#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_supervisor.py
功能：AUV 运动—刹停—悬停上位机控制节点
作者：BroXu
监听：/cmd/motion/goal (geometry_msgs/PoseStamped)
      /cmd/motion/cancel (std_msgs/Empty)
      /status/vel (geometry_msgs/TwistStamped)
      /status/auv (AUVData.msg)
      /tf
发布：/cmd/pose/ned (PoseNEDcmd.msg)
      /motion/state (MotionState.msg)
说明：
    1. 运动和刹停阶段使用 mode=2，由下位机保持深度，上位机输出 TX、TY、MZ；
    2. 捕获条件连续稳定后切换 mode=4，由下位机执行定点保持；
    3. 本节点必须是原型运行期间 /cmd/pose/ned 的唯一发布者。
记录：
2026.7.16
    新增分阶段运动、主动刹停、稳定捕获、定点接管和反馈超时保护。
2026.7.17
    TF 改为按 Time(0) 获取最新可用变换，所有目标统一使用可配置固定深度。
2026.7.17
    增加定点接管航向误差保护参数。
2026.7.18
    目标深度改为使用消息中的 z，并支持连续目标切换阈值。
2026.7.18
    增加每控制周期 CSV 完整数据日志，便于水池试验复盘和参数标定。
2026.7.18
    状态反馈固定随控制循环发布，TF 查询改为非阻塞并在首帧前发布 SAFE。
2026.7.20
    启动后锁定第一帧完整有效位姿并先完成定点接管，期间缓存最新任务目标。
    正、负计划旋转方向分别使用固定 TF；任务目标保持 base_link 最终位姿语义。
    一次动作按计划 yaw 方向锁存中心，主动刹转 MZ 反号时不切换。
"""

from __future__ import division

import csv
import json
import math
import os
import threading
from datetime import datetime

import rospy
import tf
from auv_control.msg import AUVData, MotionState, PoseNEDcmd
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from motion_supervisor_core import (
    AXIS_STATE_NAMES,
    CAPTURE,
    DEFAULT_PARAMETERS,
    HOVER,
    MotionGoal,
    MotionSupervisorCore,
    STATE_NAMES,
    SAFE,
    VehicleState,
    map_error_to_body,
    wrap_angle,
)
from lever_arm import (
    planar_origin_velocity_from_point,
)


class MotionSupervisorNode(object):
    """连接 ROS 反馈与纯算法状态机，并独占发布运动指令。"""

    LOG_FIELDS = (
        'ros_time',
        'elapsed_s',
        'state_id',
        'state',
        'reason',
        'feedback_fresh',
        'pose_age_s',
        'velocity_age_s',
        'mode_feedback_age_s',
        'reported_mode',
        'command_mode',
        'startup_complete',
        'goal_active',
        'pending_goal_active',
        'goal_sequence',
        'goal_age_s',
        'current_x',
        'current_y',
        'current_z',
        'control_x',
        'control_y',
        'control_z',
        'current_yaw_deg',
        'target_x',
        'target_y',
        'target_z',
        'control_target_x',
        'control_target_y',
        'control_target_z',
        'target_yaw_deg',
        'error_north',
        'error_east',
        'error_depth',
        'error_body_x',
        'error_body_y',
        'position_error',
        'base_position_error',
        'yaw_error_deg',
        'x_axis_state',
        'y_axis_state',
        'yaw_axis_state',
        'control_error_body_x',
        'control_error_body_y',
        'control_speed_x',
        'control_speed_y',
        'raw_imu_u',
        'raw_imu_v',
        'raw_u',
        'raw_v',
        'raw_r_deg_s',
        'filtered_u',
        'filtered_v',
        'filtered_r_deg_s',
        'horizontal_speed',
        'map_velocity_x',
        'map_velocity_y',
        'reference_velocity_x',
        'reference_velocity_y',
        'reference_speed',
        'closing_speed',
        'xy_stop_distance',
        'xy_brake_acceleration',
        'xy_brake_margin',
        'xy_brake_entry',
        'xy_brake_latched',
        'xy_braking',
        'yaw_rate_reference_deg_s',
        'map_yaw_rate_deg_s',
        'yaw_stop_angle_deg',
        'yaw_brake_entry',
        'yaw_brake_latched',
        'yaw_braking',
        'goal_static_seconds',
        'goal_static_for_capture',
        'raw_tx',
        'raw_ty',
        'raw_mz',
        'tx',
        'ty',
        'mz',
    )

    def __init__(self):
        self.control_rate_hz = float(rospy.get_param('~control_rate_hz', 20.0))
        self.feedback_timeout = float(rospy.get_param('~feedback_timeout', 0.5))
        self.velocity_filter_alpha = float(
            rospy.get_param('~velocity_filter_alpha', 0.35))
        self.pitch_offset = math.radians(
            float(rospy.get_param('~pitch_offset_deg', 0.0)))
        self.base_to_imu = self._load_base_to_imu()
        self.reference_frame = 'base_link'
        self.startup_hover_frames = int(
            rospy.get_param('~startup_hover_frames', 5))
        if self.control_rate_hz <= 0.0:
            raise ValueError('control_rate_hz 必须大于 0')
        if self.feedback_timeout <= 0.0:
            raise ValueError('feedback_timeout 必须大于 0')
        if not 0.0 < self.velocity_filter_alpha <= 1.0:
            raise ValueError('velocity_filter_alpha 必须在 (0, 1] 内')
        if self.startup_hover_frames <= 0:
            raise ValueError('startup_hover_frames 必须大于 0')
        if not all(math.isfinite(value) for value in self.base_to_imu):
            raise ValueError('base_link 到 IMU 的杆臂必须为有限值')

        self.core = MotionSupervisorCore(self._load_core_parameters())
        self.state_lock = threading.RLock()
        self.tf_listener = tf.TransformListener()
        self.last_pose = None
        self.last_pose_frame = None
        self.last_base_pose = None
        self.last_pose_stamp = None
        self.raw_velocity = None
        self.raw_imu_velocity = None
        self.filtered_velocity = None
        self.last_velocity_stamp = None
        self.reported_mode = None
        self.last_status_stamp = None
        self.goal_sequence = 0
        self.base_goal = None
        self.last_goal_stamp = None
        self.startup_latched = False
        self.startup_complete = False
        self.startup_hover_count = 0
        self.startup_pending_goal = None
        self.last_logged_state = None
        self.log_enabled = bool(rospy.get_param('~log_enabled', True))
        self.log_directory = os.path.abspath(os.path.expanduser(
            str(rospy.get_param(
                '~log_directory',
                '~/.ros/auv_logs/motion_supervisor'))))
        self.log_flush_every = int(rospy.get_param('~log_flush_every', 20))
        if self.log_flush_every <= 0:
            raise ValueError('log_flush_every 必须大于 0')
        self.log_file = None
        self.log_writer = None
        self.log_path = ''
        self.log_rows_since_flush = 0
        self.log_started_at = rospy.Time.now()
        self._open_data_log()
        rospy.on_shutdown(self._close_data_log)

        self.command_pub = rospy.Publisher(
            '/cmd/pose/ned', PoseNEDcmd, queue_size=10)
        self.state_pub = rospy.Publisher(
            '/motion/state', MotionState, queue_size=10)
        self.diagnostics_pub = rospy.Publisher(
            '/motion/diagnostics', String, queue_size=10)
        rospy.Subscriber(
            '/cmd/motion/goal', PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber(
            '/cmd/motion/cancel', Empty, self.cancel_callback, queue_size=1)
        rospy.Subscriber(
            '/status/vel', TwistStamped, self.velocity_callback, queue_size=10)
        rospy.Subscriber(
            '/status/auv', AUVData, self.status_callback, queue_size=10)

        rospy.loginfo(
            'motion_supervisor: 已启动，控制频率 %.1f Hz；'
            '使用 base_link 位姿与速度反馈进行 TX/TY/MZ 联合跟踪',
            self.control_rate_hz)
        if self.log_file is not None:
            rospy.loginfo(
                'motion_supervisor: 完整 CSV 数据日志: %s',
                self.log_path)

    def _open_data_log(self):
        """创建本次节点运行对应的 CSV 数据文件。"""
        if not self.log_enabled:
            rospy.logwarn('motion_supervisor: 完整 CSV 数据日志已禁用')
            return
        try:
            if not os.path.isdir(self.log_directory):
                os.makedirs(self.log_directory)
            filename = 'motion_supervisor_{0}.csv'.format(
                datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            self.log_path = os.path.join(self.log_directory, filename)
            self.log_file = open(
                self.log_path, 'w', encoding='utf-8', newline='')
            self.log_writer = csv.DictWriter(
                self.log_file,
                fieldnames=self.LOG_FIELDS,
                extrasaction='ignore',
            )
            self.log_writer.writeheader()
            self.log_file.flush()
        except (OSError, IOError) as error:
            self.log_file = None
            self.log_writer = None
            rospy.logerr(
                'motion_supervisor: 无法创建 CSV 数据日志，将继续控制: %s',
                error)

    def _close_data_log(self):
        """刷新并关闭 CSV 数据文件。"""
        with self.state_lock:
            if self.log_file is None:
                return
            try:
                self.log_file.flush()
                self.log_file.close()
            except (OSError, IOError) as error:
                rospy.logerr(
                    'motion_supervisor: 关闭 CSV 数据日志失败: %s', error)
            finally:
                self.log_file = None
                self.log_writer = None

    @staticmethod
    def _parameter(name, default):
        return rospy.get_param('~' + name, default)

    @staticmethod
    def _load_base_to_imu():
        """从统一的静态 TF 配置中读取 base_link 到 IMU 的杆臂。"""
        transforms = rospy.get_param('~static_transforms', None)
        if not isinstance(transforms, dict):
            raise ValueError('static_transforms 必须是字典')

        for name, transform_config in transforms.items():
            if not isinstance(transform_config, dict):
                continue
            if (transform_config.get('parent_frame') != 'base_link'
                    or transform_config.get('child_frame') != 'imu'):
                continue
            translation = transform_config.get('translation')
            try:
                return tuple(float(translation[axis]) for axis in ('x', 'y', 'z'))
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    '静态 TF %s.translation 必须包含可转换为浮点数的 x、y、z'
                    % name)

        raise ValueError('static_transforms 必须包含 base_link -> imu 变换')

    def _load_core_parameters(self):
        parameters = {}
        degree_parameters = {
            'yaw_brake_margin_positive': 'yaw_brake_margin_positive_deg',
            'yaw_brake_margin_negative': 'yaw_brake_margin_negative_deg',
            'yaw_tolerance': 'yaw_tolerance_deg',
            'yaw_rate_threshold': 'yaw_rate_threshold_deg_s',
            'hover_fault_yaw_rate': 'hover_fault_yaw_rate_deg_s',
            'hover_fault_yaw_error': 'hover_fault_yaw_error_deg',
            'yaw_max_rate': 'yaw_max_rate_deg_s',
            'yaw_max_acceleration': 'yaw_max_acceleration_deg_s2',
            'yaw_max_jerk': 'yaw_max_jerk_deg_s3',
            'yaw_brake_enter_rate': 'yaw_brake_enter_rate_deg_s',
            'yaw_brake_exit_rate': 'yaw_brake_exit_rate_deg_s',
            'goal_replan_yaw_threshold': 'goal_replan_yaw_threshold_deg',
        }
        for name, default in DEFAULT_PARAMETERS.items():
            if name in degree_parameters:
                default_degrees = math.degrees(default)
                parameters[name] = math.radians(float(self._parameter(
                    degree_parameters[name], default_degrees)))
            elif name == 'stable_frames':
                parameters[name] = int(self._parameter(name, default))
            else:
                parameters[name] = float(self._parameter(name, default))
        return parameters

    def goal_callback(self, message):
        """接收 map 坐标系最终目标。"""
        if message.header.frame_id != 'map':
            rospy.logwarn(
                'motion_supervisor: 拒绝非 map 坐标系目标: %s',
                message.header.frame_id or '<empty>')
            return
        quaternion = message.pose.orientation
        norm = math.sqrt(
            quaternion.x * quaternion.x
            + quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
            + quaternion.w * quaternion.w)
        if norm < 1e-6:
            rospy.logwarn('motion_supervisor: 拒绝无效的目标四元数')
            return
        yaw = euler_from_quaternion([
            quaternion.x / norm,
            quaternion.y / norm,
            quaternion.z / norm,
            quaternion.w / norm,
        ])[2]
        try:
            base_goal = MotionGoal(
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
                yaw,
            )
        except ValueError as error:
            rospy.logwarn('motion_supervisor: 拒绝无效目标: %s', error)
            return
        with self.state_lock:
            if self.startup_complete:
                self.base_goal = base_goal
                self.core.set_goal(base_goal)
            else:
                self.startup_pending_goal = base_goal
            self.goal_sequence += 1
            self.last_goal_stamp = rospy.Time.now()
            startup_complete = self.startup_complete
        rospy.loginfo_throttle(
            1.0,
            'motion_supervisor: 收到 base_link 目标 '
            '(x=%.2f, y=%.2f, z=%.2f, yaw=%.1fdeg)，%s',
            base_goal.x,
            base_goal.y,
            base_goal.z,
            math.degrees(base_goal.yaw),
            (
                '已交给三轴控制器'
                if startup_complete
                else '启动定点完成前缓存最新值'
            ))

    def cancel_callback(self, unused_message):
        """停止当前运动并在停稳位置悬停。"""
        del unused_message
        with self.state_lock:
            self.startup_pending_goal = None
            self.base_goal = None
            if self.startup_complete:
                self.core.cancel()
        rospy.logwarn('motion_supervisor: 收到取消指令')

    def velocity_callback(self, message):
        """把 IMU 点线速度换算到锁存计划方向的控制中心后再滤波。"""
        raw_imu = (
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.angular.z,
        )
        if not all(math.isfinite(value) for value in raw_imu):
            rospy.logwarn_throttle(1.0, 'motion_supervisor: 忽略非有限速度反馈')
            return
        with self.state_lock:
            base_u, base_v = planar_origin_velocity_from_point(
                raw_imu[0],
                raw_imu[1],
                raw_imu[2],
                self.base_to_imu,
            )
            raw = (base_u, base_v, raw_imu[2])
            self.raw_imu_velocity = raw_imu
            self.raw_velocity = raw
            if self.filtered_velocity is None:
                self.filtered_velocity = list(raw)
            else:
                alpha = self.velocity_filter_alpha
                for index, value in enumerate(raw):
                    self.filtered_velocity[index] = (
                        alpha * value
                        + (1.0 - alpha) * self.filtered_velocity[index])
            self.last_velocity_stamp = rospy.Time.now()

    def status_callback(self, message):
        """记录下位机实际运行模式，用于定点接管确认。"""
        with self.state_lock:
            self.reported_mode = int(message.control_mode)
            self.last_status_stamp = rospy.Time.now()

    def _update_pose(self, now):
        try:
            # Time(0) 读取最新可用变换；这里不能阻塞等待，否则 TF 异常时
            # /motion/state 无法保持控制频率发布。
            base_translation, base_rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0))
            base_yaw = euler_from_quaternion(base_rotation)[2]
            base_values = (
                base_translation[0],
                base_translation[1],
                base_translation[2],
                base_yaw,
            )
            if not all(math.isfinite(value) for value in base_values):
                raise ValueError('TF 包含非有限值')
            self.last_base_pose = base_values
            self.last_pose = base_values
            self.last_pose_frame = self.reference_frame
            # 查询仍使用 Time(0) 获取最新位姿，但反馈年龄必须使用 TF
            # 缓冲区中的真实最新时间，不能使用本次查询时刻代替。
            base_stamp = self.tf_listener.getLatestCommonTime(
                'map', 'base_link')
            self.last_pose_stamp = (
                base_stamp if base_stamp != rospy.Time(0) else now)
        except (tf.Exception, ValueError) as error:
            rospy.logwarn_throttle(
                2.0, 'motion_supervisor: 无法更新 AUV 位姿: %s', error)

    def _feedback_is_fresh(self, now):
        if (
                self.last_pose is None
                or self.last_pose_frame != self.reference_frame
                or self.last_pose_stamp is None
                or self.filtered_velocity is None
                or self.last_velocity_stamp is None):
            return False
        pose_age = max(0.0, (now - self.last_pose_stamp).to_sec())
        velocity_age = max(0.0, (now - self.last_velocity_stamp).to_sec())
        return (
            pose_age <= self.feedback_timeout
            and velocity_age <= self.feedback_timeout)

    def _build_vehicle_state(self, now, feedback_fresh):
        """按当前锁存虚拟中心构造纯算法核心使用的状态。"""
        velocity = self.filtered_velocity or (0.0, 0.0, 0.0)
        return VehicleState(
            now.to_sec(),
            self.last_pose[0],
            self.last_pose[1],
            self.last_pose[2],
            self.last_pose[3],
            velocity[0],
            velocity[1],
            velocity[2],
            feedback_fresh=feedback_fresh,
            reported_mode=self.reported_mode,
            reported_mode_stamp=(
                None
                if self.last_status_stamp is None
                else self.last_status_stamp.to_sec()),
        )

    @staticmethod
    def _age_seconds(now, stamp):
        """返回反馈年龄；从未收到时留空，便于 CSV 区分缺失和零延迟。"""
        if stamp is None:
            return ''
        return max(0.0, (now - stamp).to_sec())

    def _write_cycle_log(self, now, vehicle, output):
        """记录一个控制周期的完整输入、状态、误差和输出。"""
        if self.log_writer is None:
            return
        control_target = output.target
        target = control_target
        current_base_position = self.last_base_pose[0:3]
        error_north = target.x - current_base_position[0]
        error_east = target.y - current_base_position[1]
        error_body_x, error_body_y = map_error_to_body(
            error_north, error_east, vehicle.yaw)
        base_position_error = math.hypot(error_north, error_east)
        raw_velocity = self.raw_velocity or ('', '', '')
        raw_imu_velocity = self.raw_imu_velocity or ('', '', '')
        filtered_velocity = self.filtered_velocity or ('', '', '')
        row = {
            'ros_time': '{0:.9f}'.format(now.to_sec()),
            'elapsed_s': '{0:.3f}'.format(
                max(0.0, (now - self.log_started_at).to_sec())),
            'state_id': output.state,
            'state': STATE_NAMES.get(output.state, str(output.state)),
            'reason': output.reason,
            'feedback_fresh': int(vehicle.feedback_fresh),
            'pose_age_s': self._age_seconds(now, self.last_pose_stamp),
            'velocity_age_s': self._age_seconds(
                now, self.last_velocity_stamp),
            'mode_feedback_age_s': self._age_seconds(
                now, self.last_status_stamp),
            'reported_mode': (
                '' if self.reported_mode is None else self.reported_mode),
            'command_mode': output.mode,
            'startup_complete': int(self.startup_complete),
            'goal_active': int(output.goal_active),
            'pending_goal_active': int(
                self.core.pending_goal is not None
                or self.startup_pending_goal is not None),
            'goal_sequence': self.goal_sequence,
            'goal_age_s': self._age_seconds(now, self.last_goal_stamp),
            'current_x': current_base_position[0],
            'current_y': current_base_position[1],
            'current_z': current_base_position[2],
            'control_x': vehicle.x,
            'control_y': vehicle.y,
            'control_z': vehicle.z,
            'current_yaw_deg': math.degrees(vehicle.yaw),
            'target_x': target.x,
            'target_y': target.y,
            'target_z': target.z,
            'control_target_x': control_target.x,
            'control_target_y': control_target.y,
            'control_target_z': control_target.z,
            'target_yaw_deg': math.degrees(target.yaw),
            'error_north': error_north,
            'error_east': error_east,
            'error_depth': target.z - current_base_position[2],
            'error_body_x': error_body_x,
            'error_body_y': error_body_y,
            'position_error': output.position_error,
            'base_position_error': base_position_error,
            'yaw_error_deg': math.degrees(
                wrap_angle(target.yaw - vehicle.yaw)),
            'x_axis_state': AXIS_STATE_NAMES.get(
                output.x_axis_state, str(output.x_axis_state)),
            'y_axis_state': AXIS_STATE_NAMES.get(
                output.y_axis_state, str(output.y_axis_state)),
            'yaw_axis_state': AXIS_STATE_NAMES.get(
                output.yaw_axis_state, str(output.yaw_axis_state)),
            'control_error_body_x': output.x_error,
            'control_error_body_y': output.y_error,
            'control_speed_x': output.x_speed,
            'control_speed_y': output.y_speed,
            'raw_imu_u': raw_imu_velocity[0],
            'raw_imu_v': raw_imu_velocity[1],
            'raw_u': raw_velocity[0],
            'raw_v': raw_velocity[1],
            'raw_r_deg_s': (
                '' if raw_velocity[2] == ''
                else math.degrees(raw_velocity[2])),
            'filtered_u': filtered_velocity[0],
            'filtered_v': filtered_velocity[1],
            'filtered_r_deg_s': (
                '' if filtered_velocity[2] == ''
                else math.degrees(filtered_velocity[2])),
            'horizontal_speed': output.horizontal_speed,
            'tx': output.tx,
            'ty': output.ty,
            'mz': output.mz,
        }
        diagnostics = output.diagnostics
        row.update({
            'map_velocity_x': diagnostics.get('map_velocity_x', ''),
            'map_velocity_y': diagnostics.get('map_velocity_y', ''),
            'reference_velocity_x': diagnostics.get('reference_velocity_x', ''),
            'reference_velocity_y': diagnostics.get('reference_velocity_y', ''),
            'reference_speed': diagnostics.get('reference_speed', ''),
            'closing_speed': diagnostics.get('closing_speed', ''),
            'xy_stop_distance': diagnostics.get('xy_stop_distance', ''),
            'xy_brake_acceleration': diagnostics.get('xy_brake_acceleration', ''),
            'xy_brake_margin': diagnostics.get('xy_brake_margin', ''),
            'xy_brake_entry': int(bool(
                diagnostics.get('xy_brake_entry', False))),
            'xy_brake_latched': int(bool(
                diagnostics.get('xy_brake_latched', False))),
            'xy_braking': int(bool(diagnostics.get('xy_braking', False))),
            'yaw_rate_reference_deg_s': (
                math.degrees(diagnostics['yaw_rate_reference'])
                if 'yaw_rate_reference' in diagnostics else ''),
            'map_yaw_rate_deg_s': (
                math.degrees(diagnostics['map_yaw_rate'])
                if 'map_yaw_rate' in diagnostics else ''),
            'yaw_stop_angle_deg': (
                math.degrees(diagnostics['yaw_stop_angle'])
                if 'yaw_stop_angle' in diagnostics else ''),
            'yaw_brake_entry': int(bool(
                diagnostics.get('yaw_brake_entry', False))),
            'yaw_brake_latched': int(bool(
                diagnostics.get('yaw_brake_latched', False))),
            'yaw_braking': int(bool(diagnostics.get('yaw_braking', False))),
            'goal_static_seconds': diagnostics.get('goal_static_seconds', ''),
            'goal_static_for_capture': int(bool(
                diagnostics.get('goal_static_for_capture', False))),
            'raw_tx': diagnostics.get('raw_tx', ''),
            'raw_ty': diagnostics.get('raw_ty', ''),
            'raw_mz': diagnostics.get('raw_mz', ''),
        })
        try:
            self.log_writer.writerow(row)
            self.log_rows_since_flush += 1
            if self.log_rows_since_flush >= self.log_flush_every:
                self.log_file.flush()
                self.log_rows_since_flush = 0
        except (OSError, IOError, ValueError) as error:
            rospy.logerr(
                'motion_supervisor: 写入 CSV 数据日志失败，停止本次记录: %s',
                error)
            self._close_data_log()

    def _target_pose(self, goal, stamp, pitch_offset):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = 'map'
        pose.pose.position.x = goal.x
        pose.pose.position.y = goal.y
        pose.pose.position.z = goal.z
        quaternion = quaternion_from_euler(0.0, pitch_offset, goal.yaw)
        pose.pose.orientation.x = quaternion[0]
        pose.pose.orientation.y = quaternion[1]
        pose.pose.orientation.z = quaternion[2]
        pose.pose.orientation.w = quaternion[3]
        return pose

    def _publish_command(self, output, now):
        command = PoseNEDcmd()
        command.mode = output.mode
        command.target = self._target_pose(output.target, now, self.pitch_offset)
        command.force.TX = output.tx
        command.force.TY = output.ty
        command.force.TZ = 0
        command.force.MX = 0
        command.force.MY = 0
        command.force.MZ = output.mz
        self.command_pub.publish(command)

    def _publish_state(self, output, now):
        message = MotionState()
        message.header.stamp = now
        message.header.frame_id = 'map'
        message.state = (
            CAPTURE
            if output.state == HOVER and not self.startup_complete
            else output.state
        )
        message.goal_active = output.goal_active
        message.goal = self._target_pose(output.target, now, self.pitch_offset)
        message.position_error = output.position_error
        if self.last_base_pose is not None:
            current_base = self.last_base_pose[0:3]
            base_goal = output.target
            message.base_position_error = math.hypot(
                base_goal.x - current_base[0],
                base_goal.y - current_base[1],
            )
        message.yaw_error = output.yaw_error
        message.horizontal_speed = output.horizontal_speed
        message.yaw_rate = output.yaw_rate
        message.tx = output.tx
        message.ty = output.ty
        message.mz = output.mz
        message.x_axis_state = output.x_axis_state
        message.y_axis_state = output.y_axis_state
        message.yaw_axis_state = output.yaw_axis_state
        message.x_axis_error = output.x_error
        message.y_axis_error = output.y_error
        message.x_axis_speed = output.x_speed
        message.y_axis_speed = output.y_speed
        message.startup_complete = self.startup_complete
        message.reason = (
            '启动首帧定点已接管，等待稳定发布帧数'
            if output.state == HOVER and not self.startup_complete
            else output.reason
        )
        self.state_pub.publish(message)

    def _publish_diagnostics(self, output, now):
        """发布可供网页与离线工具直接消费的统一控制诊断数据。"""
        payload = dict(output.diagnostics)
        payload.update({
            'stamp_sec': now.to_sec(),
            'state': STATE_NAMES.get(output.state, str(output.state)),
            'control_frame': self.reference_frame,
            'position_error_m': output.position_error,
            'yaw_error_rad': output.yaw_error,
            'horizontal_speed_mps': output.horizontal_speed,
            'yaw_rate_radps': output.yaw_rate,
            'tx': output.tx,
            'ty': output.ty,
            'mz': output.mz,
        })
        self.diagnostics_pub.publish(String(data=json.dumps(
            payload, separators=(',', ':'), sort_keys=True)))

    def _publish_waiting_state(self, now):
        """首帧 TF 到达前以 SAFE 状态保持任务侧周期反馈。"""
        message = MotionState()
        message.header.stamp = now
        message.header.frame_id = 'map'
        message.state = SAFE
        message.goal_active = (
            self.core.goal_active
            or self.startup_pending_goal is not None
        )
        target = (
            self.core.goal
            or self.core.pending_goal
        )
        if target is not None:
            message.goal = self._target_pose(
                target, now, self.pitch_offset)
        elif self.startup_pending_goal is not None:
            # 启动前缓存的是任务侧 base_link 目标，先按当前默认计划方向
            # 换算后再复用统一的消息构造函数。
            message.goal = self._target_pose(
                self.startup_pending_goal,
                now,
                self.pitch_offset,
            )
        else:
            message.goal.header.stamp = now
            message.goal.header.frame_id = 'map'
            message.goal.pose.orientation.w = 1.0
        message.position_error = 0.0
        message.yaw_error = 0.0
        message.horizontal_speed = 0.0
        message.yaw_rate = 0.0
        message.tx = 0
        message.ty = 0
        message.mz = 0
        message.startup_complete = self.startup_complete
        message.reason = '等待首帧完整有效的 base_link TF 和 IMU 点速度'
        self.state_pub.publish(message)

    def _run_control_cycle(self, now):
        """在同一锁内完成反馈快照、状态推进和指令发布。"""
        self._update_pose(now)

        # 首次获得位姿前不能构造可靠的定深安全指令，因此只发布状态。
        if self.last_pose is None:
            self._publish_waiting_state(now)
            rospy.logwarn_throttle(2.0, 'motion_supervisor: 等待首帧 TF')
            return

        feedback_fresh = self._feedback_is_fresh(now)
        if not self.startup_latched and not feedback_fresh:
            self._publish_waiting_state(now)
            rospy.logwarn_throttle(
                2.0,
                'motion_supervisor: 等待首帧完整有效的 TF 和速度反馈')
            return
        vehicle = self._build_vehicle_state(now, feedback_fresh)
        if not self.startup_latched:
            self.core.set_goal(MotionGoal(
                vehicle.x,
                vehicle.y,
                vehicle.z,
                vehicle.yaw,
            ))
            self.startup_latched = True
            rospy.loginfo(
                'motion_supervisor: 已锁定首帧 %s 位姿 '
                '(%.3f, %.3f, %.3f, %.1fdeg)，开始刹停并定点',
                self.reference_frame,
                vehicle.x,
                vehicle.y,
                vehicle.z,
                math.degrees(vehicle.yaw),
            )
        output = self.core.step(vehicle)
        if not self.startup_complete:
            if output.state == HOVER:
                self.startup_hover_count += 1
                if self.startup_hover_count >= self.startup_hover_frames:
                    self.startup_complete = True
                    rospy.loginfo(
                        'motion_supervisor: 首帧锁定位姿已完成 mode=4 接管')
                    if self.startup_pending_goal is not None:
                        pending = self.startup_pending_goal
                        self.startup_pending_goal = None
                        self.base_goal = pending
                        self.core.set_goal(pending)
                        rospy.loginfo(
                            'motion_supervisor: 启动完成，执行缓存的最新任务目标')
                        output = self.core.step(vehicle)
            else:
                self.startup_hover_count = 0
        self._publish_command(output, now)
        self._publish_state(output, now)
        self._publish_diagnostics(output, now)
        self._write_cycle_log(now, vehicle, output)

        if output.state != self.last_logged_state:
            rospy.loginfo(
                'motion_supervisor: 状态 -> %s，原因: %s',
                STATE_NAMES.get(output.state, str(output.state)), output.reason)
            self.last_logged_state = output.state
        rospy.loginfo_throttle(
            1.0,
            'motion_supervisor: state=%s center=%s error=%.2fm/%.1fdeg '
            'speed=%.3fm/s yaw_rate=%.2fdeg/s force=(%d,%d,%d)',
            STATE_NAMES.get(output.state, str(output.state)),
            self.reference_frame,
            output.position_error,
            math.degrees(output.yaw_error),
            output.horizontal_speed,
            math.degrees(output.yaw_rate),
            output.tx,
            output.ty,
            output.mz,
        )

    def run(self):
        rate = rospy.Rate(self.control_rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            with self.state_lock:
                self._run_control_cycle(now)
            rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node('motion_supervisor')
        MotionSupervisorNode().run()
    except rospy.ROSInterruptException:
        pass
