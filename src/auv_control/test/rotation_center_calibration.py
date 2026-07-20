#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：rotation_center_calibration.py
功能：以 TX=TY=0 的自由正反多圈旋转数据自动标定水平旋转中心
作者：BroXu
监听：
    /status/vel (geometry_msgs/TwistStamped)
    /status/auv (AUVData.msg)
    /tf
发布：
    /cmd/pose/ned (PoseNEDcmd.msg)
说明：
    1. 运行时必须停止 motion_supervisor，本节点独占 /cmd/pose/ned；
    2. mode=2 仅输出 MZ，TX/TY 始终为零，避免平移控制污染旋转中心；
    3. 低、中、高三档依次测试；正向 MZ 默认 1000/2000/3000，
       负向为对应档位的 1.5 倍；
    4. 每档每 90° 刹停并回到同一个锁定位置，正反各三圈；
    5. 每个旋转段允许不同 map 圆心，各档正向和负向分别拟合独立旋转中心；
    6. 拟合只使用 TRACK 阶段样本，不把反号主动刹转样本混入计划方向中心；
    7. 结果分档输出，优先推荐中速；中速质量明显异常时选择更稳定档位。
记录：
2026.7.18
    新增自由旋转、主动刹转、定点接管和旋转中心自动拟合程序。
2026.7.18
    TF 改为读取最新 map -> base_link，并在启动时等待首帧；base_link 与
    IMU 重合时复用同一位姿，避免不同 TF 链时间不一致造成外推。
    远离目标保护只在尚未越过目标时累计，允许 BRAKE 正常超调并继续减速。
2026.7.18
    首帧有效位姿只锁定一次；全程 mode=2 和段间 mode=4 均使用同一初始
    水平位置，每段确认位置、航向、线速度和角速度稳定后才进入下一段。
2026.7.19
    按计划旋转方向分别拟合正向和负向旋转中心；拟合仅使用 TRACK 且达到
    最低角速度的样本，主动刹转样本不混入计划方向中心。
2026.7.19
    增加低、中、高三档不对称 MZ 自动序列、分档统计拟合及推荐中心输出。
2026.7.19
    取消自由旋转漂移超限中止；漂移只记录告警，完成 90° 后仍返回全程锁定的初始位置。
2026.7.19
    修复越过目标后低速阶段重新输出原方向固定 MZ 的问题；改为闭环恢复并正确刹停。
2026.7.19
    段间 mode=4 改为判稳后连续保持 12 s；恢复阶段不再触发首次接近方向保护。
"""

from __future__ import division

import csv
import math
import os
import sys
from datetime import datetime

import rosgraph
import rospy
import tf
from geometry_msgs.msg import TwistStamped
from rosgraph.masterapi import ROSMasterException
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from auv_control.msg import AUVData, PoseNEDcmd


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from rotation_center_calibration_core import (  # noqa: E402
    CalibrationSample,
    apply_fixed_track_policy,
    build_torque_profiles,
    direct_yaw_command,
    drift_exceeds_warning_threshold,
    fit_planned_direction_centers,
    fit_quality_score,
    locked_handover_is_stable,
    rotation_is_unexpectedly_moving_away,
    select_recommended_profile,
    unwrap_angle,
    update_continuous_stability,
    wrap_angle,
)


NODE_NAME = 'rotation_center_calibration'
MODE_DEPTH = 2
MODE_DPROV = 4


class RotationCenterCalibration(object):
    """执行自由偏航并分别拟合正、负计划方向的水平旋转中心。"""

    LOG_STEM = 'rotation_center'
    RESULT_GENERATOR = 'rotation_center_calibration.py'

    LOG_FIELDS = (
        'ros_time',
        'phase',
        'torque_profile',
        'track_mz_positive_limit',
        'track_mz_negative_limit',
        'segment',
        'direction',
        'turn_index',
        'quarter_index',
        'target_unwrapped_yaw_deg',
        'imu_x',
        'imu_y',
        'imu_z',
        'base_x',
        'base_y',
        'base_z',
        'target_x',
        'target_y',
        'target_z',
        'wrapped_yaw_deg',
        'unwrapped_yaw_deg',
        'yaw_error_deg',
        'raw_r_deg_s',
        'tf_yaw_rate_deg_s',
        'horizontal_speed',
        'command_mode',
        'command_mz',
        'controller_phase',
        'stopping_angle_deg',
        'reported_mode',
    )

    @staticmethod
    def _new_track_statistics():
        """创建一个力矩档位、一个计划方向的 TRACK 统计量。"""
        return {
            'cycle_count': 0,
            'yaw_rate_sum': 0.0,
            'peak_yaw_rate': 0.0,
            'horizontal_speed_sum': 0.0,
            'peak_horizontal_speed': 0.0,
            'max_drift': 0.0,
        }

    @staticmethod
    def _finalize_track_statistics(statistics):
        """把 TRACK 累计量转换为可写入结果文件的统计值。"""
        count = int(statistics['cycle_count'])
        divisor = float(max(1, count))
        return {
            'cycle_count': count,
            'mean_abs_yaw_rate': (
                statistics['yaw_rate_sum'] / divisor),
            'peak_abs_yaw_rate': statistics['peak_yaw_rate'],
            'mean_horizontal_speed': (
                statistics['horizontal_speed_sum'] / divisor),
            'peak_horizontal_speed': (
                statistics['peak_horizontal_speed']),
            'max_drift': statistics['max_drift'],
        }

    def _update_track_statistics(
            self, profile_name, direction, yaw_rate,
            horizontal_speed, drift):
        """累计当前档位和计划方向的 TRACK 运动特征。"""
        statistics = self.profile_statistics[
            profile_name][int(direction)]
        absolute_rate = abs(float(yaw_rate))
        speed = abs(float(horizontal_speed))
        statistics['cycle_count'] += 1
        statistics['yaw_rate_sum'] += absolute_rate
        statistics['peak_yaw_rate'] = max(
            statistics['peak_yaw_rate'], absolute_rate)
        statistics['horizontal_speed_sum'] += speed
        statistics['peak_horizontal_speed'] = max(
            statistics['peak_horizontal_speed'], speed)
        statistics['max_drift'] = max(
            statistics['max_drift'], float(drift))

    def __init__(self):
        self.target_z = float(rospy.get_param('~target_z', -0.9))
        self.control_to_imu_z = float(
            rospy.get_param('~control_to_imu_z', 0.0))
        self.control_rate_hz = float(
            rospy.get_param('~control_rate_hz', 5.0))
        self.turn_step = math.radians(float(
            rospy.get_param('~turn_step_deg', 90.0)))
        self.turns_each_direction = int(
            rospy.get_param('~turns_each_direction', 3))
        self.initial_hold_seconds = float(
            rospy.get_param('~initial_hold_seconds', 3.0))
        self.handover_hold_seconds = float(
            rospy.get_param('~handover_hold_seconds', 12.0))
        self.step_timeout = float(
            rospy.get_param('~step_timeout', 90.0))
        self.feedback_timeout = float(
            rospy.get_param('~feedback_timeout', 0.5))
        self.stable_frames = int(rospy.get_param('~stable_frames', 5))
        self.yaw_tolerance = math.radians(float(
            rospy.get_param('~yaw_tolerance_deg', 3.0)))
        self.yaw_rate_threshold = math.radians(float(
            rospy.get_param('~yaw_rate_threshold_deg_s', 0.5)))
        self.handover_speed_threshold = float(
            rospy.get_param('~handover_speed_threshold', 0.05))
        self.handover_position_tolerance = float(
            rospy.get_param('~handover_position_tolerance', 0.15))
        self.velocity_filter_alpha = float(
            rospy.get_param('~velocity_filter_alpha', 0.35))
        self.yaw_rate_feedback_sign = float(
            rospy.get_param('~yaw_rate_feedback_sign', -1.0))
        self.mz_to_yaw_sign = float(
            rospy.get_param('~mz_to_yaw_sign', 1.0))

        self.kp_yaw = float(rospy.get_param('~kp_yaw', 6000.0))
        self.kd_yaw = float(rospy.get_param('~kd_yaw', 2000.0))
        self.brake_gain_yaw = float(
            rospy.get_param('~brake_gain_yaw', 6000.0))
        self.positive_mz_levels = tuple(float(value) for value in
                                        rospy.get_param(
                                            '~positive_mz_levels',
                                            [1000.0, 2000.0, 3000.0]))
        self.negative_mz_scale = float(
            rospy.get_param('~negative_mz_scale', 1.5))
        self.torque_profiles = build_torque_profiles(
            self.positive_mz_levels,
            negative_scale=self.negative_mz_scale,
        )
        self.total_segments = (
            len(self.torque_profiles)
            * 2
            * self.turns_each_direction
            * 4
        )
        self.brake_max_mz_positive = float(
            rospy.get_param('~brake_max_mz_positive', 3000.0))
        self.brake_max_mz_negative = float(
            rospy.get_param('~brake_max_mz_negative', 3000.0))
        self.brake_acceleration_positive = float(rospy.get_param(
            '~angular_brake_acceleration_mz_positive', 0.025))
        self.brake_acceleration_negative = float(rospy.get_param(
            '~angular_brake_acceleration_mz_negative', 0.040))
        self.control_delay = float(
            rospy.get_param('~control_delay', 0.35))
        self.brake_margin = math.radians(float(
            rospy.get_param('~yaw_brake_margin_deg', 3.0)))
        self.minimum_brake_mz = float(
            rospy.get_param('~minimum_brake_mz', 100.0))
        legacy_drift_radius = rospy.get_param(
            '~drift_abort_radius', 0.75)
        self.drift_warning_radius = float(rospy.get_param(
            '~drift_warning_radius', legacy_drift_radius))
        self.sample_min_yaw_rate = math.radians(float(
            rospy.get_param('~sample_min_yaw_rate_deg_s', 1.0)))
        self.max_yaw_rate = math.radians(float(
            rospy.get_param('~max_yaw_rate_deg_s', 45.0)))
        self.moving_away_timeout = float(
            rospy.get_param('~moving_away_timeout', 3.0))
        self.preferred_profile = str(
            rospy.get_param('~preferred_profile', 'medium')).strip()
        self.recommendation_degradation_ratio = float(rospy.get_param(
            '~recommendation_degradation_ratio', 1.5))
        self.recommendation_degradation_margin = float(rospy.get_param(
            '~recommendation_degradation_margin', 0.01))
        self.log_directory = os.path.abspath(os.path.expanduser(str(
            rospy.get_param(
                '~log_directory',
                '~/.ros/auv_logs/rotation_center_calibration'))))

        numeric = (
            self.target_z,
            self.control_to_imu_z,
            self.control_rate_hz,
            self.turn_step,
            self.initial_hold_seconds,
            self.handover_hold_seconds,
            self.step_timeout,
            self.feedback_timeout,
            self.yaw_tolerance,
            self.yaw_rate_threshold,
            self.handover_speed_threshold,
            self.handover_position_tolerance,
            self.velocity_filter_alpha,
            self.yaw_rate_feedback_sign,
            self.mz_to_yaw_sign,
            self.kp_yaw,
            self.kd_yaw,
            self.brake_gain_yaw,
            self.brake_max_mz_positive,
            self.brake_max_mz_negative,
            self.brake_acceleration_positive,
            self.brake_acceleration_negative,
            self.control_delay,
            self.brake_margin,
            self.minimum_brake_mz,
            self.drift_warning_radius,
            self.sample_min_yaw_rate,
            self.max_yaw_rate,
            self.moving_away_timeout,
            self.negative_mz_scale,
            self.recommendation_degradation_ratio,
            self.recommendation_degradation_margin,
        ) + tuple(
            value
            for profile in self.torque_profiles
            for value in (
                profile.positive_limit,
                profile.negative_limit,
            )
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('旋转中心标定参数必须为有限值')
        if (
                self.control_rate_hz <= 0.0
                or self.turn_step <= 0.0
                or abs(self.turn_step - math.pi / 2.0) > 1e-6
                or self.turns_each_direction <= 0
                or self.initial_hold_seconds < 0.0
                or self.handover_hold_seconds < 0.0
                or self.step_timeout <= 0.0
                or self.feedback_timeout <= 0.0
                or self.stable_frames <= 0
                or self.handover_position_tolerance <= 0.0
                or not 0.0 < self.velocity_filter_alpha <= 1.0
                or self.drift_warning_radius <= 0.0
                or self.max_yaw_rate <= 0.0
                or self.moving_away_timeout <= 0.0
                or self.preferred_profile not in tuple(
                    profile.name for profile in self.torque_profiles)
                or self.recommendation_degradation_ratio < 1.0
                or self.recommendation_degradation_margin < 0.0):
            raise ValueError(
                '频率、角度、圈数、超时、力矩档位和推荐阈值不在有效范围；'
                '当前 turn_step_deg 必须为 90')

        self.tf_listener = tf.TransformListener()
        self.raw_velocity = None
        self.filtered_velocity = None
        self.last_velocity_stamp = None
        self.reported_mode = None
        self.last_status_stamp = None
        self.current_imu_pose = None
        self.current_base_pose = None
        self.locked_initial_pose = None
        self.locked_initial_yaw = None
        self.last_tf_stamp = None
        self.unwrapped_yaw = None
        self.direction_samples = {
            profile.name: {1: [], -1: []}
            for profile in self.torque_profiles
        }
        self.profile_statistics = {
            profile.name: {
                1: self._new_track_statistics(),
                -1: self._new_track_statistics(),
            }
            for profile in self.torque_profiles
        }
        self.latest_command = None
        self.completed = False
        self.log_file = None
        self.log_writer = None
        self.log_path = ''
        self.result_path = ''

        self._assert_no_other_command_publisher(before_start=True)
        self.command_pub = rospy.Publisher(
            '/cmd/pose/ned', PoseNEDcmd, queue_size=10)
        rospy.Subscriber(
            '/status/vel',
            TwistStamped,
            self.velocity_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            '/status/auv',
            AUVData,
            self.status_callback,
            queue_size=10,
        )
        self._open_log()
        rospy.on_shutdown(self._on_shutdown)

    def _publishers_on_command_topic(self):
        master = rosgraph.Master(rospy.get_name())
        publishers, unused_subscribers, unused_services = (
            master.getSystemState())
        del unused_subscribers, unused_services
        for topic, nodes in publishers:
            if topic == '/cmd/pose/ned':
                return list(nodes)
        return []

    def _assert_no_other_command_publisher(self, before_start=False):
        own_name = rospy.get_name()
        publishers = self._publishers_on_command_topic()
        others = [
            name for name in publishers
            if name != own_name
        ]
        if others:
            raise RuntimeError(
                '/cmd/pose/ned 存在其他发布者 {}；必须先停止 '
                'motion_supervisor 和其他控制节点'.format(
                    ', '.join(others)))
        if before_start and publishers:
            raise RuntimeError(
                '启动标定前 /cmd/pose/ned 必须没有任何发布者')

    def _open_log(self):
        if not os.path.isdir(self.log_directory):
            os.makedirs(self.log_directory)
        stem = '{}_{}'.format(
            self.LOG_STEM,
            datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
        )
        self.log_path = os.path.join(self.log_directory, stem + '.csv')
        self.result_path = os.path.join(
            self.log_directory, stem + '_result.yaml')
        self.log_file = open(
            self.log_path, 'w', encoding='utf-8', newline='')
        self.log_writer = csv.DictWriter(
            self.log_file, fieldnames=self.LOG_FIELDS)
        self.log_writer.writeheader()
        self.log_file.flush()

    def _close_log(self):
        if self.log_file is None:
            return
        self.log_file.flush()
        self.log_file.close()
        self.log_file = None
        self.log_writer = None

    def velocity_callback(self, message):
        values = (
            message.twist.linear.x,
            message.twist.linear.y,
            message.twist.angular.z,
        )
        if not all(math.isfinite(value) for value in values):
            return
        self.raw_velocity = values
        if self.filtered_velocity is None:
            self.filtered_velocity = list(values)
        else:
            alpha = self.velocity_filter_alpha
            for index, value in enumerate(values):
                self.filtered_velocity[index] = (
                    alpha * value
                    + (1.0 - alpha) * self.filtered_velocity[index])
        self.last_velocity_stamp = rospy.Time.now()

    def status_callback(self, message):
        self.reported_mode = int(message.control_mode)
        self.last_status_stamp = rospy.Time.now()

    def _lookup_pose(self, child):
        translation, rotation = self.tf_listener.lookupTransform(
            'map', child, rospy.Time(0))
        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('{} TF 包含非有限值'.format(child))
        yaw = euler_from_quaternion(rotation)[2]
        return tuple(translation), tuple(rotation), yaw

    def _update_tf(self):
        try:
            # 与其他任务节点一致，Time(0) 读取最新可用 map -> base_link。
            # 当前 base_link 与 IMU 重合，复用同一帧可避免旧式静态 TF
            # 与动态 TF 时间戳不同步时 map -> imu 出现外推。
            base_position, base_rotation, base_yaw = self._lookup_pose(
                'base_link')
            stamp = self.tf_listener.getLatestCommonTime(
                'map', 'base_link')
        except (tf.Exception, ValueError) as error:
            rospy.logwarn_throttle(
                2.0, '%s: TF 更新失败: %s', NODE_NAME, error)
            return False
        self.current_imu_pose = (
            base_position, base_rotation, base_yaw)
        self.current_base_pose = (
            base_position, base_rotation)
        self.last_tf_stamp = (
            stamp if stamp != rospy.Time(0) else rospy.Time.now())
        if self.unwrapped_yaw is None:
            self.unwrapped_yaw = base_yaw
        else:
            self.unwrapped_yaw = unwrap_angle(
                self.unwrapped_yaw, base_yaw)
        return True

    @staticmethod
    def _age(now, stamp):
        if stamp is None:
            return float('inf')
        return max(0.0, (now - stamp).to_sec())

    def _feedback_fresh(self):
        now = rospy.Time.now()
        return (
            self.current_imu_pose is not None
            and self.current_base_pose is not None
            and self.filtered_velocity is not None
            and self._age(now, self.last_tf_stamp) <= self.feedback_timeout
            and self._age(now, self.last_velocity_stamp)
            <= self.feedback_timeout
            and self._age(now, self.last_status_stamp)
            <= self.feedback_timeout
        )

    def _wait_for_feedback(self):
        deadline = rospy.Time.now() + rospy.Duration(30.0)
        rate = rospy.Rate(self.control_rate_hz)
        try:
            # 先等待 TF 监听器完成缓存，避免节点刚启动时立即查询产生
            # “请求时间早于缓冲区首帧”的暂态外推警告。
            self.tf_listener.waitForTransform(
                'map',
                'base_link',
                rospy.Time(0),
                rospy.Duration(5.0),
            )
        except tf.Exception as error:
            rospy.logwarn(
                '%s: 等待首帧 map -> base_link TF 超时，将继续重试: %s',
                NODE_NAME,
                error,
            )
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._update_tf()
            if self._feedback_fresh():
                return True
            rospy.loginfo_throttle(
                2.0,
                '%s: 等待最新 map->base_link、速度和模式反馈',
                NODE_NAME,
            )
            rate.sleep()
        return False

    def _make_command(self, mode, base_pose, yaw, mz=0):
        command = PoseNEDcmd()
        command.mode = int(mode)
        command.target.header.stamp = rospy.Time.now()
        command.target.header.frame_id = 'map'
        command.target.pose.position.x = base_pose[0][0]
        command.target.pose.position.y = base_pose[0][1]
        command.target.pose.position.z = self.target_z
        quaternion = quaternion_from_euler(0.0, 0.0, wrap_angle(yaw))
        command.target.pose.orientation.x = quaternion[0]
        command.target.pose.orientation.y = quaternion[1]
        command.target.pose.orientation.z = quaternion[2]
        command.target.pose.orientation.w = quaternion[3]
        command.force.TX = 0
        command.force.TY = 0
        command.force.TZ = 0
        command.force.MX = 0
        command.force.MY = 0
        command.force.MZ = int(mz)
        return command

    def _publish_command(self, command):
        command.target.header.stamp = rospy.Time.now()
        self.latest_command = command
        self.command_pub.publish(command)

    def _log_cycle(
            self, phase, profile, segment, direction,
            turn_index, quarter_index,
            target_unwrapped, yaw_error, command_mode, command_mz,
            controller_phase, stopping_angle):
        imu_position, unused_imu_rotation, imu_yaw = self.current_imu_pose
        del unused_imu_rotation
        base_position, unused_base_rotation = self.current_base_pose
        del unused_base_rotation
        velocity = self.filtered_velocity
        raw_r = (
            0.0 if self.raw_velocity is None else self.raw_velocity[2])
        yaw_rate = self.yaw_rate_feedback_sign * velocity[2]
        command_target = self.latest_command.target.pose.position
        profile_name = '' if profile is None else profile.name
        positive_limit = (
            '' if profile is None else profile.positive_limit)
        negative_limit = (
            '' if profile is None else profile.negative_limit)
        self.log_writer.writerow({
            'ros_time': rospy.Time.now().to_sec(),
            'phase': phase,
            'torque_profile': profile_name,
            'track_mz_positive_limit': positive_limit,
            'track_mz_negative_limit': negative_limit,
            'segment': segment,
            'direction': direction,
            'turn_index': turn_index,
            'quarter_index': quarter_index,
            'target_unwrapped_yaw_deg': math.degrees(target_unwrapped),
            'imu_x': imu_position[0],
            'imu_y': imu_position[1],
            'imu_z': imu_position[2],
            'base_x': base_position[0],
            'base_y': base_position[1],
            'base_z': base_position[2],
            'target_x': command_target.x,
            'target_y': command_target.y,
            'target_z': command_target.z,
            'wrapped_yaw_deg': math.degrees(imu_yaw),
            'unwrapped_yaw_deg': math.degrees(self.unwrapped_yaw),
            'yaw_error_deg': math.degrees(yaw_error),
            'raw_r_deg_s': math.degrees(raw_r),
            'tf_yaw_rate_deg_s': math.degrees(yaw_rate),
            'horizontal_speed': math.hypot(velocity[0], velocity[1]),
            'command_mode': command_mode,
            'command_mz': command_mz,
            'controller_phase': controller_phase,
            'stopping_angle_deg': math.degrees(stopping_angle),
            'reported_mode': self.reported_mode,
        })
        self.log_file.flush()

    def _hold_locked_position(
            self, duration, label, target_yaw, profile=None,
            segment=0, direction=0, turn_index=0, quarter_index=0):
        """返回锁定位置，完成判稳后再连续保持指定秒数。"""
        duration = float(duration)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError('{} 连续稳定时间不合法'.format(label))
        if not self._update_tf() or not self._feedback_fresh():
            raise RuntimeError('{} 前反馈无效'.format(label))
        if self.locked_initial_pose is None:
            raise RuntimeError('{} 前尚未锁定初始位置'.format(label))
        command = self._make_command(
            MODE_DPROV, self.locked_initial_pose, target_yaw, mz=0)
        stable_count = 0
        stable_started_at = None
        stable_elapsed = 0.0
        started_at = rospy.Time.now()
        deadline = started_at + rospy.Duration(
            self.step_timeout + duration)
        rate = rospy.Rate(self.control_rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            self._update_tf()
            if not self._feedback_fresh():
                raise RuntimeError('{} 时反馈超时'.format(label))
            self._assert_no_other_command_publisher()
            self._publish_command(command)
            velocity = self.filtered_velocity
            yaw_rate = abs(
                self.yaw_rate_feedback_sign * velocity[2])
            current_position = self.current_base_pose[0]
            locked_position = self.locked_initial_pose[0]
            position_error = math.hypot(
                current_position[0] - locked_position[0],
                current_position[1] - locked_position[1],
            )
            yaw_error = wrap_angle(
                target_yaw - self.current_imu_pose[2])
            stable = locked_handover_is_stable(
                self.reported_mode,
                position_error,
                math.hypot(velocity[0], velocity[1]),
                yaw_error,
                yaw_rate,
                MODE_DPROV,
                self.handover_position_tolerance,
                self.handover_speed_threshold,
                self.yaw_tolerance,
                self.yaw_rate_threshold,
            )
            stable_count, stable_started_at, stable_elapsed = (
                update_continuous_stability(
                    stable,
                    stable_count,
                    stable_started_at,
                    now.to_sec(),
                    self.stable_frames,
                )
            )
            self._log_cycle(
                label,
                profile,
                segment,
                direction,
                turn_index,
                quarter_index,
                target_yaw,
                yaw_error,
                MODE_DPROV,
                0,
                'HOLD',
                0.0,
            )
            if (
                    stable_count >= self.stable_frames
                    and stable_elapsed >= duration):
                return
            if now >= deadline:
                raise RuntimeError(
                    '{} mode=4 稳定接管超时，连续稳定 {:.1f}/{:.1f} s'.format(
                        label, stable_elapsed, duration))
            rospy.loginfo_throttle(
                2.0,
                '%s: %s，位置误差=%.2f m，航向误差=%.1f deg，'
                '水平速度=%.3f m/s，角速度=%.2f deg/s，稳定帧=%d/%d，'
                '连续稳定=%.1f/%.1f s',
                NODE_NAME,
                label,
                position_error,
                math.degrees(yaw_error),
                math.hypot(velocity[0], velocity[1]),
                math.degrees(yaw_rate),
                stable_count,
                self.stable_frames,
                stable_elapsed,
                duration,
            )
            rate.sleep()

    def _rotate_segment(
            self, profile, segment, direction,
            turn_index, quarter_index, turn_angle=None,
            rotation_timeout=None, description=None):
        if not self._update_tf() or not self._feedback_fresh():
            raise RuntimeError('自由旋转前反馈无效')
        if self.locked_initial_pose is None:
            raise RuntimeError('自由旋转前尚未锁定初始位置')
        turn_angle = (
            self.turn_step
            if turn_angle is None
            else float(turn_angle)
        )
        rotation_timeout = (
            self.step_timeout
            if rotation_timeout is None
            else float(rotation_timeout)
        )
        if (
                not math.isfinite(turn_angle)
                or turn_angle <= 0.0
                or not math.isfinite(rotation_timeout)
                or rotation_timeout <= 0.0):
            raise ValueError('自由旋转角度和超时必须为有限正数')
        segment_start_position = self.locked_initial_pose[0]
        segment_start_yaw = self.unwrapped_yaw
        target_unwrapped = (
            segment_start_yaw + direction * turn_angle)
        stable_count = 0
        target_crossed = False
        moving_away_started_at = None
        deadline = rospy.Time.now() + rospy.Duration(rotation_timeout)
        rate = rospy.Rate(self.control_rate_hz)
        if description is None:
            description = '{}第 {} 圈第 {}/4 段'.format(
                '正向' if direction > 0 else '负向',
                turn_index,
                quarter_index,
            )
        rospy.loginfo(
            '%s: %s档 MZ(+%.0f/-%.0f)，段 %d/%d，'
            '%s，目标增量=%+.1f deg',
            NODE_NAME,
            profile.name,
            profile.positive_limit,
            profile.negative_limit,
            segment,
            self.total_segments,
            description,
            math.degrees(direction * turn_angle),
        )
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            self._update_tf()
            if not self._feedback_fresh():
                raise RuntimeError('自由旋转段 {} 反馈超时'.format(segment))
            self._assert_no_other_command_publisher()
            imu_position = self.current_imu_pose[0]
            drift = math.hypot(
                imu_position[0] - segment_start_position[0],
                imu_position[1] - segment_start_position[1],
            )
            if drift_exceeds_warning_threshold(
                    drift, self.drift_warning_radius):
                # 漂移不再中止旋转，否则异常接管会锁定漂移后的当前位置。
                # 继续完成本段目标，随后由 mode=4 返回全程初始位置。
                rospy.logwarn_throttle(
                    2.0,
                    '%s: 自由旋转段 %d IMU 漂移 %.2f m，超过告警半径 '
                    '%.2f m；继续完成本段目标，随后返回锁定初始位置',
                    NODE_NAME,
                    segment,
                    drift,
                    self.drift_warning_radius,
                )
            yaw_rate = (
                self.yaw_rate_feedback_sign * self.filtered_velocity[2])
            yaw_error = target_unwrapped - self.unwrapped_yaw
            if abs(yaw_rate) > self.max_yaw_rate:
                raise RuntimeError(
                    '自由旋转段 {} 角速度 {:.1f} deg/s 超过安全阈值 '
                    '{:.1f} deg/s'.format(
                        segment,
                        math.degrees(abs(yaw_rate)),
                        math.degrees(self.max_yaw_rate),
                    ))
            command_mz, controller_phase, stopping_angle = (
                direct_yaw_command(
                    yaw_error,
                    yaw_rate,
                    self.kp_yaw,
                    self.kd_yaw,
                    self.brake_gain_yaw,
                    profile.positive_limit,
                    profile.negative_limit,
                    self.brake_max_mz_positive,
                    self.brake_max_mz_negative,
                    self.brake_acceleration_positive,
                    self.brake_acceleration_negative,
                    self.control_delay,
                    self.brake_margin,
                    self.yaw_tolerance,
                    self.yaw_rate_threshold,
                    self.minimum_brake_mz,
                    self.mz_to_yaw_sign,
                )
            )
            command_mz, controller_phase, target_crossed = (
                apply_fixed_track_policy(
                    command_mz,
                    controller_phase,
                    yaw_error,
                    direction,
                    target_crossed,
                    self.mz_to_yaw_sign,
                    profile.positive_limit,
                    profile.negative_limit,
                )
            )
            moving_away = rotation_is_unexpectedly_moving_away(
                yaw_error,
                yaw_rate,
                direction,
                self.yaw_rate_threshold,
                target_crossed=target_crossed,
            )
            if moving_away:
                if moving_away_started_at is None:
                    moving_away_started_at = now
                elif (
                        now - moving_away_started_at
                ).to_sec() > self.moving_away_timeout:
                    raise RuntimeError(
                        '自由旋转段 {} 在尚未越过目标时连续 {:.1f} s '
                        '远离目标，检查 MZ 与 yaw 反馈符号'.format(
                            segment, self.moving_away_timeout))
            else:
                # 越过目标后仍按原方向运动属于正常刹转超调，
                # 不累计方向错误计时；未越过时的真实反向运动仍受保护。
                moving_away_started_at = None
            command = self._make_command(
                MODE_DEPTH,
                self.locked_initial_pose,
                target_unwrapped,
                mz=command_mz,
            )
            self._publish_command(command)
            self._log_cycle(
                'ROTATE',
                profile,
                segment,
                direction,
                turn_index,
                quarter_index,
                target_unwrapped,
                yaw_error,
                MODE_DEPTH,
                command_mz,
                controller_phase,
                stopping_angle,
            )
            horizontal_speed = math.hypot(
                self.filtered_velocity[0],
                self.filtered_velocity[1],
            )
            if controller_phase == 'TRACK':
                self._update_track_statistics(
                    profile.name,
                    direction,
                    yaw_rate,
                    horizontal_speed,
                    drift,
                )
            if (
                    controller_phase == 'TRACK'
                    and abs(yaw_rate) >= self.sample_min_yaw_rate):
                sample = CalibrationSample(
                    segment,
                    imu_position[0],
                    imu_position[1],
                    self.current_imu_pose[2],
                )
                self.direction_samples[
                    profile.name][int(direction)].append(sample)
            stable = (
                controller_phase == 'HOLD'
                and abs(yaw_error) <= self.yaw_tolerance
                and abs(yaw_rate) <= self.yaw_rate_threshold
            )
            stable_count = stable_count + 1 if stable else 0
            if stable_count >= self.stable_frames:
                return target_unwrapped
            if now >= deadline:
                raise RuntimeError(
                    '自由旋转段 {} 在 {:.1f} s 内未停稳'.format(
                        segment, rotation_timeout))
            rospy.loginfo_throttle(
                2.0,
                '%s: %s档段 %d %s，yaw_error=%.1f deg，'
                'yaw_rate=%.1f deg/s，MZ=%d，漂移=%.2f m',
                NODE_NAME,
                profile.name,
                segment,
                controller_phase,
                math.degrees(yaw_error),
                math.degrees(yaw_rate),
                command_mz,
                drift,
            )
            rate.sleep()

    def _write_direction_result(
            self, stream, indent, result, statistics):
        """写入单个档位、单个计划方向的拟合及运动统计。"""
        stream.write(
            '{}control_to_imu_x: {:.6f}\n'.format(
                indent, result['offset_x']))
        stream.write(
            '{}control_to_imu_y: {:.6f}\n'.format(
                indent, result['offset_y']))
        stream.write(
            '{}control_to_imu_z: {:.6f}\n'.format(
                indent, self.control_to_imu_z))
        stream.write(
            '{}radius: {:.6f}\n'.format(
                indent,
                math.hypot(result['offset_x'], result['offset_y'])))
        stream.write(
            '{}quality_score: {:.6f}\n'.format(
                indent, fit_quality_score(result)))
        stream.write(
            '{}rms_residual: {:.6f}\n'.format(
                indent, result['rms_residual']))
        stream.write(
            '{}max_residual: {:.6f}\n'.format(
                indent, result['max_residual']))
        stream.write(
            '{}sample_count: {}\n'.format(
                indent, result['sample_count']))
        stream.write(
            '{}used_sample_count: {}\n'.format(
                indent, result['used_sample_count']))
        stream.write(
            '{}rejected_sample_count: {}\n'.format(
                indent, result['rejected_sample_count']))
        stream.write(
            '{}track_cycle_count: {}\n'.format(
                indent, statistics['cycle_count']))
        stream.write(
            '{}mean_abs_yaw_rate_deg_s: {:.6f}\n'.format(
                indent,
                math.degrees(statistics['mean_abs_yaw_rate'])))
        stream.write(
            '{}peak_abs_yaw_rate_deg_s: {:.6f}\n'.format(
                indent,
                math.degrees(statistics['peak_abs_yaw_rate'])))
        stream.write(
            '{}mean_horizontal_speed: {:.6f}\n'.format(
                indent, statistics['mean_horizontal_speed']))
        stream.write(
            '{}peak_horizontal_speed: {:.6f}\n'.format(
                indent, statistics['peak_horizontal_speed']))
        stream.write(
            '{}max_drift_from_locked_position: {:.6f}\n'.format(
                indent, statistics['max_drift']))
        stream.write('{}segment_centers:\n'.format(indent))
        for segment in result['segments']:
            center = result['centers'][segment]
            stream.write(
                '{}  {}: [{:.6f}, {:.6f}]\n'.format(
                    indent, segment, center[0], center[1]))

    def _write_result(
            self, profile_results, profile_statistics,
            recommended_positive_profile,
            recommended_negative_profile):
        """写入可直接配置的推荐中心和三档完整对比结果。"""
        positive_result = profile_results[
            recommended_positive_profile]['positive']
        negative_result = profile_results[
            recommended_negative_profile]['negative']
        direction_difference = math.hypot(
            positive_result['offset_x'] - negative_result['offset_x'],
            positive_result['offset_y'] - negative_result['offset_y'],
        )
        profile_by_name = {
            profile.name: profile for profile in self.torque_profiles}
        with open(self.result_path, 'w', encoding='utf-8') as stream:
            stream.write(
                '# 由 {} 自动生成\n'.format(self.RESULT_GENERATOR))
            stream.write(
                '# 顶层 control_to_imu_* 是推荐结果，可写入 motion_supervisor\n')
            stream.write(
                '# 推荐优先采用中速；中速质量明显异常时按方向选择更稳定档位\n')
            stream.write(
                'recommended_positive_profile: {}\n'.format(
                    recommended_positive_profile))
            stream.write(
                'recommended_negative_profile: {}\n'.format(
                    recommended_negative_profile))
            stream.write(
                'control_to_imu_positive_x: {:.6f}\n'.format(
                    positive_result['offset_x']))
            stream.write(
                'control_to_imu_positive_y: {:.6f}\n'.format(
                    positive_result['offset_y']))
            stream.write(
                'control_to_imu_positive_z: {:.6f}\n'.format(
                    self.control_to_imu_z))
            stream.write(
                'control_to_imu_negative_x: {:.6f}\n'.format(
                    negative_result['offset_x']))
            stream.write(
                'control_to_imu_negative_y: {:.6f}\n'.format(
                    negative_result['offset_y']))
            stream.write(
                'control_to_imu_negative_z: {:.6f}\n'.format(
                    self.control_to_imu_z))
            stream.write(
                'direction_offset_difference: {:.6f}\n'.format(
                    direction_difference))
            stream.write('recommendation:\n')
            stream.write(
                '  preferred_profile: {}\n'.format(
                    self.preferred_profile))
            stream.write(
                '  degradation_ratio: {:.6f}\n'.format(
                    self.recommendation_degradation_ratio))
            stream.write(
                '  degradation_margin: {:.6f}\n'.format(
                    self.recommendation_degradation_margin))
            stream.write('profiles:\n')
            for profile in self.torque_profiles:
                name = profile.name
                results = profile_results[name]
                statistics = profile_statistics[name]
                stream.write('  {}:\n'.format(name))
                stream.write(
                    '    track_mz_positive_limit: {:.1f}\n'.format(
                        profile_by_name[name].positive_limit))
                stream.write(
                    '    track_mz_negative_limit: {:.1f}\n'.format(
                        profile_by_name[name].negative_limit))
                stream.write('    positive_direction:\n')
                self._write_direction_result(
                    stream,
                    '      ',
                    results['positive'],
                    statistics['positive'],
                )
                stream.write('    negative_direction:\n')
                self._write_direction_result(
                    stream,
                    '      ',
                    results['negative'],
                    statistics['negative'],
                )

    def emergency_hold(self):
        """异常或退出时发送当前位置 mode=4、六轴力清零。"""
        try:
            self._update_tf()
            if self.current_base_pose is None:
                return
            yaw = (
                self.current_imu_pose[2]
                if self.current_imu_pose is not None
                else 0.0
            )
            command = self._make_command(
                MODE_DPROV, self.current_base_pose, yaw, mz=0)
            for unused_index in range(3):
                self._publish_command(command)
            rospy.logwarn(
                '%s: 已发送当前位置 mode=4 紧急接管指令', NODE_NAME)
        except (rospy.ROSException, tf.Exception, ValueError):
            pass

    def _on_shutdown(self):
        if not self.completed:
            self.emergency_hold()
        self._close_log()

    def _prepare_calibration(self):
        """等待反馈，锁定唯一初始位置并完成启动接管。"""
        if not self._wait_for_feedback():
            raise RuntimeError('等待 TF、速度和模式反馈超时')
        self.locked_initial_pose = (
            tuple(self.current_base_pose[0]),
            tuple(self.current_base_pose[1]),
        )
        self.locked_initial_yaw = float(self.unwrapped_yaw)
        rospy.logwarn(
            '%s: 已锁定全程初始位姿，x=%.3f m，y=%.3f m，z目标=%.3f m，'
            'yaw=%.1f deg；后续不再用实时位置更新定点目标',
            NODE_NAME,
            self.locked_initial_pose[0][0],
            self.locked_initial_pose[0][1],
            self.target_z,
            math.degrees(self.locked_initial_yaw),
        )
        self._hold_locked_position(
            self.initial_hold_seconds,
            'INITIAL_HOLD',
            self.locked_initial_yaw,
        )

    def _finalize_calibration(self):
        """拟合三档正负方向结果，写入推荐配置并关闭日志。"""
        profile_results = {}
        finalized_statistics = {}
        for profile in self.torque_profiles:
            name = profile.name
            profile_results[name] = fit_planned_direction_centers(
                self.direction_samples[name][1],
                self.direction_samples[name][-1],
            )
            finalized_statistics[name] = {
                'positive': self._finalize_track_statistics(
                    self.profile_statistics[name][1]),
                'negative': self._finalize_track_statistics(
                    self.profile_statistics[name][-1]),
            }
        recommended_positive_profile = select_recommended_profile(
            {
                name: results['positive']
                for name, results in profile_results.items()
            },
            preferred=self.preferred_profile,
            degradation_ratio=self.recommendation_degradation_ratio,
            degradation_margin=self.recommendation_degradation_margin,
        )
        recommended_negative_profile = select_recommended_profile(
            {
                name: results['negative']
                for name, results in profile_results.items()
            },
            preferred=self.preferred_profile,
            degradation_ratio=self.recommendation_degradation_ratio,
            degradation_margin=self.recommendation_degradation_margin,
        )
        self._write_result(
            profile_results,
            finalized_statistics,
            recommended_positive_profile,
            recommended_negative_profile,
        )
        self.completed = True
        self._close_log()
        positive_result = profile_results[
            recommended_positive_profile]['positive']
        negative_result = profile_results[
            recommended_negative_profile]['negative']
        rospy.loginfo(
            '%s: 三档标定完成；推荐正向=%s档，方向控制中心 -> imu='
            '(%.4f, %.4f, %.4f) m，RMS=%.4f m；'
            '推荐负向=%s档，中心=(%.4f, %.4f, %.4f) m，RMS=%.4f m',
            NODE_NAME,
            recommended_positive_profile,
            positive_result['offset_x'],
            positive_result['offset_y'],
            self.control_to_imu_z,
            positive_result['rms_residual'],
            recommended_negative_profile,
            negative_result['offset_x'],
            negative_result['offset_y'],
            self.control_to_imu_z,
            negative_result['rms_residual'])
        rospy.loginfo('%s: 结果文件: %s', NODE_NAME, self.result_path)

    def run(self):
        profile_description = ', '.join(
            '{}(+{:.0f}/-{:.0f})'.format(
                profile.name,
                profile.positive_limit,
                profile.negative_limit,
            )
            for profile in self.torque_profiles
        )
        rospy.logwarn(
            '%s: 本程序将独占 /cmd/pose/ned；TX=TY 始终为 0，'
            '每档正反各 %d 圈；力矩档位=%s；原始日志: %s',
            NODE_NAME,
            self.turns_each_direction,
            profile_description,
            self.log_path,
        )
        self._prepare_calibration()

        segment = 0
        for profile in self.torque_profiles:
            rospy.logwarn(
                '%s: 开始%s档标定，TRACK MZ 正向上限=%.0f，'
                '负向上限=%.0f',
                NODE_NAME,
                profile.name,
                profile.positive_limit,
                profile.negative_limit,
            )
            for direction in (1, -1):
                for turn_index in range(
                        1, self.turns_each_direction + 1):
                    for quarter_index in range(1, 5):
                        segment += 1
                        target_yaw = self._rotate_segment(
                            profile,
                            segment,
                            direction,
                            turn_index,
                            quarter_index,
                        )
                        self._hold_locked_position(
                            self.handover_hold_seconds,
                            'SEGMENT_HOLD',
                            target_yaw,
                            profile,
                            segment,
                            direction,
                            turn_index,
                            quarter_index,
                        )

        self._finalize_calibration()


def main():
    rospy.init_node(NODE_NAME)
    calibration = None
    try:
        calibration = RotationCenterCalibration()
        calibration.run()
    except (
            ValueError,
            RuntimeError,
            OSError,
            IOError,
            ROSMasterException) as error:
        rospy.logfatal('%s: %s', NODE_NAME, error)
        if calibration is not None:
            calibration.emergency_hold()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
