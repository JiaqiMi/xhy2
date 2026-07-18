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
    3. 默认每 90° 刹停并以当前位置 mode=4 接管，正反各三圈；
    4. 每个旋转段允许不同 map 圆心，最终拟合共享 control_link -> imu 杆臂；
    5. 程序结束后输出原始 CSV 和可直接写入 launch 的 YAML 结果。
记录：
2026.7.18
    新增自由旋转、主动刹转、定点接管和旋转中心自动拟合程序。
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
    direct_yaw_command,
    fit_segmented_rotation_center,
    unwrap_angle,
    wrap_angle,
)


NODE_NAME = 'rotation_center_calibration'
MODE_DEPTH = 2
MODE_DPROV = 4


class RotationCenterCalibration(object):
    """执行自由偏航并拟合 control_link 到 IMU 的水平杆臂。"""

    LOG_FIELDS = (
        'ros_time',
        'phase',
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
            rospy.get_param('~handover_hold_seconds', 2.0))
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
        self.max_mz_positive = float(
            rospy.get_param('~max_mz_positive', 1000.0))
        self.max_mz_negative = float(
            rospy.get_param('~max_mz_negative', 1000.0))
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
        self.drift_abort_radius = float(
            rospy.get_param('~drift_abort_radius', 0.75))
        self.sample_min_yaw_rate = math.radians(float(
            rospy.get_param('~sample_min_yaw_rate_deg_s', 1.0)))
        self.max_yaw_rate = math.radians(float(
            rospy.get_param('~max_yaw_rate_deg_s', 45.0)))
        self.moving_away_timeout = float(
            rospy.get_param('~moving_away_timeout', 3.0))
        self.direction_consistency_tolerance = float(rospy.get_param(
            '~direction_consistency_tolerance', 0.05))
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
            self.velocity_filter_alpha,
            self.yaw_rate_feedback_sign,
            self.mz_to_yaw_sign,
            self.kp_yaw,
            self.kd_yaw,
            self.brake_gain_yaw,
            self.max_mz_positive,
            self.max_mz_negative,
            self.brake_max_mz_positive,
            self.brake_max_mz_negative,
            self.brake_acceleration_positive,
            self.brake_acceleration_negative,
            self.control_delay,
            self.brake_margin,
            self.minimum_brake_mz,
            self.drift_abort_radius,
            self.sample_min_yaw_rate,
            self.max_yaw_rate,
            self.moving_away_timeout,
            self.direction_consistency_tolerance,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('旋转中心标定参数必须为有限值')
        if (
                self.control_rate_hz <= 0.0
                or self.turn_step <= 0.0
                or abs(self.turn_step - math.pi / 2.0) > 1e-6
                or self.turns_each_direction <= 0
                or self.step_timeout <= 0.0
                or self.feedback_timeout <= 0.0
                or self.stable_frames <= 0
                or not 0.0 < self.velocity_filter_alpha <= 1.0
                or self.drift_abort_radius <= 0.0
                or self.max_yaw_rate <= 0.0
                or self.moving_away_timeout <= 0.0
                or self.direction_consistency_tolerance <= 0.0):
            raise ValueError(
                '频率、角度、圈数、超时和漂移半径参数不在有效范围；'
                '当前 turn_step_deg 必须为 90')

        self.tf_listener = tf.TransformListener()
        self.raw_velocity = None
        self.filtered_velocity = None
        self.last_velocity_stamp = None
        self.reported_mode = None
        self.last_status_stamp = None
        self.current_imu_pose = None
        self.current_base_pose = None
        self.last_tf_stamp = None
        self.unwrapped_yaw = None
        self.samples = []
        self.segment_directions = {}
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
        stem = 'rotation_center_{0}'.format(
            datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
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
            imu_position, imu_rotation, imu_yaw = self._lookup_pose('imu')
            base_position, base_rotation, unused_base_yaw = (
                self._lookup_pose('base_link'))
            del unused_base_yaw
            stamp = self.tf_listener.getLatestCommonTime('map', 'imu')
        except (tf.Exception, ValueError) as error:
            rospy.logwarn_throttle(
                2.0, '%s: TF 更新失败: %s', NODE_NAME, error)
            return False
        self.current_imu_pose = (
            imu_position, imu_rotation, imu_yaw)
        self.current_base_pose = (
            base_position, base_rotation)
        self.last_tf_stamp = (
            stamp if stamp != rospy.Time(0) else rospy.Time.now())
        if self.unwrapped_yaw is None:
            self.unwrapped_yaw = imu_yaw
        else:
            self.unwrapped_yaw = unwrap_angle(
                self.unwrapped_yaw, imu_yaw)
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
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._update_tf()
            if self._feedback_fresh():
                return True
            rospy.loginfo_throttle(
                2.0,
                '%s: 等待 map->imu、map->base_link、速度和模式反馈',
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
            self, phase, segment, direction, turn_index, quarter_index,
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
        self.log_writer.writerow({
            'ros_time': rospy.Time.now().to_sec(),
            'phase': phase,
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

    def _hold_current(self, duration, label):
        if not self._update_tf() or not self._feedback_fresh():
            raise RuntimeError('{} 前反馈无效'.format(label))
        hold_pose = self.current_base_pose
        hold_yaw = self.current_imu_pose[2]
        command = self._make_command(
            MODE_DPROV, hold_pose, hold_yaw, mz=0)
        stable_count = 0
        started_at = rospy.Time.now()
        deadline = started_at + rospy.Duration(
            max(duration + 10.0, self.step_timeout))
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
            stable = (
                self.reported_mode == MODE_DPROV
                and math.hypot(velocity[0], velocity[1])
                <= self.handover_speed_threshold
                and yaw_rate <= self.yaw_rate_threshold
            )
            stable_count = stable_count + 1 if stable else 0
            self._log_cycle(
                label,
                0,
                0,
                0,
                0,
                self.unwrapped_yaw,
                0.0,
                MODE_DPROV,
                0,
                'HOLD',
                0.0,
            )
            if (
                    stable_count >= self.stable_frames
                    and (now - started_at).to_sec() >= duration):
                return
            if now >= deadline:
                raise RuntimeError('{} mode=4 稳定接管超时'.format(label))
            rate.sleep()

    def _rotate_segment(
            self, segment, direction, turn_index, quarter_index):
        if not self._update_tf() or not self._feedback_fresh():
            raise RuntimeError('自由旋转前反馈无效')
        segment_start_position = self.current_imu_pose[0]
        segment_start_yaw = self.unwrapped_yaw
        self.segment_directions[segment] = int(direction)
        target_unwrapped = (
            segment_start_yaw + direction * self.turn_step)
        stable_count = 0
        moving_away_started_at = None
        deadline = rospy.Time.now() + rospy.Duration(self.step_timeout)
        rate = rospy.Rate(self.control_rate_hz)
        rospy.loginfo(
            '%s: 段 %d，%s第 %d 圈第 %d/4 段，目标增量=%+.1f deg',
            NODE_NAME,
            segment,
            '正向' if direction > 0 else '负向',
            turn_index,
            quarter_index,
            math.degrees(direction * self.turn_step),
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
            if drift > self.drift_abort_radius:
                raise RuntimeError(
                    '自由旋转段 {} IMU 漂移 {:.2f} m，超过安全阈值 {:.2f} m'.format(
                        segment, drift, self.drift_abort_radius))
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
            moving_away = (
                yaw_error * yaw_rate < 0.0
                and abs(yaw_rate) > self.yaw_rate_threshold
            )
            if moving_away:
                if moving_away_started_at is None:
                    moving_away_started_at = now
                elif (
                        now - moving_away_started_at
                ).to_sec() > self.moving_away_timeout:
                    raise RuntimeError(
                        '自由旋转段 {} 连续 {:.1f} s 远离目标，'
                        '检查 MZ 与 yaw 反馈符号'.format(
                            segment, self.moving_away_timeout))
            else:
                moving_away_started_at = None
            command_mz, controller_phase, stopping_angle = (
                direct_yaw_command(
                    yaw_error,
                    yaw_rate,
                    self.kp_yaw,
                    self.kd_yaw,
                    self.brake_gain_yaw,
                    self.max_mz_positive,
                    self.max_mz_negative,
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
            command = self._make_command(
                MODE_DEPTH,
                self.current_base_pose,
                target_unwrapped,
                mz=command_mz,
            )
            self._publish_command(command)
            self._log_cycle(
                'ROTATE',
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
            if (
                    controller_phase != 'HOLD'
                    or abs(yaw_rate) >= self.sample_min_yaw_rate):
                self.samples.append(CalibrationSample(
                    segment,
                    imu_position[0],
                    imu_position[1],
                    self.current_imu_pose[2],
                ))
            stable = (
                controller_phase == 'HOLD'
                and abs(yaw_error) <= self.yaw_tolerance
                and abs(yaw_rate) <= self.yaw_rate_threshold
            )
            stable_count = stable_count + 1 if stable else 0
            if stable_count >= self.stable_frames:
                return
            if now >= deadline:
                raise RuntimeError(
                    '自由旋转段 {} 在 {:.1f} s 内未停稳'.format(
                        segment, self.step_timeout))
            rospy.loginfo_throttle(
                2.0,
                '%s: 段 %d %s，yaw_error=%.1f deg，'
                'yaw_rate=%.1f deg/s，MZ=%d，漂移=%.2f m',
                NODE_NAME,
                segment,
                controller_phase,
                math.degrees(yaw_error),
                math.degrees(yaw_rate),
                command_mz,
                drift,
            )
            rate.sleep()

    def _write_result(self, result, positive_result, negative_result):
        direction_difference = math.hypot(
            positive_result['offset_x'] - negative_result['offset_x'],
            positive_result['offset_y'] - negative_result['offset_y'],
        )
        with open(self.result_path, 'w', encoding='utf-8') as stream:
            stream.write(
                '# 由 rotation_center_calibration.py 自动生成\n')
            stream.write(
                'control_to_imu_x: {:.6f}\n'.format(result['offset_x']))
            stream.write(
                'control_to_imu_y: {:.6f}\n'.format(result['offset_y']))
            stream.write(
                'control_to_imu_z: {:.6f}\n'.format(
                    self.control_to_imu_z))
            stream.write(
                'radius: {:.6f}\n'.format(math.hypot(
                    result['offset_x'], result['offset_y'])))
            stream.write(
                'rms_residual: {:.6f}\n'.format(result['rms_residual']))
            stream.write(
                'max_residual: {:.6f}\n'.format(result['max_residual']))
            stream.write(
                'sample_count: {}\n'.format(result['sample_count']))
            stream.write(
                'used_sample_count: {}\n'.format(
                    result['used_sample_count']))
            stream.write(
                'rejected_sample_count: {}\n'.format(
                    result['rejected_sample_count']))
            stream.write('positive_direction:\n')
            stream.write(
                '  control_to_imu_x: {:.6f}\n'.format(
                    positive_result['offset_x']))
            stream.write(
                '  control_to_imu_y: {:.6f}\n'.format(
                    positive_result['offset_y']))
            stream.write(
                '  rms_residual: {:.6f}\n'.format(
                    positive_result['rms_residual']))
            stream.write('negative_direction:\n')
            stream.write(
                '  control_to_imu_x: {:.6f}\n'.format(
                    negative_result['offset_x']))
            stream.write(
                '  control_to_imu_y: {:.6f}\n'.format(
                    negative_result['offset_y']))
            stream.write(
                '  rms_residual: {:.6f}\n'.format(
                    negative_result['rms_residual']))
            stream.write(
                'direction_offset_difference: {:.6f}\n'.format(
                    direction_difference))
            stream.write('segment_centers:\n')
            for segment in result['segments']:
                center = result['centers'][segment]
                stream.write(
                    '  {}: [{:.6f}, {:.6f}]\n'.format(
                        segment, center[0], center[1]))

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

    def run(self):
        rospy.logwarn(
            '%s: 本程序将独占 /cmd/pose/ned；TX=TY 始终为 0，'
            '正反各 %d 圈，原始日志: %s',
            NODE_NAME,
            self.turns_each_direction,
            self.log_path,
        )
        if not self._wait_for_feedback():
            raise RuntimeError('等待 TF、速度和模式反馈超时')
        self._hold_current(self.initial_hold_seconds, 'INITIAL_HOLD')

        segment = 0
        for direction in (1, -1):
            for turn_index in range(1, self.turns_each_direction + 1):
                for quarter_index in range(1, 5):
                    segment += 1
                    self._rotate_segment(
                        segment,
                        direction,
                        turn_index,
                        quarter_index,
                    )
                    self._hold_current(
                        self.handover_hold_seconds,
                        'SEGMENT_HOLD',
                    )

        result = fit_segmented_rotation_center(self.samples)
        positive_result = fit_segmented_rotation_center([
            sample for sample in self.samples
            if self.segment_directions[sample.segment] > 0
        ])
        negative_result = fit_segmented_rotation_center([
            sample for sample in self.samples
            if self.segment_directions[sample.segment] < 0
        ])
        direction_difference = math.hypot(
            positive_result['offset_x'] - negative_result['offset_x'],
            positive_result['offset_y'] - negative_result['offset_y'],
        )
        self._write_result(result, positive_result, negative_result)
        self.completed = True
        self._close_log()
        rospy.loginfo(
            '%s: 标定完成，control_link -> imu=(%.4f, %.4f, 0.0) m，'
            '半径=%.4f m，RMS 残差=%.4f m',
            NODE_NAME,
            result['offset_x'],
            result['offset_y'],
            math.hypot(result['offset_x'], result['offset_y']),
            result['rms_residual'],
        )
        if direction_difference > self.direction_consistency_tolerance:
            rospy.logwarn(
                '%s: 正负方向杆臂差 %.4f m 超过阈值 %.4f m；'
                '本次结果不得直接写入正式参数',
                NODE_NAME,
                direction_difference,
                self.direction_consistency_tolerance,
            )
        rospy.loginfo('%s: 结果文件: %s', NODE_NAME, self.result_path)


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
