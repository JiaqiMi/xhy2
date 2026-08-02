#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：depth_wrench_calibration.py
功能：在下位机定深模式下自动执行 TX、TY、MZ 正反向阶跃标定并记录响应延迟与稳态速度
作者：BroXu
监听：
    /status/vel (geometry_msgs/TwistStamped)
    /status/auv (auv_control/AUVData)
    /tf (map -> base_link)
发布：
    /cmd/pose/ned (auv_control/PoseNEDcmd)
记录：
2026.8.2
    新增定深力/力矩自动标定节点；启动前检查控制话题独占，逐档输出 TX、TY、MZ，
    在深度、速度、位移和航向安全门槛内记录逐帧与摘要数据。
2026.8.2
    阶跃档位改为按各轴正负最大安全输出的百分比生成，并记录档位百分比；
    基线、激励、恢复和稳态统计时长均保留为 launch 可配置参数。
"""

from __future__ import division

import csv
import math
import os
import threading
from collections import deque
from datetime import datetime

import rosgraph
import rospy
import tf
from geometry_msgs.msg import TwistStamped
from rosgraph.masterapi import ROSMasterException
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from auv_control.msg import AUVData, PoseNEDcmd


NODE_NAME = 'depth_wrench_calibration'


def wrap_angle(angle):
    """将角度规约到 [-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class CalibrationStep(object):
    """单个正向或反向力/力矩阶跃。"""

    def __init__(self, index, axis, command, percentage):
        self.index = index
        self.axis = axis
        self.command = int(command)
        self.percentage = float(percentage)


class DepthWrenchCalibration(object):
    """在 /cmd/pose/ned 独占条件下执行定深开环力/力矩标定。"""

    TRACE_FIELDS = (
        'ros_time', 'elapsed_s', 'step', 'axis', 'command', 'percentage', 'phase',
        'target_depth_m', 'current_x_m', 'current_y_m', 'current_depth_m',
        'depth_error_m', 'yaw_deg', 'yaw_offset_deg', 'displacement_m',
        'u_mps', 'v_mps', 'yaw_rate_rad_s', 'reported_mode',
        'velocity_age_s', 'status_age_s', 'tx', 'ty', 'mz',
    )

    SUMMARY_FIELDS = (
        'step', 'axis', 'command', 'percentage', 'started_at', 'finished_at', 'duration_s',
        'target_depth_m', 'baseline_axis_velocity', 'response_latency_s',
        'steady_axis_velocity', 'peak_axis_velocity', 'peak_depth_error_m',
        'peak_displacement_m', 'peak_yaw_offset_deg', 'reported_mode',
        'result', 'reason',
    )

    AXIS_FIELDS = {
        'tx': 'TX',
        'ty': 'TY',
        'mz': 'MZ',
    }

    def __init__(self):
        self.enable_live_test = bool(rospy.get_param(
            '~enable_live_test', False))
        self.publish_rate_hz = float(rospy.get_param(
            '~publish_rate_hz', 10.0))
        self.hold_seconds = float(rospy.get_param('~hold_seconds', 8.0))
        self.rest_seconds = float(rospy.get_param('~rest_seconds', 6.0))
        self.baseline_seconds = float(rospy.get_param(
            '~baseline_seconds', 4.0))
        self.steady_window_seconds = float(rospy.get_param(
            '~steady_window_seconds', 3.0))
        self.startup_timeout = float(rospy.get_param(
            '~startup_timeout', 30.0))
        self.feedback_timeout = float(rospy.get_param(
            '~feedback_timeout', 0.5))
        self.max_depth_error = float(rospy.get_param(
            '~max_depth_error', 0.08))
        self.max_horizontal_speed = float(rospy.get_param(
            '~max_horizontal_speed', 0.35))
        self.max_yaw_rate = float(rospy.get_param(
            '~max_yaw_rate', 0.60))
        self.max_displacement = float(rospy.get_param(
            '~max_displacement', 4.0))
        self.max_yaw_offset = math.radians(float(rospy.get_param(
            '~max_yaw_offset_deg', 100.0)))
        self.preflight_depth_tolerance = float(rospy.get_param(
            '~preflight_depth_tolerance', 0.03))
        self.preflight_horizontal_speed = float(rospy.get_param(
            '~preflight_horizontal_speed', 0.02))
        self.preflight_yaw_rate = float(rospy.get_param(
            '~preflight_yaw_rate', math.radians(2.0)))
        self.response_speed_threshold = float(rospy.get_param(
            '~response_speed_threshold', 0.01))
        self.response_yaw_rate_threshold = float(rospy.get_param(
            '~response_yaw_rate_threshold', math.radians(1.0)))
        self.require_mode_feedback = bool(rospy.get_param(
            '~require_mode_feedback', True))
        self.required_mode = int(rospy.get_param('~required_mode', 2))
        self.target_depth = float(rospy.get_param(
            '~target_depth', float('nan')))
        self.log_directory = os.path.abspath(os.path.expanduser(str(
            rospy.get_param(
                '~log_directory',
                '~/.ros/auv_logs/depth_wrench_calibration'))))

        self.force_percentages = self._read_percentages(
            'force_percentages', (0.20, 0.40, 0.60, 0.80))
        self.axis_limits = {
            'tx': (
                int(rospy.get_param('~tx_max_positive', 3000)),
                int(rospy.get_param('~tx_max_negative', 3000))),
            'ty': (
                int(rospy.get_param('~ty_max_positive', 4000)),
                int(rospy.get_param('~ty_max_negative', 6000))),
            'mz': (
                int(rospy.get_param('~mz_max_positive', 3000)),
                int(rospy.get_param('~mz_max_negative', 3500))),
        }
        self._validate_parameters()

        self.lock = threading.RLock()
        self.tf_listener = tf.TransformListener()
        self.latest_velocity = None
        self.latest_velocity_at = None
        self.reported_mode = None
        self.latest_status_at = None
        self.initial_pose = None
        self.target_yaw = None
        self.started_at = None
        self.active_step = None
        self.trace_file = None
        self.trace_writer = None
        self.summary_file = None
        self.summary_writer = None
        self.completed = False
        self.aborted = False

        if not self.enable_live_test:
            raise ValueError(
                '拒绝启动实机力矩标定：必须显式设置 ~enable_live_test:=true')
        self._assert_command_topic_exclusive(before_start=True)
        self.command_pub = rospy.Publisher(
            '/cmd/pose/ned', PoseNEDcmd, queue_size=1)
        rospy.Subscriber(
            '/status/vel', TwistStamped, self._velocity_callback, queue_size=20)
        rospy.Subscriber(
            '/status/auv', AUVData, self._status_callback, queue_size=20)
        self._open_logs()
        rospy.on_shutdown(self._on_shutdown)

    def _read_percentages(self, parameter_name, default):
        """读取 0 到 1 的输出百分比档位。"""
        raw_percentages = rospy.get_param(
            '~' + parameter_name, list(default))
        if not isinstance(raw_percentages, (list, tuple)):
            raise ValueError('{} 必须是数值列表'.format(parameter_name))
        percentages = tuple(float(value) for value in raw_percentages)
        if (not percentages or any(
                not math.isfinite(value) or value <= 0.0 or value > 1.0
                for value in percentages)):
            raise ValueError('{} 必须包含 (0, 1] 内的数值'.format(
                parameter_name))
        if len(set(percentages)) != len(percentages):
            raise ValueError('{} 不允许重复档位'.format(parameter_name))
        return tuple(sorted(percentages))

    def _validate_parameters(self):
        """在输出前完成所有数值和安全边界校验。"""
        numeric = (
            self.publish_rate_hz, self.hold_seconds, self.rest_seconds,
            self.baseline_seconds, self.steady_window_seconds,
            self.startup_timeout, self.feedback_timeout,
            self.max_depth_error, self.max_horizontal_speed,
            self.max_yaw_rate, self.max_displacement, self.max_yaw_offset,
            self.preflight_depth_tolerance, self.preflight_horizontal_speed,
            self.preflight_yaw_rate,
            self.response_speed_threshold, self.response_yaw_rate_threshold,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('标定参数必须为有限数值')
        if any(value <= 0.0 for value in (
                self.publish_rate_hz, self.hold_seconds,
                self.baseline_seconds, self.steady_window_seconds,
                self.startup_timeout, self.feedback_timeout,
                self.max_depth_error, self.max_horizontal_speed,
                self.max_yaw_rate, self.max_displacement,
                self.max_yaw_offset, self.response_speed_threshold,
                self.response_yaw_rate_threshold)):
            raise ValueError('标定时长、阈值和限制必须大于 0')
        if self.rest_seconds < 0.0:
            raise ValueError('rest_seconds 不能小于 0')
        if math.isinf(self.target_depth):
            raise ValueError('target_depth 只能是有限值或默认 NaN（锁定当前深度）')
        if self.steady_window_seconds > self.hold_seconds:
            raise ValueError('steady_window_seconds 不能大于 hold_seconds')
        for axis, limits in self.axis_limits.items():
            if (len(limits) != 2 or any(value <= 0 or value > 30000
                                        for value in limits)):
                raise ValueError(
                    '{} 正负最大输出必须在 (0, 30000] 内'.format(axis))

    def _publishers_on_command_topic(self):
        """读取 ROS master 中当前 /cmd/pose/ned 发布者。"""
        master = rosgraph.Master(rospy.get_name())
        publishers, unused_subscribers, unused_services = (
            master.getSystemState())
        del unused_subscribers, unused_services
        for topic, nodes in publishers:
            if topic == '/cmd/pose/ned':
                return list(nodes)
        return []

    def _assert_command_topic_exclusive(self, before_start=False):
        """防止与 motion_supervisor 或人工节点争抢力矩控制话题。"""
        own_name = rospy.get_name()
        publishers = self._publishers_on_command_topic()
        others = [name for name in publishers if name != own_name]
        if others:
            raise RuntimeError(
                '/cmd/pose/ned 存在其他发布者: {}'.format(', '.join(others)))
        if before_start and publishers:
            raise RuntimeError(
                '启动标定前必须停止 motion_supervisor 和其他 '
                '/cmd/pose/ned 发布者')

    def _velocity_callback(self, message):
        """缓存原始本体系 u/v/r 速度，供阶跃响应与安全检查使用。"""
        values = (
            float(message.twist.linear.x),
            float(message.twist.linear.y),
            float(message.twist.angular.z),
        )
        if not all(math.isfinite(value) for value in values):
            rospy.logwarn_throttle(1.0, '%s: 忽略非有限速度反馈', NODE_NAME)
            return
        with self.lock:
            self.latest_velocity = values
            self.latest_velocity_at = rospy.Time.now()

    def _status_callback(self, message):
        """缓存下位机实际模式，确保持续处于定深模式。"""
        with self.lock:
            self.reported_mode = int(message.control_mode)
            self.latest_status_at = rospy.Time.now()

    def _open_logs(self):
        """创建逐帧轨迹与逐档摘要 CSV。"""
        if not os.path.isdir(self.log_directory):
            os.makedirs(self.log_directory)
        suffix = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        trace_path = os.path.join(
            self.log_directory, 'depth_wrench_calibration_trace_{}.csv'.format(
                suffix))
        summary_path = os.path.join(
            self.log_directory, 'depth_wrench_calibration_summary_{}.csv'.format(
                suffix))
        self.trace_file = open(trace_path, 'w', encoding='utf-8', newline='')
        self.trace_writer = csv.DictWriter(
            self.trace_file, fieldnames=self.TRACE_FIELDS)
        self.trace_writer.writeheader()
        self.summary_file = open(
            summary_path, 'w', encoding='utf-8', newline='')
        self.summary_writer = csv.DictWriter(
            self.summary_file, fieldnames=self.SUMMARY_FIELDS)
        self.summary_writer.writeheader()
        self.trace_file.flush()
        self.summary_file.flush()
        rospy.loginfo('%s: 逐帧日志 %s', NODE_NAME, trace_path)
        rospy.loginfo('%s: 摘要日志 %s', NODE_NAME, summary_path)

    def _close_logs(self):
        """安全关闭两个 CSV 文件。"""
        for attribute in ('trace_file', 'summary_file'):
            stream = getattr(self, attribute)
            if stream is not None:
                stream.flush()
                stream.close()
                setattr(self, attribute, None)

    def _latest_velocity_snapshot(self):
        """返回速度和反馈年龄；无新鲜数据时返回空值。"""
        with self.lock:
            velocity = self.latest_velocity
            stamp = self.latest_velocity_at
        if velocity is None or stamp is None:
            return None, float('inf')
        age = max(0.0, (rospy.Time.now() - stamp).to_sec())
        return velocity, age

    def _status_age(self):
        """返回下位机模式反馈年龄。"""
        with self.lock:
            stamp = self.latest_status_at
        if stamp is None:
            return float('inf')
        return max(0.0, (rospy.Time.now() - stamp).to_sec())

    def _read_pose(self):
        """读取 map 到 base_link 的实时位姿并验证其有限性。"""
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0))
        except tf.Exception as error:
            raise RuntimeError('无法读取 map -> base_link TF: {}'.format(error))
        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('map -> base_link TF 包含非有限值')
        yaw = euler_from_quaternion(rotation)[2]
        return (float(translation[0]), float(translation[1]),
                float(translation[2]), float(yaw))

    def _wait_for_preflight(self):
        """等待 TF、速度和模式反馈齐全，随后锁定标定深度与航向基准。"""
        deadline = rospy.Time.now() + rospy.Duration(self.startup_timeout)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_command_topic_exclusive()
            try:
                pose = self._read_pose()
            except RuntimeError:
                rate.sleep()
                continue
            velocity, velocity_age = self._latest_velocity_snapshot()
            status_age = self._status_age()
            mode_ok = (
                not self.require_mode_feedback
                or (status_age <= self.feedback_timeout
                    and self.reported_mode == self.required_mode))
            if not math.isfinite(self.target_depth):
                self.target_depth = pose[2]
            stable = velocity is not None and (
                abs(pose[2] - self.target_depth)
                <= self.preflight_depth_tolerance
                and math.hypot(velocity[0], velocity[1])
                <= self.preflight_horizontal_speed
                and abs(velocity[2]) <= self.preflight_yaw_rate)
            if (velocity is not None
                    and velocity_age <= self.feedback_timeout
                    and mode_ok and stable):
                self.initial_pose = pose
                self.target_yaw = pose[3]
                rospy.loginfo(
                    '%s: 预检通过，目标深度=%.3f m，起始航向=%.1f deg',
                    NODE_NAME, self.target_depth, math.degrees(self.target_yaw))
                return
            self._publish_command(None)
            rate.sleep()
        raise RuntimeError(
            '等待定深稳定、TF、速度或 mode={} 反馈超时'.format(
                self.required_mode))

    def _make_command(self, step):
        """构造 mode=2 定深指令，并仅在当前标定轴施加力/力矩。"""
        if self.initial_pose is None or self.target_yaw is None:
            raise RuntimeError('尚未锁定标定基准位姿')
        command = PoseNEDcmd()
        command.mode = self.required_mode
        command.target.header.stamp = rospy.Time.now()
        command.target.header.frame_id = 'map'
        command.target.pose.position.x = self.initial_pose[0]
        command.target.pose.position.y = self.initial_pose[1]
        command.target.pose.position.z = self.target_depth
        quaternion = quaternion_from_euler(0.0, 0.0, self.target_yaw)
        command.target.pose.orientation.x = quaternion[0]
        command.target.pose.orientation.y = quaternion[1]
        command.target.pose.orientation.z = quaternion[2]
        command.target.pose.orientation.w = quaternion[3]
        command.force.TX = 0
        command.force.TY = 0
        command.force.TZ = 0
        command.force.MX = 0
        command.force.MY = 0
        command.force.MZ = 0
        if step is not None:
            setattr(command.force, self.AXIS_FIELDS[step.axis], step.command)
        return command

    def _publish_command(self, step):
        """发布当前档位；所有未测试轴都显式清零。"""
        self.command_pub.publish(self._make_command(step))

    def _build_steps(self):
        """按轴、最大安全输出百分比和正负方向生成阶跃序列。"""
        steps = []
        for axis in ('tx', 'ty', 'mz'):
            positive_limit, negative_limit = self.axis_limits[axis]
            for percentage in self.force_percentages:
                for sign in (1, -1):
                    limit = positive_limit if sign > 0 else negative_limit
                    command = sign * int(round(limit * percentage))
                    if command == 0:
                        raise ValueError(
                            '{} 的 {:.3f} 档位生成了零输出'.format(
                                axis, percentage))
                    steps.append(CalibrationStep(
                        len(steps) + 1, axis, command, percentage))
        return tuple(steps)

    def _snapshot(self, step, phase):
        """读取一帧安全状态并写入逐帧数据。"""
        pose = self._read_pose()
        velocity, velocity_age = self._latest_velocity_snapshot()
        if velocity is None:
            raise RuntimeError('未收到速度反馈')
        status_age = self._status_age()
        displacement = math.hypot(
            pose[0] - self.initial_pose[0], pose[1] - self.initial_pose[1])
        yaw_offset = wrap_angle(pose[3] - self.target_yaw)
        now = rospy.Time.now().to_sec()
        row = {
            'ros_time': '{:.9f}'.format(now),
            'elapsed_s': '{:.3f}'.format(now - self.started_at),
            'step': '' if step is None else step.index,
            'axis': '' if step is None else step.axis,
            'command': '' if step is None else step.command,
            'percentage': '' if step is None else '{:.3f}'.format(
                step.percentage),
            'phase': phase,
            'target_depth_m': self.target_depth,
            'current_x_m': pose[0],
            'current_y_m': pose[1],
            'current_depth_m': pose[2],
            'depth_error_m': pose[2] - self.target_depth,
            'yaw_deg': math.degrees(pose[3]),
            'yaw_offset_deg': math.degrees(yaw_offset),
            'displacement_m': displacement,
            'u_mps': velocity[0],
            'v_mps': velocity[1],
            'yaw_rate_rad_s': velocity[2],
            'reported_mode': '' if self.reported_mode is None else self.reported_mode,
            'velocity_age_s': velocity_age,
            'status_age_s': status_age,
            'tx': 0 if step is None or step.axis != 'tx' else step.command,
            'ty': 0 if step is None or step.axis != 'ty' else step.command,
            'mz': 0 if step is None or step.axis != 'mz' else step.command,
        }
        self.trace_writer.writerow(row)
        self.trace_file.flush()
        return row, velocity, pose

    def _assert_safe(self, velocity, pose, velocity_age, status_age):
        """在每个控制周期检查深度、速度、位移、航向和模式安全门槛。"""
        if velocity_age > self.feedback_timeout:
            raise RuntimeError('速度反馈超时 {:.3f} s'.format(velocity_age))
        if self.require_mode_feedback:
            if status_age > self.feedback_timeout:
                raise RuntimeError('模式反馈超时 {:.3f} s'.format(status_age))
            if self.reported_mode != self.required_mode:
                raise RuntimeError(
                    '下位机模式异常：期望 {}，实际 {}'.format(
                        self.required_mode, self.reported_mode))
        depth_error = abs(pose[2] - self.target_depth)
        horizontal_speed = math.hypot(velocity[0], velocity[1])
        displacement = math.hypot(
            pose[0] - self.initial_pose[0], pose[1] - self.initial_pose[1])
        yaw_offset = abs(wrap_angle(pose[3] - self.target_yaw))
        if depth_error > self.max_depth_error:
            raise RuntimeError('深度偏差超限 {:.3f} m'.format(depth_error))
        if horizontal_speed > self.max_horizontal_speed:
            raise RuntimeError('水平速度超限 {:.3f} m/s'.format(horizontal_speed))
        if abs(velocity[2]) > self.max_yaw_rate:
            raise RuntimeError('航向角速度超限 {:.3f}'.format(abs(velocity[2])))
        if displacement > self.max_displacement:
            raise RuntimeError('水平位移超限 {:.3f} m'.format(displacement))
        if yaw_offset > self.max_yaw_offset:
            raise RuntimeError('航向偏移超限 {:.1f} deg'.format(
                math.degrees(yaw_offset)))

    @staticmethod
    def _axis_velocity(step, velocity):
        """返回对应力/力矩轴的实测速度分量。"""
        return velocity[{'tx': 0, 'ty': 1, 'mz': 2}[step.axis]]

    def _zero_hold(self, duration, phase):
        """显式输出零力并持续监视安全状态。"""
        deadline = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(None)
            unused_row, velocity, pose = self._snapshot(None, phase)
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_safe(velocity, pose, velocity_age, self._status_age())
            rate.sleep()

    def _baseline_velocity(self, step):
        """在零输出阶段取均值，消除环境流与传感器零偏。"""
        samples = []
        deadline = rospy.Time.now() + rospy.Duration(self.baseline_seconds)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(None)
            unused_row, velocity, pose = self._snapshot(step, 'baseline')
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_safe(velocity, pose, velocity_age, self._status_age())
            samples.append(self._axis_velocity(step, velocity))
            rate.sleep()
        if not samples:
            raise RuntimeError('未采集到零输出速度基线')
        return sum(samples) / float(len(samples))

    def _write_summary(self, step, started_at, finished_at, baseline,
                       response_latency, steady_values, peak_velocity,
                       peak_depth_error, peak_displacement, peak_yaw_offset,
                       result, reason):
        """写入单个阶跃的可直接分析摘要。"""
        self.summary_writer.writerow({
            'step': step.index,
            'axis': step.axis,
            'command': step.command,
            'percentage': '{:.3f}'.format(step.percentage),
            'started_at': '{:.9f}'.format(started_at),
            'finished_at': '{:.9f}'.format(finished_at),
            'duration_s': '{:.3f}'.format(finished_at - started_at),
            'target_depth_m': self.target_depth,
            'baseline_axis_velocity': baseline,
            'response_latency_s': '' if response_latency is None else response_latency,
            'steady_axis_velocity': (
                '' if not steady_values else
                sum(steady_values) / float(len(steady_values))),
            'peak_axis_velocity': peak_velocity,
            'peak_depth_error_m': peak_depth_error,
            'peak_displacement_m': peak_displacement,
            'peak_yaw_offset_deg': math.degrees(peak_yaw_offset),
            'reported_mode': '' if self.reported_mode is None else self.reported_mode,
            'result': result,
            'reason': reason,
        })
        self.summary_file.flush()

    def _execute_step(self, step):
        """执行一个恒定力/力矩阶跃，检测首个可观测速度响应。"""
        baseline = self._baseline_velocity(step)
        started_at = rospy.Time.now().to_sec()
        deadline = rospy.Time.now() + rospy.Duration(self.hold_seconds)
        response_latency = None
        steady_values = deque()
        peak_velocity = 0.0
        peak_depth_error = 0.0
        peak_displacement = 0.0
        peak_yaw_offset = 0.0
        threshold = (
            self.response_yaw_rate_threshold
            if step.axis == 'mz' else self.response_speed_threshold)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(step)
            unused_row, velocity, pose = self._snapshot(step, 'excite')
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_safe(velocity, pose, velocity_age, self._status_age())
            now = rospy.Time.now().to_sec()
            axis_velocity = self._axis_velocity(step, velocity)
            signed_response = math.copysign(
                1.0, step.command) * (axis_velocity - baseline)
            if response_latency is None and signed_response >= threshold:
                response_latency = now - started_at
            peak_velocity = max(peak_velocity, abs(axis_velocity - baseline))
            peak_depth_error = max(peak_depth_error, abs(pose[2] - self.target_depth))
            peak_displacement = max(peak_displacement, math.hypot(
                pose[0] - self.initial_pose[0], pose[1] - self.initial_pose[1]))
            peak_yaw_offset = max(peak_yaw_offset, abs(wrap_angle(
                pose[3] - self.target_yaw)))
            if now >= deadline.to_sec() - self.steady_window_seconds:
                steady_values.append(axis_velocity)
            rate.sleep()
        finished_at = rospy.Time.now().to_sec()
        self._write_summary(
            step, started_at, finished_at, baseline, response_latency,
            tuple(steady_values), peak_velocity, peak_depth_error,
            peak_displacement, peak_yaw_offset, 'PASS', '完成')
        rospy.loginfo(
            '%s: %s=%+d 完成，响应延迟=%s s，稳态速度=%.4f',
            NODE_NAME, step.axis.upper(), step.command,
            '未达到阈值' if response_latency is None else '{:.3f}'.format(
                response_latency),
            (sum(steady_values) / float(len(steady_values))
             if steady_values else 0.0))

    def _abort_step(self, step, error):
        """将异常步骤写入摘要，确保离线分析可区分中止原因。"""
        now = rospy.Time.now().to_sec()
        self._write_summary(
            step, now, now, 0.0, None, (), 0.0, 0.0, 0.0, 0.0,
            'FAIL', str(error))

    def _on_shutdown(self):
        """任何非正常退出都发送零输出，避免残留力/力矩。"""
        if not self.completed and not self.aborted:
            self.aborted = True
            try:
                self._publish_zero_burst()
            except (AttributeError, rospy.ROSException):
                pass
        self._close_logs()

    def _publish_zero_burst(self):
        """连续发布零力/力矩，覆盖最近一次非零阶跃命令。"""
        rate = rospy.Rate(self.publish_rate_hz)
        for unused_index in range(3):
            self._publish_command(None)
            rate.sleep()

    def run(self):
        """按 TX、TY、MZ 的正反档位完成整套定深标定。"""
        self._wait_for_preflight()
        self.started_at = rospy.Time.now().to_sec()
        steps = self._build_steps()
        rospy.logwarn(
            '%s: 开始 %d 个定深力/力矩阶跃；必须保持控制话题独占',
            NODE_NAME, len(steps))
        try:
            for step in steps:
                self.active_step = step
                rospy.loginfo(
                    '%s: [%d/%d] %s=%+d', NODE_NAME, step.index,
                    len(steps), step.axis.upper(), step.command)
                self._execute_step(step)
                self._zero_hold(self.rest_seconds, 'rest')
            self.active_step = None
            self._zero_hold(self.rest_seconds, 'complete_hold')
            self.completed = True
            rospy.loginfo('%s: 标定完成，已持续输出零力/力矩', NODE_NAME)
        except (RuntimeError, rospy.ROSException) as error:
            self.aborted = True
            if self.active_step is not None:
                self._abort_step(self.active_step, error)
            try:
                self._publish_zero_burst()
            except rospy.ROSException:
                pass
            raise
        finally:
            self._close_logs()


def main():
    rospy.init_node(NODE_NAME)
    try:
        DepthWrenchCalibration().run()
    except (
            ValueError,
            RuntimeError,
            OSError,
            IOError,
            ROSMasterException) as error:
        rospy.logfatal('%s: %s', NODE_NAME, error)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
