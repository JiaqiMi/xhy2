#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：depth_wrench_calibration.py
功能：在下位机定深模式下自动执行 TX、TY、MZ 正反向阶跃标定并记录可观测速响应与稳态速度
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
2026.8.2
    修复预检未锁定标定基准时发布零输出导致的启动异常；预检阶段改用实时位姿
    构造零力保持指令，退出时按已锁定基准或最近预检位姿安全清零。
2026.8.2
    新增正反向巡航后的反向刹停自动测试，可记录刹停时间、轴向位移和刹后反向速度。
2026.8.2
    标定步骤改为“稳定基线—驱动/反向制动—零输出停稳”的原子流程；只有恢复停稳后
    才记录 PASS。新增数据有效性门槛、基线标准差、恢复指标和参数/版本元数据，避免
    残余运动、航向漂移及部署参数不明污染力矩标定结果。
2026.8.3
    预检、运行时安全门和配置错误均改为记录 ERROR 后保持零输出并等待恢复；不会因
    单次异常退出标定节点。所有静态档位也先驱动、再反向刹停，并拒绝刹车反向超调。
2026.8.3
    默认测试档改为不撤力的固定正反循环：每轴正向输出 7 s 后立即反向输出 7 s；
    不以速度、基线、超调或停稳作为流程分支，保留硬安全门并记录正反向稳态数据。
"""

from __future__ import division

import csv
import json
import math
import os
import subprocess
import sys
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
LOG_FORMAT_VERSION = 2


def wrap_angle(angle):
    """将角度规约到 [-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class CalibrationStep(object):
    """单个正向或反向力/力矩阶跃。"""

    def __init__(self, index, axis, command, percentage,
                 profile='static', brake_command=None,
                 brake_percentage=None):
        self.index = index
        self.axis = axis
        self.command = int(command)
        self.percentage = float(percentage)
        self.profile = profile
        self.brake_command = brake_command
        self.brake_percentage = brake_percentage


class DepthWrenchCalibration(object):
    """在 /cmd/pose/ned 独占条件下执行定深开环力/力矩标定。"""

    TRACE_FIELDS = (
        'ros_time', 'elapsed_s', 'step', 'profile', 'axis', 'command',
        'percentage', 'brake_command', 'brake_percentage', 'phase',
        'target_depth_m', 'current_x_m', 'current_y_m', 'current_depth_m',
        'depth_error_m', 'yaw_deg', 'yaw_offset_deg', 'displacement_m',
        'u_mps', 'v_mps', 'yaw_rate_rad_s', 'reported_mode',
        'velocity_age_s', 'status_age_s', 'tx', 'ty', 'mz',
    )

    SUMMARY_FIELDS = (
        'step', 'profile', 'axis', 'command', 'percentage', 'brake_command',
        'brake_percentage', 'started_at', 'finished_at', 'duration_s',
        'target_depth_m', 'baseline_axis_velocity',
        'velocity_observable_response_s',
        'steady_axis_velocity', 'peak_axis_velocity', 'peak_depth_error_m',
        'peak_displacement_m', 'peak_yaw_offset_deg', 'reported_mode',
        'brake_start_axis_velocity', 'brake_stop_time_s',
        'brake_stop_displacement', 'brake_reverse_peak_velocity',
        'reverse_steady_axis_velocity',
        'baseline_axis_stddev', 'recovery_time_s',
        'recovery_axis_velocity', 'recovery_horizontal_speed_mps',
        'recovery_yaw_rate_rad_s', 'metadata_path',
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
        self.test_profile = str(rospy.get_param(
            '~test_profile', 'cycle')).strip().lower()
        self.cycle_forward_seconds = float(rospy.get_param(
            '~cycle_forward_seconds', 7.0))
        self.cycle_reverse_seconds = float(rospy.get_param(
            '~cycle_reverse_seconds', 7.0))
        self.cycle_repeat_count = int(rospy.get_param(
            '~cycle_repeat_count', 0))
        self.reverse_brake_drive_seconds = float(rospy.get_param(
            '~reverse_brake_drive_seconds', 10.0))
        self.reverse_brake_timeout = float(rospy.get_param(
            '~reverse_brake_timeout', 12.0))
        self.reverse_brake_observe_seconds = float(rospy.get_param(
            '~reverse_brake_observe_seconds', 2.0))
        self.reverse_brake_min_speed = float(rospy.get_param(
            '~reverse_brake_min_speed', 0.02))
        self.reverse_brake_min_yaw_rate = float(rospy.get_param(
            '~reverse_brake_min_yaw_rate', math.radians(5.0)))
        self.reverse_brake_stop_speed = float(rospy.get_param(
            '~reverse_brake_stop_speed', 0.012))
        self.reverse_brake_stop_yaw_rate = float(rospy.get_param(
            '~reverse_brake_stop_yaw_rate', math.radians(0.5)))
        self.reverse_brake_max_reverse_speed = float(rospy.get_param(
            '~reverse_brake_max_reverse_speed', 0.012))
        self.reverse_brake_max_reverse_yaw_rate = float(rospy.get_param(
            '~reverse_brake_max_reverse_yaw_rate', math.radians(0.5)))
        self.baseline_wait_timeout = float(rospy.get_param(
            '~baseline_wait_timeout', 30.0))
        self.baseline_stable_seconds = float(rospy.get_param(
            '~baseline_stable_seconds', 2.0))
        self.baseline_max_horizontal_speed = float(rospy.get_param(
            '~baseline_max_horizontal_speed', 0.02))
        self.baseline_max_yaw_rate = float(rospy.get_param(
            '~baseline_max_yaw_rate', math.radians(2.0)))
        self.baseline_max_axis_stddev = float(rospy.get_param(
            '~baseline_max_axis_stddev', 0.005))
        self.recovery_timeout = float(rospy.get_param(
            '~recovery_timeout', 20.0))
        self.recovery_stable_seconds = float(rospy.get_param(
            '~recovery_stable_seconds', 2.0))
        self.recovery_max_horizontal_speed = float(rospy.get_param(
            '~recovery_max_horizontal_speed', 0.015))
        self.recovery_max_yaw_rate = float(rospy.get_param(
            '~recovery_max_yaw_rate', math.radians(0.5)))
        self.valid_max_depth_error = float(rospy.get_param(
            '~valid_max_depth_error', 0.05))
        self.valid_max_displacement = float(rospy.get_param(
            '~valid_max_displacement', 1.0))
        self.valid_max_yaw_offset = math.radians(float(rospy.get_param(
            '~valid_max_yaw_offset_deg', 10.0)))
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
        self.reverse_brake_drive_percentages = self._read_percentages(
            'reverse_brake_drive_percentages', (0.40, 0.60, 0.80))
        self.reverse_brake_percentage = float(rospy.get_param(
            '~reverse_brake_percentage', 1.0))
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
        self.preflight_hold_pose = None
        self.started_at = None
        self.active_step = None
        self.trace_file = None
        self.trace_writer = None
        self.summary_file = None
        self.summary_writer = None
        self.metadata_path = None
        self.active_measurement = None
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
            self.cycle_forward_seconds, self.cycle_reverse_seconds,
            self.baseline_seconds, self.steady_window_seconds,
            self.reverse_brake_drive_seconds, self.reverse_brake_timeout,
            self.reverse_brake_observe_seconds, self.reverse_brake_min_speed,
            self.reverse_brake_min_yaw_rate, self.reverse_brake_stop_speed,
            self.reverse_brake_stop_yaw_rate,
            self.reverse_brake_max_reverse_speed,
            self.reverse_brake_max_reverse_yaw_rate,
            self.reverse_brake_percentage,
            self.baseline_wait_timeout, self.baseline_stable_seconds,
            self.baseline_max_horizontal_speed, self.baseline_max_yaw_rate,
            self.baseline_max_axis_stddev, self.recovery_timeout,
            self.recovery_stable_seconds, self.recovery_max_horizontal_speed,
            self.recovery_max_yaw_rate, self.valid_max_depth_error,
            self.valid_max_displacement, self.valid_max_yaw_offset,
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
                self.reverse_brake_drive_seconds, self.reverse_brake_timeout,
                self.reverse_brake_min_speed,
                self.reverse_brake_min_yaw_rate,
                self.reverse_brake_stop_speed,
                self.reverse_brake_stop_yaw_rate,
                self.reverse_brake_max_reverse_speed,
                self.reverse_brake_max_reverse_yaw_rate,
                self.startup_timeout, self.feedback_timeout,
                self.max_depth_error, self.max_horizontal_speed,
                self.max_yaw_rate, self.max_displacement,
                self.max_yaw_offset, self.response_speed_threshold,
                self.response_yaw_rate_threshold)):
            raise ValueError('标定时长、阈值和限制必须大于 0')
        if self.rest_seconds < 0.0:
            raise ValueError('rest_seconds 不能小于 0')
        if self.reverse_brake_observe_seconds < 0.0:
            raise ValueError('reverse_brake_observe_seconds 不能小于 0')
        if self.test_profile not in ('cycle', 'static', 'reverse_brake', 'both'):
            raise ValueError(
                'test_profile 只能是 cycle、static、reverse_brake 或 both')
        if self.cycle_repeat_count < 0:
            raise ValueError('cycle_repeat_count 只能为 0（无限循环）或正整数')
        if not 0.0 < self.reverse_brake_percentage <= 1.0:
            raise ValueError('reverse_brake_percentage 必须在 (0, 1] 内')
        if math.isinf(self.target_depth):
            raise ValueError('target_depth 只能是有限值或默认 NaN（锁定当前深度）')
        if self.steady_window_seconds > self.hold_seconds:
            raise ValueError('steady_window_seconds 不能大于 hold_seconds')
        if self.baseline_stable_seconds > self.baseline_wait_timeout:
            raise ValueError('baseline_stable_seconds 不能大于 baseline_wait_timeout')
        if self.recovery_stable_seconds > self.recovery_timeout:
            raise ValueError('recovery_stable_seconds 不能大于 recovery_timeout')
        if self.valid_max_depth_error > self.max_depth_error:
            raise ValueError('valid_max_depth_error 不能大于 max_depth_error')
        if self.valid_max_displacement > self.max_displacement:
            raise ValueError('valid_max_displacement 不能大于 max_displacement')
        if self.valid_max_yaw_offset > self.max_yaw_offset:
            raise ValueError('valid_max_yaw_offset_deg 不能大于 max_yaw_offset_deg')
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
        self.metadata_path = os.path.join(
            self.log_directory, 'depth_wrench_calibration_metadata_{}.json'.format(
                suffix))
        if sys.version_info[0] < 3:
            self.trace_file = open(trace_path, 'wb')
            self.summary_file = open(summary_path, 'wb')
        else:
            self.trace_file = open(trace_path, 'w', encoding='utf-8', newline='')
            self.summary_file = open(
                summary_path, 'w', encoding='utf-8', newline='')
        self.trace_writer = csv.DictWriter(
            self.trace_file, fieldnames=self.TRACE_FIELDS)
        self.trace_writer.writeheader()
        self.summary_writer = csv.DictWriter(
            self.summary_file, fieldnames=self.SUMMARY_FIELDS)
        self.summary_writer.writeheader()
        self.trace_file.flush()
        self.summary_file.flush()
        self._write_metadata()
        rospy.loginfo('%s: 逐帧日志 %s', NODE_NAME, trace_path)
        rospy.loginfo('%s: 摘要日志 %s', NODE_NAME, summary_path)
        rospy.loginfo('%s: 元数据日志 %s', NODE_NAME, self.metadata_path)

    @staticmethod
    def _source_revision():
        """尽力记录脚本所在 Git 仓库的提交号；安装包环境中允许为空。"""
        directory = os.path.dirname(os.path.abspath(__file__))
        while True:
            if os.path.isdir(os.path.join(directory, '.git')):
                try:
                    process = subprocess.Popen(
                        ['git', 'rev-parse', 'HEAD'], cwd=directory,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    revision, unused_error = process.communicate()
                    if process.returncode == 0:
                        return revision.decode('ascii').strip()
                except (OSError, UnicodeError):
                    return ''
                return ''
            parent = os.path.dirname(directory)
            if parent == directory:
                return ''
            directory = parent

    def _metadata_parameters(self):
        """返回会改变数据解释方式的全部标定参数。"""
        values = {}
        for name, value in sorted(self.__dict__.items()):
            if (name.startswith(('baseline_', 'recovery_', 'reverse_brake_',
                                 'cycle_',
                                 'valid_', 'max_', 'preflight_', 'response_'))
                    or name in ('publish_rate_hz', 'hold_seconds',
                                'rest_seconds', 'steady_window_seconds',
                                'test_profile', 'target_depth', 'required_mode',
                                'require_mode_feedback', 'force_percentages',
                                'axis_limits', 'feedback_timeout')):
                if isinstance(value, tuple):
                    values[name] = list(value)
                elif isinstance(value, dict):
                    values[name] = dict(value)
                elif isinstance(value, float) and not math.isfinite(value):
                    values[name] = None
                else:
                    values[name] = value
        return values

    def _write_metadata(self):
        """将格式版本、脚本版本与实际参数写入独立 JSON。"""
        metadata = {
            'format_version': LOG_FORMAT_VERSION,
            'node': NODE_NAME,
            'created_at': datetime.now().isoformat(),
            'script_path': os.path.abspath(__file__),
            'git_revision': self._source_revision(),
            'parameters': self._metadata_parameters(),
        }
        with open(self.metadata_path, 'w') as stream:
            stream.write(json.dumps(
                metadata, ensure_ascii=True, sort_keys=True, indent=2))
            stream.write('\n')

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
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown():
            deadline = rospy.Time.now() + rospy.Duration(self.startup_timeout)
            while not rospy.is_shutdown() and rospy.Time.now() < deadline:
                try:
                    self._assert_command_topic_exclusive()
                    pose = self._read_pose()
                except RuntimeError as error:
                    rospy.logerr_throttle(2.0, '%s: 预检等待：%s',
                                          NODE_NAME, error)
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
                        NODE_NAME, self.target_depth,
                        math.degrees(self.target_yaw))
                    return
                self._publish_preflight_hold(pose)
                rate.sleep()
            rospy.logerr(
                '%s: 预检等待超过 %.1f s，保持零输出并继续等待 mode=%d、TF 与稳定反馈',
                NODE_NAME, self.startup_timeout, self.required_mode)

    def _make_command(self, step, reference_pose=None):
        """构造 mode=2 定深指令，并仅在当前标定轴施加力/力矩。"""
        if self.initial_pose is not None and self.target_yaw is not None:
            reference_pose = self.initial_pose
            target_yaw = self.target_yaw
        elif reference_pose is not None:
            target_yaw = reference_pose[3]
        else:
            raise RuntimeError('尚未锁定标定基准位姿')
        target_depth = (
            self.target_depth
            if math.isfinite(self.target_depth) else reference_pose[2])
        command = PoseNEDcmd()
        command.mode = self.required_mode
        command.target.header.stamp = rospy.Time.now()
        command.target.header.frame_id = 'map'
        command.target.pose.position.x = reference_pose[0]
        command.target.pose.position.y = reference_pose[1]
        command.target.pose.position.z = target_depth
        quaternion = quaternion_from_euler(0.0, 0.0, target_yaw)
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

    def _publish_preflight_hold(self, pose):
        """在基准锁定前以实时位姿发布零力保持指令。"""
        self.preflight_hold_pose = pose
        self.command_pub.publish(self._make_command(
            None, reference_pose=pose))

    def _build_steps(self):
        """按测试档、轴、输出百分比和正负方向生成可复核序列。"""
        steps = []
        for axis in ('tx', 'ty', 'mz'):
            positive_limit, negative_limit = self.axis_limits[axis]
            if self.test_profile == 'cycle':
                for percentage in self.force_percentages:
                    command = int(round(positive_limit * percentage))
                    reverse_command = -int(round(
                        negative_limit * percentage))
                    if command == 0 or reverse_command == 0:
                        raise ValueError(
                            '{} 的 {:.3f} 循环档位生成了零输出'.format(
                                axis, percentage))
                    steps.append(CalibrationStep(
                        len(steps) + 1, axis, command, percentage,
                        profile='cycle', brake_command=reverse_command,
                        brake_percentage=percentage))
            if self.test_profile in ('static', 'both'):
                for percentage in self.force_percentages:
                    for sign in (1, -1):
                        limit = positive_limit if sign > 0 else negative_limit
                        command = sign * int(round(limit * percentage))
                        brake_limit = (
                            negative_limit if sign > 0 else positive_limit)
                        brake_command = -sign * int(round(
                            brake_limit * self.reverse_brake_percentage))
                        if command == 0 or brake_command == 0:
                            raise ValueError(
                                '{} 的 {:.3f} 档位生成了零输出'.format(
                                    axis, percentage))
                        steps.append(CalibrationStep(
                            len(steps) + 1, axis, command, percentage,
                            profile='static', brake_command=brake_command,
                            brake_percentage=self.reverse_brake_percentage))
            if self.test_profile in ('reverse_brake', 'both'):
                for percentage in self.reverse_brake_drive_percentages:
                    for sign in (1, -1):
                        drive_limit = (
                            positive_limit if sign > 0 else negative_limit)
                        brake_limit = (
                            negative_limit if sign > 0 else positive_limit)
                        command = sign * int(round(drive_limit * percentage))
                        brake_command = -sign * int(round(
                            brake_limit * self.reverse_brake_percentage))
                        if command == 0 or brake_command == 0:
                            raise ValueError(
                                '{} 的反向刹停档位生成了零输出'.format(axis))
                        steps.append(CalibrationStep(
                            len(steps) + 1, axis, command, percentage,
                            profile='reverse_brake',
                            brake_command=brake_command,
                            brake_percentage=self.reverse_brake_percentage))
        return tuple(steps)

    def _snapshot(self, step, phase, applied_command=None):
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
        command = (
            0 if step is None else
            (step.command if applied_command is None else applied_command))
        row = {
            'ros_time': '{:.9f}'.format(now),
            'elapsed_s': '{:.3f}'.format(now - self.started_at),
            'step': '' if step is None else step.index,
            'profile': '' if step is None else step.profile,
            'axis': '' if step is None else step.axis,
            'command': '' if step is None else command,
            'percentage': '' if step is None else '{:.3f}'.format(
                step.percentage),
            'brake_command': '' if step is None or step.brake_command is None else step.brake_command,
            'brake_percentage': (
                '' if step is None or step.brake_percentage is None else
                '{:.3f}'.format(step.brake_percentage)),
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
            'tx': 0 if step is None or step.axis != 'tx' else command,
            'ty': 0 if step is None or step.axis != 'ty' else command,
            'mz': 0 if step is None or step.axis != 'mz' else command,
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

    def _assert_measurement_valid(self, pose):
        """拒绝虽未触发硬安全门、但已不适合用于标定的数据。"""
        depth_error = abs(pose[2] - self.target_depth)
        displacement = math.hypot(
            pose[0] - self.initial_pose[0], pose[1] - self.initial_pose[1])
        yaw_offset = abs(wrap_angle(pose[3] - self.target_yaw))
        if depth_error > self.valid_max_depth_error:
            raise RuntimeError('深度偏差超过标定有效性门槛 {:.3f} m'.format(
                depth_error))
        if displacement > self.valid_max_displacement:
            raise RuntimeError('水平位移超过标定有效性门槛 {:.3f} m'.format(
                displacement))
        if yaw_offset > self.valid_max_yaw_offset:
            raise RuntimeError('航向偏移超过标定有效性门槛 {:.1f} deg'.format(
                math.degrees(yaw_offset)))

    def _assert_sample_acceptable(self, velocity, pose, velocity_age, status_age):
        """统一执行硬安全和标定数据有效性检查。"""
        self._assert_safe(velocity, pose, velocity_age, status_age)
        self._assert_measurement_valid(pose)

    @staticmethod
    def _axis_velocity(step, velocity):
        """返回对应力/力矩轴的实测速度分量。"""
        return velocity[{'tx': 0, 'ty': 1, 'mz': 2}[step.axis]]

    @staticmethod
    def _standard_deviation(values):
        """返回有限样本的总体标准差，避免依赖额外科学计算库。"""
        if not values:
            return 0.0
        mean = sum(values) / float(len(values))
        return math.sqrt(sum(
            (value - mean) ** 2 for value in values) / float(len(values)))

    def _baseline_is_stable(self, velocity):
        """判断零输出时是否可作为下一档标定的稳定起点。"""
        return (math.hypot(velocity[0], velocity[1])
                <= self.baseline_max_horizontal_speed
                and abs(velocity[2]) <= self.baseline_max_yaw_rate)

    def _recovery_is_stable(self, step, velocity):
        """判断零输出后是否已停止，三轴使用统一的停稳持续时间逻辑。"""
        if step.axis == 'mz':
            return abs(velocity[2]) <= self.recovery_max_yaw_rate
        return (math.hypot(velocity[0], velocity[1])
                <= self.recovery_max_horizontal_speed)

    def _zero_hold(self, duration, phase, step=None):
        """显式输出零力并持续监视安全状态。"""
        deadline = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(None)
            unused_row, velocity, pose = self._snapshot(
                step, phase, applied_command=0)
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_sample_acceptable(
                velocity, pose, velocity_age, self._status_age())
            rate.sleep()

    def _baseline_velocity(self, step):
        """仅在持续停稳后采集零输出速度基线，避免残余滑行污染下一档。"""
        samples = []
        stable_started_at = None
        deadline = rospy.Time.now() + rospy.Duration(self.baseline_wait_timeout)
        sample_count = max(
            1, int(math.ceil(self.baseline_seconds * self.publish_rate_hz)))
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(None)
            unused_row, velocity, pose = self._snapshot(
                step, 'baseline', applied_command=0)
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_sample_acceptable(
                velocity, pose, velocity_age, self._status_age())
            now = rospy.Time.now().to_sec()
            if not self._baseline_is_stable(velocity):
                stable_started_at = None
                samples = []
            else:
                if stable_started_at is None:
                    stable_started_at = now
                if now - stable_started_at >= self.baseline_stable_seconds:
                    samples.append(self._axis_velocity(step, velocity))
                    if len(samples) >= sample_count:
                        standard_deviation = self._standard_deviation(samples)
                        if standard_deviation <= self.baseline_max_axis_stddev:
                            return (
                                sum(samples) / float(len(samples)),
                                standard_deviation)
                        stable_started_at = None
                        samples = []
            rate.sleep()
        raise RuntimeError(
            '%s=%+d 在 %.1f s 内未获得稳定零输出基线'.format(
                step.axis.upper(), step.command, self.baseline_wait_timeout))

    def _recover_step(self, step):
        """零输出直到连续停稳；失败即视为该档未完成，不能记录 PASS。"""
        started_at = rospy.Time.now().to_sec()
        stable_started_at = None
        deadline = rospy.Time.now() + rospy.Duration(self.recovery_timeout)
        rate = rospy.Rate(self.publish_rate_hz)
        last_velocity = None
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(None)
            unused_row, velocity, pose = self._snapshot(
                step, 'recovery', applied_command=0)
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_sample_acceptable(
                velocity, pose, velocity_age, self._status_age())
            last_velocity = velocity
            now = rospy.Time.now().to_sec()
            if self._recovery_is_stable(step, velocity):
                if stable_started_at is None:
                    stable_started_at = now
                if now - stable_started_at >= self.recovery_stable_seconds:
                    self._zero_hold(
                        self.rest_seconds, 'post_recovery_hold', step=step)
                    return {
                        'recovery_time_s': now - started_at,
                        'recovery_axis_velocity': self._axis_velocity(step, velocity),
                        'recovery_horizontal_speed_mps': math.hypot(
                            velocity[0], velocity[1]),
                        'recovery_yaw_rate_rad_s': velocity[2],
                    }
            else:
                stable_started_at = None
            rate.sleep()
        speed = (0.0 if last_velocity is None else math.hypot(
            last_velocity[0], last_velocity[1]))
        raise RuntimeError(
            '%s=%+d 在 %.1f s 内未零输出停稳，末端水平速度 %.4f m/s'.format(
                step.axis.upper(), step.command, self.recovery_timeout, speed))

    def _write_summary(self, step, started_at, finished_at, baseline,
                       response_latency, steady_values, peak_velocity,
                       peak_depth_error, peak_displacement, peak_yaw_offset,
                       result, reason, reverse_metrics=None,
                       baseline_stddev=None, recovery_metrics=None,
                       cycle_metrics=None):
        """写入单个阶跃的可直接分析摘要。"""
        row = {
            'step': step.index,
            'profile': step.profile,
            'axis': step.axis,
            'command': step.command,
            'percentage': '{:.3f}'.format(step.percentage),
            'brake_command': (
                '' if step.brake_command is None else step.brake_command),
            'brake_percentage': (
                '' if step.brake_percentage is None else
                '{:.3f}'.format(step.brake_percentage)),
            'started_at': '{:.9f}'.format(started_at),
            'finished_at': '{:.9f}'.format(finished_at),
            'duration_s': '{:.3f}'.format(finished_at - started_at),
            'target_depth_m': self.target_depth,
            'baseline_axis_velocity': baseline,
            'velocity_observable_response_s': (
                '' if response_latency is None else response_latency),
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
        }
        row.update({
            'brake_start_axis_velocity': '',
            'brake_stop_time_s': '',
            'brake_stop_displacement': '',
            'brake_reverse_peak_velocity': '',
            'reverse_steady_axis_velocity': '',
            'baseline_axis_stddev': (
                '' if baseline_stddev is None else baseline_stddev),
            'recovery_time_s': '',
            'recovery_axis_velocity': '',
            'recovery_horizontal_speed_mps': '',
            'recovery_yaw_rate_rad_s': '',
            'metadata_path': self.metadata_path or '',
        })
        if reverse_metrics is not None:
            row.update(reverse_metrics)
        if recovery_metrics is not None:
            row.update(recovery_metrics)
        if cycle_metrics is not None:
            row.update(cycle_metrics)
        self.summary_writer.writerow(row)
        self.summary_file.flush()

    def _execute_cycle(self, step):
        """固定执行正向 7 s、反向 7 s，不插入撤力或停稳等待。"""
        started_at = rospy.Time.now().to_sec()
        steady_count = max(
            1, int(math.ceil(self.steady_window_seconds * self.publish_rate_hz)))
        forward_values = deque(maxlen=steady_count)
        reverse_values = deque(maxlen=steady_count)
        peak_velocity = 0.0
        peak_depth_error = 0.0
        peak_displacement = 0.0
        peak_yaw_offset = 0.0
        rate = rospy.Rate(self.publish_rate_hz)
        reverse_step = CalibrationStep(
            step.index, step.axis, step.brake_command, step.brake_percentage,
            profile=step.profile, brake_command=step.brake_command,
            brake_percentage=step.brake_percentage)
        phases = (
            (step, 'cycle_forward', self.cycle_forward_seconds, forward_values),
            (reverse_step, 'cycle_reverse', self.cycle_reverse_seconds,
             reverse_values),
        )
        for command_step, phase, duration, values in phases:
            deadline = rospy.Time.now() + rospy.Duration(duration)
            while not rospy.is_shutdown() and rospy.Time.now() < deadline:
                self._assert_command_topic_exclusive()
                self._publish_command(command_step)
                unused_row, velocity, pose = self._snapshot(command_step, phase)
                velocity_age = self._latest_velocity_snapshot()[1]
                self._assert_safe(velocity, pose, velocity_age, self._status_age())
                axis_velocity = self._axis_velocity(step, velocity)
                values.append(axis_velocity)
                peak_velocity = max(peak_velocity, abs(axis_velocity))
                peak_depth_error = max(peak_depth_error, abs(
                    pose[2] - self.target_depth))
                peak_displacement = max(peak_displacement, math.hypot(
                    pose[0] - self.initial_pose[0],
                    pose[1] - self.initial_pose[1]))
                peak_yaw_offset = max(peak_yaw_offset, abs(wrap_angle(
                    pose[3] - self.target_yaw)))
                rate.sleep()
        finished_at = rospy.Time.now().to_sec()
        self._write_summary(
            step, started_at, finished_at, 0.0, None, tuple(forward_values),
            peak_velocity, peak_depth_error, peak_displacement, peak_yaw_offset,
            'PASS', '固定正向/反向周期完成', cycle_metrics={
                'reverse_steady_axis_velocity': (
                    '' if not reverse_values else
                    sum(reverse_values) / float(len(reverse_values))),
            })

    def _execute_step(self, step):
        """执行静态阶跃并在零输出恢复停稳后，才确认该档通过。"""
        baseline, baseline_stddev = self._baseline_velocity(step)
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
        self.active_measurement = {
            'started_at': started_at,
            'baseline': baseline,
            'baseline_stddev': baseline_stddev,
            'response_latency': None,
            'steady_values': (),
            'peak_velocity': 0.0,
            'peak_depth_error': 0.0,
            'peak_displacement': 0.0,
            'peak_yaw_offset': 0.0,
        }
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(step)
            unused_row, velocity, pose = self._snapshot(step, 'excite')
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_sample_acceptable(
                velocity, pose, velocity_age, self._status_age())
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
            self.active_measurement.update({
                'response_latency': response_latency,
                'steady_values': tuple(steady_values),
                'peak_velocity': peak_velocity,
                'peak_depth_error': peak_depth_error,
                'peak_displacement': peak_displacement,
                'peak_yaw_offset': peak_yaw_offset,
            })
            rate.sleep()
        self.active_measurement.update({
            'response_latency': response_latency,
            'steady_values': tuple(steady_values),
            'peak_velocity': peak_velocity,
            'peak_depth_error': peak_depth_error,
            'peak_displacement': peak_displacement,
            'peak_yaw_offset': peak_yaw_offset,
        })
        drive_velocity = (sum(steady_values) / float(len(steady_values))
                          if steady_values else baseline)
        peak_metrics = {
            'peak_velocity': peak_velocity,
            'peak_depth_error': peak_depth_error,
            'peak_displacement': peak_displacement,
            'peak_yaw_offset': peak_yaw_offset,
        }
        reverse_metrics = self._brake_to_stop(
            step, baseline, drive_velocity, peak_metrics)
        peak_velocity = peak_metrics['peak_velocity']
        peak_depth_error = peak_metrics['peak_depth_error']
        peak_displacement = peak_metrics['peak_displacement']
        peak_yaw_offset = peak_metrics['peak_yaw_offset']
        self.active_measurement.update(peak_metrics)
        recovery_metrics = self._recover_step(step)
        finished_at = rospy.Time.now().to_sec()
        self._write_summary(
            step, started_at, finished_at, baseline, response_latency,
            tuple(steady_values), peak_velocity, peak_depth_error,
            peak_displacement, peak_yaw_offset, 'PASS', '完成、反向刹停并停稳',
            reverse_metrics=reverse_metrics, baseline_stddev=baseline_stddev,
            recovery_metrics=recovery_metrics)
        self.active_measurement = None
        rospy.loginfo(
            '%s: %s=%+d 完成，响应延迟=%s s，稳态速度=%.4f',
            NODE_NAME, step.axis.upper(), step.command,
            '未达到阈值' if response_latency is None else '{:.3f}'.format(
                response_latency),
            (sum(steady_values) / float(len(steady_values))
             if steady_values else 0.0))

    def _axis_displacement(self, axis, start_pose, current_pose):
        """返回从刹车开始到当前时刻沿标定轴的位移或航向变化。"""
        if axis == 'mz':
            return wrap_angle(current_pose[3] - start_pose[3])
        delta_x = current_pose[0] - start_pose[0]
        delta_y = current_pose[1] - start_pose[1]
        cosine = math.cos(self.target_yaw)
        sine = math.sin(self.target_yaw)
        if axis == 'tx':
            return cosine * delta_x + sine * delta_y
        return -sine * delta_x + cosine * delta_y

    def _brake_to_stop(self, step, baseline, drive_velocity, metrics):
        """先确认已有同向速度，再施加反向力矩并拒绝反向超调。"""
        if step.brake_command is None:
            raise RuntimeError('{}=%+d 未配置反向刹车力矩'.format(
                step.axis.upper(), step.command))
        drive_sign = math.copysign(1.0, step.command)
        minimum_speed = (
            self.reverse_brake_min_yaw_rate
            if step.axis == 'mz' else self.reverse_brake_min_speed)
        stop_threshold = (
            self.reverse_brake_stop_yaw_rate
            if step.axis == 'mz' else self.reverse_brake_stop_speed)
        max_reverse_speed = (
            self.reverse_brake_max_reverse_yaw_rate
            if step.axis == 'mz' else self.reverse_brake_max_reverse_speed)
        if drive_sign * (drive_velocity - baseline) < minimum_speed:
            raise RuntimeError(
                '{}=%+d 驱动后速度不足 {:.4f}，不执行无效刹车'.format(
                    step.axis.upper(), step.command, minimum_speed))

        brake_step = CalibrationStep(
            step.index, step.axis, step.brake_command,
            step.brake_percentage, profile=step.profile,
            brake_command=step.brake_command,
            brake_percentage=step.brake_percentage)
        brake_started_at = rospy.Time.now().to_sec()
        brake_start_pose = self._read_pose()
        stop_at = None
        stop_pose = None
        reverse_peak_velocity = 0.0
        deadline = rospy.Time.now() + rospy.Duration(self.reverse_brake_timeout)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(brake_step)
            unused_row, velocity, pose = self._snapshot(
                brake_step, 'brake')
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_sample_acceptable(
                velocity, pose, velocity_age, self._status_age())
            axis_velocity = self._axis_velocity(step, velocity)
            signed_velocity = drive_sign * (axis_velocity - baseline)
            metrics['peak_velocity'] = max(
                metrics['peak_velocity'], abs(axis_velocity - baseline))
            metrics['peak_depth_error'] = max(
                metrics['peak_depth_error'], abs(pose[2] - self.target_depth))
            metrics['peak_displacement'] = max(
                metrics['peak_displacement'], math.hypot(
                    pose[0] - self.initial_pose[0], pose[1] - self.initial_pose[1]))
            metrics['peak_yaw_offset'] = max(
                metrics['peak_yaw_offset'], abs(wrap_angle(
                    pose[3] - self.target_yaw)))
            if signed_velocity < 0.0:
                reverse_peak_velocity = max(reverse_peak_velocity, -signed_velocity)
            if abs(axis_velocity - baseline) <= stop_threshold:
                stop_at = rospy.Time.now().to_sec()
                stop_pose = pose
                break
            rate.sleep()
        if stop_at is None or stop_pose is None:
            raise RuntimeError('{}=%+d 在 {:.1f} s 内未刹停'.format(
                step.axis.upper(), step.command, self.reverse_brake_timeout))

        observe_deadline = rospy.Time.now() + rospy.Duration(
            self.reverse_brake_observe_seconds)
        while not rospy.is_shutdown() and rospy.Time.now() < observe_deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(None)
            unused_row, velocity, pose = self._snapshot(
                step, 'brake_observe', applied_command=0)
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_sample_acceptable(
                velocity, pose, velocity_age, self._status_age())
            signed_velocity = drive_sign * (
                self._axis_velocity(step, velocity) - baseline)
            if signed_velocity < 0.0:
                reverse_peak_velocity = max(reverse_peak_velocity, -signed_velocity)
            rate.sleep()
        if reverse_peak_velocity > max_reverse_speed:
            raise RuntimeError('{}=%+d 刹车反向超调 {:.4f}，上限 {:.4f}'.format(
                step.axis.upper(), step.command, reverse_peak_velocity,
                max_reverse_speed))
        return {
            'brake_start_axis_velocity': drive_velocity,
            'brake_stop_time_s': stop_at - brake_started_at,
            'brake_stop_displacement': drive_sign * self._axis_displacement(
                step.axis, brake_start_pose, stop_pose),
            'brake_reverse_peak_velocity': reverse_peak_velocity,
        }

    def _execute_reverse_brake(self, step):
        """执行同向驱动、反向制动与零输出停稳，记录完整停车指标。"""
        baseline, baseline_stddev = self._baseline_velocity(step)
        started_at = rospy.Time.now().to_sec()
        drive_values = deque(maxlen=max(
            1, int(math.ceil(
                self.steady_window_seconds * self.publish_rate_hz))))
        peak_velocity = 0.0
        peak_depth_error = 0.0
        peak_displacement = 0.0
        peak_yaw_offset = 0.0
        rate = rospy.Rate(self.publish_rate_hz)
        self.active_measurement = {
            'started_at': started_at,
            'baseline': baseline,
            'baseline_stddev': baseline_stddev,
            'response_latency': None,
            'steady_values': (),
            'peak_velocity': 0.0,
            'peak_depth_error': 0.0,
            'peak_displacement': 0.0,
            'peak_yaw_offset': 0.0,
        }
        drive_deadline = rospy.Time.now() + rospy.Duration(
            self.reverse_brake_drive_seconds)
        latest_pose = None
        while not rospy.is_shutdown() and rospy.Time.now() < drive_deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(step)
            unused_row, velocity, pose = self._snapshot(
                step, 'reverse_brake_drive')
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_sample_acceptable(
                velocity, pose, velocity_age, self._status_age())
            axis_velocity = self._axis_velocity(step, velocity)
            drive_values.append(axis_velocity)
            peak_velocity = max(peak_velocity, abs(axis_velocity - baseline))
            peak_depth_error = max(peak_depth_error, abs(
                pose[2] - self.target_depth))
            peak_displacement = max(peak_displacement, math.hypot(
                pose[0] - self.initial_pose[0],
                pose[1] - self.initial_pose[1]))
            peak_yaw_offset = max(peak_yaw_offset, abs(wrap_angle(
                pose[3] - self.target_yaw)))
            latest_pose = pose
            rate.sleep()
        if not drive_values or latest_pose is None:
            raise RuntimeError('未采集到反向刹停的驱动速度')
        drive_velocity = sum(drive_values) / float(len(drive_values))
        drive_sign = math.copysign(1.0, step.command)
        minimum_speed = (
            self.reverse_brake_min_yaw_rate
            if step.axis == 'mz' else self.reverse_brake_min_speed)
        stop_threshold = (
            self.reverse_brake_stop_yaw_rate
            if step.axis == 'mz' else self.reverse_brake_stop_speed)
        if drive_sign * (drive_velocity - baseline) < minimum_speed:
            raise RuntimeError(
                '{}=%+d 驱动后速度不足 {:.4f}，拒绝反向刹停'.format(
                    step.axis.upper(), step.command, minimum_speed))

        brake_step = CalibrationStep(
            step.index, step.axis, step.brake_command,
            step.brake_percentage, profile=step.profile,
            brake_command=step.brake_command,
            brake_percentage=step.brake_percentage)
        brake_started_at = rospy.Time.now().to_sec()
        brake_start_pose = latest_pose
        stop_at = None
        stop_pose = None
        reverse_peak_velocity = 0.0
        brake_deadline = rospy.Time.now() + rospy.Duration(
            self.reverse_brake_timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < brake_deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(brake_step)
            unused_row, velocity, pose = self._snapshot(
                brake_step, 'reverse_brake_brake')
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_sample_acceptable(
                velocity, pose, velocity_age, self._status_age())
            axis_velocity = self._axis_velocity(step, velocity)
            signed_velocity = drive_sign * (axis_velocity - baseline)
            if signed_velocity < 0.0:
                reverse_peak_velocity = max(
                    reverse_peak_velocity, -signed_velocity)
            peak_velocity = max(peak_velocity, abs(axis_velocity - baseline))
            peak_depth_error = max(peak_depth_error, abs(
                pose[2] - self.target_depth))
            peak_displacement = max(peak_displacement, math.hypot(
                pose[0] - self.initial_pose[0],
                pose[1] - self.initial_pose[1]))
            peak_yaw_offset = max(peak_yaw_offset, abs(wrap_angle(
                pose[3] - self.target_yaw)))
            if abs(axis_velocity - baseline) <= stop_threshold:
                stop_at = rospy.Time.now().to_sec()
                stop_pose = pose
                break
            rate.sleep()
        if stop_at is None or stop_pose is None:
            raise RuntimeError(
                '{}=%+d 在 {:.1f} s 内未刹停'.format(
                    step.axis.upper(), step.command,
                    self.reverse_brake_timeout))

        observe_deadline = rospy.Time.now() + rospy.Duration(
            self.reverse_brake_observe_seconds)
        while not rospy.is_shutdown() and rospy.Time.now() < observe_deadline:
            self._assert_command_topic_exclusive()
            self._publish_command(None)
            unused_row, velocity, pose = self._snapshot(
                step, 'reverse_brake_observe', applied_command=0)
            velocity_age = self._latest_velocity_snapshot()[1]
            self._assert_sample_acceptable(
                velocity, pose, velocity_age, self._status_age())
            signed_velocity = drive_sign * (
                self._axis_velocity(step, velocity) - baseline)
            if signed_velocity < 0.0:
                reverse_peak_velocity = max(
                    reverse_peak_velocity, -signed_velocity)
            peak_depth_error = max(peak_depth_error, abs(
                pose[2] - self.target_depth))
            peak_displacement = max(peak_displacement, math.hypot(
                pose[0] - self.initial_pose[0],
                pose[1] - self.initial_pose[1]))
            peak_yaw_offset = max(peak_yaw_offset, abs(wrap_angle(
                pose[3] - self.target_yaw)))
            rate.sleep()

        finished_at = rospy.Time.now().to_sec()
        reverse_metrics = {
            'brake_start_axis_velocity': drive_velocity,
            'brake_stop_time_s': stop_at - brake_started_at,
            'brake_stop_displacement': drive_sign * self._axis_displacement(
                step.axis, brake_start_pose, stop_pose),
            'brake_reverse_peak_velocity': reverse_peak_velocity,
        }
        max_reverse_speed = (
            self.reverse_brake_max_reverse_yaw_rate
            if step.axis == 'mz' else self.reverse_brake_max_reverse_speed)
        if reverse_peak_velocity > max_reverse_speed:
            raise RuntimeError('{}=%+d 刹车反向超调 {:.4f}，上限 {:.4f}'.format(
                step.axis.upper(), step.command, reverse_peak_velocity,
                max_reverse_speed))
        self.active_measurement.update({
            'steady_values': tuple(drive_values),
            'peak_velocity': peak_velocity,
            'peak_depth_error': peak_depth_error,
            'peak_displacement': peak_displacement,
            'peak_yaw_offset': peak_yaw_offset,
        })
        recovery_metrics = self._recover_step(step)
        finished_at = rospy.Time.now().to_sec()
        self._write_summary(
            step, started_at, finished_at, baseline, None,
            tuple(drive_values), peak_velocity, peak_depth_error,
            peak_displacement, peak_yaw_offset, 'PASS', '反向刹停完成',
            reverse_metrics=reverse_metrics, baseline_stddev=baseline_stddev,
            recovery_metrics=recovery_metrics)
        self.active_measurement = None
        rospy.loginfo(
            '%s: %s=%+d 后反向 %d 刹停 %.3f s，位移=%.3f，反向峰值=%.4f',
            NODE_NAME, step.axis.upper(), step.command, step.brake_command,
            reverse_metrics['brake_stop_time_s'],
            reverse_metrics['brake_stop_displacement'],
            reverse_metrics['brake_reverse_peak_velocity'])

    def _abort_step(self, step, error):
        """将异常步骤及已采集部分写入摘要，避免恢复失败伪装成通过。"""
        now = rospy.Time.now().to_sec()
        measurement = self.active_measurement or {}
        self._write_summary(
            step, measurement.get('started_at', now), now,
            measurement.get('baseline', 0.0),
            measurement.get('response_latency'),
            measurement.get('steady_values', ()),
            measurement.get('peak_velocity', 0.0),
            measurement.get('peak_depth_error', 0.0),
            measurement.get('peak_displacement', 0.0),
            measurement.get('peak_yaw_offset', 0.0), 'FAIL', str(error),
            baseline_stddev=measurement.get('baseline_stddev'))
        self.active_measurement = None

    def _on_shutdown(self):
        """任何非正常退出都发送零输出，避免残留力/力矩。"""
        if not self.completed and not self.aborted:
            self.aborted = True
            try:
                self._publish_zero_burst()
            except (AttributeError, RuntimeError, rospy.ROSException):
                pass
        self._close_logs()

    def _publish_zero_burst(self):
        """连续发布零力/力矩，覆盖最近一次非零阶跃命令。"""
        rate = rospy.Rate(self.publish_rate_hz)
        for unused_index in range(3):
            if self.initial_pose is not None and self.target_yaw is not None:
                self._publish_command(None)
            elif self.preflight_hold_pose is not None:
                self._publish_preflight_hold(self.preflight_hold_pose)
            else:
                rospy.logwarn(
                    '%s: 未获取可用位姿，无法构造零力保持指令', NODE_NAME)
                return
            rate.sleep()

    def run(self):
        """按 TX、TY、MZ 的正反档位完成整套定深标定。"""
        self._wait_for_preflight()
        self.started_at = rospy.Time.now().to_sec()
        steps = self._build_steps()
        rospy.logwarn(
            '%s: 开始 %d 个定深力/力矩阶跃；必须保持控制话题独占',
            NODE_NAME, len(steps))
        round_index = 0
        while not rospy.is_shutdown():
            for step in steps:
                while not rospy.is_shutdown():
                    self.active_step = step
                    rospy.loginfo(
                        '%s: 第 %d 轮 [%d/%d] %s %s=%+d', NODE_NAME,
                        round_index + 1, step.index, len(steps),
                        step.profile, step.axis.upper(), step.command)
                    try:
                        if step.profile == 'cycle':
                            self._execute_cycle(step)
                        elif step.profile == 'reverse_brake':
                            self._execute_reverse_brake(step)
                        else:
                            self._execute_step(step)
                        self.active_step = None
                        break
                    except (RuntimeError, rospy.ROSException) as error:
                        rospy.logerr(
                            '%s: %s=%+d 失败：%s；已记录 FAIL，保持零输出并重新等待预检',
                            NODE_NAME, step.axis.upper(), step.command, error)
                        self._abort_step(step, error)
                        try:
                            self._publish_zero_burst()
                        except (RuntimeError, rospy.ROSException) as zero_error:
                            rospy.logerr('%s: 零输出保持失败：%s', NODE_NAME,
                                         zero_error)
                        self._wait_for_preflight()
            round_index += 1
            if (self.test_profile != 'cycle'
                    or (self.cycle_repeat_count > 0
                        and round_index >= self.cycle_repeat_count)):
                break
        if not rospy.is_shutdown():
            try:
                self.active_step = None
                self._zero_hold(self.rest_seconds, 'complete_hold')
                self.completed = True
                rospy.loginfo('%s: 标定完成，已持续输出零力/力矩', NODE_NAME)
            finally:
                self._close_logs()
        else:
            self._close_logs()


def main():
    rospy.init_node(NODE_NAME)
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        try:
            DepthWrenchCalibration().run()
            return
        except (
                ValueError,
                RuntimeError,
                OSError,
                IOError,
                ROSMasterException) as error:
            rospy.logerr('%s: 初始化或完成保持失败：%s；1 s 后继续等待',
                         NODE_NAME, error)
            rate.sleep()


if __name__ == '__main__':
    main()
