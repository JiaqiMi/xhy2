#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_axis_auto_test.py
功能：自动执行 motion_supervisor 的 X、Y 或 Yaw 单轴往返测试
作者：BroXu
监听：
    /motion/state (MotionState.msg)
    /tf
发布：
    /cmd/motion/goal (geometry_msgs/PoseStamped)
    /cmd/motion/cancel (std_msgs/Empty，仅异常退出时)
说明：
    1. 平移默认测试 ±0.5、±1.0、±1.5 m，航向默认测试 ±30°、±60°、±90°；
    2. 每个幅值重复三次，顺序为正向、原点、负向、原点；
    3. 每一步必须匹配当前目标并持续处于 HOVER 后才进入下一步；
    4. 本节点只发布 motion_supervisor 目标，不能替代 motion_supervisor。
记录：
2026.7.18
    新增 X、Y、Yaw 单轴自动水池测试执行器和摘要 CSV。
2026.7.19
    X/Y 测试增加实际位置验收、连续停稳等待、停滞诊断和刹停方向分类。
"""

from __future__ import division

import copy
import csv
import math
import os
import sys
from datetime import datetime

import rosgraph
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from rosgraph.masterapi import ROSMasterException
from std_msgs.msg import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from auv_control.msg import MotionState


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from motion_auto_sequence_core import (  # noqa: E402
    build_axis_sequence,
    classify_signed_stop_error,
    goal_matches,
    relative_goal,
    signed_axis_stop_error,
    xy_motion_is_stable,
    xy_motion_is_stalled,
)


NODE_NAME = 'motion_axis_auto_test'


class MotionAxisAutoTest(object):
    """按固定初始 base_link 坐标系执行单轴自动测试。"""

    LOG_FIELDS = (
        'step',
        'axis',
        'magnitude',
        'repetition',
        'phase',
        'offset',
        'target_x',
        'target_y',
        'target_z',
        'target_yaw_deg',
        'started_at',
        'finished_at',
        'duration_s',
        'acceptance_position_tolerance',
        'stable_speed_threshold',
        'required_stable_seconds',
        'peak_position_error',
        'peak_base_position_error',
        'peak_yaw_error_deg',
        'peak_horizontal_speed',
        'peak_yaw_rate_deg_s',
        'peak_abs_tx',
        'peak_abs_ty',
        'peak_abs_mz',
        'minimum_axis_position',
        'maximum_axis_position',
        'maximum_target_overshoot',
        'final_position_error',
        'final_base_position_error',
        'final_horizontal_speed',
        'final_axis_position',
        'signed_axis_stop_error',
        'stop_classification',
        'ever_crossed_target',
        'axis_position_unit',
        'result',
        'reason',
    )

    def __init__(self):
        self.axis = str(rospy.get_param('~axis', 'x')).strip().lower()
        self.repetitions = int(rospy.get_param('~repetitions', 3))
        self.target_z = float(rospy.get_param('~target_z', -0.9))
        self.publish_rate_hz = float(
            rospy.get_param('~publish_rate_hz', 5.0))
        self.startup_timeout = float(
            rospy.get_param('~startup_timeout', 60.0))
        self.action_timeout = float(
            rospy.get_param('~action_timeout', 120.0))
        self.hover_hold_seconds = float(
            rospy.get_param('~hover_hold_seconds', 2.0))
        self.start_delay = float(rospy.get_param('~start_delay', 2.0))
        self.feedback_timeout = float(
            rospy.get_param('~feedback_timeout', 0.5))
        self.goal_position_tolerance = float(
            rospy.get_param('~goal_position_tolerance', 0.05))
        self.goal_depth_tolerance = float(
            rospy.get_param('~goal_depth_tolerance', 0.05))
        self.goal_yaw_tolerance = math.radians(float(
            rospy.get_param('~goal_yaw_tolerance_deg', 2.0)))
        self.acceptance_position_tolerance = float(
            rospy.get_param('~acceptance_position_tolerance', 0.10))
        self.inter_step_stable_seconds = float(
            rospy.get_param('~inter_step_stable_seconds', 12.0))
        self.stable_speed_threshold = float(
            rospy.get_param('~stable_speed_threshold', 0.03))
        self.stall_detection_seconds = float(
            rospy.get_param('~stall_detection_seconds', 2.0))
        self.stall_speed_threshold = float(
            rospy.get_param('~stall_speed_threshold', 0.03))
        self.stall_force_threshold = float(
            rospy.get_param('~stall_force_threshold', 1.0))
        self.stop_classification_tolerance = float(
            rospy.get_param('~stop_classification_tolerance', 0.005))
        self.log_directory = os.path.abspath(os.path.expanduser(str(
            rospy.get_param(
                '~log_directory',
                '~/.ros/auv_logs/motion_auto_test'))))

        if self.axis in ('x', 'y'):
            magnitudes = rospy.get_param(
                '~translation_magnitudes', [0.5, 1.0, 1.5])
            self.magnitudes = tuple(float(value) for value in magnitudes)
        elif self.axis == 'yaw':
            degrees = rospy.get_param(
                '~yaw_magnitudes_deg', [30.0, 60.0, 90.0])
            self.magnitudes = tuple(
                math.radians(float(value)) for value in degrees)
        else:
            raise ValueError('axis 仅支持 x、y 或 yaw')

        numeric = (
            self.target_z,
            self.publish_rate_hz,
            self.startup_timeout,
            self.action_timeout,
            self.hover_hold_seconds,
            self.start_delay,
            self.feedback_timeout,
            self.goal_position_tolerance,
            self.goal_depth_tolerance,
            self.goal_yaw_tolerance,
            self.acceptance_position_tolerance,
            self.inter_step_stable_seconds,
            self.stable_speed_threshold,
            self.stall_detection_seconds,
            self.stall_speed_threshold,
            self.stall_force_threshold,
            self.stop_classification_tolerance,
        ) + self.magnitudes
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('测试参数必须是有限值')
        if (
                self.publish_rate_hz <= 0.0
                or self.startup_timeout <= 0.0
                or self.action_timeout <= 0.0
                or self.hover_hold_seconds < 0.0
                or self.feedback_timeout <= 0.0
                or self.acceptance_position_tolerance <= 0.0
                or self.stable_speed_threshold <= 0.0
                or self.stall_detection_seconds <= 0.0
                or self.stall_speed_threshold <= 0.0
                or self.stall_force_threshold < 0.0
                or self.stop_classification_tolerance < 0.0):
            raise ValueError('频率和超时必须为正数，稳定保持时间不能为负数')
        if (
                self.axis in ('x', 'y')
                and self.inter_step_stable_seconds < 10.0):
            raise ValueError('X/Y 段间连续稳定时间不能小于 10 s')
        self.required_stable_seconds = (
            max(self.hover_hold_seconds, self.inter_step_stable_seconds)
            if self.axis in ('x', 'y')
            else self.hover_hold_seconds
        )

        self.steps = build_axis_sequence(
            self.axis, self.magnitudes, self.repetitions)
        self.tf_listener = tf.TransformListener()
        self.latest_state = None
        self.latest_state_received_at = None
        self._assert_no_other_goal_publisher(before_start=True)
        self.goal_pub = rospy.Publisher(
            '/cmd/motion/goal', PoseStamped, queue_size=1)
        self.cancel_pub = rospy.Publisher(
            '/cmd/motion/cancel', Empty, queue_size=1)
        rospy.Subscriber(
            '/motion/state',
            MotionState,
            self.motion_state_callback,
            queue_size=10,
        )
        self.start_pose = None
        self.start_yaw = None
        self.summary_file = None
        self.summary_writer = None
        self.summary_path = ''
        self.completed = False
        self.aborted = False
        self._open_summary()
        rospy.on_shutdown(self._on_shutdown)

    def _publishers_on_goal_topic(self):
        master = rosgraph.Master(rospy.get_name())
        publishers, unused_subscribers, unused_services = (
            master.getSystemState())
        del unused_subscribers, unused_services
        for topic, nodes in publishers:
            if topic == '/cmd/motion/goal':
                return list(nodes)
        return []

    def _assert_no_other_goal_publisher(self, before_start=False):
        """防止任务节点或其他自动测试同时改写运动目标。"""
        own_name = rospy.get_name()
        publishers = self._publishers_on_goal_topic()
        others = [name for name in publishers if name != own_name]
        if others:
            raise RuntimeError(
                '/cmd/motion/goal 存在其他发布者 {}；'
                '必须停止其他任务和目标测试节点'.format(
                    ', '.join(others)))
        if before_start and publishers:
            raise RuntimeError(
                '启动自动测试前 /cmd/motion/goal 必须没有任何发布者')

    def _open_summary(self):
        if not os.path.isdir(self.log_directory):
            os.makedirs(self.log_directory)
        filename = 'motion_{0}_auto_test_{1}.csv'.format(
            self.axis,
            datetime.now().strftime('%Y%m%d_%H%M%S_%f'),
        )
        self.summary_path = os.path.join(self.log_directory, filename)
        self.summary_file = open(
            self.summary_path, 'w', encoding='utf-8', newline='')
        self.summary_writer = csv.DictWriter(
            self.summary_file, fieldnames=self.LOG_FIELDS)
        self.summary_writer.writeheader()
        self.summary_file.flush()

    def _close_summary(self):
        if self.summary_file is None:
            return
        self.summary_file.flush()
        self.summary_file.close()
        self.summary_file = None
        self.summary_writer = None

    def motion_state_callback(self, message):
        self.latest_state = copy.deepcopy(message)
        self.latest_state_received_at = rospy.Time.now()

    def _state_fresh(self):
        if self.latest_state_received_at is None:
            return False
        return (
            rospy.Time.now() - self.latest_state_received_at
        ).to_sec() <= self.feedback_timeout

    def _wait_for_supervisor(self):
        deadline = rospy.Time.now() + rospy.Duration(self.startup_timeout)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            state = self.latest_state
            if (
                    self._state_fresh()
                    and state is not None
                    and state.startup_complete
                    and state.state == MotionState.HOVER):
                return True
            rospy.loginfo_throttle(
                2.0,
                '%s: 等待 motion_supervisor 完成启动定点并进入 HOVER',
                NODE_NAME,
            )
            rate.sleep()
        return False

    def _lock_start_pose(self):
        try:
            self.tf_listener.waitForTransform(
                'map',
                'base_link',
                rospy.Time(0),
                rospy.Duration(self.startup_timeout),
            )
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0))
        except tf.Exception as error:
            raise RuntimeError(
                '无法获取初始 map -> base_link TF: {}'.format(error))
        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('初始 base_link TF 包含非有限值')
        self.start_pose = tuple(translation)
        self.start_yaw = euler_from_quaternion(rotation)[2]

    def _make_goal(self, step):
        target = relative_goal(
            self.start_pose[0],
            self.start_pose[1],
            self.start_yaw,
            self.target_z,
            self.axis,
            step.offset,
        )
        message = PoseStamped()
        message.header.frame_id = 'map'
        message.pose.position.x = target[0]
        message.pose.position.y = target[1]
        message.pose.position.z = target[2]
        quaternion = quaternion_from_euler(0.0, 0.0, target[3])
        message.pose.orientation.x = quaternion[0]
        message.pose.orientation.y = quaternion[1]
        message.pose.orientation.z = quaternion[2]
        message.pose.orientation.w = quaternion[3]
        return message, target

    def _state_matches_goal(self, target):
        state = self.latest_state
        if not self._state_fresh() or state is None:
            return False
        goal = state.goal.pose
        actual_yaw = euler_from_quaternion((
            goal.orientation.x,
            goal.orientation.y,
            goal.orientation.z,
            goal.orientation.w,
        ))[2]
        return goal_matches(
            goal.position.x,
            goal.position.y,
            goal.position.z,
            actual_yaw,
            target[0],
            target[1],
            target[2],
            target[3],
            self.goal_position_tolerance,
            self.goal_depth_tolerance,
            self.goal_yaw_tolerance,
        )

    @staticmethod
    def _new_peaks():
        return {
            'position_error': 0.0,
            'base_position_error': 0.0,
            'yaw_error': 0.0,
            'horizontal_speed': 0.0,
            'yaw_rate': 0.0,
            'tx': 0,
            'ty': 0,
            'mz': 0,
            'axis_minimum': None,
            'axis_maximum': None,
            'target_overshoot': 0.0,
        }

    def _current_axis_position(self):
        """返回相对锁定起点的 X、Y 位移或 yaw 偏差。"""
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0))
        except tf.Exception:
            return None
        if self.axis == 'yaw':
            current_yaw = euler_from_quaternion(rotation)[2]
            return (
                current_yaw - self.start_yaw + math.pi
            ) % (2.0 * math.pi) - math.pi
        delta_x = translation[0] - self.start_pose[0]
        delta_y = translation[1] - self.start_pose[1]
        if self.axis == 'x':
            return (
                math.cos(self.start_yaw) * delta_x
                + math.sin(self.start_yaw) * delta_y
            )
        return (
            -math.sin(self.start_yaw) * delta_x
            + math.cos(self.start_yaw) * delta_y
        )

    def _update_peaks(self, peaks, step):
        if not self._state_fresh() or self.latest_state is None:
            return
        state = self.latest_state
        peaks['position_error'] = max(
            peaks['position_error'], abs(state.position_error))
        peaks['base_position_error'] = max(
            peaks['base_position_error'], abs(state.base_position_error))
        peaks['yaw_error'] = max(
            peaks['yaw_error'], abs(state.yaw_error))
        peaks['horizontal_speed'] = max(
            peaks['horizontal_speed'], abs(state.horizontal_speed))
        peaks['yaw_rate'] = max(
            peaks['yaw_rate'], abs(state.yaw_rate))
        peaks['tx'] = max(peaks['tx'], abs(state.tx))
        peaks['ty'] = max(peaks['ty'], abs(state.ty))
        peaks['mz'] = max(peaks['mz'], abs(state.mz))
        axis_position = self._current_axis_position()
        if axis_position is None:
            return
        peaks['axis_minimum'] = (
            axis_position
            if peaks['axis_minimum'] is None
            else min(peaks['axis_minimum'], axis_position)
        )
        peaks['axis_maximum'] = (
            axis_position
            if peaks['axis_maximum'] is None
            else max(peaks['axis_maximum'], axis_position)
        )
        if step.offset > 0.0:
            overshoot = max(0.0, axis_position - step.offset)
        elif step.offset < 0.0:
            overshoot = max(0.0, step.offset - axis_position)
        elif step.phase == 'return_after_positive':
            overshoot = max(0.0, -axis_position)
        elif step.phase == 'return_after_negative':
            overshoot = max(0.0, axis_position)
        else:
            overshoot = 0.0
        peaks['target_overshoot'] = max(
            peaks['target_overshoot'], overshoot)

    def _write_result(
            self, step, target, started_at, finished_at,
            peaks, result, reason):
        state = self.latest_state
        final_axis_position = (
            self._current_axis_position()
            if self.axis in ('x', 'y')
            else None
        )
        if final_axis_position is None:
            final_signed_axis_error = None
            stop_classification = 'UNKNOWN' if self.axis in ('x', 'y') else ''
        else:
            final_signed_axis_error = signed_axis_stop_error(
                final_axis_position,
                step.offset,
                step.phase,
            )
            stop_classification = classify_signed_stop_error(
                final_signed_axis_error,
                self.stop_classification_tolerance,
            )
        self.summary_writer.writerow({
            'step': step.index,
            'axis': step.axis,
            'magnitude': (
                math.degrees(step.magnitude)
                if self.axis == 'yaw'
                else step.magnitude
            ),
            'repetition': step.repetition,
            'phase': step.phase,
            'offset': (
                math.degrees(step.offset)
                if self.axis == 'yaw'
                else step.offset
            ),
            'target_x': target[0],
            'target_y': target[1],
            'target_z': target[2],
            'target_yaw_deg': math.degrees(target[3]),
            'started_at': started_at.to_sec(),
            'finished_at': finished_at.to_sec(),
            'duration_s': (finished_at - started_at).to_sec(),
            'acceptance_position_tolerance': (
                self.acceptance_position_tolerance
                if self.axis in ('x', 'y')
                else ''
            ),
            'stable_speed_threshold': (
                self.stable_speed_threshold
                if self.axis in ('x', 'y')
                else ''
            ),
            'required_stable_seconds': self.required_stable_seconds,
            'peak_position_error': peaks['position_error'],
            'peak_base_position_error': peaks['base_position_error'],
            'peak_yaw_error_deg': math.degrees(peaks['yaw_error']),
            'peak_horizontal_speed': peaks['horizontal_speed'],
            'peak_yaw_rate_deg_s': math.degrees(peaks['yaw_rate']),
            'peak_abs_tx': peaks['tx'],
            'peak_abs_ty': peaks['ty'],
            'peak_abs_mz': peaks['mz'],
            'minimum_axis_position': (
                ''
                if peaks['axis_minimum'] is None
                else (
                    math.degrees(peaks['axis_minimum'])
                    if self.axis == 'yaw'
                    else peaks['axis_minimum']
                )
            ),
            'maximum_axis_position': (
                ''
                if peaks['axis_maximum'] is None
                else (
                    math.degrees(peaks['axis_maximum'])
                    if self.axis == 'yaw'
                    else peaks['axis_maximum']
                )
            ),
            'maximum_target_overshoot': (
                math.degrees(peaks['target_overshoot'])
                if self.axis == 'yaw'
                else peaks['target_overshoot']
            ),
            'final_position_error': (
                '' if state is None else state.position_error),
            'final_base_position_error': (
                '' if state is None else state.base_position_error),
            'final_horizontal_speed': (
                '' if state is None else state.horizontal_speed),
            'final_axis_position': (
                '' if final_axis_position is None else final_axis_position),
            'signed_axis_stop_error': (
                ''
                if final_signed_axis_error is None
                else final_signed_axis_error
            ),
            'stop_classification': stop_classification,
            'ever_crossed_target': (
                ''
                if self.axis == 'yaw'
                else int(
                    peaks['target_overshoot']
                    > self.stop_classification_tolerance)
            ),
            'axis_position_unit': (
                'deg' if self.axis == 'yaw' else 'm'),
            'result': result,
            'reason': reason,
        })
        self.summary_file.flush()

    def _execute_step(self, step):
        goal, target = self._make_goal(step)
        started_at = rospy.Time.now()
        deadline = started_at + rospy.Duration(self.action_timeout)
        stable_started_at = None
        stall_started_at = None
        peaks = self._new_peaks()
        rate = rospy.Rate(self.publish_rate_hz)
        rospy.loginfo(
            '%s: [%d/%d] %s 幅值=%.2f%s，第 %d 次，动作=%s',
            NODE_NAME,
            step.index,
            len(self.steps),
            self.axis.upper(),
            (
                math.degrees(step.magnitude)
                if self.axis == 'yaw'
                else step.magnitude
            ),
            ' deg' if self.axis == 'yaw' else ' m',
            step.repetition,
            step.phase,
        )
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            self._assert_no_other_goal_publisher()
            goal.header.stamp = now
            self.goal_pub.publish(goal)
            self._update_peaks(peaks, step)

            state = self.latest_state
            goal_confirmed = self._state_matches_goal(target)
            arrived = (
                self._state_fresh()
                and state is not None
                and state.startup_complete
                and state.state == MotionState.HOVER
                and goal_confirmed
            )
            if arrived and self.axis in ('x', 'y'):
                arrived = xy_motion_is_stable(
                    state.base_position_error,
                    state.horizontal_speed,
                    self.acceptance_position_tolerance,
                    self.stable_speed_threshold,
                )
            if arrived:
                if stable_started_at is None:
                    stable_started_at = now
                if (
                        now - stable_started_at
                ).to_sec() >= self.required_stable_seconds:
                    self._write_result(
                        step,
                        target,
                        started_at,
                        now,
                        peaks,
                        'PASS',
                        (
                            '实际位置与速度达标，连续稳定保持 '
                            '{:.1f} s 完成'.format(
                                self.required_stable_seconds)
                        ),
                    )
                    return True
            else:
                stable_started_at = None

            stalled = (
                self.axis in ('x', 'y')
                and self._state_fresh()
                and state is not None
                and state.startup_complete
                and goal_confirmed
                and xy_motion_is_stalled(
                    state.base_position_error,
                    state.horizontal_speed,
                    state.tx,
                    state.ty,
                    self.acceptance_position_tolerance,
                    self.stall_speed_threshold,
                    self.stall_force_threshold,
                )
            )
            if stalled:
                if stall_started_at is None:
                    stall_started_at = now
                if (
                        now - stall_started_at
                ).to_sec() >= self.stall_detection_seconds:
                    axis_position = self._current_axis_position()
                    signed_error = (
                        None
                        if axis_position is None
                        else signed_axis_stop_error(
                            axis_position,
                            step.offset,
                            step.phase,
                        )
                    )
                    reason = (
                        '检测到零输出停滞：二维误差={:.3f} m，'
                        '速度={:.4f} m/s，TX/TY={}/{}，'
                        '有符号轴向误差={}'.format(
                            state.base_position_error,
                            state.horizontal_speed,
                            state.tx,
                            state.ty,
                            (
                                '未知'
                                if signed_error is None
                                else '{:+.3f} m'.format(signed_error)
                            ),
                        )
                    )
                    self._write_result(
                        step,
                        target,
                        started_at,
                        now,
                        peaks,
                        'FAIL',
                        reason,
                    )
                    rospy.logerr('%s: %s', NODE_NAME, reason)
                    return False
            else:
                stall_started_at = None

            if now >= deadline:
                reason = (
                    '动作超时；最后状态={}'.format(
                        '无新鲜反馈'
                        if state is None or not self._state_fresh()
                        else state.reason
                    )
                )
                self._write_result(
                    step,
                    target,
                    started_at,
                    now,
                    peaks,
                    'FAIL',
                    reason,
                )
                rospy.logerr('%s: %s', NODE_NAME, reason)
                return False
            rospy.loginfo_throttle(
                2.0,
                '%s: 等待当前目标稳定；state=%s，'
                'position_error=%.3f m，speed=%.4f m/s，'
                'yaw_error=%.2f deg',
                NODE_NAME,
                (
                    '无新鲜反馈'
                    if state is None or not self._state_fresh()
                    else str(state.state)
                ),
                0.0 if state is None else state.base_position_error,
                0.0 if state is None else state.horizontal_speed,
                0.0 if state is None else math.degrees(state.yaw_error),
            )
            rate.sleep()
        return False

    def _cancel(self):
        self.cancel_pub.publish(Empty())
        rospy.logwarn('%s: 已发布 /cmd/motion/cancel', NODE_NAME)

    def _on_shutdown(self):
        if not self.completed and not self.aborted:
            self.aborted = True
            try:
                self._cancel()
            except rospy.ROSException:
                pass
        self._close_summary()

    def run(self):
        rospy.loginfo(
            '%s: 启动 %s 轴自动测试，共 %d 个动作；摘要日志: %s',
            NODE_NAME,
            self.axis.upper(),
            len(self.steps),
            self.summary_path,
        )
        if self.axis in ('x', 'y'):
            rospy.loginfo(
                '%s: X/Y 验收误差≤%.3f m、稳定速度≤%.4f m/s，'
                '每步出发前连续稳定 %.1f s',
                NODE_NAME,
                self.acceptance_position_tolerance,
                self.stable_speed_threshold,
                self.required_stable_seconds,
            )
        if not self._wait_for_supervisor():
            raise RuntimeError(
                '等待 motion_supervisor startup_complete + HOVER 超时')
        self._lock_start_pose()
        rospy.loginfo(
            '%s: 已锁定初始 base_link=(%.3f, %.3f, %.3f, %.1fdeg)，'
            '目标深度=%.2f m，%.1f s 后开始',
            NODE_NAME,
            self.start_pose[0],
            self.start_pose[1],
            self.start_pose[2],
            math.degrees(self.start_yaw),
            self.target_z,
            self.start_delay,
        )
        rospy.sleep(self.start_delay)
        for step in self.steps:
            if not self._execute_step(step):
                self.aborted = True
                self._cancel()
                raise RuntimeError(
                    '自动测试在第 {} 步中止'.format(step.index))
        self.completed = True
        self._close_summary()
        rospy.loginfo(
            '%s: %s 轴全部 %d 个动作完成，AUV 已回到原点并由 mode=4 接管',
            NODE_NAME,
            self.axis.upper(),
            len(self.steps),
        )


def main():
    rospy.init_node(NODE_NAME)
    try:
        MotionAxisAutoTest().run()
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
