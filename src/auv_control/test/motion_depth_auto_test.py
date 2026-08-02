#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_depth_auto_test.py
功能：经 motion_supervisor 下发绝对深度阶跃并由 TF 独立判断到达和稳定
作者：BroXu
监听：
    /motion/state (MotionState.msg)
    /tf
发布：
    /cmd/motion/goal (geometry_msgs/PoseStamped)
记录：
2026.7.31
    新增绝对深度目标与启动基准深度往返测试。
    motion_supervisor 负责接收并透传深度目标，测试器仅用 TF 完成到达判定。
"""

from __future__ import division

import copy
import csv
import math
import os
import sys
from collections import deque
from datetime import datetime

import rosgraph
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from rosgraph.masterapi import ROSMasterException
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from auv_control.msg import MotionState


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from motion_auto_sequence_core import goal_matches  # noqa: E402
from motion_depth_auto_test_core import (  # noqa: E402
    build_depth_sequence,
    depth_motion_is_stable,
    directed_depth_overshoot,
    linear_vertical_speed,
)


NODE_NAME = 'motion_depth_auto_test'


class MotionDepthAutoTest(object):
    """执行绝对深度阶跃，保持启动时锁定的水平位置和航向。"""

    LOG_FIELDS = (
        'step',
        'cycle',
        'phase',
        'requested_depth',
        'target_depth',
        'baseline_depth',
        'target_x',
        'target_y',
        'target_yaw_deg',
        'started_at',
        'finished_at',
        'duration_s',
        'acceptance_depth_tolerance',
        'stable_vertical_speed_threshold',
        'required_stable_seconds',
        'vertical_speed_window_seconds',
        'start_depth',
        'minimum_depth',
        'maximum_depth',
        'peak_abs_depth_error',
        'peak_abs_tf_vertical_speed',
        'maximum_target_overshoot',
        'ever_crossed_target',
        'peak_base_position_error',
        'peak_horizontal_speed',
        'peak_yaw_error_deg',
        'peak_yaw_rate_deg_s',
        'peak_abs_tx',
        'peak_abs_ty',
        'peak_abs_mz',
        'final_depth',
        'final_depth_error',
        'final_tf_vertical_speed',
        'final_base_position_error',
        'final_horizontal_speed',
        'final_yaw_error_deg',
        'result',
        'reason',
    )

    def __init__(self):
        self.target_depths = tuple(float(value) for value in rospy.get_param(
            '~target_depths', [-0.3, -0.6, -0.9]))
        self.cycle_count = int(rospy.get_param('~cycle_count', 2))
        self.publish_rate_hz = float(
            rospy.get_param('~publish_rate_hz', 5.0))
        self.startup_timeout = float(
            rospy.get_param('~startup_timeout', 60.0))
        self.action_timeout = float(
            rospy.get_param('~action_timeout', 120.0))
        self.start_delay = float(rospy.get_param('~start_delay', 2.0))
        self.feedback_timeout = float(
            rospy.get_param('~feedback_timeout', 0.5))
        self.tf_feedback_timeout = float(
            rospy.get_param('~tf_feedback_timeout', 0.5))
        self.goal_position_tolerance = float(
            rospy.get_param('~goal_position_tolerance', 0.05))
        self.goal_depth_tolerance = float(
            rospy.get_param('~goal_depth_tolerance', 0.05))
        self.goal_yaw_tolerance = math.radians(float(
            rospy.get_param('~goal_yaw_tolerance_deg', 2.0)))
        # data2 的稳态绝对误差 P95 为 0.095 m，默认取 0.10 m。
        self.acceptance_depth_tolerance = float(
            rospy.get_param('~acceptance_depth_tolerance', 0.10))
        # data2 的稳态垂向速度绝对值 P99 为 0.0119 m/s，留出 TF 噪声余量。
        self.stable_vertical_speed_threshold = float(rospy.get_param(
            '~stable_vertical_speed_threshold', 0.015))
        self.required_stable_seconds = float(
            rospy.get_param('~required_stable_seconds', 5.0))
        self.vertical_speed_window_seconds = float(rospy.get_param(
            '~vertical_speed_window_seconds', 2.0))
        self.minimum_speed_window_seconds = float(rospy.get_param(
            '~minimum_speed_window_seconds', 1.0))
        self.log_directory = os.path.abspath(os.path.expanduser(str(
            rospy.get_param(
                '~log_directory',
                '~/.ros/auv_logs/motion_auto_test'))))

        numeric = (
            self.publish_rate_hz,
            self.startup_timeout,
            self.action_timeout,
            self.start_delay,
            self.feedback_timeout,
            self.tf_feedback_timeout,
            self.goal_position_tolerance,
            self.goal_depth_tolerance,
            self.goal_yaw_tolerance,
            self.acceptance_depth_tolerance,
            self.stable_vertical_speed_threshold,
            self.required_stable_seconds,
            self.vertical_speed_window_seconds,
            self.minimum_speed_window_seconds,
        ) + self.target_depths
        if not self.target_depths:
            raise ValueError('target_depths 不能为空')
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('深度测试参数必须为有限值')
        if (
                self.cycle_count <= 0
                or self.publish_rate_hz <= 0.0
                or self.startup_timeout <= 0.0
                or self.action_timeout <= 0.0
                or self.start_delay < 0.0
                or self.feedback_timeout <= 0.0
                or self.tf_feedback_timeout <= 0.0
                or self.goal_position_tolerance <= 0.0
                or self.goal_depth_tolerance <= 0.0
                or self.goal_yaw_tolerance <= 0.0
                or self.acceptance_depth_tolerance <= 0.0
                or self.stable_vertical_speed_threshold <= 0.0
                or self.required_stable_seconds <= 0.0
                or self.vertical_speed_window_seconds <= 0.0
                or self.minimum_speed_window_seconds <= 0.0
                or self.minimum_speed_window_seconds
                > self.vertical_speed_window_seconds):
            raise ValueError('次数、频率、超时、容差和稳定窗口配置无效')

        self.tf_listener = tf.TransformListener()
        self.latest_state = None
        self.latest_state_received_at = None
        self.start_pose = None
        self.start_yaw = None
        self.baseline_depth = None
        self.steps = ()
        self.speed_samples = deque()
        self.last_tf_stamp = None
        self.last_tf_pose = None
        self.summary_file = None
        self.summary_writer = None
        self.summary_path = ''
        self.completed = False
        self.aborted = False

        self._assert_no_other_goal_publisher(before_start=True)
        self.goal_pub = rospy.Publisher(
            '/cmd/motion/goal', PoseStamped, queue_size=1)
        rospy.Subscriber(
            '/motion/state',
            MotionState,
            self.motion_state_callback,
            queue_size=10,
        )
        self._open_summary()
        rospy.on_shutdown(self._on_shutdown)

    def motion_state_callback(self, message):
        self.latest_state = copy.deepcopy(message)
        self.latest_state_received_at = rospy.Time.now()

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
        """防止任务节点或其他自动测试同时改写深度目标。"""
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
                '启动深度自动测试前 /cmd/motion/goal 必须没有任何发布者')

    def _open_summary(self):
        if not os.path.isdir(self.log_directory):
            os.makedirs(self.log_directory)
        filename = 'motion_depth_auto_test_{0}.csv'.format(
            datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
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

    def _state_fresh(self):
        if self.latest_state_received_at is None:
            return False
        return (
            rospy.Time.now() - self.latest_state_received_at
        ).to_sec() <= self.feedback_timeout

    def _lookup_pose(self, require_fresh=True):
        try:
            stamp = self.tf_listener.getLatestCommonTime(
                'map', 'base_link')
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', stamp)
        except tf.Exception:
            return None
        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(value) for value in values):
            return None
        if require_fresh:
            age = (rospy.Time.now() - stamp).to_sec()
            if age > self.tf_feedback_timeout:
                return None
        self.last_tf_pose = (tuple(translation), tuple(rotation), stamp)
        return self.last_tf_pose

    def _wait_for_supervisor(self):
        deadline = rospy.Time.now() + rospy.Duration(self.startup_timeout)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            state = self.latest_state
            if (
                    self._state_fresh()
                    and state is not None
                    and state.startup_complete
                    and state.state == MotionState.HOVER
                    and self._lookup_pose() is not None):
                return True
            rospy.loginfo_throttle(
                2.0,
                '%s: 等待 motion_supervisor 启动完成、HOVER 和新鲜 TF',
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
        except tf.Exception as error:
            raise RuntimeError(
                '等待初始 map -> base_link TF 失败: {}'.format(error))
        pose = self._lookup_pose()
        if pose is None:
            raise RuntimeError('无法获取新鲜的初始 map -> base_link TF')
        translation, rotation, unused_stamp = pose
        del unused_stamp
        self.start_pose = translation
        self.start_yaw = euler_from_quaternion(rotation)[2]
        self.baseline_depth = translation[2]
        self.steps = build_depth_sequence(
            self.target_depths,
            self.baseline_depth,
            self.cycle_count,
        )

    def _make_goal(self, target_depth):
        message = PoseStamped()
        message.header.frame_id = 'map'
        message.pose.position.x = self.start_pose[0]
        message.pose.position.y = self.start_pose[1]
        message.pose.position.z = float(target_depth)
        quaternion = quaternion_from_euler(0.0, 0.0, self.start_yaw)
        message.pose.orientation.x = quaternion[0]
        message.pose.orientation.y = quaternion[1]
        message.pose.orientation.z = quaternion[2]
        message.pose.orientation.w = quaternion[3]
        return message

    def _state_matches_goal(self, target_depth):
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
            self.start_pose[0],
            self.start_pose[1],
            float(target_depth),
            self.start_yaw,
            self.goal_position_tolerance,
            self.goal_depth_tolerance,
            self.goal_yaw_tolerance,
        )

    def _sample_tf_depth(self):
        pose = self._lookup_pose()
        if pose is None:
            return None, None
        translation, unused_rotation, stamp = pose
        del unused_rotation
        stamp_seconds = stamp.to_sec()
        if (
                self.last_tf_stamp is None
                or stamp_seconds > self.last_tf_stamp):
            self.speed_samples.append((stamp_seconds, translation[2]))
            self.last_tf_stamp = stamp_seconds
            cutoff = stamp_seconds - self.vertical_speed_window_seconds
            while (
                    len(self.speed_samples) > 2
                    and self.speed_samples[0][0] < cutoff):
                self.speed_samples.popleft()
        vertical_speed = None
        if (
                len(self.speed_samples) >= 2
                and self.speed_samples[-1][0] - self.speed_samples[0][0]
                >= self.minimum_speed_window_seconds):
            vertical_speed = linear_vertical_speed(self.speed_samples)
        return translation[2], vertical_speed

    @staticmethod
    def _new_peaks(start_depth):
        return {
            'start_depth': start_depth,
            'minimum_depth': start_depth,
            'maximum_depth': start_depth,
            'peak_abs_depth_error': 0.0,
            'peak_abs_tf_vertical_speed': 0.0,
            'ever_crossed_target': False,
            'base_position_error': 0.0,
            'horizontal_speed': 0.0,
            'yaw_error': 0.0,
            'yaw_rate': 0.0,
            'tx': 0,
            'ty': 0,
            'mz': 0,
        }

    def _update_peaks(self, peaks, target_depth, depth, vertical_speed):
        if depth is not None:
            peaks['minimum_depth'] = min(peaks['minimum_depth'], depth)
            peaks['maximum_depth'] = max(peaks['maximum_depth'], depth)
            peaks['peak_abs_depth_error'] = max(
                peaks['peak_abs_depth_error'],
                abs(depth - target_depth),
            )
            if target_depth < peaks['start_depth']:
                peaks['ever_crossed_target'] = (
                    peaks['ever_crossed_target'] or depth <= target_depth)
            elif target_depth > peaks['start_depth']:
                peaks['ever_crossed_target'] = (
                    peaks['ever_crossed_target'] or depth >= target_depth)
        if vertical_speed is not None:
            peaks['peak_abs_tf_vertical_speed'] = max(
                peaks['peak_abs_tf_vertical_speed'],
                abs(vertical_speed),
            )
        if not self._state_fresh() or self.latest_state is None:
            return
        state = self.latest_state
        peaks['base_position_error'] = max(
            peaks['base_position_error'], abs(state.base_position_error))
        peaks['horizontal_speed'] = max(
            peaks['horizontal_speed'], abs(state.horizontal_speed))
        peaks['yaw_error'] = max(
            peaks['yaw_error'], abs(state.yaw_error))
        peaks['yaw_rate'] = max(
            peaks['yaw_rate'], abs(state.yaw_rate))
        peaks['tx'] = max(peaks['tx'], abs(state.tx))
        peaks['ty'] = max(peaks['ty'], abs(state.ty))
        peaks['mz'] = max(peaks['mz'], abs(state.mz))

    def _write_result(
            self, step, started_at, finished_at, peaks,
            final_depth, final_vertical_speed, result, reason):
        state = self.latest_state
        overshoot = directed_depth_overshoot(
            peaks['start_depth'],
            step.target_depth,
            peaks['minimum_depth'],
            peaks['maximum_depth'],
        )
        self.summary_writer.writerow({
            'step': step.index,
            'cycle': step.cycle,
            'phase': step.phase,
            'requested_depth': step.requested_depth,
            'target_depth': step.target_depth,
            'baseline_depth': self.baseline_depth,
            'target_x': self.start_pose[0],
            'target_y': self.start_pose[1],
            'target_yaw_deg': math.degrees(self.start_yaw),
            'started_at': started_at.to_sec(),
            'finished_at': finished_at.to_sec(),
            'duration_s': (finished_at - started_at).to_sec(),
            'acceptance_depth_tolerance': (
                self.acceptance_depth_tolerance),
            'stable_vertical_speed_threshold': (
                self.stable_vertical_speed_threshold),
            'required_stable_seconds': self.required_stable_seconds,
            'vertical_speed_window_seconds': (
                self.vertical_speed_window_seconds),
            'start_depth': peaks['start_depth'],
            'minimum_depth': peaks['minimum_depth'],
            'maximum_depth': peaks['maximum_depth'],
            'peak_abs_depth_error': peaks['peak_abs_depth_error'],
            'peak_abs_tf_vertical_speed': (
                peaks['peak_abs_tf_vertical_speed']),
            'maximum_target_overshoot': overshoot,
            'ever_crossed_target': int(peaks['ever_crossed_target']),
            'peak_base_position_error': peaks['base_position_error'],
            'peak_horizontal_speed': peaks['horizontal_speed'],
            'peak_yaw_error_deg': math.degrees(peaks['yaw_error']),
            'peak_yaw_rate_deg_s': math.degrees(peaks['yaw_rate']),
            'peak_abs_tx': peaks['tx'],
            'peak_abs_ty': peaks['ty'],
            'peak_abs_mz': peaks['mz'],
            'final_depth': '' if final_depth is None else final_depth,
            'final_depth_error': (
                ''
                if final_depth is None
                else final_depth - step.target_depth
            ),
            'final_tf_vertical_speed': (
                ''
                if final_vertical_speed is None
                else final_vertical_speed
            ),
            'final_base_position_error': (
                '' if state is None else state.base_position_error),
            'final_horizontal_speed': (
                '' if state is None else state.horizontal_speed),
            'final_yaw_error_deg': (
                '' if state is None else math.degrees(state.yaw_error)),
            'result': result,
            'reason': reason,
        })
        self.summary_file.flush()

    def _execute_step(self, step):
        initial_depth, unused_speed = self._sample_tf_depth()
        del unused_speed
        if initial_depth is None:
            raise RuntimeError('步骤开始前没有新鲜 TF 深度')
        self.speed_samples.clear()
        self.last_tf_stamp = None
        goal = self._make_goal(step.target_depth)
        started_at = rospy.Time.now()
        deadline = started_at + rospy.Duration(self.action_timeout)
        stable_started_at = None
        final_depth = initial_depth
        final_vertical_speed = None
        peaks = self._new_peaks(initial_depth)
        rate = rospy.Rate(self.publish_rate_hz)
        rospy.loginfo(
            '%s: [%d/%d] 循环=%d，动作=%s，绝对目标深度=%.3f m',
            NODE_NAME,
            step.index,
            len(self.steps),
            step.cycle,
            step.phase,
            step.target_depth,
        )

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            self._assert_no_other_goal_publisher()
            goal.header.stamp = now
            self.goal_pub.publish(goal)
            depth, vertical_speed = self._sample_tf_depth()
            if depth is not None:
                final_depth = depth
            if vertical_speed is not None:
                final_vertical_speed = vertical_speed
            self._update_peaks(
                peaks, step.target_depth, depth, vertical_speed)

            goal_confirmed = self._state_matches_goal(step.target_depth)
            arrived = (
                goal_confirmed
                and depth is not None
                and vertical_speed is not None
                and depth_motion_is_stable(
                    depth,
                    step.target_depth,
                    vertical_speed,
                    self.acceptance_depth_tolerance,
                    self.stable_vertical_speed_threshold,
                )
            )
            if arrived:
                if stable_started_at is None:
                    stable_started_at = now
                stable_duration = (now - stable_started_at).to_sec()
                if stable_duration >= self.required_stable_seconds:
                    reason = (
                        'TF 深度误差和垂向速度连续稳定 {:.1f} s'.format(
                            self.required_stable_seconds))
                    self._write_result(
                        step,
                        started_at,
                        now,
                        peaks,
                        final_depth,
                        final_vertical_speed,
                        'PASS',
                        reason,
                    )
                    return True
            else:
                stable_started_at = None

            rospy.loginfo_throttle(
                2.0,
                '%s: 等待深度稳定；目标=%.3f m，TF=%.3f m，'
                '误差=%s m，TF速度=%s m/s，目标确认=%s',
                NODE_NAME,
                step.target_depth,
                float('nan') if depth is None else depth,
                (
                    '无'
                    if depth is None
                    else '{:+.3f}'.format(depth - step.target_depth)
                ),
                (
                    '窗口不足'
                    if vertical_speed is None
                    else '{:+.4f}'.format(vertical_speed)
                ),
                goal_confirmed,
            )

            if now >= deadline:
                reason = (
                    '动作超时；最终深度={} m，TF垂向速度={} m/s，'
                    'motion_supervisor目标确认={}'.format(
                        (
                            '无新鲜反馈'
                            if final_depth is None
                            else '{:.3f}'.format(final_depth)
                        ),
                        (
                            '窗口不足'
                            if final_vertical_speed is None
                            else '{:.4f}'.format(final_vertical_speed)
                        ),
                        goal_confirmed,
                    ))
                self._write_result(
                    step,
                    started_at,
                    now,
                    peaks,
                    final_depth,
                    final_vertical_speed,
                    'FAIL',
                    reason,
                )
                rospy.logerr('%s: %s', NODE_NAME, reason)
                return False
            rate.sleep()
        return False

    def _publish_current_pose_hold(self):
        """异常退出时经 motion_supervisor 锁存当前 TF 位姿和深度。"""
        pose = self._lookup_pose(require_fresh=False)
        if pose is None:
            return
        translation, rotation, unused_stamp = pose
        del unused_stamp
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = translation[0]
        goal.pose.position.y = translation[1]
        goal.pose.position.z = translation[2]
        goal.pose.orientation.x = rotation[0]
        goal.pose.orientation.y = rotation[1]
        goal.pose.orientation.z = rotation[2]
        goal.pose.orientation.w = rotation[3]
        self.goal_pub.publish(goal)
        rospy.logwarn(
            '%s: 异常退出，已向 motion_supervisor 发布当前 TF 位姿保持目标',
            NODE_NAME,
        )

    def _on_shutdown(self):
        if not self.completed and not self.aborted:
            self.aborted = True
            try:
                self._publish_current_pose_hold()
            except rospy.ROSException:
                pass
        self._close_summary()

    def run(self):
        if not self._wait_for_supervisor():
            raise RuntimeError(
                '等待 motion_supervisor startup_complete + HOVER + TF 超时')
        self._lock_start_pose()
        rospy.loginfo(
            '%s: 锁定水平位置=(%.3f, %.3f)、航向=%.1f deg、'
            '基准深度=%.3f m；绝对目标=%s；共 %d 步；'
            '到达判据=|深度误差|≤%.3f m 且 |TF垂向速度|≤%.4f m/s '
            '连续 %.1f s；摘要=%s',
            NODE_NAME,
            self.start_pose[0],
            self.start_pose[1],
            math.degrees(self.start_yaw),
            self.baseline_depth,
            list(self.target_depths),
            len(self.steps),
            self.acceptance_depth_tolerance,
            self.stable_vertical_speed_threshold,
            self.required_stable_seconds,
            self.summary_path,
        )
        rospy.sleep(self.start_delay)
        for step in self.steps:
            if not self._execute_step(step):
                self.aborted = True
                self._publish_current_pose_hold()
                raise RuntimeError(
                    '深度自动测试在第 {} 步中止'.format(step.index))
        self.completed = True
        self._close_summary()
        rospy.loginfo(
            '%s: 全部 %d 个深度动作完成，已返回启动基准深度 %.3f m',
            NODE_NAME,
            len(self.steps),
            self.baseline_depth,
        )


def main():
    rospy.init_node(NODE_NAME)
    try:
        MotionDepthAutoTest().run()
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

