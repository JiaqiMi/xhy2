#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_yaw_continuous_test.py
功能：通过 motion_supervisor 执行锁定 base_link 位置的连续多圈航向测试

运行逻辑：
    1. 锁定起始 base_link 位置和航向；
    2. 运行阶段持续发布“当前航向 + 可配置前视偏置”的目标；
    3. 累计航向变化达到指定圈数后，发布最终航向并等待控制器定点接管；
    4. 全程只发布 /cmd/motion/goal，不直接下发 TX/TY/MZ。
"""

from __future__ import division

import copy
import csv
import json
import math
import os
from datetime import datetime

import rosgraph
import rospy
import tf
from auv_control.msg import MotionState
from geometry_msgs.msg import PoseStamped
from rosgraph.masterapi import ROSMasterException
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = 'motion_yaw_continuous_test'


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi)。"""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class MotionYawContinuousTest(object):
    """以滚动前视航向目标驱动控制器完成连续多圈旋转。"""

    LOG_FIELDS = (
        'ros_time', 'elapsed_s', 'phase', 'result', 'reason',
        'current_x', 'current_y', 'current_yaw_deg',
        'target_x', 'target_y', 'target_z', 'target_yaw_deg',
        'accumulated_yaw_deg', 'completed_turns', 'remaining_turns',
        'base_drift_m', 'motion_state', 'position_error_m',
        'yaw_error_deg', 'raw_yaw_rate_deg_s',
        'map_yaw_rate_deg_s', 'horizontal_speed_mps',
        'tx', 'ty', 'mz', 'x_axis_state', 'y_axis_state',
        'yaw_axis_state',
    )

    def __init__(self):
        self.target_z = float(rospy.get_param('~target_z', -0.9))
        self.turns = int(rospy.get_param('~turns', 3))
        self.lookahead_offset = math.radians(float(
            rospy.get_param('~lookahead_yaw_offset_deg', 90.0)))
        self.publish_rate_hz = float(
            rospy.get_param('~publish_rate_hz', 10.0))
        self.startup_timeout = float(rospy.get_param('~startup_timeout', 60.0))
        self.rotation_timeout = float(
            rospy.get_param('~rotation_timeout', 600.0))
        self.final_hold_seconds = float(
            rospy.get_param('~final_hold_seconds', 2.0))
        self.feedback_timeout = float(
            rospy.get_param('~feedback_timeout', 0.5))
        self.position_tolerance = float(
            rospy.get_param('~position_tolerance', 0.10))
        self.horizontal_speed_threshold = float(
            rospy.get_param('~horizontal_speed_threshold', 0.03))
        self.yaw_tolerance = math.radians(float(
            rospy.get_param('~yaw_tolerance_deg', 5.0)))
        self.yaw_rate_threshold = math.radians(float(
            rospy.get_param('~yaw_rate_threshold_deg_s', 1.0)))
        self.log_directory = os.path.abspath(os.path.expanduser(str(
            rospy.get_param(
                '~log_directory',
                '~/.ros/auv_logs/motion_yaw_continuous_test'))))

        numeric = (
            self.target_z, self.lookahead_offset, self.publish_rate_hz,
            self.startup_timeout, self.rotation_timeout,
            self.final_hold_seconds, self.feedback_timeout,
            self.position_tolerance, self.horizontal_speed_threshold,
            self.yaw_tolerance, self.yaw_rate_threshold,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('连续航向测试参数必须为有限值')
        if self.turns <= 0:
            raise ValueError('turns 必须为正整数')
        if abs(self.lookahead_offset) < math.radians(1.0):
            raise ValueError('lookahead_yaw_offset_deg 绝对值必须不小于 1 deg')
        if any(value <= 0.0 for value in (
                self.publish_rate_hz, self.startup_timeout,
                self.rotation_timeout, self.feedback_timeout,
                self.position_tolerance, self.horizontal_speed_threshold,
                self.yaw_tolerance, self.yaw_rate_threshold)):
            raise ValueError('连续航向测试的频率、超时和阈值必须为正数')
        if self.final_hold_seconds < 0.0:
            raise ValueError('final_hold_seconds 不能为负数')

        self.direction = 1.0 if self.lookahead_offset > 0.0 else -1.0
        self.lookahead_offset = abs(self.lookahead_offset)
        self.total_rotation = 2.0 * math.pi * self.turns
        self.tf_listener = tf.TransformListener()
        self.latest_state = None
        self.latest_state_received_at = None
        self.latest_diagnostics = None
        self.latest_diagnostics_received_at = None
        self.start_pose = None
        self.start_yaw = None
        self.last_yaw = None
        self.accumulated_yaw = 0.0
        self.log_file = None
        self.log_writer = None
        self.log_path = ''
        self.summary_path = ''
        self.peak_base_drift = 0.0
        self.peak_abs_yaw_error = 0.0
        self.peak_abs_map_yaw_rate = 0.0
        self.peak_abs_mz = 0
        self.completed = False

        self._assert_no_other_goal_publisher(before_start=True)
        self.goal_pub = rospy.Publisher(
            '/cmd/motion/goal', PoseStamped, queue_size=1)
        self.cancel_pub = rospy.Publisher(
            '/cmd/motion/cancel', Empty, queue_size=1)
        rospy.Subscriber(
            '/motion/state', MotionState, self.motion_state_callback,
            queue_size=20)
        rospy.Subscriber(
            '/motion/diagnostics', String, self.diagnostics_callback,
            queue_size=50)
        self._open_log()
        rospy.on_shutdown(self._on_shutdown)

    def _open_log(self):
        if not os.path.isdir(self.log_directory):
            os.makedirs(self.log_directory)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.log_path = os.path.join(
            self.log_directory, 'motion_yaw_continuous_{}.csv'.format(stamp))
        self.summary_path = os.path.join(
            self.log_directory, 'motion_yaw_continuous_{}.json'.format(stamp))
        self.log_file = open(self.log_path, 'w', encoding='utf-8', newline='')
        self.log_writer = csv.DictWriter(
            self.log_file, fieldnames=self.LOG_FIELDS)
        self.log_writer.writeheader()
        self.log_file.flush()

    def _close_log(self):
        if self.log_file is not None:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None
            self.log_writer = None

    def _write_summary(self, result, reason, elapsed):
        payload = {
            'result': result,
            'reason': reason,
            'elapsed_s': elapsed,
            'turns': self.turns,
            'direction': 'positive' if self.direction > 0.0 else 'negative',
            'lookahead_yaw_offset_deg': math.degrees(
                self.direction * self.lookahead_offset),
            'accumulated_yaw_deg': math.degrees(self.accumulated_yaw),
            'completed_turns': self.accumulated_yaw / (2.0 * math.pi),
            'peak_base_drift_m': self.peak_base_drift,
            'peak_abs_yaw_error_deg': math.degrees(self.peak_abs_yaw_error),
            'peak_abs_map_yaw_rate_deg_s': math.degrees(
                self.peak_abs_map_yaw_rate),
            'peak_abs_mz': self.peak_abs_mz,
            'sample_log': self.log_path,
        }
        with open(self.summary_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False,
                      indent=2, sort_keys=True)

    def motion_state_callback(self, message):
        self.latest_state = copy.deepcopy(message)
        self.latest_state_received_at = rospy.Time.now()

    def diagnostics_callback(self, message):
        try:
            diagnostics = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if not isinstance(diagnostics, dict):
            return
        self.latest_diagnostics = diagnostics
        self.latest_diagnostics_received_at = rospy.Time.now()

    def _state_fresh(self):
        return (
            self.latest_state_received_at is not None
            and (rospy.Time.now() - self.latest_state_received_at).to_sec()
            <= self.feedback_timeout
        )

    def _lookup_pose(self):
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0))
        except tf.Exception as error:
            raise RuntimeError('无法获取 map -> base_link TF: {}'.format(error))
        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError('map -> base_link TF 含有非有限值')
        return (
            float(translation[0]), float(translation[1]), float(translation[2]),
            euler_from_quaternion(rotation)[2],
        )

    def _wait_for_supervisor(self):
        deadline = rospy.Time.now() + rospy.Duration(self.startup_timeout)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            state = self.latest_state
            if (
                    self._state_fresh() and state is not None
                    and state.startup_complete
                    and state.state == MotionState.HOVER):
                return
            rate.sleep()
        raise RuntimeError('等待 motion_supervisor startup_complete + HOVER 超时')

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
        own_name = rospy.get_name()
        publishers = self._publishers_on_goal_topic()
        others = [name for name in publishers if name != own_name]
        if others:
            raise RuntimeError(
                '/cmd/motion/goal 存在其他发布者: {}'.format(
                    ', '.join(others)))
        if before_start and publishers:
            raise RuntimeError(
                '启动连续航向测试前 /cmd/motion/goal 必须没有任何发布者')

    def _goal_message(self, target_yaw, stamp):
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = 'map'
        message.pose.position.x = self.start_pose[0]
        message.pose.position.y = self.start_pose[1]
        message.pose.position.z = self.target_z
        quaternion = quaternion_from_euler(0.0, 0.0, wrap_angle(target_yaw))
        message.pose.orientation.x = quaternion[0]
        message.pose.orientation.y = quaternion[1]
        message.pose.orientation.z = quaternion[2]
        message.pose.orientation.w = quaternion[3]
        return message

    def _update_progress(self, yaw):
        if self.last_yaw is None:
            self.last_yaw = yaw
            return
        self.accumulated_yaw += self.direction * wrap_angle(yaw - self.last_yaw)
        self.last_yaw = yaw

    def _write_sample(self, now, started_at, phase, target_yaw,
                      pose=None, result='', reason=''):
        if pose is None:
            pose = self._lookup_pose()
        current_x, current_y, unused_z, current_yaw = pose
        del unused_z
        self._update_progress(current_yaw)
        base_drift = math.hypot(
            current_x - self.start_pose[0], current_y - self.start_pose[1])
        state = self.latest_state
        diagnostics = self.latest_diagnostics or {}
        try:
            map_yaw_rate = float(diagnostics.get('map_yaw_rate', 0.0))
        except (TypeError, ValueError):
            map_yaw_rate = 0.0
        self.peak_base_drift = max(self.peak_base_drift, base_drift)
        self.peak_abs_map_yaw_rate = max(
            self.peak_abs_map_yaw_rate, abs(map_yaw_rate))
        if state is not None:
            self.peak_abs_yaw_error = max(
                self.peak_abs_yaw_error, abs(state.yaw_error))
            self.peak_abs_mz = max(self.peak_abs_mz, abs(state.mz))
        self.log_writer.writerow({
            'ros_time': now.to_sec(),
            'elapsed_s': (now - started_at).to_sec(),
            'phase': phase,
            'result': result,
            'reason': reason,
            'current_x': current_x,
            'current_y': current_y,
            'current_yaw_deg': math.degrees(current_yaw),
            'target_x': self.start_pose[0],
            'target_y': self.start_pose[1],
            'target_z': self.target_z,
            'target_yaw_deg': math.degrees(wrap_angle(target_yaw)),
            'accumulated_yaw_deg': math.degrees(self.accumulated_yaw),
            'completed_turns': self.accumulated_yaw / (2.0 * math.pi),
            'remaining_turns': max(
                0.0, self.turns - self.accumulated_yaw / (2.0 * math.pi)),
            'base_drift_m': base_drift,
            'motion_state': '' if state is None else state.state,
            'position_error_m': '' if state is None else state.base_position_error,
            'yaw_error_deg': '' if state is None else math.degrees(
                state.yaw_error),
            'raw_yaw_rate_deg_s': '' if state is None else math.degrees(
                state.yaw_rate),
            'map_yaw_rate_deg_s': math.degrees(map_yaw_rate),
            'horizontal_speed_mps': '' if state is None else state.horizontal_speed,
            'tx': '' if state is None else state.tx,
            'ty': '' if state is None else state.ty,
            'mz': '' if state is None else state.mz,
            'x_axis_state': '' if state is None else state.x_axis_state,
            'y_axis_state': '' if state is None else state.y_axis_state,
            'yaw_axis_state': '' if state is None else state.yaw_axis_state,
        })
        self.log_file.flush()
        return current_yaw, state

    def _final_target_reached(self, state):
        return (
            self._state_fresh() and state is not None
            and state.startup_complete
            and state.state == MotionState.HOVER
            and abs(state.base_position_error) <= self.position_tolerance
            and abs(state.horizontal_speed) <= self.horizontal_speed_threshold
            and abs(state.yaw_error) <= self.yaw_tolerance
            and abs(state.yaw_rate) <= self.yaw_rate_threshold
        )

    def run(self):
        self._wait_for_supervisor()
        self.start_pose = self._lookup_pose()
        self.start_yaw = self.start_pose[3]
        self.last_yaw = self.start_yaw
        started_at = rospy.Time.now()
        deadline = started_at + rospy.Duration(self.rotation_timeout)
        final_hold_started_at = None
        final_target_yaw = self.start_yaw
        phase = 'ROLLING'
        rate = rospy.Rate(self.publish_rate_hz)
        rospy.loginfo(
            '%s: 开始连续旋转 %d 圈，滚动前视偏置=%+.1f deg，'
            '全程锁定 base_link=(%.3f, %.3f)',
            NODE_NAME,
            self.turns,
            math.degrees(self.direction * self.lookahead_offset),
            self.start_pose[0],
            self.start_pose[1])
        try:
            while not rospy.is_shutdown():
                now = rospy.Time.now()
                self._assert_no_other_goal_publisher()
                pose = self._lookup_pose()
                current_yaw = pose[3]
                self._update_progress(current_yaw)
                if phase == 'ROLLING' and self.accumulated_yaw >= self.total_rotation:
                    phase = 'FINAL'
                    rospy.loginfo(
                        '%s: 已累计 %.2f 圈，切换最终航向定点并等待刹停',
                        NODE_NAME, self.accumulated_yaw / (2.0 * math.pi))
                target_yaw = (
                    wrap_angle(current_yaw + self.direction * self.lookahead_offset)
                    if phase == 'ROLLING' else final_target_yaw)
                goal = self._goal_message(target_yaw, now)
                self.goal_pub.publish(goal)
                unused_yaw, state = self._write_sample(
                    now, started_at, phase, target_yaw, pose=pose)
                del unused_yaw
                if phase == 'FINAL' and self._final_target_reached(state):
                    if final_hold_started_at is None:
                        final_hold_started_at = now
                    elif (now - final_hold_started_at).to_sec() >= (
                            self.final_hold_seconds):
                        reason = '完成 {} 圈连续旋转并持续稳定 {:.1f} s'.format(
                            self.turns, self.final_hold_seconds)
                        self._write_sample(
                            now, started_at, phase, target_yaw, pose=pose,
                            result='PASS', reason=reason)
                        self._write_summary(
                            'PASS', reason, (now - started_at).to_sec())
                        self.completed = True
                        rospy.loginfo('%s: %s', NODE_NAME, reason)
                        return
                else:
                    final_hold_started_at = None
                if now >= deadline:
                    raise RuntimeError(
                        '连续旋转超时：已完成 {:.2f}/{:.2f} 圈，phase={}'.format(
                            self.accumulated_yaw / (2.0 * math.pi),
                            self.turns, phase))
                rate.sleep()
        except Exception as error:
            try:
                self._write_sample(
                    rospy.Time.now(), started_at, phase, final_target_yaw,
                    result='FAIL', reason=str(error))
                self._write_summary(
                    'FAIL', str(error),
                    (rospy.Time.now() - started_at).to_sec())
            except Exception:
                pass
            raise

    def _on_shutdown(self):
        if not self.completed:
            try:
                self.cancel_pub.publish(Empty())
            except rospy.ROSException:
                pass
        self._close_log()


def main():
    rospy.init_node(NODE_NAME)
    test = None
    try:
        test = MotionYawContinuousTest()
        test.run()
    except (ValueError, RuntimeError, OSError, IOError, ROSMasterException) as error:
        rospy.logfatal('%s: %s', NODE_NAME, error)
        if test is not None:
            test._close_log()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
