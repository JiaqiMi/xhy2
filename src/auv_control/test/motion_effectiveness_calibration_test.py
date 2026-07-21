#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通过单轴闭环位姿激励采集 TX/TY/MZ 有效性矩阵标定数据。"""

from __future__ import division

import copy
import csv
import json
import math
import os
import sys
from datetime import datetime

import rosgraph
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from rosgraph.masterapi import ROSMasterException
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from auv_control.msg import MotionState


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from motion_auto_sequence_core import (  # noqa: E402
    build_axis_sequence,
    goal_matches,
    relative_goal,
)


NODE_NAME = 'motion_effectiveness_calibration_test'


class MotionEffectivenessCalibrationTest(object):
    """安全地通过 motion_supervisor 的单轴目标收集闭环辨识数据。"""

    LOG_FIELDS = (
        'ros_time', 'elapsed_s', 'step', 'axis', 'phase', 'offset',
        'target_x', 'target_y', 'target_z', 'target_yaw_deg',
        'current_x', 'current_y', 'current_z', 'current_yaw_deg',
        'body_x_m', 'body_y_m', 'body_yaw_deg',
        'state', 'position_error_m', 'yaw_error_deg',
        'horizontal_speed_mps', 'yaw_rate_deg_s',
        'tx', 'ty', 'mz', 'xy_braking', 'yaw_braking',
    )

    def __init__(self):
        self.repetitions = int(rospy.get_param('~repetitions', 2))
        self.translation_magnitude = float(
            rospy.get_param('~translation_magnitude', 0.4))
        self.yaw_magnitude = math.radians(float(
            rospy.get_param('~yaw_magnitude_deg', 30.0)))
        self.target_z = float(rospy.get_param('~target_z', -0.9))
        self.publish_rate_hz = float(
            rospy.get_param('~publish_rate_hz', 10.0))
        self.startup_timeout = float(
            rospy.get_param('~startup_timeout', 60.0))
        self.action_timeout = float(
            rospy.get_param('~action_timeout', 90.0))
        self.settle_seconds = float(rospy.get_param('~settle_seconds', 8.0))
        self.feedback_timeout = float(
            rospy.get_param('~feedback_timeout', 0.5))
        self.goal_position_tolerance = float(
            rospy.get_param('~goal_position_tolerance', 0.05))
        self.goal_depth_tolerance = float(
            rospy.get_param('~goal_depth_tolerance', 0.05))
        self.goal_yaw_tolerance = math.radians(float(
            rospy.get_param('~goal_yaw_tolerance_deg', 2.0)))
        self.log_directory = os.path.abspath(os.path.expanduser(str(
            rospy.get_param(
                '~log_directory',
                '~/.ros/auv_logs/motion_effectiveness_calibration'))))

        numeric = (
            self.translation_magnitude, self.yaw_magnitude, self.target_z,
            self.publish_rate_hz, self.startup_timeout, self.action_timeout,
            self.settle_seconds, self.feedback_timeout,
            self.goal_position_tolerance, self.goal_depth_tolerance,
            self.goal_yaw_tolerance,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError('标定测试参数必须为有限值')
        if self.repetitions <= 0:
            raise ValueError('repetitions 必须大于 0')
        if self.translation_magnitude <= 0.0 or self.yaw_magnitude <= 0.0:
            raise ValueError('激励幅值必须大于 0')
        if any(value <= 0.0 for value in (
                self.publish_rate_hz, self.startup_timeout,
                self.action_timeout, self.feedback_timeout)):
            raise ValueError('频率和超时必须大于 0')
        if self.settle_seconds < 0.0:
            raise ValueError('settle_seconds 不能小于 0')

        self.steps = []
        for axis, magnitude in (
                ('x', self.translation_magnitude),
                ('y', self.translation_magnitude),
                ('yaw', self.yaw_magnitude)):
            for step in build_axis_sequence(
                    axis, (magnitude,), self.repetitions):
                self.steps.append(step._replace(index=len(self.steps) + 1))

        self.tf_listener = tf.TransformListener()
        self.latest_state = None
        self.latest_state_received_at = None
        self.latest_diagnostics = None
        self.start_pose = None
        self.start_yaw = None
        self.active_step = None
        self.trace_file = None
        self.trace_writer = None
        self.trace_path = ''
        self.started_at = None
        self.completed = False
        self.aborted = False

        self._assert_no_other_goal_publisher(before_start=True)
        self.goal_pub = rospy.Publisher(
            '/cmd/motion/goal', PoseStamped, queue_size=1)
        self.cancel_pub = rospy.Publisher(
            '/cmd/motion/cancel', Empty, queue_size=1)
        rospy.Subscriber(
            '/motion/state', MotionState, self._state_callback, queue_size=20)
        rospy.Subscriber(
            '/motion/diagnostics', String, self._diagnostics_callback,
            queue_size=50)
        self._open_trace()
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
        own_name = rospy.get_name()
        publishers = self._publishers_on_goal_topic()
        others = [name for name in publishers if name != own_name]
        if others:
            raise RuntimeError(
                '/cmd/motion/goal 存在其他发布者: {}'.format(
                    ', '.join(others)))
        if before_start and publishers:
            raise RuntimeError(
                '启动标定前 /cmd/motion/goal 不能存在任何发布者')

    def _state_callback(self, message):
        self.latest_state = copy.deepcopy(message)
        self.latest_state_received_at = rospy.Time.now()

    def _diagnostics_callback(self, message):
        try:
            diagnostics = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if isinstance(diagnostics, dict):
            self.latest_diagnostics = diagnostics

    def _open_trace(self):
        if not os.path.isdir(self.log_directory):
            os.makedirs(self.log_directory)
        filename = 'motion_effectiveness_calibration_{0}.csv'.format(
            datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        self.trace_path = os.path.join(self.log_directory, filename)
        self.trace_file = open(
            self.trace_path, 'w', encoding='utf-8', newline='')
        self.trace_writer = csv.DictWriter(
            self.trace_file, fieldnames=self.LOG_FIELDS)
        self.trace_writer.writeheader()
        self.trace_file.flush()

    def _close_trace(self):
        if self.trace_file is not None:
            self.trace_file.flush()
            self.trace_file.close()
            self.trace_file = None
            self.trace_writer = None

    def _state_fresh(self):
        if self.latest_state_received_at is None:
            return False
        return (rospy.Time.now() - self.latest_state_received_at).to_sec() <= (
            self.feedback_timeout)

    def _wait_for_hover(self):
        deadline = rospy.Time.now() + rospy.Duration(self.startup_timeout)
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            state = self.latest_state
            if (self._state_fresh() and state is not None
                    and state.startup_complete
                    and state.state == MotionState.HOVER):
                return
            rate.sleep()
        raise RuntimeError('等待 motion_supervisor 进入 HOVER 超时')

    def _lock_start_pose(self):
        try:
            self.tf_listener.waitForTransform(
                'map', 'base_link', rospy.Time(0),
                rospy.Duration(self.startup_timeout))
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0))
        except tf.Exception as error:
            raise RuntimeError('无法获取初始 map -> base_link TF: {}'.format(error))
        if not all(math.isfinite(value) for value in (
                tuple(translation) + tuple(rotation))):
            raise RuntimeError('初始 TF 含有非有限值')
        self.start_pose = tuple(translation)
        self.start_yaw = euler_from_quaternion(rotation)[2]

    def _make_goal(self, step):
        target = relative_goal(
            self.start_pose[0], self.start_pose[1], self.start_yaw,
            self.target_z, step.axis, step.offset)
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

    def _goal_matched(self, target):
        state = self.latest_state
        if not self._state_fresh() or state is None:
            return False
        goal = state.goal.pose
        yaw = euler_from_quaternion((
            goal.orientation.x, goal.orientation.y,
            goal.orientation.z, goal.orientation.w))[2]
        return goal_matches(
            goal.position.x, goal.position.y, goal.position.z, yaw,
            target[0], target[1], target[2], target[3],
            self.goal_position_tolerance, self.goal_depth_tolerance,
            self.goal_yaw_tolerance)

    def _trace(self, target):
        if self.trace_writer is None or self.start_pose is None:
            return
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0))
        except tf.Exception:
            return
        yaw = euler_from_quaternion(rotation)[2]
        delta_x = translation[0] - self.start_pose[0]
        delta_y = translation[1] - self.start_pose[1]
        body_x = (math.cos(self.start_yaw) * delta_x
                  + math.sin(self.start_yaw) * delta_y)
        body_y = (-math.sin(self.start_yaw) * delta_x
                  + math.cos(self.start_yaw) * delta_y)
        body_yaw = (yaw - self.start_yaw + math.pi) % (2.0 * math.pi) - math.pi
        diagnostics = self.latest_diagnostics or {}
        state = self.latest_state
        step = self.active_step
        now = rospy.Time.now().to_sec()
        self.trace_writer.writerow({
            'ros_time': '{:.9f}'.format(now),
            'elapsed_s': '{:.3f}'.format(now - self.started_at),
            'step': '' if step is None else step.index,
            'axis': '' if step is None else step.axis,
            'phase': '' if step is None else step.phase,
            'offset': '' if step is None else step.offset,
            'target_x': target[0], 'target_y': target[1],
            'target_z': target[2], 'target_yaw_deg': math.degrees(target[3]),
            'current_x': translation[0], 'current_y': translation[1],
            'current_z': translation[2], 'current_yaw_deg': math.degrees(yaw),
            'body_x_m': body_x, 'body_y_m': body_y,
            'body_yaw_deg': math.degrees(body_yaw),
            'state': '' if state is None else state.state,
            'position_error_m': '' if state is None else state.position_error,
            'yaw_error_deg': '' if state is None else math.degrees(state.yaw_error),
            'horizontal_speed_mps': '' if state is None else state.horizontal_speed,
            'yaw_rate_deg_s': '' if state is None else math.degrees(state.yaw_rate),
            'tx': diagnostics.get('tx', ''), 'ty': diagnostics.get('ty', ''),
            'mz': diagnostics.get('mz', ''),
            'xy_braking': int(bool(diagnostics.get('xy_braking', False))),
            'yaw_braking': int(bool(diagnostics.get('yaw_braking', False))),
        })
        self.trace_file.flush()

    def _execute_step(self, step):
        message, target = self._make_goal(step)
        self.active_step = step
        deadline = rospy.Time.now() + rospy.Duration(self.action_timeout)
        arrived_since = None
        rate = rospy.Rate(self.publish_rate_hz)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self._assert_no_other_goal_publisher()
            message.header.stamp = rospy.Time.now()
            self.goal_pub.publish(message)
            self._trace(target)
            state = self.latest_state
            arrived = (
                self._state_fresh() and state is not None
                and state.startup_complete
                and state.state == MotionState.HOVER
                and self._goal_matched(target))
            if arrived:
                if arrived_since is None:
                    arrived_since = rospy.Time.now()
                elif (rospy.Time.now() - arrived_since).to_sec() >= (
                        self.settle_seconds):
                    return
            else:
                arrived_since = None
            rate.sleep()
        raise RuntimeError(
            '{} {} 激励未在 {:.1f} s 内稳定'.format(
                step.axis, step.phase, self.action_timeout))

    def _cancel(self):
        self.cancel_pub.publish(Empty())

    def _on_shutdown(self):
        if not self.completed and not self.aborted:
            self.aborted = True
            try:
                self._cancel()
            except rospy.ROSException:
                pass
        self._close_trace()

    def run(self):
        self._wait_for_hover()
        self._lock_start_pose()
        self.started_at = rospy.Time.now().to_sec()
        rospy.loginfo(
            '%s: 开始 %d 个闭环单轴激励，原始数据写入 %s',
            NODE_NAME, len(self.steps), self.trace_path)
        for step in self.steps:
            rospy.loginfo(
                '%s: [%d/%d] %s %s，幅值 %.3f',
                NODE_NAME, step.index, len(self.steps), step.axis,
                step.phase, step.magnitude)
            self._execute_step(step)
        self.active_step = None
        self.completed = True
        self._close_trace()
        rospy.loginfo('%s: 标定激励完成，载体已回到锁定起点', NODE_NAME)


def main():
    rospy.init_node(NODE_NAME)
    try:
        MotionEffectivenessCalibrationTest().run()
    except (ValueError, RuntimeError, OSError, IOError, ROSMasterException) as error:
        rospy.logfatal('%s: %s', NODE_NAME, error)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
