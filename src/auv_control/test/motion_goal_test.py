#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_goal_test.py
功能：向 motion_supervisor 发布一次 map 目标
作者：BroXu
监听：/motion/state (auv_control/MotionState)
发布：/cmd/motion/goal (geometry_msgs/PoseStamped)
说明：
    1. 默认根据初始位姿和相对偏置生成 map 目标；
    2. target_mode=absolute 时直接使用 map 下绝对位置和航向；
    3. exit_after_hover=true 时确认目标进入 HOVER 后保持指定时间并退出。
记录：
2026.7.16
    新增原型目标发布工具。
2026.7.16
    改为读取初始位姿后发布 base_link 或 map 坐标系下的相对目标。
2026.7.17
    适配 test 与 driver 分目录结构，补充控制核心模块搜索路径。
2026.7.17
    增加可配置的 NED/map 绝对目标深度参数 target_z。
2026.7.18
    明确 target_z 仅用于生成测试目标，不覆盖运动管理器收到的正式目标深度。
2026.7.30
    改为接收 map 下绝对位置与航向；单帧发布后等待 HOVER 3 秒并退出。
2026.7.30
    增加 target_mode，保留原有相对目标模式并支持绝对目标模式切换。
2026.7.30
    将 HOVER 后退出行为独立为 exit_after_hover，供单次移动的两种目标模式共用。
"""

import math
import os
import sys

import rospy
import tf
from auv_control.msg import MotionState
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler


# catkin devel 模式直接执行 test 下的源码，需要显式加入相邻 driver 目录。
DRIVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'driver'))
if DRIVER_DIR not in sys.path:
    sys.path.insert(0, DRIVER_DIR)

from motion_supervisor_core import relative_target_xy, wrap_angle


DEFAULT_TARGET_Z = -0.6
SUPPORTED_OFFSET_FRAMES = ('base_link', 'map')


class MotionGoalOnceNode(object):
    """发布一次绝对目标，确认定点接管后等待并退出。"""

    def __init__(self):
        self.target = None
        self.hover_started_at = None
        self.goal_published_at = None
        self.hover_duration = 3.0
        self.state_timeout = 0.5
        self.position_tolerance = 0.01
        self.yaw_tolerance = math.radians(1.0)

    def state_callback(self, message):
        """仅接受当前目标对应的新鲜 HOVER 状态。"""
        if self.target is None or self.goal_published_at is None:
            return
        age = (rospy.Time.now() - message.header.stamp).to_sec()
        if age < 0.0 or age > self.state_timeout:
            self.hover_started_at = None
            return
        if message.header.stamp < self.goal_published_at:
            return
        if (message.state != MotionState.HOVER
                or not message.startup_complete
                or not self._goal_matches(message.goal)):
            self.hover_started_at = None
            return
        if self.hover_started_at is None:
            self.hover_started_at = rospy.Time.now()
            rospy.loginfo(
                'motion_goal_test: 目标已进入 HOVER，保持 %.1f s 后退出',
                self.hover_duration)

    def _goal_matches(self, goal):
        """判断状态机当前采用的目标是否为本节点发布的目标。"""
        position = goal.pose.position
        target_position = self.target.pose.position
        if (abs(position.x - target_position.x) > self.position_tolerance
                or abs(position.y - target_position.y) > self.position_tolerance
                or abs(position.z - target_position.z) > self.position_tolerance):
            return False
        goal_yaw = euler_from_quaternion((
            goal.pose.orientation.x,
            goal.pose.orientation.y,
            goal.pose.orientation.z,
            goal.pose.orientation.w,
        ))[2]
        target_yaw = euler_from_quaternion((
            self.target.pose.orientation.x,
            self.target.pose.orientation.y,
            self.target.pose.orientation.z,
            self.target.pose.orientation.w,
        ))[2]
        return abs(wrap_angle(goal_yaw - target_yaw)) <= self.yaw_tolerance


def main():
    rospy.init_node('motion_goal_test')
    publisher = rospy.Publisher(
        '/cmd/motion/goal', PoseStamped, queue_size=1, latch=True)
    target_mode = str(
        rospy.get_param('~target_mode', 'relative')).strip().lower()
    if target_mode not in ('relative', 'absolute'):
        rospy.logfatal(
            'motion_goal_test: target_mode=%s 无效，仅支持 relative 或 absolute',
            target_mode)
        return
    absolute_target = target_mode == 'absolute'
    exit_after_hover = bool(rospy.get_param('~exit_after_hover', False))
    publish_delay = max(0.0, float(rospy.get_param('~publish_delay', 1.0)))
    hover_duration = max(
        0.0, float(rospy.get_param('~hover_duration', 3.0)))
    if not math.isfinite(publish_delay):
        rospy.logfatal('motion_goal_test: 参数必须是有限数值')
        return

    rospy.sleep(publish_delay)

    if absolute_target:
        target_x = float(rospy.get_param('~target_x', 0.0))
        target_y = float(rospy.get_param('~target_y', 0.0))
        target_z = float(rospy.get_param('~target_z', -0.9))
        target_yaw_deg = float(rospy.get_param('~target_yaw_deg', 0.0))
        values = (target_x, target_y, target_z, target_yaw_deg,
                  hover_duration)
        if not all(math.isfinite(value) for value in values):
            rospy.logfatal('motion_goal_test: 参数必须是有限数值')
            return
        target_yaw = math.radians(target_yaw_deg)
    else:
        offset_frame = str(
            rospy.get_param('~offset_frame', 'base_link')).strip().lstrip('/')
        offset_x = float(rospy.get_param('~offset_x', 1.0))
        offset_y = float(rospy.get_param('~offset_y', 0.0))
        yaw_offset_deg = float(rospy.get_param('~yaw_offset_deg', 0.0))
        target_z = float(rospy.get_param('~target_z', DEFAULT_TARGET_Z))
        tf_timeout = max(
            0.1, float(rospy.get_param('~tf_timeout', 5.0)))
        values = (
            offset_x, offset_y, yaw_offset_deg, target_z, tf_timeout,
            hover_duration)
        if not all(math.isfinite(value) for value in values):
            rospy.logfatal('motion_goal_test: 参数必须是有限数值')
            return
        if offset_frame not in SUPPORTED_OFFSET_FRAMES:
            rospy.logfatal(
                'motion_goal_test: offset_frame=%s 无效，仅支持 %s',
                offset_frame,
                ', '.join(SUPPORTED_OFFSET_FRAMES),
            )
            return
        listener = tf.TransformListener()
        try:
            listener.waitForTransform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(tf_timeout))
            translation, rotation = listener.lookupTransform(
                'map', 'base_link', rospy.Time(0))
        except tf.Exception as error:
            rospy.logfatal(
                'motion_goal_test: 无法获取初始 map -> base_link TF: %s', error)
            return
        initial_yaw = euler_from_quaternion(rotation)[2]
        initial_values = (
            translation[0], translation[1], translation[2], initial_yaw)
        if not all(math.isfinite(value) for value in initial_values):
            rospy.logfatal('motion_goal_test: 初始 TF 包含非有限值')
            return
        target_x, target_y = relative_target_xy(
            translation[0], translation[1], initial_yaw, offset_x, offset_y,
            offset_frame)
        target_yaw = wrap_angle(initial_yaw + math.radians(yaw_offset_deg))

    target = PoseStamped()
    target.header.stamp = rospy.Time.now()
    target.header.frame_id = 'map'
    target.pose.position.x = target_x
    target.pose.position.y = target_y
    target.pose.position.z = target_z
    quaternion = quaternion_from_euler(0.0, 0.0, target_yaw)
    target.pose.orientation.x = quaternion[0]
    target.pose.orientation.y = quaternion[1]
    target.pose.orientation.z = quaternion[2]
    target.pose.orientation.w = quaternion[3]
    if exit_after_hover:
        node = MotionGoalOnceNode()
        node.target = target
        node.hover_duration = hover_duration
        node.goal_published_at = target.header.stamp
        rospy.Subscriber(
            '/motion/state', MotionState, node.state_callback, queue_size=1)
    publisher.publish(target)
    if absolute_target:
        rospy.loginfo(
            'motion_goal_test: 已单帧发布 map 绝对目标 '
            '(x=%.2f, y=%.2f, z=%.2f, yaw=%.1fdeg)',
            target_x, target_y, target_z, target_yaw_deg)
    else:
        rospy.loginfo(
            'motion_goal_test: 偏置 frame=%s '
            '(x=%.2f, y=%.2f, yaw=%.1fdeg)，已发布 map 目标 '
            '(x=%.2f, y=%.2f, z=%.2f, yaw=%.1fdeg)',
            offset_frame, offset_x, offset_y, yaw_offset_deg,
            target_x, target_y, target_z, math.degrees(target_yaw))
    if not exit_after_hover:
        rospy.spin()
        return
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if (node.hover_started_at is not None
                and (rospy.Time.now() - node.hover_started_at).to_sec()
                >= node.hover_duration):
            rospy.loginfo('motion_goal_test: HOVER 保持完成，节点退出')
            return
        rate.sleep()


if __name__ == '__main__':
    main()
