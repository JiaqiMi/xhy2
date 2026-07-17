#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_goal_test.py
功能：根据初始位姿和相对偏置向 motion_supervisor 发布一次 map 目标
作者：BroXu
发布：/cmd/motion/goal (geometry_msgs/PoseStamped)
说明：
    1. 启动后读取初始 map -> base_link 位姿；
    2. offset_frame=base_link 时，offset_x/offset_y 表示前/右偏置；
    3. offset_frame=map 时，offset_x/offset_y 表示北/东偏置；
    4. 目标航向为初始航向加 yaw_offset_deg，目标 z 固定为 -0.6 m；
    5. 发布器使用锁存模式，仅用于运动管理原型的水池测试。
记录：
2026.7.16
    新增原型目标发布工具。
2026.7.16
    改为读取初始位姿后发布 base_link 或 map 坐标系下的相对目标。
"""

import math

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from motion_supervisor_core import relative_target_xy, wrap_angle


TARGET_Z = -0.6
SUPPORTED_OFFSET_FRAMES = ('base_link', 'map')


def main():
    rospy.init_node('motion_goal_test')
    publisher = rospy.Publisher(
        '/cmd/motion/goal', PoseStamped, queue_size=1, latch=True)

    offset_frame = str(
        rospy.get_param('~offset_frame', 'base_link')).strip().lstrip('/')
    offset_x = float(rospy.get_param('~offset_x', 1.0))
    offset_y = float(rospy.get_param('~offset_y', 0.0))
    yaw_offset_deg = float(rospy.get_param('~yaw_offset_deg', 0.0))
    publish_delay = max(0.0, float(rospy.get_param('~publish_delay', 1.0)))
    tf_timeout = max(0.1, float(rospy.get_param('~tf_timeout', 5.0)))

    values = (offset_x, offset_y, yaw_offset_deg, publish_delay, tf_timeout)
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

    rospy.sleep(publish_delay)
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
        translation[0],
        translation[1],
        initial_yaw,
        offset_x,
        offset_y,
        offset_frame,
    )
    target_yaw = wrap_angle(
        initial_yaw + math.radians(yaw_offset_deg))

    target = PoseStamped()
    target.header.stamp = rospy.Time.now()
    target.header.frame_id = 'map'
    target.pose.position.x = target_x
    target.pose.position.y = target_y
    target.pose.position.z = TARGET_Z
    quaternion = quaternion_from_euler(
        0.0, 0.0, target_yaw)
    target.pose.orientation.x = quaternion[0]
    target.pose.orientation.y = quaternion[1]
    target.pose.orientation.z = quaternion[2]
    target.pose.orientation.w = quaternion[3]
    publisher.publish(target)
    rospy.loginfo(
        'motion_goal_test: 初始 map 位姿 '
        '(x=%.2f, y=%.2f, z=%.2f, yaw=%.1fdeg)',
        translation[0],
        translation[1],
        translation[2],
        math.degrees(initial_yaw),
    )
    rospy.loginfo(
        'motion_goal_test: 偏置 frame=%s '
        '(x=%.2f, y=%.2f, yaw=%.1fdeg)',
        offset_frame,
        offset_x,
        offset_y,
        yaw_offset_deg,
    )
    rospy.loginfo(
        'motion_goal_test: 已发布 map 目标 '
        '(x=%.2f, y=%.2f, z=%.2f, yaw=%.1fdeg)',
        target_x,
        target_y,
        TARGET_Z,
        math.degrees(target_yaw),
    )
    rospy.spin()


if __name__ == '__main__':
    main()
