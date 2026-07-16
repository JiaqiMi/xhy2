#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_goal_test.py
功能：向 motion_supervisor 发布一次 map 坐标系原型目标
作者：BroXu
发布：/cmd/motion/goal (geometry_msgs/PoseStamped)
说明：
    1. 目标 x、y、z 和 yaw 通过私有 ROS 参数配置；
    2. 发布器使用锁存模式，仅用于运动管理原型的水池测试。
记录：
2026.7.16
    新增原型目标发布工具。
"""

import math

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


def main():
    rospy.init_node('motion_goal_test')
    publisher = rospy.Publisher(
        '/cmd/motion/goal', PoseStamped, queue_size=1, latch=True)

    target_x = float(rospy.get_param('~target_x', 1.0))
    target_y = float(rospy.get_param('~target_y', 0.0))
    target_z = float(rospy.get_param('~target_z', 1.5))
    target_yaw_deg = float(rospy.get_param('~target_yaw_deg', 0.0))
    publish_delay = max(0.0, float(rospy.get_param('~publish_delay', 1.0)))

    rospy.sleep(publish_delay)
    target = PoseStamped()
    target.header.stamp = rospy.Time.now()
    target.header.frame_id = 'map'
    target.pose.position.x = target_x
    target.pose.position.y = target_y
    target.pose.position.z = target_z
    quaternion = quaternion_from_euler(
        0.0, 0.0, math.radians(target_yaw_deg))
    target.pose.orientation.x = quaternion[0]
    target.pose.orientation.y = quaternion[1]
    target.pose.orientation.z = quaternion[2]
    target.pose.orientation.w = quaternion[3]
    publisher.publish(target)
    rospy.loginfo(
        'motion_goal_test: 已发布目标 (x=%.2f, y=%.2f, z=%.2f, yaw=%.1fdeg)',
        target_x, target_y, target_z, target_yaw_deg)
    rospy.spin()


if __name__ == '__main__':
    main()
