#! /home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-
"""
名称：static_tf_broadcaster.py
功能：发布机器人内部刚体坐标系的静态变换
作者：buyegaid
说明：
    base_link 是动态导航 TF 的唯一艇体根节点；
    正、负计划旋转方向分别使用独立且固定的控制中心坐标系。
记录：
2026.7.20
    将旧单控制中心拆分为 control_link_positive 和 control_link_negative。
"""

import numpy as np
import rospy
import tf

from lever_arm import offset_between_origins


NODE_NAME = 'static_tf_broadcaster'
CONTROL_FRAMES = {
    1: 'control_link_positive',
    -1: 'control_link_negative',
}


class StaticTfBroadcaster(object):
    """周期发布 base_link 下的全部固定刚体变换。"""

    def __init__(self):
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.imu_trans = (
            float(rospy.get_param('~base_to_imu_x', 0.0)),
            float(rospy.get_param('~base_to_imu_y', 0.0)),
            float(rospy.get_param('~base_to_imu_z', 0.0)),
        )
        control_to_imu = {
            1: (
                float(rospy.get_param('~control_to_imu_positive_x', 0.0)),
                float(rospy.get_param('~control_to_imu_positive_y', 0.0)),
                float(rospy.get_param('~control_to_imu_positive_z', 0.0)),
            ),
            -1: (
                float(rospy.get_param('~control_to_imu_negative_x', 0.0)),
                float(rospy.get_param('~control_to_imu_negative_y', 0.0)),
                float(rospy.get_param('~control_to_imu_negative_z', 0.0)),
            ),
        }
        # offset_between_origins(A->IMU, B->IMU) 返回 A->B。
        self.control_trans = {
            direction: offset_between_origins(
                self.imu_trans,
                control_to_imu[direction],
            )
            for direction in CONTROL_FRAMES
        }
        self.identity_rot = tf.transformations.quaternion_from_euler(0, 0, 0)

        self.hand_trans = (0.632, 0, 0.068)
        self.hand_rot = self.identity_rot
        self.camera_trans = (0.658, -0.030, -0.210)
        self.camera_rot = tf.transformations.quaternion_from_euler(
            0, 0, np.radians(90))
        self.camera2_trans = (0.703, 0, -0.360)
        self.camera2_rot = self.identity_rot

        self.rate = rospy.Rate(10)
        rospy.loginfo(
            '%s: 已启动，base_link -> 正中心=(%.3f, %.3f, %.3f) m，'
            'base_link -> 负中心=(%.3f, %.3f, %.3f) m，'
            'base_link -> imu=(%.3f, %.3f, %.3f) m',
            NODE_NAME,
            *(self.control_trans[1]
              + self.control_trans[-1]
              + self.imu_trans)
        )

    def run(self):
        """主循环：周期发布固定刚体变换。"""
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            for direction, frame_name in CONTROL_FRAMES.items():
                self.tf_broadcaster.sendTransform(
                    self.control_trans[direction],
                    self.identity_rot,
                    current_time,
                    frame_name,
                    'base_link',
                )
            self.tf_broadcaster.sendTransform(
                self.imu_trans, self.identity_rot, current_time,
                'imu', 'base_link')
            self.tf_broadcaster.sendTransform(
                self.hand_trans, self.hand_rot, current_time,
                'hand', 'base_link')
            self.tf_broadcaster.sendTransform(
                self.camera_trans, self.camera_rot, current_time,
                'camera', 'base_link')
            self.tf_broadcaster.sendTransform(
                self.camera2_trans, self.camera2_rot, current_time,
                'camera_front', 'base_link')
            rospy.loginfo_throttle(10, '%s: AUV 静态 TF 广播完成', NODE_NAME)
            self.rate.sleep()


if __name__ == '__main__':
    try:
        rospy.init_node(NODE_NAME, anonymous=True)
        StaticTfBroadcaster().run()
    except rospy.ROSInterruptException:
        pass
