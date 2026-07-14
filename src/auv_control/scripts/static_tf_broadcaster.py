#! /home/xhy/xhy_env/bin/python3.8
"""
名称：static_tf_broadcaster.py
功能：机器人内部刚体坐标系的静态变换发布器
作者：buyegaid
监听：None
发布：/tf(from scan to base_link)
      /tf(from imu to base_link)
记录：
2025.7.19 11:15
    第一版完成
2025.7.22 17:38
    添加base_link到hand的变换
    添加base_link到camera的变换
2025.8.3 11:32
    改为 10s的心跳包
"""

import rospy
import tf
from geometry_msgs.msg import TransformStamped, Quaternion
import numpy as np

NODE_NAME = "static_tf_broadcaster"

class Static_tf_broadcaster:
    """静态变换发布器
    1. 发布base_link到imu的静态变换
    2. 发布base_link到scan的静态变换
    """
    def __init__(self):
        self.tf_broadcaster = tf.TransformBroadcaster()
        # base_link 到 imu的变换参数
        self.imu_trans = (0.0, 0.0, 0.0)
        self.imu_rot = tf.transformations.quaternion_from_euler(0, 0, 0)
        
        # base_link 到 hand的变换参数
        self.hand_trans = (0.632, 0, 0.068) # 夹爪中心位置
        self.hand_rot = tf.transformations.quaternion_from_euler(0, 0, 0)

        # base_link 到 down camera的变换参数
        self.camera_trans = (0.658, -0.030, -0.210) # 左眼坐标位置
        self.camera_rot = tf.transformations.quaternion_from_euler(0, 0, np.radians(90))

        # base_link 到 front camera的变换参数
        self.camera2_trans = (0.713, 0 ,-0.360)
        self.camera2_rot = tf.transformations.quaternion_from_euler(0, 0, 0)
        
        # 发布频率 10Hz
        self.rate = rospy.Rate(10)
        rospy.loginfo(f"{NODE_NAME}: 已启动")

    def run(self):
        """主循环：发布静态变换"""
        while not rospy.is_shutdown():
            current_time = rospy.Time.now()
            # 发布所有静态变换
            self.tf_broadcaster.sendTransform(self.imu_trans, self.imu_rot, current_time, "imu", "base_link")
            self.tf_broadcaster.sendTransform(self.hand_trans, self.hand_rot, current_time, "hand", "base_link")
            self.tf_broadcaster.sendTransform(self.camera_trans, self.camera_rot, current_time, "camera", "base_link")
            self.tf_broadcaster.sendTransform(self.camera2_trans, self.camera2_rot, current_time, "camera_front", "base_link")
            rospy.loginfo_throttle(10, f"{NODE_NAME}: AUV 静态tf广播完成")
            self.rate.sleep()

if __name__ == "__main__":
    try:
        rospy.init_node(f'{NODE_NAME}',anonymous=True)
        bd = Static_tf_broadcaster()
        bd.run()
    except rospy.ROSInterruptException:
        pass
