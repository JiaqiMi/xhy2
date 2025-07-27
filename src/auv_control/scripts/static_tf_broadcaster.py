#! /home/xhy/xhy_env36/bin/python
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
"""

import rospy
import tf
import tf2_ros
from geometry_msgs.msg import TransformStamped, Quaternion
import numpy as np

class Static_tf_broadcaster:
    """静态变换发布器
    1. 发布base_link到imu的静态变换
    2. 发布base_link到scan的静态变换
    """
    def __init__(self):
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        # base_link到imu的变换: 重合
        self.t_imu = TransformStamped()
        self.t_imu.header.stamp = rospy.Time.now()
        self.t_imu.header.frame_id = "base_link"
        self.t_imu.child_frame_id = "imu"
        self.t_imu.transform.translation.x = 0.0
        self.t_imu.transform.translation.y = 0.0
        self.t_imu.transform.translation.z = 0.0
        self.t_imu.transform.rotation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, 0)) # 无旋转
        
        # base_link到hand 的变换: 机械爪坐标系原点在惯导坐标系下的坐标为(1061.7mm,-2.17mm,21.235mm),坐标系设定为前右下，不用进行旋转
        self.t_hand = TransformStamped()
        self.t_hand.header.stamp = rospy.Time.now()
        self.t_hand.header.frame_id = "base_link"
        self.t_hand.child_frame_id = "hand"
        self.t_hand.transform.translation.x = 1.0617 # 前
        self.t_hand.transform.translation.y = -0.00217 # 右 
        self.t_hand.transform.translation.z = 0.021235 # 下
        self.t_hand.transform.rotation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, 0)) # 无旋转

        # base_link到camera的变换: 摄像机的坐标系原点在惯导坐标系下坐标为(693.59mm,-30.37mm,-184.04mm)，从惯导坐标系变为摄像机坐标系需要先绕z轴转90度，再绕x轴转75度
        self.t_camera = TransformStamped()
        self.t_camera.header.stamp = rospy.Time.now()
        self.t_camera.header.frame_id = "base_link"
        self.t_camera.child_frame_id = "camera"
        self.t_camera.transform.translation.x = 0.69359 # 前
        self.t_camera.transform.translation.y = -0.03037 # 右
        self.t_camera.transform.translation.z = -0.18404 # 下
        self.t_camera.transform.rotation = Quaternion(*tf.transformations.quaternion_from_euler(np.radians(75), 0, np.radians(90)))

        # 发布频率 5Hz
        self.rate = rospy.Rate(5)
        rospy.loginfo("static_tfbc: 已启动")

    def run(self):
        """主循环：发布静态变换"""
        while not rospy.is_shutdown():
            self.static_tf_broadcaster.sendTransform([self.t_imu, self.t_hand, self.t_camera])
            rospy.loginfo_throttle(10,"static_tfbc: 机器人tf发布中...")
            self.rate.sleep() # 5Hz

if __name__ == "__main__":
    try:
        rospy.init_node('static_tfbroadcaster')
        bd = Static_tf_broadcaster()
        bd.run()
    except rospy.ROSInterruptException:
        pass
