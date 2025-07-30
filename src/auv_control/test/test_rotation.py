#! /home/xhy/xhy_env36/bin/python
"""
名称: test_rotation.py
功能: 测试航向角旋转控制
作者: buyegaid
发布: /target (PoseStamped.msg)
"""

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
import tf2_ros
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np

class RotationTest:
    def __init__(self):
        # 创建发布者
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10)
        self.rate = rospy.Rate(5)  # 5Hz
        
        # 固定位置参数（从参数服务器读取）
        self.pos_x = rospy.get_param('~pos_x', 1)  # 北向位置
        self.pos_y = rospy.get_param('~pos_y', -4)  # 东向位置
        self.pos_z = rospy.get_param('~pos_z', 0.25)  # 深度
        
        # 姿态参数
        self.pitch = rospy.get_param('~pitch', 2.0)  # 固定俯仰角（度）
        self.roll = 0.0  # 横滚角固定为0
        self.yaw = 0.0  # 目标航向角
        
        # 旋转控制参数
        self.step = 2.0  # 统一步长为10度
        self.rotation_direction = 1  # 1表示顺时针，-1表示逆时针
        
        # 添加tf相关
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        rospy.loginfo(f"初始位置: ({self.pos_x}, {self.pos_y}, {self.pos_z})")
        rospy.loginfo(f"固定俯仰角: {self.pitch}度")
        rospy.loginfo(f"旋转步长: {self.step}度")

    def get_current_yaw(self):
        """获取base_link当前航向角"""
        try:
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
            quat = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            return np.degrees(yaw)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"获取当前航向角失败: {e}")
            return None

    def calculate_target_yaw(self, current_yaw):
        """
        根据当前航向计算目标航向
        
        Parameters:
            current_yaw: float, 当前航向角(度)
        
        Returns:
            target_yaw: float, 目标航向角(度)
        """
        if current_yaw is None:
            return self.yaw
            
        # 计算下一个目标航向
        next_yaw = current_yaw + (self.step * self.rotation_direction)
        
        # 归一化到[-180, 180]
        if next_yaw > 180:
            next_yaw = -180
            self.rotation_direction = -1  # 切换方向
        elif next_yaw < -180:
            next_yaw = 180
            self.rotation_direction = 1  # 切换方向
            
        return next_yaw

    def run(self):
        """主循环"""
        while not rospy.is_shutdown():
            # 获取当前航向角
            current_yaw = self.get_current_yaw()
            
            # 根据当前航向计算目标航向
            self.yaw = self.calculate_target_yaw(current_yaw)
            
            # 构造目标位姿消息
            target = PoseStamped()
            target.header.stamp = rospy.Time.now()
            target.header.frame_id = "map"
            
            # 设置固定位置
            target.pose.position.x = self.pos_x
            target.pose.position.y = self.pos_y
            target.pose.position.z = self.pos_z
            
            # 计算四元数
            q = quaternion_from_euler(
                np.radians(self.roll),
                np.radians(self.pitch),
                np.radians(self.yaw)
            )
            target.pose.orientation = Quaternion(*q)
            
            # 发布目标位姿
            self.target_pub.publish(target)
            if current_yaw is not None:
                rospy.loginfo_throttle(1, f"目标航向: {self.yaw:.1f}度, 当前航向: {current_yaw:.1f}度")
            else:
                rospy.loginfo_throttle(1, f"目标航向: {self.yaw:.1f}度")
            
            self.rate.sleep()

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('test_rotation')
    try:
        node = RotationTest()
        node.run()
    except rospy.ROSInterruptException:
        pass

