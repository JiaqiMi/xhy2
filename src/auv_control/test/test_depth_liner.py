#!/usr/bin/env python
"""
名称: test_depth_liner.py
功能: 测试task4_node中的深度插值功能
作者: Assistant
描述: 循环获取tf数据，赋值给target_posestamped，然后进行深度插值，打印日志验证插值是否正确
"""

import rospy
import tf
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Quaternion, Point

NODE_NAME = "test_depth_liner"

class DepthInterpolationTest:
    def __init__(self):
        # ros相关的初始化
        rospy.init_node(NODE_NAME, anonymous=True)
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(2)  # 2Hz

        # 初始化target_posestamped
        self.target_posestamped = PoseStamped()
        self.target_posestamped.header.frame_id = "map"
        
        # 获取参数，使用与task4_node相同的参数
        start_point_from_param = rospy.get_param('/task4_point0', [0.5, -0.5, 0.15, 0.0])  # 默认值
        end_point_from_param = rospy.get_param('/task4_point1', [0.5, -0.5, 0.15, 0.0])  # 默认值
        
        # 设置插值参数
        self.e0 = start_point_from_param[1]  # 起始点的y坐标
        self.e1 = end_point_from_param[1]    # 结束点的y坐标
        self.d0 = start_point_from_param[2]  # 起始点的深度
        self.d1 = end_point_from_param[2]    # 结束点的深度
        
        rospy.loginfo(f"{NODE_NAME}: 初始化完成")
        rospy.loginfo(f"{NODE_NAME}: 插值参数 - e0={self.e0}, e1={self.e1}, d0={self.d0}, d1={self.d1}")
        
        # 如果参数相同，添加一些测试数据
        if self.e0 == self.e1 or self.d0 == self.d1:
            rospy.logwarn(f"{NODE_NAME}: 检测到参数相同，使用测试数据")
            self.e0 = -1.0  # 起始y坐标
            self.e1 = 1.0   # 结束y坐标
            self.d0 = 0.1   # 起始深度
            self.d1 = 0.5   # 结束深度
            rospy.loginfo(f"{NODE_NAME}: 测试参数 - e0={self.e0}, e1={self.e1}, d0={self.d0}, d1={self.d1}")

    def get_current_pose(self):
        """获取当前位姿"""
        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            
            current_pose = PoseStamped()
            current_pose.header.frame_id = "map"
            current_pose.header.stamp = rospy.Time.now()
            current_pose.pose.position = Point(*trans)
            current_pose.pose.orientation = Quaternion(*rot)
            
            return current_pose
        except tf.Exception as e:
            rospy.logwarn(f"{NODE_NAME}: 获取当前位姿失败: {e}")
            return None

    def update_depth(self):
        """深度插值函数 - 从task4_node复制而来"""
        # 根据target_posestamped的e来线性化一个深度
        current_y = self.target_posestamped.pose.position.y
        
        # 线性插值计算深度
        # 当y=e0时，深度为d0；当y=e1时，深度为d1
        if self.e1 != self.e0:  # 避免除零错误
            # 计算插值比例
            ratio = (current_y - self.e0) / (self.e1 - self.e0)
            # 线性插值计算深度
            interpolated_depth = self.d0 + ratio * (self.d1 - self.d0)
        else:
            # 如果e0和e1相等，使用d0作为深度
            interpolated_depth = self.d0
        
        if current_y < self.e0:
            # 如果当前y小于e0，使用d0作为深度
            interpolated_depth = self.d0
        if current_y > self.e1:
            # 如果当前y大于e1，使用d1作为深度
            interpolated_depth = self.d1
            
        # 更新目标位置的深度
        old_depth = self.target_posestamped.pose.position.z
        self.target_posestamped.pose.position.z = interpolated_depth
        
        # 输出详细的调试信息
        rospy.loginfo(f"{NODE_NAME}: === 深度插值测试 ===")
        rospy.loginfo(f"{NODE_NAME}: 当前y坐标: {current_y:.3f}m")
        rospy.loginfo(f"{NODE_NAME}: 插值参数: e0={self.e0}, e1={self.e1}, d0={self.d0}, d1={self.d1}")
        
        if self.e1 != self.e0:
            ratio = (current_y - self.e0) / (self.e1 - self.e0)
            rospy.loginfo(f"{NODE_NAME}: 插值比例: {ratio:.3f}")
        
        rospy.loginfo(f"{NODE_NAME}: 原始深度: {old_depth:.3f}m")
        rospy.loginfo(f"{NODE_NAME}: 插值深度: {interpolated_depth:.3f}m")
        rospy.loginfo(f"{NODE_NAME}: 深度变化: {interpolated_depth - old_depth:.3f}m")
        rospy.loginfo(f"{NODE_NAME}: ========================")

    def test_with_current_tf(self):
        """使用当前tf数据进行测试"""
        current_pose = self.get_current_pose()
        if current_pose is None:
            rospy.logwarn(f"{NODE_NAME}: 无法获取当前位姿，跳过此次测试")
            return False
            
        # 将当前位姿赋值给target_posestamped
        self.target_posestamped.pose.position = current_pose.pose.position
        self.target_posestamped.pose.orientation = current_pose.pose.orientation
        self.target_posestamped.header = current_pose.header
        
        # 执行深度插值
        self.update_depth()
        
        return True

    def test_with_simulated_positions(self):
        """使用模拟位置数据进行测试"""
        rospy.loginfo(f"{NODE_NAME}: === 开始模拟位置测试 ===")
        
        # 测试多个y坐标值
        test_y_values = [
            self.e0 - 0.5,  # 小于e0
            self.e0,        # 等于e0
            (self.e0 + self.e1) / 2,  # 中间值
            self.e1,        # 等于e1
            self.e1 + 0.5,  # 大于e1
        ]
        
        for y_val in test_y_values:
            # 设置测试位置
            self.target_posestamped.pose.position.x = 0.0
            self.target_posestamped.pose.position.y = y_val
            self.target_posestamped.pose.position.z = 0.3  # 初始深度
            
            # 执行插值测试
            rospy.loginfo(f"{NODE_NAME}: 测试y坐标: {y_val:.3f}")
            self.update_depth()
            rospy.sleep(1.0)  # 等待1秒便于观察

    def run(self):
        """主循环"""
        rospy.loginfo(f"{NODE_NAME}: 开始深度插值测试")
        
        # 先进行模拟位置测试
        self.test_with_simulated_positions()
        
        rospy.loginfo(f"{NODE_NAME}: 模拟测试完成，开始实时tf测试")
        
        # 然后进行实时tf测试
        # test_count = 0
        # while not rospy.is_shutdown() and test_count < 20:  # 限制测试次数
        while not rospy.is_shutdown():
            # rospy.loginfo(f"{NODE_NAME}: 第 {test_count + 1} 次实时tf测试")
            
            self.test_with_current_tf()
                # test_count += 1
            self.rate.sleep()
        
        # rospy.loginfo(f"{NODE_NAME}: 测试完成，总共进行了 {test_count} 次tf测试")

if __name__ == '__main__':
    try:
        test_node = DepthInterpolationTest()
        test_node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo(f"{NODE_NAME}: 测试被中断")
    except Exception as e:
        rospy.logerr(f"{NODE_NAME}: 测试出错: {e}")