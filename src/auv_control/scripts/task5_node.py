#! /home/xhy/xhy_env36/bin/python
"""
名称: task5_node.py
功能: 移动到指定位置后上浮
作者: buyegaid
监听：/tf (来自tf树)
发布：/target (PoseStamped.msg) 被tf_handler订阅, 代表目标位置
      /finished (String) 被state_control订阅, 代表任务是否完成

记录：
2025.7.29 
    完成初版task5_node.py
"""

import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import numpy as np

class Task5Node:
    def __init__(self):
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10)
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10)
        self.rate = rospy.Rate(5)  # 5Hz
        self.tf_listener = tf.TransformListener()
        
        # 变量定义
        self.step = 0  # 程序运行阶段
        self.target_pose = PoseStamped()  # 期望位置消息定义
        
        # 从参数服务器获取目标点位置
        self.start_point = PoseStamped()
        self.start_point.header.frame_id = "map"
        start_point_from_param = rospy.get_param('/task5_point1', [-1.55, -7.97, 0.2, 90])  # 默认值
        self.start_point.pose.position.x = start_point_from_param[0]
        self.start_point.pose.position.y = start_point_from_param[1]
        self.start_point.pose.position.z = start_point_from_param[2]
        self.start_point.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.radians(start_point_from_param[3])))

    def numpy_distance(self, p1:Point, p2:Point):
        """计算两点间距离"""
        a = np.array([p1.x, p1.y, p1.z])
        b = np.array([p2.x, p2.y, p2.z])
        return np.linalg.norm(a - b)

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
            rospy.logwarn(f"task5 node: 获取当前位姿失败: {e}")
            return None

    def generate_smooth_pose(self, current_pose:PoseStamped, target_pose:PoseStamped, max_xy_step=0.8, max_z_step=0.1, max_yaw_step=np.radians(5)):
        """
        使用三阶贝塞尔曲线生成平滑的路径点，采用先前向移动再调整航向的策略
        
        Parameters:
            current_pose: PoseStamped, 当前位姿
            target_pose: PoseStamped, 目标位姿
            max_xy_step: float, 最大水平步长(米)
            max_z_step: float, 最大垂直步长(米)
            max_yaw_step: float, 最大偏航角步长(弧度)
            
        Returns:
            next_pose: PoseStamped, 下一个位姿点
        """
        # 创建下一个位姿点
        next_pose = PoseStamped()
        next_pose.header.frame_id = "map"
        next_pose.header.stamp = rospy.Time.now()
        
        # 获取当前和目标的姿态角
        _, _, current_yaw = euler_from_quaternion([
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ])
        _, target_pitch, target_yaw = euler_from_quaternion([
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w
        ])
        
        # 计算起点和终点
        p0 = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z])
        p3 = np.array([target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z])
        
        # 计算到目标点的距离
        dist_to_target = self.numpy_distance(current_pose.pose.position, target_pose.pose.position)

        # 如果距离目标点很近(小于1米)，则开始调整最终姿态
        if dist_to_target < max_xy_step:
            # 计算yaw角差异（处理角度环绕）
            dyaw = target_yaw - current_yaw
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            dyaw = np.clip(dyaw, -max_yaw_step, max_yaw_step) # 应用最大步长
            next_yaw = current_yaw + dyaw
            
            # 平滑过渡姿态角
            # next_roll = current_roll + np.clip(target_roll - current_roll, -max_yaw_step, max_yaw_step)
            next_pitch = target_pitch  # 保持目标俯仰角
            
            next_pose.pose.position = target_pose.pose.position
            next_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, next_pitch, next_yaw))
            return next_pose
        
        # 如果距离目标点较远，继续沿着前进方向移动
        # 计算前进方向的单位向量
        direction = p3 - p0
        direction_xy = direction[:2]
        direction_xy_norm = np.linalg.norm(direction_xy)
        if direction_xy_norm > 0:
            direction_xy = direction_xy / direction_xy_norm
            # 计算期望的航向角(前进方向)
            desired_yaw = np.arctan2(direction_xy[1], direction_xy[0])
            
            # 计算航向差
            dyaw = desired_yaw - current_yaw
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            dyaw = np.clip(dyaw, -max_yaw_step, max_yaw_step)
            next_yaw = current_yaw + dyaw
        else:
            next_yaw = current_yaw
            
        # 保持当前的俯仰和横滚角
        # 保持当前的相当于是不控制，对于横滚来说不影响，对于俯仰来说，还是需要控制的
        # next_roll = current_roll
        next_pitch = target_pitch # 保持目标俯仰角
        
        # 计算控制点（根据前进方向）
        control_dist = dist_to_target * 0.4
        p1 = p0 + control_dist * np.array([np.cos(next_yaw), np.sin(next_yaw), 0])
        p2 = p3 - control_dist * np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
        
        # 如果没有存储当前的贝塞尔曲线参数t值，初始化为0
        if not hasattr(self, 'bezier_t'):
            self.bezier_t = 0.0
        
        # 计算下一个t值（确保平滑过渡）
        dt = 0.1  # t的增量
        self.bezier_t = min(1.0, self.bezier_t + dt)
        t = self.bezier_t
        
        # 计算三阶贝塞尔曲线上的点
        next_point = (1-t)**3 * p0 + \
                    3*(1-t)**2 * t * p1 + \
                    3*(1-t) * t**2 * p2 + \
                    t**3 * p3
        
        # 应用步长限制
        dp = next_point - p0
        dist_xy = np.sqrt(dp[0]**2 + dp[1]**2)
        if dist_xy > max_xy_step:
            scale = max_xy_step / dist_xy
            dp[0] *= scale
            dp[1] *= scale
        dp[2] = np.clip(dp[2], -max_z_step, max_z_step)
        
        # 设置下一个位置
        next_pose.pose.position.x = current_pose.pose.position.x + dp[0]
        next_pose.pose.position.y = current_pose.pose.position.y + dp[1]
        next_pose.pose.position.z = current_pose.pose.position.z + dp[2]
        
        # 设置姿态
        next_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, next_pitch, next_yaw))
        
        # 如果到达目标点，重置贝塞尔曲线参数
        if dist_to_target < 0.1:  # 距离阈值
            if hasattr(self, 'bezier_t'):
                del self.bezier_t
        
        return next_pose

    def move_to_target(self, max_dist=0.2):
        """移动到目标位置"""
        try:
            current_pose = self.get_current_pose()
            if current_pose is None:
                return False
            
            # 检查是否到达目标点
            if self.numpy_distance(current_pose.pose.position, self.target_pose.pose.position) < max_dist:
                return True
            
            # 生成平滑路径点
            next_pose = self.generate_smooth_pose(current_pose, self.target_pose)
            self.target_pub.publish(next_pose)
            
            # 输出当前状态
            dist = self.numpy_distance(current_pose.pose.position, self.target_pose.pose.position)
            rospy.loginfo_throttle(1, f"task5 node: 移动到目标点，当前距离: {dist:.2f}米")
            
            return False
            
        except tf.Exception as e:
            rospy.logwarn(f"task5 node: 移动失败: {e}")
            return False

    def move_to_start_point(self):
        """移动到起始点"""
        self.target_pose = self.start_point
        return self.move_to_target()

    def ascend(self):
        """上浮到水面"""
        try:
            current_pose = self.get_current_pose()
            if current_pose is None:
                return False
            
            # 设置上浮目标点（保持当前xy位置和姿态，改变z坐标）
            self.target_pose = current_pose
            self.target_pose.pose.position.z = 0.0  # 上浮到水面
            
            return self.move_to_target(max_dist=0.1)  # 使用更小的距离阈值
            
        except tf.Exception as e:
            rospy.logwarn(f"task5 node: 上浮失败: {e}")
            return False

    def finish_task(self):
        """完成任务"""
        self.finished_pub.publish("task5 finished")
        rospy.loginfo("task5 node: 任务完成，发布完成消息")
        rospy.signal_shutdown("任务完成")
        return True

    def run(self):
        """主循环"""
        while not rospy.is_shutdown():
            if self.step == 0:  # 移动到起始点
                if self.move_to_start_point():
                    self.step = 1
                    rospy.loginfo("task5 node: 到达起始点，准备上浮")
            elif self.step == 1:  # 上浮
                if self.ascend():
                    self.step = 2
                    rospy.loginfo("task5 node: 上浮完成，任务结束")
            elif self.step == 2:  # 完成任务
                self.finish_task()
                break
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('task5_node')
    try:
        node = Task5Node()
        node.run()
    except rospy.ROSInterruptException:
        pass