#! /home/xhy/xhy_env36/bin/python
"""
名称：task1_node.py
功能：下潜、过门
作者：黄思旭
监听：/target_detection (来自视觉节点)
      /tf (来自tf树)
发布：/target (PoseStamped.msg)
      /finished (String)被state_control订阅, 代表任务是否完成

记录：
2025.7.17 10:50
    1. 完成了初版task1_node.py, 包含下潜、过门、任务完成逻辑
2025.7.19 16:18
    更正坐标系, map是北东地
    新增异步更新AUV位姿的线程, 避免阻塞主循环
2025.7.25 23:31
    更正旋转方法的逻辑
2025.7.27 17:14
    过门后最终应旋转回原来的方向上
    位置选点使用三阶贝塞尔曲线平滑
2025.7.28 18:10
    使用tf的版本
2025.7.29 11:20
    将旋转和移动统一起来，如果位置误差过大，就先移动，否则进行旋转
"""

import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String
from auv_control.msg import TargetDetection
from geometry_msgs.msg import PoseStamped, Point, Quaternion
import numpy as np
from queue import PriorityQueue  # 导入优先队列




class Task1Node:
    """下潜、过门任务"""
    def __init__(self):
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10)                     # 创建任务完成消息发布者，由state_control订阅
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10)                    # 发布目标点，PoseStamped格式
        rospy.Subscriber('/target_detection', TargetDetection, self.target_detection_callback)      # 订阅目标检测消息，来自视觉节点
        self.rate = rospy.Rate(5)                                                                   # 5Hz，控制发布频率                                                               
        self.tf_listener = tf.TransformListener()
        self.step = 0   # 当前步骤

        self.target_posestamped = PoseStamped()  # 用于存储目标点的位姿

        # 一些变量定义
        self.detect_door = rospy.get_param('/detect_door', False)  # 是否检测门
        self.init_depth = rospy.get_param('/depth', 0.3)  # 下潜深度，单位米
        self.door_color = rospy.get_param('/door_color', 'red')  # 目标颜色，默认红色 另一种是蓝色
        
        # 初始化优先队列存储门位置
        self.door_queue = PriorityQueue(maxsize=10)
        self.sequence_number = 0  # 用于优先队列中元素的唯一标识
        self.door_count = 0  # 记录收到的有效门数量

        # 准备执行任务的初始点
        self.start_point = PoseStamped()
        self.start_point.header.frame_id = "map"

        # 从参数服务器获取task1_point1参数列表
        start_point_from_param = rospy.get_param('/task1_point1', [0.5, -0.5, 0.15, 0.0])  # 默认值
        self.start_point.pose.position.x = start_point_from_param[0]
        self.start_point.pose.position.y = start_point_from_param[1]
        self.start_point.pose.position.z = start_point_from_param[2]
        self.start_point.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.radians(start_point_from_param[3])))
        
        # 红色门的点位置
        self.point_reddoor = PoseStamped()
        self.point_reddoor.header.frame_id = "map"

        # 从参数服务器获取task1_point2参数列表
        point_red_door = rospy.get_param('/task1_point2', [0.52, -2.62, 0.15, 0.0])  # 默认值
        self.point_reddoor.pose.position.x = point_red_door[0]
        self.point_reddoor.pose.position.y = point_red_door[1]
        self.point_reddoor.pose.position.z = point_red_door[2]
        self.point_reddoor.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.radians(point_red_door[3])))

        # 蓝色门的点位置
        self.point_bluedoor = PoseStamped()
        self.point_bluedoor.header.frame_id = "map"
        # 从参数服务器获取task1_point3参数列表
        point_bluedoor_params = rospy.get_param('/task1_point3', [1.99, -2.85, 0.15, -90.0])  # 默认值
        self.point_bluedoor.pose.position.x = point_bluedoor_params[0]
        self.point_bluedoor.pose.position.y = point_bluedoor_params[1]
        self.point_bluedoor.pose.position.z = point_bluedoor_params[2]
        self.point_bluedoor.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.radians(point_bluedoor_params[3])))

        # 旋转方向：红色门顺时针（-），蓝色门逆时针（+）
        self.direction = 1 if self.door_color == 'blue' else -1
        self.step_deg = 5
        self.step_rad = np.radians(self.step_deg) * self.direction

    def is_arrival(self, current_pose:PoseStamped, target_pose:PoseStamped, max_dist=0.2, max_yaw_dist=np.radians(0.2)):
        """
        检查是否到达目标位置和航向

        Parameters:
            current_pose: PoseStamped, 当前位姿
            target_pose: PoseStamped, 目标位姿
            max_dist: float, 最大位置误差(米)
            max_yaw_dist: float, 最大航向误差(弧度)

        Returns:
            bool: 是否到达目标位置和航向
        """
        # 计算位置误差
        pos_error = self.numpy_distance(current_pose.pose.position, target_pose.pose.position)

        # 计算航向误差
        yaw_error = self.yaw_distance(current_pose.pose.orientation, target_pose.pose.orientation)
        if pos_error < max_dist and yaw_error < max_yaw_dist:
            return True
        else:
            return False

    def yaw_distance(self, ori1:Quaternion, ori2:Quaternion):
        """
        计算两个航向之间的差值，并处理角度环绕问题
        """
        _, _, yaw1 = euler_from_quaternion([
            ori1.x,
            ori1.y,
            ori1.z,
            ori1.w
        ])
        _, _, yaw2 = euler_from_quaternion([
            ori2.x,
            ori2.y,
            ori2.z,
            ori2.w
        ])
        return abs((abs(yaw1 - yaw2) + np.pi) % (2 * np.pi) - np.pi)

    def numpy_distance(self, p1:Point, p2:Point):
        """
        使用NumPy计算NED距离

        Parameters:
            p1: Point 第一个点
            p2: Point 第二个点

        Returns:
            out: float 两个点之间的距离
        """
        a = np.array([p1.x, p1.y, p1.z])
        b = np.array([p2.x, p2.y, p2.z])
        return np.linalg.norm(a - b)
    
    def target_detection_callback(self, msg: TargetDetection):
        """处理目标检测消息，更新门的位置信息（NED: 北、东、地）"""
        target_name = f"{self.door_color}_door"  # 根据颜色构造目标名称
        if msg.class_name == target_name and msg.conf > 0.8:
            try:
                # 将目标点从camera坐标系转换到map坐标系
                self.tf_listener.waitForTransform("map", msg.pose.header.frame_id, msg.pose.header.stamp, rospy.Duration(1.0))
                # 转换目标点到map坐标系，类型：PoseStamped
                target_in_map = self.tf_listener.transformPose("map", msg.pose)
                
                # 如果队列未满，直接添加
                if not self.door_queue.full():
                    priority = -msg.conf  # 使用负的置信度作为优先级
                    self.door_queue.put((priority, target_in_map))
                    self.door_count += 1
                    rospy.loginfo(f"task1 node: 新门位置 #{self.door_count}: 置信度={msg.conf:.2f}, "
                                f"位置=({target_in_map.pose.position.x:.3f}, "
                                f"{target_in_map.pose.position.y:.3f}, "
                                f"{target_in_map.pose.position.z:.3f})")
                else:
                    # 如果队列已满，比较置信度
                    lowest_priority, _ = self.door_queue.get()
                    if -msg.conf < lowest_priority:  # 新目标置信度更高
                        self.door_queue.put((-msg.conf, target_in_map))
                        rospy.loginfo(f"task1 node: 更新门位置: 新置信度={msg.conf:.2f}, "
                                    f"位置=({target_in_map.pose.position.x:.3f}, "
                                    f"{target_in_map.pose.position.y:.3f}, "
                                    f"{target_in_map.pose.position.z:.3f})")
                    else:
                        self.door_queue.put((lowest_priority, _))  # 放回原来的目标
                        
            except tf.Exception as e:
                rospy.logwarn(f"task1 node: 坐标转换失败: {e}")

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

    def move_to_target(self,max_dist=0.2,max_yaw_step=np.radians(5),max_yaw_dist =0.2):
        """
        发送一次指令移动到目标位姿，通过生成平滑路径点实现
        
        Returns:
            到达目标位置返回true, 未到达目标位置返回false
        """
        try:
            # 获取当前位姿
            current_pose = self.get_current_pose()
            if current_pose is None:
                return False

            # 判断是否到达
            if self.is_arrival(current_pose, self.target_posestamped, max_dist, max_yaw_dist):
                return True
            
            # 航向控制和点控制统一起来
            next_pose = self.generate_smooth_pose(current_pose, self.target_posestamped)
            dist_to_target = self.numpy_distance(current_pose.pose.position, self.target_posestamped.pose.position)
            yaw_to_target = self.yaw_distance(current_pose.pose.orientation, self.target_posestamped.pose.orientation)
            rospy.loginfo(f"task1 node: 移动到目标点: 距离={dist_to_target:.3f}米, 航向差={np.degrees(yaw_to_target):.2f}度")
            self.target_pub.publish(next_pose)

            return False
            
        except tf.Exception as e:
            rospy.logwarn(f"task1 node: 移动失败: {e}")
            return False

    def move_down_to_point1(self):
        """
        发送一次指令运动到初始位置

        Returns:
            到达目标位置返回true,未到达目标位置返回false
        """        
        self.target_posestamped = self.start_point # 将宏定义的初始位置赋值给目标位置
        return self.move_to_target()

    def move_to_red_door(self,max_dist):
        """
        发送一次指令运动到红色门

        Returns:
            到达目标位置返回true,未到达目标位置返回false
        """        
        self.target_posestamped = self.point_reddoor # 将宏定义的初始位置赋值给目标位置
        return self.move_to_target(max_dist)
    
    def move_to_blue_door(self,max_dist):
        """
        发送一次指令运动到蓝色门

        Returns:
            到达目标位置返回true,未到达目标位置返回false
        """        
        self.target_posestamped = self.point_bluedoor # 将宏定义的初始位置赋值给目标位置
        return self.move_to_target(max_dist)

    def finish_task(self):
        """
        任务完成：
        1. 发布任务完成标志
        2. 返回True
        """
        self.finished_pub.publish("task1 finished")
        rospy.loginfo("task1 node: 任务完成，发布完成消息")
        rospy.signal_shutdown("任务完成")
        return True

    def forward(self, distance):
        """
        向前移动距离

        Parameters:
            distance: float, 移动距离(米)
        """
        # 获取当前AUV位姿
        try:
            current_pose = self.get_current_pose()
            
            # 计算当前yaw角
            _, current_pitch, yaw = euler_from_quaternion([
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z,
                current_pose.pose.orientation.w
            ])

            # 计算前进后的新位置
            new_x = current_pose.pose.position.x + np.cos(yaw) * distance
            new_y = current_pose.pose.position.y + np.sin(yaw) * distance
            new_z = current_pose.pose.position.z  # 深度不变
            
            # 构造目标点
            self.target_posestamped.header.frame_id = "map"
            self.target_posestamped.pose.position = Point(x=new_x, y=new_y, z=new_z)
            self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, current_pitch, yaw))
            rospy.loginfo(f"task1 node: 计算前进{distance}米后的目标点: ({new_x:.3f}, {new_y:.3f}, {new_z:.3f}), yaw={np.degrees(yaw):.2f}")
        except tf.Exception as e:
            rospy.logwarn(f"task1 node: 获取base_link位姿失败: {e}")

    def get_current_pose(self):
        """获取当前位姿"""
        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0)) # 返回一个三元组和一个四元组
            current_pose = PoseStamped()
            current_pose.header.frame_id = "map"
            current_pose.header.stamp = rospy.Time.now()
            current_pose.pose.position = Point(*trans)
            current_pose.pose.orientation = Quaternion(*rot)
            return current_pose
        except tf.Exception as e:
            rospy.logwarn(f"task1 node: 获取当前位姿失败: {e}")
            return None

    def get_init_pos(self) -> None:
        """获取初始位姿，增加深度赋值给目标位姿"""
        try:
            current_pose = self.get_current_pose()
        except tf.Exception as e:
            rospy.logwarn(f"task1 node: 获取base_link位姿失败: {e}")

        # 赋值给target_pose
        self.target_posestamped.header.frame_id = "map"
        self.target_posestamped.pose.position = current_pose.pose.position  # 使用当前位姿作为初始位置
        self.target_posestamped.pose.position.z = self.init_depth  # 设置目标深度
        self.target_posestamped.pose.orientation = current_pose.pose.orientation  # 使用当前位姿作为初始姿态
        rospy.loginfo(f"task1 node: 初始位姿均值: 位置=({current_pose.pose.position.x:.3f}, {current_pose.pose.position.y:.3f}, {current_pose.pose.position.z:.3f}), "
                      f"四元数=({current_pose.pose.orientation.x:.3f}, {current_pose.pose.orientation.y:.3f}, {current_pose.pose.orientation.z:.3f}, {current_pose.pose.orientation.w:.3f})")

    def rotate360(self) -> bool:
        """
        原地旋转一圈
        根据门的颜色确定旋转方向：红色顺时针，蓝色逆时针
        通过实际旋转角度判断是否完成一圈
        统一使用move_to_target方法来发布目标姿态
        """
        # 获取当前yaw角
        current_pose = self.get_current_pose()
        if current_pose is None:
            return False

        _, _, current_yaw = euler_from_quaternion(
            [current_pose.pose.orientation.x,
             current_pose.pose.orientation.y,
             current_pose.pose.orientation.z,
             current_pose.pose.orientation.w]
             )

        # 初始化旋转相关变量
        if not hasattr(self, 'rotate_start_yaw'):
            self.rotate_start_yaw = current_yaw
            # self.rotate_target_yaw = current_yaw + (step_rad * self.rotation_direction)
            self.rotate_target_yaw = current_yaw + (self.step_rad * self.direction)
            self.last_yaw = current_yaw
            self.total_rotated = 0.0
        
        # 计算实际旋转角度
        # 处理角度跳变（从-π到π或从π到-π的情况）
        delta_yaw = current_yaw - self.last_yaw
        if self.direction > 0:  # 逆时针旋转
            if delta_yaw < -np.pi:  # 从-π跳到π
                delta_yaw += 2 * np.pi
            elif delta_yaw > np.pi:  # 从π跳到-π
                delta_yaw -= 2 * np.pi
        else:  # 顺时针旋转
            if delta_yaw < -np.pi:  # 从-π跳到π
                delta_yaw += 2 * np.pi
            elif delta_yaw > np.pi:  # 从π跳到-π
                delta_yaw -= 2 * np.pi

        
        self.total_rotated += abs(delta_yaw)
        rospy.loginfo(self.total_rotated)
        self.last_yaw = current_yaw
        
        # 设置新的目标姿态
        self.rotate_target_yaw = current_yaw + (self.step_rad * self.direction)
        self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, self.rotate_target_yaw))

        # 发布新姿态
        self.move_to_target(max_dist=0.1, max_yaw_step=self.step_rad, max_yaw_dist=self.step_rad)
        if current_yaw is not None:
            rospy.loginfo_throttle(1, f"目标航向: {np.degrees(self.rotate_target_yaw)}度, 当前航向: {np.degrees(current_yaw)}度,delta yaw {delta_yaw}")
        else:
            rospy.loginfo_throttle(1, f"目标航向: {np.degrees(self.rotate_target_yaw)}度")

        # 记录旋转进度
        progress = (self.total_rotated / (2 * np.pi)) * 100
        rospy.loginfo_throttle(1, f"task1 node: 旋转进度: {progress:.1f}%")

        # 判断是否完成一圈旋转（360度）
        if self.total_rotated >= 2 * np.pi:
            # if current_yaw - self.rotate_start_yaw
            rospy.loginfo(f"task1 node: 完成360度旋转 {self.total_rotated},{np.pi}")
            # 清理旋转相关变量
            del self.rotate_start_yaw
            del self.rotate_target_yaw
            del self.last_yaw
            del self.total_rotated
            return True

        return False
       
    # 第零步，原地获取5帧AUV数据，保存NED并计算一个均值
    # 第一步，运行到左门或右门的位置
    # 第二步，经过门后
    # 第四步，原地旋转一周
    # 第五步，前进1米
    # 第六步，结束任务.           
    def run(self):
        while not rospy.is_shutdown():
            if self.step == 0:
                self.get_init_pos()  # 获取初始位姿
                self.step = 1 # 进入步骤1
                rospy.loginfo("task1 node: 已获取初始位姿, 进入步骤1")
            elif self.step == 1: # 第一步，下潜到指定深度，由宏定义确定，位置和姿态保持不变
                if self.move_to_target():
                    self.step = 2 # 进入步骤2
                    rospy.loginfo("task1 node: 已到达下潜位置, 进入步骤2")
            elif self.step == 2: # 第二步，移动到门前位置
                if self.move_down_to_point1():
                    self.step = 3 # 进入步骤3
                    rospy.loginfo("task1 node: 已到达门前位置, 进入步骤3")
            elif self.step == 3: # 进入到对应的门中
                if self.door_color == 'blue':
                    if self.move_to_blue_door(0.5):
                        self.step = 4  # 进入步骤4
                        rospy.loginfo("task1 node: 已到达蓝色门前位置, 进入步骤4")
                elif self.door_color == 'red':
                    if self.move_to_red_door(0.5):
                        self.step = 4  # 进入步骤4
                        rospy.loginfo("task1 node: 已到达红色门前位置, 进入步骤4")
                else:
                    rospy.logwarn("task1 node: 未识别的门颜色")
            elif self.step == 4:
                self.forward(1) # 计算前进1米后的目标点
                self.step = 5
                rospy.loginfo("task1 node: 已计算前进1米后的目标点, 进入步骤5")
            elif self.step == 5:
                if self.move_to_target(): # 运动到目标点
                    self.step = 6
                    rospy.loginfo("task1 node: 已到达目标点, 进入步骤6")
            elif self.step == 6:
                if self.rotate360(): # 原地旋转一圈
                    # 设置目标姿态为门的方向
                    if self.door_color == 'blue':
                        self.target_posestamped.pose.orientation = self.point_bluedoor.pose.orientation
                    else:
                        self.target_posestamped.pose.orientation = self.point_reddoor.pose.orientation
                    # 记录开始保持航向的时间
                    self.hold_heading_start_time = rospy.Time.now()
                    rospy.loginfo("task1 node: 已完成原地旋转, 进入步骤7(保持航向)")
                    self.step = 7
            elif self.step == 7:
                # 保持航向3秒
                if (rospy.Time.now() - self.hold_heading_start_time).to_sec() >= 3.0:
                    rospy.loginfo("task1 node: 已保持航向3秒, 进入步骤8(结束任务)")
                    self.step = 8
                else:
                    # 继续发送目标姿态
                    self.rotate_to_target()
            elif self.step == 8:
                self.finish_task()
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('task1_node')
    try:
        node = Task1Node()
        node.run()
    except rospy.ROSInterruptException:
        pass

