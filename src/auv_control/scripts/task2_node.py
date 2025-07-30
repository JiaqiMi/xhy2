#! /home/xhy/xhy_env36/bin/python
"""
名称: task2_node.py
功能：寻找目标释放作业
作者: buyegaid
监听：/target_detection (来自视觉节点) 检测目标是圆形的标志物, 移动目标是让T型管插入到圆形中
      /tf (来自tf树)
发布：/auv_control (Control.msg) 被sensor_driver订阅
      /finished (String) 被state_control订阅，代表任务是否完成
      /target (PoseStamped.msg) 

记录：
2025.7.20 20:34
    第一版完成
2025.7.23 00:45
    完整逻辑完成
"""

# target detect: red/green/black:对应小中大
# 宏定义一个要的位置
# 从参数列表读取到达第一个位姿
# step0: 运动到初始作业位姿，到达后跳到step1
# step1: 执行搜索，寻找目标，找到目标后跳到step2
# step2: 闭环移动到目标点，判断是否可以释放，可以释放后跳到step3
# step3: 释放后，运动到初始作业位姿，到达后跳到step4
# step4: 任务完成，发送任务完成标志，关闭节点

import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String
from auv_control.msg import TargetDetection, Control
from geometry_msgs.msg import PoseStamped, Quaternion, Point
import numpy as np
from queue import PriorityQueue
from threading import Lock

class Task2Node:
    """任务2节点: 到目标点释放钥匙"""
    def __init__(self):
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10) # 发布期望位姿话题                  
        self.control_pub = rospy.Publisher('/control', Control, queue_size=10) # 发布控制话题
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10) # 发布任务完成标志话题
        rospy.Subscriber('/obj/target_message', TargetDetection, self.target_detection_callback) # 订阅目标检测话题
        self.rate = rospy.Rate(5)  # 运行频率5Hz
        self.tf_listener = tf.TransformListener()
        self.step = 0 # 程序运行阶段

        self.target_posestamped = PoseStamped() # 期望位置消息定义
        
        # 添加优先队列用于存储目标位置
        self.target_positions_queue = PriorityQueue(maxsize=10)
        self.target_queue_count = 0  # 记录目标位置队列中的目标数量
        self.sequence_number = 0  # 用于优先队列中元素的唯一标识

        # 添加搜索相关的变量
        self.initial_yaw = None  # 初始yaw角度
        self.search_direction = 1  # 搜索方向：1表示正向，-1表示反向
        self.yaw_step = np.radians(3)  # 每次旋转3度
        self.search_phase = 1  # 搜索阶段：1为航向对齐，2为位置对齐
       # self.target_yaw = rospy.get_param('/task2_target_yaw', None)  # 从参数服务器读取目标航向
        self.target_yaw = None
        self.aligned_yaw = None  # 记录对齐后的航向
        self.best_confidence = 0.0  # 记录最佳置信度

        self.release_pub_num = 0  # 记录释放目标的次数

        # 获取宏定义参数
        self.target_depth = rospy.get_param('/depth', 0.3)  # 下潜深度，单位米

        # 获取检测目标
        self.target_color = rospy.get_param('/task2_target_class', 'red')  # 目标颜色，默认为红色
        # 添加线程锁

        self._queue_lock = Lock()
        # 存储一个初始目标点，先到这个位置，然后开始搜索
        self.start_point = PoseStamped()
        self.start_point.header.frame_id = "map" # 设置坐标系为map
        start_point_from_param = rospy.get_param('/task2_point1', [0.5, -0.5, 0.15, 0.0])  # 默认值
        self.start_point.pose.position.x = start_point_from_param[0]
        self.start_point.pose.position.y = start_point_from_param[1]
        self.start_point.pose.position.z = start_point_from_param[2] #TODO depth
        # rospy.loginfo(start_point_from_param)
        self.start_point.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.radians(start_point_from_param[3])))
        rospy.loginfo(f"task2 node: 初始位置设置为: {self.start_point.pose.position.x}, {self.start_point.pose.position.y}, {self.start_point.pose.position.z}, ")

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
            
    def numpy_distance(self,p1:Point, p2:Point):
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
        """目标检测回调函数，当检测到目标时，计算目标在世界坐标系下的位置, 并将其添加到优先队列中，不做其他处理"""
        rospy.loginfo(f"task2 node: get msg{msg.class_name}")
        with self._queue_lock: # 获取线程锁
            try:
                if msg.class_name == self.target_color and msg.conf > 0.6:  # 只有当检测到的目标颜色与宏定义的目标颜色一致时才处理
                        point_in_camera = msg.pose.pose.position
                        point_origin = Point(x=0,y=0,z=0)  # 相机坐标系下的原点
                        if self.numpy_distance(point_in_camera, point_origin) < 5.0:# 暂时先判断5米以内的
                            # 将目标点从camera坐标系转换到map坐标系
                            self.tf_listener.waitForTransform("map", msg.pose.header.frame_id, msg.pose.header.stamp, rospy.Duration(1.0))
                            target_in_map = self.tf_listener.transformPose("map", msg.pose)
                            target_in_auv = self.tf_listener.transformPose("base_link", msg.pose)  # 转换到base_link坐标系下

                            # 如果队列未满，则添加新的目标位置
                            if not self.target_positions_queue.full():
                                # 使用负的置信度作为优先级（越大的置信度优先级越高）
                                # 使用元组(priority, sequence_number, target)确保唯一性
                                priority = - msg.conf
                                self.sequence_number += 1
                                self.target_positions_queue.put((priority, self.sequence_number, target_in_map, target_in_auv))
                                self.target_queue_count += 1
                                rospy.loginfo(f"task2 node: 添加新的目标位置，置信度: {msg.conf}, 当前目标数量: {self.target_queue_count}")
                            else:
                                # 如果队列已满，且新目标的置信度高于队列中最低的置信度，则替换
                                lowest_priority, seq, map_pose, auv_pose = self.target_positions_queue.get()
                                if -msg.conf < lowest_priority:  # 记住priority是负的置信度
                                    self.target_positions_queue.put((lowest_priority, seq, map_pose, auv_pose))  # 把原来的放回去
                                else:
                                    self.sequence_number += 1
                                    self.target_positions_queue.put((-msg.conf, self.sequence_number, target_in_map, target_in_auv))  # 放入新的
                                    rospy.loginfo(f"task2 node: 更新目标位置，新置信度: {msg.conf}")                    
            except tf.Exception as e:
                rospy.logwarn(f"task2 node: 坐标转换失败: {e}")
                return

    def move_to_init_pose(self):
        """
        发送一次指令移动到初始位姿

        Returns:
            到达目标位置返回true,未到达目标位置返回false
        """
        self.target_posestamped = self.start_point # 将宏定义的初始位置赋值给目标位置
        return self.move_to_target()

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

    def move_to_target(self,max_dist=0.2,max_yaw_step=np.radians(5),max_yaw_dist =np.radians(0.2)):
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
            # rospy.loginfo(f"task1 node: 生成平滑路径点: 当前位姿: {current_pose.pose.position},  target:{self.target_posestamped.pose.position}, ")
            rospy.loginfo(f"task1 node: 移动到目标点: 距离={dist_to_target:.3f}米, 航向差={np.degrees(yaw_to_target):.2f}度")
            self.target_pub.publish(next_pose)

            return False
            
        except tf.Exception as e:
            rospy.logwarn(f"task1 node: 移动失败: {e}")
            return False
    
    def move_to_release_target(self):
        """
        移动到工作目标点：
        1. 从优先队列中获取所有目标位置
        2. 根据置信度计算加权平均位置
        3. 计算base_link的目标位置，使hand与target重合
        4. 设置target_pose并移动到目标位置

        Returns:
            是否到达目标位置
        """
        if not self.target_positions_queue.full():
            rospy.logwarn("task2 node: 没有可用的目标位置")
            return False
            
        # 获取目标位置的加权平均值
        positions = []
        weights = []
        temp_queue = PriorityQueue()
        
        while not self.target_positions_queue.empty():
            priority, seq, target = self.target_positions_queue.get()
            positions.append([target.pose.position.x, target.pose.position.y, target.pose.position.z])
            weight = -priority
            weights.append(weight)
            temp_queue.put((priority, seq, target))
            
        self.target_positions_queue = temp_queue
        
        positions = np.array(positions)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        avg_position = np.average(positions, weights=weights, axis=0)
        
        try:
            # 获取hand到base_link的静态变换
            self.tf_listener.waitForTransform("base_link", "hand", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("base_link", "hand", rospy.Time(0))
            
            # 创建目标位置的PoseStamped
            target_pose = PoseStamped()
            target_pose.header.frame_id = "map"
            target_pose.header.stamp = rospy.Time.now()
            target_pose.pose.position.x = avg_position[0]
            target_pose.pose.position.y = avg_position[1]
            target_pose.pose.position.z = avg_position[2]
            
            # 保持当前姿态
            transform = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            target_pose.pose.orientation = transform[1]
            
            # 获取当前姿态
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            (base_trans, base_rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            
            # 计算base_link的目标位置
            self.target_posestamped.header.frame_id = "map"
            self.target_posestamped.header.stamp = rospy.Time.now()
            self.target_posestamped.pose.position.x = avg_position[0] - trans[0]
            self.target_posestamped.pose.position.y = avg_position[1] - trans[1]
            self.target_posestamped.pose.position.z = avg_position[2] - trans[2]
            self.target_posestamped.pose.orientation.x = base_rot[0]
            self.target_posestamped.pose.orientation.y = base_rot[1]
            self.target_posestamped.pose.orientation.z = base_rot[2]
            self.target_posestamped.pose.orientation.w = base_rot[3]
            
            rospy.loginfo(f"计算得到的工作目标点位置: x={self.target_posestamped.pose.position.x:.3f}, "
                         f"y={self.target_posestamped.pose.position.y:.3f}, "
                         f"z={self.target_posestamped.pose.position.z:.3f}")
            
            return self.move_to_target()
            
        except tf.Exception as e:
            rospy.logwarn(f"task2 node: 获取变换失败: {e}")
            return False

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

    def search_target(self):
        """
        搜索目标的两个阶段：
        阶段1：对齐航向，将航向垂直于物体
        阶段2：对齐位置，通过识别目标位置进行微调
        
        Returns:
            是否找到目标并完成对齐
        """
        try:
            # 获取当前位姿
            current_pose = self.get_current_pose()
            if current_pose is None:
                return False

            _, _, current_yaw = euler_from_quaternion([
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z,
                current_pose.pose.orientation.w
            ])

            # 阶段1：航向对齐
            if self.search_phase == 1:
                # 如果参数服务器中有目标航向，直接使用
                if self.target_yaw is not None:
                    target_yaw = np.radians(self.target_yaw)
                    self.aligned_yaw = target_yaw
                    self.search_phase = 2
                    rospy.loginfo(f"使用参数服务器中的目标航向: {self.target_yaw}度")
                    return False

                # 初始化搜索参数
                if self.initial_yaw is None:
                    self.initial_yaw = current_yaw
                    rospy.loginfo(f"搜索初始角度: {np.degrees(self.initial_yaw)}度")

                # 检查是否找到高置信度的目标
                if self.target_positions_queue.qsize() > 0:
                    # 遍历队列找到最高置信度
                    temp_queue = PriorityQueue()
                    max_conf = 0.0
                    best_yaw = current_yaw
                    with self._queue_lock:  # 获取线程锁
                        while not self.target_positions_queue.empty():
                            priority, seq, map_pose, auv_pose = self.target_positions_queue.get()
                            conf = -priority  # 转换回置信度
                            if conf > max_conf:
                                max_conf = conf
                                # 计算目标相对于AUV的方位
                                target_dir = np.arctan2(
                                    auv_pose.pose.position.y,
                                    auv_pose.pose.position.x
                                )
                                # 计算垂直于目标的航向
                                # best_yaw = current_yaw + target_dir + np.pi/2
                                best_yaw = current_yaw + target_dir
                            temp_queue.put((priority, seq, map_pose, auv_pose))
                        
                        self.target_positions_queue = temp_queue
                        rospy.loginfo(f"best_raw: {best_raw}")

                    # 如果找到了足够高置信度的目标
                    if max_conf > self.best_confidence:
                        self.best_confidence = max_conf
                        self.aligned_yaw = best_yaw
                        rospy.loginfo(f"找到更好的目标方向，置信度: {max_conf:.2f}")

                    # 如果置信度足够高，进入第二阶段
                    if max_conf > 0.7:
                        self.search_phase = 2
                        rospy.loginfo("航向对齐完成，进入位置对齐阶段")
                        return False

                # 继续搜索
                next_yaw = current_yaw + (self.yaw_step * self.search_direction)

                # 检查是否需要改变搜索方向
                if next_yaw > self.initial_yaw + np.radians(30):
                    self.search_direction = -1
                    next_yaw = current_yaw + (self.yaw_step * self.search_direction)
                    rospy.loginfo("掉头顺时针搜索")
                elif next_yaw < self.initial_yaw - np.radians(30):
                    self.search_direction = 1
                    next_yaw = current_yaw + (self.yaw_step * self.search_direction)
                    rospy.loginfo("掉头逆时针搜索")

            # 阶段2：位置对齐
            else:
                if self.target_positions_queue.qsize()>=3:
                  

                    # 计算目标在map下的平均位置
                    temp_queue = PriorityQueue()
                    sum_pos = np.zeros(3)
                    sum_weight = 0.0
                    
                    while not self.target_positions_queue.empty():
                        priority, seq, map_pose, auv_pose = self.target_positions_queue.get()
                        weight = -priority  # 转换回置信度
                        pos = np.array([
                            map_pose.pose.position.x,
                            map_pose.pose.position.y,
                            map_pose.pose.position.z
                        ])
                        sum_pos += pos * weight
                        sum_weight += weight
                        temp_queue.put((priority, seq, map_pose, auv_pose))
                    
                    self.target_positions_queue = temp_queue
                    avg_pos = sum_pos / sum_weight

                    try:
                        # 获取hand到base_link的静态变换
                        self.tf_listener.waitForTransform("base_link", "hand", rospy.Time(0), rospy.Duration(1.0))
                        (hand_trans, hand_rot) = self.tf_listener.lookupTransform("base_link", "hand", rospy.Time(0))
                        
                        # 计算当前位置与目标的偏差，考虑夹爪偏移
                        current_pos = np.array([
                            current_pose.pose.position.x + hand_trans[0],  # 加上夹爪在x方向的偏移
                            current_pose.pose.position.y + hand_trans[1],  # 加上夹爪在y方向的偏移
                            current_pose.pose.position.z + hand_trans[2]   # 加上夹爪在z方向的偏移
                        ])
                        pos_error = np.linalg.norm(avg_pos - current_pos)
                        
                        # 如果位置和航向都对齐得比较好，完成搜索
                        yaw_error = abs((self.aligned_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi)
                        if pos_error < 0.1 and yaw_error < np.radians(5):
                            rospy.loginfo("位置和航向对齐完成")
                            return True
                        # 设置目标姿态，考虑夹爪偏移
                        self.target_posestamped.header.frame_id = "map"
                        self.target_posestamped.header.stamp = rospy.Time.now()
                        
                        # 设置目标位置，让hand对准目标
                        self.target_posestamped.pose.position.x = avg_pos[0] - hand_trans[0]
                        self.target_posestamped.pose.position.y = avg_pos[1] - hand_trans[1]
                        self.target_posestamped.pose.position.z = avg_pos[2] - hand_trans[2]
                    
                        # 设置目标航向
                        # self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, self.aligned_yaw))
                        
                        rospy.loginfo_throttle(1, f"位置对齐: 误差={pos_error:.3f}米, "
                                            f"航向误差={np.degrees(yaw_error):.1f}度")
                    except tf.Exception as e:
                        rospy.logwarn(f"task2 node: 获取hand变换失败: {e}")
                        return False
                else
                      rospy.loginfo_throttle(1, "等待足够的目标位置数据...")
            # 设置目标姿态
            # 设置目标航向
            target_yaw = self.aligned_yaw if self.search_phase == 2 else next_yaw
            # rospy.loginfo(target_yaw)
            self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, 1, target_yaw))
            self.move_to_target()  # 发布目标位置
            rospy.loginfo_throttle(1, f"搜索阶段{self.search_phase}: 当前航向={np.degrees(current_yaw):.1f}度, "
                                    f"目标航向={np.degrees(target_yaw):.1f}度")           
            return False
        except tf.Exception as e:
            rospy.logwarn(f"task2 node: 获取当前位姿失败: {e}")
            return False


    def release_target(self):
        """
        释放目标：
        1. 先发布5次张开夹爪指令
        2. 再发布5次关闭夹爪指令
        3. 返回True
        """
        # 前5次张开夹爪
        if self.release_pub_num < 5:
            control_msg = Control()
            control_msg.led_green = 0
            control_msg.led_red = 0
            control_msg.servo = 100  # 张开夹爪
            self.control_pub.publish(control_msg)
            self.move_to_target()  # 也需要按时发布位姿控制
            self.release_pub_num += 1
            return False
        # 后5次关闭夹爪
        elif self.release_pub_num < 10:
            control_msg = Control()
            control_msg.led_green = 0
            control_msg.led_red = 0
            control_msg.servo = 0  # 关闭夹爪
            self.control_pub.publish(control_msg)
            self.move_to_target()
            self.release_pub_num += 1
            return False
        else:
            return True

    def finish_task(self):
        """
        任务完成：
        1. 发布任务完成标志
        2. 返回True
        """
        self.finished_pub.publish("task2 finished")
        rospy.loginfo("task2 node: 任务完成，发布完成消息")
        rospy.signal_shutdown("任务完成")
        return True

    def run(self):
        while not rospy.is_shutdown():
            if self.step == 0:
                if self.move_to_init_pose(): #如果移动到了目标位置，则跳到step1
                    self.step = 1
            elif self.step == 1:
                if self.search_target(): # 记录到足够的目标点位置后，跳到step2
                    self.step = 2
            elif self.step == 2:
                if self.move_to_release_target(): # 如果移动到了工作目标位置，则跳到step3
                    self.step = 3
            elif self.step == 3:
                if self.release_target(): # 释放目标后，移动到初始位置
                    self.step = 4
            elif self.step == 4:
                if self.move_to_init_pose():
                    self.step = 5
            elif self.step == 5:
                self.finish_task()
                break
            self.rate.sleep()
            

if __name__ == '__main__':
    rospy.init_node('task2_node')
    try:
        node = Task2Node()
        node.run()
    except rospy.ROSInterruptException:
        pass