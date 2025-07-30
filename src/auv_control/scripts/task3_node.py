#! /home/xhy/xhy_env36/bin/python
"""
名称: task3_node.py
功能: 夹取小球上浮
作者: buyegaid
监听：/target_detection (来自视觉节点) 检测目标对应颜色的高尔夫球
      /tf (来自tf树)
发布：/auv_control (Control.msg) 被sensor_driver订阅
      /finished (String) 被state_control订阅, 代表任务是否完成
      /target (PoseStamped.msg) 被tf_handler订阅, 代表目标位置

记录：
"""
import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String
from auv_control.msg import TargetDetection, Control # control是控制led和舵机
from geometry_msgs.msg import PoseStamped, Quaternion,Point
import numpy as np
from queue import PriorityQueue
from threading import Lock

# step0: 运动到初始作业位置附近，到达后跳转到step1
# step1: 原地寻找小球,原地旋转, 当优先队列中积攒足够多的有效目标时, 跳转到step2
# step2: 闭环不断移动到小球，每次可以只前进一点距离，最后判断夹爪坐标系和小球重合, 跳转到step3
# step3: 夹取小球，随后跳转到step4
# step4: 运动到上浮的目标位置，跳转到step5
# step5: 上浮, 跳转到step6
# step6: 完成任务，发布任务完成标志，关闭节点

class Task3Node:
    def __init__(self):
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10)
        self.control_pub = rospy.Publisher('/control', Control, queue_size=10)
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10)
        rospy.Subscriber('/target_detection', TargetDetection, self.target_detection_callback)
        self.rate = rospy.Rate(5)  # 5Hz
        self.tf_listener = tf.TransformListener()
        
        # 获取宏定义参数
        self.target_depth = rospy.get_param('~depth', 0.3)  # 下潜深度，单位米
        
        # 存储一个初始目标点，先到这个位置，然后开始搜索
        self.start_point = PoseStamped()
        self.start_point.header.frame_id = "map"
        start_point_from_param = rospy.get_param('/task3_point0', [0.5, -0.5, 0.15, 0.0])  # 默认值
        self.start_point.pose.position.x = start_point_from_param[0]
        self.start_point.pose.position.y = start_point_from_param[1]
        self.start_point.pose.position.z = start_point_from_param[2]  
        self.start_point.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.radians(start_point_from_param[3])))
            
        # 变量定义
        self.step = 0
        self.target_pose = PoseStamped()
        self.target_positions = PriorityQueue(maxsize=10)
        self.target_count = 0  # 记录收到的有效目标数量
        self.sequence_number = 0  # 用于优先队列中元素的唯一标识
        self.grab_count = 0  # 记录抓取动作的次数
        
        # 搜索相关变量
        self.initial_yaw = None  # 初始yaw角度
        self.search_direction = 1  # 搜索方向：1表示正向，-1表示反向
        self.yaw_step = np.radians(3)  # 每次旋转3度
        self.search_phase = 1  # 搜索阶段：1为航向对齐，2为位置对齐
        self.target_yaw = rospy.get_param('/task3_target_yaw', None)  # 从参数服务器读取目标航向
        self.aligned_yaw = None  # 记录对齐后的航向
        self.best_confidence = 0.0  # 记录最佳置信度
        
        # 线程锁
        self._queue_lock = Lock()
        
        # 从参数服务器获取目标颜色
        self.target_color = rospy.get_param('/task3_target_color', 'red')  # 目标颜色，默认红色

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

    def get_current_pose(self):
        """获取当前位姿"""
        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
            current_pose = PoseStamped()
            current_pose.header.frame_id = "map"
            current_pose.pose.position.x = trans[0]
            current_pose.pose.position.y = trans[1]
            current_pose.pose.position.z = trans[2]
            current_pose.pose.orientation.x = rot[0]
            current_pose.pose.orientation.y = rot[1]
            current_pose.pose.orientation.z = rot[2]
            current_pose.pose.orientation.w = rot[3]
            return current_pose
        except tf.Exception as e:
            rospy.logwarn(f"task3 node: 获取当前位姿失败: {e}")
            return None

    def generate_smooth_pose(self, current_pose:PoseStamped, target_pose:PoseStamped, max_xy_step=1.0, max_z_step=0.1, max_yaw_step=np.radians(10)):
        """
        使用三阶贝塞尔曲线生成平滑的路径点
        
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
        
        # 获取当前和目标的yaw角
        _, _, current_yaw = tf.transformations.euler_from_quaternion([
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ])
        _, _, target_yaw = tf.transformations.euler_from_quaternion([
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w
        ])
        
        # 计算起点和终点
        p0 = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z])
        p3 = np.array([target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z])
        
        # 计算控制点（根据当前和目标姿态）
        control_dist = np.linalg.norm(p3 - p0) * 0.4
        
        # 根据当前姿态计算第一个控制点
        p1 = p0 + control_dist * np.array([
            np.cos(current_yaw),
            np.sin(current_yaw),
            0
        ])
        
        # 根据目标姿态计算第二个控制点
        p2 = p3 - control_dist * np.array([
            np.cos(target_yaw),
            np.sin(target_yaw),
            0
        ])
        
        # 如果没有存储当前的贝塞尔曲线参数t值，初始化为0
        if not hasattr(self, 'bezier_t'):
            self.bezier_t = 0.0
        
        # 计算下一个t值
        dt = 0.1
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
        
        # 计算yaw角差异（处理角度环绕）
        dyaw = target_yaw - current_yaw
        dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
        dyaw = np.clip(dyaw, -max_yaw_step, max_yaw_step)
        
        # 设置下一个姿态
        next_yaw = current_yaw + dyaw
        next_pose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, next_yaw))
        
        # 如果到达目标点，重置贝塞尔曲线参数
        if np.linalg.norm(p3 - p0) < 0.05:
            if hasattr(self, 'bezier_t'):
                del self.bezier_t
        
        return next_pose

    def move_to_target(self):
        """
        发送一次指令移动到目标位姿，通过生成平滑路径点实现
        
        Returns:
            到达目标位置返回true，未到达目标位置返回false
        """
        try:
            # 获取当前位姿
            current_pose = self.get_current_pose()
            if current_pose is None:
                return False
            
            # 如果已经到达目标点，返回True
            if self.numpy_distance(current_pose.pose.position, self.target_pose.pose.position) < 0.1:
                return True
                
            # 生成下一个平滑位姿点
            next_pose = self.generate_smooth_pose(current_pose, self.target_pose)
            
            # 发布下一个位姿点
            self.target_pub.publish(next_pose)
            rospy.loginfo_throttle(1, f"task3 node: Moving to target, current distance: "
                               f"{self.numpy_distance(current_pose.pose.position, self.target_pose.pose.position):.2f}m")
            
            return False
            
        except tf.Exception as e:
            rospy.logwarn(f"task3 node: 获取当前位姿失败: {e}")
            return False

    def rotate_to_target(self) -> bool:
        """
        原地执行旋转（只发送一次指令），通过是否到目标角度返回真假
        
        Returns:
            bool: 如果到达目标角度返回True，否则返回False
        """
        try:
            # 获取当前位姿
            current_pose = self.get_current_pose()
            if current_pose is None:
                return False
            
            # 获取当前和目标的yaw角
            _, _, current_yaw = tf.transformations.euler_from_quaternion([
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z,
                current_pose.pose.orientation.w
            ])
            _, _, target_yaw = tf.transformations.euler_from_quaternion([
                self.target_pose.pose.orientation.x,
                self.target_pose.pose.orientation.y,
                self.target_pose.pose.orientation.z,
                self.target_pose.pose.orientation.w
            ])
            
            # 计算yaw角差异（处理角度环绕）
            dyaw = target_yaw - current_yaw
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            
            # 如果角度差小于阈值，认为已到达目标角度
            if abs(dyaw) < np.radians(0.2):  # 0.2度的误差范围
                return True
                
            # 生成下一个平滑位姿点
            next_pose = self.generate_smooth_pose(current_pose, self.target_pose)
            rospy.loginfo(f"task3 node: target yaw {np.degrees(target_yaw)}, current yaw {np.degrees(current_yaw)}")
            # 发布下一个位姿点
            self.target_pub.publish(next_pose)
            
            return False
            
        except tf.Exception as e:
            rospy.logwarn(f"task3 node: 获取当前位姿失败: {e}")
            return False

    def target_detection_callback(self, msg: TargetDetection):
        """目标检测回调函数，当检测到目标时，计算目标在世界坐标系下的位置, 并将其添加到优先队列中"""
        with self._queue_lock:  # 获取线程锁
            if msg.class_name == self.target_color and msg.conf > 0.6:  # 只有当检测到的目标颜色与设定的目标颜色一致时才处理
                try:
                    point_in_camera = msg.pose.pose.position
                    point_origin = Point(x=0,y=0,z=0)  # 相机坐标系下的原点
                    if self.numpy_distance(point_in_camera, point_origin) < 5.0:  # 暂时先判断5米以内的
                        # 将目标点从camera坐标系转换到map坐标系
                        self.tf_listener.waitForTransform("map", msg.pose.header.frame_id, msg.pose.header.stamp, rospy.Duration(1.0))
                        target_in_map = self.tf_listener.transformPose("map", msg.pose)
                        target_in_auv = self.tf_listener.transformPose("base_link", msg.pose)  # 转换到base_link坐标系下

                        # 如果队列未满，则添加新的目标位置
                        if not self.target_positions.full():
                            priority = -msg.conf
                            self.sequence_number += 1
                            self.target_positions.put((priority, self.sequence_number, target_in_map, target_in_auv))
                            self.target_count += 1
                            rospy.loginfo(f"task3 node: 添加新的目标位置，置信度: {msg.conf}, 当前目标数量: {self.target_count}")
                        else:
                            lowest_priority, seq, map_pose, auv_pose = self.target_positions.get()
                            if -msg.conf < lowest_priority:  # 记住priority是负的置信度
                                self.target_positions.put((lowest_priority, seq, map_pose, auv_pose))  # 把原来的放回去
                            else:
                                self.sequence_number += 1
                                self.target_positions.put((-msg.conf, self.sequence_number, target_in_map, target_in_auv))  # 放入新的
                                rospy.loginfo(f"task3 node: 更新目标位置，新置信度: {msg.conf}")
                            
                except tf.Exception as e:
                    rospy.logwarn(f"task3 node: 坐标转换失败: {e}")
                    return

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
                if self.target_positions.qsize() > 0:
                    # 遍历队列找到最高置信度
                    temp_queue = PriorityQueue()
                    max_conf = 0.0
                    best_yaw = current_yaw
                    with self._queue_lock:  # 获取线程锁
                        while not self.target_positions.empty():
                            priority, seq, map_pose, auv_pose = self.target_positions.get()
                            conf = -priority  # 转换回置信度
                            if conf > max_conf:
                                max_conf = conf
                                # 计算目标相对于AUV的方位
                                target_dir = np.arctan2(
                                    auv_pose.pose.position.y,
                                    auv_pose.pose.position.x
                                )
                                # 计算垂直于目标的航向
                                best_yaw = current_yaw + target_dir + np.pi/2
                            temp_queue.put((priority, seq, map_pose, auv_pose))
                        
                        self.target_positions = temp_queue

                    # 如果找到了足够高置信度的目标
                    if max_conf > self.best_confidence:
                        self.best_confidence = max_conf
                        self.aligned_yaw = best_yaw
                        rospy.loginfo(f"找到更好的目标方向，置信度: {max_conf:.2f}")

                    # 如果置信度足够高，进入第二阶段
                    if max_conf > 0.8:
                        self.search_phase = 2
                        rospy.loginfo("航向对齐完成，进入位置对齐阶段")
                        return False

                # 继续搜索
                next_yaw = current_yaw + (self.yaw_step * self.search_direction)

                # 检查是否需要改变搜索方向
                if next_yaw > self.initial_yaw + np.radians(180):
                    self.search_direction = -1
                    next_yaw = current_yaw + (self.yaw_step * self.search_direction)
                    rospy.loginfo("掉头顺时针搜索")
                elif next_yaw < self.initial_yaw - np.radians(180):
                    self.search_direction = 1
                    next_yaw = current_yaw + (self.yaw_step * self.search_direction)
                    rospy.loginfo("掉头逆时针搜索")

            # 阶段2：位置对齐
            else:
                if not self.target_positions.full():
                    rospy.loginfo_throttle(1, "等待足够的目标位置数据...")
                    return False

                # 计算目标在map下的平均位置
                temp_queue = PriorityQueue()
                sum_pos = np.zeros(3)
                sum_weight = 0.0
                
                while not self.target_positions.empty():
                    priority, seq, map_pose, auv_pose = self.target_positions.get()
                    weight = -priority  # 转换回置信度
                    pos = np.array([
                        map_pose.pose.position.x,
                        map_pose.pose.position.y,
                        map_pose.pose.position.z
                    ])
                    sum_pos += pos * weight
                    sum_weight += weight
                    temp_queue.put((priority, seq, map_pose, auv_pose))
                
                self.target_positions = temp_queue
                avg_pos = sum_pos / sum_weight

                try:
                    # 获取hand到base_link的静态变换
                    self.tf_listener.waitForTransform("base_link", "hand", rospy.Time(0), rospy.Duration(1.0))
                    (hand_trans, hand_rot) = self.tf_listener.lookupTransform("base_link", "hand", rospy.Time(0))
                    
                    # 计算当前位置与目标的偏差，考虑夹爪偏移
                    current_pos = np.array([
                        current_pose.pose.position.x + hand_trans[0],
                        current_pose.pose.position.y + hand_trans[1],
                        current_pose.pose.position.z + hand_trans[2]
                    ])
                    pos_error = np.linalg.norm(avg_pos - current_pos)
                    
                    # 如果位置和航向都对齐得比较好，完成搜索
                    yaw_error = abs((self.aligned_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi)
                    if pos_error < 0.1 and yaw_error < np.radians(5):
                        rospy.loginfo("位置和航向对齐完成")
                        return True

                    # 设置目标姿态，考虑夹爪偏移
                    self.target_pose.header.frame_id = "map"
                    self.target_pose.header.stamp = rospy.Time.now()
                    
                    # 设置目标位置，让hand对准目标
                    self.target_pose.pose.position.x = avg_pos[0] - hand_trans[0]
                    self.target_pose.pose.position.y = avg_pos[1] - hand_trans[1]
                    self.target_pose.pose.position.z = avg_pos[2] - hand_trans[2]
                    
                    # 设置目标航向
                    self.target_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, self.aligned_yaw))
                    
                    rospy.loginfo_throttle(1, f"位置对齐: 误差={pos_error:.3f}米, "
                                         f"航向误差={np.degrees(yaw_error):.1f}度")
                    
                except tf.Exception as e:
                    rospy.logwarn(f"task3 node: 获取hand变换失败: {e}")
                    return False

            # 设置目标姿态
            self.target_pose.header.frame_id = "map"
            self.target_pose.header.stamp = rospy.Time.now()

            # 设置目标航向
            target_yaw = self.aligned_yaw if self.search_phase == 2 else next_yaw
            self.target_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, target_yaw))
            
            self.move_to_target()  # 发布目标位置
            rospy.loginfo_throttle(1, f"搜索阶段{self.search_phase}: 当前航向={np.degrees(current_yaw):.1f}度, "
                                    f"目标航向={np.degrees(target_yaw):.1f}度")
            
            return False
                
        except tf.Exception as e:
            rospy.logwarn(f"task3 node: 获取当前位姿失败: {e}")
            return False

    def move_to_init_target(self):
        """移动到初始目标位置"""
        self.target_pose = self.start_point
        return self.move_to_target()

    def grab_target(self):
        """
        抓取目标：
        1. 发布抓取指令
        2. 返回True
        """
        self.grab_count += 1
        control_msg = Control()
        control_msg.led_green = 0
        control_msg.led_red = 0
        control_msg.servo = 255  # 255是闭合，100是张开夹爪
        self.control_pub.publish(control_msg)  # 夹爪控制
        # 不发位姿控制更好控制
        # self.move_to_target()  # 也需要按时发布位姿控制
        if self.grab_count >= 5:
            return True
        return False

    def finish_task(self):
        """
        任务完成：
        1. 发布任务完成标志
        2. 返回True
        """
        self.finished_pub.publish("task3 finished")
        rospy.loginfo("task3 node: 任务完成，发布完成消息")
        rospy.signal_shutdown("任务完成")
        return True

    def run(self):
        """主循环"""
        while not rospy.is_shutdown():
            if self.step == 0:  # 移动到初始位置
                if self.move_to_init_target():
                    self.step = 1
                    rospy.loginfo("task3 node: 到达初始位置，开始搜索目标")
            elif self.step == 1:  # 搜索目标
                if self.search_target():
                    self.step = 2
                    rospy.loginfo("task3 node: 找到目标，开始移动到目标位置")
            elif self.step == 2:  # 移动到目标位置
                if self.move_to_target():
                    self.step = 3
                    rospy.loginfo("task3 node: 到达目标位置，准备抓取")
            elif self.step == 3:  # 抓取目标
                if self.grab_target():
                    self.step = 5
                    rospy.loginfo("task3 node: 抓取完成，返回初始位置")
            elif self.step == 5:  # 上浮
                if self.move_to_init_target():
                    self.step = 6
                    rospy.loginfo("task3 node: 返回初始位置，任务结束")
            elif self.step == 6:  # 完成任务
                self.finish_task()
                break
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('task3_node')
    try:
        node = Task3Node()
        node.run()
    except rospy.ROSInterruptException:
        pass