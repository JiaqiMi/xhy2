#! /home/xhy/xhy_env/bin/python
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
2025.8.1 02:38
    同步task2 完善注释和驱动
"""
import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String
from auv_control.msg import TargetDetection, Control # control是控制led和舵机
from geometry_msgs.msg import PoseStamped, Quaternion,Point
import numpy as np

NODE_NAME = "task3_node"

class Task3Node:
    """任务3节点:夹球"""
    def __init__(self):
        # ros相关的初始化
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10) # 发布任务完成标志话题
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10) # 发布期望位姿话题                  
        self.control_pub = rospy.Publisher('/sensor', Control, queue_size=10) # 发布控制话题
        rospy.Subscriber('/obj/target_message', TargetDetection, self.target_detection_callback) # 订阅目标检测话题
        self.rate = rospy.Rate(5)  # 运行频率5Hz
        self.tf_listener = tf.TransformListener()
        
        # 变量定义
        self.step = 0 # 程序运行阶段
        self.target_posestamped = PoseStamped() # 期望位置消息定义
        self.start_point = PoseStamped()
        self.end_point = PoseStamped() # 结束点，和开始点一样
        self.queue = [] # 用于保存目标队列
        self.init_yaw = None  # 初始yaw角度
        self.search_direction = 1  # 搜索方向：1表示正向，-1表示反向
        self.pitch_offset = np.radians(1.5) # 固定1.5°俯仰
        self.pub_num = 0  # 记录释放目标的次数
        self.step = 0 # 程序运行阶段
        self.sensor = [0] * 5 # 用一个列表5个数字表示传感器状态，分别代表红灯、绿灯、舵机、补光灯1、补光灯2
        self.ball_depth =None
        # 获取宏定义参数
        self.target_depth = rospy.get_param('~depth', 0.3)  # 下潜深度，单位米
        start_point_from_param = rospy.get_param('/task3_point0', [0.5, -0.5, 0.15, 0.0])  # 默认值        
        self.target_color = rospy.get_param('/task1_target_color', 'red')  # 目标小球的颜色，默认红色        
        # 准备执行任务的初始点
        self.start_point.header.frame_id = "map"
        self.start_point.pose.position = Point(*start_point_from_param[:3]) 
        self.start_point.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, np.radians(start_point_from_param[3])))
        self.end_point.pose.position = self.start_point.pose.position
        self.end_point.pose.orientation = self.start_point.pose.orientation
        # 输出log
        rospy.loginfo(f"{NODE_NAME}: 初始化完成")
        rospy.loginfo(f"{NODE_NAME}: 初始点: n={self.start_point.pose.position.x}, e={self.start_point.pose.position.y}, d={self.start_point.pose.position.z}")        

    ###############################################驱动层#################################
    def is_arrival(self, current_pose:PoseStamped, target_pose:PoseStamped, max_xyz_dist=0.2, max_yaw_dist=np.radians(0.2)):
        """
        检查是否到达目标位置和航向

        Parameters:
            current_pose: PoseStamped, 当前位姿
            target_pose: PoseStamped, 目标位姿
            max_xyz_dist: float, 最大位置误差(米)
            max_yaw_dist: float, 最大航向误差(弧度)

        Returns:
            bool: 是否到达目标位置和航向
        """
        # 计算位置误差
        pos_error = self.xyz_distance(current_pose.pose.position, target_pose.pose.position)

        # 计算航向误差
        yaw_error = self.yaw_distance(current_pose.pose.orientation, target_pose.pose.orientation)
        if pos_error < max_xyz_dist and yaw_error < max_yaw_dist:
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

    def xyz_distance(self, p1:Point, p2:Point):
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
        dist_to_target = self.xyz_distance(current_pose.pose.position, target_pose.pose.position)

        # 如果距离目标点很近(小于1米)，则开始调整最终姿态
        if dist_to_target < max_xy_step:
            # 计算yaw角差异（处理角度环绕）
            dyaw = target_yaw - current_yaw
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            dyaw = np.clip(dyaw, -max_yaw_step, max_yaw_step) # 应用最大步长
            next_yaw = current_yaw + dyaw
            
            # 平滑过渡姿态角
            # next_roll = current_roll + np.clip(target_roll - current_roll, -max_yaw_step, max_yaw_step)
            # next_pitch = target_pitch  # 保持目标俯仰角
            
            next_pose.pose.position = target_pose.pose.position
            # next_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, next_pitch, next_yaw))
            next_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, self.pitch_offset, next_yaw))
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
        # next_pitch = target_pitch # 保持目标俯仰角
        
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
        next_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, self.pitch_offset, next_yaw))

        # 如果到达目标点，重置贝塞尔曲线参数
        if dist_to_target < 0.1:  # 距离阈值
            if hasattr(self, 'bezier_t'):
                del self.bezier_t
        
        return next_pose

    def move_to_target(self, max_xy_step=0.8, max_z_step=0.2, max_yaw_step=np.radians(5), max_xyz_dist=0.2, max_yaw_dist=np.radians(1)):
        """
        发送一次指令移动到目标位姿，通过生成平滑路径点实现
        
        Parameters:
            max_xy_step: float, 最大水平步长(米)，用于平滑，超过这个距离后会先转向后移动
            max_z_step: float, 最大深度步长(米)，用于平滑
            max_yaw_step: float, 最大偏航角步长(弧度)，用于平滑
            max_xyz_dist: float, 最大三维距离误差(米)，用于判断是否到达目标位置
            max_yaw_dist: float, 最大航向误差(弧度)，用于判断是否到达目标位置
        
        Returns:
            到达目标位置返回true, 未到达目标位置返回false
        """
        try:
            # 获取当前位姿
            current_pose = self.get_current_pose()
            if current_pose is None:
                return False

            # 判断是否到达
            if self.is_arrival(current_pose, self.target_posestamped, max_xyz_dist, max_yaw_dist):
                return True
            
            # 航向控制和点控制统一起来
            next_pose = self.generate_smooth_pose(current_pose, self.target_posestamped, max_xy_step=max_xy_step, max_z_step=max_z_step, max_yaw_step=max_yaw_step)
            dist_to_target = self.xyz_distance(current_pose.pose.position, self.target_posestamped.pose.position)
            yaw_to_target = self.yaw_distance(current_pose.pose.orientation, self.target_posestamped.pose.orientation)
            rospy.loginfo_throttle(2,f"{NODE_NAME}: 移动到目标点: 距离={dist_to_target:.3f}米, 航向差={np.degrees(yaw_to_target):.2f}度,高度差={current_pose.pose.position.z-self.target_posestamped.pose.position.z}")
            self.target_pub.publish(next_pose)

            return False
            
        except tf.Exception as e:
            rospy.logwarn(f"task1 node: 移动失败: {e}")
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

            # NOTE 打印一下当前位置
            _,_,yaw = euler_from_quaternion(rot)
            rospy.loginfo_throttle(2,f"{NODE_NAME}: 当前位置为: n={current_pose.pose.position.x:.2f}m, e={current_pose.pose.position.y:.2f}m, d={current_pose.pose.position.z:.2f}m, yaw={np.degrees(yaw)}")
            return current_pose
        except tf.Exception as e:
            rospy.logwarn(f"{NODE_NAME}: 获取当前位姿失败: {e}")
            return None
        
    def control_device(self):
        """发布一次外设报文"""
        control_msg = Control(*self.sensor)
        self.control_pub.publish(control_msg)
        # NOTE 打印一下命令
        rospy.loginfo(f"{NODE_NAME}: 发布外设控制: 红色led={self.sensor[0]}, 绿色led={self.sensor[1]}, 舵机={self.sensor[2]}, 补光灯1={self.sensor[3]}, 补光灯2={self.sensor[4]}")
    ###############################################驱动层#################################


    ###############################################回调层#################################
    def target_detection_callback(self, msg: TargetDetection):
        """
        收到目标检测消息，将消息加入队列，不做操作
        存的时候就应该存减去夹爪之后的位置
        """
        rospy.loginfo(f"{NODE_NAME}: 收到目标检测消息 {msg.class_name},{msg.pose.pose.position.x},{msg.pose.pose.position.y},{msg.pose.pose.position.z}")
        if msg.class_name == self.target_color:
            point_in_camera = msg.pose.pose.position # 相机坐标系下目标点
            origin_in_camera = Point(x=0, y=0, z=0)  # 相机坐标系下的原点
            if self.xyz_distance(point_in_camera, origin_in_camera) < 5.0:
                try:
                    # 将目标点从camera坐标系转换到各个坐标系
                    self.tf_listener.waitForTransform("map", msg.pose.header.frame_id, msg.pose.header.stamp, rospy.Duration(1.0))
                    target_in_map = self.tf_listener.transformPose("map", msg.pose) # 目标点在map下
                    target_in_base = self.tf_listener.transformPose("base_link", msg.pose) # 目标点在base_link下
                    
                    # 获取auv当前位姿
                    current_pose = self.get_current_pose()
                    if current_pose is None:
                        return
                    # 根据target_in_map 和current_pose 计算两者的指向作为航向
                    p0 = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z])
                    p1 = np.array([target_in_map.pose.position.x, target_in_map.pose.position.y, target_in_map.pose.position.z])
                    direction = p1 - p0
                    direction_xy = direction[:2]
                    direction_xy_norm = np.linalg.norm(direction_xy)
                    if direction_xy_norm > 0:
                        direction_xy = direction_xy / direction_xy_norm
                        # 计算期望的航向角(前进方向)
                        desired_yaw = np.arctan2(direction_xy[1], direction_xy[0])

                    # 将目标从camera坐标系转换到hand坐标系，然后再转到map坐标系
                    # 这样可以直接得到hand应该到达的位置
                    target_in_hand = self.tf_listener.transformPose("hand", msg.pose) # 目标点在hand下
                    target_in_hand.header.frame_id = "base_link"
                    rospy.loginfo(f"{NODE_NAME}: 目标点在hand下: {target_in_hand.pose.position.x:.2f}, {target_in_hand.pose.position.y:.2f}, {target_in_hand.pose.position.z:.2f}")
                    hand_target_in_map = self.tf_listener.transformPose("map", target_in_hand)
                    
                    # 期望位姿就是让base_link移动到使得hand到达目标位置
                    expected_pose = PoseStamped()
                    expected_pose.header.frame_id = "map"
                    expected_pose.header.stamp = rospy.Time.now()
                    expected_pose.pose.position = hand_target_in_map.pose.position
                    expected_pose.pose.orientation = Quaternion(*quaternion_from_euler(0,self.pitch_offset, desired_yaw)) # 期望航向是前进方向

                    # 加入队列
                    self.queue.append((msg.conf, current_pose, expected_pose, target_in_map, target_in_base))
                    rospy.loginfo(f"{NODE_NAME}: 加入队列 (conf={msg.conf:.2f})")
                    
                except tf.Exception as e:
                    rospy.logwarn(f"{NODE_NAME}: 坐标转换失败: {e}")
    ###############################################回调层#################################
    

    ###############################################逻辑层#################################
    def search_target(self, max_rotate_rad=np.radians(20),depth_bias = -0.05,max_time_interval=5.0, min_conf=0.5,max_position_interval=0.5,rotate_step=np.radians(1),max_xyz_dist=0.3,max_yaw_dist=np.radians(0.2)):
        """
        搜索目标：
        从队列中获取三个目标点，判断三个点的时间间隔和位置间隔(在map下的),如果间隔小于阈值，时间小于阈值
        则认为找到目标，将目标点更新到self.target_posestamped中，并返回True

        Parameters:
            max_rotate_rad: 最大旋转角度（弧度），用于搜索范围
            depth_bias: 深度偏差（米）
            max_time_interval: 最大时间间隔（秒），用于判断有效
            min_conf: 最小置信度（0-1），用于判断有效
            max_position_interval: 最大位置间隔（米），用于判断有效
            rotate_step: 旋转步长（弧度），用于旋转速度
            max_xyz_dist: 最大XYZ距离（米），用于判断是否到达
            max_yaw_dist: 最大偏航距离（弧度），用于判断是否到达
        """
        # 如果处理的太快就会导致不连续的点
        # 定义三个空点
        # (msg.conf, current_pose, expected_pose, target_in_map, target_in_base)
        # 循环直到队列为空或找到目标点
        # 如果这个占用很长时间呢？
        while len(self.queue) >= 3:
            # 只要有一个条件不满足，重新取点
            target1 = self.queue[0]
            target2 = self.queue[1]
            target3 = self.queue[2]
            # rospy.loginfo(f"{NODE_NAME}: 当前队列长度: {len(self.queue)}")
            if target1 is not None and target2 is not None and target3 is not None:
                if target1[0] > min_conf and target2[0] > min_conf and target3[0] > min_conf:
                    if self.xyz_distance(target1[3].pose.position, target2[3].pose.position) < max_position_interval and \
                        self.xyz_distance(target2[3].pose.position, target3[3].pose.position) < max_position_interval and \
                        self.xyz_distance(target1[3].pose.position, target3[3].pose.position) < max_position_interval:
                        # 间距满足要求
                            # 置信度满足要求
                            if abs(target1[3].header.stamp.to_sec() - target2[3].header.stamp.to_sec()) < max_time_interval and \
                                abs(target2[3].header.stamp.to_sec() - target3[3].header.stamp.to_sec()) < max_time_interval:
                                # 时间间隔满足要求
                                rospy.loginfo(f"{NODE_NAME}: 找到目标点: {target1[4]}, {target2[4]}, {target3[4]}")
                                # 计算位置平均值：根据期望位姿
                                avg_x = (target1[2].pose.position.x + target2[2].pose.position.x + target3[2].pose.position.x) / 3.0
                                avg_y = (target1[2].pose.position.y + target2[2].pose.position.y + target3[2].pose.position.y) / 3.0
                                avg_z = (target1[2].pose.position.z + target2[2].pose.position.z + target3[2].pose.position.z) / 3.0
                                # 计算航向和俯仰平均值：根据当前位姿
                                _, pitch1, yaw1 = euler_from_quaternion([
                                    target1[1].pose.orientation.x,
                                    target1[1].pose.orientation.y,
                                    target1[1].pose.orientation.z,
                                    target1[1].pose.orientation.w
                                ])
                                _, pitch2, yaw2 = euler_from_quaternion([
                                    target2[1].pose.orientation.x,
                                    target2[1].pose.orientation.y,
                                    target2[1].pose.orientation.z,
                                    target2[1].pose.orientation.w
                                ])
                                _, pitch3, yaw3 = euler_from_quaternion([
                                    target3[1].pose.orientation.x,
                                    target3[1].pose.orientation.y,
                                    target3[1].pose.orientation.z,
                                    target3[1].pose.orientation.w
                                ])
                                avg_yaw = (yaw1 + yaw2 + yaw3) / 3.0
                                avg_pitch = (pitch1 + pitch2 + pitch3) / 3.0
                                if self.ball_depth == None:
                                    self.ball_depth = avg_z +depth_bias
                                # 设置完目标位姿后，跳转到下一步即可
                                self.target_posestamped.pose.position = Point(x=avg_x,y=avg_y,z=self.start_point.pose.position.z)
                                self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, self.pitch_offset, avg_yaw))
                                # rospy.loginfo(f"{NODE_NAME}: 当前位置为：n={avg_x:.2f}m, e={avg_y:.2f}m, d={avg_z+depth_bias:.2f}m,yaw={np.degrees(avg_yaw)}")
                                rospy.loginfo(f"{NODE_NAME}: 目标位置设置为: n={avg_x:.2f}m, e={avg_y:.2f}m, d={avg_z+depth_bias:.2f}m,yaw={np.degrees(avg_yaw)}°")
                                return True
            # 如果没有找到目标点，删除队列中的第一个元素
            self.queue.pop(0)

        # 初始化当前位姿
        current_pose = self.get_current_pose()
        if current_pose == None:
            return False
        current_yaw = euler_from_quaternion([current_pose.pose.orientation.x,
                                              current_pose.pose.orientation.y,
                                              current_pose.pose.orientation.z,
                                              current_pose.pose.orientation.w])[2]
        if self.init_yaw is None:
            self.target_posestamped.pose = current_pose.pose
            self.init_yaw = current_yaw
        next_yaw = current_yaw + (rotate_step * self.search_direction)

        if next_yaw > self.init_yaw + max_rotate_rad:
            self.search_direction = -1
            next_yaw = current_yaw + (rotate_step * self.search_direction)
            # rospy.loginfo(f"{NODE_NAME}: test search: 掉头顺时针搜索")
        elif next_yaw < self.init_yaw - max_rotate_rad:
            self.search_direction = 1
            next_yaw = current_yaw + (rotate_step * self.search_direction)
            # rospy.loginfo(f"{NODE_NAME}: test search: 掉头逆时针搜索")
        
        # 设置目标位姿，位置不变，原地开始旋转加一个旋转角度
        self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, self.pitch_offset, next_yaw))

        # 执行一次旋转指令
        self.move_to_target(max_xyz_dist=max_xyz_dist, max_yaw_step=rotate_step, max_yaw_dist=max_yaw_dist)
        return False
    
    def move_to_init_pose(self):
        """
        发送一次指令移动到初始位姿

        Returns:
            到达目标位置返回true,未到达目标位置返回false
        """
        self.target_posestamped = self.start_point # 将宏定义的初始位置赋值给目标位置
        return self.move_to_target()
    
    def move_to_end_pose(self,max_xyz_dist=0.1, max_yaw_dist=np.radians(0.5),max_xy_step=0.8):
        """
        发送一次指令移动到结束位姿

        Returns:
            到达目标位置返回true,未到达目标位置返回false
        """
        # self.end_point.pose.position.z = self.target_posestamped.pose.position.z  # 设置结束点的深度
        self.target_posestamped.pose.position = self.end_point.pose.position # 将宏定义的初始位置赋值给目标位置
        # rospy.loginfo(self.end_point.pose.position)
        return self.move_to_target(max_xyz_dist=max_xyz_dist,max_yaw_dist=max_yaw_dist,max_xy_step=max_xy_step)
        
    def grab_target(self):
        """
        抓取目标：
        1. 发布抓取指令
        2. 返回True
        """
        if self.pub_num < 10:
            self.pub_num += 1
            self.sensor[2] = 255 # 关闭舵机
            self.move_to_target()  # 也需要按时发布位姿控制
            return False
        self.pub_num = 0
        return True

    def open_light(self,light1:int,light2:int):
        """
        打开补光灯

        Parameters:
            light1: int, 补光灯1的亮度(0~100)
            light2: int, 补光灯2的亮度(0~100)
        """
        if self.pub_num < 10:
            self.sensor[2] = 100 # 同时打开舵机
            self.sensor[3] = light1
            self.sensor[4] = light2
            self.control_device()
            self.move_to_target()
            self.pub_num += 1
            return False
        self.pub_num = 0 # 重置发布次数
        return True
    
    def finish_task(self):
        """
        任务完成：
        1. 发布任务完成标志
        2. 返回True
        """
        self.finished_pub.publish(f"{NODE_NAME} finished")
        rospy.loginfo(f"{NODE_NAME}: 任务完成，发布完成消息")
        rospy.signal_shutdown("任务完成")
        return True 
    
    def dive_to_ball():
        self.target_posestamped.z = self.ball_depth
        return self.move_to_target(max_xyz_dist=0.15,max_yaw_dist=np.radians(1.5))
    ###############################################逻辑层#################################


    ###############################################主循环#################################
    def run(self):
        """主循环"""
        while not rospy.is_shutdown():
            if self.step == 0:  # 移动到初始位置
                if self.move_to_init_pose():
                    self.step = 1
                    rospy.loginfo("task3 node: 到达初始位置，开始搜索目标")
            elif self.step == 1:  # 搜索目标
                if self.open_light(50, 50):
                    self.step = 10  # 张开夹爪s
            elif self.step == 10:
                if self.search_target(max_rotate_rad=np.radians(15),min_conf=0.5,max_xyz_dist=0.3,depth_bias = -0.05,rotate_step=np.radians(0.5),max_yaw_dist=np.radians(0.2)):
                    self.step = 2
                    rospy.loginfo("task3 node: 找到目标，开始移动到目标位置")
            elif self.step == 2:  # 移动到目标位置
                if self.move_to_target(max_xyz_dist=0.1,max_xy_step=1.8,max_yaw_dist=np.radians(1)):
                    self.step = 11
                    rospy.loginfo("task3 node: 到达目标位置，准备抓取")
            elif self.step ==11:
                if self.dive_to_ball():
                    self.step == 3
                    rospy.loginfo("task3 node: 下沉完毕")
            elif self.step == 3:  # 抓取目标
                if self.grab_target():
                    self.step = 5
                    rospy.loginfo("task3 node: 抓取完成，返回初始位置")
            elif self.step == 5:  # 回到初始任务点
                if self.move_to_end_pose(max_xyz_dist=0.1,max_yaw_dist=np.radians(0.2),max_xy_step=2):
                    self.step = 6
                    rospy.loginfo("task3 node: 返回结束位置，任务结束")
            elif self.step == 6:  # 完成任务
                self.finish_task()
                break
            self.rate.sleep()
    ###############################################主循环#################################

if __name__ == '__main__':
    rospy.init_node(f'{NODE_NAME}', anonymous=True)
    try:
        node = Task3Node()
        node.run()
    except rospy.ROSInterruptException:
        pass