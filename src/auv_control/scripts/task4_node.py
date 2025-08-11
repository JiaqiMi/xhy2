#! /home/xhy/xhy_env/bin/python
"""
名称: task4_node.py
功能: 巡线 作业
    黄色三角形：机器人需水平旋转一周完成巡视。
    黑色正方形：机器人需要有一个外置能被轻易看到的LED灯，在黑色正方形处机器人需要亮红灯
    绿色圆形：机器人任意部位触碰传感器并亮绿灯
作者: buyegaid
监听：  /target_detection (来自视觉节点) 检测目标对应颜色的标志
        /tf (来自tf树)
发布：  /auv_control (Control.msg) 被sensor_driver订阅
        /finished (String) 被state_control订阅, 代表任务是否完成
        /target (PoseStamped.msg) 被tf_handler订阅, 代表目标位置

记录：
2025.8.6 21:53
    final check
2025.8.11 02:50
    add(arrive_end): 判断是否到达终点附近，直接跳转到结束
    TODO 根据实际顺序调整检测顺序
    fix(target_detection_callback): 修复目标检测回调函数，添加step判断
    fix(track_detection_callback): 修复轨迹检测回调函数，添加step判断
    fix(detect_yellow_triangle): 删掉置信度判断
    fix(detect_black_rectangle): 删掉置信度判断
    fix(detect_green_circle): 删掉置信度判断
"""
import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String
from auv_control.msg import TargetDetection, Control,TargetDetection3 # task4 特有，3点序列
from geometry_msgs.msg import PoseStamped, Quaternion, Point, Pose
import numpy as np

NODE_NAME = "task4_node"

# step0: 运动到初始作业位置附近，到达后跳转到step1
# step1: 巡线跟踪第{track_count+1}段轨迹，判断是否看到任务点标志，是则运动到任务点，执行任务后跳转到step2；如果是最后一段轨迹，则跳转到step6
# step2: 判断是哪一个任务，运动到任务点，跳转到相应任务步骤
# step3: 检测到黄色三角形，旋转360度后，track_count增加1,然后跳转到step1
# step4: 检测到黑色方形，亮红灯，track_count增加1,然后跳转到step1
# step5: 检测到绿色圆形，亮绿灯，track_count增加1,然后跳转到step1
# step6: 巡线跟踪第四段轨迹，直到轨迹尽头，跳转到step5
# step7: 完成任务，发布任务完成标志，关闭节点


class Task4Node:
    def __init__(self):
        # ros相关的初始化
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10) # 发布目标位置
        self.control_pub = rospy.Publisher('/sensor', Control, queue_size=10) # 发布外设控制
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10) # 发布任务完成标志
        rospy.Subscriber('/obj/target_message', TargetDetection, self.target_detection_callback) # 订阅目标检测消息
        rospy.Subscriber('/obj/line_message', TargetDetection3, self.track_detection_callback) # 订阅轨迹检测消息
        # rospy.Subscriber('/obj/arco_message', ArcoDetection, self.arco_detection_callback)
        self.rate = rospy.Rate(5)  # 5Hz
        self.tf_listener = tf.TransformListener() # 订阅tf变换

        # 变量定义
        self.step = 0  # 程序运行阶段
        self.target_posestamped = PoseStamped()  # 期望位置消息定义
        self.start_point = PoseStamped()
        self.end_point = PoseStamped()  # 结束点，和开始点一样
        self.yellow_triangle_queue = []  # 用于保存三角目标列表
        self.black_rectangle_queue = []  # 用于保存方形目标列表
        self.green_circle_queue = []  # 用于保存圆形目标列表
        self.track_target = None  # 存储转换后的轨迹目标
        self.init_yaw = None  # 初始yaw角度
        self.search_direction = 1  # 搜索方向：1表示正向，-1表示反向
        # self.lineupdate = 25
        # self.red_count = 0  # 记录红灯动作的次数
        # self.green_count = 0  # 记录绿灯动作的次数
        # self.track_count = 0  # 记录巡线轨迹段数
        self.direction = 1
        self.step_deg = 5
        self.step_rad = np.radians(self.step_deg) * self.direction       
        # self.light = 0  # 固定60亮度
        # self.round = False # 判断是否执行过转圈任务
        self.sensor = [0] * 5  # 用一个列表5个数字表示传感器状态，分别代表红灯、绿灯、舵机、补光灯1、补光灯2
        self.pub_num = 0 # 关灯控制
        self.done = [0, 0, 0]  # 任务完成标志，分别代表红色方形、绿色圆形、黄色三角形任务是否完成

        # 获取宏定义参数
        self.target_depth = rospy.get_param('/depth', 0.3)  # 下潜深度，单位米
        start_point_from_param = rospy.get_param('/task4_point0', [0.5, -0.5, 0.15, 0.0])  # 默认值
        end_point_from_param = rospy.get_param('/task4_point1', [0.5, -0.5, 0.15, 0.0])  # 默认值
        self.target_depth = start_point_from_param[2]  # 任务深度，单位米
        self.yellow_triangle = rospy.get_param('/task4_target_shape1', 'triangle')  # 检测目标名称
        self.black_rectangle = rospy.get_param('/task4_target_shape2', 'rectangle')  # 检测目标名称
        self.green_circle = rospy.get_param('/task4_target_shape3', 'circle')  # 检测目标名称
        self.track = rospy.get_param('/task4_target_shape', 'track')  # 检测目标名称
        self.pitch_offset = np.radians(rospy.get_param('/pitch_offset', 0.0)) # 固定俯仰角

        # 准备执行任务的初始点
        self.start_point.header.frame_id = "map"
        self.start_point.pose.position = Point(*start_point_from_param[:3])
        self.start_point.pose.orientation = Quaternion(
            *quaternion_from_euler(0, self.pitch_offset, np.radians(start_point_from_param[3])))
        
        self.end_point.header.frame_id = "map"
        self.end_point.pose.position = Point(*end_point_from_param[:3])
        self.end_point.pose.orientation = Quaternion(
            *quaternion_from_euler(0, self.pitch_offset, np.radians(end_point_from_param[3])))
        
        # 输出log
        rospy.loginfo(f"{NODE_NAME}: 初始化完成")
        rospy.loginfo(f"{NODE_NAME}: 作业深度: {self.target_depth}米")
        rospy.loginfo(f"{NODE_NAME}: 初始点: n={self.start_point.pose.position.x}, e={self.start_point.pose.position.y}, d={self.start_point.pose.position.z},{start_point_from_param[3]} ")
        rospy.loginfo(f"{NODE_NAME}: 结束点: n={self.end_point.pose.position.x}, e={self.end_point.pose.position.y}, d={self.end_point.pose.position.z},{end_point_from_param[3]} ")
    
    ############################################### 驱动层 #########################################
    def is_arrival(self, current_pose: PoseStamped, target_pose: PoseStamped, max_xyz_dist=0.2,max_yaw_dist=np.radians(0.2)):
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
        pos_error = self.xyz_distance(current_pose.pose.position, target_pose.pose.position)

        # 计算航向误差
        yaw_error = self.yaw_distance(current_pose.pose.orientation, target_pose.pose.orientation)
        if pos_error < max_xyz_dist and yaw_error < max_yaw_dist:
            return True
        else:
            return False

    def yaw_distance(self, ori1: Quaternion, ori2: Quaternion):
        """
        计算两个航向之间的差值，并处理角度环绕问题

        Parameters:
            ori1: Quaternion, 第一个四元数
            ori2: Quaternion, 第二个四元数

        Returns:
            float: 两个航向之间的差值（绝对值）弧度
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

    def xyz_distance(self, p1: Point, p2: Point):
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

    def generate_smooth_pose(self, current_pose: PoseStamped, target_pose: PoseStamped, max_xy_step=0.8,
                            max_z_step=0.1, max_yaw_step=np.radians(5)):
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
        _, _, target_yaw = euler_from_quaternion([
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
            dyaw = np.clip(dyaw, -max_yaw_step, max_yaw_step)  # 应用最大步长
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
        next_point = (1 - t) ** 3 * p0 + \
                     3 * (1 - t) ** 2 * t * p1 + \
                     3 * (1 - t) * t ** 2 * p2 + \
                     t ** 3 * p3

        # 应用步长限制
        dp = next_point - p0
        dist_xy = np.sqrt(dp[0] ** 2 + dp[1] ** 2)
        # NOTE 8.4 12:17 缩小最大速度
        if dist_xy > max_xy_step*0.6: # NOTE 0.48m max task4 特调
            scale = max_xy_step*0.6 / dist_xy
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
                rospy.loginfo(f"{NODE_NAME}: 已到达目标位置")
                return True

            # 航向控制和点控制统一起来
            next_pose = self.generate_smooth_pose(current_pose, self.target_posestamped, max_xy_step=max_xy_step,
                                                max_z_step=max_z_step, max_yaw_step=max_yaw_step)
            dist_to_target = self.xyz_distance(current_pose.pose.position, self.target_posestamped.pose.position)
            yaw_to_target = self.yaw_distance(current_pose.pose.orientation, self.target_posestamped.pose.orientation)
            rospy.loginfo_throttle(5,
                                f"{NODE_NAME}: 移动到目标位置: 距离={dist_to_target:.3f}米, 航向差={np.degrees(yaw_to_target):.2f}度,高度差={current_pose.pose.position.z - self.target_posestamped.pose.position.z}")
            self.target_pub.publish(next_pose)

            return False

        except tf.Exception as e:
            rospy.logwarn(f"{NODE_NAME}: 移动失败: {e}")
            return False

    def get_current_pose(self):
        """获取当前位姿"""
        try:
            self.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))  # 返回一个三元组和一个四元组
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

    # def light_red_led(self):
    #     """亮红灯"""
    #     control_msg = Control()
    #     control_msg.led_green = 0
    #     control_msg.led_red = 1 # 亮红灯
    #     control_msg.servo = 255
    #     control_msg.light1 = self.light
    #     control_msg.light2 = self.light
    #     rospy.loginfo(f"{NODE_NAME}: 亮红灯")
    #     self.control_pub.publish(control_msg)
    #
    # def light_green_led(self):
    #     """亮绿灯"""
    #     control_msg = Control()
    #     control_msg.led_green = 1 # 亮绿灯
    #     control_msg.led_red = 0
    #     control_msg.servo = 255
    #     control_msg.light1 = self.light
    #     control_msg.light2 = self.light
    #     rospy.loginfo(f"{NODE_NAME}: 亮绿灯")
    #     self.control_pub.publish(control_msg)

    # def light_out(self):
    #     """灭灯"""
    #     control_msg = Control()
    #     control_msg.led_green = 0
    #     control_msg.led_red = 0
    #     control_msg.servo = 255
    #     control_msg.light1 = self.light
    #     control_msg.light2 = self.light
    #     rospy.loginfo(f"{NODE_NAME}: 灭灯")
    #     self.control_pub.publish(control_msg)
    ############################################### 驱动层 #########################################

    ###############################################回调层#################################
    def target_detection_callback(self, msg: TargetDetection):
        """
        收到目标检测消息，将消息加入三个队列当中（加入的是三种形状）
        加入队列中的是体到达目标点
        """
        rospy.loginfo(
            f"{NODE_NAME}: 检测到{msg.class_name}: {msg.pose.pose.position.x},{msg.pose.pose.position.y},{msg.pose.pose.position.z}")
        # rospy.loginfo(f"{rospy.Time.now()},{msg.pose.header.stamp}")
        # point_in_camera = msg.pose.pose.position  # 相机坐标系下目标点
        # origin_in_camera = Point(x=0, y=0, z=0)  # 相机坐标系下的原点
        # if self.xyz_distance(point_in_camera, origin_in_camera) < 5.0 and self.step > 0: # 最远距离小于5m, 并且不在初始阶段
        if self.step > 0: # 不在初始阶段
            try:
                # 将目标点从camera坐标系转换到各个坐标系
                self.tf_listener.waitForTransform("map", msg.pose.header.frame_id, msg.pose.header.stamp, rospy.Duration(1.0))
                target_in_map = self.tf_listener.transformPose("map", msg.pose)  # 目标点在map下
                target_in_base = self.tf_listener.transformPose("base_link", msg.pose)  # 目标点在base_link下

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
                
                # 计算期望的航向角(前进方向)
                if direction_xy_norm > 0:
                    # 直接使用向量计算航向角，np.arctan2已经返回[-π, π]范围的角度
                    desired_yaw = np.arctan2(direction_xy[1], direction_xy[0])
                else:
                    # 如果水平距离为0（目标在正上方或正下方），保持当前航向
                    current_yaw = euler_from_quaternion([
                        current_pose.pose.orientation.x,
                        current_pose.pose.orientation.y,
                        current_pose.pose.orientation.z,
                        current_pose.pose.orientation.w
                    ])[2]
                    desired_yaw = current_yaw

                # 将目标从camera坐标系转换到hand坐标系，然后再转到map坐标系
                
                # 这样可以直接得到hand应该到达的位置
                target_in_hand = self.tf_listener.transformPose("hand", msg.pose)  # 目标点在hand下
                target_in_hand.header.frame_id = "base_link"
                # rospy.loginfo(
                #    f"{NODE_NAME}: 目标点在hand下: {target_in_hand.pose.position.x:.2f}, {target_in_hand.pose.position.y:.2f}, {target_in_hand.pose.position.z:.2f}")
                hand_target_in_map = self.tf_listener.transformPose("map", target_in_hand)

                # 期望位姿就是让base_link移动到使得hand到达目标位置
                expected_pose = PoseStamped()
                expected_pose.header.frame_id = "map"
                expected_pose.header.stamp = rospy.Time.now()
                # 用目标在map下的位置作为期望位置
                expected_pose.pose.position = target_in_map.pose.position
                expected_pose.pose.orientation = Quaternion(
                    *quaternion_from_euler(0, self.pitch_offset, desired_yaw))  # 期望航向是前进方向
                
                # 队列结构(conf, current_pose, expected_pose, target_in_map, target_in_base)
                if msg.class_name == self.yellow_triangle:
                    # 加入队列
                    self.yellow_triangle_queue.append((msg.conf, current_pose, expected_pose, target_in_map, target_in_base,target_in_hand))
                    rospy.loginfo(f"{NODE_NAME}: 加入队列 (conf={msg.conf:.2f})")
                elif msg.class_name == self.black_rectangle:
                    # 加入队列
                    self.black_rectangle_queue.append((msg.conf, current_pose, expected_pose, target_in_map, target_in_base, target_in_hand))
                    rospy.loginfo(f"{NODE_NAME}: 加入队列 (conf={msg.conf:.2f})")
                elif msg.class_name == self.green_circle:
                    # 加入队列
                    self.green_circle_queue.append((msg.conf, current_pose, expected_pose, target_in_map, target_in_base, target_in_hand))
                    rospy.loginfo(f"{NODE_NAME}: 加入队列 (conf={msg.conf:.2f})")

            except tf.Exception as e:
                rospy.logwarn(f"{NODE_NAME}: 坐标转换失败: {e}")

    def track_detection_callback(self, msg: TargetDetection3):
        """
        收到目标检测消息，将期望位姿存入到self.track_target
        加入队列中的眼睛到达的位置
        """

        # if msg.conf < 0.5:
        #     return
        rospy.loginfo(
            f"{NODE_NAME}: 收到track 1, {msg.pose1.pose.position.x},{msg.pose1.pose.position.y},{msg.pose1.pose.position.z}")
        rospy.loginfo(
            f"{NODE_NAME}: 收到track 2, {msg.pose2.pose.position.x},{msg.pose2.pose.position.y},{msg.pose2.pose.position.z}")
        rospy.loginfo(
            f"{NODE_NAME}: 收到track 3, {msg.pose3.pose.position.x},{msg.pose3.pose.position.y},{msg.pose3.pose.position.z}")

        # point3_in_camera = msg.pose3.pose.position  # 相机坐标系下目标点
        # origin_in_camera = Point(x=0, y=0, z=0)  # 相机坐标系下的原点
        # if self.xyz_distance(point3_in_camera, origin_in_camera) < 5.0: # 保证第三个点在5m以内
            # self.lineupdate = 25
        if self.step > 0:
            try:
                # 将目标点从camera坐标系转换到各个坐标系
                # self.tf_listener.waitForTransform("map", msg.pose1.header.frame_id, msg.pose1.header.stamp,
                #                                  rospy.Duration(1.0))
                # target1_in_map = self.tf_listener.transformPose("map", msg.pose1)  # 目标点在map下
                # target1_in_base = self.tf_listener.transformPose("base_link", msg.pose1)  # 目标点在base_link下

                # 第一个点不要，计算第二个点和第三个点的方向作为期望航向
                self.tf_listener.waitForTransform("map", msg.pose2.header.frame_id, msg.pose2.header.stamp, rospy.Duration(1.0))
                target2_in_map = self.tf_listener.transformPose("map", msg.pose2)  # 目标点在map下
                # target2_in_base = self.tf_listener.transformPose("base_link", msg.pose2)  # 目标点在base_link下

                self.tf_listener.waitForTransform("map", msg.pose3.header.frame_id, msg.pose3.header.stamp, rospy.Duration(1.0))
                target3_in_map = self.tf_listener.transformPose("map", msg.pose3)  # 目标点在map下
                # target3_in_base = self.tf_listener.transformPose("base_link", msg.pose3)  # 目标点在base_link下

                # 获取auv当前位姿
                current_pose = self.get_current_pose()
                if current_pose is None:
                    return
                # 计算轨迹方向：从第二个点指向第三个点的方向作为期望航向
                p2 = np.array([target2_in_map.pose.position.x, target2_in_map.pose.position.y, target2_in_map.pose.position.z])
                p3 = np.array([target3_in_map.pose.position.x, target3_in_map.pose.position.y, target3_in_map.pose.position.z])
                
                direction = p3 - p2
                direction_xy = direction[:2]
                direction_xy_norm = np.linalg.norm(direction_xy)
                
                # 计算期望的航向角(轨迹方向)
                if direction_xy_norm > 0:
                    # 直接使用向量计算航向角，np.arctan2已经返回[-π, π]范围的角度
                    desired_yaw = np.arctan2(direction_xy[1], direction_xy[0])
                else:
                    # 如果第二个点和第三个点重合，保持当前航向
                    current_yaw = euler_from_quaternion([
                        current_pose.pose.orientation.x,
                        current_pose.pose.orientation.y,
                        current_pose.pose.orientation.z,
                        current_pose.pose.orientation.w
                    ])[2]
                    desired_yaw = current_yaw

                # 将目标从camera坐标系转换到hand坐标系，然后再转到map坐标系
                # 这样可以直接得到hand应该到达的位置
                target2_in_hand = self.tf_listener.transformPose("hand", msg.pose2)  # 目标点在hand下
                target2_in_hand.header.frame_id = "base_link"
                # rospy.loginfo(
                #    f"{NODE_NAME}: 目标点在hand下: {target2_in_hand.pose.position.x:.2f}, {target2_in_hand.pose.position.y:.2f}, {target2_in_hand.pose.position.z:.2f}")
                hand_target2_in_map = self.tf_listener.transformPose("map", target2_in_hand)

                # 创建目标位姿
                self.track_target = PoseStamped()
                self.track_target.header.frame_id = "map"
                self.track_target.header.stamp = rospy.Time.now()
                self.track_target.pose.position = Point(
                    x=hand_target2_in_map.pose.position.x,
                    y=hand_target2_in_map.pose.position.y,
                    z=self.target_depth  # 使用设定的深度
                )
                self.track_target.pose.orientation = Quaternion(
                    *quaternion_from_euler(0, self.pitch_offset, desired_yaw))
                rospy.loginfo(f"{NODE_NAME}: 更新轨迹目标位置")
            except tf.Exception as e:
                rospy.logwarn(f"{NODE_NAME}: 坐标转换失败: {e}")
                self.track_target = None


    def arco_detection_callback(self, msg):
        rospy.loginfo(
            f"{NODE_NAME}: 收到arco目标检测消息")
        point_in_camera = msg.pose.pose.position  # 相机坐标系下目标点
        origin_in_camera = Point(x=0, y=0, z=0)  # 相机坐标系下的原点
        if self.xyz_distance(point_in_camera, origin_in_camera) < 1.0:
            self.step = 7

    ###############################################回调层#################################

    ###############################################逻辑层#################################

    def follow_track(self, max_rotate_rad=np.radians(30), rotate_step=np.radians(1), max_xyz_dist=0.3, max_yaw_dist=np.radians(0.2),forward_percent=0.8):
        """
        跟踪轨迹：
        1. 如果有track_target，移动到第二个点位置，航向指向第三个点
        2. 如果没有轨迹，进行小范围搜索
        """
        # # 优先检查是否有形状任务
        # if len(self.yellow_triangle_queue) > 0 or len(self.black_rectangle_queue) > 0 or len(self.green_circle_queue) > 0:
        #     return
        
        # 如果有轨迹目标，跟踪轨迹
        if self.track_target:
            self.target_posestamped.pose.position = Point(x=self.target_posestamped.pose.position.x + (self.track_target.pose.position.x-self.target_posestamped.pose.position.x)*forward_percent, 
                                                        y=self.target_posestamped.pose.position.y + (self.track_target.pose.position.y-self.target_posestamped.pose.position.y)*forward_percent, 
                                                        z=self.target_depth)    
            self.target_posestamped.pose.position.z = self.target_depth
            rospy.loginfo(f"{NODE_NAME}: 跟踪轨迹目标更新 n={self.target_posestamped.pose.position.x:.2f}, e={self.target_posestamped.pose.position.y:.2f}, d={self.target_posestamped.pose.position.z:.2f}")
            # 运动到目标点
            # self.move_to_target(max_xyz_dist=max_xyz_dist, max_yaw_dist=max_yaw_dist)
            self.track_target = None
            self.init_yaw = None
            # self.track_count += 1

            return True

        # 延时等待新的轨迹数据
        # if self.lineupdate > 0:
        #     self.lineupdate -= 1
        #     return

        # 如果找不到线，进行小范围搜索
        current_pose = self.get_current_pose()
        if current_pose is None:
            return False
            
        current_yaw = euler_from_quaternion([
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ])[2]
        
        if self.init_yaw is None:
            # 初始化目标位姿
            # self.target_posestamped.header.frame_id = "map"
            # self.target_posestamped.header.stamp = rospy.Time.now()
            # self.target_posestamped.pose.position = current_pose.pose.position
            # 不能更新位置，因为会有深度变化
            self.target_posestamped.pose.orientation = current_pose.pose.orientation
            self.init_yaw = current_yaw
            
        next_yaw = current_yaw + (rotate_step * self.search_direction)
        
        # 角度标准化：将next_yaw限制在[-π, π]范围内
        next_yaw = (next_yaw + np.pi) % (2 * np.pi) - np.pi
        
        # 计算相对于初始角度的角度差
        yaw_diff = next_yaw - self.init_yaw
        yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
        
        if yaw_diff > max_rotate_rad:
            self.search_direction = -1
            next_yaw = current_yaw + (rotate_step * self.search_direction)
            next_yaw = (next_yaw + np.pi) % (2 * np.pi) - np.pi
        elif yaw_diff < -max_rotate_rad:
            self.search_direction = 1
            next_yaw = current_yaw + (rotate_step * self.search_direction)
            next_yaw = (next_yaw + np.pi) % (2 * np.pi) - np.pi

        # 设置目标位姿，位置不变，原地旋转搜索线
        self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, self.pitch_offset, next_yaw))
        
        # 执行搜索运动
        self.move_to_target(max_xyz_dist=max_xyz_dist, max_yaw_step=rotate_step, max_yaw_dist=max_yaw_dist)
        return False

    def move_to_init_pose(self, max_xy_step=0.5, max_z_step=0.1, max_yaw_step=np.radians(5), max_xyz_dist=0.3, max_yaw_dist=np.radians(0.2)):
        """
        发送一次指令移动到初始位姿

        Returns:
            到达目标位置返回true,未到达目标位置返回false
        """
        # self.target_posestamped = self.start_point  # 将宏定义的初始位置赋值给目标位置
        self.target_posestamped.pose.position = self.start_point.pose.position
        self.target_posestamped.pose.orientation = self.start_point.pose.orientation
        # rospy.loginfo(self.target_posestamped)
        # rospy.loginfo(self.start_point)
        return self.move_to_target(max_xy_step=max_xy_step, max_z_step=max_z_step, max_yaw_step=max_yaw_step, max_xyz_dist=max_xyz_dist, max_yaw_dist=max_yaw_dist)

    def detect_yellow_triangle(self, max_time_interval=5.0, max_position_interval=0.5, rotate_step=np.radians(1), max_xyz_dist=0.3, max_yaw_dist=np.radians(0.2), forward_percent=0.8):
        """
        探测到任务点目标：
        1. 停止巡线
        2. 移动到任务点
        """
        while len(self.yellow_triangle_queue) >= 3:
            # 只要有一个条件不满足，重新取点
            target1 = self.yellow_triangle_queue[0]
            target2 = self.yellow_triangle_queue[1]
            target3 = self.yellow_triangle_queue[2]
            # rospy.loginfo(f"{NODE_NAME}: 当前队列长度: {len(self.yellow_triangle_queue)}")
            if target1 is not None and target2 is not None and target3 is not None:
                # if target1[0] > 0.5 and target2[0] > 0.5 and target3[0] > 0.5:
                if self.xyz_distance(target1[3].pose.position, target2[3].pose.position) < max_position_interval and \
                        self.xyz_distance(target2[3].pose.position, target3[3].pose.position) < max_position_interval and \
                        self.xyz_distance(target1[3].pose.position, target3[3].pose.position) < max_position_interval:
                    # 间距满足要求
                    # 置信度满足要求
                    if abs(target1[3].header.stamp.to_sec() - target2[
                        3].header.stamp.to_sec()) < max_time_interval and \
                            abs(target2[3].header.stamp.to_sec() - target3[
                                3].header.stamp.to_sec()) < max_time_interval:
                        # 时间间隔满足要求
                        rospy.loginfo(f"{NODE_NAME}: 找到yellow triangle: {target1[4]}, {target2[4]}, {target3[4]}")
                        # 计算位置平均值：根据期望位姿
                        avg_x = (target1[2].pose.position.x + target2[2].pose.position.x + target3[
                            2].pose.position.x) / 3.0
                        avg_y = (target1[2].pose.position.y + target2[2].pose.position.y + target3[
                            2].pose.position.y) / 3.0
                        avg_z = (target1[2].pose.position.z + target2[2].pose.position.z + target3[
                            2].pose.position.z) / 3.0
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
                        # 清空队列，清空初始位置
                        self.yellow_triangle_queue= []
                        # 设置完目标位姿后，跳转到下一步即可
                        self.target_posestamped.pose.position = Point(x=self.target_posestamped.pose.position.x + (avg_x-self.target_posestamped.pose.position.x)*forward_percent, 
                                                                y=self.target_posestamped.pose.position.y + (avg_y-self.target_posestamped.pose.position.y)*forward_percent, 
                                                                z =self.target_depth)
                        # self.target_posestamped.pose.position = Point(x=avg_x, y=avg_y, z=self.target_depth)
                        self.target_posestamped.pose.orientation = Quaternion(
                            *quaternion_from_euler(0, self.pitch_offset, avg_yaw))
                        rospy.loginfo(
                            f"{NODE_NAME}: 三角形位置设置为: n={avg_x:.2f}m, e={avg_y:.2f}m, d={self.target_depth}m,yaw={np.degrees(avg_yaw)}°")
                        return True
            self.yellow_triangle_queue.pop(0)
        return False


    def detect_black_rectangle(self, max_time_interval=5.0, max_position_interval=0.5, rotate_step=np.radians(1), max_xyz_dist=0.3, max_yaw_dist=np.radians(0.2), forward_percent=0.8):
        """
        探测到任务点目标：
        1. 停止巡线
        2. 移动到任务点
        """
        while len(self.black_rectangle_queue) >= 3:
            # 只要有一个条件不满足，重新取点
            target1 = self.black_rectangle_queue[0]
            target2 = self.black_rectangle_queue[1]
            target3 = self.black_rectangle_queue[2]
            rospy.loginfo(f"{NODE_NAME}: 当前队列长度: {len(self.black_rectangle_queue)}")
            if target1 is not None and target2 is not None and target3 is not None:
                # if target1[0] > 0.5 and target2[0] > 0.5 and target3[0] > 0.5:
                if self.xyz_distance(target1[3].pose.position, target2[3].pose.position) < max_position_interval and \
                        self.xyz_distance(target2[3].pose.position, target3[3].pose.position) < max_position_interval and \
                        self.xyz_distance(target1[3].pose.position, target3[3].pose.position) < max_position_interval:
                    # 间距满足要求
                    # 置信度满足要求
                    if abs(target1[3].header.stamp.to_sec() - target2[
                        3].header.stamp.to_sec()) < max_time_interval and \
                            abs(target2[3].header.stamp.to_sec() - target3[
                                3].header.stamp.to_sec()) < max_time_interval:
                        # 时间间隔满足要求
                        rospy.loginfo(f"{NODE_NAME}: 找到black rectangle: {target1[4]}, {target2[4]}, {target3[4]}")
                        # 计算位置平均值：根据期望位姿
                        avg_x = (target1[2].pose.position.x + target2[2].pose.position.x + target3[
                            2].pose.position.x) / 3.0
                        avg_y = (target1[2].pose.position.y + target2[2].pose.position.y + target3[
                            2].pose.position.y) / 3.0
                        avg_z = (target1[2].pose.position.z + target2[2].pose.position.z + target3[
                            2].pose.position.z) / 3.0
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

                        # 设置完目标位姿后，跳转到下一步即可
                        self.target_posestamped.pose.position = Point(x=self.target_posestamped.pose.position.x + (avg_x-self.target_posestamped.pose.position.x)*forward_percent, 
                                                                y=self.target_posestamped.pose.position.y + (avg_y-self.target_posestamped.pose.position.y)*forward_percent, 
                                                                z=self.target_depth)
                        # self.target_posestamped.pose.position = Point(x=avg_x, y=avg_y, z=self.target_depth)
                        self.target_posestamped.pose.orientation = Quaternion(
                            *quaternion_from_euler(0, self.pitch_offset, avg_yaw))
                        # 清空队列，清空初始位置
                        self.black_rectangle_queue = []
                        rospy.loginfo(
                            f"{NODE_NAME}: 黑色长方形位置设置为: n={avg_x:.2f}m, e={avg_y:.2f}m, d=0.00m,yaw={np.degrees(avg_yaw)}°")
                        return True
            self.black_rectangle_queue.pop(0)
        return False

    def detect_green_circle(self, min_conf=0.5, max_time_interval=5.0, max_position_interval=0.5,forward_percent=0.8):
        """
        检测绿色圆形目标，不控制运动，检测到了之后返回True
        """
        while len(self.green_circle_queue) >= 3:
            target1 = self.green_circle_queue[0]
            target2 = self.green_circle_queue[1] 
            target3 = self.green_circle_queue[2]
            # if target1[0] > min_conf and target2[0] > min_conf and target3[0] > min_conf:
            if (self.xyz_distance(target1[3].pose.position, target2[3].pose.position) < max_position_interval and 
                self.xyz_distance(target2[3].pose.position, target3[3].pose.position) < max_position_interval and 
                self.xyz_distance(target1[3].pose.position, target3[3].pose.position) < max_position_interval):
                
                if (abs(target1[3].header.stamp.to_sec() - target2[3].header.stamp.to_sec()) < max_time_interval and 
                    abs(target2[3].header.stamp.to_sec() - target3[3].header.stamp.to_sec()) < max_time_interval):
                    
                    # 计算平均位置
                    avg_x = (target1[2].pose.position.x + target2[2].pose.position.x + target3[2].pose.position.x) / 3.0
                    avg_y = (target1[2].pose.position.y + target2[2].pose.position.y + target3[2].pose.position.y) / 3.0
                    avg_z = (target1[2].pose.position.z + target2[2].pose.position.z + target3[2].pose.position.z) / 3.0
                    
                    # 计算平均航向
                    _, _, yaw1 = euler_from_quaternion([target1[1].pose.orientation.x, target1[1].pose.orientation.y, target1[1].pose.orientation.z, target1[1].pose.orientation.w])
                    _, _, yaw2 = euler_from_quaternion([target2[1].pose.orientation.x, target2[1].pose.orientation.y, target2[1].pose.orientation.z, target2[1].pose.orientation.w])
                    _, _, yaw3 = euler_from_quaternion([target3[1].pose.orientation.x, target3[1].pose.orientation.y, target3[1].pose.orientation.z, target3[1].pose.orientation.w])
                    avg_yaw = (yaw1 + yaw2 + yaw3) / 3.0
                    
                    # 设置目标位置
                    self.target_posestamped.pose.position = Point(x=self.target_posestamped.pose.position.x + (avg_x-self.target_posestamped.pose.position.x)*forward_percent, 
                                                                y=self.target_posestamped.pose.position.y + (avg_y-self.target_posestamped.pose.position.y)*forward_percent, 
                                                                z=self.start_point.pose.position.z) # X,Y前进一个百分比，z不变，但把z赋值给最终目标
                    self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, self.pitch_offset, avg_yaw))
                    
                    # 清空队列
                    self.green_circle_queue = []
                    rospy.loginfo(
                            f"{NODE_NAME}: 绿色圆形位置设置为: n={avg_x:.2f}m, e={avg_y:.2f}m, d={self.target_depth}m,yaw={np.degrees(avg_yaw)}°")
                    return True
                        
            self.green_circle_queue.pop(0)
        return False

    def rotate360(self) -> bool:
        """
        原地旋转一圈
        通过实际旋转角度判断是否完成一圈
        统一使用move_to_target方法来发布目标姿态
        """
        # 获取当前yaw角
        current_pose = self.get_current_pose()
        if current_pose is None:
            return False # 获取当前位姿失败

        _, _, current_yaw = euler_from_quaternion([
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ])

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
        # rospy.loginfo(self.total_rotated)
        self.last_yaw = current_yaw

        # 设置新的目标姿态
        self.rotate_target_yaw = current_yaw + (self.step_rad * self.direction)
        self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, self.pitch_offset, self.rotate_target_yaw))

        # 发布新姿态
        self.move_to_target(max_xyz_dist=0.2, max_yaw_step=self.step_rad, max_yaw_dist=np.radians(1))
        if current_yaw is not None:
            rospy.loginfo_throttle(1,f"{NODE_NAME}: 目标航向: {np.degrees(self.rotate_target_yaw)}度, 当前航向: {np.degrees(current_yaw)}度,delta yaw {np.degrees(delta_yaw)}")
        else:
            rospy.loginfo_throttle(1, f"{NODE_NAME}: 目标航向: {np.degrees(self.rotate_target_yaw)}度")

        # 记录旋转进度
        progress = (self.total_rotated / (2 * np.pi)) * 100
        rospy.loginfo_throttle(1, f"{NODE_NAME}: 旋转进度: {progress:.1f}%")

        # 判断是否完成一圈旋转（360度）
        if self.total_rotated >= 2 * np.pi:
            rospy.loginfo(f"{NODE_NAME}: 完成360度旋转 {np.degrees(self.total_rotated)},{np.degrees(np.pi)}")
            # 清理旋转相关变量
            del self.rotate_start_yaw
            del self.rotate_target_yaw
            del self.last_yaw
            del self.total_rotated
            return True
        return False

    def control_light(self,light1:int,light2:int):
        """
        移动到目标位置，并控制led灯开关，发布20次后停止
        Parameters:
            light1: int, 红色LED
            light2: int, 绿色LED
        """
        if self.move_to_target(max_xyz_dist=0.2,max_yaw_dist=np.radians(5)): # 移动到目标点后
            if self.pub_num < 20:
                self.sensor[2] = 255 # 保持舵机关闭
                self.sensor[0] = light1
                self.sensor[1] = light2
                self.control_device()
                self.pub_num += 1
                return False # 到了目标点但还没发布完
            elif self.pub_num < 40: # 发布20次后停止
                self.sensor[2] = 255 # 保持舵机关闭
                self.sensor[0] = 0 # 关闭红灯
                self.sensor[1] = 0 # 关闭绿灯
                self.control_device()
                self.pub_num += 1
                return False # 到了目标点但还没发布完
            self.pub_num = 0 # 重置发布次数
            return True # 到了目标点且发布完了
        else: # 还没到达目标点
            return False

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
    
    def arrive_end(self,max_xyz_dist=0.5,max_yaw_dist=np.radians(15)):
        """
        判断是否到达终点
        通过当前位姿和目标位姿的距离判断

        Parameters:
            max_xyz_dist: 最大位置误差，用于判断是否到达
            max_yaw_dist: 最大航向误差，用于判断是否到达
        """
        current_pose = self.get_current_pose()
        if current_pose is None:
            return False
        return self.is_arrival(current_pose=current_pose,target_pose=self.end_point, max_xyz_dist=max_xyz_dist, max_yaw_dist=max_yaw_dist)
    ###############################################逻辑层#################################

    ###############################################主循环#################################
    def run(self):
        """主循环"""
        self.step = 0
        self.sensor[2] = 255 # 保持舵机关闭
        self.sensor[0] = 0 # 关闭红灯
        self.sensor[1] = 0 # 关闭绿灯
        self.sensor[3] = 0 # 关闭补光灯1
        self.sensor[4] = 0 # 关闭补光灯2
        self.control_device()
        # self.target_depth = self.target_depth + 0.07
        while not rospy.is_shutdown():
            # rospy.loginfo(self.target_posestamped)
            if self.step == 0:  # 移动到初始位置
                if self.move_to_init_pose(): #如果移动到初始位置，开始轨迹跟踪
                    self.step = 1
                    rospy.loginfo("task4 node: 到达初始位置，开始跟踪轨迹")
            elif self.step == 1:  # 跟踪轨迹
                # TODO 根据实际顺序调整检测顺序
                if self.detect_green_circle(forward_percent=0.7) and not self.done[0] == 1: # 如果检测到绿色圆形目标点，则进入5
                    self.step = 5
                if self.detect_black_rectangle(forward_percent=0.7) and not self.done[1] == 1: # 如果检测到黑色方形目标点，则进入4
                    self.step = 4
                if self.detect_yellow_triangle(forward_percent=0.7) and not self.done[2] == 1: # 如果检测到黄色三角形目标点，则进入3
                    self.step = 3
                if self.arrive_end(max_xyz_dist=0.5,max_yaw_dist=np.radians(20)):
                    self.step = 7 # 如果到达终点，进入7
                if self.move_to_target(max_xyz_dist=0.15,max_yaw_dist=np.radians(2)): # 移动到目标点，不管到没到达，都开始下一次搜索，如果搜索到了就会回来，否则就会继续搜索，
                    self.step = 2
                
                # if self.track_count==3:
                #     self.step = 6
                #     continue
                # self.follow_track3()  # 跟踪轨迹
                # if len(self.yellow_triangle_queue) > 0 or len(self.black_rectangle_queue) > 0 or len(self.green_circle_queue) > 0:
                #     self.step = 2
            elif self.step == 2:  # 移动到目标位置
                rospy.loginfo_throttle(10, f"{NODE_NAME}: {self.done}")
                if self.done[0]==1 and self.done[1]==1 and self.done[2]==1:  # 如果所有任务都完成了
                    self.step = 7 # 进入完成任务步骤
                    rospy.loginfo("task4 node: 所有任务完成，进入完成任务步骤")
                if self.follow_track(max_rotate_rad=np.radians(25),rotate_step = np.radians(0.5),forward_percent=0.9): # 原地搜索轨迹，找到目标返回True，否则原地搜索
                    self.step = 1 # NOTE 扩大搜索角度
                # if len(self.yellow_triangle_queue) > 0 and self.round==False:
                #     rospy.loginfo("task4 node: 找到黄色三方形目标点，开始移动到目标位置")
                #     if self.detecte_yellow_triangle():
                #         self.step = 3
                #         if self.move_to_target():
                #             # self.step = 3
                #             pass  # 测试改动
                # if len(self.black_rectangle_queue) > 0 and self.red_count==0:
                #     rospy.loginfo("task4 node: 找到黑色方形目标点，开始移动到目标位置")
                #     if self.detecte_black_rectangle():
                #         self.step = 4
                #         if self.move_to_target():
                #             # self.step = 4
                #             pass  # 测试改动
                # if len(self.green_circle_queue) > 0 and self.green_count==0:
                #     rospy.loginfo("task4 node: 找到绿色圆形目标点，开始移动到目标位置")
                #     if self.detecte_green_circle():
                #         self.step = 5
                #         if self.move_to_target():
                #             # self.step = 5
                #             pass  # 测试改动
                # else:
                #     self.step = 1  # 防止误判进入后出不去
            elif self.step == 3:
                # if self.rotate360():
                if self.move_to_target(max_xyz_dist=0.15,max_yaw_dist=np.radians(1)):
                    self.step = 6 # 搜索目标
                # if True:
                #     rospy.loginfo("task4 node: 旋转360度")
                #     self.yellow_triangle_queue.clear()  # 清空,防止有多余的检测点造成误判
                #     self.round = True
                #     self.step = 1
                #     self.track_count += 1
            elif self.step == 4:
                if self.control_light(1, 0):  # 点亮红灯
                    self.done[1] = 1 # 红色方形任务完成
                    rospy.loginfo("task4 node: 点亮红灯")
                    # self.black_rectangle_queue.clear()  # 清空,防止有多余的检测点造成误判
                    self.step = 2
                    # self.track_count += 1
                    # rospy.sleep(2) # 非阻塞延时亮灯2s
            elif self.step == 5:
                if self.control_light(0,1):
                    rospy.loginfo("task4 node: 点亮绿灯")
                    self.done[0] = 1 # 绿色圆形任务完成
                    # self.green_circle_queue.clear()  # 清空,防止有多余的检测点造成误判
                    self.step = 2
                    # self.track_count += 1
                    # rospy.sleep(2) # 非阻塞延时亮灯2s
            elif self.step == 6:
                if self.rotate360():
                    rospy.loginfo("task4 node: 旋转360度完成")
                    self.done[2] = 1 # 黄色三角形任务完成
                    self.step = 2
            elif self.step == 7:
                self.finish_task()
                break
            self.rate.sleep()
    ###############################################主循环#################################

if __name__ == '__main__':
    rospy.init_node(f'{NODE_NAME}', anonymous=True)
    try:
        node = Task4Node()
        node.run()
    except rospy.ROSInterruptException: 
        pass