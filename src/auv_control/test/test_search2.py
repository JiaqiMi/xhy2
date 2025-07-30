#! /home/xhy/xhy_env/bin/python
"""
名称: test_search2.py
功能：测试AUV原地搜索功能
作者: buyegaid
功能点：
    1. 原地缓慢旋转
    2. 收到三包连续有效数据后计算位置运动
    3. 不用优先队列，用普通队列就行，保证顺序处理
2025.07.30 15:30
    第一版完成   
"""
import rospy
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String
from auv_control.msg import TargetDetection, Control
from geometry_msgs.msg import PoseStamped, Quaternion, Point
import numpy as np
from threading import Lock


NODE_NAME = "search2_node"

class Search2:
    def __init__(self):
        self.lock = Lock()
        self.queue = [] # 队列大小为100
        self.target_detection_sub = rospy.Subscriber("/obj/target_message", TargetDetection, self.target_detection_callback)
        self.target_posestamped = PoseStamped() # 记录最终目标位置
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10) # 发布期望位姿话题
        self.control_pub = rospy.Publisher('/control', Control, queue_size=10) # 发布控制话题
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10) # 发布任务完成标志话题
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(5)  # 运行频率5Hz
        self.step = 0 # 程序运行阶段
        self.search_direction = 1
        self.init_yaw = None
        self.pitch_offset = np.radians(1.5) # 固定1.5°俯仰
        # 获取检测目标
        # self.target_color = rospy.get_param('/task2_target_class', 'red')  # 目标颜色，默认为红色   
        self.target_color = "black"
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
        
        Parameters:
            ori1: Quaternion 第一个航向
            ori2: Quaternion 第二个航向

        Returns:
            float 两个航向之间的差值(绝对值)
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
    
    def xyz_distance(self,p1:Point, p2:Point):
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

    def move_to_target(self,max_xyz_dist=0.2,max_yaw_step=np.radians(5),max_yaw_dist=np.radians(0.2)):
        """
        发送一次指令移动到目标位姿，通过生成平滑路径点实现

        Parameters:
            max_xyz_dist: float, 最大位置误差(米)
            max_yaw_step: float, 最大偏航角步长(弧度)
            max_yaw_dist: float, 最大航向误差(弧度)

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
            next_pose = self.generate_smooth_pose(current_pose, self.target_posestamped)
            dist_to_target = self.xyz_distance(current_pose.pose.position, self.target_posestamped.pose.position)
            yaw_to_target = self.yaw_distance(current_pose.pose.orientation, self.target_posestamped.pose.orientation)
            rospy.loginfo_throttle(2,f"{NODE_NAME}: 移动到目标点: 距离={dist_to_target:.3f}米, 航向差={np.degrees(yaw_to_target):.2f}度")
            self.target_pub.publish(next_pose)

            return False
            
        except tf.Exception as e:
            rospy.logwarn(f"{NODE_NAME}: 移动失败: {e}")
            return False
        
    def get_current_pose(self) -> PoseStamped:
        """获取当前位姿，如果获取失败，返回None"""
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
            rospy.logwarn(f"{NODE_NAME}: 获取当前位姿失败: {e}")
            return None
    
    def finish_task(self):
        """
        任务完成：
        1. 发布任务完成标志
        2. 返回True
        """
        self.finished_pub.publish(F"{NODE_NAME}finished")
        rospy.loginfo(f"{NODE_NAME}: 任务完成，发布完成消息")
        rospy.signal_shutdown("任务完成")
        return True   
         
    ###############################################驱动层#################################                           
    
    ###############################################回调层#################################
    def target_detection_callback(self, msg: TargetDetection):
        """
        收到目标检测消息，将消息加入队列，不做操作
        存的时候就应该存减去夹爪之后的位置
        """
        rospy.loginfo(f"{NODE_NAME}: 收到目标检测消息 {msg.class_name},{msg.pose.pose.position.x},{msg.pose.pose.position.y},{msg.pose.pose.position.z}")
        if msg.class_name == self.target_color:
            with self.lock:
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
                        expected_pose.pose.orientation = current_pose.pose.orientation

                        # 加入队列
                        self.queue.append((msg.conf, current_pose, expected_pose, target_in_map, target_in_base))
                        rospy.loginfo(f"{NODE_NAME}: 加入队列 (conf={msg.conf:.2f})")
                        
                    except tf.Exception as e:
                        rospy.logwarn(f"{NODE_NAME}: 坐标转换失败: {e}")
    ###############################################回调层#################################

    ###############################################逻辑层#################################
    def search_target(self, max_time_interval=5.0, max_position_interval=0.5,rotate_step=np.radians(1),max_xyz_dist=0.3,max_yaw_dist=np.radians(0.2)):
        """
        搜索目标：
        从队列中获取三个目标点，判断三个点的时间间隔和位置间隔(在map下的),如果间隔小于阈值，时间小于阈值
        则认为找到目标，将目标点更新到self.target_posestamped中，并返回True
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
            rospy.loginfo(f"{NODE_NAME}: 当前队列长度: {len(self.queue)}")
            if target1 is not None and target2 is not None and target3 is not None:
                if target1[0] > 0.5 and target2[0] > 0.5 and target3[0] > 0.5:
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
                                
                                # 设置完目标位姿后，跳转到下一步即可
                                self.target_posestamped.pose.position.x = avg_x
                                self.target_posestamped.pose.position.y = avg_y
                                self.target_posestamped.pose.position.z = avg_z
                                rospy.loginfo(f"{NODE_NAME}: 目标位置设置为: x={avg_x:.2f}, y={avg_y:.2f}, z={avg_z:.2f}")
                                self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, self.pitch_offset, avg_yaw))
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

        if next_yaw > self.init_yaw + np.radians(10):
            self.search_direction = -1
            next_yaw = current_yaw + (rotate_step * self.search_direction)
            rospy.loginfo(f"{NODE_NAME}: test search: 掉头顺时针搜索")
        elif next_yaw < self.init_yaw - np.radians(10):
            self.search_direction = 1
            next_yaw = current_yaw + (rotate_step * self.search_direction)
            rospy.loginfo(f"{NODE_NAME}: test search: 掉头逆时针搜索")
        
        # 设置目标位姿，位置不变，原地开始旋转加一个旋转角度
        self.target_posestamped.pose.orientation = Quaternion(*quaternion_from_euler(0, self.pitch_offset, next_yaw))

        # 执行一次旋转指令
        self.move_to_target(max_xyz_dist=max_xyz_dist, max_yaw_step=rotate_step, max_yaw_dist=max_yaw_dist)
        return False

    ###############################################逻辑层#################################
    def run(self):
        # 先运动到一个目标点，然后运行search2，直接开始搜索
        while not rospy.is_shutdown():
            if self.step == 0:
                if self.search_target(rotate_step=np.radians(1.5)):
                    self.step = 1
                # 同时原地进行旋转运动
            if self.step == 1:
                self.move_to_target(max_xyz_dist=0.1, max_yaw_step=np.radians(0.2))
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        tester = Search2()
        tester.run()
    except rospy.ROSInterruptException:
        pass