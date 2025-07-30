# 检测aruco并正确的将其航向转换为map下的航向
"""
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
from queue import Queue
from threading import Lock

NODE_NAME = "aruco_detector"

class testaAruco:
    def __init__(self):
        self.lock = Lock()
        self.queue = []
        self.target_detection_sub = rospy.Subscriber("/aruco/pose", PoseStamped, self.target_detection_callback)
        self.target_posestamped = PoseStamped() # 记录最终目标位置
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10) # 发布期望位姿话题
        self.control_pub = rospy.Publisher('/control', Control, queue_size=10) # 发布控制话题
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10) # 发布任务完成标志话题
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(5)  # 运行频率5Hz
        self.step = 0 # 程序运行阶段
        self.search_direction = 1
        self.init_yaw = None
        self.pitch_offset = 1.5 # 固定1.5°俯仰
        # 获取检测目标
        # self.target_color = rospy.get_param('/task2_target_class', 'red')  # 目标颜色，默认为红色   
        self.target_color = "circle"
    ############################################### 驱动层 #########################################    
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
            rospy.loginfo(f"{NODE_NAME}: 移动到目标点: 距离={dist_to_target:.3f}米, 航向差={np.degrees(yaw_to_target):.2f}度")
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
         
    ############################################### 驱动层 #########################################                           
    

    ############################################### 回调层 #########################################
    def target_detection_callback(self, msg: TargetDetection):
        """
        收到目标检测消息，将消息加入队列，不做操作
        存的时候就应该存减去夹爪之后的位置
        """
        with self.lock:
            pose_in_camera = msg
            pose_in_map = self.tf_listener.transformPose("map", pose_in_camera)
            roll,pitch,yaw = euler_from_quaternion([
                pose_in_map.pose.orientation.x,
                pose_in_map.pose.orientation.y,
                pose_in_map.pose.orientation.z,
                pose_in_map.pose.orientation.w
            ])
            rospy.loginfo(f"{NODE_NAME}: 目标检测到aruco标记, 位置={pose_in_map.pose.position}, 欧拉角=({np.degrees(roll)}, {np.degrees(pitch)}, {np.degrees(yaw)})")
            self.queue.append([roll,pitch,yaw])
    ############################################### 回调层 #########################################


    ############################################### 逻辑层 #########################################
    
    ############################################### 逻辑层 #########################################


    ############################################### 主循环 #########################################
    def run(self):
        # 先运动到一个目标点，然后运行search2，直接开始搜索
        while not rospy.is_shutdown():
            self.rate.sleep()
    ############################################### 主循环 #########################################

if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        tester = testaAruco()
        tester.run()
    except rospy.ROSInterruptException:
        pass