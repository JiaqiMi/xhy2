#! /home/xhy/xhy_env36/bin/python
"""
名称: test_search.py
功能：测试AUV原地搜索功能
作者: Assistant
功能点：
1. 使用优先队列存储检测目标，使用置信度作为权重
2. 分两个阶段进行搜索：
   - 阶段1：航向对齐，将航向垂直于物体
   - 阶段2：位置对齐，通过识别目标位置进行微调
"""

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Point
from auv_control.msg import TargetDetection
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import numpy as np
from queue import PriorityQueue
from threading import Lock

class SearchTester:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('search_tester')
        
        # 创建发布者和订阅者
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10)
        rospy.Subscriber('/target_detection', TargetDetection, self.target_detection_callback)
        
        # 设置运行频率
        self.rate = rospy.Rate(5)  # 5Hz
        
        # 初始化TF相关
        self.tf_listener = tf.TransformListener()
        
        # 初始化目标队列和线程锁
        self.target_positions_queue = PriorityQueue(maxsize=10)
        self.sequence_number = 0
        self._queue_lock = Lock()
        
        # 搜索相关参数
        self.initial_yaw = None
        self.search_direction = 1
        self.yaw_step = np.radians(3)
        self.search_phase = 1  # 搜索阶段：1为航向对齐，2为位置对齐
        self.target_yaw = rospy.get_param('~target_yaw', None)  # 从参数服务器读取目标航向
        self.aligned_yaw = None
        self.best_confidence = 0.0
        
        # 目标位姿
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = "map"
        
        # 从参数服务器获取目标颜色
        self.target_color = rospy.get_param('~task3_target_class', 'red')
        
        # 获取目标深度
        self.target_depth = rospy.get_param('~depth', 0.3)

    def numpy_distance(self, p1:Point, p2:Point):
        """计算两点间的欧氏距离"""
        a = np.array([p1.x, p1.y, p1.z])
        b = np.array([p2.x, p2.y, p2.z])
        return np.linalg.norm(a - b)

    def target_detection_callback(self, msg: TargetDetection):
        """处理目标检测消息"""
        with self._queue_lock:
            try:
                if msg.class_name == self.target_color and msg.conf > 0.6:
                    point_in_camera = msg.pose.pose.position
                    point_origin = Point(x=0,y=0,z=0)
                    if self.numpy_distance(point_in_camera, point_origin) < 5.0:
                        # 转换坐标系
                        self.tf_listener.waitForTransform("map", msg.pose.header.frame_id, msg.pose.header.stamp, rospy.Duration(1.0))
                        target_in_map = self.tf_listener.transformPose("map", msg.pose)
                        target_in_auv = self.tf_listener.transformPose("base_link", msg.pose)

                        # 添加到队列
                        if not self.target_positions_queue.full():
                            priority = -msg.conf
                            self.sequence_number += 1
                            self.target_positions_queue.put((priority, self.sequence_number, target_in_map, target_in_auv))
                            rospy.loginfo(f"test search: 添加新目标，置信度: {msg.conf:.2f}")
                        else:
                            lowest_priority, seq, map_pose, auv_pose = self.target_positions_queue.get()
                            if -msg.conf < lowest_priority:
                                self.target_positions_queue.put((lowest_priority, seq, map_pose, auv_pose))
                            else:
                                self.sequence_number += 1
                                self.target_positions_queue.put((-msg.conf, self.sequence_number, target_in_map, target_in_auv))
                                rospy.loginfo(f"test search: 更新目标位置，新置信度: {msg.conf:.2f}")
                    
            except tf.Exception as e:
                rospy.logwarn(f"test search: 坐标转换失败: {e}")

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
            rospy.logwarn(f"test search: 获取当前位姿失败: {e}")
            return None

    def search_target(self):
        """
        搜索目标的两个阶段：
        阶段1：航向对齐，将航向垂直于物体
        阶段2：位置对齐，通过识别目标位置进行微调
        """
        try:
            # 第一步，获取当前位姿
            current_pose = self.get_current_pose()
            if current_pose is None:
                rospy.logwarn("test search: 获取当前位姿失败")
                return False

            _, _, current_yaw = euler_from_quaternion([
                current_pose.pose.orientation.x,
                current_pose.pose.orientation.y,
                current_pose.pose.orientation.z,
                current_pose.pose.orientation.w
            ])

            # 阶段1：航向对齐
            if self.search_phase == 1:
                # 如果宏定义了航向，直接进入到步骤二
                if self.target_yaw is not None:
                    target_yaw = np.radians(self.target_yaw)
                    self.aligned_yaw = target_yaw
                    self.search_phase = 2
                    rospy.loginfo(f"test search: 使用参数服务器中的目标航向: {self.target_yaw}度")
                    return False

                # 如果没有初始航向，则设置为当前航向
                if self.initial_yaw is None:
                    self.initial_yaw = current_yaw
                    rospy.loginfo(f"test search: 搜索初始角度: {np.degrees(self.initial_yaw)}度")

                # 获取到有用的目标方向
                if self.target_positions_queue.qsize() > 0:
                    temp_queue = PriorityQueue()
                    max_conf = 0.0
                    best_yaw = current_yaw
                    # 计算最优航向
                    with self._queue_lock:
                        while not self.target_positions_queue.empty():
                            priority, seq, map_pose, auv_pose = self.target_positions_queue.get()
                            conf = -priority
                            if conf > max_conf:
                                max_conf = conf
                                target_dir = np.arctan2(
                                    auv_pose.pose.position.y,
                                    auv_pose.pose.position.x
                                )
                                best_yaw = current_yaw + target_dir + np.pi/2
                            temp_queue.put((priority, seq, map_pose, auv_pose))
                        
                        self.target_positions_queue = temp_queue

                    if max_conf > self.best_confidence:
                        self.best_confidence = max_conf
                        self.aligned_yaw = best_yaw
                        rospy.loginfo(f"test search: 找到更好的目标方向，置信度: {max_conf:.2f}")

                    if max_conf > 0.8:
                        self.search_phase = 2
                        rospy.loginfo("test search: 航向对齐完成，进入位置对齐阶段")
                        return False

                # 继续搜索
                next_yaw = current_yaw + (self.yaw_step * self.search_direction)

                if next_yaw > self.initial_yaw + np.radians(180):
                    self.search_direction = -1
                    next_yaw = current_yaw + (self.yaw_step * self.search_direction)
                    rospy.loginfo("test search: 掉头顺时针搜索")
                elif next_yaw < self.initial_yaw - np.radians(180):
                    self.search_direction = 1
                    next_yaw = current_yaw + (self.yaw_step * self.search_direction)
                    rospy.loginfo("test search: 掉头逆时针搜索")
            # 阶段2：位置对齐
            else:
                # 没有满，则等待
                if not self.target_positions_queue.full():
                    rospy.loginfo_throttle(1, "test search: 等待足够的目标位置数据...")
                    return False

                temp_queue = PriorityQueue()
                sum_pos = np.zeros(3)
                sum_weight = 0.0
                
                while not self.target_positions_queue.empty():
                    priority, seq, map_pose, auv_pose = self.target_positions_queue.get()
                    weight = -priority
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
                        current_pose.pose.position.x + hand_trans[0],
                        current_pose.pose.position.y + hand_trans[1],
                        current_pose.pose.position.z + hand_trans[2]
                    ])
                    pos_error = np.linalg.norm(avg_pos - current_pos)
                    
                    yaw_error = abs((self.aligned_yaw - current_yaw + np.pi) % (2 * np.pi) - np.pi)
                    if pos_error < 0.1 and yaw_error < np.radians(5):
                        rospy.loginfo("test search: 位置和航向对齐完成")
                        return True

                    rospy.loginfo_throttle(1, f"test search: 位置对齐: 误差={pos_error:.3f}米, "
                                         f"航向误差={np.degrees(yaw_error):.1f}度")
                    
                except tf.Exception as e:
                    rospy.logwarn(f"test search: 获取hand变换失败: {e}")
                    return False
            # TODO 这个算的对吗？
            
            # 设置目标姿态
            self.target_pose.header.frame_id = "map"
            self.target_pose.header.stamp = rospy.Time.now()
            # 目标位置更新
            self.target_pose.pose.positon = self.target_pose.pose.position if self.search_phase == 1 else avg_pos
            # 目标航向更新
            target_yaw = self.aligned_yaw if self.search_phase == 2 else next_yaw
            self.target_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, target_yaw))

            self.target_pub.publish(self.target_pose)
            rospy.loginfo_throttle(1, f"test search: 搜索阶段{self.search_phase}: 当前航向={np.degrees(current_yaw):.1f}度, "
                                    f"目标航向={np.degrees(target_yaw):.1f}度")
            
            return False
                
        except tf.Exception as e:
            rospy.logwarn(f"test search: 搜索失败: {e}")
            return False

    def run(self):
        """主运行循环"""
        while not rospy.is_shutdown():
            self.search_target()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        tester = SearchTester()
        tester.run()
    except rospy.ROSInterruptException:
        pass
