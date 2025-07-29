#! /home/xhy/xhy_env36/bin/python
"""
测试下潜、悬停
2025.7.22 11:48
    定点测试 通过
"""

import rospy 
from std_msgs.msg import String
import tf2_ros
from geometry_msgs.msg import PoseStamped,Quaternion
import tf
import numpy as np
DEPTH = rospy.get_param('~depth', 0.3)  # 下潜深度，单位米
YAW = rospy.get_param('~yaw', 0)  # 航向，单位度

class test1:
    def __init__(self):
        rospy.init_node('test1_node')
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10)  # 发布任务完成消息
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10)  # 发布目标点
        self.Rate = rospy.Rate(5)
        self.step = 0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def run(self):
        i = 50
        while not rospy.is_shutdown():  # 主循环
            # 每次循环获取当前的tf变换
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0), rospy.Duration(1.0))
            quat = trans.transform.rotation
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
            rospy.loginfo_throttle(2, f"test1: 当前姿态: roll={np.degrees(roll)}, pitch={np.degrees(pitch)}, yaw={np.degrees(yaw)}")
            self.depth = trans.transform.translation.z 
            if self.step == 0:  # 步骤0 原地不动获取初始位置和航向（通过tf变换）
                try:
                    # map是东北天坐标系
                    init_x = trans.transform.translation.x
                    init_y = trans.transform.translation.y
                    init_depth = trans.transform.translation.z
                    target_depth = DEPTH + init_depth
                    target_yaw = yaw
                    rospy.loginfo( f"test1: 初始位置: x={init_x}, y={init_y}, z={init_depth}, yaw={np.degrees(target_yaw)}")
                    self.step = 1  # 获取到初始位置后，进入步骤1
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    rospy.logwarn("test1: 无法获取初始位置的tf变换")
            if self.step == 1:  # 步骤1 原地下潜
                if self.depth < target_depth and i > 25:
                    rospy.loginfo(f"test1: 下潜中，当前深度={self.depth}, 目标深度={target_depth}")
                    # 发布目标点
                    target_msg = PoseStamped()
                    target_msg.header.stamp = rospy.Time.now()
                    target_msg.header.frame_id = "map"
                    target_msg.pose.position.x = init_x # 原地下潜
                    target_msg.pose.position.y = init_y  # 原地下潜
                    target_msg.pose.position.z = target_depth
                    target_msg.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, target_yaw))
                    self.target_pub.publish(target_msg)
                    i -= 1
                else:
                    rospy.loginfo_throttle(2, f"test1: 已下潜到目标深度={target_depth}")
                    self.step = 2
            if self.step == 2:  # 步骤2 原地悬停
                # if i > 0:  # 继续发送运动控制帧
                target_msg = PoseStamped()
                target_msg.header.stamp = rospy.Time.now()
                target_msg.header.frame_id = "map"
                target_msg.pose.position.x = init_x  # 原地悬停
                target_msg.pose.position.y = init_y  # 原地悬停
                target_msg.pose.position.z = target_depth
                target_msg.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, target_yaw))
                rospy.loginfo_throttle(2, f"test1: 悬停中，当前深度={self.depth}, 目标深度={target_depth}")
                self.target_pub.publish(target_msg)
                #  i -= 1
                #else:
                #     self.step = 3

            if self.step == 3:  # 步骤2 原地悬停
                # if i > 0:  # 继续发送运动控制帧
                target_msg = PoseStamped()
                target_msg.header.stamp = rospy.Time.now()
                target_msg.header.frame_id = "map"
                target_msg.pose.position.x = init_x  # 原地悬停
                target_msg.pose.position.y = init_y  # 原地悬停
                target_msg.pose.position.z = target_depth
                target_msg.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(0, 0, target_yaw))
                rospy.loginfo_throttle(2, f"test1: 悬停中，当前深度={self.depth}, 目标深度={target_depth}")
                self.target_pub.publish(target_msg)
                self.step = 4
            if self.step == 4:  # 步骤3 上浮
                if i < 5:
                    target_msg = PoseStamped()
                    target_msg.header.stamp = rospy.Time.now()
                    target_msg.header.frame_id = "map"
                    target_msg.pose.position.x = init_x  # 原地上浮
                    target_msg.pose.position.y = init_y  # 原地上浮
                    target_msg.pose.position.z = init_depth
                    target_msg.pose.orientation = tf.transformations.quaternion_from_euler(0, 0, target_yaw)
                    self.target_pub.publish(target_msg)
                    i += 1
                else:
                    self.step = 5
            if self.step == 5:
                self.finish_task()
            self.Rate.sleep()
    
    def finish_task(self):
        """完成任务的清理工作"""
        self.finished_pub.publish(String(data="finished"))
        rospy.loginfo("test: 任务完成，发布完成消息")
        rospy.signal_shutdown("任务完成")
        # 这里可以添加任何需要的清理代码，比如关闭文件、释放资源等
        # 目前没有额外资源需要清理
       
if __name__ == "__main__":
    try:
        handler = test1()
        handler.run()
    except rospy.ROSInterruptException:
        pass