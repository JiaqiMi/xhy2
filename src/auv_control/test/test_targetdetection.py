#! /home/xhy/xhy_env36/bin/python
# -*- coding: utf-8 -*-

import rospy
from auv_control.msg import TargetDetection
from geometry_msgs.msg import PoseStamped
def target_publisher():
    # 初始化ROS节点
    rospy.init_node('target_detection_test', anonymous=True)
    
    # 创建发布者，发布到/obj/target_message话题
    pub = rospy.Publisher('/obj/target_message', TargetDetection, queue_size=10)
    
    # 设置发布频率为5Hz
    rate = rospy.Rate(5)
    
    # 创建一个固定的TargetDetection消息
    msg = TargetDetection()
    msg.pose = PoseStamped()
    msg.pose.header.frame_id = "camera"  # 设置坐标系
    msg.pose.pose.position.x = 1.0  # 目标在x轴的位置
    msg.pose.pose.position.y = 0.5  # 目标在y轴的位置
    msg.pose.pose.position.z = 2.0  # 目标在z轴的位置
    msg.pose.pose.orientation.w = 1.0
    msg.conf = 0.95  # 检测概率
    msg.class_name = "red"  # 目标类别ID
    
    while not rospy.is_shutdown():
        # 更新时间戳
        msg.pose.header.stamp = rospy.Time.now()
        
        # 发布消息
        pub.publish(msg)
        
        # 按照设定的频率休眠
        rate.sleep()

if __name__ == '__main__':
    try:
        target_publisher()
    except rospy.ROSInterruptException:
        pass
