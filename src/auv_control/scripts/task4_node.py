#! /home/xhy/xhy_env36/bin/python
"""
名称: task4_node.py
功能: 巡线 作业
作者: buyegaid
监听：/target_detection (来自视觉节点) 检测目标对应颜色的标志
      /tf (来自tf树)
发布：/auv_control (Control.msg) 被sensor_driver订阅
      /finished (String) 被state_control订阅, 代表任务是否完成
      /target (PoseStamped.msg) 被tf_handler订阅, 代表目标位置

记录：
"""


import rospy
from std_msgs.msg import String

class Task4Node:
    def __init__(self):
        rospy.init_node('task4_node')
        self.pub = rospy.Publisher('/finished', String, queue_size=10)
        self.rate = rospy.Rate(5)  # 5Hz
        self.count = 1

    def run(self):
        while not rospy.is_shutdown():
            if self.count <= 10:
                msg = String(data=str(self.count))
                self.pub.publish(msg)
                self.count += 1
            elif self.count == 11:
                self.pub.publish(String(data="finished"))
                rospy.signal_shutdown("Task finished")
                break
            self.rate.sleep()

if __name__ == '__main__':
    node = Task4Node()
    node.run()