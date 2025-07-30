#! /home/xhy/xhy_env36/bin/python
"""
名称: keyboard_control.py
功能: 键盘控制节点
描述:
    1. 手动设置运行阶段 
        0 不切换 
        1 过门
        2 巡线
        3 作业
        4 夹球
        5 上浮
    2. 开启自动运行 g/l
作者：黄思旭
记录：
2025.7.16 16:35
    1. 添加了键盘控制节点，支持手动设置运行阶段和开启自动运行
    2. 修改了Keyboard.msg，添加了run字段，表示是否开启自动运行
"""
import rospy
from auv_control.msg import Keyboard
import sys
import termios
import tty
import atexit
"""
Key.msg
bool run
uint8 mode
AUVData force

"""

"""
AUVMotor.msg
int16 TX 
int16 TY
int16 TZ
int16 MX 
int16 MY
int16 MZ
"""

class KeyboardControlNode:
    def __init__(self):
        rospy.init_node('keyboard_control_node')
        self.pub = rospy.Publisher('/auv_keyboard', Keyboard, queue_size=10)
        self.rate = rospy.Rate(10)
        self.running = 0 # 默认模式为0（待机）
        rospy.loginfo("键盘控制AUV状态:")
        rospy.loginfo("0-5: 手动设置运行阶段（仅非自动运行可用）")
        rospy.loginfo("g: 开启自动运行")
        rospy.loginfo("l: 停止自动运行")
        rospy.loginfo("CTRL+C: 退出")

    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key

    def run(self):
        """
        run = 1时，不关注mode是多少
        run = 0时，mode发挥作用
        run = 2时，退出自动运行
        """
        while not rospy.is_shutdown():
            key = self.get_key() # 这个是阻塞的，直到有输入
            rospy.loginfo(f"keyboard: input {key}")
            msg = Keyboard()
            # 只设置状态相关字段
            msg.run = self.running # 初始化为当前运行状态
            msg.mode = 0 # 初始化为不切换状态
            if key == 'g': # 切换自动运行状态，并将runing置1
                self.running = not self.running
                msg.run = 1
                
                self.pub.publish(msg)
            elif key == 'l':
                self.running = 0 # 停止自动运行
                msg.run = 2
                self.pub.publish(msg)
            elif key in '012345':
                msg.mode = int(key)
                self.pub.publish(msg)
            elif key == '\x03':  # CTRL+C
                break 
            self.rate.sleep()

def restore_terminal():
    fd = sys.stdin.fileno()
    termios.tcsetattr(fd, termios.TCSADRAIN, termios.tcgetattr(fd))

if __name__ == '__main__':
    atexit.register(restore_terminal)
    try:
        node = KeyboardControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass