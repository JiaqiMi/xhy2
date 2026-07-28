#! /home/xhy/xhy_env36/bin/python
# -*- coding: utf-8 -*-
"""
名称：keyboard_control.py
功能：发布任务启动与停止指令的键盘控制节点
作者：黄思旭
监听：键盘标准输入
发布：/auv_keyboard
记录：
2025.7.16
    添加键盘控制节点，支持手动设置运行阶段和自动运行。
2026.7.28
    调整为 task1、task2、task3 的直接启动与停止控制。
2026.7.28
    恢复 g 键启动自动串联任务。
"""

import atexit
import sys
import termios
import tty

import rospy
from auv_control.msg import Keyboard


class KeyboardControlNode:
    """将键盘输入转换为状态监控节点可识别的任务控制消息。"""

    def __init__(self):
        rospy.init_node('keyboard_control_node')
        self.publish_topic = rospy.get_param('~publish_topic', '/auv_keyboard')
        self.rate = rospy.Rate(rospy.get_param('~rate_hz', 10.0))
        self.publisher = rospy.Publisher(
            self.publish_topic, Keyboard, queue_size=10)

        rospy.loginfo(
            '键盘任务控制已启动：1=task1，2=task2，3=task3，'
            'g=自动串联，0/l=停止当前任务。')

    @staticmethod
    def get_key():
        """读取一个按键，并在读取后恢复终端设置。"""
        file_descriptor = sys.stdin.fileno()
        old_settings = termios.tcgetattr(file_descriptor)
        try:
            tty.setraw(file_descriptor)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(
                file_descriptor, termios.TCSADRAIN, old_settings)

    def publish_task(self, mode):
        """发布指定任务的启动指令。"""
        message = Keyboard()
        message.run = 0
        message.mode = mode
        self.publisher.publish(message)
        rospy.loginfo('请求启动 task%s。', mode)

    def stop_task(self):
        """发布当前任务的停止指令。"""
        message = Keyboard()
        message.run = 2
        message.mode = 0
        self.publisher.publish(message)
        rospy.loginfo('请求停止当前任务。')

    def start_automatic_tasks(self):
        """发布从 task1 开始的自动串联任务指令。"""
        message = Keyboard()
        message.run = 1
        message.mode = 0
        self.publisher.publish(message)
        rospy.loginfo('请求从 task1 开始自动串联任务。')

    def run(self):
        """持续读取键盘指令。"""
        while not rospy.is_shutdown():
            key = self.get_key()
            if key in '123':
                self.publish_task(int(key))
            elif key == 'g':
                self.start_automatic_tasks()
            elif key in '0l':
                self.stop_task()
            elif key == '\x03':
                break
            else:
                rospy.logwarn('无效按键 %r；可用按键：1、2、3、g、0、l。', key)
            self.rate.sleep()


def restore_terminal():
    """进程结束时尽力恢复终端属性。"""
    try:
        file_descriptor = sys.stdin.fileno()
        termios.tcsetattr(
            file_descriptor, termios.TCSADRAIN,
            termios.tcgetattr(file_descriptor))
    except termios.error:
        pass


if __name__ == '__main__':
    atexit.register(restore_terminal)
    try:
        KeyboardControlNode().run()
    except rospy.ROSInterruptException:
        pass
