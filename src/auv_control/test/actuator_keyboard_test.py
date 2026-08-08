#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
名称：actuator_keyboard_test.py
功能：通过键盘保持执行器测试状态，并以5Hz发布完整执行器控制指令
作者：buyegaid
监听：键盘标准输入
发布：/cmd/actuator (auv_control/ActuatorControl)
记录：
2026.8.5
    新增红黄绿灯、推杆和开合舵机的保持式键盘测试控制。
"""

import atexit
import os
import select
import sys
import termios
import tty

import rospy

from auv_control.msg import ActuatorControl


PUBLISH_RATE_HZ = 5.0
DRIVE_STOP = 0
DRIVE_FORWARD = 1
DRIVE_REVERSE = 2


def clamp_uint8(value):
    """将参数限制到 uint8 的有效范围。"""
    return max(0, min(255, int(value)))


class ActuatorCommandState:
    """保存键盘选择的执行器状态，并生成完整控制消息。"""

    def __init__(
            self, drive_speed=250, heading_servo=128,
            clamp_open=0, clamp_closed=255):
        self.drive_speed_setting = max(0, min(254, int(drive_speed)))
        self.heading_servo = clamp_uint8(heading_servo)
        self.clamp_open = clamp_uint8(clamp_open)
        self.clamp_closed = clamp_uint8(clamp_closed)

        self.clamp_servo = self.clamp_closed
        self.drive_cmd = DRIVE_STOP
        self.drive_speed = 0
        self.red_light = 0
        self.yellow_light = 0
        self.green_light = 0

    def apply_key(self, key):
        """应用一个按键；返回状态说明，无效按键返回 None。"""
        key = key.lower()
        if key == 'r':
            self.red_light = 0 if self.red_light else 1
            return '红灯{}'.format('亮' if self.red_light else '灭')
        if key == 'y':
            self.yellow_light = 0 if self.yellow_light else 1
            return '黄灯{}'.format('亮' if self.yellow_light else '灭')
        if key == 'g':
            self.green_light = 0 if self.green_light else 1
            return '绿灯{}'.format('亮' if self.green_light else '灭')
        if key == 'w':
            self.drive_cmd = DRIVE_FORWARD
            self.drive_speed = self.drive_speed_setting
            return '推杆前进，速度={}'.format(self.drive_speed)
        if key == 's':
            self.drive_cmd = DRIVE_REVERSE
            self.drive_speed = self.drive_speed_setting
            return '推杆后退，速度={}'.format(self.drive_speed)
        if key == 'x':
            self.stop_drive()
            return '推杆停止'
        if key == 'a':
            self.clamp_servo = self.clamp_open
            return '舵机全开，位置={}'.format(self.clamp_servo)
        if key == 'd':
            self.clamp_servo = self.clamp_closed
            return '舵机全闭，位置={}'.format(self.clamp_servo)
        return None

    def stop_drive(self):
        """停止推杆并清零速度。"""
        self.drive_cmd = DRIVE_STOP
        self.drive_speed = 0

    def set_safe_state(self):
        """切换到退出时使用的安全状态。"""
        self.stop_drive()
        self.red_light = 0
        self.yellow_light = 0
        self.green_light = 0

    def build_message(self):
        """构造 mode=2 的完整执行器控制消息。"""
        message = ActuatorControl()
        message.mode = 2
        message.light1 = 0
        message.light2 = 0
        message.heading_servo = self.heading_servo
        message.clamp_servo = self.clamp_servo
        message.drive_cmd = self.drive_cmd
        message.drive_speed = self.drive_speed
        message.red_light = self.red_light
        message.yellow_light = self.yellow_light
        message.green_light = self.green_light
        return message


class ActuatorKeyboardTestNode:
    """读取终端单键并以固定5Hz发布保持式执行器指令。"""

    def __init__(self):
        rospy.init_node('actuator_keyboard_test')
        self.command_topic = rospy.get_param(
            '~command_topic', '/cmd/actuator')
        self.state = ActuatorCommandState(
            drive_speed=rospy.get_param('~drive_speed', 250),
            heading_servo=rospy.get_param('~heading_servo', 128),
            clamp_open=rospy.get_param('~clamp_open', 0),
            clamp_closed=rospy.get_param('~clamp_closed', 255),
        )
        self.publisher = rospy.Publisher(
            self.command_topic, ActuatorControl, queue_size=1)
        self.rate = rospy.Rate(PUBLISH_RATE_HZ)

        self.input_stream = None
        self.file_descriptor = None
        self.terminal_settings = None
        self._prepare_terminal()
        atexit.register(self.restore_terminal)
        rospy.on_shutdown(self.publish_safe_state)

        rospy.loginfo(
            '执行器键盘测试已启动，固定5Hz发布到%s：'
            'r=红灯，y=黄灯，g=绿灯，w=推杆前进，s=推杆后退，'
            'x=推杆停止，a=舵机全开，d=舵机全闭；按键状态保持。',
            self.command_topic)

    def _prepare_terminal(self):
        """设置单键输入；失败时记录错误并由主循环继续重试。"""
        if self.file_descriptor is not None:
            return True
        try:
            if sys.stdin.isatty():
                self.input_stream = sys.stdin
            else:
                self.input_stream = open('/dev/tty', 'rb', buffering=0)
            self.file_descriptor = self.input_stream.fileno()
            self.terminal_settings = termios.tcgetattr(self.file_descriptor)
            tty.setcbreak(self.file_descriptor)
            return True
        except (OSError, ValueError, termios.error) as error:
            self._close_owned_input_stream()
            self.file_descriptor = None
            self.terminal_settings = None
            rospy.logerr_throttle(
                5.0, '执行器键盘测试无法访问交互终端，将继续重试：%s', error)
            return False

    def _close_owned_input_stream(self):
        """关闭节点自行打开的 /dev/tty，不关闭标准输入。"""
        if self.input_stream is not None and self.input_stream is not sys.stdin:
            try:
                self.input_stream.close()
            except OSError:
                pass
        self.input_stream = None

    def get_key(self):
        """非阻塞读取一个按键；没有输入或读取失败时返回 None。"""
        if not self._prepare_terminal():
            return None
        try:
            readable, _, _ = select.select(
                [self.file_descriptor], [], [], 0)
            if not readable:
                return None
            key = os.read(self.file_descriptor, 1)
            if not key:
                return None
            return key.decode('utf-8', errors='ignore') or None
        except (OSError, ValueError) as error:
            rospy.logerr_throttle(5.0, '读取键盘失败，将继续重试：%s', error)
            self.restore_terminal()
            return None

    def restore_terminal(self):
        """恢复节点启动前的终端设置。"""
        if self.terminal_settings is not None and self.file_descriptor is not None:
            try:
                termios.tcsetattr(
                    self.file_descriptor, termios.TCSADRAIN,
                    self.terminal_settings)
            except (OSError, termios.error):
                pass
        self.terminal_settings = None
        self.file_descriptor = None
        self._close_owned_input_stream()

    def publish_command(self):
        """发布当前保持的完整执行器状态。"""
        try:
            self.publisher.publish(self.state.build_message())
        except Exception as error:
            rospy.logerr_throttle(
                5.0, '执行器控制指令发布失败，将继续重试：%s', error)

    def publish_safe_state(self):
        """退出前发送推杆停止、三色灯全灭的安全指令。"""
        self.state.set_safe_state()
        self.publish_command()
        self.restore_terminal()

    def run(self):
        """保持5Hz处理键盘并发布控制指令。"""
        while not rospy.is_shutdown():
            key = self.get_key()
            if key == '\x03':
                rospy.signal_shutdown('收到 Ctrl-C')
                continue
            if key is not None:
                description = self.state.apply_key(key)
                if description is None:
                    rospy.logwarn(
                        '无效按键 %r；可用按键：r、y、g、w、s、x、a、d。',
                        key)
                else:
                    rospy.loginfo('执行器状态切换：%s。', description)
            self.publish_command()
            try:
                self.rate.sleep()
            except rospy.ROSInterruptException:
                break


if __name__ == '__main__':
    try:
        ActuatorKeyboardTestNode().run()
    except rospy.ROSInterruptException:
        pass
