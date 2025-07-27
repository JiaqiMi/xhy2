#!/usr/bin/env python3
"""
sensor_node.py
监听：/auv_control (Control.msg)
通过TCP发送控制报文（红色LED、绿色LED、舵机角度）
作者：黄思旭
"""

import socket
import rospy
import sys
import os
import threading
from auv_control.msg import Control

class SensorTCPClient:
    def __init__(self):
        # 获取参数服务器的IP和端口，默认192.168.1.115:5066
        self.ip = rospy.get_param('~sensor_ip', '192.168.1.115')
        self.port = rospy.get_param('~sensor_port', 5064)
        self.server_addr = (self.ip, self.port)
        self.sock = None
        self.connect()
        rospy.Subscriber('/auv_control', Control, self.control_callback)
        rospy.loginfo(f"sensor_node started, TCP to {self.ip}:{self.port}")

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(self.server_addr)
        self.sock.settimeout(1)

    def build_packet(self, led_red, led_green, servo):
        """
        构造54字节控制报文
        只设置41(红LED), 42(绿LED), 46(舵机角度)，其他全0
        """
        packet = bytearray(54)
        packet[0:2] = b'\xFE\xFE'      # 报文头
        packet[41] = led_red           # 红色LED
        packet[42] = led_green         # 绿色LED
        packet[46] = servo             # 舵机角度
        # 校验和（0~50字节异或）
        xor = 0
        for i in range(51):
            xor ^= packet[i]
        packet[51] = xor
        packet[52:54] = b'\xFD\xFD'    # 报文尾
        return packet

    def control_callback(self, msg):
        try:
            if msg.enable: # 代表当前帧为外设控制帧
                packet = self.build_packet(msg.led_red, msg.led_green, msg.servo)
                self.sock.sendall(packet)
                rospy.loginfo(f"Sent: LED_R={msg.led_red}, LED_G={msg.led_green}, Servo={msg.servo}")
        except Exception as e:
            rospy.logerr(f"Send failed: {e}")

    def spin(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node('sensor_node')
    try:
        client = SensorTCPClient()
        client.spin()
    except rospy.ROSInterruptException:
        pass

