#! /home/xhy/xhy_env36/bin/python
"""
名称：sensor_node.py
功能：通过TCP发送控制报文（红色LED、绿色LED、舵机角度、补光灯亮度）
作者：buyegaid
监听：/sensor (Control.msg)
发布：None
记录：
2025.7.15 16:26
    只和sensor芯片通信，用于控制LED和舵机，舵机50-FF，目前只有发送功能，没有接收功能
2025.7.19 15:27
    接收Control消息，只控制舵机
2025.7.22 18:13
    循环5Hz发送控制报文
    增加有效值的判断
2025.7.27 10:00
    增加补光灯控制功能(44-45字节)，亮度通过rosparam获取
"""

import socket
import rospy
import threading
from auv_control.msg import Control

class SensorDriver:
    """
    传感器节点，用于控制LED和舵机
    """
    def __init__(self):
        # 获取参数服务器的IP和端口，默认192.168.1.115:5064
        self.ip = rospy.get_param('~sensor_ip', '192.168.1.115')
        self.port = rospy.get_param('~sensor_port', 5064)
        self.server_addr = (self.ip, self.port)
        self.sock = None
        
        # 添加LED和舵机控制变量
        self.led_red = 0 # 初始值默认关闭LED_RED
        self.led_green = 0 # 初始值默认关闭LED_GREEN
        self.servo = 255  # 初始值默认关闭夹爪255对应关夹爪，100对应开夹爪
        
        # 从参数服务器获取补光灯亮度值(0-100)
        self.light1_brightness = rospy.get_param('~light1_brightness', 0)
        self.light2_brightness = rospy.get_param('~light2_brightness', 0)
        # 确保亮度值在有效范围内
        self.light1_brightness = max(0, min(100, self.light1_brightness))
        self.light2_brightness = max(0, min(100, self.light2_brightness))
        
        # 创建发送线程
        self.is_sending = True
        self.send_thread = threading.Thread(target=self.send_loop)
        
        self.connect()
        rospy.Subscriber('/sensor', Control, self.control_callback)
        self.send_thread.start()
        rospy.loginfo(f"sensor driver: 已启动")

    def connect(self):
        while not rospy.is_shutdown():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(self.server_addr)
                self.sock.settimeout(1)
                rospy.loginfo(f"sensor driver: TCP连接 {self.ip}:{self.port}")
                return
            except Exception as e:
                rospy.logerr(f"sensor driver: TCP连接失败 {e}, 2s 后重试")
                self.sock = None
                rospy.sleep(2)

    def build_packet(self, led_red, led_green, servo):
        """
        构造54字节控制报文
        41: 绿色LED
        42: 红色LED
        43: 补光灯1亮度(0-100)
        44: 补光灯2亮度(0-100)
        46: 舵机角度(100-255)
        """
        packet = bytearray(54)
        packet[0:2] = b'\xFE\xFE'      # 报文头
        packet[41] = led_green         # 绿色LED
        packet[42] = led_red           # 红色LED
        packet[43] = self.light1_brightness  # 补光灯1亮度
        packet[44] = self.light2_brightness  # 补光灯2亮度
        # 舵机角度限制在100到255
        packet[46] = max(100, min(255, servo))
        # 校验和（0~50字节异或）
        xor = 0
        for i in range(51):
            xor ^= packet[i]
        packet[51] = xor
        packet[52:54] = b'\xFD\xFD'    # 报文尾
        return packet

    def control_callback(self, msg):
        """
        接收控制消息，更新成员变量
        """
        try:
            # 验证数值范围
            if 0 <= msg.led_red <= 1 and 0 <= msg.led_green <= 1 and 100 <= msg.servo <= 255:
                self.led_red = msg.led_red
                self.led_green = msg.led_green
                self.servo = msg.servo
                rospy.loginfo(f"sensor driver: 更新控制值 LED_R={self.led_red}, LED_G={self.led_green}, Servo={self.servo}")
            else:
                rospy.logwarn("sensor driver: 控制值超出范围，忽略此次更新")
        except Exception as e:
            rospy.logerr(f"sensor driver: 更新控制值失败: {e}")

    def send_loop(self):
        """
        5Hz循环发送传感器控制报文
        """
        rate = rospy.Rate(5)  # 5Hz
        while not rospy.is_shutdown() and self.is_sending:
            try:
                if self.sock is None:
                    self.connect()
                else:
                    packet = self.build_packet(self.led_red, self.led_green, self.servo)
                    self.sock.sendall(packet)
            except Exception as e:
                rospy.logerr(f"sensor driver: 发送失败: {e}")
                self.sock = None  # 触发重连
            rate.sleep()

    def spin(self):
        """
        等待节点关闭
        """
        try:
            rospy.spin()
        finally:
            self.is_sending = False  # 停止发送线程
            if self.send_thread.is_alive():
                self.send_thread.join()
            if self.sock:
                self.sock.close()

if __name__ == "__main__":
    rospy.init_node('sensor_node')
    try:
        client = SensorDriver()
        client.spin()
    except rospy.ROSInterruptException:
        pass

