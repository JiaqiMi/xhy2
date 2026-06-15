#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
名称：sensor_driver_v2.py
功能：接收新版 sensor STATUS 报文，发布两路电源状态
监听：None
发布：/sensor_status (SensorStatus.msg)
"""

import socket
import struct

import rospy
from std_msgs.msg import Header

from auv_control.msg import SensorStatus


class SensorDriverV2:
    """
    新版 sensor 驱动：
    1. TCP 接收 64 字节上行帧
    2. 只解析 report_type=STATUS 的周期状态帧
    3. 只发布两路电源的电压、电流、功率
    """

    PACKET_LEN = 64
    HEADER = b'\xFE\xEF'
    TAIL = b'\xFA\xAF'
    REPORT_STATUS = 0x00
    STATUS_MIN_PAYLOAD_LEN = 18

    def __init__(self):
        self.ip = rospy.get_param('~sensor_ip', '192.168.1.115')
        self.port = rospy.get_param('~sensor_port', 5064)
        self.server_addr = (self.ip, self.port)

        self.sock = None
        self.buffer = bytearray()
        self.pub = rospy.Publisher('/sensor_status', SensorStatus, queue_size=10)

        self.connect()
        rospy.loginfo("sensor_driver_v2: 已启动")

    def connect(self):
        while not rospy.is_shutdown():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(self.server_addr)
                self.sock.settimeout(1.0)
                rospy.loginfo(f"sensor_driver_v2: TCP连接 {self.ip}:{self.port}")
                return
            except Exception as e:
                rospy.logerr(f"sensor_driver_v2: TCP连接失败 {e}, 2s 后重试")
                self.sock = None
                rospy.sleep(2)

    @staticmethod
    def calc_xor(packet):
        value = 0
        for byte in packet[0:61]:
            value ^= byte
        return value

    def verify_packet(self, packet):
        if len(packet) != self.PACKET_LEN:
            return False
        if packet[0:2] != self.HEADER:
            return False
        if packet[62:64] != self.TAIL:
            return False
        return self.calc_xor(packet) == packet[61]

    def recv_loop(self):
        while not rospy.is_shutdown():
            try:
                if self.sock is None:
                    self.connect()
                    continue

                data = self.sock.recv(1024)
                if not data:
                    raise RuntimeError("对端关闭连接")

                self.buffer.extend(data)
                self.process_buffer()

            except Exception as e:
                rospy.logerr(f"sensor_driver_v2: 接收失败: {e}")
                try:
                    if self.sock:
                        self.sock.close()
                except Exception:
                    pass
                self.sock = None
                self.buffer = bytearray()
                rospy.sleep(1)

    def process_buffer(self):
        while len(self.buffer) >= self.PACKET_LEN:
            idx = self.buffer.find(self.HEADER)
            if idx < 0:
                if len(self.buffer) > 1:
                    self.buffer = self.buffer[-1:]
                return

            if idx > 0:
                del self.buffer[:idx]

            if len(self.buffer) < self.PACKET_LEN:
                return

            packet = bytes(self.buffer[:self.PACKET_LEN])
            if self.verify_packet(packet):
                self.parse_and_publish(packet)
                del self.buffer[:self.PACKET_LEN]
            else:
                rospy.logwarn("sensor_driver_v2: 报文校验失败，丢弃1字节继续同步")
                del self.buffer[0]

    def parse_and_publish(self, packet):
        report_type = packet[4]
        payload_len = packet[9]
        payload = packet[10:56]

        if report_type != self.REPORT_STATUS:
            rospy.logdebug(f"sensor_driver_v2: 忽略非STATUS帧 report_type={report_type}")
            return

        if payload_len < self.STATUS_MIN_PAYLOAD_LEN:
            rospy.logwarn(f"sensor_driver_v2: STATUS payload长度不足: {payload_len}")
            return

        system_flags = payload[1]

        msg = SensorStatus()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "sensor"
        msg.checksum_ok = True
        msg.power1_valid = bool(system_flags & 0x01)
        msg.power2_valid = bool(system_flags & 0x02)

        msg.power1_voltage = struct.unpack_from('<H', payload, 2)[0] / 1000.0
        msg.power1_current = struct.unpack_from('<h', payload, 4)[0] / 1000.0
        msg.power1_power = struct.unpack_from('<i', payload, 6)[0] / 1000.0

        msg.power2_voltage = struct.unpack_from('<H', payload, 10)[0] / 1000.0
        msg.power2_current = struct.unpack_from('<h', payload, 12)[0] / 1000.0
        msg.power2_power = struct.unpack_from('<i', payload, 14)[0] / 1000.0

        self.pub.publish(msg)

    def spin(self):
        self.recv_loop()


if __name__ == "__main__":
    rospy.init_node('sensor_driver_v2')
    try:
        driver = SensorDriverV2()
        driver.spin()
    except rospy.ROSInterruptException:
        pass
