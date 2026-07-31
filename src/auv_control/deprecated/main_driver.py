#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：main_driver.py
功能：接收旧版 TCP 状态报文，仅转发 DVL 高度
作者：BroXu
监听：TCP 5062 状态报文
发布：/dvl/altitude(std_msgs/Float32)
记录：
2026.7.31
    将已弃用的主驱动收敛为 DVL 高度桥接节点，仅解析旧协议中的高度字段。
"""

import socket
import struct
import time

import rospy
from std_msgs.msg import Float32


LEGACY_PACKET_SIZE = 75
LEGACY_HEADER = b'\xFE\xFE'
LEGACY_TAIL = b'\xFD\xFD'


class DvlAltitudeBridge:
    """从旧版状态端口提取 DVL 高度并转发为独立 ROS 话题。"""

    def __init__(self):
        self.server_address = (
            rospy.get_param('~legacy_ip', '192.168.1.115'),
            int(rospy.get_param('~legacy_port', 5062)),
        )
        self.altitude_topic = rospy.get_param(
            '~dvl_altitude_topic', '/dvl/altitude'
        )
        self.connect_retry_sec = float(
            rospy.get_param('~connect_retry_sec', 2.0)
        )
        if self.connect_retry_sec <= 0.0:
            raise ValueError('connect_retry_sec 必须大于 0')

        self.publisher = rospy.Publisher(
            self.altitude_topic, Float32, queue_size=10
        )
        self.buffer = bytearray()
        rospy.loginfo(
            'main_driver: DVL 高度桥接已启动，读取 %s:%d 并发布 %s',
            self.server_address[0], self.server_address[1], self.altitude_topic,
        )

    @staticmethod
    def calculate_checksum(packet):
        """计算旧版 75 字节状态报文第 0 至 71 字节的异或校验。"""
        checksum = 0
        for byte in packet[:72]:
            checksum ^= byte
        return checksum

    @staticmethod
    def parse_altitude(packet):
        """解析旧版协议第 13 至 14 字节的大端 DVL 高度，单位为米。"""
        return struct.unpack('>H', packet[13:15])[0] / 100.0

    def publish_frames_from_buffer(self):
        """从缓存中恢复完整报文，丢弃坏帧后继续寻找下一帧。"""
        while True:
            start = self.buffer.find(LEGACY_HEADER)
            if start < 0:
                self.buffer[:] = (
                    self.buffer[-1:]
                    if self.buffer and self.buffer[-1] == LEGACY_HEADER[0]
                    else b''
                )
                return
            if start:
                del self.buffer[:start]
            if len(self.buffer) < LEGACY_PACKET_SIZE:
                return

            packet = bytes(self.buffer[:LEGACY_PACKET_SIZE])
            if packet[-2:] != LEGACY_TAIL:
                del self.buffer[:2]
                continue
            del self.buffer[:LEGACY_PACKET_SIZE]
            if self.calculate_checksum(packet) != packet[72]:
                rospy.logwarn_throttle(
                    2.0, 'main_driver: 忽略校验和错误的 DVL 状态报文'
                )
                continue

            altitude = self.parse_altitude(packet)
            self.publisher.publish(Float32(data=altitude))

    def connect(self):
        """建立 TCP 连接，并为接收循环配置超时。"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3.0)
        sock.connect(self.server_address)
        sock.settimeout(1.0)
        return sock

    def run(self):
        """持续接收旧协议状态流，断开后自动重连。"""
        sock = None
        while not rospy.is_shutdown():
            if sock is None:
                try:
                    sock = self.connect()
                    self.buffer.clear()
                    rospy.loginfo('main_driver: 已连接旧版状态端口')
                except socket.error as error:
                    rospy.logwarn_throttle(
                        2.0, 'main_driver: 连接旧版状态端口失败：%s', error
                    )
                    time.sleep(self.connect_retry_sec)
                    continue

            try:
                data = sock.recv(512)
                if not data:
                    raise ConnectionError('旧版状态端口已关闭连接')
                self.buffer.extend(data)
                self.publish_frames_from_buffer()
            except socket.timeout:
                continue
            except (socket.error, ConnectionError) as error:
                rospy.logwarn('main_driver: DVL 状态接收中断，准备重连：%s', error)
                try:
                    sock.close()
                except socket.error:
                    pass
                sock = None
                self.buffer.clear()

        if sock is not None:
            try:
                sock.close()
            except socket.error:
                pass


def main():
    """启动 DVL 高度桥接节点。"""
    rospy.init_node('dvl_altitude_bridge')
    DvlAltitudeBridge().run()


if __name__ == '__main__':
    main()
