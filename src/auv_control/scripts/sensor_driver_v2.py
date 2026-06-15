#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
名称：sensor_driver_v2.py
功能：接收新版 sensor STATUS 报文，发布两路电源状态
监听：None
发布：/sensor_status (SensorStatus.msg)
"""

import json
import os
import socket
import struct
from datetime import datetime

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
        self.raw_saving_enable = rospy.get_param('~save_raw_data', False)
        self.raw_save_dir = os.path.expanduser(rospy.get_param('~raw_save_dir', '~/.ros/auv_logs'))
        self.raw_save_file_name = rospy.get_param('~raw_save_file', '')
        self.raw_flush_every = max(1, int(rospy.get_param('~raw_flush_every', 1)))
        self.raw_write_count = 0
        self.raw_save_file = None

        if self.raw_saving_enable:
            self.open_raw_save_file()

        self.connect()
        rospy.loginfo("sensor_driver_v2: 已启动")

    def open_raw_save_file(self):
        if not self.raw_save_file_name:
            self.raw_save_file_name = datetime.now().strftime('sensor_raw_%Y%m%d_%H%M%S.jsonl')

        os.makedirs(self.raw_save_dir, exist_ok=True)
        path = os.path.join(self.raw_save_dir, self.raw_save_file_name)
        self.raw_save_file = open(path, 'a', encoding='utf-8')
        rospy.loginfo(f"sensor_driver_v2: 原始报文将保存到 {path}")

    def save_raw_packet(self, packet, checksum_ok):
        if not self.raw_saving_enable or self.raw_save_file is None:
            return

        event = {
            'pc_time': rospy.Time.now().to_sec(),
            'source': 'sensor',
            'packet_len': len(packet),
            'checksum_ok': bool(checksum_ok),
            'report_type': packet[4] if len(packet) > 4 else None,
            'packet_hex': ' '.join('{:02x}'.format(byte) for byte in packet),
        }

        try:
            self.raw_save_file.write(json.dumps(event, ensure_ascii=False) + '\n')
            self.raw_write_count += 1
            if self.raw_write_count % self.raw_flush_every == 0:
                self.raw_save_file.flush()
        except Exception as e:
            rospy.logerr(f"sensor_driver_v2: 保存原始报文失败: {e}")

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
            checksum_ok = self.verify_packet(packet)
            if packet[62:64] == self.TAIL:
                self.save_raw_packet(packet, checksum_ok)

            if checksum_ok:
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
        try:
            self.recv_loop()
        finally:
            if self.raw_save_file:
                self.raw_save_file.flush()
                self.raw_save_file.close()
                rospy.loginfo("sensor_driver_v2: 原始报文文件已保存并关闭")


if __name__ == "__main__":
    rospy.init_node('sensor_driver_v2')
    try:
        driver = SensorDriverV2()
        driver.spin()
    except rospy.ROSInterruptException:
        pass
