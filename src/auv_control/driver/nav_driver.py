#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
名称：nav_driver.py
功能：通过TCP接收导航设备140字节报文，解析完整导航数据并发布ROS消息
作者：buyegaid
监听：None
发布：/nav (NavFull.msg)
记录：
2026.03.21
    初版：
    1. 连接TCP 192.168.1.115:5066
    2. 接收140字节导航报文
    3. 校验包头和校验和
    4. 解析INS/GPS/DVL/IMU/USBL/波束速度和距离/标志位
    5. 发布完整ROS消息
2026.7.13
    调整至 driver 目录，归入硬件驱动层
"""

import json
import os
import socket
import struct
from datetime import datetime

import rospy
from genpy import Message
from std_msgs.msg import Header
from auv_control.msg import NavData


class NavDriver:
    """
    导航驱动节点：
    1. TCP接收140字节导航报文
    2. 解析所有有效字段
    3. 发布ROS消息
    """

    def __init__(self):
        self.ip = rospy.get_param('~nav_ip', '192.168.1.115')
        self.port = rospy.get_param('~nav_port', 5066)
        self.server_addr = (self.ip, self.port)

        self.sock = None
        self.buffer = bytearray()

        self.PACKET_LEN = 140
        self.HEADER = b'\xAA\x55\x5A\xA5'

        self.pub = rospy.Publisher('/nav', NavData, queue_size=10)
        self.save_data = rospy.get_param('~save_data', False)
        self.save_dir = os.path.expanduser(rospy.get_param('~save_dir', '~/.ros/auv_logs'))
        self.save_file_name = rospy.get_param('~save_file', '')
        self.flush_every = max(1, int(rospy.get_param('~flush_every', 1)))
        self.write_count = 0
        self.save_file = None
        self.raw_saving_enable = rospy.get_param('~save_raw_data', False)
        self.raw_save_dir = os.path.expanduser(rospy.get_param('~raw_save_dir', '~/.ros/auv_logs'))
        self.raw_save_file_name = rospy.get_param('~raw_save_file', '')
        self.raw_flush_every = max(1, int(rospy.get_param('~raw_flush_every', 1)))
        self.raw_write_count = 0
        self.raw_save_file = None

        if self.save_data:
            self.open_save_file()
        if self.raw_saving_enable:
            self.open_raw_save_file()

        self.connect()
        rospy.loginfo("nav driver: 已启动")

    def open_save_file(self):
        if not self.save_file_name:
            self.save_file_name = datetime.now().strftime('nav_data_%Y%m%d_%H%M%S.jsonl')

        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, self.save_file_name)
        self.save_file = open(path, 'a', encoding='utf-8')
        rospy.loginfo(f"nav driver: 数据将保存到 {path}")

    def open_raw_save_file(self):
        if not self.raw_save_file_name:
            self.raw_save_file_name = datetime.now().strftime('nav_raw_%Y%m%d_%H%M%S.jsonl')

        os.makedirs(self.raw_save_dir, exist_ok=True)
        path = os.path.join(self.raw_save_dir, self.raw_save_file_name)
        self.raw_save_file = open(path, 'a', encoding='utf-8')
        rospy.loginfo(f"nav driver: 原始报文将保存到 {path}")

    def connect(self):
        while not rospy.is_shutdown():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect(self.server_addr)
                self.sock.settimeout(1.0)
                rospy.loginfo(f"nav driver: TCP连接 {self.ip}:{self.port}")
                return
            except Exception as e:
                rospy.logerr(f"nav driver: TCP连接失败 {e}, 2s 后重试")
                self.sock = None
                rospy.sleep(2)

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
                rospy.logerr(f"nav driver: 接收失败: {e}")
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
                if len(self.buffer) > 3:
                    self.buffer = self.buffer[-3:]
                return

            if idx > 0:
                del self.buffer[:idx]

            if len(self.buffer) < self.PACKET_LEN:
                return

            packet = bytes(self.buffer[:self.PACKET_LEN])

            checksum_ok, calc_sum, recv_sum = self.verify_packet(packet)
            self.save_raw_packet(packet, checksum_ok, calc_sum, recv_sum)
            if checksum_ok:
                self.parse_and_publish(packet, calc_sum, recv_sum, checksum_ok)
                del self.buffer[:self.PACKET_LEN]
            else:
                rospy.logwarn("nav driver: 报文校验失败，丢弃1字节继续同步")
                del self.buffer[0]

    def verify_packet(self, packet):
        if len(packet) != self.PACKET_LEN:
            return False, 0, 0
        if packet[0:4] != self.HEADER:
            return False, 0, 0

        calc_sum = sum(packet[0:138]) & 0xFFFF
        recv_sum = struct.unpack_from('<H', packet, 138)[0]
        return calc_sum == recv_sum, calc_sum, recv_sum

    def parse_and_publish(self, packet, calc_sum, recv_sum, checksum_ok):
        msg = NavData()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "nav"

        # ----------------------------
        # 基础读数函数
        # ----------------------------
        def i16(offset):
            return struct.unpack_from('<h', packet, offset)[0]

        def u16(offset):
            return struct.unpack_from('<H', packet, offset)[0]

        def i32(offset):
            return struct.unpack_from('<i', packet, offset)[0]

        def u32(offset):
            return struct.unpack_from('<I', packet, offset)[0]

        def u8(offset):
            return struct.unpack_from('<B', packet, offset)[0]

        def u24(offset):
            b0 = packet[offset]
            b1 = packet[offset + 1]
            b2 = packet[offset + 2]
            return b0 | (b1 << 8) | (b2 << 16)

        def latlon_deg(raw):
            return float(raw) * 180.0 / 2147483648.0

        def angle_signed_deg(raw):
            return float(raw) * 180.0 / 32768.0

        def heading_deg(raw):
            return float(raw) * 360.0 / 65536.0

        def mm_to_m(raw):
            return float(raw) / 1000.0

        def mmps_to_mps(raw):
            return float(raw) / 1000.0

        def gyro_to_dps(raw):
            # 协议图：0.000001 deg/s (10^-6)
            return float(raw) * 1e-6

        def accel_to_mps2(raw):
            # 协议图：0.0000001 m/s^2 (10^-7)
            return float(raw) * 1e-7

        def temp_to_degC(raw):
            # 协议图：0.01℃
            return float(raw) * 0.01

        def set_u8_bits(prefix, value):
            for i in range(8):
                setattr(msg, f"{prefix}_bit{i}", bool((value >> i) & 0x01))


        # ----------------------------
        # 原始包信息
        # ----------------------------
        msg.counter = u16(4)
        msg.checksum = recv_sum
        msg.checksum_ok = checksum_ok

        # ----------------------------
        # INS
        # ----------------------------
        msg.latitude = latlon_deg(i32(6))
        msg.longitude = latlon_deg(i32(10))
        msg.altitude = mm_to_m(i32(14))
        msg.heave = mm_to_m(i16(18))
        msg.vn = mmps_to_mps(i16(20))
        msg.ve = mmps_to_mps(i16(22))
        msg.vd = mmps_to_mps(i16(24))
        msg.roll = angle_signed_deg(i16(26))
        msg.pitch = angle_signed_deg(i16(28))
        msg.heading = heading_deg(u16(30))
        msg.ins_status = u16(32)

        # ----------------------------
        # GPS
        # ----------------------------
        msg.gps_latitude = latlon_deg(i32(34))
        msg.gps_longitude = latlon_deg(i32(38))
        msg.gps_altitude = mm_to_m(i32(42))
        msg.gps_vel = mmps_to_mps(i16(46))
        msg.gps_heading = heading_deg(u16(48))
        msg.gps_status = u8(50)

        # ----------------------------
        # DVL
        # ----------------------------
        msg.dvl_vx = mmps_to_mps(i16(51))
        msg.dvl_vy = mmps_to_mps(i16(53))
        msg.dvl_vz = mmps_to_mps(i16(55))
        msg.dvl_altitude = mm_to_m(i32(57))
        msg.dvl_status = u8(61)

        # ----------------------------
        # IMU原始采样
        # ----------------------------
        msg.gyro_x = gyro_to_dps(i32(62))
        msg.gyro_y = gyro_to_dps(i32(66))
        msg.gyro_z = gyro_to_dps(i32(70))

        msg.accel_x = accel_to_mps2(i32(74))
        msg.accel_y = accel_to_mps2(i32(78))
        msg.accel_z = accel_to_mps2(i32(82))

        msg.temperature = temp_to_degC(i16(86))
        msg.imu_status = u16(88)

        # ----------------------------
        # 时间
        # ----------------------------
        msg.yymmdd_raw = u24(90)
        msg.hhmmss_raw = u24(93)
        msg.ms = u16(96)

        # ----------------------------
        # 深度
        # ----------------------------
        msg.depth = mm_to_m(i32(104))


        # ----------------------------
        # 更新标志位
        # ----------------------------
        msg.update_flags = u8(137)
        self.pub.publish(msg)
        self.save_nav_msg(msg)

    def message_to_dict(self, msg):
        if hasattr(msg, 'secs') and hasattr(msg, 'nsecs') and callable(getattr(msg, 'to_sec', None)):
            return {
                'secs': msg.secs,
                'nsecs': msg.nsecs,
                'time': msg.to_sec(),
            }

        if isinstance(msg, Message):
            result = {}
            for field in msg.__slots__:
                result[field] = self.message_to_dict(getattr(msg, field))
            return result

        if isinstance(msg, (list, tuple)):
            return [self.message_to_dict(item) for item in msg]

        return msg

    def save_nav_msg(self, msg):
        if not self.save_data or self.save_file is None:
            return

        event = {
            'pc_time': rospy.Time.now().to_sec(),
            'source': 'nav',
            'topic': '/nav',
            'msg_type': msg._type,
            'stamp': self.message_to_dict(msg.header.stamp),
            'data': self.message_to_dict(msg),
        }

        try:
            self.save_file.write(json.dumps(event, ensure_ascii=False) + '\n')
            self.write_count += 1
            if self.write_count % self.flush_every == 0:
                self.save_file.flush()
        except Exception as e:
            rospy.logerr(f"nav driver: 保存数据失败: {e}")

    def save_raw_packet(self, packet, checksum_ok, calc_sum, recv_sum):
        if not self.raw_saving_enable or self.raw_save_file is None:
            return

        event = {
            'pc_time': rospy.Time.now().to_sec(),
            'source': 'nav',
            'packet_len': len(packet),
            'checksum_ok': bool(checksum_ok),
            'calc_checksum': calc_sum,
            'recv_checksum': recv_sum,
            'packet_hex': ' '.join('{:02x}'.format(byte) for byte in packet),
        }

        try:
            self.raw_save_file.write(json.dumps(event, ensure_ascii=False) + '\n')
            self.raw_write_count += 1
            if self.raw_write_count % self.raw_flush_every == 0:
                self.raw_save_file.flush()
        except Exception as e:
            rospy.logerr(f"nav driver: 保存原始报文失败: {e}")

    def spin(self):
        try:
            self.recv_loop()
        finally:
            if self.save_file:
                self.save_file.flush()
                self.save_file.close()
                rospy.loginfo("nav driver: 数据文件已保存并关闭")
            if self.raw_save_file:
                self.raw_save_file.flush()
                self.raw_save_file.close()
                rospy.loginfo("nav driver: 原始报文文件已保存并关闭")


if __name__ == "__main__":
    rospy.init_node('nav_driver')
    try:
        driver = NavDriver()
        driver.spin()
    except rospy.ROSInterruptException:
        pass
