#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
名称：sensor_driver_v2.py
功能：接收新版 sensor STATUS 报文，发布两路电源状态
     通过debug协议下发执行器控制指令（补光灯、舵机、推杆电机、指示灯）
作者：buyegaid
监听：/auv_actuator_control (ActuatorControl.msg)
发布：/sensor_status (SensorStatus.msg)
记录：
2025.7.15 16:26
    初始版本，接收STATUS报文解析电源数据
2026.7.7
    增加 debug 协议下行控制：发送 CAMERA_LIGHT_SET 和 ACTUATOR_SET 帧
    增加 /auv_actuator_control 订阅（ActuatorControl.msg）
    增加 ACK 和 ACTUATOR_FB 上行帧解析与日志
"""

import json
import os
import socket
import struct
import threading
from datetime import datetime

import rospy
from std_msgs.msg import Header

from auv_control.msg import SensorStatus, ActuatorControl


class SensorDriverV2:
    """
    新版 sensor 驱动：
    1. TCP 接收 64 字节上行帧，解析 STATUS/ACK/ACTUATOR_FB
    2. 发布两路电源的电压、电流、功率到 /sensor_status
    3. 接收 /auv_actuator_control 控制指令，通过 debug 协议下行帧控制执行机构
    """

    PACKET_LEN = 64
    HEADER = b'\xFE\xEF'
    TAIL = b'\xFA\xAF'
    REPORT_STATUS = 0x00
    REPORT_ACK = 0x01
    REPORT_ACTUATOR_FB = 0x03
    STATUS_MIN_PAYLOAD_LEN = 18

    # --- 下行帧常量 (debug 协议) ---
    DOWNLINK_LEN = 54
    DOWNLINK_HEADER = b'\xFE\xFE'
    DOWNLINK_TAIL = b'\xFD\xFD'
    PROTOCOL_VERSION = 0x02

    CMD_CAMERA_LIGHT = 0x10
    CMD_ACTUATOR = 0x30
    OP_SET = 0x00
    FLAG_NEED_ACK = 0x01
    SEND_RATE_HZ = 5

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

        # --- 下行控制状态 ---
        self.lock = threading.Lock()
        self.seq = 0  # 下行帧序列号 (0-255)

        # 执行器状态缓存（默认值）
        self.light1 = 0
        self.light2 = 0
        self.heading_servo = 0x80  # 中间值
        self.clamp_servo = 0x00    # 全开
        self.drive_cmd = 0         # 停止
        self.drive_speed = 0
        self.red_light = 0
        self.yellow_light = 0
        self.green_light = 0
        self.control_changed = False  # 脏标志：仅在值变化时发送

        # 发送线程
        self.is_sending = True
        self.send_thread = None

        # 订阅执行器控制指令
        rospy.Subscriber('/auv_actuator_control', ActuatorControl, self.actuator_callback)

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

        if report_type == self.REPORT_STATUS:
            self._parse_status(payload, payload_len)
        elif report_type == self.REPORT_ACK:
            self._parse_ack(packet, payload_len)
        elif report_type == self.REPORT_ACTUATOR_FB:
            self._parse_actuator_fb(packet, payload_len)
        else:
            rospy.logdebug(f"sensor_driver_v2: 忽略未知帧 report_type=0x{report_type:02X}")

    def _parse_status(self, payload, payload_len):
        """解析 STATUS 周期状态帧，发布电源数据"""
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

    def _parse_ack(self, packet, payload_len):
        """解析 ACK 帧，记录命令处理结果"""
        # ACK 字段在帧头: byte5=cmd, byte6=op, byte7=result, byte8=error
        ack_cmd = packet[5]
        ack_op = packet[6]
        ack_result = packet[7]
        ack_error = packet[8]

        cmd_name = {self.CMD_CAMERA_LIGHT: 'CAMERA_LIGHT', self.CMD_ACTUATOR: 'ACTUATOR'}.get(ack_cmd, f'0x{ack_cmd:02X}')
        if ack_result != 0:
            rospy.logwarn(
                f"sensor_driver_v2: ACK错误 cmd={cmd_name} op=0x{ack_op:02X} "
                f"result=0x{ack_result:02X} error=0x{ack_error:02X}"
            )
        else:
            rospy.logdebug(f"sensor_driver_v2: ACK成功 cmd={cmd_name}")

    def _parse_actuator_fb(self, packet, payload_len):
        """解析 ACTUATOR_FB 帧，记录执行机构当前状态"""
        # ACTUATOR_FB 字段在帧头: byte5=cmd(0x30), byte7=result, byte8=error
        # payload: [heading, angle, drive_cmd, drive_speed, red, yellow, green, error]
        if payload_len < 8:
            rospy.logwarn(f"sensor_driver_v2: ACTUATOR_FB payload长度不足: {payload_len}")
            return

        payload = packet[10:10 + payload_len]
        fb_result = packet[7]
        fb_error = packet[8]

        rospy.loginfo(
            f"sensor_driver_v2: 执行机构反馈 "
            f"heading=0x{payload[0]:02X} angle=0x{payload[1]:02X} "
            f"drive=({payload[2]},{payload[3]}) "
            f"led=({payload[4]},{payload[5]},{payload[6]}) "
            f"result=0x{fb_result:02X} error=0x{fb_error:02X}"
        )

    # ============================================================
    # 下行控制方法
    # ============================================================

    def actuator_callback(self, msg):
        """接收 /auv_actuator_control 控制指令，更新缓存并置脏标志"""
        try:
            # 值域裁剪
            new_light1 = max(0, min(100, msg.light1))
            new_light2 = max(0, min(100, msg.light2))
            new_heading = max(0, min(255, msg.heading_servo))
            new_clamp = max(0, min(255, msg.clamp_servo))
            new_drive_cmd = msg.drive_cmd if msg.drive_cmd in (0, 1, 2) else 0
            new_drive_speed = max(0, min(254, msg.drive_speed))
            new_red = 1 if msg.red_light else 0
            new_yellow = 1 if msg.yellow_light else 0
            new_green = 1 if msg.green_light else 0

            changed = (
                new_light1 != self.light1 or
                new_light2 != self.light2 or
                new_heading != self.heading_servo or
                new_clamp != self.clamp_servo or
                new_drive_cmd != self.drive_cmd or
                new_drive_speed != self.drive_speed or
                new_red != self.red_light or
                new_yellow != self.yellow_light or
                new_green != self.green_light
            )

            with self.lock:
                self.light1 = new_light1
                self.light2 = new_light2
                self.heading_servo = new_heading
                self.clamp_servo = new_clamp
                self.drive_cmd = new_drive_cmd
                self.drive_speed = new_drive_speed
                self.red_light = new_red
                self.yellow_light = new_yellow
                self.green_light = new_green
                if changed:
                    self.control_changed = True

            if changed:
                rospy.loginfo(
                    f"sensor_driver_v2: 执行器控制更新 "
                    f"light=({self.light1},{self.light2}) "
                    f"heading=0x{self.heading_servo:02X} clamp=0x{self.clamp_servo:02X} "
                    f"drive=({self.drive_cmd},{self.drive_speed}) "
                    f"led=({self.red_light},{self.yellow_light},{self.green_light})"
                )
        except Exception as e:
            rospy.logerr(f"sensor_driver_v2: 执行器控制回调失败: {e}")

    def _next_seq(self):
        """递增并返回序列号 (0-255 回绕)"""
        self.seq = (self.seq + 1) & 0xFF
        return self.seq

    @staticmethod
    def _calc_downlink_xor(packet):
        """下行帧异或校验: 字节 0-50"""
        xor = 0
        for i in range(51):
            xor ^= packet[i]
        return xor & 0xFF

    def build_camera_light_frame(self):
        """
        构造 CAMERA_LIGHT_SET 下行帧 (cmd=0x10, op=0x00)
        payload: [light1(0-100), light2(0-100)]
        """
        packet = bytearray(self.DOWNLINK_LEN)
        packet[0:2] = self.DOWNLINK_HEADER
        packet[2] = self.PROTOCOL_VERSION          # version
        packet[3] = self._next_seq()               # seq
        packet[4] = self.CMD_CAMERA_LIGHT          # cmd
        packet[5] = self.OP_SET                    # op
        packet[6] = 0x00                           # index
        packet[7] = 2                              # payload_len
        packet[8] = self.light1                    # payload[0]
        packet[9] = self.light2                    # payload[1]
        # 10-39: payload剩余填0 (bytearray默认)
        packet[40] = self.FLAG_NEED_ACK            # flags
        # 41-50: reserved 填0
        packet[51] = self._calc_downlink_xor(packet)
        packet[52:54] = self.DOWNLINK_TAIL
        return packet

    def build_actuator_frame(self):
        """
        构造 ACTUATOR_SET 下行帧 (cmd=0x30, op=0x00)
        payload: [heading, angle, drive_cmd, drive_speed, red, yellow, green]
        """
        packet = bytearray(self.DOWNLINK_LEN)
        packet[0:2] = self.DOWNLINK_HEADER
        packet[2] = self.PROTOCOL_VERSION          # version
        packet[3] = self._next_seq()               # seq
        packet[4] = self.CMD_ACTUATOR              # cmd
        packet[5] = self.OP_SET                    # op
        packet[6] = 0x00                           # index
        packet[7] = 7                              # payload_len
        packet[8] = self.heading_servo             # payload[0]: heading
        packet[9] = self.clamp_servo               # payload[1]: angle
        packet[10] = self.drive_cmd                # payload[2]: drive_cmd
        packet[11] = self.drive_speed              # payload[3]: drive_speed
        packet[12] = self.red_light                # payload[4]: red
        packet[13] = self.yellow_light             # payload[5]: yellow
        packet[14] = self.green_light              # payload[6]: green
        # 15-39: payload剩余填0
        packet[40] = self.FLAG_NEED_ACK            # flags
        # 41-50: reserved 填0
        packet[51] = self._calc_downlink_xor(packet)
        packet[52:54] = self.DOWNLINK_TAIL
        return packet

    def send_loop(self):
        """5Hz 发送线程: 仅在控制值变化时发送 CAMERA_LIGHT_SET + ACTUATOR_SET 两帧"""
        rate = rospy.Rate(self.SEND_RATE_HZ)
        while not rospy.is_shutdown() and self.is_sending:
            try:
                if self.sock is None:
                    rate.sleep()
                    continue

                with self.lock:
                    changed = self.control_changed
                    if changed:
                        self.control_changed = False

                if changed:
                    light_frame = self.build_camera_light_frame()
                    self.sock.sendall(bytes(light_frame))
                    actuator_frame = self.build_actuator_frame()
                    self.sock.sendall(bytes(actuator_frame))
                    rospy.logdebug("sensor_driver_v2: 已发送控制帧")
            except Exception as e:
                rospy.logerr(f"sensor_driver_v2: 发送失败: {e}")
            rate.sleep()

    # ============================================================

    def spin(self):
        self.send_thread = threading.Thread(target=self.send_loop, daemon=True)
        self.send_thread.start()
        try:
            self.recv_loop()
        finally:
            self.is_sending = False
            if self.send_thread and self.send_thread.is_alive():
                self.send_thread.join(timeout=2)
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
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
