#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
名称：sensor_status_node.py
功能：纯 STATUS 帧接收节点，从 sensor debug 上行帧中解析电源状态并发布
     不发送任何下行控制帧，不与执行器控制逻辑耦合
作者：buyegaid
监听：无
发布：/status/power (SensorStatus.msg)
记录：
2026.7.11
    从 sensor_driver_v2.py 拆分出纯 STATUS 接收逻辑，独立 TCP 连接
    统一 loginfo 中文输出：定长小数格式，电源状态 0.2Hz 节流输出
2026.7.13
    调整至 driver 目录，归入硬件驱动层
    上行电源状态话题调整为 /status/power。
"""

import json
import os
import socket
import struct
from datetime import datetime

import rospy
from std_msgs.msg import Header

from auv_control.msg import SensorStatus


class SensorStatusNode:
    """
    sensor STATUS 帧接收节点：
    - 独立 TCP 连接到 sensor:5064
    - 仅接收 64 字节上行帧，解析 report_type=0x00 (STATUS)
    - 发布两路电源的电压、电流、功率到 /status/power
    - 忽略 ACK、ACTUATOR_FB、CONFIG_FB 等非 STATUS 帧
    """

    PACKET_LEN = 64
    HEADER = b'\xFE\xEF'
    TAIL = b'\xFA\xAF'
    REPORT_STATUS = 0x00
    STATUS_MIN_PAYLOAD_LEN = 18

    def __init__(self):
        self.ip = rospy.get_param('~sensor_ip', '192.168.1.115')
        self.port = rospy.get_param('~sensor_port', 5064)

        self.sock = None
        self.buffer = bytearray()
        self.pub = rospy.Publisher('/status/power', SensorStatus, queue_size=10)

        # --- 原始报文保存 ---
        self.raw_saving_enable = rospy.get_param('~save_raw_data', False)
        self.raw_save_dir = os.path.expanduser(rospy.get_param('~raw_save_dir', '~/.ros/auv_logs'))
        self.raw_save_file_name = rospy.get_param('~raw_save_file', '')
        self.raw_flush_every = max(1, int(rospy.get_param('~raw_flush_every', 1)))
        self.raw_write_count = 0
        self.raw_save_file = None

        if self.raw_saving_enable:
            self._open_raw_save_file()

        # STATUS 日志节流计数器（5Hz 帧率，每 25 帧 = 0.2Hz 输出一次）
        self._status_log_cnt = 0
        self._status_log_interval = 25

        self.connect()
        rospy.loginfo("sensor_status: 已启动（纯 STATUS 接收模式）")

    # ============================================================
    # 原始报文保存
    # ============================================================

    def _open_raw_save_file(self):
        if not self.raw_save_file_name:
            self.raw_save_file_name = datetime.now().strftime('sensor_status_raw_%Y%m%d_%H%M%S.jsonl')

        os.makedirs(self.raw_save_dir, exist_ok=True)
        path = os.path.join(self.raw_save_dir, self.raw_save_file_name)
        self.raw_save_file = open(path, 'a', encoding='utf-8')
        rospy.loginfo(f"sensor_status: 原始报文将保存到 {path}")

    def _save_raw_packet(self, packet, checksum_ok):
        if not self.raw_saving_enable or self.raw_save_file is None:
            return

        event = {
            'pc_time': rospy.Time.now().to_sec(),
            'source': 'sensor_status',
            'packet_len': len(packet),
            'checksum_ok': bool(checksum_ok),
            'report_type': packet[4] if len(packet) > 4 else None,
            'packet_hex': ' '.join('{:02x}'.format(b) for b in packet),
        }

        try:
            self.raw_save_file.write(json.dumps(event, ensure_ascii=False) + '\n')
            self.raw_write_count += 1
            if self.raw_write_count % self.raw_flush_every == 0:
                self.raw_save_file.flush()
        except Exception as e:
            rospy.logerr(f"sensor_status: 保存原始报文失败: {e}")

    # ============================================================
    # TCP 连接
    # ============================================================

    def connect(self):
        while not rospy.is_shutdown():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.ip, self.port))
                self.sock.settimeout(1.0)
                rospy.loginfo(f"sensor_status: TCP连接 {self.ip}:{self.port}")
                return
            except Exception as e:
                rospy.logerr(f"sensor_status: TCP连接失败 {e}, 2s 后重试")
                self.sock = None
                rospy.sleep(2)

    # ============================================================
    # 校验
    # ============================================================

    @staticmethod
    def calc_xor(packet):
        """上行帧 XOR 校验：覆盖字节 0-60"""
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

    # ============================================================
    # 接收与同步
    # ============================================================

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
                self._process_buffer()

            except Exception as e:
                rospy.logerr(f"sensor_status: 接收失败: {e}")
                try:
                    if self.sock:
                        self.sock.close()
                except Exception:
                    pass
                self.sock = None
                self.buffer = bytearray()
                rospy.sleep(1)

    def _process_buffer(self):
        while len(self.buffer) >= self.PACKET_LEN:
            idx = self.buffer.find(self.HEADER)
            if idx < 0:
                if len(self.buffer) > 1:
                    self.buffer = self.buffer[-1:]
                return

            if idx > 0:
                rospy.logdebug(f"sensor_status: 跳过 {idx} 字节同步到帧头")
                del self.buffer[:idx]

            if len(self.buffer) < self.PACKET_LEN:
                return

            packet = bytes(self.buffer[:self.PACKET_LEN])
            checksum_ok = self.verify_packet(packet)
            tail_ok = (packet[62:64] == self.TAIL)

            if tail_ok:
                self._save_raw_packet(packet, checksum_ok)

            if checksum_ok:
                self._parse_and_publish(packet)
                del self.buffer[:self.PACKET_LEN]
            else:
                if not tail_ok:
                    # 帧尾不匹配 → 假帧头，跳到下一处 FE EF
                    next_idx = self.buffer.find(self.HEADER, 2)
                    if next_idx > 0:
                        rospy.logdebug(
                            f"sensor_status: 假帧头（帧尾不匹配），跳过 {next_idx} 字节到下一帧头"
                        )
                        del self.buffer[:next_idx]
                    else:
                        del self.buffer[0]
                else:
                    # 帧头帧尾正确但 XOR 错误 → 罕见数据损坏
                    rospy.logwarn("sensor_status: 报文校验失败(XOR错误)，丢弃1字节继续同步")
                    del self.buffer[0]

    # ============================================================
    # 解析
    # ============================================================

    def _parse_and_publish(self, packet):
        """仅解析 STATUS 帧，忽略其他帧类型"""
        report_type = packet[4]

        if report_type == self.REPORT_STATUS:
            payload_len = packet[9]
            self._parse_status(packet, payload_len)
        # ACK / ACTUATOR_FB / CONFIG_FB 等帧由 actuator 节点处理，本节点忽略

    def _parse_status(self, packet, payload_len):
        """解析 STATUS 周期状态帧，发布电源数据"""
        if payload_len < self.STATUS_MIN_PAYLOAD_LEN:
            rospy.logwarn(f"sensor_status: STATUS payload长度不足: {payload_len}")
            return

        payload = packet[10:56]
        system_flags = payload[1]

        p1_valid = bool(system_flags & 0x01)
        p2_valid = bool(system_flags & 0x02)

        p1_v = struct.unpack_from('<H', payload, 2)[0] / 1000.0
        p1_c = struct.unpack_from('<h', payload, 4)[0] / 1000.0
        p1_p = struct.unpack_from('<i', payload, 6)[0] / 1000.0
        p2_v = struct.unpack_from('<H', payload, 10)[0] / 1000.0
        p2_c = struct.unpack_from('<h', payload, 12)[0] / 1000.0
        p2_p = struct.unpack_from('<i', payload, 14)[0] / 1000.0

        msg = SensorStatus()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "sensor"
        msg.checksum_ok = True
        msg.power1_valid = p1_valid
        msg.power2_valid = p2_valid
        msg.power1_voltage = p1_v
        msg.power1_current = p1_c
        msg.power1_power = p1_p
        msg.power2_voltage = p2_v
        msg.power2_current = p2_c
        msg.power2_power = p2_p

        self.pub.publish(msg)

        # STATUS 日志节流（0.2Hz，确保连接正常）
        self._status_log_cnt += 1
        if self._status_log_cnt % self._status_log_interval == 0:
            rospy.loginfo(
                "sensor_status: 电源1 %4s V=%7.3fV I=%7.3fA P=%7.1fW | "
                "电源2 %4s V=%7.3fV I=%7.3fA P=%7.1fW",
                "正常" if p1_valid else "无效", p1_v, p1_c, p1_p,
                "正常" if p2_valid else "无效", p2_v, p2_c, p2_p
            )

    # ============================================================

    def spin(self):
        try:
            self.recv_loop()
        finally:
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass
            if self.raw_save_file:
                self.raw_save_file.flush()
                self.raw_save_file.close()
                rospy.loginfo("sensor_status: 原始报文文件已保存并关闭")


if __name__ == "__main__":
    rospy.init_node('sensor_status_node')
    try:
        node = SensorStatusNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
