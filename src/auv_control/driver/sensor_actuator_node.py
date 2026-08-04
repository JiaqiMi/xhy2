#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
名称：sensor_actuator_node.py
功能：执行器控制与反馈节点，通过 sensor debug 协议下行控制执行机构，
     接收 ACK 和 ACTUATOR_FB 上行帧并发布执行机构实际状态
作者：buyegaid
监听：/cmd/actuator (ActuatorControl.msg)
发布：/status/actuator (ActuatorControl.msg)
记录：
2026.7.11
    从 sensor_driver_v2.py 拆分出执行器控制逻辑，独立 TCP 连接
    发布 /status/actuator 反馈执行机构实际状态
    统一 loginfo 中文输出：ACK/FB/CMD/SEND 定长格式化，保留原有输出频率
2026.7.13
    调整至 driver 目录，归入硬件驱动层
    下行控制话题调整为 /cmd/actuator，上行状态话题调整为 /status/actuator。
    补光灯与执行器命令按实际变化分别发送，增加 ACK 超时重发。
2026.7.15
    新增 ActuatorControl.mode 分流：mode=1 仅更新补光灯，mode=2 仅更新执行器。
    mode=0 和其他值不响应，/status/actuator 状态消息固定使用 mode=0。
2026.7.21
    补光灯与执行器控制帧改为各自 0.5Hz 持续下发，执行器帧固定错开 0.25 秒。
    下发不再等待 ACK；ACTUATOR_FB 仅更新锁存状态，/status/actuator 改为 5Hz 持续发布。
    按协议将 result=0x03 识别为 EXEC_OK，当前执行器状态使用 5 秒节流日志输出。
2026.7.30
    补光灯与执行器控制帧下发频率调整为 2Hz，缩短任务执行器闭环的命令与反馈延迟。
2026.7.31
    新增执行器命令与当前状态 JSONL 保存，默认写入 ~/.ros/auvlog/actuator_node。
2026.7.31
    修正数据保存目录为 ~/.ros/auv_logs/actuator_node，与 sensor_status_node 同级。
2026.8.5
    mode2 完整执行器控制帧调整为 5Hz 持续下发。
    停止响应和下发 mode1 补光灯控制，仅保留执行器反馈与状态发布。
"""

import json
import os
import socket
import threading
import time
from datetime import datetime

import rospy

from auv_control.msg import ActuatorControl


class SensorActuatorNode:
    """
    sensor 执行器控制与反馈节点：
    - 独立 TCP 连接到 sensor:5064
    - 订阅 /cmd/actuator，仅下发 ACTUATOR_SET
    - 接收 ACK 和 ACTUATOR_FB 上行帧
    - 发布 /status/actuator 反馈执行机构实际状态
    - 忽略 STATUS 周期帧（由 sensor_status_node 处理）
    """

    # --- 上行帧常量 ---
    PACKET_LEN = 64
    HEADER = b'\xFE\xEF'
    TAIL = b'\xFA\xAF'
    REPORT_ACK = 0x01
    REPORT_ACTUATOR_FB = 0x03

    # --- 下行帧常量 (debug 协议) ---
    DOWNLINK_LEN = 54
    DOWNLINK_HEADER = b'\xFE\xFE'
    DOWNLINK_TAIL = b'\xFD\xFD'
    PROTOCOL_VERSION = 0x02

    CMD_CAMERA_LIGHT = 0x10
    CMD_ACTUATOR = 0x30
    OP_SET = 0x00
    FLAG_NEED_ACK = 0x01
    ACTUATOR_SEND_RATE_HZ = 5.0
    STATUS_PUBLISH_RATE_HZ = 5
    SEND_LOOP_RATE_HZ = 20
    # 协议 result=OK、ACCEPTED、EXEC_OK 均表示命令已成功处理。
    SUCCESS_RESULTS = {0x00, 0x02, 0x03}
    RESULT_NAMES = {
        0x00: 'OK',
        0x01: 'SENSOR_ERROR',
        0x02: 'ACCEPTED',
        0x03: 'EXEC_OK',
        0x04: 'EXEC_ERROR',
    }

    def __init__(self):
        self.ip = rospy.get_param('~sensor_ip', '192.168.1.115')
        self.port = rospy.get_param('~sensor_port', 5064)

        self.sock = None
        self.buffer = bytearray()

        # --- 执行器状态缓存 ---
        self.lock = threading.Lock()
        self.seq = 0  # 下行帧序列号 (0-255)

        self.light1 = 0
        self.light2 = 0
        self.heading_servo = 0x80
        self.clamp_servo = 0xFF
        self.drive_cmd = 0
        self.drive_speed = 0
        self.red_light = 0
        self.yellow_light = 0
        self.green_light = 0

        # --- 执行机构实际反馈缓存（来自 ACTUATOR_FB）---
        self.fb_heading = 0x80
        self.fb_clamp = 0xFF
        self.fb_drive_cmd = 0
        self.fb_drive_speed = 0
        self.fb_red = 0
        self.fb_yellow = 0
        self.fb_green = 0
        self.fb_result = 0x00
        self.fb_error = 0x00
        self.fb_actuator_error = 0x00

        # --- 命令与状态数据保存 ---
        self.data_saving_enable = rospy.get_param('~save_data', True)
        self.data_save_dir = os.path.expanduser(rospy.get_param(
            '~save_dir', '~/.ros/auv_logs/actuator_node',
        ))
        self.data_save_file_name = rospy.get_param('~save_file', '')
        self.data_max_file_size = max(1, int(rospy.get_param(
            '~max_file_size', 50 * 1024 * 1024,
        )))
        self.data_flush_every = max(1, int(rospy.get_param('~flush_every', 1)))
        self.data_write_count = 0
        self.data_save_file = None
        self.data_save_base_name = ''
        self.data_file_index = 0
        self.data_file_lock = threading.Lock()
        if self.data_saving_enable:
            self._open_data_save_file()

        # --- 发送线程 ---
        self.is_sending = True
        self.send_thread = None

        # --- ROS 接口 ---
        rospy.Subscriber('/cmd/actuator', ActuatorControl, self.actuator_callback)
        self.status_pub = rospy.Publisher('/status/actuator', ActuatorControl, queue_size=10)
        self.status_timer = rospy.Timer(
            rospy.Duration(1.0 / self.STATUS_PUBLISH_RATE_HZ),
            self._status_timer_callback,
        )

        self.connect()
        rospy.loginfo("sensor_actuator: 已启动（执行器控制+反馈模式）")

    # ============================================================
    # 命令与状态数据保存
    # ============================================================

    def _open_data_save_file(self):
        """创建执行器数据目录并打开本次运行的 JSONL 文件。"""
        if not self.data_save_file_name:
            self.data_save_file_name = datetime.now().strftime(
                'actuator_%Y%m%d_%H%M%S.jsonl'
            )
        self.data_save_base_name = self.data_save_file_name
        self.data_file_index = 0
        os.makedirs(self.data_save_dir, exist_ok=True)
        self._open_data_file(self.data_save_base_name)

    def _open_data_file(self, file_name):
        """打开指定的数据文件分卷。"""
        path = os.path.join(self.data_save_dir, file_name)
        self.data_save_file = open(path, 'a', encoding='utf-8')
        rospy.loginfo(f"sensor_actuator: 命令与状态数据将保存到 {path}")

    def _rotate_data_file_if_needed(self, record_size):
        """写入前检查文件大小，超过上限时切换到下一分卷。"""
        if self.data_save_file is None:
            return

        current_size = self.data_save_file.tell()
        if current_size == 0 or current_size + record_size <= self.data_max_file_size:
            return

        self.data_save_file.flush()
        self.data_save_file.close()
        self.data_file_index += 1
        file_stem, extension = os.path.splitext(self.data_save_base_name)
        self._open_data_file(f'{file_stem}_{self.data_file_index}{extension}')

    @staticmethod
    def _stamp_to_dict(stamp):
        """将 ROS 时间转换为不丢失纳秒精度的可序列化字典。"""
        seconds = int(stamp.secs)
        nanoseconds = int(stamp.nsecs)
        return {
            'secs': seconds,
            'nsecs': nanoseconds,
            'timestamp_ns': seconds * 1000000000 + nanoseconds,
        }

    def _command_to_dict(self, msg):
        """完整提取收到的 ActuatorControl 原始字段。"""
        return {
            'mode': int(msg.mode),
            'light1': int(msg.light1),
            'light2': int(msg.light2),
            'heading_servo': int(msg.heading_servo),
            'clamp_servo': int(msg.clamp_servo),
            'drive_cmd': int(msg.drive_cmd),
            'drive_speed': int(msg.drive_speed),
            'red_light': int(msg.red_light),
            'yellow_light': int(msg.yellow_light),
            'green_light': int(msg.green_light),
        }

    def _status_to_dict(self):
        """在锁内读取完整命令缓存与执行器硬件反馈状态。"""
        with self.lock:
            return {
                'published_status': {
                    'mode': 0,
                    'light1': self.light1,
                    'light2': self.light2,
                    'heading_servo': self.fb_heading,
                    'clamp_servo': self.fb_clamp,
                    'drive_cmd': self.fb_drive_cmd,
                    'drive_speed': self.fb_drive_speed,
                    'red_light': self.fb_red,
                    'yellow_light': self.fb_yellow,
                    'green_light': self.fb_green,
                },
                'command_cache': {
                    'light1': self.light1,
                    'light2': self.light2,
                    'heading_servo': self.heading_servo,
                    'clamp_servo': self.clamp_servo,
                    'drive_cmd': self.drive_cmd,
                    'drive_speed': self.drive_speed,
                    'red_light': self.red_light,
                    'yellow_light': self.yellow_light,
                    'green_light': self.green_light,
                },
                'actuator_feedback': {
                    'heading_servo': self.fb_heading,
                    'clamp_servo': self.fb_clamp,
                    'drive_cmd': self.fb_drive_cmd,
                    'drive_speed': self.fb_drive_speed,
                    'red_light': self.fb_red,
                    'yellow_light': self.fb_yellow,
                    'green_light': self.fb_green,
                    'result': self.fb_result,
                    'result_name': self.RESULT_NAMES.get(
                        self.fb_result, 'UNKNOWN',
                    ),
                    'error': self.fb_error,
                    'actuator_error': self.fb_actuator_error,
                },
            }

    def _save_data_record(self, record):
        """线程安全地追加一条执行器命令或状态记录。"""
        if not self.data_saving_enable:
            return

        try:
            record_line = json.dumps(record, ensure_ascii=False) + '\n'
            with self.data_file_lock:
                if self.data_save_file is None:
                    return
                self._rotate_data_file_if_needed(len(record_line.encode('utf-8')))
                self.data_save_file.write(record_line)
                self.data_write_count += 1
                if self.data_write_count % self.data_flush_every == 0:
                    self.data_save_file.flush()
        except Exception as error:
            rospy.logerr_throttle(5.0, 'sensor_actuator: 保存数据失败: %s', error)

    def _save_command_record(self, msg, received_stamp):
        """保存收到命令的原始字段与回调入口时间戳。"""
        self._save_data_record({
            'event': 'command_received',
            'received_stamp': self._stamp_to_dict(received_stamp),
            'command': self._command_to_dict(msg),
        })

    def _save_status_record(self, stamp):
        """保存当前命令缓存与每个执行器的最新硬件反馈。"""
        self._save_data_record({
            'event': 'status_snapshot',
            'stamp': self._stamp_to_dict(stamp),
            'status': self._status_to_dict(),
        })

    # ============================================================
    # TCP 连接
    # ============================================================

    def connect(self):
        while not rospy.is_shutdown():
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.ip, self.port))
                self.sock.settimeout(1.0)
                rospy.loginfo(f"sensor_actuator: TCP连接 {self.ip}:{self.port}")
                return
            except Exception as e:
                rospy.logerr(f"sensor_actuator: TCP连接失败 {e}, 2s 后重试")
                self.sock = None
                rospy.sleep(2)

    # ============================================================
    # 上行帧校验
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
    # 上行接收与同步
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
                rospy.logerr(f"sensor_actuator: 接收失败: {e}")
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
                rospy.logdebug(f"sensor_actuator: 跳过 {idx} 字节同步到帧头")
                del self.buffer[:idx]

            if len(self.buffer) < self.PACKET_LEN:
                return

            packet = bytes(self.buffer[:self.PACKET_LEN])
            checksum_ok = self.verify_packet(packet)
            tail_ok = (packet[62:64] == self.TAIL)

            if checksum_ok:
                self._parse(packet)
                del self.buffer[:self.PACKET_LEN]
            else:
                if not tail_ok:
                    # 帧尾不匹配 → 假帧头，跳到下一处 FE EF
                    next_idx = self.buffer.find(self.HEADER, 2)
                    if next_idx > 0:
                        rospy.logdebug(
                            f"sensor_actuator: 假帧头（帧尾不匹配），跳过 {next_idx} 字节到下一帧头"
                        )
                        del self.buffer[:next_idx]
                    else:
                        del self.buffer[0]
                else:
                    rospy.logwarn("sensor_actuator: 报文校验失败(XOR错误)，丢弃1字节继续同步")
                    del self.buffer[0]

    # ============================================================
    # 上行帧解析
    # ============================================================

    def _parse(self, packet):
        """解析 ACK 和 ACTUATOR_FB 帧，忽略 STATUS 帧"""
        report_type = packet[4]

        if report_type == self.REPORT_ACK:
            self._parse_ack(packet)
        elif report_type == self.REPORT_ACTUATOR_FB:
            payload_len = packet[9]
            self._parse_actuator_fb(packet, payload_len)
        # STATUS (0x00) 和 CONFIG_FB (0x02) 由 status 节点处理，本节点忽略

    def _parse_ack(self, packet):
        """解析 ACK 帧；发送端不等待 ACK，仅记录下位机拒绝。"""
        ack_seq = packet[3]
        ack_cmd = packet[5]
        ack_result = packet[7]
        ack_error = packet[8]

        cmd_name = {
            self.CMD_CAMERA_LIGHT: 'CAMERA_LIGHT',
            self.CMD_ACTUATOR: 'ACTUATOR',
        }.get(ack_cmd, '0x%02X' % ack_cmd)

        if ack_result not in self.SUCCESS_RESULTS:
            rospy.logwarn_throttle(
                5.0,
                "sensor_actuator: ACK错误 seq=%3d cmd=%-14s result=0x%02X error=0x%02X",
                ack_seq, cmd_name, ack_result, ack_error
            )

    def _parse_actuator_fb(self, packet, payload_len):
        """解析 ACTUATOR_FB 帧并更新锁存反馈状态。"""
        if payload_len < 8:
            rospy.logwarn(f"sensor_actuator: ACTUATOR_FB payload长度不足: {payload_len}")
            return

        payload = packet[10:10 + payload_len]
        fb_result = packet[7]
        fb_error = packet[8]
        actuator_error = payload[7]

        # 更新反馈缓存
        with self.lock:
            self.fb_heading = payload[0]
            self.fb_clamp = payload[1]
            self.fb_drive_cmd = payload[2]
            self.fb_drive_speed = payload[3]
            self.fb_red = payload[4]
            self.fb_yellow = payload[5]
            self.fb_green = payload[6]
            self.fb_result = fb_result
            self.fb_error = fb_error
            self.fb_actuator_error = actuator_error

    def _publish_status(self):
        """根据反馈缓存和命令缓存组装 ActuatorControl 并发布"""
        with self.lock:
            msg = ActuatorControl()
            # 状态消息不作为控制命令，mode 固定为 0。
            msg.mode = 0
            # 补光灯控制已停用，状态固定为 0，避免旧命令被误认为实时反馈。
            msg.light1 = 0
            msg.light2 = 0
            # 执行机构：来自 ACTUATOR_FB 硬件反馈
            msg.heading_servo = self.fb_heading
            msg.clamp_servo = self.fb_clamp
            msg.drive_cmd = self.fb_drive_cmd
            msg.drive_speed = self.fb_drive_speed
            msg.red_light = self.fb_red
            msg.yellow_light = self.fb_yellow
            msg.green_light = self.fb_green

        self.status_pub.publish(msg)

    def _status_timer_callback(self, _event):
        """以 5Hz 持续发布锁存状态，并按固定频率记录当前执行器状态。"""
        self._publish_status()
        self._save_status_record(rospy.Time.now())
        with self.lock:
            values = (
                self.fb_heading,
                self.fb_clamp,
                self.fb_drive_cmd,
                self.fb_drive_speed,
                self.fb_red,
                self.fb_yellow,
                self.fb_green,
                self.fb_result,
                self.RESULT_NAMES.get(self.fb_result, 'UNKNOWN'),
                self.fb_error,
                self.fb_actuator_error,
            )
        rospy.loginfo_throttle(
            10.0,
            "sensor_actuator: 当前执行器状态 heading=%3d clamp=%3d drive=(%d,%3d) "
            "led=(%d,%d,%d) result=0x%02X(%s) error=0x%02X actuator_error=0x%02X",
            *values
        )

    # ============================================================
    # 下行控制
    # ============================================================

    def actuator_callback(self, msg):
        """仅按 mode=2 更新完整执行器命令缓存。"""
        received_stamp = rospy.Time.now()
        self._save_command_record(msg, received_stamp)
        try:
            if msg.mode == 1:
                rospy.logwarn_throttle(
                    5.0,
                    "sensor_actuator: 补光灯控制已停用，忽略 mode=1 命令",
                )
                return

            if msg.mode == 2:
                new_heading = max(0, min(255, msg.heading_servo))
                new_clamp = max(0, min(255, msg.clamp_servo))
                new_drive_cmd = msg.drive_cmd if msg.drive_cmd in (0, 1, 2) else 0
                new_drive_speed = max(0, min(254, msg.drive_speed))
                new_red = 1 if msg.red_light else 0
                new_yellow = 1 if msg.yellow_light else 0
                new_green = 1 if msg.green_light else 0

                with self.lock:
                    self.heading_servo = new_heading
                    self.clamp_servo = new_clamp
                    self.drive_cmd = new_drive_cmd
                    self.drive_speed = new_drive_speed
                    self.red_light = new_red
                    self.yellow_light = new_yellow
                    self.green_light = new_green
                return

            # mode=0 及其他值均不响应。
            if msg.mode != 0:
                rospy.logwarn_throttle(
                    5.0,
                    "sensor_actuator: 忽略不支持的执行器控制模式 mode=%d",
                    msg.mode,
                )
        except Exception as e:
            rospy.logerr(f"sensor_actuator: 执行器控制回调失败: {e}")

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

    def build_actuator_frame(
        self, heading_servo, clamp_servo, drive_cmd, drive_speed,
        red_light, yellow_light, green_light,
    ):
        """构造 ACTUATOR_SET 下行帧 (cmd=0x30, op=0x00)"""
        packet = bytearray(self.DOWNLINK_LEN)
        packet[0:2] = self.DOWNLINK_HEADER
        packet[2] = self.PROTOCOL_VERSION
        packet[3] = self._next_seq()
        packet[4] = self.CMD_ACTUATOR
        packet[5] = self.OP_SET
        packet[6] = 0x00
        packet[7] = 7
        packet[8] = int(heading_servo)
        packet[9] = int(clamp_servo)
        packet[10] = int(drive_cmd)
        packet[11] = int(drive_speed)
        packet[12] = int(red_light)
        packet[13] = int(yellow_light)
        packet[14] = int(green_light)
        packet[40] = self.FLAG_NEED_ACK
        packet[51] = self._calc_downlink_xor(packet)
        packet[52:54] = self.DOWNLINK_TAIL
        return packet

    def _send_frame(self, frame, command_name):
        """发送控制帧；不等待 ACK，接收线程独立处理反馈。"""
        try:
            if self.sock is not None:
                self.sock.sendall(bytes(frame))
        except OSError as error:
            rospy.logerr_throttle(
                5.0,
                'sensor_actuator: 发送 %s 失败: %s', command_name, error,
            )

    def send_loop(self):
        """以 5Hz 持续发送最新完整 mode2 执行器控制帧。"""
        rate = rospy.Rate(self.SEND_LOOP_RATE_HZ)
        interval = 1.0 / self.ACTUATOR_SEND_RATE_HZ
        next_actuator_send = time.monotonic()
        while not rospy.is_shutdown() and self.is_sending:
            try:
                now = time.monotonic()
                if now >= next_actuator_send:
                    with self.lock:
                        command = (
                            self.heading_servo,
                            self.clamp_servo,
                            self.drive_cmd,
                            self.drive_speed,
                            self.red_light,
                            self.yellow_light,
                            self.green_light,
                        )
                    self._send_frame(
                        self.build_actuator_frame(*command), 'ACTUATOR'
                    )
                    while next_actuator_send <= now:
                        next_actuator_send += interval
            except Exception as e:
                rospy.logerr(f"sensor_actuator: 发送失败: {e}")
            rate.sleep()

    # ============================================================

    def spin(self):
        self.send_thread = threading.Thread(target=self.send_loop, daemon=True)
        self.send_thread.start()
        try:
            self.recv_loop()
        finally:
            self.is_sending = False
            self.status_timer.shutdown()
            if self.data_save_file is not None:
                with self.data_file_lock:
                    self.data_save_file.flush()
                    self.data_save_file.close()
                    self.data_save_file = None
            if self.send_thread and self.send_thread.is_alive():
                self.send_thread.join(timeout=2)
            if self.sock:
                try:
                    self.sock.close()
                except Exception:
                    pass


if __name__ == "__main__":
    rospy.init_node('sensor_actuator_node')
    try:
        node = SensorActuatorNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
