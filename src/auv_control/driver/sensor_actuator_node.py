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
"""

import socket
import threading

import rospy

from auv_control.msg import ActuatorControl


class SensorActuatorNode:
    """
    sensor 执行器控制与反馈节点：
    - 独立 TCP 连接到 sensor:5064
    - 订阅 /cmd/actuator，下发 CAMERA_LIGHT_SET + ACTUATOR_SET
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
    SEND_RATE_HZ = 5
    # 协议 result=OK、ACCEPTED、EXEC_OK 均表示命令已成功处理。
    SUCCESS_RESULTS = {0x00, 0x02, 0x03}
    ACK_SUCCESS_RESULTS = SUCCESS_RESULTS
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
        self.ack_timeout = float(rospy.get_param('~ack_timeout', 0.5))
        self.ack_retry_count = max(0, int(rospy.get_param('~ack_retry_count', 2)))
        if self.ack_timeout <= 0.0:
            raise ValueError('~ack_timeout must be positive')

        # --- 执行器状态缓存 ---
        self.lock = threading.Lock()
        self.ack_lock = threading.Lock()
        self.pending_acks = {}
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
        self.camera_light_changed = False
        self.actuator_changed = False

        # --- 执行机构实际反馈缓存（来自 ACTUATOR_FB）---
        self.fb_heading = 0x80
        self.fb_clamp = 0xFF
        self.fb_drive_cmd = 0
        self.fb_drive_speed = 0
        self.fb_red = 0
        self.fb_yellow = 0
        self.fb_green = 0

        # --- 发送线程 ---
        self.is_sending = True
        self.send_thread = None

        # --- ROS 接口 ---
        rospy.Subscriber('/cmd/actuator', ActuatorControl, self.actuator_callback)
        self.status_pub = rospy.Publisher('/status/actuator', ActuatorControl, queue_size=10) 

        self.connect()
        rospy.loginfo("sensor_actuator: 已启动（执行器控制+反馈模式）")

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
        """解析 ACK 帧，记录命令处理结果并唤醒对应发送等待。"""
        ack_seq = packet[3]
        ack_cmd = packet[5]
        ack_result = packet[7]
        ack_error = packet[8]

        cmd_name = {
            self.CMD_CAMERA_LIGHT: 'CAMERA_LIGHT',
            self.CMD_ACTUATOR: 'ACTUATOR',
        }.get(ack_cmd, '0x%02X' % ack_cmd)

        with self.ack_lock:
            pending = self.pending_acks.get(ack_seq)
            if pending is not None:
                pending['result'] = ack_result
                pending['error'] = ack_error
                pending['event'].set()

        if ack_result not in self.ACK_SUCCESS_RESULTS:
            rospy.logwarn(
                "sensor_actuator: ACK错误 seq=%3d cmd=%-14s result=0x%02X error=0x%02X",
                ack_seq, cmd_name, ack_result, ack_error
            )
        else:
            rospy.loginfo(
                "sensor_actuator: ACK成功 seq=%3d cmd=%-14s",
                ack_seq, cmd_name
            )

    def _parse_actuator_fb(self, packet, payload_len):
        """解析 ACTUATOR_FB 帧，更新反馈缓存并发布 /status/actuator"""
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

        log_args = (
            payload[0], payload[1],
            payload[2], payload[3],
            payload[4], payload[5], payload[6],
            fb_result, self.RESULT_NAMES.get(fb_result, 'UNKNOWN'),
            fb_error, actuator_error,
        )
        log_format = (
            "sensor_actuator: 执行机构反馈 heading=%3d clamp=%3d drive=(%d,%3d) "
            "led=(%d,%d,%d) result=0x%02X(%s) error=0x%02X actuator_error=0x%02X"
        )
        if (fb_result in self.SUCCESS_RESULTS and
                fb_error == 0 and actuator_error == 0):
            rospy.loginfo(log_format, *log_args)
        else:
            rospy.logwarn(log_format, *log_args)

        # 发布 /status/actuator
        self._publish_status()

    def _publish_status(self):
        """根据反馈缓存和命令缓存组装 ActuatorControl 并发布"""
        with self.lock:
            msg = ActuatorControl()
            # 状态消息不作为控制命令，mode 固定为 0。
            msg.mode = 0
            # 补光灯：来自最后命令值（CAMERA_LIGHT 无硬件反馈）
            msg.light1 = self.light1
            msg.light2 = self.light2
            # 执行机构：来自 ACTUATOR_FB 硬件反馈
            msg.heading_servo = self.fb_heading
            msg.clamp_servo = self.fb_clamp
            msg.drive_cmd = self.fb_drive_cmd
            msg.drive_speed = self.fb_drive_speed
            msg.red_light = self.fb_red
            msg.yellow_light = self.fb_yellow
            msg.green_light = self.fb_green

        self.status_pub.publish(msg)

    # ============================================================
    # 下行控制
    # ============================================================

    def actuator_callback(self, msg):
        """按 mode 更新补光灯或执行器缓存，并置对应的脏标志。"""
        try:
            if msg.mode == 1:
                new_light1 = max(0, min(100, msg.light1))
                new_light2 = max(0, min(100, msg.light2))
                with self.lock:
                    camera_light_changed = (
                        new_light1 != self.light1 or
                        new_light2 != self.light2
                    )
                    self.light1 = new_light1
                    self.light2 = new_light2
                    if camera_light_changed:
                        self.camera_light_changed = True

                if camera_light_changed:
                    rospy.loginfo(
                        "sensor_actuator: 补光灯控制更新 light=(%3d,%3d)",
                        self.light1,
                        self.light2,
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
                    actuator_changed = (
                        new_heading != self.heading_servo or
                        new_clamp != self.clamp_servo or
                        new_drive_cmd != self.drive_cmd or
                        new_drive_speed != self.drive_speed or
                        new_red != self.red_light or
                        new_yellow != self.yellow_light or
                        new_green != self.green_light
                    )
                    self.heading_servo = new_heading
                    self.clamp_servo = new_clamp
                    self.drive_cmd = new_drive_cmd
                    self.drive_speed = new_drive_speed
                    self.red_light = new_red
                    self.yellow_light = new_yellow
                    self.green_light = new_green
                    if actuator_changed:
                        self.actuator_changed = True

                if actuator_changed:
                    rospy.loginfo(
                        "sensor_actuator: 执行器控制更新 heading=%3d clamp=%3d "
                        "drive=(%d,%3d) led=(%d,%d,%d)",
                        self.heading_servo,
                        self.clamp_servo,
                        self.drive_cmd,
                        self.drive_speed,
                        self.red_light,
                        self.yellow_light,
                        self.green_light,
                    )
                return

            # mode=0 及其他值均不响应。
            if msg.mode != 0:
                rospy.loginfo(
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

    def build_camera_light_frame(self, light1, light2):
        """构造 CAMERA_LIGHT_SET 下行帧 (cmd=0x10, op=0x00)"""
        packet = bytearray(self.DOWNLINK_LEN)
        packet[0:2] = self.DOWNLINK_HEADER
        packet[2] = self.PROTOCOL_VERSION
        packet[3] = self._next_seq()
        packet[4] = self.CMD_CAMERA_LIGHT
        packet[5] = self.OP_SET
        packet[6] = 0x00
        packet[7] = 2
        packet[8] = int(light1)
        packet[9] = int(light2)
        packet[40] = self.FLAG_NEED_ACK
        packet[51] = self._calc_downlink_xor(packet)
        packet[52:54] = self.DOWNLINK_TAIL
        return packet

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

    def _send_frame_with_ack(self, frame, command_name):
        """发送单帧并等待 ACK；超时后使用原序号重发。"""
        seq = frame[3]
        event = threading.Event()
        attempts = self.ack_retry_count + 1
        pending = {'event': event, 'result': None, 'error': None}
        with self.ack_lock:
            self.pending_acks[seq] = pending

        try:
            for attempt in range(1, attempts + 1):
                if self.sock is None:
                    rospy.logwarn(
                        'sensor_actuator: %s seq=%3d 未发送，TCP未连接',
                        command_name, seq,
                    )
                    return False

                event.clear()
                self.sock.sendall(bytes(frame))
                rospy.loginfo(
                    'sensor_actuator: 已发送 %s seq=%3d 第%d/%d次',
                    command_name, seq, attempt, attempts,
                )

                if not event.wait(self.ack_timeout):
                    rospy.logwarn(
                        'sensor_actuator: %s seq=%3d ACK超时（第%d/%d次）',
                        command_name, seq, attempt, attempts,
                    )
                    continue

                with self.ack_lock:
                    result = pending['result']
                    error = pending['error']
                if result in self.ACK_SUCCESS_RESULTS:
                    return True

                rospy.logerr(
                    'sensor_actuator: %s seq=%3d 被下位机拒绝 result=0x%02X error=0x%02X',
                    command_name, seq, result, error,
                )
                return False
        except (OSError, AttributeError) as error:
            rospy.logerr(
                'sensor_actuator: 发送 %s seq=%3d 失败: %s',
                command_name, seq, error,
            )
            return False
        finally:
            with self.ack_lock:
                self.pending_acks.pop(seq, None)

        return False

    def _requeue_camera_light_if_current(self, command):
        """未获确认时，仅在状态未被新命令覆盖时重新排队。"""
        with self.lock:
            if (self.light1, self.light2) == command:
                self.camera_light_changed = True

    def _requeue_actuator_if_current(self, command):
        """未获确认时，仅在状态未被新命令覆盖时重新排队。"""
        with self.lock:
            current = (
                self.heading_servo,
                self.clamp_servo,
                self.drive_cmd,
                self.drive_speed,
                self.red_light,
                self.yellow_light,
                self.green_light,
            )
            if current == command:
                self.actuator_changed = True

    def send_loop(self):
        """5Hz 发送线程: 仅在控制值变化时发送两帧"""
        rate = rospy.Rate(self.SEND_RATE_HZ)
        while not rospy.is_shutdown() and self.is_sending:
            try:
                if self.sock is None:
                    rate.sleep()
                    continue

                with self.lock:
                    camera_command = None
                    actuator_command = None
                    if self.camera_light_changed:
                        camera_command = (self.light1, self.light2)
                        self.camera_light_changed = False
                    if self.actuator_changed:
                        actuator_command = (
                            self.heading_servo,
                            self.clamp_servo,
                            self.drive_cmd,
                            self.drive_speed,
                            self.red_light,
                            self.yellow_light,
                            self.green_light,
                        )
                        self.actuator_changed = False
                    changed = camera_command is not None or actuator_command is not None

                if changed:
                    if camera_command is not None:
                        frame = self.build_camera_light_frame(*camera_command)
                        if not self._send_frame_with_ack(frame, 'CAMERA_LIGHT'):
                            self._requeue_camera_light_if_current(camera_command)

                    if actuator_command is not None:
                        frame = self.build_actuator_frame(*actuator_command)
                        if not self._send_frame_with_ack(frame, 'ACTUATOR'):
                            self._requeue_actuator_if_current(actuator_command)
                    rospy.logdebug(
                        "sensor_actuator: 已发送控制帧 seq=%3d light=(%3d,%3d) heading=%3d "
                        "clamp=%3d drive=(%d,%3d) led=(%d,%d,%d)",
                        self.seq,
                        self.light1, self.light2,
                        self.heading_servo, self.clamp_servo,
                        self.drive_cmd, self.drive_speed,
                        self.red_light, self.yellow_light, self.green_light
                    )
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
