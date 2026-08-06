#! /home/xhy/xhy_env36/bin/python
# -*- coding: utf-8 -*-
"""
名称：debug_driver_v2.py
功能：调试驱动V2，支持定深(02)/定深定向(03)/定点(04)三种模式
      通过 TCP 发送 54 字节 ROV 扩展控制帧到 AUV
作者：BroXu
监听：/cmd/pose/lla (PoseLLAcmd.msg，经纬度坐标系)
发布：/status/auv (AUVData.msg)
      /status/vel (geometry_msgs/TwistStamped)
记录：
2026.7.11
    基于 debug_driver.py 重构，新增定深(mode=2)和定深定向(mode=3)模式
    协议严格遵循《200502AUV扩展口协议》，偏移44固定为0x00
    力/力矩支持 0-10000 原始值直接写入，移除补光灯控制
    统一 loginfo 中文输出：CMD/SEND 带 mode 标注，定长小数对齐
2026.7.13
    调整至 driver 目录，归入硬件驱动层
    下层控制接口使用 LLA 坐标系的 PoseLLAcmd 整包消息。
    上行 AUV 状态话题调整为 /status/auv。
2026.7.16
    新增 /status/vel 速度话题，使用 TwistStamped 发布 base_link 坐标系下的三轴线速度和角速度。
    线速度单位为 m/s，角速度由 deg/s 转换为 rad/s。
2026.7.18
    明确下位机线速度参考点为 IMU/惯导点，frame_id 改为 imu。
2026.7.24
    原始调试报文按 debug_v2_raw 子目录保存，避免与其他节点数据混存。
2026.7.31
    原始报文文件达到 50 MiB 后自动分卷，文件名追加下划线编号。
2026.7.31
    订阅旧版状态端口转发的 /dvl/altitude，优先填充上行状态的 DVL 高度字段。
2026.8.5
    使用 /nav 的有效 INS 位姿、DVL 对底速度和 IMU 角速度发布 20Hz 状态反馈。
    debug 上行数据改为补充控制模式、目标、推力和设备状态，增加 nav/DVL/debug 超时保护。
    控制下发循环改用单调时钟维持配置频率，默认 20Hz。
    DVL 状态1改为低频样本软保持，状态0增加硬无效去抖，并仅在失效状态迁移时告警。
"""

import json
import math
import os
from datetime import datetime

import rospy
import socket
import struct
import threading
import time
from auv_control.msg import AUVData, NavData, PoseLLAcmd
from functools import reduce
from geometry_msgs.msg import TwistStamped
from debug_protocol import (
    DebugFrameBuffer,
    LowPassFilter,
    MovingAverageFilter,
    decode_status_words,
    require_finite,
)
from nav_driver import (
    DVL_AVAILABILITY_FRESH,
    DVL_AVAILABILITY_HARD_INVALID,
    DVL_AVAILABILITY_HOLDING,
    DVL_AVAILABILITY_STALE,
    DVL_AVAILABILITY_WAITING,
    DvlBottomTracker,
    angular_velocity_rad,
    imu_angular_velocity_deg,
    ins_data_valid,
    nav_pose_and_imu_valid,
)

# 运行模式常量
MODE_DEPTH       = 2   # 定深：闭环深度，其余开环力控
MODE_DEPTH_HDG   = 3   # 定深定向：闭环深度+航向，其余开环力控
MODE_DPROV       = 4   # 动力定位ROV：闭环经纬度+深度+姿态


class ControlTarget:
    """统一控制目标结构体"""
    def __init__(self):
        self.valid = False
        self.mode = MODE_DPROV
        self.longitude = 0.0
        self.latitude = 0.0
        self.depth = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.speed = 0.0
        self.tx = 0      # X轴力 -10000～10000
        self.ty = 0      # Y轴力 -10000～10000
        self.tz = 0      # Z轴力 -10000～10000
        self.mx = 0      # 绕X轴力矩 -10000～10000
        self.my = 0      # 绕Y轴力矩 -10000～10000
        self.mz = 0      # 绕Z轴力矩 -10000～10000


class DebugDataPacket:
    """110字节调试协议解析结构体"""
    def __init__(self):
        self.mode = 0
        self.temperature = 0.0
        self.control_voltage = 0.0
        self.power_current = 0.0
        self.water_leak = 0
        self.sensor_status = 0
        self.sensor_update = 0
        self.fault_status = 0
        self.power_status = 0
        self.force_commands = [0] * 6
        self.euler_angles = [0.0] * 3
        self.angular_velocity = [0.0] * 3
        self.linear_velocity = [0.0] * 3
        self.navigation_coords = [0.0] * 2
        self.depth = 0.0
        self.depth_filtered = 0.0
        self.depth_ma = 0.0
        self.altitude = 0.0
        self.target_longitude = 0.0
        self.target_latitude = 0.0
        self.target_depth = 0.0
        self.target_roll = 0.0
        self.target_pitch = 0.0
        self.target_yaw = 0.0
        self.target_altitude = 0.0
        self.target_speed = 0.0
        self.utc_time = [0] * 6
        self.checksum = 0

    # 模式名称映射
    MODE_NAMES = {0: "待机", 2: "定深", 3: "定深定向", 4: "动力定位"}


class DebugDriverV2:
    """
    调试串口驱动 V2
    支持三种模式：定深(02)、定深定向(03)、定点DPROV(04)
    TCP连接 192.168.1.115:5063
    """
    def __init__(self, ip=None, port=None):
        ip = ip or rospy.get_param("~debug_ip", "192.168.1.115")
        port = port or rospy.get_param("~debug_port", 5063)
        self.send_rate_hz = float(rospy.get_param("~send_rate_hz", 20.0))
        if not math.isfinite(self.send_rate_hz) or self.send_rate_hz <= 0.0:
            rospy.logerr(
                "debug_driver_v2: send_rate_hz 无效，回退为 20Hz 后继续等待控制"
            )
            self.send_rate_hz = 20.0
        self.send_period_s = 1.0 / self.send_rate_hz

        self.dvl_velocity_timeout_sec = float(rospy.get_param(
            '~dvl_velocity_timeout_sec', 2.5
        ))
        if (
                not math.isfinite(self.dvl_velocity_timeout_sec)
                or self.dvl_velocity_timeout_sec <= 0.0):
            rospy.logerr(
                'debug_driver_v2: dvl_velocity_timeout_sec 无效，回退为 2.5s'
            )
            self.dvl_velocity_timeout_sec = 2.5
        self.dvl_hard_invalid_timeout_sec = float(rospy.get_param(
            '~dvl_hard_invalid_timeout_sec', 0.15
        ))
        if (
                not math.isfinite(self.dvl_hard_invalid_timeout_sec)
                or self.dvl_hard_invalid_timeout_sec <= 0.0):
            rospy.logerr(
                'debug_driver_v2: dvl_hard_invalid_timeout_sec 无效，回退为 0.15s'
            )
            self.dvl_hard_invalid_timeout_sec = 0.15
        self.nav_timeout_sec = float(rospy.get_param('~nav_timeout_sec', 0.5))
        if not math.isfinite(self.nav_timeout_sec) or self.nav_timeout_sec <= 0.0:
            rospy.logerr('debug_driver_v2: nav_timeout_sec 无效，回退为 0.5s')
            self.nav_timeout_sec = 0.5
        self.debug_supplement_timeout_sec = float(rospy.get_param(
            '~debug_supplement_timeout_sec', 0.5
        ))
        if (
                not math.isfinite(self.debug_supplement_timeout_sec)
                or self.debug_supplement_timeout_sec <= 0.0):
            rospy.logerr(
                'debug_driver_v2: debug_supplement_timeout_sec 无效，回退为 0.5s'
            )
            self.debug_supplement_timeout_sec = 0.5

        # 原始报文保存
        self.raw_saving_enable = rospy.get_param("~save_raw_data", False)
        self.raw_save_dir = os.path.expanduser(rospy.get_param("~raw_save_dir", "~/.ros/auv_logs"))
        self.raw_save_subdir = "debug_v2_raw"
        self.raw_save_file_name = rospy.get_param("~raw_save_file", "")
        self.raw_max_file_size = max(1, int(rospy.get_param("~raw_max_file_size", 50 * 1024 * 1024)))
        self.raw_flush_every = max(1, int(rospy.get_param("~raw_flush_every", 1)))
        self.raw_write_count = 0
        self.raw_save_file = None
        self.raw_save_base_name = ""
        self.raw_file_index = 0

        self.server_address = (ip, port)
        self.tcp_sock = None
        self.latest_debug_data = None
        self.latest_debug_received_at = None
        self.latest_nav_received_at = None
        self.dvl_tracker = DvlBottomTracker(
            self.dvl_velocity_timeout_sec,
            self.dvl_hard_invalid_timeout_sec,
        )
        self.last_dvl_availability = None

        self.lock = threading.RLock()
        self.socket_lock = threading.RLock()
        self.connect_lock = threading.Lock()
        self.target = ControlTarget()
        self.last_control_time = 0
        self.send_thread = None
        self.recv_thread = None

        # 深度滤波器
        self.depth_lpf = LowPassFilter(alpha=0.2)
        self.depth_ma = MovingAverageFilter(window_size=5)

        if self.raw_saving_enable:
            self.open_raw_save_file()

        # 接收由 auv_tf_handler 转换后的 LLA 整包控制指令。
        rospy.Subscriber('/cmd/pose/lla', PoseLLAcmd, self.control_cmd_callback)
        rospy.Subscriber('/nav', NavData, self.nav_callback, queue_size=20)
        self.data_pub = rospy.Publisher('/status/auv', AUVData, queue_size=10)
        self.velocity_pub = rospy.Publisher('/status/vel', TwistStamped, queue_size=10)
        self.nav_watchdog_timer = rospy.Timer(
            rospy.Duration(0.1), self._nav_watchdog_callback
        )
        rospy.loginfo(
            "debug_driver_v2: 已启动，监听 /cmd/pose/lla 和 /nav，下发频率 %.1f Hz",
            self.send_rate_hz)

    def open_raw_save_file(self):
        """打开原始报文保存文件"""
        if not self.raw_save_file_name:
            self.raw_save_file_name = datetime.now().strftime("debug_v2_raw_%Y%m%d_%H%M%S.jsonl")
        self.raw_save_base_name = self.raw_save_file_name
        self.raw_file_index = 0
        save_dir = os.path.join(self.raw_save_dir, self.raw_save_subdir)
        os.makedirs(save_dir, exist_ok=True)
        self._open_raw_file(save_dir, self.raw_save_base_name)

    def _open_raw_file(self, save_dir, file_name):
        """打开指定的原始报文分卷文件"""
        path = os.path.join(save_dir, file_name)
        self.raw_save_file = open(path, "a", encoding="utf-8")
        rospy.loginfo(f"debug_driver_v2: 原始报文保存到 {path}")

    def _rotate_raw_file_if_needed(self, record_size):
        """在写入前检查文件大小，超过上限时切换到下一个分卷"""
        if self.raw_save_file is None:
            return

        current_size = self.raw_save_file.tell()
        if current_size == 0 or current_size + record_size <= self.raw_max_file_size:
            return

        self.raw_save_file.flush()
        self.raw_save_file.close()
        self.raw_file_index += 1
        file_stem, extension = os.path.splitext(self.raw_save_base_name)
        file_name = f"{file_stem}_{self.raw_file_index}{extension}"
        save_dir = os.path.join(self.raw_save_dir, self.raw_save_subdir)
        self._open_raw_file(save_dir, file_name)

    def save_raw_packet(self, packet, checksum_ok):
        """保存原始报文"""
        if not self.raw_saving_enable or self.raw_save_file is None:
            return
        event = {
            "pc_time": rospy.Time.now().to_sec(),
            "source": "debug_v2",
            "packet_len": len(packet),
            "checksum_ok": bool(checksum_ok),
            "packet_hex": " ".join("{:02x}".format(byte) for byte in packet),
        }
        try:
            event_line = json.dumps(event, ensure_ascii=False) + "\n"
            self._rotate_raw_file_if_needed(len(event_line.encode("utf-8")))
            self.raw_save_file.write(event_line)
            self.raw_write_count += 1
            if self.raw_write_count % self.raw_flush_every == 0:
                self.raw_save_file.flush()
        except Exception as e:
            rospy.logerr(f"debug_driver_v2: 保存原始报文失败: {e}")

    # ── 上行解析（与 V1 一致）────────────────────────────────────────

    def calc_debug_checksum(self, packet):
        """计算调试协议校验和（0-106字节异或）"""
        return reduce(lambda x, y: x ^ y, packet[:107], 0)

    def parse_debug_packet(self, packet):
        """解析 110 字节上行调试报文"""
        data = DebugDataPacket()
        try:
            data.mode = packet[2]
            data.temperature = struct.unpack('>h', packet[3:5])[0] / 100.0
            data.control_voltage = struct.unpack('>h', packet[5:7])[0] / 100.0
            data.power_current = struct.unpack('>h', packet[7:9])[0] / 100.0
            data.water_leak = packet[9]
            (
                data.sensor_status,
                data.sensor_update,
                data.fault_status,
                data.power_status,
            ) = decode_status_words(packet)
            data.force_commands = list(struct.unpack('>6h', packet[16:28]))
            data.euler_angles = [x / 100.0 for x in struct.unpack('>3h', packet[28:34])]
            data.angular_velocity = [x / 100.0 for x in struct.unpack('>3h', packet[34:40])]
            data.linear_velocity = [x / 100.0 for x in struct.unpack('>3h', packet[40:46])]
            data.navigation_coords = [x / 10000000.0 for x in struct.unpack('<2i', packet[46:54])]
            raw_depth = struct.unpack('<f', packet[54:58])[0]
            data.depth = raw_depth
            data.altitude = struct.unpack('<f', packet[58:62])[0]
            data.target_longitude = struct.unpack('<i', packet[66:70])[0] / 10000000.0
            data.target_latitude = struct.unpack('<i', packet[70:74])[0] / 10000000.0
            data.target_depth = struct.unpack('<f', packet[74:78])[0]
            data.target_roll = struct.unpack('>h', packet[78:80])[0] / 100.0
            data.target_pitch = struct.unpack('>h', packet[80:82])[0] / 100.0
            data.target_yaw = struct.unpack('>h', packet[82:84])[0] / 100.0
            data.target_altitude = struct.unpack('<f', packet[84:88])[0]
            data.target_speed = struct.unpack('>H', packet[88:90])[0] / 100.0
            data.utc_time = list(packet[90:95])
            data.utc_time.append(struct.unpack('<f', packet[95:99])[0])
            data.checksum = packet[107]
            require_finite(
                (
                    data.temperature,
                    data.control_voltage,
                    data.power_current,
                    *data.euler_angles,
                    *data.angular_velocity,
                    *data.linear_velocity,
                    *data.navigation_coords,
                    data.depth,
                    data.altitude,
                    data.target_longitude,
                    data.target_latitude,
                    data.target_depth,
                    data.target_roll,
                    data.target_pitch,
                    data.target_yaw,
                    data.target_altitude,
                    data.target_speed,
                    data.utc_time[5],
                ),
                '调试报文',
            )
            # 整包校验通过后再更新有状态滤波器，坏包不会污染后续深度。
            data.depth_filtered = self.depth_lpf.update(raw_depth)
            data.depth_ma = self.depth_ma.update(raw_depth)
        except Exception as e:
            rospy.logerr(f"debug_driver_v2: 数据解析错误: {e}")
            return None
        return data

    def _fresh_debug_snapshot(self, now):
        """返回未超时的 debug 补充数据；缺失或超时返回 None。"""
        with self.lock:
            parsed = self.latest_debug_data
            received_at = self.latest_debug_received_at
        if parsed is None or received_at is None:
            rospy.logerr_throttle(
                2.0, 'debug_driver_v2: 尚未收到 debug 补充数据，使用安全默认值'
            )
            return None
        age = max(0.0, (now - received_at).to_sec())
        if age > self.debug_supplement_timeout_sec:
            rospy.logerr_throttle(
                2.0,
                'debug_driver_v2: debug 补充数据超时 %.3fs（限制 %.3fs），使用安全默认值',
                age,
                self.debug_supplement_timeout_sec,
            )
            return None
        return parsed

    def _build_auv_data(self, nav, debug, dvl_sample, stamp):
        """以 nav 为主数据、debug 为补充组装 AUVData。"""
        msg = AUVData()
        msg.header.stamp = stamp
        msg.header.frame_id = 'nav'
        msg.pose.latitude = nav.latitude
        msg.pose.longitude = nav.longitude
        msg.pose.depth = nav.depth
        msg.pose.roll = nav.roll
        msg.pose.pitch = nav.pitch
        msg.pose.yaw = nav.heading

        dvl_velocity = (0.0, 0.0, 0.0)
        if dvl_sample is not None:
            dvl_velocity = dvl_sample.velocity
            msg.pose.altitude = dvl_sample.altitude
            msg.pose.speed = dvl_velocity[0]
        msg.linear_velocity = list(dvl_velocity)
        angular_velocity = imu_angular_velocity_deg(nav)
        msg.angular_velocity = list(angular_velocity)

        if debug is not None:
            msg.control_mode = debug.mode
            msg.target.longitude = debug.target_longitude
            msg.target.latitude = debug.target_latitude
            msg.target.depth = debug.target_depth
            msg.target.roll = debug.target_roll
            msg.target.pitch = debug.target_pitch
            msg.target.yaw = debug.target_yaw
            msg.target.altitude = debug.target_altitude
            msg.target.speed = debug.target_speed
            msg.motor_force.TX = debug.force_commands[0]
            msg.motor_force.TY = debug.force_commands[1]
            msg.motor_force.TZ = debug.force_commands[2]
            msg.motor_force.MX = debug.force_commands[3]
            msg.motor_force.MY = debug.force_commands[4]
            msg.motor_force.MZ = debug.force_commands[5]
            msg.sensor.temperature = debug.temperature
            msg.sensor.voltage = debug.control_voltage
            msg.sensor.current = debug.power_current
            msg.sensor.battery = 0
            msg.sensor.leak_alarm = bool(debug.water_leak)
            msg.sensor.sensor_valid = debug.sensor_status
            msg.sensor.sensor_updated = debug.sensor_update
            msg.sensor.fault_status = debug.fault_status
            msg.sensor.power_status = debug.power_status
            msg.time.year = debug.utc_time[0]
            msg.time.month = debug.utc_time[1]
            msg.time.day = debug.utc_time[2]
            msg.time.hour = debug.utc_time[3]
            msg.time.minute = debug.utc_time[4]
            msg.time.second = debug.utc_time[5]
        return msg, angular_velocity

    def nav_callback(self, nav):
        """隔离单帧解析异常，避免 ROS 回调线程因坏帧中断。"""
        try:
            self._process_nav_callback(nav)
        except Exception as error:
            rospy.logerr_throttle(
                1.0,
                'debug_driver_v2: nav 帧处理异常：%s，等待后续有效帧',
                error,
            )

    def _report_dvl_availability(self, availability):
        """只在真正失效或恢复时记录状态，避免状态1/2切换刷屏。"""
        previous = self.last_dvl_availability
        self.last_dvl_availability = availability
        if availability == previous:
            return
        if availability == DVL_AVAILABILITY_WAITING:
            if previous is None:
                rospy.logwarn('debug_driver_v2: 等待首个有效 DVL 对底速度')
            return
        if availability == DVL_AVAILABILITY_HARD_INVALID:
            rospy.logerr(
                'debug_driver_v2: DVL 持续硬无效，停止发布 /status/vel'
            )
            return
        if availability == DVL_AVAILABILITY_STALE:
            rospy.logerr(
                'debug_driver_v2: DVL 未获得有效新样本超过 %.2fs，停止发布 /status/vel',
                self.dvl_velocity_timeout_sec,
            )
            return
        if availability == DVL_AVAILABILITY_HOLDING:
            # 状态1是正常的低频采样间隔，不记录 warning/error。
            return
        if (
                availability == DVL_AVAILABILITY_FRESH
                and previous in (
                    DVL_AVAILABILITY_HARD_INVALID,
                    DVL_AVAILABILITY_STALE,
                    DVL_AVAILABILITY_WAITING,
                )):
            rospy.loginfo('debug_driver_v2: DVL 对底速度已恢复')

    def _process_nav_callback(self, nav):
        """以有效 nav 帧触发 AUV 位姿及速度状态发布。"""
        now = rospy.Time.now()
        if not ins_data_valid(nav.ins_status):
            rospy.logerr_throttle(
                1.0,
                'debug_driver_v2: INS 数据无效，ins_status=0x%04X，等待恢复',
                int(nav.ins_status),
            )
            return
        if not nav_pose_and_imu_valid(nav):
            rospy.logerr_throttle(
                1.0, 'debug_driver_v2: nav 位姿或 IMU 角速度无效，等待恢复'
            )
            return

        now_sec = now.to_sec()
        with self.lock:
            self.latest_nav_received_at = now
            updated = self.dvl_tracker.update(
                nav.dvl_status,
                (nav.dvl_vx, nav.dvl_vy, nav.dvl_vz),
                nav.dvl_altitude,
                now_sec,
            )
            dvl_sample = self.dvl_tracker.current(now_sec)
            dvl_availability = self.dvl_tracker.availability(now_sec)
        if int(nav.dvl_status) in (2, 4) and not updated:
            rospy.logerr_throttle(
                1.0, 'debug_driver_v2: DVL 对底速度包含无效数值，保持上一有效值'
            )
        self._report_dvl_availability(dvl_availability)

        debug = self._fresh_debug_snapshot(now)
        stamp = nav.header.stamp
        if stamp == rospy.Time():
            stamp = now
        msg, angular_velocity_deg = self._build_auv_data(
            nav, debug, dvl_sample, stamp
        )
        self.data_pub.publish(msg)

        if dvl_sample is None:
            return

        # DVL 对底速度直接使用下位机轴顺序；IMU 角速度只做单位转换。
        velocity_msg = TwistStamped()
        velocity_msg.header.stamp = msg.header.stamp
        velocity_msg.header.frame_id = "imu"
        velocity_msg.twist.linear.x = dvl_sample.velocity[0]
        velocity_msg.twist.linear.y = dvl_sample.velocity[1]
        velocity_msg.twist.linear.z = dvl_sample.velocity[2]
        angular_velocity = angular_velocity_rad(angular_velocity_deg)
        velocity_msg.twist.angular.x = angular_velocity[0]
        velocity_msg.twist.angular.y = angular_velocity[1]
        velocity_msg.twist.angular.z = angular_velocity[2]
        self.velocity_pub.publish(velocity_msg)

    def _nav_watchdog_callback(self, unused_event):
        """nav 无有效帧超过阈值时持续告警，等待数据自行恢复。"""
        del unused_event
        with self.lock:
            received_at = self.latest_nav_received_at
        if received_at is None:
            rospy.logerr_throttle(
                2.0, 'debug_driver_v2: 尚未收到有效 nav 数据，等待恢复'
            )
            return
        age = max(0.0, (rospy.Time.now() - received_at).to_sec())
        if age > self.nav_timeout_sec:
            rospy.logerr_throttle(
                1.0,
                'debug_driver_v2: nav 数据超时 %.3fs（限制 %.3fs），已停止状态更新',
                age,
                self.nav_timeout_sec,
            )

    # ── 下行组包（严格遵循协议）─────────────────────────────────────

    def build_54_packet(self):
        """
        构建 54 字节 ROV 扩展控制帧，严格遵循《200502AUV扩展口协议》
        ┌───────┬──────┬─────────────────────────────────────┐
        │ 偏移  │ 字节 │ 说明                                │
        ├───────┼──────┼─────────────────────────────────────┤
        │  0- 1 │   2  │ 报文头 FE FE                        │
        │  2- 3 │   2  │ 船号 00 01                          │
        │  4    │   1  │ 0x30 ROV扩展指令                    │
        │  5    │   1  │ 设备运行模式 02/03/04               │
        │  6    │   1  │ 开环闭环 01=闭环                    │
        │  7    │   1  │ 坐标系 00=经纬度                    │
        │  8-15 │   8  │ 期望经纬度 int32×2 ×1e7            │
        │ 16-19 │   4  │ 期望深度 float                      │
        │ 20-23 │   4  │ 期望横滚角 float                    │
        │ 24-27 │   4  │ 期望俯仰角 float                    │
        │ 28-31 │   4  │ 期望航向角 float                    │
        │ 32-43 │  12  │ 力/力矩 int16×6 (TX,TY,TZ,MX,MY,MZ)│
        │ 44    │   1  │ 是否打开模式 00=跟踪                │
        │ 45-50 │   6  │ 预留 填0                            │
        │ 51    │   1  │ 异或校验(0-50)                      │
        │ 52-53 │   2  │ 数据尾 FD FD                        │
        └───────┴──────┴─────────────────────────────────────┘
        """
        packet = bytearray(54)

        # 0-1: 报文头 FE FE
        packet[0:2] = b'\xFE\xFE'
        # 2-3: 船号 00 01
        packet[2:4] = b'\x00\x01'
        # 4: 指令类型 0x30 ROV扩展指令
        packet[4] = 0x30
        # 5: 设备运行模式（02=定深 / 03=定深定向 / 04=动力定位）
        packet[5] = self.target.mode
        # 6: 闭环模式 01
        packet[6] = 0x01
        # 7: 坐标系 00=经纬度
        packet[7] = 0x00

        # 8-15: 期望经纬度 int32 ×1e7（仅定点模式有效）
        lon = int(self.target.longitude * 1e7)
        lat = int(self.target.latitude * 1e7)
        packet[8:12] = struct.pack('<i', lon)
        packet[12:16] = struct.pack('<i', lat)

        # 16-19: 期望深度 float32
        packet[16:20] = struct.pack('<f', self.target.depth)

        # 20-23: 期望横滚角 float32
        packet[20:24] = struct.pack('<f', self.target.roll)

        # 24-27: 期望俯仰角 float32
        packet[24:28] = struct.pack('<f', self.target.pitch)

        # 28-31: 期望航向角 float32
        packet[28:32] = struct.pack('<f', self.target.yaw)

        # 32-43: 6自由度力/力矩 int16×6 大端序，协议范围 -10000～10000
        raw_forces = (
            self.target.tx, self.target.ty, self.target.tz,
            self.target.mx, self.target.my, self.target.mz,
        )
        forces = []
        for value in raw_forces:
            limited = max(-10000, min(10000, int(value)))
            if limited != value:
                rospy.logwarn_throttle(
                    1.0,
                    "debug_driver_v2: 力/力矩 %s 超出协议范围，已限制为 %d",
                    value,
                    limited,
                )
            forces.append(limited)
        struct.pack_into('>6h', packet, 32,
            forces[0], forces[1], forces[2],
            forces[3], forces[4], forces[5])

        # 44: 是否打开模式，严格保持 0x00（跟踪模式）
        packet[44] = 0x00

        # 45-50: 预留 填0
        for i in range(45, 51):
            packet[i] = 0x00

        # 51: 异或校验（0-50字节）
        xor = 0
        for i in range(0, 51):
            xor ^= packet[i]
        packet[51] = xor

        # 52-53: 数据尾 FD FD
        packet[52:54] = b'\xFD\xFD'

        return packet

    # ── TCP 连接管理 ─────────────────────────────────────────────

    def _disconnect_socket(self, expected_socket=None):
        """原子摘除并关闭当前连接；旧线程不得关闭新连接。"""
        with self.socket_lock:
            if (
                    expected_socket is not None
                    and self.tcp_sock is not expected_socket):
                return
            sock = self.tcp_sock
            self.tcp_sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def connect(self):
        """TCP 连接（阻塞重试直到成功）"""
        with self.connect_lock:
            while not rospy.is_shutdown():
                with self.socket_lock:
                    if self.tcp_sock is not None:
                        return
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    sock.settimeout(3.0)
                    sock.connect(self.server_address)
                    sock.settimeout(1)
                    if rospy.is_shutdown():
                        sock.close()
                        return
                    with self.socket_lock:
                        self.tcp_sock = sock
                    rospy.loginfo(
                        f"debug_driver_v2: TCP连接成功 {self.server_address}")
                    return
                except Exception as e:
                    try:
                        sock.close()
                    except OSError:
                        pass
                    rospy.logwarn(
                        f"debug_driver_v2: TCP连接失败 {self.server_address}: "
                        f"{e}, 2秒后重试...")
                    rospy.sleep(2)

    # ── 收发线程 ─────────────────────────────────────────────

    def recv_loop(self):
        """接收循环（子线程）"""
        active_socket = None
        frame_buffer = DebugFrameBuffer()
        while not rospy.is_shutdown():
            with self.socket_lock:
                sock = self.tcp_sock
            if sock is None:
                self.connect()
                continue
            if sock is not active_socket:
                active_socket = sock
                frame_buffer = DebugFrameBuffer()
            try:
                data = sock.recv(512)
                if not data:
                    raise ConnectionError('对端已关闭 TCP 连接')
                with self.socket_lock:
                    if self.tcp_sock is not sock:
                        continue
                for packet in frame_buffer.feed(data):
                    checksum_ok = self.calc_debug_checksum(packet) == packet[107]
                    self.save_raw_packet(packet, checksum_ok)
                    if not checksum_ok:
                        rospy.logwarn("debug_driver_v2: 校验和错误")
                        continue
                    parsed = self.parse_debug_packet(packet)
                    if parsed is not None:
                        with self.lock:
                            self.latest_debug_data = parsed
                            self.latest_debug_received_at = rospy.Time.now()
            except socket.timeout:
                continue
            except Exception as e:
                rospy.logwarn(f"debug_driver_v2: TCP连接错误: {e}, 重连中...")
                self._disconnect_socket(sock)
                active_socket = None
                frame_buffer = DebugFrameBuffer()
                self.connect()

    def send_loop(self):
        """按单调时钟以配置频率发送最新完整控制目标。"""
        next_send = time.monotonic()
        while not rospy.is_shutdown():
            monotonic_now = time.monotonic()
            if monotonic_now < next_send:
                time.sleep(min(next_send - monotonic_now, self.send_period_s))
                continue
            while next_send <= monotonic_now:
                next_send += self.send_period_s

            now = time.time()
            packet = None
            target_snapshot = None
            timed_out = False
            with self.lock:
                # 5秒未收到任一有效控制量更新则停止发送。
                if self.target.valid and (now - self.last_control_time > 5):
                    self.target.valid = False
                    timed_out = True

                if self.target.valid:
                    packet = self.build_54_packet()
                    target_snapshot = (
                        self.target.mode,
                        self.target.longitude,
                        self.target.latitude,
                        self.target.depth,
                        self.target.roll,
                        self.target.pitch,
                        self.target.yaw,
                        self.target.tx,
                        self.target.ty,
                        self.target.tz,
                        self.target.mx,
                        self.target.my,
                        self.target.mz,
                    )

            if timed_out:
                rospy.loginfo("debug_driver_v2: 5s未收到控制消息，停止发送！")

            if packet is not None:
                with self.socket_lock:
                    sock = self.tcp_sock
                if sock is None:
                    continue
                try:
                    sock.sendall(packet)
                    mode_name = DebugDataPacket.MODE_NAMES.get(
                        target_snapshot[0], f"未知({target_snapshot[0]})")
                    rospy.loginfo_throttle(2,
                        "debug_driver_v2: SEND mode=%d(%s) lon=%12.7f lat=%12.7f depth=%7.2f "
                        "roll=%6.1f pitch=%6.1f yaw=%6.1f "
                        "F=[%5d,%5d,%5d] M=[%5d,%5d,%5d]",
                        target_snapshot[0], mode_name,
                        target_snapshot[1], target_snapshot[2],
                        target_snapshot[3],
                        target_snapshot[4], target_snapshot[5], target_snapshot[6],
                        target_snapshot[7], target_snapshot[8], target_snapshot[9],
                        target_snapshot[10], target_snapshot[11], target_snapshot[12],
                    )
                except Exception as e:
                    rospy.logerr(f"debug_driver_v2: 发送扩展指令包错误: {e}")
                    self._disconnect_socket(sock)

    # ── 回调 ─────────────────────────────────────────────

    def control_cmd_callback(self, msg):
        """接收 LLA 坐标系的完整控制指令。"""
        if msg.mode not in (MODE_DEPTH, MODE_DEPTH_HDG, MODE_DPROV):
            rospy.logwarn("debug_driver_v2: 忽略不支持的控制模式 %d", msg.mode)
            return

        try:
            values = require_finite((
                msg.target.longitude,
                msg.target.latitude,
                msg.target.depth,
                msg.target.roll,
                msg.target.pitch,
                msg.target.yaw,
                msg.target.speed,
                msg.force.TX,
                msg.force.TY,
                msg.force.TZ,
                msg.force.MX,
                msg.force.MY,
                msg.force.MZ,
            ), '控制目标')
        except ValueError as error:
            rospy.logwarn('debug_driver_v2: 忽略无效控制目标: %s', error)
            return
        with self.lock:
            self.target.mode = msg.mode
            (
                self.target.longitude,
                self.target.latitude,
                self.target.depth,
                self.target.roll,
                self.target.pitch,
                self.target.yaw,
                self.target.speed,
                self.target.tx,
                self.target.ty,
                self.target.tz,
                self.target.mx,
                self.target.my,
                self.target.mz,
            ) = values
            self.target.valid = True
            self.last_control_time = time.time()

    # ── 主循环 ─────────────────────────────────────────────

    def run(self):
        """主线程"""
        while not rospy.is_shutdown():
            try:
                with self.socket_lock:
                    connected = self.tcp_sock is not None
                if not connected:
                    self.connect()
                    time.sleep(2)
                    continue

                if not self.recv_thread or not self.recv_thread.is_alive():
                    rospy.loginfo("debug_driver_v2: 启动接收线程")
                    self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
                    self.recv_thread.start()

                if not self.send_thread or not self.send_thread.is_alive():
                    rospy.loginfo("debug_driver_v2: 启动发送线程")
                    self.send_thread = threading.Thread(target=self.send_loop, daemon=True)
                    self.send_thread.start()

                time.sleep(0.01)  # 100Hz

            except Exception as e:
                rospy.logerr(f"debug_driver_v2: 运行错误: {e}")
                self._disconnect_socket()
                time.sleep(2)

        # 先关闭套接字唤醒阻塞中的 recv，再等待线程退出。
        self.nav_watchdog_timer.shutdown()
        self._disconnect_socket()
        if self.recv_thread and self.recv_thread.is_alive():
            rospy.loginfo("debug_driver_v2: 关闭接收线程")
            self.recv_thread.join(timeout=1)
        if self.send_thread and self.send_thread.is_alive():
            rospy.loginfo("debug_driver_v2: 关闭发送线程")
            self.send_thread.join(timeout=1)
        rospy.signal_shutdown("debug_driver_v2: 节点已关闭")
        self._disconnect_socket()

        if self.raw_saving_enable and self.raw_save_file:
            try:
                self.raw_save_file.close()
                rospy.loginfo("debug_driver_v2: 原始报文文件已关闭")
            except Exception as e:
                rospy.logerr(f"debug_driver_v2: 关闭原始报文文件失败: {e}")


if __name__ == "__main__":
    try:
        rospy.init_node('debug_driver_v2', anonymous=True)
        handler = DebugDriverV2()
        handler.run()
    except rospy.ROSInterruptException:
        pass
