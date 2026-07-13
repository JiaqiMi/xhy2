#! /home/xhy/xhy_env36/bin/python
"""
名称：debug_driver_v2.py
功能：调试驱动V2，支持定深(02)/定深定向(03)/定点(04)三种模式
      通过 TCP 发送 54 字节 ROV 扩展控制帧到 AUV
作者：BroXu
监听：/cmd/pose/lla (PoseLLAcmd.msg，经纬度坐标系)
发布：/status/auv (AUVData.msg)
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
"""

import json
import os
from datetime import datetime

import rospy
import socket
import struct
import threading
import time
from auv_control.msg import AUVData, PoseLLAcmd
from functools import reduce

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
        self.tx = 0      # X轴力 0-10000
        self.ty = 0      # Y轴力 0-10000
        self.tz = 0      # Z轴力 0-10000
        self.mx = 0      # 绕X轴力矩 0-10000
        self.my = 0      # 绕Y轴力矩 0-10000
        self.mz = 0      # 绕Z轴力矩 0-10000


class LowPassFilter:
    """一阶低通滤波器"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.last_value = None

    def update(self, value):
        if self.last_value is None:
            self.last_value = value
            return value
        filtered = self.alpha * value + (1 - self.alpha) * self.last_value
        self.last_value = filtered
        return filtered


class MovingAverageFilter:
    """移动平均滤波器"""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)


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

        # 原始报文保存
        self.raw_saving_enable = rospy.get_param("~save_raw_data", False)
        self.raw_save_dir = os.path.expanduser(rospy.get_param("~raw_save_dir", "~/.ros/auv_logs"))
        self.raw_save_file_name = rospy.get_param("~raw_save_file", "")
        self.raw_flush_every = max(1, int(rospy.get_param("~raw_flush_every", 1)))
        self.raw_write_count = 0
        self.raw_save_file = None

        self.server_address = (ip, port)
        self.tcp_sock = None
        self.buffer = bytearray()
        self.latest_debug_data = None

        self.lock = threading.Lock()
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
        self.data_pub = rospy.Publisher('/status/auv', AUVData, queue_size=10)
        rospy.loginfo("debug_driver_v2: 已启动, 监听 /cmd/pose/lla")

    def open_raw_save_file(self):
        """打开原始报文保存文件"""
        if not self.raw_save_file_name:
            self.raw_save_file_name = datetime.now().strftime("debug_v2_raw_%Y%m%d_%H%M%S.jsonl")
        os.makedirs(self.raw_save_dir, exist_ok=True)
        path = os.path.join(self.raw_save_dir, self.raw_save_file_name)
        self.raw_save_file = open(path, "a", encoding="utf-8")
        rospy.loginfo(f"debug_driver_v2: 原始报文保存到 {path}")

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
            self.raw_save_file.write(json.dumps(event, ensure_ascii=False) + "\n")
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
            data.sensor_status = f"{packet[10]:08b}"[::-1][:7]
            data.sensor_update = f"{packet[11]:08b}"[::-1][:5]
            data.fault_status = f"{struct.unpack('>h', packet[12:14])[0]:016b}"[::-1][:9]
            data.power_status = struct.unpack('>h', packet[14:16])[0]
            data.force_commands = list(struct.unpack('>6h', packet[16:28]))
            data.euler_angles = [x / 100.0 for x in struct.unpack('>3h', packet[28:34])]
            data.angular_velocity = [x / 100.0 for x in struct.unpack('>3h', packet[34:40])]
            data.linear_velocity = [x / 100.0 for x in struct.unpack('>3h', packet[40:46])]
            data.navigation_coords = [x / 10000000.0 for x in struct.unpack('<2i', packet[46:54])]
            raw_depth = struct.unpack('<f', packet[54:58])[0]
            data.depth = raw_depth
            data.depth_filtered = self.depth_lpf.update(raw_depth)
            data.depth_ma = self.depth_ma.update(raw_depth)
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
        except Exception as e:
            rospy.logerr(f"debug_driver_v2: 数据解析错误: {e}")
        return data

    def publish_auv_data(self, parsed):
        """将解析后的数据发布为 AUVData 消息"""
        msg = AUVData()
        msg.header.stamp = rospy.Time.now()
        msg.control_mode = parsed.mode
        msg.pose.latitude = parsed.navigation_coords[1]
        msg.pose.longitude = parsed.navigation_coords[0]
        msg.pose.depth = parsed.depth_filtered
        msg.pose.altitude = parsed.altitude
        msg.pose.roll = parsed.euler_angles[0]
        msg.pose.pitch = parsed.euler_angles[1]
        msg.pose.yaw = parsed.euler_angles[2]
        msg.pose.speed = parsed.linear_velocity[0]
        msg.motor_force.TX = parsed.force_commands[0]
        msg.motor_force.TY = parsed.force_commands[1]
        msg.motor_force.TZ = parsed.force_commands[2]
        msg.motor_force.MX = parsed.force_commands[3]
        msg.motor_force.MY = parsed.force_commands[4]
        msg.motor_force.MZ = parsed.force_commands[5]
        msg.linear_velocity = parsed.linear_velocity
        msg.angular_velocity = parsed.angular_velocity
        msg.sensor.temperature = parsed.temperature
        msg.sensor.voltage = parsed.control_voltage
        msg.sensor.current = parsed.power_current
        msg.sensor.battery = 0
        msg.sensor.leak_alarm = bool(parsed.water_leak)
        msg.sensor.sensor_valid = int(parsed.sensor_status, 2)
        msg.sensor.sensor_updated = int(parsed.sensor_update, 2)
        msg.sensor.fault_status = int(parsed.fault_status, 2)
        msg.sensor.power_status = int(parsed.power_status)
        msg.time.year = parsed.utc_time[0]
        msg.time.month = parsed.utc_time[1]
        msg.time.day = parsed.utc_time[2]
        msg.time.hour = parsed.utc_time[3]
        msg.time.minute = parsed.utc_time[4]
        msg.time.second = parsed.utc_time[5]
        self.data_pub.publish(msg)

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

        # 32-43: 6自由度力/力矩 int16×6 大端序，原始值 0-10000
        struct.pack_into('>6h', packet, 32,
            self.target.tx, self.target.ty, self.target.tz,
            self.target.mx, self.target.my, self.target.mz)

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

    def connect(self):
        """TCP 连接（阻塞重试直到成功）"""
        while not rospy.is_shutdown():
            try:
                self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_sock.connect(self.server_address)
                self.tcp_sock.settimeout(1)
                rospy.loginfo(f"debug_driver_v2: TCP连接成功 {self.server_address}")
                return
            except Exception as e:
                rospy.logwarn(f"debug_driver_v2: TCP连接失败 {self.server_address}: {e}, 2秒后重试...")
                rospy.sleep(2)

    # ── 收发线程 ─────────────────────────────────────────────

    def recv_loop(self):
        """接收循环（子线程）"""
        while not rospy.is_shutdown():
            try:
                data = self.tcp_sock.recv(512)
                if data:
                    self.buffer += data
                    while len(self.buffer) >= 110:
                        start = self.buffer.find(b'\xFE\xEF')
                        if start == -1 or len(self.buffer) - start < 110:
                            break
                        if self.buffer[start + 108:start + 110] == b'\xFA\xAF':
                            packet = self.buffer[start:start + 110]
                            self.buffer = self.buffer[start + 110:]
                            checksum_ok = self.calc_debug_checksum(packet) == packet[107]
                            self.save_raw_packet(packet, checksum_ok)
                            if checksum_ok:
                                parsed = self.parse_debug_packet(packet)
                                with self.lock:
                                    self.latest_debug_data = parsed
                                self.publish_auv_data(parsed)
                            else:
                                rospy.logwarn("debug_driver_v2: 校验和错误")
                        else:
                            self.buffer = self.buffer[start + 2:]
            except Exception as e:
                rospy.logwarn(f"debug_driver_v2: TCP连接错误: {e}, 重连中...")
                try:
                    if self.tcp_sock:
                        self.tcp_sock.close()
                except Exception:
                    pass
                self.connect()

    def send_loop(self):
        """发送循环（子线程），5Hz"""
        while not rospy.is_shutdown():
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
                try:
                    self.tcp_sock.sendall(packet)
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
            time.sleep(0.2)  # 5Hz

    # ── 回调 ─────────────────────────────────────────────

    def control_cmd_callback(self, msg):
        """接收 LLA 坐标系的完整控制指令。"""
        if msg.mode not in (MODE_DEPTH, MODE_DEPTH_HDG, MODE_DPROV):
            rospy.logwarn("debug_driver_v2: 忽略不支持的控制模式 %d", msg.mode)
            return

        with self.lock:
            self.target.mode = msg.mode
            self.target.longitude = msg.target.longitude
            self.target.latitude = msg.target.latitude
            self.target.depth = msg.target.depth
            self.target.roll = msg.target.roll
            self.target.pitch = msg.target.pitch
            self.target.yaw = msg.target.yaw
            self.target.speed = msg.target.speed
            self.target.tx = msg.force.TX
            self.target.ty = msg.force.TY
            self.target.tz = msg.force.TZ
            self.target.mx = msg.force.MX
            self.target.my = msg.force.MY
            self.target.mz = msg.force.MZ
            self.target.valid = True
            self.last_control_time = time.time()

    # ── 主循环 ─────────────────────────────────────────────

    def run(self):
        """主线程"""
        while not rospy.is_shutdown():
            try:
                if not self.tcp_sock:
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
                if self.tcp_sock:
                    try:
                        self.tcp_sock.close()
                    except:
                        pass
                    self.tcp_sock = None
                time.sleep(2)

        # 清理
        if self.recv_thread and self.recv_thread.is_alive():
            rospy.loginfo("debug_driver_v2: 关闭接收线程")
            self.recv_thread.join(timeout=1)
        if self.send_thread and self.send_thread.is_alive():
            rospy.loginfo("debug_driver_v2: 关闭发送线程")
            self.send_thread.join(timeout=1)
        rospy.signal_shutdown("debug_driver_v2: 节点已关闭")

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
