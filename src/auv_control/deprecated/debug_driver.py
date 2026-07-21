#! /home/xhy/xhy_env36/bin/python
"""
名称: debug_driver.py
功能: 连接调试串口发送54字节扩展报文
作者: buyegaid
监听: /auv_control (AUVPose.msg)
发布: /status/auv(AUVData.msg)
记录:
2025.7.15
    由于主控端需要一直发送因此, 这个改为持续5Hz发送, 当2s没有收到有效Control消息时, 停止发送
    在5s期间一直发送当前位置
2025.7.18 15:48
    精简一下发送的协议，只取有用的部分，发送给坐标转换节点来完成全局坐标转换的任务
2025.7.19 11:34
    统一log格式
2025.7.19 15:24
    控制指令改为直接接收AUVPose消息, 不再控制舵机和LED灯
    控制指令改为直接接收AUVPose消息, 不再控制舵机和LED灯
2025.7.21 11:51
    更正扩展指令报文中的错误, 更新频率是8Hz
2025.7.23 11:16
    增加数据保存功能
    对深度数据进行滤波处理
    在水下测试深度
2026.7.0 15:30
    删除csv保存，改为直接保存原始报文
2026.7.11
    统一 loginfo 中文输出：CMD/SEND 定长小数格式，保留 loginfo_throttle 节流
2026.7.13
    调整至 driver 目录，归入硬件驱动层
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
from auv_control.msg import AUVData, AUVPose
from functools import reduce
from debug_protocol import (
    DebugFrameBuffer,
    LowPassFilter,
    MovingAverageFilter,
    decode_status_words,
    require_finite,
)

class ControlTarget(object):
    """记录旧版定点控制目标。"""

    def __init__(self):
        self.valid = False
        self.longitude = 0.0
        self.latitude = 0.0
        self.depth = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.speed = 0.0


class DebugDataPacket:
    # 110字节调试协议解析结构体
    def __init__(self):
        self.mode = 0                       # 当前运行模式 02 定深，03 定向
        self.temperature = 0.0              # 舱内温度监测数据 不使用
        self.control_voltage = 0.0          # 总电压
        self.power_current = 0.0            # 总电流
        self.water_leak = 0                 # 漏水检测 00不漏 01漏
        self.sensor_status = 0              # 传感器状态 0 ahrs 1 gps 2 sbl 3 vio 4 dvl 地速 5 dvl流速 6 dvl高度
        self.sensor_update = 0              # 传感器更新 0 ahrs 1 dvl 2 gps 3 sbl 4 vio
        self.fault_status = 0               # 故障状态
        self.power_status = 0               # 电源状态 ROV默认全开启
        self.force_commands = [0] * 6        # 当前的力和力矩
        self.euler_angles = [0.0] * 3        # 欧拉角
        self.angular_velocity = [0.0] * 3    # 角速度
        self.linear_velocity = [0.0] * 3     # 线速度
        self.navigation_coords = [0.0] * 2   # 导航坐标
        self.depth = 0.0                     # 原始深度
        self.depth_filtered = 0.0            # 滤波后的深度
        self.depth_ma = 0.0                  # 移动平均后的深度
        self.altitude = 0.0                  # 高度
        self.target_longitude = 0.0     # 目标经度
        self.target_latitude = 0.0      # 目标纬度
        self.target_depth = 0.0         # 目标深度
        self.target_roll = 0.0          # 目标横滚角
        self.target_pitch = 0.0         # 目标俯仰角
        self.target_yaw = 0.0           # 目标偏航角
        self.target_altitude = 0.0      # 目标高度
        self.target_speed = 0.0         # 目标速度
        self.utc_time = [0] * 6         # UTC时间
        self.checksum = 0               # 校验和


class DebugDriver(object):
    """旧版调试口驱动。"""

    def __init__(self, ip=None, port=None):
        # 获取参数服务器的IP和端口，默认192.168.1.115:5063
        ip = ip or rospy.get_param("~debug_ip", "192.168.1.115")
        port = port or rospy.get_param("~debug_port", 5063)

        self.raw_saving_enable = rospy.get_param("~save_raw_data", False)
        self.raw_save_dir = os.path.expanduser(rospy.get_param("~raw_save_dir", "~/.ros/auv_logs"))
        self.raw_save_file_name = rospy.get_param("~raw_save_file", "")
        self.raw_flush_every = max(1, int(rospy.get_param("~raw_flush_every", 1)))
        self.raw_write_count = 0
        self.raw_save_file = None
        self.server_address = (ip, port)
        self.tcp_sock = None
        self.latest_debug_data = None

        self.lock = threading.Lock()
        self.socket_lock = threading.RLock()
        self.connect_lock = threading.Lock()
        self.target = ControlTarget()
        self.last_control_time = 0
        self.send_thread = None
        self.recv_thread = None

        # 初始化深度滤波器
        self.depth_lpf = LowPassFilter(alpha=0.2)  # 低通滤波器
        self.depth_ma = MovingAverageFilter(window_size=5)  # 移动平均滤波器
        if self.raw_saving_enable:
            self.open_raw_save_file()

        rospy.Subscriber('/auv_control', AUVPose, self.control_callback)
        self.data_pub = rospy.Publisher('/status/auv', AUVData, queue_size=10)
        self.rate = rospy.Rate(10)
        rospy.loginfo("debug_driver: 已启动")

    def open_raw_save_file(self):
        if not self.raw_save_file_name:
            self.raw_save_file_name = datetime.now().strftime("debug_raw_%Y%m%d_%H%M%S.jsonl")

        os.makedirs(self.raw_save_dir, exist_ok=True)
        path = os.path.join(self.raw_save_dir, self.raw_save_file_name)
        self.raw_save_file = open(path, "a", encoding="utf-8")
        rospy.loginfo(f"debug_driver: 原始报文将保存到 {path}")

    def save_raw_packet(self, packet, checksum_ok):
        if not self.raw_saving_enable or self.raw_save_file is None:
            return

        event = {
            "pc_time": rospy.Time.now().to_sec(),
            "source": "debug",
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
            rospy.logerr(f"debug_driver: 保存原始报文失败: {e}")
        
    def calc_debug_checksum(self, packet):
        # 计算调试协议的校验和
        return reduce(lambda x, y: x ^ y, packet[:107], 0)

    def parse_debug_packet(self, packet):
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
            # 深度滤波处理
            raw_depth = struct.unpack('<f', packet[54:58])[0]
            data.depth = raw_depth  # 保存原始深度
            data.altitude = struct.unpack('<f', packet[58:62])[0]
            # data.collision_avoidance = [x / 100.0 for x in struct.unpack('>2h', packet[62:66])]
            data.target_longitude = struct.unpack('<i', packet[66:70])[0] / 10000000.0
            data.target_latitude = struct.unpack('<i', packet[70:74])[0] / 10000000.0
            data.target_depth = struct.unpack('<f', packet[74:78])[0]
            data.target_roll = struct.unpack('>h', packet[78:80])[0] / 100.0
            data.target_pitch = struct.unpack('>h', packet[80:82])[0] / 100.0
            data.target_yaw = struct.unpack('>h', packet[82:84])[0] / 100.0
            data.target_altitude = struct.unpack('<f', packet[84:88])[0]
            data.target_speed = struct.unpack('>H', packet[88:90])[0] / 100.0
            # utc_time: 90-94为年/月/日/时/分，95-98为float秒
            data.utc_time = list(packet[90:95])  # 5字节
            data.utc_time.append(struct.unpack('<f', packet[95:99])[0])  # 秒为float
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
            # 仅在整包有效时推进滤波状态。
            data.depth_filtered = self.depth_lpf.update(raw_depth)
            data.depth_ma = self.depth_ma.update(raw_depth)
        except Exception as e:
            rospy.logerr(f"debug_driver: 数据解析错误: {e}")
            return None
        return data

    def build_54_packet_from_control(self):
        # 参考main_driver.py的build_expect_packet，构造54字节ROV扩展指令
        packet = bytearray(54)
        # 0-1: 报文头 FE FE
        packet[0:2] = b'\xFE\xFE'
        # 2-3: AUV编号 00 01
        packet[2:4] = b'\x00\x01'
        # 4: 指令类型 0x30 ROV扩展指令
        packet[4] = 0x30
        # 5: 设备运行模式，04动力定位ROV
        packet[5] = 0x04
        # 6: 开环闭环与扩展模式，01闭环模式
        packet[6] = 0x01
        # 7: 坐标系设置，00经纬度
        packet[7] = 0x00
        # 8-11: 期望经度，扩大1e7
        lon = int(self.target.longitude * 1e7)
        packet[8:12] = struct.pack('<i', lon)
        # 12-15: 期望纬度，扩大1e7
        lat = int(self.target.latitude* 1e7)
        packet[12:16] = struct.pack('<i', lat)
        # 16-19: 期望深度 float32
        packet[16:20] = struct.pack('<f', self.target.depth)
        # 20-23: 期望横滚角 float32
        packet[20:24] = struct.pack('<f', self.target.roll)
        # 24-27: 期望俯仰角 float32
        packet[24:28] = struct.pack('<f', self.target.pitch)
        # 28-31: 期望偏航角 float32
        packet[28:32] = struct.pack('<f', self.target.yaw)
        # 32-43: 6自由度力/力矩，全部填0
        for i in range(32, 44):
            packet[i] = 0x00
        # 43-44: 补光灯控制
        packet[43] = 0 # 补光灯1亮度 (0-100)
        packet[44] = 0  # 补光灯2亮度 (0-100)
        # 46-50: 预留
        for i in range(46, 51):
            packet[i] = 0x00
        # 51: 校验和
        xor = 0
        for i in range(0, 51):
            xor ^= packet[i]
        packet[51] = xor
        # 52-53: 数据尾 FD FD
        packet[52:54] = b'\xFD\xFD'
        return packet

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
                        f"debug_driver: TCP连接成功 {self.server_address}")
                    return
                except Exception as e:
                    try:
                        sock.close()
                    except OSError:
                        pass
                    rospy.logwarn(
                        f"debug_driver: TCP连接失败 {self.server_address}: "
                        f"{e}, 2秒后重试...")
                    rospy.sleep(2)

    def recv_loop(self):
        # 接收循环，子线程
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
                        rospy.logwarn("debug_driver: 校验和错误")
                        continue
                    parsed = self.parse_debug_packet(packet)
                    if parsed is None:
                        continue
                    with self.lock:
                        self.latest_debug_data = parsed
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
                    msg.sensor.sensor_valid = parsed.sensor_status
                    msg.sensor.sensor_updated = parsed.sensor_update
                    msg.sensor.fault_status = parsed.fault_status
                    msg.sensor.power_status = parsed.power_status
                    msg.time.year = parsed.utc_time[0]
                    msg.time.month = parsed.utc_time[1]
                    msg.time.day = parsed.utc_time[2]
                    msg.time.hour = parsed.utc_time[3]
                    msg.time.minute = parsed.utc_time[4]
                    msg.time.second = parsed.utc_time[5]
                    self.data_pub.publish(msg)
            except socket.timeout:
                continue
            except Exception as e:
                rospy.logwarn(f"debug_driver: TCP连接错误: {e}, 重连中...")
                self._disconnect_socket(sock)
                active_socket = None
                frame_buffer = DebugFrameBuffer()
                self.connect()

    def control_callback(self, msg: AUVPose):
        """原子更新旧版定点控制目标。"""
        try:
            values = require_finite((
                getattr(msg, 'longitude', 0.0),
                getattr(msg, 'latitude', 0.0),
                getattr(msg, 'depth', 0.0),
                getattr(msg, 'roll', 0.0),
                getattr(msg, 'pitch', 0.0),
                getattr(msg, 'yaw', 0.0),
                getattr(msg, 'speed', 0.0),
            ), '控制目标')
            with self.lock:
                (
                    self.target.longitude,
                    self.target.latitude,
                    self.target.depth,
                    self.target.roll,
                    self.target.pitch,
                    self.target.yaw,
                    self.target.speed,
                ) = values
                self.target.valid = True
                self.last_control_time = time.time()
            rospy.loginfo_throttle(5,
                "debug_driver: 接收到控制消息 lon=%12.7f lat=%12.7f depth=%7.2f "
                "roll=%6.1f pitch=%6.1f yaw=%6.1f speed=%5.2f",
                *values
            )
        except Exception as e:
            rospy.logerr(f"debug_driver: 控制消息接收错误: {e}")

    def send_loop(self):
        # 发送循环，子线程
        while not rospy.is_shutdown():
            now = time.time()
            packet = None
            target_snapshot = None
            timed_out = False
            with self.lock:
                if self.target.valid and (now - self.last_control_time > 5):
                    self.target.valid = False
                    timed_out = True
                if self.target.valid:
                    packet = self.build_54_packet_from_control()
                    target_snapshot = (
                        self.target.longitude,
                        self.target.latitude,
                        self.target.depth,
                        self.target.roll,
                        self.target.pitch,
                        self.target.yaw,
                        self.target.speed,
                    )
            if timed_out:
                rospy.loginfo("debug_driver: 5s未收到控制消息，停止发送！")

            if packet is not None:
                sock = None
                try:
                    with self.socket_lock:
                        sock = self.tcp_sock
                    if sock is None:
                        time.sleep(0.2)
                        continue
                    sock.sendall(packet)
                    rospy.loginfo_throttle(2,
                        "debug_driver: 发送扩展控制指令 lon=%12.7f lat=%12.7f depth=%7.2f "
                        "roll=%6.1f pitch=%6.1f yaw=%6.1f speed=%5.2f",
                        *target_snapshot
                    )
                except Exception as e:
                    rospy.logerr(f"debug_driver: 发送扩展指令包错误: {e}")
                    self._disconnect_socket(sock)
            time.sleep(0.2)  # 5Hz

    def run(self):
        # 主线程，主循环
        while not rospy.is_shutdown():
            try:
                with self.socket_lock:
                    connected = self.tcp_sock is not None
                if not connected:
                    self.connect()
                    time.sleep(2)
                    continue
                
                if not self.recv_thread or not self.recv_thread.is_alive():
                    rospy.loginfo("debug_driver: 启动接收线程")
                    self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
                    self.recv_thread.start()
                
                if not self.send_thread or not self.send_thread.is_alive():
                    rospy.loginfo("debug_driver: 启动发送线程")
                    self.send_thread = threading.Thread(target=self.send_loop, daemon=True)
                    self.send_thread.start()
                
                time.sleep(0.01)  # 100Hz，足够了
                
            except Exception as e:
                rospy.logerr(f"debug_driver: 运行错误: {e}")
                self._disconnect_socket()
                time.sleep(2)

        # 先关闭套接字唤醒阻塞中的 recv，再等待线程退出。
        self._disconnect_socket()
        if self.recv_thread and self.recv_thread.is_alive():
            rospy.loginfo("debug_driver: 正在关闭接收线程")
            self.recv_thread.join(timeout=1)
        if self.send_thread and self.send_thread.is_alive():
            rospy.loginfo("debug_driver: 正在关闭发送线程")
            self.send_thread.join(timeout=1)
        rospy.signal_shutdown("debug_driver: 节点已关闭")
        self._disconnect_socket()

        if self.raw_saving_enable and self.raw_save_file:
            try:
                self.raw_save_file.close()
                rospy.loginfo("debug_driver: 原始报文文件已保存并关闭")
            except Exception as e:
                rospy.logerr(f"debug_driver: 关闭原始报文文件失败: {e}")

if __name__ == "__main__":
    try:
        rospy.init_node('debug_driver', anonymous=True)
        handler = DebugDriver()
        handler.run()
    except rospy.ROSInterruptException:
        pass
