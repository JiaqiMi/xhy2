"""
名称：AUV协议发送器
功能：用于与AUV进行通信的协议发送器
作者：buyegaid

记录：
2025.7.17 18:25
    1. 根据新的通信模式更新和简化
2025.7.22 12:05
    增加数据有效值判断
2025.7.23 1:56
    修复死循环BUG, 测试完成
"""
import serial
import time
import tkinter as tk
from tkinter import ttk, messagebox, PhotoImage
import threading
import socket
from functools import reduce
import struct
import math

import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import Window  #修改主函数中控制界面
from PIL import Image, ImageTk
from ttkbootstrap import Style

# 字节 内容
# 0~1 报文头 FEFE
# 2~3 AUV编号 01
# 4 指令类型  04 空间运动控制 
# 5 运动控制模式 00 自控 01 键鼠控制 02 遥控
# 6 控制类型
# 7~32 不使用 00
# 33~38 空间运动控制，6字节对应6个自由度，以64为中值
# 39~40 设备启动
# 41~50 前4个字节控制LED灯，第6个字节控制机械手
# 51 异或校验
# 52~53 数据尾  FDFD
# 发送频率5Hz


# 设备启动2字节
# 两个字节总计16位，用二进制表示如0000 0000 0000 0000。
# （1）若控制指令类型为设备启动，则对应的控制16个用电设备电源开关，0为关闭、1为开启，
# 从最低位（最右边）开始。分别对应惯导、DVL、USBL、声通信机、高度计、深度计、侧扫声呐、
# 前视声呐、单波束、多波束、避碰声呐、频闪灯、ADCP、CTD、浅剖、PC机等。例如01 0F表示USBL、声通信机、惯导、DVL、
# 前视声呐这5个设备的电源为开启状态。


# 指令类型字典
COMMAND_TYPE = {
    "控制模式": 0x00,
    "设备启动": 0x05
}

# 运动控制模式
CONTROL_TYPE = {
    "自控": 0x00,
    "键鼠控制": 0x01,
    "遥控": 0x02
}

# 控制类型
CONTROL_MODE = {
    "闭环": 0x00,
    "开环": 0x01,
    "扩展模式": 0x02,
    "锚定": 0x03
}

# 设备启动3项
DEVICE_LIST = [
    "惯导", "DVL", "PC机"
]

# 在SenderUI类之前添加传感器数据结构体
class SensorData:
    """
    17字节传感器数据结构体
    """
    def __init__(self):
        self.control_voltage = 0.0  # 控制电压
        self.power_voltage = 0.0    # 动力电压
        self.control_current = 0.0  # 控制电流
        self.power_current = 0.0    # 动力电流
        self.servo_angle = 0        # 舵机角度

class Auv: # 用于存储auv的状态信息
    def __init__(self):
        self.yaw = 0.0             # 航向角，单位：角度
        self.pitch = 0.0           # 俯仰角，单位：角度
        self.roll = 0.0            # 横滚角，单位：角度
        self.speed = 0.0           # 速度，单位：m/s
        self.latitude = 0.0        # 纬度，单位：度
        self.longitude = 0.0       # 经度，单位：度
        self.depth = 0.0           # 深度，单位：m
        self.altitude = 0.0        # 高度，单位：m
        
def geodetic_distance(lat1, lon1, depth1, lat2, lon2, depth2):
      """
      计算两组地理坐标（经度、纬度、深度）之间的空间距离
      :param lat1: 第一个点的纬度（度）
      :param lon1: 第一个点的经度（度）
      :param depth1: 第一个点的深度（米）
      :param lat2: 第二个点的纬度（度）
      :param lon2: 第二个点的经度（度）
      :param depth2: 第二个点的深度（米）
      :return: 距离（米）
      """
      Rearth = 6378137.0  # 地球半径（米）
      # 经纬度转弧度
      lat1_rad = math.radians(lat1)
      lon1_rad = math.radians(lon1)
      lat2_rad = math.radians(lat2)
      lon2_rad = math.radians(lon2)
      # 水平球面距离（haversine公式）
      dlat = lat2_rad - lat1_rad
      dlon = lon2_rad - lon1_rad
      a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
      c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
      horizontal_dist = Rearth * c
      # 垂直距离
      ddepth = depth2 - depth1
      # 空间距离
      distance = math.sqrt(horizontal_dist**2 + ddepth**2)
      return distance

def local_point_to_global(auv_pose: Auv, local_point: tuple):
    """
    简化版：已知AUV当前经纬度和深度，local_point为(北, 东, 地)方向的增量（单位：米），
    直接计算目标点的经纬度和深度。
    :param auv_pose: Auv对象，包含latitude, longitude, depth（单位：度、度、米）
    :param local_point: (dn, de, dd)，分别为北、东、地方向的增量（单位：米）
    :return: (latitude, longitude, depth)
    """
    Rearth = 6378137.0  # 地球半径（米）
    lat0 = math.radians(auv_pose.latitude)
    lon0 = math.radians(auv_pose.longitude)
    depth0 = auv_pose.depth
    yaw = math.radians(auv_pose.yaw)  # 航向角转弧度

    forward, right, down = local_point  # 北、东、地

    # 将AUV视角下的前右坐标旋转到地理坐标系下的北东坐标
    # 使用旋转矩阵，航向角为从北向顺时针到AUV前向的角度
    dn = forward * math.cos(yaw) - right * math.sin(yaw)  # 北向分量
    de = forward * math.sin(yaw) + right * math.cos(yaw)  # 东向分量
    dd = down  # 深度增量保持不变
     # 计算经纬度增量
    dlat = dn / Rearth  # 纬度增量(弧度)
    dlon = de / (Rearth * math.cos(lat0))  # 经度增量(弧度)

    lat = lat0 + dlat
    lon = lon0 + dlon
    depth = depth0 + dd

    return math.degrees(lat), math.degrees(lon), depth

def build_packet(auv_id, command_type, control_mode, control_type, motion_values, device_start, led_values, gimbal_value):
    """
    构造AUV控制协议数据包
    """
    packet = bytearray(54)
    # Header
    packet[0:2] = b'\xFE\xFE'
    # AUV编号
    packet[2] = 0x00
    packet[3] = auv_id
    # 指令类型
    packet[4] = command_type
    # 运动控制模式
    packet[5] = control_mode
    # 控制类型
    packet[6] = control_type
    # 7~32 不使用
    for i in range(7, 33):
        packet[i] = 0x00
    packet[32] = 0xff
    # 如果是设备启动指令，将28~32字节设置为0xfc
    if command_type == COMMAND_TYPE["设备启动"]:
        for i in range(28, 33):
            packet[i] = 0xfc

    # 33~38 空间运动控制，6字节
    for i in range(6):
        packet[33 + i] = motion_values[i]
    # 39~40 设备启动
    packet[39] = device_start[1]  # 大端在前
    packet[40] = device_start[0]
    # 41~44 LED灯
    for i in range(2):
        packet[41 + i] = led_values[i]
    # 45~49 保留
    for i in range(45, 49):
        packet[i] = 0x00
    # 50 云台
    packet[46] = gimbal_value
    # 51 异或校验
    xor = 0
    for i in range(0, 51):
        xor ^= packet[i]
    packet[51] = xor
    # 52~53 数据尾
    packet[52:54] = b'\xFD\xFD'
    return packet

class DataPacket:
    """
    用于存储AUV回传数据的结构体（110字节调试协议）
    """
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
        self.force_commands = [0] * 12
        self.euler_angles = [0.0] * 3
        self.angular_velocity = [0.0] * 3
        self.linear_velocity = [0.0] * 3
        self.navigation_coords = [0.0] * 2
        self.depth = 0.0
        self.altitude = 0.0
        self.collision_avoidance = [0.0] * 2
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

class AUVSerialHandler:
    """
    串口数据接收与解析处理类（110字节调试协议）
    """
    def __init__(self, ser):
        """
        初始化，传入串口对象
        """
        self.ser = ser
        self.buffer = bytearray()

    def calculate_checksum(self, packet):
        """
        计算数据包的异或校验和
        """
        return reduce(lambda x, y: x ^ y, packet[:107], 0)

    def parse_packet(self, packet):
        """
        解析110字节数据包为DataPacket对象
        """
        data = DataPacket()
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
            data.force_commands = struct.unpack('>6h', packet[16:28])
            data.euler_angles = [x / 100.0 for x in struct.unpack('>3h', packet[28:34])]
            data.angular_velocity = [x / 100.0 for x in struct.unpack('>3h', packet[34:40])]
            data.linear_velocity = [x / 100.0 for x in struct.unpack('>3h', packet[40:46])]
            data.navigation_coords = [x / 10000000.0 for x in struct.unpack('<2i', packet[46:54])]
            data.depth = struct.unpack('<f', packet[54:58])[0]
            data.altitude = struct.unpack('<f', packet[58:62])[0]
            data.collision_avoidance = [x / 100.0 for x in struct.unpack('>2h', packet[62:66])]
            data.target_longitude = struct.unpack('<i', packet[66:70])[0] / 10000000.0
            data.target_latitude = struct.unpack('<i', packet[70:74])[0] / 10000000.0
            data.target_depth = struct.unpack('<f', packet[74:78])[0]
            data.target_roll = struct.unpack('>h', packet[78:80])[0] / 100.0
            data.target_pitch = struct.unpack('>h', packet[80:82])[0] / 100.0
            data.target_yaw = struct.unpack('>h', packet[82:84])[0] / 100.0
            data.target_altitude = struct.unpack('<f', packet[84:88])[0]
            data.target_speed = struct.unpack('>H', packet[88:90])[0] / 100.0
            data.utc_time = list(packet[90:95]) + [struct.unpack('<f', packet[95:99])[0]] # 90`94`
            data.checksum = packet[107]
        except Exception as e:
            print(f"Error parsing packet: {e}")
        return data

    def read_and_parse(self):
        """
        从串口读取数据并尝试解析完整数据包
        """
        while self.ser.in_waiting > 0:
            self.buffer += self.ser.read(self.ser.in_waiting)
            while len(self.buffer) >= 110:
                if self.buffer[0:2] == b'\xFE\xEF' and self.buffer[108:110] == b'\xFA\xAF':
                    packet = self.buffer[:110]
                    self.buffer = self.buffer[110:]
                    if self.calculate_checksum(packet) == packet[107]:
                        return self.parse_packet(packet)
                    else:
                        print("Checksum mismatch")
                else:
                    self.buffer.pop(0)
        return None

class User75Packet:
    """
    75字节用户协议数据结构体
    """
    def __init__(self):
        self.header = ""
        self.mode = 0
        self.ship_id = 0
        self.packet_id = 0
        self.motion_mode = 0
        self.task_type = 0
        self.auv_speed = 0.0
        self.auv_depth = 0.0
        self.auv_altitude = 0.0
        self.cabin_temp = 0.0
        self.attitude = [0.0, 0.0, 0.0]  # roll, pitch, yaw
        self.voltage = 0.0
        self.current = 0.0
        self.battery = 0
        self.longitude = 0.0
        self.latitude = 0.0
        self.lon_sign = 0
        self.lat_sign = 0
        self.send_time = ""
        self.sensor_valid = 0
        self.sensor_update = 0
        self.power_state = 0
        self.device_mode = 0

def parse_75_packet(packet: bytes):
    """
    解析75字节用户协议数据
    """
    if len(packet) != 75:
        return None
    p = User75Packet()
    try:
        p.header = packet[0:2].hex().upper()
        p.mode = packet[2]
        p.ship_id = packet[3]
        p.packet_id = int.from_bytes(packet[4:7], 'big')
        p.motion_mode = packet[7]
        p.task_type = packet[8]
        p.auv_speed = int.from_bytes(packet[9:11], 'big', signed=True) / 100.0
        p.auv_depth = int.from_bytes(packet[11:13], 'big', signed=True) / 100.0
        p.auv_altitude = int.from_bytes(packet[13:15], 'big', signed=True) / 100.0
        p.cabin_temp = int.from_bytes(packet[21:23], 'big', signed=True) / 100.0
        # 姿态传感器数据（roll, pitch, yaw）
        att = []
        for i in range(3):
            sign = 1 if packet[26 + i*3] == 0 else -1
            val = int.from_bytes(packet[27 + i*3:29 + i*3], 'big', signed=False) / 100.0
            att.append(sign * val)
        p.attitude = att
        p.voltage = int.from_bytes(packet[35:37], 'big', signed=True) / 100.0
        p.current = int.from_bytes(packet[37:39], 'big', signed=True) / 100.0
        p.battery = packet[39]
        p.longitude = int.from_bytes(packet[43:47], 'big', signed=True) / 1e7
        p.latitude = int.from_bytes(packet[47:51], 'big', signed=True) / 1e7
        sign_byte = packet[51]
        p.lon_sign = (sign_byte & 0x01)
        p.lat_sign = (sign_byte >> 4) & 0x01
        # 时间
        y = (packet[52]*256 *256 + packet[53] * 256 + packet[54])//10000
        m = (packet[52]*256 *256 + packet[53] * 256 + packet[54]) % 10000 // 100
        d = (packet[52]*256 *256 + packet[53] * 256 + packet[54]) % 100
        h = (packet[55]*256 *256 + packet[56] * 256 + packet[57])//10000
        mi = (packet[55]*256 *256 + packet[56] * 256 + packet[57]) % 10000 // 100
        s = (packet[55]*256 *256 + packet[56] * 256 + packet[57]) % 100
        p.send_time = f"20{y:02d}-{m:02d}-{d:02d} {h:02d}:{mi:02d}:{s:02d}"
        p.sensor_valid = packet[62]
        p.sensor_update = packet[63]
        p.power_state = int.from_bytes(packet[64:66], 'big')
        p.device_mode = packet[71]
    except Exception as e:
        print(f"75协议解析异常: {e}")
        return None
    return p

class Auv:
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.depth = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

auv_state = Auv()

class SenderUI:
    """
    主UI类，集成发送与接收功能
    """
    def __init__(self, master):
        """
        初始化UI及各参数变量
        """
        self.master = master
        master.title("AUV 协议发送器")

        # ===== UI: 背景绘制函数 =====
        self.createWidget()
        self.master.bind("<Configure>",self.on_resize)
        style = Style()
        style.configure("Transparent.TFrame", background="#CCE6FF")  # 蓝色背景

        # ==== 添加主内容框架结构 ====
        # self.sidebar_frame = ttk.Frame(self.main_frame, width=120, style="Transparent.TFrame")
        # self.sidebar_frame.grid(row=0, column=0, sticky="ns")

        # === 分割左侧边栏 ====
        self.sidebar_frame = ttk.Frame(self.main_frame, width=150, padding=10, relief="ridge")
        self.sidebar_frame.grid(row=0, column=0, sticky="nswe")

        self.sidebar_frame.rowconfigure(0, weight=2)  # 上栏占30%
        self.sidebar_frame.rowconfigure(1, weight=0)  # 分割线高度固定
        self.sidebar_frame.rowconfigure(2, weight=8)  # 下栏占70%

        # == 上方左侧栏 ==
        self.sidebar_top = ttk.Frame(self.sidebar_frame, style="White.TFrame")
        self.sidebar_top.grid(row=0, column=0, sticky="new", pady=(0, 5))

        #== 分割线 ==
        self.divider = ttk.Separator(self.sidebar_frame, orient="horizontal")
        self.divider.grid(row=1, column=0, sticky="ew", pady=5)

        # == 下方左侧栏 ==
        self.sidebar_bottom = ttk.Frame(self.sidebar_frame, style="White.TFrame")
        self.sidebar_bottom.grid(row=1, column=0, sticky="swe", pady=(5, 0))

        # self.content_frame = ttk.Frame(self.main_frame, style="Transparent.TFrame")
        # self.content_frame.grid(row=0, column=1, sticky="nsew")

        # ==== 右侧主内容区示例 =====
        main_content = ttk.Frame(self.main_frame, relief="sunken")
        main_content.grid(row=0, column=1, sticky="nswe")
        main_content.columnconfigure(0, weight=1)
        main_content.rowconfigure(0, weight=1)

        # 参数变量
        self.auv_id = tk.IntVar(value=1)
        self.command_type = tk.StringVar(value="控制模式")
        self.control_mode = tk.StringVar(value="自控")
        self.control_type = tk.StringVar(value="开环")
        self.motion_values = [tk.IntVar(value=100) for _ in range(6)]
        # 设备启动相关
        self.device_start0 = tk.IntVar(value=0)
        self.device_start1 = tk.IntVar(value=0)
        self.device_start_bits = [tk.IntVar(value=0) for _ in range(3)]  # 新增：3个设备开关
        self.led_values = [tk.IntVar(value=0) for _ in range(2)]
        self.gimbal_value = tk.IntVar(value=0)
        self.packet_hex = tk.StringVar(value="")
        self.com_port = tk.StringVar(value="COM31")
        self.baudrate = tk.IntVar(value=230400)
        self.is_sending = False
        self.ser = None
        self.send_thread = None

        # 新增：连接模式相关变量
        self.conn_mode = tk.StringVar(value="TCP")
        self.tcp_ip = tk.StringVar(value="47.104.21.115")
        self.tcp_port = tk.IntVar(value=5322)
        self.sock = None

        # 新增：调试协议TCP端口相关变量
        self.debug_tcp_ip = tk.StringVar(value="47.104.21.115")
        self.debug_tcp_port = tk.IntVar(value=5323)
        self.debug_sock = None
        self.debug_recv_thread = None
        self.is_debug_receiving = False

        # ========== 点控制相关变量 ==========
        self.point_x = tk.DoubleVar(value=0.0)
        self.point_y = tk.DoubleVar(value=0.0)
        self.point_z = tk.DoubleVar(value=0.0)
        self.point_packet_hex = tk.StringVar(value="")
        self.is_point_sending = False
        self.point_send_thread = None

        # 新增：传感器端口TCP相关变量
        self.sensor_tcp_ip = tk.StringVar(value="47.104.21.115")
        self.sensor_tcp_port = tk.IntVar(value=5324)
        self.sensor_sock = None

        # ========== 传感器报文相关变量 ==========
        self.sensor_packet_hex = tk.StringVar(value="")
        self.is_sensor_sending = False
        self.sensor_send_thread = None
        # 补光灯控制变量
        self.light1_value = tk.IntVar(value=0)  # 补光灯1亮度
        self.light2_value = tk.IntVar(value=0)  # 补光灯2亮度

        # UI布局
        row = 0
        # 连接参数区
        ttk.Label(self.sidebar_top, text="连接模式:").grid(row=row, column=0, sticky="e",padx=3)
        ttk.Combobox(self.sidebar_top, textvariable=self.conn_mode, values=["TCP","串口"], width=7, state="readonly").grid(row=row, column=1, sticky="w",padx=3)
        ttk.Label(self.sidebar_top, text="串口:").grid(row=row, column=2, sticky="e",padx=3)
        ttk.Entry(self.sidebar_top, textvariable=self.com_port, width=8).grid(row=row, column=3, sticky="w",padx=3)
        ttk.Label(self.sidebar_top, text="波特率:").grid(row=row, column=4, sticky="e",padx=3)
        ttk.Entry(self.sidebar_top, textvariable=self.baudrate, width=8).grid(row=row, column=5, sticky="w",padx=3)
        row += 1
        ttk.Label(self.sidebar_top, text="用户IP:").grid(row=row, column=0, sticky="e",pady=3,padx=2)
        ttk.Entry(self.sidebar_top, textvariable=self.tcp_ip, width=14).grid(row=row, column=1, sticky="w",pady=3,padx=2)
        ttk.Label(self.sidebar_top, text="端口:").grid(row=row, column=2, sticky="e",padx=3)
        ttk.Entry(self.sidebar_top, textvariable=self.tcp_port, width=8).grid(row=row, column=3, sticky="w",pady=3,padx=2)
        ttk.Button(self.sidebar_top, text="连接", command=self.toggle_user_connect, width=10).grid(row=row, column=4,pady=3,padx=2)
        ttk.Button(self.sidebar_top, text="断开", command=self.toggle_user_disconnect, width=10).grid(row=row, column=5,pady=3,padx=2)
        row += 1

        # 调试协议TCP参数
        ttk.Label(self.sidebar_top, text="调试IP:").grid(row=row, column=0, sticky="e",pady=3,padx=2)
        ttk.Entry(self.sidebar_top, textvariable=self.debug_tcp_ip, width=14).grid(row=row, column=1, sticky="w",pady=3,padx=2)
        ttk.Label(self.sidebar_top, text="调试端口:").grid(row=row, column=2, sticky="e",pady=3,padx=2)
        ttk.Entry(self.sidebar_top, textvariable=self.debug_tcp_port, width=8).grid(row=row, column=3, sticky="w",pady=3,padx=2)
        ttk.Button(self.sidebar_top, text="连接", command=self.toggle_debug_connect, width=10).grid(row=row, column=4,pady=3,padx=2)
        ttk.Button(self.sidebar_top, text="断开", command=self.toggle_debug_disconnect, width=10).grid(row=row, column=5,pady=3,padx=2)
        row += 1

        # 新增：传感器端口TCP参数
        ttk.Label(self.sidebar_top, text="传感器IP:").grid(row=row, column=0, sticky="e",pady=3,padx=2)
        ttk.Entry(self.sidebar_top, textvariable=self.sensor_tcp_ip, width=14).grid(row=row, column=1, sticky="w",pady=3,padx=2)
        ttk.Label(self.sidebar_top, text="传感器端口:").grid(row=row, column=2, sticky="e",pady=3,padx=2)
        ttk.Entry(self.sidebar_top, textvariable=self.sensor_tcp_port, width=8).grid(row=row, column=3, sticky="w",pady=3,padx=2)
        ttk.Button(self.sidebar_top, text="连接", command=self.toggle_sensor_connect, width=10).grid(row=row, column=4,pady=3,padx=2)
        ttk.Button(self.sidebar_top, text="断开", command=self.toggle_sensor_disconnect, width=10).grid(row=row, column=5,pady=3,padx=2)
        row += 1

        #=== 显示报文 ===
        ttk.Label(self.sidebar_bottom, text="用户报文HEX:").grid(row=row, column=0, sticky="w")
        ttk.Entry(self.sidebar_bottom, textvariable=self.packet_hex, width=60).grid(row=row, column=1, columnspan=6,sticky="w")
        row += 1

        # ==== 整合：发送报文 ======

        ttk.Button(self.sidebar_bottom, text="生成用户报文", command=self.generate_user_packet, width=12).grid(row=row,column=0,pady=2)
        ttk.Button(self.sidebar_bottom, text="发送用户报文", command=self.send_user_once, width=12).grid(row=row,column=1,pady=2)
        self.send_btn = ttk.Button(self.sidebar_bottom, text="循环发送用户报文", command=self.toggle_user_send_loop,width=16)
        self.send_btn.grid(row=row, column=2,pady=2)
        row += 1

        ttk.Button(self.sidebar_bottom, text="生成扩展报文", command=self.generate_debug_packet, width=12).grid(row=row,column=0,pady=2)
        ttk.Button(self.sidebar_bottom, text="发送扩展报文", command=self.send_debug_once, width=12).grid(row=row,column=1,pady=2)
        self.point_send_btn = ttk.Button(self.sidebar_bottom, text="循环发送扩展报文",command=self.toggle_debug_send_loop, width=16)
        self.point_send_btn.grid(row=row, column=2,pady=2)

        row += 1

        ttk.Button(self.sidebar_bottom, text="生成传感器报文", command=self.generate_sensor_packet, width=12).grid(row=row, column=0,pady=2)
        ttk.Button(self.sidebar_bottom, text="发送传感器报文", command=self.send_sensor_once, width=12).grid(row=row,column=1,pady=2)
        self.sensor_send_btn = ttk.Button(self.sidebar_bottom, text="循环发送传感器报文",command=self.toggle_sensor_send_loop, width=16)
        self.sensor_send_btn.grid(row=row, column=2,pady=2)
        row += 1

        ttk.Label(self.sidebar_bottom, text="").grid(row=row, column=0, pady=6)  # 占一行空白
        row += 1

        # 控制参数区
        ttk.Label(self.sidebar_bottom, text="AUV编号:").grid(row=row, column=0, sticky="e",pady=3)
        print(row)
        ttk.Entry(self.sidebar_bottom, textvariable=self.auv_id, width=4).grid(row=row, column=1, sticky="w",pady=5)
        row += 1
        # ========== 指令类型 + 运动模式 + 控制类型（紧凑排列，保留原文字） ==========
        ttk.Label(self.sidebar_bottom, text="指令类型:").grid(row=row, column=0, sticky="e",pady=3)
        # 创建嵌套 Frame 承载三组 Combobox
        combo_frame = ttk.Frame(self.sidebar_bottom)
        combo_frame.grid(row=row, column=1, columnspan=6, sticky="w",pady=3)
        # 指令类型
        ttk.Label(combo_frame, text="指令类型:").pack(side="left", padx=(0, 2), pady=3)
        ttk.Combobox(combo_frame, textvariable=self.command_type, values=list(COMMAND_TYPE.keys()), width=8).pack(
            side="left", padx=(0, 8))
        # 运动模式
        ttk.Label(combo_frame, text="运动模式:").pack(side="left", padx=(0, 2),pady=3)
        ttk.Combobox(combo_frame, textvariable=self.control_mode, values=list(CONTROL_TYPE.keys()), width=8).pack(
            side="left", padx=(0, 8))
        # 控制类型
        ttk.Label(combo_frame, text="控制类型:").pack(side="left", padx=(0, 2),pady=3)
        ttk.Combobox(combo_frame, textvariable=self.control_type, values=list(CONTROL_MODE.keys()), width=8).pack(
            side="left", padx=(0, 2))

        row += 1

        # 空间运动控制
        # 1. 标签
        ttk.Label(self.sidebar_bottom, text="空间运动(6自由度):").grid(row=row, column=0, sticky="e", padx=2,pady=3)
        # 2. 创建一个嵌套Frame来紧凑放置Entry
        entry_frame = ttk.Frame(self.sidebar_bottom)
        entry_frame.grid(row=row, column=1, columnspan=6, sticky="w",pady=3)
        # 3. 把所有 Entry 紧贴放进 Frame
        for i in range(6):
            ttk.Entry(entry_frame, textvariable=self.motion_values[i], width=3).pack(side="left", padx=1,pady=3)
        row += 1

        # 设备启动（只保留3项） - 紧凑排列版
        ttk.Label(self.sidebar_bottom, text="设备启动:").grid(row=row, column=0, sticky="e", padx=2,pady=3)
        # 使用一个 Frame 承载3个 Checkbutton
        device_frame = ttk.Frame(self.sidebar_bottom)
        device_frame.grid(row=row, column=1, columnspan=3, sticky="w",pady=3)  # 横跨3列
        for i in range(3):
            ttk.Checkbutton(device_frame, text=DEVICE_LIST[i], variable=self.device_start_bits[i]).pack(side="left",padx=2,pady=3)
        row += 1

        # LED与机械手（紧凑排列）
        ttk.Label(self.sidebar_bottom, text="LED与机械手:").grid(row=row, column=0, sticky="e", padx=2,pady=3)
        # 创建一个横向排列 Frame
        led_frame = ttk.Frame(self.sidebar_bottom)
        led_frame.grid(row=row, column=1, columnspan=6, sticky="w",pady=3)
        # LED(绿色)
        ttk.Label(led_frame, text="绿:").pack(side="left", padx=(0, 2),pady=3)
        ttk.Entry(led_frame, textvariable=self.led_values[0], width=6).pack(side="left", padx=(0, 6),pady=3)
        # LED(红色)
        ttk.Label(led_frame, text="红:").pack(side="left", padx=(0, 2),pady=3)
        ttk.Entry(led_frame, textvariable=self.led_values[1], width=6).pack(side="left", padx=(0, 6),pady=3)
        # 机械手
        ttk.Label(led_frame, text="机械手:").pack(side="left", padx=(0, 2),pady=3)
        ttk.Entry(led_frame, textvariable=self.gimbal_value, width=6).pack(side="left", padx=(0, 2),pady=3)

        row += 1

        # 补光灯控制UI
        # 补光灯（紧凑排列）
        ttk.Label(self.sidebar_bottom, text="补光灯:").grid(row=row, column=0, sticky="e", padx=2, pady=3)
        light_frame = ttk.Frame(self.sidebar_bottom)
        light_frame.grid(row=row, column=1, columnspan=6, sticky="w", pady=3)
        # 补光灯1
        ttk.Label(light_frame, text="补光灯1(0-100):").pack(side="left", padx=(0, 2), pady=3)
        ttk.Entry(light_frame, textvariable=self.light1_value, width=6).pack(side="left", padx=(0, 6), pady=3)
        # 补光灯2
        ttk.Label(light_frame, text="补光灯2(0-100):").pack(side="left", padx=(0, 2), pady=3)
        ttk.Entry(light_frame, textvariable=self.light2_value, width=6).pack(side="left", padx=(0, 2), pady=3)

        row +=1

        # ========== 点控制 UI（X/Y/Z） ==========
        ttk.Label(self.sidebar_bottom, text="点控制X:").grid(row=row, column=0, sticky="e", padx=2,pady=3)
        point_frame = ttk.Frame(self.sidebar_bottom)
        point_frame.grid(row=row, column=1, columnspan=6, sticky="w",pady=3)
        ttk.Label(point_frame, text="X(前,m):").pack(side="left", padx=(0, 2),pady=3)
        ttk.Entry(point_frame, textvariable=self.point_x, width=6).pack(side="left", padx=(0, 6),pady=3)
        ttk.Label(point_frame, text="Y(右,m):").pack(side="left", padx=(0, 2),pady=3)
        ttk.Entry(point_frame, textvariable=self.point_y, width=6).pack(side="left", padx=(0, 6),pady=3)
        ttk.Label(point_frame, text="Z(下,m):").pack(side="left", padx=(0, 2),pady=3)
        ttk.Entry(point_frame, textvariable=self.point_z, width=6).pack(side="left", padx=(0, 2),pady=3)
        row += 1

        # ========== 姿态偏移 UI（Roll/Pitch/Yaw） ==========
        ttk.Label(self.sidebar_bottom, text="姿态偏移(°):").grid(row=row, column=0, sticky="e", padx=2,pady=3)
        attitude_frame = ttk.Frame(self.sidebar_bottom)
        attitude_frame.grid(row=row, column=1, columnspan=6, sticky="w",pady=3)
        self.point_roll = tk.DoubleVar(value=0.0)
        ttk.Label(attitude_frame, text="横滚偏移:").pack(side="left", padx=(0, 2),pady=3)
        ttk.Entry(attitude_frame, textvariable=self.point_roll, width=6).pack(side="left", padx=(0, 6),pady=3)
        self.point_pitch = tk.DoubleVar(value=0.0)
        ttk.Label(attitude_frame, text="俯仰偏移:").pack(side="left", padx=(0, 2),pady=3)
        ttk.Entry(attitude_frame, textvariable=self.point_pitch, width=6).pack(side="left", padx=(0, 6),pady=3)
        self.point_yaw = tk.DoubleVar(value=0.0)
        ttk.Label(attitude_frame, text="航向偏移:").pack(side="left", padx=(0, 2),pady=3)
        ttk.Entry(attitude_frame, textvariable=self.point_yaw, width=6).pack(side="left", padx=(0, 2),pady=3)
        row += 1


        # # 数据显示区（左：75协议原始/解析，右：110协议原始/解析）
        # frame = ttk.Frame(self.main_frame)
        # frame.grid(row=row, column=0, columnspan=12, sticky="nsew")
        # 75协议原始HEX

        ttk.Label(main_content, text="75协议原始HEX:").grid(row=0, column=0, sticky="ne")
        # self.recv_text = tk.Text(main_content, width=55, height=8)
        self.recv_text = tk.Text(main_content, width=55, height=4)
        self.recv_text.grid(row=0, column=1, sticky="w")
        # 75协议解析
        ttk.Label(main_content, text="75协议解析:").grid(row=1, column=0, sticky="ne")
        # self.user75_text = tk.Text(main_content, width=55, height=30)
        self.user75_text = tk.Text(main_content, width=55, height=15)
        self.user75_text.grid(row=1, column=1, sticky="w")
        # 110协议原始HEX
        # ttk.Label(main_content, text="110协议原始HEX:").grid(row=0, column=2, sticky="ne")
        # self.debug_raw_text = tk.Text(main_content, width=55, height=8)
        ttk.Label(main_content, text="110协议原始HEX:").grid(row=2, column=0, sticky="ne")
        self.debug_raw_text = tk.Text(main_content, width=55, height=8)
        # self.debug_raw_text.grid(row=0, column=3, sticky="w")
        self.debug_raw_text.grid(row=2, column=1, sticky="w")
        # 110协议解析
        # ttk.Label(main_content, text="110协议解析:").grid(row=1, column=2, sticky="ne")
        ttk.Label(main_content, text="110协议解析:").grid(row=3, column=0, sticky="ne")
        self.debug_recv_text = tk.Text(main_content, width=55, height=15)  #height 30 -> 15
        # self.debug_recv_text.grid(row=1, column=3, sticky="w")
        self.debug_recv_text.grid(row=3, column=1, sticky="w")

        # 在frame中添加传感器数据显示区域
        ttk.Label(main_content, text="传感器数据:").grid(row=4, column=0, sticky="ne")
        self.sensor_text = tk.Text(main_content, width=55, height=5)
        self.sensor_text.grid(row=4, column=1, sticky="w")

        # 线程相关初始化
        self.recv_thread = None
        self.is_receiving = False
        self.debug_recv_thread = None
        self.is_debug_receiving = False

        # 新增：传感器接收相关变量
        self.is_sensor_receiving = False
        self.sensor_recv_thread = None

    # ==== UI: 绘制背景 ====
    def createWidget(self):
        self.canvas = tk.Canvas(root,width=1200, height =800)
        self.canvas.pack(fill="both",expand=True)
        self.canvas.create_rectangle(0, 0, 600, 400, fill="#CCE6FF", outline="")

        # 创建前景 Frame，放控件（用 create_window 嵌入 Canvas）
        #self.main_frame = ttk.Frame(self.canvas)
        self.main_frame = ttk.Frame(self.canvas, padding=10, style="Transparent.TFrame")
        self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

    #绑定窗口
    def on_resize(self, event):
        # 获取窗口当前大小
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        # 调整 Canvas 大小
        self.canvas.config(width=width, height=height)
        # 清空旧背景
        self.canvas.delete("bg")
        # 重新绘制背景（用 tag 标记）
        self.canvas.create_rectangle(0, 0, width, height, fill="#CCE6FF", outline="", tags="bg")


    def get_device_start_bytes(self):
        """
        获取设备启动字节，只保留惯导、DVL、PC机
        """
        bits = [v.get() for v in self.device_start_bits]
        value = 0
        for i, b in enumerate(bits):
            if b:
                # 惯导bit0, DVL bit1, PC机bit15
                if i >1:
                    value |= (1 << 15)
                else:
                    value |= (1 << i)

        return [(value & 0xFF), ((value >> 8) & 0xFF)]

    def toggle_user_connect(self):
        """
        根据连接模式选择串口或TCP连接
        """
        mode = self.conn_mode.get()
        if mode == "串口":
            if self.connect_serial():
                self.toggle_user_recv_loop()
        else:
            if self.connect_user_tcp():
                self.toggle_user_recv_loop()

    def toggle_user_disconnect(self):
        """
        根据连接模式选择断开串口或TCP
        """
        mode = self.conn_mode.get()
        # 先停止接收线程
        self.is_receiving = False
        if self.recv_thread:
            self.recv_thread.join(timeout=1.0)  # 等待接收线程结束
    
        if mode == "串口":
            self.disconnect_serial()
        else:
            self.disconnect_user_tcp()
            
    def connect_serial(self):
        """
        打开串口连接
        """
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(self.com_port.get(), self.baudrate.get(), timeout=1)
            messagebox.showinfo("用户串口连接", f"已连接到 {self.com_port.get()}")
            return True
        except Exception as e:
            messagebox.showerror("用户串口错误", str(e))
            return False

    def disconnect_serial(self):
        """
        关闭串口连接
        """
        try:
            if self.ser is not None:
                if self.ser.is_open:
                    self.ser.close()
                self.ser = None
            self.is_sending = False
            messagebox.showinfo("用户串口断开", "串口已断开")
        except Exception as e:
            # 忽略 bad file descriptor 错误
            if "Bad file descriptor" not in str(e):
                messagebox.showerror("用户串口错误", str(e))

    def toggle_sensor_connect(self):
        """
        建立传感器TCP连接
        """
        try:
            if self.sensor_sock is not None:
                self.sensor_sock.close()
            self.sensor_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sensor_sock.settimeout(3)
            self.sensor_sock.connect((self.sensor_tcp_ip.get(), self.sensor_tcp_port.get()))
            messagebox.showinfo("传感器TCP连接", f"已连接到 {self.sensor_tcp_ip.get()}:{self.sensor_tcp_port.get()}")
            return True
        except Exception as e:
            self.sensor_sock = None
            messagebox.showerror("传感器TCP错误", str(e))
            return False
        
    def toggle_sensor_disconnect(self):
        """
        断开传感器TCP连接
        """
        try:
            # 先停止发送线程
            self.is_sensor_sending = False
            if self.sensor_send_thread:
                self.sensor_send_thread.join(timeout=1.0)  # 等待发送线程结束
            
            if self.sensor_sock is not None:
                try:
                    self.sensor_sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.sensor_sock.close()
                self.sensor_sock = None
            messagebox.showinfo("传感器TCP断开", "传感器TCP已断开")
        except Exception as e:
            if "Bad file descriptor" not in str(e):
                messagebox.showerror("传感器TCP错误", str(e))

    def connect_user_tcp(self):
        """
        建立TCP连接
        """
        try:
            if self.sock is not None:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3)
            self.sock.connect((self.tcp_ip.get(), self.tcp_port.get()))
            messagebox.showinfo("用户TCP连接", f"已连接到 {self.tcp_ip.get()}:{self.tcp_port.get()}")
            return True
        except Exception as e:
            self.sock = None
            messagebox.showerror("用户TCP错误", str(e))
            return False

    def disconnect_user_tcp(self):
        """
        断开TCP连接
        """
        try:
            if self.sock is not None:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.sock.close()
                self.sock = None
            self.is_sending = False
            messagebox.showinfo("用户TCP断开", "TCP已断开")
        except Exception as e:
            if "Bad file descriptor" not in str(e):
                messagebox.showerror("用户TCP错误", str(e))

    def toggle_debug_connect(self):
        """
        连接调试协议TCP端口并自动启动接收线程
        """
        try:
            if self.debug_sock is not None:
                self.debug_sock.close()
            self.debug_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.debug_sock.settimeout(3)
            self.debug_sock.connect((self.debug_tcp_ip.get(), self.debug_tcp_port.get()))
            messagebox.showinfo("调试TCP连接", f"已连接到 {self.debug_tcp_ip.get()}:{self.debug_tcp_port.get()}")
            # 自动启动调试接收线程
            if not self.is_debug_receiving:
                self.is_debug_receiving = True
                self.debug_recv_thread = threading.Thread(target=self.debug_recv_loop, daemon=True)
                self.debug_recv_thread.start()
        except Exception as e:
            self.debug_sock = None
            messagebox.showerror("调试TCP错误", str(e))

    def toggle_debug_disconnect(self):
        """
        断开调试协议TCP端口
        """
        try:
            # 先停止接收线程
            self.is_debug_receiving = False
            if self.debug_recv_thread:
                self.debug_recv_thread.join(timeout=1.0)  # 等待接收线程结束
            
            if self.debug_sock is not None:
                try:
                    self.debug_sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.debug_sock.close()
                self.debug_sock = None
            messagebox.showinfo("调试TCP断开", "调试协议TCP已断开")
        except Exception as e:
            if "Bad file descriptor" not in str(e):
                messagebox.showerror("调试TCP错误", str(e))

    # ========== 110字节调试协议TCP相关 =========

    def toggle_debug_recv_loop(self):
        """
        启动调试协议接收线程
        """
        if not self.is_debug_receiving:
            self.is_debug_receiving = True
            self.debug_recv_thread = threading.Thread(target=self.debug_recv_loop, daemon=True)
            self.debug_recv_thread.start()

    def stop_debug_recv_loop(self):
        """
        停止调试协议接收线程
        """
        self.is_debug_receiving = False

    def debug_recv_loop(self):
        """
        调试协议接收线程函数，解析110字节协议并显示
        """
        buffer = bytearray()
        while self.is_debug_receiving:
            try:
                if self.debug_sock is None:
                    self.toggle_debug_connect()
                if self.debug_sock is not None:
                    data = self.debug_sock.recv(512)
                    if data:
                        buffer += data
                        while len(buffer) >= 110:
                            # 查找包头和包尾
                            start = buffer.find(b'\xFE\xEF')
                            if start == -1 or len(buffer) - start < 110:
                                break
                            if buffer[start+108:start+110] == b'\xFA\xAF':
                                packet = buffer[start:start+110]
                                buffer = buffer[start+110:]
                                # 保存原始包用于显示
                                self.last_debug_packet = packet
                                # 校验
                                if AUVSerialHandler.calculate_checksum(self, packet) == packet[107]:
                                    parsed = AUVSerialHandler.parse_packet(self, packet)
                                    self.show_debug_recv_data(parsed)
                                else:
                                    self.show_debug_recv_data("调试校验失败")
                            else:
                                buffer = buffer[start+2:]
            except Exception as e:
                self.is_debug_receiving = False
                messagebox.showerror("调试接收错误", str(e))
                break
                # raise e
            # 为了避免过度占用CPU，增加短暂休眠
            time.sleep(0)

    def show_debug_recv_data(self, data):
        """
        在UI中显示调试协议解析结果
        """
        # 先显示原始HEX
        if hasattr(self, 'last_debug_packet') and isinstance(self.last_debug_packet, (bytes, bytearray)):
            hex_str = " ".join(f"{b:02X}" for b in self.last_debug_packet)
            self.debug_raw_text.delete(1.0, tk.END)
            self.debug_raw_text.insert(tk.END, hex_str)
        self.debug_recv_text.delete(1.0, tk.END)
        if isinstance(data, str):
            self.debug_recv_text.insert(tk.END, data)
        else:
            # ========== 存储AUV当前经纬度/深度/姿态 ==========
            try:
                auv_state.latitude = data.navigation_coords[1]
                auv_state.longitude = data.navigation_coords[0]
                auv_state.depth = data.depth
                auv_state.yaw = data.euler_angles[2]
                auv_state.pitch = data.euler_angles[1]
                auv_state.roll = data.euler_angles[0]
            except Exception:
                pass
            for k, v in vars(data).items():
                self.debug_recv_text.insert(tk.END, f"{k}: {v}\n")

    # ========== 75字节用户协议串口/TCP相关 =========
    def toggle_user_recv_loop(self):
        """
        启动用户协议接收线程
        """
        if not self.is_receiving:
            self.is_receiving = True
            self.recv_thread = threading.Thread(target=self.user_recv_loop, daemon=True)
            self.recv_thread.start()

    def stop_user_recv_loop(self):
        """
        停止用户协议接收线程
        """
        self.is_receiving = False

    def user_recv_loop(self):
        """
        用户协议接收线程函数，显示75字节原始HEX和解析
        """
        buffer = bytearray()
        while self.is_receiving:
            try:
                if self.conn_mode.get() == "串口":
                    if self.ser is None or not self.ser.is_open:
                        self.connect_serial()
                    if self.ser is not None and self.ser.is_open:
                        if not self.is_receiving:  # 增加检查，如果停止接收则退出
                            break
                        data = self.ser.read(75)
                        if data and len(data) == 75:
                            self.show_user_recv_data(data)
                            self.show_user75_data(data)
                else:
                    if self.sock is None:
                        self.connect_user_tcp()
                    if self.sock is not None:
                        if not self.is_receiving:  # 增加检查，如果停止接收则退出
                            break
                        data = self.sock.recv(75)
                        if data and len(data) == 75:
                            self.show_user_recv_data(data)
                            self.show_user75_data(data)
            except Exception as e:
                if not self.is_receiving:  # 如果是主动停止，不显示错误
                    break
                self.is_receiving = False
                messagebox.showerror("用户接收错误", str(e))
            time.sleep(0.01)

    def show_user_recv_data(self, data):
        """
        在UI中显示用户协议原始HEX数据
        """
        self.recv_text.delete(1.0, tk.END)
        if isinstance(data, bytes) or isinstance(data, bytearray):
            hex_str = " ".join(f"{b:02X}" for b in data)
            self.recv_text.insert(tk.END, hex_str)
        else:
            self.recv_text.insert(tk.END, str(data))

    def show_user75_data(self, data):
        """
        在UI中显示75协议解析结果
        """
        self.user75_text.delete(1.0, tk.END)
        pkt = parse_75_packet(data)
        if pkt is None:
            self.user75_text.insert(tk.END, "75协议解析失败")
            return
        for k, v in vars(pkt).items():
            self.user75_text.insert(tk.END, f"{k}: {v}\n")

    def validate_number(self, value, name, allow_float=True, min_val=None, max_val=None):
        """
        验证数值是否合法
        :param value: 要验证的值
        :param name: 字段名称（用于错误提示）
        :param allow_float: 是否允许浮点数
        :param min_val: 最小值限制
        :param max_val: 最大值限制
        :return: 转换后的数值，如果无效则返回None
        """
        try:
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    messagebox.showerror("输入错误", f"{name}不能为空")
                    return None
        
            # 尝试转换为数值
            num = float(value) if allow_float else int(value)
        
            # 检查范围
            if min_val is not None and num < min_val:
                messagebox.showerror("输入错误", f"{name}不能小于{min_val}")
                return None
            if max_val is not None and num > max_val:
                messagebox.showerror("输入错误", f"{name}不能大于{max_val}")
                return None
            
            return num
        except ValueError:
            messagebox.showerror("输入错误", f"{name}必须是有效的{'数值' if allow_float else '整数'}")
            return None

    def generate_user_packet(self):
        """
        生成发送报文并显示HEX
        """
        try:
            # 验证AUV ID
            auv_id = self.validate_number(self.auv_id.get(), "AUV编号", allow_float=False, min_val=0, max_val=255)
            if auv_id is None:
                return None

            # 验证运动控制值
            motion_values = []
            for i, v in enumerate(self.motion_values):
                val = self.validate_number(v.get(), f"运动控制值{i+1}", allow_float=False, min_val=0, max_val=255)
                if val is None:
                    return None
                motion_values.append(val)

            # 验证LED值
            led_values = []
            for i, v in enumerate(self.led_values):
                val = self.validate_number(v.get(), f"LED值{i+1}", allow_float=False, min_val=0, max_val=255)
                if val is None:
                    return None
                led_values.append(val)

            # 验证云台值
            gimbal = self.validate_number(self.gimbal_value.get(), "云台值", allow_float=False, min_val=0, max_val=255)
            if gimbal is None:
                return None

            packet = build_packet(
                auv_id,
                COMMAND_TYPE[self.command_type.get()],
                CONTROL_TYPE[self.control_mode.get()],
                CONTROL_MODE[self.control_type.get()],
                motion_values,
                self.get_device_start_bytes(),
                led_values,
                gimbal
            )
            self.packet_hex.set(packet.hex())
            return packet
        except Exception as e:
            #messagebox.showerror("错误", f"生成报文失败: {str(e)}")
            raise e  # 重新抛出异常以便上层处理

    def send_user_once(self):
        """
        发送一次报文
        """
        packet = self.generate_user_packet()
        if packet is None:
            return
        mode = self.conn_mode.get()
        try:
            if mode == "串口":
                if self.ser is not None and self.ser.is_open:
                    self.ser.write(packet)
                    messagebox.showinfo("发送成功", "用户报文已发送")
            else:
                if self.sock is not None:
                    self.sock.sendall(packet)
                    messagebox.showinfo("发送成功", "用户报文已发送")
        except Exception as e:
            messagebox.showerror(f"{mode}错误", str(e))

    def toggle_user_send_loop(self):
        """
        启动或停止循环发送线程
        """
        if not self.is_sending:
            self.is_sending = True
            self.send_btn.config(text="停止循环发送")
            self.send_thread = threading.Thread(target=self.user_send_loop, daemon=True)
            self.send_thread.start()
        else:
            self.is_sending = False
            self.send_btn.config(text="循环发送用户报文")

    def user_send_loop(self):
        """
        循环发送报文线程函数
        """
        mode = self.conn_mode.get()
        while self.is_sending:
            try:
                packet = self.generate_user_packet()
                if packet is not None:
                    if mode == "串口":
                        if self.ser is not None and self.ser.is_open:
                            self.ser.write(packet)
                        else:
                            raise Exception("用户串口未连接")
                    else:
                        if self.sock is not None:
                            self.sock.sendall(packet)
                        else:
                            raise Exception("用户TCP未连接")
            except Exception as e:
                self.is_sending = False
                self.send_btn.config(text="循环发送用户报文")
                messagebox.showerror(f"{mode}错误", str(e))
                break  # 发生错误直接退出循环
            time.sleep(0.2)  # 5Hz

    # ========== SENSOR报文相关 ==========
    def build_sensor_packet(self):
        """
        构造54字节控制报文，LED和舵机字节从UI获取，其余字节全0（除头尾校验）
        """
        packet = bytearray(54)
        packet[0:2] = b'\xFE\xFE'
        packet[2] = 0x00
        packet[3] = self.auv_id.get()
        packet[4] = 0x00  # 指令类型
        packet[5] = 0x00  # 运动控制模式
        packet[6] = 0x00  # 控制类型
        for i in range(7, 33):
            packet[i] = 0x00
        packet[32] = 0xff
        # 33~38 空间运动控制
        for i in range(6):
            packet[33 + i] = 0x00
        # 39~40 设备启动
        packet[39] = 0x00
        packet[40] = 0x00
        # 41~44 LED灯
        for i in range(2):
            packet[41 + i] = self.led_values[i].get()
        # 45~49 保留
        for i in range(45, 49):
            packet[i] = 0x00
        # 50 云台
        packet[46] = self.gimbal_value.get()
        # 51 校验
        xor = 0
        for i in range(0, 51):
            xor ^= packet[i]
        packet[51] = xor
        packet[52:54] = b'\xFD\xFD'
        return packet

    def generate_sensor_packet(self):
        """
        生成传感器报文并显示HEX
        """
        try:
            packet = self.build_sensor_packet()
            self.sensor_packet_hex.set(packet.hex())
            return packet
        except Exception as e:
            raise e  # 重新抛出异常以便上层处理

    def send_sensor_once(self):
        """
        发送一次传感器报文
        """
        packet = self.generate_sensor_packet()
        if packet is None:
            return
        try:
            if self.sensor_sock is not None:
                self.sensor_sock.sendall(packet)
                messagebox.showinfo("传感器报文发送", "传感器报文已发送")
        except Exception as e:
            messagebox.showerror("传感器报文发送错误", str(e))

    def toggle_sensor_send_loop(self):
        """
        传感器报文循环发送开关
        """
        if not self.is_sensor_sending:
            self.is_sensor_sending = True
            self.sensor_send_btn.config(text="停止传感器循环")
            self.sensor_send_thread = threading.Thread(target=self.sensor_send_loop, daemon=True)
            self.sensor_send_thread.start()
        else:
            self.is_sensor_sending = False
            self.sensor_send_btn.config(text="循环发送传感器报文")

    def sensor_send_loop(self):
        """
        传感器报文循环发送线程
        """
        while self.is_sensor_sending:
            try:
                packet = self.generate_sensor_packet()
                if packet is not None:
                    if self.sensor_sock is not None:
                        self.sensor_sock.sendall(packet)
                    else:
                        raise Exception("传感器TCP未连接")
            except Exception as e:
                self.is_sensor_sending = False
                self.sensor_send_btn.config(text="循环发送传感器报文")
                messagebox.showerror("传感器报文循环发送错误", str(e))
                break  # 发生错误直接退出循环
            time.sleep(0.2)  # 5Hz
    
    # ========== DEBUG报文相关 ==========
    def build_debug_packet(self, x, y, z):
        """
        构造点控制的54字节扩展报文，前右下(m)自动转换为经纬度/深度，
        并加入姿态控制（当前姿态+偏移量）
        """
        # 1. 获取当前AUV经纬度/深度/姿态
        # 2. 调用navigation_utils.local_point_to_global
        # 3. 用转换后的经纬度/深度填充报文
        lat, lon, depth = local_point_to_global(auv_state, (x, y, z))
        # print(lat,lon,depth)

        # 计算目标姿态（当前姿态+偏移量）
        target_roll = auv_state.roll + self.point_roll.get()
        target_pitch = auv_state.pitch + self.point_pitch.get()
        target_yaw = auv_state.yaw + self.point_yaw.get()

        packet = bytearray(54)
        packet[0:2] = b'\xFE\xFE'
        packet[2:4] = b'\x00\x01'
        packet[4] = 0x30
        packet[5] = 0x04
        packet[6] = 0x01
        packet[7] = 0x00
        # 8-11: 经度
        packet[8:12] = struct.pack('<i', int(lon * 1e7))
        # 12-15: 纬度
        packet[12:16] = struct.pack('<i', int(lat * 1e7))
        # 16-19: 深度
        packet[16:20] = struct.pack('<f', depth)
        # 20-23: 横滚角
        packet[20:24] = struct.pack('<f', target_roll)
        # 24-27: 俯仰角
        packet[24:28] = struct.pack('<f', target_pitch)
        # 28-31: 偏航角
        packet[28:32] = struct.pack('<f', target_yaw)
        for i in range(32, 44):
            packet[i] = 0x00
        packet[44] = 0x00
        for i in range(45, 51):
            packet[i] = 0x00
        xor = 0
        for i in range(0, 51):
            xor ^= packet[i]
        packet[51] = xor
        packet[52:54] = b'\xFD\xFD'
        return packet

    def generate_debug_packet(self):
        """
        生成点控制报文并显示HEX
        """
        try:
            # 验证坐标值
            x = self.validate_number(self.point_x.get(), "X坐标", allow_float=True)
            if x is None:
                return None
            
            y = self.validate_number(self.point_y.get(), "Y坐标", allow_float=True)
            if y is None:
                return None
            
            z = self.validate_number(self.point_z.get(), "Z坐标", allow_float=True)
            if z is None:
                return None

            # 验证姿态偏移值
            roll = self.validate_number(self.point_roll.get(), "横滚偏移", allow_float=True, min_val=-180, max_val=180)
            if roll is None:
                return None
            
            pitch = self.validate_number(self.point_pitch.get(), "俯仰偏移", allow_float=True, min_val=-90, max_val=90)
            if pitch is None:
                return None
            
            yaw = self.validate_number(self.point_yaw.get(), "航向偏移", allow_float=True, min_val=-180, max_val=180)
            if yaw is None:
                return None

            packet = self.build_debug_packet(x, y, z)
            self.point_packet_hex.set(packet.hex())
            return packet
        except Exception as e:
            raise e  # 重新抛出异常以便上层处理
    
    def send_debug_once(self):
        """
        发送一次点控制扩展报文
        """
        packet = self.generate_debug_packet()
        if packet is None:
            return
        try:
            if self.sock is not None:
                self.sock.sendall(packet)
                messagebox.showinfo("报文发送", "调试报文已发送")
            else:
                messagebox.showerror("连接错误", "请先建立TCP连接")
        except Exception as e:
            messagebox.showerror("调试报文发送错误", str(e))

    def toggle_debug_send_loop(self):
        """
        启动或停止点控制扩展报文循环发送线程
        """
        if not self.is_point_sending:
            self.is_point_sending = True
            self.point_send_btn.config(text="停止扩展报文循环")
            self.point_send_thread = threading.Thread(target=self.debug_send_loop, daemon=True)
            self.point_send_thread.start()
        else:
            self.is_point_sending = False
            self.point_send_btn.config(text="循环发送扩展报文")

    def debug_send_loop(self):
        """
        debug扩展报文循环发送线程函数
        """
        while self.is_point_sending:
            try:
                packet = self.generate_debug_packet()
                if packet is not None:
                    if self.debug_sock is not None:
                        self.debug_sock.sendall(packet)
                    else:
                        raise Exception("DEBUG TCP未连接")
            except Exception as e:
                self.is_point_sending = False
                self.point_send_btn.config(text="循环发送扩展报文")
                messagebox.showerror("扩展报文循环发送错误", str(e))
                break  # 发生错误直接退出循环
            time.sleep(0.2)  # 5Hz发送频率

if __name__ == "__main__":
    # 保证 root 是从 Style().master 拿到的
    style = Style()
    root = style.master
    style.configure("Transparent.TFrame", background="#CCE6FF")
    style.configure("Transparent.TLabel", background="#CCE6FF")

    #root = tk.Tk()
    app = SenderUI(root)
    root.geometry("1350x825")
    root.mainloop()