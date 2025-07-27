"""
    AUV协议发送器
    时间：2025年5月26日
    作者：hsx
    版本：1.0
"""
import serial
import time
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import socket
from functools import reduce
import struct

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

# 设备启动16项
DEVICE_LIST = [
    "惯导", "DVL", "USBL", "声通信机", "高度计", "深度计", "侧扫声呐", "前视声呐",
    "单波束", "多波束", "避碰声呐", "频闪灯", "ADCP", "CTD", "浅剖", "PC机"
]


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
            data.utc_time = list(packet[90:96]) + [struct.unpack('<f', packet[95:99])[0]]
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

        # 参数变量
        self.auv_id = tk.IntVar(value=1)
        self.command_type = tk.StringVar(value="控制模式")
        self.control_mode = tk.StringVar(value="自控")
        self.control_type = tk.StringVar(value="开环")
        self.motion_values = [tk.IntVar(value=100) for _ in range(6)]
        # 设备启动相关
        self.device_start0 = tk.IntVar(value=0)
        self.device_start1 = tk.IntVar(value=0)
        self.device_start_bits = [tk.IntVar(value=0) for _ in range(16)]  # 新增：16个设备开关
        self.led_values = [tk.IntVar(value=0) for _ in range(4)]
        self.gimbal_value = tk.IntVar(value=0)
        self.packet_hex = tk.StringVar(value="")
        self.com_port = tk.StringVar(value="COM31")
        self.baudrate = tk.IntVar(value=230400)
        self.is_sending = False
        self.ser = None
        self.send_thread = None

        # 新增：连接模式相关变量
        self.conn_mode = tk.StringVar(value="串口")
        self.tcp_ip = tk.StringVar(value="47.104.21.115")
        self.tcp_port = tk.IntVar(value=5322)
        self.sock = None

        # 新增：调试协议TCP端口相关变量
        self.debug_tcp_ip = tk.StringVar(value="47.104.21.115")
        self.debug_tcp_port = tk.IntVar(value=5323)
        self.debug_sock = None
        self.debug_recv_thread = None
        self.is_debug_receiving = False

        # UI布局
        row = 0
        # 连接参数区
        ttk.Label(master, text="连接模式:").grid(row=row, column=0, sticky="e")
        ttk.Combobox(master, textvariable=self.conn_mode, values=["串口", "TCP"], width=7, state="readonly").grid(row=row, column=1, sticky="w")
        ttk.Label(master, text="串口:").grid(row=row, column=2, sticky="e")
        ttk.Entry(master, textvariable=self.com_port, width=8).grid(row=row, column=3, sticky="w")
        ttk.Label(master, text="波特率:").grid(row=row, column=4, sticky="e")
        ttk.Entry(master, textvariable=self.baudrate, width=8).grid(row=row, column=5, sticky="w")
        ttk.Label(master, text="TCP IP:").grid(row=row, column=6, sticky="e")
        ttk.Entry(master, textvariable=self.tcp_ip, width=12).grid(row=row, column=7, sticky="w")
        ttk.Label(master, text="端口:").grid(row=row, column=8, sticky="e")
        ttk.Entry(master, textvariable=self.tcp_port, width=6).grid(row=row, column=9, sticky="w")
        ttk.Button(master, text="连接", command=self.connect, width=6).grid(row=row, column=10)
        ttk.Button(master, text="断开", command=self.disconnect, width=6).grid(row=row, column=11)
        row += 1

        # 调试协议TCP参数
        ttk.Label(master, text="调试IP:").grid(row=row, column=0, sticky="e")
        ttk.Entry(master, textvariable=self.debug_tcp_ip, width=12).grid(row=row, column=1, sticky="w")
        ttk.Label(master, text="调试端口:").grid(row=row, column=2, sticky="e")
        ttk.Entry(master, textvariable=self.debug_tcp_port, width=6).grid(row=row, column=3, sticky="w")
        ttk.Button(master, text="连接调试", command=self.connect_debug_tcp, width=8).grid(row=row, column=4)
        ttk.Button(master, text="断开调试", command=self.disconnect_debug_tcp, width=8).grid(row=row, column=5)
        ttk.Button(master, text="开始调试接收", command=self.toggle_debug_recv_loop, width=12).grid(row=row, column=6)
        ttk.Button(master, text="停止调试接收", command=self.stop_debug_recv_loop, width=12).grid(row=row, column=7)
        row += 1

        # 控制参数区
        ttk.Label(master, text="AUV编号:").grid(row=row, column=0, sticky="e")
        ttk.Entry(master, textvariable=self.auv_id, width=4).grid(row=row, column=1, sticky="w")
        ttk.Label(master, text="指令类型:").grid(row=row, column=2, sticky="e")
        ttk.Combobox(master, textvariable=self.command_type, values=list(COMMAND_TYPE.keys()), width=10).grid(row=row, column=3, sticky="w")
        ttk.Label(master, text="运动模式:").grid(row=row, column=4, sticky="e")
        ttk.Combobox(master, textvariable=self.control_mode, values=list(CONTROL_TYPE.keys()), width=10).grid(row=row, column=5, sticky="w")
        ttk.Label(master, text="控制类型:").grid(row=row, column=6, sticky="e")
        ttk.Combobox(master, textvariable=self.control_type, values=list(CONTROL_MODE.keys()), width=10).grid(row=row, column=7, sticky="w")
        row += 1

        # 空间运动控制
        ttk.Label(master, text="空间运动(6自由度):").grid(row=row, column=0, sticky="e")
        for i in range(6):
            ttk.Entry(master, textvariable=self.motion_values[i], width=3).grid(row=row, column=1+i, sticky="w")
        row += 1

        # 设备启动
        ttk.Label(master, text="设备启动:").grid(row=row, column=0, sticky="ne")
        for i in range(8):
            ttk.Checkbutton(master, text=DEVICE_LIST[i], variable=self.device_start_bits[i]).grid(row=row, column=1+i, sticky="w")
        row += 1
        for i in range(8, 16):
            ttk.Checkbutton(master, text=DEVICE_LIST[i], variable=self.device_start_bits[i]).grid(row=row, column=1+(i-8), sticky="w")
        ttk.Label(master, text="(或手动)").grid(row=row, column=9, sticky="w")
        ttk.Entry(master, textvariable=self.device_start0, width=3).grid(row=row, column=10, sticky="w")
        ttk.Entry(master, textvariable=self.device_start1, width=3).grid(row=row, column=11, sticky="w")
        row += 1

        # LED与机械手
        ttk.Label(master, text="LED(2字节):").grid(row=row, column=0, sticky="e")
        for i in range(2):
            ttk.Entry(master, textvariable=self.led_values[i], width=3).grid(row=row, column=1+i, sticky="w")
        ttk.Label(master, text="机械手:").grid(row=row, column=3, sticky="e")
        ttk.Entry(master, textvariable=self.gimbal_value, width=3).grid(row=row, column=4, sticky="w")
        ttk.Button(master, text="生成报文", command=self.generate_packet, width=8).grid(row=row, column=5)
        ttk.Button(master, text="发送报文", command=self.send_packet_once, width=8).grid(row=row, column=6)
        self.send_btn = ttk.Button(master, text="开始循环发送", command=self.toggle_send_loop, width=12)
        self.send_btn.grid(row=row, column=7)
        row += 1

        # 报文HEX显示
        ttk.Label(master, text="报文HEX:").grid(row=row, column=0, sticky="ne")
        ttk.Entry(master, textvariable=self.packet_hex, width=90).grid(row=row, column=1, columnspan=10, sticky="w")
        row += 1

        # 数据显示区（左：75协议原始/解析，右：110协议原始/解析）
        frame = ttk.Frame(master)
        frame.grid(row=row, column=0, columnspan=12, sticky="nsew")
        # 75协议原始HEX
        ttk.Label(frame, text="75协议原始HEX:").grid(row=0, column=0, sticky="ne")
        self.recv_text = tk.Text(frame, width=55, height=8)
        self.recv_text.grid(row=0, column=1, sticky="w")
        # 75协议解析
        ttk.Label(frame, text="75协议解析:").grid(row=1, column=0, sticky="ne")
        self.user75_text = tk.Text(frame, width=55, height=16)
        self.user75_text.grid(row=1, column=1, sticky="w")
        # 110协议原始HEX
        ttk.Label(frame, text="110协议原始HEX:").grid(row=0, column=2, sticky="ne")
        self.debug_raw_text = tk.Text(frame, width=55, height=8)
        self.debug_raw_text.grid(row=0, column=3, sticky="w")
        # 110协议解析
        ttk.Label(frame, text="110协议解析:").grid(row=1, column=2, sticky="ne")
        self.debug_recv_text = tk.Text(frame, width=55, height=30)
        self.debug_recv_text.grid(row=1, column=3, sticky="w")

        # 线程相关初始化
        self.recv_thread = None
        self.is_receiving = False

    def get_device_start_bytes(self):
        """
        获取设备启动字节，优先使用Checkbutton，否则用手动输入
        """
        # 优先使用Checkbutton的值，如果都为0则用手动输入
        bits = [v.get() for v in self.device_start_bits]
        if any(bits):
            value = 0
            for i, b in enumerate(bits):
                if b:
                    value |= (1 << i)
            return [(value & 0xFF), ((value >> 8) & 0xFF)]
        else:
            # 兼容原有输入框
            return [self.device_start0.get(), self.device_start1.get()]

    def connect(self):
        """
        根据连接模式选择串口或TCP连接
        """
        mode = self.conn_mode.get()
        if mode == "串口":
            if self.connect_serial():
                self.toggle_recv_loop()
        else:
            if self.connect_tcp():
                self.toggle_recv_loop()

    def disconnect(self):
        """
        根据连接模式选择断开串口或TCP
        """
        mode = self.conn_mode.get()
        if mode == "串口":
            self.stop_recv_loop()
            self.disconnect_serial()
            self.stop_recv_loop()
        else:
            self.stop_recv_loop()
            self.disconnect_tcp()

    def connect_serial(self):
        """
        打开串口连接
        """
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
            self.ser = serial.Serial(self.com_port.get(), self.baudrate.get(), timeout=1)
            messagebox.showinfo("串口连接", f"已连接到 {self.com_port.get()}")
            return True
        except Exception as e:
            messagebox.showerror("串口错误", str(e))
            return False

    def disconnect_serial(self):
        """
        关闭串口连接
        """
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
                self.ser = None
            self.is_sending = False
            messagebox.showinfo("串口断开", "串口已断开")
        except Exception as e:
            messagebox.showerror("串口错误", str(e))

    def connect_tcp(self):
        """
        建立TCP连接
        """
        try:
            if self.sock is not None:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(3)
            self.sock.connect((self.tcp_ip.get(), self.tcp_port.get()))
            messagebox.showinfo("TCP连接", f"已连接到 {self.tcp_ip.get()}:{self.tcp_port.get()}")
            return True
        except Exception as e:
            self.sock = None
            messagebox.showerror("TCP错误", str(e))
            return False

    def disconnect_tcp(self):
        """
        断开TCP连接
        """
        try:
            if self.sock is not None:
                self.sock.close()
                self.sock = None
            self.is_sending = False
            messagebox.showinfo("TCP断开", "TCP已断开")
        except Exception as e:
            messagebox.showerror("TCP错误", str(e))

    # ========== 110字节调试协议TCP相关 =========
    def connect_debug_tcp(self):
        """
        连接调试协议TCP端口
        """
        try:
            if self.debug_sock is not None:
                self.debug_sock.close()
            self.debug_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.debug_sock.settimeout(3)
            self.debug_sock.connect((self.debug_tcp_ip.get(), self.debug_tcp_port.get()))
            messagebox.showinfo("调试协议TCP连接", f"已连接到 {self.debug_tcp_ip.get()}:{self.debug_tcp_port.get()}")
        except Exception as e:
            self.debug_sock = None
            messagebox.showerror("调试协议TCP错误", str(e))

    def disconnect_debug_tcp(self):
        """
        断开调试协议TCP端口
        """
        try:
            if self.debug_sock is not None:
                self.debug_sock.close()
                self.debug_sock = None
            self.is_debug_receiving = False
            messagebox.showinfo("调试协议TCP断开", "调试协议TCP已断开")
        except Exception as e:
            messagebox.showerror("调试协议TCP错误", str(e))

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
                    self.connect_debug_tcp()
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
                                    self.show_debug_recv_data("调试协议校验失败")
                            else:
                                buffer = buffer[start+2:]
            except Exception as e:
                self.is_debug_receiving = False
                messagebox.showerror("调试协议接收错误", str(e))
            time.sleep(0.1)

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
            for k, v in vars(data).items():
                self.debug_recv_text.insert(tk.END, f"{k}: {v}\n")

    # ========== 75字节用户协议串口/TCP相关 =========
    def toggle_recv_loop(self):
        """
        启动用户协议接收线程
        """
        if not self.is_receiving:
            self.is_receiving = True
            self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
            self.recv_thread.start()

    def stop_recv_loop(self):
        """
        停止用户协议接收线程
        """
        self.is_receiving = False

    def recv_loop(self):
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
                        data = self.ser.read(75)
                        if data and len(data) == 75:
                            self.show_recv_data(data)
                            self.show_user75_data(data)
                else:
                    if self.sock is None:
                        self.connect_tcp()
                    if self.sock is not None:
                        data = self.sock.recv(75)
                        if data and len(data) == 75:
                            self.show_recv_data(data)
                            self.show_user75_data(data)
            except Exception as e:
                self.is_receiving = False
                messagebox.showerror("用户协议接收错误", str(e))
            time.sleep(0.1)

    def show_recv_data(self, data):
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

    def generate_packet(self):
        """
        生成发送报文并显示HEX
        """
        try:
            packet = build_packet(
                self.auv_id.get(),
                COMMAND_TYPE[self.command_type.get()],
                CONTROL_TYPE[self.control_mode.get()],
                CONTROL_MODE[self.control_type.get()],
                [v.get() for v in self.motion_values],
                self.get_device_start_bytes(),
                [v.get() for v in self.led_values],
                self.gimbal_value.get()
            )
            self.packet_hex.set(packet.hex())
            return packet
        except Exception as e:
            messagebox.showerror("错误", str(e))
            return None

    def send_packet_once(self):
        """
        发送一次报文
        """
        packet = self.generate_packet()
        if packet is None:
            return
        mode = self.conn_mode.get()
        try:
            if mode == "串口":
                if self.ser is None or not self.ser.is_open:
                    self.connect_serial()
                if self.ser is not None and self.ser.is_open:
                    self.ser.write(packet)
                    messagebox.showinfo("发送成功", "报文已发送")
            else:
                if self.sock is None:
                    self.connect_tcp()
                if self.sock is not None:
                    self.sock.sendall(packet)
                    messagebox.showinfo("发送成功", "报文已发送")
        except Exception as e:
            messagebox.showerror(f"{mode}错误", str(e))

    def toggle_send_loop(self):
        """
        启动或停止循环发送线程
        """
        if not self.is_sending:
            self.is_sending = True
            self.send_btn.config(text="停止循环发送")
            self.send_thread = threading.Thread(target=self.send_loop, daemon=True)
            self.send_thread.start()
        else:
            self.is_sending = False
            self.send_btn.config(text="开始循环发送")

    def send_loop(self):
        """
        循环发送报文线程函数
        """
        mode = self.conn_mode.get()
        while self.is_sending:
            packet = self.generate_packet()
            if packet is not None:
                try:
                    if mode == "串口":
                        if self.ser is None or not self.ser.is_open:
                            self.connect_serial()
                        if self.ser is not None and self.ser.is_open:
                            self.ser.write(packet)
                    else:
                        if self.sock is None:
                            self.connect_tcp()
                        if self.sock is not None:
                            self.sock.sendall(packet)
                except Exception as e:
                    self.is_sending = False
                    self.send_btn.config(text="开始循环发送")
                    messagebox.showerror(f"{mode}错误", str(e))
                    break
            time.sleep(0.2)  # 5Hz

    def recv_thread_func(self):
        """
        接收线程函数，实时接收并显示数据
        """
        buffer = bytearray()
        while self.is_receiving:
            try:
                if self.conn_mode.get() == "串口":
                    if self.ser is None or not self.ser.is_open:
                        self.connect_serial()
                    if self.ser is not None and self.ser.is_open:
                        data = self.ser.read(75)
                        if data:
                            buffer += data
                            while len(buffer) >= 75:
                                self.show_recv_data(buffer[:75])
                                buffer = buffer[75:]
                else:
                    if self.sock is None:
                        self.connect_tcp()
                    if self.sock is not None:
                        data = self.sock.recv(75)
                        if data:
                            buffer += data
                            while len(buffer) >= 75:
                                self.show_recv_data(buffer[:75])
                                buffer = buffer[75:]
            except Exception as e:
                self.is_receiving = False
                messagebox.showerror("接收错误", str(e))
            time.sleep(0.1)

if __name__ == "__main__":
    # 启动主界面
    root = tk.Tk()
    SenderUI(root)
    root.mainloop()