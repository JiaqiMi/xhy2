#! /home/xhy/xhy_env36/bin/python
"""
名称: debug_driver.py
功能: 连接调试串口发送54字节扩展报文
作者: buyegaid
监听: /auv_control (Control.msg)
发布: /debug_auv_data(AUVData.msg)
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
    TODO 在水下测试深度
"""

import rospy
import socket
import struct
import threading
import time
from auv_control.msg import AUVData, AUVPose
from functools import reduce

class target:
    # 用这个结构体来记录目标位置
    def __init__(self):
        self.valid = False
        self.longitude = 0.0
        self.latitude = 0.0
        self.depth = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.speed = 0.0


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
    # 110字节调试协议解析结构体
    def __init__(self):
        self.mode = 0                       # 当前运行模式
        self.temperature = 0.0              # 舱内温度监测数据
        self.control_voltage = 0.0          # 控制电压
        self.power_current = 0.0            # 总电流
        self.water_leak = 0                 # 漏水检测
        self.sensor_status = 0              # 传感器状态
        self.sensor_update = 0              # 传感器更新
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


class debugdriver:
    """
    调试串口驱动类
    """
    def __init__(self, ip=None, port=None):
        # 获取参数服务器的IP和端口，默认192.168.1.115:5063
        ip = ip or rospy.get_param("~debug_ip", "192.168.1.115")
        port = port or rospy.get_param("~debug_port", 5063)
        self.saving_enable = rospy.get_param("~save_data", True)  # 是否保存数据
        self.saving_path = rospy.get_param("~save_path", "/home/hsx/debug_data.csv")
        self.server_address = (ip, port)
        self.tcp_sock = None
        # 初始化接收缓冲区
        self.buffer = bytearray()
        self.latest_debug_data = None

        self.lock = threading.Lock()
        self.target = target()
        self.last_control_time = 0
        self.send_thread = None
        self.recv_thread = None

        # 初始化深度滤波器
        self.depth_lpf = LowPassFilter(alpha=0.2)  # 低通滤波器
        self.depth_ma = MovingAverageFilter(window_size=5)  # 移动平均滤波器
        rospy.loginfo("debug_driver: 数据保存 %s", self.saving_enable)
        # 打开数据保存文件
        if self.saving_enable:
            try:
                self.save_file = open(self.saving_path, 'w')
                # 修改CSV表头，增加滤波后的深度字段
                header = "pc_timestamp,mode,temperature,control_voltage,power_current,water_leak,"
                header += "sensor_status,sensor_update,fault_status,power_status,"
                header += "force_cmd1,force_cmd2,force_cmd3,force_cmd4,force_cmd5,force_cmd6,"
                header += "roll,pitch,yaw,"
                header += "angular_vel_x,angular_vel_y,angular_vel_z,"
                header += "linear_vel_x,linear_vel_y,linear_vel_z,"
                header += "longitude,latitude,depth_raw,depth_lpf,depth_ma,altitude,"  # 增加滤波深度
                header += "target_longitude,target_latitude,target_depth,"
                header += "target_roll,target_pitch,target_yaw,"
                header += "target_altitude,target_speed,"
                header += "utc_year,utc_month,utc_day,utc_hour,utc_minute,utc_second\n"
                self.save_file.write(header)
                rospy.loginfo(f"debug_driver: 数据将保存到 {self.saving_path}")
            except Exception as e:
                rospy.logerr(f"debug_driver: 打开保存文件失败: {e}")
                self.save_file = None

        rospy.Subscriber('/auv_control', AUVPose, self.control_callback)
        self.data_pub = rospy.Publisher('/debug_auv_data', AUVData, queue_size=10)
        self.rate = rospy.Rate(10)
        rospy.loginfo(f"debug_driver: 已启动")
        
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
            data.sensor_status = f"{packet[10]:08b}"[::-1][:7]
            data.sensor_update = f"{packet[11]:08b}"[::-1][:5]
            data.fault_status = f"{struct.unpack('>h', packet[12:14])[0]:016b}"[::-1][:9]
            data.power_status = struct.unpack('>h', packet[14:16])[0]
            data.force_commands = list(struct.unpack('>6h', packet[16:28]))
            data.euler_angles = [x / 100.0 for x in struct.unpack('>3h', packet[28:34])]
            data.angular_velocity = [x / 100.0 for x in struct.unpack('>3h', packet[34:40])]
            data.linear_velocity = [x / 100.0 for x in struct.unpack('>3h', packet[40:46])]
            data.navigation_coords = [x / 10000000.0 for x in struct.unpack('<2i', packet[46:54])]
            # 深度滤波处理
            raw_depth = struct.unpack('<f', packet[54:58])[0]
            data.depth = raw_depth  # 保存原始深度
            data.depth_filtered = self.depth_lpf.update(raw_depth)  # 低通滤波
            data.depth_ma = self.depth_ma.update(raw_depth)  # 移动平均滤波
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
            
            # 保存所有解析到的数据
            if self.saving_enable and self.save_file:
                try:
                    pc_time = time.time()
                    # 构造完整的CSV行，包含所有字段
                    csv_line = f"{pc_time:.6f},"  # PC时间戳
                    csv_line += f"{data.mode},"  # 运行模式
                    csv_line += f"{data.temperature:.2f},{data.control_voltage:.2f},{data.power_current:.2f},"  # 温度电压电流
                    csv_line += f"{data.water_leak},"  # 漏水检测
                    csv_line += f"{data.sensor_status},{data.sensor_update},{data.fault_status},{data.power_status},"  # 状态字
                    # 力和力矩
                    csv_line += ",".join(f"{x}" for x in data.force_commands) + ","
                    # 欧拉角
                    csv_line += f"{data.euler_angles[0]:.2f},{data.euler_angles[1]:.2f},{data.euler_angles[2]:.2f},"
                    # 角速度
                    csv_line += ",".join(f"{x:.2f}" for x in data.angular_velocity) + ","
                    # 线速度
                    csv_line += ",".join(f"{x:.2f}" for x in data.linear_velocity) + ","
                    # 导航坐标和深度高度（增加滤波深度）
                    csv_line += f"{data.navigation_coords[0]:.7f},{data.navigation_coords[1]:.7f},"
                    csv_line += f"{data.depth:.3f},{data.depth_filtered:.3f},{data.depth_ma:.3f},{data.altitude:.3f},"
                    # 目标位置和姿态
                    csv_line += f"{data.target_longitude:.7f},{data.target_latitude:.7f},{data.target_depth:.3f},"
                    csv_line += f"{data.target_roll:.2f},{data.target_pitch:.2f},{data.target_yaw:.2f},"
                    csv_line += f"{data.target_altitude:.3f},{data.target_speed:.2f},"
                    # UTC时间
                    csv_line += ",".join(f"{x}" for x in data.utc_time) + "\n"
                    
                    self.save_file.write(csv_line)
                    self.save_file.flush()  # 立即写入文件
                except Exception as e:
                    rospy.logerr(f"debug_driver: 保存数据失败: {e}")
                    
        except Exception as e:
            rospy.logerr(f"debug_driver: 数据解析错误: {e}")
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
        # rospy.loginfo(f"{lon},{lat}")
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

    def connect(self):
        while not rospy.is_shutdown():
            try:
                self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_sock.connect(self.server_address)
                self.tcp_sock.settimeout(1)
                rospy.loginfo(f"debug driver: TCP连接成功 {self.server_address}")
                return
            except Exception as e:
                rospy.logwarn(f"debug driver: TCP连接失败 {self.server_address}: {e}, 2秒后重试...")
                rospy.sleep(2)

    def recv_loop(self):
        # 接收循环，子线程
        while not rospy.is_shutdown():
            try:
                data = self.tcp_sock.recv(512)
                if data:
                    self.buffer += data
                    while len(self.buffer) >= 110:
                        start = self.buffer.find(b'\xFE\xEF')
                        if start == -1 or len(self.buffer) - start < 110:
                            break
                        if self.buffer[start+108:start+110] == b'\xFA\xAF':
                            packet = self.buffer[start:start+110]
                            self.buffer = self.buffer[start+110:]
                            if self.calc_debug_checksum(packet) == packet[107]:
                                parsed = self.parse_debug_packet(packet)
                                with self.lock:
                                    self.latest_debug_data = parsed
                                msg = AUVData()
                                msg.header.stamp = rospy.Time.now()
                                msg.control_mode = parsed.mode
                                msg.pose.latitude = parsed.navigation_coords[1]
                                msg.pose.longitude = parsed.navigation_coords[0]
                                msg.pose.depth = parsed.depth_filtered  # 使用低通滤波后的深度
                                msg.pose.altitude = parsed.altitude
                                msg.pose.roll = parsed.euler_angles[0]
                                msg.pose.pitch = parsed.euler_angles[1]
                                msg.pose.yaw = parsed.euler_angles[2]
                                msg.pose.speed = parsed.linear_velocity[0]
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
                                # rospy.loginfo_throttle(2, f"DebugData published: {msg}")
                            else:
                                rospy.logwarn("debug driver: 校验和错误")
                        else:
                            self.buffer = self.buffer[start+2:]
            except Exception as e:
                rospy.logwarn(f"debug driver: TCP连接错误: {e}, 重连中...")
                try:
                    if self.tcp_sock:
                        self.tcp_sock.close()
                except Exception:
                    pass
                self.connect()

    def control_callback(self, msg:AUVPose):
        """
        收到Control消息时的回调函数,将消息存储到target结构体当中
        """
        try:
            # 更新target结构体
            self.target.longitude = getattr(msg, 'longitude', 0.0)
            self.target.latitude = getattr(msg, 'latitude', 0.0)
            self.target.depth = getattr(msg, 'depth', 0.0)
            self.target.roll = getattr(msg, 'roll', 0.0)
            self.target.pitch = getattr(msg, 'pitch', 0.0)
            self.target.yaw = getattr(msg, 'yaw', 0.0)
            self.target.speed = getattr(msg, 'speed', 0.0)
            self.target.valid = True
            self.last_control_time = time.time()
            rospy.loginfo_throttle(2, "debug_driver: 接收到控制消息")
        except Exception as e:
            rospy.logerr(f"debug_driver: 控制消息接收错误: {e}")

    def send_loop(self):
        # 发送循环，子线程
        while not rospy.is_shutdown():
            now = time.time()
            # 超过5秒未收到有效Control消息，停止发送
            if self.target.valid and (now - self.last_control_time > 5):
                self.target.valid = False
                rospy.loginfo("debug_driver: 5s未收到控制消息，停止发送！")

            if self.target.valid:
                # 构造虚拟Control.msg用于build_54_packet_from_control
                try:
                    packet = self.build_54_packet_from_control()
                    self.tcp_sock.sendall(packet)
                    rospy.loginfo_throttle(2, f"debug driver: 发送扩展控制指令{self.target.longitude}, {self.target.latitude}, {self.target.depth}, {self.target.roll}, {self.target.pitch}, {self.target.yaw}, {self.target.speed}")
                except Exception as e:
                    rospy.logerr(f"debug driver: 发送扩展指令包错误: {e}")
            time.sleep(0.2)  # 5Hz

    def run(self):
        # 主线程，主循环
        while not rospy.is_shutdown():
            try:
                if not self.tcp_sock:
                    self.connect()
                    time.sleep(2)
                    continue
                
                if not self.recv_thread or not self.recv_thread.is_alive():
                    rospy.loginfo("debug driver: 启动接收线程")
                    self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
                    self.recv_thread.start()
                
                if not self.send_thread or not self.send_thread.is_alive():
                    rospy.loginfo("debug driver: 启动发送线程")
                    self.send_thread = threading.Thread(target=self.send_loop, daemon=True)
                    self.send_thread.start()
                
                time.sleep(0.01)  # 100Hz，足够了
                
            except Exception as e:
                rospy.logerr(f"debug driver: 运行错误: {e}")
                # 关闭现有连接
                if self.tcp_sock:
                    try:
                        self.tcp_sock.close()
                    except:
                        pass
                    self.tcp_sock = None
                time.sleep(2)

        # 关闭接收和发送线程
        if self.recv_thread and self.recv_thread.is_alive():
            rospy.loginfo("debug_driver: 正在关闭接收线程")
            self.recv_thread.join(timeout=1)
        if self.send_thread and self.send_thread.is_alive():
            rospy.loginfo("debug_driver: 正在关闭发送线程")
            self.send_thread.join(timeout=1)
        rospy.signal_shutdown("debug_driver: 节点已关闭")

        # 关闭文件
        if self.saving_enable and self.save_file:
            try:
                self.save_file.close()
                rospy.loginfo("debug_driver: 数据文件已保存并关闭")
            except Exception as e:
                rospy.logerr(f"debug_driver: 关闭数据文件失败: {e}")

if __name__ == "__main__":
    try:
        rospy.init_node('debug_driver', anonymous=True)
        handler = debugdriver()
        handler.run()
    except rospy.ROSInterruptException:
        pass
