import socket
import struct
import threading
import time
import rospy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from auv_control.msg import Control, AUVData  # 增加AUVData
from functools import reduce

# 110字节调试协议结构体
class DebugDataPacket:
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

def calc_debug_checksum(packet):
    return reduce(lambda x, y: x ^ y, packet[:107], 0)

def parse_debug_packet(packet):
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
        # utc_time: 90-94为年/月/日/时/分，95-98为float秒
        data.utc_time = list(packet[90:95])  # 5字节
        data.utc_time.append(struct.unpack('<f', packet[95:99])[0])  # 秒为float
        data.checksum = packet[107]
    except Exception as e:
        rospy.logerr(f"Debug packet parse error: {e}")
    return data

def build_54_packet_from_control(msg):
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
    lon = int(getattr(msg.pose, 'longitude', 0) * 1e7)
    packet[8:12] = struct.pack('>i', lon)
    # 12-15: 期望纬度，扩大1e7
    lat = int(getattr(msg.pose, 'latitude', 0) * 1e7)
    packet[12:16] = struct.pack('>i', lat)
    # 16-19: 期望深度 float32
    packet[16:20] = struct.pack('>f', getattr(msg.pose, 'depth', 0.0))
    # 20-23: 期望横滚角 float32
    packet[20:24] = struct.pack('>f', getattr(msg.pose, 'roll', 0.0))
    # 24-27: 期望俯仰角 float32
    packet[24:28] = struct.pack('>f', getattr(msg.pose, 'pitch', 0.0))
    # 28-31: 期望偏航角 float32
    packet[28:32] = struct.pack('>f', getattr(msg.pose, 'yaw', 0.0))
    # 32-43: 6自由度力/力矩，全部填0
    for i in range(32, 44):
        packet[i] = 0x00
    # 44: 是否打开模式，00跟踪
    packet[44] = 0x00
    # 45-50: 预留
    for i in range(45, 51):
        packet[i] = 0x00
    # 51: 校验和
    xor = 0
    for i in range(0, 51):
        xor ^= packet[i]
    packet[51] = xor
    # 52-53: 数据尾 FD FD
    packet[52:54] = b'\xFD\xFD'
    return packet

class DebugTCPHandler:
    def __init__(self, ip=None, port=None):
        rospy.init_node('debug_driver', anonymous=True)
        ip = ip or rospy.get_param("~debug_ip", "192.168.1.115")
        port = port or rospy.get_param("~debug_port", 5063)
        self.server_address = (ip, port)
        self.tcp_sock = None
        self.buffer = bytearray()
        self.latest_debug_data = None
        self.lock = threading.Lock()
        rospy.Subscriber('/auv_control', Control, self.control_callback)
        self.data_pub = rospy.Publisher('/debug_auv_data', AUVData, queue_size=10)
        # self.running = True

    def connect(self):
        while not rospy.is_shutdown():
            try:
                self.tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.tcp_sock.connect(self.server_address)
                self.tcp_sock.settimeout(1)
                rospy.loginfo(f"Connected to debug TCP server {self.server_address}")
                return
            except Exception as e:
                rospy.logwarn(f"Failed to connect to debug TCP {self.server_address}: {e}, retrying in 2s...")
                time.sleep(2)

    def recv_loop(self):
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
                            if calc_debug_checksum(packet) == packet[107]:
                                parsed = parse_debug_packet(packet)
                                with self.lock:
                                    self.latest_debug_data = parsed
                                msg = AUVData()
                                msg.header.stamp = rospy.Time.now()
                                msg.control_mode = parsed.mode
                                msg.pose.latitude = parsed.navigation_coords[1]
                                msg.pose.longitude = parsed.navigation_coords[0]
                                msg.pose.depth = parsed.depth
                                msg.pose.altitude = parsed.altitude
                                msg.pose.roll = parsed.euler_angles[0]
                                msg.pose.pitch = parsed.euler_angles[1]
                                msg.pose.yaw = parsed.euler_angles[2]
                                msg.pose.velocity = parsed.linear_velocity[0]
                                msg.pose.east = 0.0
                                msg.pose.north = 0.0
                                msg.pose.up = 0.0
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
                                rospy.loginfo_throttle(2, f"DebugData published: {msg}")
                            else:
                                rospy.logwarn("Debug packet checksum mismatch")
                        else:
                            self.buffer = self.buffer[start+2:]
            except Exception as e:
                rospy.logwarn(f"TCP receive error: {e}, reconnecting...")
                try:
                    if self.tcp_sock:
                        self.tcp_sock.close()
                except Exception:
                    pass
                self.connect()

    def control_callback(self, msg):
        try:
            packet = build_54_packet_from_control(msg)
            self.tcp_sock.sendall(packet)
            rospy.loginfo("Sent 54-byte control packet")
        except Exception as e:
            rospy.logerr(f"Send control packet error: {e}")

    def run(self):
        # while not rospy.is_shutdown():
        try:
            self.connect()
            rospy.loginfo("Connected to TCP server")  # 打印连接成功
            recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
            recv_thread.start()
            while not rospy.is_shutdown():
                time.sleep(0.01)
        except Exception as e:
            rospy.logerr(f"DebugTCPHandler main loop error: {e}")
            time.sleep(2)

if __name__ == "__main__":
    try:
        handler = DebugTCPHandler()
        handler.run()
    except rospy.ROSInterruptException:
        pass
