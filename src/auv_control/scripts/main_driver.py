"""
main_driver.py
监听：auv_control/Control
发布：auv_control/AUVData

作者：黄思旭
版本：
v25.6.3.1
    # 接收AUV数据包，解析并发布AUV数据
    # 控制AUV运动数据
"""

"""
    接收协议 75字节
    字节      内容              备注
    0-1       报文头            FE FE
    2         上发模式选择      00为正常数据显示；01为透传数据显示；02为状态数据透传
    3         船只ID            船只编号
    4-6       数据包编号        大端在前,数据包编号，每发送一包数据，数据包编号自动加1
    7         运动控制模式      00 自控；01 键鼠控制；02 遥控
    8         任务类型          00 正常执行任务；01 巡航；02 虚拟锚定；03 一键返航；04 定深；05 定向；06 定点；
    9-10      AUV速度           大端在前，AUV前向速度，单位m/s，数据扩大100倍
    11-12     AUV深度           大端在前，AUV深度，单位m，数据扩大100倍
    13-14     AUV高度           大端在前，AUV高度，单位m，数据扩大100倍
    15-22     控制舱温度        大端在前，INT16类型，21-22字节有效，单位1℃，数据扩大100倍
    23-25     航线名称ID        不使用   
    26-33     姿态传感器数据    每三个字节为一组，每组第一个字节为符号位(00为正，01为负)，后两个字节为数据位，大端在前，数据扩大100倍
    35-42     电流电压测量值    大端在前，35-36为电压值，37-38为电流值，39为电池电量值，数据扩大100倍
    43-50     经纬度值          经纬度值，单位m，数据扩大10000000倍，大端在前
    51        符号位            0为正，1为负，第1位为经度符号位，第5位为纬度符号位
    52-57     数据发送时间      前三位为年月日yymmdd，后三位为时分秒hhmmss
    58-61     设备报警          四字节32位，暂不使用
    62        传感器有效位      从低到高依次是：AHRS、DVL、GPS、SBL、VIO、NAV、DS
    63        传感器更新位      从低到高依次是：AHRS、DVL、GPS、SBL、VIO
    64-65     当前电源状态      大端在前，0为关闭，1为开启，从最低位开始，分别对应惯导、DVL、USBL、声通信机、高度计、深度计、侧扫声呐、前视声呐、单波束、多波束、避碰声呐、频闪灯、ADCP、
                                CTD、浅剖
    66-70     预留位            不使用
    71        当前设备运行模式  00 定深；01 ROV模式；02非动力定位的AUV模式；03动力定位ROV模式；04 DPAUV模式
    72        校验和            0~71字节异或校验
    73-74     数据尾            FD FD
"""

"""
    发送协议 54字节 ROV扩展指令
    字节    内容            备注
    0-1     报文头          FE FE
    2-3     AUV编号         01
    4       指令类型        30 ROV扩展指令
    5       运动控制模式    00 自控； 01 键鼠控制； 02 遥控；
    6       控制类型        当指令类型为“运动控制”且“控制模式”为键鼠控制或者遥控时该位为手控类型开环或者闭环处理字节，01—为开环、00—闭环、02—扩展模式
                            当指令类型为“自控类型设置”时，对该数据进行解析，00-定深、01—定向、02—AUV无动力定位、03—定点，04—AUV有动力定位
    7       坐标系设置      00 经度纬度坐标
    8-15    期望跟踪轨迹点设置 8-11字节为经度，12-15字节为纬度，单位m，数据扩大10000000倍，大端在前
    16-19   期望跟踪轨迹点速度设置 16-19字节为速度，单位m/s，数据扩大100倍，大端在前
    20-23   期望跟踪轨迹点加速度设置 20-23字节为加速度，单位m/s²，数据扩大100倍，大端在前
    24-27   期望跟踪轨迹点姿态设置 24-27字节为姿态，单位°，数据扩大100倍，大端在前
    28-31   期望跟踪轨迹点高度设置 28-31字节为高度，单位m，数据扩大100倍，大端在前
    32-43   期望跟踪轨迹点时间设置 32-43字节为时间，单位ms，数据扩大100倍，大端在前
    44      预留位            不使用
    45-50
    51      异或校验        0·50字节异或校验和
    52-53   数据尾          FD FD 
"""

"""
    AUVData.msg 消息定义
    std_msgs/Header header      # 头部信息
    uint8           control_mode    # 运行模式
    uint8           upload_mode # 上传模式
    uint32          packet_id    # 数据包编号
    uint8           task_type   # 任务类型
    AUVPose         pose        # AUV位姿
    AUVTime         time        # AUV时间
    AUVMotor        motor_force # AUV电机力矩
    AUVSensor       sensor      # AUV传感器数据
"""

"""
    AUVMotor.msg 消息定义
    int16 TX 
    int16 TY
    int16 TZ
    int16 MX 
    int16 MY
    int16 MZ
"""

"""
    AUVPose.msg 消息定义
    float64 latitude    # 纬度
    float64 longitude   # 经度
    float32 altitude    # 高度
    float32 depth       # 深度
    float64 roll        # 横滚角
    float64 pitch       # 俯仰角
    float64 yaw         # 偏航角
    float32 velocity    # 速度
    float32 east        # ENU坐标系下的东向坐标
    float32 north       # ENU坐标系下的北向坐标
    float32 up          # ENU坐标系下的上向坐标
    float32 vx          # X轴速度
    float32 vy          # Y轴速度
    float32 vz          # Z轴速度
"""

"""
    AUVSensor.msg 消息定义
    float32 temperature       # 舱内温度监测数据（°）
    float32 voltage           # 控制电电压（V）
    float32 current           # 系统电流（A）
    uint8 battery             # 电池电量（%）
    bool leak_alarm           # 漏水报警数据，00不漏水，01漏水
    uint8 sensor_valid        # 传感器状态（有效性），从低位到高位0代表无效，1代表有效
                                # 0 AHRS
                                # 1 GPS
                                # 2 SBL
                                # 3 VIO
                                # 4 DVL地速
                                # 5 DVL流速
                                # 6 DVL高度
    uint8 sensor_updated      # 传感器状态（更新状态），从低位到高位0代表不更新，1代表更新
                                # 0 AHRS
                                # 1 DVL
                                # 2 GPS
                                # 3 SBL
                                # 4 VIO
    uint16 fault_status       # 故障状态，从低位到高位0代表无故障，1代表有故障
                                # 0 闭环超深
                                # 1 开环超深
                                # 2 漏水报警
                                # 3 通讯故障
                                # 4 电池电压报警
                                # 5 传感器采集故障
                                # 6 动力系统故障
                                # 7 闭环运行超时
                                # 8 开环运行超时
    uint16 power_status       # 电源状态，16位对应16个设备的电源，0代表关闭，1代表打开
                                # 从最低位（最右边）开始。分别对应惯导、DVL、USBL、声通信机、高度计、深度计、侧扫声呐、
                                # 前视声呐、单波束、多波束、避碰声呐、频闪灯、ADCP、CTD、浅剖、PC机
"""

"""
    AUVTime.msg 消息定义
    uint8 year                  # UTC时间年
    uint8 month                 # UTC时间月
    uint8 day                   # UTC时间日
    uint8 hour                  # UTC时间时
    uint8 minute                # UTC时间分
    uint8 second              # UTC时间秒
"""


import socket
import struct
import time
import threading
import rospy
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from auv_control.msg import AUVData, Control  # 增加Control
from functools import reduce
from geometry_msgs.msg import Point  # 新增
# from navigation_utils import geodetic_to_enu_mm  # 已有



test_data = bytearray(75)
test_data[:] = bytes.fromhex(
    "FE FE 00 01 00 26 B6 01 00 00 00 01 94 00 00 0F A0 0F A0 0F A0 0F A0 00 00 " \
    "00 01 00 00 01 00 AE 01 08 27 04 E4 00 00 04 E4 00 00 48 C8 03 90 16 61 0A " \
    "76 00 02 EA 85 02 36 13 00 00 00 00 07 07 00 00 00 00 00 00 00 01 D6 FD FD" 
)
# 定义数据结构体
class DataPacket:
    def __init__(self):
        self.mode = 0                                                                               # 上发模式选择
        self.ship_id = 0                                                                            # 船只ID
        self.packet_id = 0                                                                          # 数据包编号
        self.control_mode = 0                                                                       # 运动控制模式
        self.task_type = 0                                                                          # 任务类型
        self.speed = 0.0                                                                            # AUV速度
        self.depth = 0.0                                                                            # AUV深度
        self.altitude = 0.0                                                                         # AUV高度
        self.temperature = 0.0                                                                      # 控制舱温度
        self.roll = 0.0                                                                             # 横滚角
        self.pitch = 0.0                                                                            # 俯仰角
        self.yaw = 0.0                                                                              # 偏航角
        self.voltage = 0.0                                                                          # 系统电压
        self.current = 0.0                                                                          # 系统电流
        self.battery = 0.0                                                                          # 电池电量
        self.longitude = 0.0                                                                        # 经度
        self.latitude = 0.0                                                                         # 纬度
        self.sign = 0                                                                               # 符号位
        self.sensor_status = 0                                                                      # 传感器有效位
        self.sensor_update = 0                                                                      # 传感器更新位
        self.fault_status = 0                                                                       # 故障状态
        self.power_status = 0                                                                       # 当前电源状态
        self.utc_time = [0] * 6                                                                     # 数据发送时间
        self.checksum = 0                                                                           # 校验和
        

class TCPHandler:
    def __init__(self, ip, port):
        self.server_address = (ip, port)
        self.buffer = bytearray()
        self.socket = None
        self.packet_pub = rospy.Publisher('/auv_data', AUVData, queue_size=10)
        # self.enu_pub = rospy.Publisher('/auv_enu', Point, queue_size=10)  # 新增
        rospy.init_node('driver_node', anonymous=True)
        rospy.loginfo("driver_node successfully started!")  # 打印节点初始化信息
        self.control_msg = None  # 保存最新的Control消息
        rospy.Subscriber('/auv_control', Control, self.control_callback)
        self.send_count = 0  # 发送包计数器

    def connect(self):
        # 连接到TCP服务器
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(self.server_address)
        self.socket.settimeout(1)

    def calculate_checksum(self, packet):
        # 计算异或校验和
        return reduce(lambda x, y: x ^ y, packet[:72], 0)

    def parse_packet(self, packet):
        # 解析数据包
        data = DataPacket()
        try:
            data.mode = packet[2] # 上发模式选择
            data.ship_id = packet[3] # 船只ID
            data.packet_id = struct.unpack('>I', b'\x00' + packet[4:7])[0] # 数据包编号
            data.control_mode = packet[7] # 运动控制模式
            data.task_type = packet[8] # 任务类型
            data.speed = struct.unpack('>H', packet[9:11])[0] / 100.0 # AUV速度
            data.depth = struct.unpack('>H', packet[11:13])[0] / 100.0 # AUV深度
            data.altitude = struct.unpack('>H', packet[13:15])[0] / 100.0 # AUV高度
            data.temperature = struct.unpack('>h', packet[21:23])[0] / 100.0 # 控制舱温度
            data.roll = self.parse_signed_sensor(packet[26:29]) # 横滚角
            data.pitch = self.parse_signed_sensor(packet[29:32]) # 俯仰角
            data.yaw = self.parse_signed_sensor(packet[32:35]) # 偏航角
            data.voltage = struct.unpack('>H', packet[35:37])[0] / 100.0 # 系统电压
            data.current = struct.unpack('>H', packet[37:39])[0] / 100.0 # 系统电流
            data.battery = packet[39] # 电池电量
            data.sign = packet[51]  # 符号位
            longitude_raw = struct.unpack('>i', packet[43:47])[0]
            latitude_raw = struct.unpack('>i', packet[47:51])[0]
            if data.sign & 0x01:  # 经度为负
                data.longitude = -longitude_raw / 10000000.0
            else:  # 经度为正
                data.longitude = longitude_raw / 10000000.0
            if data.sign & 0x10:  # 纬度为负
                data.latitude = -latitude_raw / 10000000.0
            else:  # 纬度为正
                data.latitude = latitude_raw / 10000000.0
            data.fault_status = struct.unpack('>I', packet[58:62])[0] # 故障状态
            data.sensor_status = packet[62] # 传感器有效位
            data.sensor_update = packet[63] # 传感器更新位
            data.power_status = struct.unpack('>H', packet[64:66])[0] # 当前电源状态
            data.utc_time[0] = int((packet[52] * 256 * 256 + packet[53] * 256 + packet[54]) // 10000)
            data.utc_time[1] = int(((packet[52] * 256 * 256 + packet[53] * 256 + packet[54]) % 10000) // 100)
            data.utc_time[2] = int((packet[52] * 256 * 256 + packet[53] * 256 + packet[54]) % 100)
            data.utc_time[3] = int((packet[55] * 256 * 256 + packet[56] * 256 + packet[57]) // 10000)
            data.utc_time[4] = int(((packet[55] * 256 * 256 + packet[56] * 256 + packet[57]) % 10000) // 100)
            data.utc_time[5] = int((packet[55] * 256 * 256 + packet[56] * 256 + packet[57]) % 100)
            data.checksum = packet[72]
        except Exception as e:
            rospy.logerr(f"Error parsing packet: {e}")
        return data

    def parse_signed_sensor(self, sensor_data):
        # 解析带符号的传感器数据
        sign = -1 if sensor_data[0] == 0x01 else 1
        value = struct.unpack('>H', sensor_data[1:3])[0]
        return sign * value / 100.0

    def read_and_parse(self):
        # 读取数据包并解析
        while not rospy.is_shutdown():
            try:
                # 接收数据包
                data = self.socket.recv(75)
                if len(data) == 75:
                    # 检查报文头和报文尾是否正确
                    if data[0:2] == b'\xFE\xFE' and data[73:75] == b'\xFD\xFD':
                        # 校验和验证
                        if self.calculate_checksum(data) == data[72]:
                            # 解析数据包
                            parsed_data = self.parse_packet(data)
                            # 发布解析后的数据
                            msg = AUVData()
                            msg.header.stamp = rospy.Time.now()
                            msg.control_mode = parsed_data.control_mode
                            msg.upload_mode = parsed_data.mode
                            msg.packet_id = parsed_data.packet_id
                            msg.task_type = parsed_data.task_type
                            msg.pose.latitude = parsed_data.latitude
                            msg.pose.longitude = parsed_data.longitude
                            msg.pose.altitude = parsed_data.altitude
                            msg.pose.depth = parsed_data.depth
                            msg.pose.velocity = parsed_data.speed   # 前向速度
                            msg.pose.roll = parsed_data.roll
                            msg.pose.pitch = parsed_data.pitch
                            msg.pose.yaw = parsed_data.yaw
                            # 新增：ENU坐标
                            # east_mm, north_mm = geodetic_to_enu_mm(parsed_data.latitude, parsed_data.longitude)
                            msg.pose.east = 0
                            msg.pose.north = 0
                            msg.pose.up = parsed_data.depth * 1000  # 机器人在水下，关注深度，而不关注高度
                            msg.sensor.temperature = parsed_data.temperature
                            msg.sensor.voltage = parsed_data.voltage
                            msg.sensor.current = parsed_data.current
                            msg.sensor.battery = parsed_data.battery
                            msg.sensor.sensor_valid = parsed_data.sensor_status
                            msg.sensor.sensor_updated = parsed_data.sensor_update
                            msg.sensor.power_status = parsed_data.power_status
                            msg.sensor.fault_status = parsed_data.fault_status
                            msg.time.year = parsed_data.utc_time[0]
                            msg.time.month = parsed_data.utc_time[1]
                            msg.time.day = parsed_data.utc_time[2]
                            msg.time.hour = parsed_data.utc_time[3]
                            msg.time.minute = parsed_data.utc_time[4]
                            msg.time.second = parsed_data.utc_time[5]
                            self.packet_pub.publish(msg)
                            # self.rec_num +=1
                            #rospy.loginfo(f"Received packet ID: {parsed_data.packet_id}, Speed: {parsed_data.speed}, Depth: {parsed_data.depth}, Altitude: {parsed_data.altitude}")
                            # 新增：发布ENU坐标
                            # east_mm, north_mm = geodetic_to_enu_mm(parsed_data.latitude, parsed_data.longitude)
                            # enu_msg = Point()
                            # enu_msg.x = east_mm
                            # enu_msg.y = north_mm
                            # enu_msg.z = parsed_data.altitude  # 可选，或设为0
                            # self.enu_pub.publish(enu_msg)
                        else:
                            rospy.logwarn("校验和不匹配")
            except KeyboardInterrupt:
                rospy.loginfo("监听已停止")
                break
            except Exception as e: # 处理其他异常
                rospy.loginfo(f"Error reading packet: {e}")

    def control_callback(self, msg):
        # 保存最新的Control消息，并立即发送控制数据包
        self.control_msg = msg
        try:
            if not self.control_msg.enable: # 当前帧不是外设控制指令，是普通的扩展控制指令
                control_packet = self.build_control_packet()
                rospy.loginfo("Send packet: " + " ".join("{:02X}".format(b) for b in control_packet))
                self.socket.sendall(control_packet)
                self.send_count += 1  # 发送包计数
        except Exception as e:
            rospy.logerr(f"Error sending control packet: {e}")

    def build_expect_packet(self):
        packet = bytearray(54) # 54 字节数据包
        # 0 - 1: 报文头 FE FE
        # 2 - 3: AUV编号 01,先传高位，后传低位
        # 4: 对应模式 30 ROV扩展指令
        # 5 - 50： 开环闭环与扩展模式，根据第4字节的不同，协议不一样
        # 5：设备运行模式 02 定深 03 定深定向 04 动力定位ROV模式
        # 6: 开环闭环与扩展模式 01 开环 00 闭环
        # 7：坐标系设置 00 采用经纬度坐标
        # 8 - 15： 期望跟踪轨迹点设置 原始数据扩大10000000倍进行强制转换。int32型
        # 16 - 19： 期望深度设置 4byte float型，单位m
        # 20 - 23： 期望横滚角 4byte float型，单位°
        # 24 - 27： 期望俯仰角 4byte float型，单位°
        # 28 - 31： 期望偏航角 4byte float型，单位°
        # 32 - 43： 设置6个自由度上的力，开环或ROV模式下向机器人发力和力矩，对闭环的自由度无效，只对开环的环路有效
        # 44： 是否打开模式 00 跟踪以上设置状态不理会上位机的控制，使用扩展模式时此位需要开启 01 退出以上跟踪状态，控制权交还给上位机
        # 45 - 50： 预留 00
        # 51： 校验和 0-50字节异或校验和
        # 52- 53： 数据尾 FD FD
        # 设置报文头
        packet[0:2] = b'\xFE\xFE'
        # AUV编号，假设为0x00 0x01
        packet[2:4] = b'\x00\x01'
        # 指令类型: 0x30 表示ROV扩展指令
        packet[4] = 0x30

        # 设备运行模式
        if self.control_msg is not None:
            packet[5] = 0x02  # 02定深, 03定深定向, 04动力定位ROV模式
            # 开环闭环与扩展模式
            packet[6] = 0x01  # 01开环, 00闭环
            # 坐标系设置
            packet[7] = 0x00  # 00经纬度
            # 期望轨迹点
            lon = int(self.control_msg.pose.longitude * 1e7)
            lat = int(self.control_msg.pose.latitude * 1e7)
            packet[8:12] = struct.pack('>i', lon)
            packet[12:16] = struct.pack('>i', lat)
            # 期望深度
            packet[16:20] = struct.pack('>f', self.control_msg.pose.depth)
            # 期望横滚角
            packet[20:24] = struct.pack('>f', self.control_msg.pose.roll)
            # 期望俯仰角
            packet[24:28] = struct.pack('>f', self.control_msg.pose.pitch)
            # 期望偏航角
            packet[28:32] = struct.pack('>f', self.control_msg.pose.yaw)
            # 6自由度力/力矩
            # 开环或ROV模式下向机器人发力和力矩，对闭环的自由度无效，只对开环的环路有效
            # dof = [
            # self.control_msg.force.TX,
            # self.control_msg.force.TY,
            # self.control_msg.force.TZ,
            # self.control_msg.force.MX,
            # self.control_msg.force.MY,
            # self.control_msg.force.MZ
            # ]
            # for i in range(6):
            # val = 100 + int(dof[i])
            # val = max(0, min(200, val))
            # packet[32 + i] = val
            # 是否打开模式
            packet[44] = 00  # 00跟踪, 01退出
            # 预留
            for i in range(45, 51):
                packet[i] = 0x00

            # 校验和
            xor = 0
            for i in range(0, 51):
                xor ^= packet[i]
            packet[51] = xor
            # 数据尾
            packet[52:54] = b'\xFD\xFD'
        return packet
      
    def build_control_packet(self): # 已弃用
        """
        构建控制数据包，根据self.control_msg内容
        """
        packet = bytearray(54)
        packet[0:2] = b'\xFE\xFE'
        packet[2:4] = b'\x00\x01'  # AUV编号
        packet[4] = 0x00  # 指令类型，根据运动控制模式解析
        packet[5] = 0x01  # 运动控制模式 键鼠控制
        # 控制类型动态设置
        if self.send_count < 5:
            packet[6] = 0x01
        elif self.send_count < 10:
            packet[6] = 0x00
        else:
            packet[6] = 0x02
        for i in range(7, 33):
            packet[i] = 0x00
        packet[32]=0xff  # 预留位
        # 默认LED和舵机
        led_red = 0
        led_green = 0
        servo = 0
        tx = ty = tz = mx = my = mz = 0

        if self.control_msg is not None:
            led_red = self.control_msg.led_red
            led_green = self.control_msg.led_green
            servo = self.control_msg.servo
            # 力和力矩
            tx = self.control_msg.force.TX
            ty = self.control_msg.force.TY
            tz = self.control_msg.force.TZ
            mx = self.control_msg.force.MX
            my = self.control_msg.force.MY
            mz = self.control_msg.force.MZ

        # 33-38: 6自由度，100+实际值，范围0~200
        dof = [tx, ty, tz, mx, my, mz]
        for i in range(6):
            val = 100 + int(dof[i])
            val = max(0, min(200, val))
            packet[33 + i] = val

        packet[41] = led_red
        packet[42] = led_green
        packet[46] = servo

        xor = 0
        for i in range(0, 51):
            xor ^= packet[i]
        packet[51] = xor
        packet[52:54] = b'\xFD\xFD'
        return packet

    def run(self):
        """
        主函数，连接服务器并启动数据接收和解析线程
        """
        
        try:
            self.connect()  # 尝试连接到TCP服务器
            rospy.loginfo("Connected to TCP server")  # 打印连接成功
            # 启动数据接收和解析线程
            recv_thread = threading.Thread(target=self.read_and_parse, daemon=True)
            recv_thread.start()
            # 不再启动控制数据包发送线程
            # 保持主线程运行，直到ROS关闭
            while not rospy.is_shutdown():
                time.sleep(0.1)
        except Exception as e:
            rospy.logerr(f"Connection failed: {e}.")  # 打印错误信息并重试
            time.sleep(2)

if __name__ == "__main__":
    try:
        handler = TCPHandler(ip="192.168.1.115", port=5062)
        handler.run()
    except rospy.ROSInterruptException:
        pass
