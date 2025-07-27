#!/usr/bin/env python3
"""
这个节点当做main节点
订阅：keyboard，task
发布：control消息,vision mode 
"""
import rospy
from std_msgs.msg import String
from auv_control.msg import Control,Key,AUVData, AUVMotor, AUVPose
import subprocess
"""
    # 该文件定义了AUV控制消息的格式
    bool enable # 代表该帧是否是外设控制
    uint8 led_red # 红色LED开关
    uint8 led_green # 绿色LED开关
    uint8 servo # 舵机角度
    AUVMotor force # AUV的力数据
    AUVPose pose # AUV的位姿数据
"""
TASKS = [
    ['rosrun', 'auv_control', 'task1_node.py'],
    ['rosrun', 'auv_control', 'task2_node.py'],
    ['rosrun', 'auv_control', 'task3_node.py'],
    ['rosrun', 'auv_control', 'task4_node.py'],
]


class auv: # 用于存储auv的状态信息
    def __init__(self):
        self.east = 0.0            # 东向坐标，单位：mm
        self.north = 0.0            # 北向坐标，单位：mm
        self.up = 0.0            # 深度坐标，单位：mm
        self.yaw = 0.0          # 航向角，单位：度
        self.pitch = 0.0        # 俯仰角，单位：度
        self.roll = 0.0         # 横滚角，单位：度
        self.velocity =0.0           # 速度，单位：mm/s
        self.latitude = 0.0          # 纬度，单位：度
        self.longitude = 0.0          # 经度，单位：度
        self.depth = 0.0        # 深度，单位：m
        self.altitude = 0.0          # 高度，单位：m


class StateControl:
    def __init__(self):
        rospy.Subscriber('finished', String, self.finished_callback)
        rospy.Subscriber('/auv_keyboard', Key, self.keyboard_callback)  # 订阅键盘控制消息
        rospy.Subscriber('/auv_data', AUVData,self.driver_callback) # 订阅AUV数据消息
        self.control_pub = rospy.Publisher('/auv_control', Control, queue_size=10) # 发布控制消息
        self.last_key_msg = Key()  # 初始化最新的键盘消息
        self.last_key_time = rospy.get_time()  # 新增：记录时间
        self.current_task = 0
        self.process = None
        self.auv = auv() # 存储当前的auv位姿信息 
        # self.start_next_task()

    def driver_callback(self, msg):
        """
        处理导航节点消息
        更新当前AUV状态信息
        """
        # self.auv = auv()
        self.auv.east = msg.pose.east
        self.auv.north = msg.pose.north
        self.auv.up = msg.pose.up
        self.auv.yaw = msg.pose.yaw
        self.auv.pitch = msg.pose.pitch
        self.auv.roll = msg.pose.roll
        self.auv.velocity = msg.pose.velocity
        self.auv.latitude = msg.pose.latitude
        self.auv.longitude = msg.pose.longitude
        self.auv.depth = msg.pose.depth
        self.auv.altitude = msg.pose.altitude
        # rospy.loginfo(f"当前AUV状态: {self.auv}")

        
    def keyboard_callback(self, msg):
        """
        处理键盘控制消息
        根据msg.run和msg.mode字段来决定当前的运行模式
        msg.run: bool, 是否开启自动运行
        msg.mode: uint8, 运行阶段
        0: 待机
        1: task1
        2: task2
        3: task3
        4: task4
        5: task5
        g:开始自动运行
        """
        if msg.run: # 收到自动运行指令，直接开始运行，并不再理会其他指令
            if self.current_task != 0:  # 如果当前不是待机状态
                rospy.logwarn("当前不是待机状态，无法切换到自动运行模式")
                return
            else:  # 当前是待机状态
                rospy.loginfo("切换到自动运行模式")
                self.current_task = 1 # TODO 假设1是自动运行的初始模式
            rospy.loginfo("收到自动运行指令")
            return
        # 如果不是自动运行指令，那么就是手动控制指令，保存最新的键盘消息
        
        self.last_key_msg = msg  # 保存最新Key消息
        self.last_key_time = rospy.get_time()  # 新增：记录时间


    def start_next_task(self):
        if self.current_task < len(TASKS):
            rospy.loginfo(f"Starting task {self.current_task + 1}")
            self.process = subprocess.Popen(TASKS[self.current_task])
            self.current_task += 1
        else:
            rospy.loginfo("All tasks completed.")
            rospy.signal_shutdown("All tasks done.")

    def finished_callback(self, msg):
        rospy.loginfo(f"Received finished signal: {msg.data}")
        if self.process:
            self.process.terminate()
            self.process.wait()
        self.start_next_task()

    def run(self):
        rate = rospy.Rate(1000)  # 1000Hz
        control_pub_count = 0 # 发布计数器，发布频率是5Hz，因此每200次循环发布一次
        while not rospy.is_shutdown():
            control_pub_count += 1 # 计数器
            
            # 检查keyboard消息超时
            if self.last_key_msg is not None and self.last_key_time is not None:
                if rospy.get_time() - self.last_key_time > 2.0:
                    self.last_key_msg = None
                    self.last_key_time = None
                    rospy.loginfo("2秒未收到keyboard消息，已清空last_key_msg")
            
            
            # 1000Hz/5Hz=200，5Hz的频率发送控制消息
            if control_pub_count >= 200: 
                control_msg = Control()
                control_msg = self.last_key_msg.control  # 使用最新的键盘消息
                self.control_pub.publish(control_msg)
                control_pub_count = 0
                rospy.loginfo(f"发布控制消息")
            # rospy.spin() 这是阻塞的，会一直等待消息到来，不要用
            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('state_control')
    try:
        state = StateControl()
        state.run()
    except rospy.ROSInterruptException:
            # rospy.logerr("ROS Interrupt Exception occurred.")
        pass
    # rospy.spin()