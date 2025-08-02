#! /home/xhy/xhy_env36/bin/python
"""
名称： state_control.py
功能： 状态控制节点
描述：
    1. 订阅键盘控制消息，处理状态切换
        - msg.run: bool, 是否开启自动运行
        - msg.mode: uint8, 运行阶段
            0: 待机
            1: task1        
            2: task2
            3: task3
            4: task4
            5: task5
        - g: 开始自动运行
        - l: 停止自动运行
    2. 订阅AUV数据消息，更新当前AUV状态信息
    3. 订阅任务完成消息，处理任务完成逻辑
作者：buyegaid
记录：
    2025.7.16 16:37
        1. 添加了状态控制节点，支持自动运行和手动切换状态
        2. 修改了Keyboard.msg，添加了run字段，表示是否开启自动运行
"""


import rospy
from std_msgs.msg import String
from auv_control.msg import Keyboard,AUVData
import subprocess

detect_mode= [1, 2, 3] # 对应: 形状 颜色洞 球

TASKS = [
    ['rosrun', 'auv_control', 'test1.py'],
    ['rosrun', 'auv_control', 'task1_node.py'], # 过门
    ['rosrun', 'auv_control', 'task2_node.py'], # 钻洞
    ['rosrun', 'auv_control', 'task3_node.py'], # 抓球
    ['rosrun', 'auv_control', 'task4_node.py'], # 巡线
]
VISION= [
    [],
    [], # 过门
    ['rosrun', 'stereo', 'vision_task2.py'], # 钻洞
    [], # 抓球
    [], # 巡线
]

class StateControl:
    def __init__(self):
        rospy.Subscriber('/finished', String, self.finished_callback)
        rospy.Subscriber('/auv_keyboard', Keyboard, self.keyboard_callback)  # 订阅键盘控制消息
        self.last_key_msg = Keyboard()  # 初始化最新的键盘消息
        self.last_key_time = None  # 新增：记录时间
        self.current_task = 0
        self.task_process = None
        self.vision_process = None  # 视觉任务进程
        rospy.loginfo("state control: 已启动")
        
    def keyboard_callback(self, msg):
        """
        只处理状态切换相关消息
            run = 1时，不关注mode是多少
            run = 0时，mode发挥作用
        msg.run: bool, 是否开启自动运行
        msg.mode: uint8, 运行阶段
            0: 待机
            1: task1
            2: task2
            3: task3
            4: task4
            5: task5
            g:开始自动运行
            l:停止自动运行

        """
        rospy.loginfo(f"state control: get run {msg.run}, mode {msg.mode}")
        if msg.run == 1: # 收到自动运行指令，直接开始运行，并不再理会其他指令
            if self.current_task != 0:  # 如果当前不是待机状态
                rospy.logwarn("state control: 当前不是待机状态，无法切换到自动运行模式")
                return
            else:  # 当前是待机状态
                rospy.loginfo("state control: 切换到自动运行模式")
                self.start_next_task()
            rospy.loginfo("state control: 收到自动运行指令")
            return
        if msg.run == 2: # 结束自动运行，并将状态变为待机
            rospy.loginfo("state control: 收到结束自动运行指令")
            self.current_task = 0
            if self.task_process:
                self.task_process.terminate()
                self.task_process.wait()
                self.task_process = None # 关闭任务
            if self.vision_process:
                self.vision_process.terminate()
                self.vision_process.wait()
                self.vision_process = None # 关闭视觉
            return
        # 手动切换状态
        if msg.mode != self.current_task:
            rospy.loginfo(f"state control: 手动切换状态: {self.current_task} -> {msg.mode}")
            self.current_task = msg.mode
            self.start_task(self.current_task)          
            # 可在此处添加对应的任务启动逻辑

    def start_task(self, task_name:int):
        """
        启动指定任务
        :param task_name: 任务序号
        """
        if self.task_process or self.vision_process:
            rospy.logwarn("state control: 当前有任务正在运行，无法启动新任务")
            return
        try:
            self.task_process = subprocess.Popen(TASKS[task_name])
            self.vision_process = subprocess.Popen(VISION[task_name])
            rospy.loginfo(f"state control: 已启动任务: {TASKS[task_name]}")
        except Exception as e:
            rospy.logerr(f"state control: 启动任务失败: {e}")
            
    def start_next_task(self):
        self.current_task += 1
        if self.current_task <= len(TASKS):
            rospy.loginfo(f"state control: 开始执行任务 {self.current_task}")
            self.start_task(self.current_task-1)
        else:
            rospy.loginfo("state control: 所有任务已完成")
            rospy.signal_shutdown("所有任务已完成")

    def finished_callback(self, msg):
        rospy.loginfo(f"Received finished signal: {msg.data}")
        if msg.data == "finished":
            rospy.loginfo(f"state control: 任务 {self.current_task} 完成, 终止当前任务线程")
            if self.task_process:
                self.task_process.terminate()
                self.task_process.wait()
            if self.vision_process:
                self.vision_process.terminate()
                self.vision_process.wait()
            self.start_next_task()

    def run(self):
        rate = rospy.Rate(1000)  # 1000Hz
        while not rospy.is_shutdown():
            # 不再发布Control消息，只做状态管理
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