#! /home/xhy/xhy_env/bin/python
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
2025.8.5 15:20
    修复空指针bug，记录时间戳
"""


import rospy
from std_msgs.msg import String
from auv_control.msg import Keyboard
import subprocess
import time

NODE_NAME = "state_control"

detect_mode= [1, 2, 3] # 对应: 形状 颜色洞 球

task_name = ["过门", "巡线", "钻洞", "抓球", "上浮"]
TASKS = [
    ['rosrun', 'auv_control', 'task1_node.py'], # 过门
    ['rosrun', 'auv_control', 'task4_node.py'], # 巡线
    ['rosrun', 'auv_control', 'task2_node.py'], # 钻洞
    ['rosrun', 'auv_control', 'task3_node.py'], # 抓球
    ['rosrun', 'auv_control', 'task5_node.py'], # 上浮
]
VISION= [
    None,# 过门
    ['roslaunch', 'stereo_depth', 'find_line_and_shapes.launch'], # 巡线
    ['roslaunch', 'stereo_depth', 'find_holes.launch'], # 钻洞
    ['roslaunch', 'stereo_depth', 'find_balls.launch'], # 抓球
    None, # 上浮    
]

# 每个任务的超时时间（秒）
TASK_TIMEOUTS = [
    180,  # 过门 - 3分钟
    360,  # 巡线 - 6分钟
    270,  # 钻洞 - 4分半
    270,  # 抓球 - 4分半
    120,  # 上浮 - 2分钟
]

class StateControl:
    def __init__(self):
        # ros相关的初始化
        rospy.Subscriber('/finished', String, self.finished_callback)
        rospy.Subscriber('/auv_keyboard', Keyboard, self.keyboard_callback)  # 订阅键盘控制消息
        self.rate = rospy.Rate(10)  # 10Hz足够了，减少CPU占用

        # 变量定义
        self.current_task = 0 # 当前任务序号
        self.task_process = None # 任务进程
        self.vision_process = None  # 视觉任务进程
        self.task_start_time = None  # 任务开始时间
        self.auto_mode = False  # 是否为自动模式
        
        # 输出log
        rospy.loginfo(f"{NODE_NAME}: 初始化完成")

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
        rospy.loginfo(f"{NODE_NAME}: get run {msg.run}, mode {msg.mode}")
        if msg.run == 1: # 收到自动运行指令，直接开始运行，并不再理会其他指令
            if self.current_task != 0:  # 如果当前不是待机状态
                rospy.logwarn(f"{NODE_NAME}: 当前不是待机状态，无法切换到自动运行模式")
                return
            else:  # 当前是待机状态
                self.auto_mode = True  # 设置为自动模式
                self.start_next_task()
                rospy.loginfo(f"{NODE_NAME}: 切换到自动运行模式")
            rospy.loginfo(f"{NODE_NAME}: 收到自动运行指令")
            return
        if msg.run == 2: # 结束自动运行，并将状态变为待机  
            self.auto_mode = False  # 退出自动模式
            self.current_task = 0
            self.terminate_current_task()
            rospy.loginfo(f"{NODE_NAME}: 收到结束自动运行指令")
            return
        # 手动切换状态
        if msg.mode != self.current_task:      
            self.auto_mode = False  # 手动模式
            self.current_task = msg.mode
            self.start_task(self.current_task)      
            rospy.loginfo(f"{NODE_NAME}: 手动切换状态: {self.current_task} -> {msg.mode}")    
            # 可在此处添加对应的任务启动逻辑

    def terminate_current_task(self):
        """
        终止当前任务和视觉进程
        """
        if self.task_process:
            try:
                self.task_process.terminate()
                self.task_process.wait(timeout=5)  # 等待最多5秒
            except subprocess.TimeoutExpired:
                self.task_process.kill()  # 强制终止
                rospy.logwarn(f"{NODE_NAME}: 强制终止任务进程")
            finally:
                self.task_process = None
                
        if self.vision_process:
            try:
                self.vision_process.terminate()
                self.vision_process.wait(timeout=5)  # 等待最多5秒
            except subprocess.TimeoutExpired:
                self.vision_process.kill()  # 强制终止
                rospy.logwarn(f"{NODE_NAME}: 强制终止视觉进程")
            finally:
                self.vision_process = None
                
        self.task_start_time = None
        rospy.loginfo(f"{NODE_NAME}: 已终止当前任务和视觉进程")

    def start_task(self, task_index:int):
        """
        启动指定任务
        :param task_index: 任务索引 (0-4)
        """
        # 首先终止当前任务
        if self.task_process or self.vision_process:
            rospy.logwarn(f"{NODE_NAME}: 终止当前运行的任务")
            self.terminate_current_task()
            
        if task_index == 0:  # 待机状态
            rospy.loginfo(f"{NODE_NAME}: 切换到待机状态")
            return
            
        if task_index < 1 or task_index > len(TASKS):
            rospy.logerr(f"{NODE_NAME}: 无效的任务索引: {task_index}")
            return
            
        try:
            # 记录任务开始时间
            self.task_start_time = time.time()
            # 启动任务进程
            self.task_process = subprocess.Popen(TASKS[task_index-1])
            rospy.loginfo(f"{NODE_NAME}: 已启动任务: {task_name[task_index-1]} - {TASKS[task_index-1]}")
            
            # 启动视觉进程（如果需要）
            if VISION[task_index-1] is not None and len(VISION[task_index-1]) > 0:
                self.vision_process = subprocess.Popen(VISION[task_index-1])
                rospy.loginfo(f"{NODE_NAME}: 已启动视觉: {VISION[task_index-1]}")
            
            
            
        except Exception as e:
            rospy.logerr(f"{NODE_NAME}: 启动任务失败: {e}")
            self.terminate_current_task()

    def start_next_task(self):
        """
        启动下一个任务
        """
        self.current_task += 1 # 任务序号+1
        if self.current_task <= len(TASKS):
            rospy.loginfo(f"{NODE_NAME}: 开始执行任务 {self.current_task}: {task_name[self.current_task-1]}")
            self.start_task(self.current_task)
        else:
            rospy.loginfo(f"{NODE_NAME}: 所有任务已完成")
            self.auto_mode = False
            self.current_task = 0
            rospy.signal_shutdown("所有任务已完成")
    
    def check_task_timeout(self):
        """
        检查当前任务是否超时
        """
        try:
            # 安全检查：确保所有必要的条件都满足
            if not self.auto_mode:
                return False
                
            if self.current_task == 0:
                return False
                
            if self.task_start_time is None:
                rospy.logwarn(f"{NODE_NAME}: 任务 {self.current_task} 没有开始时间记录")
                return False
                
            if self.current_task > len(TASK_TIMEOUTS):
                rospy.logerr(f"{NODE_NAME}: 任务索引 {self.current_task} 超出超时配置范围")
                return False
                
            # 计算已运行时间
            current_time = time.time()
            elapsed_time = current_time - self.task_start_time
            timeout = TASK_TIMEOUTS[self.current_task-1]
            
            # 检查是否超时
            if elapsed_time > timeout:
                rospy.logwarn(f"{NODE_NAME}: 任务 {self.current_task}({task_name[self.current_task-1]}) 超时 "
                             f"({elapsed_time:.1f}s > {timeout}s)，强制进入下一任务")
                return True
            
            # 每30秒输出一次剩余时间提醒
            if int(elapsed_time) % 30 == 0 and int(elapsed_time) > 0:
                remaining_time = timeout - elapsed_time
                rospy.loginfo(f"{NODE_NAME}: 任务 {self.current_task}({task_name[self.current_task-1]}) "
                             f"剩余时间: {remaining_time:.0f}s")
            
            return False
        except Exception as e:
            rospy.logerr(f"{NODE_NAME}: 检查任务超时出错: {e}")
            return False
    # def check_task_timeout(self):
    #     """
    #     检查当前任务是否超时
    #     """
    #     # rospy.loginfo_throttle(2,f"{NODE_NAME}: {self.task_start_time},{self.auto_mode},{self.current_task}")
    #     # 这个判断有问题
    #     # 如果没有在进行任务，就直接返回
    #     if not (self.auto_mode or (self.task_start_time is not None and self.current_task != 0)):
    #         #   不在自动运行模式或者开始时间没有或者当前任务是0，返回false
    #         return False
        
            
    #     current_time = time.time()
    #     elapsed_time = current_time - self.task_start_time
    #     rospy.loginfo_throttle(2, f"{NODE_NAME}:{elapsed_time},{current_time}")
    #     timeout = TASK_TIMEOUTS[self.current_task-1]
        
    #     if elapsed_time > timeout:
    #         rospy.logwarn(f"{NODE_NAME}: 任务 {self.current_task}({task_name[self.current_task-1]}) 超时 "
    #                      f"({elapsed_time:.1f}s > {timeout}s)，强制进入下一任务或退出")
    #         return True
        
    #     # 每30秒输出一次剩余时间提醒
    #     if int(elapsed_time) % 30 == 0 and int(elapsed_time) > 0:
    #         remaining_time = timeout - elapsed_time
    #         rospy.loginfo(f"{NODE_NAME}: 任务 {self.current_task}({task_name[self.current_task-1]}) "
    #                      f"剩余时间: {remaining_time:.0f}s")
        
    #     return False

    def finished_callback(self, msg):
        """任务完成回调"""
        rospy.loginfo(f"{NODE_NAME}: 收到完成信号: {msg.data}")
        # 包含finished
        if "finished" in msg.data:
            rospy.loginfo(f"{NODE_NAME}: 任务 {self.current_task}({task_name[self.current_task-1]}) 完成, 终止当前任务线程")
            self.terminate_current_task()
            
            # 只有在自动模式下才自动进行下一个任务
            if self.auto_mode:
                self.start_next_task()
            else:
                self.current_task = 0  # 手动模式下回到待机状态

    # def run(self):
    #     """主循环"""
    #     while not rospy.is_shutdown():
    #         # 检查任务超时
    #        #  rospy.loginfo("statein loop")
    #         if self.check_task_timeout():
    #             rospy.logwarn(f"{NODE_NAME}: 任务超时，强制进入下一任务")
    #             self.terminate_current_task()
    #             if self.auto_mode:
    #                 self.start_next_task()
    #             else:
    #                 self.current_task = 0  # 手动模式下回到待机状态
            
    #         # 不再发布Control消息，只做状态管理
    #         self.rate.sleep()
    def run(self):
        """主循环"""
        while not rospy.is_shutdown():
            try:
                # 检查任务超时
                if self.auto_mode and self.current_task > 0:  # 只在自动模式且有任务运行时检查超时
                    if self.check_task_timeout():
                        rospy.logwarn(f"{NODE_NAME}: 任务超时，强制进入下一任务")
                        self.terminate_current_task()
                        if self.auto_mode:  # 再次确认是否还在自动模式
                            self.start_next_task()
                            continue  # 跳过本次循环剩余部分，避免检查刚刚初始化的任务
                        else:
                            self.current_task = 0  # 手动模式下回到待机状态
            except Exception as e:
                rospy.logerr(f"{NODE_NAME}: 主循环出错: {e}")
            
            # 不再发布Control消息，只做状态管理
            self.rate.sleep()
            
if __name__ == "__main__":
    rospy.init_node(f'{NODE_NAME}',anonymous=True)
    try:
        state = StateControl()
        state.run()
    except rospy.ROSInterruptException:
            # rospy.logerr("ROS Interrupt Exception occurred.")
        pass
    # rospy.spin()