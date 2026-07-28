#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
名称：state_control.py
功能：启动任务 launch 并监控其运行状态、完成消息和超时
作者：buyegaid
监听：/auv_keyboard、/finished
发布：无
记录：
2025.7.16
    添加状态控制节点，支持自动运行和手动切换状态。
2025.8.5
    修复空指针问题，记录任务开始时间。
2026.7.28
    改为管理 task1、task2、task3 的 launch，仅监控运行状态和超时。
2026.7.28
    恢复 /finished 驱动的 task1、task2、task3 自动串联。
"""

import subprocess
import time

import rospy
from auv_control.msg import Keyboard
from std_msgs.msg import String


NODE_NAME = 'state_control'


class StateControl:
    """根据键盘消息管理唯一的任务 launch 进程。"""

    def __init__(self):
        self.keyboard_topic = rospy.get_param('~keyboard_topic', '/auv_keyboard')
        self.finished_topic = rospy.get_param('~finished_topic', '/finished')
        self.finished_keyword = rospy.get_param('~finished_keyword', 'finished')
        self.monitor_rate_hz = rospy.get_param('~monitor_rate_hz', 2.0)
        self.terminate_wait_seconds = rospy.get_param(
            '~terminate_wait_seconds', 5.0)
        self.tasks = self._load_tasks(rospy.get_param('~tasks', []))

        self.current_task = None
        self.task_process = None
        self.task_start_time = None
        self.auto_mode = False

        rospy.Subscriber(self.keyboard_topic, Keyboard, self.keyboard_callback)
        rospy.Subscriber(self.finished_topic, String, self.finished_callback)
        self.rate = rospy.Rate(self.monitor_rate_hz)
        rospy.loginfo('%s 初始化完成，可管理任务：%s。', NODE_NAME,
                      ', '.join(task['name'] for task in self.tasks.values()))

    @staticmethod
    def _load_tasks(task_configs):
        """校验并按键盘 mode 建立任务配置索引。"""
        tasks = {}
        for config in task_configs:
            try:
                mode = int(config['mode'])
                name = str(config['name'])
                launch = str(config['launch'])
                timeout_seconds = float(config['timeout_seconds'])
            except (KeyError, TypeError, ValueError) as error:
                rospy.logerr('%s 忽略无效任务配置 %s：%s', NODE_NAME, config, error)
                continue

            if mode < 1 or timeout_seconds <= 0:
                rospy.logerr('%s 忽略无效任务配置：%s', NODE_NAME, config)
                continue

            launch_args = [str(argument) for argument in
                           config.get('launch_args', [])]
            tasks[mode] = {
                'mode': mode,
                'name': name,
                'launch': launch,
                'launch_args': launch_args,
                'timeout_seconds': timeout_seconds,
            }
        return tasks

    def keyboard_callback(self, message):
        """响应启动指定任务或停止当前任务的键盘指令。"""
        rospy.loginfo('%s 收到键盘指令：run=%s，mode=%s。', NODE_NAME,
                      message.run, message.mode)
        if message.run == 2:
            self.auto_mode = False
            self.terminate_current_task('收到停止指令')
            return

        if message.run == 1:
            self.start_automatic_tasks()
            return

        if message.run != 0:
            rospy.logwarn('%s 忽略不支持的 run 值：%s。', NODE_NAME, message.run)
            return

        task = self.tasks.get(message.mode)
        if task is None:
            rospy.logwarn('%s 未配置 mode=%s 对应的任务。', NODE_NAME, message.mode)
            return
        self.start_task(task, automatic=False)

    def start_task(self, task, automatic=False):
        """停止旧任务后，以 roslaunch 启动指定任务。"""
        self.terminate_current_task('切换任务')
        command = ['roslaunch', 'auv_control', task['launch']] + task['launch_args']
        try:
            self.task_process = subprocess.Popen(command)
            self.current_task = task
            self.task_start_time = time.monotonic()
            self.auto_mode = automatic
            rospy.loginfo('%s 已启动 %s：%s。超时 %.0f 秒。', NODE_NAME,
                          task['name'], ' '.join(command), task['timeout_seconds'])
        except OSError as error:
            self.task_process = None
            self.current_task = None
            self.task_start_time = None
            self.auto_mode = False
            rospy.logerr('%s 启动 %s 失败：%s', NODE_NAME, task['name'], error)

    def start_automatic_tasks(self):
        """从最小 mode 的任务开始自动串联。"""
        if not self.tasks:
            rospy.logerr('%s 没有可用于自动串联的任务配置。', NODE_NAME)
            return
        if self.task_process is not None and self.task_process.poll() is None:
            rospy.logwarn('%s 当前已有任务运行，拒绝启动自动串联。', NODE_NAME)
            return

        first_mode = min(self.tasks)
        self.start_task(self.tasks[first_mode], automatic=True)

    def start_next_task(self):
        """在自动模式下启动当前任务的下一个已配置任务。"""
        if self.current_task is None:
            return

        current_mode = self.current_task['mode']
        next_modes = [mode for mode in sorted(self.tasks) if mode > current_mode]
        if not next_modes:
            rospy.loginfo('%s 自动串联任务已全部完成。', NODE_NAME)
            self.terminate_current_task('自动串联任务完成')
            self.auto_mode = False
            return
        self.start_task(self.tasks[next_modes[0]], automatic=True)

    def terminate_current_task(self, reason):
        """优雅停止当前 roslaunch，必要时强制结束。"""
        if self.task_process is None:
            return

        task_name = self.current_task['name'] if self.current_task else '未知任务'
        if self.task_process.poll() is None:
            rospy.loginfo('%s 正在停止 %s：%s。', NODE_NAME, task_name, reason)
            self.task_process.terminate()
            try:
                self.task_process.wait(timeout=self.terminate_wait_seconds)
            except subprocess.TimeoutExpired:
                rospy.logwarn('%s 未在 %.1f 秒内退出，强制结束 %s。', NODE_NAME,
                              self.terminate_wait_seconds, task_name)
                self.task_process.kill()
                self.task_process.wait()

        self.task_process = None
        self.current_task = None
        self.task_start_time = None

    def finished_callback(self, message):
        """收到任务完成消息后，在自动模式推进下一任务。"""
        if self.finished_keyword not in message.data:
            return
        if self.current_task is None:
            rospy.logwarn('%s 收到完成消息，但当前没有运行任务。', NODE_NAME)
            return

        task_name = self.current_task['name']
        automatic = self.auto_mode
        rospy.loginfo('%s 收到 %s 的完成消息：%s。', NODE_NAME, task_name,
                      message.data)
        if automatic:
            self.start_next_task()
        else:
            self.terminate_current_task('收到完成消息')

    def monitor_current_task(self):
        """检测 launch 退出或超过配置时限，并清理任务状态。"""
        if self.task_process is None or self.current_task is None:
            return

        exit_code = self.task_process.poll()
        if exit_code is not None:
            rospy.loginfo('%s 已退出，退出码：%s。', self.current_task['name'], exit_code)
            self.task_process = None
            self.current_task = None
            self.task_start_time = None
            self.auto_mode = False
            return

        elapsed = time.monotonic() - self.task_start_time
        timeout = self.current_task['timeout_seconds']
        if elapsed >= timeout:
            timeout_reason = '运行超时（%.1f 秒，限制 %.1f 秒）' % (elapsed, timeout)
            if self.auto_mode:
                rospy.logwarn('%s，自动推进下一任务。', timeout_reason)
                self.start_next_task()
            else:
                self.terminate_current_task(timeout_reason)

    def run(self):
        """以固定频率监控当前 launch 进程。"""
        while not rospy.is_shutdown():
            self.monitor_current_task()
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME)
    try:
        StateControl().run()
    except rospy.ROSInterruptException:
        pass
