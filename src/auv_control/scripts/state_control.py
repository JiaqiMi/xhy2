#!/home/nvidia/venvs/xhy_ros2/bin/python
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
2026.7.28
    将内层 launch 输出逐行转发，修复终端换行错位。
2026.7.28
    使用显式 CRLF 输出内层 launch 日志，兼容 roslaunch 管道终端。
2026.7.28
    使用 PTY 启动内层 launch，确保日志逐行刷新。
2026.7.30
    在 task2 预热 task3 视觉节点，并保持至 task3 结束。
2026.7.31
    增加视觉预热开关；关闭时由 task3 自身启动视觉模型。
2026.7.31
    支持 9 键预热 task1 视觉节点，并在 task1 中复用。
2026.8.3
    将任务正常退出与异常退出区分，task2 正常结束时保留 task3 视觉预热。
2026.8.5
    task3 未检测到预热时由自身 launch 启动视觉节点。
2026.8.5
    增加 M/N 控制 motion_supervisor 的保活与单次关闭。
2026.8.7
    通过 ROS 节点 API 确认 motion_supervisor 存活，避免残留注册阻止保活重启。
    默认不向任务控制终端转发 motion_supervisor 日志。
"""

import errno
import os
import pty
import subprocess
import sys
import threading
import time

import rosgraph
import rosnode
import rospy
try:
    from xmlrpc import client as xmlrpc_client
except ImportError:
    import xmlrpclib as xmlrpc_client
from auv_control.msg import Keyboard
from std_msgs.msg import String


NODE_NAME = 'state_control'


class TimeoutTransport(xmlrpc_client.Transport):
    """为 ROS 节点 XML-RPC 存活检测设置连接超时。"""

    def __init__(self, timeout_seconds):
        xmlrpc_client.Transport.__init__(self)
        self.timeout_seconds = timeout_seconds

    def make_connection(self, host):
        """创建带超时的 HTTP 连接，防止失效节点阻塞状态监控。"""
        connection = xmlrpc_client.Transport.make_connection(self, host)
        connection.timeout = self.timeout_seconds
        return connection


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
        self.task1_vision_prewarm = self._load_vision_prewarm(
            rospy.get_param('~task1_vision_prewarm', {}), 'task1')
        self.vision_prewarm = self._load_vision_prewarm(
            rospy.get_param('~task3_vision_prewarm', {}), 'task3')
        self.motion_supervisor = self._load_motion_supervisor(
            rospy.get_param('~motion_supervisor', {}))

        self.current_task = None
        self.task_process = None
        self.task_start_time = None
        self.task_output_fd = None
        self.task_output_thread = None
        self.task1_vision_process = None
        self.task1_vision_output_fd = None
        self.task1_vision_output_thread = None
        self.vision_process = None
        self.vision_output_fd = None
        self.vision_output_thread = None
        self.motion_supervisor_process = None
        self.motion_supervisor_output_fd = None
        self.motion_supervisor_output_thread = None
        self.motion_supervisor_start_time = None
        self.motion_supervisor_keepalive = False
        self.motion_supervisor_stop_requested = False
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

    @staticmethod
    def _load_vision_prewarm(config, task_name):
        """读取指定任务的视觉预热进程配置。"""
        if not config:
            return None
        try:
            launch = str(config['launch'])
            active_modes = {int(mode) for mode in config['active_modes']}
            start_modes = {int(mode) for mode in
                           config.get('start_modes', active_modes)}
        except (KeyError, TypeError, ValueError) as error:
            rospy.logerr('%s %s 视觉预热配置无效：%s', NODE_NAME,
                         task_name, error)
            return None
        if not launch or not active_modes:
            rospy.logerr('%s %s 视觉预热配置为空。', NODE_NAME, task_name)
            return None
        return {
            'enabled': bool(config.get('enabled', True)),
            'name': str(config.get('name', '{}_vision'.format(task_name))),
            'launch': launch,
            'launch_args': [str(argument) for argument in
                            config.get('launch_args', [])],
            'reuse_launch_args': [str(argument) for argument in
                                  config.get('reuse_launch_args', [])],
            'active_modes': active_modes,
            'start_modes': start_modes,
        }

    @staticmethod
    def _load_motion_supervisor(config):
        """读取 motion_supervisor 启动、节点检测和保活配置。"""
        if not config:
            return None
        try:
            launch = str(config['launch'])
            node_name = str(config['node_name'])
            startup_timeout_seconds = float(
                config.get('startup_timeout_seconds', 15.0))
            health_check_timeout_seconds = float(
                config.get('health_check_timeout_seconds', 1.0))
        except (KeyError, TypeError, ValueError) as error:
            rospy.logerr('%s motion_supervisor 配置无效：%s', NODE_NAME, error)
            return None
        if (not launch or not node_name or startup_timeout_seconds <= 0 or
                health_check_timeout_seconds <= 0):
            rospy.logerr('%s motion_supervisor 配置为空或超时无效。', NODE_NAME)
            return None
        return {
            'name': str(config.get('name', 'motion_supervisor')),
            'launch': launch,
            'launch_args': [str(argument) for argument in
                            config.get('launch_args', [])],
            'node_name': node_name,
            'startup_timeout_seconds': startup_timeout_seconds,
            'health_check_timeout_seconds': health_check_timeout_seconds,
            'forward_output': bool(config.get('forward_output', False)),
        }

    def keyboard_callback(self, message):
        """响应启动指定任务或停止当前任务的键盘指令。"""
        rospy.loginfo('%s 收到键盘指令：run=%s，mode=%s。', NODE_NAME,
                      message.run, message.mode)
        if message.run == 2:
            self.auto_mode = False
            self.terminate_current_task('收到停止指令')
            self.terminate_task1_vision_prewarm('收到停止指令')
            self.terminate_vision_prewarm('收到停止指令')
            return

        if message.run == 3:
            self.enable_motion_supervisor_keepalive()
            return

        if message.run == 4:
            self.disable_motion_supervisor_keepalive()
            return

        if message.run == 1:
            self.start_automatic_tasks()
            return

        if message.run != 0:
            rospy.logwarn('%s 忽略不支持的 run 值：%s。', NODE_NAME, message.run)
            return

        if message.mode == 9:
            self.start_task1_vision_prewarm()
            return

        task = self.tasks.get(message.mode)
        if task is None:
            rospy.logwarn('%s 未配置 mode=%s 对应的任务。', NODE_NAME, message.mode)
            return
        self.start_task(task, automatic=False)

    def start_task(self, task, automatic=False):
        """停止旧任务后，以 roslaunch 启动指定任务。"""
        self.terminate_current_task('切换任务')
        if task['mode'] != 1:
            self.terminate_task1_vision_prewarm('切换到非 task1 任务')
        self.ensure_vision_prewarm(task['mode'])
        launch_args = list(task['launch_args'])
        if task['mode'] == 1:
            if self.task1_vision_prewarm_running():
                launch_args += self.task1_vision_prewarm['reuse_launch_args']
                rospy.loginfo('%s 检测到 task1 视觉预热，复用已有节点。',
                              NODE_NAME)
            else:
                rospy.loginfo('%s 未检测到 task1 视觉预热，由 task1 启动视觉节点。',
                              NODE_NAME)
        if task['mode'] == 3:
            if self.vision_prewarm_running():
                launch_args += self.vision_prewarm['reuse_launch_args']
                rospy.loginfo('%s 检测到 task3 视觉预热，复用已有节点。',
                              NODE_NAME)
            else:
                rospy.loginfo('%s 未检测到 task3 视觉预热，由 task3 启动视觉节点。',
                              NODE_NAME)
        command = ['roslaunch', 'auv_control', task['launch']] + launch_args
        try:
            master_fd, slave_fd = pty.openpty()
            try:
                self.task_process = subprocess.Popen(
                    command,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    close_fds=True)
            except OSError:
                os.close(master_fd)
                raise
            finally:
                os.close(slave_fd)
            self.current_task = task
            self.task_start_time = time.monotonic()
            self.auto_mode = automatic
            self.task_output_fd = master_fd
            self.task_output_thread = threading.Thread(
                target=self._forward_task_output,
                args=(task['name'], master_fd),
                name='{}_launch_output'.format(task['name']),
                daemon=True)
            self.task_output_thread.start()
            rospy.loginfo('%s 已启动 %s：%s。超时 %.0f 秒。', NODE_NAME,
                          task['name'], ' '.join(command), task['timeout_seconds'])
        except OSError as error:
            self.task_process = None
            self.current_task = None
            self.task_start_time = None
            self.task_output_fd = None
            self.task_output_thread = None
            self.auto_mode = False
            rospy.logerr('%s 启动 %s 失败：%s', NODE_NAME, task['name'], error)

    @staticmethod
    def _forward_task_output(task_name, output_fd):
        """从 PTY 逐行转发内层 roslaunch 输出。"""
        pending = b''
        try:
            while True:
                try:
                    chunk = os.read(output_fd, 4096)
                except OSError as error:
                    if error.errno in (errno.EIO, errno.EBADF):
                        break
                    raise
                if not chunk:
                    break
                pending += chunk
                while b'\n' in pending:
                    raw_line, pending = pending.split(b'\n', 1)
                    line = raw_line.decode('utf-8', errors='replace').rstrip('\r')
                    if line:
                        sys.stdout.write('{} launch: {}\r\n'.format(task_name, line))
                        sys.stdout.flush()
            if pending:
                line = pending.decode('utf-8', errors='replace').rstrip('\r')
                if line:
                    sys.stdout.write('{} launch: {}\r\n'.format(task_name, line))
                    sys.stdout.flush()
        finally:
            try:
                os.close(output_fd)
            except OSError:
                pass

    def _close_task_output(self):
        """关闭 PTY 并等待日志转发线程退出。"""
        if self.task_output_fd is not None:
            try:
                os.close(self.task_output_fd)
            except OSError:
                pass
            self.task_output_fd = None
        if self.task_output_thread is not None:
            self.task_output_thread.join(timeout=1.0)
            self.task_output_thread = None

    def ensure_vision_prewarm(self, task_mode):
        """按任务阶段启动或保留 task3 视觉预热进程。"""
        if self.vision_prewarm is None:
            return
        if not self.vision_prewarm['enabled']:
            self.terminate_vision_prewarm('视觉预热已关闭')
            return
        if task_mode not in self.vision_prewarm['active_modes']:
            self.terminate_vision_prewarm('切换到非视觉预热任务')
            return
        if self.vision_prewarm_running():
            return
        if task_mode not in self.vision_prewarm['start_modes']:
            return

        self._close_vision_output()
        command = ['roslaunch', 'auv_control', self.vision_prewarm['launch']]
        command += self.vision_prewarm['launch_args']
        try:
            master_fd, slave_fd = pty.openpty()
            try:
                self.vision_process = subprocess.Popen(
                    command,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    close_fds=True)
            except OSError:
                os.close(master_fd)
                raise
            finally:
                os.close(slave_fd)
            self.vision_output_fd = master_fd
            self.vision_output_thread = threading.Thread(
                target=self._forward_task_output,
                args=(self.vision_prewarm['name'], master_fd),
                name='{}_output'.format(self.vision_prewarm['name']),
                daemon=True)
            self.vision_output_thread.start()
            rospy.loginfo('%s 已启动 task3 视觉预热：%s。', NODE_NAME,
                          ' '.join(command))
        except OSError as error:
            self.vision_process = None
            self.vision_output_fd = None
            self.vision_output_thread = None
            rospy.logerr('%s 启动 task3 视觉预热失败：%s', NODE_NAME, error)

    def vision_prewarm_running(self):
        """判断 task3 视觉预热进程是否仍在运行。"""
        return (self.vision_prewarm is not None and
                self.vision_prewarm['enabled'] and
                self.vision_process is not None and
                self.vision_process.poll() is None)

    def motion_supervisor_node_uri(self):
        """从 ROS Master 查询控制器节点 URI，未知节点视为未运行。"""
        if self.motion_supervisor is None:
            return None
        try:
            master = rosgraph.Master(rospy.get_name())
            return master.lookupNode(self.motion_supervisor['node_name'])
        except Exception as error:
            if 'unknown node' in str(error).lower():
                return None
            rospy.logerr_throttle(
                5.0, '%s 查询 motion_supervisor 节点失败：%s',
                NODE_NAME, error)
            return None

    def motion_supervisor_node_running(self):
        """确认控制器已注册且节点 API 可响应，排除 Master 残留注册。"""
        node_uri = self.motion_supervisor_node_uri()
        if not node_uri:
            return False
        try:
            transport = TimeoutTransport(
                self.motion_supervisor['health_check_timeout_seconds'])
            node_api = xmlrpc_client.ServerProxy(node_uri, transport=transport)
            response = node_api.getPid(rospy.get_name())
            if len(response) != 3 or response[0] != 1 or int(response[2]) <= 0:
                rospy.logwarn_throttle(
                    5.0, '%s motion_supervisor 节点响应无效：%s。',
                    NODE_NAME, response)
                return False
            return True
        except Exception as error:
            rospy.logwarn_throttle(
                5.0, '%s motion_supervisor 注册存在但无法通信，将执行保活重启：%s。',
                NODE_NAME, error)
            return False

    def motion_supervisor_launch_running(self):
        """判断本节点启动的 motion_supervisor launch 是否仍在运行。"""
        return (self.motion_supervisor_process is not None and
                self.motion_supervisor_process.poll() is None)

    def enable_motion_supervisor_keepalive(self):
        """开启 motion_supervisor 保活，并在缺失时立即启动。"""
        if self.motion_supervisor is None:
            rospy.logerr('%s 未配置 motion_supervisor，无法开启保活。', NODE_NAME)
            return
        self.motion_supervisor_keepalive = True
        self.motion_supervisor_stop_requested = False
        if self.motion_supervisor_node_running():
            rospy.loginfo('%s motion_supervisor 已在运行，开始保活监控。',
                          NODE_NAME)
            return
        rospy.logwarn('%s 未检测到 motion_supervisor，开始启动并保活。', NODE_NAME)
        self.start_motion_supervisor()

    def start_motion_supervisor(self):
        """在控制器节点缺失时启动其 launch，避免重复创建进程。"""
        if self.motion_supervisor is None:
            return
        if self.motion_supervisor_node_running():
            return
        if self.motion_supervisor_launch_running():
            elapsed = time.monotonic() - self.motion_supervisor_start_time
            timeout = self.motion_supervisor['startup_timeout_seconds']
            if elapsed < timeout:
                rospy.logwarn_throttle(
                    2.0, '%s 等待 motion_supervisor 节点注册：%.1f/%.1fs。',
                    NODE_NAME, elapsed, timeout)
                return
            rospy.logerr('%s motion_supervisor 启动超时，重启 launch。', NODE_NAME)
            self.terminate_motion_supervisor_launch('节点注册超时')

        self._close_motion_supervisor_output()
        command = ['roslaunch', 'auv_control', self.motion_supervisor['launch']]
        command += self.motion_supervisor['launch_args']
        try:
            if self.motion_supervisor['forward_output']:
                master_fd, slave_fd = pty.openpty()
                try:
                    self.motion_supervisor_process = subprocess.Popen(
                        command,
                        stdin=slave_fd,
                        stdout=slave_fd,
                        stderr=slave_fd,
                        close_fds=True)
                except OSError:
                    os.close(master_fd)
                    raise
                finally:
                    os.close(slave_fd)
                self.motion_supervisor_output_fd = master_fd
                self.motion_supervisor_output_thread = threading.Thread(
                    target=self._forward_task_output,
                    args=(self.motion_supervisor['name'], master_fd),
                    name='{}_output'.format(self.motion_supervisor['name']),
                    daemon=True)
                self.motion_supervisor_output_thread.start()
            else:
                with open(os.devnull, 'wb') as devnull:
                    self.motion_supervisor_process = subprocess.Popen(
                        command,
                        stdin=devnull,
                        stdout=devnull,
                        stderr=devnull,
                        close_fds=True)
            self.motion_supervisor_start_time = time.monotonic()
            if self.motion_supervisor['forward_output']:
                rospy.loginfo('%s 已启动 motion_supervisor：%s。', NODE_NAME,
                              ' '.join(command))
            else:
                rospy.loginfo('%s 已启动 motion_supervisor，日志不转发到当前终端。',
                              NODE_NAME)
        except OSError as error:
            self.motion_supervisor_process = None
            self.motion_supervisor_start_time = None
            self.motion_supervisor_output_fd = None
            self.motion_supervisor_output_thread = None
            rospy.logerr('%s 启动 motion_supervisor 失败，将继续保活重试：%s',
                         NODE_NAME, error)

    def _close_motion_supervisor_output(self):
        """关闭 motion_supervisor launch 的 PTY 输出。"""
        if self.motion_supervisor_output_fd is not None:
            try:
                os.close(self.motion_supervisor_output_fd)
            except OSError:
                pass
            self.motion_supervisor_output_fd = None
        if self.motion_supervisor_output_thread is not None:
            self.motion_supervisor_output_thread.join(timeout=1.0)
            self.motion_supervisor_output_thread = None

    def terminate_motion_supervisor_launch(self, reason):
        """关闭本节点启动的 motion_supervisor launch。"""
        if self.motion_supervisor_process is None:
            return False
        if self.motion_supervisor_process.poll() is None:
            rospy.loginfo('%s 正在停止 motion_supervisor launch：%s。',
                          NODE_NAME, reason)
            self.motion_supervisor_process.terminate()
            try:
                self.motion_supervisor_process.wait(
                    timeout=self.terminate_wait_seconds)
            except subprocess.TimeoutExpired:
                rospy.logwarn('%s motion_supervisor 未及时退出，强制结束。',
                              NODE_NAME)
                self.motion_supervisor_process.kill()
                self.motion_supervisor_process.wait()
        self.motion_supervisor_process = None
        self.motion_supervisor_start_time = None
        self._close_motion_supervisor_output()
        return True

    def disable_motion_supervisor_keepalive(self):
        """关闭保活，并仅执行一次控制器关闭请求。"""
        self.motion_supervisor_keepalive = False
        if self.motion_supervisor_stop_requested:
            rospy.loginfo('%s 已执行过 motion_supervisor 关闭请求。', NODE_NAME)
            return
        self.motion_supervisor_stop_requested = True

        if self.terminate_motion_supervisor_launch('收到 N 键关闭指令'):
            return
        if self.motion_supervisor is None:
            rospy.logerr('%s 未配置 motion_supervisor，无法发送关闭请求。', NODE_NAME)
            return
        if not self.motion_supervisor_node_running():
            rospy.loginfo('%s motion_supervisor 当前未运行，无需关闭。', NODE_NAME)
            return
        try:
            succeeded, failed = rosnode.kill_nodes(
                [self.motion_supervisor['node_name']])
        except Exception as error:
            rospy.logerr('%s 关闭外部 motion_supervisor 失败：%s',
                         NODE_NAME, error)
            return
        if failed:
            rospy.logerr('%s motion_supervisor 关闭失败：%s', NODE_NAME, failed)
        else:
            rospy.loginfo('%s 已关闭外部 motion_supervisor：%s。', NODE_NAME,
                          succeeded)

    def task1_vision_prewarm_running(self):
        """判断 task1 视觉预热进程是否仍在运行。"""
        return (self.task1_vision_prewarm is not None and
                self.task1_vision_prewarm['enabled'] and
                self.task1_vision_process is not None and
                self.task1_vision_process.poll() is None)

    def start_task1_vision_prewarm(self):
        """按 9 键启动 task1 视觉预热进程。"""
        if self.task_process is not None and self.task_process.poll() is None:
            rospy.logwarn('%s 当前 task%s 正在运行，拒绝启动 task1 视觉预热。',
                          NODE_NAME, self.current_task['mode'])
            return
        if self.task1_vision_prewarm is None:
            rospy.logwarn('%s 未配置 task1 视觉预热。', NODE_NAME)
            return
        if not self.task1_vision_prewarm['enabled']:
            rospy.logwarn('%s task1 视觉预热已在配置中关闭。', NODE_NAME)
            return
        if self.task1_vision_prewarm_running():
            rospy.loginfo('%s task1 视觉预热已在运行。', NODE_NAME)
            return

        self._close_task1_vision_output()
        command = ['roslaunch', 'auv_control',
                   self.task1_vision_prewarm['launch']]
        command += self.task1_vision_prewarm['launch_args']
        try:
            master_fd, slave_fd = pty.openpty()
            try:
                self.task1_vision_process = subprocess.Popen(
                    command,
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    close_fds=True)
            except OSError:
                os.close(master_fd)
                raise
            finally:
                os.close(slave_fd)
            self.task1_vision_output_fd = master_fd
            self.task1_vision_output_thread = threading.Thread(
                target=self._forward_task_output,
                args=(self.task1_vision_prewarm['name'], master_fd),
                name='{}_output'.format(self.task1_vision_prewarm['name']),
                daemon=True)
            self.task1_vision_output_thread.start()
            rospy.loginfo('%s 已启动 task1 视觉预热：%s。', NODE_NAME,
                          ' '.join(command))
        except OSError as error:
            self.task1_vision_process = None
            self.task1_vision_output_fd = None
            self.task1_vision_output_thread = None
            rospy.logerr('%s 启动 task1 视觉预热失败：%s', NODE_NAME, error)

    def _close_task1_vision_output(self):
        """关闭 task1 视觉预热的 PTY 输出。"""
        if self.task1_vision_output_fd is not None:
            try:
                os.close(self.task1_vision_output_fd)
            except OSError:
                pass
            self.task1_vision_output_fd = None
        if self.task1_vision_output_thread is not None:
            self.task1_vision_output_thread.join(timeout=1.0)
            self.task1_vision_output_thread = None

    def terminate_task1_vision_prewarm(self, reason):
        """结束 task1 视觉预热进程及其节点。"""
        if self.task1_vision_process is None:
            return
        if self.task1_vision_process.poll() is None:
            rospy.loginfo('%s 正在停止 task1 视觉预热：%s。', NODE_NAME, reason)
            self.task1_vision_process.terminate()
            try:
                self.task1_vision_process.wait(
                    timeout=self.terminate_wait_seconds)
            except subprocess.TimeoutExpired:
                self.task1_vision_process.kill()
                self.task1_vision_process.wait()
        self.task1_vision_process = None
        self._close_task1_vision_output()

    def _close_vision_output(self):
        """关闭 task3 视觉预热的 PTY 输出。"""
        if self.vision_output_fd is not None:
            try:
                os.close(self.vision_output_fd)
            except OSError:
                pass
            self.vision_output_fd = None
        if self.vision_output_thread is not None:
            self.vision_output_thread.join(timeout=1.0)
            self.vision_output_thread = None

    def terminate_vision_prewarm(self, reason):
        """结束 task3 视觉预热进程及其节点。"""
        if self.vision_process is None:
            return
        if self.vision_process.poll() is None:
            rospy.loginfo('%s 正在停止 task3 视觉预热：%s。', NODE_NAME, reason)
            self.vision_process.terminate()
            try:
                self.vision_process.wait(timeout=self.terminate_wait_seconds)
            except subprocess.TimeoutExpired:
                self.vision_process.kill()
                self.vision_process.wait()
        self.vision_process = None
        self._close_vision_output()

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
            self.terminate_vision_prewarm('自动串联任务完成')
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
        self._close_task_output()

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
            completed_mode = self.current_task['mode']
            self.terminate_current_task('收到完成消息')
            if completed_mode == 1:
                self.terminate_task1_vision_prewarm('task1 完成')
            if completed_mode == 3:
                self.terminate_vision_prewarm('task3 完成')

    def monitor_current_task(self):
        """检测 launch 退出或超过配置时限，并按退出结果清理状态。"""
        if self.task_process is None or self.current_task is None:
            return

        exit_code = self.task_process.poll()
        if exit_code is not None:
            exited_mode = self.current_task['mode']
            task_name = self.current_task['name']
            automatic = self.auto_mode
            rospy.loginfo('%s 已退出，退出码：%s。', task_name, exit_code)

            if exit_code == 0 and automatic:
                rospy.loginfo('%s 正常完成，自动推进下一任务。', NODE_NAME)
                self.start_next_task()
                return

            self.task_process = None
            self.current_task = None
            self.task_start_time = None
            self._close_task_output()
            self.auto_mode = False
            if exit_code == 0:
                rospy.loginfo('%s 正常完成。', task_name)
                if exited_mode == 1:
                    self.terminate_task1_vision_prewarm('task1 正常完成')
                elif exited_mode == 3:
                    self.terminate_vision_prewarm('task3 正常完成')
                else:
                    rospy.loginfo('%s 保留 task3 视觉预热，等待启动 task3。',
                                  NODE_NAME)
                return

            rospy.logerr('%s 异常退出，退出码：%s。', task_name, exit_code)
            if exited_mode == 1:
                self.terminate_task1_vision_prewarm('任务异常退出')
            if (self.vision_prewarm is not None and
                    exited_mode in self.vision_prewarm['active_modes']):
                self.terminate_vision_prewarm('任务异常退出')
            return

        elapsed = time.monotonic() - self.task_start_time
        timeout = self.current_task['timeout_seconds']
        if elapsed >= timeout:
            timeout_reason = '运行超时（%.1f 秒，限制 %.1f 秒）' % (elapsed, timeout)
            if self.auto_mode:
                rospy.logwarn('%s，自动推进下一任务。', timeout_reason)
                self.start_next_task()
            else:
                timed_out_mode = self.current_task['mode']
                self.terminate_current_task(timeout_reason)
                if timed_out_mode == 1:
                    self.terminate_task1_vision_prewarm('task1 运行超时')

    def monitor_motion_supervisor(self):
        """保活开启后检测控制器节点，失效时继续尝试重启。"""
        if not self.motion_supervisor_keepalive:
            return
        if self.motion_supervisor_node_running():
            return
        if self.motion_supervisor_launch_running():
            self.start_motion_supervisor()
            return
        if self.motion_supervisor_process is not None:
            exit_code = self.motion_supervisor_process.poll()
            self.motion_supervisor_process = None
            self.motion_supervisor_start_time = None
            self._close_motion_supervisor_output()
            rospy.logerr_throttle(
                5.0, '%s motion_supervisor launch 已退出，退出码=%s，将重启。',
                NODE_NAME, exit_code)
        rospy.logerr_throttle(
            5.0, '%s 未检测到 motion_supervisor，执行保活重启。', NODE_NAME)
        self.start_motion_supervisor()

    def run(self):
        """以固定频率监控当前 launch 进程。"""
        while not rospy.is_shutdown():
            self.monitor_current_task()
            self.monitor_motion_supervisor()
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME)
    try:
        StateControl().run()
    except rospy.ROSInterruptException:
        pass
