#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
名称：task2_v2.py
功能：2026 Task 2——采水器采水、送水浮出与返航任务节点
作者：buyegaid
监听：/tf，/motion/state，/status/actuator
发布：/cmd/motion/goal，/cmd/actuator，/finished
记录：
    2026-07-13：
        1. 新增取水器定点采水、深度保持返航和原点保持 10 秒流程；
        2. 新增 /task_v2_sample_duration、/task_v2_pushrod_speed、
           /task_v2_return_yaw_deg 参数；
        3. 推杆前进速度由固定值 250 改为参数化配置，默认值为 250。
        4. 统一日志格式，日志正文以节点名称 task2_v2 开头。
        5. 执行器下行话题调整为 /cmd/actuator。
    2026-07-15：
        1. 通过 MissionBase 将运动控制从 /target 迁移到 /cmd/pose/ned。
        2. 推杆命令使用 mode=2，仅更新执行器字段。
    2026.7.24
        1. 迁移至 motion_supervisor 的 /cmd/motion/goal 与 /motion/state 接口。
        2. 新增送水浮出、返回起始点、执行器反馈闭环和异常安全定点流程。
        3. 新增文本日志和结构化 YAML 数据流，记录运行判断、目标与反馈快照。
"""

from datetime import datetime
import json
import logging
import math
import os
import threading
import time

import rospy
import tf
from auv_control.msg import ActuatorControl, MotionState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = 'task2_v2'
PUSHROD_STOP = 0
PUSHROD_FORWARD = 1
MAIN_RATE_HZ = 5.0


def configure_task_file_logging(node_name, log_directory):
    """将 rospy 文本日志保存到带时间戳的 UTF-8 文件。"""
    directory = os.path.abspath(os.path.expanduser(str(log_directory)))
    try:
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(directory, '{}_{}.log'.format(node_name, timestamp))
        handler = logging.FileHandler(path, mode='a', encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'))
        logging.getLogger('rosout').addHandler(handler)
        return path, handler
    except (IOError, OSError) as error:
        rospy.logerr('%s: 无法创建文本日志目录 %s：%s', node_name, directory, error)
        return None, None


class Task2V2(object):
    """采水器任务状态机，只使用控制器和执行器的反馈完成状态转场。"""

    WAIT_TF = 'WAIT_TF'
    WAIT_ACTUATOR_INIT = 'WAIT_ACTUATOR_INIT'
    INITIAL_HOLD = 'INITIAL_HOLD'
    WAIT_PUSHROD_START = 'WAIT_PUSHROD_START'
    SAMPLING = 'SAMPLING'
    WAIT_PUSHROD_STOP = 'WAIT_PUSHROD_STOP'
    DELIVERY_DEPTH = 'DELIVERY_DEPTH'
    SURFACE = 'SURFACE'
    SURFACE_HOLD = 'SURFACE_HOLD'
    DIVE = 'DIVE'
    RETURN_HOME = 'RETURN_HOME'
    FINAL_HOLD = 'FINAL_HOLD'
    FINAL_STOP_CONFIRM = 'FINAL_STOP_CONFIRM'
    SAFE_FINAL = 'SAFE_FINAL'

    def __init__(self):
        """读取参数、初始化 ROS 接口和状态缓存。"""
        self.rate_hz = MAIN_RATE_HZ
        self.rate = rospy.Rate(self.rate_hz)
        self.map_frame = str(rospy.get_param('~map_frame', 'map')).strip()
        self.base_frame = str(rospy.get_param('~base_frame', 'base_link')).strip()

        self.motion_goal_topic = str(rospy.get_param(
            '~motion_goal_topic', '/cmd/motion/goal')).strip()
        self.motion_state_topic = str(rospy.get_param(
            '~motion_state_topic', '/motion/state')).strip()
        self.actuator_topic = str(rospy.get_param(
            '~actuator_topic', '/cmd/actuator')).strip()
        self.actuator_status_topic = str(rospy.get_param(
            '~actuator_status_topic', '/status/actuator')).strip()
        self.finished_topic = str(rospy.get_param(
            '~finished_topic', '/finished')).strip()
        self.log_directory = os.path.abspath(os.path.expanduser(str(
            rospy.get_param('~log_directory', '~/.ros/auv_logs/task2'))))
        self.data_log_enabled = bool(rospy.get_param('~data_log_enabled', True))

        self.task_depth = float(rospy.get_param('~task_depth', -1.0))
        self.delivery_xy = rospy.get_param('~delivery_xy', [0.0, 0.0])
        self.delivery_yaw_deg = float(rospy.get_param('~delivery_yaw_deg', 0.0))
        self.surface_depth = float(rospy.get_param('~surface_depth', 0.0))
        self.final_yaw_deg = float(rospy.get_param('~final_yaw_deg', 0.0))
        self.pushrod_speed = int(rospy.get_param('~pushrod_speed', 250))
        self.pushrod_duration = float(rospy.get_param(
            '~pushrod_duration', 10.0))
        self.surface_hold_seconds = float(rospy.get_param(
            '~surface_hold_seconds', 10.0))
        self.final_hold_seconds = float(rospy.get_param(
            '~final_hold_seconds', 10.0))

        self.heading_servo_right = int(rospy.get_param(
            '~heading_servo_right', 0))
        self.clamp_closed = int(rospy.get_param('~clamp_closed', 255))
        self.pushrod_forward_cmd = int(rospy.get_param(
            '~pushrod_forward_cmd', PUSHROD_FORWARD))
        self.feedback_confirm_frames = int(rospy.get_param(
            '~feedback_confirm_frames', 2))

        self.tf_initial_timeout = float(rospy.get_param(
            '~tf_initial_timeout', 30.0))
        self.motion_state_timeout = float(rospy.get_param(
            '~motion_state_timeout', 0.5))
        self.motion_recovery_timeout = float(rospy.get_param(
            '~motion_recovery_timeout', 10.0))
        self.initial_hover_timeout = float(rospy.get_param(
            '~initial_hover_timeout', 120.0))
        self.delivery_depth_timeout = float(rospy.get_param(
            '~delivery_depth_timeout', 180.0))
        self.surface_timeout = float(rospy.get_param(
            '~surface_timeout', 120.0))
        self.dive_timeout = float(rospy.get_param('~dive_timeout', 120.0))
        self.return_home_timeout = float(rospy.get_param(
            '~return_home_timeout', 180.0))
        self.final_safe_warning_timeout = float(rospy.get_param(
            '~final_safe_warning_timeout', 120.0))
        self.actuator_status_timeout = float(rospy.get_param(
            '~actuator_status_timeout', 0.6))
        self.actuator_confirm_timeout = float(rospy.get_param(
            '~actuator_confirm_timeout', 5.0))

        self._validate_parameters()

        self.ros_log_path, self.ros_log_handler = configure_task_file_logging(
            NODE_NAME, self.log_directory)
        self.data_log_path = None
        self.data_log_file = None
        self.data_log_lock = threading.Lock()
        self.last_decision = {'name': 'startup', 'detail': '等待节点初始化'}
        if self.data_log_enabled:
            self._open_data_log()

        self.tf_listener = tf.TransformListener()
        self.motion_goal_pub = rospy.Publisher(
            self.motion_goal_topic, PoseStamped, queue_size=1)
        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10)
        self.finished_pub = rospy.Publisher(
            self.finished_topic, String, queue_size=1, latch=True)

        self.lock = threading.Lock()
        self.latest_motion_state = None
        self.latest_motion_state_at = None
        self.latest_actuator_status = None
        self.latest_actuator_status_at = None
        self.actuator_status_sequence = 0
        rospy.Subscriber(
            self.motion_state_topic, MotionState, self._motion_state_callback,
            queue_size=20)
        rospy.Subscriber(
            self.actuator_status_topic, ActuatorControl,
            self._actuator_status_callback, queue_size=20)

        self.state = self.WAIT_TF
        self.state_started_at = time.monotonic()
        self.state_paused_at = None
        self.active_goal = None
        self.active_goal_started_at = None
        self.initial_goal = None
        self.delivery_depth_goal = None
        self.surface_goal = None
        self.dive_goal = None
        self.return_home_goal = None
        self.final_safe_goal = None
        self.initial_tf_wait_warned = False
        self.motion_unhealthy_since = None
        self.sampling_started_at = None
        self.surface_hold_started_at = None
        self.final_hold_started_at = None
        self.safe_final_started_at = None
        self.safe_final_warned = False
        self.failure_reason = None
        self.finished = False

        self.expected_actuator = None
        self.last_actuator_command = None
        self.expected_actuator_started_at = None
        self.feedback_match_count = 0
        self.feedback_checked_sequence = 0
        self.actuator_recovering = False
        self.actuator_recovery_reason = None

        rospy.on_shutdown(self._on_shutdown)
        self._write_data_record(
            'startup',
            log_directory=self.log_directory,
            ros_log_path=self.ros_log_path,
            data_log_path=self.data_log_path,
            task_depth=self.task_depth,
            delivery_xy=self.delivery_xy,
            delivery_yaw_deg=self.delivery_yaw_deg,
            surface_depth=self.surface_depth,
            final_yaw_deg=self.final_yaw_deg,
            pushrod_speed=self.pushrod_speed,
            pushrod_duration=self.pushrod_duration,
            surface_hold_seconds=self.surface_hold_seconds,
            final_hold_seconds=self.final_hold_seconds,
            feedback_confirm_frames=self.feedback_confirm_frames,
        )
        rospy.loginfo(
            '%s: 已启动，主循环 %.1fHz，文本日志=%s，数据日志=%s，等待 %s -> %s 的首帧 TF',
            NODE_NAME, self.rate_hz, self.ros_log_path, self.data_log_path,
            self.map_frame, self.base_frame)

    def _validate_parameters(self):
        """校验启动参数，避免在水下因错误配置进入错误任务流程。"""
        positive_values = {
            'tf_initial_timeout': self.tf_initial_timeout,
            'motion_state_timeout': self.motion_state_timeout,
            'motion_recovery_timeout': self.motion_recovery_timeout,
            'initial_hover_timeout': self.initial_hover_timeout,
            'delivery_depth_timeout': self.delivery_depth_timeout,
            'surface_timeout': self.surface_timeout,
            'dive_timeout': self.dive_timeout,
            'return_home_timeout': self.return_home_timeout,
            'final_safe_warning_timeout': self.final_safe_warning_timeout,
            'actuator_status_timeout': self.actuator_status_timeout,
            'actuator_confirm_timeout': self.actuator_confirm_timeout,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError('{} 必须为有限正数'.format(name))

        non_negative_values = {
            'pushrod_duration': self.pushrod_duration,
            'surface_hold_seconds': self.surface_hold_seconds,
            'final_hold_seconds': self.final_hold_seconds,
        }
        for name, value in non_negative_values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError('{} 必须为有限非负数'.format(name))

        for name, value in {
                'task_depth': self.task_depth,
                'surface_depth': self.surface_depth,
                'delivery_yaw_deg': self.delivery_yaw_deg,
                'final_yaw_deg': self.final_yaw_deg,
        }.items():
            if not math.isfinite(value):
                raise ValueError('{} 必须为有限数'.format(name))

        if not isinstance(self.delivery_xy, (list, tuple)) or len(self.delivery_xy) != 2:
            raise ValueError('delivery_xy 必须为 [x, y]')
        self.delivery_xy = [float(value) for value in self.delivery_xy]
        if not all(math.isfinite(value) for value in self.delivery_xy):
            raise ValueError('delivery_xy 必须包含有限数')
        if not 0 <= self.pushrod_speed <= 254:
            raise ValueError('pushrod_speed 必须在 [0, 254] 内')
        if self.pushrod_forward_cmd not in (1, 2):
            raise ValueError('pushrod_forward_cmd 必须为 1 或 2')
        if self.feedback_confirm_frames <= 0:
            raise ValueError('feedback_confirm_frames 必须大于 0')
        for name, value in {
                'heading_servo_right': self.heading_servo_right,
                'clamp_closed': self.clamp_closed,
        }.items():
            if not 0 <= value <= 255:
                raise ValueError('{} 必须在 [0, 255] 内'.format(name))
        for name, value in {
                'map_frame': self.map_frame,
                'base_frame': self.base_frame,
                'motion_goal_topic': self.motion_goal_topic,
                'motion_state_topic': self.motion_state_topic,
                'actuator_topic': self.actuator_topic,
                'actuator_status_topic': self.actuator_status_topic,
                'finished_topic': self.finished_topic,
        }.items():
            if not value:
                raise ValueError('{} 不可为空'.format(name))

    def _open_data_log(self):
        """创建可解析的 YAML 文档流，用于完整复盘任务运行过程。"""
        try:
            os.makedirs(self.log_directory, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.data_log_path = os.path.join(
                self.log_directory, '{}_{}.yaml'.format(NODE_NAME, timestamp))
            self.data_log_file = open(
                self.data_log_path, 'a', encoding='utf-8', buffering=1)
        except (IOError, OSError) as error:
            self.data_log_path = None
            self.data_log_file = None
            rospy.logwarn('%s: 无法创建结构化数据日志：%s', NODE_NAME, error)

    @staticmethod
    def _safe_log_value(value):
        """将消息字段递归转换为 JSON/YAML 可安全表示的基础数据。"""
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {
                str(key): Task2V2._safe_log_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [Task2V2._safe_log_value(item) for item in value]
        return str(value)

    def _pose_snapshot(self, pose):
        """将 PoseStamped 转换为结构化日志字段。"""
        if pose is None:
            return None
        yaw = self._yaw_from_quaternion(pose.pose.orientation)
        return {
            'frame_id': pose.header.frame_id,
            'position': [
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ],
            'orientation': [
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ],
            'yaw_deg': None if yaw is None else math.degrees(yaw),
        }

    @staticmethod
    def _actuator_snapshot(status):
        """将执行器命令或反馈转换为结构化日志字段。"""
        if status is None:
            return None
        return {
            'mode': status.mode,
            'light1': status.light1,
            'light2': status.light2,
            'heading_servo': status.heading_servo,
            'clamp_servo': status.clamp_servo,
            'drive_cmd': status.drive_cmd,
            'drive_speed': status.drive_speed,
            'red_light': status.red_light,
            'yellow_light': status.yellow_light,
            'green_light': status.green_light,
        }

    def _motion_snapshot(self, status):
        """将 motion_supervisor 反馈完整映射为复盘数据。"""
        if status is None:
            return None
        return {
            'state': status.state,
            'goal_active': status.goal_active,
            'startup_complete': status.startup_complete,
            'goal': self._pose_snapshot(status.goal),
            'position_error': status.position_error,
            'base_position_error': status.base_position_error,
            'yaw_error_rad': status.yaw_error,
            'horizontal_speed': status.horizontal_speed,
            'yaw_rate': status.yaw_rate,
            'tx': status.tx,
            'ty': status.ty,
            'mz': status.mz,
            'x_axis_state': status.x_axis_state,
            'y_axis_state': status.y_axis_state,
            'yaw_axis_state': status.yaw_axis_state,
            'x_axis_error': status.x_axis_error,
            'y_axis_error': status.y_axis_error,
            'x_axis_speed': status.x_axis_speed,
            'y_axis_speed': status.y_axis_speed,
            'reason': status.reason,
        }

    def _runtime_snapshot(self):
        """汇总当前状态机、目标和最新反馈，作为每条数据记录的公共上下文。"""
        with self.lock:
            motion_state = self.latest_motion_state
            motion_at = self.latest_motion_state_at
            actuator_status = self.latest_actuator_status
            actuator_at = self.latest_actuator_status_at
            actuator_sequence = self.actuator_status_sequence
        now = time.monotonic()
        return {
            'task_state': getattr(self, 'state', None),
            'task_state_elapsed': self._stage_elapsed() if hasattr(
                self, 'state_started_at') else None,
            'active_goal': self._pose_snapshot(getattr(self, 'active_goal', None)),
            'expected_actuator': getattr(self, 'expected_actuator', None),
            'last_actuator_command': getattr(self, 'last_actuator_command', None),
            'actuator_recovering': getattr(self, 'actuator_recovering', False),
            'feedback_match_count': getattr(self, 'feedback_match_count', 0),
            'actuator_status_sequence': actuator_sequence,
            'motion_status_age': None if motion_at is None else now - motion_at,
            'actuator_status_age': None if actuator_at is None else now - actuator_at,
            'motion_status': self._motion_snapshot(motion_state),
            'actuator_status': self._actuator_snapshot(actuator_status),
            'last_decision': getattr(self, 'last_decision', None),
        }

    def _write_data_record(self, event, **data):
        """追加一条独立 YAML 文档，写入判断、状态及全部可用反馈。"""
        if getattr(self, 'data_log_file', None) is None:
            return
        record = {
            'wall_time': datetime.now().isoformat(timespec='milliseconds'),
            'ros_time': round(rospy.Time.now().to_sec(), 6),
            'event': str(event),
        }
        record.update(self._runtime_snapshot())
        record.update(data)
        try:
            encoded = json.dumps(
                self._safe_log_value(record), ensure_ascii=False,
                allow_nan=False, separators=(',', ':'))
            with self.data_log_lock:
                if self.data_log_file is not None:
                    self.data_log_file.write('--- ' + encoded + '\n')
        except (IOError, OSError, TypeError, ValueError) as error:
            rospy.logwarn_throttle(5.0, '%s: 写入结构化数据日志失败：%s', NODE_NAME, error)

    def _set_last_decision(self, name, detail, **data):
        """保存当前循环的逻辑判断，供 5 Hz 快照和状态事件共同复盘。"""
        decision = {'name': str(name), 'detail': str(detail)}
        decision.update(data)
        self.last_decision = decision

    def _motion_state_callback(self, message):
        """缓存最新运动控制器状态及本机接收时间。"""
        with self.lock:
            self.latest_motion_state = message
            self.latest_motion_state_at = time.monotonic()
        self._write_data_record('motion_state_received')

    def _actuator_status_callback(self, message):
        """缓存最新执行器反馈；消息 mode 固定为 0，不参与匹配。"""
        with self.lock:
            self.latest_actuator_status = message
            self.latest_actuator_status_at = time.monotonic()
            self.actuator_status_sequence += 1
        self._write_data_record('actuator_status_received')

    def _set_state(self, state, reason):
        """切换任务状态并重置该状态的计时与健康缓存。"""
        previous = self.state
        self.state = state
        self.state_started_at = time.monotonic()
        self.state_paused_at = None
        self.motion_unhealthy_since = None
        self._set_last_decision('state_transition', reason, previous=previous, current=state)
        self._write_data_record(
            'state_transition', previous_state=previous,
            current_state=state, reason=reason)
        rospy.loginfo('%s: 状态 %s -> %s，%s', NODE_NAME, previous, state, reason)

    def _set_active_goal(self, goal, reason):
        """锁存待持续发布的绝对运动目标。"""
        self.active_goal = goal
        self.active_goal_started_at = time.monotonic()
        self.motion_unhealthy_since = None
        self._set_last_decision('set_motion_goal', reason, goal=self._pose_snapshot(goal))
        self._write_data_record('motion_goal_set', reason=reason, goal=self._pose_snapshot(goal))
        rospy.loginfo(
            '%s: 设置运动目标 %s=(%.3f, %.3f, %.3f)，%s',
            NODE_NAME, reason,
            goal.pose.position.x, goal.pose.position.y, goal.pose.position.z,
            self._yaw_text(goal))

    def _make_goal(self, x, y, z, yaw):
        """构造 map 坐标系下的固定绝对目标位姿。"""
        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.position.z = float(z)
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]
        return goal

    @staticmethod
    def _yaw_from_quaternion(quaternion):
        """从四元数中读取偏航角；无效四元数返回 None。"""
        values = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        if not all(math.isfinite(value) for value in values):
            return None
        if math.sqrt(sum(value * value for value in values)) < 1e-6:
            return None
        return euler_from_quaternion(values)[2]

    def _yaw_text(self, goal):
        """生成目标航向的日志文本。"""
        yaw = self._yaw_from_quaternion(goal.pose.orientation)
        if yaw is None:
            return 'yaw=invalid'
        return 'yaw={:.1f}deg'.format(math.degrees(yaw))

    def _capture_initial_goals(self):
        """从 TF 锁存起始姿态，并一次性构造任务全程固定目标。"""
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                self.map_frame, self.base_frame, rospy.Time(0))
        except tf.Exception as error:
            rospy.logwarn_throttle(
                1.0, '%s: 等待 TF %s -> %s：%s', NODE_NAME,
                self.map_frame, self.base_frame, error)
            return False

        values = list(translation) + list(rotation)
        if not all(math.isfinite(float(value)) for value in values):
            rospy.logwarn_throttle(1.0, '%s: TF 含无效数值，等待下一帧', NODE_NAME)
            return False
        if math.sqrt(sum(float(value) * float(value) for value in rotation)) < 1e-6:
            rospy.logwarn_throttle(1.0, '%s: TF 四元数无效，等待下一帧', NODE_NAME)
            return False
        initial_yaw = euler_from_quaternion(rotation)[2]

        initial_x, initial_y = float(translation[0]), float(translation[1])
        delivery_yaw = math.radians(self.delivery_yaw_deg)
        final_yaw = math.radians(self.final_yaw_deg)
        self.initial_goal = self._make_goal(
            initial_x, initial_y, self.task_depth, initial_yaw)
        self.delivery_depth_goal = self._make_goal(
            self.delivery_xy[0], self.delivery_xy[1], self.task_depth,
            delivery_yaw)
        self.surface_goal = self._make_goal(
            self.delivery_xy[0], self.delivery_xy[1], self.surface_depth,
            delivery_yaw)
        self.dive_goal = self._make_goal(
            self.delivery_xy[0], self.delivery_xy[1], self.task_depth,
            delivery_yaw)
        self.return_home_goal = self._make_goal(
            initial_x, initial_y, self.task_depth, final_yaw)
        self.final_safe_goal = self.return_home_goal
        self._write_data_record(
            'tf_latched', translation=list(translation), rotation=list(rotation),
            initial_goal=self._pose_snapshot(self.initial_goal),
            delivery_depth_goal=self._pose_snapshot(self.delivery_depth_goal),
            surface_goal=self._pose_snapshot(self.surface_goal),
            return_home_goal=self._pose_snapshot(self.return_home_goal))
        rospy.loginfo(
            '%s: 已锁存起始点 XY=(%.3f, %.3f)，初始航向 %.1fdeg；后续不会跟随实时位置',
            NODE_NAME, initial_x, initial_y, math.degrees(initial_yaw))
        return True

    def _publish_goal(self):
        """以 5 Hz 持续发布当前锁存的绝对运动目标。"""
        if self.active_goal is None:
            return
        self.active_goal.header.stamp = rospy.Time.now()
        self.motion_goal_pub.publish(self.active_goal)

    def _safe_actuator_command(self):
        """返回任务安全外设状态，推杆停止且所有指示灯关闭。"""
        return {
            'heading_servo': self.heading_servo_right,
            'clamp_servo': self.clamp_closed,
            'drive_cmd': PUSHROD_STOP,
            'drive_speed': 0,
            'red_light': 0,
            'yellow_light': 0,
            'green_light': 0,
        }

    def _pushrod_command(self):
        """返回采水期间的推杆前进外设状态。"""
        command = self._safe_actuator_command()
        command['drive_cmd'] = self.pushrod_forward_cmd
        command['drive_speed'] = self.pushrod_speed
        return command

    def _publish_actuator(self, command):
        """持续下发补光灯关闭和执行器命令。"""
        light_message = ActuatorControl()
        light_message.mode = 1
        light_message.light1 = 0
        light_message.light2 = 0
        self.actuator_pub.publish(light_message)

        message = ActuatorControl()
        message.mode = 2
        message.heading_servo = command['heading_servo']
        message.clamp_servo = command['clamp_servo']
        message.drive_cmd = command['drive_cmd']
        message.drive_speed = command['drive_speed']
        message.red_light = command['red_light']
        message.yellow_light = command['yellow_light']
        message.green_light = command['green_light']
        self.actuator_pub.publish(message)
        self.last_actuator_command = dict(command)

    @staticmethod
    def _command_key(command):
        """将执行器期望命令转换为可比较的不可变键。"""
        return (
            command['heading_servo'], command['clamp_servo'],
            command['drive_cmd'], command['drive_speed'],
            command['red_light'], command['yellow_light'],
            command['green_light'],
        )

    def _start_actuator_confirmation(self, command):
        """开始等待一组新的执行器反馈确认。"""
        self.expected_actuator = dict(command)
        self.expected_actuator_started_at = time.monotonic()
        self.feedback_match_count = 0
        with self.lock:
            self.feedback_checked_sequence = self.actuator_status_sequence

    def _actuator_feedback_matches(self, command):
        """检查一帧新鲜执行器反馈是否与期望硬件状态一致。"""
        with self.lock:
            status = self.latest_actuator_status
            received_at = self.latest_actuator_status_at
            sequence = self.actuator_status_sequence
        if status is None or received_at is None:
            return None, '尚未收到执行器反馈'
        age = time.monotonic() - received_at
        if age > self.actuator_status_timeout:
            return None, '执行器反馈超时 {:.2f}s'.format(age)
        if sequence == self.feedback_checked_sequence:
            return False, '等待下一帧执行器反馈'
        self.feedback_checked_sequence = sequence
        matched = (
            status.heading_servo == command['heading_servo'] and
            status.clamp_servo == command['clamp_servo'] and
            status.drive_cmd == command['drive_cmd'] and
            status.drive_speed == command['drive_speed'] and
            status.red_light == command['red_light'] and
            status.yellow_light == command['yellow_light'] and
            status.green_light == command['green_light']
        )
        if matched:
            return True, '执行器反馈匹配'
        return False, (
            '执行器反馈不匹配 heading=%d clamp=%d drive=(%d,%d) led=(%d,%d,%d)'
            % (
                status.heading_servo, status.clamp_servo,
                status.drive_cmd, status.drive_speed,
                status.red_light, status.yellow_light, status.green_light,
            )
        )

    def _actuator_gate(self, command, stage_name):
        """确认执行器状态；异常时先安全停止，再等待硬件反馈恢复。"""
        safe_command = self._safe_actuator_command()
        if self.actuator_recovering:
            if self.expected_actuator is None or (
                    self._command_key(self.expected_actuator) !=
                    self._command_key(safe_command)):
                self._start_actuator_confirmation(safe_command)
            self._publish_actuator(safe_command)
            matched, detail = self._actuator_feedback_matches(safe_command)
            if matched is True:
                self.feedback_match_count += 1
            elif matched is False:
                self.feedback_match_count = 0
            if self.feedback_match_count >= self.feedback_confirm_frames:
                rospy.loginfo('%s: 执行器安全停止反馈已恢复，继续 %s', NODE_NAME, stage_name)
                self.actuator_recovering = False
                self.actuator_recovery_reason = None
                self._start_actuator_confirmation(command)
                self._set_last_decision(
                    'actuator_recovered', '执行器安全停止反馈确认',
                    stage=stage_name, match_count=self.feedback_match_count)
                self._write_data_record('actuator_recovered', stage=stage_name)
                return False
            self._set_last_decision(
                'actuator_safe_recovery', detail, stage=stage_name,
                match_count=self.feedback_match_count)
            rospy.logwarn_throttle(
                1.0, '%s: 执行器安全停止确认中 %d/%d，%s', NODE_NAME,
                self.feedback_match_count, self.feedback_confirm_frames, detail)
            return False

        if self.expected_actuator is None or (
                self._command_key(self.expected_actuator) != self._command_key(command)):
            self._start_actuator_confirmation(command)
        self._publish_actuator(command)
        matched, detail = self._actuator_feedback_matches(command)
        if matched is True:
            self.feedback_match_count += 1
        elif matched is False:
            self.feedback_match_count = 0
        if self.feedback_match_count >= self.feedback_confirm_frames:
            self._set_last_decision(
                'actuator_confirmed', detail, stage=stage_name,
                match_count=self.feedback_match_count)
            return True

        elapsed = time.monotonic() - self.expected_actuator_started_at
        if elapsed >= self.actuator_confirm_timeout:
            self.actuator_recovering = True
            self.actuator_recovery_reason = detail
            self._start_actuator_confirmation(safe_command)
            self._publish_actuator(safe_command)
            self._set_last_decision(
                'actuator_confirmation_timeout', detail, stage=stage_name,
                elapsed=elapsed)
            self._write_data_record(
                'actuator_confirmation_timeout', stage=stage_name,
                elapsed=elapsed, detail=detail)
            rospy.logerr(
                '%s: %s 执行器确认超过 %.1fs，安全停止并等待反馈恢复：%s',
                NODE_NAME, stage_name, self.actuator_confirm_timeout, detail)
        else:
            self._set_last_decision(
                'actuator_confirmation_pending', detail, stage=stage_name,
                match_count=self.feedback_match_count, elapsed=elapsed)
            rospy.loginfo_throttle(
                1.0, '%s: %s 执行器确认中 %d/%d，%s', NODE_NAME,
                stage_name, self.feedback_match_count,
                self.feedback_confirm_frames, detail)
        return False

    def _motion_health(self):
        """判断控制器反馈是否新鲜且未处于 SAFE 状态。"""
        with self.lock:
            state = self.latest_motion_state
            received_at = self.latest_motion_state_at
        if state is None or received_at is None:
            return False, None, '尚未收到 motion 状态'
        age = time.monotonic() - received_at
        if age > self.motion_state_timeout:
            return False, state, 'motion 状态超时 {:.2f}s'.format(age)
        if state.state == MotionState.SAFE:
            return False, state, 'motion_supervisor 处于 SAFE'
        return True, state, 'motion 状态正常'

    def _controller_hovered(self):
        """仅通过控制器 HOVER 反馈判断当前目标是否到达。"""
        healthy, state, detail = self._motion_health()
        if not healthy:
            return False, False, detail
        if self.active_goal_started_at is not None:
            with self.lock:
                received_at = self.latest_motion_state_at
            if received_at < self.active_goal_started_at:
                return False, True, '等待当前目标发布后的 motion 状态'
        if not state.startup_complete:
            return False, True, 'motion_supervisor 尚未完成启动定点'
        if not state.goal_active:
            return False, True, 'motion_supervisor 尚无活动目标'
        if state.state != MotionState.HOVER:
            return False, True, '等待 HOVER，当前状态={}'.format(state.state)
        return True, True, '控制器反馈 HOVER'

    def _pause_stage_timeout(self):
        """执行器未恢复时暂停当前导航阶段的常规超时计时。"""
        if self.state_paused_at is None:
            self.state_paused_at = time.monotonic()

    def _resume_stage_timeout(self):
        """执行器恢复后补偿暂停时间，避免反馈故障触发错误导航超时。"""
        if self.state_paused_at is not None:
            self.state_started_at += time.monotonic() - self.state_paused_at
            self.state_paused_at = None

    def _stage_elapsed(self):
        """返回扣除执行器故障暂停时间后的当前状态持续时间。"""
        now = time.monotonic()
        if self.state_paused_at is not None:
            return self.state_paused_at - self.state_started_at
        return now - self.state_started_at

    def _check_motion_or_fallback(self, stage_name):
        """处理状态反馈异常；超出恢复时间后改为最终安全定点。"""
        hovered, healthy, detail = self._controller_hovered()
        if healthy:
            self.motion_unhealthy_since = None
            self._set_last_decision(
                'motion_feedback', detail, stage=stage_name, hovered=hovered)
            return hovered, False, detail
        if self.motion_unhealthy_since is None:
            self.motion_unhealthy_since = time.monotonic()
            self._write_data_record(
                'motion_feedback_unhealthy', stage=stage_name, detail=detail)
            rospy.logwarn('%s: %s 的 motion 反馈异常：%s', NODE_NAME, stage_name, detail)
        elapsed = time.monotonic() - self.motion_unhealthy_since
        if elapsed >= self.motion_recovery_timeout:
            self._enter_safe_final(
                '{} motion 反馈异常持续 {:.1f}s：{}'.format(
                    stage_name, elapsed, detail))
            return False, True, detail
        rospy.logwarn_throttle(
            1.0, '%s: %s 等待 motion 反馈恢复 %.1f/%.1fs：%s', NODE_NAME,
            stage_name, elapsed, self.motion_recovery_timeout, detail)
        self._set_last_decision(
            'motion_feedback_recovery_pending', detail, stage=stage_name,
            elapsed=elapsed)
        return False, False, detail

    def _enter_safe_final(self, reason):
        """切换至锁存的最终安全目标，并标记本次任务为失败结束。"""
        if self.state == self.SAFE_FINAL:
            return
        self.failure_reason = reason
        self._set_active_goal(self.final_safe_goal, '最终安全定点')
        self.safe_final_started_at = time.monotonic()
        self.safe_final_warned = False
        self._write_data_record('safe_final_entered', reason=reason)
        self._set_state(self.SAFE_FINAL, reason)

    def _navigation_step(self, stage_name, timeout, next_state, next_goal=None):
        """执行一个定点导航阶段，HOVER 或超时后分别转场或安全回退。"""
        safe_command = self._safe_actuator_command()
        self._publish_goal()
        if not self._actuator_gate(safe_command, stage_name):
            self._pause_stage_timeout()
            return
        self._resume_stage_timeout()
        hovered, fallback, detail = self._check_motion_or_fallback(stage_name)
        if fallback:
            return
        if hovered:
            if next_goal is not None:
                self._set_active_goal(next_goal, next_state)
            self._set_state(next_state, '{} 已由 HOVER 确认'.format(stage_name))
            return
        if self._stage_elapsed() >= timeout:
            self._enter_safe_final(
                '{} 超过 {:.1f}s 未收到 HOVER：{}'.format(
                    stage_name, timeout, detail))

    def _hold_step(self, stage_name, hold_seconds, next_state):
        """保持当前目标，只有连续 HOVER 反馈期间才累计保持时长。"""
        safe_command = self._safe_actuator_command()
        self._publish_goal()
        if not self._actuator_gate(safe_command, stage_name):
            self._pause_stage_timeout()
            return
        self._resume_stage_timeout()
        hovered, fallback, detail = self._check_motion_or_fallback(stage_name)
        if fallback:
            return
        if not hovered:
            if self.state == self.SURFACE_HOLD:
                self.surface_hold_started_at = None
            else:
                self.final_hold_started_at = None
            rospy.loginfo_throttle(1.0, '%s: %s，保持计时等待 HOVER', NODE_NAME, detail)
            return

        now = time.monotonic()
        if self.state == self.SURFACE_HOLD:
            if self.surface_hold_started_at is None:
                self.surface_hold_started_at = now
            elapsed = now - self.surface_hold_started_at
        else:
            if self.final_hold_started_at is None:
                self.final_hold_started_at = now
            elapsed = now - self.final_hold_started_at
        if elapsed >= hold_seconds:
            self._set_state(next_state, '{} 保持 {:.1f}s 完成'.format(
                stage_name, hold_seconds))
        else:
            rospy.loginfo_throttle(
                1.0, '%s: %s 保持 %.1f/%.1fs', NODE_NAME,
                stage_name, elapsed, hold_seconds)

    def _safe_final_step(self):
        """持续前往最终安全目标，直至控制器重新确认 HOVER。"""
        safe_command = self._safe_actuator_command()
        self._publish_goal()
        if not self._actuator_gate(safe_command, '最终安全定点'):
            return
        hovered, healthy, detail = self._controller_hovered()
        elapsed = time.monotonic() - self.safe_final_started_at
        if hovered:
            self._finish(False, self.failure_reason)
            return
        if elapsed >= self.final_safe_warning_timeout:
            rospy.logerr_throttle(
                5.0,
                '%s: 最终安全定点已等待 %.1fs，继续等待控制器 HOVER：%s',
                NODE_NAME, elapsed, detail)
        elif not healthy:
            rospy.logwarn_throttle(
                1.0, '%s: 最终安全定点等待 motion 反馈恢复：%s', NODE_NAME, detail)

    def _sampling_position_ready(self):
        """采水前及采水中保持起始点 HOVER，异常时先闭环停止推杆。"""
        self._publish_goal()
        hovered, fallback, detail = self._check_motion_or_fallback('起始点采水定点')
        if fallback:
            return False
        if hovered:
            return True
        self.sampling_started_at = None
        self._actuator_gate(self._safe_actuator_command(), '采水暂停推杆')
        rospy.loginfo_throttle(
            1.0, '%s: %s，推杆保持停止并等待 HOVER', NODE_NAME, detail)
        return False

    def _finish(self, success, reason):
        """停止执行器、发布任务结果并关闭节点。"""
        if self.finished:
            return
        self.finished = True
        self._publish_actuator(self._safe_actuator_command())
        result = 'finished' if success else 'failed'
        message = '{} {}: {}'.format(NODE_NAME, result, reason)
        self._set_last_decision('task_finish', message, success=success)
        self._write_data_record('task_finish', success=success, reason=reason)
        self.finished_pub.publish(String(data=message))
        rospy.loginfo('%s: %s', NODE_NAME, message)
        rospy.signal_shutdown(message)

    def _on_shutdown(self):
        """节点被外部终止时至少下发一次推杆停止和外设安全命令。"""
        try:
            self._publish_actuator(self._safe_actuator_command())
            self._write_data_record('shutdown', finished=self.finished)
        except Exception:
            pass
        with getattr(self, 'data_log_lock', threading.Lock()):
            if getattr(self, 'data_log_file', None) is not None:
                try:
                    self.data_log_file.flush()
                    self.data_log_file.close()
                except OSError:
                    pass
                finally:
                    self.data_log_file = None
        handler = getattr(self, 'ros_log_handler', None)
        if handler is not None:
            try:
                logging.getLogger('rosout').removeHandler(handler)
                handler.close()
            except (OSError, ValueError):
                pass

    def _run_once(self):
        """执行一次 5 Hz 状态机循环。"""
        safe_command = self._safe_actuator_command()
        if self.state == self.WAIT_TF:
            self._publish_actuator(safe_command)
            if self._capture_initial_goals():
                self._set_state(self.WAIT_ACTUATOR_INIT, '已锁存起始 TF')
                return
            if not self.initial_tf_wait_warned and (
                    self._stage_elapsed() >= self.tf_initial_timeout):
                self.initial_tf_wait_warned = True
                rospy.logerr(
                    '%s: 等待首帧 TF 超过 %.1fs，继续等待，不发布运动目标',
                    NODE_NAME, self.tf_initial_timeout)
            return

        if self.state == self.WAIT_ACTUATOR_INIT:
            self._publish_goal()
            if self._actuator_gate(safe_command, '执行器初始化'):
                self._set_active_goal(self.initial_goal, '起始定点')
                self._set_state(self.INITIAL_HOLD, '执行器初始化反馈确认')
            return

        if self.state == self.INITIAL_HOLD:
            self._navigation_step(
                '起始定点', self.initial_hover_timeout,
                self.WAIT_PUSHROD_START)
            return

        if self.state == self.WAIT_PUSHROD_START:
            if not self._sampling_position_ready():
                return
            if self._actuator_gate(self._pushrod_command(), '推杆前进'):
                self.sampling_started_at = time.monotonic()
                self._set_state(self.SAMPLING, '推杆前进反馈确认，开始采水计时')
            return

        if self.state == self.SAMPLING:
            if not self._sampling_position_ready():
                return
            if not self._actuator_gate(self._pushrod_command(), '采水推杆保持'):
                self.sampling_started_at = None
                return
            if self.sampling_started_at is None:
                self.sampling_started_at = time.monotonic()
            elapsed = time.monotonic() - self.sampling_started_at
            if elapsed >= self.pushrod_duration:
                self._set_state(self.WAIT_PUSHROD_STOP, '采水计时完成，等待推杆停止确认')
            else:
                rospy.loginfo_throttle(
                    1.0, '%s: 采水推杆前进 %.1f/%.1fs', NODE_NAME,
                    elapsed, self.pushrod_duration)
            return

        if self.state == self.WAIT_PUSHROD_STOP:
            self._publish_goal()
            if self._actuator_gate(safe_command, '推杆停止'):
                self._set_active_goal(self.delivery_depth_goal, '送水点定深')
                self._set_state(self.DELIVERY_DEPTH, '推杆停止反馈确认')
            return

        if self.state == self.DELIVERY_DEPTH:
            self._navigation_step(
                '送水点定深', self.delivery_depth_timeout,
                self.SURFACE, self.surface_goal)
            return

        if self.state == self.SURFACE:
            self._navigation_step(
                '送水点浮出', self.surface_timeout,
                self.SURFACE_HOLD)
            return

        if self.state == self.SURFACE_HOLD:
            self._hold_step(
                '水面送水', self.surface_hold_seconds, self.DIVE)
            if self.state == self.DIVE:
                self._set_active_goal(self.dive_goal, '送水点下潜')
            return

        if self.state == self.DIVE:
            self._navigation_step(
                '送水点下潜', self.dive_timeout,
                self.RETURN_HOME, self.return_home_goal)
            return

        if self.state == self.RETURN_HOME:
            self._navigation_step(
                '返回起始点', self.return_home_timeout,
                self.FINAL_HOLD)
            return

        if self.state == self.FINAL_HOLD:
            self._hold_step(
                '最终位置悬停', self.final_hold_seconds,
                self.FINAL_STOP_CONFIRM)
            return

        if self.state == self.FINAL_STOP_CONFIRM:
            self._publish_goal()
            if self._actuator_gate(safe_command, '最终推杆停止'):
                self._finish(True, '采水、送水和返航流程完成')
            return

        if self.state == self.SAFE_FINAL:
            self._safe_final_step()
            return

        rospy.logerr('%s: 未知状态 %s，切换至最终安全定点', NODE_NAME, self.state)
        self._enter_safe_final('未知任务状态 {}'.format(self.state))

    def run(self):
        """以固定 5 Hz 执行任务状态机，直到任务完成或 ROS 关闭。"""
        while not rospy.is_shutdown():
            self._run_once()
            self._write_data_record('control_cycle')
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=False)
    try:
        Task2V2().run()
    except rospy.ROSInterruptException:
        pass
