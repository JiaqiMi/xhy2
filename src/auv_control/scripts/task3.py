#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
名称：task3.py
功能：在一个ROS节点内顺序执行箭头、ArUco和方框投放三个子任务
作者：Tangzongle
监听：/vision/arrow/direction、/vision/arrow/target_message
      /vision/aruco/target_message、/vision/rectangle/detections
      /vision/rectangle/target_message、/motion/state、/status/auv
发布：/cmd/motion/goal、/cmd/motion/cancel、/cmd/actuator
      /task3_final/finished
记录：
2026.8.3
    将方框子任务的三维TargetDetection话题传入嵌入式子任务，支持基于map位置的三帧确认流程。
2026.8.3
    增加固定航向/箭头调整双模式，并统一初始、ArUco和返原点三个绝对航向。
2026.8.3
    第二次箭头使用独立搜索路径参数，方框阶段继承固定ArUco航向或第二次箭头最终航向。
2026.8.4
    运行期异常统一先刹停等待HOVER，再按阶段超时固定点或返原点上浮保护继续，禁止中途退出。
2026.8.4
    航向开关扩展为模式1/2/3；仅模式3在ArUco转向完成后跳过第二次箭头。
"""

from datetime import datetime
import json
import logging
import math
import os
import sys
import threading
import time
from collections import Counter, deque

import rospkg
import rosnode
import rospy
import tf
from auv_control.msg import MotionState, TargetDetection
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, String


NODE_NAME = "task3_final"
_MISSING = object()
KEY_LOG_MARKER = "[关键]"
FLOW_LOG_MARKER = "[任务进度]"
PREREQUISITE_LOG_MARKER = "[前置条件]"
HANDOFF_LOG_MARKER = "[阶段交接]"
CONSOLE_EVENT_PHRASES = (
    KEY_LOG_MARKER,
    FLOW_LOG_MARKER,
    PREREQUISITE_LOG_MARKER,
    HANDOFF_LOG_MARKER,
    "[子任务1阶段]",
    "[子任务2阶段]",
    "[子任务3阶段]",
    "三个模型均已就绪",
    "前模型复查通过",
    "任务阶段",
    "状态切换 ",
    "运动状态切换为",
    "识别成功：ArUco",
    "箭头位置候选组确认通过",
    "逐帧候选组确认通过",
    "稳定识别成功",
    "开始灯光阶段",
    "灯光阶段完成",
    "已确认到位",
    "子任务3完成",
    "任务成功",
    "任务结束",
    "转向成功",
)
CONSOLE_PROGRESS_PHRASES = (
    "窗口=",
    "窗口进度=",
    "确认进度=",
    "等待启动定点",
    "模型新帧",
    "等待MotionState.HOVER",
    "等待严格到达",
    "到达判定",
    "到位=",
    "进行中",
    "持续上浮",
    "[灯光反馈]",
    "[执行器反馈]",
    "[执行器状态]",
    "[执行器硬件状态]",
)


class Task3ConsoleFilter(logging.Filter):
    """终端仅保留关键事件，并统一限制进度和重复警告的频率。"""

    def __init__(self, progress_interval, warning_repeat_interval):
        super().__init__()
        self.progress_interval = max(0.0, float(progress_interval))
        self.warning_repeat_interval = max(
            0.0, float(warning_repeat_interval)
        )
        self.last_progress_time = None
        self.last_warning_times = {}

    def filter(self, record):
        if record.levelno >= logging.ERROR:
            return True

        message = record.getMessage()
        if record.levelno >= logging.WARNING:
            warning_key = str(record.msg)
            now = time.monotonic()
            previous = self.last_warning_times.get(warning_key)
            if (
                previous is not None
                and now - previous < self.warning_repeat_interval
            ):
                return False
            self.last_warning_times[warning_key] = now
            return True

        if any(phrase in message for phrase in CONSOLE_EVENT_PHRASES):
            return True

        if not any(
            phrase in message for phrase in CONSOLE_PROGRESS_PHRASES
        ):
            return False

        now = time.monotonic()
        if (
            self.last_progress_time is not None
            and now - self.last_progress_time < self.progress_interval
        ):
            return False
        self.last_progress_time = now
        return True


def install_console_log_filter(
    logger,
    progress_interval,
    warning_repeat_interval,
):
    """只过滤终端处理器，不影响文件日志和ROS话题日志。"""
    console_filter = Task3ConsoleFilter(
        progress_interval,
        warning_repeat_interval,
    )
    filtered_count = 0
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            continue
        handler_name = handler.__class__.__name__.lower()
        if (
            isinstance(handler, logging.StreamHandler)
            or "stream" in handler_name
            or "console" in handler_name
        ):
            handler.addFilter(console_filter)
            filtered_count += 1
    return filtered_count


def save_parameter_snapshot(log_path):
    """把本次launch实际加载的统一配置保存到对应日志旁边。"""
    config_path = str(rospy.get_param("~task3_config_file", "")).strip()
    if not config_path:
        try:
            package_path = rospkg.RosPack().get_path("auv_control")
        except rospkg.ResourceNotFound as error:
            rospy.logwarn(
                "%s：无法定位参数快照源配置：%s",
                NODE_NAME,
                str(error),
            )
            return None
        config_path = os.path.join(package_path, "config", "task3.yaml")

    config_path = os.path.abspath(os.path.expanduser(config_path))
    snapshot_path = os.path.splitext(log_path)[0] + ".yaml"
    try:
        with open(config_path, "r", encoding="utf-8") as source_file:
            config_text = source_file.read()
        with open(snapshot_path, "w", encoding="utf-8", newline="\n") as output_file:
            output_file.write("# 任务3本次运行参数快照；请与同名.log一起保存。\n")
            output_file.write(
                "# snapshot_created_at: {}\n".format(
                    datetime.now().isoformat(timespec="microseconds")
                )
            )
            output_file.write(
                "# corresponding_log: {}\n".format(os.path.basename(log_path))
            )
            output_file.write("# source_config: {}\n\n".format(config_path))
            output_file.write(config_text)
            if config_text and not config_text.endswith("\n"):
                output_file.write("\n")
    except (IOError, OSError, UnicodeError) as error:
        rospy.logwarn(
            "%s：参数快照保存失败，源=%s，目标=%s：%s",
            NODE_NAME,
            config_path,
            snapshot_path,
            str(error),
        )
        return None
    return snapshot_path


def configure_file_logging():
    """详细日志写单文件，终端按配置只显示关键事件。"""
    log_directory = os.path.abspath(os.path.expanduser(str(
        rospy.get_param(
            "/task3_final/log_directory",
            "~/.ros/auv_logs/task3",
        )
    )))
    console_key_only = bool(rospy.get_param(
        "/task3_final/console_key_only",
        True,
    ))
    progress_interval = float(rospy.get_param(
        "/task3_final/console_progress_interval",
        0.5,
    ))
    warning_repeat_interval = float(rospy.get_param(
        "/task3_final/console_warning_repeat_interval",
        3.0,
    ))
    logger = logging.getLogger("rosout")
    try:
        os.makedirs(log_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = os.path.join(
            log_directory,
            "task3_{}.log".format(timestamp),
        )
        handler = logging.FileHandler(
            log_path,
            mode="a",
            encoding="utf-8",
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
    except (IOError, OSError) as error:
        rospy.logerr(
            "%s：无法创建文件日志目录%s：%s",
            NODE_NAME,
            log_directory,
            str(error),
        )
        return None

    parameter_snapshot_path = save_parameter_snapshot(log_path)
    filtered_count = 0
    if console_key_only:
        filtered_count = install_console_log_filter(
            logger,
            progress_interval,
            warning_repeat_interval,
        )
    rospy.loginfo(
        (
            "%s：%s 整合任务详细日志已启用：%s；"
            "本次参数快照=%s；"
            "终端关键日志过滤=%s，进度日志间隔=%.2fs，"
            "终端处理器=%d"
        ),
        NODE_NAME,
        KEY_LOG_MARKER,
        log_path,
        parameter_snapshot_path or "保存失败，请检查上方WARNING",
        "开启" if console_key_only else "关闭",
        progress_interval,
        filtered_count,
    )
    return log_path


def load_task_params(namespace):
    """从统一YAML加载后的ROS命名空间读取一个子任务参数段。"""
    parameters = rospy.get_param(namespace, None)
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(
            "统一任务3配置缺少有效参数段：{}".format(namespace)
        )
    return dict(parameters)


class ScopedRospy:
    """让三个子任务在同一ROS节点中读取各自独立的私有参数。"""

    def __init__(
        self,
        real_rospy,
        parameters,
        label,
        internal_finished_topic,
    ):
        self._real_rospy = real_rospy
        self._parameters = parameters
        self._label = label
        self._internal_finished_topic = internal_finished_topic

    def get_param(self, name, default=_MISSING):
        if str(name).startswith("~"):
            key = str(name)[1:]
            if key in self._parameters:
                return self._parameters[key]
        if default is _MISSING:
            return self._real_rospy.get_param(name)
        return self._real_rospy.get_param(name, default)

    def signal_shutdown(self, reason):
        rospy.logdebug(
            "%s：%s子函数请求结束：%s",
            NODE_NAME,
            self._label,
            str(reason),
        )

    def Publisher(self, name, *args, **kwargs):
        topic = (
            self._internal_finished_topic
            if str(name) == "/finished"
            else name
        )
        return self._real_rospy.Publisher(topic, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real_rospy, name)


def _load_subtask_modules(package_path):
    module_files = (
        "test_task3_1_acquire_area.py",
        "test_task3_2_get_task.py",
        "test_task3_3_inspect_and_drop.py",
    )
    install_prefix = os.path.dirname(os.path.dirname(package_path))
    candidate_paths = (
        os.path.join(package_path, "test"),
        os.path.dirname(os.path.realpath(__file__)),
        os.path.dirname(os.path.abspath(sys.argv[0])),
        os.path.join(install_prefix, "lib", "auv_control"),
    )
    import_path = next(
        (
            path for path in candidate_paths
            if all(
                os.path.isfile(os.path.join(path, filename))
                for filename in module_files
            )
        ),
        None,
    )
    if import_path is None:
        raise ImportError(
            "找不到三个子任务脚本，已检查：{}".format(
                "，".join(candidate_paths)
            )
        )
    if import_path not in sys.path:
        sys.path.insert(0, import_path)
    rospy.loginfo(
        "%s：从%s导入三个子任务模块",
        NODE_NAME,
        import_path,
    )

    import test_task3_1_acquire_area as task1_module
    import test_task3_2_get_task as task2_module
    import test_task3_3_inspect_and_drop as task3_module

    return task1_module, task2_module, task3_module


class Task3Final:
    MODEL_TYPES = {
        "arrow": String,
        "aruco": TargetDetection,
        "rectangle": String,
    }
    FINAL_TIMEOUT_TARGET_LABELS = {
        "aruco": "ArUco识别点",
        "box": "彩色方框点",
    }
    SUBTASK3_UNUSED_MODEL_NODES = (
        "/yolo_arrow_pose_detector",
        "/task3_final/aruco_pipeline/fisheye_aruco_node",
    )
    MODEL_SHUTDOWN_TIMEOUT = 3.0

    def __init__(self):
        package_path = rospkg.RosPack().get_path("auv_control")
        self.task1_module, self.task2_module, self.task3_module = (
            _load_subtask_modules(package_path)
        )
        self.fixed_depth_m = float(rospy.get_param(
            "/task3_target_depth_m", 0.60
        ))
        if not math.isfinite(self.fixed_depth_m) or self.fixed_depth_m <= 0.0:
            raise ValueError("task3_target_depth_m必须是大于0的有限数")
        self.fixed_map_z = -self.fixed_depth_m
        heading_mode = rospy.get_param("/task3_heading_mode", 1)
        if type(heading_mode) is not int or heading_mode not in (1, 2, 3):
            raise ValueError("task3_heading_mode必须是整数1、2或3")
        self.heading_mode = heading_mode
        self.fixed_heading_enabled = heading_mode in (2, 3)
        self.skip_second_arrow = heading_mode == 3
        self.mission_stage_count = 3 if self.skip_second_arrow else 4
        self.heading_mode_label = {
            1: "箭头调整",
            2: "固定航向",
            3: "固定航向并跳过第二次箭头",
        }[heading_mode]
        self.initial_yaw_deg = float(rospy.get_param(
            "/task3_initial_yaw_deg", 210.0
        ))
        self.aruco_yaw_deg = float(rospy.get_param(
            "/task3_aruco_yaw_deg", 120.0
        ))
        self.return_origin_yaw_deg = float(rospy.get_param(
            "/task3_return_origin_yaw_deg", 30.0
        ))
        heading_values = {
            "task3_initial_yaw_deg": self.initial_yaw_deg,
            "task3_aruco_yaw_deg": self.aruco_yaw_deg,
            "task3_return_origin_yaw_deg": self.return_origin_yaw_deg,
        }
        for name, value in heading_values.items():
            if not math.isfinite(value) or value < 0.0 or value >= 360.0:
                raise ValueError("{}必须在[0, 360)度范围内".format(name))
        self.initial_target_yaw = self.angle_difference(
            math.radians(self.initial_yaw_deg), 0.0
        )
        self.aruco_target_yaw = self.angle_difference(
            math.radians(self.aruco_yaw_deg), 0.0
        )
        self.return_origin_target_yaw = self.angle_difference(
            math.radians(self.return_origin_yaw_deg), 0.0
        )

        self.task1_params = load_task_params(
            "/test_task3_1_acquire_area"
        )
        self.task2_params = load_task_params(
            "/test_task3_2_get_task"
        )
        self.task3_params = load_task_params(
            "/test_task3_3_inspect_and_drop"
        )
        self.post_drop_step_timeout = float(
            self.task3_params.get("post_drop_step_timeout", 90.0)
        )
        self.post_drop_ascent_target_z = float(
            self.task3_params.get("post_drop_ascent_target_z", -1.3)
        )
        if (
            not math.isfinite(self.post_drop_step_timeout)
            or self.post_drop_step_timeout <= 0.0
        ):
            raise ValueError("post_drop_step_timeout必须是大于0的有限数")
        if (
            not math.isfinite(self.post_drop_ascent_target_z)
            or self.post_drop_ascent_target_z >= self.fixed_map_z
        ):
            raise ValueError(
                "post_drop_ascent_target_z必须是有限数，且必须小于"
                "任务运行深度对应的map/NED z"
            )
        self.arrow_topic = str(rospy.get_param(
            "~arrow_topic",
            "/vision/arrow/direction",
        )).strip()
        self.arrow_target_topic = str(rospy.get_param(
            "~arrow_target_topic",
            "/vision/arrow/target_message",
        )).strip()
        self.aruco_topic = str(rospy.get_param(
            "~aruco_topic",
            "/vision/aruco/target_message",
        )).strip()
        self.rectangle_topic = str(rospy.get_param(
            "~rectangle_topic",
            "/vision/rectangle/detections",
        )).strip()
        self.rectangle_target_topic = str(rospy.get_param(
            "~rectangle_target_topic",
            "/vision/rectangle/target_message",
        )).strip()
        self.motion_goal_topic = str(rospy.get_param(
            "~motion_goal_topic",
            "/cmd/motion/goal",
        )).strip()
        self.motion_cancel_topic = str(rospy.get_param(
            "~motion_cancel_topic",
            "/cmd/motion/cancel",
        )).strip()
        self.motion_state_topic = str(rospy.get_param(
            "~motion_state_topic",
            "/motion/state",
        )).strip()
        self.status_topic = str(rospy.get_param(
            "~status_topic",
            "/status/auv",
        )).strip()
        self.actuator_topic = str(rospy.get_param(
            "~actuator_topic",
            "/cmd/actuator",
        )).strip()
        self.sequence_finished_topic = str(rospy.get_param(
            "~sequence_finished_topic",
            "/task3_final/finished",
        )).strip()

        self.model_ready_timeout = float(rospy.get_param(
            "~model_ready_timeout",
            90.0,
        ))
        self.model_required_frames = int(rospy.get_param(
            "~model_required_frames",
            3,
        ))
        self.model_output_timeout = float(rospy.get_param(
            "~model_output_timeout",
            2.0,
        ))
        self.handoff_stable_seconds = float(rospy.get_param(
            "~handoff_stable_seconds",
            1.0,
        ))
        self.motion_state_timeout = float(rospy.get_param(
            "/task3_protection/motion_feedback_timeout",
            3.0,
        ))
        self.cancel_recovery_timeout = float(rospy.get_param(
            "/task3_protection/cancel_recovery_timeout",
            30.0,
        ))
        self.startup_tf_timeout = float(rospy.get_param(
            "~startup_tf_timeout",
            8.0,
        ))
        self.rate_hz = float(rospy.get_param("~rate", 10.0))
        self.arrow1_timeout_seconds = float(rospy.get_param(
            "/task3_final/arrow1_timeout_seconds",
            self.task1_params.get("max_wait_seconds", 120.0),
        ))
        self.arrow2_timeout_seconds = float(rospy.get_param(
            "/task3_final/arrow2_timeout_seconds",
            self.task1_params.get("max_wait_seconds", 120.0),
        ))
        self.box_timeout_seconds = float(rospy.get_param(
            "/task3_final/box_timeout_seconds",
            self.task3_params.get("max_wait_seconds", 120.0),
        ))
        arrow2_search_parameter_names = (
            "search_initial_forward_distance",
            "search_lateral_distance",
            "search_second_forward_distance",
            "search_third_forward_distance",
        )
        self.arrow2_search_params = {
            name: float(rospy.get_param(
                "/task3_final/arrow2_{}".format(name),
                self.task1_params.get(name),
            ))
            for name in arrow2_search_parameter_names
        }
        self.final_timeout_move_timeout = float(rospy.get_param(
            "/task3_final/final_timeout_move_timeout",
            120.0,
        ))
        self.final_timeout_arrival_stable_seconds = float(rospy.get_param(
            "/task3_final/final_timeout_arrival_stable_seconds",
            2.0,
        ))
        self.final_timeout_targets = self.load_final_timeout_targets()

        if self.model_required_frames <= 0:
            raise ValueError("model_required_frames必须大于0")
        if self.rate_hz <= 0.0:
            raise ValueError("rate必须大于0")
        if min(
            self.motion_state_timeout,
            self.cancel_recovery_timeout,
        ) <= 0.0:
            raise ValueError("运动反馈和异常悬停保护时间必须大于0")
        if (
            not all(math.isfinite(value) for value in (
                self.arrow1_timeout_seconds,
                self.arrow2_timeout_seconds,
                self.box_timeout_seconds,
            ))
            or min(
                self.arrow1_timeout_seconds,
                self.arrow2_timeout_seconds,
                self.box_timeout_seconds,
            ) <= 0.0
        ):
            raise ValueError("三个阶段最终超时必须是大于0的有限数")
        if min(
            self.final_timeout_move_timeout,
            self.final_timeout_arrival_stable_seconds,
        ) <= 0.0:
            raise ValueError("最终超时处理相关时间参数必须大于0")
        if (
            not all(math.isfinite(value) for value in (
                self.arrow2_search_params.values()
            ))
            or min(self.arrow2_search_params.values()) <= 0.0
        ):
            raise ValueError("第二次箭头搜索距离必须是大于0的有限数")

        common_topics = {
            "motion_goal_topic": self.motion_goal_topic,
            "motion_cancel_topic": self.motion_cancel_topic,
            "motion_state_topic": self.motion_state_topic,
        }
        self.task1_params.update(common_topics)
        self.task2_params.update(common_topics)
        self.task3_params.update(common_topics)
        self.task1_params["arrow_topic"] = self.arrow_topic
        self.task1_params["arrow_target_topic"] = self.arrow_target_topic
        self.task1_params["status_topic"] = self.status_topic
        self.task2_params["aruco_topic"] = self.aruco_topic
        self.task2_params["actuator_topic"] = self.actuator_topic
        self.task3_params["model_detection_topic"] = self.rectangle_topic
        self.task3_params["model_target_topic"] = self.rectangle_target_topic
        self.task3_params["status_topic"] = self.status_topic
        self.task3_params["actuator_topic"] = self.actuator_topic
        # 整合模式统一在阶段之间完成HOVER交接，取消子任务内部重复的
        # 结束保持和下一阶段启动悬停。第一次箭头的启动悬停由总调度
        # 锁存固定点完成；三个模型已统一预热，箭头模型仍在子任务启动前复查。
        self.task1_params["initial_hover_seconds"] = float(rospy.get_param(
            "~subtask1_initial_hover_seconds", 0.0
        ))
        self.task1_params["final_hold_seconds"] = float(rospy.get_param(
            "~subtask1_final_hold_seconds", 0.0
        ))
        self.task2_params["initial_hover_seconds"] = float(rospy.get_param(
            "~subtask2_initial_hover_seconds", 0.0
        ))
        self.task2_params["turn_hold_seconds"] = float(rospy.get_param(
            "~subtask2_turn_hold_seconds", 0.0
        ))
        self.task3_params["auto_initial_hover_seconds"] = float(
            rospy.get_param("~subtask3_initial_hover_seconds", 0.0)
        )

        self.task1_module.rospy = ScopedRospy(
            rospy,
            self.task1_params,
            "子任务1",
            "/task3_final/internal/subtask1/finished",
        )
        self.task2_module.rospy = ScopedRospy(
            rospy,
            self.task2_params,
            "子任务2",
            "/task3_final/internal/subtask2/finished",
        )
        self.task3_module.rospy = ScopedRospy(
            rospy,
            self.task3_params,
            "子任务3",
            "/task3_final/internal/subtask3/finished",
        )

        self.EmbeddedTask1 = self._make_embedded_task1()
        self.EmbeddedTask2 = self._make_embedded_task2()
        self.EmbeddedTask3 = self._make_embedded_task3()

        self.rate = rospy.Rate(self.rate_hz)
        self.tf_listener = tf.TransformListener()
        self.goal_pub = rospy.Publisher(
            self.motion_goal_topic,
            PoseStamped,
            queue_size=1,
        )
        self.cancel_pub = rospy.Publisher(
            self.motion_cancel_topic,
            Empty,
            queue_size=1,
        )
        self.finished_pub = rospy.Publisher(
            self.sequence_finished_topic,
            String,
            queue_size=10,
        )

        self.motion_lock = threading.Lock()
        self.latest_motion_state = None
        self.latest_motion_state_wall_time = None
        self.startup_hold_goal = None
        self.motion_sub = rospy.Subscriber(
            self.motion_state_topic,
            MotionState,
            self.motion_state_callback,
            queue_size=20,
        )

        self.model_lock = threading.Lock()
        self.aruco_history_window_size = int(
            self.task2_params.get("recognition_window_size", 10)
        )
        self.aruco_history_required_count = int(
            self.task2_params.get("required_match_count", 3)
        )
        self.aruco_history_min_confidence = float(
            self.task2_params.get("min_confidence", 0.5)
        )
        if self.aruco_history_window_size <= 0:
            raise ValueError("recognition_window_size必须大于0")
        if (
            self.aruco_history_required_count <= 0
            or self.aruco_history_required_count
            > self.aruco_history_window_size
        ):
            raise ValueError(
                "required_match_count必须大于0且不能超过recognition_window_size"
            )
        if not 0.0 <= self.aruco_history_min_confidence <= 1.0:
            raise ValueError("min_confidence必须在0到1之间")
        self.aruco_history_window = deque(
            maxlen=self.aruco_history_window_size
        )
        self.aruco_history_marker_id = None
        self.aruco_history_color = None
        self.model_counts = {
            "arrow": 0,
            "aruco": 0,
            "rectangle": 0,
        }
        self.raw_frame_counts = {
            "arrow_direction": 0,
            "arrow_target": 0,
            "aruco_target": 0,
            "rectangle_detections": 0,
        }
        self.model_latest_wall_time = {
            "arrow": None,
            "aruco": None,
            "rectangle": None,
        }
        self.model_last_source_key = {
            "arrow": None,
        }
        self.arrow_target_model_sub = rospy.Subscriber(
            self.arrow_target_topic,
            TargetDetection,
            self.arrow_target_model_callback,
            queue_size=100,
        )
        self.model_subscribers = [
            rospy.Subscriber(
                self.arrow_topic,
                String,
                self.arrow_model_callback,
                queue_size=100,
            ),
            rospy.Subscriber(
                self.aruco_topic,
                TargetDetection,
                self.aruco_model_callback,
                queue_size=100,
            ),
            rospy.Subscriber(
                self.rectangle_topic,
                String,
                self.rectangle_model_callback,
                queue_size=100,
            ),
            self.arrow_target_model_sub,
        ]

        self.finished = False
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(
            (
                "%s：%s 整合节点启动，参数来自统一config/task3.yaml："
                "子任务1=%d项，子任务2=%d项，子任务3=%d项；"
                "统一固定深度=%.3fm（map目标z=%.3f）；"
                "航向模式=%d（%s），初始/ArUco/原点绝对航向=%.1f/%.1f/%.1fdeg"
            ),
            NODE_NAME,
            KEY_LOG_MARKER,
            len(self.task1_params),
            len(self.task2_params),
            len(self.task3_params),
            self.fixed_depth_m,
            self.fixed_map_z,
            self.heading_mode,
            self.heading_mode_label,
            self.initial_yaw_deg,
            self.aruco_yaw_deg,
            self.return_origin_yaw_deg,
        )
        rospy.loginfo(
            (
                "%s：模型话题：箭头方向=%s，箭头三维位置=%s，ArUco=%s，方框=%s；"
                "阶段间不再启动或关闭子任务launch"
            ),
            NODE_NAME,
            self.arrow_topic,
            self.arrow_target_topic,
            self.aruco_topic,
            self.rectangle_topic,
        )
        rospy.loginfo(
            (
                "%s：整合模式启动保护：等待三个模型链路全部就绪期间，"
                "持续发布固定深度和初始航向目标；"
                "箭头方向唯一推理帧、箭头三维发布端、ArUco和方框就绪，"
                "且目标进入HOVER后才开始子任务；"
                "进入阶段后由各子任务自行处理模型新鲜度；"
                "阶段恢复停稳需连续保持 %.1fs"
            ),
            NODE_NAME,
            self.handoff_stable_seconds,
        )
        rospy.logwarn(
            (
                "%s：单层最终超时保护（均从机器人首次执行运动动作开始计时）："
                "箭头1=%.1fs后前往ArUco点；箭头2=%.1fs后前往方框点；"
                "方框=%.1fs后原地转向、"
                "返回map原点并上浮；所有阶段只执行一次，不cancel恢复、不重试"
            ),
            NODE_NAME,
            self.arrow1_timeout_seconds,
            self.arrow2_timeout_seconds,
            self.box_timeout_seconds,
        )
        for key, target in self.final_timeout_targets.items():
            target_yaw = (
                self.initial_yaw_deg
                if key == "aruco"
                else self.aruco_yaw_deg
            )
            rospy.logwarn(
                (
                    "%s：最终超时目标[%s] map/NED="
                    "(N=%.3f,E=%.3f,D=%.3f)，yaw=%.1fdeg"
                ),
                NODE_NAME,
                self.FINAL_TIMEOUT_TARGET_LABELS[key],
                target["N"],
                target["E"],
                self.fixed_map_z,
                target_yaw,
            )

    def load_final_timeout_targets(self):
        """解析箭头最终超时后使用的下一阶段map绝对目标。"""
        raw_targets = rospy.get_param(
            "/task3_final/final_timeout_targets",
            {},
        )
        if not isinstance(raw_targets, dict):
            raise ValueError("final_timeout_targets必须是字典")

        targets = {}
        for key in self.FINAL_TIMEOUT_TARGET_LABELS:
            raw_target = raw_targets.get(key)
            if not isinstance(raw_target, dict):
                raise ValueError(
                    "超时跳点目标{}尚未配置".format(key)
                )
            fields = ("N", "E")
            try:
                target = {
                    field: float(raw_target[field])
                    for field in fields
                }
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    "超时跳点目标{}必须填写数字{}".format(
                        key,
                        "、".join(fields),
                    )
                )
            if not all(math.isfinite(value) for value in target.values()):
                raise ValueError(
                    "超时跳点目标{}包含非有限值".format(key)
                )
            targets[key] = target
        return targets

    def _make_embedded_task1(self):
        parent = self.task1_module.Task3AcquireAreaTest

        class EmbeddedTask1(parent):
            def __init__(self):
                self.embedded_success = None
                self.embedded_detail = ""
                self.embedded_timed_out = False
                self.embedded_active = True
                super().__init__()

            def finish_task(self, success, detail):
                if self.task_finished:
                    return
                self.task_finished = True
                self.embedded_success = bool(success)
                self.embedded_detail = str(detail)
                state = "finished" if success else "failed"
                self.finished_pub.publish(String(data="{} {}: {}".format(
                    "test_task3_1_acquire_area",
                    state,
                    detail,
                )))
                if success:
                    rospy.loginfo(
                        "%s：第一次/第二次箭头子函数完成：%s；"
                        "保持最后一个motion目标，不发布cancel",
                        NODE_NAME,
                        detail,
                    )
                else:
                    rospy.logwarn(
                        "%s：第一次/第二次箭头子函数结束：%s；"
                        "等待总调度执行唯一最终超时处理，不发布cancel",
                        NODE_NAME,
                        detail,
                    )

            def on_shutdown(self):
                if self.embedded_active:
                    super().on_shutdown()

            def run(self):
                while not rospy.is_shutdown() and not self.task_finished:
                    timeout_elapsed = self.motion_timeout_elapsed()
                    if (
                        timeout_elapsed is not None
                        and timeout_elapsed >= self.max_wait_seconds
                    ):
                        self.embedded_timed_out = True
                        self.finish_task(
                            False,
                            "机器人开始运动后，搜索和对准累计达到{:.1f}s".format(
                                timeout_elapsed
                            ),
                        )
                        break

                    if not self.initialize_control():
                        self.rate.sleep()
                        continue
                    if not self.handle_motion_health():
                        self.rate.sleep()
                        continue
                    if self.get_recent_status("任务运行安全检查") is None:
                        self.finish_task(False, "/status/auv反馈超时")
                        break

                    self.control_current_state()

                    if not self.task_finished:
                        self.publish_active_goal()
                    self.rate.sleep()

                if self.embedded_success is None:
                    self.embedded_success = False
                    self.embedded_detail = "ROS关闭或子任务1未返回结果"
                return self.embedded_success, self.embedded_detail

        return EmbeddedTask1

    def _make_embedded_task2(self):
        parent = self.task2_module.Task3GetTaskTest

        class EmbeddedTask2(parent):
            def __init__(self):
                self.embedded_success = None
                self.embedded_detail = ""
                self.embedded_active = True
                super().__init__()

            def finalize_task(self, success, detail):
                self.embedded_success = bool(success)
                self.embedded_detail = str(detail)
                self.accept_detections = False
                self.publish_lights("off")
                if success:
                    self.publish_position_hold("任务结束前保持最终运动目标")
                else:
                    rospy.logwarn(
                        "%s：ArUco子函数异常结束；保留当前运动目标，"
                        "不发布cancel",
                        NODE_NAME,
                    )
                state = "finished" if success else "failed"
                self.finished_pub.publish(String(data="{} {}: {}".format(
                    "test_task3_2_get_task",
                    state,
                    detail,
                )))
                self.outputs_closed = True

            def on_shutdown(self):
                if self.embedded_active:
                    super().on_shutdown()

        return EmbeddedTask2

    def _make_embedded_task3(self):
        parent = self.task3_module.Task3InspectAndDropTest

        class EmbeddedTask3(parent):
            def __init__(self):
                self.embedded_success = None
                self.embedded_detail = ""
                self.embedded_timed_out = False
                self.embedded_active = True
                super().__init__()

            def finish_task(self, success, reason):
                if self.finished:
                    return
                self.finished = True
                reason_text = str(reason)
                self.embedded_success = bool(success)
                self.embedded_detail = reason_text
                timeout_elapsed = self.motion_timeout_elapsed()
                self.embedded_timed_out = (
                    not success
                    and timeout_elapsed is not None
                    and timeout_elapsed >= self.max_wait_seconds
                    and bool(self.max_wait_timed_out)
                )
                self.publish_actuator(self.clamp_closed, "off")
                state = "finished" if success else "failed"
                self.finished_pub.publish(String(data="{} {}: {}".format(
                    "test_task3_3_inspect_and_drop",
                    state,
                    reason_text,
                )))
                rospy.loginfo(
                    "%s：方框子函数%s：%s；保持最后一个motion目标，"
                    "不发布cancel",
                    NODE_NAME,
                    "完成" if success else "结束",
                    reason_text,
                )

            def on_shutdown(self):
                if self.embedded_active:
                    super().on_shutdown()

        return EmbeddedTask3

    def motion_state_callback(self, message):
        with self.motion_lock:
            self.latest_motion_state = message
            self.latest_motion_state_wall_time = time.monotonic()

    def _record_model_frame(self, role, source_key=None):
        with self.model_lock:
            if source_key is not None:
                if self.model_last_source_key.get(role) == source_key:
                    return False
                self.model_last_source_key[role] = source_key
            self.model_counts[role] += 1
            self.model_latest_wall_time[role] = time.monotonic()
        return True

    @staticmethod
    def arrow_direction_source_key(message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return None
        if not isinstance(payload, dict):
            return None
        if "keypoint_stamp_nsec" in payload:
            stamp_nsec = payload.get("keypoint_stamp_nsec")
            if stamp_nsec is None or not str(stamp_nsec).strip():
                return None
            return "nsec:{}".format(str(stamp_nsec).strip())
        try:
            stamp_sec = float(payload.get("stamp"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(stamp_sec) or stamp_sec <= 0.0:
            return None
        return "sec:{:.9f}".format(stamp_sec)

    def _next_raw_frame_index(self, role):
        with self.model_lock:
            self.raw_frame_counts[role] += 1
            return self.raw_frame_counts[role]

    def log_raw_string_frame(self, role, topic, message):
        """把String话题的每一帧原始内容写入现有task3日志。"""
        frame_index = self._next_raw_frame_index(role)
        rospy.loginfo(
            "%s：[原始识别帧][%s #%d] receive_ros_time=%.9f，topic=%s，raw=%s",
            NODE_NAME,
            role,
            frame_index,
            rospy.Time.now().to_sec(),
            topic,
            json.dumps(str(message.data), ensure_ascii=False),
        )

    def log_raw_target_frame(self, role, topic, message):
        """把TargetDetection消息的全部字段写入现有task3日志。"""
        frame_index = self._next_raw_frame_index(role)
        header = message.pose.header
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        raw_message = {
            "pose": {
                "header": {
                    "seq": int(getattr(header, "seq", 0)),
                    "stamp": header.stamp.to_sec(),
                    "frame_id": str(header.frame_id),
                },
                "position": {
                    "x": float(position.x),
                    "y": float(position.y),
                    "z": float(position.z),
                },
                "orientation": {
                    "x": float(orientation.x),
                    "y": float(orientation.y),
                    "z": float(orientation.z),
                    "w": float(orientation.w),
                },
            },
            "conf": float(message.conf),
            "type": str(message.type),
            "class_name": str(message.class_name),
        }
        rospy.loginfo(
            "%s：[原始识别帧][%s #%d] receive_ros_time=%.9f，topic=%s，raw=%s",
            NODE_NAME,
            role,
            frame_index,
            rospy.Time.now().to_sec(),
            topic,
            json.dumps(
                raw_message,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    def arrow_model_callback(self, message):
        self.log_raw_string_frame(
            "arrow_direction", self.arrow_topic, message
        )
        source_key = self.arrow_direction_source_key(message)
        if source_key is not None:
            self._record_model_frame("arrow", source_key)

    def arrow_target_model_callback(self, message):
        self.log_raw_target_frame(
            "arrow_target", self.arrow_target_topic, message
        )

    def aruco_model_callback(self, message):
        self.log_raw_target_frame(
            "aruco_target", self.aruco_topic, message
        )
        try:
            confidence = float(message.conf)
        except (TypeError, ValueError):
            confidence = float("nan")
        detection_type = str(message.type).strip().lower()
        marker_id = (
            self.task2_module.Task3GetTaskTest.marker_id_from_detection(
                message
            )
        )
        if (
            detection_type == "aruco_not_detected"
            or marker_id == -1
            or not math.isfinite(confidence)
            or confidence < self.aruco_history_min_confidence
            or marker_id
            not in self.task2_module.Task3GetTaskTest.COLOR_BY_MARKER
        ):
            marker_id = None

        changed = False
        confirmed_count = 0
        with self.model_lock:
            self.model_counts["aruco"] += 1
            self.model_latest_wall_time["aruco"] = time.monotonic()
            self.aruco_history_window.append(marker_id)
            counts = Counter(
                value
                for value in self.aruco_history_window
                if value is not None
            )
            if counts:
                candidate, confirmed_count = min(
                    counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
                if (
                    confirmed_count >= self.aruco_history_required_count
                    and candidate != self.aruco_history_marker_id
                ):
                    self.aruco_history_marker_id = candidate
                    self.aruco_history_color = (
                        self.task2_module.Task3GetTaskTest.color_for_marker(
                            candidate
                        )
                    )
                    changed = True

        if changed:
            rospy.loginfo(
                (
                    "%s：已记录ArUco历史结果：最近%d帧中ID=%d出现%d帧，"
                    "颜色=%s；子任务2实时识别超时后可使用该结果"
                ),
                NODE_NAME,
                self.aruco_history_window_size,
                self.aruco_history_marker_id,
                confirmed_count,
                self.aruco_history_color,
            )

    def rectangle_model_callback(self, message):
        self.log_raw_string_frame(
            "rectangle_detections", self.rectangle_topic, message
        )
        self._record_model_frame("rectangle")

    def _model_snapshot(self):
        with self.model_lock:
            return (
                dict(self.model_counts),
                dict(self.model_latest_wall_time),
            )

    def get_aruco_history_result(self):
        with self.model_lock:
            return (
                self.aruco_history_marker_id,
                self.aruco_history_color,
            )

    def capture_startup_hold_goal(self):
        """只锁存一次启动位置，并使用任务3统一初始航向。"""
        deadline = time.monotonic() + self.startup_tf_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                translation, rotation = self.tf_listener.lookupTransform(
                    "map",
                    "base_link",
                    rospy.Time(0),
                )
            except tf.Exception as error:
                rospy.logwarn_throttle(
                    1.0,
                    "%s：等待TF map -> base_link以锁存启动定点：%s",
                    NODE_NAME,
                    str(error),
                )
                self.rate.sleep()
                continue

            values = tuple(translation) + tuple(rotation)
            if not all(math.isfinite(float(value)) for value in values):
                rospy.logwarn_throttle(
                    1.0,
                    "%s：忽略包含无效值的启动TF",
                    NODE_NAME,
                )
                self.rate.sleep()
                continue

            goal = PoseStamped()
            goal.header.frame_id = "map"
            goal.pose.position.x = float(translation[0])
            goal.pose.position.y = float(translation[1])
            goal.pose.position.z = self.fixed_map_z
            half_yaw = self.initial_target_yaw * 0.5
            goal.pose.orientation.z = math.sin(half_yaw)
            goal.pose.orientation.w = math.cos(half_yaw)
            rospy.loginfo(
                (
                    "%s：已锁存任务3启动目标：map=(%.3f,%.3f,%.3f)，"
                    "初始航向=%.1fdeg；模型加载期间不会刷新该目标"
                ),
                NODE_NAME,
                goal.pose.position.x,
                goal.pose.position.y,
                goal.pose.position.z,
                self.initial_yaw_deg,
            )
            return goal
        return None

    def publish_startup_hold_goal(self, goal):
        goal.header.stamp = rospy.Time.now()
        self.goal_pub.publish(goal)

    def startup_hold_ready(self, goal):
        with self.motion_lock:
            state = self.latest_motion_state
            state_time = self.latest_motion_state_wall_time
        if (
            state is None
            or state_time is None
            or time.monotonic() - state_time > self.motion_state_timeout
        ):
            return False, "运动反馈未收到或已超时"
        if not state.startup_complete:
            return False, "motion_supervisor尚未完成启动"
        if state.state != MotionState.HOVER:
            return False, "当前state={}，等待HOVER".format(state.state)
        if not state.goal_active:
            return False, "motion_supervisor尚无活动目标"

        position_error = math.hypot(
            state.goal.pose.position.x - goal.pose.position.x,
            state.goal.pose.position.y - goal.pose.position.y,
        )
        depth_error = abs(
            state.goal.pose.position.z - goal.pose.position.z
        )
        state_goal_yaw = self.yaw_from_pose(state.goal.pose)
        expected_yaw = self.yaw_from_pose(goal.pose)
        yaw_error = abs(self.angle_difference(
            state_goal_yaw,
            expected_yaw,
        ))
        position_tolerance = float(
            self.task1_params["goal_match_position_tolerance"]
        )
        depth_tolerance = float(
            self.task1_params["goal_match_depth_tolerance"]
        )
        yaw_tolerance = math.radians(float(
            self.task1_params["goal_match_yaw_tolerance_deg"]
        ))
        if position_error > position_tolerance:
            return False, "活动目标水平偏差{:.3f}m".format(position_error)
        if depth_error > depth_tolerance:
            return False, "活动目标深度偏差{:.3f}m".format(depth_error)
        if yaw_error > yaw_tolerance:
            return False, "活动目标航向偏差{:.1f}deg".format(
                math.degrees(yaw_error)
            )
        return True, "固定启动目标已进入HOVER"

    def wait_for_all_models(self, hold_goal):
        """持续发布启动目标，等待三个模型链路就绪且目标进入HOVER。"""
        started_at = time.monotonic()
        while not rospy.is_shutdown():
            self.publish_startup_hold_goal(hold_goal)
            counts, latest = self._model_snapshot()
            now = time.monotonic()
            ready = {}
            ages = {}
            for role in self.MODEL_TYPES:
                last_time = latest[role]
                age = (
                    float("inf")
                    if last_time is None
                    else now - last_time
                )
                ages[role] = age
                ready[role] = (
                    counts[role] >= self.model_required_frames
                    and age <= self.model_output_timeout
                )
            arrow_target_connected = (
                self.arrow_target_model_sub.get_num_connections() > 0
            )
            hold_ready, hold_detail = self.startup_hold_ready(hold_goal)
            if (
                all(ready.values())
                and arrow_target_connected
                and hold_ready
            ):
                rospy.loginfo(
                    (
                        "%s：%s %s 三个模型链路均已就绪且启动目标进入HOVER："
                        "箭头唯一推理帧%d帧，箭头三维位置发布端已连接，"
                        "ArUco%d帧，方框%d帧；"
                        "开始第一次箭头子任务"
                    ),
                    NODE_NAME,
                    KEY_LOG_MARKER,
                    PREREQUISITE_LOG_MARKER,
                    counts["arrow"],
                    counts["aruco"],
                    counts["rectangle"],
                )
                return True

            elapsed = now - started_at
            if elapsed >= self.model_ready_timeout:
                rospy.logerr(
                    (
                        "%s：等待三个模型链路和启动目标超过%.1fs仍未全部就绪；"
                        "箭头方向=%d个唯一推理帧/年龄%.2fs/就绪=%s，"
                        "箭头三维位置发布端连接=%s，"
                        "ArUco=%d帧/年龄%.2fs/就绪=%s，"
                        "方框=%d帧/年龄%.2fs/就绪=%s；启动目标=%s"
                    ),
                    NODE_NAME,
                    self.model_ready_timeout,
                    counts["arrow"],
                    ages["arrow"],
                    "是" if ready["arrow"] else "否",
                    "是" if arrow_target_connected else "否",
                    counts["aruco"],
                    ages["aruco"],
                    "是" if ready["aruco"] else "否",
                    counts["rectangle"],
                    ages["rectangle"],
                    "是" if ready["rectangle"] else "否",
                    hold_detail,
                )
                return False
            rospy.loginfo_throttle(
                2.0,
                (
                    "%s：%s 等待三个模型链路和启动目标，已等待%.1f/%.1fs："
                    "箭头方向=%d/%d个唯一推理帧(%s)，"
                    "箭头三维位置发布端(%s)，ArUco=%d/%d帧(%s)，"
                    "方框=%d/%d帧(%s)；启动目标=%s；"
                    "持续发布深度和初始航向目标"
                ),
                NODE_NAME,
                PREREQUISITE_LOG_MARKER,
                elapsed,
                self.model_ready_timeout,
                counts["arrow"],
                self.model_required_frames,
                "就绪" if ready["arrow"] else "等待",
                "已连接" if arrow_target_connected else "未连接",
                counts["aruco"],
                self.model_required_frames,
                "就绪" if ready["aruco"] else "等待",
                counts["rectangle"],
                self.model_required_frames,
                "就绪" if ready["rectangle"] else "等待",
                hold_detail,
            )
            self.rate.sleep()
        return False

    @staticmethod
    def yaw_from_pose(pose):
        """读取仅含yaw的四元数航向。"""
        quaternion = pose.orientation
        return math.atan2(
            2.0 * (
                quaternion.w * quaternion.z
                + quaternion.x * quaternion.y
            ),
            1.0 - 2.0 * (
                quaternion.y * quaternion.y
                + quaternion.z * quaternion.z
            ),
        )

    @staticmethod
    def angle_difference(angle_a, angle_b):
        """返回归一化到[-pi, pi)的角度差。"""
        return (angle_a - angle_b + math.pi) % (
            2.0 * math.pi
        ) - math.pi

    def make_final_timeout_goal(self, target_key):
        """把最终超时后的人工测量点转换为motion_supervisor绝对目标。"""
        target = self.final_timeout_targets[target_key]
        yaw = (
            self.initial_target_yaw
            if target_key == "aruco"
            else self.aruco_target_yaw
        )
        half_yaw = yaw * 0.5
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = target["N"]
        goal.pose.position.y = target["E"]
        goal.pose.position.z = self.fixed_map_z
        goal.pose.orientation.z = math.sin(half_yaw)
        goal.pose.orientation.w = math.cos(half_yaw)
        return goal

    @staticmethod
    def make_map_goal(north, east, z, yaw):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.pose.position.x = float(north)
        goal.pose.position.y = float(east)
        goal.pose.position.z = float(z)
        half_yaw = float(yaw) * 0.5
        goal.pose.orientation.z = math.sin(half_yaw)
        goal.pose.orientation.w = math.cos(half_yaw)
        return goal

    def capture_current_map_goal(self, z, context):
        """读取当前map位姿；TF不可用时退回motion反馈中的锁存目标。"""
        deadline = time.monotonic() + self.startup_tf_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                self.tf_listener.waitForTransform(
                    "map",
                    "base_link",
                    rospy.Time(0),
                    rospy.Duration(0.5),
                )
                translation, rotation = self.tf_listener.lookupTransform(
                    "map",
                    "base_link",
                    rospy.Time(0),
                )
            except (
                tf.Exception,
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ):
                self.rate.sleep()
                continue

            yaw = math.atan2(
                2.0 * (
                    float(rotation[3]) * float(rotation[2])
                    + float(rotation[0]) * float(rotation[1])
                ),
                1.0 - 2.0 * (
                    float(rotation[1]) * float(rotation[1])
                    + float(rotation[2]) * float(rotation[2])
                ),
            )
            return self.make_map_goal(
                translation[0],
                translation[1],
                z,
                yaw,
            )

        with self.motion_lock:
            state = self.latest_motion_state
        if (
            state is not None
            and state.goal.header.frame_id == "map"
        ):
            rospy.logwarn(
                "%s：%s读取TF失败，退回motion反馈锁存目标作为安全位置",
                NODE_NAME,
                context,
            )
            return self.make_map_goal(
                state.goal.pose.position.x,
                state.goal.pose.position.y,
                z,
                self.yaw_from_pose(state.goal.pose),
            )
        rospy.logerr("%s：%s无法获得当前map位置", NODE_NAME, context)
        return None

    def return_origin_and_ascend(self, context):
        """方框最终超时收尾：对准原点绝对航向、返原点并安全上浮。"""
        current_goal = self.capture_current_map_goal(
            self.fixed_map_z,
            "{}生成返航目标".format(context),
        )
        if current_goal is None:
            if self.startup_hold_goal is not None:
                current_goal = self.make_map_goal(
                    self.startup_hold_goal.pose.position.x,
                    self.startup_hold_goal.pose.position.y,
                    self.fixed_map_z,
                    self.return_origin_target_yaw,
                )
                rospy.logwarn(
                    "%s：%s无法获得当前位姿，退回任务启动锁存点生成返航目标",
                    NODE_NAME,
                    context,
                )
            else:
                current_goal = self.make_map_goal(
                    0.0,
                    0.0,
                    self.fixed_map_z,
                    self.return_origin_target_yaw,
                )
                rospy.logerr(
                    (
                        "%s：%s无法获得当前位姿和启动锁存点；"
                        "不退出，直接发布map原点返航目标"
                    ),
                    NODE_NAME,
                    context,
                )

        start_yaw = self.yaw_from_pose(current_goal.pose)
        turn_yaw = self.return_origin_target_yaw
        turn_goal = self.make_map_goal(
            current_goal.pose.position.x,
            current_goal.pose.position.y,
            self.fixed_map_z,
            turn_yaw,
        )
        rospy.logwarn(
            (
                "%s：%s开始原地对准返原点绝对航向："
                "配置原点航向=%.1fdeg，"
                "保持位置=(%.3f,%.3f,%.3f)，当前航向=%.1fdeg -> 目标航向=%.1fdeg"
            ),
            NODE_NAME,
            context,
            self.return_origin_yaw_deg,
            turn_goal.pose.position.x,
            turn_goal.pose.position.y,
            turn_goal.pose.position.z,
            math.degrees(start_yaw),
            math.degrees(turn_yaw),
        )
        turned = self.wait_for_motion_goal(
            turn_goal,
            self.post_drop_step_timeout,
            self.handoff_stable_seconds,
            "{}原地对准返原点绝对航向{:.1f}度".format(
                context,
                math.degrees(self.return_origin_target_yaw),
            ),
        )
        yaw = turn_yaw
        if not turned:
            rospy.logwarn(
                "%s：%s原地航向对准超时，返回原点时继续使用绝对航向%.1fdeg",
                NODE_NAME,
                context,
                math.degrees(yaw),
            )

        origin_goal = self.make_map_goal(
            0.0,
            0.0,
            self.fixed_map_z,
            yaw,
        )
        returned = self.wait_for_motion_goal(
            origin_goal,
            self.post_drop_step_timeout,
            self.handoff_stable_seconds,
            "{}返回map/NED原点".format(context),
        )
        safe_ascent_goal = current_goal
        if not returned:
            recovered_return_goal = self.capture_current_map_goal(
                self.fixed_map_z,
                "{}返航超时后锁存当前上浮点".format(context),
            )
            if recovered_return_goal is not None:
                safe_ascent_goal = recovered_return_goal

        ascent_base_pose = (
            origin_goal.pose if returned else safe_ascent_goal.pose
        )
        ascent_goal = self.make_map_goal(
            ascent_base_pose.position.x,
            ascent_base_pose.position.y,
            self.post_drop_ascent_target_z,
            yaw,
        )

        ascent_seconds = float(
            self.task3_params.get("post_drop_ascent_seconds", 5.0)
        )
        started_at = time.monotonic()
        while (
            not rospy.is_shutdown()
            and time.monotonic() - started_at < ascent_seconds
        ):
            ascent_goal.header.stamp = rospy.Time.now()
            self.goal_pub.publish(ascent_goal)
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：%s持续上浮 %.1f/%.1fs，"
                    "NED z向下为正，目标=(N=%.3f,E=%.3f,z=%.3f)"
                ),
                NODE_NAME,
                context,
                time.monotonic() - started_at,
                ascent_seconds,
                ascent_goal.pose.position.x,
                ascent_goal.pose.position.y,
                ascent_goal.pose.position.z,
            )
            self.rate.sleep()

        if rospy.is_shutdown():
            return False, "ROS关闭，上浮保持被中止"
        turn_detail = (
            "已原地对准返原点绝对航向%.1f度" % math.degrees(yaw)
            if turned
            else "原地航向对准超时，返原点时继续对准绝对航向%.1f度"
            % math.degrees(yaw)
        )
        return_detail = (
            "已返回map原点并向z=%.2f上浮、持续%.1fs"
            % (self.post_drop_ascent_target_z, ascent_seconds)
            if returned
            else "返回原点超时，已在安全锁存点向z=%.2f上浮、持续%.1fs"
            % (self.post_drop_ascent_target_z, ascent_seconds)
        )
        return (
            returned,
            "{}；{}".format(turn_detail, return_detail),
        )

    def wait_for_motion_goal(
        self,
        goal,
        timeout,
        stable_seconds,
        context,
    ):
        """持续发布绝对目标，直到目标匹配并稳定进入HOVER。"""
        started_at = time.monotonic()
        stable_started_at = None
        while not rospy.is_shutdown():
            goal.header.stamp = rospy.Time.now()
            self.goal_pub.publish(goal)
            now = time.monotonic()
            ready, detail = self.startup_hold_ready(goal)
            if ready:
                if stable_started_at is None:
                    stable_started_at = now
                stable_elapsed = now - stable_started_at
                if stable_elapsed >= stable_seconds:
                    rospy.loginfo(
                        "%s：%s [%s] 目标匹配并稳定HOVER %.1fs",
                        NODE_NAME,
                        KEY_LOG_MARKER,
                        context,
                        stable_elapsed,
                    )
                    return True
            else:
                stable_started_at = None

            elapsed = now - started_at
            if elapsed >= timeout:
                rospy.logerr(
                    "%s：[%s] 到达等待超过%.1fs；不发布cancel；%s",
                    NODE_NAME,
                    context,
                    timeout,
                    detail,
                )
                return False

            rospy.loginfo_throttle(
                1.0,
                "%s：[%s] 已用时%.1f/%.1fs，%s",
                NODE_NAME,
                context,
                elapsed,
                timeout,
                detail,
            )
            self.rate.sleep()
        return False

    def request_safety_hover(self, context):
        """运行期异常时请求主动刹停，并有界等待新的HOVER反馈。"""
        requested_at = time.monotonic()
        try:
            self.cancel_pub.publish(Empty())
        except Exception as error:
            rospy.logexception(
                "%s：[%s] 发布motion cancel失败；不退出，继续固定点保护：%s",
                NODE_NAME,
                context,
                str(error),
            )
            return False
        rospy.logwarn(
            "%s：[%s] 已发布motion cancel；保持当前位置并等待HOVER，最长%.1fs",
            NODE_NAME,
            context,
            self.cancel_recovery_timeout,
        )
        while not rospy.is_shutdown():
            with self.motion_lock:
                state = self.latest_motion_state
                received_at = self.latest_motion_state_wall_time
            fresh = (
                state is not None
                and received_at is not None
                and received_at >= requested_at
                and time.monotonic() - received_at <= self.motion_state_timeout
            )
            if (
                fresh
                and state.startup_complete
                and state.state == MotionState.HOVER
            ):
                rospy.loginfo(
                    "%s：%s [%s] 异常后已进入HOVER，交回总调度自救",
                    NODE_NAME,
                    HANDOFF_LOG_MARKER,
                    context,
                )
                return True

            elapsed = time.monotonic() - requested_at
            if elapsed >= self.cancel_recovery_timeout:
                rospy.logwarn(
                    (
                        "%s：[%s] 等待异常后HOVER超过%.1fs；"
                        "不退出，继续执行下一固定点保护"
                    ),
                    NODE_NAME,
                    context,
                    self.cancel_recovery_timeout,
                )
                return False
            rospy.loginfo_throttle(
                1.0,
                "%s：[%s] 异常悬停等待%.1f/%.1fs，motion_state=%s，反馈新鲜=%s",
                NODE_NAME,
                context,
                elapsed,
                self.cancel_recovery_timeout,
                "未收到" if state is None else str(state.state),
                "是" if fresh else "否",
            )
            self.rate.sleep()
        return False

    def attempt_final_timeout_target(self, target_key, context):
        """固定点移动失败或内部抛错时只记录，不终止后续保护链。"""
        try:
            reached = self.move_to_final_timeout_target(target_key, context)
        except Exception as error:
            rospy.logexception(
                "%s：[%s] 固定点保护发生异常，继续后续保护：%s",
                NODE_NAME,
                context,
                str(error),
            )
            return False
        if not reached and not rospy.is_shutdown():
            rospy.logwarn(
                "%s：[%s] 未确认到达固定点；保留该目标并继续后续保护",
                NODE_NAME,
                context,
            )
        return reached

    def align_current_position_to_yaw(self, yaw, context):
        """保持当前位置对准阶段航向；失败时由下一阶段目标继续接管。"""
        try:
            current_goal = self.capture_current_map_goal(
                self.fixed_map_z,
                "{}锁存当前悬停点".format(context),
            )
            if current_goal is None:
                rospy.logwarn(
                    "%s：[%s] 无法锁存当前位置，下一阶段直接使用配置航向",
                    NODE_NAME,
                    context,
                )
                return False
            goal = self.make_map_goal(
                current_goal.pose.position.x,
                current_goal.pose.position.y,
                self.fixed_map_z,
                yaw,
            )
            return self.wait_for_motion_goal(
                goal,
                self.final_timeout_move_timeout,
                self.final_timeout_arrival_stable_seconds,
                context,
            )
        except Exception as error:
            rospy.logexception(
                "%s：[%s] 阶段航向保护发生异常，继续后续任务：%s",
                NODE_NAME,
                context,
                str(error),
            )
            return False

    def move_to_final_timeout_target(self, target_key, context):
        """阶段达到唯一最终超时后，移动到下一阶段人工测量点。"""
        goal = self.make_final_timeout_goal(target_key)
        target_label = self.FINAL_TIMEOUT_TARGET_LABELS[target_key]
        rospy.logwarn(
            (
                "%s：[%s] 移动到%s：map=(%.3f,%.3f,%.3f)，"
                "yaw=%.1fdeg"
            ),
            NODE_NAME,
            context,
            target_label,
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z,
            math.degrees(self.yaw_from_pose(goal.pose)),
        )
        return self.wait_for_motion_goal(
            goal,
            self.final_timeout_move_timeout,
            self.final_timeout_arrival_stable_seconds,
            "{}到{}".format(context, target_label),
        )

    @staticmethod
    def deactivate_task(task):
        task.embedded_active = False
        for name, resource in list(vars(task).items()):
            if resource is None:
                continue
            if not (name.endswith("_sub") or name.endswith("_pub")):
                continue
            unregister = getattr(resource, "unregister", None)
            if callable(unregister):
                try:
                    unregister()
                except Exception:
                    pass
        listener = getattr(task, "tf_listener", None)
        unregister = getattr(listener, "unregister", None)
        if callable(unregister):
            try:
                unregister()
            except Exception:
                pass

    @staticmethod
    def log_flow_stage(
        flow_step,
        task_name,
        stage,
        prerequisite,
        next_action,
    ):
        """输出一行不会被关键日志过滤器隐藏的总流程定位信息。"""
        rospy.loginfo(
            (
                "%s：%s 总流程=%s，当前任务=%s，当前阶段=%s；"
                "前置条件=%s；通过后=%s"
            ),
            NODE_NAME,
            FLOW_LOG_MARKER,
            flow_step,
            task_name,
            stage,
            prerequisite,
            next_action,
        )

    def run_subtask1(self, run_index):
        label = "第{}次箭头子任务".format(run_index)
        flow_step = (
            "1/{}".format(self.mission_stage_count)
            if run_index == 1
            else "3/4"
        )
        search_yaw_deg = (
            self.initial_yaw_deg
            if run_index == 1
            else self.aruco_yaw_deg
        )
        self.task1_params["search_yaw_deg"] = search_yaw_deg
        timeout_seconds = (
            self.arrow1_timeout_seconds
            if run_index == 1
            else self.arrow2_timeout_seconds
        )
        self.task1_params["max_wait_seconds"] = timeout_seconds
        if self.fixed_heading_enabled:
            alignment_prerequisite = "位置滑动窗稳定；箭头方向仅记录、不参与判定"
            alignment_result = "箭头map位置通过并保持阶段固定航向"
        else:
            alignment_prerequisite = "位置和方向滑动窗稳定"
            alignment_result = "箭头map位置与航向误差通过并定点保持"
        if run_index == 2:
            self.task1_params.update(self.arrow2_search_params)
            rospy.loginfo(
                (
                    "%s：%s 第二次箭头使用独立搜索参数："
                    "首次前进=%.2fm，左右横扫=%.2fm，"
                    "第二层前进=%.2fm，第三层前进=%.2fm"
                ),
                NODE_NAME,
                KEY_LOG_MARKER,
                self.arrow2_search_params[
                    "search_initial_forward_distance"
                ],
                self.arrow2_search_params["search_lateral_distance"],
                self.arrow2_search_params[
                    "search_second_forward_distance"
                ],
                self.arrow2_search_params[
                    "search_third_forward_distance"
                ],
            )
        self.log_flow_stage(
            flow_step,
            label,
            "进入箭头子任务",
            "总任务启动时已确认箭头模型可用；阶段内由箭头子任务检查模型新鲜度",
            "进入箭头搜索和滑动窗闭环移动对准",
        )

        self.log_flow_stage(
            flow_step,
            label,
            "搜索与对准执行",
            alignment_prerequisite,
            alignment_result,
        )

        rospy.loginfo(
            (
                "%s：%s [%s开始] 直接进入子函数，不启动子任务launch；"
                "本次启动悬停=%.1fs，搜索固定航向=%.1fdeg，"
                "模式=%d（%s），首次运动后最终超时=%.1fs"
            ),
            NODE_NAME,
            KEY_LOG_MARKER,
            label,
            self.task1_params["initial_hover_seconds"],
            search_yaw_deg,
            self.heading_mode,
            self.heading_mode_label,
            timeout_seconds,
        )
        task = None
        timed_out = False
        final_yaw = None
        try:
            task = self.EmbeddedTask1()
            success, detail = task.run()
            timed_out = bool(task.embedded_timed_out)
            if success:
                final_yaw = task.final_target_yaw
        except Exception as error:
            rospy.logexception("%s：%s发生未处理异常", NODE_NAME, label)
            success = False
            detail = str(error)
        finally:
            if task is not None:
                self.deactivate_task(task)

        rospy.loginfo(
            "%s：%s [%s结束] success=%s，%s",
            NODE_NAME,
            KEY_LOG_MARKER,
            label,
            str(success),
            detail,
        )
        return success, detail, timed_out, final_yaw

    def run_subtask2(self):
        label = "ArUco子任务"
        flow_step = "2/{}".format(self.mission_stage_count)
        self.log_flow_stage(
            flow_step,
            label,
            "进入ArUco子任务",
            "总任务启动时已确认ArUco模型可用；识别超时完全由ArUco子任务自行处理",
            "进入定点悬停、ID确认、亮灯和原地转向",
        )

        history_marker_id, history_color = self.get_aruco_history_result()
        self.task2_params["history_marker_id"] = (
            -1 if history_marker_id is None else history_marker_id
        )
        if history_marker_id is None:
            rospy.logwarn(
                (
                    "%s：子任务2启动前尚无三帧一致的ArUco历史结果；"
                    "实时识别超时后将使用人工设置颜色%s"
                ),
                NODE_NAME,
                self.task2_params.get("recognition_fallback_color", "red"),
            )
        else:
            rospy.loginfo(
                (
                    "%s：子任务2已锁存ArUco历史结果：ID=%d，颜色=%s；"
                    "实时识别超时后优先使用"
                ),
                NODE_NAME,
                history_marker_id,
                history_color,
            )

        self.log_flow_stage(
            flow_step,
            label,
            "识别、灯光与转向执行",
            "实时识别优先；超时后依次使用三帧一致历史结果和人工颜色",
            "等待灯光反馈到位并保持，再灭灯并执行配置的原地转向",
        )

        rospy.loginfo(
            "%s：%s [%s开始] 直接进入子函数，不启动子任务launch",
            NODE_NAME,
            KEY_LOG_MARKER,
            label,
        )
        task = None
        try:
            task = self.EmbeddedTask2()
            task.run()
            if task.embedded_success is None:
                success = False
                detail = "ROS关闭或ArUco子任务未返回结果"
            else:
                success = bool(task.embedded_success)
                detail = task.embedded_detail
            marker_id = task.confirmed_marker_id
            color = task.confirmed_color
            if color is None and success and marker_id is not None:
                color = task.color_for_marker(marker_id)
        except Exception as error:
            rospy.logexception("%s：%s发生未处理异常", NODE_NAME, label)
            success = False
            detail = str(error)
            color = None
        finally:
            if task is not None:
                self.deactivate_task(task)

        rospy.loginfo(
            "%s：%s [%s结束] success=%s，颜色=%s，%s",
            NODE_NAME,
            KEY_LOG_MARKER,
            label,
            str(success),
            str(color),
            detail,
        )
        return success, detail, color

    def run_subtask3(self, target_color, search_yaw):
        label = "彩色方框投放子任务"
        flow_step = (
            "3/3" if self.skip_second_arrow else "4/4"
        )
        self.task3_params["max_wait_seconds"] = self.box_timeout_seconds
        self.task3_params["search_yaw_deg"] = (
            math.degrees(search_yaw) % 360.0
        )
        self.log_flow_stage(
            flow_step,
            label,
            "进入方框子任务",
            "总任务启动时已确认方框模型可用；阶段内由方框子任务检查模型新鲜度",
            "搜索并对准目标颜色{}方框".format(target_color),
        )

        self.log_flow_stage(
            flow_step,
            label,
            "方框搜索、对准与投放执行",
            "目标颜色={}，识别和运动反馈满足当前状态门槛".format(
                target_color
            ),
            "投放前前移、灯光夹爪动作、返回N/E原点并持续上浮",
        )

        self.task3_params["target_color"] = str(target_color)
        if str(self.task3_params.get("operation_mode", "")).lower() != "auto":
            return (
                False,
                (
                    "整合任务要求子任务3的operation_mode=auto，当前为{}"
                ).format(self.task3_params.get("operation_mode")),
                False,
                False,
            )

        self.shutdown_unused_models_for_subtask3()

        rospy.loginfo(
            (
                "%s：%s [%s开始] 目标颜色=%s，超时=%.1fs，"
                "搜索方式=%s；直接进入子函数，不启动子任务launch"
            ),
            NODE_NAME,
            KEY_LOG_MARKER,
            label,
            target_color,
            float(self.task3_params["max_wait_seconds"]),
            "正常自动搜索",
        )
        task = None
        timed_out = False
        drop_action_started = False
        try:
            task = self.EmbeddedTask3()
            task.run()
            drop_action_started = bool(task.drop_action_started)
            if task.embedded_success is None:
                success = False
                detail = "ROS关闭或彩色方框子任务未返回结果"
            else:
                success = bool(task.embedded_success)
                detail = task.embedded_detail
            timed_out = bool(task.embedded_timed_out)
        except Exception as error:
            rospy.logexception("%s：%s发生未处理异常", NODE_NAME, label)
            success = False
            detail = str(error)
        finally:
            if task is not None:
                self.deactivate_task(task)

        rospy.loginfo(
            "%s：%s [%s结束] success=%s，%s",
            NODE_NAME,
            KEY_LOG_MARKER,
            label,
            str(success),
            detail,
        )
        return success, detail, timed_out, drop_action_started

    def shutdown_unused_models_for_subtask3(self):
        """进入子任务3时关闭后续不再使用的箭头和ArUco检测进程。"""
        requested_nodes = list(self.SUBTASK3_UNUSED_MODEL_NODES)
        try:
            active_nodes = set(rosnode.get_node_names())
        except Exception as error:
            rospy.logwarn(
                "%s：进入子任务3前无法读取ROS节点列表，未关闭箭头和ArUco模型：%s",
                NODE_NAME,
                str(error),
            )
            return False

        active_targets = [
            node_name for node_name in requested_nodes
            if node_name in active_nodes
        ]
        already_stopped = [
            node_name for node_name in requested_nodes
            if node_name not in active_nodes
        ]
        if already_stopped:
            rospy.loginfo(
                "%s：进入子任务3时以下模型节点已关闭：%s",
                NODE_NAME,
                ", ".join(already_stopped),
            )
        if not active_targets:
            rospy.loginfo(
                "%s：进入子任务3时箭头和ArUco模型均已关闭",
                NODE_NAME,
            )
            return True

        try:
            stopped_nodes, failed_nodes = rosnode.kill_nodes(active_targets)
        except Exception as error:
            rospy.logwarn(
                "%s：进入子任务3时关闭箭头和ArUco模型失败，任务继续：%s",
                NODE_NAME,
                str(error),
            )
            return False

        deadline = time.monotonic() + self.MODEL_SHUTDOWN_TIMEOUT
        remaining_nodes = set(active_targets)
        while (
            remaining_nodes
            and not rospy.is_shutdown()
            and time.monotonic() < deadline
        ):
            try:
                remaining_nodes.intersection_update(rosnode.get_node_names())
            except Exception:
                break
            if remaining_nodes:
                rospy.sleep(0.1)

        if stopped_nodes:
            rospy.loginfo(
                "%s：进入子任务3，已关闭模型节点：%s",
                NODE_NAME,
                ", ".join(str(node) for node in stopped_nodes),
            )
        unresolved_nodes = set(str(node) for node in failed_nodes)
        unresolved_nodes.update(remaining_nodes)
        if unresolved_nodes:
            rospy.logwarn(
                "%s：以下模型节点未确认关闭，子任务3继续执行：%s",
                NODE_NAME,
                ", ".join(sorted(unresolved_nodes)),
            )
            return False
        return True

    def finish(self, success, detail):
        if self.finished:
            return
        self.finished = True
        state = "finished" if success else "failed"
        message = "{} {}: {}".format(NODE_NAME, state, detail)
        self.finished_pub.publish(String(data=message))
        if success:
            rospy.loginfo(
                "%s：%s 完整任务3成功：%s",
                NODE_NAME,
                KEY_LOG_MARKER,
                detail,
            )
        else:
            rospy.logerr("%s：完整任务3失败：%s", NODE_NAME, detail)
        rospy.sleep(0.2)

    def fail(self, detail):
        self.finish(False, detail)
        return False

    def finish_with_return_protection(self, context, detail):
        """任何方框阶段失败或未处理异常都收口到返原点上浮。"""
        try:
            unused_returned, return_detail = self.return_origin_and_ascend(
                context
            )
            del unused_returned
        except Exception as error:
            rospy.logexception(
                "%s：%s标准返航保护异常，改为直接发布原点上浮目标：%s",
                NODE_NAME,
                context,
                str(error),
            )
            ascent_seconds = float(
                self.task3_params.get("post_drop_ascent_seconds", 5.0)
            )
            emergency_goal = self.make_map_goal(
                0.0,
                0.0,
                self.post_drop_ascent_target_z,
                self.return_origin_target_yaw,
            )
            started_at = time.monotonic()
            while (
                not rospy.is_shutdown()
                and time.monotonic() - started_at < ascent_seconds
            ):
                emergency_goal.header.stamp = rospy.Time.now()
                self.goal_pub.publish(emergency_goal)
                self.rate.sleep()
            return_detail = (
                "标准返航保护异常，已直接发布map原点、原点航向和上浮目标"
            )

        if rospy.is_shutdown():
            return self.fail("{}期间ROS关闭".format(context))
        self.finish(
            True,
            "{}；运行期异常已由总调度保护收尾；{}".format(
                detail,
                return_detail,
            ),
        )
        return True

    def run_all_timeout_protection(self, context):
        """模型或阶段整体不可用时，按全部超时路径走完固定点和返航。"""
        self.request_safety_hover("{}：进入全部超时保护".format(context))
        if rospy.is_shutdown():
            return self.fail("{}期间ROS关闭".format(context))
        self.attempt_final_timeout_target(
            "aruco",
            "{}：第一次箭头按超时处理".format(context),
        )
        if rospy.is_shutdown():
            return self.fail("{}前往ArUco固定点期间ROS关闭".format(context))
        self.align_current_position_to_yaw(
            self.aruco_target_yaw,
            "{}：ArUco按异常处理并对准固定航向".format(context),
        )
        if rospy.is_shutdown():
            return self.fail("{}对准ArUco航向期间ROS关闭".format(context))
        self.attempt_final_timeout_target(
            "box",
            "{}：第二次箭头按超时处理".format(context),
        )
        if rospy.is_shutdown():
            return self.fail("{}前往方框固定点期间ROS关闭".format(context))
        return self.finish_with_return_protection(
            "{}：方框按超时处理".format(context),
            "全部视觉阶段均按最终超时固定点执行",
        )

    def run_mission(self):
        self.log_flow_stage(
            "准备/{}".format(self.mission_stage_count),
            "整合任务启动",
            "三个模型与启动定点就绪检查",
            (
                "箭头/ArUco/方框各至少{}帧且消息年龄不超过{:.1f}s；"
                "深度目标z={:.2f}、航向目标{:.1f}度进入HOVER"
            ).format(
                self.model_required_frames,
                self.model_output_timeout,
                self.fixed_map_z,
                self.initial_yaw_deg,
            ),
            "开始第1/{}阶段：第一次箭头".format(
                self.mission_stage_count
            ),
        )
        rospy.loginfo(
            (
                "%s：%s 完整任务3开始，等待三个模型全部就绪；"
                "等待期间持续发布入口位置、固定深度和初始航向目标"
            ),
            NODE_NAME,
            KEY_LOG_MARKER,
        )
        startup_hold_goal = self.capture_startup_hold_goal()
        if startup_hold_goal is None:
            fallback_goal = self.capture_current_map_goal(
                self.fixed_map_z,
                "启动固定点TF异常保护",
            )
            if fallback_goal is not None:
                startup_hold_goal = self.make_map_goal(
                    fallback_goal.pose.position.x,
                    fallback_goal.pose.position.y,
                    self.fixed_map_z,
                    self.initial_target_yaw,
                )
                rospy.logwarn(
                    (
                        "%s：启动TF锁存失败，退回motion反馈位置，"
                        "继续保持初始绝对航向%.1fdeg"
                    ),
                    NODE_NAME,
                    self.initial_yaw_deg,
                )
            else:
                startup_hold_goal = self.make_map_goal(
                    0.0,
                    0.0,
                    self.fixed_map_z,
                    self.initial_target_yaw,
                )
                rospy.logerr(
                    (
                        "%s：启动TF和motion反馈均不可用；不退出，"
                        "使用map原点、固定深度和初始航向进入全部超时保护"
                    ),
                    NODE_NAME,
                )
        self.startup_hold_goal = startup_hold_goal
        if not self.wait_for_all_models(startup_hold_goal):
            if rospy.is_shutdown():
                return self.fail("等待模型和启动HOVER期间ROS关闭")
            rospy.logwarn(
                (
                    "%s：三个识别模型没有全部就绪或启动目标未进入HOVER；"
                    "不退出，按全部阶段超时执行固定点自救"
                ),
                NODE_NAME,
            )
            return self.run_all_timeout_protection("启动就绪异常")

        success, detail, timed_out, _ = self.run_subtask1(1)
        if rospy.is_shutdown():
            return self.fail("第一次箭头阶段期间ROS关闭")
        if not success:
            rospy.logwarn(
                (
                    "%s：第一次箭头%s：%s；不退出，"
                    "先保持悬停，再前往ArUco固定点"
                ),
                NODE_NAME,
                "达到最终超时" if timed_out else "运行期异常按超时降级",
                detail,
            )
            self.request_safety_hover("第一次箭头异常/超时交接")
            if rospy.is_shutdown():
                return self.fail("第一次箭头异常悬停期间ROS关闭")
            self.attempt_final_timeout_target(
                "aruco",
                "第一次箭头异常/最终超时",
            )
            if rospy.is_shutdown():
                return self.fail("第一次箭头保护移动期间ROS关闭")

        success, detail, target_color = self.run_subtask2()
        if rospy.is_shutdown():
            return self.fail("ArUco阶段期间ROS关闭")
        if not success:
            rospy.logwarn(
                (
                    "%s：ArUco阶段运行期异常：%s；不退出，先保持悬停，"
                    "使用预设颜色并重新对准ArUco绝对航向"
                ),
                NODE_NAME,
                detail,
            )
            self.request_safety_hover("ArUco异常交接")
            if rospy.is_shutdown():
                return self.fail("ArUco异常悬停期间ROS关闭")
            self.align_current_position_to_yaw(
                self.aruco_target_yaw,
                "ArUco异常后的绝对航向保护",
            )
            if rospy.is_shutdown():
                return self.fail("ArUco异常航向保护期间ROS关闭")
            target_color = None
        if target_color is None:
            target_color = str(
                self.task2_params.get("recognition_fallback_color", "red")
            ).strip().lower()
            if target_color not in ("yellow", "green", "red"):
                rospy.logerr(
                    (
                        "%s：recognition_fallback_color=%s无效，"
                        "为保证流程继续，使用red"
                    ),
                    NODE_NAME,
                    target_color,
                )
                target_color = "red"
            rospy.logwarn(
                "%s：ArUco未得到颜色，容错流程使用预设颜色%s",
                NODE_NAME,
                target_color,
            )

        if self.skip_second_arrow:
            second_arrow_yaw = self.aruco_target_yaw
            rospy.logwarn(
                (
                    "%s：%s 航向模式3已启用：ArUco阶段完成后"
                    "跳过第二次箭头，直接进入方框任务；方框固定航向=%.1fdeg"
                ),
                NODE_NAME,
                HANDOFF_LOG_MARKER,
                self.aruco_yaw_deg,
            )
        else:
            success, detail, timed_out, second_arrow_yaw = self.run_subtask1(2)
            if rospy.is_shutdown():
                return self.fail("第二次箭头阶段期间ROS关闭")
            if not success:
                rospy.logwarn(
                    (
                        "%s：第二次箭头%s：%s；不退出，"
                        "先保持悬停，再前往彩色方框固定点"
                    ),
                    NODE_NAME,
                    "达到最终超时" if timed_out else "运行期异常按超时降级",
                    detail,
                )
                self.request_safety_hover("第二次箭头异常/超时交接")
                if rospy.is_shutdown():
                    return self.fail("第二次箭头异常悬停期间ROS关闭")
                self.attempt_final_timeout_target(
                    "box",
                    "第二次箭头异常/最终超时",
                )
                if rospy.is_shutdown():
                    return self.fail("第二次箭头保护移动期间ROS关闭")
                second_arrow_yaw = self.aruco_target_yaw
            elif second_arrow_yaw is None:
                rospy.logwarn(
                    (
                        "%s：第二次箭头成功但没有返回最终目标航向；"
                        "不退出，方框阶段退回ArUco绝对航向%.1fdeg"
                    ),
                    NODE_NAME,
                    self.aruco_yaw_deg,
                )
                second_arrow_yaw = self.aruco_target_yaw

        box_search_yaw = (
            self.aruco_target_yaw
            if self.fixed_heading_enabled
            else second_arrow_yaw
        )
        rospy.loginfo(
            (
                "%s：%s 方框阶段继承航向=%.1fdeg，来源=%s；"
                "不再执行额外的方框入口转向"
            ),
            NODE_NAME,
            HANDOFF_LOG_MARKER,
            math.degrees(box_search_yaw),
            "固定ArUco绝对航向"
            if self.fixed_heading_enabled
            else "第二次箭头最终航向",
        )

        success, detail, timed_out, drop_action_started = self.run_subtask3(
            target_color,
            box_search_yaw,
        )
        if rospy.is_shutdown():
            return self.fail("彩色方框阶段期间ROS关闭")

        if success:
            self.finish(
                True,
                "目标颜色{}方框投放子任务完成：{}"
                .format(target_color, detail),
            )
            return True

        rospy.logwarn(
            (
                "%s：彩色方框%s：%s；投放动作已开始=%s；"
                "不退出，先保持悬停，再按原点绝对航向%.1f度返航上浮"
            ),
            NODE_NAME,
            "达到最终超时" if timed_out else "运行期异常按超时收尾",
            detail,
            "是" if drop_action_started else "否",
            math.degrees(self.return_origin_target_yaw),
        )
        self.request_safety_hover("彩色方框异常/超时交接")
        if rospy.is_shutdown():
            return self.fail("方框异常悬停期间ROS关闭")
        return self.finish_with_return_protection(
            "彩色方框异常/最终超时",
            (
                "彩色方框保护已触发：{}；投放动作已开始={}"
                .format(detail, "是" if drop_action_started else "否")
            ),
        )

    def run(self):
        """兜住所有未预料的运行期异常，统一保持悬停并返航上浮。"""
        try:
            return self.run_mission()
        except rospy.ROSInterruptException:
            raise
        except Exception as error:
            rospy.logexception(
                "%s：整合任务出现未处理运行期异常，不退出并启动总保护：%s",
                NODE_NAME,
                str(error),
            )
            if rospy.is_shutdown():
                return self.fail("未处理异常后ROS已关闭：{}".format(error))
            self.request_safety_hover("整合任务未处理异常")
            if rospy.is_shutdown():
                return self.fail("未处理异常悬停期间ROS关闭")
            return self.finish_with_return_protection(
                "整合任务未处理异常",
                "未处理异常={}，已转入返航保护".format(error),
            )

    def on_shutdown(self):
        if not self.finished:
            rospy.logwarn(
                "%s：完整任务节点关闭；保留motion_supervisor当前目标，"
                "不发布cancel",
                NODE_NAME,
            )


def main():
    rospy.init_node(NODE_NAME)
    configure_file_logging()
    try:
        Task3Final().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("%s：未处理异常：%s", NODE_NAME, str(error))
        raise


if __name__ == "__main__":
    main()
