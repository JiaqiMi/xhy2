#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""任务3整合版：在一个ROS节点内顺序执行三个子任务。

执行顺序：
    第一次箭头 -> ArUco识别、亮灯和转向 -> 第二次箭头 -> 彩色方框投放

三个识别模型由task3.launch一次性启动并保持常驻。本节点不再为每个
阶段启动或关闭子任务launch，而是复用现有三个子任务类的控制方法。每个
子任务结束时原本请求的rospy全局关闭会被转换为当前子函数返回。

三个子任务和整合调度参数统一由config/task3.yaml加载到ROS参数服务器。
独立子任务与整合任务读取同一份参数，避免两套实验值逐渐不一致。
"""

from datetime import datetime
import logging
import math
import os
import sys
import threading
import time

import rospkg
import rospy
import tf
from auv_control.msg import MotionState, TargetDetection
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, String


NODE_NAME = "task3_final"
_MISSING = object()
KEY_LOG_MARKER = "[关键]"
CONSOLE_EVENT_PHRASES = (
    KEY_LOG_MARKER,
    "三个模型均已就绪",
    "前模型复查通过",
    "任务阶段",
    "状态切换 ",
    "运动状态切换为",
    "识别成功：ArUco",
    "箭头位置候选组确认通过",
    "逐帧候选组确认通过",
    "稳定识别成功",
    "残缺方框安全点已确认",
    "残缺方框一次移动目标已发布",
    "直接投放复核通过",
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
            "task3_final_{}.log".format(timestamp),
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
            "终端关键日志过滤=%s，进度日志间隔=%.2fs，"
            "终端处理器=%d"
        ),
        NODE_NAME,
        KEY_LOG_MARKER,
        log_path,
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
    TIMEOUT_SKIP_TARGET_LABELS = {
        "subtask2": "子任务2 ArUco识别点",
        "second_arrow": "第二个箭头起始点",
        "subtask3": "子任务3彩色方框起始点",
        "box_red": "红色方框人工复核点",
        "box_yellow": "黄色方框人工复核点",
        "box_green": "绿色方框人工复核点",
    }

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
        self.initial_yaw_deg = float(rospy.get_param(
            "/task3_initial_yaw_deg", 0.0
        ))
        if not math.isfinite(self.initial_yaw_deg):
            raise ValueError("task3_initial_yaw_deg必须是有限数")

        self.task1_params = load_task_params(
            "/test_task3_1_acquire_area"
        )
        self.task2_params = load_task_params(
            "/test_task3_2_get_task"
        )
        self.task3_params = load_task_params(
            "/test_task3_3_inspect_and_drop"
        )
        self.arrow_topic = str(rospy.get_param(
            "~arrow_topic",
            "/vision/arrow/direction",
        )).strip()
        self.aruco_topic = str(rospy.get_param(
            "~aruco_topic",
            "/vision/aruco/target_message",
        )).strip()
        self.rectangle_topic = str(rospy.get_param(
            "~rectangle_topic",
            "/vision/rectangle/detections",
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
        self.model_recovery_timeout = float(rospy.get_param(
            "~model_recovery_timeout",
            10.0,
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
        self.timeout_skip_enabled = bool(rospy.get_param(
            "/task3_final/timeout_skip_enabled",
            False,
        ))
        self.timeout_skip_move_timeout = float(rospy.get_param(
            "/task3_final/timeout_skip_move_timeout",
            120.0,
        ))
        self.timeout_skip_arrival_stable_seconds = float(rospy.get_param(
            "/task3_final/timeout_skip_arrival_stable_seconds",
            2.0,
        ))
        self.box_point_recheck_timeout = float(rospy.get_param(
            "/task3_final/box_point_recheck_timeout",
            30.0,
        ))
        self.timeout_skip_targets = self.load_timeout_skip_targets()

        if self.model_required_frames <= 0:
            raise ValueError("model_required_frames必须大于0")
        if self.rate_hz <= 0.0:
            raise ValueError("rate必须大于0")
        if min(
            self.motion_state_timeout,
            self.cancel_recovery_timeout,
        ) <= 0.0:
            raise ValueError("task3_protection时间参数必须大于0")
        if min(
            self.timeout_skip_move_timeout,
            self.timeout_skip_arrival_stable_seconds,
            self.box_point_recheck_timeout,
        ) <= 0.0:
            raise ValueError("超时跳点相关时间参数必须大于0")

        common_topics = {
            "motion_goal_topic": self.motion_goal_topic,
            "motion_cancel_topic": self.motion_cancel_topic,
            "motion_state_topic": self.motion_state_topic,
        }
        self.task1_params.update(common_topics)
        self.task2_params.update(common_topics)
        self.task3_params.update(common_topics)
        self.task1_params["arrow_topic"] = self.arrow_topic
        self.task1_params["status_topic"] = self.status_topic
        self.task2_params["aruco_topic"] = self.aruco_topic
        self.task2_params["actuator_topic"] = self.actuator_topic
        self.task3_params["model_detection_topic"] = self.rectangle_topic
        self.task3_params["status_topic"] = self.status_topic
        self.task3_params["actuator_topic"] = self.actuator_topic
        # 整合模式统一在阶段之间完成HOVER交接，取消子任务内部重复的
        # 结束保持和下一阶段启动悬停。第一次箭头的启动悬停由总调度
        # 锁存固定点完成，箭头模型在子任务启动前单独检查。
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
        self.motion_sub = rospy.Subscriber(
            self.motion_state_topic,
            MotionState,
            self.motion_state_callback,
            queue_size=20,
        )

        self.model_lock = threading.Lock()
        self.model_counts = {
            "arrow": 0,
            "aruco": 0,
            "rectangle": 0,
        }
        self.model_latest_wall_time = {
            "arrow": None,
            "aruco": None,
            "rectangle": None,
        }
        self.model_subscribers = [
            rospy.Subscriber(
                self.arrow_topic,
                String,
                self.arrow_model_callback,
                queue_size=20,
            ),
            rospy.Subscriber(
                self.aruco_topic,
                TargetDetection,
                self.aruco_model_callback,
                queue_size=20,
            ),
            rospy.Subscriber(
                self.rectangle_topic,
                String,
                self.rectangle_model_callback,
                queue_size=20,
            ),
        ]

        self.finished = False
        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo(
            (
                "%s：%s 整合节点启动，参数来自统一config/task3.yaml："
                "子任务1=%d项，子任务2=%d项，子任务3=%d项；"
                "统一固定深度=%.3fm（map目标z=%.3f）；"
                "任务3初始航向=%.1fdeg"
            ),
            NODE_NAME,
            KEY_LOG_MARKER,
            len(self.task1_params),
            len(self.task2_params),
            len(self.task3_params),
            self.fixed_depth_m,
            self.fixed_map_z,
            self.initial_yaw_deg,
        )
        rospy.loginfo(
            (
                "%s：模型话题：箭头=%s，ArUco=%s，方框=%s；"
                "阶段间不再启动或关闭子任务launch"
            ),
            NODE_NAME,
            self.arrow_topic,
            self.aruco_topic,
            self.rectangle_topic,
        )
        rospy.loginfo(
            (
                "%s：整合模式悬停优化：启动后锁存一个固定点；"
                "每个识别模型仅在对应子任务开始前检查；"
                "阶段恢复停稳需连续保持 %.1fs"
            ),
            NODE_NAME,
            self.handoff_stable_seconds,
        )
        rospy.loginfo(
            (
                "%s：三级超时保护：运动反馈第一级=%.1fs，"
                "cancel恢复第二级=%.1fs；恢复后当前阶段固定重试一次，"
                "再次失败进入第三级跳过并继续"
            ),
            NODE_NAME,
            self.motion_state_timeout,
            self.cancel_recovery_timeout,
        )
        if self.timeout_skip_enabled:
            rospy.logwarn(
                (
                    "%s：时间超时跳点容错已开启；不使用目标视野范围推断。"
                    "子任务1=%.1fs，子任务2识别=%.1fs，"
                    "子任务3首次=%.1fs，颜色点原地复核=%.1fs"
                ),
                NODE_NAME,
                float(self.task1_params["max_wait_seconds"]),
                float(self.task2_params["recognition_fallback_seconds"]),
                float(self.task3_params["max_wait_seconds"]),
                self.box_point_recheck_timeout,
            )
            for key, target in self.timeout_skip_targets.items():
                if key == "task3_start":
                    rospy.logwarn(
                        (
                            "%s：容错目标[%s] 启动时锁存当前map x/y，"
                            "固定深度=%.3fm，yaw=%.1fdeg"
                        ),
                        NODE_NAME,
                        self.TIMEOUT_SKIP_TARGET_LABELS[key],
                        self.fixed_depth_m,
                        target["yaw_deg"],
                    )
                    continue
                rospy.logwarn(
                    (
                        "%s：容错目标[%s] map/NED="
                        "(N=%.3f,E=%.3f,D=%.3f)，yaw=%.1fdeg"
                    ),
                    NODE_NAME,
                    self.TIMEOUT_SKIP_TARGET_LABELS[key],
                    target["N"],
                    target["E"],
                    self.fixed_map_z,
                    target["yaw_deg"],
                )
        else:
            rospy.loginfo(
                (
                    "%s：人工坐标跳点已关闭；阶段重试后仍失败时"
                    "在当前位置继续后续流程"
                ),
                NODE_NAME,
            )

    def load_timeout_skip_targets(self):
        """仅在容错开启时解析入口航向和人工测量的map绝对目标。"""
        if not self.timeout_skip_enabled:
            return {}
        raw_targets = rospy.get_param(
            "/task3_final/timeout_skip_targets",
            {},
        )
        if not isinstance(raw_targets, dict):
            raise ValueError("timeout_skip_targets必须是字典")

        targets = {}
        for key in self.TIMEOUT_SKIP_TARGET_LABELS:
            raw_target = raw_targets.get(key)
            if not isinstance(raw_target, dict):
                raise ValueError(
                    "超时跳点目标{}尚未配置".format(key)
                )
            fields = ("N", "E", "yaw_deg")
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
                self.embedded_success = bool(success)
                self.embedded_detail = str(detail)
                super().finish_task(success, detail)

            def on_shutdown(self):
                if self.embedded_active:
                    super().on_shutdown()

            def run(self):
                while not rospy.is_shutdown() and not self.task_finished:
                    elapsed = (
                        rospy.Time.now() - self.task_started
                    ).to_sec()
                    if (
                        elapsed >= self.max_wait_seconds
                        and self.state != self.FINAL_HOLD
                    ):
                        self.embedded_timed_out = True
                        self.finish_task(
                            False,
                            "搜索和对准累计超过{:.1f}s".format(
                                self.max_wait_seconds
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
                super().finalize_task(success, detail)

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
                reason_text = str(reason)
                self.embedded_success = bool(success)
                self.embedded_detail = reason_text
                self.embedded_timed_out = (
                    not success
                    and (
                        reason_text.startswith("自动搜索/等待 ")
                        or reason_text.startswith("等待 ")
                    )
                    and "后仍未" in reason_text
                )
                super().finish_task(success, reason)

            def on_shutdown(self):
                if self.embedded_active:
                    super().on_shutdown()

        return EmbeddedTask3

    def motion_state_callback(self, message):
        with self.motion_lock:
            self.latest_motion_state = message
            self.latest_motion_state_wall_time = time.monotonic()

    def _record_model_frame(self, role):
        with self.model_lock:
            self.model_counts[role] += 1
            self.model_latest_wall_time[role] = time.monotonic()

    def arrow_model_callback(self, _message):
        self._record_model_frame("arrow")

    def aruco_model_callback(self, _message):
        self._record_model_frame("aruco")

    def rectangle_model_callback(self, _message):
        self._record_model_frame("rectangle")

    def _model_snapshot(self):
        with self.model_lock:
            return (
                dict(self.model_counts),
                dict(self.model_latest_wall_time),
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
            half_yaw = math.radians(self.initial_yaw_deg) * 0.5
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

    def wait_for_startup_hold(self, hold_goal):
        started_at = time.monotonic()
        while not rospy.is_shutdown():
            self.publish_startup_hold_goal(hold_goal)
            counts, _ = self._model_snapshot()
            hold_ready, hold_detail = self.startup_hold_ready(hold_goal)
            if hold_ready:
                rospy.loginfo(
                    (
                        "%s：%s 启动定点保持正常；模型改为各阶段开始前"
                        "单独检查：箭头%d帧，ArUco%d帧，方框%d帧"
                    ),
                    NODE_NAME,
                    KEY_LOG_MARKER,
                    counts["arrow"],
                    counts["aruco"],
                    counts["rectangle"],
                )
                return True

            elapsed = time.monotonic() - started_at
            if elapsed >= self.model_ready_timeout:
                rospy.logerr(
                    (
                        "%s：启动定点超过%.1fs仍未就绪；"
                        "模型累计：箭头=%d帧，ArUco=%d帧，方框=%d帧"
                    ),
                    NODE_NAME,
                    self.model_ready_timeout,
                    counts["arrow"],
                    counts["aruco"],
                    counts["rectangle"],
                )
                return False
            rospy.loginfo_throttle(
                2.0,
                (
                    "%s：等待启动定点，已等待%.1f/%.1fs："
                    "箭头=%d，ArUco=%d，方框=%d；启动定点=%s"
                ),
                NODE_NAME,
                elapsed,
                self.model_ready_timeout,
                counts["arrow"],
                counts["aruco"],
                counts["rectangle"],
                hold_detail,
            )
            self.rate.sleep()
        return False

    def wait_for_new_model_frames(self, role, context):
        counts, _ = self._model_snapshot()
        baseline = counts[role]
        started_at = time.monotonic()
        while not rospy.is_shutdown():
            counts, latest = self._model_snapshot()
            new_count = counts[role] - baseline
            last_time = latest[role]
            age = (
                float("inf")
                if last_time is None
                else time.monotonic() - last_time
            )
            if (
                new_count >= self.model_required_frames
                and age <= self.model_output_timeout
            ):
                rospy.loginfo(
                    "%s：%s前模型复查通过：%s新增%d/%d帧",
                    NODE_NAME,
                    context,
                    role,
                    new_count,
                    self.model_required_frames,
                )
                return True

            elapsed = time.monotonic() - started_at
            if elapsed >= self.model_recovery_timeout:
                rospy.logerr(
                    (
                        "%s：%s前模型复查失败：%s新增%d/%d帧，"
                        "最新消息年龄=%.2fs"
                    ),
                    NODE_NAME,
                    context,
                    role,
                    new_count,
                    self.model_required_frames,
                    age,
                )
                return False
            rospy.loginfo_throttle(
                1.0,
                "%s：%s前等待%s模型新帧%d/%d",
                NODE_NAME,
                context,
                role,
                new_count,
                self.model_required_frames,
            )
            self.rate.sleep()
        return False

    def wait_for_motion_recovery(self, context):
        """第二级保护：发布一次cancel，等待HOVER或刹车低速稳定。"""
        started_at = time.monotonic()
        stable_started_at = None
        max_horizontal_speed = float(
            self.task3_params.get("auto_action_max_horizontal_speed", 0.03)
        )
        max_yaw_rate = float(
            self.task3_params.get("auto_action_max_yaw_rate", 0.05)
        )
        brake_states = {
            MotionState.TRANSLATE_BRAKE,
            MotionState.FINAL_BRAKE,
            MotionState.SAFE,
        }
        self.cancel_pub.publish(Empty())

        while not rospy.is_shutdown():
            now = time.monotonic()

            with self.motion_lock:
                state = self.latest_motion_state
                state_time = self.latest_motion_state_wall_time

            fresh = (
                state is not None
                and state_time is not None
                and now - state_time <= self.motion_state_timeout
            )
            hovering = (
                fresh
                and state.startup_complete
                and state.state == MotionState.HOVER
            )
            brake_stopped = (
                fresh
                and state.startup_complete
                and state.state in brake_states
                and abs(state.horizontal_speed) <= max_horizontal_speed
                and abs(state.yaw_rate) <= max_yaw_rate
            )
            stopped = hovering or brake_stopped
            if stopped:
                if stable_started_at is None:
                    stable_started_at = now
                stable_elapsed = now - stable_started_at
                if stable_elapsed >= self.handoff_stable_seconds:
                    rospy.logwarn(
                        (
                            "%s：%s第二级恢复完成：方式=%s，state=%d，"
                            "水平速度=%.3fm/s，yaw_rate=%.3frad/s，稳定%.1fs"
                        ),
                        NODE_NAME,
                        context,
                        "HOVER" if hovering else "刹车低速降级停稳",
                        state.state,
                        state.horizontal_speed,
                        state.yaw_rate,
                        stable_elapsed,
                    )
                    return True
            else:
                stable_started_at = None

            elapsed = now - started_at
            if elapsed >= self.cancel_recovery_timeout:
                rospy.logerr(
                    (
                        "%s：%s第二级cancel恢复超过%.1fs；"
                        "反馈新鲜=%s，state=%s，跳过本阶段重试"
                    ),
                    NODE_NAME,
                    context,
                    self.cancel_recovery_timeout,
                    "是" if fresh else "否",
                    "无反馈" if state is None else str(state.state),
                )
                return False

            rospy.loginfo_throttle(
                1.0,
                "%s：%s第二级cancel恢复 %.1f/%.1fs",
                NODE_NAME,
                context,
                elapsed,
                self.cancel_recovery_timeout,
            )
            self.rate.sleep()
        return False

    @staticmethod
    def replace_result_detail(result, detail):
        return (result[0], str(detail)) + tuple(result[2:])

    def run_stage_with_single_retry(
        self,
        label,
        runner,
        skip_retry_on_timeout=False,
    ):
        """第一级失败后恢复并固定重试一次；再次失败交给总流程跳过。"""
        first_result = runner()
        if first_result[0] or rospy.is_shutdown():
            return first_result

        first_detail = str(first_result[1] or "未返回失败原因")
        timed_out = len(first_result) >= 3 and bool(first_result[2])
        if skip_retry_on_timeout and timed_out:
            rospy.logwarn(
                (
                    "%s：%s已达到配置的总超时：%s；"
                    "不再重复整轮搜索，直接进入人工点交接"
                ),
                NODE_NAME,
                label,
                first_detail,
            )
            return first_result
        rospy.logwarn(
            "%s：%s第一次执行失败：%s；进入第二级cancel恢复",
            NODE_NAME,
            label,
            first_detail,
        )
        if not self.wait_for_motion_recovery("{}重试前".format(label)):
            return self.replace_result_detail(
                first_result,
                "{}；cancel恢复超时，未执行重试".format(first_detail),
            )

        rospy.logwarn(
            "%s：%s第二级恢复完成，开始唯一一次重试",
            NODE_NAME,
            label,
        )
        retry_result = runner()
        if retry_result[0]:
            rospy.loginfo(
                "%s：%s唯一一次重试成功",
                NODE_NAME,
                label,
            )
            return retry_result

        retry_detail = str(retry_result[1] or "未返回失败原因")
        rospy.logwarn(
            (
                "%s：%s唯一一次重试仍失败：%s；"
                "进入第三级保护，跳过当前阶段"
            ),
            NODE_NAME,
            label,
            retry_detail,
        )
        return self.replace_result_detail(
            retry_result,
            "首次失败={}；重试失败={}".format(
                first_detail,
                retry_detail,
            ),
        )

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

    def make_timeout_skip_goal(self, target_key):
        """把人工测量点转换为motion_supervisor绝对目标。"""
        if target_key == "task3_start":
            raise ValueError(
                "task3_start只配置初始航向，必须通过启动位姿锁存生成目标"
            )
        target = self.timeout_skip_targets[target_key]
        yaw = math.radians(target["yaw_deg"])
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
        """第三级收尾：返回N/E原点；失败也在安全锁存点上浮5秒。"""
        current_goal = self.capture_current_map_goal(
            self.fixed_map_z,
            "{}生成返航目标".format(context),
        )
        if current_goal is None:
            return None, "无法获得返航起始位姿，不能安全生成上浮目标"

        yaw = self.yaw_from_pose(current_goal.pose)
        origin_goal = self.make_map_goal(
            0.0,
            0.0,
            self.fixed_map_z,
            yaw,
        )
        step_timeout = float(
            self.task3_params.get("post_drop_step_timeout", 90.0)
        )
        returned = self.wait_for_motion_goal(
            origin_goal,
            step_timeout,
            self.handoff_stable_seconds,
            "{}返回map/NED原点".format(context),
        )
        if not returned:
            self.wait_for_motion_recovery(
                "{}返航超时后的安全停稳".format(context)
            )

        ascent_goal = self.capture_current_map_goal(
            0.0,
            "{}生成上浮目标".format(context),
        )
        if ascent_goal is None:
            fallback_pose = (
                origin_goal.pose if returned else current_goal.pose
            )
            ascent_goal = self.make_map_goal(
                fallback_pose.position.x,
                fallback_pose.position.y,
                0.0,
                yaw,
            )
            rospy.logwarn(
                (
                    "%s：%s无法重新读取当前位姿，"
                    "使用最近安全锁存点持续上浮"
                ),
                NODE_NAME,
                context,
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
                    "目标=(N=%.3f,E=%.3f,z=0.000)"
                ),
                NODE_NAME,
                context,
                time.monotonic() - started_at,
                ascent_seconds,
                ascent_goal.pose.position.x,
                ascent_goal.pose.position.y,
            )
            self.rate.sleep()

        if rospy.is_shutdown():
            return False, "ROS关闭，上浮保持被中止"
        return (
            returned,
            (
                "已返回map原点并上浮%.1fs" % ascent_seconds
                if returned
                else "返回原点超时，已在安全锁存点上浮%.1fs" % ascent_seconds
            ),
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
                self.cancel_pub.publish(Empty())
                rospy.logerr(
                    "%s：[%s] 到达等待超过%.1fs，已cancel；%s",
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

    def move_to_stage_target(self, target_key, context):
        """取消当前动作后，移动到下一阶段的人工绝对测量点。"""
        self.cancel_pub.publish(Empty())
        rospy.sleep(0.2)

        goal = self.make_timeout_skip_goal(target_key)
        target_label = self.TIMEOUT_SKIP_TARGET_LABELS[target_key]
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
            self.timeout_skip_targets[target_key]["yaw_deg"],
        )
        return self.wait_for_motion_goal(
            goal,
            self.timeout_skip_move_timeout,
            self.timeout_skip_arrival_stable_seconds,
            "{}到{}".format(context, target_label),
        )

    def prepare_next_stage(self, target_key, context):
        """阶段交接失败只记录，不再终止整个任务。"""
        if self.timeout_skip_enabled:
            if self.move_to_stage_target(target_key, context):
                return True
            self.wait_for_motion_recovery(
                "{}人工点移动超时后的安全停稳".format(context)
            )
            return False
        return self.wait_for_motion_recovery(context)

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

    def run_subtask1(self, run_index):
        label = "第{}次箭头子任务".format(run_index)
        if not self.wait_for_new_model_frames("arrow", label):
            return (
                False,
                "{}启动前箭头模型等待超时".format(label),
                True,
            )

        rospy.loginfo(
            (
                "%s：%s [%s开始] 直接进入子函数，不启动子任务launch；"
                "本次启动悬停=%.1fs"
            ),
            NODE_NAME,
            KEY_LOG_MARKER,
            label,
            self.task1_params["initial_hover_seconds"],
        )
        task = None
        timed_out = False
        try:
            task = self.EmbeddedTask1()
            success, detail = task.run()
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
        return success, detail, timed_out

    def run_subtask2(self):
        label = "ArUco子任务"
        if not self.wait_for_new_model_frames("aruco", label):
            return False, "{}启动前模型等待超时".format(label), None

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

    def run_subtask3(self, target_color, stationary_recheck=False):
        label = (
            "彩色方框人工点原地复核"
            if stationary_recheck
            else "彩色方框投放子任务"
        )
        if not self.wait_for_new_model_frames("rectangle", label):
            return (
                False,
                "{}启动前模型等待超时".format(label),
                True,
                False,
            )

        self.task3_params["target_color"] = str(target_color)
        if stationary_recheck:
            self.task3_params["max_wait_seconds"] = (
                self.box_point_recheck_timeout
            )
        if str(self.task3_params.get("operation_mode", "")).lower() != "auto":
            return (
                False,
                (
                    "整合任务要求子任务3的operation_mode=auto，当前为{}"
                ).format(self.task3_params.get("operation_mode")),
                False,
                False,
            )

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
            "人工颜色点原地识别" if stationary_recheck else "正常自动搜索",
        )
        task = None
        timed_out = False
        drop_action_started = False
        try:
            task = self.EmbeddedTask3()
            if stationary_recheck:
                task.auto_search_plan = [("hover", 0.0)]
                task.auto_search_index = 0
                task.reset_auto_search_step()
                task.reset_stability()
                rospy.logwarn(
                    (
                        "%s：[%s] 已关闭前进和左右搜索；仅在当前人工点"
                        "保持HOVER并重新识别，识别成功后复用原细对准和投放流程"
                    ),
                    NODE_NAME,
                    label,
                    )
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

    def finish(self, success, detail):
        if self.finished:
            return
        self.finished = True
        if not success:
            self.cancel_pub.publish(Empty())
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
        self.cancel_pub.publish(Empty())
        self.finish(False, detail)
        return False

    def run(self):
        skipped_stages = []
        movement_timeouts = []
        rospy.loginfo(
            (
                "%s：%s 完整任务3开始，锁存入口位置并调整初始航向；"
                "识别模型改为各子任务启动前检查"
            ),
            NODE_NAME,
            KEY_LOG_MARKER,
        )
        startup_hold_goal = self.capture_startup_hold_goal()
        if startup_hold_goal is None:
            return self.fail("无法锁存任务启动固定点")
        if not self.wait_for_startup_hold(startup_hold_goal):
            return self.fail("启动定点未在限定时间内进入HOVER")

        success, detail, timed_out = self.run_stage_with_single_retry(
            "第一次箭头",
            lambda: self.run_subtask1(1),
            skip_retry_on_timeout=self.timeout_skip_enabled,
        )
        if not success:
            skipped_stages.append("第一次箭头")
            rospy.logwarn(
                "%s：第三级保护跳过第一次箭头：%s",
                NODE_NAME,
                detail,
            )
        if rospy.is_shutdown():
            return self.fail("第一次箭头阶段期间ROS关闭")
        if not self.prepare_next_stage(
            "subtask2",
            "第一次箭头到ArUco子任务交接",
        ):
            movement_timeouts.append(
                "ArUco识别点"
                if self.timeout_skip_enabled
                else "第一次箭头到ArUco交接"
            )
            rospy.logwarn(
                "%s：第一次箭头到ArUco交接恢复超时，仍继续子任务2",
                NODE_NAME,
            )

        success, detail, target_color = self.run_stage_with_single_retry(
            "ArUco子任务",
            self.run_subtask2,
        )
        if not success:
            skipped_stages.append("ArUco识别")
            rospy.logwarn(
                "%s：第三级保护跳过ArUco识别：%s",
                NODE_NAME,
                detail,
            )
        if rospy.is_shutdown():
            return self.fail("ArUco阶段期间ROS关闭")
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
        if not self.prepare_next_stage(
            "second_arrow",
            "ArUco子任务到第二个箭头交接",
        ):
            movement_timeouts.append(
                "第二个箭头起始点"
                if self.timeout_skip_enabled
                else "ArUco到第二个箭头交接"
            )
            rospy.logwarn(
                "%s：ArUco到第二个箭头交接恢复超时，仍继续第二次箭头",
                NODE_NAME,
            )

        success, detail, timed_out = self.run_stage_with_single_retry(
            "第二次箭头",
            lambda: self.run_subtask1(2),
            skip_retry_on_timeout=self.timeout_skip_enabled,
        )
        if not success:
            skipped_stages.append("第二次箭头")
            rospy.logwarn(
                "%s：第三级保护跳过第二次箭头：%s",
                NODE_NAME,
                detail,
            )
        if rospy.is_shutdown():
            return self.fail("第二次箭头阶段期间ROS关闭")
        if not self.prepare_next_stage(
            "subtask3",
            "第二次箭头到彩色方框子任务交接",
        ):
            movement_timeouts.append(
                "彩色方框起始点"
                if self.timeout_skip_enabled
                else "第二次箭头到彩色方框交接"
            )
            rospy.logwarn(
                "%s：第二次箭头到彩色方框交接恢复超时，仍继续子任务3",
                NODE_NAME,
            )

        success, detail, timed_out, drop_action_started = self.run_subtask3(
            target_color
        )
        if not success:
            rospy.logwarn(
                (
                    "%s：彩色方框投放第一次执行失败：%s；"
                    "投放动作已开始=%s"
                ),
                NODE_NAME,
                detail,
                "是" if drop_action_started else "否",
            )
            if not drop_action_started and not rospy.is_shutdown():
                first_detail = detail
                if self.wait_for_motion_recovery("彩色方框投放重试前"):
                    stationary_recheck = False
                    box_target_key = "box_{}".format(
                        str(target_color).lower()
                    )
                    if (
                        self.timeout_skip_enabled
                        and timed_out
                        and box_target_key in self.timeout_skip_targets
                    ):
                        stationary_recheck = True
                        if not self.move_to_stage_target(
                            box_target_key,
                            "彩色方框首次识别超时",
                        ):
                            movement_timeouts.append(
                                "{}方框人工点".format(target_color)
                            )
                            rospy.logwarn(
                                (
                                    "%s：移动到%s方框人工点超时，"
                                    "仍在当前位置执行唯一一次原地复核"
                                ),
                                NODE_NAME,
                                target_color,
                            )
                    rospy.logwarn(
                        "%s：彩色方框投放开始唯一一次重试",
                        NODE_NAME,
                    )
                    (
                        success,
                        detail,
                        timed_out,
                        retry_drop_started,
                    ) = self.run_subtask3(
                        target_color,
                        stationary_recheck=stationary_recheck,
                    )
                    drop_action_started = (
                        drop_action_started or retry_drop_started
                    )
                    if not success:
                        detail = "首次失败={}；重试失败={}".format(
                            first_detail,
                            detail,
                        )
                else:
                    detail = "{}；cancel恢复超时，未执行重试".format(
                        first_detail
                    )
            elif drop_action_started:
                rospy.logwarn(
                    (
                        "%s：投放动作已经开始，为避免重复开爪和重复投放，"
                        "不重试子任务3"
                    ),
                    NODE_NAME,
                )

            if not success:
                skipped_stages.append(
                    "{}方框投放后异常收尾".format(target_color)
                    if drop_action_started
                    else "{}方框投放".format(target_color)
                )
                rospy.logwarn(
                    (
                        "%s：第三级保护结束%s方框当前阶段：%s；"
                        "投放动作已开始=%s，"
                        "开始统一返航上浮收尾"
                    ),
                    NODE_NAME,
                    target_color,
                    detail,
                    "是" if drop_action_started else "否",
                )

        if rospy.is_shutdown():
            return self.fail("彩色方框阶段期间ROS关闭")

        if success:
            finish_detail = (
                "目标颜色{}方框投放子任务完成：{}"
            ).format(target_color, detail)
            if skipped_stages:
                finish_detail += "；此前已跳过阶段：{}".format(
                    "、".join(skipped_stages)
                )
            if movement_timeouts:
                finish_detail += "；移动超时点={}".format(
                    "、".join(movement_timeouts)
                )
            self.finish(True, finish_detail)
            return True

        returned_to_origin, return_detail = self.return_origin_and_ascend(
            "投放后异常收尾" if drop_action_started else "未投放安全收尾"
        )
        if rospy.is_shutdown():
            return self.fail("统一返航上浮期间ROS关闭")
        if returned_to_origin is None:
            return self.fail("统一返航上浮无法安全执行：{}".format(
                return_detail
            ))
        if not returned_to_origin:
            movement_timeouts.append("map/NED原点")

        finish_detail = "三级容错流程完成，已跳过阶段：{}；{}".format(
            "、".join(skipped_stages),
            return_detail,
        )
        if movement_timeouts:
            finish_detail += "；移动超时点={}".format(
                "、".join(movement_timeouts)
            )
        self.finish(
            True,
            finish_detail,
        )
        return True

    def on_shutdown(self):
        if not self.finished and hasattr(self, "cancel_pub"):
            self.cancel_pub.publish(Empty())


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
