#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""任务3整合版：在一个ROS节点内顺序执行三个子任务。

执行顺序：
    第一次箭头 -> ArUco识别、亮灯和转向 -> 第二次箭头 -> 彩色方框投放

三个识别模型由task3_final.launch一次性启动并保持常驻。本节点不再为每个
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


def configure_file_logging():
    """把整合任务日志保存到task3目录。"""
    log_directory = os.path.abspath(os.path.expanduser(str(
        rospy.get_param("~log_directory", "~/.ros/auv_logs/task3")
    )))
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
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        logging.getLogger("rosout").addHandler(handler)
    except (IOError, OSError) as error:
        rospy.logerr(
            "%s：无法创建文件日志目录%s：%s",
            NODE_NAME,
            log_directory,
            str(error),
        )
        return None
    rospy.loginfo("%s：整合任务文件日志已启用：%s", NODE_NAME, log_path)
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
        self.handoff_timeout = float(rospy.get_param(
            "~handoff_timeout",
            30.0,
        ))
        self.handoff_stable_seconds = float(rospy.get_param(
            "~handoff_stable_seconds",
            1.0,
        ))
        self.motion_state_timeout = float(rospy.get_param(
            "~motion_state_timeout",
            2.0,
        ))
        self.startup_tf_timeout = float(rospy.get_param(
            "~startup_tf_timeout",
            8.0,
        ))
        self.rate_hz = float(rospy.get_param("~rate", 10.0))

        if self.model_required_frames <= 0:
            raise ValueError("model_required_frames必须大于0")
        if self.rate_hz <= 0.0:
            raise ValueError("rate必须大于0")

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
        # 结束保持和下一阶段启动悬停。第一次箭头的启动悬停由模型预热
        # 阶段统一完成，模型就绪后立即进入箭头任务。
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
        self.model_topics = {
            "arrow": self.arrow_topic,
            "aruco": self.aruco_topic,
            "rectangle": self.rectangle_topic,
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
                "%s：整合节点启动，参数来自统一config/task3.yaml："
                "子任务1=%d项，子任务2=%d项，子任务3=%d项；"
                "统一固定深度=%.3fm（map目标z=%.3f）"
            ),
            NODE_NAME,
            len(self.task1_params),
            len(self.task2_params),
            len(self.task3_params),
            self.fixed_depth_m,
            self.fixed_map_z,
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
                "%s：整合模式悬停优化：启动后锁存一个固定点并在三个"
                "模型预热期间持续保持；模型就绪后立即开始第一次箭头；"
                "后续阶段间只由总脚本连续保持HOVER %.1fs"
            ),
            NODE_NAME,
            self.handoff_stable_seconds,
        )

    def _make_embedded_task1(self):
        parent = self.task1_module.Task3AcquireAreaTest

        class EmbeddedTask1(parent):
            def __init__(self):
                self.embedded_success = None
                self.embedded_detail = ""
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

                    if self.state == self.INITIAL_HOVER:
                        self.control_initial_hover()
                    elif self.state == self.SEARCH_PATTERN:
                        self.control_search_pattern()
                    elif self.state == self.CANCEL_WAIT:
                        self.control_cancel_wait()
                    elif self.state == self.WAIT_FOR_ARROW:
                        self.control_wait_for_arrow()
                    elif self.state == self.COARSE_LATERAL_ALIGN:
                        self.control_coarse_lateral_align()
                    elif self.state == self.CONFIRM_DIRECTION:
                        self.control_confirm_direction()
                    elif self.state == self.ALIGN_HEADING:
                        self.control_align_heading()
                    elif self.state == self.FINE_FORWARD_ALIGN:
                        self.control_fine_forward_align()
                    elif self.state == self.MOVE_BASE_OVER_ARROW:
                        self.control_move_base_over_arrow()
                    elif self.state == self.FINAL_HOLD:
                        self.control_final_hold()

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
                self.embedded_active = True
                super().__init__()

            def finish_task(self, success, reason):
                if self.finished:
                    return
                self.embedded_success = bool(success)
                self.embedded_detail = str(reason)
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
        """只读取一次启动位姿，避免定点目标跟随漂移。"""
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
            goal.pose.orientation.x = float(rotation[0])
            goal.pose.orientation.y = float(rotation[1])
            goal.pose.orientation.z = float(rotation[2])
            goal.pose.orientation.w = float(rotation[3])
            rospy.loginfo(
                (
                    "%s：已锁存模型预热固定点：x=%.3f，y=%.3f，"
                    "固定深度=%.3fm（map z=%.3f）；"
                    "模型加载期间不会刷新该目标"
                ),
                NODE_NAME,
                goal.pose.position.x,
                goal.pose.position.y,
                self.fixed_depth_m,
                goal.pose.position.z,
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
        position_tolerance = float(
            self.task1_params["goal_match_position_tolerance"]
        )
        depth_tolerance = float(
            self.task1_params["goal_match_depth_tolerance"]
        )
        if position_error > position_tolerance:
            return False, "活动目标水平偏差{:.3f}m".format(position_error)
        if depth_error > depth_tolerance:
            return False, "活动目标深度偏差{:.3f}m".format(depth_error)
        return True, "固定启动目标已进入HOVER"

    def wait_for_all_models(self, hold_goal):
        started_at = time.monotonic()
        while not rospy.is_shutdown():
            self.publish_startup_hold_goal(hold_goal)
            counts, latest = self._model_snapshot()
            now = time.monotonic()
            ready = {}
            for role in self.MODEL_TYPES:
                last_time = latest[role]
                fresh = (
                    last_time is not None
                    and now - last_time <= self.model_output_timeout
                )
                ready[role] = (
                    counts[role] >= self.model_required_frames and fresh
                )
            hold_ready, hold_detail = self.startup_hold_ready(hold_goal)
            if all(ready.values()) and hold_ready:
                rospy.loginfo(
                    (
                        "%s：三个模型均已就绪且启动定点保持正常："
                        "箭头%d帧，ArUco%d帧，方框%d帧；立即开始子任务1"
                    ),
                    NODE_NAME,
                    counts["arrow"],
                    counts["aruco"],
                    counts["rectangle"],
                )
                return True

            elapsed = time.monotonic() - started_at
            if elapsed >= self.model_ready_timeout:
                rospy.logerr(
                    (
                        "%s：等待模型超过%.1fs：箭头=%d帧，ArUco=%d帧，"
                        "方框=%d帧"
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
                    "%s：等待三个模型预热，已等待%.1f/%.1fs："
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

    def wait_for_hover(self, context):
        started_at = time.monotonic()
        stable_started_at = None
        while not rospy.is_shutdown():
            with self.motion_lock:
                state = self.latest_motion_state
                state_time = self.latest_motion_state_wall_time

            now = time.monotonic()
            fresh = (
                state is not None
                and state_time is not None
                and now - state_time <= self.motion_state_timeout
            )
            hovering = fresh and state.state == MotionState.HOVER
            if hovering:
                if stable_started_at is None:
                    stable_started_at = now
                stable_elapsed = now - stable_started_at
                if stable_elapsed >= self.handoff_stable_seconds:
                    rospy.loginfo(
                        "%s：%s完成，MotionState.HOVER稳定%.1fs",
                        NODE_NAME,
                        context,
                        stable_elapsed,
                    )
                    return True
            else:
                stable_started_at = None

            elapsed = now - started_at
            if elapsed >= self.handoff_timeout:
                state_text = (
                    "无反馈"
                    if state is None
                    else str(state.state)
                )
                rospy.logerr(
                    (
                        "%s：%s等待HOVER超过%.1fs，当前state=%s，"
                        "反馈新鲜=%s"
                    ),
                    NODE_NAME,
                    context,
                    self.handoff_timeout,
                    state_text,
                    "是" if fresh else "否",
                )
                return False
            rospy.loginfo_throttle(
                1.0,
                "%s：%s等待MotionState.HOVER，已等待%.1f/%.1fs",
                NODE_NAME,
                context,
                elapsed,
                self.handoff_timeout,
            )
            self.rate.sleep()
        return False

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
        if (
            run_index != 1
            and not self.wait_for_new_model_frames("arrow", label)
        ):
            return False, "{}启动前箭头模型没有持续输出".format(label)

        rospy.loginfo(
            (
                "%s：[%s开始] 直接进入子函数，不启动子任务launch；"
                "本次启动悬停=%.1fs"
            ),
            NODE_NAME,
            label,
            self.task1_params["initial_hover_seconds"],
        )
        task = None
        try:
            task = self.EmbeddedTask1()
            success, detail = task.run()
        except Exception as error:
            rospy.logexception("%s：%s发生未处理异常", NODE_NAME, label)
            success = False
            detail = str(error)
        finally:
            if task is not None:
                self.deactivate_task(task)

        rospy.loginfo(
            "%s：[%s结束] success=%s，%s",
            NODE_NAME,
            label,
            str(success),
            detail,
        )
        return success, detail

    def run_subtask2(self):
        label = "ArUco子任务"
        if not self.wait_for_new_model_frames("aruco", label):
            return False, "{}启动前模型没有持续输出".format(label), None

        rospy.loginfo(
            "%s：[%s开始] 直接进入子函数，不启动子任务launch",
            NODE_NAME,
            label,
        )
        task = None
        try:
            task = self.EmbeddedTask2()
            task.run()
            success = bool(task.embedded_success)
            detail = task.embedded_detail
            marker_id = task.confirmed_marker_id
            color = task.confirmed_color if success else None
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
            "%s：[%s结束] success=%s，颜色=%s，%s",
            NODE_NAME,
            label,
            str(success),
            str(color),
            detail,
        )
        return success, detail, color

    def run_subtask3(self, target_color):
        label = "彩色方框投放子任务"
        if not self.wait_for_new_model_frames("rectangle", label):
            return False, "{}启动前模型没有持续输出".format(label)

        self.task3_params["target_color"] = str(target_color)
        if str(self.task3_params.get("operation_mode", "")).lower() != "auto":
            return False, (
                "整合任务要求子任务3的operation_mode=auto，当前为{}"
            ).format(self.task3_params.get("operation_mode"))

        rospy.loginfo(
            (
                "%s：[%s开始] 目标颜色=%s，直接进入子函数，"
                "不启动子任务launch"
            ),
            NODE_NAME,
            label,
            target_color,
        )
        task = None
        try:
            task = self.EmbeddedTask3()
            task.run()
            success = bool(task.embedded_success)
            detail = task.embedded_detail
        except Exception as error:
            rospy.logexception("%s：%s发生未处理异常", NODE_NAME, label)
            success = False
            detail = str(error)
        finally:
            if task is not None:
                self.deactivate_task(task)

        rospy.loginfo(
            "%s：[%s结束] success=%s，%s",
            NODE_NAME,
            label,
            str(success),
            detail,
        )
        return success, detail

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
            rospy.loginfo("%s：完整任务3成功：%s", NODE_NAME, detail)
        else:
            rospy.logerr("%s：完整任务3失败：%s", NODE_NAME, detail)
        rospy.sleep(0.2)

    def fail(self, detail):
        self.cancel_pub.publish(Empty())
        self.finish(False, detail)
        return False

    def run(self):
        startup_hold_goal = self.capture_startup_hold_goal()
        if startup_hold_goal is None:
            return self.fail("无法锁存模型预热期间的启动固定点")
        if not self.wait_for_all_models(startup_hold_goal):
            return self.fail("三个识别模型没有全部就绪或启动定点未进入HOVER")

        success, detail = self.run_subtask1(1)
        if not success:
            return self.fail("第一次箭头子任务失败：{}".format(detail))
        if not self.wait_for_hover("第一次箭头到子任务2交接"):
            return self.fail("第一次箭头结束后没有稳定进入HOVER")

        success, detail, target_color = self.run_subtask2()
        if not success or target_color is None:
            return self.fail("ArUco子任务失败：{}".format(detail))
        if not self.wait_for_hover("子任务2到第二次箭头交接"):
            return self.fail("子任务2结束后没有稳定进入HOVER")

        success, detail = self.run_subtask1(2)
        if not success:
            return self.fail("第二次箭头子任务失败：{}".format(detail))
        if not self.wait_for_hover("第二次箭头到子任务3交接"):
            return self.fail("第二次箭头结束后没有稳定进入HOVER")

        success, detail = self.run_subtask3(target_color)
        if not success:
            return self.fail("彩色方框投放子任务失败：{}".format(detail))
        if not self.wait_for_hover("完整任务结束定点"):
            return self.fail("子任务3结束后没有稳定进入HOVER")

        self.finish(
            True,
            (
                "第一次箭头、ArUco识别亮灯转向、第二次箭头和"
                "目标颜色{}方框投放均完成"
            ).format(target_color),
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
