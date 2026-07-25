#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务3子任务1、2、3完整联调协调节点。"""

import copy
import json
import math
import os
import re
import signal
import subprocess
import threading
import time

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion

from auv_control.msg import MotionState, TargetDetection


NODE_NAME = "test_task3_1_2_3_sequence"


def normalize_angle(angle):
    """把角度限制到[-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Task3Subtask123Sequence:
    """预热三个模型，并顺序调度任务3的三个执行子任务。"""

    TASK1_PREFIX = "test_task3_1_acquire_area"
    TASK2_PREFIX = "test_task3_2_get_task"
    TASK3_PREFIX = "test_task3_3_inspect_and_drop"
    ARROW_MODEL_ROLE = "arrow_model"
    MODEL_BUNDLE_ROLE = "model_bundle"
    TASK2_COLOR_BY_MARKER = {
        1: "yellow",
        2: "yellow",
        3: "green",
        4: "green",
        5: "red",
        6: "red",
    }
    MOTION_STATE_NAMES = {
        0: "IDLE",
        1: "ALIGN_PATH",
        2: "ALIGN_PATH_BRAKE",
        3: "TRANSLATE",
        4: "TRANSLATE_BRAKE",
        5: "ALIGN_FINAL",
        6: "FINAL_BRAKE",
        7: "CAPTURE",
        8: "HOVER",
        9: "SAFE",
    }

    def __init__(self):
        self.rate_hz = float(rospy.get_param("~rate", 10.0))
        self.map_frame = str(rospy.get_param("~map_frame", "map"))
        self.base_frame = str(rospy.get_param("~base_frame", "base_link"))
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 8.0))

        self.arrow_topic = str(rospy.get_param(
            "~arrow_topic", "/vision/arrow/direction"))
        self.aruco_topic = str(rospy.get_param(
            "~aruco_topic", "/vision/aruco/target_message"))
        self.aruco_web_detection_topic = str(rospy.get_param(
            "~aruco_web_detection_topic",
            "/vision/aruco/detections"))
        self.aruco_web_pose_topic = str(rospy.get_param(
            "~aruco_web_pose_topic", "/vision/aruco/pose"))
        self.rectangle_target_topic = str(rospy.get_param(
            "~rectangle_target_topic",
            "/vision/rectangle/target_message"))
        self.rectangle_detection_topic = str(rospy.get_param(
            "~rectangle_detection_topic",
            "/vision/rectangle/detections"))
        self.rectangle_pose_topic = str(rospy.get_param(
            "~rectangle_pose_topic", "/vision/rectangle/pose"))
        self.start_arrow_model = bool(rospy.get_param(
            "~start_arrow_model", True))
        self.start_aruco_model = bool(rospy.get_param(
            "~start_aruco_model", True))
        self.start_rectangle_model = bool(rospy.get_param(
            "~start_rectangle_model", True))
        self.start_fisheye_driver = bool(rospy.get_param(
            "~start_fisheye_driver", False))
        self.start_down_camera = bool(rospy.get_param(
            "~start_down_camera", False))
        self.start_down_splitter = bool(rospy.get_param(
            "~start_down_splitter", False))
        self.start_arrow_web = bool(rospy.get_param(
            "~start_arrow_web", True))
        self.start_aruco_web = bool(rospy.get_param(
            "~start_aruco_web", True))
        self.start_rectangle_web = bool(rospy.get_param(
            "~start_rectangle_web", True))
        self.arrow_web_port = int(rospy.get_param("~arrow_web_port", 8083))
        self.aruco_web_port = int(rospy.get_param("~aruco_web_port", 8082))
        self.rectangle_web_port = int(rospy.get_param(
            "~rectangle_web_port", 8080))
        self.model_ready_timeout = float(rospy.get_param(
            "~model_ready_timeout", 90.0))
        self.model_settle_seconds = float(rospy.get_param(
            "~model_settle_seconds", 1.0))
        self.model_output_timeout = float(rospy.get_param(
            "~model_output_timeout", 2.0))
        self.model_recovery_timeout = float(rospy.get_param(
            "~model_recovery_timeout", 10.0))
        self.model_handoff_required_frames = int(rospy.get_param(
            "~model_handoff_required_frames", 3))

        self.motion_goal_topic = str(rospy.get_param(
            "~motion_goal_topic", "/cmd/motion/goal"))
        self.motion_cancel_topic = str(rospy.get_param(
            "~motion_cancel_topic", "/cmd/motion/cancel"))
        self.motion_state_topic = str(rospy.get_param(
            "~motion_state_topic", "/motion/state"))
        self.motion_state_timeout = float(rospy.get_param(
            "~motion_state_timeout", 0.5))
        self.initial_hold_timeout = float(rospy.get_param(
            "~initial_hold_timeout", 30.0))
        self.initial_hold_stable_seconds = float(rospy.get_param(
            "~initial_hold_stable_seconds", 1.0))
        self.post_task1_hold_timeout = float(rospy.get_param(
            "~post_task1_hold_timeout", 30.0))
        self.post_task1_stable_seconds = float(rospy.get_param(
            "~post_task1_stable_seconds", 1.0))
        self.post_task2_hold_timeout = float(rospy.get_param(
            "~post_task2_hold_timeout", 30.0))
        self.post_task2_stable_seconds = float(rospy.get_param(
            "~post_task2_stable_seconds", 1.0))
        self.post_second_arrow_hold_timeout = float(rospy.get_param(
            "~post_second_arrow_hold_timeout", 30.0))
        self.post_second_arrow_stable_seconds = float(rospy.get_param(
            "~post_second_arrow_stable_seconds", 1.0))

        self.goal_match_position_tolerance = float(rospy.get_param(
            "~goal_match_position_tolerance", 0.03))
        self.goal_match_depth_tolerance = float(rospy.get_param(
            "~goal_match_depth_tolerance", 0.03))
        self.goal_match_yaw_tolerance = math.radians(float(rospy.get_param(
            "~goal_match_yaw_tolerance_deg", 2.0)))
        self.arrival_position_tolerance = float(rospy.get_param(
            "~arrival_position_tolerance", 0.05))
        self.arrival_yaw_tolerance = math.radians(float(rospy.get_param(
            "~arrival_yaw_tolerance_deg", 5.0)))
        self.arrival_max_horizontal_speed = float(rospy.get_param(
            "~arrival_max_horizontal_speed", 0.03))
        self.arrival_max_yaw_rate = float(rospy.get_param(
            "~arrival_max_yaw_rate", 0.05))

        self.task1_stage_timeout = float(rospy.get_param(
            "~task1_stage_timeout", 360.0))
        self.second_arrow_stage_timeout = float(rospy.get_param(
            "~second_arrow_stage_timeout", 360.0))
        self.task2_stage_timeout = float(rospy.get_param(
            "~task2_stage_timeout", 180.0))
        self.task3_stage_timeout = float(rospy.get_param(
            "~task3_stage_timeout", 360.0))
        self.task3_color_source = str(rospy.get_param(
            "~task3_color_source", "task2")).strip().lower()
        self.task3_handoff_timeout = float(rospy.get_param(
            "~task3_handoff_timeout", 10.0))
        self.task3_handoff_goal_count = int(rospy.get_param(
            "~task3_handoff_goal_count", 2))
        self.handoff_goal_position_tolerance = float(rospy.get_param(
            "~handoff_goal_position_tolerance", 0.10))
        self.handoff_goal_depth_tolerance = float(rospy.get_param(
            "~handoff_goal_depth_tolerance", 0.10))
        self.handoff_goal_yaw_tolerance = math.radians(float(rospy.get_param(
            "~handoff_goal_yaw_tolerance_deg", 10.0)))
        self.safe_stop_timeout = float(rospy.get_param(
            "~safe_stop_timeout", 30.0))
        self.safe_stop_stable_seconds = float(rospy.get_param(
            "~safe_stop_stable_seconds", 1.0))
        self.control_health_grace_timeout = float(rospy.get_param(
            "~control_health_grace_timeout", 3.0))

        self.finished_topic = str(rospy.get_param(
            "~finished_topic", "/finished"))
        self.sequence_finished_topic = str(rospy.get_param(
            "~sequence_finished_topic", "/task3_sequence/finished"))
        self.stage_shutdown_grace = float(rospy.get_param(
            "~stage_shutdown_grace", 1.0))
        self.process_sigint_timeout = float(rospy.get_param(
            "~process_sigint_timeout", 5.0))
        self.process_sigterm_timeout = float(rospy.get_param(
            "~process_sigterm_timeout", 2.0))

        self._validate_parameters()

        self.tf_listener = tf.TransformListener()
        self.motion_state_lock = threading.Lock()
        self.latest_motion_state = None
        self.latest_motion_state_wall_time = None

        self.arrow_ready = False
        self.aruco_ready = False
        self.rectangle_ready = False
        self.arrow_message_count = 0
        self.aruco_message_count = 0
        self.rectangle_message_count = 0
        self.arrow_last_message_wall_time = None
        self.aruco_last_message_wall_time = None
        self.rectangle_last_message_wall_time = None
        self.model_lock = threading.Lock()

        self.stage_lock = threading.Lock()
        self.current_stage = None
        self.stage_result = None
        self.stage_event = threading.Event()
        self.child_process = None
        self.child_process_role = None
        self.managed_processes = {}
        self.process_lock = threading.Lock()
        self.shutdown_started = False
        self.final_result_published = False
        self.task2_marker_id = None
        self.task2_target_color = None

        self.handoff_lock = threading.Lock()
        self.task3_handoff_goal_count_received = 0
        self.latest_task3_handoff_goal = None
        self.latest_task3_handoff_caller = ""
        self.task3_handoff_goals = []
        self.task3_handoff_unauthorized_caller = ""

        self.motion_goal_pub = rospy.Publisher(
            self.motion_goal_topic, PoseStamped, queue_size=1)
        self.motion_cancel_pub = rospy.Publisher(
            self.motion_cancel_topic, Empty, queue_size=1)
        self.sequence_finished_pub = rospy.Publisher(
            self.sequence_finished_topic, String, queue_size=1, latch=True)
        rospy.Subscriber(
            self.motion_state_topic,
            MotionState,
            self.motion_state_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.motion_goal_topic,
            PoseStamped,
            self.motion_goal_observer_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.arrow_topic,
            String,
            self.arrow_ready_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.aruco_topic,
            TargetDetection,
            self.aruco_ready_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.rectangle_detection_topic,
            String,
            self.rectangle_ready_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.finished_topic,
            String,
            self.finished_callback,
            queue_size=10,
        )
        rospy.on_shutdown(self.on_shutdown)
        self.rate = rospy.Rate(self.rate_hz)
        self.log_configuration()

    def _validate_parameters(self):
        positive_values = {
            "rate": self.rate_hz,
            "tf_timeout": self.tf_timeout,
            "model_ready_timeout": self.model_ready_timeout,
            "model_output_timeout": self.model_output_timeout,
            "model_recovery_timeout": self.model_recovery_timeout,
            "motion_state_timeout": self.motion_state_timeout,
            "initial_hold_timeout": self.initial_hold_timeout,
            "post_task1_hold_timeout": self.post_task1_hold_timeout,
            "post_task2_hold_timeout": self.post_task2_hold_timeout,
            "post_second_arrow_hold_timeout": self.post_second_arrow_hold_timeout,
            "task1_stage_timeout": self.task1_stage_timeout,
            "second_arrow_stage_timeout": self.second_arrow_stage_timeout,
            "task2_stage_timeout": self.task2_stage_timeout,
            "task3_stage_timeout": self.task3_stage_timeout,
            "task3_handoff_timeout": self.task3_handoff_timeout,
            "safe_stop_timeout": self.safe_stop_timeout,
            "control_health_grace_timeout": self.control_health_grace_timeout,
            "process_sigint_timeout": self.process_sigint_timeout,
            "process_sigterm_timeout": self.process_sigterm_timeout,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError("参数{}必须大于0".format(name))
        if self.task3_handoff_goal_count <= 0:
            raise ValueError("task3_handoff_goal_count必须大于0")
        if self.model_handoff_required_frames <= 0:
            raise ValueError("model_handoff_required_frames必须大于0")
        if self.task3_color_source not in ("task2", "manual"):
            raise ValueError("task3_color_source必须为task2或manual")
        if (
                not self.start_arrow_model
                or not self.start_aruco_model
                or not self.start_rectangle_model):
            raise ValueError(
                "完整联调要求三个识别模型都为true，"
                "以便协调节点统一管理模型生命周期")
        for name, port in (
                ("arrow_web_port", self.arrow_web_port),
                ("aruco_web_port", self.aruco_web_port),
                ("rectangle_web_port", self.rectangle_web_port)):
            if port <= 0 or port > 65535:
                raise ValueError("参数{}必须在1到65535之间".format(name))
        enabled_web_ports = []
        if self.start_arrow_web:
            enabled_web_ports.append(self.arrow_web_port)
        if self.start_aruco_web:
            enabled_web_ports.append(self.aruco_web_port)
        if self.start_rectangle_web:
            enabled_web_ports.append(self.rectangle_web_port)
        if len(enabled_web_ports) != len(set(enabled_web_ports)):
            raise ValueError("三个模型启用的Web页面不能使用相同端口")
        isolated_topics = {
            "箭头方向": self.arrow_topic,
            "ArUco目标结果": self.aruco_topic,
            "ArUco网页检测": self.aruco_web_detection_topic,
            "ArUco网页位姿": self.aruco_web_pose_topic,
            "方框目标结果": self.rectangle_target_topic,
            "方框网页检测": self.rectangle_detection_topic,
            "方框网页位姿": self.rectangle_pose_topic,
        }
        expected_prefixes = {
            "箭头方向": "/vision/arrow/",
            "ArUco目标结果": "/vision/aruco/",
            "ArUco网页检测": "/vision/aruco/",
            "ArUco网页位姿": "/vision/aruco/",
            "方框目标结果": "/vision/rectangle/",
            "方框网页检测": "/vision/rectangle/",
            "方框网页位姿": "/vision/rectangle/",
        }
        for label, topic in isolated_topics.items():
            expected_prefix = expected_prefixes[label]
            if not topic.startswith(expected_prefix):
                raise ValueError(
                    "{}话题必须使用最新{}前缀".format(label, expected_prefix))
        duplicate_topics = sorted({
            topic for topic in isolated_topics.values()
            if list(isolated_topics.values()).count(topic) > 1
        })
        if duplicate_topics:
            raise ValueError(
                "模型隔离话题不能重复：{}".format(", ".join(duplicate_topics)))
        non_negative_values = {
            "model_settle_seconds": self.model_settle_seconds,
            "initial_hold_stable_seconds": self.initial_hold_stable_seconds,
            "post_task1_stable_seconds": self.post_task1_stable_seconds,
            "post_task2_stable_seconds": self.post_task2_stable_seconds,
            "post_second_arrow_stable_seconds": (
                self.post_second_arrow_stable_seconds),
            "handoff_goal_position_tolerance": self.handoff_goal_position_tolerance,
            "handoff_goal_depth_tolerance": self.handoff_goal_depth_tolerance,
            "handoff_goal_yaw_tolerance": self.handoff_goal_yaw_tolerance,
            "safe_stop_stable_seconds": self.safe_stop_stable_seconds,
        }
        for name, value in non_negative_values.items():
            if value < 0.0:
                raise ValueError("参数{}不能小于0".format(name))

    def log_configuration(self):
        rospy.loginfo(
            (
                "%s：[联调职责] 本节点只管理模型和执行顺序；"
                "两次箭头的前进搜索由子任务1负责，ArUco后的转向由子任务2负责，"
                "方框搜索和投放由子任务3负责"
            ),
            NODE_NAME,
        )
        rospy.loginfo(
            (
                "%s：[联调参数] 到达阈值：位置<=%.3fm，航向<=%.1fdeg，"
                "水平速度<=%.3fm/s，航向角速度<=%.3frad/s"
            ),
            NODE_NAME,
            self.arrival_position_tolerance,
            math.degrees(self.arrival_yaw_tolerance),
            self.arrival_max_horizontal_speed,
            self.arrival_max_yaw_rate,
        )
        rospy.loginfo(
            (
                "%s：[联调超时] 模型就绪=%.1fs，子任务1=%.1fs，"
                "子任务2=%.1fs，第二次箭头=%.1fs，子任务3=%.1fs"
            ),
            NODE_NAME,
            self.model_ready_timeout,
            self.task1_stage_timeout,
            self.task2_stage_timeout,
            self.second_arrow_stage_timeout,
            self.task3_stage_timeout,
        )
        rospy.loginfo(
            (
                "%s：[交接参数] 子任务1后复稳=%.1fs，子任务2后复稳=%.1fs；"
                "第二次箭头后复稳=%.1fs；子任务3接管超时=%.1fs，"
                "需连续收到%d个目标；"
                "模型交接需%d个新帧；最终安全悬停=%.1fs"
            ),
            NODE_NAME,
            self.post_task1_stable_seconds,
            self.post_task2_stable_seconds,
            self.post_second_arrow_stable_seconds,
            self.task3_handoff_timeout,
            self.task3_handoff_goal_count,
            self.model_handoff_required_frames,
            self.safe_stop_stable_seconds,
        )
        rospy.loginfo(
            "%s：[话题隔离] 子任务1=%s，子任务2=%s，子任务3=%s",
            NODE_NAME,
            self.arrow_topic,
            self.aruco_topic,
            self.rectangle_detection_topic,
        )
        rospy.loginfo(
            (
                "%s：[模型输出隔离] 箭头方向=%s；ArUco={目标:%s,检测:%s,位姿:%s}；"
                "方框={目标:%s,检测:%s,位姿:%s}"
            ),
            NODE_NAME,
            self.arrow_topic,
            self.aruco_topic,
            self.aruco_web_detection_topic,
            self.aruco_web_pose_topic,
            self.rectangle_target_topic,
            self.rectangle_detection_topic,
            self.rectangle_pose_topic,
        )
        rospy.loginfo(
            (
                "%s：[进程生命周期] 默认启动环境只有公共控制和传感器；"
                "不扫描、不清理历史任务或模型。三个模型同时预热；"
                "箭头模型保持到第二次箭头完成，ArUco和方框模型常驻到总任务结束"
            ),
            NODE_NAME,
        )
        rospy.loginfo(
            "%s：[进程关闭] SIGINT等待=%.1fs，SIGTERM等待=%.1fs",
            NODE_NAME,
            self.process_sigint_timeout,
            self.process_sigterm_timeout,
        )
        rospy.loginfo(
            "%s：[模型进程] 箭头Web=%s:%d，ArUco Web=%s:%d，方框 Web=%s:%d",
            NODE_NAME,
            "开启" if self.start_arrow_web else "关闭",
            self.arrow_web_port,
            "开启" if self.start_aruco_web else "关闭",
            self.aruco_web_port,
            "开启" if self.start_rectangle_web else "关闭",
            self.rectangle_web_port,
        )
        rospy.loginfo(
            (
                "%s：[参数归属] 两次箭头的搜索和对准参数读取"
                "task3_subtask1_acquire_area.launch；"
                "ArUco识别、亮灯和转向参数读取task3_subtask2_get_task.launch；"
                "子任务3参数读取task3_subtask3_inspect_and_drop.launch；"
                "完整联调launch只保留阶段超时和交接参数"
            ),
            NODE_NAME,
        )
        rospy.loginfo(
            "%s：[颜色交接] 子任务3颜色来源=%s",
            NODE_NAME,
            (
                "子任务2的ArUco识别结果"
                if self.task3_color_source == "task2"
                else "子任务3 launch中的手动target_color"
            ),
        )

    @staticmethod
    def _yaw_from_pose(pose):
        quaternion = pose.orientation
        norm = math.sqrt(
            quaternion.x * quaternion.x
            + quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
            + quaternion.w * quaternion.w
        )
        if norm < 1e-6:
            return None
        return euler_from_quaternion([
            quaternion.x / norm,
            quaternion.y / norm,
            quaternion.z / norm,
            quaternion.w / norm,
        ])[2]

    def motion_state_callback(self, message):
        with self.motion_state_lock:
            self.latest_motion_state = message
            self.latest_motion_state_wall_time = time.monotonic()

    def motion_goal_observer_callback(self, message):
        """监听子任务3的首批目标，用于确认控制权已经接管。"""
        connection_header = getattr(message, "_connection_header", {}) or {}
        caller_id = str(connection_header.get("callerid", ""))
        if caller_id == rospy.get_name():
            return
        with self.stage_lock:
            current_stage = self.current_stage
        if current_stage != self.TASK3_PREFIX:
            return
        if self.TASK3_PREFIX not in caller_id:
            with self.handoff_lock:
                self.task3_handoff_unauthorized_caller = caller_id or "未知节点"
            rospy.logwarn_throttle(
                1.0,
                "%s：[控制权警告] 子任务3阶段检测到其他目标发布者：%s",
                NODE_NAME,
                caller_id or "未知节点",
            )
            return
        with self.handoff_lock:
            self.task3_handoff_goal_count_received += 1
            self.latest_task3_handoff_goal = copy.deepcopy(message)
            self.latest_task3_handoff_caller = caller_id
            self.task3_handoff_goals.append((
                self.task3_handoff_goal_count_received,
                copy.deepcopy(message),
                caller_id,
            ))
            self.task3_handoff_goals = self.task3_handoff_goals[-20:]

    def arrow_ready_callback(self, message):
        now = time.monotonic()
        try:
            payload = json.loads(message.data)
            if not isinstance(payload, dict):
                raise ValueError("箭头方向消息不是JSON对象")
            valid = bool(payload.get("valid", False))
            confidence = payload.get("confidence", "未知")
            angle = payload.get("angle_deg", "未知")
        except (TypeError, ValueError, AttributeError):
            rospy.logwarn_throttle(
                2.0,
                "%s：[箭头模型监测] 收到无法解析的消息，不计入模型就绪帧",
                NODE_NAME,
            )
            return
        with self.model_lock:
            first_message = not self.arrow_ready
            self.arrow_ready = True
            self.arrow_message_count += 1
            self.arrow_last_message_wall_time = now
            message_count = self.arrow_message_count
        if first_message:
            rospy.loginfo(
                "%s：箭头模型已有输出，话题=%s",
                NODE_NAME,
                self.arrow_topic,
            )
        rospy.loginfo_throttle(
            5.0,
            (
                "%s：[箭头模型监测] 第%d帧，valid=%s，confidence=%s，"
                "angle_deg=%s"
            ),
            NODE_NAME,
            message_count,
            str(valid),
            str(confidence),
            str(angle),
        )

    def aruco_ready_callback(self, message):
        now = time.monotonic()
        with self.model_lock:
            first_message = not self.aruco_ready
            self.aruco_ready = True
            self.aruco_message_count += 1
            self.aruco_last_message_wall_time = now
            message_count = self.aruco_message_count
        if first_message:
            rospy.loginfo("%s：ArUco模型已有输出，话题=%s", NODE_NAME, self.aruco_topic)
        rospy.loginfo_throttle(
            5.0,
            (
                "%s：[ArUco模型监测] 第%d帧，type=%s，ID=%s，conf=%.3f；"
                "子任务2启动后由其自身日志判定3/10帧"
            ),
            NODE_NAME,
            message_count,
            str(message.type),
            str(message.class_name),
            float(message.conf),
        )

    def rectangle_ready_callback(self, message):
        now = time.monotonic()
        try:
            payload = json.loads(message.data)
            detections = payload.get("detections")
            if not isinstance(payload, dict) or not isinstance(detections, list):
                raise ValueError("JSON缺少detections列表")
            detection_count = str(payload.get("count", len(detections)))
        except (TypeError, ValueError, AttributeError):
            rospy.logwarn_throttle(
                2.0,
                "%s：[方框模型监测] 收到无法解析的消息，不计入模型就绪帧",
                NODE_NAME,
            )
            return
        with self.model_lock:
            first_message = not self.rectangle_ready
            self.rectangle_ready = True
            self.rectangle_message_count += 1
            self.rectangle_last_message_wall_time = now
            message_count = self.rectangle_message_count
        if first_message:
            rospy.loginfo(
                "%s：彩色方框模型已有输出，话题=%s",
                NODE_NAME,
                self.rectangle_detection_topic,
            )
        rospy.loginfo_throttle(
            5.0,
            (
                "%s：[方框模型监测] 第%d帧，本帧检测数量=%s；"
                "模型全程常驻，子任务3启动前不执行方框动作"
            ),
            NODE_NAME,
            message_count,
            detection_count,
        )

    def finished_callback(self, message):
        text = str(message.data).strip()
        with self.stage_lock:
            expected = self.current_stage
            if expected is None or not text.startswith(expected):
                rospy.loginfo_throttle(
                    2.0,
                    "%s：忽略非当前阶段的完成消息：%s",
                    NODE_NAME,
                    text,
                )
                return
            self.stage_result = text
            self.stage_event.set()
        rospy.loginfo("%s：收到当前阶段完成消息：%s", NODE_NAME, text)

    def capture_current_goal(self):
        deadline = time.monotonic() + self.tf_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                translation, rotation = self.tf_listener.lookupTransform(
                    self.map_frame, self.base_frame, rospy.Time(0))
            except tf.Exception as error:
                rospy.logwarn_throttle(
                    1.0,
                    "%s：等待TF %s -> %s：%s",
                    NODE_NAME,
                    self.map_frame,
                    self.base_frame,
                    str(error),
                )
                self.rate.sleep()
                continue

            values = tuple(translation) + tuple(rotation)
            if not all(math.isfinite(float(value)) for value in values):
                rospy.logwarn_throttle(1.0, "%s：忽略包含无效值的TF", NODE_NAME)
                self.rate.sleep()
                continue

            goal = PoseStamped()
            goal.header.frame_id = self.map_frame
            goal.pose.position.x = float(translation[0])
            goal.pose.position.y = float(translation[1])
            goal.pose.position.z = float(translation[2])
            goal.pose.orientation.x = float(rotation[0])
            goal.pose.orientation.y = float(rotation[1])
            goal.pose.orientation.z = float(rotation[2])
            goal.pose.orientation.w = float(rotation[3])
            yaw = self._yaw_from_pose(goal.pose)
            if yaw is None:
                self.rate.sleep()
                continue
            rospy.loginfo(
                "%s：锁存启动定点 x=%.3f, y=%.3f, z=%.3f, yaw=%.1fdeg",
                NODE_NAME,
                goal.pose.position.x,
                goal.pose.position.y,
                goal.pose.position.z,
                math.degrees(yaw),
            )
            return goal

        return None

    def publish_goal(self, goal):
        goal.header.stamp = rospy.Time.now()
        self.motion_goal_pub.publish(goal)

    def goal_arrival_status(self, expected_goal, goal_started_wall_time):
        with self.motion_state_lock:
            state = self.latest_motion_state
            received_at = self.latest_motion_state_wall_time

        if state is None or received_at is None:
            return False, "尚未收到motion状态"
        age = time.monotonic() - received_at
        if age > self.motion_state_timeout:
            return False, "motion状态超时{:.2f}s".format(age)
        if received_at < goal_started_wall_time:
            return False, "等待目标发布后的新motion状态"
        if not state.startup_complete:
            return False, "motion_supervisor尚未完成启动定点"
        if state.state != MotionState.HOVER:
            return False, "当前控制状态={}，等待HOVER".format(state.state)
        if not state.goal_active:
            return False, "motion状态尚无活动目标"

        expected_yaw = self._yaw_from_pose(expected_goal.pose)
        state_goal_yaw = self._yaw_from_pose(state.goal.pose)
        if expected_yaw is None or state_goal_yaw is None:
            return False, "目标四元数无效"
        goal_xy_error = math.hypot(
            state.goal.pose.position.x - expected_goal.pose.position.x,
            state.goal.pose.position.y - expected_goal.pose.position.y,
        )
        goal_z_error = abs(
            state.goal.pose.position.z - expected_goal.pose.position.z)
        goal_yaw_error = abs(normalize_angle(state_goal_yaw - expected_yaw))
        if goal_xy_error > self.goal_match_position_tolerance:
            return False, "状态目标与任务目标水平不匹配{:.3f}m".format(goal_xy_error)
        if goal_z_error > self.goal_match_depth_tolerance:
            return False, "状态目标与任务目标深度不匹配{:.3f}m".format(goal_z_error)
        if goal_yaw_error > self.goal_match_yaw_tolerance:
            return False, "状态目标与任务目标航向不匹配{:.1f}deg".format(
                math.degrees(goal_yaw_error))
        if abs(state.base_position_error) > self.arrival_position_tolerance:
            return False, "位置误差{:.3f}m".format(state.base_position_error)
        if abs(state.yaw_error) > self.arrival_yaw_tolerance:
            return False, "航向误差{:.1f}deg".format(math.degrees(state.yaw_error))
        if abs(state.horizontal_speed) > self.arrival_max_horizontal_speed:
            return False, "水平速度{:.3f}m/s".format(state.horizontal_speed)
        if abs(state.yaw_rate) > self.arrival_max_yaw_rate:
            return False, "航向角速度{:.3f}rad/s".format(state.yaw_rate)
        return True, "HOVER且位置、航向、速度均达到阈值"

    def motion_state_debug_text(self):
        with self.motion_state_lock:
            state = self.latest_motion_state
            received_at = self.latest_motion_state_wall_time
        if state is None or received_at is None:
            return "motion状态=未收到"
        return (
            "状态={}，启动={}，位置误差={:.3f}m，航向误差={:.1f}deg，"
            "水平速度={:.3f}m/s，航向角速度={:.3f}rad/s，消息年龄={:.2f}s"
        ).format(
            self.MOTION_STATE_NAMES.get(state.state, str(state.state)),
            "完成" if state.startup_complete else "未完成",
            state.base_position_error,
            math.degrees(state.yaw_error),
            state.horizontal_speed,
            state.yaw_rate,
            max(0.0, time.monotonic() - received_at),
        )

    def wait_for_goal(self, goal, timeout, stable_seconds, stage_name):
        started_at = time.monotonic()
        stable_started_at = None
        while not rospy.is_shutdown():
            now = time.monotonic()
            if now - started_at >= timeout:
                rospy.logerr("%s：%s超过%.1fs仍未稳定到达", NODE_NAME, stage_name, timeout)
                return False

            self.publish_goal(goal)
            arrived, detail = self.goal_arrival_status(goal, started_at)
            if arrived:
                if stable_started_at is None:
                    stable_started_at = now
                    rospy.loginfo("%s：%s首次达到阈值，开始稳定计时", NODE_NAME, stage_name)
                stable_elapsed = now - stable_started_at
                rospy.loginfo_throttle(
                    1.0,
                    "%s：%s稳定确认 %.1f/%.1fs，%s；%s",
                    NODE_NAME,
                    stage_name,
                    min(stable_elapsed, stable_seconds),
                    stable_seconds,
                    detail,
                    self.motion_state_debug_text(),
                )
                if stable_elapsed >= stable_seconds:
                    return True
            else:
                if stable_started_at is not None:
                    rospy.logwarn("%s：%s稳定计时中断：%s", NODE_NAME, stage_name, detail)
                stable_started_at = None
                rospy.loginfo_throttle(
                    1.0,
                    "%s：等待%s，剩余%.1fs：%s；%s",
                    NODE_NAME,
                    stage_name,
                    max(0.0, timeout - (now - started_at)),
                    detail,
                    self.motion_state_debug_text(),
                )
            self.rate.sleep()
        return False

    def wait_for_models(self, hold_goal):
        started_at = time.monotonic()
        all_ready_at = None
        ready_start_counts = None
        while not rospy.is_shutdown():
            self.publish_goal(hold_goal)
            if not self.managed_process_alive(self.ARROW_MODEL_ROLE):
                rospy.logerr("%s：箭头模型进程组已提前退出", NODE_NAME)
                return False
            if not self.managed_process_alive(self.MODEL_BUNDLE_ROLE):
                rospy.logerr("%s：ArUco和方框模型进程组已提前退出", NODE_NAME)
                return False
            with self.model_lock:
                arrow_ready = self.arrow_ready
                aruco_ready = self.aruco_ready
                rectangle_ready = self.rectangle_ready
                arrow_count = self.arrow_message_count
                aruco_count = self.aruco_message_count
                rectangle_count = self.rectangle_message_count
                arrow_last = self.arrow_last_message_wall_time
                aruco_last = self.aruco_last_message_wall_time
                rectangle_last = self.rectangle_last_message_wall_time

            now = time.monotonic()
            arrow_age = None if arrow_last is None else now - arrow_last
            aruco_age = None if aruco_last is None else now - aruco_last
            rectangle_age = (
                None if rectangle_last is None else now - rectangle_last)
            arrow_fresh = (
                arrow_ready
                and arrow_age is not None
                and arrow_age <= self.model_output_timeout
            )
            aruco_fresh = (
                aruco_ready
                and aruco_age is not None
                and aruco_age <= self.model_output_timeout
            )
            rectangle_fresh = (
                rectangle_ready
                and rectangle_age is not None
                and rectangle_age <= self.model_output_timeout
            )
            if arrow_fresh and aruco_fresh and rectangle_fresh:
                if all_ready_at is None:
                    all_ready_at = now
                    ready_start_counts = (
                        arrow_count,
                        aruco_count,
                        rectangle_count,
                    )
                    rospy.loginfo(
                        (
                            "%s：三个识别模型输出均新鲜，再等待%.1fs且"
                            "各接收%d个新帧确认连续输出"
                        ),
                        NODE_NAME,
                        self.model_settle_seconds,
                        self.model_handoff_required_frames,
                    )
                arrow_new_frames = arrow_count - ready_start_counts[0]
                aruco_new_frames = aruco_count - ready_start_counts[1]
                rectangle_new_frames = rectangle_count - ready_start_counts[2]
                if (
                        now - all_ready_at >= self.model_settle_seconds
                        and arrow_new_frames >= self.model_handoff_required_frames
                        and aruco_new_frames >= self.model_handoff_required_frames
                        and rectangle_new_frames >= self.model_handoff_required_frames):
                    rospy.loginfo(
                        "%s：三个模型已完成预热并持续输出，开始执行子任务1",
                        NODE_NAME,
                    )
                    return True
            else:
                all_ready_at = None
                ready_start_counts = None

            elapsed = now - started_at
            if elapsed >= self.model_ready_timeout:
                rospy.logerr(
                    (
                        "%s：等待模型超过%.1fs，箭头=%s(%d帧，年龄=%s)，"
                        "ArUco=%s(%d帧，年龄=%s)，"
                        "方框=%s(%d帧，年龄=%s)"
                    ),
                    NODE_NAME,
                    self.model_ready_timeout,
                    "新鲜" if arrow_fresh else "未就绪/超时",
                    arrow_count,
                    "无" if arrow_age is None else "{:.2f}s".format(arrow_age),
                    "新鲜" if aruco_fresh else "未就绪/超时",
                    aruco_count,
                    "无" if aruco_age is None else "{:.2f}s".format(aruco_age),
                    "新鲜" if rectangle_fresh else "未就绪/超时",
                    rectangle_count,
                    "无" if rectangle_age is None else "{:.2f}s".format(rectangle_age),
                )
                return False
            rospy.loginfo_throttle(
                2.0,
                (
                    "%s：模型预热中，箭头=%s(%d帧，年龄=%s)，"
                    "ArUco=%s(%d帧，年龄=%s)，"
                    "方框=%s(%d帧，年龄=%s)，已等待%.1fs"
                ),
                NODE_NAME,
                "新鲜" if arrow_fresh else "等待",
                arrow_count,
                "无" if arrow_age is None else "{:.2f}s".format(arrow_age),
                "新鲜" if aruco_fresh else "等待",
                aruco_count,
                "无" if aruco_age is None else "{:.2f}s".format(aruco_age),
                "新鲜" if rectangle_fresh else "等待",
                rectangle_count,
                "无" if rectangle_age is None else "{:.2f}s".format(rectangle_age),
                elapsed,
            )
            self.rate.sleep()
        return False

    def _wait_for_model_fresh(
            self, hold_goal, reason, model_label, count_attribute,
            last_message_attribute, process_role, process_label):
        """在阶段交接前确认指定常驻模型仍持续出帧。"""
        started_at = time.monotonic()
        with self.model_lock:
            baseline_count = getattr(self, count_attribute)
        while not rospy.is_shutdown():
            self.publish_goal(hold_goal)
            if not self.managed_process_alive(process_role):
                rospy.logerr(
                    "%s：[模型交接检查] %s失败，%s进程组已经退出",
                    NODE_NAME,
                    reason,
                    process_label,
                )
                return False
            with self.model_lock:
                last_message = getattr(self, last_message_attribute)
                message_count = getattr(self, count_attribute)
            now = time.monotonic()
            age = None if last_message is None else now - last_message
            new_frame_count = max(0, message_count - baseline_count)
            if (
                    age is not None
                    and age <= self.model_output_timeout
                    and new_frame_count >= self.model_handoff_required_frames):
                rospy.loginfo(
                    (
                        "%s：[模型交接检查] %s通过，新收到%d/%d帧，"
                        "%s总帧=%d，消息年龄=%.2fs"
                    ),
                    NODE_NAME,
                    reason,
                    new_frame_count,
                    self.model_handoff_required_frames,
                    model_label,
                    message_count,
                    age,
                )
                return True
            elapsed = now - started_at
            if elapsed >= self.model_recovery_timeout:
                rospy.logerr(
                    "%s：[模型交接检查] %s失败，%.1fs内%s没有恢复连续输出",
                    NODE_NAME,
                    reason,
                    self.model_recovery_timeout,
                    model_label,
                )
                return False
            rospy.logwarn_throttle(
                1.0,
                "%s：[模型交接检查] %s等待%s恢复，消息年龄=%s，剩余%.1fs",
                NODE_NAME,
                reason,
                model_label,
                "无" if age is None else "{:.2f}s".format(age),
                max(0.0, self.model_recovery_timeout - elapsed),
            )
            rospy.loginfo_throttle(
                1.0,
                "%s：[模型交接检查] %s新帧进度%d/%d",
                NODE_NAME,
                reason,
                new_frame_count,
                self.model_handoff_required_frames,
            )
            self.rate.sleep()
        return False

    def wait_for_arrow_model_fresh(self, hold_goal, reason):
        return self._wait_for_model_fresh(
            hold_goal,
            reason,
            "箭头模型",
            "arrow_message_count",
            "arrow_last_message_wall_time",
            self.ARROW_MODEL_ROLE,
            "箭头模型",
        )

    def wait_for_aruco_model_fresh(self, hold_goal, reason):
        return self._wait_for_model_fresh(
            hold_goal,
            reason,
            "ArUco模型",
            "aruco_message_count",
            "aruco_last_message_wall_time",
            self.MODEL_BUNDLE_ROLE,
            "ArUco和方框模型",
        )

    def wait_for_rectangle_model_fresh(self, hold_goal, reason):
        return self._wait_for_model_fresh(
            hold_goal,
            reason,
            "方框模型",
            "rectangle_message_count",
            "rectangle_last_message_wall_time",
            self.MODEL_BUNDLE_ROLE,
            "ArUco和方框模型",
        )

    @staticmethod
    def _launch_bool(value):
        return "true" if value else "false"

    def arrow_model_command(self):
        """递归启动仅箭头模型模式，使箭头Web节点处于独立命名空间。"""
        return [
            "roslaunch", "auv_control", "task3_subtask1_2_3_sequence.launch",
            "coordinator_enabled:=false",
            "arrow_model_only:=true",
            "start_arrow_model:={}".format(
                self._launch_bool(self.start_arrow_model)),
            "start_down_camera:=false",
            "start_down_splitter:=false",
            "start_arrow_web:={}".format(
                self._launch_bool(self.start_arrow_web)),
            "arrow_web_port:={}".format(self.arrow_web_port),
            "arrow_topic:={}".format(self.arrow_topic),
        ]

    def model_bundle_command(self):
        """递归启动同一launch的仅模型模式，保留两套模型的话题隔离。"""
        return [
            "roslaunch", "auv_control", "task3_subtask1_2_3_sequence.launch",
            "coordinator_enabled:=false",
            "model_bundle_only:=true",
            "start_aruco_model:={}".format(
                self._launch_bool(self.start_aruco_model)),
            "start_rectangle_model:={}".format(
                self._launch_bool(self.start_rectangle_model)),
            "start_fisheye_driver:={}".format(
                self._launch_bool(self.start_fisheye_driver)),
            "start_down_camera:={}".format(
                self._launch_bool(self.start_down_camera)),
            "start_down_splitter:={}".format(
                self._launch_bool(self.start_down_splitter)),
            "start_aruco_web:={}".format(
                self._launch_bool(self.start_aruco_web)),
            "start_rectangle_web:={}".format(
                self._launch_bool(self.start_rectangle_web)),
            "aruco_web_port:={}".format(self.aruco_web_port),
            "rectangle_web_port:={}".format(self.rectangle_web_port),
            "aruco_topic:={}".format(self.aruco_topic),
            "aruco_web_detection_topic:={}".format(
                self.aruco_web_detection_topic),
            "aruco_web_pose_topic:={}".format(self.aruco_web_pose_topic),
            "rectangle_target_topic:={}".format(self.rectangle_target_topic),
            "rectangle_detection_topic:={}".format(
                self.rectangle_detection_topic),
            "rectangle_pose_topic:={}".format(self.rectangle_pose_topic),
        ]

    def task1_command(self):
        return [
            "roslaunch", "auv_control", "task3_subtask1_acquire_area.launch",
            "start_arrow_model:=false",
            "arrow_topic:={}".format(self.arrow_topic),
            "motion_goal_topic:={}".format(self.motion_goal_topic),
            "motion_cancel_topic:={}".format(self.motion_cancel_topic),
            "motion_state_topic:={}".format(self.motion_state_topic),
        ]

    def task2_command(self):
        return [
            "roslaunch", "auv_control", "task3_subtask2_get_task.launch",
            "start_aruco_model:=false",
            "aruco_topic:={}".format(self.aruco_topic),
            "motion_goal_topic:={}".format(self.motion_goal_topic),
            "motion_cancel_topic:={}".format(self.motion_cancel_topic),
            "motion_state_topic:={}".format(self.motion_state_topic),
        ]

    @staticmethod
    def parse_task2_result(task2_result):
        marker_match = re.search(
            r"ArUco\s*ID\s*[=：:]\s*(\d+)",
            str(task2_result),
            re.IGNORECASE,
        )
        color_match = re.search(
            r"(?:颜色|color)\s*[=：:]\s*(yellow|green|red)\b",
            str(task2_result),
            re.IGNORECASE,
        )
        marker_id = int(marker_match.group(1)) if marker_match else None
        color = color_match.group(1).lower() if color_match else None
        return marker_id, color

    def prepare_task3_color(self, task2_result):
        marker_id, reported_color = self.parse_task2_result(task2_result)
        mapped_color = self.TASK2_COLOR_BY_MARKER.get(marker_id)
        self.task2_marker_id = marker_id
        self.task2_target_color = mapped_color
        if self.task3_color_source == "manual":
            rospy.logwarn(
                (
                    "%s：[颜色交接-手动] 子任务2结果为ID=%s、颜色=%s；"
                    "联调不覆盖target_color，子任务3使用自身launch中的手动值"
                ),
                NODE_NAME,
                "未知" if marker_id is None else str(marker_id),
                "未知" if reported_color is None else reported_color,
            )
            return True
        if mapped_color is None:
            rospy.logerr(
                "%s：[颜色交接失败] 无法从子任务2成功消息解析有效ArUco ID：%s",
                NODE_NAME,
                task2_result,
            )
            return False
        if reported_color is not None and reported_color != mapped_color:
            rospy.logerr(
                (
                    "%s：[颜色交接失败] ArUco ID=%d应映射为%s，"
                    "但子任务2上报颜色=%s"
                ),
                NODE_NAME,
                marker_id,
                mapped_color,
                reported_color,
            )
            return False
        rospy.loginfo(
            (
                "%s：[颜色交接-自动] ArUco ID=%d -> target_color=%s，"
                "子任务2上报颜色=%s；"
                "启动子任务3时将覆盖其launch默认颜色"
            ),
            NODE_NAME,
            marker_id,
            mapped_color,
            "未提供" if reported_color is None else reported_color,
        )
        return True

    def task3_command(self):
        command = [
            "roslaunch", "auv_control", "task3_subtask3_inspect_and_drop.launch",
            "start_rectangle_model:=false",
            "model_detection_topic:={}".format(self.rectangle_detection_topic),
            "motion_goal_topic:={}".format(self.motion_goal_topic),
            "motion_cancel_topic:={}".format(self.motion_cancel_topic),
            "motion_state_topic:={}".format(self.motion_state_topic),
        ]
        if self.task3_color_source == "task2":
            command.append("target_color:={}".format(self.task2_target_color))
        return command

    def start_managed_process(self, role, label, command):
        rospy.loginfo(
            "%s：[%s] 启动进程：%s", NODE_NAME, label, " ".join(command))
        with self.process_lock:
            existing = self.managed_processes.get(role)
        if existing is not None:
            rospy.logerr("%s：[%s] 同角色进程已经存在", NODE_NAME, label)
            return None
        try:
            process = subprocess.Popen(command, start_new_session=True)
            process_group_id = process.pid if os.name == "posix" else None
        except (OSError, ValueError) as error:
            rospy.logerr("%s：[%s] 进程启动失败：%s", NODE_NAME, label, str(error))
            return None
        entry = {
            "process": process,
            "process_group_id": process_group_id,
            "label": label,
        }
        with self.process_lock:
            self.managed_processes[role] = entry
        rospy.loginfo(
            "%s：[%s] 已启动，PID=%d，进程组=%s",
            NODE_NAME,
            label,
            process.pid,
            str(process_group_id) if process_group_id is not None else "不适用",
        )
        return process

    @staticmethod
    def _managed_entry_alive(entry):
        process = entry["process"]
        process.poll()
        if os.name != "posix":
            return process.poll() is None
        process_group_id = entry["process_group_id"]
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def managed_process_alive(self, role):
        with self.process_lock:
            entry = self.managed_processes.get(role)
        return entry is not None and self._managed_entry_alive(entry)

    def _wait_managed_process_exit(self, entry, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._managed_entry_alive(entry):
                return True
            time.sleep(0.05)
        return not self._managed_entry_alive(entry)

    def stop_managed_process(self, role):
        with self.process_lock:
            entry = self.managed_processes.get(role)
        if entry is None:
            return True

        def forget_entry():
            with self.process_lock:
                if self.managed_processes.get(role) is entry:
                    self.managed_processes.pop(role, None)

        process = entry["process"]
        label = entry["label"]
        process_group_id = entry["process_group_id"]
        if not self._managed_entry_alive(entry):
            forget_entry()
            rospy.loginfo("%s：[%s] 进程组已经退出", NODE_NAME, label)
            return True

        rospy.loginfo(
            "%s：[%s] 发送SIGINT，准备关闭整个进程组", NODE_NAME, label)
        try:
            if os.name == "posix":
                os.killpg(process_group_id, signal.SIGINT)
            else:
                process.terminate()
        except OSError as error:
            rospy.logwarn("%s：[%s] SIGINT失败：%s", NODE_NAME, label, str(error))
        if self._wait_managed_process_exit(entry, self.process_sigint_timeout):
            forget_entry()
            rospy.loginfo("%s：[%s] 进程组已正常退出", NODE_NAME, label)
            return True

        rospy.logwarn("%s：[%s] SIGINT超时，升级为SIGTERM", NODE_NAME, label)
        try:
            if os.name == "posix":
                os.killpg(process_group_id, signal.SIGTERM)
            else:
                process.terminate()
        except OSError as error:
            rospy.logwarn("%s：[%s] SIGTERM失败：%s", NODE_NAME, label, str(error))
        if self._wait_managed_process_exit(entry, self.process_sigterm_timeout):
            forget_entry()
            rospy.loginfo("%s：[%s] 进程组已在SIGTERM后退出", NODE_NAME, label)
            return True

        rospy.logerr("%s：[%s] SIGTERM仍未退出，发送SIGKILL", NODE_NAME, label)
        try:
            if os.name == "posix":
                os.killpg(process_group_id, signal.SIGKILL)
            else:
                process.kill()
        except OSError as error:
            rospy.logerr("%s：[%s] SIGKILL失败：%s", NODE_NAME, label, str(error))
        stopped = self._wait_managed_process_exit(entry, 1.0)
        if stopped:
            forget_entry()
        return stopped

    def start_model_bundle(self):
        with self.model_lock:
            self.aruco_ready = False
            self.rectangle_ready = False
            self.aruco_message_count = 0
            self.rectangle_message_count = 0
            self.aruco_last_message_wall_time = None
            self.rectangle_last_message_wall_time = None
        process = self.start_managed_process(
            self.MODEL_BUNDLE_ROLE,
            "ArUco和方框常驻进程组",
            self.model_bundle_command(),
        )
        return process is not None

    def start_arrow_model_process(self):
        with self.model_lock:
            self.arrow_ready = False
            self.arrow_message_count = 0
            self.arrow_last_message_wall_time = None
        process = self.start_managed_process(
            self.ARROW_MODEL_ROLE,
            "箭头模型进程组",
            self.arrow_model_command(),
        )
        return process is not None

    def start_all_models(self):
        if not self.start_arrow_model_process():
            return False
        if self.start_model_bundle():
            return True
        self.stop_arrow_model_process("ArUco和方框模型启动失败回收")
        return False

    def stop_arrow_model_process(self, label="第二次箭头结束模型清理"):
        rospy.loginfo("%s：[%s] 关闭本联调启动的箭头模型进程组", NODE_NAME, label)
        return self.stop_managed_process(self.ARROW_MODEL_ROLE)

    def stop_model_bundle(self, label="总任务模型清理"):
        rospy.loginfo(
            "%s：[%s] 关闭本联调启动的ArUco和方框模型进程组",
            NODE_NAME,
            label,
        )
        return self.stop_managed_process(self.MODEL_BUNDLE_ROLE)

    def start_child(self, stage_prefix, stage_name, command):
        role = "task_{}".format(stage_prefix)
        process = self.start_managed_process(role, stage_name + "执行launch", command)
        self.child_process = process
        self.child_process_role = role if process is not None else None
        return process is not None

    def stop_child(self, stage_name="当前子任务"):
        role = self.child_process_role
        self.child_process = None
        self.child_process_role = None
        if role is None:
            return True
        rospy.loginfo("%s：[%s阶段结束] 关闭本联调启动的执行launch", NODE_NAME, stage_name)
        return self.stop_managed_process(role)

    def stop_remaining_managed_processes(self):
        """异常路径兜底，关闭仍由本节点登记的所有进程组。"""
        with self.process_lock:
            roles = list(self.managed_processes.keys())
        success = True
        for role in roles:
            success = self.stop_managed_process(role) and success
        return success

    def reset_task3_handoff(self):
        with self.handoff_lock:
            self.task3_handoff_goal_count_received = 0
            self.latest_task3_handoff_goal = None
            self.latest_task3_handoff_caller = ""
            self.task3_handoff_goals = []
            self.task3_handoff_unauthorized_caller = ""

    def handoff_goal_status(self, expected_goal, task3_goal):
        if task3_goal.header.frame_id != self.map_frame:
            return False, "目标坐标系为{}，要求{}".format(
                task3_goal.header.frame_id or "空",
                self.map_frame,
            )
        expected_yaw = self._yaw_from_pose(expected_goal.pose)
        task3_yaw = self._yaw_from_pose(task3_goal.pose)
        if expected_yaw is None or task3_yaw is None:
            return False, "目标四元数无效"
        xy_error = math.hypot(
            task3_goal.pose.position.x - expected_goal.pose.position.x,
            task3_goal.pose.position.y - expected_goal.pose.position.y,
        )
        depth_error = abs(
            task3_goal.pose.position.z - expected_goal.pose.position.z)
        yaw_error = abs(normalize_angle(task3_yaw - expected_yaw))
        detail = "目标差：水平={:.3f}m，深度={:.3f}m，航向={:.1f}deg".format(
            xy_error,
            depth_error,
            math.degrees(yaw_error),
        )
        if xy_error > self.handoff_goal_position_tolerance:
            return False, detail
        if depth_error > self.handoff_goal_depth_tolerance:
            return False, detail
        if yaw_error > self.handoff_goal_yaw_tolerance:
            return False, detail
        return True, detail

    def wait_for_task3_handoff(self, hold_goal):
        """保持过渡定点，直到子任务3连续发出合法目标。"""
        started_at = time.monotonic()
        processed_count = 0
        valid_count = 0
        while not rospy.is_shutdown():
            self.publish_goal(hold_goal)
            if not self.managed_process_alive(self.MODEL_BUNDLE_ROLE):
                rospy.logerr(
                    "%s：[控制权交接失败] ArUco和方框模型进程组已经退出",
                    NODE_NAME,
                )
                return False, "子任务3接管前ArUco和方框模型进程组已经退出"

            if self.stage_event.is_set():
                with self.stage_lock:
                    result = self.stage_result
                return False, "接管前子任务3已经结束：{}".format(result)
            if self.child_process is None or self.child_process.poll() is not None:
                return False, "接管前子任务3进程已经退出"

            with self.handoff_lock:
                new_goals = [
                    item for item in self.task3_handoff_goals
                    if item[0] > processed_count
                ]
                unauthorized_caller = self.task3_handoff_unauthorized_caller
            if unauthorized_caller:
                return False, "接管期间检测到未授权目标发布者：{}".format(
                    unauthorized_caller)
            for sequence, task3_goal, caller_id in new_goals:
                processed_count = max(processed_count, sequence)
                valid, detail = self.handoff_goal_status(hold_goal, task3_goal)
                if not valid:
                    rospy.logerr(
                        "%s：[控制权交接失败] %s发布的第%d个目标不安全：%s",
                        NODE_NAME,
                        caller_id,
                        sequence,
                        detail,
                    )
                    return False, "子任务3首批目标与交接定点不一致：{}".format(detail)
                valid_count += 1
                rospy.loginfo(
                    "%s：[控制权交接] 收到子任务3合法目标%d/%d，发布者=%s，%s",
                    NODE_NAME,
                    valid_count,
                    self.task3_handoff_goal_count,
                    caller_id,
                    detail,
                )
                if valid_count >= self.task3_handoff_goal_count:
                    rospy.loginfo(
                        (
                            "%s：[控制权交接完成] 子任务3已连续发布%d个合法目标；"
                            "联调节点停止续发固定点，后续运动由子任务3控制"
                        ),
                        NODE_NAME,
                        valid_count,
                    )
                    return True, "子任务3目标发布握手完成"

            elapsed = time.monotonic() - started_at
            if elapsed >= self.task3_handoff_timeout:
                return False, (
                    "等待子任务3发布控制目标超过{:.1f}s；请确认"
                    "task3_subtask3_inspect_and_drop.launch中的operation_mode=auto"
                ).format(self.task3_handoff_timeout)
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：[控制权交接] 联调节点继续保持第二次箭头结束定点，"
                    "等待子任务3目标%d/%d，剩余%.1fs"
                ),
                NODE_NAME,
                valid_count,
                self.task3_handoff_goal_count,
                max(0.0, self.task3_handoff_timeout - elapsed),
            )
            self.rate.sleep()
        return False, "ROS关闭，控制权交接中止"

    def run_child_stage(
            self, stage_prefix, stage_name, command, timeout,
            hold_goal=None, handoff_goal=None, required_model_roles=None):
        if required_model_roles is None:
            required_model_roles = (
                (self.MODEL_BUNDLE_ROLE, "ArUco和方框模型"),
            )
        with self.stage_lock:
            self.current_stage = stage_prefix
            self.stage_result = None
            self.stage_event.clear()
        if handoff_goal is not None:
            self.reset_task3_handoff()

        if not self.start_child(stage_prefix, stage_name, command):
            with self.stage_lock:
                self.current_stage = None
            return False, "无法启动{}".format(stage_name)

        if handoff_goal is not None:
            handoff_success, handoff_detail = self.wait_for_task3_handoff(
                handoff_goal)
            if not handoff_success:
                cleanup_success = self.stop_child(stage_name)
                with self.stage_lock:
                    self.current_stage = None
                    self.stage_result = None
                    self.stage_event.clear()
                if not cleanup_success:
                    handoff_detail += "；{}执行launch清理未通过".format(stage_name)
                return False, handoff_detail

        started_at = time.monotonic()
        process_exit_at = None
        hold_unhealthy_at = None
        stage_failure_detail = None
        result = None
        while not rospy.is_shutdown():
            failed_model = next(
                (
                    label for role, label in required_model_roles
                    if not self.managed_process_alive(role)
                ),
                None,
            )
            if failed_model is not None:
                stage_failure_detail = "{}期间{}进程组异常退出".format(
                    stage_name,
                    failed_model,
                )
                rospy.logerr("%s：%s", NODE_NAME, stage_failure_detail)
                break
            if hold_goal is not None:
                self.publish_goal(hold_goal)
                hold_healthy, hold_detail = self.goal_arrival_status(
                    hold_goal, started_at)
                if hold_healthy:
                    hold_unhealthy_at = None
                else:
                    if hold_unhealthy_at is None:
                        hold_unhealthy_at = time.monotonic()
                    unhealthy_elapsed = time.monotonic() - hold_unhealthy_at
                    rospy.logwarn_throttle(
                        1.0,
                        "%s：%s定点健康检查异常%.1f/%.1fs：%s；%s",
                        NODE_NAME,
                        stage_name,
                        unhealthy_elapsed,
                        self.control_health_grace_timeout,
                        hold_detail,
                        self.motion_state_debug_text(),
                    )
                    if unhealthy_elapsed >= self.control_health_grace_timeout:
                        stage_failure_detail = (
                            "{}期间固定点控制连续异常{:.1f}s：{}".format(
                                stage_name,
                                unhealthy_elapsed,
                                hold_detail,
                            )
                        )
                        break

            if self.stage_event.is_set():
                with self.stage_lock:
                    result = self.stage_result
                break

            if self.child_process is not None and self.child_process.poll() is not None:
                if process_exit_at is None:
                    process_exit_at = time.monotonic()
                elif time.monotonic() - process_exit_at >= self.stage_shutdown_grace:
                    break

            elapsed = time.monotonic() - started_at
            if elapsed >= timeout:
                rospy.logerr("%s：%s运行超过%.1fs", NODE_NAME, stage_name, timeout)
                break
            rospy.loginfo_throttle(
                2.0,
                "%s：%s运行中，已用时%.1fs，等待完成消息",
                NODE_NAME,
                stage_name,
                elapsed,
            )
            self.rate.sleep()

        cleanup_success = self.stop_child(stage_name)
        with self.stage_lock:
            self.current_stage = None
            self.stage_result = None
            self.stage_event.clear()

        if not cleanup_success:
            return False, "{}结束后执行launch未完全关闭".format(stage_name)
        if result is None:
            if stage_failure_detail is not None:
                return False, stage_failure_detail
            return False, "{}未返回完成消息或已超时".format(stage_name)
        success = result.startswith("{} finished".format(stage_prefix))
        return success, result

    def safe_hover_status(self, cancel_started_at):
        with self.motion_state_lock:
            state = self.latest_motion_state
            received_at = self.latest_motion_state_wall_time
        if state is None or received_at is None:
            return False, "尚未收到motion状态"
        age = time.monotonic() - received_at
        if age > self.motion_state_timeout:
            return False, "motion状态超时{:.2f}s".format(age)
        if received_at < cancel_started_at:
            return False, "等待取消指令后的新motion状态"
        if not state.startup_complete:
            return False, "motion_supervisor尚未完成启动定点"
        if state.state != MotionState.HOVER:
            return False, "当前状态={}，等待HOVER".format(
                self.MOTION_STATE_NAMES.get(state.state, state.state))
        if abs(state.base_position_error) > self.arrival_position_tolerance:
            return False, "位置误差{:.3f}m".format(state.base_position_error)
        if abs(state.yaw_error) > self.arrival_yaw_tolerance:
            return False, "航向误差{:.1f}deg".format(math.degrees(state.yaw_error))
        if abs(state.horizontal_speed) > self.arrival_max_horizontal_speed:
            return False, "水平速度{:.3f}m/s".format(state.horizontal_speed)
        if abs(state.yaw_rate) > self.arrival_max_yaw_rate:
            return False, "航向角速度{:.3f}rad/s".format(state.yaw_rate)
        return True, "已HOVER且位置、航向和速度均稳定"

    def request_safe_stop(self, reason):
        """取消当前目标，确认机器人停稳后再结束联调。"""
        cancel_started_at = time.monotonic()
        stable_started_at = None
        self.motion_cancel_pub.publish(Empty())
        rospy.logwarn(
            "%s：[安全刹停] 已发布%s，原因：%s",
            NODE_NAME,
            self.motion_cancel_topic,
            reason,
        )
        while not rospy.is_shutdown():
            now = time.monotonic()
            safe, detail = self.safe_hover_status(cancel_started_at)
            if safe:
                if stable_started_at is None:
                    stable_started_at = now
                    rospy.loginfo("%s：[安全刹停] 首次达到HOVER稳定阈值", NODE_NAME)
                stable_elapsed = now - stable_started_at
                rospy.loginfo_throttle(
                    1.0,
                    "%s：[安全刹停] 稳定确认%.1f/%.1fs，%s；%s",
                    NODE_NAME,
                    min(stable_elapsed, self.safe_stop_stable_seconds),
                    self.safe_stop_stable_seconds,
                    detail,
                    self.motion_state_debug_text(),
                )
                if stable_elapsed >= self.safe_stop_stable_seconds:
                    return True
            else:
                if stable_started_at is not None:
                    rospy.logwarn("%s：[安全刹停] 稳定计时中断：%s", NODE_NAME, detail)
                stable_started_at = None

            elapsed = now - cancel_started_at
            if elapsed >= self.safe_stop_timeout:
                rospy.logerr(
                    "%s：[安全刹停] 超过%.1fs仍未确认HOVER：%s；%s",
                    NODE_NAME,
                    self.safe_stop_timeout,
                    detail,
                    self.motion_state_debug_text(),
                )
                return False
            rospy.loginfo_throttle(
                1.0,
                "%s：[安全刹停] 等待停稳，剩余%.1fs：%s；%s",
                NODE_NAME,
                max(0.0, self.safe_stop_timeout - elapsed),
                detail,
                self.motion_state_debug_text(),
            )
            self.rate.sleep()
        return False

    def finish_after_safe_stop(self, requested_success, detail):
        safe = self.request_safe_stop(detail)
        child_cleanup = self.stop_child()
        arrow_cleanup = self.stop_arrow_model_process("总任务结束箭头模型清理")
        model_cleanup = self.stop_model_bundle("总任务结束模型清理")
        process_cleanup = self.stop_remaining_managed_processes()
        cleanup = (
            child_cleanup
            and arrow_cleanup
            and model_cleanup
            and process_cleanup
        )
        if not safe or not cleanup:
            failure_parts = [detail]
            if not safe:
                failure_parts.append("最终安全悬停未确认")
            if not cleanup:
                failure_parts.append("本联调启动的执行launch或模型进程未完全关闭")
            self.finish(False, "；".join(failure_parts))
            return False
        self.finish(
            requested_success,
            (
                "{}；机器人已安全悬停；本联调启动的各阶段执行launch和"
                "三个识别模型均已关闭"
            ).format(detail),
        )
        return requested_success

    def finish(self, success, detail):
        if self.final_result_published:
            rospy.logwarn("%s：忽略重复的联调最终结果：%s", NODE_NAME, detail)
            return
        self.final_result_published = True
        state = "finished" if success else "failed"
        message = "{} {}: {}".format(NODE_NAME, state, detail)
        self.sequence_finished_pub.publish(String(data=message))
        rospy.loginfo(
            "%s：联调流程%s：%s",
            NODE_NAME,
            "完成" if success else "失败",
            detail,
        )

    def run(self):
        rospy.sleep(0.5)
        rospy.loginfo(
            "%s：[启动] 默认环境仅有公共控制和传感器，直接启动三个模型；"
            "不扫描或关闭其他任务、模型、Web节点和端口",
            NODE_NAME,
        )
        if not self.start_all_models():
            self.stop_remaining_managed_processes()
            self.finish(False, "三个识别模型进程组启动失败")
            return

        hold_goal = self.capture_current_goal()
        if hold_goal is None:
            arrow_cleanup = self.stop_arrow_model_process("启动失败箭头模型清理")
            model_cleanup = self.stop_model_bundle("启动失败常驻模型清理")
            process_cleanup = self.stop_remaining_managed_processes()
            cleanup = arrow_cleanup and model_cleanup and process_cleanup
            detail = "未能获取启动时map到base_link的固定姿态"
            if not cleanup:
                detail += "；本联调启动的识别模型未完全关闭"
            self.finish(False, detail)
            return

        if not self.wait_for_goal(
                hold_goal,
                self.initial_hold_timeout,
                self.initial_hold_stable_seconds,
                "启动定点"):
            self.finish_after_safe_stop(False, "motion_supervisor未能完成启动定点")
            return

        if not self.wait_for_models(hold_goal):
            self.finish_after_safe_stop(False, "三个识别模型未能全部就绪")
            return

        rospy.loginfo(
            (
                "%s：[阶段1/4 第一次箭头] 三个模型均已持续输出；启动箭头任务节点，"
                "参数来自task3_subtask1_acquire_area.launch，协调节点暂停发布运动目标"
            ),
            NODE_NAME,
        )
        task1_success, task1_detail = self.run_child_stage(
            self.TASK1_PREFIX,
            "子任务1",
            self.task1_command(),
            self.task1_stage_timeout,
            required_model_roles=(
                (self.ARROW_MODEL_ROLE, "箭头模型"),
                (self.MODEL_BUNDLE_ROLE, "ArUco和方框模型"),
            ),
        )
        if not task1_success:
            self.finish_after_safe_stop(
                False,
                "第一次箭头任务失败，不执行后续阶段：{}".format(task1_detail),
            )
            return
        rospy.loginfo(
            "%s：[阶段1/4 第一次箭头] 成功：%s",
            NODE_NAME,
            task1_detail,
        )

        rospy.loginfo(
            "%s：[阶段衔接1->2] 重新锁存子任务1结束位置，禁止使用总任务启动点",
            NODE_NAME,
        )
        hold_goal = self.capture_current_goal()
        if hold_goal is None:
            self.finish_after_safe_stop(
                False,
                "子任务1成功，但未能获取其结束位置，禁止启动子任务2",
            )
            return
        if not self.wait_for_goal(
                hold_goal,
                self.post_task1_hold_timeout,
                self.post_task1_stable_seconds,
                "子任务1结束位置复稳"):
            self.finish_after_safe_stop(
                False,
                "子任务1成功，但机器人未能稳定锁定子任务1结束位置",
            )
            return
        if not self.wait_for_aruco_model_fresh(
                hold_goal, "启动子任务2前ArUco模型复查"):
            self.finish_after_safe_stop(
                False,
                "子任务1成功，但ArUco模型在子任务2启动前没有持续输出",
            )
            return

        rospy.loginfo(
            (
                "%s：[阶段2/4 子任务2] 启动后自行锁存第一次箭头结束位置，"
                "依次完成ArUco识别、亮灯和原地转向；左右方向和角度来自"
                "task3_subtask2_get_task.launch。协调节点暂停发布运动目标"
            ),
            NODE_NAME,
        )
        task2_success, task2_detail = self.run_child_stage(
            self.TASK2_PREFIX,
            "子任务2",
            self.task2_command(),
            self.task2_stage_timeout,
            required_model_roles=(
                (self.ARROW_MODEL_ROLE, "箭头模型"),
                (self.MODEL_BUNDLE_ROLE, "ArUco和方框模型"),
            ),
        )
        if not task2_success:
            self.finish_after_safe_stop(
                False,
                "子任务2识别、亮灯或转向失败，不执行第二次箭头和子任务3：{}".format(
                    task2_detail),
            )
            return

        rospy.loginfo(
            "%s：[阶段2/4 子任务2] 成功：%s",
            NODE_NAME,
            task2_detail,
        )
        if not self.prepare_task3_color(task2_detail):
            self.finish_after_safe_stop(
                False,
                "子任务2虽然成功，但颜色数据无法交给子任务3",
            )
            return
        rospy.loginfo(
            "%s：[阶段衔接2->3] 子任务2已完成内部转向，重新锁存转向结束位置",
            NODE_NAME,
        )
        hold_goal = self.capture_current_goal()
        if hold_goal is None:
            self.finish_after_safe_stop(
                False,
                "子任务2成功，但未能获取转向结束位置，禁止启动第二次箭头",
            )
            return
        if not self.wait_for_goal(
                hold_goal,
                self.post_task2_hold_timeout,
                self.post_task2_stable_seconds,
                "子任务2转向结束位置复稳"):
            self.finish_after_safe_stop(
                False,
                "子任务2成功，但机器人未能稳定锁定转向结束位置",
            )
            return
        if not self.wait_for_arrow_model_fresh(
                hold_goal, "启动第二次箭头前模型复查"):
            self.finish_after_safe_stop(
                False,
                "子任务2转向成功，但箭头模型在第二次箭头任务启动前没有持续输出",
            )
            return
        rospy.loginfo(
            (
                "%s：[阶段3/4 第二次箭头] 转向结束位置稳定且箭头模型输出正常；"
                "再次完整启动子任务1，由其自行前进、左右搜索、粗对准和细对准。"
                "协调节点暂停发布运动目标"
            ),
            NODE_NAME,
        )
        second_arrow_success, second_arrow_detail = self.run_child_stage(
            self.TASK1_PREFIX,
            "第二次箭头",
            self.task1_command(),
            self.second_arrow_stage_timeout,
            required_model_roles=(
                (self.ARROW_MODEL_ROLE, "箭头模型"),
                (self.MODEL_BUNDLE_ROLE, "ArUco和方框模型"),
            ),
        )
        if not second_arrow_success:
            self.finish_after_safe_stop(
                False,
                "第二次箭头任务失败，不执行子任务3：{}".format(
                    second_arrow_detail),
            )
            return
        rospy.loginfo(
            "%s：[阶段3/4 第二次箭头] 成功：%s",
            NODE_NAME,
            second_arrow_detail,
        )

        rospy.loginfo(
            "%s：[阶段衔接3->4] 重新锁存第二次箭头结束位置，作为子任务3接管定点",
            NODE_NAME,
        )
        task3_entry_goal = self.capture_current_goal()
        if task3_entry_goal is None:
            self.finish_after_safe_stop(
                False,
                "第二次箭头任务成功，但未能获取其结束位置，禁止启动子任务3",
            )
            return
        if not self.wait_for_goal(
                task3_entry_goal,
                self.post_second_arrow_hold_timeout,
                self.post_second_arrow_stable_seconds,
                "第二次箭头结束位置复稳"):
            self.finish_after_safe_stop(
                False,
                "第二次箭头任务成功，但机器人未能稳定锁定其结束位置",
            )
            return
        if not self.stop_arrow_model_process("第二次箭头完成模型清理"):
            self.finish_after_safe_stop(
                False,
                "第二次箭头任务成功，但箭头模型未能完全关闭，禁止启动子任务3",
            )
            return
        if not self.wait_for_rectangle_model_fresh(
                task3_entry_goal, "启动子任务3前方框模型复查"):
            self.finish_after_safe_stop(
                False,
                "第二次箭头任务成功，但方框模型在子任务3接管前没有持续输出",
            )
            return
        rospy.loginfo(
            (
                "%s：[阶段4/4 子任务3] 第二次箭头结束位置已稳定且方框模型输出正常；"
                "先启动子任务3并保持该定点，握手成功后再交出控制权。参数来自"
                "task3_subtask3_inspect_and_drop.launch"
            ),
            NODE_NAME,
        )
        task3_success, task3_detail = self.run_child_stage(
            self.TASK3_PREFIX,
            "子任务3",
            self.task3_command(),
            self.task3_stage_timeout,
            handoff_goal=task3_entry_goal,
        )
        if not task3_success:
            self.finish_after_safe_stop(
                False,
                "子任务3失败：{}".format(task3_detail),
            )
            return

        self.finish_after_safe_stop(
            True,
            "第一次箭头、子任务2识别亮灯及转向、第二次箭头和子任务3均执行成功",
        )

    def on_shutdown(self):
        if self.shutdown_started:
            return
        self.shutdown_started = True
        self.stop_child()
        self.stop_remaining_managed_processes()
        if (
                not self.final_result_published
                and hasattr(self, "motion_cancel_pub")):
            self.motion_cancel_pub.publish(Empty())


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=False)
    try:
        Task3Subtask123Sequence().run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    except Exception as error:
        rospy.logfatal("%s：未处理异常：%s", NODE_NAME, str(error))
        raise
