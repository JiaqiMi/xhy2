#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务3子任务2、3联调协调节点。"""

import copy
import json
import math
import os
import re
import signal
import socket
import subprocess
import threading
import time

import rosgraph
import rosnode
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from auv_control.msg import MotionState, TargetDetection


NODE_NAME = "test_task3_2_3_sequence"


def normalize_angle(angle):
    """把角度限制到[-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Task3Subtask23Sequence:
    """同时保持两个模型常驻，并顺序调度、清理两个执行子任务。"""

    TASK2_PREFIX = "test_task3_2_get_task"
    TASK3_PREFIX = "test_task3_3_inspect_and_drop"
    MODEL_BUNDLE_ROLE = "model_bundle"
    TASK_NODE_TOKENS = (
        "task1",
        "task2",
        "task3",
        "task4",
        "task5",
        "aruco",
        "arrow",
        "yolo",
        "stereo_depth",
        "object_detection_locate",
        "holes_location",
        "balls_detector",
        "disparity_to_point_node",
        "line_location",
        "pipeline_detector",
        "vision_web_dashboard",
        "world_origin_reset",
        "motion_goal_test",
        "motion_axis_auto_test",
        "motion_yaw_continuous_test",
        "test_point_control",
        "test_pose_control",
        "test_rotation",
        "keyboard_control_node",
        "search_tester",
        "target_detection_test",
    )
    MODEL_NODE_TOKENS = (
        "aruco",
        "arrow",
        "yolo",
        "stereo_depth",
        "object_detection_locate",
        "holes_location",
        "balls_detector",
        "disparity_to_point_node",
        "line_location",
        "pipeline_detector",
        "vision_web_dashboard",
    )
    TASK_LAUNCH_COMMAND_TOKENS = (
        "roslaunch auv_control task",
        "roslaunch auv_control motion_axis_auto_test.launch",
        "roslaunch auv_control motion_x_auto_test.launch",
        "roslaunch auv_control motion_y_auto_test.launch",
        "roslaunch auv_control motion_yaw_auto_test.launch",
        "roslaunch auv_control motion_yaw_continuous_test.launch",
        "roslaunch auv_control reset_world_origin.launch",
        "roslaunch stereo_depth ",
        "roslaunch yolo_bridge ",
        "roslaunch vision_web ",
    )
    PROTECTED_PUBLIC_NODES = (
        "/sensor_status_node",
        "/sensor_actuator_node",
        "/debug_driver_v2",
        "/map_initer",
        "/auv_tf_handler",
        "/state_web",
        "/usb_cam",
        "/split_stereo_image",
        "/rtsp_fisheye_driver",
        "/vo_nav_fusion",
        "/motion_supervisor",
    )
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

        self.aruco_topic = str(rospy.get_param(
            "~aruco_topic", "/obj/target_message"))
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
        self.start_aruco_web = bool(rospy.get_param(
            "~start_aruco_web", True))
        self.start_rectangle_web = bool(rospy.get_param(
            "~start_rectangle_web", True))
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
        self.transition_turn_direction = str(rospy.get_param(
            "~transition_turn_direction", "right")).strip().lower()
        self.transition_turn_angle_deg = float(rospy.get_param(
            "~transition_turn_angle_deg", 90.0))
        self.transition_timeout = float(rospy.get_param(
            "~transition_timeout", 90.0))
        self.transition_stable_seconds = float(rospy.get_param(
            "~transition_stable_seconds", 1.0))
        self.transition_hold_seconds = float(rospy.get_param(
            "~transition_hold_seconds", 1.0))
        self.post_task2_hold_timeout = float(rospy.get_param(
            "~post_task2_hold_timeout", 30.0))
        self.post_task2_stable_seconds = float(rospy.get_param(
            "~post_task2_stable_seconds", 1.0))

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

        self.task2_pose_cmd_topic = str(rospy.get_param(
            "~task2_pose_cmd_topic",
            "/task3_sequence/task2_isolated_pose_cmd"))
        self.task2_stage_timeout = float(rospy.get_param(
            "~task2_stage_timeout", 100.0))
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
        self.cleanup_timeout = float(rospy.get_param(
            "~cleanup_timeout", 10.0))
        self.control_backbone_timeout = float(rospy.get_param(
            "~control_backbone_timeout", 10.0))
        self.cleanup_poll_interval = float(rospy.get_param(
            "~cleanup_poll_interval", 0.2))
        self.cleanup_stable_seconds = float(rospy.get_param(
            "~cleanup_stable_seconds", 1.0))
        self.process_sigint_timeout = float(rospy.get_param(
            "~process_sigint_timeout", 5.0))
        self.process_sigterm_timeout = float(rospy.get_param(
            "~process_sigterm_timeout", 2.0))

        self._validate_parameters()

        self.tf_listener = tf.TransformListener()
        self.motion_state_lock = threading.Lock()
        self.latest_motion_state = None
        self.latest_motion_state_wall_time = None

        self.aruco_ready = False
        self.rectangle_ready = False
        self.aruco_message_count = 0
        self.rectangle_message_count = 0
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
            "transition_timeout": self.transition_timeout,
            "post_task2_hold_timeout": self.post_task2_hold_timeout,
            "task2_stage_timeout": self.task2_stage_timeout,
            "task3_stage_timeout": self.task3_stage_timeout,
            "task3_handoff_timeout": self.task3_handoff_timeout,
            "safe_stop_timeout": self.safe_stop_timeout,
            "control_health_grace_timeout": self.control_health_grace_timeout,
            "cleanup_timeout": self.cleanup_timeout,
            "control_backbone_timeout": self.control_backbone_timeout,
            "cleanup_poll_interval": self.cleanup_poll_interval,
            "process_sigint_timeout": self.process_sigint_timeout,
            "process_sigterm_timeout": self.process_sigterm_timeout,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError("参数{}必须大于0".format(name))
        if self.transition_turn_direction not in ("left", "right"):
            raise ValueError("transition_turn_direction必须为left或right")
        if self.transition_turn_angle_deg <= 0.0:
            raise ValueError("transition_turn_angle_deg必须大于0")
        if self.transition_turn_angle_deg >= 180.0:
            raise ValueError(
                "transition_turn_angle_deg必须小于180；180度时绝对航向方向存在二义性")
        if self.task3_handoff_goal_count <= 0:
            raise ValueError("task3_handoff_goal_count必须大于0")
        if self.model_handoff_required_frames <= 0:
            raise ValueError("model_handoff_required_frames必须大于0")
        if self.task3_color_source not in ("task2", "manual"):
            raise ValueError("task3_color_source必须为task2或manual")
        if not self.start_aruco_model or not self.start_rectangle_model:
            raise ValueError(
                "联调模式要求start_aruco_model和start_rectangle_model都为true，"
                "以便协调节点统一管理模型生命周期")
        for name, port in (
                ("aruco_web_port", self.aruco_web_port),
                ("rectangle_web_port", self.rectangle_web_port)):
            if port <= 0 or port > 65535:
                raise ValueError("参数{}必须在1到65535之间".format(name))
        if (
                self.start_aruco_web
                and self.start_rectangle_web
                and self.aruco_web_port == self.rectangle_web_port):
            raise ValueError("两个Web页面不能使用相同端口")
        isolated_topics = {
            "ArUco目标结果": self.aruco_topic,
            "ArUco网页检测": self.aruco_web_detection_topic,
            "ArUco网页位姿": self.aruco_web_pose_topic,
            "方框目标结果": self.rectangle_target_topic,
            "方框网页检测": self.rectangle_detection_topic,
            "方框网页位姿": self.rectangle_pose_topic,
        }
        expected_prefixes = {
            "ArUco目标结果": "/obj/",
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
            "transition_stable_seconds": self.transition_stable_seconds,
            "transition_hold_seconds": self.transition_hold_seconds,
            "post_task2_stable_seconds": self.post_task2_stable_seconds,
            "handoff_goal_position_tolerance": self.handoff_goal_position_tolerance,
            "handoff_goal_depth_tolerance": self.handoff_goal_depth_tolerance,
            "handoff_goal_yaw_tolerance": self.handoff_goal_yaw_tolerance,
            "safe_stop_stable_seconds": self.safe_stop_stable_seconds,
            "cleanup_stable_seconds": self.cleanup_stable_seconds,
        }
        for name, value in non_negative_values.items():
            if value < 0.0:
                raise ValueError("参数{}不能小于0".format(name))

    def signed_turn_angle_deg(self):
        """NED航向正方向为右转，左转使用负角度。"""
        sign = 1.0 if self.transition_turn_direction == "right" else -1.0
        return sign * self.transition_turn_angle_deg

    def turn_direction_text(self):
        return "右转" if self.transition_turn_direction == "right" else "左转"

    def log_configuration(self):
        rospy.loginfo(
            (
                "%s：[联调参数] 转向=%s%.1f度，旋转超时=%.1fs，"
                "稳定确认=%.1fs，完成后保持=%.1fs"
            ),
            NODE_NAME,
            self.turn_direction_text(),
            self.transition_turn_angle_deg,
            self.transition_timeout,
            self.transition_stable_seconds,
            self.transition_hold_seconds,
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
                "%s：[联调超时] 模型就绪=%.1fs，子任务2总时限=%.1fs，"
                "子任务3总时限=%.1fs"
            ),
            NODE_NAME,
            self.model_ready_timeout,
            self.task2_stage_timeout,
            self.task3_stage_timeout,
        )
        rospy.loginfo(
            (
                "%s：[交接参数] 子任务2后复稳=%.1fs；子任务3接管超时=%.1fs，"
                "需连续收到%d个目标；模型交接需%d个新帧；最终安全悬停=%.1fs"
            ),
            NODE_NAME,
            self.post_task2_stable_seconds,
            self.task3_handoff_timeout,
            self.task3_handoff_goal_count,
            self.model_handoff_required_frames,
            self.safe_stop_stable_seconds,
        )
        rospy.loginfo(
            "%s：[话题隔离] 子任务2只读取%s，子任务3只读取%s",
            NODE_NAME,
            self.aruco_topic,
            self.rectangle_detection_topic,
        )
        rospy.loginfo(
            (
                "%s：[模型输出隔离] ArUco={目标:%s,检测:%s,位姿:%s}；"
                "方框={目标:%s,检测:%s,位姿:%s}"
            ),
            NODE_NAME,
            self.aruco_topic,
            self.aruco_web_detection_topic,
            self.aruco_web_pose_topic,
            self.rectangle_target_topic,
            self.rectangle_detection_topic,
            self.rectangle_pose_topic,
        )
        rospy.loginfo(
            (
                "%s：[进程生命周期] 启动时无条件清理任务、模型和模型Web节点；"
                "begin公共节点与motion_supervisor受保护；两个模型同时启动并全程常驻；"
                "阶段结束只关闭当前执行launch；"
                "总任务结束统一关闭模型"
            ),
            NODE_NAME,
        )
        rospy.loginfo(
            (
                "%s：[清场参数] 节点/控制清理超时=%.1fs，公共控制器等待=%.1fs，"
                "轮询周期=%.2fs，连续无残留确认=%.1fs，SIGINT等待=%.1fs，"
                "SIGTERM等待=%.1fs"
            ),
            NODE_NAME,
            self.cleanup_timeout,
            self.control_backbone_timeout,
            self.cleanup_poll_interval,
            self.cleanup_stable_seconds,
            self.process_sigint_timeout,
            self.process_sigterm_timeout,
        )
        rospy.loginfo(
            "%s：[模型进程] ArUco Web=%s:%d，方框 Web=%s:%d",
            NODE_NAME,
            "开启" if self.start_aruco_web else "关闭",
            self.aruco_web_port,
            "开启" if self.start_rectangle_web else "关闭",
            self.rectangle_web_port,
        )
        rospy.loginfo(
            (
                "%s：[参数归属] 子任务2参数读取task3_subtask2_get_task.launch；"
                "子任务3参数读取task3_subtask3_inspect_and_drop.launch；"
                "本节点只管理联调衔接"
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

    def verify_model_topic_graph(self):
        """确认隔离话题的实际发布节点和消息类型与设计一致。"""
        expected = (
            (
                "ArUco目标结果",
                self.aruco_topic,
                "/task3_sequence/aruco_pipeline/fisheye_aruco_node",
                "auv_control/TargetDetection",
            ),
            (
                "ArUco网页检测",
                self.aruco_web_detection_topic,
                "/task3_sequence/aruco_pipeline/fisheye_aruco_node",
                "std_msgs/String",
            ),
            (
                "ArUco网页位姿",
                self.aruco_web_pose_topic,
                "/task3_sequence/aruco_pipeline/fisheye_aruco_node",
                "std_msgs/String",
            ),
            (
                "方框目标结果",
                self.rectangle_target_topic,
                "/stereo_depth_rectangle",
                "auv_control/TargetDetection",
            ),
            (
                "方框网页检测",
                self.rectangle_detection_topic,
                "/yolo_rectangle_detector",
                "std_msgs/String",
            ),
            (
                "方框网页位姿",
                self.rectangle_pose_topic,
                "/stereo_depth_rectangle",
                "std_msgs/String",
            ),
        )
        try:
            master = rosgraph.Master(rospy.get_name())
            publishers, _, _ = master.getSystemState()
            topic_types = dict(master.getTopicTypes())
        except Exception as error:
            return False, "无法读取ROS话题图：{}".format(error)

        publisher_map = {
            topic: set(nodes) for topic, nodes in publishers
        }
        failures = []
        for label, topic, expected_node, expected_type in expected:
            actual_publishers = publisher_map.get(topic, set())
            actual_type = topic_types.get(topic)
            if actual_publishers != {expected_node}:
                failures.append(
                    "{} {}发布者={}，要求={}".format(
                        label,
                        topic,
                        ",".join(sorted(actual_publishers)) or "无",
                        expected_node,
                    )
                )
            if actual_type != expected_type:
                failures.append(
                    "{} {}类型={}，要求={}".format(
                        label,
                        topic,
                        actual_type or "未知",
                        expected_type,
                    )
                )
        if failures:
            return False, "；".join(failures)

        expected_web_nodes = []
        if self.start_aruco_web:
            expected_web_nodes.append(
                "/task3_sequence/aruco_pipeline/vision_web_dashboard")
        if self.start_rectangle_web:
            expected_web_nodes.append("/vision_web_dashboard")
        try:
            current_nodes = set(rosnode.get_node_names())
        except Exception as error:
            return False, "无法核对Web节点：{}".format(error)
        missing_web_nodes = sorted(set(expected_web_nodes) - current_nodes)
        if missing_web_nodes:
            return False, "模型Web节点缺失：{}".format(", ".join(missing_web_nodes))

        web_ports_ok, web_ports_detail = self.verify_web_ports(
            expected_open=True)
        if not web_ports_ok:
            return False, web_ports_detail

        rospy.loginfo(
            (
                "%s：[模型隔离检查] 通过：ArUco由鱼眼节点独占%s，"
                "方框YOLO独占%s，六个输出话题的发布者和类型均正确"
            ),
            NODE_NAME,
            self.aruco_topic,
            self.rectangle_detection_topic,
        )
        return True, "双模型话题图隔离正确"

    def configured_web_ports(self):
        ports = []
        if self.start_aruco_web:
            ports.append(("ArUco Web", self.aruco_web_port))
        if self.start_rectangle_web:
            ports.append(("方框 Web", self.rectangle_web_port))
        return ports

    @staticmethod
    def _local_port_accepting(port):
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.3)
        try:
            return probe.connect_ex(("127.0.0.1", int(port))) == 0
        finally:
            probe.close()

    def verify_web_ports(self, expected_open):
        failures = []
        for label, port in self.configured_web_ports():
            accepting = self._local_port_accepting(port)
            if accepting != expected_open:
                failures.append(
                    "{}端口{}{}".format(
                        label,
                        port,
                        "未监听" if expected_open else "已被占用",
                    )
                )
        if failures:
            return False, "；".join(failures)
        return True, (
            "Web端口均已监听" if expected_open else "Web端口启动前均空闲")

    def wait_for_models(self, hold_goal):
        started_at = time.monotonic()
        both_ready_at = None
        ready_start_counts = None
        while not rospy.is_shutdown():
            self.publish_goal(hold_goal)
            if not self.managed_process_alive(self.MODEL_BUNDLE_ROLE):
                rospy.logerr("%s：双模型常驻进程组已提前退出", NODE_NAME)
                return False
            with self.model_lock:
                aruco_ready = self.aruco_ready
                rectangle_ready = self.rectangle_ready
                aruco_count = self.aruco_message_count
                rectangle_count = self.rectangle_message_count
                aruco_last = self.aruco_last_message_wall_time
                rectangle_last = self.rectangle_last_message_wall_time

            now = time.monotonic()
            aruco_age = None if aruco_last is None else now - aruco_last
            rectangle_age = (
                None if rectangle_last is None else now - rectangle_last)
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
            if aruco_fresh and rectangle_fresh:
                if both_ready_at is None:
                    both_ready_at = now
                    ready_start_counts = (aruco_count, rectangle_count)
                    rospy.loginfo(
                        (
                            "%s：两个识别模型输出均新鲜，再等待%.1fs且"
                            "各接收%d个新帧确认连续输出"
                        ),
                        NODE_NAME,
                        self.model_settle_seconds,
                        self.model_handoff_required_frames,
                    )
                aruco_new_frames = aruco_count - ready_start_counts[0]
                rectangle_new_frames = rectangle_count - ready_start_counts[1]
                if (
                        now - both_ready_at >= self.model_settle_seconds
                        and aruco_new_frames >= self.model_handoff_required_frames
                        and rectangle_new_frames >= self.model_handoff_required_frames):
                    graph_ok, graph_detail = self.verify_model_topic_graph()
                    if not graph_ok:
                        rospy.logerr(
                            "%s：[模型隔离检查失败] %s",
                            NODE_NAME,
                            graph_detail,
                        )
                        return False
                    return True
            else:
                both_ready_at = None
                ready_start_counts = None

            elapsed = now - started_at
            if elapsed >= self.model_ready_timeout:
                rospy.logerr(
                    (
                        "%s：等待模型超过%.1fs，ArUco=%s(%d帧，年龄=%s)，"
                        "方框=%s(%d帧，年龄=%s)"
                    ),
                    NODE_NAME,
                    self.model_ready_timeout,
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
                    "%s：模型预热中，ArUco=%s(%d帧，年龄=%s)，"
                    "方框=%s(%d帧，年龄=%s)，已等待%.1fs"
                ),
                NODE_NAME,
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

    def wait_for_rectangle_model_fresh(self, hold_goal, reason):
        """在交接前确认方框模型仍持续出帧，等待期间保持固定点。"""
        started_at = time.monotonic()
        with self.model_lock:
            baseline_count = self.rectangle_message_count
        while not rospy.is_shutdown():
            self.publish_goal(hold_goal)
            if not self.managed_process_alive(self.MODEL_BUNDLE_ROLE):
                rospy.logerr(
                    "%s：[模型交接检查] %s失败，双模型进程组已经退出",
                    NODE_NAME,
                    reason,
                )
                return False
            with self.model_lock:
                last_message = self.rectangle_last_message_wall_time
                message_count = self.rectangle_message_count
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
                        "方框模型总帧=%d，消息年龄=%.2fs"
                    ),
                    NODE_NAME,
                    reason,
                    new_frame_count,
                    self.model_handoff_required_frames,
                    message_count,
                    age,
                )
                return True
            elapsed = now - started_at
            if elapsed >= self.model_recovery_timeout:
                rospy.logerr(
                    "%s：[模型交接检查] %s失败，%.1fs内方框模型没有恢复连续输出",
                    NODE_NAME,
                    reason,
                    self.model_recovery_timeout,
                )
                return False
            rospy.logwarn_throttle(
                1.0,
                "%s：[模型交接检查] %s等待方框模型恢复，消息年龄=%s，剩余%.1fs",
                NODE_NAME,
                reason,
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

    @staticmethod
    def _launch_bool(value):
        return "true" if value else "false"

    def model_bundle_command(self):
        """递归启动同一launch的仅模型模式，保留两套模型的话题隔离。"""
        return [
            "roslaunch", "auv_control", "task3_subtask2_3_sequence.launch",
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

    def task2_command(self):
        return [
            "roslaunch", "auv_control", "task3_subtask2_get_task.launch",
            "start_aruco_model:=false",
            "pose_cmd_topic:={}".format(self.task2_pose_cmd_topic),
            "aruco_topic:={}".format(self.aruco_topic),
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

    @staticmethod
    def _node_matches(node_name, tokens):
        return any(token in node_name for token in tokens)

    def protected_node_names(self):
        """公共底层节点和当前协调节点不属于任务级清理范围。"""
        return set(self.PROTECTED_PUBLIC_NODES) | {rospy.get_name()}

    def matching_ros_nodes(self, tokens):
        try:
            nodes = rosnode.get_node_names()
        except Exception as error:
            rospy.logerr("%s：[进程检查] 无法读取ROS节点列表：%s", NODE_NAME, str(error))
            return None
        protected = self.protected_node_names()
        return sorted(
            node for node in nodes
            if node not in protected and self._node_matches(node, tokens))

    def model_node_tokens(self):
        return self.MODEL_NODE_TOKENS

    def all_task_node_tokens(self):
        return tuple(sorted(set(self.TASK_NODE_TOKENS + self.model_node_tokens())))

    def request_ros_nodes_shutdown(self, label, nodes):
        if not nodes:
            return
        rospy.logwarn(
            "%s：[%s] 发现残留节点，发送关闭请求：%s",
            NODE_NAME,
            label,
            ", ".join(nodes),
        )
        try:
            success, failed = rosnode.kill_nodes(nodes)
            rospy.loginfo(
                "%s：[%s] rosnode关闭结果：成功=%s，失败=%s",
                NODE_NAME,
                label,
                ", ".join(success) if success else "无",
                ", ".join(failed) if failed else "无",
            )
        except Exception as error:
            rospy.logerr(
                "%s：[%s] 发送ROS节点关闭请求失败：%s",
                NODE_NAME,
                label,
                str(error),
            )

    def ensure_nodes_stopped(self, label, tokens, request_shutdown=True):
        """关闭节点，并连续确认一段时间无残留，防止respawn造成误判。"""
        deadline = time.monotonic() + self.cleanup_timeout
        stable_started_at = None
        last_shutdown_request_at = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            now = time.monotonic()
            remaining = self.matching_ros_nodes(tokens)
            if remaining is None:
                return False
            if not remaining:
                if stable_started_at is None:
                    stable_started_at = now
                    rospy.loginfo(
                        "%s：[%s] 当前无残留，开始连续确认%.1fs",
                        NODE_NAME,
                        label,
                        self.cleanup_stable_seconds,
                    )
                if now - stable_started_at >= self.cleanup_stable_seconds:
                    rospy.loginfo(
                        "%s：[%s] ROS节点清理确认通过，连续%.1fs无残留",
                        NODE_NAME,
                        label,
                        self.cleanup_stable_seconds,
                    )
                    return True
            else:
                if stable_started_at is not None:
                    rospy.logwarn(
                        "%s：[%s] 无残留确认被新出现的节点打断：%s",
                        NODE_NAME,
                        label,
                        ", ".join(remaining),
                    )
                stable_started_at = None
                if (
                        request_shutdown
                        and (
                            last_shutdown_request_at is None
                            or now - last_shutdown_request_at >= 1.0)):
                    self.request_ros_nodes_shutdown(label, remaining)
                    last_shutdown_request_at = now
            rospy.loginfo_throttle(
                1.0,
                "%s：[%s] 清理确认中，当前残留：%s",
                NODE_NAME,
                label,
                ", ".join(remaining) if remaining else "无",
            )
            rospy.sleep(self.cleanup_poll_interval)

        remaining = self.matching_ros_nodes(tokens)
        rospy.logerr(
            "%s：[%s] 清理超过%.1fs，仍残留：%s",
            NODE_NAME,
            label,
            self.cleanup_timeout,
            "无法读取节点列表" if remaining is None else ", ".join(remaining),
        )
        return False

    @staticmethod
    def _pid_still_matches(pid, token):
        """再次核对PID命令行，避免PID复用后误发升级信号。"""
        if os.name != "posix":
            return False
        try:
            with open("/proc/{}/cmdline".format(pid), "rb") as stream:
                command = stream.read().replace(b"\x00", b" ").decode(
                    "utf-8", errors="replace")
        except OSError:
            return False
        return token in command

    def find_task_launch_processes(self):
        if os.name != "posix":
            return []
        try:
            output = subprocess.check_output(
                ["ps", "-eo", "pid=,ppid=,args="],
                universal_newlines=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            rospy.logerr("%s：[启动清理] 无法扫描任务roslaunch：%s", NODE_NAME, str(error))
            return None

        process_rows = {}
        for line in output.splitlines():
            fields = line.strip().split(None, 2)
            if len(fields) != 3:
                continue
            try:
                pid = int(fields[0])
                parent_pid = int(fields[1])
            except ValueError:
                continue
            process_rows[pid] = (parent_pid, fields[2])

        # 当前协调节点由外层联调roslaunch启动。保护整个父进程链，避免清场时
        # 把正在运行的本次联调连同协调节点一起关闭。
        protected_pids = set()
        current_pid = os.getpid()
        while current_pid > 1 and current_pid not in protected_pids:
            protected_pids.add(current_pid)
            row = process_rows.get(current_pid)
            if row is None:
                break
            current_pid = row[0]

        matches = []
        for pid, (_, command) in process_rows.items():
            if pid in protected_pids:
                continue
            token = next(
                (
                    candidate for candidate in self.TASK_LAUNCH_COMMAND_TOKENS
                    if candidate in command
                ),
                None,
            )
            if token is not None:
                matches.append((pid, token))
        return matches

    def stop_task_launch_processes(self):
        if os.name != "posix":
            rospy.logwarn("%s：[启动清理] 当前系统不支持/proc扫描，仅执行ROS节点清理", NODE_NAME)
            return True
        matches = self.find_task_launch_processes()
        if matches is None:
            return False
        if not matches:
            rospy.loginfo("%s：[启动清理] 未发现需要关闭的任务或模型roslaunch进程", NODE_NAME)
            return True

        rospy.logwarn(
            "%s：[启动清理] 发现%d个任务或模型roslaunch，公共底层launch不会关闭：%s",
            NODE_NAME,
            len(matches),
            ", ".join("PID={}({})".format(pid, token) for pid, token in matches),
        )
        for pid, _ in matches:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            except OSError as error:
                rospy.logwarn(
                    "%s：[启动清理] 无法向任务roslaunch PID=%d发送SIGINT：%s",
                    NODE_NAME,
                    pid,
                    str(error),
                )

        deadline = time.monotonic() + self.process_sigint_timeout
        while time.monotonic() < deadline:
            if not any(
                    self._pid_still_matches(pid, token)
                    for pid, token in matches):
                return True
            time.sleep(0.1)

        remaining = [
            (pid, token) for pid, token in matches
            if self._pid_still_matches(pid, token)
        ]
        rospy.logwarn(
            "%s：[启动清理] 任务roslaunch未响应SIGINT，向PID=%s发送SIGTERM",
            NODE_NAME,
            ",".join(str(pid) for pid, _ in remaining),
        )
        for pid, _ in remaining:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as error:
                rospy.logwarn(
                    "%s：[启动清理] 无法向PID=%d发送SIGTERM：%s",
                    NODE_NAME,
                    pid,
                    str(error),
                )
        deadline = time.monotonic() + self.process_sigterm_timeout
        while time.monotonic() < deadline:
            if not any(
                    self._pid_still_matches(pid, token)
                    for pid, token in remaining):
                return True
            time.sleep(0.1)

        remaining = [
            (pid, token) for pid, token in remaining
            if self._pid_still_matches(pid, token)
        ]
        for pid, _ in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.2)
        remaining = [
            (pid, token) for pid, token in remaining
            if self._pid_still_matches(pid, token)
        ]
        if remaining:
            rospy.logerr(
                "%s：[启动清理] 任务roslaunch仍未退出，PID=%s",
                NODE_NAME,
                ",".join(str(pid) for pid, _ in remaining),
            )
            return False
        return True

    def unexpected_control_publishers(self):
        """返回会与本次联调争用控制权的发布节点。"""
        current_node = rospy.get_name()
        allowed_by_topic = {
            self.motion_goal_topic: {current_node},
            self.motion_cancel_topic: {current_node},
            "/cmd/pose/ned": {"/motion_supervisor"},
            "/cmd/actuator": set(),
            self.task2_pose_cmd_topic: set(),
        }
        try:
            publishers, _, _ = rosgraph.Master(current_node).getSystemState()
        except Exception as error:
            rospy.logerr("%s：[控制权检查] 无法读取ROS发布者：%s", NODE_NAME, str(error))
            return None
        publisher_map = {topic: set(nodes) for topic, nodes in publishers}
        conflicts = {}
        for topic, allowed_nodes in allowed_by_topic.items():
            unexpected = publisher_map.get(topic, set()) - allowed_nodes
            if unexpected:
                conflicts[topic] = unexpected
        return conflicts

    def ensure_control_publishers_exclusive(self, label):
        """关闭旧任务控制发布者，并确认控制权不再冲突。"""
        deadline = time.monotonic() + self.cleanup_timeout
        stable_started_at = None
        last_shutdown_request_at = None
        protected = self.protected_node_names()
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            now = time.monotonic()
            conflicts = self.unexpected_control_publishers()
            if conflicts is None:
                return False
            unexpected_nodes = sorted(set().union(*conflicts.values())) if conflicts else []
            protected_conflicts = sorted(set(unexpected_nodes) & protected)
            if protected_conflicts:
                rospy.logerr(
                    "%s：[%s] 受保护公共节点正在异常发布任务控制话题，不会强制关闭：%s",
                    NODE_NAME,
                    label,
                    ", ".join(protected_conflicts),
                )
                return False
            if not unexpected_nodes:
                if stable_started_at is None:
                    stable_started_at = now
                    rospy.loginfo(
                        "%s：[%s] 控制话题无旧任务发布者，开始连续确认%.1fs",
                        NODE_NAME,
                        label,
                        self.cleanup_stable_seconds,
                    )
                if now - stable_started_at >= self.cleanup_stable_seconds:
                    rospy.loginfo(
                        "%s：[%s] 控制权检查通过：goal/cancel由协调节点独占，"
                        "/cmd/pose/ned仅允许motion_supervisor发布，执行器无旧发布者",
                        NODE_NAME,
                        label,
                    )
                    return True
            else:
                stable_started_at = None
                detail = "；".join(
                    "{}={}".format(topic, ",".join(sorted(nodes)))
                    for topic, nodes in sorted(conflicts.items())
                )
                rospy.logwarn_throttle(
                    1.0,
                    "%s：[%s] 发现旧任务控制发布者：%s",
                    NODE_NAME,
                    label,
                    detail,
                )
                if (
                        last_shutdown_request_at is None
                        or now - last_shutdown_request_at >= 1.0):
                    self.request_ros_nodes_shutdown(label, unexpected_nodes)
                    last_shutdown_request_at = now
            rospy.sleep(self.cleanup_poll_interval)

        conflicts = self.unexpected_control_publishers()
        detail = "无法读取" if conflicts is None else "；".join(
            "{}={}".format(topic, ",".join(sorted(nodes)))
            for topic, nodes in sorted(conflicts.items())
        ) or "无"
        rospy.logerr(
            "%s：[%s] 控制权清理超过%.1fs，冲突发布者：%s",
            NODE_NAME,
            label,
            self.cleanup_timeout,
            detail,
        )
        return False

    def wait_for_motion_supervisor_backbone(self):
        """启动模型前确认公共控制器的话题接口已经完整注册。"""
        deadline = time.monotonic() + self.control_backbone_timeout
        last_detail = "尚未检查"
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            try:
                master = rosgraph.Master(rospy.get_name())
                publishers, subscribers, _ = master.getSystemState()
                topic_types = dict(master.getTopicTypes())
                current_nodes = set(rosnode.get_node_names())
            except Exception as error:
                last_detail = "无法读取ROS图：{}".format(error)
                rospy.logwarn_throttle(
                    1.0,
                    "%s：[公共控制检查] %s",
                    NODE_NAME,
                    last_detail,
                )
                rospy.sleep(self.cleanup_poll_interval)
                continue

            publisher_map = {topic: set(nodes) for topic, nodes in publishers}
            subscriber_map = {topic: set(nodes) for topic, nodes in subscribers}
            failures = []
            if "/motion_supervisor" not in current_nodes:
                failures.append("/motion_supervisor节点不存在")
            if publisher_map.get("/cmd/pose/ned", set()) != {"/motion_supervisor"}:
                failures.append(
                    "/cmd/pose/ned发布者={}".format(
                        ",".join(sorted(publisher_map.get("/cmd/pose/ned", set())))
                        or "无"))
            if publisher_map.get(self.motion_state_topic, set()) != {"/motion_supervisor"}:
                failures.append(
                    "{}发布者={}".format(
                        self.motion_state_topic,
                        ",".join(sorted(
                            publisher_map.get(self.motion_state_topic, set()))) or "无"))
            if "/motion_supervisor" not in subscriber_map.get(
                    self.motion_goal_topic, set()):
                failures.append("{}缺少motion_supervisor订阅".format(
                    self.motion_goal_topic))
            if "/motion_supervisor" not in subscriber_map.get(
                    self.motion_cancel_topic, set()):
                failures.append("{}缺少motion_supervisor订阅".format(
                    self.motion_cancel_topic))
            if topic_types.get("/cmd/pose/ned") != "auv_control/PoseNEDcmd":
                failures.append(
                    "/cmd/pose/ned类型={}".format(
                        topic_types.get("/cmd/pose/ned") or "未知"))
            if topic_types.get(self.motion_state_topic) != "auv_control/MotionState":
                failures.append(
                    "{}类型={}".format(
                        self.motion_state_topic,
                        topic_types.get(self.motion_state_topic) or "未知"))

            if not failures:
                rospy.loginfo(
                    "%s：[公共控制检查] motion_supervisor接口完整，公共控制节点保持运行",
                    NODE_NAME,
                )
                return True
            last_detail = "；".join(failures)
            rospy.logwarn_throttle(
                1.0,
                "%s：[公共控制检查] 等待公共控制器就绪：%s",
                NODE_NAME,
                last_detail,
            )
            rospy.sleep(self.cleanup_poll_interval)

        rospy.logerr(
            "%s：[公共控制检查] 超过%.1fs仍未就绪：%s；不会启动识别模型",
            NODE_NAME,
            self.control_backbone_timeout,
            last_detail,
        )
        return False

    def preflight_cleanup(self):
        rospy.loginfo(
            "%s：[启动清理] 无条件关闭全部任务级节点、识别模型和模型Web；"
            "保留begin公共节点、相机驱动、导航融合及motion_supervisor",
            NODE_NAME,
        )
        try:
            current_nodes = set(rosnode.get_node_names())
        except Exception:
            current_nodes = set()
        protected_running = sorted(self.protected_node_names() & current_nodes)
        rospy.loginfo(
            "%s：[启动清理] 当前受保护节点：%s",
            NODE_NAME,
            ", ".join(protected_running) if protected_running else "未发现",
        )
        processes_stopped = self.stop_task_launch_processes()
        nodes_stopped = self.ensure_nodes_stopped(
            "启动清理", self.all_task_node_tokens(), request_shutdown=True)
        self.motion_cancel_pub.publish(Empty())
        rospy.loginfo(
            "%s：[启动清理] 旧任务源清除后已发布一次%s，取消旧目标并进入停稳悬停",
            NODE_NAME,
            self.motion_cancel_topic,
        )
        control_exclusive = self.ensure_control_publishers_exclusive("启动控制权复核")
        control_backbone_ready = self.wait_for_motion_supervisor_backbone()
        ports_available, ports_detail = self.verify_web_ports(
            expected_open=False)
        if ports_available:
            rospy.loginfo("%s：[启动清理] %s", NODE_NAME, ports_detail)
        else:
            rospy.logerr("%s：[启动清理] %s", NODE_NAME, ports_detail)
        cleanup_ok = (
            processes_stopped
            and nodes_stopped
            and control_exclusive
            and control_backbone_ready
            and ports_available
        )
        rospy.loginfo(
            "%s：[启动清理] 清场%s；下一步%s启动任务3双模型",
            NODE_NAME,
            "完成" if cleanup_ok else "失败",
            "允许" if cleanup_ok else "禁止",
        )
        return cleanup_ok

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
            entry = self.managed_processes.pop(role, None)
        if entry is None:
            return True
        process = entry["process"]
        label = entry["label"]
        process_group_id = entry["process_group_id"]
        if not self._managed_entry_alive(entry):
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
        return self._wait_managed_process_exit(entry, 1.0)

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
            "双模型常驻进程组",
            self.model_bundle_command(),
        )
        return process is not None

    def stop_model_bundle(self, label="总任务模型清理"):
        process_stopped = self.stop_managed_process(self.MODEL_BUNDLE_ROLE)
        nodes_stopped = self.ensure_nodes_stopped(
            label,
            self.model_node_tokens(),
            request_shutdown=True,
        )
        return process_stopped and nodes_stopped

    def start_child(self, stage_prefix, stage_name, command):
        role = "task_{}".format(stage_prefix)
        process = self.start_managed_process(role, stage_name + "执行launch", command)
        self.child_process = process
        self.child_process_role = role if process is not None else None
        return process is not None

    def stop_child(self, stage_prefix=None, stage_name="当前子任务"):
        role = self.child_process_role
        self.child_process = None
        self.child_process_role = None
        process_stopped = True if role is None else self.stop_managed_process(role)
        if stage_prefix is None:
            return process_stopped
        nodes_stopped = self.ensure_nodes_stopped(
            stage_name + "阶段清理",
            (stage_prefix,),
            request_shutdown=True,
        )
        return process_stopped and nodes_stopped

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
                    "%s：[控制权交接失败] 双模型进程组已经退出",
                    NODE_NAME,
                )
                return False, "子任务3接管前双模型进程组已经退出"

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
                    return False, "子任务3首批目标与转向完成点不一致：{}".format(detail)
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
                    "%s：[控制权交接] 联调节点继续保持转向完成点，"
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
            hold_goal=None, handoff_goal=None):
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
                cleanup_success = self.stop_child(stage_prefix, stage_name)
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
            if not self.managed_process_alive(self.MODEL_BUNDLE_ROLE):
                stage_failure_detail = "{}期间双模型常驻进程组异常退出".format(stage_name)
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

        cleanup_success = self.stop_child(stage_prefix, stage_name)
        with self.stage_lock:
            self.current_stage = None
            self.stage_result = None
            self.stage_event.clear()

        if not cleanup_success:
            return False, "{}结束后执行launch或ROS节点未完全关闭".format(stage_name)
        if result is None:
            if stage_failure_detail is not None:
                return False, stage_failure_detail
            return False, "{}未返回完成消息或已超时".format(stage_name)
        success = result.startswith("{} finished".format(stage_prefix))
        return success, result

    def make_rotation_goal(self, hold_goal):
        rotation_goal = copy.deepcopy(hold_goal)
        start_yaw = self._yaw_from_pose(hold_goal.pose)
        target_yaw = normalize_angle(
            start_yaw + math.radians(self.signed_turn_angle_deg()))
        quaternion = quaternion_from_euler(0.0, 0.0, target_yaw)
        rotation_goal.pose.orientation.x = quaternion[0]
        rotation_goal.pose.orientation.y = quaternion[1]
        rotation_goal.pose.orientation.z = quaternion[2]
        rotation_goal.pose.orientation.w = quaternion[3]
        rospy.loginfo(
            (
                "%s：[转向目标] %s%.1f度，保持位置(%.3f,%.3f,%.3f)不变，"
                "航向%.1fdeg -> %.1fdeg"
            ),
            NODE_NAME,
            self.turn_direction_text(),
            self.transition_turn_angle_deg,
            rotation_goal.pose.position.x,
            rotation_goal.pose.position.y,
            rotation_goal.pose.position.z,
            math.degrees(start_yaw),
            math.degrees(target_yaw),
        )
        return rotation_goal

    def hold_goal_for(self, goal, seconds, reason):
        started_at = time.monotonic()
        while not rospy.is_shutdown() and time.monotonic() - started_at < seconds:
            self.publish_goal(goal)
            rospy.loginfo_throttle(
                1.0,
                "%s：%s %.1f/%.1fs",
                NODE_NAME,
                reason,
                min(time.monotonic() - started_at, seconds),
                seconds,
            )
            self.rate.sleep()

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
        model_cleanup = self.stop_model_bundle("总任务结束模型清理")
        process_cleanup = self.stop_remaining_managed_processes()
        task_launch_cleanup = self.stop_task_launch_processes()
        node_cleanup = self.ensure_nodes_stopped(
            "总任务结束残留复查",
            self.all_task_node_tokens(),
            request_shutdown=True,
        )
        control_cleanup = self.ensure_control_publishers_exclusive(
            "总任务结束控制权复查")
        cleanup = (
            child_cleanup
            and model_cleanup
            and process_cleanup
            and task_launch_cleanup
            and node_cleanup
            and control_cleanup
        )
        if not safe or not cleanup:
            failure_parts = [detail]
            if not safe:
                failure_parts.append("最终安全悬停未确认")
            if not cleanup:
                failure_parts.append("任务或模型相关进程未完全关闭")
            self.finish(False, "；".join(failure_parts))
            return False
        self.finish(
            requested_success,
            (
                "{}；机器人已安全悬停；两个子任务执行launch和"
                "两个识别模型均已关闭"
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
        if not self.preflight_cleanup():
            self.finish(False, "启动任务级清场、控制权复核或Web端口复核未通过")
            return
        if not self.start_model_bundle():
            self.finish(False, "两个识别模型进程组启动失败")
            return

        hold_goal = self.capture_current_goal()
        if hold_goal is None:
            cleanup = self.stop_model_bundle("启动失败模型清理")
            detail = "未能获取启动时map到base_link的固定姿态"
            if not cleanup:
                detail += "；两个识别模型未完全关闭"
            self.finish(False, detail)
            return

        if not self.wait_for_goal(
                hold_goal,
                self.initial_hold_timeout,
                self.transition_stable_seconds,
                "启动定点"):
            self.finish_after_safe_stop(False, "motion_supervisor未能完成启动定点")
            return

        if not self.wait_for_models(hold_goal):
            self.finish_after_safe_stop(False, "两个识别模型未能全部就绪")
            return

        rospy.loginfo(
            (
                "%s：[阶段1/3 子任务2] 启动任务节点；参数来自"
                "task3_subtask2_get_task.launch。方框模型继续预热，"
                "但其识别信息不会进入子任务2"
            ),
            NODE_NAME,
        )
        task2_success, task2_detail = self.run_child_stage(
            self.TASK2_PREFIX,
            "子任务2",
            self.task2_command(),
            self.task2_stage_timeout,
            hold_goal=hold_goal,
        )
        if not task2_success:
            self.finish_after_safe_stop(
                False,
                "子任务2失败，不执行旋转和子任务3：{}".format(task2_detail),
            )
            return

        rospy.loginfo(
            "%s：[阶段1/3 子任务2] 成功：%s",
            NODE_NAME,
            task2_detail,
        )
        if not self.prepare_task3_color(task2_detail):
            self.finish_after_safe_stop(
                False,
                "子任务2虽然成功，但颜色数据无法交给子任务3",
            )
            return
        graph_ok, graph_detail = self.verify_model_topic_graph()
        if not graph_ok:
            self.finish_after_safe_stop(
                False,
                "子任务2结束后双模型隔离检查失败：{}".format(graph_detail),
            )
            return
        rospy.loginfo(
            "%s：[阶段衔接2->转向] 子任务2退出后重新确认启动固定点，不直接开始旋转",
            NODE_NAME,
        )
        if not self.wait_for_goal(
                hold_goal,
                self.post_task2_hold_timeout,
                self.post_task2_stable_seconds,
                "子任务2结束后的固定点复稳"):
            self.finish_after_safe_stop(
                False,
                "子任务2成功，但机器人未能重新稳定在启动固定点",
            )
            return
        if not self.wait_for_rectangle_model_fresh(
                hold_goal, "转向前方框模型健康检查"):
            self.finish_after_safe_stop(
                False,
                "子任务2成功，但方框模型在转向前没有持续输出",
            )
            return
        rospy.loginfo(
            "%s：[阶段2/3 联调转向] 开始%s%.1f度",
            NODE_NAME,
            self.turn_direction_text(),
            self.transition_turn_angle_deg,
        )
        rotation_goal = self.make_rotation_goal(hold_goal)
        if not self.wait_for_goal(
                rotation_goal,
                self.transition_timeout,
                self.transition_stable_seconds,
                "原地{}过渡".format(self.turn_direction_text())):
            self.finish_after_safe_stop(
                False,
                "子任务2成功，但原地{}{}度未稳定到达，已停止后续子任务".format(
                    self.turn_direction_text(),
                    self.transition_turn_angle_deg,
                ),
            )
            return
        self.hold_goal_for(
            rotation_goal,
            self.transition_hold_seconds,
            "旋转完成后的定点保持",
        )
        if not self.wait_for_rectangle_model_fresh(
                rotation_goal, "启动子任务3前方框模型复查"):
            self.finish_after_safe_stop(
                False,
                "转向成功，但方框模型在子任务3接管前没有持续输出",
            )
            return
        graph_ok, graph_detail = self.verify_model_topic_graph()
        if not graph_ok:
            self.finish_after_safe_stop(
                False,
                "启动子任务3前双模型隔离检查失败：{}".format(graph_detail),
            )
            return

        rospy.loginfo(
            (
                "%s：[阶段3/3 子任务3] 转向稳定且方框模型输出正常；"
                "先启动子任务3并保持转向完成点，握手成功后再交出控制权。参数来自"
                "task3_subtask3_inspect_and_drop.launch"
            ),
            NODE_NAME,
        )
        task3_success, task3_detail = self.run_child_stage(
            self.TASK3_PREFIX,
            "子任务3",
            self.task3_command(),
            self.task3_stage_timeout,
            handoff_goal=rotation_goal,
        )
        if not task3_success:
            self.finish_after_safe_stop(
                False,
                "子任务3失败：{}".format(task3_detail),
            )
            return

        self.finish_after_safe_stop(
            True,
            "子任务2、原地转向、控制权交接和子任务3均执行成功",
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
        Task3Subtask23Sequence().run()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        pass
    except Exception as error:
        rospy.logfatal("%s：未处理异常：%s", NODE_NAME, str(error))
        raise
