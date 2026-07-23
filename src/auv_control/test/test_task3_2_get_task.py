#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务3子任务2：定点识别 ArUco、点亮对应颜色灯并原地转向。

人工把机器人停在目标正前方后，节点记录当前 ``map -> base_link`` 位姿，
通过 ``motion_supervisor`` 做动力定位。启动后先悬停10秒，再开始60秒识别计时。

识别采用最近10个模型消息组成的滑动窗口。任一合法 ArUco ID 在窗口内
出现3次即立即确认，不要求连续，也不需要等待窗口填满。例如第1、3、7帧
是同一个ID时，第7帧到达后立即成功。成功后点亮对应颜色灯3秒，再按配置
原地左转或右转；60秒内未确认或转向未稳定到达都按失败结束。
"""

import math
import re
import threading
import time
from collections import Counter, deque

import rospy
import tf

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from auv_control.msg import ActuatorControl, MotionState, TargetDetection


NODE_NAME = "test_task3_2_get_task"

DEFAULT_RATE = 10.0
DEFAULT_MOTION_GOAL_TOPIC = "/cmd/motion/goal"
DEFAULT_MOTION_CANCEL_TOPIC = "/cmd/motion/cancel"
DEFAULT_MOTION_STATE_TOPIC = "/motion/state"
DEFAULT_HOLD_MAP_FRAME = "map"
DEFAULT_HOLD_BASE_FRAME = "base_link"
DEFAULT_INITIAL_HOVER_SECONDS = 10.0
DEFAULT_INITIAL_HOVER_TIMEOUT = 30.0
DEFAULT_HOLD_POSE_TIMEOUT = 5.0
DEFAULT_MOTION_STATE_TIMEOUT = 0.5

DEFAULT_TURN_ENABLED = True
DEFAULT_TURN_DIRECTION = "right"
DEFAULT_TURN_ANGLE_DEG = 90.0
DEFAULT_TURN_TIMEOUT = 90.0
DEFAULT_TURN_STABLE_SECONDS = 1.0
DEFAULT_TURN_HOLD_SECONDS = 1.0

DEFAULT_GOAL_MATCH_POSITION_TOLERANCE = 0.03
DEFAULT_GOAL_MATCH_DEPTH_TOLERANCE = 0.03
DEFAULT_GOAL_MATCH_YAW_TOLERANCE_DEG = 2.0
DEFAULT_ARRIVAL_POSITION_TOLERANCE = 0.05
DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG = 5.0
DEFAULT_ARRIVAL_MAX_HORIZONTAL_SPEED = 0.03
DEFAULT_ARRIVAL_MAX_YAW_RATE = 0.05

DEFAULT_INPUT_MODE = "topic"
DEFAULT_ARUCO_TOPIC = "/vision/aruco/target_message"
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_RECOGNITION_WINDOW_SIZE = 10
DEFAULT_REQUIRED_MATCH_COUNT = 3
DEFAULT_RECOGNITION_TIMEOUT = 60.0
DEFAULT_MOCK_FRAME_INTERVAL = 0.2
# 模拟第1、3、7帧为同一个ID，-1表示空帧。
DEFAULT_MOCK_ARUCO_IDS = [1, -1, 1, 2, -1, -1, 1]

DEFAULT_LIGHT_SECONDS = 3.0
DEFAULT_GAP_SECONDS = 0.5
DEFAULT_ACTUATOR_TOPIC = "/cmd/actuator"
DEFAULT_ACTUATOR_MODE = 2
DEFAULT_LIGHT1 = 0
DEFAULT_LIGHT2 = 0
DEFAULT_HEADING_SERVO = 0x80
DEFAULT_CLAMP_SERVO = 0xFF
DEFAULT_DRIVE_CMD = 0
DEFAULT_DRIVE_SPEED = 0


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Task3GetTaskTest:
    COLOR_BY_MARKER = {
        1: "yellow",
        2: "yellow",
        3: "green",
        4: "green",
        5: "red",
        6: "red",
    }

    ACTUATOR_LIGHTS = {
        "yellow": (0, 1, 0),
        "green": (0, 0, 1),
        "red": (1, 0, 0),
        "off": (0, 0, 0),
    }

    def __init__(self):
        self.rate_hz = float(rospy.get_param("~rate", DEFAULT_RATE))
        self.rate = rospy.Rate(self.rate_hz)

        self.motion_goal_topic = str(rospy.get_param(
            "~motion_goal_topic", DEFAULT_MOTION_GOAL_TOPIC
        )).strip()
        self.motion_cancel_topic = str(rospy.get_param(
            "~motion_cancel_topic", DEFAULT_MOTION_CANCEL_TOPIC
        )).strip()
        self.motion_state_topic = str(rospy.get_param(
            "~motion_state_topic", DEFAULT_MOTION_STATE_TOPIC
        )).strip()
        self.hold_map_frame = str(
            rospy.get_param("~hold_map_frame", DEFAULT_HOLD_MAP_FRAME)
        ).strip()
        self.hold_base_frame = str(
            rospy.get_param("~hold_base_frame", DEFAULT_HOLD_BASE_FRAME)
        ).strip()
        self.initial_hover_seconds = float(rospy.get_param(
            "~initial_hover_seconds", DEFAULT_INITIAL_HOVER_SECONDS
        ))
        self.initial_hover_timeout = float(rospy.get_param(
            "~initial_hover_timeout", DEFAULT_INITIAL_HOVER_TIMEOUT
        ))
        self.hold_pose_timeout = float(rospy.get_param(
            "~hold_pose_timeout", DEFAULT_HOLD_POSE_TIMEOUT
        ))
        self.motion_state_timeout = float(rospy.get_param(
            "~motion_state_timeout", DEFAULT_MOTION_STATE_TIMEOUT
        ))

        self.turn_enabled = bool(rospy.get_param(
            "~turn_enabled", DEFAULT_TURN_ENABLED
        ))
        self.turn_direction = str(rospy.get_param(
            "~turn_direction", DEFAULT_TURN_DIRECTION
        )).strip().lower()
        self.turn_angle_deg = float(rospy.get_param(
            "~turn_angle_deg", DEFAULT_TURN_ANGLE_DEG
        ))
        self.turn_timeout = float(rospy.get_param(
            "~turn_timeout", DEFAULT_TURN_TIMEOUT
        ))
        self.turn_stable_seconds = float(rospy.get_param(
            "~turn_stable_seconds", DEFAULT_TURN_STABLE_SECONDS
        ))
        self.turn_hold_seconds = float(rospy.get_param(
            "~turn_hold_seconds", DEFAULT_TURN_HOLD_SECONDS
        ))

        self.goal_match_position_tolerance = float(rospy.get_param(
            "~goal_match_position_tolerance",
            DEFAULT_GOAL_MATCH_POSITION_TOLERANCE,
        ))
        self.goal_match_depth_tolerance = float(rospy.get_param(
            "~goal_match_depth_tolerance",
            DEFAULT_GOAL_MATCH_DEPTH_TOLERANCE,
        ))
        self.goal_match_yaw_tolerance = math.radians(float(rospy.get_param(
            "~goal_match_yaw_tolerance_deg",
            DEFAULT_GOAL_MATCH_YAW_TOLERANCE_DEG,
        )))
        self.arrival_position_tolerance = float(rospy.get_param(
            "~arrival_position_tolerance",
            DEFAULT_ARRIVAL_POSITION_TOLERANCE,
        ))
        self.arrival_yaw_tolerance = math.radians(float(rospy.get_param(
            "~arrival_yaw_tolerance_deg",
            DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG,
        )))
        self.arrival_max_horizontal_speed = float(rospy.get_param(
            "~arrival_max_horizontal_speed",
            DEFAULT_ARRIVAL_MAX_HORIZONTAL_SPEED,
        ))
        self.arrival_max_yaw_rate = float(rospy.get_param(
            "~arrival_max_yaw_rate",
            DEFAULT_ARRIVAL_MAX_YAW_RATE,
        ))

        self.input_mode = str(
            rospy.get_param("~input_mode", DEFAULT_INPUT_MODE)
        ).strip().lower()
        self.aruco_topic = str(
            rospy.get_param("~aruco_topic", DEFAULT_ARUCO_TOPIC)
        ).strip()
        self.min_confidence = float(
            rospy.get_param("~min_confidence", DEFAULT_MIN_CONFIDENCE)
        )
        self.recognition_window_size = int(rospy.get_param(
            "~recognition_window_size", DEFAULT_RECOGNITION_WINDOW_SIZE
        ))
        self.required_match_count = int(rospy.get_param(
            "~required_match_count", DEFAULT_REQUIRED_MATCH_COUNT
        ))
        self.recognition_timeout = float(rospy.get_param(
            "~recognition_timeout", DEFAULT_RECOGNITION_TIMEOUT
        ))
        self.mock_frame_interval = float(rospy.get_param(
            "~mock_frame_interval", DEFAULT_MOCK_FRAME_INTERVAL
        ))
        self.mock_aruco_ids = self.parse_marker_sequence(
            rospy.get_param("~mock_aruco_ids", DEFAULT_MOCK_ARUCO_IDS)
        )

        self.light_seconds = float(
            rospy.get_param("~light_seconds", DEFAULT_LIGHT_SECONDS)
        )
        self.gap_seconds = float(
            rospy.get_param("~gap_seconds", DEFAULT_GAP_SECONDS)
        )
        self.actuator_topic = str(
            rospy.get_param("~actuator_topic", DEFAULT_ACTUATOR_TOPIC)
        ).strip()
        self.actuator_mode = int(
            rospy.get_param("~actuator_mode", DEFAULT_ACTUATOR_MODE)
        )
        self.light1 = int(rospy.get_param("~light1", DEFAULT_LIGHT1))
        self.light2 = int(rospy.get_param("~light2", DEFAULT_LIGHT2))
        self.heading_servo = int(
            rospy.get_param("~heading_servo", DEFAULT_HEADING_SERVO)
        )
        self.clamp_servo = int(
            rospy.get_param("~clamp_servo", DEFAULT_CLAMP_SERVO)
        )
        self.drive_cmd = int(
            rospy.get_param("~drive_cmd", DEFAULT_DRIVE_CMD)
        )
        self.drive_speed = int(
            rospy.get_param("~drive_speed", DEFAULT_DRIVE_SPEED)
        )

        self.validate_params()

        self.hold_goal = None
        self.active_goal = None
        self.motion_state_lock = threading.Lock()
        self.latest_motion_state = None
        self.latest_motion_state_wall_time = None
        self.accept_detections = False
        self.model_message_index = 0
        self.recognition_frame_index = 0
        self.mock_index = 0
        self.confirmed_marker_id = None
        self.recognition_lock = threading.Lock()
        self.recognition_window = deque(maxlen=self.recognition_window_size)
        self.outputs_closed = False

        self.motion_goal_pub = rospy.Publisher(
            self.motion_goal_topic, PoseStamped, queue_size=1
        )
        self.motion_cancel_pub = rospy.Publisher(
            self.motion_cancel_topic, Empty, queue_size=1
        )
        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.finished_pub = rospy.Publisher(
            "/finished", String, queue_size=10
        )
        self.tf_listener = tf.TransformListener()
        self.actuator_mode_supported = hasattr(ActuatorControl(), "mode")
        self.motion_state_sub = rospy.Subscriber(
            self.motion_state_topic,
            MotionState,
            self.motion_state_callback,
            queue_size=20,
        )

        self.aruco_sub = None
        if self.input_mode == "topic":
            self.aruco_sub = rospy.Subscriber(
                self.aruco_topic,
                TargetDetection,
                self.aruco_callback,
                queue_size=20,
            )

        rospy.on_shutdown(self.on_shutdown)
        self.log_startup_config()

    def validate_params(self):
        if self.rate_hz <= 0.0:
            raise ValueError("rate 必须大于 0")
        if self.input_mode not in ("topic", "mock"):
            raise ValueError("input_mode 必须是 topic 或 mock")
        if self.input_mode == "topic" and not self.aruco_topic:
            raise ValueError("aruco_topic 不能为空")
        if self.input_mode == "mock" and not self.mock_aruco_ids:
            raise ValueError("mock_aruco_ids 不能为空")
        if not all((
                self.motion_goal_topic,
                self.motion_cancel_topic,
                self.motion_state_topic)):
            raise ValueError("motion goal、cancel和state话题不能为空")
        if not self.hold_map_frame or not self.hold_base_frame:
            raise ValueError("hold_map_frame 和 hold_base_frame 不能为空")
        if self.initial_hover_seconds < 0.0:
            raise ValueError("initial_hover_seconds 不能小于 0")
        if min(
                self.initial_hover_timeout,
                self.hold_pose_timeout,
                self.motion_state_timeout,
                self.turn_timeout) <= 0.0:
            raise ValueError("悬停、TF、运动状态和转向超时必须大于0")
        if self.turn_direction not in ("left", "right"):
            raise ValueError("turn_direction 必须是 left 或 right")
        if self.turn_angle_deg <= 0.0 or self.turn_angle_deg >= 180.0:
            raise ValueError("turn_angle_deg 必须在0到180度之间")
        if min(
                self.turn_stable_seconds,
                self.turn_hold_seconds,
                self.goal_match_position_tolerance,
                self.goal_match_depth_tolerance,
                self.goal_match_yaw_tolerance,
                self.arrival_position_tolerance,
                self.arrival_yaw_tolerance,
                self.arrival_max_horizontal_speed,
                self.arrival_max_yaw_rate) < 0.0:
            raise ValueError("转向稳定时间、保持时间和到达门槛不能小于0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence 必须在 0 到 1 之间")
        if self.recognition_window_size <= 0:
            raise ValueError("recognition_window_size 必须为正数")
        if self.required_match_count <= 0:
            raise ValueError("required_match_count 必须为正数")
        if self.required_match_count > self.recognition_window_size:
            raise ValueError("required_match_count 不能大于 recognition_window_size")
        if self.recognition_timeout <= 0.0:
            raise ValueError("recognition_timeout 必须大于 0")
        if self.mock_frame_interval <= 0.0:
            raise ValueError("mock_frame_interval 必须大于 0")
        if min(self.light_seconds, self.gap_seconds) < 0.0:
            raise ValueError("灯光持续时间不能小于 0")
        if self.actuator_mode not in (0, 1, 2):
            raise ValueError("actuator_mode 必须是 0、1 或 2")
        if self.actuator_mode != 2:
            rospy.logwarn(
                "%s：三色指示灯属于执行器，actuator_mode 应设置为 2",
                NODE_NAME,
            )

    def log_startup_config(self):
        rospy.loginfo(
            (
                "%s：流程=motion_supervisor定点悬停%.1fs -> "
                "最近%d帧内同ID达到%d帧 -> 亮灯%.1fs -> %s%.1f度；"
                "识别最长%.1fs"
            ),
            NODE_NAME,
            self.initial_hover_seconds,
            self.recognition_window_size,
            self.required_match_count,
            self.light_seconds,
            "右转" if self.turn_direction == "right" else "左转",
            self.turn_angle_deg,
            self.recognition_timeout,
        )
        rospy.loginfo(
            (
                "%s：运动接口={目标:%s,取消:%s,状态:%s}，TF=%s -> %s；"
                "识别模式=%s，话题=%s"
            ),
            NODE_NAME,
            self.motion_goal_topic,
            self.motion_cancel_topic,
            self.motion_state_topic,
            self.hold_map_frame,
            self.hold_base_frame,
            self.input_mode,
            self.aruco_topic,
        )
        rospy.loginfo(
            (
                "%s：转向参数：启用=%s，方向=%s，角度=%.1fdeg，超时=%.1fs，"
                "稳定确认=%.1fs，完成后保持=%.1fs"
            ),
            NODE_NAME,
            str(self.turn_enabled),
            "右转" if self.turn_direction == "right" else "左转",
            self.turn_angle_deg,
            self.turn_timeout,
            self.turn_stable_seconds,
            self.turn_hold_seconds,
        )
        rospy.loginfo(
            (
                "%s：到达门槛：位置<=%.3fm，航向<=%.1fdeg，"
                "水平速度<=%.3fm/s，航向角速度<=%.3frad/s"
            ),
            NODE_NAME,
            self.arrival_position_tolerance,
            math.degrees(self.arrival_yaw_tolerance),
            self.arrival_max_horizontal_speed,
            self.arrival_max_yaw_rate,
        )
        rospy.loginfo(
            "%s：执行器话题=%s，actuator_mode=%d，最低置信度=%.2f",
            NODE_NAME,
            self.actuator_topic,
            self.actuator_mode,
            self.min_confidence,
        )

    @staticmethod
    def parse_marker_sequence(raw_value):
        if isinstance(raw_value, (list, tuple)):
            return [int(value) for value in raw_value]
        return [int(value) for value in re.findall(r"[+-]?\d+", str(raw_value))]

    @staticmethod
    def parse_marker_id_text(raw_value):
        text = str(raw_value).strip()
        if not text:
            return None
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)

        patterns = (
            r"(?i)(?:aruco|marker)\s*(?:id)?\s*[:=_#-]?\s*(\d+)",
            r"(?i)\bid\s*[:=_#-]?\s*(\d+)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return None

    @classmethod
    def marker_id_from_detection(cls, message):
        for raw_value in (message.class_name, message.type):
            marker_id = cls.parse_marker_id_text(raw_value)
            if marker_id is not None:
                return marker_id
        return None

    @classmethod
    def color_for_marker(cls, marker_id):
        return cls.COLOR_BY_MARKER.get(marker_id)

    @staticmethod
    def yaw_from_pose(pose):
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
        rospy.loginfo_throttle(
            1.0,
            (
                "%s：运动反馈：state=%d，goal_active=%s，位置误差=%.3fm，"
                "航向误差=%+.1fdeg，水平速度=%.3fm/s，"
                "航向角速度=%+.2fdeg/s，输出=(%d,%d,%d)，原因=%s"
            ),
            NODE_NAME,
            message.state,
            str(bool(message.goal_active)),
            message.base_position_error,
            math.degrees(message.yaw_error),
            message.horizontal_speed,
            math.degrees(message.yaw_rate),
            message.tx,
            message.ty,
            message.mz,
            message.reason or "无",
        )

    def capture_hold_pose(self):
        deadline = rospy.Time.now() + rospy.Duration(self.hold_pose_timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                self.tf_listener.waitForTransform(
                    self.hold_map_frame,
                    self.hold_base_frame,
                    rospy.Time(0),
                    rospy.Duration(0.5),
                )
                translation, rotation = self.tf_listener.lookupTransform(
                    self.hold_map_frame,
                    self.hold_base_frame,
                    rospy.Time(0),
                )
            except tf.Exception as error:
                rospy.logwarn_throttle(
                    1.0,
                    "%s：等待定点 TF %s -> %s：%s",
                    NODE_NAME,
                    self.hold_map_frame,
                    self.hold_base_frame,
                    str(error),
                )
                self.rate.sleep()
                continue

            values = tuple(translation) + tuple(rotation)
            if not all(math.isfinite(float(value)) for value in values):
                rospy.logwarn_throttle(
                    1.0,
                    "%s：定点 TF 包含无效数值，等待下一帧",
                    NODE_NAME,
                )
                self.rate.sleep()
                continue

            goal = PoseStamped()
            goal.header.frame_id = self.hold_map_frame
            goal.pose.position.x = float(translation[0])
            goal.pose.position.y = float(translation[1])
            goal.pose.position.z = float(translation[2])
            goal.pose.orientation.x = float(rotation[0])
            goal.pose.orientation.y = float(rotation[1])
            goal.pose.orientation.z = float(rotation[2])
            goal.pose.orientation.w = float(rotation[3])
            yaw = self.yaw_from_pose(goal.pose)
            if yaw is None:
                rospy.logwarn_throttle(1.0, "%s：定点四元数无效，等待下一帧", NODE_NAME)
                self.rate.sleep()
                continue
            self.hold_goal = goal
            self.active_goal = goal
            rospy.loginfo(
                (
                    "%s：已锁存motion_supervisor固定点：%s坐标=(%.3f,%.3f,%.3f)，"
                    "yaw=%.1fdeg；后续漂移不会更新该点"
                ),
                NODE_NAME,
                self.hold_map_frame,
                goal.pose.position.x,
                goal.pose.position.y,
                goal.pose.position.z,
                math.degrees(yaw),
            )
            return True

        rospy.logerr(
            "%s：%.1fs 内未获得 TF %s -> %s",
            NODE_NAME,
            self.hold_pose_timeout,
            self.hold_map_frame,
            self.hold_base_frame,
        )
        return False

    def publish_position_hold(self, reason):
        if self.active_goal is None:
            return False
        self.active_goal.header.stamp = rospy.Time.now()
        self.motion_goal_pub.publish(self.active_goal)
        yaw = self.yaw_from_pose(self.active_goal.pose)
        rospy.loginfo_throttle(
            1.0,
            (
                "%s：持续发布运动目标=(%.3f,%.3f,%.3f)，yaw=%.1fdeg，阶段=%s"
            ),
            NODE_NAME,
            self.active_goal.pose.position.x,
            self.active_goal.pose.position.y,
            self.active_goal.pose.position.z,
            math.degrees(yaw) if yaw is not None else float("nan"),
            reason,
        )
        return True

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
            return False, "当前状态={}，等待HOVER".format(state.state)
        if not state.goal_active:
            return False, "motion状态尚无活动目标"

        expected_yaw = self.yaw_from_pose(expected_goal.pose)
        state_goal_yaw = self.yaw_from_pose(state.goal.pose)
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
        return True, "HOVER且位置、航向和速度均达到阈值"

    def motion_state_debug_text(self):
        with self.motion_state_lock:
            state = self.latest_motion_state
            received_at = self.latest_motion_state_wall_time
        if state is None or received_at is None:
            return "motion状态=未收到"
        return (
            "state={}，位置误差={:.3f}m，航向误差={:.1f}deg，"
            "水平速度={:.3f}m/s，yaw_rate={:.3f}rad/s，消息年龄={:.2f}s"
        ).format(
            state.state,
            state.base_position_error,
            math.degrees(state.yaw_error),
            state.horizontal_speed,
            state.yaw_rate,
            max(0.0, time.monotonic() - received_at),
        )

    def wait_for_goal(self, goal, timeout, stable_seconds, reason):
        self.active_goal = goal
        started_at = time.monotonic()
        stable_started_at = None
        while not rospy.is_shutdown():
            now = time.monotonic()
            elapsed = now - started_at
            if elapsed >= timeout:
                rospy.logerr(
                    "%s：%s超过%.1fs仍未稳定到达；%s",
                    NODE_NAME,
                    reason,
                    timeout,
                    self.motion_state_debug_text(),
                )
                return False
            if not self.publish_position_hold(reason):
                return False
            arrived, detail = self.goal_arrival_status(goal, started_at)
            if arrived:
                if stable_started_at is None:
                    stable_started_at = now
                    rospy.loginfo("%s：%s首次达到阈值，开始稳定计时", NODE_NAME, reason)
                stable_elapsed = now - stable_started_at
                rospy.loginfo_throttle(
                    1.0,
                    "%s：%s稳定确认%.1f/%.1fs：%s；%s",
                    NODE_NAME,
                    reason,
                    min(stable_elapsed, stable_seconds),
                    stable_seconds,
                    detail,
                    self.motion_state_debug_text(),
                )
                if stable_elapsed >= stable_seconds:
                    return True
            else:
                if stable_started_at is not None:
                    rospy.logwarn("%s：%s稳定计时中断：%s", NODE_NAME, reason, detail)
                stable_started_at = None
                rospy.loginfo_throttle(
                    1.0,
                    "%s：等待%s，剩余%.1fs：%s；%s",
                    NODE_NAME,
                    reason,
                    max(0.0, timeout - elapsed),
                    detail,
                    self.motion_state_debug_text(),
                )
            self.rate.sleep()
        return False

    def hold_position_for(self, seconds, reason):
        return self.wait_for_goal(
            self.hold_goal,
            self.initial_hover_timeout,
            seconds,
            reason,
        )

    def make_rotation_goal(self):
        start_yaw = self.yaw_from_pose(self.hold_goal.pose)
        if start_yaw is None:
            return None
        signed_angle = self.turn_angle_deg
        if self.turn_direction == "left":
            signed_angle = -signed_angle
        target_yaw = normalize_angle(start_yaw + math.radians(signed_angle))
        goal = PoseStamped()
        goal.header.frame_id = self.hold_goal.header.frame_id
        goal.pose.position.x = self.hold_goal.pose.position.x
        goal.pose.position.y = self.hold_goal.pose.position.y
        goal.pose.position.z = self.hold_goal.pose.position.z
        quaternion = quaternion_from_euler(0.0, 0.0, target_yaw)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]
        rospy.loginfo(
            (
                "%s：[转向目标] %s%.1f度，保持位置=(%.3f,%.3f,%.3f)，"
                "航向=%.1fdeg -> %.1fdeg"
            ),
            NODE_NAME,
            "右转" if self.turn_direction == "right" else "左转",
            self.turn_angle_deg,
            goal.pose.position.x,
            goal.pose.position.y,
            goal.pose.position.z,
            math.degrees(start_yaw),
            math.degrees(target_yaw),
        )
        return goal

    def hold_active_goal_for(self, seconds, reason):
        started_at = time.monotonic()
        while not rospy.is_shutdown() and time.monotonic() - started_at < seconds:
            self.publish_position_hold(reason)
            rospy.loginfo_throttle(
                1.0,
                "%s：%s %.1f/%.1fs",
                NODE_NAME,
                reason,
                min(time.monotonic() - started_at, seconds),
                seconds,
            )
            self.rate.sleep()
        return not rospy.is_shutdown()

    @staticmethod
    def window_text(window_values):
        return "[{}]".format(
            ",".join("-" if value is None else str(value) for value in window_values)
        )

    @staticmethod
    def count_text(counts):
        if not counts:
            return "无有效ID"
        return ",".join(
            "ID{}:{}帧".format(marker_id, counts[marker_id])
            for marker_id in sorted(counts)
        )

    def record_recognition_frame(self, marker_id, frame_index, detail):
        with self.recognition_lock:
            if self.confirmed_marker_id is not None:
                return

            self.recognition_window.append(marker_id)
            window_values = list(self.recognition_window)
            counts = Counter(
                value for value in window_values if value is not None
            )
            winners = [
                value
                for value, count in counts.items()
                if count >= self.required_match_count
            ]
            if winners:
                # 优先选择出现次数更多的ID；次数相同则选择数值较小的ID。
                self.confirmed_marker_id = min(
                    winners,
                    key=lambda value: (-counts[value], value),
                )

            confirmed_marker_id = self.confirmed_marker_id
            window_snapshot = self.window_text(window_values)
            count_snapshot = self.count_text(counts)
            window_frame_count = len(window_values)
            current_count = counts.get(marker_id, 0) if marker_id is not None else 0
            best_marker_id = None
            best_count = 0
            if counts:
                best_marker_id = min(
                    counts,
                    key=lambda value: (-counts[value], value),
                )
                best_count = counts[best_marker_id]
            confirmed_count = (
                counts.get(confirmed_marker_id, 0)
                if confirmed_marker_id is not None
                else 0
            )

        if marker_id is None:
            if best_marker_id is None:
                rospy.loginfo(
                    (
                        "%s：[识别第%d帧] 本帧未识别到有效ID，原因=%s；"
                        "最近%d帧中识别到0帧（0/%d）；队列=%s"
                    ),
                    NODE_NAME,
                    frame_index,
                    detail,
                    window_frame_count,
                    window_frame_count,
                    window_snapshot,
                )
            else:
                rospy.loginfo(
                    (
                        "%s：[识别第%d帧] 本帧未识别到有效ID，原因=%s；"
                        "当前最多的是ID=%d，最近%d帧中识别到%d帧（%d/%d），"
                        "确认进度=%d/%d；队列=%s，统计=%s"
                    ),
                    NODE_NAME,
                    frame_index,
                    detail,
                    best_marker_id,
                    window_frame_count,
                    best_count,
                    best_count,
                    window_frame_count,
                    best_count,
                    self.required_match_count,
                    window_snapshot,
                    count_snapshot,
                )
        else:
            rospy.loginfo(
                (
                    "%s：[识别第%d帧] 本帧识别到有效ID=%d；"
                    "最近%d帧中该ID识别到%d帧（%d/%d），确认进度=%d/%d；"
                    "队列=%s，%s"
                ),
                NODE_NAME,
                frame_index,
                marker_id,
                window_frame_count,
                current_count,
                current_count,
                window_frame_count,
                current_count,
                self.required_match_count,
                window_snapshot,
                detail,
            )

        if confirmed_marker_id is not None:
            rospy.loginfo(
                (
                    "%s：识别成功：ArUco ID=%d，最近%d帧中识别到%d帧"
                    "（%d/%d），已达到确认要求%d帧，不再等待窗口填满"
                ),
                NODE_NAME,
                confirmed_marker_id,
                window_frame_count,
                confirmed_count,
                confirmed_count,
                window_frame_count,
                self.required_match_count,
            )

    def aruco_callback(self, message):
        self.model_message_index += 1

        if not self.accept_detections:
            rospy.loginfo_throttle(
                1.0,
                "%s：[模型消息 #%d] 当前尚未进入60秒识别阶段，本帧不入队",
                NODE_NAME,
                self.model_message_index,
            )
            return

        with self.recognition_lock:
            self.recognition_frame_index += 1
            frame_index = self.recognition_frame_index

        try:
            confidence = float(message.conf)
        except (TypeError, ValueError):
            confidence = float("nan")
        detection_type = str(message.type).strip().lower()
        marker_id = self.marker_id_from_detection(message)
        detail = "conf={:.3f}, type={!r}, class_name={!r}".format(
            confidence,
            message.type,
            message.class_name,
        )

        if detection_type == "aruco_not_detected" or marker_id == -1:
            self.record_recognition_frame(
                None, frame_index, "未检测到ArUco；{}".format(detail)
            )
            return
        if not math.isfinite(confidence) or confidence < self.min_confidence:
            self.record_recognition_frame(
                None, frame_index, "置信度不足；{}".format(detail)
            )
            return
        if marker_id is None:
            self.record_recognition_frame(
                None, frame_index, "无法解析ID；{}".format(detail)
            )
            return
        if marker_id not in self.COLOR_BY_MARKER:
            self.record_recognition_frame(
                None,
                frame_index,
                "ID={}不在任务范围1~6；{}".format(marker_id, detail),
            )
            return

        self.record_recognition_frame(marker_id, frame_index, detail)

    def reset_recognition(self):
        with self.recognition_lock:
            self.recognition_window.clear()
            self.confirmed_marker_id = None
            self.recognition_frame_index = 0
        self.mock_index = 0

    def get_confirmed_marker_id(self):
        with self.recognition_lock:
            return self.confirmed_marker_id

    def feed_next_mock_frame(self):
        if self.mock_index >= len(self.mock_aruco_ids):
            return False

        raw_marker_id = int(self.mock_aruco_ids[self.mock_index])
        self.mock_index += 1
        self.model_message_index += 1
        with self.recognition_lock:
            self.recognition_frame_index += 1
            frame_index = self.recognition_frame_index
        marker_id = raw_marker_id if raw_marker_id in self.COLOR_BY_MARKER else None
        detail = "mock序号={}/{}，原始值={}".format(
            self.mock_index,
            len(self.mock_aruco_ids),
            raw_marker_id,
        )
        self.record_recognition_frame(
            marker_id,
            frame_index,
            detail,
        )
        return True

    def wait_for_recognition(self):
        self.reset_recognition()
        self.accept_detections = self.input_mode == "topic"
        start_time = rospy.Time.now()
        next_mock_time = start_time

        rospy.loginfo(
            (
                "%s：正式开始识别，计时%.1fs；窗口大小=%d，"
                "同一ID命中%d帧立即成功"
            ),
            NODE_NAME,
            self.recognition_timeout,
            self.recognition_window_size,
            self.required_match_count,
        )

        while not rospy.is_shutdown():
            self.publish_position_hold("60秒ArUco识别阶段")
            now = rospy.Time.now()

            if self.input_mode == "mock" and now >= next_mock_time:
                self.feed_next_mock_frame()
                next_mock_time = now + rospy.Duration(self.mock_frame_interval)

            marker_id = self.get_confirmed_marker_id()
            if marker_id is not None:
                self.accept_detections = False
                return marker_id

            elapsed = (now - start_time).to_sec()
            if elapsed >= self.recognition_timeout:
                self.accept_detections = False
                return None

            with self.recognition_lock:
                window_snapshot = self.window_text(list(self.recognition_window))
            rospy.loginfo_throttle(
                1.0,
                "%s：识别进行中，剩余 %.1fs，当前窗口=%s",
                NODE_NAME,
                max(0.0, self.recognition_timeout - elapsed),
                window_snapshot,
            )
            self.rate.sleep()

        return None

    def publish_lights(self, color):
        red, yellow, green = self.ACTUATOR_LIGHTS[color]
        message = ActuatorControl()
        if not hasattr(message, "mode"):
            rospy.logerr_throttle(
                5.0,
                "%s：ActuatorControl 缺少 mode 字段，灯光指令未发送",
                NODE_NAME,
            )
            return False

        message.mode = self.actuator_mode
        message.light1 = self.light1
        message.light2 = self.light2
        message.heading_servo = self.heading_servo
        message.clamp_servo = self.clamp_servo
        message.drive_cmd = self.drive_cmd
        message.drive_speed = self.drive_speed
        message.red_light = red
        message.yellow_light = yellow
        message.green_light = green
        self.actuator_pub.publish(message)
        return True

    def hold_color(self, color, seconds):
        red, yellow, green = self.ACTUATOR_LIGHTS[color]
        start_time = rospy.Time.now()
        rospy.loginfo(
            "%s：开始灯光阶段：颜色=%s，持续%.1fs",
            NODE_NAME,
            color,
            seconds,
        )
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed >= seconds:
                return True
            self.publish_position_hold("灯光阶段继续保持识别固定点")
            self.publish_lights(color)
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：灯光执行中：颜色=%s，三色指示=(红%d,黄%d,绿%d)，"
                    "剩余%.1fs"
                ),
                NODE_NAME,
                color,
                red,
                yellow,
                green,
                max(0.0, seconds - elapsed),
            )
            self.rate.sleep()
        return False

    def finalize_task(self, success, detail):
        self.accept_detections = False
        self.publish_lights("off")
        if success:
            self.publish_position_hold("任务结束前保持最终运动目标")
        else:
            self.motion_cancel_pub.publish(Empty())
            rospy.logwarn("%s：任务失败，已发布motion cancel要求主动刹停", NODE_NAME)
        state = "finished" if success else "failed"
        self.finished_pub.publish(String(
            data="{} {}: {}".format(NODE_NAME, state, detail)
        ))
        self.outputs_closed = True
        rospy.loginfo(
            "%s：任务%s：%s；灯光已关闭，节点即将退出",
            NODE_NAME,
            "成功" if success else "失败",
            detail,
        )

    def on_shutdown(self):
        self.accept_detections = False
        if self.outputs_closed:
            return
        if hasattr(self, "actuator_pub") and self.actuator_mode_supported:
            self.publish_lights("off")
        if hasattr(self, "motion_cancel_pub"):
            self.motion_cancel_pub.publish(Empty())
            rospy.logwarn("%s：节点中途退出，已发布motion cancel", NODE_NAME)
        self.outputs_closed = True

    def run(self):
        rospy.sleep(0.5)
        if not self.actuator_mode_supported:
            reason = "ActuatorControl 缺少 mode 字段，请重新编译最新消息定义"
            self.finalize_task(False, reason)
            rospy.signal_shutdown(reason)
            return

        if not self.capture_hold_pose():
            reason = "无法记录启动定点，请检查 map -> base_link TF"
            self.finalize_task(False, reason)
            rospy.signal_shutdown(reason)
            return

        rospy.loginfo(
            (
                "%s：先使用motion_supervisor锁定启动点并稳定悬停%.1fs，"
                "期间模型消息不入识别窗口"
            ),
            NODE_NAME,
            self.initial_hover_seconds,
        )
        if not self.hold_position_for(
            self.initial_hover_seconds,
            "识别前定点悬停",
        ):
            reason = "识别前固定点未能在规定时间稳定"
            self.finalize_task(False, reason)
            rospy.signal_shutdown(reason)
            return

        marker_id = self.wait_for_recognition()
        if marker_id is None:
            reason = "识别时间超过{:.1f}s，未满足最近{}帧内同ID达到{}帧".format(
                self.recognition_timeout,
                self.recognition_window_size,
                self.required_match_count,
            )
            self.finalize_task(False, reason)
            rospy.signal_shutdown(reason)
            return

        color = self.color_for_marker(marker_id)
        rospy.loginfo(
            "%s：ArUco ID=%d 确认成功，对应颜色=%s，开始亮灯",
            NODE_NAME,
            marker_id,
            color,
        )
        if not self.hold_color(color, self.light_seconds):
            reason = "亮灯阶段被中止"
            self.finalize_task(False, reason)
            rospy.signal_shutdown(reason)
            return
        if not self.hold_color("off", self.gap_seconds):
            reason = "灭灯间隔阶段被中止"
            self.finalize_task(False, reason)
            rospy.signal_shutdown(reason)
            return

        turn_text = "未启用转向"
        if self.turn_enabled:
            rotation_goal = self.make_rotation_goal()
            if rotation_goal is None:
                reason = "无法生成ArUco识别后的转向目标"
                self.finalize_task(False, reason)
                rospy.signal_shutdown(reason)
                return
            direction_text = (
                "右转" if self.turn_direction == "right" else "左转")
            rospy.loginfo(
                "%s：灯光阶段完成，开始原地%s%.1f度",
                NODE_NAME,
                direction_text,
                self.turn_angle_deg,
            )
            if not self.wait_for_goal(
                    rotation_goal,
                    self.turn_timeout,
                    self.turn_stable_seconds,
                    "ArUco识别后原地{}{}".format(
                        direction_text, self.turn_angle_deg),
            ):
                reason = "ArUco识别成功，但原地{}{}度未稳定到达".format(
                    direction_text, self.turn_angle_deg)
                self.finalize_task(False, reason)
                rospy.signal_shutdown(reason)
                return
            if not self.hold_active_goal_for(
                    self.turn_hold_seconds, "转向完成后的定点保持"):
                reason = "转向完成后的定点保持被中止"
                self.finalize_task(False, reason)
                rospy.signal_shutdown(reason)
                return
            turn_text = "{}{:.1f}度".format(
                direction_text, self.turn_angle_deg)
            rospy.loginfo(
                "%s：转向成功：%s，位置和深度保持不变",
                NODE_NAME,
                turn_text,
            )
        else:
            rospy.logwarn("%s：turn_enabled=false，本次识别亮灯后不执行转向", NODE_NAME)

        detail = "ArUco ID={}，颜色={}，亮灯{:.1f}s，转向={}".format(
            marker_id,
            color,
            self.light_seconds,
            turn_text,
        )
        self.finalize_task(True, detail)
        rospy.signal_shutdown("{} complete".format(NODE_NAME))


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3GetTaskTest().run()
    except rospy.ROSInterruptException:
        pass
