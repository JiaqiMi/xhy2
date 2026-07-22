#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
任务 3 子任务 3：识别指定颜色方框并执行投放动作。

节点支持两种操作模式：
  1. manual：人工控制机器人到方框上方，本节点只识别并执行灯光、夹爪动作；
  2. auto：向 motion_supervisor 发布绝对位置目标，自动搜索并根据方框中心像素对齐。

两种模式共用颜色过滤、滑动窗口候选组判断和执行器动作流程。
"""

import copy
import json
import math
import statistics

import rospy
import tf
from auv_control.msg import AUVData, ActuatorControl, MotionState
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task3_3_inspect_and_drop"
MODE_POSITION = 4


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def yaw_from_quaternion(quaternion):
    return euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ])[2]


def normalize_angle_rad(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# 模型识别参数。
DEFAULT_RATE = 10.0
DEFAULT_DETECTION_TOPIC = "/vision/rectangle/detections"
DEFAULT_TARGET_COLOR = "yellow"
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_STABLE_DETECTION_COUNT = 5
DEFAULT_AUTO_SEARCH_STABLE_DETECTION_COUNT = 3
DEFAULT_AUTO_CENTER_STABLE_DETECTION_COUNT = 5
DEFAULT_STABLE_DETECTION_WINDOW_SIZE = 10
DEFAULT_AUTO_HOVER_CONFIRM_SETTLE_SECONDS = 0.5
DEFAULT_AUTO_HOVER_CONFIRM_TIMEOUT = 4.0
DEFAULT_STABLE_CENTER_TOLERANCE_PX = 40.0
DEFAULT_STABLE_AREA_TOLERANCE_RATIO = 0.35
DEFAULT_DETECTION_TIMEOUT = 2.0
DEFAULT_MAX_WAIT_SECONDS = 300.0

# 操作模式和 motion_supervisor 接口参数。
DEFAULT_OPERATION_MODE = "manual"
DEFAULT_MOTION_GOAL_TOPIC = "/cmd/motion/goal"
DEFAULT_MOTION_CANCEL_TOPIC = "/cmd/motion/cancel"
DEFAULT_MOTION_STATE_TOPIC = "/motion/state"
DEFAULT_STATUS_TOPIC = "/status/auv"
DEFAULT_MOTION_STATE_TIMEOUT = 0.5
DEFAULT_MOTION_STARTUP_TIMEOUT = 10.0
DEFAULT_CANCEL_TIMEOUT = 15.0
DEFAULT_STATUS_TIMEOUT = 0.5
DEFAULT_STATUS_LINEAR_VELOCITY_SCALE = 1.0
DEFAULT_GOAL_MATCH_POSITION_TOLERANCE = 0.03
DEFAULT_GOAL_MATCH_DEPTH_TOLERANCE = 0.03
DEFAULT_GOAL_MATCH_YAW_TOLERANCE_DEG = 2.0
DEFAULT_ARRIVAL_POSITION_TOLERANCE = 0.05
DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG = 5.0
DEFAULT_ARRIVAL_MAX_HORIZONTAL_SPEED = 0.02
DEFAULT_ARRIVAL_MAX_YAW_RATE_DEG_S = 0.5
DEFAULT_AUTO_INITIAL_HOVER_SECONDS = 10.0
DEFAULT_AUTO_SEARCH_FIRST_FORWARD_DISTANCE = 0.30
DEFAULT_AUTO_SEARCH_SECOND_FORWARD_DISTANCE = 0.20
DEFAULT_AUTO_SEARCH_THIRD_FORWARD_DISTANCE = 0.10
DEFAULT_AUTO_FIRST_SEGMENT_ZIGZAG_ENABLED = False
DEFAULT_AUTO_SEARCH_LEFT_DISTANCE = 0.20
DEFAULT_AUTO_SEARCH_RIGHT_DISTANCE = 0.40
DEFAULT_AUTO_VISUAL_FORWARD_GAIN_M = 0.10
DEFAULT_AUTO_VISUAL_LATERAL_GAIN_M = 0.10
DEFAULT_AUTO_VISUAL_MIN_STEP_M = 0.005
DEFAULT_AUTO_VISUAL_MAX_STEP_M = 0.03
DEFAULT_AUTO_VISUAL_GOAL_MIN_INTERVAL = 0.50
DEFAULT_AUTO_FORWARD_SIGN = 1.0
DEFAULT_AUTO_LATERAL_SIGN = 1.0
DEFAULT_AUTO_ACTION_MAX_HORIZONTAL_SPEED = 0.03
DEFAULT_AUTO_ACTION_MAX_VERTICAL_SPEED = 0.03
DEFAULT_AUTO_ACTION_MAX_YAW_RATE = 0.05
DEFAULT_AUTO_ACTION_MAX_DEPTH_ERROR = 0.08
DEFAULT_AUTO_ACTION_MAX_YAW_ERROR_DEG = 5.0
DEFAULT_AUTO_TARGET_CENTER_U_RATIO = 0.5
DEFAULT_AUTO_TARGET_CENTER_V_RATIO = 0.5
DEFAULT_AUTO_CENTER_TOLERANCE_U_PX = 35.0
DEFAULT_AUTO_CENTER_TOLERANCE_V_PX = 35.0
DEFAULT_AUTO_IMAGE_WIDTH = 640.0
DEFAULT_AUTO_IMAGE_HEIGHT = 480.0
DEFAULT_LOG_INTERVAL = 1.0
DEFAULT_WARNING_LOG_INTERVAL = 2.0

# 识别成功后的动作参数。
DEFAULT_HOLD_SECONDS = 1.0
DEFAULT_OPEN_SECONDS = 3.0
DEFAULT_CLOSE_SECONDS = 0.0

# 执行器参数。
DEFAULT_ACTUATOR_TOPIC = "/cmd/actuator"
DEFAULT_ACTUATOR_MODE = 2  # 0=不响应，1=仅补光灯，2=仅执行器
DEFAULT_CLAMP_OPEN = 0x00
DEFAULT_CLAMP_CLOSED = 0xFF
DEFAULT_HEADING_SERVO = 0x80
DEFAULT_DRIVE_CMD = 0
DEFAULT_DRIVE_SPEED = 0
DEFAULT_LIGHT1 = 0
DEFAULT_LIGHT2 = 0


class Task3InspectAndDropTest:
    WAIT_FOR_TARGET = 0
    AUTO_HOVER_CONFIRM = 1
    AUTO_APPROACH = 2
    HOLD_BEFORE_ACTION = 3
    OPEN_CLAMP = 4
    CLOSE_CLAMP = 5

    STATE_NAMES = {
        WAIT_FOR_TARGET: "等待目标颜色方框",
        AUTO_HOVER_CONFIRM: "刹停后悬停复核方框",
        AUTO_APPROACH: "自动靠近并对齐方框",
        HOLD_BEFORE_ACTION: "识别确认后保持",
        OPEN_CLAMP: "打开夹爪",
        CLOSE_CLAMP: "关闭夹爪",
    }

    SEARCH_STEP_NAMES = {
        "hover": "启动悬停",
        "forward": "向前移动",
        "left": "向左横移",
        "right": "向右横移",
        "zigzag_left": "之字形向前并左移",
        "zigzag_right": "之字形向前并右移",
        "zigzag_center": "之字形向前并回中线",
    }

    MOTION_STATE_NAMES = {
        MotionState.IDLE: "IDLE",
        MotionState.ALIGN_PATH: "ALIGN_PATH",
        MotionState.ALIGN_PATH_BRAKE: "ALIGN_PATH_BRAKE",
        MotionState.TRANSLATE: "TRANSLATE",
        MotionState.TRANSLATE_BRAKE: "TRANSLATE_BRAKE",
        MotionState.ALIGN_FINAL: "ALIGN_FINAL",
        MotionState.FINAL_BRAKE: "FINAL_BRAKE",
        MotionState.CAPTURE: "CAPTURE",
        MotionState.HOVER: "HOVER",
        MotionState.SAFE: "SAFE",
    }

    COLOR_LIGHTS = {
        "yellow": (0, 1, 0),
        "green": (0, 0, 1),
        "red": (1, 0, 0),
        "off": (0, 0, 0),
    }

    def __init__(self):
        self.rate_hz = float(rospy.get_param("~rate", DEFAULT_RATE))
        if self.rate_hz <= 0.0:
            raise ValueError("rate 必须大于 0")
        self.rate = rospy.Rate(self.rate_hz)
        self.operation_mode = str(
            rospy.get_param("~operation_mode", DEFAULT_OPERATION_MODE)
        ).strip().lower()
        self.auto_enabled = self.operation_mode == "auto"

        # 旧 launch 的 detection_topic 指向三维 TargetDetection；当前人工测试
        # 直接读取 YOLO 全候选 JSON，使用独立参数避免修改团队共享 launch。
        self.detection_topic = str(
            rospy.get_param("~model_detection_topic", DEFAULT_DETECTION_TOPIC)
        ).strip()
        self.target_color = self.normalize_label(
            rospy.get_param("~target_color", DEFAULT_TARGET_COLOR)
        )
        self.min_confidence = float(
            rospy.get_param("~min_confidence", DEFAULT_MIN_CONFIDENCE)
        )
        self.stable_detection_count = int(
            rospy.get_param(
                "~stable_detection_count", DEFAULT_STABLE_DETECTION_COUNT
            )
        )
        self.auto_search_stable_detection_count = int(rospy.get_param(
            "~auto_search_stable_detection_count",
            DEFAULT_AUTO_SEARCH_STABLE_DETECTION_COUNT,
        ))
        self.auto_center_stable_detection_count = int(rospy.get_param(
            "~auto_center_stable_detection_count",
            DEFAULT_AUTO_CENTER_STABLE_DETECTION_COUNT,
        ))
        self.stable_detection_window_size = int(rospy.get_param(
            "~stable_detection_window_size",
            DEFAULT_STABLE_DETECTION_WINDOW_SIZE,
        ))
        self.auto_hover_confirm_settle_seconds = float(rospy.get_param(
            "~auto_hover_confirm_settle_seconds",
            DEFAULT_AUTO_HOVER_CONFIRM_SETTLE_SECONDS,
        ))
        self.auto_hover_confirm_timeout = float(rospy.get_param(
            "~auto_hover_confirm_timeout",
            DEFAULT_AUTO_HOVER_CONFIRM_TIMEOUT,
        ))
        self.stable_center_tolerance_px = float(
            rospy.get_param(
                "~stable_center_tolerance_px",
                DEFAULT_STABLE_CENTER_TOLERANCE_PX,
            )
        )
        self.stable_area_tolerance_ratio = float(
            rospy.get_param(
                "~stable_area_tolerance_ratio",
                DEFAULT_STABLE_AREA_TOLERANCE_RATIO,
            )
        )
        self.detection_timeout = float(
            rospy.get_param("~detection_timeout", DEFAULT_DETECTION_TIMEOUT)
        )
        self.max_wait_seconds = float(
            rospy.get_param("~max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS)
        )
        self.hold_seconds = float(
            rospy.get_param("~hold_seconds", DEFAULT_HOLD_SECONDS)
        )
        self.open_seconds = float(
            rospy.get_param("~open_seconds", DEFAULT_OPEN_SECONDS)
        )
        self.close_seconds = float(
            rospy.get_param("~close_seconds", DEFAULT_CLOSE_SECONDS)
        )

        self.motion_goal_topic = str(rospy.get_param(
            "~motion_goal_topic", DEFAULT_MOTION_GOAL_TOPIC
        )).strip()
        self.motion_cancel_topic = str(rospy.get_param(
            "~motion_cancel_topic", DEFAULT_MOTION_CANCEL_TOPIC
        )).strip()
        self.motion_state_topic = str(rospy.get_param(
            "~motion_state_topic", DEFAULT_MOTION_STATE_TOPIC
        )).strip()
        self.status_topic = str(
            rospy.get_param("~status_topic", DEFAULT_STATUS_TOPIC)
        ).strip()
        self.motion_state_timeout = float(rospy.get_param(
            "~motion_state_timeout", DEFAULT_MOTION_STATE_TIMEOUT
        ))
        self.motion_startup_timeout = float(rospy.get_param(
            "~motion_startup_timeout", DEFAULT_MOTION_STARTUP_TIMEOUT
        ))
        self.cancel_timeout = float(rospy.get_param(
            "~cancel_timeout", DEFAULT_CANCEL_TIMEOUT
        ))
        self.status_timeout = float(rospy.get_param(
            "~status_timeout", DEFAULT_STATUS_TIMEOUT
        ))
        self.status_linear_velocity_scale = float(rospy.get_param(
            "~status_linear_velocity_scale",
            DEFAULT_STATUS_LINEAR_VELOCITY_SCALE,
        ))
        self.goal_match_position_tolerance = float(rospy.get_param(
            "~goal_match_position_tolerance",
            DEFAULT_GOAL_MATCH_POSITION_TOLERANCE,
        ))
        self.goal_match_depth_tolerance = float(rospy.get_param(
            "~goal_match_depth_tolerance",
            DEFAULT_GOAL_MATCH_DEPTH_TOLERANCE,
        ))
        self.goal_match_yaw_tolerance_deg = float(rospy.get_param(
            "~goal_match_yaw_tolerance_deg",
            DEFAULT_GOAL_MATCH_YAW_TOLERANCE_DEG,
        ))
        self.arrival_position_tolerance = float(rospy.get_param(
            "~arrival_position_tolerance",
            DEFAULT_ARRIVAL_POSITION_TOLERANCE,
        ))
        self.arrival_yaw_tolerance_deg = float(rospy.get_param(
            "~arrival_yaw_tolerance_deg",
            DEFAULT_ARRIVAL_YAW_TOLERANCE_DEG,
        ))
        self.arrival_max_horizontal_speed = float(rospy.get_param(
            "~arrival_max_horizontal_speed",
            DEFAULT_ARRIVAL_MAX_HORIZONTAL_SPEED,
        ))
        self.arrival_max_yaw_rate_deg_s = float(rospy.get_param(
            "~arrival_max_yaw_rate_deg_s",
            DEFAULT_ARRIVAL_MAX_YAW_RATE_DEG_S,
        ))
        self.auto_initial_hover_seconds = float(rospy.get_param(
            "~auto_initial_hover_seconds", DEFAULT_AUTO_INITIAL_HOVER_SECONDS
        ))
        self.auto_search_first_forward_distance = float(rospy.get_param(
            "~auto_search_first_forward_distance",
            DEFAULT_AUTO_SEARCH_FIRST_FORWARD_DISTANCE,
        ))
        self.auto_search_second_forward_distance = float(rospy.get_param(
            "~auto_search_second_forward_distance",
            DEFAULT_AUTO_SEARCH_SECOND_FORWARD_DISTANCE,
        ))
        self.auto_search_third_forward_distance = float(rospy.get_param(
            "~auto_search_third_forward_distance",
            DEFAULT_AUTO_SEARCH_THIRD_FORWARD_DISTANCE,
        ))
        self.auto_first_segment_zigzag_enabled = bool(rospy.get_param(
            "~auto_first_segment_zigzag_enabled",
            DEFAULT_AUTO_FIRST_SEGMENT_ZIGZAG_ENABLED,
        ))
        self.auto_search_left_distance = float(rospy.get_param(
            "~auto_search_left_distance", DEFAULT_AUTO_SEARCH_LEFT_DISTANCE
        ))
        self.auto_search_right_distance = float(rospy.get_param(
            "~auto_search_right_distance", DEFAULT_AUTO_SEARCH_RIGHT_DISTANCE
        ))
        self.auto_visual_forward_gain_m = float(rospy.get_param(
            "~auto_visual_forward_gain_m", DEFAULT_AUTO_VISUAL_FORWARD_GAIN_M
        ))
        self.auto_visual_lateral_gain_m = float(rospy.get_param(
            "~auto_visual_lateral_gain_m", DEFAULT_AUTO_VISUAL_LATERAL_GAIN_M
        ))
        self.auto_visual_min_step_m = float(rospy.get_param(
            "~auto_visual_min_step_m", DEFAULT_AUTO_VISUAL_MIN_STEP_M
        ))
        self.auto_visual_max_step_m = float(rospy.get_param(
            "~auto_visual_max_step_m", DEFAULT_AUTO_VISUAL_MAX_STEP_M
        ))
        self.auto_visual_goal_min_interval = float(rospy.get_param(
            "~auto_visual_goal_min_interval",
            DEFAULT_AUTO_VISUAL_GOAL_MIN_INTERVAL,
        ))
        self.auto_forward_sign = float(rospy.get_param(
            "~auto_forward_sign", DEFAULT_AUTO_FORWARD_SIGN
        ))
        self.auto_lateral_sign = float(rospy.get_param(
            "~auto_lateral_sign", DEFAULT_AUTO_LATERAL_SIGN
        ))
        self.auto_action_max_horizontal_speed = float(rospy.get_param(
            "~auto_action_max_horizontal_speed",
            DEFAULT_AUTO_ACTION_MAX_HORIZONTAL_SPEED,
        ))
        self.auto_action_max_vertical_speed = float(rospy.get_param(
            "~auto_action_max_vertical_speed",
            DEFAULT_AUTO_ACTION_MAX_VERTICAL_SPEED,
        ))
        self.auto_action_max_yaw_rate = float(rospy.get_param(
            "~auto_action_max_yaw_rate", DEFAULT_AUTO_ACTION_MAX_YAW_RATE
        ))
        self.auto_action_max_depth_error = float(rospy.get_param(
            "~auto_action_max_depth_error", DEFAULT_AUTO_ACTION_MAX_DEPTH_ERROR
        ))
        self.auto_action_max_yaw_error_deg = float(rospy.get_param(
            "~auto_action_max_yaw_error_deg",
            DEFAULT_AUTO_ACTION_MAX_YAW_ERROR_DEG,
        ))
        self.auto_target_center_u_ratio = float(rospy.get_param(
            "~auto_target_center_u_ratio", DEFAULT_AUTO_TARGET_CENTER_U_RATIO
        ))
        self.auto_target_center_v_ratio = float(rospy.get_param(
            "~auto_target_center_v_ratio", DEFAULT_AUTO_TARGET_CENTER_V_RATIO
        ))
        self.auto_center_tolerance_u_px = float(rospy.get_param(
            "~auto_center_tolerance_u_px", DEFAULT_AUTO_CENTER_TOLERANCE_U_PX
        ))
        self.auto_center_tolerance_v_px = float(rospy.get_param(
            "~auto_center_tolerance_v_px", DEFAULT_AUTO_CENTER_TOLERANCE_V_PX
        ))
        self.auto_image_width = float(rospy.get_param(
            "~auto_image_width", DEFAULT_AUTO_IMAGE_WIDTH
        ))
        self.auto_image_height = float(rospy.get_param(
            "~auto_image_height", DEFAULT_AUTO_IMAGE_HEIGHT
        ))
        self.log_interval = float(rospy.get_param(
            "~log_interval", DEFAULT_LOG_INTERVAL
        ))
        self.warning_log_interval = float(rospy.get_param(
            "~warning_log_interval", DEFAULT_WARNING_LOG_INTERVAL
        ))

        self.actuator_topic = str(
            rospy.get_param("~actuator_topic", DEFAULT_ACTUATOR_TOPIC)
        ).strip()
        self.actuator_mode = int(
            rospy.get_param("~actuator_mode", DEFAULT_ACTUATOR_MODE)
        )
        self.clamp_open = int(
            rospy.get_param("~clamp_open", DEFAULT_CLAMP_OPEN)
        )
        self.clamp_closed = int(
            rospy.get_param("~clamp_closed", DEFAULT_CLAMP_CLOSED)
        )
        self.heading_servo = int(
            rospy.get_param("~heading_servo", DEFAULT_HEADING_SERVO)
        )
        self.drive_cmd = int(
            rospy.get_param("~drive_cmd", DEFAULT_DRIVE_CMD)
        )
        self.drive_speed = int(
            rospy.get_param("~drive_speed", DEFAULT_DRIVE_SPEED)
        )
        self.light1 = int(rospy.get_param("~light1", DEFAULT_LIGHT1))
        self.light2 = int(rospy.get_param("~light2", DEFAULT_LIGHT2))

        self.validate_params()

        # mode 字段由传感器协议新增；团队消息定义合并并重新编译后才会存在。
        self.actuator_mode_supported = hasattr(ActuatorControl(), "mode")

        self.finished_pub = rospy.Publisher(
            "/finished", String, queue_size=10
        )
        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.goal_pub = None
        self.cancel_pub = None
        self.tf_listener = None
        if self.auto_enabled:
            self.goal_pub = rospy.Publisher(
                self.motion_goal_topic, PoseStamped, queue_size=1
            )
            self.cancel_pub = rospy.Publisher(
                self.motion_cancel_topic, Empty, queue_size=1
            )
            self.tf_listener = tf.TransformListener()

        self.state = self.WAIT_FOR_TARGET
        self.state_started = rospy.Time.now()
        self.task_started = rospy.Time.now()
        self.last_model_message_time = None
        self.last_target_time = None
        self.detection_frame_window = []
        self.hover_confirmation_ready = False
        self.hover_confirmation_hover_at = None
        self.hover_confirmation_started_at = None
        self.hover_confirmation_resume_goal = None
        self.model_frame_index = 0
        self.current_auto_target = None
        self.auto_hold_z = None
        self.auto_hold_yaw = None
        self.auto_centered_frame_count = 0
        self.active_goal = None
        self.active_goal_reason = ""
        self.latest_motion_state = None
        self.last_motion_state_received = None
        self.last_motion_state_value = None
        self.motion_ready_once = False
        self.motion_cancel_requested_at = None
        self.motion_cancel_reason = ""
        self.auto_search_resume_goal = None
        self.auto_search_paused_for_model = False
        self.last_visual_goal_frame = 0
        self.last_visual_goal_time = None
        self.visual_center_hold_requested = False
        self.visual_stop_locked = False
        if self.auto_first_segment_zigzag_enabled:
            zigzag_forward_step = (
                self.auto_search_first_forward_distance / 3.0
            )
            first_segment_steps = [
                ("zigzag_left", zigzag_forward_step),
                ("zigzag_right", zigzag_forward_step),
                (
                    "zigzag_center",
                    self.auto_search_first_forward_distance
                    - 2.0 * zigzag_forward_step,
                ),
            ]
        else:
            first_segment_steps = [
                ("forward", self.auto_search_first_forward_distance),
                ("left", self.auto_search_left_distance),
                ("right", self.auto_search_right_distance),
            ]
        self.auto_search_plan = [
            ("hover", self.auto_initial_hover_seconds),
        ] + first_segment_steps + [
            ("forward", self.auto_search_second_forward_distance),
            ("left", self.auto_search_left_distance),
            ("right", self.auto_search_right_distance),
            ("forward", self.auto_search_third_forward_distance),
            ("left", self.auto_search_left_distance),
            ("right", self.auto_search_right_distance),
        ]
        self.auto_search_index = 0
        self.auto_search_step_started = None
        self.auto_search_step_goal = None
        self.last_status_time = None
        self.current_status = None
        self.status_hold_depth = None
        self.status_hold_yaw_deg = None
        self.auto_action_hold_position = None
        self.last_actuator_command = None
        self.finished = False

        self.detection_sub = rospy.Subscriber(
            self.detection_topic,
            String,
            self.detection_callback,
            queue_size=10,
        )
        self.status_sub = None
        self.motion_state_sub = None
        if self.auto_enabled:
            self.status_sub = rospy.Subscriber(
                self.status_topic,
                AUVData,
                self.status_callback,
                queue_size=20,
            )
            self.motion_state_sub = rospy.Subscriber(
                self.motion_state_topic,
                MotionState,
                self.motion_state_callback,
                queue_size=20,
            )
        rospy.on_shutdown(self.on_shutdown)

        if self.auto_enabled:
            rospy.loginfo(
                (
                    "%s：启动自动寻找模式，运动目标=%s，取消=%s，反馈=%s；"
                    "底层 mode=4、推力、阻尼和刹车全部由 motion_supervisor 管理"
                ),
                NODE_NAME,
                self.motion_goal_topic,
                self.motion_cancel_topic,
                self.motion_state_topic,
            )
        else:
            rospy.loginfo(
                "%s：启动人工操作模式，只识别和执行动作，不发布机器人运动指令",
                NODE_NAME,
            )
        rospy.loginfo(
            "%s：主循环频率=%.1fHz，总任务超时=%.1fs",
            NODE_NAME,
            self.rate_hz,
            self.max_wait_seconds,
        )
        rospy.loginfo(
            (
                "%s：模型话题=%s，目标颜色=%s，最低置信度=%.2f"
            ),
            NODE_NAME,
            self.detection_topic,
            self.target_color,
            self.min_confidence,
        )
        rospy.loginfo(
            (
                "%s：逐帧候选组：最近%d个模型帧内保留有效检测，"
                "位置误差<=%.1fpx，面积变化比例<=%.2f；"
                "识别超时 %.1fs，总等待上限 %.1fs"
            ),
            NODE_NAME,
            self.stable_detection_window_size,
            self.stable_center_tolerance_px,
            self.stable_area_tolerance_ratio,
            self.detection_timeout,
            self.max_wait_seconds,
        )
        if self.auto_enabled:
            rospy.loginfo(
                (
                    "%s：自动模式帧数门槛：搜索候选组=%d/%d帧，"
                    "悬停复核候选组=%d/%d帧，细对准居中确认=%d帧"
                ),
                NODE_NAME,
                self.auto_search_stable_detection_count,
                self.stable_detection_window_size,
                self.auto_search_stable_detection_count,
                self.stable_detection_window_size,
                self.auto_center_stable_detection_count,
            )
            rospy.loginfo(
                (
                    "%s：悬停复核时序：新HOVER后先稳定等待%.2fs，"
                    "再开始重新识别，最长复核%.1fs；超时后恢复被打断的搜索步骤"
                ),
                NODE_NAME,
                self.auto_hover_confirm_settle_seconds,
                self.auto_hover_confirm_timeout,
            )
            rospy.loginfo(
                (
                    "%s：自动动作时序：居中确认后同时开灯和打开夹爪，"
                    "悬停=%.1fs，然后关闭夹爪、熄灯，结束前确认=%.1fs"
                ),
                NODE_NAME,
                self.open_seconds,
                self.close_seconds,
            )
            rospy.loginfo(
                (
                    "%s：motion_supervisor 判定：状态超时=%.2fs，启动等待=%.1fs，"
                    "取消等待=%.1fs，目标匹配容差=(水平%.3fm,深度%.3fm,航向%.1fdeg)"
                ),
                NODE_NAME,
                self.motion_state_timeout,
                self.motion_startup_timeout,
                self.cancel_timeout,
                self.goal_match_position_tolerance,
                self.goal_match_depth_tolerance,
                self.goal_match_yaw_tolerance_deg,
            )
            rospy.loginfo(
                (
                    "%s：实际到达门槛：base_link误差<=%.3fm，航向误差<=%.1fdeg，"
                    "水平速度<=%.3fm/s，航向角速度<=%.2fdeg/s；"
                    "以上条件全部通过才接受HOVER"
                ),
                NODE_NAME,
                self.arrival_position_tolerance,
                self.arrival_yaw_tolerance_deg,
                self.arrival_max_horizontal_speed,
                self.arrival_max_yaw_rate_deg_s,
            )
            rospy.loginfo(
                (
                    "%s：动作放行附加门槛：底层mode=4，水平速度<=%.3fm/s，"
                    "垂直速度<=%.3fm/s，航向角速度<=%.3frad/s，"
                    "深度误差<=%.3fm，航向误差<=%.1fdeg；"
                    "/status/auv超时=%.2fs，线速度缩放=%.3f"
                ),
                NODE_NAME,
                self.auto_action_max_horizontal_speed,
                self.auto_action_max_vertical_speed,
                self.auto_action_max_yaw_rate,
                self.auto_action_max_depth_error,
                self.auto_action_max_yaw_error_deg,
                self.status_timeout,
                self.status_linear_velocity_scale,
            )
            rospy.loginfo(
                (
                    "%s：视觉位置小步：增益=(前后%.3fm,左右%.3fm)，"
                    "步长范围=[%.3f,%.3f]m，最短更新间隔=%.2fs，"
                    "方向符号=(前后%.0f,左右%.0f)"
                ),
                NODE_NAME,
                self.auto_visual_forward_gain_m,
                self.auto_visual_lateral_gain_m,
                self.auto_visual_min_step_m,
                self.auto_visual_max_step_m,
                self.auto_visual_goal_min_interval,
                self.auto_forward_sign,
                self.auto_lateral_sign,
            )
            rospy.loginfo(
                (
                    "%s：自动对齐：目标中心=(%.2fW, %.2fH)，容差=(%.1fpx, %.1fpx)，"
                    "连续居中确认=%d帧"
                ),
                NODE_NAME,
                self.auto_target_center_u_ratio,
                self.auto_target_center_v_ratio,
                self.auto_center_tolerance_u_px,
                self.auto_center_tolerance_v_px,
                self.auto_center_stable_detection_count,
            )
            rospy.loginfo(
                (
                    "%s：搜索顺序：悬停%.1fs；第一段之字形=%s，"
                    "第一段总前进%.2fm；第二段前进%.2fm后左右搜索；"
                    "第三段前进%.2fm后左右搜索"
                ),
                NODE_NAME,
                self.auto_initial_hover_seconds,
                "开启" if self.auto_first_segment_zigzag_enabled else "关闭",
                self.auto_search_first_forward_distance,
                self.auto_search_second_forward_distance,
                self.auto_search_third_forward_distance,
            )
            rospy.loginfo(
                (
                    "%s：横移距离定义：左移从当前点向左走%.2fm；"
                    "随后右移从左侧位置向右走%.2fm"
                ),
                NODE_NAME,
                self.auto_search_left_distance,
                self.auto_search_right_distance,
            )
        else:
            rospy.loginfo(
                "%s：人工模式连续稳定识别=%d帧，动作前确认=%.1fs",
                NODE_NAME,
                self.stable_detection_count,
                self.hold_seconds,
            )
        rospy.loginfo(
            "%s：执行器话题=%s，mode=%d（2=仅执行器），夹爪开=%d，夹爪关=%d",
            NODE_NAME,
            self.actuator_topic,
            self.actuator_mode,
            self.clamp_open,
            self.clamp_closed,
        )
        rospy.loginfo(
            (
                "%s：执行器固定字段：补光灯=(%d,%d)，航向舵机=%d，"
                "推进电机=(动作%d,转速%d)；颜色灯随目标颜色自动选择"
            ),
            NODE_NAME,
            self.light1,
            self.light2,
            self.heading_servo,
            self.drive_cmd,
            self.drive_speed,
        )
        if not self.actuator_mode_supported:
            rospy.logerr(
                (
                    "%s：当前 auv_control/ActuatorControl 尚无 mode 字段；"
                    "请同步新消息定义并重新 catkin 编译后再执行子任务3"
                ),
                NODE_NAME,
            )

    @staticmethod
    def normalize_label(value):
        text = str(value).strip().lower()
        text = text.replace("-", "_").replace(" ", "_")
        return "_".join(part for part in text.split("_") if part)

    def status_callback(self, message):
        raw_values = (
            message.linear_velocity[0],
            message.linear_velocity[1],
            message.linear_velocity[2],
            message.pose.latitude,
            message.pose.longitude,
            message.pose.depth,
            message.pose.altitude,
            message.pose.roll,
            message.pose.pitch,
            message.pose.yaw,
        )
        if not all(math.isfinite(value) for value in raw_values):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：/status/auv 包含无效位姿或速度，本帧已忽略",
                NODE_NAME,
            )
            return

        self.current_status = {
            "control_mode": int(message.control_mode),
            "vx": float(raw_values[0]) * self.status_linear_velocity_scale,
            "vy": float(raw_values[1]) * self.status_linear_velocity_scale,
            "vz": float(raw_values[2]) * self.status_linear_velocity_scale,
            "latitude": float(raw_values[3]),
            "longitude": float(raw_values[4]),
            "depth": float(raw_values[5]),
            "altitude": float(raw_values[6]),
            "roll_deg": float(raw_values[7]),
            "pitch_deg": float(raw_values[8]),
            "yaw_deg": float(raw_values[9]),
        }
        self.last_status_time = rospy.Time.now()

        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：/status/auv：mode=%d，深度=%.3fm，高度=%.3fm，"
                "航向=%.2fdeg，速度前右下=(%+.3f,%+.3f,%+.3f)m/s"
            ),
            NODE_NAME,
            self.current_status["control_mode"],
            self.current_status["depth"],
            self.current_status["altitude"],
            self.current_status["yaw_deg"],
            self.current_status["vx"],
            self.current_status["vy"],
            self.current_status["vz"],
        )

    def motion_state_callback(self, message):
        self.latest_motion_state = message
        self.last_motion_state_received = rospy.Time.now()
        state_name = self.MOTION_STATE_NAMES.get(
            message.state, "UNKNOWN({})".format(message.state)
        )
        if message.state != self.last_motion_state_value:
            rospy.loginfo(
                "%s：运动状态切换为 %s，原因=%s",
                NODE_NAME,
                state_name,
                message.reason or "无",
            )
            self.last_motion_state_value = message.state
        if message.state != MotionState.SAFE:
            self.motion_ready_once = True
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：运动反馈：state=%s，goal_active=%s，"
                "控制误差=%.3fm，base_link误差=%.3fm，"
                "航向误差=%+.2fdeg，水平速度=%.3fm/s，航向角速度=%+.2fdeg/s，"
                "输出=(TX=%d,TY=%d,MZ=%d)，原因=%s"
            ),
            NODE_NAME,
            state_name,
            str(bool(message.goal_active)),
            message.position_error,
            message.base_position_error,
            math.degrees(message.yaw_error),
            message.horizontal_speed,
            math.degrees(message.yaw_rate),
            message.tx,
            message.ty,
            message.mz,
            message.reason or "无",
        )

    def get_recent_status(self, context):
        if self.current_status is None or self.last_status_time is None:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：等待状态话题 %s，%s暂停",
                NODE_NAME,
                self.status_topic,
                context,
            )
            return None
        age = (rospy.Time.now() - self.last_status_time).to_sec()
        if age > self.status_timeout:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：/status/auv 已超时 %.2fs（限制 %.2fs），%s暂停",
                NODE_NAME,
                age,
                self.status_timeout,
                context,
            )
            return None
        return self.current_status

    @staticmethod
    def angle_difference_deg(angle_a, angle_b):
        return (angle_a - angle_b + 180.0) % 360.0 - 180.0

    def status_pose_errors(self, status):
        if self.status_hold_depth is None or self.status_hold_yaw_deg is None:
            return None
        depth_error = status["depth"] - self.status_hold_depth
        yaw_error_deg = self.angle_difference_deg(
            status["yaw_deg"],
            self.status_hold_yaw_deg,
        )
        return depth_error, yaw_error_deg

    def validate_params(self):
        if self.operation_mode not in ("manual", "auto"):
            raise ValueError("operation_mode 必须是 manual 或 auto")
        if not self.detection_topic:
            raise ValueError("model_detection_topic 不能为空")
        if not self.target_color:
            raise ValueError("target_color 不能为空")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence 必须在 0 到 1 之间")
        if self.stable_detection_count < 1:
            raise ValueError("stable_detection_count 必须大于等于 1")
        if self.stable_detection_window_size < 1:
            raise ValueError("stable_detection_window_size 必须大于等于 1")
        if self.stable_detection_count > self.stable_detection_window_size:
            raise ValueError(
                "stable_detection_count 不能大于 stable_detection_window_size"
            )
        if self.auto_hover_confirm_settle_seconds < 0.0:
            raise ValueError("auto_hover_confirm_settle_seconds 不能小于 0")
        if self.auto_hover_confirm_timeout <= 0.0:
            raise ValueError("auto_hover_confirm_timeout 必须大于 0")
        if self.stable_center_tolerance_px < 0.0:
            raise ValueError("stable_center_tolerance_px 不能小于 0")
        if not 0.0 <= self.stable_area_tolerance_ratio <= 1.0:
            raise ValueError("stable_area_tolerance_ratio 必须在 0 到 1 之间")
        if self.detection_timeout <= 0.0:
            raise ValueError("detection_timeout 必须大于 0")
        if self.max_wait_seconds <= 0.0:
            raise ValueError("max_wait_seconds 必须大于 0")
        if self.actuator_mode not in (0, 1, 2):
            raise ValueError("actuator_mode 必须是 0、1 或 2")
        if self.actuator_mode != 2:
            rospy.logwarn(
                "%s：子任务3需要控制夹爪和三色灯，actuator_mode 应设置为 2",
                NODE_NAME,
            )
        if min(self.hold_seconds, self.open_seconds, self.close_seconds) < 0.0:
            raise ValueError("动作持续时间不能小于 0")
        if self.target_color not in self.COLOR_LIGHTS:
            raise ValueError("target_color 必须是 yellow、green 或 red")
        if min(self.log_interval, self.warning_log_interval) <= 0.0:
            raise ValueError("日志间隔必须大于 0")

        if not self.auto_enabled:
            return
        if self.auto_search_stable_detection_count < 1:
            raise ValueError(
                "auto_search_stable_detection_count 必须大于等于 1"
            )
        if (
            self.auto_search_stable_detection_count
            > self.stable_detection_window_size
        ):
            raise ValueError(
                "auto_search_stable_detection_count 不能大于 "
                "stable_detection_window_size"
            )
        if self.auto_center_stable_detection_count < 1:
            raise ValueError(
                "auto_center_stable_detection_count 必须大于等于 1"
            )
        topics = (
            self.motion_goal_topic,
            self.motion_cancel_topic,
            self.motion_state_topic,
            self.status_topic,
        )
        if not all(topics):
            raise ValueError("motion_supervisor 和状态反馈话题不能为空")
        if min(
            self.motion_state_timeout,
            self.motion_startup_timeout,
            self.cancel_timeout,
            self.status_timeout,
            self.status_linear_velocity_scale,
        ) <= 0.0:
            raise ValueError("运动反馈超时和状态缩放参数必须大于 0")
        if min(
            self.goal_match_position_tolerance,
            self.goal_match_depth_tolerance,
            self.goal_match_yaw_tolerance_deg,
            self.arrival_position_tolerance,
            self.arrival_yaw_tolerance_deg,
            self.arrival_max_horizontal_speed,
            self.arrival_max_yaw_rate_deg_s,
        ) < 0.0:
            raise ValueError("运动目标匹配和实际到达阈值不能小于 0")
        search_distances = (
            self.auto_search_first_forward_distance,
            self.auto_search_second_forward_distance,
            self.auto_search_third_forward_distance,
            self.auto_search_left_distance,
            self.auto_search_right_distance,
        )
        if min(search_distances) <= 0.0:
            raise ValueError("自动搜索的前进和横移距离必须大于 0")
        if self.auto_initial_hover_seconds < 0.0:
            raise ValueError("auto_initial_hover_seconds 不能小于 0")
        if min(
            self.auto_visual_forward_gain_m,
            self.auto_visual_lateral_gain_m,
            self.auto_visual_min_step_m,
            self.auto_visual_max_step_m,
            self.auto_visual_goal_min_interval,
        ) < 0.0:
            raise ValueError("视觉位置小步参数不能小于 0")
        if self.auto_visual_max_step_m <= 0.0:
            raise ValueError("auto_visual_max_step_m 必须大于 0")
        if self.auto_visual_min_step_m > self.auto_visual_max_step_m:
            raise ValueError("auto_visual_min_step_m 不能大于最大步长")
        if self.auto_forward_sign == 0.0 or self.auto_lateral_sign == 0.0:
            raise ValueError("视觉前后和左右方向符号不能为 0")
        if min(
            self.auto_action_max_horizontal_speed,
            self.auto_action_max_vertical_speed,
            self.auto_action_max_yaw_rate,
            self.auto_action_max_depth_error,
            self.auto_action_max_yaw_error_deg,
        ) < 0.0:
            raise ValueError("动作放行速度和位姿阈值不能小于 0")
        if not 0.0 <= self.auto_target_center_u_ratio <= 1.0:
            raise ValueError("auto_target_center_u_ratio 必须在 0 到 1 之间")
        if not 0.0 <= self.auto_target_center_v_ratio <= 1.0:
            raise ValueError("auto_target_center_v_ratio 必须在 0 到 1 之间")
        if min(
            self.auto_center_tolerance_u_px,
            self.auto_center_tolerance_v_px,
        ) < 0.0:
            raise ValueError("自动居中容差不能小于 0")
        if min(self.auto_image_width, self.auto_image_height) <= 0.0:
            raise ValueError("自动控制默认图像尺寸必须大于 0")

    def get_current_pose(self, context="自动控制"):
        if not self.auto_enabled or self.tf_listener is None:
            return None
        try:
            self.tf_listener.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.5)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
        except tf.Exception as exc:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：%s无法读取 map -> base_link：%s",
                NODE_NAME,
                context,
                str(exc),
            )
            return None

        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(value) for value in values):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：%s读取到非有限 TF，本帧忽略",
                NODE_NAME,
                context,
            )
            return None
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def make_goal(self, x_value, y_value, z_value, yaw):
        values = (x_value, y_value, z_value, yaw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("任务生成了非有限运动目标")
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = x_value
        goal.pose.position.y = y_value
        goal.pose.position.z = z_value
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]
        return goal

    def set_active_goal(self, x_value, y_value, z_value, yaw, reason):
        self.active_goal = self.make_goal(
            x_value, y_value, z_value, yaw
        )
        self.active_goal_reason = reason
        rospy.loginfo(
            (
                "%s：设置 motion_supervisor 绝对目标：map=(%.3f,%.3f,%.3f)，"
                "yaw=%.2fdeg，原因=%s"
            ),
            NODE_NAME,
            x_value,
            y_value,
            z_value,
            math.degrees(yaw),
            reason,
        )

    def set_body_offset_goal(self, current, forward, right, reason):
        goal_x = (
            current.pose.position.x
            + math.cos(self.auto_hold_yaw) * forward
            - math.sin(self.auto_hold_yaw) * right
        )
        goal_y = (
            current.pose.position.y
            + math.sin(self.auto_hold_yaw) * forward
            + math.cos(self.auto_hold_yaw) * right
        )
        self.set_active_goal(
            goal_x,
            goal_y,
            self.auto_hold_z,
            self.auto_hold_yaw,
            reason,
        )
        return self.active_goal

    def initialize_auto_pose(self):
        if not self.auto_enabled:
            return True
        if self.auto_hold_z is not None and self.auto_hold_yaw is not None:
            return True

        status = self.get_recent_status("初始化固定悬停点")
        current = self.get_current_pose("初始化固定悬停点")
        if status is None or current is None:
            return False

        self.auto_hold_z = current.pose.position.z
        self.auto_hold_yaw = yaw_from_quaternion(current.pose.orientation)
        self.status_hold_depth = status["depth"]
        self.status_hold_yaw_deg = status["yaw_deg"]
        self.set_active_goal(
            current.pose.position.x,
            current.pose.position.y,
            self.auto_hold_z,
            self.auto_hold_yaw,
            "只锁存一次启动位置，漂移时仍返回该固定悬停点",
        )
        rospy.loginfo(
            (
                "%s：固定悬停点已锁存：map=(%.3f,%.3f,%.3f)，yaw=%.2fdeg，"
                "启动深度=%.3fm；后续不会跟随漂移位置更新"
            ),
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            self.auto_hold_z,
            math.degrees(self.auto_hold_yaw),
            self.status_hold_depth,
        )
        return True

    def publish_active_goal(self):
        if (
            not self.auto_enabled
            or self.goal_pub is None
            or self.active_goal is None
        ):
            return False
        self.active_goal.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.active_goal)
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：持续发布运动目标：map=(%.3f,%.3f,%.3f)，"
                "yaw=%.2fdeg，任务状态=%s，原因=%s"
            ),
            NODE_NAME,
            self.active_goal.pose.position.x,
            self.active_goal.pose.position.y,
            self.active_goal.pose.position.z,
            math.degrees(yaw_from_quaternion(
                self.active_goal.pose.orientation
            )),
            self.STATE_NAMES.get(self.state, "未知状态"),
            self.active_goal_reason,
        )
        return True

    def motion_state_age(self):
        if self.latest_motion_state is None:
            return None
        stamp = self.latest_motion_state.header.stamp
        if stamp == rospy.Time(0):
            return None
        return max(0.0, (rospy.Time.now() - stamp).to_sec())

    def motion_state_is_fresh(self):
        if (
            self.latest_motion_state is None
            or self.last_motion_state_received is None
        ):
            return False
        receipt_age = (
            rospy.Time.now() - self.last_motion_state_received
        ).to_sec()
        stamp_age = self.motion_state_age()
        return (
            receipt_age <= self.motion_state_timeout
            and stamp_age is not None
            and stamp_age <= self.motion_state_timeout
        )

    def goal_match_errors(self):
        if self.active_goal is None or self.latest_motion_state is None:
            return None
        actual_goal = self.latest_motion_state.goal
        if actual_goal.header.frame_id != "map":
            return None
        dx = actual_goal.pose.position.x - self.active_goal.pose.position.x
        dy = actual_goal.pose.position.y - self.active_goal.pose.position.y
        dz = actual_goal.pose.position.z - self.active_goal.pose.position.z
        desired_yaw = yaw_from_quaternion(self.active_goal.pose.orientation)
        actual_yaw = yaw_from_quaternion(actual_goal.pose.orientation)
        yaw_error_deg = abs(math.degrees(normalize_angle_rad(
            actual_yaw - desired_yaw
        )))
        return math.hypot(dx, dy), abs(dz), yaw_error_deg

    def goal_matches_motion_state(self):
        errors = self.goal_match_errors()
        if errors is None:
            return False
        position_error, depth_error, yaw_error_deg = errors
        return (
            position_error <= self.goal_match_position_tolerance
            and depth_error <= self.goal_match_depth_tolerance
            and yaw_error_deg <= self.goal_match_yaw_tolerance_deg
        )

    def actual_arrival_checks(self):
        message = self.latest_motion_state
        if message is None:
            return None
        values = (
            message.base_position_error,
            message.yaw_error,
            message.horizontal_speed,
            message.yaw_rate,
        )
        if not all(math.isfinite(value) for value in values):
            return None
        return {
            "position_error": abs(message.base_position_error),
            "position_ok": (
                abs(message.base_position_error)
                <= self.arrival_position_tolerance
            ),
            "yaw_error_deg": abs(math.degrees(message.yaw_error)),
            "yaw_ok": (
                abs(math.degrees(message.yaw_error))
                <= self.arrival_yaw_tolerance_deg
            ),
            "horizontal_speed": abs(message.horizontal_speed),
            "speed_ok": (
                abs(message.horizontal_speed)
                <= self.arrival_max_horizontal_speed
            ),
            "yaw_rate_deg_s": abs(math.degrees(message.yaw_rate)),
            "yaw_rate_ok": (
                abs(math.degrees(message.yaw_rate))
                <= self.arrival_max_yaw_rate_deg_s
            ),
        }

    def actual_arrival_satisfied(self):
        checks = self.actual_arrival_checks()
        return (
            checks is not None
            and checks["position_ok"]
            and checks["yaw_ok"]
            and checks["speed_ok"]
            and checks["yaw_rate_ok"]
        )

    def motion_arrived(self):
        return (
            self.motion_state_is_fresh()
            and self.latest_motion_state.state == MotionState.HOVER
            and self.goal_matches_motion_state()
            and self.actual_arrival_satisfied()
        )

    def current_motion_state_name(self):
        if self.latest_motion_state is None:
            return "未收到"
        return self.MOTION_STATE_NAMES.get(
            self.latest_motion_state.state,
            "UNKNOWN({})".format(self.latest_motion_state.state),
        )

    def log_arrival_gate(self, context):
        message = self.latest_motion_state
        if message is None:
            return
        fresh = self.motion_state_is_fresh()
        hover = message.state == MotionState.HOVER
        goal_match = self.goal_matches_motion_state()
        goal_errors = self.goal_match_errors()
        actual_checks = self.actual_arrival_checks()
        if goal_errors is None:
            goal_error_text = "未知（反馈goal坐标系={}）".format(
                message.goal.header.frame_id or "空"
            )
        else:
            goal_error_text = (
                "水平{:.3f}/<={:.3f}m，z{:.3f}/<={:.3f}m，"
                "yaw{:.2f}/<={:.2f}deg"
            ).format(
                goal_errors[0],
                self.goal_match_position_tolerance,
                goal_errors[1],
                self.goal_match_depth_tolerance,
                goal_errors[2],
                self.goal_match_yaw_tolerance_deg,
            )
        if actual_checks is None:
            actual_error_text = "未知"
            actual_ok = False
        else:
            actual_ok = self.actual_arrival_satisfied()
            actual_error_text = (
                "位置{:.3f}/<={:.3f}m，航向{:.2f}/<={:.2f}deg，"
                "速度{:.3f}/<={:.3f}m/s，yaw_rate{:.2f}/<={:.2f}deg/s"
            ).format(
                actual_checks["position_error"],
                self.arrival_position_tolerance,
                actual_checks["yaw_error_deg"],
                self.arrival_yaw_tolerance_deg,
                actual_checks["horizontal_speed"],
                self.arrival_max_horizontal_speed,
                actual_checks["yaw_rate_deg_s"],
                self.arrival_max_yaw_rate_deg_s,
            )
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：%s：反馈新鲜[%s]，state=%s/HOVER[%s]，"
                "目标一致[%s]，目标差值=(%s)，实际到达[%s]，"
                "实际门槛=(%s)，输出=(TX=%d,TY=%d,MZ=%d)"
            ),
            NODE_NAME,
            context,
            "通过" if fresh else "未通过",
            self.current_motion_state_name(),
            "通过" if hover else "未通过",
            "通过" if goal_match else "未通过",
            goal_error_text,
            "通过" if actual_ok else "未通过",
            actual_error_text,
            message.tx,
            message.ty,
            message.mz,
        )

    def handle_motion_health(self):
        elapsed = (rospy.Time.now() - self.task_started).to_sec()
        if self.latest_motion_state is None:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：等待运动反馈 %s，已等待 %.1f/%.1fs",
                NODE_NAME,
                self.motion_state_topic,
                elapsed,
                self.motion_startup_timeout,
            )
            if elapsed >= self.motion_startup_timeout:
                self.finish_task(False, "启动后未收到 /motion/state")
            return False
        if not self.motion_state_is_fresh():
            age = self.motion_state_age()
            rospy.logerr_throttle(
                self.warning_log_interval,
                "%s：运动反馈不新鲜，消息年龄=%s，限制=%.2fs",
                NODE_NAME,
                "未知" if age is None else "{:.2f}s".format(age),
                self.motion_state_timeout,
            )
            if self.motion_ready_once or elapsed >= self.motion_startup_timeout:
                self.finish_task(False, "运动状态反馈超时")
            return False
        if self.latest_motion_state.state not in self.MOTION_STATE_NAMES:
            self.finish_task(
                False,
                "motion_supervisor 返回未知状态 {}".format(
                    self.latest_motion_state.state
                ),
            )
            return False
        if self.latest_motion_state.state == MotionState.SAFE:
            rospy.logerr_throttle(
                self.warning_log_interval,
                "%s：motion_supervisor 进入 SAFE，原因=%s",
                NODE_NAME,
                self.latest_motion_state.reason or "未知",
            )
            if self.motion_ready_once or elapsed >= self.motion_startup_timeout:
                self.finish_task(
                    False,
                    "motion_supervisor 进入 SAFE：{}".format(
                        self.latest_motion_state.reason or "未知原因"
                    ),
                )
            return False
        return True

    def request_motion_cancel(self, reason, discard_search_resume=False):
        if not self.auto_enabled or self.cancel_pub is None:
            return False
        if discard_search_resume:
            self.auto_search_resume_goal = None
            self.auto_search_paused_for_model = False
        if self.motion_cancel_requested_at is None:
            self.cancel_pub.publish(Empty())
            self.motion_cancel_requested_at = rospy.Time.now()
            rospy.logwarn(
                "%s：发布 %s，要求 motion_supervisor 主动刹停并进入 HOVER；原因=%s",
                NODE_NAME,
                self.motion_cancel_topic,
                reason,
            )
        self.motion_cancel_reason = reason
        self.active_goal = None
        return True

    def cancel_has_completed(self):
        if (
            self.motion_cancel_requested_at is None
            or not self.motion_state_is_fresh()
            or self.latest_motion_state.state != MotionState.HOVER
        ):
            return False
        return (
            self.latest_motion_state.header.stamp
            >= self.motion_cancel_requested_at
        )

    def wait_for_motion_cancel(self, context):
        if self.motion_cancel_requested_at is None:
            return True
        elapsed = (
            rospy.Time.now() - self.motion_cancel_requested_at
        ).to_sec()
        if elapsed >= self.cancel_timeout:
            self.finish_task(
                False,
                "等待 motion_supervisor 主动刹停超时：{}".format(
                    self.motion_cancel_reason
                ),
            )
            return False
        if not self.cancel_has_completed():
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：%s，等待刹停进入 HOVER %.1f/%.1fs",
                NODE_NAME,
                context,
                elapsed,
                self.cancel_timeout,
            )
            return False

        current = self.get_current_pose("刹停完成后锁定当前位置")
        if current is None:
            return False
        reason = self.motion_cancel_reason
        self.motion_cancel_requested_at = None
        self.motion_cancel_reason = ""
        self.set_active_goal(
            current.pose.position.x,
            current.pose.position.y,
            self.auto_hold_z,
            self.auto_hold_yaw,
            "主动刹停完成后锁定当前位置",
        )
        rospy.loginfo(
            "%s：主动刹停完成并已锁定固定点，原因为：%s",
            NODE_NAME,
            reason,
        )
        return True

    def action_status_is_stable(self, status):
        pose_errors = self.status_pose_errors(status)
        checks = self.actual_arrival_checks()
        if pose_errors is None or checks is None:
            return False
        depth_error, yaw_error_deg = pose_errors
        message = self.latest_motion_state
        return (
            self.motion_arrived()
            and status["control_mode"] == MODE_POSITION
            and message.horizontal_speed
            <= self.auto_action_max_horizontal_speed
            and abs(status["vz"]) <= self.auto_action_max_vertical_speed
            and abs(message.yaw_rate) <= self.auto_action_max_yaw_rate
            and abs(depth_error) <= self.auto_action_max_depth_error
            and abs(yaw_error_deg) <= self.auto_action_max_yaw_error_deg
        )

    def capture_action_hold_position(self):
        if self.active_goal is None:
            current = self.get_current_pose("记录最终动作定点")
            if current is None:
                return False
            self.set_active_goal(
                current.pose.position.x,
                current.pose.position.y,
                self.auto_hold_z,
                self.auto_hold_yaw,
                "记录开灯和夹爪动作期间的固定定点",
            )
        self.auto_action_hold_position = (
            self.active_goal.pose.position.x,
            self.active_goal.pose.position.y,
        )
        rospy.loginfo(
            "%s：最终动作定点已锁定：map=(%.3f,%.3f,%.3f)，后续只重发同一目标",
            NODE_NAME,
            self.auto_action_hold_position[0],
            self.auto_action_hold_position[1],
            self.active_goal.pose.position.z,
        )
        return True

    def publish_action_position_hold(self, reason):
        if self.auto_action_hold_position is None or self.active_goal is None:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：%s时最终动作定点尚未记录",
                NODE_NAME,
                reason,
            )
            return False
        published = self.publish_active_goal()
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：motion_supervisor 最终定点保持：map=(%.3f,%.3f)，阶段=%s",
            NODE_NAME,
            self.auto_action_hold_position[0],
            self.auto_action_hold_position[1],
            reason,
        )
        return published

    def reset_auto_search_step(self):
        self.auto_search_step_started = None
        self.auto_search_step_goal = None
        self.auto_search_resume_goal = None
        self.auto_search_paused_for_model = False

    def search_step_offsets(self, step_kind, step_amount):
        if step_kind == "forward":
            return step_amount, 0.0
        if step_kind == "left":
            return 0.0, -step_amount
        if step_kind == "right":
            return 0.0, step_amount
        if step_kind == "zigzag_left":
            return step_amount, -self.auto_search_left_distance
        if step_kind == "zigzag_right":
            return step_amount, self.auto_search_right_distance
        if step_kind == "zigzag_center":
            return (
                step_amount,
                self.auto_search_left_distance
                - self.auto_search_right_distance,
            )
        return 0.0, 0.0

    def search_step_description(self, step_kind, step_amount):
        if step_kind == "hover":
            return "启动悬停 %.2fs" % step_amount
        forward, right = self.search_step_offsets(step_kind, step_amount)
        return "%s：前后%+.2fm，左右%+.2fm" % (
            self.SEARCH_STEP_NAMES[step_kind],
            forward,
            right,
        )

    def complete_auto_search_step(self, step_kind, step_amount):
        rospy.loginfo(
            "%s：搜索步骤 %d/%d 完成并由 HOVER 确认：%s",
            NODE_NAME,
            self.auto_search_index + 1,
            len(self.auto_search_plan),
            self.search_step_description(step_kind, step_amount),
        )
        if step_kind == "hover":
            self.reset_stability()
            rospy.loginfo(
                "%s：启动悬停结束，已清空悬停期间的识别帧，从搜索移动阶段重新统计",
                NODE_NAME,
            )
        self.auto_search_index += 1
        self.reset_auto_search_step()
        if self.auto_search_index >= len(self.auto_search_plan):
            rospy.logwarn(
                "%s：预设搜索路径已经执行完毕，仍未稳定识别方框，保持最后定点等待",
                NODE_NAME,
            )

    def pause_search_for_model(self):
        if not self.auto_search_paused_for_model:
            if self.auto_search_step_goal is not None:
                self.auto_search_resume_goal = copy.deepcopy(
                    self.auto_search_step_goal
                )
            self.auto_search_paused_for_model = True
            self.request_motion_cancel(
                "模型话题未就绪或已超时，暂停当前搜索位移"
            )
        self.wait_for_motion_cancel("模型不可用，搜索暂停")

    def resume_search_after_model_ready(self):
        if not self.auto_search_paused_for_model:
            return True
        if not self.wait_for_motion_cancel("等待模型恢复前先完成刹停"):
            return False
        if self.auto_search_resume_goal is None:
            self.auto_search_paused_for_model = False
            return True
        self.active_goal = copy.deepcopy(self.auto_search_resume_goal)
        self.active_goal_reason = "模型恢复，继续原搜索目标"
        self.auto_search_paused_for_model = False
        self.auto_search_resume_goal = None
        rospy.loginfo(
            "%s：模型话题恢复，继续当前搜索步骤的原绝对目标",
            NODE_NAME,
        )
        return True

    def search_target_automatically(self, model_ready):
        self.auto_centered_frame_count = 0
        if self.state != self.WAIT_FOR_TARGET:
            return
        if not self.wait_for_motion_cancel("搜索阶段等待主动刹停"):
            return
        if self.auto_search_index >= len(self.auto_search_plan):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：搜索路径已完成，保持最后定点等待 %s 方框，任务总超时 %.1fs",
                NODE_NAME,
                self.target_color,
                self.max_wait_seconds,
            )
            return

        step_kind, step_amount = self.auto_search_plan[self.auto_search_index]
        if step_kind == "hover":
            status = self.get_recent_status("启动固定点悬停")
            if status is None:
                return
            stable = self.action_status_is_stable(status)
            if not stable:
                if self.auto_search_step_started is not None:
                    rospy.logwarn(
                        "%s：启动悬停稳定条件中断，10秒计时重新开始",
                        NODE_NAME,
                    )
                self.auto_search_step_started = None
                checks = self.actual_arrival_checks()
                pose_errors = self.status_pose_errors(status)
                depth_error = 0.0 if pose_errors is None else pose_errors[0]
                yaw_error_deg = 0.0 if pose_errors is None else pose_errors[1]
                rospy.loginfo_throttle(
                    self.log_interval,
                    (
                        "%s：等待 motion_supervisor 在固定启动点进入 HOVER；"
                        "state=%s，base误差=%s，水平速度=%s；"
                        "附加门槛：mode=%d/4，下向速度=%.3f/<=%.3fm/s，"
                        "深度误差=%.3f/<=%.3fm，航向误差=%.2f/<=%.2fdeg"
                    ),
                    NODE_NAME,
                    self.MOTION_STATE_NAMES.get(
                        self.latest_motion_state.state, "未知"
                    ),
                    "未知" if checks is None else "{:.3f}m".format(
                        checks["position_error"]
                    ),
                    "未知" if checks is None else "{:.3f}m/s".format(
                        checks["horizontal_speed"]
                    ),
                    status["control_mode"],
                    abs(status["vz"]),
                    self.auto_action_max_vertical_speed,
                    abs(depth_error),
                    self.auto_action_max_depth_error,
                    abs(yaw_error_deg),
                    self.auto_action_max_yaw_error_deg,
                )
                self.log_arrival_gate("启动固定点悬停到达判定")
                return
            if self.auto_search_step_started is None:
                self.auto_search_step_started = rospy.Time.now()
                rospy.loginfo(
                    "%s：固定启动点已稳定接管，开始连续悬停 %.1fs",
                    NODE_NAME,
                    step_amount,
                )
            elapsed = (
                rospy.Time.now() - self.auto_search_step_started
            ).to_sec()
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：启动固定点悬停 %.1f/%.1fs，HOVER和动作门槛均通过",
                NODE_NAME,
                min(elapsed, step_amount),
                step_amount,
            )
            if elapsed >= step_amount:
                self.complete_auto_search_step(step_kind, step_amount)
            return

        if not model_ready:
            self.pause_search_for_model()
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：模型话题未就绪，搜索步骤 %d/%d 已主动刹停暂停",
                NODE_NAME,
                self.auto_search_index + 1,
                len(self.auto_search_plan),
            )
            return
        if not self.resume_search_after_model_ready():
            return
        if self.state != self.WAIT_FOR_TARGET:
            return

        if self.auto_search_step_goal is None:
            current = self.get_current_pose("生成搜索绝对目标")
            if current is None:
                return
            forward, right = self.search_step_offsets(
                step_kind,
                step_amount,
            )
            step_description = self.search_step_description(
                step_kind,
                step_amount,
            )
            self.auto_search_step_goal = copy.deepcopy(
                self.set_body_offset_goal(
                    current,
                    forward,
                    right,
                    "搜索步骤 {}/{}：{}".format(
                        self.auto_search_index + 1,
                        len(self.auto_search_plan),
                        step_description,
                    ),
                )
            )
            if self.state != self.WAIT_FOR_TARGET:
                self.active_goal = None
                self.auto_search_step_goal = None
                return
            rospy.loginfo(
                "%s：开始搜索步骤 %d/%d：%s，等待匹配目标的 HOVER",
                NODE_NAME,
                self.auto_search_index + 1,
                len(self.auto_search_plan),
                step_description,
            )

        if self.state != self.WAIT_FOR_TARGET:
            return
        if self.motion_arrived():
            if self.state != self.WAIT_FOR_TARGET:
                return
            self.complete_auto_search_step(step_kind, step_amount)
            return

        checks = self.actual_arrival_checks()
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：搜索步骤 %d/%d 进行中：%s，motion=%s，"
                "base误差=%s，水平速度=%s"
            ),
            NODE_NAME,
            self.auto_search_index + 1,
            len(self.auto_search_plan),
            self.search_step_description(step_kind, step_amount),
            self.MOTION_STATE_NAMES.get(
                self.latest_motion_state.state, "未知"
            ),
            "未知" if checks is None else "{:.3f}m".format(
                checks["position_error"]
            ),
            "未知" if checks is None else "{:.3f}m/s".format(
                checks["horizontal_speed"]
            ),
        )
        self.log_arrival_gate(
            "搜索步骤 {}/{} 到达判定".format(
                self.auto_search_index + 1,
                len(self.auto_search_plan),
            )
        )

    def auto_target_errors(self, target):
        image_width = target.get("image_width", self.auto_image_width)
        image_height = target.get("image_height", self.auto_image_height)
        desired_u = image_width * self.auto_target_center_u_ratio
        desired_v = image_height * self.auto_target_center_v_ratio
        error_u_px = target["center_u"] - desired_u
        error_v_px = target["center_v"] - desired_v
        normalized_u = error_u_px / max(image_width * 0.5, 1.0)
        normalized_v = error_v_px / max(image_height * 0.5, 1.0)
        return error_u_px, error_v_px, normalized_u, normalized_v

    def visual_step(self, normalized_error, gain, sign):
        raw_step = gain * normalized_error * sign
        raw_step = clamp(
            raw_step,
            -self.auto_visual_max_step_m,
            self.auto_visual_max_step_m,
        )
        if raw_step == 0.0 or abs(raw_step) >= self.auto_visual_min_step_m:
            return raw_step
        return math.copysign(self.auto_visual_min_step_m, raw_step)

    def visual_goal_interval_ready(self, frame_index):
        if frame_index <= self.last_visual_goal_frame:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：模型帧#%d已经生成过视觉目标，本周期不重复叠加",
                NODE_NAME,
                frame_index,
            )
            return False
        if self.last_visual_goal_time is None:
            return True
        interval = (
            rospy.Time.now() - self.last_visual_goal_time
        ).to_sec()
        if interval >= self.auto_visual_goal_min_interval:
            return True
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：视觉目标更新间隔%.2f/%.2fs，本帧只更新识别结果",
            NODE_NAME,
            interval,
            self.auto_visual_goal_min_interval,
        )
        return False

    def approach_target_automatically(self):
        now = rospy.Time.now()
        status = self.get_recent_status("方框细对准")
        if status is None:
            if not self.visual_stop_locked:
                self.request_motion_cancel(
                    "/status/auv 不可用，暂停方框细对准"
                )
                self.visual_stop_locked = True
            self.reset_auto_center_stability("/status/auv 不可用或超时")
            return
        if not self.wait_for_motion_cancel("细对准等待主动刹停"):
            return

        target_age = None
        if self.last_target_time is not None:
            target_age = (now - self.last_target_time).to_sec()
        if target_age is not None and target_age > self.detection_timeout:
            self.current_auto_target = None
            if not self.visual_stop_locked:
                self.request_motion_cancel(
                    "目标识别结果超时，停止细对准",
                    discard_search_resume=True,
                )
                self.visual_stop_locked = True
            if not self.wait_for_motion_cancel("目标超时后等待定点接管"):
                return
            self.reset_auto_center_stability("目标识别结果超时")
            self.reset_stability()
            self.reset_auto_search_step()
            self.set_state(self.WAIT_FOR_TARGET, "目标丢失超时，重新执行当前搜索步骤")
            return

        target = self.current_auto_target
        if target is None:
            if not self.visual_stop_locked:
                self.request_motion_cancel(
                    "当前模型帧未识别到目标，停止水平运动",
                    discard_search_resume=True,
                )
                self.visual_stop_locked = True
            self.reset_auto_center_stability("当前模型帧未识别到目标")
            return

        self.visual_stop_locked = False
        error_u_px, error_v_px, normalized_u, normalized_v = (
            self.auto_target_errors(target)
        )
        centered_u = abs(error_u_px) <= self.auto_center_tolerance_u_px
        centered_v = abs(error_v_px) <= self.auto_center_tolerance_v_px
        centered = centered_u and centered_v
        frame_index = int(target.get("frame_index", 0))

        if not centered:
            self.visual_center_hold_requested = False
            if self.visual_goal_interval_ready(frame_index):
                current = self.get_current_pose("生成方框视觉小步目标")
                if current is None or self.current_auto_target is None:
                    self.reset_auto_center_stability(
                        "无法读取当前位姿或最新模型帧已丢失目标"
                    )
                    return
                forward_step = 0.0
                if not centered_v:
                    forward_step = self.visual_step(
                        -normalized_v,
                        self.auto_visual_forward_gain_m,
                        self.auto_forward_sign,
                    )
                right_step = 0.0
                if not centered_u:
                    right_step = self.visual_step(
                        normalized_u,
                        self.auto_visual_lateral_gain_m,
                        self.auto_lateral_sign,
                    )
                self.set_body_offset_goal(
                    current,
                    forward_step,
                    right_step,
                    "依据方框中心像素生成细对准位置小步",
                )
                self.last_visual_goal_frame = frame_index
                self.last_visual_goal_time = now
                rospy.loginfo(
                    (
                        "%s：[模型帧 #%d] 视觉位置小步已发布："
                        "像素误差=(u=%+.1f,v=%+.1f)，本体偏置=(前%+.3f,右%+.3f)m"
                    ),
                    NODE_NAME,
                    frame_index,
                    error_u_px,
                    error_v_px,
                    forward_step,
                    right_step,
                )
        elif not self.visual_center_hold_requested:
            current = self.get_current_pose("方框进入中心后锁定当前位置")
            if current is None or self.current_auto_target is None:
                return
            self.set_active_goal(
                current.pose.position.x,
                current.pose.position.y,
                self.auto_hold_z,
                self.auto_hold_yaw,
                "方框进入中心容差，锁定固定点等待 HOVER",
            )
            self.visual_center_hold_requested = True

        pose_errors = self.status_pose_errors(status)
        depth_error = 0.0 if pose_errors is None else pose_errors[0]
        yaw_error_deg = 0.0 if pose_errors is None else pose_errors[1]
        message = self.latest_motion_state
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：自动对齐：中心=(%.1f,%.1f)，像素误差=(u=%+.1f,v=%+.1f)，"
                "motion=%s，base误差=%.3fm，水平速度=%.3fm/s，"
                "航向角速度=%+.2fdeg/s，mode=%d，深度误差=%+.3fm，"
                "航向误差=%+.2fdeg"
            ),
            NODE_NAME,
            target["center_u"],
            target["center_v"],
            error_u_px,
            error_v_px,
            self.MOTION_STATE_NAMES.get(message.state, "未知"),
            message.base_position_error,
            message.horizontal_speed,
            math.degrees(message.yaw_rate),
            status["control_mode"],
            depth_error,
            yaw_error_deg,
        )

        if not centered:
            self.reset_auto_center_stability("方框中心超出允许范围")
            return
        if (
            self.auto_centered_frame_count
            < self.auto_center_stable_detection_count
        ):
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：方框已进入中心范围，等待连续居中识别 %d/%d 帧",
                NODE_NAME,
                self.auto_centered_frame_count,
                self.auto_center_stable_detection_count,
            )
            return
        if not self.action_status_is_stable(status):
            self.log_arrival_gate("方框居中后的动作放行到达判定")
            checks = self.actual_arrival_checks()
            goal_errors = self.goal_match_errors()
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：动作放行等待：居中=%d/%d帧；motion=%s/HOVER；"
                    "目标匹配误差=%s；base误差=%s；水平速度=%s；"
                    "动作水平速度=%.3f<=%.3f；下向速度=%.3f<=%.3f；"
                    "航向角速度=%.3f<=%.3frad/s；深度误差=%.3f<=%.3f；"
                    "航向误差=%.2f<=%.2f；mode=%d/4"
                ),
                NODE_NAME,
                self.auto_centered_frame_count,
                self.auto_center_stable_detection_count,
                self.MOTION_STATE_NAMES.get(message.state, "未知"),
                "未知" if goal_errors is None else (
                    "水平{:.3f}m/深度{:.3f}m/航向{:.2f}deg".format(
                        goal_errors[0], goal_errors[1], goal_errors[2]
                    )
                ),
                "未知" if checks is None else "{:.3f}m".format(
                    checks["position_error"]
                ),
                "未知" if checks is None else "{:.3f}m/s".format(
                    checks["horizontal_speed"]
                ),
                abs(message.horizontal_speed),
                self.auto_action_max_horizontal_speed,
                abs(status["vz"]),
                self.auto_action_max_vertical_speed,
                abs(message.yaw_rate),
                self.auto_action_max_yaw_rate,
                abs(depth_error),
                self.auto_action_max_depth_error,
                abs(yaw_error_deg),
                self.auto_action_max_yaw_error_deg,
                status["control_mode"],
            )
            return

        if not self.capture_action_hold_position():
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：无法记录最终定点位置，暂不执行夹爪动作",
                NODE_NAME,
            )
            return
        self.publish_actuator(self.clamp_open, self.target_color)
        self.set_state(
            self.OPEN_CLAMP,
            "方框连续居中 {} 帧且 motion_supervisor 已稳定 HOVER".format(
                self.auto_center_stable_detection_count
            ),
        )

    def confirm_target_after_hover(self):
        if self.state != self.AUTO_HOVER_CONFIRM:
            return
        if not self.wait_for_motion_cancel("首次识别后等待刹停悬停"):
            return

        now = rospy.Time.now()
        if self.hover_confirmation_hover_at is None:
            self.hover_confirmation_hover_at = now
            self.reset_stability()
            self.current_auto_target = None
            rospy.loginfo(
                (
                    "%s：motion_supervisor 已完成刹停并锁定当前位置；"
                    "先稳定悬停 %.2fs，期间模型帧不参与第二轮复核"
                ),
                NODE_NAME,
                self.auto_hover_confirm_settle_seconds,
            )

        settle_elapsed = (
            now - self.hover_confirmation_hover_at
        ).to_sec()
        if settle_elapsed < self.auto_hover_confirm_settle_seconds:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：悬停画面稳定等待 %.2f/%.2fs",
                NODE_NAME,
                settle_elapsed,
                self.auto_hover_confirm_settle_seconds,
            )
            return

        if not self.hover_confirmation_ready:
            self.reset_stability()
            self.current_auto_target = None
            self.hover_confirmation_ready = True
            self.hover_confirmation_started_at = now
            rospy.loginfo(
                (
                    "%s：悬停画面稳定等待完成；已再次清空旧帧，"
                    "开始第二轮 %d/%d 帧候选组复核，最长等待 %.1fs"
                ),
                NODE_NAME,
                self.auto_search_stable_detection_count,
                self.stable_detection_window_size,
                self.auto_hover_confirm_timeout,
            )

        confirm_elapsed = (
            now - self.hover_confirmation_started_at
        ).to_sec()
        if confirm_elapsed >= self.auto_hover_confirm_timeout:
            self.resume_search_after_hover_confirmation(
                "悬停复核 %.1fs 内未形成 %d/%d 帧稳定候选组"
                % (
                    self.auto_hover_confirm_timeout,
                    self.auto_search_stable_detection_count,
                    self.stable_detection_window_size,
                )
            )
            return

        if self.last_model_message_time is None:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：悬停复核等待模型话题 %s",
                NODE_NAME,
                self.detection_topic,
            )
            return

        model_age = (
            rospy.Time.now() - self.last_model_message_time
        ).to_sec()
        if model_age > self.detection_timeout:
            self.reset_stability()
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：悬停复核期间模型话题已 %.2fs 没有新消息，继续保持定点",
                NODE_NAME,
                model_age,
            )
            return

        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：悬停复核进行中：时间=%.1f/%.1fs，窗口进度=%d/%d帧，"
                "达到同一位置候选组 %d 帧后进入现有细对准"
            ),
            NODE_NAME,
            confirm_elapsed,
            self.auto_hover_confirm_timeout,
            len(self.detection_frame_window),
            self.stable_detection_window_size,
            self.auto_search_stable_detection_count,
        )

    def resume_search_after_hover_confirmation(self, reason):
        resume_goal = self.hover_confirmation_resume_goal
        self.hover_confirmation_ready = False
        self.hover_confirmation_hover_at = None
        self.hover_confirmation_started_at = None
        self.hover_confirmation_resume_goal = None
        self.current_auto_target = None
        self.visual_stop_locked = False
        self.reset_auto_center_stability("悬停复核未通过")
        self.reset_stability()

        if resume_goal is not None:
            self.active_goal = copy.deepcopy(resume_goal)
            self.active_goal_reason = "悬停复核未通过，继续被打断的搜索目标"
            rospy.logwarn(
                (
                    "%s：%s；恢复搜索绝对目标 map=(%.3f,%.3f,%.3f)，"
                    "搜索步骤不前进"
                ),
                NODE_NAME,
                reason,
                self.active_goal.pose.position.x,
                self.active_goal.pose.position.y,
                self.active_goal.pose.position.z,
            )
        else:
            rospy.logwarn(
                "%s：%s；没有被打断的位移目标，保持当前定点继续搜索",
                NODE_NAME,
                reason,
            )

        self.set_state(
            self.WAIT_FOR_TARGET,
            "悬停重新识别未通过，恢复原搜索流程",
        )

    def label_matches(self, class_name):
        normalized = self.normalize_label(class_name)
        if normalized == self.target_color:
            return True

        # 兼容 yellow_box、yellow_rectangle 等带颜色后缀的模型标签。
        return self.target_color in normalized.split("_")

    @staticmethod
    def finite_number(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    def parse_detection(self, raw_detection, stamp, image_width, image_height):
        if not isinstance(raw_detection, dict):
            return None

        class_name = str(raw_detection.get("class_name", "")).strip()
        confidence = self.finite_number(raw_detection.get("confidence"))
        center = raw_detection.get("center")
        bbox = raw_detection.get("bbox")

        if not class_name or confidence is None:
            return None
        if not isinstance(center, dict) or not isinstance(bbox, dict):
            return None

        center_u = self.finite_number(center.get("u"))
        center_v = self.finite_number(center.get("v"))
        x1 = self.finite_number(bbox.get("x1"))
        y1 = self.finite_number(bbox.get("y1"))
        x2 = self.finite_number(bbox.get("x2"))
        y2 = self.finite_number(bbox.get("y2"))
        if None in (center_u, center_v, x1, y1, x2, y2):
            return None

        width = x2 - x1
        height = y2 - y1
        if width <= 0.0 or height <= 0.0:
            return None

        return {
            "stamp": stamp,
            "class_id": raw_detection.get("class_id"),
            "class_name": class_name,
            "normalized_label": self.normalize_label(class_name),
            "confidence": confidence,
            "center_u": center_u,
            "center_v": center_v,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "area": width * height,
            "image_width": image_width,
            "image_height": image_height,
        }

    @staticmethod
    def detection_summary(detection):
        return (
            "%s(id=%s, conf=%.3f, center=(%.0f,%.0f), "
            "bbox=(%.0f,%.0f,%.0f,%.0f))"
            % (
                detection["class_name"],
                detection["class_id"],
                detection["confidence"],
                detection["center_u"],
                detection["center_v"],
                detection["x1"],
                detection["y1"],
                detection["x2"],
                detection["y2"],
            )
        )

    def detection_callback(self, message):
        now = rospy.Time.now()
        previous_model_message_time = self.last_model_message_time
        self.last_model_message_time = now
        self.model_frame_index += 1
        frame_index = self.model_frame_index

        if (
            self.state in (self.WAIT_FOR_TARGET, self.AUTO_HOVER_CONFIRM)
            and previous_model_message_time is not None
            and (now - previous_model_message_time).to_sec()
            > self.detection_timeout
        ):
            gap = (now - previous_model_message_time).to_sec()
            self.reset_stability()
            rospy.logwarn(
                (
                    "%s：[模型帧 #%d] 模型消息中断 %.2fs，超过 %.2fs，"
                    "已清空过期的%d帧候选窗口"
                ),
                NODE_NAME,
                frame_index,
                gap,
                self.detection_timeout,
                self.stable_detection_window_size,
            )

        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as exc:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：无法解析模型 JSON：%s",
                NODE_NAME,
                str(exc),
            )
            if self.state == self.AUTO_APPROACH:
                self.current_auto_target = None
                self.reset_auto_center_stability("模型 JSON 解析失败")
            elif self.state == self.WAIT_FOR_TARGET or (
                self.state == self.AUTO_HOVER_CONFIRM
                and self.hover_confirmation_ready
            ):
                self.add_detection_sample(
                    None, frame_index, "模型 JSON 解析失败"
                )
            return

        if not isinstance(payload, dict):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：模型 JSON 根节点不是对象",
                NODE_NAME,
            )
            if self.state == self.AUTO_APPROACH:
                self.current_auto_target = None
                self.reset_auto_center_stability("模型 JSON 根节点无效")
            elif self.state == self.WAIT_FOR_TARGET or (
                self.state == self.AUTO_HOVER_CONFIRM
                and self.hover_confirmation_ready
            ):
                self.add_detection_sample(
                    None, frame_index, "模型 JSON 根节点无效"
                )
            return

        raw_detections = payload.get("detections", [])
        if not isinstance(raw_detections, list):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：模型 JSON 的 detections 不是数组",
                NODE_NAME,
            )
            if self.state == self.AUTO_APPROACH:
                self.current_auto_target = None
                self.reset_auto_center_stability("模型 detections 字段无效")
            elif self.state == self.WAIT_FOR_TARGET or (
                self.state == self.AUTO_HOVER_CONFIRM
                and self.hover_confirmation_ready
            ):
                self.add_detection_sample(
                    None, frame_index, "模型 detections 字段无效"
                )
            return

        image_width = self.finite_number(payload.get("image_width"))
        image_height = self.finite_number(payload.get("image_height"))
        if image_width is None or image_width <= 0.0:
            image_width = self.auto_image_width
        if image_height is None or image_height <= 0.0:
            image_height = self.auto_image_height

        detections = []
        for raw_detection in raw_detections:
            detection = self.parse_detection(
                raw_detection, now, image_width, image_height
            )
            if detection is not None:
                detections.append(detection)

        summaries = [self.detection_summary(item) for item in detections]
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：模型有效候选=%d：%s",
            NODE_NAME,
            len(detections),
            "; ".join(summaries) if summaries else "无目标",
        )

        if self.state not in (
            self.WAIT_FOR_TARGET,
            self.AUTO_HOVER_CONFIRM,
            self.AUTO_APPROACH,
        ):
            return
        if (
            self.state == self.AUTO_HOVER_CONFIRM
            and not self.hover_confirmation_ready
        ):
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：[模型帧 #%d] 首次识别已通过，"
                    "机器人尚未完成刹停，本帧不计入悬停复核窗口"
                ),
                NODE_NAME,
                frame_index,
            )
            return

        candidates = [
            item
            for item in detections
            if self.label_matches(item["class_name"])
            and item["confidence"] >= self.min_confidence
        ]
        if not candidates:
            if self.state == self.AUTO_APPROACH:
                self.current_auto_target = None
                self.reset_auto_center_stability("本帧未识别到目标颜色方框")
                rospy.loginfo(
                    (
                        "%s：[模型帧 #%d] 自动跟踪帧无效：没有找到 %s 方框，"
                        "将请求 motion_supervisor 主动刹停并锁定当前位置"
                    ),
                    NODE_NAME,
                    frame_index,
                    self.target_color,
                )
                return

            self.add_detection_sample(
                None,
                frame_index,
                "没有找到 %s 方框或置信度低于 %.2f"
                % (self.target_color, self.min_confidence),
            )
            return

        best = max(candidates, key=lambda item: item["confidence"])
        if (
            self.auto_enabled
            and self.state == self.WAIT_FOR_TARGET
            and self.auto_search_index == 0
        ):
            self.reset_stability()
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：[模型帧 #%d] 启动悬停尚未结束，暂不锁定目标：%s；"
                    "悬停完成后重新累计稳定帧"
                ),
                NODE_NAME,
                frame_index,
                self.detection_summary(best),
            )
            return

        if self.state == self.AUTO_APPROACH:
            best["frame_index"] = frame_index
            self.current_auto_target = best
            self.last_target_time = now
            self.visual_stop_locked = False
            error_u_px, error_v_px, _, _ = self.auto_target_errors(best)
            self.update_auto_center_stability(
                best,
                frame_index,
                error_u_px,
                error_v_px,
            )
            rospy.loginfo(
                (
                    "%s：[模型帧 #%d] 自动跟踪有效：%s，"
                    "中心误差=(u=%+.1fpx,v=%+.1fpx)"
                ),
                NODE_NAME,
                frame_index,
                self.detection_summary(best),
                error_u_px,
                error_v_px,
            )
            return

        self.add_detection_sample(best, frame_index)

    def reset_stability(self):
        self.detection_frame_window = []
        self.last_target_time = None

    def required_stable_detection_count(self):
        if self.auto_enabled:
            return self.auto_search_stable_detection_count
        return self.stable_detection_count

    def reset_auto_center_stability(self, reason=""):
        previous_count = self.auto_centered_frame_count
        self.auto_centered_frame_count = 0
        if previous_count > 0:
            rospy.loginfo(
                "%s：连续居中识别 %d -> 0，原因：%s",
                NODE_NAME,
                previous_count,
                reason or "中心稳定条件未通过",
            )

    def update_auto_center_stability(
        self,
        detection,
        frame_index,
        error_u_px,
        error_v_px,
    ):
        centered = (
            abs(error_u_px) <= self.auto_center_tolerance_u_px
            and abs(error_v_px) <= self.auto_center_tolerance_v_px
        )
        if not centered:
            self.reset_auto_center_stability("方框中心超出像素容差")
            rospy.loginfo(
                (
                    "%s：[模型帧 #%d] 对齐帧有效但尚未居中：%s，"
                    "中心误差=(u=%+.1fpx,v=%+.1fpx)"
                ),
                NODE_NAME,
                frame_index,
                self.detection_summary(detection),
                error_u_px,
                error_v_px,
            )
            return

        self.auto_centered_frame_count = min(
            self.auto_centered_frame_count + 1,
            self.auto_center_stable_detection_count,
        )
        rospy.loginfo(
            (
                "%s：[模型帧 #%d] 居中确认第 %d/%d 帧有效：%s，"
                "中心误差=(u=%+.1fpx,v=%+.1fpx)"
            ),
            NODE_NAME,
            frame_index,
            self.auto_centered_frame_count,
            self.auto_center_stable_detection_count,
            self.detection_summary(detection),
            error_u_px,
            error_v_px,
        )

    def add_detection_sample(self, detection, frame_index, invalid_reason=""):
        stage_name = (
            "悬停复核"
            if self.state == self.AUTO_HOVER_CONFIRM
            else "搜索识别" if self.auto_enabled else "人工识别"
        )
        if detection is not None:
            detection["frame_index"] = frame_index
            self.last_target_time = detection["stamp"]

        self.detection_frame_window.append({
            "frame_index": frame_index,
            "detection": detection,
        })
        self.detection_frame_window = self.detection_frame_window[
            -self.stable_detection_window_size :
        ]

        valid_samples = [
            item["detection"]
            for item in self.detection_frame_window
            if item["detection"] is not None
        ]
        candidate_groups = self.build_detection_candidate_groups(valid_samples)
        required_count = self.required_stable_detection_count()
        window_count = len(self.detection_frame_window)
        best_group_count = max(
            (len(group) for group in candidate_groups),
            default=0,
        )

        if detection is None:
            rospy.loginfo(
                (
                    "%s：[%s][模型帧 #%d] 本帧无效：%s；窗口进度=%d/%d帧，"
                    "有效位置帧=%d/%d，最佳位置候选组=%d/%d；"
                    "保留窗口内旧有效帧"
                ),
                NODE_NAME,
                stage_name,
                frame_index,
                invalid_reason or "没有有效目标",
                window_count,
                self.stable_detection_window_size,
                len(valid_samples),
                window_count,
                best_group_count,
                required_count,
            )
            return

        current_group_index = 0
        current_group = [detection]
        for index, group in enumerate(candidate_groups, start=1):
            if any(item is detection for item in group):
                current_group_index = index
                current_group = group
                break

        stable, center_jitter, area_change = self.samples_are_stable(
            current_group,
            required_count,
        )
        frame_ids = [item["frame_index"] for item in current_group]
        rospy.loginfo(
                (
                    "%s：[%s][模型帧 #%d] 本帧有效并加入候选组%d：%s；"
                    "窗口进度=%d/%d帧，有效位置帧=%d/%d，候选组=%d/%d，"
                    "组内帧=%s，中心抖动=%.1f/%.1fpx，面积变化=%.3f/%.3f"
                ),
            NODE_NAME,
            stage_name,
            frame_index,
            current_group_index,
            self.detection_summary(detection),
            window_count,
            self.stable_detection_window_size,
            len(valid_samples),
            window_count,
            len(current_group),
            required_count,
            frame_ids,
            center_jitter,
            self.stable_center_tolerance_px,
            area_change,
            self.stable_area_tolerance_ratio,
        )

        if len(current_group) < required_count:
            return
        if not stable:
            rospy.logwarn(
                (
                    "%s：[%s][模型帧 #%d] 候选组帧数已达到%d，"
                    "但最终一致性未通过；"
                    "继续保留最近%d帧并等待新的匹配帧"
                ),
                NODE_NAME,
                stage_name,
                frame_index,
                required_count,
                self.stable_detection_window_size,
            )
            return

        rospy.loginfo(
            (
                "%s：[%s][模型帧 #%d] 逐帧候选组确认通过：最近%d帧窗口内"
                "位置一致的有效帧=%d/%d，命中帧=%s"
            ),
            NODE_NAME,
            stage_name,
            frame_index,
            self.stable_detection_window_size,
            len(current_group),
            required_count,
            frame_ids,
        )
        self.lock_target(current_group)

    def build_detection_candidate_groups(self, samples):
        groups = []
        for sample in samples:
            matches = []
            for index, group in enumerate(groups):
                median_u, median_v, median_area = self.sample_medians(group)
                center_distance = math.hypot(
                    sample["center_u"] - median_u,
                    sample["center_v"] - median_v,
                )
                area_change = self.area_change_ratio(
                    sample["area"], median_area
                )
                if (
                    center_distance <= self.stable_center_tolerance_px
                    and area_change <= self.stable_area_tolerance_ratio
                ):
                    matches.append((center_distance, area_change, index))

            if not matches:
                groups.append([sample])
                continue

            _, _, best_index = min(matches)
            groups[best_index].append(sample)
        return groups

    @staticmethod
    def sample_medians(samples):
        return (
            statistics.median(item["center_u"] for item in samples),
            statistics.median(item["center_v"] for item in samples),
            statistics.median(item["area"] for item in samples),
        )

    @staticmethod
    def area_change_ratio(area_a, area_b):
        denominator = max(area_a, area_b)
        if denominator <= 0.0:
            return 1.0
        return abs(area_a - area_b) / denominator

    def samples_are_stable(self, samples, required_count):
        if not samples:
            return False, 0.0, 0.0

        median_u, median_v, median_area = self.sample_medians(samples)
        center_jitter = max(
            math.hypot(
                item["center_u"] - median_u,
                item["center_v"] - median_v,
            )
            for item in samples
        )
        area_change = max(
            self.area_change_ratio(item["area"], median_area)
            for item in samples
        )
        stable = (
            len(samples) >= required_count
            and center_jitter <= self.stable_center_tolerance_px
            and area_change <= self.stable_area_tolerance_ratio
        )
        return stable, center_jitter, area_change

    def lock_target(self, samples):
        latest = dict(samples[-1])
        latest["mean_confidence"] = sum(
            item["confidence"] for item in samples
        ) / len(samples)
        latest["mean_center_u"] = statistics.median(
            item["center_u"] for item in samples
        )
        latest["mean_center_v"] = statistics.median(
            item["center_v"] for item in samples
        )
        rospy.loginfo(
            (
                "%s：稳定识别成功：颜色=%s，模型标签=%s，平均置信度=%.3f，"
                "平均中心=(%.1f, %.1f)，最新 bbox=(%.0f, %.0f, %.0f, %.0f)"
            ),
            NODE_NAME,
            self.target_color,
            latest["class_name"],
            latest["mean_confidence"],
            latest["mean_center_u"],
            latest["mean_center_v"],
            latest["x1"],
            latest["y1"],
            latest["x2"],
            latest["y2"],
        )
        if self.auto_enabled:
            if self.state == self.WAIT_FOR_TARGET:
                resume_source = (
                    self.auto_search_step_goal
                    if self.auto_search_step_goal is not None
                    else self.active_goal
                )
                self.hover_confirmation_resume_goal = (
                    None
                    if resume_source is None
                    else copy.deepcopy(resume_source)
                )
                self.current_auto_target = None
                self.reset_auto_center_stability()
                self.visual_stop_locked = False
                self.hover_confirmation_ready = False
                self.hover_confirmation_hover_at = None
                self.hover_confirmation_started_at = None
                self.reset_stability()
                self.request_motion_cancel(
                    "搜索阶段首次稳定识别目标，刹停后重新识别",
                    discard_search_resume=True,
                )
                self.set_state(
                    self.AUTO_HOVER_CONFIRM,
                    "搜索中首次识别通过，等待刹停后重新采集识别帧",
                )
            elif self.state == self.AUTO_HOVER_CONFIRM:
                latest["center_u"] = latest["mean_center_u"]
                latest["center_v"] = latest["mean_center_v"]
                self.current_auto_target = latest
                self.reset_auto_center_stability()
                self.visual_stop_locked = False
                self.hover_confirmation_ready = False
                self.hover_confirmation_hover_at = None
                self.hover_confirmation_started_at = None
                self.hover_confirmation_resume_goal = None
                self.detection_frame_window = []
                self.set_state(
                    self.AUTO_APPROACH,
                    "悬停后第二轮目标识别通过，开始依据中心像素细对准",
                )
        else:
            self.set_state(self.HOLD_BEFORE_ACTION, "模型目标已稳定确认")

    def state_elapsed(self):
        return (rospy.Time.now() - self.state_started).to_sec()

    def set_state(self, state, reason=""):
        previous = self.state
        previous_elapsed = self.state_elapsed()
        self.state = state
        self.state_started = rospy.Time.now()
        rospy.loginfo(
            "%s：状态切换 %s -> %s，上一状态持续 %.1fs，原因：%s",
            NODE_NAME,
            self.STATE_NAMES.get(previous, "未知状态"),
            self.STATE_NAMES.get(state, "未知状态"),
            previous_elapsed,
            reason or "无",
        )

    def publish_actuator(self, clamp_servo, color="off"):
        red, yellow, green = self.COLOR_LIGHTS.get(
            color, self.COLOR_LIGHTS["off"]
        )

        message = ActuatorControl()
        if not hasattr(message, "mode"):
            rospy.logerr_throttle(
                5.0,
                "%s：ActuatorControl 缺少 mode 字段，本次执行器指令未发送",
                NODE_NAME,
            )
            return False

        message.mode = self.actuator_mode
        message.light1 = self.light1
        message.light2 = self.light2
        message.heading_servo = self.heading_servo
        message.clamp_servo = int(clamp_servo)
        message.drive_cmd = self.drive_cmd
        message.drive_speed = self.drive_speed
        message.red_light = red
        message.yellow_light = yellow
        message.green_light = green
        self.actuator_pub.publish(message)

        command = (
            message.mode,
            message.light1,
            message.light2,
            message.heading_servo,
            message.clamp_servo,
            message.drive_cmd,
            message.drive_speed,
            message.red_light,
            message.yellow_light,
            message.green_light,
        )
        if command != getattr(self, "last_actuator_command", None):
            rospy.loginfo(
                (
                    "%s：执行器指令已发布：mode=%d，夹爪=%d，"
                    "颜色灯=(红%d,黄%d,绿%d)，补光灯=(%d,%d)，"
                    "航向舵机=%d，推进电机=(动作%d,转速%d)"
                ),
                NODE_NAME,
                message.mode,
                message.clamp_servo,
                message.red_light,
                message.yellow_light,
                message.green_light,
                message.light1,
                message.light2,
                message.heading_servo,
                message.drive_cmd,
                message.drive_speed,
            )
            self.last_actuator_command = command
        return True

    def finish_task(self, success, reason):
        if self.finished:
            return
        self.finished = True
        if self.auto_enabled and self.cancel_pub is not None:
            self.cancel_pub.publish(Empty())
            self.active_goal = None
            rospy.loginfo(
                "%s：任务结束，已发布 %s 请求主动刹停并定点接管",
                NODE_NAME,
                self.motion_cancel_topic,
            )
        self.publish_actuator(self.clamp_closed, "off")

        if success:
            message = "%s finished" % NODE_NAME
            self.finished_pub.publish(String(data=message))
            rospy.loginfo(
                "%s：子任务3完成，目标颜色=%s，%s，已发布 /finished",
                NODE_NAME,
                self.target_color,
                reason,
            )
        else:
            message = "%s 失败：%s" % (NODE_NAME, reason)
            self.finished_pub.publish(String(data=message))
            rospy.logerr(message)

        rospy.signal_shutdown(message)

    def on_shutdown(self):
        if (
            getattr(self, "auto_enabled", False)
            and getattr(self, "cancel_pub", None) is not None
        ):
            self.cancel_pub.publish(Empty())
            self.active_goal = None
        if hasattr(self, "actuator_pub"):
            self.publish_actuator(self.clamp_closed, "off")

    def run(self):
        if not self.actuator_mode_supported:
            self.finish_task(
                False,
                "ActuatorControl 缺少 mode 字段，请同步消息定义并重新编译",
            )
            return

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            elapsed = (now - self.task_started).to_sec()

            if self.auto_enabled and not self.handle_motion_health():
                if self.finished:
                    return
                self.rate.sleep()
                continue

            if (
                self.state in (
                    self.WAIT_FOR_TARGET,
                    self.AUTO_HOVER_CONFIRM,
                    self.AUTO_APPROACH,
                )
                and elapsed >= self.max_wait_seconds
            ):
                self.finish_task(
                    False,
                    "自动搜索/等待 %.1fs 后仍未完成 %s 方框确认与对齐"
                    % (self.max_wait_seconds, self.target_color)
                    if self.auto_enabled
                    else "等待 %.1fs 后仍未稳定识别到 %s 方框"
                    % (self.max_wait_seconds, self.target_color),
                )
                return

            if self.state == self.WAIT_FOR_TARGET:
                self.publish_actuator(self.clamp_closed, "off")
                model_ready = False
                if self.last_model_message_time is None:
                    rospy.logwarn_throttle(
                        self.warning_log_interval,
                        "%s：等待模型话题 %s，已等待 %.1fs",
                        NODE_NAME,
                        self.detection_topic,
                        elapsed,
                    )
                else:
                    model_age = (now - self.last_model_message_time).to_sec()
                    if model_age > self.detection_timeout:
                        self.reset_stability()
                        rospy.logwarn_throttle(
                            self.warning_log_interval,
                            "%s：模型话题已 %.1fs 没有新消息",
                            NODE_NAME,
                            model_age,
                        )
                    else:
                        model_ready = True

                if self.auto_enabled:
                    if not self.initialize_auto_pose():
                        self.rate.sleep()
                        continue
                    self.search_target_automatically(model_ready)

            elif self.state == self.AUTO_HOVER_CONFIRM:
                self.publish_actuator(self.clamp_closed, "off")
                self.confirm_target_after_hover()

            elif self.state == self.AUTO_APPROACH:
                self.publish_actuator(self.clamp_closed, "off")
                self.approach_target_automatically()

            elif self.state == self.HOLD_BEFORE_ACTION:
                if self.auto_enabled:
                    self.publish_action_position_hold("动作前最终定点保持")
                self.publish_actuator(self.clamp_closed, self.target_color)
                if self.state_elapsed() >= self.hold_seconds:
                    rospy.loginfo(
                        "%s：识别确认完成，打开夹爪执行投放，颜色灯=%s",
                        NODE_NAME,
                        self.target_color,
                    )
                    self.publish_actuator(self.clamp_open, self.target_color)
                    self.set_state(self.OPEN_CLAMP, "开始执行投放动作")

            elif self.state == self.OPEN_CLAMP:
                if self.auto_enabled:
                    self.publish_action_position_hold(
                        "开灯和夹爪打开期间最终定点保持"
                    )
                self.publish_actuator(self.clamp_open, self.target_color)
                open_elapsed = self.state_elapsed()
                rospy.loginfo_throttle(
                    self.log_interval,
                    "%s：已开%s灯并打开夹爪，最终定点保持 %.1f/%.1fs",
                    NODE_NAME,
                    self.target_color,
                    min(open_elapsed, self.open_seconds),
                    self.open_seconds,
                )
                if open_elapsed >= self.open_seconds:
                    rospy.loginfo(
                        "%s：开灯并打开夹爪后已定点保持 %.1fs，关闭夹爪并熄灯",
                        NODE_NAME,
                        self.open_seconds,
                    )
                    self.publish_actuator(self.clamp_closed, "off")
                    self.set_state(self.CLOSE_CLAMP, "夹爪打开时间完成")

            elif self.state == self.CLOSE_CLAMP:
                if self.auto_enabled:
                    self.publish_action_position_hold(
                        "夹爪关闭期间最终定点保持"
                    )
                self.publish_actuator(self.clamp_closed, "off")
                if self.state_elapsed() >= self.close_seconds:
                    self.finish_task(True, "识别和投放动作执行完成")
                    return

            if self.auto_enabled and not self.finished:
                self.publish_active_goal()
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3InspectAndDropTest().run()
    except rospy.ROSInterruptException:
        pass
