#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
名称：test_task3_3_inspect_and_drop.py
功能：识别指定颜色方框，基于map位置完成camera粗对准、X误差精对准、投放和离场
作者：Tangzongle
监听：/vision/rectangle/target_message (auv_control/TargetDetection)
      /vision/rectangle/detections (std_msgs/String，人工模式兼容)
      /motion/state (auv_control/MotionState)
      /status/auv (auv_control/AUVData)
      /status/actuator (auv_control/ActuatorControl)
发布：/cmd/motion/goal (geometry_msgs/PoseStamped)
      /cmd/actuator (auv_control/ActuatorControl)
      /finished (std_msgs/String)
记录：
2026.8.3
    自动模式改为使用带时间戳的三维方框map位置队列，按三帧稳定位置完成camera粗对准和X误差精对准。
2026.8.3
    投放后返原点改为使用任务初始航向向左偏移配置角度得到的绝对航向。
"""

import copy
from datetime import datetime
import itertools
import json
import logging
import math
import os
import statistics

import rospy
import tf
from auv_control.msg import AUVData, ActuatorControl, MotionState, TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task3_3_inspect_and_drop"


def configure_task_file_logging(subtask_name):
    """将本节点的rospy日志同时保存到带时间戳的UTF-8文件。"""
    log_directory = os.path.abspath(os.path.expanduser(str(
        rospy.get_param("~log_directory", "~/.ros/auv_logs/task3")
    )))
    try:
        os.makedirs(log_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = os.path.join(
            log_directory, "{}_{}.log".format(subtask_name, timestamp)
        )
        handler = logging.FileHandler(
            log_path, mode="a", encoding="utf-8"
        )
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        ))
        ros_logger = logging.getLogger("rosout")
        ros_logger.addHandler(handler)
    except (IOError, OSError) as error:
        rospy.logerr(
            "%s：无法创建文件日志目录%s：%s",
            NODE_NAME,
            log_directory,
            str(error),
        )
        return None
    rospy.loginfo("%s：文件日志已启用：%s", NODE_NAME, log_path)
    return log_path


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
DEFAULT_TARGET_TOPIC = "/vision/rectangle/target_message"
DEFAULT_TARGET_COLOR = "yellow"
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_STABLE_DETECTION_COUNT = 5
DEFAULT_AUTO_SEARCH_STABLE_DETECTION_COUNT = 3
DEFAULT_AUTO_CENTER_STABLE_DETECTION_COUNT = 5
DEFAULT_STABLE_DETECTION_WINDOW_SIZE = 10
DEFAULT_STABLE_MAP_POSITION_TOLERANCE_M = 0.20
DEFAULT_AUTO_HOVER_CONFIRM_SETTLE_SECONDS = 0.5
DEFAULT_AUTO_HOVER_CONFIRM_TIMEOUT = 4.0
DEFAULT_STABLE_CENTER_TOLERANCE_PX = 40.0
DEFAULT_STABLE_AREA_TOLERANCE_RATIO = 0.35
DEFAULT_DETECTION_TIMEOUT = 2.0
DEFAULT_MAX_WAIT_SECONDS = 300.0

# 操作模式和 motion_supervisor 接口参数。
DEFAULT_OPERATION_MODE = "manual"
DEFAULT_MOTION_GOAL_TOPIC = "/cmd/motion/goal"
DEFAULT_MOTION_STATE_TOPIC = "/motion/state"
DEFAULT_STATUS_TOPIC = "/status/auv"
DEFAULT_MOTION_STARTUP_TIMEOUT = 10.0
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
DEFAULT_AUTO_SEARCH_LEFT_DISTANCE = 0.20
DEFAULT_AUTO_SEARCH_RIGHT_DISTANCE = 0.40
DEFAULT_AUTO_COLOR_FAST_SEARCH_ENABLED = True
DEFAULT_AUTO_VISUAL_LATERAL_GAIN_M = 0.10
DEFAULT_AUTO_VISUAL_MIN_STEP_M = 0.01
DEFAULT_AUTO_VISUAL_MAX_STEP_M = 0.05
DEFAULT_AUTO_VISUAL_GOAL_MIN_INTERVAL = 0.50
DEFAULT_AUTO_LATERAL_SIGN = 1.0
DEFAULT_AUTO_ACTION_MAX_HORIZONTAL_SPEED = 0.03
DEFAULT_AUTO_ACTION_MAX_VERTICAL_SPEED = 0.03
DEFAULT_AUTO_ACTION_MAX_YAW_RATE = 0.05
DEFAULT_AUTO_ACTION_MAX_DEPTH_ERROR = 0.08
DEFAULT_AUTO_ACTION_MAX_YAW_ERROR_DEG = 5.0
DEFAULT_FINE_POSITION_X_TOLERANCE_M = 0.10
DEFAULT_AUTO_TARGET_CENTER_U_RATIO = 0.5
DEFAULT_AUTO_TARGET_CENTER_V_RATIO = 0.5
DEFAULT_AUTO_CENTER_TOLERANCE_U_PX = 60.0
DEFAULT_AUTO_IMAGE_WIDTH = 640.0
DEFAULT_AUTO_IMAGE_HEIGHT = 480.0
DEFAULT_LOG_INTERVAL = 1.0
DEFAULT_WARNING_LOG_INTERVAL = 2.0

# 识别成功后的动作参数。
DEFAULT_HOLD_SECONDS = 1.0
DEFAULT_OPEN_SECONDS = 3.0
DEFAULT_RETURN_RIGHT_SECONDS = 1.0
DEFAULT_CLOSE_SECONDS = 0.0
DEFAULT_PRE_DROP_FORWARD_DISTANCE = 0.20
DEFAULT_PRE_DROP_FORWARD_TIMEOUT = 90.0
DEFAULT_POST_DROP_MOTION_ENABLED = True
DEFAULT_TASK3_INITIAL_YAW_DEG = 215.0
DEFAULT_RETURN_ORIGIN_YAW_OFFSET_DEG = 180.0
DEFAULT_POST_DROP_STEP_TIMEOUT = 90.0
DEFAULT_POST_DROP_ASCENT_SECONDS = 5.0
DEFAULT_POST_DROP_ASCENT_TARGET_Z = -1.3
POST_DROP_ORIGIN_X = 0.0
POST_DROP_ORIGIN_Y = 0.0

# 执行器参数。
DEFAULT_ACTUATOR_TOPIC = "/cmd/actuator"
DEFAULT_ACTUATOR_STATUS_TOPIC = "/status/actuator"
DEFAULT_ACTUATOR_MODE = 2  # 0=不响应，1=仅补光灯，2=仅执行器
DEFAULT_CLAMP_OPEN = 0x00
DEFAULT_CLAMP_CLOSED = 0xFF
DEFAULT_HEADING_SERVO_ENABLED = False
DEFAULT_HEADING_SERVO_RIGHT = 0x00
DEFAULT_HEADING_SERVO_CENTER = 0x80
DEFAULT_ACTUATOR_STATUS_TIMEOUT = 1.0
DEFAULT_ACTUATOR_FEEDBACK_CONFIRM_FRAMES = 2
DEFAULT_ACTUATOR_SERVO_TOLERANCE = 2
DEFAULT_ACTUATOR_STAGE_TIMEOUT = 15.0
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
    RETURN_GRIPPER_RIGHT = 5
    CLOSE_CLAMP = 6
    POST_DROP_TURN = 7
    POST_DROP_RETURN_ORIGIN = 8
    PRE_DROP_FORWARD = 9
    POST_DROP_ASCEND = 10

    STATE_NAMES = {
        WAIT_FOR_TARGET: "等待目标颜色方框",
        AUTO_HOVER_CONFIRM: "camera粗对准后等待HOVER复核方框",
        AUTO_APPROACH: "方框map位置X误差精对准并保持航向",
        HOLD_BEFORE_ACTION: "夹爪移动到中间",
        OPEN_CLAMP: "打开夹爪",
        RETURN_GRIPPER_RIGHT: "打开状态回到右侧",
        CLOSE_CLAMP: "关闭夹爪",
        POST_DROP_TURN: "投放后左转",
        POST_DROP_RETURN_ORIGIN: "投放后返回map原点",
        PRE_DROP_FORWARD: "投放前固定前进",
        POST_DROP_ASCEND: "原点持续上浮",
    }

    SEARCH_STEP_NAMES = {
        "hover": "启动悬停",
        "forward": "向前移动",
        "left": "向左横移",
        "right": "向右横移",
        "observe_center": "中间位置颜色观察",
        "observe_left": "左侧位置颜色观察",
        "observe_inferred_left": "推断左侧目标复核",
        "observe_inferred_right": "推断右侧目标复核",
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
        self.target_topic = str(
            rospy.get_param("~model_target_topic", DEFAULT_TARGET_TOPIC)
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
        self.stable_map_position_tolerance_m = float(rospy.get_param(
            "~stable_map_position_tolerance_m",
            DEFAULT_STABLE_MAP_POSITION_TOLERANCE_M,
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
        self.return_right_seconds = float(rospy.get_param(
            "~return_right_seconds", DEFAULT_RETURN_RIGHT_SECONDS
        ))
        self.close_seconds = float(
            rospy.get_param("~close_seconds", DEFAULT_CLOSE_SECONDS)
        )
        self.pre_drop_forward_distance = float(rospy.get_param(
            "~pre_drop_forward_distance",
            DEFAULT_PRE_DROP_FORWARD_DISTANCE,
        ))
        self.pre_drop_forward_timeout = float(rospy.get_param(
            "~pre_drop_forward_timeout",
            DEFAULT_PRE_DROP_FORWARD_TIMEOUT,
        ))
        self.post_drop_motion_enabled = bool(rospy.get_param(
            "~post_drop_motion_enabled",
            DEFAULT_POST_DROP_MOTION_ENABLED,
        ))
        self.task3_initial_yaw_deg = float(rospy.get_param(
            "/task3_initial_yaw_deg",
            DEFAULT_TASK3_INITIAL_YAW_DEG,
        ))
        self.return_origin_yaw_offset_deg = float(rospy.get_param(
            "/task3_return_origin_yaw_offset_deg",
            DEFAULT_RETURN_ORIGIN_YAW_OFFSET_DEG,
        ))
        self.post_drop_step_timeout = float(rospy.get_param(
            "~post_drop_step_timeout",
            DEFAULT_POST_DROP_STEP_TIMEOUT,
        ))
        self.post_drop_ascent_seconds = float(rospy.get_param(
            "~post_drop_ascent_seconds",
            DEFAULT_POST_DROP_ASCENT_SECONDS,
        ))
        self.post_drop_ascent_target_z = float(rospy.get_param(
            "~post_drop_ascent_target_z",
            DEFAULT_POST_DROP_ASCENT_TARGET_Z,
        ))

        self.motion_goal_topic = str(rospy.get_param(
            "~motion_goal_topic", DEFAULT_MOTION_GOAL_TOPIC
        )).strip()
        self.motion_state_topic = str(rospy.get_param(
            "~motion_state_topic", DEFAULT_MOTION_STATE_TOPIC
        )).strip()
        self.status_topic = str(
            rospy.get_param("~status_topic", DEFAULT_STATUS_TOPIC)
        ).strip()
        self.motion_state_timeout = float(rospy.get_param(
            "/task3_protection/motion_feedback_timeout", 3.0
        ))
        self.motion_startup_timeout = float(rospy.get_param(
            "~motion_startup_timeout", DEFAULT_MOTION_STARTUP_TIMEOUT
        ))
        self.hold_timeout = float(rospy.get_param(
            "/task3_protection/cancel_recovery_timeout", 30.0
        ))
        self.status_timeout = float(rospy.get_param(
            "~status_timeout", DEFAULT_STATUS_TIMEOUT
        ))
        self.status_linear_velocity_scale = float(rospy.get_param(
            "~status_linear_velocity_scale",
            DEFAULT_STATUS_LINEAR_VELOCITY_SCALE,
        ))
        self.fixed_depth_m = float(rospy.get_param(
            "/task3_target_depth_m", 0.60
        ))
        self.fixed_map_z = -self.fixed_depth_m
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
        self.auto_search_left_distance = float(rospy.get_param(
            "~auto_search_left_distance", DEFAULT_AUTO_SEARCH_LEFT_DISTANCE
        ))
        self.auto_search_right_distance = float(rospy.get_param(
            "~auto_search_right_distance", DEFAULT_AUTO_SEARCH_RIGHT_DISTANCE
        ))
        self.auto_color_fast_search_enabled = bool(rospy.get_param(
            "~auto_color_fast_search_enabled",
            DEFAULT_AUTO_COLOR_FAST_SEARCH_ENABLED,
        ))
        if self.auto_enabled and self.auto_color_fast_search_enabled:
            self.auto_color_fast_search_enabled = False
            rospy.logwarn(
                (
                    "%s：自动map对准模式下 /vision/rectangle/detections "
                    "仅用于模型在线检查，已禁用依赖二维检测内容的快速颜色搜索；"
                    "继续使用固定步长搜索路径"
                ),
                NODE_NAME,
            )
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
        self.fine_position_x_tolerance_m = float(rospy.get_param(
            "~fine_position_x_tolerance_m",
            DEFAULT_FINE_POSITION_X_TOLERANCE_M,
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
        self.actuator_status_topic = str(rospy.get_param(
            "~actuator_status_topic", DEFAULT_ACTUATOR_STATUS_TOPIC
        )).strip()
        self.actuator_mode = int(
            rospy.get_param("~actuator_mode", DEFAULT_ACTUATOR_MODE)
        )
        self.clamp_open = int(
            rospy.get_param("~clamp_open", DEFAULT_CLAMP_OPEN)
        )
        self.clamp_closed = int(
            rospy.get_param("~clamp_closed", DEFAULT_CLAMP_CLOSED)
        )
        self.heading_servo_enabled = bool(rospy.get_param(
            "~heading_servo_enabled", DEFAULT_HEADING_SERVO_ENABLED
        ))
        self.heading_servo_right = int(
            rospy.get_param(
                "~heading_servo_right", DEFAULT_HEADING_SERVO_RIGHT
            )
        )
        self.heading_servo_center = int(
            rospy.get_param(
                "~heading_servo_center", DEFAULT_HEADING_SERVO_CENTER
            )
        )
        self.actuator_status_timeout = float(rospy.get_param(
            "~actuator_status_timeout", DEFAULT_ACTUATOR_STATUS_TIMEOUT
        ))
        self.actuator_feedback_confirm_frames = int(rospy.get_param(
            "~actuator_feedback_confirm_frames",
            DEFAULT_ACTUATOR_FEEDBACK_CONFIRM_FRAMES,
        ))
        self.actuator_servo_tolerance = int(rospy.get_param(
            "~actuator_servo_tolerance", DEFAULT_ACTUATOR_SERVO_TOLERANCE
        ))
        self.actuator_stage_timeout = float(rospy.get_param(
            "~actuator_stage_timeout", DEFAULT_ACTUATOR_STAGE_TIMEOUT
        ))
        self.drive_cmd = int(
            rospy.get_param("~drive_cmd", DEFAULT_DRIVE_CMD)
        )
        self.drive_speed = int(
            rospy.get_param("~drive_speed", DEFAULT_DRIVE_SPEED)
        )
        self.light1 = int(rospy.get_param("~light1", DEFAULT_LIGHT1))
        self.light2 = int(rospy.get_param("~light2", DEFAULT_LIGHT2))

        self.validate_params()
        self.return_origin_target_yaw = normalize_angle_rad(math.radians(
            self.task3_initial_yaw_deg - self.return_origin_yaw_offset_deg
        ))

        # mode 字段由传感器协议新增；团队消息定义合并并重新编译后才会存在。
        self.actuator_mode_supported = hasattr(ActuatorControl(), "mode")

        self.finished_pub = rospy.Publisher(
            "/finished", String, queue_size=10
        )
        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.goal_pub = None
        self.tf_listener = None
        if self.auto_enabled:
            self.goal_pub = rospy.Publisher(
                self.motion_goal_topic, PoseStamped, queue_size=1
            )
            self.tf_listener = tf.TransformListener()

        self.state = self.WAIT_FOR_TARGET
        self.state_started = rospy.Time.now()
        self.task_started = rospy.Time.now()
        self.motion_timeout_started_at = None
        self.max_wait_timed_out = False
        self.last_model_message_time = None
        self.last_target_time = None
        self.detection_frame_window = []
        self.color_scene_frame_window = []
        self.smart_search_first_color = None
        self.hover_confirmation_ready = False
        self.hover_confirmation_hover_at = None
        self.hover_confirmation_started_at = None
        self.hover_confirmation_resume_goal = None
        self.model_frame_index = 0
        self.box_map_frame_index = 0
        self.box_position_samples = []
        self.box_coarse_map_x = None
        self.box_coarse_map_y = None
        self.box_coarse_camera_frame = None
        self.box_fine_candidate = None
        self.box_recheck_collecting = False
        self.box_precision_goal_pending = False
        self.box_final_goal_pending = False
        self.box_final_map_x = None
        self.box_final_map_y = None
        self.box_final_camera_frame = None
        self.box_last_target_time = None
        self.box_position_lock_ready = False
        self.current_auto_target = None
        self.auto_tracking_frame_window = []
        self.auto_tracking_waiting_for_fresh_frame = False
        self.auto_hold_z = None
        self.auto_hold_yaw = None
        self.auto_centered_frame_count = 0
        self.active_goal = None
        self.active_goal_reason = ""
        self.latest_motion_state = None
        self.last_motion_state_received = None
        self.last_motion_state_value = None
        self.motion_ready_once = False
        self.motion_hold_requested_at = None
        self.motion_hold_reason = ""
        self.auto_search_resume_goal = None
        self.auto_search_paused_for_model = False
        self.last_visual_goal_frame = 0
        self.last_visual_goal_time = None
        self.visual_center_hold_requested = False
        self.visual_stop_locked = False
        self.fallback_search_tail = [
            ("forward", self.auto_search_second_forward_distance),
            ("left", self.auto_search_left_distance),
            ("right", self.auto_search_right_distance),
            ("forward", self.auto_search_third_forward_distance),
            ("left", self.auto_search_left_distance),
            ("right", self.auto_search_right_distance),
        ]
        self.auto_search_plan = [
            ("hover", self.auto_initial_hover_seconds),
            ("forward", self.auto_search_first_forward_distance),
        ]
        if self.auto_color_fast_search_enabled:
            self.auto_search_plan.append(("observe_center", 0.0))
        else:
            self.auto_search_plan.extend([
                ("left", self.auto_search_left_distance),
                ("right", self.auto_search_right_distance),
            ])
            self.auto_search_plan.extend(self.fallback_search_tail)
        self.auto_search_index = 0
        self.auto_search_step_started = None
        self.auto_search_step_goal = None
        self.last_status_time = None
        self.current_status = None
        self.status_hold_depth = None
        self.status_hold_yaw_deg = None
        self.auto_action_hold_position = None
        self.pending_drop_reason = ""
        self.drop_action_started = False
        self.post_drop_target_yaw = None
        self.last_actuator_command = None
        self.latest_actuator_status = None
        self.last_actuator_status_time = None
        self.actuator_status_sequence = 0
        self.actuator_feedback_baseline_sequence = 0
        self.actuator_feedback_last_checked_sequence = 0
        self.actuator_feedback_match_count = 0
        self.actuator_feedback_confirmed_at = None
        self.actuator_safe_feedback_count = 0
        self.actuator_safe_feedback_ready = False
        self.actuator_pre_action_wait_started_at = None
        self.finished = False

        self.model_health_sub = None
        if self.auto_enabled:
            self.model_health_sub = rospy.Subscriber(
                self.detection_topic,
                String,
                self.rectangle_health_callback,
                queue_size=10,
            )
            self.detection_sub = rospy.Subscriber(
                self.target_topic,
                TargetDetection,
                self.rectangle_target_callback,
                queue_size=20,
            )
        else:
            self.detection_sub = rospy.Subscriber(
                self.detection_topic,
                String,
                self.detection_callback,
                queue_size=10,
            )
        self.actuator_status_sub = rospy.Subscriber(
            self.actuator_status_topic,
            ActuatorControl,
            self.actuator_status_callback,
            queue_size=20,
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
                    "%s：启动自动寻找模式，运动目标=%s，反馈=%s；"
                    "所有暂停和刹停都改为下发机器人当前map位姿；"
                    "底层 mode=4、推力、阻尼和刹车全部由 motion_supervisor 管理"
                ),
                NODE_NAME,
                self.motion_goal_topic,
                self.motion_state_topic,
            )
        else:
            rospy.loginfo(
                "%s：启动人工操作模式，只识别和执行动作，不发布机器人运动指令",
                NODE_NAME,
            )
        rospy.loginfo(
            "%s：主循环频率=%.1fHz，首次运动后总超时=%.1fs",
            NODE_NAME,
            self.rate_hz,
            self.max_wait_seconds,
        )
        rospy.loginfo(
            (
                "%s：模型话题=%s，三维目标话题=%s，目标颜色=%s，最低置信度=%.2f"
            ),
            NODE_NAME,
            self.detection_topic,
            self.target_topic,
            self.target_color,
            self.min_confidence,
        )
        rospy.loginfo(
            (
                "%s：逐帧候选组：最近%d个模型帧内保留有效检测，"
                "位置误差<=%.1fpx，面积变化比例<=%.2f；"
                "识别超时 %.1fs，首次运动后总等待上限 %.1fs"
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
                    "%s：自动模式三维位置门槛：最多%d帧队列，"
                    "任意%d帧map位置相近即通过；精确认X误差<=%.3fm"
                ),
                NODE_NAME,
                self.stable_detection_window_size,
                self.auto_search_stable_detection_count,
                self.fine_position_x_tolerance_m,
            )
            rospy.loginfo(
                (
                    "%s：自动识别流程：首次三帧平均位置 -> camera粗对准并等待HOVER -> "
                    "再次三帧平均位置；X误差超限时按XY小步靠近，航向保持不变；"
                    "X误差通过后锁定位置和航向，前移%.2fm后执行投放"
                ),
                NODE_NAME,
                self.pre_drop_forward_distance,
            )
            rospy.loginfo(
                (
                    "%s：自动动作时序：投放前前进%.2fm -> "
                    "右侧闭合 -> 中间闭合%.1fs -> "
                    "中间打开%.1fs -> 打开状态回右侧%.1fs -> "
                    "右侧闭合%.1fs"
                ),
                NODE_NAME,
                self.pre_drop_forward_distance,
                self.hold_seconds,
                self.open_seconds,
                self.return_right_seconds,
                self.close_seconds,
            )
            rospy.loginfo(
                (
                    "%s：投放后离场：启用=%s，任务初始航向=%.1fdeg，"
                    "向左偏移=%.1fdeg，返原点绝对航向=%.1fdeg；"
                    "随后返回map原点(0,0)，每一步到达超时=%.1fs；"
                    "到原点后向NED z=%.2f上浮、持续%.1fs并结束"
                ),
                NODE_NAME,
                "是" if self.post_drop_motion_enabled else "否",
                self.task3_initial_yaw_deg,
                self.return_origin_yaw_offset_deg,
                math.degrees(self.return_origin_target_yaw),
                self.post_drop_step_timeout,
                self.post_drop_ascent_target_z,
                self.post_drop_ascent_seconds,
            )
            rospy.loginfo(
                (
                    "%s：motion_supervisor 判定：状态超时=%.2fs，启动等待=%.1fs，"
                    "当前位置保持等待=%.1fs，目标匹配容差=(水平%.3fm,深度%.3fm,航向%.1fdeg)"
                ),
                NODE_NAME,
                self.motion_state_timeout,
                self.motion_startup_timeout,
                self.hold_timeout,
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
                    "%s：map精对准小步范围=[%.3f,%.3f]m，"
                    "航向固定为进入子任务3时锁存的yaw；"
                    "精确认只用X误差判定，Y参与移动但不作为门槛"
                ),
                NODE_NAME,
                self.auto_visual_min_step_m,
                self.auto_visual_max_step_m,
            )
            rospy.loginfo(
                (
                    "%s：搜索顺序：悬停%.1fs；第一段直行%.2fm后进行快速颜色判定；"
                    "第二段前进%.2fm后左右搜索；"
                    "第三段前进%.2fm后左右搜索"
                ),
                NODE_NAME,
                self.auto_initial_hover_seconds,
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
            rospy.loginfo(
                (
                    "%s：三色方框快速搜索=%s；每个观察位置使用最近%d帧中的"
                    "%d帧一致结论。中间识别到非目标单色后按原步长左移，"
                    "两种非目标颜色可直接推断缺失目标方向；推断失败后继续原搜索路径"
                ),
                NODE_NAME,
                "开启" if self.auto_color_fast_search_enabled else "关闭",
                self.stable_detection_window_size,
                self.auto_search_stable_detection_count,
            )
        else:
            rospy.loginfo(
                "%s：人工模式连续稳定识别=%d帧，动作前确认=%.1fs",
                NODE_NAME,
                self.stable_detection_count,
                self.hold_seconds,
            )
        rospy.loginfo(
            (
                "%s：执行器指令=%s，反馈=%s，mode=%d（2=仅执行器），"
                "夹爪开=%d，夹爪关=%d，方向舵机=%s，右=%d，中=%d"
            ),
            NODE_NAME,
            self.actuator_topic,
            self.actuator_status_topic,
            self.actuator_mode,
            self.clamp_open,
            self.clamp_closed,
            "启用" if self.heading_servo_enabled else "关闭",
            self.heading_servo_right,
            self.heading_servo_center,
        )
        if not self.heading_servo_enabled:
            rospy.logwarn(
                (
                    "%s：方向舵机控制已关闭；投放时跳过移到中间和回到右侧，"
                    "只依据夹爪反馈执行打开、闭合"
                ),
                NODE_NAME,
            )
        rospy.loginfo(
            (
                "%s：执行器固定字段：补光灯=(%d,%d)，"
                "推进电机=(动作%d,转速%d)；颜色灯随目标颜色自动选择"
            ),
            NODE_NAME,
            self.light1,
            self.light2,
            self.drive_cmd,
            self.drive_speed,
        )
        rospy.loginfo(
            (
                "%s：执行器反馈判定：舵机误差<=%d，连续%d帧到位，"
                "反馈超时=%.1fs，单阶段到位超时=%.1fs；"
                "动作保持时间均从反馈确认到位后开始计时"
            ),
            NODE_NAME,
            self.actuator_servo_tolerance,
            self.actuator_feedback_confirm_frames,
            self.actuator_status_timeout,
            self.actuator_stage_timeout,
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

    def actuator_status_callback(self, message):
        now = rospy.Time.now()
        self.actuator_status_sequence += 1
        self.last_actuator_status_time = now
        self.latest_actuator_status = {
            "sequence": self.actuator_status_sequence,
            "heading_servo": int(message.heading_servo),
            "clamp_servo": int(message.clamp_servo),
            "red_light": int(message.red_light),
            "yellow_light": int(message.yellow_light),
            "green_light": int(message.green_light),
        }
        if self.drop_action_started:
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：[执行器硬件状态] 阶段=%s，夹爪=%d，"
                    "颜色灯=(红%d,黄%d,绿%d)，反馈帧#%d"
                ),
                NODE_NAME,
                self.STATE_NAMES.get(self.state, "未知状态"),
                message.clamp_servo,
                message.red_light,
                message.yellow_light,
                message.green_light,
                self.actuator_status_sequence,
            )

        safe_match = self.actuator_values_match(
            message.heading_servo,
            message.clamp_servo,
            self.heading_servo_right,
            self.clamp_closed,
        )
        if safe_match:
            self.actuator_safe_feedback_count = min(
                self.actuator_safe_feedback_count + 1,
                self.actuator_feedback_confirm_frames,
            )
            if (
                not self.actuator_safe_feedback_ready
                and self.actuator_safe_feedback_count
                >= self.actuator_feedback_confirm_frames
            ):
                self.actuator_safe_feedback_ready = True
                if self.heading_servo_enabled:
                    rospy.loginfo(
                        (
                            "%s：[执行器反馈] 初始安全位置已确认："
                            "航向舵机=%d（右），夹爪=%d（闭合），连续%d帧"
                        ),
                        NODE_NAME,
                        message.heading_servo,
                        message.clamp_servo,
                        self.actuator_safe_feedback_count,
                    )
                else:
                    rospy.loginfo(
                        (
                            "%s：[执行器反馈] 初始闭合位置已确认："
                            "夹爪=%d（闭合），方向舵机不参与判断，连续%d帧"
                        ),
                        NODE_NAME,
                        message.clamp_servo,
                        self.actuator_safe_feedback_count,
                    )
        elif not self.actuator_safe_feedback_ready:
            self.actuator_safe_feedback_count = 0

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
        # SAFE只作为普通状态记录，不再触发任务失败或阻止反馈就绪。
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
        if not math.isfinite(self.fixed_depth_m) or self.fixed_depth_m <= 0.0:
            raise ValueError("task3_target_depth_m必须是大于0的有限数")
        if not self.detection_topic:
            raise ValueError("model_detection_topic 不能为空")
        if self.auto_enabled and not self.target_topic:
            raise ValueError("model_target_topic 不能为空")
        if not self.actuator_topic or not self.actuator_status_topic:
            raise ValueError("执行器指令和反馈话题不能为空")
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
        if self.stable_map_position_tolerance_m < 0.0:
            raise ValueError("stable_map_position_tolerance_m 不能小于 0")
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
        if min(
                self.hold_seconds,
                self.open_seconds,
                self.return_right_seconds,
                self.close_seconds) < 0.0:
            raise ValueError("动作持续时间不能小于 0")
        actuator_values = (
            ("clamp_open", self.clamp_open),
            ("clamp_closed", self.clamp_closed),
            ("heading_servo_right", self.heading_servo_right),
            ("heading_servo_center", self.heading_servo_center),
        )
        for name, value in actuator_values:
            if not 0 <= value <= 255:
                raise ValueError("{} 必须在0到255之间".format(name))
        if self.actuator_status_timeout <= 0.0:
            raise ValueError("actuator_status_timeout 必须大于0")
        if self.actuator_feedback_confirm_frames < 1:
            raise ValueError("actuator_feedback_confirm_frames 必须大于等于1")
        if not 0 <= self.actuator_servo_tolerance <= 255:
            raise ValueError("actuator_servo_tolerance 必须在0到255之间")
        if self.actuator_stage_timeout <= 0.0:
            raise ValueError("actuator_stage_timeout 必须大于0")
        if self.target_color not in self.COLOR_LIGHTS:
            raise ValueError("target_color 必须是 yellow、green 或 red")
        if min(self.log_interval, self.warning_log_interval) <= 0.0:
            raise ValueError("日志间隔必须大于 0")

        if not self.auto_enabled:
            return
        if (
            not math.isfinite(self.fine_position_x_tolerance_m)
            or self.fine_position_x_tolerance_m <= 0.0
        ):
            raise ValueError(
                "fine_position_x_tolerance_m 必须是大于0的有限数"
            )
        if (
            not math.isfinite(self.pre_drop_forward_distance)
            or self.pre_drop_forward_distance <= 0.0
        ):
            raise ValueError("pre_drop_forward_distance 必须是大于0的有限数")
        if (
            not math.isfinite(self.pre_drop_forward_timeout)
            or self.pre_drop_forward_timeout <= 0.0
        ):
            raise ValueError("pre_drop_forward_timeout 必须是大于0的有限数")
        if self.post_drop_motion_enabled:
            if (
                not math.isfinite(self.task3_initial_yaw_deg)
                or not math.isfinite(self.return_origin_yaw_offset_deg)
                or self.return_origin_yaw_offset_deg < 0.0
                or self.return_origin_yaw_offset_deg >= 360.0
            ):
                raise ValueError(
                    "task3初始航向必须是有限数，返原点航向左偏角必须在0到360度之间"
                )
            if (
                not math.isfinite(self.post_drop_step_timeout)
                or self.post_drop_step_timeout <= 0.0
            ):
                raise ValueError(
                    "post_drop_step_timeout 必须是大于0的有限数"
                )
            if (
                not math.isfinite(self.post_drop_ascent_seconds)
                or self.post_drop_ascent_seconds <= 0.0
            ):
                raise ValueError(
                    "post_drop_ascent_seconds 必须是大于0的有限数"
                )
            if (
                not math.isfinite(self.post_drop_ascent_target_z)
                or self.post_drop_ascent_target_z >= self.fixed_map_z
            ):
                raise ValueError(
                    "post_drop_ascent_target_z 必须是有限数，且必须小于"
                    "任务运行深度对应的map/NED z"
                )
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
            self.motion_state_topic,
            self.status_topic,
        )
        if not all(topics):
            raise ValueError("motion_supervisor 和状态反馈话题不能为空")
        if min(
            self.motion_state_timeout,
            self.motion_startup_timeout,
            self.hold_timeout,
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
        if (
            self.auto_color_fast_search_enabled
            and self.auto_search_right_distance
            <= self.auto_search_left_distance
        ):
            raise ValueError(
                "开启三色方框快速搜索时，auto_search_right_distance 必须大于 "
                "auto_search_left_distance，才能从中间位置计算右侧定步长"
            )
        if self.auto_initial_hover_seconds < 0.0:
            raise ValueError("auto_initial_hover_seconds 不能小于 0")
        if min(
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
        if self.auto_lateral_sign == 0.0:
            raise ValueError("视觉左右方向符号不能为 0")
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
        if self.auto_center_tolerance_u_px < 0.0:
            raise ValueError("自动左右居中容差不能小于 0")
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

    def get_camera_pose(self, camera_frame, context):
        try:
            self.tf_listener.waitForTransform(
                "map", camera_frame, rospy.Time(0), rospy.Duration(0.5)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                "map", camera_frame, rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：无法读取map -> %s，%s暂停：%s",
                NODE_NAME,
                camera_frame,
                context,
                str(error),
            )
            return None
        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(value) for value in values):
            return None
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def get_base_to_camera_offset(self, camera_frame, context):
        try:
            translation, _ = self.tf_listener.lookupTransform(
                "base_link", camera_frame, rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：无法读取base_link -> %s，%s暂停：%s",
                NODE_NAME,
                camera_frame,
                context,
                str(error),
            )
            return None
        if not all(math.isfinite(value) for value in translation):
            return None
        return translation

    def set_camera_xy_goal(
        self, target_x, target_y, target_yaw, camera_frame, reason
    ):
        offset = self.get_base_to_camera_offset(camera_frame, reason)
        if offset is None:
            return False
        offset_map_x = (
            math.cos(target_yaw) * offset[0]
            - math.sin(target_yaw) * offset[1]
        )
        offset_map_y = (
            math.sin(target_yaw) * offset[0]
            + math.cos(target_yaw) * offset[1]
        )
        goal_x = target_x - offset_map_x
        goal_y = target_y - offset_map_y
        self.start_motion_timeout_clock(reason)
        self.set_active_goal(
            goal_x,
            goal_y,
            self.auto_hold_z,
            target_yaw,
            reason,
        )
        rospy.logwarn(
            (
                "%s：camera xy目标换算：camera_frame=%s，方框map=(%.3f,%.3f)，"
                "base_link->camera偏置map=(%.3f,%.3f)m，保持yaw=%.2fdeg，"
                "下发base_link目标=(%.3f,%.3f)"
            ),
            NODE_NAME,
            camera_frame,
            target_x,
            target_y,
            offset_map_x,
            offset_map_y,
            math.degrees(target_yaw),
            goal_x,
            goal_y,
        )
        return True

    def set_limited_camera_goal(
        self, target_x, target_y, target_yaw, camera_frame, reason
    ):
        camera_pose = self.get_camera_pose(camera_frame, reason)
        if camera_pose is None:
            return False
        error_x = target_x - camera_pose.pose.position.x
        error_y = target_y - camera_pose.pose.position.y
        distance = math.hypot(error_x, error_y)
        if distance <= 1e-6:
            goal_x = target_x
            goal_y = target_y
            step_distance = 0.0
        else:
            step_distance = min(self.auto_visual_max_step_m, distance)
            if step_distance < self.auto_visual_min_step_m:
                step_distance = min(self.auto_visual_min_step_m, distance)
            scale = step_distance / distance
            goal_x = camera_pose.pose.position.x + error_x * scale
            goal_y = camera_pose.pose.position.y + error_y * scale
        if not self.set_camera_xy_goal(
            goal_x,
            goal_y,
            target_yaw,
            camera_frame,
            reason,
        ):
            return False
        rospy.loginfo(
            (
                "%s：方框map精对准小步：camera当前=(%.3f,%.3f)，"
                "目标=(%.3f,%.3f)，XY误差=(%+.3f,%+.3f)m，"
                "本次步长=%.3fm，航向保持%.2fdeg"
            ),
            NODE_NAME,
            camera_pose.pose.position.x,
            camera_pose.pose.position.y,
            target_x,
            target_y,
            error_x,
            error_y,
            step_distance,
            math.degrees(target_yaw),
        )
        return True

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

    def start_motion_timeout_clock(self, reason):
        """首次实际运动目标生成时启动唯一总超时，后续不得重置。"""
        if self.motion_timeout_started_at is not None:
            return
        self.motion_timeout_started_at = rospy.Time.now()
        rospy.logwarn(
            "%s：机器人开始执行运动动作，启动唯一总超时计时：%.1fs；原因=%s",
            NODE_NAME,
            self.max_wait_seconds,
            reason,
        )

    def motion_timeout_elapsed(self):
        if self.motion_timeout_started_at is None:
            return None
        return max(
            0.0,
            (rospy.Time.now() - self.motion_timeout_started_at).to_sec(),
        )

    def set_body_offset_goal(self, current, forward, right, reason):
        self.start_motion_timeout_clock(reason)
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

    def transform_box_target_to_map(self, message):
        source_frame = str(message.pose.header.frame_id).strip()
        stamp = message.pose.header.stamp
        if not source_frame:
            return None, "三维方框位置缺少frame_id"
        if stamp == rospy.Time(0):
            return None, "三维方框位置缺少原始图像时间戳"
        age = (rospy.Time.now() - stamp).to_sec()
        if age < -0.1:
            return None, "三维方框位置时间戳来自未来"
        if age > self.detection_timeout:
            return None, "三维方框位置已过期{:.2f}s".format(age)
        try:
            self.tf_listener.waitForTransform(
                "map", source_frame, stamp, rospy.Duration(1.0)
            )
            transformed = self.tf_listener.transformPose("map", message.pose)
        except tf.Exception as error:
            return None, "原始时间戳map<-{} TF不可用：{}".format(
                source_frame, str(error)
            )
        values = (
            transformed.pose.position.x,
            transformed.pose.position.y,
            transformed.pose.position.z,
        )
        if not all(math.isfinite(value) for value in values):
            return None, "转换后的方框map位置包含无效数值"
        return transformed, ""

    @staticmethod
    def box_position_group_summary(samples):
        mean_x = sum(item["map_x"] for item in samples) / len(samples)
        mean_y = sum(item["map_y"] for item in samples) / len(samples)
        jitter = max(
            math.hypot(
                item["map_x"] - mean_x,
                item["map_y"] - mean_y,
            )
            for item in samples
        )
        return mean_x, mean_y, jitter

    def stable_box_position_groups(self):
        now = rospy.Time.now()
        fresh_samples = [
            item
            for item in self.box_position_samples
            if (now - item["received_time"]).to_sec()
            <= self.detection_timeout
        ]
        self.box_position_samples = fresh_samples[
            -self.stable_detection_window_size:
        ]
        required_count = self.auto_search_stable_detection_count
        if len(self.box_position_samples) < required_count:
            return []
        groups = []
        for group in itertools.combinations(
            self.box_position_samples, required_count
        ):
            _, _, jitter = self.box_position_group_summary(group)
            if jitter <= self.stable_map_position_tolerance_m:
                groups.append(list(group))
        return groups

    def best_box_position_group(self):
        groups = self.stable_box_position_groups()
        if not groups:
            return None
        return min(
            groups,
            key=lambda group: (
                self.box_position_group_summary(group)[2],
                -max(item["frame_index"] for item in group),
            ),
        )

    def reset_box_position_queue(self):
        self.box_position_samples = []
        self.box_fine_candidate = None
        self.box_last_target_time = None

    def add_box_position_sample(self, sample):
        self.box_position_samples.append(sample)
        self.box_position_samples = self.box_position_samples[
            -self.stable_detection_window_size:
        ]
        group = self.best_box_position_group()
        if group is None:
            rospy.loginfo(
                (
                    "%s：[方框map帧#%d] 有效位置写入队列=%d/%d，"
                    "尚未找到相近的%d帧"
                ),
                NODE_NAME,
                sample["frame_index"],
                len(self.box_position_samples),
                self.stable_detection_window_size,
                self.auto_search_stable_detection_count,
            )
            return

        mean_x, mean_y, jitter = self.box_position_group_summary(group)
        candidate = {
            "map_x": mean_x,
            "map_y": mean_y,
            "jitter": jitter,
            "camera_frame": group[-1]["camera_frame"],
            "frame_ids": [item["frame_index"] for item in group],
            "received_time": group[-1]["received_time"],
        }
        rospy.loginfo(
            (
                "%s：[方框map帧#%d] 三帧位置确认通过：队列=%d/%d，"
                "命中帧=%s，平均map=(%.3f,%.3f)，抖动=%.3f/%.3fm，阶段=%s"
            ),
            NODE_NAME,
            sample["frame_index"],
            len(self.box_position_samples),
            self.stable_detection_window_size,
            candidate["frame_ids"],
            mean_x,
            mean_y,
            jitter,
            self.stable_map_position_tolerance_m,
            self.STATE_NAMES.get(self.state, "未知状态"),
        )
        if self.state == self.WAIT_FOR_TARGET:
            self.box_coarse_map_x = mean_x
            self.box_coarse_map_y = mean_y
            self.box_coarse_camera_frame = candidate["camera_frame"]
            self.box_position_lock_ready = True
        elif self.state == self.AUTO_APPROACH and self.box_recheck_collecting:
            self.box_fine_candidate = candidate
            self.box_recheck_collecting = False

    def rectangle_health_callback(self, _message):
        """只用持续发布的二维检测话题判断方框模型是否在线。"""
        self.last_model_message_time = rospy.Time.now()

    def rectangle_target_callback(self, message):
        now = rospy.Time.now()
        self.box_last_target_time = now
        self.box_map_frame_index += 1
        frame_index = self.box_map_frame_index
        if self.state not in (
            self.WAIT_FOR_TARGET,
            self.AUTO_HOVER_CONFIRM,
            self.AUTO_APPROACH,
        ):
            return
        if self.state == self.WAIT_FOR_TARGET and self.auto_search_index == 0:
            return
        if self.state == self.AUTO_HOVER_CONFIRM:
            return
        if self.state == self.AUTO_APPROACH and not self.box_recheck_collecting:
            return
        class_name = str(message.class_name).strip()
        confidence = self.finite_number(message.conf)
        if self.detection_color(class_name) != self.target_color:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：[方框map帧#%d] 忽略类别%s，目标颜色=%s",
                NODE_NAME,
                frame_index,
                class_name or "空",
                self.target_color,
            )
            return
        if confidence is None or confidence < self.min_confidence:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：[方框map帧#%d] 忽略置信度%s，最低=%.2f",
                NODE_NAME,
                frame_index,
                "无效" if confidence is None else "{:.3f}".format(confidence),
                self.min_confidence,
            )
            return
        transformed, reason = self.transform_box_target_to_map(message)
        if transformed is None:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：[方框map帧#%d] 无效：%s",
                NODE_NAME,
                frame_index,
                reason,
            )
            return
        if (
            self.box_position_samples
            and (now - self.box_position_samples[-1]["received_time"]).to_sec()
            > self.detection_timeout
        ):
            self.reset_box_position_queue()
            rospy.logwarn(
                "%s：[方框map] 消息间隔超过%.2fs，清空过期位置队列",
                NODE_NAME,
                self.detection_timeout,
            )
        source = message.pose.pose.position
        target = transformed.pose.position
        sample = {
            "frame_index": frame_index,
            "received_time": now,
            "source_stamp": message.pose.header.stamp,
            "source_stamp_sec": message.pose.header.stamp.to_sec(),
            "confidence": confidence,
            "camera_frame": str(message.pose.header.frame_id).strip(),
            "camera_x": float(source.x),
            "camera_y": float(source.y),
            "camera_z": float(source.z),
            "map_x": float(target.x),
            "map_y": float(target.y),
            "map_z": float(target.z),
        }
        self.add_box_position_sample(sample)

    def start_pre_drop_forward(self, reason):
        if self.active_goal is None or self.auto_hold_yaw is None:
            return False
        source_goal = copy.deepcopy(self.active_goal)
        self.pending_drop_reason = str(reason)
        self.auto_action_hold_position = None
        self.set_body_offset_goal(
            source_goal,
            self.pre_drop_forward_distance,
            0.0,
            "开灯和开爪前沿当前航向前进%.2fm"
            % self.pre_drop_forward_distance,
        )
        rospy.loginfo(
            (
                "%s：[投放前前进] 识别和对准已通过，先沿当前航向"
                "前进%.2fm，目标=(%.3f,%.3f,%.3f)"
            ),
            NODE_NAME,
            self.pre_drop_forward_distance,
            self.active_goal.pose.position.x,
            self.active_goal.pose.position.y,
            self.active_goal.pose.position.z,
        )
        self.set_state(
            self.PRE_DROP_FORWARD,
            "投放动作前固定前进目标已发布",
        )
        return True

    def handle_pre_drop_forward(self):
        self.publish_actuator(self.clamp_closed, "off")
        if self.motion_step_timed_out(
            "投放前固定前进",
            self.pre_drop_forward_timeout,
        ):
            return
        if self.motion_arrived():
            if not self.capture_action_hold_position():
                self.finish_task(False, "投放前前进到达，但无法锁定动作定点")
                return
            reason = self.pending_drop_reason
            self.pending_drop_reason = ""
            self.begin_drop_actuator_action(
                "%s；投放前已前进%.2fm并进入HOVER"
                % (reason, self.pre_drop_forward_distance)
            )
            return
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：[投放前前进] 进行中 %.1f/%.1fs，motion=%s",
            NODE_NAME,
            self.state_elapsed(),
            self.pre_drop_forward_timeout,
            self.current_motion_state_name(),
        )
        self.log_arrival_gate("投放前固定前进到达判定")

    def start_post_drop_turn(self):
        source_goal = self.active_goal
        if source_goal is None:
            source_goal = self.get_current_pose("生成投放后左转目标")
        if source_goal is None:
            return False

        start_yaw = yaw_from_quaternion(source_goal.pose.orientation)
        self.post_drop_target_yaw = self.return_origin_target_yaw
        self.set_active_goal(
            source_goal.pose.position.x,
            source_goal.pose.position.y,
            self.auto_hold_z,
            self.post_drop_target_yaw,
            "投放完成后原地对准返原点绝对航向%.1f度"
            % math.degrees(self.post_drop_target_yaw),
        )
        rospy.loginfo(
            (
                "%s：[投放后离场] 开始原地对准返原点绝对航向，"
                "任务初始航向=%.1fdeg，向左偏移=%.1fdeg，"
                "保持位置=(%.3f,%.3f,%.3f)，当前航向=%.1fdeg -> 目标航向=%.1fdeg"
            ),
            NODE_NAME,
            self.task3_initial_yaw_deg,
            self.return_origin_yaw_offset_deg,
            self.active_goal.pose.position.x,
            self.active_goal.pose.position.y,
            self.active_goal.pose.position.z,
            math.degrees(start_yaw),
            math.degrees(self.post_drop_target_yaw),
        )
        self.set_state(
            self.POST_DROP_TURN,
            "夹爪关闭并熄灯后开始离场左转",
        )
        return True

    def start_post_drop_return_origin(self):
        if self.post_drop_target_yaw is None:
            return False
        self.auto_hold_yaw = self.post_drop_target_yaw
        self.set_active_goal(
            POST_DROP_ORIGIN_X,
            POST_DROP_ORIGIN_Y,
            self.auto_hold_z,
            self.post_drop_target_yaw,
            "投放后左转完成，返回map原点(0,0)",
        )
        rospy.loginfo(
            (
                "%s：[投放后离场] 左转已由HOVER确认，"
                "开始返回map原点，目标=(%.3f,%.3f,%.3f)"
            ),
            NODE_NAME,
            self.active_goal.pose.position.x,
            self.active_goal.pose.position.y,
            self.active_goal.pose.position.z,
        )
        self.set_state(
            self.POST_DROP_RETURN_ORIGIN,
            "左转目标已到达，开始返回map原点",
        )
        return True

    def start_post_drop_ascent(self):
        if self.post_drop_target_yaw is None or self.auto_hold_z is None:
            return False
        self.set_active_goal(
            POST_DROP_ORIGIN_X,
            POST_DROP_ORIGIN_Y,
            self.post_drop_ascent_target_z,
            self.post_drop_target_yaw,
            "到达map原点后向NED z=%.2f上浮"
            % self.post_drop_ascent_target_z,
        )
        rospy.loginfo(
            (
                "%s：[投放后离场] map原点已由HOVER确认，"
                "NED z向下为正，从z=%.2f向目标z=%.2f上浮，"
                "持续发布%.1fs"
            ),
            NODE_NAME,
            self.auto_hold_z,
            self.post_drop_ascent_target_z,
            self.post_drop_ascent_seconds,
        )
        self.set_state(
            self.POST_DROP_ASCEND,
            "已到达map原点，开始持续上浮",
        )
        return True

    def motion_step_timed_out(self, step_name, timeout):
        elapsed = self.state_elapsed()
        if elapsed < timeout:
            return False
        self.finish_task(
            False,
            "%s超过%.1fs仍未到达HOVER"
            % (step_name, timeout),
        )
        return True

    def handle_post_drop_turn(self):
        self.publish_actuator(self.clamp_closed, "off")
        if self.motion_step_timed_out(
            "投放后左转",
            self.post_drop_step_timeout,
        ):
            return
        if self.motion_arrived():
            if not self.start_post_drop_return_origin():
                self.finish_task(False, "无法生成投放后返回map原点目标")
            return
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：[投放后离场] 左转进行中 %.1f/%.1fs，motion=%s",
            NODE_NAME,
            self.state_elapsed(),
            self.post_drop_step_timeout,
            self.current_motion_state_name(),
        )
        self.log_arrival_gate("投放后左转到达判定")

    def handle_post_drop_return_origin(self):
        self.publish_actuator(self.clamp_closed, "off")
        if self.motion_step_timed_out(
            "投放后返回map原点",
            self.post_drop_step_timeout,
        ):
            return
        if self.motion_arrived():
            if not self.start_post_drop_ascent():
                self.finish_task(False, "无法生成map原点上浮目标")
            return
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：[投放后离场] 返回map原点进行中 %.1f/%.1fs，motion=%s",
            NODE_NAME,
            self.state_elapsed(),
            self.post_drop_step_timeout,
            self.current_motion_state_name(),
        )
        self.log_arrival_gate("投放后返回map原点到达判定")

    def handle_post_drop_ascent(self):
        self.publish_actuator(self.clamp_closed, "off")
        elapsed = self.state_elapsed()
        if elapsed >= self.post_drop_ascent_seconds:
            self.finish_task(
                True,
                (
                    "识别和投放完成，对准返原点绝对航向%.1f度、返回map原点(0,0)，"
                    "向NED z=%.2f上浮、持续%.1fs后结束"
                )
                % (
                    math.degrees(self.post_drop_target_yaw),
                    self.post_drop_ascent_target_z,
                    self.post_drop_ascent_seconds,
                ),
            )
            return
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：[投放后离场] 原点上浮进行中 %.1f/%.1fs，目标z=%.2f",
            NODE_NAME,
            elapsed,
            self.post_drop_ascent_seconds,
            self.active_goal.pose.position.z,
        )

    def initialize_auto_pose(self):
        if not self.auto_enabled:
            return True
        if self.auto_hold_z is not None and self.auto_hold_yaw is not None:
            return True

        status = self.get_recent_status("初始化固定悬停点")
        current = self.get_current_pose("初始化固定悬停点")
        if status is None or current is None:
            return False

        self.auto_hold_z = self.fixed_map_z
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
                "统一固定深度=%.3fm，启动传感器深度=%.3fm，启动TF z=%.3f；"
                "后续不会跟随漂移位置更新"
            ),
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            self.auto_hold_z,
            math.degrees(self.auto_hold_yaw),
            self.fixed_depth_m,
            self.status_hold_depth,
            current.pose.position.z,
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
        return True

    def request_motion_hold(
        self,
        reason,
        discard_search_resume=False,
        force_refresh=False,
    ):
        if not self.auto_enabled or self.goal_pub is None:
            return False
        if discard_search_resume:
            self.auto_search_resume_goal = None
            self.auto_search_paused_for_model = False
        if force_refresh:
            self.motion_hold_requested_at = None
        if self.motion_hold_requested_at is None:
            current = self.get_current_pose("生成当前位置保持目标")
            if current is None:
                return False
            current_yaw = yaw_from_quaternion(current.pose.orientation)
            self.set_active_goal(
                current.pose.position.x,
                current.pose.position.y,
                current.pose.position.z,
                current_yaw,
                "当前位置保持：{}".format(reason),
            )
            self.motion_hold_requested_at = rospy.Time.now()
            rospy.logwarn(
                "%s：不再发布 /cmd/motion/cancel；已下发当前位置保持目标，"
                "等待 motion_supervisor 进入 HOVER；原因=%s",
                NODE_NAME,
                reason,
            )
        self.motion_hold_reason = reason
        return True

    def hold_has_completed(self):
        if (
            self.motion_hold_requested_at is None
            or not self.motion_arrived()
        ):
            return False
        return (
            self.latest_motion_state.header.stamp
            >= self.motion_hold_requested_at
        )

    def wait_for_motion_hold(self, context):
        if self.motion_hold_requested_at is None:
            return True
        elapsed = (
            rospy.Time.now() - self.motion_hold_requested_at
        ).to_sec()
        if elapsed >= self.hold_timeout:
            self.finish_task(
                False,
                "等待当前位置保持目标进入 HOVER 超时：{}".format(
                    self.motion_hold_reason
                ),
            )
            return False
        if not self.hold_has_completed():
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：%s，等待当前位置保持目标进入 HOVER %.1f/%.1fs",
                NODE_NAME,
                context,
                elapsed,
                self.hold_timeout,
            )
            return False

        reason = self.motion_hold_reason
        self.motion_hold_requested_at = None
        self.motion_hold_reason = ""
        rospy.loginfo(
            "%s：当前位置保持目标已进入 HOVER，原因为：%s",
            NODE_NAME,
            reason,
        )
        return True

    def action_status_is_stable(self, status):
        return (
            self.status_pose_errors(status) is not None
            and self.motion_arrived()
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

    @staticmethod
    def is_color_observation_step(step_kind):
        return step_kind.startswith("observe_")

    def current_search_step_kind(self):
        if self.auto_search_index >= len(self.auto_search_plan):
            return None
        return self.auto_search_plan[self.auto_search_index][0]

    def color_observation_is_ready(self):
        step_kind = self.current_search_step_kind()
        return (
            self.auto_enabled
            and self.state == self.WAIT_FOR_TARGET
            and self.auto_color_fast_search_enabled
            and step_kind is not None
            and self.is_color_observation_step(step_kind)
            and self.auto_search_step_started is not None
        )

    def replace_search_plan_after_observation(self, steps, reason):
        completed_prefix = self.auto_search_plan[
            : self.auto_search_index + 1
        ]
        self.auto_search_plan = completed_prefix + list(steps)
        self.auto_search_index += 1
        self.reset_auto_search_step()
        self.reset_stability()
        rospy.loginfo(
            "%s：快速颜色搜索决定下一段路径：%s；后续步骤=%s",
            NODE_NAME,
            reason,
            " -> ".join(
                self.search_step_description(kind, amount)
                for kind, amount in steps
            ) if steps else "保持当前定点",
        )

    def start_fallback_search(self, observation_step, reason):
        if observation_step == "observe_center":
            fallback_steps = [
                ("left", self.auto_search_left_distance),
                ("right", self.auto_search_right_distance),
            ] + self.fallback_search_tail
        elif observation_step in (
            "observe_left",
            "observe_inferred_left",
        ):
            fallback_steps = [
                ("right", self.auto_search_right_distance),
            ] + self.fallback_search_tail
        elif observation_step == "observe_inferred_right":
            fallback_steps = [
                ("left", self.auto_search_right_distance),
                ("right", self.auto_search_right_distance),
            ] + self.fallback_search_tail
        else:
            fallback_steps = list(self.fallback_search_tail)

        self.smart_search_first_color = None
        self.replace_search_plan_after_observation(
            fallback_steps,
            "快速判定未找到目标：%s；从当前位置接回原有定步长搜索"
            % reason,
        )

    def schedule_smart_search_direction(self, direction, source_step, reason):
        if direction == "left":
            steps = [
                ("left", self.auto_search_left_distance),
                ("observe_inferred_left", 0.0),
            ]
        elif source_step == "observe_center":
            center_to_right = (
                self.auto_search_right_distance
                - self.auto_search_left_distance
            )
            steps = [
                ("right", center_to_right),
                ("observe_inferred_right", 0.0),
            ]
        else:
            steps = [
                ("right", self.auto_search_right_distance),
                ("observe_inferred_right", 0.0),
            ]
        self.replace_search_plan_after_observation(steps, reason)

    def handle_color_observation_step(self, model_ready, step_kind):
        if not model_ready:
            self.pause_search_for_model()
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：%s等待模型恢复，当前定点保持不动",
                NODE_NAME,
                self.SEARCH_STEP_NAMES[step_kind],
            )
            return
        if not self.resume_search_after_model_ready():
            return
        if self.state != self.WAIT_FOR_TARGET:
            return

        if self.auto_search_step_started is None:
            if not self.motion_arrived():
                rospy.loginfo_throttle(
                    self.log_interval,
                    "%s：等待HOVER后开始%s",
                    NODE_NAME,
                    self.SEARCH_STEP_NAMES[step_kind],
                )
                self.log_arrival_gate(
                    "%s到达判定" % self.SEARCH_STEP_NAMES[step_kind]
                )
                return
            self.reset_stability()
            self.auto_search_step_started = rospy.Time.now()
            rospy.loginfo(
                (
                    "%s：已到达HOVER，开始%s；清空移动途中的旧帧，"
                    "按最近%d帧中%d帧一致进行颜色判定"
                ),
                NODE_NAME,
                self.SEARCH_STEP_NAMES[step_kind],
                self.stable_detection_window_size,
                self.auto_search_stable_detection_count,
            )
            return

        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：%s进行中：颜色场景窗口=%d/%d帧，"
                "目标颜色=%s，首次正面颜色=%s"
            ),
            NODE_NAME,
            self.SEARCH_STEP_NAMES[step_kind],
            len(self.color_scene_frame_window),
            self.stable_detection_window_size,
            self.target_color,
            self.smart_search_first_color or "尚未确认",
        )

    def search_step_offsets(self, step_kind, step_amount):
        if step_kind == "forward":
            return step_amount, 0.0
        if step_kind == "left":
            return 0.0, -step_amount
        if step_kind == "right":
            return 0.0, step_amount
        return 0.0, 0.0

    def search_step_description(self, step_kind, step_amount):
        if step_kind == "hover":
            return "启动悬停 %.2fs" % step_amount
        if self.is_color_observation_step(step_kind):
            return self.SEARCH_STEP_NAMES[step_kind]
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
                "%s：本轮预设搜索路径已经执行完毕，仍未稳定识别方框；"
                "保持最后搜索点继续识别，不提前结束，等待唯一总超时",
                NODE_NAME,
            )

    def pause_search_for_model(self):
        if not self.auto_search_paused_for_model:
            if self.auto_search_step_goal is not None:
                self.auto_search_resume_goal = copy.deepcopy(
                    self.auto_search_step_goal
                )
            self.auto_search_paused_for_model = True
        if self.motion_hold_requested_at is None:
            self.request_motion_hold(
                "模型话题未就绪或已超时，暂停当前搜索位移"
            )
        self.wait_for_motion_hold("模型不可用，搜索暂停")

    def resume_search_after_model_ready(self):
        if not self.auto_search_paused_for_model:
            return True
        if not self.wait_for_motion_hold("等待模型恢复前先完成当前位置保持"):
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

    def begin_box_coarse_camera_alignment(self):
        if (
            not self.box_position_lock_ready
            or self.box_coarse_map_x is None
            or self.box_coarse_map_y is None
            or not self.box_coarse_camera_frame
        ):
            return False
        if not self.set_camera_xy_goal(
            self.box_coarse_map_x,
            self.box_coarse_map_y,
            self.auto_hold_yaw,
            self.box_coarse_camera_frame,
            "首次三帧方框map位置通过，camera xy粗对准",
        ):
            return False
        self.box_position_lock_ready = False
        self.box_precision_goal_pending = False
        self.box_final_goal_pending = False
        self.box_recheck_collecting = False
        self.hover_confirmation_hover_at = None
        self.reset_box_position_queue()
        self.set_state(
            self.AUTO_HOVER_CONFIRM,
            "首次稳定方框map位置已冻结，等待camera粗对准目标HOVER",
        )
        return True

    def confirm_box_after_coarse_hover(self):
        if self.state != self.AUTO_HOVER_CONFIRM:
            return
        if self.state_elapsed() >= self.hold_timeout:
            self.finish_task(False, "camera粗对准方框目标未在规定时间进入HOVER")
            return
        if not self.motion_arrived():
            self.log_arrival_gate("等待camera粗对准方框目标HOVER")
            return
        if self.auto_hover_confirm_settle_seconds > 0.0:
            if self.hover_confirmation_hover_at is None:
                self.hover_confirmation_hover_at = rospy.Time.now()
                rospy.loginfo(
                    "%s：camera粗对准已进入HOVER，先稳定%.2fs再重新识别方框",
                    NODE_NAME,
                    self.auto_hover_confirm_settle_seconds,
                )
                return
            if (
                rospy.Time.now() - self.hover_confirmation_hover_at
            ).to_sec() < self.auto_hover_confirm_settle_seconds:
                return
        self.reset_box_position_queue()
        self.box_recheck_collecting = True
        self.hover_confirmation_hover_at = None
        self.set_state(
            self.AUTO_APPROACH,
            "camera粗对准已HOVER，清空旧帧并开始三帧方框精确认",
        )

    def finish_box_position_alignment(self, candidate):
        self.box_final_map_x = candidate["map_x"]
        self.box_final_map_y = candidate["map_y"]
        self.box_final_camera_frame = candidate["camera_frame"]
        if not self.set_camera_xy_goal(
            self.box_final_map_x,
            self.box_final_map_y,
            self.auto_hold_yaw,
            self.box_final_camera_frame,
            "方框X误差通过，锁定最终位置和固定航向",
        ):
            return False
        self.box_final_goal_pending = True
        self.box_recheck_collecting = False
        self.reset_box_position_queue()
        rospy.logwarn(
            (
                "%s：方框精确认完成：锁定map=(%.3f,%.3f)，"
                "X误差门槛=%.3fm，航向固定=%.2fdeg；"
                "最终位置HOVER后前移%.2fm"
            ),
            NODE_NAME,
            self.box_final_map_x,
            self.box_final_map_y,
            self.fine_position_x_tolerance_m,
            math.degrees(self.auto_hold_yaw),
            self.pre_drop_forward_distance,
        )
        return True

    def approach_box_by_map(self):
        if self.state != self.AUTO_APPROACH:
            return
        if self.state_elapsed() >= self.hold_timeout:
            self.finish_task(False, "方框map精对准目标未在规定时间进入HOVER")
            return
        if self.box_final_goal_pending:
            if not self.motion_arrived():
                self.log_arrival_gate("等待方框最终固定位置HOVER")
                return
            if not self.require_safe_actuator_feedback(
                "方框最终位置HOVER后的投放动作放行"
            ):
                self.publish_actuator(
                    self.clamp_closed,
                    "off",
                    self.heading_servo_right,
                )
                return
            self.box_final_goal_pending = False
            if not self.start_pre_drop_forward(
                "方框X误差已通过，最终位置和航向已锁定"
            ):
                self.finish_task(False, "无法生成方框投放前20厘米前进目标")
            return
        if self.box_precision_goal_pending:
            if not self.motion_arrived():
                self.log_arrival_gate("等待方框map精对准小步HOVER")
                return
            self.box_precision_goal_pending = False
            self.box_recheck_collecting = True
            self.reset_box_position_queue()
            rospy.loginfo(
                "%s：方框map精对准小步已HOVER，重新累计三帧位置",
                NODE_NAME,
            )
            return
        if self.box_fine_candidate is None:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：方框精确认中：等待%d帧相近map位置，队列=%d/%d",
                NODE_NAME,
                self.auto_search_stable_detection_count,
                len(self.box_position_samples),
                self.stable_detection_window_size,
            )
            return

        candidate = self.box_fine_candidate
        self.box_fine_candidate = None
        x_error = candidate["map_x"] - self.box_coarse_map_x
        rospy.loginfo(
            (
                "%s：方框三帧精确认候选：map=(%.3f,%.3f)，"
                "相对首次点误差=(%+.3f,%+.3f)m，X门槛=%.3fm"
            ),
            NODE_NAME,
            candidate["map_x"],
            candidate["map_y"],
            x_error,
            candidate["map_y"] - self.box_coarse_map_y,
            self.fine_position_x_tolerance_m,
        )
        if abs(x_error) <= self.fine_position_x_tolerance_m:
            if not self.finish_box_position_alignment(candidate):
                self.finish_task(False, "无法生成方框最终固定位置目标")
            return
        if not self.set_limited_camera_goal(
            candidate["map_x"],
            candidate["map_y"],
            self.auto_hold_yaw,
            candidate["camera_frame"],
            "方框X误差超限，保持航向按XY小步靠近",
        ):
            return
        self.box_precision_goal_pending = True
        self.box_recheck_collecting = False
        self.reset_box_position_queue()
        self.set_state(
            self.AUTO_APPROACH,
            "方框X误差超过门槛，已下发一个XY精对准小步",
        )

    def search_target_automatically(self, model_ready):
        self.auto_centered_frame_count = 0
        if self.state != self.WAIT_FOR_TARGET:
            return
        if not self.wait_for_motion_hold("搜索阶段等待当前位置保持"):
            return
        if self.box_position_lock_ready:
            self.begin_box_coarse_camera_alignment()
            return
        if self.auto_search_index >= len(self.auto_search_plan):
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：预设搜索路径已完成，保持最后搜索点继续识别%s方框；"
                "不提前失败、不恢复、不重试，等待唯一总超时 %.1fs",
                NODE_NAME,
                self.target_color,
                self.max_wait_seconds,
            )
            return

        step_kind, step_amount = self.auto_search_plan[self.auto_search_index]
        if self.is_color_observation_step(step_kind):
            self.handle_color_observation_step(model_ready, step_kind)
            return
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
                "%s：模型话题未就绪，搜索步骤 %d/%d 已下发当前位置保持目标暂停",
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

    def horizontal_edge_state(self, detection):
        image_width = detection.get("image_width", self.auto_image_width)
        edge_margin = max(2.0, image_width * 0.01)
        touches_left = detection["x1"] <= edge_margin
        touches_right = detection["x2"] >= image_width - edge_margin
        # 只贴一侧说明目标仍横向偏出画面；两侧都不贴或同时贴住时，
        # 水平方向是对称的，仍可继续使用中心误差判断。
        return (
            touches_left == touches_right,
            touches_left,
            touches_right,
            edge_margin,
        )

    def select_auto_tracking_candidate(self, candidates):
        if self.current_auto_target is None:
            return max(candidates, key=lambda item: item["confidence"])

        previous = self.current_auto_target
        image_width = previous.get("image_width", self.auto_image_width)
        max_center_change = max(
            self.stable_center_tolerance_px * 2.0,
            image_width * 0.125,
        )
        max_area_change = min(
            1.0,
            max(self.stable_area_tolerance_ratio * 2.0, 0.70),
        )
        matches = []
        for candidate in candidates:
            center_change = abs(
                candidate["center_u"] - previous["center_u"]
            )
            area_change = self.area_change_ratio(
                candidate["area"], previous["area"]
            )
            if (
                center_change <= max_center_change
                and area_change <= max_area_change
            ):
                matches.append((center_change, area_change, candidate))

        if not matches:
            rospy.logwarn(
                (
                    "%s：本帧同色候选均与上一跟踪目标不连续，"
                    "不切换目标：上一中心u=%.1f，允许变化<=%.1fpx，"
                    "允许面积变化<=%.2f"
                ),
                NODE_NAME,
                previous["center_u"],
                max_center_change,
                max_area_change,
            )
            return None

        _, _, selected = min(
            matches,
            key=lambda item: (
                item[0],
                item[1],
                -item[2]["confidence"],
            ),
        )
        return selected

    def append_auto_tracking_detection(self, detection):
        self.auto_tracking_frame_window.append(dict(detection))
        window_size = max(self.auto_center_stable_detection_count, 3)
        self.auto_tracking_frame_window = self.auto_tracking_frame_window[
            -window_size:
        ]

        smoothed = dict(detection)
        smoothed["center_u"] = statistics.median(
            item["center_u"] for item in self.auto_tracking_frame_window
        )
        smoothed["center_v"] = statistics.median(
            item["center_v"] for item in self.auto_tracking_frame_window
        )
        smoothed["area"] = statistics.median(
            item["area"] for item in self.auto_tracking_frame_window
        )
        return smoothed

    def record_auto_tracking_miss(self, frame_index, reason):
        self.auto_tracking_frame_window = []
        self.auto_tracking_waiting_for_fresh_frame = True
        self.reset_auto_center_stability(reason)
        rospy.loginfo(
            (
                "%s：[模型帧 #%d] 自动跟踪帧无效：%s；"
                "清空最终连续帧，但保留上一小步目标，短暂丢帧不立即改保持目标"
            ),
            NODE_NAME,
            frame_index,
            reason,
        )

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
                self.visual_stop_locked = self.request_motion_hold(
                    "/status/auv 不可用，暂停方框细对准"
                )
            self.reset_auto_center_stability("/status/auv 不可用或超时")
            return
        if not self.wait_for_motion_hold("细对准等待当前位置保持"):
            return

        target_age = None
        if self.last_target_time is not None:
            target_age = (now - self.last_target_time).to_sec()
        if target_age is not None and target_age > self.detection_timeout:
            self.current_auto_target = None
            self.auto_tracking_frame_window = []
            if not self.visual_stop_locked:
                self.visual_stop_locked = self.request_motion_hold(
                    "目标识别结果超时，停止细对准",
                    discard_search_resume=True,
                )
            if not self.wait_for_motion_hold("目标超时后等待当前位置保持"):
                return
            self.reset_auto_center_stability("目标识别结果超时")
            self.reset_stability()
            self.reset_auto_search_step()
            self.set_state(self.WAIT_FOR_TARGET, "目标丢失超时，重新执行当前搜索步骤")
            return

        target = self.current_auto_target
        if target is None:
            if not self.visual_stop_locked:
                self.visual_stop_locked = self.request_motion_hold(
                    "当前模型帧未识别到目标，停止水平运动",
                    discard_search_resume=True,
                )
            self.reset_auto_center_stability("当前模型帧未识别到目标")
            return
        if self.auto_tracking_waiting_for_fresh_frame:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：等待新的有效方框帧，保留当前小步目标但不发布新横移目标",
                NODE_NAME,
            )
            return

        self.visual_stop_locked = False
        error_u_px, error_v_px, normalized_u, _ = (
            self.auto_target_errors(target)
        )
        (
            edge_balanced,
            touches_left,
            touches_right,
            edge_margin,
        ) = self.horizontal_edge_state(target)
        centered = (
            abs(error_u_px) <= self.auto_center_tolerance_u_px
            and edge_balanced
        )
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
                right_step = self.visual_step(
                    normalized_u,
                    self.auto_visual_lateral_gain_m,
                    self.auto_lateral_sign,
                )
                self.set_body_offset_goal(
                    current,
                    forward_step,
                    right_step,
                    "依据方框水平中心生成左右细对准小步，保持锁存航向",
                )
                self.last_visual_goal_frame = frame_index
                self.last_visual_goal_time = now
                rospy.loginfo(
                    (
                        "%s：[模型帧 #%d] 横向细对准小步已发布："
                        "左右误差u=%+.1fpx，纵向误差v=%+.1fpx仅观察，"
                        "单侧贴边=(左%s,右%s,边界%.1fpx)，"
                        "本体偏置=(前%+.3f,右%+.3f)m，航向保持不变"
                    ),
                    NODE_NAME,
                    frame_index,
                    error_u_px,
                    error_v_px,
                    "是" if touches_left else "否",
                    "是" if touches_right else "否",
                    edge_margin,
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
                "方框左右进入容差，锁定位置和航向等待HOVER",
            )
            self.visual_center_hold_requested = True

        pose_errors = self.status_pose_errors(status)
        depth_error = 0.0 if pose_errors is None else pose_errors[0]
        yaw_error_deg = 0.0 if pose_errors is None else pose_errors[1]
        message = self.latest_motion_state
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：横向细对准：中心=(%.1f,%.1f)，左右误差u=%+.1fpx，"
                "纵向误差v=%+.1fpx仅观察，单侧贴边=%s，"
                "motion=%s，base误差=%.3fm，水平速度=%.3fm/s，"
                "航向角速度=%+.2fdeg/s，mode=%d，深度误差=%+.3fm，"
                "航向误差=%+.2fdeg"
            ),
            NODE_NAME,
            target["center_u"],
            target["center_v"],
            error_u_px,
            error_v_px,
            "否" if edge_balanced else "是",
            self.MOTION_STATE_NAMES.get(message.state, "未知"),
            message.base_position_error,
            message.horizontal_speed,
            math.degrees(message.yaw_rate),
            status["control_mode"],
            depth_error,
            yaw_error_deg,
        )

        if not centered:
            self.reset_auto_center_stability("方框左右中心超出允许范围")
            return
        if (
            self.auto_centered_frame_count
            < self.auto_center_stable_detection_count
        ):
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：方框左右已进入中心范围，等待连续居中识别 %d/%d 帧",
                NODE_NAME,
                self.auto_centered_frame_count,
                self.auto_center_stable_detection_count,
            )
            return
        if not self.action_status_is_stable(status):
            self.log_arrival_gate("方框左右居中后的动作放行到达判定")
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

        if not self.require_safe_actuator_feedback("自动模式动作放行"):
            self.publish_actuator(
                self.clamp_closed,
                "off",
                self.heading_servo_right,
            )
            return
        if not self.capture_action_hold_position():
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：无法记录最终定点位置，暂不执行夹爪动作",
                NODE_NAME,
            )
            return
        self.start_drop_action(
            (
                "方框左右连续居中 {} 帧、航向保持且motion_supervisor已稳定HOVER"
                .format(self.auto_center_stable_detection_count)
            )
        )

    def confirm_target_after_hover(self):
        if self.state != self.AUTO_HOVER_CONFIRM:
            return
        if not self.wait_for_motion_hold("首次识别后等待当前位置悬停"):
            return

        now = rospy.Time.now()
        if self.hover_confirmation_hover_at is None:
            self.hover_confirmation_hover_at = now
            self.reset_stability()
            self.current_auto_target = None
            rospy.loginfo(
                (
                    "%s：motion_supervisor 已完成当前位置保持并进入HOVER；"
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
        return self.detection_color(class_name) == self.target_color

    def detection_color(self, class_name):
        normalized = self.normalize_label(class_name)
        label_parts = normalized.split("_")
        for color in self.COLOR_LIGHTS:
            if color == "off":
                continue
            if normalized == color or color in label_parts:
                return color
        return None

    def build_color_scene_observation(
        self,
        detections,
        step_kind,
    ):
        by_color = {}
        for detection in detections:
            if detection["confidence"] < self.min_confidence:
                continue
            color = self.detection_color(detection["class_name"])
            if color is None or color == self.target_color:
                continue
            previous = by_color.get(color)
            if (
                previous is None
                or detection["confidence"] > previous["confidence"]
            ):
                by_color[color] = detection

        if not by_color:
            return None

        first_detection = next(iter(by_color.values()))
        image_width = first_detection.get(
            "image_width", self.auto_image_width
        )
        desired_u = image_width * self.auto_target_center_u_ratio
        visible_colors = tuple(sorted(by_color))

        if step_kind == "observe_center":
            center_candidates = [
                (color, detection)
                for color, detection in by_color.items()
                if detection["x1"] <= desired_u <= detection["x2"]
            ]
            if not center_candidates:
                return None
            front_color, front = min(
                center_candidates,
                key=lambda item: abs(item[1]["center_u"] - desired_u),
            )
            if len(by_color) == 1:
                return {
                    "key": ("single", front_color),
                    "kind": "single",
                    "front_color": front_color,
                    "visible_colors": visible_colors,
                }

            other_color, other = min(
                (
                    (color, detection)
                    for color, detection in by_color.items()
                    if color != front_color
                ),
                key=lambda item: abs(
                    item[1]["center_u"] - front["center_u"]
                ),
            )
            direction = (
                "right"
                if other["center_u"] < front["center_u"]
                else "left"
            )
            return {
                "key": (
                    "infer",
                    direction,
                    front_color,
                    other_color,
                ),
                "kind": "infer",
                "direction": direction,
                "front_color": front_color,
                "other_color": other_color,
                "visible_colors": visible_colors,
            }

        if step_kind == "observe_left":
            if self.smart_search_first_color is None:
                return None
            different_colors = [
                (color, detection)
                for color, detection in by_color.items()
                if color != self.smart_search_first_color
            ]
            if different_colors:
                second_color, _ = min(
                    different_colors,
                    key=lambda item: abs(item[1]["center_u"] - desired_u),
                )
                return {
                    "key": (
                        "infer",
                        "right",
                        self.smart_search_first_color,
                        second_color,
                    ),
                    "kind": "infer",
                    "direction": "right",
                    "front_color": self.smart_search_first_color,
                    "other_color": second_color,
                    "visible_colors": visible_colors,
                }

            if self.smart_search_first_color in by_color:
                return {
                    "key": ("same", self.smart_search_first_color),
                    "kind": "same",
                    "front_color": self.smart_search_first_color,
                    "visible_colors": visible_colors,
                }
        return None

    @staticmethod
    def color_scene_summary(observation):
        if observation is None:
            return "没有可用于推断的非目标颜色"
        colors = ",".join(observation["visible_colors"])
        if observation["kind"] == "single":
            return "正面颜色=%s，可见颜色=[%s]" % (
                observation["front_color"],
                colors,
            )
        if observation["kind"] == "same":
            return "左移后仍只确认到原颜色=%s，可见颜色=[%s]" % (
                observation["front_color"],
                colors,
            )
        return "非目标颜色=%s/%s，推断目标在%s侧，可见颜色=[%s]" % (
            observation["front_color"],
            observation["other_color"],
            "左" if observation["direction"] == "left" else "右",
            colors,
        )

    def add_color_scene_sample(
        self,
        observation,
        frame_index,
        invalid_reason="",
    ):
        if not self.color_observation_is_ready():
            return
        step_kind = self.current_search_step_kind()
        self.color_scene_frame_window.append({
            "frame_index": frame_index,
            "observation": observation,
        })
        self.color_scene_frame_window = self.color_scene_frame_window[
            -self.stable_detection_window_size :
        ]
        window_count = len(self.color_scene_frame_window)
        required_count = self.auto_search_stable_detection_count

        if observation is None:
            rospy.loginfo(
                (
                    "%s：[快速颜色判定][模型帧 #%d] 本帧无有效推断：%s；"
                    "窗口=%d/%d帧，保留旧的有效颜色结论"
                ),
                NODE_NAME,
                frame_index,
                invalid_reason or "未看到非目标颜色方框",
                window_count,
                self.stable_detection_window_size,
            )
            self.maybe_start_color_search_fallback(step_kind)
            return

        matching_frames = [
            item
            for item in self.color_scene_frame_window
            if item["observation"] is not None
            and item["observation"]["key"] == observation["key"]
        ]
        frame_ids = [item["frame_index"] for item in matching_frames]
        rospy.loginfo(
            (
                "%s：[快速颜色判定][模型帧 #%d] %s；窗口=%d/%d帧，"
                "同一颜色布局=%d/%d，命中帧=%s"
            ),
            NODE_NAME,
            frame_index,
            self.color_scene_summary(observation),
            window_count,
            self.stable_detection_window_size,
            len(matching_frames),
            required_count,
            frame_ids,
        )

        if (
            observation["kind"] != "same"
            and len(matching_frames) >= required_count
        ):
            self.handle_color_scene_decision(
                step_kind,
                observation,
                frame_ids,
            )
            return
        self.maybe_start_color_search_fallback(step_kind)

    def handle_color_scene_decision(
        self,
        step_kind,
        observation,
        frame_ids,
    ):
        if not self.color_observation_is_ready():
            return
        if observation["kind"] == "single" and step_kind == "observe_center":
            self.smart_search_first_color = observation["front_color"]
            self.replace_search_plan_after_observation(
                [
                    ("left", self.auto_search_left_distance),
                    ("observe_left", 0.0),
                ],
                (
                    "最近%d帧内%d帧确认正面非目标颜色=%s，命中帧=%s；"
                    "按原左移步长%.2fm观察相邻方框"
                )
                % (
                    self.stable_detection_window_size,
                    len(frame_ids),
                    self.smart_search_first_color,
                    frame_ids,
                    self.auto_search_left_distance,
                ),
            )
            return

        if observation["kind"] == "infer":
            self.schedule_smart_search_direction(
                observation["direction"],
                step_kind,
                (
                    "最近%d帧内%d帧确认非目标颜色%s/%s，命中帧=%s，"
                    "推断目标在%s侧"
                )
                % (
                    self.stable_detection_window_size,
                    len(frame_ids),
                    observation["front_color"],
                    observation["other_color"],
                    frame_ids,
                    "左" if observation["direction"] == "left" else "右",
                ),
            )

    def maybe_start_color_search_fallback(self, step_kind):
        if len(self.color_scene_frame_window) < (
            self.stable_detection_window_size
        ):
            return
        if not self.color_observation_is_ready():
            return

        hypothesis_counts = {}
        for item in self.color_scene_frame_window:
            observation = item["observation"]
            if observation is None:
                continue
            summary = self.color_scene_summary(observation)
            hypothesis_counts[summary] = hypothesis_counts.get(summary, 0) + 1
        summary = (
            "；".join(
                "%s=%d帧" % item
                for item in sorted(hypothesis_counts.items())
            )
            if hypothesis_counts
            else "10帧内没有可用颜色结论"
        )
        self.start_fallback_search(
            step_kind,
            "窗口已满但没有形成可执行的3帧一致结论（%s）" % summary,
        )

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

    def record_invalid_model_frame(self, frame_index, reason):
        if self.state == self.AUTO_APPROACH:
            self.record_auto_tracking_miss(frame_index, reason)
            return
        if self.state == self.WAIT_FOR_TARGET or (
            self.state == self.AUTO_HOVER_CONFIRM
            and self.hover_confirmation_ready
        ):
            self.add_detection_sample(None, frame_index, reason)
            self.add_color_scene_sample(None, frame_index, reason)

    def detection_callback(self, message):
        now = rospy.Time.now()
        previous_model_message_time = self.last_model_message_time
        self.last_model_message_time = now
        self.model_frame_index += 1
        frame_index = self.model_frame_index

        if (
            self.state in (
                self.WAIT_FOR_TARGET,
                self.AUTO_HOVER_CONFIRM,
            )
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
            self.record_invalid_model_frame(
                frame_index, "模型 JSON 解析失败"
            )
            return

        if not isinstance(payload, dict):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：模型 JSON 根节点不是对象",
                NODE_NAME,
            )
            self.record_invalid_model_frame(
                frame_index, "模型 JSON 根节点无效"
            )
            return

        raw_detections = payload.get("detections", [])
        if not isinstance(raw_detections, list):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：模型 JSON 的 detections 不是数组",
                NODE_NAME,
            )
            self.record_invalid_model_frame(
                frame_index, "模型 detections 字段无效"
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
                    "机器人尚未完成当前位置保持，本帧不计入悬停复核窗口"
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
                self.record_auto_tracking_miss(
                    frame_index,
                    "没有找到 %s 方框或置信度低于 %.2f"
                    % (self.target_color, self.min_confidence),
                )
                return

            self.add_detection_sample(
                None,
                frame_index,
                "没有找到 %s 方框或置信度低于 %.2f"
                % (self.target_color, self.min_confidence),
            )
            if self.color_observation_is_ready():
                step_kind = self.current_search_step_kind()
                observation = self.build_color_scene_observation(
                    detections,
                    step_kind,
                )
                self.add_color_scene_sample(
                    observation,
                    frame_index,
                    "未看到目标颜色，也没有稳定的其他颜色布局",
                )
            return

        if self.state == self.AUTO_APPROACH:
            best = self.select_auto_tracking_candidate(candidates)
            if best is None:
                self.record_auto_tracking_miss(
                    frame_index,
                    "同色候选与上一跟踪目标的中心或面积不连续",
                )
                return
        else:
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
            self.current_auto_target = self.append_auto_tracking_detection(best)
            self.auto_tracking_waiting_for_fresh_frame = False
            self.last_target_time = now
            self.visual_stop_locked = False
            error_u_px, error_v_px, _, _ = self.auto_target_errors(
                self.current_auto_target
            )
            self.update_auto_center_stability(
                self.current_auto_target,
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
                self.detection_summary(self.current_auto_target),
                error_u_px,
                error_v_px,
            )
            return

        self.add_detection_sample(best, frame_index)
        if self.color_observation_is_ready():
            self.add_color_scene_sample(
                None,
                frame_index,
                "本帧看到目标颜色，优先累计目标位置候选组",
            )

    def reset_stability(self):
        self.detection_frame_window = []
        self.color_scene_frame_window = []
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
        required_count = self.auto_center_stable_detection_count
        samples = self.auto_tracking_frame_window[-required_count:]
        if not samples:
            self.reset_auto_center_stability("没有可用的细对准连续帧")
            return

        latest = samples[-1]
        latest_error_u, _, _, _ = self.auto_target_errors(latest)
        (
            latest_edge_balanced,
            latest_touches_left,
            latest_touches_right,
            _,
        ) = self.horizontal_edge_state(latest)
        if abs(latest_error_u) > self.auto_center_tolerance_u_px:
            self.reset_auto_center_stability("方框左右中心超出像素容差")
            rospy.loginfo(
                (
                    "%s：[模型帧 #%d] 横向细对准帧有效但左右尚未居中：%s，"
                    "中位数误差u=%+.1fpx，最新误差u=%+.1fpx，"
                    "纵向误差v=%+.1fpx仅观察"
                ),
                NODE_NAME,
                frame_index,
                self.detection_summary(detection),
                error_u_px,
                latest_error_u,
                error_v_px,
            )
            return
        if not latest_edge_balanced:
            self.reset_auto_center_stability("方框仍单侧贴住画面边缘")
            rospy.loginfo(
                (
                    "%s：[模型帧 #%d] 方框中心已进入容差但仍单侧贴边："
                    "左=%s，右=%s；继续横向细对准，不计入最终确认"
                ),
                NODE_NAME,
                frame_index,
                "是" if latest_touches_left else "否",
                "是" if latest_touches_right else "否",
            )
            return

        trailing_count = 0
        for sample in reversed(samples):
            sample_error_u, _, _, _ = self.auto_target_errors(sample)
            edge_balanced, _, _, _ = self.horizontal_edge_state(sample)
            if (
                abs(sample_error_u) > self.auto_center_tolerance_u_px
                or not edge_balanced
            ):
                break
            trailing_count += 1

        self.auto_centered_frame_count = min(
            trailing_count,
            required_count,
        )
        if self.auto_centered_frame_count < required_count:
            rospy.loginfo(
                (
                    "%s：[模型帧 #%d] 左右居中连续帧=%d/%d：%s，"
                    "中位数误差u=%+.1fpx，等待更多连续帧"
                ),
                NODE_NAME,
                frame_index,
                self.auto_centered_frame_count,
                required_count,
                self.detection_summary(detection),
                error_u_px,
            )
            return

        median_u = statistics.median(
            item["center_u"] for item in samples
        )
        median_area = statistics.median(item["area"] for item in samples)
        center_jitter = max(
            abs(item["center_u"] - median_u) for item in samples
        )
        area_change = max(
            self.area_change_ratio(item["area"], median_area)
            for item in samples
        )
        if (
            center_jitter > self.stable_center_tolerance_px
            or area_change > self.stable_area_tolerance_ratio
        ):
            self.reset_auto_center_stability(
                "最终连续帧的水平中心或面积不一致"
            )
            rospy.logwarn(
                (
                    "%s：[模型帧 #%d] 最终%d帧一致性未通过："
                    "水平中心抖动=%.1f/%.1fpx，面积变化=%.3f/%.3f；"
                    "继续识别，不执行投放"
                ),
                NODE_NAME,
                frame_index,
                required_count,
                center_jitter,
                self.stable_center_tolerance_px,
                area_change,
                self.stable_area_tolerance_ratio,
            )
            return

        rospy.loginfo(
            (
                "%s：[模型帧 #%d] 最终左右细对准%d帧确认通过：%s，"
                "中位数误差u=%+.1fpx，水平中心抖动=%.1fpx，"
                "面积变化=%.3f，纵向误差v=%+.1fpx仅观察"
            ),
            NODE_NAME,
            frame_index,
            required_count,
            self.detection_summary(detection),
            error_u_px,
            center_jitter,
            area_change,
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
                self.auto_tracking_frame_window = []
                self.auto_tracking_waiting_for_fresh_frame = False
                self.reset_auto_center_stability()
                self.visual_stop_locked = False
                self.hover_confirmation_ready = False
                self.hover_confirmation_hover_at = None
                self.hover_confirmation_started_at = None
                self.reset_stability()
                if not self.request_motion_hold(
                    "搜索阶段首次稳定识别目标，保持当前位置后重新识别",
                    discard_search_resume=True,
                ):
                    return
                self.set_state(
                    self.AUTO_HOVER_CONFIRM,
                    "搜索中首次识别通过，等待当前位置保持后重新采集识别帧",
                )
            elif self.state == self.AUTO_HOVER_CONFIRM:
                latest["center_u"] = latest["mean_center_u"]
                latest["center_v"] = latest["mean_center_v"]
                self.hover_confirmation_ready = False
                self.hover_confirmation_hover_at = None
                self.hover_confirmation_started_at = None
                self.hover_confirmation_resume_goal = None
                self.detection_frame_window = []
                self.current_auto_target = latest
                self.auto_tracking_frame_window = []
                self.auto_tracking_waiting_for_fresh_frame = True
                self.reset_auto_center_stability()
                self.visual_stop_locked = False
                self.set_state(
                    self.AUTO_APPROACH,
                    "悬停后第二轮目标识别通过，开始左右细对准并保持锁存航向",
                )
        else:
            if not self.require_safe_actuator_feedback("人工模式动作放行"):
                return
            self.start_drop_action("模型目标已稳定确认")

    def actuator_values_match(
        self,
        actual_heading,
        actual_clamp,
        expected_heading,
        expected_clamp,
    ):
        clamp_matched = (
            abs(int(actual_clamp) - int(expected_clamp))
            <= self.actuator_servo_tolerance
        )
        if not self.heading_servo_enabled:
            return clamp_matched
        return (
            abs(int(actual_heading) - int(expected_heading))
            <= self.actuator_servo_tolerance
            and clamp_matched
        )

    def start_drop_action(self, reason):
        if self.auto_enabled:
            if not self.start_pre_drop_forward(reason):
                self.finish_task(False, "无法生成投放前固定前进目标")
            return
        self.begin_drop_actuator_action(reason)

    def begin_drop_actuator_action(self, reason):
        self.drop_action_started = True
        if self.heading_servo_enabled:
            self.publish_actuator(
                self.clamp_closed,
                self.target_color,
                self.heading_servo_center,
            )
            self.set_state(
                self.HOLD_BEFORE_ACTION,
                "%s；夹爪先移到中间" % reason,
            )
            return

        self.publish_actuator(
            self.clamp_open,
            self.target_color,
        )
        self.set_state(
            self.OPEN_CLAMP,
            "%s；方向舵机关闭，跳过回中并直接打开夹爪" % reason,
        )

    def require_safe_actuator_feedback(self, context):
        if self.actuator_safe_feedback_ready:
            self.actuator_pre_action_wait_started_at = None
            return True

        now = rospy.Time.now()
        if self.actuator_pre_action_wait_started_at is None:
            self.actuator_pre_action_wait_started_at = now
        waiting_seconds = (
            now - self.actuator_pre_action_wait_started_at
        ).to_sec()

        if self.latest_actuator_status is None:
            detail = "尚未收到反馈"
        else:
            status_age = (
                now - self.last_actuator_status_time
            ).to_sec()
            if self.heading_servo_enabled:
                detail = (
                    "实际=(航向%d,夹爪%d)，目标右侧闭合=(%d,%d)，"
                    "连续=%d/%d，反馈年龄=%.2fs"
                    % (
                        self.latest_actuator_status["heading_servo"],
                        self.latest_actuator_status["clamp_servo"],
                        self.heading_servo_right,
                        self.clamp_closed,
                        self.actuator_safe_feedback_count,
                        self.actuator_feedback_confirm_frames,
                        status_age,
                    )
                )
            else:
                detail = (
                    "实际夹爪=%d，目标闭合=%d，方向舵机不参与判断，"
                    "连续=%d/%d，反馈年龄=%.2fs"
                    % (
                        self.latest_actuator_status["clamp_servo"],
                        self.clamp_closed,
                        self.actuator_safe_feedback_count,
                        self.actuator_feedback_confirm_frames,
                        status_age,
                    )
                )

        rospy.logwarn_throttle(
            self.warning_log_interval,
            "%s：[执行器反馈] %s等待初始闭合到位；%s",
            NODE_NAME,
            context,
            detail,
        )
        if waiting_seconds >= self.actuator_stage_timeout:
            self.finish_task(
                False,
                (
                    "%s等待执行器初始闭合反馈超过%.1fs；%s"
                    % (context, self.actuator_stage_timeout, detail)
                ),
            )
        return False

    def actuator_stage_complete(
        self,
        expected_heading,
        expected_clamp,
        hold_seconds,
        stage_name,
        expected_color="off",
    ):
        now = rospy.Time.now()
        status = self.latest_actuator_status
        expected_lights = self.COLOR_LIGHTS.get(
            expected_color, self.COLOR_LIGHTS["off"]
        )
        status_is_fresh = (
            status is not None
            and self.last_actuator_status_time is not None
            and (now - self.last_actuator_status_time).to_sec()
            <= self.actuator_status_timeout
        )

        if not status_is_fresh:
            self.actuator_feedback_match_count = 0
            self.actuator_feedback_confirmed_at = None
            if status is None:
                detail = "尚未收到反馈"
            else:
                detail = "反馈年龄=%.2fs，限制=%.2fs" % (
                    (now - self.last_actuator_status_time).to_sec(),
                    self.actuator_status_timeout,
                )
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：[执行器反馈][%s] %s",
                NODE_NAME,
                stage_name,
                detail,
            )
        elif (
            status["sequence"] > self.actuator_feedback_baseline_sequence
            and status["sequence"]
            > self.actuator_feedback_last_checked_sequence
        ):
            self.actuator_feedback_last_checked_sequence = status["sequence"]
            actuator_matched = self.actuator_values_match(
                status["heading_servo"],
                status["clamp_servo"],
                expected_heading,
                expected_clamp,
            )
            actual_lights = (
                status["red_light"],
                status["yellow_light"],
                status["green_light"],
            )
            lights_matched = actual_lights == expected_lights
            matched = actuator_matched and lights_matched
            if matched:
                self.actuator_feedback_match_count = min(
                    self.actuator_feedback_match_count + 1,
                    self.actuator_feedback_confirm_frames,
                )
            else:
                self.actuator_feedback_match_count = 0
                self.actuator_feedback_confirmed_at = None

            if self.heading_servo_enabled:
                feedback_detail = (
                    "航向=%d/%d，夹爪=%d/%d，误差=(%d,%d)，"
                    "颜色灯=(红%d/%d,黄%d/%d,绿%d/%d)"
                    % (
                        status["heading_servo"],
                        expected_heading,
                        status["clamp_servo"],
                        expected_clamp,
                        abs(status["heading_servo"] - expected_heading),
                        abs(status["clamp_servo"] - expected_clamp),
                        actual_lights[0],
                        expected_lights[0],
                        actual_lights[1],
                        expected_lights[1],
                        actual_lights[2],
                        expected_lights[2],
                    )
                )
            else:
                feedback_detail = (
                    "夹爪=%d/%d，误差=%d，方向舵机不参与判断，"
                    "颜色灯=(红%d/%d,黄%d/%d,绿%d/%d)"
                    % (
                        status["clamp_servo"],
                        expected_clamp,
                        abs(status["clamp_servo"] - expected_clamp),
                        actual_lights[0],
                        expected_lights[0],
                        actual_lights[1],
                        expected_lights[1],
                        actual_lights[2],
                        expected_lights[2],
                    )
                )
            rospy.loginfo(
                (
                    "%s：[执行器反馈][%s][反馈帧#%d] %s，"
                    "到位=%s，连续=%d/%d"
                ),
                NODE_NAME,
                stage_name,
                status["sequence"],
                feedback_detail,
                "通过" if matched else "未通过",
                self.actuator_feedback_match_count,
                self.actuator_feedback_confirm_frames,
            )
            if (
                self.actuator_feedback_match_count
                >= self.actuator_feedback_confirm_frames
                and self.actuator_feedback_confirmed_at is None
            ):
                self.actuator_feedback_confirmed_at = now
                rospy.loginfo(
                    (
                        "%s：[执行器反馈][%s] 夹爪和颜色灯均已确认到位，"
                        "从现在开始共同保持%.1fs"
                    ),
                    NODE_NAME,
                    stage_name,
                    hold_seconds,
                )

        if self.actuator_feedback_confirmed_at is not None:
            held_seconds = (
                now - self.actuator_feedback_confirmed_at
            ).to_sec()
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：[执行器状态][%s] 夹爪=%d/%d，"
                    "颜色灯=(红%d/%d,黄%d/%d,绿%d/%d)，"
                    "共同保持 %.1f/%.1fs"
                ),
                NODE_NAME,
                stage_name,
                status["clamp_servo"],
                expected_clamp,
                status["red_light"],
                expected_lights[0],
                status["yellow_light"],
                expected_lights[1],
                status["green_light"],
                expected_lights[2],
                min(held_seconds, hold_seconds),
                hold_seconds,
            )
            return held_seconds >= hold_seconds

        if self.state_elapsed() >= self.actuator_stage_timeout:
            if status is None:
                actual = "无"
            elif self.heading_servo_enabled:
                actual = "航向%d,夹爪%d,灯=(%d,%d,%d)" % (
                    status["heading_servo"],
                    status["clamp_servo"],
                    status["red_light"],
                    status["yellow_light"],
                    status["green_light"],
                )
            else:
                actual = "夹爪%d,灯=(%d,%d,%d)，方向舵机不参与判断" % (
                    status["clamp_servo"],
                    status["red_light"],
                    status["yellow_light"],
                    status["green_light"],
                )
            target = (
                "航向%d,夹爪%d,灯=%s"
                % (expected_heading, expected_clamp, expected_lights)
                if self.heading_servo_enabled
                else "夹爪%d,灯=%s" % (expected_clamp, expected_lights)
            )
            self.finish_task(
                False,
                (
                    "执行器阶段[%s]到位超时%.1fs，实际=%s，目标=%s"
                    % (
                        stage_name,
                        self.actuator_stage_timeout,
                        actual,
                        target,
                    )
                ),
            )
        return False

    def state_elapsed(self):
        return (rospy.Time.now() - self.state_started).to_sec()

    def set_state(self, state, reason=""):
        previous = self.state
        previous_elapsed = self.state_elapsed()
        self.state = state
        self.state_started = rospy.Time.now()
        self.actuator_feedback_baseline_sequence = getattr(
            self, "actuator_status_sequence", 0
        )
        self.actuator_feedback_last_checked_sequence = (
            self.actuator_feedback_baseline_sequence
        )
        self.actuator_feedback_match_count = 0
        self.actuator_feedback_confirmed_at = None
        rospy.loginfo(
            (
                "%s：[子任务3阶段] 当前阶段=%s；上一阶段=%s，"
                "上一阶段持续%.1fs，进入原因=%s"
            ),
            NODE_NAME,
            self.STATE_NAMES.get(state, "未知状态"),
            self.STATE_NAMES.get(previous, "未知状态"),
            previous_elapsed,
            reason or "无",
        )

    def publish_actuator(self, clamp_servo, color="off", heading_servo=None):
        red, yellow, green = self.COLOR_LIGHTS.get(
            color, self.COLOR_LIGHTS["off"]
        )
        if heading_servo is None:
            heading_servo = self.heading_servo_right
        if not self.heading_servo_enabled:
            status = getattr(self, "latest_actuator_status", None)
            if status is None:
                rospy.logwarn_throttle(
                    self.warning_log_interval,
                    (
                        "%s：方向舵机控制关闭，尚未收到执行器反馈；"
                        "为避免主动改变舵机位置，本次夹爪指令暂不发送"
                    ),
                    NODE_NAME,
                )
                return False
            heading_servo = status["heading_servo"]

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
        message.heading_servo = int(heading_servo)
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
        if self.auto_enabled:
            if self.request_motion_hold(
                "子任务3结束，保持机器人当前位姿",
                discard_search_resume=True,
                force_refresh=True,
            ):
                self.publish_active_goal()
                rospy.loginfo(
                    "%s：任务结束，已下发机器人当前map位姿保持目标",
                    NODE_NAME,
                )
            else:
                rospy.logerr(
                    "%s：任务结束时无法读取当前map位姿，未生成新的保持目标",
                    NODE_NAME,
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
            and not getattr(self, "finished", False)
            and getattr(self, "goal_pub", None) is not None
        ):
            if self.request_motion_hold(
                "子任务3节点关闭，保持机器人当前位姿",
                discard_search_resume=True,
                force_refresh=True,
            ):
                self.publish_active_goal()
        if hasattr(self, "actuator_pub"):
            self.publish_actuator(self.clamp_closed, "off")

    def run(self):
        if not self.actuator_mode_supported:
            self.finish_task(
                False,
                "ActuatorControl 缺少 mode 字段，请同步消息定义并重新编译",
            )
            return

        while not rospy.is_shutdown() and not self.finished:
            now = rospy.Time.now()
            elapsed = (now - self.task_started).to_sec()
            timeout_elapsed = self.motion_timeout_elapsed()

            if (
                self.state in (
                    self.WAIT_FOR_TARGET,
                    self.AUTO_HOVER_CONFIRM,
                    self.AUTO_APPROACH,
                )
                and timeout_elapsed is not None
                and timeout_elapsed >= self.max_wait_seconds
            ):
                self.max_wait_timed_out = True
                self.finish_task(
                    False,
                    "机器人开始运动后自动搜索/等待 %.1fs，仍未完成 %s 方框确认与对齐"
                    % (timeout_elapsed, self.target_color)
                    if self.auto_enabled
                    else "机器人开始运动后等待 %.1fs，仍未稳定识别到 %s 方框"
                    % (timeout_elapsed, self.target_color),
                )
                return

            if self.auto_enabled and not self.handle_motion_health():
                if self.finished:
                    return
                self.rate.sleep()
                continue

            if self.state == self.WAIT_FOR_TARGET:
                self.publish_actuator(self.clamp_closed, "off")
                model_ready = False
                model_topic = self.detection_topic
                if self.last_model_message_time is None:
                    rospy.logwarn_throttle(
                        self.warning_log_interval,
                        "%s：等待模型话题 %s，已等待 %.1fs",
                        NODE_NAME,
                        model_topic,
                        elapsed,
                    )
                else:
                    model_age = (now - self.last_model_message_time).to_sec()
                    if model_age > self.detection_timeout:
                        self.reset_stability()
                        if self.auto_enabled:
                            self.reset_box_position_queue()
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
                self.confirm_box_after_coarse_hover()

            elif self.state == self.AUTO_APPROACH:
                self.publish_actuator(self.clamp_closed, "off")
                self.approach_box_by_map()

            elif self.state == self.PRE_DROP_FORWARD:
                self.handle_pre_drop_forward()

            elif self.state == self.HOLD_BEFORE_ACTION:
                if self.auto_enabled:
                    self.publish_action_position_hold("动作前最终定点保持")
                self.publish_actuator(
                    self.clamp_closed,
                    self.target_color,
                    self.heading_servo_center,
                )
                if self.actuator_stage_complete(
                    self.heading_servo_center,
                    self.clamp_closed,
                    self.hold_seconds,
                    "回中并保持闭合",
                    self.target_color,
                ):
                    rospy.loginfo(
                        (
                            "%s：反馈确认夹爪已在中间闭合并保持%.1fs，"
                            "开始打开，颜色灯=%s"
                        ),
                        NODE_NAME,
                        self.hold_seconds,
                        self.target_color,
                    )
                    self.publish_actuator(
                        self.clamp_open,
                        self.target_color,
                        self.heading_servo_center,
                    )
                    self.set_state(self.OPEN_CLAMP, "开始执行投放动作")

            elif self.state == self.OPEN_CLAMP:
                if self.auto_enabled:
                    self.publish_action_position_hold(
                        "开灯和夹爪打开期间最终定点保持"
                    )
                self.publish_actuator(
                    self.clamp_open,
                    self.target_color,
                    self.heading_servo_center,
                )
                if self.actuator_stage_complete(
                    self.heading_servo_center,
                    self.clamp_open,
                    self.open_seconds,
                    (
                        "中间打开"
                        if self.heading_servo_enabled
                        else "夹爪打开"
                    ),
                    self.target_color,
                ):
                    if not self.heading_servo_enabled:
                        rospy.loginfo(
                            (
                                "%s：方向舵机控制关闭；夹爪已打开且%s灯"
                                "持续亮%.1fs，跳过回右侧并开始关闭夹爪"
                            ),
                            NODE_NAME,
                            self.target_color,
                            self.open_seconds,
                        )
                        self.publish_actuator(
                            self.clamp_closed,
                            "off",
                        )
                        self.set_state(
                            self.CLOSE_CLAMP,
                            "夹爪打开时间完成，直接开始关闭",
                        )
                        continue
                    rospy.loginfo(
                        (
                            "%s：反馈确认夹爪在中间打开并保持%.1fs，"
                            "现在保持打开、熄灯并回到右侧"
                        ),
                        NODE_NAME,
                        self.open_seconds,
                    )
                    self.publish_actuator(
                        self.clamp_open,
                        "off",
                        self.heading_servo_right,
                    )
                    self.set_state(
                        self.RETURN_GRIPPER_RIGHT,
                        "夹爪打开时间完成，开始回到右侧",
                    )

            elif self.state == self.RETURN_GRIPPER_RIGHT:
                if self.auto_enabled:
                    self.publish_action_position_hold(
                        "夹爪打开状态回右侧期间最终定点保持"
                    )
                self.publish_actuator(
                    self.clamp_open,
                    "off",
                    self.heading_servo_right,
                )
                if self.actuator_stage_complete(
                    self.heading_servo_right,
                    self.clamp_open,
                    self.return_right_seconds,
                    "打开状态回右侧",
                ):
                    rospy.loginfo(
                        (
                            "%s：反馈确认夹爪已打开回到右侧并保持%.1fs，"
                            "开始关闭夹爪"
                        ),
                        NODE_NAME,
                        self.return_right_seconds,
                    )
                    self.publish_actuator(
                        self.clamp_closed,
                        "off",
                        self.heading_servo_right,
                    )
                    self.set_state(
                        self.CLOSE_CLAMP,
                        "夹爪已回右侧，开始关闭",
                    )

            elif self.state == self.CLOSE_CLAMP:
                if self.auto_enabled:
                    self.publish_action_position_hold(
                        "夹爪关闭期间最终定点保持"
                )
                self.publish_actuator(
                    self.clamp_closed,
                    "off",
                    self.heading_servo_right,
                )
                if self.actuator_stage_complete(
                    self.heading_servo_right,
                    self.clamp_closed,
                    self.close_seconds,
                    (
                        "右侧关闭"
                        if self.heading_servo_enabled
                        else "夹爪关闭"
                    ),
                ):
                    if self.auto_enabled and self.post_drop_motion_enabled:
                        if not self.start_post_drop_turn():
                            self.finish_task(
                                False,
                                "投放完成，但无法生成投放后左转目标",
                            )
                            return
                    else:
                        self.finish_task(
                            True,
                            "识别和投放动作执行完成；未执行自动离场"
                            if self.auto_enabled
                            else "人工模式识别和投放动作执行完成",
                        )
                        return

            elif self.state == self.POST_DROP_TURN:
                self.handle_post_drop_turn()

            elif self.state == self.POST_DROP_RETURN_ORIGIN:
                self.handle_post_drop_return_origin()

            elif self.state == self.POST_DROP_ASCEND:
                self.handle_post_drop_ascent()

            if self.auto_enabled and not self.finished:
                self.publish_active_goal()
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    configure_task_file_logging("subtask3")
    try:
        Task3InspectAndDropTest().run()
    except rospy.ROSInterruptException:
        pass
