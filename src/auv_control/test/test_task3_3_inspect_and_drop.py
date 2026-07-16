#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
任务 3 子任务 3：识别指定颜色方框并执行投放动作。

节点支持两种操作模式：
  1. manual：人工控制机器人到方框上方，本节点只识别并执行灯光、夹爪动作；
  2. auto：使用定深定向手控模式向前搜索，再根据方框中心像素控制 TX/TY 对齐。

两种模式共用颜色过滤、连续帧稳定判断和执行器动作流程。
"""

import json
import math

import rospy
import tf
from auv_control.msg import AUVData, ActuatorControl, PoseNEDcmd
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task3_3_inspect_and_drop"
MODE_DEPTH_HDG = 3


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def yaw_from_quaternion(quaternion):
    return euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ])[2]


# 模型识别参数。
DEFAULT_RATE = 10.0
DEFAULT_DETECTION_TOPIC = "/web/detections"
DEFAULT_TARGET_COLOR = "yellow"
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_STABLE_DETECTION_COUNT = 5
DEFAULT_AUTO_SEARCH_STABLE_DETECTION_COUNT = 3
DEFAULT_AUTO_CENTER_STABLE_DETECTION_COUNT = 5
DEFAULT_STABLE_CENTER_TOLERANCE_PX = 40.0
DEFAULT_STABLE_AREA_TOLERANCE_RATIO = 0.35
DEFAULT_DETECTION_TIMEOUT = 2.0
DEFAULT_MAX_WAIT_SECONDS = 300.0

# 操作模式和自动视觉控制参数。
DEFAULT_OPERATION_MODE = "manual"
DEFAULT_POSE_CMD_TOPIC = "/cmd/pose/ned"
DEFAULT_STATUS_TOPIC = "/status/auv"
DEFAULT_STATUS_TIMEOUT = 0.5
DEFAULT_STATUS_LINEAR_VELOCITY_SCALE = 1.0
DEFAULT_STATUS_ANGULAR_VELOCITY_SCALE = math.pi / 180.0
DEFAULT_AUTO_INITIAL_HOVER_SECONDS = 10.0
DEFAULT_AUTO_SEARCH_FORWARD_FORCE = 120.0
DEFAULT_AUTO_SEARCH_LATERAL_FORCE = 120.0
DEFAULT_AUTO_SEARCH_FIRST_FORWARD_DISTANCE = 0.30
DEFAULT_AUTO_SEARCH_SECOND_FORWARD_DISTANCE = 0.20
DEFAULT_AUTO_SEARCH_THIRD_FORWARD_DISTANCE = 0.10
DEFAULT_AUTO_SEARCH_LATERAL_DISTANCE = 0.20
DEFAULT_AUTO_SEARCH_DISTANCE_TOLERANCE = 0.03
DEFAULT_AUTO_SEARCH_FORWARD_BRAKING_DISTANCE = 0.08
DEFAULT_AUTO_SEARCH_LATERAL_BRAKING_DISTANCE = 0.08
DEFAULT_AUTO_FORWARD_GAIN = 250.0
DEFAULT_AUTO_LATERAL_GAIN = 250.0
DEFAULT_AUTO_MAX_FORWARD_FORCE = 180.0
DEFAULT_AUTO_MAX_LATERAL_FORCE = 180.0
DEFAULT_AUTO_MIN_CORRECTION_FORCE = 50.0
DEFAULT_AUTO_FORCE_STEP = 50.0
DEFAULT_AUTO_FORWARD_VELOCITY_DAMPING = 300.0
DEFAULT_AUTO_LATERAL_VELOCITY_DAMPING = 300.0
DEFAULT_AUTO_SEARCH_STOP_SPEED = 0.05
DEFAULT_AUTO_ACTION_MAX_HORIZONTAL_SPEED = 0.03
DEFAULT_AUTO_ACTION_MAX_VERTICAL_SPEED = 0.03
DEFAULT_AUTO_ACTION_MAX_YAW_RATE = 0.05
DEFAULT_AUTO_ACTION_MAX_DEPTH_ERROR = 0.08
DEFAULT_AUTO_ACTION_MAX_YAW_ERROR_DEG = 5.0
DEFAULT_AUTO_HOLD_FORWARD_POSITION_GAIN = 600.0
DEFAULT_AUTO_HOLD_LATERAL_POSITION_GAIN = 600.0
DEFAULT_AUTO_HOLD_MAX_FORCE = 120.0
DEFAULT_AUTO_HOLD_POSITION_TOLERANCE = 0.02
DEFAULT_AUTO_TX_SIGN = 1.0
DEFAULT_AUTO_TY_SIGN = 1.0
DEFAULT_AUTO_TARGET_CENTER_U_RATIO = 0.5
DEFAULT_AUTO_TARGET_CENTER_V_RATIO = 0.5
DEFAULT_AUTO_CENTER_TOLERANCE_U_PX = 35.0
DEFAULT_AUTO_CENTER_TOLERANCE_V_PX = 35.0
DEFAULT_AUTO_IMAGE_WIDTH = 640.0
DEFAULT_AUTO_IMAGE_HEIGHT = 480.0

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
    AUTO_APPROACH = 1
    HOLD_BEFORE_ACTION = 2
    OPEN_CLAMP = 3
    CLOSE_CLAMP = 4

    STATE_NAMES = {
        WAIT_FOR_TARGET: "等待目标颜色方框",
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
    }

    COLOR_LIGHTS = {
        "yellow": (0, 1, 0),
        "green": (0, 0, 1),
        "red": (1, 0, 0),
        "off": (0, 0, 0),
    }

    def __init__(self):
        self.rate = rospy.Rate(float(rospy.get_param("~rate", DEFAULT_RATE)))
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

        self.pose_cmd_topic = str(
            rospy.get_param("~pose_cmd_topic", DEFAULT_POSE_CMD_TOPIC)
        ).strip()
        self.status_topic = str(
            rospy.get_param("~status_topic", DEFAULT_STATUS_TOPIC)
        ).strip()
        self.status_timeout = float(rospy.get_param(
            "~status_timeout", DEFAULT_STATUS_TIMEOUT
        ))
        self.status_linear_velocity_scale = float(rospy.get_param(
            "~status_linear_velocity_scale",
            DEFAULT_STATUS_LINEAR_VELOCITY_SCALE,
        ))
        self.status_angular_velocity_scale = float(rospy.get_param(
            "~status_angular_velocity_scale",
            DEFAULT_STATUS_ANGULAR_VELOCITY_SCALE,
        ))
        self.auto_initial_hover_seconds = float(rospy.get_param(
            "~auto_initial_hover_seconds", DEFAULT_AUTO_INITIAL_HOVER_SECONDS
        ))
        self.auto_search_forward_force = float(rospy.get_param(
            "~auto_search_forward_force", DEFAULT_AUTO_SEARCH_FORWARD_FORCE
        ))
        self.auto_search_lateral_force = float(rospy.get_param(
            "~auto_search_lateral_force", DEFAULT_AUTO_SEARCH_LATERAL_FORCE
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
        self.auto_search_lateral_distance = float(rospy.get_param(
            "~auto_search_lateral_distance", DEFAULT_AUTO_SEARCH_LATERAL_DISTANCE
        ))
        self.auto_search_distance_tolerance = float(rospy.get_param(
            "~auto_search_distance_tolerance",
            DEFAULT_AUTO_SEARCH_DISTANCE_TOLERANCE,
        ))
        self.auto_search_forward_braking_distance = float(rospy.get_param(
            "~auto_search_forward_braking_distance",
            DEFAULT_AUTO_SEARCH_FORWARD_BRAKING_DISTANCE,
        ))
        self.auto_search_lateral_braking_distance = float(rospy.get_param(
            "~auto_search_lateral_braking_distance",
            DEFAULT_AUTO_SEARCH_LATERAL_BRAKING_DISTANCE,
        ))
        self.auto_forward_gain = float(rospy.get_param(
            "~auto_forward_gain", DEFAULT_AUTO_FORWARD_GAIN
        ))
        self.auto_lateral_gain = float(rospy.get_param(
            "~auto_lateral_gain", DEFAULT_AUTO_LATERAL_GAIN
        ))
        self.auto_max_forward_force = float(rospy.get_param(
            "~auto_max_forward_force", DEFAULT_AUTO_MAX_FORWARD_FORCE
        ))
        self.auto_max_lateral_force = float(rospy.get_param(
            "~auto_max_lateral_force", DEFAULT_AUTO_MAX_LATERAL_FORCE
        ))
        self.auto_min_correction_force = float(rospy.get_param(
            "~auto_min_correction_force", DEFAULT_AUTO_MIN_CORRECTION_FORCE
        ))
        self.auto_force_step = float(rospy.get_param(
            "~auto_force_step", DEFAULT_AUTO_FORCE_STEP
        ))
        self.auto_forward_velocity_damping = float(rospy.get_param(
            "~auto_forward_velocity_damping",
            DEFAULT_AUTO_FORWARD_VELOCITY_DAMPING,
        ))
        self.auto_lateral_velocity_damping = float(rospy.get_param(
            "~auto_lateral_velocity_damping",
            DEFAULT_AUTO_LATERAL_VELOCITY_DAMPING,
        ))
        self.auto_search_stop_speed = float(rospy.get_param(
            "~auto_search_stop_speed", DEFAULT_AUTO_SEARCH_STOP_SPEED
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
        self.auto_hold_forward_position_gain = float(rospy.get_param(
            "~auto_hold_forward_position_gain",
            DEFAULT_AUTO_HOLD_FORWARD_POSITION_GAIN,
        ))
        self.auto_hold_lateral_position_gain = float(rospy.get_param(
            "~auto_hold_lateral_position_gain",
            DEFAULT_AUTO_HOLD_LATERAL_POSITION_GAIN,
        ))
        self.auto_hold_max_force = float(rospy.get_param(
            "~auto_hold_max_force", DEFAULT_AUTO_HOLD_MAX_FORCE
        ))
        self.auto_hold_position_tolerance = float(rospy.get_param(
            "~auto_hold_position_tolerance",
            DEFAULT_AUTO_HOLD_POSITION_TOLERANCE,
        ))
        self.auto_tx_sign = float(rospy.get_param(
            "~auto_tx_sign", DEFAULT_AUTO_TX_SIGN
        ))
        self.auto_ty_sign = float(rospy.get_param(
            "~auto_ty_sign", DEFAULT_AUTO_TY_SIGN
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
        self.pose_cmd_pub = None
        self.tf_listener = None
        if self.auto_enabled:
            self.pose_cmd_pub = rospy.Publisher(
                self.pose_cmd_topic, PoseNEDcmd, queue_size=10
            )
            self.tf_listener = tf.TransformListener()

        self.state = self.WAIT_FOR_TARGET
        self.state_started = rospy.Time.now()
        self.task_started = rospy.Time.now()
        self.last_model_message_time = None
        self.last_target_time = None
        self.detection_samples = []
        self.model_frame_index = 0
        self.current_auto_target = None
        self.auto_hold_z = None
        self.auto_hold_yaw = None
        self.auto_centered_frame_count = 0
        self.last_auto_tx = 0
        self.last_auto_ty = 0
        self.auto_search_plan = [
            ("hover", self.auto_initial_hover_seconds),
            ("forward", self.auto_search_first_forward_distance),
            ("left", self.auto_search_lateral_distance),
            ("right", self.auto_search_lateral_distance),
            ("forward", self.auto_search_second_forward_distance),
            ("left", self.auto_search_lateral_distance),
            ("right", self.auto_search_lateral_distance),
            ("forward", self.auto_search_third_forward_distance),
            ("left", self.auto_search_lateral_distance),
            ("right", self.auto_search_lateral_distance),
        ]
        self.auto_search_index = 0
        self.auto_search_step_started = None
        self.auto_search_step_origin = None
        self.auto_search_step_braking = False
        self.last_status_time = None
        self.current_status = None
        self.status_hold_depth = None
        self.status_hold_yaw_deg = None
        self.auto_action_hold_position = None
        self.finished = False

        self.detection_sub = rospy.Subscriber(
            self.detection_topic,
            String,
            self.detection_callback,
            queue_size=10,
        )
        self.status_sub = None
        if self.auto_enabled:
            self.status_sub = rospy.Subscriber(
                self.status_topic,
                AUVData,
                self.status_callback,
                queue_size=20,
            )
        rospy.on_shutdown(self.on_shutdown)

        if self.auto_enabled:
            rospy.loginfo(
                (
                    "%s：启动自动寻找模式，运动话题=%s，底层模式=3（定深定向），"
                    "状态反馈=%s，仅发送 TX/TY"
                ),
                NODE_NAME,
                self.pose_cmd_topic,
                self.status_topic,
            )
        else:
            rospy.loginfo(
                "%s：启动人工操作模式，只识别和执行动作，不发布机器人运动指令",
                NODE_NAME,
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
                "%s：稳定条件：中心最大抖动 %.1fpx，检测框面积变化比例 %.2f，"
                "识别超时 %.1fs，总等待上限 %.1fs"
            ),
            NODE_NAME,
            self.stable_center_tolerance_px,
            self.stable_area_tolerance_ratio,
            self.detection_timeout,
            self.max_wait_seconds,
        )
        if self.auto_enabled:
            rospy.loginfo(
                "%s：自动模式帧数门槛：搜索锁定=%d帧，居中确认=%d帧",
                NODE_NAME,
                self.auto_search_stable_detection_count,
                self.auto_center_stable_detection_count,
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
                    "%s：自动控制：保持启动时当前航向，搜索力=(前进%.0f,横移%.0f)，"
                    "修正增益=(前后%.0f,左右%.0f)，"
                    "最大力=(%.0f,%.0f)，单次变化<=%.0f"
                ),
                NODE_NAME,
                self.auto_search_forward_force,
                self.auto_search_lateral_force,
                self.auto_forward_gain,
                self.auto_lateral_gain,
                self.auto_max_forward_force,
                self.auto_max_lateral_force,
                self.auto_force_step,
            )
            rospy.loginfo(
                (
                    "%s：状态反馈：超时=%.2fs，速度缩放=(线速度%.6f,角速度%.6f)，"
                    "阻尼=(前后%.0f,左右%.0f)，"
                    "搜索停稳<=%.3fm/s，动作前速度<=水平%.3f/垂直%.3fm/s，"
                    "航向角速度<=%.3frad/s，深度误差<=%.2fm，航向误差<=%.1fdeg"
                ),
                NODE_NAME,
                self.status_timeout,
                self.status_linear_velocity_scale,
                self.status_angular_velocity_scale,
                self.auto_forward_velocity_damping,
                self.auto_lateral_velocity_damping,
                self.auto_search_stop_speed,
                self.auto_action_max_horizontal_speed,
                self.auto_action_max_vertical_speed,
                self.auto_action_max_yaw_rate,
                self.auto_action_max_depth_error,
                self.auto_action_max_yaw_error_deg,
            )
            rospy.loginfo(
                (
                    "%s：提前刹车距离=(前进%.3fm,横移%.3fm)，"
                    "最终定点增益=(前后%.0f,左右%.0f)，"
                    "定点最大力=%.0f，位置容差=%.3fm"
                ),
                NODE_NAME,
                self.auto_search_forward_braking_distance,
                self.auto_search_lateral_braking_distance,
                self.auto_hold_forward_position_gain,
                self.auto_hold_lateral_position_gain,
                self.auto_hold_max_force,
                self.auto_hold_position_tolerance,
            )
            rospy.loginfo(
                (
                    "%s：自动对齐：目标中心=(%.2fW, %.2fH)，容差=(%.1fpx, %.1fpx)，"
                    "连续居中确认=%d帧，方向符号=(TX %.0f, TY %.0f)"
                ),
                NODE_NAME,
                self.auto_target_center_u_ratio,
                self.auto_target_center_v_ratio,
                self.auto_center_tolerance_u_px,
                self.auto_center_tolerance_v_px,
                self.auto_center_stable_detection_count,
                self.auto_tx_sign,
                self.auto_ty_sign,
            )
            rospy.loginfo(
                (
                    "%s：搜索顺序：悬停%.1fs -> 前进%.2fm -> 左%.2fm -> 右%.2fm -> "
                    "前进%.2fm -> 左%.2fm -> 右%.2fm -> 前进%.2fm -> 左%.2fm -> 右%.2fm"
                ),
                NODE_NAME,
                self.auto_initial_hover_seconds,
                self.auto_search_first_forward_distance,
                self.auto_search_lateral_distance,
                self.auto_search_lateral_distance,
                self.auto_search_second_forward_distance,
                self.auto_search_lateral_distance,
                self.auto_search_lateral_distance,
                self.auto_search_third_forward_distance,
                self.auto_search_lateral_distance,
                self.auto_search_lateral_distance,
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
            message.angular_velocity[0],
            message.angular_velocity[1],
            message.angular_velocity[2],
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
                2.0,
                "%s：/status/auv 包含无效位姿或速度，本帧已忽略",
                NODE_NAME,
            )
            return

        self.current_status = {
            "control_mode": int(message.control_mode),
            "vx": float(raw_values[0]) * self.status_linear_velocity_scale,
            "vy": float(raw_values[1]) * self.status_linear_velocity_scale,
            "vz": float(raw_values[2]) * self.status_linear_velocity_scale,
            "wx": float(raw_values[3]) * self.status_angular_velocity_scale,
            "wy": float(raw_values[4]) * self.status_angular_velocity_scale,
            "wz": float(raw_values[5]) * self.status_angular_velocity_scale,
            "latitude": float(raw_values[6]),
            "longitude": float(raw_values[7]),
            "depth": float(raw_values[8]),
            "altitude": float(raw_values[9]),
            "roll_deg": float(raw_values[10]),
            "pitch_deg": float(raw_values[11]),
            "yaw_deg": float(raw_values[12]),
        }
        self.last_status_time = rospy.Time.now()

        if self.status_hold_depth is None:
            self.status_hold_depth = self.current_status["depth"]
            self.status_hold_yaw_deg = self.current_status["yaw_deg"]
            rospy.loginfo(
                "%s：记录 /status/auv 启动基准：深度=%.3fm，航向=%.2fdeg",
                NODE_NAME,
                self.status_hold_depth,
                self.status_hold_yaw_deg,
            )

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：/status/auv：mode=%d，位姿=(lat=%.7f,lon=%.7f,"
                "深度=%.3f,航向=%.2fdeg)，速度前右下=(%+.3f,%+.3f,%+.3f)m/s，"
                "角速度=(%+.3f,%+.3f,%+.3f)rad/s"
            ),
            NODE_NAME,
            self.current_status["control_mode"],
            self.current_status["latitude"],
            self.current_status["longitude"],
            self.current_status["depth"],
            self.current_status["yaw_deg"],
            self.current_status["vx"],
            self.current_status["vy"],
            self.current_status["vz"],
            self.current_status["wx"],
            self.current_status["wy"],
            self.current_status["wz"],
        )

    def get_recent_status(self, context):
        if self.current_status is None or self.last_status_time is None:
            rospy.logwarn_throttle(
                2.0,
                "%s：等待状态话题 %s，%s暂停",
                NODE_NAME,
                self.status_topic,
                context,
            )
            return None

        age = (rospy.Time.now() - self.last_status_time).to_sec()
        if age > self.status_timeout:
            rospy.logwarn_throttle(
                2.0,
                "%s：/status/auv 已超时 %.2fs（限制 %.2fs），%s暂停",
                NODE_NAME,
                age,
                self.status_timeout,
                context,
            )
            return None
        return self.current_status

    @staticmethod
    def horizontal_speed(status):
        return math.hypot(status["vx"], status["vy"])

    def velocity_feedback_command(
        self,
        base_forward_force,
        base_right_force,
        status,
        speed_deadband,
        max_forward_force=None,
        max_right_force=None,
    ):
        if max_forward_force is None:
            max_forward_force = self.auto_max_forward_force
        if max_right_force is None:
            max_right_force = self.auto_max_lateral_force

        if self.horizontal_speed(status) > speed_deadband:
            damping_vx = status["vx"]
            damping_vy = status["vy"]
        else:
            damping_vx = 0.0
            damping_vy = 0.0

        physical_forward = clamp(
            base_forward_force
            - self.auto_forward_velocity_damping * damping_vx,
            -max_forward_force,
            max_forward_force,
        )
        physical_right = clamp(
            base_right_force
            - self.auto_lateral_velocity_damping * damping_vy,
            -max_right_force,
            max_right_force,
        )
        return (
            self.auto_tx_sign * physical_forward,
            self.auto_ty_sign * physical_right,
        )

    def publish_velocity_brake(self, reason, speed_deadband):
        status = self.get_recent_status(reason)
        if status is None:
            self.publish_auto_stop("/status/auv 不可用，清零水平推力")
            return False
        desired_tx, desired_ty = self.velocity_feedback_command(
            0.0,
            0.0,
            status,
            speed_deadband,
        )
        return self.publish_auto_motion(desired_tx, desired_ty, reason)

    @staticmethod
    def angle_difference_deg(angle_a, angle_b):
        return (angle_a - angle_b + 180.0) % 360.0 - 180.0

    def status_pose_errors(self, status):
        depth_error = status["depth"] - self.status_hold_depth
        yaw_error_deg = self.angle_difference_deg(
            status["yaw_deg"],
            self.status_hold_yaw_deg,
        )
        return depth_error, yaw_error_deg

    def action_status_is_stable(self, status):
        depth_error, yaw_error_deg = self.status_pose_errors(status)
        return (
            status["control_mode"] == MODE_DEPTH_HDG
            and self.horizontal_speed(status)
            <= self.auto_action_max_horizontal_speed
            and abs(status["vz"]) <= self.auto_action_max_vertical_speed
            and abs(status["wz"]) <= self.auto_action_max_yaw_rate
            and abs(depth_error) <= self.auto_action_max_depth_error
            and abs(yaw_error_deg) <= self.auto_action_max_yaw_error_deg
        )

    def capture_action_hold_position(self):
        current = self.get_current_pose()
        if current is None:
            return False
        self.auto_action_hold_position = (
            current.pose.position.x,
            current.pose.position.y,
        )
        rospy.loginfo(
            "%s：记录最终定点位置：map=(%.3f, %.3f)",
            NODE_NAME,
            self.auto_action_hold_position[0],
            self.auto_action_hold_position[1],
        )
        return True

    def publish_action_position_hold(self, reason):
        if self.auto_action_hold_position is None:
            return self.publish_velocity_brake(
                "%s，最终定点尚未记录" % reason,
                self.auto_action_max_horizontal_speed,
            )

        status = self.get_recent_status(reason)
        current = self.get_current_pose()
        if status is None or current is None:
            self.publish_auto_stop("最终定点反馈不可用，清零水平推力")
            return False

        dx = self.auto_action_hold_position[0] - current.pose.position.x
        dy = self.auto_action_hold_position[1] - current.pose.position.y
        forward_error = (
            dx * math.cos(self.auto_hold_yaw)
            + dy * math.sin(self.auto_hold_yaw)
        )
        right_error = (
            -dx * math.sin(self.auto_hold_yaw)
            + dy * math.cos(self.auto_hold_yaw)
        )

        base_forward_force = 0.0
        if abs(forward_error) > self.auto_hold_position_tolerance:
            base_forward_force = (
                self.auto_hold_forward_position_gain * forward_error
            )
        base_right_force = 0.0
        if abs(right_error) > self.auto_hold_position_tolerance:
            base_right_force = (
                self.auto_hold_lateral_position_gain * right_error
            )

        desired_tx, desired_ty = self.velocity_feedback_command(
            base_forward_force,
            base_right_force,
            status,
            self.auto_action_max_horizontal_speed,
            self.auto_hold_max_force,
            self.auto_hold_max_force,
        )
        if not self.publish_auto_motion(desired_tx, desired_ty, reason):
            return False

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：最终定点保持：位置误差=(前%+.3f,右%+.3f)m，"
                "速度=(前%+.3f,右%+.3f)m/s，指令=(TX=%d,TY=%d)"
            ),
            NODE_NAME,
            forward_error,
            right_error,
            status["vx"],
            status["vy"],
            self.last_auto_tx,
            self.last_auto_ty,
        )
        return True

    def validate_params(self):
        if self.operation_mode not in ("manual", "auto"):
            raise ValueError("operation_mode 必须是 manual 或 auto")
        if not self.detection_topic:
            raise ValueError("detection_topic 不能为空")
        if not self.target_color:
            raise ValueError("target_color 不能为空")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence 必须在 0 到 1 之间")
        if self.stable_detection_count < 1:
            raise ValueError("stable_detection_count 必须大于等于 1")
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

        if self.auto_enabled:
            if self.auto_search_stable_detection_count < 1:
                raise ValueError(
                    "auto_search_stable_detection_count 必须大于等于 1"
                )
            if self.auto_center_stable_detection_count < 1:
                raise ValueError(
                    "auto_center_stable_detection_count 必须大于等于 1"
                )
            if not self.pose_cmd_topic:
                raise ValueError("pose_cmd_topic 不能为空")
            if not self.status_topic:
                raise ValueError("status_topic 不能为空")
            if self.status_timeout <= 0.0:
                raise ValueError("status_timeout 必须大于 0")
            if min(
                self.status_linear_velocity_scale,
                self.status_angular_velocity_scale,
            ) <= 0.0:
                raise ValueError("/status/auv 速度缩放参数必须大于 0")
            if min(
                self.auto_initial_hover_seconds,
                self.auto_search_forward_force,
                self.auto_search_lateral_force,
                self.auto_search_distance_tolerance,
            ) < 0.0:
                raise ValueError("自动悬停、搜索力和距离容差不能小于 0")
            search_distances = (
                self.auto_search_first_forward_distance,
                self.auto_search_second_forward_distance,
                self.auto_search_third_forward_distance,
                self.auto_search_lateral_distance,
            )
            if min(search_distances) <= 0.0:
                raise ValueError("自动搜索的前进和横移距离必须大于 0")
            if self.auto_search_distance_tolerance >= min(search_distances):
                raise ValueError("auto_search_distance_tolerance 必须小于搜索距离")
            if min(
                self.auto_search_forward_braking_distance,
                self.auto_search_lateral_braking_distance,
            ) <= self.auto_search_distance_tolerance:
                raise ValueError("提前刹车距离必须大于搜索距离容差")
            if self.auto_search_forward_braking_distance > min(
                self.auto_search_first_forward_distance,
                self.auto_search_second_forward_distance,
                self.auto_search_third_forward_distance,
            ):
                raise ValueError("前进提前刹车距离不能大于最短前进距离")
            if (
                self.auto_search_lateral_braking_distance
                > self.auto_search_lateral_distance
            ):
                raise ValueError("横移提前刹车距离不能大于横移距离")
            if min(self.auto_forward_gain, self.auto_lateral_gain) < 0.0:
                raise ValueError("自动修正增益不能小于 0")
            if min(
                self.auto_max_forward_force,
                self.auto_max_lateral_force,
                self.auto_min_correction_force,
                self.auto_force_step,
                self.auto_forward_velocity_damping,
                self.auto_lateral_velocity_damping,
                self.auto_search_stop_speed,
                self.auto_action_max_horizontal_speed,
                self.auto_action_max_vertical_speed,
                self.auto_action_max_yaw_rate,
                self.auto_action_max_depth_error,
                self.auto_action_max_yaw_error_deg,
                self.auto_hold_forward_position_gain,
                self.auto_hold_lateral_position_gain,
                self.auto_hold_max_force,
                self.auto_hold_position_tolerance,
            ) < 0.0:
                raise ValueError("自动推力限制和速度反馈参数不能小于 0")
            if self.auto_hold_max_force <= 0.0:
                raise ValueError("auto_hold_max_force 必须大于 0")
            if self.auto_min_correction_force > min(
                self.auto_max_forward_force, self.auto_max_lateral_force
            ):
                raise ValueError("auto_min_correction_force 不能大于最大修正力")
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

    def get_current_pose(self):
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
                2.0,
                "%s：自动模式无法读取 map -> base_link：%s",
                NODE_NAME,
                str(exc),
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def initialize_auto_pose(self):
        if not self.auto_enabled:
            return True
        if self.auto_hold_z is not None and self.auto_hold_yaw is not None:
            return True

        current = self.get_current_pose()
        if current is None:
            return False

        self.auto_hold_z = current.pose.position.z
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        self.auto_hold_yaw = current_yaw

        rospy.loginfo(
            (
                "%s：自动控制基准已记录：当前位置=(%.2f, %.2f, %.2f)，"
                "启动航向=%.1fdeg，后续保持该航向"
            ),
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(current_yaw),
        )
        return True

    @staticmethod
    def force_value(value):
        return int(round(clamp(value, -10000.0, 10000.0)))

    def limit_auto_force(self, desired, previous):
        return int(round(clamp(
            desired,
            previous - self.auto_force_step,
            previous + self.auto_force_step,
        )))

    @staticmethod
    def apply_minimum_force(value, minimum, maximum):
        value = clamp(value, -maximum, maximum)
        if value == 0.0 or abs(value) >= minimum:
            return value
        return math.copysign(minimum, value)

    def publish_auto_motion(self, desired_tx, desired_ty, reason, rate_limit=True):
        if not self.auto_enabled or self.pose_cmd_pub is None:
            return False
        if not self.initialize_auto_pose():
            return False

        current = self.get_current_pose()
        if current is None:
            return False

        if rate_limit:
            tx = self.limit_auto_force(desired_tx, self.last_auto_tx)
            ty = self.limit_auto_force(desired_ty, self.last_auto_ty)
        else:
            tx = self.force_value(desired_tx)
            ty = self.force_value(desired_ty)
        self.last_auto_tx = tx
        self.last_auto_ty = ty

        command = PoseNEDcmd()
        command.mode = MODE_DEPTH_HDG
        command.target.header.frame_id = "map"
        command.target.header.stamp = rospy.Time.now()
        command.target.pose.position.x = current.pose.position.x
        command.target.pose.position.y = current.pose.position.y
        command.target.pose.position.z = self.auto_hold_z
        command.target.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0, 0.0, self.auto_hold_yaw
        ))
        command.force.TX = self.force_value(tx)
        command.force.TY = self.force_value(ty)
        self.pose_cmd_pub.publish(command)

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：运动指令 mode=3，保持深度=%.2f，保持航向=%.1fdeg，"
                "TX=%d，TY=%d，TZ/MX/MY/MZ=0，原因=%s"
            ),
            NODE_NAME,
            self.auto_hold_z,
            math.degrees(self.auto_hold_yaw),
            command.force.TX,
            command.force.TY,
            reason,
        )
        return True

    def publish_auto_stop(self, reason):
        if not self.auto_enabled:
            return
        self.publish_auto_motion(0.0, 0.0, reason, rate_limit=False)

    def reset_auto_search_step(self):
        self.auto_search_step_started = None
        self.auto_search_step_origin = None
        self.auto_search_step_braking = False

    def complete_auto_search_step(self, step_kind, step_amount):
        step_number = self.auto_search_index + 1
        rospy.loginfo(
            "%s：搜索步骤 %d/%d 完成：%s %.2f%s",
            NODE_NAME,
            step_number,
            len(self.auto_search_plan),
            self.SEARCH_STEP_NAMES[step_kind],
            step_amount,
            "s" if step_kind == "hover" else "m",
        )
        self.publish_auto_stop("当前搜索步骤完成，切换前先停止")
        self.auto_search_index += 1
        self.reset_auto_search_step()

        if self.auto_search_index >= len(self.auto_search_plan):
            rospy.logwarn(
                "%s：预设搜索路径已经执行完毕，仍未稳定识别方框，原地悬停等待",
                NODE_NAME,
            )

    def search_target_automatically(self, model_ready):
        self.auto_centered_frame_count = 0
        if self.state != self.WAIT_FOR_TARGET:
            self.publish_auto_stop("已退出搜索状态，停止搜索推力")
            return

        if self.auto_search_index >= len(self.auto_search_plan):
            self.publish_velocity_brake(
                "预设搜索路径完成，速度闭环悬停等待识别",
                self.auto_search_stop_speed,
            )
            rospy.logwarn_throttle(
                2.0,
                "%s：搜索路径已完成，等待 %s 方框，任务总超时为 %.1fs",
                NODE_NAME,
                self.target_color,
                self.max_wait_seconds,
            )
            return

        current = self.get_current_pose()
        if current is None:
            return

        step_kind, step_amount = self.auto_search_plan[self.auto_search_index]
        hover_status = None
        if step_kind == "hover":
            hover_status = self.get_recent_status("启动悬停")
            if hover_status is None:
                self.publish_auto_stop("等待 /status/auv，尚未开始10秒悬停计时")
                return

        if self.auto_search_step_started is None:
            self.auto_search_step_started = rospy.Time.now()
            self.auto_search_step_origin = (
                current.pose.position.x,
                current.pose.position.y,
            )
            rospy.loginfo(
                "%s：开始搜索步骤 %d/%d：%s %.2f%s",
                NODE_NAME,
                self.auto_search_index + 1,
                len(self.auto_search_plan),
                self.SEARCH_STEP_NAMES[step_kind],
                step_amount,
                "s" if step_kind == "hover" else "m",
            )

        if step_kind == "hover":
            self.publish_velocity_brake(
                "任务启动后速度闭环悬停，等待人工放置方框",
                self.auto_search_stop_speed,
            )
            elapsed = (rospy.Time.now() - self.auto_search_step_started).to_sec()
            horizontal_speed = self.horizontal_speed(hover_status)
            rospy.loginfo_throttle(
                1.0,
                "%s：启动悬停 %.1f/%.1fs，水平速度=%.3fm/s",
                NODE_NAME,
                min(elapsed, step_amount),
                step_amount,
                horizontal_speed,
            )
            if (
                elapsed >= step_amount
                and horizontal_speed <= self.auto_search_stop_speed
                and self.last_auto_tx == 0
                and self.last_auto_ty == 0
            ):
                self.complete_auto_search_step(step_kind, step_amount)
            elif elapsed >= step_amount:
                rospy.loginfo_throttle(
                    1.0,
                    "%s：悬停时间已到，等待速度降至 %.3fm/s 以下再开始搜索",
                    NODE_NAME,
                    self.auto_search_stop_speed,
                )
            return

        if not model_ready:
            self.publish_velocity_brake(
                "模型话题未就绪或已超时，速度闭环暂停搜索",
                self.auto_search_stop_speed,
            )
            rospy.logwarn_throttle(
                2.0,
                "%s：模型话题未就绪，搜索步骤 %d/%d 暂停",
                NODE_NAME,
                self.auto_search_index + 1,
                len(self.auto_search_plan),
            )
            return

        status = self.get_recent_status("自动搜索")
        if status is None:
            self.publish_auto_stop("/status/auv 不可用，暂停搜索位移")
            return

        origin_x, origin_y = self.auto_search_step_origin
        dx = current.pose.position.x - origin_x
        dy = current.pose.position.y - origin_y
        forward_displacement = (
            dx * math.cos(self.auto_hold_yaw)
            + dy * math.sin(self.auto_hold_yaw)
        )
        right_displacement = (
            -dx * math.sin(self.auto_hold_yaw)
            + dy * math.cos(self.auto_hold_yaw)
        )

        if step_kind == "forward":
            progress = forward_displacement
            base_forward_force = self.auto_search_forward_force
            base_right_force = 0.0
            braking_distance = self.auto_search_forward_braking_distance
        elif step_kind == "left":
            progress = -right_displacement
            base_forward_force = 0.0
            base_right_force = -self.auto_search_lateral_force
            braking_distance = self.auto_search_lateral_braking_distance
        else:
            progress = right_displacement
            base_forward_force = 0.0
            base_right_force = self.auto_search_lateral_force
            braking_distance = self.auto_search_lateral_braking_distance

        remaining_distance = max(step_amount - progress, 0.0)
        if remaining_distance <= braking_distance:
            force_ratio = clamp(
                remaining_distance / braking_distance,
                0.0,
                1.0,
            )
            base_forward_force *= force_ratio
            base_right_force *= force_ratio
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：进入提前减速区：剩余=%.3fm/%.3fm，"
                    "基础推力比例=%.2f，速度阻尼将根据惯性自动补反向力"
                ),
                NODE_NAME,
                remaining_distance,
                braking_distance,
                force_ratio,
            )

        if progress >= step_amount - self.auto_search_distance_tolerance:
            if not self.auto_search_step_braking:
                self.auto_search_step_braking = True
                rospy.loginfo(
                    "%s：搜索步骤达到距离目标，进入速度反馈刹停阶段",
                    NODE_NAME,
                )

        if self.auto_search_step_braking:
            desired_tx, desired_ty = self.velocity_feedback_command(
                0.0,
                0.0,
                status,
                self.auto_search_stop_speed,
            )
            if not self.publish_auto_motion(
                desired_tx,
                desired_ty,
                "搜索步骤达到目标距离，依据速度反馈刹停",
            ):
                return
            horizontal_speed = self.horizontal_speed(status)
            if (
                horizontal_speed <= self.auto_search_stop_speed
                and self.last_auto_tx == 0
                and self.last_auto_ty == 0
            ):
                self.complete_auto_search_step(step_kind, step_amount)
            else:
                rospy.loginfo_throttle(
                    1.0,
                    (
                        "%s：搜索刹停中：位移=%.3f/%.3fm，"
                        "速度=(前%+.3f,右%+.3f)m/s，指令=(TX=%d,TY=%d)"
                    ),
                    NODE_NAME,
                    progress,
                    step_amount,
                    status["vx"],
                    status["vy"],
                    self.last_auto_tx,
                    self.last_auto_ty,
                )
            return

        desired_tx, desired_ty = self.velocity_feedback_command(
            base_forward_force,
            base_right_force,
            status,
            self.auto_search_stop_speed,
        )

        if self.state != self.WAIT_FOR_TARGET:
            self.publish_auto_stop("识别状态已切换，停止搜索推力")
            return

        if not self.publish_auto_motion(
            desired_tx,
            desired_ty,
            "执行分段方框搜索路径",
        ):
            return

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：搜索步骤 %d/%d：%s，位移=%.3f/%.3fm，"
                "速度=(前%+.3f,右%+.3f)m/s，目标力=(TX=%+.0f,TY=%+.0f)"
            ),
            NODE_NAME,
            self.auto_search_index + 1,
            len(self.auto_search_plan),
            self.SEARCH_STEP_NAMES[step_kind],
            progress,
            step_amount,
            status["vx"],
            status["vy"],
            desired_tx,
            desired_ty,
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

    def approach_target_automatically(self):
        now = rospy.Time.now()
        status = self.get_recent_status("方框细对准")
        if status is None:
            self.publish_auto_stop("/status/auv 不可用，暂停方框细对准")
            self.reset_auto_center_stability("/status/auv 不可用或超时")
            return

        if (
            self.last_target_time is not None
            and (now - self.last_target_time).to_sec() > self.detection_timeout
        ):
            self.current_auto_target = None
            self.publish_auto_stop("目标识别结果已超时，立即停止水平运动")
            self.reset_auto_center_stability("目标识别结果超时")
            self.reset_stability()
            self.reset_auto_search_step()
            self.set_state(self.WAIT_FOR_TARGET, "目标丢失超时，重新向前搜索")
            return

        if self.current_auto_target is None:
            self.publish_auto_stop("当前模型帧未识别到目标，立即停止水平运动")
            self.reset_auto_center_stability("当前模型帧未识别到目标")
            if (
                self.last_target_time is not None
                and (now - self.last_target_time).to_sec() > self.detection_timeout
            ):
                self.reset_stability()
                self.reset_auto_search_step()
                self.set_state(self.WAIT_FOR_TARGET, "目标丢失超时，重新向前搜索")
            return

        error_u_px, error_v_px, normalized_u, normalized_v = (
            self.auto_target_errors(self.current_auto_target)
        )
        centered_u = abs(error_u_px) <= self.auto_center_tolerance_u_px
        centered_v = abs(error_v_px) <= self.auto_center_tolerance_v_px

        visual_forward_force = 0.0
        if not centered_v:
            visual_forward_force = self.apply_minimum_force(
                -self.auto_forward_gain * normalized_v,
                self.auto_min_correction_force,
                self.auto_max_forward_force,
            )

        visual_right_force = 0.0
        if not centered_u:
            visual_right_force = self.apply_minimum_force(
                self.auto_lateral_gain * normalized_u,
                self.auto_min_correction_force,
                self.auto_max_lateral_force,
            )

        desired_tx, desired_ty = self.velocity_feedback_command(
            visual_forward_force,
            visual_right_force,
            status,
            self.auto_action_max_horizontal_speed,
        )

        if not self.publish_auto_motion(
            desired_tx,
            desired_ty,
            "依据方框中心像素进行前后和左右修正",
        ):
            self.reset_auto_center_stability("自动运动指令未能发布")
            return
        depth_error, yaw_error_deg = self.status_pose_errors(status)
        rospy.loginfo_throttle(
            1.0,
            (
                "%s：自动对齐：中心=(%.1f,%.1f)，像素误差=(u=%+.1f,v=%+.1f)，"
                "速度=(前%+.3f,右%+.3f,下%+.3f)m/s，航向角速度=%+.3frad/s，"
                "mode=%d，深度误差=%+.3fm，航向误差=%+.2fdeg，"
                "目标力=(TX=%+.0f,TY=%+.0f)"
            ),
            NODE_NAME,
            self.current_auto_target["center_u"],
            self.current_auto_target["center_v"],
            error_u_px,
            error_v_px,
            status["vx"],
            status["vy"],
            status["vz"],
            status["wz"],
            status["control_mode"],
            depth_error,
            yaw_error_deg,
            desired_tx,
            desired_ty,
        )

        horizontal_command_stopped = (
            self.last_auto_tx == 0 and self.last_auto_ty == 0
        )
        physical_motion_stopped = self.action_status_is_stable(status)
        if not (centered_u and centered_v):
            self.reset_auto_center_stability("方框中心超出允许范围")
        elif (
            self.auto_centered_frame_count
            < self.auto_center_stable_detection_count
        ):
            rospy.loginfo_throttle(
                1.0,
                "%s：方框已进入中心范围，等待连续居中识别 %d/%d 帧",
                NODE_NAME,
                self.auto_centered_frame_count,
                self.auto_center_stable_detection_count,
            )
        elif not horizontal_command_stopped or not physical_motion_stopped:
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：方框已连续居中 %d 帧，等待实际停稳："
                    "速度=(水平%.3f,下%+.3f)m/s，航向角速度=%+.3frad/s，"
                    "mode=%d，深度误差=%+.3fm，航向误差=%+.2fdeg，"
                    "当前指令=(TX=%d,TY=%d)"
                ),
                NODE_NAME,
                self.auto_centered_frame_count,
                self.horizontal_speed(status),
                status["vz"],
                status["wz"],
                status["control_mode"],
                depth_error,
                yaw_error_deg,
                self.last_auto_tx,
                self.last_auto_ty,
            )
        else:
            if not self.capture_action_hold_position():
                self.publish_auto_stop("无法记录最终定点位置，暂不执行夹爪动作")
                rospy.logwarn_throttle(
                    1.0,
                    "%s：等待 TF 可用后记录最终定点位置",
                    NODE_NAME,
                )
                return
            self.publish_auto_stop(
                "方框已连续%d帧居中，准备同时开灯和打开夹爪"
                % self.auto_center_stable_detection_count
            )
            self.publish_actuator(self.clamp_open, self.target_color)
            self.set_state(
                self.OPEN_CLAMP,
                "方框中心连续稳定识别达到 %d 帧"
                % self.auto_center_stable_detection_count,
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
        self.last_model_message_time = now
        self.model_frame_index += 1
        frame_index = self.model_frame_index

        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as exc:
            rospy.logwarn_throttle(
                2.0,
                "%s：无法解析模型 JSON：%s",
                NODE_NAME,
                str(exc),
            )
            if self.state == self.AUTO_APPROACH:
                self.current_auto_target = None
                self.reset_auto_center_stability("模型 JSON 解析失败")
            else:
                self.reset_stability()
            return

        if not isinstance(payload, dict):
            rospy.logwarn_throttle(2.0, "%s：模型 JSON 根节点不是对象", NODE_NAME)
            if self.state == self.AUTO_APPROACH:
                self.current_auto_target = None
                self.reset_auto_center_stability("模型 JSON 根节点无效")
            else:
                self.reset_stability()
            return

        raw_detections = payload.get("detections", [])
        if not isinstance(raw_detections, list):
            rospy.logwarn_throttle(
                2.0, "%s：模型 JSON 的 detections 不是数组", NODE_NAME
            )
            if self.state == self.AUTO_APPROACH:
                self.current_auto_target = None
                self.reset_auto_center_stability("模型 detections 字段无效")
            else:
                self.reset_stability()
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
            1.0,
            "%s：模型有效候选=%d：%s",
            NODE_NAME,
            len(detections),
            "; ".join(summaries) if summaries else "无目标",
        )

        if self.state not in (self.WAIT_FOR_TARGET, self.AUTO_APPROACH):
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
                        "当前水平推力将停止"
                    ),
                    NODE_NAME,
                    frame_index,
                    self.target_color,
                )
                return

            previous_count = len(self.detection_samples)
            self.reset_stability()
            rospy.loginfo(
                (
                    "%s：[模型帧 #%d] 本帧无效：没有找到 %s 方框，"
                    "要求置信度 >= %.2f，连续有效帧 %d -> 0"
                ),
                NODE_NAME,
                frame_index,
                self.target_color,
                self.min_confidence,
                previous_count,
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
                1.0,
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
            self.current_auto_target = best
            self.last_target_time = now
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
        self.detection_samples = []
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

    def add_detection_sample(self, detection, frame_index):
        now = detection["stamp"]
        if (
            self.last_target_time is not None
            and (now - self.last_target_time).to_sec() > self.detection_timeout
        ):
            previous_count = len(self.detection_samples)
            self.reset_stability()
            rospy.logwarn(
                (
                    "%s：[模型帧 #%d] 相邻目标间隔超过 %.1fs，"
                    "连续有效帧 %d -> 0"
                ),
                NODE_NAME,
                frame_index,
                self.detection_timeout,
                previous_count,
            )

        self.last_target_time = now
        detection["frame_index"] = frame_index
        self.detection_samples.append(detection)
        required_count = self.required_stable_detection_count()
        self.detection_samples = self.detection_samples[
            -required_count :
        ]

        stable, center_jitter, area_change = self.samples_are_stable(required_count)
        progress = len(self.detection_samples)
        rospy.loginfo(
            (
                "%s：[模型帧 #%d] 第 %d/%d 帧有效：%s，"
                "中心抖动=%.1f/%.1fpx，面积变化=%.3f/%.3f"
            ),
            NODE_NAME,
            frame_index,
            progress,
            required_count,
            self.detection_summary(detection),
            center_jitter,
            self.stable_center_tolerance_px,
            area_change,
            self.stable_area_tolerance_ratio,
        )

        if progress < required_count:
            return

        if not stable:
            rospy.logwarn(
                (
                    "%s：[模型帧 #%d] 目标帧有效，但连续稳定性未通过："
                    "中心抖动=%.1fpx/%.1fpx，面积变化=%.3f/%.3f，"
                    "保留当前帧作为新的第 1/%d 帧"
                ),
                NODE_NAME,
                frame_index,
                center_jitter,
                self.stable_center_tolerance_px,
                area_change,
                self.stable_area_tolerance_ratio,
                required_count,
            )
            self.detection_samples = [detection]
            return

        rospy.loginfo(
            "%s：[模型帧 #%d] 第 %d/%d 帧有效，连续稳定识别确认通过",
            NODE_NAME,
            frame_index,
            progress,
            required_count,
        )
        self.lock_target()

    def samples_are_stable(self, required_count):
        if not self.detection_samples:
            return False, 0.0, 0.0

        mean_u = sum(item["center_u"] for item in self.detection_samples) / len(
            self.detection_samples
        )
        mean_v = sum(item["center_v"] for item in self.detection_samples) / len(
            self.detection_samples
        )
        center_jitter = max(
            math.hypot(item["center_u"] - mean_u, item["center_v"] - mean_v)
            for item in self.detection_samples
        )

        areas = [item["area"] for item in self.detection_samples]
        max_area = max(areas)
        area_change = (max_area - min(areas)) / max_area if max_area > 0.0 else 1.0

        stable = (
            len(self.detection_samples) >= required_count
            and center_jitter <= self.stable_center_tolerance_px
            and area_change <= self.stable_area_tolerance_ratio
        )
        return stable, center_jitter, area_change

    def lock_target(self):
        samples = self.detection_samples
        latest = dict(samples[-1])
        latest["mean_confidence"] = sum(
            item["confidence"] for item in samples
        ) / len(samples)
        latest["mean_center_u"] = sum(
            item["center_u"] for item in samples
        ) / len(samples)
        latest["mean_center_v"] = sum(
            item["center_v"] for item in samples
        ) / len(samples)
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
            latest["center_u"] = latest["mean_center_u"]
            latest["center_v"] = latest["mean_center_v"]
            self.current_auto_target = latest
            self.reset_auto_center_stability()
            self.publish_auto_stop("搜索阶段稳定识别目标，停止搜索后进入细对准")
            self.set_state(
                self.AUTO_APPROACH,
                "模型目标已稳定确认，开始依据中心像素自动对齐",
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
        return True

    def finish_task(self, success, reason):
        if self.finished:
            return
        self.finished = True
        self.publish_auto_stop("任务结束，清零水平推力")
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
        if getattr(self, "auto_enabled", False):
            self.publish_auto_stop("节点关闭，清零水平推力")
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

            if (
                self.state in (self.WAIT_FOR_TARGET, self.AUTO_APPROACH)
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
                        2.0,
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
                            2.0,
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
                    1.0,
                    "%s：已开%s灯并打开夹爪，零水平推力悬停 %.1f/%.1fs",
                    NODE_NAME,
                    self.target_color,
                    min(open_elapsed, self.open_seconds),
                    self.open_seconds,
                )
                if open_elapsed >= self.open_seconds:
                    rospy.loginfo(
                        "%s：开灯并打开夹爪后已悬停 %.1fs，关闭夹爪并熄灯",
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

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3InspectAndDropTest().run()
    except rospy.ROSInterruptException:
        pass
