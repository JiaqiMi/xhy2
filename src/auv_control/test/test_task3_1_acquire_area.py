#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""任务3子任务1：识别箭头后完成中心、航向细对准并定点。

人工先把机器人放在箭头后方约0.5米。节点启动后使用模式3悬停10秒，
再保持深度和启动航向低速向前搜索。``/arrow/direction`` 连续稳定3帧后，
先对齐箭头航向，再通过 TX/TY 缓慢视觉靠近。摄像头居中后使用现有TF
取得箭头map坐标，并将机器人本体移动到该点，最终稳定悬停10秒。
"""

import json
import math

import rospy
import tf
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 添加src的父目
from auv_control.msg import AUVData, PoseNEDcmd
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from driver.motion_supervisor_core import map_error_to_body


NODE_NAME = "test_task3_1_acquire_area"
MODE_DEPTH_HEADING = 4


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_angle_deg(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


def angle_difference_deg(angle_a_deg, angle_b_deg):
    return normalize_angle_deg(angle_a_deg - angle_b_deg)


def yaw_from_quaternion(quaternion):
    return euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ])[2]


# 模型与状态话题。
DEFAULT_RATE = 10.0
DEFAULT_ARROW_TOPIC = "/arrow/direction"
DEFAULT_STATUS_TOPIC = "/status/auv"
DEFAULT_POSE_CMD_TOPIC = "/cmd/pose/ned"
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_DETECTION_TIMEOUT = 1.0
DEFAULT_MAX_WAIT_SECONDS = 300.0

# 启动悬停和直行搜索。
DEFAULT_INITIAL_HOVER_SECONDS = 10.0
DEFAULT_SEARCH_FORWARD_FORCE = 80.0
DEFAULT_MAX_SEARCH_DISTANCE = 1.20
DEFAULT_CAMERA_FRAME = "camera"

# 箭头首次锁定条件。
DEFAULT_STABLE_DETECTION_COUNT = 3
DEFAULT_STABLE_CENTER_TOLERANCE_PX = 40.0
DEFAULT_STABLE_ANGLE_TOLERANCE_DEG = 12.0

# 图像中心和航向换算。
DEFAULT_IMAGE_WIDTH = 640.0
DEFAULT_IMAGE_HEIGHT = 480.0
DEFAULT_TARGET_CENTER_U_RATIO = 0.5
DEFAULT_TARGET_CENTER_V_RATIO = 0.5
DEFAULT_CENTER_TOLERANCE_U_PX = 35.0
DEFAULT_CENTER_TOLERANCE_V_PX = 35.0
DEFAULT_CENTER_STABLE_DETECTION_COUNT = 5
DEFAULT_HEADING_STABLE_DETECTION_COUNT = 3
# 箭头协议中图像向右为0度、向上为90度；默认图像向上对应机器人前方。
DEFAULT_CAMERA_FORWARD_ANGLE_DEG = 90.0
DEFAULT_YAW_CORRECTION_SIGN = 1.0
DEFAULT_YAW_TOLERANCE_DEG = 10.0
DEFAULT_YAW_TARGET_FILTER_ALPHA = 0.35
DEFAULT_YAW_TARGET_MAX_STEP_DEG = 10.0

# 视觉细对准推力与速度阻尼。
DEFAULT_FORWARD_GAIN = 160.0
DEFAULT_LATERAL_GAIN = 160.0
DEFAULT_MAX_FORWARD_FORCE = 100.0
DEFAULT_MAX_LATERAL_FORCE = 100.0
DEFAULT_MIN_CORRECTION_FORCE = 35.0
DEFAULT_FORCE_STEP = 30.0
DEFAULT_FORWARD_VELOCITY_DAMPING = 300.0
DEFAULT_LATERAL_VELOCITY_DAMPING = 300.0
DEFAULT_SPEED_DEADBAND = 0.03
DEFAULT_TX_SIGN = 1.0
DEFAULT_TY_SIGN = 1.0

# 等待识别和最终阶段的位置外环。
DEFAULT_HOLD_FORWARD_POSITION_GAIN = 600.0
DEFAULT_HOLD_LATERAL_POSITION_GAIN = 600.0
DEFAULT_HOLD_MAX_FORCE = 100.0
DEFAULT_HOLD_POSITION_TOLERANCE = 0.02
DEFAULT_FINAL_POSITION_TOLERANCE = 0.05

# /status/auv 反馈及成功判定。
DEFAULT_STATUS_TIMEOUT = 0.5
DEFAULT_STATUS_LINEAR_VELOCITY_SCALE = 1.0
# AUVData 中角速度沿用下位机 deg/s，乘该系数后转成 rad/s。
DEFAULT_STATUS_ANGULAR_VELOCITY_SCALE = math.pi / 180.0
DEFAULT_MAX_HORIZONTAL_SPEED = 0.03
DEFAULT_MAX_VERTICAL_SPEED = 0.03
DEFAULT_MAX_YAW_RATE = 0.05
DEFAULT_MAX_DEPTH_ERROR = 0.08
DEFAULT_MIN_GROUND_CLEARANCE = 0.40

# 最终定点持续时间。
DEFAULT_FINAL_HOLD_SECONDS = 10.0
DEFAULT_FINAL_HOLD_TIMEOUT = 30.0


class Task3AcquireAreaTest:
    INITIAL_HOVER = "启动悬停"
    FORWARD_SEARCH = "直行搜索箭头"
    WAIT_FOR_ARROW = "丢失后定点重识别"
    ALIGN_HEADING = "箭头航向对准"
    APPROACH_CAMERA = "摄像头视觉居中"
    BODY_CENTERING = "机器人本体移到箭头上方"
    FINAL_HOLD = "最终定点"

    def __init__(self):
        self.rate_hz = float(rospy.get_param("~rate", DEFAULT_RATE))
        self.rate = rospy.Rate(self.rate_hz)

        self.arrow_topic = str(
            rospy.get_param("~arrow_topic", DEFAULT_ARROW_TOPIC)
        ).strip()
        self.status_topic = str(
            rospy.get_param("~status_topic", DEFAULT_STATUS_TOPIC)
        ).strip()
        self.pose_cmd_topic = str(
            rospy.get_param("~pose_cmd_topic", DEFAULT_POSE_CMD_TOPIC)
        ).strip()
        self.min_confidence = float(rospy.get_param(
            "~min_confidence", DEFAULT_MIN_CONFIDENCE
        ))
        self.detection_timeout = float(rospy.get_param(
            "~detection_timeout", DEFAULT_DETECTION_TIMEOUT
        ))
        self.max_wait_seconds = float(rospy.get_param(
            "~max_wait_seconds", DEFAULT_MAX_WAIT_SECONDS
        ))
        self.initial_hover_seconds = float(rospy.get_param(
            "~initial_hover_seconds", DEFAULT_INITIAL_HOVER_SECONDS
        ))
        self.search_forward_force = float(rospy.get_param(
            "~search_forward_force", DEFAULT_SEARCH_FORWARD_FORCE
        ))
        self.max_search_distance = float(rospy.get_param(
            "~max_search_distance", DEFAULT_MAX_SEARCH_DISTANCE
        ))
        self.camera_frame = str(rospy.get_param(
            "~camera_frame", DEFAULT_CAMERA_FRAME
        )).strip()

        self.stable_detection_count = int(rospy.get_param(
            "~stable_detection_count", DEFAULT_STABLE_DETECTION_COUNT
        ))
        self.stable_center_tolerance_px = float(rospy.get_param(
            "~stable_center_tolerance_px", DEFAULT_STABLE_CENTER_TOLERANCE_PX
        ))
        self.stable_angle_tolerance_deg = float(rospy.get_param(
            "~stable_angle_tolerance_deg", DEFAULT_STABLE_ANGLE_TOLERANCE_DEG
        ))

        self.image_width = float(rospy.get_param(
            "~image_width", DEFAULT_IMAGE_WIDTH
        ))
        self.image_height = float(rospy.get_param(
            "~image_height", DEFAULT_IMAGE_HEIGHT
        ))
        self.target_center_u_ratio = float(rospy.get_param(
            "~target_center_u_ratio", DEFAULT_TARGET_CENTER_U_RATIO
        ))
        self.target_center_v_ratio = float(rospy.get_param(
            "~target_center_v_ratio", DEFAULT_TARGET_CENTER_V_RATIO
        ))
        self.center_tolerance_u_px = float(rospy.get_param(
            "~center_tolerance_u_px", DEFAULT_CENTER_TOLERANCE_U_PX
        ))
        self.center_tolerance_v_px = float(rospy.get_param(
            "~center_tolerance_v_px", DEFAULT_CENTER_TOLERANCE_V_PX
        ))
        self.center_stable_detection_count = int(rospy.get_param(
            "~center_stable_detection_count",
            DEFAULT_CENTER_STABLE_DETECTION_COUNT,
        ))
        self.heading_stable_detection_count = int(rospy.get_param(
            "~heading_stable_detection_count",
            DEFAULT_HEADING_STABLE_DETECTION_COUNT,
        ))
        self.camera_forward_angle_deg = float(rospy.get_param(
            "~camera_forward_angle_deg", DEFAULT_CAMERA_FORWARD_ANGLE_DEG
        ))
        self.yaw_correction_sign = float(rospy.get_param(
            "~yaw_correction_sign", DEFAULT_YAW_CORRECTION_SIGN
        ))
        self.yaw_tolerance_deg = float(rospy.get_param(
            "~yaw_tolerance_deg", DEFAULT_YAW_TOLERANCE_DEG
        ))
        self.yaw_target_filter_alpha = float(rospy.get_param(
            "~yaw_target_filter_alpha", DEFAULT_YAW_TARGET_FILTER_ALPHA
        ))
        self.yaw_target_max_step_deg = float(rospy.get_param(
            "~yaw_target_max_step_deg", DEFAULT_YAW_TARGET_MAX_STEP_DEG
        ))

        self.forward_gain = float(rospy.get_param(
            "~forward_gain", DEFAULT_FORWARD_GAIN
        ))
        self.lateral_gain = float(rospy.get_param(
            "~lateral_gain", DEFAULT_LATERAL_GAIN
        ))
        self.max_forward_force = float(rospy.get_param(
            "~max_forward_force", DEFAULT_MAX_FORWARD_FORCE
        ))
        self.max_lateral_force = float(rospy.get_param(
            "~max_lateral_force", DEFAULT_MAX_LATERAL_FORCE
        ))
        self.min_correction_force = float(rospy.get_param(
            "~min_correction_force", DEFAULT_MIN_CORRECTION_FORCE
        ))
        self.force_step = float(rospy.get_param(
            "~force_step", DEFAULT_FORCE_STEP
        ))
        self.forward_velocity_damping = float(rospy.get_param(
            "~forward_velocity_damping", DEFAULT_FORWARD_VELOCITY_DAMPING
        ))
        self.lateral_velocity_damping = float(rospy.get_param(
            "~lateral_velocity_damping", DEFAULT_LATERAL_VELOCITY_DAMPING
        ))
        self.speed_deadband = float(rospy.get_param(
            "~speed_deadband", DEFAULT_SPEED_DEADBAND
        ))
        self.tx_sign = float(rospy.get_param("~tx_sign", DEFAULT_TX_SIGN))
        self.ty_sign = float(rospy.get_param("~ty_sign", DEFAULT_TY_SIGN))

        self.hold_forward_position_gain = float(rospy.get_param(
            "~hold_forward_position_gain",
            DEFAULT_HOLD_FORWARD_POSITION_GAIN,
        ))
        self.hold_lateral_position_gain = float(rospy.get_param(
            "~hold_lateral_position_gain",
            DEFAULT_HOLD_LATERAL_POSITION_GAIN,
        ))
        self.hold_max_force = float(rospy.get_param(
            "~hold_max_force", DEFAULT_HOLD_MAX_FORCE
        ))
        self.hold_position_tolerance = float(rospy.get_param(
            "~hold_position_tolerance", DEFAULT_HOLD_POSITION_TOLERANCE
        ))
        self.final_position_tolerance = float(rospy.get_param(
            "~final_position_tolerance", DEFAULT_FINAL_POSITION_TOLERANCE
        ))

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
        self.max_horizontal_speed = float(rospy.get_param(
            "~max_horizontal_speed", DEFAULT_MAX_HORIZONTAL_SPEED
        ))
        self.max_vertical_speed = float(rospy.get_param(
            "~max_vertical_speed", DEFAULT_MAX_VERTICAL_SPEED
        ))
        self.max_yaw_rate = float(rospy.get_param(
            "~max_yaw_rate", DEFAULT_MAX_YAW_RATE
        ))
        self.max_depth_error = float(rospy.get_param(
            "~max_depth_error", DEFAULT_MAX_DEPTH_ERROR
        ))
        self.min_ground_clearance = float(rospy.get_param(
            "~min_ground_clearance", DEFAULT_MIN_GROUND_CLEARANCE
        ))
        self.final_hold_seconds = float(rospy.get_param(
            "~final_hold_seconds", DEFAULT_FINAL_HOLD_SECONDS
        ))
        self.final_hold_timeout = float(rospy.get_param(
            "~final_hold_timeout", DEFAULT_FINAL_HOLD_TIMEOUT
        ))

        self.validate_params()

        self.pose_cmd_pub = rospy.Publisher(
            self.pose_cmd_topic, PoseNEDcmd, queue_size=10
        )
        self.finished_pub = rospy.Publisher(
            "/finished", String, queue_size=10
        )
        self.tf_listener = tf.TransformListener()

        self.task_started = rospy.Time.now()
        self.state = self.INITIAL_HOVER
        self.state_started = self.task_started
        self.current_status = None
        self.last_status_time = None
        self.control_initialized = False
        self.hold_z = None
        self.hold_depth = None
        self.wait_hold_position = None
        self.wait_hold_yaw = None
        self.initial_hover_started = None
        self.search_origin = None
        self.search_yaw = None
        self.target_yaw = None
        self.heading_hold_position = None
        self.heading_aligned_frame_count = 0
        self.last_heading_frame_index = 0
        self.arrow_map_position = None
        self.final_hold_position = None
        self.final_hold_stable_started = None

        self.model_frame_index = 0
        self.last_model_message_time = None
        self.latest_detection = None
        self.last_valid_detection_time = None
        self.detection_samples = []
        self.arrow_locked = False
        self.locked_detection = None
        self.centered_frame_count = 0
        self.last_alignment_frame_index = 0
        self.last_target_yaw_frame_index = 0
        self.alignment_loss_hold_position = None

        self.last_tx = 0
        self.last_ty = 0
        self.task_finished = False

        self.status_sub = rospy.Subscriber(
            self.status_topic, AUVData, self.status_callback, queue_size=20
        )
        self.arrow_sub = rospy.Subscriber(
            self.arrow_topic, String, self.arrow_callback, queue_size=20
        )
        rospy.on_shutdown(self.on_shutdown)
        self.log_startup_config()

    def validate_params(self):
        if self.rate_hz <= 0.0:
            raise ValueError("rate 必须大于0")
        if not all((
            self.arrow_topic,
            self.status_topic,
            self.pose_cmd_topic,
            self.camera_frame,
        )):
            raise ValueError("箭头、状态、运动指令话题和相机坐标系不能为空")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence 必须在0到1之间")
        if min(self.detection_timeout, self.max_wait_seconds) <= 0.0:
            raise ValueError("识别超时和任务超时必须大于0")
        if min(
            self.stable_detection_count,
            self.center_stable_detection_count,
            self.heading_stable_detection_count,
        ) <= 0:
            raise ValueError("稳定识别帧数必须大于0")
        if min(
            self.stable_center_tolerance_px,
            self.stable_angle_tolerance_deg,
            self.center_tolerance_u_px,
            self.center_tolerance_v_px,
            self.yaw_tolerance_deg,
        ) < 0.0:
            raise ValueError("识别与对准容差不能小于0")
        if min(self.image_width, self.image_height) <= 0.0:
            raise ValueError("图像宽度和高度必须大于0")
        if not 0.0 <= self.target_center_u_ratio <= 1.0:
            raise ValueError("target_center_u_ratio 必须在0到1之间")
        if not 0.0 <= self.target_center_v_ratio <= 1.0:
            raise ValueError("target_center_v_ratio 必须在0到1之间")
        if self.yaw_correction_sign not in (-1.0, 1.0):
            raise ValueError("yaw_correction_sign 必须是1.0或-1.0")
        if not 0.0 < self.yaw_target_filter_alpha <= 1.0:
            raise ValueError("yaw_target_filter_alpha 必须在0到1之间")
        if self.yaw_target_max_step_deg <= 0.0:
            raise ValueError("yaw_target_max_step_deg 必须大于0")
        if min(
            self.forward_gain,
            self.lateral_gain,
            self.initial_hover_seconds,
            self.search_forward_force,
            self.max_search_distance,
            self.max_forward_force,
            self.max_lateral_force,
            self.min_correction_force,
            self.force_step,
            self.forward_velocity_damping,
            self.lateral_velocity_damping,
            self.speed_deadband,
            self.hold_forward_position_gain,
            self.hold_lateral_position_gain,
            self.hold_max_force,
            self.hold_position_tolerance,
            self.final_position_tolerance,
            self.status_timeout,
            self.status_linear_velocity_scale,
            self.status_angular_velocity_scale,
            self.max_horizontal_speed,
            self.max_vertical_speed,
            self.max_yaw_rate,
            self.max_depth_error,
            self.min_ground_clearance,
            self.final_hold_seconds,
            self.final_hold_timeout,
        ) < 0.0:
            raise ValueError("控制增益、限制、阻尼和时间参数不能小于0")
        if min(
            self.search_forward_force,
            self.max_search_distance,
            self.max_forward_force,
            self.max_lateral_force,
            self.force_step,
            self.hold_max_force,
            self.status_timeout,
            self.status_linear_velocity_scale,
            self.status_angular_velocity_scale,
            self.min_ground_clearance,
            self.final_hold_timeout,
        ) <= 0.0:
            raise ValueError("推力限制、状态缩放、离地距离和最终超时必须大于0")
        if self.min_correction_force > min(
            self.max_forward_force, self.max_lateral_force
        ):
            raise ValueError("min_correction_force 不能大于最大修正力")
        if self.tx_sign not in (-1.0, 1.0) or self.ty_sign not in (-1.0, 1.0):
            raise ValueError("tx_sign 和 ty_sign 必须是1.0或-1.0")
        if self.final_hold_timeout < self.final_hold_seconds:
            raise ValueError("final_hold_timeout 不能小于 final_hold_seconds")

    def log_startup_config(self):
        rospy.loginfo(
            (
                "%s：启动子任务1；流程=mode3悬停%.1fs -> 低速直行搜索 -> "
                "稳定识别%d帧 -> 先对齐航向 -> 缓慢视觉居中 -> "
                "相机/本体TF补偿 -> 最终定点%.1fs"
            ),
            NODE_NAME,
            self.initial_hover_seconds,
            self.stable_detection_count,
            self.final_hold_seconds,
        )
        rospy.loginfo(
            (
                "%s：箭头话题=%s，状态话题=%s，运动话题=%s，最低置信度=%.2f，"
                "识别超时=%.1fs，任务超时=%.1fs"
            ),
            NODE_NAME,
            self.arrow_topic,
            self.status_topic,
            self.pose_cmd_topic,
            self.min_confidence,
            self.detection_timeout,
            self.max_wait_seconds,
        )
        rospy.loginfo(
            (
                "%s：直行搜索：启动航向保持不变，前进力=%.1f，"
                "最大搜索距离=%.2fm；相机坐标系=%s，最终通过TF把camera位置"
                "转换成机器人本体map目标"
            ),
            NODE_NAME,
            self.search_forward_force,
            self.max_search_distance,
            self.camera_frame,
        )
        rospy.loginfo(
            (
                "%s：图像=%.0fx%.0f，目标中心=(%.0f%%,%.0f%%)，中心容差=(%.1f,%.1f)px，"
                "航向容差=%.1fdeg，图像中机器人前方角度=%.1fdeg，航向方向符号=%+.0f"
            ),
            NODE_NAME,
            self.image_width,
            self.image_height,
            100.0 * self.target_center_u_ratio,
            100.0 * self.target_center_v_ratio,
            self.center_tolerance_u_px,
            self.center_tolerance_v_px,
            self.yaw_tolerance_deg,
            self.camera_forward_angle_deg,
            self.yaw_correction_sign,
        )
        rospy.loginfo(
            (
                "%s：视觉控制增益=(前后%.1f,左右%.1f)，最大力=(%.1f,%.1f)，"
                "最小修正力=%.1f，推力步长=%.1f，速度阻尼=(%.1f,%.1f)，"
                "TX/TY方向符号=(%+.0f,%+.0f)，最低对地距离=%.2fm"
            ),
            NODE_NAME,
            self.forward_gain,
            self.lateral_gain,
            self.max_forward_force,
            self.max_lateral_force,
            self.min_correction_force,
            self.force_step,
            self.forward_velocity_damping,
            self.lateral_velocity_damping,
            self.tx_sign,
            self.ty_sign,
            self.min_ground_clearance,
        )

    @staticmethod
    def finite_number(value):
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def mean_angle_deg(angles_deg):
        x_value = sum(math.cos(math.radians(value)) for value in angles_deg)
        y_value = sum(math.sin(math.radians(value)) for value in angles_deg)
        if abs(x_value) < 1e-9 and abs(y_value) < 1e-9:
            return normalize_angle_deg(angles_deg[-1])
        return normalize_angle_deg(math.degrees(math.atan2(y_value, x_value)))

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

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：状态反馈：mode=%d，深度=%.3fm，高度=%.3fm，航向=%.2fdeg，"
                "速度前右下=(%+.3f,%+.3f,%+.3f)m/s，航向角速度=%+.3frad/s"
            ),
            NODE_NAME,
            self.current_status["control_mode"],
            self.current_status["depth"],
            self.current_status["altitude"],
            self.current_status["yaw_deg"],
            self.current_status["vx"],
            self.current_status["vy"],
            self.current_status["vz"],
            self.current_status["wz"],
        )

    def reject_arrow_frame(self, frame_index, reason):
        self.latest_detection = None
        self.arrow_locked = False
        self.locked_detection = None
        if self.state in (self.FORWARD_SEARCH, self.WAIT_FOR_ARROW):
            self.detection_samples = []
        if self.state in (self.ALIGN_HEADING, self.APPROACH_CAMERA):
            self.reset_center_confirmation(reason)
            self.heading_aligned_frame_count = 0
        rospy.loginfo(
            "%s：[箭头识别第%d帧] 无效，原因=%s",
            NODE_NAME,
            frame_index,
            reason,
        )

    def arrow_callback(self, message):
        self.model_frame_index += 1
        frame_index = self.model_frame_index
        now = rospy.Time.now()
        self.last_model_message_time = now

        if self.state == self.INITIAL_HOVER:
            rospy.loginfo_throttle(
                1.0,
                "%s：[箭头识别第%d帧] 当前处于启动悬停，识别帧暂不计数",
                NODE_NAME,
                frame_index,
            )
            return
        if self.state in (self.BODY_CENTERING, self.FINAL_HOLD):
            return

        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as error:
            self.reject_arrow_frame(
                frame_index, "JSON解析失败：{}".format(error)
            )
            return
        if not isinstance(payload, dict):
            self.reject_arrow_frame(frame_index, "JSON根节点不是对象")
            return
        if not bool(payload.get("valid", False)):
            self.reject_arrow_frame(
                frame_index,
                "模型未识别到箭头：{}".format(
                    payload.get("reason") or "valid=false"
                ),
            )
            return

        class_name = str(payload.get("class_name", "")).strip().lower()
        confidence = self.finite_number(payload.get("confidence"))
        center = payload.get("center")
        angle_deg = self.finite_number(payload.get("angle_deg"))
        if class_name != "arrow":
            self.reject_arrow_frame(
                frame_index, "类别={}，不是arrow".format(class_name or "空")
            )
            return
        if confidence is None or confidence < self.min_confidence:
            self.reject_arrow_frame(
                frame_index,
                "置信度={}，低于{:.2f}".format(confidence, self.min_confidence),
            )
            return
        if not isinstance(center, dict):
            self.reject_arrow_frame(frame_index, "缺少center字段")
            return
        center_u = self.finite_number(center.get("u"))
        center_v = self.finite_number(center.get("v"))
        if center_u is None or center_v is None or angle_deg is None:
            self.reject_arrow_frame(frame_index, "中心或箭头角度无效")
            return

        detection = {
            "frame_index": frame_index,
            "received_time": now,
            "confidence": confidence,
            "center_u": center_u,
            "center_v": center_v,
            "angle_deg": normalize_angle_deg(angle_deg),
            "discrete_direction": str(
                payload.get("discrete_direction", "")
            ).strip(),
        }
        self.latest_detection = detection
        self.last_valid_detection_time = now

        if self.state in (self.FORWARD_SEARCH, self.WAIT_FOR_ARROW):
            self.add_detection_sample(detection)
        else:
            error_u, error_v, _, _ = self.detection_center_errors(detection)
            rospy.loginfo(
                (
                    "%s：[箭头识别第%d帧] 有效：置信度=%.3f，中心=(%.1f,%.1f)，"
                    "中心误差=(u=%+.1f,v=%+.1f)px，箭头角度=%.1fdeg，方向=%s"
                ),
                NODE_NAME,
                frame_index,
                confidence,
                center_u,
                center_v,
                error_u,
                error_v,
                detection["angle_deg"],
                detection["discrete_direction"] or "未知",
            )

    def add_detection_sample(self, detection):
        if self.detection_samples:
            gap = (
                detection["received_time"]
                - self.detection_samples[-1]["received_time"]
            ).to_sec()
            if gap > self.detection_timeout:
                rospy.logwarn(
                    "%s：有效箭头帧间隔%.2fs超过%.2fs，稳定计数清零",
                    NODE_NAME,
                    gap,
                    self.detection_timeout,
                )
                self.detection_samples = []

        self.detection_samples.append(detection)
        self.detection_samples = self.detection_samples[
            -self.stable_detection_count :
        ]

        mean_u = sum(item["center_u"] for item in self.detection_samples) / len(
            self.detection_samples
        )
        mean_v = sum(item["center_v"] for item in self.detection_samples) / len(
            self.detection_samples
        )
        mean_angle = self.mean_angle_deg([
            item["angle_deg"] for item in self.detection_samples
        ])
        center_jitter = max(
            math.hypot(item["center_u"] - mean_u, item["center_v"] - mean_v)
            for item in self.detection_samples
        )
        angle_jitter = max(
            abs(angle_difference_deg(item["angle_deg"], mean_angle))
            for item in self.detection_samples
        )
        progress = len(self.detection_samples)

        rospy.loginfo(
            (
                "%s：[箭头识别第%d帧] 第%d/%d帧有效：置信度=%.3f，"
                "中心=(%.1f,%.1f)，箭头角度=%.1fdeg；"
                "中心抖动=%.1f/%.1fpx，角度抖动=%.1f/%.1fdeg"
            ),
            NODE_NAME,
            detection["frame_index"],
            progress,
            self.stable_detection_count,
            detection["confidence"],
            detection["center_u"],
            detection["center_v"],
            detection["angle_deg"],
            center_jitter,
            self.stable_center_tolerance_px,
            angle_jitter,
            self.stable_angle_tolerance_deg,
        )

        if progress < self.stable_detection_count:
            return
        if (
            center_jitter > self.stable_center_tolerance_px
            or angle_jitter > self.stable_angle_tolerance_deg
        ):
            rospy.logwarn(
                (
                    "%s：[箭头识别第%d帧] 已收到%d帧，但稳定性未通过；"
                    "保留当前帧作为新的第1/%d帧"
                ),
                NODE_NAME,
                detection["frame_index"],
                progress,
                self.stable_detection_count,
            )
            self.detection_samples = [detection]
            return

        locked = dict(detection)
        locked["center_u"] = mean_u
        locked["center_v"] = mean_v
        locked["angle_deg"] = mean_angle
        locked["confidence"] = sum(
            item["confidence"] for item in self.detection_samples
        ) / len(self.detection_samples)
        self.locked_detection = locked
        self.latest_detection = locked
        self.arrow_locked = True
        rospy.loginfo(
            (
                "%s：箭头连续%d帧稳定识别成功：平均置信度=%.3f，"
                "平均中心=(%.1f,%.1f)，平均角度=%.1fdeg"
            ),
            NODE_NAME,
            self.stable_detection_count,
            locked["confidence"],
            locked["center_u"],
            locked["center_v"],
            locked["angle_deg"],
        )

    def get_recent_status(self, context):
        if self.current_status is None or self.last_status_time is None:
            rospy.logwarn_throttle(
                2.0,
                "%s：等待状态话题%s，%s暂停",
                NODE_NAME,
                self.status_topic,
                context,
            )
            return None
        age = (rospy.Time.now() - self.last_status_time).to_sec()
        if age > self.status_timeout:
            rospy.logwarn_throttle(
                2.0,
                "%s：状态话题已超时%.2fs（限制%.2fs），%s暂停",
                NODE_NAME,
                age,
                self.status_timeout,
                context,
            )
            return None
        return self.current_status

    def get_frame_pose(self, frame_name, context):
        try:
            self.tf_listener.waitForTransform(
                "map", frame_name, rospy.Time(0), rospy.Duration(0.5)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                "map", frame_name, rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0,
                "%s：无法读取map -> %s坐标变换，%s暂停：%s",
                NODE_NAME,
                frame_name,
                context,
                str(error),
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def get_current_pose(self):
        return self.get_frame_pose("base_link", "机器人本体控制")

    def get_camera_pose(self):
        return self.get_frame_pose(self.camera_frame, "相机与本体坐标补偿")

    def initialize_control(self):
        if self.control_initialized:
            return True
        status = self.get_recent_status("初始化控制基准")
        current = self.get_current_pose()
        if status is None or current is None:
            return False

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        self.hold_z = current.pose.position.z
        self.hold_depth = status["depth"]
        self.wait_hold_position = (
            current.pose.position.x,
            current.pose.position.y,
        )
        self.wait_hold_yaw = current_yaw
        self.target_yaw = current_yaw
        self.control_initialized = True
        self.update_ground_clearance_target(status, current)

        rospy.loginfo(
            (
                "%s：控制基准初始化完成：map位置=(%.3f,%.3f,%.3f)，"
                "当前航向=%.2fdeg，状态深度=%.3fm，高度=%.3fm"
            ),
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            self.hold_z,
            math.degrees(current_yaw),
            status["depth"],
            status["altitude"],
        )
        return True

    def update_ground_clearance_target(self, status, current):
        altitude = status["altitude"]
        if altitude <= 0.0:
            rospy.logwarn_throttle(
                3.0,
                (
                    "%s：高度反馈%.3fm无效，无法主动校验%.2fm最低对地距离；"
                    "当前只保持启动深度"
                ),
                NODE_NAME,
                altitude,
                self.min_ground_clearance,
            )
            return
        if altitude >= self.min_ground_clearance:
            return

        upward_correction = self.min_ground_clearance - altitude
        safe_z = current.pose.position.z - upward_correction
        safe_depth = status["depth"] - upward_correction
        changed = False
        if safe_z < self.hold_z:
            self.hold_z = safe_z
            changed = True
        if safe_depth < self.hold_depth:
            self.hold_depth = safe_depth
            changed = True
        if changed:
            rospy.logwarn(
                (
                    "%s：高度%.3fm低于最低对地距离%.3fm，目标向上修正%.3fm，"
                    "新目标z=%.3f、目标深度=%.3f"
                ),
                NODE_NAME,
                altitude,
                self.min_ground_clearance,
                upward_correction,
                self.hold_z,
                self.hold_depth,
            )

    @staticmethod
    def force_value(value):
        return int(round(clamp(value, -10000.0, 10000.0)))

    def limit_force_step(self, desired, previous):
        return int(round(clamp(
            desired,
            previous - self.force_step,
            previous + self.force_step,
        )))

    @staticmethod
    def apply_minimum_force(value, minimum, maximum):
        value = clamp(value, -maximum, maximum)
        if value == 0.0 or abs(value) >= minimum:
            return value
        return math.copysign(minimum, value)

    def velocity_feedback_command(
        self,
        base_forward_force,
        base_right_force,
        status,
        max_forward_force,
        max_right_force,
    ):
        damping_vx = (
            status["vx"] if abs(status["vx"]) > self.speed_deadband else 0.0
        )
        damping_vy = (
            status["vy"] if abs(status["vy"]) > self.speed_deadband else 0.0
        )
        physical_forward = clamp(
            base_forward_force - self.forward_velocity_damping * damping_vx,
            -max_forward_force,
            max_forward_force,
        )
        physical_right = clamp(
            base_right_force - self.lateral_velocity_damping * damping_vy,
            -max_right_force,
            max_right_force,
        )
        return (
            self.tx_sign * physical_forward,
            self.ty_sign * physical_right,
        )

    def publish_motion(
        self,
        desired_tx,
        desired_ty,
        target_yaw,
        reason,
        rate_limit=True,
    ):
        if not self.initialize_control():
            return False
        status = self.get_recent_status(reason)
        current = self.get_current_pose()
        if status is None or current is None:
            return False
        self.update_ground_clearance_target(status, current)

        if rate_limit:
            tx_value = self.limit_force_step(desired_tx, self.last_tx)
            ty_value = self.limit_force_step(desired_ty, self.last_ty)
        else:
            tx_value = self.force_value(desired_tx)
            ty_value = self.force_value(desired_ty)
        self.last_tx = tx_value
        self.last_ty = ty_value

        command = PoseNEDcmd()
        command.mode = MODE_DEPTH_HEADING
        command.target.header.frame_id = "map"
        command.target.header.stamp = rospy.Time.now()
        command.target.pose.position.x = current.pose.position.x
        command.target.pose.position.y = current.pose.position.y
        command.target.pose.position.z = self.hold_z
        command.target.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0, 0.0, target_yaw
        ))
        command.force.TX = self.force_value(tx_value)
        command.force.TY = self.force_value(ty_value)
        command.force.TZ = 0
        command.force.MX = 0
        command.force.MY = 0
        command.force.MZ = 0
        self.pose_cmd_pub.publish(command)

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：控制指令：mode=3，目标深度z=%.3f，目标航向=%.2fdeg，"
                "TX=%d，TY=%d，其余力/力矩=0，原因=%s"
            ),
            NODE_NAME,
            self.hold_z,
            math.degrees(target_yaw),
            command.force.TX,
            command.force.TY,
            reason,
        )
        return True

    def publish_position_hold(self, position, target_yaw, reason):
        status = self.get_recent_status(reason)
        current = self.get_current_pose()
        if status is None or current is None or position is None:
            if self.control_initialized:
                self.publish_motion(
                    0.0, 0.0, target_yaw, "{}，反馈不可用".format(reason), False
                )
            return None

        dx = position[0] - current.pose.position.x
        dy = position[1] - current.pose.position.y
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        forward_error, right_error = map_error_to_body(
            dx, dy, current_yaw
        )

        base_forward_force = 0.0
        if abs(forward_error) > self.hold_position_tolerance:
            base_forward_force = self.hold_forward_position_gain * forward_error
        base_right_force = 0.0
        if abs(right_error) > self.hold_position_tolerance:
            base_right_force = self.hold_lateral_position_gain * right_error

        desired_tx, desired_ty = self.velocity_feedback_command(
            base_forward_force,
            base_right_force,
            status,
            self.hold_max_force,
            self.hold_max_force,
        )
        if not self.publish_motion(
            desired_tx, desired_ty, target_yaw, reason
        ):
            return None

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：定点保持：位置误差=(前%+.3f,右%+.3f)m，"
                "速度=(前%+.3f,右%+.3f)m/s，指令=(TX=%d,TY=%d)"
            ),
            NODE_NAME,
            forward_error,
            right_error,
            status["vx"],
            status["vy"],
            self.last_tx,
            self.last_ty,
        )
        return {
            "status": status,
            "current": current,
            "forward_error": forward_error,
            "right_error": right_error,
        }

    def detection_center_errors(self, detection):
        desired_u = self.image_width * self.target_center_u_ratio
        desired_v = self.image_height * self.target_center_v_ratio
        error_u = detection["center_u"] - desired_u
        error_v = detection["center_v"] - desired_v
        normalized_u = error_u / max(0.5 * self.image_width, 1.0)
        normalized_v = error_v / max(0.5 * self.image_height, 1.0)
        return error_u, error_v, normalized_u, normalized_v

    def update_target_yaw(self, detection, current):
        if detection["frame_index"] == self.last_target_yaw_frame_index:
            return
        self.last_target_yaw_frame_index = detection["frame_index"]

        current_yaw_deg = math.degrees(yaw_from_quaternion(
            current.pose.orientation
        ))
        relative_correction_deg = self.yaw_correction_sign * normalize_angle_deg(
            self.camera_forward_angle_deg - detection["angle_deg"]
        )
        measured_target_deg = normalize_angle_deg(
            current_yaw_deg + relative_correction_deg
        )

        if self.target_yaw is None:
            filtered_target_deg = measured_target_deg
        else:
            previous_target_deg = math.degrees(self.target_yaw)
            target_delta_deg = angle_difference_deg(
                measured_target_deg, previous_target_deg
            )
            applied_step_deg = clamp(
                self.yaw_target_filter_alpha * target_delta_deg,
                -self.yaw_target_max_step_deg,
                self.yaw_target_max_step_deg,
            )
            filtered_target_deg = normalize_angle_deg(
                previous_target_deg + applied_step_deg
            )
        self.target_yaw = math.radians(filtered_target_deg)

        rospy.loginfo(
            (
                "%s：[箭头识别第%d帧] 航向换算：当前航向=%.2fdeg，"
                "箭头图像角度=%.2fdeg，相对修正=%+.2fdeg，"
                "瞬时目标=%.2fdeg，滤波目标=%.2fdeg"
            ),
            NODE_NAME,
            detection["frame_index"],
            current_yaw_deg,
            detection["angle_deg"],
            relative_correction_deg,
            measured_target_deg,
            filtered_target_deg,
        )

    def start_alignment(self):
        current = self.get_current_pose()
        status = self.get_recent_status("进入箭头细对准")
        if current is None or status is None or self.locked_detection is None:
            return False

        self.target_yaw = None
        self.last_target_yaw_frame_index = 0
        self.update_target_yaw(self.locked_detection, current)
        self.latest_detection = self.locked_detection
        self.heading_hold_position = (
            current.pose.position.x,
            current.pose.position.y,
        )
        self.heading_aligned_frame_count = 0
        self.last_heading_frame_index = 0
        self.centered_frame_count = 0
        self.last_alignment_frame_index = 0
        self.alignment_loss_hold_position = None
        self.set_state(
            self.ALIGN_HEADING,
            "箭头已连续{}帧稳定识别，停止直行并优先对齐航向".format(
                self.stable_detection_count
            ),
        )
        return True

    def reset_center_confirmation(self, reason):
        previous = self.centered_frame_count
        self.centered_frame_count = 0
        if previous > 0:
            rospy.loginfo(
                "%s：连续对准确认%d -> 0，原因=%s",
                NODE_NAME,
                previous,
                reason,
            )

    def set_state(self, state, reason):
        previous = self.state
        self.state = state
        self.state_started = rospy.Time.now()
        rospy.loginfo(
            "%s：状态切换：%s -> %s，原因=%s",
            NODE_NAME,
            previous,
            state,
            reason,
        )

    def reset_arrow_lock(self):
        self.detection_samples = []
        self.arrow_locked = False
        self.locked_detection = None
        self.latest_detection = None
        self.last_valid_detection_time = None

    def control_initial_hover(self):
        if not self.initialize_control():
            return
        if self.initial_hover_started is None:
            self.initial_hover_started = rospy.Time.now()
            rospy.loginfo(
                "%s：mode3启动悬停正式开始，持续%.1fs",
                NODE_NAME,
                self.initial_hover_seconds,
            )

        self.publish_position_hold(
            self.wait_hold_position,
            self.wait_hold_yaw,
            "任务开始后mode3定深定向悬停",
        )
        elapsed = (rospy.Time.now() - self.initial_hover_started).to_sec()
        rospy.loginfo_throttle(
            1.0,
            "%s：启动悬停 %.1f/%.1fs，识别帧暂不计数",
            NODE_NAME,
            elapsed,
            self.initial_hover_seconds,
        )
        if elapsed < self.initial_hover_seconds:
            return

        current = self.get_current_pose()
        if current is None:
            return
        self.search_origin = (
            current.pose.position.x,
            current.pose.position.y,
        )
        self.search_yaw = yaw_from_quaternion(current.pose.orientation)
        self.wait_hold_yaw = self.search_yaw
        self.reset_arrow_lock()
        self.set_state(
            self.FORWARD_SEARCH,
            "启动悬停完成，保持当前航向并低速向前搜索箭头",
        )

    def control_forward_search(self):
        if self.arrow_locked:
            self.start_alignment()
            return
        status = self.get_recent_status("低速直行搜索箭头")
        current = self.get_current_pose()
        if status is None or current is None:
            return
        if self.search_origin is None or self.search_yaw is None:
            self.search_origin = (
                current.pose.position.x,
                current.pose.position.y,
            )
            self.search_yaw = yaw_from_quaternion(current.pose.orientation)

        delta_north = current.pose.position.x - self.search_origin[0]
        delta_east = current.pose.position.y - self.search_origin[1]
        forward_distance, lateral_drift = map_error_to_body(
            delta_north,
            delta_east,
            self.search_yaw,
        )
        if forward_distance >= self.max_search_distance:
            self.finish_task(
                False,
                "直行搜索达到安全距离{:.2f}m仍未稳定识别箭头".format(
                    self.max_search_distance
                ),
            )
            return

        desired_tx, desired_ty = self.velocity_feedback_command(
            self.search_forward_force,
            0.0,
            status,
            self.search_forward_force,
            self.max_lateral_force,
        )
        self.publish_motion(
            desired_tx,
            desired_ty,
            self.search_yaw,
            "mode3保持启动航向并低速直行搜索箭头",
        )
        rospy.loginfo_throttle(
            1.0,
            (
                "%s：直行搜索：前进距离=%.3f/%.3fm，横向漂移=%+.3fm，"
                "速度=(前%+.3f,右%+.3f)m/s，指令=(TX=%d,TY=%d)，"
                "稳定识别进度=%d/%d帧"
            ),
            NODE_NAME,
            forward_distance,
            self.max_search_distance,
            lateral_drift,
            status["vx"],
            status["vy"],
            self.last_tx,
            self.last_ty,
            len(self.detection_samples),
            self.stable_detection_count,
        )
        if self.last_model_message_time is None:
            rospy.logwarn_throttle(
                2.0,
                "%s：直行搜索中尚未收到%s，请检查箭头模型",
                NODE_NAME,
                self.arrow_topic,
            )

    def return_to_wait_for_arrow(self, reason):
        current = self.get_current_pose()
        if current is not None:
            self.wait_hold_position = (
                current.pose.position.x,
                current.pose.position.y,
            )
            self.wait_hold_yaw = (
                self.target_yaw
                if self.target_yaw is not None
                else yaw_from_quaternion(current.pose.orientation)
            )
        self.detection_samples = []
        self.arrow_locked = False
        self.locked_detection = None
        self.latest_detection = None
        self.heading_aligned_frame_count = 0
        self.last_heading_frame_index = 0
        self.reset_center_confirmation(reason)
        self.set_state(self.WAIT_FOR_ARROW, reason)

    def control_wait_for_arrow(self):
        if not self.initialize_control():
            return
        self.publish_position_hold(
            self.wait_hold_position,
            self.wait_hold_yaw,
            "箭头丢失后保持当前位置，重新等待连续稳定识别",
        )
        if self.arrow_locked:
            self.start_alignment()
            return

        if self.last_model_message_time is None:
            rospy.logwarn_throttle(
                2.0,
                "%s：尚未收到箭头模型话题%s，请检查模型节点是否启动",
                NODE_NAME,
                self.arrow_topic,
            )
        else:
            model_age = (rospy.Time.now() - self.last_model_message_time).to_sec()
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：箭头丢失后定点重识别，模型消息距今%.2fs，"
                    "当前稳定识别进度=%d/%d帧"
                ),
                NODE_NAME,
                model_age,
                len(self.detection_samples),
                self.stable_detection_count,
            )

    def detection_is_fresh(self, now):
        if self.latest_detection is None or self.last_valid_detection_time is None:
            return False
        return (
            now - self.last_valid_detection_time
        ).to_sec() <= self.detection_timeout

    def control_heading_alignment(self):
        now = rospy.Time.now()
        if not self.detection_is_fresh(now):
            self.publish_position_hold(
                self.heading_hold_position,
                self.target_yaw,
                "航向对准时箭头暂时丢失，保持当前位置",
            )
            if (
                self.last_valid_detection_time is None
                or (now - self.last_valid_detection_time).to_sec()
                > self.detection_timeout
            ):
                self.return_to_wait_for_arrow(
                    "航向对准时箭头丢失超过{:.1f}s，定点重新识别3帧".format(
                        self.detection_timeout
                    )
                )
            return

        detection = self.latest_detection
        status = self.get_recent_status("优先对齐箭头航向")
        current = self.get_current_pose()
        if status is None or current is None:
            return
        self.update_target_yaw(detection, current)
        hold_result = self.publish_position_hold(
            self.heading_hold_position,
            self.target_yaw,
            "先保持当前位置，仅对齐机器人和箭头航向",
        )
        if hold_result is None:
            self.heading_aligned_frame_count = 0
            return

        current_yaw_deg = math.degrees(yaw_from_quaternion(
            current.pose.orientation
        ))
        yaw_error_deg = angle_difference_deg(
            current_yaw_deg, math.degrees(self.target_yaw)
        )
        if detection["frame_index"] != self.last_heading_frame_index:
            self.last_heading_frame_index = detection["frame_index"]
            if abs(yaw_error_deg) <= self.yaw_tolerance_deg:
                self.heading_aligned_frame_count = min(
                    self.heading_aligned_frame_count + 1,
                    self.heading_stable_detection_count,
                )
                rospy.loginfo(
                    (
                        "%s：[箭头识别第%d帧] 航向对准第%d/%d帧有效："
                        "当前航向=%.1fdeg，目标航向=%.1fdeg，误差=%+.1fdeg"
                    ),
                    NODE_NAME,
                    detection["frame_index"],
                    self.heading_aligned_frame_count,
                    self.heading_stable_detection_count,
                    current_yaw_deg,
                    math.degrees(self.target_yaw),
                    yaw_error_deg,
                )
            else:
                if self.heading_aligned_frame_count > 0:
                    rospy.loginfo(
                        "%s：航向连续确认%d -> 0，当前误差=%+.1fdeg",
                        NODE_NAME,
                        self.heading_aligned_frame_count,
                        yaw_error_deg,
                    )
                self.heading_aligned_frame_count = 0

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：优先航向对准：箭头角度=%.1fdeg，当前/目标航向="
                "(%.1f/%.1f)deg，误差=%+.1f/%.1fdeg，wz=%+.3f/%.3frad/s，"
                "水平速度=%.3f/%.3fm/s，确认=%d/%d帧"
            ),
            NODE_NAME,
            detection["angle_deg"],
            current_yaw_deg,
            math.degrees(self.target_yaw),
            yaw_error_deg,
            self.yaw_tolerance_deg,
            status["wz"],
            self.max_yaw_rate,
            math.hypot(status["vx"], status["vy"]),
            self.max_horizontal_speed,
            self.heading_aligned_frame_count,
            self.heading_stable_detection_count,
        )

        if self.heading_aligned_frame_count < self.heading_stable_detection_count:
            return
        if not self.alignment_physical_stable(status, current, yaw_error_deg):
            rospy.loginfo_throttle(
                1.0,
                "%s：航向已稳定%d帧，等待机器人停稳后开始缓慢靠近",
                NODE_NAME,
                self.heading_aligned_frame_count,
            )
            return

        self.centered_frame_count = 0
        self.last_alignment_frame_index = 0
        self.set_state(
            self.APPROACH_CAMERA,
            "机器人航向已与箭头一致，开始保持航向并缓慢视觉靠近",
        )

    def alignment_physical_stable(self, status, current, yaw_error_deg):
        depth_error = status["depth"] - self.hold_depth
        return (
            status["control_mode"] == MODE_DEPTH_HEADING
            and math.hypot(status["vx"], status["vy"])
            <= self.max_horizontal_speed
            and abs(status["vz"]) <= self.max_vertical_speed
            and abs(status["wz"]) <= self.max_yaw_rate
            and abs(depth_error) <= self.max_depth_error
            and abs(yaw_error_deg) <= self.yaw_tolerance_deg
            and self.last_tx == 0
            and self.last_ty == 0
        )

    def update_alignment_confirmation(
        self,
        detection,
        error_u,
        error_v,
        yaw_error_deg,
        current_yaw_deg,
    ):
        if detection["frame_index"] == self.last_alignment_frame_index:
            return
        self.last_alignment_frame_index = detection["frame_index"]

        centered = (
            abs(error_u) <= self.center_tolerance_u_px
            and abs(error_v) <= self.center_tolerance_v_px
        )
        yaw_aligned = abs(yaw_error_deg) <= self.yaw_tolerance_deg
        if centered and yaw_aligned:
            self.centered_frame_count = min(
                self.centered_frame_count + 1,
                self.center_stable_detection_count,
            )
            rospy.loginfo(
                (
                    "%s：[箭头识别第%d帧] 对准确认第%d/%d帧有效："
                    "中心误差=(u=%+.1f,v=%+.1f)px，箭头角度=%.1fdeg，"
                    "当前/目标航向=(%.1f/%.1f)deg，航向误差=%+.1fdeg"
                ),
                NODE_NAME,
                detection["frame_index"],
                self.centered_frame_count,
                self.center_stable_detection_count,
                error_u,
                error_v,
                detection["angle_deg"],
                current_yaw_deg,
                math.degrees(self.target_yaw),
                yaw_error_deg,
            )
        else:
            self.reset_center_confirmation(
                "中心或航向超出容差"
            )
            rospy.loginfo(
                (
                    "%s：[箭头识别第%d帧] 本帧有效但尚未完成对准："
                    "中心误差=(u=%+.1f/%+.1f,v=%+.1f/%+.1f)px，"
                    "航向误差=%+.1f/%.1fdeg"
                ),
                NODE_NAME,
                detection["frame_index"],
                error_u,
                self.center_tolerance_u_px,
                error_v,
                self.center_tolerance_v_px,
                yaw_error_deg,
                self.yaw_tolerance_deg,
            )

    def control_camera_approach(self):
        now = rospy.Time.now()
        if not self.detection_is_fresh(now):
            current = self.get_current_pose()
            if current is not None and self.alignment_loss_hold_position is None:
                self.alignment_loss_hold_position = (
                    current.pose.position.x,
                    current.pose.position.y,
                )
            self.publish_position_hold(
                self.alignment_loss_hold_position,
                self.target_yaw,
                "视觉靠近时箭头暂时丢失，停止移动并保持丢失位置",
            )
            if (
                self.last_valid_detection_time is None
                or (now - self.last_valid_detection_time).to_sec()
                > self.detection_timeout
            ):
                self.return_to_wait_for_arrow(
                    "视觉靠近时箭头超过{:.1f}s未恢复，定点重新识别3帧".format(
                        self.detection_timeout
                    )
                )
            return

        detection = self.latest_detection
        status = self.get_recent_status("保持箭头航向并缓慢视觉靠近")
        current = self.get_current_pose()
        if status is None or current is None:
            self.publish_motion(
                0.0, 0.0, self.target_yaw, "视觉靠近反馈不可用，清零水平推力", False
            )
            self.reset_center_confirmation("状态或TF反馈不可用")
            return
        self.alignment_loss_hold_position = None
        self.update_target_yaw(detection, current)

        error_u, error_v, normalized_u, normalized_v = (
            self.detection_center_errors(detection)
        )
        centered_u = abs(error_u) <= self.center_tolerance_u_px
        centered_v = abs(error_v) <= self.center_tolerance_v_px

        visual_forward_force = 0.0
        if not centered_v:
            visual_forward_force = self.apply_minimum_force(
                -self.forward_gain * normalized_v,
                self.min_correction_force,
                self.max_forward_force,
            )
        visual_right_force = 0.0
        if not centered_u:
            visual_right_force = self.apply_minimum_force(
                self.lateral_gain * normalized_u,
                self.min_correction_force,
                self.max_lateral_force,
            )

        desired_tx, desired_ty = self.velocity_feedback_command(
            visual_forward_force,
            visual_right_force,
            status,
            self.max_forward_force,
            self.max_lateral_force,
        )
        if not self.publish_motion(
            desired_tx,
            desired_ty,
            self.target_yaw,
            "保持箭头航向并依据中心像素缓慢靠近",
        ):
            self.reset_center_confirmation("运动指令发布失败")
            return

        current_yaw_deg = math.degrees(yaw_from_quaternion(
            current.pose.orientation
        ))
        yaw_error_deg = angle_difference_deg(
            current_yaw_deg, math.degrees(self.target_yaw)
        )
        depth_error = status["depth"] - self.hold_depth
        self.update_alignment_confirmation(
            detection,
            error_u,
            error_v,
            yaw_error_deg,
            current_yaw_deg,
        )

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：缓慢视觉靠近：中心=(%.1f,%.1f)，误差=(u=%+.1f,v=%+.1f)px，"
                "箭头角度=%.1fdeg，目标航向=%.1fdeg，航向误差=%+.1fdeg，"
                "速度=(前%+.3f,右%+.3f,下%+.3f)m/s，wz=%+.3frad/s，"
                "mode=%d，深度误差=%+.3fm，指令=(TX=%d,TY=%d)"
            ),
            NODE_NAME,
            detection["center_u"],
            detection["center_v"],
            error_u,
            error_v,
            detection["angle_deg"],
            math.degrees(self.target_yaw),
            yaw_error_deg,
            status["vx"],
            status["vy"],
            status["vz"],
            status["wz"],
            status["control_mode"],
            depth_error,
            self.last_tx,
            self.last_ty,
        )

        if self.centered_frame_count < self.center_stable_detection_count:
            return
        if not self.alignment_physical_stable(status, current, yaw_error_deg):
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：中心和航向已连续满足%d帧，等待机器人实际停稳："
                    "水平速度=%.3f/%.3f，垂直速度=%+.3f/%.3f，"
                    "wz=%+.3f/%.3f，mode=%d，深度误差=%+.3f/%.3f，"
                    "航向误差=%+.1f/%.1f，指令=(%d,%d)"
                ),
                NODE_NAME,
                self.centered_frame_count,
                math.hypot(status["vx"], status["vy"]),
                self.max_horizontal_speed,
                status["vz"],
                self.max_vertical_speed,
                status["wz"],
                self.max_yaw_rate,
                status["control_mode"],
                depth_error,
                self.max_depth_error,
                yaw_error_deg,
                self.yaw_tolerance_deg,
                self.last_tx,
                self.last_ty,
            )
            return

        camera_pose = self.get_camera_pose()
        if camera_pose is None:
            rospy.logwarn_throttle(
                1.0,
                "%s：摄像头已居中，但暂时无法读取map -> %s，保持当前点等待TF",
                NODE_NAME,
                self.camera_frame,
            )
            return

        self.arrow_map_position = (
            camera_pose.pose.position.x,
            camera_pose.pose.position.y,
        )
        self.final_hold_position = self.arrow_map_position
        camera_forward, camera_right = map_error_to_body(
            camera_pose.pose.position.x - current.pose.position.x,
            camera_pose.pose.position.y - current.pose.position.y,
            yaw_from_quaternion(current.pose.orientation),
        )
        rospy.loginfo(
            (
                "%s：摄像头已位于箭头正上方，使用现有TF捕获箭头map坐标："
                "base_link=(%.3f,%.3f)，%s=(%.3f,%.3f)，"
                "相机相对本体=(前%+.3f,右%+.3f)m；"
                "下一阶段将base_link移动到箭头map坐标"
            ),
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            self.camera_frame,
            camera_pose.pose.position.x,
            camera_pose.pose.position.y,
            camera_forward,
            camera_right,
        )
        self.set_state(
            self.BODY_CENTERING,
            "视觉中心已稳定，开始补偿相机与机器人本体坐标偏移",
        )

    def control_body_centering(self):
        hold_result = self.publish_position_hold(
            self.arrow_map_position,
            self.target_yaw,
            "调用map_error_to_body控制base_link移到箭头map坐标",
        )
        if hold_result is None:
            return

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：本体坐标补偿：base_link目标=(%.3f,%.3f)，"
                "误差=(前%+.3f,右%+.3f)m，速度=(前%+.3f,右%+.3f)m/s，"
                "指令=(TX=%d,TY=%d)"
            ),
            NODE_NAME,
            self.arrow_map_position[0],
            self.arrow_map_position[1],
            hold_result["forward_error"],
            hold_result["right_error"],
            hold_result["status"]["vx"],
            hold_result["status"]["vy"],
            self.last_tx,
            self.last_ty,
        )
        if not self.final_hold_is_stable(hold_result):
            return

        self.final_hold_stable_started = None
        self.set_state(
            self.FINAL_HOLD,
            "base_link已移动到箭头map坐标，开始最终稳定悬停10秒",
        )

    def final_hold_is_stable(self, hold_result):
        status = hold_result["status"]
        current = hold_result["current"]
        current_yaw_deg = math.degrees(yaw_from_quaternion(
            current.pose.orientation
        ))
        yaw_error_deg = angle_difference_deg(
            current_yaw_deg, math.degrees(self.target_yaw)
        )
        depth_error = status["depth"] - self.hold_depth
        return (
            status["control_mode"] == MODE_DEPTH_HEADING
            and abs(hold_result["forward_error"])
            <= self.final_position_tolerance
            and abs(hold_result["right_error"])
            <= self.final_position_tolerance
            and math.hypot(status["vx"], status["vy"])
            <= self.max_horizontal_speed
            and abs(status["vz"]) <= self.max_vertical_speed
            and abs(status["wz"]) <= self.max_yaw_rate
            and abs(depth_error) <= self.max_depth_error
            and abs(yaw_error_deg) <= self.yaw_tolerance_deg
            and self.last_tx == 0
            and self.last_ty == 0
        )

    def control_final_hold(self):
        hold_result = self.publish_position_hold(
            self.final_hold_position,
            self.target_yaw,
            "箭头对准完成后的最终定点保持",
        )
        if hold_result is None:
            self.final_hold_stable_started = None
            return

        now = rospy.Time.now()
        stable = self.final_hold_is_stable(hold_result)
        if stable:
            if self.final_hold_stable_started is None:
                self.final_hold_stable_started = now
                rospy.loginfo(
                    "%s：最终定点已满足稳定条件，开始累计%.1fs稳定保持时间",
                    NODE_NAME,
                    self.final_hold_seconds,
                )
            stable_elapsed = (now - self.final_hold_stable_started).to_sec()
            rospy.loginfo_throttle(
                1.0,
                "%s：最终定点稳定保持 %.1f/%.1fs",
                NODE_NAME,
                stable_elapsed,
                self.final_hold_seconds,
            )
            if stable_elapsed >= self.final_hold_seconds:
                self.finish_task(
                    True,
                    "机器人本体位于箭头正上方、航向一致，并稳定悬停{:.1f}s".format(
                        self.final_hold_seconds
                    ),
                )
                return
        else:
            if self.final_hold_stable_started is not None:
                rospy.loginfo(
                    "%s：最终定点稳定条件被打断，稳定保持计时重新开始",
                    NODE_NAME,
                )
            self.final_hold_stable_started = None

        state_elapsed = (now - self.state_started).to_sec()
        if state_elapsed >= self.final_hold_timeout:
            self.finish_task(
                False,
                "最终定点{:.1f}s内未能连续稳定保持{:.1f}s".format(
                    self.final_hold_timeout,
                    self.final_hold_seconds,
                ),
            )

    def finish_task(self, success, detail):
        if self.task_finished:
            return
        self.task_finished = True
        if self.control_initialized:
            hold_position = self.final_hold_position or self.wait_hold_position
            if self.final_hold_position is None:
                current = self.get_current_pose()
                if current is not None:
                    hold_position = (
                        current.pose.position.x,
                        current.pose.position.y,
                    )
            hold_yaw = (
                self.target_yaw
                if self.target_yaw is not None
                else self.wait_hold_yaw
            )
            self.publish_position_hold(
                hold_position,
                hold_yaw,
                "任务结束前保持最后定点",
            )
        state = "finished" if success else "failed"
        self.finished_pub.publish(String(
            data="{} {}: {}".format(NODE_NAME, state, detail)
        ))
        if success:
            rospy.loginfo("%s：任务成功：%s", NODE_NAME, detail)
        else:
            rospy.logerr("%s：任务失败：%s", NODE_NAME, detail)
        rospy.signal_shutdown("{} {}".format(NODE_NAME, state))

    def on_shutdown(self):
        if not self.control_initialized:
            return
        hold_position = self.final_hold_position or self.wait_hold_position
        if self.final_hold_position is None:
            current = self.get_current_pose()
            if current is not None:
                hold_position = (
                    current.pose.position.x,
                    current.pose.position.y,
                )
        hold_yaw = (
            self.target_yaw
            if self.target_yaw is not None
            else self.wait_hold_yaw
        )
        self.publish_position_hold(
            hold_position,
            hold_yaw,
            "节点退出，保持最后目标并清除额外运动意图",
        )

    def run(self):
        rospy.loginfo(
            (
                "%s：等待/status/auv和TF；随后执行mode3悬停10秒、低速直行搜索、"
                "航向优先对准、缓慢视觉靠近、本体坐标补偿和最终悬停"
            ),
            NODE_NAME,
        )
        while not rospy.is_shutdown():
            if not self.task_finished:
                elapsed = (rospy.Time.now() - self.task_started).to_sec()
                if (
                    self.state != self.FINAL_HOLD
                    and elapsed >= self.max_wait_seconds
                ):
                    self.finish_task(
                        False,
                        "搜索、对准或本体坐标补偿超过{:.1f}s仍未完成".format(
                            self.max_wait_seconds
                        ),
                    )
                    break

                if self.state == self.INITIAL_HOVER:
                    self.control_initial_hover()
                elif self.state == self.FORWARD_SEARCH:
                    self.control_forward_search()
                elif self.state == self.WAIT_FOR_ARROW:
                    self.control_wait_for_arrow()
                elif self.state == self.ALIGN_HEADING:
                    self.control_heading_alignment()
                elif self.state == self.APPROACH_CAMERA:
                    self.control_camera_approach()
                elif self.state == self.BODY_CENTERING:
                    self.control_body_centering()
                elif self.state == self.FINAL_HOLD:
                    self.control_final_hold()
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME)
    try:
        Task3AcquireAreaTest().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("%s：未处理异常：%s", NODE_NAME, str(error))
        raise
