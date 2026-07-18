#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
任务3子任务1：使用 motion_supervisor 完成箭头搜索、对准和最终定位。

本节点只生成 map 绝对目标，不直接发布 /cmd/pose/ned，也不计算 TX、TY、MZ。
motion_supervisor 负责平移、主动刹停、最终转向和 mode=4 定点接管。
"""

import json
import math

import rospy
import tf
from auv_control.msg import AUVData, MotionState
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task3_1_acquire_area_motion"


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_angle_rad(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def normalize_angle_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def yaw_from_quaternion(quaternion):
    return euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ])[2]


class Task3AcquireAreaMotionTest(object):
    WAIT_FOR_CONTROL = "等待运动状态机和反馈"
    INITIAL_HOVER = "启动定点悬停"
    FORWARD_SEARCH = "向前搜索箭头"
    CANCEL_WAIT = "主动刹停并等待定点接管"
    WAIT_FOR_ARROW = "定点重新识别箭头"
    CENTER_CAMERA = "箭头图像居中"
    CONFIRM_DIRECTION = "定点确认箭头方向"
    ALIGN_HEADING = "保持居中并对齐箭头航向"
    MOVE_FRONT_THRUSTER = "前移到前垂推对齐位置"
    FINAL_HOLD = "最终定点保持"

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

    def __init__(self):
        self.rate_hz = float(rospy.get_param("~rate", 5.0))
        self.arrow_topic = str(rospy.get_param(
            "~arrow_topic", "/arrow/direction"
        )).strip()
        self.motion_goal_topic = str(rospy.get_param(
            "~motion_goal_topic", "/cmd/motion/goal"
        )).strip()
        self.motion_cancel_topic = str(rospy.get_param(
            "~motion_cancel_topic", "/cmd/motion/cancel"
        )).strip()
        self.motion_state_topic = str(rospy.get_param(
            "~motion_state_topic", "/motion/state"
        )).strip()
        self.status_topic = str(rospy.get_param(
            "~status_topic", "/status/auv"
        )).strip()

        self.min_confidence = float(rospy.get_param(
            "~min_confidence", 0.35
        ))
        self.detection_timeout = float(rospy.get_param(
            "~detection_timeout", 1.0
        ))
        self.visual_loss_cancel_seconds = float(rospy.get_param(
            "~visual_loss_cancel_seconds", 0.5
        ))
        self.stable_detection_count = int(rospy.get_param(
            "~stable_detection_count", 3
        ))
        self.stable_center_tolerance_px = float(rospy.get_param(
            "~stable_center_tolerance_px", 40.0
        ))
        self.stable_angle_tolerance_deg = float(rospy.get_param(
            "~stable_angle_tolerance_deg", 12.0
        ))
        self.center_stable_detection_count = int(rospy.get_param(
            "~center_stable_detection_count", 5
        ))
        self.heading_stable_detection_count = int(rospy.get_param(
            "~heading_stable_detection_count", 3
        ))

        self.image_width = float(rospy.get_param("~image_width", 640.0))
        self.image_height = float(rospy.get_param("~image_height", 480.0))
        self.target_center_u_ratio = float(rospy.get_param(
            "~target_center_u_ratio", 0.5
        ))
        self.target_center_v_ratio = float(rospy.get_param(
            "~target_center_v_ratio", 0.5
        ))
        self.center_tolerance_u_px = float(rospy.get_param(
            "~center_tolerance_u_px", 35.0
        ))
        self.center_tolerance_v_px = float(rospy.get_param(
            "~center_tolerance_v_px", 35.0
        ))
        self.visual_forward_gain_m = float(rospy.get_param(
            "~visual_forward_gain_m", 0.20
        ))
        self.visual_lateral_gain_m = float(rospy.get_param(
            "~visual_lateral_gain_m", 0.20
        ))
        self.visual_max_step_m = float(rospy.get_param(
            "~visual_max_step_m", 0.08
        ))
        self.visual_min_step_m = float(rospy.get_param(
            "~visual_min_step_m", 0.01
        ))
        self.visual_forward_sign = float(rospy.get_param(
            "~visual_forward_sign", 1.0
        ))
        self.visual_lateral_sign = float(rospy.get_param(
            "~visual_lateral_sign", 1.0
        ))

        self.camera_forward_angle_deg = float(rospy.get_param(
            "~camera_forward_angle_deg", 90.0
        ))
        self.yaw_correction_sign = float(rospy.get_param(
            "~yaw_correction_sign", 1.0
        ))
        self.yaw_tolerance_deg = float(rospy.get_param(
            "~yaw_tolerance_deg", 10.0
        ))

        self.initial_hover_seconds = float(rospy.get_param(
            "~initial_hover_seconds", 10.0
        ))
        self.search_forward_distance = float(rospy.get_param(
            "~search_forward_distance", 1.20
        ))
        self.front_thruster_forward_offset = float(rospy.get_param(
            "~front_thruster_forward_offset", 0.35
        ))
        self.final_hold_seconds = float(rospy.get_param(
            "~final_hold_seconds", 10.0
        ))
        self.final_hold_timeout = float(rospy.get_param(
            "~final_hold_timeout", 30.0
        ))
        self.max_wait_seconds = float(rospy.get_param(
            "~max_wait_seconds", 300.0
        ))
        self.cancel_timeout = float(rospy.get_param(
            "~cancel_timeout", 15.0
        ))

        self.motion_state_timeout = float(rospy.get_param(
            "~motion_state_timeout", 0.5
        ))
        self.motion_startup_timeout = float(rospy.get_param(
            "~motion_startup_timeout", 10.0
        ))
        self.status_timeout = float(rospy.get_param(
            "~status_timeout", 0.5
        ))
        self.goal_match_position_tolerance = float(rospy.get_param(
            "~goal_match_position_tolerance", 0.03
        ))
        self.goal_match_depth_tolerance = float(rospy.get_param(
            "~goal_match_depth_tolerance", 0.03
        ))
        self.goal_match_yaw_tolerance_deg = float(rospy.get_param(
            "~goal_match_yaw_tolerance_deg", 2.0
        ))
        self.max_depth_error = float(rospy.get_param(
            "~max_depth_error", 0.08
        ))
        self.min_ground_clearance = float(rospy.get_param(
            "~min_ground_clearance", 0.40
        ))
        self.ground_clearance_goal_update_threshold = float(rospy.get_param(
            "~ground_clearance_goal_update_threshold", 0.01
        ))
        self.log_interval = float(rospy.get_param(
            "~log_interval", 1.0
        ))
        self.warning_log_interval = float(rospy.get_param(
            "~warning_log_interval", 2.0
        ))

        self.validate_params()
        self.rate = rospy.Rate(self.rate_hz)
        self.tf_listener = tf.TransformListener()

        self.goal_pub = rospy.Publisher(
            self.motion_goal_topic, PoseStamped, queue_size=1
        )
        self.cancel_pub = rospy.Publisher(
            self.motion_cancel_topic, Empty, queue_size=1
        )
        self.finished_pub = rospy.Publisher(
            "/finished", String, queue_size=10
        )
        self.arrow_sub = rospy.Subscriber(
            self.arrow_topic, String, self.arrow_callback, queue_size=20
        )
        self.motion_state_sub = rospy.Subscriber(
            self.motion_state_topic,
            MotionState,
            self.motion_state_callback,
            queue_size=20,
        )
        self.status_sub = rospy.Subscriber(
            self.status_topic, AUVData, self.status_callback, queue_size=20
        )

        self.task_started = rospy.Time.now()
        self.state = self.WAIT_FOR_CONTROL
        self.state_started = self.task_started
        self.task_finished = False
        self.control_initialized = False

        self.current_status = None
        self.last_status_received = None
        self.latest_motion_state = None
        self.last_motion_state_received = None
        self.last_motion_state_value = None
        self.motion_ready_once = False
        self.active_goal = None
        self.active_goal_reason = ""
        self.target_z = None
        self.target_depth = None

        self.model_frame_index = 0
        self.last_model_message_time = None
        self.last_valid_detection_time = None
        self.latest_detection = None
        self.detection_samples = []
        self.direction_samples = []
        self.arrow_locked = False
        self.direction_locked = False
        self.direction_locked_angle_deg = None
        self.centered_frame_count = 0
        self.aligned_frame_count = 0
        self.last_visual_goal_frame = 0

        self.initial_hover_stable_started = None
        self.final_hold_stable_started = None
        self.final_target_yaw = None
        self.cancel_requested_at = None
        self.cancel_next_state = None
        self.cancel_reason = ""

        rospy.on_shutdown(self.on_shutdown)
        self.log_startup_config()

    def validate_params(self):
        if self.rate_hz <= 0.0:
            raise ValueError("rate 必须大于0")
        if not all((
            self.arrow_topic,
            self.motion_goal_topic,
            self.motion_cancel_topic,
            self.motion_state_topic,
            self.status_topic,
        )):
            raise ValueError("任务话题参数不能为空")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence 必须在0到1之间")
        if min(
            self.stable_detection_count,
            self.center_stable_detection_count,
            self.heading_stable_detection_count,
        ) < 1:
            raise ValueError("连续确认帧数必须大于等于1")
        if min(self.image_width, self.image_height) <= 0.0:
            raise ValueError("图像宽度和高度必须大于0")
        if not 0.0 <= self.target_center_u_ratio <= 1.0:
            raise ValueError("target_center_u_ratio 必须在0到1之间")
        if not 0.0 <= self.target_center_v_ratio <= 1.0:
            raise ValueError("target_center_v_ratio 必须在0到1之间")
        if min(
            self.stable_center_tolerance_px,
            self.stable_angle_tolerance_deg,
            self.center_tolerance_u_px,
            self.center_tolerance_v_px,
            self.visual_forward_gain_m,
            self.visual_lateral_gain_m,
            self.visual_max_step_m,
            self.visual_min_step_m,
            self.yaw_tolerance_deg,
            self.initial_hover_seconds,
            self.search_forward_distance,
            self.front_thruster_forward_offset,
            self.final_hold_seconds,
            self.final_hold_timeout,
            self.max_wait_seconds,
            self.cancel_timeout,
            self.motion_state_timeout,
            self.motion_startup_timeout,
            self.status_timeout,
            self.goal_match_position_tolerance,
            self.goal_match_depth_tolerance,
            self.goal_match_yaw_tolerance_deg,
            self.max_depth_error,
            self.min_ground_clearance,
            self.detection_timeout,
            self.visual_loss_cancel_seconds,
            self.ground_clearance_goal_update_threshold,
            self.log_interval,
            self.warning_log_interval,
        ) < 0.0:
            raise ValueError("距离、时间、增益和容差不能小于0")
        if min(
            self.visual_max_step_m,
            self.search_forward_distance,
            self.front_thruster_forward_offset,
            self.final_hold_timeout,
            self.max_wait_seconds,
            self.cancel_timeout,
            self.motion_state_timeout,
            self.motion_startup_timeout,
            self.status_timeout,
            self.min_ground_clearance,
            self.detection_timeout,
            self.visual_loss_cancel_seconds,
            self.log_interval,
            self.warning_log_interval,
        ) <= 0.0:
            raise ValueError("关键距离、时间和超时参数必须大于0")
        if self.visual_min_step_m > self.visual_max_step_m:
            raise ValueError("visual_min_step_m 不能大于 visual_max_step_m")
        if self.visual_forward_sign not in (-1.0, 1.0):
            raise ValueError("visual_forward_sign 必须是1或-1")
        if self.visual_lateral_sign not in (-1.0, 1.0):
            raise ValueError("visual_lateral_sign 必须是1或-1")
        if self.yaw_correction_sign not in (-1.0, 1.0):
            raise ValueError("yaw_correction_sign 必须是1或-1")
        if self.final_hold_timeout < self.final_hold_seconds:
            raise ValueError("final_hold_timeout 不能小于 final_hold_seconds")
        if self.visual_loss_cancel_seconds > self.detection_timeout:
            raise ValueError(
                "visual_loss_cancel_seconds 不能大于 detection_timeout"
            )

    def log_startup_config(self):
        rospy.loginfo(
            (
                "%s：启动新运动状态机版本；本节点不发布/cmd/pose/ned，"
                "只以%.1fHz发布%s并订阅%s"
            ),
            NODE_NAME,
            self.rate_hz,
            self.motion_goal_topic,
            self.motion_state_topic,
        )
        rospy.loginfo(
            (
                "%s：流程：HOVER悬停%.1fs -> 前方%.2fm搜索 -> 识别%d帧后取消刹停 -> "
                "视觉小步居中 -> 定点确认方向%d帧 -> 航向和中心确认%d帧 -> "
                "base_link前移%.2fm -> 最终HOVER保持%.1fs"
            ),
            NODE_NAME,
            self.initial_hover_seconds,
            self.search_forward_distance,
            self.stable_detection_count,
            self.heading_stable_detection_count,
            self.center_stable_detection_count,
            self.front_thruster_forward_offset,
            self.final_hold_seconds,
        )
        rospy.loginfo(
            (
                "%s：识别：话题=%s，最低置信度=%.2f，稳定判定超时=%.2fs，"
                "运动阶段丢失刹停=%.2fs，"
                "图像=%.0fx%.0f，中心容差=(%.1f,%.1f)px"
            ),
            NODE_NAME,
            self.arrow_topic,
            self.min_confidence,
            self.detection_timeout,
            self.visual_loss_cancel_seconds,
            self.image_width,
            self.image_height,
            self.center_tolerance_u_px,
            self.center_tolerance_v_px,
        )
        rospy.loginfo(
            (
                "%s：视觉目标步长：增益=(前后%.3f,左右%.3f)m/归一化误差，"
                "步长范围=%.3f~%.3fm，方向符号=(前后%+.0f,左右%+.0f)"
            ),
            NODE_NAME,
            self.visual_forward_gain_m,
            self.visual_lateral_gain_m,
            self.visual_min_step_m,
            self.visual_max_step_m,
            self.visual_forward_sign,
            self.visual_lateral_sign,
        )
        rospy.loginfo(
            (
                "%s：运动反馈超时=%.2fs，启动等待=%.1fs，取消刹停超时=%.1fs；"
                "HOVER目标匹配容差=(水平%.3fm,深度%.3fm,航向%.1fdeg)"
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
                "%s：保护与日志：最低离地=%.2fm，离地目标更新阈值=%.3fm，"
                "普通/警告日志周期=(%.1f/%.1f)s"
            ),
            NODE_NAME,
            self.min_ground_clearance,
            self.ground_clearance_goal_update_threshold,
            self.log_interval,
            self.warning_log_interval,
        )

    @staticmethod
    def finite_number(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def mean_angle_deg(values):
        x_value = sum(math.cos(math.radians(value)) for value in values)
        y_value = sum(math.sin(math.radians(value)) for value in values)
        if abs(x_value) < 1e-9 and abs(y_value) < 1e-9:
            return normalize_angle_deg(values[-1])
        return normalize_angle_deg(math.degrees(math.atan2(y_value, x_value)))

    def status_callback(self, message):
        values = (
            message.pose.depth,
            message.pose.altitude,
            message.pose.yaw,
        )
        if not all(math.isfinite(value) for value in values):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：/status/auv深度、高度或航向包含无效值，本帧忽略",
                NODE_NAME,
            )
            return
        self.current_status = {
            "control_mode": int(message.control_mode),
            "depth": float(message.pose.depth),
            "altitude": float(message.pose.altitude),
            "yaw_deg": float(message.pose.yaw),
        }
        self.last_status_received = rospy.Time.now()
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：/status/auv：mode=%d，深度=%.3fm，高度=%.3fm，航向=%.2fdeg",
            NODE_NAME,
            self.current_status["control_mode"],
            self.current_status["depth"],
            self.current_status["altitude"],
            self.current_status["yaw_deg"],
        )

    def motion_state_callback(self, message):
        self.latest_motion_state = message
        self.last_motion_state_received = rospy.Time.now()
        state_name = self.MOTION_STATE_NAMES.get(
            message.state, "UNKNOWN({})".format(message.state)
        )
        if message.state != self.last_motion_state_value:
            rospy.loginfo(
                "%s：运动状态切换为%s，原因=%s",
                NODE_NAME,
                state_name,
                message.reason or "无",
            )
            self.last_motion_state_value = message.state
        if message.state not in (MotionState.SAFE,):
            self.motion_ready_once = True
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：运动反馈：state=%s，goal_active=%s，位置误差=%.3fm，"
                "航向误差=%+.2fdeg，水平速度=%.3fm/s，航向角速度=%+.2fdeg/s，"
                "输出=(TX=%d,TY=%d,MZ=%d)，原因=%s"
            ),
            NODE_NAME,
            state_name,
            str(bool(message.goal_active)),
            message.position_error,
            math.degrees(message.yaw_error),
            message.horizontal_speed,
            math.degrees(message.yaw_rate),
            message.tx,
            message.ty,
            message.mz,
            message.reason or "无",
        )

    def reject_arrow_frame(self, frame_index, reason):
        self.latest_detection = None
        if self.state in (self.FORWARD_SEARCH, self.WAIT_FOR_ARROW):
            previous = len(self.detection_samples)
            self.detection_samples = []
            self.arrow_locked = False
        else:
            previous = 0
        if self.state == self.CONFIRM_DIRECTION:
            self.direction_samples = []
            self.direction_locked = False
        if self.state == self.CENTER_CAMERA:
            self.reset_center_progress(reason)
        if self.state == self.ALIGN_HEADING:
            self.reset_alignment_progress(reason)
        if self.state in (self.FORWARD_SEARCH, self.WAIT_FOR_ARROW):
            rospy.loginfo(
                "%s：[箭头帧#%d] 无效：%s，首次锁定进度%d -> 0",
                NODE_NAME,
                frame_index,
                reason,
                previous,
            )
        else:
            rospy.loginfo(
                "%s：[箭头帧#%d] 无效：%s，阶段=%s",
                NODE_NAME,
                frame_index,
                reason,
                self.state,
            )

    def arrow_callback(self, message):
        self.model_frame_index += 1
        frame_index = self.model_frame_index
        now = rospy.Time.now()
        self.last_model_message_time = now

        if self.state == self.INITIAL_HOVER:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：[箭头帧#%d] 启动悬停中，本帧暂不计数",
                NODE_NAME,
                frame_index,
            )
            return
        if self.state in (self.MOVE_FRONT_THRUSTER, self.FINAL_HOLD):
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
                frame_index, "类别{}不是arrow".format(class_name or "空")
            )
            return
        if confidence is None or confidence < self.min_confidence:
            self.reject_arrow_frame(
                frame_index,
                "置信度{}低于{:.2f}".format(confidence, self.min_confidence),
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
            "direction": str(
                payload.get("discrete_direction", "")
            ).strip(),
        }
        self.latest_detection = detection
        self.last_valid_detection_time = now
        error_u, error_v, _, _ = self.detection_center_errors(detection)
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 有效：conf=%.3f，中心=(%.1f,%.1f)，"
                "误差=(u=%+.1f,v=%+.1f)px，角度=%.1fdeg，方向=%s，阶段=%s"
            ),
            NODE_NAME,
            frame_index,
            confidence,
            center_u,
            center_v,
            error_u,
            error_v,
            detection["angle_deg"],
            detection["direction"] or "未知",
            self.state,
        )

        if self.state in (self.FORWARD_SEARCH, self.WAIT_FOR_ARROW):
            self.add_detection_sample(detection)
        elif self.state == self.CENTER_CAMERA:
            self.update_center_progress(detection, error_u, error_v)
        elif self.state == self.CONFIRM_DIRECTION:
            self.add_direction_sample(detection, error_u, error_v)
        elif self.state == self.ALIGN_HEADING:
            self.update_alignment_progress(detection, error_u, error_v)

    def add_detection_sample(self, detection):
        if self.detection_samples:
            gap = (
                detection["received_time"]
                - self.detection_samples[-1]["received_time"]
            ).to_sec()
            if gap > self.detection_timeout:
                rospy.logwarn(
                    "%s：有效箭头帧间隔%.2fs超过%.2fs，计数清零",
                    NODE_NAME,
                    gap,
                    self.detection_timeout,
                )
                self.detection_samples = []
        self.detection_samples.append(detection)
        self.detection_samples = self.detection_samples[
            -self.stable_detection_count:
        ]
        mean_u = sum(
            item["center_u"] for item in self.detection_samples
        ) / len(self.detection_samples)
        mean_v = sum(
            item["center_v"] for item in self.detection_samples
        ) / len(self.detection_samples)
        center_jitter = max(
            math.hypot(item["center_u"] - mean_u, item["center_v"] - mean_v)
            for item in self.detection_samples
        )
        progress = len(self.detection_samples)
        rospy.loginfo(
            "%s：[箭头帧#%d] 首次锁定第%d/%d帧，中心抖动=%.1f/%.1fpx",
            NODE_NAME,
            detection["frame_index"],
            progress,
            self.stable_detection_count,
            center_jitter,
            self.stable_center_tolerance_px,
        )
        if progress < self.stable_detection_count:
            return
        if center_jitter > self.stable_center_tolerance_px:
            rospy.logwarn(
                "%s：首次锁定稳定性未通过，保留当前帧作为第1/%d帧",
                NODE_NAME,
                self.stable_detection_count,
            )
            self.detection_samples = [detection]
            return
        locked = dict(detection)
        locked["center_u"] = mean_u
        locked["center_v"] = mean_v
        locked["confidence"] = sum(
            item["confidence"] for item in self.detection_samples
        ) / len(self.detection_samples)
        self.latest_detection = locked
        self.arrow_locked = True
        rospy.loginfo(
            "%s：箭头连续%d帧稳定锁定，平均中心=(%.1f,%.1f)，平均置信度=%.3f",
            NODE_NAME,
            self.stable_detection_count,
            locked["center_u"],
            locked["center_v"],
            locked["confidence"],
        )

    def add_direction_sample(self, detection, error_u, error_v):
        centered = (
            abs(error_u) <= self.center_tolerance_u_px
            and abs(error_v) <= self.center_tolerance_v_px
        )
        if not centered:
            previous = len(self.direction_samples)
            self.direction_samples = []
            self.direction_locked = False
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] 方向确认时偏离中心，"
                    "误差=(%+.1f,%+.1f)px，方向计数%d -> 0"
                ),
                NODE_NAME,
                detection["frame_index"],
                error_u,
                error_v,
                previous,
            )
            return
        self.direction_samples.append(detection)
        self.direction_samples = self.direction_samples[
            -self.heading_stable_detection_count:
        ]
        mean_angle = self.mean_angle_deg([
            item["angle_deg"] for item in self.direction_samples
        ])
        angle_jitter = max(
            abs(normalize_angle_deg(item["angle_deg"] - mean_angle))
            for item in self.direction_samples
        )
        progress = len(self.direction_samples)
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 定点方向确认第%d/%d帧，"
                "平均角度=%.1fdeg，抖动=%.1f/%.1fdeg"
            ),
            NODE_NAME,
            detection["frame_index"],
            progress,
            self.heading_stable_detection_count,
            mean_angle,
            angle_jitter,
            self.stable_angle_tolerance_deg,
        )
        if progress < self.heading_stable_detection_count:
            return
        if angle_jitter > self.stable_angle_tolerance_deg:
            self.direction_samples = [detection]
            rospy.logwarn(
                "%s：方向角度抖动超限，保留当前帧重新累计",
                NODE_NAME,
            )
            return
        self.direction_locked = True
        self.direction_locked_angle_deg = mean_angle
        rospy.loginfo(
            "%s：箭头方向连续%d帧稳定，平均角度=%.1fdeg",
            NODE_NAME,
            self.heading_stable_detection_count,
            mean_angle,
        )

    def reset_center_progress(self, reason):
        previous = self.centered_frame_count
        self.centered_frame_count = 0
        if previous > 0:
            rospy.loginfo(
                "%s：图像居中计数%d -> 0，原因=%s",
                NODE_NAME,
                previous,
                reason,
            )

    def update_center_progress(self, detection, error_u, error_v):
        centered = (
            abs(error_u) <= self.center_tolerance_u_px
            and abs(error_v) <= self.center_tolerance_v_px
        )
        if not centered:
            self.reset_center_progress("箭头超出中心容差")
            return
        self.centered_frame_count = min(
            self.centered_frame_count + 1,
            self.center_stable_detection_count,
        )
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 图像居中第%d/%d帧有效，"
                "误差=(u=%+.1f,v=%+.1f)px"
            ),
            NODE_NAME,
            detection["frame_index"],
            self.centered_frame_count,
            self.center_stable_detection_count,
            error_u,
            error_v,
        )

    def reset_alignment_progress(self, reason):
        previous = self.aligned_frame_count
        self.aligned_frame_count = 0
        if previous > 0:
            rospy.loginfo(
                "%s：最终中心和航向计数%d -> 0，原因=%s",
                NODE_NAME,
                previous,
                reason,
            )

    def update_alignment_progress(self, detection, error_u, error_v):
        centered = (
            abs(error_u) <= self.center_tolerance_u_px
            and abs(error_v) <= self.center_tolerance_v_px
        )
        arrow_heading_error = self.yaw_correction_sign * normalize_angle_deg(
            self.camera_forward_angle_deg - detection["angle_deg"]
        )
        heading_ok = abs(arrow_heading_error) <= self.yaw_tolerance_deg
        if not (centered and heading_ok):
            self.reset_alignment_progress("中心或箭头方向超出容差")
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] 最终对准未通过：中心误差=(%+.1f,%+.1f)px，"
                    "箭头方向误差=%+.1f/%.1fdeg"
                ),
                NODE_NAME,
                detection["frame_index"],
                error_u,
                error_v,
                arrow_heading_error,
                self.yaw_tolerance_deg,
            )
            return
        self.aligned_frame_count = min(
            self.aligned_frame_count + 1,
            self.center_stable_detection_count,
        )
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 最终中心和航向第%d/%d帧有效，"
                "中心误差=(%+.1f,%+.1f)px，方向误差=%+.1fdeg"
            ),
            NODE_NAME,
            detection["frame_index"],
            self.aligned_frame_count,
            self.center_stable_detection_count,
            error_u,
            error_v,
            arrow_heading_error,
        )

    def get_current_pose(self, context):
        try:
            self.tf_listener.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(0.5)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：无法读取map -> base_link，%s暂停：%s",
                NODE_NAME,
                context,
                str(error),
            )
            return None
        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(value) for value in values):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：map -> base_link含无效值，%s暂停",
                NODE_NAME,
                context,
            )
            return None
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def get_recent_status(self, context):
        if self.current_status is None or self.last_status_received is None:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：等待状态话题%s，%s暂停",
                NODE_NAME,
                self.status_topic,
                context,
            )
            return None
        age = (rospy.Time.now() - self.last_status_received).to_sec()
        if age > self.status_timeout:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：状态话题已超时%.2fs（限制%.2fs），%s暂停",
                NODE_NAME,
                age,
                self.status_timeout,
                context,
            )
            return None
        return self.current_status

    def initialize_control(self):
        if self.control_initialized:
            return True
        status = self.get_recent_status("初始化任务绝对目标")
        current = self.get_current_pose("初始化任务绝对目标")
        if status is None or current is None:
            return False
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        self.target_z = current.pose.position.z
        self.target_depth = status["depth"]
        self.control_initialized = True
        self.set_active_goal(
            current.pose.position.x,
            current.pose.position.y,
            self.target_z,
            current_yaw,
            "记录启动位置并等待motion_supervisor完成初始HOVER接管",
        )
        self.set_state(
            self.INITIAL_HOVER,
            "TF和/status/auv已就绪，开始初始定点悬停",
        )
        return True

    def set_active_goal(self, x_value, y_value, z_value, yaw, reason):
        values = (x_value, y_value, z_value, yaw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("任务生成了非有限运动目标")
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = "map"
        goal.pose.position.x = x_value
        goal.pose.position.y = y_value
        goal.pose.position.z = z_value
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]
        self.active_goal = goal
        self.active_goal_reason = reason
        rospy.loginfo(
            (
                "%s：设置map绝对目标：x=%.3f，y=%.3f，z=%.3f，"
                "yaw=%.2fdeg，原因=%s"
            ),
            NODE_NAME,
            x_value,
            y_value,
            z_value,
            math.degrees(yaw),
            reason,
        )

    def set_body_offset_goal(self, current, forward, right, yaw, reason):
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        goal_x = (
            current.pose.position.x
            + math.cos(current_yaw) * forward
            - math.sin(current_yaw) * right
        )
        goal_y = (
            current.pose.position.y
            + math.sin(current_yaw) * forward
            + math.cos(current_yaw) * right
        )
        self.set_active_goal(
            goal_x,
            goal_y,
            self.target_z,
            yaw,
            reason,
        )
        return goal_x, goal_y

    def update_ground_clearance_goal(self):
        if self.active_goal is None:
            return
        status = self.get_recent_status("最低对地距离保护")
        if status is None:
            return
        altitude = status["altitude"]
        if altitude <= 0.0:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：高度%.3fm无效，无法校验最低对地距离%.2fm",
                NODE_NAME,
                altitude,
                self.min_ground_clearance,
            )
            return
        if altitude >= self.min_ground_clearance:
            return
        current = self.get_current_pose("最低对地距离保护")
        if current is None:
            return
        correction = self.min_ground_clearance - altitude
        safe_z = current.pose.position.z - correction
        safe_depth = status["depth"] - correction
        target_adjustment = self.target_z - safe_z
        if target_adjustment >= self.ground_clearance_goal_update_threshold:
            previous_target_z = self.target_z
            self.target_z = safe_z
            self.active_goal.pose.position.z = safe_z
            self.target_depth = min(self.target_depth, safe_depth)
            rospy.logwarn(
                (
                    "%s：离地保护触发：高度%.3fm<%.3fm，按当前位姿需上移%.3fm；"
                    "目标z从%.3f改为%.3f（改写%.3fm），目标深度=%.3f"
                ),
                NODE_NAME,
                altitude,
                self.min_ground_clearance,
                correction,
                previous_target_z,
                safe_z,
                target_adjustment,
                self.target_depth,
            )
        else:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                (
                    "%s：高度%.3fm<%.3fm，但目标z无需继续改写："
                    "候选改写=%+.3fm，更新阈值=%.3fm，当前目标z=%.3f"
                ),
                NODE_NAME,
                altitude,
                self.min_ground_clearance,
                target_adjustment,
                self.ground_clearance_goal_update_threshold,
                self.target_z,
            )

    def publish_active_goal(self):
        if self.active_goal is None:
            return False
        self.update_ground_clearance_goal()
        self.active_goal.header.stamp = rospy.Time.now()
        self.goal_pub.publish(self.active_goal)
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：持续发布运动目标：x=%.3f，y=%.3f，z=%.3f，"
                "yaw=%.2fdeg，阶段=%s"
            ),
            NODE_NAME,
            self.active_goal.pose.position.x,
            self.active_goal.pose.position.y,
            self.active_goal.pose.position.z,
            math.degrees(yaw_from_quaternion(
                self.active_goal.pose.orientation
            )),
            self.state,
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
        actual = self.latest_motion_state.goal
        if actual.header.frame_id != "map":
            return None
        dx = actual.pose.position.x - self.active_goal.pose.position.x
        dy = actual.pose.position.y - self.active_goal.pose.position.y
        dz = actual.pose.position.z - self.active_goal.pose.position.z
        desired_yaw = yaw_from_quaternion(self.active_goal.pose.orientation)
        actual_yaw = yaw_from_quaternion(actual.pose.orientation)
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

    def motion_hover_fresh(self):
        return (
            self.motion_state_is_fresh()
            and self.latest_motion_state.state == MotionState.HOVER
        )

    def motion_arrived(self):
        return self.motion_hover_fresh() and self.goal_matches_motion_state()

    def handle_motion_health(self):
        elapsed = (rospy.Time.now() - self.task_started).to_sec()
        if self.latest_motion_state is None:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：等待运动反馈%s，已等待%.1f/%.1fs",
                NODE_NAME,
                self.motion_state_topic,
                elapsed,
                self.motion_startup_timeout,
            )
            if elapsed >= self.motion_startup_timeout:
                self.finish_task(False, "启动后未收到/motion/state")
            return False
        if not self.motion_state_is_fresh():
            age = self.motion_state_age()
            rospy.logerr_throttle(
                self.warning_log_interval,
                "%s：运动反馈不新鲜，header年龄=%s，限制=%.2fs",
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
                "运动状态机返回未知状态{}".format(
                    self.latest_motion_state.state
                ),
            )
            return False
        if self.latest_motion_state.state == MotionState.SAFE:
            rospy.logerr_throttle(
                self.warning_log_interval,
                "%s：motion_supervisor进入SAFE，原因=%s",
                NODE_NAME,
                self.latest_motion_state.reason or "未知",
            )
            if self.motion_ready_once or elapsed >= self.motion_startup_timeout:
                self.finish_task(
                    False,
                    "motion_supervisor进入SAFE：{}".format(
                        self.latest_motion_state.reason or "未知原因"
                    ),
                )
            return False
        return True

    def set_state(self, state, reason):
        now = rospy.Time.now()
        previous_elapsed = (now - self.state_started).to_sec()
        task_elapsed = (now - self.task_started).to_sec()
        previous = self.state
        self.state = state
        self.state_started = now
        rospy.loginfo(
            "%s：任务阶段%s -> %s，上一阶段%.1fs，累计%.1fs，原因=%s",
            NODE_NAME,
            previous,
            state,
            previous_elapsed,
            task_elapsed,
            reason,
        )

    def begin_cancel(self, next_state, reason):
        self.cancel_pub.publish(Empty())
        self.active_goal = None
        self.cancel_requested_at = rospy.Time.now()
        self.cancel_next_state = next_state
        self.cancel_reason = reason
        rospy.logwarn(
            "%s：发布%s，要求主动刹停后HOVER；后续阶段=%s，原因=%s",
            NODE_NAME,
            self.motion_cancel_topic,
            next_state,
            reason,
        )
        self.set_state(self.CANCEL_WAIT, reason)

    def cancel_has_completed(self):
        if not self.motion_hover_fresh() or self.cancel_requested_at is None:
            return False
        return self.latest_motion_state.header.stamp >= self.cancel_requested_at

    def reset_first_lock(self):
        self.detection_samples = []
        self.arrow_locked = False

    def reset_direction_lock(self):
        self.direction_samples = []
        self.direction_locked = False
        self.direction_locked_angle_deg = None

    def valid_detection_age(self):
        if self.last_valid_detection_time is None:
            return None
        return max(
            0.0,
            (rospy.Time.now() - self.last_valid_detection_time).to_sec(),
        )

    def detection_available_within(self, timeout, context):
        valid_age = self.valid_detection_age()
        available = (
            self.latest_detection is not None
            and valid_age is not None
            and valid_age <= timeout
        )
        if available:
            return True
        model_age = None
        if self.last_model_message_time is not None:
            model_age = max(
                0.0,
                (rospy.Time.now() - self.last_model_message_time).to_sec(),
            )
        rospy.logwarn_throttle(
            self.warning_log_interval,
            (
                "%s：%s当前无可用箭头：最近有效帧年龄=%s，"
                "模型消息年龄=%s，保护阈值=%.2fs"
            ),
            NODE_NAME,
            context,
            "从未有效" if valid_age is None else "{:.2f}s".format(valid_age),
            "未收到" if model_age is None else "{:.2f}s".format(model_age),
            timeout,
        )
        return False

    def detection_center_errors(self, detection):
        desired_u = self.image_width * self.target_center_u_ratio
        desired_v = self.image_height * self.target_center_v_ratio
        error_u = detection["center_u"] - desired_u
        error_v = detection["center_v"] - desired_v
        normalized_u = error_u / max(0.5 * self.image_width, 1.0)
        normalized_v = error_v / max(0.5 * self.image_height, 1.0)
        return error_u, error_v, normalized_u, normalized_v

    def minimum_visual_step(self, value):
        value = clamp(
            value, -self.visual_max_step_m, self.visual_max_step_m
        )
        if value == 0.0 or abs(value) >= self.visual_min_step_m:
            return value
        return math.copysign(self.visual_min_step_m, value)

    def update_visual_goal(self, target_yaw, context):
        detection = self.latest_detection
        if detection is None:
            return False
        if detection["frame_index"] == self.last_visual_goal_frame:
            return True
        error_u, error_v, normalized_u, normalized_v = (
            self.detection_center_errors(detection)
        )
        forward_step = 0.0
        if abs(error_v) > self.center_tolerance_v_px:
            forward_step = self.minimum_visual_step(
                self.visual_forward_sign
                * -self.visual_forward_gain_m
                * normalized_v
            )
        right_step = 0.0
        if abs(error_u) > self.center_tolerance_u_px:
            right_step = self.minimum_visual_step(
                self.visual_lateral_sign
                * self.visual_lateral_gain_m
                * normalized_u
            )
        if forward_step == 0.0 and right_step == 0.0:
            self.last_visual_goal_frame = detection["frame_index"]
            rospy.loginfo(
                "%s：[箭头帧#%d] 已在中心容差，不生成新的平移目标",
                NODE_NAME,
                detection["frame_index"],
            )
            return True
        current = self.get_current_pose(context)
        if current is None:
            return False
        goal_x, goal_y = self.set_body_offset_goal(
            current,
            forward_step,
            right_step,
            target_yaw,
            "{}：按当前箭头像素误差生成一次小步绝对目标".format(context),
        )
        self.last_visual_goal_frame = detection["frame_index"]
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 视觉小步目标：误差=(u=%+.1f,v=%+.1f)px，"
                "本体偏置=(前%+.3f,右%+.3f)m，map目标=(%.3f,%.3f)"
            ),
            NODE_NAME,
            detection["frame_index"],
            error_u,
            error_v,
            forward_step,
            right_step,
            goal_x,
            goal_y,
        )
        return True

    def control_initial_hover(self):
        if self.motion_arrived():
            if self.initial_hover_stable_started is None:
                self.initial_hover_stable_started = rospy.Time.now()
                rospy.loginfo(
                    "%s：初始目标已进入新鲜HOVER，开始累计%.1fs悬停",
                    NODE_NAME,
                    self.initial_hover_seconds,
                )
            elapsed = (
                rospy.Time.now() - self.initial_hover_stable_started
            ).to_sec()
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：启动HOVER稳定保持%.1f/%.1fs",
                NODE_NAME,
                elapsed,
                self.initial_hover_seconds,
            )
            if elapsed >= self.initial_hover_seconds:
                current = self.get_current_pose("生成直行搜索目标")
                if current is None:
                    return
                search_yaw = yaw_from_quaternion(current.pose.orientation)
                self.reset_first_lock()
                self.set_body_offset_goal(
                    current,
                    self.search_forward_distance,
                    0.0,
                    search_yaw,
                    "在阶段开始时一次性计算前方搜索终点",
                )
                self.set_state(
                    self.FORWARD_SEARCH,
                    "启动悬停完成，交由motion_supervisor向前搜索",
                )
        else:
            self.initial_hover_stable_started = None
            self.log_arrival_gate("等待初始HOVER接管")

    def control_forward_search(self):
        if self.arrow_locked:
            self.begin_cancel(
                self.CENTER_CAMERA,
                "箭头连续稳定识别，取消远处搜索目标并主动刹停",
            )
            return
        if self.motion_arrived():
            self.finish_task(
                False,
                "已到达前方{:.2f}m搜索终点仍未稳定识别箭头".format(
                    self.search_forward_distance
                ),
            )
            return
        model_age = None
        if self.last_model_message_time is not None:
            model_age = (
                rospy.Time.now() - self.last_model_message_time
            ).to_sec()
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：向前搜索：motion=%s，位置误差=%.3fm，"
                "识别进度=%d/%d帧，模型消息年龄=%s"
            ),
            NODE_NAME,
            self.current_motion_state_name(),
            self.latest_motion_state.position_error,
            len(self.detection_samples),
            self.stable_detection_count,
            "未收到" if model_age is None else "{:.2f}s".format(model_age),
        )
        if model_age is None or model_age > self.detection_timeout:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：搜索时箭头模型话题未更新，请检查%s",
                NODE_NAME,
                self.arrow_topic,
            )

    def control_cancel_wait(self):
        elapsed = (rospy.Time.now() - self.state_started).to_sec()
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：等待取消刹停完成：motion=%s，速度=%.3fm/s，"
                "输出=(%d,%d,%d)，%.1f/%.1fs"
            ),
            NODE_NAME,
            self.current_motion_state_name(),
            self.latest_motion_state.horizontal_speed,
            self.latest_motion_state.tx,
            self.latest_motion_state.ty,
            self.latest_motion_state.mz,
            elapsed,
            self.cancel_timeout,
        )
        if elapsed >= self.cancel_timeout:
            self.finish_task(False, "取消后未在规定时间进入HOVER")
            return
        if not self.cancel_has_completed():
            return
        current = self.get_current_pose("取消完成后记录停稳位置")
        if current is None:
            return
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        next_state = self.cancel_next_state
        self.set_active_goal(
            current.pose.position.x,
            current.pose.position.y,
            self.target_z,
            current_yaw,
            "取消刹停完成，锁定实际停稳位置",
        )
        self.last_visual_goal_frame = 0
        if next_state == self.WAIT_FOR_ARROW:
            self.reset_first_lock()
        elif next_state == self.CENTER_CAMERA:
            self.reset_center_progress("进入视觉居中")
        elif next_state == self.CONFIRM_DIRECTION:
            self.reset_direction_lock()
        self.set_state(
            next_state,
            "motion_supervisor已完成刹停并由mode4接管",
        )

    def control_wait_for_arrow(self):
        if self.arrow_locked:
            self.reset_center_progress("重新锁定箭头")
            self.last_visual_goal_frame = 0
            self.set_state(
                self.CENTER_CAMERA,
                "定点重新连续识别成功，恢复视觉居中",
            )
            return
        model_age = None
        if self.last_model_message_time is not None:
            model_age = (
                rospy.Time.now() - self.last_model_message_time
            ).to_sec()
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：定点重识别：进度=%d/%d帧，模型年龄=%s，motion=%s",
            NODE_NAME,
            len(self.detection_samples),
            self.stable_detection_count,
            "未收到" if model_age is None else "{:.2f}s".format(model_age),
            self.current_motion_state_name(),
        )

    def control_center_camera(self):
        if not self.detection_available_within(
            self.visual_loss_cancel_seconds, "视觉居中阶段"
        ):
            valid_age = self.valid_detection_age()
            if valid_age is None or valid_age > self.visual_loss_cancel_seconds:
                self.begin_cancel(
                    self.WAIT_FOR_ARROW,
                    (
                        "视觉居中时箭头丢失{}，超过{:.2f}s阈值，"
                        "刹停后定点重识别"
                    ).format(
                        "未知" if valid_age is None else "{:.2f}s".format(
                            valid_age
                        ),
                        self.visual_loss_cancel_seconds,
                    ),
                )
            return
        if self.centered_frame_count >= self.center_stable_detection_count:
            self.begin_cancel(
                self.CONFIRM_DIRECTION,
                "箭头连续居中，先主动刹停并HOVER后再确认方向",
            )
            return
        target_yaw = yaw_from_quaternion(self.active_goal.pose.orientation)
        self.update_visual_goal(target_yaw, "保持发现航向并视觉居中")
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：视觉居中进度=%d/%d帧，motion=%s，位置误差=%.3fm",
            NODE_NAME,
            self.centered_frame_count,
            self.center_stable_detection_count,
            self.current_motion_state_name(),
            self.latest_motion_state.position_error,
        )

    def control_confirm_direction(self):
        if not self.detection_available_within(
            self.detection_timeout, "定点方向确认阶段"
        ):
            valid_age = self.valid_detection_age()
            if valid_age is None or valid_age > self.detection_timeout:
                self.begin_cancel(
                    self.WAIT_FOR_ARROW,
                    "定点确认方向时箭头丢失超过{:.2f}s".format(
                        self.detection_timeout
                    ),
                )
            return
        error_u, error_v, _, _ = self.detection_center_errors(
            self.latest_detection
        )
        if (
            abs(error_u) > self.center_tolerance_u_px
            or abs(error_v) > self.center_tolerance_v_px
        ):
            self.reset_center_progress("方向确认时箭头偏离中心")
            self.last_visual_goal_frame = 0
            self.set_state(
                self.CENTER_CAMERA,
                "方向确认时偏离中心，重新生成视觉小步目标",
            )
            return
        if not self.direction_locked:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：保持HOVER确认箭头方向%d/%d帧",
                NODE_NAME,
                len(self.direction_samples),
                self.heading_stable_detection_count,
            )
            return
        if not self.motion_arrived():
            self.log_arrival_gate("方向已稳定，等待当前定点目标HOVER")
            return
        current = self.get_current_pose("根据稳定箭头方向生成最终航向")
        if current is None:
            return
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        correction_deg = self.yaw_correction_sign * normalize_angle_deg(
            self.camera_forward_angle_deg - self.direction_locked_angle_deg
        )
        self.final_target_yaw = normalize_angle_rad(
            current_yaw + math.radians(correction_deg)
        )
        self.aligned_frame_count = 0
        self.last_visual_goal_frame = 0
        self.set_active_goal(
            current.pose.position.x,
            current.pose.position.y,
            self.target_z,
            self.final_target_yaw,
            "箭头方向稳定，发布完整最终航向并由状态机主动刹转",
        )
        rospy.loginfo(
            (
                "%s：航向目标换算：当前=%.2fdeg，箭头角度=%.2fdeg，"
                "相对修正=%+.2fdeg，最终map航向=%.2fdeg"
            ),
            NODE_NAME,
            math.degrees(current_yaw),
            self.direction_locked_angle_deg,
            correction_deg,
            math.degrees(self.final_target_yaw),
        )
        self.set_state(
            self.ALIGN_HEADING,
            "开始由motion_supervisor完成最终转向并用视觉小步保持中心",
        )

    def control_align_heading(self):
        if not self.detection_available_within(
            self.visual_loss_cancel_seconds, "航向对准阶段"
        ):
            valid_age = self.valid_detection_age()
            if valid_age is None or valid_age > self.visual_loss_cancel_seconds:
                self.begin_cancel(
                    self.WAIT_FOR_ARROW,
                    (
                        "航向对准时箭头丢失{}，超过{:.2f}s阈值，"
                        "取消并定点重识别"
                    ).format(
                        "未知" if valid_age is None else "{:.2f}s".format(
                            valid_age
                        ),
                        self.visual_loss_cancel_seconds,
                    ),
                )
            return
        self.update_visual_goal(
            self.final_target_yaw,
            "最终航向对准期间保持箭头居中",
        )
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：最终对准：图像确认=%d/%d帧，motion=%s，"
                "位置/航向误差=(%.3fm,%+.2fdeg)，目标匹配=%s"
            ),
            NODE_NAME,
            self.aligned_frame_count,
            self.center_stable_detection_count,
            self.current_motion_state_name(),
            self.latest_motion_state.position_error,
            math.degrees(self.latest_motion_state.yaw_error),
            "通过" if self.goal_matches_motion_state() else "未通过",
        )
        if self.aligned_frame_count < self.center_stable_detection_count:
            return
        if not self.motion_arrived():
            self.log_arrival_gate(
                "图像中心和箭头方向已稳定，等待最新目标HOVER"
            )
            return
        self.start_front_thruster_offset()

    def start_front_thruster_offset(self):
        current = self.get_current_pose("生成前垂推对齐目标")
        if current is None:
            return
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        goal_x = (
            current.pose.position.x
            + math.cos(current_yaw) * self.front_thruster_forward_offset
        )
        goal_y = (
            current.pose.position.y
            + math.sin(current_yaw) * self.front_thruster_forward_offset
        )
        self.set_active_goal(
            goal_x,
            goal_y,
            self.target_z,
            self.final_target_yaw,
            "航向和相机中心已对准，base_link沿当前前方前移固定距离",
        )
        rospy.loginfo(
            (
                "%s：前垂推补偿只计算一次：起点=(%.3f,%.3f)，"
                "当前航向=%.2fdeg，base_link前移=%.3fm，终点=(%.3f,%.3f)"
            ),
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            math.degrees(current_yaw),
            self.front_thruster_forward_offset,
            goal_x,
            goal_y,
        )
        self.set_state(
            self.MOVE_FRONT_THRUSTER,
            "目标位置将与前垂推基本对齐，等待平移、刹停和mode4接管",
        )

    def control_move_front_thruster(self):
        if self.motion_arrived():
            self.final_hold_stable_started = None
            self.set_state(
                self.FINAL_HOLD,
                "前移目标已进入HOVER，开始最终稳定保持",
            )
            return
        self.log_arrival_gate("等待base_link前移目标完成")

    def control_final_hold(self):
        now = rospy.Time.now()
        status = self.get_recent_status("最终深度确认")
        hover_ok = self.motion_arrived()
        depth_error = None
        depth_ok = False
        if status is not None and self.target_depth is not None:
            depth_error = status["depth"] - self.target_depth
            depth_ok = abs(depth_error) <= self.max_depth_error
        if hover_ok and depth_ok:
            if self.final_hold_stable_started is None:
                self.final_hold_stable_started = now
                rospy.loginfo(
                    "%s：最终HOVER和深度均通过，开始累计%.1fs",
                    NODE_NAME,
                    self.final_hold_seconds,
                )
            stable_elapsed = (
                now - self.final_hold_stable_started
            ).to_sec()
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：最终保持%.1f/%.1fs；HOVER[通过]；"
                    "深度误差=%.3f<=%.3f[通过]"
                ),
                NODE_NAME,
                stable_elapsed,
                self.final_hold_seconds,
                abs(depth_error),
                self.max_depth_error,
            )
            if stable_elapsed >= self.final_hold_seconds:
                self.finish_task(
                    True,
                    "机器人航向与箭头一致，base_link前移{:.2f}m使目标与前垂推基本对齐".format(
                        self.front_thruster_forward_offset
                    ),
                )
                return
        else:
            if self.final_hold_stable_started is not None:
                rospy.loginfo(
                    "%s：最终稳定条件被打断，保持计时清零",
                    NODE_NAME,
                )
            self.final_hold_stable_started = None
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：最终保持门槛：motion=%s，HOVER目标匹配[%s]；"
                    "深度误差=%s<=%.3f[%s]"
                ),
                NODE_NAME,
                self.current_motion_state_name(),
                "通过" if hover_ok else "未通过",
                "未知" if depth_error is None else "{:.3f}".format(
                    abs(depth_error)
                ),
                self.max_depth_error,
                "通过" if depth_ok else "未通过",
            )
        if (now - self.state_started).to_sec() >= self.final_hold_timeout:
            self.finish_task(
                False,
                "最终定点{:.1f}s内未连续稳定保持{:.1f}s".format(
                    self.final_hold_timeout,
                    self.final_hold_seconds,
                ),
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
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：%s：反馈新鲜[%s]，state=%s/HOVER[%s]，"
                "目标一致[%s]，目标差值=(%s)，控制误差=(位置%.3fm,航向%+.2fdeg)，"
                "速度=%.3fm/s，yaw_rate=%+.2fdeg/s，输出=(%d,%d,%d)"
            ),
            NODE_NAME,
            context,
            "通过" if fresh else "未通过",
            self.current_motion_state_name(),
            "通过" if hover else "未通过",
            "通过" if goal_match else "未通过",
            goal_error_text,
            message.position_error,
            math.degrees(message.yaw_error),
            message.horizontal_speed,
            math.degrees(message.yaw_rate),
            message.tx,
            message.ty,
            message.mz,
        )

    def finish_task(self, success, detail):
        if self.task_finished:
            return
        self.task_finished = True
        self.active_goal = None
        self.cancel_pub.publish(Empty())
        state = "finished" if success else "failed"
        message = "{} {}: {}".format(NODE_NAME, state, detail)
        self.finished_pub.publish(String(data=message))
        if success:
            rospy.loginfo(
                "%s：任务成功：%s；已发布cancel保持停稳位置",
                NODE_NAME,
                detail,
            )
        else:
            rospy.logerr(
                "%s：任务失败：%s；已发布cancel要求主动刹停",
                NODE_NAME,
                detail,
            )
        rospy.signal_shutdown(message)

    def on_shutdown(self):
        if hasattr(self, "cancel_pub"):
            self.cancel_pub.publish(Empty())

    def run(self):
        while not rospy.is_shutdown():
            if self.task_finished:
                self.rate.sleep()
                continue
            elapsed = (rospy.Time.now() - self.task_started).to_sec()
            if elapsed >= self.max_wait_seconds and self.state != self.FINAL_HOLD:
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
            elif self.state == self.FORWARD_SEARCH:
                self.control_forward_search()
            elif self.state == self.CANCEL_WAIT:
                self.control_cancel_wait()
            elif self.state == self.WAIT_FOR_ARROW:
                self.control_wait_for_arrow()
            elif self.state == self.CENTER_CAMERA:
                self.control_center_camera()
            elif self.state == self.CONFIRM_DIRECTION:
                self.control_confirm_direction()
            elif self.state == self.ALIGN_HEADING:
                self.control_align_heading()
            elif self.state == self.MOVE_FRONT_THRUSTER:
                self.control_move_front_thruster()
            elif self.state == self.FINAL_HOLD:
                self.control_final_hold()

            if not self.task_finished:
                self.publish_active_goal()
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME)
    try:
        Task3AcquireAreaMotionTest().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("%s：未处理异常：%s", NODE_NAME, str(error))
        raise
