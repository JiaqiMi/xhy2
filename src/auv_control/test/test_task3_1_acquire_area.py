#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
任务3子任务1：识别箭头并通过 motion_supervisor 完成搜索、对准和最终定位。

本节点只生成 map 绝对目标，不直接发布 /cmd/pose/ned，也不计算 TX、TY、MZ。
motion_supervisor 负责平移、主动刹停、最终转向和 mode=4 定点接管。
"""

import json
import math
import statistics

import rospy
import tf
from auv_control.msg import AUVData, MotionState
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task3_1_acquire_area"


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


class Task3AcquireAreaTest(object):
    WAIT_FOR_CONTROL = "等待运动状态机和反馈"
    INITIAL_HOVER = "启动定点悬停"
    SEARCH_PATTERN = "固定路径搜索箭头"
    CANCEL_WAIT = "主动刹停并等待定点接管"
    WAIT_FOR_ARROW = "定点重新识别箭头"
    COARSE_LATERAL_ALIGN = "箭头图像粗居中"
    CONFIRM_DIRECTION = "定点确认箭头方向"
    ALIGN_HEADING = "细对准：持续看箭头并慢速对齐航向"
    FINE_FORWARD_ALIGN = "细对准：航向对齐后慢速前后居中"
    MOVE_BASE_OVER_ARROW = "持续视觉跟踪并将base_link移到箭头上方"
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
            "~arrow_topic", "/vision/arrow/direction"
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
        self.stable_detection_window_size = int(rospy.get_param(
            "~stable_detection_window_size", 10
        ))
        self.stable_center_tolerance_px = float(rospy.get_param(
            "~stable_center_tolerance_px", 40.0
        ))
        self.stable_area_tolerance_ratio = float(rospy.get_param(
            "~stable_area_tolerance_ratio", 0.35
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
        self.heading_aligned_detection_count = int(rospy.get_param(
            "~heading_aligned_detection_count", 3
        ))

        self.image_width = float(rospy.get_param("~image_width", 640.0))
        self.image_height = float(rospy.get_param("~image_height", 480.0))
        self.full_arrow_edge_margin_px = float(rospy.get_param(
            "~full_arrow_edge_margin_px", 15.0
        ))
        self.full_arrow_min_bbox_width_px = float(rospy.get_param(
            "~full_arrow_min_bbox_width_px", 30.0
        ))
        self.full_arrow_min_bbox_height_px = float(rospy.get_param(
            "~full_arrow_min_bbox_height_px", 30.0
        ))
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
        self.visual_lateral_gain_m = float(rospy.get_param(
            "~visual_lateral_gain_m", 0.20
        ))
        self.visual_forward_gain_m = float(rospy.get_param(
            "~visual_forward_gain_m", 0.20
        ))
        self.visual_max_step_m = float(rospy.get_param(
            "~visual_max_step_m", 0.08
        ))
        self.visual_min_step_m = float(rospy.get_param(
            "~visual_min_step_m", 0.01
        ))
        self.visual_goal_min_interval = float(rospy.get_param(
            "~visual_goal_min_interval", 1.0
        ))
        self.visual_forward_sign = float(rospy.get_param(
            "~visual_forward_sign", 1.0
        ))
        self.visual_lateral_sign = float(rospy.get_param(
            "~visual_lateral_sign", 1.0
        ))
        self.fine_forward_gain_m = float(rospy.get_param(
            "~fine_forward_gain_m", 0.10
        ))
        self.fine_lateral_gain_m = float(rospy.get_param(
            "~fine_lateral_gain_m", 0.10
        ))
        self.fine_visual_max_step_m = float(rospy.get_param(
            "~fine_visual_max_step_m", 0.03
        ))
        self.fine_visual_min_step_m = float(rospy.get_param(
            "~fine_visual_min_step_m", 0.005
        ))
        self.fine_yaw_max_step_deg = float(rospy.get_param(
            "~fine_yaw_max_step_deg", 3.0
        ))
        self.fine_goal_min_interval = float(rospy.get_param(
            "~fine_goal_min_interval", 0.5
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
        self.search_initial_forward_distance = float(rospy.get_param(
            "~search_initial_forward_distance", 0.50
        ))
        self.search_lateral_distance = float(rospy.get_param(
            "~search_lateral_distance", 0.20
        ))
        self.search_second_forward_distance = float(rospy.get_param(
            "~search_second_forward_distance", 0.30
        ))
        self.base_link_forward_offset = float(rospy.get_param(
            "~base_link_forward_offset", 0.35
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
        self.arrival_position_tolerance = float(rospy.get_param(
            "~arrival_position_tolerance", 0.05
        ))
        self.arrival_yaw_tolerance_deg = float(rospy.get_param(
            "~arrival_yaw_tolerance_deg", 5.0
        ))
        self.arrival_max_horizontal_speed = float(rospy.get_param(
            "~arrival_max_horizontal_speed", 0.02
        ))
        self.arrival_max_yaw_rate_deg_s = float(rospy.get_param(
            "~arrival_max_yaw_rate_deg_s", 0.5
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
        self.initial_hold_x = None
        self.initial_hold_y = None
        self.initial_hold_yaw = None
        self.search_waypoints = []
        self.search_waypoint_index = -1
        self.first_position_detected = False

        self.model_frame_index = 0
        self.last_model_message_time = None
        self.last_valid_detection_time = None
        self.last_full_direction_detection_time = None
        self.latest_detection = None
        self.detection_samples = []
        self.direction_samples = []
        self.arrow_locked = False
        self.direction_locked = False
        self.direction_locked_angle_deg = None
        self.centered_frame_count = 0
        self.heading_aligned_frame_count = 0
        self.aligned_frame_count = 0
        self.base_tracking_frame_count = 0
        self.last_visual_goal_frame = 0
        self.last_visual_goal_time = None

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
            self.stable_detection_window_size,
            self.center_stable_detection_count,
            self.heading_stable_detection_count,
            self.heading_aligned_detection_count,
        ) < 1:
            raise ValueError("识别窗口和确认帧数必须大于等于1")
        if self.stable_detection_count > self.stable_detection_window_size:
            raise ValueError(
                "stable_detection_count 不能大于 stable_detection_window_size"
            )
        if not 0.0 <= self.stable_area_tolerance_ratio <= 1.0:
            raise ValueError("stable_area_tolerance_ratio 必须在0到1之间")
        if min(self.image_width, self.image_height) <= 0.0:
            raise ValueError("图像宽度和高度必须大于0")
        if not 0.0 <= self.target_center_u_ratio <= 1.0:
            raise ValueError("target_center_u_ratio 必须在0到1之间")
        if not 0.0 <= self.target_center_v_ratio <= 1.0:
            raise ValueError("target_center_v_ratio 必须在0到1之间")
        if min(
            self.stable_center_tolerance_px,
            self.stable_angle_tolerance_deg,
            self.full_arrow_edge_margin_px,
            self.full_arrow_min_bbox_width_px,
            self.full_arrow_min_bbox_height_px,
            self.center_tolerance_u_px,
            self.center_tolerance_v_px,
            self.visual_forward_gain_m,
            self.visual_lateral_gain_m,
            self.visual_max_step_m,
            self.visual_min_step_m,
            self.visual_goal_min_interval,
            self.fine_forward_gain_m,
            self.fine_lateral_gain_m,
            self.fine_visual_max_step_m,
            self.fine_visual_min_step_m,
            self.fine_yaw_max_step_deg,
            self.fine_goal_min_interval,
            self.yaw_tolerance_deg,
            self.initial_hover_seconds,
            self.search_initial_forward_distance,
            self.search_lateral_distance,
            self.search_second_forward_distance,
            self.base_link_forward_offset,
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
            self.arrival_position_tolerance,
            self.arrival_yaw_tolerance_deg,
            self.arrival_max_horizontal_speed,
            self.arrival_max_yaw_rate_deg_s,
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
            self.fine_visual_max_step_m,
            self.fine_yaw_max_step_deg,
            self.search_initial_forward_distance,
            self.search_lateral_distance,
            self.search_second_forward_distance,
            self.base_link_forward_offset,
            self.final_hold_timeout,
            self.max_wait_seconds,
            self.cancel_timeout,
            self.motion_state_timeout,
            self.motion_startup_timeout,
            self.status_timeout,
            self.min_ground_clearance,
            self.detection_timeout,
            self.visual_loss_cancel_seconds,
            self.visual_goal_min_interval,
            self.fine_goal_min_interval,
            self.arrival_position_tolerance,
            self.arrival_yaw_tolerance_deg,
            self.arrival_max_horizontal_speed,
            self.arrival_max_yaw_rate_deg_s,
            self.log_interval,
            self.warning_log_interval,
        ) <= 0.0:
            raise ValueError("关键距离、时间和超时参数必须大于0")
        if self.visual_min_step_m > self.visual_max_step_m:
            raise ValueError("visual_min_step_m 不能大于 visual_max_step_m")
        if self.fine_visual_min_step_m > self.fine_visual_max_step_m:
            raise ValueError(
                "fine_visual_min_step_m 不能大于 fine_visual_max_step_m"
            )
        if 2.0 * self.full_arrow_edge_margin_px >= min(
            self.image_width, self.image_height
        ):
            raise ValueError("full_arrow_edge_margin_px 不能占满整幅图像")
        if (
            self.full_arrow_min_bbox_width_px
            + 2.0 * self.full_arrow_edge_margin_px
            > self.image_width
        ):
            raise ValueError(
                "bbox最小宽度与两侧边缘留白之和不能大于图像宽度"
            )
        if (
            self.full_arrow_min_bbox_height_px
            + 2.0 * self.full_arrow_edge_margin_px
            > self.image_height
        ):
            raise ValueError(
                "bbox最小高度与上下边缘留白之和不能大于图像高度"
            )
        if max(self.yaw_tolerance_deg, self.fine_yaw_max_step_deg) > 180.0:
            raise ValueError("航向容差和单次航向步长不能大于180度")
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
                "%s：启动子任务1；本节点不发布/cmd/pose/ned，"
                "只以%.1fHz发布%s并订阅%s"
            ),
            NODE_NAME,
            self.rate_hz,
            self.motion_goal_topic,
            self.motion_state_topic,
        )
        rospy.loginfo(
            (
                "%s：流程：固定点HOVER悬停%.1fs -> 前%.2fm -> 左右各%.2fm -> "
                "再前%.2fm -> 左右各%.2fm搜索 -> "
                "最近%d帧内位置一致%d帧 -> 图像中心粗对准 -> "
                "完整箭头方向%d帧 -> 慢速航向对齐%d帧 -> "
                "慢速前后居中%d帧 -> base_link前移%.2fm并持续视觉修正 -> "
                "最终HOVER保持%.1fs"
            ),
            NODE_NAME,
            self.initial_hover_seconds,
            self.search_initial_forward_distance,
            self.search_lateral_distance,
            self.search_second_forward_distance,
            self.search_lateral_distance,
            self.stable_detection_window_size,
            self.stable_detection_count,
            self.heading_stable_detection_count,
            self.heading_aligned_detection_count,
            self.center_stable_detection_count,
            self.base_link_forward_offset,
            self.final_hold_seconds,
        )
        rospy.loginfo(
            (
                "%s：识别：话题=%s，最低置信度=%.2f，稳定判定超时=%.2fs，"
                "运动阶段丢失刹停=%.2fs，"
                "位置候选组=最近%d帧命中%d帧，中心抖动<=%.1fpx，面积变化<=%.3f；"
                "图像=%.0fx%.0f，目标中心=(%.1f,%.1f)px，"
                "中心容差=(%.1f,%.1f)px；"
                "完整箭头门槛=距边缘>=%.1fpx且bbox>=%.1fx%.1fpx"
            ),
            NODE_NAME,
            self.arrow_topic,
            self.min_confidence,
            self.detection_timeout,
            self.visual_loss_cancel_seconds,
            self.stable_detection_window_size,
            self.stable_detection_count,
            self.stable_center_tolerance_px,
            self.stable_area_tolerance_ratio,
            self.image_width,
            self.image_height,
            self.image_width * self.target_center_u_ratio,
            self.image_height * self.target_center_v_ratio,
            self.center_tolerance_u_px,
            self.center_tolerance_v_px,
            self.full_arrow_edge_margin_px,
            self.full_arrow_min_bbox_width_px,
            self.full_arrow_min_bbox_height_px,
        )
        rospy.loginfo(
            (
                "%s：粗对准居中参数：增益=(前后%.3f,左右%.3f)m/归一化误差，"
                "步长范围=%.3f~%.3fm，最短目标间隔=%.2fs，"
                "方向符号=(前后%+.0f,左右%+.0f)"
            ),
            NODE_NAME,
            self.visual_forward_gain_m,
            self.visual_lateral_gain_m,
            self.visual_min_step_m,
            self.visual_max_step_m,
            self.visual_goal_min_interval,
            self.visual_forward_sign,
            self.visual_lateral_sign,
        )
        rospy.loginfo(
            (
                "%s：细对准慢速参数：增益=(前后%.3f,左右%.3f)，"
                "平移步长=%.3f~%.3fm，单次航向<=%.1fdeg，目标间隔>=%.2fs，"
                "方向符号=(前后%+.0f,左右%+.0f,yaw%+.0f)"
            ),
            NODE_NAME,
            self.fine_forward_gain_m,
            self.fine_lateral_gain_m,
            self.fine_visual_min_step_m,
            self.fine_visual_max_step_m,
            self.fine_yaw_max_step_deg,
            self.fine_goal_min_interval,
            self.visual_forward_sign,
            self.visual_lateral_sign,
            self.yaw_correction_sign,
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
                "%s：任务侧实际到达门槛：base_link水平误差<=%.3fm，"
                "航向误差<=%.1fdeg，水平速度<=%.3fm/s，"
                "航向角速度<=%.2fdeg/s，深度误差<=%.3fm"
            ),
            NODE_NAME,
            self.arrival_position_tolerance,
            self.arrival_yaw_tolerance_deg,
            self.arrival_max_horizontal_speed,
            self.arrival_max_yaw_rate_deg_s,
            self.max_depth_error,
        )
        rospy.loginfo(
            (
                "%s：保护与日志：按map地面z=0计算，最低离地=%.2fm，"
                "离地目标更新阈值=%.3fm，"
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
            message.pose.yaw,
        )
        if not all(math.isfinite(value) for value in values):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：/status/auv深度或航向包含无效值，本帧忽略",
                NODE_NAME,
            )
            return
        self.current_status = {
            "control_mode": int(message.control_mode),
            "depth": float(message.pose.depth),
            "yaw_deg": float(message.pose.yaw),
        }
        self.last_status_received = rospy.Time.now()
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：/status/auv：mode=%d，深度=%.3fm，航向=%.2fdeg",
            NODE_NAME,
            self.current_status["control_mode"],
            self.current_status["depth"],
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
                "%s：运动反馈：state=%s，goal_active=%s，"
                "控制位置误差=%.3fm，base_link实际误差=%.3fm，"
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

    def reject_arrow_frame(self, frame_index, reason):
        self.latest_detection = None
        if self.state in (self.SEARCH_PATTERN, self.WAIT_FOR_ARROW):
            self.add_detection_sample(None, frame_index, reason)
        if self.state in (
            self.COARSE_LATERAL_ALIGN,
            self.CONFIRM_DIRECTION,
            self.ALIGN_HEADING,
            self.FINE_FORWARD_ALIGN,
            self.MOVE_BASE_OVER_ARROW,
        ):
            self.reset_direction_lock()
        if self.state == self.COARSE_LATERAL_ALIGN:
            self.reset_center_progress(reason)
        if self.state == self.ALIGN_HEADING:
            self.reset_heading_alignment_progress(reason)
        if self.state == self.FINE_FORWARD_ALIGN:
            self.reset_alignment_progress(reason)
        if self.state == self.MOVE_BASE_OVER_ARROW:
            self.reset_base_tracking_progress(reason)
        if self.state in (self.SEARCH_PATTERN, self.WAIT_FOR_ARROW):
            return
        else:
            rospy.loginfo(
                "%s：[箭头帧#%d] 无效：%s，阶段=%s",
                NODE_NAME,
                frame_index,
                reason,
                self.state,
            )

    def full_arrow_visible(self, detection):
        bbox = detection.get("bbox")
        if bbox is None:
            return False, "缺少有效bbox"
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if width < self.full_arrow_min_bbox_width_px:
            return False, "bbox宽度{:.1f}px不足".format(width)
        if height < self.full_arrow_min_bbox_height_px:
            return False, "bbox高度{:.1f}px不足".format(height)
        margin = self.full_arrow_edge_margin_px
        edge_distances = (x1, y1, self.image_width - x2, self.image_height - y2)
        if min(edge_distances) < margin:
            return False, "bbox距最近图像边缘{:.1f}px不足".format(
                min(edge_distances)
            )
        return True, "bbox完整且距边缘最小{:.1f}px".format(
            min(edge_distances)
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
        if self.state == self.FINAL_HOLD:
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
        bbox = payload.get("bbox")
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
        if center_u is None or center_v is None:
            self.reject_arrow_frame(frame_index, "箭头中心位置无效")
            return

        bbox_values = None
        if isinstance(bbox, dict):
            candidate = tuple(
                self.finite_number(bbox.get(key))
                for key in ("x1", "y1", "x2", "y2")
            )
            if all(value is not None for value in candidate):
                bbox_values = candidate
        if (
            bbox_values is None
            or bbox_values[2] <= bbox_values[0]
            or bbox_values[3] <= bbox_values[1]
        ):
            self.reject_arrow_frame(
                frame_index, "bbox无效，无法进行位置候选组一致性判断"
            )
            return

        detection = {
            "frame_index": frame_index,
            "received_time": now,
            "confidence": confidence,
            "center_u": center_u,
            "center_v": center_v,
            "angle_deg": (
                None if angle_deg is None else normalize_angle_deg(angle_deg)
            ),
            "direction": str(
                payload.get("discrete_direction", "")
            ).strip(),
            "bbox": bbox_values,
            "area": (
                (bbox_values[2] - bbox_values[0])
                * (bbox_values[3] - bbox_values[1])
            ),
        }
        full_visible, full_visible_reason = self.full_arrow_visible(detection)
        detection["full_visible"] = full_visible
        detection["full_visible_reason"] = full_visible_reason
        self.latest_detection = detection
        self.last_valid_detection_time = now
        if full_visible and detection["angle_deg"] is not None:
            self.last_full_direction_detection_time = now
        error_u, error_v, _, _ = self.detection_center_errors(detection)
        bbox_text = "缺失"
        if bbox_values is not None:
            bbox_text = "({:.0f},{:.0f},{:.0f},{:.0f})".format(*bbox_values)
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 有效：conf=%.3f，中心=(%.1f,%.1f)，"
                "误差=(u=%+.1f,v=%+.1f)px，bbox=%s，完整可见=%s（%s），"
                "角度=%s，方向=%s，阶段=%s"
            ),
            NODE_NAME,
            frame_index,
            confidence,
            center_u,
            center_v,
            error_u,
            error_v,
            bbox_text,
            "是" if full_visible else "否",
            full_visible_reason,
            (
                "未提供"
                if detection["angle_deg"] is None
                else "{:.1f}deg".format(detection["angle_deg"])
            ),
            detection["direction"] or "未知",
            self.state,
        )

        if self.state == self.SEARCH_PATTERN:
            if not self.first_position_detected:
                self.first_position_detected = True
                rospy.logwarn(
                    "%s：[箭头帧#%d] 搜索中首次获得有效位置，"
                    "立即退出固定路径搜索；方向字段本次不参与判断",
                    NODE_NAME,
                    frame_index,
                )
            self.add_detection_sample(detection, frame_index)
        elif self.state == self.WAIT_FOR_ARROW:
            self.add_detection_sample(detection, frame_index)
        elif self.state == self.COARSE_LATERAL_ALIGN:
            self.update_center_progress(detection, error_u, error_v)
            self.add_direction_sample(detection, error_u, error_v)
        elif self.state == self.CONFIRM_DIRECTION:
            self.add_direction_sample(detection, error_u, error_v)
        elif self.state == self.ALIGN_HEADING:
            self.add_direction_sample(detection, error_u, error_v)
            self.update_heading_alignment_progress(
                detection, error_u, error_v
            )
        elif self.state == self.FINE_FORWARD_ALIGN:
            self.add_direction_sample(detection, error_u, error_v)
            self.update_alignment_progress(detection, error_u, error_v)
        elif self.state == self.MOVE_BASE_OVER_ARROW:
            self.add_direction_sample(detection, error_u, error_v)
            self.update_base_tracking_progress(
                detection, error_u, error_v
            )

    def add_detection_sample(self, detection, frame_index, invalid_reason=""):
        self.detection_samples.append({
            "frame_index": frame_index,
            "detection": detection,
        })
        self.detection_samples = self.detection_samples[
            -self.stable_detection_window_size:
        ]

        valid_samples = [
            item["detection"]
            for item in self.detection_samples
            if item["detection"] is not None
        ]
        candidate_groups = self.build_detection_candidate_groups(valid_samples)
        window_count = len(self.detection_samples)
        best_group_count = max(
            (len(group) for group in candidate_groups),
            default=0,
        )

        if detection is None:
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] 本帧无效：%s；窗口=%d/%d帧，"
                    "有效位置帧=%d/%d，最佳候选组=%d/%d；保留旧有效帧"
                ),
                NODE_NAME,
                frame_index,
                invalid_reason or "没有有效箭头",
                window_count,
                self.stable_detection_window_size,
                len(valid_samples),
                window_count,
                best_group_count,
                self.stable_detection_count,
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
            current_group
        )
        frame_ids = [item["frame_index"] for item in current_group]
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 有效位置加入候选组%d；窗口=%d/%d帧，"
                "有效位置帧=%d/%d，当前候选组=%d/%d，命中帧=%s，"
                "中心抖动=%.1f/%.1fpx，面积变化=%.3f/%.3f"
            ),
            NODE_NAME,
            frame_index,
            current_group_index,
            window_count,
            self.stable_detection_window_size,
            len(valid_samples),
            window_count,
            len(current_group),
            self.stable_detection_count,
            frame_ids,
            center_jitter,
            self.stable_center_tolerance_px,
            area_change,
            self.stable_area_tolerance_ratio,
        )
        if not stable:
            return

        locked = dict(current_group[-1])
        locked["center_u"] = statistics.median(
            item["center_u"] for item in current_group
        )
        locked["center_v"] = statistics.median(
            item["center_v"] for item in current_group
        )
        locked["confidence"] = sum(
            item["confidence"] for item in current_group
        ) / len(current_group)
        self.latest_detection = locked
        self.arrow_locked = True
        rospy.loginfo(
            (
                "%s：箭头位置候选组确认通过：最近%d帧内命中%d/%d帧，"
                "命中帧=%s，中位中心=(%.1f,%.1f)，平均置信度=%.3f；"
                "方向暂不参与判断"
            ),
            NODE_NAME,
            self.stable_detection_window_size,
            len(current_group),
            self.stable_detection_count,
            frame_ids,
            locked["center_u"],
            locked["center_v"],
            locked["confidence"],
        )

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

    def samples_are_stable(self, samples):
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
            len(samples) >= self.stable_detection_count
            and center_jitter <= self.stable_center_tolerance_px
            and area_change <= self.stable_area_tolerance_ratio
        )
        return stable, center_jitter, area_change

    def detection_window_progress(self):
        valid_samples = [
            item["detection"]
            for item in self.detection_samples
            if item["detection"] is not None
        ]
        groups = self.build_detection_candidate_groups(valid_samples)
        return (
            len(self.detection_samples),
            len(valid_samples),
            max((len(group) for group in groups), default=0),
        )

    def add_direction_sample(self, detection, error_u, error_v):
        del error_u
        del error_v
        if not detection["full_visible"]:
            previous = len(self.direction_samples)
            self.reset_direction_lock()
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] 箭头未完整留在画面内，"
                    "方向计数%d -> 0；原因=%s"
                ),
                NODE_NAME,
                detection["frame_index"],
                previous,
                detection["full_visible_reason"],
            )
            return
        if detection["angle_deg"] is None:
            previous = len(self.direction_samples)
            self.reset_direction_lock()
            rospy.loginfo(
                "%s：[箭头帧#%d] 位置有效但方向字段无效，方向计数%d -> 0",
                NODE_NAME,
                detection["frame_index"],
                previous,
            )
            return
        if self.direction_samples:
            gap = (
                detection["received_time"]
                - self.direction_samples[-1]["received_time"]
            ).to_sec()
            if gap > self.detection_timeout:
                rospy.logwarn(
                    "%s：完整箭头方向帧间隔%.2fs超过%.2fs，方向计数清零",
                    NODE_NAME,
                    gap,
                    self.detection_timeout,
                )
                self.reset_direction_lock()
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
                "%s：[箭头帧#%d] 方向识别第%d/%d帧，"
                "完整箭头[通过]，平均角度=%.1fdeg，抖动=%.1f/%.1fdeg"
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
            "%s：完整箭头方向连续%d帧稳定，平均角度=%.1fdeg",
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
            self.reset_center_progress("箭头中心超出粗对准容差")
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] 图像粗居中未通过："
                    "u误差=%+.1f/%.1fpx，v误差=%+.1f/%.1fpx"
                ),
                NODE_NAME,
                detection["frame_index"],
                error_u,
                self.center_tolerance_u_px,
                error_v,
                self.center_tolerance_v_px,
            )
            return
        self.centered_frame_count = min(
            self.centered_frame_count + 1,
            self.center_stable_detection_count,
        )
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 图像粗居中第%d/%d帧有效，"
                "误差=(u=%+.1f,v=%+.1f)px"
            ),
            NODE_NAME,
            detection["frame_index"],
            self.centered_frame_count,
            self.center_stable_detection_count,
            error_u,
            error_v,
        )

    def arrow_heading_error_deg(self):
        if not self.direction_locked or self.direction_locked_angle_deg is None:
            return None
        return self.yaw_correction_sign * normalize_angle_deg(
            self.camera_forward_angle_deg - self.direction_locked_angle_deg
        )

    def reset_heading_alignment_progress(self, reason):
        previous = self.heading_aligned_frame_count
        self.heading_aligned_frame_count = 0
        if previous > 0:
            rospy.loginfo(
                "%s：航向对齐计数%d -> 0，原因=%s",
                NODE_NAME,
                previous,
                reason,
            )

    def update_heading_alignment_progress(self, detection, error_u, error_v):
        del error_v
        heading_error = self.arrow_heading_error_deg()
        if (
            not detection["full_visible"]
            or heading_error is None
            or abs(heading_error) > self.yaw_tolerance_deg
            or abs(error_u) > self.center_tolerance_u_px
        ):
            self.reset_heading_alignment_progress(
                "完整可见、左右位置或箭头航向未同时通过"
            )
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] 慢速航向对齐未通过：完整可见=%s，"
                    "u误差=%+.1f/%.1fpx，方向误差=%s/%.1fdeg"
                ),
                NODE_NAME,
                detection["frame_index"],
                "是" if detection["full_visible"] else "否",
                error_u,
                self.center_tolerance_u_px,
                "未知" if heading_error is None else "{:+.1f}".format(
                    heading_error
                ),
                self.yaw_tolerance_deg,
            )
            return
        self.heading_aligned_frame_count = min(
            self.heading_aligned_frame_count + 1,
            self.heading_aligned_detection_count,
        )
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 慢速航向对齐第%d/%d帧有效："
                "完整箭头[通过]，u误差=%+.1fpx，方向误差=%+.1fdeg"
            ),
            NODE_NAME,
            detection["frame_index"],
            self.heading_aligned_frame_count,
            self.heading_aligned_detection_count,
            error_u,
            heading_error,
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
        if not detection["full_visible"] or not self.direction_locked:
            self.reset_alignment_progress("细对准阶段缺少箭头方向")
            rospy.loginfo(
                "%s：[箭头帧#%d] 箭头不完整或方向未稳定，细对准帧不计数",
                NODE_NAME,
                detection["frame_index"],
            )
            return
        centered = (
            abs(error_u) <= self.center_tolerance_u_px
            and abs(error_v) <= self.center_tolerance_v_px
        )
        arrow_heading_error = self.arrow_heading_error_deg()
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

    def reset_base_tracking_progress(self, reason):
        previous = self.base_tracking_frame_count
        self.base_tracking_frame_count = 0
        if previous > 0:
            rospy.loginfo(
                "%s：base_link前移视觉跟踪计数%d -> 0，原因=%s",
                NODE_NAME,
                previous,
                reason,
            )

    def update_base_tracking_progress(self, detection, error_u, error_v):
        heading_error = self.arrow_heading_error_deg()
        tracking_ok = (
            detection["full_visible"]
            and heading_error is not None
            and abs(error_u) <= self.center_tolerance_u_px
            and abs(heading_error) <= self.yaw_tolerance_deg
        )
        if not tracking_ok:
            self.reset_base_tracking_progress(
                "完整箭头、左右位置或箭头航向未同时通过"
            )
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] base_link前移视觉跟踪未通过："
                    "完整可见=%s，u误差=%+.1f/%.1fpx，方向误差=%s/%.1fdeg，"
                    "v误差=%+.1fpx仅记录"
                ),
                NODE_NAME,
                detection["frame_index"],
                "是" if detection["full_visible"] else "否",
                error_u,
                self.center_tolerance_u_px,
                "未知" if heading_error is None else "{:+.1f}".format(
                    heading_error
                ),
                self.yaw_tolerance_deg,
                error_v,
            )
            return
        self.base_tracking_frame_count = min(
            self.base_tracking_frame_count + 1,
            self.center_stable_detection_count,
        )
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] base_link前移视觉跟踪第%d/%d帧有效："
                "u误差=%+.1fpx，方向误差=%+.1fdeg，v误差=%+.1fpx仅记录"
            ),
            NODE_NAME,
            detection["frame_index"],
            self.base_tracking_frame_count,
            self.center_stable_detection_count,
            error_u,
            heading_error,
            error_v,
        )

    def get_current_pose(self, context):
        try:
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
        self.initial_hold_x = current.pose.position.x
        self.initial_hold_y = current.pose.position.y
        self.initial_hold_yaw = current_yaw
        self.target_z = current.pose.position.z
        self.target_depth = status["depth"]
        self.control_initialized = True
        self.set_active_goal(
            current.pose.position.x,
            current.pose.position.y,
            self.target_z,
            current_yaw,
            "只锁存一次启动位置，后续漂移时仍追踪该固定悬停点",
        )
        self.set_state(
            self.INITIAL_HOVER,
            "TF和/status/auv已就绪，开始追踪固定启动点",
        )
        rospy.loginfo(
            "%s：固定悬停点已锁存：map=(%.3f,%.3f,%.3f)，yaw=%.2fdeg；"
            "悬停期间不会随当前漂移位置更新",
            NODE_NAME,
            self.initial_hold_x,
            self.initial_hold_y,
            self.target_z,
            math.degrees(self.initial_hold_yaw),
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
        current = self.get_current_pose("最低对地距离保护")
        if current is None:
            return
        current_z = current.pose.position.z
        current_clearance = -current_z
        safe_z = -self.min_ground_clearance
        target_adjustment = self.target_z - safe_z
        if target_adjustment >= self.ground_clearance_goal_update_threshold:
            previous_target_z = self.target_z
            self.target_z = safe_z
            self.active_goal.pose.position.z = safe_z
            self.target_depth -= target_adjustment
            rospy.logwarn(
                (
                    "%s：离地保护触发：map实际z=%.3f（离底约%.3fm），"
                    "目标不得低于%.3fm；"
                    "目标z从%.3f改为%.3f（改写%.3fm），目标深度=%.3f"
                ),
                NODE_NAME,
                current_z,
                current_clearance,
                self.min_ground_clearance,
                previous_target_z,
                safe_z,
                target_adjustment,
                self.target_depth,
            )
            return
        if current_clearance < self.min_ground_clearance:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                (
                    "%s：map实际z=%.3f（离底约%.3fm）低于%.3fm，"
                    "当前安全目标z=%.3f，等待motion_supervisor抬升并定点"
                ),
                NODE_NAME,
                current_z,
                current_clearance,
                self.min_ground_clearance,
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

    def actual_arrival_checks(self):
        message = self.latest_motion_state
        status = self.current_status
        if message is None or status is None or self.target_depth is None:
            return None
        values = (
            message.base_position_error,
            message.yaw_error,
            message.horizontal_speed,
            message.yaw_rate,
            status["depth"],
        )
        if not all(math.isfinite(value) for value in values):
            return None
        depth_error = abs(status["depth"] - self.target_depth)
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
            "depth_error": depth_error,
            "depth_ok": depth_error <= self.max_depth_error,
        }

    def actual_arrival_satisfied(self):
        checks = self.actual_arrival_checks()
        return (
            checks is not None
            and checks["position_ok"]
            and checks["yaw_ok"]
            and checks["speed_ok"]
            and checks["yaw_rate_ok"]
            and checks["depth_ok"]
        )

    def motion_arrived(self):
        return (
            self.motion_hover_fresh()
            and self.goal_matches_motion_state()
            and self.actual_arrival_satisfied()
        )

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

    def full_direction_detection_age(self):
        if self.last_full_direction_detection_time is None:
            return None
        return max(
            0.0,
            (
                rospy.Time.now() - self.last_full_direction_detection_time
            ).to_sec(),
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

    def fine_detection_available_within(self, timeout, context):
        full_age = self.full_direction_detection_age()
        available = (
            self.latest_detection is not None
            and self.latest_detection["full_visible"]
            and self.latest_detection["angle_deg"] is not None
            and full_age is not None
            and full_age <= timeout
        )
        if available:
            return True
        valid_age = self.valid_detection_age()
        latest_reason = "无有效位置帧"
        if self.latest_detection is not None:
            latest_reason = self.latest_detection["full_visible_reason"]
            if self.latest_detection["angle_deg"] is None:
                latest_reason += "，且缺少方向"
        rospy.logwarn_throttle(
            self.warning_log_interval,
            (
                "%s：%s没有可用于细对准的完整箭头方向帧："
                "完整方向帧年龄=%s，普通位置帧年龄=%s，原因=%s，保护阈值=%.2fs"
            ),
            NODE_NAME,
            context,
            "从未获得" if full_age is None else "{:.2f}s".format(full_age),
            "从未获得" if valid_age is None else "{:.2f}s".format(valid_age),
            latest_reason,
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

    def minimum_visual_step(self, value, min_step=None, max_step=None):
        if min_step is None:
            min_step = self.visual_min_step_m
        if max_step is None:
            max_step = self.visual_max_step_m
        value = clamp(value, -max_step, max_step)
        if value == 0.0 or abs(value) >= min_step:
            return value
        return math.copysign(min_step, value)

    def visual_goal_update_ready(self, detection):
        if detection["frame_index"] == self.last_visual_goal_frame:
            return False
        if self.last_visual_goal_time is not None:
            goal_age = (
                rospy.Time.now() - self.last_visual_goal_time
            ).to_sec()
            if goal_age < self.visual_goal_min_interval:
                rospy.loginfo_throttle(
                    self.log_interval,
                    "%s：视觉小步间隔%.2f/%.2fs，本帧暂不生成新目标",
                    NODE_NAME,
                    goal_age,
                    self.visual_goal_min_interval,
                )
                return False
        if self.active_goal is not None and not self.motion_arrived():
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：上一视觉小步尚未满足实际到达门槛，"
                    "本帧只更新识别结果，不叠加新目标"
                ),
                NODE_NAME,
            )
            return False
        return True

    def fine_goal_update_ready(self, detection):
        if detection["frame_index"] == self.last_visual_goal_frame:
            return False
        if self.last_visual_goal_time is None:
            return True
        goal_age = (rospy.Time.now() - self.last_visual_goal_time).to_sec()
        if goal_age >= self.fine_goal_min_interval:
            return True
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：细对准实时目标间隔%.2f/%.2fs，本帧只更新识别结果",
            NODE_NAME,
            goal_age,
            self.fine_goal_min_interval,
        )
        return False

    def update_coarse_center_goal(self, target_yaw, context):
        detection = self.latest_detection
        if detection is None:
            return False
        if not self.visual_goal_update_ready(detection):
            return True
        error_u, error_v, normalized_u, normalized_v = (
            self.detection_center_errors(detection)
        )
        if (
            abs(error_u) <= self.center_tolerance_u_px
            and abs(error_v) <= self.center_tolerance_v_px
        ):
            self.last_visual_goal_frame = detection["frame_index"]
            rospy.loginfo(
                "%s：[箭头帧#%d] 箭头中心已进入粗对准容差；"
                "误差=(u=%+.1f,v=%+.1f)px",
                NODE_NAME,
                detection["frame_index"],
                error_u,
                error_v,
            )
            return True
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
        current = self.get_current_pose(context)
        if current is None:
            return False
        goal_x, goal_y = self.set_body_offset_goal(
            current,
            forward_step,
            right_step,
            target_yaw,
            "{}：粗对准生成前后和左右居中目标，航向保持不变".format(context),
        )
        self.last_visual_goal_frame = detection["frame_index"]
        self.last_visual_goal_time = rospy.Time.now()
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 图像粗居中目标：误差=(u=%+.1f,v=%+.1f)px，"
                "本体偏置=(前%+.3f,右%+.3f)m，"
                "map目标=(%.3f,%.3f)，航向保持=%.2fdeg"
            ),
            NODE_NAME,
            detection["frame_index"],
            error_u,
            error_v,
            forward_step,
            right_step,
            goal_x,
            goal_y,
            math.degrees(target_yaw),
        )
        return True

    def update_fine_visual_goal(self, allow_forward, context):
        detection = self.latest_detection
        if detection is None:
            return False
        if not self.fine_goal_update_ready(detection):
            return True
        error_u, error_v, normalized_u, normalized_v = (
            self.detection_center_errors(detection)
        )
        forward_step = 0.0
        if allow_forward and abs(error_v) > self.center_tolerance_v_px:
            forward_step = self.minimum_visual_step(
                self.visual_forward_sign
                * -self.fine_forward_gain_m
                * normalized_v,
                self.fine_visual_min_step_m,
                self.fine_visual_max_step_m,
            )
        right_step = 0.0
        if abs(error_u) > self.center_tolerance_u_px:
            right_step = self.minimum_visual_step(
                self.visual_lateral_sign
                * self.fine_lateral_gain_m
                * normalized_u,
                self.fine_visual_min_step_m,
                self.fine_visual_max_step_m,
            )
        heading_error = self.arrow_heading_error_deg()
        if heading_error is None:
            return False
        yaw_step_deg = 0.0
        if abs(heading_error) > self.yaw_tolerance_deg:
            yaw_step_deg = clamp(
                heading_error,
                -self.fine_yaw_max_step_deg,
                self.fine_yaw_max_step_deg,
            )
        current = self.get_current_pose(context)
        if current is None:
            return False
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        target_yaw = normalize_angle_rad(
            current_yaw + math.radians(yaw_step_deg)
        )
        self.final_target_yaw = target_yaw
        if (
            forward_step == 0.0
            and right_step == 0.0
            and yaw_step_deg == 0.0
        ):
            self.last_visual_goal_frame = detection["frame_index"]
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] 细对准已在本阶段容差内："
                    "允许前后=%s，误差=(u=%+.1f,v=%+.1f)px，方向误差=%+.1fdeg"
                ),
                NODE_NAME,
                detection["frame_index"],
                "是" if allow_forward else "否",
                error_u,
                error_v,
                heading_error,
            )
            return True
        goal_x, goal_y = self.set_body_offset_goal(
            current,
            forward_step,
            right_step,
            target_yaw,
            "{}：按最新完整箭头位置和方向生成慢速小步目标".format(context),
        )
        self.last_visual_goal_frame = detection["frame_index"]
        self.last_visual_goal_time = rospy.Time.now()
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 细对准实时目标：完整箭头[通过]，"
                "允许前后=%s，误差=(u=%+.1f,v=%+.1f)px，"
                "本体偏置=(前%+.3f,右%+.3f)m，"
                "方向误差/航向小步=(%+.2f/%+.2f)deg，"
                "map目标=(%.3f,%.3f, yaw=%.2fdeg)"
            ),
            NODE_NAME,
            detection["frame_index"],
            "是" if allow_forward else "否",
            error_u,
            error_v,
            forward_step,
            right_step,
            heading_error,
            yaw_step_deg,
            goal_x,
            goal_y,
            math.degrees(target_yaw),
        )
        return True

    def update_base_tracking_goal(self):
        detection = self.latest_detection
        if detection is None or self.active_goal is None:
            return False
        if not self.fine_goal_update_ready(detection):
            return True
        error_u, error_v, normalized_u, unused_normalized_v = (
            self.detection_center_errors(detection)
        )
        del unused_normalized_v
        right_step = 0.0
        if abs(error_u) > self.center_tolerance_u_px:
            right_step = self.minimum_visual_step(
                self.visual_lateral_sign
                * self.fine_lateral_gain_m
                * normalized_u,
                self.fine_visual_min_step_m,
                self.fine_visual_max_step_m,
            )
        heading_error = self.arrow_heading_error_deg()
        if heading_error is None:
            return False
        yaw_step_deg = 0.0
        if abs(heading_error) > self.yaw_tolerance_deg:
            yaw_step_deg = clamp(
                heading_error,
                -self.fine_yaw_max_step_deg,
                self.fine_yaw_max_step_deg,
            )
        if right_step == 0.0 and yaw_step_deg == 0.0:
            self.last_visual_goal_frame = detection["frame_index"]
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] base_link前移期间不改目标："
                    "u/v误差=(%+.1f,%+.1f)px，方向误差=%+.1fdeg，均在控制容差内"
                ),
                NODE_NAME,
                detection["frame_index"],
                error_u,
                error_v,
                heading_error,
            )
            return True
        current = self.get_current_pose("base_link前移期间实时视觉修正")
        if current is None:
            return False
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        target_yaw = normalize_angle_rad(
            current_yaw + math.radians(yaw_step_deg)
        )
        target_x = self.active_goal.pose.position.x
        target_y = self.active_goal.pose.position.y
        if right_step != 0.0:
            target_x -= math.sin(current_yaw) * right_step
            target_y += math.cos(current_yaw) * right_step
        self.final_target_yaw = target_yaw
        self.set_active_goal(
            target_x,
            target_y,
            self.target_z,
            target_yaw,
            "base_link前移期间按最新完整箭头做慢速横向和航向修正",
        )
        self.last_visual_goal_frame = detection["frame_index"]
        self.last_visual_goal_time = rospy.Time.now()
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] base_link前移实时修正："
                "u/v误差=(%+.1f,%+.1f)px，右移修正=%+.3fm，"
                "方向误差/航向修正=(%+.2f/%+.2f)deg，"
                "map目标=(%.3f,%.3f,yaw=%.2fdeg)，"
                "前后目标保持为已标定的%.3fm偏置"
            ),
            NODE_NAME,
            detection["frame_index"],
            error_u,
            error_v,
            right_step,
            heading_error,
            yaw_step_deg,
            target_x,
            target_y,
            math.degrees(target_yaw),
            self.base_link_forward_offset,
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
                self.reset_first_lock()
                self.first_position_detected = False
                self.build_search_waypoints()
                self.activate_search_waypoint(0)
                self.set_state(
                    self.SEARCH_PATTERN,
                    "固定点悬停完成，开始执行固定绝对坐标搜索路径",
                )
        else:
            self.initial_hover_stable_started = None
            self.log_arrival_gate("等待初始HOVER接管")

    def build_search_waypoints(self):
        first_forward = self.search_initial_forward_distance
        second_forward = (
            first_forward + self.search_second_forward_distance
        )
        lateral = self.search_lateral_distance
        offsets = (
            (first_forward, 0.0, "前进{:.2f}m".format(first_forward)),
            (first_forward, -lateral, "第一层左移{:.2f}m".format(lateral)),
            (first_forward, lateral, "第一层右移{:.2f}m".format(lateral)),
            (first_forward, 0.0, "第一层回到中线"),
            (second_forward, 0.0, "沿中线再前进{:.2f}m".format(
                self.search_second_forward_distance
            )),
            (second_forward, -lateral, "第二层左移{:.2f}m".format(lateral)),
            (second_forward, lateral, "第二层右移{:.2f}m".format(lateral)),
        )
        cos_yaw = math.cos(self.initial_hold_yaw)
        sin_yaw = math.sin(self.initial_hold_yaw)
        self.search_waypoints = []
        for forward, right, label in offsets:
            self.search_waypoints.append({
                "x": self.initial_hold_x + cos_yaw * forward - sin_yaw * right,
                "y": self.initial_hold_y + sin_yaw * forward + cos_yaw * right,
                "forward": forward,
                "right": right,
                "label": label,
            })
        rospy.loginfo(
            (
                "%s：固定搜索路径已生成，共%d点；所有点均相对启动悬停点计算，"
                "不会随机器人漂移位置重新累加"
            ),
            NODE_NAME,
            len(self.search_waypoints),
        )

    def activate_search_waypoint(self, index):
        waypoint = self.search_waypoints[index]
        self.search_waypoint_index = index
        self.set_active_goal(
            waypoint["x"],
            waypoint["y"],
            self.target_z,
            self.initial_hold_yaw,
            "搜索第{}/{}点：{}".format(
                index + 1, len(self.search_waypoints), waypoint["label"]
            ),
        )
        rospy.loginfo(
            (
                "%s：搜索路径第%d/%d点：%s，本体固定偏置=(前%.2f,右%+.2f)m，"
                "map目标=(%.3f,%.3f)，航向固定=%.2fdeg"
            ),
            NODE_NAME,
            index + 1,
            len(self.search_waypoints),
            waypoint["label"],
            waypoint["forward"],
            waypoint["right"],
            waypoint["x"],
            waypoint["y"],
            math.degrees(self.initial_hold_yaw),
        )

    def control_search_pattern(self):
        if self.first_position_detected:
            self.begin_cancel(
                self.WAIT_FOR_ARROW,
                "搜索中出现首帧有效箭头位置，立即退出搜索并刹停；"
                "随后按最近{}帧候选组重新确认".format(
                    self.stable_detection_window_size
                ),
            )
            return
        if self.motion_arrived():
            next_index = self.search_waypoint_index + 1
            if next_index >= len(self.search_waypoints):
                self.finish_task(
                    False,
                    "固定搜索路径全部完成仍未获得有效箭头位置",
                )
                return
            self.activate_search_waypoint(next_index)
            return
        model_age = None
        if self.last_model_message_time is not None:
            model_age = (
                rospy.Time.now() - self.last_model_message_time
            ).to_sec()
        window_count, valid_count, best_group_count = (
            self.detection_window_progress()
        )
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：固定路径搜索第%d/%d点：motion=%s，实际位置误差=%.3fm，"
                "识别窗口=%d/%d帧，有效=%d帧，最佳候选组=%d/%d帧，"
                "模型消息年龄=%s"
            ),
            NODE_NAME,
            self.search_waypoint_index + 1,
            len(self.search_waypoints),
            self.current_motion_state_name(),
            self.latest_motion_state.base_position_error,
            window_count,
            self.stable_detection_window_size,
            valid_count,
            best_group_count,
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
        self.last_visual_goal_time = None
        if next_state == self.WAIT_FOR_ARROW:
            self.reset_first_lock()
        elif next_state == self.COARSE_LATERAL_ALIGN:
            self.reset_center_progress("进入保持航向的图像中心粗对准")
        elif next_state == self.CONFIRM_DIRECTION:
            self.reset_direction_lock()
        self.set_state(
            next_state,
            "motion_supervisor已完成刹停并由mode4接管",
        )

    def control_wait_for_arrow(self):
        if self.arrow_locked:
            self.reset_center_progress("重新锁定箭头位置")
            self.reset_direction_lock()
            self.last_visual_goal_frame = 0
            self.last_visual_goal_time = None
            self.set_state(
                self.COARSE_LATERAL_ALIGN,
                "定点完成{}/{}位置候选组确认，进入图像中心粗对准".format(
                    self.stable_detection_count,
                    self.stable_detection_window_size,
                ),
            )
            return
        model_age = None
        if self.last_model_message_time is not None:
            model_age = (
                rospy.Time.now() - self.last_model_message_time
            ).to_sec()
        window_count, valid_count, best_group_count = (
            self.detection_window_progress()
        )
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：定点重识别：窗口=%d/%d帧，有效=%d帧，"
                "最佳候选组=%d/%d帧，模型年龄=%s，motion=%s"
            ),
            NODE_NAME,
            window_count,
            self.stable_detection_window_size,
            valid_count,
            best_group_count,
            self.stable_detection_count,
            "未收到" if model_age is None else "{:.2f}s".format(model_age),
            self.current_motion_state_name(),
        )

    def control_coarse_lateral_align(self):
        if not self.detection_available_within(
            self.visual_loss_cancel_seconds, "图像中心粗对准阶段"
        ):
            valid_age = self.valid_detection_age()
            if valid_age is None or valid_age > self.visual_loss_cancel_seconds:
                self.begin_cancel(
                    self.WAIT_FOR_ARROW,
                    (
                        "图像中心粗对准时箭头丢失{}，超过{:.2f}s阈值，"
                        "刹停后定点重识别"
                    ).format(
                        "未知" if valid_age is None else "{:.2f}s".format(
                            valid_age
                        ),
                        self.visual_loss_cancel_seconds,
                    ),
                )
            return
        if (
            self.direction_locked
            and self.centered_frame_count >= self.center_stable_detection_count
        ):
            self.begin_cancel(
                self.CONFIRM_DIRECTION,
                "图像中心粗对准和完整箭头方向均稳定，刹停后定点复核方向",
            )
            return
        if self.centered_frame_count >= self.center_stable_detection_count:
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：图像中心粗对准已稳定%d/%d帧，保持当前固定目标等待"
                    "完整箭头方向%d/%d帧"
                ),
                NODE_NAME,
                self.centered_frame_count,
                self.center_stable_detection_count,
                len(self.direction_samples),
                self.heading_stable_detection_count,
            )
            return
        target_yaw = yaw_from_quaternion(self.active_goal.pose.orientation)
        self.update_coarse_center_goal(
            target_yaw, "保持发现航向并做前后、左右图像粗居中"
        )
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：图像中心粗对准进度=%d/%d帧，motion=%s，位置误差=%.3fm",
            NODE_NAME,
            self.centered_frame_count,
            self.center_stable_detection_count,
            self.current_motion_state_name(),
            self.latest_motion_state.position_error,
        )

    def control_confirm_direction(self):
        if not self.fine_detection_available_within(
            self.detection_timeout, "定点复核完整箭头方向阶段"
        ):
            full_age = self.full_direction_detection_age()
            if full_age is None or full_age > self.detection_timeout:
                self.begin_cancel(
                    self.WAIT_FOR_ARROW,
                    "定点复核时完整箭头方向丢失超过{:.2f}s".format(
                        self.detection_timeout
                    ),
                )
            return
        error_u, error_v, _, _ = self.detection_center_errors(
            self.latest_detection
        )
        if not self.direction_locked:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：保持HOVER复核完整箭头方向%d/%d帧；"
                "当前位置误差=(u=%+.1f,v=%+.1f)px，v误差将在细对准处理",
                NODE_NAME,
                len(self.direction_samples),
                self.heading_stable_detection_count,
                error_u,
                error_v,
            )
            return
        if not self.motion_arrived():
            self.log_arrival_gate("方向已稳定，等待当前定点目标HOVER")
            return
        current = self.get_current_pose("进入实时慢速航向对准")
        if current is None:
            return
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        correction_deg = self.yaw_correction_sign * normalize_angle_deg(
            self.camera_forward_angle_deg - self.direction_locked_angle_deg
        )
        self.final_target_yaw = current_yaw
        self.heading_aligned_frame_count = 0
        self.last_visual_goal_frame = 0
        self.last_visual_goal_time = None
        self.set_active_goal(
            current.pose.position.x,
            current.pose.position.y,
            self.target_z,
            current_yaw,
            "完整箭头方向稳定，先保持当前位置并开始实时慢速航向对准",
        )
        rospy.loginfo(
            (
                "%s：进入慢速航向对准：当前=%.2fdeg，箭头平均角度=%.2fdeg，"
                "当前相对误差=%+.2fdeg；后续每次只按最新完整箭头修正，"
                "单次不超过%.2fdeg"
            ),
            NODE_NAME,
            math.degrees(current_yaw),
            self.direction_locked_angle_deg,
            correction_deg,
            self.fine_yaw_max_step_deg,
        )
        self.set_state(
            self.ALIGN_HEADING,
            "完整箭头方向已复核，禁止前后移动，只慢速横移并实时对齐航向",
        )

    def control_align_heading(self):
        if not self.fine_detection_available_within(
            self.visual_loss_cancel_seconds, "慢速航向对准阶段"
        ):
            full_age = self.full_direction_detection_age()
            if full_age is None or full_age > self.visual_loss_cancel_seconds:
                self.begin_cancel(
                    self.WAIT_FOR_ARROW,
                    (
                        "航向对准时完整箭头方向丢失{}，超过{:.2f}s阈值，"
                        "取消并定点重识别"
                    ).format(
                        "未知" if full_age is None else "{:.2f}s".format(
                            full_age
                        ),
                        self.visual_loss_cancel_seconds,
                    ),
                )
            return
        if not self.direction_locked:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：慢速航向对准暂停，等待完整箭头方向重新稳定",
                NODE_NAME,
            )
            return
        self.update_fine_visual_goal(
            False, "细对准第一段：禁止前后移动，实时慢速横移和转向"
        )
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：慢速航向对准：确认=%d/%d帧，motion=%s，"
                "位置/航向误差=(%.3fm,%+.2fdeg)，目标匹配=%s"
            ),
            NODE_NAME,
            self.heading_aligned_frame_count,
            self.heading_aligned_detection_count,
            self.current_motion_state_name(),
            self.latest_motion_state.position_error,
            math.degrees(self.latest_motion_state.yaw_error),
            "通过" if self.goal_matches_motion_state() else "未通过",
        )
        if self.heading_aligned_frame_count < self.heading_aligned_detection_count:
            return
        if not self.motion_arrived():
            self.log_arrival_gate(
                "箭头航向已稳定，等待最新慢速航向目标HOVER"
            )
            return
        self.aligned_frame_count = 0
        self.last_visual_goal_frame = 0
        self.last_visual_goal_time = None
        self.set_state(
            self.FINE_FORWARD_ALIGN,
            "航向连续稳定且已停稳，开始保持实时方向并慢速前后居中",
        )

    def control_fine_forward_align(self):
        if not self.fine_detection_available_within(
            self.visual_loss_cancel_seconds, "航向对齐后的慢速前后居中阶段"
        ):
            full_age = self.full_direction_detection_age()
            if full_age is None or full_age > self.visual_loss_cancel_seconds:
                self.begin_cancel(
                    self.WAIT_FOR_ARROW,
                    (
                        "慢速前后居中时完整箭头方向丢失{}，超过{:.2f}s阈值，"
                        "取消并定点重识别"
                    ).format(
                        "未知" if full_age is None else "{:.2f}s".format(
                            full_age
                        ),
                        self.visual_loss_cancel_seconds,
                    ),
                )
            return
        if not self.direction_locked:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：慢速前后居中暂停，等待完整箭头方向重新稳定",
                NODE_NAME,
            )
            return
        self.update_fine_visual_goal(
            True, "细对准第二段：保持方向并慢速前后、左右居中"
        )
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：慢速前后居中：最终确认=%d/%d帧，motion=%s，"
                "位置/航向误差=(%.3fm,%+.2fdeg)"
            ),
            NODE_NAME,
            self.aligned_frame_count,
            self.center_stable_detection_count,
            self.current_motion_state_name(),
            self.latest_motion_state.position_error,
            math.degrees(self.latest_motion_state.yaw_error),
        )
        if self.aligned_frame_count < self.center_stable_detection_count:
            return
        if not self.motion_arrived():
            self.log_arrival_gate(
                "完整箭头中心和方向已稳定，等待最新视觉目标HOVER"
            )
            return
        self.start_base_over_arrow_offset()

    def start_base_over_arrow_offset(self):
        current = self.get_current_pose("生成base_link移动到箭头上方的目标")
        if current is None:
            return
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        goal_x = (
            current.pose.position.x
            + math.cos(current_yaw) * self.base_link_forward_offset
        )
        goal_y = (
            current.pose.position.y
            + math.sin(current_yaw) * self.base_link_forward_offset
        )
        self.set_active_goal(
            goal_x,
            goal_y,
            self.target_z,
            current_yaw,
            "航向和图像中心已对准，base_link沿当前前方移动到箭头上方",
        )
        self.final_target_yaw = current_yaw
        rospy.loginfo(
            (
                "%s：base_link位置补偿只计算一次：起点=(%.3f,%.3f)，"
                "当前航向=%.2fdeg，base_link前移=%.3fm，终点=(%.3f,%.3f)"
            ),
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            math.degrees(current_yaw),
            self.base_link_forward_offset,
            goal_x,
            goal_y,
        )
        self.set_state(
            self.MOVE_BASE_OVER_ARROW,
            "等待base_link到达箭头上方、刹停并由mode4接管",
        )
        self.base_tracking_frame_count = 0
        self.last_visual_goal_frame = 0
        self.last_visual_goal_time = None

    def control_move_base_over_arrow(self):
        if not self.fine_detection_available_within(
            self.visual_loss_cancel_seconds, "base_link前移视觉跟踪阶段"
        ):
            full_age = self.full_direction_detection_age()
            if full_age is None or full_age > self.visual_loss_cancel_seconds:
                self.begin_cancel(
                    self.WAIT_FOR_ARROW,
                    (
                        "base_link前移时完整箭头方向丢失{}，超过{:.2f}s阈值，"
                        "立即刹停并重新识别"
                    ).format(
                        "未知" if full_age is None else "{:.2f}s".format(
                            full_age
                        ),
                        self.visual_loss_cancel_seconds,
                    ),
                )
            return
        if not self.direction_locked:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：base_link前移目标保持，等待完整箭头方向重新稳定",
                NODE_NAME,
            )
            return
        self.update_base_tracking_goal()
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：base_link前移：视觉跟踪=%d/%d帧，motion=%s，位置误差=%.3fm",
            NODE_NAME,
            self.base_tracking_frame_count,
            self.center_stable_detection_count,
            self.current_motion_state_name(),
            self.latest_motion_state.position_error,
        )
        if (
            self.base_tracking_frame_count >= self.center_stable_detection_count
            and self.motion_arrived()
        ):
            self.final_hold_stable_started = None
            self.set_state(
                self.FINAL_HOLD,
                "前移目标已进入HOVER，开始最终稳定保持",
            )
            return
        if self.base_tracking_frame_count >= self.center_stable_detection_count:
            self.log_arrival_gate("视觉跟踪已稳定，等待base_link前移目标完成")

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
                    "机器人航向与箭头一致，base_link按标定偏置前移{:.2f}m并稳定在箭头上方".format(
                        self.base_link_forward_offset
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
                "速度{:.3f}/<={:.3f}m/s，yaw_rate{:.2f}/<={:.2f}deg/s，"
                "深度{:.3f}/<={:.3f}m"
            ).format(
                actual_checks["position_error"],
                self.arrival_position_tolerance,
                actual_checks["yaw_error_deg"],
                self.arrival_yaw_tolerance_deg,
                actual_checks["horizontal_speed"],
                self.arrival_max_horizontal_speed,
                actual_checks["yaw_rate_deg_s"],
                self.arrival_max_yaw_rate_deg_s,
                actual_checks["depth_error"],
                self.max_depth_error,
            )
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：%s：反馈新鲜[%s]，state=%s/HOVER[%s]，"
                "目标一致[%s]，目标差值=(%s)，实际到达[%s]，"
                "实际门槛=(%s)，控制器输出=(%d,%d,%d)"
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


if __name__ == "__main__":
    rospy.init_node(NODE_NAME)
    try:
        Task3AcquireAreaTest().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("%s：未处理异常：%s", NODE_NAME, str(error))
        raise
