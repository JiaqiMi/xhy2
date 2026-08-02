#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
名称：test_task3_1_acquire_area.py
功能：识别箭头并通过 motion_supervisor 完成搜索、对准和最终定位
作者：BroXu
监听：视觉识别、/motion/state、/status/auv、/tf
发布：/cmd/motion/goal、/cmd/motion/cancel、任务诊断
记录：
2026.8.2
    将 THRUSTER_RECOVERY 视为有效等待状态，避免自动恢复期间误判任务失败。

说明：本节点只生成 map 绝对目标，不直接发布 /cmd/pose/ned，也不计算 TX、TY、MZ。
"""

from datetime import datetime
import json
import logging
import math
import os
import statistics
import rospy
import tf
from auv_control.msg import AUVData, MotionState, TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task3_1_acquire_area"


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
    SEARCH_POSITION = "固定路径只搜索箭头位置"
    SEARCH_PATTERN = SEARCH_POSITION
    HOLD_WAIT = "锁定当前位姿并等待定点稳定"
    RECOVER_POSITION = "定点重新识别箭头位置"
    WAIT_FOR_ARROW = RECOVER_POSITION
    COARSE_POSITION_APPROACH = "位置窗口锁定后缓慢靠近"
    COLLECT_DIRECTION = "位置靠近中收集完整箭头方向"
    JOINT_POSITION_HEADING_ALIGN = "位置和方向联合闭环对准"
    TRACK_AND_ALIGN = JOINT_POSITION_HEADING_ALIGN
    FINAL_BASE_LINK_APPROACH = "冻结箭头位姿并移动base_link"
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
        MotionState.THRUSTER_RECOVERY: "THRUSTER_RECOVERY",
    }

    def __init__(self):
        self.rate_hz = float(rospy.get_param("~rate", 5.0))
        self.arrow_topic = str(rospy.get_param(
            "~arrow_topic", "/vision/arrow/direction"
        )).strip()
        self.arrow_target_topic = str(rospy.get_param(
            "~arrow_target_topic", "/vision/arrow/target_message"
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
        self.direction_start_confidence = float(rospy.get_param(
            "~direction_start_confidence", 0.50
        ))
        self.detection_timeout = float(rospy.get_param(
            "~detection_timeout", 1.0
        ))
        self.stable_detection_count = int(rospy.get_param(
            "~stable_detection_count", 3
        ))
        self.stable_detection_window_size = int(rospy.get_param(
            "~stable_detection_window_size", 10
        ))
        self.stable_map_position_tolerance_m = float(rospy.get_param(
            "~stable_map_position_tolerance_m", 0.20
        ))
        self.map_alignment_tolerance_m = float(rospy.get_param(
            "~map_alignment_tolerance_m", 0.10
        ))
        self.map_goal_step_ratio = float(rospy.get_param(
            "~map_goal_step_ratio", 0.50
        ))
        self.stable_angle_tolerance_deg = float(rospy.get_param(
            "~stable_angle_tolerance_deg", 12.0
        ))
        self.direction_confirm_window_size = int(rospy.get_param(
            "~direction_confirm_window_size", 10
        ))
        self.direction_confirm_required_count = int(rospy.get_param(
            "~direction_confirm_required_count", 3
        ))
        self.alignment_window_size = int(rospy.get_param(
            "~alignment_window_size", 10
        ))
        self.alignment_required_count = int(rospy.get_param(
            "~alignment_required_count", 3
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
        self.coarse_visual_max_step_m = float(rospy.get_param(
            "~coarse_visual_max_step_m", 0.20
        ))
        self.coarse_visual_min_step_m = float(rospy.get_param(
            "~coarse_visual_min_step_m", 0.05
        ))
        self.coarse_goal_min_interval = float(rospy.get_param(
            "~coarse_goal_min_interval", 1.0
        ))
        self.coarse_position_stop_distance_m = float(rospy.get_param(
            "~coarse_position_stop_distance_m", 0.70
        ))
        self.fine_visual_max_step_m = float(rospy.get_param(
            "~fine_visual_max_step_m", 0.20
        ))
        self.fine_visual_min_step_m = float(rospy.get_param(
            "~fine_visual_min_step_m", 0.05
        ))
        self.fine_yaw_max_step_deg = float(rospy.get_param(
            "~fine_yaw_max_step_deg", 10.0
        ))
        self.fine_goal_min_interval = float(rospy.get_param(
            "~fine_goal_min_interval", 2.0
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
            "~search_initial_forward_distance", 0.40
        ))
        self.search_lateral_distance = float(rospy.get_param(
            "~search_lateral_distance", 0.75
        ))
        self.search_second_forward_distance = float(rospy.get_param(
            "~search_second_forward_distance", 0.65
        ))
        self.search_third_forward_distance = float(rospy.get_param(
            "~search_third_forward_distance", 0.65
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
            "/task3_protection/cancel_recovery_timeout", 30.0
        ))

        self.motion_state_timeout = float(rospy.get_param(
            "/task3_protection/motion_feedback_timeout", 3.0
        ))
        self.motion_startup_timeout = float(rospy.get_param(
            "~motion_startup_timeout", 10.0
        ))
        self.status_timeout = float(rospy.get_param(
            "~status_timeout", 0.5
        ))
        self.fixed_depth_m = float(rospy.get_param(
            "/task3_target_depth_m", 0.60
        ))
        self.fixed_map_z = -self.fixed_depth_m
        self.goal_match_position_tolerance = float(rospy.get_param(
            "~goal_match_position_tolerance", 0.03
        ))
        self.goal_match_depth_tolerance = float(rospy.get_param(
            "~goal_match_depth_tolerance", 0.03
        ))
        self.goal_match_yaw_tolerance_deg = float(rospy.get_param(
            "~goal_match_yaw_tolerance_deg", 2.0
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
        self.task_started = rospy.Time.now()
        self.motion_timeout_started_at = None
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
        self.target_z = None
        self.target_depth = None
        self.initial_hold_x = None
        self.initial_hold_y = None
        self.initial_hold_yaw = None
        self.search_waypoints = []
        self.search_waypoint_index = -1
        self.search_recovery_resume_index = None
        self.first_position_detected = False

        self.model_frame_index = 0
        self.map_target_frame_index = 0
        self.last_model_message_time = None
        self.last_map_target_message_time = None
        self.last_direction_source_key = None
        self.latest_detection = None
        self.latest_map_target = None
        self.locked_arrow_map_x = None
        self.locked_arrow_map_y = None
        self.locked_arrow_received_time = None
        self.detection_samples = []
        self.direction_confirmation_samples = []
        self.alignment_samples = []
        self.arrow_locked = False
        self.direction_locked = False
        self.direction_locked_angle_deg = None
        self.direction_locked_frame_index = None
        self.direction_locked_received_time = None
        self.direction_collection_active = False
        self.last_tracking_input_frames = None
        self.last_visual_goal_time = None
        self.final_arrow_map_x = None
        self.final_arrow_map_y = None
        self.final_target_yaw = None
        self.initial_hover_stable_started = None
        self.final_hold_stable_started = None
        self.hold_requested_at = None
        self.hold_next_state = None

        # 所有运行状态初始化完成后再订阅，避免启动瞬间回调读取未初始化字段。
        self.arrow_sub = rospy.Subscriber(
            self.arrow_topic, String, self.arrow_callback, queue_size=20
        )
        self.arrow_target_sub = rospy.Subscriber(
            self.arrow_target_topic,
            TargetDetection,
            self.arrow_target_callback,
            queue_size=20,
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

        rospy.on_shutdown(self.on_shutdown)
        self.log_startup_config()

    def validate_params(self):
        if self.rate_hz <= 0.0:
            raise ValueError("rate 必须大于0")
        if not math.isfinite(self.fixed_depth_m) or self.fixed_depth_m <= 0.0:
            raise ValueError("task3_target_depth_m必须是大于0的有限数")
        if not all((
            self.arrow_topic,
            self.arrow_target_topic,
            self.motion_goal_topic,
            self.motion_cancel_topic,
            self.motion_state_topic,
            self.status_topic,
        )):
            raise ValueError("任务话题参数不能为空")
        if not (
            0.0 <= self.min_confidence <= 1.0
            and 0.0 <= self.direction_start_confidence <= 1.0
        ):
            raise ValueError("位置和方向置信度必须在0到1之间")
        if min(
            self.stable_detection_count,
            self.stable_detection_window_size,
            self.direction_confirm_window_size,
            self.direction_confirm_required_count,
            self.alignment_window_size,
            self.alignment_required_count,
        ) < 1:
            raise ValueError("识别窗口和确认帧数必须大于等于1")
        if self.stable_detection_count > self.stable_detection_window_size:
            raise ValueError(
                "stable_detection_count 不能大于 stable_detection_window_size"
            )
        if (
            self.direction_confirm_required_count
            > self.direction_confirm_window_size
        ):
            raise ValueError(
                "direction_confirm_required_count 不能大于 "
                "direction_confirm_window_size"
            )
        if self.alignment_required_count > self.alignment_window_size:
            raise ValueError(
                "alignment_required_count 不能大于 alignment_window_size"
            )
        if not 0.0 < self.map_goal_step_ratio <= 1.0:
            raise ValueError("map_goal_step_ratio 必须大于0且不大于1")
        if min(self.image_width, self.image_height) <= 0.0:
            raise ValueError("图像宽度和高度必须大于0")
        if not 0.0 <= self.target_center_u_ratio <= 1.0:
            raise ValueError("target_center_u_ratio 必须在0到1之间")
        if not 0.0 <= self.target_center_v_ratio <= 1.0:
            raise ValueError("target_center_v_ratio 必须在0到1之间")
        if min(
            self.stable_map_position_tolerance_m,
            self.map_alignment_tolerance_m,
            self.stable_angle_tolerance_deg,
            self.full_arrow_edge_margin_px,
            self.full_arrow_min_bbox_width_px,
            self.full_arrow_min_bbox_height_px,
            self.coarse_visual_max_step_m,
            self.coarse_visual_min_step_m,
            self.coarse_goal_min_interval,
            self.coarse_position_stop_distance_m,
            self.fine_visual_max_step_m,
            self.fine_visual_min_step_m,
            self.fine_yaw_max_step_deg,
            self.fine_goal_min_interval,
            self.yaw_tolerance_deg,
            self.initial_hover_seconds,
            self.search_initial_forward_distance,
            self.search_lateral_distance,
            self.search_second_forward_distance,
            self.search_third_forward_distance,
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
            self.min_ground_clearance,
            self.detection_timeout,
            self.ground_clearance_goal_update_threshold,
            self.log_interval,
            self.warning_log_interval,
        ) < 0.0:
            raise ValueError("距离、时间、增益和容差不能小于0")
        if min(
            self.coarse_visual_max_step_m,
            self.coarse_goal_min_interval,
            self.coarse_position_stop_distance_m,
            self.fine_visual_max_step_m,
            self.fine_yaw_max_step_deg,
            self.search_initial_forward_distance,
            self.search_lateral_distance,
            self.search_second_forward_distance,
            self.search_third_forward_distance,
            self.final_hold_timeout,
            self.max_wait_seconds,
            self.cancel_timeout,
            self.motion_state_timeout,
            self.motion_startup_timeout,
            self.status_timeout,
            self.min_ground_clearance,
            self.detection_timeout,
            self.fine_goal_min_interval,
            self.log_interval,
            self.warning_log_interval,
        ) <= 0.0:
            raise ValueError("关键距离、时间和超时参数必须大于0")
        if self.fine_visual_min_step_m > self.fine_visual_max_step_m:
            raise ValueError(
                "fine_visual_min_step_m 不能大于 fine_visual_max_step_m"
            )
        if self.coarse_visual_min_step_m > self.coarse_visual_max_step_m:
            raise ValueError(
                "coarse_visual_min_step_m 不能大于 coarse_visual_max_step_m"
            )
        if self.coarse_position_stop_distance_m <= self.map_alignment_tolerance_m:
            raise ValueError(
                "coarse_position_stop_distance_m 必须大于最终位置容差"
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
        if self.yaw_correction_sign not in (-1.0, 1.0):
            raise ValueError("yaw_correction_sign 必须是1或-1")
        if self.final_hold_timeout < self.final_hold_seconds:
            raise ValueError("final_hold_timeout 不能小于 final_hold_seconds")

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
                "再前%.2fm -> 左右各%.2fm搜索 -> "
                "位置滑动窗%d帧命中%d帧 -> "
                "保持航向按%.2f~%.2fm小步靠近，%.2fm内等待方向 -> "
                "位置置信度>=%.2f后方向窗%d帧命中%d帧 -> "
                "位置和航向联合对准%d帧通过%d帧 -> "
                "冻结base_link最终目标 -> HOVER保持%.1fs"
            ),
            NODE_NAME,
            self.initial_hover_seconds,
            self.search_initial_forward_distance,
            self.search_lateral_distance,
            self.search_second_forward_distance,
            self.search_lateral_distance,
            self.search_third_forward_distance,
            self.search_lateral_distance,
            self.stable_detection_window_size,
            self.stable_detection_count,
            self.coarse_visual_min_step_m,
            self.coarse_visual_max_step_m,
            self.coarse_position_stop_distance_m,
            self.direction_start_confidence,
            self.direction_confirm_window_size,
            self.direction_confirm_required_count,
            self.alignment_window_size,
            self.alignment_required_count,
            self.final_hold_seconds,
        )
        rospy.loginfo(
            (
                "%s：识别：方向话题=%s，三维位置话题=%s，位置最低置信度=%.2f，"
                "方向启用和有效帧置信度=%.2f；"
                "位置窗和方向窗各自数据超时=%.2fs，不要求两个话题时间戳配对；"
                "位置滑动窗=最近%d帧命中%d帧，map二维抖动<=%.3fm；"
                "方向滑动窗候选组=最近%d个唯一推理帧命中%d帧，"
                "角度抖动<=%.1fdeg；"
                "最终误差窗=最近%d帧通过%d帧且最新帧必须通过；"
                "图像=%.0fx%.0f，u/v只记录、不参与平移和完成判定；"
                "完整箭头门槛=距边缘>=%.1fpx且bbox>=%.1fx%.1fpx"
            ),
            NODE_NAME,
            self.arrow_topic,
            self.arrow_target_topic,
            self.min_confidence,
            self.direction_start_confidence,
            self.detection_timeout,
            self.stable_detection_window_size,
            self.stable_detection_count,
            self.stable_map_position_tolerance_m,
            self.direction_confirm_window_size,
            self.direction_confirm_required_count,
            self.stable_angle_tolerance_deg,
            self.alignment_window_size,
            self.alignment_required_count,
            self.image_width,
            self.image_height,
            self.full_arrow_edge_margin_px,
            self.full_arrow_min_bbox_width_px,
            self.full_arrow_min_bbox_height_px,
        )
        rospy.loginfo(
            (
                "%s：base_link位置控制参数：最终误差容差=%.3fm，"
                "粗靠近步长=%.3f~%.3fm、方向观察距离=%.3fm；"
                "联合闭环每次取误差比例=%.2f，步长范围=%.3f~%.3fm，"
                "单次航向<=%.1fdeg，目标间隔>=%.2fs，yaw符号=%+.0f"
            ),
            NODE_NAME,
            self.map_alignment_tolerance_m,
            self.coarse_visual_min_step_m,
            self.coarse_visual_max_step_m,
            self.coarse_position_stop_distance_m,
            self.map_goal_step_ratio,
            self.fine_visual_min_step_m,
            self.fine_visual_max_step_m,
            self.fine_yaw_max_step_deg,
            self.fine_goal_min_interval,
            self.yaw_correction_sign,
        )
        rospy.loginfo(
            (
                "%s：完成条件：联合滑动窗通过后冻结箭头map位姿；"
                "base_link距离<=%.3fm、航向误差<=%.1fdeg，"
                "并且冻结目标对应的motion状态进入HOVER"
            ),
            NODE_NAME,
            self.map_alignment_tolerance_m,
            self.yaw_tolerance_deg,
        )
        rospy.loginfo(
            (
                "%s：运动反馈超时=%.2fs，启动等待=%.1fs，当前位置保持超时=%.1fs；"
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
            "%s：到达判定只读取当前目标对应的新鲜MotionState.HOVER；"
            "位置、航向、速度和角速度门槛由motion_supervisor统一负责",
            NODE_NAME,
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
        # SAFE只作为普通状态记录，不再触发任务失败或阻止反馈就绪。
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
        direction_states = (
            self.COARSE_POSITION_APPROACH,
            self.COLLECT_DIRECTION,
            self.JOINT_POSITION_HEADING_ALIGN,
        )
        direction_waiting = (
            self.state in direction_states
            or (
                self.state == self.HOLD_WAIT
                and self.hold_next_state in direction_states
            )
        )
        if self.direction_collection_active and direction_waiting:
            self.add_direction_confirmation_sample(
                None, frame_index, reason
            )
        if not direction_waiting:
            return
        rospy.loginfo(
            "%s：[箭头帧#%d] 无效：%s，阶段=%s",
            NODE_NAME,
            frame_index,
            reason,
            self.state,
        )

    def reject_map_target_frame(self, frame_index, reason):
        self.latest_map_target = None
        position_states = (
            self.SEARCH_POSITION,
            self.RECOVER_POSITION,
            self.COARSE_POSITION_APPROACH,
            self.COLLECT_DIRECTION,
            self.JOINT_POSITION_HEADING_ALIGN,
        )
        if self.state in position_states:
            self.add_detection_sample(None, frame_index, reason)
        elif (
            self.state == self.HOLD_WAIT
            and self.hold_next_state in position_states
        ):
            self.add_detection_sample(None, frame_index, reason)
        if self.state == self.JOINT_POSITION_HEADING_ALIGN:
            self.add_alignment_sample(frame_index, reason)
        rospy.loginfo(
            "%s：[箭头map帧#%d] 无效：%s，阶段=%s",
            NODE_NAME,
            frame_index,
            reason,
            self.state,
        )

    def transform_arrow_target_to_map(self, message):
        source_frame = str(message.pose.header.frame_id).strip()
        stamp = message.pose.header.stamp
        if not source_frame:
            return None, "三维箭头位置缺少frame_id"
        if stamp == rospy.Time(0):
            return None, "三维箭头位置缺少原始图像时间戳"
        age = (rospy.Time.now() - stamp).to_sec()
        if age < -0.1:
            return None, "三维箭头位置时间戳来自未来"
        if age > self.detection_timeout:
            return None, "三维箭头位置已过期{:.2f}s".format(age)
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
            return None, "转换后的箭头map位置包含无效数值"
        return transformed, ""

    def arrow_target_callback(self, message):
        self.map_target_frame_index += 1
        frame_index = self.map_target_frame_index
        now = rospy.Time.now()
        self.last_map_target_message_time = now

        if self.state in (self.INITIAL_HOVER, self.FINAL_HOLD):
            return
        class_name = str(message.class_name).strip().lower()
        confidence = self.finite_number(message.conf)
        target_type = str(message.type).strip().lower()
        if class_name != "arrow":
            self.reject_map_target_frame(
                frame_index, "三维目标类别{}不是arrow".format(
                    class_name or "空"
                )
            )
            return
        if target_type and target_type != "center":
            self.reject_map_target_frame(
                frame_index, "三维目标类型{}不是center".format(target_type)
            )
            return
        if confidence is None or confidence < self.min_confidence:
            self.reject_map_target_frame(
                frame_index,
                "三维目标置信度{}低于{:.2f}".format(
                    confidence, self.min_confidence
                ),
            )
            return
        transformed, reason = self.transform_arrow_target_to_map(message)
        if transformed is None:
            self.reject_map_target_frame(frame_index, reason)
            return

        source = message.pose.pose.position
        target = transformed.pose.position
        detection = {
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
        self.latest_map_target = detection
        rospy.loginfo(
            (
                "%s：[箭头map帧#%d] 三维位置有效：conf=%.3f，"
                "camera=(%.3f,%.3f,%.3f)，map=(%.3f,%.3f,%.3f)，阶段=%s"
            ),
            NODE_NAME,
            frame_index,
            confidence,
            detection["camera_x"],
            detection["camera_y"],
            detection["camera_z"],
            detection["map_x"],
            detection["map_y"],
            detection["map_z"],
            self.state,
        )

        position_states = (
            self.SEARCH_POSITION,
            self.RECOVER_POSITION,
            self.COARSE_POSITION_APPROACH,
            self.COLLECT_DIRECTION,
            self.JOINT_POSITION_HEADING_ALIGN,
        )
        position_waiting = (
            self.state in position_states
            or (
                self.state == self.HOLD_WAIT
                and self.hold_next_state in position_states
            )
        )
        if self.state == self.SEARCH_POSITION:
            if not self.first_position_detected:
                self.first_position_detected = True
                rospy.logwarn(
                    "%s：[箭头map帧#%d] 搜索中首次获得可转换到map的三维位置，"
                    "搜索移动不中断；本阶段只累计位置滑动窗",
                    NODE_NAME,
                    frame_index,
                )
        if position_waiting:
            self.add_detection_sample(detection, frame_index)
            self.maybe_start_direction_collection(confidence, frame_index)
        if self.state == self.JOINT_POSITION_HEADING_ALIGN:
            self.add_alignment_sample(frame_index)

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


    def direction_source_identity(self, payload):
        if "keypoint_stamp_nsec" in payload:
            stamp_nsec = payload.get("keypoint_stamp_nsec")
            if stamp_nsec is None or not str(stamp_nsec).strip():
                return None, None
            source_key = "nsec:{}".format(str(stamp_nsec).strip())
            source_stamp_sec = self.finite_number(
                payload.get("keypoint_stamp")
            )
            return source_key, source_stamp_sec

        source_stamp_sec = self.finite_number(payload.get("stamp"))
        if source_stamp_sec is None or source_stamp_sec <= 0.0:
            return None, None
        return "sec:{:.9f}".format(source_stamp_sec), source_stamp_sec

    def arrow_callback(self, message):
        now = rospy.Time.now()

        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as error:
            self.model_frame_index += 1
            self.reject_arrow_frame(
                self.model_frame_index, "JSON解析失败：{}".format(error)
            )
            return
        if not isinstance(payload, dict):
            self.model_frame_index += 1
            self.reject_arrow_frame(
                self.model_frame_index, "JSON根节点不是对象"
            )
            return

        source_key, source_stamp_sec = self.direction_source_identity(payload)
        if source_key is None:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：箭头方向消息没有关键点源帧标识，只记录话题存活，"
                "不推进方向滑动窗",
                NODE_NAME,
            )
            return
        if source_key == self.last_direction_source_key:
            rospy.logdebug_throttle(
                self.log_interval,
                "%s：忽略定时器重复发布的箭头关键点源帧%s",
                NODE_NAME,
                source_key,
            )
            return
        self.last_direction_source_key = source_key
        self.last_model_message_time = now
        self.model_frame_index += 1
        frame_index = self.model_frame_index

        if self.state == self.INITIAL_HOVER:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：[箭头唯一推理帧#%d] 启动悬停中，本帧暂不计数",
                NODE_NAME,
                frame_index,
            )
            return
        if self.state == self.FINAL_HOLD:
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
        if (
            confidence is None
            or confidence < self.direction_start_confidence
        ):
            self.reject_arrow_frame(
                frame_index,
                "方向置信度{}低于{:.2f}".format(
                    confidence,
                    self.direction_start_confidence,
                ),
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
        if source_stamp_sec is None or source_stamp_sec <= 0.0:
            self.reject_arrow_frame(frame_index, "箭头方向缺少关键点源时间戳")
            return
        source_age = now.to_sec() - source_stamp_sec
        if source_age < -0.1:
            self.reject_arrow_frame(frame_index, "箭头方向时间戳来自未来")
            return
        if source_age > self.detection_timeout:
            self.reject_arrow_frame(
                frame_index,
                "箭头方向已过期{:.2f}s".format(source_age),
            )
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
            "received_sec": now.to_sec(),
            "source_stamp_sec": source_stamp_sec,
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

        direction_states = (
            self.COARSE_POSITION_APPROACH,
            self.COLLECT_DIRECTION,
            self.JOINT_POSITION_HEADING_ALIGN,
        )
        direction_waiting = (
            self.state in direction_states
            or (
                self.state == self.HOLD_WAIT
                and self.hold_next_state in direction_states
            )
        )
        if self.direction_collection_active and direction_waiting:
            self.add_direction_confirmation_sample(detection, frame_index)
        else:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：[箭头唯一推理帧#%d] 当前阶段只使用位置，方向帧不计数",
                NODE_NAME,
                frame_index,
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
        best_stable_group = None
        for group in candidate_groups:
            stable, map_jitter = self.samples_are_stable(group)
            if not stable:
                continue
            if best_stable_group is None or (
                len(group), group[-1]["frame_index"]
            ) > (
                len(best_stable_group),
                best_stable_group[-1]["frame_index"],
            ):
                best_stable_group = group

        was_locked = self.arrow_locked
        if best_stable_group is None:
            self.arrow_locked = False
            self.locked_arrow_map_x = None
            self.locked_arrow_map_y = None
            self.locked_arrow_received_time = None
            if was_locked:
                rospy.loginfo(
                    "%s：位置滑动窗中的旧候选组已跌出最近%d帧，位置锁定撤销",
                    NODE_NAME,
                    self.stable_detection_window_size,
                )
        else:
            locked = dict(best_stable_group[-1])
            locked["map_x"] = statistics.median(
                item["map_x"] for item in best_stable_group
            )
            locked["map_y"] = statistics.median(
                item["map_y"] for item in best_stable_group
            )
            locked["confidence"] = sum(
                item["confidence"] for item in best_stable_group
            ) / len(best_stable_group)
            self.latest_map_target = locked
            self.locked_arrow_map_x = locked["map_x"]
            self.locked_arrow_map_y = locked["map_y"]
            self.locked_arrow_received_time = locked["received_time"]
            self.arrow_locked = True

        if detection is None:
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] 本帧无效：%s；窗口=%d/%d帧，"
                    "有效位置帧=%d/%d，最佳候选组=%d/%d，当前锁定=%s；"
                    "保留窗口内旧有效帧"
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
                "是" if self.arrow_locked else "否",
            )
            return

        current_group_index = 0
        current_group = [detection]
        for index, group in enumerate(candidate_groups, start=1):
            if any(item is detection for item in group):
                current_group_index = index
                current_group = group
                break

        _, map_jitter = self.samples_are_stable(current_group)
        frame_ids = [item["frame_index"] for item in current_group]
        rospy.loginfo(
            (
                "%s：[箭头map帧#%d] 有效位置加入候选组%d；窗口=%d/%d帧，"
                "有效位置帧=%d/%d，当前候选组=%d/%d，命中帧=%s，"
                "map二维抖动=%.3f/%.3fm"
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
            map_jitter,
            self.stable_map_position_tolerance_m,
        )
        if best_stable_group is None:
            return

        locked_frame_ids = [
            item["frame_index"] for item in best_stable_group
        ]
        rospy.loginfo(
            (
                "%s：位置滑动窗候选组确认通过：最近%d帧内命中%d/%d帧，"
                "命中帧=%s，中位map位置=(%.3f,%.3f)，平均置信度=%.3f；"
                "方向暂不参与判断"
            ),
            NODE_NAME,
            self.stable_detection_window_size,
            len(best_stable_group),
            self.stable_detection_count,
            locked_frame_ids,
            locked["map_x"],
            locked["map_y"],
            locked["confidence"],
        )

    def build_detection_candidate_groups(self, samples):
        groups = []
        for sample in samples:
            matches = []
            for index, group in enumerate(groups):
                median_x, median_y = self.sample_medians(group)
                map_distance = math.hypot(
                    sample["map_x"] - median_x,
                    sample["map_y"] - median_y,
                )
                if map_distance <= self.stable_map_position_tolerance_m:
                    matches.append((map_distance, index))
            if not matches:
                groups.append([sample])
                continue
            _, best_index = min(matches)
            groups[best_index].append(sample)
        return groups

    @staticmethod
    def sample_medians(samples):
        return (
            statistics.median(item["map_x"] for item in samples),
            statistics.median(item["map_y"] for item in samples),
        )

    def samples_are_stable(self, samples):
        if not samples:
            return False, 0.0
        median_x, median_y = self.sample_medians(samples)
        map_jitter = max(
            math.hypot(
                item["map_x"] - median_x,
                item["map_y"] - median_y,
            )
            for item in samples
        )
        stable = (
            len(samples) >= self.stable_detection_count
            and map_jitter <= self.stable_map_position_tolerance_m
        )
        return stable, map_jitter

    def detection_window_progress(self):
        valid_samples = [
            item["detection"]
            for item in self.detection_samples
            if item["detection"] is not None
        ]
        groups = self.build_detection_candidate_groups(valid_samples)
        stable_group_counts = [
            len(group)
            for group in groups
            if self.samples_are_stable(group)[0]
        ]
        return (
            len(self.detection_samples),
            len(valid_samples),
            max(stable_group_counts, default=0),
        )

    def add_direction_confirmation_sample(
        self, detection, frame_index, invalid_reason=""
    ):
        reason = invalid_reason
        if detection is not None and not detection["full_visible"]:
            reason = detection["full_visible_reason"]
            detection = None
        elif detection is not None and detection["angle_deg"] is None:
            reason = "位置有效但方向字段无效"
            detection = None

        self.direction_confirmation_samples.append({
            "frame_index": frame_index,
            "detection": detection,
        })
        self.direction_confirmation_samples = (
            self.direction_confirmation_samples[
                -self.direction_confirm_window_size:
            ]
        )
        valid_samples = [
            item["detection"]
            for item in self.direction_confirmation_samples
            if item["detection"] is not None
        ]
        candidate_groups = self.build_direction_candidate_groups(valid_samples)
        window_count = len(self.direction_confirmation_samples)
        best_group_count = max(
            (len(group) for group in candidate_groups),
            default=0,
        )
        best_stable_group = None
        best_stable_mean = None
        best_stable_jitter = None
        for group in candidate_groups:
            mean_angle = self.mean_angle_deg([
                item["angle_deg"] for item in group
            ])
            angle_jitter = max(
                abs(normalize_angle_deg(item["angle_deg"] - mean_angle))
                for item in group
            )
            if (
                len(group) < self.direction_confirm_required_count
                or angle_jitter > self.stable_angle_tolerance_deg
            ):
                continue
            if best_stable_group is None or (
                len(group), group[-1]["frame_index"]
            ) > (
                len(best_stable_group),
                best_stable_group[-1]["frame_index"],
            ):
                best_stable_group = group
                best_stable_mean = mean_angle
                best_stable_jitter = angle_jitter

        was_locked = self.direction_locked
        if best_stable_group is None:
            self.direction_locked = False
            self.direction_locked_angle_deg = None
            self.direction_locked_frame_index = None
            self.direction_locked_received_time = None
            if was_locked:
                rospy.loginfo(
                    "%s：方向滑动窗中的旧候选组已跌出最近%d帧，方向锁定撤销",
                    NODE_NAME,
                    self.direction_confirm_window_size,
                )
        else:
            self.direction_locked = True
            self.direction_locked_angle_deg = best_stable_mean
            self.direction_locked_frame_index = best_stable_group[-1][
                "frame_index"
            ]
            self.direction_locked_received_time = best_stable_group[-1][
                "received_time"
            ]

        if detection is None:
            rospy.loginfo(
                (
                    "%s：[箭头帧#%d] 方向滑动窗本帧无效：%s；"
                    "窗口=%d/%d帧，有效方向帧=%d/%d，"
                    "最佳角度候选组=%d/%d，当前锁定=%s；"
                    "保留窗口内旧有效帧"
                ),
                NODE_NAME,
                frame_index,
                reason or "没有有效完整箭头方向",
                window_count,
                self.direction_confirm_window_size,
                len(valid_samples),
                window_count,
                best_group_count,
                self.direction_confirm_required_count,
                "是" if self.direction_locked else "否",
            )
            return

        current_group_index = 0
        current_group = [detection]
        for index, group in enumerate(candidate_groups, start=1):
            if any(item is detection for item in group):
                current_group_index = index
                current_group = group
                break

        mean_angle = self.mean_angle_deg([
            item["angle_deg"] for item in current_group
        ])
        angle_jitter = max(
            abs(normalize_angle_deg(item["angle_deg"] - mean_angle))
            for item in current_group
        )
        frame_ids = [item["frame_index"] for item in current_group]
        rospy.loginfo(
            (
                "%s：[箭头帧#%d] 有效方向加入角度候选组%d；"
                "窗口=%d/%d帧，有效方向帧=%d/%d，"
                "当前候选组=%d/%d，命中帧=%s，"
                "平均角度=%.1fdeg，抖动=%.1f/%.1fdeg"
            ),
            NODE_NAME,
            frame_index,
            current_group_index,
            window_count,
            self.direction_confirm_window_size,
            len(valid_samples),
            window_count,
            len(current_group),
            self.direction_confirm_required_count,
            frame_ids,
            mean_angle,
            angle_jitter,
            self.stable_angle_tolerance_deg,
        )
        if best_stable_group is None:
            return

        locked_frame_ids = [
            item["frame_index"] for item in best_stable_group
        ]
        rospy.loginfo(
            (
                "%s：方向滑动窗候选组确认通过：最近%d帧内命中%d/%d帧，"
                "不要求连续，命中帧=%s，平均角度=%.1fdeg，"
                "抖动=%.1f/%.1fdeg"
            ),
            NODE_NAME,
            self.direction_confirm_window_size,
            len(best_stable_group),
            self.direction_confirm_required_count,
            locked_frame_ids,
            best_stable_mean,
            best_stable_jitter,
            self.stable_angle_tolerance_deg,
        )

    def build_direction_candidate_groups(self, samples):
        groups = []
        for sample in samples:
            matches = []
            for index, group in enumerate(groups):
                mean_angle = self.mean_angle_deg([
                    item["angle_deg"] for item in group
                ])
                angle_distance = abs(normalize_angle_deg(
                    sample["angle_deg"] - mean_angle
                ))
                if angle_distance <= self.stable_angle_tolerance_deg:
                    matches.append((angle_distance, index))
            if not matches:
                groups.append([sample])
                continue
            _, best_index = min(matches)
            groups[best_index].append(sample)
        return groups

    def direction_confirmation_window_progress(self):
        valid_samples = [
            item["detection"]
            for item in self.direction_confirmation_samples
            if item["detection"] is not None
        ]
        groups = self.build_direction_candidate_groups(valid_samples)
        stable_group_counts = []
        for group in groups:
            mean_angle = self.mean_angle_deg([
                item["angle_deg"] for item in group
            ])
            angle_jitter = max(
                abs(normalize_angle_deg(item["angle_deg"] - mean_angle))
                for item in group
            )
            if (
                len(group) >= self.direction_confirm_required_count
                and angle_jitter <= self.stable_angle_tolerance_deg
            ):
                stable_group_counts.append(len(group))
        return (
            len(self.direction_confirmation_samples),
            len(valid_samples),
            max(stable_group_counts, default=0),
        )

    def add_alignment_sample(self, frame_index, invalid_reason=""):
        sample = {
            "frame_index": frame_index,
            "passed": False,
            "reason": invalid_reason or "位置窗或方向窗尚未同时通过",
            "map_distance": None,
            "heading_error": None,
        }
        position_ready, direction_ready = self.dual_windows_ready()
        if not invalid_reason and position_ready and direction_ready:
            alignment = self.map_alignment_state(
                "更新闭环对准滑动窗",
                self.latest_map_target,
            )
            heading_error = self.arrow_heading_error_deg()
            sample.update({
                "map_distance": (
                    None if alignment is None else alignment["distance"]
                ),
                "heading_error": heading_error,
            })
            gates = (
                alignment is not None,
                alignment is not None
                and alignment["distance"] <= self.map_alignment_tolerance_m,
                heading_error is not None
                and abs(heading_error) <= self.yaw_tolerance_deg,
            )
            sample["passed"] = all(gates)
            sample["reason"] = "全部误差门槛通过" if sample["passed"] else (
                "map位置或航向仍有误差"
            )

        self.alignment_samples.append(sample)
        self.alignment_samples = self.alignment_samples[
            -self.alignment_window_size:
        ]
        passed_count = sum(
            1 for item in self.alignment_samples if item["passed"]
        )
        rospy.loginfo(
            (
                "%s：闭环对准滑动窗由三维位置帧推进：最新map帧=%s，通过=%s，"
                "map距离=%s/%.3fm，"
                "航向误差=%s/%.1fdeg；窗口通过=%d/%d（需要%d帧）"
            ),
            NODE_NAME,
            "无" if sample["frame_index"] is None else str(
                sample["frame_index"]
            ),
            "是" if sample["passed"] else "否",
            "未知" if sample["map_distance"] is None else "{:.3f}".format(
                sample["map_distance"]
            ),
            self.map_alignment_tolerance_m,
            "未知" if sample["heading_error"] is None else "{:+.1f}".format(
                sample["heading_error"]
            ),
            self.yaw_tolerance_deg,
            passed_count,
            len(self.alignment_samples),
            self.alignment_required_count,
        )

    def alignment_window_progress(self):
        passed_count = sum(
            1 for item in self.alignment_samples if item["passed"]
        )
        latest_passed = bool(
            self.alignment_samples and self.alignment_samples[-1]["passed"]
        )
        return len(self.alignment_samples), passed_count, latest_passed

    def map_alignment_state(self, context, map_target=None):
        target_x = self.locked_arrow_map_x
        target_y = self.locked_arrow_map_y
        if map_target is not None:
            target_x = map_target["map_x"]
            target_y = map_target["map_y"]
        if target_x is None or target_y is None:
            return None
        current = self.get_current_pose(context)
        if current is None:
            return None
        error_x = target_x - current.pose.position.x
        error_y = target_y - current.pose.position.y
        return {
            "current": current,
            "error_x": error_x,
            "error_y": error_y,
            "distance": math.hypot(error_x, error_y),
        }

    def arrow_heading_error_deg(self):
        if not self.direction_locked or self.direction_locked_angle_deg is None:
            return None
        return self.yaw_correction_sign * normalize_angle_deg(
            self.camera_forward_angle_deg - self.direction_locked_angle_deg
        )

    def get_frame_pose(self, frame, context):
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                "map", frame, rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：无法读取map -> %s，%s暂停：%s",
                NODE_NAME,
                frame,
                context,
                str(error),
            )
            return None
        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(value) for value in values):
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：map -> %s含无效值，%s暂停",
                NODE_NAME,
                frame,
                context,
            )
            return None
        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def get_current_pose(self, context):
        return self.get_frame_pose("base_link", context)

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
        self.target_z = self.fixed_map_z
        self.target_depth = status["depth"]
        self.control_initialized = True
        rospy.loginfo(
            "%s：任务统一固定深度=%.3fm，map目标z=%.3f，启动TF z=%.3f",
            NODE_NAME,
            self.fixed_depth_m,
            self.target_z,
            current.pose.position.z,
        )
        self.set_active_goal(
            current.pose.position.x,
            current.pose.position.y,
            self.target_z,
            current_yaw,
            "锁存启动水平位置和航向，并使用任务统一固定深度",
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

    def motion_arrived(self):
        return (
            self.motion_hover_fresh()
            and self.latest_motion_state.startup_complete
            and self.goal_matches_motion_state()
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
        return True

    def set_state(self, state, reason):
        now = rospy.Time.now()
        previous_elapsed = (now - self.state_started).to_sec()
        task_elapsed = (now - self.task_started).to_sec()
        previous = self.state
        self.state = state
        self.state_started = now
        rospy.loginfo(
            (
                "%s：[子任务1阶段] 当前阶段=%s；上一阶段=%s，"
                "上一阶段持续%.1fs，子任务累计%.1fs，进入原因=%s"
            ),
            NODE_NAME,
            state,
            previous,
            previous_elapsed,
            task_elapsed,
            reason,
        )

    def begin_hold(self, next_state, reason):
        current = self.get_current_pose("锁定阶段切换保持位姿")
        if current is None:
            return False
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        self.hold_requested_at = rospy.Time.now()
        self.hold_next_state = next_state
        self.set_active_goal(
            current.pose.position.x,
            current.pose.position.y,
            self.target_z,
            current_yaw,
            "阶段切换时锁定当前实际位姿，不发布cancel",
        )
        rospy.logwarn(
            (
                "%s：不发布%s；改为通过%s锁定当前位姿并等待HOVER；"
                "后续阶段=%s，原因=%s"
            ),
            NODE_NAME,
            self.motion_cancel_topic,
            self.motion_goal_topic,
            next_state,
            reason,
        )
        self.set_state(self.HOLD_WAIT, reason)
        return True

    def hold_has_completed(self):
        if not self.motion_arrived() or self.hold_requested_at is None:
            return False
        return self.latest_motion_state.header.stamp >= self.hold_requested_at

    def reset_first_lock(self):
        self.detection_samples = []
        self.alignment_samples = []
        self.arrow_locked = False
        self.direction_collection_active = False
        self.latest_map_target = None
        self.last_map_target_message_time = None
        self.locked_arrow_map_x = None
        self.locked_arrow_map_y = None
        self.locked_arrow_received_time = None

    def reset_direction_lock(self):
        self.direction_confirmation_samples = []
        self.direction_locked = False
        self.direction_locked_angle_deg = None
        self.direction_locked_frame_index = None
        self.direction_locked_received_time = None

    def maybe_start_direction_collection(self, confidence, frame_index):
        if (
            self.direction_collection_active
            or not self.position_window_ready()
            or confidence < self.direction_start_confidence
        ):
            return False
        self.reset_direction_lock()
        self.direction_collection_active = True
        rospy.logwarn(
            (
                "%s：[箭头map帧#%d] 位置窗口已锁定且置信度%.3f>=%.2f；"
                "从空窗口开始判断完整箭头方向"
            ),
            NODE_NAME,
            frame_index,
            confidence,
            self.direction_start_confidence,
        )
        return True

    def reset_tracking_alignment(self):
        self.alignment_samples = []
        self.last_tracking_input_frames = None

    def locked_direction_age(self):
        if self.direction_locked_received_time is None:
            return None
        return max(
            0.0,
            (
                rospy.Time.now() - self.direction_locked_received_time
            ).to_sec(),
        )

    def detection_center_errors(self, detection):
        desired_u = self.image_width * self.target_center_u_ratio
        desired_v = self.image_height * self.target_center_v_ratio
        error_u = detection["center_u"] - desired_u
        error_v = detection["center_v"] - desired_v
        normalized_u = error_u / max(0.5 * self.image_width, 1.0)
        normalized_v = error_v / max(0.5 * self.image_height, 1.0)
        return error_u, error_v, normalized_u, normalized_v

    def tracking_goal_update_ready(
        self, position_frame_index, direction_frame_index
    ):
        input_frames = (position_frame_index, direction_frame_index)
        if input_frames == self.last_tracking_input_frames:
            return False
        if self.last_visual_goal_time is None:
            return True
        goal_age = (rospy.Time.now() - self.last_visual_goal_time).to_sec()
        if goal_age >= self.fine_goal_min_interval:
            return True
        rospy.loginfo_throttle(
            self.log_interval,
            "%s：闭环目标更新间隔%.2f/%.2fs，本次只更新独立滑动窗",
            NODE_NAME,
            goal_age,
            self.fine_goal_min_interval,
        )
        return False

    def limit_map_step(self, error_x, error_y, min_step, max_step):
        distance = math.hypot(error_x, error_y)
        if distance <= self.map_alignment_tolerance_m or distance <= 1e-9:
            return 0.0, 0.0
        requested = distance * self.map_goal_step_ratio
        step_length = clamp(requested, min_step, max_step)
        scale = step_length / distance
        return error_x * scale, error_y * scale

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
                    self.SEARCH_POSITION,
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
        third_forward = (
            second_forward + self.search_third_forward_distance
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
            (second_forward, 0.0, "第二层回到中线"),
            (third_forward, 0.0, "沿中线再前进{:.2f}m".format(
                self.search_third_forward_distance
            )),
            (third_forward, -lateral, "第三层左移{:.2f}m".format(lateral)),
            (third_forward, lateral, "第三层右移{:.2f}m".format(lateral)),
            (third_forward, 0.0, "第三层回到中线"),
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
                "%s：三段中线优先搜索路径已生成，共%d点；"
                "所有点均相对启动悬停点计算，不会随机器人漂移位置重新累加"
            ),
            NODE_NAME,
            len(self.search_waypoints),
        )

    def activate_search_waypoint(self, index):
        waypoint = self.search_waypoints[index]
        self.search_waypoint_index = index
        self.start_motion_timeout_clock(
            "开始执行搜索路径第{}/{}个运动目标".format(
                index + 1,
                len(self.search_waypoints),
            )
        )
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

    def begin_search_recovery(
        self,
        reason,
        reset_position_window,
        reset_direction_window,
    ):
        """定点复核无进展时回到当前层中轴，再恢复被中断的搜索点。"""
        if not (
            0 <= self.search_waypoint_index < len(self.search_waypoints)
        ):
            self.finish_task(False, "无法确定二级恢复对应的搜索步骤")
            return

        interrupted_index = self.search_waypoint_index
        interrupted = self.search_waypoints[interrupted_index]
        forward = interrupted["forward"]
        cos_yaw = math.cos(self.initial_hold_yaw)
        sin_yaw = math.sin(self.initial_hold_yaw)
        center_x = self.initial_hold_x + cos_yaw * forward
        center_y = self.initial_hold_y + sin_yaw * forward

        self.search_recovery_resume_index = interrupted_index
        if reset_position_window:
            self.reset_first_lock()
            self.first_position_detected = False
        else:
            self.first_position_detected = self.arrow_locked
        if reset_direction_window:
            self.reset_direction_lock()
            self.latest_detection = None
        self.last_tracking_input_frames = None
        self.last_visual_goal_time = None
        self.set_active_goal(
            center_x,
            center_y,
            self.target_z,
            self.initial_hold_yaw,
            (
                "二级恢复：返回搜索步骤{}/{}所在层的中轴，"
                "到达后恢复{}"
            ).format(
                interrupted_index + 1,
                len(self.search_waypoints),
                interrupted["label"],
            ),
        )
        self.set_state(
            self.SEARCH_POSITION,
            "{}；识别回调保持启用，返回中轴途中出现有效数据仍会锁定当前位置复核".format(
                reason
            ),
        )

    def control_search_pattern(self):
        if self.position_window_ready():
            self.last_tracking_input_frames = None
            self.last_visual_goal_time = None
            self.set_state(
                self.COARSE_POSITION_APPROACH,
                "搜索移动期间位置滑动窗已锁定，开始保持航向缓慢靠近",
            )
            return
        if self.first_position_detected:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：已发现箭头粗位置，搜索目标暂不刹停；"
                "本阶段只等待位置窗口稳定，不判断方向",
                NODE_NAME,
            )
        if self.search_recovery_resume_index is not None:
            if not self.motion_arrived():
                rospy.loginfo_throttle(
                    self.log_interval,
                    (
                        "%s：二级恢复返回当前层中轴进行中："
                        "待恢复搜索点=%d/%d，motion=%s，实际位置误差=%.3fm；"
                        "箭头识别持续运行"
                    ),
                    NODE_NAME,
                    self.search_recovery_resume_index + 1,
                    len(self.search_waypoints),
                    self.current_motion_state_name(),
                    self.latest_motion_state.base_position_error,
                )
                return

            resume_index = self.search_recovery_resume_index
            self.search_recovery_resume_index = None
            self.activate_search_waypoint(resume_index)
            rospy.logwarn(
                (
                    "%s：二级恢复已到达当前层中轴；"
                    "恢复搜索步骤%d/%d：%s，识别继续运行"
                ),
                NODE_NAME,
                resume_index + 1,
                len(self.search_waypoints),
                self.search_waypoints[resume_index]["label"],
            )
            return
        if self.motion_arrived():
            next_index = self.search_waypoint_index + 1
            if next_index >= len(self.search_waypoints):
                rospy.logwarn_throttle(
                    self.warning_log_interval,
                    (
                        "%s：三段固定搜索路径已全部完成，保持最后搜索点继续识别；"
                        "不提前结束，等待唯一总超时%.1fs"
                    ),
                    NODE_NAME,
                    self.max_wait_seconds,
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
                "位置窗=%d/%d帧、有效=%d、最佳稳定组=%d/%d；"
                "方向帧本阶段不计数；模型消息年龄=%s"
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

    def control_hold_wait(self):
        elapsed = (rospy.Time.now() - self.state_started).to_sec()
        position_progress = self.detection_window_progress()
        direction_progress = self.direction_confirmation_window_progress()
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：等待当前位置保持目标稳定：motion=%s，速度=%.3fm/s，"
                "输出=(%d,%d,%d)，%.1f/%.1fs；"
                "位置窗=%d/%d、方向窗=%d/%d"
            ),
            NODE_NAME,
            self.current_motion_state_name(),
            self.latest_motion_state.horizontal_speed,
            self.latest_motion_state.tx,
            self.latest_motion_state.ty,
            self.latest_motion_state.mz,
            elapsed,
            self.cancel_timeout,
            position_progress[2],
            self.stable_detection_count,
            direction_progress[2],
            self.direction_confirm_required_count,
        )
        if elapsed >= self.cancel_timeout:
            self.finish_task(False, "当前位置保持目标未在规定时间进入HOVER")
            return
        if not self.hold_has_completed():
            return
        next_state = self.hold_next_state
        self.last_tracking_input_frames = None
        self.last_visual_goal_time = None
        self.set_state(
            next_state,
            "motion_supervisor已完成当前位置保持目标并进入HOVER",
        )
        self.hold_requested_at = None
        self.hold_next_state = None

    def locked_map_target_age(self):
        if self.locked_arrow_received_time is None:
            return None
        return max(
            0.0,
            (rospy.Time.now() - self.locked_arrow_received_time).to_sec(),
        )

    def position_window_ready(self):
        target_age = self.locked_map_target_age()
        return (
            self.arrow_locked
            and self.latest_map_target is not None
            and target_age is not None
            and target_age <= self.detection_timeout
        )

    def direction_window_ready(self):
        direction_age = self.locked_direction_age()
        return (
            self.direction_collection_active
            and self.direction_locked
            and self.direction_locked_angle_deg is not None
            and self.direction_locked_frame_index is not None
            and direction_age is not None
            and direction_age <= self.detection_timeout
        )

    def dual_windows_ready(self):
        return self.position_window_ready(), self.direction_window_ready()

    def control_wait_for_arrow(self):
        target_age = self.locked_map_target_age()
        window_count, valid_count, best_group_count = (
            self.detection_window_progress()
        )
        if self.position_window_ready():
            self.last_tracking_input_frames = None
            self.last_visual_goal_time = None
            self.set_state(
                self.COARSE_POSITION_APPROACH,
                "重新获得稳定位置窗口，恢复位置优先靠近流程",
            )
            return
        state_elapsed = (rospy.Time.now() - self.state_started).to_sec()
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：定点重新获取位置：窗口=%d/%d帧、有效=%d、"
                "最佳稳定组=%d/%d、锁定=%s、年龄=%s；"
                "方向窗口暂不作为恢复条件；motion=%s"
            ),
            NODE_NAME,
            window_count,
            self.stable_detection_window_size,
            valid_count,
            best_group_count,
            self.stable_detection_count,
            "是" if self.position_window_ready() else "否",
            "未收到" if target_age is None else "{:.2f}s".format(target_age),
            self.current_motion_state_name(),
        )
        position_window_failed = (
            window_count >= self.stable_detection_window_size
            and best_group_count < self.stable_detection_count
        )
        position_not_updating = (
            state_elapsed >= self.detection_timeout
            and (
                target_age is None
                or target_age > self.detection_timeout
            )
        )
        if position_window_failed or position_not_updating:
            if position_window_failed:
                reason = "位置滑动窗已满但最佳稳定组仅{}/{}帧".format(
                    best_group_count,
                    self.stable_detection_count,
                )
            else:
                reason = "三维map位置超过{:.2f}s未更新".format(
                    self.detection_timeout
                )
            self.direction_collection_active = False
            self.reset_direction_lock()
            self.begin_search_recovery(
                "位置重新识别未通过：{}".format(reason),
                reset_position_window=True,
                reset_direction_window=True,
            )

    def position_goal_update_ready(self, position_frame_index):
        input_frames = (position_frame_index, None)
        if input_frames == self.last_tracking_input_frames:
            return False
        if self.last_visual_goal_time is None:
            return True
        goal_age = (rospy.Time.now() - self.last_visual_goal_time).to_sec()
        return goal_age >= self.coarse_goal_min_interval

    def update_position_approach_goal(self):
        if not self.position_window_ready():
            return False
        position_frame_index = self.latest_map_target["frame_index"]
        if not self.position_goal_update_ready(position_frame_index):
            return False
        alignment = self.map_alignment_state(
            "位置优先缓慢靠近",
            self.latest_map_target,
        )
        if alignment is None:
            return False
        if alignment["distance"] <= self.coarse_position_stop_distance_m:
            rospy.loginfo_throttle(
                self.log_interval,
                "%s：base_link距箭头%.3fm，已进入方向观察距离%.3fm，"
                "暂停继续靠近并等待完整方向",
                NODE_NAME,
                alignment["distance"],
                self.coarse_position_stop_distance_m,
            )
            return True

        step_x, step_y = self.limit_map_step(
            alignment["error_x"],
            alignment["error_y"],
            self.coarse_visual_min_step_m,
            self.coarse_visual_max_step_m,
        )
        current = alignment["current"]
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        self.last_tracking_input_frames = (position_frame_index, None)
        self.set_active_goal(
            current.pose.position.x + step_x,
            current.pose.position.y + step_y,
            self.target_z,
            current_yaw,
            "只根据稳定位置窗口缓慢靠近箭头，暂不修正方向",
        )
        self.last_visual_goal_time = rospy.Time.now()
        rospy.loginfo(
            (
                "%s：位置优先靠近：base_link误差=(%+.3f,%+.3f)m，"
                "本次小步=(%+.3f,%+.3f)m，保持yaw=%.2fdeg"
            ),
            NODE_NAME,
            alignment["error_x"],
            alignment["error_y"],
            step_x,
            step_y,
            math.degrees(current_yaw),
        )
        return True

    def control_coarse_position_approach(self):
        if not self.position_window_ready():
            self.direction_collection_active = False
            self.reset_direction_lock()
            self.begin_hold(
                self.RECOVER_POSITION,
                "位置优先靠近期间位置窗口失效，先保持当前位置重新获取位置",
            )
            return
        if self.direction_collection_active:
            self.set_state(
                self.COLLECT_DIRECTION,
                "稳定位置帧置信度达到方向启用阈值，开始收集完整箭头方向",
            )
            return
        self.update_position_approach_goal()

    def control_collect_direction(self):
        if not self.position_window_ready():
            self.direction_collection_active = False
            self.reset_direction_lock()
            self.begin_hold(
                self.RECOVER_POSITION,
                "方向收集期间位置窗口失效，先保持当前位置重新获取位置",
            )
            return
        if self.direction_window_ready():
            self.reset_tracking_alignment()
            self.last_tracking_input_frames = None
            self.last_visual_goal_time = None
            self.set_state(
                self.JOINT_POSITION_HEADING_ALIGN,
                "完整箭头方向窗口通过，开始边移动边修正位置和航向",
            )
            return

        self.update_position_approach_goal()
        window_count, valid_count, best_group_count = (
            self.direction_confirmation_window_progress()
        )
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：靠近中收集方向：窗口=%d/%d帧、有效=%d、"
                "最佳稳定组=%d/%d；位置控制继续，方向尚不参与航向目标"
            ),
            NODE_NAME,
            window_count,
            self.direction_confirm_window_size,
            valid_count,
            best_group_count,
            self.direction_confirm_required_count,
        )

    def update_tracking_goal(self):
        position_ready, direction_ready = self.dual_windows_ready()
        if not position_ready or not direction_ready:
            return False
        position_frame_index = self.latest_map_target["frame_index"]
        direction_frame_index = self.direction_locked_frame_index
        if not self.tracking_goal_update_ready(
            position_frame_index,
            direction_frame_index,
        ):
            return False
        alignment = self.map_alignment_state(
            "闭环移动对准",
            self.latest_map_target,
        )
        heading_error = self.arrow_heading_error_deg()
        if alignment is None or heading_error is None:
            return False
        current = alignment["current"]
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        step_x, step_y = self.limit_map_step(
            alignment["error_x"],
            alignment["error_y"],
            self.fine_visual_min_step_m,
            self.fine_visual_max_step_m,
        )
        yaw_step_deg = 0.0
        if abs(heading_error) > self.yaw_tolerance_deg:
            yaw_step_deg = clamp(
                heading_error,
                -self.fine_yaw_max_step_deg,
                self.fine_yaw_max_step_deg,
            )
        goal_x = current.pose.position.x + step_x
        goal_y = current.pose.position.y + step_y
        goal_yaw = normalize_angle_rad(
            current_yaw + math.radians(yaw_step_deg)
        )
        self.last_tracking_input_frames = (
            position_frame_index,
            direction_frame_index,
        )
        if step_x == 0.0 and step_y == 0.0 and yaw_step_deg == 0.0:
            return True
        self.set_active_goal(
            goal_x,
            goal_y,
            self.target_z,
            goal_yaw,
            "根据独立位置窗和方向窗持续更新map平移及航向目标",
        )
        self.last_visual_goal_time = rospy.Time.now()
        rospy.loginfo(
            (
                "%s：滑动窗闭环目标：map误差=(%+.3f,%+.3f)m，"
                "本次map小步=(%+.3f,%+.3f)m；"
                "方向误差/航向小步=(%+.2f/%+.2f)deg；"
                "新目标=(%.3f,%.3f,yaw=%.2fdeg)"
            ),
            NODE_NAME,
            alignment["error_x"],
            alignment["error_y"],
            step_x,
            step_y,
            heading_error,
            yaw_step_deg,
            goal_x,
            goal_y,
            math.degrees(goal_yaw),
        )
        return True

    def lock_final_base_goal(self):
        position_ready, direction_ready = self.dual_windows_ready()
        if not position_ready or not direction_ready:
            return False
        current = self.get_current_pose("冻结base_link最终箭头目标")
        heading_error = self.arrow_heading_error_deg()
        if current is None or heading_error is None:
            return False
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        self.final_arrow_map_x = self.locked_arrow_map_x
        self.final_arrow_map_y = self.locked_arrow_map_y
        self.final_target_yaw = normalize_angle_rad(
            current_yaw + math.radians(heading_error)
        )
        self.set_active_goal(
            self.final_arrow_map_x,
            self.final_arrow_map_y,
            self.target_z,
            self.final_target_yaw,
            "冻结稳定箭头位姿，移动base_link到箭头map位置",
        )
        self.direction_collection_active = False
        rospy.logwarn(
            (
                "%s：最终base_link目标已冻结：map=(%.3f,%.3f,%.3f)，"
                "yaw=%.2fdeg；后续不再使用视觉更新该目标"
            ),
            NODE_NAME,
            self.final_arrow_map_x,
            self.final_arrow_map_y,
            self.target_z,
            math.degrees(self.final_target_yaw),
        )
        return True

    def final_base_goal_errors(self):
        if (
            self.final_arrow_map_x is None
            or self.final_arrow_map_y is None
            or self.final_target_yaw is None
        ):
            return None
        current = self.get_current_pose("检查base_link最终箭头目标")
        if current is None:
            return None
        position_error = math.hypot(
            current.pose.position.x - self.final_arrow_map_x,
            current.pose.position.y - self.final_arrow_map_y,
        )
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        yaw_error_deg = abs(math.degrees(normalize_angle_rad(
            current_yaw - self.final_target_yaw
        )))
        return position_error, yaw_error_deg

    def control_track_and_align(self):
        position_ready, direction_ready = self.dual_windows_ready()
        if not position_ready:
            self.direction_collection_active = False
            self.reset_direction_lock()
            self.begin_hold(
                self.RECOVER_POSITION,
                "联合对准期间位置窗口失效，保持当前位置后重新获取位置",
            )
            return
        if not direction_ready:
            self.begin_hold(
                self.COLLECT_DIRECTION,
                "联合对准期间方向窗口失效，保持当前位置继续收集方向",
            )
            return

        window_count, passed_count, latest_passed = (
            self.alignment_window_progress()
        )
        current_alignment = self.map_alignment_state(
            "检查闭环对准当前误差",
            self.latest_map_target,
        )
        current_heading_error = self.arrow_heading_error_deg()
        current_passed = (
            current_alignment is not None
            and current_alignment["distance"]
            <= self.map_alignment_tolerance_m
            and current_heading_error is not None
            and abs(current_heading_error) <= self.yaw_tolerance_deg
        )
        if (
            latest_passed
            and passed_count >= self.alignment_required_count
            and current_passed
        ):
            if self.lock_final_base_goal():
                self.final_hold_stable_started = None
                self.set_state(
                    self.FINAL_BASE_LINK_APPROACH,
                    (
                        "联合对准窗口最近{}帧内通过{}/{}帧且最新帧通过；"
                        "冻结箭头位姿并移动base_link到最终目标"
                    ).format(
                        self.alignment_window_size,
                        passed_count,
                        self.alignment_required_count,
                    ),
                )
            return

        self.update_tracking_goal()
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：滑动窗闭环移动中：最终对准窗口=%d/%d帧，"
                "通过=%d/%d，最新帧=%s，motion=%s，"
                "当前控制误差=(位置%.3fm,航向%+.2fdeg)"
            ),
            NODE_NAME,
            window_count,
            self.alignment_window_size,
            passed_count,
            self.alignment_required_count,
            "通过" if latest_passed else "未通过",
            self.current_motion_state_name(),
            self.latest_motion_state.base_position_error,
            math.degrees(self.latest_motion_state.yaw_error),
        )

    def control_final_base_link_approach(self):
        errors = self.final_base_goal_errors()
        if errors is None:
            rospy.logwarn_throttle(
                self.warning_log_interval,
                "%s：暂时无法读取最终base_link目标误差，保持冻结目标等待TF恢复",
                NODE_NAME,
            )
            if (
                (rospy.Time.now() - self.state_started).to_sec()
                >= self.final_hold_timeout
            ):
                self.finish_task(False, "无法读取最终base_link目标误差")
            return
        position_error, yaw_error_deg = errors
        arrived = (
            position_error <= self.map_alignment_tolerance_m
            and yaw_error_deg <= self.yaw_tolerance_deg
            and self.motion_arrived()
        )
        rospy.loginfo_throttle(
            self.log_interval,
            (
                "%s：base_link最终靠近：位置误差=%.3f/%.3fm，"
                "航向误差=%.2f/%.2fdeg，motion=%s，到达=%s"
            ),
            NODE_NAME,
            position_error,
            self.map_alignment_tolerance_m,
            yaw_error_deg,
            self.yaw_tolerance_deg,
            self.current_motion_state_name(),
            "是" if arrived else "否",
        )
        if arrived:
            self.final_hold_stable_started = None
            self.set_state(
                self.FINAL_HOLD,
                "base_link已到达冻结箭头目标并进入HOVER，开始最终稳定保持",
            )
            return
        if (rospy.Time.now() - self.state_started).to_sec() >= self.final_hold_timeout:
            self.finish_task(
                False,
                "base_link未在{:.1f}s内到达冻结箭头目标".format(
                    self.final_hold_timeout
                ),
            )


    def control_final_hold(self):
        now = rospy.Time.now()
        errors = self.final_base_goal_errors()
        hover_ok = (
            errors is not None
            and errors[0] <= self.map_alignment_tolerance_m
            and errors[1] <= self.yaw_tolerance_deg
            and self.motion_arrived()
        )
        if hover_ok:
            if self.final_hold_stable_started is None:
                self.final_hold_stable_started = now
                rospy.loginfo(
                    "%s：当前固定目标已由motion_supervisor报告HOVER，"
                    "开始累计%.1fs",
                    NODE_NAME,
                    self.final_hold_seconds,
                )
            stable_elapsed = (
                now - self.final_hold_stable_started
            ).to_sec()
            rospy.loginfo_throttle(
                self.log_interval,
                (
                    "%s：最终保持%.1f/%.1fs；"
                    "当前目标对应的新鲜HOVER[通过]"
                ),
                NODE_NAME,
                stable_elapsed,
                self.final_hold_seconds,
            )
            if stable_elapsed >= self.final_hold_seconds:
                self.finish_task(
                    True,
                    "机器人已稳定到达箭头map位置容差内，且航向与箭头方向一致",
                )
                return
        else:
            if self.final_hold_stable_started is not None:
                rospy.loginfo(
                    "%s：最终稳定条件被打断，保持计时清零",
                    NODE_NAME,
                )
            self.final_hold_stable_started = None
            self.log_arrival_gate("最终保持等待当前目标对应的新鲜HOVER")
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
        startup_complete = bool(message.startup_complete)
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
                "%s：%s：反馈新鲜[%s]，startup_complete[%s]，"
                "state=%s/HOVER[%s]，当前目标一致[%s]，目标差值=(%s)；"
                "控制器诊断=(位置误差%.3fm，航向误差%+.2fdeg，"
                "速度%.3fm/s，yaw_rate%+.2fdeg/s，输出=%d,%d,%d)"
            ),
            NODE_NAME,
            context,
            "通过" if fresh else "未通过",
            "通过" if startup_complete else "未通过",
            self.current_motion_state_name(),
            "通过" if hover else "未通过",
            "通过" if goal_match else "未通过",
            goal_error_text,
            message.base_position_error,
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

    def control_current_state(self):
        if self.state == self.INITIAL_HOVER:
            self.control_initial_hover()
        elif self.state == self.SEARCH_POSITION:
            self.control_search_pattern()
        elif self.state == self.HOLD_WAIT:
            self.control_hold_wait()
        elif self.state == self.RECOVER_POSITION:
            self.control_wait_for_arrow()
        elif self.state == self.COARSE_POSITION_APPROACH:
            self.control_coarse_position_approach()
        elif self.state == self.COLLECT_DIRECTION:
            self.control_collect_direction()
        elif self.state == self.JOINT_POSITION_HEADING_ALIGN:
            self.control_track_and_align()
        elif self.state == self.FINAL_BASE_LINK_APPROACH:
            self.control_final_base_link_approach()
        elif self.state == self.FINAL_HOLD:
            self.control_final_hold()

    def run(self):
        while not rospy.is_shutdown():
            if self.task_finished:
                self.rate.sleep()
                continue
            timeout_elapsed = self.motion_timeout_elapsed()
            if (
                timeout_elapsed is not None
                and timeout_elapsed >= self.max_wait_seconds
            ):
                self.finish_task(
                    False,
                    "机器人开始运动后，搜索和对准累计达到{:.1f}s".format(
                        timeout_elapsed
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

            self.control_current_state()

            if not self.task_finished:
                self.publish_active_goal()
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME)
    configure_task_file_logging("subtask1")
    try:
        Task3AcquireAreaTest().run()
    except rospy.ROSInterruptException:
        pass
    except Exception as error:
        rospy.logfatal("%s：未处理异常：%s", NODE_NAME, str(error))
        raise
