#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task1 比赛完整任务：连续巡线，并按巡线前向进度处理黄色和黑色标志。

流程：
    1. 复用 task1_line_follow.py 完成红线搜索、拟合、LOS 巡线和终点判断；
    2. 巡线期间同时接收黄色/黑色识别，最多保存最近 N 条有效帧；
    3. 任意 K 条有效帧在同一位置聚类后立即把标志投影到红线弧长进度；
    4. 黑色按 base_link、黄色按 hand 前向进度触发；黄色用 hand
       水平投影对准图形，灯光后按绝对深度下潜，再回到任务保持深度；
    5. 动作完成后继续向前巡线，绝不反向寻找已经错过的标志；
    6. 红线任务正常到达终点后结束，并打印实际/要求动作次数。
"""

import copy
import math
import threading

import rospy
import tf
from auv_control.msg import ActuatorControl, TargetDetection
from std_msgs.msg import String
from task1_yaml_logger import TimestampedYamlLogger

from task1_line_follow import (
    Task1LineFollow,
    class_names,
    wrap_angle,
    xy_distance,
    yaw_from_quaternion,
)


NODE_NAME = "task1"


class MarkerObservationWindow:
    """只保存有效识别；达到聚类数量后无需等待窗口装满。"""

    def __init__(self, max_size, required_count, max_age, cluster_distance):
        self.max_size = max(1, int(max_size))
        self.required_count = min(
            self.max_size, max(1, int(required_count))
        )
        self.max_age = max(0.1, float(max_age))
        self.cluster_distance = max(0.0, float(cluster_distance))
        self.samples = []

    def clear(self):
        self.samples = []

    def prune(self, now_seconds):
        self.samples = [
            item for item in self.samples
            if now_seconds - item[0] <= self.max_age
        ]

    def add(self, now_seconds, marker):
        self.prune(now_seconds)
        if marker.header.stamp.to_sec() > 0.0 and any(
            marker.header.stamp == item[1].header.stamp
            for item in self.samples
        ):
            return None, False
        self.samples.append((now_seconds, copy.deepcopy(marker)))
        self.samples = self.samples[-self.max_size:]

        poses = [item[1] for item in self.samples]
        best_cluster = []
        for seed in poses:
            cluster = [
                item for item in poses
                if xy_distance(item.pose.position, seed.pose.position)
                <= self.cluster_distance
            ]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
        if len(best_cluster) < self.required_count:
            return None, True

        confirmed = copy.deepcopy(best_cluster[-1])
        confirmed.pose.position.x = sum(
            item.pose.position.x for item in best_cluster
        ) / len(best_cluster)
        confirmed.pose.position.y = sum(
            item.pose.position.y for item in best_cluster
        ) / len(best_cluster)
        confirmed.pose.position.z = sum(
            item.pose.position.z for item in best_cluster
        ) / len(best_cluster)
        return confirmed, True


class Task1(Task1LineFollow):
    """在巡线状态机上增加按弧长触发的黄色/黑色动作。"""

    MARKER_ACTION = "MARKER_ACTION"

    def open_data_log(self):
        """为完整任务创建独立数据文件。"""
        try:
            self.data_logger = TimestampedYamlLogger(
                NODE_NAME, self.log_directory
            )
            self.data_log_path = self.data_logger.path
            self.write_data_record(
                "startup",
                log_directory=self.log_directory,
                line_min_confidence=self.line_min_confidence,
                los_midpoint_ratio=self.los_midpoint_ratio,
                endpoint_min_completed_path_length=(
                    self.endpoint_min_completed_path_length
                ),
                use_reference_depth=self.use_reference_depth,
                reference_depth=self.reference_depth,
                use_reference_yaw=self.use_reference_yaw,
                reference_yaw_deg=round(
                    math.degrees(self.reference_yaw), 6
                ),
            )
            rospy.loginfo("%s: 完整数据文件=%s", NODE_NAME, self.data_log_path)
        except OSError as error:
            self.data_logger = None
            self.data_log_path = None
            rospy.logwarn("%s: 无法创建完整数据文件: %s", NODE_NAME, error)

    def __init__(self):
        super().__init__()

        self.target_topic = rospy.get_param(
            "~target_topic", "/obj/target_message"
        )
        self.line_detection_ready_topic = rospy.get_param(
            "~line_detection_ready_topic", "/vision/line/detections"
        )
        self.shape_detection_ready_topic = rospy.get_param(
            "~shape_detection_ready_topic", "/vision/shapes/detections"
        )
        self.config_file = str(rospy.get_param("~config_file", ""))
        self.last_line_detection_time = None
        self.last_shape_detection_time = None
        self.actuator_topic = rospy.get_param(
            "~actuator_topic", "/cmd/actuator"
        )
        self.yellow_classes = class_names(
            "~yellow_classes", ["triangle", "circle"]
        )
        self.black_classes = class_names("~black_classes", ["rectangle"])
        self.yellow_min_confidence = float(rospy.get_param(
            "~yellow_min_confidence", 0.30
        ))
        self.black_min_confidence = float(rospy.get_param(
            "~black_min_confidence", 0.30
        ))
        self.max_camera_distance = max(0.0, float(rospy.get_param(
            "~max_camera_distance", 5.0
        )))

        marker_window_size = max(1, int(rospy.get_param(
            "~marker_window_size", 10
        )))
        marker_required_valid = max(1, int(rospy.get_param(
            "~marker_required_valid", 3
        )))
        marker_sample_timeout = max(0.1, float(rospy.get_param(
            "~marker_sample_timeout", 10.0
        )))
        marker_cluster_distance = max(0.0, float(rospy.get_param(
            "~marker_cluster_distance", 0.25
        )))
        self.marker_windows = {
            kind: MarkerObservationWindow(
                marker_window_size,
                marker_required_valid,
                marker_sample_timeout,
                marker_cluster_distance,
            )
            for kind in ("yellow", "black")
        }
        self.marker_duplicate_distance = max(0.0, float(rospy.get_param(
            "~marker_duplicate_distance", 0.60
        )))
        self.marker_line_max_distance = max(0.0, float(rospy.get_param(
            "~marker_line_max_distance", 0.50
        )))
        self.marker_progress_tolerance = max(0.0, float(rospy.get_param(
            "~marker_progress_tolerance", 0.15
        )))

        self.required_counts = {
            "yellow": max(0, int(rospy.get_param(
                "~yellow_required_count", 1
            ))),
            "black": max(0, int(rospy.get_param(
                "~black_required_count", 1
            ))),
        }
        self.handled_counts = {"yellow": 0, "black": 0}
        self.known_marker_points = {"yellow": [], "black": []}
        self.pending_markers = []
        self.handled_markers = []
        self.next_marker_id = 1
        self.marker_lock = threading.Lock()

        self.light_seconds = max(0.0, float(rospy.get_param(
            "~light_seconds", 3.0
        )))
        self.gap_seconds = max(0.0, float(rospy.get_param(
            "~gap_seconds", 0.5
        )))
        self.yellow_light_count = max(1, int(rospy.get_param(
            "~yellow_light_count", 1
        )))
        self.black_light_count = max(1, int(rospy.get_param(
            "~black_light_count", 2
        )))
        self.yellow_contact_enabled = bool(rospy.get_param(
            "~yellow_contact_enabled", False
        ))
        self.yellow_alignment_frame = str(rospy.get_param(
            "~yellow_alignment_frame", "hand"
        )).strip().lstrip("/") or "hand"
        self.yellow_contact_depth = float(rospy.get_param(
            "~yellow_contact_depth", -0.8
        ))
        self.yellow_contact_depth_tolerance = max(0.0, float(
            rospy.get_param("~yellow_contact_depth_tolerance", 0.05)
        ))
        self.yellow_dive_timeout = max(0.1, float(rospy.get_param(
            "~yellow_dive_timeout", 10.0
        )))
        self.light1 = int(rospy.get_param("~light1", 0))
        self.light2 = int(rospy.get_param("~light2", 0))
        self.heading_servo = int(rospy.get_param("~heading_servo", 0x80))
        self.clamp_servo = int(rospy.get_param("~clamp_servo", 0x00))
        self.drive_cmd = int(rospy.get_param("~drive_cmd", 0))
        self.drive_speed = int(rospy.get_param("~drive_speed", 0))

        self.black_rotation_angle = math.radians(abs(float(rospy.get_param(
            "~black_rotation_angle_deg", 360.0
        ))))
        direction = float(rospy.get_param("~black_rotation_direction", 1.0))
        self.black_rotation_direction = 1.0 if direction >= 0.0 else -1.0
        self.black_rotation_lookahead = math.radians(min(
            170.0,
            max(1.0, abs(float(rospy.get_param(
                "~black_rotation_step_deg", 90.0
            )))),
        ))
        self.active_marker = None
        self.marker_resume_state = None
        self.marker_action_phase = None
        self.marker_action_hold_goal = None
        self.marker_action_dive_goal = None
        self.marker_light_started_at = None
        self.yellow_phase_started_at = None
        self.yellow_dive_start_depth = None
        self.marker_rotation_state = None
        self.commanded_lights = {"red": 0, "green": 0}

        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        rospy.Subscriber(
            self.target_topic,
            TargetDetection,
            self.marker_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.line_detection_ready_topic,
            rospy.AnyMsg,
            self.line_detection_ready_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.shape_detection_ready_topic,
            rospy.AnyMsg,
            self.shape_detection_ready_callback,
            queue_size=1,
        )
        rospy.on_shutdown(self.shutdown_actuators)
        rospy.loginfo(
            "%s: 完整任务启动；黄色要求=%d，黑色要求=%d，"
            "有效识别=%d/%d，样本最长保存=%.1f s；"
            "黄色接触=%s，对准坐标系=%s，绝对目标深度=%.2f m",
            NODE_NAME,
            self.required_counts["yellow"],
            self.required_counts["black"],
            marker_required_valid,
            marker_window_size,
            marker_sample_timeout,
            "开启" if self.yellow_contact_enabled else "关闭",
            self.yellow_alignment_frame,
            self.yellow_contact_depth,
        )
        rospy.loginfo(
            "%s: 启动就绪只检查任意消息；line=%s，shape=%s",
            NODE_NAME,
            self.line_detection_ready_topic,
            self.shape_detection_ready_topic,
        )
        self.write_data_record(
            "task1_configuration",
            target_topic=self.target_topic,
            line_detection_ready_topic=self.line_detection_ready_topic,
            shape_detection_ready_topic=self.shape_detection_ready_topic,
            actuator_topic=self.actuator_topic,
            yellow_classes=sorted(self.yellow_classes),
            black_classes=sorted(self.black_classes),
            yellow_min_confidence=self.yellow_min_confidence,
            black_min_confidence=self.black_min_confidence,
            max_camera_distance=self.max_camera_distance,
            handled_counts=copy.deepcopy(self.handled_counts),
            required_counts=copy.deepcopy(self.required_counts),
            marker_window_size=marker_window_size,
            marker_required_valid=marker_required_valid,
            marker_sample_timeout=marker_sample_timeout,
            marker_cluster_distance=marker_cluster_distance,
            marker_duplicate_distance=self.marker_duplicate_distance,
            marker_line_max_distance=self.marker_line_max_distance,
            marker_progress_tolerance=self.marker_progress_tolerance,
            use_reference_depth=self.use_reference_depth,
            reference_depth=self.reference_depth,
            use_reference_yaw=self.use_reference_yaw,
            reference_yaw_deg=math.degrees(self.reference_yaw),
            light_seconds=self.light_seconds,
            gap_seconds=self.gap_seconds,
            yellow_light_count=self.yellow_light_count,
            black_light_count=self.black_light_count,
            yellow_contact_enabled=self.yellow_contact_enabled,
            yellow_alignment_frame=self.yellow_alignment_frame,
            yellow_contact_depth=self.yellow_contact_depth,
            yellow_contact_depth_tolerance=(
                self.yellow_contact_depth_tolerance
            ),
            yellow_dive_timeout=self.yellow_dive_timeout,
            black_rotation_angle_deg=math.degrees(self.black_rotation_angle),
            black_rotation_direction=self.black_rotation_direction,
            black_rotation_lookahead_deg=math.degrees(
                self.black_rotation_lookahead
            ),
            use_known_line_length=self.use_known_line_length,
            known_line_length=self.known_line_length,
            known_line_stop_margin=self.known_line_stop_margin,
        )
        self.write_data_record(
            "parameter_snapshot",
            config_file=self.config_file,
            parameters=copy.deepcopy(rospy.get_param("~", {})),
        )

    def shutdown_actuators(self):
        if hasattr(self, "actuator_pub"):
            self.publish_lights(0, 0)

    def startup_readiness(self):
        return {
            "line_detections": self.last_line_detection_time is not None,
            "shape_detections": self.last_shape_detection_time is not None,
            "motion": self.motion_state_fresh(),
        }

    def line_detection_ready_callback(self, _message):
        if self.last_line_detection_time is None:
            self.write_data_record(
                "detection_node_ready",
                detection="line",
                topic=self.line_detection_ready_topic,
            )
        self.last_line_detection_time = rospy.Time.now()

    def shape_detection_ready_callback(self, _message):
        if self.last_shape_detection_time is None:
            self.write_data_record(
                "detection_node_ready",
                detection="shape",
                topic=self.shape_detection_ready_topic,
            )
        self.last_shape_detection_time = rospy.Time.now()

    def marker_kind(self, message):
        if message.class_name in self.yellow_classes:
            return "yellow"
        if message.class_name in self.black_classes:
            return "black"
        return None

    def transform_marker_to_map(self, pose):
        try:
            self.tf_listener.waitForTransform(
                self.map_frame,
                pose.header.frame_id,
                pose.header.stamp,
                rospy.Duration(self.tf_timeout_seconds),
            )
            return (
                self.tf_listener.transformPose(self.map_frame, pose),
                "image_stamp",
            )
        except tf.Exception:
            try:
                latest = copy.deepcopy(pose)
                latest.header.stamp = rospy.Time(0)
                self.tf_listener.waitForTransform(
                    self.map_frame,
                    latest.header.frame_id,
                    rospy.Time(0),
                    rospy.Duration(self.tf_timeout_seconds),
                )
                return (
                    self.tf_listener.transformPose(self.map_frame, latest),
                    "latest",
                )
            except tf.Exception as error:
                rospy.logwarn_throttle(
                    2.0, "%s: 图形坐标转换失败: %s", NODE_NAME, error
                )
                return None, "unavailable"

    def marker_already_known(self, kind, point):
        with self.marker_lock:
            return any(
                xy_distance(point, known) <= self.marker_duplicate_distance
                for known in self.known_marker_points[kind]
            )

    def marker_curve_projection(self, point):
        with self.curve_lock:
            if self.curve_ready(
                self.line_committed_curve_points,
                self.line_committed_curve_s,
            ):
                points = [
                    copy.deepcopy(item)
                    for item in self.line_committed_curve_points
                ]
                distances = list(self.line_committed_curve_s)
            elif self.curve_ready(self.line_curve_points, self.line_curve_s):
                points = [copy.deepcopy(item) for item in self.line_curve_points]
                distances = list(self.line_curve_s)
            else:
                return None
        return self.project_to_curve(point, points, distances)

    def marker_entry_record(self, marker):
        if marker is None:
            return None
        return {
            "id": marker["id"],
            "kind": marker["kind"],
            "pose": self.pose_record(marker["pose"]),
            "confidence": round(float(marker["confidence"]), 6),
            "path_s": (
                round(float(marker["path_s"]), 6)
                if marker["path_s"] is not None else None
            ),
        }

    def marker_snapshot(self):
        with self.marker_lock:
            windows = {
                kind: {
                    "size": len(window.samples),
                    "samples": [
                        {
                            "arrival_ros_time": round(arrival_time, 6),
                            "image_stamp": round(
                                pose.header.stamp.to_sec(), 6
                            ),
                            "pose": self.pose_record(pose),
                        }
                        for arrival_time, pose in window.samples
                    ],
                }
                for kind, window in self.marker_windows.items()
            }
            return {
                "windows": windows,
                "pending": [
                    self.marker_entry_record(marker)
                    for marker in self.pending_markers
                ],
                "handled": [
                    self.marker_entry_record(marker)
                    for marker in self.handled_markers
                ],
                "known_points": {
                    kind: [
                        self.point_record(point)
                        for point in points
                    ]
                    for kind, points in self.known_marker_points.items()
                },
                "handled_counts": copy.deepcopy(self.handled_counts),
                "required_counts": copy.deepcopy(self.required_counts),
            }

    def record_marker_frame(
        self,
        message,
        status,
        reason,
        kind=None,
        transformed=None,
        transform_mode=None,
        confirmed=None,
        registration=None,
    ):
        with self.marker_lock:
            window_sizes = {
                name: len(window.samples)
                for name, window in self.marker_windows.items()
            }
        self.write_data_record(
            "marker_frame",
            status=status,
            reason=reason,
            marker_kind=kind,
            class_name=message.class_name,
            target_type=message.type,
            confidence=round(float(message.conf), 6),
            source_frame=message.pose.header.frame_id,
            image_stamp=round(message.pose.header.stamp.to_sec(), 6),
            camera_position=self.point_record(message.pose.pose.position),
            map_pose=self.pose_record(transformed),
            transform_mode=transform_mode,
            confirmed_pose=self.pose_record(confirmed),
            registration=registration,
            window_sizes=window_sizes,
            completed_path=round(self.completed_path_length, 6),
            handled_counts=copy.deepcopy(self.handled_counts),
            required_counts=copy.deepcopy(self.required_counts),
        )

    def motion_record(self):
        message = self.latest_motion_state
        if message is None:
            return None
        return {
            "stamp": round(message.header.stamp.to_sec(), 6),
            "state": message.state,
            "reason": message.reason,
            "goal_active": message.goal_active,
            "goal": self.pose_record(message.goal),
            "position_error": round(float(message.position_error), 6),
            "base_position_error": round(
                float(message.base_position_error), 6
            ),
            "yaw_error_deg": round(
                math.degrees(float(message.yaw_error)), 6
            ),
            "horizontal_speed": round(float(message.horizontal_speed), 6),
            "yaw_rate_deg_s": round(
                math.degrees(float(message.yaw_rate)), 6
            ),
            "tx": message.tx,
            "ty": message.ty,
            "mz": message.mz,
            "x_axis_state": message.x_axis_state,
            "y_axis_state": message.y_axis_state,
            "yaw_axis_state": message.yaw_axis_state,
            "x_axis_error": round(float(message.x_axis_error), 6),
            "y_axis_error": round(float(message.y_axis_error), 6),
            "x_axis_speed": round(float(message.x_axis_speed), 6),
            "y_axis_speed": round(float(message.y_axis_speed), 6),
            "startup_complete": message.startup_complete,
        }

    def rotation_record(self):
        state = self.marker_rotation_state
        if state is None:
            return None
        return {
            "start_yaw_deg": round(math.degrees(state["start_yaw"]), 6),
            "last_yaw_deg": round(math.degrees(state["last_yaw"]), 6),
            "completed_deg": round(math.degrees(state["completed"]), 6),
            "goal": self.pose_record(state["goal"]),
            "final_goal_active": state["final_goal_active"],
        }

    def stage_description(self):
        if self.state == self.WAIT_CAMERA:
            if self.startup_hold_started is None:
                return "保持启动位置并等待航向、识别和运动数据就绪"
            return "保持启动位置并执行启动缓冲"
        if (
            self.extension_search_active
            and self.state in self.SEARCH_STATES
        ):
            endpoint_search = {
                self.SEARCH_LEFT: "终点固定位置向左搜索",
                self.SEARCH_RIGHT: "终点固定位置向右搜索",
                self.SEARCH_RETURN: "终点固定位置返回原航向",
            }
            return endpoint_search.get(
                self.state, "终点固定位置搜索红线延伸"
            )
        descriptions = {
            self.SEARCH_LEFT: "向左搜索红线",
            self.SEARCH_RIGHT: "向右搜索红线",
            self.SEARCH_RETURN: "返回启动航向搜索红线",
            self.SEARCH_FORWARD: "向前搜索红线",
            self.WAIT_FIXED_LINE: "等待红线拟合固定",
            self.FOLLOW_LINE: "沿红线进行 LOS 巡航",
            self.HOLD_END: "保持终点并确认结束",
            self.FINISH: "任务结束",
        }
        if self.state != self.MARKER_ACTION:
            return descriptions.get(self.state, "执行任务")
        if self.active_marker is None:
            return "准备执行标志动作"
        kind_name = "黄色" if self.active_marker["kind"] == "yellow" else "黑色"
        action_descriptions = {
            "WAIT_HOVER": "在标志进度处原地定点",
            "LIGHT": "执行%s标志灯光动作" % kind_name,
            "YELLOW_DIVE": "黄色标志保持水平位置并下潜",
            "YELLOW_RETURN": "黄色标志保持水平位置并回到任务深度",
            "ROTATE": "执行黑色标志连续旋转",
        }
        return action_descriptions.get(
            self.marker_action_phase, "执行%s标志动作" % kind_name
        )

    def record_task1_cycle(self):
        current = self.get_current_pose()
        camera = self.get_camera_pose()
        yellow_alignment_pose = (
            self.get_frame_pose(self.yellow_alignment_frame)
            if (
                self.active_marker is not None
                and self.active_marker["kind"] == "yellow"
            )
            else None
        )
        with self.curve_lock:
            line_data = {
                "locked": self.line_locked,
                "status": self.latest_line_status,
                "confidence": round(self.latest_line_confidence, 6),
                "fit_residual": round(self.line_fit_residual, 6),
                "current_path": round(self.current_path_s, 6),
                "projected_path": round(self.projected_path_s, 6),
                "completed_path": round(self.completed_path_length, 6),
                "fitted_curve_length": (
                    round(self.line_curve_s[-1], 6)
                    if self.line_curve_s else 0.0
                ),
                "fixed_curve_length": (
                    round(self.line_committed_curve_s[-1], 6)
                    if self.line_committed_curve_s else 0.0
                ),
                "executing_curve_length": (
                    round(self.tracking_curve_s[-1], 6)
                    if self.tracking_curve_s else 0.0
                ),
                "tracking_curve_version": self.tracking_curve_version,
                "fixed_curve_version": self.line_version,
                "active_los_target_s": (
                    round(self.active_los_target_s, 6)
                    if self.active_los_target_s is not None else None
                ),
                "endpoint_hold_started": (
                    round(self.endpoint_hold_started.to_sec(), 6)
                    if self.endpoint_hold_started is not None else None
                ),
                "extension_search_active": self.extension_search_active,
                "endpoint_progress_ready": self.endpoint_progress_ready(),
                "endpoint_pending_extension": round(
                    self.endpoint_pending_extension(), 6
                ),
                "endpoint_pending_extension_tolerance": round(
                    self.endpoint_pending_extension_tolerance, 6
                ),
                "endpoint_finish_ready": self.endpoint_finish_ready(),
                "use_known_line_length": self.use_known_line_length,
                "known_line_stop_progress": (
                    round(self.known_line_stop_progress(), 6)
                    if self.use_known_line_length else None
                ),
            }
        self.write_data_record(
            "task1_cycle",
            stage_description=self.stage_description(),
            base=self.pose_record(current),
            camera=self.pose_record(camera),
            command_goal=self.pose_record(self.last_motion_goal),
            motion=self.motion_record(),
            line=line_data,
            markers=self.marker_snapshot(),
            active_marker=self.marker_entry_record(self.active_marker),
            marker_action_phase=self.marker_action_phase,
            marker_action_hold_goal=self.pose_record(
                self.marker_action_hold_goal
            ),
            marker_action_dive_goal=self.pose_record(
                self.marker_action_dive_goal
            ),
            rotation=self.rotation_record(),
            commanded_lights=copy.deepcopy(self.commanded_lights),
            startup_readiness=self.startup_readiness(),
            use_reference_depth=self.use_reference_depth,
            reference_depth=self.reference_depth,
            active_depth=self.hold_z,
            yellow_contact_enabled=self.yellow_contact_enabled,
            yellow_alignment_frame=self.yellow_alignment_frame,
            yellow_alignment_pose=self.pose_record(
                yellow_alignment_pose
            ),
            yellow_contact_depth=self.yellow_contact_depth,
            yellow_contact_depth_tolerance=(
                self.yellow_contact_depth_tolerance
            ),
            yellow_dive_timeout=self.yellow_dive_timeout,
            use_reference_yaw=self.use_reference_yaw,
            reference_yaw_deg=round(
                math.degrees(self.reference_yaw), 6
            ),
            active_start_yaw_deg=(
                round(math.degrees(self.search_base_yaw), 6)
                if self.search_base_yaw is not None else None
            ),
        )

    def log_task_summary(self):
        with self.marker_lock:
            pending_count = len(self.pending_markers)
        motion_state = (
            self.latest_motion_state.state
            if self.latest_motion_state is not None else "-"
        )
        marker_text = ""
        if self.active_marker is not None:
            marker_text = "；标志=%s#%d，动作=%s" % (
                self.active_marker["kind"],
                self.active_marker["id"],
                self.marker_action_phase,
            )
        rospy.loginfo_throttle(
            2.0,
            "%s: 当前阶段=%s；正在%s；进度=%.2f m；"
            "黄色=%d/%d，黑色=%d/%d；待处理=%d；MotionState=%s%s",
            NODE_NAME,
            self.state,
            self.stage_description(),
            self.completed_path_length,
            self.handled_counts["yellow"],
            self.required_counts["yellow"],
            self.handled_counts["black"],
            self.required_counts["black"],
            pending_count,
            motion_state,
            marker_text,
        )

    def marker_callback(self, message):
        if not self.line_locked:
            self.record_marker_frame(
                message, "ignored", "line_not_locked"
            )
            return
        kind = self.marker_kind(message)
        if kind is None:
            self.record_marker_frame(
                message, "ignored", "class_not_target"
            )
            return
        if self.handled_counts[kind] >= self.required_counts[kind]:
            self.record_marker_frame(
                message, "ignored", "required_count_reached", kind=kind
            )
            return
        minimum_confidence = (
            self.yellow_min_confidence
            if kind == "yellow" else self.black_min_confidence
        )
        point = message.pose.pose.position
        confidence = float(message.conf)
        if message.type and message.type != "center":
            self.record_marker_frame(
                message, "rejected", "target_type_not_center", kind=kind
            )
            return
        if not math.isfinite(confidence):
            self.record_marker_frame(
                message, "rejected", "confidence_not_finite", kind=kind
            )
            return
        if confidence < minimum_confidence:
            self.record_marker_frame(
                message, "rejected", "confidence_below_minimum", kind=kind
            )
            return
        if not all(
            math.isfinite(value) for value in (point.x, point.y, point.z)
        ):
            self.record_marker_frame(
                message, "rejected", "position_not_finite", kind=kind
            )
            return
        camera_distance = math.sqrt(
            point.x ** 2 + point.y ** 2 + point.z ** 2
        )
        if camera_distance > self.max_camera_distance:
            self.record_marker_frame(
                message, "rejected", "camera_distance_too_large", kind=kind
            )
            return

        marker, transform_mode = self.transform_marker_to_map(message.pose)
        if marker is None:
            self.record_marker_frame(
                message,
                "rejected",
                "transform_unavailable",
                kind=kind,
                transform_mode=transform_mode,
            )
            return
        marker.header.stamp = message.pose.header.stamp
        if self.marker_already_known(kind, marker.pose.position):
            self.record_marker_frame(
                message,
                "ignored",
                "marker_already_known",
                kind=kind,
                transformed=marker,
                transform_mode=transform_mode,
            )
            return

        now_seconds = rospy.Time.now().to_sec()
        with self.marker_lock:
            confirmed, added = self.marker_windows[kind].add(
                now_seconds, marker
            )
            sample_count = len(self.marker_windows[kind].samples)
            if confirmed is not None:
                self.marker_windows[kind].clear()
        if not added:
            self.record_marker_frame(
                message,
                "ignored",
                "duplicate_image_stamp",
                kind=kind,
                transformed=marker,
                transform_mode=transform_mode,
            )
            return
        rospy.loginfo_throttle(
            1.0,
            "%s: %s有效识别=%d，位置=(%.2f, %.2f)",
            NODE_NAME,
            kind,
            sample_count,
            marker.pose.position.x,
            marker.pose.position.y,
        )
        if confirmed is None:
            self.record_marker_frame(
                message,
                "accepted",
                "window_collecting",
                kind=kind,
                transformed=marker,
                transform_mode=transform_mode,
            )
            return

        registration_status, registration = self.register_marker(
            kind, confirmed, confidence
        )
        self.record_marker_frame(
            message,
            "confirmed",
            registration_status,
            kind=kind,
            transformed=marker,
            transform_mode=transform_mode,
            confirmed=confirmed,
            registration=registration,
        )

    def register_marker(self, kind, marker, confidence):
        point = marker.pose.position
        projection = self.marker_curve_projection(point)
        trigger_progress = self.marker_trigger_progress(kind)
        with self.marker_lock:
            if any(
                xy_distance(point, known) <= self.marker_duplicate_distance
                for known in self.known_marker_points[kind]
            ):
                return "duplicate_marker", None
            self.known_marker_points[kind].append(copy.deepcopy(point))

            projection_matches = (
                projection is not None
                and projection["distance"] <= self.marker_line_max_distance
            )
            path_s = projection["path_s"] if projection_matches else None
            if (
                path_s is not None
                and trigger_progress is not None
                and path_s
                < trigger_progress - (
                    0.0
                    if kind == "yellow"
                    else self.marker_progress_tolerance
                )
            ):
                ignored_record = {
                    "kind": kind,
                    "pose": self.pose_record(marker),
                    "confidence": round(confidence, 6),
                    "path_s": round(path_s, 6),
                    "line_distance": round(projection["distance"], 6),
                }
            else:
                ignored_record = None

            if ignored_record is None:
                marker_data = {
                    "id": self.next_marker_id,
                    "kind": kind,
                    "pose": copy.deepcopy(marker),
                    "confidence": confidence,
                    "path_s": path_s,
                }
                self.next_marker_id += 1
                self.pending_markers.append(marker_data)

        if ignored_record is not None:
            rospy.loginfo(
                "%s: 忽略已越过的%s标志；标志进度=%.2f m，"
                "机器人进度=%.2f m",
                NODE_NAME,
                kind,
                path_s,
                self.completed_path_length,
            )
            self.write_data_record(
                "marker_ignored",
                reason="progress_already_passed",
                marker=ignored_record,
                completed_path=round(self.completed_path_length, 6),
                trigger_progress=round(trigger_progress, 6),
                trigger_frame=(
                    self.yellow_alignment_frame
                    if kind == "yellow" else "base_link"
                ),
            )
            return "progress_already_passed", ignored_record

        rospy.loginfo(
            "%s: 标记%s点位 id=%d，位置=(%.2f, %.2f)，巡线进度=%s",
            NODE_NAME,
            kind,
            marker_data["id"],
            point.x,
            point.y,
            "%.2f m" % path_s if path_s is not None else "等待轨迹投影",
        )
        self.write_data_record(
            "marker_registered",
            marker_id=marker_data["id"],
            marker_kind=kind,
            marker=self.pose_record(marker),
            path_s=path_s,
            confidence=confidence,
        )
        return (
            "queued" if path_s is not None else "queued_waiting_curve_match",
            self.marker_entry_record(marker_data),
        )

    def marker_trigger_progress(self, kind):
        """黄色按 hand、其他标志按 base_link 巡线进度触发。"""
        if kind != "yellow":
            return self.completed_path_length
        if not self.tracking_curve_ready():
            return None
        alignment_pose = self.get_frame_pose(self.yellow_alignment_frame)
        if alignment_pose is None:
            return None
        projection = self.project_to_curve(
            alignment_pose.pose.position,
            self.tracking_curve_points,
            self.tracking_curve_s,
        )
        return projection["path_s"] if projection is not None else None

    def update_marker_progress(self):
        if not self.tracking_curve_ready():
            return
        tracking = self.get_tracking_pose()
        if tracking is None:
            return
        projection = self.project_to_curve(
            tracking.pose.position,
            self.tracking_curve_points,
            self.tracking_curve_s,
        )
        if projection is None:
            return
        self.projected_path_s = max(
            self.projected_path_s, self.current_path_s, projection["path_s"]
        )
        self.current_path_s = max(self.current_path_s, self.projected_path_s)
        self.completed_path_length = max(
            self.completed_path_length, self.current_path_s
        )

    def next_due_marker(self):
        due = []
        trigger_progresses = {
            "yellow": self.marker_trigger_progress("yellow"),
            "black": self.completed_path_length,
        }
        with self.marker_lock:
            retained = []
            for marker in self.pending_markers:
                kind = marker["kind"]
                if self.handled_counts[kind] >= self.required_counts[kind]:
                    self.write_data_record(
                        "marker_skipped",
                        reason="required_count_reached",
                        marker=self.marker_entry_record(marker),
                        completed_path=round(
                            self.completed_path_length, 6
                        ),
                    )
                    continue
                projection = self.marker_curve_projection(
                    marker["pose"].pose.position
                )
                if (
                    projection is None
                    or projection["distance"] > self.marker_line_max_distance
                ):
                    marker["path_s"] = None
                    retained.append(marker)
                    continue
                marker["path_s"] = projection["path_s"]
                trigger_progress = trigger_progresses[kind]
                if trigger_progress is None:
                    retained.append(marker)
                    continue
                if marker["path_s"] < (
                    trigger_progress - (
                        0.0
                        if kind == "yellow"
                        else self.marker_progress_tolerance
                    )
                ):
                    rospy.loginfo(
                        "%s: 忽略已越过的%s标志 id=%d；"
                        "标志/触发进度=%.2f/%.2f m（%s）",
                        NODE_NAME,
                        kind,
                        marker["id"],
                        marker["path_s"],
                        trigger_progress,
                        (
                            self.yellow_alignment_frame
                            if kind == "yellow" else "base_link"
                        ),
                    )
                    self.write_data_record(
                        "marker_skipped",
                        reason="progress_already_passed",
                        marker=self.marker_entry_record(marker),
                        completed_path=round(
                            self.completed_path_length, 6
                        ),
                        trigger_progress=round(trigger_progress, 6),
                        trigger_frame=(
                            self.yellow_alignment_frame
                            if kind == "yellow" else "base_link"
                        ),
                    )
                    continue
                retained.append(marker)
                if marker["path_s"] <= (
                    trigger_progress + self.marker_progress_tolerance
                ):
                    due.append(marker)
            self.pending_markers = retained
            if not due:
                return None
            selected = min(due, key=lambda item: (item["path_s"], item["id"]))
            self.pending_markers.remove(selected)
            return selected

    def start_marker_action(self, marker, current):
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        alignment_pose = None
        trigger_progress = self.completed_path_length
        hold_x = current.pose.position.x
        hold_y = current.pose.position.y
        if marker["kind"] == "yellow":
            alignment_pose = self.get_frame_pose(
                self.yellow_alignment_frame
            )
            if alignment_pose is None:
                self.publish_motion_goal(self.make_pose(
                    hold_x, hold_y, current_yaw
                ))
                with self.marker_lock:
                    self.pending_markers.append(marker)
                rospy.logwarn_throttle(
                    2.0,
                    "%s: 黄色标志已到触发进度，但 %s 位姿不可用；"
                    "保持当前位置并等待重试",
                    NODE_NAME,
                    self.yellow_alignment_frame,
                )
                return False
            hold_x += (
                marker["pose"].pose.position.x
                - alignment_pose.pose.position.x
            )
            hold_y += (
                marker["pose"].pose.position.y
                - alignment_pose.pose.position.y
            )
            alignment_projection = self.project_to_curve(
                alignment_pose.pose.position,
                self.tracking_curve_points,
                self.tracking_curve_s,
            )
            trigger_progress = (
                alignment_projection["path_s"]
                if alignment_projection is not None else None
            )

        self.active_marker = marker
        self.marker_resume_state = self.state
        self.marker_action_hold_goal = self.make_pose(
            hold_x, hold_y, current_yaw
        )
        self.marker_action_dive_goal = None
        if marker["kind"] == "yellow" and self.yellow_contact_enabled:
            self.marker_action_dive_goal = copy.deepcopy(
                self.marker_action_hold_goal
            )
            self.marker_action_dive_goal.pose.position.z = (
                self.yellow_contact_depth
            )
        self.marker_action_phase = "WAIT_HOVER"
        self.marker_light_started_at = None
        self.yellow_phase_started_at = None
        self.yellow_dive_start_depth = None
        self.marker_rotation_state = None
        self.current_tracking_point = copy.deepcopy(
            self.marker_action_hold_goal.pose.position
        )
        self.set_state(self.MARKER_ACTION)
        rospy.loginfo(
            "%s: 巡线进度到达%s标志 id=%d；标志/触发进度="
            "%.2f/%.2f m，触发坐标系=%s，定点后执行动作",
            NODE_NAME,
            marker["kind"],
            marker["id"],
            marker["path_s"],
            (
                trigger_progress
                if trigger_progress is not None
                else self.completed_path_length
            ),
            (
                self.yellow_alignment_frame
                if marker["kind"] == "yellow" else "base_link"
            ),
        )
        self.write_data_record(
            "marker_action_start",
            marker=self.marker_entry_record(marker),
            completed_path=round(self.completed_path_length, 6),
            hold_goal=self.pose_record(self.marker_action_hold_goal),
            dive_goal=self.pose_record(self.marker_action_dive_goal),
            trigger_frame=(
                self.yellow_alignment_frame
                if marker["kind"] == "yellow" else "base_link"
            ),
            trigger_progress=(
                round(trigger_progress, 6)
                if trigger_progress is not None else None
            ),
            base_at_trigger=self.pose_record(current),
            alignment_frame_pose_at_trigger=self.pose_record(
                alignment_pose
            ),
            resume_state=self.marker_resume_state,
        )
        return True

    def publish_lights(self, red, green):
        next_lights = {
            "red": int(red),
            "green": int(green),
        }
        lights_changed = next_lights != self.commanded_lights
        self.commanded_lights = next_lights
        if lights_changed:
            self.write_data_record(
                "light_command",
                commanded_lights=copy.deepcopy(self.commanded_lights),
                active_marker=self.marker_entry_record(self.active_marker),
                marker_action_phase=self.marker_action_phase,
            )
        camera_light = ActuatorControl()
        camera_light.mode = 1
        camera_light.light1 = self.light1
        camera_light.light2 = self.light2
        self.actuator_pub.publish(camera_light)

        actuator = ActuatorControl()
        actuator.mode = 2
        actuator.heading_servo = self.heading_servo
        actuator.clamp_servo = self.clamp_servo
        actuator.drive_cmd = self.drive_cmd
        actuator.drive_speed = self.drive_speed
        actuator.red_light = int(red)
        actuator.yellow_light = 0
        actuator.green_light = int(green)
        self.actuator_pub.publish(actuator)

    def run_marker_light(self):
        self.publish_motion_goal(self.marker_action_hold_goal)
        kind = self.active_marker["kind"]
        count = (
            self.yellow_light_count if kind == "yellow"
            else self.black_light_count
        )
        elapsed = (rospy.Time.now() - self.marker_light_started_at).to_sec()
        cycle_seconds = self.light_seconds + self.gap_seconds
        cycle_index = int(elapsed // max(1e-6, cycle_seconds))
        if cycle_index >= count:
            self.publish_lights(0, 0)
            if kind == "yellow":
                if self.yellow_contact_enabled:
                    self.marker_action_phase = "YELLOW_DIVE"
                    self.yellow_phase_started_at = rospy.Time.now()
                    current = self.get_current_pose()
                    self.yellow_dive_start_depth = (
                        current.pose.position.z
                        if current is not None else None
                    )
                    rospy.loginfo(
                        "%s: 黄色标志灯光完成；保持 X/Y/航向，下潜到"
                        "绝对深度 %.2f m；由任务节点判断深度到达",
                        NODE_NAME,
                        self.yellow_contact_depth,
                    )
                    self.write_data_record(
                        "marker_action_phase",
                        marker=self.marker_entry_record(self.active_marker),
                        next_phase=self.marker_action_phase,
                        goal=self.pose_record(self.marker_action_dive_goal),
                        start_depth=self.yellow_dive_start_depth,
                        depth_tolerance=self.yellow_contact_depth_tolerance,
                        timeout=self.yellow_dive_timeout,
                    )
                else:
                    self.complete_marker_action()
            else:
                self.marker_action_phase = "ROTATE"
                self.marker_rotation_state = None
                self.write_data_record(
                    "marker_action_phase",
                    marker=self.marker_entry_record(self.active_marker),
                    next_phase=self.marker_action_phase,
                )
            return

        light_on = elapsed - cycle_index * cycle_seconds < self.light_seconds
        self.publish_lights(
            1 if kind == "yellow" and light_on else 0,
            1 if kind == "black" and light_on else 0,
        )

    def run_yellow_contact(self):
        if self.marker_action_phase == "YELLOW_DIVE":
            goal = self.marker_action_dive_goal
            stage = "下潜"
        else:
            goal = self.marker_action_hold_goal
            stage = "回到任务深度"
        if goal is None:
            return

        self.publish_motion_goal(goal)
        current = self.get_current_pose()
        if current is None:
            return
        current_depth = current.pose.position.z
        depth_error = abs(current_depth - goal.pose.position.z)

        if self.marker_action_phase == "YELLOW_DIVE":
            if self.yellow_phase_started_at is None:
                self.yellow_phase_started_at = rospy.Time.now()
            elapsed = (
                rospy.Time.now() - self.yellow_phase_started_at
            ).to_sec()
            start_depth = self.yellow_dive_start_depth
            if start_depth is None:
                depth_reached = (
                    depth_error <= self.yellow_contact_depth_tolerance
                )
            elif goal.pose.position.z >= start_depth:
                depth_reached = current_depth >= (
                    goal.pose.position.z
                    - self.yellow_contact_depth_tolerance
                )
            else:
                depth_reached = current_depth <= (
                    goal.pose.position.z
                    + self.yellow_contact_depth_tolerance
                )
            timed_out = elapsed >= self.yellow_dive_timeout
            rospy.loginfo_throttle(
                2.0,
                "%s: 黄色接触下潜；当前/目标深度=%.2f/%.2f m，"
                "误差=%.2f m，计时=%.1f/%.1f s",
                NODE_NAME,
                current_depth,
                goal.pose.position.z,
                depth_error,
                elapsed,
                self.yellow_dive_timeout,
            )
            if not depth_reached and not timed_out:
                return

            completion_reason = (
                "depth_reached" if depth_reached else "timeout"
            )
            self.marker_action_phase = "YELLOW_RETURN"
            self.yellow_phase_started_at = rospy.Time.now()
            rospy.loginfo(
                "%s: 黄色接触下潜阶段结束（%s）；实际/目标深度="
                "%.2f/%.2f m，回到任务深度 %.2f m",
                NODE_NAME,
                completion_reason,
                current_depth,
                goal.pose.position.z,
                self.marker_action_hold_goal.pose.position.z,
            )
            self.write_data_record(
                "marker_action_phase",
                marker=self.marker_entry_record(self.active_marker),
                next_phase=self.marker_action_phase,
                completion_reason=completion_reason,
                actual_depth=round(current_depth, 6),
                target_depth=round(goal.pose.position.z, 6),
                depth_error=round(depth_error, 6),
                elapsed=round(elapsed, 6),
                goal=self.pose_record(self.marker_action_hold_goal),
            )
            return

        rospy.loginfo_throttle(
            2.0,
            "%s: 黄色接触%s；当前/目标深度=%.2f/%.2f m，等待 HOVER",
            NODE_NAME,
            stage,
            current_depth,
            goal.pose.position.z,
        )
        if not self.motion_arrived():
            return

        rospy.loginfo(
            "%s: 黄色接触已回到任务深度 %.2f m",
            NODE_NAME,
            self.marker_action_hold_goal.pose.position.z,
        )
        self.complete_marker_action()

    def run_black_rotation(self):
        current = self.get_current_pose()
        if current is None:
            return False
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        if self.marker_rotation_state is None:
            self.marker_rotation_state = {
                "start_yaw": current_yaw,
                "last_yaw": current_yaw,
                "completed": 0.0,
                "goal": None,
                "final_goal_active": False,
            }
            self.write_data_record(
                "black_rotation_start",
                marker=self.marker_entry_record(self.active_marker),
                start_yaw_deg=round(math.degrees(current_yaw), 6),
                target_rotation_deg=round(
                    math.degrees(self.black_rotation_angle), 6
                ),
                lookahead_deg=round(
                    math.degrees(self.black_rotation_lookahead), 6
                ),
                direction=self.black_rotation_direction,
            )

        state = self.marker_rotation_state
        yaw_delta = wrap_angle(current_yaw - state["last_yaw"])
        state["completed"] = max(
            0.0,
            state["completed"] + self.black_rotation_direction * yaw_delta,
        )
        state["last_yaw"] = current_yaw
        final_phase_start = max(
            0.0, self.black_rotation_angle - self.black_rotation_lookahead
        )
        anchor = self.marker_action_hold_goal.pose.position

        if (
            not state["final_goal_active"]
            and state["completed"] >= final_phase_start - 1e-6
        ):
            final_yaw = wrap_angle(
                state["start_yaw"]
                + self.black_rotation_direction * self.black_rotation_angle
            )
            state["goal"] = self.make_pose(anchor.x, anchor.y, final_yaw)
            state["final_goal_active"] = True
            self.write_data_record(
                "black_rotation_final_goal",
                marker=self.marker_entry_record(self.active_marker),
                completed_deg=round(
                    math.degrees(state["completed"]), 6
                ),
                goal=self.pose_record(state["goal"]),
            )
        elif not state["final_goal_active"]:
            target_yaw = wrap_angle(
                current_yaw
                + self.black_rotation_direction * self.black_rotation_lookahead
            )
            state["goal"] = self.make_pose(anchor.x, anchor.y, target_yaw)

        self.publish_motion_goal(state["goal"])
        if not state["final_goal_active"]:
            rospy.loginfo_throttle(
                2.0,
                "%s: 黑色动作连续旋转=%.1f/%.1f deg，动态超前=%.1f deg",
                NODE_NAME,
                math.degrees(state["completed"]),
                math.degrees(self.black_rotation_angle),
                math.degrees(self.black_rotation_lookahead),
            )
            return False

        return self.motion_arrived()

    def run_active_marker_action(self):
        if self.active_marker is None:
            return
        if self.marker_action_phase == "WAIT_HOVER":
            self.publish_motion_goal(self.marker_action_hold_goal)
            if self.motion_arrived():
                self.marker_action_phase = "LIGHT"
                self.marker_light_started_at = rospy.Time.now()
                rospy.loginfo(
                    "%s: %s标志定点完成，开始灯光动作",
                    NODE_NAME,
                    self.active_marker["kind"],
                )
                self.write_data_record(
                    "marker_action_phase",
                    marker=self.marker_entry_record(self.active_marker),
                    next_phase=self.marker_action_phase,
                    light_started_at=round(
                        self.marker_light_started_at.to_sec(), 6
                    ),
                )
        elif self.marker_action_phase == "LIGHT":
            self.run_marker_light()
        elif self.marker_action_phase in ("YELLOW_DIVE", "YELLOW_RETURN"):
            self.run_yellow_contact()
        elif self.marker_action_phase == "ROTATE":
            if self.run_black_rotation():
                self.complete_marker_action()

    def complete_marker_action(self):
        marker = self.active_marker
        if marker is None:
            return
        kind = marker["kind"]
        self.publish_lights(0, 0)
        self.handled_counts[kind] += 1
        self.handled_markers.append(copy.deepcopy(marker))
        rospy.loginfo(
            "%s: %s标志动作完成；黄色=%d/%d，黑色=%d/%d",
            NODE_NAME,
            kind,
            self.handled_counts["yellow"],
            self.required_counts["yellow"],
            self.handled_counts["black"],
            self.required_counts["black"],
        )
        self.write_data_record(
            "marker_action_complete",
            marker_id=marker["id"],
            marker_kind=kind,
            marker=self.pose_record(marker["pose"]),
            path_s=marker["path_s"],
            handled_counts=copy.deepcopy(self.handled_counts),
            required_counts=copy.deepcopy(self.required_counts),
        )

        resume_state = self.marker_resume_state
        self.active_marker = None
        self.marker_resume_state = None
        self.marker_action_phase = None
        self.marker_action_hold_goal = None
        self.marker_action_dive_goal = None
        self.marker_light_started_at = None
        self.yellow_phase_started_at = None
        self.yellow_dive_start_depth = None
        self.marker_rotation_state = None
        if resume_state == self.FOLLOW_LINE:
            self.last_los_goal = None
            self.hold_target = None
            self.clear_active_los_target()
        self.set_state(
            resume_state
            if resume_state in (self.FOLLOW_LINE, self.HOLD_END)
            else self.FOLLOW_LINE
        )

    def run_task_override_cycle(self):
        now_seconds = rospy.Time.now().to_sec()
        with self.marker_lock:
            for window in self.marker_windows.values():
                window.prune(now_seconds)

        if self.active_marker is not None:
            self.run_active_marker_action()
            return True
        if self.state not in (self.FOLLOW_LINE, self.HOLD_END):
            return False

        self.update_marker_progress()
        current = self.get_current_pose()
        if current is None:
            return False
        marker = self.next_due_marker()
        if marker is None:
            rospy.loginfo_throttle(
                5.0,
                "%s: 巡线动作计数 黄色=%d/%d，黑色=%d/%d，待处理=%d",
                NODE_NAME,
                self.handled_counts["yellow"],
                self.required_counts["yellow"],
                self.handled_counts["black"],
                self.required_counts["black"],
                len(self.pending_markers),
            )
            return False
        self.start_marker_action(marker, current)
        return True

    def after_control_cycle(self):
        self.record_task1_cycle()
        self.log_task_summary()

    def finish(self):
        self.publish_lights(0, 0)
        self.cancel_motion()
        self.finished_pub.publish(String(data="task1 finished"))
        rospy.loginfo(
            "%s: FINISH；巡线完成，黄色动作=%d/%d，黑色动作=%d/%d；"
            "次数不足时也不反向巡航",
            NODE_NAME,
            self.handled_counts["yellow"],
            self.required_counts["yellow"],
            self.handled_counts["black"],
            self.required_counts["black"],
        )
        self.write_data_record(
            "task1_finish",
            completed_path=round(self.completed_path_length, 6),
            handled_counts=copy.deepcopy(self.handled_counts),
            required_counts=copy.deepcopy(self.required_counts),
            pending_marker_count=len(self.pending_markers),
            markers=self.marker_snapshot(),
            active_marker=self.marker_entry_record(self.active_marker),
            marker_action_phase=self.marker_action_phase,
            commanded_lights=copy.deepcopy(self.commanded_lights),
            final_base=self.pose_record(self.get_current_pose()),
            final_motion=self.motion_record(),
            endpoint_progress_ready=self.endpoint_progress_ready(),
            use_known_line_length=self.use_known_line_length,
            known_line_stop_progress=(
                round(self.known_line_stop_progress(), 6)
                if self.use_known_line_length else None
            ),
        )
        rospy.signal_shutdown("task1 complete")


def main():
    rospy.init_node(NODE_NAME)
    Task1().run()


if __name__ == "__main__":
    main()
