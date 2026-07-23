#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task1 v3 完整任务：连续巡线，并按巡线前向进度处理黄色和黑色标志。

流程：
    1. 复用 test_task1_v3_line_follow.py 完成红线搜索、拟合、LOS 巡线和终点判断；
    2. 巡线期间同时接收黄色/黑色识别，最多保存最近 N 条有效帧；
    3. 任意 K 条有效帧在同一位置聚类后立即把标志投影到红线弧长进度；
    4. 机器人前向进度到达标志进度时原地定点，执行对应灯光/旋转动作；
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
from task1_v3_yaml_logger import TimestampedYamlLogger

from test_task1_v3_line_follow import (
    Task1LineFollowTest,
    class_names,
    wrap_angle,
    xy_distance,
    yaw_from_quaternion,
)


NODE_NAME = "task1_v3"


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
            return None
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
            return None

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
        return confirmed


class Task1V3(Task1LineFollowTest):
    """在 v3 巡线状态机上增加按弧长触发的黄色/黑色动作。"""

    MARKER_ACTION = "MARKER_ACTION"

    def open_data_log(self):
        """覆盖巡线测试的日志名称，为完整任务创建独立数据文件。"""
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
            "~marker_duplicate_distance", 0.40
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
        self.black_return_hold_seconds = max(0.0, float(rospy.get_param(
            "~black_return_hold_seconds", 2.0
        )))

        self.active_marker = None
        self.marker_resume_state = None
        self.marker_action_phase = None
        self.marker_action_hold_goal = None
        self.marker_light_started_at = None
        self.marker_rotation_state = None

        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        rospy.Subscriber(
            self.target_topic,
            TargetDetection,
            self.marker_callback,
            queue_size=10,
        )
        rospy.on_shutdown(self.shutdown_actuators)
        rospy.loginfo(
            "%s: 完整任务启动；黄色要求=%d，黑色要求=%d，"
            "有效识别=%d/%d，样本最长保存=%.1f s",
            NODE_NAME,
            self.required_counts["yellow"],
            self.required_counts["black"],
            marker_required_valid,
            marker_window_size,
            marker_sample_timeout,
        )
        self.write_data_record(
            "task1_configuration",
            handled_counts=copy.deepcopy(self.handled_counts),
            required_counts=copy.deepcopy(self.required_counts),
            marker_window_size=marker_window_size,
            marker_required_valid=marker_required_valid,
            marker_sample_timeout=marker_sample_timeout,
            marker_cluster_distance=marker_cluster_distance,
            marker_duplicate_distance=self.marker_duplicate_distance,
            marker_line_max_distance=self.marker_line_max_distance,
            marker_progress_tolerance=self.marker_progress_tolerance,
            black_rotation_angle_deg=math.degrees(self.black_rotation_angle),
            black_rotation_lookahead_deg=math.degrees(
                self.black_rotation_lookahead
            ),
        )

    def shutdown_actuators(self):
        if hasattr(self, "actuator_pub"):
            self.publish_lights(0, 0)

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
            return self.tf_listener.transformPose(self.map_frame, pose)
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
                return self.tf_listener.transformPose(self.map_frame, latest)
            except tf.Exception as error:
                rospy.logwarn_throttle(
                    2.0, "%s: 图形坐标转换失败: %s", NODE_NAME, error
                )
                return None

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

    def marker_callback(self, message):
        if not self.line_locked:
            return
        kind = self.marker_kind(message)
        if kind is None or self.handled_counts[kind] >= self.required_counts[kind]:
            return
        minimum_confidence = (
            self.yellow_min_confidence
            if kind == "yellow" else self.black_min_confidence
        )
        point = message.pose.pose.position
        valid = (
            (not message.type or message.type == "center")
            and float(message.conf) >= minimum_confidence
            and math.isfinite(point.x)
            and math.isfinite(point.y)
            and math.isfinite(point.z)
            and math.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)
            <= self.max_camera_distance
        )
        if not valid:
            return

        marker = self.transform_marker_to_map(message.pose)
        if marker is None:
            return
        marker.header.stamp = message.pose.header.stamp
        if self.marker_already_known(kind, marker.pose.position):
            return

        now_seconds = rospy.Time.now().to_sec()
        with self.marker_lock:
            confirmed = self.marker_windows[kind].add(now_seconds, marker)
            sample_count = len(self.marker_windows[kind].samples)
            if confirmed is not None:
                self.marker_windows[kind].clear()
        rospy.loginfo_throttle(
            1.0,
            "%s: %s有效识别=%d，位置=(%.2f, %.2f)",
            NODE_NAME,
            kind,
            sample_count,
            marker.pose.position.x,
            marker.pose.position.y,
        )
        if confirmed is not None:
            self.register_marker(kind, confirmed, float(message.conf))

    def register_marker(self, kind, marker, confidence):
        point = marker.pose.position
        projection = self.marker_curve_projection(point)
        with self.marker_lock:
            if any(
                xy_distance(point, known) <= self.marker_duplicate_distance
                for known in self.known_marker_points[kind]
            ):
                return
            self.known_marker_points[kind].append(copy.deepcopy(point))

            projection_matches = (
                projection is not None
                and projection["distance"] <= self.marker_line_max_distance
            )
            path_s = projection["path_s"] if projection_matches else None
            if (
                path_s is not None
                and path_s
                < self.completed_path_length - self.marker_progress_tolerance
            ):
                rospy.loginfo(
                    "%s: 忽略已越过的%s标志；标志进度=%.2f m，"
                    "机器人进度=%.2f m",
                    NODE_NAME,
                    kind,
                    path_s,
                    self.completed_path_length,
                )
                return

            marker_data = {
                "id": self.next_marker_id,
                "kind": kind,
                "pose": copy.deepcopy(marker),
                "confidence": confidence,
                "path_s": path_s,
            }
            self.next_marker_id += 1
            self.pending_markers.append(marker_data)

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
        with self.marker_lock:
            retained = []
            for marker in self.pending_markers:
                kind = marker["kind"]
                if self.handled_counts[kind] >= self.required_counts[kind]:
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
                if marker["path_s"] < (
                    self.completed_path_length - self.marker_progress_tolerance
                ):
                    rospy.loginfo(
                        "%s: 忽略已越过的%s标志 id=%d；%.2f < %.2f m",
                        NODE_NAME,
                        kind,
                        marker["id"],
                        marker["path_s"],
                        self.completed_path_length,
                    )
                    continue
                retained.append(marker)
                if marker["path_s"] <= (
                    self.completed_path_length + self.marker_progress_tolerance
                ):
                    due.append(marker)
            self.pending_markers = retained
            if not due:
                return None
            selected = min(due, key=lambda item: (item["path_s"], item["id"]))
            self.pending_markers.remove(selected)
            return selected

    def start_marker_action(self, marker, current):
        self.active_marker = marker
        self.marker_resume_state = self.state
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        self.marker_action_hold_goal = self.make_pose(
            current.pose.position.x,
            current.pose.position.y,
            current_yaw,
        )
        self.marker_action_phase = "WAIT_HOVER"
        self.marker_light_started_at = None
        self.marker_rotation_state = None
        self.current_tracking_point = copy.deepcopy(marker["pose"].pose.position)
        self.set_state(self.MARKER_ACTION)
        rospy.loginfo(
            "%s: 巡线进度到达%s标志 id=%d；标志进度=%.2f m，"
            "机器人进度=%.2f m，原地定点后执行动作",
            NODE_NAME,
            marker["kind"],
            marker["id"],
            marker["path_s"],
            self.completed_path_length,
        )

    def publish_lights(self, red, green):
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
                self.complete_marker_action()
            else:
                self.marker_action_phase = "ROTATE"
                self.marker_rotation_state = None
            return

        light_on = elapsed - cycle_index * cycle_seconds < self.light_seconds
        self.publish_lights(
            1 if kind == "yellow" and light_on else 0,
            1 if kind == "black" and light_on else 0,
        )

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
                "hover_started": None,
            }

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
            state["hover_started"] = None
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

        if self.motion_arrived():
            if state["hover_started"] is None:
                state["hover_started"] = rospy.Time.now()
        else:
            state["hover_started"] = None
        held_seconds = (
            (rospy.Time.now() - state["hover_started"]).to_sec()
            if state["hover_started"] is not None else 0.0
        )
        return held_seconds >= self.black_return_hold_seconds

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
        elif self.marker_action_phase == "LIGHT":
            self.run_marker_light()
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
        self.marker_light_started_at = None
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

    def finish(self):
        self.publish_lights(0, 0)
        self.cancel_motion()
        self.finished_pub.publish(String(data="task1_v3 finished"))
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
        )
        rospy.signal_shutdown("task1_v3 complete")


def main():
    rospy.init_node(NODE_NAME)
    Task1V3().run()


if __name__ == "__main__":
    main()
