#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v2_line_follow.py
功能：Task1 巡线单项测试。

流程：
    1. 以节点启动时机器人当前位置为起点，并记录当前 z，不主动改变高度；
    2. 按设定初始航向前进，直到识别到底部红色长线；
    3. 将持续识别到的红线点融合为拟合曲线；
    4. 使用定深定向手控模式沿曲线进行 LOS 巡线；
    5. 红线丢失超过阈值后，认为巡线结束并发布 /finished。

监听：/obj/line_message，/tf
发布：/cmd/pose/ned，/finished

记录：
2026.7.14
    初版，用于单独验证 Task1 巡线通信与控制流程。
"""

import copy
import math

import numpy as np
import rospy
import tf
from auv_control.msg import PoseNEDcmd, TargetDetection3
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task1_v2_line_follow"
MODE_DEPTH_HDG = 3
DEFAULT_INITIAL_HEADING_DEG = 0.0


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(quaternion):
    return euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ])[2]


def xy_distance(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


def class_names(param_name, default):
    value = rospy.get_param(param_name, default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


class Task1LineFollowTest:
    """只测试 Task1 红线搜索和巡线。"""

    def __init__(self):
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))
        self.tf_listener = tf.TransformListener()

        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd/pose/ned")
        self.line_topic = rospy.get_param("~line_topic", "/obj/line_message")
        self.finished_topic = rospy.get_param("~finished_topic", "/finished")

        self.initial_search_yaw = math.radians(float(rospy.get_param(
            "~initial_heading_deg", DEFAULT_INITIAL_HEADING_DEG
        )))
        self.search_forward_force = float(rospy.get_param("~search_forward_force", 120.0))
        self.manual_forward_force = float(rospy.get_param("~manual_forward_force", 260.0))
        self.manual_slow_forward_force = float(rospy.get_param(
            "~manual_slow_forward_force", 120.0
        ))
        self.manual_lateral_gain = float(rospy.get_param("~manual_lateral_gain", 250.0))
        self.manual_max_lateral_force = float(rospy.get_param(
            "~manual_max_lateral_force", 180.0
        ))
        self.manual_force_step = float(rospy.get_param("~manual_force_step", 50.0))
        self.manual_tx_sign = float(rospy.get_param("~manual_tx_sign", 1.0))
        self.manual_ty_sign = float(rospy.get_param("~manual_ty_sign", 1.0))

        self.los_lookahead_distance = float(rospy.get_param("~los_lookahead_distance", 0.6))
        self.manual_slow_yaw_error = math.radians(float(rospy.get_param(
            "~manual_slow_yaw_error_deg", 20.0
        )))
        self.manual_slow_lateral_error = float(rospy.get_param(
            "~manual_slow_lateral_error", 0.25
        ))

        self.line_lost_timeout = float(rospy.get_param("~line_lost_timeout", 5.0))
        self.curve_blind_follow_timeout = float(rospy.get_param(
            "~curve_blind_follow_timeout", 2.0
        ))
        self.line_point_merge_distance = float(rospy.get_param(
            "~line_point_merge_distance", 0.15
        ))
        self.line_curve_max_points = int(rospy.get_param("~line_curve_max_points", 120))
        self.line_curve_sample_count = int(rospy.get_param("~line_curve_sample_count", 80))
        self.line_curve_degree = int(rospy.get_param("~line_curve_degree", 3))
        self.line_curve_min_length = float(rospy.get_param("~line_curve_min_length", 0.4))
        self.line_classes = class_names("~line_classes", ["line"])

        self.cmd_pub = rospy.Publisher(self.cmd_topic, PoseNEDcmd, queue_size=10)
        self.finished_pub = rospy.Publisher(self.finished_topic, String, queue_size=10)
        rospy.Subscriber(self.line_topic, TargetDetection3, self.line_callback)

        self.start_pose = None
        self.hold_z = None
        self.last_line_time = None
        self.last_line_yaw = None
        self.line_started = False
        self.line_axis_origin = None
        self.line_axis_yaw = None
        self.line_raw_points = []
        self.line_curve_points = []
        self.line_curve_s = []
        self.current_path_s = 0.0
        self.completed_path_length = 0.0
        self.last_manual_tx = 0
        self.last_manual_ty = 0

        rospy.loginfo(
            "%s: initialized, initial_heading=%.1fdeg, cmd_topic=%s, line_topic=%s",
            NODE_NAME,
            math.degrees(self.initial_search_yaw),
            self.cmd_topic,
            self.line_topic,
        )

    def get_current_pose(self):
        try:
            self.tf_listener.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(1.0)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(2, "%s: cannot read current pose: %s", NODE_NAME, error)
            return None

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def initialize_start_pose(self):
        if self.start_pose is not None:
            return True

        current = self.get_current_pose()
        if current is None:
            return False

        self.start_pose = copy.deepcopy(current)
        self.hold_z = current.pose.position.z
        rospy.loginfo(
            "%s: start pose recorded x=%.2f, y=%.2f, z=%.2f, current_yaw=%.1fdeg, "
            "search_yaw=%.1fdeg",
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(yaw_from_quaternion(current.pose.orientation)),
            math.degrees(self.initial_search_yaw),
        )
        return True

    def make_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = self.hold_z
        pose.pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw))
        return pose

    @staticmethod
    def force_value(value):
        return int(round(clamp(value, -10000, 10000)))

    def publish_pose_cmd(self, yaw, tx=0, ty=0):
        current = self.get_current_pose()
        if current is None or self.hold_z is None:
            return False

        cmd = PoseNEDcmd()
        cmd.mode = MODE_DEPTH_HDG
        cmd.target = self.make_pose(current.pose.position.x, current.pose.position.y, yaw)
        cmd.force.TX = self.force_value(tx)
        cmd.force.TY = self.force_value(ty)
        self.cmd_pub.publish(cmd)

        rospy.loginfo_throttle(
            1.0,
            "%s: cmd mode=3 yaw=%.1fdeg force=(%d,%d), current=(%.2f, %.2f, %.2f)",
            NODE_NAME,
            math.degrees(yaw),
            cmd.force.TX,
            cmd.force.TY,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
        )
        return True

    def publish_stop(self):
        current = self.get_current_pose()
        yaw = self.initial_search_yaw
        if current is not None:
            yaw = yaw_from_quaternion(current.pose.orientation)
        self.publish_pose_cmd(yaw, 0, 0)
        self.last_manual_tx = 0
        self.last_manual_ty = 0

    def transform_pose_to_map(self, pose):
        try:
            self.tf_listener.waitForTransform(
                "map", pose.header.frame_id, pose.header.stamp, rospy.Duration(1.0)
            )
            return self.tf_listener.transformPose("map", pose)
        except tf.Exception:
            try:
                self.tf_listener.waitForTransform(
                    "map", pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0)
                )
                return self.tf_listener.transformPose("map", pose)
            except tf.Exception as error:
                rospy.logwarn_throttle(2, "%s: line transform failed: %s", NODE_NAME, error)
                return None

    def line_axis_s(self, point):
        if self.line_axis_origin is None or self.line_axis_yaw is None:
            return 0.0
        dx = point.x - self.line_axis_origin.x
        dy = point.y - self.line_axis_origin.y
        return dx * math.cos(self.line_axis_yaw) + dy * math.sin(self.line_axis_yaw)

    def add_line_point(self, point):
        new_point = Point(point.x, point.y, self.hold_z)
        for old_point in self.line_raw_points:
            if xy_distance(old_point, new_point) <= self.line_point_merge_distance:
                old_point.x = 0.8 * old_point.x + 0.2 * new_point.x
                old_point.y = 0.8 * old_point.y + 0.2 * new_point.y
                self.line_raw_points.sort(key=self.line_axis_s)
                return

        self.line_raw_points.append(new_point)
        self.line_raw_points.sort(key=self.line_axis_s)
        if len(self.line_raw_points) > self.line_curve_max_points:
            ordered = sorted(self.line_raw_points, key=self.line_axis_s)
            keep_count = max(2, self.line_curve_max_points)
            indexes = [
                int(round(index * (len(ordered) - 1) / float(keep_count - 1)))
                for index in range(keep_count)
            ]
            self.line_raw_points = [copy.deepcopy(ordered[index]) for index in indexes]

    @staticmethod
    def cumulative_distance(points):
        distances = [0.0]
        for index in range(1, len(points)):
            distances.append(distances[-1] + xy_distance(points[index], points[index - 1]))
        return distances

    def fit_line_curve(self):
        if len(self.line_raw_points) < 2:
            return

        ordered = sorted(self.line_raw_points, key=self.line_axis_s)
        filtered = [ordered[0]]
        for point in ordered[1:]:
            if xy_distance(point, filtered[-1]) > 1e-3:
                filtered.append(point)
        if len(filtered) < 2:
            return

        raw_s = self.cumulative_distance(filtered)
        total_length = raw_s[-1]
        if total_length < self.line_curve_min_length:
            self.line_curve_points = [copy.deepcopy(point) for point in filtered]
            self.line_curve_s = raw_s
            return

        degree = min(max(1, self.line_curve_degree), len(filtered) - 1)
        sample_count = max(2, self.line_curve_sample_count)
        try:
            x_curve = np.poly1d(np.polyfit(raw_s, [point.x for point in filtered], degree))
            y_curve = np.poly1d(np.polyfit(raw_s, [point.y for point in filtered], degree))
            self.line_curve_points = [
                Point(float(x_curve(value)), float(y_curve(value)), self.hold_z)
                for value in np.linspace(0.0, total_length, sample_count)
            ]
            self.line_curve_s = self.cumulative_distance(self.line_curve_points)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            rospy.logwarn_throttle(2, "%s: curve fit failed: %s", NODE_NAME, error)
            self.line_curve_points = [copy.deepcopy(point) for point in filtered]
            self.line_curve_s = raw_s

    def line_curve_ready(self):
        return len(self.line_curve_points) >= 2 and len(self.line_curve_s) == len(
            self.line_curve_points
        )

    def line_callback(self, message):
        if self.hold_z is None and not self.initialize_start_pose():
            return

        if message.class_name and message.class_name not in self.line_classes:
            return

        first = self.transform_pose_to_map(message.pose1)
        second = self.transform_pose_to_map(message.pose2)
        third = self.transform_pose_to_map(message.pose3)
        if first is None or second is None or third is None:
            return

        if self.line_axis_origin is None:
            self.line_axis_origin = copy.deepcopy(first.pose.position)
            self.line_axis_yaw = math.atan2(
                third.pose.position.y - first.pose.position.y,
                third.pose.position.x - first.pose.position.x,
            )
            rospy.loginfo(
                "%s: first red line detected, axis_yaw=%.1fdeg",
                NODE_NAME,
                math.degrees(self.line_axis_yaw),
            )

        self.line_started = True
        self.last_line_time = rospy.Time.now()
        self.last_line_yaw = math.atan2(
            third.pose.position.y - second.pose.position.y,
            third.pose.position.x - second.pose.position.x,
        )
        self.add_line_point(first.pose.position)
        self.add_line_point(second.pose.position)
        self.add_line_point(third.pose.position)
        self.fit_line_curve()

        known_curve = self.line_curve_s[-1] if self.line_curve_ready() else 0.0
        rospy.loginfo_throttle(
            1.0,
            "%s: line update raw=%d curve=%d known_curve=%.2fm line_yaw=%.1fdeg",
            NODE_NAME,
            len(self.line_raw_points),
            len(self.line_curve_points),
            known_curve,
            math.degrees(self.last_line_yaw),
        )

    def line_is_recent(self):
        if self.last_line_time is None:
            return False
        return (rospy.Time.now() - self.last_line_time).to_sec() <= self.line_lost_timeout

    def blind_follow_allowed(self):
        if self.last_line_time is None:
            return False
        return (
            rospy.Time.now() - self.last_line_time
        ).to_sec() <= self.curve_blind_follow_timeout

    def project_to_curve(self, point):
        best = None
        for index in range(len(self.line_curve_points) - 1):
            start = self.line_curve_points[index]
            end = self.line_curve_points[index + 1]
            vx = end.x - start.x
            vy = end.y - start.y
            segment_sq = vx * vx + vy * vy
            if segment_sq < 1e-9:
                continue
            wx = point.x - start.x
            wy = point.y - start.y
            ratio = clamp((wx * vx + wy * vy) / segment_sq, 0.0, 1.0)
            proj_x = start.x + ratio * vx
            proj_y = start.y + ratio * vy
            segment_length = math.sqrt(segment_sq)
            lateral = (vx * (point.y - start.y) - vy * (point.x - start.x)) / segment_length
            path_s = self.line_curve_s[index] + ratio * segment_length
            distance = math.hypot(point.x - proj_x, point.y - proj_y)
            if best is None or distance < best["distance"]:
                best = {
                    "distance": distance,
                    "lateral": lateral,
                    "path_s": path_s,
                }
        return best

    def point_at_curve_s(self, target_s):
        target_s = clamp(target_s, 0.0, self.line_curve_s[-1])
        for index in range(len(self.line_curve_s) - 1):
            start_s = self.line_curve_s[index]
            end_s = self.line_curve_s[index + 1]
            if target_s > end_s:
                continue
            start = self.line_curve_points[index]
            end = self.line_curve_points[index + 1]
            if end_s <= start_s:
                return copy.deepcopy(start)
            ratio = (target_s - start_s) / (end_s - start_s)
            return Point(
                start.x + ratio * (end.x - start.x),
                start.y + ratio * (end.y - start.y),
                self.hold_z,
            )
        return copy.deepcopy(self.line_curve_points[-1])

    def limit_force(self, desired, previous):
        return int(round(clamp(
            desired,
            previous - self.manual_force_step,
            previous + self.manual_force_step,
        )))

    def search_line(self):
        self.publish_pose_cmd(
            self.initial_search_yaw,
            tx=self.manual_tx_sign * self.search_forward_force,
            ty=0,
        )
        rospy.loginfo_throttle(
            1.0,
            "%s: searching red line heading=%.1fdeg force=%.0f",
            NODE_NAME,
            math.degrees(self.initial_search_yaw),
            self.search_forward_force,
        )

    def follow_line(self):
        current = self.get_current_pose()
        if current is None or not self.line_curve_ready():
            self.publish_stop()
            return False

        projection = self.project_to_curve(current.pose.position)
        if projection is None:
            self.publish_stop()
            return False

        self.current_path_s = projection["path_s"]
        self.completed_path_length = max(self.completed_path_length, self.current_path_s)
        los_target = self.point_at_curve_s(self.current_path_s + self.los_lookahead_distance)
        desired_yaw = math.atan2(
            los_target.y - current.pose.position.y,
            los_target.x - current.pose.position.x,
        )
        yaw_error = wrap_angle(desired_yaw - yaw_from_quaternion(current.pose.orientation))
        lateral_error = projection["lateral"]

        forward_force = self.manual_forward_force
        if (
            abs(yaw_error) > self.manual_slow_yaw_error
            or abs(lateral_error) > self.manual_slow_lateral_error
            or not self.line_is_recent()
        ):
            forward_force = self.manual_slow_forward_force

        desired_tx = self.manual_tx_sign * forward_force
        desired_ty = self.manual_ty_sign * clamp(
            -self.manual_lateral_gain * lateral_error,
            -self.manual_max_lateral_force,
            self.manual_max_lateral_force,
        )
        tx = self.limit_force(desired_tx, self.last_manual_tx)
        ty = self.limit_force(desired_ty, self.last_manual_ty)
        self.last_manual_tx = tx
        self.last_manual_ty = ty
        self.publish_pose_cmd(desired_yaw, tx=tx, ty=ty)

        rospy.loginfo_throttle(
            1.0,
            "%s: follow completed=%.2fm known_curve=%.2fm lateral=%.2f "
            "yaw_error=%.1fdeg force=(%d,%d)",
            NODE_NAME,
            self.completed_path_length,
            self.line_curve_s[-1],
            lateral_error,
            math.degrees(yaw_error),
            tx,
            ty,
        )
        return True

    def finish(self):
        self.publish_stop()
        self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
        rospy.loginfo("%s: finished line-only test", NODE_NAME)
        rospy.signal_shutdown("%s complete" % NODE_NAME)

    def run(self):
        while not rospy.is_shutdown():
            if not self.initialize_start_pose():
                self.rate.sleep()
                continue

            if not self.line_started:
                self.search_line()
            elif self.line_curve_ready() and (self.line_is_recent() or self.blind_follow_allowed()):
                self.follow_line()
            else:
                rospy.loginfo("%s: red line lost, line-follow test complete", NODE_NAME)
                self.finish()

            self.rate.sleep()


def main():
    rospy.init_node(NODE_NAME)
    Task1LineFollowTest().run()


if __name__ == "__main__":
    main()
