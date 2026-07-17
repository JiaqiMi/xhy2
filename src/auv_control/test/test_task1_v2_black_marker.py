#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v2_black_marker.py
功能：Task1 黑色方形单项测试。

流程：
    1. 定点保持启动位姿，等待相机节点持续发布图像；
    2. 未识别到黑色方形时，定点向启动航向前进，静止后使用 TY 左右横移搜索；
    3. 同一 rectangle 连续 3 帧且每帧置信度不低于 0.30 后确认位置；
    4. 使用动力定位模式直接前往黑色方形中心，只以 XY 距离判断到达；
    5. 到达后亮绿灯，并以 MZ 和 TF 航向累计完成默认两圈旋转；
    6. 动作完成后发布完成消息。

监听：/obj/target_message，/left/image_raw，/status/vel（可选），/tf
发布：/cmd/pose/ned，/cmd/actuator，/finished

记录：
2026.7.14
    初版，用于单独验证 Task1 黑色方形识别、前往图形、亮灯和旋转流程。
2026.7.17
    同步黄色图形的相机等待、定点前进、TY 横移搜索、定点靠近和精简日志流程。
    保留黑色方形连续 3 帧且置信度至少 30% 的确认条件。
    保留黑色方形独立的绿灯次数、旋转角度、MZ、减速和反馈过滤参数。
    旋转前记录起始航向，支持提前撤力/可选反向制动；超过目标角后定点返回起始航向。
"""

import copy
import math

import rospy
from auv_control.msg import ActuatorControl
from std_msgs.msg import String

from test_task1_v2_yellow_marker import (
    MODE_DPROV,
    YellowMarkerTest,
    clamp,
    class_names,
    wrap_angle,
    xy_distance,
    yaw_from_quaternion,
)


class BlackMarkerTest(YellowMarkerTest):
    """复用黄色子任务的搜索流程，执行黑色方形专属动作。"""

    def __init__(self):
        # 父类构造时会立即创建目标订阅；先准备回调最早可能访问的黑色参数。
        self.black_classes = class_names("~black_classes", ["rectangle"])
        self.black_min_confidence = float(rospy.get_param(
            "~black_min_confidence", 0.30
        ))
        super().__init__()
        self.node_name = "test_task1_v2_black_marker"
        self.marker_display_name = "黑色方形"

        # 黑色方形默认只要求连续 3 帧；launch 可以继续覆盖该值。
        self.marker_sample_count = max(1, int(rospy.get_param(
            "~marker_sample_count", 3
        )))

        self.black_light_count = max(1, int(rospy.get_param(
            "~black_light_count", 2
        )))
        self.black_rotation_angle = math.radians(float(rospy.get_param(
            "~black_rotation_angle_deg", 720.0
        )))
        self.black_rotation_drive_stop_angle = math.radians(float(rospy.get_param(
            "~black_rotation_drive_stop_angle_deg", 700.0
        )))
        self.black_rotation_mz = abs(float(rospy.get_param(
            "~black_rotation_mz", 3000.0
        )))
        direction = float(rospy.get_param("~black_rotation_direction", 1.0))
        self.black_rotation_direction = 1.0 if direction >= 0.0 else -1.0
        self.black_rotation_mz_step = abs(float(rospy.get_param(
            "~black_rotation_mz_step", 500.0
        )))
        self.black_rotation_slow_angle = math.radians(float(rospy.get_param(
            "~black_rotation_slow_angle_deg", 30.0
        )))
        self.black_rotation_slow_mz = abs(float(rospy.get_param(
            "~black_rotation_slow_mz", 1000.0
        )))
        self.black_rotation_brake_mz = abs(float(rospy.get_param(
            "~black_rotation_brake_mz", 0.0
        )))
        self.rotation_feedback_deadband = math.radians(float(rospy.get_param(
            "~rotation_feedback_deadband_deg", 0.05
        )))
        self.rotation_feedback_max_delta = math.radians(float(rospy.get_param(
            "~rotation_feedback_max_delta_deg", 45.0
        )))
        self.black_return_yaw_tolerance = math.radians(float(rospy.get_param(
            "~black_return_yaw_tolerance_deg", 5.0
        )))
        self.black_return_hold_seconds = float(rospy.get_param(
            "~black_return_hold_seconds", 2.0
        ))

        self.black_action_phase = "LIGHT"
        self.rotation_state = None

    def reset_marker_samples(self):
        self.marker_samples = []
        self.last_marker_sample_time = None

    def target_callback(self, message):
        """只接收满足连续帧和置信度要求的黑色 rectangle。"""
        if self.hold_z is None and not self.initialize_start_pose():
            return
        if self.detected_marker is not None or not self.camera_ready():
            return
        if message.type and message.type != "center":
            self.reset_marker_samples()
            return
        if message.class_name not in self.black_classes:
            self.reset_marker_samples()
            return
        if message.conf < self.black_min_confidence:
            self.reset_marker_samples()
            return

        camera_point = message.pose.pose.position
        if math.sqrt(
            camera_point.x ** 2 + camera_point.y ** 2 + camera_point.z ** 2
        ) > self.max_camera_distance:
            self.reset_marker_samples()
            return

        marker = self.transform_pose_to_map(message.pose)
        if marker is None:
            self.reset_marker_samples()
            return

        now = rospy.Time.now()
        if (
            self.last_marker_sample_time is None
            or (now - self.last_marker_sample_time).to_sec()
            > self.marker_sample_timeout
        ):
            self.marker_samples = []
        self.last_marker_sample_time = now

        if self.marker_samples:
            center_x = sum(
                item.pose.position.x for item in self.marker_samples
            ) / len(self.marker_samples)
            center_y = sum(
                item.pose.position.y for item in self.marker_samples
            ) / len(self.marker_samples)
            if math.hypot(
                marker.pose.position.x - center_x,
                marker.pose.position.y - center_y,
            ) > self.marker_cluster_distance:
                self.marker_samples = []

        self.marker_samples.append(copy.deepcopy(marker))
        if len(self.marker_samples) < self.marker_sample_count:
            return

        marker.pose.position.x = sum(
            item.pose.position.x for item in self.marker_samples
        ) / len(self.marker_samples)
        marker.pose.position.y = sum(
            item.pose.position.y for item in self.marker_samples
        ) / len(self.marker_samples)
        marker.pose.position.z = self.hold_z
        self.detected_marker = copy.deepcopy(marker)
        self.move_target = copy.deepcopy(marker.pose.position)
        self.search_target = None
        self.search_stable_since = None
        self.last_search_ty = 0.0

        current = self.get_current_pose()
        if current is None:
            self.detected_marker = None
            self.move_target = None
            self.reset_marker_samples()
            return

        distance = xy_distance(current.pose.position, self.move_target)
        rospy.loginfo(
            "%s: 识别状态=已识别；当前位置=(%.2f, %.2f, %.2f)，"
            "目标位置=(%.2f, %.2f, %.2f)，水平距离=%.2f m",
            self.node_name,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            self.move_target.x,
            self.move_target.y,
            self.move_target.z,
            distance,
        )
        self.step = self.STEP_MOVE

    def run_move_to_marker(self):
        current = self.get_current_pose()
        if current is None or self.move_target is None:
            return

        distance = xy_distance(current.pose.position, self.move_target)
        rospy.loginfo_throttle(
            2.0,
            "%s: 识别状态=已识别；当前位置=(%.2f, %.2f, %.2f)，"
            "目标位置=(%.2f, %.2f, %.2f)，水平距离=%.2f m",
            self.node_name,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            self.move_target.x,
            self.move_target.y,
            self.move_target.z,
            distance,
        )
        self.publish_position_target(self.move_target)
        if distance <= self.position_tolerance:
            rospy.loginfo(
                "%s: 已到达黑色方形位置，开始执行绿灯和旋转动作",
                self.node_name,
            )
            self.black_action_phase = "LIGHT"
            self.light_started_at = rospy.Time.now()
            self.rotation_state = None
            self.step = self.STEP_LIGHT

    def publish_lights(self, green):
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
        actuator.red_light = 0
        actuator.yellow_light = 0
        actuator.green_light = int(green)
        self.actuator_pub.publish(actuator)

    def run_light_action(self):
        if self.black_action_phase == "ROTATE":
            if self.run_black_rotation():
                self.step = self.STEP_FINISH
            return

        self.publish_position_target(self.move_target)
        elapsed = (rospy.Time.now() - self.light_started_at).to_sec()
        cycle_seconds = self.light_seconds + self.gap_seconds
        cycle_index = int(elapsed // cycle_seconds)
        if cycle_index >= self.black_light_count:
            self.publish_lights(0)
            self.black_action_phase = "ROTATE"
            self.rotation_state = None
            rospy.loginfo(
                "%s: 黑色方形亮灯完成，开始旋转 %.1f 度",
                self.node_name,
                math.degrees(self.black_rotation_angle),
            )
            return

        cycle_elapsed = elapsed - cycle_index * cycle_seconds
        self.publish_lights(1 if cycle_elapsed < self.light_seconds else 0)

    def publish_rotation_command(self, current, mz):
        target = copy.deepcopy(current)
        target.pose.position.x = self.move_target.x
        target.pose.position.y = self.move_target.y
        target.pose.position.z = self.hold_z
        self.publish_pose_command(2, target, mz=mz)

    def run_black_rotation(self):
        current = self.get_current_pose()
        if current is None:
            return False
        current_yaw = yaw_from_quaternion(current.pose.orientation)

        if self.rotation_state is None:
            self.rotation_state = {
                "phase": "DRIVE",
                "start_yaw": current_yaw,
                "last_yaw": current_yaw,
                "accumulated": 0.0,
                "commanded_mz": 0.0,
                "return_stable_since": None,
            }
            rospy.loginfo(
                "%s: 黑色方形旋转开始，起始航向=%.1f 度，"
                "提前撤力角=%.1f 度，目标累计角=%.1f 度",
                self.node_name,
                math.degrees(current_yaw),
                math.degrees(self.black_rotation_drive_stop_angle),
                math.degrees(self.black_rotation_angle),
            )

        delta = wrap_angle(current_yaw - self.rotation_state["last_yaw"])
        directed_delta = delta * self.black_rotation_direction
        if (
            abs(delta) <= self.rotation_feedback_max_delta
            and directed_delta > self.rotation_feedback_deadband
        ):
            self.rotation_state["accumulated"] += directed_delta
        self.rotation_state["last_yaw"] = current_yaw

        phase = self.rotation_state["phase"]
        accumulated = self.rotation_state["accumulated"]

        if phase == "DRIVE":
            if accumulated >= self.black_rotation_drive_stop_angle:
                self.rotation_state["phase"] = "COAST"
                self.rotation_state["commanded_mz"] = 0.0
                self.publish_rotation_command(current, 0.0)
                rospy.loginfo(
                    "%s: 累计旋转达到 %.1f 度，正向 MZ 已清零；"
                    "反向制动 MZ=%.0f",
                    self.node_name,
                    math.degrees(accumulated),
                    self.black_rotation_brake_mz,
                )
                return False

            remaining_to_stop = max(
                0.0, self.black_rotation_drive_stop_angle - accumulated
            )
            mz_magnitude = (
                self.black_rotation_slow_mz
                if remaining_to_stop <= self.black_rotation_slow_angle
                else self.black_rotation_mz
            )
            desired_mz = self.black_rotation_direction * mz_magnitude
            self.rotation_state["commanded_mz"] = clamp(
                desired_mz,
                self.rotation_state["commanded_mz"] - self.black_rotation_mz_step,
                self.rotation_state["commanded_mz"] + self.black_rotation_mz_step,
            )
            self.publish_rotation_command(
                current, self.rotation_state["commanded_mz"]
            )
            rospy.loginfo_throttle(
                2.0,
                "%s: 黑色方形正向旋转累计=%.1f 度，撤力角=%.1f 度，MZ=%d",
                self.node_name,
                math.degrees(accumulated),
                math.degrees(self.black_rotation_drive_stop_angle),
                int(self.rotation_state["commanded_mz"]),
            )
            return False

        if phase == "COAST":
            if accumulated >= self.black_rotation_angle:
                self.rotation_state["phase"] = "RETURN_HEADING"
                self.rotation_state["commanded_mz"] = 0.0
                self.rotation_state["return_stable_since"] = None
                self.pose_speed_sample = None
                rospy.loginfo(
                    "%s: 累计旋转达到 %.1f 度，开始返回起始航向 %.1f 度",
                    self.node_name,
                    math.degrees(accumulated),
                    math.degrees(self.rotation_state["start_yaw"]),
                )
            else:
                desired_brake_mz = (
                    -self.black_rotation_direction * self.black_rotation_brake_mz
                )
                self.rotation_state["commanded_mz"] = clamp(
                    desired_brake_mz,
                    self.rotation_state["commanded_mz"] - self.black_rotation_mz_step,
                    self.rotation_state["commanded_mz"] + self.black_rotation_mz_step,
                )
                self.publish_rotation_command(
                    current, self.rotation_state["commanded_mz"]
                )
                rospy.loginfo_throttle(
                    2.0,
                    "%s: 黑色方形惯性/制动阶段累计=%.1f/%.1f 度，MZ=%d",
                    self.node_name,
                    math.degrees(accumulated),
                    math.degrees(self.black_rotation_angle),
                    int(self.rotation_state["commanded_mz"]),
                )
                return False

        start_yaw = self.rotation_state["start_yaw"]
        return_target = self.make_pose(
            self.move_target.x,
            self.move_target.y,
            self.hold_z,
            start_yaw,
        )
        self.publish_pose_command(MODE_DPROV, return_target, mz=0.0)
        yaw_error = abs(wrap_angle(start_yaw - current_yaw))
        stable = self.motion_is_stable(current)
        if yaw_error <= self.black_return_yaw_tolerance and stable:
            if self.rotation_state["return_stable_since"] is None:
                self.rotation_state["return_stable_since"] = rospy.Time.now()
        else:
            self.rotation_state["return_stable_since"] = None

        held_seconds = 0.0
        if self.rotation_state["return_stable_since"] is not None:
            held_seconds = (
                rospy.Time.now() - self.rotation_state["return_stable_since"]
            ).to_sec()
        rospy.loginfo_throttle(
            2.0,
            "%s: 返回起始航向 current=%.1f target=%.1f error=%.1f 度，"
            "stable=%s hold=%.1f/%.1f s",
            self.node_name,
            math.degrees(current_yaw),
            math.degrees(start_yaw),
            math.degrees(yaw_error),
            stable,
            held_seconds,
            self.black_return_hold_seconds,
        )
        if held_seconds >= self.black_return_hold_seconds:
            self.publish_lights(0)
            return True
        return False

    def finish(self):
        self.publish_lights(0)
        if self.move_target is not None:
            self.publish_position_target(self.move_target)
        self.finished_pub.publish(String(data="black marker finished"))
        rospy.loginfo("%s: FINISH 黑色方形动作完成", self.node_name)
        rospy.signal_shutdown("black marker test complete")


def main():
    rospy.init_node("test_task1_v2_black_marker")
    BlackMarkerTest().run()


if __name__ == "__main__":
    main()
