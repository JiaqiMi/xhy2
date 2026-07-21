#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v3_black_marker.py
功能：Task1 黑色方形单项测试，运动接口迁移到 motion_supervisor。

流程：
    1. 定点保持启动位姿一段可调时间，等待相机和识别模块启动；
    2. 未识别到黑色方形时，依次定点左转、右转、回正，再向前移动后重复；
    3. 最近多条识别消息中达到可调有效帧数和置信度后确认位置；
    4. 发布 map 绝对目标前往黑色方形中心，以 HOVER 确认到达；
    5. 到达后亮绿灯，把累计旋转拆成多个绝对航向子目标完成默认两圈旋转；
    6. 动作完成后发布完成消息。

监听：/obj/target_message，/left/image_raw，/motion/state，/tf
发布：/cmd/motion/goal，/cmd/motion/cancel，/cmd/actuator，/finished

记录：
2026.7.14
    初版，用于单独验证 Task1 黑色方形识别、前往图形、亮灯和旋转流程。
2026.7.17
    同步黄色图形的相机等待、定点前进、TY 横移搜索、定点靠近和精简日志流程。
    保留黑色方形连续 3 帧且置信度至少 30% 的确认条件。
    保留黑色方形独立的绿灯次数、旋转角度、MZ、减速和反馈过滤参数。
    旋转前记录起始航向，支持提前撤力/可选反向制动；超过目标角后定点返回起始航向。
2026.7.18
    同步启动定点等待、N 取 K 滑动识别窗口和左右转后前进的搜索流程。
    新增 v3：迁移到 motion_supervisor；不再输出 MZ，累计两圈旋转改为逐个
    绝对航向目标，每个子目标均等待新鲜 HOVER 后才进入下一段。
2026.7.20
    复用黄色子任务的 YAML 数据记录，并增加黑色旋转阶段及累计角度数据。
"""

import math

import rospy
from auv_control.msg import ActuatorControl
from std_msgs.msg import String

from test_task1_v3_yellow_marker import (
    YellowMarkerTest,
    class_names,
    wrap_angle,
    xy_distance,
    yaw_from_quaternion,
)


class BlackMarkerTest(YellowMarkerTest):
    """复用黄色子任务的搜索流程，执行黑色方形专属动作。"""

    def __init__(self):
        # 父类构造时会立即创建目标订阅；先准备回调最早可能访问的黑色参数。
        self.node_name = "test_task1_v3_black_marker"
        self.marker_display_name = "黑色方形"
        self.black_classes = class_names("~black_classes", ["rectangle"])
        self.black_min_confidence = float(rospy.get_param(
            "~black_min_confidence", 0.30
        ))
        super().__init__()
        self.node_name = "test_task1_v3_black_marker"
        self.marker_display_name = "黑色方形"

        self.black_light_count = max(1, int(rospy.get_param(
            "~black_light_count", 2
        )))
        self.black_rotation_angle = math.radians(abs(float(rospy.get_param(
            "~black_rotation_angle_deg", 720.0
        ))))
        direction = float(rospy.get_param("~black_rotation_direction", 1.0))
        self.black_rotation_direction = 1.0 if direction >= 0.0 else -1.0
        self.black_rotation_step = math.radians(min(
            170.0,
            max(1.0, abs(float(rospy.get_param(
                "~black_rotation_step_deg", 90.0
            )))),
        ))
        self.black_rotation_step_hold_seconds = max(0.0, float(rospy.get_param(
            "~black_rotation_step_hold_seconds", 0.5
        )))
        self.black_return_hold_seconds = float(rospy.get_param(
            "~black_return_hold_seconds", 2.0
        ))

        self.black_action_phase = "LIGHT"
        self.rotation_state = None
        self.write_data_record(
            "black_configuration",
            black_min_confidence=self.black_min_confidence,
            black_light_count=self.black_light_count,
            black_rotation_angle_deg=math.degrees(
                self.black_rotation_angle
            ),
            black_rotation_step_deg=math.degrees(self.black_rotation_step),
        )

    def target_callback(self, message):
        """最近 N 条识别输出中有 K 条有效结果时确认黑色 rectangle。"""
        if self.hold_z is None and not self.initialize_start_pose():
            self.record_target_message(message, "robot_pose_unavailable")
            return
        if (
            self.step == self.STEP_WAIT_CAMERA
            or self.detected_marker is not None
            or not self.camera_ready()
        ):
            self.record_target_message(message, "ignored_in_current_step")
            return

        marker = None
        valid_message = (
            (not message.type or message.type == "center")
            and message.class_name in self.black_classes
            and float(message.conf) >= self.black_min_confidence
        )
        if valid_message:
            camera_point = message.pose.pose.position
            valid_message = (
                math.isfinite(camera_point.x)
                and math.isfinite(camera_point.y)
                and math.isfinite(camera_point.z)
                and math.sqrt(
                    camera_point.x ** 2
                    + camera_point.y ** 2
                    + camera_point.z ** 2
                ) <= self.max_camera_distance
            )
        if valid_message:
            marker = self.transform_pose_to_map(message.pose)

        confirmed = self.add_marker_observation(marker)
        self.record_target_message(
            message,
            "confirmed" if confirmed is not None else (
                "accepted" if marker is not None else "rejected"
            ),
            marker,
        )
        if confirmed is not None:
            self.lock_confirmed_marker(confirmed)

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
        if self.motion_arrived(self.move_goal):
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

    def run_black_rotation(self):
        """以多个小于 180 度的绝对航向目标表达累计旋转角。"""
        current = self.get_current_pose()
        if current is None:
            return False
        current_yaw = yaw_from_quaternion(current.pose.orientation)

        if self.rotation_state is None:
            self.rotation_state = {
                "start_yaw": current_yaw,
                "completed": 0.0,
                "step_angle": 0.0,
                "goal": None,
                "hover_started": None,
            }
            rospy.loginfo(
                "%s: 黑色方形旋转开始；起始航向=%.1f 度，"
                "累计目标=%.1f 度，单段最大=%.1f 度",
                self.node_name,
                math.degrees(current_yaw),
                math.degrees(self.black_rotation_angle),
                math.degrees(self.black_rotation_step),
            )

        if self.rotation_state["goal"] is None:
            remaining = max(
                0.0,
                self.black_rotation_angle - self.rotation_state["completed"],
            )
            step_angle = min(self.black_rotation_step, remaining)
            cumulative_target = self.rotation_state["completed"] + step_angle
            target_yaw = wrap_angle(
                self.rotation_state["start_yaw"]
                + self.black_rotation_direction * cumulative_target
            )
            self.rotation_state["step_angle"] = step_angle
            self.rotation_state["goal"] = self.make_pose(
                self.move_target.x,
                self.move_target.y,
                self.hold_z,
                target_yaw,
            )
            self.rotation_state["hover_started"] = None
            rospy.loginfo(
                "%s: 发布旋转子目标 %.1f/%.1f 度，目标航向=%.1f 度",
                self.node_name,
                math.degrees(cumulative_target),
                math.degrees(self.black_rotation_angle),
                math.degrees(target_yaw),
            )

        goal = self.rotation_state["goal"]
        self.publish_motion_goal(goal)
        arrived = self.motion_arrived(goal)
        if arrived:
            if self.rotation_state["hover_started"] is None:
                self.rotation_state["hover_started"] = rospy.Time.now()
        else:
            self.rotation_state["hover_started"] = None

        completed_after_step = (
            self.rotation_state["completed"]
            + self.rotation_state["step_angle"]
        )
        final_step = (
            completed_after_step >= self.black_rotation_angle - 1e-6
        )
        required_hold = (
            self.black_return_hold_seconds
            if final_step
            else self.black_rotation_step_hold_seconds
        )
        held_seconds = (
            (rospy.Time.now() - self.rotation_state["hover_started"]).to_sec()
            if self.rotation_state["hover_started"] is not None
            else 0.0
        )
        rospy.loginfo_throttle(
            2.0,
            "%s: 旋转子目标累计=%.1f/%.1f 度，当前=%.1f 度，"
            "HOVER=%s，保持=%.1f/%.1f s",
            self.node_name,
            math.degrees(completed_after_step),
            math.degrees(self.black_rotation_angle),
            math.degrees(current_yaw),
            "是" if arrived else "否",
            held_seconds,
            required_hold,
        )

        if not arrived or held_seconds < required_hold:
            return False
        self.rotation_state["completed"] = completed_after_step
        if final_step:
            self.publish_lights(0)
            return True
        self.rotation_state["goal"] = None
        self.rotation_state["hover_started"] = None
        return False

    def finish(self):
        self.publish_lights(0)
        self.cancel_motion()
        self.finished_pub.publish(String(data="black marker finished"))
        rospy.loginfo("%s: FINISH 黑色方形动作完成", self.node_name)
        rospy.signal_shutdown("black marker test complete")


def main():
    rospy.init_node("test_task1_v3_black_marker")
    BlackMarkerTest().run()


if __name__ == "__main__":
    main()
