#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v2_yellow_marker.py
功能：Task1 黄色图形单项测试。

流程：
    1. 以节点启动时机器人当前位置为起点，并记录当前 z，不主动改变高度；
    2. 按设定初始航向手控前进，直到识别到指定图形；
    3. 将图形位置转换到 map 坐标系；
    4. 使用动力定位 ROV 模式只在 XY 平面前往图形上方；
    5. 到达后执行对应灯光动作。

监听：/obj/target_message，/tf
发布：/cmd/pose/ned，/cmd/actuator，/finished

记录：
2026.7.14
    初版，用于单独验证 Task1 图形识别、前往图形和黄色动作流程。
"""

import copy
import math

import rospy
import tf
from auv_control.msg import ActuatorControl, PoseNEDcmd, TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


MODE_DEPTH_HDG = 3
MODE_DPROV = 4
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


class Task1MarkerActionTest:
    """搜索一个指定图形，前往图形上方并执行动作。"""

    STEP_SEARCH = "SEARCH_MARKER"
    STEP_MOVE = "MOVE_TO_MARKER"
    STEP_LIGHT = "LIGHT_ACTION"
    STEP_ROTATE = "ROTATE_BLACK"
    STEP_FINISH = "FINISH"

    def __init__(self, node_name, marker_kind):
        self.node_name = node_name
        self.marker_kind = marker_kind
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))
        self.tf_listener = tf.TransformListener()

        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd/pose/ned")
        self.target_topic = rospy.get_param("~target_topic", "/obj/target_message")
        self.actuator_topic = rospy.get_param("~actuator_topic", "/cmd/actuator")
        self.finished_topic = rospy.get_param("~finished_topic", "/finished")

        self.initial_search_yaw = math.radians(float(rospy.get_param(
            "~initial_heading_deg", DEFAULT_INITIAL_HEADING_DEG
        )))
        self.search_forward_force = float(rospy.get_param("~search_forward_force", 120.0))
        self.manual_tx_sign = float(rospy.get_param("~manual_tx_sign", 1.0))

        self.max_xy_step = float(rospy.get_param("~max_xy_step", 0.4))
        self.position_tolerance = float(rospy.get_param("~position_tolerance", 0.15))
        self.yaw_tolerance = math.radians(float(rospy.get_param("~yaw_tolerance_deg", 3.0)))
        self.max_yaw_step = math.radians(float(rospy.get_param("~max_yaw_step_deg", 5.0)))
        self.max_camera_distance = float(rospy.get_param("~max_camera_distance", 5.0))

        self.light_seconds = float(rospy.get_param("~light_seconds", 3.0))
        self.gap_seconds = float(rospy.get_param("~gap_seconds", 0.5))
        self.black_rotation_angle = math.radians(float(rospy.get_param(
            "~black_rotation_angle_deg", 360.0
        )))
        self.rotation_yaw_step = math.radians(float(rospy.get_param(
            "~rotation_yaw_step_deg", 3.0
        )))
        self.rotation_stop_margin = math.radians(float(rospy.get_param(
            "~rotation_stop_margin_deg", 10.0
        )))

        self.yellow_classes = class_names("~yellow_classes", ["triangle", "circle"])
        self.black_classes = class_names("~black_classes", ["rectangle"])

        self.light1 = int(rospy.get_param("~light1", 0))
        self.light2 = int(rospy.get_param("~light2", 0))
        self.heading_servo = int(rospy.get_param("~heading_servo", 0x80))
        self.clamp_servo = int(rospy.get_param("~clamp_servo", 0x00))
        self.drive_cmd = int(rospy.get_param("~drive_cmd", 0))
        self.drive_speed = int(rospy.get_param("~drive_speed", 0))

        self.cmd_pub = rospy.Publisher(self.cmd_topic, PoseNEDcmd, queue_size=10)
        self.actuator_pub = rospy.Publisher(self.actuator_topic, ActuatorControl, queue_size=10)
        self.finished_pub = rospy.Publisher(self.finished_topic, String, queue_size=10)
        rospy.Subscriber(self.target_topic, TargetDetection, self.target_callback)

        self.step = self.STEP_SEARCH
        self.step_started = rospy.Time.now()
        self.start_pose = None
        self.hold_z = None
        self.detected_marker = None
        self.move_target = None
        self.light_action_state = None
        self.rotation_state = None

        rospy.loginfo(
            "%s: initialized marker_kind=%s initial_heading=%.1fdeg target_topic=%s",
            self.node_name,
            self.marker_kind,
            math.degrees(self.initial_search_yaw),
            self.target_topic,
        )

    def set_step(self, step):
        old_step = self.step
        elapsed = (rospy.Time.now() - self.step_started).to_sec()
        self.step = step
        self.step_started = rospy.Time.now()
        if old_step != step:
            rospy.loginfo(
                "%s: step %s -> %s, previous_step_elapsed=%.1fs",
                self.node_name,
                old_step,
                step,
                elapsed,
            )

    def step_elapsed(self):
        return (rospy.Time.now() - self.step_started).to_sec()

    def get_current_pose(self):
        try:
            self.tf_listener.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(1.0)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(2, "%s: cannot read current pose: %s", self.node_name, error)
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
            self.node_name,
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

    def publish_pose_cmd(self, mode, target, tx=0, ty=0):
        cmd = PoseNEDcmd()
        cmd.mode = int(mode)
        cmd.target = copy.deepcopy(target)
        cmd.target.header.frame_id = "map"
        cmd.target.header.stamp = rospy.Time.now()
        cmd.force.TX = self.force_value(tx)
        cmd.force.TY = self.force_value(ty)
        self.cmd_pub.publish(cmd)
        rospy.loginfo_throttle(
            1.0,
            "%s: pose cmd mode=%d target=(%.2f, %.2f, %.2f, yaw=%.1fdeg) force=(%d,%d)",
            self.node_name,
            cmd.mode,
            cmd.target.pose.position.x,
            cmd.target.pose.position.y,
            cmd.target.pose.position.z,
            math.degrees(yaw_from_quaternion(cmd.target.pose.orientation)),
            cmd.force.TX,
            cmd.force.TY,
        )

    def publish_current_manual_cmd(self, yaw, tx=0, ty=0):
        current = self.get_current_pose()
        if current is None:
            return False
        self.publish_pose_cmd(
            MODE_DEPTH_HDG,
            self.make_pose(current.pose.position.x, current.pose.position.y, yaw),
            tx=tx,
            ty=ty,
        )
        return True

    def publish_lights(self, red=0, green=0):
        msg = ActuatorControl()
        msg.light1 = self.light1
        msg.light2 = self.light2
        msg.heading_servo = self.heading_servo
        msg.clamp_servo = self.clamp_servo
        msg.drive_cmd = self.drive_cmd
        msg.drive_speed = self.drive_speed
        msg.red_light = int(red)
        msg.yellow_light = 0
        msg.green_light = int(green)
        self.actuator_pub.publish(msg)
        rospy.loginfo_throttle(
            1.0,
            "%s: actuator red=%d green=%d light=(%d,%d) servo=(%d,%d)",
            self.node_name,
            msg.red_light,
            msg.green_light,
            msg.light1,
            msg.light2,
            msg.heading_servo,
            msg.clamp_servo,
        )

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
                rospy.logwarn_throttle(2, "%s: marker transform failed: %s", self.node_name, error)
                return None

    def marker_type_from_class(self, class_name):
        if class_name in self.yellow_classes:
            return "yellow"
        if class_name in self.black_classes:
            return "black"
        return None

    def target_callback(self, message):
        if self.hold_z is None and not self.initialize_start_pose():
            return

        if self.detected_marker is not None:
            return
        if message.type and message.type != "center":
            return

        detected_kind = self.marker_type_from_class(message.class_name)
        if detected_kind != self.marker_kind:
            return

        camera_origin = Point()
        if math.sqrt(
            message.pose.pose.position.x ** 2
            + message.pose.pose.position.y ** 2
            + message.pose.pose.position.z ** 2
        ) > self.max_camera_distance:
            rospy.loginfo_throttle(
                2.0,
                "%s: ignore far %s marker camera_pos=(%.2f, %.2f, %.2f)",
                self.node_name,
                detected_kind,
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            )
            return

        marker = self.transform_pose_to_map(message.pose)
        if marker is None:
            return

        self.detected_marker = marker
        current = self.get_current_pose()
        if current is not None:
            yaw = math.atan2(
                marker.pose.position.y - current.pose.position.y,
                marker.pose.position.x - current.pose.position.x,
            )
        else:
            yaw = self.initial_search_yaw
        self.move_target = self.make_pose(marker.pose.position.x, marker.pose.position.y, yaw)

        rospy.loginfo(
            "%s: detected %s marker class=%s conf=%.2f map=(%.2f, %.2f, %.2f), "
            "move_target=(%.2f, %.2f, %.2f, yaw=%.1fdeg)",
            self.node_name,
            detected_kind,
            message.class_name,
            message.conf,
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
            self.move_target.pose.position.x,
            self.move_target.pose.position.y,
            self.move_target.pose.position.z,
            math.degrees(yaw),
        )
        self.set_step(self.STEP_MOVE)

    def search_marker(self):
        self.publish_current_manual_cmd(
            self.initial_search_yaw,
            tx=self.manual_tx_sign * self.search_forward_force,
            ty=0,
        )
        rospy.loginfo_throttle(
            1.0,
            "%s: searching %s marker heading=%.1fdeg force=%.0f",
            self.node_name,
            self.marker_kind,
            math.degrees(self.initial_search_yaw),
            self.search_forward_force,
        )

    def move_to_marker(self):
        if self.move_target is None:
            self.set_step(self.STEP_SEARCH)
            return False

        current = self.get_current_pose()
        if current is None:
            return False

        dx = self.move_target.pose.position.x - current.pose.position.x
        dy = self.move_target.pose.position.y - current.pose.position.y
        distance = math.hypot(dx, dy)
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        target_yaw = yaw_from_quaternion(self.move_target.pose.orientation)

        if distance > self.position_tolerance:
            move_yaw = math.atan2(dy, dx)
            yaw_error = wrap_angle(move_yaw - current_yaw)
            if abs(yaw_error) > self.yaw_tolerance:
                cmd_yaw = current_yaw + clamp(yaw_error, -self.max_yaw_step, self.max_yaw_step)
                target = self.make_pose(current.pose.position.x, current.pose.position.y, cmd_yaw)
                self.publish_pose_cmd(MODE_DPROV, target)
                rospy.loginfo_throttle(
                    1.0,
                    "%s: align to marker distance=%.2f yaw_error=%.1fdeg",
                    self.node_name,
                    distance,
                    math.degrees(yaw_error),
                )
                return False

            step = min(self.max_xy_step, distance)
            scale = step / distance
            target = self.make_pose(
                current.pose.position.x + dx * scale,
                current.pose.position.y + dy * scale,
                move_yaw,
            )
            self.publish_pose_cmd(MODE_DPROV, target)
            rospy.loginfo_throttle(
                1.0,
                "%s: move to marker distance=%.2f step=%.2f current=(%.2f, %.2f) "
                "target=(%.2f, %.2f)",
                self.node_name,
                distance,
                step,
                current.pose.position.x,
                current.pose.position.y,
                self.move_target.pose.position.x,
                self.move_target.pose.position.y,
            )
            return False

        yaw_error = wrap_angle(target_yaw - current_yaw)
        if abs(yaw_error) > self.yaw_tolerance:
            cmd_yaw = current_yaw + clamp(yaw_error, -self.max_yaw_step, self.max_yaw_step)
            self.publish_pose_cmd(
                MODE_DPROV,
                self.make_pose(self.move_target.pose.position.x, self.move_target.pose.position.y, cmd_yaw),
            )
            rospy.loginfo_throttle(
                1.0,
                "%s: final marker yaw align yaw_error=%.1fdeg",
                self.node_name,
                math.degrees(yaw_error),
            )
            return False

        self.publish_pose_cmd(MODE_DPROV, self.move_target)
        rospy.loginfo(
            "%s: arrived above %s marker, distance=%.2f",
            self.node_name,
            self.marker_kind,
            distance,
        )
        return True

    def start_light_action(self):
        if self.light_action_state is not None:
            return

        if self.marker_kind == "yellow":
            self.light_action_state = {"count": 1, "red": 1, "green": 0}
        else:
            self.light_action_state = {"count": 2, "red": 0, "green": 1}

        rospy.loginfo(
            "%s: light action start marker=%s count=%d on=%.1fs off=%.1fs",
            self.node_name,
            self.marker_kind,
            self.light_action_state["count"],
            self.light_seconds,
            self.gap_seconds,
        )

    def run_light_action(self):
        self.start_light_action()
        self.publish_pose_cmd(MODE_DPROV, self.move_target)

        elapsed = self.step_elapsed()
        cycle = self.light_seconds + self.gap_seconds
        current_count = int(elapsed // cycle)
        if current_count >= self.light_action_state["count"]:
            self.publish_lights(0, 0)
            rospy.loginfo(
                "%s: light action complete marker=%s elapsed=%.1fs",
                self.node_name,
                self.marker_kind,
                elapsed,
            )
            return True

        in_cycle = elapsed - current_count * cycle
        if in_cycle < self.light_seconds:
            self.publish_lights(self.light_action_state["red"], self.light_action_state["green"])
            rospy.loginfo_throttle(
                1.0,
                "%s: light action running marker=%s cycle=%d/%d elapsed=%.1fs",
                self.node_name,
                self.marker_kind,
                current_count + 1,
                self.light_action_state["count"],
                elapsed,
            )
        else:
            self.publish_lights(0, 0)
            rospy.loginfo_throttle(
                1.0,
                "%s: light action off-gap marker=%s cycle=%d/%d elapsed=%.1fs",
                self.node_name,
                self.marker_kind,
                current_count + 1,
                self.light_action_state["count"],
                elapsed,
            )
        return False

    def rotate_black(self):
        current = self.get_current_pose()
        if current is None:
            return False

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        if self.rotation_state is None:
            self.rotation_state = {
                "last_yaw": current_yaw,
                "accumulated": 0.0,
                "direction": 1.0,
            }
            rospy.loginfo(
                "%s: black rotation start yaw=%.1fdeg target=%.1fdeg",
                self.node_name,
                math.degrees(current_yaw),
                math.degrees(self.black_rotation_angle),
            )

        delta = wrap_angle(current_yaw - self.rotation_state["last_yaw"])
        if delta * self.rotation_state["direction"] > 0.0:
            self.rotation_state["accumulated"] += abs(delta)
        self.rotation_state["last_yaw"] = current_yaw

        finish_angle = max(0.0, self.black_rotation_angle - self.rotation_stop_margin)
        if self.rotation_state["accumulated"] >= finish_angle:
            self.publish_lights(0, 0)
            self.publish_pose_cmd(MODE_DPROV, self.move_target)
            rospy.loginfo(
                "%s: black rotation complete accumulated=%.1fdeg",
                self.node_name,
                math.degrees(self.rotation_state["accumulated"]),
            )
            return True

        commanded_yaw = wrap_angle(current_yaw + self.rotation_state["direction"] * self.rotation_yaw_step)
        target = self.make_pose(
            self.move_target.pose.position.x,
            self.move_target.pose.position.y,
            commanded_yaw,
        )
        self.publish_lights(0, 1)
        self.publish_pose_cmd(MODE_DPROV, target)
        rospy.loginfo_throttle(
            1.0,
            "%s: black rotating accumulated=%.1f/%.1fdeg current_yaw=%.1fdeg cmd_yaw=%.1fdeg",
            self.node_name,
            math.degrees(self.rotation_state["accumulated"]),
            math.degrees(self.black_rotation_angle),
            math.degrees(current_yaw),
            math.degrees(commanded_yaw),
        )
        return False

    def finish(self):
        self.publish_lights(0, 0)
        current = self.get_current_pose()
        if current is not None:
            self.publish_pose_cmd(
                MODE_DPROV,
                self.make_pose(
                    current.pose.position.x,
                    current.pose.position.y,
                    yaw_from_quaternion(current.pose.orientation),
                ),
            )
        self.finished_pub.publish(String(data="%s finished" % self.node_name))
        rospy.loginfo("%s: finished %s marker test", self.node_name, self.marker_kind)
        rospy.signal_shutdown("%s complete" % self.node_name)

    def run(self):
        while not rospy.is_shutdown():
            if not self.initialize_start_pose():
                self.rate.sleep()
                continue

            if self.step == self.STEP_SEARCH:
                self.search_marker()
            elif self.step == self.STEP_MOVE:
                if self.move_to_marker():
                    self.set_step(self.STEP_LIGHT)
            elif self.step == self.STEP_LIGHT:
                if self.run_light_action():
                    if self.marker_kind == "black":
                        self.rotation_state = None
                        self.set_step(self.STEP_ROTATE)
                    else:
                        self.set_step(self.STEP_FINISH)
            elif self.step == self.STEP_ROTATE:
                if self.rotate_black():
                    self.set_step(self.STEP_FINISH)
            elif self.step == self.STEP_FINISH:
                self.finish()

            self.rate.sleep()


def main():
    rospy.init_node("test_task1_v2_yellow_marker")
    Task1MarkerActionTest("test_task1_v2_yellow_marker", "yellow").run()


if __name__ == "__main__":
    main()
