#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v2_yellow_marker.py
功能：Task1 黄色图形单项测试。

流程：
    1. 保持启动位姿，等待相机稳定出图后慢速定点对准指定航向；
    2. 按设定航向手控前进，以连续 10 帧确定指定图形位置；
    3. 逐步制动并定点稳定，再只在 XY 平面前往图形上方；
    4. 到达图形中心稳定 3~5 s 后执行灯光动作；
    5. 动作完成后回图形中心及原航向，再次稳定后结束单项测试。

监听：/obj/target_message，/left/image_raw，/status/vel（可选），/tf
发布：/cmd/pose/ned，/cmd/actuator，/finished

记录：
2026.7.14
    初版，用于单独验证 Task1 图形识别、前往图形和黄色动作流程。
2026.7.16
    同步正式 Task1 的启动等待、10 帧稳定识别、手控制动、速度稳定、动作前后定点和黑色 MZ 旋转。
    增加亮灯次数、旋转方向、MZ 步长、减速区和航向反馈过滤参数。
    黑色单项测试改为连续 3 帧 rectangle 且每帧置信度不低于 0.30。
"""

import copy
import math

import rospy
import tf
from auv_control.msg import ActuatorControl, PoseNEDcmd, TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TwistStamped
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
    STEP_WAIT_READY = "WAIT_READY"
    STEP_SETTLE = "SETTLE"

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
        self.search_forward_force = float(rospy.get_param("~search_forward_force", 1000.0))
        self.manual_force_step = float(rospy.get_param("~manual_force_step", 200.0))
        self.manual_brake_step = float(rospy.get_param("~manual_brake_step", 300.0))
        self.manual_tx_sign = float(rospy.get_param("~manual_tx_sign", 1.0))

        self.max_xy_step = float(rospy.get_param("~max_xy_step", 0.4))
        self.position_tolerance = float(rospy.get_param("~position_tolerance", 0.15))
        self.yaw_tolerance = math.radians(float(rospy.get_param("~yaw_tolerance_deg", 3.0)))
        self.max_yaw_step = math.radians(float(rospy.get_param("~max_yaw_step_deg", 2.0)))
        self.max_camera_distance = float(rospy.get_param("~max_camera_distance", 5.0))
        self.marker_sample_count = int(rospy.get_param("~marker_sample_count", 10))
        self.black_min_confidence = float(rospy.get_param(
            "~black_min_confidence", 0.30
        ))
        self.marker_cluster_distance = float(rospy.get_param(
            "~marker_cluster_distance", 0.25
        ))

        self.light_seconds = float(rospy.get_param("~light_seconds", 3.0))
        self.gap_seconds = float(rospy.get_param("~gap_seconds", 0.5))
        self.yellow_light_count = int(rospy.get_param("~yellow_light_count", 1))
        self.black_light_count = int(rospy.get_param("~black_light_count", 2))
        self.black_rotation_angle = math.radians(float(rospy.get_param(
            "~black_rotation_angle_deg", 720.0
        )))
        self.rotation_stop_margin = math.radians(float(rospy.get_param(
            "~rotation_stop_margin_deg", 10.0
        )))
        self.black_rotation_mz = float(rospy.get_param("~black_rotation_mz", 3000.0))
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
        self.rotation_feedback_deadband = math.radians(float(rospy.get_param(
            "~rotation_feedback_deadband_deg", 0.05
        )))
        self.rotation_feedback_max_delta = math.radians(float(rospy.get_param(
            "~rotation_feedback_max_delta_deg", 45.0
        )))

        self.camera_topic = rospy.get_param("~camera_topic", "/left/image_raw")
        self.camera_message_timeout = float(rospy.get_param(
            "~camera_message_timeout", 2.0
        ))
        self.startup_hold_seconds = float(rospy.get_param("~startup_hold_seconds", 10.0))
        self.transition_hold_seconds = float(rospy.get_param(
            "~transition_hold_seconds", 4.0
        ))
        self.velocity_topic = rospy.get_param("~velocity_topic", "/status/vel")
        self.velocity_message_timeout = float(rospy.get_param(
            "~velocity_message_timeout", 1.0
        ))
        self.stable_linear_speed = float(rospy.get_param("~stable_linear_speed", 0.05))
        self.stable_angular_speed = math.radians(float(rospy.get_param(
            "~stable_angular_speed_deg", 3.0
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
        rospy.Subscriber(self.camera_topic, rospy.AnyMsg, self.camera_callback, queue_size=1)
        rospy.Subscriber(self.velocity_topic, TwistStamped, self.velocity_callback, queue_size=5)

        self.step = self.STEP_WAIT_READY
        self.step_started = rospy.Time.now()
        self.start_pose = None
        self.hold_z = None
        self.detected_marker = None
        self.move_target = None
        self.light_action_state = None
        self.rotation_state = None
        self.marker_samples = []
        self.last_manual_tx = 0
        self.last_manual_ty = 0
        self.last_camera_time = None
        self.latest_velocity = None
        self.latest_velocity_time = None
        self.pose_speed_sample = None
        self.settle_target = None
        self.settle_next_step = None
        self.settle_reason = ""
        self.settle_stable_since = None

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

    def camera_callback(self, _message):
        self.last_camera_time = rospy.Time.now()

    def velocity_callback(self, message):
        self.latest_velocity = copy.deepcopy(message.twist)
        self.latest_velocity_time = rospy.Time.now()

    def camera_ready(self):
        return (
            self.last_camera_time is not None
            and (rospy.Time.now() - self.last_camera_time).to_sec()
            <= self.camera_message_timeout
        )

    @staticmethod
    def approach_zero(value, step):
        if value > 0:
            return max(0, value - step)
        if value < 0:
            return min(0, value + step)
        return 0

    def limit_force(self, desired, previous):
        return int(round(clamp(
            desired,
            previous - self.manual_force_step,
            previous + self.manual_force_step,
        )))

    def motion_is_stable(self, current):
        now = rospy.Time.now()
        if (
            self.latest_velocity is not None
            and self.latest_velocity_time is not None
            and (now - self.latest_velocity_time).to_sec()
            <= self.velocity_message_timeout
        ):
            linear = math.hypot(
                self.latest_velocity.linear.x, self.latest_velocity.linear.y
            )
            angular = abs(self.latest_velocity.angular.z)
            source = "velocity_topic"
        else:
            yaw = yaw_from_quaternion(current.pose.orientation)
            sample = (now, current.pose.position.x, current.pose.position.y, yaw)
            previous = self.pose_speed_sample
            self.pose_speed_sample = sample
            if previous is None or (now - previous[0]).to_sec() <= 0.05:
                rospy.loginfo_throttle(1.0, "%s: TF speed estimate warming", self.node_name)
                return False
            elapsed = (now - previous[0]).to_sec()
            linear = math.hypot(sample[1] - previous[1], sample[2] - previous[2]) / elapsed
            angular = abs(wrap_angle(sample[3] - previous[3])) / elapsed
            source = "tf_difference"
        stable = (
            linear <= self.stable_linear_speed
            and angular <= self.stable_angular_speed
        )
        rospy.loginfo_throttle(
            1.0,
            "%s: stable check source=%s linear=%.3fm/s angular=%.2fdeg/s result=%s",
            self.node_name,
            source,
            linear,
            math.degrees(angular),
            stable,
        )
        return stable

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

    def publish_pose_cmd(self, mode, target, tx=0, ty=0, mz=0):
        cmd = PoseNEDcmd()
        cmd.mode = int(mode)
        cmd.target = copy.deepcopy(target)
        cmd.target.header.frame_id = "map"
        cmd.target.header.stamp = rospy.Time.now()
        cmd.force.TX = self.force_value(tx)
        cmd.force.TY = self.force_value(ty)
        cmd.force.MZ = self.force_value(mz)
        self.cmd_pub.publish(cmd)
        rospy.loginfo_throttle(
            1.0,
            "%s: pose cmd mode=%d target=(%.2f, %.2f, %.2f, yaw=%.1fdeg) force=(%d,%d,MZ=%d)",
            self.node_name,
            cmd.mode,
            cmd.target.pose.position.x,
            cmd.target.pose.position.y,
            cmd.target.pose.position.z,
            math.degrees(yaw_from_quaternion(cmd.target.pose.orientation)),
            cmd.force.TX,
            cmd.force.TY,
            cmd.force.MZ,
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
        light_msg = ActuatorControl()
        light_msg.mode = 1
        light_msg.light1 = self.light1
        light_msg.light2 = self.light2
        self.actuator_pub.publish(light_msg)

        msg = ActuatorControl()
        msg.mode = 2
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
            light_msg.light1,
            light_msg.light2,
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

        if self.detected_marker is not None or self.step != self.STEP_SEARCH:
            return
        if message.type and message.type != "center":
            if self.marker_kind == "black" and self.marker_samples:
                rospy.loginfo(
                    "%s: black rectangle streak interrupted by type=%s; restart",
                    self.node_name,
                    message.type,
                )
                self.marker_samples = []
            return

        detected_kind = self.marker_type_from_class(message.class_name)
        if detected_kind != self.marker_kind:
            if self.marker_kind == "black" and self.marker_samples:
                rospy.loginfo(
                    "%s: black rectangle streak interrupted by class=%s; restart",
                    self.node_name,
                    message.class_name,
                )
                self.marker_samples = []
            return

        if self.marker_kind == "black" and message.conf < self.black_min_confidence:
            rospy.loginfo_throttle(
                1.0,
                "%s: reject black rectangle conf=%.2f < %.2f; restart streak",
                self.node_name,
                message.conf,
                self.black_min_confidence,
            )
            self.marker_samples = []
            return

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
            if self.marker_kind == "black":
                self.marker_samples = []
            return

        marker = self.transform_pose_to_map(message.pose)
        if marker is None:
            if self.marker_kind == "black":
                self.marker_samples = []
            return

        if self.marker_samples and xy_distance(
            marker.pose.position, self.marker_samples[0].pose.position
        ) > self.marker_cluster_distance:
            rospy.loginfo(
                "%s: marker cluster moved %.2fm; restart frame collection",
                self.node_name,
                xy_distance(marker.pose.position, self.marker_samples[0].pose.position),
            )
            self.marker_samples = []
        self.marker_samples.append(copy.deepcopy(marker))
        rospy.loginfo_throttle(
            1.0,
            "%s: collecting %s marker samples=%d/%d latest=(%.2f,%.2f)",
            self.node_name,
            detected_kind,
            len(self.marker_samples),
            self.marker_sample_count,
            marker.pose.position.x,
            marker.pose.position.y,
        )
        if len(self.marker_samples) < self.marker_sample_count:
            return

        marker.pose.position.x = sum(
            sample.pose.position.x for sample in self.marker_samples
        ) / len(self.marker_samples)
        marker.pose.position.y = sum(
            sample.pose.position.y for sample in self.marker_samples
        ) / len(self.marker_samples)
        marker.pose.position.z = self.hold_z

        self.detected_marker = marker
        current = self.get_current_pose()
        if current is None:
            self.detected_marker = None
            self.marker_samples = []
            return
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
        if current is not None:
            hold_target = self.make_pose(
                current.pose.position.x,
                current.pose.position.y,
                self.initial_search_yaw,
            )
            self.begin_settle(
                hold_target,
                self.STEP_MOVE,
                "stable_marker_detected",
            )

    def search_marker(self):
        self.last_manual_tx = self.limit_force(
            self.manual_tx_sign * self.search_forward_force,
            self.last_manual_tx,
        )
        self.last_manual_ty = self.approach_zero(
            self.last_manual_ty, self.manual_brake_step
        )
        self.publish_current_manual_cmd(
            self.initial_search_yaw,
            tx=self.last_manual_tx,
            ty=self.last_manual_ty,
        )
        rospy.loginfo_throttle(
            1.0,
            "%s: searching %s marker heading=%.1fdeg force=%.0f",
            self.node_name,
            self.marker_kind,
            math.degrees(self.initial_search_yaw),
            self.last_manual_tx,
        )

    def move_to_pose(self, target):
        """慢速定点到指定 XY，最后恢复目标航向。"""
        if target is None:
            return False
        current = self.get_current_pose()
        if current is None:
            return False

        dx = target.pose.position.x - current.pose.position.x
        dy = target.pose.position.y - current.pose.position.y
        distance = math.hypot(dx, dy)
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        target_yaw = yaw_from_quaternion(target.pose.orientation)

        if distance > self.position_tolerance:
            move_yaw = math.atan2(dy, dx)
            yaw_error = wrap_angle(move_yaw - current_yaw)
            if abs(yaw_error) > self.yaw_tolerance:
                cmd_yaw = current_yaw + clamp(yaw_error, -self.max_yaw_step, self.max_yaw_step)
                target = self.make_pose(current.pose.position.x, current.pose.position.y, cmd_yaw)
                self.publish_pose_cmd(MODE_DPROV, target)
                rospy.loginfo_throttle(
                    1.0,
                    "%s: DPROV align distance=%.2f yaw_error=%.1fdeg",
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
                target.pose.position.x,
                target.pose.position.y,
            )
            return False

        yaw_error = wrap_angle(target_yaw - current_yaw)
        if abs(yaw_error) > self.yaw_tolerance:
            cmd_yaw = current_yaw + clamp(yaw_error, -self.max_yaw_step, self.max_yaw_step)
            self.publish_pose_cmd(
                MODE_DPROV,
                self.make_pose(target.pose.position.x, target.pose.position.y, cmd_yaw),
            )
            rospy.loginfo_throttle(
                1.0,
                "%s: final marker yaw align yaw_error=%.1fdeg",
                self.node_name,
                math.degrees(yaw_error),
            )
            return False

        self.publish_pose_cmd(MODE_DPROV, target)
        return True

    def move_to_marker(self):
        if self.move_target is None:
            self.set_step(self.STEP_SEARCH)
            return False
        reached = self.move_to_pose(self.move_target)
        if reached:
            rospy.loginfo_throttle(
                1.0,
                "%s: arrived above %s marker",
                self.node_name,
                self.marker_kind,
            )
        return reached

    def begin_settle(self, target, next_step, reason):
        self.settle_target = copy.deepcopy(target)
        self.settle_next_step = next_step
        self.settle_reason = reason
        self.settle_stable_since = None
        self.pose_speed_sample = None
        rospy.loginfo(
            "%s: begin settle reason=%s target=(%.2f,%.2f,yaw=%.1fdeg) hold=%.1fs",
            self.node_name,
            reason,
            target.pose.position.x,
            target.pose.position.y,
            math.degrees(yaw_from_quaternion(target.pose.orientation)),
            self.transition_hold_seconds,
        )
        self.set_step(self.STEP_SETTLE)

    def run_settle(self):
        if self.last_manual_tx != 0 or self.last_manual_ty != 0:
            self.last_manual_tx = self.approach_zero(
                self.last_manual_tx, self.manual_brake_step
            )
            self.last_manual_ty = self.approach_zero(
                self.last_manual_ty, self.manual_brake_step
            )
            self.publish_current_manual_cmd(
                yaw_from_quaternion(self.settle_target.pose.orientation),
                self.last_manual_tx,
                self.last_manual_ty,
            )
            self.settle_stable_since = None
            rospy.loginfo_throttle(
                1.0,
                "%s: braking before DPROV force=(%d,%d)",
                self.node_name,
                self.last_manual_tx,
                self.last_manual_ty,
            )
            return
        if not self.move_to_pose(self.settle_target):
            self.settle_stable_since = None
            return
        current = self.get_current_pose()
        if current is None or not self.motion_is_stable(current):
            self.settle_stable_since = None
            return
        if self.settle_stable_since is None:
            self.settle_stable_since = rospy.Time.now()
        stable_seconds = (rospy.Time.now() - self.settle_stable_since).to_sec()
        rospy.loginfo_throttle(
            1.0,
            "%s: point hold reason=%s stable=%.1f/%.1fs",
            self.node_name,
            self.settle_reason,
            stable_seconds,
            self.transition_hold_seconds,
        )
        if stable_seconds >= self.transition_hold_seconds:
            next_step = self.settle_next_step
            rospy.loginfo("%s: settle complete reason=%s", self.node_name, self.settle_reason)
            self.settle_target = None
            self.settle_stable_since = None
            self.set_step(next_step)

    def start_light_action(self):
        if self.light_action_state is not None:
            return

        if self.marker_kind == "yellow":
            self.light_action_state = {
                "count": self.yellow_light_count,
                "red": 1,
                "green": 0,
            }
        else:
            self.light_action_state = {
                "count": self.black_light_count,
                "red": 0,
                "green": 1,
            }

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
                "direction": self.black_rotation_direction,
                "commanded_mz": 0.0,
            }
            rospy.loginfo(
                "%s: black rotation start yaw=%.1fdeg target=%.1fdeg",
                self.node_name,
                math.degrees(current_yaw),
                math.degrees(self.black_rotation_angle),
            )

        delta = wrap_angle(current_yaw - self.rotation_state["last_yaw"])
        directed_delta = delta * self.rotation_state["direction"]
        if abs(delta) > self.rotation_feedback_max_delta:
            rospy.logwarn_throttle(
                1.0,
                "%s: ignore abnormal yaw feedback jump %.1fdeg",
                self.node_name,
                math.degrees(delta),
            )
        elif directed_delta > self.rotation_feedback_deadband:
            self.rotation_state["accumulated"] += directed_delta
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

        target = self.make_pose(
            self.move_target.pose.position.x,
            self.move_target.pose.position.y,
            current_yaw,
        )
        remaining = max(
            0.0,
            finish_angle - self.rotation_state["accumulated"],
        )
        desired_mz_magnitude = (
            self.black_rotation_slow_mz
            if remaining <= self.black_rotation_slow_angle
            else abs(self.black_rotation_mz)
        )
        desired_mz = self.rotation_state["direction"] * desired_mz_magnitude
        self.rotation_state["commanded_mz"] = clamp(
            desired_mz,
            self.rotation_state["commanded_mz"] - self.black_rotation_mz_step,
            self.rotation_state["commanded_mz"] + self.black_rotation_mz_step,
        )
        self.publish_lights(0, 0)
        self.publish_pose_cmd(
            MODE_DPROV,
            target,
            mz=self.rotation_state["commanded_mz"],
        )
        rospy.loginfo_throttle(
            1.0,
            "%s: black rotating with point hold accumulated=%.1f/%.1fdeg "
            "remaining=%.1fdeg current_yaw=%.1fdeg MZ=%d",
            self.node_name,
            math.degrees(self.rotation_state["accumulated"]),
            math.degrees(self.black_rotation_angle),
            math.degrees(remaining),
            math.degrees(current_yaw),
            int(self.rotation_state["commanded_mz"]),
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

            if self.step == self.STEP_WAIT_READY:
                hold = self.make_pose(
                    self.start_pose.pose.position.x,
                    self.start_pose.pose.position.y,
                    yaw_from_quaternion(self.start_pose.pose.orientation),
                )
                self.publish_pose_cmd(MODE_DPROV, hold)
                rospy.loginfo_throttle(
                    1.0,
                    "%s: startup point hold elapsed=%.1f/%.1fs camera_ready=%s topic=%s",
                    self.node_name,
                    self.step_elapsed(),
                    self.startup_hold_seconds,
                    self.camera_ready(),
                    self.camera_topic,
                )
                if self.step_elapsed() >= self.startup_hold_seconds and self.camera_ready():
                    self.begin_settle(
                        self.make_pose(
                            self.start_pose.pose.position.x,
                            self.start_pose.pose.position.y,
                            self.initial_search_yaw,
                        ),
                        self.STEP_SEARCH,
                        "startup_heading_alignment",
                    )
            elif self.step == self.STEP_SETTLE:
                self.run_settle()
            elif self.step == self.STEP_SEARCH:
                self.search_marker()
            elif self.step == self.STEP_MOVE:
                if self.move_to_marker():
                    self.begin_settle(
                        self.move_target,
                        self.STEP_LIGHT,
                        "arrived_above_marker",
                    )
            elif self.step == self.STEP_LIGHT:
                if self.run_light_action():
                    if self.marker_kind == "black":
                        self.rotation_state = None
                        self.begin_settle(
                            self.move_target,
                            self.STEP_ROTATE,
                            "light_to_black_rotation",
                        )
                    else:
                        self.begin_settle(
                            self.move_target,
                            self.STEP_FINISH,
                            "return_to_yellow_marker",
                        )
            elif self.step == self.STEP_ROTATE:
                if self.rotate_black():
                    self.begin_settle(
                        self.move_target,
                        self.STEP_FINISH,
                        "return_to_black_marker",
                    )
            elif self.step == self.STEP_FINISH:
                self.finish()

            self.rate.sleep()


def main():
    rospy.init_node("test_task1_v2_yellow_marker")
    Task1MarkerActionTest("test_task1_v2_yellow_marker", "yellow").run()


if __name__ == "__main__":
    main()
