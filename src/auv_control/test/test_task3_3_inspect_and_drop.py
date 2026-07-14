#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
Task 3 subtask 3 test: find the selected colored frame and drop the beacon.

Detection mode logic:
  1. read the target colored frame from /obj/target_message;
  2. lock several stable detections;
  3. move roughly near the detected frame center;
  4. keep reading vision and fine-align the robot above the frame center;
  5. stop, open the clamp, close the clamp, and finish.

This script uses /target for motion and the newer /cmd/actuator topic for
lights and clamp control.

记录：
2026.7.13
  执行器下行话题调整为 /cmd/actuator。
"""

import math

import rospy
import tf
from auv_control.msg import ActuatorControl, TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


NODE_NAME = "test_task3_3_inspect_and_drop"

# =========================
# Tunable defaults
# =========================
# These values are the first place to edit during pool/robot debugging.
# They are still ROS params, so roslaunch can override them without editing code.

DEFAULT_RATE = 5.0
DEFAULT_TARGET_MODE = "mock"  # mock, topic, or detection
DEFAULT_TARGET_TOPIC = "/task3/pipeline_target"
DEFAULT_DETECTION_TOPIC = "/obj/target_message"
DEFAULT_DETECTION_FRAME = "camera"
DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_TARGET_COLOR = "yellow"
DEFAULT_MOCK_DETECTED_COLORS = ["yellow", "green", "red"]

# Mock mode only: old temporary drop point relative to base_link.
# base_link convention in this project: x=front, y=right, z=down.
DEFAULT_DROP_FORWARD = 0.50
DEFAULT_DROP_LEFT = 0.30
DEFAULT_DROP_DOWN = 0.00

# Simple mock/topic arrival tolerance.
DEFAULT_ARRIVE_DIST = 0.12
DEFAULT_ARRIVE_YAW_DEG = 5.0

# Detection lock before coarse movement.
DEFAULT_STABLE_DETECTION_COUNT = 5
DEFAULT_STABLE_POSITION_TOLERANCE = 0.15
DEFAULT_DETECTION_TIMEOUT = 2.0
DEFAULT_MAX_DETECTION_WAIT_SECONDS = 60.0

# Coarse movement to the detected colored-frame center.
DEFAULT_COARSE_ARRIVE_DIST = 0.18
DEFAULT_COARSE_ARRIVE_YAW_DEG = 8.0

# Fine visual alignment above the colored frame.
DEFAULT_FINE_TOLERANCE_X = 0.08
DEFAULT_FINE_TOLERANCE_Y = 0.08
DEFAULT_FINE_MAX_STEP = 0.10
DEFAULT_FINE_MIN_STEP = 0.03
DEFAULT_FINE_GAIN = 0.8
DEFAULT_FINE_COMMAND_PERIOD = 0.4
DEFAULT_FINE_HOLD_SECONDS = 1.0
DEFAULT_FRAME_LOST_TIMEOUT = 2.0

# Drop action.
DEFAULT_HOLD_SECONDS = 1.0
DEFAULT_OPEN_SECONDS = 3.0
DEFAULT_CLOSE_SECONDS = 1.0

# Actuator defaults.
DEFAULT_ACTUATOR_TOPIC = "/cmd/actuator"
DEFAULT_CLAMP_OPEN = 0x00
DEFAULT_CLAMP_CLOSED = 0xFF
DEFAULT_HEADING_SERVO = 0x80
DEFAULT_DRIVE_CMD = 0
DEFAULT_DRIVE_SPEED = 0
DEFAULT_LIGHT1 = 0
DEFAULT_LIGHT2 = 0
DEFAULT_SHOW_COLOR_LIGHT = True


class Task3InspectAndDropTest:
    LOCK_FRAME = 0
    MOVE_NEAR_FRAME = 1
    FINE_ALIGN_FRAME = 2
    HOLD_BEFORE_DROP = 3
    OPEN_CLAMP = 4
    CLOSE_CLAMP = 5

    COLOR_LIGHTS = {
        "yellow": (0, 1, 0),
        "green": (0, 0, 1),
        "red": (1, 0, 0),
        "off": (0, 0, 0),
    }

    VALID_COLORS = ("yellow", "green", "red")

    def __init__(self):
        self.target_pub = rospy.Publisher("/target", PoseStamped, queue_size=10)
        self.finished_pub = rospy.Publisher("/finished", String, queue_size=10)
        self.actuator_pub = rospy.Publisher(
            rospy.get_param("~actuator_topic", DEFAULT_ACTUATOR_TOPIC),
            ActuatorControl,
            queue_size=10,
        )
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(rospy.get_param("~rate", DEFAULT_RATE))

        self.target_mode = (
            rospy.get_param("~target_mode", DEFAULT_TARGET_MODE).strip().lower()
        )
        self.target_topic = rospy.get_param("~target_topic", DEFAULT_TARGET_TOPIC)
        self.detection_topic = rospy.get_param(
            "~detection_topic", DEFAULT_DETECTION_TOPIC
        )
        self.detection_frame = rospy.get_param(
            "~detection_frame", DEFAULT_DETECTION_FRAME
        )
        self.min_confidence = float(
            rospy.get_param("~min_confidence", DEFAULT_MIN_CONFIDENCE)
        )
        self.target_color = (
            rospy.get_param("~target_color", DEFAULT_TARGET_COLOR).strip().lower()
        )
        self.mock_detected_colors = self.parse_colors(
            rospy.get_param("~mock_detected_colors", DEFAULT_MOCK_DETECTED_COLORS)
        )

        self.drop_forward = float(
            rospy.get_param("~drop_forward", DEFAULT_DROP_FORWARD)
        )
        self.drop_left = float(rospy.get_param("~drop_left", DEFAULT_DROP_LEFT))
        self.drop_down = float(rospy.get_param("~drop_down", DEFAULT_DROP_DOWN))

        self.arrive_dist = float(rospy.get_param("~arrive_dist", DEFAULT_ARRIVE_DIST))
        self.arrive_yaw = math.radians(
            float(rospy.get_param("~arrive_yaw_deg", DEFAULT_ARRIVE_YAW_DEG))
        )

        self.stable_detection_count = int(
            rospy.get_param("~stable_detection_count", DEFAULT_STABLE_DETECTION_COUNT)
        )
        self.stable_position_tolerance = float(
            rospy.get_param(
                "~stable_position_tolerance", DEFAULT_STABLE_POSITION_TOLERANCE
            )
        )
        self.detection_timeout = float(
            rospy.get_param("~detection_timeout", DEFAULT_DETECTION_TIMEOUT)
        )
        self.max_detection_wait_seconds = float(
            rospy.get_param(
                "~max_detection_wait_seconds", DEFAULT_MAX_DETECTION_WAIT_SECONDS
            )
        )

        self.coarse_arrive_dist = float(
            rospy.get_param("~coarse_arrive_dist", DEFAULT_COARSE_ARRIVE_DIST)
        )
        self.coarse_arrive_yaw = math.radians(
            float(
                rospy.get_param(
                    "~coarse_arrive_yaw_deg", DEFAULT_COARSE_ARRIVE_YAW_DEG
                )
            )
        )

        self.fine_tolerance_x = float(
            rospy.get_param("~fine_tolerance_x", DEFAULT_FINE_TOLERANCE_X)
        )
        self.fine_tolerance_y = float(
            rospy.get_param("~fine_tolerance_y", DEFAULT_FINE_TOLERANCE_Y)
        )
        self.fine_max_step = float(
            rospy.get_param("~fine_max_step", DEFAULT_FINE_MAX_STEP)
        )
        self.fine_min_step = float(
            rospy.get_param("~fine_min_step", DEFAULT_FINE_MIN_STEP)
        )
        self.fine_gain = float(rospy.get_param("~fine_gain", DEFAULT_FINE_GAIN))
        self.fine_command_period = float(
            rospy.get_param("~fine_command_period", DEFAULT_FINE_COMMAND_PERIOD)
        )
        self.fine_hold_seconds = float(
            rospy.get_param("~fine_hold_seconds", DEFAULT_FINE_HOLD_SECONDS)
        )
        self.frame_lost_timeout = float(
            rospy.get_param("~frame_lost_timeout", DEFAULT_FRAME_LOST_TIMEOUT)
        )

        self.hold_seconds = float(rospy.get_param("~hold_seconds", DEFAULT_HOLD_SECONDS))
        self.open_seconds = float(rospy.get_param("~open_seconds", DEFAULT_OPEN_SECONDS))
        self.close_seconds = float(
            rospy.get_param("~close_seconds", DEFAULT_CLOSE_SECONDS)
        )

        self.clamp_open = int(rospy.get_param("~clamp_open", DEFAULT_CLAMP_OPEN))
        self.clamp_closed = int(
            rospy.get_param("~clamp_closed", DEFAULT_CLAMP_CLOSED)
        )
        self.heading_servo = int(
            rospy.get_param("~heading_servo", DEFAULT_HEADING_SERVO)
        )
        self.drive_cmd = int(rospy.get_param("~drive_cmd", DEFAULT_DRIVE_CMD))
        self.drive_speed = int(rospy.get_param("~drive_speed", DEFAULT_DRIVE_SPEED))
        self.light1 = int(rospy.get_param("~light1", DEFAULT_LIGHT1))
        self.light2 = int(rospy.get_param("~light2", DEFAULT_LIGHT2))
        self.show_color_light = bool(
            rospy.get_param("~show_color_light", DEFAULT_SHOW_COLOR_LIGHT)
        )

        self.state = self.LOCK_FRAME
        self.target_pose = None
        self.topic_target_pose = None
        self.detection_samples = []
        self.state_started = rospy.Time.now()
        self.search_started = rospy.Time.now()
        self.align_started = None
        self.frame_lost_started = None
        self.last_fine_command_time = rospy.Time(0)

        self.validate_params()

        if self.target_mode == "topic":
            rospy.Subscriber(self.target_topic, PoseStamped, self.target_callback)
        elif self.target_mode == "detection":
            rospy.Subscriber(
                self.detection_topic,
                TargetDetection,
                self.detection_callback,
                queue_size=10,
            )

        rospy.loginfo(
            "%s: target_color=%s target_mode=%s detection_topic=%s mock_offset front=%.2fm left=%.2fm down=%.2fm",
            NODE_NAME,
            self.target_color,
            self.target_mode,
            self.detection_topic,
            self.drop_forward,
            self.drop_left,
            self.drop_down,
        )

    @staticmethod
    def parse_colors(raw_value):
        if isinstance(raw_value, (list, tuple)):
            return [str(value).strip().lower() for value in raw_value]

        text = str(raw_value).strip().lower()
        if not text:
            return []

        normalized = text.replace(",", " ").replace(";", " ")
        return [part for part in normalized.split() if part]

    def validate_params(self):
        if self.target_mode not in ("mock", "topic", "detection"):
            raise ValueError("target_mode must be mock, topic, or detection")

        if self.target_color not in self.VALID_COLORS:
            raise ValueError("target_color must be one of yellow, green, red")

        if self.stable_detection_count <= 0:
            raise ValueError("stable_detection_count must be positive")

        if self.fine_min_step > self.fine_max_step:
            raise ValueError("fine_min_step cannot be larger than fine_max_step")

        unknown_colors = [
            color for color in self.mock_detected_colors if color not in self.VALID_COLORS
        ]
        if unknown_colors:
            raise ValueError("unsupported mock_detected_colors: {}".format(unknown_colors))

        if self.target_mode == "mock" and self.target_color not in self.mock_detected_colors:
            raise ValueError(
                "target_color {} is not in mock_detected_colors {}".format(
                    self.target_color, self.mock_detected_colors
                )
            )

    def target_callback(self, message):
        self.topic_target_pose = message

    def detection_callback(self, message):
        detected_color = message.class_name.strip().lower()
        if detected_color != self.target_color:
            return

        if message.conf < self.min_confidence:
            rospy.logwarn_throttle(
                2.0,
                "%s: ignore %s detection with low conf %.2f < %.2f",
                NODE_NAME,
                message.class_name,
                message.conf,
                self.min_confidence,
            )
            return

        if not message.pose.header.frame_id:
            message.pose.header.frame_id = self.detection_frame

        self.detection_samples.append((rospy.Time.now(), message))
        max_samples = max(self.stable_detection_count * 3, 10)
        if len(self.detection_samples) > max_samples:
            self.detection_samples = self.detection_samples[-max_samples:]

    def get_current_pose(self):
        try:
            self.tf_listener.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(1.0)
            )
            trans, rot = self.tf_listener.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(2, "%s: cannot get current pose: %s", NODE_NAME, error)
            return None

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = "map"
        pose.pose.position = Point(*trans)
        pose.pose.orientation = Quaternion(*rot)
        return pose

    @staticmethod
    def yaw_from_pose(pose):
        q = pose.pose.orientation
        return euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

    @staticmethod
    def yaw_distance(first, second):
        error = (
            Task3InspectAndDropTest.yaw_from_pose(first)
            - Task3InspectAndDropTest.yaw_from_pose(second)
        )
        return abs((error + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def xyz_distance(first, second):
        dx = first.pose.position.x - second.pose.position.x
        dy = first.pose.position.y - second.pose.position.y
        dz = first.pose.position.z - second.pose.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def detection_position_distance(first, second):
        p1 = first.pose.pose.position
        p2 = second.pose.pose.position
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def bounded_step(error, tolerance, min_step, max_step, gain):
        if abs(error) <= tolerance:
            return 0.0

        step = min(abs(error) * gain, max_step)
        step = max(step, min_step)
        return math.copysign(step, error)

    def transform_pose_to_frame(self, pose, target_frame):
        source = PoseStamped()
        source.header.stamp = rospy.Time(0)
        source.header.frame_id = pose.header.frame_id or self.detection_frame
        source.pose = pose.pose

        if source.header.frame_id == target_frame:
            result = PoseStamped()
            result.header.stamp = rospy.Time.now()
            result.header.frame_id = target_frame
            result.pose = source.pose
            return result

        try:
            self.tf_listener.waitForTransform(
                target_frame,
                source.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            result = self.tf_listener.transformPose(target_frame, source)
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0,
                "%s: cannot transform pose from %s to %s: %s",
                NODE_NAME,
                source.header.frame_id,
                target_frame,
                error,
            )
            return None

        result.header.stamp = rospy.Time.now()
        result.header.frame_id = target_frame
        return result

    def transform_pose_to_map(self, pose):
        return self.transform_pose_to_frame(pose, "map")

    def transform_base_link_offset_to_map(self, forward, right, down):
        local_target = PoseStamped()
        local_target.header.stamp = rospy.Time(0)
        local_target.header.frame_id = "base_link"
        local_target.pose.position.x = forward
        local_target.pose.position.y = right
        local_target.pose.position.z = down
        local_target.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        return self.transform_pose_to_map(local_target)

    def build_mock_drop_target(self):
        """
        Convert the temporary mock pipe position from base_link to map.

        This project uses x=front, y=right, z=down in base_link, so a positive
        drop_left value becomes a negative y offset.
        """
        return self.transform_base_link_offset_to_map(
            self.drop_forward,
            -self.drop_left,
            self.drop_down,
        )

    def build_topic_drop_target(self):
        if self.topic_target_pose is None:
            rospy.logwarn_throttle(
                2.0,
                "%s: waiting for selected pipeline target on %s",
                NODE_NAME,
                self.target_topic,
            )
            return None

        target = self.transform_pose_to_map(self.topic_target_pose)
        current = self.get_current_pose()
        if target is not None and current is not None:
            target.pose.orientation = current.pose.orientation
        return target

    def build_simple_drop_target(self):
        if self.target_mode == "mock":
            rospy.loginfo(
                "%s: mock detected pipe colors=%s, selected=%s",
                NODE_NAME,
                ",".join(self.mock_detected_colors),
                self.target_color,
            )
            return self.build_mock_drop_target()

        return self.build_topic_drop_target()

    def recent_detection_samples(self):
        now = rospy.Time.now()
        recent = [
            sample
            for sample in self.detection_samples
            if (now - sample[0]).to_sec() <= self.detection_timeout
        ]
        self.detection_samples = recent
        return recent

    def latest_detection(self):
        recent = self.recent_detection_samples()
        if not recent:
            return None
        return recent[-1][1]

    def current_stable_detection(self):
        recent = self.recent_detection_samples()
        if len(recent) < self.stable_detection_count:
            if recent:
                rospy.loginfo_throttle(
                    1.0,
                    "%s: %s frame seen, locking stability %d/%d",
                    NODE_NAME,
                    self.target_color,
                    len(recent),
                    self.stable_detection_count,
                )
            return None

        selected = [sample[1] for sample in recent[-self.stable_detection_count :]]
        latest = selected[-1]
        max_distance = max(
            self.detection_position_distance(sample, latest) for sample in selected
        )
        if max_distance > self.stable_position_tolerance:
            rospy.loginfo_throttle(
                1.0,
                "%s: %s frame not stable yet, max_delta=%.3fm > %.3fm",
                NODE_NAME,
                self.target_color,
                max_distance,
                self.stable_position_tolerance,
            )
            return None

        return latest

    def build_coarse_target_from_detection(self, detection):
        target = self.transform_pose_to_map(detection.pose)
        current = self.get_current_pose()
        if target is None or current is None:
            return None

        target.pose.orientation = current.pose.orientation
        rospy.loginfo(
            "%s: locked %s frame target in map x=%.3f y=%.3f z=%.3f",
            NODE_NAME,
            self.target_color,
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
        )
        return target

    def build_fine_alignment_target(self, detection):
        frame_in_base = self.transform_pose_to_frame(detection.pose, "base_link")
        if frame_in_base is None:
            return None, None

        error_x = frame_in_base.pose.position.x
        error_y = frame_in_base.pose.position.y
        step_x = self.bounded_step(
            error_x,
            self.fine_tolerance_x,
            self.fine_min_step,
            self.fine_max_step,
            self.fine_gain,
        )
        step_y = self.bounded_step(
            error_y,
            self.fine_tolerance_y,
            self.fine_min_step,
            self.fine_max_step,
            self.fine_gain,
        )

        if step_x == 0.0 and step_y == 0.0:
            current = self.get_current_pose()
            return current, (error_x, error_y)

        target = self.transform_base_link_offset_to_map(step_x, step_y, 0.0)
        return target, (error_x, error_y)

    def is_arrived(self, current, target, max_dist=None, max_yaw=None, label="target"):
        max_dist = self.arrive_dist if max_dist is None else max_dist
        max_yaw = self.arrive_yaw if max_yaw is None else max_yaw
        pos_error = self.xyz_distance(current, target)
        yaw_error = self.yaw_distance(current, target)
        rospy.loginfo_throttle(
            1.0,
            "%s: moving to %s, pos_error=%.3fm yaw_error=%.2fdeg",
            NODE_NAME,
            label,
            pos_error,
            math.degrees(yaw_error),
        )
        return pos_error <= max_dist and yaw_error <= max_yaw

    def publish_target(self):
        if self.target_pose is None:
            return
        self.target_pose.header.stamp = rospy.Time.now()
        self.target_pub.publish(self.target_pose)

    def publish_actuator(self, clamp_servo, color=None):
        red, yellow, green = self.COLOR_LIGHTS[color or "off"]

        message = ActuatorControl()
        message.light1 = self.light1
        message.light2 = self.light2
        message.heading_servo = self.heading_servo
        message.clamp_servo = int(clamp_servo)
        message.drive_cmd = self.drive_cmd
        message.drive_speed = self.drive_speed
        message.red_light = red
        message.yellow_light = yellow
        message.green_light = green
        self.actuator_pub.publish(message)

    def active_color(self):
        return self.target_color if self.show_color_light else "off"

    def state_elapsed(self):
        return (rospy.Time.now() - self.state_started).to_sec()

    def set_state(self, state):
        self.state = state
        self.state_started = rospy.Time.now()

    def finish_task(self, success=True, reason=""):
        self.publish_actuator(self.clamp_closed, "off")
        if success:
            self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
            rospy.loginfo("%s: finished", NODE_NAME)
            rospy.signal_shutdown("%s finished" % NODE_NAME)
            return

        message = "%s failed: %s" % (NODE_NAME, reason)
        self.finished_pub.publish(String(data=message))
        rospy.logerr(message)
        rospy.signal_shutdown(message)

    def run_simple_mode(self):
        while not rospy.is_shutdown():
            if self.state == self.LOCK_FRAME:
                self.target_pose = self.build_simple_drop_target()
                if self.target_pose is None:
                    self.rate.sleep()
                    continue

                rospy.loginfo(
                    "%s: fixed drop target in map x=%.3f y=%.3f z=%.3f",
                    NODE_NAME,
                    self.target_pose.pose.position.x,
                    self.target_pose.pose.position.y,
                    self.target_pose.pose.position.z,
                )
                self.set_state(self.MOVE_NEAR_FRAME)

            elif self.state == self.MOVE_NEAR_FRAME:
                current = self.get_current_pose()
                if current is None:
                    self.rate.sleep()
                    continue

                self.publish_target()
                if self.is_arrived(current, self.target_pose, label="drop area"):
                    rospy.loginfo("%s: arrived at selected pipe area", NODE_NAME)
                    self.set_state(self.HOLD_BEFORE_DROP)

            elif self.state == self.HOLD_BEFORE_DROP:
                self.publish_target()
                self.publish_actuator(self.clamp_closed, self.active_color())
                if self.state_elapsed() >= self.hold_seconds:
                    rospy.loginfo("%s: opening clamp to drop beacon", NODE_NAME)
                    self.set_state(self.OPEN_CLAMP)

            elif self.state == self.OPEN_CLAMP:
                self.publish_target()
                self.publish_actuator(self.clamp_open, self.active_color())
                if self.state_elapsed() >= self.open_seconds:
                    rospy.loginfo("%s: beacon drop window complete, closing clamp", NODE_NAME)
                    self.set_state(self.CLOSE_CLAMP)

            elif self.state == self.CLOSE_CLAMP:
                self.publish_target()
                self.publish_actuator(self.clamp_closed, "off")
                if self.state_elapsed() >= self.close_seconds:
                    self.finish_task(success=True)

            self.rate.sleep()

    def run_detection_mode(self):
        while not rospy.is_shutdown():
            self.publish_actuator(self.clamp_closed, self.active_color())

            if (
                self.max_detection_wait_seconds > 0.0
                and (rospy.Time.now() - self.search_started).to_sec()
                >= self.max_detection_wait_seconds
                and self.state == self.LOCK_FRAME
            ):
                self.finish_task(success=False, reason="colored frame not found before timeout")
                return

            if self.state == self.LOCK_FRAME:
                detection = self.current_stable_detection()
                if detection is None:
                    rospy.logwarn_throttle(
                        2.0,
                        "%s: waiting for stable %s frame on %s",
                        NODE_NAME,
                        self.target_color,
                        self.detection_topic,
                    )
                    self.rate.sleep()
                    continue

                self.target_pose = self.build_coarse_target_from_detection(detection)
                if self.target_pose is None:
                    self.rate.sleep()
                    continue

                self.set_state(self.MOVE_NEAR_FRAME)

            elif self.state == self.MOVE_NEAR_FRAME:
                current = self.get_current_pose()
                if current is None:
                    self.rate.sleep()
                    continue

                self.publish_target()
                if self.is_arrived(
                    current,
                    self.target_pose,
                    self.coarse_arrive_dist,
                    self.coarse_arrive_yaw,
                    label="%s frame coarse target" % self.target_color,
                ):
                    rospy.loginfo(
                        "%s: arrived near %s frame, start fine visual alignment",
                        NODE_NAME,
                        self.target_color,
                    )
                    self.align_started = None
                    self.frame_lost_started = None
                    self.last_fine_command_time = rospy.Time(0)
                    self.set_state(self.FINE_ALIGN_FRAME)

            elif self.state == self.FINE_ALIGN_FRAME:
                detection = self.latest_detection()
                if detection is None:
                    if self.frame_lost_started is None:
                        self.frame_lost_started = rospy.Time.now()

                    self.publish_target()
                    lost_seconds = (rospy.Time.now() - self.frame_lost_started).to_sec()
                    rospy.logwarn_throttle(
                        1.0,
                        "%s: lost %s frame during fine alignment for %.1fs",
                        NODE_NAME,
                        self.target_color,
                        lost_seconds,
                    )
                    if lost_seconds >= self.frame_lost_timeout:
                        rospy.logwarn(
                            "%s: return to frame lock because %s frame was lost",
                            NODE_NAME,
                            self.target_color,
                        )
                        self.detection_samples = []
                        self.target_pose = None
                        self.align_started = None
                        self.frame_lost_started = None
                        self.search_started = rospy.Time.now()
                        self.set_state(self.LOCK_FRAME)
                    self.rate.sleep()
                    continue

                self.frame_lost_started = None
                now = rospy.Time.now()
                if (
                    self.target_pose is None
                    or (now - self.last_fine_command_time).to_sec()
                    >= self.fine_command_period
                ):
                    target, error_xy = self.build_fine_alignment_target(detection)
                    if target is None or error_xy is None:
                        self.rate.sleep()
                        continue

                    error_x, error_y = error_xy
                    self.target_pose = target
                    self.last_fine_command_time = now

                    rospy.loginfo_throttle(
                        0.5,
                        "%s: fine align %s frame error_x=%.3fm error_y=%.3fm",
                        NODE_NAME,
                        self.target_color,
                        error_x,
                        error_y,
                    )

                    aligned = (
                        abs(error_x) <= self.fine_tolerance_x
                        and abs(error_y) <= self.fine_tolerance_y
                    )
                    if aligned:
                        if self.align_started is None:
                            self.align_started = rospy.Time.now()
                        elif (
                            rospy.Time.now() - self.align_started
                        ).to_sec() >= self.fine_hold_seconds:
                            rospy.loginfo(
                                "%s: fine alignment stable, ready to drop beacon",
                                NODE_NAME,
                            )
                            self.set_state(self.HOLD_BEFORE_DROP)
                    else:
                        self.align_started = None

                self.publish_target()

            elif self.state == self.HOLD_BEFORE_DROP:
                self.publish_target()
                self.publish_actuator(self.clamp_closed, self.active_color())
                if self.state_elapsed() >= self.hold_seconds:
                    rospy.loginfo("%s: opening clamp to drop beacon", NODE_NAME)
                    self.set_state(self.OPEN_CLAMP)

            elif self.state == self.OPEN_CLAMP:
                self.publish_target()
                self.publish_actuator(self.clamp_open, self.active_color())
                if self.state_elapsed() >= self.open_seconds:
                    rospy.loginfo("%s: beacon drop window complete, closing clamp", NODE_NAME)
                    self.set_state(self.CLOSE_CLAMP)

            elif self.state == self.CLOSE_CLAMP:
                self.publish_target()
                self.publish_actuator(self.clamp_closed, "off")
                if self.state_elapsed() >= self.close_seconds:
                    self.finish_task(success=True)

            self.rate.sleep()

    def run(self):
        if self.target_mode == "detection":
            self.run_detection_mode()
        else:
            self.run_simple_mode()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3InspectAndDropTest().run()
    except rospy.ROSInterruptException:
        pass
