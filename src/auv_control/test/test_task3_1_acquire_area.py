#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
Task 3 subtask 1 test: find the arrow and move into the task acquisition frame.

The real Task 3 acquisition area is the 90 x 90 cm frame around the floor arrow.
In topic mode this script no longer uses a fixed front/right offset after seeing
the arrow.  It searches for the arrow, locks a stable detection, transforms the
detected arrow pose to map, and publishes that pose as the navigation target.

Current limitation:
TargetDetection currently only guarantees a 3D position.  If the detector later
publishes the real arrow heading in pose.orientation, set arrow_yaw_mode to
"detection" so the AUV aligns with the arrow direction.  Until then the default
arrow_yaw_mode keeps the current yaw after moving to the arrow.
"""

import math

import rospy
import tf
from auv_control.msg import TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task3_1_acquire_area"

# =========================
# Tunable defaults
# =========================
# These values are the first place to edit during pool/robot debugging.
# They are still ROS params, so roslaunch can override them without editing code.

# Basic input.
DEFAULT_RATE = 5.0
DEFAULT_INPUT_MODE = "mock"  # mock or topic
DEFAULT_ARROW_TOPIC = "/obj/target_message"
DEFAULT_ARROW_CLASS = "arrow"
DEFAULT_MIN_CONFIDENCE = 0.2

# Mock mode only: old temporary target relative to base_link.
# base_link convention in this project: x=front, y=right, z=down.
DEFAULT_ARROW_FORWARD = 0.50
DEFAULT_ARROW_RIGHT = 0.30
DEFAULT_ARROW_DOWN = 0.00

# Search motion.  The arrow is inside a 90 x 90 cm frame, so 0.30 m is a
# conservative first search step.
DEFAULT_SEARCH_STEP = 0.30
DEFAULT_MAX_SEARCH_POINTS = 9
DEFAULT_SCAN_YAW_OFFSETS_DEG = [0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0]
DEFAULT_SCAN_HOLD_SECONDS = 1.5
DEFAULT_MAX_SEARCH_SECONDS = 60.0
DEFAULT_SEARCH_ARRIVE_DIST = 0.15
DEFAULT_SEARCH_ARRIVE_YAW_DEG = 8.0

# Detection lock.  Raise stable_detection_count or lower tolerance if false
# detections are common; lower them if the detector is slow or noisy.
DEFAULT_STABLE_DETECTION_COUNT = 5
DEFAULT_STABLE_POSITION_TOLERANCE = 0.15
DEFAULT_DETECTION_TIMEOUT = 2.0

# Arrow heading.  Use detection only after the detector writes real arrow yaw to
# TargetDetection.pose.orientation.
DEFAULT_ARROW_YAW_MODE = "current"  # current, detection, or fixed
DEFAULT_FIXED_ARROW_YAW_DEG = 0.0

# Final arrival at the arrow/frame.
DEFAULT_ARRIVE_DIST = 0.25
DEFAULT_ARRIVE_YAW_DEG = 8.0
DEFAULT_HOLD_SECONDS = 2.0


class Task3AcquireAreaTest:
    SEARCH_MOVE = 0
    SEARCH_SCAN = 1
    MOVE_TO_ARROW = 2
    HOLD = 3

    def __init__(self):
        self.target_pub = rospy.Publisher("/target", PoseStamped, queue_size=10)
        self.finished_pub = rospy.Publisher("/finished", String, queue_size=10)
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(rospy.get_param("~rate", DEFAULT_RATE))

        self.input_mode = rospy.get_param("~input_mode", DEFAULT_INPUT_MODE).strip().lower()
        self.arrow_topic = rospy.get_param("~arrow_topic", DEFAULT_ARROW_TOPIC)
        self.arrow_class = rospy.get_param("~arrow_class", DEFAULT_ARROW_CLASS).strip().lower()
        self.min_confidence = float(
            rospy.get_param("~min_confidence", DEFAULT_MIN_CONFIDENCE)
        )

        # Mock mode keeps the old fixed parameter target.  Topic mode goes to
        # the detected arrow/frame position instead.
        self.arrow_forward = float(
            rospy.get_param("~arrow_forward", DEFAULT_ARROW_FORWARD)
        )
        self.arrow_right = float(rospy.get_param("~arrow_right", DEFAULT_ARROW_RIGHT))
        self.arrow_down = float(rospy.get_param("~arrow_down", DEFAULT_ARROW_DOWN))

        self.search_step = float(rospy.get_param("~search_step", DEFAULT_SEARCH_STEP))
        self.max_search_points = int(
            rospy.get_param("~max_search_points", DEFAULT_MAX_SEARCH_POINTS)
        )
        self.scan_yaw_offsets_deg = self.parse_float_list(
            rospy.get_param("~scan_yaw_offsets_deg", DEFAULT_SCAN_YAW_OFFSETS_DEG),
            DEFAULT_SCAN_YAW_OFFSETS_DEG,
        )
        self.scan_hold_seconds = float(
            rospy.get_param("~scan_hold_seconds", DEFAULT_SCAN_HOLD_SECONDS)
        )
        self.max_search_seconds = float(
            rospy.get_param("~max_search_seconds", DEFAULT_MAX_SEARCH_SECONDS)
        )
        self.search_arrive_dist = float(
            rospy.get_param("~search_arrive_dist", DEFAULT_SEARCH_ARRIVE_DIST)
        )
        self.search_arrive_yaw = math.radians(
            float(rospy.get_param("~search_arrive_yaw_deg", DEFAULT_SEARCH_ARRIVE_YAW_DEG))
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

        self.arrow_yaw_mode = (
            rospy.get_param("~arrow_yaw_mode", DEFAULT_ARROW_YAW_MODE).strip().lower()
        )
        self.fixed_arrow_yaw_deg = float(
            rospy.get_param("~fixed_arrow_yaw_deg", DEFAULT_FIXED_ARROW_YAW_DEG)
        )

        self.arrive_dist = float(rospy.get_param("~arrive_dist", DEFAULT_ARRIVE_DIST))
        self.arrive_yaw = math.radians(
            float(rospy.get_param("~arrive_yaw_deg", DEFAULT_ARRIVE_YAW_DEG))
        )
        self.hold_seconds = float(rospy.get_param("~hold_seconds", DEFAULT_HOLD_SECONDS))

        self.state = self.SEARCH_MOVE
        self.target_pose = None
        self.search_targets = []
        self.search_point_index = 0
        self.search_yaw_index = 0
        self.scan_started = None
        self.search_started = None
        self.arrive_time = None
        self.arrow_samples = []

        self.validate_params()

        if self.input_mode == "topic":
            rospy.Subscriber(
                self.arrow_topic,
                TargetDetection,
                self.arrow_detection_callback,
                queue_size=10,
            )

        rospy.loginfo(
            "%s: input_mode=%s arrow_topic=%s arrow_class=%s min_confidence=%.2f",
            NODE_NAME,
            self.input_mode,
            self.arrow_topic,
            self.arrow_class,
            self.min_confidence,
        )

    @staticmethod
    def parse_float_list(raw_value, default_value):
        if isinstance(raw_value, (list, tuple)):
            try:
                return [float(value) for value in raw_value]
            except (TypeError, ValueError):
                return default_value

        text = str(raw_value).strip()
        if not text:
            return default_value

        normalized = text.replace(",", " ").replace(";", " ")
        try:
            return [float(part) for part in normalized.split()]
        except ValueError:
            return default_value

    def validate_params(self):
        if self.input_mode not in ("mock", "topic"):
            raise ValueError("input_mode must be mock or topic")

        if self.arrow_yaw_mode not in ("current", "detection", "fixed"):
            raise ValueError("arrow_yaw_mode must be current, detection, or fixed")

        if self.stable_detection_count <= 0:
            raise ValueError("stable_detection_count must be positive")

        if not self.scan_yaw_offsets_deg:
            raise ValueError("scan_yaw_offsets_deg cannot be empty")

    def arrow_detection_callback(self, message):
        if message.class_name.strip().lower() != self.arrow_class:
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

        now = rospy.Time.now()
        self.arrow_samples.append((now, message))
        max_samples = max(self.stable_detection_count * 3, 10)
        if len(self.arrow_samples) > max_samples:
            self.arrow_samples = self.arrow_samples[-max_samples:]

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
        error = Task3AcquireAreaTest.yaw_from_pose(first) - Task3AcquireAreaTest.yaw_from_pose(second)
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
    def quaternion_is_default(quaternion):
        return (
            abs(quaternion.x) < 1e-6
            and abs(quaternion.y) < 1e-6
            and abs(quaternion.z) < 1e-6
            and abs(quaternion.w - 1.0) < 1e-6
        )

    def transform_pose_to_map(self, pose):
        if pose.header.frame_id == "map":
            target = PoseStamped()
            target.header.stamp = rospy.Time.now()
            target.header.frame_id = "map"
            target.pose = pose.pose
            return target

        source = PoseStamped()
        source.header.stamp = rospy.Time(0)
        source.header.frame_id = pose.header.frame_id
        source.pose = pose.pose

        try:
            self.tf_listener.waitForTransform(
                "map", source.header.frame_id, rospy.Time(0), rospy.Duration(1.0)
            )
            target = self.tf_listener.transformPose("map", source)
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0, "%s: cannot transform pose to map: %s", NODE_NAME, error
            )
            return None

        target.header.stamp = rospy.Time.now()
        target.header.frame_id = "map"
        return target

    def transform_base_link_offset_to_map(self, forward, right, down, yaw_offset_deg=0.0):
        local_target = PoseStamped()
        local_target.header.stamp = rospy.Time(0)
        local_target.header.frame_id = "base_link"
        local_target.pose.position.x = forward
        local_target.pose.position.y = right
        local_target.pose.position.z = down
        local_target.pose.orientation = Quaternion(
            *quaternion_from_euler(0.0, 0.0, math.radians(yaw_offset_deg))
        )
        return self.transform_pose_to_map(local_target)

    def build_mock_acquisition_target(self):
        """
        Convert the temporary mock arrow offset from base_link to map.

        In this project base_link uses x=front, y=right, z=down.  Mock mode is
        only for bench tests before perception is running.
        """
        return self.transform_base_link_offset_to_map(
            self.arrow_forward,
            self.arrow_right,
            self.arrow_down,
            0.0,
        )

    def build_search_offsets(self):
        step = self.search_step
        offsets = [
            (0.0, 0.0),
            (step, 0.0),
            (step, -step),
            (0.0, -step),
            (-step, -step),
            (-step, 0.0),
            (-step, step),
            (0.0, step),
            (step, step),
        ]
        return offsets[: self.max_search_points]

    def prepare_search_plan(self):
        self.search_targets = []
        offsets = self.build_search_offsets()
        for forward, right in offsets:
            yaw_targets = []
            for yaw_offset_deg in self.scan_yaw_offsets_deg:
                target = self.transform_base_link_offset_to_map(
                    forward, right, 0.0, yaw_offset_deg
                )
                if target is None:
                    return False
                yaw_targets.append(target)
            self.search_targets.append(yaw_targets)

        self.search_started = rospy.Time.now()
        self.search_point_index = 0
        self.search_yaw_index = 0
        self.scan_started = None
        self.arrow_samples = []
        self.target_pose = self.current_search_target()

        rospy.loginfo(
            "%s: prepared arrow search plan with %d points and yaw offsets %s deg",
            NODE_NAME,
            len(self.search_targets),
            self.scan_yaw_offsets_deg,
        )
        return True

    def current_search_target(self):
        return self.search_targets[self.search_point_index][self.search_yaw_index]

    def search_timed_out(self):
        if self.search_started is None:
            return False
        elapsed = (rospy.Time.now() - self.search_started).to_sec()
        return elapsed >= self.max_search_seconds

    def current_stable_arrow(self):
        now = rospy.Time.now()
        recent = [
            sample
            for sample in self.arrow_samples
            if (now - sample[0]).to_sec() <= self.detection_timeout
        ]
        self.arrow_samples = recent

        if len(recent) < self.stable_detection_count:
            if recent:
                rospy.loginfo_throttle(
                    1.0,
                    "%s: arrow seen, locking stability %d/%d",
                    NODE_NAME,
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
                "%s: arrow detections not stable yet, max_delta=%.3fm > %.3fm",
                NODE_NAME,
                max_distance,
                self.stable_position_tolerance,
            )
            return None

        return latest

    def apply_arrow_orientation(self, target, detection, current):
        if self.arrow_yaw_mode == "current":
            target.pose.orientation = current.pose.orientation
            return target

        if self.arrow_yaw_mode == "fixed":
            target.pose.orientation = Quaternion(
                *quaternion_from_euler(
                    0.0,
                    0.0,
                    math.radians(self.fixed_arrow_yaw_deg),
                )
            )
            return target

        if self.quaternion_is_default(detection.pose.pose.orientation):
            rospy.logwarn_throttle(
                2.0,
                "%s: arrow_yaw_mode=detection but detector orientation is default; keep current yaw",
                NODE_NAME,
            )
            target.pose.orientation = current.pose.orientation
            return target

        return target

    def build_arrow_target_from_detection(self, detection):
        current = self.get_current_pose()
        if current is None:
            return None

        target = self.transform_pose_to_map(detection.pose)
        if target is None:
            return None

        target = self.apply_arrow_orientation(target, detection, current)
        rospy.loginfo(
            "%s: locked arrow target in map x=%.3f y=%.3f z=%.3f yaw_mode=%s",
            NODE_NAME,
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
            self.arrow_yaw_mode,
        )
        return target

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

    def advance_search_target(self):
        self.scan_started = None
        self.arrow_samples = []

        self.search_yaw_index += 1
        if self.search_yaw_index < len(self.scan_yaw_offsets_deg):
            self.target_pose = self.current_search_target()
            self.state = self.SEARCH_MOVE
            rospy.loginfo(
                "%s: scan next yaw offset %.1f deg at search point %d",
                NODE_NAME,
                self.scan_yaw_offsets_deg[self.search_yaw_index],
                self.search_point_index + 1,
            )
            return True

        self.search_point_index += 1
        self.search_yaw_index = 0
        if self.search_point_index < len(self.search_targets):
            self.target_pose = self.current_search_target()
            self.state = self.SEARCH_MOVE
            rospy.loginfo(
                "%s: move to next arrow search point %d/%d",
                NODE_NAME,
                self.search_point_index + 1,
                len(self.search_targets),
            )
            return True

        return False

    def finish_task(self, success=True, reason=""):
        if success:
            self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
            rospy.loginfo("%s: finished", NODE_NAME)
            rospy.signal_shutdown("%s finished" % NODE_NAME)
            return

        message = "%s failed: %s" % (NODE_NAME, reason)
        self.finished_pub.publish(String(data=message))
        rospy.logerr(message)
        rospy.signal_shutdown(message)

    def run_mock_mode(self):
        while not rospy.is_shutdown():
            if self.target_pose is None:
                self.target_pose = self.build_mock_acquisition_target()
                if self.target_pose is None:
                    self.rate.sleep()
                    continue

                rospy.loginfo(
                    "%s: mock acquisition target in map x=%.3f y=%.3f z=%.3f",
                    NODE_NAME,
                    self.target_pose.pose.position.x,
                    self.target_pose.pose.position.y,
                    self.target_pose.pose.position.z,
                )

            current = self.get_current_pose()
            if current is None:
                self.rate.sleep()
                continue

            self.publish_target()
            if self.arrive_time is None:
                if self.is_arrived(current, self.target_pose, label="mock acquisition area"):
                    self.arrive_time = rospy.Time.now()
                    rospy.loginfo("%s: arrived at mock acquisition area", NODE_NAME)
            elif (rospy.Time.now() - self.arrive_time).to_sec() >= self.hold_seconds:
                self.finish_task(success=True)

            self.rate.sleep()

    def run_topic_mode(self):
        while not rospy.is_shutdown():
            if not self.search_targets:
                if not self.prepare_search_plan():
                    self.rate.sleep()
                    continue

            if self.search_timed_out():
                self.finish_task(success=False, reason="arrow not found before timeout")
                return

            stable_arrow = None
            if self.state == self.SEARCH_SCAN:
                stable_arrow = self.current_stable_arrow()

            if stable_arrow is not None:
                target = self.build_arrow_target_from_detection(stable_arrow)
                if target is not None:
                    self.target_pose = target
                    self.state = self.MOVE_TO_ARROW
                    self.arrow_samples = []

            current = self.get_current_pose()
            if current is None:
                self.rate.sleep()
                continue

            if self.state == self.SEARCH_MOVE:
                self.publish_target()
                if self.is_arrived(
                    current,
                    self.target_pose,
                    self.search_arrive_dist,
                    self.search_arrive_yaw,
                    label="arrow search pose",
                ):
                    self.scan_started = rospy.Time.now()
                    self.arrow_samples = []
                    self.state = self.SEARCH_SCAN
                    rospy.loginfo(
                        "%s: scanning arrow at point %d/%d yaw %.1f deg",
                        NODE_NAME,
                        self.search_point_index + 1,
                        len(self.search_targets),
                        self.scan_yaw_offsets_deg[self.search_yaw_index],
                    )

            elif self.state == self.SEARCH_SCAN:
                self.publish_target()
                if self.scan_started is None:
                    self.scan_started = rospy.Time.now()

                elapsed = (rospy.Time.now() - self.scan_started).to_sec()
                if elapsed >= self.scan_hold_seconds:
                    if not self.advance_search_target():
                        self.finish_task(success=False, reason="arrow not found in search area")
                        return

            elif self.state == self.MOVE_TO_ARROW:
                self.publish_target()
                if self.is_arrived(current, self.target_pose, label="arrow acquisition frame"):
                    self.arrive_time = rospy.Time.now()
                    self.state = self.HOLD
                    rospy.loginfo("%s: arrived at arrow acquisition frame", NODE_NAME)

            elif self.state == self.HOLD:
                self.publish_target()
                if (rospy.Time.now() - self.arrive_time).to_sec() >= self.hold_seconds:
                    self.finish_task(success=True)

            self.rate.sleep()

    def run(self):
        if self.input_mode == "mock":
            self.run_mock_mode()
        else:
            self.run_topic_mode()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3AcquireAreaTest().run()
    except rospy.ROSInterruptException:
        pass
