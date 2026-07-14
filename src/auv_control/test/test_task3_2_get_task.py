#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
Task 3 subtask 2 test: read the ArUco marker id and show the matching light.

This script does not move the robot and does not publish /target.  It only reads
the ArUco id and drives the new /auv_actuator_control light fields.

Default mode reads real ArUco ids from /task3/aruco_id.  Mock mode is still kept
for bench tests without the camera/perception node.

Task rule mapping:
  1,2 -> yellow
  3,4 -> green
  5,6 -> red

This script only uses the newer /auv_actuator_control topic.
"""

import rospy

from std_msgs.msg import Int32, String

from auv_control.msg import ActuatorControl


NODE_NAME = "test_task3_2_get_task"

# =========================
# Tunable defaults
# =========================
# These values are the first place to edit during pool/robot debugging.
# They are still ROS params, so roslaunch can override them without editing code.

DEFAULT_RATE = 10.0
DEFAULT_INPUT_MODE = "topic"  # topic or mock
DEFAULT_ARUCO_TOPIC = "/task3/aruco_id"
DEFAULT_MOCK_ARUCO_IDS = [1, 3, 5, 2, 4, 6]

# max_topic_markers=1 means one real marker is enough for this test.
# Set it to 0 if you want the script to keep responding to markers.
DEFAULT_MAX_TOPIC_MARKERS = 1
DEFAULT_STABLE_MARKER_COUNT = 1
DEFAULT_MARKER_TIMEOUT = 1.0

DEFAULT_LIGHT_SECONDS = 3.0
DEFAULT_GAP_SECONDS = 0.5

DEFAULT_ACTUATOR_TOPIC = "/auv_actuator_control"
DEFAULT_LIGHT1 = 0
DEFAULT_LIGHT2 = 0
DEFAULT_HEADING_SERVO = 0x80
DEFAULT_CLAMP_SERVO = 0xFF
DEFAULT_DRIVE_CMD = 0
DEFAULT_DRIVE_SPEED = 0


class Task3GetTaskTest:
    COLOR_BY_MARKER = {
        1: "yellow",
        2: "yellow",
        3: "green",
        4: "green",
        5: "red",
        6: "red",
    }

    ACTUATOR_LIGHTS = {
        "yellow": (0, 1, 0),
        "green": (0, 0, 1),
        "red": (1, 0, 0),
        "off": (0, 0, 0),
    }

    def __init__(self):
        self.rate = rospy.Rate(rospy.get_param("~rate", DEFAULT_RATE))

        self.input_mode = rospy.get_param("~input_mode", DEFAULT_INPUT_MODE).strip().lower()
        self.aruco_topic = rospy.get_param("~aruco_topic", DEFAULT_ARUCO_TOPIC)
        self.max_topic_markers = int(
            rospy.get_param("~max_topic_markers", DEFAULT_MAX_TOPIC_MARKERS)
        )
        self.stable_marker_count = int(
            rospy.get_param("~stable_marker_count", DEFAULT_STABLE_MARKER_COUNT)
        )
        self.marker_timeout = float(
            rospy.get_param("~marker_timeout", DEFAULT_MARKER_TIMEOUT)
        )
        self.marker_samples = []

        self.mock_aruco_ids = self.parse_marker_sequence(
            rospy.get_param("~mock_aruco_ids", DEFAULT_MOCK_ARUCO_IDS)
        )
        self.light_seconds = float(
            rospy.get_param("~light_seconds", DEFAULT_LIGHT_SECONDS)
        )
        self.gap_seconds = float(rospy.get_param("~gap_seconds", DEFAULT_GAP_SECONDS))

        self.actuator_topic = rospy.get_param("~actuator_topic", DEFAULT_ACTUATOR_TOPIC)

        self.light1 = int(rospy.get_param("~light1", DEFAULT_LIGHT1))
        self.light2 = int(rospy.get_param("~light2", DEFAULT_LIGHT2))

        self.heading_servo = int(rospy.get_param("~heading_servo", DEFAULT_HEADING_SERVO))
        self.clamp_servo = int(rospy.get_param("~clamp_servo", DEFAULT_CLAMP_SERVO))
        self.drive_cmd = int(rospy.get_param("~drive_cmd", DEFAULT_DRIVE_CMD))
        self.drive_speed = int(rospy.get_param("~drive_speed", DEFAULT_DRIVE_SPEED))

        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.finished_pub = rospy.Publisher("/finished", String, queue_size=10)
        self.mock_index = 0

        if self.input_mode not in ("mock", "topic"):
            raise ValueError("input_mode must be mock or topic")

        if self.input_mode == "mock" and not self.mock_aruco_ids:
            raise ValueError("mock_aruco_ids cannot be empty")

        if self.stable_marker_count <= 0:
            raise ValueError("stable_marker_count must be positive")

        if self.input_mode == "topic":
            rospy.Subscriber(self.aruco_topic, Int32, self.aruco_callback)

    @staticmethod
    def parse_marker_sequence(raw_value):
        if isinstance(raw_value, (list, tuple)):
            return [int(value) for value in raw_value]

        if isinstance(raw_value, int):
            return [int(char) for char in str(abs(raw_value))]

        text = str(raw_value).strip()
        if not text:
            return []

        normalized = text.replace(",", " ").replace(";", " ")
        parts = [part for part in normalized.split() if part]
        if len(parts) > 1:
            return [int(part) for part in parts]

        return [int(char) for char in text if char.isdigit()]

    def mock_read_aruco_marker(self):
        if self.mock_index >= len(self.mock_aruco_ids):
            return None

        marker_id = self.mock_aruco_ids[self.mock_index]
        self.mock_index += 1
        return marker_id

    def aruco_callback(self, message):
        marker_id = int(message.data)
        now = rospy.Time.now()
        self.marker_samples.append((now, marker_id))

        max_samples = max(self.stable_marker_count * 3, 10)
        if len(self.marker_samples) > max_samples:
            self.marker_samples = self.marker_samples[-max_samples:]

    def topic_read_aruco_marker(self):
        now = rospy.Time.now()
        recent_samples = [
            sample
            for sample in self.marker_samples
            if (now - sample[0]).to_sec() <= self.marker_timeout
        ]
        self.marker_samples = recent_samples

        if len(recent_samples) < self.stable_marker_count:
            return None

        selected_ids = [
            sample[1] for sample in recent_samples[-self.stable_marker_count :]
        ]
        if len(set(selected_ids)) != 1:
            rospy.loginfo_throttle(
                1.0,
                "%s: ArUco id not stable yet: %s",
                NODE_NAME,
                selected_ids,
            )
            return None

        marker_id = selected_ids[-1]
        self.marker_samples = []
        return marker_id

    def read_aruco_marker(self):
        if self.input_mode == "mock":
            return self.mock_read_aruco_marker()
        return self.topic_read_aruco_marker()

    @classmethod
    def color_for_marker(cls, marker_id):
        return cls.COLOR_BY_MARKER.get(marker_id)

    def publish_lights(self, color):
        red, yellow, green = self.ACTUATOR_LIGHTS[color]

        message = ActuatorControl()
        message.light1 = self.light1
        message.light2 = self.light2
        message.heading_servo = self.heading_servo
        message.clamp_servo = self.clamp_servo
        message.drive_cmd = self.drive_cmd
        message.drive_speed = self.drive_speed
        message.red_light = red
        message.yellow_light = yellow
        message.green_light = green
        self.actuator_pub.publish(message)

    def hold_color(self, color, seconds):
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed >= seconds:
                return
            self.publish_lights(color)
            self.rate.sleep()

    def finish_task(self):
        self.publish_lights("off")
        self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
        rospy.loginfo("%s: finished", NODE_NAME)

    def run(self):
        rospy.sleep(0.5)

        if self.input_mode == "mock":
            rospy.loginfo(
                "%s: mock ArUco sequence=%s",
                NODE_NAME,
                ",".join(str(marker_id) for marker_id in self.mock_aruco_ids),
            )
        else:
            rospy.loginfo(
                "%s: waiting for ArUco ids on %s, stable_marker_count=%d",
                NODE_NAME,
                self.aruco_topic,
                self.stable_marker_count,
            )

        handled_count = 0
        while not rospy.is_shutdown():
            marker_id = self.read_aruco_marker()
            if marker_id is None:
                if self.input_mode == "mock":
                    break
                rospy.logwarn_throttle(
                    2.0,
                    "%s: waiting for ArUco id from %s",
                    NODE_NAME,
                    self.aruco_topic,
                )
                self.rate.sleep()
                continue

            if rospy.is_shutdown():
                return

            color = self.color_for_marker(marker_id)
            if color is None:
                rospy.logwarn(
                    "%s: ignore unsupported ArUco ID %d, expected 1~6",
                    NODE_NAME,
                    marker_id,
                )
                self.rate.sleep()
                continue

            rospy.logwarn(
                "%s: detected ArUco ID %d -> %s light",
                NODE_NAME,
                marker_id,
                color,
            )

            self.hold_color(color, self.light_seconds)
            self.hold_color("off", self.gap_seconds)
            handled_count += 1

            if (
                self.input_mode == "topic"
                and self.max_topic_markers > 0
                and handled_count >= self.max_topic_markers
            ):
                break

        self.finish_task()
        rospy.signal_shutdown("%s complete" % NODE_NAME)


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3GetTaskTest().run()
    except rospy.ROSInterruptException:
        pass
