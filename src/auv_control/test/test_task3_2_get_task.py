#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
Task 3 subtask 2 test: get the ArUco task id and show the matching light.

Real ArUco recognition is not ready yet, so this script defaults to a
fixed mock recognition sequence.  The default sequence is 1,3,5,2,4,6.

Task rule mapping:
  1,2 -> yellow
  3,4 -> green
  5,6 -> red

This script only uses the newer /cmd/actuator topic.  When the
vision model is ready, switch input_mode from mock to topic and publish
std_msgs/Int32 marker ids on aruco_topic.

记录：
2026.7.13
  执行器下行话题调整为 /cmd/actuator。
"""

import rospy

from std_msgs.msg import Int32, String

from auv_control.msg import ActuatorControl


NODE_NAME = "test_task3_2_get_task"


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
        self.rate = rospy.Rate(rospy.get_param("~rate", 10.0))

        self.input_mode = rospy.get_param("~input_mode", "mock").strip().lower()
        self.aruco_topic = rospy.get_param("~aruco_topic", "/task3/aruco_id")
        self.max_topic_markers = int(rospy.get_param("~max_topic_markers", 1))
        self.pending_marker_id = None

        self.mock_aruco_ids = self.parse_marker_sequence(
            rospy.get_param("~mock_aruco_ids", [1, 3, 5, 2, 4, 6])
        )
        self.light_seconds = float(rospy.get_param("~light_seconds", 3.0))
        self.gap_seconds = float(rospy.get_param("~gap_seconds", 0.5))

        self.actuator_topic = rospy.get_param("~actuator_topic", "/cmd/actuator")

        self.light1 = int(rospy.get_param("~light1", 0))
        self.light2 = int(rospy.get_param("~light2", 0))

        self.heading_servo = int(rospy.get_param("~heading_servo", 0x80))
        self.clamp_servo = int(rospy.get_param("~clamp_servo", 0xFF))
        self.drive_cmd = int(rospy.get_param("~drive_cmd", 0))
        self.drive_speed = int(rospy.get_param("~drive_speed", 0))

        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.finished_pub = rospy.Publisher("/finished", String, queue_size=10)
        self.mock_index = 0

        if self.input_mode not in ("mock", "topic"):
            raise ValueError("input_mode must be mock or topic")

        if self.input_mode == "mock" and not self.mock_aruco_ids:
            raise ValueError("mock_aruco_ids cannot be empty")

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
        self.pending_marker_id = int(message.data)

    def topic_read_aruco_marker(self):
        marker_id = self.pending_marker_id
        self.pending_marker_id = None
        return marker_id

    def read_aruco_marker(self):
        if self.input_mode == "mock":
            return self.mock_read_aruco_marker()
        return self.topic_read_aruco_marker()

    @classmethod
    def color_for_marker(cls, marker_id):
        if marker_id not in cls.COLOR_BY_MARKER:
            raise ValueError("unsupported ArUco marker id: {}".format(marker_id))
        return cls.COLOR_BY_MARKER[marker_id]

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
                "%s: waiting for ArUco ids on %s",
                NODE_NAME,
                self.aruco_topic,
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
