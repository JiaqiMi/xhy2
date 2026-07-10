#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
Task 3 subtask 3 test: inspect the selected pipe area and drop the beacon.

Current mock assumption before the pipe detector is ready:
  target pipe area is 0.50 m in front of the robot and 0.30 m to the left.

This file only uses the newer /auv_actuator_control topic.  The movement target
is still parameterized so the future detector can be connected by switching
target_mode to topic and publishing a geometry_msgs/PoseStamped target.
"""

import math

import rospy
import tf
from auv_control.msg import ActuatorControl
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


NODE_NAME = "test_task3_3_inspect_and_drop"


class Task3InspectAndDropTest:
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
            rospy.get_param("~actuator_topic", "/auv_actuator_control"),
            ActuatorControl,
            queue_size=10,
        )
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(rospy.get_param("~rate", 5.0))

        self.target_mode = rospy.get_param("~target_mode", "mock").strip().lower()
        self.target_topic = rospy.get_param("~target_topic", "/task3/pipeline_target")
        self.target_color = rospy.get_param("~target_color", "yellow").strip().lower()
        self.mock_detected_colors = self.parse_colors(
            rospy.get_param("~mock_detected_colors", ["yellow", "green", "red"])
        )

        self.drop_forward = float(rospy.get_param("~drop_forward", 0.50))
        self.drop_left = float(rospy.get_param("~drop_left", 0.30))
        self.drop_down = float(rospy.get_param("~drop_down", 0.0))

        self.arrive_dist = float(rospy.get_param("~arrive_dist", 0.12))
        self.arrive_yaw = math.radians(float(rospy.get_param("~arrive_yaw_deg", 5.0)))
        self.hold_seconds = float(rospy.get_param("~hold_seconds", 1.0))
        self.open_seconds = float(rospy.get_param("~open_seconds", 3.0))
        self.close_seconds = float(rospy.get_param("~close_seconds", 1.0))

        self.clamp_open = int(rospy.get_param("~clamp_open", 0x00))
        self.clamp_closed = int(rospy.get_param("~clamp_closed", 0xFF))
        self.heading_servo = int(rospy.get_param("~heading_servo", 0x80))
        self.drive_cmd = int(rospy.get_param("~drive_cmd", 0))
        self.drive_speed = int(rospy.get_param("~drive_speed", 0))
        self.light1 = int(rospy.get_param("~light1", 0))
        self.light2 = int(rospy.get_param("~light2", 0))
        self.show_color_light = bool(rospy.get_param("~show_color_light", True))

        self.step = 0
        self.target_pose = None
        self.topic_target_pose = None
        self.step_started = rospy.Time.now()

        self.validate_params()

        if self.target_mode == "topic":
            rospy.Subscriber(self.target_topic, PoseStamped, self.target_callback)

        rospy.loginfo(
            "%s: target_color=%s target_mode=%s mock_offset front=%.2fm left=%.2fm down=%.2fm",
            NODE_NAME,
            self.target_color,
            self.target_mode,
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
        if self.target_mode not in ("mock", "topic"):
            raise ValueError("target_mode must be mock or topic")

        if self.target_color not in self.VALID_COLORS:
            raise ValueError("target_color must be one of yellow, green, red")

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
            rospy.logwarn_throttle(2, "%s: cannot transform target pose: %s", NODE_NAME, error)
            return None

        target.header.stamp = rospy.Time.now()
        target.header.frame_id = "map"
        return target

    def build_mock_drop_target(self):
        """
        Convert the temporary mock pipe position from base_link to map.

        This project uses x=front, y=right, z=down in base_link, so a positive
        drop_left value becomes a negative y offset.
        """
        local_target = PoseStamped()
        local_target.header.stamp = rospy.Time(0)
        local_target.header.frame_id = "base_link"
        local_target.pose.position.x = self.drop_forward
        local_target.pose.position.y = -self.drop_left
        local_target.pose.position.z = self.drop_down
        local_target.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        return self.transform_pose_to_map(local_target)

    def build_topic_drop_target(self):
        if self.topic_target_pose is None:
            rospy.logwarn_throttle(
                2.0,
                "%s: waiting for selected pipeline target on %s",
                NODE_NAME,
                self.target_topic,
            )
            return None

        return self.transform_pose_to_map(self.topic_target_pose)

    def build_drop_target(self):
        if self.target_mode == "mock":
            rospy.loginfo(
                "%s: mock detected pipe colors=%s, selected=%s",
                NODE_NAME,
                ",".join(self.mock_detected_colors),
                self.target_color,
            )
            return self.build_mock_drop_target()

        return self.build_topic_drop_target()

    def is_arrived(self, current):
        pos_error = self.xyz_distance(current, self.target_pose)
        yaw_error = self.yaw_distance(current, self.target_pose)
        rospy.loginfo_throttle(
            1,
            "%s: moving to drop area, pos_error=%.3fm yaw_error=%.2fdeg",
            NODE_NAME,
            pos_error,
            math.degrees(yaw_error),
        )
        return pos_error <= self.arrive_dist and yaw_error <= self.arrive_yaw

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

    def step_elapsed(self):
        return (rospy.Time.now() - self.step_started).to_sec()

    def set_step(self, step):
        self.step = step
        self.step_started = rospy.Time.now()

    def finish_task(self):
        self.publish_actuator(self.clamp_closed, "off")
        self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
        rospy.loginfo("%s: finished", NODE_NAME)
        rospy.signal_shutdown("%s finished" % NODE_NAME)

    def run(self):
        while not rospy.is_shutdown():
            if self.step == 0:
                self.target_pose = self.build_drop_target()
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
                self.set_step(1)

            elif self.step == 1:
                current = self.get_current_pose()
                if current is None:
                    self.rate.sleep()
                    continue

                self.publish_target()
                if self.is_arrived(current):
                    rospy.loginfo("%s: arrived at selected pipe area", NODE_NAME)
                    self.set_step(2)

            elif self.step == 2:
                self.publish_target()
                color = self.target_color if self.show_color_light else "off"
                self.publish_actuator(self.clamp_closed, color)
                if self.step_elapsed() >= self.hold_seconds:
                    rospy.loginfo("%s: opening clamp to drop beacon", NODE_NAME)
                    self.set_step(3)

            elif self.step == 3:
                self.publish_target()
                color = self.target_color if self.show_color_light else "off"
                self.publish_actuator(self.clamp_open, color)
                if self.step_elapsed() >= self.open_seconds:
                    rospy.loginfo("%s: beacon drop window complete, closing clamp", NODE_NAME)
                    self.set_step(4)

            elif self.step == 4:
                self.publish_target()
                self.publish_actuator(self.clamp_closed, "off")
                if self.step_elapsed() >= self.close_seconds:
                    self.finish_task()

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3InspectAndDropTest().run()
    except rospy.ROSInterruptException:
        pass
