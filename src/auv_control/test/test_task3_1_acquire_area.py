#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
Task 3 subtask 1 test: move to the task acquisition area.

This script keeps the same style as the existing ROS test scripts:
it reads the current AUV pose from TF, publishes a target PoseStamped to
/target, and publishes /finished when the target is reached.

Temporary assumption before the arrow detector is ready:
the arrow result is hard-coded as a local target in base_link:
  x = +0.50 m  (front)
  y = +0.30 m  (right, according to this project's base_link convention)
  z =  0.00 m  (keep current depth)
"""

import math

import rospy
import tf
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


NODE_NAME = "test_task3_1_acquire_area"


class Task3AcquireAreaTest:
    def __init__(self):
        self.target_pub = rospy.Publisher("/target", PoseStamped, queue_size=10)
        self.finished_pub = rospy.Publisher("/finished", String, queue_size=10)
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(rospy.get_param("~rate", 5.0))

        # Hard-coded arrow result. Keep them as ROS params so they can be tuned
        # from rosrun/roslaunch without editing this file.
        self.arrow_forward = rospy.get_param("~arrow_forward", 0.50)
        self.arrow_right = rospy.get_param("~arrow_right", 0.30)
        self.arrow_down = rospy.get_param("~arrow_down", 0.0)

        self.arrive_dist = rospy.get_param("~arrive_dist", 0.12)
        self.arrive_yaw = math.radians(rospy.get_param("~arrive_yaw_deg", 5.0))
        self.hold_seconds = rospy.get_param("~hold_seconds", 2.0)

        self.target_pose = None
        self.arrive_time = None
        self.step = 0

        rospy.loginfo(
            "%s: arrow offset front=%.2fm right=%.2fm down=%.2fm",
            NODE_NAME,
            self.arrow_forward,
            self.arrow_right,
            self.arrow_down,
        )

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

    def build_acquisition_target(self):
        """
        Convert the hard-coded arrow result from base_link to map.

        In this project base_link uses x=front, y=right, z=down.  By creating
        the local target in base_link and transforming it through TF, the target
        remains correct no matter what yaw the AUV currently has.
        """
        local_target = PoseStamped()
        local_target.header.stamp = rospy.Time(0)
        local_target.header.frame_id = "base_link"
        local_target.pose.position.x = self.arrow_forward
        local_target.pose.position.y = self.arrow_right
        local_target.pose.position.z = self.arrow_down
        local_target.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)

        try:
            self.tf_listener.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(1.0)
            )
            target = self.tf_listener.transformPose("map", local_target)
        except tf.Exception as error:
            rospy.logwarn_throttle(2, "%s: cannot transform arrow target: %s", NODE_NAME, error)
            return None

        target.header.stamp = rospy.Time.now()
        target.header.frame_id = "map"
        return target

    def is_arrived(self, current):
        pos_error = self.xyz_distance(current, self.target_pose)
        yaw_error = self.yaw_distance(current, self.target_pose)
        rospy.loginfo_throttle(
            1,
            "%s: moving to acquisition area, pos_error=%.3fm yaw_error=%.2fdeg",
            NODE_NAME,
            pos_error,
            math.degrees(yaw_error),
        )
        return pos_error <= self.arrive_dist and yaw_error <= self.arrive_yaw

    def publish_finished(self):
        self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
        rospy.loginfo("%s: finished", NODE_NAME)
        rospy.signal_shutdown("%s finished" % NODE_NAME)

    def run(self):
        while not rospy.is_shutdown():
            if self.step == 0:
                self.target_pose = self.build_acquisition_target()
                if self.target_pose is None:
                    self.rate.sleep()
                    continue

                rospy.loginfo(
                    "%s: fixed acquisition target in map x=%.3f y=%.3f z=%.3f",
                    NODE_NAME,
                    self.target_pose.pose.position.x,
                    self.target_pose.pose.position.y,
                    self.target_pose.pose.position.z,
                )
                self.step = 1

            elif self.step == 1:
                current = self.get_current_pose()
                if current is None:
                    self.rate.sleep()
                    continue

                self.target_pose.header.stamp = rospy.Time.now()
                self.target_pub.publish(self.target_pose)

                if self.is_arrived(current):
                    self.arrive_time = rospy.Time.now()
                    rospy.loginfo("%s: arrived, holding position", NODE_NAME)
                    self.step = 2

            elif self.step == 2:
                self.target_pose.header.stamp = rospy.Time.now()
                self.target_pub.publish(self.target_pose)
                if (rospy.Time.now() - self.arrive_time).to_sec() >= self.hold_seconds:
                    self.publish_finished()

            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3AcquireAreaTest().run()
    except rospy.ROSInterruptException:
        pass
