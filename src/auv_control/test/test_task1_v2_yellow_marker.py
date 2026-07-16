#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v2_yellow_marker.py
功能：Task1 黄色图形单项测试。

流程：
    1. 定点保持启动位姿，等待相机节点持续发布图像；
    2. 未识别到黄色图形时，定点向初始航向前进 0.5 m，静止后左右旋转 30 度搜索；
    3. 连续多帧确认黄色图形位置后，使用动力定位模式直接前往图形中心；
    4. 前往图形时只判断水平位置误差，不规划或校正目标航向；
    5. 到达黄色图形上方后亮红灯，动作完成后发布完成消息。

监听：/obj/target_message，/left/image_raw，/status/vel（可选），/tf
发布：/cmd/pose/ned，/cmd/actuator，/finished

记录：
2026.7.14
    初版，用于单独验证 Task1 图形识别、前往图形和黄色动作流程。
2026.7.16
    将脚本改为纯黄色图形测试，删除黑色图形识别、旋转和无关状态。
    搜索基准航向直接读取节点启动时的机器人当前航向，不再使用人工配置参数。
    锁定黄色图形后直接发布其中心为动力定位目标，只以 XY 距离判断到达。
    未识别时采用“前进 0.5 m、左转 30 度、右转 30 度”的定点搜索流程。
    精简运行日志，只保留相机状态、识别状态、当前位置、目标位置、距离和完成状态。
"""

import copy
import math

import rospy
import tf
from auv_control.msg import ActuatorControl, PoseNEDcmd, TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TwistStamped
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


MODE_DPROV = 4


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


class YellowMarkerTest:
    """搜索黄色图形，前往图形中心并执行亮灯动作。"""

    STEP_WAIT_CAMERA = "WAIT_CAMERA"
    STEP_SEARCH_FORWARD = "SEARCH_FORWARD"
    STEP_SEARCH_LEFT = "SEARCH_LEFT"
    STEP_SEARCH_RIGHT = "SEARCH_RIGHT"
    STEP_SEARCH_CENTER = "SEARCH_CENTER"
    STEP_MOVE = "MOVE_TO_YELLOW"
    STEP_LIGHT = "LIGHT_ACTION"
    STEP_FINISH = "FINISH"

    SEARCH_STEPS = {
        STEP_SEARCH_FORWARD,
        STEP_SEARCH_LEFT,
        STEP_SEARCH_RIGHT,
        STEP_SEARCH_CENTER,
    }

    def __init__(self):
        self.node_name = "test_task1_v2_yellow_marker"
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))
        self.tf_listener = tf.TransformListener()

        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd/pose/ned")
        self.target_topic = rospy.get_param("~target_topic", "/obj/target_message")
        self.actuator_topic = rospy.get_param("~actuator_topic", "/cmd/actuator")
        self.finished_topic = rospy.get_param("~finished_topic", "/finished")
        self.camera_topic = rospy.get_param("~camera_topic", "/left/image_raw")
        self.velocity_topic = rospy.get_param("~velocity_topic", "/status/vel")

        self.search_forward_distance = float(rospy.get_param(
            "~search_forward_distance", 0.5
        ))
        self.search_yaw_offset = math.radians(float(rospy.get_param(
            "~search_yaw_offset_deg", 30.0
        )))
        self.position_tolerance = float(rospy.get_param(
            "~position_tolerance", 0.15
        ))
        self.yaw_tolerance = math.radians(float(rospy.get_param(
            "~yaw_tolerance_deg", 3.0
        )))
        self.max_yaw_step = math.radians(float(rospy.get_param(
            "~max_yaw_step_deg", 2.0
        )))

        self.camera_message_timeout = float(rospy.get_param(
            "~camera_message_timeout", 2.0
        ))
        self.velocity_message_timeout = float(rospy.get_param(
            "~velocity_message_timeout", 1.0
        ))
        self.transition_hold_seconds = float(rospy.get_param(
            "~transition_hold_seconds", 4.0
        ))
        self.stable_linear_speed = float(rospy.get_param(
            "~stable_linear_speed", 0.05
        ))
        self.stable_angular_speed = math.radians(float(rospy.get_param(
            "~stable_angular_speed_deg", 3.0
        )))

        self.yellow_classes = class_names(
            "~yellow_classes", ["triangle", "circle"]
        )
        self.marker_sample_count = max(1, int(rospy.get_param(
            "~marker_sample_count", 10
        )))
        self.marker_cluster_distance = float(rospy.get_param(
            "~marker_cluster_distance", 0.25
        ))
        self.marker_sample_timeout = float(rospy.get_param(
            "~marker_sample_timeout", 1.0
        ))
        self.max_camera_distance = float(rospy.get_param(
            "~max_camera_distance", 5.0
        ))

        self.light_seconds = float(rospy.get_param("~light_seconds", 3.0))
        self.gap_seconds = float(rospy.get_param("~gap_seconds", 0.5))
        self.yellow_light_count = max(1, int(rospy.get_param(
            "~yellow_light_count", 1
        )))
        self.light1 = int(rospy.get_param("~light1", 0))
        self.light2 = int(rospy.get_param("~light2", 0))
        self.heading_servo = int(rospy.get_param("~heading_servo", 0x80))
        self.clamp_servo = int(rospy.get_param("~clamp_servo", 0x00))
        self.drive_cmd = int(rospy.get_param("~drive_cmd", 0))
        self.drive_speed = int(rospy.get_param("~drive_speed", 0))

        self.cmd_pub = rospy.Publisher(
            self.cmd_topic, PoseNEDcmd, queue_size=10
        )
        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.finished_pub = rospy.Publisher(
            self.finished_topic, String, queue_size=10
        )
        rospy.Subscriber(
            self.target_topic, TargetDetection, self.target_callback, queue_size=10
        )
        rospy.Subscriber(
            self.camera_topic, rospy.AnyMsg, self.camera_callback, queue_size=1
        )
        rospy.Subscriber(
            self.velocity_topic, TwistStamped, self.velocity_callback, queue_size=5
        )

        self.step = self.STEP_WAIT_CAMERA
        self.start_pose = None
        self.hold_z = None
        self.search_base_yaw = None
        self.last_camera_time = None
        self.latest_velocity = None
        self.latest_velocity_time = None
        self.pose_speed_sample = None

        self.marker_samples = []
        self.last_marker_sample_time = None
        self.detected_marker = None
        self.move_target = None

        self.search_target = None
        self.search_stable_since = None
        self.search_arrival_logged = False
        self.light_started_at = None

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

    def get_current_pose(self):
        try:
            self.tf_listener.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(1.0)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0, "%s: 无法获取当前位姿: %s", self.node_name, error
            )
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
        self.search_base_yaw = yaw_from_quaternion(current.pose.orientation)
        return True

    @staticmethod
    def make_pose(x, y, z, yaw):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw))
        return pose

    def publish_dprov(self, target):
        command = PoseNEDcmd()
        command.mode = MODE_DPROV
        command.target = copy.deepcopy(target)
        command.target.header.frame_id = "map"
        command.target.header.stamp = rospy.Time.now()
        self.cmd_pub.publish(command)

    def publish_position_target(self, point):
        """发布定点目标，但把当前航向原样带入，不主动规划航向。"""
        current = self.get_current_pose()
        if current is None:
            return False
        target = PoseStamped()
        target.header.frame_id = "map"
        target.header.stamp = rospy.Time.now()
        target.pose.position = Point(point.x, point.y, self.hold_z)
        target.pose.orientation = copy.deepcopy(current.pose.orientation)
        self.publish_dprov(target)
        return True

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
                latest_pose = copy.deepcopy(pose)
                latest_pose.header.stamp = rospy.Time(0)
                return self.tf_listener.transformPose("map", latest_pose)
            except tf.Exception as error:
                rospy.logwarn_throttle(
                    2.0, "%s: 黄色图形坐标转换失败: %s", self.node_name, error
                )
                return None

    def target_callback(self, message):
        """只接收黄色图形；多帧稳定后立即切换到定点靠近。"""
        if self.hold_z is None and not self.initialize_start_pose():
            return
        if self.detected_marker is not None or not self.camera_ready():
            return
        if message.type and message.type != "center":
            return
        if message.class_name not in self.yellow_classes:
            return

        camera_point = message.pose.pose.position
        if math.sqrt(
            camera_point.x ** 2 + camera_point.y ** 2 + camera_point.z ** 2
        ) > self.max_camera_distance:
            return

        marker = self.transform_pose_to_map(message.pose)
        if marker is None:
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

        current = self.get_current_pose()
        if current is None:
            self.detected_marker = None
            self.move_target = None
            self.marker_samples = []
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

    def motion_is_stable(self, current):
        now = rospy.Time.now()
        if (
            self.latest_velocity is not None
            and self.latest_velocity_time is not None
            and (now - self.latest_velocity_time).to_sec()
            <= self.velocity_message_timeout
        ):
            linear_speed = math.hypot(
                self.latest_velocity.linear.x,
                self.latest_velocity.linear.y,
            )
            angular_speed = abs(self.latest_velocity.angular.z)
        else:
            yaw = yaw_from_quaternion(current.pose.orientation)
            sample = (
                now,
                current.pose.position.x,
                current.pose.position.y,
                yaw,
            )
            previous = self.pose_speed_sample
            self.pose_speed_sample = sample
            if previous is None:
                return False
            elapsed = (now - previous[0]).to_sec()
            if elapsed <= 0.05:
                return False
            linear_speed = math.hypot(
                sample[1] - previous[1], sample[2] - previous[2]
            ) / elapsed
            angular_speed = abs(wrap_angle(sample[3] - previous[3])) / elapsed

        return (
            linear_speed <= self.stable_linear_speed
            and angular_speed <= self.stable_angular_speed
        )

    def set_search_step(self, step):
        self.step = step
        self.search_target = None
        self.search_stable_since = None
        self.search_arrival_logged = False
        self.pose_speed_sample = None

    def wait_until_search_target_stable(self, current, reached):
        if not reached:
            self.search_stable_since = None
            return False
        if not self.search_arrival_logged:
            rospy.loginfo("%s: SEARCH 目标已到达，等待机器人静止", self.node_name)
            self.search_arrival_logged = True
        if not self.motion_is_stable(current):
            self.search_stable_since = None
            return False
        if self.search_stable_since is None:
            self.search_stable_since = rospy.Time.now()
        return (
            rospy.Time.now() - self.search_stable_since
        ).to_sec() >= self.transition_hold_seconds

    def run_search_forward(self, current):
        if self.search_target is None:
            self.search_target = Point(
                current.pose.position.x
                + self.search_forward_distance * math.cos(self.search_base_yaw),
                current.pose.position.y
                + self.search_forward_distance * math.sin(self.search_base_yaw),
                self.hold_z,
            )
            rospy.loginfo(
                "%s: 识别状态=未识别；SEARCH 定点向前移动 %.2f m",
                self.node_name,
                self.search_forward_distance,
            )

        self.publish_position_target(self.search_target)
        reached = xy_distance(
            current.pose.position, self.search_target
        ) <= self.position_tolerance
        if self.wait_until_search_target_stable(current, reached):
            self.set_search_step(self.STEP_SEARCH_LEFT)

    def run_search_rotation(self, current, offset, next_step, label):
        if self.search_target is None:
            self.search_target = copy.deepcopy(current)
            rospy.loginfo(
                "%s: 识别状态=未识别；SEARCH 在当前位置%s %.1f 度",
                self.node_name,
                label,
                abs(math.degrees(offset)),
            )

        desired_yaw = wrap_angle(self.search_base_yaw + offset)
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        yaw_error = wrap_angle(desired_yaw - current_yaw)
        command_yaw = current_yaw + clamp(
            yaw_error, -self.max_yaw_step, self.max_yaw_step
        )
        target = self.make_pose(
            self.search_target.pose.position.x,
            self.search_target.pose.position.y,
            self.hold_z,
            command_yaw,
        )
        self.publish_dprov(target)

        reached = abs(yaw_error) <= self.yaw_tolerance
        if self.wait_until_search_target_stable(current, reached):
            self.set_search_step(next_step)

    def run_search(self):
        if self.detected_marker is not None:
            self.step = self.STEP_MOVE
            return
        current = self.get_current_pose()
        if current is None:
            return

        rospy.loginfo_throttle(
            3.0, "%s: 识别状态=未识别", self.node_name
        )
        if self.step == self.STEP_SEARCH_FORWARD:
            self.run_search_forward(current)
        elif self.step == self.STEP_SEARCH_LEFT:
            self.run_search_rotation(
                current,
                self.search_yaw_offset,
                self.STEP_SEARCH_RIGHT,
                "左转",
            )
        elif self.step == self.STEP_SEARCH_RIGHT:
            self.run_search_rotation(
                current,
                -self.search_yaw_offset,
                self.STEP_SEARCH_CENTER,
                "右转",
            )
        elif self.step == self.STEP_SEARCH_CENTER:
            self.run_search_rotation(
                current,
                0.0,
                self.STEP_SEARCH_FORWARD,
                "回到初始航向",
            )

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

        # 直接把黄色图形中心作为动力定位目标。目标姿态使用机器人当前姿态，
        # 不根据位置差计算航向，也不把航向误差作为到达条件。
        self.publish_position_target(self.move_target)
        if distance <= self.position_tolerance:
            rospy.loginfo(
                "%s: 已到达黄色图形位置，开始执行亮灯动作", self.node_name
            )
            self.light_started_at = rospy.Time.now()
            self.step = self.STEP_LIGHT

    def publish_lights(self, red):
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
        actuator.red_light = int(red)
        actuator.yellow_light = 0
        actuator.green_light = 0
        self.actuator_pub.publish(actuator)

    def run_light_action(self):
        self.publish_position_target(self.move_target)
        elapsed = (rospy.Time.now() - self.light_started_at).to_sec()
        cycle_seconds = self.light_seconds + self.gap_seconds
        cycle_index = int(elapsed // cycle_seconds)
        if cycle_index >= self.yellow_light_count:
            self.publish_lights(0)
            self.step = self.STEP_FINISH
            return

        cycle_elapsed = elapsed - cycle_index * cycle_seconds
        self.publish_lights(1 if cycle_elapsed < self.light_seconds else 0)

    def finish(self):
        self.publish_lights(0)
        if self.move_target is not None:
            self.publish_position_target(self.move_target)
        self.finished_pub.publish(String(data="yellow marker finished"))
        rospy.loginfo("%s: FINISH 黄色图形动作完成", self.node_name)
        rospy.signal_shutdown("yellow marker test complete")

    def run(self):
        while not rospy.is_shutdown():
            if not self.initialize_start_pose():
                self.rate.sleep()
                continue

            current = self.get_current_pose()
            if current is None:
                self.rate.sleep()
                continue

            if self.step == self.STEP_WAIT_CAMERA:
                self.publish_dprov(self.start_pose)
                camera_state = "已开启" if self.camera_ready() else "未开启"
                rospy.loginfo_throttle(
                    2.0,
                    "%s: 摄像头状态=%s；当前自身位置=(%.2f, %.2f, %.2f)，"
                    "当前航向=%.1f 度",
                    self.node_name,
                    camera_state,
                    current.pose.position.x,
                    current.pose.position.y,
                    current.pose.position.z,
                    math.degrees(yaw_from_quaternion(current.pose.orientation)),
                )
                if self.camera_ready():
                    rospy.loginfo(
                        "%s: 摄像头状态=已开启，进入黄色图形识别阶段",
                        self.node_name,
                    )
                    self.set_search_step(self.STEP_SEARCH_FORWARD)
            elif self.step in self.SEARCH_STEPS:
                self.run_search()
            elif self.step == self.STEP_MOVE:
                self.run_move_to_marker()
            elif self.step == self.STEP_LIGHT:
                self.run_light_action()
            elif self.step == self.STEP_FINISH:
                self.finish()

            self.rate.sleep()


def main():
    rospy.init_node("test_task1_v2_yellow_marker")
    YellowMarkerTest().run()


if __name__ == "__main__":
    main()
