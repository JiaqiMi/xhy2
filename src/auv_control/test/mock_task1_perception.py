#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：mock_task1_perception.py
功能：Task1 V2 真实下位机通信测试用虚拟视觉节点
描述：
    1. 读取真实 TF 中的 map -> base_link，将启动时机器人位置作为正方形起点；
    2. 以机器人初始航向为“前”，生成边长约 1 m 的前、右、后、左闭合轨迹；
    3. 将轨迹前瞻点转换到 camera 坐标系，发布 /obj/line_message；
    4. 在四条边上布置黄色圆形、黑色方形、黄色三角形、黑色方形；
    5. 当机器人接近图形前方可发现范围时，发布 /obj/target_message。
监听：/tf
发布：/obj/line_message (TargetDetection3)，/obj/target_message (TargetDetection)
说明：本节点只虚拟识别输出，不发布 /tf、不发布 /target、不控制下位机。
"""

import math

import rospy
import tf
from auv_control.msg import TargetDetection, TargetDetection3
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = 'mock_task1_perception'


def yaw_from_tf_quaternion(quaternion):
    """从 TF 四元数元组中提取 yaw，单位为弧度。"""
    return euler_from_quaternion(quaternion)[2]


def xy_distance(first, second):
    """计算两个三维点在水平面上的距离。"""
    return math.hypot(first[0] - second[0], first[1] - second[1])


class MockTask1Perception:
    """根据真实机器人位置生成虚拟管线和虚拟图形识别结果。"""

    def __init__(self):
        """初始化参数、发布器、TF 监听器和虚拟场景。"""
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.camera_frame = rospy.get_param('~camera_frame', 'camera')
        self.side_length = float(rospy.get_param('~side_length', 1.0))
        self.rate_hz = float(rospy.get_param('~rate', 5.0))
        self.line_pose1_ahead = float(rospy.get_param('~line_pose1_ahead', 0.20))
        self.line_pose2_ahead = float(rospy.get_param('~line_pose2_ahead', 0.35))
        self.line_pose3_ahead = float(rospy.get_param('~line_pose3_ahead', 0.70))
        self.marker_detection_distance = float(
            rospy.get_param('~marker_detection_distance', 0.60)
        )
        self.marker_reached_distance = float(
            rospy.get_param('~marker_reached_distance', 0.15)
        )
        self.marker_max_publish_seconds = float(
            rospy.get_param('~marker_max_publish_seconds', 120.0)
        )
        self.marker_confidence = float(rospy.get_param('~marker_confidence', 1.0))

        self.tf_listener = tf.TransformListener()
        self.line_pub = rospy.Publisher('/obj/line_message', TargetDetection3, queue_size=10)
        self.target_pub = rospy.Publisher('/obj/target_message', TargetDetection, queue_size=10)
        self.rate = rospy.Rate(self.rate_hz)

        self.start_position, self.start_yaw = self.wait_for_initial_pose()
        self.track_depth = float(rospy.get_param('~track_depth', self.start_position[2]))
        self.vertices = self.build_square_track()
        self.segment_lengths, self.total_length = self.build_segment_lengths()
        self.markers = self.build_markers()

        rospy.loginfo(
            '%s: square track initialized at (%.2f, %.2f, %.2f), yaw %.1f deg',
            NODE_NAME,
            self.start_position[0],
            self.start_position[1],
            self.track_depth,
            math.degrees(self.start_yaw),
        )

    ############################################### 初始化层 #########################################
    def wait_for_initial_pose(self):
        """等待真实 TF 可用，并读取机器人启动时的 map 坐标和航向。"""
        while not rospy.is_shutdown():
            try:
                self.tf_listener.waitForTransform(
                    self.map_frame, self.base_frame, rospy.Time(0), rospy.Duration(1.0)
                )
                translation, rotation = self.tf_listener.lookupTransform(
                    self.map_frame, self.base_frame, rospy.Time(0)
                )
                return translation, yaw_from_tf_quaternion(rotation)
            except tf.Exception as error:
                rospy.logwarn_throttle(2, '%s: waiting for initial TF: %s', NODE_NAME, error)
                rospy.sleep(0.2)
        raise rospy.ROSInterruptException('shutdown while waiting for initial pose')

    def build_square_track(self):
        """按“前、右、后、左”构造 1 m 正方形轨迹。"""
        forward = (math.cos(self.start_yaw), math.sin(self.start_yaw))
        right = (-math.sin(self.start_yaw), math.cos(self.start_yaw))
        start = (self.start_position[0], self.start_position[1], self.track_depth)
        first = (
            start[0] + forward[0] * self.side_length,
            start[1] + forward[1] * self.side_length,
            self.track_depth,
        )
        second = (
            first[0] + right[0] * self.side_length,
            first[1] + right[1] * self.side_length,
            self.track_depth,
        )
        third = (
            second[0] - forward[0] * self.side_length,
            second[1] - forward[1] * self.side_length,
            self.track_depth,
        )
        return [start, first, second, third, start]

    def build_segment_lengths(self):
        """计算每条边长度和总长度。"""
        lengths = []
        total = 0.0
        for index in range(len(self.vertices) - 1):
            length = xy_distance(self.vertices[index], self.vertices[index + 1])
            lengths.append(length)
            total += length
        return lengths, total

    def build_markers(self):
        """在四条边中部附近放置两类动作各两次。"""
        side = self.side_length
        marker_specs = [
            (0.50 * side, 'yellow_circle'),
            (1.50 * side, 'black_square'),
            (2.50 * side, 'yellow_triangle'),
            (3.50 * side, 'black_square'),
        ]
        markers = []
        for progress, class_name in marker_specs:
            markers.append({
                'progress': progress,
                'class_name': class_name,
                'state': 'idle',
                'started_at': None,
            })
        return markers

    ############################################### 轨迹计算层 #######################################
    def point_at_progress(self, progress):
        """根据沿轨迹距离返回 map 坐标系下的轨迹点。"""
        remaining = progress % self.total_length
        for index, length in enumerate(self.segment_lengths):
            if remaining <= length:
                start = self.vertices[index]
                end = self.vertices[index + 1]
                ratio = 0.0 if length <= 1e-6 else remaining / length
                return (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                    self.track_depth,
                )
            remaining -= length
        return self.vertices[-1]

    def closest_progress(self, point):
        """计算机器人当前位置在正方形轨迹上的最近投影进度。"""
        best_distance = None
        best_progress = 0.0
        accumulated = 0.0

        for index, length in enumerate(self.segment_lengths):
            start = self.vertices[index]
            end = self.vertices[index + 1]
            vx = end[0] - start[0]
            vy = end[1] - start[1]
            wx = point[0] - start[0]
            wy = point[1] - start[1]
            denominator = vx * vx + vy * vy
            ratio = 0.0 if denominator <= 1e-9 else (wx * vx + wy * vy) / denominator
            ratio = max(0.0, min(1.0, ratio))
            projection = (start[0] + vx * ratio, start[1] + vy * ratio, self.track_depth)
            distance = xy_distance(point, projection)

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_progress = accumulated + ratio * length
            accumulated += length

        return best_progress

    ############################################### 消息生成层 #######################################
    def pose_in_map(self, point):
        """将 map 坐标点封装为 PoseStamped。"""
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = rospy.Time(0)
        pose.pose.position = Point(point[0], point[1], point[2])
        pose.pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, 0.0))
        return pose

    def transform_pose(self, target_frame, pose):
        """将 PoseStamped 转换到指定坐标系。"""
        try:
            self.tf_listener.waitForTransform(
                target_frame, pose.header.frame_id, rospy.Time(0), rospy.Duration(0.5)
            )
            return self.tf_listener.transformPose(target_frame, pose)
        except tf.Exception as error:
            rospy.logwarn_throttle(2, '%s: transform to %s failed: %s',
                                   NODE_NAME, target_frame, error)
            return None

    def publish_line(self, robot_progress):
        """发布三个相机坐标系下的管线前瞻点。"""
        pose1 = self.transform_pose(
            self.camera_frame,
            self.pose_in_map(self.point_at_progress(robot_progress + self.line_pose1_ahead)),
        )
        pose2 = self.transform_pose(
            self.camera_frame,
            self.pose_in_map(self.point_at_progress(robot_progress + self.line_pose2_ahead)),
        )
        pose3 = self.transform_pose(
            self.camera_frame,
            self.pose_in_map(self.point_at_progress(robot_progress + self.line_pose3_ahead)),
        )
        if pose1 is None or pose2 is None or pose3 is None:
            return

        message = TargetDetection3()
        message.pose1 = pose1
        message.pose2 = pose2
        message.pose3 = pose3
        message.conf = self.marker_confidence
        message.type = 'center'
        message.class_name = 'pipe_line'
        self.line_pub.publish(message)

    def publish_marker(self, marker):
        """发布一个相机坐标系下的虚拟图形识别结果。"""
        marker_pose = self.transform_pose(
            self.camera_frame,
            self.pose_in_map(self.point_at_progress(marker['progress'])),
        )
        if marker_pose is None:
            return

        message = TargetDetection()
        message.pose = marker_pose
        message.conf = self.marker_confidence
        message.type = 'center'
        message.class_name = marker['class_name']
        self.target_pub.publish(message)

    ############################################### 图形状态层 #######################################
    def update_markers(self, robot_position, robot_progress):
        """根据机器人沿轨迹进度，决定是否发布和关闭虚拟图形。"""
        now = rospy.Time.now()
        for marker in self.markers:
            if marker['state'] == 'used':
                continue

            marker_point = self.point_at_progress(marker['progress'])
            ahead = (marker['progress'] - robot_progress) % self.total_length

            if marker['state'] == 'idle' and ahead <= self.marker_detection_distance:
                marker['state'] = 'publishing'
                marker['started_at'] = now
                rospy.loginfo(
                    '%s: start publishing %s at progress %.2f',
                    NODE_NAME,
                    marker['class_name'],
                    marker['progress'],
                )

            if marker['state'] != 'publishing':
                continue

            self.publish_marker(marker)
            if xy_distance(robot_position, marker_point) <= self.marker_reached_distance:
                marker['state'] = 'used'
                rospy.loginfo('%s: marker %s reached; stop publishing',
                              NODE_NAME, marker['class_name'])
                continue

            if (now - marker['started_at']).to_sec() >= self.marker_max_publish_seconds:
                marker['state'] = 'used'
                rospy.logwarn('%s: marker %s publish timeout; mark as used',
                              NODE_NAME, marker['class_name'])

    ############################################### 主循环 ###########################################
    def run(self):
        """持续根据真实机器人位姿发布虚拟管线和虚拟图形。"""
        while not rospy.is_shutdown():
            try:
                translation, _ = self.tf_listener.lookupTransform(
                    self.map_frame, self.base_frame, rospy.Time(0)
                )
            except tf.Exception as error:
                rospy.logwarn_throttle(2, '%s: cannot read robot pose: %s',
                                       NODE_NAME, error)
                self.rate.sleep()
                continue

            robot_position = (translation[0], translation[1], translation[2])
            robot_progress = self.closest_progress(robot_position)
            self.publish_line(robot_progress)
            self.update_markers(robot_position, robot_progress)
            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=False)
    try:
        MockTask1Perception().run()
    except rospy.ROSInterruptException:
        pass
