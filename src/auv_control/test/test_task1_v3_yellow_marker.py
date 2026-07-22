#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v3_yellow_marker.py
功能：Task1 黄色图形单项测试，运动接口迁移到 motion_supervisor。

流程：
    1. 定点保持启动位姿一段可调时间，等待相机和识别模块启动；
    2. 未识别到黄色图形时，依次定点左转、右转、回正，再向前移动后重复；
    3. 最近 N 条有效识别中有 K 条位置聚类后，发布 map 绝对目标前往图形中心；
    4. 前往图形时只判断水平位置误差，不规划或校正目标航向；
    5. 到达黄色图形上方后亮红灯，动作完成后发布完成消息。

监听：/obj/target_message，/left/image_raw，/motion/state，/tf
发布：/cmd/motion/goal，/cmd/motion/cancel，/cmd/actuator，/finished

记录：
2026.7.14
    初版，用于单独验证 Task1 图形识别、前往图形和黄色动作流程。
2026.7.16
    将脚本改为纯黄色图形测试，删除黑色图形识别、旋转和无关状态。
    搜索基准航向直接读取节点启动时的机器人当前航向，不再使用人工配置参数。
    锁定黄色图形后直接发布其中心为动力定位目标，只以 XY 距离判断到达。
    未识别时采用“前进、左移、回中心、右移、回中心”的搜索流程，横移仅输出 TY。
    精简运行日志，只保留相机状态、识别状态、当前位置、目标位置、距离和完成状态。
2026.7.18
    启动阶段改为定点悬停可调时间，避免识别模块尚未就绪时提前运动。
    图形确认改为最近 N 条识别输出中至少 K 条有效，窗口、有效数和置信度均可配置。
    搜索改为 mode=4 左右转、回到启动航向并稳定后定点前进，循环执行。
    新增 v3：任务不再发布 PoseNEDcmd，不直接控制 mode 或六轴力；所有运动
    改为向 motion_supervisor 发布 map 绝对目标，并只用新鲜 HOVER 判定到达。
2026.7.20
    增加带时间戳的 YAML 数据文件，记录图形消息处理结果以及每个控制周期的
    位姿、目标、MotionState、识别窗口和动作阶段。
2026.7.22
    识别窗口只保留有效识别帧，每条样本独立按最大保存时间淘汰。
    增加参考深度参数；-1 使用启动时深度，其他值作为全程运动深度。
"""

import copy
import math

import rospy
import tf
from auv_control.msg import ActuatorControl, MotionState, TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Empty, String
from task1_v3_yaml_logger import TimestampedYamlLogger
from tf.transformations import euler_from_quaternion, quaternion_from_euler


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
    STEP_SEARCH_LEFT = "SEARCH_LEFT"
    STEP_SEARCH_RIGHT = "SEARCH_RIGHT"
    STEP_SEARCH_RETURN = "SEARCH_RETURN"
    STEP_SEARCH_FORWARD = "SEARCH_FORWARD"
    STEP_MOVE = "MOVE_TO_YELLOW"
    STEP_LIGHT = "LIGHT_ACTION"
    STEP_FINISH = "FINISH"

    SEARCH_STEPS = {
        STEP_SEARCH_LEFT,
        STEP_SEARCH_RIGHT,
        STEP_SEARCH_RETURN,
        STEP_SEARCH_FORWARD,
    }

    def __init__(self):
        self.node_name = getattr(
            self, "node_name", "test_task1_v3_yellow_marker"
        )
        self.marker_display_name = getattr(
            self, "marker_display_name", "黄色图形"
        )
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))
        self.reference_depth = float(rospy.get_param("~reference_depth", -1.0))
        self.tf_listener = tf.TransformListener()

        self.motion_goal_topic = rospy.get_param(
            "~motion_goal_topic", "/cmd/motion/goal"
        )
        self.motion_cancel_topic = rospy.get_param(
            "~motion_cancel_topic", "/cmd/motion/cancel"
        )
        self.motion_state_topic = rospy.get_param(
            "~motion_state_topic", "/motion/state"
        )
        self.motion_state_timeout = max(0.1, float(rospy.get_param(
            "~motion_state_timeout", 0.5
        )))
        self.motion_goal_position_tolerance = max(0.001, float(rospy.get_param(
            "~motion_goal_position_tolerance", 0.05
        )))
        self.motion_goal_yaw_tolerance = math.radians(max(
            0.1,
            float(rospy.get_param("~motion_goal_yaw_tolerance_deg", 3.0)),
        ))
        self.target_topic = rospy.get_param("~target_topic", "/obj/target_message")
        self.actuator_topic = rospy.get_param("~actuator_topic", "/cmd/actuator")
        self.finished_topic = rospy.get_param("~finished_topic", "/finished")
        self.camera_topic = rospy.get_param("~camera_topic", "/left/image_raw")
        self.log_directory = rospy.get_param(
            "~log_directory", "~/.ros/auv_logs/task1"
        )

        self.search_forward_distance = float(rospy.get_param(
            "~search_forward_distance", 0.5
        ))
        self.search_yaw_angle = math.radians(float(rospy.get_param(
            "~search_yaw_angle_deg", 30.0
        )))
        self.search_yaw_sign = 1.0 if float(rospy.get_param(
            "~search_yaw_sign", 1.0
        )) >= 0.0 else -1.0
        self.search_yaw_tolerance = math.radians(float(rospy.get_param(
            "~search_yaw_tolerance_deg", 5.0
        )))
        self.search_yaw_hold_seconds = max(0.0, float(rospy.get_param(
            "~search_yaw_hold_seconds", 2.0
        )))
        self.search_return_hold_seconds = max(0.0, float(rospy.get_param(
            "~search_return_hold_seconds", 2.0
        )))
        self.search_forward_hold_seconds = max(0.0, float(rospy.get_param(
            "~search_forward_hold_seconds", 4.0
        )))
        self.position_tolerance = float(rospy.get_param(
            "~position_tolerance", 0.15
        ))

        self.camera_message_timeout = float(rospy.get_param(
            "~camera_message_timeout", 2.0
        ))
        self.startup_hold_seconds = max(0.0, float(rospy.get_param(
            "~startup_hold_seconds", 10.0
        )))
        self.yellow_classes = class_names(
            "~yellow_classes", ["triangle", "circle"]
        )
        self.yellow_min_confidence = float(rospy.get_param(
            "~yellow_min_confidence", 0.30
        ))
        self.marker_window_size = max(1, int(rospy.get_param(
            "~marker_window_size", 10
        )))
        self.marker_required_valid = max(1, int(rospy.get_param(
            "~marker_required_valid", 3
        )))
        self.marker_required_valid = min(
            self.marker_required_valid, self.marker_window_size
        )
        self.marker_cluster_distance = float(rospy.get_param(
            "~marker_cluster_distance", 0.25
        ))
        self.marker_sample_timeout = max(0.1, float(rospy.get_param(
            "~marker_sample_timeout", 10.0
        )))
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

        self.motion_goal_pub = rospy.Publisher(
            self.motion_goal_topic, PoseStamped, queue_size=1
        )
        self.motion_cancel_pub = rospy.Publisher(
            self.motion_cancel_topic, Empty, queue_size=1
        )
        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.finished_pub = rospy.Publisher(
            self.finished_topic, String, queue_size=10
        )
        self.step = self.STEP_WAIT_CAMERA
        self.start_pose = None
        self.hold_z = None
        self.search_base_yaw = None
        self.startup_hold_started = None
        self.last_camera_time = None
        self.latest_motion_state = None
        self.last_motion_goal = None
        self.cancel_sent = False

        self.marker_observation_window = []
        self.detected_marker = None
        self.move_target = None
        self.move_goal = None

        self.search_target = None
        self.search_cycle_anchor = None
        self.search_stable_since = None
        self.search_arrival_logged = False
        self.light_started_at = None
        self.data_logger = None
        self.open_data_log()
        rospy.on_shutdown(self.shutdown)

        # 状态字段全部就绪后再创建订阅，避免首帧消息在构造期间触发回调。
        rospy.Subscriber(
            self.target_topic, TargetDetection, self.target_callback, queue_size=10
        )
        rospy.Subscriber(
            self.camera_topic, rospy.AnyMsg, self.camera_callback, queue_size=1
        )
        rospy.Subscriber(
            self.motion_state_topic,
            MotionState,
            self.motion_state_callback,
            queue_size=1,
        )

    @staticmethod
    def point_record(point):
        if point is None:
            return None
        return [point.x, point.y, point.z]

    @staticmethod
    def pose_record(pose):
        if pose is None:
            return None
        return {
            "position": YellowMarkerTest.point_record(pose.pose.position),
            "yaw_deg": math.degrees(yaw_from_quaternion(
                pose.pose.orientation
            )),
        }

    def open_data_log(self):
        try:
            self.data_logger = TimestampedYamlLogger(
                self.node_name, self.log_directory
            )
            self.write_data_record(
                "startup",
                log_directory=self.log_directory,
                marker_display_name=self.marker_display_name,
            )
            rospy.loginfo(
                "%s: 完整数据文件=%s",
                self.node_name,
                self.data_logger.path,
            )
        except OSError as error:
            self.data_logger = None
            rospy.logwarn(
                "%s: 无法创建完整数据文件: %s", self.node_name, error
            )

    def shutdown(self):
        self.cancel_motion()
        if self.data_logger is not None:
            self.data_logger.close()
            self.data_logger = None

    def write_data_record(self, event, **data):
        if self.data_logger is None:
            return
        data.setdefault("step", self.step)
        try:
            self.data_logger.write(event, **data)
        except (OSError, TypeError, ValueError) as error:
            rospy.logwarn_throttle(
                5.0, "%s: 完整数据写入失败: %s", self.node_name, error
            )

    def record_target_message(self, message, status, transformed=None):
        self.write_data_record(
            "target_frame",
            status=status,
            class_name=message.class_name,
            target_type=message.type,
            confidence=float(message.conf),
            source_frame=message.pose.header.frame_id,
            camera_position=self.point_record(message.pose.pose.position),
            map_position=(
                self.point_record(transformed.pose.position)
                if transformed is not None else None
            ),
            observation_window_size=len(self.marker_observation_window),
            observation_valid_count=len(self.marker_observation_window),
        )

    def record_control_cycle(self, current):
        self.prune_marker_observations()
        motion = self.latest_motion_state
        rotation = getattr(self, "rotation_state", None)
        rotation_data = None
        if rotation is not None:
            rotation_data = {
                "completed_deg": math.degrees(rotation["completed"]),
                "lookahead_deg": math.degrees(getattr(
                    self, "black_rotation_step", 0.0
                )),
                "goal": self.pose_record(rotation["goal"]),
                "final_goal_active": rotation["final_goal_active"],
                "hover_started": (
                    rotation["hover_started"].to_sec()
                    if rotation["hover_started"] is not None else None
                ),
            }
        self.write_data_record(
            "control_cycle",
            camera_ready=self.camera_ready(),
            current_pose=self.pose_record(current),
            command_goal=self.pose_record(self.last_motion_goal),
            move_target=self.point_record(self.move_target),
            detected_marker=self.pose_record(self.detected_marker),
            observation_window_size=len(self.marker_observation_window),
            observation_valid_count=len(self.marker_observation_window),
            action_phase=getattr(self, "black_action_phase", None),
            rotation=rotation_data,
            motion=(
                {
                    "state": motion.state,
                    "reason": motion.reason,
                    "goal": self.pose_record(motion.goal),
                    "tx": motion.tx,
                    "ty": motion.ty,
                    "mz": motion.mz,
                }
                if motion is not None else None
            ),
        )

    def camera_callback(self, _message):
        self.last_camera_time = rospy.Time.now()

    def motion_state_callback(self, message):
        self.latest_motion_state = copy.deepcopy(message)

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
        self.hold_z = (
            current.pose.position.z
            if self.reference_depth == -1.0
            else self.reference_depth
        )
        self.start_pose.pose.position.z = self.hold_z
        self.search_base_yaw = yaw_from_quaternion(current.pose.orientation)
        self.startup_hold_started = rospy.Time.now()
        rospy.loginfo(
            "%s: 参考深度=%.2f m（%s）",
            self.node_name,
            self.hold_z,
            "当前深度" if self.reference_depth == -1.0 else "launch 设置",
        )
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

    def publish_motion_goal(self, target):
        goal = copy.deepcopy(target)
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        self.last_motion_goal = copy.deepcopy(goal)
        self.motion_goal_pub.publish(goal)

    def publish_dprov(self, target):
        """兼容原状态机函数名；v3 实际发布 motion_supervisor 目标。"""
        self.publish_motion_goal(target)

    def motion_state_fresh(self):
        return (
            self.latest_motion_state is not None
            and (rospy.Time.now() - self.latest_motion_state.header.stamp).to_sec()
            <= self.motion_state_timeout
        )

    def motion_goal_matches(self, target):
        if not self.motion_state_fresh() or target is None:
            return False
        actual = self.latest_motion_state.goal
        position_error = math.sqrt(
            (actual.pose.position.x - target.pose.position.x) ** 2
            + (actual.pose.position.y - target.pose.position.y) ** 2
            + (actual.pose.position.z - target.pose.position.z) ** 2
        )
        yaw_error = abs(wrap_angle(
            yaw_from_quaternion(actual.pose.orientation)
            - yaw_from_quaternion(target.pose.orientation)
        ))
        return (
            position_error <= self.motion_goal_position_tolerance
            and yaw_error <= self.motion_goal_yaw_tolerance
        )

    def motion_arrived(self, target=None):
        target = self.last_motion_goal if target is None else target
        return (
            self.motion_state_fresh()
            and self.latest_motion_state.state == MotionState.HOVER
            and self.motion_goal_matches(target)
        )

    def motion_failed(self):
        return (
            self.motion_state_fresh()
            and self.latest_motion_state.state == MotionState.SAFE
        )

    def cancel_motion(self):
        if not self.cancel_sent:
            self.motion_cancel_pub.publish(Empty())
            self.cancel_sent = True

    def publish_position_target(self, point):
        """发布定点目标，但把当前航向原样带入，不主动规划航向。"""
        if self.move_goal is None:
            current = self.get_current_pose()
            if current is None:
                return False
            self.move_goal = self.make_pose(
                point.x,
                point.y,
                self.hold_z,
                yaw_from_quaternion(current.pose.orientation),
            )
        self.publish_motion_goal(self.move_goal)
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

    def reset_marker_window(self):
        """清空只包含有效识别帧的滑动窗口。"""
        self.marker_observation_window = []

    def prune_marker_observations(self, now=None):
        """独立淘汰超过最大保存时间的有效识别帧。"""
        now = rospy.Time.now() if now is None else now
        self.marker_observation_window = [
            item for item in self.marker_observation_window
            if (now - item[0]).to_sec() <= self.marker_sample_timeout
        ]

    def add_marker_observation(self, marker):
        """只加入有效识别，并返回最近有效帧中满足聚类条件的图形位姿。"""
        now = rospy.Time.now()
        self.prune_marker_observations(now)
        if marker is None:
            return None

        self.marker_observation_window.append((now, copy.deepcopy(marker)))
        self.marker_observation_window = self.marker_observation_window[
            -self.marker_window_size:
        ]
        valid = [item[1] for item in self.marker_observation_window]
        rospy.loginfo_throttle(
            2.0,
            "%s: %s有效识别窗口=%d/%d，同位置需要=%d，最大保存=%.1f s",
            self.node_name,
            self.marker_display_name,
            len(self.marker_observation_window),
            self.marker_window_size,
            self.marker_required_valid,
            self.marker_sample_timeout,
        )
        if len(valid) < self.marker_window_size:
            return None

        best_cluster = []
        for seed in valid:
            cluster = [
                item for item in valid
                if xy_distance(item.pose.position, seed.pose.position)
                <= self.marker_cluster_distance
            ]
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
        if len(best_cluster) < self.marker_required_valid:
            return None

        confirmed = copy.deepcopy(best_cluster[-1])
        confirmed.pose.position.x = sum(
            item.pose.position.x for item in best_cluster
        ) / len(best_cluster)
        confirmed.pose.position.y = sum(
            item.pose.position.y for item in best_cluster
        ) / len(best_cluster)
        confirmed.pose.position.z = self.hold_z
        return confirmed

    def lock_confirmed_marker(self, marker):
        """锁定确认后的图形位置并切换到定点靠近。"""
        self.detected_marker = copy.deepcopy(marker)
        self.move_target = copy.deepcopy(marker.pose.position)
        self.move_goal = None
        self.search_target = None
        self.search_cycle_anchor = None
        self.search_stable_since = None

        current = self.get_current_pose()
        if current is None:
            self.detected_marker = None
            self.move_target = None
            self.reset_marker_window()
            return

        distance = xy_distance(current.pose.position, self.move_target)
        self.move_goal = self.make_pose(
            self.move_target.x,
            self.move_target.y,
            self.hold_z,
            yaw_from_quaternion(current.pose.orientation),
        )
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

    def target_callback(self, message):
        """最近 N 条有效识别中有 K 条位置聚类时确认黄色图形。"""
        if self.hold_z is None and not self.initialize_start_pose():
            self.record_target_message(message, "robot_pose_unavailable")
            return
        if (
            self.step == self.STEP_WAIT_CAMERA
            or self.detected_marker is not None
            or not self.camera_ready()
        ):
            self.record_target_message(message, "ignored_in_current_step")
            return

        marker = None
        valid_message = (
            (not message.type or message.type == "center")
            and message.class_name in self.yellow_classes
            and float(message.conf) >= self.yellow_min_confidence
        )
        if valid_message:
            camera_point = message.pose.pose.position
            valid_message = (
                math.isfinite(camera_point.x)
                and math.isfinite(camera_point.y)
                and math.isfinite(camera_point.z)
                and math.sqrt(
                    camera_point.x ** 2
                    + camera_point.y ** 2
                    + camera_point.z ** 2
                ) <= self.max_camera_distance
            )
        if valid_message:
            marker = self.transform_pose_to_map(message.pose)

        confirmed = self.add_marker_observation(marker)
        self.record_target_message(
            message,
            "confirmed" if confirmed is not None else (
                "accepted" if marker is not None else "rejected"
            ),
            marker,
        )
        if confirmed is not None:
            self.lock_confirmed_marker(confirmed)

    def motion_is_stable(self, _current):
        """HOVER 已包含位置、航向、速度和下位机 mode=4 接管确认。"""
        return self.motion_arrived()

    def set_search_step(self, step):
        self.step = step
        self.search_target = None
        self.search_stable_since = None
        self.search_arrival_logged = False

    def wait_until_search_target_stable(
        self, current, position_reached, yaw_reached, hold_seconds
    ):
        if not position_reached or not yaw_reached:
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
        ).to_sec() >= hold_seconds

    def run_search_rotation(self, current, yaw_offset, next_step, label, hold_seconds):
        """保持搜索锚点，只向运动监督器提交绝对航向目标。"""
        if self.search_cycle_anchor is None:
            self.search_cycle_anchor = copy.deepcopy(current.pose.position)
        target_yaw = wrap_angle(self.search_base_yaw + yaw_offset)
        if self.search_target is None:
            self.search_target = self.make_pose(
                self.search_cycle_anchor.x,
                self.search_cycle_anchor.y,
                self.hold_z,
                target_yaw,
            )
            rospy.loginfo(
                "%s: 识别状态=未识别；SEARCH %s %.1f 度",
                self.node_name,
                label,
                math.degrees(abs(yaw_offset)),
            )

        self.publish_dprov(self.search_target)
        position_reached = xy_distance(
            current.pose.position, self.search_cycle_anchor
        ) <= self.position_tolerance
        yaw_error = abs(wrap_angle(
            target_yaw - yaw_from_quaternion(current.pose.orientation)
        ))
        if self.wait_until_search_target_stable(
            current,
            position_reached,
            yaw_error <= self.search_yaw_tolerance,
            hold_seconds,
        ):
            self.set_search_step(next_step)

    def run_search_forward(self, current):
        if self.search_target is None:
            target_point = Point(
                current.pose.position.x
                + self.search_forward_distance * math.cos(self.search_base_yaw),
                current.pose.position.y
                + self.search_forward_distance * math.sin(self.search_base_yaw),
                self.hold_z,
            )
            self.search_target = self.make_pose(
                target_point.x,
                target_point.y,
                self.hold_z,
                self.search_base_yaw,
            )
            rospy.loginfo(
                "%s: 识别状态=未识别；SEARCH 定点向前移动 %.2f m",
                self.node_name,
                self.search_forward_distance,
            )

        self.publish_dprov(self.search_target)
        reached = xy_distance(
            current.pose.position, self.search_target.pose.position
        ) <= self.position_tolerance
        yaw_error = abs(wrap_angle(
            self.search_base_yaw - yaw_from_quaternion(current.pose.orientation)
        ))
        if self.wait_until_search_target_stable(
            current,
            reached,
            yaw_error <= self.search_yaw_tolerance,
            self.search_forward_hold_seconds,
        ):
            self.search_cycle_anchor = None
            self.set_search_step(self.STEP_SEARCH_LEFT)

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
        if self.step == self.STEP_SEARCH_LEFT:
            self.run_search_rotation(
                current,
                -self.search_yaw_sign * self.search_yaw_angle,
                self.STEP_SEARCH_RIGHT,
                "向左旋转",
                self.search_yaw_hold_seconds,
            )
        elif self.step == self.STEP_SEARCH_RIGHT:
            self.run_search_rotation(
                current,
                self.search_yaw_sign * self.search_yaw_angle,
                self.STEP_SEARCH_RETURN,
                "向右旋转",
                self.search_yaw_hold_seconds,
            )
        elif self.step == self.STEP_SEARCH_RETURN:
            self.run_search_rotation(
                current,
                0.0,
                self.STEP_SEARCH_FORWARD,
                "返回初始航向",
                self.search_return_hold_seconds,
            )
        elif self.step == self.STEP_SEARCH_FORWARD:
            self.run_search_forward(current)

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

        # 图形中心和锁定时航向组成固定的 map 绝对目标；只用匹配该目标的
        # 新鲜 HOVER 进入动作阶段，不在任务节点自行判断刹车或模式。
        self.publish_position_target(self.move_target)
        if self.motion_arrived(self.move_goal):
            rospy.loginfo(
                "%s: 已到达%s位置，开始执行亮灯动作",
                self.node_name,
                self.marker_display_name,
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
        self.cancel_motion()
        self.finished_pub.publish(String(data="yellow marker finished"))
        rospy.loginfo("%s: FINISH 黄色图形动作完成", self.node_name)
        rospy.signal_shutdown("yellow marker test complete")

    def run(self):
        while not rospy.is_shutdown():
            if not self.initialize_start_pose():
                self.record_control_cycle(None)
                self.rate.sleep()
                continue

            current = self.get_current_pose()
            if current is None:
                self.record_control_cycle(None)
                self.rate.sleep()
                continue
            if self.motion_failed():
                rospy.logerr_throttle(
                    2.0,
                    "%s: motion_supervisor=SAFE，暂停任务推进，原因=%s",
                    self.node_name,
                    self.latest_motion_state.reason,
                )
                self.record_control_cycle(current)
                self.rate.sleep()
                continue

            if self.step == self.STEP_WAIT_CAMERA:
                self.publish_dprov(self.start_pose)
                camera_state = "已开启" if self.camera_ready() else "未开启"
                hold_elapsed = (
                    (rospy.Time.now() - self.startup_hold_started).to_sec()
                    if self.startup_hold_started is not None else 0.0
                )
                rospy.loginfo_throttle(
                    2.0,
                    "%s: 摄像头状态=%s；当前自身位置=(%.2f, %.2f, %.2f)，"
                    "当前航向=%.1f 度；启动定点=%.1f/%.1f s",
                    self.node_name,
                    camera_state,
                    current.pose.position.x,
                    current.pose.position.y,
                    current.pose.position.z,
                    math.degrees(yaw_from_quaternion(current.pose.orientation)),
                    hold_elapsed,
                    self.startup_hold_seconds,
                )
                if (
                    self.camera_ready()
                    and hold_elapsed >= self.startup_hold_seconds
                    and self.motion_arrived(self.start_pose)
                ):
                    rospy.loginfo(
                        "%s: 启动定点完成，进入%s识别和左右转搜索阶段",
                        self.node_name,
                        self.marker_display_name,
                    )
                    self.set_search_step(self.STEP_SEARCH_LEFT)
            elif self.step in self.SEARCH_STEPS:
                self.run_search()
            elif self.step == self.STEP_MOVE:
                self.run_move_to_marker()
            elif self.step == self.STEP_LIGHT:
                self.run_light_action()
            elif self.step == self.STEP_FINISH:
                self.finish()

            self.record_control_cycle(current)
            self.rate.sleep()


def main():
    rospy.init_node("test_task1_v3_yellow_marker")
    YellowMarkerTest().run()


if __name__ == "__main__":
    main()
