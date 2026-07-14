#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
任务 3 子任务 1 测试：寻找箭头并进入任务获取区方框。

topic 模式下只搜索机器人前方区域。稳定锁定箭头识别结果后，机器人会在保持
可配置靠近距离的同时靠近箭头中心，并让自身航向对齐识别到的箭头航向。
旧版 /target 输出仍可通过 motion_output=target 或 motion_output=both 启用。
"""

import math

import rospy
import tf
from auv_control.msg import PoseNEDcmd, TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task3_1_acquire_area"

# =========================
# 可调默认参数
# =========================
# 这些值是水池/实艇调试时优先调整的位置。
# 它们仍然是 ROS 参数，因此 roslaunch 可以在不改代码的情况下覆盖。

# 基础输入。
DEFAULT_RATE = 5.0
DEFAULT_INPUT_MODE = "mock"  # 可选 mock 或 topic
DEFAULT_ARROW_TOPIC = "/obj/target_message"
DEFAULT_ARROW_CLASS = "arrow"
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_DETECTION_FRAME = "camera"

# 运动指令输出。
# cmd：发布最新的 /cmd/pose/ned PoseNEDcmd 接口。
# target：发布旧版 /target PoseStamped 接口。
# both：迁移/调试阶段同时发布两个接口。
DEFAULT_MOTION_OUTPUT = "cmd"  # 可选 cmd、target 或 both
DEFAULT_LEGACY_TARGET_TOPIC = "/target"
DEFAULT_POSE_CMD_TOPIC = "/cmd/pose/ned"
DEFAULT_POSE_CMD_MODE = 4  # 2=定深，3=定深+定航向，4=DPROV 定点保持
DEFAULT_POSE_CMD_FORCE = [0, 0, 0, 0, 0, 0]  # 六自由度力/力矩：TX、TY、TZ、MX、MY、MZ

# 仅 mock 模式使用：相对 base_link 的旧临时目标。
# 本项目 base_link 约定：x=前方，y=右方，z=向下。
DEFAULT_ARROW_FORWARD = 0.50
DEFAULT_ARROW_RIGHT = 0.30
DEFAULT_ARROW_DOWN = 0.00

# 搜索运动。已知箭头位于机器人前方，因此 topic 模式只搜索前方行，
# 不会下发任何后退搜索点。
DEFAULT_SEARCH_STEP = 0.30
DEFAULT_MAX_SEARCH_POINTS = 14
DEFAULT_SCAN_YAW_OFFSETS_DEG = [0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0]
DEFAULT_SCAN_HOLD_SECONDS = 1.5
DEFAULT_MAX_SEARCH_SECONDS = 300.0
DEFAULT_SEARCH_ARRIVE_DIST = 0.15
DEFAULT_SEARCH_ARRIVE_YAW_DEG = 8.0

# 识别锁定。如果误检较多，可以提高 stable_detection_count 或降低容差；
# 如果检测器较慢或噪声较大，可以适当放宽。
DEFAULT_STABLE_DETECTION_COUNT = 5
DEFAULT_STABLE_POSITION_TOLERANCE = 0.15
DEFAULT_DETECTION_TIMEOUT = 2.0

# 箭头航向。验收条件要求机器人航向与箭头一致，因此默认使用 detection。
# 如果感知暂未填充 TargetDetection.pose.orientation，调试时可使用 current 或 fixed。
DEFAULT_ARROW_YAW_MODE = "detection"  # 可选 current、detection 或 fixed
DEFAULT_FIXED_ARROW_YAW_DEG = 0.0

# 最终视觉靠近箭头中心。成功条件是箭头中心约位于 base_link 前方 0.30 m，
# 且机器人航向与箭头航向一致。
DEFAULT_ARRIVE_DIST = 0.25
DEFAULT_ARRIVE_YAW_DEG = 8.0
DEFAULT_MIN_GROUND_CLEARANCE = 0.40
DEFAULT_APPROACH_DISTANCE = 0.30
DEFAULT_APPROACH_DISTANCE_TOLERANCE = 0.10
DEFAULT_APPROACH_LATERAL_TOLERANCE = 0.10
DEFAULT_APPROACH_YAW_TOLERANCE_DEG = 10.0
DEFAULT_HOLD_SECONDS = 2.0


class Task3AcquireAreaTest:
    SEARCH_MOVE = 0
    SEARCH_SCAN = 1
    MOVE_TO_ARROW = 2
    HOLD = 3

    def __init__(self):
        self.motion_output = (
            rospy.get_param("~motion_output", DEFAULT_MOTION_OUTPUT).strip().lower()
        )
        self.legacy_target_topic = rospy.get_param(
            "~legacy_target_topic", DEFAULT_LEGACY_TARGET_TOPIC
        )
        self.pose_cmd_topic = rospy.get_param("~pose_cmd_topic", DEFAULT_POSE_CMD_TOPIC)
        self.pose_cmd_mode = int(rospy.get_param("~pose_cmd_mode", DEFAULT_POSE_CMD_MODE))
        self.pose_cmd_force = self.parse_int_list(
            rospy.get_param("~pose_cmd_force", DEFAULT_POSE_CMD_FORCE),
            DEFAULT_POSE_CMD_FORCE,
        )

        self.target_pub = rospy.Publisher(
            self.legacy_target_topic, PoseStamped, queue_size=10
        )
        self.pose_cmd_pub = rospy.Publisher(
            self.pose_cmd_topic, PoseNEDcmd, queue_size=10
        )
        self.finished_pub = rospy.Publisher("/finished", String, queue_size=10)
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(rospy.get_param("~rate", DEFAULT_RATE))

        self.input_mode = rospy.get_param("~input_mode", DEFAULT_INPUT_MODE).strip().lower()
        self.arrow_topic = rospy.get_param("~arrow_topic", DEFAULT_ARROW_TOPIC)
        self.arrow_class = rospy.get_param("~arrow_class", DEFAULT_ARROW_CLASS).strip().lower()
        self.detection_frame = rospy.get_param(
            "~detection_frame", DEFAULT_DETECTION_FRAME
        )
        self.min_confidence = float(
            rospy.get_param("~min_confidence", DEFAULT_MIN_CONFIDENCE)
        )

        # mock 模式保留旧的固定参数目标；topic 模式改为靠近识别到的箭头/方框位置。
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
        self.min_ground_clearance = float(
            rospy.get_param("~min_ground_clearance", DEFAULT_MIN_GROUND_CLEARANCE)
        )
        self.approach_distance = float(
            rospy.get_param("~approach_distance", DEFAULT_APPROACH_DISTANCE)
        )
        self.approach_distance_tolerance = float(
            rospy.get_param(
                "~approach_distance_tolerance",
                DEFAULT_APPROACH_DISTANCE_TOLERANCE,
            )
        )
        self.approach_lateral_tolerance = float(
            rospy.get_param(
                "~approach_lateral_tolerance",
                DEFAULT_APPROACH_LATERAL_TOLERANCE,
            )
        )
        self.approach_yaw_tolerance = math.radians(
            float(
                rospy.get_param(
                    "~approach_yaw_tolerance_deg",
                    DEFAULT_APPROACH_YAW_TOLERANCE_DEG,
                )
            )
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
            "%s：启动子任务1，输入模式=%s，箭头话题=%s，目标类别=%s，最低置信度=%.2f",
            NODE_NAME,
            self.input_mode,
            self.arrow_topic,
            self.arrow_class,
            self.min_confidence,
        )
        rospy.loginfo(
            (
                "%s：搜索参数 search_step=%.2fm，搜索点数量=%d，扫描航向=%s，"
                "每个航向停留=%.1fs，最大搜索时间=%.1fs"
            ),
            NODE_NAME,
            self.search_step,
            self.max_search_points,
            self.scan_yaw_offsets_deg,
            self.scan_hold_seconds,
            self.max_search_seconds,
        )
        rospy.loginfo(
            (
                "%s：靠近判定 approach_distance=%.2fm，前后容差=%.2fm，"
                "左右容差=%.2fm，航向容差=%.1fdeg，对地最小距离=%.2fm"
            ),
            NODE_NAME,
            self.approach_distance,
            self.approach_distance_tolerance,
            self.approach_lateral_tolerance,
            math.degrees(self.approach_yaw_tolerance),
            self.min_ground_clearance,
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
            raise ValueError("input_mode 必须是 mock 或 topic")

        if self.arrow_yaw_mode not in ("current", "detection", "fixed"):
            raise ValueError("arrow_yaw_mode 必须是 current、detection 或 fixed")

        if self.stable_detection_count <= 0:
            raise ValueError("stable_detection_count 必须为正数")

        if not self.scan_yaw_offsets_deg:
            raise ValueError("scan_yaw_offsets_deg 不能为空")

        if self.max_search_seconds <= 0.0:
            raise ValueError("max_search_seconds 必须为正数")

        if self.motion_output not in ("cmd", "target", "both"):
            raise ValueError("motion_output 必须是 cmd、target 或 both")

        if self.pose_cmd_mode not in (2, 3, 4):
            raise ValueError("pose_cmd_mode 必须是 2、3 或 4")

        if len(self.pose_cmd_force) != 6:
            raise ValueError("pose_cmd_force 必须包含 6 个整数")

        if self.min_ground_clearance <= 0.0:
            raise ValueError("min_ground_clearance 必须为正数")

        if self.approach_distance <= 0.0:
            raise ValueError("approach_distance 必须为正数")

        if self.approach_distance_tolerance <= 0.0:
            raise ValueError("approach_distance_tolerance 必须为正数")

        if self.approach_lateral_tolerance <= 0.0:
            raise ValueError("approach_lateral_tolerance 必须为正数")

        if self.approach_yaw_tolerance <= 0.0:
            raise ValueError("approach_yaw_tolerance_deg 必须为正数")

    @staticmethod
    def parse_int_list(raw_value, default_value):
        if isinstance(raw_value, (list, tuple)):
            try:
                return [int(value) for value in raw_value]
            except (TypeError, ValueError):
                return list(default_value)

        text = str(raw_value).strip()
        if not text:
            return list(default_value)

        normalized = text.replace(",", " ").replace(";", " ")
        try:
            return [int(part) for part in normalized.split()]
        except ValueError:
            return list(default_value)

    def arrow_detection_callback(self, message):
        if message.class_name.strip().lower() != self.arrow_class:
            return

        if message.conf < self.min_confidence:
            rospy.logwarn_throttle(
                2.0,
                "%s：忽略低置信度识别，类别=%s，置信度=%.2f < %.2f",
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
        rospy.loginfo_throttle(
            0.8,
            "%s：收到有效箭头样本，累计样本=%d，置信度=%.2f，坐标系=%s，位置=(%.3f, %.3f, %.3f)",
            NODE_NAME,
            len(self.arrow_samples),
            message.conf,
            message.pose.header.frame_id or self.detection_frame,
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            message.pose.pose.position.z,
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
            rospy.logwarn_throttle(2, "%s：无法获取当前位姿：%s", NODE_NAME, error)
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

    def transform_pose_to_frame(self, pose, target_frame):
        source_frame = pose.header.frame_id or self.detection_frame
        if source_frame == target_frame:
            target = PoseStamped()
            target.header.stamp = rospy.Time.now()
            target.header.frame_id = target_frame
            target.pose = pose.pose
            return target

        source = PoseStamped()
        source.header.stamp = rospy.Time(0)
        source.header.frame_id = source_frame
        source.pose = pose.pose

        try:
            self.tf_listener.waitForTransform(
                target_frame,
                source.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            target = self.tf_listener.transformPose(target_frame, source)
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0,
                "%s：坐标转换失败，%s -> %s：%s",
                NODE_NAME,
                source.header.frame_id,
                target_frame,
                error,
            )
            return None

        target.header.stamp = rospy.Time.now()
        target.header.frame_id = target_frame
        return target

    def transform_pose_to_map(self, pose):
        return self.transform_pose_to_frame(pose, "map")

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
        将临时 mock 箭头偏移量从 base_link 转换到 map。

        本项目中 base_link 使用 x=前方、y=右方、z=向下。mock 模式仅用于
        感知链路运行前的台架测试。
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
            (step, step),
            (2.0 * step, 0.0),
            (2.0 * step, -step),
            (2.0 * step, step),
            (2.0 * step, -2.0 * step),
            (2.0 * step, 2.0 * step),
            (3.0 * step, 0.0),
            (3.0 * step, -step),
            (3.0 * step, step),
            (3.0 * step, -2.0 * step),
            (3.0 * step, 2.0 * step),
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
            "%s：已生成箭头搜索计划，搜索点=%d，扫描航向=%s deg",
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

    def search_elapsed_seconds(self):
        if self.search_started is None:
            return 0.0
        return (rospy.Time.now() - self.search_started).to_sec()

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
                    "%s：已看到箭头，稳定锁定进度=%d/%d",
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
                "%s：箭头位置还不稳定，最大抖动=%.3fm > %.3fm",
                NODE_NAME,
                max_distance,
                self.stable_position_tolerance,
            )
            return None

        rospy.loginfo_throttle(
            1.0,
            "%s：箭头稳定锁定，样本数=%d，最大抖动=%.3fm，置信度=%.2f",
            NODE_NAME,
            self.stable_detection_count,
            max_distance,
            latest.conf,
        )
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
                "%s：arrow_yaw_mode=detection，但识别器暂未给出箭头方向，先按默认 yaw=0 处理",
                NODE_NAME,
            )

        return target

    def depth_with_ground_clearance(self, current, ground_pose):
        # 本项目中的 PoseNEDcmd 使用 NED 语义：z 轴向下为正。
        # 识别到的箭头中心按地面点处理。
        max_safe_depth = ground_pose.pose.position.z - self.min_ground_clearance
        current_depth = current.pose.position.z
        target_depth = min(current_depth, max_safe_depth)

        current_clearance = ground_pose.pose.position.z - current_depth
        target_clearance = ground_pose.pose.position.z - target_depth
        if current_clearance < self.min_ground_clearance:
            rospy.logwarn_throttle(
                1.0,
                "%s：当前对地距离 %.3fm 小于安全距离 %.3fm，下发更浅目标 z=%.3f",
                NODE_NAME,
                current_clearance,
                self.min_ground_clearance,
                target_depth,
            )

        return target_depth, target_clearance

    def build_arrow_target_from_detection(self, detection):
        current = self.get_current_pose()
        if current is None:
            return None

        arrow_in_map = self.transform_pose_to_map(detection.pose)
        if arrow_in_map is None:
            return None

        arrow_in_map = self.apply_arrow_orientation(arrow_in_map, detection, current)
        target_yaw = self.yaw_from_pose(arrow_in_map)

        target = PoseStamped()
        target.header.stamp = rospy.Time.now()
        target.header.frame_id = "map"
        target.pose.position.x = (
            arrow_in_map.pose.position.x - self.approach_distance * math.cos(target_yaw)
        )
        target.pose.position.y = (
            arrow_in_map.pose.position.y - self.approach_distance * math.sin(target_yaw)
        )
        target.pose.position.z, target_clearance = self.depth_with_ground_clearance(
            current, arrow_in_map
        )
        target.pose.orientation = arrow_in_map.pose.orientation

        rospy.loginfo(
            "%s：箭头靠近目标 map=(%.3f, %.3f, %.3f)，目标航向=%.1fdeg，期望前距=%.2fm，对地距离=%.2fm",
            NODE_NAME,
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
            math.degrees(target_yaw),
            self.approach_distance,
            target_clearance,
        )
        return target

    def is_arrived(self, current, target, max_dist=None, max_yaw=None, label="目标"):
        max_dist = self.arrive_dist if max_dist is None else max_dist
        max_yaw = self.arrive_yaw if max_yaw is None else max_yaw
        pos_error = self.xyz_distance(current, target)
        yaw_error = self.yaw_distance(current, target)
        rospy.loginfo_throttle(
            1.0,
            "%s：正在移动到 %s，位置误差=%.3fm，航向误差=%.2fdeg",
            NODE_NAME,
            label,
            pos_error,
            math.degrees(yaw_error),
        )
        return pos_error <= max_dist and yaw_error <= max_yaw

    def is_arrow_approach_ready(self, current, detection):
        if self.target_pose is None:
            return False

        arrow_in_base = self.transform_pose_to_frame(detection.pose, "base_link")
        if arrow_in_base is None:
            return False

        forward = arrow_in_base.pose.position.x
        lateral = arrow_in_base.pose.position.y
        horizontal_distance = math.sqrt(forward * forward + lateral * lateral)
        forward_error = forward - self.approach_distance
        yaw_error = self.yaw_distance(current, self.target_pose)

        rospy.loginfo_throttle(
            0.5,
            (
                "%s：箭头靠近检查，置信度=%.2f，前向=%.3fm，横向=%.3fm，"
                "水平距离=%.3fm，前向误差=%.3fm，航向误差=%.2fdeg"
            ),
            NODE_NAME,
            detection.conf,
            forward,
            lateral,
            horizontal_distance,
            forward_error,
            math.degrees(yaw_error),
        )

        return (
            abs(forward_error) <= self.approach_distance_tolerance
            and abs(lateral) <= self.approach_lateral_tolerance
            and yaw_error <= self.approach_yaw_tolerance
        )

    def publish_target(self):
        if self.target_pose is None:
            return
        self.target_pose.header.stamp = rospy.Time.now()
        if self.motion_output in ("target", "both"):
            self.target_pub.publish(self.target_pose)
        if self.motion_output in ("cmd", "both"):
            self.publish_pose_cmd(self.target_pose)

    def publish_pose_cmd(self, target_pose):
        command = PoseNEDcmd()
        command.mode = self.pose_cmd_mode
        command.target = target_pose
        (
            command.force.TX,
            command.force.TY,
            command.force.TZ,
            command.force.MX,
            command.force.MY,
            command.force.MZ,
        ) = self.pose_cmd_force
        self.pose_cmd_pub.publish(command)

    def advance_search_target(self):
        self.scan_started = None
        self.arrow_samples = []

        self.search_yaw_index += 1
        if self.search_yaw_index < len(self.scan_yaw_offsets_deg):
            self.target_pose = self.current_search_target()
            self.state = self.SEARCH_MOVE
            rospy.loginfo(
                "%s：切换到下一个扫描航向 %.1fdeg，当前搜索点=%d",
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
                "%s：移动到下一个箭头搜索点 %d/%d",
                NODE_NAME,
                self.search_point_index + 1,
                len(self.search_targets),
            )
            return True

        return False

    def restart_search_cycle(self):
        self.search_point_index = 0
        self.search_yaw_index = 0
        self.scan_started = None
        self.arrow_samples = []
        self.target_pose = self.current_search_target()
        self.state = self.SEARCH_MOVE
        rospy.logwarn(
            "%s：一整轮箭头搜索未锁定，已搜索 %.1fs < %.1fs，重新开始下一轮搜索",
            NODE_NAME,
            self.search_elapsed_seconds(),
            self.max_search_seconds,
        )

    def finish_task(self, success=True, reason=""):
        if success:
            self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
            rospy.loginfo("%s：子任务1完成，已发布 /finished", NODE_NAME)
            rospy.signal_shutdown("%s finished" % NODE_NAME)
            return

        message = "%s 失败：%s" % (NODE_NAME, reason)
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
                    "%s：mock 获取区目标 map=(%.3f, %.3f, %.3f)",
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
                if self.is_arrived(current, self.target_pose, label="mock 获取区"):
                    self.arrive_time = rospy.Time.now()
                    rospy.loginfo("%s：已到达 mock 获取区，开始保持计时", NODE_NAME)
            elif (rospy.Time.now() - self.arrive_time).to_sec() >= self.hold_seconds:
                self.finish_task(success=True)

            self.rate.sleep()

    def run_topic_mode(self):
        while not rospy.is_shutdown():
            if not self.search_targets:
                if not self.prepare_search_plan():
                    self.rate.sleep()
                    continue

            stable_arrow = None
            if self.state in (self.SEARCH_SCAN, self.MOVE_TO_ARROW):
                stable_arrow = self.current_stable_arrow()

            if self.state == self.SEARCH_SCAN and stable_arrow is not None:
                target = self.build_arrow_target_from_detection(stable_arrow)
                if target is not None:
                    self.target_pose = target
                    self.state = self.MOVE_TO_ARROW
                    self.arrive_time = None
                    rospy.loginfo("%s：稳定锁定箭头，开始视觉靠近", NODE_NAME)

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
                    label="箭头搜索位姿",
                ):
                    self.scan_started = rospy.Time.now()
                    self.arrow_samples = []
                    self.state = self.SEARCH_SCAN
                    rospy.loginfo(
                        "%s：开始扫描箭头，搜索点=%d/%d，扫描航向=%.1fdeg",
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
                        if self.search_timed_out():
                            self.finish_task(
                                success=False,
                                reason=(
                                    "完整搜索一轮且超过 %.1fs，仍未找到箭头"
                                    % self.max_search_seconds
                                ),
                            )
                            return
                        self.restart_search_cycle()

            elif self.state == self.MOVE_TO_ARROW:
                if stable_arrow is not None:
                    target = self.build_arrow_target_from_detection(stable_arrow)
                    if target is not None:
                        self.target_pose = target

                self.publish_target()
                if stable_arrow is None:
                    self.arrive_time = None
                    rospy.logwarn_throttle(
                        2.0,
                        "%s：正在靠近箭头，但暂未获得稳定识别，继续等待",
                        NODE_NAME,
                    )
                elif self.is_arrow_approach_ready(current, stable_arrow):
                    if self.arrive_time is None:
                        self.arrive_time = rospy.Time.now()
                        rospy.loginfo(
                            "%s：箭头靠近条件已稳定满足，开始保持计时",
                            NODE_NAME,
                        )
                    elif (
                        rospy.Time.now() - self.arrive_time
                    ).to_sec() >= self.hold_seconds:
                        self.finish_task(success=True)
                else:
                    self.arrive_time = None

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
