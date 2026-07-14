#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
任务 3 子任务 3 测试：寻找指定颜色方框并投放信标球。

detection 模式逻辑：
  1. 从 /obj/target_message 读取目标颜色方框；
  2. 按子任务 1 的前方搜索逻辑寻找并稳定锁定方框；
  3. 根据实时中心误差分步靠近方框中心；
  4. 持续读取视觉结果并在方框中心附近细对齐；
  5. 停稳、打开夹爪、关闭夹爪并结束。

本脚本默认通过 /cmd/pose/ned 发布运动指令，也可以通过 motion_output=target
或 motion_output=both 继续发布旧版 /target 话题。灯光和夹爪控制使用
/cmd/actuator。

记录：
2026.7.13
  执行器下行话题调整为 /cmd/actuator。
"""

import math

import rospy
import tf
from auv_control.msg import ActuatorControl, PoseNEDcmd, TargetDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task3_3_inspect_and_drop"

# =========================
# 可调默认参数
# =========================
# 这些值是水池/实艇调试时优先调整的位置。
# 它们仍然是 ROS 参数，因此 roslaunch 可以在不改代码的情况下覆盖。

DEFAULT_RATE = 5.0
DEFAULT_TARGET_MODE = "detection"  # 可选 mock、topic 或 detection
DEFAULT_TARGET_TOPIC = "/task3/pipeline_target"
DEFAULT_DETECTION_TOPIC = "/obj/target_message"
DEFAULT_DETECTION_FRAME = "camera"
DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_TARGET_COLOR = "yellow"
DEFAULT_MOCK_DETECTED_COLORS = ["yellow", "green", "red"]
DEFAULT_DEBUG_LOG_DETECTIONS = True
DEFAULT_DEBUG_LOG_TARGETS = True

# 运动指令输出。
# cmd：发布最新的 /cmd/pose/ned PoseNEDcmd 接口。
# target：发布旧版 /target PoseStamped 接口。
# both：迁移/调试阶段同时发布两个接口。
DEFAULT_MOTION_OUTPUT = "cmd"  # 可选 cmd、target 或 both
DEFAULT_LEGACY_TARGET_TOPIC = "/target"
DEFAULT_POSE_CMD_TOPIC = "/cmd/pose/ned"
DEFAULT_POSE_CMD_MODE = 4  # 2=定深，3=定深+定航向，4=DPROV 定点保持
DEFAULT_POSE_CMD_FORCE = [0, 0, 0, 0, 0, 0]  # 六自由度力/力矩：TX、TY、TZ、MX、MY、MZ

# 仅 mock 模式使用：相对 base_link 的旧临时投放点。
# 本项目 base_link 约定：x=前方，y=右方，z=向下。
DEFAULT_DROP_FORWARD = 0.50
DEFAULT_DROP_LEFT = 0.30
DEFAULT_DROP_DOWN = 0.00

# mock/topic 简单模式下的到达容差。
DEFAULT_ARRIVE_DIST = 0.12
DEFAULT_ARRIVE_YAW_DEG = 5.0

# 粗靠近前的识别搜索和稳定锁定。
DEFAULT_SEARCH_STEP = 0.30
DEFAULT_MAX_SEARCH_POINTS = 14
DEFAULT_SCAN_YAW_OFFSETS_DEG = [0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0]
DEFAULT_SCAN_HOLD_SECONDS = 1.5
DEFAULT_MAX_SEARCH_SECONDS = 300.0
DEFAULT_SEARCH_ARRIVE_DIST = 0.15
DEFAULT_SEARCH_ARRIVE_YAW_DEG = 8.0
DEFAULT_STABLE_DETECTION_COUNT = 5
DEFAULT_STABLE_POSITION_TOLERANCE = 0.15
DEFAULT_DETECTION_TIMEOUT = 2.0
DEFAULT_MAX_DETECTION_WAIT_SECONDS = DEFAULT_MAX_SEARCH_SECONDS

# 机器人允许投放时，方框中心在 base_link 下的期望位置。
# x=前方，y=右方。如果投放机构与 base_link 对齐，保持为 0 即可；
# 如果相机或夹爪存在固定偏移，后续再调这些值。
DEFAULT_CENTER_TARGET_FORWARD = 0.0
DEFAULT_CENTER_TARGET_RIGHT = 0.0

# 对识别到的彩色方框中心进行粗视觉靠近。
DEFAULT_COARSE_ARRIVE_DIST = 0.18
DEFAULT_COARSE_ARRIVE_YAW_DEG = 8.0
DEFAULT_COARSE_CENTER_TOLERANCE_X = 0.18
DEFAULT_COARSE_CENTER_TOLERANCE_Y = 0.18
DEFAULT_COARSE_MAX_STEP = 0.20
DEFAULT_COARSE_MIN_STEP = 0.05
DEFAULT_COARSE_GAIN = 0.8
DEFAULT_COARSE_COMMAND_PERIOD = 0.5

# 在彩色方框上方进行精细视觉对齐。
DEFAULT_FINE_TOLERANCE_X = 0.08
DEFAULT_FINE_TOLERANCE_Y = 0.08
DEFAULT_FINE_MAX_STEP = 0.10
DEFAULT_FINE_MIN_STEP = 0.03
DEFAULT_FINE_GAIN = 0.8
DEFAULT_FINE_COMMAND_PERIOD = 0.4
DEFAULT_FINE_HOLD_SECONDS = 1.0
DEFAULT_FRAME_LOST_TIMEOUT = 2.0

# 投放动作。
DEFAULT_HOLD_SECONDS = 1.0
DEFAULT_OPEN_SECONDS = 3.0
DEFAULT_CLOSE_SECONDS = 1.0

# 执行器默认值。
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
    SEARCH_MOVE = 1
    SEARCH_SCAN = 2
    MOVE_NEAR_FRAME = 3
    FINE_ALIGN_FRAME = 4
    HOLD_BEFORE_DROP = 5
    OPEN_CLAMP = 6
    CLOSE_CLAMP = 7

    STATE_NAMES = {
        LOCK_FRAME: "锁定方框",
        SEARCH_MOVE: "移动到搜索位姿",
        SEARCH_SCAN: "扫描方框",
        MOVE_NEAR_FRAME: "粗靠近方框",
        FINE_ALIGN_FRAME: "细对齐方框",
        HOLD_BEFORE_DROP: "投放前保持",
        OPEN_CLAMP: "打开夹爪",
        CLOSE_CLAMP: "关闭夹爪",
    }

    COLOR_LIGHTS = {
        "yellow": (0, 1, 0),
        "green": (0, 0, 1),
        "red": (1, 0, 0),
        "off": (0, 0, 0),
    }

    VALID_COLORS = ("yellow", "green", "red")

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
        self.debug_log_detections = bool(
            rospy.get_param("~debug_log_detections", DEFAULT_DEBUG_LOG_DETECTIONS)
        )
        self.debug_log_targets = bool(
            rospy.get_param("~debug_log_targets", DEFAULT_DEBUG_LOG_TARGETS)
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
        legacy_max_wait = float(
            rospy.get_param(
                "~max_detection_wait_seconds", DEFAULT_MAX_DETECTION_WAIT_SECONDS
            )
        )
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
            rospy.get_param("~max_search_seconds", legacy_max_wait)
        )
        self.search_arrive_dist = float(
            rospy.get_param("~search_arrive_dist", DEFAULT_SEARCH_ARRIVE_DIST)
        )
        self.search_arrive_yaw = math.radians(
            float(
                rospy.get_param(
                    "~search_arrive_yaw_deg", DEFAULT_SEARCH_ARRIVE_YAW_DEG
                )
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
        self.center_target_forward = float(
            rospy.get_param(
                "~center_target_forward", DEFAULT_CENTER_TARGET_FORWARD
            )
        )
        self.center_target_right = float(
            rospy.get_param("~center_target_right", DEFAULT_CENTER_TARGET_RIGHT)
        )
        self.coarse_center_tolerance_x = float(
            rospy.get_param(
                "~coarse_center_tolerance_x", DEFAULT_COARSE_CENTER_TOLERANCE_X
            )
        )
        self.coarse_center_tolerance_y = float(
            rospy.get_param(
                "~coarse_center_tolerance_y", DEFAULT_COARSE_CENTER_TOLERANCE_Y
            )
        )
        self.coarse_max_step = float(
            rospy.get_param("~coarse_max_step", DEFAULT_COARSE_MAX_STEP)
        )
        self.coarse_min_step = float(
            rospy.get_param("~coarse_min_step", DEFAULT_COARSE_MIN_STEP)
        )
        self.coarse_gain = float(
            rospy.get_param("~coarse_gain", DEFAULT_COARSE_GAIN)
        )
        self.coarse_command_period = float(
            rospy.get_param("~coarse_command_period", DEFAULT_COARSE_COMMAND_PERIOD)
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
        self.search_targets = []
        self.search_point_index = 0
        self.search_yaw_index = 0
        self.state_started = rospy.Time.now()
        self.search_started = None
        self.scan_started = None
        self.align_started = None
        self.frame_lost_started = None
        self.last_center_error = None
        self.last_detection_conf = None
        self.last_coarse_command_time = rospy.Time(0)
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
            (
                "%s：启动子任务3，目标颜色=%s，目标模式=%s，识别话题=%s，"
                "最低置信度=%.2f，中心期望=(前 %.2fm，右 %.2fm)"
            ),
            NODE_NAME,
            self.target_color,
            self.target_mode,
            self.detection_topic,
            self.min_confidence,
            self.center_target_forward,
            self.center_target_right,
        )
        rospy.loginfo(
            (
                "%s：粗靠近容差=(x %.2fm，y %.2fm)，步长=[%.2f, %.2f]m；"
                "细对齐容差=(x %.2fm，y %.2fm)，步长=[%.2f, %.2f]m"
            ),
            NODE_NAME,
            self.coarse_center_tolerance_x,
            self.coarse_center_tolerance_y,
            self.coarse_min_step,
            self.coarse_max_step,
            self.fine_tolerance_x,
            self.fine_tolerance_y,
            self.fine_min_step,
            self.fine_max_step,
        )
        rospy.loginfo(
            (
                "%s：方框搜索参数 step=%.2fm，搜索点=%d，扫描航向=%s，"
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
            "%s：mock 偏移：前 %.2fm，左 %.2fm，下 %.2fm",
            NODE_NAME,
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
            raise ValueError("target_mode 必须是 mock、topic 或 detection")

        if not self.target_color:
            raise ValueError("target_color 不能为空")

        if self.target_color not in self.VALID_COLORS:
            rospy.logwarn(
                "%s：目标颜色 %s 没有内置灯光映射，执行器颜色灯将保持关闭",
                NODE_NAME,
                self.target_color,
            )

        if self.stable_detection_count <= 0:
            raise ValueError("stable_detection_count 必须为正数")

        if self.max_search_points <= 0:
            raise ValueError("max_search_points 必须为正数")

        if not self.scan_yaw_offsets_deg:
            raise ValueError("scan_yaw_offsets_deg 不能为空")

        if self.max_search_seconds <= 0.0:
            raise ValueError("max_search_seconds 必须为正数")

        if self.coarse_min_step > self.coarse_max_step:
            raise ValueError("coarse_min_step 不能大于 coarse_max_step")

        if self.fine_min_step > self.fine_max_step:
            raise ValueError("fine_min_step 不能大于 fine_max_step")

        if self.motion_output not in ("cmd", "target", "both"):
            raise ValueError("motion_output 必须是 cmd、target 或 both")

        if self.pose_cmd_mode not in (2, 3, 4):
            raise ValueError("pose_cmd_mode 必须是 2、3 或 4")

        if len(self.pose_cmd_force) != 6:
            raise ValueError("pose_cmd_force 必须包含 6 个整数")

        if self.target_mode == "mock" and self.target_color not in self.mock_detected_colors:
            raise ValueError(
                "target_color {} 不在 mock_detected_colors {} 中".format(
                    self.target_color, self.mock_detected_colors
                )
            )

    def target_callback(self, message):
        self.topic_target_pose = message

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

    @staticmethod
    def parse_float_list(raw_value, default_value):
        if isinstance(raw_value, (list, tuple)):
            try:
                return [float(value) for value in raw_value]
            except (TypeError, ValueError):
                return list(default_value)

        text = str(raw_value).strip()
        if not text:
            return list(default_value)

        normalized = text.replace(",", " ").replace(";", " ")
        try:
            return [float(part) for part in normalized.split()]
        except ValueError:
            return list(default_value)

    def detection_callback(self, message):
        detected_color = message.class_name.strip().lower()
        pose = message.pose.pose.position
        frame_id = message.pose.header.frame_id or self.detection_frame

        if self.debug_log_detections:
            rospy.loginfo_throttle(
                0.8,
                (
                    "%s：识别输入，状态=%s，类别=%s，目标颜色=%s，置信度=%.2f，"
                    "类型=%s，坐标系=%s，位置=(%.3f, %.3f, %.3f)"
                ),
                NODE_NAME,
                self.state_name(),
                message.class_name,
                self.target_color,
                message.conf,
                message.type,
                frame_id,
                pose.x,
                pose.y,
                pose.z,
            )

        if detected_color != self.target_color:
            if self.debug_log_detections:
                rospy.loginfo_throttle(
                    2.0,
                    "%s：忽略非目标颜色，识别类别=%s，目标颜色=%s",
                    NODE_NAME,
                    message.class_name,
                    self.target_color,
                )
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

        if not message.pose.header.frame_id:
            message.pose.header.frame_id = self.detection_frame

        now = rospy.Time.now()
        self.detection_samples.append((now, message))
        self.last_detection_conf = message.conf
        max_samples = max(self.stable_detection_count * 3, 10)
        if len(self.detection_samples) > max_samples:
            self.detection_samples = self.detection_samples[-max_samples:]

        if self.debug_log_detections:
            rospy.loginfo_throttle(
                0.8,
                "%s：接受目标方框样本，颜色=%s，样本数=%d，置信度=%.2f",
                NODE_NAME,
                self.target_color,
                len(self.detection_samples),
                message.conf,
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
                "%s：坐标转换失败，%s -> %s：%s",
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

    def build_mock_drop_target(self):
        """
        将临时 mock 管道/方框位置从 base_link 转换到 map。

        本项目中 base_link 使用 x=前方、y=右方、z=向下，因此正的 drop_left
        会转换成负的 y 偏移。
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
                "%s：等待外部投放目标，话题=%s",
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
                "%s：mock 可用颜色=%s，当前选择=%s",
                NODE_NAME,
                ",".join(self.mock_detected_colors),
                self.target_color,
            )
            return self.build_mock_drop_target()

        return self.build_topic_drop_target()

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
        self.detection_samples = []
        self.target_pose = self.current_search_target()
        self.set_state(self.SEARCH_MOVE, "已生成彩色方框搜索计划")

        rospy.loginfo(
            "%s：已生成 %s 方框搜索计划，前方搜索点=%d，扫描航向=%s deg",
            NODE_NAME,
            self.target_color,
            len(self.search_targets),
            self.scan_yaw_offsets_deg,
        )
        return True

    def current_search_target(self):
        return self.search_targets[self.search_point_index][self.search_yaw_index]

    def search_elapsed_seconds(self):
        if self.search_started is None:
            return 0.0
        return (rospy.Time.now() - self.search_started).to_sec()

    def search_timed_out(self):
        return self.search_elapsed_seconds() >= self.max_search_seconds

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
                    "%s：已看到 %s 方框，稳定锁定进度=%d/%d",
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
                "%s：%s 方框位置还不稳定，最大抖动=%.3fm > %.3fm",
                NODE_NAME,
                self.target_color,
                max_distance,
                self.stable_position_tolerance,
            )
            return None

        rospy.loginfo_throttle(
            1.0,
            "%s：稳定锁定 %s 方框，样本数=%d，最大抖动=%.3fm，置信度=%.2f",
            NODE_NAME,
            self.target_color,
            self.stable_detection_count,
            max_distance,
            latest.conf,
        )
        return latest

    def frame_error_in_base(self, detection):
        frame_in_base = self.transform_pose_to_frame(detection.pose, "base_link")
        if frame_in_base is None:
            return None, None

        error_x = frame_in_base.pose.position.x - self.center_target_forward
        error_y = frame_in_base.pose.position.y - self.center_target_right
        return (error_x, error_y), frame_in_base

    def build_visual_step_target(
        self,
        detection,
        tolerance_x,
        tolerance_y,
        min_step,
        max_step,
        gain,
        label,
    ):
        error_xy, frame_in_base = self.frame_error_in_base(detection)
        if error_xy is None:
            return None, None

        error_x, error_y = error_xy
        step_x = self.bounded_step(
            error_x,
            tolerance_x,
            min_step,
            max_step,
            gain,
        )
        step_y = self.bounded_step(
            error_y,
            tolerance_y,
            min_step,
            max_step,
            gain,
        )
        self.last_center_error = (error_x, error_y)

        if step_x == 0.0 and step_y == 0.0:
            current = self.get_current_pose()
            return current, (error_x, error_y, step_x, step_y)

        target = self.transform_base_link_offset_to_map(step_x, step_y, 0.0)
        if target is not None:
            rospy.loginfo_throttle(
                0.5,
                (
                    "%s：%s，%s 方框中心 base_link=(%.3f, %.3f, %.3f)，"
                    "期望=(%.3f, %.3f)，误差=(%.3f, %.3f)，步长=(%.3f, %.3f)"
                ),
                NODE_NAME,
                label,
                self.target_color,
                frame_in_base.pose.position.x,
                frame_in_base.pose.position.y,
                frame_in_base.pose.position.z,
                self.center_target_forward,
                self.center_target_right,
                error_x,
                error_y,
                step_x,
                step_y,
            )
        return target, (error_x, error_y, step_x, step_y)

    def build_coarse_target_from_detection(self, detection):
        return self.build_visual_step_target(
            detection,
            self.coarse_center_tolerance_x,
            self.coarse_center_tolerance_y,
            self.coarse_min_step,
            self.coarse_max_step,
            self.coarse_gain,
            "粗靠近",
        )

    def build_fine_alignment_target(self, detection):
        return self.build_visual_step_target(
            detection,
            self.fine_tolerance_x,
            self.fine_tolerance_y,
            self.fine_min_step,
            self.fine_max_step,
            self.fine_gain,
            "细对齐",
        )

    @staticmethod
    def center_is_inside(error_x, error_y, tolerance_x, tolerance_y):
        return abs(error_x) <= tolerance_x and abs(error_y) <= tolerance_y

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

    def publish_target(self):
        if self.target_pose is None:
            return
        self.target_pose.header.stamp = rospy.Time.now()
        if self.motion_output in ("target", "both"):
            self.target_pub.publish(self.target_pose)
        if self.motion_output in ("cmd", "both"):
            self.publish_pose_cmd(self.target_pose)
        if self.debug_log_targets:
            rospy.loginfo_throttle(
                1.0,
                "%s：发布 %s 目标 map=(%.3f, %.3f, %.3f)，输出=%s",
                NODE_NAME,
                self.state_name(),
                self.target_pose.pose.position.x,
                self.target_pose.pose.position.y,
                self.target_pose.pose.position.z,
                self.motion_output,
            )

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
        if not self.show_color_light:
            return "off"
        if self.target_color not in self.COLOR_LIGHTS:
            return "off"
        return self.target_color

    def state_elapsed(self):
        return (rospy.Time.now() - self.state_started).to_sec()

    def state_name(self, state=None):
        state = self.state if state is None else state
        return self.STATE_NAMES.get(state, "未知状态")

    def set_state(self, state, reason=""):
        old_state = self.state
        old_elapsed = self.state_elapsed()
        self.state = state
        self.state_started = rospy.Time.now()
        detail = ", %s" % reason if reason else ""
        rospy.loginfo(
            "%s：状态切换 %s -> %s，上一状态持续 %.1fs%s",
            NODE_NAME,
            self.state_name(old_state),
            self.state_name(state),
            old_elapsed,
            detail,
        )

    def advance_search_target(self):
        self.scan_started = None
        self.detection_samples = []

        self.search_yaw_index += 1
        if self.search_yaw_index < len(self.scan_yaw_offsets_deg):
            self.target_pose = self.current_search_target()
            rospy.loginfo(
                "%s：切换到下一个扫描航向 %.1fdeg，%s 方框搜索点=%d",
                NODE_NAME,
                self.scan_yaw_offsets_deg[self.search_yaw_index],
                self.target_color,
                self.search_point_index + 1,
            )
            self.set_state(self.SEARCH_MOVE, "切换扫描航向")
            return True

        self.search_point_index += 1
        self.search_yaw_index = 0
        if self.search_point_index < len(self.search_targets):
            self.target_pose = self.current_search_target()
            rospy.loginfo(
                "%s：移动到下一个 %s 方框搜索点 %d/%d",
                NODE_NAME,
                self.target_color,
                self.search_point_index + 1,
                len(self.search_targets),
            )
            self.set_state(self.SEARCH_MOVE, "切换搜索点")
            return True

        return False

    def restart_search_cycle(self):
        self.search_point_index = 0
        self.search_yaw_index = 0
        self.scan_started = None
        self.detection_samples = []
        self.target_pose = self.current_search_target()
        rospy.logwarn(
            (
                "%s：一整轮 %s 方框搜索未锁定，已搜索 %.1fs < %.1fs，"
                "重新开始下一轮搜索"
            ),
            NODE_NAME,
            self.target_color,
            self.search_elapsed_seconds(),
            self.max_search_seconds,
        )
        self.set_state(self.SEARCH_MOVE, "重新开始搜索")

    def finish_task(self, success=True, reason=""):
        self.publish_actuator(self.clamp_closed, "off")
        if success:
            self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
            rospy.loginfo("%s：子任务3完成，已发布 /finished", NODE_NAME)
            rospy.signal_shutdown("%s finished" % NODE_NAME)
            return

        message = "%s 失败：%s" % (NODE_NAME, reason)
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
                    "%s：固定投放目标 map=(%.3f, %.3f, %.3f)",
                    NODE_NAME,
                    self.target_pose.pose.position.x,
                    self.target_pose.pose.position.y,
                    self.target_pose.pose.position.z,
                )
                self.set_state(self.MOVE_NEAR_FRAME, "已生成固定投放目标")

            elif self.state == self.MOVE_NEAR_FRAME:
                current = self.get_current_pose()
                if current is None:
                    self.rate.sleep()
                    continue

                self.publish_target()
                if self.is_arrived(current, self.target_pose, label="投放区域"):
                    rospy.loginfo("%s：已到达投放区域", NODE_NAME)
                    self.set_state(self.HOLD_BEFORE_DROP, "已到达投放区域")

            elif self.state == self.HOLD_BEFORE_DROP:
                self.publish_target()
                self.publish_actuator(self.clamp_closed, self.active_color())
                if self.state_elapsed() >= self.hold_seconds:
                    rospy.loginfo("%s：投放前保持完成，打开夹爪投放信标球", NODE_NAME)
                    self.set_state(self.OPEN_CLAMP, "投放前保持完成")

            elif self.state == self.OPEN_CLAMP:
                self.publish_target()
                self.publish_actuator(self.clamp_open, self.active_color())
                if self.state_elapsed() >= self.open_seconds:
                    rospy.loginfo("%s：夹爪打开时间完成，关闭夹爪", NODE_NAME)
                    self.set_state(self.CLOSE_CLAMP, "夹爪打开时间完成")

            elif self.state == self.CLOSE_CLAMP:
                self.publish_target()
                self.publish_actuator(self.clamp_closed, "off")
                if self.state_elapsed() >= self.close_seconds:
                    self.finish_task(success=True)

            self.rate.sleep()

    def run_detection_mode(self):
        while not rospy.is_shutdown():
            self.publish_actuator(self.clamp_closed, self.active_color())

            if not self.search_targets:
                if not self.prepare_search_plan():
                    self.rate.sleep()
                    continue

            stable_detection = None
            if self.state in (self.SEARCH_SCAN, self.MOVE_NEAR_FRAME):
                stable_detection = self.current_stable_detection()

            if self.state == self.SEARCH_SCAN and stable_detection is not None:
                target, error_step = self.build_coarse_target_from_detection(
                    stable_detection
                )
                if target is None or error_step is None:
                    self.rate.sleep()
                    continue

                error_x, error_y, step_x, step_y = error_step
                self.target_pose = target
                self.last_coarse_command_time = rospy.Time.now()
                self.align_started = None
                self.frame_lost_started = None
                rospy.loginfo(
                    (
                        "%s：锁定 %s 方框，开始粗视觉靠近，"
                        "误差=(%.3f, %.3f)，步长=(%.3f, %.3f)"
                    ),
                    NODE_NAME,
                    self.target_color,
                    error_x,
                    error_y,
                    step_x,
                    step_y,
                )
                self.set_state(
                    self.MOVE_NEAR_FRAME,
                    "稳定锁定 %s 方框" % self.target_color,
                )

            if self.state == self.SEARCH_MOVE:
                current = self.get_current_pose()
                if current is None:
                    self.rate.sleep()
                    continue

                self.publish_target()
                if self.is_arrived(
                    current,
                    self.target_pose,
                    self.search_arrive_dist,
                    self.search_arrive_yaw,
                    label="%s 方框搜索位姿" % self.target_color,
                ):
                    self.scan_started = rospy.Time.now()
                    self.detection_samples = []
                    rospy.loginfo(
                        "%s：开始扫描 %s 方框，搜索点=%d/%d，扫描航向=%.1fdeg",
                        NODE_NAME,
                        self.target_color,
                        self.search_point_index + 1,
                        len(self.search_targets),
                        self.scan_yaw_offsets_deg[self.search_yaw_index],
                    )
                    self.set_state(self.SEARCH_SCAN, "已到达搜索位姿")

            elif self.state == self.SEARCH_SCAN:
                self.publish_target()
                if self.scan_started is None:
                    self.scan_started = rospy.Time.now()

                elapsed = (rospy.Time.now() - self.scan_started).to_sec()
                rospy.loginfo_throttle(
                    1.0,
                    (
                        "%s：正在扫描 %s 方框，话题=%s，搜索点=%d/%d，"
                        "航向=%.1fdeg，已停留=%.1fs/%.1fs"
                    ),
                    NODE_NAME,
                    self.target_color,
                    self.detection_topic,
                    self.search_point_index + 1,
                    len(self.search_targets),
                    self.scan_yaw_offsets_deg[self.search_yaw_index],
                    elapsed,
                    self.scan_hold_seconds,
                )
                if elapsed >= self.scan_hold_seconds:
                    if not self.advance_search_target():
                        if self.search_timed_out():
                            self.finish_task(
                                success=False,
                                reason=(
                                    "%s 方框完整搜索一轮且超过 %.1fs 后仍未找到"
                                    % (self.target_color, self.max_search_seconds)
                                ),
                            )
                            return
                        self.restart_search_cycle()

            elif self.state == self.MOVE_NEAR_FRAME:
                detection = stable_detection
                if detection is None:
                    if self.frame_lost_started is None:
                        self.frame_lost_started = rospy.Time.now()

                    self.publish_target()
                    lost_seconds = (rospy.Time.now() - self.frame_lost_started).to_sec()
                    rospy.logwarn_throttle(
                        1.0,
                        "%s：粗靠近阶段等待稳定 %s 方框，已等待 %.1fs",
                        NODE_NAME,
                        self.target_color,
                        lost_seconds,
                    )
                    if lost_seconds >= self.frame_lost_timeout:
                        rospy.logwarn(
                            "%s：粗靠近阶段 %s 方框持续不稳定，返回搜索",
                            NODE_NAME,
                            self.target_color,
                        )
                        self.detection_samples = []
                        self.target_pose = self.current_search_target()
                        self.align_started = None
                        self.frame_lost_started = None
                        self.scan_started = None
                        self.set_state(
                            self.SEARCH_MOVE,
                            "粗靠近阶段丢失稳定方框",
                        )
                    self.rate.sleep()
                    continue

                self.frame_lost_started = None
                now = rospy.Time.now()
                if (
                    self.target_pose is None
                    or (now - self.last_coarse_command_time).to_sec()
                    >= self.coarse_command_period
                ):
                    target, error_step = self.build_coarse_target_from_detection(
                        detection
                    )
                    if target is None or error_step is None:
                        self.rate.sleep()
                        continue

                    error_x, error_y, step_x, step_y = error_step
                    self.target_pose = target
                    self.last_coarse_command_time = now

                    rospy.loginfo_throttle(
                        0.5,
                        (
                            "%s：粗靠近中心检查，颜色=%s，误差=(%.3f, %.3f)m，"
                            "步长=(%.3f, %.3f)m，容差=(%.3f, %.3f)m"
                        ),
                        NODE_NAME,
                        self.target_color,
                        error_x,
                        error_y,
                        step_x,
                        step_y,
                        self.coarse_center_tolerance_x,
                        self.coarse_center_tolerance_y,
                    )

                    if self.center_is_inside(
                        error_x,
                        error_y,
                        self.coarse_center_tolerance_x,
                        self.coarse_center_tolerance_y,
                    ):
                        rospy.loginfo(
                            "%s：方框中心进入粗容差，开始细视觉对齐",
                            NODE_NAME,
                        )
                        self.align_started = None
                        self.frame_lost_started = None
                        self.last_fine_command_time = rospy.Time(0)
                        self.set_state(
                            self.FINE_ALIGN_FRAME,
                            "中心进入粗容差",
                        )

                self.publish_target()

            elif self.state == self.FINE_ALIGN_FRAME:
                detection = self.latest_detection()
                if detection is None:
                    if self.frame_lost_started is None:
                        self.frame_lost_started = rospy.Time.now()

                    self.publish_target()
                    lost_seconds = (rospy.Time.now() - self.frame_lost_started).to_sec()
                    rospy.logwarn_throttle(
                        1.0,
                        "%s：细对齐阶段丢失 %s 方框，已丢失 %.1fs",
                        NODE_NAME,
                        self.target_color,
                        lost_seconds,
                    )
                    if lost_seconds >= self.frame_lost_timeout:
                        rospy.logwarn(
                            "%s：细对齐阶段 %s 方框丢失超时，返回搜索",
                            NODE_NAME,
                            self.target_color,
                        )
                        self.detection_samples = []
                        self.target_pose = self.current_search_target()
                        self.align_started = None
                        self.frame_lost_started = None
                        self.scan_started = None
                        self.set_state(
                            self.SEARCH_MOVE,
                            "细对齐阶段丢失方框",
                        )
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

                    error_x, error_y, step_x, step_y = error_xy
                    self.target_pose = target
                    self.last_fine_command_time = now

                    rospy.loginfo_throttle(
                        0.5,
                        (
                            "%s：细对齐中心检查，颜色=%s，误差=(%.3f, %.3f)m，"
                            "步长=(%.3f, %.3f)m，容差=(%.3f, %.3f)m"
                        ),
                        NODE_NAME,
                        self.target_color,
                        error_x,
                        error_y,
                        step_x,
                        step_y,
                        self.fine_tolerance_x,
                        self.fine_tolerance_y,
                    )

                    aligned = self.center_is_inside(
                        error_x,
                        error_y,
                        self.fine_tolerance_x,
                        self.fine_tolerance_y,
                    )
                    if aligned:
                        if self.align_started is None:
                            self.align_started = now
                            rospy.loginfo(
                                "%s：%s 方框中心进入细容差，开始保持确认",
                                NODE_NAME,
                                self.target_color,
                            )
                        elif (
                            now - self.align_started
                        ).to_sec() >= self.fine_hold_seconds:
                            rospy.loginfo(
                                "%s：细对齐已稳定 %.1fs，准备投放信标球",
                                NODE_NAME,
                                self.fine_hold_seconds,
                            )
                            self.set_state(
                                self.HOLD_BEFORE_DROP,
                                "方框中心已确认",
                            )
                    else:
                        if self.align_started is not None:
                            rospy.loginfo(
                                "%s：%s 方框中心离开细容差，重置保持确认",
                                NODE_NAME,
                                self.target_color,
                            )
                        self.align_started = None

                self.publish_target()

            elif self.state == self.HOLD_BEFORE_DROP:
                self.publish_target()
                self.publish_actuator(self.clamp_closed, self.active_color())
                if self.state_elapsed() >= self.hold_seconds:
                    if self.last_center_error is not None:
                        rospy.loginfo(
                            (
                                "%s：打开夹爪投放信标球，目标颜色=%s，"
                                "最终中心误差=(%.3f, %.3f)m"
                            ),
                            NODE_NAME,
                            self.target_color,
                            self.last_center_error[0],
                            self.last_center_error[1],
                        )
                    else:
                        rospy.loginfo(
                            "%s：打开夹爪投放信标球，目标颜色=%s",
                            NODE_NAME,
                            self.target_color,
                        )
                    self.set_state(self.OPEN_CLAMP, "中心保持完成，打开夹爪")

            elif self.state == self.OPEN_CLAMP:
                self.publish_target()
                self.publish_actuator(self.clamp_open, self.active_color())
                if self.state_elapsed() >= self.open_seconds:
                    rospy.loginfo("%s：投放窗口完成，关闭夹爪", NODE_NAME)
                    self.set_state(self.CLOSE_CLAMP, "投放窗口完成")

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
