#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
任务 3 子任务 2 测试：读取 ArUco 标记编号并点亮对应颜色灯。

本脚本不主动移动机器人。人工把机器人停在 ArUco 正前方后，脚本记录启动时
map -> base_link 位姿，并通过 /cmd/pose/ned 的 mode=3 持续保持深度和航向，
同时根据位置误差及 /status/auv 速度反馈计算 TX/TY 保持水平定点。
启动后先悬停10秒，悬停结束才开始累计 ArUco 稳定识别帧；识别成功后点亮
对应颜色灯3秒，随后熄灯并结束任务。

默认模式从 /obj/target_message 读取鱼眼 ArUco 的 TargetDetection，并从
class_name/type 解析编号。mock 模式仍保留，用于没有相机/感知节点时的台架测试。

任务规则映射：
  1,2 -> yellow
  3,4 -> green
  5,6 -> red

本脚本只使用较新的 /cmd/actuator 话题。

记录：
2026.7.13
  执行器下行话题调整为 /cmd/actuator。
"""

import math
import re

import rospy
import tf

from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from auv_control.msg import AUVData, ActuatorControl, PoseNEDcmd, TargetDetection


NODE_NAME = "test_task3_2_get_task"

# =========================
# 可调默认参数
# =========================
# 这些值是水池/实艇调试时优先调整的位置。
# 它们仍然是 ROS 参数，因此 roslaunch 可以在不改代码的情况下覆盖。

DEFAULT_RATE = 10.0
DEFAULT_INPUT_MODE = "topic"  # 可选 topic 或 mock
DEFAULT_ARUCO_TOPIC = "/obj/target_message"
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MOCK_ARUCO_IDS = [1, 3, 5, 2, 4, 6]

# max_topic_markers=1 表示本测试读到一个真实标记即可结束。
# 如果希望脚本持续响应新的标记，可以设置为 0。
DEFAULT_MAX_TOPIC_MARKERS = 1
DEFAULT_STABLE_MARKER_COUNT = 3
DEFAULT_MARKER_TIMEOUT = 1.0

DEFAULT_LIGHT_SECONDS = 3.0
DEFAULT_GAP_SECONDS = 0.5

# 子任务2只保持人工放置后的当前位置，不进行搜索或视觉移动。
MODE_DEPTH_HEADING = 3
DEFAULT_POSE_CMD_TOPIC = "/cmd/pose/ned"
DEFAULT_STATUS_TOPIC = "/status/auv"
DEFAULT_STATUS_TIMEOUT = 0.5
DEFAULT_STATUS_WAIT_TIMEOUT = 5.0
DEFAULT_STATUS_LINEAR_VELOCITY_SCALE = 1.0
DEFAULT_HOLD_MAP_FRAME = "map"
DEFAULT_HOLD_BASE_FRAME = "base_link"
DEFAULT_INITIAL_HOVER_SECONDS = 10.0
DEFAULT_HOLD_POSE_TIMEOUT = 5.0
DEFAULT_HOLD_FORWARD_POSITION_GAIN = 600.0
DEFAULT_HOLD_LATERAL_POSITION_GAIN = 600.0
DEFAULT_HOLD_FORWARD_VELOCITY_DAMPING = 300.0
DEFAULT_HOLD_LATERAL_VELOCITY_DAMPING = 300.0
DEFAULT_HOLD_MAX_FORCE = 120.0
DEFAULT_HOLD_POSITION_TOLERANCE = 0.02
DEFAULT_HOLD_SPEED_DEADBAND = 0.03
DEFAULT_HOLD_FORCE_STEP = 50.0
DEFAULT_HOLD_TX_SIGN = 1.0
DEFAULT_HOLD_TY_SIGN = 1.0

DEFAULT_ACTUATOR_TOPIC = "/cmd/actuator"
DEFAULT_ACTUATOR_MODE = 2
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
        self.min_confidence = float(
            rospy.get_param("~min_confidence", DEFAULT_MIN_CONFIDENCE)
        )
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
        self.model_message_index = 0

        self.mock_aruco_ids = self.parse_marker_sequence(
            rospy.get_param("~mock_aruco_ids", DEFAULT_MOCK_ARUCO_IDS)
        )
        self.light_seconds = float(
            rospy.get_param("~light_seconds", DEFAULT_LIGHT_SECONDS)
        )
        self.gap_seconds = float(rospy.get_param("~gap_seconds", DEFAULT_GAP_SECONDS))

        self.pose_cmd_topic = str(
            rospy.get_param("~pose_cmd_topic", DEFAULT_POSE_CMD_TOPIC)
        ).strip()
        self.status_topic = str(
            rospy.get_param("~status_topic", DEFAULT_STATUS_TOPIC)
        ).strip()
        self.status_timeout = float(
            rospy.get_param("~status_timeout", DEFAULT_STATUS_TIMEOUT)
        )
        self.status_wait_timeout = float(rospy.get_param(
            "~status_wait_timeout", DEFAULT_STATUS_WAIT_TIMEOUT
        ))
        self.status_linear_velocity_scale = float(rospy.get_param(
            "~status_linear_velocity_scale",
            DEFAULT_STATUS_LINEAR_VELOCITY_SCALE,
        ))
        self.hold_map_frame = str(
            rospy.get_param("~hold_map_frame", DEFAULT_HOLD_MAP_FRAME)
        ).strip()
        self.hold_base_frame = str(
            rospy.get_param("~hold_base_frame", DEFAULT_HOLD_BASE_FRAME)
        ).strip()
        self.initial_hover_seconds = float(rospy.get_param(
            "~initial_hover_seconds", DEFAULT_INITIAL_HOVER_SECONDS
        ))
        self.hold_pose_timeout = float(rospy.get_param(
            "~hold_pose_timeout", DEFAULT_HOLD_POSE_TIMEOUT
        ))
        self.hold_forward_position_gain = float(rospy.get_param(
            "~hold_forward_position_gain", DEFAULT_HOLD_FORWARD_POSITION_GAIN
        ))
        self.hold_lateral_position_gain = float(rospy.get_param(
            "~hold_lateral_position_gain", DEFAULT_HOLD_LATERAL_POSITION_GAIN
        ))
        self.hold_forward_velocity_damping = float(rospy.get_param(
            "~hold_forward_velocity_damping", DEFAULT_HOLD_FORWARD_VELOCITY_DAMPING
        ))
        self.hold_lateral_velocity_damping = float(rospy.get_param(
            "~hold_lateral_velocity_damping", DEFAULT_HOLD_LATERAL_VELOCITY_DAMPING
        ))
        self.hold_max_force = float(
            rospy.get_param("~hold_max_force", DEFAULT_HOLD_MAX_FORCE)
        )
        self.hold_position_tolerance = float(rospy.get_param(
            "~hold_position_tolerance", DEFAULT_HOLD_POSITION_TOLERANCE
        ))
        self.hold_speed_deadband = float(rospy.get_param(
            "~hold_speed_deadband", DEFAULT_HOLD_SPEED_DEADBAND
        ))
        self.hold_force_step = float(
            rospy.get_param("~hold_force_step", DEFAULT_HOLD_FORCE_STEP)
        )
        self.hold_tx_sign = float(
            rospy.get_param("~hold_tx_sign", DEFAULT_HOLD_TX_SIGN)
        )
        self.hold_ty_sign = float(
            rospy.get_param("~hold_ty_sign", DEFAULT_HOLD_TY_SIGN)
        )
        self.hold_translation = None
        self.hold_rotation = None
        self.hold_yaw = None
        self.current_status = None
        self.last_status_time = None
        self.last_hold_tx = 0
        self.last_hold_ty = 0
        self.accept_detections = False

        self.actuator_topic = rospy.get_param("~actuator_topic", DEFAULT_ACTUATOR_TOPIC)
        self.actuator_mode = int(
            rospy.get_param("~actuator_mode", DEFAULT_ACTUATOR_MODE)
        )
        # mode 字段由执行器协议新增，团队消息定义合并并重新编译后才会存在。
        self.actuator_mode_supported = hasattr(ActuatorControl(), "mode")

        self.light1 = int(rospy.get_param("~light1", DEFAULT_LIGHT1))
        self.light2 = int(rospy.get_param("~light2", DEFAULT_LIGHT2))

        self.heading_servo = int(rospy.get_param("~heading_servo", DEFAULT_HEADING_SERVO))
        self.clamp_servo = int(rospy.get_param("~clamp_servo", DEFAULT_CLAMP_SERVO))
        self.drive_cmd = int(rospy.get_param("~drive_cmd", DEFAULT_DRIVE_CMD))
        self.drive_speed = int(rospy.get_param("~drive_speed", DEFAULT_DRIVE_SPEED))

        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.pose_cmd_pub = rospy.Publisher(
            self.pose_cmd_topic, PoseNEDcmd, queue_size=10
        )
        self.finished_pub = rospy.Publisher("/finished", String, queue_size=10)
        self.tf_listener = tf.TransformListener()
        self.status_sub = rospy.Subscriber(
            self.status_topic,
            AUVData,
            self.status_callback,
            queue_size=20,
        )
        rospy.on_shutdown(self.on_shutdown)
        self.mock_index = 0

        if self.input_mode not in ("mock", "topic"):
            raise ValueError("input_mode 必须是 mock 或 topic")

        if self.input_mode == "mock" and not self.mock_aruco_ids:
            raise ValueError("mock_aruco_ids 不能为空")

        if self.stable_marker_count <= 0:
            raise ValueError("stable_marker_count 必须为正数")

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence 必须在 0 到 1 之间")
        if self.marker_timeout <= 0.0:
            raise ValueError("marker_timeout 必须大于 0")
        if self.input_mode == "topic" and not self.aruco_topic:
            raise ValueError("aruco_topic 不能为空")
        if not self.pose_cmd_topic:
            raise ValueError("pose_cmd_topic 不能为空")
        if not self.status_topic:
            raise ValueError("status_topic 不能为空")
        if not self.hold_map_frame or not self.hold_base_frame:
            raise ValueError("hold_map_frame 和 hold_base_frame 不能为空")
        if self.initial_hover_seconds < 0.0:
            raise ValueError("initial_hover_seconds 不能小于 0")
        if self.hold_pose_timeout <= 0.0:
            raise ValueError("hold_pose_timeout 必须大于 0")
        if min(
            self.status_timeout,
            self.status_wait_timeout,
            self.status_linear_velocity_scale,
            self.hold_max_force,
            self.hold_force_step,
        ) <= 0.0:
            raise ValueError("状态超时、速度缩放、最大定点力和推力步长必须大于 0")
        if min(
            self.hold_forward_position_gain,
            self.hold_lateral_position_gain,
            self.hold_forward_velocity_damping,
            self.hold_lateral_velocity_damping,
            self.hold_position_tolerance,
            self.hold_speed_deadband,
        ) < 0.0:
            raise ValueError("定点增益、阻尼和死区不能小于 0")
        if self.hold_tx_sign not in (-1.0, 1.0) or self.hold_ty_sign not in (-1.0, 1.0):
            raise ValueError("hold_tx_sign 和 hold_ty_sign 必须是 1.0 或 -1.0")
        if min(self.light_seconds, self.gap_seconds) < 0.0:
            raise ValueError("灯光持续时间不能小于 0")
        if self.actuator_mode not in (0, 1, 2):
            raise ValueError("actuator_mode 必须是 0、1 或 2")
        if self.actuator_mode != 2:
            rospy.logwarn(
                "%s：子任务2需要控制三色灯，actuator_mode 应设置为 2",
                NODE_NAME,
            )

        if self.input_mode == "topic":
            rospy.Subscriber(
                self.aruco_topic,
                TargetDetection,
                self.aruco_callback,
                queue_size=20,
            )

        rospy.loginfo(
            (
                "%s：启动子任务2，输入模式=%s，ArUco话题=%s，"
                "消息类型=TargetDetection，最低置信度=%.2f，稳定样本数=%d，"
                "样本超时=%.1fs，最多处理数量=%d"
            ),
            NODE_NAME,
            self.input_mode,
            self.aruco_topic,
            self.min_confidence,
            self.stable_marker_count,
            self.marker_timeout,
            self.max_topic_markers,
        )
        rospy.loginfo(
            "%s：执行器话题=%s，mode=%d（2=仅执行器）",
            NODE_NAME,
            self.actuator_topic,
            self.actuator_mode,
        )
        rospy.loginfo(
            (
                "%s：定点控制：话题=%s，mode=3，状态=%s，TF=%s -> %s，"
                "启动悬停=%.1fs，获取初始位姿超时=%.1fs"
            ),
            NODE_NAME,
            self.pose_cmd_topic,
            self.status_topic,
            self.hold_map_frame,
            self.hold_base_frame,
            self.initial_hover_seconds,
            self.hold_pose_timeout,
        )
        rospy.loginfo(
            (
                "%s：定点参数：位置增益=(前%.1f,右%.1f)，速度阻尼=(前%.1f,右%.1f)，"
                "最大力=%.1f，位置死区=%.3fm，速度死区=%.3fm/s，"
                "推力步长=%.1f，方向符号=(TX %.0f,TY %.0f)"
            ),
            NODE_NAME,
            self.hold_forward_position_gain,
            self.hold_lateral_position_gain,
            self.hold_forward_velocity_damping,
            self.hold_lateral_velocity_damping,
            self.hold_max_force,
            self.hold_position_tolerance,
            self.hold_speed_deadband,
            self.hold_force_step,
            self.hold_tx_sign,
            self.hold_ty_sign,
        )

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
        rospy.loginfo(
            "%s：mock 读取 ArUco 编号=%d，序号=%d/%d",
            NODE_NAME,
            marker_id,
            self.mock_index,
            len(self.mock_aruco_ids),
        )
        return marker_id

    @staticmethod
    def parse_marker_id_text(raw_value):
        text = str(raw_value).strip()
        if not text:
            return None

        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)

        patterns = (
            r"(?i)(?:aruco|marker)\s*(?:id)?\s*[:=_#-]?\s*(\d+)",
            r"(?i)\bid\s*[:=_#-]?\s*(\d+)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return None

    @classmethod
    def marker_id_from_detection(cls, message):
        for raw_value in (message.class_name, message.type):
            marker_id = cls.parse_marker_id_text(raw_value)
            if marker_id is not None:
                return marker_id
        return None

    def reset_marker_samples(self, reason, frame_index):
        previous_count = len(self.marker_samples)
        self.marker_samples = []
        if previous_count > 0:
            rospy.loginfo(
                "%s：[识别消息 #%d] 连续有效编号 %d -> 0，原因：%s",
                NODE_NAME,
                frame_index,
                previous_count,
                reason,
            )

    def aruco_callback(self, message):
        self.model_message_index += 1
        frame_index = self.model_message_index

        if not self.accept_detections:
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：[识别消息 #%d] 当前处于初始悬停或任务动作阶段，"
                    "本帧不计入连续识别"
                ),
                NODE_NAME,
                frame_index,
            )
            return

        confidence = float(message.conf)
        marker_id = self.marker_id_from_detection(message)

        detection_type = str(message.type).strip().lower()
        if detection_type == "aruco_not_detected" or marker_id == -1:
            self.reset_marker_samples("本帧未检测到 ArUco", frame_index)
            rospy.loginfo(
                (
                    "%s：[识别消息 #%d] 本帧无 ArUco："
                    "conf=%.3f，type=%r，class_name=%r"
                ),
                NODE_NAME,
                frame_index,
                confidence,
                message.type,
                message.class_name,
            )
            return

        if not math.isfinite(confidence) or confidence < self.min_confidence:
            self.reset_marker_samples("置信度不足", frame_index)
            rospy.loginfo(
                (
                    "%s：[识别消息 #%d] 本帧无效：置信度 %.3f < %.3f，"
                    "class_name=%r，type=%r"
                ),
                NODE_NAME,
                frame_index,
                confidence,
                self.min_confidence,
                message.class_name,
                message.type,
            )
            return

        if marker_id is None:
            self.reset_marker_samples("无法从消息文本解析 ArUco ID", frame_index)
            rospy.logwarn(
                (
                    "%s：[识别消息 #%d] 无法解析 ArUco ID："
                    "class_name=%r，type=%r"
                ),
                NODE_NAME,
                frame_index,
                message.class_name,
                message.type,
            )
            return

        if marker_id not in self.COLOR_BY_MARKER:
            self.reset_marker_samples("编号不在任务范围1~6", frame_index)
            rospy.logwarn(
                "%s：[识别消息 #%d] 忽略不支持的 ArUco ID=%d",
                NODE_NAME,
                frame_index,
                marker_id,
            )
            return

        now = rospy.Time.now()
        if self.marker_samples:
            sample_age = (now - self.marker_samples[-1][0]).to_sec()
            if sample_age > self.marker_timeout:
                self.reset_marker_samples("相邻识别消息间隔超时", frame_index)
            elif self.marker_samples[-1][1] != marker_id:
                self.reset_marker_samples("识别编号发生跳变", frame_index)

        self.marker_samples.append((now, marker_id))
        self.marker_samples = self.marker_samples[-self.stable_marker_count :]
        progress = len(self.marker_samples)
        position = message.pose.pose.position
        rospy.loginfo(
            (
                "%s：[识别消息 #%d] 第 %d/%d 帧有效：ArUco ID=%d，"
                "conf=%.3f，class_name=%r，type=%r，"
                "frame=%s，位置=(%.3f,%.3f,%.3f)m"
            ),
            NODE_NAME,
            frame_index,
            progress,
            self.stable_marker_count,
            marker_id,
            confidence,
            message.class_name,
            message.type,
            message.pose.header.frame_id or "未提供",
            position.x,
            position.y,
            position.z,
        )

    def topic_read_aruco_marker(self):
        now = rospy.Time.now()
        recent_samples = [
            sample
            for sample in self.marker_samples
            if (now - sample[0]).to_sec() <= self.marker_timeout
        ]
        self.marker_samples = recent_samples

        if len(recent_samples) < self.stable_marker_count:
            if recent_samples:
                rospy.loginfo_throttle(
                    1.0,
                    "%s：ArUco 稳定样本不足，进度=%d/%d",
                    NODE_NAME,
                    len(recent_samples),
                    self.stable_marker_count,
                )
            return None

        selected_ids = [
            sample[1] for sample in recent_samples[-self.stable_marker_count :]
        ]
        if len(set(selected_ids)) != 1:
            rospy.loginfo_throttle(
                1.0,
                "%s：ArUco 编号暂不稳定，最近样本=%s",
                NODE_NAME,
                selected_ids,
            )
            return None

        marker_id = selected_ids[-1]
        self.marker_samples = []
        rospy.loginfo("%s：稳定读取 ArUco 编号=%d", NODE_NAME, marker_id)
        return marker_id

    def read_aruco_marker(self):
        if self.input_mode == "mock":
            return self.mock_read_aruco_marker()
        return self.topic_read_aruco_marker()

    @classmethod
    def color_for_marker(cls, marker_id):
        return cls.COLOR_BY_MARKER.get(marker_id)

    def status_callback(self, message):
        values = (
            message.linear_velocity[0],
            message.linear_velocity[1],
            message.linear_velocity[2],
            message.pose.depth,
            message.pose.yaw,
        )
        if not all(math.isfinite(float(value)) for value in values):
            rospy.logwarn_throttle(
                2.0,
                "%s：/status/auv 包含无效速度或位姿，本帧已忽略",
                NODE_NAME,
            )
            return

        self.current_status = {
            "control_mode": int(message.control_mode),
            "vx": float(values[0]) * self.status_linear_velocity_scale,
            "vy": float(values[1]) * self.status_linear_velocity_scale,
            "vz": float(values[2]) * self.status_linear_velocity_scale,
            "depth": float(values[3]),
            "yaw_deg": float(values[4]),
        }
        self.last_status_time = rospy.Time.now()
        rospy.loginfo_throttle(
            1.0,
            (
                "%s：/status/auv：mode=%d，深度=%.3fm，航向=%.2fdeg，"
                "速度前右下=(%+.3f,%+.3f,%+.3f)m/s"
            ),
            NODE_NAME,
            self.current_status["control_mode"],
            self.current_status["depth"],
            self.current_status["yaw_deg"],
            self.current_status["vx"],
            self.current_status["vy"],
            self.current_status["vz"],
        )

    def get_recent_status(self):
        if self.current_status is None or self.last_status_time is None:
            rospy.logwarn_throttle(
                1.0,
                "%s：等待状态反馈 %s",
                NODE_NAME,
                self.status_topic,
            )
            return None

        age = (rospy.Time.now() - self.last_status_time).to_sec()
        if age > self.status_timeout:
            rospy.logwarn_throttle(
                1.0,
                "%s：状态反馈超时 %.2fs（限制 %.2fs），水平推力暂时清零",
                NODE_NAME,
                age,
                self.status_timeout,
            )
            return None
        return self.current_status

    def get_current_transform(self):
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                self.hold_map_frame,
                self.hold_base_frame,
                rospy.Time(0),
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                1.0,
                "%s：无法读取定点 TF %s -> %s：%s",
                NODE_NAME,
                self.hold_map_frame,
                self.hold_base_frame,
                str(error),
            )
            return None

        values = tuple(translation) + tuple(rotation)
        if not all(math.isfinite(float(value)) for value in values):
            rospy.logwarn_throttle(
                1.0,
                "%s：定点 TF 包含无效数值，本次反馈已忽略",
                NODE_NAME,
            )
            return None
        return (
            tuple(float(value) for value in translation),
            tuple(float(value) for value in rotation),
        )

    def capture_hold_pose(self):
        deadline = rospy.Time.now() + rospy.Duration(self.hold_pose_timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            try:
                self.tf_listener.waitForTransform(
                    self.hold_map_frame,
                    self.hold_base_frame,
                    rospy.Time(0),
                    rospy.Duration(0.5),
                )
                translation, rotation = self.tf_listener.lookupTransform(
                    self.hold_map_frame,
                    self.hold_base_frame,
                    rospy.Time(0),
                )
            except tf.Exception as error:
                rospy.logwarn_throttle(
                    1.0,
                    "%s：等待定点 TF %s -> %s：%s",
                    NODE_NAME,
                    self.hold_map_frame,
                    self.hold_base_frame,
                    str(error),
                )
                self.rate.sleep()
                continue

            values = tuple(translation) + tuple(rotation)
            if not all(math.isfinite(float(value)) for value in values):
                rospy.logwarn_throttle(
                    1.0,
                    "%s：定点 TF 包含无效数值，等待下一帧",
                    NODE_NAME,
                )
                self.rate.sleep()
                continue

            self.hold_translation = tuple(float(value) for value in translation)
            self.hold_rotation = tuple(float(value) for value in rotation)
            self.hold_yaw = euler_from_quaternion(self.hold_rotation)[2]
            self.hold_rotation = tuple(quaternion_from_euler(0.0, 0.0, self.hold_yaw))
            rospy.loginfo(
                (
                    "%s：已记录启动定点：%s 坐标=(%.3f,%.3f,%.3f)，"
                    "航向=%.1fdeg，后续固定使用 mode=3 和水平反馈保持该点"
                ),
                NODE_NAME,
                self.hold_map_frame,
                self.hold_translation[0],
                self.hold_translation[1],
                self.hold_translation[2],
                math.degrees(self.hold_yaw),
            )
            return True

        rospy.logerr(
            "%s：%.1fs 内未获得定点 TF %s -> %s",
            NODE_NAME,
            self.hold_pose_timeout,
            self.hold_map_frame,
            self.hold_base_frame,
        )
        return False

    @staticmethod
    def clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))

    def publish_hold_command(self, tx, ty, reason):
        if self.hold_translation is None or self.hold_rotation is None:
            return False

        command = PoseNEDcmd()
        command.mode = MODE_DEPTH_HEADING
        command.target.header.frame_id = self.hold_map_frame
        command.target.header.stamp = rospy.Time.now()
        command.target.pose.position.x = self.hold_translation[0]
        command.target.pose.position.y = self.hold_translation[1]
        command.target.pose.position.z = self.hold_translation[2]
        command.target.pose.orientation.x = self.hold_rotation[0]
        command.target.pose.orientation.y = self.hold_rotation[1]
        command.target.pose.orientation.z = self.hold_rotation[2]
        command.target.pose.orientation.w = self.hold_rotation[3]
        command.force.TX = int(round(self.clamp(tx, -10000.0, 10000.0)))
        command.force.TY = int(round(self.clamp(ty, -10000.0, 10000.0)))
        command.force.TZ = 0
        command.force.MX = 0
        command.force.MY = 0
        command.force.MZ = 0
        self.pose_cmd_pub.publish(command)

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：定点指令 mode=3，保持深度=%.3f，航向=%.1fdeg，"
                "TX=%d，TY=%d，TZ/MX/MY/MZ=0，阶段=%s"
            ),
            NODE_NAME,
            self.hold_translation[2],
            math.degrees(self.hold_yaw),
            command.force.TX,
            command.force.TY,
            reason,
        )
        return True

    def wait_for_status_feedback(self):
        deadline = rospy.Time.now() + rospy.Duration(self.status_wait_timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self.publish_hold_command(0.0, 0.0, "等待 /status/auv 反馈")
            status = self.get_recent_status()
            if status is not None:
                rospy.loginfo(
                    (
                        "%s：状态反馈已就绪：下位机模式=%d，"
                        "速度前右下=(%+.3f,%+.3f,%+.3f)m/s"
                    ),
                    NODE_NAME,
                    status["control_mode"],
                    status["vx"],
                    status["vy"],
                    status["vz"],
                )
                return True
            self.rate.sleep()

        rospy.logerr(
            "%s：%.1fs 内未收到有效状态反馈 %s",
            NODE_NAME,
            self.status_wait_timeout,
            self.status_topic,
        )
        return False

    def publish_position_hold(self, reason):
        status = self.get_recent_status()
        current = self.get_current_transform()
        if status is None or current is None:
            self.last_hold_tx = 0
            self.last_hold_ty = 0
            self.publish_hold_command(0.0, 0.0, "%s，反馈不可用" % reason)
            return False

        translation, _ = current
        dx = self.hold_translation[0] - translation[0]
        dy = self.hold_translation[1] - translation[1]
        forward_error = dx * math.cos(self.hold_yaw) + dy * math.sin(self.hold_yaw)
        right_error = -dx * math.sin(self.hold_yaw) + dy * math.cos(self.hold_yaw)

        base_tx = 0.0
        if abs(forward_error) > self.hold_position_tolerance:
            base_tx = self.hold_forward_position_gain * forward_error
        base_ty = 0.0
        if abs(right_error) > self.hold_position_tolerance:
            base_ty = self.hold_lateral_position_gain * right_error

        horizontal_speed = math.hypot(status["vx"], status["vy"])
        damping_vx = status["vx"] if horizontal_speed > self.hold_speed_deadband else 0.0
        damping_vy = status["vy"] if horizontal_speed > self.hold_speed_deadband else 0.0
        desired_tx = self.hold_tx_sign * self.clamp(
            base_tx - self.hold_forward_velocity_damping * damping_vx,
            -self.hold_max_force,
            self.hold_max_force,
        )
        desired_ty = self.hold_ty_sign * self.clamp(
            base_ty - self.hold_lateral_velocity_damping * damping_vy,
            -self.hold_max_force,
            self.hold_max_force,
        )

        tx = self.clamp(
            desired_tx,
            self.last_hold_tx - self.hold_force_step,
            self.last_hold_tx + self.hold_force_step,
        )
        ty = self.clamp(
            desired_ty,
            self.last_hold_ty - self.hold_force_step,
            self.last_hold_ty + self.hold_force_step,
        )
        self.last_hold_tx = int(round(tx))
        self.last_hold_ty = int(round(ty))
        if not self.publish_hold_command(self.last_hold_tx, self.last_hold_ty, reason):
            return False

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：水平定点反馈：位置误差=(前%+.3f,右%+.3f)m，"
                "速度=(前%+.3f,右%+.3f)m/s，指令=(TX=%d,TY=%d)，"
                "下位机模式=%d"
            ),
            NODE_NAME,
            forward_error,
            right_error,
            status["vx"],
            status["vy"],
            self.last_hold_tx,
            self.last_hold_ty,
            status["control_mode"],
        )
        return True

    def hold_position_for(self, seconds, reason):
        valid_elapsed = 0.0
        last_loop_time = rospy.Time.now()
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            loop_seconds = max(0.0, (now - last_loop_time).to_sec())
            last_loop_time = now
            if self.publish_position_hold(reason):
                valid_elapsed += loop_seconds
            if valid_elapsed >= seconds:
                return True
            rospy.loginfo_throttle(
                1.0,
                "%s：%s，剩余 %.1fs",
                NODE_NAME,
                reason,
                max(0.0, seconds - valid_elapsed),
            )
            self.rate.sleep()
        return False

    def publish_lights(self, color):
        red, yellow, green = self.ACTUATOR_LIGHTS[color]

        message = ActuatorControl()
        if not hasattr(message, "mode"):
            rospy.logerr_throttle(
                5.0,
                "%s：ActuatorControl 缺少 mode 字段，本次灯光指令未发送",
                NODE_NAME,
            )
            return False

        message.mode = self.actuator_mode
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
        return True

    def hold_color(self, color, seconds):
        rospy.loginfo("%s：灯光保持，颜色=%s，持续 %.1fs", NODE_NAME, color, seconds)
        red, yellow, green = self.ACTUATOR_LIGHTS[color]
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed >= seconds:
                return
            self.publish_lights(color)
            self.publish_position_hold("灯光动作期间保持启动定点")
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：灯光执行中：颜色=%s，三色指示=(红%d,黄%d,绿%d)，"
                    "剩余 %.1fs"
                ),
                NODE_NAME,
                color,
                red,
                yellow,
                green,
                max(0.0, seconds - elapsed),
            )
            self.rate.sleep()

    def finish_task(self):
        self.accept_detections = False
        self.publish_lights("off")
        self.last_hold_tx = 0
        self.last_hold_ty = 0
        self.publish_hold_command(0.0, 0.0, "任务结束，清零水平推力")
        self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
        rospy.loginfo(
            "%s：子任务2完成，已熄灯、清零 TX/TY 并发布 /finished",
            NODE_NAME,
        )

    def fail_task(self, reason):
        self.accept_detections = False
        self.publish_lights("off")
        self.last_hold_tx = 0
        self.last_hold_ty = 0
        self.publish_hold_command(0.0, 0.0, "任务失败，清零水平推力")
        self.finished_pub.publish(String(data="%s failed: %s" % (NODE_NAME, reason)))
        rospy.logerr("%s：子任务2失败：%s", NODE_NAME, reason)

    def on_shutdown(self):
        self.accept_detections = False
        if self.actuator_mode_supported:
            self.publish_lights("off")
        self.last_hold_tx = 0
        self.last_hold_ty = 0
        self.publish_hold_command(0.0, 0.0, "节点退出，清零水平推力")

    def run(self):
        rospy.sleep(0.5)

        rospy.loginfo(
            (
                "%s：人工定点测试流程：记录启动位置 -> mode=3定点悬停%.1fs -> "
                "连续%d帧识别 -> 对应颜色亮灯%.1fs -> 熄灯并清零TX/TY"
            ),
            NODE_NAME,
            self.initial_hover_seconds,
            self.stable_marker_count,
            self.light_seconds,
        )

        if not self.actuator_mode_supported:
            reason = "ActuatorControl 缺少 mode 字段，请同步消息定义并重新编译"
            self.fail_task(reason)
            rospy.signal_shutdown("%s failed" % NODE_NAME)
            return

        if not self.capture_hold_pose():
            reason = "无法记录启动定点，请检查 map -> base_link TF"
            self.fail_task(reason)
            rospy.signal_shutdown("%s failed" % NODE_NAME)
            return

        if not self.wait_for_status_feedback():
            reason = "无法获得 /status/auv 速度反馈，不能执行 mode=3 水平定点"
            self.fail_task(reason)
            rospy.signal_shutdown("%s failed" % NODE_NAME)
            return

        self.marker_samples = []
        rospy.loginfo(
            "%s：开始初始定点悬停 %.1fs，期间识别帧不参与稳定计数",
            NODE_NAME,
            self.initial_hover_seconds,
        )
        if not self.hold_position_for(
            self.initial_hover_seconds,
            "任务开始前定点悬停",
        ):
            if not rospy.is_shutdown():
                self.fail_task("初始定点悬停失败")
                rospy.signal_shutdown("%s failed" % NODE_NAME)
            return

        self.marker_samples = []
        self.accept_detections = True
        rospy.loginfo(
            "%s：初始悬停完成，现在开始正式累计 ArUco 连续识别帧",
            NODE_NAME,
        )

        if self.input_mode == "mock":
            rospy.loginfo(
                "%s：使用 mock ArUco 序列=%s",
                NODE_NAME,
                ",".join(str(marker_id) for marker_id in self.mock_aruco_ids),
            )
        else:
            rospy.loginfo(
                "%s：等待 ArUco 编号，话题=%s，稳定样本数=%d",
                NODE_NAME,
                self.aruco_topic,
                self.stable_marker_count,
            )

        handled_count = 0
        while not rospy.is_shutdown():
            if not self.publish_position_hold("等待 ArUco 稳定识别"):
                self.accept_detections = False
                self.reset_marker_samples(
                    "定点位置或速度反馈暂时不可用",
                    self.model_message_index,
                )
                self.rate.sleep()
                continue

            if not self.accept_detections:
                self.marker_samples = []
                self.accept_detections = True
                rospy.loginfo(
                    "%s：定点反馈恢复，重新开始累计 ArUco 连续识别帧",
                    NODE_NAME,
                )

            marker_id = self.read_aruco_marker()
            if marker_id is None:
                if self.input_mode == "mock":
                    break
                rospy.logwarn_throttle(
                    2.0,
                    "%s：仍在等待 ArUco 编号，话题=%s",
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
                    "%s：忽略不支持的 ArUco 编号=%d，期望范围为 1~6",
                    NODE_NAME,
                    marker_id,
                )
                self.rate.sleep()
                continue

            rospy.loginfo(
                "%s：识别到 ArUco 编号=%d，对应目标颜色=%s，开始亮灯",
                NODE_NAME,
                marker_id,
                color,
            )

            self.accept_detections = False
            self.hold_color(color, self.light_seconds)
            self.hold_color("off", self.gap_seconds)
            handled_count += 1

            if (
                self.input_mode == "topic"
                and self.max_topic_markers > 0
                and handled_count >= self.max_topic_markers
            ):
                rospy.loginfo(
                    "%s：已处理 %d 个 ArUco 编号，达到 max_topic_markers=%d",
                    NODE_NAME,
                    handled_count,
                    self.max_topic_markers,
                )
                break

            self.marker_samples = []
            self.accept_detections = True

        self.finish_task()
        rospy.signal_shutdown("%s complete" % NODE_NAME)


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3GetTaskTest().run()
    except rospy.ROSInterruptException:
        pass
