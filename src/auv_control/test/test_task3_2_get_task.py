#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务3子任务2：定点识别 ArUco 并点亮对应颜色灯。

人工把机器人停在目标正前方后，节点记录当前 ``map -> base_link`` 位姿，
固定使用 ``PoseNEDcmd.mode=4`` 做动力定位。启动后先悬停10秒，再开始
60秒识别计时。

识别采用最近10个模型消息组成的滑动窗口。任一合法 ArUco ID 在窗口内
出现3次即立即确认，不要求连续，也不需要等待窗口填满。例如第1、3、7帧
是同一个ID时，第7帧到达后立即成功。成功后点亮对应颜色灯3秒并结束；
60秒内未确认则按失败结束。
"""

import math
import re
import threading
from collections import Counter, deque

import rospy
import tf

from std_msgs.msg import String

from auv_control.msg import ActuatorControl, PoseNEDcmd, TargetDetection


NODE_NAME = "test_task3_2_get_task"
MODE_DYNAMIC_POSITIONING = 4

DEFAULT_RATE = 10.0
DEFAULT_POSE_CMD_TOPIC = "/cmd/pose/ned"
DEFAULT_HOLD_MAP_FRAME = "map"
DEFAULT_HOLD_BASE_FRAME = "base_link"
DEFAULT_INITIAL_HOVER_SECONDS = 10.0
DEFAULT_HOLD_POSE_TIMEOUT = 5.0

DEFAULT_INPUT_MODE = "topic"
DEFAULT_ARUCO_TOPIC = "/obj/target_message"
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_RECOGNITION_WINDOW_SIZE = 10
DEFAULT_REQUIRED_MATCH_COUNT = 3
DEFAULT_RECOGNITION_TIMEOUT = 60.0
DEFAULT_MOCK_FRAME_INTERVAL = 0.2
# 模拟第1、3、7帧为同一个ID，-1表示空帧。
DEFAULT_MOCK_ARUCO_IDS = [1, -1, 1, 2, -1, -1, 1]

DEFAULT_LIGHT_SECONDS = 3.0
DEFAULT_GAP_SECONDS = 0.5
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
        self.rate_hz = float(rospy.get_param("~rate", DEFAULT_RATE))
        self.rate = rospy.Rate(self.rate_hz)

        self.pose_cmd_topic = str(
            rospy.get_param("~pose_cmd_topic", DEFAULT_POSE_CMD_TOPIC)
        ).strip()
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

        self.input_mode = str(
            rospy.get_param("~input_mode", DEFAULT_INPUT_MODE)
        ).strip().lower()
        self.aruco_topic = str(
            rospy.get_param("~aruco_topic", DEFAULT_ARUCO_TOPIC)
        ).strip()
        self.min_confidence = float(
            rospy.get_param("~min_confidence", DEFAULT_MIN_CONFIDENCE)
        )
        self.recognition_window_size = int(rospy.get_param(
            "~recognition_window_size", DEFAULT_RECOGNITION_WINDOW_SIZE
        ))
        self.required_match_count = int(rospy.get_param(
            "~required_match_count", DEFAULT_REQUIRED_MATCH_COUNT
        ))
        self.recognition_timeout = float(rospy.get_param(
            "~recognition_timeout", DEFAULT_RECOGNITION_TIMEOUT
        ))
        self.mock_frame_interval = float(rospy.get_param(
            "~mock_frame_interval", DEFAULT_MOCK_FRAME_INTERVAL
        ))
        self.mock_aruco_ids = self.parse_marker_sequence(
            rospy.get_param("~mock_aruco_ids", DEFAULT_MOCK_ARUCO_IDS)
        )

        self.light_seconds = float(
            rospy.get_param("~light_seconds", DEFAULT_LIGHT_SECONDS)
        )
        self.gap_seconds = float(
            rospy.get_param("~gap_seconds", DEFAULT_GAP_SECONDS)
        )
        self.actuator_topic = str(
            rospy.get_param("~actuator_topic", DEFAULT_ACTUATOR_TOPIC)
        ).strip()
        self.actuator_mode = int(
            rospy.get_param("~actuator_mode", DEFAULT_ACTUATOR_MODE)
        )
        self.light1 = int(rospy.get_param("~light1", DEFAULT_LIGHT1))
        self.light2 = int(rospy.get_param("~light2", DEFAULT_LIGHT2))
        self.heading_servo = int(
            rospy.get_param("~heading_servo", DEFAULT_HEADING_SERVO)
        )
        self.clamp_servo = int(
            rospy.get_param("~clamp_servo", DEFAULT_CLAMP_SERVO)
        )
        self.drive_cmd = int(
            rospy.get_param("~drive_cmd", DEFAULT_DRIVE_CMD)
        )
        self.drive_speed = int(
            rospy.get_param("~drive_speed", DEFAULT_DRIVE_SPEED)
        )

        self.validate_params()

        self.hold_translation = None
        self.hold_rotation = None
        self.accept_detections = False
        self.model_message_index = 0
        self.recognition_frame_index = 0
        self.mock_index = 0
        self.confirmed_marker_id = None
        self.recognition_lock = threading.Lock()
        self.recognition_window = deque(maxlen=self.recognition_window_size)
        self.outputs_closed = False

        self.pose_cmd_pub = rospy.Publisher(
            self.pose_cmd_topic, PoseNEDcmd, queue_size=10
        )
        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.finished_pub = rospy.Publisher(
            "/finished", String, queue_size=10
        )
        self.tf_listener = tf.TransformListener()
        self.actuator_mode_supported = hasattr(ActuatorControl(), "mode")

        self.aruco_sub = None
        if self.input_mode == "topic":
            self.aruco_sub = rospy.Subscriber(
                self.aruco_topic,
                TargetDetection,
                self.aruco_callback,
                queue_size=20,
            )

        rospy.on_shutdown(self.on_shutdown)
        self.log_startup_config()

    def validate_params(self):
        if self.rate_hz <= 0.0:
            raise ValueError("rate 必须大于 0")
        if self.input_mode not in ("topic", "mock"):
            raise ValueError("input_mode 必须是 topic 或 mock")
        if self.input_mode == "topic" and not self.aruco_topic:
            raise ValueError("aruco_topic 不能为空")
        if self.input_mode == "mock" and not self.mock_aruco_ids:
            raise ValueError("mock_aruco_ids 不能为空")
        if not self.pose_cmd_topic:
            raise ValueError("pose_cmd_topic 不能为空")
        if not self.hold_map_frame or not self.hold_base_frame:
            raise ValueError("hold_map_frame 和 hold_base_frame 不能为空")
        if self.initial_hover_seconds < 0.0:
            raise ValueError("initial_hover_seconds 不能小于 0")
        if self.hold_pose_timeout <= 0.0:
            raise ValueError("hold_pose_timeout 必须大于 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence 必须在 0 到 1 之间")
        if self.recognition_window_size <= 0:
            raise ValueError("recognition_window_size 必须为正数")
        if self.required_match_count <= 0:
            raise ValueError("required_match_count 必须为正数")
        if self.required_match_count > self.recognition_window_size:
            raise ValueError("required_match_count 不能大于 recognition_window_size")
        if self.recognition_timeout <= 0.0:
            raise ValueError("recognition_timeout 必须大于 0")
        if self.mock_frame_interval <= 0.0:
            raise ValueError("mock_frame_interval 必须大于 0")
        if min(self.light_seconds, self.gap_seconds) < 0.0:
            raise ValueError("灯光持续时间不能小于 0")
        if self.actuator_mode not in (0, 1, 2):
            raise ValueError("actuator_mode 必须是 0、1 或 2")
        if self.actuator_mode != 2:
            rospy.logwarn(
                "%s：三色指示灯属于执行器，actuator_mode 应设置为 2",
                NODE_NAME,
            )

    def log_startup_config(self):
        rospy.loginfo(
            (
                "%s：流程=mode4定点悬停%.1fs -> 最近%d帧内同ID达到%d帧 -> "
                "亮灯%.1fs；识别最长%.1fs"
            ),
            NODE_NAME,
            self.initial_hover_seconds,
            self.recognition_window_size,
            self.required_match_count,
            self.light_seconds,
            self.recognition_timeout,
        )
        rospy.loginfo(
            "%s：定点话题=%s，TF=%s -> %s；识别模式=%s，话题=%s",
            NODE_NAME,
            self.pose_cmd_topic,
            self.hold_map_frame,
            self.hold_base_frame,
            self.input_mode,
            self.aruco_topic,
        )
        rospy.loginfo(
            "%s：执行器话题=%s，actuator_mode=%d，最低置信度=%.2f",
            NODE_NAME,
            self.actuator_topic,
            self.actuator_mode,
            self.min_confidence,
        )

    @staticmethod
    def parse_marker_sequence(raw_value):
        if isinstance(raw_value, (list, tuple)):
            return [int(value) for value in raw_value]
        return [int(value) for value in re.findall(r"[+-]?\d+", str(raw_value))]

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

    @classmethod
    def color_for_marker(cls, marker_id):
        return cls.COLOR_BY_MARKER.get(marker_id)

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
            rospy.loginfo(
                "%s：已记录 mode=4 定点：%s 坐标=(%.3f,%.3f,%.3f)",
                NODE_NAME,
                self.hold_map_frame,
                self.hold_translation[0],
                self.hold_translation[1],
                self.hold_translation[2],
            )
            return True

        rospy.logerr(
            "%s：%.1fs 内未获得 TF %s -> %s",
            NODE_NAME,
            self.hold_pose_timeout,
            self.hold_map_frame,
            self.hold_base_frame,
        )
        return False

    def publish_position_hold(self, reason):
        if self.hold_translation is None or self.hold_rotation is None:
            return False

        command = PoseNEDcmd()
        command.mode = MODE_DYNAMIC_POSITIONING
        command.target.header.frame_id = self.hold_map_frame
        command.target.header.stamp = rospy.Time.now()
        command.target.pose.position.x = self.hold_translation[0]
        command.target.pose.position.y = self.hold_translation[1]
        command.target.pose.position.z = self.hold_translation[2]
        command.target.pose.orientation.x = self.hold_rotation[0]
        command.target.pose.orientation.y = self.hold_rotation[1]
        command.target.pose.orientation.z = self.hold_rotation[2]
        command.target.pose.orientation.w = self.hold_rotation[3]
        command.force.TX = 0
        command.force.TY = 0
        command.force.TZ = 0
        command.force.MX = 0
        command.force.MY = 0
        command.force.MZ = 0
        self.pose_cmd_pub.publish(command)

        rospy.loginfo_throttle(
            1.0,
            (
                "%s：定点指令 mode=4，目标=(%.3f,%.3f,%.3f)，"
                "附加力全为0，阶段=%s"
            ),
            NODE_NAME,
            self.hold_translation[0],
            self.hold_translation[1],
            self.hold_translation[2],
            reason,
        )
        return True

    def hold_position_for(self, seconds, reason):
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed >= seconds:
                return True
            if not self.publish_position_hold(reason):
                return False
            rospy.loginfo_throttle(
                1.0,
                "%s：%s，剩余 %.1fs",
                NODE_NAME,
                reason,
                max(0.0, seconds - elapsed),
            )
            self.rate.sleep()
        return False

    @staticmethod
    def window_text(window_values):
        return "[{}]".format(
            ",".join("-" if value is None else str(value) for value in window_values)
        )

    @staticmethod
    def count_text(counts):
        if not counts:
            return "无有效ID"
        return ",".join(
            "ID{}:{}帧".format(marker_id, counts[marker_id])
            for marker_id in sorted(counts)
        )

    def record_recognition_frame(self, marker_id, frame_index, detail):
        with self.recognition_lock:
            if self.confirmed_marker_id is not None:
                return

            self.recognition_window.append(marker_id)
            window_values = list(self.recognition_window)
            counts = Counter(
                value for value in window_values if value is not None
            )
            winners = [
                value
                for value, count in counts.items()
                if count >= self.required_match_count
            ]
            if winners:
                # 优先选择出现次数更多的ID；次数相同则选择数值较小的ID。
                self.confirmed_marker_id = min(
                    winners,
                    key=lambda value: (-counts[value], value),
                )

            confirmed_marker_id = self.confirmed_marker_id
            window_snapshot = self.window_text(window_values)
            count_snapshot = self.count_text(counts)
            window_frame_count = len(window_values)
            current_count = counts.get(marker_id, 0) if marker_id is not None else 0
            best_marker_id = None
            best_count = 0
            if counts:
                best_marker_id = min(
                    counts,
                    key=lambda value: (-counts[value], value),
                )
                best_count = counts[best_marker_id]
            confirmed_count = (
                counts.get(confirmed_marker_id, 0)
                if confirmed_marker_id is not None
                else 0
            )

        if marker_id is None:
            if best_marker_id is None:
                rospy.loginfo(
                    (
                        "%s：[识别第%d帧] 本帧未识别到有效ID，原因=%s；"
                        "最近%d帧中识别到0帧（0/%d）；队列=%s"
                    ),
                    NODE_NAME,
                    frame_index,
                    detail,
                    window_frame_count,
                    window_frame_count,
                    window_snapshot,
                )
            else:
                rospy.loginfo(
                    (
                        "%s：[识别第%d帧] 本帧未识别到有效ID，原因=%s；"
                        "当前最多的是ID=%d，最近%d帧中识别到%d帧（%d/%d），"
                        "确认进度=%d/%d；队列=%s，统计=%s"
                    ),
                    NODE_NAME,
                    frame_index,
                    detail,
                    best_marker_id,
                    window_frame_count,
                    best_count,
                    best_count,
                    window_frame_count,
                    best_count,
                    self.required_match_count,
                    window_snapshot,
                    count_snapshot,
                )
        else:
            rospy.loginfo(
                (
                    "%s：[识别第%d帧] 本帧识别到有效ID=%d；"
                    "最近%d帧中该ID识别到%d帧（%d/%d），确认进度=%d/%d；"
                    "队列=%s，%s"
                ),
                NODE_NAME,
                frame_index,
                marker_id,
                window_frame_count,
                current_count,
                current_count,
                window_frame_count,
                current_count,
                self.required_match_count,
                window_snapshot,
                detail,
            )

        if confirmed_marker_id is not None:
            rospy.loginfo(
                (
                    "%s：识别成功：ArUco ID=%d，最近%d帧中识别到%d帧"
                    "（%d/%d），已达到确认要求%d帧，不再等待窗口填满"
                ),
                NODE_NAME,
                confirmed_marker_id,
                window_frame_count,
                confirmed_count,
                confirmed_count,
                window_frame_count,
                self.required_match_count,
            )

    def aruco_callback(self, message):
        self.model_message_index += 1

        if not self.accept_detections:
            rospy.loginfo_throttle(
                1.0,
                "%s：[模型消息 #%d] 当前尚未进入60秒识别阶段，本帧不入队",
                NODE_NAME,
                self.model_message_index,
            )
            return

        with self.recognition_lock:
            self.recognition_frame_index += 1
            frame_index = self.recognition_frame_index

        try:
            confidence = float(message.conf)
        except (TypeError, ValueError):
            confidence = float("nan")
        detection_type = str(message.type).strip().lower()
        marker_id = self.marker_id_from_detection(message)
        detail = "conf={:.3f}, type={!r}, class_name={!r}".format(
            confidence,
            message.type,
            message.class_name,
        )

        if detection_type == "aruco_not_detected" or marker_id == -1:
            self.record_recognition_frame(
                None, frame_index, "未检测到ArUco；{}".format(detail)
            )
            return
        if not math.isfinite(confidence) or confidence < self.min_confidence:
            self.record_recognition_frame(
                None, frame_index, "置信度不足；{}".format(detail)
            )
            return
        if marker_id is None:
            self.record_recognition_frame(
                None, frame_index, "无法解析ID；{}".format(detail)
            )
            return
        if marker_id not in self.COLOR_BY_MARKER:
            self.record_recognition_frame(
                None,
                frame_index,
                "ID={}不在任务范围1~6；{}".format(marker_id, detail),
            )
            return

        self.record_recognition_frame(marker_id, frame_index, detail)

    def reset_recognition(self):
        with self.recognition_lock:
            self.recognition_window.clear()
            self.confirmed_marker_id = None
            self.recognition_frame_index = 0
        self.mock_index = 0

    def get_confirmed_marker_id(self):
        with self.recognition_lock:
            return self.confirmed_marker_id

    def feed_next_mock_frame(self):
        if self.mock_index >= len(self.mock_aruco_ids):
            return False

        raw_marker_id = int(self.mock_aruco_ids[self.mock_index])
        self.mock_index += 1
        self.model_message_index += 1
        with self.recognition_lock:
            self.recognition_frame_index += 1
            frame_index = self.recognition_frame_index
        marker_id = raw_marker_id if raw_marker_id in self.COLOR_BY_MARKER else None
        detail = "mock序号={}/{}，原始值={}".format(
            self.mock_index,
            len(self.mock_aruco_ids),
            raw_marker_id,
        )
        self.record_recognition_frame(
            marker_id,
            frame_index,
            detail,
        )
        return True

    def wait_for_recognition(self):
        self.reset_recognition()
        self.accept_detections = self.input_mode == "topic"
        start_time = rospy.Time.now()
        next_mock_time = start_time

        rospy.loginfo(
            (
                "%s：正式开始识别，计时%.1fs；窗口大小=%d，"
                "同一ID命中%d帧立即成功"
            ),
            NODE_NAME,
            self.recognition_timeout,
            self.recognition_window_size,
            self.required_match_count,
        )

        while not rospy.is_shutdown():
            self.publish_position_hold("60秒ArUco识别阶段")
            now = rospy.Time.now()

            if self.input_mode == "mock" and now >= next_mock_time:
                self.feed_next_mock_frame()
                next_mock_time = now + rospy.Duration(self.mock_frame_interval)

            marker_id = self.get_confirmed_marker_id()
            if marker_id is not None:
                self.accept_detections = False
                return marker_id

            elapsed = (now - start_time).to_sec()
            if elapsed >= self.recognition_timeout:
                self.accept_detections = False
                return None

            with self.recognition_lock:
                window_snapshot = self.window_text(list(self.recognition_window))
            rospy.loginfo_throttle(
                1.0,
                "%s：识别进行中，剩余 %.1fs，当前窗口=%s",
                NODE_NAME,
                max(0.0, self.recognition_timeout - elapsed),
                window_snapshot,
            )
            self.rate.sleep()

        return None

    def publish_lights(self, color):
        red, yellow, green = self.ACTUATOR_LIGHTS[color]
        message = ActuatorControl()
        if not hasattr(message, "mode"):
            rospy.logerr_throttle(
                5.0,
                "%s：ActuatorControl 缺少 mode 字段，灯光指令未发送",
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
        red, yellow, green = self.ACTUATOR_LIGHTS[color]
        start_time = rospy.Time.now()
        rospy.loginfo(
            "%s：开始灯光阶段：颜色=%s，持续%.1fs",
            NODE_NAME,
            color,
            seconds,
        )
        while not rospy.is_shutdown():
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed >= seconds:
                return True
            self.publish_position_hold("灯光阶段继续mode4定点")
            self.publish_lights(color)
            rospy.loginfo_throttle(
                1.0,
                (
                    "%s：灯光执行中：颜色=%s，三色指示=(红%d,黄%d,绿%d)，"
                    "剩余%.1fs"
                ),
                NODE_NAME,
                color,
                red,
                yellow,
                green,
                max(0.0, seconds - elapsed),
            )
            self.rate.sleep()
        return False

    def finalize_task(self, success, detail):
        self.accept_detections = False
        self.publish_lights("off")
        self.publish_position_hold("任务结束前保持mode4定点，附加力清零")
        state = "finished" if success else "failed"
        self.finished_pub.publish(String(
            data="{} {}: {}".format(NODE_NAME, state, detail)
        ))
        self.outputs_closed = True
        rospy.loginfo(
            "%s：任务%s：%s；灯光已关闭，节点即将退出",
            NODE_NAME,
            "成功" if success else "失败",
            detail,
        )

    def on_shutdown(self):
        self.accept_detections = False
        if self.outputs_closed:
            return
        if hasattr(self, "actuator_pub") and self.actuator_mode_supported:
            self.publish_lights("off")
        if hasattr(self, "pose_cmd_pub"):
            self.publish_position_hold("节点退出，保持mode4定点")
        self.outputs_closed = True

    def run(self):
        rospy.sleep(0.5)
        if not self.actuator_mode_supported:
            reason = "ActuatorControl 缺少 mode 字段，请重新编译最新消息定义"
            self.finalize_task(False, reason)
            rospy.signal_shutdown(reason)
            return

        if not self.capture_hold_pose():
            reason = "无法记录启动定点，请检查 map -> base_link TF"
            self.finalize_task(False, reason)
            rospy.signal_shutdown(reason)
            return

        rospy.loginfo(
            "%s：先进行mode4定点悬停%.1fs，期间模型消息不入识别窗口",
            NODE_NAME,
            self.initial_hover_seconds,
        )
        if not self.hold_position_for(
            self.initial_hover_seconds,
            "识别前定点悬停",
        ):
            return

        marker_id = self.wait_for_recognition()
        if marker_id is None:
            reason = "识别时间超过{:.1f}s，未满足最近{}帧内同ID达到{}帧".format(
                self.recognition_timeout,
                self.recognition_window_size,
                self.required_match_count,
            )
            self.finalize_task(False, reason)
            rospy.signal_shutdown(reason)
            return

        color = self.color_for_marker(marker_id)
        rospy.loginfo(
            "%s：ArUco ID=%d 确认成功，对应颜色=%s，开始亮灯",
            NODE_NAME,
            marker_id,
            color,
        )
        self.hold_color(color, self.light_seconds)
        self.hold_color("off", self.gap_seconds)

        detail = "ArUco ID={}，颜色={}，亮灯{:.1f}s".format(
            marker_id,
            color,
            self.light_seconds,
        )
        self.finalize_task(True, detail)
        rospy.signal_shutdown("{} complete".format(NODE_NAME))


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3GetTaskTest().run()
    except rospy.ROSInterruptException:
        pass
