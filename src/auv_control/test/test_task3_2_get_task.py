#! /home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
任务 3 子任务 2 测试：读取 ArUco 标记编号并点亮对应颜色灯。

本脚本不控制机器人移动，也不发布 /cmd/pose/ned 或 /target。它只读取 ArUco
编号，并驱动新的 /cmd/actuator 灯光字段。

默认模式从 /task3/aruco_id 读取真实 ArUco 编号。mock 模式仍保留，用于没有
相机/感知节点时的台架测试。

任务规则映射：
  1,2 -> yellow
  3,4 -> green
  5,6 -> red

本脚本只使用较新的 /cmd/actuator 话题。

记录：
2026.7.13
  执行器下行话题调整为 /cmd/actuator。
"""

import rospy

from std_msgs.msg import Int32, String

from auv_control.msg import ActuatorControl


NODE_NAME = "test_task3_2_get_task"

# =========================
# 可调默认参数
# =========================
# 这些值是水池/实艇调试时优先调整的位置。
# 它们仍然是 ROS 参数，因此 roslaunch 可以在不改代码的情况下覆盖。

DEFAULT_RATE = 10.0
DEFAULT_INPUT_MODE = "topic"  # 可选 topic 或 mock
DEFAULT_ARUCO_TOPIC = "/task3/aruco_id"
DEFAULT_MOCK_ARUCO_IDS = [1, 3, 5, 2, 4, 6]

# max_topic_markers=1 表示本测试读到一个真实标记即可结束。
# 如果希望脚本持续响应新的标记，可以设置为 0。
DEFAULT_MAX_TOPIC_MARKERS = 1
DEFAULT_STABLE_MARKER_COUNT = 1
DEFAULT_MARKER_TIMEOUT = 1.0

DEFAULT_LIGHT_SECONDS = 3.0
DEFAULT_GAP_SECONDS = 0.5

DEFAULT_ACTUATOR_TOPIC = "/cmd/actuator"
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

        self.mock_aruco_ids = self.parse_marker_sequence(
            rospy.get_param("~mock_aruco_ids", DEFAULT_MOCK_ARUCO_IDS)
        )
        self.light_seconds = float(
            rospy.get_param("~light_seconds", DEFAULT_LIGHT_SECONDS)
        )
        self.gap_seconds = float(rospy.get_param("~gap_seconds", DEFAULT_GAP_SECONDS))

        self.actuator_topic = rospy.get_param("~actuator_topic", DEFAULT_ACTUATOR_TOPIC)

        self.light1 = int(rospy.get_param("~light1", DEFAULT_LIGHT1))
        self.light2 = int(rospy.get_param("~light2", DEFAULT_LIGHT2))

        self.heading_servo = int(rospy.get_param("~heading_servo", DEFAULT_HEADING_SERVO))
        self.clamp_servo = int(rospy.get_param("~clamp_servo", DEFAULT_CLAMP_SERVO))
        self.drive_cmd = int(rospy.get_param("~drive_cmd", DEFAULT_DRIVE_CMD))
        self.drive_speed = int(rospy.get_param("~drive_speed", DEFAULT_DRIVE_SPEED))

        self.actuator_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.finished_pub = rospy.Publisher("/finished", String, queue_size=10)
        self.mock_index = 0

        if self.input_mode not in ("mock", "topic"):
            raise ValueError("input_mode 必须是 mock 或 topic")

        if self.input_mode == "mock" and not self.mock_aruco_ids:
            raise ValueError("mock_aruco_ids 不能为空")

        if self.stable_marker_count <= 0:
            raise ValueError("stable_marker_count 必须为正数")

        if self.input_mode == "topic":
            rospy.Subscriber(self.aruco_topic, Int32, self.aruco_callback)

        rospy.loginfo(
            (
                "%s：启动子任务2，输入模式=%s，ArUco话题=%s，稳定样本数=%d，"
                "样本超时=%.1fs，最多处理数量=%d"
            ),
            NODE_NAME,
            self.input_mode,
            self.aruco_topic,
            self.stable_marker_count,
            self.marker_timeout,
            self.max_topic_markers,
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

    def aruco_callback(self, message):
        marker_id = int(message.data)
        now = rospy.Time.now()
        self.marker_samples.append((now, marker_id))
        rospy.loginfo_throttle(
            0.8,
            "%s：收到 ArUco 编号样本=%d，当前缓存样本数=%d",
            NODE_NAME,
            marker_id,
            len(self.marker_samples),
        )

        max_samples = max(self.stable_marker_count * 3, 10)
        if len(self.marker_samples) > max_samples:
            self.marker_samples = self.marker_samples[-max_samples:]

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
        rospy.loginfo("%s：灯光保持，颜色=%s，持续 %.1fs", NODE_NAME, color, seconds)
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
        rospy.loginfo("%s：子任务2完成，已熄灯并发布 /finished", NODE_NAME)

    def run(self):
        rospy.sleep(0.5)

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

        self.finish_task()
        rospy.signal_shutdown("%s complete" % NODE_NAME)


if __name__ == "__main__":
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3GetTaskTest().run()
    except rospy.ROSInterruptException:
        pass
