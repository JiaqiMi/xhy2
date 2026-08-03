#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-
"""
名称：state_web_node.py
功能：订阅 AUV 图像、位姿、速度和控制状态，并提供只读 Web 仪表盘
作者：xhy
监听：AUV 状态、TF、三路相机图像及视觉检测结果
发布：HTTP 状态接口与三路 MJPEG 相机流
记录：
2026.7.17
    新增三路原始图像、NED 位置、航向图、人工地平图和运行状态监控。
2026.7.30
    按检测图像分辨率缩放视觉坐标，修正鱼眼 ArUco 框位置。
    在最终视频尺寸上绘制标注，并增大字体、字宽和标签间距。
2026.8.2
    统计各视觉结果话题的独立帧率并加入状态接口。
    缓存最后一次有效视觉结果，标注失效后延迟一秒消失。
2026.8.3
    新增视觉目标 map 历史、时间戳 TF 重试、ArUco 历史和清除接口。
    新增 base_link 一分钟轨迹、轨迹清除接口和最近两帧目标位姿。
    三路实时 TF 统一使用 Time(0) 获取，公共时间查询失败不再影响位姿。
    增加 map 到 hand 的最新 TF，并通过状态接口提供地图点位。
    目标历史按 0.01m 位置阈值或 1° 航向阈值过滤重复命令。
    将相机画面中的 Arrow 识别标注改为青蓝色。
    base_link 轨迹默认按 1Hz 保留最多 1000 个点。
    实际箭头终点改用 camera_center TF，地图文字仍显示 camera。
"""

import copy
import json
import logging
import math
import os
import threading
import time

import cv2
import numpy as np
import rospkg
import rospy
import tf
from auv_control.msg import (
    AUVData,
    ActuatorControl,
    MotionState,
    PoseNEDcmd,
    SensorStatus,
)
from cv_bridge import CvBridge, CvBridgeError
from flask import Flask, Response, abort, jsonify, send_from_directory
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image, NavSatFix
from std_msgs.msg import String

from state_web_core import (
    ACTUATOR_MODE_NAMES,
    CONTROL_MODE_NAMES,
    MOTION_STATE_NAMES,
    NavigationHistoryState,
    OriginRevision,
    VisionHistoryState,
    extract_vision_pose_points,
    has_vision_detections,
    health_state,
    map_pixel_to_frame,
    normalize_heading,
    quaternion_to_euler_deg,
    safe_float,
    sanitize_json,
    select_attitude,
    shortest_heading_error,
    transform_visual_geometry,
    update_fps,
    vision_map_category,
    vision_packet_status,
)


def ros_stamp_sec(header):
    """提取 ROS Header 时间戳，零时间返回 None。"""
    if header is None:
        return None
    try:
        stamp = header.stamp
        if stamp is None or stamp == rospy.Time(0):
            return None
        return stamp.to_sec()
    except (AttributeError, TypeError):
        return None


def serialize_motor(message):
    """序列化六自由度力/力矩。"""
    return {
        "tx": int(message.TX),
        "ty": int(message.TY),
        "tz": int(message.TZ),
        "mx": int(message.MX),
        "my": int(message.MY),
        "mz": int(message.MZ),
    }


def serialize_auv_pose(message):
    """序列化经纬深和姿态。"""
    return {
        "latitude_deg": float(message.latitude),
        "longitude_deg": float(message.longitude),
        "altitude_m": float(message.altitude),
        "depth_m": float(message.depth),
        "roll_deg": float(message.roll),
        "pitch_deg": float(message.pitch),
        "yaw_deg": float(message.yaw),
        "heading_deg": normalize_heading(message.yaw),
        "speed_mps": float(message.speed),
    }


def serialize_pose_stamped(message):
    """序列化 PoseStamped，并附加欧拉角。"""
    pose = message.pose
    orientation = quaternion_to_euler_deg(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    return {
        "frame_id": message.header.frame_id,
        "stamp_sec": ros_stamp_sec(message.header),
        "position_m": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation_quaternion": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
        "orientation_deg": orientation,
    }


def serialize_actuator(message):
    """序列化补光灯和执行器状态。"""
    mode = int(message.mode)
    return {
        "mode": mode,
        "mode_name": ACTUATOR_MODE_NAMES.get(mode, "未知"),
        "light1": int(message.light1),
        "light2": int(message.light2),
        "heading_servo": int(message.heading_servo),
        "clamp_servo": int(message.clamp_servo),
        "drive_cmd": int(message.drive_cmd),
        "drive_speed": int(message.drive_speed),
        "red_light": int(message.red_light),
        "yellow_light": int(message.yellow_light),
        "green_light": int(message.green_light),
    }


VISION_SOURCE_DEFAULTS = {
    "line": {
        "camera": "left",
        "label": "Line",
        "color": (0, 214, 255),
        "topics": {
            "detection": "/vision/line/detections",
            "pose": "/vision/line/pose",
        },
    },
    "red_circle": {
        "camera": "left",
        "label": "RedCircle",
        "color": (55, 80, 255),
        "topics": {
            "detection": "/vision/red_circle/detections",
            "pose": "/vision/red_circle/pose",
        },
    },
    "shapes": {
        "camera": "left",
        "label": "Shapes",
        "color": (86, 220, 113),
        "topics": {
            "detection": "/vision/shapes/detections",
            "pose": "/vision/shapes/pose",
        },
    },
    "rectangle": {
        "camera": "left",
        "label": "Rectangle",
        "color": (255, 163, 62),
        "topics": {
            "detection": "/vision/rectangle/detections",
            "pose": "/vision/rectangle/pose",
        },
    },
    "arrow": {
        "camera": "left",
        "label": "Arrow",
        "color": (238, 211, 34),
        "topics": {
            "detection": "/vision/arrow/detections",
            "arrow": "/vision/arrow/direction",
            "pose": "/vision/arrow/pose",
        },
    },
    "aruco": {
        "camera": "fisheye",
        "label": "ArUco",
        "color": (0, 235, 235),
        "topics": {
            "detection": "/vision/aruco/detections",
            "pose": "/vision/aruco/pose",
        },
    },
}


class VisionOverlayStore:
    """缓存各视觉任务结果，并按相机帧筛选可绘制数据。"""

    def __init__(self, sources, timeout, frame_tolerance, hold_time):
        self.sources = copy.deepcopy(sources)
        self.timeout = float(timeout)
        self.frame_tolerance = float(frame_tolerance)
        self.hold_time = max(0.001, float(hold_time))
        self.lock = threading.RLock()
        self.values = {
            source: {kind: None for kind in config["topics"]}
            for source, config in self.sources.items()
        }
        self.drawable_values = {
            source: {kind: None for kind in config["topics"]}
            for source, config in self.sources.items()
        }

    @staticmethod
    def _is_drawable(source, kind, payload):
        """判断消息能否更新最后一次有效标注缓存。"""
        if kind == "detection":
            return has_vision_detections(payload)
        return bool(
            isinstance(payload, dict) and payload.get("valid") is True
        )

    def store(self, source, kind, payload, received_at):
        """保存一类视觉 JSON 的最近一次有效格式消息。"""
        if source not in self.values or kind not in self.values[source]:
            return
        with self.lock:
            previous = self.values[source][kind]
            fps = update_fps(
                previous.get("fps", 0.0) if previous else 0.0,
                previous.get("received_at") if previous else None,
                received_at,
            )
            packet = {
                "payload": copy.deepcopy(payload),
                "received_at": float(received_at),
                "fps": fps,
                "drawn_once": False,
            }
            self.values[source][kind] = packet
            if self._is_drawable(source, kind, payload):
                self.drawable_values[source][kind] = copy.deepcopy(packet)

    def latest(self, source, kind):
        """返回指定视觉通道的最新原始数据包副本。"""
        with self.lock:
            source_values = self.values.get(source, {})
            return copy.deepcopy(source_values.get(kind))

    def _mark_drawn(self, source, kind, received_at):
        """标记结果已通过严格帧同步检查，可进入短暂保留阶段。"""
        with self.lock:
            packet = self.drawable_values[source][kind]
            if (
                    packet is not None
                    and packet.get("received_at") == received_at):
                packet["drawn_once"] = True

    def packets_for_frame(self, camera, frame_stamp, now):
        """返回与当前相机帧同相机、同时间窗口的有效任务结果。"""
        with self.lock:
            values = copy.deepcopy(self.drawable_values)

        active = {}
        for source, config in self.sources.items():
            if config["camera"] != camera:
                continue
            packets = values.get(source, {})
            selected = {}
            for kind, packet in packets.items():
                strict_status = vision_packet_status(
                    packet,
                    now,
                    self.timeout,
                    frame_stamp,
                    self.frame_tolerance,
                )
                if strict_status["online"]:
                    self._mark_drawn(
                        source,
                        kind,
                        packet.get("received_at"),
                    )
                else:
                    held_status = vision_packet_status(
                        packet,
                        now,
                        self.hold_time,
                    )
                    if (
                            not packet
                            or not packet.get("drawn_once")
                            or not held_status["online"]):
                        continue
                payload = packet.get("payload") if packet else None
                if not isinstance(payload, dict):
                    continue
                selected[kind] = payload

            if source == "arrow":
                arrow = selected.get("arrow")
                if arrow and arrow.get("valid") is True:
                    active[source] = selected
            else:
                detection = selected.get("detection")
                if detection and has_vision_detections(detection):
                    active[source] = selected
        return active

    def status(self, now):
        """生成不依赖单帧图像的视觉诊断状态。"""
        with self.lock:
            values = copy.deepcopy(self.values)
        payload = {}
        for source, config in self.sources.items():
            channels = {}
            for kind, packet in values[source].items():
                status = vision_packet_status(packet, now, self.timeout)
                status["fps"] = (
                    packet.get("fps", 0.0) if packet else 0.0
                )
                message = packet.get("payload") if packet else None
                if kind == "detection":
                    status["valid"] = has_vision_detections(message)
                else:
                    status["valid"] = bool(
                        isinstance(message, dict) and message.get("valid")
                    )
                channels[kind] = status
            payload[source] = {
                "camera": config["camera"],
                "label": config["label"],
                "topics": copy.deepcopy(config["topics"]),
                "channels": channels,
                "fps": channels.get("detection", {}).get("fps", 0.0),
            }
        return payload


class CameraStream:
    """保存单路相机最新 JPEG 和流状态。"""

    def __init__(self, name, topic, bridge, jpeg_quality, max_width,
                 timeout, stream_fps, overlay_callback=None):
        self.name = name
        self.topic = topic
        self.bridge = bridge
        self.jpeg_quality = int(max(20, min(100, jpeg_quality)))
        self.max_width = int(max_width)
        self.timeout = float(timeout)
        self.stream_fps = float(max(0.5, stream_fps))
        self.overlay_callback = overlay_callback

        self.lock = threading.Lock()
        self.jpeg = None
        self.sequence = 0
        self.received_at = None
        self.fps = 0.0
        self.width = 0
        self.height = 0

    def callback(self, message):
        """转换 ROS 图像并只保留最新 JPEG。"""
        try:
            frame = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
        except CvBridgeError as exc:
            rospy.logerr_throttle(
                2.0,
                "state_web: %s cv_bridge 转换失败: %s",
                self.name,
                str(exc),
            )
            return

        height, width = frame.shape[:2]
        if self.max_width > 0 and width > self.max_width:
            scale = self.max_width / float(width)
            width = self.max_width
            height = max(1, int(round(height * scale)))
            frame = cv2.resize(
                frame,
                (width, height),
                interpolation=cv2.INTER_AREA,
            )

        # 在最终输出尺寸上绘制，防止文字随原始视频一起缩小。
        if self.overlay_callback is not None:
            try:
                frame = self.overlay_callback(
                    self.name,
                    frame,
                    ros_stamp_sec(message.header),
                )
            except Exception as exc:
                rospy.logerr_throttle(
                    2.0,
                    "state_web: %s 视觉标注绘制失败: %s",
                    self.name,
                    str(exc),
                )

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            rospy.logwarn_throttle(
                2.0,
                "state_web: %s JPEG 编码失败",
                self.name,
            )
            return

        now = time.time()
        with self.lock:
            self.fps = update_fps(
                self.fps,
                self.received_at,
                now,
            )
            self.jpeg = encoded.tobytes()
            self.received_at = now
            self.width = width
            self.height = height
            self.sequence += 1

    def frame(self):
        """返回当前 JPEG 和帧序号。"""
        with self.lock:
            return self.jpeg, self.sequence

    def status(self, now):
        """返回图像在线状态。"""
        with self.lock:
            received_at = self.received_at
            payload = {
                "name": self.name,
                "topic": self.topic,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "sequence": self.sequence,
            }
        payload.update(health_state(received_at, now, self.timeout))
        return payload


class StateWebNode:
    """AUV 运行状态 Web 监控节点。"""

    def __init__(self):
        self.host = rospy.get_param("~host", "0.0.0.0")
        self.port = int(rospy.get_param("~port", 8088))

        self.world_frame = rospy.get_param("~world_frame", "map")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.imu_frame = rospy.get_param("~imu_frame", "imu")
        self.camera_frame = rospy.get_param(
            "~camera_frame", "camera_center"
        )
        self.camera_label = rospy.get_param("~camera_label", "camera")
        self.hand_frame = rospy.get_param("~hand_frame", "hand")

        self.stream_fps = float(rospy.get_param("~stream_fps", 8.0))
        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 75))
        self.image_max_width = int(
            rospy.get_param("~image_max_width", 960)
        )
        self.image_timeout = float(
            rospy.get_param("~image_timeout", 3.0)
        )
        self.state_timeout = float(
            rospy.get_param("~state_timeout", 2.0)
        )
        self.command_timeout = float(
            rospy.get_param("~command_timeout", 2.0)
        )
        self.tf_timeout = float(rospy.get_param("~tf_timeout", 2.0))
        self.tf_poll_hz = float(rospy.get_param("~tf_poll_hz", 10.0))
        self.vision_timeout = float(
            rospy.get_param("~vision_timeout", 2.0)
        )
        self.vision_frame_tolerance = float(
            rospy.get_param("~vision_frame_tolerance", 0.5)
        )
        self.vision_overlay_hold = float(
            rospy.get_param("~vision_overlay_hold", 1.0)
        )
        self.vision_map_history_limit = int(
            rospy.get_param("~vision_map_history_limit", 20)
        )
        self.aruco_history_window = int(
            rospy.get_param("~aruco_history_window", 10)
        )
        self.aruco_required_count = int(
            rospy.get_param("~aruco_required_count", 3)
        )
        self.aruco_min_confidence = float(
            rospy.get_param("~aruco_min_confidence", 0.5)
        )
        self.trajectory_sample_hz = float(
            rospy.get_param("~trajectory_sample_hz", 1.0)
        )
        self.trajectory_duration_sec = float(
            rospy.get_param("~trajectory_duration_sec", 1000.0)
        )
        self.target_position_threshold_m = float(
            rospy.get_param("~target_position_threshold_m", 0.01)
        )
        self.target_heading_threshold_deg = float(
            rospy.get_param("~target_heading_threshold_deg", 1.0)
        )

        self.topics = {
            "left": rospy.get_param(
                "~left_image_topic", "/left/image_raw"
            ),
            "right": rospy.get_param(
                "~right_image_topic", "/right/image_raw"
            ),
            "fisheye": rospy.get_param(
                "~fisheye_image_topic", "/fisheye_camera/image_raw"
            ),
            "feedback": rospy.get_param(
                "~auv_status_topic", "/status/auv"
            ),
            "velocity": rospy.get_param(
                "~velocity_topic", "/status/vel"
            ),
            "pose_command": rospy.get_param(
                "~pose_command_topic", "/cmd/pose/ned"
            ),
            "actuator_command": rospy.get_param(
                "~actuator_command_topic", "/cmd/actuator"
            ),
            "actuator_feedback": rospy.get_param(
                "~actuator_status_topic", "/status/actuator"
            ),
            "power": rospy.get_param(
                "~power_topic", "/status/power"
            ),
            "motion_state": rospy.get_param(
                "~motion_state_topic", "/motion/state"
            ),
            "motion_diagnostics": rospy.get_param(
                "~motion_diagnostics_topic", "/motion/diagnostics"
            ),
            "origin": rospy.get_param(
                "~world_origin_topic", "/world_origin"
            ),
        }

        self.vision_sources = copy.deepcopy(VISION_SOURCE_DEFAULTS)
        for source, config in self.vision_sources.items():
            for kind, default_topic in config["topics"].items():
                parameter_kind = (
                    "direction" if source == "arrow" and kind == "arrow"
                    else kind
                )
                config["topics"][kind] = rospy.get_param(
                    "~{}_{}_topic".format(source, parameter_kind),
                    default_topic,
                )

        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        self.state_lock = threading.RLock()
        self.values = {}
        self.vision_store = VisionOverlayStore(
            self.vision_sources,
            self.vision_timeout,
            self.vision_frame_tolerance,
            self.vision_overlay_hold,
        )
        self.vision_history = VisionHistoryState(
            map_limit=self.vision_map_history_limit,
            aruco_window=self.aruco_history_window,
            aruco_required_count=self.aruco_required_count,
            aruco_min_confidence=self.aruco_min_confidence,
        )
        self.vision_history_operation_lock = threading.RLock()
        self.vision_pending_lock = threading.RLock()
        self.vision_pending = {}
        self.navigation_history = NavigationHistoryState(
            trajectory_hz=self.trajectory_sample_hz,
            trajectory_duration_sec=self.trajectory_duration_sec,
            target_limit=2,
            target_position_threshold_m=self.target_position_threshold_m,
            target_heading_threshold_deg=self.target_heading_threshold_deg,
        )
        self.origin_revision = OriginRevision()
        # 使用哨兵值，确保时间戳为 0 的首组 TF 也能被记录。
        self.last_tf_signature = object()

        self.cameras = {
            name: CameraStream(
                name=name,
                topic=self.topics[name],
                bridge=self.bridge,
                jpeg_quality=self.jpeg_quality,
                max_width=self.image_max_width,
                timeout=self.image_timeout,
                stream_fps=self.stream_fps,
                overlay_callback=self._annotate_frame,
            )
            for name in ("left", "right", "fisheye")
        }

        self._create_subscribers()
        period = 1.0 / max(0.5, self.tf_poll_hz)
        self.tf_timer = rospy.Timer(
            rospy.Duration(period),
            self._update_tf,
        )

        package_path = rospkg.RosPack().get_path("state_web")
        self.www_path = os.path.join(package_path, "www")
        self.app = Flask(
            __name__,
            static_folder=self.www_path,
            static_url_path="/assets",
        )
        self._configure_routes()
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        rospy.loginfo(
            "state_web: 已启动，Web=http://%s:%d，TF=%s -> [%s, %s, %s, %s]",
            self.host,
            self.port,
            self.world_frame,
            self.imu_frame,
            self.base_frame,
            self.camera_frame,
            self.hand_frame,
        )
        for key in (
                "left", "right", "fisheye", "feedback", "velocity",
                "pose_command", "actuator_command", "actuator_feedback",
                "power", "motion_state", "motion_diagnostics", "origin"):
            rospy.loginfo("state_web: %s 话题 %s", key, self.topics[key])
        for source, config in self.vision_sources.items():
            for kind, topic in config["topics"].items():
                rospy.loginfo(
                    "state_web: 视觉%s/%s话题 %s",
                    source,
                    kind,
                    topic,
                )

    def _create_subscribers(self):
        """创建所有只读 ROS 订阅。"""
        for name in ("left", "right", "fisheye"):
            rospy.Subscriber(
                self.topics[name],
                Image,
                self.cameras[name].callback,
                queue_size=1,
                buff_size=2 ** 24,
            )

        rospy.Subscriber(
            self.topics["feedback"],
            AUVData,
            self._feedback_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["velocity"],
            TwistStamped,
            self._velocity_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["pose_command"],
            PoseNEDcmd,
            self._pose_command_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["actuator_command"],
            ActuatorControl,
            self._actuator_command_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["actuator_feedback"],
            ActuatorControl,
            self._actuator_feedback_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["power"],
            SensorStatus,
            self._power_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["motion_state"],
            MotionState,
            self._motion_state_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["motion_diagnostics"],
            String,
            self._motion_diagnostics_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.topics["origin"],
            NavSatFix,
            self._origin_callback,
            queue_size=1,
        )
        for source, config in self.vision_sources.items():
            for kind, topic in config["topics"].items():
                rospy.Subscriber(
                    topic,
                    String,
                    lambda message, task=source, channel=kind:
                    self._vision_callback(task, channel, message),
                    queue_size=1,
                )

    def _vision_callback(self, source, kind, message):
        """接收并缓存一类视觉任务 JSON，格式错误时直接忽略。"""
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(
                2.0,
                "state_web: 视觉%s/%s JSON无效: %s",
                source,
                kind,
                error,
            )
            return
        if not isinstance(payload, dict):
            rospy.logwarn_throttle(
                2.0,
                "state_web: 视觉%s/%s JSON不是对象",
                source,
                kind,
            )
            return
        received_at = time.time()
        self.vision_store.store(source, kind, payload, received_at)
        with self.vision_history_operation_lock:
            if source == "aruco" and kind == "detection":
                self.vision_history.append_aruco_payload(payload, received_at)
            if source in (
                    "line", "red_circle", "shapes", "rectangle", "arrow"):
                self._queue_visual_map_input(source, kind, received_at)

    def _queue_visual_map_input(self, source, kind, received_at):
        """把有效三维视觉结果放入等待时间戳 TF 的短期队列。"""
        if source != "arrow" and kind != "pose":
            return
        if source == "arrow" and kind not in ("pose", "arrow"):
            return

        pose_packet = self.vision_store.latest(source, "pose")
        pose_payload = pose_packet.get("payload") if pose_packet else None
        if not isinstance(pose_payload, dict):
            return
        stamp = safe_float(pose_payload.get("stamp"))
        confidence = safe_float(pose_payload.get("confidence"))
        frame_id = str(pose_payload.get("frame_id") or "").strip().lstrip("/")
        points = extract_vision_pose_points(source, pose_payload)
        category = vision_map_category(
            source,
            pose_payload.get("class_name"),
        )
        if (
                stamp is None
                or stamp <= 0.0
                or confidence is None
                or not 0.0 <= confidence <= 1.0
                or not frame_id
                or category is None
                or (source == "line" and len(points) < 2)
                or (source != "line" and len(points) != 1)):
            return

        direction = None
        if source == "arrow":
            arrow_packet = self.vision_store.latest("arrow", "arrow")
            arrow_payload = arrow_packet.get("payload") if arrow_packet else None
            if not isinstance(arrow_payload, dict):
                return
            arrow_stamp = safe_float(arrow_payload.get("stamp"))
            vector = arrow_payload.get("direction_2d")
            if (
                    arrow_payload.get("valid") is not True
                    or arrow_stamp is None
                    or abs(arrow_stamp - stamp) > self.vision_frame_tolerance
                    or not isinstance(vector, dict)):
                return
            direction_x = safe_float(vector.get("x"))
            direction_y = safe_float(vector.get("y"))
            direction_confidence = safe_float(
                arrow_payload.get(
                    "direction_confidence",
                    arrow_payload.get("confidence"),
                )
            )
            if (
                    direction_x is None
                    or direction_y is None
                    or direction_confidence is None
                    or not 0.0 <= direction_confidence <= 1.0
                    or math.hypot(direction_x, direction_y) <= 1e-9):
                return
            direction = {"x": direction_x, "y": direction_y, "z": 0.0}
            confidence = min(confidence, direction_confidence)

        dedupe_key = "{}:{:.9f}".format(source, stamp)
        event = {
            "source": source,
            "category": category,
            "stamp": stamp,
            "frame_id": frame_id,
            "confidence": confidence,
            "points": points,
            "direction": direction,
            "received_at": float(received_at),
            "expires_at": float(received_at) + max(0.05, self.tf_timeout),
            "dedupe_key": dedupe_key,
        }
        with self.vision_pending_lock:
            if dedupe_key not in self.vision_pending:
                self.vision_pending[dedupe_key] = event

    def _transform_visual_event(self, event):
        """使用检测时刻 TF 把相机点和箭头方向转换到 map/NED。"""
        stamp = rospy.Time.from_sec(float(event["stamp"]))
        translation, quaternion = self.tf_listener.lookupTransform(
            self.world_frame,
            event["frame_id"],
            stamp,
        )
        map_points, map_direction = transform_visual_geometry(
            event["points"],
            event.get("direction"),
            translation,
            quaternion,
        )

        record = {
            "stamp": float(event["stamp"]),
            "received_at": float(event["received_at"]),
            "confidence": float(event["confidence"]),
            "points": map_points,
        }
        if map_direction is not None:
            record["direction_ne"] = {
                "north": map_direction["north"],
                "east": map_direction["east"],
            }
            record["heading_deg"] = map_direction["heading_deg"]
        return record

    def _process_pending_visual_map(self):
        """重试待处理视觉结果，直至成功或超过 TF 等待时间。"""
        with self.vision_history_operation_lock:
            self._process_pending_visual_map_locked()

    def _process_pending_visual_map_locked(self):
        """在清除互斥锁保护下处理待转换的视觉结果。"""
        now = time.time()
        with self.vision_pending_lock:
            pending = list(self.vision_pending.items())
        for key, event in pending:
            if now > event["expires_at"]:
                with self.vision_pending_lock:
                    self.vision_pending.pop(key, None)
                rospy.logwarn_throttle(
                    2.0,
                    "state_web: 视觉%s在检测时刻的TF等待超时，已丢弃",
                    event["source"],
                )
                continue
            try:
                record = self._transform_visual_event(event)
            except (
                    tf.Exception,
                    tf.LookupException,
                    tf.ConnectivityException,
                    tf.ExtrapolationException,
                    ValueError,
            ):
                continue
            self.vision_history.append_map(
                event["category"],
                record,
                event["dedupe_key"],
            )
            with self.vision_pending_lock:
                self.vision_pending.pop(key, None)

    def _clear_pending_visual_map(self):
        """清除尚未转换的旧坐标视觉结果。"""
        with self.vision_pending_lock:
            self.vision_pending.clear()

    def _store(self, name, data, ros_stamp=None, received_at=None):
        """线程安全保存一份话题快照。"""
        with self.state_lock:
            self.values[name] = {
                "data": data,
                "received_at": (
                    time.time() if received_at is None else received_at
                ),
                "ros_stamp": ros_stamp,
            }

    def _snapshot(self, name, timeout, now, persistent=False):
        """复制话题快照并附加健康状态。"""
        with self.state_lock:
            value = copy.deepcopy(self.values.get(name))
        if value is None:
            result = health_state(None, now, timeout)
            result.update({
                "data": None,
                "received_at": None,
                "ros_stamp": None,
            })
            return result

        effective_timeout = None if persistent else timeout
        result = health_state(
            value.get("received_at"),
            now,
            effective_timeout,
        )
        result.update(value)
        return result

    @staticmethod
    def _pixel(point, width, height, source_width=None, source_height=None):
        """将检测图像中的 JSON 像素点映射到当前相机帧。"""
        return map_pixel_to_frame(
            point,
            width,
            height,
            source_width,
            source_height,
        )

    @staticmethod
    def _finite_text(value, digits=2, prefix=""):
        """仅格式化有限数值，避免把无效值画到画面上。"""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return ""
        if not (-float("inf") < number < float("inf")):
            return ""
        return "{}{:0.{precision}f}".format(
            prefix,
            number,
            precision=digits,
        )

    @staticmethod
    def _position_text(pose):
        """提取位姿结果中最有价值的距离信息。"""
        if not isinstance(pose, dict) or pose.get("valid") is not True:
            return ""
        position = pose.get("position_m")
        if not isinstance(position, dict):
            return ""
        distance = StateWebNode._finite_text(position.get("z"), 2, "Z=")
        return "{}m".format(distance) if distance else ""

    @staticmethod
    def _draw_label(frame, text, anchor, color, slot):
        """在最终视频尺寸上绘制带半透明底板的大号标签。"""
        if not text:
            return
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, scale, thickness
        )
        line_step = text_height + baseline + 10
        x = max(4, min(width - 4, int(anchor[0])))
        y = max(
            text_height + baseline + 8,
            min(height - 5, int(anchor[1]) + slot * line_step),
        )
        left = max(0, min(width - text_width - 12, x))
        top = max(0, y - text_height - baseline - 8)
        right = min(width - 1, left + text_width + 12)
        bottom = min(height - 1, y + 4)
        overlay = frame.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), (8, 16, 24), -1)
        cv2.addWeighted(overlay, 0.66, frame, 0.34, 0, frame)
        cv2.putText(
            frame,
            text,
            (left + 6, bottom - baseline - 3),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_detection_source(self, frame, source, packets, label_slot):
        """绘制通用 YOLO、线段和 ArUco 检测结果。"""
        config = self.vision_sources[source]
        color = config["color"]
        payload = packets["detection"]
        pose_text = self._position_text(packets.get("pose"))
        height, width = frame.shape[:2]
        source_width = payload.get("image_width")
        source_height = payload.get("image_height")
        for index, item in enumerate(payload.get("detections", [])):
            if not isinstance(item, dict):
                continue
            anchor = None
            corners = [
                self._pixel(
                    point, width, height, source_width, source_height
                )
                for point in item.get("corners", [])
            ]
            corners = [point for point in corners if point is not None]
            keypoints = [
                self._pixel(
                    point, width, height, source_width, source_height
                )
                for point in item.get("keypoints", [])
            ]
            keypoints = [point for point in keypoints if point is not None]
            polygon = [
                self._pixel(
                    point, width, height, source_width, source_height
                )
                for point in item.get("polygon", [])
            ]
            polygon = [point for point in polygon if point is not None]
            if len(corners) >= 3:
                cv2.polylines(
                    frame,
                    [np.array(corners, dtype="int32")],
                    True,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                anchor = corners[0]
            elif len(keypoints) >= 2:
                cv2.polylines(
                    frame,
                    [np.array(keypoints, dtype="int32")],
                    False,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                for point in keypoints:
                    cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
                anchor = keypoints[0]
            elif len(polygon) >= 3:
                cv2.polylines(
                    frame,
                    [np.array(polygon, dtype="int32")],
                    True,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                anchor = polygon[0]
            else:
                bbox = item.get("bbox") or {}
                try:
                    top_left = self._pixel(
                        {"u": bbox["x1"], "v": bbox["y1"]},
                        width,
                        height,
                        source_width,
                        source_height,
                    )
                    bottom_right = self._pixel(
                        {"u": bbox["x2"], "v": bbox["y2"]},
                        width,
                        height,
                        source_width,
                        source_height,
                    )
                    if top_left is None or bottom_right is None:
                        raise ValueError("invalid bbox")
                    x1, y1 = top_left
                    x2, y2 = bottom_right
                except (KeyError, TypeError, ValueError):
                    center = self._pixel(
                        item.get("center"),
                        width,
                        height,
                        source_width,
                        source_height,
                    )
                    if center is None:
                        continue
                    cv2.circle(frame, center, 6, color, 2, cv2.LINE_AA)
                    anchor = center
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    anchor = (x1, y1)
            class_name = str(item.get("class_name") or config["label"])
            confidence = self._finite_text(item.get("confidence"), 2, "C=")
            details = " ".join(
                value for value in (config["label"], class_name, confidence,
                                    pose_text if index == 0 else "") if value
            )
            self._draw_label(frame, details, anchor, color, label_slot + index)

    def _draw_arrow_source(self, frame, packets, label_slot):
        """绘制箭头方向估计的框、尖端、尾端和方向线。"""
        payload = packets["arrow"]
        config = self.vision_sources["arrow"]
        color = config["color"]
        height, width = frame.shape[:2]
        source_width = payload.get("image_width")
        source_height = payload.get("image_height")
        bbox = payload.get("bbox") or {}
        anchor = self._pixel(
            payload.get("center"),
            width,
            height,
            source_width,
            source_height,
        )
        try:
            top_left = self._pixel(
                {"u": bbox["x1"], "v": bbox["y1"]},
                width,
                height,
                source_width,
                source_height,
            )
            bottom_right = self._pixel(
                {"u": bbox["x2"], "v": bbox["y2"]},
                width,
                height,
                source_width,
                source_height,
            )
            if top_left is None or bottom_right is None:
                raise ValueError("invalid bbox")
            x1, y1 = top_left
            x2, y2 = bottom_right
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            anchor = (x1, y1)
        except (KeyError, TypeError, ValueError):
            pass
        tail = self._pixel(
            payload.get("tail"),
            width,
            height,
            source_width,
            source_height,
        )
        tip = self._pixel(
            payload.get("tip"),
            width,
            height,
            source_width,
            source_height,
        )
        if tail is not None and tip is not None:
            cv2.arrowedLine(frame, tail, tip, color, 3, cv2.LINE_AA, tipLength=0.22)
            cv2.circle(frame, tail, 4, color, -1, cv2.LINE_AA)
            anchor = anchor or tail
        if anchor is None:
            return
        direction = str(payload.get("discrete_direction") or "unknown")
        angle = self._finite_text(payload.get("angle_deg"), 1, "A=")
        confidence = self._finite_text(payload.get("direction_confidence"), 2, "C=")
        label = " ".join(
            value for value in (config["label"], direction, angle, confidence)
            if value
        )
        self._draw_label(frame, label, anchor, color, label_slot)

    def _annotate_frame(self, camera, frame, frame_stamp):
        """在左目和鱼眼原始帧上叠加当前有效的视觉检测。"""
        if camera not in ("left", "fisheye"):
            return frame
        active = self.vision_store.packets_for_frame(
            camera,
            frame_stamp,
            time.time(),
        )
        if not active:
            return frame
        frame = frame.copy()
        labels = []
        label_slot = 0
        for source, packets in active.items():
            if source == "arrow":
                self._draw_arrow_source(frame, packets, label_slot)
            else:
                self._draw_detection_source(frame, source, packets, label_slot)
            labels.append(self.vision_sources[source]["label"])
            label_slot += max(1, len(
                packets.get("detection", {}).get("detections", [])
            ))
        summary = "Active: {}".format(", ".join(labels))
        self._draw_label(frame, summary, (8, 20), (235, 235, 235), 0)
        return frame

    def _feedback_callback(self, message):
        """接收 AUV 原始反馈。"""
        mode = int(message.control_mode)
        sensor = message.sensor
        data = {
            "control_mode": mode,
            "control_mode_name": CONTROL_MODE_NAMES.get(mode, "未知"),
            "pose": serialize_auv_pose(message.pose),
            "target": serialize_auv_pose(message.target),
            "motor_force": serialize_motor(message.motor_force),
            "linear_velocity_raw": [
                float(value) for value in message.linear_velocity
            ],
            "angular_velocity_raw_degps": [
                float(value) for value in message.angular_velocity
            ],
            "sensor": {
                "temperature_c": float(sensor.temperature),
                "voltage_v": float(sensor.voltage),
                "current_a": float(sensor.current),
                "battery_percent": int(sensor.battery),
                "leak_alarm": bool(sensor.leak_alarm),
                "sensor_valid": int(sensor.sensor_valid),
                "sensor_updated": int(sensor.sensor_updated),
                "fault_status": int(sensor.fault_status),
                "power_status": int(sensor.power_status),
            },
        }
        stamp = ros_stamp_sec(message.header)
        self._store("feedback", data, ros_stamp=stamp)

    def _velocity_callback(self, message):
        """接收本体速度反馈。"""
        twist = message.twist
        data = {
            "frame_id": message.header.frame_id,
            "linear_mps": {
                "x": float(twist.linear.x),
                "y": float(twist.linear.y),
                "z": float(twist.linear.z),
            },
            "angular_radps": {
                "x": float(twist.angular.x),
                "y": float(twist.angular.y),
                "z": float(twist.angular.z),
            },
        }
        stamp = ros_stamp_sec(message.header)
        self._store("velocity", data, ros_stamp=stamp)

    def _pose_command_callback(self, message):
        """接收 NED 运动控制指令。"""
        mode = int(message.mode)
        received_at = time.time()
        target = serialize_pose_stamped(message.target)
        data = {
            "mode": mode,
            "mode_name": CONTROL_MODE_NAMES.get(mode, "未知"),
            "target": target,
            "force": serialize_motor(message.force),
        }
        self.navigation_history.append_target(target, received_at)
        self._store(
            "pose_command",
            data,
            ros_stamp=ros_stamp_sec(message.target.header),
            received_at=received_at,
        )

    def _actuator_command_callback(self, message):
        """接收执行器控制指令。"""
        self._store("actuator_command", serialize_actuator(message))

    def _actuator_feedback_callback(self, message):
        """接收执行器硬件反馈。"""
        self._store("actuator_feedback", serialize_actuator(message))

    def _power_callback(self, message):
        """接收两路电源状态。"""
        data = {
            "checksum_ok": bool(message.checksum_ok),
            "power1": {
                "valid": bool(message.power1_valid),
                "voltage_v": float(message.power1_voltage),
                "current_a": float(message.power1_current),
                "power_w": float(message.power1_power),
            },
            "power2": {
                "valid": bool(message.power2_valid),
                "voltage_v": float(message.power2_voltage),
                "current_a": float(message.power2_current),
                "power_w": float(message.power2_power),
            },
        }
        self._store(
            "power",
            data,
            ros_stamp=ros_stamp_sec(message.header),
        )

    def _motion_state_callback(self, message):
        """接收运动监督状态。"""
        state = int(message.state)
        data = {
            "state": state,
            "state_name": MOTION_STATE_NAMES.get(state, "未知"),
            "goal_active": bool(message.goal_active),
            "goal": serialize_pose_stamped(message.goal),
            "position_error_m": float(message.position_error),
            "yaw_error_rad": float(message.yaw_error),
            "horizontal_speed_mps": float(message.horizontal_speed),
            "yaw_rate_radps": float(message.yaw_rate),
            "force": {
                "tx": int(message.tx),
                "ty": int(message.ty),
                "mz": int(message.mz),
            },
            "reason": message.reason,
        }
        self._store(
            "motion_state",
            data,
            ros_stamp=ros_stamp_sec(message.header),
        )

    def _motion_diagnostics_callback(self, message):
        """接收统一三轴控制器的 JSON 诊断快照。"""
        try:
            data = json.loads(message.data)
        except (TypeError, ValueError) as error:
            rospy.logwarn_throttle(
                2.0, "state_web: 运动控制诊断 JSON 无效: %s", error)
            return
        if not isinstance(data, dict):
            rospy.logwarn_throttle(2.0, "state_web: 运动控制诊断不是对象")
            return
        self._store("motion_diagnostics", data)

    def _origin_callback(self, message):
        """接收锁存世界原点并维护版本号。"""
        changed, revision = self.origin_revision.update(
            message.latitude,
            message.longitude,
            message.altitude,
        )
        data = {
            "frame_id": message.header.frame_id,
            "latitude_deg": float(message.latitude),
            "longitude_deg": float(message.longitude),
            "depth_m": float(message.altitude),
            "revision": revision,
            "changed": changed,
        }
        self._store(
            "origin",
            data,
            ros_stamp=ros_stamp_sec(message.header),
        )
        if changed:
            with self.vision_history_operation_lock:
                self._clear_pending_visual_map()
                self.vision_history.clear_map(time.time())
            self.navigation_history.clear_trajectory(time.time())
            rospy.logwarn(
                "state_web: 世界原点版本更新为 %d，Web 轨迹将清空",
                revision,
            )

    def _lookup_tf_pose(self, child_frame):
        """查询世界坐标系到指定机体坐标系的最新位姿。"""
        try:
            translation, quaternion = self.tf_listener.lookupTransform(
                self.world_frame,
                child_frame,
                rospy.Time(0),
            )
        except (
                tf.Exception,
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
        ):
            return None

        lookup_received_at = time.time()
        stamp_sec = None
        try:
            stamp = self.tf_listener.getLatestCommonTime(
                self.world_frame,
                child_frame,
            )
            if stamp != rospy.Time(0):
                stamp_sec = stamp.to_sec()
        except (
                tf.Exception,
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
        ):
            # 位姿已经通过 Time(0) 成功获取，时间戳失败不能使该帧失效。
            pass
        orientation = quaternion_to_euler_deg(*quaternion)
        return {
            "frame_id": self.world_frame,
            "child_frame_id": child_frame,
            "stamp_sec": stamp_sec,
            "lookup_received_at": lookup_received_at,
            "position_m": {
                "x": float(translation[0]),
                "y": float(translation[1]),
                "z": float(translation[2]),
            },
            "orientation_quaternion": {
                "x": float(quaternion[0]),
                "y": float(quaternion[1]),
                "z": float(quaternion[2]),
                "w": float(quaternion[3]),
            },
            "orientation_deg": orientation,
        }

    def _update_tf(self, unused_event):
        """轮询世界坐标系到 IMU、机体、相机和 hand 的最新动态 TF。"""
        del unused_event
        self._process_pending_visual_map()
        frame_poses = {
            "imu": self._lookup_tf_pose(self.imu_frame),
            "base": self._lookup_tf_pose(self.base_frame),
            "camera": self._lookup_tf_pose(self.camera_frame),
            "hand": self._lookup_tf_pose(self.hand_frame),
        }
        base_pose = frame_poses["base"]
        if base_pose is None:
            return
        self.navigation_history.append_base_pose(base_pose, time.time())

        # 任意坐标系的新时间戳都会触发页面快照更新。
        signature = tuple(
            (
                key,
                None if pose is None else (
                    pose.get("stamp_sec")
                    if pose.get("stamp_sec") is not None
                    else pose.get("lookup_received_at")
                ),
            )
            for key, pose in sorted(frame_poses.items())
        )
        if signature == self.last_tf_signature:
            return
        self.last_tf_signature = signature

        data = copy.deepcopy(base_pose)
        data["frame_poses"] = frame_poses
        self._store(
            "tf",
            data,
            ros_stamp=base_pose.get("stamp_sec"),
        )

    @staticmethod
    def _attitude_candidate(snapshot, source):
        """从反馈或 TF 快照提取统一姿态。"""
        data = snapshot.get("data")
        if not data:
            return None

        if source == "status_auv":
            pose = data.get("pose") or {}
            roll = pose.get("roll_deg")
            pitch = pose.get("pitch_deg")
            heading = pose.get("heading_deg")
        else:
            orientation = data.get("orientation_deg") or {}
            roll = orientation.get("roll_deg")
            pitch = orientation.get("pitch_deg")
            heading = orientation.get("heading_deg")

        if None in (roll, pitch, heading):
            return None
        return {
            "valid": bool(snapshot.get("online")),
            "source": source,
            "age_sec": snapshot.get("age_sec"),
            "received_at": snapshot.get("received_at"),
            "roll_deg": roll,
            "pitch_deg": pitch,
            "heading_deg": normalize_heading(heading),
        }

    def _build_attitude(self, feedback, tf_pose, pose_command):
        """按 AUV 反馈优先、TF 回退生成姿态仪表数据。"""
        feedback_candidate = self._attitude_candidate(
            feedback,
            "status_auv",
        )
        tf_candidate = self._attitude_candidate(tf_pose, "tf")

        actual = select_attitude(feedback_candidate, tf_candidate)

        target = None
        command_data = pose_command.get("data")
        if command_data:
            orientation = (
                command_data.get("target", {})
                .get("orientation_deg") or {}
            )
            heading = orientation.get("heading_deg")
            if heading is not None:
                target = {
                    "valid": bool(pose_command.get("online")),
                    "heading_deg": normalize_heading(heading),
                    "age_sec": pose_command.get("age_sec"),
                }

        error = None
        if (
                actual is not None
                and actual.get("valid")
                and target is not None
                and target.get("valid")):
            error = shortest_heading_error(
                target["heading_deg"],
                actual["heading_deg"],
            )

        return {
            "valid": bool(actual and actual.get("valid")),
            "actual": actual,
            "target": target,
            "heading_error_deg": error,
        }

    def status_payload(self):
        """生成网页使用的统一只读状态快照。"""
        now = time.time()
        streams = {
            name: camera.status(now)
            for name, camera in self.cameras.items()
        }
        vision = self.vision_store.status(now)
        feedback = self._snapshot(
            "feedback", self.state_timeout, now
        )
        velocity = self._snapshot(
            "velocity", self.state_timeout, now
        )
        pose_command = self._snapshot(
            "pose_command", self.command_timeout, now
        )
        actuator_command = self._snapshot(
            "actuator_command", self.command_timeout, now
        )
        actuator_feedback = self._snapshot(
            "actuator_feedback", self.state_timeout, now
        )
        power = self._snapshot("power", self.state_timeout, now)
        motion_state = self._snapshot(
            "motion_state", self.state_timeout, now
        )
        motion_diagnostics = self._snapshot(
            "motion_diagnostics", self.state_timeout, now
        )
        origin = self._snapshot(
            "origin", None, now, persistent=True
        )
        tf_pose = self._snapshot("tf", self.tf_timeout, now)
        attitude = self._build_attitude(
            feedback,
            tf_pose,
            pose_command,
        )
        unused_origin_values, origin_revision = self.origin_revision.snapshot()
        del unused_origin_values
        vision_history = self.vision_history.snapshot(
            now,
            self.world_frame,
            origin_revision,
        )
        navigation_history = self.navigation_history.snapshot(
            now,
            self.world_frame,
        )

        snapshots = {
            "feedback": feedback,
            "velocity": velocity,
            "pose_command": pose_command,
            "actuator_command": actuator_command,
            "actuator_feedback": actuator_feedback,
            "power": power,
            "motion_state": motion_state,
            "motion_diagnostics": motion_diagnostics,
            "origin": origin,
            "tf": tf_pose,
        }
        topic_health = {
            name: {
                "online": bool(snapshot.get("online")),
                "age_sec": snapshot.get("age_sec"),
                "topic": self.topics.get(name),
            }
            for name, snapshot in snapshots.items()
        }
        for name, stream in streams.items():
            topic_health[name] = {
                "online": bool(stream.get("online")),
                "age_sec": stream.get("age_sec"),
                "topic": stream.get("topic"),
            }
        for source, source_status in vision.items():
            for kind, channel in source_status["channels"].items():
                topic_health["vision_{}_{}".format(source, kind)] = {
                    "online": bool(channel.get("online")),
                    "age_sec": channel.get("age_sec"),
                    "topic": source_status["topics"].get(kind),
                    "valid": bool(channel.get("valid")),
                    "fps": channel.get("fps", 0.0),
                }

        payload = {
            "server_time": now,
            "ready": bool(origin.get("online") and tf_pose.get("online")),
            "frames": {
                "world": self.world_frame,
                "base": self.base_frame,
                "imu": self.imu_frame,
                "camera": self.camera_label,
                "camera_tf": self.camera_frame,
                "hand": self.hand_frame,
            },
            "streams": streams,
            "tf": tf_pose,
            "feedback": feedback,
            "velocity": velocity,
            "pose_command": pose_command,
            "actuator_command": actuator_command,
            "actuator_feedback": actuator_feedback,
            "power": power,
            "motion_state": motion_state,
            "motion_diagnostics": motion_diagnostics,
            "origin": origin,
            "attitude": attitude,
            "vision": vision,
            "vision_map": vision_history["vision_map"],
            "aruco_history": vision_history["aruco_history"],
            "base_trajectory": navigation_history["base_trajectory"],
            "target_history": navigation_history["target_history"],
            "topic_health": topic_health,
        }
        return sanitize_json(payload)

    def _configure_routes(self):
        """注册 Flask 只读接口。"""

        @self.app.after_request
        def disable_cache(response):
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, max-age=0"
            )
            return response

        @self.app.route("/")
        def index():
            return send_from_directory(self.www_path, "index.html")

        @self.app.route("/stream/<camera_name>")
        def stream(camera_name):
            camera = self.cameras.get(camera_name)
            if camera is None:
                abort(404)
            return Response(
                self._mjpeg_generator(camera),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/api/status")
        def api_status():
            return jsonify(self.status_payload())

        @self.app.route("/api/vision-history/clear", methods=["POST"])
        def clear_vision_history():
            cleared_at = time.time()
            with self.vision_history_operation_lock:
                self._clear_pending_visual_map()
                version = self.vision_history.clear_all(cleared_at)
            return jsonify({
                "ok": True,
                "cleared_at": cleared_at,
                "history_version": version,
            })

        @self.app.route("/api/base-trajectory/clear", methods=["POST"])
        def clear_base_trajectory():
            cleared_at = time.time()
            version = self.navigation_history.clear_trajectory(cleared_at)
            return jsonify({
                "ok": True,
                "cleared_at": cleared_at,
                "history_version": version,
            })

        @self.app.route("/health")
        def health():
            payload = self.status_payload()
            return jsonify({
                "ok": True,
                "ros_shutdown": rospy.is_shutdown(),
                "ready": payload["ready"],
                "origin_ready": payload["origin"]["online"],
                "tf_ready": payload["tf"]["online"],
                "server_time": payload["server_time"],
            })

    @staticmethod
    def _mjpeg_generator(camera):
        """持续输出单路 MJPEG，缺流时保持连接等待。"""
        period = 1.0 / camera.stream_fps
        while not rospy.is_shutdown():
            jpeg, unused_sequence = camera.frame()
            del unused_sequence
            if jpeg is None:
                time.sleep(0.1)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            time.sleep(period)

    def run(self):
        """运行多线程 Flask 服务。"""
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )


if __name__ == "__main__":
    try:
        rospy.init_node("state_web", anonymous=False)
        StateWebNode().run()
    except rospy.ROSInterruptException:
        pass
