#!/home/nvidia/venvs/xhy_ros2/bin/python
# -*- coding: utf-8 -*-
"""
名称：state_web_core.py
功能：state_web 使用的纯 Python 数据处理工具
作者：xhy
监听：无
发布：无
记录：
2026.7.17
    新增角度归一化、姿态换算、话题健康状态和原点版本管理。
2026.7.30
    新增检测图像坐标到当前相机帧的分辨率映射。
2026.8.3
    新增视觉地图分类、三维坐标转换、滚动历史和 ArUco 判色工具。
    新增 base_link 定时轨迹与最近两帧目标位姿历史工具。
    目标历史仅在位置变化 0.01m 或航向变化 1° 时滚动。
    base_link 轨迹默认按 1Hz 保留最多 1000 个点。
2026.8.5
    新增 6S4P 锂离子电池电压平滑、SOC 插值和两路电源摘要工具。
    新增按控制模式合并的执行器指令缓存，独立计算灯光与机械指令年龄。
"""

import copy
import math
import threading
from collections import Counter, deque


CONTROL_MODE_NAMES = {
    0: "未知",
    2: "定深",
    3: "定深定向",
    4: "动力定位ROV",
}

MOTION_STATE_NAMES = {
    0: "空闲",
    1: "路径对准",
    2: "路径对准刹停",
    3: "平移",
    4: "平移刹停",
    5: "最终航向对准",
    6: "最终刹停",
    7: "定点接管",
    8: "悬停",
    9: "安全模式",
}

ACTUATOR_MODE_NAMES = {
    0: "状态/不响应",
    1: "补光灯控制",
    2: "执行器控制",
}


VISION_MAP_CATEGORIES = (
    "red_circle",
    "black_square",
    "yellow_circle",
    "red_line",
    "arrow",
    "rectangle_red",
    "rectangle_yellow",
    "rectangle_green",
)

ARUCO_COLOR_BY_ID = {
    1: "yellow",
    2: "yellow",
    3: "green",
    4: "green",
    5: "red",
    6: "red",
}

BATTERY_SOC_CURVE = (
    (3.20, 0.0),
    (3.50, 5.0),
    (3.68, 10.0),
    (3.74, 20.0),
    (3.77, 30.0),
    (3.79, 40.0),
    (3.82, 50.0),
    (3.87, 60.0),
    (3.92, 70.0),
    (4.00, 80.0),
    (4.10, 90.0),
    (4.20, 100.0),
)


def safe_float(value):
    """将数值转换为有限浮点数，无效值返回 None。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def interpolate_battery_soc(cell_voltage, curve=BATTERY_SOC_CURVE):
    """按单节开路电压曲线分段线性估算锂离子电池 SOC。"""
    voltage = safe_float(cell_voltage)
    points = tuple(curve or ())
    if voltage is None or len(points) < 2:
        return None
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        point_voltage = safe_float(point[0])
        point_soc = safe_float(point[1])
        if point_voltage is None or point_soc is None:
            return None
        normalized.append((point_voltage, point_soc))
    normalized.sort(key=lambda item: item[0])
    if voltage <= normalized[0][0]:
        return max(0.0, min(100.0, normalized[0][1]))
    if voltage >= normalized[-1][0]:
        return max(0.0, min(100.0, normalized[-1][1]))
    for lower, upper in zip(normalized, normalized[1:]):
        if lower[0] <= voltage <= upper[0]:
            span = upper[0] - lower[0]
            if span <= 1e-12:
                return max(0.0, min(100.0, upper[1]))
            ratio = (voltage - lower[0]) / span
            soc = lower[1] + ratio * (upper[1] - lower[1])
            return max(0.0, min(100.0, soc))
    return None


class ActuatorCommandState:
    """按消息模式保存执行器指令，避免不同模式的零值互相覆盖。"""

    LIGHT_FIELDS = ("light1", "light2")
    ACTUATOR_FIELDS = (
        "heading_servo",
        "clamp_servo",
        "drive_cmd",
        "drive_speed",
        "red_light",
        "yellow_light",
        "green_light",
    )

    def __init__(self):
        self.lock = threading.RLock()
        self.values = {}
        self.feedback_values = {}

    def update(self, payload, received_at):
        """合并一帧指令；模式 1 和模式 2 只更新各自负责的字段。"""
        if not isinstance(payload, dict):
            return self.snapshot(received_at, 0.0)
        timestamp = safe_float(received_at)
        with self.lock:
            mode = int(payload.get("mode", 0))
            self.values["mode"] = mode
            self.values["mode_name"] = payload.get("mode_name", "未知")
            self.values["last_mode"] = mode
            self.values["last_mode_name"] = payload.get("mode_name", "未知")
            if timestamp is not None:
                self.values["received_at"] = timestamp

            if mode == 1:
                self._copy_fields(payload, self.LIGHT_FIELDS)
                if timestamp is not None:
                    self.values["light_received_at"] = timestamp
            elif mode == 2:
                previous = tuple(
                    self.values.get(field) for field in self.ACTUATOR_FIELDS
                )
                self._copy_fields(payload, self.ACTUATOR_FIELDS)
                current = tuple(
                    self.values.get(field) for field in self.ACTUATOR_FIELDS
                )
                if timestamp is not None:
                    self.values["actuator_received_at"] = timestamp
                    if previous != current:
                        self.values["actuator_changed_at"] = timestamp
                        self.values["actuator_ack_received_at"] = None
                        self.values["actuator_ack_delay_sec"] = None
            else:
                self._copy_fields(
                    payload,
                    self.LIGHT_FIELDS + self.ACTUATOR_FIELDS,
                )
                if timestamp is not None:
                    self.values["light_received_at"] = timestamp
                    self.values["actuator_received_at"] = timestamp
            return copy.deepcopy(self.values)

    def observe_feedback(self, payload, received_at):
        """记录反馈，并锁存指令变化后首次匹配反馈的 Web 接收差。"""
        if not isinstance(payload, dict):
            return
        timestamp = safe_float(received_at)
        with self.lock:
            self.feedback_values = copy.deepcopy(payload)
            if (
                    timestamp is None
                    or self.values.get("actuator_ack_received_at") is not None
                    or not self._matches_locked(payload)):
                return
            changed_at = safe_float(self.values.get("actuator_changed_at"))
            if changed_at is None:
                return
            self.values["actuator_ack_received_at"] = timestamp
            self.values["actuator_ack_delay_sec"] = max(
                0.0,
                timestamp - changed_at,
            )

    def _matches_locked(self, feedback):
        exact_fields = (
            "drive_cmd",
            "red_light",
            "yellow_light",
            "green_light",
        )
        close_fields = ("clamp_servo", "drive_speed")
        for field in exact_fields:
            expected = safe_float(self.values.get(field))
            actual = safe_float(feedback.get(field))
            if expected is None or actual is None or expected != actual:
                return False
        for field in close_fields:
            expected = safe_float(self.values.get(field))
            actual = safe_float(feedback.get(field))
            if (
                    expected is None
                    or actual is None
                    or abs(expected - actual) > 2.0):
                return False
        return True

    def _copy_fields(self, payload, fields):
        for field in fields:
            if field in payload:
                self.values[field] = payload[field]

    def snapshot(self, now, timeout):
        """返回合并结果及两个模式各自的年龄和在线状态。"""
        timestamp = safe_float(now)
        limit = max(0.0, safe_float(timeout) or 0.0)
        with self.lock:
            result = copy.deepcopy(self.values)
        for prefix in ("light", "actuator"):
            received_at = safe_float(result.get(prefix + "_received_at"))
            age = (
                max(0.0, timestamp - received_at)
                if timestamp is not None and received_at is not None
                else None
            )
            result[prefix + "_age_sec"] = age
            result[prefix + "_online"] = bool(
                age is not None and age <= limit
            )
        return result


class PowerSummaryState:
    """维护动力电压滑动窗口并生成控制、动力和电池摘要。"""

    def __init__(self, series_count=6, parallel_count=4,
                 cell_capacity_ah=4.0, pack_capacity_ah=16.0,
                 smoothing_sec=5.0):
        self.series_count = max(1, int(series_count))
        self.parallel_count = max(1, int(parallel_count))
        self.cell_capacity_ah = max(0.0, float(cell_capacity_ah))
        self.pack_capacity_ah = max(0.0, float(pack_capacity_ah))
        self.smoothing_sec = max(0.0, float(smoothing_sec))
        self.samples = deque()
        self.lock = threading.RLock()

    @staticmethod
    def _branch(payload, name):
        branch = payload.get(name) if isinstance(payload, dict) else None
        if not isinstance(branch, dict) or branch.get("valid") is not True:
            return None
        voltage = safe_float(branch.get("voltage_v"))
        current = safe_float(branch.get("current_a"))
        power = safe_float(branch.get("power_w"))
        if voltage is None or current is None or power is None:
            return None
        return {
            "voltage_v": voltage,
            "current_a": current,
            "power_w": power,
        }

    def _trim(self, now):
        cutoff = now - self.smoothing_sec
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def update(self, payload, received_at):
        """接收一帧两路电源反馈并返回不伪造部分数据的状态摘要。"""
        timestamp = safe_float(received_at)
        if timestamp is None:
            return self.snapshot(payload, None)
        checksum_ok = bool(
            isinstance(payload, dict) and payload.get("checksum_ok") is True
        )
        control = self._branch(payload, "power1") if checksum_ok else None
        motive = self._branch(payload, "power2") if checksum_ok else None
        with self.lock:
            if motive is not None and motive["voltage_v"] > 0.0:
                self.samples.append((timestamp, motive["voltage_v"]))
            self._trim(timestamp)
            return self._snapshot_locked(control, motive, checksum_ok)

    def snapshot(self, payload, now):
        """不增加样本地读取当前电源摘要，供测试和诊断使用。"""
        timestamp = safe_float(now)
        checksum_ok = bool(
            isinstance(payload, dict) and payload.get("checksum_ok") is True
        )
        control = self._branch(payload, "power1") if checksum_ok else None
        motive = self._branch(payload, "power2") if checksum_ok else None
        with self.lock:
            if timestamp is not None:
                self._trim(timestamp)
            return self._snapshot_locked(control, motive, checksum_ok)

    def _snapshot_locked(self, control, motive, checksum_ok):
        voltages = [item[1] for item in self.samples]
        average_voltage = (
            sum(voltages) / len(voltages) if voltages else None
        )
        battery_valid = bool(
            checksum_ok and motive is not None and average_voltage is not None
        )
        cell_voltage = (
            average_voltage / self.series_count if battery_valid else None
        )
        soc = interpolate_battery_soc(cell_voltage) if battery_valid else None
        remaining_ah = (
            self.pack_capacity_ah * soc / 100.0 if soc is not None else None
        )
        both_valid = control is not None and motive is not None
        return {
            "valid": bool(checksum_ok and (control or motive)),
            "checksum_ok": checksum_ok,
            "control": copy.deepcopy(control),
            "motive": copy.deepcopy(motive),
            "battery_voltage_v": (
                motive["voltage_v"] if motive is not None else None
            ),
            "control_current_a": (
                control["current_a"] if control is not None else None
            ),
            "control_power_w": (
                control["power_w"] if control is not None else None
            ),
            "motive_current_a": (
                motive["current_a"] if motive is not None else None
            ),
            "motive_power_w": (
                motive["power_w"] if motive is not None else None
            ),
            "total_power_w": (
                control["power_w"] + motive["power_w"]
                if both_valid else None
            ),
            "battery": {
                "valid": battery_valid and soc is not None,
                "chemistry": "li_ion",
                "series_count": self.series_count,
                "parallel_count": self.parallel_count,
                "cell_capacity_ah": self.cell_capacity_ah,
                "pack_capacity_ah": self.pack_capacity_ah,
                "smoothing_sec": self.smoothing_sec,
                "sample_count": len(voltages),
                "smoothed_voltage_v": average_voltage,
                "cell_voltage_v": cell_voltage,
                "soc_percent": soc,
                "remaining_ah": remaining_ah,
                "estimated": True,
            },
        }


def map_pixel_to_frame(point, frame_width, frame_height,
                       source_width=None, source_height=None):
    """将检测图像中的像素点缩放并裁剪到当前相机帧。"""
    try:
        target_width = int(frame_width)
        target_height = int(frame_height)
    except (TypeError, ValueError):
        return None
    if target_width <= 0 or target_height <= 0:
        return None

    if isinstance(point, dict):
        u = safe_float(point.get("u", point.get("x")))
        v = safe_float(point.get("v", point.get("y")))
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        u = safe_float(point[0])
        v = safe_float(point[1])
    else:
        return None
    if u is None or v is None:
        return None

    source_width = safe_float(source_width)
    source_height = safe_float(source_height)
    if source_width is not None and source_width > 0:
        u *= target_width / source_width
    if source_height is not None and source_height > 0:
        v *= target_height / source_height

    return (
        max(0, min(target_width - 1, int(round(u)))),
        max(0, min(target_height - 1, int(round(v)))),
    )


def vision_packet_status(packet, now, timeout, frame_stamp=None,
                         frame_tolerance=None):
    """返回视觉结果的时效和与当前图像的同步状态。"""
    if not isinstance(packet, dict):
        return {
            "online": False,
            "age_sec": None,
            "frame_delta_sec": None,
            "frame_synced": False,
        }

    received_at = safe_float(packet.get("received_at"))
    status = health_state(received_at, now, timeout)
    payload = packet.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    payload_stamp = safe_float(payload.get("stamp"))
    current_stamp = safe_float(frame_stamp)
    tolerance = safe_float(frame_tolerance)

    frame_delta = None
    frame_synced = True
    if payload_stamp is not None and current_stamp is not None:
        frame_delta = abs(payload_stamp - current_stamp)
        if tolerance is not None and tolerance > 0.0:
            frame_synced = frame_delta <= tolerance

    status.update({
        "frame_delta_sec": frame_delta,
        "frame_synced": frame_synced,
    })
    status["online"] = bool(status["online"] and frame_synced)
    return status


def has_vision_detections(payload):
    """判断通用检测 JSON 是否包含至少一个检测结果。"""
    if not isinstance(payload, dict):
        return False
    if payload.get("valid") is False:
        return False
    detections = payload.get("detections")
    return isinstance(detections, list) and bool(detections)


def sanitize_json(value):
    """递归清理 JSON 中不能可靠传输的非有限浮点数。"""
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def normalize_heading(angle_deg):
    """将航向角归一化到 [0, 360)。"""
    value = safe_float(angle_deg)
    if value is None:
        return None
    return value % 360.0


def shortest_heading_error(target_deg, actual_deg):
    """计算目标减实际的最短航向误差，范围为 [-180, 180)。"""
    target = normalize_heading(target_deg)
    actual = normalize_heading(actual_deg)
    if target is None or actual is None:
        return None
    return (target - actual + 180.0) % 360.0 - 180.0


def quaternion_to_euler_deg(x, y, z, w):
    """将四元数转换为 Roll/Pitch/Yaw 角度。"""
    values = [safe_float(item) for item in (x, y, z, w)]
    if any(item is None for item in values):
        return None

    qx, qy, qz, qw = values
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        return None
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    sin_roll = 2.0 * (qw * qx + qy * qz)
    cos_roll = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (qw * qy - qz * qx)
    if abs(sin_pitch) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sin_pitch)
    else:
        pitch = math.asin(sin_pitch)

    sin_yaw = 2.0 * (qw * qz + qx * qy)
    cos_yaw = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(sin_yaw, cos_yaw)

    return {
        "roll_deg": math.degrees(roll),
        "pitch_deg": math.degrees(pitch),
        "yaw_deg": math.degrees(yaw),
        "heading_deg": normalize_heading(math.degrees(yaw)),
    }


def health_state(received_at, now, timeout):
    """根据墙钟接收时间计算数据年龄和在线状态。"""
    received = safe_float(received_at)
    current = safe_float(now)
    limit = safe_float(timeout)
    if received is None or current is None:
        return {
            "online": False,
            "age_sec": None,
            "timeout_sec": limit,
        }

    age = max(0.0, current - received)
    online = True if limit is None or limit <= 0.0 else age <= limit
    return {
        "online": online,
        "age_sec": age,
        "timeout_sec": limit,
    }


def update_fps(previous_fps, previous_received_at, now, alpha=0.1):
    """使用指数滑动平均更新图像帧率。"""
    current_time = safe_float(now)
    previous_time = safe_float(previous_received_at)
    old_fps = safe_float(previous_fps)
    smoothing = safe_float(alpha)
    if current_time is None or previous_time is None:
        return old_fps or 0.0
    delta = current_time - previous_time
    if delta <= 1e-6:
        return old_fps or 0.0

    instant_fps = 1.0 / delta
    if old_fps is None or old_fps <= 0.0:
        return instant_fps
    smoothing = 0.1 if smoothing is None else max(0.0, min(1.0, smoothing))
    return (1.0 - smoothing) * old_fps + smoothing * instant_fps


def select_attitude(feedback_candidate, tf_candidate):
    """选择有效反馈姿态，反馈无效时回退到 TF。"""
    if feedback_candidate and feedback_candidate.get("valid"):
        return feedback_candidate
    if tf_candidate and tf_candidate.get("valid"):
        return tf_candidate

    candidates = [
        item for item in (feedback_candidate, tf_candidate)
        if item is not None
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: item.get("received_at") or 0.0,
    )


def horizon_transform(roll_deg, pitch_deg, pixels_per_degree=2.0,
                      pitch_limit_deg=45.0):
    """返回人工地平图的背景旋转角和俯仰位移。"""
    roll = safe_float(roll_deg)
    pitch = safe_float(pitch_deg)
    scale = safe_float(pixels_per_degree)
    limit = safe_float(pitch_limit_deg)
    if None in (roll, pitch, scale, limit):
        return None

    clamped_pitch = max(-abs(limit), min(abs(limit), pitch))
    return {
        "rotation_deg": -roll,
        "offset_px": clamped_pitch * scale,
        "clamped_pitch_deg": clamped_pitch,
    }


def normalize_visual_label(value):
    """将模型类别名归一化为便于匹配的英文下划线形式。"""
    text = str(value or "").strip().lower()
    for separator in ("-", " ", "/"):
        text = text.replace(separator, "_")
    return "_".join(part for part in text.split("_") if part)


def vision_map_category(source, class_name):
    """将视觉任务和模型类别映射为地图历史类别。"""
    source = str(source or "").strip().lower()
    label = normalize_visual_label(class_name)
    if source == "red_circle":
        return "red_circle"
    if source == "line":
        return "red_line"
    if source == "arrow":
        return "arrow"
    if source == "shapes":
        if label == "rectangle":
            return "black_square"
        if label == "circle":
            return "yellow_circle"
        return None
    if source == "rectangle":
        parts = set(label.split("_"))
        for color in ("red", "yellow", "green"):
            if label == color or color in parts:
                return "rectangle_{}".format(color)
    return None


def finite_position(value):
    """校验并复制一个有限三维位置字典。"""
    if not isinstance(value, dict):
        return None
    values = [safe_float(value.get(axis)) for axis in ("x", "y", "z")]
    if any(item is None for item in values):
        return None
    return dict(zip(("x", "y", "z"), values))


def extract_vision_pose_points(source, payload):
    """从单点、固定三点或 LineN payload 中提取有序三维点。"""
    if not isinstance(payload, dict) or payload.get("valid") is not True:
        return []
    if str(source).strip().lower() != "line":
        point = finite_position(payload.get("position_m"))
        return [point] if point is not None else []

    positions = payload.get("positions_m")
    if isinstance(positions, list):
        return [
            point for point in (finite_position(item) for item in positions)
            if point is not None
        ]

    samples = payload.get("samples")
    if isinstance(samples, list):
        points = []
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("valid") is False:
                continue
            point = finite_position(sample.get("position_m"))
            if point is not None:
                points.append(point)
        return points
    return []


def transform_visual_geometry(points, direction, translation, quaternion):
    """使用刚体 TF 将相机点和可选方向向量转换到 map/NED。"""
    if not isinstance(points, list) or not points:
        raise ValueError("视觉点不能为空")
    if not isinstance(translation, (list, tuple)) or len(translation) != 3:
        raise ValueError("TF平移无效")
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise ValueError("TF四元数无效")
    translation_values = [safe_float(value) for value in translation]
    quaternion_values = [safe_float(value) for value in quaternion]
    if len(translation_values) != 3 or any(
            value is None for value in translation_values):
        raise ValueError("TF平移无效")
    if len(quaternion_values) != 4 or any(
            value is None for value in quaternion_values):
        raise ValueError("TF四元数无效")
    qx, qy, qz, qw = quaternion_values
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        raise ValueError("TF四元数长度为零")
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    rotation = (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )

    def rotate(vector):
        return [
            sum(rotation[row][column] * vector[column] for column in range(3))
            for row in range(3)
        ]

    map_points = []
    for point in points:
        finite = finite_position(point)
        if finite is None:
            raise ValueError("视觉点包含无效坐标")
        rotated = rotate([finite["x"], finite["y"], finite["z"]])
        world = [
            rotated[index] + translation_values[index]
            for index in range(3)
        ]
        map_points.append({
            "north_m": world[0],
            "east_m": world[1],
            "down_m": world[2],
        })

    map_direction = None
    if direction is not None:
        finite = finite_position(direction)
        if finite is None:
            raise ValueError("箭头方向包含无效坐标")
        rotated = rotate([finite["x"], finite["y"], finite["z"]])
        horizontal_length = math.hypot(rotated[0], rotated[1])
        if horizontal_length <= 1e-9:
            raise ValueError("箭头转换后的水平向量长度为零")
        north = rotated[0] / horizontal_length
        east = rotated[1] / horizontal_length
        map_direction = {
            "north": north,
            "east": east,
            "heading_deg": normalize_heading(
                math.degrees(math.atan2(east, north))
            ),
        }
    return map_points, map_direction


class VisionHistoryState:
    """线程安全维护地图视觉历史和鱼眼 ArUco 判色历史。"""

    def __init__(self, map_limit=20, aruco_window=10,
                 aruco_required_count=3, aruco_min_confidence=0.5):
        self.map_limit = max(1, int(map_limit))
        self.aruco_window_size = max(1, int(aruco_window))
        self.aruco_required_count = max(1, int(aruco_required_count))
        if self.aruco_required_count > self.aruco_window_size:
            raise ValueError("ArUco确认次数不能大于历史窗口")
        confidence = safe_float(aruco_min_confidence)
        if confidence is None or not 0.0 <= confidence <= 1.0:
            raise ValueError("ArUco最低置信度必须在0到1之间")
        self.aruco_min_confidence = confidence
        self.lock = threading.RLock()
        self.map_values = {
            category: deque(maxlen=self.map_limit)
            for category in VISION_MAP_CATEGORIES
        }
        self.map_seen_order = deque(maxlen=max(128, self.map_limit * 16))
        self.map_seen = set()
        self.aruco_values = deque(maxlen=self.aruco_window_size)
        self.aruco_seen_order = deque(maxlen=max(64, self.aruco_window_size * 8))
        self.aruco_seen = set()
        self.confirmed_marker_id = None
        self.expected_color = None
        self.version = 0
        self.cleared_at = None

    @staticmethod
    def _remember(key, order, values):
        if key in values:
            return False
        if order.maxlen and len(order) >= order.maxlen:
            values.discard(order[0])
        order.append(key)
        values.add(key)
        return True

    def append_map(self, category, record, dedupe_key):
        """向指定图形队列追加一条已经转换到 map 的记录。"""
        if category not in self.map_values or not isinstance(record, dict):
            return False
        with self.lock:
            if not self._remember(
                    str(dedupe_key), self.map_seen_order, self.map_seen):
                return False
            item = copy.deepcopy(record)
            item["category"] = category
            self.map_values[category].append(item)
            self.version += 1
            return True

    def append_aruco_payload(self, payload, received_at):
        """将一帧中的全部合法 ArUco 依次加入最近十次历史。"""
        if not isinstance(payload, dict):
            return 0
        stamp = safe_float(payload.get("stamp"))
        detections = payload.get("detections")
        if stamp is None or not isinstance(detections, list):
            return 0

        valid_items = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                continue
            marker_value = detection.get("marker_id", detection.get("class_id"))
            try:
                marker_id = int(marker_value)
            except (TypeError, ValueError):
                continue
            confidence = safe_float(detection.get("confidence"))
            if (
                    marker_id not in ARUCO_COLOR_BY_ID
                    or confidence is None
                    or confidence < self.aruco_min_confidence):
                continue
            valid_items.append({
                "marker_id": marker_id,
                "confidence": confidence,
                "stamp": stamp,
                "received_at": float(received_at),
                "source_index": index,
            })

        if not valid_items:
            return 0
        packet_key = "{:.9f}".format(stamp)
        with self.lock:
            if not self._remember(
                    packet_key, self.aruco_seen_order, self.aruco_seen):
                return 0
            for item in valid_items:
                self.aruco_values.append(item)
            counts = Counter(
                item["marker_id"] for item in self.aruco_values
            )
            candidate, count = min(
                counts.items(),
                key=lambda pair: (-pair[1], pair[0]),
            )
            if count >= self.aruco_required_count:
                self.confirmed_marker_id = candidate
                self.expected_color = ARUCO_COLOR_BY_ID[candidate]
            self.version += 1
            return len(valid_items)

    def clear_map(self, cleared_at=None):
        """仅清空依赖当前世界原点的地图视觉历史。"""
        with self.lock:
            for values in self.map_values.values():
                values.clear()
            self.map_seen_order.clear()
            self.map_seen.clear()
            self.version += 1
            self.cleared_at = cleared_at
            return self.version

    def clear_all(self, cleared_at=None):
        """原子清空地图、ArUco历史和已经锁存的期望颜色。"""
        with self.lock:
            for values in self.map_values.values():
                values.clear()
            self.map_seen_order.clear()
            self.map_seen.clear()
            self.aruco_values.clear()
            self.aruco_seen_order.clear()
            self.aruco_seen.clear()
            self.confirmed_marker_id = None
            self.expected_color = None
            self.version += 1
            self.cleared_at = cleared_at
            return self.version

    def snapshot(self, now, frame_id, origin_revision):
        """生成 /api/status 使用的地图与 ArUco 历史快照。"""
        current_time = safe_float(now)
        with self.lock:
            categories = {
                category: [copy.deepcopy(item) for item in values]
                for category, values in self.map_values.items()
            }
            aruco_items = []
            for item in reversed(self.aruco_values):
                output = copy.deepcopy(item)
                received_at = safe_float(output.get("received_at"))
                output["age_sec"] = (
                    None if current_time is None or received_at is None
                    else max(0.0, current_time - received_at)
                )
                aruco_items.append(output)
            counts = Counter(
                item["marker_id"] for item in self.aruco_values
            )
            confirmed_count = (
                counts.get(self.confirmed_marker_id, 0)
                if self.confirmed_marker_id is not None else 0
            )
            version = self.version
            cleared_at = self.cleared_at
            marker_id = self.confirmed_marker_id
            expected_color = self.expected_color

        return {
            "vision_map": {
                "frame_id": str(frame_id),
                "origin_revision": int(origin_revision),
                "history_version": version,
                "limit_per_category": self.map_limit,
                "categories": categories,
            },
            "aruco_history": {
                "history_version": version,
                "window_size": self.aruco_window_size,
                "required_count": self.aruco_required_count,
                "min_confidence": self.aruco_min_confidence,
                "items": aruco_items,
                "confirmed_marker_id": marker_id,
                "confirmed_count": confirmed_count,
                "expected_color": expected_color,
                "latched": marker_id is not None,
                "cleared_at": cleared_at,
            },
        }


class NavigationHistoryState:
    """线程安全维护 base_link 轨迹和最近目标位姿。"""

    def __init__(self, trajectory_hz=1.0, trajectory_duration_sec=1000.0,
                 target_limit=2, target_position_threshold_m=0.01,
                 target_heading_threshold_deg=1.0):
        frequency = safe_float(trajectory_hz)
        duration = safe_float(trajectory_duration_sec)
        position_threshold = safe_float(target_position_threshold_m)
        heading_threshold = safe_float(target_heading_threshold_deg)
        if frequency is None or frequency <= 0.0:
            raise ValueError("轨迹采样频率必须大于零")
        if duration is None or duration <= 0.0:
            raise ValueError("轨迹保留时间必须大于零")
        if position_threshold is None or position_threshold <= 0.0:
            raise ValueError("目标位置变化阈值必须大于零")
        if heading_threshold is None or heading_threshold <= 0.0:
            raise ValueError("目标航向变化阈值必须大于零")
        self.trajectory_hz = frequency
        self.trajectory_duration_sec = duration
        self.sample_period_sec = 1.0 / frequency
        self.trajectory_limit = max(1, int(math.ceil(frequency * duration)))
        self.target_limit = max(1, int(target_limit))
        self.target_position_threshold_m = position_threshold
        self.target_heading_threshold_deg = heading_threshold
        self.lock = threading.RLock()
        self.trajectory = deque(maxlen=self.trajectory_limit)
        self.targets = deque(maxlen=self.target_limit)
        self.trajectory_version = 0
        self.target_version = 0
        self.trajectory_cleared_at = None

    def append_base_pose(self, pose, sampled_at):
        """按配置频率保存一条有效 map 下 base_link 位姿。"""
        if not isinstance(pose, dict):
            return False
        position = finite_position(pose.get("position_m"))
        sample_time = safe_float(sampled_at)
        tf_stamp = safe_float(pose.get("stamp_sec"))
        if position is None or sample_time is None:
            return False
        with self.lock:
            if self.trajectory:
                previous = self.trajectory[-1]
                if (
                        sample_time - previous["sampled_at"]
                        < self.sample_period_sec):
                    return False
                previous_stamp = safe_float(previous.get("stamp_sec"))
                if (
                        tf_stamp is not None
                        and previous_stamp is not None
                        and tf_stamp == previous_stamp):
                    return False
            self.trajectory.append({
                "north_m": position["x"],
                "east_m": position["y"],
                "down_m": position["z"],
                "stamp_sec": tf_stamp,
                "sampled_at": sample_time,
            })
            self.trajectory_version += 1
            return True

    def append_target(self, target, received_at):
        """目标位置或航向达到阈值时，滚动保存最近两帧。"""
        if not isinstance(target, dict):
            return False
        position = finite_position(target.get("position_m"))
        sample_time = safe_float(received_at)
        if position is None or sample_time is None:
            return False
        item = copy.deepcopy(target)
        item["received_at"] = sample_time
        with self.lock:
            if self.targets:
                previous = self.targets[-1]
                previous_position = finite_position(
                    previous.get("position_m")
                )
                position_changed = any(
                    abs(position[axis] - previous_position[axis])
                    >= self.target_position_threshold_m
                    for axis in ("x", "y", "z")
                )
                heading = safe_float(
                    (target.get("orientation_deg") or {}).get(
                        "heading_deg"
                    )
                )
                previous_heading = safe_float(
                    (previous.get("orientation_deg") or {}).get(
                        "heading_deg"
                    )
                )
                heading_error = shortest_heading_error(
                    heading,
                    previous_heading,
                )
                heading_changed = (
                    heading_error is not None
                    and abs(heading_error)
                    >= self.target_heading_threshold_deg
                )
                if not position_changed and not heading_changed:
                    # 只刷新时间；保留已接纳目标作为累计变化的比较基准。
                    previous["received_at"] = sample_time
                    if item.get("stamp_sec") is not None:
                        previous["stamp_sec"] = item["stamp_sec"]
                    return False
            self.targets.append(item)
            self.target_version += 1
            return True

    def clear_trajectory(self, cleared_at=None):
        """只清空 base_link 轨迹，不影响目标与视觉历史。"""
        with self.lock:
            self.trajectory.clear()
            self.trajectory_version += 1
            self.trajectory_cleared_at = safe_float(cleared_at)
            return self.trajectory_version

    def snapshot(self, now, frame_id):
        """生成地图轨迹和目标最近两帧的状态快照。"""
        current_time = safe_float(now)
        with self.lock:
            trajectory = []
            for item in self.trajectory:
                sampled_at = safe_float(item.get("sampled_at"))
                if (
                        current_time is not None
                        and sampled_at is not None
                        and current_time - sampled_at
                        > self.trajectory_duration_sec):
                    continue
                output = copy.deepcopy(item)
                output["age_sec"] = (
                    None if current_time is None or sampled_at is None
                    else max(0.0, current_time - sampled_at)
                )
                trajectory.append(output)
            targets = []
            for item in reversed(self.targets):
                output = copy.deepcopy(item)
                received_at = safe_float(output.get("received_at"))
                output["age_sec"] = (
                    None if current_time is None or received_at is None
                    else max(0.0, current_time - received_at)
                )
                targets.append(output)
            trajectory_version = self.trajectory_version
            target_version = self.target_version
            cleared_at = self.trajectory_cleared_at

        return {
            "base_trajectory": {
                "frame_id": str(frame_id),
                "sample_hz": self.trajectory_hz,
                "duration_sec": self.trajectory_duration_sec,
                "limit": self.trajectory_limit,
                "history_version": trajectory_version,
                "cleared_at": cleared_at,
                "points": trajectory,
            },
            "target_history": {
                "frame_id": str(frame_id),
                "limit": self.target_limit,
                "position_change_threshold_m": (
                    self.target_position_threshold_m
                ),
                "heading_change_threshold_deg": (
                    self.target_heading_threshold_deg
                ),
                "history_version": target_version,
                "items": targets,
            },
        }


class OriginRevision:
    """维护世界原点数值和递增版本号。"""

    def __init__(self, epsilon=1e-9):
        self.epsilon = float(epsilon)
        self.values = None
        self.revision = 0
        self.lock = threading.Lock()

    def update(self, latitude, longitude, depth):
        """原点首次出现或发生变化时递增版本号。"""
        values = tuple(safe_float(item) for item in (
            latitude, longitude, depth
        ))
        if any(item is None for item in values):
            return False, self.revision

        with self.lock:
            changed = (
                self.values is None
                or any(
                    abs(current - previous) > self.epsilon
                    for current, previous in zip(values, self.values)
                )
            )
            if changed:
                self.values = values
                self.revision += 1
            return changed, self.revision

    def snapshot(self):
        """返回当前原点与版本号。"""
        with self.lock:
            return self.values, self.revision
