#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-
"""
名称：state_web_core.py
功能：state_web 使用的纯 Python 数据处理工具
记录：
2026.7.17
    新增角度归一化、姿态换算、话题健康状态和原点版本管理。
"""

import math
import threading


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


def safe_float(value):
    """将数值转换为有限浮点数，无效值返回 None。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


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
