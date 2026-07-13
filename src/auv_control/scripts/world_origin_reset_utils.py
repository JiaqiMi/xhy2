#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：world_origin_reset_utils.py
功能：提供红色圆形原点重置的目标校验与鲁棒位置估计
作者：buyegaid
订阅：无
发布：无
记录：
2026.7.13
    新增中位数与 MAD 离群值剔除工具，供 world_origin_reset.py 调用。
"""

import numpy as np


def is_matching_target(class_name, target_type, confidence, expected_class, min_confidence):
    """检查感知消息是否是满足置信度要求的目标类别。"""
    try:
        return (
            class_name == expected_class
            and target_type == "center"
            and np.isfinite(float(confidence))
            and float(confidence) >= min_confidence
        )
    except (TypeError, ValueError):
        return False


def is_valid_camera_point(point, min_depth, max_depth):
    """检查相机坐标点是否有限且处于允许的深度范围。"""
    values = np.asarray(point, dtype=float)
    return (
        values.shape == (3,)
        and np.all(np.isfinite(values))
        and min_depth <= values[2] <= max_depth
    )


class RobustPointEstimator:
    """以滑动窗口、中位数和 MAD 估计稳定的三维目标位置。"""

    def __init__(self, sample_count, min_inliers, max_spread, min_inlier_radius):
        if sample_count <= 0:
            raise ValueError("sample_count 必须大于 0")
        if not 0 < min_inliers <= sample_count:
            raise ValueError("min_inliers 必须位于 (0, sample_count] 范围内")
        self.sample_count = int(sample_count)
        self.min_inliers = int(min_inliers)
        self.max_spread = float(max_spread)
        self.min_inlier_radius = float(min_inlier_radius)
        self._points = []

    @property
    def sample_size(self):
        """返回当前滑动窗口中的样本数。"""
        return len(self._points)

    def add(self, point):
        """加入样本；仅在当前窗口形成稳定簇时返回三维中位数。"""
        values = np.asarray(point, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            return None
        self._points.append(values)
        if len(self._points) > self.sample_count:
            self._points.pop(0)
        if len(self._points) < self.sample_count:
            return None

        points = np.asarray(self._points)
        median = np.median(points, axis=0)
        distances = np.linalg.norm(points - median, axis=1)
        median_distance = np.median(distances)
        mad = np.median(np.abs(distances - median_distance))
        inlier_radius = max(
            self.min_inlier_radius,
            median_distance + 3.0 * mad,
        )
        inliers = points[distances <= inlier_radius]
        if len(inliers) < self.min_inliers:
            return None

        center = np.median(inliers, axis=0)
        inlier_distances = np.linalg.norm(inliers - center, axis=1)
        if np.max(inlier_distances) > self.max_spread:
            return None
        return center
