#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：rotation_center_calibration_core.py
功能：提供自由偏航控制和分段共享杆臂的旋转中心最小二乘拟合
作者：BroXu
监听：无
发布：无
说明：
    每个自由旋转段允许具有不同的 map 圆心，但所有段共享同一个
    control_link -> imu 机体系杆臂，从而排除段间 mode=4 接管平移的影响。
记录：
2026.7.18
    新增直接 MZ 转向控制、角度展开和旋转中心稳健拟合算法。
2026.7.18
    增加分阶段远离目标判断，目标尚未越过时保留符号保护，刹转越过目标后
    不再误报方向错误。
2026.7.18
    增加固定初始位置接管的稳定条件判断。
"""

from __future__ import division

import math
from collections import namedtuple

import numpy as np


CalibrationSample = namedtuple(
    'CalibrationSample',
    ('segment', 'x', 'y', 'yaw'),
)


def wrap_angle(angle):
    """把角度归一化到 [-pi, pi)。"""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_angle(previous_unwrapped, current_wrapped):
    """根据上一展开角度连续展开当前航向。"""
    previous_unwrapped = float(previous_unwrapped)
    return previous_unwrapped + wrap_angle(
        float(current_wrapped) - wrap_angle(previous_unwrapped))


def clamp_directional(value, positive_limit, negative_limit):
    """按实际 MZ 正负号应用不对称限幅。"""
    return max(
        -float(negative_limit),
        min(float(positive_limit), float(value)),
    )


def rotation_is_unexpectedly_moving_away(
        yaw_error, yaw_rate, direction, yaw_rate_threshold):
    """判断目标尚未越过时是否持续向目标反方向旋转。"""
    values = (yaw_error, yaw_rate, direction, yaw_rate_threshold)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('远离目标判断参数必须为有限值')
    if abs(float(direction)) < 1e-9:
        raise ValueError('旋转方向不能为 0')
    return (
        float(direction) * float(yaw_error) > 0.0
        and float(yaw_error) * float(yaw_rate) < 0.0
        and abs(float(yaw_rate)) > float(yaw_rate_threshold)
    )


def locked_handover_is_stable(
        reported_mode,
        position_error,
        horizontal_speed,
        yaw_error,
        yaw_rate,
        required_mode,
        position_tolerance,
        speed_threshold,
        yaw_tolerance,
        yaw_rate_threshold):
    """判断固定初始位置的闭环接管是否已经连续稳定。"""
    values = (
        position_error,
        horizontal_speed,
        yaw_error,
        yaw_rate,
        position_tolerance,
        speed_threshold,
        yaw_tolerance,
        yaw_rate_threshold,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('接管稳定条件必须为有限值')
    if min(
            float(position_tolerance),
            float(speed_threshold),
            float(yaw_tolerance),
            float(yaw_rate_threshold)) < 0.0:
        raise ValueError('接管稳定阈值不能为负数')
    return (
        int(reported_mode) == int(required_mode)
        and abs(float(position_error)) <= float(position_tolerance)
        and abs(float(horizontal_speed)) <= float(speed_threshold)
        and abs(float(yaw_error)) <= float(yaw_tolerance)
        and abs(float(yaw_rate)) <= float(yaw_rate_threshold)
    )


def direct_yaw_command(
        yaw_error,
        yaw_rate,
        kp,
        kd,
        brake_gain,
        track_positive_limit,
        track_negative_limit,
        brake_positive_limit,
        brake_negative_limit,
        brake_acceleration_positive,
        brake_acceleration_negative,
        control_delay,
        brake_margin,
        yaw_tolerance,
        yaw_rate_threshold,
        minimum_brake_mz,
        mz_to_yaw_sign=1.0):
    """计算自由转向阶段的 MZ、子阶段和停车角阈值。"""
    values = (
        yaw_error,
        yaw_rate,
        kp,
        kd,
        brake_gain,
        track_positive_limit,
        track_negative_limit,
        brake_positive_limit,
        brake_negative_limit,
        brake_acceleration_positive,
        brake_acceleration_negative,
        control_delay,
        brake_margin,
        yaw_tolerance,
        yaw_rate_threshold,
        minimum_brake_mz,
        mz_to_yaw_sign,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('直接转向控制参数必须为有限值')
    if abs(mz_to_yaw_sign) < 1e-9:
        raise ValueError('mz_to_yaw_sign 不能为 0')

    yaw_error = float(yaw_error)
    yaw_rate = float(yaw_rate)
    brake_candidate = (
        -float(brake_gain) * yaw_rate / float(mz_to_yaw_sign))
    acceleration = (
        float(brake_acceleration_positive)
        if brake_candidate >= 0.0
        else float(brake_acceleration_negative)
    )
    if acceleration <= 0.0:
        raise ValueError('有效角减速度必须大于 0')
    stopping_angle = (
        yaw_rate * yaw_rate / (2.0 * acceleration)
        + abs(yaw_rate) * float(control_delay)
        + float(brake_margin)
    )
    stable = (
        abs(yaw_error) <= float(yaw_tolerance)
        and abs(yaw_rate) <= float(yaw_rate_threshold)
    )
    if stable:
        return 0, 'HOLD', stopping_angle

    moving_toward = yaw_error * yaw_rate > 0.0
    moving_away = yaw_error * yaw_rate < 0.0
    braking = (
        (moving_toward and abs(yaw_error) <= stopping_angle)
        or (moving_away and abs(yaw_rate) > float(yaw_rate_threshold))
        or (
            abs(yaw_error) <= float(yaw_tolerance)
            and abs(yaw_rate) > float(yaw_rate_threshold)
        )
    )
    if braking:
        command = brake_candidate
        if (
                abs(yaw_rate) > float(yaw_rate_threshold)
                and abs(command) < float(minimum_brake_mz)):
            command = math.copysign(float(minimum_brake_mz), command)
        command = clamp_directional(
            command,
            brake_positive_limit,
            brake_negative_limit,
        )
        return int(round(command)), 'BRAKE', stopping_angle

    command = (
        float(kp) * yaw_error - float(kd) * yaw_rate
    ) / float(mz_to_yaw_sign)
    command = clamp_directional(
        command,
        track_positive_limit,
        track_negative_limit,
    )
    return int(round(command)), 'TRACK', stopping_angle


def _build_fit(samples):
    segments = sorted(set(int(sample.segment) for sample in samples))
    segment_index = {
        segment: index for index, segment in enumerate(segments)}
    matrix = np.zeros((2 * len(samples), 2 * len(segments) + 2))
    vector = np.zeros(2 * len(samples))
    offset_column = 2 * len(segments)
    for sample_index, sample in enumerate(samples):
        center_column = 2 * segment_index[int(sample.segment)]
        cosine = math.cos(float(sample.yaw))
        sine = math.sin(float(sample.yaw))
        row_x = 2 * sample_index
        row_y = row_x + 1
        matrix[row_x, center_column] = 1.0
        matrix[row_y, center_column + 1] = 1.0
        matrix[row_x, offset_column] = cosine
        matrix[row_x, offset_column + 1] = -sine
        matrix[row_y, offset_column] = sine
        matrix[row_y, offset_column + 1] = cosine
        vector[row_x] = float(sample.x)
        vector[row_y] = float(sample.y)
    solution, unused_residuals, rank, unused_singular = np.linalg.lstsq(
        matrix, vector, rcond=None)
    predicted = matrix.dot(solution)
    point_residuals = np.hypot(
        predicted[0::2] - vector[0::2],
        predicted[1::2] - vector[1::2],
    )
    centers = {
        segment: (
            float(solution[2 * index]),
            float(solution[2 * index + 1]),
        )
        for segment, index in segment_index.items()
    }
    return {
        'segments': segments,
        'centers': centers,
        'offset_x': float(solution[offset_column]),
        'offset_y': float(solution[offset_column + 1]),
        'rank': int(rank),
        'required_rank': int(matrix.shape[1]),
        'point_residuals': point_residuals,
    }


def fit_segmented_rotation_center(samples, outlier_sigma=3.5):
    """稳健拟合共享 control_link -> imu 杆臂和每段 map 圆心。"""
    converted = [
        CalibrationSample(
            int(sample.segment),
            float(sample.x),
            float(sample.y),
            float(sample.yaw),
        )
        for sample in samples
    ]
    if len(converted) < 8:
        raise ValueError('旋转中心拟合至少需要 8 个样本')
    if not all(
            math.isfinite(value)
            for sample in converted
            for value in (sample.x, sample.y, sample.yaw)):
        raise ValueError('旋转中心样本包含非有限值')

    first = _build_fit(converted)
    if first['rank'] < first['required_rank']:
        raise ValueError('旋转中心样本角度变化不足，拟合矩阵秩不足')
    residuals = first['point_residuals']
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    robust_sigma = 1.4826 * mad
    threshold = max(0.01, median + float(outlier_sigma) * robust_sigma)
    kept = [
        sample
        for sample, residual in zip(converted, residuals)
        if residual <= threshold
    ]
    minimum_kept = max(8, first['required_rank'] + 2)
    result = (
        _build_fit(kept)
        if len(kept) >= minimum_kept and len(kept) < len(converted)
        else first
    )
    if result['rank'] < result['required_rank']:
        result = first
        kept = converted
    final_residuals = result['point_residuals']
    result.update({
        'sample_count': len(converted),
        'used_sample_count': len(kept),
        'rejected_sample_count': len(converted) - len(kept),
        'rms_residual': float(math.sqrt(np.mean(final_residuals ** 2))),
        'max_residual': float(np.max(final_residuals)),
        'outlier_threshold': threshold,
    })
    return result
