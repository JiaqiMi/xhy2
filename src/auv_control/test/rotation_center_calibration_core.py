#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：rotation_center_calibration_core.py
功能：提供自由偏航控制和正负计划方向独立旋转中心的最小二乘拟合
作者：BroXu
监听：无
发布：无
说明：
    每个自由旋转段允许具有不同的 map 圆心；同一计划方向的各段共享一个
    control_link -> imu 机体系杆臂，正、负计划方向分别拟合，从而同时排除
    段间 mode=4 接管平移和正反桨差异的影响。
记录：
2026.7.18
    新增直接 MZ 转向控制、角度展开和旋转中心稳健拟合算法。
2026.7.18
    增加分阶段远离目标判断，目标尚未越过时保留符号保护，刹转越过目标后
    不再误报方向错误。
2026.7.18
    增加固定初始位置接管的稳定条件判断。
2026.7.19
    正负计划旋转方向分别拟合独立旋转中心，不再强制共享同一杆臂。
2026.7.19
    增加低、中、高三档不对称 MZ 配置和分档结果推荐算法。
2026.7.19
    将旋转漂移半径改为告警判定，不再因平移漂移中止 90° 旋转。
2026.7.19
    固定档位 MZ 仅用于首次接近目标；越过目标后锁存 RECOVER，使用闭环指令刹停。
2026.7.19
    增加连续稳定计时；方向保护读取目标越过锁存，避免恢复阶段误报方向错误。
"""

from __future__ import division

import math
from collections import namedtuple

import numpy as np


CalibrationSample = namedtuple(
    'CalibrationSample',
    ('segment', 'x', 'y', 'yaw'),
)
TorqueProfile = namedtuple(
    'TorqueProfile',
    ('name', 'positive_limit', 'negative_limit'),
)
ContinuousRotationStep = namedtuple(
    'ContinuousRotationStep',
    ('segment', 'profile', 'direction', 'turns', 'turn_angle'),
)


def build_torque_profiles(
        positive_levels, negative_scale=1.5,
        names=('low', 'medium', 'high')):
    """构造低、中、高三档正负不对称 MZ 限幅。"""
    levels = tuple(float(value) for value in positive_levels)
    profile_names = tuple(str(value).strip() for value in names)
    scale = float(negative_scale)
    if len(levels) != 3 or len(profile_names) != 3:
        raise ValueError('旋转中心标定必须配置低、中、高三个力矩档位')
    if not all(
            math.isfinite(value) and 0.0 < value <= 10000.0
            for value in levels):
        raise ValueError('正向力矩档位必须在 (0, 10000] 内')
    if not levels[0] < levels[1] < levels[2]:
        raise ValueError('正向力矩档位必须严格递增')
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError('负向力矩倍率必须为有限正数')
    negative_levels = tuple(value * scale for value in levels)
    if any(value > 10000.0 for value in negative_levels):
        raise ValueError('负向力矩档位乘倍率后不能超过协议上限 10000')
    if any(not name for name in profile_names):
        raise ValueError('力矩档位名称不能为空')
    return tuple(
        TorqueProfile(name, positive, negative)
        for name, positive, negative in zip(
            profile_names, levels, negative_levels)
    )


def build_continuous_rotation_steps(torque_profiles, turns=3):
    """构造低中高三档、正负方向各一次的连续多圈旋转序列。"""
    profiles = tuple(torque_profiles)
    turns = int(turns)
    if not profiles:
        raise ValueError('连续旋转标定至少需要一个力矩档位')
    if turns <= 0:
        raise ValueError('连续旋转圈数必须大于 0')
    if any(not isinstance(profile, TorqueProfile) for profile in profiles):
        raise TypeError('连续旋转标定档位类型不合法')

    steps = []
    for profile in profiles:
        for direction in (1, -1):
            steps.append(ContinuousRotationStep(
                len(steps) + 1,
                profile,
                direction,
                turns,
                2.0 * math.pi * turns,
            ))
    return tuple(steps)


def fit_quality_score(result):
    """用稳健 RMS 和最大残差评价单方向旋转中心拟合质量。"""
    rms = float(result['rms_residual'])
    maximum = float(result['max_residual'])
    if not all(math.isfinite(value) and value >= 0.0
               for value in (rms, maximum)):
        raise ValueError('旋转中心拟合质量必须为有限非负数')
    return rms + 0.25 * maximum


def select_recommended_profile(
        profile_results, preferred='medium',
        degradation_ratio=1.5, degradation_margin=0.01):
    """优先中速；中速质量明显劣化时选择评分最低的档位。"""
    results = dict(profile_results)
    preferred = str(preferred)
    ratio = float(degradation_ratio)
    margin = float(degradation_margin)
    if not results:
        raise ValueError('至少需要一个力矩档位结果')
    if preferred not in results:
        raise ValueError('首选力矩档位不存在: {}'.format(preferred))
    if (
            not math.isfinite(ratio)
            or ratio < 1.0
            or not math.isfinite(margin)
            or margin < 0.0):
        raise ValueError('推荐档位劣化阈值不合法')
    scores = {
        name: fit_quality_score(result)
        for name, result in results.items()
    }
    best = min(scores, key=lambda name: (scores[name], name))
    preferred_limit = scores[best] * ratio + margin
    return preferred if scores[preferred] <= preferred_limit else best


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


def fixed_track_command(
        direction, mz_to_yaw_sign,
        positive_limit, negative_limit):
    """按计划 yaw 方向输出当前档位的固定 TRACK MZ。"""
    values = (
        direction,
        mz_to_yaw_sign,
        positive_limit,
        negative_limit,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('固定 TRACK MZ 参数必须为有限值')
    if abs(float(direction)) < 1e-9:
        raise ValueError('计划旋转方向不能为 0')
    if abs(float(mz_to_yaw_sign)) < 1e-9:
        raise ValueError('mz_to_yaw_sign 不能为 0')
    if min(float(positive_limit), float(negative_limit)) <= 0.0:
        raise ValueError('固定 TRACK MZ 限幅必须为正数')
    command_sign = (
        1.0
        if float(direction) * float(mz_to_yaw_sign) > 0.0
        else -1.0
    )
    magnitude = (
        float(positive_limit)
        if command_sign > 0.0
        else float(negative_limit)
    )
    return int(round(command_sign * magnitude))


def apply_fixed_track_policy(
        command_mz,
        controller_phase,
        yaw_error,
        direction,
        target_crossed,
        mz_to_yaw_sign,
        positive_limit,
        negative_limit):
    """首次接近目标时使用固定 MZ，越过目标后保留闭环恢复指令。"""
    values = (
        command_mz,
        yaw_error,
        direction,
        mz_to_yaw_sign,
        positive_limit,
        negative_limit,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('固定 TRACK 策略参数必须为有限值')
    if abs(float(direction)) < 1e-9:
        raise ValueError('计划旋转方向不能为 0')

    crossed = (
        bool(target_crossed)
        or float(direction) * float(yaw_error) <= 0.0
    )
    phase = str(controller_phase)
    if phase != 'TRACK':
        return int(round(float(command_mz))), phase, crossed
    if crossed:
        return int(round(float(command_mz))), 'RECOVER', True
    return (
        fixed_track_command(
            direction,
            mz_to_yaw_sign,
            positive_limit,
            negative_limit,
        ),
        'TRACK',
        False,
    )


def rotation_is_unexpectedly_moving_away(
        yaw_error, yaw_rate, direction, yaw_rate_threshold,
        target_crossed=False):
    """判断首次越过目标前是否持续向目标反方向旋转。"""
    values = (yaw_error, yaw_rate, direction, yaw_rate_threshold)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('远离目标判断参数必须为有限值')
    if abs(float(direction)) < 1e-9:
        raise ValueError('旋转方向不能为 0')
    return (
        not bool(target_crossed)
        and float(direction) * float(yaw_error) > 0.0
        and float(yaw_error) * float(yaw_rate) < 0.0
        and abs(float(yaw_rate)) > float(yaw_rate_threshold)
    )


def drift_exceeds_warning_threshold(drift, warning_radius):
    """判断自由旋转漂移是否超过只用于记录的告警半径。"""
    values = (drift, warning_radius)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('旋转漂移告警参数必须为有限值')
    if float(drift) < 0.0 or float(warning_radius) <= 0.0:
        raise ValueError('旋转漂移必须非负，告警半径必须为正数')
    return float(drift) > float(warning_radius)


def update_continuous_stability(
        stable,
        stable_count,
        stable_started_at,
        now,
        required_frames):
    """更新连续判稳帧数和判稳后的连续保持时间。"""
    count = int(stable_count)
    frame_limit = int(required_frames)
    now = float(now)
    if count < 0 or frame_limit <= 0:
        raise ValueError('连续判稳帧数必须合法')
    if not math.isfinite(now):
        raise ValueError('连续判稳时间必须为有限值')
    if stable_started_at is not None:
        stable_started_at = float(stable_started_at)
        if not math.isfinite(stable_started_at):
            raise ValueError('连续判稳起始时间必须为有限值')

    if not bool(stable):
        return 0, None, 0.0

    count += 1
    if count >= frame_limit and stable_started_at is None:
        stable_started_at = now
    elapsed = (
        0.0
        if stable_started_at is None
        else max(0.0, now - stable_started_at)
    )
    return count, stable_started_at, elapsed


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
    """稳健拟合同一计划方向共享的杆臂和每段 map 圆心。"""
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


def fit_planned_direction_centers(
        positive_samples, negative_samples, outlier_sigma=3.5):
    """分别拟合正向和负向计划旋转对应的 control_link -> imu 杆臂。"""
    return {
        'positive': fit_segmented_rotation_center(
            positive_samples, outlier_sigma=outlier_sigma),
        'negative': fit_segmented_rotation_center(
            negative_samples, outlier_sigma=outlier_sigma),
    }
