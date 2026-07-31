#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_depth_auto_test_core.py
功能：生成绝对深度阶跃序列并提供基于 TF 的深度稳定性计算
作者：BroXu
监听：无
发布：无
记录：
2026.7.31
    新增绝对深度目标、启动基准深度往返序列及 TF 垂向速度和超调计算。
"""

from __future__ import division

import math
from collections import namedtuple


DepthSequenceStep = namedtuple(
    'DepthSequenceStep',
    ('index', 'cycle', 'phase', 'requested_depth', 'target_depth'),
)


def _finite_values(values, name):
    converted = tuple(float(value) for value in values)
    if not converted:
        raise ValueError('{} 不能为空'.format(name))
    if not all(math.isfinite(value) for value in converted):
        raise ValueError('{} 必须全部为有限值'.format(name))
    return converted


def build_depth_sequence(target_depths, baseline_depth, cycle_count=2):
    """生成“绝对目标深度、启动基准深度”的循环往返序列。"""
    targets = _finite_values(target_depths, 'target_depths')
    baseline_depth = float(baseline_depth)
    cycle_count = int(cycle_count)
    if not math.isfinite(baseline_depth):
        raise ValueError('baseline_depth 必须为有限值')
    if cycle_count <= 0:
        raise ValueError('cycle_count 必须大于 0')

    steps = []
    for cycle in range(1, cycle_count + 1):
        for target_depth in targets:
            steps.append(DepthSequenceStep(
                len(steps) + 1,
                cycle,
                'target',
                target_depth,
                target_depth,
            ))
            steps.append(DepthSequenceStep(
                len(steps) + 1,
                cycle,
                'return_to_baseline',
                target_depth,
                baseline_depth,
            ))
    return steps


def linear_vertical_speed(samples):
    """用时间窗口内的 TF 深度做最小二乘拟合，返回垂向速度。"""
    if len(samples) < 2:
        raise ValueError('计算垂向速度至少需要两个 TF 样本')
    converted = tuple(
        (float(sample[0]), float(sample[1])) for sample in samples)
    if not all(
            math.isfinite(stamp) and math.isfinite(depth)
            for stamp, depth in converted):
        raise ValueError('TF 时间和深度必须为有限值')

    mean_time = sum(sample[0] for sample in converted) / len(converted)
    mean_depth = sum(sample[1] for sample in converted) / len(converted)
    time_variance = sum(
        (sample[0] - mean_time) ** 2 for sample in converted)
    if time_variance <= 0.0:
        raise ValueError('TF 样本时间必须递增')
    covariance = sum(
        (stamp - mean_time) * (depth - mean_depth)
        for stamp, depth in converted)
    return covariance / time_variance


def depth_motion_is_stable(
        current_depth, target_depth, vertical_speed,
        depth_tolerance, speed_threshold):
    """判断深度误差和 TF 垂向速度是否同时满足稳定条件。"""
    values = (
        current_depth,
        target_depth,
        vertical_speed,
        depth_tolerance,
        speed_threshold,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('深度稳定性参数必须为有限值')
    if float(depth_tolerance) <= 0.0 or float(speed_threshold) <= 0.0:
        raise ValueError('深度容差和速度阈值必须大于 0')
    return (
        abs(float(current_depth) - float(target_depth))
        <= float(depth_tolerance)
        and abs(float(vertical_speed)) <= float(speed_threshold)
    )


def directed_depth_overshoot(
        start_depth, target_depth, minimum_depth, maximum_depth):
    """按本步运动方向计算越过目标后的最大深度超调量。"""
    values = (
        start_depth,
        target_depth,
        minimum_depth,
        maximum_depth,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('深度超调参数必须为有限值')
    start_depth = float(start_depth)
    target_depth = float(target_depth)
    if target_depth < start_depth:
        return max(0.0, target_depth - float(minimum_depth))
    if target_depth > start_depth:
        return max(0.0, float(maximum_depth) - target_depth)
    return 0.0

