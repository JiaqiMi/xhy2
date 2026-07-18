#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_auto_sequence_core.py
功能：生成运动管理器单轴自动测试序列并完成目标坐标换算
作者：BroXu
监听：无
发布：无
说明：
    1. X、Y 偏置均相对测试开始时锁定的 base_link 坐标系；
    2. Yaw 偏置相对测试开始时锁定的初始航向；
    3. 每个幅值按“正向、原点、负向、原点”排列。
记录：
2026.7.18
    新增 X、Y、Yaw 单轴自动往返测试的纯算法工具。
"""

from __future__ import division

import math
from collections import namedtuple


SUPPORTED_AXES = ('x', 'y', 'yaw')
SequenceStep = namedtuple(
    'SequenceStep',
    ('index', 'axis', 'magnitude', 'repetition', 'phase', 'offset'),
)


def wrap_angle(angle):
    """把角度归一化到 [-pi, pi)。"""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _positive_finite(values, name):
    converted = tuple(float(value) for value in values)
    if not converted:
        raise ValueError('{} 不能为空'.format(name))
    if not all(math.isfinite(value) and value > 0.0 for value in converted):
        raise ValueError('{} 必须全部为有限正数'.format(name))
    return converted


def build_axis_sequence(axis, magnitudes, repetitions=3):
    """生成正向、原点、负向、原点的重复测试序列。"""
    axis = str(axis).strip().lower()
    if axis not in SUPPORTED_AXES:
        raise ValueError('axis 仅支持 {}'.format(', '.join(SUPPORTED_AXES)))
    values = _positive_finite(magnitudes, 'magnitudes')
    repetitions = int(repetitions)
    if repetitions <= 0:
        raise ValueError('repetitions 必须大于 0')

    steps = []
    phases = (
        ('positive', 1.0),
        ('return_after_positive', 0.0),
        ('negative', -1.0),
        ('return_after_negative', 0.0),
    )
    for magnitude in values:
        for repetition in range(1, repetitions + 1):
            for phase, sign in phases:
                steps.append(SequenceStep(
                    len(steps) + 1,
                    axis,
                    magnitude,
                    repetition,
                    phase,
                    sign * magnitude,
                ))
    return steps


def relative_goal(start_x, start_y, start_yaw, target_z, axis, offset):
    """由固定初始 base_link 位姿生成 map 下的绝对目标。"""
    values = (start_x, start_y, start_yaw, target_z, offset)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('初始位姿、深度和偏置必须为有限值')
    axis = str(axis).strip().lower()
    if axis not in SUPPORTED_AXES:
        raise ValueError('axis 仅支持 {}'.format(', '.join(SUPPORTED_AXES)))

    x = float(start_x)
    y = float(start_y)
    yaw = float(start_yaw)
    offset = float(offset)
    if axis == 'x':
        x += math.cos(yaw) * offset
        y += math.sin(yaw) * offset
    elif axis == 'y':
        x -= math.sin(yaw) * offset
        y += math.cos(yaw) * offset
    else:
        yaw = wrap_angle(yaw + offset)
    return x, y, float(target_z), yaw


def goal_matches(
        actual_x, actual_y, actual_z, actual_yaw,
        expected_x, expected_y, expected_z, expected_yaw,
        position_tolerance, depth_tolerance, yaw_tolerance):
    """判断状态反馈中的目标是否对应当前测试动作。"""
    return (
        math.hypot(
            float(actual_x) - float(expected_x),
            float(actual_y) - float(expected_y),
        ) <= float(position_tolerance)
        and abs(float(actual_z) - float(expected_z))
        <= float(depth_tolerance)
        and abs(wrap_angle(float(actual_yaw) - float(expected_yaw)))
        <= float(yaw_tolerance)
    )
