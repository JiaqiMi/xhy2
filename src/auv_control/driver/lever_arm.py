#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：lever_arm.py
功能：提供导航传感器与 base_link 之间的刚体杆臂位置换算
作者：BroXu
监听：无
发布：无
说明：
    1. 偏置统一定义为 base_link 原点指向导航传感器原点；
    2. 位置和偏置均采用前、右、下的 NED/base_link 轴约定；
    3. 姿态四元数表示 base_link 到 map/NED 的旋转。
记录：
2026.7.18
    新增 IMU/GNSS 定位点与前移 base_link 之间的双向杆臂补偿。
2026.7.18
    增加旋转中心、base_link 和 IMU 之间的通用位置及水平速度换算。
2026.7.18
    base_link 默认恢复与 IMU 重合；保留通用杆臂换算供 control_link
    旋转中心标定及非零传感器偏置配置使用。
"""

import math


def _finite_vector(values, name, expected_length):
    """校验并转换有限数值向量。"""
    converted = tuple(float(value) for value in values)
    if len(converted) != expected_length:
        raise ValueError(
            '{} 必须包含 {} 个元素'.format(name, expected_length))
    if not all(math.isfinite(value) for value in converted):
        raise ValueError('{} 包含非有限值'.format(name))
    return converted


def rotate_vector_by_quaternion(vector, quaternion):
    """使用单位四元数把 base_link 向量旋转到 map/NED。"""
    x, y, z = _finite_vector(vector, 'vector', 3)
    qx, qy, qz, qw = _finite_vector(quaternion, 'quaternion', 4)
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-12:
        raise ValueError('quaternion 模长不能为零')
    qx, qy, qz, qw = (
        qx / norm,
        qy / norm,
        qz / norm,
        qw / norm,
    )

    # 单位四元数对应的主动旋转矩阵。
    return (
        (1.0 - 2.0 * (qy * qy + qz * qz)) * x
        + 2.0 * (qx * qy - qz * qw) * y
        + 2.0 * (qx * qz + qy * qw) * z,
        2.0 * (qx * qy + qz * qw) * x
        + (1.0 - 2.0 * (qx * qx + qz * qz)) * y
        + 2.0 * (qy * qz - qx * qw) * z,
        2.0 * (qx * qz - qy * qw) * x
        + 2.0 * (qy * qz + qx * qw) * y
        + (1.0 - 2.0 * (qx * qx + qy * qy)) * z,
    )


def sensor_position_from_base(
        base_position, orientation, base_to_sensor):
    """由 base_link 位置计算导航传感器位置。"""
    base = _finite_vector(base_position, 'base_position', 3)
    rotated_offset = rotate_vector_by_quaternion(
        base_to_sensor, orientation)
    return tuple(
        base[index] + rotated_offset[index]
        for index in range(3)
    )


def base_position_from_sensor(
        sensor_position, orientation, base_to_sensor):
    """由导航传感器位置反算 base_link 位置。"""
    sensor = _finite_vector(sensor_position, 'sensor_position', 3)
    rotated_offset = rotate_vector_by_quaternion(
        base_to_sensor, orientation)
    return tuple(
        sensor[index] - rotated_offset[index]
        for index in range(3)
    )


def offset_point_from_origin(origin_position, orientation, origin_to_point):
    """由刚体参考原点位置计算偏置点位置。"""
    return sensor_position_from_base(
        origin_position,
        orientation,
        origin_to_point,
    )


def origin_from_offset_point(point_position, orientation, origin_to_point):
    """由刚体偏置点位置反算参考原点位置。"""
    return base_position_from_sensor(
        point_position,
        orientation,
        origin_to_point,
    )


def planar_origin_velocity_from_point(
        point_forward_velocity,
        point_lateral_velocity,
        yaw_rate,
        origin_to_point):
    """把偏置点线速度换算为同姿态刚体原点的前、右速度。"""
    offset = _finite_vector(origin_to_point, 'origin_to_point', 3)
    point_u = float(point_forward_velocity)
    point_v = float(point_lateral_velocity)
    rate = float(yaw_rate)
    if not all(math.isfinite(value) for value in (point_u, point_v, rate)):
        raise ValueError('速度包含非有限值')
    # v_point = v_origin + omega × r，FRD 中 omega=(0, 0, r)。
    return (
        point_u + rate * offset[1],
        point_v - rate * offset[0],
    )


def offset_between_origins(origin_a_to_point, origin_b_to_point):
    """由两个原点到同一刚体点的杆臂计算原点 A 到原点 B 的偏置。"""
    offset_a = _finite_vector(
        origin_a_to_point, 'origin_a_to_point', 3)
    offset_b = _finite_vector(
        origin_b_to_point, 'origin_b_to_point', 3)
    return tuple(
        offset_a[index] - offset_b[index]
        for index in range(3)
    )
