#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_lever_arm.py
功能：验证 base_link 与 IMU/GNSS 定位点之间的杆臂补偿
作者：BroXu
监听：无
发布：无
记录：
2026.7.18
    增加 0°、±90° 航向和双向位置换算回归测试。
2026.7.18
    增加 control_link、base_link 和 IMU 之间的位置与速度换算测试。
"""

import math
import os
import sys
import unittest


DRIVER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'driver'))
if DRIVER_DIR not in sys.path:
    sys.path.insert(0, DRIVER_DIR)

from lever_arm import (  # noqa: E402
    base_position_from_sensor,
    offset_point_from_origin,
    offset_between_origins,
    origin_from_offset_point,
    planar_origin_velocity_from_point,
    sensor_position_from_base,
)
from world_frame import WorldFrameManager  # noqa: E402


def yaw_quaternion(yaw):
    """生成只有 yaw 的单位四元数。"""
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def euler_quaternion(roll, pitch, yaw):
    """生成 roll、pitch、yaw 对应的单位四元数。"""
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class LeverArmTest(unittest.TestCase):

    def setUp(self):
        self.base_to_imu = (-0.35, 0.0, 0.0)

    def assert_vector_almost_equal(self, actual, expected):
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=9)

    def test_zero_yaw_moves_base_forward_from_imu(self):
        base = base_position_from_sensor(
            (-0.35, 0.0, 0.0),
            yaw_quaternion(0.0),
            self.base_to_imu,
        )
        self.assert_vector_almost_equal(base, (0.0, 0.0, 0.0))

    def test_positive_90_yaw_rotates_forward_offset_to_east(self):
        sensor = sensor_position_from_base(
            (0.0, 0.0, 0.0),
            yaw_quaternion(math.pi / 2.0),
            self.base_to_imu,
        )
        self.assert_vector_almost_equal(sensor, (0.0, -0.35, 0.0))
        recovered = base_position_from_sensor(
            sensor,
            yaw_quaternion(math.pi / 2.0),
            self.base_to_imu,
        )
        self.assert_vector_almost_equal(recovered, (0.0, 0.0, 0.0))

    def test_negative_90_yaw_rotates_forward_offset_to_west(self):
        sensor = sensor_position_from_base(
            (1.0, 2.0, -0.8),
            yaw_quaternion(-math.pi / 2.0),
            self.base_to_imu,
        )
        self.assert_vector_almost_equal(sensor, (1.0, 2.35, -0.8))

    def test_round_trip_for_arbitrary_pose(self):
        base = (3.2, -1.4, 0.7)
        orientation = euler_quaternion(
            math.radians(8.0),
            math.radians(-12.0),
            math.radians(37.0),
        )
        sensor = sensor_position_from_base(
            base, orientation, self.base_to_imu)
        recovered = base_position_from_sensor(
            sensor, orientation, self.base_to_imu)
        self.assert_vector_almost_equal(recovered, base)

    def test_target_lld_round_trip_preserves_base_position(self):
        frame = WorldFrameManager(30.0, 120.0, 0.8)
        base_target = (4.0, -2.0, 1.1)
        orientation = yaw_quaternion(math.radians(65.0))
        sensor_target = sensor_position_from_base(
            base_target, orientation, self.base_to_imu)
        target_lld = frame.ned_to_lld(*sensor_target)
        recovered_sensor = frame.lld_to_ned(*target_lld)
        recovered_base = base_position_from_sensor(
            recovered_sensor, orientation, self.base_to_imu)
        for actual_value, expected_value in zip(
                recovered_base, base_target):
            self.assertAlmostEqual(
                actual_value, expected_value, places=5)

    def test_control_and_base_goal_round_trip(self):
        control_to_base = (0.35, 0.0, 0.0)
        orientation = yaw_quaternion(math.radians(60.0))
        control = (1.0, -2.0, -0.8)
        base = offset_point_from_origin(
            control, orientation, control_to_base)
        recovered = origin_from_offset_point(
            base, orientation, control_to_base)
        self.assert_vector_almost_equal(recovered, control)

    def test_imu_velocity_is_shifted_to_rotation_center(self):
        control_to_imu = (-0.20, 0.10, 0.0)
        yaw_rate = 0.5
        # 静止旋转中心对应的 IMU 杆臂速度 omega × r。
        imu_u = -yaw_rate * control_to_imu[1]
        imu_v = yaw_rate * control_to_imu[0]
        control_u, control_v = planar_origin_velocity_from_point(
            imu_u, imu_v, yaw_rate, control_to_imu)
        self.assertAlmostEqual(control_u, 0.0, places=9)
        self.assertAlmostEqual(control_v, 0.0, places=9)

    def test_fixed_control_center_reaches_final_base_pose_after_yaw(self):
        control_to_base = (0.35, 0.0, 0.0)
        final_yaw = math.pi / 2.0
        final_orientation = yaw_quaternion(final_yaw)
        requested_base = (1.0, 2.0, -0.8)
        control_target = origin_from_offset_point(
            requested_base,
            final_orientation,
            control_to_base,
        )
        recovered_base = offset_point_from_origin(
            control_target,
            final_orientation,
            control_to_base,
        )
        self.assert_vector_almost_equal(recovered_base, requested_base)

    def test_control_to_base_is_derived_from_common_imu_point(self):
        control_to_imu = (0.0, 0.0, 0.0)
        base_to_imu = (-0.35, 0.0, 0.0)
        self.assert_vector_almost_equal(
            offset_between_origins(control_to_imu, base_to_imu),
            (0.35, 0.0, 0.0),
        )


if __name__ == '__main__':
    unittest.main()
