#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_rotation_center_calibration_core.py
功能：验证自由转向控制、角度展开和分段旋转中心拟合
作者：BroXu
监听：无
发布：无
记录：
2026.7.18
    新增旋转中心标定纯算法单元测试。
2026.7.18
    增加目标前方向错误和越过目标后 BRAKE 正常超调的保护逻辑测试。
2026.7.18
    增加固定初始位置 mode=4 接管稳定条件测试。
2026.7.19
    验证正负计划方向可拟合不同旋转中心。
"""

import math
import os
import sys
import unittest


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from rotation_center_calibration_core import (  # noqa: E402
    CalibrationSample,
    direct_yaw_command,
    fit_planned_direction_centers,
    fit_segmented_rotation_center,
    locked_handover_is_stable,
    rotation_is_unexpectedly_moving_away,
    unwrap_angle,
)


class RotationCenterCalibrationCoreTest(unittest.TestCase):

    def command(self, error, rate):
        return direct_yaw_command(
            error,
            rate,
            kp=1000.0,
            kd=500.0,
            brake_gain=3000.0,
            track_positive_limit=1000.0,
            track_negative_limit=1200.0,
            brake_positive_limit=2000.0,
            brake_negative_limit=2500.0,
            brake_acceleration_positive=0.5,
            brake_acceleration_negative=0.4,
            control_delay=0.2,
            brake_margin=math.radians(2.0),
            yaw_tolerance=math.radians(2.0),
            yaw_rate_threshold=math.radians(1.0),
            minimum_brake_mz=100.0,
        )

    def test_unwrap_crosses_pi_continuously(self):
        value = unwrap_angle(
            math.radians(179.0), math.radians(-179.0))
        self.assertAlmostEqual(value, math.radians(181.0))

    def test_tracking_and_braking_commands(self):
        command, phase, unused_stop = self.command(
            math.radians(60.0), 0.0)
        self.assertEqual(phase, 'TRACK')
        self.assertGreater(command, 0)

        command, phase, unused_stop = self.command(
            math.radians(3.0), math.radians(20.0))
        self.assertEqual(phase, 'BRAKE')
        self.assertLess(command, 0)

    def test_hold_requires_angle_and_rate_stable(self):
        command, phase, unused_stop = self.command(
            math.radians(1.0), math.radians(0.5))
        self.assertEqual((command, phase), (0, 'HOLD'))

    def test_moving_away_before_crossing_target_is_protected(self):
        threshold = math.radians(0.5)
        self.assertTrue(rotation_is_unexpectedly_moving_away(
            math.radians(30.0),
            math.radians(-1.1),
            1,
            threshold,
        ))
        self.assertTrue(rotation_is_unexpectedly_moving_away(
            math.radians(-30.0),
            math.radians(1.1),
            -1,
            threshold,
        ))

    def test_braking_overshoot_from_log_is_not_direction_error(self):
        command, phase, unused_stop = self.command(
            math.radians(-4.8), math.radians(1.1))
        self.assertEqual(phase, 'BRAKE')
        self.assertLess(command, 0)
        self.assertFalse(rotation_is_unexpectedly_moving_away(
            math.radians(-4.8),
            math.radians(1.1),
            1,
            math.radians(0.5),
        ))

    def test_locked_handover_requires_all_conditions(self):
        arguments = dict(
            reported_mode=4,
            position_error=0.10,
            horizontal_speed=0.03,
            yaw_error=math.radians(2.0),
            yaw_rate=math.radians(0.3),
            required_mode=4,
            position_tolerance=0.15,
            speed_threshold=0.05,
            yaw_tolerance=math.radians(3.0),
            yaw_rate_threshold=math.radians(0.5),
        )
        self.assertTrue(locked_handover_is_stable(**arguments))
        for key, invalid_value in (
                ('reported_mode', 2),
                ('position_error', 0.16),
                ('horizontal_speed', 0.06),
                ('yaw_error', math.radians(4.0)),
                ('yaw_rate', math.radians(0.6))):
            changed = dict(arguments)
            changed[key] = invalid_value
            self.assertFalse(locked_handover_is_stable(**changed))

    def test_fit_recovers_shared_offset_with_different_centers(self):
        offset = (0.32, -0.07)
        centers = ((1.0, 2.0), (-0.5, 3.0), (4.0, -2.0))
        samples = []
        for segment, center in enumerate(centers, 1):
            for degrees in range(-80, 81, 10):
                yaw = math.radians(degrees + 15 * segment)
                x = (
                    center[0]
                    + math.cos(yaw) * offset[0]
                    - math.sin(yaw) * offset[1]
                )
                y = (
                    center[1]
                    + math.sin(yaw) * offset[0]
                    + math.cos(yaw) * offset[1]
                )
                samples.append(CalibrationSample(
                    segment, x, y, yaw))
        samples.append(CalibrationSample(1, 99.0, -99.0, 0.2))

        result = fit_segmented_rotation_center(samples)

        self.assertAlmostEqual(result['offset_x'], offset[0], places=6)
        self.assertAlmostEqual(result['offset_y'], offset[1], places=6)
        self.assertGreaterEqual(result['rejected_sample_count'], 1)
        self.assertLess(result['rms_residual'], 1e-6)

    def test_fit_rejects_insufficient_samples(self):
        with self.assertRaises(ValueError):
            fit_segmented_rotation_center([
                CalibrationSample(1, 0.0, 0.0, 0.0)
            ])

    def test_planned_directions_recover_distinct_centers(self):
        offsets = {
            'positive': (0.20, -0.10),
            'negative': (-0.08, 0.04),
        }
        samples = {'positive': [], 'negative': []}
        for direction_index, direction_name in enumerate(
                ('positive', 'negative'), 1):
            offset = offsets[direction_name]
            for segment in range(1, 3):
                center = (
                    direction_index + 0.3 * segment,
                    -direction_index + 0.2 * segment,
                )
                for degrees in range(-70, 71, 10):
                    yaw = math.radians(degrees + 20 * segment)
                    samples[direction_name].append(CalibrationSample(
                        segment,
                        center[0]
                        + math.cos(yaw) * offset[0]
                        - math.sin(yaw) * offset[1],
                        center[1]
                        + math.sin(yaw) * offset[0]
                        + math.cos(yaw) * offset[1],
                        yaw,
                    ))
        results = fit_planned_direction_centers(
            samples['positive'], samples['negative'])
        for direction_name, expected in offsets.items():
            self.assertAlmostEqual(
                results[direction_name]['offset_x'], expected[0], places=6)
            self.assertAlmostEqual(
                results[direction_name]['offset_y'], expected[1], places=6)


if __name__ == '__main__':
    unittest.main()
