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
2026.7.19
    验证低、中、高不对称力矩档位和优先中速的推荐策略。
2026.7.19
    验证旋转漂移超过半径时只产生告警判定，不中止后续定点接管流程。
2026.7.19
    回归验证越过 90° 目标后不再恢复原方向固定 MZ，而是使用闭环指令刹停。
2026.7.19
    验证判稳后连续保持计时，以及越过目标后的方向保护锁存。
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
    apply_fixed_track_policy,
    build_torque_profiles,
    direct_yaw_command,
    drift_exceeds_warning_threshold,
    fixed_track_command,
    fit_planned_direction_centers,
    fit_segmented_rotation_center,
    fit_quality_score,
    locked_handover_is_stable,
    rotation_is_unexpectedly_moving_away,
    select_recommended_profile,
    unwrap_angle,
    update_continuous_stability,
)


class RotationCenterCalibrationCoreTest(unittest.TestCase):

    def test_builds_three_asymmetric_torque_profiles(self):
        profiles = build_torque_profiles(
            (1000.0, 2000.0, 3000.0), negative_scale=1.5)
        self.assertEqual(
            [(profile.name,
              profile.positive_limit,
              profile.negative_limit)
             for profile in profiles],
            [
                ('low', 1000.0, 1500.0),
                ('medium', 2000.0, 3000.0),
                ('high', 3000.0, 4500.0),
            ],
        )

    def test_torque_profiles_require_three_increasing_safe_levels(self):
        for levels, scale in (
                ((1000.0, 2000.0), 1.5),
                ((1000.0, 1000.0, 3000.0), 1.5),
                ((1000.0, 2000.0, 8000.0), 1.5)):
            with self.assertRaises(ValueError):
                build_torque_profiles(levels, negative_scale=scale)

    def test_fixed_track_command_uses_planned_direction_and_asymmetry(self):
        self.assertEqual(
            fixed_track_command(1, 1.0, 1000.0, 1500.0), 1000)
        self.assertEqual(
            fixed_track_command(-1, 1.0, 1000.0, 1500.0), -1500)
        self.assertEqual(
            fixed_track_command(1, -1.0, 1000.0, 1500.0), -1500)

    def test_fixed_track_is_latched_off_after_crossing_target(self):
        command, phase, crossed = apply_fixed_track_policy(
            command_mz=600,
            controller_phase='TRACK',
            yaw_error=math.radians(30.0),
            direction=1,
            target_crossed=False,
            mz_to_yaw_sign=1.0,
            positive_limit=1000.0,
            negative_limit=1500.0,
        )
        self.assertEqual((command, phase, crossed), (1000, 'TRACK', False))

        closed_loop_command, closed_loop_phase, unused_stop = self.command(
            math.radians(-24.0), math.radians(0.4))
        self.assertEqual(closed_loop_phase, 'TRACK')
        self.assertLess(closed_loop_command, 0)
        command, phase, crossed = apply_fixed_track_policy(
            command_mz=closed_loop_command,
            controller_phase=closed_loop_phase,
            yaw_error=math.radians(-24.0),
            direction=1,
            target_crossed=crossed,
            mz_to_yaw_sign=1.0,
            positive_limit=1000.0,
            negative_limit=1500.0,
        )
        self.assertEqual(
            (command, phase, crossed),
            (closed_loop_command, 'RECOVER', True),
        )

        command, phase, crossed = apply_fixed_track_policy(
            command_mz=420,
            controller_phase='TRACK',
            yaw_error=math.radians(4.0),
            direction=1,
            target_crossed=crossed,
            mz_to_yaw_sign=1.0,
            positive_limit=1000.0,
            negative_limit=1500.0,
        )
        self.assertEqual(
            (command, phase, crossed), (420, 'RECOVER', True))

    def test_recommendation_prefers_medium_when_quality_is_close(self):
        results = {
            'low': {'rms_residual': 0.020, 'max_residual': 0.060},
            'medium': {'rms_residual': 0.024, 'max_residual': 0.065},
            'high': {'rms_residual': 0.018, 'max_residual': 0.055},
        }
        self.assertAlmostEqual(
            fit_quality_score(results['medium']), 0.04025)
        self.assertEqual(
            select_recommended_profile(results), 'medium')

    def test_recommendation_rejects_clearly_unstable_medium(self):
        results = {
            'low': {'rms_residual': 0.020, 'max_residual': 0.040},
            'medium': {'rms_residual': 0.090, 'max_residual': 0.200},
            'high': {'rms_residual': 0.015, 'max_residual': 0.030},
        }
        self.assertEqual(
            select_recommended_profile(results), 'high')

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

    def test_moving_away_protection_is_disabled_after_target_crossing(self):
        arguments = (
            math.radians(-2.9),
            math.radians(0.65),
            -1,
            math.radians(0.5),
        )
        self.assertTrue(rotation_is_unexpectedly_moving_away(*arguments))
        self.assertFalse(rotation_is_unexpectedly_moving_away(
            *arguments, target_crossed=True))

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

    def test_large_rotation_drift_only_triggers_warning(self):
        self.assertFalse(drift_exceeds_warning_threshold(0.75, 0.75))
        self.assertTrue(drift_exceeds_warning_threshold(0.90, 0.75))
        for drift, warning_radius in (
                (-0.01, 0.75),
                (0.50, 0.0)):
            with self.assertRaises(ValueError):
                drift_exceeds_warning_threshold(
                    drift, warning_radius)

    def test_continuous_stability_starts_after_frames_and_resets(self):
        count = 0
        started_at = None
        elapsed = 0.0
        for now in (0.0, 0.2, 0.4, 0.6, 0.8):
            count, started_at, elapsed = update_continuous_stability(
                True, count, started_at, now, required_frames=5)
        self.assertEqual(count, 5)
        self.assertAlmostEqual(started_at, 0.8)
        self.assertAlmostEqual(elapsed, 0.0)

        count, started_at, elapsed = update_continuous_stability(
            True, count, started_at, 12.8, required_frames=5)
        self.assertAlmostEqual(elapsed, 12.0)

        count, started_at, elapsed = update_continuous_stability(
            False, count, started_at, 13.0, required_frames=5)
        self.assertEqual((count, started_at, elapsed), (0, None, 0.0))

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
