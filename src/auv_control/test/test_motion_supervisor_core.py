#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_motion_supervisor_core.py
功能：motion_supervisor 纯算法核心单元测试
作者：BroXu
说明：
    覆盖坐标转换、角度归一化、力限幅、停车距离、完整状态转换、
    反馈超时、目标抢占、取消和定点异常回退。
记录：
2026.7.16
    新增运动—刹停—悬停控制核心自动测试。
2026.7.16
    增加正负方向刹车限幅与减速度选择测试。
2026.7.17
    增加固定目标深度在目标、兜底和取消路径中的回归测试。
2026.7.17
    增加最终转向期间水平刹停和漂出捕获区回退测试。
2026.7.17
    增加停车主轴、最小刹转力矩和定点接管保护测试。
"""

import math
import os
import sys
import unittest


DRIVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'driver'))
if DRIVER_DIR not in sys.path:
    sys.path.insert(0, DRIVER_DIR)

from motion_supervisor_core import (  # noqa: E402
    ALIGN_FINAL,
    ALIGN_PATH,
    ALIGN_PATH_BRAKE,
    CAPTURE,
    FINAL_BRAKE,
    HOVER,
    MODE_DPROV,
    MODE_DEPTH,
    SAFE,
    TRANSLATE,
    TRANSLATE_BRAKE,
    MotionGoal,
    MotionSupervisorCore,
    VehicleState,
    map_error_to_body,
    protocol_force,
    relative_target_xy,
    slew_value,
    stopping_distance,
    wrap_angle,
)


class MotionSupervisorCoreTest(unittest.TestCase):

    def setUp(self):
        self.parameters = {
            'stable_frames': 2,
            'force_slew_per_cycle': 10000.0,
            'brake_force_slew_per_cycle': 10000.0,
            'mode_ack_timeout': 1.0,
        }
        self.core = MotionSupervisorCore(self.parameters)
        self.now = 0.0

    def vehicle(
            self, x=0.0, y=0.0, z=1.5, yaw=0.0, u=0.0, v=0.0,
            r=0.0, fresh=True, mode=MODE_DEPTH):
        self.now += 0.2
        return VehicleState(
            self.now, x, y, z, yaw, u, v, r,
            feedback_fresh=fresh, reported_mode=mode)

    def test_coordinate_conversion_for_cardinal_yaws(self):
        cases = (
            (0.0, (1.0, 0.0)),
            (math.pi / 2.0, (0.0, -1.0)),
            (-math.pi / 2.0, (0.0, 1.0)),
            (math.pi, (-1.0, 0.0)),
        )
        for yaw, expected in cases:
            actual = map_error_to_body(1.0, 0.0, yaw)
            self.assertAlmostEqual(actual[0], expected[0], places=6)
            self.assertAlmostEqual(actual[1], expected[1], places=6)

    def test_relative_base_offset_for_cardinal_yaws(self):
        cases = (
            (0.0, (2.0, 3.0)),
            (math.pi / 2.0, (-1.0, 2.0)),
            (-math.pi / 2.0, (3.0, 0.0)),
            (math.pi, (0.0, -1.0)),
        )
        for yaw, expected in cases:
            actual = relative_target_xy(
                1.0, 1.0, yaw, 1.0, 2.0, 'base_link')
            self.assertAlmostEqual(actual[0], expected[0], places=6)
            self.assertAlmostEqual(actual[1], expected[1], places=6)

    def test_relative_map_offset_does_not_rotate(self):
        actual = relative_target_xy(
            1.0, 2.0, math.pi / 2.0, 3.0, -4.0, 'map')
        self.assertEqual(actual, (4.0, -2.0))
        with self.assertRaises(ValueError):
            relative_target_xy(0.0, 0.0, 0.0, 1.0, 0.0, 'odom')

    def test_wrap_angle_crosses_180_degrees(self):
        error = wrap_angle(math.radians(-179.0) - math.radians(179.0))
        self.assertAlmostEqual(error, math.radians(2.0), places=6)

    def test_force_protocol_and_slew_limits(self):
        self.assertEqual(protocol_force(12000), 10000)
        self.assertEqual(protocol_force(-12000), -10000)
        self.assertEqual(protocol_force(-123.6), -124)
        self.assertEqual(slew_value(0.0, 300.0, 50.0), 50.0)
        self.assertEqual(slew_value(100.0, -300.0, 50.0), 50.0)

    def test_directional_brake_force_limits(self):
        core = MotionSupervisorCore({
            'brake_max_tx_positive': 2000.0,
            'brake_max_tx_negative': 3000.0,
            'brake_max_ty_positive': 2100.0,
            'brake_max_ty_negative': 4000.0,
            'brake_max_mz_positive': 2500.0,
            'brake_max_mz_negative': 3500.0,
            'brake_force_slew_per_cycle': 10000.0,
        })
        self.assertEqual(
            core._limited_forces(
                9000.0, -9000.0, 9000.0, braking=True),
            (2000, -4000, 2500),
        )
        self.assertEqual(
            core._limited_forces(
                -9000.0, 9000.0, -9000.0, braking=True),
            (-3000, 2100, -3500),
        )

    def test_horizontal_deceleration_uses_brake_force_direction(self):
        core = MotionSupervisorCore({
            'brake_acceleration_tx_positive': 0.02,
            'brake_acceleration_tx_negative': 0.08,
            'brake_acceleration_ty_positive': 0.03,
            'brake_acceleration_ty_negative': 0.06,
        })
        forward = self.vehicle(u=0.2)
        backward = self.vehicle(u=-0.2)
        right = self.vehicle(v=0.1)
        left = self.vehicle(v=-0.1)
        self.assertAlmostEqual(
            core._horizontal_brake_acceleration(forward, 1.0, 0.0),
            0.08,
        )
        self.assertAlmostEqual(
            core._horizontal_brake_acceleration(backward, -1.0, 0.0),
            0.02,
        )
        self.assertAlmostEqual(
            core._horizontal_brake_acceleration(right, 0.0, 1.0),
            0.06,
        )
        self.assertAlmostEqual(
            core._horizontal_brake_acceleration(left, 0.0, -1.0),
            0.03,
        )

    def test_horizontal_deceleration_ignores_minor_cross_axis(self):
        core = MotionSupervisorCore({
            'brake_axis_relevance_ratio': 0.20,
            'brake_acceleration_tx_positive': 0.10,
            'brake_acceleration_tx_negative': 0.10,
            'brake_acceleration_ty_positive': 0.05,
            'brake_acceleration_ty_negative': 0.05,
        })
        mostly_forward = self.vehicle(u=0.20, v=0.01)
        diagonal = self.vehicle(u=0.20, v=0.10)
        self.assertAlmostEqual(
            core._horizontal_brake_acceleration(
                mostly_forward, 1.0, 0.05),
            0.10,
        )
        self.assertAlmostEqual(
            core._horizontal_brake_acceleration(
                diagonal, 1.0, 0.50),
            0.05,
        )

    def test_angular_deceleration_uses_mz_direction(self):
        core = MotionSupervisorCore({
            'brake_gain_yaw': -6000.0,
            'angular_brake_acceleration_mz_positive': 0.20,
            'angular_brake_acceleration_mz_negative': 0.40,
            'yaw_brake_margin': 0.0,
        })
        self.assertAlmostEqual(
            core._angular_stop_threshold(0.4),
            0.4,
        )
        self.assertAlmostEqual(
            core._angular_stop_threshold(-0.4),
            0.2,
        )

    def test_calibrated_brake_output_uses_actual_force_signs(self):
        positive_velocity = self.core._brake_output(
            self.vehicle(u=0.2, v=0.2, r=0.5)
        )
        self.assertEqual(
            (positive_velocity.tx, positive_velocity.ty, positive_velocity.mz),
            (-3000, -4000, 3000),
        )

        negative_velocity = self.core._brake_output(
            self.vehicle(u=-0.2, v=-0.2, r=-0.5)
        )
        self.assertEqual(
            (negative_velocity.tx, negative_velocity.ty, negative_velocity.mz),
            (2000, 2000, -3000),
        )

    def test_yaw_brake_uses_minimum_effective_moment(self):
        core = MotionSupervisorCore({
            'brake_gain_yaw': -6000.0,
            'brake_min_mz': 100.0,
            'yaw_rate_threshold': math.radians(0.3),
        })
        self.assertEqual(
            core._yaw_brake_command(math.radians(0.5)),
            100.0,
        )
        self.assertAlmostEqual(
            core._yaw_brake_command(math.radians(0.1)),
            6000.0 * math.radians(0.1),
        )

    def test_stopping_distance_includes_delay_and_margin(self):
        distance = stopping_distance(0.2, 0.1, 0.35, 0.15)
        self.assertAlmostEqual(distance, 0.42, places=6)
        self.assertAlmostEqual(
            stopping_distance(-0.2, 0.1, 0.35, 0.15), 0.15, places=6)

    def test_target_depth_is_fixed_for_all_output_paths(self):
        output = self.core.step(self.vehicle(z=-0.2))
        self.assertAlmostEqual(output.target.z, -0.6)

        self.core.set_goal(MotionGoal(2.0, 0.0, -1.2, 0.0))
        output = self.core.step(self.vehicle(z=-0.3))
        self.assertAlmostEqual(output.target.z, -0.6)
        self.assertAlmostEqual(self.core.goal.z, -0.6)

        configured_core = MotionSupervisorCore({'fixed_target_z': -1.2})
        configured_core.set_goal(MotionGoal(1.0, 0.0, -0.6, 0.0))
        configured_output = configured_core.step(self.vehicle())
        self.assertAlmostEqual(configured_output.target.z, -1.2)

    def test_complete_sequence_uses_depth_then_dprov(self):
        self.core.set_goal(MotionGoal(2.0, 0.0, 1.5, 0.0))

        output = self.core.step(self.vehicle())
        self.assertEqual(output.state, ALIGN_PATH_BRAKE)
        self.assertEqual(output.mode, MODE_DEPTH)

        self.core.step(self.vehicle())
        output = self.core.step(self.vehicle())
        self.assertEqual(output.state, TRANSLATE)
        self.assertEqual(output.mode, MODE_DEPTH)

        output = self.core.step(self.vehicle())
        self.assertEqual(output.state, TRANSLATE)
        self.assertGreater(output.tx, 0)
        self.assertEqual(output.mode, MODE_DEPTH)

        output = self.core.step(self.vehicle(x=1.90, u=0.20))
        self.assertEqual(output.state, TRANSLATE_BRAKE)
        self.assertLess(output.tx, 0)

        self.core.step(self.vehicle(x=1.90))
        output = self.core.step(self.vehicle(x=1.90))
        self.assertEqual(output.state, ALIGN_FINAL)

        output = self.core.step(self.vehicle(x=1.90))
        self.assertEqual(output.state, FINAL_BRAKE)
        self.core.step(self.vehicle(x=1.90))
        output = self.core.step(self.vehicle(x=1.90))
        self.assertEqual(output.state, HOVER)
        self.assertEqual(output.mode, MODE_DPROV)
        self.assertEqual((output.tx, output.ty, output.mz), (0, 0, 0))

    def test_align_final_brakes_translation_and_reacquires_after_drift(self):
        self.core.goal = MotionGoal(0.0, 0.0, -0.6, 0.2)
        self.core.state = ALIGN_FINAL

        output = self.core.step(
            self.vehicle(x=0.1, yaw=0.0, u=0.02, v=-0.01))
        self.assertEqual(output.state, ALIGN_FINAL)
        self.assertLess(output.tx, 0)
        self.assertGreater(output.ty, 0)
        self.assertGreater(output.mz, 0)

        output = self.core.step(self.vehicle(x=0.4, yaw=0.0))
        self.assertEqual(output.state, ALIGN_PATH)
        self.assertEqual(output.mode, MODE_DEPTH)

    def test_final_brake_requires_capture_radius_before_hover(self):
        self.core.goal = MotionGoal(0.0, 0.0, -0.6, 0.0)
        self.core.state = FINAL_BRAKE
        output = self.core.step(self.vehicle(x=0.20))
        self.assertEqual(output.state, ALIGN_PATH)
        self.assertEqual(output.mode, MODE_DEPTH)

    def test_feedback_timeout_enters_safe_with_zero_force(self):
        self.core.set_goal(MotionGoal(2.0, 0.0, 1.5, 0.0))
        self.core.step(self.vehicle())
        output = self.core.step(self.vehicle(u=0.5, fresh=False))
        self.assertEqual(output.state, SAFE)
        self.assertEqual(output.mode, MODE_DEPTH)
        self.assertEqual((output.tx, output.ty, output.mz), (0, 0, 0))

    def test_new_goal_preempts_through_brake(self):
        self.core.set_goal(MotionGoal(2.0, 0.0, 1.5, 0.0))
        self.core.step(self.vehicle())
        self.core.set_goal(MotionGoal(0.0, 2.0, 1.5, math.pi / 2.0))
        output = self.core.step(self.vehicle(u=0.2))
        self.assertEqual(output.state, TRANSLATE_BRAKE)
        self.assertLess(output.tx, 0)

        self.core.step(self.vehicle())
        output = self.core.step(self.vehicle())
        self.assertEqual(output.state, ALIGN_PATH)
        self.assertAlmostEqual(output.target.y, 2.0)

    def test_cancel_brakes_and_hovers_at_current_pose(self):
        self.core.set_goal(MotionGoal(2.0, 0.0, 1.5, 0.0))
        self.core.step(self.vehicle())
        self.core.cancel()
        self.core.step(self.vehicle(x=0.4))
        output = self.core.step(self.vehicle(x=0.4))
        self.assertEqual(output.state, CAPTURE)
        self.assertAlmostEqual(output.target.x, 0.4)
        self.assertAlmostEqual(output.target.z, -0.6)

    def test_hover_ack_timeout_and_abnormal_speed_fallback(self):
        self.core.goal = MotionGoal(0.0, 0.0, 1.5, 0.0)
        self.core.state = HOVER
        self.core.hover_started_at = 0.0
        output = self.core.step(self.vehicle(mode=MODE_DEPTH))
        self.assertEqual(output.state, HOVER)

        self.now = 1.2
        output = self.core.step(self.vehicle(mode=MODE_DEPTH))
        self.assertEqual(output.state, SAFE)
        self.assertEqual(output.mode, MODE_DEPTH)

        self.core.state = HOVER
        self.core.hover_started_at = self.now
        output = self.core.step(self.vehicle(u=0.09, mode=MODE_DPROV))
        self.assertEqual(output.state, TRANSLATE_BRAKE)
        self.assertEqual(output.mode, MODE_DEPTH)

    def test_hover_yaw_error_and_rate_fallback(self):
        self.core.goal = MotionGoal(0.0, 0.0, -0.6, 0.0)
        self.core.state = HOVER
        self.core.hover_started_at = self.now
        output = self.core.step(
            self.vehicle(yaw=math.radians(11.0), mode=MODE_DPROV))
        self.assertEqual(output.state, TRANSLATE_BRAKE)
        self.assertEqual(output.mode, MODE_DEPTH)

    def test_hover_position_error_fallback(self):
        self.core.goal = MotionGoal(0.0, 0.0, -0.6, 0.0)
        self.core.state = HOVER
        self.core.hover_started_at = self.now
        output = self.core.step(
            self.vehicle(x=0.30, mode=MODE_DPROV))
        self.assertEqual(output.state, TRANSLATE_BRAKE)
        self.assertEqual(output.mode, MODE_DEPTH)

        self.core.state = HOVER
        self.core.hover_started_at = self.now
        output = self.core.step(
            self.vehicle(r=math.radians(2.1), mode=MODE_DPROV))
        self.assertEqual(output.state, TRANSLATE_BRAKE)
        self.assertEqual(output.mode, MODE_DEPTH)


if __name__ == '__main__':
    unittest.main()
