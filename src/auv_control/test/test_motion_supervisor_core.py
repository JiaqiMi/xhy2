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
    slew_value,
    stopping_distance,
    wrap_angle,
)


class MotionSupervisorCoreTest(unittest.TestCase):

    def setUp(self):
        self.parameters = {
            'stable_frames': 2,
            'force_slew_per_cycle': 10000.0,
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

    def test_wrap_angle_crosses_180_degrees(self):
        error = wrap_angle(math.radians(-179.0) - math.radians(179.0))
        self.assertAlmostEqual(error, math.radians(2.0), places=6)

    def test_force_protocol_and_slew_limits(self):
        self.assertEqual(protocol_force(12000), 10000)
        self.assertEqual(protocol_force(-12000), -10000)
        self.assertEqual(protocol_force(-123.6), -124)
        self.assertEqual(slew_value(0.0, 300.0, 50.0), 50.0)
        self.assertEqual(slew_value(100.0, -300.0, 50.0), 50.0)

    def test_stopping_distance_includes_delay_and_margin(self):
        distance = stopping_distance(0.2, 0.1, 0.35, 0.15)
        self.assertAlmostEqual(distance, 0.42, places=6)
        self.assertAlmostEqual(
            stopping_distance(-0.2, 0.1, 0.35, 0.15), 0.15, places=6)

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

        output = self.core.step(self.vehicle(x=1.80, u=0.20))
        self.assertEqual(output.state, TRANSLATE_BRAKE)
        self.assertLess(output.tx, 0)

        self.core.step(self.vehicle(x=1.80))
        output = self.core.step(self.vehicle(x=1.80))
        self.assertEqual(output.state, ALIGN_FINAL)

        output = self.core.step(self.vehicle(x=1.80))
        self.assertEqual(output.state, FINAL_BRAKE)
        self.core.step(self.vehicle(x=1.80))
        output = self.core.step(self.vehicle(x=1.80))
        self.assertEqual(output.state, CAPTURE)

        self.core.step(self.vehicle(x=1.80))
        output = self.core.step(self.vehicle(x=1.80))
        self.assertEqual(output.state, HOVER)
        self.assertEqual(output.mode, MODE_DPROV)
        self.assertEqual((output.tx, output.ty, output.mz), (0, 0, 0))

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


if __name__ == '__main__':
    unittest.main()
