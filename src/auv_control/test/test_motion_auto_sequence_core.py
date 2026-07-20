#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_motion_auto_sequence_core.py
功能：验证单轴自动测试序列和 base_link 相对目标换算
作者：BroXu
监听：无
发布：无
记录：
2026.7.18
    新增自动测试动作顺序、坐标转换和目标匹配测试。
2026.7.19
    增加刹停方向分类、稳定等待和零输出停滞判据测试。
"""

import math
import os
import sys
import unittest


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from motion_auto_sequence_core import (  # noqa: E402
    EARLY_STOP,
    ON_TARGET,
    OVERSHOOT,
    build_axis_sequence,
    classify_signed_stop_error,
    goal_matches,
    relative_goal,
    signed_axis_stop_error,
    xy_motion_is_stable,
    xy_motion_is_stalled,
)


class MotionAutoSequenceCoreTest(unittest.TestCase):

    def test_sequence_order_and_repetitions(self):
        steps = build_axis_sequence('x', (0.5, 1.0, 1.5), 3)
        self.assertEqual(len(steps), 36)
        self.assertEqual(
            [step.offset for step in steps[:4]],
            [0.5, 0.0, -0.5, 0.0],
        )
        self.assertEqual(steps[4].repetition, 2)
        self.assertEqual(steps[12].magnitude, 1.0)

    def test_base_link_x_and_y_offsets_rotate_with_initial_yaw(self):
        x_goal = relative_goal(
            1.0, 2.0, math.pi / 2.0, -0.9, 'x', 0.5)
        y_goal = relative_goal(
            1.0, 2.0, math.pi / 2.0, -0.9, 'y', 0.5)
        self.assertAlmostEqual(x_goal[0], 1.0)
        self.assertAlmostEqual(x_goal[1], 2.5)
        self.assertAlmostEqual(y_goal[0], 0.5)
        self.assertAlmostEqual(y_goal[1], 2.0)

    def test_yaw_offset_keeps_position(self):
        goal = relative_goal(
            1.0, 2.0, math.radians(170.0), -0.9,
            'yaw', math.radians(30.0))
        self.assertAlmostEqual(goal[0], 1.0)
        self.assertAlmostEqual(goal[1], 2.0)
        self.assertAlmostEqual(goal[3], math.radians(-160.0))

    def test_goal_match_wraps_yaw(self):
        self.assertTrue(goal_matches(
            1.0, 2.0, -0.9, math.radians(179.0),
            1.01, 2.01, -0.91, math.radians(-179.0),
            0.05, 0.05, math.radians(3.0),
        ))

    def test_signed_stop_error_distinguishes_early_and_overshoot(self):
        self.assertAlmostEqual(
            signed_axis_stop_error(0.45, 0.5, 'positive'), -0.05)
        self.assertAlmostEqual(
            signed_axis_stop_error(0.55, 0.5, 'positive'), 0.05)
        self.assertAlmostEqual(
            signed_axis_stop_error(0.05, 0.0, 'return_after_positive'),
            -0.05,
        )
        self.assertAlmostEqual(
            signed_axis_stop_error(-0.05, 0.0, 'return_after_positive'),
            0.05,
        )

    def test_stop_classification_uses_signed_error(self):
        self.assertEqual(classify_signed_stop_error(-0.02), EARLY_STOP)
        self.assertEqual(classify_signed_stop_error(0.02), OVERSHOOT)
        self.assertEqual(classify_signed_stop_error(0.004), ON_TARGET)

    def test_xy_stability_requires_position_and_speed(self):
        self.assertTrue(xy_motion_is_stable(0.04, 0.004, 0.05, 0.005))
        self.assertFalse(xy_motion_is_stable(0.06, 0.004, 0.05, 0.005))
        self.assertFalse(xy_motion_is_stable(0.04, 0.006, 0.05, 0.005))

    def test_xy_stall_requires_error_low_speed_and_near_zero_force(self):
        self.assertTrue(xy_motion_is_stalled(
            0.08, 0.004, 0.0, 1.0, 0.05, 0.005, 1.0))
        self.assertFalse(xy_motion_is_stalled(
            0.08, 0.004, 200.0, 0.0, 0.05, 0.005, 1.0))
        self.assertFalse(xy_motion_is_stalled(
            0.04, 0.004, 0.0, 0.0, 0.05, 0.005, 1.0))

    def test_invalid_sequence_parameters_are_rejected(self):
        with self.assertRaises(ValueError):
            build_axis_sequence('z', (0.5,), 3)
        with self.assertRaises(ValueError):
            build_axis_sequence('x', (0.0,), 3)
        with self.assertRaises(ValueError):
            build_axis_sequence('x', (0.5,), 0)


if __name__ == '__main__':
    unittest.main()
