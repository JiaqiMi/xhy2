#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_motion_depth_auto_test_core.py
功能：验证绝对深度阶跃序列、TF 垂向速度、稳定判据和超调计算
作者：BroXu
监听：无
发布：无
记录：
2026.7.31
    新增深度自动测试纯算法单元测试。
"""

from __future__ import division

import os
import sys
import unittest


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from motion_depth_auto_test_core import (  # noqa: E402
    build_depth_sequence,
    depth_motion_is_stable,
    directed_depth_overshoot,
    linear_vertical_speed,
)


class DepthSequenceTest(unittest.TestCase):
    """验证绝对深度数组与启动基准深度交替执行。"""

    def test_targets_are_absolute_and_return_to_baseline(self):
        steps = build_depth_sequence(
            (-0.3, -0.6, -0.9), baseline_depth=-0.12, cycle_count=2)

        self.assertEqual(
            [step.target_depth for step in steps],
            [
                -0.3, -0.12, -0.6, -0.12, -0.9, -0.12,
                -0.3, -0.12, -0.6, -0.12, -0.9, -0.12,
            ],
        )
        self.assertEqual([step.cycle for step in steps], [1] * 6 + [2] * 6)
        self.assertEqual(
            [step.phase for step in steps],
            ['target', 'return_to_baseline'] * 6,
        )

    def test_rejects_invalid_sequence_parameters(self):
        for targets in ((), (float('nan'),), (float('inf'),)):
            with self.assertRaises(ValueError):
                build_depth_sequence(targets, -0.1, 1)
        with self.assertRaises(ValueError):
            build_depth_sequence((-0.3,), float('nan'), 1)
        with self.assertRaises(ValueError):
            build_depth_sequence((-0.3,), -0.1, 0)


class DepthStabilityTest(unittest.TestCase):
    """验证 TF 速度拟合和宽泛到达判据。"""

    def test_linear_vertical_speed_uses_window_regression(self):
        samples = (
            (10.0, -0.20),
            (10.5, -0.22),
            (11.0, -0.24),
            (11.5, -0.26),
        )

        self.assertAlmostEqual(linear_vertical_speed(samples), -0.04)

    def test_depth_and_speed_must_both_be_stable(self):
        self.assertTrue(depth_motion_is_stable(
            -0.51, -0.60, -0.010, 0.10, 0.015))
        self.assertFalse(depth_motion_is_stable(
            -0.49, -0.60, -0.010, 0.10, 0.015))
        self.assertFalse(depth_motion_is_stable(
            -0.51, -0.60, -0.020, 0.10, 0.015))

    def test_overshoot_respects_dive_and_ascent_direction(self):
        self.assertAlmostEqual(
            directed_depth_overshoot(-0.1, -0.6, -0.68, -0.1),
            0.08,
        )
        self.assertAlmostEqual(
            directed_depth_overshoot(-0.7, -0.1, -0.7, -0.03),
            0.07,
        )


if __name__ == '__main__':
    unittest.main()

