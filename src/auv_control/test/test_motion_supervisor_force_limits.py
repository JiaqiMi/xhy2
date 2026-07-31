#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_motion_supervisor_force_limits.py
功能：验证正常跟踪与主动刹车共用方向性最大输出和力矩变化步长
作者：BroXu
监听：无
发布：无
记录：
2026.7.28
    新增统一最大输出、统一步长和制动前馈限幅测试。
"""

from __future__ import division

import os
import sys
import unittest


DRIVER_DIRECTORY = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'driver'))
if DRIVER_DIRECTORY not in sys.path:
    sys.path.insert(0, DRIVER_DIRECTORY)

from motion_supervisor_core import MotionSupervisorCore  # noqa: E402


class UnifiedForceLimitTest(unittest.TestCase):
    """验证制动标志不再切换另一组最大输出或步长参数。"""

    def _core(self):
        return MotionSupervisorCore({
            'max_tx_positive': 2000.0,
            'max_tx_negative': 2250.0,
            'max_ty_positive': 4000.0,
            'max_ty_negative': 6600.0,
            'max_mz_positive': 3000.0,
            'max_mz_negative': 3500.0,
            'force_slew_per_cycle': 1000.0,
            'brake_acceleration_tx_positive': 0.04,
            'brake_acceleration_tx_negative': 0.03,
        })

    def test_braking_uses_same_per_cycle_slew(self):
        core = self._core()

        tx, ty, mz = core._limited_forces(
            5000.0,
            -7000.0,
            5000.0,
            x_braking=True,
            y_braking=True,
            yaw_braking=True,
        )

        self.assertEqual((1000, -1000, 1000), (tx, ty, mz))

    def test_braking_uses_same_directional_limits(self):
        core = self._core()
        core.last_tx = 2000.0
        core.last_ty = -6600.0
        core.last_mz = -3500.0

        tx, ty, mz = core._limited_forces(
            5000.0,
            -7000.0,
            -5000.0,
            x_braking=True,
            y_braking=True,
            yaw_braking=True,
        )

        self.assertEqual((2000, -6600, -3500), (tx, ty, mz))

    def test_brake_feedforward_uses_unified_maximum_force(self):
        core = self._core()

        positive = core._acceleration_force_feedforward(
            'tx', 0.04, 'brake_acceleration_tx')
        negative = core._acceleration_force_feedforward(
            'tx', -0.03, 'brake_acceleration_tx')

        self.assertEqual(2000.0, positive)
        self.assertEqual(-2250.0, negative)

    def test_obsolete_limit_parameters_are_not_defaults(self):
        core = self._core()

        self.assertNotIn('brake_max_tx_positive', core.parameters)
        self.assertNotIn('brake_max_ty_negative', core.parameters)
        self.assertNotIn('brake_max_mz_negative', core.parameters)
        self.assertNotIn('brake_force_slew_per_cycle', core.parameters)


if __name__ == '__main__':
    unittest.main()
