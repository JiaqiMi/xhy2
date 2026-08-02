#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_motion_supervisor_thruster_recovery.py
功能：验证 TY/MZ 协议级侧推异常检测、反向恢复与状态边界
作者：BroXu
监听：无
发布：无
记录：
2026.8.2
    新增空载功率基线、异常防误触发、反向比例、恢复时序及中断条件测试。
"""

from __future__ import division

import os
import sys
import unittest


DRIVER_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'driver'))
if DRIVER_DIR not in sys.path:
    sys.path.insert(0, DRIVER_DIR)

from motion_supervisor_core import (  # noqa: E402
    MODE_DEPTH,
    HOVER,
    RECOVERY_RECHECK_DELAY,
    RECOVERY_REVERSE_PULSE,
    RECOVERY_ZERO_BEFORE_TRACK,
    SAFE,
    THRUSTER_RECOVERY,
    TRANSLATE,
    TRANSLATE_BRAKE,
    MotionGoal,
    MotionSupervisorCore,
    VehicleState,
)


def vehicle(
        now, power=16.0, lateral_velocity=0.0, yaw_rate=0.0,
        feedback_fresh=True, startup_complete=True):
    """构造带有效动力功率反馈的静止测试状态。"""
    return VehicleState(
        now,
        x=0.0,
        y=0.0,
        z=-1.2,
        yaw=0.0,
        forward_velocity=0.0,
        lateral_velocity=lateral_velocity,
        yaw_rate=yaw_rate,
        feedback_fresh=feedback_fresh,
        startup_complete=startup_complete,
        power_feedback_fresh=True,
        power2_valid=True,
        power2_power=power,
    )


class ThrusterRecoveryTest(unittest.TestCase):
    """验证侧推恢复只依赖 TY/MZ、动力功率和运动响应。"""

    def setUp(self):
        self.core = MotionSupervisorCore({
            'max_ty_positive': 4000.0,
            'max_ty_negative': 6000.0,
            'max_mz_positive': 3000.0,
            'max_mz_negative': 3500.0,
            'force_slew_per_cycle': 10000.0,
        })
        self.core.goal = MotionGoal(5.0, 2.0, -1.2, 1.0)
        self.core.state = TRANSLATE
        self._establish_idle_baseline()

    def _establish_idle_baseline(self):
        for now in (0.0, 0.5, 1.0, 1.5, 2.0, 2.1):
            self.core._output(
                vehicle(now), MODE_DEPTH, immediate_zero=True)
        self.assertAlmostEqual(self.core.thruster_power_baseline, 16.0)

    def _fault_until_triggered(self, ty=4000.0, mz=2655.0, power=20.0):
        output = None
        triggered = False
        now = 3.0
        while now <= 5.01:
            state = vehicle(now, power=power)
            output = self.core._output(
                state, MODE_DEPTH, ty=ty, mz=mz)
            triggered = self.core._evaluate_thruster_fault(state, output)
            if triggered:
                break
            now += 0.25
        return now, output, triggered

    def test_combined_fault_triggers_after_two_seconds(self):
        now, output, triggered = self._fault_until_triggered()

        self.assertTrue(triggered)
        self.assertGreaterEqual(now, 5.0)
        self.assertEqual(self.core.state, THRUSTER_RECOVERY)
        self.assertAlmostEqual(self.core.thruster_fault_ty, output.ty)
        self.assertAlmostEqual(self.core.thruster_fault_mz, output.mz)
        recovery_output = self.core._thruster_recovery_output(vehicle(now))
        self.assertGreaterEqual(
            recovery_output.diagnostics['thruster_fault_wait_s'], 2.0)

    def test_detection_requires_valid_context_and_power(self):
        scenarios = (
            {'startup_complete': False},
            {'power_feedback_fresh': False},
            {'power2_valid': False},
        )
        for overrides in scenarios:
            state = vehicle(3.0)
            for name, value in overrides.items():
                setattr(state, name, value)
            output = self.core._output(
                state, MODE_DEPTH, ty=4000.0, mz=2655.0)
            self.assertFalse(output.diagnostics['thruster_fault_requested'])

        output = self.core._output(
            vehicle(3.0), MODE_DEPTH, tx=501.0,
            ty=4000.0, mz=2655.0)
        self.assertFalse(output.diagnostics['thruster_fault_requested'])
        for excluded_state in (TRANSLATE_BRAKE, HOVER, SAFE):
            self.core.state = excluded_state
            output = self.core._output(
                vehicle(3.0), MODE_DEPTH, ty=4000.0, mz=2655.0)
            self.assertFalse(output.diagnostics['thruster_fault_requested'])
        self.core.state = TRANSLATE

    def test_high_power_or_any_required_motion_prevents_false_trigger(self):
        scenarios = (
            {'power': 35.0, 'lateral_velocity': 0.0, 'yaw_rate': 0.0},
            {'power': 20.0, 'lateral_velocity': 0.02, 'yaw_rate': 0.0},
            {'power': 20.0, 'lateral_velocity': 0.0,
             'yaw_rate': -0.01},
        )
        for scenario in scenarios:
            core = MotionSupervisorCore(self.core.parameters)
            core.goal = MotionGoal(5.0, 2.0, -1.2, 1.0)
            core.state = TRANSLATE
            for now in (0.0, 0.5, 1.0, 1.5, 2.0, 2.1):
                core._output(vehicle(now), MODE_DEPTH, immediate_zero=True)
            triggered = False
            now = 3.0
            while now <= 5.5:
                state = vehicle(now, **scenario)
                output = core._output(
                    state, MODE_DEPTH, ty=4000.0, mz=2655.0)
                triggered = core._evaluate_thruster_fault(state, output)
                now += 0.25
            self.assertFalse(triggered, scenario)
            self.assertEqual(core.state, TRANSLATE, scenario)

    def test_pure_ty_and_pure_mz_faults_are_detected(self):
        for ty, mz in ((4000.0, 0.0), (0.0, 3000.0)):
            core = MotionSupervisorCore(self.core.parameters)
            core.goal = MotionGoal(5.0, 2.0, -1.2, 1.0)
            core.state = TRANSLATE
            for now in (0.0, 0.5, 1.0, 1.5, 2.0, 2.1):
                core._output(vehicle(now), MODE_DEPTH, immediate_zero=True)
            triggered = False
            for index in range(10):
                now = 3.0 + 0.25 * index
                state = vehicle(now, power=20.0)
                output = core._output(
                    state, MODE_DEPTH, ty=ty, mz=mz)
                triggered = core._evaluate_thruster_fault(state, output)
                if triggered:
                    break
            self.assertTrue(triggered, (ty, mz))

    def test_reverse_preserves_ratio_and_obeys_both_limits(self):
        now, unused_output, triggered = self._fault_until_triggered()
        self.assertTrue(triggered)
        del unused_output

        zero_output = self.core._thruster_recovery_output(vehicle(now))
        self.assertEqual((zero_output.tx, zero_output.ty, zero_output.mz),
                         (0, 0, 0))
        reverse_output = self.core._thruster_recovery_output(
            vehicle(now + 1.5))

        self.assertEqual(self.core.thruster_recovery_phase,
                         RECOVERY_REVERSE_PULSE)
        self.assertLessEqual(abs(reverse_output.ty), 2000)
        self.assertLessEqual(abs(reverse_output.mz), 1500)
        self.assertLess(reverse_output.ty * self.core.thruster_fault_ty, 0.0)
        self.assertLess(reverse_output.mz * self.core.thruster_fault_mz, 0.0)
        self.assertAlmostEqual(
            self.core.thruster_reverse_ty / self.core.thruster_fault_ty,
            self.core.thruster_reverse_mz / self.core.thruster_fault_mz,
        )

    def test_power_and_motion_response_ends_reverse_early(self):
        now, unused_output, triggered = self._fault_until_triggered()
        self.assertTrue(triggered)
        del unused_output
        self.core._thruster_recovery_output(vehicle(now, power=16.0))
        self.core._thruster_recovery_output(vehicle(now + 1.5, power=16.0))
        self.core._thruster_recovery_output(vehicle(
            now + 1.60, power=35.0, lateral_velocity=-0.02))
        output = self.core._thruster_recovery_output(vehicle(
            now + 1.86, power=35.0, lateral_velocity=-0.02))

        self.assertEqual(self.core.thruster_recovery_phase,
                         RECOVERY_ZERO_BEFORE_TRACK)
        self.assertTrue(self.core.thruster_recovery_preliminary_success)
        self.assertEqual((output.tx, output.ty, output.mz), (0, 0, 0))

    def test_wrong_direction_motion_is_not_recovery_evidence(self):
        now, unused_output, triggered = self._fault_until_triggered()
        self.assertTrue(triggered)
        del unused_output
        self.core._thruster_recovery_output(vehicle(now, power=16.0))
        self.core._thruster_recovery_output(vehicle(now + 1.5, power=16.0))
        output = self.core._thruster_recovery_output(vehicle(
            now + 1.80, power=35.0, lateral_velocity=0.02))

        self.assertEqual(self.core.thruster_recovery_phase,
                         RECOVERY_REVERSE_PULSE)
        self.assertTrue(output.diagnostics['thruster_recovery_power_response'])
        self.assertFalse(
            output.diagnostics['thruster_recovery_lateral_response'])
        self.assertFalse(
            output.diagnostics['thruster_recovery_preliminary_success'])

    def test_no_response_still_returns_to_latest_goal_and_rechecks(self):
        now, unused_output, triggered = self._fault_until_triggered()
        self.assertTrue(triggered)
        del unused_output
        latest_goal = MotionGoal(8.0, -3.0, -1.6, -0.5)
        self.core.set_goal(latest_goal)
        self.core._thruster_recovery_output(vehicle(now, power=16.0))
        self.core._thruster_recovery_output(vehicle(now + 1.5, power=16.0))
        zero_output = self.core._thruster_recovery_output(
            vehicle(now + 2.51, power=16.0))
        self.assertEqual(self.core.thruster_recovery_phase,
                         RECOVERY_ZERO_BEFORE_TRACK)
        self.assertFalse(self.core.thruster_recovery_preliminary_success)
        self.assertEqual((zero_output.tx, zero_output.ty, zero_output.mz),
                         (0, 0, 0))

        track_output = self.core._thruster_recovery_output(
            vehicle(now + 3.02, power=16.0))
        self.assertEqual(self.core.state, TRANSLATE)
        self.assertEqual(self.core.thruster_recovery_phase,
                         RECOVERY_RECHECK_DELAY)
        self.assertEqual(track_output.mode, MODE_DEPTH)
        self.assertIs(track_output.target, latest_goal)
        self.assertAlmostEqual(track_output.target.z, -1.6)
        self.assertTrue(track_output.diagnostics['thruster_recheck_active'])

    def test_cancel_and_feedback_timeout_interrupt_recovery(self):
        self._fault_until_triggered()
        self.core.cancel()
        self.assertEqual(self.core.state, TRANSLATE_BRAKE)
        self.assertEqual(self.core.thruster_recovery_phase, '')

        other = MotionSupervisorCore(self.core.parameters)
        other.goal = MotionGoal(5.0, 2.0, -1.2, 1.0)
        other.state = THRUSTER_RECOVERY
        other.thruster_recovery_phase = RECOVERY_REVERSE_PULSE
        other.thruster_recovery_phase_started_at = 1.0
        output = other.step(vehicle(1.1, feedback_fresh=False))
        self.assertEqual(other.state, SAFE)
        self.assertEqual(other.thruster_recovery_phase, '')
        self.assertEqual(output.state, SAFE)


if __name__ == '__main__':
    unittest.main()
