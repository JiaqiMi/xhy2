#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：rotation_center_continuous_calibration.py
功能：按低、中、高三档执行正负方向连续三圈旋转中心标定
作者：BroXu
监听：
    /status/vel (geometry_msgs/TwistStamped)
    /status/auv (AUVData.msg)
    /tf
发布：
    /cmd/pose/ned (PoseNEDcmd.msg)
说明：
    1. 运行时必须停止 motion_supervisor，本节点独占 /cmd/pose/ned；
    2. 每个档位先正向、再负向，各连续旋转三圈，中途不执行 90° 刹停；
    3. 每段仅在三圈目标结束时主动刹停，然后 mode=4 返回同一初始位置；
    4. 返回位置并完成判稳后连续稳定 12 s，才进入下一档位或方向；
    5. 低、中、高三档共六个连续旋转段，TRACK 样本按档位和方向独立拟合。
记录：
2026.7.19
    新增不同力矩档位下连续三圈旋转、末端刹停和定点回归标定流程。
"""

from __future__ import division

import math
import os
import sys

import rospy
from rosgraph.masterapi import ROSMasterException


TEST_DIR = os.path.abspath(os.path.dirname(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

import rotation_center_calibration as calibration_module  # noqa: E402
from rotation_center_calibration_core import (  # noqa: E402
    build_continuous_rotation_steps,
)


NODE_NAME = 'rotation_center_continuous_calibration'
calibration_module.NODE_NAME = NODE_NAME


class ContinuousRotationCenterCalibration(
        calibration_module.RotationCenterCalibration):
    """执行六段连续多圈自由偏航并拟合正负方向虚拟旋转中心。"""

    LOG_STEM = 'rotation_center_continuous'
    RESULT_GENERATOR = 'rotation_center_continuous_calibration.py'

    def __init__(self):
        super(ContinuousRotationCenterCalibration, self).__init__()
        self.continuous_turns = int(
            rospy.get_param('~continuous_turns', 3))
        self.continuous_step_timeout = float(
            rospy.get_param('~continuous_step_timeout', 600.0))
        if (
                self.continuous_turns <= 0
                or not math.isfinite(self.continuous_step_timeout)
                or self.continuous_step_timeout <= 0.0):
            raise ValueError('连续旋转圈数和单段超时必须为有限正数')
        self.continuous_steps = build_continuous_rotation_steps(
            self.torque_profiles,
            turns=self.continuous_turns,
        )
        self.total_segments = len(self.continuous_steps)

    def run(self):
        profile_description = ', '.join(
            '{}(+{:.0f}/-{:.0f})'.format(
                profile.name,
                profile.positive_limit,
                profile.negative_limit,
            )
            for profile in self.torque_profiles
        )
        rospy.logwarn(
            '%s: 本程序将独占 /cmd/pose/ned；TX=TY 始终为 0；'
            '每档正负方向各连续 %d 圈，仅在末端刹停；'
            '力矩档位=%s；共 %d 段；原始日志: %s',
            NODE_NAME,
            self.continuous_turns,
            profile_description,
            self.total_segments,
            self.log_path,
        )
        self._prepare_calibration()

        for step in self.continuous_steps:
            direction_name = '正向' if step.direction > 0 else '负向'
            description = '{}连续 {} 圈'.format(
                direction_name, step.turns)
            rospy.logwarn(
                '%s: 开始第 %d/%d 段，%s档%s，'
                'TRACK MZ(+%.0f/-%.0f)',
                NODE_NAME,
                step.segment,
                self.total_segments,
                step.profile.name,
                description,
                step.profile.positive_limit,
                step.profile.negative_limit,
            )
            target_yaw = self._rotate_segment(
                step.profile,
                step.segment,
                step.direction,
                step.turns,
                0,
                turn_angle=step.turn_angle,
                rotation_timeout=self.continuous_step_timeout,
                description=description,
            )
            self._hold_locked_position(
                self.handover_hold_seconds,
                'CONTINUOUS_SEGMENT_HOLD',
                target_yaw,
                step.profile,
                step.segment,
                step.direction,
                step.turns,
                0,
            )

        self._finalize_calibration()


def main():
    rospy.init_node(NODE_NAME)
    calibration = None
    try:
        calibration = ContinuousRotationCenterCalibration()
        calibration.run()
    except (
            ValueError,
            RuntimeError,
            OSError,
            IOError,
            ROSMasterException) as error:
        rospy.logfatal('%s: %s', NODE_NAME, error)
        if calibration is not None:
            calibration.emergency_hold()
        raise SystemExit(1)


if __name__ == '__main__':
    main()
