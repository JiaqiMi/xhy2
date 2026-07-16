#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_world_frame.py
功能：验证 WorldFrameManager 的坐标换算与原点重置精度
作者：Codex
订阅：无
发布：无
记录：
2026.7.13
    新增世界坐标系原点与重置后零点回归测试。
"""

import os
import sys
import unittest

DRIVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "driver"))
if DRIVER_DIR not in sys.path:
    sys.path.insert(0, DRIVER_DIR)

from world_frame import WorldFrameManager


class WorldFrameManagerTest(unittest.TestCase):
    def test_origin_is_zero(self):
        frame = WorldFrameManager(30.0, 120.0, 5.0)
        north, east, down = frame.lld_to_ned(30.0, 120.0, 5.0)
        self.assertAlmostEqual(north, 0.0, places=6)
        self.assertAlmostEqual(east, 0.0, places=6)
        self.assertAlmostEqual(down, 0.0, places=6)

    def test_rebased_target_is_zero(self):
        old_frame = WorldFrameManager(30.0, 120.0, 5.0)
        target_lld = old_frame.ned_to_lld(8.0, -3.0, 1.5)
        new_frame = WorldFrameManager(*target_lld)
        north, east, down = new_frame.lld_to_ned(*target_lld)
        self.assertAlmostEqual(north, 0.0, places=5)
        self.assertAlmostEqual(east, 0.0, places=5)
        self.assertAlmostEqual(down, 0.0, places=5)

    def test_multiple_rebases_are_supported(self):
        first_frame = WorldFrameManager(30.0, 120.0, 5.0)
        first_origin = first_frame.ned_to_lld(8.0, -3.0, 1.5)
        second_frame = WorldFrameManager(*first_origin)
        second_origin = second_frame.ned_to_lld(-2.0, 4.0, 0.3)
        third_frame = WorldFrameManager(*second_origin)

        north, east, down = third_frame.lld_to_ned(*second_origin)
        self.assertAlmostEqual(north, 0.0, places=5)
        self.assertAlmostEqual(east, 0.0, places=5)
        self.assertAlmostEqual(down, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
