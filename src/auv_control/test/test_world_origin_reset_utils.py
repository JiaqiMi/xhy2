#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_world_origin_reset_utils.py
功能：验证红色圆形原点重置的目标筛选与鲁棒样本估计
作者：Codex
订阅：无
发布：无
记录：
2026.7.13
    新增类别、置信度、深度、离群值与离散样本测试。
"""

import os
import sys
import unittest

import numpy as np

SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from world_origin_reset_utils import (
    RobustPointEstimator,
    is_matching_target,
    is_valid_camera_point,
)


class WorldOriginResetUtilsTest(unittest.TestCase):
    def test_target_filters(self):
        self.assertTrue(is_matching_target("red", "center", 0.7, "red", 0.7))
        self.assertFalse(is_matching_target("blue", "center", 0.9, "red", 0.7))
        self.assertFalse(is_matching_target("red", "edge", 0.9, "red", 0.7))
        self.assertFalse(is_matching_target("red", "center", 0.69, "red", 0.7))
        self.assertTrue(is_valid_camera_point((0.1, -0.2, 1.0), 0.1, 3.0))
        self.assertFalse(is_valid_camera_point((0.1, 0.2, np.nan), 0.1, 3.0))
        self.assertFalse(is_valid_camera_point((0.1, 0.2, 3.1), 0.1, 3.0))

    def test_single_outlier_is_rejected(self):
        estimator = RobustPointEstimator(10, 8, 0.20, 0.03)
        stable_points = [
            (2.00, -1.00, 0.50), (2.01, -1.01, 0.50),
            (1.99, -0.99, 0.51), (2.00, -1.00, 0.49),
            (2.01, -1.00, 0.50), (2.00, -1.01, 0.50),
            (1.99, -1.00, 0.50), (2.00, -0.99, 0.50),
            (2.01, -1.00, 0.49),
        ]
        for point in stable_points:
            self.assertIsNone(estimator.add(point))
        estimate = estimator.add((4.0, 2.0, 1.5))
        np.testing.assert_allclose(estimate, (2.0, -1.0, 0.5), atol=0.02)

    def test_scattered_points_do_not_form_candidate(self):
        estimator = RobustPointEstimator(10, 8, 0.20, 0.03)
        points = [
            (0.0, 0.0, 1.0), (0.4, 0.0, 1.0), (0.0, 0.4, 1.0),
            (-0.4, 0.0, 1.0), (0.0, -0.4, 1.0), (0.3, 0.3, 1.0),
            (-0.3, 0.3, 1.0), (0.3, -0.3, 1.0), (-0.3, -0.3, 1.0),
            (0.5, 0.5, 1.0),
        ]
        for point in points:
            self.assertIsNone(estimator.add(point))


if __name__ == "__main__":
    unittest.main()
