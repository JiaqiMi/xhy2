#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_state_web_core.py
功能：state_web 纯数据处理单元测试
记录：
2026.7.17
    覆盖角度、姿态、超时、FPS、原点版本和无效数值处理。
"""

import math
import os
import sys
import unittest


SCRIPT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scripts")
)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from state_web_core import (  # noqa: E402
    OriginRevision,
    health_state,
    horizon_transform,
    has_vision_detections,
    normalize_heading,
    quaternion_to_euler_deg,
    sanitize_json,
    select_attitude,
    shortest_heading_error,
    update_fps,
    vision_packet_status,
)


class StateWebCoreTest(unittest.TestCase):

    def test_heading_normalization(self):
        self.assertAlmostEqual(normalize_heading(361.5), 1.5)
        self.assertAlmostEqual(normalize_heading(-1.0), 359.0)
        self.assertIsNone(normalize_heading(float("nan")))

    def test_shortest_heading_error_crosses_zero(self):
        self.assertAlmostEqual(shortest_heading_error(1.0, 359.0), 2.0)
        self.assertAlmostEqual(shortest_heading_error(359.0, 1.0), -2.0)
        self.assertAlmostEqual(shortest_heading_error(180.0, 0.0), -180.0)

    def test_quaternion_to_euler(self):
        yaw = math.radians(90.0)
        result = quaternion_to_euler_deg(
            0.0,
            0.0,
            math.sin(yaw / 2.0),
            math.cos(yaw / 2.0),
        )
        self.assertAlmostEqual(result["roll_deg"], 0.0, places=6)
        self.assertAlmostEqual(result["pitch_deg"], 0.0, places=6)
        self.assertAlmostEqual(result["heading_deg"], 90.0, places=6)
        self.assertIsNone(quaternion_to_euler_deg(0.0, 0.0, 0.0, 0.0))

    def test_health_timeout(self):
        online = health_state(10.0, 11.0, 2.0)
        stale = health_state(10.0, 13.0, 2.0)
        persistent = health_state(10.0, 100.0, None)
        self.assertTrue(online["online"])
        self.assertFalse(stale["online"])
        self.assertTrue(persistent["online"])
        self.assertAlmostEqual(stale["age_sec"], 3.0)

    def test_fps_exponential_average(self):
        first = update_fps(0.0, 10.0, 10.2)
        second = update_fps(first, 10.2, 10.3, alpha=0.1)
        self.assertAlmostEqual(first, 5.0)
        self.assertAlmostEqual(second, 5.5)
        self.assertAlmostEqual(update_fps(second, 10.3, 10.3), second)

    def test_attitude_source_priority_and_stale_fallback(self):
        feedback = {
            "valid": True,
            "source": "status_auv",
            "received_at": 10.0,
        }
        tf_pose = {
            "valid": True,
            "source": "tf",
            "received_at": 11.0,
        }
        self.assertIs(select_attitude(feedback, tf_pose), feedback)

        feedback["valid"] = False
        self.assertIs(select_attitude(feedback, tf_pose), tf_pose)

        tf_pose["valid"] = False
        self.assertIs(select_attitude(feedback, tf_pose), tf_pose)

    def test_horizon_transform_uses_opposite_roll(self):
        result = horizon_transform(
            roll_deg=20.0,
            pitch_deg=60.0,
            pixels_per_degree=3.0,
            pitch_limit_deg=45.0,
        )
        self.assertAlmostEqual(result["rotation_deg"], -20.0)
        self.assertAlmostEqual(result["clamped_pitch_deg"], 45.0)
        self.assertAlmostEqual(result["offset_px"], 135.0)

    def test_origin_revision_only_changes_for_new_origin(self):
        tracker = OriginRevision()
        changed, revision = tracker.update(30.0, 120.0, 2.0)
        self.assertTrue(changed)
        self.assertEqual(revision, 1)

        changed, revision = tracker.update(30.0, 120.0, 2.0)
        self.assertFalse(changed)
        self.assertEqual(revision, 1)

        changed, revision = tracker.update(30.0, 120.0, 2.5)
        self.assertTrue(changed)
        self.assertEqual(revision, 2)

    def test_sanitize_json_replaces_nonfinite_values(self):
        result = sanitize_json({
            "valid": 1.0,
            "nan": float("nan"),
            "values": [float("inf"), 2.0],
        })
        self.assertEqual(result["valid"], 1.0)
        self.assertIsNone(result["nan"])
        self.assertIsNone(result["values"][0])

    def test_vision_packet_requires_fresh_and_synchronized_frame(self):
        packet = {
            "received_at": 10.0,
            "payload": {"stamp": 9.8, "detections": [{}]},
        }
        fresh = vision_packet_status(
            packet,
            now=11.0,
            timeout=2.0,
            frame_stamp=10.0,
            frame_tolerance=0.5,
        )
        self.assertTrue(fresh["online"])
        self.assertTrue(fresh["frame_synced"])
        self.assertAlmostEqual(fresh["frame_delta_sec"], 0.2)

        unsynchronized = vision_packet_status(
            packet,
            now=11.0,
            timeout=2.0,
            frame_stamp=10.5,
            frame_tolerance=0.5,
        )
        self.assertFalse(unsynchronized["online"])
        self.assertFalse(unsynchronized["frame_synced"])

        stale = vision_packet_status(
            packet,
            now=13.0,
            timeout=2.0,
            frame_stamp=10.0,
            frame_tolerance=0.5,
        )
        self.assertFalse(stale["online"])

    def test_vision_detection_payload_requires_nonempty_list(self):
        self.assertTrue(has_vision_detections({"detections": [{}]}))
        self.assertFalse(has_vision_detections({
            "valid": False,
            "detections": [{}],
        }))
        self.assertFalse(has_vision_detections({"detections": []}))
        self.assertFalse(has_vision_detections({"detections": {}}))
        self.assertFalse(has_vision_detections(None))


if __name__ == "__main__":
    unittest.main()
