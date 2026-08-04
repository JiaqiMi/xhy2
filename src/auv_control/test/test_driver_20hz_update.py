#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：test_driver_20hz_update.py
功能：验证 nav/DVL/debug 融合字段和 actuator mode2 下发协议
作者：buyegaid
监听：无
发布：无
记录：
2026.8.5
    新增 INS/DVL 有效性、IMU 角速度直通、debug 补充和 mode2 组帧测试。
"""

import math
import os
import sys
import threading
import types
import unittest


DRIVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'driver'))
if DRIVER_DIR not in sys.path:
    sys.path.insert(0, DRIVER_DIR)


class _TimeValue:
    @staticmethod
    def now():
        return _TimeValue()


class _Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ''


class _Pose:
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.depth = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.speed = 0.0


class _Motor:
    def __init__(self):
        self.TX = self.TY = self.TZ = 0
        self.MX = self.MY = self.MZ = 0


class _Sensor:
    def __init__(self):
        self.temperature = 0.0
        self.voltage = 0.0
        self.current = 0.0
        self.battery = 0
        self.leak_alarm = False
        self.sensor_valid = 0
        self.sensor_updated = 0
        self.fault_status = 0
        self.power_status = 0


class _AuvTime:
    def __init__(self):
        self.year = self.month = self.day = 0
        self.hour = self.minute = 0
        self.second = 0.0


class _AUVData:
    def __init__(self):
        self.header = _Header()
        self.control_mode = 0
        self.pose = _Pose()
        self.target = _Pose()
        self.time = _AuvTime()
        self.motor_force = _Motor()
        self.linear_velocity = [0.0, 0.0, 0.0]
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.sensor = _Sensor()


class _ActuatorControl:
    pass


def _install_ros_stubs():
    rospy = types.ModuleType('rospy')
    rospy.Time = _TimeValue
    rospy.logwarn_throttle = lambda *args, **kwargs: None
    rospy.logerr_throttle = lambda *args, **kwargs: None
    rospy.logerr = lambda *args, **kwargs: None
    rospy.logwarn = lambda *args, **kwargs: None
    rospy.loginfo = lambda *args, **kwargs: None
    sys.modules.setdefault('rospy', rospy)

    auv_control = types.ModuleType('auv_control')
    auv_msg = types.ModuleType('auv_control.msg')
    auv_msg.AUVData = _AUVData
    auv_msg.NavData = type('NavData', (), {})
    auv_msg.PoseLLAcmd = type('PoseLLAcmd', (), {})
    auv_msg.ActuatorControl = _ActuatorControl
    auv_control.msg = auv_msg
    sys.modules.setdefault('auv_control', auv_control)
    sys.modules.setdefault('auv_control.msg', auv_msg)

    geometry_msgs = types.ModuleType('geometry_msgs')
    geometry_msg = types.ModuleType('geometry_msgs.msg')
    geometry_msg.TwistStamped = type('TwistStamped', (), {})
    geometry_msgs.msg = geometry_msg
    sys.modules.setdefault('geometry_msgs', geometry_msgs)
    sys.modules.setdefault('geometry_msgs.msg', geometry_msg)

    std_msgs = types.ModuleType('std_msgs')
    std_msgs_msg = types.ModuleType('std_msgs.msg')
    std_msgs_msg.Header = _Header
    std_msgs.msg = std_msgs_msg
    sys.modules.setdefault('std_msgs', std_msgs)
    sys.modules.setdefault('std_msgs.msg', std_msgs_msg)

    genpy = types.ModuleType('genpy')
    genpy.Message = type('Message', (), {})
    sys.modules.setdefault('genpy', genpy)


_install_ros_stubs()

from debug_driver_v2 import DebugDataPacket, DebugDriverV2  # noqa: E402
from nav_driver import (  # noqa: E402
    DvlBottomSample,
    DvlBottomTracker,
    angular_velocity_rad,
    imu_angular_velocity_deg,
    ins_data_valid,
)
from sensor_actuator_node import SensorActuatorNode  # noqa: E402


class Driver20HzUpdateTest(unittest.TestCase):

    def test_ins_valid_bit(self):
        self.assertTrue(ins_data_valid(0x60A0))
        self.assertFalse(ins_data_valid(0x20A0))
        self.assertFalse(ins_data_valid(None))

    def test_dvl_bottom_speed_latch_and_timeout(self):
        tracker = DvlBottomTracker(0.3)
        self.assertFalse(tracker.update(1, (9.0, 9.0, 9.0), 9.0, 1.0))
        self.assertIsNone(tracker.current(1.0))
        self.assertTrue(tracker.update(2, (0.1, -0.2, 0.3), 1.5, 1.1))
        self.assertFalse(tracker.update(1, (8.0, 8.0, 8.0), 8.0, 1.2))
        self.assertEqual(tracker.current(1.39).velocity, (0.1, -0.2, 0.3))
        self.assertIsNone(tracker.current(1.41))
        self.assertTrue(tracker.update(4, (0.4, 0.5, 0.6), 2.0, 1.5))
        self.assertEqual(tracker.current(1.5).altitude, 2.0)
        self.assertFalse(tracker.update(2, ('bad', 0.0, 0.0), 1.0, 1.6))
        self.assertEqual(tracker.current(1.6).velocity, (0.4, 0.5, 0.6))

        alternating = DvlBottomTracker(0.3)
        for index in range(20):
            stamp = index * 0.05
            status = 2 if index % 2 == 0 else 1
            alternating.update(status, (0.1, 0.2, 0.3), 1.0, stamp)
            self.assertIsNotNone(alternating.current(stamp))

    def test_imu_angular_velocity_is_direct_and_only_changes_unit(self):
        nav = types.SimpleNamespace(gyro_x=1.0, gyro_y=-2.0, gyro_z=3.0)
        angular_deg = imu_angular_velocity_deg(nav)
        self.assertEqual(angular_deg, (1.0, -2.0, 3.0))
        angular_rad = angular_velocity_rad(angular_deg)
        self.assertAlmostEqual(angular_rad[0], math.radians(1.0))
        self.assertAlmostEqual(angular_rad[1], math.radians(-2.0))
        self.assertAlmostEqual(angular_rad[2], math.radians(3.0))

    def test_nav_fields_override_debug_and_debug_only_supplements(self):
        nav = types.SimpleNamespace(
            latitude=23.1,
            longitude=113.2,
            depth=-0.8,
            roll=1.0,
            pitch=2.0,
            heading=3.0,
            gyro_x=4.0,
            gyro_y=5.0,
            gyro_z=6.0,
        )
        debug = DebugDataPacket()
        debug.mode = 2
        debug.navigation_coords = [99.0, 88.0]
        debug.depth_filtered = 77.0
        debug.euler_angles = [66.0, 55.0, 44.0]
        debug.linear_velocity = [33.0, 22.0, 11.0]
        debug.angular_velocity = [10.0, 20.0, 30.0]
        debug.target_longitude = 120.0
        debug.target_latitude = 30.0
        debug.target_depth = 2.5
        debug.force_commands = [1, 2, 3, 4, 5, 6]
        debug.sensor_status = 0x12
        dvl = DvlBottomSample((0.11, -0.22, 0.33), 1.25, 10.0, 0.05)
        node = object.__new__(DebugDriverV2)

        message, angular = node._build_auv_data(nav, debug, dvl, 'stamp')

        self.assertEqual(message.header.stamp, 'stamp')
        self.assertEqual(message.pose.latitude, 23.1)
        self.assertEqual(message.pose.longitude, 113.2)
        self.assertEqual(message.pose.depth, -0.8)
        self.assertEqual(message.pose.yaw, 3.0)
        self.assertEqual(message.linear_velocity, [0.11, -0.22, 0.33])
        self.assertEqual(message.angular_velocity, [4.0, 5.0, 6.0])
        self.assertEqual(angular, (4.0, 5.0, 6.0))
        self.assertEqual(message.control_mode, 2)
        self.assertEqual(message.target.longitude, 120.0)
        self.assertEqual(message.motor_force.MZ, 6)
        self.assertEqual(message.sensor.sensor_valid, 0x12)

        safe_message, _ = node._build_auv_data(nav, None, dvl, 'stamp')
        self.assertEqual(safe_message.control_mode, 0)
        self.assertEqual(safe_message.motor_force.MZ, 0)
        self.assertEqual(safe_message.sensor.sensor_valid, 0)

    def test_actuator_frame_keeps_all_mode2_fields(self):
        node = object.__new__(SensorActuatorNode)
        node.seq = 0
        frame = node.build_actuator_frame(128, 255, 2, 100, 1, 0, 1)
        self.assertEqual(len(frame), 54)
        self.assertEqual(frame[0:2], b'\xFE\xFE')
        self.assertEqual(frame[4], SensorActuatorNode.CMD_ACTUATOR)
        self.assertEqual(list(frame[8:15]), [128, 255, 2, 100, 1, 0, 1])
        self.assertEqual(frame[51], node._calc_downlink_xor(frame))
        self.assertEqual(frame[52:54], b'\xFD\xFD')
        self.assertEqual(SensorActuatorNode.ACTUATOR_SEND_RATE_HZ, 5.0)
        self.assertFalse(hasattr(SensorActuatorNode, 'build_camera_light_frame'))

    def test_mode1_does_not_update_light_cache(self):
        node = object.__new__(SensorActuatorNode)
        node.lock = threading.Lock()
        node.light1 = 0
        node.light2 = 0
        node._save_command_record = lambda *args, **kwargs: None
        command = types.SimpleNamespace(mode=1, light1=80, light2=90)
        node.actuator_callback(command)
        self.assertEqual((node.light1, node.light2), (0, 0))


if __name__ == '__main__':
    unittest.main()
