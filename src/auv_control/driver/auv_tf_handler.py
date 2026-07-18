#! /home/xhy/xhy_env36/bin/python
# -*- coding: utf-8 -*-
"""
名称：auv_tf_handler.py
功能：完成机器人坐标到世界坐标的转换
作者：buyegaid
监听：/status/auv (AUVData.msg)
      /cmd/pose/ned (PoseNEDcmd.msg)
      /world_origin (NavSatFix.msg)
发布：/cmd/pose/lla (PoseLLAcmd.msg)
      /tf (from control_link to map)
记录：
2025.7.19 10:56
    第一版完成
2025.7.19 15:21
    控制指令改为直接发布AUVPose消息，不再控制舵机和LED灯
2026.7.13
    新增 /world_origin 更新订阅，收到红色圆形对应的新原点后原子更新坐标换算器。
    新增 PoseNEDcmd（NED）→ PoseLLAcmd（LLA）控制指令转换。
    AUV 状态订阅话题调整为 /status/auv。
2026.7.15
    删除旧版 /target 订阅、/auv_control 发布器及对应转换回调。
    取消旧链路兼容，仅保留 /cmd/pose/ned 到 /cmd/pose/lla 的整包指令转换。
2026.7.18
    增加 base_link 到 IMU/GNSS 定位点的杆臂补偿，同时修正状态 TF 和定点目标。
2026.7.18
    动态导航 TF 改为 map -> control_link；base_link 通过静态刚体关系派生，
    避免把前移后的 base_link 误当作水平旋转中心。
"""

import threading

import numpy as np
import rospy
import tf
from auv_control.msg import AUVData, PoseLLAcmd, PoseNEDcmd
from sensor_msgs.msg import NavSatFix
from tf import transformations

from lever_arm import origin_from_offset_point, sensor_position_from_base
from world_frame import WorldFrameManager


class AUVTfHandler:
    """维护当前 map 原点，并将导航与控制目标转换到对应的坐标系。"""

    def __init__(self):
        self.origin_lock = threading.RLock()
        origin = rospy.wait_for_message('/world_origin', NavSatFix)
        self.wfm = WorldFrameManager(
            origin.latitude, origin.longitude, origin.altitude
        )
        self.origin_values = (origin.latitude, origin.longitude, origin.altitude)
        rospy.loginfo("auv_tf_handler: 初始世界坐标系已就绪: %s", self.origin_values)

        self.tf_broadcaster = tf.TransformBroadcaster()
        self.base_to_imu = (
            float(rospy.get_param('~base_to_imu_x', -0.35)),
            float(rospy.get_param('~base_to_imu_y', 0.0)),
            float(rospy.get_param('~base_to_imu_z', 0.0)),
        )
        if not np.all(np.isfinite(self.base_to_imu)):
            raise ValueError('base_link 到 IMU 的杆臂参数必须为有限值')
        self.control_to_imu = (
            float(rospy.get_param('~control_to_imu_x', 0.0)),
            float(rospy.get_param('~control_to_imu_y', 0.0)),
            float(rospy.get_param('~control_to_imu_z', 0.0)),
        )
        if not np.all(np.isfinite(self.control_to_imu)):
            raise ValueError('control_link 到 IMU 的杆臂参数必须为有限值')
        self.current_pose = None
        self.current_yaw = 0.0
        self.control_cmd_pub = rospy.Publisher(
            '/cmd/pose/lla', PoseLLAcmd, queue_size=10
        )

        rospy.Subscriber('/status/auv', AUVData, self.debug_callback)
        rospy.Subscriber('/cmd/pose/ned', PoseNEDcmd, self.target_cmd_callback)
        # map_initer 是 /world_origin 的唯一发布者；锁存初始值会被安全忽略。
        rospy.Subscriber('/world_origin', NavSatFix, self.origin_callback, queue_size=1)
        rospy.loginfo(
            'auv_tf_handler: control_link -> imu=(%.3f, %.3f, %.3f) m，'
            'base_link -> imu=(%.3f, %.3f, %.3f) m',
            *(self.control_to_imu + self.base_to_imu)
        )

    def origin_callback(self, origin):
        """原子替换坐标换算器，使下一帧 TF 按新原点计算。"""
        values = (origin.latitude, origin.longitude, origin.altitude)
        if not np.all(np.isfinite(values)):
            rospy.logwarn("auv_tf_handler: 忽略包含无效值的世界原点")
            return

        with self.origin_lock:
            if np.allclose(values, self.origin_values, rtol=0.0, atol=1e-9):
                return
            self.wfm = WorldFrameManager(*values)
            self.origin_values = values
        rospy.logwarn("auv_tf_handler: 原点已更新，后续 map 坐标以红圆为零点")

    def get_world_frame_manager(self):
        """获取当前不可变的坐标换算器实例。"""
        with self.origin_lock:
            return self.wfm

    def debug_callback(self, msg):
        """将 AUV 导航位姿转换为当前 map 中的 NED 位姿并发布 TF。"""
        wfm = self.get_world_frame_manager()
        north, east, down = wfm.lld_to_ned(
            msg.pose.latitude, msg.pose.longitude, msg.pose.depth
        )
        self.current_yaw = msg.pose.yaw
        orientation = transformations.quaternion_from_euler(
            np.radians(msg.pose.roll),
            np.radians(msg.pose.pitch),
            np.radians(msg.pose.yaw),
        )
        control_position = origin_from_offset_point(
            (north, east, down),
            orientation,
            self.control_to_imu,
        )
        self.current_pose = [
            control_position[0],
            control_position[1],
            control_position[2],
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
        ]
        self.publish_tf()
        rospy.loginfo_throttle(10, "auv_tf_handler: TF 已发布")

    def target_cmd_callback(self, msg):
        """将 NED 整包控制指令转换为 LLA 整包控制指令。"""
        wfm = self.get_world_frame_manager()
        pose = msg.target.pose
        orientation = (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        sensor_target = sensor_position_from_base(
            (
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ),
            orientation,
            self.base_to_imu,
        )
        latitude, longitude, depth = wfm.ned_to_lld(
            sensor_target[0],
            sensor_target[1],
            sensor_target[2],
        )
        roll, pitch, yaw = tf.transformations.euler_from_quaternion([
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
        ])

        output = PoseLLAcmd()
        output.mode = msg.mode
        output.target.latitude = latitude
        output.target.longitude = longitude
        output.target.depth = depth
        output.target.roll = np.degrees(roll)
        output.target.pitch = np.degrees(pitch)
        output.target.yaw = np.degrees(yaw)
        output.force = msg.force
        self.control_cmd_pub.publish(output)
        rospy.loginfo_throttle(5, "auv_tf_handler: 已发布 /cmd/pose/lla")

    def publish_tf(self):
        """发布当前 map 到 control_link 的 NED 变换。"""
        self.tf_broadcaster.sendTransform(
            tuple(self.current_pose[0:3]),
            tuple(self.current_pose[3:7]),
            rospy.Time.now(),
            'control_link',
            'map',
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        rospy.init_node('auv_tf_handler_node')
        AUVTfHandler().run()
    except rospy.ROSInterruptException:
        pass
