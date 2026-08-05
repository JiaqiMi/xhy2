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
      /tf (map -> base_link)
      /tf_static (base_link -> sensors)
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
2026.7.20
    动态导航 TF 固定为 map -> base_link；方向控制中心全部由静态 TF 派生。
2026.8.5
    /status/auv 订阅改为仅保留最新帧，动态 TF 沿用上游状态时间戳。
    静态 TF 改用锁存广播器，仅在节点初始化时发布一次。
"""

import threading

import numpy as np
import rospy
import tf
import tf2_ros
from auv_control.msg import AUVData, PoseLLAcmd, PoseNEDcmd
from geometry_msgs.msg import TransformStamped
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
        self.static_tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self.static_transforms = self.load_static_transforms()
        self.base_to_imu = self.get_imu_translation(self.static_transforms)
        self.publish_static_transforms()
        self.current_pose = None
        self.control_cmd_pub = rospy.Publisher(
            '/cmd/pose/lla', PoseLLAcmd, queue_size=10
        )

        rospy.Subscriber(
            '/status/auv',
            AUVData,
            self.debug_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.Subscriber('/cmd/pose/ned', PoseNEDcmd, self.target_cmd_callback)
        # map_initer 是 /world_origin 的唯一发布者；锁存初始值会被安全忽略。
        rospy.Subscriber('/world_origin', NavSatFix, self.origin_callback, queue_size=1)
        rospy.loginfo(
            'auv_tf_handler: base_link -> imu=(%.3f, %.3f, %.3f) m',
            *self.base_to_imu
        )

    @staticmethod
    def _vector_from_config(config, field_name, axes=('x', 'y', 'z')):
        """读取三维向量配置并校验其数值有效性。"""
        try:
            vector = tuple(float(config[axis]) for axis in axes)
        except (KeyError, TypeError, ValueError):
            raise ValueError(
                '%s 必须包含可转换为浮点数的 %s'
                % (field_name, '、'.join(axes)))
        if not np.all(np.isfinite(vector)):
            raise ValueError('%s 必须全部为有限值' % field_name)
        return vector

    def load_static_transforms(self):
        """从私有参数加载全部静态 TF，并转换为可直接发布的格式。"""
        configs = rospy.get_param('~static_transforms', None)
        if not isinstance(configs, dict) or not configs:
            raise ValueError('static_transforms 必须是非空字典')

        transforms = []
        child_frames = set()
        for name, config in configs.items():
            if not isinstance(config, dict):
                raise ValueError('静态 TF %s 的配置必须是字典' % name)
            parent_frame = config.get('parent_frame')
            child_frame = config.get('child_frame')
            if not isinstance(parent_frame, str) or not parent_frame:
                raise ValueError('静态 TF %s 缺少 parent_frame' % name)
            if not isinstance(child_frame, str) or not child_frame:
                raise ValueError('静态 TF %s 缺少 child_frame' % name)
            if child_frame in child_frames:
                raise ValueError('静态 TF 子坐标系重复: %s' % child_frame)
            child_frames.add(child_frame)

            translation = self._vector_from_config(
                config.get('translation'), '%s.translation' % name)
            rotation_rpy_deg = self._vector_from_config(
                config.get('rotation_rpy_deg'), '%s.rotation_rpy_deg' % name,
                ('roll', 'pitch', 'yaw'))
            rotation = transformations.quaternion_from_euler(
                *np.radians(rotation_rpy_deg))
            transforms.append({
                'name': name,
                'parent_frame': parent_frame,
                'child_frame': child_frame,
                'translation': translation,
                'rotation': rotation,
            })
        return transforms

    @staticmethod
    def get_imu_translation(static_transforms):
        """从 base_link -> imu 静态 TF 提取导航杆臂。"""
        for transform_config in static_transforms:
            if (transform_config['parent_frame'] == 'base_link'
                    and transform_config['child_frame'] == 'imu'):
                return transform_config['translation']
        raise ValueError('static_transforms 必须包含 base_link -> imu 变换')

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
        navigation_values = (
            msg.pose.latitude,
            msg.pose.longitude,
            msg.pose.depth,
            msg.pose.roll,
            msg.pose.pitch,
            msg.pose.yaw,
        )
        if (
                not np.all(np.isfinite(navigation_values))
                or not -90.0 <= msg.pose.latitude <= 90.0
                or not -180.0 <= msg.pose.longitude <= 180.0):
            rospy.logwarn_throttle(
                1.0, 'auv_tf_handler: 忽略非有限或超出范围的导航位姿')
            return
        wfm = self.get_world_frame_manager()
        north, east, down = wfm.lld_to_ned(
            msg.pose.latitude, msg.pose.longitude, msg.pose.depth
        )
        orientation = transformations.quaternion_from_euler(
            np.radians(msg.pose.roll),
            np.radians(msg.pose.pitch),
            np.radians(msg.pose.yaw),
        )
        base_position = origin_from_offset_point(
            (north, east, down),
            orientation,
            self.base_to_imu,
        )
        self.current_pose = [
            base_position[0],
            base_position[1],
            base_position[2],
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
        ]
        source_stamp = msg.header.stamp
        if source_stamp == rospy.Time():
            rospy.logwarn_throttle(
                2.0,
                'auv_tf_handler: /status/auv 时间戳为空，使用当前时间',
            )
            source_stamp = rospy.Time.now()
        self.publish_tf(source_stamp)
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

    def publish_tf(self, source_stamp):
        """使用上游状态时间戳发布 map -> base_link 动态 TF。"""
        self.tf_broadcaster.sendTransform(
            tuple(self.current_pose[0:3]),
            tuple(self.current_pose[3:7]),
            source_stamp,
            'base_link',
            'map',
        )

    def publish_static_transforms(self):
        """通过 /tf_static 锁存发布配置中的全部静态刚体变换。"""
        current_time = rospy.Time.now()
        messages = []
        for transform_config in self.static_transforms:
            message = TransformStamped()
            message.header.stamp = current_time
            message.header.frame_id = transform_config['parent_frame']
            message.child_frame_id = transform_config['child_frame']
            message.transform.translation.x = transform_config['translation'][0]
            message.transform.translation.y = transform_config['translation'][1]
            message.transform.translation.z = transform_config['translation'][2]
            message.transform.rotation.x = transform_config['rotation'][0]
            message.transform.rotation.y = transform_config['rotation'][1]
            message.transform.rotation.z = transform_config['rotation'][2]
            message.transform.rotation.w = transform_config['rotation'][3]
            messages.append(message)
        self.static_tf_broadcaster.sendTransform(messages)
        rospy.loginfo(
            'auv_tf_handler: 已通过 /tf_static 锁存发布 %d 组静态 TF',
            len(messages),
        )

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        rospy.init_node('auv_tf_handler_node')
        AUVTfHandler().run()
    except rospy.ROSInterruptException:
        pass
