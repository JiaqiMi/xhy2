#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：world_origin_reset.py
功能：通过稳定识别红色圆形，一次性重置 map 坐标系原点
作者：buyegaid
订阅：/obj/target_message(TargetDetection.msg)
      /status/auv(AUVData.msg)
      /world_origin_reset_result(Bool.msg)
发布：/world_origin_reset_candidate(PoseStamped.msg)
记录：
2026.7.13
    新增红色圆形稳定观测、TF 转换、原点重置请求与确认超时保护。
2026.7.31
    红圆深度改用 DVL 高度与 base_link 到相机的 TF 高度计算，视觉深度仅记录对比。
2026.7.31
    DVL 高度有效性改为与 map_initer 一致，要求 sensor_valid 的第 0、1 位同时有效。
2026.7.31
    候选点深度直接采用 DVL 推导的相机到底距离，不再采用未可靠初始化的 map z。
2026.7.31
    修正池底原点深度：以当前 AUV 绝对深度加 DVL 高度计算，确保 NED 的 D=0 位于池底。
"""

import math
import threading

import rospy
import tf
from auv_control.msg import AUVData, TargetDetection
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool

from world_origin_reset_utils import (
    RobustPointEstimator,
    camera_depth_from_dvl,
    is_matching_target,
    is_valid_camera_xy,
)


class WorldOriginResetNode:
    """收集红圆观测，确认稳定后请求 map_initer 更新原点。"""

    def __init__(self):
        self.target_topic = rospy.get_param("~target_topic", "/obj/target_message")
        self.candidate_topic = rospy.get_param(
            "~candidate_topic", "/world_origin_reset_candidate"
        )
        self.result_topic = rospy.get_param(
            "~result_topic", "/world_origin_reset_result"
        )
        self.status_topic = rospy.get_param("~status_topic", "/status/auv")
        self.target_class = rospy.get_param("~target_class", "red")
        self.source_frame = rospy.get_param("~source_frame", "camera")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.reference_frame = rospy.get_param("~reference_frame", "map")
        self.min_confidence = float(rospy.get_param("~min_confidence", 0.7))
        self.status_timeout = float(rospy.get_param("~status_timeout_sec", 0.5))
        self.require_dvl_altitude_valid = bool(
            rospy.get_param("~require_dvl_altitude_valid", True)
        )
        self.tf_ready_timeout = float(rospy.get_param("~tf_ready_timeout_sec", 30.0))
        # 限时
        self.observation_timeout = float(rospy.get_param("~observation_timeout_sec", 180.0))
        self.result_timeout = float(rospy.get_param("~result_timeout_sec", 5.0))

        self.estimator = RobustPointEstimator(
            sample_count=int(rospy.get_param("~sample_count", 10)),
            min_inliers=int(rospy.get_param("~min_inliers", 8)),
            max_spread=float(rospy.get_param("~max_spread", 0.20)),
            min_inlier_radius=float(rospy.get_param("~min_inlier_radius", 0.03)),
        )
        self.tf_listener = tf.TransformListener()
        self.started_at = rospy.Time.now()
        self.sampling_started_at = None
        self.candidate_sent_at = None
        self.candidate_sent = False
        self.status_lock = threading.RLock()
        self.latest_dvl_altitude = None
        self.latest_auv_depth = None
        self.latest_status_stamp = None
        self.latest_status_received_at = None

        self.candidate_pub = rospy.Publisher(
            self.candidate_topic, PoseStamped, queue_size=1
        )

        # 订阅目标检测消息
        rospy.Subscriber(
            self.target_topic, TargetDetection, self.target_callback, queue_size=10
        )
        rospy.Subscriber(
            self.status_topic, AUVData, self.status_callback, queue_size=10
        )

        # 订阅结果更新
        rospy.Subscriber(self.result_topic, Bool, self.result_callback, queue_size=1)
        self.watchdog = rospy.Timer(rospy.Duration(0.5), self.watchdog_callback)

        rospy.loginfo(
            "world_origin_reset: 已启动，等待 class_name=%s 的稳定目标和 DVL 高度",
            self.target_class,
        )

    def status_callback(self, msg):
        """缓存同一状态帧的 DVL 高度和 AUV 绝对深度。"""
        altitude = float(msg.pose.altitude)
        auv_depth = float(msg.pose.depth)
        if not math.isfinite(altitude) or not math.isfinite(auv_depth):
            with self.status_lock:
                self.latest_dvl_altitude = None
                self.latest_auv_depth = None
                self.latest_status_stamp = None
                self.latest_status_received_at = None
            rospy.logwarn_throttle(
                2.0, "world_origin_reset: 忽略非有限的 DVL 高度或 AUV 深度"
            )
            return

        # 与 map_initer 保持一致：第 0、1 位分别表示惯导数据有效。
        required_sensor_valid = (1 << 0) | (1 << 1)
        if (
                self.require_dvl_altitude_valid
                and (int(msg.sensor.sensor_valid) & required_sensor_valid)
                != required_sensor_valid):
            with self.status_lock:
                self.latest_dvl_altitude = None
                self.latest_auv_depth = None
                self.latest_status_stamp = None
                self.latest_status_received_at = None
            rospy.logwarn_throttle(
                2.0,
                "world_origin_reset: DVL 高度无效，sensor_valid=0x%02X",
                msg.sensor.sensor_valid,
            )
            return

        with self.status_lock:
            self.latest_dvl_altitude = altitude
            self.latest_auv_depth = auv_depth
            self.latest_status_stamp = msg.header.stamp
            self.latest_status_received_at = rospy.Time.now()

    def get_current_dvl_measurement(self):
        """返回未超时的 DVL 高度、AUV 绝对深度及状态帧时间戳。"""
        with self.status_lock:
            altitude = self.latest_dvl_altitude
            auv_depth = self.latest_auv_depth
            stamp = self.latest_status_stamp
            received_at = self.latest_status_received_at

        if altitude is None or auv_depth is None or received_at is None:
            rospy.logwarn_throttle(
                2.0, "world_origin_reset: 尚未收到有效的 DVL 高度和 AUV 深度"
            )
            return None

        reference_time = stamp if stamp is not None and not stamp.is_zero() else received_at
        age = (rospy.Time.now() - reference_time).to_sec()
        if age > self.status_timeout:
            rospy.logwarn_throttle(
                2.0,
                "world_origin_reset: DVL 高度已超时 %.3fs（限制 %.3fs）",
                age,
                self.status_timeout,
            )
            return None
        return altitude, auv_depth, reference_time, max(age, 0.0)

    def target_callback(self, msg):
        """验证观测；使用 map 平面坐标和相机到底深度发布唯一候选点。"""

        # 如果已经发送过候选点，则不再处理新的观测。
        if self.candidate_sent:
            return

        # 验证观测的坐标系和目标类型
        if msg.pose.header.frame_id != self.source_frame:
            rospy.logwarn_throttle(
                2.0,
                "world_origin_reset: 忽略非 %s 坐标系的目标: %s",
                self.source_frame,
                msg.pose.header.frame_id,
            )
            return

        # 验证观测的目标类型和置信度
        if not is_matching_target(
            msg.class_name,
            msg.type,
            msg.conf,
            self.target_class,
            self.min_confidence,
        ):
            return

        # 视觉仅提供平面位置；其 z 仅用于与 DVL 推导深度进行日志对比。
        visual_camera_z = float(msg.pose.pose.position.z)
        camera_xy = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
        )
        if not is_valid_camera_xy(camera_xy):
            rospy.logwarn_throttle(2.0, "world_origin_reset: 忽略无效的相机平面坐标")
            return

        dvl_data = self.get_current_dvl_measurement()
        if dvl_data is None:
            return
        dvl_altitude, auv_depth, _status_stamp, dvl_age = dvl_data

        try:
            self.tf_listener.waitForTransform(
                self.reference_frame,
                self.source_frame,
                msg.pose.header.stamp,
                rospy.Duration(0.1),
            )
            self.tf_listener.waitForTransform(
                self.base_frame,
                self.source_frame,
                msg.pose.header.stamp,
                rospy.Duration(0.1),
            )
            base_to_camera_translation, _ = self.tf_listener.lookupTransform(
                self.base_frame,
                self.source_frame,
                msg.pose.header.stamp,
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0,
                "world_origin_reset: 等待 %s 到 %s/base_link 的 TF: %s",
                self.source_frame,
                self.reference_frame,
                error,
            )
            return

        try:
            dvl_camera_z = camera_depth_from_dvl(
                dvl_altitude, base_to_camera_translation[2]
            )
        except ValueError as error:
            rospy.logwarn_throttle(
                2.0, "world_origin_reset: 无法计算 DVL 推导深度: %s", error
            )
            return

        # DVL 高度相对于 base_link；DVL 与 IMU 杆臂 z 为 0，因此当前
        # AUV 绝对深度加 DVL 高度就是池底的绝对深度。以此为 map 原点后，
        # 池底平面在 NED 中恒为 D=0。
        pool_bottom_depth = auv_depth + dvl_altitude

        point = PointStamped()
        point.header = msg.pose.header
        point.point.x, point.point.y, point.point.z = camera_xy[0], camera_xy[1], dvl_camera_z
        try:
            map_point = self.tf_listener.transformPoint(self.reference_frame, point)
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0, "world_origin_reset: 转换 DVL 推导点到 map 失败: %s", error
            )
            return

        rospy.loginfo(
            "world_origin_reset: 深度对比 AUV深度=%.3fm, DVL高度=%.3fm, "
            "相机TF z=%.3fm, 相机到底=%.3fm, 采用池底绝对深度=%.3fm, "
            "TF map z(仅对比)=%.3fm, 视觉相机z=%s, "
            "DVL帧龄=%.3fs",
            auv_depth,
            dvl_altitude,
            base_to_camera_translation[2],
            dvl_camera_z,
            pool_bottom_depth,
            map_point.point.z,
            "%.3f" % visual_camera_z if math.isfinite(visual_camera_z) else "无效",
            dvl_age,
        )

        # 开启采样计时器，并将观测点添加到稳健估计器中
        if self.sampling_started_at is None:
            self.sampling_started_at = rospy.Time.now()
            rospy.loginfo("world_origin_reset: TF 已就绪，开始收集稳定观测")

        # 平面坐标使用 map 变换结果；z 使用池底的绝对深度建立 NED 原点。
        candidate_point = (
            map_point.point.x,
            map_point.point.y,
            pool_bottom_depth,
        )
        candidate = self.estimator.add(
            candidate_point
        )
        rospy.loginfo(
            "world_origin_reset: 加入有效点 %s",
            candidate_point,
        )
        if candidate is None:
            return

        candidate_msg = PoseStamped()
        candidate_msg.header.stamp = rospy.Time.now()
        candidate_msg.header.frame_id = self.reference_frame
        candidate_msg.pose.position.x = float(candidate[0])
        candidate_msg.pose.position.y = float(candidate[1])
        candidate_msg.pose.position.z = float(candidate[2])
        candidate_msg.pose.orientation.w = 1.0
        self.candidate_sent = True
        self.candidate_sent_at = rospy.Time.now()
        self.candidate_pub.publish(candidate_msg)
        rospy.loginfo(
            "world_origin_reset: 已提交稳定候选点 (N=%.3f, E=%.3f, D=%.3f)",
            candidate[0], candidate[1], candidate[2],
        )

    def result_callback(self, msg):
        """收到 map_initer 的处理结果后结束本次一次性操作。"""
        if not self.candidate_sent:
            return
        if msg.data:
            rospy.loginfo("world_origin_reset: 世界原点已更新，本次操作完成")
        else:
            rospy.logerr("world_origin_reset: map_initer 拒绝了原点更新请求")
        rospy.signal_shutdown("世界原点重置已结束")

    def watchdog_callback(self, _event):
        """在依赖未就绪、观测不稳定或确认失败时安全结束。"""
        now = rospy.Time.now()
        if self.candidate_sent:
            if (now - self.candidate_sent_at).to_sec() > self.result_timeout:
                rospy.logerr("world_origin_reset: 等待原点更新确认超时，未再发送请求")
                rospy.signal_shutdown("等待更新确认超时")
            return

        if self.sampling_started_at is None:
            if (now - self.started_at).to_sec() > self.tf_ready_timeout:
                rospy.logerr("world_origin_reset: map 到 camera 的 TF 未在限定时间内就绪")
                rospy.signal_shutdown("TF 未就绪")
            return

        if (now - self.sampling_started_at).to_sec() > self.observation_timeout:
            rospy.logerr(
                "world_origin_reset: 未在限定时间内取得 %d 个稳定观测",
                self.estimator.sample_count,
            )
            rospy.signal_shutdown("稳定观测超时")


if __name__ == "__main__":
    try:
        rospy.init_node("world_origin_reset")
        WorldOriginResetNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
