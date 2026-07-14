#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：world_origin_reset.py
功能：通过稳定识别红色圆形，一次性重置 map 坐标系原点
作者：buyegaid
订阅：/obj/target_message(TargetDetection.msg)
      /world_origin_reset_result(Bool.msg)
发布：/world_origin_reset_candidate(PoseStamped.msg)
记录：
2026.7.13
    新增红色圆形稳定观测、TF 转换、原点重置请求与确认超时保护。
"""

import rospy
import tf
from auv_control.msg import TargetDetection
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool

from world_origin_reset_utils import (
    RobustPointEstimator,
    is_matching_target,
    is_valid_camera_point,
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
        self.target_class = rospy.get_param("~target_class", "red")
        self.source_frame = rospy.get_param("~source_frame", "camera")
        self.reference_frame = rospy.get_param("~reference_frame", "map")
        self.min_confidence = float(rospy.get_param("~min_confidence", 0.7))
        self.min_depth = float(rospy.get_param("~min_depth", 0.1))
        self.max_depth = float(rospy.get_param("~max_depth", 3.0))
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

        self.candidate_pub = rospy.Publisher(
            self.candidate_topic, PoseStamped, queue_size=1
        )

        # 订阅目标检测消息
        rospy.Subscriber(
            self.target_topic, TargetDetection, self.target_callback, queue_size=10
        )

        # 订阅结果更新
        rospy.Subscriber(self.result_topic, Bool, self.result_callback, queue_size=1)
        self.watchdog = rospy.Timer(rospy.Duration(0.5), self.watchdog_callback)

        rospy.loginfo(
            "world_origin_reset: 已启动，等待 class_name=%s 的稳定目标", self.target_class
        )

    def target_callback(self, msg):
        """验证观测、转换到 map，并在稳定后发布唯一候选点。"""

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

        # 将相机坐标系下的观测点转换到 map 坐标系
        camera_point = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        )
        if not is_valid_camera_point(camera_point, self.min_depth, self.max_depth):
            rospy.logwarn_throttle(2.0, "world_origin_reset: 忽略无效的相机坐标")
            return

        # 将相机坐标系下的点转换到 map 坐标系
        point = PointStamped()
        point.header = msg.pose.header
        point.point.x, point.point.y, point.point.z = camera_point
        try:
            self.tf_listener.waitForTransform(
                self.reference_frame,
                point.header.frame_id,
                point.header.stamp,
                rospy.Duration(0.1),
            )
            map_point = self.tf_listener.transformPoint(self.reference_frame, point)
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0,
                "world_origin_reset: 等待 %s 到 %s 的 TF: %s",
                point.header.frame_id,
                self.reference_frame,
                error,
            )
            return

        # 开启采样计时器，并将观测点添加到稳健估计器中
        if self.sampling_started_at is None:
            self.sampling_started_at = rospy.Time.now()
            rospy.loginfo("world_origin_reset: TF 已就绪，开始收集稳定观测")

        # 添加到候选
        candidate = self.estimator.add(
            (map_point.point.x, map_point.point.y, map_point.point.z)
        )
        rospy.loginfo("world_origin_reset: 加入有效point %s", (map_point.point.x, map_point.point.y, map_point.point.z))
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
