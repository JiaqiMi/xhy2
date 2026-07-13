#! /home/xhy/xhy_env36/bin/python
# -*- coding: utf-8 -*-
"""
名称：map_initer.py
功能：创建map坐标系原点
作者：buyegaid
监听：/debug_auv_data(AUVData.msg)
发布：/world_origin(NavSatFix.msg)
记录：
2025.7.19 10:50
    第一版完成
2026.7.13
    新增一次性红色圆形原点重置，监听 /world_origin_reset_candidate，
    更新后锁存发布 /world_origin，并通过 /world_origin_reset_result 返回结果。
"""

import threading

import numpy as np
import rospy
from auv_control.msg import AUVData
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool

from world_frame import WorldFrameManager


class MapIniter:
    """先由导航数据初始化原点，再接受一次稳定红圆候选点重置原点。"""

    def __init__(self):
        self.lock = threading.RLock()
        self.init_lat_list = []
        self.init_lon_list = []
        self.init_dep_list = []
        self.initialized = False
        self.reset_done = False
        self.wfm = None

        self.pub = rospy.Publisher('/world_origin', NavSatFix, queue_size=1, latch=True)
        self.reset_result_pub = rospy.Publisher(
            '/world_origin_reset_result', Bool, queue_size=1
        )
        rospy.Subscriber('/debug_auv_data', AUVData, self.debug_callback)
        rospy.Subscriber(
            '/world_origin_reset_candidate', PoseStamped, self.reset_callback, queue_size=1
        )
        rospy.loginfo("map_initer: 已启动")

    def debug_callback(self, msg):
        """取前 50 个有效导航数据的平均值作为初始世界坐标系原点。"""
        with self.lock:
            if self.initialized:
                return
            if (msg.sensor.sensor_valid >> 6) == 0:
                rospy.loginfo_throttle(2, "map_initer: 惯导数据无效，等待有效数据...")
                return

            rospy.loginfo_throttle(2, "map_initer: 惯导数据有效")
            if len(self.init_lon_list) < 50:
                self.init_lat_list.append(msg.pose.latitude)
                self.init_lon_list.append(msg.pose.longitude)
                self.init_dep_list.append(msg.pose.depth)
            if len(self.init_lon_list) != 50:
                return

            latitude = sum(self.init_lat_list) / len(self.init_lat_list)
            longitude = sum(self.init_lon_list) / len(self.init_lon_list)
            depth = sum(self.init_dep_list) / len(self.init_dep_list)
            self.wfm = WorldFrameManager(latitude, longitude, depth)
            self.initialized = True
            self.publish_origin(latitude, longitude, depth)
            rospy.loginfo(
                "map_initer: 初始世界坐标系原点已发布: lat=%s, lon=%s, depth=%s",
                latitude, longitude, depth,
            )

    def reset_callback(self, candidate):
        """将旧 map 中的红圆候选点换算为新的经纬深原点。"""
        with self.lock:
            if not self.initialized or self.wfm is None:
                rospy.logwarn("map_initer: 初始原点未就绪，拒绝重置请求")
                self.reset_result_pub.publish(Bool(data=False))
                return
            if self.reset_done:
                rospy.logwarn("map_initer: 本次运行已经完成过原点重置")
                self.reset_result_pub.publish(Bool(data=False))
                return
            if candidate.header.frame_id != 'map':
                rospy.logwarn("map_initer: 候选点坐标系必须为 map")
                self.reset_result_pub.publish(Bool(data=False))
                return

            point = np.array([
                candidate.pose.position.x,
                candidate.pose.position.y,
                candidate.pose.position.z,
            ], dtype=float)
            if not np.all(np.isfinite(point)):
                rospy.logwarn("map_initer: 候选点包含无效坐标")
                self.reset_result_pub.publish(Bool(data=False))
                return

            latitude, longitude, depth = self.wfm.ned_to_lld(*point)
            if not np.all(np.isfinite([latitude, longitude, depth])):
                rospy.logerr("map_initer: 候选点无法换算为有效经纬深")
                self.reset_result_pub.publish(Bool(data=False))
                return

            self.wfm = WorldFrameManager(latitude, longitude, depth)
            self.reset_done = True
            self.publish_origin(latitude, longitude, depth)
            self.reset_result_pub.publish(Bool(data=True))
            rospy.loginfo(
                "map_initer: 已将红圆设为新原点: lat=%s, lon=%s, depth=%s",
                latitude, longitude, depth,
            )

    def publish_origin(self, latitude, longitude, depth):
        """锁存发布当前生效的世界坐标系原点。"""
        origin = NavSatFix()
        origin.header.stamp = rospy.Time.now()
        origin.header.frame_id = 'map'
        origin.latitude = latitude
        origin.longitude = longitude
        origin.altitude = depth
        self.pub.publish(origin)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        rospy.init_node('map_initer_node')
        MapIniter().run()
    except rospy.ROSInterruptException:
        pass
