#!/home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from stereo_depth.msg import TargetDetection, BoundingBox
from geometry_msgs.msg import PoseStamped, Quaternion


class PoseEstimatorNode:
    def __init__(self):
        rospy.init_node("target_pose_estimator", anonymous=True)

        # 订阅点云和目标检测框
        self.pc_sub = rospy.Subscriber("/points2", PointCloud2, self.pc_callback)
        self.box_sub = rospy.Subscriber("/yolov8/target_bbox", BoundingBox, self.box_callback)

        # 发布目标位姿
        self.pose_pub = rospy.Publisher("/obj/target_message", TargetDetection, queue_size=1)

        self.latest_box = None
        self.latest_stamp = rospy.Time.now()

        rospy.loginfo("Pose Estimator Node Initialized.")

    def box_callback(self, msg):
        self.latest_box = msg
        self.latest_stamp = msg.header.stamp

    def pc_callback(self, cloud_msg):
        if self.latest_box is None:
            return

        # 提取边界框
        x1, y1, x2, y2 = self.latest_box.x1, self.latest_box.y1, self.latest_box.x2, self.latest_box.y2

        points = []
        # for u in range(x1, x2):
        #     for v in range(y1, y2):
        #         gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), uvs=[[u, v]])
        #         for p in gen:
        #             if not any(np.isnan(p)) and not all(np.isclose(p, [0, 0, 0])):
        #                 points.append(np.array(p))
        #                 print(f"Point at ({u}, {v}): {p}")
        p1 = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), uvs=[[x1, y1]])
        p2 = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), uvs=[[x2, y1]])
        p3 = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), uvs=[[x1, y2]])
        p4 = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), uvs=[[x2, y2]])
        
        print(f"Points at corners: {list(p1)}, {list(p2)}, {list(p3)}, {list(p4)}")
        
        if len(points) < 10:
            rospy.logwarn("Too few valid points in bounding box.")
            return

        points_np = np.array(points)
        center = np.mean(points_np, axis=0)

        # 姿态估计：计算法向量作为 Z 轴方向
        if len(points_np) >= 3:
            cov = np.cov(points_np.T)
            eig_vals, eig_vecs = np.linalg.eig(cov)
            z_axis = eig_vecs[:, np.argmin(eig_vals)]  # 最小特征值对应法向量
            z_axis /= np.linalg.norm(z_axis)
        else:
            z_axis = np.array([0, 0, 1])

        x_axis = np.array([1, 0, 0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        R = np.column_stack((x_axis, y_axis, z_axis))
        T = np.eye(4)
        T[:3, :3] = R
        quat = tf.transformations.quaternion_from_matrix(T)

        pose = PoseStamped()
        pose.header.stamp = self.latest_stamp
        pose.header.frame_id = cloud_msg.header.frame_id
        pose.pose.position.x = center[0]
        pose.pose.position.y = center[1]
        pose.pose.position.z = center[2]
        pose.pose.orientation = Quaternion(*quat)

        msg = TargetDetection()
        msg.pose = pose
        msg.type = "center"
        msg.conf = self.latest_box.conf
        msg.class_name = self.latest_box.header.frame_id

        self.pose_pub.publish(msg)
        rospy.loginfo("Published pose: class=%s, conf=%.2f, X=%.2f Y=%.2f Z=%.2f",
                      msg.class_name, msg.conf,
                      pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)


if __name__ == "__main__":
    try:
        PoseEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
