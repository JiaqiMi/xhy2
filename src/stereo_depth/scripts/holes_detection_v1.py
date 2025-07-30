#!/home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import tf
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
# from auv_control.msg import TargetDetection
from stereo_depth.msg import TargetDetection, BoundingBox
from geometry_msgs.msg import PointStamped,PoseStamped,Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import Quaternion


def pixel_to_camera_coords(u, v, depth, fx, fy, cx, cy):
    """calculate the location in camera axis of any pixel in picture."""
    Z = depth[v][u]  # 注意 OpenCV 顺序是 (row, col) = (v, u)
    if Z == 0:
        return None  # 无效深度
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])


def get_stable_depth(u, v, depth, fx, fy, cx, cy, window_size=11):
    half_w = window_size // 2
    h, w = depth.shape

    u, v = int(u), int(v)

    umin = max(u - half_w, 0)
    umax = min(u + half_w + 1, w)
    vmin = max(v - half_w, 0)
    vmax = min(v + half_w + 1, h)

    region = depth[vmin:vmax, umin:umax]
    valid = region[np.isfinite(region) & (region > 0)]

    if valid.size < 3:
        return np.array([np.nan, np.nan, np.nan])  # too few valid points

    Z = np.min(valid)  # or np.mean(valid)
    if Z == 0:
        rospy.logwarn("Invalid depth at pixel ({}, {}): Z = 0".format(u, v))
        return np.array([np.nan, np.nan, np.nan])
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])



def compute_pose_from_quad(P1, P2, P3, P4):
    center = (P1 + P2 + P3 + P4) / 4.0
    vec1 = P2 - P1
    vec2 = P4 - P1
    z_axis = np.cross(vec1, vec2)
    z_axis /= np.linalg.norm(z_axis)

    x_axis = vec1 / np.linalg.norm(vec1)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)

    T = np.eye(4)
    T[:3, :3] = R
    quat = tf.transformations.quaternion_from_matrix(T)

    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "camera"
    pose.pose.position.x = center[0]
    pose.pose.position.y = center[1]
    pose.pose.position.z = center[2]
    pose.pose.orientation = Quaternion(*quat)

    return pose


class StereoDepthNode:
    def __init__(self):
        rospy.init_node("holes_location", anonymous=True)

        # 相机参数（来自你的标定结果）
        # air - old
        # self.fx = 218.510120
        # self.fy = 218.510120
        # self.cx = 175.566744
        # self.cy = 124.191102
        # self.baseline = 13.041602 / 218.510120  # m

        # water - old
        # self.fx = 360.607260
        # self.fy = 360.607260
        # self.cx = 177.368463
        # self.cy = 121.148735
        # self.baseline = 21.507233 / 360.607260  # m
        
        # air - new
        # self.fx = 572.993971
        # self.fy = 572.993971
        # self.cx = 374.534946
        # self.cy = 271.474743
        # self.baseline = 34.309807 / 572.993971  # m
        
        # # water - new
        # self.fx = 1080.689861
        # self.fy = 1080.689861
        # self.cx = 559.908498
        # self.cy = 261.932663
        # self.baseline = 81.420154 / 1080.689861  # m
        
        # water - new - 0727
        self.fx = 798.731044
        self.fy = 798.731044
        self.cx = 348.127430
        self.cy = 269.935493
        self.baseline = 47.694354 / 798.731044  # m
        
        
        # self.fx = 686.32092
        # self.fy = 685.83026
        # self.cx = 316.41091
        # self.cy = 279.42833
        # self.baseline = 47.694354 / 686.32092  # m
        
        
        self.bridge = CvBridge()
        self.target_uv = None  
        self.target_conf = 0.0
        self.target_class = ""

        # 图像订阅（注意：需要有两个图像 topic）
        self.left_sub = Subscriber("/left/image_raw", Image)
        self.right_sub = Subscriber("/right/image_raw", Image)

        # Book YOLO center pixel 
        rospy.Subscriber("/yolov8/target_bbox", BoundingBox, self.target_callback)

        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.callback)

        # 可选：发布深度图
        # self.depth_pub = rospy.Publisher("/stereo/depth_image", Image, queue_size=1)

        # Publish the target message
        self.target_message = rospy.Publisher("/obj/target_message", TargetDetection, queue_size=1)

        rospy.loginfo("Stereo Depth Node Initialized.")


    def target_callback(self, msg):
        """保存最新目标的像素位置"""
        # self.target_uv = (int(msg.point.x), int(msg.point.y))
        # self.target_conf = msg.point.z
        # self.target_class = msg.header.frame_id
        # self.target_check_time = msg.header.stamp
        self.target_x1 = int(msg.x1)
        self.target_y1 = int(msg.y1)
        self.target_x2 = int(msg.x2)
        self.target_y2 = int(msg.y2)
        self.target_conf = float(msg.conf)
        self.target_class = msg.header.frame_id
        self.target_check_time = msg.header.stamp
        


    def callback(self, left_img_msg, right_img_msg):
        try:
            left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
            right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", str(e))
            return

        # 转灰度
        grayL = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        # print("grayL", grayL.shape)

        # 创建 StereoSGBM 匹配器
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 6,  # 必须是16的倍数
            blockSize=7,
            P1=8 * 3 * 7 ** 2,
            P2=32 * 3 * 7 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        disparity = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
        # print("disparity", disparity.shape)

        # 避免除以0
        disparity[disparity <= 0.0] = 0.1

        # 计算深度（Z = fx * baseline / disparity）
        depth = self.fx * self.baseline / disparity  # 单位：米

        # calculate the center pixel location
        # (height, width) = disparity.shape
        # X, Y, Z = pixel_to_camera_coords(
        #     u=width//2,
        #     v=height//2,
        #     depth=depth,
        #     fx=self.fx,
        #     fy=self.fy,
        #     cx=self.cx,
        #     cy=self.cy
        # )
        # print("X: {}, Y: {}, Z:{}".format(X, Y, Z))

        # 判断是否有来自YOLO的像素坐标
        P1, P2, P3, P4 = None, None, None, None
        if self.target_conf >= 0.6:
            
            rospy.loginfo("Target detected: class=%s, conf=%.2f, x1: %d, y1: %d, x2: %d, y2: %d",
                          self.target_class, self.target_conf,
                          self.target_x1, self.target_y1, self.target_x2, self.target_y2)
        
            if (0 <= self.target_x1 <= disparity.shape[1]) and (0 <= self.target_y1 <= disparity.shape[0]) and \
                (0 <= self.target_x2 <= disparity.shape[1]) and (0 <= self.target_y2 <= disparity.shape[0]):
               
                # 获取每个角点的相机坐标
                P1 = get_stable_depth(self.target_x1, self.target_y1, depth, self.fx, self.fy, self.cx, self.cy)
                P2 = get_stable_depth(self.target_x2, self.target_y1, depth, self.fx, self.fy, self.cx, self.cy)
                P3 = get_stable_depth(self.target_x2, self.target_y2, depth, self.fx, self.fy, self.cx, self.cy)
                P4 = get_stable_depth(self.target_x1, self.target_y2, depth, self.fx, self.fy, self.cx, self.cy)
                
                rospy.loginfo("class_name: %s, P1.X: %.2f, P1.Y: %.2f, P1.Z: %.2f", self.target_class, P1[0], P1[1], P1[2])
                rospy.loginfo("class_name: %s, P2.X: %.2f, P2.Y: %.2f, P2.Z: %.2f", self.target_class, P2[0], P2[1], P2[2])
                rospy.loginfo("class_name: %s, P3.X: %.2f, P3.Y: %.2f, P3.Z: %.2f", self.target_class, P3[0], P3[1], P3[2])
                rospy.loginfo("class_name: %s, P4.X: %.2f, P4.Y: %.2f, P4.Z: %.2f", self.target_class, P4[0], P4[1], P4[2])
                
                # 计算中心点和姿态
                pose_msg = compute_pose_from_quad(P1, P2, P3, P4)
                
                rospy.loginfo("pose_msg.pose.position.x: %.2f", pose_msg.pose.position.x)
                rospy.loginfo("pose_msg.pose.position.y: %.2f", pose_msg.pose.position.y)
                rospy.loginfo("pose_msg.pose.position.z: %.2f", pose_msg.pose.position.z)
                
                if (-1.0 < pose_msg.pose.position.x < 1.0) and (-1.0 < pose_msg.pose.position.y < 1.0):
                    rospy.loginfo(
                        "Valid target: time=%s, class=%s conf=%.2f -> X=%.2f Y=%.2f Z=%.2f", \
                        self.target_check_time, self.target_class, self.target_conf,  \
                        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z)
                    
                    # 发布target message
                    try:
                        msg = TargetDetection()
                        msg.pose = pose_msg
                        msg.type = 'center'
                        msg.conf = self.target_conf
                        msg.class_name = self.target_class
                        self.target_message.publish(msg)
                    except Exception as e:
                        rospy.logerr("Error publishing depth: %s", str(e))
                    
                else:
                    # rospy.logwarn("Invalid location of objection.")
                    rospy.loginfo(
                        "InValid target: time=%s, class=%s conf=%.2f -> X=%.2f Y=%.2f Z=%.2f", \
                        self.target_check_time, self.target_class, self.target_conf,  \
                        pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z)
            else:
                rospy.logwarn("Target pixel coordinates are out of bounds.")
        

        # 发布相机事件
        # try:
        #     if (-1.0 < pose_msg.pose.position.x < 1.0) and (-1.0 < pose_msg.pose.position.y < 1.0):
        #         depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        #         depth_msg.header = left_img_msg.header
        #         # self.depth_pub.publish(depth_msg)

        #         # 发布target message
        #         msg = TargetDetection()
        #         # pos_msg = PoseStamped()
        #         # pos_msg.header.stamp = self.target_check_time
        #         # pos_msg.header.frame_id = "camera"
        #         # pos_msg.pose.position.x = X
        #         # pos_msg.pose.position.y = Y
        #         # pos_msg.pose.position.z = Z 
        #         # # pos_msg.pose.orientation = Quaternion(0, 0, 0, 1) 
        #         # # tf.transformations.quaternion_from_euler(0, 0, 1)
        #         # pos_msg.pose.orientation = pos_msg.pose.orientation = Quaternion(0, 0, 0, 1) 
        #         msg.pose = pos_msg
        #         msg.type = 'center'
        #         msg.conf = self.target_conf
        #         msg.class_name = self.target_class
        #         self.target_message.publish(msg)
        # except Exception as e:
        #     rospy.logerr("Error publishing depth: %s", str(e))

        # 可视化（归一化视差）
        # disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        # disp_vis = np.uint8(disp_vis)
        # cv2.imshow("Disparity", disp_vis)
        # cv2.imshow("Left", left_img)
        # cv2.waitKey(1)

if __name__ == '__main__':
    try:
        StereoDepthNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

