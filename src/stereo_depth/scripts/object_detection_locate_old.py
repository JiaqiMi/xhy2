#!/home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import tf
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from stereo_depth.msg import TargetDetection
from geometry_msgs.msg import PointStamped,PoseStamped,Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber


def pixel_to_camera_coords(u, v, depth, fx, fy, cx, cy):
    """calculate the location in camera axis of any pixel in picture."""
    Z = depth[v][u]  # 注意 OpenCV 顺序是 (row, col) = (v, u)
    if Z == 0:
        return None  # 无效深度
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z


class StereoDepthNode:
    def __init__(self):
        rospy.init_node("stereo_depth", anonymous=True)

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
        
        # water - new
        self.fx = 1080.689861
        self.fy = 1080.689861
        self.cx = 559.908498
        self.cy = 261.932663
        self.baseline = 81.420154 / 1080.689861  # m
        
        self.bridge = CvBridge()
        self.target_uv = None  
        self.target_conf = 0.0
        self.target_class = ""

        # 图像订阅（注意：需要有两个图像 topic）
        self.left_sub = Subscriber("/left/image_raw", Image)
        self.right_sub = Subscriber("/right/image_raw", Image)

        # Book YOLO center pixel 
        rospy.Subscriber("/yolov8/target_center", PointStamped, self.target_callback)

        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.callback)

        # 可选：发布深度图
        # self.depth_pub = rospy.Publisher("/stereo/depth_image", Image, queue_size=1)

        # Publish the target message
        self.target_message = rospy.Publisher("/obj/target_message", TargetDetection, queue_size=1)

        rospy.loginfo("Stereo Depth Node Initialized.")


    def target_callback(self, msg):
        """保存最新目标的像素位置"""
        self.target_uv = (int(msg.point.x), int(msg.point.y))
        self.target_conf = msg.point.z
        self.target_class = msg.header.frame_id
        self.target_check_time = msg.header.time


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
        X, Y, Z = None, None, None
        if self.target_uv is not None and self.target_conf >= 0.2:
            u, v = self.target_uv
            if 0 <= u < disparity.shape[1] and 0 <= v < disparity.shape[0]:
                X, Y, Z = pixel_to_camera_coords(u, v, depth, self.fx, self.fy, self.cx, self.cy)
                if X is not None:
                    rospy.loginfo(
                        "Valid target: time=%s, class=%s conf=%.2f -> X=%.2f Y=%.2f Z=%.2f", \
                        self.target_check_time, self.target_class, self.target_conf, X, Y, Z)
                else:
                    rospy.logwarn("Valid location of objection.")
            else:
                rospy.logwarn("Target pixel coordinates are out of bounds.")
        

        # 发布相机事件
        try:
            if X is not None:
                depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
                depth_msg.header = left_img_msg.header
                # self.depth_pub.publish(depth_msg)

                # 发布target message
                msg = TargetDetection()
                pos_msg = PoseStamped()
                pos_msg.header.stamp = self.target_check_time
                pos_msg.header.frame_id = "camera"
                pos_msg.pose.position.x = X
                pos_msg.pose.position.y = Y
                pos_msg.pose.position.z = Z 
                pos_msg.pose.orientation = tf.transformations.quaternion_from_euler(0, 0, 1)
                msg.pose = pos_msg
                msg.type = 'center'
                msg.conf = self.target_conf
                msg.class_name = self.target_class
                self.target_message.publish(msg)
        except Exception as e:
            rospy.logerr("Error publishing depth: %s", str(e))

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

