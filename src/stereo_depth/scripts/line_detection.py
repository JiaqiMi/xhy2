#!/home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class PipelineDetector:
    def __init__(self):
        rospy.init_node("pipeline_detector", anonymous=True)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/left/image_raw", Image, self.image_callback, queue_size=1)

        # 相机内参（需替换为你的标定结果）
        self.fx = 519.1519
        self.fy = 519.712551
        self.cx = 319.174292
        self.cy = 277.976296
        self.assumed_depth = 0.8  # 设定深度（单位：米）

        rospy.loginfo("Pipeline Detector Initialized.")

    def image_callback(self, img_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge error: {e}")
            return

        # 转换为 HSV 并提取红色区域
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # 红色分两段，低+高
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学操作清除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # 骨架提取
        skeleton = self.get_skeleton(mask_clean)

        # 获取骨架上的像素点
        points = np.column_stack(np.where(skeleton > 0))  # [[y,x], ...]

        if len(points) < 10:
            rospy.logwarn("Not enough skeleton points found.")
            return

        # 选取等间距10个骨架点
        indices = np.linspace(0, len(points) - 1, 10, dtype=int)
        selected_points = points[indices]  # shape: (10, 2)

        coords_3d = []
        for pt in selected_points:
            y, x = pt  # 行是y，列是x
            X, Y, Z = self.pixel_to_camera_coords(x, y, self.assumed_depth)
            coords_3d.append((X, Y, Z))

        rospy.loginfo("Detected 3D pipeline points (camera frame):")
        for i, (X, Y, Z) in enumerate(coords_3d):
            rospy.loginfo(f"Point {i+1}: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

    def pixel_to_camera_coords(self, u, v, depth):
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return x, y, z

    def get_skeleton(self, binary_img):
        size = np.size(binary_img)
        skel = np.zeros(binary_img.shape, np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        while not done:
            eroded = cv2.erode(binary_img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary_img, temp)
            skel = cv2.bitwise_or(skel, temp)
            binary_img = eroded.copy()

            zeros = size - cv2.countNonZero(binary_img)
            if zeros == size:
                done = True

        return skel

if __name__ == '__main__':
    try:
        PipelineDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
