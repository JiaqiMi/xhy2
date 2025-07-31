#!/home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from stereo_depth.msg import TargetDetection, BoundingBox
from message_filters import ApproximateTimeSynchronizer, Subscriber
from geometry_msgs.msg import PointStamped,PoseStamped,Quaternion


def pixel_to_camera_coords(u, v, depth, fx, fy, cx, cy):
    """calculate the location in camera axis of any pixel in picture."""
    Z = depth[v][u]  # 注意 OpenCV 顺序是 (row, col) = (v, u)
    if Z == 0:
        return None  # 无效深度
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z


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


class BallsDetector:
    def __init__(self):
        rospy.init_node("balls_detector", anonymous=True)

        self.bridge = CvBridge()
        # 图像订阅（注意：需要有两个图像 topic）
        # self.image_sub = rospy.Subscriber("/left/image_raw", Image, self.image_callback, queue_size=1)

        self.left_sub = Subscriber("/left/image_raw", Image)
        self.right_sub = Subscriber("/right/image_raw", Image)
        
        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        # # 相机内参（需替换为你的标定结果）
        # air - new - 0727 - P
        self.fx = 572.993971
        self.fy = 572.993971
        self.cx = 374.534946
        self.cy = 271.474743
        self.baseline = 34.309807 / 572.993971  # m
        
        # air - new - 0727 - K
        # self.fx = 519.151900
        # self.fy = 519.712551
        # self.cx = 319.174292
        # self.cy = 277.976296
        # self.baseline = 34.309807 / 519.151900  # m
        
        # water - new - 0727 - P
        # self.fx = 798.731044
        # self.fy = 798.731044
        # self.cx = 348.127430
        # self.cy = 269.935493
        # self.baseline = 47.694354 / 798.731044  # m
        
        # water - new - 0727 - K
        # self.fx = 686.32092
        # self.fy = 685.83026
        # self.cx = 316.41091
        # self.cy = 279.42833
        # self.baseline = 47.694354 / 686.32092  # m
        
        self.image_topic = rospy.get_param("~va", "/usb_cam/image_raw")
        
        # Publish the target message
        self.target_message = rospy.Publisher("/obj/target_message", TargetDetection, queue_size=1)

        rospy.loginfo("Pipeline Detector Initialized.")

    def image_callback(self, left_img_msg, right_img_msg):
        try:
            left_img = self.bridge.imgmsg_to_cv2(left_img_msg, "bgr8")
            right_img = self.bridge.imgmsg_to_cv2(right_img_msg, "bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", str(e))
            return
        
        # ============================  计算立体匹配的视差图   ============================ #
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
        
        # 转换为 HSV
        hsv = cv2.cvtColor(left_img, cv2.COLOR_BGR2HSV)
        
        # ============================  获取红色球   ============================ #
        # 红色分两段，低+高
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # lower_green = np.array([35, 70, 50])
        # upper_green = np.array([85, 255, 255])
        # mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # lower_blue = np.array([100, 70, 50])
        # upper_blue = np.array([130, 255, 255])
        # mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 形态学操作清除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # 骨架提取
        skeleton = self.get_skeleton(mask_clean)

        # 获取骨架上的像素点
        points = np.column_stack(np.where(skeleton > 0))  # [[y,x], ...]

        if len(points) < 1:
            rospy.logwarn("Not enough skeleton points found.")
            return

        # 选取等间距10个骨架点
        indices = np.linspace(0, len(points) - 1, 10, dtype=int)
        selected_points = points[indices]  # shape: (10, 2)
        
        coords_3d = []
        for idx, pt in enumerate(selected_points):
            u, v = pt  # 行是y，列是x
            rospy.loginfo(f"Processing point at pixel coordinates: idx={idx}, (u={u}, v={v})")
            # X, Y, Z = self.pixel_to_camera_coords(u, v, depth[y, x])
            X, Y, Z = get_stable_depth(u, v, depth, self.fx, self.fy, self.cx, self.cy)
            
            if (-1 < X < 1) and (-1 < Y < 1) and (0 < Z < 3):
                rospy.loginfo("Valid target:  class=red -> X=%.2f Y=%.2f Z=%.2f", X, Y, Z)
                coords_3d.append([X, Y, Z])
            else:
                # rospy.logwarn("Invalid location of objection.")
                rospy.loginfo(f"Invalid target: ({X}, {Y}, {Z})")
        
        if len(coords_3d) > 0:
            # 打印找到的最终目标的像素坐标和置信度
            mean_X = np.mean([coord[0] for coord in coords_3d])
            mean_Y = np.mean([coord[1] for coord in coords_3d])
            mean_Z = np.mean([coord[2] for coord in coords_3d])
                    
            # 发布相机事件
            try:
                # depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
                # depth_msg.header = left_img_msg.header
                # self.depth_pub.publish(depth_msg)

                # 发布target message
                msg = TargetDetection()
                pos_msg = PoseStamped()
                pos_msg.header.stamp = rospy.Time.now()
                pos_msg.header.frame_id = "camera"
                pos_msg.pose.position.x = mean_X
                pos_msg.pose.position.y = mean_Y
                pos_msg.pose.position.z = mean_Z
                pos_msg.pose.orientation = Quaternion(0, 0, 0, 1)
                msg.pose = pos_msg
                msg.type = 'center'
                msg.conf = 0.8  # 假设置信度为0.8
                msg.class_name = 'red'
                self.target_message.publish(msg)
            except Exception as e:
                rospy.logerr("Error publishing depth: %s", str(e))
        
            
            
        # ============================  获取蓝色球   ============================ #
        # lower_blue = np.array([100, 70, 50])
        # upper_blue = np.array([130, 255, 255])
        
        lower_blue = np.array([105, 150, 40])  # 更高的S，更低的V
        upper_blue = np.array([125, 255, 200])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        

        # 形态学操作清除噪声
        kernel = np.ones((5, 5), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # 骨架提取
        skeleton = self.get_skeleton(mask_clean)

        # 获取骨架上的像素点
        points = np.column_stack(np.where(skeleton > 0))  # [[y,x], ...]

        if len(points) < 5:
            rospy.logwarn("Not enough skeleton points found.")
            return

        # 选取等间距10个骨架点
        indices = np.linspace(0, len(points) - 1, 10, dtype=int)
        selected_points = points[indices]  # shape: (10, 2)

        coords_3d = []
        for idx, pt in enumerate(selected_points):
            u, v = pt  # 行是y，列是x
            rospy.loginfo(f"Processing point at pixel coordinates: idx={idx}, (u={u}, v={v})")
            # X, Y, Z = self.pixel_to_camera_coords(u, v, depth[y, x])
            X, Y, Z = get_stable_depth(u, v, depth, self.fx, self.fy, self.cx, self.cy)
            
            if (-1 < X < 1) and (-1 < Y < 1) and (0 < Z < 3):
                rospy.loginfo("Valid target:  class=blue -> X=%.2f Y=%.2f Z=%.2f", X, Y, Z)
                coords_3d.append([X, Y, Z])
            else:
                # rospy.logwarn("Invalid location of objection.")
                rospy.loginfo(f"Invalid target: ({X}, {Y}, {Z})")

        # 打印找到的最终目标的像素坐标和置信度
        if len(coords_3d) > 0:
            mean_X = np.mean([coord[0] for coord in coords_3d])
            mean_Y = np.mean([coord[1] for coord in coords_3d])
            mean_Z = np.mean([coord[2] for coord in coords_3d])
                    
            # 发布相机事件
            try:
                # depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
                # depth_msg.header = left_img_msg.header
                # self.depth_pub.publish(depth_msg)

                # 发布target message
                msg = TargetDetection()
                pos_msg = PoseStamped()
                pos_msg.header.stamp = rospy.Time.now()
                pos_msg.header.frame_id = "camera"
                pos_msg.pose.position.x = mean_X
                pos_msg.pose.position.y = mean_Y
                pos_msg.pose.position.z = mean_Z
                pos_msg.pose.orientation = Quaternion(0, 0, 0, 1)
                msg.pose = pos_msg
                msg.type = 'center'
                msg.conf = 0.8  # 假设置信度为0.8
                msg.class_name = 'blue'
                self.target_message.publish(msg)
            except Exception as e:
                rospy.logerr("Error publishing depth: %s", str(e))
        
        # for i, (X, Y, Z) in enumerate(coords_3d):
        #     rospy.loginfo(f"Point {i+1}: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
         
        
        # ============================  可视化（归一化视差）   ============================ #
        
        # 可视化分割结果和骨架线
        # seg_vis = cv2.bitwise_and(left_img, left_img, mask=mask_clean)

        # skeleton_vis = left_img.copy()
        # for y, x in np.column_stack(np.where(skeleton > 0)):
        #     cv2.circle(skeleton_vis, (x, y), 1, (0, 0, 255), -1)

        # # 本地调试时可使用imshow
        # cv2.imshow("Segmented Region", seg_vis)
        # cv2.imshow("Skeleton", skeleton_vis)
        # cv2.waitKey(1)

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
        BallsDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
