#!/home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-

# 基于中心像素点的目标位姿计算
import rospy
import cv2
import tf
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from auv_control.msg import TargetDetection
# from stereo_depth.msg import TargetDetection
from geometry_msgs.msg import PointStamped,PoseStamped,Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber, Cache
from geometry_msgs.msg import Quaternion


def pixel_to_camera_coords(u, v, depth, fx, fy, cx, cy):
    """calculate the location in camera axis of any pixel in picture."""
    Z = depth[v][u]  # 注意 OpenCV 顺序是 (row, col) = (v, u)
    if Z == 0:
        # rospy.logwarn("Invalid depth at pixel ({}, {}): Z = 0".format(u, v))
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
    return X, Y, Z


class StereoDepthNode:
    def __init__(self):
        rospy.init_node("stereo_depth", anonymous=True)
        
        self.exp_env = rospy.get_param("~exp_env", "water") 
        self.visualization = rospy.get_param("~visualization", 0)
        self.conf_thre = rospy.get_param("~conf_thre", 0.5)
        
        if self.exp_env == "air":
            self.fx = 572.993971
            self.fy = 572.993971
            self.cx = 374.534946
            self.cy = 271.474743
            self.baseline = 34.309807 / 572.993971  # m
        elif self.exp_env == "water":
            self.fx = 798.731044
            self.fy = 798.731044
            self.cx = 348.127430
            self.cy = 269.935493
            self.baseline = 47.694354 / 798.731044  # m
        else:
            rospy.logerr("Invalid exp_env parameter: %s. Use 'air' or 'water'.", self.exp_env)
            rospy.signal_shutdown("Invalid exp_env parameter")
            return
        
        # Book the image message
        self.bridge = CvBridge()
        self.last_target_msg = None
        self.left_sub = Subscriber("/left/image_raw", Image)
        self.right_sub = Subscriber("/right/image_raw", Image)
        self.target_sub = Subscriber("/yolov8/target_center", PointStamped)
        self.target_cache = Cache(self.target_sub, cache_size=100)
        # rospy.Subscriber("/yolov8/target_center", PointStamped, self.target_callback)
        self.ts = ApproximateTimeSynchronizer([self.left_sub, self.right_sub], queue_size=5, slop=0.1)
        self.ts.registerCallback(self.callback)

        # Publish the target message
        self.target_message = rospy.Publisher("/obj/target_message", TargetDetection, queue_size=1)
        
        # 控制推断频率
        self.last_infer_time = rospy.Time.now()
        self.infer_interval = rospy.Duration(0.5)  # 单位秒 
        
        rospy.loginfo("Stereo Depth Node Initialized.")


    def target_callback(self, msg):
        """保存最新目标的像素位置"""
        self.target_uv = (int(msg.point.x), int(msg.point.y))
        self.target_conf = msg.point.z
        self.target_class = msg.header.frame_id
        self.target_check_time = msg.header.stamp

    def callback(self, left_img_msg, right_img_msg):
        
        now = rospy.Time.now()
        if now - self.last_infer_time < self.infer_interval:
            return  # 距离上次推理太近，跳过此次图像
        self.last_infer_time = now
        
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
        
        # 避免除以0
        disparity[disparity <= 0.0] = 0.1
        
        # 计算深度（Z = fx * baseline / disparity）, 单位：米
        depth = self.fx * self.baseline / disparity  
        
        # ============================  获取目标检测结果   ============================ #
        # 获取目标检测结果msg
        target_msg = self.target_cache.getElemBeforeTime(left_img_msg.header.stamp + rospy.Duration(1))
        if target_msg is not None:
            self.last_target_msg = target_msg
        elif self.last_target_msg:
            dt = (left_img_msg.header.stamp - self.last_target_msg.header.stamp).to_sec()
            if dt > 0.5:
                rospy.logwarn("Last target too old (%.2f s ago), skip.", dt)
                return
            else:
                rospy.loginfo("Using cached target (%.2f s old)", dt)
                target_msg = self.last_target_msg
        else:
            rospy.logwarn("No valid target_center found.")
            return
        
        # 获取目标的像素坐标和置信度
        u, v = int(target_msg.point.x), int(target_msg.point.y)
        target_conf = target_msg.point.z
        target_class = target_msg.header.frame_id
        check_time = target_msg.header.stamp

        # 打印找到的所有目标的像素坐标和置信度
        if target_conf is not None:
            rospy.loginfo("Find target: time=%s, class=%s conf=%.2f", check_time, target_class, target_conf)
            
        # 计算目标的相机坐标系位置
        X, Y, Z = None, None, None
        if target_conf >= self.conf_thre:
            # 获取稳定的深度值
            X, Y, Z = get_stable_depth(u, v, depth, self.fx, self.fy, self.cx, self.cy, window_size=11)
            # 过滤无效的深度值
            if (-1 < X < 1) and (-1 < Y < 1) and (0 < Z < 3):
                rospy.loginfo("Valid target: time=%s, class=%s conf=%.2f -> X=%.2f Y=%.2f Z=%.2f", \
                    check_time, target_class, target_conf, X, Y, Z)
                # 发布相机事件
                try:
                    msg = TargetDetection()
                    pos_msg = PoseStamped()
                    pos_msg.header.stamp = check_time
                    pos_msg.header.frame_id = "camera"
                    pos_msg.pose.position.x = X
                    pos_msg.pose.position.y = Y
                    pos_msg.pose.position.z = Z 
                    pos_msg.pose.orientation = Quaternion(0, 0, 0, 1)
                    msg.pose = pos_msg
                    msg.type = 'center'
                    msg.conf = target_conf
                    msg.class_name = target_class
                    self.target_message.publish(msg)
                except Exception as e:
                    rospy.logerr("Error publishing depth: %s", str(e))
            else:
                rospy.loginfo(
                    "Invalid target: time=%s, class=%s conf=%.2f -> X=%.2f Y=%.2f Z=%.2f", \
                    check_time, target_class, target_conf, X, Y, Z
                )
        else:
            rospy.logwarn("Confidence too low to calculate the location.")
        
        # ============================  可视化   ============================ #
        if self.visualization:
            # 可视化深度图
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth Map", depth_vis)

            # 可视化目标检测结果
            if X is not None and Y is not None:
                cv2.circle(left_img, (u, v), 5, (0, 255, 0), -1)

if __name__ == '__main__':
    try:
        StereoDepthNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

