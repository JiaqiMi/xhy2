#!/home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
"""
Unified Stereo Depth & Pose Estimation Node
Supports three modes via ROS parameters:
    mode: 1=center, 2=quad corners, 3=line points
    exp_env: air|water
    conf_thre: confidence threshold
    is_visual: 0|1
"""
import rospy
import cv2
import tf
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Custom msgs
from auv_control.msg import TargetDetection, TargetDetection3
from stereo_depth.msg import BoundingBox, LineBox


def get_stable_depth(u, v, depth, fx, fy, cx, cy, window_size=11):
    half = window_size // 2
    h, w = depth.shape
    u, v = int(u), int(v)
    region = depth[max(v-half, 0):min(v+half+1, h), max(u-half, 0):min(u+half+1, w)]
    valid = region[np.isfinite(region) & (region > 0)]
    if valid.size < 3:
        return np.array([np.nan, np.nan, np.nan])
    valid = valid[(valid >= 0.5) & (valid <= 3)]
    Z = np.mean(valid)
    if Z <= 0:
        return np.array([np.nan, np.nan, np.nan])
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])


def compute_pose_from_quad(P1, P2, P3, P4):
    """计算四个点的中心位置和姿态"""
    center = None
    dis_thre = 3   # 单位米
    if (P1[2] < dis_thre) and (P2[2] < dis_thre) and (P3[2] < dis_thre) and (P4[2] < dis_thre):
        center = (P1 + P2 + P3 + P4) / 4.0
    elif (P1[2] < dis_thre) and (P3[2] < dis_thre):
        center = (P1 + P3) / 2.0
    elif (P2[2] < dis_thre) and (P4[2] < dis_thre):
        center = (P2 + P4) / 2.0
    elif (P1[2] < dis_thre) and (P2[2] < dis_thre) and (P3[2]<dis_thre):
        center = (P1 + P3) / 2.0
    elif (P1[2] < dis_thre) and (P2[2] < dis_thre) and (P4[2]<dis_thre):
        center = (P2 + P4) / 2.0
    elif (P1[2] < dis_thre) and (P3[2] < dis_thre) and (P4[2]<dis_thre):
        center = (P1 + P3) / 2.0
    elif (P2[2] < dis_thre) and (P3[2] < dis_thre) and (P4[2]<dis_thre):
        center = (P2 + P4) / 2.0
    else:
        rospy.logwarn("Invalid depth for all points, cannot compute pose.")
        return None
    
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
        rospy.init_node('stereo_depth_node', anonymous=True)
        # Load parameters via rospy.get_param
        self.mode = rospy.get_param('~mode', 1)
        self.exp_env = rospy.get_param('~exp_env', 'water')
        self.conf_thre = rospy.get_param('~conf_thre', 0.5)
        self.is_visual = rospy.get_param('~is_visual', 0)

        # Camera intrinsics
        # if self.exp_env == 'air':
        #     self.fx = 572.993971
        #     self.fy = 572.993971
        #     self.cx = 374.534946
        #     self.cy = 271.474743
        #     self.baseline = 34.309807 / self.fx
        # elif self.exp_env == 'water':
        #     self.fx = 798.731044
        #     self.fy = 798.731044
        #     self.cx = 348.127430
        #     self.cy = 269.935493
        #     self.baseline = 47.694354 / self.fx
        # else:
        #     rospy.logerr(f"Invalid exp_env: {self.exp_env}, use water or air.")
        #     return 
        
        # 相机标定参数（从你的 YAML 拷贝过来）
        # 左相机
        K1 = np.array([[519.1519, 0,       319.174292],
                       [0,        519.712551,277.976296],
                       [0,        0,         1]], dtype=np.float64)
        D1 = np.array([-0.019985, 0.106889, 0.000070, 0.002679, 0], dtype=np.float64)
        R1 = np.array([[0.997406,  0.009347, -0.071366],
                       [-0.009146, 0.999953,  0.003147],
                       [0.071392, -0.002486,  0.997445]], dtype=np.float64)
        P1 = np.array([[572.993971, 0,         374.534946, 0],
                       [0,          572.993971,271.474743, 0],
                       [0,          0,         1,          0]], dtype=np.float64)
        # 右相机
        K2 = np.array([[523.139499, 0,         319.655966],
                       [0,          523.394088,267.733782],
                       [0,          0,         1]], dtype=np.float64)
        D2 = np.array([0.000599, 0.071806, 0.002352, 0.001851, 0], dtype=np.float64)
        R2 = np.array([[0.996861,  0.008680, -0.078688],
                       [-0.008901, 0.999957, -0.002469],
                       [0.078663,  0.003162,  0.996896]], dtype=np.float64)
        P2 = np.array([[572.993971, 0,         374.534946, -34.309807],
                       [0,          572.993971,271.474743,  0],
                       [0,          0,         1,           0]], dtype=np.float64)

        # 初始化去畸变+校正映射表
        img_size = (640, 480)
        self.left_map1, self.left_map2   = cv2.initUndistortRectifyMap(
            K1, D1, R1, P1[:3,:3], img_size, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            K2, D2, R2, P2[:3,:3], img_size, cv2.CV_16SC2)

        # 基线（只要一个就行）
        self.fx       = P1[0,0]
        self.fy       = P1[1,1]
        self.cx       = P1[0,2]
        self.cy       = P1[1,2]
        self.baseline = abs(P2[0,3]) / self.fx

        self.bridge = CvBridge()
        self.rate = rospy.Rate(5.0)
        self.left_img = None
        self.right_img = None
        self.reset_target()

        # Subscribers for stereo images
        left_sub = Subscriber('/left/image_raw', Image)
        right_sub = Subscriber('/right/image_raw', Image)
        self.ts = ApproximateTimeSynchronizer([left_sub, right_sub], 5, 0.1)
        self.ts.registerCallback(self.img_callback)

        # Mode-specific topic subscriptions & publishers
        if self.mode == 1:
            rospy.Subscriber('/yolov8/target_center', PointStamped, self.cb_center)
            self.pub = rospy.Publisher('/obj/target_message', TargetDetection, queue_size=1)
        elif self.mode == 2:
            rospy.Subscriber('/yolov8/target_bbox', BoundingBox, self.cb_bbox)
            self.pub = rospy.Publisher('/obj/target_message', TargetDetection, queue_size=1)
        elif self.mode == 3:
            rospy.Subscriber('/yolov8/line_bbox', LineBox, self.cb_line)
            self.pub = rospy.Publisher('/obj/line_message', TargetDetection3, queue_size=1)
        else:
            rospy.logerr('Invalid mode: %d. Use 1, 2, or 3.', self.mode)
            return 
        
        rospy.loginfo('StereoDepthNode initialized in mode %d (%s env)', self.mode, self.exp_env)

    def reset_target(self):
        self.u1 = self.v1 = self.u2 = self.v2 = self.u3 = self.v3 = None
        self.conf = None; self.cls = None; self.tstamp = None

    def cb_center(self, msg):
        self.u1, self.v1 = int(msg.point.x), int(msg.point.y)
        self.conf = msg.point.z; self.cls = msg.header.frame_id; self.tstamp = msg.header.stamp

    def cb_bbox(self, msg):
        self.u1, self.v1, self.u2, self.v2 = msg.x1, msg.y1, msg.x2, msg.y2
        self.conf, self.cls, self.tstamp = msg.conf, msg.header.frame_id, msg.header.stamp

    def cb_line(self, msg):
        self.u1, self.v1 = msg.x1, msg.y1
        self.u2, self.v2 = msg.x2, msg.y2
        self.u3, self.v3 = msg.x3, msg.y3
        self.conf, self.cls, self.tstamp = msg.conf, msg.header.frame_id, msg.header.stamp

    def img_callback(self, left_msg, right_msg):
        try:
            self.left_img = self.bridge.imgmsg_to_cv2(left_msg, 'bgr8')
            self.right_img = self.bridge.imgmsg_to_cv2(right_msg, 'bgr8')
        except cv2.error as e:
            rospy.logerr('cv_bridge error: %s', str(e))

    def run(self):
        while not rospy.is_shutdown():
            if self.left_img is None or self.conf is None:
                self.rate.sleep(); 
                continue

            # grayL = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY)
            # grayR = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2GRAY)
            # stereo = cv2.StereoSGBM_create(
            #     minDisparity=0, numDisparities=96, blockSize=7,
            #     P1=8*3*7**2, P2=32*3*7**2,
            #     disp12MaxDiff=1, uniquenessRatio=10,
            #     speckleWindowSize=100, speckleRange=32)
            # disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
            # disp[disp <= 0] = 0.1
            # depth = self.fx * self.baseline / disp
            
            # 1) 去畸变+校正
            left_rect  = cv2.remap(self.left_img,  self.left_map1,  self.left_map2,  cv2.INTER_LINEAR)
            right_rect = cv2.remap(self.right_img, self.right_map1, self.right_map2, cv2.INTER_LINEAR)

            # 2) 生成视差 & 深度
            grayL = cv2.cvtColor(left_rect,  cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
            stereo = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=96, blockSize=7,
                P1=8*3*7**2, P2=32*3*7**2,
                disp12MaxDiff=1, uniquenessRatio=10,
                speckleWindowSize=100, speckleRange=32)
            disp  = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
            disp[disp <= 0] = 0.1
            depth = self.fx * self.baseline / disp            

            if self.mode == 1:
                if self.conf < self.conf_thre:
                    rospy.logwarn('Low conf center: %.2f', self.conf)
                else:
                    P = get_stable_depth(self.u1, self.v1, depth, self.fx, self.fy, self.cx, self.cy)
                    if np.all(np.isfinite(P)):
                        msg = TargetDetection()
                        pose = PoseStamped(); pose.header.stamp = self.tstamp; pose.header.frame_id = 'camera'
                        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = P
                        pose.pose.orientation = Quaternion(0,0,0,1)
                        msg.pose = pose; msg.type = 'center'; msg.conf = self.conf; msg.class_name = self.cls
                        self.pub.publish(msg)

            elif self.mode == 2:
                if self.conf < self.conf_thre:
                    rospy.logwarn('Low conf bbox: %.2f', self.conf)
                else:
                    P1 = get_stable_depth(self.u1, self.v1, depth, self.fx, self.fy, self.cx, self.cy)
                    P2 = get_stable_depth(self.u2, self.v1, depth, self.fx, self.fy, self.cx, self.cy)
                    P3 = get_stable_depth(self.u2, self.v2, depth, self.fx, self.fy, self.cx, self.cy)
                    P4 = get_stable_depth(self.u1, self.v2, depth, self.fx, self.fy, self.cx, self.cy)
                    pose = compute_pose_from_quad(P1, P2, P3, P4)
                    if pose:
                        msg = TargetDetection()
                        msg.pose = pose; msg.type = 'center'; msg.conf = self.conf; msg.class_name = self.cls
                        self.pub.publish(msg)

            else:
                if self.conf < self.conf_thre:
                    rospy.logwarn('Low conf line: %.2f', self.conf)
                else:
                    P1 = get_stable_depth(self.u3, self.v3, depth, self.fx, self.fy, self.cx, self.cy)
                    P2 = get_stable_depth(self.u2, self.v2, depth, self.fx, self.fy, self.cx, self.cy)
                    P3 = get_stable_depth(self.u1, self.v1, depth, self.fx, self.fy, self.cx, self.cy)
                    if np.all(np.isfinite(P1)) and np.all(np.isfinite(P2)) and np.all(np.isfinite(P3)):
                        msg = TargetDetection3()
                        # fill poses
                        msg.pose1 = PoseStamped(header=rospy.Header(stamp=self.tstamp, frame_id='camera'))
                        msg.pose1.pose.position.x, msg.pose1.pose.position.y, msg.pose1.pose.position.z = P1
                        msg.pose2 = PoseStamped(header=rospy.Header(stamp=self.tstamp, frame_id='camera'))
                        msg.pose2.pose.position.x, msg.pose2.pose.position.y, msg.pose2.pose.position.z = P2
                        msg.pose3 = PoseStamped(header=rospy.Header(stamp=self.tstamp, frame_id='camera'))
                        msg.pose3.pose.position.x, msg.pose3.pose.position.y, msg.pose3.pose.position.z = P3
                        # set orientations
                        for p in (msg.pose1, msg.pose2, msg.pose3): p.pose.orientation = Quaternion(0,0,0,1)
                        msg.type = 'center'; msg.conf = self.conf; msg.class_name = self.cls
                        self.pub.publish(msg)

            if self.is_visual:
                # visualization omitted for brevity
                pass

            self.reset_target()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        node = StereoDepthNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
