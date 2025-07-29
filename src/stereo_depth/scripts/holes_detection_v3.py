#!/home/xhy/xhy_env/bin/python

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from stereo_msgs.msg import DisparityImage
from geometry_msgs.msg import PointStamped
from stereo_depth.msg import TargetDetection, BoundingBox

def pixel_to_camera_coords(u, v, disparity_map, fx, fy, cx, cy, baseline):
    disparity = disparity_map[int(v), int(u)]
    if disparity <= 0.0 or np.isnan(disparity):
        return np.array([None, None, None])
    Z = fx * baseline / disparity
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

class DisparityToPoint:
    def __init__(self):
        rospy.init_node("disparity_to_point_node")
        self.bridge = CvBridge()
        
        self.fx = rospy.get_param("~fx", 572.993971)
        self.fy = rospy.get_param("~fy", 572.993971)
        self.cx = rospy.get_param("~cx", 374.534946)
        self.cy = rospy.get_param("~cy", 271.474743)
        self.baseline = rospy.get_param("~baseline", 34.309807 / 572.993971)  # baseline in meters

        self.target_uv = (640, 240)  # 可以替换成目标点（u,v）
        self.target_conf = 0.9       # 设置一个用于判断目标可信度的阈值
        self.target_class = 'None'       # 设置一个用于判断目标可信度的阈值
        
        # Book YOLO center pixel 
        rospy.Subscriber("/yolov8/target_bbox", BoundingBox, self.target_callback)

        rospy.Subscriber("/disparity", DisparityImage, self.disparity_callback, queue_size=1)
        self.pub = rospy.Publisher("/target_point", PointStamped, queue_size=1)
        
        
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


    def disparity_callback(self, msg):
        disparity = None
        try:
            disparity = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="32FC1")
        except Exception as e:
            rospy.logerr("Failed to convert disparity image: %s", str(e))
            return
        
        # 判断是否有来自YOLO的像素坐标
        P1, P2, P3, P4 = None, None, None, None
        
        if self.target_conf >= 0.5:
            
            rospy.loginfo("Target detected: class=%s, conf=%.2f, x1: %d, y1: %d, x2: %d, y2: %d",
                          self.target_class, self.target_conf,
                          self.target_x1, self.target_y1, self.target_x2, self.target_y2)
            
            if (0 <= self.target_x1 <= disparity.shape[1]) and (0 <= self.target_y1 <= disparity.shape[0]) and \
                (0 <= self.target_x2 <= disparity.shape[1]) and (0 <= self.target_y2 <= disparity.shape[0]):
                
                # X, Y, Z = pixel_to_camera_coords(u, v, disparity, self.fx, self.fy, self.cx, self.cy, self.baseline)
                # 获取每个角点的相机坐标
                P1 = pixel_to_camera_coords(self.target_x1, self.target_y1, disparity, self.fx, self.fy, self.cx, self.cy, self.baseline)
                P2 = pixel_to_camera_coords(self.target_x2, self.target_y1, disparity, self.fx, self.fy, self.cx, self.cy, self.baseline)
                P3 = pixel_to_camera_coords(self.target_x2, self.target_y2, disparity, self.fx, self.fy, self.cx, self.cy, self.baseline)
                P4 = pixel_to_camera_coords(self.target_x1, self.target_y2, disparity, self.fx, self.fy, self.cx, self.cy, self.baseline)
                # if X is not None:
                #     point_msg = PointStamped()
                #     point_msg.header = msg.header
                #     point_msg.point.x = X
                #     point_msg.point.y = Y
                #     point_msg.point.z = Z
                #     self.pub.publish(point_msg)
                #     rospy.loginfo(f"[INFO] Valid target: class=custom conf={self.target_conf:.2f} -> X={X:.2f} Y={Y:.2f} Z={Z:.2f}")
                # else:
                #     rospy.logwarn("Disparity at target is invalid (zero or NaN)")
                
                rospy.loginfo("class_name: %s, P1.X: %.2f, P1.Y: %.2f, P1.Z: %.2f", self.target_class, P1[0], P1[1], P1[2])
                rospy.loginfo("class_name: %s, P2.X: %.2f, P2.Y: %.2f, P2.Z: %.2f", self.target_class, P2[0], P2[1], P2[2])
                rospy.loginfo("class_name: %s, P3.X: %.2f, P3.Y: %.2f, P3.Z: %.2f", self.target_class, P3[0], P3[1], P3[2])
                rospy.loginfo("class_name: %s, P4.X: %.2f, P4.Y: %.2f, P4.Z: %.2f", self.target_class, P4[0], P4[1], P4[2])
                
            else:
                rospy.logwarn("Target pixel out of bounds")

if __name__ == "__main__":
    try:
        DisparityToPoint()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
