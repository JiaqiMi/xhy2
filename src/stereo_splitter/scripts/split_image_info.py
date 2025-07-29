#!/home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-

import rospy
import yaml
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from std_msgs.msg import Header

class StereoSplitter:
    def __init__(self):
        rospy.init_node('split_stereo_image')

        # 参数加载
        self.left_yaml = rospy.get_param("~left_camera_yaml", "/home/xhy/catkin_ws/camera/left_nc_water_0727.yaml")
        self.right_yaml = rospy.get_param("~right_camera_yaml", "/home/xhy/catkin_ws/camera/left_nc_water_0727.yaml")
        self.image_topic = rospy.get_param("~input_image_topic", "/usb_cam/image_raw")

        # 加载 CameraInfo
        self.left_info = self.load_camera_info(self.left_yaml)
        self.right_info = self.load_camera_info(self.right_yaml)

        # 发布器
        self.bridge = CvBridge()
        self.left_pub = rospy.Publisher("/left/image_raw", Image, queue_size=1)
        self.right_pub = rospy.Publisher("/right/image_raw", Image, queue_size=1)
        self.left_info_pub = rospy.Publisher("/left/camera_info", CameraInfo, queue_size=1)
        self.right_info_pub = rospy.Publisher("/right/camera_info", CameraInfo, queue_size=1)

        # 订阅拼接图像
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.loginfo("StereoSplitter initialized. Listening to %s" % self.image_topic)

    def load_camera_info(self, yaml_file):
        with open(yaml_file, 'r') as f:
            calib_data = yaml.safe_load(f)

        info = CameraInfo()
        info.width = calib_data["image_width"]
        info.height = calib_data["image_height"]
        info.K = calib_data["camera_matrix"]["data"]
        info.D = calib_data["distortion_coefficients"]["data"]
        info.R = calib_data["rectification_matrix"]["data"]
        info.P = calib_data["projection_matrix"]["data"]
        info.distortion_model = calib_data["distortion_model"]
        return info

    def image_callback(self, msg):
        # 拆图像
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", e)
            return

        # 假设图像尺寸为1280x480（水平拼接）
        height, width, _ = img.shape
        mid = width // 2

        left = img[:, :mid]
        right = img[:, mid:]

        # 构造消息头
        header = Header()
        # header.stamp = msg.header.stamp
        header.stamp = rospy.Time.now()
        header.frame_id = "camera"

        # 发布图像和CameraInfo
        self.left_info.header = header
        self.right_info.header = header
        # self.left_info.header = header
        # self.right_info.header = header
        
        left_msg = self.bridge.cv2_to_imgmsg(left, encoding="bgr8")
        right_msg = self.bridge.cv2_to_imgmsg(right, encoding="bgr8")
        left_msg.header = header
        right_msg.header = header
        

        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)
        self.left_info_pub.publish(self.left_info)
        self.right_info_pub.publish(self.right_info)
        
        # rospy.loginfo("Published left and right images with CameraInfo.")

if __name__ == '__main__':
    try:
        StereoSplitter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
