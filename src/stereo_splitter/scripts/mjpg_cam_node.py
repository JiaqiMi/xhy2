#!/home/nvidia/venvs/xhy_ros2/bin/python
# -*- coding: utf-8 -*-
"""MJPG USB 相机采集节点,替代 usb_cam。

背景:ROS-O (Ubuntu 22.04) 的 usb_cam 0.3.7 对本项目双目相机
(Sunplus 1bcf:0b15) 的 MJPEG 解码路径直接段错误,yuyv 模式受 USB2
带宽限制只有 1~10 fps。本节点用 V4L2 + OpenCV 直读 MJPG 压缩流,
1280x480 可跑满相机标称 30fps(暗光下受自动曝光限制会降低)。

话题与 usb_cam 保持一致(默认 /usb_cam/image_raw),下游
split_stereo_image.py 无需任何改动。
"""

import re
import subprocess

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def device_index(path):
    m = re.search(r"(\d+)$", str(path))
    return int(m.group(1)) if m else 0


class MjpgCamNode:
    def __init__(self):
        self.device = rospy.get_param("~video_device", "/dev/video0")
        self.width = int(rospy.get_param("~image_width", 1280))
        self.height = int(rospy.get_param("~image_height", 480))
        self.fps = float(rospy.get_param("~framerate", 30))
        self.frame_id = rospy.get_param("~frame_id", "usb_cam")
        self.topic = rospy.get_param("~topic", "/usb_cam/image_raw")
        # 可选: 传给 v4l2-ctl 的控制参数,如锁定曝光稳帧率:
        #   "-c auto_exposure=1 -c exposure_time_absolute=100"
        self.v4l2_ctl = str(rospy.get_param("~v4l2_ctl_args", "")).strip()
        self.pub = rospy.Publisher(self.topic, Image, queue_size=1)
        self.bridge = CvBridge()
        self.cap = None

    def open(self):
        cap = cv2.VideoCapture(device_index(self.device), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            return None
        if self.v4l2_ctl:
            subprocess.call(
                ["v4l2-ctl", "-d", self.device] + self.v4l2_ctl.split()
            )
        h, w = frame.shape[:2]
        rospy.loginfo(
            "mjpg_cam: %s MJPG %dx%d, 请求 %.0f fps -> %s",
            self.device, w, h, self.fps, self.topic,
        )
        if (w, h) != (self.width, self.height):
            rospy.logwarn(
                "mjpg_cam: 协商到 %dx%d, 与请求的 %dx%d 不一致",
                w, h, self.width, self.height,
            )
        return cap

    def spin(self):
        fails = 0
        while not rospy.is_shutdown():
            if self.cap is None:
                self.cap = self.open()
                if self.cap is None:
                    rospy.logerr_throttle(
                        5.0, "mjpg_cam: 无法打开 %s, 1s 后重试", self.device
                    )
                    rospy.sleep(1.0)
                    continue
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails >= 10:
                    rospy.logwarn("mjpg_cam: 连续读帧失败, 重新打开设备")
                    self.cap.release()
                    self.cap = None
                    fails = 0
                continue
            fails = 0
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = self.frame_id
            self.pub.publish(msg)
        if self.cap is not None:
            self.cap.release()


if __name__ == "__main__":
    rospy.init_node("mjpg_cam")
    try:
        MjpgCamNode().spin()
    except rospy.ROSInterruptException:
        pass
