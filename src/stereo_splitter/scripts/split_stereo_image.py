#!/home/xhy/xhy_env/bin/python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class StereoSplitter:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.callback)
        self.left_pub = rospy.Publisher("/left/image_raw", Image, queue_size=1)
        self.right_pub = rospy.Publisher("/right/image_raw", Image, queue_size=1)

    def callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            height, width, _ = cv_image.shape
            mid = width // 2
            left = cv_image[:, :mid]
            right = cv_image[:, mid:]

            left_msg = self.bridge.cv2_to_imgmsg(left, encoding="bgr8")
            right_msg = self.bridge.cv2_to_imgmsg(right, encoding="bgr8")

            left_msg.header = msg.header
            right_msg.header = msg.header

            self.left_pub.publish(left_msg)
            self.right_pub.publish(right_msg)
        except Exception as e:
            rospy.logerr("Image split failed: %s", e)

if __name__ == '__main__':
    rospy.init_node('stereo_splitter')
    splitter = StereoSplitter()
    rospy.spin()
