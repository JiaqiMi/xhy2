#!/home/xhy/xhy_env/bin/python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Quaternion
from scipy.spatial.transform import Rotation as R
import tf

class ArucoPosePublisher:
    def __init__(self):
        rospy.init_node("aruco_pose_publisher", anonymous=True)

        # 初始化CvBridge
        self.bridge = CvBridge()

        # 相机内参（替换为实际标定值）
        # air
        # self.K = np.array([[519.1519, 0, 319.174292],
        #                    [0, 519.712551, 277.976296],
        #                    [0, 0, 1]], dtype=np.float64)
        # self.dist_coeffs = np.array([[-0.019985, 0.106889, 0.000070, 0.002679, 0.000000]], dtype=np.float64).T
        
        # water
        self.K = np.array([[686.32092, 0, 316.41091],
                           [0, 685.83026, 279.42833],
                           [0, 0, 1]], dtype=np.float64)
        self.dist_coeffs = np.array([[0.287829, 0.605589, 0.005716, -0.000247, 0.000000]], dtype=np.float64).T

        # ArUco 参数
        self.marker_length = 0.105  # 单位：米
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # 订阅图像
        self.image_sub = rospy.Subscriber("/left/image_raw", Image, self.image_callback, queue_size=1)

        # 发布姿态信息
        self.pose_pub = rospy.Publisher("/aruco/pose", PoseStamped, queue_size=10)

        rospy.loginfo("Aruco Pose Publisher Initialized.")
        
        # 控制推断频率
        self.last_infer_time = rospy.Time.now()
        self.infer_interval = rospy.Duration(2)  # 单位秒 

    def image_callback(self, img_msg):
        
        now = rospy.Time.now()
        if now - self.last_infer_time < self.infer_interval:
            return  # 距离上次推理太近，跳过此次图像

        self.last_infer_time = now
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge error: {e}")
            return

        # 检测 ArUco 标记
        corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict)

        if ids is not None:
            for i, corner in enumerate(corners):
                retval, rvec, tvec = cv2.solvePnP(
                    objectPoints=np.array([
                        [-0.5, 0.5, 0],
                        [0.5, 0.5, 0],
                        [0.5, -0.5, 0],
                        [-0.5, -0.5, 0]
                    ]) * self.marker_length,
                    imagePoints=corner[0],
                    cameraMatrix=self.K,
                    distCoeffs=self.dist_coeffs
                )

                # 将旋转向量转换为四元数
                R, _ = cv2.Rodrigues(rvec)
                quat = tf.transformations.quaternion_from_matrix(np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1])))

                # 发布位姿
                pose_msg = PoseStamped()
                pose_msg.header.stamp = img_msg.header.stamp
                pose_msg.header.frame_id = "camera"
                pose_msg.pose.position.x = tvec[0][0]
                pose_msg.pose.position.y = tvec[1][0]
                pose_msg.pose.position.z = tvec[2][0]
                pose_msg.pose.orientation = Quaternion(*quat)

                self.pose_pub.publish(pose_msg)

                rospy.loginfo(f"Published pose for ArUco ID={ids[i][0]}: tvec={tvec.ravel()}, quat={quat}")
                
                rospy.sleep(0.2)

if __name__ == '__main__':
    try:
        ArucoPosePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass