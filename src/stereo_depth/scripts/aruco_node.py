#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

import json

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion
from scipy.spatial.transform import Rotation as Rotation
from sensor_msgs.msg import Image
from std_msgs.msg import String


class ArucoPosePublisher:
    def __init__(self):
        rospy.init_node("aruco_pose_publisher", anonymous=False)

        self.bridge = CvBridge()

        self.image_topic = rospy.get_param(
            "~image_topic", "/left/image_raw"
        )
        self.pose_topic = rospy.get_param(
            "~pose_topic", "/aruco/pose"
        )
        self.annotated_topic = rospy.get_param(
            "~annotated_topic", "/aruco/annotated_image"
        )
        self.web_detection_topic = rospy.get_param(
            "~web_detection_topic", "/web/detections"
        )
        self.web_pose_topic = rospy.get_param(
            "~web_pose_topic", "/web/pose"
        )

        self.exp_env = str(
            rospy.get_param("~exp_env", "air")
        ).strip().lower()

        self.marker_length = float(
            rospy.get_param("~marker_length", 0.2)
        )
        self.dictionary_name = str(
            rospy.get_param(
                "~dictionary", "DICT_4X4_1000"
            )
        ).strip()

        self.infer_rate = max(
            0.5,
            float(rospy.get_param("~infer_rate", 5.0)),
        )
        self.visualization = int(
            rospy.get_param("~visualization", 0)
        )
        self.camera_frame = str(
            rospy.get_param("~camera_frame", "camera")
        )
        self.primary_marker_policy = str(
            rospy.get_param(
                "~primary_marker_policy", "nearest"
            )
        ).strip().lower()

        self.K, self.dist_coeffs = self._load_camera_parameters()
        self.aruco_dict = self._create_aruco_dictionary()
        self.detector_parameters = self._create_detector_parameters()

        self.axis_length = float(
            rospy.get_param(
                "~axis_length", self.marker_length * 0.5
            )
        )

        self.last_infer_time = rospy.Time(0)
        self.infer_interval = rospy.Duration(
            1.0 / self.infer_rate
        )

        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

        # 原有ROS位姿话题
        self.pose_pub = rospy.Publisher(
            self.pose_topic,
            PoseStamped,
            queue_size=10,
        )

        # Web所需话题
        self.annotated_pub = rospy.Publisher(
            self.annotated_topic,
            Image,
            queue_size=1,
        )
        self.web_detection_pub = rospy.Publisher(
            self.web_detection_topic,
            String,
            queue_size=1,
        )
        self.web_pose_pub = rospy.Publisher(
            self.web_pose_topic,
            String,
            queue_size=1,
        )

        rospy.loginfo("Aruco Pose Publisher initialized")
        rospy.loginfo("image_topic=%s", self.image_topic)
        rospy.loginfo("pose_topic=%s", self.pose_topic)
        rospy.loginfo("annotated_topic=%s", self.annotated_topic)
        rospy.loginfo(
            "web_detection_topic=%s",
            self.web_detection_topic,
        )
        rospy.loginfo("web_pose_topic=%s", self.web_pose_topic)
        rospy.loginfo(
            "env=%s, marker_length=%.3f m, dictionary=%s, rate=%.2f Hz",
            self.exp_env,
            self.marker_length,
            self.dictionary_name,
            self.infer_rate,
        )

    def _load_camera_parameters(self):
        """
        优先使用ROS参数：
          ~camera_matrix: 9个数
          ~dist_coeffs: 4或5个数

        未提供时使用air/water预设值。
        """
        custom_k = rospy.get_param("~camera_matrix", [])
        custom_dist = rospy.get_param("~dist_coeffs", [])

        if len(custom_k) == 9 and len(custom_dist) >= 4:
            camera_matrix = np.asarray(
                custom_k, dtype=np.float64
            ).reshape(3, 3)
            dist_coeffs = np.asarray(
                custom_dist, dtype=np.float64
            ).reshape(-1, 1)
            rospy.loginfo(
                "Using custom camera parameters from ROS"
            )
            return camera_matrix, dist_coeffs

        if self.exp_env == "air":
            camera_matrix = np.array(
                [
                    [519.1519, 0.0, 319.174292],
                    [0.0, 519.712551, 277.976296],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist_coeffs = np.array(
                [
                    -0.019985,
                    0.106889,
                    0.000070,
                    0.002679,
                    0.000000,
                ],
                dtype=np.float64,
            ).reshape(-1, 1)

        elif self.exp_env == "water":
            camera_matrix = np.array(
                [
                    [686.32092, 0.0, 316.41091],
                    [0.0, 685.83026, 279.42833],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            dist_coeffs = np.array(
                [
                    0.287829,
                    0.605589,
                    0.005716,
                    -0.000247,
                    0.000000,
                ],
                dtype=np.float64,
            ).reshape(-1, 1)

        else:
            raise ValueError(
                "exp_env must be air or water, got: {}".format(
                    self.exp_env
                )
            )

        rospy.loginfo(
            "Using preset %s camera parameters",
            self.exp_env,
        )
        return camera_matrix, dist_coeffs

    def _create_aruco_dictionary(self):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco is unavailable; install opencv-contrib-python"
            )

        if not hasattr(cv2.aruco, self.dictionary_name):
            raise ValueError(
                "Unsupported ArUco dictionary: {}".format(
                    self.dictionary_name
                )
            )

        dictionary_id = getattr(
            cv2.aruco, self.dictionary_name
        )
        return cv2.aruco.getPredefinedDictionary(
            dictionary_id
        )

    @staticmethod
    def _create_detector_parameters():
        if hasattr(
            cv2.aruco, "DetectorParameters_create"
        ):
            return cv2.aruco.DetectorParameters_create()
        return cv2.aruco.DetectorParameters()

    @staticmethod
    def _valid_stamp(header):
        if header.stamp == rospy.Time():
            return rospy.Time.now()
        return header.stamp

    def _detect_markers(self, image):
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(
                self.aruco_dict,
                self.detector_parameters,
            )
            return detector.detectMarkers(image)

        return cv2.aruco.detectMarkers(
            image,
            self.aruco_dict,
            parameters=self.detector_parameters,
        )

    def _solve_marker_pose(self, corner):
        half = self.marker_length / 2.0

        object_points = np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

        image_points = np.asarray(
            corner, dtype=np.float64
        ).reshape(4, 2)

        flag = getattr(
            cv2,
            "SOLVEPNP_IPPE_SQUARE",
            cv2.SOLVEPNP_ITERATIVE,
        )

        success, rvec, tvec = cv2.solvePnP(
            objectPoints=object_points,
            imagePoints=image_points,
            cameraMatrix=self.K,
            distCoeffs=self.dist_coeffs,
            flags=flag,
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        quat_xyzw = Rotation.from_matrix(
            rotation_matrix
        ).as_quat()

        return rvec, tvec, quat_xyzw

    def _draw_axis(self, image, rvec, tvec):
        try:
            if hasattr(cv2, "drawFrameAxes"):
                cv2.drawFrameAxes(
                    image,
                    self.K,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.axis_length,
                    2,
                )
            elif hasattr(cv2.aruco, "drawAxis"):
                cv2.aruco.drawAxis(
                    image,
                    self.K,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.axis_length,
                )
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0,
                "Failed to draw ArUco axis: %s",
                str(exc),
            )

    def _publish_pose(self, stamp, position, quaternion):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.camera_frame

        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])

        pose_msg.pose.orientation = Quaternion(
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        )

        self.pose_pub.publish(pose_msg)

    def _select_primary(self, detections):
        if not detections:
            return None

        if self.primary_marker_policy == "lowest_id":
            return min(
                detections,
                key=lambda item: item["marker_id"],
            )

        return min(
            detections,
            key=lambda item: item["distance_m"],
        )

    def _publish_web(self, stamp, header, image, detections):
        detection_payload = {
            "stamp": stamp.to_sec(),
            "source": "aruco",
            "node": "aruco_pose_publisher",
            "frame_id": header.frame_id,
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "count": len(detections),
            "detections": detections,
        }

        self.web_detection_pub.publish(
            String(
                data=json.dumps(
                    detection_payload,
                    ensure_ascii=False,
                )
            )
        )

        primary = self._select_primary(detections)

        if primary is None:
            pose_payload = {
                "stamp": stamp.to_sec(),
                "source": "aruco",
                "frame_id": self.camera_frame,
                "valid": False,
                "reason": "no_marker_detected",
            }
        else:
            pose_payload = {
                "stamp": stamp.to_sec(),
                "source": "aruco",
                "frame_id": self.camera_frame,
                "marker_id": primary["marker_id"],
                "class_name": primary["class_name"],
                "confidence": 1.0,
                "valid": True,
                "pixel_center": primary["center"],
                "position_m": primary["position_m"],
                "orientation_xyzw": primary[
                    "orientation_xyzw"
                ],
                "distance_m": primary["distance_m"],
            }

        self.web_pose_pub.publish(
            String(
                data=json.dumps(
                    pose_payload,
                    ensure_ascii=False,
                )
            )
        )

    def _publish_annotated(self, image, header, stamp):
        try:
            msg = self.bridge.cv2_to_imgmsg(
                image,
                encoding="bgr8",
            )
            msg.header = header
            msg.header.stamp = stamp
            self.annotated_pub.publish(msg)

        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "Failed to publish annotated image: %s",
                str(exc),
            )

    def image_callback(self, img_msg):
        now = rospy.Time.now()

        if (
            self.last_infer_time != rospy.Time(0)
            and now - self.last_infer_time
            < self.infer_interval
        ):
            return

        self.last_infer_time = now

        try:
            image = self.bridge.imgmsg_to_cv2(
                img_msg,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "CvBridge error: %s",
                str(exc),
            )
            return

        annotated = image.copy()
        stamp = self._valid_stamp(img_msg.header)

        try:
            corners, ids, _ = self._detect_markers(image)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "ArUco detection failed: %s",
                str(exc),
            )
            return

        detections = []

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(
                annotated,
                corners,
                ids,
            )

            ids_flat = ids.reshape(-1)

            for index, corner in enumerate(corners):
                marker_id = int(ids_flat[index])
                pose = self._solve_marker_pose(corner[0])

                if pose is None:
                    rospy.logwarn(
                        "solvePnP failed for ArUco ID=%d",
                        marker_id,
                    )
                    continue

                rvec, tvec, quat_xyzw = pose
                position = np.asarray(
                    tvec,
                    dtype=np.float64,
                ).reshape(3)

                center = np.mean(
                    np.asarray(corner[0]),
                    axis=0,
                )
                distance = float(
                    np.linalg.norm(position)
                )

                self._publish_pose(
                    stamp,
                    position,
                    quat_xyzw,
                )
                self._draw_axis(
                    annotated,
                    rvec,
                    tvec,
                )

                item = {
                    "marker_id": marker_id,
                    "class_id": marker_id,
                    "class_name": "ArUco ID {}".format(
                        marker_id
                    ),
                    "confidence": 1.0,
                    "center": {
                        "u": int(round(center[0])),
                        "v": int(round(center[1])),
                    },
                    "corners": [
                        {
                            "u": round(float(point[0]), 2),
                            "v": round(float(point[1]), 2),
                        }
                        for point in corner[0]
                    ],
                    "position_m": {
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                    },
                    "orientation_xyzw": {
                        "x": float(quat_xyzw[0]),
                        "y": float(quat_xyzw[1]),
                        "z": float(quat_xyzw[2]),
                        "w": float(quat_xyzw[3]),
                    },
                    "distance_m": distance,
                    "task": "aruco_pose",
                    "output_type": "pose",
                }

                detections.append(item)

                cv2.putText(
                    annotated,
                    "ID {} Z={:.2f}m".format(
                        marker_id,
                        position[2],
                    ),
                    (
                        int(round(center[0])) + 8,
                        int(round(center[1])) - 8,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        self._publish_web(
            stamp,
            img_msg.header,
            image,
            detections,
        )
        self._publish_annotated(
            annotated,
            img_msg.header,
            stamp,
        )

        primary = self._select_primary(detections)
        if primary is not None:
            position = primary["position_m"]
            rospy.loginfo_throttle(
                1.0,
                (
                    "ArUco primary ID={}: "
                    "X={:.3f}, Y={:.3f}, Z={:.3f} m"
                ).format(
                    primary["marker_id"],
                    position["x"],
                    position["y"],
                    position["z"],
                ),
            )

        if self.visualization == 1:
            cv2.imshow(
                "ArUco Detection",
                annotated,
            )
            cv2.waitKey(1)


if __name__ == "__main__":
    try:
        node = ArucoPosePublisher()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass

    finally:
        cv2.destroyAllWindows()