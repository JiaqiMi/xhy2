#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

import json
import math
import time

import cv2
import numpy as np
import rospy
from auv_control.msg import TargetDetection
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Quaternion
from sensor_msgs.msg import Image
from std_msgs.msg import String


def rotation_matrix_to_quaternion(matrix):
    """Return quaternion [x, y, z, w] from a 3x3 rotation matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quat / norm


class FisheyeArucoNode:
    def __init__(self):
        rospy.init_node("fisheye_aruco_node", anonymous=False)
        self.bridge = CvBridge()

        # Topics
        self.image_topic = rospy.get_param(
            "~image_topic", "/fisheye_camera/image_raw"
        )
        self.pose_topic = rospy.get_param("~pose_topic", "/aruco/pose")
        self.annotated_topic = rospy.get_param(
            "~annotated_topic", "/aruco/annotated_image"
        )
        self.rectified_topic = rospy.get_param(
            "~rectified_topic", "/fisheye_camera/image_rect"
        )
        self.web_detection_topic = rospy.get_param(
            "~web_detection_topic", "/web/detections"
        )
        self.web_pose_topic = rospy.get_param(
            "~web_pose_topic", "/web/pose"
        )
        self.target_output_topic = rospy.get_param(
            "~target_output_topic", "/obj/target_message"
        )

        # ArUco settings
        self.marker_length = float(rospy.get_param("~marker_length", 0.20))
        self.dictionary_name = str(
            rospy.get_param("~dictionary", "DICT_4X4_1000")
        ).strip()
        self.infer_rate = max(
            0.2, float(rospy.get_param("~infer_rate", 5.0))
        )
        self.visualization = int(rospy.get_param("~visualization", 0))
        self.camera_frame = str(
            rospy.get_param("~camera_frame", "fisheye_camera")
        ).strip()
        self.primary_marker_policy = str(
            rospy.get_param("~primary_marker_policy", "nearest")
        ).strip().lower()

        # Camera model settings
        self.camera_model = str(
            rospy.get_param("~camera_model", "fisheye")
        ).strip().lower()
        if self.camera_model not in ("fisheye", "pinhole"):
            raise ValueError("camera_model must be fisheye or pinhole")

        self.enable_pose = bool(rospy.get_param("~enable_pose", False))
        self.undistort_before_detection = bool(
            rospy.get_param("~undistort_before_detection", True)
        )
        self.processing_width = int(
            rospy.get_param("~processing_width", 1280)
        )
        self.fisheye_balance = min(
            max(float(rospy.get_param("~fisheye_balance", 0.25)), 0.0),
            1.0,
        )
        self.calibration_width = int(
            rospy.get_param("~calibration_width", 0)
        )
        self.calibration_height = int(
            rospy.get_param("~calibration_height", 0)
        )

        camera_matrix = rospy.get_param("~camera_matrix", [])
        dist_coeffs = rospy.get_param("~dist_coeffs", [])
        self.K_calib = None
        self.D_calib = None

        if len(camera_matrix) == 9:
            self.K_calib = np.asarray(
                camera_matrix, dtype=np.float64
            ).reshape(3, 3)
        if len(dist_coeffs) >= 4:
            self.D_calib = np.asarray(
                dist_coeffs, dtype=np.float64
            ).reshape(-1, 1)

        if self.enable_pose:
            if self.K_calib is None or self.D_calib is None:
                raise ValueError(
                    "enable_pose=true requires camera_matrix and dist_coeffs"
                )
            if self.camera_model == "fisheye" and self.D_calib.size != 4:
                raise ValueError(
                    "fisheye model requires four coefficients [k1,k2,k3,k4]"
                )
            # This node estimates pose on the rectified image.
            self.undistort_before_detection = True

        self.axis_length = float(
            rospy.get_param("~axis_length", self.marker_length * 0.5)
        )

        self.aruco_dict = self._create_dictionary()
        self.detector_parameters = self._create_detector_parameters()
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            self.detector_parameters.cornerRefinementMethod = (
                cv2.aruco.CORNER_REFINE_SUBPIX
            )
        self.detector_parameters.minMarkerPerimeterRate = float(
            rospy.get_param("~min_marker_perimeter_rate", 0.01)
        )
        self.detector_parameters.maxMarkerPerimeterRate = float(
            rospy.get_param("~max_marker_perimeter_rate", 4.0)
        )

        self.map_cache = {}
        self.last_infer_wall_time = 0.0
        self.infer_period = 1.0 / self.infer_rate

        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 26,
        )
        self.pose_pub = rospy.Publisher(
            self.pose_topic, PoseStamped, queue_size=10
        )
        self.annotated_pub = rospy.Publisher(
            self.annotated_topic, Image, queue_size=1
        )
        self.rectified_pub = rospy.Publisher(
            self.rectified_topic, Image, queue_size=1
        )
        self.web_detection_pub = rospy.Publisher(
            self.web_detection_topic, String, queue_size=1
        )
        self.web_pose_pub = rospy.Publisher(
            self.web_pose_topic, String, queue_size=1
        )
        self.target_pub = rospy.Publisher(
            self.target_output_topic,
            TargetDetection,
            queue_size=1,
        )

        rospy.loginfo("Fisheye ArUco node initialized")
        rospy.loginfo("image_topic=%s", self.image_topic)
        rospy.loginfo(
            "camera_model=%s, enable_pose=%s, undistort=%s",
            self.camera_model,
            self.enable_pose,
            self.undistort_before_detection,
        )
        rospy.loginfo(
            "dictionary=%s, marker_length=%.3f m, rate=%.2f Hz",
            self.dictionary_name,
            self.marker_length,
            self.infer_rate,
        )
        rospy.loginfo(
            "target_output_topic=%s",
            self.target_output_topic,
        )
        if not self.enable_pose:
            rospy.logwarn(
                "Pose disabled: detection and annotated image are available, "
                "but /aruco/pose will not be published."
            )

    def _create_dictionary(self):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("cv2.aruco is unavailable")
        if not hasattr(cv2.aruco, self.dictionary_name):
            raise ValueError(
                "Unsupported ArUco dictionary: {}".format(
                    self.dictionary_name
                )
            )
        dictionary_id = getattr(cv2.aruco, self.dictionary_name)
        return cv2.aruco.getPredefinedDictionary(dictionary_id)

    @staticmethod
    def _create_detector_parameters():
        if hasattr(cv2.aruco, "DetectorParameters_create"):
            return cv2.aruco.DetectorParameters_create()
        return cv2.aruco.DetectorParameters()

    @staticmethod
    def _valid_stamp(header):
        if header.stamp == rospy.Time():
            return rospy.Time.now()
        return header.stamp

    def _detect_markers(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(
                self.aruco_dict, self.detector_parameters
            )
            return detector.detectMarkers(gray)
        return cv2.aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.detector_parameters,
        )

    def _resize_for_processing(self, image):
        if self.processing_width <= 0:
            return image
        height, width = image.shape[:2]
        if width == self.processing_width:
            return image
        scale = self.processing_width / float(width)
        new_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(
            image,
            (self.processing_width, new_height),
            interpolation=interpolation,
        )

    def _scaled_K(self, width, height):
        if self.K_calib is None:
            return None
        if self.calibration_width > 0 and self.calibration_height > 0:
            sx = width / float(self.calibration_width)
            sy = height / float(self.calibration_height)
        else:
            sx = 1.0
            sy = 1.0

        K = self.K_calib.copy()
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy
        return K

    def _get_rectification(self, width, height):
        key = (
            self.camera_model,
            width,
            height,
            round(self.fisheye_balance, 4),
        )
        if key in self.map_cache:
            return self.map_cache[key]

        K = self._scaled_K(width, height)
        if K is None or self.D_calib is None:
            raise RuntimeError("camera calibration is unavailable")

        size = (width, height)
        identity = np.eye(3, dtype=np.float64)

        if self.camera_model == "fisheye":
            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K,
                self.D_calib,
                size,
                identity,
                self.fisheye_balance,
                size,
                1.0,
            )
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K,
                self.D_calib,
                identity,
                new_K,
                size,
                cv2.CV_16SC2,
            )
        else:
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                K,
                self.D_calib,
                size,
                self.fisheye_balance,
                size,
            )
            map1, map2 = cv2.initUndistortRectifyMap(
                K,
                self.D_calib,
                None,
                new_K,
                size,
                cv2.CV_16SC2,
            )

        zero_dist = np.zeros((5, 1), dtype=np.float64)
        self.map_cache[key] = (map1, map2, new_K, zero_dist)
        return self.map_cache[key]

    def _prepare_image(self, original):
        image = self._resize_for_processing(original)
        height, width = image.shape[:2]

        if (
            self.undistort_before_detection
            and self.K_calib is not None
            and self.D_calib is not None
        ):
            map1, map2, pose_K, pose_D = self._get_rectification(
                width, height
            )
            rectified = cv2.remap(
                image,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            return rectified, pose_K, pose_D, True

        return image, self._scaled_K(width, height), self.D_calib, False

    def _solve_pose(self, corner, K, D):
        if not self.enable_pose or K is None or D is None:
            return None

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
        image_points = np.asarray(corner, dtype=np.float64).reshape(4, 2)
        flag = getattr(
            cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE
        )
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            K,
            D,
            flags=flag,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        quaternion = rotation_matrix_to_quaternion(rotation_matrix)
        return rvec, tvec, quaternion

    def _publish_target_message(self, stamp, detections):
        """
        Publish ArUco detection state and one ID through the existing
        auv_control/TargetDetection message.

        Field mapping:
          type:
            "aruco_detected"     -> at least one marker detected
            "aruco_not_detected" -> no marker detected

          conf:
            1.0 -> detected
            0.0 -> not detected

          class_name:
            detected     -> selected ArUco ID as a decimal string
            not detected -> "-1"

        Pose is not used by this ID-only output and is filled with zero
        position plus an identity quaternion.

        When multiple markers are visible, the lowest numerical ID is
        published to keep the output deterministic.
        """
        msg = TargetDetection()
        msg.pose.header.stamp = stamp
        msg.pose.header.frame_id = self.camera_frame

        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = 0.0
        msg.pose.pose.orientation.w = 1.0

        if detections:
            marker_id = min(
                int(item["marker_id"])
                for item in detections
            )
            msg.type = "aruco_detected"
            msg.conf = 1.0
            msg.class_name = str(marker_id)
        else:
            msg.type = "aruco_not_detected"
            msg.conf = 0.0
            msg.class_name = "-1"

        self.target_pub.publish(msg)

    def _select_primary(self, detections):
        valid = [item for item in detections if item.get("pose_valid")]
        if not valid:
            return None
        if self.primary_marker_policy == "lowest_id":
            return min(valid, key=lambda item: item["marker_id"])
        return min(valid, key=lambda item: item["distance_m"])

    def _publish_pose(self, stamp, primary):
        if primary is None:
            return
        position = primary["position_m"]
        quaternion = primary["orientation_xyzw"]
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.camera_frame
        msg.pose.position.x = float(position["x"])
        msg.pose.position.y = float(position["y"])
        msg.pose.position.z = float(position["z"])
        msg.pose.orientation = Quaternion(
            float(quaternion["x"]),
            float(quaternion["y"]),
            float(quaternion["z"]),
            float(quaternion["w"]),
        )
        self.pose_pub.publish(msg)

    def _publish_image(self, publisher, image, header, stamp):
        try:
            msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            msg.header = header
            msg.header.stamp = stamp
            msg.header.frame_id = self.camera_frame
            publisher.publish(msg)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0, "Image publication failed: %s", str(exc)
            )

    def _publish_web(self, stamp, image, detections, rectified):
        detection_payload = {
            "stamp": stamp.to_sec(),
            "source": "aruco",
            "node": "fisheye_aruco_node",
            "frame_id": self.camera_frame,
            "camera_model": self.camera_model,
            "rectified": bool(rectified),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "count": len(detections),
            "detections": detections,
        }
        self.web_detection_pub.publish(
            String(data=json.dumps(detection_payload, ensure_ascii=False))
        )

        primary = self._select_primary(detections)
        if primary is None:
            pose_payload = {
                "stamp": stamp.to_sec(),
                "source": "aruco",
                "frame_id": self.camera_frame,
                "camera_model": self.camera_model,
                "valid": False,
                "reason": (
                    "pose_disabled_or_uncalibrated"
                    if not self.enable_pose
                    else "no_valid_marker_pose"
                ),
            }
        else:
            pose_payload = {
                "stamp": stamp.to_sec(),
                "source": "aruco",
                "frame_id": self.camera_frame,
                "camera_model": self.camera_model,
                "marker_id": primary["marker_id"],
                "class_name": primary["class_name"],
                "confidence": 1.0,
                "valid": True,
                "pixel_center": primary["center"],
                "position_m": primary["position_m"],
                "orientation_xyzw": primary["orientation_xyzw"],
                "distance_m": primary["distance_m"],
            }
        self.web_pose_pub.publish(
            String(data=json.dumps(pose_payload, ensure_ascii=False))
        )

    def image_callback(self, image_msg):
        wall_now = time.monotonic()
        if wall_now - self.last_infer_wall_time < self.infer_period:
            return
        self.last_infer_wall_time = wall_now

        try:
            original = self.bridge.imgmsg_to_cv2(
                image_msg, desired_encoding="bgr8"
            )
            image, pose_K, pose_D, rectified = self._prepare_image(original)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0, "Image conversion/rectification failed: %s", str(exc)
            )
            return

        stamp = self._valid_stamp(image_msg.header)
        annotated = image.copy()

        try:
            corners, ids, _ = self._detect_markers(image)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0, "ArUco detection failed: %s", str(exc)
            )
            return

        detections = []
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            ids_flat = ids.reshape(-1)

            for index, corner in enumerate(corners):
                marker_id = int(ids_flat[index])
                points = np.asarray(corner[0], dtype=np.float64).reshape(4, 2)
                center = np.mean(points, axis=0)

                item = {
                    "marker_id": marker_id,
                    "class_id": marker_id,
                    "class_name": "ArUco ID {}".format(marker_id),
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
                        for point in points
                    ],
                    "pose_valid": False,
                    "task": "aruco_pose",
                    "output_type": "pose",
                }

                pose = self._solve_pose(points, pose_K, pose_D)
                if pose is not None:
                    rvec, tvec, quaternion = pose
                    position = np.asarray(tvec, dtype=np.float64).reshape(3)
                    distance = float(np.linalg.norm(position))
                    item.update(
                        {
                            "pose_valid": True,
                            "position_m": {
                                "x": float(position[0]),
                                "y": float(position[1]),
                                "z": float(position[2]),
                            },
                            "orientation_xyzw": {
                                "x": float(quaternion[0]),
                                "y": float(quaternion[1]),
                                "z": float(quaternion[2]),
                                "w": float(quaternion[3]),
                            },
                            "distance_m": distance,
                        }
                    )
                    try:
                        cv2.drawFrameAxes(
                            annotated,
                            pose_K,
                            pose_D,
                            rvec,
                            tvec,
                            self.axis_length,
                            2,
                        )
                    except Exception as exc:
                        rospy.logwarn_throttle(
                            2.0, "Axis drawing failed: %s", str(exc)
                        )
                    label = "ID {} Z={:.2f}m".format(
                        marker_id, position[2]
                    )
                else:
                    label = "ID {} DETECTED".format(marker_id)

                cv2.putText(
                    annotated,
                    label,
                    (
                        int(round(center[0])) + 8,
                        int(round(center[1])) - 8,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                detections.append(item)

        # This topic is independent of camera calibration and pose.
        self._publish_target_message(stamp, detections)

        primary = self._select_primary(detections)
        self._publish_pose(stamp, primary)
        self._publish_web(stamp, image, detections, rectified)
        self._publish_image(
            self.annotated_pub, annotated, image_msg.header, stamp
        )
        if rectified:
            self._publish_image(
                self.rectified_pub, image, image_msg.header, stamp
            )

        if primary is not None:
            p = primary["position_m"]
            rospy.loginfo_throttle(
                1.0,
                "ArUco ID=%d X=%.3f Y=%.3f Z=%.3f m",
                primary["marker_id"],
                p["x"],
                p["y"],
                p["z"],
            )
        else:
            rospy.loginfo_throttle(
                2.0,
                "ArUco detections=%d, pose_enabled=%s",
                len(detections),
                self.enable_pose,
            )

        if self.visualization == 1:
            cv2.imshow("Fisheye ArUco Detection", annotated)
            cv2.waitKey(1)


if __name__ == "__main__":
    try:
        FisheyeArucoNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()