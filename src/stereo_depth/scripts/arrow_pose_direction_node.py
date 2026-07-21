#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

"""
Arrow direction node based directly on YOLO Pose keypoints.

Input:
    /left/image_raw                       sensor_msgs/Image
    /yolo_unified/arrow_keypoints         stereo_depth/ArrowKeypoints

Outputs:
    /arrow/direction                      std_msgs/String (JSON)
    /arrow/angle_deg                      std_msgs/Float32
    /arrow/discrete_direction             std_msgs/String
    /arrow/direction_vector               geometry_msgs/Vector3Stamped
    /arrow/annotated_image                sensor_msgs/Image

Angle convention:
    image right = 0 deg
    image up    = 90 deg
    image left  = 180 deg
    image down  = 270 deg

No thresholding, morphology, contour extraction, or PCA is used.
"""

import json
import math
import threading
import time

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from stereo_depth.msg import ArrowKeypoints


class ArrowPoseDirectionNode:
    def __init__(self):
        rospy.init_node("arrow_pose_direction_node", anonymous=False)

        self.image_topic = rospy.get_param(
            "~image_topic",
            "/left/image_raw",
        )
        self.keypoint_topic = rospy.get_param(
            "~keypoint_topic",
            "/yolo_unified/arrow_keypoints",
        )
        self.annotated_topic = rospy.get_param(
            "~annotated_topic",
            "/arrow/annotated_image",
        )
        self.direction_topic = rospy.get_param(
            "~direction_topic",
            "/arrow/direction",
        )
        self.angle_topic = rospy.get_param(
            "~angle_topic",
            "/arrow/angle_deg",
        )
        self.discrete_topic = rospy.get_param(
            "~discrete_topic",
            "/arrow/discrete_direction",
        )
        self.vector_topic = rospy.get_param(
            "~vector_topic",
            "/arrow/direction_vector",
        )

        self.target_class_name = str(
            rospy.get_param("~target_class_name", "arrow")
        ).strip().lower()

        self.publish_rate = max(
            0.5,
            float(rospy.get_param("~publish_rate", 5.0)),
        )
        self.keypoint_timeout = max(
            0.05,
            float(rospy.get_param("~keypoint_timeout", 1.5)),
        )
        self.max_sync_dt = max(
            0.0,
            float(rospy.get_param("~max_sync_dt", 1.2)),
        )
        self.min_keypoint_confidence = float(
            rospy.get_param("~min_keypoint_confidence", 0.35)
        )
        self.min_direction_length_px = max(
            1.0,
            float(
                rospy.get_param(
                    "~min_direction_length_px",
                    15.0,
                )
            ),
        )
        self.visualization = int(
            rospy.get_param("~visualization", 0)
        )

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_image = None
        self.latest_image_header = None

        self.latest_keypoints = None
        self.latest_keypoint_receive_wall = 0.0

        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 26,
        )
        self.keypoint_sub = rospy.Subscriber(
            self.keypoint_topic,
            ArrowKeypoints,
            self.keypoint_callback,
            queue_size=1,
        )

        self.annotated_pub = rospy.Publisher(
            self.annotated_topic,
            Image,
            queue_size=1,
        )
        self.direction_pub = rospy.Publisher(
            self.direction_topic,
            String,
            queue_size=1,
        )
        self.angle_pub = rospy.Publisher(
            self.angle_topic,
            Float32,
            queue_size=1,
        )
        self.discrete_pub = rospy.Publisher(
            self.discrete_topic,
            String,
            queue_size=1,
        )
        self.vector_pub = rospy.Publisher(
            self.vector_topic,
            Vector3Stamped,
            queue_size=1,
        )

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self.timer_callback,
        )

        rospy.loginfo("Arrow Pose direction node initialized")
        rospy.loginfo("image_topic=%s", self.image_topic)
        rospy.loginfo("keypoint_topic=%s", self.keypoint_topic)
        rospy.loginfo("direction_topic=%s", self.direction_topic)
        rospy.loginfo(
            "publish_rate=%.2f Hz, keypoint_timeout=%.2f s",
            self.publish_rate,
            self.keypoint_timeout,
        )

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "Arrow Pose direction image conversion failed: %s",
                str(exc),
            )
            return

        with self.lock:
            self.latest_image = image
            self.latest_image_header = msg.header

    def keypoint_callback(self, msg):
        with self.lock:
            self.latest_keypoints = msg
            self.latest_keypoint_receive_wall = time.monotonic()

    @staticmethod
    def valid_stamp(stamp):
        if stamp == rospy.Time():
            return rospy.Time.now()
        return stamp

    @staticmethod
    def finite_xy(point):
        return (
            np.isfinite(point.x)
            and np.isfinite(point.y)
        )

    def timer_callback(self, _event):
        now = rospy.Time.now()

        with self.lock:
            image = (
                None
                if self.latest_image is None
                else self.latest_image.copy()
            )
            image_header = self.latest_image_header
            keypoints = self.latest_keypoints
            receive_wall = self.latest_keypoint_receive_wall

        if image is None or image_header is None:
            self.publish_invalid(
                now=now,
                image=None,
                image_header=None,
                reason="no_image",
                keypoints=keypoints,
            )
            return

        annotated = image.copy()

        if keypoints is None:
            self.draw_invalid_overlay(
                annotated,
                "no_arrow_keypoints",
            )
            self.publish_invalid(
                now=now,
                image=annotated,
                image_header=image_header,
                reason="no_arrow_keypoints",
                keypoints=None,
            )
            return

        age = (
            time.monotonic() - receive_wall
            if receive_wall > 0.0
            else float("inf")
        )
        if age > self.keypoint_timeout:
            self.draw_bbox(
                annotated,
                keypoints,
                (0, 0, 255),
            )
            self.draw_invalid_overlay(
                annotated,
                "arrow_keypoint_timeout",
            )
            self.publish_invalid(
                now=now,
                image=annotated,
                image_header=image_header,
                reason="arrow_keypoint_timeout",
                keypoints=keypoints,
                extra={"keypoint_age_sec": age},
            )
            return

        if (
            self.max_sync_dt > 0.0
            and keypoints.header.stamp != rospy.Time()
            and image_header.stamp != rospy.Time()
        ):
            sync_dt = abs(
                (
                    keypoints.header.stamp
                    - image_header.stamp
                ).to_sec()
            )
            if sync_dt > self.max_sync_dt:
                self.draw_bbox(
                    annotated,
                    keypoints,
                    (0, 0, 255),
                )
                self.draw_invalid_overlay(
                    annotated,
                    "keypoint_image_unsynchronized",
                )
                self.publish_invalid(
                    now=now,
                    image=annotated,
                    image_header=image_header,
                    reason="keypoint_image_unsynchronized",
                    keypoints=keypoints,
                    extra={"sync_dt_sec": sync_dt},
                )
                return

        result, reason = self.compute_direction(keypoints)

        if result is None:
            self.draw_bbox(
                annotated,
                keypoints,
                (0, 0, 255),
            )
            self.draw_keypoints(
                annotated,
                keypoints,
                valid=False,
            )
            self.draw_invalid_overlay(
                annotated,
                reason,
            )
            self.publish_invalid(
                now=now,
                image=annotated,
                image_header=image_header,
                reason=reason,
                keypoints=keypoints,
            )
            return

        annotated = self.draw_valid_overlay(
            annotated,
            keypoints,
            result,
        )
        self.publish_valid(
            image=annotated,
            image_header=image_header,
            keypoints=keypoints,
            result=result,
        )

    def compute_direction(self, msg):
        if not msg.valid:
            return None, msg.reason or "invalid_arrow_pose"

        class_name = str(msg.class_name).strip().lower()
        if class_name != self.target_class_name:
            return None, "non_arrow_class"

        if not (
            self.finite_xy(msg.tip)
            and self.finite_xy(msg.tail_left)
            and self.finite_xy(msg.tail_right)
        ):
            return None, "non_finite_keypoint"

        confidences = [
            float(msg.tip_confidence),
            float(msg.tail_left_confidence),
            float(msg.tail_right_confidence),
        ]
        if min(confidences) < self.min_keypoint_confidence:
            return None, "keypoint_confidence_too_low"

        tip = np.array(
            [msg.tip.x, msg.tip.y],
            dtype=np.float64,
        )
        tail_left = np.array(
            [msg.tail_left.x, msg.tail_left.y],
            dtype=np.float64,
        )
        tail_right = np.array(
            [msg.tail_right.x, msg.tail_right.y],
            dtype=np.float64,
        )

        tail = 0.5 * (tail_left + tail_right)
        direction = tip - tail
        length = float(np.linalg.norm(direction))

        if length < self.min_direction_length_px:
            return None, "direction_vector_too_short"

        direction /= length
        dir_x = float(direction[0])
        dir_y = float(direction[1])

        angle_rad = math.atan2(-dir_y, dir_x)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0.0:
            angle_deg += 360.0

        center = 0.5 * (tip + tail)
        confidence = min(
            float(msg.detection_confidence),
            confidences[0],
            confidences[1],
            confidences[2],
        )

        result = {
            "tip": (
                int(round(tip[0])),
                int(round(tip[1])),
            ),
            "tail_left": (
                int(round(tail_left[0])),
                int(round(tail_left[1])),
            ),
            "tail_right": (
                int(round(tail_right[0])),
                int(round(tail_right[1])),
            ),
            "tail": (
                int(round(tail[0])),
                int(round(tail[1])),
            ),
            "center": (
                int(round(center[0])),
                int(round(center[1])),
            ),
            "direction_x": dir_x,
            "direction_y": dir_y,
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "discrete_direction": self.angle_to_direction(
                angle_deg
            ),
            "direction_length_px": length,
            "direction_confidence": confidence,
        }
        return result, ""

    def publish_valid(
        self,
        image,
        image_header,
        keypoints,
        result,
    ):
        stamp = self.valid_stamp(image_header.stamp)
        frame_id = image_header.frame_id or "camera"

        payload = {
            "stamp": stamp.to_sec(),
            "source": "arrow_pose_direction",
            "valid": True,
            "reason": "",
            "class_name": self.target_class_name,
            "confidence": float(
                result["direction_confidence"]
            ),
            "detection_confidence": float(
                keypoints.detection_confidence
            ),
            "bbox": {
                "x1": int(keypoints.x1),
                "y1": int(keypoints.y1),
                "x2": int(keypoints.x2),
                "y2": int(keypoints.y2),
            },
            "tip": {
                "u": int(result["tip"][0]),
                "v": int(result["tip"][1]),
                "confidence": float(
                    keypoints.tip_confidence
                ),
            },
            "tail_left": {
                "u": int(result["tail_left"][0]),
                "v": int(result["tail_left"][1]),
                "confidence": float(
                    keypoints.tail_left_confidence
                ),
            },
            "tail_right": {
                "u": int(result["tail_right"][0]),
                "v": int(result["tail_right"][1]),
                "confidence": float(
                    keypoints.tail_right_confidence
                ),
            },
            "tail": {
                "u": int(result["tail"][0]),
                "v": int(result["tail"][1]),
            },
            "center": {
                "u": int(result["center"][0]),
                "v": int(result["center"][1]),
            },
            "direction_2d": {
                "x": float(result["direction_x"]),
                "y": float(result["direction_y"]),
            },
            "angle_rad": float(result["angle_rad"]),
            "angle_deg": float(result["angle_deg"]),
            "discrete_direction": result[
                "discrete_direction"
            ],
            "direction_length_px": float(
                result["direction_length_px"]
            ),
            "direction_confidence": float(
                result["direction_confidence"]
            ),
        }

        self.direction_pub.publish(
            String(
                data=json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            )
        )
        self.angle_pub.publish(
            Float32(data=float(result["angle_deg"]))
        )
        self.discrete_pub.publish(
            String(data=result["discrete_direction"])
        )

        vector = Vector3Stamped()
        vector.header.stamp = stamp
        vector.header.frame_id = frame_id
        vector.vector.x = float(result["direction_x"])
        vector.vector.y = float(result["direction_y"])
        vector.vector.z = 0.0
        self.vector_pub.publish(vector)

        self.publish_annotated_image(
            image,
            image_header,
            stamp,
        )

        rospy.loginfo_throttle(
            1.0,
            "Arrow Pose direction valid: %s, %.1f deg, conf=%.3f",
            result["discrete_direction"],
            result["angle_deg"],
            result["direction_confidence"],
        )

    def publish_invalid(
        self,
        now,
        image,
        image_header,
        reason,
        keypoints=None,
        extra=None,
    ):
        stamp = (
            self.valid_stamp(image_header.stamp)
            if image_header is not None
            else now
        )
        frame_id = (
            (image_header.frame_id or "camera")
            if image_header is not None
            else "camera"
        )

        payload = {
            "stamp": stamp.to_sec(),
            "source": "arrow_pose_direction",
            "valid": False,
            "reason": reason,
            "class_name": self.target_class_name,
            "confidence": 0.0,
            "detection_confidence": (
                float(keypoints.detection_confidence)
                if keypoints is not None
                else 0.0
            ),
            "bbox": (
                {
                    "x1": int(keypoints.x1),
                    "y1": int(keypoints.y1),
                    "x2": int(keypoints.x2),
                    "y2": int(keypoints.y2),
                }
                if keypoints is not None
                else None
            ),
            "tip": None,
            "tail_left": None,
            "tail_right": None,
            "tail": None,
            "center": None,
            "direction_2d": None,
            "angle_rad": None,
            "angle_deg": None,
            "discrete_direction": "none",
            "direction_length_px": 0.0,
            "direction_confidence": 0.0,
        }
        if extra:
            payload.update(extra)

        self.direction_pub.publish(
            String(
                data=json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            )
        )
        self.angle_pub.publish(
            Float32(data=float("nan"))
        )
        self.discrete_pub.publish(
            String(data="none")
        )

        vector = Vector3Stamped()
        vector.header.stamp = stamp
        vector.header.frame_id = frame_id
        vector.vector.x = 0.0
        vector.vector.y = 0.0
        vector.vector.z = 0.0
        self.vector_pub.publish(vector)

        if image is not None and image_header is not None:
            self.publish_annotated_image(
                image,
                image_header,
                stamp,
            )

        rospy.loginfo_throttle(
            2.0,
            "Arrow Pose direction invalid: %s",
            reason,
        )

    def publish_annotated_image(
        self,
        image,
        image_header,
        stamp,
    ):
        try:
            msg = self.bridge.cv2_to_imgmsg(
                image,
                encoding="bgr8",
            )
            msg.header = image_header
            msg.header.stamp = stamp
            self.annotated_pub.publish(msg)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "Arrow Pose annotated publish failed: %s",
                str(exc),
            )

        if self.visualization == 1:
            cv2.imshow(
                "Arrow Pose Direction",
                image,
            )
            cv2.waitKey(1)

    def draw_valid_overlay(
        self,
        image,
        keypoints,
        result,
    ):
        self.draw_bbox(
            image,
            keypoints,
            (255, 0, 0),
        )

        cv2.circle(
            image,
            result["tip"],
            8,
            (0, 0, 255),
            -1,
        )
        cv2.circle(
            image,
            result["tail_left"],
            7,
            (255, 0, 0),
            -1,
        )
        cv2.circle(
            image,
            result["tail_right"],
            7,
            (255, 255, 0),
            -1,
        )
        cv2.line(
            image,
            result["tail_left"],
            result["tail_right"],
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            image,
            result["tail"],
            7,
            (0, 255, 255),
            -1,
        )
        cv2.arrowedLine(
            image,
            result["tail"],
            result["tip"],
            (0, 255, 0),
            4,
            cv2.LINE_AA,
            tipLength=0.20,
        )

        x = max(10, int(keypoints.x1))
        y = max(30, int(keypoints.y1) - 35)

        cv2.putText(
            image,
            "Arrow: {}".format(
                result["discrete_direction"]
            ),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Angle: {:.1f} deg".format(
                result["angle_deg"]
            ),
            (x, y + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Pose conf: {:.3f}".format(
                result["direction_confidence"]
            ),
            (x, y + 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image

    def draw_keypoints(
        self,
        image,
        msg,
        valid=False,
    ):
        if msg is None:
            return

        color = (
            (0, 255, 0)
            if valid
            else (0, 0, 255)
        )
        for point in (
            msg.tip,
            msg.tail_left,
            msg.tail_right,
        ):
            if self.finite_xy(point):
                cv2.circle(
                    image,
                    (
                        int(round(point.x)),
                        int(round(point.y)),
                    ),
                    6,
                    color,
                    -1,
                )

    @staticmethod
    def draw_bbox(image, msg, color):
        if msg is None:
            return

        height, width = image.shape[:2]
        x1 = max(0, min(width - 1, int(msg.x1)))
        y1 = max(0, min(height - 1, int(msg.y1)))
        x2 = max(0, min(width - 1, int(msg.x2)))
        y2 = max(0, min(height - 1, int(msg.y2)))

        if x2 > x1 and y2 > y1:
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

    @staticmethod
    def draw_invalid_overlay(image, reason):
        cv2.rectangle(
            image,
            (8, 8),
            (610, 52),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            "Arrow Pose: NOT VALID ({})".format(reason),
            (16, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def angle_to_direction(angle_deg):
        if angle_deg >= 337.5 or angle_deg < 22.5:
            return "right"
        if angle_deg < 67.5:
            return "up_right"
        if angle_deg < 112.5:
            return "up"
        if angle_deg < 157.5:
            return "up_left"
        if angle_deg < 202.5:
            return "left"
        if angle_deg < 247.5:
            return "down_left"
        if angle_deg < 292.5:
            return "down"
        return "down_right"


if __name__ == "__main__":
    try:
        ArrowPoseDirectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
