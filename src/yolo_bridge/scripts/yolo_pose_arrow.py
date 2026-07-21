#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

"""
ROS1 Melodic + Ultralytics YOLO Pose arrow detector.

Input:
    /left/image_raw                       sensor_msgs/Image

Outputs:
    /yolo_unified/arrow_keypoints         stereo_depth/ArrowKeypoints
    /yolo_unified/target_bbox             stereo_depth/BoundingBox
    /yolo_unified/annotated_image         sensor_msgs/Image
    /web/detections                       std_msgs/String

Keypoint order:
    0: tip
    1: tail_left
    2: tail_right

The node publishes ArrowKeypoints on every processed image. When no valid
arrow pose is found, valid=false and reason explains the failure.
"""

import argparse
import json
import math
import os
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from ultralytics import YOLO

from stereo_depth.msg import ArrowKeypoints, BoundingBox


KEYPOINT_NAMES = ("tip", "tail_left", "tail_right")
EXPECTED_KEYPOINT_COUNT = 3


def normalize_optional_model_path(model_path):
    if model_path is None:
        return ""

    model_path = str(model_path).strip()

    if (
        len(model_path) >= 2
        and model_path[0] == model_path[-1]
        and model_path[0] in ("'", '"')
    ):
        model_path = model_path[1:-1].strip()

    if model_path.lower() in ("", "none", "null", "~"):
        return ""

    return os.path.expanduser(model_path)


class ArrowPoseYOLODetector:
    def __init__(self, args):
        rospy.init_node("yolo_arrow_pose_detector", anonymous=False)

        self.model_path = normalize_optional_model_path(args.model_path)
        if not self.model_path:
            raise ValueError("model_path must be provided for arrow pose")
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "pose model does not exist: {}".format(self.model_path)
            )

        self.top_k = max(1, int(args.top_k))
        self.visualization = int(args.visualization)
        self.conf_thre = float(args.conf_thre)
        self.keypoint_conf_thre = float(args.keypoint_conf_thre)
        self.infer_rate = max(0.2, float(args.infer_rate))
        self.imgsz = max(32, int(args.imgsz))
        self.device = str(args.device).strip()
        self.rate = rospy.Rate(self.infer_rate)

        self.input_topic = args.input_topic
        self.annotated_topic = args.annotated_topic
        self.web_topic = args.web_topic
        self.keypoint_topic = args.keypoint_topic
        self.bbox_topic = args.bbox_topic

        rospy.loginfo("Loading arrow pose model: %s", self.model_path)
        self.model = YOLO(self.model_path)

        model_task = str(getattr(self.model, "task", "")).lower()
        if model_task and model_task != "pose":
            rospy.logwarn(
                "Loaded model task is '%s', expected 'pose'. "
                "Check that the weight file is a YOLO Pose model.",
                model_task,
            )

        self.bridge = CvBridge()
        self.image_lock = threading.Lock()

        self.latest_image = None
        self.latest_header = Header()
        self.image_version = 0
        self.processed_version = -1

        self.image_sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 26,
        )

        self.keypoint_pub = rospy.Publisher(
            self.keypoint_topic,
            ArrowKeypoints,
            queue_size=1,
        )
        self.bbox_pub = rospy.Publisher(
            self.bbox_topic,
            BoundingBox,
            queue_size=1,
        )
        self.annotated_pub = rospy.Publisher(
            self.annotated_topic,
            Image,
            queue_size=1,
        )
        self.web_pub = rospy.Publisher(
            self.web_topic,
            String,
            queue_size=1,
        )

        rospy.loginfo("Arrow Pose YOLO node initialized")
        rospy.loginfo("input_topic=%s", self.input_topic)
        rospy.loginfo("keypoint_topic=%s", self.keypoint_topic)
        rospy.loginfo("bbox_topic=%s", self.bbox_topic)
        rospy.loginfo(
            "conf=%.2f, keypoint_conf=%.2f, rate=%.2f Hz, imgsz=%d",
            self.conf_thre,
            self.keypoint_conf_thre,
            self.infer_rate,
            self.imgsz,
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
                "Arrow Pose cv_bridge error: %s",
                str(exc),
            )
            return

        with self.image_lock:
            self.latest_image = image
            self.latest_header = msg.header
            self.image_version += 1

    @staticmethod
    def valid_stamp(stamp):
        if stamp == rospy.Time():
            return rospy.Time.now()
        return stamp

    @staticmethod
    def class_name_from_result(result, class_id):
        names = result.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def finite_point(point):
        return (
            len(point) >= 2
            and np.isfinite(point[0])
            and np.isfinite(point[1])
        )

    def build_detections(self, result, image_shape):
        if result.boxes is None or len(result.boxes) == 0:
            return []

        if result.keypoints is None or result.keypoints.xy is None:
            rospy.logwarn_throttle(
                2.0,
                "Pose model returned boxes but no keypoints",
            )
            return []

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        box_confs = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        keypoints_xy = result.keypoints.xy.detach().cpu().numpy()

        if result.keypoints.conf is None:
            keypoints_conf = np.ones(
                keypoints_xy.shape[:2],
                dtype=np.float32,
            )
        else:
            keypoints_conf = (
                result.keypoints.conf.detach().cpu().numpy()
            )

        count = min(
            len(boxes),
            len(box_confs),
            len(class_ids),
            len(keypoints_xy),
            len(keypoints_conf),
        )

        if count <= 0:
            return []

        height, width = image_shape[:2]
        order = np.argsort(box_confs[:count])[::-1]
        detections = []

        for raw_index in order:
            if len(detections) >= self.top_k:
                break

            box_conf = float(box_confs[raw_index])
            if box_conf < self.conf_thre:
                continue

            box = boxes[raw_index]
            class_id = int(class_ids[raw_index])
            class_name = self.class_name_from_result(
                result,
                class_id,
            )

            x1, y1, x2, y2 = [
                int(round(float(value)))
                for value in box
            ]
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))

            xy = np.asarray(
                keypoints_xy[raw_index],
                dtype=np.float32,
            )
            kp_conf = np.asarray(
                keypoints_conf[raw_index],
                dtype=np.float32,
            ).reshape(-1)

            item = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(box_conf, 4),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "task": "pose_arrow",
                "output_type": "three_keypoints",
                "valid": False,
                "reason": "",
                "keypoints": [],
            }

            if xy.ndim != 2 or xy.shape[0] < EXPECTED_KEYPOINT_COUNT:
                item["reason"] = "keypoint_count_less_than_3"
                detections.append(item)
                continue

            point_validity = []
            for keypoint_index, keypoint_name in enumerate(KEYPOINT_NAMES):
                point = xy[keypoint_index]
                confidence = (
                    float(kp_conf[keypoint_index])
                    if keypoint_index < len(kp_conf)
                    else 0.0
                )

                valid = (
                    self.finite_point(point)
                    and confidence >= self.keypoint_conf_thre
                    and 0.0 <= float(point[0]) < width
                    and 0.0 <= float(point[1]) < height
                )

                point_validity.append(valid)
                item["keypoints"].append(
                    {
                        "index": keypoint_index,
                        "name": keypoint_name,
                        "u": round(float(point[0]), 2),
                        "v": round(float(point[1]), 2),
                        "confidence": round(confidence, 4),
                        "valid": bool(valid),
                    }
                )

            if not all(point_validity):
                item["reason"] = "keypoint_confidence_too_low"
                detections.append(item)
                continue

            tip = xy[0]
            tail_left = xy[1]
            tail_right = xy[2]
            tail_center = 0.5 * (tail_left + tail_right)

            dx = float(tip[0] - tail_center[0])
            dy = float(tip[1] - tail_center[1])
            vector_length = math.hypot(dx, dy)

            if vector_length < 1.0:
                item["reason"] = "keypoint_direction_too_short"
                detections.append(item)
                continue

            angle_deg = math.degrees(math.atan2(-dy, dx))
            if angle_deg < 0.0:
                angle_deg += 360.0

            item["valid"] = True
            item["reason"] = ""
            item["tail_center"] = {
                "u": round(float(tail_center[0]), 2),
                "v": round(float(tail_center[1]), 2),
            }
            item["angle_deg"] = round(angle_deg, 3)
            item["direction_length_px"] = round(vector_length, 3)
            detections.append(item)

        return detections

    def make_keypoint_message(self, detection, stamp, frame_id):
        msg = ArrowKeypoints()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.valid = False
        msg.reason = "no_arrow_detection"
        msg.class_name = "arrow"
        msg.detection_confidence = 0.0

        msg.x1 = 0
        msg.y1 = 0
        msg.x2 = 0
        msg.y2 = 0

        msg.tip = Point32(0.0, 0.0, 0.0)
        msg.tail_left = Point32(0.0, 0.0, 0.0)
        msg.tail_right = Point32(0.0, 0.0, 0.0)

        msg.tip_confidence = 0.0
        msg.tail_left_confidence = 0.0
        msg.tail_right_confidence = 0.0

        if detection is None:
            return msg

        msg.valid = bool(detection.get("valid", False))
        msg.reason = str(detection.get("reason", ""))
        msg.class_name = str(detection.get("class_name", "arrow"))
        msg.detection_confidence = float(
            detection.get("confidence", 0.0)
        )

        bbox = detection.get("bbox", {})
        msg.x1 = int(bbox.get("x1", 0))
        msg.y1 = int(bbox.get("y1", 0))
        msg.x2 = int(bbox.get("x2", 0))
        msg.y2 = int(bbox.get("y2", 0))

        keypoints = detection.get("keypoints", [])
        if len(keypoints) >= 3:
            tip = keypoints[0]
            tail_left = keypoints[1]
            tail_right = keypoints[2]

            msg.tip = Point32(
                float(tip["u"]),
                float(tip["v"]),
                0.0,
            )
            msg.tail_left = Point32(
                float(tail_left["u"]),
                float(tail_left["v"]),
                0.0,
            )
            msg.tail_right = Point32(
                float(tail_right["u"]),
                float(tail_right["v"]),
                0.0,
            )

            msg.tip_confidence = float(tip["confidence"])
            msg.tail_left_confidence = float(
                tail_left["confidence"]
            )
            msg.tail_right_confidence = float(
                tail_right["confidence"]
            )

        return msg

    def publish_bbox(self, detection, stamp):
        if detection is None:
            return

        bbox = detection.get("bbox")
        if not bbox:
            return

        msg = BoundingBox()
        msg.header.stamp = stamp
        msg.header.frame_id = str(
            detection.get("class_name", "arrow")
        )
        msg.x1 = int(bbox["x1"])
        msg.y1 = int(bbox["y1"])
        msg.x2 = int(bbox["x2"])
        msg.y2 = int(bbox["y2"])
        msg.conf = float(detection.get("confidence", 0.0))
        self.bbox_pub.publish(msg)

    @staticmethod
    def choose_best_detection(detections):
        if not detections:
            return None

        valid = [
            item
            for item in detections
            if item.get("valid", False)
        ]
        if valid:
            return max(
                valid,
                key=lambda item: float(
                    item.get("confidence", 0.0)
                ),
            )

        return max(
            detections,
            key=lambda item: float(
                item.get("confidence", 0.0)
            ),
        )

    def draw_semantic_keypoints(self, image, detections):
        output = image

        for detection in detections:
            bbox = detection.get("bbox", {})
            color = (
                (0, 255, 0)
                if detection.get("valid", False)
                else (0, 0, 255)
            )

            x1 = int(bbox.get("x1", 0))
            y1 = int(bbox.get("y1", 0))
            x2 = int(bbox.get("x2", 0))
            y2 = int(bbox.get("y2", 0))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2,
                )

            points = detection.get("keypoints", [])
            point_map = {
                item["name"]: item
                for item in points
            }

            point_colors = {
                "tip": (0, 0, 255),
                "tail_left": (255, 0, 0),
                "tail_right": (255, 255, 0),
            }

            for name in KEYPOINT_NAMES:
                item = point_map.get(name)
                if item is None:
                    continue

                center = (
                    int(round(float(item["u"]))),
                    int(round(float(item["v"]))),
                )
                cv2.circle(
                    output,
                    center,
                    7,
                    point_colors[name],
                    -1,
                )
                cv2.putText(
                    output,
                    "{} {:.2f}".format(
                        name,
                        float(item["confidence"]),
                    ),
                    (center[0] + 7, center[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    point_colors[name],
                    1,
                    cv2.LINE_AA,
                )

            if detection.get("valid", False) and len(points) >= 3:
                tip = point_map["tip"]
                tail_left = point_map["tail_left"]
                tail_right = point_map["tail_right"]

                tip_xy = (
                    int(round(float(tip["u"]))),
                    int(round(float(tip["v"]))),
                )
                tail_center = (
                    int(round(
                        0.5
                        * (
                            float(tail_left["u"])
                            + float(tail_right["u"])
                        )
                    )),
                    int(round(
                        0.5
                        * (
                            float(tail_left["v"])
                            + float(tail_right["v"])
                        )
                    )),
                )

                cv2.circle(
                    output,
                    tail_center,
                    6,
                    (0, 255, 255),
                    -1,
                )
                cv2.arrowedLine(
                    output,
                    tail_center,
                    tip_xy,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                    tipLength=0.20,
                )
                cv2.putText(
                    output,
                    "Angle {:.1f} deg".format(
                        float(detection["angle_deg"])
                    ),
                    (max(5, x1), max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                reason = str(
                    detection.get("reason", "invalid")
                )
                cv2.putText(
                    output,
                    reason,
                    (max(5, x1), max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        return output

    def publish_annotated(self, image, header, stamp):
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
                "Failed to publish pose annotated image: %s",
                str(exc),
            )

    def run(self):
        while not rospy.is_shutdown():
            with self.image_lock:
                if (
                    self.latest_image is None
                    or self.image_version == self.processed_version
                ):
                    image = None
                    header = None
                    version = self.processed_version
                else:
                    image = self.latest_image.copy()
                    header = self.latest_header
                    version = self.image_version

            if image is None:
                self.rate.sleep()
                continue

            try:
                results = self.model(
                    image,
                    conf=self.conf_thre,
                    max_det=self.top_k,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            except Exception as exc:
                rospy.logerr_throttle(
                    2.0,
                    "Arrow Pose inference failed: %s",
                    str(exc),
                )
                self.rate.sleep()
                continue

            self.processed_version = version
            stamp = self.valid_stamp(header.stamp)
            frame_id = header.frame_id or "camera"

            result = results[0] if results else None
            detections = (
                self.build_detections(result, image.shape)
                if result is not None
                else []
            )
            best = self.choose_best_detection(detections)

            keypoint_msg = self.make_keypoint_message(
                best,
                stamp,
                frame_id,
            )
            self.keypoint_pub.publish(keypoint_msg)
            self.publish_bbox(best, stamp)

            payload = {
                "stamp": stamp.to_sec(),
                "source": "ultralytics_pose",
                "node": "yolo_arrow_pose_detector",
                "task_mode": "pose_arrow",
                "frame_id": frame_id,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "count": len(detections),
                "detections": detections,
            }
            self.web_pub.publish(
                String(
                    data=json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                )
            )

            try:
                annotated = (
                    result.plot()
                    if result is not None
                    else image.copy()
                )
                annotated = self.draw_semantic_keypoints(
                    annotated,
                    detections,
                )
                self.publish_annotated(
                    annotated,
                    header,
                    stamp,
                )

                if self.visualization == 1:
                    cv2.imshow(
                        "YOLO Arrow Pose",
                        annotated,
                    )
                    cv2.waitKey(1)

            except Exception as exc:
                rospy.logerr_throttle(
                    2.0,
                    "Arrow Pose annotated image failed: %s",
                    str(exc),
                )

            if best is None:
                rospy.loginfo_throttle(
                    2.0,
                    "Arrow Pose: no detection",
                )
            elif best.get("valid", False):
                rospy.loginfo_throttle(
                    1.0,
                    "Arrow Pose valid: conf=%.3f angle=%.1f deg",
                    float(best["confidence"]),
                    float(best["angle_deg"]),
                )
            else:
                rospy.loginfo_throttle(
                    2.0,
                    "Arrow Pose invalid: %s",
                    str(best.get("reason", "unknown")),
                )

            self.rate.sleep()

        cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Ultralytics Arrow Pose ROS node"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--visualization", type=int, default=0)
    parser.add_argument("--conf_thre", type=float, default=0.20)
    parser.add_argument(
        "--keypoint_conf_thre",
        type=float,
        default=0.35,
    )
    parser.add_argument("--infer_rate", type=float, default=5.0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")

    parser.add_argument(
        "--input_topic",
        default="/left/image_raw",
    )
    parser.add_argument(
        "--annotated_topic",
        default="/yolo_unified/annotated_image",
    )
    parser.add_argument(
        "--web_topic",
        default="/web/detections",
    )
    parser.add_argument(
        "--keypoint_topic",
        default="/yolo_unified/arrow_keypoints",
    )
    parser.add_argument(
        "--bbox_topic",
        default="/yolo_unified/target_bbox",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(rospy.myargv()[1:])

    try:
        ArrowPoseYOLODetector(args).run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("Arrow Pose YOLO node failed: %s", str(exc))
        raise
