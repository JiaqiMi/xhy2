#!/home/nvidia/venvs/xhy_ros2/bin/python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from ultralytics import YOLO

from stereo_depth.msg import BoundingBox, LineBox


def normalize_optional_model_path(model_path):
    """
    清理来自 ROS 参数服务器或 Shell 的可选模型路径。

    ROS 空字符串经过 rosparam get 后可能表现为：
        ''
        ""
        null
        None
        ~

    这些情况都视为“未显式指定模型路径”。
    """
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


def resolve_model_path(task_mode, detect_mode, model_path):
    """
    模型路径解析规则：

    1. model_path 非空时，优先使用显式路径；
    2. model_path 为空时，根据 task_mode + detect_mode 选择预设模型。
    """
    task_mode = str(task_mode).strip().lower()
    detect_mode = int(detect_mode)
    model_path = normalize_optional_model_path(model_path)

    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                "explicit model file does not exist: {}".format(model_path)
            )

        rospy.loginfo("Use explicit model: %s", model_path)
        return model_path

    model_root = os.path.join(
        os.path.expanduser("~"),
        "catkin_ws",
        "models",
    )

    detect_models = {
        1: os.path.join(model_root, "shapes0709.pt"),
        2: os.path.join(model_root, "rectangle0710.pt"),
        3: os.path.join(model_root, "line0709.pt"),
        4: os.path.join(model_root, "arrow0709.pt"),
    }

    segment_models = {
        1: os.path.join(model_root, "shapes_model0719.pt"),
        2: os.path.join(model_root, "holes_model0719.pt"),
        3: os.path.join(model_root, "balls_model0725.pt"),
        4: os.path.join(model_root, "line0709.pt"),
    }

    if task_mode == "detect":
        model_map = detect_models
    elif task_mode == "segment3":
        model_map = segment_models
    else:
        raise ValueError(
            "invalid task_mode: {}. Expected 'detect' or 'segment3'.".format(
                task_mode
            )
        )

    if detect_mode not in model_map:
        raise ValueError(
            "no preset model for task_mode={}, detect_mode={}".format(
                task_mode,
                detect_mode,
            )
        )

    resolved_path = model_map[detect_mode]

    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(
            "preset model file does not exist: {}".format(resolved_path)
        )

    rospy.loginfo(
        "Use preset model: task_mode=%s, detect_mode=%d, path=%s",
        task_mode,
        detect_mode,
        resolved_path,
    )

    return resolved_path


def parse_csv_set(value):
    """Parse comma separated class names into a normalized set."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = str(value).split(",")
    return {
        str(item).strip().lower()
        for item in items
        if str(item).strip()
    }


def parse_class_thresholds(value):
    """
    Parse class thresholds from a compact string, for example:
        red:0.20,green:0.10,yello:0.20

    Unknown/missing classes fall back to --conf_thre.
    """
    thresholds = {}
    if value is None:
        return thresholds

    if isinstance(value, dict):
        raw_items = value.items()
    else:
        raw_items = []
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if ":" not in token:
                rospy.logwarn(
                    "Ignore invalid class threshold token: %s",
                    token,
                )
                continue
            name, threshold = token.split(":", 1)
            raw_items.append((name, threshold))

    for name, threshold in raw_items:
        class_name = str(name).strip().lower()
        if not class_name:
            continue
        try:
            value_f = float(threshold)
        except (TypeError, ValueError):
            rospy.logwarn(
                "Ignore invalid threshold for class %s: %s",
                class_name,
                str(threshold),
            )
            continue
        thresholds[class_name] = min(max(value_f, 0.0), 1.0)

    return thresholds


def parse_class_priority(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = str(value).split(",")
    return [
        str(item).strip().lower()
        for item in items
        if str(item).strip()
    ]


class UnifiedYOLODetector:
    def __init__(self, args):
        rospy.init_node("yolo_unified_detector", anonymous=False)

        self.task_mode = str(args.task_mode).strip().lower()
        self.detect_mode = int(args.detect_mode)

        # top_k is now only the maximum number of filtered detections shown
        # in Web JSON. It no longer limits the candidates used for per-class
        # target selection.
        self.top_k = max(1, int(args.top_k))
        self.max_det = max(self.top_k, int(args.max_det))

        self.visualization = int(args.visualization)

        # Two-stage confidence filtering:
        # 1) candidate_conf_thre controls Ultralytics pre-filtering;
        # 2) conf_thre / class_conf_thresholds perform the final class-aware
        #    filtering before publishing targets.
        self.conf_thre = min(max(float(args.conf_thre), 0.0), 1.0)
        self.candidate_conf_thre = min(
            max(float(args.candidate_conf_thre), 0.001),
            1.0,
        )
        self.class_conf_thresholds = parse_class_thresholds(
            args.class_conf_thresholds
        )

        # Never let Ultralytics remove a class that has a lower configured
        # final threshold.
        threshold_candidates = [
            self.candidate_conf_thre,
            self.conf_thre,
        ]
        threshold_candidates.extend(self.class_conf_thresholds.values())
        self.inference_conf_thre = max(
            0.001,
            min(threshold_candidates),
        )

        self.target_selection_mode = str(
            args.target_selection_mode
        ).strip().lower()
        if self.target_selection_mode not in (
            "global_best",
            "per_class_best",
            "all",
        ):
            raise ValueError(
                "target_selection_mode must be global_best, "
                "per_class_best or all"
            )

        self.target_pub_queue_size = max(
            1,
            int(args.target_pub_queue_size),
        )

        self.class_whitelist = parse_csv_set(args.class_whitelist)
        self.class_priority = parse_class_priority(args.class_priority)
        self.class_priority_index = {
            name: index
            for index, name in enumerate(self.class_priority)
        }

        self.detc_type = str(args.detc_type).strip().lower()
        self.output_type = str(args.output_type).strip().lower()
        self.rate = rospy.Rate(max(0.5, float(args.infer_rate)))

        if self.task_mode not in ("detect", "segment3"):
            raise ValueError("task_mode must be detect or segment3")

        if self.task_mode == "detect" and self.detc_type not in (
            "center",
            "bbox",
        ):
            raise ValueError(
                "detect task requires detc_type=center or bbox"
            )

        if (
            self.task_mode == "segment3"
            and self.output_type != "quartiles"
        ):
            raise ValueError(
                "segment3 currently supports output_type=quartiles only"
            )

        self.input_topic = args.input_topic
        self.annotated_topic = args.annotated_topic
        self.web_topic = args.web_topic
        self.center_topic = args.center_topic
        self.bbox_topic = args.bbox_topic
        self.line_topic = args.line_topic

        raw_model_path = getattr(args, "model_path", "")

        self.model_path = resolve_model_path(
            task_mode=self.task_mode,
            detect_mode=self.detect_mode,
            model_path=raw_model_path,
        )

        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                "resolved model file does not exist: {}".format(
                    self.model_path
                )
            )

        rospy.loginfo("Final model path: %s", self.model_path)

        self.model = YOLO(self.model_path)

        self.bridge = CvBridge()
        self.image_lock = threading.Lock()

        self.left_img = None
        self.left_header = Header()
        self.image_version = 0
        self.processed_version = -1

        self.image_sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

        if self.task_mode == "detect":
            if self.detc_type == "center":
                self.target_pub = rospy.Publisher(
                    self.center_topic,
                    PointStamped,
                    queue_size=self.target_pub_queue_size,
                )
            else:
                self.target_pub = rospy.Publisher(
                    self.bbox_topic,
                    BoundingBox,
                    queue_size=self.target_pub_queue_size,
                )
        else:
            self.target_pub = rospy.Publisher(
                self.line_topic,
                LineBox,
                queue_size=self.target_pub_queue_size,
            )

        self.annotated_pub = rospy.Publisher(
            self.annotated_topic,
            Image,
            queue_size=1,
        )

        self.web_detection_pub = rospy.Publisher(
            self.web_topic,
            String,
            queue_size=1,
        )

        rospy.loginfo("YOLO node initialized")
        rospy.loginfo(
            "task_mode=%s, detc_type=%s",
            self.task_mode,
            self.detc_type,
        )
        rospy.loginfo("model=%s", self.model_path)
        rospy.loginfo("input=%s", self.input_topic)
        rospy.loginfo("annotated=%s", self.annotated_topic)
        rospy.loginfo("web=%s", self.web_topic)
        rospy.loginfo(
            "selection=%s, inference_conf=%.3f, default_class_conf=%.3f, "
            "max_det=%d, web_top_k=%d",
            self.target_selection_mode,
            self.inference_conf_thre,
            self.conf_thre,
            self.max_det,
            self.top_k,
        )
        rospy.loginfo(
            "class_conf_thresholds=%s, class_whitelist=%s, class_priority=%s",
            str(self.class_conf_thresholds),
            str(sorted(self.class_whitelist)),
            str(self.class_priority),
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
                "cv_bridge error: %s",
                str(exc),
            )
            return

        with self.image_lock:
            self.left_img = image
            self.left_header = msg.header
            self.image_version += 1

    @staticmethod
    def valid_stamp(header):
        if header.stamp == rospy.Time():
            return rospy.Time.now()
        return header.stamp

    @staticmethod
    def get_skeleton(binary_img):
        binary_img = binary_img.copy().astype(np.uint8)
        size = binary_img.size
        skeleton = np.zeros(binary_img.shape, np.uint8)
        element = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (3, 3),
        )

        while True:
            eroded = cv2.erode(binary_img, element)
            opened = cv2.dilate(eroded, element)
            residue = cv2.subtract(binary_img, opened)
            skeleton = cv2.bitwise_or(skeleton, residue)
            binary_img = eroded

            if size - cv2.countNonZero(binary_img) == size:
                break

        return skeleton

    @staticmethod
    def largest_component(binary_img):
        num_labels, labels = cv2.connectedComponents(
            (binary_img > 0).astype(np.uint8),
            connectivity=8,
        )

        if num_labels <= 1:
            return np.zeros_like(binary_img, dtype=np.uint8)

        counts = [
            int(np.count_nonzero(labels == label))
            for label in range(1, num_labels)
        ]

        best_label = int(np.argmax(counts)) + 1

        return (labels == best_label).astype(np.uint8) * 255

    @staticmethod
    def select_three_quartile_points(skeleton):
        points_yx = np.column_stack(np.where(skeleton > 0))

        if len(points_yx) < 3:
            return None

        order = np.lexsort(
            (
                points_yx[:, 1],
                points_yx[:, 0],
            )
        )
        points_yx = points_yx[order]

        count = len(points_yx)
        indices = [
            count // 4,
            count // 2,
            (3 * count) // 4,
        ]

        selected = []

        for index in indices:
            y, x = points_yx[min(count - 1, index)]
            selected.append((int(x), int(y)))

        return selected

    @staticmethod
    def build_mask(result, index, image_shape):
        if result.masks is None:
            return None

        height, width = image_shape[:2]

        if index < len(result.masks.xy):
            polygon = result.masks.xy[index]

            if polygon is not None and len(polygon) >= 3:
                mask = np.zeros(
                    (height, width),
                    dtype=np.uint8,
                )
                contour = np.round(polygon).astype(np.int32)
                cv2.fillPoly(mask, [contour], 255)
                return mask

        data = result.masks.data[index].cpu().numpy()
        data = (data > 0.5).astype(np.uint8) * 255

        if data.shape[:2] != (height, width):
            data = cv2.resize(
                data,
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )

        return data

    @staticmethod
    def normalize_class_name(class_name):
        return str(class_name).strip().lower()

    def class_allowed(self, class_name):
        if not self.class_whitelist:
            return True
        return self.normalize_class_name(class_name) in self.class_whitelist

    def class_threshold(self, class_name):
        key = self.normalize_class_name(class_name)
        return float(
            self.class_conf_thresholds.get(
                key,
                self.conf_thre,
            )
        )

    def build_detections(self, result, image):
        """
        Build all class-filtered detections returned by Ultralytics.

        Important: there is intentionally NO global top_k cutoff here. A
        high-confidence class therefore cannot consume all available slots
        and suppress a lower-confidence class before per-class selection.
        """
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        order = confs.argsort()[::-1]
        detections = []

        for index in order:
            confidence = float(confs[index])
            class_id = int(classes[index])
            class_name = str(result.names[class_id])
            class_key = self.normalize_class_name(class_name)

            if not self.class_allowed(class_key):
                continue

            threshold = self.class_threshold(class_key)
            if confidence < threshold:
                continue

            box = boxes[index]
            x1, y1, x2, y2 = [
                int(round(value))
                for value in box
            ]

            center_u = int(round((x1 + x2) / 2.0))
            center_v = int(round((y1 + y2) / 2.0))

            item = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "class_threshold": round(threshold, 4),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "center": {
                    "u": center_u,
                    "v": center_v,
                },
            }

            if (
                result.masks is not None
                and index < len(result.masks.xy)
            ):
                polygon = result.masks.xy[index]
                item["polygon"] = [
                    [
                        round(float(point[0]), 2),
                        round(float(point[1]), 2),
                    ]
                    for point in polygon
                ]

            if self.task_mode == "detect":
                item["task"] = "detect"
                item["output_type"] = self.detc_type
                detections.append(item)
                continue

            mask = self.build_mask(
                result,
                index,
                image.shape,
            )
            if mask is None:
                continue

            skeleton = self.largest_component(
                self.get_skeleton(mask)
            )
            keypoints = self.select_three_quartile_points(
                skeleton
            )
            if keypoints is None:
                rospy.logwarn(
                    "not enough skeleton points for %s",
                    class_name,
                )
                continue

            item["task"] = "segment3"
            item["output_type"] = "quartiles"
            item["keypoints"] = [
                {
                    "x": point[0],
                    "y": point[1],
                }
                for point in keypoints
            ]
            detections.append(item)

        return detections

    def target_sort_key(self, detection):
        class_key = self.normalize_class_name(
            detection["class_name"]
        )
        priority = self.class_priority_index.get(
            class_key,
            len(self.class_priority_index),
        )
        return (
            priority,
            class_key,
            -float(detection["confidence"]),
        )

    def select_targets_for_publish(self, detections):
        """
        Select detections sent to the depth node.

        per_class_best:
            retain exactly one highest-confidence target per class;
        global_best:
            legacy behavior, only one global best target;
        all:
            publish every filtered detection.
        """
        if not detections:
            return []

        # Preserve legacy segment3 behavior unless explicitly changed.
        if self.task_mode == "segment3":
            return [detections[0]]

        if self.target_selection_mode == "global_best":
            return [detections[0]]

        if self.target_selection_mode == "all":
            return sorted(
                detections,
                key=self.target_sort_key,
            )

        best_by_class = {}
        for detection in detections:
            class_key = self.normalize_class_name(
                detection["class_name"]
            )
            previous = best_by_class.get(class_key)
            if (
                previous is None
                or float(detection["confidence"])
                > float(previous["confidence"])
            ):
                best_by_class[class_key] = detection

        return sorted(
            best_by_class.values(),
            key=self.target_sort_key,
        )

    def publish_target(self, detection, stamp):
        """Publish one selected target using the existing ROS topic/type."""
        if detection is None:
            return

        if self.task_mode == "segment3":
            keypoints = [
                (
                    point["x"],
                    point["y"],
                )
                for point in detection["keypoints"]
            ]

            msg = LineBox()
            msg.header.stamp = stamp
            msg.header.frame_id = detection["class_name"]
            msg.x1, msg.y1 = keypoints[0]
            msg.x2, msg.y2 = keypoints[1]
            msg.x3, msg.y3 = keypoints[2]
            msg.conf = float(detection["confidence"])
            self.target_pub.publish(msg)
            return

        if self.detc_type == "center":
            msg = PointStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = detection["class_name"]
            msg.point.x = float(detection["center"]["u"])
            msg.point.y = float(detection["center"]["v"])
            msg.point.z = float(detection["confidence"])
            self.target_pub.publish(msg)
        else:
            msg = BoundingBox()
            msg.header.stamp = stamp
            msg.header.frame_id = detection["class_name"]
            msg.x1 = int(detection["bbox"]["x1"])
            msg.y1 = int(detection["bbox"]["y1"])
            msg.x2 = int(detection["bbox"]["x2"])
            msg.y2 = int(detection["bbox"]["y2"])
            msg.conf = float(detection["confidence"])
            self.target_pub.publish(msg)

    def build_annotated_image(self, image, result, detections, selected):
        """Draw only detections that passed the final class-aware filters."""
        if self.task_mode != "detect":
            annotated = result.plot()
            for item in detections[: self.top_k]:
                for index, point in enumerate(item.get("keypoints", [])):
                    center = (
                        int(point["x"]),
                        int(point["y"]),
                    )
                    cv2.circle(
                        annotated,
                        center,
                        6,
                        (0, 255, 255),
                        -1,
                    )
                    cv2.putText(
                        annotated,
                        "P{}".format(index + 1),
                        (center[0] + 7, center[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
            return annotated

        annotated = image.copy()
        selected_ids = {id(item) for item in selected}

        for detection in detections[: self.top_k]:
            bbox = detection["bbox"]
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])
            is_selected = id(detection) in selected_ids
            thickness = 3 if is_selected else 1
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0) if is_selected else (0, 200, 255),
                thickness,
            )
            label = "{} {:.2f}{}".format(
                detection["class_name"],
                float(detection["confidence"]),
                " *" if is_selected else "",
            )
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0) if is_selected else (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def run(self):
        while not rospy.is_shutdown():
            with self.image_lock:
                if (
                    self.left_img is None
                    or self.image_version == self.processed_version
                ):
                    image = None
                    header = None
                    version = self.processed_version
                else:
                    image = self.left_img.copy()
                    header = self.left_header
                    version = self.image_version

            if image is None:
                self.rate.sleep()
                continue

            try:
                results = self.model(
                    image,
                    conf=self.inference_conf_thre,
                    max_det=self.max_det,
                    verbose=False,
                )
            except Exception as exc:
                rospy.logerr_throttle(
                    2.0,
                    "YOLO inference failed: %s",
                    str(exc),
                )
                self.rate.sleep()
                continue

            self.processed_version = version

            if not results:
                self.rate.sleep()
                continue

            result = results[0]
            stamp = self.valid_stamp(header)
            detections = self.build_detections(
                result,
                image,
            )

            selected = self.select_targets_for_publish(
                detections
            )

            # All selected targets from this inference frame share exactly the
            # same stamp. The depth node can therefore batch them reliably.
            for detection in selected:
                self.publish_target(
                    detection,
                    stamp,
                )

            selected_keys = [
                "{}:{:.3f}".format(
                    item["class_name"],
                    float(item["confidence"]),
                )
                for item in selected
            ]
            rospy.loginfo_throttle(
                2.0,
                "YOLO selected %d target(s): %s",
                len(selected),
                ", ".join(selected_keys) if selected_keys else "none",
            )

            web_detections = detections[: self.top_k]
            payload = {
                "stamp": stamp.to_sec(),
                "source": "ultralytics",
                "node": "yolo_unified_detector",
                "task_mode": self.task_mode,
                "detect_mode": self.detect_mode,
                "frame_id": header.frame_id,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "candidate_conf_thre": self.inference_conf_thre,
                "default_class_conf_thre": self.conf_thre,
                "target_selection_mode": self.target_selection_mode,
                "candidate_count": int(
                    0 if result.boxes is None else len(result.boxes)
                ),
                "count": len(web_detections),
                "selected_count": len(selected),
                "selected_classes": [
                    item["class_name"]
                    for item in selected
                ],
                "detections": web_detections,
            }

            self.web_detection_pub.publish(
                String(
                    data=json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                )
            )

            annotated = None
            try:
                annotated = self.build_annotated_image(
                    image,
                    result,
                    detections,
                    selected,
                )
                annotated_msg = self.bridge.cv2_to_imgmsg(
                    annotated,
                    encoding="bgr8",
                )
                annotated_msg.header = header
                annotated_msg.header.stamp = stamp
                self.annotated_pub.publish(
                    annotated_msg
                )
            except Exception as exc:
                rospy.logerr_throttle(
                    2.0,
                    "failed to publish annotated image: %s",
                    str(exc),
                )

            if (
                self.visualization == 1
                and annotated is not None
            ):
                cv2.imshow(
                    "Unified YOLO Detection",
                    annotated,
                )
                cv2.waitKey(1)

            self.rate.sleep()

        cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Unified Ultralytics YOLO ROS node"
    )

    parser.add_argument(
        "--task_mode",
        choices=["detect", "segment3"],
        default="detect",
    )

    parser.add_argument(
        "--detect_mode",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--model_path",
        default="",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="maximum filtered detections included in Web JSON",
    )

    parser.add_argument(
        "--max_det",
        type=int,
        default=30,
        help="Ultralytics maximum candidate detections before class selection",
    )

    parser.add_argument(
        "--visualization",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--conf_thre",
        type=float,
        default=0.2,
        help="default final confidence threshold for classes",
    )

    parser.add_argument(
        "--candidate_conf_thre",
        type=float,
        default=0.05,
        help="low pre-filter threshold used for Ultralytics inference",
    )

    parser.add_argument(
        "--class_conf_thresholds",
        default="",
        help="per-class thresholds, e.g. red:0.20,green:0.10,yello:0.20",
    )

    parser.add_argument(
        "--target_selection_mode",
        choices=["global_best", "per_class_best", "all"],
        default="global_best",
    )

    parser.add_argument(
        "--target_pub_queue_size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--class_whitelist",
        default="",
        help="optional comma separated class names",
    )

    parser.add_argument(
        "--class_priority",
        default="",
        help="optional publish order, comma separated class names",
    )

    parser.add_argument(
        "--detc_type",
        choices=["center", "bbox"],
        default="center",
    )

    parser.add_argument(
        "--output_type",
        choices=["quartiles"],
        default="quartiles",
    )

    parser.add_argument(
        "--infer_rate",
        type=float,
        default=5.0,
    )

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
        "--center_topic",
        default="/yolo_unified/target_center",
    )

    parser.add_argument(
        "--bbox_topic",
        default="/yolo_unified/target_bbox",
    )

    parser.add_argument(
        "--line_topic",
        default="/yolo_unified/line_bbox",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    parsed_args = parser.parse_args(
        rospy.myargv()[1:]
    )

    try:
        node = UnifiedYOLODetector(
            parsed_args
        )
        node.run()

    except rospy.ROSInterruptException:
        pass

    except Exception as exc:
        rospy.logfatal(
            "YOLO node failed: %s",
            str(exc),
        )
        raise