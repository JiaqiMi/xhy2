#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

import argparse
import heapq
import json
import os
import threading
import time
import torch

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32, PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from ultralytics import YOLO

from stereo_depth.msg import BoundingBox, LinePixels


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

    detect_models = {
        1: "/home/xhy/catkin_ws/models/shapes0709.pt",
        2: "/home/xhy/catkin_ws/models/rectangle0710.pt",
        3: "/home/xhy/catkin_ws/models/line0709.pt",
        4: "/home/xhy/catkin_ws/models/arrow0709.pt"
    }

    segment_models = {
        1: "/home/xhy/catkin_ws/models/shapes_model0719.pt",
        2: "/home/xhy/catkin_ws/models/holes_model0719.pt",
        3: "/home/xhy/catkin_ws/models/balls_model0725.pt",
        4: "/home/xhy/catkin_ws/models/line0709.pt",
    }

    if task_mode == "segment3":
        # Backward-compatible alias.
        task_mode = "segment_line"

    if task_mode == "detect":
        model_map = detect_models
    elif task_mode == "segment_line":
        model_map = segment_models
    else:
        raise ValueError(
            "invalid task_mode: {}. Expected detect or segment_line.".format(
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


class UnifiedYOLODetector:
    def __init__(self, args):
        rospy.init_node("yolo_unified_detector", anonymous=False)

        self.task_mode = str(args.task_mode).strip().lower()
        if self.task_mode == "segment_line":
            # Backward-compatible alias used by the old launch files.
            self.task_mode = "segment_line"

        self.detect_mode = int(args.detect_mode)
        self.top_k = max(1, int(args.top_k))
        self.visualization = int(args.visualization)
        self.conf_thre = float(args.conf_thre)
        self.detc_type = str(args.detc_type).strip().lower()
        self.output_type = str(args.output_type).strip().lower()
        self.line_sample_count = max(2, int(args.line_sample_count))
        self.rate = rospy.Rate(max(0.5, float(args.infer_rate)))

        if self.task_mode not in ("detect", "segment_line"):
            raise ValueError("task_mode must be detect or segment_line")

        if self.task_mode == "detect" and self.detc_type not in (
            "center",
            "bbox",
        ):
            raise ValueError(
                "detect task requires detc_type=center or bbox"
            )

        if (
            self.task_mode == "segment_line"
            and self.output_type not in ("uniform_path", "quartiles")
        ):
            raise ValueError(
                "segment_line supports output_type=uniform_path or quartiles"
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
                    queue_size=1,
                )
            else:
                self.target_pub = rospy.Publisher(
                    self.bbox_topic,
                    BoundingBox,
                    queue_size=1,
                )
        else:
            self.target_pub = rospy.Publisher(
                self.line_topic,
                LinePixels,
                queue_size=1,
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
        if self.task_mode == "segment_line":
            rospy.loginfo(
                "line_sample_count=%d, line_topic=%s",
                self.line_sample_count,
                self.line_topic,
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
    def _farthest_skeleton_node(start, skeleton_points):
        """
        Dijkstra search on an 8-connected skeleton graph.

        Returns:
            farthest_node: (y, x)
            parent: dict used to reconstruct the path
            distance: shortest-path distance from start
        """
        point_set = set(skeleton_points)
        queue = [(0.0, start)]
        distance = {start: 0.0}
        parent = {start: None}

        neighbors = (
            (-1, -1, 1.41421356237),
            (-1, 0, 1.0),
            (-1, 1, 1.41421356237),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (1, -1, 1.41421356237),
            (1, 0, 1.0),
            (1, 1, 1.41421356237),
        )

        while queue:
            current_distance, current = heapq.heappop(queue)
            if current_distance > distance[current]:
                continue

            cy, cx = current
            for dy, dx, edge_cost in neighbors:
                nxt = (cy + dy, cx + dx)
                if nxt not in point_set:
                    continue

                candidate = current_distance + edge_cost
                if candidate < distance.get(nxt, float("inf")):
                    distance[nxt] = candidate
                    parent[nxt] = current
                    heapq.heappush(queue, (candidate, nxt))

        farthest = max(distance, key=distance.get)
        return farthest, parent, distance

    @classmethod
    def sample_skeleton_path(cls, skeleton, sample_count):
        """
        Extract the main geodesic path of a skeleton and sample points
        uniformly by arc length.

        This is more appropriate than lexicographically sorting all skeleton
        pixels, especially for tilted or curved lines.
        """
        points_yx = np.column_stack(np.where(skeleton > 0))
        if points_yx.shape[0] < 2:
            return None

        skeleton_points = [
            (int(point[0]), int(point[1]))
            for point in points_yx
        ]

        # Two-sweep graph diameter. It is exact for tree-like skeletons and
        # a good approximation for skeletons containing small loops.
        first_start = skeleton_points[0]
        endpoint_a, _, _ = cls._farthest_skeleton_node(
            first_start,
            skeleton_points,
        )
        endpoint_b, parent, _ = cls._farthest_skeleton_node(
            endpoint_a,
            skeleton_points,
        )

        path_yx = []
        current = endpoint_b
        while current is not None:
            path_yx.append(current)
            if current == endpoint_a:
                break
            current = parent.get(current)

        if not path_yx or path_yx[-1] != endpoint_a:
            return None

        path_yx.reverse()

        # Keep point order deterministic from frame to frame:
        # horizontal lines: left -> right
        # vertical lines: top -> bottom
        y0, x0 = path_yx[0]
        y1, x1 = path_yx[-1]
        if abs(x1 - x0) >= abs(y1 - y0):
            if x0 > x1:
                path_yx.reverse()
        elif y0 > y1:
            path_yx.reverse()

        path_xy = np.asarray(
            [(float(x), float(y)) for y, x in path_yx],
            dtype=np.float64,
        )

        if path_xy.shape[0] == 1:
            x, y = path_xy[0]
            return [(int(round(x)), int(round(y)))]

        segment_lengths = np.linalg.norm(
            np.diff(path_xy, axis=0),
            axis=1,
        )
        cumulative = np.concatenate(
            ([0.0], np.cumsum(segment_lengths))
        )
        total_length = float(cumulative[-1])

        if total_length <= 1e-6:
            return None

        requested = max(2, int(sample_count))
        actual_count = min(requested, path_xy.shape[0])
        targets = np.linspace(0.0, total_length, actual_count)

        selected = []
        used = set()

        for target in targets:
            index = int(np.searchsorted(cumulative, target))
            index = min(max(index, 0), path_xy.shape[0] - 1)

            if index > 0:
                previous = index - 1
                if abs(cumulative[previous] - target) <= abs(
                    cumulative[index] - target
                ):
                    index = previous

            x, y = path_xy[index]
            point = (int(round(x)), int(round(y)))

            if point not in used:
                selected.append(point)
                used.add(point)

        # Very short paths can cause nearest-neighbor duplicates.
        # Fill remaining slots from evenly spaced path indices.
        if len(selected) < actual_count:
            fallback_indices = np.linspace(
                0,
                path_xy.shape[0] - 1,
                actual_count,
            ).round().astype(int)

            for index in fallback_indices:
                x, y = path_xy[int(index)]
                point = (int(round(x)), int(round(y)))
                if point not in used:
                    selected.append(point)
                    used.add(point)
                if len(selected) >= actual_count:
                    break

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

    def build_detections(self, result, image):
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        order = confs.argsort()[::-1]
        detections = []

        for index in order:
            confidence = float(confs[index])

            if confidence < self.conf_thre:
                continue

            if len(detections) >= self.top_k:
                break

            box = boxes[index]
            class_id = int(classes[index])
            class_name = str(result.names[class_id])

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

            if self.output_type == "quartiles":
                sample_count = 3
            else:
                sample_count = self.line_sample_count

            keypoints = self.sample_skeleton_path(
                skeleton,
                sample_count,
            )

            if keypoints is None:
                rospy.logwarn(
                    "not enough skeleton points for %s",
                    class_name,
                )
                continue

            item["task"] = "segment_line"
            item["output_type"] = self.output_type
            item["sample_count"] = len(keypoints)
            item["keypoints"] = [
                {
                    "x": point[0],
                    "y": point[1],
                }
                for point in keypoints
            ]

            detections.append(item)

        return detections

    def publish_best_target(self, detection, stamp):
        """
        定位链路只发布最高置信度目标；
        Web JSON 中仍保留 top_k 目标。
        """
        if detection is None:
            return

        if self.task_mode == "segment_line":
            msg = LinePixels()
            msg.header.stamp = stamp
            msg.header.frame_id = self.left_header.frame_id
            msg.class_name = detection["class_name"]
            msg.conf = float(detection["confidence"])

            msg.points = []
            for point in detection["keypoints"]:
                pixel = Point32()
                pixel.x = float(point["x"])
                pixel.y = float(point["y"])
                pixel.z = 0.0
                msg.points.append(pixel)

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

            # 推理前
            model_device = next(
                self.model.model.parameters()
            ).device

            use_cuda = model_device.type == "cuda"

            if use_cuda:
                torch.cuda.synchronize()

            t0 = time.perf_counter()

            try:
                results = self.model(
                    image,
                    conf=self.conf_thre,
                    max_det=self.top_k,
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
            
            # 模型推理
            if use_cuda:
                torch.cuda.synchronize()

            t1 = time.perf_counter()

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

            # 管线后处理
            t2 = time.perf_counter()

            self.publish_best_target(
                detections[0] if detections else None,
                stamp,
            )

            payload = {
                "stamp": stamp.to_sec(),
                "source": "ultralytics",
                "node": "yolo_unified_detector",
                "task_mode": self.task_mode,
                "detect_mode": self.detect_mode,
                "frame_id": header.frame_id,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "count": len(detections),
                "detections": detections,
            }

            self.web_detection_pub.publish(
                String(
                    data=json.dumps(
                        payload,
                        ensure_ascii=False,
                    )
                )
            )

            # 发布管线点和JSON
            t3 = time.perf_counter()
            annotated = None

            try:
                annotated = result.plot()

                if self.task_mode == "segment_line":
                    for item in detections:
                        for index, point in enumerate(
                            item.get("keypoints", [])
                        ):
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

                            # Avoid clutter when 10-20 points are drawn.
                            if (
                                index == 0
                                or index == len(item.get("keypoints", [])) - 1
                                or index == len(item.get("keypoints", [])) // 2
                            ):
                                cv2.putText(
                                    annotated,
                                    "P{}".format(index + 1),
                                    (
                                        center[0] + 7,
                                        center[1] - 7,
                                    ),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 255, 255),
                                    2,
                                    cv2.LINE_AA,
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

                t4 = time.perf_counter()

            except Exception as exc:
                rospy.logerr_throttle(
                    2.0,
                    "failed to publish annotated image: %s",
                    str(exc),
                )
            
            # rospy.loginfo_throttle(
            #     2.0,
            #     (
            #         "YOLO timing: "
            #         "infer=%.1f ms, "
            #         "line_post=%.1f ms, "
            #         "message=%.1f ms, "
            #         "plot_publish=%.1f ms, "
            #         "total=%.1f ms"
            #     ),
            #     (t1 - t0) * 1000.0,
            #     (t2 - t1) * 1000.0,
            #     (t3 - t2) * 1000.0,
            #     (t4 - t3) * 1000.0,
            #     (t4 - t0) * 1000.0,
            # )

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
        choices=["detect", "segment3", "segment_line"],
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
        default=3,
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
    )

    parser.add_argument(
        "--detc_type",
        choices=["center", "bbox"],
        default="center",
    )

    parser.add_argument(
        "--output_type",
        choices=["quartiles", "uniform_path"],
        default="uniform_path",
    )

    parser.add_argument(
        "--line_sample_count",
        type=int,
        default=20,
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
        default="/yolo_unified/line_points",
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