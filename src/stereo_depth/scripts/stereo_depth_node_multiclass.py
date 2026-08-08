#!/home/nvidia/venvs/xhy_ros2/bin/python
# -*- coding: utf-8 -*-

"""
加速推理脚本,并对不同目标进行同步处理
"""

import json
import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
from auv_control.msg import TargetDetection, TargetDetection3
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from std_msgs.msg import String
from stereo_depth.msg import BoundingBox, LineBox


def is_finite_point(point):
    return bool(
        point is not None
        and len(point) == 3
        and np.all(np.isfinite(point))
    )


class UnifiedStereoDepthNode:
    def __init__(self):
        rospy.init_node("stereo_depth_unified", anonymous=False)

        self.task_mode = str(rospy.get_param("~task_mode", "center")).lower()
        if self.task_mode not in ("center", "bbox", "line3"):
            raise ValueError("task_mode must be center, bbox or line3")

        self.exp_env = str(rospy.get_param("~exp_env", "water")).lower()
        self.visualization = int(rospy.get_param("~visualization", 0))
        self.conf_thre = float(rospy.get_param("~conf_thre", 0.2))
        self.rate = rospy.Rate(max(0.5, float(rospy.get_param("~rate", 5.0))))

        self.window_size = int(rospy.get_param("~window_size", 25))
        self.min_depth = float(rospy.get_param("~min_depth", 0.5))
        self.max_depth = float(rospy.get_param("~max_depth", 2.0))
        self.min_valid_pixels = int(rospy.get_param("~min_valid_pixels", 3))
        self.depth_statistic = str(
            rospy.get_param(
                "~depth_statistic",
                "min" if self.task_mode == "line3" else "mean",
            )
        ).lower()

        if self.depth_statistic not in ("mean", "median", "min"):
            raise ValueError("depth_statistic must be mean, median or min")

        self.require_line_depth_order = bool(
            rospy.get_param("~require_line_depth_order", False)
        )
        self.reverse_line_points = bool(
            rospy.get_param("~reverse_line_points", False)
        )

        self.max_sync_dt = float(rospy.get_param("~max_sync_dt", 0.15))
        self.frame_buffer_size = int(rospy.get_param("~frame_buffer_size", 20))

        # Multi-class target batching. For center/bbox tasks, detections from
        # the same inference frame are grouped by timestamp. Only the highest
        # confidence target of each class is retained.
        self.multi_class_targets = bool(
            rospy.get_param("~multi_class_targets", True)
        )
        self.target_batch_wait_sec = max(
            0.0,
            float(rospy.get_param("~target_batch_wait_sec", 0.03)),
        )
        self.target_group_tolerance = max(
            0.0,
            float(rospy.get_param("~target_group_tolerance", 0.01)),
        )
        self.max_pending_target_batches = max(
            2,
            int(rospy.get_param("~max_pending_target_batches", 10)),
        )
        self.target_sub_queue_size = max(
            1,
            int(rospy.get_param("~target_sub_queue_size", 20)),
        )
        self.publisher_queue_size = max(
            1,
            int(rospy.get_param("~publisher_queue_size", 10)),
        )

        self.multi_target_depth_mode = str(
            rospy.get_param("~multi_target_depth_mode", "auto")
        ).strip().lower()
        if self.multi_target_depth_mode not in (
            "auto",
            "combined",
            "per_target",
        ):
            raise ValueError(
                "multi_target_depth_mode must be auto, combined or per_target"
            )

        self.combined_roi_cost_ratio = max(
            1.0,
            float(rospy.get_param("~combined_roi_cost_ratio", 1.20)),
        )

        raw_whitelist = rospy.get_param("~class_whitelist", [])
        if isinstance(raw_whitelist, str):
            raw_whitelist = [
                item.strip()
                for item in raw_whitelist.split(",")
                if item.strip()
            ]
        self.class_whitelist = {
            str(item).strip().lower()
            for item in raw_whitelist
            if str(item).strip()
        }

        raw_priority = rospy.get_param("~class_priority", [])
        if isinstance(raw_priority, str):
            raw_priority = [
                item.strip()
                for item in raw_priority.split(",")
                if item.strip()
            ]
        self.class_priority = [
            str(item).strip().lower()
            for item in raw_priority
            if str(item).strip()
        ]
        self.class_priority_index = {
            class_name: index
            for index, class_name in enumerate(self.class_priority)
        }

        self.left_topic = rospy.get_param("~left_topic", "/left/image_raw")
        self.right_topic = rospy.get_param("~right_topic", "/right/image_raw")
        self.center_topic = rospy.get_param(
            "~center_topic", "/yolo_unified/target_center"
        )
        self.bbox_topic = rospy.get_param(
            "~bbox_topic", "/yolo_unified/target_bbox"
        )
        self.line_topic = rospy.get_param(
            "~line_topic", "/yolo_unified/line_bbox"
        )
        self.target_output_topic = rospy.get_param(
            "~target_output_topic", "/obj/target_message"
        )
        self.line_output_topic = rospy.get_param(
            "~line_output_topic", "/obj/line_message"
        )
        self.web_pose_topic = rospy.get_param("~web_pose_topic", "/web/pose")

        self.load_camera_parameters()

        self.bridge = CvBridge()
        self.data_lock = threading.Lock()
        self.frame_buffer = deque(maxlen=max(2, self.frame_buffer_size))

        # Legacy single-target storage is retained for line3 and for cases
        # where multi_class_targets is disabled.
        self.pending_target = None

        # Each batch stores one highest-confidence target per class.
        self.pending_target_batches = deque(
            maxlen=self.max_pending_target_batches
        )

        left_sub = Subscriber(self.left_topic, Image)
        right_sub = Subscriber(self.right_topic, Image)
        self.sync = ApproximateTimeSynchronizer(
            [left_sub, right_sub],
            queue_size=10,
            slop=0.1,
        )
        self.sync.registerCallback(self.stereo_callback)

        if self.task_mode == "center":
            self.target_sub = rospy.Subscriber(
                self.center_topic,
                PointStamped,
                self.center_callback,
                queue_size=self.target_sub_queue_size,
            )
        elif self.task_mode == "bbox":
            self.target_sub = rospy.Subscriber(
                self.bbox_topic,
                BoundingBox,
                self.bbox_callback,
                queue_size=self.target_sub_queue_size,
            )
        else:
            self.target_sub = rospy.Subscriber(
                self.line_topic,
                LineBox,
                self.line_callback,
                queue_size=self.target_sub_queue_size,
            )

        self.target_pub = rospy.Publisher(
            self.target_output_topic,
            TargetDetection,
            queue_size=self.publisher_queue_size,
        )
        self.line_pub = rospy.Publisher(
            self.line_output_topic,
            TargetDetection3,
            queue_size=self.publisher_queue_size,
        )
        self.web_pub = rospy.Publisher(
            self.web_pose_topic,
            String,
            queue_size=self.publisher_queue_size,
        )

        self.min_disparity = int(rospy.get_param("~min_disparity", 0))
        self.num_disparities = int(rospy.get_param("~num_disparities", 96))
        self.block_size = int(rospy.get_param("~block_size", 7))

        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer >= 3")

        # Depth calculation mode:
        #   roi  - run SGBM only around requested target pixels;
        #   full - preserve the original full-frame SGBM behavior.
        self.depth_compute_mode = str(
            rospy.get_param("~depth_compute_mode", "roi")
        ).strip().lower()
        if self.depth_compute_mode not in ("roi", "full"):
            raise ValueError("depth_compute_mode must be roi or full")

        self.use_sgbm_3way = bool(
            rospy.get_param("~use_sgbm_3way", True)
        )
        self.roi_extra_margin_x = max(
            0,
            int(rospy.get_param("~roi_extra_margin_x", 0)),
        )
        self.roi_extra_margin_y = max(
            0,
            int(rospy.get_param("~roi_extra_margin_y", 0)),
        )
        self.roi_min_width = max(
            0,
            int(rospy.get_param("~roi_min_width", 0)),
        )
        self.roi_min_height = max(
            0,
            int(rospy.get_param("~roi_min_height", 64)),
        )
        self.roi_full_frame_threshold = float(
            rospy.get_param("~roi_full_frame_threshold", 0.90)
        )
        self.roi_full_frame_threshold = min(
            max(self.roi_full_frame_threshold, 0.05),
            1.0,
        )
        self.roi_fallback_full_frame = bool(
            rospy.get_param("~roi_fallback_full_frame", True)
        )
        self.log_depth_timing = bool(
            rospy.get_param("~log_depth_timing", True)
        )

        cv2.setUseOptimized(True)

        if self.use_sgbm_3way:
            self.sgbm_mode = getattr(
                cv2,
                "STEREO_SGBM_MODE_SGBM_3WAY",
                cv2.STEREO_SGBM_MODE_SGBM,
            )
        else:
            self.sgbm_mode = cv2.STEREO_SGBM_MODE_SGBM

        channels = 1
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * channels * self.block_size ** 2,
            P2=32 * channels * self.block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            mode=self.sgbm_mode,
        )

        self.last_depth_roi = None
        self.last_depth_roi_ratio = 1.0

        rospy.loginfo(
            "Depth node initialized: task=%s, max_sync_dt=%.3f, "
            "depth_mode=%s, sgbm_mode=%s, multi_class=%s, "
            "multi_depth_mode=%s",
            self.task_mode,
            self.max_sync_dt,
            self.depth_compute_mode,
            "3WAY"
            if self.sgbm_mode
            == getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", -1)
            else "SGBM",
            str(self.multi_class_targets),
            self.multi_target_depth_mode,
        )

    def load_camera_parameters(self):
        if self.exp_env == "air":
            defaults = {
                "fx": 572.993971,
                "fy": 572.993971,
                "cx": 374.534946,
                "cy": 271.474743,
                "baseline": 34.309807 / 572.993971,
            }
        elif self.exp_env == "water":
            defaults = {
                "fx": 798.731044,
                "fy": 798.731044,
                "cx": 348.127430,
                "cy": 269.935493,
                "baseline": 47.694354 / 798.731044,
            }
        else:
            raise ValueError("exp_env must be air or water")

        self.fx = float(rospy.get_param("~fx", defaults["fx"]))
        self.fy = float(rospy.get_param("~fy", defaults["fy"]))
        self.cx = float(rospy.get_param("~cx", defaults["cx"]))
        self.cy = float(rospy.get_param("~cy", defaults["cy"]))
        self.baseline = float(
            rospy.get_param("~baseline", defaults["baseline"])
        )

    @staticmethod
    def valid_stamp(stamp):
        if stamp is None or stamp == rospy.Time():
            return rospy.Time.now()
        return stamp

    def stereo_callback(self, left_msg, right_msg):
        try:
            left = self.bridge.imgmsg_to_cv2(
                left_msg, desired_encoding="bgr8"
            )
            right = self.bridge.imgmsg_to_cv2(
                right_msg, desired_encoding="bgr8"
            )
        except Exception as exc:
            rospy.logerr_throttle(2.0, "cv_bridge error: %s", str(exc))
            return

        stamp = self.valid_stamp(left_msg.header.stamp)

        with self.data_lock:
            self.frame_buffer.append(
                {
                    "stamp": stamp,
                    "stamp_sec": stamp.to_sec(),
                    "left": left,
                    "right": right,
                }
            )

    @staticmethod
    def normalize_class_name(class_name):
        return str(class_name).strip().lower()

    def target_class_allowed(self, class_name):
        if not self.class_whitelist:
            return True
        return (
            self.normalize_class_name(class_name)
            in self.class_whitelist
        )

    def set_target(self, target):
        """Legacy single-target path."""
        with self.data_lock:
            self.pending_target = target

    def add_target_to_batch(self, target):
        """
        Group center/bbox targets by image timestamp and retain the maximum
        confidence target for each class within that batch.
        """
        class_key = self.normalize_class_name(
            target["class_name"]
        )
        if not self.target_class_allowed(class_key):
            return

        target_sec = target["stamp"].to_sec()
        now_wall = time.monotonic()

        with self.data_lock:
            matched_batch = None

            for batch in reversed(self.pending_target_batches):
                if abs(batch["stamp_sec"] - target_sec) <= (
                    self.target_group_tolerance
                ):
                    matched_batch = batch
                    break

            if matched_batch is None:
                matched_batch = {
                    "stamp": target["stamp"],
                    "stamp_sec": target_sec,
                    "first_receive_wall": now_wall,
                    "targets_by_class": {},
                }
                self.pending_target_batches.append(
                    matched_batch
                )

            previous = matched_batch[
                "targets_by_class"
            ].get(class_key)

            if (
                previous is None
                or target["confidence"]
                > previous["confidence"]
            ):
                matched_batch["targets_by_class"][
                    class_key
                ] = target

    def enqueue_target(self, target):
        if (
            self.multi_class_targets
            and self.task_mode in ("center", "bbox")
        ):
            self.add_target_to_batch(target)
        else:
            self.set_target(target)

    def center_callback(self, msg):
        self.enqueue_target(
            {
                "class_name": str(msg.header.frame_id),
                "confidence": float(msg.point.z),
                "stamp": self.valid_stamp(msg.header.stamp),
                "pixels": [(int(msg.point.x), int(msg.point.y))],
                "input_type": "center",
            }
        )

    def bbox_callback(self, msg):
        x1, y1, x2, y2 = (
            int(msg.x1),
            int(msg.y1),
            int(msg.x2),
            int(msg.y2),
        )
        u = int(round((x1 + x2) / 2.0))
        v = int(round((y1 + y2) / 2.0))

        self.enqueue_target(
            {
                "class_name": str(msg.header.frame_id),
                "confidence": float(msg.conf),
                "stamp": self.valid_stamp(msg.header.stamp),
                "pixels": [(u, v)],
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "input_type": "bbox",
            }
        )

    def line_callback(self, msg):
        pixels = [
            (int(msg.x1), int(msg.y1)),
            (int(msg.x2), int(msg.y2)),
            (int(msg.x3), int(msg.y3)),
        ]
        if self.reverse_line_points:
            pixels.reverse()

        self.set_target(
            {
                "class_name": str(msg.header.frame_id),
                "confidence": float(msg.conf),
                "stamp": self.valid_stamp(msg.header.stamp),
                "pixels": pixels,
                "input_type": "line3",
            }
        )

    def sort_targets(self, targets):
        def sort_key(target):
            class_key = self.normalize_class_name(
                target["class_name"]
            )
            priority = self.class_priority_index.get(
                class_key,
                len(self.class_priority_index),
            )
            return (
                priority,
                class_key,
                -float(target["confidence"]),
            )

        return sorted(targets, key=sort_key)

    def take_targets_and_frame(self):
        """
        Return one synchronized stereo frame and one or more targets.

        For center/bbox multi-class mode, one batch contains at most one target
        per class. For line3 or disabled multi-class mode, the original
        single-target behavior is preserved.
        """
        with self.data_lock:
            if not self.frame_buffer:
                return None, None, None

            frames = list(self.frame_buffer)

            use_batch = (
                self.multi_class_targets
                and self.task_mode in ("center", "bbox")
            )

            if use_batch:
                if not self.pending_target_batches:
                    return None, None, None

                batch = self.pending_target_batches[0]
                batch_age = (
                    time.monotonic()
                    - batch["first_receive_wall"]
                )

                # Wait briefly so all detections from one inference frame can
                # arrive before the per-class maxima are finalized.
                if (
                    batch_age < self.target_batch_wait_sec
                    and len(self.pending_target_batches) == 1
                ):
                    return None, None, None

                batch = self.pending_target_batches.popleft()
                targets = list(
                    batch["targets_by_class"].values()
                )
                target_sec = batch["stamp_sec"]
            else:
                if self.pending_target is None:
                    return None, None, None

                target = self.pending_target
                self.pending_target = None
                targets = [target]
                target_sec = target["stamp"].to_sec()

        if not targets:
            return None, None, None

        frame = min(
            frames,
            key=lambda item: abs(
                item["stamp_sec"] - target_sec
            ),
        )
        sync_dt = abs(frame["stamp_sec"] - target_sec)

        targets = self.sort_targets(targets)

        batch_size = len(targets)
        for index, target in enumerate(targets):
            target["batch_size"] = batch_size
            target["batch_index"] = index

        return targets, frame, sync_dt

    @staticmethod
    def _expand_interval(start, end, minimum_size, limit):
        """
        Expand [start, end) to at least minimum_size while keeping it inside
        [0, limit). This is used only for the internal SGBM crop.
        """
        start = int(max(0, min(start, limit)))
        end = int(max(start, min(end, limit)))

        current_size = end - start
        if current_size >= minimum_size or current_size >= limit:
            return start, end

        missing = int(minimum_size - current_size)
        grow_before = missing // 2
        grow_after = missing - grow_before

        start -= grow_before
        end += grow_after

        if start < 0:
            end = min(limit, end - start)
            start = 0

        if end > limit:
            start = max(0, start - (end - limit))
            end = limit

        return int(start), int(end)

    def calculate_depth_roi(self, image_shape, pixels):
        """
        Calculate a safe stereo ROI around all requested target pixels.

        Left and right images use the same crop origin, so the resulting
        disparity remains expressed in the original image coordinate system.
        Extra pixels are retained on the left to cover the positive disparity
        search range required by StereoSGBM.
        """
        height, width = image_shape[:2]
        full_roi = (0, 0, width, height)

        if self.depth_compute_mode == "full":
            return full_roi

        valid_pixels = [
            (int(u), int(v))
            for u, v in (pixels or [])
            if 0 <= int(u) < width and 0 <= int(v) < height
        ]
        if not valid_pixels:
            return full_roi

        u_values = [point[0] for point in valid_pixels]
        v_values = [point[1] for point in valid_pixels]

        half_window = max(1, self.window_size // 2)
        block_margin = (
            max(8, self.block_size * 2)
            + self.roi_extra_margin_x
        )
        vertical_margin = (
            half_window
            + max(16, self.block_size * 3)
            + self.roi_extra_margin_y
        )

        max_positive_disparity = max(
            0,
            self.min_disparity + self.num_disparities - 1,
        )
        max_negative_disparity = max(0, -self.min_disparity)

        x0 = (
            min(u_values)
            - half_window
            - block_margin
            - max_positive_disparity
        )
        x1 = (
            max(u_values)
            + half_window
            + block_margin
            + max_negative_disparity
            + 1
        )
        y0 = min(v_values) - vertical_margin
        y1 = max(v_values) + vertical_margin + 1

        x0 = max(0, int(x0))
        x1 = min(width, int(x1))
        y0 = max(0, int(y0))
        y1 = min(height, int(y1))

        automatic_min_width = (
            self.num_disparities + 2 * block_margin + 16
        )
        minimum_width = min(
            width,
            max(automatic_min_width, self.roi_min_width),
        )

        automatic_min_height = max(
            64,
            self.window_size + 2 * vertical_margin,
        )
        minimum_height = min(
            height,
            max(automatic_min_height, self.roi_min_height),
        )

        x0, x1 = self._expand_interval(
            x0,
            x1,
            minimum_width,
            width,
        )
        y0, y1 = self._expand_interval(
            y0,
            y1,
            minimum_height,
            height,
        )

        roi_area = max(0, x1 - x0) * max(0, y1 - y0)
        full_area = max(1, width * height)
        roi_ratio = float(roi_area) / float(full_area)

        # For an ROI that is already close to the full image, use the original
        # full frame to avoid changing border behavior for little speed gain.
        if roi_ratio >= self.roi_full_frame_threshold:
            return full_roi

        return int(x0), int(y0), int(x1), int(y1)

    def _compute_depth_crop(self, left, right, roi):
        x0, y0, x1, y1 = roi

        left_roi = np.ascontiguousarray(
            left[y0:y1, x0:x1]
        )
        right_roi = np.ascontiguousarray(
            right[y0:y1, x0:x1]
        )

        if left_roi.size == 0 or right_roi.size == 0:
            raise ValueError("empty stereo ROI")

        gray_left = cv2.cvtColor(
            left_roi,
            cv2.COLOR_BGR2GRAY,
        )
        gray_right = cv2.cvtColor(
            right_roi,
            cv2.COLOR_BGR2GRAY,
        )

        disparity = (
            self.stereo_matcher.compute(
                gray_left,
                gray_right,
            ).astype(np.float32)
            / 16.0
        )

        depth_roi = np.full(
            disparity.shape,
            np.nan,
            dtype=np.float32,
        )
        valid = disparity > 0.0
        depth_roi[valid] = (
            self.fx * self.baseline / disparity[valid]
        )
        return depth_roi

    def compute_depth(self, left, right, pixels=None):
        """
        Run StereoSGBM either on the target ROI or on the full image.

        To keep pixel_to_3d(), visualization, publishers and output messages
        unchanged, the returned array still has the original image size.
        Pixels outside the calculated ROI remain NaN; no full-frame disparity
        calculation is performed in ROI mode.
        """
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("left and right image sizes differ")

        height, width = left.shape[:2]
        full_roi = (0, 0, width, height)
        roi = self.calculate_depth_roi(
            left.shape,
            pixels,
        )

        try:
            depth_roi = self._compute_depth_crop(
                left,
                right,
                roi,
            )
        except cv2.error as exc:
            if (
                roi == full_roi
                or not self.roi_fallback_full_frame
            ):
                raise

            rospy.logwarn_throttle(
                2.0,
                "ROI SGBM failed (%s); falling back to full frame",
                str(exc),
            )
            roi = full_roi
            depth_roi = self._compute_depth_crop(
                left,
                right,
                roi,
            )

        x0, y0, x1, y1 = roi
        depth = np.full(
            (height, width),
            np.nan,
            dtype=np.float32,
        )
        depth[y0:y1, x0:x1] = depth_roi

        self.last_depth_roi = roi
        self.last_depth_roi_ratio = (
            float((x1 - x0) * (y1 - y0))
            / float(max(1, width * height))
        )

        return depth

    def pixel_to_3d(self, u, v, depth):
        height, width = depth.shape
        u, v = int(u), int(v)

        if not (0 <= u < width and 0 <= v < height):
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        half = self.window_size // 2
        region = depth[
            max(0, v - half):min(height, v + half + 1),
            max(0, u - half):min(width, u + half + 1),
        ]

        values = region[
            np.isfinite(region)
            & (region >= self.min_depth)
            & (region <= self.max_depth)
        ]

        if values.size < self.min_valid_pixels:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        if self.depth_statistic == "min":
            z = float(np.min(values))
        elif self.depth_statistic == "median":
            z = float(np.median(values))
        else:
            z = float(np.mean(values))

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.array([x, y, z], dtype=np.float64)

    def point_valid(self, point):
        return bool(
            is_finite_point(point)
            and -1.0 < point[0] < 1.0
            and -1.0 < point[1] < 1.0
            and self.min_depth <= point[2] <= self.max_depth
        )

    @staticmethod
    def make_pose(point, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "camera"
        pose.pose.position.x = float(point[0])
        pose.pose.position.y = float(point[1])
        pose.pose.position.z = float(point[2])
        pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        return pose

    def publish_web(self, payload):
        self.web_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False))
        )

    def publish_invalid(self, target, reason, sync_dt=None):
        payload = {
            "stamp": target["stamp"].to_sec(),
            "source": "stereo_depth_unified",
            "task_mode": self.task_mode,
            "input_type": target["input_type"],
            "frame_id": "camera",
            "class_name": target["class_name"],
            "confidence": target["confidence"],
            "valid": False,
            "reason": reason,
        }
        if sync_dt is not None:
            payload["sync_dt_sec"] = float(sync_dt)
        self.publish_web(payload)

    def process_single(self, target, depth):
        u, v = target["pixels"][0]
        point = self.pixel_to_3d(u, v, depth)
        valid = self.point_valid(point)

        if valid:
            msg = TargetDetection()
            msg.pose = self.make_pose(point, target["stamp"])
            msg.type = target["input_type"]
            msg.conf = target["confidence"]
            msg.class_name = target["class_name"]
            self.target_pub.publish(msg)

        payload = {
            "stamp": target["stamp"].to_sec(),
            "source": "stereo_depth_unified",
            "task_mode": self.task_mode,
            "input_type": target["input_type"],
            "frame_id": "camera",
            "class_name": target["class_name"],
            "confidence": target["confidence"],
            "valid": valid,
            "pixel_center": {"u": u, "v": v},
        }
        if "bbox" in target:
            payload["bbox"] = target["bbox"]

        if "batch_size" in target:
            payload["batch_size"] = int(target["batch_size"])
            payload["batch_index"] = int(target["batch_index"])

        if valid:
            payload["position_m"] = {
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
            }
        else:
            payload["reason"] = "invalid_depth_or_position"

        self.publish_web(payload)

    def process_line(self, target, depth):
        points = [
            self.pixel_to_3d(u, v, depth)
            for u, v in target["pixels"]
        ]
        valid = all(self.point_valid(point) for point in points)

        if valid and self.require_line_depth_order:
            valid = bool(points[0][2] <= points[1][2] <= points[2][2])

        if valid:
            poses = [self.make_pose(point, target["stamp"]) for point in points]
            msg = TargetDetection3()
            msg.pose1, msg.pose2, msg.pose3 = poses
            msg.type = "line3"
            msg.conf = target["confidence"]
            msg.class_name = target["class_name"]
            self.line_pub.publish(msg)

        payload = {
            "stamp": target["stamp"].to_sec(),
            "source": "stereo_depth_unified",
            "task_mode": "line3",
            "input_type": "line3",
            "frame_id": "camera",
            "class_name": target["class_name"],
            "confidence": target["confidence"],
            "valid": valid,
            "pixel_keypoints": [
                {"x": u, "y": v} for u, v in target["pixels"]
            ],
        }

        if valid:
            payload["positions_m"] = [
                {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                for p in points
            ]
            payload["position_m"] = payload["positions_m"][1]
            payload["representative_point"] = "P2"
        else:
            payload["reason"] = "invalid_line_depth_or_order"

        self.publish_web(payload)

    @staticmethod
    def roi_area(roi):
        x0, y0, x1, y1 = roi
        return max(0, x1 - x0) * max(0, y1 - y0)

    def choose_multi_target_depth_mode(
        self,
        image_shape,
        targets,
    ):
        """
        auto:
          - use one combined ROI when it is not much larger than the sum of
            individual ROIs;
          - otherwise run one small ROI per class.
        """
        if len(targets) <= 1:
            return "per_target"

        if self.multi_target_depth_mode in (
            "combined",
            "per_target",
        ):
            return self.multi_target_depth_mode

        all_pixels = [
            pixel
            for target in targets
            for pixel in target["pixels"]
        ]
        combined_roi = self.calculate_depth_roi(
            image_shape,
            all_pixels,
        )
        combined_area = self.roi_area(combined_roi)

        individual_area_sum = 0
        for target in targets:
            individual_roi = self.calculate_depth_roi(
                image_shape,
                target["pixels"],
            )
            individual_area_sum += self.roi_area(
                individual_roi
            )

        if combined_area <= (
            individual_area_sum
            * self.combined_roi_cost_ratio
        ):
            return "combined"

        return "per_target"

    def process_target_batch(self, targets, frame):
        """
        Compute and publish all selected targets while keeping the existing
        output topics and message types unchanged.
        """
        if self.task_mode == "line3":
            target = targets[0]
            t0 = time.perf_counter()
            depth = self.compute_depth(
                frame["left"],
                frame["right"],
                target["pixels"],
            )
            t1 = time.perf_counter()
            self.process_line(target, depth)
            t2 = time.perf_counter()
            return {
                "mode": "per_target",
                "sgbm_ms": (t1 - t0) * 1000.0,
                "process_ms": (t2 - t1) * 1000.0,
                "total_ms": (t2 - t0) * 1000.0,
                "last_depth": depth,
                "processed_count": 1,
            }

        chosen_mode = self.choose_multi_target_depth_mode(
            frame["left"].shape,
            targets,
        )

        total_sgbm_ms = 0.0
        total_process_ms = 0.0
        last_depth = None

        if chosen_mode == "combined":
            all_pixels = [
                pixel
                for target in targets
                for pixel in target["pixels"]
            ]

            t0 = time.perf_counter()
            depth = self.compute_depth(
                frame["left"],
                frame["right"],
                all_pixels,
            )
            t1 = time.perf_counter()

            for target in targets:
                self.process_single(target, depth)

            t2 = time.perf_counter()
            total_sgbm_ms = (t1 - t0) * 1000.0
            total_process_ms = (t2 - t1) * 1000.0
            last_depth = depth
        else:
            for target in targets:
                t0 = time.perf_counter()
                depth = self.compute_depth(
                    frame["left"],
                    frame["right"],
                    target["pixels"],
                )
                t1 = time.perf_counter()

                self.process_single(target, depth)
                t2 = time.perf_counter()

                total_sgbm_ms += (
                    (t1 - t0) * 1000.0
                )
                total_process_ms += (
                    (t2 - t1) * 1000.0
                )
                last_depth = depth

        return {
            "mode": chosen_mode,
            "sgbm_ms": total_sgbm_ms,
            "process_ms": total_process_ms,
            "total_ms": (
                total_sgbm_ms + total_process_ms
            ),
            "last_depth": last_depth,
            "processed_count": len(targets),
        }

    def run(self):
        while not rospy.is_shutdown():
            targets, frame, sync_dt = (
                self.take_targets_and_frame()
            )

            if not targets:
                self.rate.sleep()
                continue

            valid_targets = []
            for target in targets:
                if target["confidence"] < self.conf_thre:
                    self.publish_invalid(
                        target,
                        "confidence_too_low",
                    )
                    continue

                if sync_dt > self.max_sync_dt:
                    self.publish_invalid(
                        target,
                        "no_synchronized_stereo_frame",
                        sync_dt=sync_dt,
                    )
                    continue

                valid_targets.append(target)

            if not valid_targets:
                self.rate.sleep()
                continue

            try:
                result = self.process_target_batch(
                    valid_targets,
                    frame,
                )
            except Exception as exc:
                rospy.logerr_throttle(
                    2.0,
                    "depth computation failed: %s",
                    str(exc),
                )
                for target in valid_targets:
                    self.publish_invalid(
                        target,
                        "depth_computation_failed",
                    )
                self.rate.sleep()
                continue

            if self.log_depth_timing:
                class_summary = ",".join(
                    "{}:{:.3f}".format(
                        target["class_name"],
                        target["confidence"],
                    )
                    for target in valid_targets
                )

                rospy.loginfo_throttle(
                    2.0,
                    (
                        "DEPTH batch: count=%d, classes=[%s], "
                        "multi_mode=%s, sgbm=%.1f ms, "
                        "point_process=%.1f ms, total=%.1f ms, "
                        "last_roi=%s, last_roi_ratio=%.3f"
                    ),
                    result["processed_count"],
                    class_summary,
                    result["mode"],
                    result["sgbm_ms"],
                    result["process_ms"],
                    result["total_ms"],
                    str(self.last_depth_roi),
                    float(self.last_depth_roi_ratio),
                )

            if (
                self.visualization
                and result["last_depth"] is not None
            ):
                self.show_visualization_batch(
                    frame["left"],
                    result["last_depth"],
                    valid_targets,
                )

            self.rate.sleep()

        cv2.destroyAllWindows()

    def show_visualization_batch(
        self,
        left,
        depth,
        targets,
    ):
        image = left.copy()

        for target in targets:
            for point_index, (u, v) in enumerate(
                target["pixels"]
            ):
                cv2.circle(
                    image,
                    (u, v),
                    6,
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    image,
                    "{}:P{}".format(
                        target["class_name"],
                        point_index + 1,
                    ),
                    (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        depth_image = cv2.normalize(
            np.nan_to_num(depth, nan=0.0),
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        ).astype(np.uint8)
        depth_image = cv2.applyColorMap(
            depth_image,
            cv2.COLORMAP_JET,
        )

        cv2.imshow("Target", image)
        cv2.imshow("Depth", depth_image)
        cv2.waitKey(1)

    def show_visualization(self, left, depth, target):
        self.show_visualization_batch(
            left,
            depth,
            [target],
        )


if __name__ == "__main__":
    try:
        node = UnifiedStereoDepthNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("depth node failed: %s", str(exc))
        raise