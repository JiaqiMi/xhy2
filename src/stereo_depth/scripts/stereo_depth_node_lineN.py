#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

import json
import threading
from collections import deque

import cv2
import numpy as np
import rospy
from auv_control.msg import LineDetection, TargetDetection
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Point32, PointStamped, PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image
from std_msgs.msg import String
from stereo_depth.msg import BoundingBox, LinePixels


def is_finite_point(point):
    return bool(
        point is not None
        and len(point) == 3
        and np.all(np.isfinite(point))
    )


class UnifiedStereoDepthNode:
    def __init__(self):
        rospy.init_node("stereo_depth_unified", anonymous=False)

        self.task_mode = str(
            rospy.get_param("~task_mode", "center")
        ).strip().lower()

        if self.task_mode == "line3":
            # Backward-compatible alias used by old launch files.
            self.task_mode = "line"

        if self.task_mode not in ("center", "bbox", "line"):
            raise ValueError("task_mode must be center, bbox or line")

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
                "median" if self.task_mode == "line" else "mean",
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

        # With 10-20 samples it is usually too strict to require every depth
        # sample to be valid. Overall line validity is determined by both an
        # absolute valid-point count and a valid ratio.
        self.min_valid_line_points = max(
            1,
            int(rospy.get_param("~min_valid_line_points", 3)),
        )
        self.min_valid_line_ratio = float(
            rospy.get_param("~min_valid_line_ratio", 0.5)
        )
        self.min_valid_line_ratio = min(
            max(self.min_valid_line_ratio, 0.0),
            1.0,
        )
        self.require_all_line_points = bool(
            rospy.get_param("~require_all_line_points", False)
        )

        self.max_sync_dt = float(rospy.get_param("~max_sync_dt", 0.15))
        self.frame_buffer_size = int(rospy.get_param("~frame_buffer_size", 20))

        self.left_topic = rospy.get_param("~left_topic", "/left/image_raw")
        self.right_topic = rospy.get_param("~right_topic", "/right/image_raw")
        self.center_topic = rospy.get_param(
            "~center_topic", "/yolo_unified/target_center"
        )
        self.bbox_topic = rospy.get_param(
            "~bbox_topic", "/yolo_unified/target_bbox"
        )
        self.line_topic = rospy.get_param(
            "~line_topic", "/yolo_unified/line_points"
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
        self.pending_target = None

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
                queue_size=1,
            )
        elif self.task_mode == "bbox":
            self.target_sub = rospy.Subscriber(
                self.bbox_topic,
                BoundingBox,
                self.bbox_callback,
                queue_size=1,
            )
        else:
            self.target_sub = rospy.Subscriber(
                self.line_topic,
                LinePixels,
                self.line_callback,
                queue_size=1,
            )

        self.target_pub = rospy.Publisher(
            self.target_output_topic, TargetDetection, queue_size=1
        )
        self.line_pub = rospy.Publisher(
            self.line_output_topic, LineDetection, queue_size=1
        )
        self.web_pub = rospy.Publisher(
            self.web_pose_topic, String, queue_size=1
        )

        self.min_disparity = int(rospy.get_param("~min_disparity", 0))
        self.num_disparities = int(rospy.get_param("~num_disparities", 96))
        self.block_size = int(rospy.get_param("~block_size", 7))

        if self.num_disparities <= 0 or self.num_disparities % 16 != 0:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if self.block_size < 3 or self.block_size % 2 == 0:
            raise ValueError("block_size must be an odd integer >= 3")

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
        )

        rospy.loginfo(
            "Depth node initialized: task=%s, max_sync_dt=%.3f",
            self.task_mode,
            self.max_sync_dt,
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

    def set_target(self, target):
        with self.data_lock:
            self.pending_target = target

    def center_callback(self, msg):
        self.set_target(
            {
                "class_name": str(msg.header.frame_id),
                "confidence": float(msg.point.z),
                "stamp": self.valid_stamp(msg.header.stamp),
                "pixels": [(int(msg.point.x), int(msg.point.y))],
                "input_type": "center",
            }
        )

    def bbox_callback(self, msg):
        x1, y1, x2, y2 = int(msg.x1), int(msg.y1), int(msg.x2), int(msg.y2)
        u = int(round((x1 + x2) / 2.0))
        v = int(round((y1 + y2) / 2.0))
        self.set_target(
            {
                "class_name": str(msg.header.frame_id),
                "confidence": float(msg.conf),
                "stamp": self.valid_stamp(msg.header.stamp),
                "pixels": [(u, v)],
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "input_type": "bbox",
            }
        )

    def line_callback(self, msg):
        pixels = [
            (int(round(point.x)), int(round(point.y)))
            for point in msg.points
        ]

        if self.reverse_line_points:
            pixels.reverse()

        self.set_target(
            {
                "class_name": str(msg.class_name),
                "confidence": float(msg.conf),
                "stamp": self.valid_stamp(msg.header.stamp),
                "pixels": pixels,
                "input_type": "line",
            }
        )

    def take_target_and_frame(self):
        with self.data_lock:
            if self.pending_target is None or not self.frame_buffer:
                return None, None, None

            target = self.pending_target
            self.pending_target = None
            frames = list(self.frame_buffer)

        target_sec = target["stamp"].to_sec()
        frame = min(
            frames,
            key=lambda item: abs(item["stamp_sec"] - target_sec),
        )
        sync_dt = abs(frame["stamp_sec"] - target_sec)

        return target, frame, sync_dt

    def compute_depth(self, left, right):
        if left.shape[:2] != right.shape[:2]:
            raise ValueError("left and right image sizes differ")

        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disparity = (
            self.stereo_matcher.compute(gray_left, gray_right).astype(np.float32)
            / 16.0
        )

        depth = np.full(disparity.shape, np.nan, dtype=np.float32)
        valid = disparity > 0.0
        depth[valid] = self.fx * self.baseline / disparity[valid]
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

    @staticmethod
    def make_pixel_message(u, v):
        pixel = Point32()
        pixel.x = float(u)
        pixel.y = float(v)
        pixel.z = 0.0
        return pixel

    @staticmethod
    def make_position_message(point):
        position = Point()
        position.x = float(point[0])
        position.y = float(point[1])
        position.z = float(point[2])
        return position

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
        pixels = list(target["pixels"])
        points = [
            self.pixel_to_3d(u, v, depth)
            for u, v in pixels
        ]
        point_valid = [
            self.point_valid(point)
            for point in points
        ]

        total_count = len(points)
        valid_count = int(sum(point_valid))
        valid_ratio = (
            float(valid_count) / float(total_count)
            if total_count > 0
            else 0.0
        )

        depth_order_valid = True
        if self.require_line_depth_order:
            valid_depths = [
                float(point[2])
                for point, valid in zip(points, point_valid)
                if valid
            ]
            depth_order_valid = all(
                valid_depths[index] <= valid_depths[index + 1]
                for index in range(len(valid_depths) - 1)
            )

        if self.require_all_line_points:
            overall_valid = (
                total_count > 0
                and valid_count == total_count
                and depth_order_valid
            )
        else:
            overall_valid = (
                valid_count >= self.min_valid_line_points
                and valid_ratio >= self.min_valid_line_ratio
                and depth_order_valid
            )

        if total_count == 0:
            reason = "no_line_samples"
        elif not depth_order_valid:
            reason = "line_depth_order_invalid"
        elif valid_count < self.min_valid_line_points:
            reason = "too_few_valid_line_points"
        elif valid_ratio < self.min_valid_line_ratio:
            reason = "valid_line_ratio_too_low"
        elif self.require_all_line_points and valid_count != total_count:
            reason = "not_all_line_points_valid"
        else:
            reason = ""

        # Publish every line result. Per-point validity allows downstream
        # modules to retain valid samples even if a few stereo depths fail.
        msg = LineDetection()
        msg.header.stamp = target["stamp"]
        msg.header.frame_id = "camera"
        msg.type = "line"
        msg.conf = float(target["confidence"])
        msg.class_name = str(target["class_name"])
        msg.valid = bool(overall_valid)
        msg.point_count = int(total_count)
        msg.valid_count = int(valid_count)
        msg.valid_ratio = float(valid_ratio)
        msg.reason = reason

        msg.pixels = [
            self.make_pixel_message(u, v)
            for u, v in pixels
        ]
        msg.positions = [
            self.make_position_message(point)
            for point in points
        ]
        msg.point_valid = [
            bool(valid)
            for valid in point_valid
        ]

        self.line_pub.publish(msg)

        samples = []
        for index, ((u, v), point, valid) in enumerate(
            zip(pixels, points, point_valid)
        ):
            sample = {
                "index": index,
                "pixel": {
                    "u": int(u),
                    "v": int(v),
                },
                "valid": bool(valid),
                "position_m": None,
            }
            if valid:
                sample["position_m"] = {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                }
            samples.append(sample)

        payload = {
            "stamp": target["stamp"].to_sec(),
            "source": "stereo_depth_unified",
            "task_mode": "line",
            "input_type": "line",
            "frame_id": "camera",
            "class_name": target["class_name"],
            "confidence": target["confidence"],
            "valid": bool(overall_valid),
            "point_count": total_count,
            "valid_count": valid_count,
            "valid_ratio": valid_ratio,
            "samples": samples,
        }

        valid_indices = [
            index
            for index, valid in enumerate(point_valid)
            if valid
        ]
        if valid_indices:
            representative_index = valid_indices[
                len(valid_indices) // 2
            ]
            representative = points[representative_index]
            payload["position_m"] = {
                "x": float(representative[0]),
                "y": float(representative[1]),
                "z": float(representative[2]),
            }
            payload["representative_index"] = representative_index

        if reason:
            payload["reason"] = reason

        self.publish_web(payload)

    def run(self):
        while not rospy.is_shutdown():
            target, frame, sync_dt = self.take_target_and_frame()

            if target is None:
                self.rate.sleep()
                continue

            if target["confidence"] < self.conf_thre:
                self.publish_invalid(target, "confidence_too_low")
                self.rate.sleep()
                continue

            if sync_dt > self.max_sync_dt:
                self.publish_invalid(
                    target,
                    "no_synchronized_stereo_frame",
                    sync_dt=sync_dt,
                )
                self.rate.sleep()
                continue

            try:
                depth = self.compute_depth(frame["left"], frame["right"])
            except Exception as exc:
                rospy.logerr_throttle(2.0, "depth computation failed: %s", str(exc))
                self.publish_invalid(target, "depth_computation_failed")
                self.rate.sleep()
                continue

            if self.task_mode in ("center", "bbox"):
                self.process_single(target, depth)
            else:
                self.process_line(target, depth)

            if self.visualization:
                self.show_visualization(frame["left"], depth, target)

            self.rate.sleep()

        cv2.destroyAllWindows()

    def show_visualization(self, left, depth, target):
        image = left.copy()
        pixels = target["pixels"]
        for index, (u, v) in enumerate(pixels):
            cv2.circle(image, (u, v), 4, (0, 255, 0), -1)

            if (
                index == 0
                or index == len(pixels) - 1
                or index == len(pixels) // 2
            ):
                cv2.putText(
                    image,
                    "P{}".format(index + 1),
                    (u + 8, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
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
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

        cv2.imshow("Target", image)
        cv2.imshow("Depth", depth_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    try:
        node = UnifiedStereoDepthNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("depth node failed: %s", str(exc))
        raise