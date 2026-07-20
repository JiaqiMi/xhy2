#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

"""
ROS1 Melodic 二维箭头方向识别节点。

输入：
    /left/image_raw                 sensor_msgs/Image
    /yolo_unified/target_bbox       stereo_depth/BoundingBox

输出：
    /arrow/direction                std_msgs/String，JSON，始终按固定频率发布
    /arrow/angle_deg                std_msgs/Float32，无效时为 NaN
    /arrow/discrete_direction       std_msgs/String，无效时为 "none"
    /arrow/direction_vector         geometry_msgs/Vector3Stamped，无效时为零向量
    /arrow/annotated_image          sensor_msgs/Image，始终按固定频率发布

角度约定：
    图像向右 = 0°
    图像向上 = 90°
    图像向左 = 180°
    图像向下 = 270°
"""

import json
import math
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String

from stereo_depth.msg import BoundingBox


class ArrowDirectionNode:
    def __init__(self):
        rospy.init_node("arrow_direction_node", anonymous=False)

        # -------------------------
        # 话题参数
        # -------------------------
        self.image_topic = rospy.get_param(
            "~image_topic", "/left/image_raw"
        )
        self.bbox_topic = rospy.get_param(
            "~bbox_topic", "/yolo_unified/target_bbox"
        )
        self.annotated_topic = rospy.get_param(
            "~annotated_topic", "/arrow/annotated_image"
        )
        self.direction_topic = rospy.get_param(
            "~direction_topic", "/arrow/direction"
        )
        self.angle_topic = rospy.get_param(
            "~angle_topic", "/arrow/angle_deg"
        )
        self.discrete_topic = rospy.get_param(
            "~discrete_topic", "/arrow/discrete_direction"
        )
        self.vector_topic = rospy.get_param(
            "~vector_topic", "/arrow/direction_vector"
        )

        # -------------------------
        # 算法参数
        # -------------------------
        self.target_class_name = str(
            rospy.get_param("~target_class_name", "arrow")
        ).strip().lower()
        self.publish_rate = max(
            0.5, float(rospy.get_param("~publish_rate", 5.0))
        )
        self.bbox_timeout = max(
            0.05, float(rospy.get_param("~bbox_timeout", 0.6))
        )
        self.max_sync_dt = max(
            0.0, float(rospy.get_param("~max_sync_dt", 0.35))
        )
        self.bbox_margin_ratio = max(
            0.0, float(rospy.get_param("~bbox_margin_ratio", 0.10))
        )
        self.min_bbox_size = max(
            5, int(rospy.get_param("~min_bbox_size", 20))
        )
        self.min_contour_area = max(
            1.0, float(rospy.get_param("~min_contour_area", 80.0))
        )
        self.min_contour_fill_ratio = float(
            rospy.get_param("~min_contour_fill_ratio", 0.015)
        )
        self.max_contour_fill_ratio = float(
            rospy.get_param("~max_contour_fill_ratio", 0.85)
        )
        self.threshold_mode = str(
            rospy.get_param("~threshold_mode", "combined")
        ).strip().lower()
        self.visualization = int(
            rospy.get_param("~visualization", 0)
        )

        if self.threshold_mode not in ("otsu", "adaptive", "combined"):
            raise ValueError(
                "threshold_mode must be otsu, adaptive or combined"
            )

        self.bridge = CvBridge()
        self.lock = threading.Lock()

        self.latest_image = None
        self.latest_image_header = None
        self.latest_image_receive_time = rospy.Time(0)

        self.latest_bbox = None
        self.latest_bbox_receive_time = rospy.Time(0)
        self.latest_bbox_class = ""

        # -------------------------
        # ROS接口
        # -------------------------
        self.image_sub = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )
        self.bbox_sub = rospy.Subscriber(
            self.bbox_topic,
            BoundingBox,
            self.bbox_callback,
            queue_size=1,
        )

        self.annotated_pub = rospy.Publisher(
            self.annotated_topic, Image, queue_size=1
        )
        self.direction_pub = rospy.Publisher(
            self.direction_topic, String, queue_size=1
        )
        self.angle_pub = rospy.Publisher(
            self.angle_topic, Float32, queue_size=1
        )
        self.discrete_pub = rospy.Publisher(
            self.discrete_topic, String, queue_size=1
        )
        self.vector_pub = rospy.Publisher(
            self.vector_topic, Vector3Stamped, queue_size=1
        )

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate),
            self.timer_callback,
        )

        rospy.loginfo("Arrow direction node initialized")
        rospy.loginfo("image_topic=%s", self.image_topic)
        rospy.loginfo("bbox_topic=%s", self.bbox_topic)
        rospy.loginfo("direction_topic=%s", self.direction_topic)
        rospy.loginfo("annotated_topic=%s", self.annotated_topic)
        rospy.loginfo(
            "publish_rate=%.2f Hz, bbox_timeout=%.2f s",
            self.publish_rate,
            self.bbox_timeout,
        )

    # ================================================================
    # ROS回调
    # ================================================================
    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as exc:
            rospy.logerr_throttle(
                2.0, "Arrow image conversion failed: %s", str(exc)
            )
            return

        with self.lock:
            self.latest_image = image
            self.latest_image_header = msg.header
            self.latest_image_receive_time = rospy.Time.now()

    def bbox_callback(self, msg):
        class_name = str(msg.header.frame_id).strip().lower()

        with self.lock:
            self.latest_bbox_receive_time = rospy.Time.now()
            self.latest_bbox_class = class_name

            if class_name == self.target_class_name:
                self.latest_bbox = msg
            else:
                # YOLO发布了其他类别时，立即清除旧箭头框，避免沿用旧结果。
                self.latest_bbox = None

    def timer_callback(self, _event):
        """固定频率发布状态、方向结果和标注图。"""
        now = rospy.Time.now()

        with self.lock:
            image = (
                None
                if self.latest_image is None
                else self.latest_image.copy()
            )
            image_header = self.latest_image_header
            bbox = self.latest_bbox
            bbox_receive_time = self.latest_bbox_receive_time
            bbox_class = self.latest_bbox_class

        if image is None or image_header is None:
            self.publish_invalid_state(
                now=now,
                image=None,
                image_header=None,
                reason="no_image",
                bbox_class=bbox_class,
            )
            return

        annotated = image.copy()

        if bbox is None:
            reason = (
                "non_arrow_class"
                if bbox_class and bbox_class != self.target_class_name
                else "no_arrow_bbox"
            )
            self.draw_invalid_overlay(annotated, reason)
            self.publish_invalid_state(
                now=now,
                image=annotated,
                image_header=image_header,
                reason=reason,
                bbox_class=bbox_class,
            )
            return

        bbox_age = (now - bbox_receive_time).to_sec()
        if bbox_age > self.bbox_timeout:
            self.draw_invalid_overlay(annotated, "arrow_bbox_timeout")
            self.publish_invalid_state(
                now=now,
                image=annotated,
                image_header=image_header,
                reason="arrow_bbox_timeout",
                bbox_class=bbox_class,
                extra={"bbox_age_sec": bbox_age},
            )
            return

        if (
            self.max_sync_dt > 0.0
            and bbox.header.stamp != rospy.Time()
            and image_header.stamp != rospy.Time()
        ):
            sync_dt = abs(
                (bbox.header.stamp - image_header.stamp).to_sec()
            )
            if sync_dt > self.max_sync_dt:
                self.draw_invalid_overlay(
                    annotated, "bbox_image_unsynchronized"
                )
                self.publish_invalid_state(
                    now=now,
                    image=annotated,
                    image_header=image_header,
                    reason="bbox_image_unsynchronized",
                    bbox_class=bbox_class,
                    extra={"sync_dt_sec": sync_dt},
                )
                return

        result, reason = self.process_arrow(image, bbox)

        if result is None:
            self.draw_bbox_if_valid(annotated, bbox, (0, 0, 255))
            self.draw_invalid_overlay(annotated, reason)
            self.publish_invalid_state(
                now=now,
                image=annotated,
                image_header=image_header,
                reason=reason,
                bbox_class=bbox_class,
                bbox=bbox,
            )
            return

        annotated = self.draw_valid_overlay(
            image.copy(), bbox, result
        )
        self.publish_valid_state(
            image=annotated,
            image_header=image_header,
            bbox=bbox,
            result=result,
        )

    # ================================================================
    # 算法主体
    # ================================================================
    def process_arrow(self, image, bbox):
        height, width = image.shape[:2]

        x1 = int(bbox.x1)
        y1 = int(bbox.y1)
        x2 = int(bbox.x2)
        y2 = int(bbox.y2)

        if x2 <= x1 or y2 <= y1:
            return None, "invalid_bbox"

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if (
            bbox_width < self.min_bbox_size
            or bbox_height < self.min_bbox_size
        ):
            return None, "bbox_too_small"

        margin_x = int(round(bbox_width * self.bbox_margin_ratio))
        margin_y = int(round(bbox_height * self.bbox_margin_ratio))

        roi_x1 = max(0, x1 - margin_x)
        roi_y1 = max(0, y1 - margin_y)
        roi_x2 = min(width, x2 + margin_x)
        roi_y2 = min(height, y2 + margin_y)

        if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
            return None, "empty_roi"

        roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
        mask = self.extract_black_mask(roi)
        contour = self.select_main_contour(mask)

        if contour is None:
            return None, "no_valid_contour"

        contour_area = float(cv2.contourArea(contour))
        roi_area = float(roi.shape[0] * roi.shape[1])
        fill_ratio = contour_area / max(roi_area, 1.0)

        if contour_area < self.min_contour_area:
            return None, "contour_too_small"
        if fill_ratio < self.min_contour_fill_ratio:
            return None, "contour_fill_too_low"
        if fill_ratio > self.max_contour_fill_ratio:
            return None, "contour_fill_too_high"

        direction_result = self.estimate_tip_tail(mask, contour)
        if direction_result is None:
            return None, "direction_estimation_failed"

        tip_roi, tail_roi, center_roi, direction_xy = direction_result

        tip = (
            int(round(tip_roi[0] + roi_x1)),
            int(round(tip_roi[1] + roi_y1)),
        )
        tail = (
            int(round(tail_roi[0] + roi_x1)),
            int(round(tail_roi[1] + roi_y1)),
        )
        center = (
            int(round(center_roi[0] + roi_x1)),
            int(round(center_roi[1] + roi_y1)),
        )

        dx = float(direction_xy[0])
        dy = float(direction_xy[1])
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return None, "zero_direction_vector"

        dir_x = dx / norm
        dir_y = dy / norm

        angle_rad = math.atan2(-dir_y, dir_x)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0.0:
            angle_deg += 360.0

        result = {
            "tip": tip,
            "tail": tail,
            "center": center,
            "direction_x": dir_x,
            "direction_y": dir_y,
            "angle_rad": angle_rad,
            "angle_deg": angle_deg,
            "discrete_direction": self.angle_to_direction(angle_deg),
            "contour": contour,
            "contour_offset": (roi_x1, roi_y1),
            "contour_area": contour_area,
            "contour_fill_ratio": fill_ratio,
        }
        return result, ""

    def extract_black_mask(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, otsu = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )

        if self.threshold_mode == "otsu":
            mask = otsu
        elif self.threshold_mode == "adaptive":
            mask = adaptive
        else:
            mask = cv2.bitwise_and(otsu, adaptive)

        kernel3 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )
        kernel5 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, kernel3, iterations=1
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, kernel5, iterations=2
        )
        return mask

    @staticmethod
    def select_main_contour(mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def estimate_tip_tail(mask, contour):
        """
        通过轮廓PCA得到主轴，再比较主轴两端的截面宽度变化判断箭头尖端。

        箭头尖端附近的截面通常在极端位置接近0，并向内部快速增宽；
        箭尾端附近的截面通常保持近似恒定的杆宽。
        """
        points_yx = np.column_stack(np.where(mask > 0))
        if points_yx.shape[0] < 20:
            return None

        points_xy = points_yx[:, [1, 0]].astype(np.float32)
        center = np.mean(points_xy, axis=0)
        centered = points_xy - center

        covariance = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        perpendicular = np.array([-axis[1], axis[0]], dtype=np.float32)

        longitudinal = centered @ axis
        lateral = centered @ perpendicular
        t_min = float(np.min(longitudinal))
        t_max = float(np.max(longitudinal))
        span = t_max - t_min
        if span < 5.0:
            return None

        def slice_width(start_ratio, end_ratio, from_min):
            if from_min:
                low = t_min + start_ratio * span
                high = t_min + end_ratio * span
            else:
                low = t_max - end_ratio * span
                high = t_max - start_ratio * span

            selected = lateral[
                (longitudinal >= low) & (longitudinal <= high)
            ]
            if selected.size < 3:
                return 0.0
            return float(np.percentile(selected, 95) - np.percentile(selected, 5))

        # 极端薄层宽度与稍靠内层宽度。
        min_outer = slice_width(0.00, 0.035, True)
        min_inner = slice_width(0.07, 0.15, True)
        max_outer = slice_width(0.00, 0.035, False)
        max_inner = slice_width(0.07, 0.15, False)

        # 尖端：外层窄、向内增宽快。
        min_tip_score = (
            (min_inner - min_outer)
            + 0.5 * min_inner
            - 0.8 * min_outer
        )
        max_tip_score = (
            (max_inner - max_outer)
            + 0.5 * max_inner
            - 0.8 * max_outer
        )

        if max_tip_score >= min_tip_score:
            direction = axis
            tip_projection = t_max
            tail_side_is_min = True
        else:
            direction = -axis
            tip_projection = -t_min
            tail_side_is_min = False

        # 在选定方向坐标中重新投影，保证tip位于最大投影端。
        directed_longitudinal = centered @ direction
        d_min = float(np.min(directed_longitudinal))
        d_max = float(np.max(directed_longitudinal))
        d_span = d_max - d_min

        contour_points = contour[:, 0, :].astype(np.float32)
        contour_centered = contour_points - center
        contour_projection = contour_centered @ direction
        tip = contour_points[int(np.argmax(contour_projection))]

        tail_threshold = d_min + 0.08 * d_span
        tail_candidates = points_xy[
            directed_longitudinal <= tail_threshold
        ]
        if tail_candidates.shape[0] < 3:
            tail = center - direction * (0.45 * d_span)
        else:
            tail = np.mean(tail_candidates, axis=0)

        # 最终方向由tail指向tip，避免PCA数值误差。
        final_direction = tip - tail
        final_norm = np.linalg.norm(final_direction)
        if final_norm < 1e-6:
            return None
        final_direction = final_direction / final_norm

        return tip, tail, center, final_direction

    # ================================================================
    # 发布与绘图
    # ================================================================
    def publish_valid_state(self, image, image_header, bbox, result):
        stamp = self.valid_stamp(image_header.stamp)
        frame_id = image_header.frame_id or "camera"

        payload = {
            "stamp": stamp.to_sec(),
            "source": "arrow_direction",
            "valid": True,
            "reason": "",
            "class_name": self.target_class_name,
            "confidence": float(bbox.conf),
            "bbox": {
                "x1": int(bbox.x1),
                "y1": int(bbox.y1),
                "x2": int(bbox.x2),
                "y2": int(bbox.y2),
            },
            "tip": {
                "u": int(result["tip"][0]),
                "v": int(result["tip"][1]),
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
            "discrete_direction": result["discrete_direction"],
            "contour_area": float(result["contour_area"]),
            "contour_fill_ratio": float(
                result["contour_fill_ratio"]
            ),
        }

        self.direction_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False))
        )
        self.angle_pub.publish(
            Float32(data=float(result["angle_deg"]))
        )
        self.discrete_pub.publish(
            String(data=result["discrete_direction"])
        )

        vector_msg = Vector3Stamped()
        vector_msg.header.stamp = stamp
        vector_msg.header.frame_id = frame_id
        vector_msg.vector.x = float(result["direction_x"])
        vector_msg.vector.y = float(result["direction_y"])
        vector_msg.vector.z = 0.0
        self.vector_pub.publish(vector_msg)

        self.publish_annotated_image(image, image_header, stamp)

        rospy.loginfo_throttle(
            1.0,
            "Arrow valid: direction=%s, angle=%.1f deg",
            result["discrete_direction"],
            result["angle_deg"],
        )

    def publish_invalid_state(
        self,
        now,
        image,
        image_header,
        reason,
        bbox_class="",
        bbox=None,
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
            "source": "arrow_direction",
            "valid": False,
            "reason": reason,
            "class_name": self.target_class_name,
            "observed_class_name": bbox_class,
            "confidence": 0.0,
            "bbox": None,
            "tip": None,
            "tail": None,
            "center": None,
            "direction_2d": None,
            "angle_rad": None,
            "angle_deg": None,
            "discrete_direction": "none",
        }

        if bbox is not None:
            payload["confidence"] = float(bbox.conf)
            payload["bbox"] = {
                "x1": int(bbox.x1),
                "y1": int(bbox.y1),
                "x2": int(bbox.x2),
                "y2": int(bbox.y2),
            }
        if extra:
            payload.update(extra)

        self.direction_pub.publish(
            String(data=json.dumps(payload, ensure_ascii=False))
        )
        self.angle_pub.publish(Float32(data=float("nan")))
        self.discrete_pub.publish(String(data="none"))

        vector_msg = Vector3Stamped()
        vector_msg.header.stamp = stamp
        vector_msg.header.frame_id = frame_id
        vector_msg.vector.x = 0.0
        vector_msg.vector.y = 0.0
        vector_msg.vector.z = 0.0
        self.vector_pub.publish(vector_msg)

        if image is not None and image_header is not None:
            self.publish_annotated_image(
                image, image_header, stamp
            )

        rospy.loginfo_throttle(
            2.0, "Arrow invalid: %s", reason
        )

    def publish_annotated_image(self, image, header, stamp):
        try:
            image_msg = self.bridge.cv2_to_imgmsg(
                image, encoding="bgr8"
            )
            image_msg.header = header
            image_msg.header.stamp = stamp
            self.annotated_pub.publish(image_msg)
        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "Arrow annotated image publish failed: %s",
                str(exc),
            )

        if self.visualization == 1:
            cv2.imshow("Arrow Direction", image)
            cv2.waitKey(1)

    def draw_valid_overlay(self, image, bbox, result):
        self.draw_bbox_if_valid(image, bbox, (255, 0, 0))

        contour = result["contour"].copy()
        offset_x, offset_y = result["contour_offset"]
        contour[:, 0, 0] += offset_x
        contour[:, 0, 1] += offset_y
        cv2.drawContours(
            image, [contour], -1, (0, 255, 255), 2
        )

        tip = result["tip"]
        tail = result["tail"]
        center = result["center"]

        cv2.circle(image, tip, 7, (0, 0, 255), -1)
        cv2.circle(image, tail, 7, (255, 0, 0), -1)
        cv2.circle(image, center, 5, (0, 255, 255), -1)
        cv2.arrowedLine(
            image,
            tail,
            tip,
            (0, 255, 0),
            3,
            cv2.LINE_AA,
            tipLength=0.20,
        )

        x1 = max(10, int(bbox.x1))
        y1 = max(30, int(bbox.y1) - 35)
        cv2.putText(
            image,
            "Arrow: {}".format(result["discrete_direction"]),
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "Angle: {:.1f} deg".format(result["angle_deg"]),
            (x1, y1 + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return image

    @staticmethod
    def draw_invalid_overlay(image, reason):
        cv2.rectangle(image, (8, 8), (470, 52), (0, 0, 0), -1)
        cv2.putText(
            image,
            "Arrow: NOT VALID ({})".format(reason),
            (16, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def draw_bbox_if_valid(image, bbox, color):
        if bbox is None:
            return
        height, width = image.shape[:2]
        x1 = max(0, min(width - 1, int(bbox.x1)))
        y1 = max(0, min(height - 1, int(bbox.y1)))
        x2 = max(0, min(width - 1, int(bbox.x2)))
        y2 = max(0, min(height - 1, int(bbox.y2)))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    @staticmethod
    def valid_stamp(stamp):
        if stamp == rospy.Time():
            return rospy.Time.now()
        return stamp

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
        ArrowDirectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()