#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

import argparse
import json
import threading

import cv2
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from ultralytics import YOLO

from stereo_depth.msg import BoundingBox


class YOLO26nDetector:
    """Ultralytics YOLO ROS detector with Web-facing image and JSON topics."""

    def __init__(self, param):
        rospy.init_node("yolo26n_detector", anonymous=True)

        self.DetectMode = int(param.detect_mode)
        self.top_k = max(1, int(param.top_k))
        self.visualization = int(param.visualization)
        self.conf_thre = float(param.conf_thre)
        self.detc_type = str(param.detc_type).lower()
        # self.rate = rospy.Rate(float(param.infer_rate))
        self.rate = rospy.Rate(5.0)

        if self.detc_type not in ("center", "bbox"):
            raise ValueError("detc_type must be 'center' or 'bbox'")

        self.bridge = CvBridge()
        self.image_lock = threading.Lock()
        self.left_img = None
        self.left_header = Header()

        model_list = [
            "/home/xhy/catkin_ws/models/shapes0709.pt",
            "/home/xhy/catkin_ws/models/rectangle0710.pt",
            "/home/xhy/catkin_ws/models/line0709.pt",
        ]

        if not 1 <= self.DetectMode <= len(model_list):
            raise ValueError("DetectMode error: {}".format(self.DetectMode))

        model_path = model_list[self.DetectMode - 1]
        self.model = YOLO(model_path)

        mode_names = {
            1: "shapes detection model",
            2: "rectangle detection model",
            3: "line detection model",
        }
        rospy.loginfo("Load %s", mode_names[self.DetectMode])
        rospy.loginfo("Model Path: %s", model_path)

        self.image_sub = rospy.Subscriber(
            "/left/image_raw",
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

        # 保留原有定位接口。
        if self.detc_type == "center":
            self.target_pub = rospy.Publisher(
                "/yolo26n/target_center",
                PointStamped,
                queue_size=1,
            )
        else:
            self.target_pub = rospy.Publisher(
                "/yolo26n/target_bbox",
                BoundingBox,
                queue_size=1,
            )

        # Web端使用的两个标准ROS话题。
        self.annotated_pub = rospy.Publisher(
            "/yolo26n/annotated_image",
            Image,
            queue_size=1,
        )
        self.web_detection_pub = rospy.Publisher(
            "/web/detections",
            String,
            queue_size=1,
        )

        rospy.loginfo("YOLO26n detector initialized")
        rospy.loginfo("Annotated image topic: /yolo26n/annotated_image")
        rospy.loginfo("Web detection topic: /web/detections")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            rospy.logerr_throttle(2.0, "cv_bridge error: %s", str(exc))
            return

        # 回调线程和推理线程之间加锁，避免读取到正在更新的图像。
        with self.image_lock:
            self.left_img = image
            self.left_header = msg.header

    @staticmethod
    def _valid_stamp(header):
        if header.stamp == rospy.Time():
            return rospy.Time.now()
        return header.stamp

    def _publish_target(self, detection, stamp):
        cls_name = detection["class_name"]
        conf = detection["confidence"]
        bbox = detection["bbox"]
        center = detection["center"]

        if self.detc_type == "center":
            point_msg = PointStamped()
            point_msg.header.stamp = stamp
            point_msg.header.frame_id = cls_name
            point_msg.point.x = float(center["u"])
            point_msg.point.y = float(center["v"])
            point_msg.point.z = float(conf)
            self.target_pub.publish(point_msg)

            rospy.loginfo(
                "object %s, conf: %.2f, u: %d, v: %d",
                cls_name,
                conf,
                center["u"],
                center["v"],
            )
        else:
            bbox_msg = BoundingBox()
            bbox_msg.header.stamp = stamp
            bbox_msg.header.frame_id = cls_name
            bbox_msg.x1 = bbox["x1"]
            bbox_msg.y1 = bbox["y1"]
            bbox_msg.x2 = bbox["x2"]
            bbox_msg.y2 = bbox["y2"]
            bbox_msg.conf = float(conf)
            self.target_pub.publish(bbox_msg)

            rospy.loginfo(
                "object %s, conf: %.2f, x1: %d, y1: %d, x2: %d, y2: %d",
                cls_name,
                conf,
                bbox["x1"],
                bbox["y1"],
                bbox["x2"],
                bbox["y2"],
            )

    def _build_detections(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        class_names = result.names

        # 按置信度降序排列，并只保留top_k。
        order = confs.argsort()[::-1]
        detections = []

        for index in order:
            conf = float(confs[index])
            if conf < self.conf_thre:
                continue
            if len(detections) >= self.top_k:
                break

            box = boxes[index]
            cls_id = int(classes[index])
            cls_name = str(class_names[cls_id])

            x1 = int(round(box[0]))
            y1 = int(round(box[1]))
            x2 = int(round(box[2]))
            y2 = int(round(box[3]))
            u = int(round((x1 + x2) / 2.0))
            v = int(round((y1 + y2) / 2.0))

            item = {
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "center": {
                    "u": u,
                    "v": v,
                },
            }

            # 分割模型存在mask时，把轮廓点也写入JSON。
            if result.masks is not None and index < len(result.masks.xy):
                polygon = result.masks.xy[index]
                item["polygon"] = [
                    [round(float(point[0]), 2), round(float(point[1]), 2)]
                    for point in polygon
                ]

            detections.append(item)

        return detections

    def run(self):
        while not rospy.is_shutdown():
            with self.image_lock:
                if self.left_img is None:
                    image = None
                    header = None
                else:
                    image = self.left_img.copy()
                    header = self.left_header

            if image is None:
                self.rate.sleep()
                continue

            try:
                results = self.model(
                    image,
                    conf=self.conf_thre,
                    verbose=False,
                )
            except Exception as exc:
                rospy.logerr_throttle(2.0, "YOLO inference failed: %s", str(exc))
                self.rate.sleep()
                continue

            if not results:
                self.rate.sleep()
                continue

            result = results[0]
            stamp = self._valid_stamp(header)
            detections = self._build_detections(result)

            # 保持原有定位话题发布行为。
            for detection in detections:
                self._publish_target(detection, stamp)

            payload = {
                "stamp": stamp.to_sec(),
                "source": "yolo26n_ultralytics",
                "detect_mode": self.DetectMode,
                "frame_id": header.frame_id,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "count": len(detections),
                "detections": detections,
            }

            self.web_detection_pub.publish(
                String(data=json.dumps(payload, ensure_ascii=False))
            )

            try:
                # 对分割模型会同时绘制mask、边界框、类别和置信度。
                annotated = result.plot()
                annotated_msg = self.bridge.cv2_to_imgmsg(
                    annotated,
                    encoding="bgr8",
                )
                annotated_msg.header = header
                annotated_msg.header.stamp = stamp
                self.annotated_pub.publish(annotated_msg)
            except Exception as exc:
                rospy.logerr_throttle(
                    2.0,
                    "Failed to publish annotated image: %s",
                    str(exc),
                )
                annotated = None

            if self.visualization == 1 and annotated is not None:
                cv2.imshow("YOLO26n Detection", annotated)
                cv2.waitKey(1)

            self.rate.sleep()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26n Detector Node")
    parser.add_argument("--detect_mode", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--visualization", type=int, default=0)
    parser.add_argument("--conf_thre", type=float, default=0.2)
    parser.add_argument("--detc_type", choices=["center", "bbox"], default="center")
    parser.add_argument("--infer_rate", type=float, default=5.0)
    args = parser.parse_args(rospy.myargv()[1:])

    try:
        detector = YOLO26nDetector(param=args)
        detector.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("YOLO26n detector startup failed: %s", str(exc))
        raise