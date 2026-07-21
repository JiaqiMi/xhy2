#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

import traceback
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image


class SafeStereoSplitter:
    def __init__(self):
        rospy.init_node("split_stereo_image", anonymous=False)

        self.input_topic = rospy.get_param("~input_topic", "/usb_cam/image_raw")
        self.left_topic = rospy.get_param("~left_topic", "/left/image_raw")
        self.right_topic = rospy.get_param("~right_topic", "/right/image_raw")
        self.frame_id = rospy.get_param("~frame_id", "camera")

        self.left_pub = rospy.Publisher(self.left_topic, Image, queue_size=1)
        self.right_pub = rospy.Publisher(self.right_topic, Image, queue_size=1)

        self.sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 26,
        )

        rospy.loginfo("Safe stereo splitter initialized")
        rospy.loginfo("input=%s", self.input_topic)
        rospy.loginfo("left=%s", self.left_topic)
        rospy.loginfo("right=%s", self.right_topic)

    @staticmethod
    def _rows_from_buffer(msg, dtype):
        itemsize = np.dtype(dtype).itemsize
        if msg.step <= 0:
            raise ValueError("invalid image step: {}".format(msg.step))
        if msg.step % itemsize != 0:
            raise ValueError(
                "step={} is not divisible by itemsize={}".format(
                    msg.step, itemsize
                )
            )

        row_elements = msg.step // itemsize
        expected = int(msg.height) * int(row_elements)
        data = np.frombuffer(msg.data, dtype=dtype)

        if data.size < expected:
            raise ValueError(
                "image data too short: got {}, expected {}".format(
                    data.size, expected
                )
            )

        return data[:expected].reshape(int(msg.height), int(row_elements))

    def decode_to_bgr8(self, msg):
        encoding = str(msg.encoding).strip().lower()
        width = int(msg.width)
        height = int(msg.height)

        if width <= 0 or height <= 0:
            raise ValueError("invalid image size: {}x{}".format(width, height))

        if encoding in ("bgr8", "8uc3"):
            rows = self._rows_from_buffer(msg, np.uint8)
            required = width * 3
            return np.ascontiguousarray(
                rows[:, :required].reshape(height, width, 3)
            )

        if encoding == "rgb8":
            rows = self._rows_from_buffer(msg, np.uint8)
            required = width * 3
            rgb = rows[:, :required].reshape(height, width, 3)
            return cv2.cvtColor(
                np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR
            )

        if encoding in ("bgra8", "8uc4"):
            rows = self._rows_from_buffer(msg, np.uint8)
            required = width * 4
            bgra = rows[:, :required].reshape(height, width, 4)
            return cv2.cvtColor(
                np.ascontiguousarray(bgra), cv2.COLOR_BGRA2BGR
            )

        if encoding == "rgba8":
            rows = self._rows_from_buffer(msg, np.uint8)
            required = width * 4
            rgba = rows[:, :required].reshape(height, width, 4)
            return cv2.cvtColor(
                np.ascontiguousarray(rgba), cv2.COLOR_RGBA2BGR
            )

        if encoding in ("mono8", "8uc1"):
            rows = self._rows_from_buffer(msg, np.uint8)
            gray = rows[:, :width]
            return cv2.cvtColor(
                np.ascontiguousarray(gray), cv2.COLOR_GRAY2BGR
            )

        if encoding in ("mono16", "16uc1"):
            dtype = np.dtype(">u2") if bool(msg.is_bigendian) else np.dtype("<u2")
            rows = self._rows_from_buffer(msg, dtype)
            gray16 = rows[:, :width].astype(np.float32)

            minimum = float(np.min(gray16))
            maximum = float(np.max(gray16))
            if maximum > minimum:
                gray8 = np.clip(
                    (gray16 - minimum) * (255.0 / (maximum - minimum)),
                    0,
                    255,
                ).astype(np.uint8)
            else:
                gray8 = np.zeros((height, width), dtype=np.uint8)

            return cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

        if encoding in ("yuv422", "yuyv", "yuy2"):
            rows = self._rows_from_buffer(msg, np.uint8)
            required = width * 2
            yuv = rows[:, :required].reshape(height, width, 2)
            return cv2.cvtColor(
                np.ascontiguousarray(yuv), cv2.COLOR_YUV2BGR_YUY2
            )

        if encoding == "uyvy":
            rows = self._rows_from_buffer(msg, np.uint8)
            required = width * 2
            yuv = rows[:, :required].reshape(height, width, 2)
            return cv2.cvtColor(
                np.ascontiguousarray(yuv), cv2.COLOR_YUV2BGR_UYVY
            )

        raise ValueError("unsupported input encoding: {!r}".format(msg.encoding))

    @staticmethod
    def make_bgr8_message(image, source_header, frame_id):
        image = np.ascontiguousarray(image, dtype=np.uint8)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("output image must be HxWx3, got {}".format(image.shape))

        output = Image()
        output.header = source_header
        if output.header.stamp == rospy.Time():
            output.header.stamp = rospy.Time.now()
        if frame_id:
            output.header.frame_id = frame_id

        output.height = int(image.shape[0])
        output.width = int(image.shape[1])
        output.encoding = "bgr8"
        output.is_bigendian = 0
        output.step = int(image.shape[1] * 3)
        output.data = image.tobytes()
        return output

    def image_callback(self, msg):
        try:
            image = self.decode_to_bgr8(msg)

            height, width = image.shape[:2]
            if width < 2:
                raise ValueError("input image width too small: {}".format(width))

            middle = width // 2
            left = image[:, :middle]
            right = image[:, middle:]

            left_msg = self.make_bgr8_message(left, msg.header, self.frame_id)
            right_msg = self.make_bgr8_message(right, msg.header, self.frame_id)

            self.left_pub.publish(left_msg)
            self.right_pub.publish(right_msg)

            rospy.loginfo_throttle(
                5.0,
                "Split image: input=%dx%d encoding=%s -> left=%dx%d right=%dx%d",
                width,
                height,
                msg.encoding,
                left.shape[1],
                left.shape[0],
                right.shape[1],
                right.shape[0],
            )

        except Exception as exc:
            rospy.logerr_throttle(
                2.0,
                "Image split failed: type=%s, repr=%r, encoding=%r, size=%dx%d, step=%d",
                type(exc).__name__,
                exc,
                msg.encoding,
                msg.width,
                msg.height,
                msg.step,
            )
            rospy.logdebug(traceback.format_exc())


if __name__ == "__main__":
    SafeStereoSplitter()
    rospy.spin()