#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

import os
import time
from urllib.parse import quote

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class RtspFisheyeDriver:
    """读取 RTSP 原始画面并发布标准 ROS Image。"""

    def __init__(self):
        self.host = str(rospy.get_param("~host", "192.168.1.122")).strip()
        self.port = int(rospy.get_param("~port", 554))
        self.username = str(rospy.get_param("~username", "admin"))
        self.password = str(rospy.get_param("~password", "admin"))
        self.channel = int(rospy.get_param("~channel", 1))
        self.subtype = int(rospy.get_param("~subtype", 0))
        self.rtsp_path = str(
            rospy.get_param(
                "~rtsp_path",
                "/cam/realmonitor?channel={channel}&subtype={subtype}",
            )
        )
        self.direct_rtsp_url = str(rospy.get_param("~rtsp_url", "")).strip()
        self.transport = str(rospy.get_param("~transport", "tcp")).lower()

        self.image_topic = str(
            rospy.get_param("~image_topic", "/fisheye_camera/image_raw")
        )
        self.frame_id = str(rospy.get_param("~frame_id", "fisheye_camera"))
        self.queue_size = max(1, int(rospy.get_param("~queue_size", 1)))
        self.output_width = max(0, int(rospy.get_param("~output_width", 0)))
        self.output_height = max(0, int(rospy.get_param("~output_height", 0)))

        self.reconnect_delay = max(
            0.1, float(rospy.get_param("~reconnect_delay", 2.0))
        )
        self.failure_threshold = max(
            1, int(rospy.get_param("~failure_threshold", 3))
        )
        self.log_interval = max(
            1.0, float(rospy.get_param("~log_interval", 5.0))
        )

        if not self.host and not self.direct_rtsp_url:
            raise ValueError("参数 ~host 和 ~rtsp_url 不能同时为空")
        if not 1 <= self.port <= 65535:
            raise ValueError("参数 ~port 必须在 1 到 65535 之间")
        if self.transport not in ("tcp", "udp"):
            raise ValueError("参数 ~transport 仅支持 tcp 或 udp")

        self.bridge = CvBridge()
        self.publisher = rospy.Publisher(
            self.image_topic,
            Image,
            queue_size=self.queue_size,
        )
        self.capture = None
        self.rtsp_url = self._build_rtsp_url()
        self.sequence = 0
        self.frame_count = 0
        self.fps_start_time = time.monotonic()

        rospy.on_shutdown(self.close)
        rospy.loginfo("RTSP 鱼眼相机驱动已初始化")
        rospy.loginfo("RTSP 地址: %s", self._masked_url())
        rospy.loginfo("发布话题: %s, frame_id: %s", self.image_topic, self.frame_id)

    def _build_rtsp_url(self):
        if self.direct_rtsp_url:
            return self.direct_rtsp_url

        path = self.rtsp_path.format(
            channel=self.channel,
            subtype=self.subtype,
        )
        if not path.startswith("/"):
            path = "/" + path

        auth = ""
        if self.username:
            user = quote(self.username, safe="")
            password = quote(self.password, safe="")
            auth = "{}:{}@".format(user, password)

        return "rtsp://{}{}:{}{}".format(auth, self.host, self.port, path)

    def _masked_url(self):
        if "@" not in self.rtsp_url:
            return self.rtsp_url
        scheme, remainder = self.rtsp_url.split("://", 1)
        return "{}://***@{}".format(scheme, remainder.split("@", 1)[1])

    def _open_capture(self):
        self.close()

        # OpenCV 的 FFmpeg 后端在打开流时读取该环境变量。
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;{}".format(self.transport)
        )

        backend = getattr(cv2, "CAP_FFMPEG", getattr(cv2, "CAP_ANY", 0))
        try:
            capture = cv2.VideoCapture(self.rtsp_url, backend)
        except TypeError:
            # 兼容不支持 apiPreference 参数的旧版 OpenCV Python 绑定。
            capture = cv2.VideoCapture(self.rtsp_url)

        buffer_property = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        if buffer_property is not None:
            capture.set(buffer_property, 1)

        if not capture.isOpened():
            capture.release()
            rospy.logwarn("无法打开 RTSP 视频流，%.1f 秒后重试", self.reconnect_delay)
            return False

        self.capture = capture
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        rospy.loginfo(
            "RTSP 视频流已连接: %dx%d, 源帧率 %.2f",
            width,
            height,
            source_fps,
        )
        return True

    def _resize_frame(self, frame):
        if self.output_width <= 0 and self.output_height <= 0:
            return frame

        height, width = frame.shape[:2]
        if self.output_width > 0 and self.output_height > 0:
            target_size = (self.output_width, self.output_height)
        elif self.output_width > 0:
            target_height = max(1, int(height * self.output_width / float(width)))
            target_size = (self.output_width, target_height)
        else:
            target_width = max(1, int(width * self.output_height / float(height)))
            target_size = (target_width, self.output_height)

        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

    def _publish_frame(self, frame):
        frame = self._resize_frame(frame)
        message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        message.header.seq = self.sequence
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.frame_id
        self.publisher.publish(message)

        self.sequence = (self.sequence + 1) % (2 ** 32)
        self.frame_count += 1
        now = time.monotonic()
        elapsed = now - self.fps_start_time
        if elapsed >= self.log_interval:
            rospy.loginfo(
                "正在发布 %s: %.2f FPS, %dx%d",
                self.image_topic,
                self.frame_count / elapsed,
                frame.shape[1],
                frame.shape[0],
            )
            self.frame_count = 0
            self.fps_start_time = now

    def run(self):
        consecutive_failures = 0

        while not rospy.is_shutdown():
            if self.capture is None and not self._open_capture():
                rospy.sleep(self.reconnect_delay)
                continue

            ok, frame = self.capture.read()
            if not ok or frame is None or frame.size == 0:
                consecutive_failures += 1
                rospy.logwarn_throttle(
                    2.0,
                    "RTSP 读取失败 (%d/%d)",
                    consecutive_failures,
                    self.failure_threshold,
                )
                if consecutive_failures >= self.failure_threshold:
                    rospy.logwarn("RTSP 连续读取失败，准备重新连接")
                    self.close()
                    consecutive_failures = 0
                    rospy.sleep(self.reconnect_delay)
                continue

            consecutive_failures = 0
            try:
                self._publish_frame(frame)
            except Exception as exc:
                rospy.logerr_throttle(2.0, "发布图像失败: %s", str(exc))

    def close(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def main():
    rospy.init_node("rtsp_fisheye_driver", anonymous=False)
    try:
        RtspFisheyeDriver().run()
    except (ValueError, KeyError) as exc:
        rospy.logfatal("RTSP 鱼眼相机参数错误: %s", str(exc))
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
