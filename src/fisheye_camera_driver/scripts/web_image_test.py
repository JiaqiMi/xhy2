#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

import logging
import threading
import time

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from flask import Flask, Response, jsonify
from sensor_msgs.msg import Image


PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>鱼眼相机画面测试</title>
  <style>
    body { margin: 0; background: #111827; color: #e5e7eb;
           font-family: Arial, "Microsoft YaHei", sans-serif; }
    main { max-width: 1200px; margin: auto; padding: 20px; }
    .status { margin-bottom: 12px; color: #93c5fd; }
    img { display: block; width: 100%; min-height: 240px;
          background: #000; object-fit: contain; border-radius: 8px; }
  </style>
</head>
<body>
  <main>
    <h1>鱼眼相机原始画面</h1>
    <div id="status" class="status">等待 ROS 图像...</div>
    <img src="/video_feed" alt="鱼眼相机画面">
  </main>
  <script>
    async function updateStatus() {
      try {
        const response = await fetch('/health', {cache: 'no-store'});
        const data = await response.json();
        document.getElementById('status').textContent = data.image_online
          ? `在线 | ${data.width}x${data.height} | ${data.fps.toFixed(1)} FPS`
          : '等待 ROS 图像...';
      } catch (_) {
        document.getElementById('status').textContent = 'Web 服务状态读取失败';
      }
    }
    setInterval(updateStatus, 1000);
    updateStatus();
  </script>
</body>
</html>
"""


class WebImageTest:
    """订阅 ROS Image，并以 MJPEG 网页显示。"""

    def __init__(self):
        self.host = str(rospy.get_param("~host", "0.0.0.0"))
        self.port = int(rospy.get_param("~port", 8081))
        self.image_topic = str(
            rospy.get_param("~image_topic", "/fisheye_camera/image_raw")
        )
        self.stream_fps = max(0.5, float(rospy.get_param("~stream_fps", 10.0)))
        self.jpeg_quality = min(
            100, max(1, int(rospy.get_param("~jpeg_quality", 80)))
        )
        self.web_width = max(0, int(rospy.get_param("~web_width", 0)))
        self.image_timeout = max(
            0.1, float(rospy.get_param("~image_timeout", 3.0))
        )

        if not 1 <= self.port <= 65535:
            raise ValueError("参数 ~port 必须在 1 到 65535 之间")

        self.bridge = CvBridge()
        self.condition = threading.Condition()
        self.latest_jpeg = None
        self.frame_sequence = 0
        self.last_frame_time = 0.0
        self.last_callback_time = 0.0
        self.measured_fps = 0.0
        self.width = 0
        self.height = 0

        self.subscriber = rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/video_feed", "video_feed", self.video_feed)
        self.app.add_url_rule("/health", "health", self.health)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        rospy.loginfo("Web 图像测试节点已初始化，订阅: %s", self.image_topic)

    def image_callback(self, message):
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(2.0, "ROS 图像转换失败: %s", str(exc))
            return

        if self.web_width > 0 and frame.shape[1] > self.web_width:
            scale = self.web_width / float(frame.shape[1])
            target_height = max(1, int(frame.shape[0] * scale))
            frame = cv2.resize(
                frame,
                (self.web_width, target_height),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            rospy.logwarn_throttle(2.0, "JPEG 编码失败")
            return

        now = time.monotonic()
        with self.condition:
            if self.last_callback_time > 0:
                interval = now - self.last_callback_time
                if interval > 0:
                    current_fps = 1.0 / interval
                    self.measured_fps = (
                        current_fps
                        if self.measured_fps <= 0
                        else 0.9 * self.measured_fps + 0.1 * current_fps
                    )
            self.last_callback_time = now
            self.last_frame_time = now
            self.latest_jpeg = encoded.tobytes()
            self.width = frame.shape[1]
            self.height = frame.shape[0]
            self.frame_sequence += 1
            self.condition.notify_all()

    def index(self):
        return Response(PAGE_HTML, mimetype="text/html")

    def mjpeg_generator(self):
        period = 1.0 / self.stream_fps
        last_sequence = -1

        while not rospy.is_shutdown():
            with self.condition:
                self.condition.wait_for(
                    lambda: (
                        self.frame_sequence != last_sequence
                        or rospy.is_shutdown()
                    ),
                    timeout=1.0,
                )
                jpeg = self.latest_jpeg
                last_sequence = self.frame_sequence

            if jpeg is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            time.sleep(period)

    def video_feed(self):
        return Response(
            self.mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def health(self):
        with self.condition:
            frame_age = (
                time.monotonic() - self.last_frame_time
                if self.last_frame_time > 0
                else None
            )
            return jsonify(
                {
                    "ok": True,
                    "image_online": (
                        frame_age is not None and frame_age < self.image_timeout
                    ),
                    "image_age_sec": frame_age,
                    "fps": self.measured_fps,
                    "width": self.width,
                    "height": self.height,
                    "image_topic": self.image_topic,
                }
            )

    def run(self):
        rospy.loginfo("浏览器访问: http://设备IP:%d", self.port)
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )


def main():
    rospy.init_node("fisheye_web_image_test", anonymous=False)
    try:
        WebImageTest().run()
    except ValueError as exc:
        rospy.logfatal("Web 测试参数错误: %s", str(exc))
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
