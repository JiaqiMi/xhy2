#!/home/xhy/xhy_env/bin/python3.8
# -*- coding: utf-8 -*-

import copy
import json
import logging
import threading
import time

import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from flask import Flask, Response, jsonify
from sensor_msgs.msg import Image
from std_msgs.msg import String


PAGE_HTML = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AUV视觉识别系统</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            padding: 20px;
            background: #111827;
            color: #e5e7eb;
            font-family: Arial, "Microsoft YaHei", sans-serif;
        }
        h1 { margin-top: 0; font-size: 24px; }
        h2 { margin-top: 0; font-size: 18px; }
        .status-row {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .badge {
            padding: 7px 12px;
            border-radius: 6px;
            background: #374151;
        }
        .online { background: #065f46; }
        .offline { background: #991b1b; }
        .layout {
            display: grid;
            grid-template-columns: minmax(0, 2fr) minmax(340px, 1fr);
            gap: 18px;
        }
        .panel {
            background: #1f2937;
            border-radius: 10px;
            padding: 14px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        }
        .panel + .panel { margin-top: 18px; }
        .video-container {
            width: 100%;
            background: #000;
            border-radius: 8px;
            overflow: hidden;
        }
        #video-stream {
            display: block;
            width: 100%;
            min-height: 320px;
            object-fit: contain;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        th, td {
            padding: 8px;
            border-bottom: 1px solid #374151;
            text-align: left;
        }
        th { color: #93c5fd; }
        .value {
            color: #fbbf24;
            font-family: monospace;
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }
        .item {
            background: #111827;
            padding: 9px;
            border-radius: 6px;
            min-height: 40px;
        }
        .muted { color: #9ca3af; }
        .valid { color: #34d399; }
        .invalid { color: #f87171; }
        @media (max-width: 900px) {
            .layout { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
<h1>AUV视觉识别与箭头方向估计</h1>

<div class="status-row">
    <div id="connection-status" class="badge offline">Web服务：等待数据</div>
    <div class="badge">图像FPS：<span id="fps" class="value">0.0</span></div>
    <div class="badge">目标数量：<span id="target-count" class="value">0</span></div>
    <div class="badge">更新时间：<span id="update-time" class="value">--</span></div>
</div>

<div class="layout">
    <div class="panel">
        <h2>箭头方向标注画面</h2>
        <div class="video-container">
            <img id="video-stream" src="/video_feed" alt="等待视频流">
        </div>
    </div>

    <div>
        <div class="panel">
            <h2>箭头二维方向</h2>
            <div id="arrow-source" class="muted">暂无箭头方向数据</div>
            <div class="grid" style="margin-top:10px;">
                <div class="item">状态：<span id="arrow-valid" class="value">--</span></div>
                <div class="item">方向：<span id="arrow-direction" class="value">--</span></div>
                <div class="item">角度：<span id="arrow-angle" class="value">--</span> °</div>
                <div class="item">置信度：<span id="arrow-confidence" class="value">--</span></div>
                <div class="item">方向向量X：<span id="arrow-vector-x" class="value">--</span></div>
                <div class="item">方向向量Y：<span id="arrow-vector-y" class="value">--</span></div>
                <div class="item">尖端像素：<span id="arrow-tip" class="value">--</span></div>
                <div class="item">尾部像素：<span id="arrow-tail" class="value">--</span></div>
            </div>
            <div id="arrow-reason" class="muted" style="margin-top:10px;">--</div>
        </div>

        <div class="panel">
            <h2>三维位置</h2>
            <div id="pose-source" class="muted">暂无位置数据</div>
            <div class="grid" style="margin-top:10px;">
                <div class="item">X：<span id="pose-x" class="value">--</span> m</div>
                <div class="item">Y：<span id="pose-y" class="value">--</span> m</div>
                <div class="item">Z：<span id="pose-z" class="value">--</span> m</div>
                <div class="item">状态：<span id="pose-valid" class="value">--</span></div>
            </div>
        </div>

        <div class="panel">
            <h2>YOLO识别结果</h2>
            <table>
                <thead>
                    <tr><th>类别</th><th>置信度</th><th>中心像素</th></tr>
                </thead>
                <tbody id="detections-body">
                    <tr><td colspan="3" class="muted">暂无识别结果</td></tr>
                </tbody>
            </table>
        </div>
    </div>
</div>

<script>
function numberText(value, digits = 3) {
    if (value === undefined || value === null || Number.isNaN(Number(value))) {
        return "--";
    }
    return Number(value).toFixed(digits);
}

function pointText(point) {
    if (!point) return "--";
    const u = point.u !== undefined ? point.u : point.x;
    const v = point.v !== undefined ? point.v : point.y;
    return `(${numberText(u, 0)}, ${numberText(v, 0)})`;
}

function updateDetections(payload) {
    const body = document.getElementById("detections-body");
    const detections = payload && Array.isArray(payload.detections)
        ? payload.detections : [];

    document.getElementById("target-count").textContent = detections.length;

    if (detections.length === 0) {
        body.innerHTML = '<tr><td colspan="3" class="muted">当前未检测到目标</td></tr>';
        return;
    }

    body.innerHTML = detections.map(det => {
        const center = det.center || det.pixel_center || {};
        return `
            <tr>
                <td>${det.class_name || "--"}</td>
                <td>${numberText(det.confidence, 3)}</td>
                <td>(${numberText(center.u, 0)}, ${numberText(center.v, 0)})</td>
            </tr>`;
    }).join("");
}

function updatePose(payload) {
    const position = payload && payload.position_m ? payload.position_m : null;
    if (!position) {
        document.getElementById("pose-source").textContent = "暂无位置数据";
        document.getElementById("pose-x").textContent = "--";
        document.getElementById("pose-y").textContent = "--";
        document.getElementById("pose-z").textContent = "--";
        document.getElementById("pose-valid").textContent = "--";
        return;
    }

    const source = payload.source || "unknown";
    const className = payload.class_name || "";
    document.getElementById("pose-source").textContent = `来源：${source} ${className}`;
    document.getElementById("pose-x").textContent = numberText(position.x);
    document.getElementById("pose-y").textContent = numberText(position.y);
    document.getElementById("pose-z").textContent = numberText(position.z);
    document.getElementById("pose-valid").textContent = payload.valid === false ? "无效" : "有效";
}

function updateArrow(payload) {
    const validElement = document.getElementById("arrow-valid");

    if (!payload) {
        document.getElementById("arrow-source").textContent = "暂无箭头方向数据";
        validElement.textContent = "--";
        validElement.className = "value";
        document.getElementById("arrow-direction").textContent = "--";
        document.getElementById("arrow-angle").textContent = "--";
        document.getElementById("arrow-confidence").textContent = "--";
        document.getElementById("arrow-vector-x").textContent = "--";
        document.getElementById("arrow-vector-y").textContent = "--";
        document.getElementById("arrow-tip").textContent = "--";
        document.getElementById("arrow-tail").textContent = "--";
        document.getElementById("arrow-reason").textContent = "--";
        return;
    }

    const source = payload.source || "arrow_direction";
    document.getElementById("arrow-source").textContent = `来源：${source}`;

    const isValid = payload.valid === true;
    validElement.textContent = isValid ? "有效" : "无效";
    validElement.className = isValid ? "value valid" : "value invalid";

    const vector = payload.direction_2d || {};
    document.getElementById("arrow-direction").textContent =
        isValid ? (payload.discrete_direction || "--") : "none";
    document.getElementById("arrow-angle").textContent =
        isValid ? numberText(payload.angle_deg, 1) : "--";
    document.getElementById("arrow-confidence").textContent =
        numberText(payload.confidence, 3);
    document.getElementById("arrow-vector-x").textContent =
        isValid ? numberText(vector.x, 3) : "--";
    document.getElementById("arrow-vector-y").textContent =
        isValid ? numberText(vector.y, 3) : "--";
    document.getElementById("arrow-tip").textContent =
        isValid ? pointText(payload.tip) : "--";
    document.getElementById("arrow-tail").textContent =
        isValid ? pointText(payload.tail) : "--";
    document.getElementById("arrow-reason").textContent =
        isValid ? "方向估计正常" : `原因：${payload.reason || "unknown"}`;
}

async function refreshStatus() {
    const statusElement = document.getElementById("connection-status");
    try {
        const response = await fetch("/api/status", {cache: "no-store"});
        if (!response.ok) throw new Error("HTTP " + response.status);
        const data = await response.json();

        statusElement.textContent = data.image_online
            ? "Web服务：图像在线" : "Web服务：等待图像";
        statusElement.className = data.image_online
            ? "badge online" : "badge offline";

        document.getElementById("fps").textContent = numberText(data.image_fps, 1);
        document.getElementById("update-time").textContent = new Date().toLocaleTimeString();

        updateDetections(data.detection);
        updatePose(data.pose);
        updateArrow(data.arrow);
    } catch (error) {
        statusElement.textContent = "Web服务：连接失败";
        statusElement.className = "badge offline";
    }
}

setInterval(refreshStatus, 300);
refreshStatus();
</script>
</body>
</html>
"""


class VisionWebDashboard:
    def __init__(self):
        self.bridge = CvBridge()

        self.host = rospy.get_param("~host", "0.0.0.0")
        self.port = int(rospy.get_param("~port", 8080))

        self.image_topic = rospy.get_param(
            "~image_topic", "/arrow/annotated_image"
        )
        self.detection_topic = rospy.get_param(
            "~detection_topic", "/web/detections"
        )
        self.pose_topic = rospy.get_param(
            "~pose_topic", "/web/pose"
        )
        self.arrow_topic = rospy.get_param(
            "~arrow_topic", "/arrow/direction"
        )

        self.jpeg_quality = int(rospy.get_param("~jpeg_quality", 80))
        self.stream_fps = float(rospy.get_param("~stream_fps", 8.0))
        self.web_width = int(rospy.get_param("~web_width", 960))

        self.detection_timeout = float(
            rospy.get_param("~detection_timeout", 2.0)
        )
        self.pose_timeout = float(
            rospy.get_param("~pose_timeout", 2.0)
        )
        self.arrow_timeout = float(
            rospy.get_param("~arrow_timeout", 2.0)
        )

        self.frame_lock = threading.Lock()
        self.state_lock = threading.Lock()

        self.latest_jpeg = None
        self.last_frame_wall_time = 0.0
        self.image_fps = 0.0

        self.latest_detection = {"count": 0, "detections": []}
        self.latest_pose = None
        self.latest_arrow = None

        rospy.Subscriber(
            self.image_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.Subscriber(
            self.detection_topic,
            String,
            self.detection_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.pose_topic,
            String,
            self.pose_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.arrow_topic,
            String,
            self.arrow_callback,
            queue_size=1,
        )

        self.app = Flask(__name__)
        self.app.add_url_rule("/", "index", self.index)
        self.app.add_url_rule("/video_feed", "video_feed", self.video_feed)
        self.app.add_url_rule("/api/status", "api_status", self.api_status)
        self.app.add_url_rule("/health", "health", self.health)

        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        rospy.loginfo("Vision Web Dashboard initialized")
        rospy.loginfo("Image topic: %s", self.image_topic)
        rospy.loginfo("Detection topic: %s", self.detection_topic)
        rospy.loginfo("Pose topic: %s", self.pose_topic)
        rospy.loginfo("Arrow topic: %s", self.arrow_topic)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except CvBridgeError as exc:
            rospy.logerr_throttle(
                2.0, "Web cv_bridge error: %s", str(exc)
            )
            return

        if self.web_width > 0:
            height, width = frame.shape[:2]
            if width > self.web_width:
                scale = self.web_width / float(width)
                new_height = max(1, int(height * scale))
                frame = cv2.resize(
                    frame,
                    (self.web_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            rospy.logwarn_throttle(2.0, "Failed to encode Web JPEG")
            return

        now = time.time()
        with self.frame_lock:
            if self.last_frame_wall_time > 0:
                dt = now - self.last_frame_wall_time
                if dt > 0:
                    current_fps = 1.0 / dt
                    self.image_fps = (
                        current_fps
                        if self.image_fps <= 0
                        else 0.9 * self.image_fps + 0.1 * current_fps
                    )
            self.last_frame_wall_time = now
            self.latest_jpeg = encoded.tobytes()

    def detection_callback(self, msg):
        payload = self.decode_json(msg.data, "detection")
        if payload is None:
            return
        payload["received_at"] = time.time()
        with self.state_lock:
            self.latest_detection = payload

    def pose_callback(self, msg):
        payload = self.decode_json(msg.data, "pose")
        if payload is None:
            return
        payload["received_at"] = time.time()
        with self.state_lock:
            self.latest_pose = payload

    def arrow_callback(self, msg):
        payload = self.decode_json(msg.data, "arrow")
        if payload is None:
            return
        payload["received_at"] = time.time()
        with self.state_lock:
            self.latest_arrow = payload

    @staticmethod
    def decode_json(data, name):
        try:
            return json.loads(data)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0, "Invalid %s JSON: %s", name, str(exc)
            )
            return None

    def index(self):
        return Response(PAGE_HTML, mimetype="text/html")

    def mjpeg_generator(self):
        period = 1.0 / max(self.stream_fps, 0.5)
        while not rospy.is_shutdown():
            with self.frame_lock:
                jpeg = self.latest_jpeg
            if jpeg is None:
                time.sleep(0.1)
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

    @staticmethod
    def apply_timeout(payload, timeout, now, stream_name):
        if payload is None:
            return None
        received_at = payload.get("received_at")
        if received_at is None or timeout <= 0:
            return payload
        if now - float(received_at) <= timeout:
            return payload

        stale = copy.deepcopy(payload)
        stale["valid"] = False
        stale["reason"] = "{}_timeout".format(stream_name)
        return stale

    def api_status(self):
        with self.frame_lock:
            last_frame = self.last_frame_wall_time
            fps = self.image_fps

        with self.state_lock:
            detection = copy.deepcopy(self.latest_detection)
            pose = copy.deepcopy(self.latest_pose)
            arrow = copy.deepcopy(self.latest_arrow)

        now = time.time()
        image_online = last_frame > 0 and now - last_frame < 3.0

        detection = self.apply_timeout(
            detection, self.detection_timeout, now, "detection"
        )
        pose = self.apply_timeout(
            pose, self.pose_timeout, now, "pose"
        )
        arrow = self.apply_timeout(
            arrow, self.arrow_timeout, now, "arrow"
        )

        return jsonify(
            {
                "server_time": now,
                "image_online": image_online,
                "image_fps": fps,
                "topics": {
                    "image": self.image_topic,
                    "detection": self.detection_topic,
                    "pose": self.pose_topic,
                    "arrow": self.arrow_topic,
                },
                "detection": detection,
                "pose": pose,
                "arrow": arrow,
            }
        )

    def health(self):
        with self.frame_lock:
            frame_age = (
                time.time() - self.last_frame_wall_time
                if self.last_frame_wall_time > 0
                else None
            )
        return jsonify(
            {
                "ok": True,
                "ros_shutdown": rospy.is_shutdown(),
                "image_age_sec": frame_age,
            }
        )

    def run(self):
        rospy.loginfo("Open browser: http://%s:%d", self.host, self.port)
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )


if __name__ == "__main__":
    try:
        rospy.init_node("vision_web_dashboard", anonymous=False)
        VisionWebDashboard().run()
    except rospy.ROSInterruptException:
        pass