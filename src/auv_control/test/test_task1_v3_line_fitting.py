#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v3_line_fitting.py
功能：Task1 红线曲线拟合观察工具，只处理感知数据，不发布运动控制命令。

流程：
    1. 监听相机红线多点识别消息，将有效点从 camera 变换到 map；
    2. 保留骨架路径点顺序，验证相邻间距和各组三点的局部几何关系；
    3. 首次锁定短时间窗内置信度最高的合理红线；
    4. 后续点只有能与已锁曲线合理关联时，才进入点集并重新拟合；
    5. 当前拟合经后续合理帧确认后立即冻结，固定段重复点不再进入拟合；
    6. 活动点集上限仅用于内存保护，超限时只保留远端衔接点；
    7. Web 显示实时拟合曲线、固定曲线、最新多点识别、机器人航向和手动运动轨迹。

重要：
    本节点不创建 /cmd/pose/ned 发布器，也不会发布任何控制指令；
    测试期间由操作人员手动控制机器人沿红线运动。

监听：/obj/line_message，/left/image_raw，/tf
发布：/task1/line_fitting/trajectory（仅调试 JSON）
网页：默认 http://192.168.1.117:8083

记录：
2026.7.18
    初版。拆出纯视觉曲线拟合和 Web 观察功能，不参与机器人控制。
    将曲线改为“实时拟合、后续帧确认、分段冻结”；固定段上的重复点不再参与拟合，
    只接收能从固定曲线末端合理延伸的新点。
    新增 v3 独立入口。本子任务本来就不发布运动指令，因此无需接入
    motion_supervisor；保留为验证 v3 巡线所使用冻结曲线的纯感知工具。
    将曲线点融合权重和实际轨迹最大保存点数开放为 launch 参数，保持与
    v3 巡线节点使用相同的可调拟合逻辑，并限制 Web 历史轨迹数据量。
2026.7.20
    Web 改为 Reset 原点下的 NED 俯视图，使用固定比例尺并支持滚轮缩放、
    拖动平移；规划和实际轨迹显示 base_link，实际 camera 位置作为航向箭头。
2026.7.20
    红线输入由固定三点 TargetDetection3 迁移到多点 LineDetection；逐点过滤
    point_valid，launch 可限制每帧参与拟合的有效点数量，Web 显示总/有效/使用点数。
"""

import copy
import json
import math
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rospy
import tf
from auv_control.msg import LineDetection
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


NODE_NAME = "test_task1_v3_line_fitting"


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(quaternion):
    return euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ])[2]


def xy_distance(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


def class_names(param_name, default):
    value = rospy.get_param(param_name, default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


TRAJECTORY_HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Task1 红线拟合观察</title><style>
body{margin:0;background:#101820;color:#eef;font-family:Arial,"Microsoft YaHei"}
header{padding:12px 18px;background:#172630}canvas{display:block;margin:16px auto;background:#f7fbfd;border-radius:8px}
#s{margin-left:25px;color:#bcd0d8}.legend{font-size:13px;color:#bcd0d8;margin-top:8px}.legend span{margin-right:18px}
</style></head><body><header><b>Task1 红线曲线拟合观察（无运动控制）</b><span id="s">等待数据</span>
<div class="legend"><span>红/橙线：base_link 拟合/固定轨迹</span><span>灰点：原始活动拟合点</span>
<span>蓝线：base_link 手动运动轨迹</span><span>青色圆点→箭头：base_link→camera</span>
<span>粉点：最新有效识别点</span><span>滚轮缩放，拖动平移</span></div></header>
<canvas id="c" width="960" height="680"></canvas><script>
const c=document.getElementById('c'),x=c.getContext('2d');let k=70,ox=90,oy=c.height-80,last={};
function q(a){return[ox+a[1]*k,oy-a[0]*k]}
function line(a,col,w){if(!a||a.length<2)return;x.beginPath();a.forEach((v,i)=>{let z=q(v);i?x.lineTo(...z):x.moveTo(...z)});x.strokeStyle=col;x.lineWidth=w;x.stroke()}
function dot(a,col,r){if(!a)return;let z=q(a);x.beginPath();x.arc(z[0],z[1],r,0,7);x.fillStyle=col;x.fill()}
function tag(a,text,col){if(!a)return;let z=q(a);x.fillStyle=col;x.font='bold 13px Arial';x.fillText(text,z[0]+7,z[1]-7)}
function nice(v){let p=10**Math.floor(Math.log10(Math.max(v,.01))),n=v/p;return(n<=1?1:n<=2?2:n<=5?5:10)*p}
function grid(){let minN=(oy-c.height)/k,maxN=oy/k,minE=-ox/k,maxE=(c.width-ox)/k,st=nice(Math.max(maxN-minN,maxE-minE)/8);x.font='11px Arial';x.lineWidth=1;
for(let n=Math.ceil(minN/st)*st;n<=maxN+1e-9;n+=st){let y=q([n,0])[1];x.strokeStyle=Math.abs(n)<1e-9?'#71848d':'#dce6ea';x.beginPath();x.moveTo(0,y);x.lineTo(c.width,y);x.stroke();x.fillStyle='#52646d';x.fillText(`N ${n.toFixed(1)}`,5,y-4)}
for(let e=Math.ceil(minE/st)*st;e<=maxE+1e-9;e+=st){let z=q([0,e]);x.strokeStyle=Math.abs(e)<1e-9?'#71848d':'#dce6ea';x.beginPath();x.moveTo(z[0],0);x.lineTo(z[0],c.height);x.stroke();x.fillStyle='#52646d';x.fillText(`E ${e.toFixed(1)}`,z[0]+4,c.height-7)}
if(ox>=0&&ox<=c.width){x.fillStyle='#263942';x.font='bold 13px Arial';x.fillText('N / North',ox+8,16)}if(oy>=0&&oy<=c.height){x.fillStyle='#263942';x.font='bold 13px Arial';x.fillText('E / East',c.width-68,oy-8)}let o=q([0,0]);if(o[0]>=0&&o[0]<=c.width&&o[1]>=0&&o[1]<=c.height){dot([0,0],'#111',5);tag([0,0],'Reset O(0,0)','#111')}x.fillStyle='#52646d';x.fillText(`${k.toFixed(0)} px/m`,c.width-72,18)}
function bodyArrow(base,camera){if(!base||!camera)return;let a=q(base),b=q(camera),r=Math.atan2(b[1]-a[1],b[0]-a[0]),h=10;x.strokeStyle='#00a9c7';x.lineWidth=4;x.beginPath();x.moveTo(...a);x.lineTo(...b);x.stroke();x.beginPath();x.moveTo(...b);x.lineTo(b[0]-h*Math.cos(r-.55),b[1]-h*Math.sin(r-.55));x.moveTo(...b);x.lineTo(b[0]-h*Math.cos(r+.55),b[1]-h*Math.sin(r+.55));x.stroke();dot(camera,'#00a9c7',4)}
function draw(d){last=d;let latest=d.latest_line_points||[];x.clearRect(0,0,c.width,c.height);grid();
(d.accepted_points||[]).forEach(v=>dot(v,'#879aa3',3));line(d.fitted_curve,'#e74c3c',3);line(d.fixed_curve,'#f39c12',5);line(d.actual_path,'#1677ff',3);
latest.forEach((v,i)=>{dot(v,'#ff2d91',4);if(i===0||i===latest.length-1||i===Math.floor(latest.length/2))tag(v,`P${i+1}`,'#ff2d91')});
dot(d.robot,'#00cfe8',8);bodyArrow(d.robot,d.camera);
document.getElementById('s').textContent=`状态 ${d.state}　相机 ${d.camera_ready?'已开启':'未开启'}　base航向 ${(d.robot_yaw_deg||0).toFixed(1)}°　D(base/camera) ${(d.robot_down??0).toFixed(2)}/${(d.camera_down??0).toFixed(2)} m　识别点 总/有效/使用 ${d.line_input_point_count||0}/${d.line_valid_point_count||0}/${d.line_used_point_count||0}　锁线 ${d.line_locked?'是':'否'}　固定/拟合 ${d.fixed_length||0}/${d.curve_length||0} m　确认 ${d.freeze_confirmations||0}/${d.freeze_required||0}　有效帧/拒绝帧 ${d.accepted_triplets||0}/${d.rejected_triplets||0}`}
function pos(e){let r=c.getBoundingClientRect();return[(e.clientX-r.left)*c.width/r.width,(e.clientY-r.top)*c.height/r.height]}
c.addEventListener('wheel',e=>{e.preventDefault();let m=pos(e),old=k,ne=(m[0]-ox)/old,nn=(oy-m[1])/old;k=Math.max(10,Math.min(500,k*(e.deltaY<0?1.15:1/1.15)));ox=m[0]-ne*k;oy=m[1]+nn*k;draw(last)},{passive:false});
let drag=false,pm=null;c.addEventListener('mousedown',e=>{drag=true;pm=pos(e)});window.addEventListener('mouseup',()=>drag=false);window.addEventListener('mousemove',e=>{if(!drag)return;let m=pos(e);ox+=m[0]-pm[0];oy+=m[1]-pm[1];pm=m;draw(last)});
async function t(){try{draw(await(await fetch('/data',{cache:'no-store'})).json())}catch(e){}setTimeout(t,500)}draw({});t();
</script></body></html>"""


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class TrajectoryWebServer:
    def __init__(self, host, port):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler):
                if handler.path == "/data":
                    body = handler.server.latest.encode("utf-8")
                    kind = "application/json; charset=utf-8"
                else:
                    body = TRAJECTORY_HTML.encode("utf-8")
                    kind = "text/html; charset=utf-8"
                handler.send_response(200)
                handler.send_header("Content-Type", kind)
                handler.send_header("Content-Length", str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self.server = ReusableThreadingHTTPServer((str(host), int(port)), Handler)
        self.server.daemon_threads = True
        self.server.latest = "{}"
        thread = threading.Thread(target=self.server.serve_forever)
        thread.daemon = True
        thread.start()

    def update(self, payload):
        self.server.latest = payload


class LineFittingTest:
    """只观察视觉拟合效果，不向机器人发布运动控制。"""

    WAIT_LINE = "WAIT_LINE"
    SELECT_CANDIDATE = "SELECT_CANDIDATE"
    FITTING = "FITTING"

    def __init__(self):
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))
        self.tf_listener = tf.TransformListener()

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.line_tracking_frame = str(rospy.get_param(
            "~line_tracking_frame", "camera"
        )).strip().lstrip("/") or "camera"
        self.line_topic = rospy.get_param("~line_topic", "/obj/line_message")
        # 每帧最多使用的有效红线点数；0 表示使用消息中的全部有效点。
        requested_line_point_count = int(rospy.get_param(
            "~line_accept_point_count", 20
        ))
        self.line_accept_point_count = (
            0 if requested_line_point_count <= 0
            else max(2, requested_line_point_count)
        )
        self.camera_topic = rospy.get_param("~camera_topic", "/left/image_raw")
        self.trajectory_topic = rospy.get_param(
            "~trajectory_topic", "/task1/v3/line_fitting/trajectory"
        )
        self.tf_timeout_seconds = max(0.0, float(rospy.get_param(
            "~tf_timeout_seconds", 1.0
        )))
        self.camera_message_timeout = max(0.0, float(rospy.get_param(
            "~camera_message_timeout", 2.0
        )))
        self.log_period_seconds = max(0.1, float(rospy.get_param(
            "~log_period_seconds", 2.0
        )))

        self.line_classes = class_names("~line_classes", ["line"])
        self.max_camera_distance = max(0.1, float(rospy.get_param(
            "~max_camera_distance", 6.0
        )))
        self.line_lock_window_seconds = max(0.0, float(rospy.get_param(
            "~line_lock_window_seconds", 0.3
        )))
        self.line_low_confidence_threshold = float(rospy.get_param(
            "~line_low_confidence_threshold", 0.50
        ))
        self.line_min_point_spacing = max(0.0, float(rospy.get_param(
            "~line_min_point_spacing", 0.005
        )))
        self.line_max_point_spacing = max(0.0, float(rospy.get_param(
            "~line_max_point_spacing", 4.0
        )))
        self.line_triplet_max_residual = max(0.0, float(rospy.get_param(
            "~line_triplet_max_residual", 0.35
        )))
        self.line_triplet_max_bend = math.radians(float(rospy.get_param(
            "~line_triplet_max_bend_deg", 60.0
        )))
        self.line_association_distance = max(0.0, float(rospy.get_param(
            "~line_association_distance", 0.35
        )))
        self.line_high_confidence_association_distance = max(0.0, float(
            rospy.get_param("~line_high_confidence_association_distance", 0.50)
        ))
        self.line_association_angle = math.radians(float(rospy.get_param(
            "~line_association_angle_deg", 35.0
        )))
        self.line_high_confidence_association_angle = math.radians(float(
            rospy.get_param("~line_high_confidence_association_angle_deg", 45.0)
        ))
        self.line_extension_max_gap = max(0.0, float(rospy.get_param(
            "~line_extension_max_gap", 1.0
        )))

        self.line_point_merge_distance = max(0.001, float(rospy.get_param(
            "~line_point_merge_distance", 0.12
        )))
        self.line_point_update_alpha = max(0.0, min(1.0, float(
            rospy.get_param("~line_point_update_alpha", 0.20)
        )))
        self.line_curve_max_points = max(3, int(rospy.get_param(
            "~line_curve_max_points", 200
        )))
        requested_overlap = int(rospy.get_param(
            "~line_curve_overlap_points", 40
        ))
        self.line_curve_overlap_points = max(
            2, min(requested_overlap, self.line_curve_max_points - 1)
        )
        self.line_window_backtrack_distance = max(0.0, float(rospy.get_param(
            "~line_window_backtrack_distance", 1.0
        )))
        self.line_curve_sample_count = max(3, int(rospy.get_param(
            "~line_curve_sample_count", 100
        )))
        self.line_curve_degree = max(1, int(rospy.get_param(
            "~line_curve_degree", 3
        )))
        self.line_curve_min_length = max(0.01, float(rospy.get_param(
            "~line_curve_min_length", 0.15
        )))

        # 实时拟合曲线经后续帧确认后立即冻结，不再等待原始点达到上限。
        self.curve_freeze_required_frames = max(1, int(rospy.get_param(
            "~curve_freeze_required_frames", 2
        )))
        self.curve_freeze_min_length = max(0.01, float(rospy.get_param(
            "~curve_freeze_min_length", 0.15
        )))
        self.curve_freeze_min_advance = max(0.0, float(rospy.get_param(
            "~curve_freeze_min_advance", 0.10
        )))
        self.curve_freeze_ignore_distance = max(0.0, float(rospy.get_param(
            "~curve_freeze_ignore_distance", 0.12
        )))
        self.curve_freeze_endpoint_guard = max(0.0, float(rospy.get_param(
            "~curve_freeze_endpoint_guard", 0.05
        )))
        self.curve_freeze_overlap_distance = max(0.0, float(rospy.get_param(
            "~curve_freeze_overlap_distance", 0.25
        )))

        self.actual_path_min_spacing = max(0.001, float(rospy.get_param(
            "~actual_path_min_spacing", 0.02
        )))
        self.actual_path_max_points = max(10, int(rospy.get_param(
            "~actual_path_max_points", 2000
        )))
        self.trajectory_publish_period = max(0.05, float(rospy.get_param(
            "~trajectory_publish_period", 0.5
        )))
        self.trajectory_web_enabled = bool(rospy.get_param(
            "~trajectory_web_enabled", True
        ))
        self.trajectory_web_host = rospy.get_param(
            "~trajectory_web_host", "192.168.1.117"
        )
        self.trajectory_web_port = int(rospy.get_param(
            "~trajectory_web_port", 8083
        ))

        # 本节点只有调试 JSON 发布器，明确不创建运动控制发布器。
        self.trajectory_pub = rospy.Publisher(
            self.trajectory_topic, String, queue_size=2
        )

        self.trajectory_web = None
        if self.trajectory_web_enabled:
            try:
                self.trajectory_web = TrajectoryWebServer(
                    self.trajectory_web_host, self.trajectory_web_port
                )
                rospy.loginfo(
                    "%s: 拟合网页 http://%s:%d（本节点不发布控制指令）",
                    NODE_NAME,
                    self.trajectory_web_host,
                    self.trajectory_web_port,
                )
            except OSError as error:
                rospy.logwarn("%s: 拟合网页启动失败: %s", NODE_NAME, error)

        self.state = self.WAIT_LINE
        self.last_camera_time = None
        self.line_lock_candidate = None
        self.line_locked = False
        self.line_lock_confidence = 0.0
        self.line_reference_point = None
        self.line_axis = None
        self.line_raw_points = []
        self.line_committed_curve_points = []
        self.line_committed_curve_s = []
        self.line_curve_points = []
        self.line_curve_s = []
        self.line_start_point = None
        self.line_end_point = None
        self.line_fit_residual = 0.0
        self.freeze_confirmation_count = 0
        self.freeze_version = 0
        self.ignored_fixed_points = 0
        self.latest_line_points = []
        self.latest_line_input_count = 0
        self.latest_line_valid_count = 0
        self.latest_line_used_count = 0
        self.latest_confidence = 0.0
        self.accepted_triplets = 0
        self.rejected_triplets = 0
        self.last_reject_reason = "none"
        self.actual_trajectory = []
        self.tracking_lever_arm = None
        self.last_trajectory_publish_time = rospy.Time(0)

        rospy.Subscriber(
            self.line_topic, LineDetection, self.line_callback, queue_size=10
        )
        rospy.Subscriber(
            self.camera_topic, rospy.AnyMsg, self.camera_callback, queue_size=1
        )
        rospy.loginfo(
            "%s: 红线接口=auv_control/LineDetection；每帧拟合点上限=%s",
            NODE_NAME,
            "全部有效点" if self.line_accept_point_count == 0
            else str(self.line_accept_point_count),
        )

    def camera_callback(self, _message):
        self.last_camera_time = rospy.Time.now()

    def camera_ready(self):
        return (
            self.last_camera_time is not None
            and (rospy.Time.now() - self.last_camera_time).to_sec()
            <= self.camera_message_timeout
        )

    def get_frame_pose(self, frame):
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                self.map_frame, frame, rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0, "%s: 等待 %s -> %s TF: %s",
                NODE_NAME, self.map_frame, frame, error
            )
            return None
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def get_current_pose(self):
        """返回 Web 机器人圆点和实际轨迹使用的 base_link 位姿。"""
        return self.get_frame_pose(self.robot_frame)

    def get_tracking_pose(self):
        """返回 Web 航向箭头终点使用的 camera 位姿。"""
        return self.get_frame_pose(self.line_tracking_frame)

    def get_tracking_lever_arm(self):
        """读取 base_link 指向 camera 的水平杆臂。"""
        if self.tracking_lever_arm is not None:
            return self.tracking_lever_arm
        try:
            translation, _rotation = self.tf_listener.lookupTransform(
                self.robot_frame, self.line_tracking_frame, rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0,
                "%s: 等待 %s -> %s 杆臂 TF: %s",
                NODE_NAME,
                self.robot_frame,
                self.line_tracking_frame,
                error,
            )
            return None
        self.tracking_lever_arm = (float(translation[0]), float(translation[1]))
        rospy.loginfo(
            "%s: Web 规划轨迹使用 %s 水平杆臂=(%.3f, %.3f) m",
            NODE_NAME,
            self.line_tracking_frame,
            self.tracking_lever_arm[0],
            self.tracking_lever_arm[1],
        )
        return self.tracking_lever_arm

    def tracking_point_to_base_point(self, camera_point, target_yaw):
        """按曲线切向把 camera 曲线点换算为 base_link 规划点。"""
        lever_arm = self.get_tracking_lever_arm()
        if lever_arm is None:
            return None
        cosine = math.cos(target_yaw)
        sine = math.sin(target_yaw)
        return Point(
            camera_point.x - (cosine * lever_arm[0] - sine * lever_arm[1]),
            camera_point.y - (sine * lever_arm[0] + cosine * lever_arm[1]),
            camera_point.z,
        )

    def tracking_curve_to_base_points(self, points):
        """生成仅供 Web 显示的 base_link 拟合轨迹。"""
        if len(points) < 2:
            return []
        converted = []
        for index, point in enumerate(points):
            if index < len(points) - 1:
                other = points[index + 1]
                yaw = math.atan2(other.y - point.y, other.x - point.x)
            else:
                other = points[index - 1]
                yaw = math.atan2(point.y - other.y, point.x - other.x)
            base_point = self.tracking_point_to_base_point(point, yaw)
            if base_point is None:
                return []
            converted.append(base_point)
        return converted

    def transform_pose_to_map(self, pose):
        try:
            self.tf_listener.waitForTransform(
                self.map_frame,
                pose.header.frame_id,
                pose.header.stamp,
                rospy.Duration(self.tf_timeout_seconds),
            )
            return self.tf_listener.transformPose(self.map_frame, pose)
        except tf.Exception:
            try:
                latest_pose = copy.deepcopy(pose)
                latest_pose.header.stamp = rospy.Time(0)
                self.tf_listener.waitForTransform(
                    self.map_frame,
                    latest_pose.header.frame_id,
                    rospy.Time(0),
                    rospy.Duration(self.tf_timeout_seconds),
                )
                return self.tf_listener.transformPose(
                    self.map_frame, latest_pose
                )
            except tf.Exception as error:
                rospy.logwarn_throttle(
                    2.0, "%s: 红线坐标转换失败: %s", NODE_NAME, error
                )
                return None

    @staticmethod
    def point_to_chord_distance(point, start, end):
        vx = end.x - start.x
        vy = end.y - start.y
        length_sq = vx * vx + vy * vy
        if length_sq < 1e-9:
            return float("inf")
        ratio = clamp(
            ((point.x - start.x) * vx + (point.y - start.y) * vy) / length_sq,
            0.0,
            1.0,
        )
        projection = Point(start.x + ratio * vx, start.y + ratio * vy, point.z)
        return xy_distance(point, projection)

    @staticmethod
    def undirected_angle_error(first, second):
        error = abs(wrap_angle(first - second))
        return min(error, abs(math.pi - error))

    def order_and_validate_points(self, poses, reference):
        """验证一帧有序多点，并把靠近当前参考位置的一端作为起点。"""
        points = [copy.deepcopy(pose.pose.position) for pose in poses]
        if len(points) < 2:
            return None, None, "too_few_valid_points"
        coordinates = np.array([[point.x, point.y] for point in points], dtype=float)
        if not np.isfinite(coordinates).all():
            return None, None, "non_finite_point"
        # lineN 发布端已经按骨架主路径排列，不能再按单一主轴排序，否则弯线会乱序。
        ordered = points
        if xy_distance(ordered[-1], reference) < xy_distance(ordered[0], reference):
            ordered.reverse()

        spacings = [
            xy_distance(ordered[index - 1], ordered[index])
            for index in range(1, len(ordered))
        ]
        if min(spacings) < self.line_min_point_spacing:
            return None, None, "point_spacing_too_small"
        if max(spacings) > self.line_max_point_spacing:
            return None, None, "point_spacing_too_large"

        maximum_bend = 0.0
        maximum_residual = 0.0
        for index in range(1, len(ordered) - 1):
            previous = ordered[index - 1]
            current = ordered[index]
            following = ordered[index + 1]
            first_yaw = math.atan2(
                current.y - previous.y, current.x - previous.x
            )
            second_yaw = math.atan2(
                following.y - current.y, following.x - current.x
            )
            maximum_bend = max(
                maximum_bend,
                self.undirected_angle_error(first_yaw, second_yaw),
            )
            maximum_residual = max(
                maximum_residual,
                self.point_to_chord_distance(current, previous, following),
            )
        if maximum_bend > self.line_triplet_max_bend:
            return None, None, "local_bend_too_large"
        if maximum_residual > self.line_triplet_max_residual:
            return None, None, "local_residual_too_large"

        detected_yaw = math.atan2(
            ordered[-1].y - ordered[0].y,
            ordered[-1].x - ordered[0].x,
        )
        return ordered, detected_yaw, "valid"

    def curve_ready(self):
        return (
            len(self.line_curve_points) >= 2
            and len(self.line_curve_s) == len(self.line_curve_points)
        )

    @staticmethod
    def cumulative_distance(points):
        distances = [0.0]
        for index in range(1, len(points)):
            distances.append(
                distances[-1] + xy_distance(points[index - 1], points[index])
            )
        return distances

    def project_to_polyline(self, point, curve_points, curve_s=None):
        if len(curve_points) < 2:
            return None
        if curve_s is None:
            curve_s = self.cumulative_distance(curve_points)
        best = None
        for index in range(len(curve_points) - 1):
            start = curve_points[index]
            end = curve_points[index + 1]
            vx = end.x - start.x
            vy = end.y - start.y
            segment_sq = vx * vx + vy * vy
            if segment_sq < 1e-9:
                continue
            ratio = clamp(
                ((point.x - start.x) * vx + (point.y - start.y) * vy)
                / segment_sq,
                0.0,
                1.0,
            )
            projected_x = start.x + ratio * vx
            projected_y = start.y + ratio * vy
            distance = math.hypot(point.x - projected_x, point.y - projected_y)
            candidate = {
                "distance": distance,
                "path_s": curve_s[index] + ratio * math.sqrt(segment_sq),
                "segment_yaw": math.atan2(vy, vx),
            }
            if best is None or distance < best["distance"]:
                best = candidate
        return best

    def project_to_curve(self, point):
        if not self.curve_ready():
            return None
        return self.project_to_polyline(
            point, self.line_curve_points, self.line_curve_s
        )

    def valid_curve_extension(self, points, lateral_limit):
        if len(self.line_curve_points) < 2:
            return False
        # 起点在首次锁线时已经固定，后续只允许从曲线远端继续向前延伸。
        endpoint = self.line_curve_points[-1]
        inner_point = self.line_curve_points[-2]
        tangent_x = endpoint.x - inner_point.x
        tangent_y = endpoint.y - inner_point.y
        tangent_length = math.hypot(tangent_x, tangent_y)
        if tangent_length < 1e-9:
            return False
        tangent_x /= tangent_length
        tangent_y /= tangent_length
        for point in points:
            dx = point.x - endpoint.x
            dy = point.y - endpoint.y
            outward = dx * tangent_x + dy * tangent_y
            lateral = abs(tangent_x * dy - tangent_y * dx)
            if (
                0.0 < outward <= self.line_extension_max_gap
                and lateral <= lateral_limit
            ):
                return True
        return False

    def line_segment_associated(self, points, detected_yaw, confidence):
        projections = [self.project_to_curve(point) for point in points]
        projections = [item for item in projections if item is not None]
        if not projections:
            return False, float("inf"), math.pi, "curve_projection_failed"
        best = min(projections, key=lambda item: item["distance"])
        fit_distance = best["distance"]
        angle_error = self.undirected_angle_error(
            detected_yaw, best["segment_yaw"]
        )
        low_confidence = confidence < self.line_low_confidence_threshold
        distance_limit = (
            self.line_association_distance
            if low_confidence
            else self.line_high_confidence_association_distance
        )
        angle_limit = (
            self.line_association_angle
            if low_confidence
            else self.line_high_confidence_association_angle
        )
        close_enough = (
            fit_distance <= distance_limit
            or self.valid_curve_extension(points, distance_limit)
        )
        if not close_enough:
            return False, fit_distance, angle_error, "line_too_far"
        if angle_error > angle_limit:
            return False, fit_distance, angle_error, "line_direction_mismatch"
        return True, fit_distance, angle_error, "associated"

    def roll_line_fit_window(self):
        """原始点上限只作为内存保护，不再触发未经确认的曲线冻结。"""
        if (
            len(self.line_raw_points) <= self.line_curve_max_points
            or not self.curve_ready()
            or self.line_reference_point is None
        ):
            return
        ordered = sorted(self.line_raw_points, key=lambda point: (
            self.project_to_curve(point)["path_s"]
            if self.project_to_curve(point) is not None else -1.0
        ))
        self.line_raw_points = [
            copy.deepcopy(point)
            for point in ordered[-self.line_curve_overlap_points:]
        ]
        rospy.loginfo(
            "%s: 活动点达到上限，仅保留远端点；固定曲线=%d点，活动原始点=%d点",
            NODE_NAME,
            len(self.line_committed_curve_points),
            len(self.line_raw_points),
        )

    def points_not_on_fixed_curve(self, points):
        """忽略固定段重复点，只保留固定末端之后可用于延伸的点。"""
        if not self.line_committed_curve_points:
            return points
        fixed_end = self.line_committed_curve_points[-1]
        fixed_inner = self.line_committed_curve_points[-2]
        tangent_x = fixed_end.x - fixed_inner.x
        tangent_y = fixed_end.y - fixed_inner.y
        tangent_length = math.hypot(tangent_x, tangent_y)
        if tangent_length < 1e-9:
            return []
        tangent_x /= tangent_length
        tangent_y /= tangent_length
        fixed_length = self.line_committed_curve_s[-1]
        active = []
        for point in points:
            projection = self.project_to_polyline(
                point,
                self.line_committed_curve_points,
                self.line_committed_curve_s,
            )
            dx = point.x - fixed_end.x
            dy = point.y - fixed_end.y
            outward = dx * tangent_x + dy * tangent_y
            lateral = abs(tangent_x * dy - tangent_y * dx)
            forward_extension = (
                outward > self.curve_freeze_endpoint_guard
                and outward <= self.line_extension_max_gap
                and lateral <= self.line_high_confidence_association_distance
            )
            on_fixed_curve = (
                projection is not None
                and projection["distance"] <= self.curve_freeze_ignore_distance
            )
            far_behind_fixed_end = (
                projection is not None
                and projection["path_s"]
                < fixed_length - self.line_window_backtrack_distance
            )
            if forward_extension or (not on_fixed_curve and not far_behind_fixed_end):
                active.append(point)
            else:
                self.ignored_fixed_points += 1
        return active

    def freeze_confirmed_curve(self):
        """把经连续合理帧确认的当前拟合曲线整体冻结。"""
        if (
            not self.curve_ready()
            or self.freeze_confirmation_count < self.curve_freeze_required_frames
        ):
            return False
        current_length = self.line_curve_s[-1]
        fixed_length = (
            self.line_committed_curve_s[-1]
            if self.line_committed_curve_s else 0.0
        )
        required_advance = (
            self.curve_freeze_min_length
            if not self.line_committed_curve_points
            else self.curve_freeze_min_advance
        )
        if current_length < fixed_length + required_advance:
            return False

        self.line_committed_curve_points = [
            copy.deepcopy(point) for point in self.line_curve_points
        ]
        self.line_committed_curve_s = self.cumulative_distance(
            self.line_committed_curve_points
        )
        self.freeze_version += 1
        self.freeze_confirmation_count = 0

        # 仅保留固定末端附近的原始点作为下一段拟合的平滑衔接，不允许其改变固定段。
        minimum_s = max(
            0.0,
            self.line_committed_curve_s[-1] - self.curve_freeze_overlap_distance,
        )
        retained = []
        for point in self.line_raw_points:
            projection = self.project_to_polyline(
                point,
                self.line_committed_curve_points,
                self.line_committed_curve_s,
            )
            if projection is not None and projection["path_s"] >= minimum_s:
                retained.append(copy.deepcopy(point))
        if len(retained) < 2:
            retained = [copy.deepcopy(point) for point in self.line_raw_points[-2:]]
        self.line_raw_points = retained
        rospy.loginfo(
            "%s: 固定曲线版本=%d，固定长度=%.2f m，保留末端衔接点=%d",
            NODE_NAME,
            self.freeze_version,
            self.line_committed_curve_s[-1],
            len(self.line_raw_points),
        )
        return True

    def fuse_line_points(self, points):
        for point in points:
            point = Point(point.x, point.y, point.z)
            if not self.line_raw_points:
                self.line_raw_points.append(point)
                continue
            nearest = min(
                self.line_raw_points, key=lambda old: xy_distance(old, point)
            )
            if xy_distance(nearest, point) <= self.line_point_merge_distance:
                old_weight = 1.0 - self.line_point_update_alpha
                nearest.x = (
                    old_weight * nearest.x
                    + self.line_point_update_alpha * point.x
                )
                nearest.y = (
                    old_weight * nearest.y
                    + self.line_point_update_alpha * point.y
                )
                nearest.z = (
                    old_weight * nearest.z
                    + self.line_point_update_alpha * point.z
                )
            else:
                self.line_raw_points.append(point)
        self.roll_line_fit_window()

    def fit_line_curve(self):
        if len(self.line_raw_points) < 2 or self.line_reference_point is None:
            return False
        coordinates = np.array(
            [[point.x, point.y] for point in self.line_raw_points], dtype=float
        )
        center = coordinates.mean(axis=0)
        try:
            _, _, axes = np.linalg.svd(coordinates - center)
        except np.linalg.LinAlgError:
            return False
        axis = axes[0]
        if self.line_axis is not None and np.dot(axis, self.line_axis) < 0.0:
            axis = -axis
        elif self.line_axis is None:
            parameters = np.dot(coordinates - center, axis)
            low = self.line_raw_points[int(np.argmin(parameters))]
            high = self.line_raw_points[int(np.argmax(parameters))]
            if xy_distance(high, self.line_reference_point) < xy_distance(
                low, self.line_reference_point
            ):
                axis = -axis
        self.line_axis = axis

        parameters = np.dot(coordinates - center, axis)
        order = np.argsort(parameters)
        parameters = parameters[order]
        ordered = coordinates[order]
        span = float(parameters[-1] - parameters[0])
        if span < self.line_curve_min_length:
            return False

        degree = min(self.line_curve_degree, len(self.line_raw_points) - 1)
        samples = np.linspace(
            float(parameters[0]),
            float(parameters[-1]),
            self.line_curve_sample_count,
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x_model = np.poly1d(np.polyfit(parameters, ordered[:, 0], degree))
                y_model = np.poly1d(np.polyfit(parameters, ordered[:, 1], degree))
            mean_z = sum(point.z for point in self.line_raw_points) / len(
                self.line_raw_points
            )
            curve = [
                Point(float(x_model(value)), float(y_model(value)), mean_z)
                for value in samples
            ]
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return False

        local_start = min(
            self.line_raw_points,
            key=lambda point: xy_distance(point, self.line_reference_point),
        )
        if xy_distance(curve[-1], local_start) < xy_distance(curve[0], local_start):
            curve.reverse()
            self.line_axis = -self.line_axis
        if self.line_start_point is None or not self.line_locked:
            self.line_start_point = copy.deepcopy(local_start)
        curve[0] = copy.deepcopy(self.line_start_point)

        if self.line_committed_curve_points:
            committed = [
                copy.deepcopy(point) for point in self.line_committed_curve_points
            ]
            join_index = min(
                range(len(curve)),
                key=lambda index: xy_distance(committed[-1], curve[index]),
            )
            suffix = [copy.deepcopy(point) for point in curve[join_index:]]
            if (
                suffix
                and xy_distance(committed[-1], suffix[0])
                <= self.line_point_merge_distance
            ):
                suffix = suffix[1:]
            curve = committed + suffix

        self.line_curve_points = curve
        self.line_curve_s = self.cumulative_distance(curve)
        if self.line_curve_s[-1] < self.line_curve_min_length:
            return False
        self.line_end_point = copy.deepcopy(curve[-1])

        residuals = []
        for point in self.line_raw_points:
            projection = self.project_to_curve(point)
            if projection is not None:
                residuals.append(projection["distance"])
        self.line_fit_residual = (
            sum(residuals) / len(residuals) if residuals else 0.0
        )
        return True

    @staticmethod
    def normalized_confidence(value):
        confidence = float(value)
        if confidence > 1.0:
            confidence /= 100.0
        return clamp(confidence, 0.0, 1.0)

    def valid_line_poses(self, message):
        """从 LineDetection 提取有效点，并按配置等间隔限制参与拟合的数量。"""
        declared_count = int(message.point_count)
        positions = list(message.positions)
        validity = list(message.point_valid)
        self.latest_line_input_count = declared_count
        self.latest_line_valid_count = int(message.valid_count)
        self.latest_line_used_count = 0

        if not message.valid:
            return None, message.reason or "line_message_invalid"
        if declared_count <= 0:
            return None, "empty_line_message"
        if declared_count != len(positions) or declared_count != len(validity):
            return None, "line_array_size_mismatch"
        if not message.header.frame_id:
            return None, "line_frame_id_empty"

        poses = []
        for position, point_valid in zip(positions, validity):
            if not point_valid:
                continue
            if (
                not math.isfinite(position.x)
                or not math.isfinite(position.y)
                or not math.isfinite(position.z)
                or math.sqrt(
                    position.x ** 2 + position.y ** 2 + position.z ** 2
                ) > self.max_camera_distance
            ):
                continue
            pose = PoseStamped()
            pose.header = copy.deepcopy(message.header)
            pose.pose.position = copy.deepcopy(position)
            pose.pose.orientation.w = 1.0
            poses.append(pose)

        if len(poses) < 2:
            return None, "too_few_usable_points"
        if self.line_accept_point_count > 0 and len(poses) > self.line_accept_point_count:
            indices = np.linspace(
                0, len(poses) - 1, self.line_accept_point_count
            ).round().astype(int)
            poses = [poses[int(index)] for index in indices]
        self.latest_line_used_count = len(poses)
        return poses, "valid"

    def reject_triplet(self, reason):
        self.rejected_triplets += 1
        self.last_reject_reason = reason
        rospy.loginfo_throttle(
            self.log_period_seconds,
            "%s: 本帧红线已忽略，原因=%s",
            NODE_NAME,
            reason,
        )

    def line_callback(self, message):
        if message.class_name and message.class_name not in self.line_classes:
            return
        camera_poses, reason = self.valid_line_poses(message)
        if camera_poses is None:
            self.reject_triplet(reason)
            return

        transformed = [
            self.transform_pose_to_map(pose)
            for pose in camera_poses
        ]
        if any(pose is None for pose in transformed):
            self.reject_triplet("tf_failed")
            return
        self.latest_line_points = [
            copy.deepcopy(pose.pose.position) for pose in transformed
        ]

        current = self.get_current_pose()
        if current is None:
            self.reject_triplet("robot_pose_unavailable")
            return
        reference = (
            self.line_reference_point
            if self.line_reference_point is not None
            else current.pose.position
        )
        points, detected_yaw, reason = self.order_and_validate_points(
            transformed, reference
        )
        if points is None:
            self.reject_triplet(reason)
            return

        confidence = self.normalized_confidence(message.conf)
        self.latest_confidence = confidence
        now = rospy.Time.now()
        if not self.line_locked:
            if self.line_lock_candidate is None:
                self.line_lock_candidate = {
                    "started": now,
                    "confidence": confidence,
                    "points": points,
                    "reference": copy.deepcopy(current.pose.position),
                }
            elif confidence > self.line_lock_candidate["confidence"]:
                self.line_lock_candidate["confidence"] = confidence
                self.line_lock_candidate["points"] = points
            self.state = self.SELECT_CANDIDATE
            return

        associated, fit_distance, angle_error, reason = self.line_segment_associated(
            points, detected_yaw, confidence
        )
        if not associated:
            self.reject_triplet(reason)
            rospy.loginfo_throttle(
                self.log_period_seconds,
                "%s: 隔离信息 conf=%.2f，拟合距离=%.2f m，方向差=%.1f deg",
                NODE_NAME,
                confidence,
                fit_distance,
                math.degrees(angle_error),
            )
            return

        active_points = self.points_not_on_fixed_curve(points)
        if not active_points:
            rospy.loginfo_throttle(
                self.log_period_seconds,
                "%s: 本帧点均位于已固定曲线上，已忽略",
                NODE_NAME,
            )
            return
        self.fuse_line_points(active_points)
        if self.fit_line_curve():
            self.accepted_triplets += 1
            self.last_reject_reason = "none"
            self.freeze_confirmation_count = min(
                self.curve_freeze_required_frames,
                self.freeze_confirmation_count + 1,
            )
            self.freeze_confirmed_curve()
        else:
            self.reject_triplet("curve_fit_failed")

    def try_lock_line(self):
        if self.line_locked or self.line_lock_candidate is None:
            return False
        candidate = self.line_lock_candidate
        if (
            rospy.Time.now() - candidate["started"]
        ).to_sec() < self.line_lock_window_seconds:
            return False
        self.line_reference_point = copy.deepcopy(candidate["reference"])
        self.fuse_line_points(candidate["points"])
        if not self.fit_line_curve():
            self.line_lock_candidate = None
            self.line_reference_point = None
            self.line_raw_points = []
            self.state = self.WAIT_LINE
            self.reject_triplet("initial_curve_too_short")
            return False
        self.line_locked = True
        self.line_lock_confidence = candidate["confidence"]
        self.line_lock_candidate = None
        self.accepted_triplets += 1
        self.freeze_confirmation_count = 1
        self.freeze_confirmed_curve()
        self.state = self.FITTING
        rospy.loginfo(
            "%s: 红线已锁定；conf=%.2f，起点=(%.2f, %.2f)，"
            "当前曲线长度=%.2f m，之后只接受可关联点",
            NODE_NAME,
            self.line_lock_confidence,
            self.line_start_point.x,
            self.line_start_point.y,
            self.line_curve_s[-1],
        )
        return True

    def record_actual_trajectory(self, current):
        if (
            not self.actual_trajectory
            or xy_distance(current.pose.position, self.actual_trajectory[-1])
            >= self.actual_path_min_spacing
        ):
            self.actual_trajectory.append(copy.deepcopy(current.pose.position))
            if len(self.actual_trajectory) > self.actual_path_max_points:
                self.actual_trajectory = self.actual_trajectory[
                    -self.actual_path_max_points:
                ]

    def publish_trajectory_status(self, current):
        now = rospy.Time.now()
        if (
            now - self.last_trajectory_publish_time
        ).to_sec() < self.trajectory_publish_period:
            return

        def point_data(point):
            return [round(point.x, 3), round(point.y, 3)] if point else None

        tracking = self.get_tracking_pose()
        fitted_base_curve = self.tracking_curve_to_base_points(
            self.line_curve_points
        )
        fixed_base_curve = self.tracking_curve_to_base_points(
            self.line_committed_curve_points
        )

        payload = {
            "stamp": round(now.to_sec(), 3),
            "state": self.state,
            "camera_ready": self.camera_ready(),
            "line_locked": self.line_locked,
            "lock_confidence": round(self.line_lock_confidence, 3),
            "latest_confidence": round(self.latest_confidence, 3),
            "fit_residual": round(self.line_fit_residual, 3),
            "fixed_length": round(
                self.line_committed_curve_s[-1]
                if self.line_committed_curve_s else 0.0,
                3,
            ),
            "curve_length": round(
                self.line_curve_s[-1] if self.line_curve_s else 0.0, 3
            ),
            "freeze_confirmations": self.freeze_confirmation_count,
            "freeze_required": self.curve_freeze_required_frames,
            "freeze_version": self.freeze_version,
            "ignored_fixed_points": self.ignored_fixed_points,
            "accepted_triplets": self.accepted_triplets,
            "rejected_triplets": self.rejected_triplets,
            "last_reject_reason": self.last_reject_reason,
            "line_input_point_count": self.latest_line_input_count,
            "line_valid_point_count": self.latest_line_valid_count,
            "line_used_point_count": self.latest_line_used_count,
            "robot": point_data(current.pose.position) if current else None,
            "camera": point_data(tracking.pose.position) if tracking else None,
            "robot_down": round(current.pose.position.z, 3) if current else None,
            "camera_down": round(tracking.pose.position.z, 3) if tracking else None,
            "robot_yaw_deg": round(math.degrees(yaw_from_quaternion(
                current.pose.orientation
            )), 2) if current else 0.0,
            "tracking_frame": self.line_tracking_frame,
            "fitted_curve": [point_data(point) for point in fitted_base_curve],
            "fixed_curve": [point_data(point) for point in fixed_base_curve],
            "accepted_points": [point_data(point) for point in self.line_raw_points],
            "latest_line_points": [
                point_data(point) for point in self.latest_line_points
            ],
            "actual_path": [point_data(point) for point in self.actual_trajectory],
        }
        encoded = json.dumps(payload, separators=(",", ":"))
        self.trajectory_pub.publish(String(data=encoded))
        if self.trajectory_web is not None:
            self.trajectory_web.update(encoded)
        self.last_trajectory_publish_time = now

    def log_status(self, current):
        camera_state = "已开启" if self.camera_ready() else "未开启"
        if current is None:
            rospy.loginfo_throttle(
                self.log_period_seconds,
                "%s: 相机=%s；等待机器人 TF；本节点不发布控制指令",
                NODE_NAME,
                camera_state,
            )
            return
        curve_length = self.line_curve_s[-1] if self.line_curve_s else 0.0
        fixed_length = (
            self.line_committed_curve_s[-1]
            if self.line_committed_curve_s else 0.0
        )
        rospy.loginfo_throttle(
            self.log_period_seconds,
            "%s: 相机=%s；状态=%s；位置=(%.2f, %.2f, %.2f)，"
            "航向=%.1f deg；固定/拟合=%.2f/%.2f m，确认=%d/%d，"
            "识别点(总/有效/使用)=%d/%d/%d，有效帧/拒绝帧=%d/%d；无控制输出",
            NODE_NAME,
            camera_state,
            self.state,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(yaw_from_quaternion(current.pose.orientation)),
            fixed_length,
            curve_length,
            self.freeze_confirmation_count,
            self.curve_freeze_required_frames,
            self.latest_line_input_count,
            self.latest_line_valid_count,
            self.latest_line_used_count,
            self.accepted_triplets,
            self.rejected_triplets,
        )

    def run(self):
        rospy.loginfo(
            "%s: 启动纯拟合观察；请手动控制机器人，本节点不会发布运动命令",
            NODE_NAME,
        )
        while not rospy.is_shutdown():
            self.try_lock_line()
            current = self.get_current_pose()
            if current is not None:
                self.record_actual_trajectory(current)
            self.log_status(current)
            self.publish_trajectory_status(current)
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME)
    LineFittingTest().run()
