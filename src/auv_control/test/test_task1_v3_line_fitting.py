#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v3_line_fitting.py
功能：Task1 红线曲线拟合观察工具，只处理感知数据，不发布运动控制命令。

流程：
    1. 监听相机红线多点识别消息，将 positions 中坐标有限的点从 camera 变换到 map；
    2. conf >= 0.70 时保留全部有限识别点，密集近邻点不再合并压缩；
    3. 首次短时间窗内累计所有高置信识别点并锁定红线；
    4. 后续 conf >= 0.70 的 positions 全部点进入点集并重新拟合；
    5. 当前拟合经后续高置信帧确认后立即冻结；
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
2026.7.20
    增加置信度硬下限、单向连续性检查和自动降阶拟合；出现长跳变、反向
    延伸或近 180 度回折的曲线不允许冻结，Web 曲线直接使用 base_link 规划坐标。
2026.7.20
    红线一帧只查询一次原时间戳 TF，并用同一变换批量转换全部识别点；
    过期帧、零时间戳和原时刻 TF 不可用的帧直接拒绝，订阅队列缩短为 1。
    曲线关联改为多点比例、中位数、最大距离和末端局部切向联合判断；
    点融合与拟合改在临时状态中试算，全部验证通过后才原子提交。
2026.7.20
    根据高密度实测点集放宽预处理：近邻点交由融合函数合并，不再因点距过小、
    消息数组局部弯角或弦线残差拒绝整帧。高置信普通关联只检查点到现有曲线
    的距离统计；只有末端延伸继续使用局部 PCA 方向，保留误线隔离能力。
2026.7.20
    Web 增加历史识别点图层；高置信且成功转换到 map 的 positions 全部点会
    持续保留。显示历史支持独立的空间合并距离和数量上限，不影响曲线拟合。
2026.7.20
    根据实测确认 conf >= 0.70 的红线点均可信，停用曲线关联、内点比例、延伸
    方向、固定段过滤和曲线回折等几何拒绝条件；所有达标点直接融合并重新拟合。
    进一步取消 point_valid、坐标合理性和每帧取点上限；置信度达标后直接使用
    positions 数组的全部点。密集点融合距离默认改为 0，避免正常近邻点被压缩。
2026.7.20
    跳过 positions 中无法参与数值计算的 NaN/Inf 点，其余高置信点仍全部用于拟合；
    Web 历史点增加非有限值保护，避免显示处理打断感知回调。
2026.7.20
    增加带时间戳的 YAML 数据文件，记录每帧识别结果及定期发布的完整拟合快照。
"""

import copy
from collections import deque
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
from task1_v3_yaml_logger import TimestampedYamlLogger
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
<span>浅粉点：历史有效识别点</span><span>亮粉点：最新有效识别点</span><span>滚轮缩放，拖动平移</span></div></header>
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
function draw(d){last=d;let history=d.valid_line_point_history||[],latest=d.latest_line_points||[];x.clearRect(0,0,c.width,c.height);grid();
history.forEach(v=>dot(v,'rgba(255,45,145,.32)',2));(d.accepted_points||[]).forEach(v=>dot(v,'#879aa3',3));line(d.fitted_curve,'#e74c3c',3);line(d.fixed_curve,'#f39c12',5);line(d.actual_path,'#1677ff',3);
latest.forEach((v,i)=>{dot(v,'#ff2d91',4);if(i===0||i===latest.length-1||i===Math.floor(latest.length/2))tag(v,`P${i+1}`,'#ff2d91')});
dot(d.robot,'#00cfe8',8);bodyArrow(d.robot,d.camera);
document.getElementById('s').textContent=`状态 ${d.state}　相机 ${d.camera_ready?'已开启':'未开启'}　base航向 ${(d.robot_yaw_deg||0).toFixed(1)}°　D(base/camera) ${(d.robot_down??0).toFixed(2)}/${(d.camera_down??0).toFixed(2)} m　识别点 总/有效/使用 ${d.line_input_point_count||0}/${d.line_valid_point_count||0}/${d.line_used_point_count||0}　历史 ${d.valid_line_point_history_count||0}　锁线 ${d.line_locked?'是':'否'}　固定/拟合 ${d.fixed_length||0}/${d.curve_length||0} m　确认 ${d.freeze_confirmations||0}/${d.freeze_required||0}　有效帧/拒绝帧 ${d.accepted_triplets||0}/${d.rejected_triplets||0}`}
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
        self.camera_topic = rospy.get_param("~camera_topic", "/left/image_raw")
        self.trajectory_topic = rospy.get_param(
            "~trajectory_topic", "/task1/v3/line_fitting/trajectory"
        )
        self.tf_timeout_seconds = max(0.0, float(rospy.get_param(
            "~tf_timeout_seconds", 0.1
        )))
        self.line_message_max_age_seconds = max(0.0, float(rospy.get_param(
            "~line_message_max_age_seconds", 0.5
        )))
        self.camera_message_timeout = max(0.0, float(rospy.get_param(
            "~camera_message_timeout", 2.0
        )))
        self.log_period_seconds = max(0.1, float(rospy.get_param(
            "~log_period_seconds", 2.0
        )))
        self.log_directory = rospy.get_param(
            "~log_directory", "~/.ros/auv_logs/task1"
        )

        self.line_classes = class_names("~line_classes", ["line"])
        self.line_lock_window_seconds = max(0.0, float(rospy.get_param(
            "~line_lock_window_seconds", 0.3
        )))
        self.line_min_confidence = clamp(float(rospy.get_param(
            "~line_min_confidence", 0.70
        )), 0.0, 1.0)
        self.line_point_merge_distance = max(0.0, float(rospy.get_param(
            "~line_point_merge_distance", 0.0
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
        self.line_curve_sample_count = max(3, int(rospy.get_param(
            "~line_curve_sample_count", 100
        )))
        self.line_curve_degree = max(1, int(rospy.get_param(
            "~line_curve_degree", 3
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
        self.curve_freeze_overlap_distance = max(0.0, float(rospy.get_param(
            "~curve_freeze_overlap_distance", 0.25
        )))

        self.actual_path_min_spacing = max(0.001, float(rospy.get_param(
            "~actual_path_min_spacing", 0.02
        )))
        self.actual_path_max_points = max(10, int(rospy.get_param(
            "~actual_path_max_points", 2000
        )))
        self.web_valid_point_merge_distance = max(0.0, float(rospy.get_param(
            "~web_valid_point_merge_distance", 0.01
        )))
        self.web_valid_point_max_count = max(100, int(rospy.get_param(
            "~web_valid_point_max_count", 20000
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
        self.curve_lock = threading.Lock()
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
        self.latest_line_points = []
        self.latest_line_input_count = 0
        self.latest_line_valid_count = 0
        self.latest_line_used_count = 0
        self.latest_confidence = 0.0
        self.accepted_triplets = 0
        self.rejected_triplets = 0
        self.last_reject_reason = "none"
        self.actual_trajectory = []
        self.web_valid_point_history = deque()
        self.web_valid_point_cells = set()
        self.web_valid_point_lock = threading.Lock()
        self.last_trajectory_publish_time = rospy.Time(0)
        self.data_logger = None
        self.open_data_log()
        rospy.on_shutdown(self.shutdown)

        rospy.Subscriber(
            self.line_topic, LineDetection, self.line_callback, queue_size=1
        )
        rospy.Subscriber(
            self.camera_topic, rospy.AnyMsg, self.camera_callback, queue_size=1
        )
        rospy.loginfo(
            "%s: 红线接口=auv_control/LineDetection；置信度达标后使用 positions 全部有限点；下限=%.2f",
            NODE_NAME,
            self.line_min_confidence,
        )

    def open_data_log(self):
        try:
            self.data_logger = TimestampedYamlLogger(
                NODE_NAME, self.log_directory
            )
            self.write_data_record(
                "startup",
                log_directory=self.log_directory,
                line_min_confidence=self.line_min_confidence,
            )
            rospy.loginfo(
                "%s: 完整数据文件=%s", NODE_NAME, self.data_logger.path
            )
        except OSError as error:
            self.data_logger = None
            rospy.logwarn("%s: 无法创建完整数据文件: %s", NODE_NAME, error)

    def shutdown(self):
        if self.data_logger is not None:
            self.data_logger.close()
            self.data_logger = None

    def write_data_record(self, event, **data):
        if self.data_logger is None:
            return
        data.setdefault("state", self.state)
        try:
            self.data_logger.write(event, **data)
        except (OSError, TypeError, ValueError) as error:
            rospy.logwarn_throttle(
                5.0, "%s: 完整数据写入失败: %s", NODE_NAME, error
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

    def tracking_curve_to_base_points(self, points):
        """拟合曲线现在直接作为 base_link 规划轨迹显示。"""
        return [copy.deepcopy(point) for point in points]

    def line_message_time_reason(self, stamp):
        """拒绝无时间戳、过期或明显来自未来的感知帧。"""
        if stamp == rospy.Time(0):
            return "line_timestamp_missing"
        age = (rospy.Time.now() - stamp).to_sec()
        if age < -0.1:
            return "line_timestamp_in_future"
        if age > self.line_message_max_age_seconds:
            return "line_message_stale"
        return None

    def transform_frame_to_map(self, poses):
        """同一视觉帧只查询一次 TF，并用同一刚体变换转换全部点。"""
        if not poses:
            return None, "empty_line_frame"
        source_frame = poses[0].header.frame_id
        stamp = poses[0].header.stamp
        try:
            self.tf_listener.waitForTransform(
                self.map_frame,
                source_frame,
                stamp,
                rospy.Duration(self.tf_timeout_seconds),
            )
            translation, rotation = self.tf_listener.lookupTransform(
                self.map_frame, source_frame, stamp
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0, "%s: 红线原时间戳 TF 不可用: %s", NODE_NAME, error
            )
            return None, "tf_unavailable_at_frame_stamp"

        matrix = tf.transformations.quaternion_matrix(rotation)
        matrix[0:3, 3] = np.asarray(translation, dtype=float)
        transformed = []
        for pose in poses:
            if (
                pose.header.frame_id != source_frame
                or pose.header.stamp != stamp
            ):
                return None, "line_frame_header_mismatch"
            source = pose.pose.position
            target = np.dot(matrix, np.array([
                source.x, source.y, source.z, 1.0
            ], dtype=float))
            output = PoseStamped()
            output.header.frame_id = self.map_frame
            output.header.stamp = stamp
            output.pose.position = Point(
                float(target[0]), float(target[1]), float(target[2])
            )
            output.pose.orientation.w = 1.0
            transformed.append(output)
        stale_reason = self.line_message_time_reason(stamp)
        if stale_reason is not None:
            return None, stale_reason
        return transformed, "valid"

    def ordered_line_points(self, poses, reference):
        """按任务起点统一点列方向，不判断识别点是否合理。"""
        points = [copy.deepcopy(pose.pose.position) for pose in poses]
        if xy_distance(points[-1], reference) < xy_distance(points[0], reference):
            points.reverse()
        return points

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

    def freeze_confirmed_curve(self):
        """把经连续高置信帧确认的当前拟合曲线整体冻结。"""
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

    @staticmethod
    def fit_state_fields():
        """拟合试算成功后需要一次性提交的全部可变字段。"""
        return (
            "line_reference_point",
            "line_axis",
            "line_raw_points",
            "line_curve_points",
            "line_curve_s",
            "line_start_point",
            "line_end_point",
            "line_fit_residual",
            "last_reject_reason",
        )

    def reset_tentative_line_state(self):
        """首次锁线失败后清除全部暂定点集、曲线和 PCA 方向。"""
        with self.curve_lock:
            self.line_reference_point = None
            self.line_axis = None
            self.line_raw_points = []
            self.line_curve_points = []
            self.line_curve_s = []
            self.line_start_point = None
            self.line_end_point = None
            self.line_fit_residual = 0.0
            self.freeze_confirmation_count = 0

    def fit_points_transaction(self, points, reference=None, fresh=False):
        """在隔离的临时状态中融合和拟合，通过后才提交正式状态。"""
        trial = copy.copy(self)
        for field in self.fit_state_fields():
            setattr(trial, field, copy.deepcopy(getattr(self, field)))
        if fresh:
            trial.line_reference_point = copy.deepcopy(reference)
            trial.line_axis = None
            trial.line_raw_points = []
            trial.line_curve_points = []
            trial.line_curve_s = []
            trial.line_start_point = None
            trial.line_end_point = None
            trial.line_fit_residual = 0.0
            trial.last_reject_reason = "curve_fit_failed"
            trial.line_locked = False
        elif reference is not None:
            trial.line_reference_point = copy.deepcopy(reference)

        trial.fuse_line_points(points)
        if not trial.fit_line_curve():
            return False, trial.last_reject_reason or "curve_fit_failed"

        with self.curve_lock:
            for field in self.fit_state_fields():
                setattr(self, field, copy.deepcopy(getattr(trial, field)))
        return True, "none"

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
        # 所有高置信点均视为合理；这里只排除无法形成曲线的数值零跨度。
        if span <= 1e-6:
            self.last_reject_reason = "curve_span_too_small"
            return False

        samples = np.linspace(
            float(parameters[0]),
            float(parameters[-1]),
            self.line_curve_sample_count,
        )
        local_start = min(
            self.line_raw_points,
            key=lambda point: xy_distance(point, self.line_reference_point),
        )
        selected_start = (
            copy.deepcopy(local_start)
            if self.line_start_point is None or not self.line_locked
            else copy.deepcopy(self.line_start_point)
        )
        mean_z = sum(point.z for point in self.line_raw_points) / len(
            self.line_raw_points
        )
        curve = None
        selected_axis = axis
        requested_degree = min(
            self.line_curve_degree, len(self.line_raw_points) - 1
        )
        for degree in range(requested_degree, 0, -1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x_model = np.poly1d(np.polyfit(
                        parameters, ordered[:, 0], degree
                    ))
                    y_model = np.poly1d(np.polyfit(
                        parameters, ordered[:, 1], degree
                    ))
                candidate = [
                    Point(
                        float(x_model(value)),
                        float(y_model(value)),
                        mean_z,
                    )
                    for value in samples
                ]
            except (TypeError, ValueError, np.linalg.LinAlgError):
                continue

            candidate_axis = axis
            if xy_distance(candidate[-1], selected_start) < xy_distance(
                candidate[0], selected_start
            ):
                candidate.reverse()
                candidate_axis = -axis
            candidate_end = max(
                self.line_raw_points,
                key=lambda point: (
                    (point.x - center[0]) * candidate_axis[0]
                    + (point.y - center[1]) * candidate_axis[1]
                ),
            )
            candidate[0] = copy.deepcopy(selected_start)
            candidate[-1] = copy.deepcopy(candidate_end)
            if self.line_committed_curve_points:
                committed = [
                    copy.deepcopy(point)
                    for point in self.line_committed_curve_points
                ]
                join_index = min(
                    range(len(candidate)),
                    key=lambda index: xy_distance(
                        committed[-1], candidate[index]
                    ),
                )
                suffix = [
                    copy.deepcopy(point) for point in candidate[join_index:]
                ]
                if (
                    suffix
                    and xy_distance(committed[-1], suffix[0])
                    <= self.line_point_merge_distance
                ):
                    suffix = suffix[1:]
                candidate = committed + suffix
            candidate[0] = copy.deepcopy(selected_start)
            # 实测输入已由置信度保证，本处不再以回退、转角、间距等几何条件拒绝。
            curve = candidate
            selected_axis = candidate_axis
            break

        if curve is None:
            self.last_reject_reason = "curve_fit_failed"
            return False
        self.line_axis = selected_axis
        self.line_start_point = selected_start
        self.line_curve_points = curve
        self.line_curve_s = self.cumulative_distance(curve)
        if self.line_curve_s[-1] <= 1e-6:
            self.last_reject_reason = "curve_span_too_small"
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

    def line_poses(self, message):
        """置信度达标后，将 positions 中坐标有限的识别点全部用于拟合。"""
        declared_count = int(message.point_count)
        positions = list(message.positions)
        self.latest_line_input_count = declared_count
        self.latest_line_valid_count = int(message.valid_count)
        self.latest_line_used_count = 0

        if declared_count <= 0:
            return None, "empty_line_message"
        if declared_count != len(positions):
            return None, "line_array_size_mismatch"
        if not message.header.frame_id:
            return None, "line_frame_id_empty"

        poses = []
        for position in positions:
            if not all(math.isfinite(value) for value in (
                position.x, position.y, position.z
            )):
                continue
            pose = PoseStamped()
            pose.header = copy.deepcopy(message.header)
            pose.pose.position = copy.deepcopy(position)
            pose.pose.orientation.w = 1.0
            poses.append(pose)

        if len(poses) < 2:
            return None, "too_few_finite_line_points"
        self.latest_line_used_count = len(poses)
        return poses, "valid"

    def retain_web_valid_points(self, points):
        """累计 Web 有效识别点；显示用去重和限长不参与拟合判断。"""
        spacing = self.web_valid_point_merge_distance
        with self.web_valid_point_lock:
            for point in points:
                if not all(math.isfinite(value) for value in (
                    point.x, point.y, point.z
                )):
                    continue
                cell = None
                if spacing > 0.0:
                    cell = (
                        int(round(point.x / spacing)),
                        int(round(point.y / spacing)),
                    )
                    if cell in self.web_valid_point_cells:
                        continue
                while len(self.web_valid_point_history) >= self.web_valid_point_max_count:
                    _, expired_cell = self.web_valid_point_history.popleft()
                    if expired_cell is not None:
                        self.web_valid_point_cells.discard(expired_cell)
                self.web_valid_point_history.append((copy.deepcopy(point), cell))
                if cell is not None:
                    self.web_valid_point_cells.add(cell)

    def web_valid_points_snapshot(self):
        """返回 Web 历史点的一致快照，避免回调追加时序列发生变化。"""
        with self.web_valid_point_lock:
            return [copy.deepcopy(item[0]) for item in self.web_valid_point_history]

    def reject_triplet(self, reason):
        self.rejected_triplets += 1
        self.last_reject_reason = reason
        self.write_data_record(
            "line_frame",
            status="rejected",
            reason=reason,
            confidence=round(self.latest_confidence, 6),
            input_count=self.latest_line_input_count,
            valid_count=self.latest_line_valid_count,
            used_count=self.latest_line_used_count,
        )
        rospy.loginfo_throttle(
            self.log_period_seconds,
            "%s: 本帧红线已忽略，原因=%s",
            NODE_NAME,
            reason,
        )

    def line_callback(self, message):
        if message.class_name and message.class_name not in self.line_classes:
            return
        self.latest_line_input_count = int(message.point_count)
        self.latest_line_valid_count = int(message.valid_count)
        self.latest_line_used_count = 0
        confidence = self.normalized_confidence(message.conf)
        self.latest_confidence = confidence
        time_reason = self.line_message_time_reason(message.header.stamp)
        if time_reason is not None:
            self.reject_triplet(time_reason)
            return
        if confidence < self.line_min_confidence:
            self.reject_triplet("confidence_below_minimum")
            return
        camera_poses, reason = self.line_poses(message)
        if camera_poses is None:
            self.reject_triplet(reason)
            return

        transformed, reason = self.transform_frame_to_map(camera_poses)
        if transformed is None:
            self.reject_triplet(reason)
            return
        self.latest_line_points = [
            copy.deepcopy(pose.pose.position) for pose in transformed
        ]
        self.retain_web_valid_points(self.latest_line_points)

        current = self.get_current_pose()
        if current is None:
            self.reject_triplet("robot_pose_unavailable")
            return
        reference = (
            self.line_reference_point
            if self.line_reference_point is not None
            else current.pose.position
        )
        points = self.ordered_line_points(transformed, reference)

        now = rospy.Time.now()
        if not self.line_locked:
            if self.line_lock_candidate is None:
                self.line_lock_candidate = {
                    "started": now,
                    "confidence": confidence,
                    "points": points,
                    "reference": copy.deepcopy(current.pose.position),
                }
            else:
                self.line_lock_candidate["confidence"] = max(
                    self.line_lock_candidate["confidence"], confidence
                )
                self.line_lock_candidate["points"].extend(
                    copy.deepcopy(points)
                )
            self.state = self.SELECT_CANDIDATE
            self.write_data_record(
                "line_frame",
                status="lock_candidate",
                confidence=round(confidence, 6),
                input_count=self.latest_line_input_count,
                valid_count=self.latest_line_valid_count,
                used_count=self.latest_line_used_count,
                points=[
                    [point.x, point.y, point.z] for point in points
                ],
            )
            return

        # conf 已通过硬下限后，positions 全部点直接进入融合和拟合；旧的曲线关联、
        # 内点比例、延伸方向及固定段重复点过滤均按实测要求停用。
        fitted, fit_reason = self.fit_points_transaction(points)
        if fitted:
            self.accepted_triplets += 1
            self.last_reject_reason = "none"
            self.freeze_confirmation_count = min(
                self.curve_freeze_required_frames,
                self.freeze_confirmation_count + 1,
            )
            self.freeze_confirmed_curve()
            self.write_data_record(
                "line_frame",
                status="accepted",
                confidence=round(confidence, 6),
                input_count=self.latest_line_input_count,
                valid_count=self.latest_line_valid_count,
                used_count=self.latest_line_used_count,
                points=[
                    [point.x, point.y, point.z] for point in points
                ],
                curve_length=(
                    self.line_curve_s[-1] if self.line_curve_s else 0.0
                ),
                fixed_length=(
                    self.line_committed_curve_s[-1]
                    if self.line_committed_curve_s else 0.0
                ),
                freeze_version=self.freeze_version,
            )
        else:
            self.freeze_confirmation_count = 0
            self.reject_triplet(fit_reason)

    def try_lock_line(self):
        if self.line_locked or self.line_lock_candidate is None:
            return False
        candidate = self.line_lock_candidate
        if (
            rospy.Time.now() - candidate["started"]
        ).to_sec() < self.line_lock_window_seconds:
            return False
        fitted, fit_reason = self.fit_points_transaction(
            candidate["points"], reference=candidate["reference"], fresh=True
        )
        if not fitted:
            self.line_lock_candidate = None
            self.reset_tentative_line_state()
            self.state = self.WAIT_LINE
            self.reject_triplet(fit_reason)
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
            "当前曲线长度=%.2f m，后续高置信 positions 全部点参与拟合",
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
        web_valid_points = self.web_valid_points_snapshot()

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
            "valid_line_point_history": [
                point_data(point) for point in web_valid_points
            ],
            "valid_line_point_history_count": len(web_valid_points),
            "actual_path": [point_data(point) for point in self.actual_trajectory],
        }
        self.write_data_record("trajectory_update", **payload)
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
