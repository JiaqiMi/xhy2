#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v3_line_follow.py
功能：Task1 巡线单项测试，运动接口迁移到 motion_supervisor。

流程：
    1. 记录节点启动时的当前位置、高度和航向，定点等待相机和识别模块；
    2. 未识别红线时，依次定点左转、右转、回正，再向前移动后重复；
    3. 在短时间窗内选择置信度最高的第一条红线并锁定；
    4. 仅接收能与已锁轨迹合理关联的新点，后续合理帧确认后立即冻结曲线段；
    5. LOS 选择 map 前视目标，经 motion_supervisor 前往曲线起点并巡线；
    6. 起点定点稳定后先对准曲线方向，再只沿已固定曲线巡线；
    7. 到达当前最远点后定点等待；同一最远点被连续多次观测且轨迹不再
       增长时，才将其确认为真实终点并结束测试。

起终点定义：
    以首次锁线时的机器人位置为固定参考点。所有已接受点中，距参考点
    最近的点记为红线起点，最远的点记为红线终点。参考点不会随机器人
    运动而改变，因此起终点不会在巡线过程中反转。

监听：/obj/line_message，/left/image_raw，/motion/state，/tf
发布：/cmd/motion/goal，/cmd/motion/cancel，/task1/v3/trajectory，/finished
网页：默认 http://192.168.1.117:8082

记录：
2026.7.14
    初版，用于单独验证 Task1 巡线通信与控制流程。
2026.7.16
    增加启动等待、搜索、曲线拟合、速度稳定和网页轨迹显示。
2026.7.17
    取消人工初始航向，启动航向改为机器人节点启动时的实际航向。
    搜索方式与图形子任务一致：定点前进后仅用 TY 左右横移。
    重构为“锁线、LOS 到起点、起点定点、对向、LOS 巡线、终点确认”。
    第一条线按短时间窗内最高置信度锁定，锁定后不再返回搜索状态。
    P1/P2/P3 不再解释为整条线端点，改为三点重排、几何验证和曲线关联。
    低于 0.70 置信度且难以拟合的点直接丢弃。
    当前最远点只作为临时巡线目标；连续多帧仍为同一最远点后才形成终点证据。
    原始点集改为滑动窗口；旧点超限后冻结历史曲线，只用远端重叠点和新点拟合后缀。
2026.7.18
    启动阶段增加可调的 mode=4 定点悬停时间。
    未识别红线时改为定点左转、右转、回到启动航向，再定点前进后循环搜索。
    轨迹网页增加机器人航向箭头、当前跟踪点和最新识别 P1/P2/P3。
    将持续更新的拟合曲线与 LOS 当前执行曲线分离；执行曲线走完前不随识别帧改变，
    到达当前曲线末端后才接入已经拟合出的新增路径，避免跟踪点前后跳变。
    取消切入 mode=4 定点前由任务代码逐步卸载 TX，交由新版定点控制器自动刹车。
    适当放宽首次红线锁定的默认几何门槛，所有门槛仍由 launch 参数配置。
    曲线改为后续帧确认后立即分段冻结；固定段上的重复识别点不再参与拟合。
    LOS 只跟踪已固定曲线，走完当前固定快照后才接入新固定段。
    新增 v3：搜索、起点、对向、巡线和终点全部改发 map 绝对目标；LOS
    不再直接输出 TX/MZ，相邻目标经位置和 yaw 步长限制后交由运动监督器。
    将曲线点融合权重、实际轨迹记录间距和最大保存点数开放为 launch 参数，
    便于实测时调整拟合平滑程度和 Web 轨迹数据量。
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
from auv_control.msg import MotionState, TargetDetection3
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Empty, String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task1_v3_line_follow"


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
<title>Task1 巡线测试轨迹</title><style>
body{margin:0;background:#101820;color:#eef;font-family:Arial,"Microsoft YaHei"}
header{padding:12px 18px;background:#172630}canvas{display:block;margin:16px auto;background:#f7fbfd;border-radius:8px}
#s{margin-left:25px;color:#bcd0d8}.legend{font-size:13px;color:#bcd0d8;margin-top:8px}.legend span{margin-right:18px}
</style></head><body><header><b>Task1 巡线测试轨迹</b><span id="s">等待数据</span>
<div class="legend"><span>蓝线：实际轨迹</span><span>红线：实时拟合轨迹</span><span>橙线：已确认固定轨迹</span><span>绿线：LOS 当前执行轨迹</span><span>青色箭头：机器人航向</span>
<span>紫点：当前跟踪点</span><span>粉/橙/黄：最新 P1/P2/P3</span></div></header>
<canvas id="c" width="960" height="680"></canvas><script>
const c=document.getElementById('c'),x=c.getContext('2d'),p=45;let k=1,mx=0,my=0;
function q(a){return[p+(a[0]-mx)*k,c.height-p-(a[1]-my)*k]}
function line(a,col,w){if(!a||a.length<2)return;x.beginPath();a.forEach((v,i)=>{let z=q(v);i?x.lineTo(...z):x.moveTo(...z)});x.strokeStyle=col;x.lineWidth=w;x.stroke()}
function dot(a,col,r){if(!a)return;let z=q(a);x.beginPath();x.arc(z[0],z[1],r,0,7);x.fillStyle=col;x.fill()}
function tag(a,text,col){if(!a)return;let z=q(a);x.fillStyle=col;x.font='bold 13px Arial';x.fillText(text,z[0]+7,z[1]-7)}
function arrow(a,yawDeg){if(!a)return;let z=q(a),r=yawDeg*Math.PI/180,L=34,ex=z[0]+L*Math.cos(r),ey=z[1]-L*Math.sin(r),h=9;
x.beginPath();x.moveTo(z[0],z[1]);x.lineTo(ex,ey);x.strokeStyle='#00a9c7';x.lineWidth=4;x.stroke();
x.beginPath();x.moveTo(ex,ey);x.lineTo(ex-h*Math.cos(r-.55),ey+h*Math.sin(r-.55));x.moveTo(ex,ey);x.lineTo(ex-h*Math.cos(r+.55),ey+h*Math.sin(r+.55));x.stroke()}
function draw(d){let latest=d.latest_line_points||[],a=[...(d.actual_path||[]),...(d.planned_curve||[]),...(d.fixed_curve||[]),...(d.tracking_curve||[]),...(d.raw_line||[]),...latest];
if(d.robot)a.push(d.robot);if(d.tracking_point)a.push(d.tracking_point);if(d.line_start)a.push(d.line_start);if(d.line_end)a.push(d.line_end);if(d.endpoint_candidate)a.push(d.endpoint_candidate);
x.clearRect(0,0,c.width,c.height);if(!a.length)return;let xs=a.map(v=>v[0]),ys=a.map(v=>v[1]),xx=Math.max(...xs)+.2,yy=Math.max(...ys)+.2;
mx=Math.min(...xs)-.2;my=Math.min(...ys)-.2;k=Math.min((c.width-2*p)/Math.max(.5,xx-mx),(c.height-2*p)/Math.max(.5,yy-my));
(d.raw_line||[]).forEach(v=>dot(v,'#879aa3',2));line(d.planned_curve,'#e74c3c',3);line(d.fixed_curve,'#f39c12',5);line(d.tracking_curve,'#21a366',4);line(d.actual_path,'#1677ff',3);
dot(d.line_start,'#21a366',7);dot(d.line_end,'#f39c12',7);dot(d.endpoint_candidate,'#8e44ad',5);dot(d.tracking_point,'#9b2cff',7);tag(d.tracking_point,'跟踪点','#6f13ba');
['#ff2d91','#ff8c1a','#ffd000'].forEach((col,i)=>{if(latest[i]){dot(latest[i],col,6);tag(latest[i],`P${i+1}`,col)}});
dot(d.robot,'#00cfe8',8);arrow(d.robot,d.robot_yaw_deg||0);
document.getElementById('s').textContent=`任务 ${d.state}　监督器 ${d.motion_state??'-'}　航向 ${(d.robot_yaw_deg||0).toFixed(1)}°　锁线 ${d.line_locked?'是':'否'}　固定/拟合 ${d.fixed_length||0}/${d.fitted_length||0} m　执行/固定版本 ${d.tracking_curve_version||0}/${d.fixed_curve_version||0}　确认 ${d.freeze_confirmations||0}/${d.freeze_required||0}　已完成 ${d.completed_length||0} m　终点证据 ${d.endpoint_stable_count||0}/${d.endpoint_stable_required||0}`}
async function t(){try{draw(await(await fetch('/data',{cache:'no-store'})).json())}catch(e){}setTimeout(t,500)}t();
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


class Task1LineFollowTest:
    """只测试 Task1 红线搜索和巡线。"""

    WAIT_CAMERA = "WAIT_CAMERA"
    SEARCH_LEFT = "SEARCH_LEFT"
    SEARCH_RIGHT = "SEARCH_RIGHT"
    SEARCH_RETURN = "SEARCH_RETURN"
    SEARCH_FORWARD = "SEARCH_FORWARD"
    WAIT_FIXED_LINE = "WAIT_FIXED_LINE"
    GO_TO_START = "GO_TO_START"
    HOLD_START = "HOLD_START"
    ALIGN_LINE = "ALIGN_LINE"
    FOLLOW_LINE = "FOLLOW_LINE"
    HOLD_END = "HOLD_END"
    FINISH = "FINISH"

    SEARCH_STATES = {
        SEARCH_LEFT,
        SEARCH_RIGHT,
        SEARCH_RETURN,
        SEARCH_FORWARD,
    }

    def __init__(self):
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))
        self.tf_listener = tf.TransformListener()

        self.motion_goal_topic = rospy.get_param(
            "~motion_goal_topic", "/cmd/motion/goal"
        )
        self.motion_cancel_topic = rospy.get_param(
            "~motion_cancel_topic", "/cmd/motion/cancel"
        )
        self.motion_state_topic = rospy.get_param(
            "~motion_state_topic", "/motion/state"
        )
        self.motion_state_timeout = max(0.1, float(rospy.get_param(
            "~motion_state_timeout", 0.5
        )))
        self.motion_goal_position_tolerance = max(0.001, float(rospy.get_param(
            "~motion_goal_position_tolerance", 0.05
        )))
        self.motion_goal_yaw_tolerance = math.radians(max(
            0.1,
            float(rospy.get_param("~motion_goal_yaw_tolerance_deg", 3.0)),
        ))
        self.line_topic = rospy.get_param("~line_topic", "/obj/line_message")
        self.finished_topic = rospy.get_param("~finished_topic", "/finished")
        self.camera_topic = rospy.get_param("~camera_topic", "/left/image_raw")
        self.trajectory_topic = rospy.get_param(
            "~trajectory_topic", "/task1/v3/trajectory"
        )

        # 未识别红线时先定点左右转，再回正并定点向前移动。
        self.search_forward_distance = float(rospy.get_param(
            "~search_forward_distance", 0.5
        ))
        self.search_yaw_angle = math.radians(float(rospy.get_param(
            "~search_yaw_angle_deg", 30.0
        )))
        self.search_yaw_sign = 1.0 if float(rospy.get_param(
            "~search_yaw_sign", 1.0
        )) >= 0.0 else -1.0
        self.search_yaw_tolerance = math.radians(float(rospy.get_param(
            "~search_yaw_tolerance_deg", 5.0
        )))
        self.search_yaw_hold_seconds = max(0.0, float(rospy.get_param(
            "~search_yaw_hold_seconds", 2.0
        )))
        self.search_return_hold_seconds = max(0.0, float(rospy.get_param(
            "~search_return_hold_seconds", 2.0
        )))
        self.search_forward_hold_seconds = max(0.0, float(rospy.get_param(
            "~search_forward_hold_seconds", 4.0
        )))
        self.position_tolerance = float(rospy.get_param(
            "~position_tolerance", 0.15
        ))

        # 相机和任务阶段等待时间；静止与接管统一由 MotionState.HOVER 确认。
        self.camera_message_timeout = float(rospy.get_param(
            "~camera_message_timeout", 2.0
        ))
        self.startup_hold_seconds = max(0.0, float(rospy.get_param(
            "~startup_hold_seconds", 10.0
        )))
        self.transition_hold_seconds = float(rospy.get_param(
            "~transition_hold_seconds", 4.0
        ))

        # 红线首帧选择、局部三点验证和误识别隔离参数。
        self.line_classes = class_names("~line_classes", ["line"])
        self.max_camera_distance = float(rospy.get_param(
            "~max_camera_distance", 6.0
        ))
        self.line_lock_window_seconds = float(rospy.get_param(
            "~line_lock_window_seconds", 0.3
        ))
        self.line_low_confidence_threshold = float(rospy.get_param(
            "~line_low_confidence_threshold", 0.50
        ))
        self.line_min_point_spacing = float(rospy.get_param(
            "~line_min_point_spacing", 0.005
        ))
        self.line_max_point_spacing = float(rospy.get_param(
            "~line_max_point_spacing", 4.0
        ))
        self.line_triplet_max_residual = float(rospy.get_param(
            "~line_triplet_max_residual", 0.35
        ))
        self.line_triplet_max_bend = math.radians(float(rospy.get_param(
            "~line_triplet_max_bend_deg", 60.0
        )))
        self.line_association_distance = float(rospy.get_param(
            "~line_association_distance", 0.35
        ))
        self.line_high_confidence_association_distance = float(rospy.get_param(
            "~line_high_confidence_association_distance", 0.50
        ))
        self.line_association_angle = math.radians(float(rospy.get_param(
            "~line_association_angle_deg", 35.0
        )))
        self.line_high_confidence_association_angle = math.radians(float(
            rospy.get_param("~line_high_confidence_association_angle_deg", 45.0)
        ))
        self.line_extension_max_gap = float(rospy.get_param(
            "~line_extension_max_gap", 1.0
        ))

        # 已接受点集上限和平滑曲线拟合参数。
        self.line_point_merge_distance = float(rospy.get_param(
            "~line_point_merge_distance", 0.12
        ))
        self.line_point_update_alpha = clamp(float(rospy.get_param(
            "~line_point_update_alpha", 0.20
        )), 0.0, 1.0)
        self.line_curve_max_points = max(3, int(rospy.get_param(
            "~line_curve_max_points", 200
        )))
        requested_overlap = int(rospy.get_param(
            "~line_curve_overlap_points", 40
        ))
        self.line_curve_overlap_points = max(
            2, min(requested_overlap, self.line_curve_max_points - 1)
        )
        self.line_window_backtrack_distance = float(rospy.get_param(
            "~line_window_backtrack_distance", 1.0
        ))
        self.line_curve_sample_count = max(3, int(rospy.get_param(
            "~line_curve_sample_count", 100
        )))
        self.line_curve_degree = max(1, int(rospy.get_param(
            "~line_curve_degree", 3
        )))
        self.line_curve_min_length = float(rospy.get_param(
            "~line_curve_min_length", 0.15
        ))
        self.curve_freeze_required_frames = max(1, int(rospy.get_param(
            "~curve_freeze_required_frames", 2
        )))
        self.curve_freeze_min_length = float(rospy.get_param(
            "~curve_freeze_min_length", 0.15
        ))
        self.curve_freeze_min_advance = float(rospy.get_param(
            "~curve_freeze_min_advance", 0.10
        ))
        self.curve_freeze_ignore_distance = float(rospy.get_param(
            "~curve_freeze_ignore_distance", 0.12
        ))
        self.curve_freeze_endpoint_guard = float(rospy.get_param(
            "~curve_freeze_endpoint_guard", 0.05
        ))
        self.curve_freeze_overlap_distance = float(rospy.get_param(
            "~curve_freeze_overlap_distance", 0.25
        ))
        self.endpoint_growth_tolerance = float(rospy.get_param(
            "~endpoint_growth_tolerance", 0.08
        ))

        # LOS 只产生连续的 map 绝对目标；相邻变化小于监督器抢占阈值。
        self.los_lookahead_distance = float(rospy.get_param(
            "~los_lookahead_distance", 0.6
        ))
        self.los_goal_max_step = max(0.01, float(rospy.get_param(
            "~los_goal_max_step", 0.20
        )))
        self.los_goal_max_yaw_step = math.radians(max(
            0.1,
            float(rospy.get_param("~los_goal_max_yaw_step_deg", 20.0)),
        ))
        self.line_start_hold_seconds = float(rospy.get_param(
            "~line_start_hold_seconds", 4.0
        ))
        self.line_alignment_hold_seconds = float(rospy.get_param(
            "~line_alignment_hold_seconds", 2.0
        ))
        self.endpoint_arrival_tolerance = float(rospy.get_param(
            "~endpoint_arrival_tolerance", 0.18
        ))
        self.endpoint_path_tolerance = float(rospy.get_param(
            "~endpoint_path_tolerance", 0.25
        ))
        self.endpoint_confirm_seconds = float(rospy.get_param(
            "~endpoint_confirm_seconds", 5.0
        ))
        self.endpoint_stable_frames = max(2, int(rospy.get_param(
            "~endpoint_stable_frames", 8
        )))
        self.endpoint_stable_position_tolerance = float(rospy.get_param(
            "~endpoint_stable_position_tolerance", 0.12
        ))
        self.endpoint_evidence_robot_distance = float(rospy.get_param(
            "~endpoint_evidence_robot_distance", 0.8
        ))

        self.trajectory_publish_period = float(rospy.get_param(
            "~trajectory_publish_period", 0.5
        ))
        self.actual_path_min_spacing = max(0.001, float(rospy.get_param(
            "~actual_path_min_spacing", 0.03
        )))
        self.actual_path_max_points = max(10, int(rospy.get_param(
            "~actual_path_max_points", 2000
        )))
        self.trajectory_web_enabled = bool(rospy.get_param(
            "~trajectory_web_enabled", True
        ))
        self.trajectory_web_host = rospy.get_param(
            "~trajectory_web_host", "192.168.1.117"
        )
        self.trajectory_web_port = int(rospy.get_param(
            "~trajectory_web_port", 8082
        ))

        self.motion_goal_pub = rospy.Publisher(
            self.motion_goal_topic, PoseStamped, queue_size=1
        )
        self.motion_cancel_pub = rospy.Publisher(
            self.motion_cancel_topic, Empty, queue_size=1
        )
        self.finished_pub = rospy.Publisher(self.finished_topic, String, queue_size=10)
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
                    "%s: 轨迹网页 http://%s:%d",
                    NODE_NAME,
                    self.trajectory_web_host,
                    self.trajectory_web_port,
                )
            except OSError as error:
                rospy.logwarn("%s: 轨迹网页启动失败: %s", NODE_NAME, error)

        self.state = self.WAIT_CAMERA
        self.start_pose = None
        self.hold_z = None
        self.search_base_yaw = None
        self.startup_hold_started = None
        self.last_camera_time = None
        self.latest_motion_state = None
        self.last_motion_goal = None
        self.last_los_goal = None
        self.cancel_sent = False
        rospy.on_shutdown(self.cancel_motion)

        self.search_target = None
        self.search_cycle_anchor = None

        self.line_lock_candidate = None
        self.curve_lock = threading.Lock()
        self.line_locked = False
        self.line_lock_confidence = 0.0
        self.line_reference_point = None
        self.line_axis = None
        self.line_raw_points = []
        self.line_committed_curve_points = []
        self.line_committed_curve_s = []
        self.line_curve_points = []
        self.line_curve_s = []
        # line_curve_* 持续接收识别点并重拟合；tracking_curve_* 是 LOS 当前
        # 正在执行的只读快照。快照走完前不更新，避免识别帧直接改变运动目标。
        self.tracking_curve_points = []
        self.tracking_curve_s = []
        self.tracking_curve_version = 0
        self.line_start_point = None
        self.line_end_point = None
        self.line_fit_residual = 0.0
        self.freeze_confirmation_count = 0
        self.ignored_fixed_points = 0
        self.confirmed_end_distance = -1.0
        self.line_version = 0
        self.initial_line_hold_pose = None

        self.hold_target = None
        self.stable_since = None
        self.current_path_s = 0.0
        self.completed_path_length = 0.0
        self.endpoint_hold_started = None
        self.endpoint_candidate_point = None
        self.endpoint_candidate_count = 0

        self.actual_trajectory = []
        self.current_tracking_point = None
        self.latest_line_points = []
        self.last_trajectory_publish_time = rospy.Time(0)

        # 状态字段全部就绪后再订阅，避免构造期间首帧回调访问未初始化字段。
        rospy.Subscriber(
            self.line_topic, TargetDetection3, self.line_callback, queue_size=10
        )
        rospy.Subscriber(
            self.camera_topic, rospy.AnyMsg, self.camera_callback, queue_size=1
        )
        rospy.Subscriber(
            self.motion_state_topic,
            MotionState,
            self.motion_state_callback,
            queue_size=1,
        )

    def set_state(self, state):
        if self.state != state:
            rospy.loginfo("%s: 阶段 %s -> %s", NODE_NAME, self.state, state)
        self.state = state
        self.stable_since = None

    def camera_callback(self, _message):
        self.last_camera_time = rospy.Time.now()

    def motion_state_callback(self, message):
        self.latest_motion_state = copy.deepcopy(message)

    def camera_ready(self):
        return (
            self.last_camera_time is not None
            and (rospy.Time.now() - self.last_camera_time).to_sec()
            <= self.camera_message_timeout
        )

    def get_current_pose(self):
        try:
            self.tf_listener.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(1.0)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0, "%s: 无法获取当前位姿: %s", NODE_NAME, error
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def initialize_start_pose(self):
        if self.start_pose is not None:
            return True
        current = self.get_current_pose()
        if current is None:
            return False
        self.start_pose = copy.deepcopy(current)
        self.hold_z = current.pose.position.z
        self.search_base_yaw = yaw_from_quaternion(current.pose.orientation)
        self.startup_hold_started = rospy.Time.now()
        rospy.loginfo(
            "%s: 当前位姿=(%.2f, %.2f, %.2f)，启动航向=%.1f deg",
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(self.search_base_yaw),
        )
        return True

    def make_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(x, y, self.hold_z)
        pose.pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw))
        return pose

    def publish_motion_goal(self, target):
        goal = copy.deepcopy(target)
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        self.last_motion_goal = copy.deepcopy(goal)
        self.motion_goal_pub.publish(goal)

    def publish_dprov(self, target):
        """兼容原状态机函数名；v3 实际发布 motion_supervisor 目标。"""
        self.publish_motion_goal(target)

    def publish_los_goal(self, point, desired_yaw):
        """平滑连续前视目标，避免触发监督器的大目标抢占刹停。"""
        target_x = point.x
        target_y = point.y
        target_yaw = desired_yaw
        if self.last_los_goal is not None:
            previous = self.last_los_goal
            dx = point.x - previous.pose.position.x
            dy = point.y - previous.pose.position.y
            distance = math.hypot(dx, dy)
            if distance > self.los_goal_max_step:
                ratio = self.los_goal_max_step / distance
                target_x = previous.pose.position.x + ratio * dx
                target_y = previous.pose.position.y + ratio * dy
            previous_yaw = yaw_from_quaternion(previous.pose.orientation)
            yaw_step = clamp(
                wrap_angle(desired_yaw - previous_yaw),
                -self.los_goal_max_yaw_step,
                self.los_goal_max_yaw_step,
            )
            target_yaw = wrap_angle(previous_yaw + yaw_step)
        goal = self.make_pose(target_x, target_y, target_yaw)
        self.last_los_goal = copy.deepcopy(goal)
        self.publish_motion_goal(goal)
        return goal

    def publish_position_target(self, point, yaw):
        self.publish_dprov(self.make_pose(point.x, point.y, yaw))

    def motion_state_fresh(self):
        return (
            self.latest_motion_state is not None
            and (rospy.Time.now() - self.latest_motion_state.header.stamp).to_sec()
            <= self.motion_state_timeout
        )

    def motion_goal_matches(self, target):
        if not self.motion_state_fresh() or target is None:
            return False
        actual = self.latest_motion_state.goal
        position_error = math.sqrt(
            (actual.pose.position.x - target.pose.position.x) ** 2
            + (actual.pose.position.y - target.pose.position.y) ** 2
            + (actual.pose.position.z - target.pose.position.z) ** 2
        )
        yaw_error = abs(wrap_angle(
            yaw_from_quaternion(actual.pose.orientation)
            - yaw_from_quaternion(target.pose.orientation)
        ))
        return (
            position_error <= self.motion_goal_position_tolerance
            and yaw_error <= self.motion_goal_yaw_tolerance
        )

    def motion_arrived(self, target=None):
        target = self.last_motion_goal if target is None else target
        return (
            self.motion_state_fresh()
            and self.latest_motion_state.state == MotionState.HOVER
            and self.motion_goal_matches(target)
        )

    def motion_failed(self):
        return (
            self.motion_state_fresh()
            and self.latest_motion_state.state == MotionState.SAFE
        )

    def cancel_motion(self):
        if not self.cancel_sent:
            self.motion_cancel_pub.publish(Empty())
            self.cancel_sent = True

    def motion_is_stable(self, _current):
        return self.motion_arrived()

    def transform_pose_to_map(self, pose):
        try:
            self.tf_listener.waitForTransform(
                "map", pose.header.frame_id, pose.header.stamp, rospy.Duration(1.0)
            )
            return self.tf_listener.transformPose("map", pose)
        except tf.Exception:
            try:
                latest_pose = copy.deepcopy(pose)
                latest_pose.header.stamp = rospy.Time(0)
                self.tf_listener.waitForTransform(
                    "map", latest_pose.header.frame_id, rospy.Time(0),
                    rospy.Duration(1.0)
                )
                return self.tf_listener.transformPose("map", latest_pose)
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

    def order_and_validate_triplet(self, poses, reference):
        """按三点主方向排序；P1 虽通常更远，但三点标签不作为轨迹顺序。"""
        points = [copy.deepcopy(pose.pose.position) for pose in poses]
        coordinates = np.array([[point.x, point.y] for point in points], dtype=float)
        if not np.isfinite(coordinates).all():
            return None, None, None, "non_finite_point"
        center = coordinates.mean(axis=0)
        try:
            _, _, axes = np.linalg.svd(coordinates - center)
        except np.linalg.LinAlgError:
            return None, None, None, "triplet_svd_failed"
        axis = axes[0]
        order = np.argsort(np.dot(coordinates - center, axis))
        ordered = [points[int(index)] for index in order]
        if xy_distance(ordered[-1], reference) < xy_distance(ordered[0], reference):
            ordered.reverse()

        first_spacing = xy_distance(ordered[0], ordered[1])
        second_spacing = xy_distance(ordered[1], ordered[2])
        if min(first_spacing, second_spacing) < self.line_min_point_spacing:
            return None, None, None, "point_spacing_too_small"
        if max(first_spacing, second_spacing) > self.line_max_point_spacing:
            return None, None, None, "point_spacing_too_large"

        first_yaw = math.atan2(
            ordered[1].y - ordered[0].y, ordered[1].x - ordered[0].x
        )
        second_yaw = math.atan2(
            ordered[2].y - ordered[1].y, ordered[2].x - ordered[1].x
        )
        bend = self.undirected_angle_error(first_yaw, second_yaw)
        residual = self.point_to_chord_distance(
            ordered[1], ordered[0], ordered[2]
        )
        if bend > self.line_triplet_max_bend:
            return None, None, residual, "triplet_bend_too_large"
        if residual > self.line_triplet_max_residual:
            return None, None, residual, "triplet_residual_too_large"

        detected_yaw = math.atan2(
            ordered[-1].y - ordered[0].y,
            ordered[-1].x - ordered[0].x,
        )
        return ordered, detected_yaw, residual, "valid"

    def curve_ready(self, points=None, distances=None):
        points = self.line_curve_points if points is None else points
        distances = self.line_curve_s if distances is None else distances
        return (
            len(points) >= 2
            and len(distances) == len(points)
        )

    def tracking_curve_ready(self):
        return self.curve_ready(
            self.tracking_curve_points, self.tracking_curve_s
        )

    @staticmethod
    def cumulative_distance(points):
        distances = [0.0]
        for index in range(1, len(points)):
            distances.append(
                distances[-1] + xy_distance(points[index - 1], points[index])
            )
        return distances

    def project_to_curve(self, point, curve_points=None, curve_s=None):
        curve_points = (
            self.line_curve_points if curve_points is None else curve_points
        )
        curve_s = self.line_curve_s if curve_s is None else curve_s
        if not self.curve_ready(curve_points, curve_s):
            return None
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
            segment_length = math.sqrt(segment_sq)
            candidate = {
                "distance": distance,
                "path_s": curve_s[index] + ratio * segment_length,
                "segment_yaw": math.atan2(vy, vx),
            }
            if best is None or distance < best["distance"]:
                best = candidate
        return best

    def point_at_curve_s(self, target_s, curve_points=None, curve_s=None):
        curve_points = (
            self.line_curve_points if curve_points is None else curve_points
        )
        curve_s = self.line_curve_s if curve_s is None else curve_s
        target_s = clamp(target_s, 0.0, curve_s[-1])
        for index in range(len(curve_s) - 1):
            start_s = curve_s[index]
            end_s = curve_s[index + 1]
            if target_s > end_s:
                continue
            start = curve_points[index]
            end = curve_points[index + 1]
            if end_s <= start_s:
                return copy.deepcopy(start)
            ratio = (target_s - start_s) / (end_s - start_s)
            return Point(
                start.x + ratio * (end.x - start.x),
                start.y + ratio * (end.y - start.y),
                self.hold_z,
            )
        return copy.deepcopy(curve_points[-1])

    def tracking_point_at_s(self, target_s):
        return self.point_at_curve_s(
            target_s, self.tracking_curve_points, self.tracking_curve_s
        )

    def activate_latest_tracking_curve(self, current=None, reset_progress=False):
        """在一个 LOS 固定段走完时，原子地换入最新固定曲线快照。"""
        # 回调线程冻结新曲线和主循环切换 LOS 快照使用同一把锁。
        with self.curve_lock:
            if not self.curve_ready(
                self.line_committed_curve_points, self.line_committed_curve_s
            ):
                return False
            fixed_version = self.line_version
            tracking_curve = [
                copy.deepcopy(point) for point in self.line_committed_curve_points
            ]
        self.tracking_curve_points = tracking_curve
        self.tracking_curve_s = self.cumulative_distance(tracking_curve)
        self.tracking_curve_version = fixed_version
        if reset_progress:
            self.current_path_s = 0.0
        elif current is not None:
            projection = self.project_to_curve(
                current.pose.position,
                self.tracking_curve_points,
                self.tracking_curve_s,
            )
            if projection is not None:
                # 新旧曲线都从已锁定起点向前编号；即使重拟合让最近投影略向后，
                # 也不允许 LOS 的路径进度退回到已完成路段。
                completed_on_new_curve = min(
                    self.completed_path_length, self.tracking_curve_s[-1]
                )
                self.current_path_s = max(
                    completed_on_new_curve, projection["path_s"]
                )
        return True

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
        extension_allowed = self.valid_curve_extension(points, distance_limit)
        close_enough = fit_distance <= distance_limit or extension_allowed
        if not close_enough:
            return False, fit_distance, angle_error, "line_too_far"
        if angle_error > angle_limit:
            return False, fit_distance, angle_error, "line_direction_mismatch"
        return True, fit_distance, angle_error, "associated"

    def valid_curve_extension(self, points, lateral_limit):
        """仅允许沿曲线首尾切向向外延伸，隔离端点附近的平行误识别线。"""
        if len(self.line_curve_points) < 2:
            return False
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

    def roll_line_fit_window(self):
        """原始点上限只作为内存保护，不再触发未经确认的曲线冻结。"""
        if (
            len(self.line_raw_points) <= self.line_curve_max_points
            or not self.curve_ready()
            or self.line_reference_point is None
        ):
            return
        ordered = sorted(
            self.line_raw_points,
            key=lambda point: (
                self.project_to_curve(point)["path_s"]
                if self.project_to_curve(point) is not None else -1.0
            ),
        )
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
            projection = self.project_to_curve(
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
        """把经连续合理帧确认的当前拟合曲线整体冻结，并形成新 LOS 版本。"""
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

        fixed_curve = [
            copy.deepcopy(point) for point in self.line_curve_points
        ]
        fixed_curve_s = self.cumulative_distance(fixed_curve)
        with self.curve_lock:
            self.line_committed_curve_points = fixed_curve
            self.line_committed_curve_s = fixed_curve_s
            self.line_version += 1
            fixed_version = self.line_version
        self.freeze_confirmation_count = 0

        minimum_s = max(
            0.0,
            self.line_committed_curve_s[-1] - self.curve_freeze_overlap_distance,
        )
        retained = []
        for point in self.line_raw_points:
            projection = self.project_to_curve(
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
            fixed_version,
            self.line_committed_curve_s[-1],
            len(self.line_raw_points),
        )
        return True

    def fuse_line_points(self, points):
        for point in points:
            point = Point(point.x, point.y, self.hold_z)
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
            low_index = int(np.argmin(np.dot(coordinates - center, axis)))
            high_index = int(np.argmax(np.dot(coordinates - center, axis)))
            low = self.line_raw_points[low_index]
            high = self.line_raw_points[high_index]
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
            curve = [
                Point(float(x_model(value)), float(y_model(value)), self.hold_z)
                for value in samples
            ]
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return False

        local_start_point = min(
            self.line_raw_points,
            key=lambda point: xy_distance(point, self.line_reference_point),
        )
        local_end_point = max(
            self.line_raw_points,
            key=lambda point: xy_distance(point, self.line_reference_point),
        )
        if xy_distance(curve[-1], local_start_point) < xy_distance(
            curve[0], local_start_point
        ):
            curve.reverse()
            self.line_axis = -self.line_axis
        curve[0] = copy.deepcopy(local_start_point)
        curve[-1] = copy.deepcopy(local_end_point)

        if (
            self.line_start_point is None
            or (
                not self.line_locked
                and xy_distance(local_start_point, self.line_reference_point)
                < xy_distance(self.line_start_point, self.line_reference_point)
            )
        ):
            self.line_start_point = copy.deepcopy(local_start_point)

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

        curve[0] = copy.deepcopy(self.line_start_point)
        curve_s = self.cumulative_distance(curve)
        if curve_s[-1] < self.line_curve_min_length:
            return False

        end_point = local_end_point
        if (
            self.line_end_point is not None
            and xy_distance(self.line_end_point, self.line_reference_point)
            > xy_distance(local_end_point, self.line_reference_point)
        ):
            end_point = self.line_end_point
        new_end_distance = xy_distance(end_point, self.line_reference_point)
        self.line_curve_points = curve
        self.line_curve_s = curve_s
        self.line_end_point = copy.deepcopy(end_point)

        residuals = []
        for point in self.line_raw_points:
            projection = self.project_to_curve(point)
            if projection is not None:
                residuals.append(projection["distance"])
        self.line_fit_residual = (
            sum(residuals) / len(residuals) if residuals else 0.0
        )

        if (
            self.confirmed_end_distance < 0.0
            or new_end_distance
            > self.confirmed_end_distance + self.endpoint_growth_tolerance
        ):
            had_previous_end = self.confirmed_end_distance >= 0.0
            self.confirmed_end_distance = new_end_distance
            if had_previous_end:
                self.reset_endpoint_evidence()
        return True

    def reset_endpoint_evidence(self):
        self.endpoint_candidate_point = None
        self.endpoint_candidate_count = 0

    def update_endpoint_evidence(self, observed_points, robot_point):
        """机器人接近最远点后，当前消息再次看到该点才累计稳定帧。"""
        if self.line_end_point is None or self.line_reference_point is None:
            return
        if xy_distance(
            robot_point, self.line_end_point
        ) > self.endpoint_evidence_robot_distance:
            self.reset_endpoint_evidence()
            return
        observed_farthest = max(
            observed_points,
            key=lambda point: xy_distance(point, self.line_reference_point),
        )
        if xy_distance(
            observed_farthest, self.line_end_point
        ) > self.endpoint_stable_position_tolerance:
            self.reset_endpoint_evidence()
            return

        if (
            self.endpoint_candidate_point is None
            or xy_distance(
                observed_farthest, self.endpoint_candidate_point
            ) > self.endpoint_stable_position_tolerance
        ):
            self.endpoint_candidate_point = copy.deepcopy(observed_farthest)
            self.endpoint_candidate_count = 1
            return

        self.endpoint_candidate_count += 1
        weight = 1.0 / float(self.endpoint_candidate_count)
        self.endpoint_candidate_point.x = (
            (1.0 - weight) * self.endpoint_candidate_point.x
            + weight * observed_farthest.x
        )
        self.endpoint_candidate_point.y = (
            (1.0 - weight) * self.endpoint_candidate_point.y
            + weight * observed_farthest.y
        )

    def endpoint_confirmed(self):
        return (
            self.endpoint_candidate_point is not None
            and self.endpoint_candidate_count >= self.endpoint_stable_frames
            and xy_distance(
                self.endpoint_candidate_point, self.line_end_point
            ) <= self.endpoint_stable_position_tolerance
        )

    @staticmethod
    def normalized_confidence(value):
        confidence = float(value)
        if confidence > 1.0:
            confidence /= 100.0
        return clamp(confidence, 0.0, 1.0)

    def line_callback(self, message):
        if self.state == self.WAIT_CAMERA or not self.camera_ready():
            return
        if self.hold_z is None and not self.initialize_start_pose():
            return
        if message.class_name and message.class_name not in self.line_classes:
            return

        camera_points = [
            message.pose1.pose.position,
            message.pose2.pose.position,
            message.pose3.pose.position,
        ]
        if any(
            not math.isfinite(point.x)
            or not math.isfinite(point.y)
            or not math.isfinite(point.z)
            or math.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)
            > self.max_camera_distance
            for point in camera_points
        ):
            return

        transformed = [
            self.transform_pose_to_map(pose)
            for pose in (message.pose1, message.pose2, message.pose3)
        ]
        if any(pose is None for pose in transformed):
            return
        # Web 始终显示最近一条可转换到 map 的识别消息，便于观察被后续规则拒绝的三点。
        self.latest_line_points = [
            copy.deepcopy(pose.pose.position) for pose in transformed
        ]

        current = self.get_current_pose()
        if current is None:
            return
        reference = (
            self.line_reference_point
            if self.line_reference_point is not None
            else current.pose.position
        )
        points, detected_yaw, residual, reason = self.order_and_validate_triplet(
            transformed, reference
        )
        if points is None:
            rospy.loginfo_throttle(
                3.0, "%s: 红线点已忽略，原因=%s", NODE_NAME, reason
            )
            return

        confidence = self.normalized_confidence(message.conf)
        now = rospy.Time.now()
        if not self.line_locked:
            if self.line_lock_candidate is None:
                self.line_lock_candidate = {
                    "started": now,
                    "confidence": confidence,
                    "points": points,
                    "reference": copy.deepcopy(current.pose.position),
                    "hold_pose": copy.deepcopy(current),
                }
            elif confidence > self.line_lock_candidate["confidence"]:
                self.line_lock_candidate["confidence"] = confidence
                self.line_lock_candidate["points"] = points
            return

        associated, fit_distance, angle_error, reason = self.line_segment_associated(
            points, detected_yaw, confidence
        )
        if not associated:
            rospy.loginfo_throttle(
                3.0,
                "%s: 红线点已隔离，conf=%.2f，拟合距离=%.2f m，方向差=%.1f deg，原因=%s",
                NODE_NAME,
                confidence,
                fit_distance,
                math.degrees(angle_error),
                reason,
            )
            return

        active_points = self.points_not_on_fixed_curve(points)
        if not active_points:
            # 固定段重复点不再参与拟合，但到达固定末端时仍可作为终点重复观测证据。
            self.update_endpoint_evidence(points, current.pose.position)
            rospy.loginfo_throttle(
                3.0, "%s: 本帧红线点均位于已固定曲线上，已忽略", NODE_NAME
            )
            return
        self.fuse_line_points(active_points)
        if self.fit_line_curve():
            self.freeze_confirmation_count = min(
                self.curve_freeze_required_frames,
                self.freeze_confirmation_count + 1,
            )
            self.freeze_confirmed_curve()
            self.update_endpoint_evidence(active_points, current.pose.position)

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
            return False
        self.line_lock_confidence = candidate["confidence"]
        self.line_locked = True
        self.line_lock_candidate = None
        self.initial_line_hold_pose = copy.deepcopy(candidate["hold_pose"])
        self.freeze_confirmation_count = 1
        self.freeze_confirmed_curve()
        self.current_tracking_point = copy.deepcopy(self.line_start_point)
        current = self.get_current_pose()
        if current is not None:
            self.update_endpoint_evidence(
                candidate["points"], current.pose.position
            )
        distance = (
            xy_distance(current.pose.position, self.line_start_point)
            if current is not None else float("nan")
        )
        rospy.loginfo(
            "%s: 识别状态=已锁定；conf=%.2f，当前位置=(%.2f, %.2f)，"
            "起点=(%.2f, %.2f)，当前最远点=(%.2f, %.2f)，距起点=%.2f m",
            NODE_NAME,
            self.line_lock_confidence,
            current.pose.position.x if current is not None else float("nan"),
            current.pose.position.y if current is not None else float("nan"),
            self.line_start_point.x,
            self.line_start_point.y,
            self.line_end_point.x,
            self.line_end_point.y,
            distance,
        )
        if self.activate_latest_tracking_curve(reset_progress=True):
            self.set_state(self.GO_TO_START)
        else:
            self.set_state(self.WAIT_FIXED_LINE)
            rospy.loginfo(
                "%s: 首条红线已拟合，等待后续合理帧确认固定（%d/%d）",
                NODE_NAME,
                self.freeze_confirmation_count,
                self.curve_freeze_required_frames,
            )
        return True

    def run_wait_fixed_line(self):
        """首段拟合尚未确认时保持原位；固定后才允许机器人开始 LOS。"""
        current = self.get_current_pose()
        if current is None:
            return
        hold_pose = self.initial_line_hold_pose or current
        self.publish_dprov(hold_pose)
        self.current_tracking_point = copy.deepcopy(self.line_start_point)
        rospy.loginfo_throttle(
            2.0,
            "%s: 红线已拟合，等待固定确认=%d/%d；机器人保持定点",
            NODE_NAME,
            self.freeze_confirmation_count,
            self.curve_freeze_required_frames,
        )
        if self.activate_latest_tracking_curve(reset_progress=True):
            rospy.loginfo(
                "%s: 首段曲线已固定，固定长度=%.2f m，开始前往曲线起点",
                NODE_NAME,
                self.tracking_curve_s[-1],
            )
            self.set_state(self.GO_TO_START)

    def set_search_state(self, state):
        self.set_state(state)
        self.search_target = None

    def wait_until_target_stable(self, current, reached, hold_seconds):
        if not reached or not self.motion_is_stable(current):
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = rospy.Time.now()
        return (rospy.Time.now() - self.stable_since).to_sec() >= hold_seconds

    def run_search_rotation(self, current, yaw_offset, next_state, label, hold_seconds):
        """保持搜索锚点，只向监督器提交绝对航向目标。"""
        if self.search_cycle_anchor is None:
            self.search_cycle_anchor = copy.deepcopy(current.pose.position)
        target_yaw = wrap_angle(self.search_base_yaw + yaw_offset)
        if self.search_target is None:
            self.search_target = self.make_pose(
                self.search_cycle_anchor.x,
                self.search_cycle_anchor.y,
                target_yaw,
            )
            rospy.loginfo(
                "%s: 识别状态=未识别；SEARCH %s %.1f deg",
                NODE_NAME,
                label,
                math.degrees(abs(yaw_offset)),
            )

        self.current_tracking_point = copy.deepcopy(self.search_target.pose.position)
        self.publish_dprov(self.search_target)
        position_reached = xy_distance(
            current.pose.position, self.search_cycle_anchor
        ) <= self.position_tolerance
        yaw_error = abs(wrap_angle(
            target_yaw - yaw_from_quaternion(current.pose.orientation)
        ))
        if self.wait_until_target_stable(
            current,
            position_reached and yaw_error <= self.search_yaw_tolerance,
            hold_seconds,
        ):
            self.set_search_state(next_state)

    def run_search_forward(self, current):
        if self.search_target is None:
            target_point = Point(
                current.pose.position.x
                + self.search_forward_distance * math.cos(self.search_base_yaw),
                current.pose.position.y
                + self.search_forward_distance * math.sin(self.search_base_yaw),
                self.hold_z,
            )
            self.search_target = self.make_pose(
                target_point.x,
                target_point.y,
                self.search_base_yaw,
            )
            rospy.loginfo(
                "%s: 识别状态=未识别；SEARCH 定点向前移动 %.2f m",
                NODE_NAME,
                self.search_forward_distance,
            )
        self.current_tracking_point = copy.deepcopy(self.search_target.pose.position)
        self.publish_dprov(self.search_target)
        reached = xy_distance(
            current.pose.position, self.search_target.pose.position
        ) <= self.position_tolerance
        yaw_error = abs(wrap_angle(
            self.search_base_yaw - yaw_from_quaternion(current.pose.orientation)
        ))
        if self.wait_until_target_stable(
            current,
            reached and yaw_error <= self.search_yaw_tolerance,
            self.search_forward_hold_seconds,
        ):
            self.search_cycle_anchor = None
            self.set_search_state(self.SEARCH_LEFT)

    def run_search(self):
        current = self.get_current_pose()
        if current is None:
            return
        rospy.loginfo_throttle(3.0, "%s: 识别状态=未识别", NODE_NAME)
        if self.state == self.SEARCH_LEFT:
            self.run_search_rotation(
                current,
                -self.search_yaw_sign * self.search_yaw_angle,
                self.SEARCH_RIGHT,
                "向左旋转",
                self.search_yaw_hold_seconds,
            )
        elif self.state == self.SEARCH_RIGHT:
            self.run_search_rotation(
                current,
                self.search_yaw_sign * self.search_yaw_angle,
                self.SEARCH_RETURN,
                "向右旋转",
                self.search_yaw_hold_seconds,
            )
        elif self.state == self.SEARCH_RETURN:
            self.run_search_rotation(
                current,
                0.0,
                self.SEARCH_FORWARD,
                "返回初始航向",
                self.search_return_hold_seconds,
            )
        elif self.state == self.SEARCH_FORWARD:
            self.run_search_forward(current)

    def enter_hold_start(self, current):
        if self.hold_target is None:
            self.hold_target = self.make_pose(
                self.line_start_point.x,
                self.line_start_point.y,
                yaw_from_quaternion(current.pose.orientation),
            )
        self.publish_dprov(self.hold_target)
        rospy.loginfo(
            "%s: 已到红线起点，进入定点稳定，起点=(%.2f, %.2f)",
            NODE_NAME,
            self.line_start_point.x,
            self.line_start_point.y,
        )
        self.set_state(self.HOLD_START)

    def run_go_to_start(self):
        current = self.get_current_pose()
        if current is None or self.line_start_point is None:
            return
        self.current_tracking_point = copy.deepcopy(self.line_start_point)

        distance = xy_distance(current.pose.position, self.line_start_point)
        if self.hold_target is None:
            self.hold_target = self.make_pose(
                self.line_start_point.x,
                self.line_start_point.y,
                yaw_from_quaternion(current.pose.orientation),
            )
        self.publish_motion_goal(self.hold_target)
        if self.motion_arrived(self.hold_target):
            self.enter_hold_start(current)
            return
        rospy.loginfo_throttle(
            2.0,
            "%s: 监督器前往起点；当前位置=(%.2f, %.2f)，起点=(%.2f, %.2f)，"
            "距离=%.2f m，等待 HOVER",
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            self.line_start_point.x,
            self.line_start_point.y,
            distance,
        )

    def run_hold_start(self):
        current = self.get_current_pose()
        if current is None:
            return
        self.hold_target.pose.position.x = self.line_start_point.x
        self.hold_target.pose.position.y = self.line_start_point.y
        self.current_tracking_point = copy.deepcopy(self.line_start_point)
        self.publish_dprov(self.hold_target)
        distance = xy_distance(current.pose.position, self.line_start_point)
        stable = self.wait_until_target_stable(
            current,
            distance <= self.position_tolerance,
            self.line_start_hold_seconds,
        )
        rospy.loginfo_throttle(
            2.0,
            "%s: 起点定点；距离=%.2f m，稳定=%s",
            NODE_NAME,
            distance,
            "是" if stable else "否",
        )
        if stable:
            self.hold_target = None
            self.set_state(self.ALIGN_LINE)

    def line_start_yaw(self):
        lookahead = self.tracking_point_at_s(
            min(self.los_lookahead_distance, self.tracking_curve_s[-1])
        )
        return math.atan2(
            lookahead.y - self.line_start_point.y,
            lookahead.x - self.line_start_point.x,
        )

    def run_align_line(self):
        current = self.get_current_pose()
        if current is None or not self.tracking_curve_ready():
            return
        desired_yaw = self.line_start_yaw()
        self.current_tracking_point = self.tracking_point_at_s(
            min(self.los_lookahead_distance, self.tracking_curve_s[-1])
        )
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        yaw_error = wrap_angle(desired_yaw - current_yaw)
        if self.hold_target is None:
            self.hold_target = self.make_pose(
                self.line_start_point.x,
                self.line_start_point.y,
                desired_yaw,
            )
        self.publish_motion_goal(self.hold_target)
        aligned = self.motion_arrived(self.hold_target)
        if aligned:
            if self.stable_since is None:
                self.stable_since = rospy.Time.now()
        else:
            self.stable_since = None
        held = (
            self.stable_since is not None
            and (rospy.Time.now() - self.stable_since).to_sec()
            >= self.line_alignment_hold_seconds
        )
        rospy.loginfo_throttle(
            2.0,
            "%s: 起点对向；当前=%.1f deg，目标=%.1f deg，误差=%.1f deg",
            NODE_NAME,
            math.degrees(current_yaw),
            math.degrees(desired_yaw),
            math.degrees(yaw_error),
        )
        if held:
            self.current_path_s = 0.0
            self.last_los_goal = copy.deepcopy(self.hold_target)
            self.set_state(self.FOLLOW_LINE)

    def enter_hold_end(self, current):
        self.last_los_goal = None
        active_end = self.tracking_curve_points[-1]
        self.hold_target = self.make_pose(
            active_end.x,
            active_end.y,
            yaw_from_quaternion(current.pose.orientation),
        )
        self.publish_dprov(self.hold_target)
        self.endpoint_hold_started = rospy.Time.now()
        rospy.loginfo(
            "%s: 已到当前最远点，位置=(%.2f, %.2f)，定点等待终点重复观测和轨迹增长",
            NODE_NAME,
            active_end.x,
            active_end.y,
        )
        self.set_state(self.HOLD_END)

    def run_follow_line(self):
        current = self.get_current_pose()
        if current is None or not self.tracking_curve_ready():
            return
        projection = self.project_to_curve(
            current.pose.position,
            self.tracking_curve_points,
            self.tracking_curve_s,
        )
        if projection is None:
            return
        self.current_path_s = max(self.current_path_s, projection["path_s"])
        self.completed_path_length = max(
            self.completed_path_length, self.current_path_s
        )
        remaining_path = max(
            0.0, self.tracking_curve_s[-1] - self.current_path_s
        )
        active_end = self.tracking_curve_points[-1]
        endpoint_distance = xy_distance(current.pose.position, active_end)

        if (
            remaining_path <= self.endpoint_path_tolerance
            and endpoint_distance <= self.endpoint_arrival_tolerance
        ):
            # 当前快照走完后才允许把识别线程已经拟合出的新增曲线交给 LOS。
            if self.line_version > self.tracking_curve_version:
                old_version = self.tracking_curve_version
                if self.activate_latest_tracking_curve(current=current):
                    rospy.loginfo(
                        "%s: 当前 LOS 固定段已走完；接入固定曲线版本 %d -> %d",
                        NODE_NAME,
                        old_version,
                        self.tracking_curve_version,
                    )
                    return
            self.current_tracking_point = copy.deepcopy(active_end)
            self.enter_hold_end(current)
            return

        los_target = self.tracking_point_at_s(
            self.current_path_s + self.los_lookahead_distance
        )
        self.current_tracking_point = copy.deepcopy(los_target)
        desired_yaw = math.atan2(
            los_target.y - current.pose.position.y,
            los_target.x - current.pose.position.x,
        )
        commanded_goal = self.publish_los_goal(los_target, desired_yaw)
        state_name = (
            str(self.latest_motion_state.state)
            if self.motion_state_fresh()
            else "无新鲜反馈"
        )
        rospy.loginfo_throttle(
            2.0,
            "%s: LOS 巡线；已完成=%.2f m，已知轨迹=%.2f m，距终点=%.2f m，"
            "下发目标=(%.2f, %.2f)，motion_state=%s",
            NODE_NAME,
            self.completed_path_length,
            self.tracking_curve_s[-1],
            endpoint_distance,
            commanded_goal.pose.position.x,
            commanded_goal.pose.position.y,
            state_name,
        )

    def run_hold_end(self):
        current = self.get_current_pose()
        if current is None:
            return
        # line_version 只在后续帧确认并冻结新曲线时增加。
        grew = self.line_version > self.tracking_curve_version
        if grew:
            old_version = self.tracking_curve_version
            if self.activate_latest_tracking_curve(current=current):
                rospy.loginfo(
                    "%s: 发现新固定红线；接入固定曲线版本 %d -> %d，继续 LOS 巡线",
                    NODE_NAME,
                    old_version,
                    self.tracking_curve_version,
                )
                self.last_los_goal = copy.deepcopy(self.hold_target)
                self.set_state(self.FOLLOW_LINE)
                return

        active_end = self.tracking_curve_points[-1]
        self.hold_target.pose.position.x = active_end.x
        self.hold_target.pose.position.y = active_end.y
        self.current_tracking_point = copy.deepcopy(active_end)
        self.publish_dprov(self.hold_target)
        distance = xy_distance(current.pose.position, active_end)
        stable = self.wait_until_target_stable(
            current,
            distance <= self.position_tolerance,
            self.transition_hold_seconds,
        )
        no_growth_seconds = (
            rospy.Time.now() - self.endpoint_hold_started
        ).to_sec()
        fixed_length = (
            self.line_committed_curve_s[-1]
            if self.line_committed_curve_s else 0.0
        )
        pending_extension = max(
            0.0,
            (self.line_curve_s[-1] if self.line_curve_s else 0.0) - fixed_length,
        )
        rospy.loginfo_throttle(
            2.0,
            "%s: 终点候选定点；距离=%.2f m，稳定=%s，重复观测=%d/%d，"
            "待固定延伸=%.2f m，确认=%d/%d，轨迹未增长=%.1f/%.1f s",
            NODE_NAME,
            distance,
            "是" if stable else "否",
            self.endpoint_candidate_count,
            self.endpoint_stable_frames,
            pending_extension,
            self.freeze_confirmation_count,
            self.curve_freeze_required_frames,
            no_growth_seconds,
            self.endpoint_confirm_seconds,
        )
        if (
            stable
            and pending_extension < self.curve_freeze_min_advance
            and self.endpoint_confirmed()
            and no_growth_seconds >= self.endpoint_confirm_seconds
        ):
            self.set_state(self.FINISH)

    def publish_trajectory_status(self):
        now = rospy.Time.now()
        if (
            now - self.last_trajectory_publish_time
        ).to_sec() < self.trajectory_publish_period:
            return
        current = self.get_current_pose()
        if current is not None and (
            not self.actual_trajectory
            or xy_distance(current.pose.position, self.actual_trajectory[-1])
            >= self.actual_path_min_spacing
        ):
            self.actual_trajectory.append(copy.deepcopy(current.pose.position))
            if len(self.actual_trajectory) > self.actual_path_max_points:
                self.actual_trajectory = self.actual_trajectory[
                    -self.actual_path_max_points:
                ]

        def point_data(point):
            return [round(point.x, 3), round(point.y, 3)] if point else None

        payload = {
            "stamp": round(now.to_sec(), 3),
            "state": self.state,
            "motion_state": (
                self.latest_motion_state.state
                if self.motion_state_fresh() else None
            ),
            "motion_reason": (
                self.latest_motion_state.reason
                if self.motion_state_fresh() else ""
            ),
            "tx": self.latest_motion_state.tx if self.motion_state_fresh() else 0,
            "ty": self.latest_motion_state.ty if self.motion_state_fresh() else 0,
            "mz": self.latest_motion_state.mz if self.motion_state_fresh() else 0,
            "line_locked": self.line_locked,
            "lock_confidence": round(self.line_lock_confidence, 3),
            "fit_residual": round(self.line_fit_residual, 3),
            "fixed_length": round(
                self.line_committed_curve_s[-1]
                if self.line_committed_curve_s else 0.0,
                3,
            ),
            "fitted_length": round(
                self.line_curve_s[-1] if self.line_curve_s else 0.0,
                3,
            ),
            "completed_length": round(self.completed_path_length, 3),
            "robot": point_data(current.pose.position) if current else None,
            "robot_yaw_deg": round(math.degrees(yaw_from_quaternion(
                current.pose.orientation
            )), 2) if current else 0.0,
            "tracking_point": point_data(self.current_tracking_point),
            "tracking_curve_version": self.tracking_curve_version,
            "fixed_curve_version": self.line_version,
            "freeze_confirmations": self.freeze_confirmation_count,
            "freeze_required": self.curve_freeze_required_frames,
            "ignored_fixed_points": self.ignored_fixed_points,
            "latest_line_points": [
                point_data(point) for point in self.latest_line_points
            ],
            "actual_path": [point_data(point) for point in self.actual_trajectory],
            "planned_curve": [point_data(point) for point in self.line_curve_points],
            "fixed_curve": [
                point_data(point) for point in self.line_committed_curve_points
            ],
            "tracking_curve": [
                point_data(point) for point in self.tracking_curve_points
            ],
            "raw_line": [point_data(point) for point in self.line_raw_points],
            "line_start": point_data(self.line_start_point),
            "line_end": point_data(self.line_end_point),
            "endpoint_candidate": point_data(self.endpoint_candidate_point),
            "endpoint_stable_count": self.endpoint_candidate_count,
            "endpoint_stable_required": self.endpoint_stable_frames,
            "endpoint_confirmed": self.endpoint_confirmed(),
        }
        encoded = json.dumps(payload, separators=(",", ":"))
        self.trajectory_pub.publish(String(data=encoded))
        if self.trajectory_web is not None:
            self.trajectory_web.update(encoded)
        self.last_trajectory_publish_time = now

    def finish(self):
        self.cancel_motion()
        self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
        rospy.loginfo(
            "%s: FINISH；红线起点=(%.2f, %.2f)，终点=(%.2f, %.2f)，"
            "已完成路径=%.2f m",
            NODE_NAME,
            self.line_start_point.x,
            self.line_start_point.y,
            self.line_end_point.x,
            self.line_end_point.y,
            self.completed_path_length,
        )
        rospy.signal_shutdown("%s complete" % NODE_NAME)

    def run(self):
        while not rospy.is_shutdown():
            if not self.initialize_start_pose():
                self.rate.sleep()
                continue
            if self.motion_failed():
                self.publish_trajectory_status()
                rospy.logerr_throttle(
                    2.0,
                    "%s: motion_supervisor=SAFE，暂停任务推进，原因=%s",
                    NODE_NAME,
                    self.latest_motion_state.reason,
                )
                self.rate.sleep()
                continue

            self.try_lock_line()
            self.publish_trajectory_status()

            if self.line_lock_candidate is not None and not self.line_locked:
                # 首帧候选出现后立即停止搜索，短暂定点选择窗口内最高置信线。
                self.current_tracking_point = copy.deepcopy(
                    self.line_lock_candidate["points"][0]
                )
                self.publish_dprov(self.line_lock_candidate["hold_pose"])
                rospy.loginfo_throttle(
                    2.0,
                    "%s: 识别状态=候选已发现；定点选择最高置信红线",
                    NODE_NAME,
                )
            elif self.state == self.WAIT_CAMERA:
                self.publish_dprov(self.start_pose)
                self.current_tracking_point = copy.deepcopy(
                    self.start_pose.pose.position
                )
                current = self.get_current_pose()
                hold_elapsed = (
                    (rospy.Time.now() - self.startup_hold_started).to_sec()
                    if self.startup_hold_started is not None else 0.0
                )
                rospy.loginfo_throttle(
                    2.0,
                    "%s: 摄像头节点=%s；当前位姿=(%.2f, %.2f, %.2f)；"
                    "启动定点=%.1f/%.1f s",
                    NODE_NAME,
                    "已开启" if self.camera_ready() else "未开启",
                    current.pose.position.x if current else float("nan"),
                    current.pose.position.y if current else float("nan"),
                    current.pose.position.z if current else float("nan"),
                    hold_elapsed,
                    self.startup_hold_seconds,
                )
                if (
                    self.camera_ready()
                    and hold_elapsed >= self.startup_hold_seconds
                    and self.motion_arrived(self.start_pose)
                ):
                    rospy.loginfo(
                        "%s: 启动定点完成，进入红线识别和左右转搜索阶段",
                        NODE_NAME,
                    )
                    self.set_search_state(self.SEARCH_LEFT)
            elif self.state in self.SEARCH_STATES:
                self.run_search()
            elif self.state == self.WAIT_FIXED_LINE:
                self.run_wait_fixed_line()
            elif self.state == self.GO_TO_START:
                self.run_go_to_start()
            elif self.state == self.HOLD_START:
                self.run_hold_start()
            elif self.state == self.ALIGN_LINE:
                self.run_align_line()
            elif self.state == self.FOLLOW_LINE:
                self.run_follow_line()
            elif self.state == self.HOLD_END:
                self.run_hold_end()
            elif self.state == self.FINISH:
                self.finish()

            self.rate.sleep()


def main():
    rospy.init_node(NODE_NAME)
    Task1LineFollowTest().run()


if __name__ == "__main__":
    main()
