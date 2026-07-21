#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v3_line_follow.py
功能：Task1 巡线单项测试，运动接口迁移到 motion_supervisor。

流程：
    1. 记录节点启动时的当前位置、高度和航向，定点等待相机和识别模块；
    2. 未识别红线时，依次定点左转、右转、回正，再向前移动后重复；
    3. 在短时间窗内选择置信度最高的第一条红线并锁定；
    4. conf >= 0.70 时 positions 中坐标有限的点全部进入拟合，后续高置信帧确认后立即冻结曲线段；
    5. LOS 以 base_link 投影选择曲线前方目标，实际向 motion_supervisor
       下发 base_link 与 LOS 目标之间的中点；新目标直接覆盖旧目标；
    6. 已知曲线没有新目标时保持最后目标，由控制器完成平移和最终转向，
       HOVER 连续确认后固定 XY 左右旋转寻找曲线延伸；
    7. 到达当前最远点后继续终点判断；同一最远点被连续多次观测且轨迹不再
       增长时，才将其确认为真实终点并结束测试。

起终点定义：
    以首次锁线时的 base_link 位置为固定参考点。所有已接受点中，距参考点
    最近的点记为红线起点，最远的点记为红线终点。参考点不会随机器人
    运动而改变，因此起终点不会在巡线过程中反转。

监听：/obj/line_message，/left/image_raw，/motion/state，/tf
发布：/cmd/motion/goal，/cmd/motion/cancel，/task1/v3/trajectory，/finished
网页：默认 http://192.168.1.117:8083

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
    识别点不再解释为整条线端点，改为多点局部验证和曲线关联。
    低于 0.70 置信度且难以拟合的点直接丢弃。
    当前最远点只作为临时巡线目标；连续多帧仍为同一最远点后才形成终点证据。
    原始点集改为滑动窗口；旧点超限后冻结历史曲线，只用远端重叠点和新点拟合后缀。
2026.7.18
    启动阶段增加可调的 mode=4 定点悬停时间。
    未识别红线时改为定点左转、右转、回到启动航向，再定点前进后循环搜索。
    轨迹网页增加机器人航向箭头、当前跟踪点和最新多点识别结果。
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
2026.7.18
    巡线投影、起终点距离和 Web 实际轨迹改用双目摄像头位置；下发运动目标前
    根据 base_link -> camera 静态 TF 补偿水平杆臂。
    LOS 当前航点保持不变，摄像头位置和机器人航向同时到达门槛后才切换下一点。
2026.7.20
    每个控制周期的完整位姿、目标、误差、路径和监督器输出改用 logdebug；
    loginfo 只保留阶段变化、航点到达和节流后的测试摘要。
2026.7.20
    Web 改为 Reset 原点下的 NED 俯视图，N 向上、E 向右；机器人和实际
    轨迹统一显示 base_link，base_link 到 camera 的实时 TF 作为航向箭头。
    LOS 航点改为先保持当前航向到达并定点，再按当前点到下一点的方向对向。
2026.7.20
    红线输入由固定三点 TargetDetection3 迁移到多点 LineDetection；逐点过滤
    point_valid，launch 可限制每帧参与拟合的有效点数量，Web 显示总/有效/使用点数。
2026.7.20
    增加可调的红线置信度硬下限；低于下限的整帧识别不参与首次锁线、
    曲线关联、点融合或轨迹冻结，避免低置信误识别污染执行轨迹。
2026.7.20
    巡线定位改回 base_link，取消 camera 杆臂补偿和逐航点对向；LOS 下发
    base_link 与前视点的中点，新固定曲线立即更新目标。没有新目标时保持
    最后目标，待 MotionState.HOVER 连续 3 秒后固定 XY 左右旋转搜索延伸。
    增加曲线单向性检查和降阶拟合，禁止往返折线进入固定/执行轨迹；完整
    控制、感知和轨迹数据按运行批次写入可配置目录下的 YAML 文件。
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
    持续保留。显示历史支持独立的空间合并距离和数量上限，不影响拟合与控制。
2026.7.20
    根据实测确认 conf >= 0.70 的红线点均可信，停用曲线关联、内点比例、延伸
    方向、固定段过滤和曲线回折等几何拒绝条件；所有达标点直接融合并重新拟合。
    进一步取消 point_valid、坐标合理性和每帧取点上限；置信度达标后直接使用
    positions 数组的全部点。密集点融合距离默认改为 0，避免正常近邻点被压缩。
2026.7.20
    跳过 positions 中无法参与数值计算的 NaN/Inf 点，其余高置信点仍全部用于拟合；
    Web 历史点增加非有限值保护，避免显示处理打断感知回调。
    终点结束增加已完成路径长度下限，防止把相机当前视野末端误判为整条红线终点。
2026.7.20
    当前 LOS 点与固定曲线末端的剩余弧长小于目标更新最小距离时，直接视为已走完
    当前固定曲线，避免末端微小目标更新被抑制后永久停留在 FOLLOW_LINE。
2026.7.20
    结构化测试数据由 JSONL 改为带启动时间戳的 YAML 文档流；每条感知、目标、
    状态和控制周期记录都保留独立墙钟时间与 ROS 时间，便于复现实测过程。
"""

import copy
from collections import deque
import json
import math
import os
import threading
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rospy
import tf
from auv_control.msg import LineDetection, MotionState
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Empty, String
from task1_v3_yaml_logger import TimestampedYamlLogger
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
<div class="legend"><span>蓝线：base_link 实际轨迹</span><span>红/橙/绿：base_link 拟合/固定/执行轨迹</span><span>青色圆点→箭头：base_link→camera</span>
<span>紫点：下发中点目标</span><span>棕点：LOS曲线目标</span><span>浅粉点：历史有效识别点</span><span>亮粉点：最新有效识别点</span><span>滚轮缩放，拖动平移</span></div></header>
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
function draw(d){let history=d.valid_line_point_history||[],latest=d.latest_line_points||[],a=[...(d.actual_path||[]),...(d.planned_curve||[]),...(d.fixed_curve||[]),...(d.tracking_curve||[]),...(d.raw_line||[]),...history,...latest];
last=d;x.clearRect(0,0,c.width,c.height);grid();
history.forEach(v=>dot(v,'rgba(255,45,145,.32)',2));(d.raw_line||[]).forEach(v=>dot(v,'#879aa3',2));line(d.planned_curve,'#e74c3c',3);line(d.fixed_curve,'#f39c12',5);line(d.tracking_curve,'#21a366',4);line(d.actual_path,'#1677ff',3);
dot(d.line_start,'#21a366',7);dot(d.line_end,'#f39c12',7);dot(d.endpoint_candidate,'#8e44ad',5);dot(d.tracking_point,'#9b2cff',7);tag(d.tracking_point,'目标','#6f13ba');
dot(d.los_target,'#8b5a2b',6);tag(d.los_target,'LOS','#8b5a2b');
latest.forEach((v,i)=>{dot(v,'#ff2d91',4);if(i===0||i===latest.length-1||i===Math.floor(latest.length/2))tag(v,`P${i+1}`,'#ff2d91')});
dot(d.robot,'#00cfe8',8);bodyArrow(d.robot,d.camera);
document.getElementById('s').textContent=`任务 ${d.state}/${d.los_phase||'-'}　监督器 ${d.motion_state??'-'}　base航向 ${(d.robot_yaw_deg||0).toFixed(1)}°　D(base/camera) ${(d.robot_down??0).toFixed(2)}/${(d.camera_down??0).toFixed(2)} m　识别点 总/有效/使用 ${d.line_input_point_count||0}/${d.line_valid_point_count||0}/${d.line_used_point_count||0}　历史 ${d.valid_line_point_history_count||0}　锁线 ${d.line_locked?'是':'否'}　固定/拟合 ${d.fixed_length||0}/${d.fitted_length||0} m　已完成/投影 ${d.completed_length||0}/${d.projected_length||0} m　终点证据 ${d.endpoint_stable_count||0}/${d.endpoint_stable_required||0}`}
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


class Task1LineFollowTest:
    """只测试 Task1 红线搜索和巡线。"""

    WAIT_CAMERA = "WAIT_CAMERA"
    SEARCH_LEFT = "SEARCH_LEFT"
    SEARCH_RIGHT = "SEARCH_RIGHT"
    SEARCH_RETURN = "SEARCH_RETURN"
    SEARCH_FORWARD = "SEARCH_FORWARD"
    WAIT_FIXED_LINE = "WAIT_FIXED_LINE"
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
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.line_topic = rospy.get_param("~line_topic", "/obj/line_message")
        self.finished_topic = rospy.get_param("~finished_topic", "/finished")
        self.camera_topic = rospy.get_param("~camera_topic", "/left/image_raw")
        self.line_tracking_frame = str(rospy.get_param(
            "~line_tracking_frame", "camera"
        )).strip().lstrip("/") or "camera"
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

        # 相机和任务阶段等待时间；静止与接管统一由 MotionState.HOVER 确认。
        self.camera_message_timeout = float(rospy.get_param(
            "~camera_message_timeout", 2.0
        ))
        self.tf_timeout_seconds = max(0.0, float(rospy.get_param(
            "~tf_timeout_seconds", 0.1
        )))
        self.line_message_max_age_seconds = max(0.0, float(rospy.get_param(
            "~line_message_max_age_seconds", 0.5
        )))
        self.startup_hold_seconds = max(0.0, float(rospy.get_param(
            "~startup_hold_seconds", 10.0
        )))

        # 红线首帧选择和置信度硬下限。
        self.line_classes = class_names("~line_classes", ["line"])
        self.line_lock_window_seconds = float(rospy.get_param(
            "~line_lock_window_seconds", 0.3
        ))
        self.line_min_confidence = clamp(float(rospy.get_param(
            "~line_min_confidence", 0.70
        )), 0.0, 1.0)
        # 已接受点集上限和平滑曲线拟合参数。
        self.line_point_merge_distance = max(0.0, float(rospy.get_param(
            "~line_point_merge_distance", 0.0
        )))
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
        self.line_curve_sample_count = max(3, int(rospy.get_param(
            "~line_curve_sample_count", 100
        )))
        self.line_curve_degree = max(1, int(rospy.get_param(
            "~line_curve_degree", 3
        )))
        self.curve_freeze_required_frames = max(1, int(rospy.get_param(
            "~curve_freeze_required_frames", 2
        )))
        self.curve_freeze_min_length = float(rospy.get_param(
            "~curve_freeze_min_length", 0.15
        ))
        self.curve_freeze_min_advance = float(rospy.get_param(
            "~curve_freeze_min_advance", 0.10
        ))
        self.curve_freeze_overlap_distance = float(rospy.get_param(
            "~curve_freeze_overlap_distance", 0.25
        ))
        self.endpoint_growth_tolerance = float(rospy.get_param(
            "~endpoint_growth_tolerance", 0.08
        ))

        # LOS 使用 base_link 选择前视点；向监督器下发当前位置到前视点的中点。
        self.los_lookahead_distance = float(rospy.get_param(
            "~los_lookahead_distance", 0.6
        ))
        self.los_midpoint_ratio = clamp(float(rospy.get_param(
            "~los_midpoint_ratio", 0.50
        )), 0.05, 1.0)
        self.los_target_update_min_distance = max(0.001, float(
            rospy.get_param("~los_target_update_min_distance", 0.08)
        ))
        self.motion_hover_confirm_seconds = max(0.0, float(rospy.get_param(
            "~motion_hover_confirm_seconds", 3.0
        )))
        self.endpoint_confirm_seconds = float(rospy.get_param(
            "~endpoint_confirm_seconds", 5.0
        ))
        self.endpoint_min_completed_path_length = max(0.0, float(
            rospy.get_param("~endpoint_min_completed_path_length", 4.5)
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
        self.web_valid_point_merge_distance = max(0.0, float(rospy.get_param(
            "~web_valid_point_merge_distance", 0.01
        )))
        self.web_valid_point_max_count = max(100, int(rospy.get_param(
            "~web_valid_point_max_count", 20000
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
        self.log_directory = os.path.expanduser(str(rospy.get_param(
            "~log_directory", "~/.ros/auv_logs/task1"
        )))

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
        self.data_logger = None
        self.data_log_path = None
        rospy.on_shutdown(self.shutdown)

        self.search_target = None
        self.search_cycle_anchor = None
        self.extension_search_active = False

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
        self.confirmed_end_distance = -1.0
        self.line_version = 0
        self.initial_line_hold_pose = None

        self.hold_target = None
        self.stable_since = None
        self.current_path_s = 0.0
        self.projected_path_s = 0.0
        self.completed_path_length = 0.0
        self.endpoint_hold_started = None
        self.endpoint_candidate_point = None
        self.endpoint_candidate_count = 0

        self.actual_trajectory = []
        self.current_tracking_point = None
        self.active_los_target_s = None
        self.active_los_target = None
        self.active_los_yaw = None
        self.active_los_phase = None
        self.active_los_curve_version = 0
        self.latest_line_points = []
        self.latest_line_input_count = 0
        self.latest_line_valid_count = 0
        self.latest_line_used_count = 0
        self.latest_line_confidence = 0.0
        self.latest_line_status = "none"
        self.web_valid_point_history = deque()
        self.web_valid_point_cells = set()
        self.web_valid_point_lock = threading.Lock()
        self.last_trajectory_publish_time = rospy.Time(0)

        self.open_data_log()

        # 状态字段全部就绪后再订阅，避免构造期间首帧回调访问未初始化字段。
        rospy.Subscriber(
            self.line_topic, LineDetection, self.line_callback, queue_size=1
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
        rospy.loginfo(
            "%s: 红线接口=auv_control/LineDetection；置信度达标后使用 positions 全部有限点；下限=%.2f",
            NODE_NAME,
            self.line_min_confidence,
        )

    def open_data_log(self):
        """为本次运行创建独立 YAML 数据文件；失败时不影响任务控制。"""
        try:
            self.data_logger = TimestampedYamlLogger(
                NODE_NAME, self.log_directory
            )
            self.data_log_path = self.data_logger.path
            self.write_data_record(
                "startup",
                log_directory=self.log_directory,
                line_min_confidence=self.line_min_confidence,
                los_midpoint_ratio=self.los_midpoint_ratio,
                endpoint_min_completed_path_length=(
                    self.endpoint_min_completed_path_length
                ),
            )
            rospy.loginfo("%s: 完整数据文件=%s", NODE_NAME, self.data_log_path)
        except OSError as error:
            self.data_logger = None
            rospy.logwarn("%s: 无法创建完整数据文件: %s", NODE_NAME, error)

    def shutdown(self):
        self.cancel_motion()
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

    @staticmethod
    def point_record(point):
        if point is None:
            return None
        return [round(point.x, 6), round(point.y, 6), round(point.z, 6)]

    @staticmethod
    def pose_record(pose):
        if pose is None:
            return None
        return {
            "position": Task1LineFollowTest.point_record(pose.pose.position),
            "yaw_deg": round(math.degrees(
                yaw_from_quaternion(pose.pose.orientation)
            ), 6),
        }

    def set_state(self, state):
        if self.state != state:
            rospy.loginfo("%s: 阶段 %s -> %s", NODE_NAME, self.state, state)
            self.write_data_record(
                "state_change", previous_state=self.state, next_state=state
            )
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

    def get_frame_pose(self, frame):
        try:
            self.tf_listener.waitForTransform(
                self.map_frame, frame, rospy.Time(0), rospy.Duration(1.0)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                self.map_frame, frame, rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0,
                "%s: 无法获取 %s -> %s 位姿: %s",
                NODE_NAME,
                self.map_frame,
                frame,
                error,
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    def get_current_pose(self):
        """返回控制器使用的 base_link 当前位姿。"""
        return self.get_frame_pose("base_link")

    def get_tracking_pose(self):
        """巡线定位、投影和到达判断统一使用 base_link。"""
        return self.get_current_pose()

    def get_camera_pose(self):
        """camera 只用于 Web 航向箭头和数据记录，不参与巡线控制。"""
        return self.get_frame_pose(self.line_tracking_frame)

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
        pose.header.frame_id = self.map_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(x, y, self.hold_z)
        pose.pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw))
        return pose

    def tracking_curve_to_base_points(self, points):
        """曲线已经是 base_link 规划轨迹，Web 直接显示。"""
        return [copy.deepcopy(point) for point in points]

    def publish_motion_goal(self, target):
        goal = copy.deepcopy(target)
        goal.header.frame_id = self.map_frame
        goal.header.stamp = rospy.Time.now()
        self.last_motion_goal = copy.deepcopy(goal)
        self.motion_goal_pub.publish(goal)

    def publish_dprov(self, target):
        """兼容原状态机函数名；v3 实际发布 motion_supervisor 目标。"""
        self.publish_motion_goal(target)

    def publish_los_goal(self, los_point, desired_yaw):
        """只在 LOS 产生新点时计算一次 base_link 到该点的中点目标。"""
        current = self.get_current_pose()
        if current is None:
            return None
        ratio = self.los_midpoint_ratio
        target_x = current.pose.position.x + ratio * (
            los_point.x - current.pose.position.x
        )
        target_y = current.pose.position.y + ratio * (
            los_point.y - current.pose.position.y
        )
        goal = self.make_pose(target_x, target_y, desired_yaw)
        self.last_los_goal = copy.deepcopy(goal)
        self.publish_motion_goal(goal)
        self.write_data_record(
            "los_goal_update",
            base=self.pose_record(current),
            los_target=self.point_record(los_point),
            command_goal=self.pose_record(goal),
            target_s=round(self.active_los_target_s, 6)
            if self.active_los_target_s is not None else None,
            curve_version=self.tracking_curve_version,
        )
        return goal

    def publish_position_target(self, point, yaw):
        self.publish_dprov(self.make_pose(point.x, point.y, yaw))

    def motion_state_fresh(self):
        return (
            self.latest_motion_state is not None
            and (rospy.Time.now() - self.latest_motion_state.header.stamp).to_sec()
            <= self.motion_state_timeout
        )

    def motion_arrived(self):
        return (
            self.motion_state_fresh()
            and self.latest_motion_state.state == MotionState.HOVER
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

    def tracking_yaw_at_s(self, target_s):
        """以当前航点到下一前视点的方向作为机器人目标航向。"""
        curve_end_s = self.tracking_curve_s[-1]
        target = self.tracking_point_at_s(target_s)
        next_s = min(curve_end_s, target_s + self.los_lookahead_distance)
        next_point = self.tracking_point_at_s(next_s)
        if xy_distance(target, next_point) > 1e-6:
            return math.atan2(next_point.y - target.y, next_point.x - target.x)

        previous_s = max(0.0, target_s - self.los_lookahead_distance)
        previous = self.tracking_point_at_s(previous_s)
        return math.atan2(target.y - previous.y, target.x - previous.x)

    def clear_active_los_target(self):
        self.active_los_target_s = None
        self.active_los_target = None
        self.active_los_yaw = None
        self.active_los_phase = None

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
        self.clear_active_los_target()
        if reset_progress:
            self.current_path_s = 0.0
            self.projected_path_s = 0.0
        elif current is not None:
            # 新旧曲线都从已锁定起点向前编号。正式进度只由已经完成“移动+对向”
            # 的航点推进；最近投影只用于 Web 观察，不能跳过尚未完成的航点。
            self.current_path_s = min(
                self.completed_path_length, self.tracking_curve_s[-1]
            )
            projection = self.project_to_curve(
                current.pose.position,
                self.tracking_curve_points,
                self.tracking_curve_s,
            )
            self.projected_path_s = max(
                self.current_path_s,
                projection["path_s"] if projection is not None else 0.0,
            )
        return True

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

    def freeze_confirmed_curve(self):
        """把经连续高置信帧确认的当前拟合曲线整体冻结，并形成新 LOS 版本。"""
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
            "confirmed_end_distance",
            "endpoint_candidate_point",
            "endpoint_candidate_count",
        )

    def reset_tentative_line_state(self):
        """首次锁线失败后清除全部暂定点集、曲线、方向和终点证据。"""
        with self.curve_lock:
            self.line_reference_point = None
            self.line_axis = None
            self.line_raw_points = []
            self.line_curve_points = []
            self.line_curve_s = []
            self.line_start_point = None
            self.line_end_point = None
            self.line_fit_residual = 0.0
            self.confirmed_end_distance = -1.0
            self.endpoint_candidate_point = None
            self.endpoint_candidate_count = 0
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
            trial.confirmed_end_distance = -1.0
            trial.endpoint_candidate_point = None
            trial.endpoint_candidate_count = 0
            trial.line_locked = False
        elif reference is not None:
            trial.line_reference_point = copy.deepcopy(reference)

        trial.fuse_line_points(points)
        if not trial.fit_line_curve():
            return False

        with self.curve_lock:
            for field in self.fit_state_fields():
                setattr(self, field, copy.deepcopy(getattr(trial, field)))
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
        # 所有高置信点均视为合理；这里只排除无法形成曲线的数值零跨度。
        if span <= 1e-6:
            return False

        samples = np.linspace(
            float(parameters[0]),
            float(parameters[-1]),
            self.line_curve_sample_count,
        )
        local_start_point = min(
            self.line_raw_points,
            key=lambda point: xy_distance(point, self.line_reference_point),
        )
        local_end_point = max(
            self.line_raw_points,
            key=lambda point: (
                (point.x - center[0]) * axis[0]
                + (point.y - center[1]) * axis[1]
            ),
        )
        selected_start = self.line_start_point
        if (
            self.line_start_point is None
            or (
                not self.line_locked
                and xy_distance(local_start_point, self.line_reference_point)
                < xy_distance(self.line_start_point, self.line_reference_point)
            )
        ):
            selected_start = copy.deepcopy(local_start_point)

        curve = None
        selected_axis = axis
        selected_end = local_end_point
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
                        self.hold_z,
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
            selected_end = candidate_end
            break

        if curve is None:
            return False
        self.line_axis = selected_axis
        self.line_start_point = copy.deepcopy(selected_start)
        local_end_point = selected_end
        curve_s = self.cumulative_distance(curve)
        if curve_s[-1] <= 1e-6:
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

    def update_endpoint_evidence(self, observed_points, tracking_point):
        """base_link 接近最远点后，再次看到该点才累计稳定帧。"""
        if self.line_end_point is None or self.line_reference_point is None:
            return
        if xy_distance(
            tracking_point, self.line_end_point
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

    def record_line_frame(self, status, confidence, reason="", points=None):
        self.latest_line_confidence = confidence
        self.latest_line_status = status if not reason else "%s:%s" % (
            status, reason
        )
        self.write_data_record(
            "line_frame",
            status=status,
            reason=reason,
            confidence=round(confidence, 6),
            input_count=self.latest_line_input_count,
            valid_count=self.latest_line_valid_count,
            used_count=self.latest_line_used_count,
            points=[self.point_record(point) for point in (points or [])],
            fitted_curve=[
                self.point_record(point) for point in self.line_curve_points
            ],
            fixed_curve=[
                self.point_record(point)
                for point in self.line_committed_curve_points
            ],
            curve_version=self.line_version,
        )

    def line_callback(self, message):
        if self.state == self.WAIT_CAMERA or not self.camera_ready():
            return
        if self.hold_z is None and not self.initialize_start_pose():
            return
        if message.class_name and message.class_name not in self.line_classes:
            return

        self.latest_line_input_count = int(message.point_count)
        self.latest_line_valid_count = int(message.valid_count)
        self.latest_line_used_count = 0
        confidence = self.normalized_confidence(message.conf)
        time_reason = self.line_message_time_reason(message.header.stamp)
        if time_reason is not None:
            self.record_line_frame("rejected", confidence, time_reason)
            return
        if confidence < self.line_min_confidence:
            self.record_line_frame(
                "rejected", confidence, "confidence_below_minimum"
            )
            rospy.loginfo_throttle(
                3.0,
                "%s: 红线消息已忽略，conf=%.2f 低于下限 %.2f",
                NODE_NAME,
                confidence,
                self.line_min_confidence,
            )
            return

        camera_poses, reason = self.line_poses(message)
        if camera_poses is None:
            self.record_line_frame("rejected", confidence, reason)
            rospy.loginfo_throttle(
                3.0, "%s: 红线消息已忽略，原因=%s", NODE_NAME, reason
            )
            return

        transformed, reason = self.transform_frame_to_map(camera_poses)
        if transformed is None:
            self.record_line_frame("rejected", confidence, reason)
            return
        # Web 始终显示最近一条可转换到 map 的多点识别消息。
        self.latest_line_points = [
            copy.deepcopy(pose.pose.position) for pose in transformed
        ]
        self.retain_web_valid_points(self.latest_line_points)

        current = self.get_current_pose()
        tracking = self.get_tracking_pose()
        if current is None or tracking is None:
            self.record_line_frame(
                "rejected", confidence, "base_link_pose_unavailable"
            )
            return
        reference = (
            self.line_reference_point
            if self.line_reference_point is not None
            else tracking.pose.position
        )
        points = self.ordered_line_points(transformed, reference)

        now = rospy.Time.now()
        if not self.line_locked:
            if self.line_lock_candidate is None:
                self.line_lock_candidate = {
                    "started": now,
                    "confidence": confidence,
                    "points": points,
                    "reference": copy.deepcopy(tracking.pose.position),
                    "hold_pose": copy.deepcopy(current),
                }
            else:
                self.line_lock_candidate["confidence"] = max(
                    self.line_lock_candidate["confidence"], confidence
                )
                self.line_lock_candidate["points"].extend(
                    copy.deepcopy(points)
                )
            self.record_line_frame("lock_candidate", confidence, points=points)
            return

        # conf 已通过硬下限后，positions 全部点直接进入融合和拟合；旧的曲线关联、
        # 内点比例、延伸方向及固定段重复点过滤均按实测要求停用。
        if self.fit_points_transaction(points):
            self.freeze_confirmation_count = min(
                self.curve_freeze_required_frames,
                self.freeze_confirmation_count + 1,
            )
            self.freeze_confirmed_curve()
            self.update_endpoint_evidence(points, tracking.pose.position)
            self.record_line_frame("accepted", confidence, points=points)
        else:
            self.freeze_confirmation_count = 0
            self.record_line_frame(
                "rejected", confidence, "curve_fit_failed", points
            )

    def try_lock_line(self):
        if self.line_locked or self.line_lock_candidate is None:
            return False
        candidate = self.line_lock_candidate
        if (
            rospy.Time.now() - candidate["started"]
        ).to_sec() < self.line_lock_window_seconds:
            return False

        if not self.fit_points_transaction(
            candidate["points"], reference=candidate["reference"], fresh=True
        ):
            self.line_lock_candidate = None
            self.reset_tentative_line_state()
            return False
        self.line_lock_confidence = candidate["confidence"]
        self.line_locked = True
        self.line_lock_candidate = None
        self.initial_line_hold_pose = copy.deepcopy(candidate["hold_pose"])
        self.freeze_confirmation_count = 1
        self.freeze_confirmed_curve()
        self.current_tracking_point = copy.deepcopy(self.line_start_point)
        tracking = self.get_tracking_pose()
        if tracking is not None:
            self.update_endpoint_evidence(
                candidate["points"], tracking.pose.position
            )
        distance = (
            xy_distance(tracking.pose.position, self.line_start_point)
            if tracking is not None else float("nan")
        )
        rospy.loginfo(
            "%s: 识别状态=已锁定；conf=%.2f，base_link=(%.2f, %.2f)，"
            "起点=(%.2f, %.2f)，当前最远点=(%.2f, %.2f)，距起点=%.2f m",
            NODE_NAME,
            self.line_lock_confidence,
            tracking.pose.position.x if tracking is not None else float("nan"),
            tracking.pose.position.y if tracking is not None else float("nan"),
            self.line_start_point.x,
            self.line_start_point.y,
            self.line_end_point.x,
            self.line_end_point.y,
            distance,
        )
        if self.activate_latest_tracking_curve(reset_progress=True):
            rospy.loginfo(
                "%s: 首段固定曲线已就绪，直接开始 base_link 中点连续巡线",
                NODE_NAME,
            )
            self.hold_target = None
            self.last_los_goal = None
            self.clear_active_los_target()
            self.set_state(self.FOLLOW_LINE)
        else:
            self.set_state(self.WAIT_FIXED_LINE)
            rospy.loginfo(
                "%s: 首条红线已拟合，等待后续高置信帧确认固定（%d/%d）",
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
                "%s: 首段曲线已固定，固定长度=%.2f m，开始连续巡线",
                NODE_NAME,
                self.tracking_curve_s[-1],
            )
            self.hold_target = None
            self.last_los_goal = None
            self.clear_active_los_target()
            self.set_state(self.FOLLOW_LINE)

    def set_search_state(self, state):
        self.set_state(state)
        self.search_target = None

    def hover_confirmed(self):
        """只采用 MotionState.HOVER，并要求状态连续保持配置时长。"""
        if not self.motion_arrived():
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = rospy.Time.now()
        return (
            rospy.Time.now() - self.stable_since
        ).to_sec() >= self.motion_hover_confirm_seconds

    def run_search_rotation(self, current, yaw_offset, next_state, label):
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
        if self.hover_confirmed():
            if self.extension_search_active and self.endpoint_finish_ready():
                self.set_state(self.FINISH)
                return
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
        if self.hover_confirmed():
            self.search_cycle_anchor = None
            self.set_search_state(self.SEARCH_LEFT)

    def run_search(self):
        current = self.get_current_pose()
        if current is None:
            return
        if (
            self.extension_search_active
            and self.line_version > self.tracking_curve_version
            and self.activate_latest_tracking_curve(current=current)
        ):
            rospy.loginfo(
                "%s: 搜索期间发现新的固定曲线，恢复连续 LOS 巡线",
                NODE_NAME,
            )
            self.extension_search_active = False
            self.search_cycle_anchor = None
            self.search_target = None
            self.hold_target = None
            self.last_los_goal = None
            self.clear_active_los_target()
            self.set_state(self.FOLLOW_LINE)
            return
        rospy.loginfo_throttle(3.0, "%s: 识别状态=未识别", NODE_NAME)
        if self.state == self.SEARCH_LEFT:
            self.run_search_rotation(
                current,
                -self.search_yaw_sign * self.search_yaw_angle,
                self.SEARCH_RIGHT,
                "向左旋转",
            )
        elif self.state == self.SEARCH_RIGHT:
            self.run_search_rotation(
                current,
                self.search_yaw_sign * self.search_yaw_angle,
                self.SEARCH_RETURN,
                "向右旋转",
            )
        elif self.state == self.SEARCH_RETURN:
            self.run_search_rotation(
                current,
                0.0,
                self.SEARCH_LEFT if self.extension_search_active
                else self.SEARCH_FORWARD,
                "返回搜索基准航向",
            )
        elif self.state == self.SEARCH_FORWARD:
            self.run_search_forward(current)

    def enter_hold_end(self, current):
        active_end = self.tracking_curve_points[-1]
        self.hold_target = copy.deepcopy(self.last_los_goal)
        if self.hold_target is None:
            endpoint_yaw = self.tracking_yaw_at_s(self.tracking_curve_s[-1])
            self.active_los_target_s = self.tracking_curve_s[-1]
            self.active_los_target = copy.deepcopy(active_end)
            self.active_los_yaw = endpoint_yaw
            self.hold_target = self.publish_los_goal(active_end, endpoint_yaw)
        if self.hold_target is None:
            return
        self.publish_dprov(self.hold_target)
        self.endpoint_hold_started = rospy.Time.now()
        self.extension_search_active = False
        self.active_los_phase = "WAIT_FINAL_HOVER"
        rospy.loginfo(
            "%s: 已无新的 LOS 目标；保持最后中点目标，等待控制器完成平移和最终转向",
            NODE_NAME,
        )
        self.set_state(self.HOLD_END)

    def run_follow_line(self):
        current = self.get_current_pose()
        tracking = self.get_tracking_pose()
        if current is None or tracking is None or not self.tracking_curve_ready():
            return

        previous_version = self.tracking_curve_version
        if (
            self.line_version > self.tracking_curve_version
            and self.activate_latest_tracking_curve(current=tracking)
        ):
            rospy.loginfo(
                "%s: 固定曲线更新 %d -> %d，立即更新 LOS 目标",
                NODE_NAME,
                previous_version,
                self.tracking_curve_version,
            )

        projection = self.project_to_curve(
            tracking.pose.position,
            self.tracking_curve_points,
            self.tracking_curve_s,
        )
        if projection is None:
            return

        self.projected_path_s = max(self.current_path_s, projection["path_s"])
        self.current_path_s = self.projected_path_s
        self.completed_path_length = max(
            self.completed_path_length, self.current_path_s
        )
        active_end = self.tracking_curve_points[-1]
        endpoint_distance = xy_distance(tracking.pose.position, active_end)
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        next_target_s = min(
            self.tracking_curve_s[-1],
            self.current_path_s + self.los_lookahead_distance,
        )
        next_target = self.tracking_point_at_s(next_target_s)
        next_yaw = self.tracking_yaw_at_s(next_target_s)
        target_changed = (
            self.active_los_target is None
            or self.active_los_curve_version != self.tracking_curve_version
            or (
                next_target_s > (self.active_los_target_s or 0.0) + 1e-6
                and xy_distance(next_target, self.active_los_target)
                >= self.los_target_update_min_distance
            )
        )

        if target_changed:
            self.active_los_target_s = next_target_s
            self.active_los_target = copy.deepcopy(next_target)
            self.active_los_yaw = next_yaw
            self.active_los_phase = "CONTINUOUS_TRANSLATION"
            self.active_los_curve_version = self.tracking_curve_version
            commanded_goal = self.publish_los_goal(next_target, next_yaw)
        else:
            commanded_goal = self.last_los_goal
            if commanded_goal is not None:
                self.publish_motion_goal(commanded_goal)
        if commanded_goal is None:
            return

        self.current_tracking_point = copy.deepcopy(self.active_los_target)
        position_error = xy_distance(
            tracking.pose.position, self.active_los_target
        )
        yaw_error = abs(wrap_angle(self.active_los_yaw - current_yaw))

        # 小于 los_target_update_min_distance 的末端更新本来就不会重新下发；
        # 终点判断使用同一容差，否则活动 LOS 点可能永远差几厘米而无法进入 HOLD_END。
        remaining_curve_length = max(
            0.0, self.tracking_curve_s[-1] - self.active_los_target_s
        )
        at_known_end = (
            remaining_curve_length <= self.los_target_update_min_distance
            and self.line_version <= self.tracking_curve_version
        )
        if at_known_end:
            self.enter_hold_end(current)
            return

        state_name = (
            str(self.latest_motion_state.state)
            if self.motion_state_fresh()
            else "无新鲜反馈"
        )
        rospy.loginfo_throttle(
            2.0,
            "%s: LOS 连续巡线；已完成/投影=%.2f/%.2f m，"
            "已知轨迹=%.2f m，距末端=%.2f m，base目标误差=%.2f m，"
            "航向误差=%.1f deg，"
            "LOS点=(%.2f, %.2f)，中点目标=(%.2f, %.2f)，motion_state=%s",
            NODE_NAME,
            self.completed_path_length,
            self.projected_path_s,
            self.tracking_curve_s[-1],
            endpoint_distance,
            position_error,
            math.degrees(yaw_error),
            self.active_los_target.x,
            self.active_los_target.y,
            commanded_goal.pose.position.x,
            commanded_goal.pose.position.y,
            state_name,
        )

    def endpoint_finish_ready(self):
        if self.endpoint_hold_started is None:
            return False
        fixed_length = (
            self.line_committed_curve_s[-1]
            if self.line_committed_curve_s else 0.0
        )
        pending_extension = max(
            0.0,
            (self.line_curve_s[-1] if self.line_curve_s else 0.0)
            - fixed_length,
        )
        no_growth_seconds = (
            rospy.Time.now() - self.endpoint_hold_started
        ).to_sec()
        return (
            self.completed_path_length
            >= self.endpoint_min_completed_path_length
            and pending_extension < self.curve_freeze_min_advance
            and self.endpoint_confirmed()
            and no_growth_seconds >= self.endpoint_confirm_seconds
        )

    def run_hold_end(self):
        current = self.get_current_pose()
        tracking = self.get_tracking_pose()
        if current is None or tracking is None:
            return
        # line_version 只在后续帧确认并冻结新曲线时增加。
        grew = self.line_version > self.tracking_curve_version
        if grew:
            old_version = self.tracking_curve_version
            if self.activate_latest_tracking_curve(current=tracking):
                rospy.loginfo(
                    "%s: 等待期间发现新固定红线；版本 %d -> %d，立即恢复连续巡线",
                    NODE_NAME,
                    old_version,
                    self.tracking_curve_version,
                )
                self.hold_target = None
                self.last_los_goal = None
                self.clear_active_los_target()
                self.set_state(self.FOLLOW_LINE)
                return

        active_end = self.tracking_curve_points[-1]
        if self.hold_target is None:
            return
        self.current_tracking_point = copy.deepcopy(active_end)
        self.publish_dprov(self.hold_target)
        distance = xy_distance(tracking.pose.position, active_end)
        stable = self.hover_confirmed()
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
            "%s: 最后目标保持；LOS末端距离=%.2f m，HOVER连续确认=%s，"
            "已完成/结束下限=%.2f/%.2f m，重复观测=%d/%d，"
            "待固定延伸=%.2f m，确认=%d/%d，轨迹未增长=%.1f/%.1f s",
            NODE_NAME,
            distance,
            "是" if stable else "否",
            self.completed_path_length,
            self.endpoint_min_completed_path_length,
            self.endpoint_candidate_count,
            self.endpoint_stable_frames,
            pending_extension,
            self.freeze_confirmation_count,
            self.curve_freeze_required_frames,
            no_growth_seconds,
            self.endpoint_confirm_seconds,
        )
        if stable and self.endpoint_finish_ready():
            self.set_state(self.FINISH)
            return
        if stable:
            self.search_base_yaw = yaw_from_quaternion(
                self.hold_target.pose.orientation
            )
            self.search_cycle_anchor = copy.deepcopy(current.pose.position)
            self.extension_search_active = True
            rospy.loginfo(
                "%s: 最后目标已由下位机接管 %.1f s；固定当前位置，开始左右旋转搜索新目标",
                NODE_NAME,
                self.motion_hover_confirm_seconds,
            )
            self.set_search_state(self.SEARCH_LEFT)

    def publish_trajectory_status(self):
        now = rospy.Time.now()
        if (
            now - self.last_trajectory_publish_time
        ).to_sec() < self.trajectory_publish_period:
            return
        current = self.get_current_pose()
        camera = self.get_camera_pose()
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

        planned_base_curve = self.tracking_curve_to_base_points(
            self.line_curve_points
        )
        fixed_base_curve = self.tracking_curve_to_base_points(
            self.line_committed_curve_points
        )
        tracking_base_curve = self.tracking_curve_to_base_points(
            self.tracking_curve_points
        )
        base_target = (
            self.last_motion_goal.pose.position
            if self.last_motion_goal is not None else None
        )
        endpoint_base_candidate = None
        if (
            self.endpoint_candidate_point is not None
            and len(self.tracking_curve_points) >= 2
        ):
            endpoint_base_candidate = copy.deepcopy(self.endpoint_candidate_point)
        web_valid_points = self.web_valid_points_snapshot()

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
            "camera": point_data(camera.pose.position) if camera else None,
            "robot_down": round(current.pose.position.z, 3) if current else None,
            "camera_down": round(camera.pose.position.z, 3) if camera else None,
            "robot_yaw_deg": round(math.degrees(yaw_from_quaternion(
                current.pose.orientation
            )), 2) if current else 0.0,
            "tracking_frame": "base_link",
            "tracking_point": point_data(base_target),
            "los_target": point_data(self.current_tracking_point),
            "los_phase": self.active_los_phase,
            "tracking_curve_version": self.tracking_curve_version,
            "fixed_curve_version": self.line_version,
            "freeze_confirmations": self.freeze_confirmation_count,
            "freeze_required": self.curve_freeze_required_frames,
            "line_input_point_count": self.latest_line_input_count,
            "line_valid_point_count": self.latest_line_valid_count,
            "line_used_point_count": self.latest_line_used_count,
            "latest_line_points": [
                point_data(point) for point in self.latest_line_points
            ],
            "valid_line_point_history": [
                point_data(point) for point in web_valid_points
            ],
            "valid_line_point_history_count": len(web_valid_points),
            "actual_path": [point_data(point) for point in self.actual_trajectory],
            "planned_curve": [point_data(point) for point in planned_base_curve],
            "fixed_curve": [point_data(point) for point in fixed_base_curve],
            "tracking_curve": [point_data(point) for point in tracking_base_curve],
            "raw_line": [point_data(point) for point in self.line_raw_points],
            "line_start": point_data(
                fixed_base_curve[0] if fixed_base_curve else None
            ),
            "line_end": point_data(
                fixed_base_curve[-1] if fixed_base_curve else None
            ),
            "endpoint_candidate": point_data(endpoint_base_candidate),
            "endpoint_stable_count": self.endpoint_candidate_count,
            "endpoint_stable_required": self.endpoint_stable_frames,
            "endpoint_confirmed": self.endpoint_confirmed(),
            "projected_length": round(self.projected_path_s, 3),
        }
        encoded = json.dumps(payload, separators=(",", ":"))
        self.trajectory_pub.publish(String(data=encoded))
        if self.trajectory_web is not None:
            self.trajectory_web.update(encoded)
        self.last_trajectory_publish_time = now

    def log_debug_cycle(self):
        """以 DEBUG 级别记录每个控制周期的完整诊断。"""
        current = self.get_current_pose()
        camera = self.get_camera_pose()
        if current is None:
            return
        target = self.current_tracking_point
        goal = self.last_motion_goal
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        target_yaw = (
            self.active_los_yaw
            if self.active_los_yaw is not None
            else (
                yaw_from_quaternion(goal.pose.orientation)
                if goal is not None else current_yaw
            )
        )
        position_error = (
            xy_distance(current.pose.position, target)
            if target is not None else float("nan")
        )
        yaw_error = (
            math.degrees(abs(wrap_angle(target_yaw - current_yaw)))
            if target is not None else float("nan")
        )
        motion_state = self.latest_motion_state
        rospy.logdebug(
            "%s: FULL state=%s base=(%.3f,%.3f,%.3f,%.2fdeg) "
            "camera=(%.3f,%.3f,%.3f) los_target=(%.3f,%.3f) "
            "base_goal=(%.3f,%.3f,%.3f,%.2fdeg) error=(%.3fm,%.2fdeg) "
            "path=%.3f/%.3f/%.3f active=(%s,%s) "
            "line=(%s,%.3f,%.3f,%s/%s,points=%d/%d/%d) "
            "motion=(%s,%s,%s,%s,%s)",
            NODE_NAME,
            self.state,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(current_yaw),
            camera.pose.position.x if camera is not None else float("nan"),
            camera.pose.position.y if camera is not None else float("nan"),
            camera.pose.position.z if camera is not None else float("nan"),
            target.x if target is not None else float("nan"),
            target.y if target is not None else float("nan"),
            goal.pose.position.x if goal is not None else float("nan"),
            goal.pose.position.y if goal is not None else float("nan"),
            goal.pose.position.z if goal is not None else float("nan"),
            math.degrees(yaw_from_quaternion(goal.pose.orientation))
            if goal is not None else float("nan"),
            position_error,
            yaw_error,
            self.completed_path_length,
            self.projected_path_s,
            self.tracking_curve_s[-1] if self.tracking_curve_s else 0.0,
            "%.3f" % self.active_los_target_s
            if self.active_los_target_s is not None else "-",
            self.active_los_phase or "-",
            self.line_locked,
            self.line_lock_confidence,
            self.line_fit_residual,
            self.tracking_curve_version,
            self.line_version,
            self.latest_line_input_count,
            self.latest_line_valid_count,
            self.latest_line_used_count,
            motion_state.state if motion_state is not None else "-",
            motion_state.reason if motion_state is not None else "-",
            motion_state.tx if motion_state is not None else 0,
            motion_state.ty if motion_state is not None else 0,
            motion_state.mz if motion_state is not None else 0,
        )
        self.write_data_record(
            "control_cycle",
            base=self.pose_record(current),
            camera=self.pose_record(camera),
            los_target=self.point_record(target),
            command_goal=self.pose_record(goal),
            position_error=round(position_error, 6)
            if math.isfinite(position_error) else None,
            yaw_error_deg=round(yaw_error, 6)
            if math.isfinite(yaw_error) else None,
            completed_path=round(self.completed_path_length, 6),
            projected_path=round(self.projected_path_s, 6),
            fitted_curve=[
                self.point_record(point) for point in self.line_curve_points
            ],
            fixed_curve=[
                self.point_record(point)
                for point in self.line_committed_curve_points
            ],
            executing_curve=[
                self.point_record(point) for point in self.tracking_curve_points
            ],
            latest_line_points=[
                self.point_record(point) for point in self.latest_line_points
            ],
            latest_line_confidence=round(self.latest_line_confidence, 6),
            latest_line_status=self.latest_line_status,
            tracking_curve_version=self.tracking_curve_version,
            fixed_curve_version=self.line_version,
            motion={
                "state": motion_state.state if motion_state is not None else None,
                "reason": motion_state.reason if motion_state is not None else "",
                "tx": motion_state.tx if motion_state is not None else 0,
                "ty": motion_state.ty if motion_state is not None else 0,
                "mz": motion_state.mz if motion_state is not None else 0,
            },
        )

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
                self.log_debug_cycle()
                self.rate.sleep()
                continue

            self.try_lock_line()
            self.publish_trajectory_status()

            if self.line_lock_candidate is not None and not self.line_locked:
                # 首帧候选出现后立即停止搜索，短暂定点累计窗口内高置信点。
                self.current_tracking_point = copy.deepcopy(
                    self.line_lock_candidate["points"][0]
                )
                self.publish_dprov(self.line_lock_candidate["hold_pose"])
                rospy.loginfo_throttle(
                    2.0,
                    "%s: 识别状态=候选已发现；定点累计高置信红线点",
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
                hover_ready = self.hover_confirmed()
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
                    and hover_ready
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
            elif self.state == self.FOLLOW_LINE:
                self.run_follow_line()
            elif self.state == self.HOLD_END:
                self.run_hold_end()
            elif self.state == self.FINISH:
                self.finish()

            self.log_debug_cycle()
            self.rate.sleep()


def main():
    rospy.init_node(NODE_NAME)
    Task1LineFollowTest().run()


if __name__ == "__main__":
    main()
