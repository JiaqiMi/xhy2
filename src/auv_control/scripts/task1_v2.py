#! /home/xhy/xhy_env/bin/python
"""
名称：task1_v2.py
功能：2026 Task 1——主管道检修
描述：
    1. 任务启动时记录当前位姿，按比赛约定认为 map 原点已经位于红色圆形处；
    2. 从起点向前搜索底部红色长线，首次看到管线时记录红线起点；
    3. 根据红色长线识别结果拟合全局巡线路径，运行高度统一使用参考高度；
    4. 巡线过程中稳定识别黄色圆形/三角形和黑色方形，并在图形上方执行动作；
    5. 黄色图形亮红灯，黑色图形亮绿灯并根据实时航向累计旋转角度；
    6. 红色长线丢失超过设定时间后记录终点并结束任务。
监听：/obj/target_message，/obj/line_message，/left/image_raw，/status/vel（可选），/tf
发布：/cmd/pose/ned，/cmd/actuator，/task1/trajectory，/finished
说明：
    - 识别类别按感知方话题格式默认使用 triangle、circle、rectangle、line；
    - /task1_v2_reference_height 可设置相对红色圆形原点的运行高度，默认 0.4 m；
      map 为 NED 坐标，z/down 向下为正，因此默认运行目标 z 为 -0.4；
    - /task1_v2_initial_heading_deg 可设置搜索红线起点的初始航向，0 度沿 map +X；
    - 巡线阶段使用 PoseNEDcmd 的定深定向手控模式，按拟合曲线执行 LOS；
    - 图形只作为巡线过程中的触发点，不直接导航到图形位置；
    - 到达图形上方后使用 PoseNEDcmd 的动力定位 ROV 模式保持当前动作位姿；
    - 轨迹网页默认监听 192.168.1.117:8082；
    - 本文件自行实现发布、TF、当前位姿、外设控制、任务完成和状态机计时，
      不依赖 task_v2_common.py / MissionBase。

修改记录：
    2026.7.13：
        1. 将任务起点改为启动时当前位姿，符合 map 已在红色圆形处建系的约定；
        2. 增加参考运行高度参数，巡线和图形动作均只控制 XY，z 固定为参考运行高度；
        3. 将运动控制改为先转向目标点，再向目标点移动；
        4. 图形识别改为约 10 帧聚类稳定后入队，完成动作后按位置去重；
        5. 灯光动作改为每次亮灯 3 秒；
        6. 黑色图形旋转改为根据当前航向反馈累计角度，避免按命令步长误判；
        7. 增加红线起点、终点记录，红线结束后结束任务；
        8. 增加图形动作后的红线重捕获宽限，避免动作期间视觉中断导致提前结束；
        9. 降低黑色图形旋转默认航向步长，减少旋转时机器人中心漂移。
    2026.7.14：
        1. 合并 main 最新框架，执行器下行话题调整为 /cmd/actuator；
        2. 解决 task1_v2.py 合并冲突，保留当前 Task1 状态机和稳定识别逻辑。
        3. 修复最新 MissionBase 中不存在 publish_target 方法导致的目标发布错误。
        4. 移除 MissionBase 依赖，将 Task1 所需公共功能全部放入本文件。
        5. 运行高度改为相对红色圆形原点的参考高度，默认 0.4 m，便于现场修改。
        6. 巡线改为 PoseNEDcmd 手控模式，红线点集限量后拟合为曲线并进行 LOS 跟踪。
        7. 巡线日志不再输出未知总路径进度，改为输出已完成路径长度。
        8. 增加真实下水调试用节流日志，覆盖状态、感知、曲线、控制和动作执行。
        9. 图形识别后不直接前往图形坐标，改为沿巡线曲线到达图形上方后再触发动作。
        10. 增加可人工配置的初始搜索航向，任务开始按该航向寻找红色长线。
    2026.7.16：
        1. 按感知最新约定将 Pose1 作为远点、Pose3 作为近点，内部按近中远顺序融合；
        2. 巡线前进力默认改为 1000，横移力非零时限制在 1500~3000，并在定点前逐步制动；
        3. 增加相机就绪等待、慢速初始航向对准和所有任务过渡的定点稳定等待；
        4. 无红线时按 0/+30/-30/0 度定点扫描，未发现后定深定向前进 0.5 m；
        5. 图形动作前后均回到图形中心定点，黑色图形以 MZ=3000 旋转并由 TF 航向累计判定；
        6. 任务结束增加“黄色两个、黑色两个、远端点稳定、红线超时丢失”联合判据；
        7. 发布 Task1 轨迹 JSON，供测试网页显示实际轨迹、拟合曲线和图形位置。
        8. 轨迹网页默认地址调整为 192.168.1.117:8082，监听地址和端口支持 ROS 参数配置；
        9. 增加正式 task1_v2.launch，集中配置任务参数、话题和现场调试阈值。
        10. 增加黑色旋转方向、MZ 步长、减速区、慢速 MZ 和姿态反馈过滤参数。
        11. 黑色方形改为同一 rectangle 连续 3 帧且每帧置信度不低于 0.30 后入队。
        12. 轨迹网页和节流日志增加黄色、黑色图形已完成操作次数。
        13. P1/P2/P3 仅解释为当前局部管线上的近中远三点，不再视为整条管线端点。
        14. 增加局部三点几何验证、连续帧确认及距离/方向/进度关联，隔离误识别管线。
        15. 全局原始点按已确认管线前进顺序维护，不再使用首帧固定轴投影排序。
"""

import copy
import json
import math
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import rospy
import tf
from auv_control.msg import ActuatorControl, PoseNEDcmd, TargetDetection, TargetDetection3
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TwistStamped
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = 'task1_v2'
DEFAULT_REFERENCE_HEIGHT = 0.4
DEFAULT_INITIAL_HEADING_DEG = 0.0
MODE_DEPTH = 2
MODE_DEPTH_HDG = 3
MODE_DPROV = 4
MODE_NAMES = {
    MODE_DEPTH: 'depth_manual',
    MODE_DEPTH_HDG: 'depth_heading_manual',
    MODE_DPROV: 'dprov',
}


def clamp(value, lower, upper):
    """将数值限制到指定闭区间内。"""
    return max(lower, min(upper, value))


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi]，便于计算航向误差。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(quaternion):
    """从 geometry_msgs/Quaternion 中取出偏航角。"""
    angles = euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ])
    return angles[2]


def class_names(param_name, defaults):
    """读取类别参数，并统一转换为字符串列表。"""
    value = rospy.get_param(param_name, defaults)
    if isinstance(value, str):
        names = value.replace(',', ' ').split()
        return names if names else [value]
    return list(value)


def xy_distance(first, second):
    """只计算水平面距离，忽略 z 方向差异。"""
    return math.hypot(first.x - second.x, first.y - second.y)


def median(values):
    """返回一组数的中位数，用于降低识别抖动影响。"""
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


TRAJECTORY_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>Task1 实时轨迹</title>
<style>
body{margin:0;background:#101820;color:#e8f1f5;font-family:Arial,"Microsoft YaHei",sans-serif}
header{padding:12px 18px;background:#172630;display:flex;gap:28px;align-items:center}
#status{font-size:14px;color:#a8c7d5}canvas{display:block;margin:16px auto;background:#f6fbfd;border-radius:8px}
.legend{padding:0 18px 14px;text-align:center;color:#bcd0d8;font-size:13px}
</style></head><body><header><b>Task1 实时轨迹</b><span id="status">等待数据</span></header>
<canvas id="map" width="960" height="680"></canvas>
<div class="legend">蓝：实际轨迹　红：拟合巡线　灰：视觉点　黄/黑：图形　青：机器人</div>
<script>
const c=document.getElementById('map'),ctx=c.getContext('2d'),pad=45;
function pts(d){let a=[];['actual_path','planned_curve','raw_line'].forEach(k=>a=a.concat(d[k]||[]));
(d.pending_markers||[]).forEach(m=>a.push(m.point));(d.handled_markers||[]).forEach(m=>a.push(m.point));
if(d.robot)a.push(d.robot);return a}
function line(arr,color,width){if(!arr||arr.length<2)return;ctx.beginPath();arr.forEach((p,i)=>{let q=xy(p);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])});ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
let scale=1,minx=0,miny=0,maxy=1;
function xy(p){return[pad+(p[0]-minx)*scale,c.height-pad-(p[1]-miny)*scale]}
function dot(p,color,r){let q=xy(p);ctx.beginPath();ctx.arc(q[0],q[1],r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#27404c';ctx.stroke()}
function render(d){let a=pts(d);ctx.clearRect(0,0,c.width,c.height);if(!a.length)return;
let xs=a.map(p=>p[0]),ys=a.map(p=>p[1]);minx=Math.min(...xs)-.2;let maxx=Math.max(...xs)+.2;miny=Math.min(...ys)-.2;maxy=Math.max(...ys)+.2;
scale=Math.min((c.width-2*pad)/Math.max(.5,maxx-minx),(c.height-2*pad)/Math.max(.5,maxy-miny));
ctx.strokeStyle='#d5e2e8';ctx.lineWidth=1;ctx.strokeRect(pad,pad,c.width-2*pad,c.height-2*pad);
(d.raw_line||[]).forEach(p=>dot(p,'#879aa3',2));line(d.planned_curve,'#e74c3c',3);line(d.actual_path,'#1677ff',3);
(d.pending_markers||[]).forEach(m=>dot(m.point,m.type==='yellow'?'#ffd400':'#202020',7));
(d.handled_markers||[]).forEach(m=>dot(m.point,m.type==='yellow'?'#d9a800':'#555',6));
if(d.active_marker)dot(d.active_marker.point,'#ff5b9a',9);if(d.endpoint_candidate)dot(d.endpoint_candidate,'#8e44ad',7);if(d.robot)dot(d.robot,'#00cfe8',8);
let n=d.counts||{},r=d.required_counts||{};document.getElementById('status').textContent=`状态 ${d.state}　已完成 ${d.completed_length||0} m　黄色操作 ${n.yellow||0}/${r.yellow||0}　黑色操作 ${n.black||0}/${r.black||0}`}
async function tick(){try{let r=await fetch('/data',{cache:'no-store'});render(await r.json())}catch(e){}setTimeout(tick,500)}tick();
</script></body></html>"""


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """允许任务节点快速重启后重新绑定同一网页端口。"""

    allow_reuse_address = True


class Task1TrajectoryWebServer:
    """在独立线程提供只读轨迹页面，不参与控制状态机。"""

    def __init__(self, host, port):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(handler):
                if handler.path == '/data':
                    body = handler.server.latest_payload.encode('utf-8')
                    content_type = 'application/json; charset=utf-8'
                else:
                    body = TRAJECTORY_HTML.encode('utf-8')
                    content_type = 'text/html; charset=utf-8'
                handler.send_response(200)
                handler.send_header('Content-Type', content_type)
                handler.send_header('Content-Length', str(len(body)))
                handler.end_headers()
                handler.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        self.server = ReusableThreadingHTTPServer((str(host), int(port)), Handler)
        self.server.daemon_threads = True
        self.server.latest_payload = '{}'
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

    def update(self, payload):
        self.server.latest_payload = payload


class Task1V2:
    """主管道检修任务状态机。"""

    STEP_SEARCH_LINE = 0
    STEP_FOLLOW_LINE = 1
    STEP_MOVE_TO_MARKER = 2
    STEP_LIGHT_ACTION = 3
    STEP_ROTATE_BLACK = 4
    STEP_FINISH = 5
    STEP_WAIT_READY = 6
    STEP_SETTLE = 7
    STEP_NAMES = {
        STEP_SEARCH_LINE: 'SEARCH_LINE',
        STEP_FOLLOW_LINE: 'FOLLOW_LINE',
        STEP_MOVE_TO_MARKER: 'MOVE_TO_MARKER',
        STEP_LIGHT_ACTION: 'LIGHT_ACTION',
        STEP_ROTATE_BLACK: 'ROTATE_BLACK',
        STEP_FINISH: 'FINISH',
        STEP_WAIT_READY: 'WAIT_READY',
        STEP_SETTLE: 'SETTLE',
    }

    def __init__(self):
        """初始化任务参数、视觉缓存以及目标检测订阅。"""
        self.node_name = NODE_NAME
        self.pose_cmd_topic = rospy.get_param(
            '/task1_v2_pose_cmd_topic', '/cmd/pose/ned'
        )
        self.actuator_topic = rospy.get_param(
            '/task1_v2_actuator_topic', '/cmd/actuator'
        )
        self.finished_topic = rospy.get_param(
            '/task1_v2_finished_topic', '/finished'
        )
        self.target_topic = rospy.get_param(
            '/task1_v2_target_topic', '/obj/target_message'
        )
        self.line_topic = rospy.get_param(
            '/task1_v2_line_topic', '/obj/line_message'
        )
        self.pose_cmd_pub = rospy.Publisher(
            self.pose_cmd_topic, PoseNEDcmd, queue_size=10
        )
        self.finished_pub = rospy.Publisher(
            self.finished_topic, String, queue_size=10
        )
        self.device_pub = rospy.Publisher(
            self.actuator_topic, ActuatorControl, queue_size=10
        )
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(rospy.get_param('~rate_hz', 5.0))

        self.pitch_offset = math.radians(rospy.get_param('/pitch_offset', 0.0))
        self.default_heading_servo = int(rospy.get_param('/task_v2_heading_servo', 0x80))
        self.default_clamp_servo = int(rospy.get_param('/task_v2_clamp_servo', 0x00))
        self.default_drive_cmd = int(rospy.get_param('/task_v2_drive_cmd', 0))
        self.default_drive_speed = int(rospy.get_param('/task_v2_drive_speed', 0))
        self.max_xy_step = rospy.get_param('~max_xy_step', 0.5)
        self.max_yaw_step = math.radians(rospy.get_param('~max_yaw_step_deg', 2.0))
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.15)
        self.yaw_tolerance = math.radians(rospy.get_param('~yaw_tolerance_deg', 2.0))

        self.step = self.STEP_WAIT_READY
        self.step_started = rospy.Time.now()

        self.reference_height = float(rospy.get_param(
            '/task1_v2_reference_height', DEFAULT_REFERENCE_HEIGHT
        ))
        reference_z_value = rospy.get_param('/task1_v2_reference_z', 'auto')
        self.reference_z = (
            -self.reference_height
            if str(reference_z_value).strip().lower() == 'auto'
            else float(reference_z_value)
        )
        self.initial_search_yaw = math.radians(float(rospy.get_param(
            '/task1_v2_initial_heading_deg', DEFAULT_INITIAL_HEADING_DEG
        )))

        self.search_forward_step = rospy.get_param('/task1_v2_search_forward_step', 0.5)
        self.search_forward_force = rospy.get_param('/task1_v2_search_forward_force', 1000)
        self.search_slow_forward_force = rospy.get_param(
            '/task1_v2_search_slow_forward_force', 300
        )
        self.search_deceleration_distance = rospy.get_param(
            '/task1_v2_search_deceleration_distance', 0.20
        )
        self.search_scan_hold_seconds = rospy.get_param(
            '/task1_v2_search_scan_hold_seconds', 2.0
        )
        self.search_scan_offsets = [
            math.radians(float(value))
            for value in rospy.get_param(
                '/task1_v2_search_scan_offsets_deg', [0.0, 30.0, -30.0, 0.0]
            )
        ]
        self.line_lost_timeout = rospy.get_param('/task1_v2_line_lost_timeout', 5.0)
        self.curve_blind_follow_timeout = rospy.get_param(
            '/task1_v2_curve_blind_follow_timeout', 2.0
        )

        self.line_point_merge_distance = rospy.get_param(
            '/task1_v2_line_point_merge_distance', 0.15
        )
        self.line_min_point_spacing = rospy.get_param(
            '/task1_v2_line_min_point_spacing', 0.03
        )
        self.line_max_point_spacing = rospy.get_param(
            '/task1_v2_line_max_point_spacing', 3.0
        )
        self.line_middle_offset_tolerance = rospy.get_param(
            '/task1_v2_line_middle_offset_tolerance', 0.25
        )
        self.line_point_order_tolerance = rospy.get_param(
            '/task1_v2_line_point_order_tolerance', 0.15
        )
        self.line_local_max_bend = math.radians(rospy.get_param(
            '/task1_v2_line_local_max_bend_deg', 45.0
        ))
        self.line_candidate_confirm_frames = int(rospy.get_param(
            '/task1_v2_line_candidate_confirm_frames', 3
        ))
        self.line_candidate_center_distance = rospy.get_param(
            '/task1_v2_line_candidate_center_distance', 0.50
        )
        self.line_candidate_yaw_tolerance = math.radians(rospy.get_param(
            '/task1_v2_line_candidate_yaw_tolerance_deg', 20.0
        ))
        self.line_association_distance = rospy.get_param(
            '/task1_v2_line_association_distance', 0.50
        )
        self.line_association_angle = math.radians(rospy.get_param(
            '/task1_v2_line_association_angle_deg', 35.0
        ))
        self.line_association_backtrack = rospy.get_param(
            '/task1_v2_line_association_backtrack', 0.60
        )
        self.line_extension_max_gap = rospy.get_param(
            '/task1_v2_line_extension_max_gap', 1.0
        )
        self.line_curve_max_points = int(rospy.get_param(
            '/task1_v2_line_curve_max_points', 120
        ))
        self.line_curve_sample_count = int(rospy.get_param(
            '/task1_v2_line_curve_sample_count', 80
        ))
        self.line_curve_degree = int(rospy.get_param('/task1_v2_line_curve_degree', 3))
        self.line_curve_min_length = rospy.get_param('/task1_v2_line_curve_min_length', 0.4)

        self.los_lookahead_distance = rospy.get_param('/task1_v2_los_lookahead_distance', 0.6)
        self.line_end_margin = rospy.get_param('/task1_v2_line_end_margin', 0.5)
        self.manual_forward_force = rospy.get_param('/task1_v2_manual_forward_force', 1000)
        self.manual_slow_forward_force = rospy.get_param(
            '/task1_v2_manual_slow_forward_force', 300
        )
        self.manual_lateral_gain = rospy.get_param('/task1_v2_manual_lateral_gain', 6000.0)
        self.manual_min_lateral_force = rospy.get_param(
            '/task1_v2_manual_min_lateral_force', 1500
        )
        self.manual_max_lateral_force = rospy.get_param(
            '/task1_v2_manual_max_lateral_force', 3000
        )
        self.manual_lateral_deadband = rospy.get_param(
            '/task1_v2_manual_lateral_deadband', 0.03
        )
        self.manual_force_step = rospy.get_param('/task1_v2_manual_force_step', 200)
        self.manual_brake_step = rospy.get_param('/task1_v2_manual_brake_step', 300)
        self.manual_slow_yaw_error = math.radians(
            rospy.get_param('/task1_v2_manual_slow_yaw_error_deg', 20.0)
        )
        self.manual_slow_lateral_error = rospy.get_param(
            '/task1_v2_manual_slow_lateral_error', 0.25
        )
        self.manual_tx_sign = rospy.get_param('/task1_v2_manual_tx_sign', 1.0)
        self.manual_ty_sign = rospy.get_param('/task1_v2_manual_ty_sign', 1.0)
        self.marker_slow_distance = rospy.get_param('/task1_v2_marker_slow_distance', 0.8)

        self.marker_sample_count = int(rospy.get_param('/task1_v2_marker_sample_count', 10))
        self.black_marker_sample_count = int(rospy.get_param(
            '/task1_v2_black_marker_sample_count', 3
        ))
        self.black_min_confidence = float(rospy.get_param(
            '/task1_v2_black_min_confidence', 0.30
        ))
        self.marker_cluster_distance = rospy.get_param('/task1_v2_marker_cluster_distance', 0.25)
        self.marker_ignore_distance = rospy.get_param('/task1_v2_marker_ignore_distance', 0.5)
        self.marker_arrival_tolerance = rospy.get_param('/task1_v2_marker_arrival_tolerance', 0.15)
        self.marker_max_camera_distance = rospy.get_param(
            '/task1_v2_marker_max_camera_distance', 5.0
        )

        self.light_on_seconds = rospy.get_param('/task1_v2_light_on_seconds', 3.0)
        self.light_off_seconds = rospy.get_param('/task1_v2_light_off_seconds', 0.5)
        self.yellow_light_count = int(rospy.get_param('/task1_v2_yellow_light_count', 1))
        self.black_light_count = int(rospy.get_param('/task1_v2_black_light_count', 2))

        self.black_rotation_angle = math.radians(
            rospy.get_param('/task1_v2_black_rotation_angle_deg', 720.0)
        )
        self.rotation_stop_margin = math.radians(
            rospy.get_param('/task1_v2_rotation_stop_margin_deg', 10.0)
        )
        self.black_rotation_mz = rospy.get_param('/task1_v2_black_rotation_mz', 3000)
        direction = rospy.get_param('/task1_v2_black_rotation_direction', 1.0)
        self.black_rotation_direction = 1.0 if direction >= 0.0 else -1.0
        self.black_rotation_mz_step = abs(rospy.get_param(
            '/task1_v2_black_rotation_mz_step', 500
        ))
        self.black_rotation_slow_angle = math.radians(rospy.get_param(
            '/task1_v2_black_rotation_slow_angle_deg', 30.0
        ))
        self.black_rotation_slow_mz = abs(rospy.get_param(
            '/task1_v2_black_rotation_slow_mz', 1000
        ))
        self.rotation_feedback_deadband = math.radians(rospy.get_param(
            '/task1_v2_rotation_feedback_deadband_deg', 0.05
        ))
        self.rotation_feedback_max_delta = math.radians(rospy.get_param(
            '/task1_v2_rotation_feedback_max_delta_deg', 45.0
        ))

        self.camera_topic = rospy.get_param('/task1_v2_camera_topic', '/left/image_raw')
        self.camera_message_timeout = rospy.get_param(
            '/task1_v2_camera_message_timeout', 2.0
        )
        self.startup_hold_seconds = rospy.get_param(
            '/task1_v2_startup_hold_seconds', 10.0
        )
        self.transition_hold_seconds = rospy.get_param(
            '/task1_v2_transition_hold_seconds', 4.0
        )
        self.velocity_topic = rospy.get_param('/task1_v2_velocity_topic', '/status/vel')
        self.velocity_message_timeout = rospy.get_param(
            '/task1_v2_velocity_message_timeout', 1.0
        )
        self.stable_linear_speed = rospy.get_param(
            '/task1_v2_stable_linear_speed', 0.05
        )
        self.stable_angular_speed = math.radians(rospy.get_param(
            '/task1_v2_stable_angular_speed_deg', 3.0
        ))

        self.required_yellow_count = int(rospy.get_param(
            '/task1_v2_required_yellow_count', 2
        ))
        self.required_black_count = int(rospy.get_param(
            '/task1_v2_required_black_count', 2
        ))
        self.endpoint_stable_count = int(rospy.get_param(
            '/task1_v2_endpoint_stable_count', 8
        ))
        self.endpoint_position_tolerance = rospy.get_param(
            '/task1_v2_endpoint_position_tolerance', 0.12
        )
        self.endpoint_growth_tolerance = rospy.get_param(
            '/task1_v2_endpoint_growth_tolerance', 0.05
        )
        self.endpoint_stall_seconds = rospy.get_param(
            '/task1_v2_endpoint_stall_seconds', 3.0
        )
        self.endpoint_min_completed_length = rospy.get_param(
            '/task1_v2_endpoint_min_completed_length', 1.0
        )
        self.endpoint_robot_distance = rospy.get_param(
            '/task1_v2_endpoint_robot_distance', 0.8
        )

        self.trajectory_topic = rospy.get_param(
            '/task1_v2_trajectory_topic', '/task1/trajectory'
        )
        self.trajectory_max_points = int(rospy.get_param(
            '/task1_v2_trajectory_max_points', 2000
        ))
        self.trajectory_sample_distance = rospy.get_param(
            '/task1_v2_trajectory_sample_distance', 0.03
        )
        self.trajectory_publish_period = rospy.get_param(
            '/task1_v2_trajectory_publish_period', 0.5
        )
        self.trajectory_pub = rospy.Publisher(
            self.trajectory_topic, String, queue_size=2
        )
        self.trajectory_web_enabled = bool(rospy.get_param(
            '/task1_v2_trajectory_web_enabled', True
        ))
        self.trajectory_web_host = rospy.get_param(
            '/task1_v2_trajectory_web_host', '192.168.1.117'
        )
        self.trajectory_web_port = int(rospy.get_param(
            '/task1_v2_trajectory_web_port', 8082
        ))
        self.trajectory_web = None
        if self.trajectory_web_enabled:
            try:
                self.trajectory_web = Task1TrajectoryWebServer(
                    self.trajectory_web_host,
                    self.trajectory_web_port,
                )
                rospy.loginfo(
                    '%s: trajectory web available at http://%s:%d',
                    NODE_NAME,
                    self.trajectory_web_host,
                    self.trajectory_web_port,
                )
            except OSError as error:
                rospy.logwarn('%s: cannot start trajectory web: %s', NODE_NAME, error)

        self.yellow_classes = class_names('/task1_v2_yellow_classes', ['triangle', 'circle'])
        self.black_classes = class_names('/task1_v2_black_classes', ['rectangle'])
        self.line_classes = class_names('/task1_v2_line_classes', ['line'])
        self.max_line_direction_change = math.radians(rospy.get_param(
            '/task1_v2_max_line_direction_change_deg', 75.0
        ))

        self.initial_pose = None
        self.initial_yaw = None
        self.search_base_yaw = self.initial_search_yaw
        self.search_anchor_pose = None
        self.search_scan_index = 0
        self.search_scan_stable_since = None
        self.search_advance_target = None

        self.line_start_pose = None
        self.line_end_pose = None
        self.last_line_yaw = None
        self.last_line_time = None
        self.line_axis_origin = None
        self.line_axis_yaw = None
        self.line_candidate = None
        self.line_raw_points = []
        self.line_curve_points = []
        self.line_curve_s = []
        self.current_path_s = 0.0
        self.completed_path_length = 0.0
        self.completed_segment_offset = 0.0
        self.last_manual_tx = 0
        self.last_manual_ty = 0
        self.max_known_curve_length = 0.0
        self.last_curve_growth_time = rospy.Time.now()
        self.far_endpoint_samples = deque(maxlen=max(2, self.endpoint_stable_count))
        self.line_end_candidate = None
        self.verifying_line_end = False

        self.last_camera_time = None
        self.latest_velocity = None
        self.latest_velocity_time = None
        self.pose_speed_sample = None
        self.settle_target = None
        self.settle_next_step = None
        self.settle_reason = ''
        self.settle_hold_seconds = self.transition_hold_seconds
        self.settle_stable_since = None
        self.settle_completion = None

        self.actual_trajectory = []
        self.last_trajectory_publish_time = rospy.Time(0)

        self.marker_clusters = []
        self.pending_defects = []
        self.active_defect = None
        self.handled_markers = []
        self.handled_counts = {'yellow': 0, 'black': 0}

        self.light_action_state = None
        self.rotation_feedback_state = None

        rospy.Subscriber(self.target_topic, TargetDetection, self.defect_callback)
        rospy.Subscriber(self.line_topic, TargetDetection3, self.line_callback)
        rospy.Subscriber(self.camera_topic, rospy.AnyMsg, self.camera_callback, queue_size=1)
        rospy.Subscriber(
            self.velocity_topic, TwistStamped, self.velocity_callback, queue_size=5
        )
        rospy.loginfo(
            '%s: initialized, reference_height=%.2f, target_z=%.2f, '
            'initial_heading=%.1fdeg, manual_force=(fast=%s, slow=%s), '
            'line_curve(max_raw=%d, sample=%d, degree=%d)',
            NODE_NAME,
            self.reference_height,
            self.reference_z,
            math.degrees(self.initial_search_yaw),
            self.manual_forward_force,
            self.manual_slow_forward_force,
            self.line_curve_max_points,
            self.line_curve_sample_count,
            self.line_curve_degree,
        )

    ############################################### 基础工具层 #######################################
    def step_name(self, step):
        """返回状态机步骤名称，便于日志定位。"""
        return self.STEP_NAMES.get(step, str(step))

    def set_step(self, step):
        """切换状态机步骤，并记录进入该步骤的时间。"""
        old_step = self.step
        elapsed = self.step_elapsed()
        self.step = step
        self.step_started = rospy.Time.now()
        if old_step != step:
            rospy.loginfo(
                '%s: step %s -> %s, previous_step_elapsed=%.1fs',
                NODE_NAME,
                self.step_name(old_step),
                self.step_name(step),
                elapsed,
            )

    def step_elapsed(self):
        """返回当前步骤已持续时间，单位为秒。"""
        return (rospy.Time.now() - self.step_started).to_sec()

    def camera_callback(self, _message):
        """记录前视相机最近一次出图时间；消息内容由感知节点处理。"""
        self.last_camera_time = rospy.Time.now()

    def velocity_callback(self, message):
        """缓存驱动发布的速度；没有该话题时自动使用 TF 差分速度。"""
        self.latest_velocity = copy.deepcopy(message.twist)
        self.latest_velocity_time = rospy.Time.now()

    def camera_ready(self):
        """判断相机是否仍在持续发布，避免只收到一次旧消息便启动。"""
        if self.last_camera_time is None:
            return False
        return (
            rospy.Time.now() - self.last_camera_time
        ).to_sec() <= self.camera_message_timeout

    @staticmethod
    def approach_zero(value, step):
        """按固定步长将手控量收敛到零，用于切换动力定位前制动。"""
        if value > 0:
            return max(0, value - step)
        if value < 0:
            return min(0, value + step)
        return 0

    def motion_speed(self, current):
        """返回水平线速度、航向角速度及数据源。"""
        now = rospy.Time.now()
        if (
            self.latest_velocity is not None
            and self.latest_velocity_time is not None
            and (now - self.latest_velocity_time).to_sec()
            <= self.velocity_message_timeout
        ):
            linear = math.hypot(
                self.latest_velocity.linear.x,
                self.latest_velocity.linear.y,
            )
            angular = abs(self.latest_velocity.angular.z)
            return linear, angular, 'velocity_topic'

        yaw = yaw_from_quaternion(current.pose.orientation)
        sample = {
            'time': now,
            'x': current.pose.position.x,
            'y': current.pose.position.y,
            'yaw': yaw,
        }
        previous = self.pose_speed_sample
        self.pose_speed_sample = sample
        if previous is None:
            return None, None, 'tf_warming'

        elapsed = (now - previous['time']).to_sec()
        if elapsed <= 0.05:
            return None, None, 'tf_warming'
        linear = math.hypot(
            sample['x'] - previous['x'],
            sample['y'] - previous['y'],
        ) / elapsed
        angular = abs(wrap_angle(sample['yaw'] - previous['yaw'])) / elapsed
        return linear, angular, 'tf_difference'

    def motion_is_stable(self, current):
        """结合速度话题或 TF 差分判断机器人是否基本静止。"""
        linear, angular, source = self.motion_speed(current)
        stable = (
            linear is not None
            and angular is not None
            and linear <= self.stable_linear_speed
            and angular <= self.stable_angular_speed
        )
        rospy.loginfo_throttle(
            1.0,
            '%s: settle velocity source=%s linear=%s m/s angular=%s deg/s stable=%s',
            NODE_NAME,
            source,
            '{:.3f}'.format(linear) if linear is not None else 'warming',
            '{:.2f}'.format(math.degrees(angular)) if angular is not None else 'warming',
            stable,
        )
        return stable

    def begin_settle(self, target, next_step, reason, completion=None, hold_seconds=None):
        """进入动力定位稳定阶段；所有手控到定点的切换均从这里完成。"""
        self.settle_target = copy.deepcopy(target)
        self.settle_next_step = next_step
        self.settle_reason = reason
        self.settle_completion = completion
        self.settle_hold_seconds = (
            self.transition_hold_seconds if hold_seconds is None else hold_seconds
        )
        self.settle_stable_since = None
        self.pose_speed_sample = None
        rospy.loginfo(
            '%s: begin settle reason=%s target=(%.2f, %.2f, yaw=%.1fdeg) hold=%.1fs',
            NODE_NAME,
            reason,
            target.pose.position.x,
            target.pose.position.y,
            math.degrees(yaw_from_quaternion(target.pose.orientation)),
            self.settle_hold_seconds,
        )
        self.set_step(self.STEP_SETTLE)

    def finish_settle(self):
        """完成稳定等待，并按过渡类型进入下一阶段。"""
        completion = self.settle_completion
        next_step = self.settle_next_step
        reason = self.settle_reason
        self.settle_target = None
        self.settle_next_step = None
        self.settle_completion = None
        self.settle_stable_since = None
        rospy.loginfo('%s: settle completed reason=%s', NODE_NAME, reason)

        if completion == 'reset_search':
            self.reset_search_cycle()
        elif completion == 'complete_defect':
            self.complete_active_defect()
            return
        elif completion == 'evaluate_line_loss':
            if self.line_curve_ready() and self.curve_blind_follow_allowed():
                self.set_step(self.STEP_FOLLOW_LINE)
                return
            if self.mission_end_evidence_ready():
                self.verifying_line_end = True
                self.reset_search_cycle()
                rospy.loginfo(
                    '%s: finish evidence found; scan +/-30deg once before finishing',
                    NODE_NAME,
                )
                self.set_step(self.STEP_SEARCH_LINE)
                return
            self.prepare_line_reacquire()
            self.reset_search_cycle()
            self.set_step(self.STEP_SEARCH_LINE)
            return
        elif completion == 'finish_if_endpoint':
            if self.line_curve_ready() and self.curve_blind_follow_allowed():
                self.verifying_line_end = False
                self.set_step(self.STEP_FOLLOW_LINE)
                return
            if self.mission_end_evidence_ready():
                self.set_step(self.STEP_FINISH)
                return
            self.verifying_line_end = False
            self.prepare_line_reacquire()
            self.reset_search_cycle()
            self.set_step(self.STEP_SEARCH_LINE)
            return
        self.set_step(next_step)

    def run_settle(self):
        """先平滑卸掉手控力，再定点到目标并连续稳定指定时长。"""
        if self.settle_target is None:
            rospy.logerr('%s: settle target is empty', NODE_NAME)
            self.set_step(self.STEP_SEARCH_LINE)
            return

        if self.last_manual_tx != 0 or self.last_manual_ty != 0:
            current = self.get_current_pose()
            if current is None:
                return
            self.last_manual_tx = self.approach_zero(
                self.last_manual_tx, self.manual_brake_step
            )
            self.last_manual_ty = self.approach_zero(
                self.last_manual_ty, self.manual_brake_step
            )
            target = self.make_level_pose(
                current.pose.position.x,
                current.pose.position.y,
                yaw_from_quaternion(self.settle_target.pose.orientation),
            )
            self.publish_pose_cmd(
                MODE_DEPTH_HDG,
                target,
                tx=self.last_manual_tx,
                ty=self.last_manual_ty,
            )
            self.settle_stable_since = None
            rospy.loginfo_throttle(
                1.0,
                '%s: braking before DPROV reason=%s force=(%d,%d)',
                NODE_NAME,
                self.settle_reason,
                self.last_manual_tx,
                self.last_manual_ty,
            )
            return

        if not self.move_to_pose_level(self.settle_target):
            self.settle_stable_since = None
            return

        current = self.get_current_pose()
        if current is None or not self.motion_is_stable(current):
            self.settle_stable_since = None
            return

        if self.settle_stable_since is None:
            self.settle_stable_since = rospy.Time.now()
        stable_seconds = (rospy.Time.now() - self.settle_stable_since).to_sec()
        rospy.loginfo_throttle(
            1.0,
            '%s: point hold reason=%s stable=%.1f/%.1fs',
            NODE_NAME,
            self.settle_reason,
            stable_seconds,
            self.settle_hold_seconds,
        )
        if stable_seconds >= self.settle_hold_seconds:
            self.finish_settle()

    def get_current_pose(self):
        """从 TF 树读取 map -> base_link 当前位姿。"""
        try:
            self.tf_listener.waitForTransform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(2, '%s: cannot read AUV pose: %s', NODE_NAME, error)
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    @staticmethod
    def position_distance(first, second):
        """计算两个 Point 之间的三维欧氏距离。"""
        dx = first.x - second.x
        dy = first.y - second.y
        dz = first.z - second.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def publish_device(self, red=0, green=0, servo=None, light1=0, light2=0):
        """发布执行器控制消息到 /cmd/actuator。"""
        light_message = ActuatorControl()
        light_message.mode = 1
        light_message.light1 = int(light1)
        light_message.light2 = int(light2)
        self.device_pub.publish(light_message)

        actuator_message = ActuatorControl()
        actuator_message.mode = 2
        actuator_message.heading_servo = self.default_heading_servo
        actuator_message.clamp_servo = (
            self.default_clamp_servo if servo is None else int(servo)
        )
        actuator_message.drive_cmd = self.default_drive_cmd
        actuator_message.drive_speed = self.default_drive_speed
        actuator_message.red_light = int(red)
        actuator_message.yellow_light = 0
        actuator_message.green_light = int(green)
        self.device_pub.publish(actuator_message)
        rospy.loginfo_throttle(
            1.0,
            '%s: actuator cmd red=%d green=%d light1=%d light2=%d '
            'heading_servo=%d clamp_servo=%d drive=(cmd=%d, speed=%d)',
            NODE_NAME,
            actuator_message.red_light,
            actuator_message.green_light,
            light_message.light1,
            light_message.light2,
            actuator_message.heading_servo,
            actuator_message.clamp_servo,
            actuator_message.drive_cmd,
            actuator_message.drive_speed,
        )

    @staticmethod
    def force_value(value):
        """将手控力/力矩限制到 int16 安全范围。"""
        return int(round(clamp(value, -10000, 10000)))

    def publish_pose_cmd(self, mode, target, tx=0, ty=0, tz=0, mx=0, my=0, mz=0):
        """按 PoseNEDcmd 格式发布运动控制指令。"""
        command = PoseNEDcmd()
        command.mode = int(mode)
        command.target = copy.deepcopy(target)
        command.target.header.frame_id = 'map'
        command.target.header.stamp = rospy.Time.now()
        command.force.TX = self.force_value(tx)
        command.force.TY = self.force_value(ty)
        command.force.TZ = self.force_value(tz)
        command.force.MX = self.force_value(mx)
        command.force.MY = self.force_value(my)
        command.force.MZ = self.force_value(mz)
        self.pose_cmd_pub.publish(command)
        rospy.loginfo_throttle(
            1.0,
            '%s: pose cmd mode=%d(%s) target=(%.2f, %.2f, %.2f, yaw=%.1fdeg) '
            'force=(TX=%d, TY=%d, TZ=%d, MX=%d, MY=%d, MZ=%d)',
            NODE_NAME,
            command.mode,
            MODE_NAMES.get(command.mode, 'unknown'),
            command.target.pose.position.x,
            command.target.pose.position.y,
            command.target.pose.position.z,
            math.degrees(yaw_from_quaternion(command.target.pose.orientation)),
            command.force.TX,
            command.force.TY,
            command.force.TZ,
            command.force.MX,
            command.force.MY,
            command.force.MZ,
        )

    def publish_stop_cmd(self):
        """发布零力手控指令，停止巡线阶段的开环推进。"""
        current = self.get_current_pose()
        if current is None:
            yaw = self.initial_yaw if self.initial_yaw is not None else 0.0
            target = self.make_level_pose(0.0, 0.0, yaw)
        else:
            yaw = yaw_from_quaternion(current.pose.orientation)
            target = self.make_level_pose(
                current.pose.position.x,
                current.pose.position.y,
                yaw,
            )
        self.publish_pose_cmd(MODE_DEPTH_HDG, target)
        self.last_manual_tx = 0
        self.last_manual_ty = 0

    def finish_task(self):
        """关闭执行器并发布任务完成消息。"""
        self.publish_stop_cmd()
        self.publish_device()
        self.finished_pub.publish('{} finished'.format(NODE_NAME))
        rospy.loginfo('%s: mission finished', NODE_NAME)
        rospy.signal_shutdown('mission finished')

    def wait_for_current_pose(self, timeout=5.0):
        """等待并返回当前 map -> base_link 位姿；超时返回 None。"""
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            current = self.get_current_pose()
            if current is not None:
                return current
            rospy.sleep(0.1)
        return None

    def initialize_mission_frame(self):
        """记录任务启动位姿和参考运行高度。"""
        if self.initial_pose is not None:
            return True

        current = self.wait_for_current_pose()
        if current is None:
            rospy.logwarn_throttle(2, '%s: waiting for current pose', NODE_NAME)
            return False

        self.initial_pose = copy.deepcopy(current)
        self.initial_yaw = yaw_from_quaternion(current.pose.orientation)

        rospy.loginfo(
            '%s: mission origin pose recorded at x=%.2f, y=%.2f, z=%.2f, '
            'current_yaw=%.1fdeg, search_yaw=%.1fdeg; reference_height=%.2f, target_z=%.2f',
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(self.initial_yaw),
            math.degrees(self.initial_search_yaw),
            self.reference_height,
            self.reference_z,
        )
        return True

    def make_pose(self, x, y, z, yaw):
        """生成 map 坐标系下的目标位姿。"""
        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rospy.Time.now()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        target.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0, self.pitch_offset, yaw
        ))
        return target

    def make_level_pose(self, x, y, yaw):
        """生成固定参考运行高度的目标位姿。"""
        return self.make_pose(x, y, self.reference_z, yaw)

    def publish_level_target(self, x, y, yaw):
        """以动力定位 ROV 模式发布固定参考运行高度目标。"""
        self.publish_pose_cmd(MODE_DPROV, self.make_level_pose(x, y, yaw))

    def move_to_pose_level(self, target, position_tolerance=None, yaw_tolerance=None):
        """先控制航向，再控制 XY 位置，z 固定为参考运行高度。"""
        current = self.get_current_pose()
        if current is None:
            return False

        if position_tolerance is None:
            position_tolerance = self.position_tolerance
        if yaw_tolerance is None:
            yaw_tolerance = self.yaw_tolerance

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        dx = target.pose.position.x - current.pose.position.x
        dy = target.pose.position.y - current.pose.position.y
        horizontal_distance = math.hypot(dx, dy)
        final_yaw = yaw_from_quaternion(target.pose.orientation)

        if horizontal_distance > position_tolerance:
            move_yaw = math.atan2(dy, dx)
            yaw_error = wrap_angle(move_yaw - current_yaw)
            if abs(yaw_error) > yaw_tolerance:
                commanded_yaw = current_yaw + clamp(
                    yaw_error, -self.max_yaw_step, self.max_yaw_step
                )
                rospy.loginfo_throttle(
                    1.0,
                    '%s: DPROV align to move target distance=%.2f yaw_error=%.1fdeg',
                    NODE_NAME,
                    horizontal_distance,
                    math.degrees(yaw_error),
                )
                self.publish_level_target(
                    current.pose.position.x,
                    current.pose.position.y,
                    commanded_yaw,
                )
                return False

            step = min(self.max_xy_step, horizontal_distance)
            scale = step / horizontal_distance
            rospy.loginfo_throttle(
                1.0,
                '%s: DPROV move to target distance=%.2f step=%.2f target=(%.2f, %.2f)',
                NODE_NAME,
                horizontal_distance,
                step,
                target.pose.position.x,
                target.pose.position.y,
            )
            self.publish_level_target(
                current.pose.position.x + dx * scale,
                current.pose.position.y + dy * scale,
                move_yaw,
            )
            return False

        yaw_error = wrap_angle(final_yaw - current_yaw)
        if abs(yaw_error) > yaw_tolerance:
            commanded_yaw = current_yaw + clamp(yaw_error, -self.max_yaw_step, self.max_yaw_step)
            rospy.loginfo_throttle(
                1.0,
                '%s: DPROV final yaw align yaw_error=%.1fdeg target=(%.2f, %.2f)',
                NODE_NAME,
                math.degrees(yaw_error),
                target.pose.position.x,
                target.pose.position.y,
            )
            self.publish_level_target(
                target.pose.position.x,
                target.pose.position.y,
                commanded_yaw,
            )
            return False

        self.publish_level_target(
            target.pose.position.x,
            target.pose.position.y,
            final_yaw,
        )
        return True

    def hold_active_pose(self):
        """保持当前动作位姿，避免亮灯或旋转时目标点丢失。"""
        if self.active_defect is not None:
            self.active_defect['action'].header.stamp = rospy.Time.now()
            self.publish_pose_cmd(MODE_DPROV, self.active_defect['action'])
            return

        current = self.get_current_pose()
        if current is None:
            return
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        self.publish_level_target(current.pose.position.x, current.pose.position.y, current_yaw)

    ############################################### 管线处理层 #######################################
    def transform_pose_to_map(self, pose):
        """将任意检测位姿转换到 map 坐标系。"""
        try:
            self.tf_listener.waitForTransform(
                'map',
                pose.header.frame_id,
                pose.header.stamp,
                rospy.Duration(1.0),
            )
            return self.tf_listener.transformPose('map', pose)
        except tf.Exception as error:
            rospy.logwarn_throttle(2, '%s: transform failed: %s', NODE_NAME, error)
            return None

    def downsample_line_points(self):
        """按已确认的管线顺序均匀降采样，不改变前后拓扑关系。"""
        if len(self.line_raw_points) <= self.line_curve_max_points:
            return
        keep_count = max(2, self.line_curve_max_points)
        indexes = [
            int(round(index * (len(self.line_raw_points) - 1) / float(keep_count - 1)))
            for index in range(keep_count)
        ]
        self.line_raw_points = [
            copy.deepcopy(self.line_raw_points[index]) for index in indexes
        ]
        rospy.loginfo(
            '%s: line raw points downsampled to %d points',
            NODE_NAME,
            len(self.line_raw_points),
        )

    def smooth_existing_line_point(self, point):
        """将局部观测与已有路径的最近点平滑融合。"""
        if not self.line_raw_points:
            return False
        new_point = Point(point.x, point.y, self.reference_z)
        nearest = min(
            self.line_raw_points,
            key=lambda old_point: xy_distance(old_point, new_point),
        )
        if xy_distance(nearest, new_point) > self.line_point_merge_distance:
            return False
        nearest.x = 0.8 * nearest.x + 0.2 * new_point.x
        nearest.y = 0.8 * nearest.y + 0.2 * new_point.y
        nearest.z = self.reference_z
        return True

    def fuse_ordered_line_segment(self, poses):
        """
        将当前局部近→中→远三点融入全局有序路径。

        三点可以是管线上的任意采样点：已观测部分只做平滑，只有与当前
        路径尾部连通的前向部分才会追加，避免把整帧当成新端点。
        """
        points = [
            Point(pose.pose.position.x, pose.pose.position.y, self.reference_z)
            for pose in poses
        ]
        if not self.line_raw_points:
            self.line_raw_points = [copy.deepcopy(point) for point in points]
            self.downsample_line_points()
            return

        old_tail = copy.deepcopy(self.line_raw_points[-1])
        tail_distances = [xy_distance(old_tail, point) for point in points]
        connection_index = min(range(len(points)), key=lambda index: tail_distances[index])
        connection_distance = tail_distances[connection_index]

        for point in points:
            self.smooth_existing_line_point(point)

        if connection_distance > self.line_association_distance:
            rospy.loginfo_throttle(
                2.0,
                '%s: associated line segment overlaps known curve but does not extend tail, '
                'tail_gap=%.2fm',
                NODE_NAME,
                connection_distance,
            )
            return

        for point in points[connection_index:]:
            if len(self.line_raw_points) >= 2:
                previous = self.line_raw_points[-2]
                tail = self.line_raw_points[-1]
                tail_yaw = math.atan2(tail.y - previous.y, tail.x - previous.x)
                forward_projection = (
                    (point.x - tail.x) * math.cos(tail_yaw)
                    + (point.y - tail.y) * math.sin(tail_yaw)
                )
                if forward_projection < -self.line_point_merge_distance:
                    continue
            gap = xy_distance(self.line_raw_points[-1], point)
            if gap <= self.line_point_merge_distance:
                tail = self.line_raw_points[-1]
                tail.x = 0.8 * tail.x + 0.2 * point.x
                tail.y = 0.8 * tail.y + 0.2 * point.y
                continue
            if gap > self.line_extension_max_gap:
                rospy.logwarn_throttle(
                    2.0,
                    '%s: stop line extension at discontinuous gap %.2fm > %.2fm',
                    NODE_NAME,
                    gap,
                    self.line_extension_max_gap,
                )
                break
            self.line_raw_points.append(copy.deepcopy(point))
        self.downsample_line_points()

    @staticmethod
    def cumulative_distance(points):
        """计算点列的累计水平距离。"""
        distances = [0.0]
        for index in range(1, len(points)):
            distances.append(distances[-1] + xy_distance(points[index], points[index - 1]))
        return distances

    def fit_line_curve(self):
        """将有限红线点集拟合为平滑曲线，并采样为固定数量控制点。"""
        if len(self.line_raw_points) < 2:
            return

        filtered = [self.line_raw_points[0]]
        for point in self.line_raw_points[1:]:
            if xy_distance(point, filtered[-1]) > 1e-3:
                filtered.append(point)

        if len(filtered) < 2:
            return

        raw_s = self.cumulative_distance(filtered)
        total_length = raw_s[-1]
        if total_length < self.line_curve_min_length:
            self.line_curve_points = [copy.deepcopy(point) for point in filtered]
            self.line_curve_s = raw_s
            return

        degree = min(max(1, self.line_curve_degree), len(filtered) - 1)
        xs = [point.x for point in filtered]
        ys = [point.y for point in filtered]
        sample_count = max(2, self.line_curve_sample_count)

        try:
            x_curve = np.poly1d(np.polyfit(raw_s, xs, degree))
            y_curve = np.poly1d(np.polyfit(raw_s, ys, degree))
            sample_s = np.linspace(0.0, total_length, sample_count)
            fitted = []
            for value in sample_s:
                fitted.append(Point(float(x_curve(value)), float(y_curve(value)), self.reference_z))
            self.line_curve_points = fitted
            self.line_curve_s = self.cumulative_distance(self.line_curve_points)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            rospy.logwarn_throttle(2, '%s: line curve fit failed: %s', NODE_NAME, error)
            self.line_curve_points = [copy.deepcopy(point) for point in filtered]
            self.line_curve_s = raw_s

    def update_line_curve(self, poses):
        """用一帧已关联的局部近/中/远三点更新全局曲线。"""
        first, second, third = poses
        if self.line_axis_origin is None:
            self.line_axis_origin = copy.deepcopy(first.pose.position)
            self.line_axis_yaw = math.atan2(
                third.pose.position.y - first.pose.position.y,
                third.pose.position.x - first.pose.position.x,
            )
            rospy.loginfo(
                '%s: line axis initialized origin=(%.2f, %.2f), yaw=%.1fdeg',
                NODE_NAME,
                self.line_axis_origin.x,
                self.line_axis_origin.y,
                math.degrees(self.line_axis_yaw),
            )

        self.fuse_ordered_line_segment((first, second, third))
        self.fit_line_curve()
        curve_length = self.line_curve_s[-1] if self.line_curve_ready() else 0.0
        if curve_length > self.max_known_curve_length + self.endpoint_growth_tolerance:
            self.max_known_curve_length = curve_length
            self.last_curve_growth_time = rospy.Time.now()
            self.line_end_candidate = None
        rospy.loginfo_throttle(
            1.0,
            '%s: line fusion raw_points=%d curve_points=%d known_curve=%.2fm '
            'line_yaw=%.1fdeg',
            NODE_NAME,
            len(self.line_raw_points),
            len(self.line_curve_points),
            curve_length,
            math.degrees(self.line_axis_yaw if self.line_axis_yaw is not None else 0.0),
        )

    def update_line_end_evidence(self, far_pose):
        """
        用局部前向观测点稳定、曲线停止增长和机器人接近尾部共同生成终点候选。

        P1 只是当前局部三点中的前向点；本函数不单独依赖 P1 判断物理终点。
        任务结束仍需后续红线超时丢失和额外扫描确认。
        """
        self.far_endpoint_samples.append(copy.deepcopy(far_pose.pose.position))
        if len(self.far_endpoint_samples) < self.far_endpoint_samples.maxlen:
            return

        center = Point(
            sum(point.x for point in self.far_endpoint_samples)
            / len(self.far_endpoint_samples),
            sum(point.y for point in self.far_endpoint_samples)
            / len(self.far_endpoint_samples),
            self.reference_z,
        )
        spread = max(xy_distance(point, center) for point in self.far_endpoint_samples)
        stalled = (
            rospy.Time.now() - self.last_curve_growth_time
        ).to_sec() >= self.endpoint_stall_seconds
        long_enough = self.completed_path_length >= self.endpoint_min_completed_length
        current = self.get_current_pose()
        robot_distance = (
            xy_distance(current.pose.position, center) if current is not None else float('inf')
        )
        robot_near_end = (
            self.near_curve_end() and robot_distance <= self.endpoint_robot_distance
        )
        if (
            spread <= self.endpoint_position_tolerance
            and stalled
            and long_enough
            and robot_near_end
        ):
            candidate = copy.deepcopy(far_pose)
            candidate.pose.position = center
            self.line_end_candidate = candidate
            rospy.loginfo_throttle(
                1.0,
                '%s: red line end candidate stable, local_far_spread=%.3fm '
                'robot_to_end=%.2fm completed=%.2fm growth_stalled=%.1fs',
                NODE_NAME,
                spread,
                robot_distance,
                self.completed_path_length,
                (rospy.Time.now() - self.last_curve_growth_time).to_sec(),
            )
        else:
            self.line_end_candidate = None

    def line_curve_ready(self):
        """判断当前是否已有可用于 LOS 的拟合曲线。"""
        return len(self.line_curve_points) >= 2 and len(self.line_curve_s) == len(
            self.line_curve_points
        )

    def project_to_line_curve(self, point):
        """将机器人当前位置投影到拟合曲线，返回路径进度和横向误差。"""
        if not self.line_curve_ready():
            return None

        best = None
        for index in range(len(self.line_curve_points) - 1):
            start = self.line_curve_points[index]
            end = self.line_curve_points[index + 1]
            vx = end.x - start.x
            vy = end.y - start.y
            segment_sq = vx * vx + vy * vy
            if segment_sq < 1e-9:
                continue

            wx = point.x - start.x
            wy = point.y - start.y
            ratio = clamp((wx * vx + wy * vy) / segment_sq, 0.0, 1.0)
            proj_x = start.x + ratio * vx
            proj_y = start.y + ratio * vy
            dx = point.x - proj_x
            dy = point.y - proj_y
            distance = math.hypot(dx, dy)
            segment_length = math.sqrt(segment_sq)
            signed_lateral = (vx * (point.y - start.y) - vy * (point.x - start.x)) / (
                segment_length
            )
            path_s = self.line_curve_s[index] + ratio * segment_length
            segment_yaw = math.atan2(vy, vx)

            if best is None or distance < best['distance']:
                best = {
                    'distance': distance,
                    'lateral': signed_lateral,
                    'path_s': path_s,
                    'segment_yaw': segment_yaw,
                    'projection': Point(proj_x, proj_y, self.reference_z),
                }
        return best

    def point_at_curve_s(self, target_s):
        """按路径进度从拟合曲线上插值得到前视目标点。"""
        if not self.line_curve_ready():
            return None

        target_s = clamp(target_s, 0.0, self.line_curve_s[-1])
        for index in range(len(self.line_curve_s) - 1):
            start_s = self.line_curve_s[index]
            end_s = self.line_curve_s[index + 1]
            if target_s > end_s:
                continue

            start = self.line_curve_points[index]
            end = self.line_curve_points[index + 1]
            if end_s <= start_s:
                return copy.deepcopy(start)
            ratio = (target_s - start_s) / (end_s - start_s)
            return Point(
                start.x + ratio * (end.x - start.x),
                start.y + ratio * (end.y - start.y),
                self.reference_z,
            )
        return copy.deepcopy(self.line_curve_points[-1])

    def near_curve_end(self):
        """判断机器人是否已经接近当前已知曲线末端。"""
        if not self.line_curve_ready():
            return False
        return self.current_path_s >= self.line_curve_s[-1] - self.line_end_margin

    def curve_blind_follow_allowed(self):
        """红线短暂丢失时，允许沿已知曲线低速继续一小段。"""
        if self.last_line_time is None:
            return False
        return (
            rospy.Time.now() - self.last_line_time
        ).to_sec() <= self.curve_blind_follow_timeout

    def limit_manual_force(self, desired, previous):
        """限制手控力单周期变化量，降低惯性过冲。"""
        return int(round(clamp(
            desired,
            previous - self.manual_force_step,
            previous + self.manual_force_step,
        )))

    def limit_lateral_force(self, desired, previous):
        """横移非零时使用有效力区间；换向前先卸力，避免左右突变。"""
        if desired == 0:
            return self.approach_zero(previous, self.manual_brake_step)
        desired_sign = 1 if desired > 0 else -1
        previous_sign = 1 if previous > 0 else -1 if previous < 0 else 0
        if previous_sign not in (0, desired_sign):
            return self.approach_zero(previous, self.manual_brake_step)
        if previous_sign == 0:
            return desired_sign * int(self.manual_min_lateral_force)
        return self.limit_manual_force(desired, previous)

    def nearest_pending_marker_distance(self):
        """返回最近待处理图形到机器人当前位置的水平距离。"""
        if not self.pending_defects:
            return None
        current = self.get_current_pose()
        if current is None:
            return None
        distances = [
            xy_distance(current.pose.position, defect['marker'].pose.position)
            for defect in self.pending_defects
        ]
        return min(distances) if distances else None

    def ready_defect_index(self):
        """判断是否已经到达某个稳定图形附近，可以切换到动作阶段。"""
        if not self.pending_defects:
            return None
        current = self.get_current_pose()
        if current is None:
            return None

        best_index = None
        best_distance = None
        for index, defect in enumerate(self.pending_defects):
            distance = xy_distance(current.pose.position, defect['marker'].pose.position)
            if distance <= self.marker_arrival_tolerance:
                if best_distance is None or distance < best_distance:
                    best_index = index
                    best_distance = distance
        return best_index

    def follow_line_curve_manual(self):
        """使用拟合曲线 LOS 计算定深定向手控指令。"""
        current = self.get_current_pose()
        if current is None or not self.line_curve_ready():
            self.publish_stop_cmd()
            return False

        projection = self.project_to_line_curve(current.pose.position)
        if projection is None:
            self.publish_stop_cmd()
            return False

        self.current_path_s = projection['path_s']
        self.completed_path_length = max(
            self.completed_path_length,
            self.completed_segment_offset + self.current_path_s,
        )
        los_target = self.point_at_curve_s(self.current_path_s + self.los_lookahead_distance)
        if los_target is None:
            self.publish_stop_cmd()
            return False

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        desired_yaw = math.atan2(
            los_target.y - current.pose.position.y,
            los_target.x - current.pose.position.x,
        )
        yaw_error = wrap_angle(desired_yaw - current_yaw)
        lateral_error = projection['lateral']

        forward_force = self.manual_forward_force
        marker_distance = self.nearest_pending_marker_distance()
        line_age = (
            (rospy.Time.now() - self.last_line_time).to_sec()
            if self.last_line_time is not None else -1.0
        )
        if (
            abs(yaw_error) > self.manual_slow_yaw_error
            or abs(lateral_error) > self.manual_slow_lateral_error
            or (marker_distance is not None and marker_distance < self.marker_slow_distance)
            or not self.line_is_recent()
        ):
            forward_force = self.manual_slow_forward_force

        desired_tx = self.manual_tx_sign * forward_force
        raw_lateral_force = -self.manual_lateral_gain * lateral_error
        if abs(lateral_error) <= self.manual_lateral_deadband:
            desired_ty = 0
        else:
            desired_ty = math.copysign(
                clamp(
                    abs(raw_lateral_force),
                    self.manual_min_lateral_force,
                    self.manual_max_lateral_force,
                ),
                raw_lateral_force,
            ) * self.manual_ty_sign
        tx = self.limit_manual_force(desired_tx, self.last_manual_tx)
        ty = self.limit_lateral_force(desired_ty, self.last_manual_ty)
        self.last_manual_tx = tx
        self.last_manual_ty = ty

        target = self.make_level_pose(
            current.pose.position.x,
            current.pose.position.y,
            desired_yaw,
        )
        self.publish_pose_cmd(MODE_DEPTH_HDG, target, tx=tx, ty=ty)

        rospy.loginfo_throttle(
            1.0,
            '%s: LOS manual mode=3 completed=%.2fm known_curve=%.2fm lateral=%.2f '
            'yaw_error=%.1fdeg line_age=%.1fs raw_points=%d curve_points=%d '
            'pending=%d nearest_marker=%.2f force=(%d,%d)',
            NODE_NAME,
            self.completed_path_length,
            self.line_curve_s[-1],
            lateral_error,
            math.degrees(yaw_error),
            line_age,
            len(self.line_raw_points),
            len(self.line_curve_points),
            len(self.pending_defects),
            marker_distance if marker_distance is not None else -1.0,
            tx,
            ty,
        )
        return True

    @staticmethod
    def point_to_chord_distance(point, start, end):
        """计算点到线段弦线的水平距离。"""
        vx = end.x - start.x
        vy = end.y - start.y
        length_sq = vx * vx + vy * vy
        if length_sq < 1e-9:
            return float('inf')
        ratio = clamp(
            ((point.x - start.x) * vx + (point.y - start.y) * vy) / length_sq,
            0.0,
            1.0,
        )
        projection = Point(start.x + ratio * vx, start.y + ratio * vy, point.z)
        return xy_distance(point, projection)

    def validate_local_line_triplet(self, near, middle, far):
        """验证任意局部三点的顺序、间距、共线性和局部转角。"""
        near_point = near.pose.position
        middle_point = middle.pose.position
        far_point = far.pose.position
        near_middle = xy_distance(near_point, middle_point)
        middle_far = xy_distance(middle_point, far_point)
        if (
            near_middle < self.line_min_point_spacing
            or middle_far < self.line_min_point_spacing
        ):
            return False, None, 'point_spacing_too_small'
        if (
            near_middle > self.line_max_point_spacing
            or middle_far > self.line_max_point_spacing
        ):
            return False, None, 'point_spacing_too_large'

        yaw_near_middle = math.atan2(
            middle_point.y - near_point.y,
            middle_point.x - near_point.x,
        )
        yaw_middle_far = math.atan2(
            far_point.y - middle_point.y,
            far_point.x - middle_point.x,
        )
        if abs(wrap_angle(yaw_middle_far - yaw_near_middle)) > self.line_local_max_bend:
            return False, None, 'local_bend_too_large'

        middle_offset = self.point_to_chord_distance(
            middle_point, near_point, far_point
        )
        if middle_offset > self.line_middle_offset_tolerance:
            return False, None, 'middle_point_off_chord'

        current = self.get_current_pose()
        if current is not None:
            robot = current.pose.position
            distances = [
                xy_distance(robot, near_point),
                xy_distance(robot, middle_point),
                xy_distance(robot, far_point),
            ]
            if (
                distances[0] > distances[1] + self.line_point_order_tolerance
                or distances[1] > distances[2] + self.line_point_order_tolerance
            ):
                return False, None, 'near_middle_far_order_invalid'

        detected_yaw = math.atan2(
            far_point.y - near_point.y,
            far_point.x - near_point.x,
        )
        return True, detected_yaw, 'valid'

    def line_segment_associated(self, poses, detected_yaw):
        """根据与已知曲线的距离、切向和当前进度判断是否为同一管线。"""
        if not self.line_curve_ready():
            return True, 'first_line_segment'

        projections = []
        minimum_progress = max(0.0, self.current_path_s - self.line_association_backtrack)
        for pose in poses:
            projection = self.project_to_line_curve(pose.pose.position)
            if projection is None or projection['path_s'] < minimum_progress:
                continue
            projections.append(projection)
        if not projections:
            return False, 'segment_behind_current_progress'

        best = min(projections, key=lambda value: value['distance'])
        if best['distance'] > self.line_association_distance:
            return False, 'segment_too_far_from_active_curve'
        if abs(wrap_angle(detected_yaw - best['segment_yaw'])) > self.line_association_angle:
            return False, 'segment_direction_mismatch'
        return True, 'associated_with_active_curve'

    def reset_line_candidate(self, reason):
        """清除未确认的管线候选，防止非连续误检累加。"""
        if self.line_candidate is not None:
            rospy.loginfo_throttle(
                1.0,
                '%s: reset line candidate reason=%s confirmed_frames=%d',
                NODE_NAME,
                reason,
                self.line_candidate['count'],
            )
        self.line_candidate = None

    def confirm_line_candidate(self, poses, detected_yaw):
        """要求空间位置和方向一致的局部管线连续多帧出现。"""
        points = [pose.pose.position for pose in poses]
        center = Point(
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
            self.reference_z,
        )
        compatible = (
            self.line_candidate is not None
            and xy_distance(center, self.line_candidate['center'])
            <= self.line_candidate_center_distance
            and abs(wrap_angle(detected_yaw - self.line_candidate['yaw']))
            <= self.line_candidate_yaw_tolerance
        )
        if compatible:
            self.line_candidate['count'] += 1
            self.line_candidate['center'] = center
            self.line_candidate['yaw'] = detected_yaw
        else:
            self.line_candidate = {
                'count': 1,
                'center': center,
                'yaw': detected_yaw,
            }

        rospy.loginfo_throttle(
            1.0,
            '%s: line candidate frames=%d/%d center=(%.2f,%.2f) yaw=%.1fdeg',
            NODE_NAME,
            self.line_candidate['count'],
            self.line_candidate_confirm_frames,
            center.x,
            center.y,
            math.degrees(detected_yaw),
        )
        return self.line_candidate['count'] >= self.line_candidate_confirm_frames

    def line_callback(self, message):
        """接收红色长线识别结果，并更新巡线目标。"""
        if message.class_name and message.class_name not in self.line_classes:
            self.reset_line_candidate('unexpected_line_class')
            return

        # Pose1/Pose2/Pose3 是局部管线上的远/中/近点，不代表整条管线端点。
        far = self.transform_pose_to_map(message.pose1)
        middle = self.transform_pose_to_map(message.pose2)
        near = self.transform_pose_to_map(message.pose3)
        if far is None or middle is None or near is None:
            self.reset_line_candidate('line_tf_failed')
            return

        geometry_valid, detected_yaw, reason = self.validate_local_line_triplet(
            near, middle, far
        )
        if not geometry_valid:
            rospy.logwarn_throttle(
                1.0,
                '%s: reject local line triplet reason=%s',
                NODE_NAME,
                reason,
            )
            self.reset_line_candidate(reason)
            return

        if (
            self.last_line_yaw is not None
            and abs(wrap_angle(detected_yaw - self.last_line_yaw))
            > self.max_line_direction_change
        ):
            rospy.logwarn_throttle(
                2.0,
                '%s: ignore reverse line candidate detected=%.1fdeg previous=%.1fdeg',
                NODE_NAME,
                math.degrees(detected_yaw),
                math.degrees(self.last_line_yaw),
            )
            self.reset_line_candidate('gross_direction_change')
            return

        associated, reason = self.line_segment_associated(
            (near, middle, far), detected_yaw
        )
        if not associated:
            rospy.logwarn_throttle(
                1.0,
                '%s: isolate unrelated line candidate reason=%s yaw=%.1fdeg',
                NODE_NAME,
                reason,
                math.degrees(detected_yaw),
            )
            self.reset_line_candidate(reason)
            return

        if not self.confirm_line_candidate((near, middle, far), detected_yaw):
            return

        if self.line_start_pose is None:
            self.line_start_pose = copy.deepcopy(near)
            rospy.loginfo(
                '%s: first confirmed line observation recorded at x=%.2f, y=%.2f, z=%.2f',
                NODE_NAME,
                near.pose.position.x,
                near.pose.position.y,
                near.pose.position.z,
            )

        self.line_end_pose = copy.deepcopy(far)
        self.last_line_time = rospy.Time.now()
        self.last_line_yaw = detected_yaw
        self.update_line_curve((near, middle, far))
        self.update_line_end_evidence(far)
        rospy.loginfo_throttle(
            1.0,
            '%s: accepted local line class=%s P1_local_far=(%.2f, %.2f) '
            'P2_local_mid=(%.2f, %.2f) P3_local_near=(%.2f, %.2f) yaw=%.1fdeg',
            NODE_NAME,
            message.class_name,
            far.pose.position.x,
            far.pose.position.y,
            middle.pose.position.x,
            middle.pose.position.y,
            near.pose.position.x,
            near.pose.position.y,
            math.degrees(self.last_line_yaw),
        )

    def line_is_recent(self):
        """判断红线识别是否仍然有效。"""
        if self.last_line_time is None:
            return False
        return (rospy.Time.now() - self.last_line_time).to_sec() <= self.line_lost_timeout

    def reset_search_cycle(self):
        """以当前点为扫描中心，重置“左右扫描→前进 0.5m”搜索周期。"""
        current = self.get_current_pose()
        if current is None:
            return False
        self.search_base_yaw = (
            self.last_line_yaw if self.last_line_yaw is not None
            else self.initial_search_yaw
        )
        self.search_anchor_pose = copy.deepcopy(current)
        self.search_scan_index = 0
        self.search_scan_stable_since = None
        self.search_advance_target = self.make_level_pose(
            current.pose.position.x
            + self.search_forward_step * math.cos(self.search_base_yaw),
            current.pose.position.y
            + self.search_forward_step * math.sin(self.search_base_yaw),
            self.search_base_yaw,
        )
        rospy.loginfo(
            '%s: reset line search anchor=(%.2f, %.2f), base_yaw=%.1fdeg',
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            math.degrees(self.search_base_yaw),
        )
        return True

    def search_line_forward(self):
        """定点左右扫描；仍未发现红线时定深定向前进 0.5 m 后重复。"""
        if self.search_anchor_pose is None and not self.reset_search_cycle():
            return

        current = self.get_current_pose()
        if current is None:
            return

        if self.search_scan_index < len(self.search_scan_offsets):
            scan_yaw = wrap_angle(
                self.search_base_yaw + self.search_scan_offsets[self.search_scan_index]
            )
            scan_target = self.make_level_pose(
                self.search_anchor_pose.pose.position.x,
                self.search_anchor_pose.pose.position.y,
                scan_yaw,
            )
            reached = self.move_to_pose_level(scan_target)
            stable = reached and self.motion_is_stable(current)
            if not stable:
                self.search_scan_stable_since = None
                return
            if self.search_scan_stable_since is None:
                self.search_scan_stable_since = rospy.Time.now()
            stable_seconds = (
                rospy.Time.now() - self.search_scan_stable_since
            ).to_sec()
            rospy.loginfo_throttle(
                1.0,
                '%s: scanning line angle=%+.1fdeg index=%d/%d stable=%.1f/%.1fs',
                NODE_NAME,
                math.degrees(self.search_scan_offsets[self.search_scan_index]),
                self.search_scan_index + 1,
                len(self.search_scan_offsets),
                stable_seconds,
                self.search_scan_hold_seconds,
            )
            if stable_seconds >= self.search_scan_hold_seconds:
                self.search_scan_index += 1
                self.search_scan_stable_since = None
                self.pose_speed_sample = None
            return

        if self.verifying_line_end:
            target = self.make_level_pose(
                self.search_anchor_pose.pose.position.x,
                self.search_anchor_pose.pose.position.y,
                self.search_base_yaw,
            )
            self.begin_settle(
                target,
                self.STEP_FINISH,
                'endpoint_scan_completed',
                completion='finish_if_endpoint',
            )
            return

        dx = current.pose.position.x - self.search_anchor_pose.pose.position.x
        dy = current.pose.position.y - self.search_anchor_pose.pose.position.y
        progress = dx * math.cos(self.search_base_yaw) + dy * math.sin(
            self.search_base_yaw
        )
        remaining = self.search_forward_step - progress
        if remaining <= self.position_tolerance:
            self.begin_settle(
                self.search_advance_target,
                self.STEP_SEARCH_LINE,
                'search_advance_completed',
                completion='reset_search',
            )
            return

        desired_force = (
            self.search_slow_forward_force
            if remaining <= self.search_deceleration_distance
            else self.search_forward_force
        ) * self.manual_tx_sign
        self.last_manual_tx = self.limit_manual_force(
            desired_force, self.last_manual_tx
        )
        self.last_manual_ty = self.approach_zero(
            self.last_manual_ty, self.manual_brake_step
        )
        target = self.make_level_pose(
            current.pose.position.x,
            current.pose.position.y,
            self.search_base_yaw,
        )
        self.publish_pose_cmd(
            MODE_DEPTH_HDG,
            target,
            tx=self.last_manual_tx,
            ty=self.last_manual_ty,
        )
        rospy.loginfo_throttle(
            1.0,
            '%s: line search advance progress=%.2f/%.2fm remaining=%.2fm TX=%d',
            NODE_NAME,
            progress,
            self.search_forward_step,
            remaining,
            self.last_manual_tx,
        )

    ############################################### 图形处理层 #######################################
    def marker_already_known(self, point, defect_type):
        """判断同类图形是否已入队、正在处理或已完成。"""
        for marker in self.handled_markers:
            if marker['type'] == defect_type and xy_distance(point, marker['point']) < (
                self.marker_ignore_distance
            ):
                return True

        for defect in self.pending_defects:
            if defect['type'] == defect_type and xy_distance(
                point, defect['marker'].pose.position
            ) < self.marker_ignore_distance:
                return True

        if self.active_defect is not None:
            if self.active_defect['type'] == defect_type and xy_distance(
                point, self.active_defect['marker'].pose.position
            ) < self.marker_ignore_distance:
                return True

        return False

    def marker_sample_requirement(self, defect_type):
        """返回不同图形的稳定识别帧数。"""
        if defect_type == 'black':
            return self.black_marker_sample_count
        return self.marker_sample_count

    def reset_unqueued_marker_clusters(self, defect_type, reason):
        """清除未入队的同类聚类，用于保证黑色方形是连续识别。"""
        removed = sum(
            1 for cluster in self.marker_clusters
            if cluster['type'] == defect_type and not cluster['queued']
        )
        if removed == 0:
            return
        self.marker_clusters = [
            cluster for cluster in self.marker_clusters
            if cluster['type'] != defect_type or cluster['queued']
        ]
        rospy.loginfo(
            '%s: reset %s marker streak reason=%s removed_clusters=%d',
            NODE_NAME,
            defect_type,
            reason,
            removed,
        )

    def defect_callback(self, message):
        """接收图形识别结果，聚类稳定后加入动作队列。"""
        if message.type and message.type != 'center':
            if message.class_name in self.black_classes:
                self.reset_unqueued_marker_clusters('black', 'non_center_detection')
            return

        if message.class_name in self.yellow_classes:
            defect_type = 'yellow'
        elif message.class_name in self.black_classes:
            defect_type = 'black'
        else:
            self.reset_unqueued_marker_clusters('black', 'non_rectangle_detection')
            return

        if defect_type != 'black':
            self.reset_unqueued_marker_clusters('black', 'rectangle_sequence_interrupted')
        elif message.conf < self.black_min_confidence:
            rospy.loginfo_throttle(
                1.0,
                '%s: reject black rectangle conf=%.2f < %.2f; restart streak',
                NODE_NAME,
                message.conf,
                self.black_min_confidence,
            )
            self.reset_unqueued_marker_clusters('black', 'confidence_below_threshold')
            return

        camera_origin = Point()
        if self.position_distance(
            message.pose.pose.position, camera_origin
        ) > self.marker_max_camera_distance:
            rospy.loginfo_throttle(
                2.0,
                '%s: ignore far marker class=%s camera_pos=(%.2f, %.2f, %.2f)',
                NODE_NAME,
                message.class_name,
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            )
            if defect_type == 'black':
                self.reset_unqueued_marker_clusters('black', 'camera_distance_invalid')
            return

        rospy.loginfo_throttle(
            1.0,
            '%s: marker detection class=%s defect=%s conf=%.2f camera_pos=(%.2f, %.2f, %.2f)',
            NODE_NAME,
            message.class_name,
            defect_type,
            message.conf,
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            message.pose.pose.position.z,
        )

        marker = self.transform_pose_to_map(message.pose)
        if marker is None:
            if defect_type == 'black':
                self.reset_unqueued_marker_clusters('black', 'tf_transform_failed')
            return
        if self.marker_already_known(marker.pose.position, defect_type):
            rospy.loginfo_throttle(
                2.0,
                '%s: ignore known %s marker at x=%.2f, y=%.2f',
                NODE_NAME,
                defect_type,
                marker.pose.position.x,
                marker.pose.position.y,
            )
            return

        self.add_marker_sample(defect_type, marker, message.conf)

    def add_marker_sample(self, defect_type, marker, confidence):
        """用多帧识别结果确定一个稳定图形位置。"""
        point = marker.pose.position
        required_samples = self.marker_sample_requirement(defect_type)
        for cluster in self.marker_clusters:
            if cluster['type'] != defect_type:
                continue
            if xy_distance(point, cluster['center']) > self.marker_cluster_distance:
                continue

            cluster['samples'].append(copy.deepcopy(marker))
            cluster['confidences'].append(confidence)
            cluster['center'] = self.cluster_center(cluster['samples'])
            rospy.loginfo_throttle(
                1.0,
                '%s: marker cluster %s samples=%d/%d center=(%.2f, %.2f)',
                NODE_NAME,
                defect_type,
                len(cluster['samples']),
                required_samples,
                cluster['center'].x,
                cluster['center'].y,
            )
            self.queue_marker_if_stable(cluster)
            return

        if defect_type == 'black':
            self.reset_unqueued_marker_clusters('black', 'rectangle_position_changed')

        cluster = {
            'type': defect_type,
            'samples': [copy.deepcopy(marker)],
            'confidences': [confidence],
            'center': copy.deepcopy(marker.pose.position),
            'queued': False,
        }
        self.marker_clusters.append(cluster)
        rospy.loginfo(
            '%s: new marker cluster %s at x=%.2f, y=%.2f, conf=%.2f',
            NODE_NAME,
            defect_type,
            point.x,
            point.y,
            confidence,
        )
        self.queue_marker_if_stable(cluster)

    def cluster_center(self, samples):
        """使用中位数计算图形聚类中心。"""
        center = Point()
        center.x = median([sample.pose.position.x for sample in samples])
        center.y = median([sample.pose.position.y for sample in samples])
        center.z = median([sample.pose.position.z for sample in samples])
        return center

    def queue_marker_if_stable(self, cluster):
        """当图形连续稳定识别达到阈值后，生成动作并入队。"""
        required_samples = self.marker_sample_requirement(cluster['type'])
        if cluster['queued'] or len(cluster['samples']) < required_samples:
            return

        if self.marker_already_known(cluster['center'], cluster['type']):
            cluster['queued'] = True
            return

        marker = copy.deepcopy(cluster['samples'][-1])
        marker.pose.position = copy.deepcopy(cluster['center'])

        self.pending_defects.append({
            'type': cluster['type'],
            'action': None,
            'marker': marker,
            'confidence': sum(cluster['confidences']) / len(cluster['confidences']),
        })
        cluster['queued'] = True
        rospy.loginfo(
            '%s: queued stable %s marker at x=%.2f, y=%.2f, z=%.2f, samples=%d',
            NODE_NAME,
            cluster['type'],
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
            len(cluster['samples']),
        )

    def action_pose_from_current(self):
        """生成图形中心上方动作位姿；触发前机器人必须已沿曲线到达附近。"""
        current = self.get_current_pose()
        if current is None or self.active_defect is None:
            return None
        marker = self.active_defect['marker'].pose.position
        action_yaw = (
            self.last_line_yaw if self.last_line_yaw is not None
            else yaw_from_quaternion(current.pose.orientation)
        )
        return self.make_level_pose(
            marker.x,
            marker.y,
            action_yaw,
        )

    def prepare_line_reacquire(self):
        """保存已完成里程并清空图形遮挡前的局部红线路径。"""
        self.completed_segment_offset = max(
            self.completed_segment_offset,
            self.completed_path_length,
        )
        self.last_line_time = None
        self.line_axis_origin = None
        self.line_axis_yaw = None
        self.line_candidate = None
        self.line_raw_points = []
        self.line_curve_points = []
        self.line_curve_s = []
        self.current_path_s = 0.0
        self.max_known_curve_length = 0.0
        self.last_curve_growth_time = rospy.Time.now()
        self.far_endpoint_samples.clear()
        self.line_end_candidate = None
        self.verifying_line_end = False
        self.search_anchor_pose = None
        self.search_scan_index = 0
        self.search_scan_stable_since = None
        self.search_advance_target = None
        rospy.loginfo(
            '%s: local line cleared for reacquisition, completed=%.2fm, '
            'continue_yaw=%.1fdeg',
            NODE_NAME,
            self.completed_path_length,
            math.degrees(
                self.last_line_yaw
                if self.last_line_yaw is not None else self.initial_search_yaw
            ),
        )

    def all_required_actions_complete(self):
        """判断比赛要求的两个黄色动作和两个黑色动作是否均已完成。"""
        return (
            self.handled_counts['yellow'] >= self.required_yellow_count
            and self.handled_counts['black'] >= self.required_black_count
        )

    def mission_end_evidence_ready(self):
        """结束必须同时满足动作数量、稳定远端点及红线超时丢失。"""
        line_lost = (
            self.last_line_time is None
            or (rospy.Time.now() - self.last_line_time).to_sec()
            >= self.line_lost_timeout
        )
        ready = (
            self.all_required_actions_complete()
            and self.line_end_candidate is not None
            and line_lost
        )
        rospy.loginfo_throttle(
            1.0,
            '%s: finish evidence yellow=%d/%d black=%d/%d '
            'stable_endpoint=%s line_lost=%s ready=%s',
            NODE_NAME,
            self.handled_counts['yellow'],
            self.required_yellow_count,
            self.handled_counts['black'],
            self.required_black_count,
            self.line_end_candidate is not None,
            line_lost,
            ready,
        )
        return ready

    def complete_active_defect(self):
        """记录图形动作完成，清空被图形遮挡前的红线并重新搜索。"""
        if self.active_defect is None:
            return
        self.handled_markers.append({
            'type': self.active_defect['type'],
            'point': copy.deepcopy(self.active_defect['marker'].pose.position),
        })
        self.handled_counts[self.active_defect['type']] += 1
        rospy.loginfo(
            '%s: completed %s marker, yellow=%d, black=%d',
            NODE_NAME,
            self.active_defect['type'],
            self.handled_counts['yellow'],
            self.handled_counts['black'],
        )
        self.active_defect = None
        self.light_action_state = None
        self.rotation_feedback_state = None
        self.prepare_line_reacquire()
        self.reset_search_cycle()
        self.set_step(self.STEP_SEARCH_LINE)

    ############################################### 动作执行层 #######################################
    def run_light_action(self):
        """按图形类型执行灯光动作；每次亮灯持续 light_on_seconds。"""
        if self.active_defect is None:
            return True

        defect_type = self.active_defect['type']
        if self.light_action_state is None:
            self.light_action_state = {
                'count': (
                    self.yellow_light_count
                    if defect_type == 'yellow' else self.black_light_count
                ),
                'red': 1 if defect_type == 'yellow' else 0,
                'green': 0 if defect_type == 'yellow' else 1,
            }
            rospy.loginfo(
                '%s: light action start type=%s count=%d light_on=%.1fs light_off=%.1fs',
                NODE_NAME,
                defect_type,
                self.light_action_state['count'],
                self.light_on_seconds,
                self.light_off_seconds,
            )

        self.hold_active_pose()

        elapsed = self.step_elapsed()
        cycle = self.light_on_seconds + self.light_off_seconds
        current_count = int(elapsed // cycle)
        if current_count >= self.light_action_state['count']:
            self.publish_device(red=0, green=0)
            rospy.loginfo(
                '%s: light action completed type=%s elapsed=%.1fs',
                NODE_NAME,
                defect_type,
                elapsed,
            )
            return True

        in_cycle = elapsed - current_count * cycle
        if in_cycle < self.light_on_seconds:
            self.publish_device(
                red=self.light_action_state['red'],
                green=self.light_action_state['green'],
            )
            rospy.loginfo_throttle(
                1.0,
                '%s: light action running type=%s cycle=%d/%d elapsed=%.1fs red=%d green=%d',
                NODE_NAME,
                defect_type,
                current_count + 1,
                self.light_action_state['count'],
                elapsed,
                self.light_action_state['red'],
                self.light_action_state['green'],
            )
        else:
            self.publish_device(red=0, green=0)
            rospy.loginfo_throttle(
                1.0,
                '%s: light action off-gap type=%s cycle=%d/%d elapsed=%.1fs',
                NODE_NAME,
                defect_type,
                current_count + 1,
                self.light_action_state['count'],
                elapsed,
            )
        return False

    def rotate_black_by_feedback(self):
        """根据当前航向反馈累计旋转角度，完成黑色图形旋转动作。"""
        current = self.get_current_pose()
        if current is None or self.active_defect is None:
            return False

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        if self.rotation_feedback_state is None:
            self.rotation_feedback_state = {
                'last_yaw': current_yaw,
                'accumulated': 0.0,
                'direction': self.black_rotation_direction,
                'commanded_mz': 0.0,
            }
            rospy.loginfo(
                '%s: black marker rotation start current_yaw=%.1fdeg target_angle=%.1fdeg',
                NODE_NAME,
                math.degrees(current_yaw),
                math.degrees(self.black_rotation_angle),
            )

        state = self.rotation_feedback_state
        delta = wrap_angle(current_yaw - state['last_yaw'])
        directed_delta = delta * state['direction']
        if abs(delta) > self.rotation_feedback_max_delta:
            rospy.logwarn_throttle(
                1.0,
                '%s: ignore abnormal yaw feedback jump %.1fdeg',
                NODE_NAME,
                math.degrees(delta),
            )
        elif directed_delta > self.rotation_feedback_deadband:
            state['accumulated'] += directed_delta
        state['last_yaw'] = current_yaw

        finish_angle = max(0.0, self.black_rotation_angle - self.rotation_stop_margin)
        if state['accumulated'] >= finish_angle:
            self.publish_device(red=0, green=0)
            self.publish_pose_cmd(MODE_DPROV, self.active_defect['action'])
            rospy.loginfo(
                '%s: black marker rotation completed, accumulated=%.1f deg',
                NODE_NAME,
                math.degrees(state['accumulated']),
            )
            return True

        action = self.active_defect['action']
        rotating_target = self.make_level_pose(
            action.pose.position.x,
            action.pose.position.y,
            current_yaw,
        )
        remaining = max(0.0, finish_angle - state['accumulated'])
        desired_mz_magnitude = (
            self.black_rotation_slow_mz
            if remaining <= self.black_rotation_slow_angle
            else abs(self.black_rotation_mz)
        )
        desired_mz = state['direction'] * desired_mz_magnitude
        state['commanded_mz'] = clamp(
            desired_mz,
            state['commanded_mz'] - self.black_rotation_mz_step,
            state['commanded_mz'] + self.black_rotation_mz_step,
        )
        self.publish_pose_cmd(
            MODE_DPROV,
            rotating_target,
            mz=state['commanded_mz'],
        )
        rospy.loginfo_throttle(
            1.0,
            '%s: black marker rotating with point hold, accumulated=%.1f/%.1f deg '
            'remaining=%.1fdeg MZ=%d',
            NODE_NAME,
            math.degrees(state['accumulated']),
            math.degrees(self.black_rotation_angle),
            math.degrees(remaining),
            int(state['commanded_mz']),
        )
        return False

    def publish_trajectory_status(self):
        """发布轻量 JSON，供 Task1 测试网页实时绘制规划线和机器人轨迹。"""
        now = rospy.Time.now()
        if (
            now - self.last_trajectory_publish_time
        ).to_sec() < self.trajectory_publish_period:
            return

        current = self.get_current_pose()
        if current is not None:
            point = current.pose.position
            if (
                not self.actual_trajectory
                or xy_distance(point, self.actual_trajectory[-1])
                >= self.trajectory_sample_distance
            ):
                self.actual_trajectory.append(Point(point.x, point.y, self.reference_z))
                if len(self.actual_trajectory) > self.trajectory_max_points:
                    self.actual_trajectory = self.actual_trajectory[-self.trajectory_max_points:]

        def point_data(point):
            return [round(point.x, 3), round(point.y, 3)]

        payload = {
            'stamp': round(now.to_sec(), 3),
            'state': self.step_name(self.step),
            'completed_length': round(self.completed_path_length, 3),
            'robot': (
                point_data(current.pose.position) if current is not None else None
            ),
            'actual_path': [point_data(point) for point in self.actual_trajectory],
            'planned_curve': [point_data(point) for point in self.line_curve_points],
            'raw_line': [point_data(point) for point in self.line_raw_points],
            'pending_markers': [
                {
                    'type': defect['type'],
                    'point': point_data(defect['marker'].pose.position),
                }
                for defect in self.pending_defects
            ],
            'handled_markers': [
                {'type': marker['type'], 'point': point_data(marker['point'])}
                for marker in self.handled_markers
            ],
            'active_marker': (
                {
                    'type': self.active_defect['type'],
                    'point': point_data(self.active_defect['marker'].pose.position),
                }
                if self.active_defect is not None else None
            ),
            'counts': copy.deepcopy(self.handled_counts),
            'required_counts': {
                'yellow': self.required_yellow_count,
                'black': self.required_black_count,
            },
            'endpoint_candidate': (
                point_data(self.line_end_candidate.pose.position)
                if self.line_end_candidate is not None else None
            ),
        }
        encoded = json.dumps(payload, separators=(',', ':'))
        self.trajectory_pub.publish(String(data=encoded))
        if self.trajectory_web is not None:
            self.trajectory_web.update(encoded)
        rospy.loginfo_throttle(
            2.0,
            '%s: action counts completed yellow=%d/%d black=%d/%d',
            NODE_NAME,
            self.handled_counts['yellow'],
            self.required_yellow_count,
            self.handled_counts['black'],
            self.required_black_count,
        )
        self.last_trajectory_publish_time = now

    ############################################### 结束处理层 #######################################
    def log_line_end(self):
        """记录任务结束时可用的红线终点。"""
        if self.line_end_pose is None:
            rospy.logwarn('%s: finished without red line end pose', NODE_NAME)
            return

        rospy.loginfo(
            '%s: red line end recorded at x=%.2f, y=%.2f, z=%.2f',
            NODE_NAME,
            self.line_end_pose.pose.position.x,
            self.line_end_pose.pose.position.y,
            self.line_end_pose.pose.position.z,
        )

    ############################################### 主循环 ###########################################
    def run(self):
        """执行“起点建系→寻找红线→巡线→图形动作→红线结束”的任务流程。"""
        while not rospy.is_shutdown():
            if not self.initialize_mission_frame():
                self.rate.sleep()
                continue

            self.publish_trajectory_status()

            # 启动阶段：固定初始位置，至少等待 10 s 且确认相机持续出图。
            if self.step == self.STEP_WAIT_READY:
                hold_target = self.make_level_pose(
                    self.initial_pose.pose.position.x,
                    self.initial_pose.pose.position.y,
                    self.initial_yaw,
                )
                self.publish_pose_cmd(MODE_DPROV, hold_target)
                ready = self.camera_ready()
                rospy.loginfo_throttle(
                    1.0,
                    '%s: startup point hold elapsed=%.1f/%.1fs camera_ready=%s topic=%s',
                    NODE_NAME,
                    self.step_elapsed(),
                    self.startup_hold_seconds,
                    ready,
                    self.camera_topic,
                )
                if self.step_elapsed() >= self.startup_hold_seconds and ready:
                    heading_target = self.make_level_pose(
                        self.initial_pose.pose.position.x,
                        self.initial_pose.pose.position.y,
                        self.initial_search_yaw,
                    )
                    self.begin_settle(
                        heading_target,
                        self.STEP_SEARCH_LINE,
                        'startup_heading_alignment',
                        completion='reset_search',
                    )

            # 所有任务过渡统一先制动、回目标点，并验证速度稳定。
            elif self.step == self.STEP_SETTLE:
                self.run_settle()

            # 步骤0：从红色圆形原点向前搜索红色长线。
            elif self.step == self.STEP_SEARCH_LINE:
                if self.line_curve_ready() and self.line_is_recent():
                    self.verifying_line_end = False
                    current = self.get_current_pose()
                    if current is not None:
                        target = self.make_level_pose(
                            current.pose.position.x,
                            current.pose.position.y,
                            self.last_line_yaw,
                        )
                        rospy.loginfo('%s: red line found, settle before following', NODE_NAME)
                        self.begin_settle(
                            target,
                            self.STEP_FOLLOW_LINE,
                            'line_acquired',
                        )
                else:
                    self.search_line_forward()

            # 步骤1：正常巡线；若当前位置到达稳定图形上方，则停止巡线并执行动作。
            elif self.step == self.STEP_FOLLOW_LINE:
                ready_index = self.ready_defect_index()
                if ready_index is not None:
                    self.active_defect = self.pending_defects.pop(ready_index)
                    self.active_defect['action'] = self.action_pose_from_current()
                    if self.active_defect['action'] is None:
                        rospy.logwarn('%s: cannot get current pose for marker action', NODE_NAME)
                        self.pending_defects.insert(ready_index, self.active_defect)
                        self.active_defect = None
                        self.rate.sleep()
                        continue
                    rospy.loginfo(
                        '%s: trigger %s marker action on path, marker=(%.2f, %.2f), '
                        'hold_pose=(%.2f, %.2f), queue_left=%d',
                        NODE_NAME,
                        self.active_defect['type'],
                        self.active_defect['marker'].pose.position.x,
                        self.active_defect['marker'].pose.position.y,
                        self.active_defect['action'].pose.position.x,
                        self.active_defect['action'].pose.position.y,
                        len(self.pending_defects),
                    )
                    self.light_action_state = None
                    self.rotation_feedback_state = None
                    self.begin_settle(
                        self.active_defect['action'],
                        self.STEP_LIGHT_ACTION,
                        'arrive_{}_marker'.format(self.active_defect['type']),
                    )
                elif self.line_curve_ready() and self.curve_blind_follow_allowed():
                    self.follow_line_curve_manual()
                elif self.line_start_pose is not None:
                    current = self.get_current_pose()
                    if current is not None:
                        hold_yaw = (
                            self.last_line_yaw if self.last_line_yaw is not None
                            else yaw_from_quaternion(current.pose.orientation)
                        )
                        target = self.make_level_pose(
                            current.pose.position.x,
                            current.pose.position.y,
                            hold_yaw,
                        )
                        self.begin_settle(
                            target,
                            self.STEP_SEARCH_LINE,
                            'line_temporarily_lost',
                            completion='evaluate_line_loss',
                        )
                else:
                    self.search_line_forward()

            # 步骤2：兼容保留；正常流程已经沿巡线到达图形上方，不再主动导航到图形坐标。
            elif self.step == self.STEP_MOVE_TO_MARKER:
                if self.active_defect is None:
                    self.set_step(self.STEP_FOLLOW_LINE)
                else:
                    if self.active_defect.get('action') is None:
                        self.active_defect['action'] = self.action_pose_from_current()
                    if self.active_defect.get('action') is None:
                        self.set_step(self.STEP_FOLLOW_LINE)
                        continue
                    self.begin_settle(
                        self.active_defect['action'],
                        self.STEP_LIGHT_ACTION,
                        'legacy_marker_transition',
                    )

            # 步骤3：执行灯光动作。
            elif self.step == self.STEP_LIGHT_ACTION:
                if self.run_light_action():
                    if self.active_defect is not None and self.active_defect['type'] == 'black':
                        self.rotation_feedback_state = None
                        self.begin_settle(
                            self.active_defect['action'],
                            self.STEP_ROTATE_BLACK,
                            'light_to_black_rotation',
                        )
                    else:
                        self.begin_settle(
                            self.active_defect['action'],
                            self.STEP_SEARCH_LINE,
                            'return_to_yellow_marker',
                            completion='complete_defect',
                        )

            # 步骤4：黑色方形额外执行原地旋转，完成后返回巡线。
            elif self.step == self.STEP_ROTATE_BLACK:
                if self.rotate_black_by_feedback():
                    self.begin_settle(
                        self.active_defect['action'],
                        self.STEP_SEARCH_LINE,
                        'return_to_black_marker',
                        completion='complete_defect',
                    )

            # 步骤5：记录红线终点并发布任务完成。
            elif self.step == self.STEP_FINISH:
                self.log_line_end()
                self.finish_task()

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME)
    try:
        Task1V2().run()
    except rospy.ROSInterruptException:
        pass
