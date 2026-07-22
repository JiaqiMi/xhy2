#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v3_los_square.py
功能：Task1 LOS 固定正方形轨迹单项测试，使用 motion_supervisor 运动接口。

流程：
    1. 节点启动时记录机器人和双目摄像头当前位置、当前高度和当前航向；
    2. 以双目摄像头位置为起点生成“前、右、后、回原点”的正方形轨迹；
    3. 启动阶段向运动监督器发布启动位姿并等待 HOVER；
    4. LOS 按顺序选择固定轨迹前视点，只有位置和航向均到达后才推进；
    5. 把摄像头前视点补偿为 base_link 目标后发布，由运动监督器完成控制和刹车；
    6. 回到原点附近后发布原点目标并等待 HOVER；
    7. Web 实时显示固定轨迹、LOS 跟踪点、机器人航向和实际运动轨迹。

监听：/motion/state，/tf
发布：/cmd/motion/goal，/cmd/motion/cancel，/task1/v3/los_square/trajectory，/finished
网页：默认 http://192.168.1.117:8083

记录：
2026.7.18
    初版。增加基于启动位姿的 1 m 正方形固定轨迹、LOS 控制和 Web 显示。
    明确正方形在启动时一次生成并整体冻结；运行期间任何外部点都不能修改该轨迹。
    新增 v3：LOS 只计算前视目标，不再直接下发 TX 和目标模式；相邻目标位置
    与 yaw 做步长限制后交给 motion_supervisor，并以匹配目标的新鲜 HOVER 判定到达。
    将 Web 实际轨迹最大保存点数开放为 launch 参数，避免长时间测试持续增长。
2026.7.20
    修复闭合正方形起终点重合时，全局最近投影可能直接跳到路径末段并提前完成的问题。
    轨迹和 LOS 改用双目摄像头位置，控制目标按 TF 杆臂补偿回 base_link；
    当前航点的位置与目标航向同时满足门槛后才顺序推进。
    控制周期完整诊断使用 logdebug，控制台 loginfo 只保留阶段和节流摘要。
2026.7.20
    Web 改为 Reset 原点下的 NED 俯视图并固定比例尺，支持滚轮缩放和拖动；
    轨迹、机器人和目标显示 base_link，实际 camera 位置作为航向箭头终点。
    航点改为先到达并定点，再对准下一航点；四个正方形拐角强制成为航点。
2026.7.20
    增加带时间戳的 YAML 数据文件，定期保存固定轨迹、机器人位姿、LOS目标、
    监督器状态和已完成路径，便于复盘控制过程。
2026.7.22
    增加参考深度参数；-1 使用启动时深度，其他值作为全程运动深度。
"""

import copy
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rospy
import tf
from auv_control.msg import MotionState
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import Empty, String
from task1_v3_yaml_logger import TimestampedYamlLogger
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task1_v3_los_square"


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


TRAJECTORY_HTML = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Task1 LOS 正方形测试</title><style>
body{margin:0;background:#101820;color:#eef;font-family:Arial,"Microsoft YaHei"}
header{padding:12px 18px;background:#172630}canvas{display:block;margin:16px auto;background:#f7fbfd;border-radius:8px}
#s{margin-left:25px;color:#bcd0d8}.legend{font-size:13px;color:#bcd0d8;margin-top:8px}.legend span{margin-right:18px}
</style></head><body><header><b>Task1 LOS 固定正方形测试</b><span id="s">等待数据</span>
<div class="legend"><span>红线：base_link 固定正方形轨迹</span><span>蓝线：base_link 实际轨迹</span>
<span>青色圆点→箭头：base_link→camera</span><span>紫点：当前 base_link 目标</span><span>滚轮缩放，拖动平移</span></div></header>
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
function draw(d){last=d;x.clearRect(0,0,c.width,c.height);grid();line(d.planned_path,'#e74c3c',4);line(d.actual_path,'#1677ff',3);dot(d.tracking_point,'#9b2cff',7);tag(d.tracking_point,'目标','#6f13ba');dot(d.robot,'#00cfe8',8);bodyArrow(d.robot,d.camera);
document.getElementById('s').textContent=`任务 ${d.state}/${d.los_phase||'-'}　监督器 ${d.motion_state??'-'}　base航向 ${(d.robot_yaw_deg||0).toFixed(1)}°　D(base/camera) ${(d.robot_down??0).toFixed(2)}/${(d.camera_down??0).toFixed(2)} m　已完成 ${d.completed_length||0}/${d.total_length||0} m　监督器力 ${d.tx||0}/${d.ty||0}/${d.mz||0}`}
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


class LosSquareTest:
    """使用启动位姿生成固定正方形，并单独验证 LOS 控制。"""

    WAIT_START = "WAIT_START"
    FOLLOW = "FOLLOW"
    HOLD_END = "HOLD_END"
    FINISH = "FINISH"
    LOS_MOVE = "MOVE_TO_WAYPOINT"
    LOS_ALIGN = "ALIGN_AT_WAYPOINT"

    def __init__(self):
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))
        self.tf_listener = tf.TransformListener()

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.reference_depth = float(rospy.get_param("~reference_depth", -1.0))
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.line_tracking_frame = str(rospy.get_param(
            "~line_tracking_frame", "camera"
        )).strip().lstrip("/") or "camera"
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
            "~motion_state_timeout", 5
        )))
        self.motion_goal_position_tolerance = max(0.001, float(rospy.get_param(
            "~motion_goal_position_tolerance", 0.05
        )))
        self.motion_goal_yaw_tolerance = math.radians(max(
            0.1,
            float(rospy.get_param("~motion_goal_yaw_tolerance_deg", 3.0)),
        ))
        self.finished_topic = rospy.get_param("~finished_topic", "/finished")
        self.trajectory_topic = rospy.get_param(
            "~trajectory_topic", "/task1/v3/los_square/trajectory"
        )
        self.log_directory = rospy.get_param(
            "~log_directory", "~/.ros/auv_logs/task1"
        )

        self.square_side_length = max(0.1, float(rospy.get_param(
            "~square_side_length", 1.0
        )))
        self.path_sample_spacing = max(0.01, float(rospy.get_param(
            "~path_sample_spacing", 0.05
        )))
        self.startup_hold_seconds = max(0.0, float(rospy.get_param(
            "~startup_hold_seconds", 10.0
        )))
        self.log_period_seconds = max(0.1, float(rospy.get_param(
            "~log_period_seconds", 2.0
        )))

        self.los_lookahead_distance = max(0.01, float(rospy.get_param(
            "~los_lookahead_distance", 0.40
        )))
        self.los_goal_max_step = max(0.01, float(rospy.get_param(
            "~los_goal_max_step", 0.20
        )))
        self.los_goal_max_yaw_step = math.radians(max(
            0.1,
            float(rospy.get_param("~los_goal_max_yaw_step_deg", 20.0)),
        ))
        self.los_waypoint_yaw_tolerance = math.radians(max(
            0.1,
            float(rospy.get_param(
                "~los_waypoint_yaw_tolerance_deg", 10.0
            )),
        ))
        self.los_waypoint_hold_seconds = max(0.0, float(rospy.get_param(
            "~los_waypoint_hold_seconds", 0.5
        )))

        self.endpoint_path_tolerance = max(0.0, float(rospy.get_param(
            "~endpoint_path_tolerance", 0.15
        )))
        self.endpoint_position_tolerance = max(0.0, float(rospy.get_param(
            "~endpoint_position_tolerance", 0.15
        )))
        self.endpoint_hold_seconds = max(0.0, float(rospy.get_param(
            "~endpoint_hold_seconds", 4.0
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

        self.motion_goal_pub = rospy.Publisher(
            self.motion_goal_topic, PoseStamped, queue_size=1
        )
        self.motion_cancel_pub = rospy.Publisher(
            self.motion_cancel_topic, Empty, queue_size=1
        )
        self.finished_pub = rospy.Publisher(
            self.finished_topic, String, queue_size=10
        )
        self.trajectory_pub = rospy.Publisher(
            self.trajectory_topic, String, queue_size=2
        )
        rospy.Subscriber(
            self.motion_state_topic,
            MotionState,
            self.motion_state_callback,
            queue_size=1,
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

        self.state = self.WAIT_START
        self.start_pose = None
        self.start_yaw = None
        self.hold_z = None
        self.startup_started = None
        self.planned_path = []
        self.planned_path_s = []
        self.current_path_s = 0.0
        self.completed_path_length = 0.0
        self.current_tracking_point = None
        self.active_los_target_s = None
        self.active_los_target = None
        self.active_los_yaw = None
        self.active_los_move_yaw = None
        self.active_los_phase = None
        self.active_los_hold_started = None
        self.last_los_goal = None
        self.tracking_lever_arm = None
        self.end_goal = None
        self.end_hold_started = None
        self.latest_motion_state = None
        self.last_motion_goal = None
        self.cancel_sent = False
        self.actual_trajectory = []
        self.last_trajectory_publish_time = rospy.Time(0)
        self.data_logger = None
        self.open_data_log()
        rospy.on_shutdown(self.shutdown)

    def open_data_log(self):
        try:
            self.data_logger = TimestampedYamlLogger(
                NODE_NAME, self.log_directory
            )
            self.write_data_record(
                "startup",
                log_directory=self.log_directory,
                square_side_length=self.square_side_length,
                los_lookahead_distance=self.los_lookahead_distance,
            )
            rospy.loginfo(
                "%s: 完整数据文件=%s", NODE_NAME, self.data_logger.path
            )
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

    def motion_state_callback(self, message):
        self.latest_motion_state = copy.deepcopy(message)

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
        """返回控制器使用的 base_link 当前位姿。"""
        return self.get_frame_pose(self.robot_frame)

    def get_tracking_pose(self):
        """返回固定轨迹和 LOS 使用的双目摄像头当前位姿。"""
        return self.get_frame_pose(self.line_tracking_frame)

    def get_tracking_lever_arm(self):
        """读取 base_link 指向双目摄像头的水平杆臂。"""
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
            "%s: LOS 定位坐标=%s，%s 水平杆臂=(%.3f, %.3f) m",
            NODE_NAME,
            self.line_tracking_frame,
            self.robot_frame,
            self.tracking_lever_arm[0],
            self.tracking_lever_arm[1],
        )
        return self.tracking_lever_arm

    @staticmethod
    def cumulative_distance(points):
        distances = [0.0]
        for index in range(1, len(points)):
            distances.append(
                distances[-1] + xy_distance(points[index - 1], points[index])
            )
        return distances

    def sample_square(self, origin, yaw):
        forward = (math.cos(yaw), math.sin(yaw))
        right = (-math.sin(yaw), math.cos(yaw))
        side = self.square_side_length
        vertices = [
            Point(origin.x, origin.y, self.hold_z),
            Point(origin.x + side * forward[0],
                  origin.y + side * forward[1], self.hold_z),
            Point(origin.x + side * (forward[0] + right[0]),
                  origin.y + side * (forward[1] + right[1]), self.hold_z),
            Point(origin.x + side * right[0],
                  origin.y + side * right[1], self.hold_z),
            Point(origin.x, origin.y, self.hold_z),
        ]
        sampled = [copy.deepcopy(vertices[0])]
        for start, end in zip(vertices[:-1], vertices[1:]):
            length = xy_distance(start, end)
            steps = max(1, int(math.ceil(length / self.path_sample_spacing)))
            for step in range(1, steps + 1):
                ratio = float(step) / steps
                sampled.append(Point(
                    start.x + ratio * (end.x - start.x),
                    start.y + ratio * (end.y - start.y),
                    self.hold_z,
                ))
        return sampled

    def initialize_start_pose(self):
        if self.start_pose is not None:
            return True
        current = self.get_current_pose()
        tracking = self.get_tracking_pose()
        if current is None or tracking is None:
            return False
        self.start_pose = copy.deepcopy(current)
        self.start_yaw = yaw_from_quaternion(current.pose.orientation)
        self.hold_z = (
            current.pose.position.z
            if self.reference_depth == -1.0
            else self.reference_depth
        )
        self.start_pose.pose.position.z = self.hold_z
        self.startup_started = rospy.Time.now()
        self.planned_path = self.sample_square(
            tracking.pose.position, self.start_yaw
        )
        self.planned_path_s = self.cumulative_distance(self.planned_path)
        self.current_tracking_point = copy.deepcopy(self.planned_path[0])
        self.actual_trajectory.append(copy.deepcopy(current.pose.position))
        rospy.loginfo(
            "%s: 双目起点=(%.2f, %.2f, %.2f)，参考深度=%.2f m（%s），"
            "初始航向=%.1f deg，"
            "正方形边长=%.2f m，总轨迹=%.2f m",
            NODE_NAME,
            tracking.pose.position.x,
            tracking.pose.position.y,
            tracking.pose.position.z,
            self.hold_z,
            "当前深度" if self.reference_depth == -1.0 else "launch 设置",
            math.degrees(self.start_yaw),
            self.square_side_length,
            self.planned_path_s[-1],
        )
        return True

    def make_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(x, y, self.hold_z)
        pose.pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw))
        return pose

    def make_tracking_goal(self, camera_point, target_yaw):
        """把 map 下的摄像头 XY 目标补偿为 base_link 控制目标。"""
        base_point = self.tracking_point_to_base_point(camera_point, target_yaw)
        if base_point is None:
            return None
        return self.make_pose(base_point.x, base_point.y, target_yaw)

    def tracking_point_to_base_point(self, camera_point, target_yaw):
        """按目标航向将 camera 轨迹点换算为 base_link 轨迹点。"""
        lever_arm = self.get_tracking_lever_arm()
        if lever_arm is None:
            return None
        cosine = math.cos(target_yaw)
        sine = math.sin(target_yaw)
        offset_x = cosine * lever_arm[0] - sine * lever_arm[1]
        offset_y = sine * lever_arm[0] + cosine * lever_arm[1]
        return Point(
            camera_point.x - offset_x,
            camera_point.y - offset_y,
            camera_point.z,
        )

    def tracking_curve_to_base_points(self, points):
        """生成仅供 Web 显示的 base_link 正方形轨迹。"""
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

    def publish_motion_goal(self, target):
        goal = copy.deepcopy(target)
        goal.header.frame_id = self.map_frame
        goal.header.stamp = rospy.Time.now()
        self.last_motion_goal = copy.deepcopy(goal)
        self.motion_goal_pub.publish(goal)

    def publish_dprov(self, target):
        """兼容原函数名；v3 只向 motion_supervisor 发布目标。"""
        self.publish_motion_goal(target)

    def publish_los_goal(self, camera_point, desired_yaw):
        """补偿相机杆臂并限制 base_link 目标变化。"""
        target_yaw = desired_yaw
        if self.last_los_goal is not None:
            previous = self.last_los_goal
            previous_yaw = yaw_from_quaternion(previous.pose.orientation)
            yaw_step = clamp(
                wrap_angle(desired_yaw - previous_yaw),
                -self.los_goal_max_yaw_step,
                self.los_goal_max_yaw_step,
            )
            target_yaw = wrap_angle(previous_yaw + yaw_step)

        compensated = self.make_tracking_goal(camera_point, target_yaw)
        if compensated is None:
            return None
        target_x = compensated.pose.position.x
        target_y = compensated.pose.position.y
        if self.last_los_goal is not None:
            previous = self.last_los_goal
            dx = target_x - previous.pose.position.x
            dy = target_y - previous.pose.position.y
            distance = math.hypot(dx, dy)
            if distance > self.los_goal_max_step:
                ratio = self.los_goal_max_step / distance
                target_x = previous.pose.position.x + ratio * dx
                target_y = previous.pose.position.y + ratio * dy
        goal = self.make_pose(target_x, target_y, target_yaw)
        self.last_los_goal = copy.deepcopy(goal)
        self.publish_motion_goal(goal)
        return goal

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

    def point_at_path_s(self, target_s):
        target_s = clamp(target_s, 0.0, self.planned_path_s[-1])
        for index in range(len(self.planned_path_s) - 1):
            start_s = self.planned_path_s[index]
            end_s = self.planned_path_s[index + 1]
            if target_s > end_s:
                continue
            start = self.planned_path[index]
            end = self.planned_path[index + 1]
            if end_s <= start_s:
                return copy.deepcopy(start)
            ratio = (target_s - start_s) / (end_s - start_s)
            return Point(
                start.x + ratio * (end.x - start.x),
                start.y + ratio * (end.y - start.y),
                self.hold_z,
            )
        return copy.deepcopy(self.planned_path[-1])

    def path_yaw_at_s(self, target_s):
        """使用当前航点到下一前视点的方向作为机器人目标航向。"""
        path_end_s = self.planned_path_s[-1]
        target = self.point_at_path_s(target_s)
        next_s = min(path_end_s, target_s + self.los_lookahead_distance)
        next_point = self.point_at_path_s(next_s)
        if xy_distance(target, next_point) > 1e-6:
            return math.atan2(next_point.y - target.y, next_point.x - target.x)
        previous = self.point_at_path_s(max(
            0.0, target_s - self.los_lookahead_distance
        ))
        return math.atan2(target.y - previous.y, target.x - previous.x)

    def next_waypoint_s(self, current_s):
        """前视航点不能跨过正方形拐角。"""
        path_end_s = self.planned_path_s[-1]
        proposed = min(path_end_s, current_s + self.los_lookahead_distance)
        for corner_index in range(1, 5):
            corner_s = min(path_end_s, corner_index * self.square_side_length)
            if current_s + 1e-6 < corner_s < proposed - 1e-6:
                return corner_s
        return proposed

    def yaw_to_next_waypoint(self, target_s):
        """到点后使用当前航点到下一航点的方向对向。"""
        target = self.point_at_path_s(target_s)
        next_s = self.next_waypoint_s(target_s)
        next_point = self.point_at_path_s(next_s)
        if xy_distance(target, next_point) > 1e-6:
            return math.atan2(next_point.y - target.y, next_point.x - target.x)
        return self.path_yaw_at_s(target_s)

    def clear_active_los_target(self):
        self.active_los_target_s = None
        self.active_los_target = None
        self.active_los_yaw = None
        self.active_los_move_yaw = None
        self.active_los_phase = None
        self.active_los_hold_started = None

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

    def enter_hold_end(self):
        self.last_los_goal = None
        self.clear_active_los_target()
        self.end_hold_started = None
        self.current_tracking_point = copy.deepcopy(self.planned_path[-1])
        self.end_goal = self.make_tracking_goal(
            self.planned_path[-1], self.start_yaw
        )
        if self.end_goal is None:
            return
        self.state = self.HOLD_END
        self.publish_dprov(self.end_goal)
        rospy.loginfo(
            "%s: 已走完固定轨迹，发布原点目标并等待运动监督器 HOVER",
            NODE_NAME,
        )

    def run_follow(self, current, tracking):
        remaining = max(0.0, self.planned_path_s[-1] - self.current_path_s)
        endpoint_distance = xy_distance(
            tracking.pose.position, self.planned_path[-1]
        )
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        endpoint_yaw = self.path_yaw_at_s(self.planned_path_s[-1])
        endpoint_yaw_error = abs(wrap_angle(endpoint_yaw - current_yaw))
        if (
            self.active_los_target is None
            and remaining <= self.endpoint_path_tolerance
            and endpoint_distance <= self.endpoint_position_tolerance
            and endpoint_yaw_error <= self.los_waypoint_yaw_tolerance
        ):
            self.enter_hold_end()
            return

        if self.active_los_target is None:
            self.active_los_target_s = self.next_waypoint_s(self.current_path_s)
            self.active_los_target = self.point_at_path_s(
                self.active_los_target_s
            )
            self.active_los_yaw = self.yaw_to_next_waypoint(
                self.active_los_target_s
            )
            self.active_los_move_yaw = current_yaw
            self.active_los_phase = self.LOS_MOVE
            self.active_los_hold_started = None

        self.current_tracking_point = copy.deepcopy(self.active_los_target)
        position_error = xy_distance(
            tracking.pose.position, self.active_los_target
        )
        desired_yaw = (
            self.active_los_move_yaw
            if self.active_los_phase == self.LOS_MOVE
            else self.active_los_yaw
        )
        yaw_error = abs(wrap_angle(desired_yaw - current_yaw))
        commanded_goal = self.publish_los_goal(self.active_los_target, desired_yaw)
        if commanded_goal is None:
            return

        if self.active_los_phase == self.LOS_MOVE:
            position_stable = (
                position_error <= self.endpoint_position_tolerance
                and self.motion_arrived(commanded_goal)
            )
            if position_stable:
                if self.active_los_hold_started is None:
                    self.active_los_hold_started = rospy.Time.now()
            else:
                self.active_los_hold_started = None
            held = (
                self.active_los_hold_started is not None
                and (rospy.Time.now() - self.active_los_hold_started).to_sec()
                >= self.los_waypoint_hold_seconds
            )
            if held:
                self.active_los_phase = self.LOS_ALIGN
                self.active_los_hold_started = None
                rospy.loginfo(
                    "%s: 正方形航点位置已到达并定点；s=%.2f m，开始调整航向 %.1f deg",
                    NODE_NAME,
                    self.active_los_target_s,
                    math.degrees(self.active_los_yaw),
                )
        elif (
            position_error <= self.endpoint_position_tolerance
            and abs(wrap_angle(self.active_los_yaw - current_yaw))
            <= self.los_waypoint_yaw_tolerance
            and self.motion_arrived(commanded_goal)
        ):
            reached_s = self.active_los_target_s
            self.current_path_s = max(self.current_path_s, reached_s)
            self.completed_path_length = self.current_path_s
            rospy.loginfo(
                "%s: 正方形航点移动和对向均完成；camera位置误差=%.2f m，"
                "航向误差=%.1f deg，推进到 s=%.2f/%.2f m",
                NODE_NAME,
                position_error,
                math.degrees(abs(wrap_angle(
                    self.active_los_yaw - current_yaw
                ))),
                reached_s,
                self.planned_path_s[-1],
            )
            self.clear_active_los_target()

        motion_state = (
            str(self.latest_motion_state.state)
            if self.motion_state_fresh()
            else "无新鲜反馈"
        )
        rospy.loginfo_throttle(
            self.log_period_seconds,
            "%s: LOS跟踪；阶段=%s，camera位置=(%.2f, %.2f)，跟踪点=(%.2f, %.2f)，"
            "位置误差=%.2f m，航向误差=%.1f deg，进度=%.2f/%.2f m，"
            "motion_state=%s",
            NODE_NAME,
            self.active_los_phase or "航点完成",
            tracking.pose.position.x,
            tracking.pose.position.y,
            self.current_tracking_point.x,
            self.current_tracking_point.y,
            position_error,
            math.degrees(yaw_error),
            self.completed_path_length,
            self.planned_path_s[-1],
            motion_state,
        )

    def run_hold_end(self, current, tracking):
        self.publish_dprov(self.end_goal)
        position_error = xy_distance(
            tracking.pose.position, self.planned_path[-1]
        )
        yaw_error = abs(wrap_angle(
            self.start_yaw - yaw_from_quaternion(current.pose.orientation)
        ))
        stable = self.motion_arrived(self.end_goal)
        if stable:
            if self.end_hold_started is None:
                self.end_hold_started = rospy.Time.now()
        else:
            self.end_hold_started = None
        held = (
            (rospy.Time.now() - self.end_hold_started).to_sec()
            if self.end_hold_started is not None else 0.0
        )
        rospy.loginfo_throttle(
            self.log_period_seconds,
            "%s: 原点定点；位置误差=%.2f m，航向误差=%.1f deg，"
            "HOVER=%s，保持=%.1f/%.1f s",
            NODE_NAME,
            position_error,
            math.degrees(yaw_error),
            "是" if stable else "否",
            held,
            self.endpoint_hold_seconds,
        )
        if stable and held >= self.endpoint_hold_seconds:
            self.state = self.FINISH

    def publish_trajectory_status(self, current, tracking):
        now = rospy.Time.now()
        if (
            now - self.last_trajectory_publish_time
        ).to_sec() < self.trajectory_publish_period:
            return

        def point_data(point):
            return [round(point.x, 3), round(point.y, 3)] if point else None

        planned_base_path = self.tracking_curve_to_base_points(self.planned_path)
        base_target = (
            self.last_motion_goal.pose.position
            if self.last_motion_goal is not None else None
        )

        payload = {
            "stamp": round(now.to_sec(), 3),
            "state": self.state,
            "motion_state": (
                self.latest_motion_state.state
                if self.motion_state_fresh() else None
            ),
            "path_fixed": True,
            "robot": point_data(current.pose.position),
            "camera": point_data(tracking.pose.position),
            "robot_down": round(current.pose.position.z, 3),
            "camera_down": round(tracking.pose.position.z, 3),
            "robot_yaw_deg": round(math.degrees(yaw_from_quaternion(
                current.pose.orientation
            )), 2),
            "tracking_frame": self.line_tracking_frame,
            "tracking_point": point_data(base_target),
            "camera_tracking_point": point_data(self.current_tracking_point),
            "los_phase": self.active_los_phase,
            "planned_path": [point_data(point) for point in planned_base_path],
            "actual_path": [point_data(point) for point in self.actual_trajectory],
            "completed_length": round(self.completed_path_length, 3),
            "total_length": round(self.planned_path_s[-1], 3),
            "tx": self.latest_motion_state.tx if self.motion_state_fresh() else 0,
            "ty": self.latest_motion_state.ty if self.motion_state_fresh() else 0,
            "mz": self.latest_motion_state.mz if self.motion_state_fresh() else 0,
        }
        self.write_data_record("trajectory_update", **payload)
        encoded = json.dumps(payload, separators=(",", ":"))
        self.trajectory_pub.publish(String(data=encoded))
        if self.trajectory_web is not None:
            self.trajectory_web.update(encoded)
        self.last_trajectory_publish_time = now

    def log_debug_cycle(self, current, tracking):
        """以 DEBUG 级别记录每个控制周期的完整诊断。"""
        target = self.current_tracking_point
        goal = self.last_motion_goal
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        target_yaw = (
            self.active_los_move_yaw
            if self.active_los_phase == self.LOS_MOVE
            else (
                self.active_los_yaw
                if self.active_los_yaw is not None else current_yaw
            )
        )
        position_error = (
            xy_distance(tracking.pose.position, target)
            if target is not None else float("nan")
        )
        yaw_error = (
            math.degrees(abs(wrap_angle(target_yaw - current_yaw)))
            if target is not None else float("nan")
        )
        motion_state = self.latest_motion_state
        rospy.logdebug(
            "%s: FULL state=%s base=(%.3f,%.3f,%.3f,%.2fdeg) "
            "%s=(%.3f,%.3f,%.3f) camera_target=(%.3f,%.3f) "
            "base_goal=(%.3f,%.3f,%.3f,%.2fdeg) error=(%.3fm,%.2fdeg) "
            "path=%.3f/%.3f active=(%s,%s) motion=(%s,%s,%s,%s,%s)",
            NODE_NAME,
            self.state,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(current_yaw),
            self.line_tracking_frame,
            tracking.pose.position.x,
            tracking.pose.position.y,
            tracking.pose.position.z,
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
            self.planned_path_s[-1] if self.planned_path_s else 0.0,
            "%.3f" % self.active_los_target_s
            if self.active_los_target_s is not None else "-",
            self.active_los_phase or "-",
            motion_state.state if motion_state is not None else "-",
            motion_state.reason if motion_state is not None else "-",
            motion_state.tx if motion_state is not None else 0,
            motion_state.ty if motion_state is not None else 0,
            motion_state.mz if motion_state is not None else 0,
        )
        self.write_data_record(
            "control_cycle",
            base={
                "position": [
                    current.pose.position.x,
                    current.pose.position.y,
                    current.pose.position.z,
                ],
                "yaw_deg": math.degrees(current_yaw),
            },
            camera={
                "position": [
                    tracking.pose.position.x,
                    tracking.pose.position.y,
                    tracking.pose.position.z,
                ],
            },
            camera_target=(
                [target.x, target.y, target.z]
                if target is not None else None
            ),
            command_goal=(
                {
                    "position": [
                        goal.pose.position.x,
                        goal.pose.position.y,
                        goal.pose.position.z,
                    ],
                    "yaw_deg": math.degrees(yaw_from_quaternion(
                        goal.pose.orientation
                    )),
                }
                if goal is not None else None
            ),
            position_error=position_error,
            yaw_error_deg=yaw_error,
            completed_path=self.completed_path_length,
            total_path=(
                self.planned_path_s[-1] if self.planned_path_s else 0.0
            ),
            active_target_s=self.active_los_target_s,
            los_phase=self.active_los_phase,
            motion=(
                {
                    "state": motion_state.state,
                    "reason": motion_state.reason,
                    "tx": motion_state.tx,
                    "ty": motion_state.ty,
                    "mz": motion_state.mz,
                }
                if motion_state is not None else None
            ),
        )

    def finish(self):
        self.cancel_motion()
        self.finished_pub.publish(String(data="los square finished"))
        rospy.loginfo(
            "%s: FINISH；固定正方形 LOS 测试完成，总轨迹=%.2f m",
            NODE_NAME,
            self.planned_path_s[-1],
        )
        rospy.signal_shutdown("los square test complete")

    def run(self):
        while not rospy.is_shutdown():
            if not self.initialize_start_pose():
                self.rate.sleep()
                continue
            current = self.get_current_pose()
            tracking = self.get_tracking_pose()
            if current is None or tracking is None:
                self.rate.sleep()
                continue
            if self.motion_failed():
                rospy.logerr_throttle(
                    self.log_period_seconds,
                    "%s: motion_supervisor=SAFE，暂停任务推进，原因=%s",
                    NODE_NAME,
                    self.latest_motion_state.reason,
                )
                self.log_debug_cycle(current, tracking)
                self.rate.sleep()
                continue
            self.record_actual_trajectory(current)

            if self.state == self.WAIT_START:
                self.publish_dprov(self.start_pose)
                elapsed = (rospy.Time.now() - self.startup_started).to_sec()
                rospy.loginfo_throttle(
                    self.log_period_seconds,
                    "%s: 启动定点；当前位置=(%.2f, %.2f, %.2f)，"
                    "航向=%.1f deg，等待=%.1f/%.1f s",
                    NODE_NAME,
                    current.pose.position.x,
                    current.pose.position.y,
                    current.pose.position.z,
                    math.degrees(yaw_from_quaternion(current.pose.orientation)),
                    elapsed,
                    self.startup_hold_seconds,
                )
                if (
                    elapsed >= self.startup_hold_seconds
                    and self.motion_arrived(self.start_pose)
                ):
                    self.last_los_goal = copy.deepcopy(self.start_pose)
                    self.state = self.FOLLOW
                    rospy.loginfo("%s: 启动定点完成，开始 LOS 正方形跟踪", NODE_NAME)
            elif self.state == self.FOLLOW:
                self.run_follow(current, tracking)
            elif self.state == self.HOLD_END:
                self.run_hold_end(current, tracking)
            elif self.state == self.FINISH:
                self.publish_trajectory_status(current, tracking)
                self.finish()
                break

            self.log_debug_cycle(current, tracking)
            self.publish_trajectory_status(current, tracking)
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME)
    LosSquareTest().run()
