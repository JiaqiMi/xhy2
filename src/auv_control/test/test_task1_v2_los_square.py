#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v2_los_square.py
功能：Task1 LOS 固定正方形轨迹单项测试，不使用视觉识别结果。

流程：
    1. 节点启动时记录机器人当前位置、当前高度和当前航向；
    2. 以该位姿为起点生成“前、右、后、回原点”的正方形轨迹；
    3. 启动阶段使用 mode=4 在原位定点悬停；
    4. LOS 在固定轨迹上投影机器人位置，并沿轨迹向前选择前视点；
    5. 使用 mode=3 发布目标航向和 TX，沿完整正方形运行；
    6. 回到原点附近后直接切入 mode=4，由下位机自动刹车并定点稳定；
    7. Web 实时显示固定轨迹、LOS 跟踪点、机器人航向和实际运动轨迹。

监听：/status/vel（可选），/tf
发布：/cmd/pose/ned，/task1/los_square/trajectory，/finished
网页：默认 http://192.168.1.117:8082

记录：
2026.7.18
    初版。增加基于启动位姿的 1 m 正方形固定轨迹、LOS 控制和 Web 显示。
    明确正方形在启动时一次生成并整体冻结；运行期间任何外部点都不能修改该轨迹。
"""

import copy
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rospy
import tf
from auv_control.msg import PoseNEDcmd
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TwistStamped
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task1_v2_los_square"
MODE_DEPTH_HDG = 3
MODE_DPROV = 4


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
<div class="legend"><span>红线：固定正方形轨迹</span><span>蓝线：机器人实际轨迹</span>
<span>青色箭头：机器人位置与航向</span><span>紫点：LOS 当前跟踪点</span></div></header>
<canvas id="c" width="960" height="680"></canvas><script>
const c=document.getElementById('c'),x=c.getContext('2d'),p=45;let k=1,mx=0,my=0;
function q(a){return[p+(a[0]-mx)*k,c.height-p-(a[1]-my)*k]}
function line(a,col,w){if(!a||a.length<2)return;x.beginPath();a.forEach((v,i)=>{let z=q(v);i?x.lineTo(...z):x.moveTo(...z)});x.strokeStyle=col;x.lineWidth=w;x.stroke()}
function dot(a,col,r){if(!a)return;let z=q(a);x.beginPath();x.arc(z[0],z[1],r,0,7);x.fillStyle=col;x.fill()}
function tag(a,text,col){if(!a)return;let z=q(a);x.fillStyle=col;x.font='bold 13px Arial';x.fillText(text,z[0]+7,z[1]-7)}
function arrow(a,yawDeg){if(!a)return;let z=q(a),r=yawDeg*Math.PI/180,L=34,ex=z[0]+L*Math.cos(r),ey=z[1]-L*Math.sin(r),h=9;
x.beginPath();x.moveTo(z[0],z[1]);x.lineTo(ex,ey);x.strokeStyle='#00a9c7';x.lineWidth=4;x.stroke();
x.beginPath();x.moveTo(ex,ey);x.lineTo(ex-h*Math.cos(r-.55),ey+h*Math.sin(r-.55));x.moveTo(ex,ey);x.lineTo(ex-h*Math.cos(r+.55),ey+h*Math.sin(r+.55));x.stroke()}
function draw(d){let a=[...(d.planned_path||[]),...(d.actual_path||[])];if(d.robot)a.push(d.robot);if(d.tracking_point)a.push(d.tracking_point);
x.clearRect(0,0,c.width,c.height);if(!a.length)return;let xs=a.map(v=>v[0]),ys=a.map(v=>v[1]),xx=Math.max(...xs)+.2,yy=Math.max(...ys)+.2;
mx=Math.min(...xs)-.2;my=Math.min(...ys)-.2;k=Math.min((c.width-2*p)/Math.max(.5,xx-mx),(c.height-2*p)/Math.max(.5,yy-my));
line(d.planned_path,'#e74c3c',4);line(d.actual_path,'#1677ff',3);dot(d.tracking_point,'#9b2cff',7);tag(d.tracking_point,'LOS','#6f13ba');dot(d.robot,'#00cfe8',8);arrow(d.robot,d.robot_yaw_deg||0);
document.getElementById('s').textContent=`状态 ${d.state}　固定轨迹 是　航向 ${(d.robot_yaw_deg||0).toFixed(1)}°　已完成 ${d.completed_length||0}/${d.total_length||0} m　TX ${d.tx||0}`}
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


class LosSquareTest:
    """使用启动位姿生成固定正方形，并单独验证 LOS 控制。"""

    WAIT_START = "WAIT_START"
    FOLLOW = "FOLLOW"
    HOLD_END = "HOLD_END"
    FINISH = "FINISH"

    def __init__(self):
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))
        self.tf_listener = tf.TransformListener()

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd/pose/ned")
        self.velocity_topic = rospy.get_param("~velocity_topic", "/status/vel")
        self.finished_topic = rospy.get_param("~finished_topic", "/finished")
        self.trajectory_topic = rospy.get_param(
            "~trajectory_topic", "/task1/los_square/trajectory"
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
        self.los_forward_force = abs(float(rospy.get_param(
            "~los_forward_force", 1000.0
        )))
        self.los_slow_forward_force = abs(float(rospy.get_param(
            "~los_slow_forward_force", 300.0
        )))
        self.los_force_step = abs(float(rospy.get_param(
            "~los_force_step", 200.0
        )))
        self.los_tx_sign = 1.0 if float(rospy.get_param(
            "~los_tx_sign", 1.0
        )) >= 0.0 else -1.0
        self.los_stop_yaw_error = math.radians(float(rospy.get_param(
            "~los_stop_yaw_error_deg", 25.0
        )))
        self.los_slow_yaw_error = math.radians(float(rospy.get_param(
            "~los_slow_yaw_error_deg", 12.0
        )))
        self.los_slow_distance = max(0.0, float(rospy.get_param(
            "~los_slow_distance", 0.40
        )))

        self.endpoint_path_tolerance = max(0.0, float(rospy.get_param(
            "~endpoint_path_tolerance", 0.15
        )))
        self.endpoint_position_tolerance = max(0.0, float(rospy.get_param(
            "~endpoint_position_tolerance", 0.15
        )))
        self.endpoint_yaw_tolerance = math.radians(float(rospy.get_param(
            "~endpoint_yaw_tolerance_deg", 5.0
        )))
        self.endpoint_hold_seconds = max(0.0, float(rospy.get_param(
            "~endpoint_hold_seconds", 4.0
        )))
        self.velocity_message_timeout = max(0.0, float(rospy.get_param(
            "~velocity_message_timeout", 1.0
        )))
        self.stable_linear_speed = max(0.0, float(rospy.get_param(
            "~stable_linear_speed", 0.05
        )))
        self.stable_angular_speed = math.radians(float(rospy.get_param(
            "~stable_angular_speed_deg", 3.0
        )))

        self.actual_path_min_spacing = max(0.001, float(rospy.get_param(
            "~actual_path_min_spacing", 0.02
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
            "~trajectory_web_port", 8082
        ))

        self.cmd_pub = rospy.Publisher(self.cmd_topic, PoseNEDcmd, queue_size=10)
        self.finished_pub = rospy.Publisher(
            self.finished_topic, String, queue_size=10
        )
        self.trajectory_pub = rospy.Publisher(
            self.trajectory_topic, String, queue_size=2
        )
        rospy.Subscriber(
            self.velocity_topic, TwistStamped, self.velocity_callback, queue_size=5
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
        self.last_los_tx = 0.0
        self.end_hold_started = None
        self.latest_velocity = None
        self.latest_velocity_time = None
        self.pose_speed_sample = None
        self.actual_trajectory = []
        self.last_trajectory_publish_time = rospy.Time(0)

    def velocity_callback(self, message):
        self.latest_velocity = copy.deepcopy(message.twist)
        self.latest_velocity_time = rospy.Time.now()

    def get_current_pose(self):
        try:
            translation, rotation = self.tf_listener.lookupTransform(
                self.map_frame, self.robot_frame, rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(
                2.0, "%s: 等待 %s -> %s TF: %s",
                NODE_NAME, self.map_frame, self.robot_frame, error
            )
            return None
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

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
        if current is None:
            return False
        self.start_pose = copy.deepcopy(current)
        self.start_yaw = yaw_from_quaternion(current.pose.orientation)
        self.hold_z = current.pose.position.z
        self.startup_started = rospy.Time.now()
        self.planned_path = self.sample_square(
            current.pose.position, self.start_yaw
        )
        self.planned_path_s = self.cumulative_distance(self.planned_path)
        self.current_tracking_point = copy.deepcopy(self.planned_path[0])
        self.actual_trajectory.append(copy.deepcopy(current.pose.position))
        rospy.loginfo(
            "%s: 起点=(%.2f, %.2f, %.2f)，初始航向=%.1f deg，"
            "正方形边长=%.2f m，总轨迹=%.2f m",
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
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

    @staticmethod
    def force_value(value):
        return int(round(clamp(value, -10000.0, 10000.0)))

    def publish_pose_command(self, mode, target, tx=0.0):
        command = PoseNEDcmd()
        command.mode = int(mode)
        command.target = copy.deepcopy(target)
        command.target.header.frame_id = self.map_frame
        command.target.header.stamp = rospy.Time.now()
        command.force.TX = self.force_value(tx)
        self.cmd_pub.publish(command)

    def publish_dprov(self, target):
        self.publish_pose_command(MODE_DPROV, target)

    def publish_manual(self, current, yaw, tx):
        target = self.make_pose(
            current.pose.position.x, current.pose.position.y, yaw
        )
        self.publish_pose_command(MODE_DEPTH_HDG, target, tx=tx)

    def project_to_path(self, point):
        best = None
        for index in range(len(self.planned_path) - 1):
            start = self.planned_path[index]
            end = self.planned_path[index + 1]
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
                "path_s": self.planned_path_s[index] + ratio * segment_length,
            }
            if best is None or distance < best["distance"]:
                best = candidate
        return best

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

    def desired_los_force(self, yaw_error, remaining_distance):
        if abs(yaw_error) > self.los_stop_yaw_error:
            return 0.0
        force = self.los_forward_force
        if (
            abs(yaw_error) > self.los_slow_yaw_error
            or remaining_distance <= self.los_slow_distance
        ):
            force = self.los_slow_forward_force
        return self.los_tx_sign * force

    def smooth_los_force(self, desired):
        self.last_los_tx = clamp(
            desired,
            self.last_los_tx - self.los_force_step,
            self.last_los_tx + self.los_force_step,
        )
        return self.last_los_tx

    def motion_is_stable(self, current):
        now = rospy.Time.now()
        if (
            self.latest_velocity is not None
            and self.latest_velocity_time is not None
            and (now - self.latest_velocity_time).to_sec()
            <= self.velocity_message_timeout
        ):
            linear_speed = math.hypot(
                self.latest_velocity.linear.x,
                self.latest_velocity.linear.y,
            )
            angular_speed = abs(self.latest_velocity.angular.z)
        else:
            yaw = yaw_from_quaternion(current.pose.orientation)
            sample = (now, current.pose.position.x, current.pose.position.y, yaw)
            previous = self.pose_speed_sample
            self.pose_speed_sample = sample
            if previous is None:
                return False
            elapsed = (now - previous[0]).to_sec()
            if elapsed <= 0.05:
                return False
            linear_speed = math.hypot(
                sample[1] - previous[1], sample[2] - previous[2]
            ) / elapsed
            angular_speed = abs(wrap_angle(sample[3] - previous[3])) / elapsed
        return (
            linear_speed <= self.stable_linear_speed
            and angular_speed <= self.stable_angular_speed
        )

    def record_actual_trajectory(self, current):
        if (
            not self.actual_trajectory
            or xy_distance(current.pose.position, self.actual_trajectory[-1])
            >= self.actual_path_min_spacing
        ):
            self.actual_trajectory.append(copy.deepcopy(current.pose.position))

    def enter_hold_end(self):
        self.last_los_tx = 0.0
        self.end_hold_started = None
        self.current_tracking_point = copy.deepcopy(self.planned_path[-1])
        self.state = self.HOLD_END
        self.publish_dprov(self.start_pose)
        rospy.loginfo(
            "%s: 已走完固定轨迹，切入 mode=4 返回原点并由系统自动刹车",
            NODE_NAME,
        )

    def run_follow(self, current):
        projection = self.project_to_path(current.pose.position)
        if projection is None:
            return
        self.current_path_s = max(self.current_path_s, projection["path_s"])
        self.completed_path_length = max(
            self.completed_path_length, self.current_path_s
        )
        remaining = max(0.0, self.planned_path_s[-1] - self.current_path_s)
        endpoint_distance = xy_distance(
            current.pose.position, self.planned_path[-1]
        )
        if (
            remaining <= self.endpoint_path_tolerance
            and endpoint_distance <= self.endpoint_position_tolerance
        ):
            self.enter_hold_end()
            return

        target = self.point_at_path_s(
            self.current_path_s + self.los_lookahead_distance
        )
        self.current_tracking_point = copy.deepcopy(target)
        desired_yaw = math.atan2(
            target.y - current.pose.position.y,
            target.x - current.pose.position.x,
        )
        yaw_error = wrap_angle(
            desired_yaw - yaw_from_quaternion(current.pose.orientation)
        )
        desired_tx = self.desired_los_force(yaw_error, remaining)
        tx = self.smooth_los_force(desired_tx)
        self.publish_manual(current, desired_yaw, tx)
        rospy.loginfo_throttle(
            self.log_period_seconds,
            "%s: LOS跟踪；位置=(%.2f, %.2f)，跟踪点=(%.2f, %.2f)，"
            "进度=%.2f/%.2f m，航向误差=%.1f deg，TX=%d",
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            target.x,
            target.y,
            self.completed_path_length,
            self.planned_path_s[-1],
            math.degrees(yaw_error),
            self.force_value(tx),
        )

    def run_hold_end(self, current):
        self.publish_dprov(self.start_pose)
        position_error = xy_distance(
            current.pose.position, self.start_pose.pose.position
        )
        yaw_error = abs(wrap_angle(
            self.start_yaw - yaw_from_quaternion(current.pose.orientation)
        ))
        stable = (
            position_error <= self.endpoint_position_tolerance
            and yaw_error <= self.endpoint_yaw_tolerance
            and self.motion_is_stable(current)
        )
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
            "稳定=%s，保持=%.1f/%.1f s",
            NODE_NAME,
            position_error,
            math.degrees(yaw_error),
            "是" if stable else "否",
            held,
            self.endpoint_hold_seconds,
        )
        if stable and held >= self.endpoint_hold_seconds:
            self.state = self.FINISH

    def publish_trajectory_status(self, current):
        now = rospy.Time.now()
        if (
            now - self.last_trajectory_publish_time
        ).to_sec() < self.trajectory_publish_period:
            return

        def point_data(point):
            return [round(point.x, 3), round(point.y, 3)] if point else None

        payload = {
            "stamp": round(now.to_sec(), 3),
            "state": self.state,
            "path_fixed": True,
            "robot": point_data(current.pose.position),
            "robot_yaw_deg": round(math.degrees(yaw_from_quaternion(
                current.pose.orientation
            )), 2),
            "tracking_point": point_data(self.current_tracking_point),
            "planned_path": [point_data(point) for point in self.planned_path],
            "actual_path": [point_data(point) for point in self.actual_trajectory],
            "completed_length": round(self.completed_path_length, 3),
            "total_length": round(self.planned_path_s[-1], 3),
            "tx": self.force_value(self.last_los_tx),
        }
        encoded = json.dumps(payload, separators=(",", ":"))
        self.trajectory_pub.publish(String(data=encoded))
        if self.trajectory_web is not None:
            self.trajectory_web.update(encoded)
        self.last_trajectory_publish_time = now

    def finish(self):
        self.publish_dprov(self.start_pose)
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
            if current is None:
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
                if elapsed >= self.startup_hold_seconds:
                    self.state = self.FOLLOW
                    rospy.loginfo("%s: 启动定点完成，开始 LOS 正方形跟踪", NODE_NAME)
            elif self.state == self.FOLLOW:
                self.run_follow(current)
            elif self.state == self.HOLD_END:
                self.run_hold_end(current)
            elif self.state == self.FINISH:
                self.publish_trajectory_status(current)
                self.finish()
                break

            self.publish_trajectory_status(current)
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node(NODE_NAME)
    LosSquareTest().run()
