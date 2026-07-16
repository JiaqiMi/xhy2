#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v2_line_follow.py
功能：Task1 巡线单项测试。

流程：
    1. 以节点启动时机器人当前位置为起点并保持当前 z，等待相机稳定出图；
    2. 定点慢速对准初始航向，左右扫描；未发现红线则前进 0.5 m 后重复；
    3. 将持续识别到的红线点融合为拟合曲线；
    4. 使用定深定向手控模式沿曲线进行 LOS 巡线；
    5. 局部前向点稳定、曲线停止增长且红线超时丢失后才发布 /finished。

监听：/obj/line_message，/left/image_raw，/status/vel（可选），/tf
发布：/cmd/pose/ned，/task1/trajectory，/finished；网页默认地址 192.168.1.117:8082

记录：
2026.7.14
    初版，用于单独验证 Task1 巡线通信与控制流程。
2026.7.16
    同步正式 Task1 的 P1/P3、启动等待、扫描搜索、制动定点、速度稳定、终点判据和轨迹发布。
    轨迹网页默认地址调整为 192.168.1.117:8082，并开放绑定地址和端口参数。
    将 P1/P2/P3 解释为局部近中远三点，增加几何验证、连续帧确认和主管线数据关联。
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
from auv_control.msg import PoseNEDcmd, TargetDetection3
from geometry_msgs.msg import Point, PoseStamped, Quaternion, TwistStamped
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = "test_task1_v2_line_follow"
MODE_DEPTH_HDG = 3
MODE_DPROV = 4
DEFAULT_INITIAL_HEADING_DEG = 0.0


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
<title>Task1 巡线测试轨迹</title><style>body{margin:0;background:#101820;color:#eef;font-family:Arial,"Microsoft YaHei"}
header{padding:12px 18px;background:#172630}canvas{display:block;margin:16px auto;background:#f7fbfd;border-radius:8px}
#s{margin-left:25px;color:#bcd0d8}</style></head><body><header><b>Task1 巡线测试轨迹</b><span id="s">等待数据</span></header>
<canvas id="c" width="960" height="680"></canvas><script>const c=document.getElementById('c'),x=c.getContext('2d'),p=45;let k=1,mx=0,my=0;
function q(a){return[p+(a[0]-mx)*k,c.height-p-(a[1]-my)*k]}function line(a,col,w){if(!a||a.length<2)return;x.beginPath();a.forEach((v,i)=>{let z=q(v);i?x.lineTo(...z):x.moveTo(...z)});x.strokeStyle=col;x.lineWidth=w;x.stroke()}
function dot(a,col,r){let z=q(a);x.beginPath();x.arc(z[0],z[1],r,0,7);x.fillStyle=col;x.fill()}
function draw(d){let a=[...(d.actual_path||[]),...(d.planned_curve||[]),...(d.raw_line||[])];if(d.robot)a.push(d.robot);x.clearRect(0,0,c.width,c.height);if(!a.length)return;
let xs=a.map(v=>v[0]),ys=a.map(v=>v[1]),xx=Math.max(...xs)+.2,yy=Math.max(...ys)+.2;mx=Math.min(...xs)-.2;my=Math.min(...ys)-.2;k=Math.min((c.width-2*p)/Math.max(.5,xx-mx),(c.height-2*p)/Math.max(.5,yy-my));
(d.raw_line||[]).forEach(v=>dot(v,'#879aa3',2));line(d.planned_curve,'#e74c3c',3);line(d.actual_path,'#1677ff',3);if(d.endpoint_candidate)dot(d.endpoint_candidate,'#8e44ad',7);if(d.robot)dot(d.robot,'#00cfe8',8);
document.getElementById('s').textContent=`状态 ${d.state}　已完成 ${d.completed_length||0} m`}
async function t(){try{draw(await(await fetch('/data',{cache:'no-store'})).json())}catch(e){}setTimeout(t,500)}t();</script></body></html>"""


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

    def __init__(self):
        self.rate = rospy.Rate(float(rospy.get_param("~rate", 5.0)))
        self.tf_listener = tf.TransformListener()

        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd/pose/ned")
        self.line_topic = rospy.get_param("~line_topic", "/obj/line_message")
        self.finished_topic = rospy.get_param("~finished_topic", "/finished")

        self.initial_search_yaw = math.radians(float(rospy.get_param(
            "~initial_heading_deg", DEFAULT_INITIAL_HEADING_DEG
        )))
        self.search_forward_force = float(rospy.get_param("~search_forward_force", 1000.0))
        self.search_slow_forward_force = float(rospy.get_param(
            "~search_slow_forward_force", 300.0
        ))
        self.search_forward_step = float(rospy.get_param("~search_forward_step", 0.5))
        self.search_deceleration_distance = float(rospy.get_param(
            "~search_deceleration_distance", 0.2
        ))
        self.search_scan_hold_seconds = float(rospy.get_param(
            "~search_scan_hold_seconds", 2.0
        ))
        self.search_scan_offsets = [
            math.radians(float(value)) for value in rospy.get_param(
                "~search_scan_offsets_deg", [0.0, 30.0, -30.0, 0.0]
            )
        ]
        self.manual_forward_force = float(rospy.get_param("~manual_forward_force", 1000.0))
        self.manual_slow_forward_force = float(rospy.get_param(
            "~manual_slow_forward_force", 300.0
        ))
        self.manual_lateral_gain = float(rospy.get_param("~manual_lateral_gain", 6000.0))
        self.manual_min_lateral_force = float(rospy.get_param(
            "~manual_min_lateral_force", 1500.0
        ))
        self.manual_max_lateral_force = float(rospy.get_param(
            "~manual_max_lateral_force", 3000.0
        ))
        self.manual_lateral_deadband = float(rospy.get_param(
            "~manual_lateral_deadband", 0.03
        ))
        self.manual_force_step = float(rospy.get_param("~manual_force_step", 200.0))
        self.manual_brake_step = float(rospy.get_param("~manual_brake_step", 300.0))
        self.manual_tx_sign = float(rospy.get_param("~manual_tx_sign", 1.0))
        self.manual_ty_sign = float(rospy.get_param("~manual_ty_sign", 1.0))

        self.los_lookahead_distance = float(rospy.get_param("~los_lookahead_distance", 0.6))
        self.manual_slow_yaw_error = math.radians(float(rospy.get_param(
            "~manual_slow_yaw_error_deg", 20.0
        )))
        self.manual_slow_lateral_error = float(rospy.get_param(
            "~manual_slow_lateral_error", 0.25
        ))

        self.line_lost_timeout = float(rospy.get_param("~line_lost_timeout", 5.0))
        self.curve_blind_follow_timeout = float(rospy.get_param(
            "~curve_blind_follow_timeout", 2.0
        ))
        self.line_point_merge_distance = float(rospy.get_param(
            "~line_point_merge_distance", 0.15
        ))
        self.line_min_point_spacing = float(rospy.get_param(
            "~line_min_point_spacing", 0.03
        ))
        self.line_max_point_spacing = float(rospy.get_param(
            "~line_max_point_spacing", 3.0
        ))
        self.line_middle_offset_tolerance = float(rospy.get_param(
            "~line_middle_offset_tolerance", 0.25
        ))
        self.line_point_order_tolerance = float(rospy.get_param(
            "~line_point_order_tolerance", 0.15
        ))
        self.line_local_max_bend = math.radians(float(rospy.get_param(
            "~line_local_max_bend_deg", 45.0
        )))
        self.line_candidate_confirm_frames = int(rospy.get_param(
            "~line_candidate_confirm_frames", 3
        ))
        self.line_candidate_center_distance = float(rospy.get_param(
            "~line_candidate_center_distance", 0.50
        ))
        self.line_candidate_yaw_tolerance = math.radians(float(rospy.get_param(
            "~line_candidate_yaw_tolerance_deg", 20.0
        )))
        self.line_association_distance = float(rospy.get_param(
            "~line_association_distance", 0.50
        ))
        self.line_association_angle = math.radians(float(rospy.get_param(
            "~line_association_angle_deg", 35.0
        )))
        self.line_association_backtrack = float(rospy.get_param(
            "~line_association_backtrack", 0.60
        ))
        self.line_extension_max_gap = float(rospy.get_param(
            "~line_extension_max_gap", 1.0
        ))
        self.line_curve_max_points = int(rospy.get_param("~line_curve_max_points", 120))
        self.line_curve_sample_count = int(rospy.get_param("~line_curve_sample_count", 80))
        self.line_curve_degree = int(rospy.get_param("~line_curve_degree", 3))
        self.line_curve_min_length = float(rospy.get_param("~line_curve_min_length", 0.4))
        self.line_classes = class_names("~line_classes", ["line"])
        self.max_line_direction_change = math.radians(float(rospy.get_param(
            "~max_line_direction_change_deg", 75.0
        )))

        self.camera_topic = rospy.get_param("~camera_topic", "/left/image_raw")
        self.camera_message_timeout = float(rospy.get_param(
            "~camera_message_timeout", 2.0
        ))
        self.startup_hold_seconds = float(rospy.get_param("~startup_hold_seconds", 10.0))
        self.transition_hold_seconds = float(rospy.get_param(
            "~transition_hold_seconds", 4.0
        ))
        self.velocity_topic = rospy.get_param("~velocity_topic", "/status/vel")
        self.velocity_message_timeout = float(rospy.get_param(
            "~velocity_message_timeout", 1.0
        ))
        self.stable_linear_speed = float(rospy.get_param("~stable_linear_speed", 0.05))
        self.stable_angular_speed = math.radians(float(rospy.get_param(
            "~stable_angular_speed_deg", 3.0
        )))
        self.max_yaw_step = math.radians(float(rospy.get_param("~max_yaw_step_deg", 2.0)))
        self.max_xy_step = float(rospy.get_param("~max_xy_step", 0.4))
        self.position_tolerance = float(rospy.get_param("~position_tolerance", 0.15))
        self.yaw_tolerance = math.radians(float(rospy.get_param("~yaw_tolerance_deg", 2.0)))

        self.endpoint_stable_count = int(rospy.get_param("~endpoint_stable_count", 8))
        self.endpoint_position_tolerance = float(rospy.get_param(
            "~endpoint_position_tolerance", 0.12
        ))
        self.endpoint_growth_tolerance = float(rospy.get_param(
            "~endpoint_growth_tolerance", 0.05
        ))
        self.endpoint_stall_seconds = float(rospy.get_param(
            "~endpoint_stall_seconds", 3.0
        ))
        self.endpoint_min_completed_length = float(rospy.get_param(
            "~endpoint_min_completed_length", 1.0
        ))
        self.endpoint_robot_distance = float(rospy.get_param(
            "~endpoint_robot_distance", 0.8
        ))
        self.trajectory_topic = rospy.get_param("~trajectory_topic", "/task1/trajectory")
        self.trajectory_publish_period = float(rospy.get_param(
            "~trajectory_publish_period", 0.5
        ))
        self.trajectory_web_enabled = bool(rospy.get_param("~trajectory_web_enabled", True))
        self.trajectory_web_host = rospy.get_param(
            "~trajectory_web_host", "192.168.1.117"
        )
        self.trajectory_web_port = int(rospy.get_param("~trajectory_web_port", 8082))

        self.cmd_pub = rospy.Publisher(self.cmd_topic, PoseNEDcmd, queue_size=10)
        self.finished_pub = rospy.Publisher(self.finished_topic, String, queue_size=10)
        self.trajectory_pub = rospy.Publisher(self.trajectory_topic, String, queue_size=2)
        self.trajectory_web = None
        if self.trajectory_web_enabled:
            try:
                self.trajectory_web = TrajectoryWebServer(
                    self.trajectory_web_host,
                    self.trajectory_web_port,
                )
                rospy.loginfo(
                    "%s: trajectory web available at http://%s:%d",
                    NODE_NAME,
                    self.trajectory_web_host,
                    self.trajectory_web_port,
                )
            except OSError as error:
                rospy.logwarn("%s: cannot start trajectory web: %s", NODE_NAME, error)
        rospy.Subscriber(self.line_topic, TargetDetection3, self.line_callback)
        rospy.Subscriber(self.camera_topic, rospy.AnyMsg, self.camera_callback, queue_size=1)
        rospy.Subscriber(self.velocity_topic, TwistStamped, self.velocity_callback, queue_size=5)

        self.start_pose = None
        self.hold_z = None
        self.last_line_time = None
        self.last_line_yaw = None
        self.line_started = False
        self.line_axis_origin = None
        self.line_axis_yaw = None
        self.line_candidate = None
        self.line_raw_points = []
        self.line_curve_points = []
        self.line_curve_s = []
        self.current_path_s = 0.0
        self.completed_path_length = 0.0
        self.last_manual_tx = 0
        self.last_manual_ty = 0
        self.state = "WAIT_READY"
        self.state_started = rospy.Time.now()
        self.last_camera_time = None
        self.latest_velocity = None
        self.latest_velocity_time = None
        self.pose_speed_sample = None
        self.settle_target = None
        self.settle_next_state = None
        self.settle_reason = ""
        self.settle_stable_since = None
        self.search_anchor_pose = None
        self.search_scan_index = 0
        self.search_scan_stable_since = None
        self.search_advance_target = None
        self.max_known_curve_length = 0.0
        self.last_curve_growth_time = rospy.Time.now()
        self.far_endpoint_samples = deque(maxlen=max(2, self.endpoint_stable_count))
        self.endpoint_candidate = None
        self.verifying_line_end = False
        self.actual_trajectory = []
        self.last_trajectory_publish_time = rospy.Time(0)

        rospy.loginfo(
            "%s: initialized, initial_heading=%.1fdeg, cmd_topic=%s, line_topic=%s",
            NODE_NAME,
            math.degrees(self.initial_search_yaw),
            self.cmd_topic,
            self.line_topic,
        )

    def set_state(self, state):
        old_state = self.state
        self.state = state
        self.state_started = rospy.Time.now()
        if old_state != state:
            rospy.loginfo("%s: state %s -> %s", NODE_NAME, old_state, state)

    def state_elapsed(self):
        return (rospy.Time.now() - self.state_started).to_sec()

    def camera_callback(self, _message):
        self.last_camera_time = rospy.Time.now()

    def velocity_callback(self, message):
        self.latest_velocity = copy.deepcopy(message.twist)
        self.latest_velocity_time = rospy.Time.now()

    def camera_ready(self):
        return (
            self.last_camera_time is not None
            and (rospy.Time.now() - self.last_camera_time).to_sec()
            <= self.camera_message_timeout
        )

    @staticmethod
    def approach_zero(value, step):
        if value > 0:
            return max(0, value - step)
        if value < 0:
            return min(0, value + step)
        return 0

    def motion_is_stable(self, current):
        now = rospy.Time.now()
        if (
            self.latest_velocity is not None
            and self.latest_velocity_time is not None
            and (now - self.latest_velocity_time).to_sec()
            <= self.velocity_message_timeout
        ):
            linear = math.hypot(
                self.latest_velocity.linear.x, self.latest_velocity.linear.y
            )
            angular = abs(self.latest_velocity.angular.z)
            source = "velocity_topic"
        else:
            yaw = yaw_from_quaternion(current.pose.orientation)
            sample = (now, current.pose.position.x, current.pose.position.y, yaw)
            previous = self.pose_speed_sample
            self.pose_speed_sample = sample
            if previous is None or (now - previous[0]).to_sec() <= 0.05:
                rospy.loginfo_throttle(1.0, "%s: TF speed estimate warming", NODE_NAME)
                return False
            elapsed = (now - previous[0]).to_sec()
            linear = math.hypot(sample[1] - previous[1], sample[2] - previous[2]) / elapsed
            angular = abs(wrap_angle(sample[3] - previous[3])) / elapsed
            source = "tf_difference"
        stable = (
            linear <= self.stable_linear_speed
            and angular <= self.stable_angular_speed
        )
        rospy.loginfo_throttle(
            1.0,
            "%s: stable check source=%s linear=%.3fm/s angular=%.2fdeg/s result=%s",
            NODE_NAME,
            source,
            linear,
            math.degrees(angular),
            stable,
        )
        return stable

    def get_current_pose(self):
        try:
            self.tf_listener.waitForTransform(
                "map", "base_link", rospy.Time(0), rospy.Duration(1.0)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                "map", "base_link", rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(2, "%s: cannot read current pose: %s", NODE_NAME, error)
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
        rospy.loginfo(
            "%s: start pose recorded x=%.2f, y=%.2f, z=%.2f, current_yaw=%.1fdeg, "
            "search_yaw=%.1fdeg",
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(yaw_from_quaternion(current.pose.orientation)),
            math.degrees(self.initial_search_yaw),
        )
        return True

    def make_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = rospy.Time.now()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = self.hold_z
        pose.pose.orientation = Quaternion(*quaternion_from_euler(0.0, 0.0, yaw))
        return pose

    @staticmethod
    def force_value(value):
        return int(round(clamp(value, -10000, 10000)))

    def publish_pose_cmd(self, yaw, tx=0, ty=0):
        current = self.get_current_pose()
        if current is None or self.hold_z is None:
            return False

        cmd = PoseNEDcmd()
        cmd.mode = MODE_DEPTH_HDG
        cmd.target = self.make_pose(current.pose.position.x, current.pose.position.y, yaw)
        cmd.force.TX = self.force_value(tx)
        cmd.force.TY = self.force_value(ty)
        self.cmd_pub.publish(cmd)

        rospy.loginfo_throttle(
            1.0,
            "%s: cmd mode=3 yaw=%.1fdeg force=(%d,%d), current=(%.2f, %.2f, %.2f)",
            NODE_NAME,
            math.degrees(yaw),
            cmd.force.TX,
            cmd.force.TY,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
        )
        return True

    def publish_dprov(self, target, mz=0):
        cmd = PoseNEDcmd()
        cmd.mode = MODE_DPROV
        cmd.target = copy.deepcopy(target)
        cmd.target.header.stamp = rospy.Time.now()
        cmd.force.MZ = self.force_value(mz)
        self.cmd_pub.publish(cmd)
        rospy.loginfo_throttle(
            1.0,
            "%s: DPROV target=(%.2f, %.2f, yaw=%.1fdeg) MZ=%d",
            NODE_NAME,
            target.pose.position.x,
            target.pose.position.y,
            math.degrees(yaw_from_quaternion(target.pose.orientation)),
            cmd.force.MZ,
        )

    def move_to_pose(self, target):
        current = self.get_current_pose()
        if current is None:
            return False
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        dx = target.pose.position.x - current.pose.position.x
        dy = target.pose.position.y - current.pose.position.y
        distance = math.hypot(dx, dy)
        final_yaw = yaw_from_quaternion(target.pose.orientation)
        if distance > self.position_tolerance:
            move_yaw = math.atan2(dy, dx)
            yaw_error = wrap_angle(move_yaw - current_yaw)
            if abs(yaw_error) > self.yaw_tolerance:
                yaw = current_yaw + clamp(yaw_error, -self.max_yaw_step, self.max_yaw_step)
                self.publish_dprov(self.make_pose(
                    current.pose.position.x, current.pose.position.y, yaw
                ))
                return False
            step = min(self.max_xy_step, distance)
            self.publish_dprov(self.make_pose(
                current.pose.position.x + dx * step / distance,
                current.pose.position.y + dy * step / distance,
                move_yaw,
            ))
            return False
        yaw_error = wrap_angle(final_yaw - current_yaw)
        if abs(yaw_error) > self.yaw_tolerance:
            yaw = current_yaw + clamp(yaw_error, -self.max_yaw_step, self.max_yaw_step)
            self.publish_dprov(self.make_pose(
                target.pose.position.x, target.pose.position.y, yaw
            ))
            return False
        self.publish_dprov(target)
        return True

    def begin_settle(self, target, next_state, reason):
        self.settle_target = copy.deepcopy(target)
        self.settle_next_state = next_state
        self.settle_reason = reason
        self.settle_stable_since = None
        self.pose_speed_sample = None
        rospy.loginfo(
            "%s: begin settle reason=%s target=(%.2f, %.2f, yaw=%.1fdeg)",
            NODE_NAME,
            reason,
            target.pose.position.x,
            target.pose.position.y,
            math.degrees(yaw_from_quaternion(target.pose.orientation)),
        )
        self.set_state("SETTLE")

    def run_settle(self):
        if self.last_manual_tx != 0 or self.last_manual_ty != 0:
            self.last_manual_tx = self.approach_zero(
                self.last_manual_tx, self.manual_brake_step
            )
            self.last_manual_ty = self.approach_zero(
                self.last_manual_ty, self.manual_brake_step
            )
            self.publish_pose_cmd(
                yaw_from_quaternion(self.settle_target.pose.orientation),
                self.last_manual_tx,
                self.last_manual_ty,
            )
            self.settle_stable_since = None
            rospy.loginfo_throttle(
                1.0,
                "%s: braking before point hold force=(%d,%d)",
                NODE_NAME,
                self.last_manual_tx,
                self.last_manual_ty,
            )
            return
        if not self.move_to_pose(self.settle_target):
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
            "%s: point hold reason=%s stable=%.1f/%.1fs",
            NODE_NAME,
            self.settle_reason,
            stable_seconds,
            self.transition_hold_seconds,
        )
        if stable_seconds >= self.transition_hold_seconds:
            next_state = self.settle_next_state
            self.settle_target = None
            self.settle_stable_since = None
            if next_state == "SEARCH":
                self.reset_search_cycle()
            self.set_state(next_state)

    def publish_stop(self):
        current = self.get_current_pose()
        yaw = self.initial_search_yaw
        if current is not None:
            yaw = yaw_from_quaternion(current.pose.orientation)
        self.publish_pose_cmd(yaw, 0, 0)
        self.last_manual_tx = 0
        self.last_manual_ty = 0

    def transform_pose_to_map(self, pose):
        try:
            self.tf_listener.waitForTransform(
                "map", pose.header.frame_id, pose.header.stamp, rospy.Duration(1.0)
            )
            return self.tf_listener.transformPose("map", pose)
        except tf.Exception:
            try:
                self.tf_listener.waitForTransform(
                    "map", pose.header.frame_id, rospy.Time(0), rospy.Duration(1.0)
                )
                return self.tf_listener.transformPose("map", pose)
            except tf.Exception as error:
                rospy.logwarn_throttle(2, "%s: line transform failed: %s", NODE_NAME, error)
                return None

    def downsample_line_points(self):
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

    def smooth_existing_line_point(self, point):
        if not self.line_raw_points:
            return False
        new_point = Point(point.x, point.y, self.hold_z)
        nearest = min(
            self.line_raw_points,
            key=lambda old_point: xy_distance(old_point, new_point),
        )
        if xy_distance(nearest, new_point) > self.line_point_merge_distance:
            return False
        nearest.x = 0.8 * nearest.x + 0.2 * new_point.x
        nearest.y = 0.8 * nearest.y + 0.2 * new_point.y
        return True

    def fuse_ordered_line_segment(self, poses):
        points = [
            Point(pose.pose.position.x, pose.pose.position.y, self.hold_z)
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
                    "%s: reject discontinuous line extension gap=%.2fm",
                    NODE_NAME,
                    gap,
                )
                break
            self.line_raw_points.append(copy.deepcopy(point))
        self.downsample_line_points()

    @staticmethod
    def cumulative_distance(points):
        distances = [0.0]
        for index in range(1, len(points)):
            distances.append(distances[-1] + xy_distance(points[index], points[index - 1]))
        return distances

    def fit_line_curve(self):
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
        sample_count = max(2, self.line_curve_sample_count)
        try:
            x_curve = np.poly1d(np.polyfit(raw_s, [point.x for point in filtered], degree))
            y_curve = np.poly1d(np.polyfit(raw_s, [point.y for point in filtered], degree))
            self.line_curve_points = [
                Point(float(x_curve(value)), float(y_curve(value)), self.hold_z)
                for value in np.linspace(0.0, total_length, sample_count)
            ]
            self.line_curve_s = self.cumulative_distance(self.line_curve_points)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            rospy.logwarn_throttle(2, "%s: curve fit failed: %s", NODE_NAME, error)
            self.line_curve_points = [copy.deepcopy(point) for point in filtered]
            self.line_curve_s = raw_s

    def line_curve_ready(self):
        return len(self.line_curve_points) >= 2 and len(self.line_curve_s) == len(
            self.line_curve_points
        )

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

    def validate_local_line_triplet(self, near, middle, far):
        points = [near.pose.position, middle.pose.position, far.pose.position]
        near_middle = xy_distance(points[0], points[1])
        middle_far = xy_distance(points[1], points[2])
        if min(near_middle, middle_far) < self.line_min_point_spacing:
            return False, None, "point_spacing_too_small"
        if max(near_middle, middle_far) > self.line_max_point_spacing:
            return False, None, "point_spacing_too_large"

        yaw_first = math.atan2(points[1].y - points[0].y, points[1].x - points[0].x)
        yaw_second = math.atan2(points[2].y - points[1].y, points[2].x - points[1].x)
        if abs(wrap_angle(yaw_second - yaw_first)) > self.line_local_max_bend:
            return False, None, "local_bend_too_large"
        if self.point_to_chord_distance(
            points[1], points[0], points[2]
        ) > self.line_middle_offset_tolerance:
            return False, None, "middle_point_off_chord"

        current = self.get_current_pose()
        if current is not None:
            distances = [xy_distance(current.pose.position, point) for point in points]
            if (
                distances[0] > distances[1] + self.line_point_order_tolerance
                or distances[1] > distances[2] + self.line_point_order_tolerance
            ):
                return False, None, "near_middle_far_order_invalid"
        yaw = math.atan2(points[2].y - points[0].y, points[2].x - points[0].x)
        return True, yaw, "valid"

    def line_segment_associated(self, poses, detected_yaw):
        if not self.line_curve_ready():
            return True, "first_line_segment"
        minimum_progress = max(0.0, self.current_path_s - self.line_association_backtrack)
        projections = []
        for pose in poses:
            projection = self.project_to_curve(pose.pose.position)
            if projection is not None and projection["path_s"] >= minimum_progress:
                projections.append(projection)
        if not projections:
            return False, "segment_behind_current_progress"
        best = min(projections, key=lambda value: value["distance"])
        if best["distance"] > self.line_association_distance:
            return False, "segment_too_far_from_active_curve"
        if abs(wrap_angle(detected_yaw - best["segment_yaw"])) > self.line_association_angle:
            return False, "segment_direction_mismatch"
        return True, "associated_with_active_curve"

    def reset_line_candidate(self, reason):
        if self.line_candidate is not None:
            rospy.loginfo_throttle(
                1.0,
                "%s: reset line candidate reason=%s frames=%d",
                NODE_NAME,
                reason,
                self.line_candidate["count"],
            )
        self.line_candidate = None

    def confirm_line_candidate(self, poses, detected_yaw):
        points = [pose.pose.position for pose in poses]
        center = Point(
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
            self.hold_z,
        )
        compatible = (
            self.line_candidate is not None
            and xy_distance(center, self.line_candidate["center"])
            <= self.line_candidate_center_distance
            and abs(wrap_angle(detected_yaw - self.line_candidate["yaw"]))
            <= self.line_candidate_yaw_tolerance
        )
        if compatible:
            self.line_candidate["count"] += 1
            self.line_candidate["center"] = center
            self.line_candidate["yaw"] = detected_yaw
        else:
            self.line_candidate = {"count": 1, "center": center, "yaw": detected_yaw}
        rospy.loginfo_throttle(
            1.0,
            "%s: line candidate frames=%d/%d center=(%.2f,%.2f) yaw=%.1fdeg",
            NODE_NAME,
            self.line_candidate["count"],
            self.line_candidate_confirm_frames,
            center.x,
            center.y,
            math.degrees(detected_yaw),
        )
        return self.line_candidate["count"] >= self.line_candidate_confirm_frames

    def line_callback(self, message):
        if self.hold_z is None and not self.initialize_start_pose():
            return

        if message.class_name and message.class_name not in self.line_classes:
            self.reset_line_candidate("unexpected_line_class")
            return

        # P1/P2/P3 是局部管线上的远/中/近点，不代表整条管线端点。
        far = self.transform_pose_to_map(message.pose1)
        middle = self.transform_pose_to_map(message.pose2)
        near = self.transform_pose_to_map(message.pose3)
        if far is None or middle is None or near is None:
            self.reset_line_candidate("line_tf_failed")
            return

        valid, detected_yaw, reason = self.validate_local_line_triplet(
            near, middle, far
        )
        if not valid:
            rospy.logwarn_throttle(1.0, "%s: reject line reason=%s", NODE_NAME, reason)
            self.reset_line_candidate(reason)
            return
        if (
            self.last_line_yaw is not None
            and abs(wrap_angle(detected_yaw - self.last_line_yaw))
            > self.max_line_direction_change
        ):
            rospy.logwarn_throttle(
                2.0,
                "%s: ignore reverse line candidate detected=%.1fdeg previous=%.1fdeg",
                NODE_NAME,
                math.degrees(detected_yaw),
                math.degrees(self.last_line_yaw),
            )
            self.reset_line_candidate("gross_direction_change")
            return

        associated, reason = self.line_segment_associated(
            (near, middle, far), detected_yaw
        )
        if not associated:
            rospy.logwarn_throttle(
                1.0,
                "%s: isolate unrelated line reason=%s",
                NODE_NAME,
                reason,
            )
            self.reset_line_candidate(reason)
            return
        if not self.confirm_line_candidate((near, middle, far), detected_yaw):
            return

        if self.line_axis_origin is None:
            self.line_axis_origin = copy.deepcopy(near.pose.position)
            self.line_axis_yaw = math.atan2(
                far.pose.position.y - near.pose.position.y,
                far.pose.position.x - near.pose.position.x,
            )
            rospy.loginfo(
                "%s: first red line detected, axis_yaw=%.1fdeg",
                NODE_NAME,
                math.degrees(self.line_axis_yaw),
            )

        self.line_started = True
        self.last_line_time = rospy.Time.now()
        self.last_line_yaw = detected_yaw
        self.fuse_ordered_line_segment((near, middle, far))
        self.fit_line_curve()

        known_curve = self.line_curve_s[-1] if self.line_curve_ready() else 0.0
        if known_curve > self.max_known_curve_length + self.endpoint_growth_tolerance:
            self.max_known_curve_length = known_curve
            self.last_curve_growth_time = rospy.Time.now()
            self.endpoint_candidate = None
        self.update_endpoint_evidence(far)
        rospy.loginfo_throttle(
            1.0,
            "%s: line update P1_local_far=(%.2f,%.2f) P3_local_near=(%.2f,%.2f) "
            "raw=%d curve=%d known_curve=%.2fm line_yaw=%.1fdeg",
            NODE_NAME,
            far.pose.position.x,
            far.pose.position.y,
            near.pose.position.x,
            near.pose.position.y,
            len(self.line_raw_points),
            len(self.line_curve_points),
            known_curve,
            math.degrees(self.last_line_yaw),
        )

    def update_endpoint_evidence(self, far_pose):
        self.far_endpoint_samples.append(copy.deepcopy(far_pose.pose.position))
        if len(self.far_endpoint_samples) < self.far_endpoint_samples.maxlen:
            return
        center = Point(
            sum(point.x for point in self.far_endpoint_samples)
            / len(self.far_endpoint_samples),
            sum(point.y for point in self.far_endpoint_samples)
            / len(self.far_endpoint_samples),
            self.hold_z,
        )
        spread = max(xy_distance(point, center) for point in self.far_endpoint_samples)
        stalled = (
            rospy.Time.now() - self.last_curve_growth_time
        ).to_sec() >= self.endpoint_stall_seconds
        current = self.get_current_pose()
        robot_distance = (
            xy_distance(current.pose.position, center) if current is not None else float("inf")
        )
        near_curve_end = (
            bool(self.line_curve_s)
            and self.current_path_s >= self.line_curve_s[-1] - 0.5
        )
        if (
            spread <= self.endpoint_position_tolerance
            and stalled
            and self.completed_path_length >= self.endpoint_min_completed_length
            and near_curve_end
            and robot_distance <= self.endpoint_robot_distance
        ):
            self.endpoint_candidate = copy.deepcopy(far_pose)
            self.endpoint_candidate.pose.position = center
            rospy.loginfo_throttle(
                1.0,
                "%s: endpoint candidate stable spread=%.3fm robot_to_end=%.2fm completed=%.2fm",
                NODE_NAME,
                spread,
                robot_distance,
                self.completed_path_length,
            )
        else:
            self.endpoint_candidate = None

    def line_is_recent(self):
        if self.last_line_time is None:
            return False
        return (rospy.Time.now() - self.last_line_time).to_sec() <= self.line_lost_timeout

    def blind_follow_allowed(self):
        if self.last_line_time is None:
            return False
        return (
            rospy.Time.now() - self.last_line_time
        ).to_sec() <= self.curve_blind_follow_timeout

    def project_to_curve(self, point):
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
            segment_length = math.sqrt(segment_sq)
            lateral = (vx * (point.y - start.y) - vy * (point.x - start.x)) / segment_length
            path_s = self.line_curve_s[index] + ratio * segment_length
            distance = math.hypot(point.x - proj_x, point.y - proj_y)
            if best is None or distance < best["distance"]:
                best = {
                    "distance": distance,
                    "lateral": lateral,
                    "path_s": path_s,
                    "segment_yaw": math.atan2(vy, vx),
                }
        return best

    def point_at_curve_s(self, target_s):
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
                self.hold_z,
            )
        return copy.deepcopy(self.line_curve_points[-1])

    def limit_force(self, desired, previous):
        return int(round(clamp(
            desired,
            previous - self.manual_force_step,
            previous + self.manual_force_step,
        )))

    def limit_lateral_force(self, desired, previous):
        if desired == 0:
            return self.approach_zero(previous, self.manual_brake_step)
        desired_sign = 1 if desired > 0 else -1
        previous_sign = 1 if previous > 0 else -1 if previous < 0 else 0
        if previous_sign not in (0, desired_sign):
            return self.approach_zero(previous, self.manual_brake_step)
        if previous_sign == 0:
            return desired_sign * int(self.manual_min_lateral_force)
        return self.limit_force(desired, previous)

    def reset_search_cycle(self):
        current = self.get_current_pose()
        if current is None:
            return False
        search_yaw = self.last_line_yaw if self.last_line_yaw is not None else self.initial_search_yaw
        self.search_anchor_pose = copy.deepcopy(current)
        self.search_base_yaw = search_yaw
        self.search_scan_index = 0
        self.search_scan_stable_since = None
        self.search_advance_target = self.make_pose(
            current.pose.position.x + self.search_forward_step * math.cos(search_yaw),
            current.pose.position.y + self.search_forward_step * math.sin(search_yaw),
            search_yaw,
        )
        rospy.loginfo(
            "%s: search cycle anchor=(%.2f,%.2f) yaw=%.1fdeg",
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            math.degrees(search_yaw),
        )
        return True

    def search_line(self):
        if self.search_anchor_pose is None and not self.reset_search_cycle():
            return
        current = self.get_current_pose()
        if current is None:
            return
        if self.search_scan_index < len(self.search_scan_offsets):
            offset = self.search_scan_offsets[self.search_scan_index]
            target = self.make_pose(
                self.search_anchor_pose.pose.position.x,
                self.search_anchor_pose.pose.position.y,
                wrap_angle(self.search_base_yaw + offset),
            )
            reached = self.move_to_pose(target)
            if not reached or not self.motion_is_stable(current):
                self.search_scan_stable_since = None
                return
            if self.search_scan_stable_since is None:
                self.search_scan_stable_since = rospy.Time.now()
            stable_seconds = (
                rospy.Time.now() - self.search_scan_stable_since
            ).to_sec()
            rospy.loginfo_throttle(
                1.0,
                "%s: scan offset=%+.1fdeg index=%d/%d stable=%.1f/%.1fs",
                NODE_NAME,
                math.degrees(offset),
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
            self.begin_settle(
                self.make_pose(
                    self.search_anchor_pose.pose.position.x,
                    self.search_anchor_pose.pose.position.y,
                    self.search_base_yaw,
                ),
                "VERIFY_ENDPOINT",
                "endpoint_scan_completed",
            )
            return

        dx = current.pose.position.x - self.search_anchor_pose.pose.position.x
        dy = current.pose.position.y - self.search_anchor_pose.pose.position.y
        progress = dx * math.cos(self.search_base_yaw) + dy * math.sin(self.search_base_yaw)
        remaining = self.search_forward_step - progress
        if remaining <= self.position_tolerance:
            self.begin_settle(
                self.search_advance_target,
                "SEARCH",
                "search_advance_completed",
            )
            return
        force = (
            self.search_slow_forward_force
            if remaining <= self.search_deceleration_distance
            else self.search_forward_force
        ) * self.manual_tx_sign
        self.last_manual_tx = self.limit_force(force, self.last_manual_tx)
        self.last_manual_ty = self.approach_zero(
            self.last_manual_ty, self.manual_brake_step
        )
        self.publish_pose_cmd(
            self.search_base_yaw,
            tx=self.last_manual_tx,
            ty=self.last_manual_ty,
        )
        rospy.loginfo_throttle(
            1.0,
            "%s: search advance progress=%.2f/%.2fm heading=%.1fdeg TX=%d",
            NODE_NAME,
            progress,
            self.search_forward_step,
            math.degrees(self.search_base_yaw),
            self.last_manual_tx,
        )

    def follow_line(self):
        current = self.get_current_pose()
        if current is None or not self.line_curve_ready():
            self.publish_stop()
            return False

        projection = self.project_to_curve(current.pose.position)
        if projection is None:
            self.publish_stop()
            return False

        self.current_path_s = projection["path_s"]
        self.completed_path_length = max(self.completed_path_length, self.current_path_s)
        los_target = self.point_at_curve_s(self.current_path_s + self.los_lookahead_distance)
        desired_yaw = math.atan2(
            los_target.y - current.pose.position.y,
            los_target.x - current.pose.position.x,
        )
        yaw_error = wrap_angle(desired_yaw - yaw_from_quaternion(current.pose.orientation))
        lateral_error = projection["lateral"]

        forward_force = self.manual_forward_force
        if (
            abs(yaw_error) > self.manual_slow_yaw_error
            or abs(lateral_error) > self.manual_slow_lateral_error
            or not self.line_is_recent()
        ):
            forward_force = self.manual_slow_forward_force

        desired_tx = self.manual_tx_sign * forward_force
        raw_ty = -self.manual_lateral_gain * lateral_error
        if abs(lateral_error) <= self.manual_lateral_deadband:
            desired_ty = 0
        else:
            desired_ty = self.manual_ty_sign * math.copysign(
                clamp(
                    abs(raw_ty),
                    self.manual_min_lateral_force,
                    self.manual_max_lateral_force,
                ),
                raw_ty,
            )
        tx = self.limit_force(desired_tx, self.last_manual_tx)
        ty = self.limit_lateral_force(desired_ty, self.last_manual_ty)
        self.last_manual_tx = tx
        self.last_manual_ty = ty
        self.publish_pose_cmd(desired_yaw, tx=tx, ty=ty)

        rospy.loginfo_throttle(
            1.0,
            "%s: follow completed=%.2fm known_curve=%.2fm lateral=%.2f "
            "yaw_error=%.1fdeg force=(%d,%d)",
            NODE_NAME,
            self.completed_path_length,
            self.line_curve_s[-1],
            lateral_error,
            math.degrees(yaw_error),
            tx,
            ty,
        )
        return True

    def publish_trajectory_status(self):
        now = rospy.Time.now()
        if (
            now - self.last_trajectory_publish_time
        ).to_sec() < self.trajectory_publish_period:
            return
        current = self.get_current_pose()
        if current is not None and (
            not self.actual_trajectory
            or xy_distance(current.pose.position, self.actual_trajectory[-1]) >= 0.03
        ):
            self.actual_trajectory.append(copy.deepcopy(current.pose.position))
            if len(self.actual_trajectory) > 2000:
                self.actual_trajectory = self.actual_trajectory[-2000:]

        def point_data(point):
            return [round(point.x, 3), round(point.y, 3)]

        payload = {
            "stamp": round(now.to_sec(), 3),
            "state": self.state,
            "completed_length": round(self.completed_path_length, 3),
            "robot": point_data(current.pose.position) if current is not None else None,
            "actual_path": [point_data(point) for point in self.actual_trajectory],
            "planned_curve": [point_data(point) for point in self.line_curve_points],
            "raw_line": [point_data(point) for point in self.line_raw_points],
            "pending_markers": [],
            "handled_markers": [],
            "active_marker": None,
            "counts": {"yellow": 0, "black": 0},
            "endpoint_candidate": (
                point_data(self.endpoint_candidate.pose.position)
                if self.endpoint_candidate is not None else None
            ),
        }
        encoded = json.dumps(payload, separators=(",", ":"))
        self.trajectory_pub.publish(String(data=encoded))
        if self.trajectory_web is not None:
            self.trajectory_web.update(encoded)
        self.last_trajectory_publish_time = now

    def finish(self):
        current = self.get_current_pose()
        if current is not None:
            yaw = self.last_line_yaw if self.last_line_yaw is not None else (
                yaw_from_quaternion(current.pose.orientation)
            )
            self.publish_dprov(self.make_pose(
                current.pose.position.x, current.pose.position.y, yaw
            ))
        self.finished_pub.publish(String(data="%s finished" % NODE_NAME))
        rospy.loginfo("%s: finished line-only test", NODE_NAME)
        rospy.signal_shutdown("%s complete" % NODE_NAME)

    def run(self):
        while not rospy.is_shutdown():
            if not self.initialize_start_pose():
                self.rate.sleep()
                continue

            self.publish_trajectory_status()

            if self.state == "WAIT_READY":
                hold = self.make_pose(
                    self.start_pose.pose.position.x,
                    self.start_pose.pose.position.y,
                    yaw_from_quaternion(self.start_pose.pose.orientation),
                )
                self.publish_dprov(hold)
                rospy.loginfo_throttle(
                    1.0,
                    "%s: startup hold elapsed=%.1f/%.1fs camera_ready=%s",
                    NODE_NAME,
                    self.state_elapsed(),
                    self.startup_hold_seconds,
                    self.camera_ready(),
                )
                if self.state_elapsed() >= self.startup_hold_seconds and self.camera_ready():
                    self.begin_settle(
                        self.make_pose(
                            self.start_pose.pose.position.x,
                            self.start_pose.pose.position.y,
                            self.initial_search_yaw,
                        ),
                        "SEARCH",
                        "startup_heading_alignment",
                    )
            elif self.state == "SETTLE":
                self.run_settle()
            elif self.state == "SEARCH":
                if self.line_curve_ready() and self.blind_follow_allowed():
                    self.verifying_line_end = False
                    current = self.get_current_pose()
                    if current is not None:
                        self.begin_settle(
                            self.make_pose(
                                current.pose.position.x,
                                current.pose.position.y,
                                self.last_line_yaw,
                            ),
                            "FOLLOW",
                            "line_acquired",
                        )
                else:
                    self.search_line()
            elif self.state == "FOLLOW":
                if self.line_curve_ready() and self.blind_follow_allowed():
                    self.follow_line()
                else:
                    current = self.get_current_pose()
                    if current is not None:
                        yaw = self.last_line_yaw if self.last_line_yaw is not None else (
                            yaw_from_quaternion(current.pose.orientation)
                        )
                        self.begin_settle(
                            self.make_pose(current.pose.position.x, current.pose.position.y, yaw),
                            "EVALUATE_LOSS",
                            "line_temporarily_lost",
                        )
            elif self.state == "EVALUATE_LOSS":
                if self.line_curve_ready() and self.blind_follow_allowed():
                    self.set_state("FOLLOW")
                elif (
                    self.endpoint_candidate is not None
                    and self.last_line_time is not None
                    and (rospy.Time.now() - self.last_line_time).to_sec()
                    >= self.line_lost_timeout
                ):
                    rospy.loginfo(
                        "%s: stable endpoint found; scan +/-30deg before finishing",
                        NODE_NAME,
                    )
                    self.verifying_line_end = True
                    self.reset_search_cycle()
                    self.set_state("SEARCH")
                else:
                    rospy.loginfo(
                        "%s: no valid endpoint evidence; resume scan search",
                        NODE_NAME,
                    )
                    self.reset_search_cycle()
                    self.set_state("SEARCH")
            elif self.state == "VERIFY_ENDPOINT":
                if self.line_curve_ready() and self.blind_follow_allowed():
                    self.verifying_line_end = False
                    self.set_state("FOLLOW")
                elif (
                    self.endpoint_candidate is not None
                    and self.last_line_time is not None
                    and (rospy.Time.now() - self.last_line_time).to_sec()
                    >= self.line_lost_timeout
                ):
                    self.finish()
                else:
                    self.verifying_line_end = False
                    self.reset_search_cycle()
                    self.set_state("SEARCH")
            elif self.state == "FINISH":
                self.finish()
            elif not self.line_started:
                self.search_line()

            self.rate.sleep()


def main():
    rospy.init_node(NODE_NAME)
    Task1LineFollowTest().run()


if __name__ == "__main__":
    main()
