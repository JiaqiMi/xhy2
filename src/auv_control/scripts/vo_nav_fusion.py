#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vo_nav_fusion.py

功能：
1. 订阅 /nav (auv_control/NavData)，读取导航设备高频输出
2. 订阅 /orb_slam3/body_odom（默认），读取视觉里程计载体系位姿
3. 在 nav_ned 世界系下做弱耦合融合：
   - 高频：用 nav 的姿态/速度/深度进行预测
   - 低频：用 VO 的位置和 yaw 进行校正
4. 发布：
   - /fusion/pose  (geometry_msgs/PoseStamped)
   - /fusion/odom  (nav_msgs/Odometry)

坐标系约定：
- 世界系 nav_ned：
    x = North
    y = East
    z = Down
- 机体系 base_link：
    x = Forward
    y = Right
    z = Down
- 默认 VO 输入直接使用 /orb_slam3/body_odom，因此不再重复做 T_cb 变换

说明：
- 这是 V1 版本的弱耦合融合，不是严格误差状态 EKF
- 先跑通链路，再升级到“原始 IMU 预测 + VO 更新”的完整版
"""

import math
import threading
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from auv_control.msg import NavData


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def rot_x(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, ca, -sa],
        [0.0, sa, ca]
    ], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [ca, 0.0, sa],
        [0.0, 1.0, 0.0],
        [-sa, 0.0, ca]
    ], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [ca, -sa, 0.0],
        [sa,  ca, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def rpy_to_rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    采用 ZYX 顺序：
        R = Rz(yaw) * Ry(pitch) * Rx(roll)
    输出为 body->nav_ned 的旋转矩阵 R_nb
    """
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rot_to_quat_xyzw(R: np.ndarray):
    """
    旋转矩阵 -> 四元数 (x,y,z,w)
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    q /= n
    return q[0], q[1], q[2], q[3]


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q /= n
    x, y, z, w = q

    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float64)


def yaw_from_rot(R: np.ndarray) -> float:
    """
    从 R_nb 中取 yaw（nav_ned 下）
    """
    return math.atan2(R[1, 0], R[0, 0])


def pose_msg_to_T(pose) -> np.ndarray:
    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w
    R = quat_xyzw_to_rot(qx, qy, qz, qw)
    t = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


class VONavFusion:
    def __init__(self):
        self.lock = threading.Lock()

        # ----------------------------
        # Topics / frames
        # ----------------------------
        self.nav_topic = rospy.get_param("~nav_topic", "/nav")
        self.vo_topic = rospy.get_param("~vo_topic", "/orb_slam3/body_odom")
        self.use_body_odom = rospy.get_param("~use_body_odom", True)

        self.world_frame = rospy.get_param("~world_frame", "nav_ned")
        self.body_frame = rospy.get_param("~body_frame", "base_link")

        # ----------------------------
        # Gains / switches
        # ----------------------------
        self.kp_pos_xy = rospy.get_param("~kp_pos_xy", 0.20)
        self.kp_pos_z = rospy.get_param("~kp_pos_z", 0.10)
        self.kp_yaw = rospy.get_param("~kp_yaw", 0.20)

        self.use_depth_as_z = rospy.get_param("~use_depth_as_z", True)
        self.max_vo_age = rospy.get_param("~max_vo_age", 1.5)   # s
        self.max_dt = rospy.get_param("~max_nav_dt", 0.2)       # 防止断流后大步长积分
        self.publish_rate = rospy.get_param("~publish_rate", 50.0)

        # ----------------------------
        # Nav angle convention tuning
        # 如后续发现姿态方向不一致，可直接调参数，不改代码
        # ----------------------------
        self.nav_roll_sign = rospy.get_param("~nav_roll_sign", 1.0)
        self.nav_pitch_sign = rospy.get_param("~nav_pitch_sign", 1.0)
        self.nav_heading_sign = rospy.get_param("~nav_heading_sign", 1.0)
        self.nav_heading_offset_deg = rospy.get_param("~nav_heading_offset_deg", 0.0)

        # ----------------------------
        # If using camera_odom instead of body_odom, keep T_cb interface
        # T_cb: body -> camera
        # ----------------------------
        self.T_cb = np.eye(4, dtype=np.float64)
        if not self.use_body_odom:
            tx = rospy.get_param("~Tcb_tx", 0.0)
            ty = rospy.get_param("~Tcb_ty", 0.0)
            tz = rospy.get_param("~Tcb_tz", 0.0)
            qx = rospy.get_param("~Tcb_qx", 0.0)
            qy = rospy.get_param("~Tcb_qy", 0.0)
            qz = rospy.get_param("~Tcb_qz", 0.0)
            qw = rospy.get_param("~Tcb_qw", 1.0)
            R_cb = quat_xyzw_to_rot(qx, qy, qz, qw)
            self.T_cb = make_T(R_cb, np.array([tx, ty, tz], dtype=np.float64))

        # ----------------------------
        # State
        # ----------------------------
        self.nav_initialized = False
        self.alignment_initialized = False

        self.last_nav_stamp: Optional[float] = None
        self.latest_vo_stamp: Optional[float] = None

        self.p_fused_n = np.zeros(3, dtype=np.float64)   # [N, E, D]
        self.v_fused_n = np.zeros(3, dtype=np.float64)   # [vn, ve, vd]
        self.roll_nav = 0.0
        self.pitch_nav = 0.0
        self.yaw_nav = 0.0
        self.yaw_bias = 0.0

        self.T_nm = np.eye(4, dtype=np.float64)          # map -> nav_ned

        # optional cache
        self.last_nav_msg = None
        self.last_vo_msg = None

        # ----------------------------
        # Publishers
        # ----------------------------
        self.pub_pose = rospy.Publisher("/fusion/pose", PoseStamped, queue_size=10)
        self.pub_odom = rospy.Publisher("/fusion/odom", Odometry, queue_size=10)

        # ----------------------------
        # Subscribers
        # ----------------------------
        rospy.Subscriber(self.nav_topic, NavData, self.nav_callback, queue_size=50)
        rospy.Subscriber(self.vo_topic, Odometry, self.vo_callback, queue_size=10)

        # Timer publish
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.publish_rate), self.timer_publish)

        rospy.loginfo("vo_nav_fusion: started")
        rospy.loginfo("  nav_topic      = %s", self.nav_topic)
        rospy.loginfo("  vo_topic       = %s", self.vo_topic)
        rospy.loginfo("  use_body_odom  = %s", str(self.use_body_odom))
        rospy.loginfo("  world_frame    = %s", self.world_frame)
        rospy.loginfo("  body_frame     = %s", self.body_frame)

    # ============================================================
    # Utility
    # ============================================================
    def current_fused_yaw(self) -> float:
        return wrap_pi(self.yaw_nav + self.yaw_bias)

    def current_fused_R_nb(self) -> np.ndarray:
        return rpy_to_rot_zyx(self.roll_nav, self.pitch_nav, self.current_fused_yaw())

    def current_fused_T_nb(self) -> np.ndarray:
        return make_T(self.current_fused_R_nb(), self.p_fused_n)

    def nav_rpy_from_msg(self, msg: NavData):
        roll = self.nav_roll_sign * math.radians(msg.roll)
        pitch = self.nav_pitch_sign * math.radians(msg.pitch)
        yaw = self.nav_heading_sign * math.radians(msg.heading) + math.radians(self.nav_heading_offset_deg)
        yaw = wrap_pi(yaw)
        return roll, pitch, yaw

    def vo_msg_to_T_mb(self, msg: Odometry) -> np.ndarray:
        """
        将 VO 消息转成 map->body 的位姿
        - 若 use_body_odom=True，msg 本身就是 map->body
        - 否则，认为 msg 是 map->camera，需要再乘 T_cb
        """
        T_mx = pose_msg_to_T(msg.pose.pose)
        if self.use_body_odom:
            return T_mx
        else:
            return T_mx @ self.T_cb

    # ============================================================
    # Callbacks
    # ============================================================
    def nav_callback(self, msg: NavData):
        with self.lock:
            stamp = msg.header.stamp.to_sec() if msg.header.stamp != rospy.Time() else rospy.Time.now().to_sec()

            roll, pitch, yaw = self.nav_rpy_from_msg(msg)
            v_nav = np.array([msg.vn, msg.ve, msg.vd], dtype=np.float64)

            if not self.nav_initialized:
                self.nav_initialized = True
                self.last_nav_stamp = stamp

                self.roll_nav = roll
                self.pitch_nav = pitch
                self.yaw_nav = yaw
                self.yaw_bias = 0.0

                self.v_fused_n = v_nav.copy()
                self.p_fused_n[:] = 0.0
                if self.use_depth_as_z:
                    self.p_fused_n[2] = msg.depth

                self.last_nav_msg = msg
                return

            dt = stamp - self.last_nav_stamp if self.last_nav_stamp is not None else 0.0
            dt = clamp(dt, 0.0, self.max_dt)
            self.last_nav_stamp = stamp

            # 1) 先用上一时刻速度做位置传播
            self.p_fused_n = self.p_fused_n + self.v_fused_n * dt

            # 2) 再写入当前 nav 解算量
            self.v_fused_n = v_nav.copy()
            self.roll_nav = roll
            self.pitch_nav = pitch
            self.yaw_nav = yaw

            # 3) 深度约束
            if self.use_depth_as_z:
                self.p_fused_n[2] = msg.depth

            self.last_nav_msg = msg

    def vo_callback(self, msg: Odometry):
        with self.lock:
            if not self.nav_initialized:
                return

            T_mb = self.vo_msg_to_T_mb(msg)
            T_nb_pred = self.current_fused_T_nb()

            # 还没完成 map->nav_ned 对齐：第一次有效 VO 时初始化
            if not self.alignment_initialized:
                self.T_nm = T_nb_pred @ inv_T(T_mb)
                self.alignment_initialized = True
                self.latest_vo_stamp = msg.header.stamp.to_sec()
                self.last_vo_msg = msg
                rospy.loginfo("vo_nav_fusion: map->nav_ned alignment initialized.")
                return

            # map -> body 先变到 nav_ned -> body
            T_nb_vo = self.T_nm @ T_mb
            p_vo = T_nb_vo[:3, 3]
            yaw_vo = yaw_from_rot(T_nb_vo[:3, :3])

            # 当前预测状态
            p_pred = self.p_fused_n.copy()
            yaw_pred = self.current_fused_yaw()

            # 位置校正
            err_p = p_vo - p_pred
            self.p_fused_n[0] += self.kp_pos_xy * err_p[0]
            self.p_fused_n[1] += self.kp_pos_xy * err_p[1]
            self.p_fused_n[2] += self.kp_pos_z * err_p[2]

            # yaw 校正：只修 heading 漂移
            err_yaw = wrap_pi(yaw_vo - yaw_pred)
            self.yaw_bias = wrap_pi(self.yaw_bias + self.kp_yaw * err_yaw)

            self.latest_vo_stamp = msg.header.stamp.to_sec()
            self.last_vo_msg = msg

    # ============================================================
    # Publish
    # ============================================================
    def build_fused_pose_msg(self, stamp: rospy.Time) -> PoseStamped:
        R_nb = self.current_fused_R_nb()
        qx, qy, qz, qw = rot_to_quat_xyzw(R_nb)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.world_frame
        pose_msg.pose.position.x = float(self.p_fused_n[0])
        pose_msg.pose.position.y = float(self.p_fused_n[1])
        pose_msg.pose.position.z = float(self.p_fused_n[2])
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw
        return pose_msg

    def build_fused_odom_msg(self, pose_msg: PoseStamped) -> Odometry:
        odom_msg = Odometry()
        odom_msg.header = pose_msg.header
        odom_msg.child_frame_id = self.body_frame
        odom_msg.pose.pose = pose_msg.pose

        # twist 默认发布 body 系速度
        R_nb = self.current_fused_R_nb()
        v_body = R_nb.T @ self.v_fused_n

        odom_msg.twist.twist.linear.x = float(v_body[0])
        odom_msg.twist.twist.linear.y = float(v_body[1])
        odom_msg.twist.twist.linear.z = float(v_body[2])

        if self.last_nav_msg is not None:
            # 原始角速度直接带上，单位已在 nav_driver.py 中转成 deg/s
            # 这里转换成 rad/s 再发
            odom_msg.twist.twist.angular.x = math.radians(self.last_nav_msg.gyro_x)
            odom_msg.twist.twist.angular.y = math.radians(self.last_nav_msg.gyro_y)
            odom_msg.twist.twist.angular.z = math.radians(self.last_nav_msg.gyro_z)

        # 给一个简单占位协方差，后面可以再细化
        for i in range(36):
            odom_msg.pose.covariance[i] = 0.0
            odom_msg.twist.covariance[i] = 0.0

        # 位置协方差（可按需要再调）
        odom_msg.pose.covariance[0] = 0.5 * 0.5
        odom_msg.pose.covariance[7] = 0.5 * 0.5
        odom_msg.pose.covariance[14] = 0.3 * 0.3

        # 姿态协方差
        odom_msg.pose.covariance[21] = math.radians(5.0) ** 2
        odom_msg.pose.covariance[28] = math.radians(5.0) ** 2
        odom_msg.pose.covariance[35] = math.radians(10.0) ** 2

        return odom_msg

    def timer_publish(self, _event):
        with self.lock:
            if not self.nav_initialized:
                return

            now = rospy.Time.now()

            # 如果 VO 太久没更新，只继续发布 nav 预测结果，不额外报错
            if self.alignment_initialized and self.latest_vo_stamp is not None:
                vo_age = now.to_sec() - self.latest_vo_stamp
                if vo_age > self.max_vo_age:
                    rospy.logwarn_throttle(2.0, "vo_nav_fusion: VO stale for %.3f s", vo_age)

            pose_msg = self.build_fused_pose_msg(now)
            odom_msg = self.build_fused_odom_msg(pose_msg)

            self.pub_pose.publish(pose_msg)
            self.pub_odom.publish(odom_msg)


def main():
    rospy.init_node("vo_nav_fusion")
    VONavFusion()
    rospy.spin()


if __name__ == "__main__":
    main()
