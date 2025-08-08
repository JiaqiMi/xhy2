#!/home/xhy/xhy_env/bin/python
# -*- coding: utf-8 -*-
import re
import tf
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion


def get_stable_depth(u, v, depth, fx, fy, cx, cy, window_size=11):
    half = window_size // 2
    h, w = depth.shape
    u, v = int(u), int(v)
    region = depth[max(v-half, 0):min(v+half+1, h), max(u-half, 0):min(u+half+1, w)]
    
    # 有效深度区域面积判断
    valid = region[np.isfinite(region) & (region > 0)]
    if valid.size < 3:
        return np.array([np.nan, np.nan, np.nan])
    
    # 添加距离过滤条件
    valid = valid[(valid >= 0.3) & (valid <= 2.5)]
    if valid.size == 0:
        return np.array([np.nan, np.nan, np.nan])
    
    Z = np.mean(valid)
    if Z <= 0:
        return np.array([np.nan, np.nan, np.nan])
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])

def load_stereo_params(txt_path):
    """
    从 new_camera.txt 中解析左右相机标定参数：
        K1, D1, R1, P1, K2, D2, R2, P2
    返回 dict:
        { 'K1':…, 'D1':…, 'R1':…, 'P1':…, 'K2':…, 'D2':…, 'R2':…, 'P2':… }
    """
    with open(txt_path, 'r') as f:
        txt = f.read()

    # 先拆分成左右两大块
    left_block  = re.search(r'\[stereo/left\](.*?)\[stereo/right\]', txt, re.S).group(1)
    right_block = re.search(r'\[stereo/right\](.*)', txt, re.S).group(1)

    def parse_matrix(block, key, rows, cols):
        # 找到 key 后面的 rows×cols 数值
        pat = key + r'\s*([\d\.\-\s]+(?:\n[\d\.\-\s]+){' + f'{rows-1}' + r'})'
        m = re.search(pat, block)
        nums = list(map(float, re.split(r'\s+', m.group(1).strip())))
        return np.array(nums, dtype=np.float64).reshape(rows, cols)

    def parse_vector(block, key, length):
        pat = key + r'\s*([\d\.\-\s]+(?:\s+[\d\.\-\s]+){' + f'{length-1}' + r'})'
        m = re.search(pat, block)
        nums = list(map(float, re.split(r'\s+', m.group(1).strip())))
        return np.array(nums, dtype=np.float64)

    out = {}
    out['K1'] = parse_matrix(left_block,  'camera matrix',        3, 3)
    out['D1'] = parse_vector(left_block,  'distortion',            5)
    out['R1'] = parse_matrix(left_block,  'rectification',         3, 3)
    out['P1'] = parse_matrix(left_block,  'projection',            3, 4)
    out['K2'] = parse_matrix(right_block, 'camera matrix',        3, 3)
    out['D2'] = parse_vector(right_block, 'distortion',            5)
    out['R2'] = parse_matrix(right_block, 'rectification',         3, 3)
    out['P2'] = parse_matrix(right_block, 'projection',            3, 4)
    return out



def compute_pose_from_quad(P1, P2, P3, P4):
    """计算四个点的中心位置和姿态"""
    center = None
    dis_thre = 2.5   # 单位米
    if (P1[2] < dis_thre) and (P2[2] < dis_thre) and (P3[2] < dis_thre) and (P4[2] < dis_thre):
        center = (P1 + P2 + P3 + P4) / 4.0
    elif (P1[2] < dis_thre) and (P3[2] < dis_thre):
        center = (P1 + P3) / 2.0
    elif (P2[2] < dis_thre) and (P4[2] < dis_thre):
        center = (P2 + P4) / 2.0
    elif (P1[2] < dis_thre) and (P2[2] < dis_thre) and (P3[2]<dis_thre):
        center = (P1 + P3) / 2.0
    elif (P1[2] < dis_thre) and (P2[2] < dis_thre) and (P4[2]<dis_thre):
        center = (P2 + P4) / 2.0
    elif (P1[2] < dis_thre) and (P3[2] < dis_thre) and (P4[2]<dis_thre):
        center = (P1 + P3) / 2.0
    elif (P2[2] < dis_thre) and (P3[2] < dis_thre) and (P4[2]<dis_thre):
        center = (P2 + P4) / 2.0
    else:
        rospy.logwarn("Invalid depth for all points, cannot compute pose.")
        return None
    
    vec1 = P2 - P1
    vec2 = P4 - P1
    z_axis = np.cross(vec1, vec2)
    z_axis /= np.linalg.norm(z_axis)
    x_axis = vec1 / np.linalg.norm(vec1)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    T = np.eye(4)
    T[:3, :3] = R
    quat = tf.transformations.quaternion_from_matrix(T)

    pose = PoseStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = "camera"
    pose.pose.position.x = center[0]
    pose.pose.position.y = center[1]
    pose.pose.position.z = center[2]
    pose.pose.orientation = Quaternion(*quat)
    return pose