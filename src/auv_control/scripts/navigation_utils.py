import math

# class auv: # 用于存储auv的状态信息
#     def __init__(self):
#         self.east = 0.0            # 东向坐标，单位：mm
#         self.north = 0.0            # 北向坐标，单位：mm
#         self.up = 0.0            # 深度坐标，单位：mm
#         self.yaw = 0.0          # 航向角，单位：弧度
#         self.pitch = 0.0        # 俯仰角，单位：弧度
#         self.roll = 0.0         # 横滚角，单位：弧度
#         self.velocity =0.0           # 速度，单位：mm/s
#         self.latitude = 0.0          # 纬度，单位：度
#         self.longitude = 0.0          # 经度，单位：度
#         self.depth = 0.0        # 深度，单位：m
#         self.altitude = 0.0          # 高度，单位：m

def local_point_to_global(auv_pose, local_point):
    """
    将AUV坐标系下的点转换为地理坐标（经度、纬度、深度）和航向（视角）。
    :param auv_pose: auv对象，包含AUV的位姿信息（latitude, longitude, depth, yaw, pitch, roll），单位分别为度、度、米、度、度、度
    :param local_point: tuple/list，AUV坐标系下的点 (x, y, z)，单位为米（前右下）
    :return: (latitude, longitude, depth, heading)
    """
    # 地球半径（WGS84椭球体平均半径，单位：米）
    Rearth = 6378137.0

    # 提取AUV位姿
    lat0 = math.radians(auv_pose.latitude)
    lon0 = math.radians(auv_pose.longitude)
    depth0 = auv_pose.depth
    # 姿态角由度转为弧度
    yaw = math.radians(auv_pose.yaw)
    pitch = math.radians(getattr(auv_pose, 'pitch', 0.0))
    roll = math.radians(getattr(auv_pose, 'roll', 0.0))

    # AUV坐标系下的点
    x, y, z = local_point  # 前右下

    # 旋转矩阵（ZYX顺序：yaw, pitch, roll）
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    # 方向余弦矩阵
    Rz = [[cy, -sy, 0],
          [sy,  cy, 0],
          [ 0,   0, 1]]
    Ry = [[cp, 0, sp],
          [ 0, 1,  0],
          [-sp, 0, cp]]
    Rx = [[1,  0,   0],
          [0, cr, -sr],
          [0, sr,  cr]]

    # 矩阵乘法
    def matmul(A, B):
        return [[sum(a*b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

    Rzy = matmul(Rz, Ry)
    R = matmul(Rzy, Rx)

    # 坐标变换
    dx = R[0][0]*x + R[0][1]*y + R[0][2]*z
    dy = R[1][0]*x + R[1][1]*y + R[1][2]*z
    dz = R[2][0]*x + R[2][1]*y + R[2][2]*z

    # 东北天坐标系下的增量
    dn = dx  # 北
    de = dy  # 东
    # du = -dz  # 下为正，天为正（未使用）

    # 经纬度换算
    dlat = dn / Rearth
    dlon = de / (Rearth * math.cos(lat0))
    lat = lat0 + dlat
    lon = lon0 + dlon
    depth = depth0 + dz  # 深度向下为正

    # 计算点的航向（视角），假设视角为AUV航向加上点在AUV坐标系下的前右方向
    heading = (yaw + math.atan2(y, x)) % (2 * math.pi)
    heading = math.degrees(heading)

    return math.degrees(lat), math.degrees(lon), depth, heading



