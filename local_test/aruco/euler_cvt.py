# import numpy as np
# from scipy.spatial.transform import Rotation as R

# # example 
# yaw1, pitch1, roll1 = -30, 0, 0
# yaw2, pitch2, roll2 = 90, 0, 0

# # 输入：导航系下两个坐标系的欧拉角
# euler_body_nav = [yaw1, pitch1, roll1]  # 单位：度
# euler_qr_nav = [yaw2, pitch2, roll2]    # 单位：度

# # 步骤1：欧拉角 -> 旋转矩阵
# R_body_nav = R.from_euler('zyx', euler_body_nav, degrees=True).as_matrix()
# R_qr_nav = R.from_euler('zyx', euler_qr_nav, degrees=True).as_matrix()

# # 步骤2：分别提取导航系下的单位向量
# x_body_nav = R_body_nav[:, 0]  # 载体坐标系 x 轴在导航坐标系下的表示
# z_qr_nav = R_qr_nav[:, 2]      # 二维码坐标系 z 轴在导航坐标系下的表示

# # 步骤3：计算将 x_body_nav 旋转到 z_qr_nav 的旋转
# v = np.cross(x_body_nav, z_qr_nav)
# s = np.linalg.norm(v)
# c = np.dot(x_body_nav, z_qr_nav)

# if s == 0:
#     R_align = np.eye(3) if c > 0 else -np.eye(3)
# else:
#     vx = np.array([
#         [0, -v[2], v[1]],
#         [v[2], 0, -v[0]],
#         [-v[1], v[0], 0]
#     ])
#     R_align = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

# # 步骤4：旋转后的载体姿态（仍在导航坐标系下）
# R_new_body_nav = R_align @ R_body_nav

# # 输出旋转后的欧拉角
# euler_new = R.from_matrix(R_new_body_nav).as_euler('zyx', degrees=True)
# print("新的姿态欧拉角（导航系下）：", euler_new)


from scipy.spatial.transform import Rotation as R
import numpy as np


# 从导航坐标系到二维码坐标系的欧拉变换
yaw = -90 #-90
pitch = 0
roll = 90

# 示例欧拉角，单位为度
euler_nav2qr = [yaw, pitch, roll]  # ZYX 顺序
r = R.from_euler('zyx', euler_nav2qr, degrees=True)
R_nav2qr = r.as_matrix()

R_qr2nav = R_nav2qr.T

v_qr = np.array([0, 0, -1])  # 二维码坐标系的Z轴
v_nav = R_qr2nav @ v_qr     # 在导航系下的表示

# 在导航坐标系xy平面上的投影向量 K
K = v_nav.copy()
K[2] = 0

# 单位化投影向量
K_unit = K / np.linalg.norm(K)

x_axis = np.array([1, 0, 0])
cos_theta = np.dot(K_unit, x_axis)
angle_deg = np.degrees(np.arccos(cos_theta))

print(angle_deg)
