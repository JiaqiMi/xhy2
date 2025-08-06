import cv2
import numpy as np

# -------------------------------
# 相机内参（你需要替换为你的相机标定结果）
# -------------------------------
# K = np.array([[800, 0, 320],
#               [0, 800, 240],
#               [0,   0,   1]], dtype=np.float64)
K = np.array([[519.1519, 0, 319.174292],
              [0, 519.712551, 277.976296],
              [0, 0, 1]], dtype=np.float64)

# dist_coeffs = np.zeros((5, 1))  # 无畸变或略微畸变
dist_coeffs = np.array([[-0.019985, 0.106889, 0.000070, 0.002679, 0.000000]], dtype=np.float64).T


# -------------------------------
# 载入图像
# -------------------------------
image = cv2.imread('./local_test/aruco/data/my_photo-8.jpg')
if image is None:
    raise FileNotFoundError("图像未找到，请确认路径正确")

# -------------------------------
# 设置 ArUco 字典
# -------------------------------
# The code snippet you provided is setting up the ArUco marker detection parameters using OpenCV's
# ArUco module. Here's a breakdown of what each line is doing:
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# parameters = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

used_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# -------------------------------
# 检测二维码
# -------------------------------
# corners, ids, rejected = detector.detectMarkers(image)
corners, ids, rejected = cv2.aruco.detectMarkers(image, used_dict)

# -------------------------------
# 估计姿态（假设每个标记边长为 0.05 米）
# -------------------------------
marker_length = 0.035  # 5cm
if ids is not None:
    for i, corner in enumerate(corners):
        retval, rvec, tvec = cv2.solvePnP(
            objectPoints=np.array([
                [-0.5, 0.5, 0],
                [ 0.5, 0.5, 0],
                [ 0.5,-0.5, 0],
                [-0.5,-0.5, 0]
            ]) * marker_length,  # 世界坐标
            imagePoints=corner[0],
            cameraMatrix=K,
            distCoeffs=dist_coeffs
        )

        # 画坐标轴
        # cv2.aruco.drawAxis(image, K, dist_coeffs, rvec, tvec, marker_length * 0.5)
        
        # 定义坐标轴的 3D 点（以标记中心为原点）
        axis = np.float32([
            [0, 0, 0],  # 原点
            [marker_length, 0, 0],  # X 轴
            [0, marker_length, 0],  # Y 轴
            [0, 0, -marker_length]  # Z 轴
        ]).reshape(-1, 3)

        # 将 3D 点投影到 2D 图像平面
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, K, dist_coeffs)
        
        # 提取起点和终点坐标
        origin = tuple(map(int, imgpts[0].ravel()))
        x_axis = tuple(map(int, imgpts[1].ravel()))
        y_axis = tuple(map(int, imgpts[2].ravel()))
        z_axis = tuple(map(int, imgpts[3].ravel()))

        # 绘制坐标轴：从原点出发
        image = cv2.line(image, origin, x_axis, (0, 0, 255), 1)   # X 轴（红）
        image = cv2.line(image, origin, y_axis, (0, 255, 0), 1)   # Y 轴（绿）
        image = cv2.line(image, origin, z_axis, (255, 0, 0), 1)   # Z 轴（蓝）

        # 显示结果
        print(f"ID={ids[i][0]} 平移向量 tvec: {tvec.ravel()} (单位：米)")
        R, _ = cv2.Rodrigues(rvec)
        print(f"ID={ids[i][0]} 旋转矩阵 R:\n{R}\n")

    # 画出标记边框
    # cv2.aruco.drawDetectedMarkers(image, corners, ids)
else:
    print("未检测到任何 ArUco 标记")

# -------------------------------
# 显示图像
# -------------------------------
cv2.imshow("Pose Estimation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()