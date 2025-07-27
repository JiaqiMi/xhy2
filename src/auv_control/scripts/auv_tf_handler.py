#! /home/xhy/xhy_env36/bin/python
"""
名称：auv_tf_handler.py
功能：完成机器人坐标到世界坐标的转换
作者：buyegaid
监听：/debug_auv_data (AUVData.msg)
      /target (PoseStamped.msg)
      /world_origin (NavSatFix.msg)
发布：/auv_control (AUVPose.msg)
      /tf (from base_link to map)
记录：
2025.7.19 10:56
    第一版完成
2025.7.19 15:21
    控制指令改为直接发布AUVPose消息，不再控制舵机和LED灯
"""


# from geographic_msgs.msg import GeoPoint,GeoPose # 需要安装 sudo apt install ros-noetic-geodesy
"""
float64 latitude   # 纬度(degrees)
float64 longitude  # 经度(degrees)
float64 altitude   # 高度(meters)
"""
import rospy
import tf2_ros
import tf
from sensor_msgs.msg import NavSatFix
from auv_control.msg import AUVData,AUVPose
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped # 目标点，目标位姿

class AUV_tfhandler:
    """
    1. 订阅/debug_auv_data，获取机器人在世界坐标系下的位姿
    2. 订阅/target，获取机器人坐标系下的目标位置
    3. 将目标位置转换成世界坐标系下的位置，再转换成经纬度发送给debug_driver
    4. 发布/auv_control消息，包含目标经纬度和深度
    5. 发布TF变换，将世界坐标系下的base_link变换
    """
    def __init__(self):
        # rospy.init_node('auv_pose_control_publisher')
        origin = rospy.wait_for_message('/world_origin', NavSatFix)
        self.wfm = WorldFrameManager(origin.latitude, origin.longitude, origin.altitude)
        rospy.loginfo(f"auv_tfhandler: 世界坐标系初始化完成{origin.latitude, origin.longitude, origin.altitude}")
        # TF相关
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        rospy.Subscriber('/debug_auv_data', AUVData, self.debug_callback) # 订阅imu数据
        rospy.Subscriber('/target', PoseStamped, self.target_callback) # 订阅目标点数据
        self.current_pose = None
        self.current_yaw = 0.0  # 记录当前yaw角
        self.control_pub = rospy.Publisher('/auv_control', AUVPose, queue_size=10)
        self.Rate = rospy.Rate(100)
        # rospy.loginfo("auv_tfhandler:世界坐标系初始化完成")

    def debug_callback(self, msg):
        """将AUV的位姿转换为世界坐标系下的NED坐标，并发布TF变换"""
        # 转换为NED坐标
        n, e, d = self.wfm.lld_to_ned(msg.pose.latitude, msg.pose.longitude, msg.pose.depth) # 深度直接用正值
        if self.current_pose is None:
            self.current_pose = [n, e, d, 0, 0, 0, 1]  # n,e,d + 初始四元数(无旋转)
        else:
            self.current_pose[0:3] = [n, e, d]
        # 记录当前yaw角（单位：度）
        self.current_yaw = msg.pose.yaw
        if self.current_pose is not None:
            # 欧拉角使用北东地坐标系,且使用弧度制
            self.current_pose[3:7] = tf.transformations.quaternion_from_euler(
                np.radians(msg.pose.roll), 
                np.radians(msg.pose.pitch), 
                np.radians(msg.pose.yaw)
            )
            self.publish_tf()
            rospy.loginfo_throttle(10, f"auv_tfhandler: tf已发布")

    def target_callback(self, msg):
        """将目标点(PoseStamped)转换为经纬度和欧拉角，并发送control消息"""
        # 目标点已经是map坐标系(NED)下的点，直接转换为LLD
        n = msg.pose.position.x
        e = msg.pose.position.y
        d = msg.pose.position.z
        # NED转LLD
        lat, lon, depth = self.wfm.ned_to_lld(n, e, d)
        # 四元数转欧拉角
        q = msg.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
        # 发布Control消息
        ctrl_msg = AUVPose()
        ctrl_msg.latitude = lat
        ctrl_msg.longitude = lon
        ctrl_msg.depth = depth  # NED下直接赋值
        ctrl_msg.roll = np.degrees(roll)
        ctrl_msg.pitch = np.degrees(pitch)
        ctrl_msg.yaw = np.degrees(yaw)
        self.control_pub.publish(ctrl_msg)
        # rospy.loginfo_(f"auv_tfhandler: control 已发布lat:{lat},lon:{lon},dep:{depth},rpy:{ctrl_msg.roll:.2f},{ctrl_msg.pitch:.2f},{ctrl_msg.yaw:.2f}")

    def publish_tf(self):
        """发布map到base_link的TF（NED）"""
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.current_pose[0]  # n
        t.transform.translation.y = self.current_pose[1]  # e
        t.transform.translation.z = self.current_pose[2]  # d
        t.transform.rotation.x = self.current_pose[3]
        t.transform.rotation.y = self.current_pose[4]
        t.transform.rotation.z = self.current_pose[5]
        t.transform.rotation.w = self.current_pose[6]
        self.tf_broadcaster.sendTransform(t)
    
    def run(self):
        while not rospy.is_shutdown():
            self.Rate.sleep()

class WorldFrameManager:
    """
    世界坐标系定义
    1.保存世界坐标系原点
    2.完成tf计算和世界坐标系到经纬度的转换
    """
    def __init__(self, init_lat:float, init_lon:float, init_depth:float):
        # 保存初始纬度、经度和深度(参考点)
        # init_lat, init_lon 单位: 度(degree)
        # init_depth 单位: 米(m)
        self.init_lat = init_lat
        self.init_lon = init_lon
        self.init_depth = init_depth
        
        # WGS84椭球参数
        self.a = 6378137.0  # 半长轴(m)
        self.f = 1/298.257223563  # 扁率
        self.e_sq = self.f * (2 - self.f)  # 第一偏心率的平方

    def lld_to_ned(self, lat, lon, depth):
        """
        将经纬深(LLD)坐标转换为NED坐标系下的坐标(相对于初始点)
        输入:
            lat, lon 单位: 度(degree)
            depth 单位: 米(m)
        输出:
            (north, east, down) 单位: 米(m)
        """
        # 1. 将当前LLD转换为ECEF(地心地固坐标系)
        x, y, z = self.lld_to_ecef(lat, lon, depth)
        
        # 2. 将初始LLD转换为ECEF
        x0, y0, z0 = self.lld_to_ecef(self.init_lat, self.init_lon, self.init_depth)
        
        # 3. 计算NED坐标
        return self.ecef_to_ned(x, y, z, x0, y0, z0)
    
    def lld_to_ecef(self, lat, lon, depth):
        """
        LLd(经纬高)转ECEF(地心地固坐标系)
        输入:
            lat, lon 单位: 度(degree)
            depth 单位: 米(m)
        输出:
            x, y, z 单位: 米(m)
        """
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        N = self.a / np.sqrt(1 - self.e_sq * np.sin(lat_rad)**2)
        
        x = (N - depth) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (N - depth) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (N * (1 - self.e_sq) - depth) * np.sin(lat_rad)
        
        return x, y, z
    
    def ecef_to_ned(self, x, y, z, x0, y0, z0):
        """
        ECEF(地心地固)转NED(北东地)
        输入:
            x, y, z, x0, y0, z0 单位: 米(m)
        输出:
            n, e, d 单位: 米(m)
        """
        dx = x - x0
        dy = y - y0
        dz = z - z0

        lat0_rad = np.radians(self.init_lat)
        lon0_rad = np.radians(self.init_lon)

        # 标准ECEF到NED旋转矩阵
        R = np.array([
            [-np.sin(lat0_rad)*np.cos(lon0_rad), -np.sin(lat0_rad)*np.sin(lon0_rad),  np.cos(lat0_rad)],
            [-np.sin(lon0_rad),                   np.cos(lon0_rad),                  0],
            [-np.cos(lat0_rad)*np.cos(lon0_rad), -np.cos(lat0_rad)*np.sin(lon0_rad), -np.sin(lat0_rad)]
        ])
        ned = R @ np.array([dx, dy, dz])
        return ned[0], ned[1], ned[2]

    def ned_to_lld(self, n, e, d):
        """
        NED(北东地)坐标转LLD(纬经深)
        输入:
            n, e, d 单位: 米(m)
        输出:
            lat, lon 单位: 度(degree)
            depth 单位: 米(m)
        """
        # 1. 将原点LLD转换为ECEF
        lat0_rad = np.radians(self.init_lat)
        lon0_rad = np.radians(self.init_lon)
        depth0 = self.init_depth
        N0 = self.a / np.sqrt(1 - self.e_sq * np.sin(lat0_rad)**2)
        x0 = (N0 - depth0) * np.cos(lat0_rad) * np.cos(lon0_rad)
        y0 = (N0 - depth0) * np.cos(lat0_rad) * np.sin(lon0_rad)
        z0 = (N0 * (1 - self.e_sq) - depth0) * np.sin(lat0_rad)
        # 2. NED到ECEF的旋转矩阵（标准）
        R = np.array([
            [-np.sin(lat0_rad)*np.cos(lon0_rad), -np.sin(lat0_rad)*np.sin(lon0_rad),  np.cos(lat0_rad)],
            [-np.sin(lon0_rad),                   np.cos(lon0_rad),                  0],
            [-np.cos(lat0_rad)*np.cos(lon0_rad), -np.cos(lat0_rad)*np.sin(lon0_rad), -np.sin(lat0_rad)]
        ])
        # 3. 计算目标点ECEF坐标
        ned = np.array([n, e, d])
        ecef = R.T @ ned + np.array([x0, y0, z0])
        x, y, z = ecef
        # 4. ECEF转LLD
        lld = self.ecef_to_lld(x, y, z)
        # 返回纬度、经度(度)，深度(米)
        return np.degrees(lld[0]), np.degrees(lld[1]), lld[2]

    def ecef_to_lld(self, x, y, z):
        """
        ECEF(地心地固)坐标转LLD(纬经深)
        输入:
            x, y, z 单位: 米(m)
        输出:
            lat_rad, lon_rad 单位: 弧度(radian)
            depth 单位: 米(m)
        """
        # 经度
        lon_rad = np.arctan2(y, x)
        
        # 初始纬度估计
        p = np.sqrt(x**2 + y**2)
        lat_rad = np.arctan2(z, p * (1 - self.e_sq))
        
        # 迭代计算精确纬度
        for _ in range(10):
            N = self.a / np.sqrt(1 - self.e_sq * np.sin(lat_rad)**2)
            depth = N - p / np.cos(lat_rad)
            lat_new = np.arctan2(z, p * (1 - self.e_sq * N / (N - depth)))
            if abs(lat_new - lat_rad) < 1e-12:
                break
            lat_rad = lat_new
        
        # 最终高度
        N = self.a / np.sqrt(1 - self.e_sq * np.sin(lat_rad)**2)
        depth = N - p / np.cos(lat_rad)

        return lat_rad, lon_rad, depth


if __name__ == "__main__":
    try:
        rospy.init_node('auv_tf_handler_node')
        pub = AUV_tfhandler()
        pub.run()
    except rospy.ROSInterruptException:
        pass
