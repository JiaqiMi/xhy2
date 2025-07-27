#! /home/xhy/xhy_env36/bin/python
"""
名称：map_initer.py
功能：创建map坐标系原点
作者：buyegaid
监听：/debug_auv_data(AUVData.msg)
发布：/world_origin(NavSatFix.msg)
记录：
2025.7.19 10:50
    第一版完成
"""

import rospy
from auv_control.msg import AUVData
from sensor_msgs.msg import NavSatFix
init_lat_list = [] # 初始纬度列表
init_lon_list = [] # 初始经度列表
init_dep_list = [] # 初始深度列表

class map_initer:
    def __init__(self):
        rospy.Subscriber('/debug_auv_data',AUVData, self.debug_callback)
        self.pub = rospy.Publisher('/world_origin',NavSatFix,queue_size=10)
        self.done = False # 记录是否已经发布了世界坐标系原点 
        rospy.loginfo("map_initer: 已启动")
    
    def debug_callback(self, msg:AUVData):
        """监听AUVData,取前50个惯导有效数据计算世界坐标系原点"""
        # 只看pose和sensor中的ahrs有效
        if not self.done:
            if msg.sensor.sensor_valid&0x40!=0:
                rospy.loginfo_throttle(2,"map_initer:  惯导数据有效")
                if len(init_lon_list) < 50: # 保存前50帧作为世界坐标系原点
                    init_lat_list.append(msg.pose.latitude)
                    init_lon_list.append(msg.pose.longitude)
                    init_dep_list.append(msg.pose.depth)
                if len(init_lon_list) == 50:
                    init_lat = sum(init_lat_list) / len(init_lat_list)
                    init_lon = sum(init_lon_list) / len(init_lon_list)
                    init_alt = sum(init_dep_list) / len(init_dep_list)
                    msg = NavSatFix()
                    msg.altitude = init_alt
                    msg.latitude = init_lat
                    msg.longitude = init_lon
                    self.pub.publish(msg)
                    self.done = True
                    rospy.loginfo("map_initer: 世界坐标系原点已发布: lat:{}, lon:{}, alt:{}".format(init_lat, init_lon, init_alt)) # 纬经深
    
    def run(self):
        """主循环"""
        if not rospy.is_shutdown():
            rospy.spin()


if __name__ == "__main__":
    try:
        rospy.init_node('map_initer_node')
        mi =  map_initer()
        mi.run()
    except rospy.ROSInterruptException:
        pass
