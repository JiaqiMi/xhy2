#!/usr/bin/env python3
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

if __name__ == '__main__':
    rospy.init_node('my_tf_listener')
    
    # 创建缓存区和监听器
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    print("正在监听 map -> base_link 的变换...")
    rate = rospy.Rate(10)  # 10Hz
    
    while not rospy.is_shutdown():
        try:
            # 获取最新的变换 (rospy.Time(0) 表示最新时刻)
            trans: TransformStamped = tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
            
            # 格式化打印
            print("\n[时间戳: {:.3f}]".format(trans.header.stamp.to_sec()))
            print("  平移: x={:.3f}, y={:.3f}, z={:.3f}".format(
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ))
            print("  旋转(四元数): x={:.3f}, y={:.3f}, z={:.3f}, w={:.3f}".format(
                trans.transform.rotation.x,
                trans.transform.rotation.y,
                trans.transform.rotation.z,
                trans.transform.rotation.w
            ))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # 如果 TF 还没发布或者 lookup 失败，打印警告
            print("等待 TF 数据: {}".format(e))
        
        rate.sleep()