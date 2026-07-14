#! /home/xhy/xhy_env36/bin/python
"""
名称：test_debug_v2.py
功能：测试 debug_driver_v2 的三种控制模式
      发布 PoseNEDcmd（NED 坐标系）
      → auv_tf_handler 转换 → PoseLLAcmd（LLA 坐标系）
      → debug_driver_v2 → TCP → AUV
作者：BroXu
记录：
2026.7.11
    初版，支持 --mode 2/3/4 三种模式，走完整 TF 链路
2026.7.13
    上层改用包含 PoseStamped 的 PoseNEDcmd 整包消息。
用法：
    python test_debug_v2.py --mode 2    # 定深模式
    python test_debug_v2.py --mode 3    # 定深定向模式
    python test_debug_v2.py --mode 4    # 定点模式
"""

import argparse
import math

import rospy
from auv_control.msg import PoseNEDcmd
from tf.transformations import quaternion_from_euler

# 运行模式常量
MODE_DEPTH     = 2
MODE_DEPTH_HDG = 3
MODE_DPROV     = 4

MODE_NAMES = {2: "定深", 3: "定深定向", 4: "定点(DPROV)"}


def make_cmd(mode, north, east, down, roll, pitch, yaw, forces):
    """
    构造 NED 坐标系的上层整包控制指令。

    roll、pitch、yaw 为度，转换为四元数后写入 PoseStamped。
    """
    cmd = PoseNEDcmd()
    cmd.mode = mode
    cmd.target.header.stamp = rospy.Time.now()
    cmd.target.header.frame_id = "map"
    cmd.target.pose.position.x = north
    cmd.target.pose.position.y = east
    cmd.target.pose.position.z = down

    quaternion = quaternion_from_euler(
        math.radians(roll),
        math.radians(pitch),
        math.radians(yaw),
    )
    cmd.target.pose.orientation.x = quaternion[0]
    cmd.target.pose.orientation.y = quaternion[1]
    cmd.target.pose.orientation.z = quaternion[2]
    cmd.target.pose.orientation.w = quaternion[3]
    cmd.force.TX, cmd.force.TY, cmd.force.TZ = forces[:3]
    cmd.force.MX, cmd.force.MY, cmd.force.MZ = forces[3:]
    return cmd


def main():
    parser = argparse.ArgumentParser(description="debug_driver_v2 测试程序")
    parser.add_argument("--mode", type=int, default=4, choices=[2, 3, 4],
                        help="控制模式: 2=定深, 3=定深定向, 4=定点 (默认:4)")
    parser.add_argument("--rate", type=float, default=5.0,
                        help="发布频率 Hz (默认:5)")
    args = parser.parse_args()

    rospy.init_node("test_debug_v2", anonymous=True)
    pub = rospy.Publisher("/cmd/pose/ned", PoseNEDcmd, queue_size=10)
    rate = rospy.Rate(args.rate)

    mode_name = MODE_NAMES.get(args.mode, f"未知({args.mode})")
    rospy.loginfo(f"test_debug_v2: 测试模式={args.mode}({mode_name}), "
                  f"发布频率={args.rate}Hz, 发布到 /cmd/pose/ned")

    # ── 预设测试参数 ──
    if args.mode == MODE_DEPTH:
        # 定深：仅深度闭环，其余开环力控
        north, east, down = 0.0, 0.0, 1.5
        roll, pitch, yaw = 0.0, 0.0, 0.0
        forces = [500, 0, 0, 0, 0, 0]  # TX=500g 前进推力
        rospy.loginfo("  预设: depth=-1.5m, TX=500, 其余力=0")

    elif args.mode == MODE_DEPTH_HDG:
        # 定深定向：深度+航向闭环，其余开环力控
        north, east, down = 0.0, 0.0, 1.5
        roll, pitch, yaw = 0.0, 0.0, 90.0
        forces = [500, 200, 0, 0, 0, 0]  # TX=500g 前进, TY=200g 侧推
        rospy.loginfo("  预设: depth=-1.5m, yaw=90°, TX=500, TY=200")

    else:  # MODE_DPROV
        # 定点：全部闭环
        north, east, down = 5.0, 3.0, 1.5
        roll, pitch, yaw = 0.0, 0.0, 0.0
        forces = [0, 0, 0, 0, 0, 0]  # 定点模式力全为0
        rospy.loginfo("  预设: NED(5,3,-1.5), yaw=0°, 力全0")

    rospy.loginfo("  Ctrl+C 停止发送")

    while not rospy.is_shutdown():
        pub.publish(make_cmd(args.mode, north, east, down, roll, pitch, yaw, forces))
        rate.sleep()


if __name__ == "__main__":
    main()
