#!/usr/bin/env python3
# 功能，测试点控制功能
# 输入：ui界面上输入的一个机器人坐标系下的一个点的坐标，单位m（前右下坐标系）
# 输出：发送一个Control.msg消息，main_driver和sensor_driver会订阅该消息并控制机器人移动到目标为止

import rospy
from auv_control.msg import Control, AUVPose, AUVMotor, AUVData
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import tkinter as tk
from tkinter import messagebox
import navigation_utils

class auv: # 用于存储auv的状态信息
    def __init__(self):
        self.east = 0.0            # 东向坐标，单位：mm
        self.north = 0.0            # 北向坐标，单位：mm.
        self.up = 0.0              # 上向坐标，单位：mm
        self.yaw = 0.0             # 航向角，单位：角度
        self.pitch = 0.0           # 俯仰角，单位：角度
        self.roll = 0.0            # 横滚角，单位：角度
        self.velocity = 0.0        # 速度，单位：mm/s
        self.latitude = 0.0        # 纬度，单位：度
        self.longitude = 0.0       # 经度，单位：度
        self.depth = 0.0           # 深度，单位：m
        self.altitude = 0.0        # 高度，单位：m

auv_state = auv()  # 全局AUV状态实例

def auv_data_callback(msg):
    # 解析AUVData消息并保存到auv_state
    auv_state.east = msg.pose.east
    auv_state.north = msg.pose.north
    auv_state.up = msg.pose.up
    auv_state.yaw = msg.pose.yaw
    auv_state.pitch = msg.pose.pitch
    auv_state.roll = msg.pose.roll
    auv_state.velocity = msg.pose.velocity
    auv_state.latitude = msg.pose.latitude
    auv_state.longitude = msg.pose.longitude
    auv_state.depth = msg.pose.depth
    auv_state.altitude = msg.pose.altitude
    # 可根据需要打印或处理
    # print(f"AUV状态: 东{auv_state.east} 北{auv_state.north} 上{auv_state.up} 深度{auv_state.depth}")

class ControlUI:
    def __init__(self, master, pub):
        self.master = master
        self.pub = pub
        master.title("AUV 控制界面")

        # 坐标输入
        tk.Label(master, text="X（前向，m）：").grid(row=0, column=0)
        self.x_entry = tk.Entry(master)
        self.x_entry.grid(row=0, column=1)

        tk.Label(master, text="Y（右向，m）：").grid(row=1, column=0)
        self.y_entry = tk.Entry(master)
        self.y_entry.grid(row=1, column=1)

        tk.Label(master, text="Z（下向，m）：").grid(row=2, column=0)
        self.z_entry = tk.Entry(master)
        self.z_entry.grid(row=2, column=1)

        # LED
        self.led_red_var = tk.IntVar()
        self.led_green_var = tk.IntVar()
        tk.Checkbutton(master, text="红色LED", variable=self.led_red_var).grid(row=3, column=0)
        tk.Checkbutton(master, text="绿色LED", variable=self.led_green_var).grid(row=3, column=1)

        # 舵机
        tk.Label(master, text="舵机(50-250)：").grid(row=4, column=0)
        self.servo_spin = tk.Spinbox(master, from_=50, to=250, width=5)
        self.servo_spin.grid(row=4, column=1)

        # 外设控制帧选项
        self.enable_var = tk.IntVar()
        tk.Checkbutton(master, text="外设控制帧（LED/舵机）", variable=self.enable_var).grid(row=5, column=0, columnspan=2)

        # 发送按钮
        self.send_btn = tk.Button(master, text="发送", command=self.send_control)
        self.send_btn.grid(row=6, column=0, columnspan=2, pady=10)

        # AUV位姿显示
        self.pose_frame = tk.LabelFrame(master, text="AUV当前位姿", padx=5, pady=5)
        self.pose_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=10)

        self.pose_labels = {}
        pose_items = [
            ("前向X(mm)", "east"),
            ("右向Y(mm)", "north"),
            ("下向Z(mm)", "up"),
            ("深度(m)", "depth"),
            ("纬度(°)", "latitude"),
            ("经度(°)", "longitude"),
            ("速度(mm/s)", "velocity"),
            ("横滚角(°)", "roll"),
            ("俯仰角(°)", "pitch"),
            ("航向角(°)", "yaw"),
            ("高度(m)", "altitude"),
        ]
        for i, (label, key) in enumerate(pose_items):
            tk.Label(self.pose_frame, text=label).grid(row=i, column=0, sticky="w")
            self.pose_labels[key] = tk.Label(self.pose_frame, text="0.0")
            self.pose_labels[key].grid(row=i, column=1, sticky="w")

        # 定时刷新位姿显示
        self.update_pose_display()

    def update_pose_display(self):
        # 刷新AUV位姿显示
        self.pose_labels["east"].config(text="%.2f" % auv_state.east)
        self.pose_labels["north"].config(text="%.2f" % auv_state.north)
        self.pose_labels["up"].config(text="%.2f" % auv_state.up)
        self.pose_labels["depth"].config(text="%.2f" % auv_state.depth)
        self.pose_labels["latitude"].config(text="%.6f" % auv_state.latitude)
        self.pose_labels["longitude"].config(text="%.6f" % auv_state.longitude)
        self.pose_labels["velocity"].config(text="%.2f" % auv_state.velocity)
        self.pose_labels["roll"].config(text="%.2f" % auv_state.roll)
        self.pose_labels["pitch"].config(text="%.2f" % auv_state.pitch)
        self.pose_labels["yaw"].config(text="%.2f" % auv_state.yaw)
        self.pose_labels["altitude"].config(text="%.2f" % auv_state.altitude)
        self.master.after(500, self.update_pose_display)  # 500ms刷新一次

    def send_control(self):
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            z = float(self.z_entry.get())
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字坐标")
            return

        ctrl = Control()
        ctrl.enable = bool(self.enable_var.get())
        ctrl.led_red = 1 if self.led_red_var.get() else 0
        ctrl.led_green = 1 if self.led_green_var.get() else 0
        ctrl.servo = int(self.servo_spin.get())
        ctrl.force = AUVMotor()

        # 使用AUV当前位姿和目标点（前右下，单位m）转换为全局坐标
        lat, lon, depth, heading = navigation_utils.local_point_to_global(auv_state, (x, y, z))

        pose = AUVPose()
        pose.latitude = lat
        pose.longitude = lon
        pose.depth = depth
        pose.yaw = heading
        # 其他字段可根据需要补充
        ctrl.pose = pose

        if self.pub:
            self.pub.publish(ctrl)
            print("已发送: 目标点(经纬度/深度): %.6f, %.6f, %.2f, heading: %.2f, 红LED:%d, 绿LED:%d, 舵机:%d, enable:%d" %
                  (lat, lon, depth, heading, ctrl.led_red, ctrl.led_green, ctrl.servo, ctrl.enable))

def main():
    rospy.init_node('test_point_control')
    pub = rospy.Publisher('/auv_control', Control, queue_size=1)
    rospy.Subscriber('/debug_auv_data', AUVData, auv_data_callback)  # 订阅AUVData消息
    root = tk.Tk()
    ui = ControlUI(root, pub)
    root.mainloop()

if __name__ == '__main__':
    main()