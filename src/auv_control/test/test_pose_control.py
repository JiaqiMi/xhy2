#! /home/xhy/xhy_env/bin/python
"""
名称: test_pose_control.py
功能: 通过UI输入目标位姿 (在map坐标系下), 发布到/target话题
作者: buyegaid
监听：/tf (来自tf树)
发布：/auv_control (Control.msg) 被sensor_driver订阅
      /target (PoseStamped.msg) 

记录：
2025.7.23 1:21
    第一版完成
    TODO 测试
"""
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped,Point
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf
from auv_control.msg import Control
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']  # 或 'Noto Sans CJK TC'（繁体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
class PoseControlUI:
    """AUV位姿控制UI界面"""
    def __init__(self):
        # 初始化ROS节点
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10) # 发布目标位姿消息
        self.control_pub = rospy.Publisher('/sensor', Control, queue_size=10) # 发布控制消息
        self.tf_listener = tf.TransformListener() # tf监听器
        self.rate = rospy.Rate(5)
        # 添加绘图相关变量
        self.fig = Figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(111)
        self.current_pos = [0, 0]  # 当前位置
        self.target_pos = [0, 0]  # 目标位置

        # 创建UI窗口
        self.root = tk.Tk()
        self.root.title("AUV位姿控制测试")
        self.setup_ui()

        self.is_sending = False  # 添加循环发送标志
        self.input_widgets = []  # 存储需要锁定的输入控件



    def setup_ui(self):
        # 目标位姿输入区域
        input_frame = ttk.LabelFrame(self.root, text="目标位姿输入(map坐标系)", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        # 位置输入
        ttk.Label(input_frame, text="n(m):").grid(row=0, column=0)
        self.x_entry = ttk.Entry(input_frame, width=10)
        self.x_entry.grid(row=0, column=1)

        ttk.Label(input_frame, text="e(m):").grid(row=1, column=0)
        self.y_entry = ttk.Entry(input_frame, width=10)
        self.y_entry.grid(row=1, column=1)

        ttk.Label(input_frame, text="d(m):").grid(row=2, column=0)
        self.z_entry = ttk.Entry(input_frame, width=10)
        self.z_entry.grid(row=2, column=1)

        # 姿态输入
        ttk.Label(input_frame, text="Roll(°):").grid(row=3, column=0)
        self.roll_entry = ttk.Entry(input_frame, width=10)
        self.roll_entry.grid(row=3, column=1)

        ttk.Label(input_frame, text="Pitch(°):").grid(row=4, column=0)
        self.pitch_entry = ttk.Entry(input_frame, width=10)
        self.pitch_entry.grid(row=4, column=1)

        ttk.Label(input_frame, text="Yaw(°):").grid(row=5, column=0)
        self.yaw_entry = ttk.Entry(input_frame, width=10)
        self.yaw_entry.grid(row=5, column=1)
        # LED
        self.led_red_var = tk.IntVar()
        self.led_green_var = tk.IntVar()
        tk.Checkbutton(input_frame, text="红色LED", variable=self.led_red_var).grid(row=6, column=0)
        tk.Checkbutton(input_frame, text="绿色LED", variable=self.led_green_var).grid(row=6, column=1)
        # 舵机
        tk.Label(input_frame, text="舵机(50-250)：").grid(row=7, column=0)
        self.servo_spin = tk.Spinbox(input_frame, from_=50, to=250, width=5)
        self.servo_spin.grid(row=7, column=1)
        # 外设控制帧选项
        self.enable_var = tk.IntVar()
        tk.Checkbutton(input_frame, text="外设控制帧（LED/舵机）", variable=self.enable_var).grid(row=8, column=0, columnspan=2)

        # 修改控制按钮区域
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=1, column=0, pady=5)
        self.send_once_btn = ttk.Button(btn_frame, text="发送一次", command=self.send_target)
        self.send_once_btn.grid(row=0, column=0, padx=5)
        self.send_loop_btn = ttk.Button(btn_frame, text="循环发送", command=self.toggle_send_loop)
        self.send_loop_btn.grid(row=0, column=1, padx=5)

        # 保存所有输入控件的引用
        self.input_widgets = [
            self.x_entry, self.y_entry, self.z_entry,
            self.roll_entry, self.pitch_entry, self.yaw_entry,
            self.servo_spin
        ]

        # 当前位姿显示区域
        pose_frame = ttk.LabelFrame(self.root, text="当前位姿(map坐标系)", padding=10)
        pose_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")

        self.pose_labels = {}
        pose_items = [
            ("N(m)", "n"), 
            ("E(m)", "e"), 
            ("D(m)", "d"),
            ("Roll(°)", "roll"), 
            ("Pitch(°)", "pitch"), 
            ("Yaw(°)", "yaw")
        ]
        for i, (label, key) in enumerate(pose_items):
            ttk.Label(pose_frame, text=label).grid(row=i, column=0)
            self.pose_labels[key] = ttk.Label(pose_frame, text="0.0")
            self.pose_labels[key].grid(row=i, column=1)

        # 添加绘图区域
        plot_frame = ttk.LabelFrame(self.root, text="位置显示", padding=10)
        plot_frame.grid(row=3, column=0, padx=10, pady=5, sticky="nsew")
        
        # 设置绘图
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.grid(True)
        self.ax.set_aspect('equal')  # 等比例显示
        
        # 初始化散点图
        self.current_point, = self.ax.plot([], [], 'bo', label='now pos')
        self.target_point, = self.ax.plot([], [], 'ro', label='target')
        self.ax.legend()
        
        # 将matplotlib图形嵌入Tkinter窗口
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update_plot(self):
        """更新位置图"""
        try:
            # 更新当前位置
            (trans, rot) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
            self.current_pos = [trans[1],  # East
                              trans[0]]    # North
            
            # 更新目标位置（如果有）
            if hasattr(self, 'last_target'):
                self.target_pos = [self.last_target.pose.position.y,  # East
                                 self.last_target.pose.position.x]    # North
            
            # 更新散点位置
            self.current_point.set_data(self.current_pos[0], self.current_pos[1])
            self.target_point.set_data(self.target_pos[0], self.target_pos[1])
            
            # 调整显示范围
            all_x = [self.current_pos[0], self.target_pos[0]]
            all_y = [self.current_pos[1], self.target_pos[1]]
            margin = 1.0  # 边距1米
            self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            
            # 刷新画布
            self.canvas.draw()
            
        except Exception as e:
            rospy.logwarn_throttle(1, f"更新位置图失败: {e}")

    def update_pose_display(self):
        """更新当前位姿显示"""
        try:
            # 获取最新的tf变换
            (trans, rot) = self.tf_listener.lookupTransform('map', 'base_link', rospy.Time(0))
            # 获取位置
            self.pose_labels["n"].config(text=f"{trans[0]:.3f}")
            self.pose_labels["e"].config(text=f"{trans[1]:.3f}")
            self.pose_labels["d"].config(text=f"{trans[2]:.3f}")

            # 获取姿态
            quat = rot
            roll, pitch, yaw = euler_from_quaternion(quat)
            self.pose_labels["roll"].config(text=f"{np.degrees(roll):.1f}")
            self.pose_labels["pitch"].config(text=f"{np.degrees(pitch):.1f}")
            self.pose_labels["yaw"].config(text=f"{np.degrees(yaw):.1f}")
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"获取当前位姿失败: {e}")

        self.update_plot()  # 同时更新位置图
        self.root.after(200, self.update_pose_display)  # 5Hz更新

    def validate_input(self):
        """验证输入值"""
        try:
            x = float(self.x_entry.get())
            y = float(self.y_entry.get())
            z = float(self.z_entry.get())
            roll = float(self.roll_entry.get())
            pitch = float(self.pitch_entry.get())
            yaw = float(self.yaw_entry.get())
            return x, y, z, roll, pitch, yaw
        except ValueError:
            messagebox.showerror("输入错误", "请输入有效的数字")
            # raise ValueError("输入数据无效")
            return None

    def lock_inputs(self, lock=True):
        """锁定/解锁输入控件"""
        state = 'disabled' if lock else 'normal'
        for widget in self.input_widgets:
            widget.configure(state=state)
        self.led_red_var.set(self.led_red_var.get())  # 保持当前值
        self.led_green_var.set(self.led_green_var.get())
        self.enable_var.set(self.enable_var.get())

    def toggle_send_loop(self):
        """切换循环发送状态"""
        if not self.is_sending:
            # 验证输入
            if self.validate_input() is None:
                return
            
            self.is_sending = True
            self.send_loop_btn.configure(text="停止发送")
            self.send_once_btn.configure(state='disabled')
            self.lock_inputs(True)
            self.send_loop()
        else:
            self.is_sending = False
            self.send_loop_btn.configure(text="循环发送")
            self.send_once_btn.configure(state='normal')
            self.lock_inputs(False)

    def send_loop(self):
        """循环发送数据"""
        if not self.is_sending:
            return
        
        self.send_target()  # 发送一次数据
        self.root.after(200, self.send_loop)  # 5Hz循环发送

    def numpy_distance(self, p1:Point, p2:Point):
        """
        使用NumPy计算NED距离

        Parameters:
            p1: Point 第一个点
            p2: Point 第二个点

        Returns:
            out: float 两个点之间的距离
        """
        a = np.array([p1.x, p1.y, p1.z])
        b = np.array([p2.x, p2.y, p2.z])
        return np.linalg.norm(a - b)
    def generate_smooth_pose(self, current_pose, target_pose,max_xy_step=0.8, max_z_step=0.1, max_yaw_step=np.radians(5)):
        """
        使用三阶贝塞尔曲线生成平滑的路径点，采用先前向移动再调整航向的策略
        
        Parameters:
            current_pose: PoseStamped, 当前位姿
            target_pose: PoseStamped, 目标位姿
            max_xy_step: float, 最大水平步长(米)
            max_z_step: float, 最大垂直步长(米)
            max_yaw_step: float, 最大偏航角步长(弧度)
            
        Returns:
            next_pose: PoseStamped, 下一个位姿点
        """
        # 创建下一个位姿点
        next_pose = PoseStamped()
        next_pose.header.frame_id = "map"
        next_pose.header.stamp = rospy.Time.now()
        
        # 获取当前和目标的姿态角
        _, _, current_yaw = euler_from_quaternion([
            current_pose.pose.orientation.x,
            current_pose.pose.orientation.y,
            current_pose.pose.orientation.z,
            current_pose.pose.orientation.w
        ])
        _, target_pitch, target_yaw = euler_from_quaternion([
            target_pose.pose.orientation.x,
            target_pose.pose.orientation.y,
            target_pose.pose.orientation.z,
            target_pose.pose.orientation.w
        ])
        
        # 计算起点和终点
        p0 = np.array([current_pose.pose.position.x, current_pose.pose.position.y, current_pose.pose.position.z])
        p3 = np.array([target_pose.pose.position.x, target_pose.pose.position.y, target_pose.pose.position.z])
        
        # 计算到目标点的距离
        dist_to_target = self.numpy_distance(current_pose.pose.position, target_pose.pose.position)

        # 如果距离目标点很近(小于1米)，则开始调整最终姿态
        if dist_to_target < max_xy_step:
            # 计算yaw角差异（处理角度环绕）
            dyaw = target_yaw - current_yaw
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            dyaw = np.clip(dyaw, -max_yaw_step, max_yaw_step) # 应用最大步长
            next_yaw = current_yaw + dyaw
            
            # 平滑过渡姿态角
            # next_roll = current_roll + np.clip(target_roll - current_roll, -max_yaw_step, max_yaw_step)
            next_pitch = target_pitch  # 保持目标俯仰角
            
            next_pose.pose.position = target_pose.pose.position
            next_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, next_pitch, next_yaw))
            return next_pose
        
        # 如果距离目标点较远，继续沿着前进方向移动
        # 计算前进方向的单位向量
        direction = p3 - p0
        direction_xy = direction[:2]
        direction_xy_norm = np.linalg.norm(direction_xy)
        if direction_xy_norm > 0:
            direction_xy = direction_xy / direction_xy_norm
            # 计算期望的航向角(前进方向)
            desired_yaw = np.arctan2(direction_xy[1], direction_xy[0])
            
            # 计算航向差
            dyaw = desired_yaw - current_yaw
            dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
            dyaw = np.clip(dyaw, -max_yaw_step, max_yaw_step)
            next_yaw = current_yaw + dyaw
        else:
            next_yaw = current_yaw
            
        # 保持当前的俯仰和横滚角
        # 保持当前的相当于是不控制，对于横滚来说不影响，对于俯仰来说，还是需要控制的
        # next_roll = current_roll
        next_pitch = target_pitch # 保持目标俯仰角
        
        # 计算控制点（根据前进方向）
        control_dist = dist_to_target * 0.4
        p1 = p0 + control_dist * np.array([np.cos(next_yaw), np.sin(next_yaw), 0])
        p2 = p3 - control_dist * np.array([np.cos(target_yaw), np.sin(target_yaw), 0])
        
        # 如果没有存储当前的贝塞尔曲线参数t值，初始化为0
        if not hasattr(self, 'bezier_t'):
            self.bezier_t = 0.0
        
        # 计算下一个t值（确保平滑过渡）
        dt = 0.1  # t的增量
        self.bezier_t = min(1.0, self.bezier_t + dt)
        t = self.bezier_t
        
        # 计算三阶贝塞尔曲线上的点
        next_point = (1-t)**3 * p0 + \
                    3*(1-t)**2 * t * p1 + \
                    3*(1-t) * t**2 * p2 + \
                    t**3 * p3
        
        # 应用步长限制
        dp = next_point - p0
        dist_xy = np.sqrt(dp[0]**2 + dp[1]**2)
        if dist_xy > max_xy_step:
            scale = max_xy_step / dist_xy
            dp[0] *= scale
            dp[1] *= scale
        dp[2] = np.clip(dp[2], -max_z_step, max_z_step)
        
        # 设置下一个位置
        next_pose.pose.position.x = current_pose.pose.position.x + dp[0]
        next_pose.pose.position.y = current_pose.pose.position.y + dp[1]
        next_pose.pose.position.z = current_pose.pose.position.z + dp[2]
        
        # 设置姿态
        next_pose.pose.orientation = Quaternion(*quaternion_from_euler(0, next_pitch, next_yaw))
        
        # 如果到达目标点，重置贝塞尔曲线参数
        if dist_to_target < 0.1:  # 距离阈值
            if hasattr(self, 'bezier_t'):
                del self.bezier_t
        
        return next_pose

    def send_target(self):
        """发送目标位姿或外设控制帧"""
        if not self.is_sending:  # 非循环发送时才验证输入
            values = self.validate_input()
            if values is None:
                return
        else:
            # 循环发送时直接使用已验证的输入值
            try:
                values = (
                    float(self.x_entry.get()),
                    float(self.y_entry.get()),
                    float(self.z_entry.get()),
                    float(self.roll_entry.get()),
                    float(self.pitch_entry.get()),
                    float(self.yaw_entry.get())
                )
            except ValueError:
                self.is_sending = False
                self.send_loop_btn.configure(text="循环发送")
                self.send_once_btn.configure(state='normal')
                self.lock_inputs(False)
                messagebox.showerror("错误", "输入数据无效，停止发送")
                return

        x, y, z, roll, pitch, yaw = values

        if self.enable_var.get() == 0:
            try:
                # 获取当前位姿
                (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
                current_pose = PoseStamped()
                current_pose.header.frame_id = "map"
                current_pose.pose.position.x = trans[0]
                current_pose.pose.position.y = trans[1]
                current_pose.pose.position.z = trans[2]
                current_pose.pose.orientation.x = rot[0]
                current_pose.pose.orientation.y = rot[1]
                current_pose.pose.orientation.z = rot[2]
                current_pose.pose.orientation.w = rot[3]
                
                # 构造目标位姿
                target = PoseStamped()
                target.header.stamp = rospy.Time.now()
                target.header.frame_id = "map"
                target.pose.position.x = x
                target.pose.position.y = y
                target.pose.position.z = z
                q = quaternion_from_euler(np.radians(roll), np.radians(pitch), np.radians(yaw))
                target.pose.orientation = Quaternion(*q)
                
                # 生成平滑位姿点
                smooth_target = self.generate_smooth_pose(current_pose, target)
                
                # 发布平滑位姿点
                self.target_pub.publish(smooth_target)
                self.last_target = target  # 保存最终目标位置用于绘图
                
                rospy.loginfo(f"发送平滑位姿: 位置({smooth_target.pose.position.x:.3f}, "
                            f"{smooth_target.pose.position.y:.3f}, {smooth_target.pose.position.z:.3f})")
                
            except (tf.LookupException, tf.ConnectivityException, 
                    tf.ExtrapolationException) as e:
                rospy.logwarn(f"获取当前位姿失败: {e}")
                return
        else:
            # 勾选，发送Control.msg
            control_msg = Control()
            control_msg.led_red = bool(self.led_red_var.get())
            control_msg.led_green = bool(self.led_green_var.get())
            try:
                control_msg.servo = int(self.servo_spin.get())
            except ValueError:
                messagebox.showerror("输入错误", "舵机数值无效")
                return
            self.control_pub.publish(control_msg)
            rospy.loginfo(f"发送外设控制帧: 红LED={control_msg.led_red}, 绿LED={control_msg.led_green}, 舵机={control_msg.servo}")
    
    def run(self):
        """运行UI"""
        self.update_pose_display()
        self.root.mainloop()

if __name__ == "__main__":
    try:
        rospy.init_node('test_pose_control')
        ui = PoseControlUI()
        ui.run()
    except rospy.ROSInterruptException:
        pass
