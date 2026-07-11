#! /home/xhy/xhy_env/bin/python
"""
名称：task_v2_common.py
功能：任务节点的公共驱动模块
描述：
    1. 从 TF 树获取 AUV 在 map 坐标系下的当前位姿；
    2. 将任务目标拆分为带步长限制的中间目标，并发布到 /target；
    3. 通过 /auv_actuator_control 控制红绿灯、补光灯和执行器；
    4. 提供定时亮灯、闪灯、往复搜索、原地旋转和接触点计算功能。
监听：/tf
发布：/target (PoseStamped)，/auv_actuator_control (ActuatorControl)，/finished (String)
说明：本文件只封装多个任务共同使用的功能，不单独启动 ROS 节点。
"""

import copy
import math

import rospy
import tf
from auv_control.msg import ActuatorControl
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def clamp(value, lower, upper):
    """将数值限制在 [lower, upper] 范围内。"""
    return max(lower, min(upper, value))


def wrap_angle(angle):
    """将弧度角归一化到 [-pi, pi) 区间。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quaternion(quaternion):
    """从 geometry_msgs/Quaternion 中提取偏航角，单位为弧度。"""
    return euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ])[2]


class MissionBase:
    """V2 任务状态机的公共基类。

    子类只需要定义任务步骤；位姿获取、目标发布、外设控制和任务结束
    等通用操作由本类统一完成。
    """

    def __init__(self, node_name, rate_hz=5.0):
        """初始化公共话题、TF 监听器和运动控制参数。"""
        self.node_name = node_name
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10)
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10)
        self.device_pub = rospy.Publisher(
            '/auv_actuator_control', ActuatorControl, queue_size=10
        )
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(rate_hz)

        self.pitch_offset = math.radians(rospy.get_param('/pitch_offset', 0.0))
        self.default_heading_servo = int(rospy.get_param('/task_v2_heading_servo', 0x80))
        self.default_clamp_servo = int(rospy.get_param('/task_v2_clamp_servo', 0x00))
        self.default_drive_cmd = int(rospy.get_param('/task_v2_drive_cmd', 0))
        self.default_drive_speed = int(rospy.get_param('/task_v2_drive_speed', 0))
        self.max_xy_step = rospy.get_param('~max_xy_step', 0.5)
        self.max_z_step = rospy.get_param('~max_z_step', 0.1)
        self.max_yaw_step = math.radians(rospy.get_param('~max_yaw_step_deg', 5.0))
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.15)
        self.yaw_tolerance = math.radians(rospy.get_param('~yaw_tolerance_deg', 2.0))

        self.step = 0
        self.step_started = rospy.Time.now()
        self.hold_pose = None
        self.rotation_state = None
        self.sweep_state = None

    ############################################### 状态管理 #########################################
    def set_step(self, step):
        """切换状态机步骤，并清空只对上一步有效的临时动作状态。"""
        self.step = step
        self.step_started = rospy.Time.now()
        self.hold_pose = None
        self.rotation_state = None
        self.sweep_state = None

    def step_elapsed(self):
        """返回进入当前状态后经过的时间，单位为秒。"""
        return (rospy.Time.now() - self.step_started).to_sec()

    ############################################### 参数层 ###########################################
    def pose_from_param(self, name, default):
        """从 ROS 参数读取一个 map 坐标系目标位姿。

        Parameters:
            name: str，ROS 参数名。
            default: list，默认值 [north, east, down, yaw_deg]。

        Returns:
            PoseStamped，位置单位为米，航向输入单位为度。

        未配置比赛场地坐标时会输出警告，提醒部署前完成标定。
        """
        if not rospy.has_param(name):
            rospy.logwarn('%s: %s not set; using placeholder %s', self.node_name, name, default)
        values = rospy.get_param(name, default)
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError('{} must be [north, east, down, yaw_deg]'.format(name))

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position = Point(*[float(value) for value in values[:3]])
        pose.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0,
            self.pitch_offset,
            math.radians(float(values[3])),
        ))
        return pose

    ############################################### 驱动层 ###########################################
    def get_current_pose(self):
        """从 TF 树读取 map 到 base_link 的变换并生成当前位姿。

        Returns:
            PoseStamped：读取成功时返回当前位姿；失败时返回 None。
        """
        try:
            self.tf_listener.waitForTransform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(2, '%s: cannot read AUV pose: %s', self.node_name, error)
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    @staticmethod
    def position_distance(first, second):
        """计算两个 Point 之间的三维欧氏距离，单位为米。"""
        dx = first.x - second.x
        dy = first.y - second.y
        dz = first.z - second.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def move_to_pose(self, target, position_tolerance=None, yaw_tolerance=None):
        """向目标位姿运动一次，并判断是否到达。

        每次调用只发布一个带水平、深度和航向步长限制的中间目标，
        因此任务主循环需要持续调用本函数，直到返回 True。

        Parameters:
            target: PoseStamped，map 坐标系下的最终目标位姿。
            position_tolerance: float，可选的位置到达阈值，单位为米。
            yaw_tolerance: float，可选的航向到达阈值，单位为弧度。

        Returns:
            bool：位置和航向均满足阈值时返回 True。
        """
        current = self.get_current_pose()
        if current is None:
            return False

        position_tolerance = (
            self.position_tolerance if position_tolerance is None else position_tolerance
        )
        yaw_tolerance = self.yaw_tolerance if yaw_tolerance is None else yaw_tolerance

        target_yaw = yaw_from_quaternion(target.pose.orientation)
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        distance = self.position_distance(current.pose.position, target.pose.position)
        yaw_error = abs(wrap_angle(target_yaw - current_yaw))

        if distance <= position_tolerance and yaw_error <= yaw_tolerance:
            final_pose = copy.deepcopy(target)
            final_pose.header.stamp = rospy.Time.now()
            self.target_pub.publish(final_pose)
            return True

        dx = target.pose.position.x - current.pose.position.x
        dy = target.pose.position.y - current.pose.position.y
        horizontal_distance = math.hypot(dx, dy)

        if horizontal_distance > position_tolerance:
            desired_yaw = math.atan2(dy, dx)
        else:
            desired_yaw = target_yaw

        yaw_step = clamp(
            wrap_angle(desired_yaw - current_yaw),
            -self.max_yaw_step,
            self.max_yaw_step,
        )
        next_yaw = wrap_angle(current_yaw + yaw_step)

        if horizontal_distance > self.max_xy_step:
            scale = self.max_xy_step / horizontal_distance
        else:
            scale = 1.0

        next_pose = PoseStamped()
        next_pose.header.frame_id = 'map'
        next_pose.header.stamp = rospy.Time.now()
        next_pose.pose.position.x = current.pose.position.x + dx * scale
        next_pose.pose.position.y = current.pose.position.y + dy * scale
        next_pose.pose.position.z = current.pose.position.z + clamp(
            target.pose.position.z - current.pose.position.z,
            -self.max_z_step,
            self.max_z_step,
        )
        next_pose.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0,
            self.pitch_offset,
            next_yaw,
        ))
        self.target_pub.publish(next_pose)
        return False

    def hold_position(self):
        """记录并持续发布当前位姿，使执行灯光或机构动作时保持定点。"""
        if self.hold_pose is None:
            self.hold_pose = self.get_current_pose()
        if self.hold_pose is not None:
            self.hold_pose.header.stamp = rospy.Time.now()
            self.target_pub.publish(self.hold_pose)

    ############################################### 外设层 ###########################################
    def publish_device(self, red=0, green=0, servo=None, light1=0, light2=0):
        """发布一次外设控制消息。

        Parameters:
            red/green: int，红灯和绿灯开关，取值 0 或 1。
            servo: int，可选，开合舵机控制值；None 时使用 /task_v2_clamp_servo。
            light1/light2: int，两路补光灯亮度，取值 0～100。
        """
        message = ActuatorControl()
        message.light1 = int(light1)
        message.light2 = int(light2)
        message.heading_servo = self.default_heading_servo
        message.clamp_servo = self.default_clamp_servo if servo is None else int(servo)
        message.drive_cmd = self.default_drive_cmd
        message.drive_speed = self.default_drive_speed
        message.red_light = int(red)
        message.yellow_light = 0
        message.green_light = int(green)
        self.device_pub.publish(message)

    def blink_lights(self, red, green, count, half_period=0.5):
        """非阻塞闪灯；一次亮和一次灭构成一次完整闪烁。

        Returns:
            bool：完成指定闪烁次数后返回 True。
        """
        self.hold_position()
        phase = int(self.step_elapsed() / half_period)
        total_phases = int(count) * 2
        if phase >= total_phases:
            self.publish_device()
            return True

        lights_on = phase % 2 == 0
        self.publish_device(red=red if lights_on else 0, green=green if lights_on else 0)
        return False

    def show_lights(self, red, green, duration):
        """保持指定灯光 duration 秒，结束后自动熄灯。"""
        self.hold_position()
        if self.step_elapsed() >= duration:
            self.publish_device()
            return True
        self.publish_device(red=red, green=green)
        return False

    ############################################### 搜索与动作层 #####################################
    def rotate_360(self, direction=1, step_deg=5.0):
        """保持当前位置完成一次 360 度旋转，随后恢复初始航向。

        Parameters:
            direction: int，1 为逆时针，-1 为顺时针。
            step_deg: float，每次发布的航向增量，单位为度。

        Returns:
            bool：旋转一周并恢复初始航向后返回 True。
        """
        current = self.get_current_pose()
        if current is None:
            return False

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        if self.rotation_state is None:
            self.rotation_state = {
                'position': copy.deepcopy(current.pose.position),
                'initial_yaw': current_yaw,
                'last_yaw': current_yaw,
                'total': 0.0,
                'restoring': False,
            }

        state = self.rotation_state
        delta = wrap_angle(current_yaw - state['last_yaw'])
        if delta * direction > 0.0:
            state['total'] += abs(delta)
        state['last_yaw'] = current_yaw

        if state['total'] >= 2.0 * math.pi - math.radians(8.0):
            state['restoring'] = True

        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rospy.Time.now()
        target.pose.position = copy.deepcopy(state['position'])

        if state['restoring']:
            target.pose.orientation = Quaternion(*quaternion_from_euler(
                0.0, self.pitch_offset, state['initial_yaw']
            ))
            return self.move_to_pose(target, position_tolerance=0.2)

        command_yaw = wrap_angle(current_yaw + direction * math.radians(step_deg))
        target.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0, self.pitch_offset, command_yaw
        ))
        self.target_pub.publish(target)
        return False

    def sweep_for_target(self, max_angle_deg=30.0, step_deg=2.0):
        """在初始航向左右往复转动，用于视觉目标搜索。

        本函数不会自行结束；视觉回调获得目标后，由任务状态机切换步骤。
        """
        current = self.get_current_pose()
        if current is None:
            return

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        if self.sweep_state is None:
            self.sweep_state = {
                'pose': copy.deepcopy(current),
                'origin_yaw': current_yaw,
                'direction': 1,
            }

        state = self.sweep_state
        relative_yaw = wrap_angle(current_yaw - state['origin_yaw'])
        max_angle = math.radians(max_angle_deg)
        if relative_yaw >= max_angle:
            state['direction'] = -1
        elif relative_yaw <= -max_angle:
            state['direction'] = 1

        command = copy.deepcopy(state['pose'])
        command.header.stamp = rospy.Time.now()
        command.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0,
            self.pitch_offset,
            wrap_angle(current_yaw + state['direction'] * math.radians(step_deg)),
        ))
        self.target_pub.publish(command)

    ############################################### 坐标转换层 #######################################
    def contact_pose(self, detection_pose, standoff=0.0):
        """根据视觉目标计算使 hand 坐标系到达标记处的 base_link 位姿。

        Parameters:
            detection_pose: PoseStamped，相机坐标系下的标记位姿。
            standoff: float，hand 与标记之间保留的距离，单位为米。

        Returns:
            (target, marker_in_map)：base_link 目标位姿和标记的 map 位姿；
            TF 转换失败时返回 (None, None)。
        """
        try:
            self.tf_listener.waitForTransform(
                'map', detection_pose.header.frame_id, detection_pose.header.stamp,
                rospy.Duration(1.0),
            )
            marker_in_map = self.tf_listener.transformPose('map', detection_pose)
            hand_translation, _ = self.tf_listener.lookupTransform(
                'base_link', 'hand', rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(2, '%s: marker transform failed: %s', self.node_name, error)
            return None, None

        current = self.get_current_pose()
        if current is None:
            return None, None

        dx = marker_in_map.pose.position.x - current.pose.position.x
        dy = marker_in_map.pose.position.y - current.pose.position.y
        desired_yaw = math.atan2(dy, dx)

        hand_x = (
            hand_translation[0] * math.cos(desired_yaw)
            - hand_translation[1] * math.sin(desired_yaw)
        )
        hand_y = (
            hand_translation[0] * math.sin(desired_yaw)
            + hand_translation[1] * math.cos(desired_yaw)
        )

        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rospy.Time.now()
        target.pose.position.x = (
            marker_in_map.pose.position.x - hand_x - standoff * math.cos(desired_yaw)
        )
        target.pose.position.y = (
            marker_in_map.pose.position.y - hand_y - standoff * math.sin(desired_yaw)
        )
        target.pose.position.z = marker_in_map.pose.position.z - hand_translation[2]
        target.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0, self.pitch_offset, desired_yaw
        ))
        return target, marker_in_map

    ############################################### 任务结束 #########################################
    def finish_task(self):
        """关闭外设、发布任务完成消息，并结束当前 ROS 节点。"""
        self.publish_device()
        self.finished_pub.publish('{} finished'.format(self.node_name))
        rospy.loginfo('%s: mission finished', self.node_name)
        rospy.signal_shutdown('mission finished')
