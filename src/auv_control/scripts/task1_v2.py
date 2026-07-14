#! /home/xhy/xhy_env/bin/python
"""
名称：task1_v2.py
功能：2026 Task 1——主管道检修
描述：
    1. 任务启动时记录当前位姿，按比赛约定认为 map 原点已经位于红色圆形处；
    2. 从起点向前搜索底部红色长线，首次看到管线时记录红线起点；
    3. 根据红色长线识别结果拟合全局巡线路径，运行高度统一使用参考高度；
    4. 巡线过程中稳定识别黄色圆形/三角形和黑色方形，并在图形上方执行动作；
    5. 黄色图形亮红灯，黑色图形亮绿灯并根据实时航向累计旋转角度；
    6. 红色长线丢失超过设定时间后记录终点并结束任务。
监听：/obj/target_message，/obj/line_message，/tf
发布：/cmd/pose/ned，/cmd/actuator，/finished
说明：
    - 识别类别按感知方话题格式默认使用 triangle、circle、rectangle、line；
    - /task1_v2_reference_height 可设置相对红色圆形原点的运行高度，默认 0.4 m；
      map 为 NED 坐标，z/down 向下为正，因此默认运行目标 z 为 -0.4；
    - /task1_v2_initial_heading_deg 可设置搜索红线起点的初始航向，0 度沿 map +X；
    - 巡线阶段使用 PoseNEDcmd 的定深定向手控模式，按拟合曲线执行 LOS；
    - 图形只作为巡线过程中的触发点，不直接导航到图形位置；
    - 到达图形上方后使用 PoseNEDcmd 的动力定位 ROV 模式保持当前动作位姿；
    - 本文件自行实现发布、TF、当前位姿、外设控制、任务完成和状态机计时，
      不依赖 task_v2_common.py / MissionBase。

修改记录：
    2026.7.13：
        1. 将任务起点改为启动时当前位姿，符合 map 已在红色圆形处建系的约定；
        2. 增加参考运行高度参数，巡线和图形动作均只控制 XY，z 固定为参考运行高度；
        3. 将运动控制改为先转向目标点，再向目标点移动；
        4. 图形识别改为约 10 帧聚类稳定后入队，完成动作后按位置去重；
        5. 灯光动作改为每次亮灯 3 秒；
        6. 黑色图形旋转改为根据当前航向反馈累计角度，避免按命令步长误判；
        7. 增加红线起点、终点记录，红线结束后结束任务；
        8. 增加图形动作后的红线重捕获宽限，避免动作期间视觉中断导致提前结束；
        9. 降低黑色图形旋转默认航向步长，减少旋转时机器人中心漂移。
    2026.7.14：
        1. 合并 main 最新框架，执行器下行话题调整为 /cmd/actuator；
        2. 解决 task1_v2.py 合并冲突，保留当前 Task1 状态机和稳定识别逻辑。
        3. 修复最新 MissionBase 中不存在 publish_target 方法导致的目标发布错误。
        4. 移除 MissionBase 依赖，将 Task1 所需公共功能全部放入本文件。
        5. 运行高度改为相对红色圆形原点的参考高度，默认 0.4 m，便于现场修改。
        6. 巡线改为 PoseNEDcmd 手控模式，红线点集限量后拟合为曲线并进行 LOS 跟踪。
        7. 巡线日志不再输出未知总路径进度，改为输出已完成路径长度。
        8. 增加真实下水调试用节流日志，覆盖状态、感知、曲线、控制和动作执行。
        9. 图形识别后不直接前往图形坐标，改为沿巡线曲线到达图形上方后再触发动作。
        10. 增加可人工配置的初始搜索航向，任务开始按该航向寻找红色长线。
"""

import copy
import math

import numpy as np
import rospy
import tf
from auv_control.msg import ActuatorControl, PoseNEDcmd, TargetDetection, TargetDetection3
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = 'task1_v2'
DEFAULT_REFERENCE_HEIGHT = 0.4
DEFAULT_INITIAL_HEADING_DEG = 0.0
MODE_DEPTH = 2
MODE_DEPTH_HDG = 3
MODE_DPROV = 4
MODE_NAMES = {
    MODE_DEPTH: 'depth_manual',
    MODE_DEPTH_HDG: 'depth_heading_manual',
    MODE_DPROV: 'dprov',
}


def clamp(value, lower, upper):
    """将数值限制到指定闭区间内。"""
    return max(lower, min(upper, value))


def wrap_angle(angle):
    """将角度归一化到 [-pi, pi]，便于计算航向误差。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(quaternion):
    """从 geometry_msgs/Quaternion 中取出偏航角。"""
    angles = euler_from_quaternion([
        quaternion.x,
        quaternion.y,
        quaternion.z,
        quaternion.w,
    ])
    return angles[2]


def class_names(param_name, defaults):
    """读取类别参数，并统一转换为字符串列表。"""
    value = rospy.get_param(param_name, defaults)
    if isinstance(value, str):
        names = value.replace(',', ' ').split()
        return names if names else [value]
    return list(value)


def xy_distance(first, second):
    """只计算水平面距离，忽略 z 方向差异。"""
    return math.hypot(first.x - second.x, first.y - second.y)


def median(values):
    """返回一组数的中位数，用于降低识别抖动影响。"""
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


class Task1V2:
    """主管道检修任务状态机。"""

    STEP_SEARCH_LINE = 0
    STEP_FOLLOW_LINE = 1
    STEP_MOVE_TO_MARKER = 2
    STEP_LIGHT_ACTION = 3
    STEP_ROTATE_BLACK = 4
    STEP_FINISH = 5
    STEP_NAMES = {
        STEP_SEARCH_LINE: 'SEARCH_LINE',
        STEP_FOLLOW_LINE: 'FOLLOW_LINE',
        STEP_MOVE_TO_MARKER: 'MOVE_TO_MARKER',
        STEP_LIGHT_ACTION: 'LIGHT_ACTION',
        STEP_ROTATE_BLACK: 'ROTATE_BLACK',
        STEP_FINISH: 'FINISH',
    }

    def __init__(self):
        """初始化任务参数、视觉缓存以及目标检测订阅。"""
        self.node_name = NODE_NAME
        self.pose_cmd_pub = rospy.Publisher('/cmd/pose/ned', PoseNEDcmd, queue_size=10)
        self.finished_pub = rospy.Publisher('/finished', String, queue_size=10)
        self.device_pub = rospy.Publisher('/cmd/actuator', ActuatorControl, queue_size=10)
        self.tf_listener = tf.TransformListener()
        self.rate = rospy.Rate(rospy.get_param('~rate_hz', 5.0))

        self.pitch_offset = math.radians(rospy.get_param('/pitch_offset', 0.0))
        self.default_heading_servo = int(rospy.get_param('/task_v2_heading_servo', 0x80))
        self.default_clamp_servo = int(rospy.get_param('/task_v2_clamp_servo', 0x00))
        self.default_drive_cmd = int(rospy.get_param('/task_v2_drive_cmd', 0))
        self.default_drive_speed = int(rospy.get_param('/task_v2_drive_speed', 0))
        self.max_xy_step = rospy.get_param('~max_xy_step', 0.5)
        self.max_yaw_step = math.radians(rospy.get_param('~max_yaw_step_deg', 5.0))
        self.position_tolerance = rospy.get_param('~position_tolerance', 0.15)
        self.yaw_tolerance = math.radians(rospy.get_param('~yaw_tolerance_deg', 2.0))

        self.step = self.STEP_SEARCH_LINE
        self.step_started = rospy.Time.now()

        self.reference_height = float(rospy.get_param(
            '/task1_v2_reference_height', DEFAULT_REFERENCE_HEIGHT
        ))
        self.reference_z = float(rospy.get_param(
            '/task1_v2_reference_z', -self.reference_height
        ))
        self.initial_search_yaw = math.radians(float(rospy.get_param(
            '/task1_v2_initial_heading_deg', DEFAULT_INITIAL_HEADING_DEG
        )))

        self.search_forward_step = rospy.get_param('/task1_v2_search_forward_step', 0.3)
        self.line_forward_fraction = rospy.get_param('/task1_v2_line_forward_fraction', 0.8)
        self.line_lost_timeout = rospy.get_param('/task1_v2_line_lost_timeout', 5.0)
        self.curve_blind_follow_timeout = rospy.get_param(
            '/task1_v2_curve_blind_follow_timeout', 2.0
        )

        self.line_point_merge_distance = rospy.get_param(
            '/task1_v2_line_point_merge_distance', 0.15
        )
        self.line_curve_max_points = int(rospy.get_param(
            '/task1_v2_line_curve_max_points', 120
        ))
        self.line_curve_sample_count = int(rospy.get_param(
            '/task1_v2_line_curve_sample_count', 80
        ))
        self.line_curve_degree = int(rospy.get_param('/task1_v2_line_curve_degree', 3))
        self.line_curve_min_length = rospy.get_param('/task1_v2_line_curve_min_length', 0.4)

        self.los_lookahead_distance = rospy.get_param('/task1_v2_los_lookahead_distance', 0.6)
        self.line_end_margin = rospy.get_param('/task1_v2_line_end_margin', 0.5)
        self.manual_forward_force = rospy.get_param('/task1_v2_manual_forward_force', 300)
        self.manual_slow_forward_force = rospy.get_param(
            '/task1_v2_manual_slow_forward_force', 120
        )
        self.manual_lateral_gain = rospy.get_param('/task1_v2_manual_lateral_gain', 250.0)
        self.manual_max_lateral_force = rospy.get_param(
            '/task1_v2_manual_max_lateral_force', 180
        )
        self.manual_force_step = rospy.get_param('/task1_v2_manual_force_step', 50)
        self.manual_slow_yaw_error = math.radians(
            rospy.get_param('/task1_v2_manual_slow_yaw_error_deg', 20.0)
        )
        self.manual_slow_lateral_error = rospy.get_param(
            '/task1_v2_manual_slow_lateral_error', 0.25
        )
        self.manual_tx_sign = rospy.get_param('/task1_v2_manual_tx_sign', 1.0)
        self.manual_ty_sign = rospy.get_param('/task1_v2_manual_ty_sign', 1.0)
        self.marker_action_distance = rospy.get_param('/task1_v2_marker_action_distance', 0.35)
        self.marker_slow_distance = rospy.get_param('/task1_v2_marker_slow_distance', 0.8)

        self.marker_sample_count = int(rospy.get_param('/task1_v2_marker_sample_count', 10))
        self.marker_cluster_distance = rospy.get_param('/task1_v2_marker_cluster_distance', 0.25)
        self.marker_ignore_distance = rospy.get_param('/task1_v2_marker_ignore_distance', 0.5)
        self.marker_arrival_tolerance = rospy.get_param('/task1_v2_marker_arrival_tolerance', 0.15)

        self.light_on_seconds = rospy.get_param('/task1_v2_light_on_seconds', 3.0)
        self.light_off_seconds = rospy.get_param('/task1_v2_light_off_seconds', 0.5)

        self.black_rotation_angle = math.radians(
            rospy.get_param('/task1_v2_black_rotation_angle_deg', 360.0)
        )
        self.rotation_yaw_step = math.radians(
            rospy.get_param('/task1_v2_rotation_yaw_step_deg', 3.0)
        )
        self.rotation_stop_margin = math.radians(
            rospy.get_param('/task1_v2_rotation_stop_margin_deg', 10.0)
        )

        self.yellow_classes = class_names('/task1_v2_yellow_classes', ['triangle', 'circle'])
        self.black_classes = class_names('/task1_v2_black_classes', ['rectangle'])
        self.line_classes = class_names('/task1_v2_line_classes', ['line'])

        self.initial_pose = None
        self.initial_yaw = None
        self.search_target = None

        self.line_target = None
        self.line_start_pose = None
        self.line_end_pose = None
        self.last_line_yaw = None
        self.last_line_time = None
        self.line_reacquire_until = None
        self.line_axis_origin = None
        self.line_axis_yaw = None
        self.line_raw_points = []
        self.line_curve_points = []
        self.line_curve_s = []
        self.current_path_s = 0.0
        self.completed_path_length = 0.0
        self.last_manual_tx = 0
        self.last_manual_ty = 0

        self.marker_clusters = []
        self.pending_defects = []
        self.active_defect = None
        self.handled_markers = []
        self.handled_counts = {'yellow': 0, 'black': 0}

        self.light_action_state = None
        self.rotation_feedback_state = None

        rospy.Subscriber('/obj/target_message', TargetDetection, self.defect_callback)
        rospy.Subscriber('/obj/line_message', TargetDetection3, self.line_callback)
        rospy.loginfo(
            '%s: initialized, reference_height=%.2f, target_z=%.2f, '
            'initial_heading=%.1fdeg, manual_force=(fast=%s, slow=%s), '
            'line_curve(max_raw=%d, sample=%d, degree=%d)',
            NODE_NAME,
            self.reference_height,
            self.reference_z,
            math.degrees(self.initial_search_yaw),
            self.manual_forward_force,
            self.manual_slow_forward_force,
            self.line_curve_max_points,
            self.line_curve_sample_count,
            self.line_curve_degree,
        )

    ############################################### 基础工具层 #######################################
    def step_name(self, step):
        """返回状态机步骤名称，便于日志定位。"""
        return self.STEP_NAMES.get(step, str(step))

    def set_step(self, step):
        """切换状态机步骤，并记录进入该步骤的时间。"""
        old_step = self.step
        elapsed = self.step_elapsed()
        self.step = step
        self.step_started = rospy.Time.now()
        if old_step != step:
            rospy.loginfo(
                '%s: step %s -> %s, previous_step_elapsed=%.1fs',
                NODE_NAME,
                self.step_name(old_step),
                self.step_name(step),
                elapsed,
            )

    def step_elapsed(self):
        """返回当前步骤已持续时间，单位为秒。"""
        return (rospy.Time.now() - self.step_started).to_sec()

    def get_current_pose(self):
        """从 TF 树读取 map -> base_link 当前位姿。"""
        try:
            self.tf_listener.waitForTransform(
                'map', 'base_link', rospy.Time(0), rospy.Duration(1.0)
            )
            translation, rotation = self.tf_listener.lookupTransform(
                'map', 'base_link', rospy.Time(0)
            )
        except tf.Exception as error:
            rospy.logwarn_throttle(2, '%s: cannot read AUV pose: %s', NODE_NAME, error)
            return None

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(*translation)
        pose.pose.orientation = Quaternion(*rotation)
        return pose

    @staticmethod
    def position_distance(first, second):
        """计算两个 Point 之间的三维欧氏距离。"""
        dx = first.x - second.x
        dy = first.y - second.y
        dz = first.z - second.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def publish_device(self, red=0, green=0, servo=None, light1=0, light2=0):
        """发布执行器控制消息到 /cmd/actuator。"""
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
        rospy.loginfo_throttle(
            1.0,
            '%s: actuator cmd red=%d green=%d light1=%d light2=%d '
            'heading_servo=%d clamp_servo=%d drive=(cmd=%d, speed=%d)',
            NODE_NAME,
            message.red_light,
            message.green_light,
            message.light1,
            message.light2,
            message.heading_servo,
            message.clamp_servo,
            message.drive_cmd,
            message.drive_speed,
        )

    @staticmethod
    def force_value(value):
        """将手控力/力矩限制到 int16 安全范围。"""
        return int(round(clamp(value, -10000, 10000)))

    def publish_pose_cmd(self, mode, target, tx=0, ty=0, tz=0, mx=0, my=0, mz=0):
        """按 PoseNEDcmd 格式发布运动控制指令。"""
        command = PoseNEDcmd()
        command.mode = int(mode)
        command.target = copy.deepcopy(target)
        command.target.header.frame_id = 'map'
        command.target.header.stamp = rospy.Time.now()
        command.force.TX = self.force_value(tx)
        command.force.TY = self.force_value(ty)
        command.force.TZ = self.force_value(tz)
        command.force.MX = self.force_value(mx)
        command.force.MY = self.force_value(my)
        command.force.MZ = self.force_value(mz)
        self.pose_cmd_pub.publish(command)
        rospy.loginfo_throttle(
            1.0,
            '%s: pose cmd mode=%d(%s) target=(%.2f, %.2f, %.2f, yaw=%.1fdeg) '
            'force=(TX=%d, TY=%d, TZ=%d, MX=%d, MY=%d, MZ=%d)',
            NODE_NAME,
            command.mode,
            MODE_NAMES.get(command.mode, 'unknown'),
            command.target.pose.position.x,
            command.target.pose.position.y,
            command.target.pose.position.z,
            math.degrees(yaw_from_quaternion(command.target.pose.orientation)),
            command.force.TX,
            command.force.TY,
            command.force.TZ,
            command.force.MX,
            command.force.MY,
            command.force.MZ,
        )

    def publish_stop_cmd(self):
        """发布零力手控指令，停止巡线阶段的开环推进。"""
        current = self.get_current_pose()
        if current is None:
            yaw = self.initial_yaw if self.initial_yaw is not None else 0.0
            target = self.make_level_pose(0.0, 0.0, yaw)
        else:
            yaw = yaw_from_quaternion(current.pose.orientation)
            target = self.make_level_pose(
                current.pose.position.x,
                current.pose.position.y,
                yaw,
            )
        self.publish_pose_cmd(MODE_DEPTH_HDG, target)
        self.last_manual_tx = 0
        self.last_manual_ty = 0

    def finish_task(self):
        """关闭执行器并发布任务完成消息。"""
        self.publish_stop_cmd()
        self.publish_device()
        self.finished_pub.publish('{} finished'.format(NODE_NAME))
        rospy.loginfo('%s: mission finished', NODE_NAME)
        rospy.signal_shutdown('mission finished')

    def wait_for_current_pose(self, timeout=5.0):
        """等待并返回当前 map -> base_link 位姿；超时返回 None。"""
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            current = self.get_current_pose()
            if current is not None:
                return current
            rospy.sleep(0.1)
        return None

    def initialize_mission_frame(self):
        """记录任务启动位姿和参考运行高度。"""
        if self.initial_pose is not None:
            return True

        current = self.wait_for_current_pose()
        if current is None:
            rospy.logwarn_throttle(2, '%s: waiting for current pose', NODE_NAME)
            return False

        self.initial_pose = copy.deepcopy(current)
        self.initial_yaw = yaw_from_quaternion(current.pose.orientation)

        rospy.loginfo(
            '%s: mission origin pose recorded at x=%.2f, y=%.2f, z=%.2f, '
            'current_yaw=%.1fdeg, search_yaw=%.1fdeg; reference_height=%.2f, target_z=%.2f',
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(self.initial_yaw),
            math.degrees(self.initial_search_yaw),
            self.reference_height,
            self.reference_z,
        )
        return True

    def make_pose(self, x, y, z, yaw):
        """生成 map 坐标系下的目标位姿。"""
        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rospy.Time.now()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        target.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0, self.pitch_offset, yaw
        ))
        return target

    def make_level_pose(self, x, y, yaw):
        """生成固定参考运行高度的目标位姿。"""
        return self.make_pose(x, y, self.reference_z, yaw)

    def publish_level_target(self, x, y, yaw):
        """以动力定位 ROV 模式发布固定参考运行高度目标。"""
        self.publish_pose_cmd(MODE_DPROV, self.make_level_pose(x, y, yaw))

    def move_to_pose_level(self, target, position_tolerance=None, yaw_tolerance=None):
        """先控制航向，再控制 XY 位置，z 固定为参考运行高度。"""
        current = self.get_current_pose()
        if current is None:
            return False

        if position_tolerance is None:
            position_tolerance = self.position_tolerance
        if yaw_tolerance is None:
            yaw_tolerance = self.yaw_tolerance

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        dx = target.pose.position.x - current.pose.position.x
        dy = target.pose.position.y - current.pose.position.y
        horizontal_distance = math.hypot(dx, dy)
        final_yaw = yaw_from_quaternion(target.pose.orientation)

        if horizontal_distance > position_tolerance:
            move_yaw = math.atan2(dy, dx)
            yaw_error = wrap_angle(move_yaw - current_yaw)
            if abs(yaw_error) > yaw_tolerance:
                commanded_yaw = current_yaw + clamp(
                    yaw_error, -self.max_yaw_step, self.max_yaw_step
                )
                rospy.loginfo_throttle(
                    1.0,
                    '%s: DPROV align to move target distance=%.2f yaw_error=%.1fdeg',
                    NODE_NAME,
                    horizontal_distance,
                    math.degrees(yaw_error),
                )
                self.publish_level_target(
                    current.pose.position.x,
                    current.pose.position.y,
                    commanded_yaw,
                )
                return False

            step = min(self.max_xy_step, horizontal_distance)
            scale = step / horizontal_distance
            rospy.loginfo_throttle(
                1.0,
                '%s: DPROV move to target distance=%.2f step=%.2f target=(%.2f, %.2f)',
                NODE_NAME,
                horizontal_distance,
                step,
                target.pose.position.x,
                target.pose.position.y,
            )
            self.publish_level_target(
                current.pose.position.x + dx * scale,
                current.pose.position.y + dy * scale,
                move_yaw,
            )
            return False

        yaw_error = wrap_angle(final_yaw - current_yaw)
        if abs(yaw_error) > yaw_tolerance:
            commanded_yaw = current_yaw + clamp(yaw_error, -self.max_yaw_step, self.max_yaw_step)
            rospy.loginfo_throttle(
                1.0,
                '%s: DPROV final yaw align yaw_error=%.1fdeg target=(%.2f, %.2f)',
                NODE_NAME,
                math.degrees(yaw_error),
                target.pose.position.x,
                target.pose.position.y,
            )
            self.publish_level_target(
                target.pose.position.x,
                target.pose.position.y,
                commanded_yaw,
            )
            return False

        self.publish_level_target(
            target.pose.position.x,
            target.pose.position.y,
            final_yaw,
        )
        return True

    def hold_active_pose(self):
        """保持当前动作位姿，避免亮灯或旋转时目标点丢失。"""
        if self.active_defect is not None:
            self.active_defect['action'].header.stamp = rospy.Time.now()
            self.publish_pose_cmd(MODE_DPROV, self.active_defect['action'])
            return

        current = self.get_current_pose()
        if current is None:
            return
        current_yaw = yaw_from_quaternion(current.pose.orientation)
        self.publish_level_target(current.pose.position.x, current.pose.position.y, current_yaw)

    ############################################### 管线处理层 #######################################
    def transform_pose_to_map(self, pose):
        """将任意检测位姿转换到 map 坐标系。"""
        try:
            self.tf_listener.waitForTransform(
                'map',
                pose.header.frame_id,
                pose.header.stamp,
                rospy.Duration(1.0),
            )
            return self.tf_listener.transformPose('map', pose)
        except tf.Exception as error:
            rospy.logwarn_throttle(2, '%s: transform failed: %s', NODE_NAME, error)
            return None

    def line_axis_s(self, point):
        """计算点在初始红线主方向上的投影进度，用于稳定排序。"""
        if self.line_axis_origin is None or self.line_axis_yaw is None:
            return 0.0
        dx = point.x - self.line_axis_origin.x
        dy = point.y - self.line_axis_origin.y
        return dx * math.cos(self.line_axis_yaw) + dy * math.sin(self.line_axis_yaw)

    def downsample_line_points(self):
        """限制原始红线点数量，避免长时间运行后点集过大。"""
        if len(self.line_raw_points) <= self.line_curve_max_points:
            return
        ordered = sorted(self.line_raw_points, key=self.line_axis_s)
        keep_count = max(2, self.line_curve_max_points)
        indexes = [
            int(round(index * (len(ordered) - 1) / float(keep_count - 1)))
            for index in range(keep_count)
        ]
        self.line_raw_points = [copy.deepcopy(ordered[index]) for index in indexes]
        rospy.loginfo(
            '%s: line raw points downsampled to %d points',
            NODE_NAME,
            len(self.line_raw_points),
        )

    def add_line_map_point(self, point):
        """将红线点加入全局点集；近距离点做平滑融合。"""
        new_point = Point(point.x, point.y, self.reference_z)
        for old_point in self.line_raw_points:
            if xy_distance(old_point, new_point) <= self.line_point_merge_distance:
                old_point.x = 0.8 * old_point.x + 0.2 * new_point.x
                old_point.y = 0.8 * old_point.y + 0.2 * new_point.y
                old_point.z = self.reference_z
                self.line_raw_points.sort(key=self.line_axis_s)
                return
        self.line_raw_points.append(new_point)
        self.line_raw_points.sort(key=self.line_axis_s)
        self.downsample_line_points()

    @staticmethod
    def cumulative_distance(points):
        """计算点列的累计水平距离。"""
        distances = [0.0]
        for index in range(1, len(points)):
            distances.append(distances[-1] + xy_distance(points[index], points[index - 1]))
        return distances

    def fit_line_curve(self):
        """将有限红线点集拟合为平滑曲线，并采样为固定数量控制点。"""
        if len(self.line_raw_points) < 2:
            return

        ordered = sorted(self.line_raw_points, key=self.line_axis_s)
        filtered = [ordered[0]]
        for point in ordered[1:]:
            if xy_distance(point, filtered[-1]) > 1e-3:
                filtered.append(point)

        if len(filtered) < 2:
            return

        raw_s = self.cumulative_distance(filtered)
        total_length = raw_s[-1]
        if total_length < self.line_curve_min_length:
            self.line_curve_points = [copy.deepcopy(point) for point in filtered]
            self.line_curve_s = raw_s
            return

        degree = min(max(1, self.line_curve_degree), len(filtered) - 1)
        xs = [point.x for point in filtered]
        ys = [point.y for point in filtered]
        sample_count = max(2, self.line_curve_sample_count)

        try:
            x_curve = np.poly1d(np.polyfit(raw_s, xs, degree))
            y_curve = np.poly1d(np.polyfit(raw_s, ys, degree))
            sample_s = np.linspace(0.0, total_length, sample_count)
            fitted = []
            for value in sample_s:
                fitted.append(Point(float(x_curve(value)), float(y_curve(value)), self.reference_z))
            self.line_curve_points = fitted
            self.line_curve_s = self.cumulative_distance(self.line_curve_points)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            rospy.logwarn_throttle(2, '%s: line curve fit failed: %s', NODE_NAME, error)
            self.line_curve_points = [copy.deepcopy(point) for point in filtered]
            self.line_curve_s = raw_s

    def update_line_curve(self, poses):
        """用一帧红线起点/中点/终点更新全局曲线。"""
        first, second, third = poses
        if self.line_axis_origin is None:
            self.line_axis_origin = copy.deepcopy(first.pose.position)
            self.line_axis_yaw = math.atan2(
                third.pose.position.y - first.pose.position.y,
                third.pose.position.x - first.pose.position.x,
            )
            rospy.loginfo(
                '%s: line axis initialized origin=(%.2f, %.2f), yaw=%.1fdeg',
                NODE_NAME,
                self.line_axis_origin.x,
                self.line_axis_origin.y,
                math.degrees(self.line_axis_yaw),
            )

        self.add_line_map_point(first.pose.position)
        self.add_line_map_point(second.pose.position)
        self.add_line_map_point(third.pose.position)
        self.fit_line_curve()
        curve_length = self.line_curve_s[-1] if self.line_curve_ready() else 0.0
        rospy.loginfo_throttle(
            1.0,
            '%s: line fusion raw_points=%d curve_points=%d known_curve=%.2fm '
            'line_yaw=%.1fdeg',
            NODE_NAME,
            len(self.line_raw_points),
            len(self.line_curve_points),
            curve_length,
            math.degrees(self.line_axis_yaw if self.line_axis_yaw is not None else 0.0),
        )

    def line_curve_ready(self):
        """判断当前是否已有可用于 LOS 的拟合曲线。"""
        return len(self.line_curve_points) >= 2 and len(self.line_curve_s) == len(
            self.line_curve_points
        )

    def project_to_line_curve(self, point):
        """将机器人当前位置投影到拟合曲线，返回路径进度和横向误差。"""
        if not self.line_curve_ready():
            return None

        best = None
        for index in range(len(self.line_curve_points) - 1):
            start = self.line_curve_points[index]
            end = self.line_curve_points[index + 1]
            vx = end.x - start.x
            vy = end.y - start.y
            segment_sq = vx * vx + vy * vy
            if segment_sq < 1e-9:
                continue

            wx = point.x - start.x
            wy = point.y - start.y
            ratio = clamp((wx * vx + wy * vy) / segment_sq, 0.0, 1.0)
            proj_x = start.x + ratio * vx
            proj_y = start.y + ratio * vy
            dx = point.x - proj_x
            dy = point.y - proj_y
            distance = math.hypot(dx, dy)
            segment_length = math.sqrt(segment_sq)
            signed_lateral = (vx * (point.y - start.y) - vy * (point.x - start.x)) / (
                segment_length
            )
            path_s = self.line_curve_s[index] + ratio * segment_length
            segment_yaw = math.atan2(vy, vx)

            if best is None or distance < best['distance']:
                best = {
                    'distance': distance,
                    'lateral': signed_lateral,
                    'path_s': path_s,
                    'segment_yaw': segment_yaw,
                    'projection': Point(proj_x, proj_y, self.reference_z),
                }
        return best

    def point_at_curve_s(self, target_s):
        """按路径进度从拟合曲线上插值得到前视目标点。"""
        if not self.line_curve_ready():
            return None

        target_s = clamp(target_s, 0.0, self.line_curve_s[-1])
        for index in range(len(self.line_curve_s) - 1):
            start_s = self.line_curve_s[index]
            end_s = self.line_curve_s[index + 1]
            if target_s > end_s:
                continue

            start = self.line_curve_points[index]
            end = self.line_curve_points[index + 1]
            if end_s <= start_s:
                return copy.deepcopy(start)
            ratio = (target_s - start_s) / (end_s - start_s)
            return Point(
                start.x + ratio * (end.x - start.x),
                start.y + ratio * (end.y - start.y),
                self.reference_z,
            )
        return copy.deepcopy(self.line_curve_points[-1])

    def near_curve_end(self):
        """判断机器人是否已经接近当前已知曲线末端。"""
        if not self.line_curve_ready():
            return False
        return self.current_path_s >= self.line_curve_s[-1] - self.line_end_margin

    def curve_blind_follow_allowed(self):
        """红线短暂丢失时，允许沿已知曲线低速继续一小段。"""
        if self.last_line_time is None:
            return False
        return (
            rospy.Time.now() - self.last_line_time
        ).to_sec() <= self.curve_blind_follow_timeout

    def limit_manual_force(self, desired, previous):
        """限制手控力单周期变化量，降低惯性过冲。"""
        return int(round(clamp(
            desired,
            previous - self.manual_force_step,
            previous + self.manual_force_step,
        )))

    def nearest_pending_marker_distance(self):
        """返回最近待处理图形到机器人当前位置的水平距离。"""
        if not self.pending_defects:
            return None
        current = self.get_current_pose()
        if current is None:
            return None
        distances = [
            xy_distance(current.pose.position, defect['marker'].pose.position)
            for defect in self.pending_defects
        ]
        return min(distances) if distances else None

    def ready_defect_index(self):
        """判断是否已经到达某个稳定图形附近，可以切换到动作阶段。"""
        if not self.pending_defects:
            return None
        current = self.get_current_pose()
        if current is None:
            return None

        best_index = None
        best_distance = None
        for index, defect in enumerate(self.pending_defects):
            distance = xy_distance(current.pose.position, defect['marker'].pose.position)
            if distance <= self.marker_action_distance:
                if best_distance is None or distance < best_distance:
                    best_index = index
                    best_distance = distance
        return best_index

    def follow_line_curve_manual(self):
        """使用拟合曲线 LOS 计算定深定向手控指令。"""
        current = self.get_current_pose()
        if current is None or not self.line_curve_ready():
            self.publish_stop_cmd()
            return False

        projection = self.project_to_line_curve(current.pose.position)
        if projection is None:
            self.publish_stop_cmd()
            return False

        self.current_path_s = projection['path_s']
        self.completed_path_length = max(self.completed_path_length, self.current_path_s)
        los_target = self.point_at_curve_s(self.current_path_s + self.los_lookahead_distance)
        if los_target is None:
            self.publish_stop_cmd()
            return False

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        desired_yaw = math.atan2(
            los_target.y - current.pose.position.y,
            los_target.x - current.pose.position.x,
        )
        yaw_error = wrap_angle(desired_yaw - current_yaw)
        lateral_error = projection['lateral']

        forward_force = self.manual_forward_force
        marker_distance = self.nearest_pending_marker_distance()
        line_age = (
            (rospy.Time.now() - self.last_line_time).to_sec()
            if self.last_line_time is not None else -1.0
        )
        if (
            abs(yaw_error) > self.manual_slow_yaw_error
            or abs(lateral_error) > self.manual_slow_lateral_error
            or (marker_distance is not None and marker_distance < self.marker_slow_distance)
            or not self.line_is_recent()
        ):
            forward_force = self.manual_slow_forward_force

        desired_tx = self.manual_tx_sign * forward_force
        desired_ty = self.manual_ty_sign * clamp(
            -self.manual_lateral_gain * lateral_error,
            -self.manual_max_lateral_force,
            self.manual_max_lateral_force,
        )
        tx = self.limit_manual_force(desired_tx, self.last_manual_tx)
        ty = self.limit_manual_force(desired_ty, self.last_manual_ty)
        self.last_manual_tx = tx
        self.last_manual_ty = ty

        target = self.make_level_pose(
            current.pose.position.x,
            current.pose.position.y,
            desired_yaw,
        )
        self.publish_pose_cmd(MODE_DEPTH_HDG, target, tx=tx, ty=ty)

        rospy.loginfo_throttle(
            1.0,
            '%s: LOS manual mode=3 completed=%.2fm known_curve=%.2fm lateral=%.2f '
            'yaw_error=%.1fdeg line_age=%.1fs raw_points=%d curve_points=%d '
            'pending=%d nearest_marker=%.2f force=(%d,%d)',
            NODE_NAME,
            self.completed_path_length,
            self.line_curve_s[-1],
            lateral_error,
            math.degrees(yaw_error),
            line_age,
            len(self.line_raw_points),
            len(self.line_curve_points),
            len(self.pending_defects),
            marker_distance if marker_distance is not None else -1.0,
            tx,
            ty,
        )
        return True

    def line_callback(self, message):
        """接收红色长线识别结果，并更新巡线目标。"""
        if message.class_name and message.class_name not in self.line_classes:
            return

        first = self.transform_pose_to_map(message.pose1)
        second = self.transform_pose_to_map(message.pose2)
        third = self.transform_pose_to_map(message.pose3)
        if first is None or second is None or third is None:
            return

        current = self.get_current_pose()
        if current is None:
            return

        line_dx = third.pose.position.x - second.pose.position.x
        line_dy = third.pose.position.y - second.pose.position.y
        if math.hypot(line_dx, line_dy) < 1e-6:
            return

        if self.line_start_pose is None:
            self.line_start_pose = copy.deepcopy(first)
            rospy.loginfo(
                '%s: red line start recorded at x=%.2f, y=%.2f, z=%.2f',
                NODE_NAME,
                first.pose.position.x,
                first.pose.position.y,
                first.pose.position.z,
            )

        self.line_end_pose = copy.deepcopy(third)
        self.last_line_time = rospy.Time.now()
        self.line_reacquire_until = None
        self.last_line_yaw = math.atan2(line_dy, line_dx)
        self.update_line_curve((first, second, third))
        rospy.loginfo_throttle(
            1.0,
            '%s: line detection class=%s first=(%.2f, %.2f) mid=(%.2f, %.2f) '
            'end=(%.2f, %.2f) yaw=%.1fdeg',
            NODE_NAME,
            message.class_name,
            first.pose.position.x,
            first.pose.position.y,
            second.pose.position.x,
            second.pose.position.y,
            third.pose.position.x,
            third.pose.position.y,
            math.degrees(self.last_line_yaw),
        )

        target_x = current.pose.position.x + self.line_forward_fraction * (
            second.pose.position.x - current.pose.position.x
        )
        target_y = current.pose.position.y + self.line_forward_fraction * (
            second.pose.position.y - current.pose.position.y
        )
        self.line_target = self.make_level_pose(target_x, target_y, self.last_line_yaw)

    def line_is_recent(self):
        """判断红线识别是否仍然有效。"""
        if self.last_line_time is None:
            return False
        return (rospy.Time.now() - self.last_line_time).to_sec() <= self.line_lost_timeout

    def search_line_forward(self):
        """未看到红线时，沿任务启动时的前方小步搜索。"""
        if self.initial_pose is None or self.initial_yaw is None:
            return

        if self.search_target is None:
            current = self.get_current_pose()
            if current is None:
                return
            search_yaw = self.initial_search_yaw
            if self.line_start_pose is not None and self.last_line_yaw is not None:
                search_yaw = self.last_line_yaw
            self.search_target = self.make_level_pose(
                current.pose.position.x + self.search_forward_step * math.cos(search_yaw),
                current.pose.position.y + self.search_forward_step * math.sin(search_yaw),
                search_yaw,
            )
            rospy.loginfo(
                '%s: search line target=(%.2f, %.2f, yaw=%.1fdeg)',
                NODE_NAME,
                self.search_target.pose.position.x,
                self.search_target.pose.position.y,
                math.degrees(search_yaw),
            )

        if self.move_to_pose_level(self.search_target):
            self.search_target = None

    ############################################### 图形处理层 #######################################
    def marker_already_known(self, point, defect_type):
        """判断同类图形是否已入队、正在处理或已完成。"""
        for marker in self.handled_markers:
            if marker['type'] == defect_type and xy_distance(point, marker['point']) < (
                self.marker_ignore_distance
            ):
                return True

        for defect in self.pending_defects:
            if defect['type'] == defect_type and xy_distance(
                point, defect['marker'].pose.position
            ) < self.marker_ignore_distance:
                return True

        if self.active_defect is not None:
            if self.active_defect['type'] == defect_type and xy_distance(
                point, self.active_defect['marker'].pose.position
            ) < self.marker_ignore_distance:
                return True

        return False

    def defect_callback(self, message):
        """接收图形识别结果，聚类稳定后加入动作队列。"""
        if message.type and message.type != 'center':
            return

        if message.class_name in self.yellow_classes:
            defect_type = 'yellow'
        elif message.class_name in self.black_classes:
            defect_type = 'black'
        else:
            return

        camera_origin = Point()
        if self.position_distance(message.pose.pose.position, camera_origin) > 5.0:
            rospy.loginfo_throttle(
                2.0,
                '%s: ignore far marker class=%s camera_pos=(%.2f, %.2f, %.2f)',
                NODE_NAME,
                message.class_name,
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            )
            return

        rospy.loginfo_throttle(
            1.0,
            '%s: marker detection class=%s defect=%s conf=%.2f camera_pos=(%.2f, %.2f, %.2f)',
            NODE_NAME,
            message.class_name,
            defect_type,
            message.conf,
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            message.pose.pose.position.z,
        )

        marker = self.transform_pose_to_map(message.pose)
        if marker is None:
            return
        if self.marker_already_known(marker.pose.position, defect_type):
            rospy.loginfo_throttle(
                2.0,
                '%s: ignore known %s marker at x=%.2f, y=%.2f',
                NODE_NAME,
                defect_type,
                marker.pose.position.x,
                marker.pose.position.y,
            )
            return

        self.add_marker_sample(defect_type, marker, message.conf)

    def add_marker_sample(self, defect_type, marker, confidence):
        """用多帧识别结果确定一个稳定图形位置。"""
        point = marker.pose.position
        for cluster in self.marker_clusters:
            if cluster['type'] != defect_type:
                continue
            if xy_distance(point, cluster['center']) > self.marker_cluster_distance:
                continue

            cluster['samples'].append(copy.deepcopy(marker))
            cluster['confidences'].append(confidence)
            cluster['center'] = self.cluster_center(cluster['samples'])
            rospy.loginfo_throttle(
                1.0,
                '%s: marker cluster %s samples=%d/%d center=(%.2f, %.2f)',
                NODE_NAME,
                defect_type,
                len(cluster['samples']),
                self.marker_sample_count,
                cluster['center'].x,
                cluster['center'].y,
            )
            self.queue_marker_if_stable(cluster)
            return

        cluster = {
            'type': defect_type,
            'samples': [copy.deepcopy(marker)],
            'confidences': [confidence],
            'center': copy.deepcopy(marker.pose.position),
            'queued': False,
        }
        self.marker_clusters.append(cluster)
        rospy.loginfo(
            '%s: new marker cluster %s at x=%.2f, y=%.2f, conf=%.2f',
            NODE_NAME,
            defect_type,
            point.x,
            point.y,
            confidence,
        )
        self.queue_marker_if_stable(cluster)

    def cluster_center(self, samples):
        """使用中位数计算图形聚类中心。"""
        center = Point()
        center.x = median([sample.pose.position.x for sample in samples])
        center.y = median([sample.pose.position.y for sample in samples])
        center.z = median([sample.pose.position.z for sample in samples])
        return center

    def queue_marker_if_stable(self, cluster):
        """当图形连续稳定识别达到阈值后，生成动作并入队。"""
        if cluster['queued'] or len(cluster['samples']) < self.marker_sample_count:
            return

        if self.marker_already_known(cluster['center'], cluster['type']):
            cluster['queued'] = True
            return

        marker = copy.deepcopy(cluster['samples'][-1])
        marker.pose.position = copy.deepcopy(cluster['center'])

        self.pending_defects.append({
            'type': cluster['type'],
            'action': None,
            'marker': marker,
            'confidence': sum(cluster['confidences']) / len(cluster['confidences']),
        })
        cluster['queued'] = True
        rospy.loginfo(
            '%s: queued stable %s marker at x=%.2f, y=%.2f, z=%.2f, samples=%d',
            NODE_NAME,
            cluster['type'],
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
            len(cluster['samples']),
        )

    def action_pose_from_current(self):
        """生成当前动作保持位姿；图形只触发动作，不作为导航目标。"""
        current = self.get_current_pose()
        if current is None:
            return None

        return self.make_level_pose(
            current.pose.position.x,
            current.pose.position.y,
            yaw_from_quaternion(current.pose.orientation),
        )

    def complete_active_defect(self):
        """记录当前图形动作已完成，并返回巡线。"""
        self.handled_markers.append({
            'type': self.active_defect['type'],
            'point': copy.deepcopy(self.active_defect['marker'].pose.position),
        })
        self.handled_counts[self.active_defect['type']] += 1
        rospy.loginfo(
            '%s: completed %s marker, yellow=%d, black=%d',
            NODE_NAME,
            self.active_defect['type'],
            self.handled_counts['yellow'],
            self.handled_counts['black'],
        )
        self.active_defect = None
        self.light_action_state = None
        self.rotation_feedback_state = None
        if self.line_start_pose is not None and not self.line_is_recent():
            self.line_target = None
            self.search_target = None
            self.line_reacquire_until = rospy.Time.now() + rospy.Duration(
                self.line_lost_timeout
            )
        self.set_step(self.STEP_FOLLOW_LINE)

    ############################################### 动作执行层 #######################################
    def run_light_action(self):
        """按图形类型执行灯光动作；每次亮灯持续 light_on_seconds。"""
        if self.active_defect is None:
            return True

        defect_type = self.active_defect['type']
        if self.light_action_state is None:
            self.light_action_state = {
                'count': 1 if defect_type == 'yellow' else 2,
                'red': 1 if defect_type == 'yellow' else 0,
                'green': 0 if defect_type == 'yellow' else 1,
            }
            rospy.loginfo(
                '%s: light action start type=%s count=%d light_on=%.1fs light_off=%.1fs',
                NODE_NAME,
                defect_type,
                self.light_action_state['count'],
                self.light_on_seconds,
                self.light_off_seconds,
            )

        self.hold_active_pose()

        elapsed = self.step_elapsed()
        cycle = self.light_on_seconds + self.light_off_seconds
        current_count = int(elapsed // cycle)
        if current_count >= self.light_action_state['count']:
            self.publish_device(red=0, green=0)
            rospy.loginfo(
                '%s: light action completed type=%s elapsed=%.1fs',
                NODE_NAME,
                defect_type,
                elapsed,
            )
            return True

        in_cycle = elapsed - current_count * cycle
        if in_cycle < self.light_on_seconds:
            self.publish_device(
                red=self.light_action_state['red'],
                green=self.light_action_state['green'],
            )
            rospy.loginfo_throttle(
                1.0,
                '%s: light action running type=%s cycle=%d/%d elapsed=%.1fs red=%d green=%d',
                NODE_NAME,
                defect_type,
                current_count + 1,
                self.light_action_state['count'],
                elapsed,
                self.light_action_state['red'],
                self.light_action_state['green'],
            )
        else:
            self.publish_device(red=0, green=0)
            rospy.loginfo_throttle(
                1.0,
                '%s: light action off-gap type=%s cycle=%d/%d elapsed=%.1fs',
                NODE_NAME,
                defect_type,
                current_count + 1,
                self.light_action_state['count'],
                elapsed,
            )
        return False

    def rotate_black_by_feedback(self):
        """根据当前航向反馈累计旋转角度，完成黑色图形旋转动作。"""
        current = self.get_current_pose()
        if current is None or self.active_defect is None:
            return False

        current_yaw = yaw_from_quaternion(current.pose.orientation)
        if self.rotation_feedback_state is None:
            self.rotation_feedback_state = {
                'last_yaw': current_yaw,
                'accumulated': 0.0,
                'direction': 1.0,
            }
            rospy.loginfo(
                '%s: black marker rotation start current_yaw=%.1fdeg target_angle=%.1fdeg',
                NODE_NAME,
                math.degrees(current_yaw),
                math.degrees(self.black_rotation_angle),
            )

        state = self.rotation_feedback_state
        delta = wrap_angle(current_yaw - state['last_yaw'])
        if delta * state['direction'] > 0.0:
            state['accumulated'] += abs(delta)
        state['last_yaw'] = current_yaw

        finish_angle = max(0.0, self.black_rotation_angle - self.rotation_stop_margin)
        if state['accumulated'] >= finish_angle:
            self.publish_device(red=0, green=0)
            self.hold_active_pose()
            rospy.loginfo(
                '%s: black marker rotation completed, accumulated=%.1f deg',
                NODE_NAME,
                math.degrees(state['accumulated']),
            )
            return True

        commanded_yaw = wrap_angle(current_yaw + state['direction'] * self.rotation_yaw_step)
        action = self.active_defect['action']
        self.publish_level_target(
            action.pose.position.x,
            action.pose.position.y,
            commanded_yaw,
        )
        rospy.loginfo_throttle(
            1.0,
            '%s: black marker rotating, accumulated=%.1f/%.1f deg',
            NODE_NAME,
            math.degrees(state['accumulated']),
            math.degrees(self.black_rotation_angle),
        )
        return False

    ############################################### 结束处理层 #######################################
    def log_line_end(self):
        """记录任务结束时可用的红线终点。"""
        if self.line_end_pose is None:
            rospy.logwarn('%s: finished without red line end pose', NODE_NAME)
            return

        rospy.loginfo(
            '%s: red line end recorded at x=%.2f, y=%.2f, z=%.2f',
            NODE_NAME,
            self.line_end_pose.pose.position.x,
            self.line_end_pose.pose.position.y,
            self.line_end_pose.pose.position.z,
        )

    ############################################### 主循环 ###########################################
    def run(self):
        """执行“起点建系→寻找红线→巡线→图形动作→红线结束”的任务流程。"""
        while not rospy.is_shutdown():
            if not self.initialize_mission_frame():
                self.rate.sleep()
                continue

            # 步骤0：从红色圆形原点向前搜索红色长线。
            if self.step == self.STEP_SEARCH_LINE:
                if self.line_curve_ready() and self.line_is_recent():
                    rospy.loginfo('%s: red line found, start line following', NODE_NAME)
                    self.set_step(self.STEP_FOLLOW_LINE)
                else:
                    self.search_line_forward()

            # 步骤1：正常巡线；若当前位置到达稳定图形上方，则停止巡线并执行动作。
            elif self.step == self.STEP_FOLLOW_LINE:
                ready_index = self.ready_defect_index()
                if ready_index is not None:
                    self.publish_stop_cmd()
                    self.active_defect = self.pending_defects.pop(ready_index)
                    self.active_defect['action'] = self.action_pose_from_current()
                    if self.active_defect['action'] is None:
                        rospy.logwarn('%s: cannot get current pose for marker action', NODE_NAME)
                        self.pending_defects.insert(ready_index, self.active_defect)
                        self.active_defect = None
                        self.rate.sleep()
                        continue
                    rospy.loginfo(
                        '%s: trigger %s marker action on path, marker=(%.2f, %.2f), '
                        'hold_pose=(%.2f, %.2f), queue_left=%d',
                        NODE_NAME,
                        self.active_defect['type'],
                        self.active_defect['marker'].pose.position.x,
                        self.active_defect['marker'].pose.position.y,
                        self.active_defect['action'].pose.position.x,
                        self.active_defect['action'].pose.position.y,
                        len(self.pending_defects),
                    )
                    self.light_action_state = None
                    self.rotation_feedback_state = None
                    self.set_step(self.STEP_LIGHT_ACTION)
                elif self.line_curve_ready() and (
                    self.line_is_recent() or self.curve_blind_follow_allowed()
                ):
                    self.follow_line_curve_manual()
                elif (
                    self.line_start_pose is not None
                    and self.line_reacquire_until is not None
                    and rospy.Time.now() <= self.line_reacquire_until
                ):
                    self.search_line_forward()
                elif self.line_start_pose is not None and self.near_curve_end():
                    self.publish_stop_cmd()
                    rospy.loginfo('%s: red line lost, task will finish', NODE_NAME)
                    self.set_step(self.STEP_FINISH)
                elif self.line_start_pose is not None:
                    self.publish_stop_cmd()
                    self.search_line_forward()
                else:
                    self.search_line_forward()

            # 步骤2：兼容保留；正常流程已经沿巡线到达图形上方，不再主动导航到图形坐标。
            elif self.step == self.STEP_MOVE_TO_MARKER:
                if self.active_defect is None:
                    self.set_step(self.STEP_FOLLOW_LINE)
                else:
                    if self.active_defect.get('action') is None:
                        self.active_defect['action'] = self.action_pose_from_current()
                    if self.active_defect.get('action') is None:
                        self.set_step(self.STEP_FOLLOW_LINE)
                        continue
                    rospy.loginfo('%s: marker action uses current hold pose', NODE_NAME)
                    self.set_step(self.STEP_LIGHT_ACTION)

            # 步骤3：执行灯光动作。
            elif self.step == self.STEP_LIGHT_ACTION:
                if self.run_light_action():
                    if self.active_defect is not None and self.active_defect['type'] == 'black':
                        self.rotation_feedback_state = None
                        self.set_step(self.STEP_ROTATE_BLACK)
                    else:
                        self.complete_active_defect()

            # 步骤4：黑色方形额外执行原地旋转，完成后返回巡线。
            elif self.step == self.STEP_ROTATE_BLACK:
                if self.rotate_black_by_feedback():
                    self.complete_active_defect()

            # 步骤5：记录红线终点并发布任务完成。
            elif self.step == self.STEP_FINISH:
                self.log_line_end()
                self.finish_task()

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task1V2().run()
    except rospy.ROSInterruptException:
        pass
