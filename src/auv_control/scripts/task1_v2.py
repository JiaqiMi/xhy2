#! /home/xhy/xhy_env/bin/python
"""
名称：task1_v2.py
功能：2026 Task 1——主管道检修
描述：
    1. 任务启动时记录当前位姿，按比赛约定认为 map 原点已经位于红色圆形处；
    2. 从起点向前搜索底部红色长线，首次看到管线时记录红线起点；
    3. 根据红色长线识别结果在 XY 平面巡线，运行高度统一使用参考高度；
    4. 巡线过程中稳定识别黄色圆形/三角形和黑色方形，并在图形上方执行动作；
    5. 黄色图形亮红灯，黑色图形亮绿灯并根据实时航向累计旋转角度；
    6. 红色长线丢失超过设定时间后记录终点并结束任务。
监听：/obj/target_message，/obj/line_message，/tf
发布：/target，/cmd/actuator，/finished
说明：
    - 识别类别按感知方话题格式默认使用 triangle、circle、rectangle、line；
    - /task1_v2_reference_height 可设置相对红色圆形原点的运行高度，默认 0.4 m；
      map 为 NED 坐标，z/down 向下为正，因此默认运行目标 z 为 -0.4；
    - 运动控制先调整航向指向目标点，再执行 XY 位置移动；
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
"""

import copy
import math

import rospy
import tf
from auv_control.msg import ActuatorControl, TargetDetection, TargetDetection3
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion, quaternion_from_euler


NODE_NAME = 'task1_v2'


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

    def __init__(self):
        """初始化任务参数、视觉缓存以及目标检测订阅。"""
        self.node_name = NODE_NAME
        self.target_pub = rospy.Publisher('/target', PoseStamped, queue_size=10)
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

        self.reference_height = float(rospy.get_param('/task1_v2_reference_height', 0.4))
        self.reference_z = float(rospy.get_param(
            '/task1_v2_reference_z', -self.reference_height
        ))

        self.search_forward_step = rospy.get_param('/task1_v2_search_forward_step', 0.3)
        self.line_forward_fraction = rospy.get_param('/task1_v2_line_forward_fraction', 0.8)
        self.line_lost_timeout = rospy.get_param('/task1_v2_line_lost_timeout', 5.0)

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

        self.marker_clusters = []
        self.pending_defects = []
        self.active_defect = None
        self.handled_markers = []
        self.handled_counts = {'yellow': 0, 'black': 0}

        self.light_action_state = None
        self.rotation_feedback_state = None

        rospy.Subscriber('/obj/target_message', TargetDetection, self.defect_callback)
        rospy.Subscriber('/obj/line_message', TargetDetection3, self.line_callback)
        rospy.loginfo('%s: initialized', NODE_NAME)

    ############################################### 基础工具层 #######################################
    def set_step(self, step):
        """切换状态机步骤，并记录进入该步骤的时间。"""
        self.step = step
        self.step_started = rospy.Time.now()

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

    def finish_task(self):
        """关闭执行器并发布任务完成消息。"""
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
            '%s: mission origin pose recorded at x=%.2f, y=%.2f, z=%.2f, yaw=%.1f deg; '
            'reference_height=%.2f, target_z=%.2f',
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
            math.degrees(self.initial_yaw),
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
        """发布固定参考运行高度目标。"""
        self.target_pub.publish(self.make_level_pose(x, y, yaw))

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
                self.publish_level_target(
                    current.pose.position.x,
                    current.pose.position.y,
                    commanded_yaw,
                )
                return False

            step = min(self.max_xy_step, horizontal_distance)
            scale = step / horizontal_distance
            self.publish_level_target(
                current.pose.position.x + dx * scale,
                current.pose.position.y + dy * scale,
                move_yaw,
            )
            return False

        yaw_error = wrap_angle(final_yaw - current_yaw)
        if abs(yaw_error) > yaw_tolerance:
            commanded_yaw = current_yaw + clamp(yaw_error, -self.max_yaw_step, self.max_yaw_step)
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
            self.target_pub.publish(self.active_defect['action'])
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
            search_yaw = self.initial_yaw
            if self.line_start_pose is not None and self.last_line_yaw is not None:
                search_yaw = self.last_line_yaw
            self.search_target = self.make_level_pose(
                current.pose.position.x + self.search_forward_step * math.cos(search_yaw),
                current.pose.position.y + self.search_forward_step * math.sin(search_yaw),
                search_yaw,
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
            return

        marker = self.transform_pose_to_map(message.pose)
        if marker is None or self.marker_already_known(marker.pose.position, defect_type):
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
        action = self.action_pose_from_marker(marker)
        if action is None:
            return

        self.pending_defects.append({
            'type': cluster['type'],
            'action': action,
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

    def action_pose_from_marker(self, marker):
        """生成图形上方动作位姿，XY 使用图形位置，z 使用参考运行高度。"""
        current = self.get_current_pose()
        if current is None:
            return None

        dx = marker.pose.position.x - current.pose.position.x
        dy = marker.pose.position.y - current.pose.position.y
        if math.hypot(dx, dy) > 0.05:
            desired_yaw = math.atan2(dy, dx)
        else:
            desired_yaw = yaw_from_quaternion(current.pose.orientation)

        return self.make_level_pose(marker.pose.position.x, marker.pose.position.y, desired_yaw)

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

        self.hold_active_pose()

        elapsed = self.step_elapsed()
        cycle = self.light_on_seconds + self.light_off_seconds
        current_count = int(elapsed // cycle)
        if current_count >= self.light_action_state['count']:
            self.publish_device(red=0, green=0)
            return True

        in_cycle = elapsed - current_count * cycle
        if in_cycle < self.light_on_seconds:
            self.publish_device(
                red=self.light_action_state['red'],
                green=self.light_action_state['green'],
            )
        else:
            self.publish_device(red=0, green=0)
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
                if self.line_target is not None and self.line_is_recent():
                    rospy.loginfo('%s: red line found, start line following', NODE_NAME)
                    self.set_step(self.STEP_FOLLOW_LINE)
                else:
                    self.search_line_forward()

            # 步骤1：正常巡线；若图形队列中有稳定目标，则优先执行动作。
            elif self.step == self.STEP_FOLLOW_LINE:
                if self.pending_defects:
                    self.active_defect = self.pending_defects.pop(0)
                    self.light_action_state = None
                    self.rotation_feedback_state = None
                    self.set_step(self.STEP_MOVE_TO_MARKER)
                elif self.line_target is not None and self.line_is_recent():
                    self.move_to_pose_level(self.line_target)
                elif (
                    self.line_start_pose is not None
                    and self.line_reacquire_until is not None
                    and rospy.Time.now() <= self.line_reacquire_until
                ):
                    self.search_line_forward()
                elif self.line_start_pose is not None:
                    rospy.loginfo('%s: red line lost, task will finish', NODE_NAME)
                    self.set_step(self.STEP_FINISH)
                else:
                    self.search_line_forward()

            # 步骤2：移动到图形正上方，移动过程中仍然保持参考运行高度。
            elif self.step == self.STEP_MOVE_TO_MARKER:
                if self.active_defect is None:
                    self.set_step(self.STEP_FOLLOW_LINE)
                elif self.move_to_pose_level(
                    self.active_defect['action'],
                    position_tolerance=self.marker_arrival_tolerance,
                ):
                    rospy.loginfo('%s: reached marker action pose', NODE_NAME)
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
