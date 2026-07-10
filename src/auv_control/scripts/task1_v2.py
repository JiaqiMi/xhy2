#! /home/xhy/xhy_env/bin/python
"""
名称：task1_v2.py
功能：2026 Task 1——主管道检修
描述：
    1. 移动到主管道入口并根据视觉检测结果沿管道前进；
    2. 识别黄色圆形/三角形泄漏标记和黑色方形污染标记；
    3. 当前测试阶段将黄色“接触”简化为当前位置下潜 10 cm；
    4. 黄色标记闪烁一次红灯；
    5. 黑色标记闪烁两次绿灯并原地旋转 360 度；
    6. 完成规定数量的两类标记后移动到管道终点。
监听：/obj/target_message，/obj/line_message，/tf
发布：/target，/auv_actuator_control，/finished
说明：比赛场地坐标、目标类别名称和标记数量通过 ROS 参数配置。
      测试时可将 /task1_v2_use_current_pose_as_start 设为 true，使任务以
      启动瞬间机器人当前位置作为入口点和返回终点。
"""

import copy
import math

import rospy
import tf
from auv_control.msg import TargetDetection, TargetDetection3
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

from task_v2_common import MissionBase, yaw_from_quaternion


NODE_NAME = 'task1_v2'


def class_names(param_name, defaults):
    """读取类别参数，并统一转换为字符串列表。"""
    value = rospy.get_param(param_name, defaults)
    if isinstance(value, str):
        return [value]
    return list(value)


class Task1V2(MissionBase):
    """主管道检修任务状态机。"""

    def __init__(self):
        """初始化任务参数、视觉队列以及目标检测订阅。"""
        super().__init__(NODE_NAME)

        self.entry_pose = self.pose_from_param('/task1_v2_entry_point', [0.0, 0.0, 0.3, 0.0])
        self.end_pose = self.pose_from_param('/task1_v2_end_point', [0.0, 0.0, 0.3, 0.0])
        self.target_depth = rospy.get_param('/task1_v2_depth', self.entry_pose.pose.position.z)
        self.use_current_pose_as_start = rospy.get_param(
            '/task1_v2_use_current_pose_as_start', False
        )
        self.required_yellow = int(rospy.get_param('/task1_v2_required_yellow', 2))
        self.required_black = int(rospy.get_param('/task1_v2_required_black', 2))
        self.touch_hold_seconds = rospy.get_param('/task1_v2_touch_hold_seconds', 1.0)
        self.yellow_dive_depth = rospy.get_param('/task1_v2_yellow_dive_depth', 0.10)
        self.marker_merge_distance = rospy.get_param('/task1_v2_marker_merge_distance', 0.5)
        self.line_forward_fraction = rospy.get_param('/task1_v2_line_forward_fraction', 0.8)
        self.yellow_classes = class_names(
            '/task1_v2_yellow_classes', ['yellow_circle', 'yellow_triangle', 'yellow']
        )
        self.black_classes = class_names(
            '/task1_v2_black_classes', ['black_square', 'black_rectangle', 'black']
        )

        self.line_target = None
        self.pending_defects = []
        self.handled_markers = []
        self.handled_counts = {'yellow': 0, 'black': 0}
        self.active_defect = None

        if self.use_current_pose_as_start:
            self.apply_current_pose_as_start()

        rospy.Subscriber('/obj/target_message', TargetDetection, self.defect_callback)
        rospy.Subscriber('/obj/line_message', TargetDetection3, self.line_callback)
        rospy.loginfo('%s: initialized', NODE_NAME)

    ############################################### 起点设置层 #######################################
    def apply_current_pose_as_start(self):
        """将任务启动时的当前位姿作为入口点、终点和巡线深度。"""
        current = self.wait_for_current_pose()
        if current is None:
            rospy.logwarn(
                '%s: cannot use current pose as start; fallback to configured points',
                NODE_NAME,
            )
            return

        self.entry_pose = copy.deepcopy(current)
        self.end_pose = copy.deepcopy(current)
        self.target_depth = current.pose.position.z
        rospy.loginfo(
            '%s: using current pose as test start/end: %.2f, %.2f, %.2f',
            NODE_NAME,
            current.pose.position.x,
            current.pose.position.y,
            current.pose.position.z,
        )

    def wait_for_current_pose(self, timeout=5.0):
        """等待并返回当前 map -> base_link 位姿；超时返回 None。"""
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            current = self.get_current_pose()
            if current is not None:
                return current
            rospy.sleep(0.1)
        return None

    ############################################### 回调辅助层 #######################################
    def marker_already_known(self, point):
        """判断标记是否已处理或已进入等待队列，避免重复响应同一标记。"""
        known_points = self.handled_markers + [
            defect['marker'].pose.position for defect in self.pending_defects
        ]
        if self.active_defect is not None:
            known_points.append(self.active_defect['marker'].pose.position)
        return any(
            self.position_distance(point, known) < self.marker_merge_distance
            for known in known_points
        )

    ############################################### 回调层 ###########################################
    def defect_callback(self, message):
        """接收缺陷目标，将有效且未重复的标记加入任务队列。

        视觉消息中的目标点先从 camera 转换到 map。当前真实下水通信测试
        还没有真实接触识别，因此黄色图形动作用当前位置下潜近似，黑色
        图形仍移动到图形坐标执行闪灯和旋转。
        """
        if message.class_name in self.yellow_classes:
            defect_type = 'yellow'
        elif message.class_name in self.black_classes:
            defect_type = 'black'
        else:
            return

        camera_origin = Point()
        if self.position_distance(message.pose.pose.position, camera_origin) > 5.0:
            return

        marker = self.marker_pose_in_map(message.pose)
        if marker is None or self.marker_already_known(marker.pose.position):
            return
        contact = self.action_pose_from_marker(marker, defect_type)
        if contact is None:
            return

        self.pending_defects.append({
            'type': defect_type,
            'contact': contact,
            'marker': marker,
            'confidence': message.conf,
        })
        rospy.loginfo(
            '%s: queued %s defect at %.2f, %.2f, %.2f',
            NODE_NAME,
            defect_type,
            marker.pose.position.x,
            marker.pose.position.y,
            marker.pose.position.z,
        )

    def marker_pose_in_map(self, detection_pose):
        """将视觉节点发布的 camera 坐标系图形位置转换到 map 坐标系。"""
        try:
            self.tf_listener.waitForTransform(
                'map', detection_pose.header.frame_id, detection_pose.header.stamp,
                rospy.Duration(1.0),
            )
            return self.tf_listener.transformPose('map', detection_pose)
        except tf.Exception as error:
            rospy.logwarn_throttle(2, '%s: marker transform failed: %s', NODE_NAME, error)
            return None

    def action_pose_from_marker(self, marker, defect_type):
        """根据图形位置生成当前测试用动作位姿。

        黄色图形：base_link 保持当前 XY 和航向，仅下潜 yellow_dive_depth；
        黑色图形：base_link 到达图形 XY，并在当前巡线深度执行动作。
        """
        current = self.get_current_pose()
        if current is None:
            return None

        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rospy.Time.now()
        if defect_type == 'yellow':
            target.pose.position = copy.deepcopy(current.pose.position)
            target.pose.position.z += self.yellow_dive_depth
            desired_yaw = yaw_from_quaternion(current.pose.orientation)
        else:
            dx = marker.pose.position.x - current.pose.position.x
            dy = marker.pose.position.y - current.pose.position.y
            if math.hypot(dx, dy) > 0.05:
                desired_yaw = math.atan2(dy, dx)
            else:
                desired_yaw = yaw_from_quaternion(current.pose.orientation)
            target.pose.position.x = marker.pose.position.x
            target.pose.position.y = marker.pose.position.y
            target.pose.position.z = self.target_depth
        target.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0, self.pitch_offset, desired_yaw
        ))
        return target

    def line_callback(self, message):
        """根据管线上的第二、第三检测点更新循线目标。

        第二个点用于确定下一位置，第二点指向第三点的方向作为期望航向，
        深度保持为 /task1_v2_depth。
        """
        try:
            self.tf_listener.waitForTransform(
                'map', message.pose2.header.frame_id, message.pose2.header.stamp,
                rospy.Duration(1.0),
            )
            second = self.tf_listener.transformPose('map', message.pose2)
            third = self.tf_listener.transformPose('map', message.pose3)
        except tf.Exception as error:
            rospy.logwarn_throttle(2, '%s: line transform failed: %s', NODE_NAME, error)
            return

        current = self.get_current_pose()
        if current is None:
            return

        line_dx = third.pose.position.x - second.pose.position.x
        line_dy = third.pose.position.y - second.pose.position.y
        if math.hypot(line_dx, line_dy) < 1e-6:
            return

        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rospy.Time.now()
        target.pose.position.x = current.pose.position.x + self.line_forward_fraction * (
            second.pose.position.x - current.pose.position.x
        )
        target.pose.position.y = current.pose.position.y + self.line_forward_fraction * (
            second.pose.position.y - current.pose.position.y
        )
        target.pose.position.z = self.target_depth
        target.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0, self.pitch_offset, math.atan2(line_dy, line_dx)
        ))
        self.line_target = target

    ############################################### 逻辑层 ###########################################
    def complete_active_defect(self):
        """记录当前缺陷已完成，并返回管道跟踪步骤。"""
        self.handled_markers.append(copy.deepcopy(self.active_defect['marker'].pose.position))
        self.handled_counts[self.active_defect['type']] += 1
        rospy.loginfo(
            '%s: completed %s defect (yellow=%d/%d, black=%d/%d)',
            NODE_NAME,
            self.active_defect['type'],
            self.handled_counts['yellow'],
            self.required_yellow,
            self.handled_counts['black'],
            self.required_black,
        )
        self.active_defect = None
        self.set_step(1)

    ############################################### 主循环 ###########################################
    def run(self):
        """按照“入口→循线→触碰→标记动作→终点”的顺序执行任务。"""
        while not rospy.is_shutdown():
            # 步骤0：移动到主管道入口。
            if self.step == 0:
                if self.move_to_pose(self.entry_pose):
                    rospy.loginfo('%s: reached pipeline entry', NODE_NAME)
                    self.set_step(1)

            # 步骤1：优先处理缺陷；无缺陷时继续循线或原地搜索。
            elif self.step == 1:
                if self.pending_defects:
                    self.active_defect = self.pending_defects.pop(0)
                    if self.active_defect['type'] == 'yellow':
                        contact = self.action_pose_from_marker(
                            self.active_defect['marker'], self.active_defect['type']
                        )
                        if contact is None:
                            self.active_defect = None
                            continue
                        self.active_defect['contact'] = contact
                    self.set_step(2)
                elif (
                    self.handled_counts['yellow'] >= self.required_yellow
                    and self.handled_counts['black'] >= self.required_black
                ):
                    self.set_step(6)
                elif self.line_target is not None:
                    self.move_to_pose(self.line_target)
                else:
                    self.sweep_for_target(max_angle_deg=20.0, step_deg=1.0)

            # 步骤2：移动到动作位姿；黄色图形保持当前XY并下潜10cm。
            elif self.step == 2:
                if self.move_to_pose(self.active_defect['contact'], position_tolerance=0.1):
                    rospy.loginfo('%s: reached marker action pose', NODE_NAME)
                    self.set_step(3)

            # 步骤3：保持动作位姿指定时间，避免瞬时到达造成误判。
            elif self.step == 3:
                self.hold_position()
                if self.step_elapsed() >= self.touch_hold_seconds:
                    self.set_step(4)

            # 步骤4：黄色闪一次红灯；黑色闪两次绿灯。
            elif self.step == 4:
                if self.active_defect['type'] == 'yellow':
                    finished = self.blink_lights(red=1, green=0, count=1)
                else:
                    finished = self.blink_lights(red=0, green=1, count=2)

                if finished:
                    if self.active_defect['type'] == 'black':
                        self.set_step(5)
                    else:
                        self.complete_active_defect()

            # 步骤5：黑色污染标记要求额外原地旋转360度。
            elif self.step == 5:
                if self.rotate_360(direction=1):
                    self.complete_active_defect()

            # 步骤6：两类标记均达到要求后移动到管道终点并结束。
            elif self.step == 6:
                if self.move_to_pose(self.end_pose):
                    self.finish_task()

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task1V2().run()
    except rospy.ROSInterruptException:
        pass
