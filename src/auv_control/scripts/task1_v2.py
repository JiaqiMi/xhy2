#! /home/xhy/xhy_env/bin/python
"""
名称：task1_v2.py
功能：2026 Task 1——主管道检修
描述：
    1. 移动到主管道入口并根据视觉检测结果沿管道前进；
    2. 识别黄色圆形泄漏标记和黑色方形污染标记；
    3. 控制 hand 坐标系触碰标记；
    4. 黄色标记闪烁一次红灯；
    5. 黑色标记闪烁两次绿灯并原地旋转 360 度；
    6. 完成规定数量的两类标记后移动到管道终点。
监听：/obj/target_message，/obj/line_message，/tf
发布：/target，/sensor，/finished
说明：比赛场地坐标、目标类别名称和标记数量通过 ROS 参数配置。
"""

import copy
import math

import rospy
import tf
from auv_control.msg import TargetDetection, TargetDetection3
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

from task_v2_common import MissionBase


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
        self.required_yellow = int(rospy.get_param('/task1_v2_required_yellow', 1))
        self.required_black = int(rospy.get_param('/task1_v2_required_black', 1))
        self.touch_hold_seconds = rospy.get_param('/task1_v2_touch_hold_seconds', 1.0)
        self.marker_merge_distance = rospy.get_param('/task1_v2_marker_merge_distance', 0.5)
        self.contact_standoff = rospy.get_param('/task1_v2_contact_standoff', 0.0)
        self.line_forward_fraction = rospy.get_param('/task1_v2_line_forward_fraction', 0.8)
        self.yellow_classes = class_names(
            '/task1_v2_yellow_classes', ['yellow_circle', 'yellow']
        )
        self.black_classes = class_names(
            '/task1_v2_black_classes', ['black_square', 'black_rectangle', 'black']
        )

        self.line_target = None
        self.pending_defects = []
        self.handled_markers = []
        self.handled_counts = {'yellow': 0, 'black': 0}
        self.active_defect = None

        rospy.Subscriber('/obj/target_message', TargetDetection, self.defect_callback)
        rospy.Subscriber('/obj/line_message', TargetDetection3, self.line_callback)
        rospy.loginfo('%s: initialized', NODE_NAME)

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

        视觉消息中的目标点先从 camera 转换到 map，再根据 hand 到
        base_link 的静态 TF 计算能够完成触碰动作的 AUV 目标位姿。
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

        contact, marker = self.contact_pose(message.pose, self.contact_standoff)
        if contact is None or self.marker_already_known(marker.pose.position):
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

            # 步骤2：移动到接触位姿，使现有 hand 机构触碰标记。
            elif self.step == 2:
                if self.move_to_pose(self.active_defect['contact'], position_tolerance=0.1):
                    rospy.loginfo('%s: marker touched', NODE_NAME)
                    self.set_step(3)

            # 步骤3：保持接触指定时间，避免瞬时到达造成误判。
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
