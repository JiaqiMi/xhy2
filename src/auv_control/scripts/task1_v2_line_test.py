#! /home/xhy/xhy_env/bin/python
"""
名称：task1_v2_line_test.py
功能：Task1 V2 巡线运动功能单独测试节点
描述：
    1. 启动后读取机器人当前 map -> base_link 位姿；
    2. 以当前航向为前方，先向前运动 0.5 m；
    3. 到达前方目标后原地转向 180 度；
    4. 再沿转向后的前方运动 0.5 m，回到启动位置；
    5. 全过程目标 z 使用当前深度，不主动上浮或下潜。
监听：/tf
发布：/target，/auv_actuator_control，/finished
说明：本节点不订阅视觉话题，只用于验证最小巡线运动控制链路。
"""

import copy
import math

import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

from task_v2_common import MissionBase, wrap_angle, yaw_from_quaternion


NODE_NAME = 'task1_v2_line_test'


class Task1V2LineTest(MissionBase):
    """Task1 巡线运动功能单独测试状态机。"""

    def __init__(self):
        """初始化测试距离和动作目标。"""
        super().__init__(NODE_NAME)

        self.test_distance = float(rospy.get_param('~test_distance', 0.5))
        self.start_pose = None
        self.forward_pose = None
        self.turn_pose = None
        self.return_pose = None

        rospy.loginfo('%s: initialized', NODE_NAME)

    ############################################### 位姿生成层 #######################################
    def make_pose(self, x, y, z, yaw):
        """根据 map 坐标和 yaw 生成目标位姿。"""
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = rospy.Time.now()
        pose.pose.position = Point(x, y, z)
        pose.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0, self.pitch_offset, yaw
        ))
        return pose

    def initialize_targets(self):
        """以启动时当前位置为基准生成前进、转向和返回目标。"""
        current = self.get_current_pose()
        if current is None:
            return False

        start_yaw = yaw_from_quaternion(current.pose.orientation)
        return_yaw = wrap_angle(start_yaw + math.pi)
        start_x = current.pose.position.x
        start_y = current.pose.position.y
        start_z = current.pose.position.z
        forward_x = start_x + self.test_distance * math.cos(start_yaw)
        forward_y = start_y + self.test_distance * math.sin(start_yaw)

        self.start_pose = copy.deepcopy(current)
        self.forward_pose = self.make_pose(forward_x, forward_y, start_z, start_yaw)
        self.turn_pose = self.make_pose(forward_x, forward_y, start_z, return_yaw)
        self.return_pose = self.make_pose(start_x, start_y, start_z, return_yaw)

        rospy.loginfo(
            '%s: test path ready, start=(%.2f, %.2f), forward=(%.2f, %.2f)',
            NODE_NAME,
            start_x,
            start_y,
            forward_x,
            forward_y,
        )
        return True

    ############################################### 运动辅助层 #######################################
    def move_to_pose_xy(self, target, position_tolerance=None, yaw_tolerance=None):
        """只按 XY 和航向移动，发布目标前将 z 改为机器人当前深度。"""
        current = self.get_current_pose()
        if current is None:
            return False

        xy_target = copy.deepcopy(target)
        xy_target.header.stamp = rospy.Time.now()
        xy_target.pose.position.z = current.pose.position.z
        return self.move_to_pose(xy_target, position_tolerance, yaw_tolerance)

    ############################################### 主循环 ###########################################
    def run(self):
        """执行前进、转向、返回三段动作后结束。"""
        while not rospy.is_shutdown():
            # 步骤0：读取启动位姿并生成测试目标。
            if self.step == 0:
                if self.initialize_targets():
                    self.set_step(1)

            # 步骤1：沿当前航向向前运动 0.5 m。
            elif self.step == 1:
                if self.move_to_pose_xy(self.forward_pose):
                    rospy.loginfo('%s: reached forward point', NODE_NAME)
                    self.set_step(2)

            # 步骤2：在前方目标点原地转向 180 度。
            elif self.step == 2:
                if self.move_to_pose_xy(self.turn_pose, position_tolerance=0.2):
                    rospy.loginfo('%s: turned around', NODE_NAME)
                    self.set_step(3)

            # 步骤3：沿转向后的前方运动 0.5 m，回到启动位置。
            elif self.step == 3:
                if self.move_to_pose_xy(self.return_pose):
                    rospy.loginfo('%s: returned to start point', NODE_NAME)
                    self.set_step(4)

            # 步骤4：测试完成。
            elif self.step == 4:
                self.finish_task()

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task1V2LineTest().run()
    except rospy.ROSInterruptException:
        pass
