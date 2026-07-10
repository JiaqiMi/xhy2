#! /home/xhy/xhy_env/bin/python
"""
名称：task1_v2_black_test.py
功能：Task1 V2 黑色方形动作单独测试节点
描述：
    1. 启动后读取机器人当前 map -> base_link 位姿；
    2. 以当前航向为前方，先向前运动 0.5 m；
    3. 到达前方目标后保持当前位置；
    4. 发布 /auv_actuator_control，使绿灯闪烁两次；
    5. 绿灯动作完成后原地旋转 360 度。
监听：/tf
发布：/target，/auv_actuator_control，/finished
说明：本节点不订阅视觉话题，用固定前进距离模拟“检测到黑色方形后执行动作”。
"""

import copy
import math

import rospy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

from task_v2_common import MissionBase, yaw_from_quaternion


NODE_NAME = 'task1_v2_black_test'


class Task1V2BlackTest(MissionBase):
    """Task1 黑色方形动作单独测试状态机。"""

    def __init__(self):
        """初始化测试距离和前进目标。"""
        super().__init__(NODE_NAME)

        self.test_distance = float(rospy.get_param('~test_distance', 0.5))
        self.forward_pose = None

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

    def initialize_target(self):
        """以启动时当前位置为基准生成前进 0.5 m 目标。"""
        current = self.get_current_pose()
        if current is None:
            return False

        start_yaw = yaw_from_quaternion(current.pose.orientation)
        target_x = current.pose.position.x + self.test_distance * math.cos(start_yaw)
        target_y = current.pose.position.y + self.test_distance * math.sin(start_yaw)
        self.forward_pose = self.make_pose(
            target_x,
            target_y,
            current.pose.position.z,
            start_yaw,
        )

        rospy.loginfo(
            '%s: black test target ready, target=(%.2f, %.2f)',
            NODE_NAME,
            target_x,
            target_y,
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
        """执行前进 0.5 m、绿灯动作和旋转动作后结束。"""
        while not rospy.is_shutdown():
            # 步骤0：读取启动位姿并生成前进目标。
            if self.step == 0:
                if self.initialize_target():
                    self.set_step(1)

            # 步骤1：向前运动 0.5 m。
            elif self.step == 1:
                if self.move_to_pose_xy(self.forward_pose):
                    rospy.loginfo('%s: reached black action point', NODE_NAME)
                    self.set_step(2)

            # 步骤2：黑色方形动作，绿灯闪烁两次。
            elif self.step == 2:
                if self.blink_lights(red=0, green=1, count=2):
                    self.set_step(3)

            # 步骤3：原地旋转 360 度。
            elif self.step == 3:
                if self.rotate_360(direction=1):
                    self.set_step(4)

            # 步骤4：测试完成。
            elif self.step == 4:
                self.finish_task()

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task1V2BlackTest().run()
    except rospy.ROSInterruptException:
        pass
