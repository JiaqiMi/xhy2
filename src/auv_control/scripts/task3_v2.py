#! /home/xhy/xhy_env/bin/python
"""
名称：task3_v2.py
功能：2026 Task 3——识别指定管段并投放信标
描述：
    1. 移动到任务获取区域；
    2. 使用 mock_aruco_id() 示例函数随机生成 1～6 的任务编号；
    3. 根据编号确定管段颜色：1/2 黄色、3/4 绿色、5/6 红色；
    4. 显示对应颜色的灯并移动到目标管段；
    5. 使用 /cmd/actuator 舵机接口释放高尔夫球信标。
监听：/tf
发布：/cmd/pose/ned，/cmd/actuator，/finished
说明：真实 ArUco 识别不在本节点实现，当前用随机编号跳过识别流程。
      黄色灯由红灯和绿灯同时点亮表示。
"""

import random

import rospy

from task_v2_common import MissionBase


NODE_NAME = 'task3_v2'


class Task3V2(MissionBase):
    """任务编号获取、彩色管段导航和信标投放状态机。"""

    def __init__(self):
        """初始化任务获取点、三种管段位置和信标舵机参数。"""
        super().__init__(NODE_NAME)
        self.acquisition_pose = self.pose_from_param(
            '/task3_v2_acquisition_point', [0.0, 0.0, 0.3, 0.0]
        )
        self.region_poses = {
            'yellow': self.pose_from_param(
                '/task3_v2_yellow_point', [0.0, 0.0, 0.3, 0.0]
            ),
            'green': self.pose_from_param(
                '/task3_v2_green_point', [0.0, 0.0, 0.3, 0.0]
            ),
            'red': self.pose_from_param(
                '/task3_v2_red_point', [0.0, 0.0, 0.3, 0.0]
            ),
        }
        self.assignment_light_seconds = rospy.get_param(
            '/task3_v2_assignment_light_seconds', 2.0
        )
        self.beacon_open_seconds = rospy.get_param(
            '/task3_v2_beacon_open_seconds', 2.0
        )
        self.beacon_close_seconds = rospy.get_param(
            '/task3_v2_beacon_close_seconds', 1.0
        )
        self.beacon_open_servo = int(
            rospy.get_param('/task3_v2_beacon_open_servo', 100)
        )
        self.beacon_closed_servo = int(
            rospy.get_param('/task3_v2_beacon_closed_servo', 255)
        )

        self.marker_id = None
        self.assigned_color = None
        self.target_region = None

        rospy.loginfo('%s: initialized', NODE_NAME)

    ############################################### 任务映射层 #######################################
    @staticmethod
    def mock_aruco_id():
        """示例任务获取函数：随机返回一个 1～6 的编号。

        真实比赛中，这里应替换为外部 ArUco 识别模块返回的编号。本代码只负责
        后续任务流程，因此用随机数临时跳过识别部分。
        """
        return random.randint(1, 6)

    @staticmethod
    def color_for_marker(marker_id):
        """将 ArUco 编号 1～6 映射为黄色、绿色或红色管段。"""
        if marker_id in (1, 2):
            return 'yellow'
        if marker_id in (3, 4):
            return 'green'
        if marker_id in (5, 6):
            return 'red'
        return None

    ############################################### 逻辑层 ###########################################
    def show_assignment_color(self):
        """按照目标管段颜色亮灯，并在设定时间后自动熄灯。"""
        if self.assigned_color == 'yellow':
            return self.show_lights(red=1, green=1, duration=self.assignment_light_seconds)
        if self.assigned_color == 'green':
            return self.show_lights(red=0, green=1, duration=self.assignment_light_seconds)
        return self.show_lights(red=1, green=0, duration=self.assignment_light_seconds)

    def release_beacon(self):
        """通过现有舵机接口释放高尔夫球信标。

        动作顺序：先发送开舵机值并保持 beacon_open_seconds，再发送关闭值
        保持 beacon_close_seconds。整个动作期间持续发布定点目标。

        Returns:
            bool：舵机打开和复位阶段均完成后返回 True。
        """
        self.hold_position()
        elapsed = self.step_elapsed()
        if elapsed < self.beacon_open_seconds:
            self.publish_device(servo=self.beacon_open_servo)
            return False

        if elapsed < self.beacon_open_seconds + self.beacon_close_seconds:
            self.publish_device(servo=self.beacon_closed_servo)
            return False

        self.publish_device(servo=self.beacon_closed_servo)
        rospy.loginfo('%s: beacon released', NODE_NAME)
        return True

    ############################################### 主循环 ###########################################
    def run(self):
        """依次执行“获取任务→显示颜色→到达管段→投放信标”。"""
        while not rospy.is_shutdown():
            # 步骤0：到达 90×90 cm 的任务获取区域。
            if self.step == 0:
                if self.move_to_pose(self.acquisition_pose):
                    rospy.loginfo('%s: reached task acquisition area', NODE_NAME)
                    self.set_step(1)

            # 步骤1：用示例函数随机生成任务编号，并确定目标管段。
            elif self.step == 1:
                self.marker_id = self.mock_aruco_id()
                self.assigned_color = self.color_for_marker(self.marker_id)
                self.target_region = self.region_poses[self.assigned_color]
                rospy.logwarn(
                    '%s: mock ArUco ID %d selects %s region',
                    NODE_NAME,
                    self.marker_id,
                    self.assigned_color,
                )
                self.set_step(2)

            # 步骤2：显示与指定管段对应的颜色。
            elif self.step == 2:
                if self.show_assignment_color():
                    self.set_step(3)

            # 步骤3：移动到黄色、绿色或红色目标管段框架内。
            elif self.step == 3:
                if self.move_to_pose(self.target_region, position_tolerance=0.1):
                    rospy.loginfo('%s: reached assigned pipeline region', NODE_NAME)
                    self.set_step(4)

            # 步骤4：打开投放机构释放信标，复位舵机后结束任务。
            elif self.step == 4:
                if self.release_beacon():
                    self.finish_task()

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task3V2().run()
    except rospy.ROSInterruptException:
        pass
