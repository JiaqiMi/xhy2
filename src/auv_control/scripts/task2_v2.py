#! /home/xhy/xhy_env/bin/python
"""
名称：task2_v2.py
功能：2026 Task 2——环境监测与水样采集
描述：
    1. 移动到预设采样位置；
    2. 保持定点并执行采水推拉杆示例动作；
    3. 携带水样返回起始区域；
    4. 在起始区域上浮，实现水样自动送达。
监听：/tf
发布：/target，/sensor，/finished
说明：目前工程没有采水推拉杆话题，operate_sampling_pushrod() 仅为占位示例，
      不会向不存在的硬件接口发布命令。接入硬件后应使用驱动确认信号替换
      sample_collected 的直接赋值。
"""

import copy

import rospy

from task_v2_common import MissionBase


NODE_NAME = 'task2_v2'


class Task2V2(MissionBase):
    """环境监测与水样采集任务状态机。"""

    def __init__(self):
        """初始化采样点、起始点、上浮深度和采样动作时间。"""
        super().__init__(NODE_NAME)
        self.sample_pose = self.pose_from_param(
            '/task2_v2_sample_point', [0.0, 0.0, 0.3, 0.0]
        )
        self.start_pose = self.pose_from_param(
            '/task_v2_start_point', [0.0, 0.0, 0.3, 0.0]
        )
        self.surface_depth = rospy.get_param('/task_v2_surface_depth', 0.0)
        self.sample_duration = rospy.get_param('/task2_v2_sample_duration', 3.0)
        self.sample_collected = False
        self.pushrod_action_announced = False
        rospy.loginfo('%s: initialized', NODE_NAME)

    ############################################### 采水机构层 #######################################
    def operate_sampling_pushrod(self):
        """采水推拉杆示例函数。

        当前行为：
            1. 保持 AUV 位于采样点；
            2. 第一次调用时输出明确的占位警告；
            3. 等待 /task2_v2_sample_duration 指定的时间；
            4. 假定推拉杆已运动且采水器已经完成采水。

        Returns:
            bool：示例采样时间达到后返回 True。

        TODO：后续用真实推拉杆话题或服务替换占位逻辑，并等待硬件反馈。
        """
        self.hold_position()
        if not self.pushrod_action_announced:
            rospy.logwarn(
                '%s: SAMPLE PUSHROD EXAMPLE - assume the pushrod has extended '
                'and the sampler is collecting water; no hardware topic is published',
                NODE_NAME,
            )
            self.pushrod_action_announced = True

        if self.step_elapsed() < self.sample_duration:
            return False

        # TODO: replace this assignment with acknowledgement from the pushrod driver.
        self.sample_collected = True
        rospy.loginfo('%s: example sampling action completed', NODE_NAME)
        return True

    ############################################### 主循环 ###########################################
    def run(self):
        """依次执行“前往采样点→采水→返回起点→上浮送达”。"""
        while not rospy.is_shutdown():
            # 步骤0：移动到允许采集水样的预设位置和深度。
            if self.step == 0:
                if self.move_to_pose(self.sample_pose):
                    rospy.loginfo('%s: reached sampling point', NODE_NAME)
                    self.set_step(1)

            # 步骤1：保持定点并执行推拉杆采水示例动作。
            elif self.step == 1:
                if self.operate_sampling_pushrod():
                    self.set_step(2)

            # 步骤2：携带采集到的水样返回起始区域。
            elif self.step == 2:
                if self.move_to_pose(self.start_pose):
                    rospy.loginfo('%s: sample returned to start area', NODE_NAME)
                    self.surface_pose = copy.deepcopy(self.start_pose)
                    self.surface_pose.pose.position.z = self.surface_depth
                    self.set_step(3)

            # 步骤3：保持起始区域水平位置并上浮到水面。
            elif self.step == 3:
                if self.move_to_pose(self.surface_pose, position_tolerance=0.1):
                    rospy.loginfo(
                        '%s: surfaced in start area for automatic delivery', NODE_NAME
                    )
                    self.finish_task()

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task2V2().run()
    except rospy.ROSInterruptException:
        pass
