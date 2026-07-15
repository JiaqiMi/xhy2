#! /home/xhy/xhy_env/bin/python
"""
名称：task4_v2.py
功能：2026 Task 4——返回起始区域并上浮
描述：
    1. 移动到 90×90 cm 起始区域内的预设目标点；
    2. 保持起始区域水平位置，上浮到设定水面深度；
    3. 发布任务完成消息，提示队员向裁判说“Stop”。
监听：/tf
发布：/cmd/pose/ned，/cmd/actuator，/finished
说明：程序只能完成返航和上浮，规则要求的口头“Stop”仍需参赛队员执行。
"""

import copy

import rospy

from task_v2_common import MissionBase


NODE_NAME = 'task4_v2'


class Task4V2(MissionBase):
    """返回起点并上浮的任务状态机。"""

    def __init__(self):
        """初始化起始区域目标位姿和水面深度。"""
        super().__init__(NODE_NAME)
        self.start_pose = self.pose_from_param(
            '/task_v2_start_point', [0.0, 0.0, 0.3, 0.0]
        )
        self.surface_depth = rospy.get_param('/task_v2_surface_depth', 0.0)
        self.surface_pose = copy.deepcopy(self.start_pose)
        self.surface_pose.pose.position.z = self.surface_depth
        rospy.loginfo('%s: initialized', NODE_NAME)

    ############################################### 主循环 ###########################################
    def run(self):
        """依次执行“返回起始区域→原地上浮→结束任务”。"""
        while not rospy.is_shutdown():
            # 步骤0：先在水下返回起始区域中心附近。
            if self.step == 0:
                if self.move_to_pose(self.start_pose):
                    rospy.loginfo('%s: returned to start area', NODE_NAME)
                    self.set_step(1)

            # 步骤1：保持起点的 north/east 和航向，仅改变 down 坐标上浮。
            elif self.step == 1:
                if self.move_to_pose(self.surface_pose, position_tolerance=0.1):
                    rospy.loginfo(
                        '%s: surfaced; team member should now call Stop', NODE_NAME
                    )
                    self.finish_task()

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task4V2().run()
    except rospy.ROSInterruptException:
        pass
