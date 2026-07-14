#! /home/xhy/xhy_env/bin/python
"""
名称：task2_v2.py
功能：2026 Task 2——取水器采水与返航
作者：buyegaid
订阅：/tf
发布：/target，/cmd/actuator，/finished
记录：
    2026-07-13：
        1. 新增取水器定点采水、深度保持返航和原点保持 10 秒流程；
        2. 新增 /task_v2_sample_duration、/task_v2_pushrod_speed、
           /task_v2_return_yaw_deg 参数；
        3. 推杆前进速度由固定值 250 改为参数化配置，默认值为 250。
        4. 统一日志格式，日志正文以节点名称 task2_v2 开头。
        5. 执行器下行话题调整为 /cmd/actuator。
"""

import math

import rospy
from auv_control.msg import ActuatorControl
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_from_euler

from task_v2_common import MissionBase


NODE_NAME = 'task2_v2'
PUSHROD_FORWARD = 1
ARRIVAL_HOLD_SECONDS = 10.0


class Task2V2(MissionBase):
    """取水、返航和到达保持的任务状态机。"""

    def __init__(self):
        """读取采水时长、推杆速度和返航航向参数。"""
        super().__init__(NODE_NAME)
        self.sample_duration = float(
            rospy.get_param('/task_v2_sample_duration', 10.0)
        )
        self.pushrod_speed = int(
            rospy.get_param('/task_v2_pushrod_speed', 250)
        )
        self.return_yaw_deg = float(
            rospy.get_param('/task_v2_return_yaw_deg', 0.0)
        )
        if self.sample_duration < 0.0:
            raise ValueError('/task_v2_sample_duration must be non-negative')
        if not 0 <= self.pushrod_speed <= 254:
            raise ValueError('/task_v2_pushrod_speed must be in [0, 254]')

        # 任务结束时必须停止推杆，不能继承其他任务的默认推杆动作。
        self.default_drive_cmd = 0
        self.default_drive_speed = 0
        self.collection_started_at = None
        self.return_pose = None
        self.log_info(
            'initialized (sample_duration=%.2fs, pushrod_speed=%d, '
            'return_yaw=%.1fdeg)',
            self.sample_duration,
            self.pushrod_speed,
            self.return_yaw_deg,
        )

    ############################################### 日志层 ###########################################
    def log_info(self, message, *args):
        """输出以节点名称开头的 INFO 日志。"""
        rospy.loginfo('%s: ' + message, NODE_NAME, *args)

    ############################################### 采水机构层 #######################################
    def publish_pushrod(self, command, speed):
        """发布取水推杆控制，并保持其他执行器处于默认状态。"""
        message = ActuatorControl()
        message.light1 = 0
        message.light2 = 0
        message.heading_servo = self.default_heading_servo
        message.clamp_servo = self.default_clamp_servo
        message.drive_cmd = int(command)
        message.drive_speed = int(speed)
        message.red_light = 0
        message.yellow_light = 0
        message.green_light = 0
        self.device_pub.publish(message)

    def collect_water(self):
        """在采水位置持续定点控制，并驱动推杆前进至采水结束。"""
        self.hold_position()
        if self.hold_pose is None:
            return False

        if self.collection_started_at is None:
            self.collection_started_at = rospy.Time.now()
            self.log_info('water collection started')

        elapsed = (rospy.Time.now() - self.collection_started_at).to_sec()
        if elapsed >= self.sample_duration:
            self.publish_pushrod(0, 0)
            return True

        self.publish_pushrod(PUSHROD_FORWARD, self.pushrod_speed)
        return False

    ############################################### 返航目标层 #######################################
    def build_return_pose(self):
        """构造返回世界坐标原点且保持采水深度的目标位姿。"""
        target = PoseStamped()
        target.header.frame_id = 'map'
        target.header.stamp = rospy.Time.now()
        # 位置是当前位置，深度保持原有深度
        target.pose.position.x = 0.0
        target.pose.position.y = 0.0
        target.pose.position.z = self.hold_pose.pose.position.z
        target.pose.orientation = Quaternion(*quaternion_from_euler(
            0.0,
            0.0,
            math.radians(self.return_yaw_deg),
        ))
        return target

    ############################################### 主循环 ###########################################
    def run(self):
        """依次执行采水、返航、到达保持和任务结束。"""
        while not rospy.is_shutdown():
            # 步骤0：采水期间持续发布当前位置，确保 AUV 原地定点。
            if self.step == 0:
                if self.collect_water():
                    self.return_pose = self.build_return_pose()
                    self.log_info('water collection completed')
                    self.set_step(1)

            # 步骤1：保持采水深度，返回 map 原点并调整为指定航向。
            elif self.step == 1:
                if self.move_to_pose(self.return_pose):
                    self.log_info('reached world origin')
                    self.set_step(2)

            # 步骤2：在原点定点保持 10 秒后结束任务。
            elif self.step == 2:
                self.hold_position()
                if self.step_elapsed() >= ARRIVAL_HOLD_SECONDS:
                    self.publish_pushrod(0, 0)
                    self.finish_task()

            self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=True)
    try:
        Task2V2().run()
    except rospy.ROSInterruptException:
        pass
