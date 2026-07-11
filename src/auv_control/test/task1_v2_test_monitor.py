#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：task1_v2_test_monitor.py
功能：Task1 V2 通信测试话题监控节点
描述：
    1. 统计虚拟巡线、虚拟图形、任务目标点、执行器控制和任务完成消息；
    2. 通过红灯/绿灯上升沿估算黄色/黑色图形动作是否触发；
    3. 收到 /finished 后输出一次测试结果摘要。
监听：/obj/line_message，/obj/target_message，/target，/auv_actuator_control，/finished
发布：None
说明：本节点只做监控和日志输出，不参与控制闭环。
"""

import rospy
from auv_control.msg import ActuatorControl, TargetDetection, TargetDetection3
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


NODE_NAME = 'task1_v2_test_monitor'


class Task1V2TestMonitor:
    """Task1 V2 真实通信测试的话题统计器。"""

    def __init__(self):
        """初始化计数器和订阅器。"""
        self.line_count = 0
        self.marker_count = 0
        self.target_count = 0
        self.actuator_count = 0
        self.red_rising_edges = 0
        self.green_rising_edges = 0
        self.last_red = 0
        self.last_green = 0
        self.finished_message = ''
        self.marker_class_counts = {}

        self.last_line_time = None
        self.last_marker_time = None
        self.last_target_time = None
        self.last_actuator_time = None

        self.expected_yellow = int(rospy.get_param('/task1_v2_required_yellow', 2))
        self.expected_black = int(rospy.get_param('/task1_v2_required_black', 2))

        rospy.Subscriber('/obj/line_message', TargetDetection3, self.line_callback)
        rospy.Subscriber('/obj/target_message', TargetDetection, self.marker_callback)
        rospy.Subscriber('/target', PoseStamped, self.target_callback)
        rospy.Subscriber('/auv_actuator_control', ActuatorControl, self.actuator_callback)
        rospy.Subscriber('/finished', String, self.finished_callback)
        self.report_timer_handle = rospy.Timer(rospy.Duration(5.0), self.report_timer)

        rospy.loginfo('%s: monitor initialized', NODE_NAME)

    ############################################### 回调层 ###########################################
    def line_callback(self, _):
        """统计巡线识别消息。"""
        self.line_count += 1
        self.last_line_time = rospy.Time.now()

    def marker_callback(self, message):
        """统计图形识别消息及类别。"""
        self.marker_count += 1
        self.last_marker_time = rospy.Time.now()
        self.marker_class_counts[message.class_name] = (
            self.marker_class_counts.get(message.class_name, 0) + 1
        )

    def target_callback(self, _):
        """统计任务目标点输出。"""
        self.target_count += 1
        self.last_target_time = rospy.Time.now()

    def actuator_callback(self, message):
        """统计执行器控制消息，并记录红绿灯上升沿。"""
        self.actuator_count += 1
        self.last_actuator_time = rospy.Time.now()

        red = 1 if message.red_light else 0
        green = 1 if message.green_light else 0
        if red == 1 and self.last_red == 0:
            self.red_rising_edges += 1
            rospy.loginfo('%s: red light rising edge #%d',
                          NODE_NAME, self.red_rising_edges)
        if green == 1 and self.last_green == 0:
            self.green_rising_edges += 1
            rospy.loginfo('%s: green light rising edge #%d',
                          NODE_NAME, self.green_rising_edges)

        self.last_red = red
        self.last_green = green

    def finished_callback(self, message):
        """收到任务完成消息后输出最终摘要。"""
        self.finished_message = message.data
        rospy.loginfo('%s: received finished message: %s', NODE_NAME, message.data)
        self.print_summary(final=True)
        rospy.signal_shutdown('task finished')

    ############################################### 报告层 ###########################################
    @staticmethod
    def age_seconds(last_time):
        """计算某话题距离上次收到消息的时间。"""
        if last_time is None:
            return None
        return (rospy.Time.now() - last_time).to_sec()

    @staticmethod
    def format_age(age):
        """格式化消息年龄。"""
        if age is None:
            return 'never'
        return '{:.1f}s'.format(age)

    def report_timer(self, _):
        """周期性输出当前测试统计。"""
        self.print_summary(final=False)

    def print_summary(self, final=False):
        """打印测试统计摘要。"""
        expected_green_edges = self.expected_black * 2
        result = (
            self.line_count > 0
            and self.marker_count > 0
            and self.target_count > 0
            and self.actuator_count > 0
            and self.red_rising_edges >= self.expected_yellow
            and self.green_rising_edges >= expected_green_edges
        )

        prefix = 'FINAL' if final else 'STATUS'
        rospy.loginfo(
            '%s %s: line=%d(age=%s), marker=%d(age=%s), target=%d(age=%s), '
            'actuator=%d(age=%s), red_edges=%d/%d, green_edges=%d/%d, '
            'finished="%s", classes=%s, result=%s',
            NODE_NAME,
            prefix,
            self.line_count,
            self.format_age(self.age_seconds(self.last_line_time)),
            self.marker_count,
            self.format_age(self.age_seconds(self.last_marker_time)),
            self.target_count,
            self.format_age(self.age_seconds(self.last_target_time)),
            self.actuator_count,
            self.format_age(self.age_seconds(self.last_actuator_time)),
            self.red_rising_edges,
            self.expected_yellow,
            self.green_rising_edges,
            expected_green_edges,
            self.finished_message,
            self.marker_class_counts,
            'PASS' if result else 'RUNNING/FAIL',
        )


if __name__ == '__main__':
    rospy.init_node(NODE_NAME, anonymous=False)
    try:
        Task1V2TestMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
