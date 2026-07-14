#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v2_black_marker.py
功能：Task1 黑色方形单项测试。

流程：
    1. 以节点启动时机器人当前位置为起点，并记录当前 z，不主动改变高度；
    2. 按设定初始航向手控前进，直到识别到黑色方形；
    3. 将图形位置转换到 map 坐标系；
    4. 使用动力定位 ROV 模式只在 XY 平面前往图形上方；
    5. 到达后亮绿灯并根据航向反馈执行旋转动作。

监听：/obj/target_message，/tf
发布：/cmd/pose/ned，/cmd/actuator，/finished

记录：
2026.7.14
    初版，用于单独验证 Task1 黑色方形识别、前往图形、亮灯和旋转流程。
"""

import rospy

from test_task1_v2_yellow_marker import Task1MarkerActionTest


def main():
    rospy.init_node("test_task1_v2_black_marker")
    Task1MarkerActionTest("test_task1_v2_black_marker", "black").run()


if __name__ == "__main__":
    main()
