#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
名称：test_task1_v2_black_marker.py
功能：Task1 黑色方形单项测试。

流程：
    复用黄色图形测试的启动等待、制动和定点过渡；
    同一 rectangle 连续 3 帧且每帧置信度不低于 0.30 后确认；
    到达黑色方形后亮绿灯，并以 MZ=3000 和 TF 航向累计完成默认两圈旋转，
    最后回到图形中心及初始航向，稳定后结束。

监听：/obj/target_message，/left/image_raw，/status/vel（可选），/tf
发布：/cmd/pose/ned，/cmd/actuator，/finished

记录：
2026.7.14
    初版，用于单独验证 Task1 黑色方形识别、前往图形、亮灯和旋转流程。
2026.7.16
    同步 Task1MarkerActionTest 的启动等待、稳定定点和 MZ 旋转控制。
    旋转方向、MZ 步长、减速区和反馈过滤均可通过 launch 调整。
    降低黑色方形确认要求：连续 3 帧、置信度至少 30%。
"""

import rospy

from test_task1_v2_yellow_marker import Task1MarkerActionTest


def main():
    rospy.init_node("test_task1_v2_black_marker")
    Task1MarkerActionTest("test_task1_v2_black_marker", "black").run()


if __name__ == "__main__":
    main()
