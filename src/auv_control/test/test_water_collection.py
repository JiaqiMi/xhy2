#! /home/xhy/xhy_env/bin/python
"""
名称：test_water_collection.py
功能：测试取水器完整流程
描述：
    1.
监听：/tf
发布：/target (PoseStamped)，/sensor (Control)，/finished (String)
说明：本文件只封装多个任务共同使用的功能，不单独启动 ROS 节点。
"""
