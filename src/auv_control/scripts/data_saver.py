#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
名称：data_saver.py
功能：将 debug/sensor/nav 三类 ROS 消息保存为同一个 JSONL 事件流文件
记录：
2026.7.13
    订阅话题调整为 /status/auv 与 /status/power。
"""

import json
import os
from datetime import datetime

import rospy
from genpy import Message

from auv_control.msg import AUVData, NavData, SensorStatus


class DataSaver:
    """
    事件流保存节点：
    任一订阅话题收到消息，就立即写入一行 JSON。
    """

    def __init__(self):
        self.enabled = rospy.get_param('~enabled', True)
        self.save_dir = os.path.expanduser(rospy.get_param('~save_dir', '~/.ros/auv_logs'))
        self.file_name = rospy.get_param('~file_name', '')
        self.flush_every = max(1, int(rospy.get_param('~flush_every', 1)))
        self.write_count = 0
        self.file = None

        if self.enabled:
            self.open_file()

        rospy.Subscriber('/status/auv', AUVData, self.debug_callback)
        rospy.Subscriber('/status/power', SensorStatus, self.sensor_callback)
        rospy.Subscriber('/nav', NavData, self.nav_callback)

        rospy.loginfo("data_saver: 已启动")

    def open_file(self):
        if not self.file_name:
            self.file_name = datetime.now().strftime('auv_data_%Y%m%d_%H%M%S.jsonl')

        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, self.file_name)
        self.file = open(path, 'a', encoding='utf-8')
        rospy.loginfo(f"data_saver: 数据将保存到 {path}")

    def message_to_dict(self, msg):
        if hasattr(msg, 'secs') and hasattr(msg, 'nsecs') and callable(getattr(msg, 'to_sec', None)):
            return {
                'secs': msg.secs,
                'nsecs': msg.nsecs,
                'time': msg.to_sec(),
            }

        if isinstance(msg, Message):
            result = {}
            for field in msg.__slots__:
                result[field] = self.message_to_dict(getattr(msg, field))
            return result

        if isinstance(msg, (list, tuple)):
            return [self.message_to_dict(item) for item in msg]

        return msg

    def write_event(self, source, topic, msg):
        if not self.enabled or self.file is None:
            return

        stamp = None
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            stamp = self.message_to_dict(msg.header.stamp)

        event = {
            'pc_time': rospy.Time.now().to_sec(),
            'source': source,
            'topic': topic,
            'msg_type': msg._type,
            'stamp': stamp,
            'data': self.message_to_dict(msg),
        }

        try:
            self.file.write(json.dumps(event, ensure_ascii=False) + '\n')
            self.write_count += 1
            if self.write_count % self.flush_every == 0:
                self.file.flush()
        except Exception as e:
            rospy.logerr(f"data_saver: 写入失败: {e}")

    def debug_callback(self, msg):
        self.write_event('debug', '/status/auv', msg)

    def sensor_callback(self, msg):
        self.write_event('sensor', '/status/power', msg)

    def nav_callback(self, msg):
        self.write_event('nav', '/nav', msg)

    def spin(self):
        try:
            rospy.spin()
        finally:
            if self.file:
                self.file.flush()
                self.file.close()
                rospy.loginfo("data_saver: 数据文件已保存并关闭")


if __name__ == "__main__":
    rospy.init_node('data_saver')
    try:
        saver = DataSaver()
        saver.spin()
    except rospy.ROSInterruptException:
        pass
