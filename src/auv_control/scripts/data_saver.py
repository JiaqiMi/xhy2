#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
名称：data_saver.py
功能：将 debug/sensor/nav 三类 ROS 消息保存为同一个 JSONL 事件流文件
作者：项目组
监听：/status/auv、/status/power、/nav
发布：无
记录：
2026.7.13
    订阅话题调整为 /status/auv 与 /status/power。
2026.7.31
    整合数据按 auv_data 子目录保存，文件达到 50 MiB 后自动分卷并追加下划线编号。
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
        self.save_subdir = 'auv_data'
        self.file_name = rospy.get_param('~file_name', '')
        self.max_file_size = max(1, int(rospy.get_param('~max_file_size', 50 * 1024 * 1024)))
        self.flush_every = max(1, int(rospy.get_param('~flush_every', 1)))
        self.write_count = 0
        self.file = None
        self.file_save_dir = ''
        self.base_file_name = ''
        self.file_index = 0

        if self.enabled:
            self.open_file()

        rospy.Subscriber('/status/auv', AUVData, self.debug_callback)
        rospy.Subscriber('/status/power', SensorStatus, self.sensor_callback)
        rospy.Subscriber('/nav', NavData, self.nav_callback)

        rospy.loginfo("data_saver: 已启动")

    def open_file(self):
        if not self.file_name:
            self.file_name = datetime.now().strftime('auv_data_%Y%m%d_%H%M%S.jsonl')
        self.base_file_name = self.file_name
        self.file_index = 0

        self.file_save_dir = os.path.join(self.save_dir, self.save_subdir)
        os.makedirs(self.file_save_dir, exist_ok=True)
        self._open_file(self.base_file_name)

    def _open_file(self, file_name):
        """打开指定的整合数据分卷文件"""
        path = os.path.join(self.file_save_dir, file_name)
        self.file = open(path, 'a', encoding='utf-8')
        rospy.loginfo(f"data_saver: 数据将保存到 {path}")

    def _rotate_file_if_needed(self, record_size):
        """在写入前检查文件大小，超过上限时切换到下一个分卷"""
        if self.file is None:
            return

        current_size = self.file.tell()
        if current_size == 0 or current_size + record_size <= self.max_file_size:
            return

        self.file.flush()
        self.file.close()
        self.file_index += 1
        file_stem, extension = os.path.splitext(self.base_file_name)
        file_name = f'{file_stem}_{self.file_index}{extension}'
        self._open_file(file_name)

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
            event_line = json.dumps(event, ensure_ascii=False) + '\n'
            self._rotate_file_if_needed(len(event_line.encode('utf-8')))
            self.file.write(event_line)
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
