#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task1 v3 测试节点共用的带时间戳 YAML 数据记录器。"""

from datetime import datetime
import json
import math
import os
import threading

import rospy


def yaml_safe_value(value):
    """把测试快照递归转换为 YAML/JSON 都可安全表示的基础类型。"""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): yaml_safe_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [yaml_safe_value(item) for item in value]
    return str(value)


class TimestampedYamlLogger:
    """每次启动创建一个 YAML 文档流文件，每条记录都带独立时间。"""

    def __init__(self, node_name, log_directory):
        self._lock = threading.Lock()
        self._file = None
        directory = os.path.abspath(os.path.expanduser(str(log_directory)))
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = os.path.join(
            directory, "%s_%s.yaml" % (node_name, timestamp)
        )
        self._file = open(
            self.path, "a", encoding="utf-8", buffering=1
        )

    def write(self, event, **data):
        """追加一个独立 YAML 文档；JSON 流式写法属于有效的 YAML 1.2。"""
        record = {
            "wall_time": datetime.now().isoformat(timespec="milliseconds"),
            "ros_time": round(rospy.Time.now().to_sec(), 6),
            "event": str(event),
        }
        record.update(yaml_safe_value(data))
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock:
            if self._file is not None:
                self._file.write("--- " + encoded + "\n")

    def close(self):
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.flush()
                self._file.close()
            except OSError:
                pass
            finally:
                self._file = None
