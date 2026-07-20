#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试口协议的纯 Python 公共解析与滤波工具。"""

from __future__ import division

import math
import struct


DEBUG_PACKET_SIZE = 110
DEBUG_HEADER = b'\xFE\xEF'
DEBUG_TAIL = b'\xFA\xAF'


def require_finite(values, name):
    """要求一组协议浮点量全部有限，并返回浮点元组。"""
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError('{}包含 NaN 或 Inf'.format(name))
    return converted


def decode_status_words(packet):
    """按协议位权直接解析状态字段，最低位保持为 bit 0。"""
    if len(packet) < 16:
        raise ValueError('调试报文长度不足 16 字节')
    return (
        packet[10] & 0x7F,
        packet[11] & 0x1F,
        struct.unpack('>H', packet[12:14])[0] & 0x01FF,
        # AUVSensor.power_status 的既有消息类型为 int16，保持接口 ABI。
        struct.unpack('>h', packet[14:16])[0],
    )


class DebugFrameBuffer(object):
    """从任意 TCP 字节分片中提取完整 110 字节调试报文。"""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        """追加一段非空字节流并返回本次提取出的完整报文。"""
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError('调试字节流必须是 bytes 或 bytearray')
        if not data:
            raise ValueError('调试字节流不能为空')
        self.buffer.extend(data)
        packets = []
        while True:
            start = self.buffer.find(DEBUG_HEADER)
            if start < 0:
                # 保留可能作为下一段报文头首字节的末尾 FE，丢弃其余垃圾。
                self.buffer[:] = (
                    self.buffer[-1:]
                    if self.buffer and self.buffer[-1] == DEBUG_HEADER[0]
                    else b''
                )
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < DEBUG_PACKET_SIZE:
                break
            if (
                    self.buffer[DEBUG_PACKET_SIZE - 2:DEBUG_PACKET_SIZE]
                    == DEBUG_TAIL):
                packets.append(bytes(self.buffer[:DEBUG_PACKET_SIZE]))
                del self.buffer[:DEBUG_PACKET_SIZE]
            else:
                # 当前 FE EF 不是合法报文头，从其后继续寻找。
                del self.buffer[:len(DEBUG_HEADER)]
        return packets


class LowPassFilter(object):
    """拒绝非有限输入的一阶低通滤波器。"""

    def __init__(self, alpha=0.1):
        self.alpha = float(alpha)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError('低通滤波 alpha 必须位于 (0, 1]')
        self.last_value = None

    def update(self, value):
        value = require_finite((value,), '低通滤波输入')[0]
        if self.last_value is None:
            self.last_value = value
            return value
        filtered = (
            self.alpha * value
            + (1.0 - self.alpha) * self.last_value
        )
        self.last_value = filtered
        return filtered


class MovingAverageFilter(object):
    """拒绝非有限输入的固定窗口移动平均滤波器。"""

    def __init__(self, window_size=5):
        self.window_size = int(window_size)
        if self.window_size <= 0:
            raise ValueError('移动平均窗口必须大于 0')
        self.values = []

    def update(self, value):
        value = require_finite((value,), '移动平均输入')[0]
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)
