#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：world_frame.py
功能：提供世界坐标系与经纬深坐标之间的转换
作者：buyegaid
订阅：无
发布：无
记录：
2026.7.13
    新增 WorldFrameManager，供世界原点初始化与重置流程复用。
"""

import numpy as np


class WorldFrameManager:
    """维护一个以经纬深为原点的北东地（NED）坐标系。"""

    def __init__(self, init_lat, init_lon, init_depth):
        self.init_lat, self.init_lon, self.init_depth = self._validated_lld(
            init_lat, init_lon, init_depth, '世界原点')
        self.a = 6378137.0
        self.f = 1 / 298.257223563
        self.e_sq = self.f * (2 - self.f)

    def lld_to_ned(self, lat, lon, depth):
        """将纬度、经度、深度转换为相对原点的北东地坐标。"""
        x, y, z = self.lld_to_ecef(lat, lon, depth)
        x0, y0, z0 = self.lld_to_ecef(
            self.init_lat, self.init_lon, self.init_depth
        )
        return self.ecef_to_ned(x, y, z, x0, y0, z0)

    def lld_to_ecef(self, lat, lon, depth):
        """将纬度、经度、深度转换为 ECEF 坐标。"""
        lat, lon, depth = self._validated_lld(
            lat, lon, depth, '经纬深输入')
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        radius = self.a / np.sqrt(1 - self.e_sq * np.sin(lat_rad) ** 2)
        x = (radius - depth) * np.cos(lat_rad) * np.cos(lon_rad)
        y = (radius - depth) * np.cos(lat_rad) * np.sin(lon_rad)
        z = (radius * (1 - self.e_sq) - depth) * np.sin(lat_rad)
        return x, y, z

    def ecef_to_ned(self, x, y, z, x0, y0, z0):
        """将 ECEF 坐标转换为以当前原点为参考的 NED 坐标。"""
        dx = x - x0
        dy = y - y0
        dz = z - z0
        rotation = self._ecef_to_ned_rotation()
        ned = rotation.dot(np.array([dx, dy, dz]))
        return ned[0], ned[1], ned[2]

    def ned_to_lld(self, north, east, down):
        """将相对原点的 NED 坐标转换为纬度、经度、深度。"""
        if not np.all(np.isfinite([north, east, down])):
            raise ValueError('NED 输入必须为有限值')
        x0, y0, z0 = self.lld_to_ecef(
            self.init_lat, self.init_lon, self.init_depth
        )
        ecef = self._ecef_to_ned_rotation().T.dot(
            np.array([north, east, down])
        ) + np.array([x0, y0, z0])
        lat_rad, lon_rad, depth = self.ecef_to_lld(*ecef)
        return np.degrees(lat_rad), np.degrees(lon_rad), depth

    def ecef_to_lld(self, x, y, z):
        """将 ECEF 坐标转换为纬度、经度、深度。"""
        if not np.all(np.isfinite([x, y, z])):
            raise ValueError('ECEF 输入必须为有限值')
        lon_rad = np.arctan2(y, x)
        horizontal_distance = np.sqrt(x ** 2 + y ** 2)
        lat_rad = np.arctan2(z, horizontal_distance * (1 - self.e_sq))

        for _ in range(10):
            radius = self.a / np.sqrt(1 - self.e_sq * np.sin(lat_rad) ** 2)
            depth = radius - horizontal_distance / np.cos(lat_rad)
            next_lat_rad = np.arctan2(
                z,
                horizontal_distance * (1 - self.e_sq * radius / (radius - depth)),
            )
            if abs(next_lat_rad - lat_rad) < 1e-12:
                lat_rad = next_lat_rad
                break
            lat_rad = next_lat_rad

        radius = self.a / np.sqrt(1 - self.e_sq * np.sin(lat_rad) ** 2)
        depth = radius - horizontal_distance / np.cos(lat_rad)
        return lat_rad, lon_rad, depth

    @staticmethod
    def _validated_lld(lat, lon, depth, name):
        """校验经纬深有限性和经纬度物理范围。"""
        values = tuple(float(value) for value in (lat, lon, depth))
        if not np.all(np.isfinite(values)):
            raise ValueError('{}必须为有限值'.format(name))
        if not -90.0 <= values[0] <= 90.0:
            raise ValueError('{}纬度必须位于 [-90, 90]'.format(name))
        if not -180.0 <= values[1] <= 180.0:
            raise ValueError('{}经度必须位于 [-180, 180]'.format(name))
        return values

    def _ecef_to_ned_rotation(self):
        """构造从 ECEF 到当前原点 NED 的标准旋转矩阵。"""
        lat0_rad = np.radians(self.init_lat)
        lon0_rad = np.radians(self.init_lon)
        return np.array([
            [-np.sin(lat0_rad) * np.cos(lon0_rad),
             -np.sin(lat0_rad) * np.sin(lon0_rad),
             np.cos(lat0_rad)],
            [-np.sin(lon0_rad), np.cos(lon0_rad), 0],
            [-np.cos(lat0_rad) * np.cos(lon0_rad),
             -np.cos(lat0_rad) * np.sin(lon0_rad),
             -np.sin(lat0_rad)],
        ])
