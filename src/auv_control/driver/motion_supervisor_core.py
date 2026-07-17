#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_supervisor_core.py
功能：运动—刹停—悬停控制器的纯算法核心
作者：BroXu
说明：
    1. 实现航向对准、水平运动、主动刹停、捕获和悬停状态机；
    2. 提供坐标转换、停车距离、力限幅和变化率限制等纯函数；
    3. 本模块不依赖 ROS，ROS 话题、TF 和消息转换由 motion_supervisor.py 负责。
记录：
2026.7.16
    新增纯 Python 状态机核心，支持离线单元测试。
2026.7.16
    根据实测数据增加正负方向独立刹车力、减速度和刹车步进参数。
2026.7.17
    将目标深度统一固定为参数值，避免兜底和取消路径跟随实时深度变化。
2026.7.17
    根据入水日志提高三轴运动力矩，并在最终转向期间持续水平刹停和监测漂移。
2026.7.17
    根据大角度测试修正停车主轴选择、航向刹车参数和定点接管保护。
"""

from __future__ import division

import math


MODE_DEPTH = 2
MODE_DPROV = 4
PROTOCOL_FORCE_LIMIT = 10000

IDLE = 0
ALIGN_PATH = 1
ALIGN_PATH_BRAKE = 2
TRANSLATE = 3
TRANSLATE_BRAKE = 4
ALIGN_FINAL = 5
FINAL_BRAKE = 6
CAPTURE = 7
HOVER = 8
SAFE = 9

STATE_NAMES = {
    IDLE: 'IDLE',
    ALIGN_PATH: 'ALIGN_PATH',
    ALIGN_PATH_BRAKE: 'ALIGN_PATH_BRAKE',
    TRANSLATE: 'TRANSLATE',
    TRANSLATE_BRAKE: 'TRANSLATE_BRAKE',
    ALIGN_FINAL: 'ALIGN_FINAL',
    FINAL_BRAKE: 'FINAL_BRAKE',
    CAPTURE: 'CAPTURE',
    HOVER: 'HOVER',
    SAFE: 'SAFE',
}


def clamp(value, lower, upper):
    """将数值限制到闭区间。"""
    return max(lower, min(upper, value))


def wrap_angle(angle):
    """将弧度角归一化到 [-pi, pi)。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def map_error_to_body(error_north, error_east, yaw):
    """把 map/NED 水平误差旋转到 base_link。"""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * error_north + sine * error_east,
        -sine * error_north + cosine * error_east,
    )


def body_velocity_to_map(forward, lateral, yaw):
    """把 base_link 水平速度旋转到 map/NED。"""
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        cosine * forward - sine * lateral,
        sine * forward + cosine * lateral,
    )


def relative_target_xy(
        initial_x, initial_y, initial_yaw, offset_x, offset_y, offset_frame):
    """把初始 base_link 或 map 下的水平偏置换算为 map 绝对坐标。"""
    if offset_frame == 'base_link':
        cosine = math.cos(initial_yaw)
        sine = math.sin(initial_yaw)
        delta_x = cosine * offset_x - sine * offset_y
        delta_y = sine * offset_x + cosine * offset_y
    elif offset_frame == 'map':
        delta_x = offset_x
        delta_y = offset_y
    else:
        raise ValueError('offset_frame 仅支持 base_link 或 map')
    return initial_x + delta_x, initial_y + delta_y


def stopping_distance(closing_speed, brake_acceleration, delay, margin):
    """根据闭合速度、有效减速度和链路延迟估算停车距离。"""
    speed = max(0.0, closing_speed)
    acceleration = max(1e-6, brake_acceleration)
    return speed * speed / (2.0 * acceleration) + speed * delay + margin


def protocol_force(value):
    """转换成协议允许的有符号整数力。"""
    return int(round(clamp(value, -PROTOCOL_FORCE_LIMIT, PROTOCOL_FORCE_LIMIT)))


def slew_value(previous, desired, maximum_step):
    """限制单周期力变化量。"""
    step = max(0.0, maximum_step)
    return previous + clamp(desired - previous, -step, step)


class MotionGoal(object):
    """map 坐标系下的最终目标。"""

    def __init__(self, x, y, z, yaw):
        values = (x, y, z, yaw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('目标位姿包含非有限值')
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw = wrap_angle(float(yaw))


class VehicleState(object):
    """控制器单周期所需的 AUV 状态。"""

    def __init__(
            self, now, x, y, z, yaw, forward_velocity, lateral_velocity,
            yaw_rate, feedback_fresh=True, reported_mode=None):
        self.now = float(now)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.yaw = wrap_angle(float(yaw))
        self.forward_velocity = float(forward_velocity)
        self.lateral_velocity = float(lateral_velocity)
        self.yaw_rate = float(yaw_rate)
        self.feedback_fresh = bool(feedback_fresh)
        self.reported_mode = reported_mode


class ControlOutput(object):
    """单周期控制输出与诊断量。"""

    def __init__(
            self, state, mode, target, tx, ty, mz, position_error,
            yaw_error, horizontal_speed, yaw_rate, reason, goal_active):
        self.state = state
        self.mode = mode
        self.target = target
        self.tx = tx
        self.ty = ty
        self.mz = mz
        self.position_error = position_error
        self.yaw_error = yaw_error
        self.horizontal_speed = horizontal_speed
        self.yaw_rate = yaw_rate
        self.reason = reason
        self.goal_active = goal_active


DEFAULT_PARAMETERS = {
    'fixed_target_z': -0.6,
    'max_tx': 2000.0,
    'max_ty': 2000.0,
    'max_mz': 1000.0,
    'brake_max_tx_positive': 2000.0,
    'brake_max_tx_negative': 3000.0,
    'brake_max_ty_positive': 2000.0,
    'brake_max_ty_negative': 4000.0,
    'brake_max_mz_positive': 3000.0,
    'brake_max_mz_negative': 3000.0,
    'force_slew_per_cycle': 10000.0,
    'brake_force_slew_per_cycle': 10000.0,
    'kp_x': 2000.0,
    'kp_y': 4000.0,
    'kv_x': 1000.0,
    'kv_y': 2000.0,
    'kp_yaw': 6000.0,
    'kr_yaw': -2000.0,
    'brake_gain_x': 30000.0,
    'brake_gain_y': 40000.0,
    'brake_gain_yaw': -6000.0,
    'brake_min_mz': 100.0,
    'brake_axis_relevance_ratio': 0.20,
    'brake_acceleration_tx_positive': 0.10,
    'brake_acceleration_tx_negative': 0.10,
    'brake_acceleration_ty_positive': 0.05,
    'brake_acceleration_ty_negative': 0.05,
    'angular_brake_acceleration_mz_positive': 0.10,
    'angular_brake_acceleration_mz_negative': 0.10,
    'control_delay': 0.35,
    'brake_margin': 0.10,
    'yaw_brake_margin': math.radians(3.0),
    'capture_radius': 0.15,
    'capture_exit_radius': 0.25,
    'horizontal_speed_threshold': 0.015,
    'yaw_tolerance': math.radians(5.0),
    'path_yaw_tolerance': math.radians(5.0),
    'yaw_rate_threshold': math.radians(0.3),
    'stable_frames': 5,
    'hover_fault_speed': 0.08,
    'hover_fault_yaw_rate': math.radians(2.0),
    'hover_fault_yaw_error': math.radians(10.0),
    'mode_ack_timeout': 1.0,
}


class MotionSupervisorCore(object):
    """分阶段完成航向对准、平移、刹停和定点接管。"""

    def __init__(self, parameters=None):
        self.parameters = dict(DEFAULT_PARAMETERS)
        if parameters:
            self.parameters.update(parameters)
        self._validate_parameters()

        self.state = IDLE
        self.reason = '等待目标'
        self.goal = None
        self.pending_goal = None
        self.cancel_requested = False
        self.path_yaw = 0.0
        self.stable_count = 0
        self.hover_started_at = None
        self.last_tx = 0.0
        self.last_ty = 0.0
        self.last_mz = 0.0

    def _validate_parameters(self):
        if not math.isfinite(self.parameters['fixed_target_z']):
            raise ValueError('fixed_target_z 必须是有限数值')
        positive_names = (
            'max_tx', 'max_ty', 'max_mz',
            'brake_max_tx_positive', 'brake_max_tx_negative',
            'brake_max_ty_positive', 'brake_max_ty_negative',
            'brake_max_mz_positive', 'brake_max_mz_negative',
            'force_slew_per_cycle', 'brake_force_slew_per_cycle',
            'brake_min_mz',
            'brake_acceleration_tx_positive',
            'brake_acceleration_tx_negative',
            'brake_acceleration_ty_positive',
            'brake_acceleration_ty_negative',
            'angular_brake_acceleration_mz_positive',
            'angular_brake_acceleration_mz_negative', 'capture_radius',
            'capture_exit_radius', 'horizontal_speed_threshold',
            'yaw_tolerance', 'path_yaw_tolerance', 'yaw_rate_threshold',
            'stable_frames', 'hover_fault_yaw_rate',
            'hover_fault_yaw_error', 'mode_ack_timeout',
        )
        for name in positive_names:
            if self.parameters[name] <= 0:
                raise ValueError('{} 必须大于 0'.format(name))
        if self.parameters['capture_exit_radius'] <= self.parameters['capture_radius']:
            raise ValueError('capture_exit_radius 必须大于 capture_radius')
        axis_ratio = self.parameters['brake_axis_relevance_ratio']
        if not 0.0 < axis_ratio <= 1.0:
            raise ValueError('brake_axis_relevance_ratio 必须在 (0, 1] 内')
        if self.parameters['brake_min_mz'] > min(
                self.parameters['brake_max_mz_positive'],
                self.parameters['brake_max_mz_negative']):
            raise ValueError('brake_min_mz 不能超过航向刹车限幅')
        for name in (
                'max_tx', 'max_ty', 'max_mz',
                'brake_max_tx_positive', 'brake_max_tx_negative',
                'brake_max_ty_positive', 'brake_max_ty_negative',
                'brake_max_mz_positive', 'brake_max_mz_negative'):
            if self.parameters[name] > PROTOCOL_FORCE_LIMIT:
                raise ValueError('{} 不能超过协议限制 {}'.format(
                    name, PROTOCOL_FORCE_LIMIT))

    @property
    def goal_active(self):
        return self.goal is not None or self.pending_goal is not None

    def set_goal(self, goal):
        """提交目标；运动中收到的新目标先触发刹停。"""
        if not isinstance(goal, MotionGoal):
            raise TypeError('goal 必须是 MotionGoal')
        goal = MotionGoal(
            goal.x,
            goal.y,
            self.parameters['fixed_target_z'],
            goal.yaw,
        )
        if self.goal is None and self.state in (IDLE, SAFE):
            self.pending_goal = goal
            self.reason = '收到新目标'
            return
        self.pending_goal = goal
        self.cancel_requested = False
        self._transition(TRANSLATE_BRAKE, '新目标抢占，先刹停')

    def cancel(self):
        """取消当前运动；停稳后以当前位置进入悬停。"""
        if not self.goal_active and self.state == IDLE:
            return
        self.pending_goal = None
        self.cancel_requested = True
        self._transition(TRANSLATE_BRAKE, '收到取消指令，先刹停')

    def _transition(self, new_state, reason):
        if new_state != self.state:
            self.state = new_state
            self.stable_count = 0
        self.reason = reason
        if new_state != HOVER:
            self.hover_started_at = None

    def _activate_pending_goal(self, vehicle):
        if self.pending_goal is None:
            return False
        self.goal = self.pending_goal
        self.pending_goal = None
        self.cancel_requested = False
        dx = self.goal.x - vehicle.x
        dy = self.goal.y - vehicle.y
        distance = math.hypot(dx, dy)
        self.path_yaw = math.atan2(dy, dx) if distance > 1e-6 else vehicle.yaw
        if distance <= self.parameters['capture_radius']:
            self._transition(ALIGN_FINAL, '已在目标位置附近，调整最终航向')
        else:
            self._transition(ALIGN_PATH, '对准目标运动方向')
        return True

    def _goal_metrics(self, vehicle):
        if self.goal is None:
            return 0.0, 0.0, 0.0, 0.0
        dx = self.goal.x - vehicle.x
        dy = self.goal.y - vehicle.y
        return dx, dy, math.hypot(dx, dy), wrap_angle(self.goal.yaw - vehicle.yaw)

    def _stable(self, condition, frames=None):
        required = int(frames or self.parameters['stable_frames'])
        if condition:
            self.stable_count += 1
        else:
            self.stable_count = 0
        return self.stable_count >= required

    def _directional_parameter(self, prefix, command):
        """按实际输出力符号选择正向或负向参数。"""
        positive = self.parameters[prefix + '_positive']
        negative = self.parameters[prefix + '_negative']
        if command > 0.0:
            return positive
        if command < 0.0:
            return negative
        return min(positive, negative)

    def _angular_stop_threshold(self, yaw_rate):
        brake_command = (
            -self.parameters['brake_gain_yaw'] * yaw_rate
        )
        acceleration = self._directional_parameter(
            'angular_brake_acceleration_mz', brake_command)
        return (
            yaw_rate * yaw_rate / (2.0 * acceleration)
            + self.parameters['yaw_brake_margin']
        )

    def _horizontal_brake_acceleration(
            self, vehicle, error_x_body, error_y_body):
        """忽略微小伴随分量，按主要运动轴选择保守有效减速度。"""
        reference_x = (
            vehicle.forward_velocity
            if abs(vehicle.forward_velocity) > 1e-6
            else error_x_body
        )
        reference_y = (
            vehicle.lateral_velocity
            if abs(vehicle.lateral_velocity) > 1e-6
            else error_y_body
        )
        tx_command = -self.parameters['brake_gain_x'] * reference_x
        ty_command = -self.parameters['brake_gain_y'] * reference_y
        ratio = self.parameters['brake_axis_relevance_ratio']
        error_scale = max(abs(error_x_body), abs(error_y_body), 1e-6)
        speed_scale = max(
            abs(vehicle.forward_velocity),
            abs(vehicle.lateral_velocity),
            1e-6,
        )
        x_relevant = (
            abs(error_x_body) >= ratio * error_scale
            or abs(vehicle.forward_velocity) >= ratio * speed_scale
        )
        y_relevant = (
            abs(error_y_body) >= ratio * error_scale
            or abs(vehicle.lateral_velocity) >= ratio * speed_scale
        )
        accelerations = []
        if x_relevant:
            accelerations.append(self._directional_parameter(
                'brake_acceleration_tx', tx_command))
        if y_relevant:
            accelerations.append(self._directional_parameter(
                'brake_acceleration_ty', ty_command))
        if not accelerations:
            accelerations.extend((
                min(
                    self.parameters['brake_acceleration_tx_positive'],
                    self.parameters['brake_acceleration_tx_negative'],
                ),
                min(
                    self.parameters['brake_acceleration_ty_positive'],
                    self.parameters['brake_acceleration_ty_negative'],
                ),
            ))
        return min(accelerations)

    def _yaw_brake_command(self, yaw_rate):
        """角速度超过停稳阈值时，保证刹转力矩越过实测有效下限。"""
        command = -self.parameters['brake_gain_yaw'] * yaw_rate
        if (
                abs(yaw_rate) > self.parameters['yaw_rate_threshold']
                and abs(command) < self.parameters['brake_min_mz']):
            command = math.copysign(
                self.parameters['brake_min_mz'], command)
        return command

    def _limited_forces(self, tx, ty, mz, braking=False, immediate_zero=False):
        if immediate_zero:
            self.last_tx = self.last_ty = self.last_mz = 0.0
            return 0, 0, 0

        if braking:
            tx = clamp(
                tx,
                -self.parameters['brake_max_tx_negative'],
                self.parameters['brake_max_tx_positive'],
            )
            ty = clamp(
                ty,
                -self.parameters['brake_max_ty_negative'],
                self.parameters['brake_max_ty_positive'],
            )
            mz = clamp(
                mz,
                -self.parameters['brake_max_mz_negative'],
                self.parameters['brake_max_mz_positive'],
            )
            maximum_step = self.parameters['brake_force_slew_per_cycle']
        else:
            tx = clamp(tx, -self.parameters['max_tx'], self.parameters['max_tx'])
            ty = clamp(ty, -self.parameters['max_ty'], self.parameters['max_ty'])
            mz = clamp(mz, -self.parameters['max_mz'], self.parameters['max_mz'])
            maximum_step = self.parameters['force_slew_per_cycle']
        tx = slew_value(self.last_tx, tx, maximum_step)
        ty = slew_value(self.last_ty, ty, maximum_step)
        mz = slew_value(self.last_mz, mz, maximum_step)
        self.last_tx, self.last_ty, self.last_mz = tx, ty, mz
        return protocol_force(tx), protocol_force(ty), protocol_force(mz)

    def _output(self, vehicle, mode, tx=0.0, ty=0.0, mz=0.0,
                braking=False, immediate_zero=False):
        source_target = self.goal or self.pending_goal
        if source_target is None:
            source_target = MotionGoal(
                vehicle.x,
                vehicle.y,
                self.parameters['fixed_target_z'],
                vehicle.yaw,
            )
        target = MotionGoal(
            source_target.x,
            source_target.y,
            self.parameters['fixed_target_z'],
            source_target.yaw,
        )
        dx, dy, distance, yaw_error = self._goal_metrics(vehicle)
        del dx, dy
        horizontal_speed = math.hypot(
            vehicle.forward_velocity, vehicle.lateral_velocity)
        tx, ty, mz = self._limited_forces(
            tx, ty, mz, braking=braking, immediate_zero=immediate_zero)
        return ControlOutput(
            self.state, mode, target, tx, ty, mz, distance, yaw_error,
            horizontal_speed, vehicle.yaw_rate, self.reason, self.goal_active)

    def _brake_output(self, vehicle):
        return self._output(
            vehicle,
            MODE_DEPTH,
            -self.parameters['brake_gain_x'] * vehicle.forward_velocity,
            -self.parameters['brake_gain_y'] * vehicle.lateral_velocity,
            self._yaw_brake_command(vehicle.yaw_rate),
            braking=True,
        )

    def _final_alignment_output(self, vehicle, yaw_error):
        """在目标附近转向时持续消除水平速度，并限制正常转向力矩。"""
        mz = (
            self.parameters['kp_yaw'] * yaw_error
            - self.parameters['kr_yaw'] * vehicle.yaw_rate
        )
        mz = clamp(
            mz,
            -self.parameters['max_mz'],
            self.parameters['max_mz'],
        )
        return self._output(
            vehicle,
            MODE_DEPTH,
            -self.parameters['brake_gain_x'] * vehicle.forward_velocity,
            -self.parameters['brake_gain_y'] * vehicle.lateral_velocity,
            mz,
            braking=True,
        )

    def step(self, vehicle):
        """推进一个控制周期。"""
        if not isinstance(vehicle, VehicleState):
            raise TypeError('vehicle 必须是 VehicleState')

        speed = math.hypot(vehicle.forward_velocity, vehicle.lateral_velocity)
        yaw_rate_abs = abs(vehicle.yaw_rate)

        if not vehicle.feedback_fresh:
            self._transition(SAFE, 'TF 或速度反馈超时')
            return self._output(
                vehicle, MODE_DEPTH, immediate_zero=True)

        if self.state == SAFE:
            if self.goal_active or self.cancel_requested:
                self._transition(TRANSLATE_BRAKE, '反馈恢复，先确认停稳')
            else:
                self._transition(IDLE, '反馈恢复，等待目标')

        if self.state == IDLE:
            if not self._activate_pending_goal(vehicle):
                return self._output(
                    vehicle, MODE_DEPTH, immediate_zero=True)

        if self.state == ALIGN_PATH:
            error = wrap_angle(self.path_yaw - vehicle.yaw)
            if abs(error) <= max(
                    self.parameters['path_yaw_tolerance'],
                    self._angular_stop_threshold(vehicle.yaw_rate)):
                self._transition(ALIGN_PATH_BRAKE, '路径航向进入刹转区')
                return self._brake_output(vehicle)
            mz = (
                self.parameters['kp_yaw'] * error
                - self.parameters['kr_yaw'] * vehicle.yaw_rate
            )
            return self._output(vehicle, MODE_DEPTH, mz=mz)

        if self.state == ALIGN_PATH_BRAKE:
            error = wrap_angle(self.path_yaw - vehicle.yaw)
            if abs(error) > 2.0 * self.parameters['path_yaw_tolerance']:
                self._transition(ALIGN_PATH, '刹转后航向偏差过大，重新对准')
                return self._output(vehicle, MODE_DEPTH)
            if self._stable(
                    abs(error) <= self.parameters['path_yaw_tolerance']
                    and yaw_rate_abs <= self.parameters['yaw_rate_threshold']):
                self._transition(TRANSLATE, '路径航向稳定，开始平移')
            return self._brake_output(vehicle)

        if self.state == TRANSLATE:
            dx, dy, distance, unused_yaw_error = self._goal_metrics(vehicle)
            del unused_yaw_error
            error_x, error_y = map_error_to_body(dx, dy, vehicle.yaw)
            north_velocity, east_velocity = body_velocity_to_map(
                vehicle.forward_velocity, vehicle.lateral_velocity, vehicle.yaw)
            if distance > 1e-6:
                closing_speed = (
                    north_velocity * dx / distance
                    + east_velocity * dy / distance
                )
            else:
                closing_speed = 0.0
            brake_acceleration = self._horizontal_brake_acceleration(
                vehicle, error_x, error_y)
            stop_distance = stopping_distance(
                closing_speed,
                brake_acceleration,
                self.parameters['control_delay'],
                self.parameters['brake_margin'],
            )
            if distance <= max(self.parameters['capture_radius'], stop_distance):
                self._transition(TRANSLATE_BRAKE, '进入水平停车距离')
                return self._brake_output(vehicle)

            tx = (
                self.parameters['kp_x'] * error_x
                - self.parameters['kv_x'] * vehicle.forward_velocity
            )
            ty = (
                self.parameters['kp_y'] * error_y
                - self.parameters['kv_y'] * vehicle.lateral_velocity
            )
            path_error = wrap_angle(self.path_yaw - vehicle.yaw)
            mz = (
                self.parameters['kp_yaw'] * path_error
                - self.parameters['kr_yaw'] * vehicle.yaw_rate
            )
            return self._output(vehicle, MODE_DEPTH, tx=tx, ty=ty, mz=mz)

        if self.state == TRANSLATE_BRAKE:
            stopped = (
                speed <= self.parameters['horizontal_speed_threshold']
                and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
            )
            if self._stable(stopped):
                if self.pending_goal is not None:
                    self._activate_pending_goal(vehicle)
                elif self.cancel_requested:
                    self.goal = MotionGoal(
                        vehicle.x,
                        vehicle.y,
                        self.parameters['fixed_target_z'],
                        vehicle.yaw,
                    )
                    self.cancel_requested = False
                    self._transition(CAPTURE, '取消后已停稳，悬停当前位置')
                elif self.goal is not None:
                    dx, dy, distance, unused_yaw_error = self._goal_metrics(vehicle)
                    del dx, dy, unused_yaw_error
                    if distance > self.parameters['capture_radius']:
                        self.path_yaw = math.atan2(
                            self.goal.y - vehicle.y, self.goal.x - vehicle.x)
                        self._transition(ALIGN_PATH, '刹停点尚未进入目标区，重新接近')
                    else:
                        self._transition(ALIGN_FINAL, '水平运动停稳，调整最终航向')
            return self._brake_output(vehicle)

        if self.state == ALIGN_FINAL:
            dx, dy, distance, error = self._goal_metrics(vehicle)
            del dx, dy
            if distance > self.parameters['capture_exit_radius']:
                self.path_yaw = math.atan2(
                    self.goal.y - vehicle.y, self.goal.x - vehicle.x)
                self._transition(ALIGN_PATH, '最终转向时漂出捕获区，重新接近')
                return self._brake_output(vehicle)
            if abs(error) <= max(
                    self.parameters['yaw_tolerance'],
                    self._angular_stop_threshold(vehicle.yaw_rate)):
                self._transition(FINAL_BRAKE, '最终航向进入刹转区')
                return self._brake_output(vehicle)
            return self._final_alignment_output(vehicle, error)

        if self.state == FINAL_BRAKE:
            dx, dy, distance, yaw_error = self._goal_metrics(vehicle)
            del dx, dy
            if distance > self.parameters['capture_exit_radius']:
                self.path_yaw = math.atan2(
                    self.goal.y - vehicle.y, self.goal.x - vehicle.x)
                self._transition(ALIGN_PATH, '最终刹转时漂出捕获区')
                return self._brake_output(vehicle)
            if abs(yaw_error) > 2.0 * self.parameters['yaw_tolerance']:
                self._transition(ALIGN_FINAL, '最终航向偏差过大，重新调整')
                return self._output(vehicle, MODE_DEPTH)
            pose_stopped = (
                abs(yaw_error) <= self.parameters['yaw_tolerance']
                and speed <= self.parameters['horizontal_speed_threshold']
                and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
            )
            if (
                    distance > self.parameters['capture_radius']
                    and pose_stopped):
                self.path_yaw = math.atan2(
                    self.goal.y - vehicle.y, self.goal.x - vehicle.x)
                self._transition(ALIGN_PATH, '最终刹停点仍在捕获区外，重新接近')
                return self._brake_output(vehicle)
            if self._stable(
                    distance <= self.parameters['capture_radius']
                    and pose_stopped):
                self._transition(HOVER, '目标位姿刹停稳定，切换下位机定点')
                self.hover_started_at = vehicle.now
                return self._output(
                    vehicle, MODE_DPROV, immediate_zero=True)
            return self._brake_output(vehicle)

        if self.state == CAPTURE:
            dx, dy, distance, yaw_error = self._goal_metrics(vehicle)
            del dx, dy
            if distance > self.parameters['capture_exit_radius']:
                self.path_yaw = math.atan2(
                    self.goal.y - vehicle.y, self.goal.x - vehicle.x)
                self._transition(ALIGN_PATH, '捕获期间漂出位置范围')
                return self._output(vehicle, MODE_DEPTH)
            if abs(yaw_error) > 2.0 * self.parameters['yaw_tolerance']:
                self._transition(ALIGN_FINAL, '捕获期间航向偏差过大')
                return self._output(vehicle, MODE_DEPTH)
            captured = (
                distance <= self.parameters['capture_radius']
                and abs(yaw_error) <= self.parameters['yaw_tolerance']
                and speed <= self.parameters['horizontal_speed_threshold']
                and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
            )
            if self._stable(captured):
                self._transition(HOVER, '捕获稳定，切换下位机定点')
                self.hover_started_at = vehicle.now
                return self._output(
                    vehicle, MODE_DPROV, immediate_zero=True)
            return self._brake_output(vehicle)

        if self.state == HOVER:
            unused_dx, unused_dy, hover_distance, hover_yaw_error = (
                self._goal_metrics(vehicle))
            del unused_dx, unused_dy
            if speed > self.parameters['hover_fault_speed']:
                self._transition(TRANSLATE_BRAKE, '定点接管后水平速度异常')
                return self._brake_output(vehicle)
            if hover_distance > self.parameters['capture_exit_radius']:
                self._transition(TRANSLATE_BRAKE, '定点接管后位置误差超限')
                return self._brake_output(vehicle)
            if yaw_rate_abs > self.parameters['hover_fault_yaw_rate']:
                self._transition(TRANSLATE_BRAKE, '定点接管后航向角速度异常')
                return self._brake_output(vehicle)
            if abs(hover_yaw_error) > self.parameters['hover_fault_yaw_error']:
                self._transition(TRANSLATE_BRAKE, '定点接管后航向误差异常')
                return self._brake_output(vehicle)
            if (
                    vehicle.reported_mode != MODE_DPROV
                    and self.hover_started_at is not None
                    and vehicle.now - self.hover_started_at
                    > self.parameters['mode_ack_timeout']):
                self._transition(SAFE, '定点模式确认超时')
                return self._output(
                    vehicle, MODE_DEPTH, immediate_zero=True)
            return self._output(
                vehicle, MODE_DPROV, immediate_zero=True)

        self._transition(SAFE, '未知状态')
        return self._output(vehicle, MODE_DEPTH, immediate_zero=True)
