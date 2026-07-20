#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_supervisor_core.py
功能：运动—刹停—悬停控制器的纯算法核心
作者：BroXu
说明：
    1. 实现保持当前航向平移、主动刹停、最终转向和悬停状态机；
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
2026.7.18
    目标深度改为跟随 goal，取消路径预对准并增加连续目标切换机制。
2026.7.18
    收紧 HOVER 语义，仅在下位机反馈 mode=4 后表示目标到达并完成定点接管。
2026.7.18
    水平 X/Y 改为独立 TRACK、BRAKE、HOLD 子状态，连续目标始终采用最新值。
    正常控制、主动刹车、限幅、减速度和停车余量均支持按输出正负方向配置。
2026.7.18
    最终转向和刹转阶段增加 control_link 平面位置保持，固定实际旋转中心。
2026.7.19
    增加计划 yaw 方向锁存逻辑，主动刹转阶段不随 MZ 反号切换旋转中心。
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

AXIS_HOLD = 0
AXIS_TRACK = 1
AXIS_BRAKE = 2

AXIS_STATE_NAMES = {
    AXIS_HOLD: 'HOLD',
    AXIS_TRACK: 'TRACK',
    AXIS_BRAKE: 'BRAKE',
}

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


def select_planned_rotation_direction(
        yaw_error, current_direction=1, direction_locked=False,
        deadband=0.0):
    """按计划 yaw 误差选择正负方向；锁存或死区内保持当前方向。"""
    values = (yaw_error, current_direction, deadband)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError('计划旋转方向参数必须为有限值')
    if float(deadband) < 0.0:
        raise ValueError('计划旋转方向死区不能为负数')
    current_direction = 1 if float(current_direction) >= 0.0 else -1
    if direction_locked or abs(float(yaw_error)) <= float(deadband):
        return current_direction
    return 1 if float(yaw_error) > 0.0 else -1


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
            yaw_rate, feedback_fresh=True, reported_mode=None,
            reported_mode_stamp=None):
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
        self.reported_mode_stamp = (
            None
            if reported_mode_stamp is None
            else float(reported_mode_stamp)
        )


class ControlOutput(object):
    """单周期控制输出与诊断量。"""

    def __init__(
            self, state, mode, target, tx, ty, mz, position_error,
            yaw_error, horizontal_speed, yaw_rate, reason, goal_active,
            x_axis_state=AXIS_HOLD, y_axis_state=AXIS_HOLD,
            yaw_axis_state=AXIS_HOLD, x_error=0.0, y_error=0.0,
            x_speed=0.0, y_speed=0.0):
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
        self.x_axis_state = x_axis_state
        self.y_axis_state = y_axis_state
        self.yaw_axis_state = yaw_axis_state
        self.x_error = x_error
        self.y_error = y_error
        self.x_speed = x_speed
        self.y_speed = y_speed


DEFAULT_PARAMETERS = {
    'max_tx_positive': 2000.0,
    'max_tx_negative': 2000.0,
    'max_ty_positive': 2000.0,
    'max_ty_negative': 2000.0,
    'max_mz_positive': 1000.0,
    'max_mz_negative': 1000.0,
    'brake_max_tx_positive': 2000.0,
    'brake_max_tx_negative': 3000.0,
    'brake_max_ty_positive': 2000.0,
    'brake_max_ty_negative': 4000.0,
    'brake_max_mz_positive': 3000.0,
    'brake_max_mz_negative': 3000.0,
    'force_slew_per_cycle': 10000.0,
    'brake_force_slew_per_cycle': 10000.0,
    'kp_x_positive': 3000.0,
    'kp_x_negative': 3000.0,
    'kp_y_positive': 4000.0,
    'kp_y_negative': 4000.0,
    'kv_x_positive': 1000.0,
    'kv_x_negative': 1000.0,
    'kv_y_positive': 2000.0,
    'kv_y_negative': 2000.0,
    'kp_yaw_positive': 6000.0,
    'kp_yaw_negative': 6000.0,
    'kr_yaw_positive': -2000.0,
    'kr_yaw_negative': -2000.0,
    'brake_gain_tx_positive': 30000.0,
    'brake_gain_tx_negative': 30000.0,
    'brake_gain_ty_positive': 40000.0,
    'brake_gain_ty_negative': 40000.0,
    'brake_gain_mz_positive': -6000.0,
    'brake_gain_mz_negative': -6000.0,
    'brake_min_mz': 100.0,
    'brake_acceleration_tx_positive': 0.10,
    'brake_acceleration_tx_negative': 0.10,
    'brake_acceleration_ty_positive': 0.05,
    'brake_acceleration_ty_negative': 0.05,
    'angular_brake_acceleration_mz_positive': 0.025,
    'angular_brake_acceleration_mz_negative': 0.040,
    'control_delay': 0.35,
    'brake_margin_tx_positive': 0.10,
    'brake_margin_tx_negative': 0.10,
    'brake_margin_ty_positive': 0.10,
    'brake_margin_ty_negative': 0.10,
    'yaw_brake_margin_positive': math.radians(3.0),
    'yaw_brake_margin_negative': math.radians(3.0),
    'capture_radius': 0.15,
    'capture_exit_radius': 0.25,
    'control_center_hold_tolerance': 0.03,
    'axis_brake_exit_hysteresis': 0.05,
    'horizontal_speed_threshold': 0.015,
    'yaw_tolerance': math.radians(5.0),
    'yaw_rate_threshold': math.radians(0.3),
    'stable_frames': 5,
    'hover_fault_position_error': 0.40,
    'hover_fault_speed': 0.15,
    'hover_fault_yaw_rate': math.radians(5.0),
    'hover_fault_yaw_error': math.radians(20.0),
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
        self.hover_direct_update = False
        self.recovery_brake_requested = False
        self.translation_yaw = 0.0
        self.x_axis_state = AXIS_HOLD
        self.y_axis_state = AXIS_HOLD
        self.yaw_axis_state = AXIS_HOLD
        self.stable_count = 0
        self.handover_started_at = None
        self.last_tx = 0.0
        self.last_ty = 0.0
        self.last_mz = 0.0

    def _validate_parameters(self):
        positive_names = (
            'max_tx_positive', 'max_tx_negative',
            'max_ty_positive', 'max_ty_negative',
            'max_mz_positive', 'max_mz_negative',
            'brake_max_tx_positive', 'brake_max_tx_negative',
            'brake_max_ty_positive', 'brake_max_ty_negative',
            'brake_max_mz_positive', 'brake_max_mz_negative',
            'force_slew_per_cycle', 'brake_force_slew_per_cycle',
            'kp_x_positive', 'kp_x_negative',
            'kp_y_positive', 'kp_y_negative',
            'kv_x_positive', 'kv_x_negative',
            'kv_y_positive', 'kv_y_negative',
            'kp_yaw_positive', 'kp_yaw_negative',
            'brake_gain_tx_positive', 'brake_gain_tx_negative',
            'brake_gain_ty_positive', 'brake_gain_ty_negative',
            'brake_min_mz',
            'brake_acceleration_tx_positive',
            'brake_acceleration_tx_negative',
            'brake_acceleration_ty_positive',
            'brake_acceleration_ty_negative',
            'angular_brake_acceleration_mz_positive',
            'angular_brake_acceleration_mz_negative', 'capture_radius',
            'capture_exit_radius', 'control_center_hold_tolerance',
            'axis_brake_exit_hysteresis',
            'horizontal_speed_threshold',
            'yaw_tolerance', 'yaw_rate_threshold',
            'stable_frames', 'hover_fault_position_error',
            'hover_fault_speed', 'hover_fault_yaw_rate',
            'hover_fault_yaw_error', 'mode_ack_timeout',
            'brake_margin_tx_positive', 'brake_margin_tx_negative',
            'brake_margin_ty_positive', 'brake_margin_ty_negative',
            'yaw_brake_margin_positive', 'yaw_brake_margin_negative',
        )
        for name in positive_names:
            if self.parameters[name] <= 0:
                raise ValueError('{} 必须大于 0'.format(name))
        for name, value in self.parameters.items():
            if not math.isfinite(float(value)):
                raise ValueError('{} 必须为有限值'.format(name))
        for name in (
                'kr_yaw_positive', 'kr_yaw_negative',
                'brake_gain_mz_positive', 'brake_gain_mz_negative'):
            if self.parameters[name] == 0:
                raise ValueError('{} 不能为 0'.format(name))
        if self.parameters['capture_exit_radius'] <= self.parameters['capture_radius']:
            raise ValueError('capture_exit_radius 必须大于 capture_radius')
        if self.parameters['control_center_hold_tolerance'] >= (
                self.parameters['capture_radius']):
            raise ValueError('control_center_hold_tolerance 二维死区二维死区必须小于 capture_radius')
        if self.parameters['hover_fault_position_error'] <= (
                self.parameters['capture_exit_radius']):
            raise ValueError(
                'hover_fault_position_error 必须大于 capture_exit_radius')
        if self.parameters['hover_fault_speed'] <= (
                self.parameters['horizontal_speed_threshold']):
            raise ValueError(
                'hover_fault_speed 必须大于 horizontal_speed_threshold')
        if self.parameters['hover_fault_yaw_rate'] <= (
                self.parameters['yaw_rate_threshold']):
            raise ValueError(
                'hover_fault_yaw_rate 必须大于 yaw_rate_threshold')
        if self.parameters['hover_fault_yaw_error'] <= (
                self.parameters['yaw_tolerance']):
            raise ValueError(
                'hover_fault_yaw_error 必须大于 yaw_tolerance')
        if self.parameters['brake_min_mz'] > min(
                self.parameters['brake_max_mz_positive'],
                self.parameters['brake_max_mz_negative']):
            raise ValueError('brake_min_mz 不能超过航向刹车限幅')
        for name in (
                'max_tx_positive', 'max_tx_negative',
                'max_ty_positive', 'max_ty_negative',
                'max_mz_positive', 'max_mz_negative',
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
        """始终采用最新目标，由各轴独立决定跟踪、刹车或保持。"""
        if not isinstance(goal, MotionGoal):
            raise TypeError('goal 必须是 MotionGoal')

        if self.goal is None:
            self.pending_goal = goal
            self.hover_direct_update = False
            self.reason = '收到首个目标'
            return

        changed = (
            abs(goal.x - self.goal.x) > 1e-6
            or abs(goal.y - self.goal.y) > 1e-6
            or abs(goal.z - self.goal.z) > 1e-6
            or abs(wrap_angle(goal.yaw - self.goal.yaw)) > 1e-6
        )
        self.goal = goal
        self.pending_goal = None
        self.cancel_requested = False
        self.hover_direct_update = self.state == HOVER and changed
        if self.state == CAPTURE and changed:
            self.handover_started_at = None
            self.stable_count = 0
        if changed:
            self.reason = '采用最新目标，由各轴独立切换'

    def cancel(self):
        """取消当前运动；停稳后以当前位置进入悬停。"""
        if not self.goal_active and self.state == IDLE:
            return
        self.pending_goal = None
        self.cancel_requested = True
        self.recovery_brake_requested = False
        self.hover_direct_update = False
        self.x_axis_state = AXIS_BRAKE
        self.y_axis_state = AXIS_BRAKE
        self.yaw_axis_state = AXIS_BRAKE
        self._transition(TRANSLATE_BRAKE, '收到取消指令，先刹停')

    def _transition(self, new_state, reason):
        if new_state != self.state:
            self.state = new_state
            self.stable_count = 0
        self.reason = reason
        if new_state != CAPTURE:
            self.handover_started_at = None

    def _activate_pending_goal(self, vehicle):
        if self.pending_goal is None:
            return False
        self.goal = self.pending_goal
        self.pending_goal = None
        self.cancel_requested = False
        self.hover_direct_update = False
        dx = self.goal.x - vehicle.x
        dy = self.goal.y - vehicle.y
        distance = math.hypot(dx, dy)
        self.translation_yaw = vehicle.yaw
        self.x_axis_state = AXIS_HOLD
        self.y_axis_state = AXIS_HOLD
        self.yaw_axis_state = AXIS_HOLD
        if distance <= self.parameters['capture_radius']:
            self._transition(ALIGN_FINAL, '已在目标位置附近，调整最终航向')
        else:
            self.x_axis_state = AXIS_TRACK
            self.y_axis_state = AXIS_TRACK
            self._transition(TRANSLATE, '保持当前航向，开始水平平移')
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

    def _motion_parameter(self, prefix, error):
        """正常跟踪按目标误差方向选择参数。"""
        if error >= 0.0:
            return self.parameters[prefix + '_positive']
        return self.parameters[prefix + '_negative']

    def _brake_command(self, force_prefix, velocity):
        """按最终刹车输出符号选择对应增益。"""
        positive_gain = self.parameters[
            'brake_gain_' + force_prefix + '_positive']
        negative_gain = self.parameters[
            'brake_gain_' + force_prefix + '_negative']
        positive_candidate = -positive_gain * velocity
        negative_candidate = -negative_gain * velocity
        if positive_candidate > 0.0:
            return positive_candidate
        if negative_candidate < 0.0:
            return negative_candidate
        return (
            positive_candidate
            if abs(positive_candidate) >= abs(negative_candidate)
            else negative_candidate
        )

    def _angular_stop_threshold(self, yaw_rate):
        brake_command = self._brake_command('mz', yaw_rate)
        acceleration = self._directional_parameter(
            'angular_brake_acceleration_mz', brake_command)
        margin = self._directional_parameter(
            'yaw_brake_margin', brake_command)
        return (
            yaw_rate * yaw_rate / (2.0 * acceleration)
            + margin
        )

    def _yaw_brake_command(self, yaw_rate):
        """角速度超过停稳阈值时，保证刹转力矩越过实测有效下限。"""
        command = self._brake_command('mz', yaw_rate)
        if (
                abs(yaw_rate) > self.parameters['yaw_rate_threshold']
                and abs(command) < self.parameters['brake_min_mz']):
            command = math.copysign(
                self.parameters['brake_min_mz'], command)
        return command

    def _axis_stop_distance(self, force_prefix, error, velocity):
        """计算一个本体水平轴的有向停车距离。"""
        if abs(error) <= 1e-9:
            closing_speed = abs(velocity)
        else:
            closing_speed = max(
                0.0,
                velocity if error > 0.0 else -velocity,
            )
        brake_command = self._brake_command(force_prefix, velocity)
        acceleration = self._directional_parameter(
            'brake_acceleration_' + force_prefix, brake_command)
        margin = self._directional_parameter(
            'brake_margin_' + force_prefix, brake_command)
        return stopping_distance(
            closing_speed,
            acceleration,
            self.parameters['control_delay'],
            margin,
        )

    def _axis_track_command(self, axis_name, error, velocity):
        """计算水平轴 TRACK 输出，不执行单轴捕获状态切换。"""
        kp_prefix = 'kp_' + axis_name
        kv_prefix = 'kv_' + axis_name
        kp = self._motion_parameter(kp_prefix, error)
        kv = self._motion_parameter(kv_prefix, error)
        return kp * error - kv * velocity

    def _axis_track_command(self, axis_name, error, velocity):
        """计算水平轴 TRACK 输出，不执行单轴捕获状态切换。"""
        kp_prefix = 'kp_' + axis_name
        kv_prefix = 'kv_' + axis_name
        kp = self._motion_parameter(kp_prefix, error)
        kv = self._motion_parameter(kv_prefix, error)
        return kp * error - kv * velocity

    def _axis_control(self, axis_name, state, error, velocity):
        """推进单个水平轴的 TRACK、BRAKE、HOLD 子状态。"""
        force_prefix = 'tx' if axis_name == 'x' else 'ty'
        threshold = self.parameters['horizontal_speed_threshold']
        capture = self.parameters['capture_radius']
        stop_distance = self._axis_stop_distance(
            force_prefix, error, velocity)
        moving_toward = error * velocity > 0.0

        if state == AXIS_HOLD:
            if abs(velocity) > threshold:
                state = AXIS_BRAKE
            elif abs(error) > capture:
                state = AXIS_TRACK
        elif state == AXIS_TRACK:
            if (
                    abs(error) <= capture
                    and abs(velocity) <= threshold):
                state = AXIS_HOLD
            elif moving_toward and abs(error) <= stop_distance:
                state = AXIS_BRAKE
        elif state == AXIS_BRAKE:
            if abs(velocity) <= threshold:
                state = (
                    AXIS_HOLD
                    if abs(error) <= capture
                    else AXIS_TRACK
                )
            elif (
                    moving_toward
                    and abs(error) > (
                        stop_distance
                        + self.parameters['axis_brake_exit_hysteresis'])):
                state = AXIS_TRACK
        else:
            state = AXIS_BRAKE

        if state == AXIS_TRACK:
            return (
                state,
                self._axis_track_command(axis_name, error, velocity),
                False,
            )
            return (
                state,
                self._axis_track_command(axis_name, error, velocity),
                False,
            )
        if state == AXIS_BRAKE:
            return (
                state,
                self._brake_command(force_prefix, velocity),
                True,
            )
        return state, 0.0, False

    def _axis_force_limit(self, force_prefix, value, braking):
        """按轴、控制阶段和实际输出符号限制力。"""
        limit_prefix = (
            'brake_max_' + force_prefix
            if braking
            else 'max_' + force_prefix
        )
        return clamp(
            value,
            -self.parameters[limit_prefix + '_negative'],
            self.parameters[limit_prefix + '_positive'],
        )

    def _limited_forces(
            self, tx, ty, mz, x_braking=False, y_braking=False,
            yaw_braking=False, immediate_zero=False):
        if immediate_zero:
            self.last_tx = self.last_ty = self.last_mz = 0.0
            return 0, 0, 0

        tx = self._axis_force_limit('tx', tx, x_braking)
        ty = self._axis_force_limit('ty', ty, y_braking)
        mz = self._axis_force_limit('mz', mz, yaw_braking)
        normal_step = self.parameters['force_slew_per_cycle']
        brake_step = self.parameters['brake_force_slew_per_cycle']
        tx = slew_value(
            self.last_tx, tx, brake_step if x_braking else normal_step)
        ty = slew_value(
            self.last_ty, ty, brake_step if y_braking else normal_step)
        mz = slew_value(
            self.last_mz, mz, brake_step if yaw_braking else normal_step)
        self.last_tx, self.last_ty, self.last_mz = tx, ty, mz
        return protocol_force(tx), protocol_force(ty), protocol_force(mz)

    def _output(self, vehicle, mode, tx=0.0, ty=0.0, mz=0.0,
                x_braking=False, y_braking=False, yaw_braking=False,
                immediate_zero=False):
        source_target = self.goal or self.pending_goal
        if source_target is None:
            source_target = MotionGoal(
                vehicle.x,
                vehicle.y,
                vehicle.z,
                vehicle.yaw,
            )
        target = source_target
        dx, dy, distance, yaw_error = self._goal_metrics(vehicle)
        del dx, dy
        horizontal_speed = math.hypot(
            vehicle.forward_velocity, vehicle.lateral_velocity)
        tx, ty, mz = self._limited_forces(
            tx, ty, mz,
            x_braking=x_braking,
            y_braking=y_braking,
            yaw_braking=yaw_braking,
            immediate_zero=immediate_zero)
        error_x, error_y = map_error_to_body(
            target.x - vehicle.x,
            target.y - vehicle.y,
            vehicle.yaw,
        )
        return ControlOutput(
            self.state, mode, target, tx, ty, mz, distance, yaw_error,
            horizontal_speed, vehicle.yaw_rate, self.reason, self.goal_active,
            x_axis_state=self.x_axis_state,
            y_axis_state=self.y_axis_state,
            yaw_axis_state=self.yaw_axis_state,
            x_error=error_x,
            y_error=error_y,
            x_speed=vehicle.forward_velocity,
            y_speed=vehicle.lateral_velocity,
        )

    def _brake_output(self, vehicle):
        self.x_axis_state = AXIS_BRAKE
        self.y_axis_state = AXIS_BRAKE
        self.yaw_axis_state = AXIS_BRAKE
        return self._output(
            vehicle,
            MODE_DEPTH,
            self._brake_command('tx', vehicle.forward_velocity),
            self._brake_command('ty', vehicle.lateral_velocity),
            self._yaw_brake_command(vehicle.yaw_rate),
            x_braking=True,
            y_braking=True,
            yaw_braking=True,
        )

    def _center_hold_commands(self, vehicle):
        """用二维圆形死区保持 control_link，并同时补偿 X/Y 耦合漂移。"""
        """用二维圆形死区保持 control_link，并同时补偿 X/Y 耦合漂移。"""
        dx, dy, unused_distance, unused_yaw_error = (
            self._goal_metrics(vehicle))
        del unused_distance, unused_yaw_error
        error_x, error_y = map_error_to_body(dx, dy, vehicle.yaw)
        center_stable = (
            math.hypot(error_x, error_y)
            <= self.parameters['control_center_hold_tolerance']
            and math.hypot(
                vehicle.forward_velocity,
                vehicle.lateral_velocity,
            ) <= self.parameters['horizontal_speed_threshold']
        )
        if center_stable:
            self.x_axis_state = AXIS_HOLD
            self.y_axis_state = AXIS_HOLD
            return 0.0, 0.0

        # 最终调航向时两轴始终共同闭环；小误差轴的输出由自身 PD 自然减小。
        self.x_axis_state = AXIS_TRACK
        self.y_axis_state = AXIS_TRACK
        return (
            self._axis_track_command(
                'x', error_x, vehicle.forward_velocity),
            self._axis_track_command(
                'y', error_y, vehicle.lateral_velocity),
        )
        center_stable = (
            math.hypot(error_x, error_y)
            <= self.parameters['control_center_hold_tolerance']
            and math.hypot(
                vehicle.forward_velocity,
                vehicle.lateral_velocity,
            ) <= self.parameters['horizontal_speed_threshold']
        )
        if center_stable:
            self.x_axis_state = AXIS_HOLD
            self.y_axis_state = AXIS_HOLD
            return 0.0, 0.0

        # 最终调航向时两轴始终共同闭环；小误差轴的输出由自身 PD 自然减小。
        self.x_axis_state = AXIS_TRACK
        self.y_axis_state = AXIS_TRACK
        return (
            self._axis_track_command(
                'x', error_x, vehicle.forward_velocity),
            self._axis_track_command(
                'y', error_y, vehicle.lateral_velocity),
        )

    def _final_alignment_output(self, vehicle, yaw_error):
        """保持 control_link 位置并调整最终航向。"""
        kp = self._motion_parameter('kp_yaw', yaw_error)
        kr = self._motion_parameter('kr_yaw', yaw_error)
        mz = kp * yaw_error - kr * vehicle.yaw_rate
        tx, ty = self._center_hold_commands(vehicle)
        self.yaw_axis_state = AXIS_TRACK
        return self._output(
            vehicle,
            MODE_DEPTH,
            tx,
            ty,
            mz,
        )

    def _final_brake_output(self, vehicle):
        """保持 control_link 位置，同时主动消除最终航向角速度。"""
        tx, ty = self._center_hold_commands(vehicle)
        self.yaw_axis_state = AXIS_BRAKE
        return self._output(
            vehicle,
            MODE_DEPTH,
            tx,
            ty,
            self._yaw_brake_command(vehicle.yaw_rate),
            yaw_braking=True,
        )

    def _final_yaw_needs_realign(self, yaw_error, yaw_rate_abs):
        """判断刹转后是否必须恢复主动调航向，避免停在误差死区。"""
        tolerance = self.parameters['yaw_tolerance']
        return (
            abs(yaw_error) > 2.0 * tolerance
            or (
                abs(yaw_error) > tolerance
                and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
            )
        )

    def step(self, vehicle):
        """推进一个控制周期。"""
        if not isinstance(vehicle, VehicleState):
            raise TypeError('vehicle 必须是 VehicleState')

        speed = math.hypot(vehicle.forward_velocity, vehicle.lateral_velocity)
        yaw_rate_abs = abs(vehicle.yaw_rate)

        if not vehicle.feedback_fresh:
            if self.goal_active:
                self.recovery_brake_requested = True
            self._transition(SAFE, 'TF 或速度反馈超时')
            return self._output(
                vehicle, MODE_DEPTH, immediate_zero=True)

        if self.state == SAFE:
            if self.goal_active or self.cancel_requested:
                self.recovery_brake_requested = True
                self.x_axis_state = AXIS_BRAKE
                self.y_axis_state = AXIS_BRAKE
                self.yaw_axis_state = AXIS_BRAKE
                self._transition(TRANSLATE_BRAKE, '反馈恢复，先确认停稳')
            else:
                self._transition(IDLE, '反馈恢复，等待目标')

        if self.state == IDLE:
            if not self._activate_pending_goal(vehicle):
                return self._output(
                    vehicle, MODE_DEPTH, immediate_zero=True)

        if self.state in (ALIGN_PATH, ALIGN_PATH_BRAKE):
            # 兼容旧状态编号；新流程不再朝向目标点，直接保持当前航向平移。
            self.translation_yaw = vehicle.yaw
            self._transition(TRANSLATE, '取消路径预对准，保持当前航向平移')

        if self.state in (TRANSLATE, TRANSLATE_BRAKE):
            if self.recovery_brake_requested:
                stopped = (
                    speed <= self.parameters['horizontal_speed_threshold']
                    and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
                )
                if self._stable(stopped):
                    self.recovery_brake_requested = False
                    dx, dy, distance, unused_yaw_error = (
                        self._goal_metrics(vehicle))
                    del dx, dy, unused_yaw_error
                    self.translation_yaw = vehicle.yaw
                    if distance <= self.parameters['capture_radius']:
                        self.x_axis_state = AXIS_HOLD
                        self.y_axis_state = AXIS_HOLD
                        self._transition(
                            ALIGN_FINAL,
                            '反馈恢复且已停稳，继续最终航向',
                        )
                    else:
                        self.x_axis_state = AXIS_TRACK
                        self.y_axis_state = AXIS_TRACK
                        self._transition(
                            TRANSLATE,
                            '反馈恢复且已停稳，继续水平跟踪',
                        )
                return self._brake_output(vehicle)

            if self.cancel_requested:
                stopped = (
                    speed <= self.parameters['horizontal_speed_threshold']
                    and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
                )
                if self._stable(stopped):
                    self.goal = MotionGoal(
                        vehicle.x,
                        vehicle.y,
                        vehicle.z,
                        vehicle.yaw,
                    )
                    self.cancel_requested = False
                    self.x_axis_state = AXIS_HOLD
                    self.y_axis_state = AXIS_HOLD
                    self.yaw_axis_state = AXIS_HOLD
                    self._transition(CAPTURE, '取消后已停稳，悬停当前位置')
                    self.handover_started_at = vehicle.now
                    return self._output(
                        vehicle, MODE_DPROV, immediate_zero=True)
                return self._brake_output(vehicle)

            dx, dy, distance, unused_yaw_error = self._goal_metrics(vehicle)
            del unused_yaw_error
            error_x, error_y = map_error_to_body(dx, dy, vehicle.yaw)
            self.x_axis_state, tx, x_braking = self._axis_control(
                'x',
                self.x_axis_state,
                error_x,
                vehicle.forward_velocity,
            )
            self.y_axis_state, ty, y_braking = self._axis_control(
                'y',
                self.y_axis_state,
                error_y,
                vehicle.lateral_velocity,
            )

            if distance > self.parameters['capture_radius']:
                # 二维距离在捕获区外时，不能忽略已经进入 HOLD 的另一轴。
                # 两轴都保持闭环，可用较小的反馈输出抑制推进耦合造成的漂移。
                if self.x_axis_state == AXIS_HOLD:
                    self.x_axis_state = AXIS_TRACK
                    tx = self._axis_track_command(
                        'x', error_x, vehicle.forward_velocity)
                    x_braking = False
                if self.y_axis_state == AXIS_HOLD:
                    self.y_axis_state = AXIS_TRACK
                    ty = self._axis_track_command(
                        'y', error_y, vehicle.lateral_velocity)
                    y_braking = False

            if distance > self.parameters['capture_radius']:
                # 二维距离在捕获区外时，不能忽略已经进入 HOLD 的另一轴。
                # 两轴都保持闭环，可用较小的反馈输出抑制推进耦合造成的漂移。
                if self.x_axis_state == AXIS_HOLD:
                    self.x_axis_state = AXIS_TRACK
                    tx = self._axis_track_command(
                        'x', error_x, vehicle.forward_velocity)
                    x_braking = False
                if self.y_axis_state == AXIS_HOLD:
                    self.y_axis_state = AXIS_TRACK
                    ty = self._axis_track_command(
                        'y', error_y, vehicle.lateral_velocity)
                    y_braking = False

            axes_hold = (
                self.x_axis_state == AXIS_HOLD
                and self.y_axis_state == AXIS_HOLD
            )

            if axes_hold:
                self.yaw_axis_state = AXIS_HOLD
                self._transition(ALIGN_FINAL, '水平两轴停稳，调整最终航向')
                return self._output(
                    vehicle,
                    MODE_DEPTH,
                    x_braking=x_braking,
                    y_braking=y_braking,
                )

            any_tracking = (
                self.x_axis_state == AXIS_TRACK
                or self.y_axis_state == AXIS_TRACK
            )
            any_braking = (
                self.x_axis_state == AXIS_BRAKE
                or self.y_axis_state == AXIS_BRAKE
            )
            if any_tracking:
                self._transition(
                    TRANSLATE,
                    '水平轴独立跟踪，进入停车距离的轴单独刹车',
                )
            elif any_braking:
                self._transition(
                    TRANSLATE_BRAKE,
                    '水平两轴均处于刹车或保持',
                )

            path_error = wrap_angle(self.translation_yaw - vehicle.yaw)
            kp_yaw = self._motion_parameter('kp_yaw', path_error)
            kr_yaw = self._motion_parameter('kr_yaw', path_error)
            mz = (
                kp_yaw * path_error
                - kr_yaw * vehicle.yaw_rate
            )
            self.yaw_axis_state = (
                AXIS_HOLD
                if (
                    abs(path_error) <= self.parameters['yaw_tolerance']
                    and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
                )
                else AXIS_TRACK
            )
            return self._output(
                vehicle,
                MODE_DEPTH,
                tx=tx,
                ty=ty,
                mz=mz,
                x_braking=x_braking,
                y_braking=y_braking,
            )

        if self.state == ALIGN_FINAL:
            dx, dy, distance, error = self._goal_metrics(vehicle)
            del dx, dy
            if distance > self.parameters['capture_exit_radius']:
                self.translation_yaw = vehicle.yaw
                self._transition(
                    TRANSLATE,
                    '最终转向时漂出捕获区，保持当前航向重新接近',
                )
                return self._brake_output(vehicle)
            if abs(error) <= max(
                    self.parameters['yaw_tolerance'],
                    self._angular_stop_threshold(vehicle.yaw_rate)):
                self.yaw_axis_state = AXIS_BRAKE
                self._transition(FINAL_BRAKE, '最终航向进入刹转区')
                return self._final_brake_output(vehicle)
            return self._final_alignment_output(vehicle, error)

        if self.state == FINAL_BRAKE:
            dx, dy, distance, yaw_error = self._goal_metrics(vehicle)
            del dx, dy
            if distance > self.parameters['capture_exit_radius']:
                self.translation_yaw = vehicle.yaw
                self._transition(
                    TRANSLATE,
                    '最终刹转时漂出捕获区，保持当前航向重新接近',
                )
                return self._brake_output(vehicle)
            if self._final_yaw_needs_realign(yaw_error, yaw_rate_abs):
                self.yaw_axis_state = AXIS_TRACK
                self._transition(ALIGN_FINAL, '最终刹转后航向未收敛，重新调整')
                return self._final_alignment_output(vehicle, yaw_error)
            pose_stopped = (
                abs(yaw_error) <= self.parameters['yaw_tolerance']
                and speed <= self.parameters['horizontal_speed_threshold']
                and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
            )
            if (
                    distance > self.parameters['capture_radius']
                    and pose_stopped):
                self.translation_yaw = vehicle.yaw
                self._transition(
                    TRANSLATE,
                    '最终刹停点仍在捕获区外，保持当前航向重新接近',
                )
                return self._brake_output(vehicle)
            if self._stable(
                    distance <= self.parameters['capture_radius']
                    and pose_stopped):
                self.x_axis_state = AXIS_HOLD
                self.y_axis_state = AXIS_HOLD
                self.yaw_axis_state = AXIS_HOLD
                self._transition(CAPTURE, '目标位姿刹停稳定，等待下位机定点接管')
                self.handover_started_at = vehicle.now
                return self._output(
                    vehicle, MODE_DPROV, immediate_zero=True)
            return self._final_brake_output(vehicle)

        if self.state == CAPTURE:
            self.x_axis_state = AXIS_HOLD
            self.y_axis_state = AXIS_HOLD
            self.yaw_axis_state = AXIS_HOLD
            dx, dy, distance, yaw_error = self._goal_metrics(vehicle)
            del dx, dy
            if distance > self.parameters['capture_exit_radius']:
                self.translation_yaw = vehicle.yaw
                self._transition(
                    TRANSLATE,
                    '捕获期间漂出位置范围，保持当前航向重新接近',
                )
                return self._brake_output(vehicle)
            if self._final_yaw_needs_realign(yaw_error, yaw_rate_abs):
                self.yaw_axis_state = AXIS_TRACK
                self._transition(ALIGN_FINAL, '捕获期间航向未收敛，重新调整')
                return self._final_alignment_output(vehicle, yaw_error)
            captured = (
                distance <= self.parameters['capture_radius']
                and abs(yaw_error) <= self.parameters['yaw_tolerance']
                and speed <= self.parameters['horizontal_speed_threshold']
                and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
            )
            if self.handover_started_at is not None:
                if not captured:
                    self.handover_started_at = None
                    self.stable_count = 0
                    self.reason = '定点接管等待期间离开捕获条件，重新刹停确认'
                    return self._final_brake_output(vehicle)
                mode_acknowledged = (
                    vehicle.reported_mode == MODE_DPROV
                    and vehicle.reported_mode_stamp is not None
                    and vehicle.reported_mode_stamp
                    >= self.handover_started_at
                )
                if mode_acknowledged:
                    self._transition(
                        HOVER,
                        '下位机定点接管已确认，目标到达',
                    )
                    return self._output(
                        vehicle, MODE_DPROV, immediate_zero=True)
                if (
                        vehicle.now - self.handover_started_at
                        > self.parameters['mode_ack_timeout']):
                    self._transition(SAFE, '定点模式确认超时')
                    return self._output(
                        vehicle, MODE_DEPTH, immediate_zero=True)
                return self._output(
                    vehicle, MODE_DPROV, immediate_zero=True)
            if self._stable(captured):
                self.handover_started_at = vehicle.now
                self.reason = '捕获稳定，等待下位机定点接管'
                return self._output(
                    vehicle, MODE_DPROV, immediate_zero=True)
            return self._final_brake_output(vehicle)

        if self.state == HOVER:
            self.x_axis_state = AXIS_HOLD
            self.y_axis_state = AXIS_HOLD
            self.yaw_axis_state = AXIS_HOLD
            unused_dx, unused_dy, hover_distance, hover_yaw_error = (
                self._goal_metrics(vehicle))
            del unused_dx, unused_dy
            if self.hover_direct_update:
                self.hover_direct_update = False
                if hover_distance > self.parameters['capture_radius']:
                    self.translation_yaw = vehicle.yaw
                    self.x_axis_state = AXIS_TRACK
                    self.y_axis_state = AXIS_TRACK
                    self._transition(
                        TRANSLATE,
                        '定点期间收到近距离目标，直接平移跟踪',
                    )
                    return self._output(
                        vehicle, MODE_DEPTH, immediate_zero=True)
                if abs(hover_yaw_error) > self.parameters['yaw_tolerance']:
                    self.yaw_axis_state = AXIS_TRACK
                    self._transition(
                        ALIGN_FINAL,
                        '定点期间收到近航向目标，直接调整航向',
                    )
                    return self._final_alignment_output(
                        vehicle, hover_yaw_error)
            if speed > self.parameters['hover_fault_speed']:
                self._transition(TRANSLATE_BRAKE, '定点接管后水平速度异常')
                return self._brake_output(vehicle)
            if hover_distance > self.parameters['hover_fault_position_error']:
                self._transition(TRANSLATE_BRAKE, '定点接管后位置误差超限')
                return self._brake_output(vehicle)
            if yaw_rate_abs > self.parameters['hover_fault_yaw_rate']:
                self._transition(TRANSLATE_BRAKE, '定点接管后航向角速度异常')
                return self._brake_output(vehicle)
            if abs(hover_yaw_error) > self.parameters['hover_fault_yaw_error']:
                self._transition(TRANSLATE_BRAKE, '定点接管后航向误差异常')
                return self._brake_output(vehicle)
            mode_feedback_timed_out = (
                vehicle.reported_mode_stamp is None
                or vehicle.now - vehicle.reported_mode_stamp
                > self.parameters['mode_ack_timeout']
            )
            if mode_feedback_timed_out:
                self._transition(SAFE, '定点模式反馈超时')
                return self._output(
                    vehicle, MODE_DEPTH, immediate_zero=True)
            if vehicle.reported_mode != MODE_DPROV:
                self._transition(SAFE, '定点模式反馈丢失')
                return self._output(
                    vehicle, MODE_DEPTH, immediate_zero=True)
            return self._output(
                vehicle, MODE_DPROV, immediate_zero=True)

        self._transition(SAFE, '未知状态')
        return self._output(vehicle, MODE_DEPTH, immediate_zero=True)
