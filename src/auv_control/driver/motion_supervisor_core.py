#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称：motion_supervisor_core.py
功能：运动—刹停—悬停控制器的纯算法核心
作者：BroXu
说明：
    1. 实现统一三轴位姿跟踪、主动刹停、定点接管和悬停状态机；
    2. 提供坐标转换、停车距离、力限幅和参考轨迹限制等纯函数；
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
    最终转向和刹转阶段增加控制中心平面位置保持，固定实际旋转中心。
2026.7.20
    删除统一三轴控制后不可达的旧分阶段状态机，并补齐参考状态复位和硬限幅。
"""

from __future__ import division

import math


MODE_DEPTH = 2
MODE_DPROV = 4
PROTOCOL_FORCE_LIMIT = 10000

IDLE = 0
# 1、2、5、6 仅保留与 MotionState 消息的编号兼容，当前流程不再产生。
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


def map_vector_to_body(north, east, yaw):
    """将 map/NED 水平向量旋转到 base_link。"""
    return map_error_to_body(north, east, yaw)


def vector_limit(x, y, maximum):
    """按二维向量模长限幅，保持方向不变。"""
    magnitude = math.hypot(x, y)
    if maximum <= 0.0 or magnitude <= maximum:
        return x, y
    scale = maximum / magnitude
    return x * scale, y * scale


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


def goals_equal(first, second, tolerance=1e-6):
    """判断两个位姿目标在给定容差内是否相同。"""
    if first is None or second is None:
        return first is second
    if not isinstance(first, MotionGoal) or not isinstance(second, MotionGoal):
        raise TypeError('目标必须是 MotionGoal')
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('目标比较容差必须是非负有限值')
    return (
        abs(first.x - second.x) <= tolerance
        and abs(first.y - second.y) <= tolerance
        and abs(first.z - second.z) <= tolerance
        and abs(wrap_angle(first.yaw - second.yaw)) <= tolerance
    )


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
            x_speed=0.0, y_speed=0.0, diagnostics=None):
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
        self.diagnostics = diagnostics or {}


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
    'kv_x_positive': 1000.0,
    'kv_x_negative': 1000.0,
    'kv_y_positive': 2000.0,
    'kv_y_negative': 2000.0,
    'kp_yaw_positive': 6000.0,
    'kp_yaw_negative': 6000.0,
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
    'horizontal_speed_threshold': 0.015,
    'yaw_tolerance': math.radians(5.0),
    'yaw_rate_threshold': math.radians(0.3),
    'stable_frames': 5,
    'hover_fault_position_error': 0.40,
    'hover_fault_speed': 0.15,
    'hover_fault_yaw_rate': math.radians(5.0),
    'hover_fault_yaw_error': math.radians(20.0),
    'mode_ack_timeout': 1.0,
    # 统一三轴位置控制：位置环生成速度参考，速度环生成力/力矩。
    'xy_max_speed': 0.30,
    'xy_position_gain': 1.00,
    'xy_max_acceleration': 0.20,
    'xy_max_jerk': 0.40,
    # 预测到停车距离后进入刹停；速度降至退出阈值后才恢复位置跟踪。
    'xy_brake_enter_speed': 0.020,
    'xy_brake_exit_speed': 0.012,
    'yaw_max_rate': math.radians(25.0),
    'yaw_position_gain': 1.50,
    'yaw_max_acceleration': math.radians(20.0),
    'yaw_max_jerk': math.radians(60.0),
    # 航向刹停使用独立迟滞，避免 r 在停稳阈值附近抖动。
    'yaw_brake_enter_rate': math.radians(0.6),
    'yaw_brake_exit_rate': math.radians(0.2),
    # 原始 /status/vel.r 到 map 航向角速度的符号转换。
    # 当前下位机约定：MZ 与 r 方向相反，故 map 航向角速度 = -r。
    'yaw_rate_to_map_sign': -1.0,
    'goal_static_capture_seconds': 0.80,
    'goal_replan_position_threshold': 0.10,
    'goal_replan_yaw_threshold': math.radians(5.0),
    'control_dt_min': 0.02,
    'control_dt_max': 0.20,
    # 推进器有效性矩阵。默认单位阵，标定后填写交叉项以补偿 TX/TY/MZ 耦合。
    'effectiveness_x_tx': 1.0,
    'effectiveness_x_ty': 0.0,
    'effectiveness_x_mz': 0.0,
    'effectiveness_y_tx': 0.0,
    'effectiveness_y_ty': 1.0,
    'effectiveness_y_mz': 0.0,
    'effectiveness_yaw_tx': 0.0,
    'effectiveness_yaw_ty': 0.0,
    'effectiveness_yaw_mz': 1.0,
    'effectiveness_min_determinant': 0.05,
}


class MotionSupervisorCore(object):
    """统一完成三轴位姿跟踪、刹停和定点接管。"""

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
        self.recovery_brake_requested = False
        self.x_axis_state = AXIS_HOLD
        self.y_axis_state = AXIS_HOLD
        self.yaw_axis_state = AXIS_HOLD
        self.stable_count = 0
        self.handover_started_at = None
        self.last_tx = 0.0
        self.last_ty = 0.0
        self.last_mz = 0.0
        self.last_control_time = None
        self.last_velocity_reference = (0.0, 0.0)
        self.last_velocity_acceleration = (0.0, 0.0)
        self.last_yaw_rate_reference = 0.0
        self.last_yaw_acceleration = 0.0
        self.xy_brake_latched = False
        self.yaw_brake_latched = False
        self.reference_reset_pending = True
        self.goal_changed_pending = False
        self.goal_changed_at = None

    def _validate_parameters(self):
        positive_names = (
            'max_tx_positive', 'max_tx_negative',
            'max_ty_positive', 'max_ty_negative',
            'max_mz_positive', 'max_mz_negative',
            'brake_max_tx_positive', 'brake_max_tx_negative',
            'brake_max_ty_positive', 'brake_max_ty_negative',
            'brake_max_mz_positive', 'brake_max_mz_negative',
            'force_slew_per_cycle', 'brake_force_slew_per_cycle',
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
            'horizontal_speed_threshold',
            'yaw_tolerance', 'yaw_rate_threshold',
            'stable_frames', 'hover_fault_position_error',
            'hover_fault_speed', 'hover_fault_yaw_rate',
            'hover_fault_yaw_error', 'mode_ack_timeout',
            'brake_margin_tx_positive', 'brake_margin_tx_negative',
            'brake_margin_ty_positive', 'brake_margin_ty_negative',
            'yaw_brake_margin_positive', 'yaw_brake_margin_negative',
            'xy_max_speed', 'xy_position_gain', 'xy_max_acceleration',
            'xy_max_jerk', 'xy_brake_enter_speed', 'xy_brake_exit_speed',
            'yaw_max_rate', 'yaw_position_gain',
            'yaw_max_acceleration', 'yaw_max_jerk',
            'yaw_brake_enter_rate', 'yaw_brake_exit_rate',
            'goal_static_capture_seconds',
            'goal_replan_position_threshold', 'goal_replan_yaw_threshold',
            'control_dt_min',
            'control_dt_max', 'effectiveness_min_determinant',
        )
        for name in positive_names:
            if self.parameters[name] <= 0:
                raise ValueError('{} 必须大于 0'.format(name))
        for name, value in self.parameters.items():
            if not math.isfinite(float(value)):
                raise ValueError('{} 必须为有限值'.format(name))
        if abs(abs(self.parameters['yaw_rate_to_map_sign']) - 1.0) > 1e-9:
            raise ValueError('yaw_rate_to_map_sign 必须为 +1 或 -1')
        if (self.parameters['xy_brake_exit_speed'] >=
                self.parameters['xy_brake_enter_speed']):
            raise ValueError('xy_brake_exit_speed 必须小于 xy_brake_enter_speed')
        if (self.parameters['yaw_brake_exit_rate'] >=
                self.parameters['yaw_brake_enter_rate']):
            raise ValueError('yaw_brake_exit_rate 必须小于 yaw_brake_enter_rate')
        for name in (
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
        if self.parameters['control_dt_min'] > self.parameters['control_dt_max']:
            raise ValueError('control_dt_min 不能大于 control_dt_max')
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

    def _effectiveness_matrix(self):
        """返回推进器命令到本体控制效果的三自由度矩阵。"""
        p = self.parameters
        return (
            (p['effectiveness_x_tx'], p['effectiveness_x_ty'], p['effectiveness_x_mz']),
            (p['effectiveness_y_tx'], p['effectiveness_y_ty'], p['effectiveness_y_mz']),
            (p['effectiveness_yaw_tx'], p['effectiveness_yaw_ty'], p['effectiveness_yaw_mz']),
        )

    def _effectiveness_determinant(self):
        """计算三阶有效性矩阵行列式。"""
        matrix = self._effectiveness_matrix()
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    @property
    def goal_active(self):
        return self.goal is not None or self.pending_goal is not None

    def _goal_requires_replan(self, previous, current):
        """判断是否需要为新目标重置轨迹参考。"""
        if previous is None:
            return True
        position_delta = math.sqrt(
            (current.x - previous.x) ** 2
            + (current.y - previous.y) ** 2
            + (current.z - previous.z) ** 2
        )
        yaw_delta = abs(wrap_angle(current.yaw - previous.yaw))
        return (
            position_delta >= self.parameters['goal_replan_position_threshold']
            or yaw_delta >= self.parameters['goal_replan_yaw_threshold']
        )

    def set_goal(self, goal):
        """采用最新目标；只有超过重规划阈值才重置捕获计时和轨迹参考。"""
        if not isinstance(goal, MotionGoal):
            raise TypeError('goal 必须是 MotionGoal')

        if self.goal is None:
            changed = self._goal_requires_replan(self.pending_goal, goal)
            self.pending_goal = goal
            if changed:
                self.goal_changed_pending = True
                self._reset_motion_references()
                self.reason = '收到首个目标'
            return changed

        changed = self._goal_requires_replan(self.goal, goal)
        self.goal = goal
        self.pending_goal = None
        self.cancel_requested = False
        if self.state == CAPTURE and changed:
            self.handover_started_at = None
            self.stable_count = 0
        if changed:
            self.goal_changed_pending = True
            self._reset_motion_references()
            self.reason = '采用最新目标，恢复统一三轴跟踪'
        return changed

    def cancel(self):
        """取消当前运动；停稳后以当前位置进入悬停。"""
        if not self.goal_active and self.state == IDLE:
            return
        self.pending_goal = None
        self.cancel_requested = True
        self.recovery_brake_requested = False
        self._reset_motion_references()
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
        self.x_axis_state = AXIS_TRACK
        self.y_axis_state = AXIS_TRACK
        self.yaw_axis_state = AXIS_TRACK
        self._transition(TRANSLATE, '开始统一三轴位姿跟踪')
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

    def _yaw_brake_command(self, yaw_rate):
        """角速度超过停稳阈值时，保证刹转力矩越过实测有效下限。"""
        command = self._brake_command('mz', yaw_rate)
        if (
                abs(yaw_rate) > self.parameters['yaw_rate_threshold']
                and abs(command) < self.parameters['brake_min_mz']):
            command = math.copysign(
                self.parameters['brake_min_mz'], command)
        return command

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

    def _reset_motion_references(self, now=None):
        """清空跨目标、SAFE 和定点阶段不得沿用的轨迹参考状态。"""
        self.last_control_time = now
        self.last_velocity_reference = (0.0, 0.0)
        self.last_velocity_acceleration = (0.0, 0.0)
        self.last_yaw_rate_reference = 0.0
        self.last_yaw_acceleration = 0.0
        self.xy_brake_latched = False
        self.yaw_brake_latched = False
        self.reference_reset_pending = True

    def _initialize_motion_references(self, vehicle, map_vx, map_vy):
        """从当前实测速度无扰初始化新动作参考，并立即施加配置硬限幅。"""
        reference_vx, reference_vy = vector_limit(
            map_vx,
            map_vy,
            self.parameters['xy_max_speed'],
        )
        self.last_velocity_reference = (reference_vx, reference_vy)
        self.last_velocity_acceleration = (0.0, 0.0)
        self.last_yaw_rate_reference = clamp(
            self.parameters['yaw_rate_to_map_sign'] * vehicle.yaw_rate,
            -self.parameters['yaw_max_rate'],
            self.parameters['yaw_max_rate'],
        )
        self.last_yaw_acceleration = 0.0
        self.last_control_time = vehicle.now
        self.reference_reset_pending = False

    def _control_dt(self, now):
        """返回受限控制周期，避免调度抖动造成参考突跳。"""
        if self.last_control_time is None:
            dt = self.parameters['control_dt_max']
        else:
            dt = now - self.last_control_time
        self.last_control_time = now
        return clamp(
            dt,
            self.parameters['control_dt_min'],
            self.parameters['control_dt_max'],
        )

    def _directional_xy_brake_model(self, direction_body_x, direction_body_y):
        """沿目标方向估计可达二维减速度和停止余量。"""
        tx_brake = -direction_body_x
        ty_brake = -direction_body_y
        ax = self._directional_parameter('brake_acceleration_tx', tx_brake)
        ay = self._directional_parameter('brake_acceleration_ty', ty_brake)
        mx = self._directional_parameter('brake_margin_tx', tx_brake)
        my = self._directional_parameter('brake_margin_ty', ty_brake)
        denominator = math.sqrt(
            (direction_body_x / max(ax, 1e-6)) ** 2
            + (direction_body_y / max(ay, 1e-6)) ** 2
        )
        acceleration = 1.0 / max(denominator, 1e-6)
        margin = math.hypot(direction_body_x * mx, direction_body_y * my)
        return acceleration, margin

    def _slew_velocity_reference(self, desired_x, desired_y, dt):
        """以加速度和 jerk 双重约束平滑二维速度参考。"""
        last_x, last_y = self.last_velocity_reference
        desired_ax = (desired_x - last_x) / dt
        desired_ay = (desired_y - last_y) / dt
        desired_ax, desired_ay = vector_limit(
            desired_ax,
            desired_ay,
            self.parameters['xy_max_acceleration'],
        )
        last_ax, last_ay = self.last_velocity_acceleration
        delta_ax = desired_ax - last_ax
        delta_ay = desired_ay - last_ay
        delta_ax, delta_ay = vector_limit(
            delta_ax,
            delta_ay,
            self.parameters['xy_max_jerk'] * dt,
        )
        acceleration_x = last_ax + delta_ax
        acceleration_y = last_ay + delta_ay
        reference_x = last_x + acceleration_x * dt
        reference_y = last_y + acceleration_y * dt
        reference_x, reference_y = vector_limit(
            reference_x,
            reference_y,
            self.parameters['xy_max_speed'],
        )
        # 限速后保存实际产生的加速度，避免内部状态继续向限幅外累积。
        acceleration_x = (reference_x - last_x) / dt
        acceleration_y = (reference_y - last_y) / dt
        self.last_velocity_reference = (reference_x, reference_y)
        self.last_velocity_acceleration = (acceleration_x, acceleration_y)
        return reference_x, reference_y

    def _slew_yaw_rate_reference(self, desired_rate, dt):
        """以角加速度和角 jerk 约束平滑角速度参考。"""
        desired_acceleration = (
            desired_rate - self.last_yaw_rate_reference
        ) / dt
        desired_acceleration = clamp(
            desired_acceleration,
            -self.parameters['yaw_max_acceleration'],
            self.parameters['yaw_max_acceleration'],
        )
        delta = clamp(
            desired_acceleration - self.last_yaw_acceleration,
            -self.parameters['yaw_max_jerk'] * dt,
            self.parameters['yaw_max_jerk'] * dt,
        )
        acceleration = self.last_yaw_acceleration + delta
        reference = self.last_yaw_rate_reference + acceleration * dt
        reference = clamp(
            reference,
            -self.parameters['yaw_max_rate'],
            self.parameters['yaw_max_rate'],
        )
        self.last_yaw_acceleration = (
            reference - self.last_yaw_rate_reference
        ) / dt
        self.last_yaw_rate_reference = reference
        return reference

    def _compensate_effectiveness(self, tx, ty, mz):
        """使用三阶有效性矩阵逆补偿推进器交叉耦合。"""
        matrix = self._effectiveness_matrix()
        a, b, c = matrix[0]
        d, e, f = matrix[1]
        g, h, i = matrix[2]
        determinant = self._effectiveness_determinant()
        if abs(determinant) < self.parameters['effectiveness_min_determinant']:
            return tx, ty, mz
        inverse = (
            ((e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant),
            ((f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant),
            ((d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant),
        )
        return (
            inverse[0][0] * tx + inverse[0][1] * ty + inverse[0][2] * mz,
            inverse[1][0] * tx + inverse[1][1] * ty + inverse[1][2] * mz,
            inverse[2][0] * tx + inverse[2][1] * ty + inverse[2][2] * mz,
        )

    def _limit_xy_force_vector(self, tx, ty, braking):
        """按可用正负推力形成椭圆限幅，保持二维合力方向。"""
        limit_prefix = 'brake_max_' if braking else 'max_'
        tx_limit = self.parameters[
            limit_prefix + ('tx_positive' if tx >= 0.0 else 'tx_negative')]
        ty_limit = self.parameters[
            limit_prefix + ('ty_positive' if ty >= 0.0 else 'ty_negative')]
        normalized = math.sqrt(
            (tx / max(tx_limit, 1e-6)) ** 2
            + (ty / max(ty_limit, 1e-6)) ** 2
        )
        if normalized <= 1.0:
            return tx, ty
        return tx / normalized, ty / normalized

    def _output(self, vehicle, mode, tx=0.0, ty=0.0, mz=0.0,
                x_braking=False, y_braking=False, yaw_braking=False,
                immediate_zero=False, diagnostics=None):
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
        raw_tx, raw_ty, raw_mz = tx, ty, mz
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
        diagnostics = dict(diagnostics or {})
        diagnostics.update({
            'raw_tx': raw_tx,
            'raw_ty': raw_ty,
            'raw_mz': raw_mz,
            'limited_tx': tx,
            'limited_ty': ty,
            'limited_mz': mz,
        })
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
            diagnostics=diagnostics,
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

    def _goal_static_seconds(self, now):
        """返回最新几何目标保持不变的时间。"""
        if self.goal_changed_at is None:
            return 0.0
        return max(0.0, now - self.goal_changed_at)

    def _unified_pose_output(self, vehicle):
        """同时生成 XY 与 yaw 的受限速度参考和推进器命令。"""
        if self.goal_changed_pending:
            self.goal_changed_at = vehicle.now
            self.goal_changed_pending = False
        if self.goal_changed_at is None:
            self.goal_changed_at = vehicle.now

        dx, dy, distance, yaw_error = self._goal_metrics(vehicle)
        map_vx, map_vy = body_velocity_to_map(
            vehicle.forward_velocity,
            vehicle.lateral_velocity,
            vehicle.yaw,
        )
        if self.reference_reset_pending:
            self._initialize_motion_references(vehicle, map_vx, map_vy)
        dt = self._control_dt(vehicle.now)
        if distance > 1e-6:
            direction_x = dx / distance
            direction_y = dy / distance
        else:
            direction_x = 0.0
            direction_y = 0.0
        direction_body_x, direction_body_y = map_vector_to_body(
            direction_x, direction_y, vehicle.yaw)
        closing_speed = max(0.0, map_vx * direction_x + map_vy * direction_y)
        brake_acceleration, brake_margin = self._directional_xy_brake_model(
            direction_body_x, direction_body_y)
        stop_distance = stopping_distance(
            closing_speed,
            brake_acceleration,
            self.parameters['control_delay'],
            brake_margin,
        )
        xy_brake_entry = (
            closing_speed >= self.parameters['xy_brake_enter_speed']
            and distance <= stop_distance)
        if xy_brake_entry:
            self.xy_brake_latched = True
        elif (self.xy_brake_latched and
              math.hypot(map_vx, map_vy) <=
              self.parameters['xy_brake_exit_speed']):
            self.xy_brake_latched = False
        braking_xy = self.xy_brake_latched
        position_speed = self.parameters['xy_position_gain'] * distance
        stopping_speed = math.sqrt(
            2.0 * brake_acceleration * max(distance - brake_margin, 0.0)
        )
        close_position_speed = self.parameters['xy_position_gain'] * min(
            distance, brake_margin)
        desired_speed = min(
            self.parameters['xy_max_speed'],
            position_speed,
            max(stopping_speed, close_position_speed),
        )
        desired_vx = desired_speed * direction_x
        desired_vy = desired_speed * direction_y
        if braking_xy:
            # 锁存刹停期间始终把速度参考收敛到零，避免越过目标后立即恢复加速。
            desired_vx = 0.0
            desired_vy = 0.0
        reference_vx, reference_vy = self._slew_velocity_reference(
            desired_vx, desired_vy, dt)
        reference_body_x, reference_body_y = map_vector_to_body(
            reference_vx, reference_vy, vehicle.yaw)
        velocity_error_x = reference_body_x - vehicle.forward_velocity
        velocity_error_y = reference_body_y - vehicle.lateral_velocity
        tx = self._motion_parameter('kv_x', velocity_error_x) * velocity_error_x
        ty = self._motion_parameter('kv_y', velocity_error_y) * velocity_error_y

        # 位置误差和期望角速度属于 map 航向约定；原始 r 与其方向相反。
        map_yaw_rate = (
            self.parameters['yaw_rate_to_map_sign'] * vehicle.yaw_rate)
        # 刹车 MZ 与 map 航向角速度反向，按最终刹车 MZ 的符号选择模型。
        yaw_brake_mz_direction = -map_yaw_rate
        # 位置误差和期望角速度属于 map 航向约定；原始 r 与其方向相反。
        map_yaw_rate = (
            self.parameters['yaw_rate_to_map_sign'] * vehicle.yaw_rate)
        # 刹车 MZ 与 map 航向角速度反向，按最终刹车 MZ 的符号选择模型。
        yaw_brake_mz_direction = -map_yaw_rate
        yaw_brake_acceleration = self._directional_parameter(
            'angular_brake_acceleration_mz', yaw_brake_mz_direction)
        yaw_margin = self._directional_parameter(
            'yaw_brake_margin', yaw_brake_mz_direction)
        yaw_stop_angle = stopping_distance(
            abs(map_yaw_rate),
            yaw_brake_acceleration,
            self.parameters['control_delay'],
            yaw_margin,
        )
        yaw_brake_entry = (
            abs(map_yaw_rate) >= self.parameters['yaw_brake_enter_rate']
            and abs(yaw_error) <= yaw_stop_angle)
        if yaw_brake_entry:
            self.yaw_brake_latched = True
        elif (self.yaw_brake_latched and
              abs(map_yaw_rate) <= self.parameters['yaw_brake_exit_rate']):
            self.yaw_brake_latched = False
        braking_yaw = self.yaw_brake_latched
        yaw_stopping_rate = math.sqrt(
            2.0 * yaw_brake_acceleration * max(
                abs(yaw_error) - yaw_margin, 0.0)
        )
        yaw_close_rate = self.parameters['yaw_position_gain'] * min(
            abs(yaw_error), yaw_margin)
        desired_yaw_rate = math.copysign(
            min(
                self.parameters['yaw_max_rate'],
                self.parameters['yaw_position_gain'] * abs(yaw_error),
                max(yaw_stopping_rate, yaw_close_rate),
            ),
            yaw_error,
        ) if abs(yaw_error) > 1e-9 else 0.0
        if braking_yaw:
            # 锁存刹转期间将 map 航向角速度参考置零，持续施加反向阻尼。
            desired_yaw_rate = 0.0
        reference_yaw_rate = self._slew_yaw_rate_reference(
            desired_yaw_rate, dt)
        yaw_rate_error = reference_yaw_rate - map_yaw_rate
        mz = self._motion_parameter('kp_yaw', yaw_rate_error) * yaw_rate_error

        tx, ty, mz = self._compensate_effectiveness(tx, ty, mz)
        tx, ty = self._limit_xy_force_vector(tx, ty, braking_xy)
        if distance <= self.parameters['control_center_hold_tolerance'] and (
                math.hypot(map_vx, map_vy)
                <= self.parameters['horizontal_speed_threshold']):
            self.x_axis_state = AXIS_HOLD
            self.y_axis_state = AXIS_HOLD
        else:
            self.x_axis_state = AXIS_BRAKE if braking_xy else AXIS_TRACK
            self.y_axis_state = self.x_axis_state
        self.yaw_axis_state = (
            AXIS_HOLD if (
                abs(yaw_error) <= self.parameters['yaw_tolerance']
                and abs(vehicle.yaw_rate) <= self.parameters['yaw_rate_threshold']
            ) else (AXIS_BRAKE if braking_yaw else AXIS_TRACK)
        )
        self._transition(
            TRANSLATE_BRAKE if (braking_xy or braking_yaw) else TRANSLATE,
            '统一三轴制动跟踪' if (braking_xy or braking_yaw) else '统一三轴位姿跟踪',
        )
        return self._output(
            vehicle,
            MODE_DEPTH,
            tx=tx,
            ty=ty,
            mz=mz,
            x_braking=braking_xy,
            y_braking=braking_xy,
            yaw_braking=braking_yaw,
            diagnostics={
                'map_velocity_x': map_vx,
                'map_velocity_y': map_vy,
                'reference_velocity_x': reference_vx,
                'reference_velocity_y': reference_vy,
                'reference_speed': math.hypot(reference_vx, reference_vy),
                'closing_speed': closing_speed,
                'xy_stop_distance': stop_distance,
                'xy_brake_acceleration': brake_acceleration,
                'xy_brake_margin': brake_margin,
                'xy_brake_entry': xy_brake_entry,
                'xy_brake_latched': self.xy_brake_latched,
                'xy_braking': braking_xy,
                'yaw_rate_reference': reference_yaw_rate,
                'map_yaw_rate': map_yaw_rate,
                'map_yaw_rate': map_yaw_rate,
                'yaw_stop_angle': yaw_stop_angle,
                'yaw_brake_entry': yaw_brake_entry,
                'yaw_brake_latched': self.yaw_brake_latched,
                'yaw_braking': braking_yaw,
                'goal_static_seconds': self._goal_static_seconds(vehicle.now),
                'goal_static_for_capture': (
                    self._goal_static_seconds(vehicle.now)
                    >= self.parameters['goal_static_capture_seconds']
                ),
            },
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
            self._reset_motion_references(vehicle.now)
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

        # 正常目标始终采用统一三轴轨迹，不再按 X/Y/yaw 分阶段控制。
        if self.state == IDLE:
            if not self._activate_pending_goal(vehicle):
                return self._output(
                    vehicle, MODE_DEPTH, immediate_zero=True)

        if self.state in (HOVER, CAPTURE) and self.goal_changed_pending:
            self.handover_started_at = None
            self.stable_count = 0
            self._transition(TRANSLATE, '目标更新，恢复统一三轴跟踪')

        if self.state not in (HOVER, CAPTURE):
            stopped = (
                speed <= self.parameters['horizontal_speed_threshold']
                and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
            )
            if self.recovery_brake_requested:
                if self._stable(stopped):
                    self.recovery_brake_requested = False
                    self._reset_motion_references(vehicle.now)
                    self._transition(TRANSLATE, '反馈恢复且已停稳，恢复统一三轴跟踪')
                else:
                    return self._brake_output(vehicle)
            if self.cancel_requested:
                if self._stable(stopped):
                    self.goal = MotionGoal(
                        vehicle.x, vehicle.y, vehicle.z, vehicle.yaw)
                    self.goal_changed_at = vehicle.now
                    self.goal_changed_pending = False
                    self.cancel_requested = False
                    self._reset_motion_references(vehicle.now)
                    self.x_axis_state = AXIS_HOLD
                    self.y_axis_state = AXIS_HOLD
                    self.yaw_axis_state = AXIS_HOLD
                    self._transition(CAPTURE, '取消后已停稳，等待下位机定点接管')
                    self.handover_started_at = vehicle.now
                    return self._output(
                        vehicle, MODE_DPROV, immediate_zero=True)
                return self._brake_output(vehicle)

            output = self._unified_pose_output(vehicle)
            goal_static = (
                output.diagnostics.get('goal_static_seconds', 0.0)
                >= self.parameters['goal_static_capture_seconds']
            )
            captured = (
                output.position_error <= self.parameters['capture_radius']
                and abs(output.yaw_error) <= self.parameters['yaw_tolerance']
                and output.horizontal_speed <= self.parameters['horizontal_speed_threshold']
                and abs(output.yaw_rate) <= self.parameters['yaw_rate_threshold']
                and goal_static
            )
            if self._stable(captured):
                self.x_axis_state = AXIS_HOLD
                self.y_axis_state = AXIS_HOLD
                self.yaw_axis_state = AXIS_HOLD
                self._reset_motion_references(vehicle.now)
                self._transition(CAPTURE, '目标位姿同步刹停稳定，等待下位机定点接管')
                self.handover_started_at = vehicle.now
                return self._output(
                    vehicle, MODE_DPROV, immediate_zero=True,
                    diagnostics=output.diagnostics)
            return output

        if self.state == CAPTURE:
            self.x_axis_state = AXIS_HOLD
            self.y_axis_state = AXIS_HOLD
            self.yaw_axis_state = AXIS_HOLD
            dx, dy, distance, yaw_error = self._goal_metrics(vehicle)
            del dx, dy
            capture_entry_satisfied = (
                distance <= self.parameters['capture_radius']
                and abs(yaw_error) <= self.parameters['yaw_tolerance']
                and speed <= self.parameters['horizontal_speed_threshold']
                and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
            )
            # mode=4 已发出后使用更大的退出半径，避免测量噪声在捕获边界
            # 反复触发 mode=2/mode=4 切换；其余停稳条件仍必须持续满足。
            capture_retained = (
                distance <= self.parameters['capture_exit_radius']
                and abs(yaw_error) <= self.parameters['yaw_tolerance']
                and speed <= self.parameters['horizontal_speed_threshold']
                and yaw_rate_abs <= self.parameters['yaw_rate_threshold']
            )
            if self.handover_started_at is not None:
                if not capture_retained:
                    self.handover_started_at = None
                    self.stable_count = 0
                    self._transition(
                        TRANSLATE,
                        '定点接管等待期间离开捕获条件，恢复统一三轴跟踪',
                    )
                    return self._unified_pose_output(vehicle)
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
                    self._reset_motion_references(vehicle.now)
                    self._transition(SAFE, '定点模式确认超时')
                    return self._output(
                        vehicle, MODE_DEPTH, immediate_zero=True)
                return self._output(
                    vehicle, MODE_DPROV, immediate_zero=True)
            if self._stable(capture_entry_satisfied):
                self._reset_motion_references(vehicle.now)
                self.handover_started_at = vehicle.now
                self.reason = '捕获稳定，等待下位机定点接管'
                return self._output(
                    vehicle, MODE_DPROV, immediate_zero=True)
            if not capture_entry_satisfied:
                self._transition(
                    TRANSLATE,
                    '捕获条件未保持，恢复统一三轴跟踪',
                )
                return self._unified_pose_output(vehicle)
            return self._output(
                vehicle, MODE_DEPTH, immediate_zero=True)

        if self.state == HOVER:
            self.x_axis_state = AXIS_HOLD
            self.y_axis_state = AXIS_HOLD
            self.yaw_axis_state = AXIS_HOLD
            unused_dx, unused_dy, hover_distance, hover_yaw_error = (
                self._goal_metrics(vehicle))
            del unused_dx, unused_dy
            if speed > self.parameters['hover_fault_speed']:
                self.recovery_brake_requested = True
                self._reset_motion_references(vehicle.now)
                self._transition(TRANSLATE_BRAKE, '定点接管后水平速度异常')
                return self._brake_output(vehicle)
            if hover_distance > self.parameters['hover_fault_position_error']:
                self.recovery_brake_requested = True
                self._reset_motion_references(vehicle.now)
                self._transition(TRANSLATE_BRAKE, '定点接管后位置误差超限')
                return self._brake_output(vehicle)
            if yaw_rate_abs > self.parameters['hover_fault_yaw_rate']:
                self.recovery_brake_requested = True
                self._reset_motion_references(vehicle.now)
                self._transition(TRANSLATE_BRAKE, '定点接管后航向角速度异常')
                return self._brake_output(vehicle)
            if abs(hover_yaw_error) > self.parameters['hover_fault_yaw_error']:
                self.recovery_brake_requested = True
                self._reset_motion_references(vehicle.now)
                self._transition(TRANSLATE_BRAKE, '定点接管后航向误差异常')
                return self._brake_output(vehicle)
            mode_feedback_timed_out = (
                vehicle.reported_mode_stamp is None
                or vehicle.now - vehicle.reported_mode_stamp
                > self.parameters['mode_ack_timeout']
            )
            if mode_feedback_timed_out:
                self._reset_motion_references(vehicle.now)
                self._transition(SAFE, '定点模式反馈超时')
                return self._output(
                    vehicle, MODE_DEPTH, immediate_zero=True)
            if vehicle.reported_mode != MODE_DPROV:
                self._reset_motion_references(vehicle.now)
                self._transition(SAFE, '定点模式反馈丢失')
                return self._output(
                    vehicle, MODE_DEPTH, immediate_zero=True)
            return self._output(
                vehicle, MODE_DPROV, immediate_zero=True)

        self._reset_motion_references(vehicle.now)
        self._transition(SAFE, '未知状态')
        return self._output(vehicle, MODE_DEPTH, immediate_zero=True)
