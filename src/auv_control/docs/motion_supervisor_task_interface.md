# 运动状态机任务节点接口

## 1. 用途与边界

本文面向 `task1～task4` 等任务节点，说明如何调用 `motion_supervisor`
完成水平平移、主动刹停、最终转向和下位机定点接管。

`/cmd/motion/goal` 中的位姿始终表示最终 `base_link` 位姿。控制器内部根据
本次计划 yaw 方向选择正向或负向虚拟旋转中心，并自动完成刚体换算。
任务节点不需要也不能指定旋转中心。

任务节点负责：

- 生成 `map` 坐标系下的最终目标 `x、y、z、yaw`；
- 以 5 Hz 发布目标；
- 订阅 `/motion/state`，根据 `HOVER` 判断到达；
- 根据任务超时、取消和异常状态决定任务流程。

任务节点不负责：

- 不直接发布 `/cmd/pose/ned`；
- 不直接控制 `TX、TY、MZ`；
- 不发送速度指令；
- 不切换下位机模式；
- 不自行判断何时开始主动刹车。

运行期间，`motion_supervisor` 必须是 `/cmd/pose/ned` 的唯一发布者。

## 2. 接口总览

| 方向 | Topic | 消息类型 | 频率 | 用途 |
|---|---|---|---:|---|
| 任务 → 状态机 | `/cmd/motion/goal` | `geometry_msgs/PoseStamped` | 建议 5 Hz | 发布最终目标位姿 |
| 任务 → 状态机 | `/cmd/motion/cancel` | `std_msgs/Empty` | 单次 | 刹停后悬停当前位置 |
| 状态机 → 任务 | `/motion/state` | `auv_control/MotionState` | 5 Hz | 状态、误差、速度和力矩反馈 |


## 3. 目标输入 `/cmd/motion/goal`

### 3.1 消息约定

```text
header.frame_id = "map"
pose.position.x = NED/map 北向绝对位置，单位 m
pose.position.y = NED/map 东向绝对位置，单位 m
pose.position.z = NED/map 绝对目标深度，单位 m
pose.orientation = 最终 yaw 对应的有效单位四元数
```

注意：

- `frame_id` 必须严格为 `map`，否则目标会被拒绝；
- `x、y、z、yaw` 必须为有限值；
- 四元数模长不能接近零；
- 当前只使用 yaw，任务节点通常令 roll、pitch 为 0；
- 深度来自每条 goal 的 `position.z`；
- 当前系统原点为水池底部，向下为正，例如高 0.6 m 使用 `z=-0.6`。

Python 发布示例：

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import rospy
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler


def build_goal(x, y, z, yaw):
    """构造 map 坐标系绝对目标。"""
    message = PoseStamped()
    message.header.stamp = rospy.Time.now()
    message.header.frame_id = 'map'
    message.pose.position.x = x
    message.pose.position.y = y
    message.pose.position.z = z
    quaternion = quaternion_from_euler(0.0, 0.0, yaw)
    message.pose.orientation.x = quaternion[0]
    message.pose.orientation.y = quaternion[1]
    message.pose.orientation.z = quaternion[2]
    message.pose.orientation.w = quaternion[3]
    return message
```

### 3.2 发布频率

建议任务节点以 5 Hz 持续发布当前最终目标，与运动状态机控制频率一致。

- 停止发布不会取消运动，状态机会继续执行最后一个有效目标；
- 需要终止任务时，必须发布 `/cmd/motion/cancel`；
- 发布频率短时抖动不会清空目标；
- 任务节点应自行设置任务级超时，不能只依赖目标发布是否正常。

节点启动后会锁定第一帧完整有效的 `base_link` 位姿，并按默认计划方向换算
启动控制点，先在该处刹停并完成 `mode=4` 接管。启动定点完成前收到的任务
目标不会改变锁定位姿，只缓存最新一条；达到配置的稳定发布帧数前，对外
保持 `CAPTURE`，不会提前发布 `HOVER`。`startup_complete=true` 后自动
执行缓存目标。

## 4. 状态反馈 `/motion/state`

`motion_supervisor` 每个控制周期发布一次，默认频率为 5 Hz。

TF 或 IMU 点速度尚未产生第一帧时也会发布 `SAFE`，此时不会下发运动指令。
获得第一帧完整有效反馈后，状态机只锁定一次启动位姿，不再跟随实时位置改写。
TF 后续短时失败不会阻塞状态话题；状态机使用最后有效位姿，并在真实 TF
年龄超过 `feedback_timeout` 后进入 `SAFE`。

### 4.1 字段

| 字段 | 单位 | 说明 |
|---|---|---|
| `header.stamp` | ROS 时间 | 本周期反馈时间 |
| `header.frame_id` | - | 固定为 `map` |
| `state` | - | 当前状态编号 |
| `goal_active` | - | 是否持有当前或待切换目标，不能作为到达标志 |
| `goal` | - | 当前采用的最终 `base_link` 绝对目标 |
| `position_error` | m | 当前计划方向虚拟旋转中心到内部控制目标的水平误差 |
| `base_position_error` | m | 当前 `base_link` 到任务最终目标的水平误差 |
| `yaw_error` | rad | 最终 yaw 误差，已归一化 |
| `horizontal_speed` | m/s | 低通后的水平合速度 |
| `yaw_rate` | rad/s | 低通后的航向角速度 |
| `tx/ty/mz` | 协议单位 | 本周期实际下发值 |
| `x/y/yaw_axis_state` | - | 三轴 `AXIS_HOLD/TRACK/BRAKE` 子状态 |
| `x/y_axis_error` | m | 当前虚拟旋转中心本体前、右方向控制误差 |
| `x/y_axis_speed` | m/s | 杆臂补偿后的旋转中心前、右速度 |
| `startup_complete` | - | 是否已完成首帧锁定和首次 mode=4 接管 |
| `reason` | - | 当前状态或最近切换原因 |

### 4.2 状态含义

| 状态 | 数值 | 任务节点含义 |
|---|---:|---|
| `IDLE` | 0 | 尚未执行目标 |
| `ALIGN_PATH` | 1 | 兼容保留，当前正常流程不使用 |
| `ALIGN_PATH_BRAKE` | 2 | 兼容保留，当前正常流程不使用 |
| `TRANSLATE` | 3 | 统一 XY/yaw 位姿跟踪，当前未进入停车距离 |
| `TRANSLATE_BRAKE` | 4 | 统一 XY 或 yaw 轨迹已进入停车距离 |
| `ALIGN_FINAL` | 5 | 仅保留消息编号，当前控制器不再产生 |
| `FINAL_BRAKE` | 6 | 仅保留消息编号，当前控制器不再产生 |
| `CAPTURE` | 7 | 判断稳定或已发送 `mode=4`、正在等待接管确认 |
| `HOVER` | 8 | 下位机已反馈 `mode=4`，目标到达并完成定点接管 |
| `SAFE` | 9 | 反馈超时、接管超时或模式异常，任务不得继续推进 |

典型流程：

```text
IDLE
→ TRANSLATE
↔ TRANSLATE_BRAKE
→ CAPTURE
→ HOVER
```

纯旋转与平移旋转组合也使用同一流程。取消后通常为：

```text
TRANSLATE_BRAKE → CAPTURE → HOVER
```

## 5. 任务如何判断到达

到达条件为：

```text
/motion/state.state == MotionState.HOVER
```

`HOVER` 同时保证：

1. 水平位置误差、yaw 误差、水平速度和角速度已连续满足捕获条件；
2. 状态机已发送最终目标的 `mode=4` 且六轴外部力为零；
3. 本次接管开始后收到新的 `/status/auv.control_mode==4` 反馈；
4. 下位机模式反馈仍在 `mode_ack_timeout` 允许的新鲜度范围内。

任务节点还应确认状态反馈新鲜：

```python
def motion_arrived(message):
    """仅将新鲜的 HOVER 状态视为到达。"""
    age = (rospy.Time.now() - message.header.stamp).to_sec()
    return (
        age <= 0.5
        and message.startup_complete
        and message.state == message.HOVER
    )
```

重要限制：

- `goal_active` 在 `HOVER` 中仍为 `true`，不能用于判断完成；
- `CAPTURE` 不代表到达，因为此时可能仍在等待 `mode=4` 反馈；
- `position_error` 只表示水平误差，深度由下位机控制；
- `HOVER` 表示水平位置和 yaw 到达并完成定点接管，不单独保证深度误差；
- 若任务必须确认深度，应额外使用 `/status/auv` 的实际深度与自己的阈值；
- `HOVER` 可能因新目标或定点异常退出，任务节点不应永久锁存该状态。

### 5.1 连续目标下的判断

目标持续变化时不使用目标编号。任务节点应：

1. 保存自己最近发布的目标；
2. 确认 `startup_complete==true`；
3. 收到 `/motion/state` 后确认 `state==HOVER`；
4. 确认 `message.goal` 与最近目标一致；
5. 发布明显不同的新目标后，清除任务节点内部的上一阶段完成状态。

`message.goal` 是状态机当前实际采用的目标，可用于避免把上一目标遗留的
`HOVER` 当作新目标到达。比较 yaw 时必须进行 `±π` 归一化。

## 6. 不同运动类型的下发方式

所有目标最终都必须转换为 `map` 下的绝对位姿。

### 6.1 纯平移

在运动开始时固定目标 yaw 为当前航向，只改变目标 `x、y`：

```text
goal.x = 目标绝对 x
goal.y = 目标绝对 y
goal.z = 任务目标深度
goal.yaw = 平移开始时的当前 yaw
```

执行过程：

```text
TRANSLATE ↔ TRANSLATE_BRAKE → CAPTURE → HOVER
```

即使最终 yaw 与起始 yaw 相同，也会经过最终稳定确认。

注意：不要每周期用“当前 AUV 位置 + 相对偏置”重新计算目标，否则目标会随
AUV 一起移动，造成无法到达。相对偏置必须在阶段开始时计算一次。

连续平移时，需要任务节点把控步长，如果只是移动到目标位置，可无视步长。

### 6.2 纯旋转

在旋转开始时锁定当前 `x、y`，只改变最终 yaw：

```text
goal.x = 旋转开始时的当前 x
goal.y = 旋转开始时的当前 y
goal.z = 任务目标深度
goal.yaw = 目标绝对 yaw
```

纯旋转仍由统一 XY/yaw 轨迹同步控制：

```text
TRANSLATE ↔ TRANSLATE_BRAKE → CAPTURE → HOVER
```

当前 `base_link` 与 IMU/GNSS 定位点重合。TF 树固定提供
`control_link_positive` 和 `control_link_negative`。控制器按目标 yaw
相对当前 yaw 的最短归一化误差选择一次：正误差使用正中心，负误差使用
负中心。同一目标重复发布时方向和中心保持锁存，主动刹转 `MZ` 反号也不会
切换中心。若标定中心与 IMU 存在杆臂，`base_link` 与 IMU 会按刚体关系
沿小圆弧运动；最终 `base_link` 仍到达任务指定的位姿。

任务节点只需发布最终 yaw，不要根据本周期 `MZ` 正负修改目标，也不要自行
选择正负中心。5 Hz 重复发布同一目标不会改变本次动作已锁存的计划方向。

连续旋转时，旋转步长仍需要任务节点来把控，如果只是旋转到目标角度，可无视步长。

### 6.3 既有平移又有旋转

一次发布完整最终位姿：

```text
goal.x/y = 最终绝对水平位置
goal.z = 最终绝对深度
goal.yaw = 最终绝对航向
```

状态机固定采用“先平移、后转向”：

```text
保持平移开始时航向，以 TX/TY 到达并刹停
→ 调整最终 yaw 并刹转
→ mode=4 定点接管
```

如果只需到目标位置，则不考虑步长。
如果需要先指向目标位置，再到达目标位置和航向，则应该分两阶段发布指令，先原地旋转，再移动

### 6.4 base_link 相对偏置转换

任务需要“前 `dx`、右 `dy`”时，在阶段开始时读取一次当前
`map → base_link` 位姿：

```text
goal_x = current_x + cos(yaw) × dx - sin(yaw) × dy
goal_y = current_y + sin(yaw) × dx + cos(yaw) × dy
```

- `dx > 0`：向前；
- `dx < 0`：向后；
- `dy > 0`：向右；
- `dy < 0`：向左。

转换完成后持续发布同一个 `map` 绝对目标，不要重复累加偏置。

## 7. 连续目标更新与轴解耦

状态机始终采用最后收到的有效目标，不再按目标变化量触发整艇预抢占刹停。
每个周期把目标位置误差转换到当前本体前、右轴，然后独立计算停车距离：

```text
X: TRACK / BRAKE / HOLD → TX
Y: TRACK / BRAKE / HOLD → TY
Yaw: 水平两轴 HOLD 后执行 TRACK / BRAKE / HOLD → MZ
```

矩形轨迹转角时，旧方向轴进入 `BRAKE` 的同时，新方向轴可以保持 `TRACK`。
例如 X 边结束并开始 Y 边时，允许 `X=BRAKE、Y=TRACK`，避免 X 惯性未处理
却直接叠加横移。目标继续向外移动并超过停车区滞环时，相应轴可由 `BRAKE`
返回 `TRACK`。

相同目标重复发布不会重置 `CAPTURE` 或 mode=4 接管计时。目标停止更新后，
状态机会继续执行最后一条目标并按正常流程进入 `HOVER`。

## 8. 取消与异常处理

### 8.1 取消

```python
cancel_pub.publish(Empty())
```

状态机会：

```text
主动刹停
→ 将停稳时的实际 x/y/z/yaw 设为目标
→ CAPTURE
→ mode=4 接管
→ HOVER
```

取消不是立即切断深度控制，也不是立即清空目标。

### 8.2 SAFE

以下情况会进入 `SAFE`：

- TF 或速度反馈超过 `feedback_timeout`；
- 发送 `mode=4` 后超过 `mode_ack_timeout` 未确认；
- 已在 `HOVER` 时下位机模式反馈不再为 4 或反馈超时；
- 状态机遇到未知状态。

任务节点收到 `SAFE` 后：

1. 不得判定任务完成；
2. 停止推进任务阶段；
3. 记录 `reason`；
4. 根据任务安全策略选择等待恢复、取消或人工接管。

反馈恢复后，状态机会先刹停确认，不会直接恢复大力运动。

## 9. 任务节点最小订阅示例

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from auv_control.msg import MotionState


class TaskMotionMonitor(object):
    """任务节点侧的运动状态监视器。"""

    def __init__(self):
        self.latest_state = None
        rospy.Subscriber(
            '/motion/state',
            MotionState,
            self.state_callback,
            queue_size=1,
        )

    def state_callback(self, message):
        self.latest_state = message

    def arrived(self):
        """判断当前目标是否已接管到 HOVER。"""
        if self.latest_state is None:
            return False
        age = (
            rospy.Time.now() - self.latest_state.header.stamp
        ).to_sec()
        return (
            age <= 0.5
            and self.latest_state.startup_complete
            and self.latest_state.state == MotionState.HOVER
        )

    def failed(self):
        """判断运动状态机是否进入安全状态。"""
        return (
            self.latest_state is not None
            and self.latest_state.state == MotionState.SAFE
        )
```

实际任务还应增加：

- 目标与 `latest_state.goal` 的对应检查；
- 每个任务阶段的最长执行时间；
- `SAFE`、任务取消和人工接管处理；
- 深度需要严格到达时的独立误差判断。

## 10. 接入检查表

- [ ] 任务目标均为 `map` 下最终 `base_link` 绝对位姿；
- [ ] 相对偏置只在阶段开始时转换一次；
- [ ] 目标包含有效 z 和单位四元数；
- [ ] 目标以约 5 Hz 发布；
- [ ] 任务节点不发布 `/cmd/pose/ned`；
- [ ] `/motion/state` 反馈频率约 5 Hz；
- [ ] 任务启动前确认 `startup_complete`；
- [ ] 只用新鲜的 `HOVER` 判断到达；
- [ ] 不用 `goal_active` 或 `CAPTURE` 判断到达；
- [ ] 新任务阶段会清除本地旧完成状态；
- [ ] 任务配置了超时、取消和 `SAFE` 处理；
- [ ] 对深度到达有要求时单独检查实际深度。

## 11. 文件说明

`motion_supervisor_core.py` 核心算法

`motion_supervisor.py` 状态机实现
