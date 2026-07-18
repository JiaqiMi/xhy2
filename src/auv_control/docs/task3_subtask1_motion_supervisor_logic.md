# 子任务1 motion_supervisor版本说明

## 1. 文件与版本边界

本版本是新增实现，旧版子任务1继续保留：

```text
旧版：
test/test_task3_1_acquire_area.py
launch/task3_subtask1_acquire_area.launch

新版：
test/test_task3_1_acquire_area_motion.py
launch/task3_subtask1_acquire_area_motion.launch
```

新版严格使用 `motion_supervisor` 任务接口：

| 方向 | 话题 | 类型 | 用途 |
| --- | --- | --- | --- |
| 任务发布 | `/cmd/motion/goal` | `geometry_msgs/PoseStamped` | 5Hz持续发布map绝对目标 |
| 任务发布 | `/cmd/motion/cancel` | `std_msgs/Empty` | 主动刹停并在实际停稳位置HOVER |
| 任务订阅 | `/motion/state` | `auv_control/MotionState` | 判断状态、误差、速度、输出和HOVER接管 |
| 任务订阅 | `/arrow/direction` | `std_msgs/String` | 箭头中心、角度和置信度 |
| 任务订阅 | `/status/auv` | `auv_control/AUVData` | 独立检查深度和最低对地距离 |

新版任务节点不会发布 `/cmd/pose/ned`，也不会计算或设置 `TX/TY/MZ`。
`motion_supervisor` 必须是 `/cmd/pose/ned` 的唯一发布者。

## 2. 当前本地代码兼容提醒

当前本地旧原型 `motion_supervisor_core.py` 的 `set_goal()` 会把每次收到的goal都当成新目标抢占，即使目标数值没有变化，也可能反复进入刹停。新接口说明要求任务节点以5Hz持续发布同一个目标，并由状态机区分相同目标、小幅更新和大幅更新。

因此：

1. 新版任务代码按照最新接口说明编写。
2. 队友的新 `motion_supervisor` 合并前，不要使用本地旧原型进行实船测试。
3. 新版launch不自动包含当前旧原型，必须先单独启动合并后的新运动状态机。
4. 合并后先在陆上或低推力条件确认重复目标不会持续触发抢占刹停。

## 3. 完整任务逻辑

```text
等待map -> base_link、/status/auv和/motion/state
  -> 发布启动位置的map绝对目标
  -> 只在新鲜MotionState.HOVER且目标一致时开始悬停计时
  -> 连续HOVER 10秒
  -> 在阶段开始时一次性计算base_link前方1.20m搜索终点
  -> 持续发布同一个搜索终点
  -> 箭头连续稳定识别3帧
  -> 发布/cmd/motion/cancel
  -> 等待取消之后的新鲜HOVER
  -> 根据每个新箭头帧生成一次视觉小步map绝对目标
  -> 箭头连续居中5帧
  -> 再次cancel并等待HOVER
  -> 定点状态下重新确认箭头方向连续3帧
  -> 发布包含最终x/y/z/yaw的完整目标
  -> motion_supervisor完成平移、转向、刹停和mode4接管
  -> 图像中心和箭头方向连续满足5帧
  -> 最新目标对应新鲜HOVER
  -> 在阶段开始时计算一次base_link前方0.35m绝对终点
  -> 等待该终点对应HOVER
  -> HOVER和深度误差连续满足10秒
  -> 发布/finished和cancel，任务结束
```

### 3.1 启动悬停

任务读取一次当前 `map -> base_link`，把当前 `x/y/z/yaw` 作为初始目标。只有以下条件同时满足才开始累计10秒：

```text
/motion/state消息新鲜
state == MotionState.HOVER
MotionState.goal与任务最近目标一致
```

如果HOVER中断，10秒计时清零。

### 3.2 向前搜索

搜索阶段不会直接给前进推力。程序在阶段开始时读取一次当前位姿，并按当前航向计算前方 `1.20m` 的map绝对终点：

```text
goal_x = current_x + cos(current_yaw) * 1.20
goal_y = current_y + sin(current_yaw) * 1.20
goal_z = 启动目标深度
goal_yaw = 搜索开始时航向
```

该目标只计算一次，之后以5Hz重复发布同一个值。箭头连续稳定识别3帧后不再发布远处终点，而是发布cancel，由状态机主动刹停并在实际停稳位置完成mode4接管。

如果先到达1.20m搜索终点仍未识别箭头，任务失败并取消运动。

### 3.3 视觉小步居中

图像误差转换为本体前、右方向的小步位移，不再转换成推力：

```text
normalized_u = error_u / (image_width / 2)
normalized_v = error_v / (image_height / 2)

forward_step = -visual_forward_gain_m * normalized_v
right_step = visual_lateral_gain_m * normalized_u
```

小步经过 `visual_min_step_m` 和 `visual_max_step_m` 限制，再使用当前 `base_link` 航向转换为map绝对目标。每个新模型帧最多计算一次；主循环重复运行时不会对同一帧重复累加偏置。

视觉目标默认小于 `0.50m`，属于运动状态机的小幅目标更新范围。运动速度、最大推力、主动刹车和捕获精度应在 `motion_supervisor.yaml` 中调整，任务launch不再提供推力参数。

### 3.4 方向确认和最终转向

箭头先连续居中5帧，然后cancel刹停并进入HOVER。定点状态下重新确认箭头方向连续3帧，角度抖动不能超过 `stable_angle_tolerance_deg`。

航向换算仍沿用旧版逻辑：

```text
相对修正 = yaw_correction_sign
         * wrap(camera_forward_angle_deg - arrow_angle_deg)

最终map航向 = 当前map航向 + 相对修正
```

任务一次发布完整最终航向，不再自己分步设置航向或控制MZ。转向速度、角速度阻尼和主动刹转全部由 `motion_supervisor` 负责。转向期间如果箭头偏离图像中心，任务仍可按新模型帧小幅更新目标 `x/y`，最终yaw保持不变。

### 3.5 前垂推位置补偿

航向、图像中心和最新目标HOVER全部通过后，任务读取一次当前 `base_link` 位姿，并沿机器人实际前方计算 `0.35m` 绝对终点：

```text
goal_x = current_x + cos(current_yaw) * 0.35
goal_y = current_y + sin(current_yaw) * 0.35
goal_z = 当前任务目标深度
goal_yaw = 箭头最终航向
```

这个0.35m偏置只在阶段开始时计算一次，不会每周期使用“当前位置+0.35m”造成目标不断向前移动。到达后目标位置与前垂推基本对齐。

本版本不再使用旧版的 `map -> camera` 坐标去把 `base_link` 完全移动到箭头中心，因为新的机械目标是让箭头与前垂推位置对齐。

## 4. 到达与保护逻辑

### 4.1 到达条件

任务只把以下结果视为当前目标到达：

```text
MotionState消息年龄 <= motion_state_timeout
MotionState.state == MotionState.HOVER
MotionState.goal与任务最近发布的目标一致
```

不会使用 `goal_active` 或 `CAPTURE` 判断完成。

目标一致性默认允许：

| 项目 | 默认容差 |
| --- | ---: |
| 水平位置 | `0.03m` |
| z目标 | `0.03m` |
| yaw目标 | `2deg` |

### 4.2 取消

以下情况发布cancel：

- 搜索期间稳定识别到箭头；
- 视觉居中连续满足5帧；
- 视觉居中或航向对准仍在运动时，箭头丢失超过 `visual_loss_cancel_seconds`；
- 定点方向确认时，箭头丢失超过 `detection_timeout`；
- 任务成功、失败或节点关闭。

cancel后必须收到时间戳晚于取消指令的新鲜HOVER，才认为主动刹停完成。超过 `cancel_timeout` 未完成则任务失败。

### 4.3 SAFE和反馈超时

- 启动时允许在 `motion_startup_timeout` 内等待首个有效运动反馈。
- 运动状态机正常运行过一次后，只要进入SAFE，任务立即发布cancel并失败退出。
- `/motion/state` 超过 `motion_state_timeout` 未更新时，任务取消并失败。
- `/status/auv` 超时后无法继续执行深度和离地保护，任务取消并失败。
- 总搜索和对准时间超过 `max_wait_seconds` 时失败。

### 4.4 深度和最低对地距离

`HOVER`只保证水平位置、yaw和mode4接管，不单独保证深度误差。最终阶段还要求：

```text
abs(current_depth - target_depth) <= max_depth_error
```

高度有效且低于 `min_ground_clearance=0.40m` 时，任务只把goal的z向上修正，不会继续靠近地面。候选安全z与当前目标z至少相差 `ground_clearance_goal_update_threshold` 才改写目标，避免高度噪声造成频繁的小目标更新。高度无效时打印警告。

### 4.5 调试日志

新版本会持续输出以下关键信息：

- 每个模型消息的帧号、有效或无效原因、置信度、中心坐标、中心误差、箭头角度和稳定帧进度；
- 每次视觉小步对应的箭头帧号、本体前/右偏置和转换后的map绝对目标；
- 当前任务阶段、`MotionState.state`、位置/航向控制误差、速度、角速度和TX/TY/MZ反馈；
- 等待HOVER时，任务最新目标与 `MotionState.goal` 的水平、z、yaw三项差值及各自容差；
- 视觉丢失时，最近有效帧年龄、模型消息年龄、当前保护阈值和cancel原因；
- 离地保护触发时，当前高度、所需上移量、目标z改写前后值和目标深度；
- 最终HOVER与深度稳定保持的累计时间，以及计时被打断的原因。

高频日志由 `log_interval` 和 `warning_log_interval` 节流，逐帧识别结果不节流，便于把终端帧号与网页视频逐帧对应。

## 5. 主要可调参数

识别参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `min_confidence` | `0.35` | 箭头最低置信度 |
| `stable_detection_count` | `3` | 搜索时首次锁定帧数 |
| `stable_center_tolerance_px` | `40` | 首次锁定中心抖动容差 |
| `center_stable_detection_count` | `5` | 图像居中和最终对准帧数 |
| `heading_stable_detection_count` | `3` | 定点确认方向帧数 |
| `stable_angle_tolerance_deg` | `12` | 方向确认角度抖动容差 |
| `detection_timeout` | `1.0s` | 识别帧连续性和定点方向确认的丢失超时 |
| `visual_loss_cancel_seconds` | `0.5s` | 居中或航向运动时丢失箭头后的刹停阈值，必须不大于 `detection_timeout` |

任务位置参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `search_forward_distance` | `1.20m` | 直行搜索最大终点距离 |
| `front_thruster_forward_offset` | `0.35m` | 最终base_link前移距离 |
| `initial_hover_seconds` | `10s` | 启动HOVER稳定时间 |
| `final_hold_seconds` | `10s` | 最终HOVER和深度连续稳定时间 |
| `max_wait_seconds` | `300s` | 搜索和对准总超时 |

视觉目标参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `center_tolerance_u_px/v_px` | `35/35` | 图像中心容差 |
| `visual_forward_gain_m` | `0.20` | 垂直归一化误差到前后位移增益 |
| `visual_lateral_gain_m` | `0.20` | 水平归一化误差到左右位移增益 |
| `visual_min_step_m` | `0.01m` | 非零视觉目标最小步长 |
| `visual_max_step_m` | `0.08m` | 单个模型帧最大单轴步长 |
| `visual_forward_sign` | `1` | 前后方向相反时改为 `-1` |
| `visual_lateral_sign` | `1` | 左右方向相反时改为 `-1` |

航向标定参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `camera_forward_angle_deg` | `90deg` | 图像协议中“箭头正对相机前方”对应的角度 |
| `yaw_correction_sign` | `1` | 航向修正方向相反时改为 `-1` |
| `yaw_tolerance_deg` | `10deg` | 最终箭头方向允许误差 |

运动接口参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `rate` | `5Hz` | 任务目标发布频率 |
| `motion_state_timeout` | `0.5s` | 运动反馈新鲜度限制 |
| `motion_startup_timeout` | `10s` | 启动等待状态机时间 |
| `cancel_timeout` | `15s` | 取消到HOVER最长时间 |
| `goal_match_position_tolerance` | `0.03m` | 任务目标与状态机目标的水平匹配容差 |
| `goal_match_depth_tolerance` | `0.03m` | 任务目标与状态机目标的z匹配容差 |
| `goal_match_yaw_tolerance_deg` | `2deg` | 任务目标与状态机目标的yaw匹配容差 |
| `status_timeout` | `0.5s` | `/status/auv` 接收超时 |
| `max_depth_error` | `0.08m` | 最终深度误差容差 |
| `min_ground_clearance` | `0.40m` | 最低对地距离 |
| `ground_clearance_goal_update_threshold` | `0.01m` | 离地保护改写目标z的最小变化量 |
| `final_hold_timeout` | `30s` | 最终阶段累计稳定失败的最长等待时间 |
| `log_interval` | `1.0s` | 普通周期状态日志的最小间隔 |
| `warning_log_interval` | `2.0s` | 重复警告和错误日志的最小间隔 |

## 6. 推荐调试顺序

1. 合并新版 `motion_supervisor` 后，先确认重复发布完全相同goal不会反复触发抢占。
2. 不启动任务，只测试绝对目标、cancel、SAFE和HOVER反馈。
3. 将任务 `start_arrow_model=true`，机器人保持不动，只检查逐帧中心和角度日志。
4. 使用较小的 `search_forward_distance` 验证前方目标方向。
5. 验证 `visual_forward_sign/visual_lateral_sign`，再调整视觉增益和步长。
6. 验证 `yaw_correction_sign`，再由状态机配置调转向和刹转速度。
7. 最后单独测试 `front_thruster_forward_offset=0.35`，确认目标与前垂推位置关系。

## 7. 运行步骤

编译并加载工作空间：

```bash
catkin_make
source devel/setup.bash
```

启动底层、TF和传感器：

```bash
roslaunch auv_control begin.launch
```

单独启动合并后的新版 `motion_supervisor`。它必须独占 `/cmd/pose/ned`。

随后启动新子任务：

```bash
roslaunch auv_control task3_subtask1_acquire_area_motion.launch
```

调试时建议观察：

```bash
rostopic echo /arrow/direction
rostopic echo /cmd/motion/goal
rostopic echo /motion/state
rostopic echo /status/auv
rostopic echo /cmd/pose/ned
```

确认 `/cmd/pose/ned` 只有 `motion_supervisor` 一个发布者：

```bash
rostopic info /cmd/pose/ned
```
