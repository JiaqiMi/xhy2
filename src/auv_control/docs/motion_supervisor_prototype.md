# 运动—刹停—悬停控制原型

## 控制边界

- `motion_supervisor` 是原型运行时 `/cmd/pose/ned` 的唯一发布者。
- 运动和刹停使用 `mode=2`：下位机保持目标深度，上位机仅输出 `TX、TY、MZ`。
- 捕获稳定后使用 `mode=4`：六轴外部力清零，由下位机完成定点。
- 本原型不修改 `task1～task4`，也不包含下位机 PID 重置逻辑。

## 话题

| 方向 | 话题                   | 类型                           | 说明                     |
| ---- | ---------------------- | ------------------------------ | ------------------------ |
| 输入 | `/cmd/motion/goal`   | `geometry_msgs/PoseStamped`  | `map` 坐标系最终目标   |
| 输入 | `/cmd/motion/cancel` | `std_msgs/Empty`             | 刹停后悬停当前位置       |
| 输入 | `/status/vel`        | `geometry_msgs/TwistStamped` | `base_link` 速度反馈   |
| 输入 | `/status/auv`        | `auv_control/AUVData`        | 下位机模式反馈           |
| 输出 | `/cmd/pose/ned`      | `auv_control/PoseNEDcmd`     | 唯一运动控制输出         |
| 输出 | `/motion/state`      | `auv_control/MotionState`    | 状态、误差、速度和输出力 |

## 启动

新增了 `MotionState.msg`，首次使用前需要重新构建并加载工作空间：

```bash
catkin_make
source devel/setup.bash
```

先按现有流程启动驱动、地图原点和 TF 链路，并确认没有任务节点或其他测试节点发布 `/cmd/pose/ned`。随后启动原型：

```bash
roslaunch auv_control motion_supervisor_prototype.launch
```

可以同时发布一个测试目标：

```bash
roslaunch auv_control motion_supervisor_prototype.launch \
  start_goal_test:=true offset_frame:=base_link \
  offset_x:=1.0 offset_y:=0.0 yaw_offset_deg:=0.0
```

测试节点启动后读取初始 `map -> base_link` 位姿。`offset_frame=base_link`
时，`offset_x/offset_y` 分别表示初始艇体坐标系下的前/右偏置；
设置为 `map` 时表示北/东偏置。目标航向为初始航向加
`yaw_offset_deg`，目标 z 固定为 `-0.6 m`。

观察状态：

```bash
rostopic echo /motion/state
```

取消运动并在停稳位置悬停：

```bash
rostopic pub -1 /cmd/motion/cancel std_msgs/Empty '{}'
```

## 历史数据参考

对 `C:\Users\sixuh\Documents\B_matlab_ws\AUV\research\data\segments_0616` 中约 8.3 Hz 的历史数据进行离线统计，结果如下：

| 轴     | 样本类型             | 有效段数 |  减速度中位值 |       20% 分位值 |
| ------ | -------------------- | -------: | ------------: | ---------------: |
| TX / u | 推力归零后的被动减速 |       16 |  0.0486 m/s² |     0.0285 m/s² |
| TY / v | 推力归零后的被动减速 |       11 |  0.0269 m/s² |     0.0179 m/s² |
| MZ / r | 推力归零后的被动减速 |        5 |  5.93 deg/s² |     4.54 deg/s² |
| MZ / r | ±10000 主动反向刹转 |        2 | 66.58 deg/s² | 约 65.56 deg/s² |

历史 TX、TY 数据没有有效的主动反向刹车段，因此只能给出被动减速参考。MZ 主动结果对应 `|MZ|=10000`，不能直接用于当时旧配置中的 `brake_max_mz=120`。

历史数据还显示 `MZ > 0` 时 `r < 0`。下水前必须重新确认当前固件和坐标系是否仍保持该符号关系。如果保持不变，当前计算公式中的以下三个参数需要同时取负值：

```yaml
kp_yaw: -100.0
kr_yaw: -100.0
brake_gain_yaw: -1000.0
```

如果 `MZ > 0` 对应 `r > 0`，则三个参数均保持正值。禁止只修改其中一个参数。

## 2026-07-16 实测主动刹车参数

本次原始帧使用手动协议，十进制 `100` 为停止中值。通道 1、2、4 分别对应 `TX、MZ、TY`，手动值每偏离中值 1，debug 实际力约变化 100。推进器正反转推力不同，因此刹车限幅和有效减速度均按实际输出力符号拆分。

| 参数方向 | 刹车限幅 | 有效减速度 |
|---|---:|---:|
| `TX > 0` | 2000 | 0.05 m/s²，暂用负向标定值 |
| `TX < 0` | 3000 | 0.05 m/s² |
| `TY > 0` | 2000 | 0.025 m/s² |
| `TY < 0` | 4000 | 0.025 m/s² |
| `MZ > 0` | 3000 | 0.25 rad/s² |
| `MZ < 0` | 3000 | 0.30 rad/s² |

停车距离计算不再使用单一减速度：

- `u > 0` 时需要负 `TX` 刹车，选择 `brake_acceleration_tx_negative`；
- `u < 0` 时需要正 `TX` 刹车，选择 `brake_acceleration_tx_positive`；
- `v > 0` 时需要负 `TY` 刹车，选择 `brake_acceleration_ty_negative`；
- `v < 0` 时需要正 `TY` 刹车，选择 `brake_acceleration_ty_positive`；
- yaw 根据实际刹车 `MZ` 的符号选择对应角减速度；
- 目标同时包含 x/y 分量时，使用相关轴中较小的减速度，避免高估制动能力。

实测还确认 `MZ > 0` 时 `r < 0`，因此 yaw 相关增益统一为负值：

```yaml
kp_yaw: -100.0
kr_yaw: -100.0
brake_gain_yaw: -6000.0
```

为达到本次标定使用的实际反向力，刹车增益调整为：

```yaml
brake_gain_x: 15000.0
brake_gain_y: 30000.0
brake_gain_yaw: -6000.0
```

上位机运动与刹车变化率均设置为 `10000/周期`，实际步进保护由底层驱动完成：

```yaml
force_slew_per_cycle: 10000.0
brake_force_slew_per_cycle: 10000.0
```

当前 TX 正向刹车尚无完整的反向运动—停稳样本，暂时使用 `2000` 和 `0.05 m/s²`。后续补测后只需更新 `brake_max_tx_positive` 与 `brake_acceleration_tx_positive`。
