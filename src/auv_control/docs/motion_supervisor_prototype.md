# 运动—刹停—悬停控制原型

## 控制边界

- `motion_supervisor` 是原型运行时 `/cmd/pose/ned` 的唯一发布者。
- 运动和刹停使用 `mode=2`：下位机保持目标深度，上位机仅输出 `TX、TY、MZ`。
- 捕获稳定后使用 `mode=4`：六轴外部力清零，由下位机完成定点。
- 本原型不修改 `task1～task4`，也不包含下位机 PID 重置逻辑。

## 话题

| 方向 | 话题 | 类型 | 说明 |
|---|---|---|---|
| 输入 | `/cmd/motion/goal` | `geometry_msgs/PoseStamped` | `map` 坐标系最终目标 |
| 输入 | `/cmd/motion/cancel` | `std_msgs/Empty` | 刹停后悬停当前位置 |
| 输入 | `/status/vel` | `geometry_msgs/TwistStamped` | `base_link` 速度反馈 |
| 输入 | `/status/auv` | `auv_control/AUVData` | 下位机模式反馈 |
| 输出 | `/cmd/pose/ned` | `auv_control/PoseNEDcmd` | 唯一运动控制输出 |
| 输出 | `/motion/state` | `auv_control/MotionState` | 状态、误差、速度和输出力 |

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
  start_goal_test:=true target_x:=1.0 target_y:=0.0 \
  target_z:=1.5 target_yaw_deg:=0.0
```

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

| 轴 | 样本类型 | 有效段数 | 减速度中位值 | 20% 分位值 |
|---|---|---:|---:|---:|
| TX / u | 推力归零后的被动减速 | 16 | 0.0486 m/s² | 0.0285 m/s² |
| TY / v | 推力归零后的被动减速 | 11 | 0.0269 m/s² | 0.0179 m/s² |
| MZ / r | 推力归零后的被动减速 | 5 | 5.93 deg/s² | 4.54 deg/s² |
| MZ / r | ±10000 主动反向刹转 | 2 | 66.58 deg/s² | 约 65.56 deg/s² |

历史 TX、TY 数据没有有效的主动反向刹车段，因此只能给出被动减速参考。MZ 主动结果对应 `|MZ|=10000`，不能直接用于默认的 `brake_max_mz=120`。

历史数据还显示 `MZ > 0` 时 `r < 0`。下水前必须重新确认当前固件和坐标系是否仍保持该符号关系。如果保持不变，当前计算公式中的以下三个参数需要同时取负值：

```yaml
kp_yaw: -100.0
kr_yaw: -100.0
brake_gain_yaw: -1000.0
```

如果 `MZ > 0` 对应 `r > 0`，则三个参数均保持正值。禁止只修改其中一个参数。

## 完整标定顺序

### 1. 准备与记录

1. 确认水池有足够停车距离，设置人工急停并安排安全观察人员。
2. 保证 `motion_supervisor` 是 `/cmd/pose/ned` 的唯一发布者。
3. 使用 `mode=2` 保持深度，关闭正式任务节点。
4. 记录 `/status/vel`、`/status/auv`、`/cmd/pose/ned`、`/motion/state` 和 `/tf`。
5. 每组试验记录电池电压、载荷、深度和水流方向；正反方向分别重复至少五次。

### 2. 确认符号

从小力开始，每次保持 1～2 秒并立即归零：

1. 正 TX 应产生正 u，负 TX 应产生负 u。
2. 正 TY 应产生正 v，负 TY 应产生负 v。
3. 记录正 MZ 对应的 r 符号，并按“历史数据参考”统一设置三个 yaw 增益的符号。
4. 任一轴方向与预期不一致时停止后续试验，先修正符号。

### 3. 标定死区和安全运动限幅

1. 从绝对值 50 开始，每次增加 50，保持 2 秒。
2. 连续三个采样点出现同向、可重复的速度变化时，将该力记为该方向死区上沿。
3. 正反方向分别测量，取较大的死区值。
4. 初始运动限幅设置为死区上沿的 1.5～2 倍，并在低风险试验中逐步增加。

需要调节：

```yaml
max_tx: 300.0
max_ty: 180.0
max_mz: 120.0
force_slew_per_cycle: 50.0
```

如果默认值低于实测死区，必须提高对应 `max_*`；变化率过猛时减小 `force_slew_per_cycle`。

### 4. 标定主动刹车力与增益

每次先加速到目标速度，再切换到实际刹车控制律：

```text
TX = clamp(-brake_gain_x × u)
TY = clamp(-brake_gain_y × v)
MZ = clamp(-brake_gain_yaw × r)
```

推荐测试点：

| 轴 | 初始速度测试点 | 停止阈值 |
|---|---|---|
| TX | 0.15、0.20、0.30 m/s | 0.03 m/s |
| TY | 0.08、0.12、0.18 m/s | 0.03 m/s |
| MZ | 10、20、30 deg/s | 3 deg/s |

调节顺序：

1. 先设置较小的 `brake_max_*`，确认反向力确实降低速度而不是继续加速。
2. 增大 `brake_gain_*`，直到中高速阶段能够较快达到 `brake_max_*`。
3. 逐步增大 `brake_max_*`，直到停车距离满足要求且不出现明显反向速度。
4. 如果末段频繁正反切换，降低 `brake_gain_*` 或增大停止阈值。

需要调节：

```yaml
brake_gain_x: 1000.0
brake_gain_y: 1000.0
brake_gain_yaw: -1000.0  # 仅当实测仍为 MZ>0、r<0；否则使用 +1000.0
brake_max_tx: 300.0
brake_max_ty: 180.0
brake_max_mz: 120.0
```

### 5. 计算有效刹车减速度

对每个有效刹车区间定义：

```text
t0：反向刹车开始生效
t1：速度降到停止阈值
a_eff = (|v(t0)| - |v(t1)|) / (t1 - t0)
```

也可以对 `|v| = v0 - a_eff × t` 做线性拟合。排除命令切换后的第一个采样点，在速度反向前结束；拟合 `R² < 0.7` 的记录不参与统计。

最终参数取正反方向有效结果的较小 20% 分位值，再乘以 0.8 安全系数：

```text
配置减速度 = 0.8 × min(P20(正方向), P20(反方向))
```

需要调节：

```yaml
brake_acceleration: 0.02          # 无主动 TX/TY 数据前的临时保守值
angular_brake_acceleration: 0.30 # 必须按实际 brake_max_mz 重新测量
```

现有默认 `brake_acceleration=0.10` 相对历史被动减速偏乐观，主动刹车标定完成前建议使用 `0.02`。

### 6. 标定延迟和停车余量

1. `control_delay` 取刹车命令时间到速度斜率明显反向时间的 90% 分位值。
2. 用实测初速度计算预测停车距离，并与速度积分得到的实际停车距离比较。
3. `brake_margin` 取“实际距离减预测距离”的 95% 分位值，并至少保留 0.10 m。
4. yaw 使用相同方法调节 `yaw_brake_margin_deg`。

需要调节：

```yaml
control_delay: 0.35
brake_margin: 0.15
yaw_brake_margin_deg: 3.0
```

### 7. 标定运动 PD 增益

1. 保持速度阻尼项，先从较小 `kp_x/kp_y` 开始，逐步提高接近速度。
2. 出现超调时优先提高 `kv_x/kv_y`，仍无法改善时再降低 `kp_x/kp_y`。
3. yaw 先调 `kp_yaw`，再增加 `kr_yaw` 抑制角速度和转向超调。
4. 每次只修改一个参数，并使用相同目标、初始位置和电压重复试验。

需要调节：

```yaml
kp_x: 200.0
kp_y: 200.0
kv_x: 300.0
kv_y: 300.0
kp_yaw: -100.0  # 仅当实测仍为 MZ>0、r<0；否则使用 +100.0
kr_yaw: -100.0  # 必须与 kp_yaw、brake_gain_yaw 同号
```

### 8. 标定捕获与接管条件

1. 单轴和 yaw 均能稳定刹停后，再启用定点接管。
2. 首先使用较严格的速度阈值和较长稳定帧数。
3. 如果长期无法接管，可小幅放宽位置/航向范围，但不能先放宽速度条件。
4. 定点接管后一秒内若水平速度超过 0.08 m/s，应视为接管失败并回退刹车。

需要调节：

```yaml
capture_radius: 0.25
capture_exit_radius: 0.35
horizontal_speed_threshold: 0.05
yaw_tolerance_deg: 5.0
path_yaw_tolerance_deg: 5.0
yaw_rate_threshold_deg_s: 3.0
stable_frames: 5
hover_fault_speed: 0.08
hover_fault_yaw_rate_deg_s: 6.0
mode_ack_timeout: 1.0
```

### 9. 最终验收

依次执行单轴 TX、单轴 TY、单独 MZ、组合 x/y/yaw 和定点接管。每项与原定点控制使用相同初始条件重复至少五次，检查：

- 最大超调中位数至少降低 50%；
- 刹停后水平速度不超过 0.05 m/s、角速度不超过 3 deg/s；
- 定点接管后一秒内水平速度峰值不超过 0.08 m/s；
- 连续三秒保持在 0.25 m、5 deg 捕获范围内；
- 全过程无力指令越界、无反向加速、无状态频繁振荡。

所有默认力均为保守起点，未经实船标定不得直接提高。协议绝对范围为 `[-10000,10000]`。
