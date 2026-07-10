# Task 3 三个测试子任务实现逻辑

本文档说明 `test` 目录下 Task 3 三个子任务脚本的当前实现思路。现在识别模型还没有训练好，所以代码先用参数模拟视觉识别结果；后续模型完成后，尽量只通过切换参数或接入 ROS topic 来使用真实识别结果。

相关脚本：

- `test_task3_1_acquire_area.py`：到达任务获取区
- `test_task3_2_get_task.py`：读取 ArUco 任务编号并亮灯
- `test_task3_3_inspect_and_drop.py`：移动到指定颜色管道区域并打开夹爪放球

## 公共约定

三个脚本都按 ROS 节点方式运行，核心思路是“测试先跑通流程，识别结果暂时写死”。

坐标约定：

- `map`：全局坐标系，控制器最终接收的目标位姿在这个坐标系下。
- `base_link`：机器人自身坐标系。
- 当前脚本按项目已有约定使用 `base_link`：`x` 表示前方，`y` 表示右方，`z` 表示向下。
- 因此“左侧 0.30 m”在代码里对应 `y = -0.30`。

主要话题：

- `/target`：发布 `geometry_msgs/PoseStamped`，告诉底层控制器机器人要去哪里。
- `/finished`：发布 `std_msgs/String`，表示当前测试子任务完成。
- `/auv_actuator_control`：发布 `auv_control/ActuatorControl`，控制新接口下的三色指示灯和夹爪舵机。

## 子任务 1：到达任务获取区

脚本：

```bash
rosrun auv_control test_task3_1_acquire_area.py
```

### 当前测试假设

比赛里原本应该识别水底箭头标志，根据箭头位置前往任务获取区。现在箭头识别模型还没有训练好，所以脚本先把箭头识别结果写成固定参数：

- 前方 `0.50 m`
- 右侧 `0.30 m`
- 深度不变，也就是 `down = 0.00 m`

对应默认参数：

```text
~arrow_forward = 0.50
~arrow_right   = 0.30
~arrow_down    = 0.00
```

### 实现流程

1. 节点启动后创建 `/target` 和 `/finished` 发布器，并创建 TF 监听器。
2. 在 `base_link` 坐标系下构造一个局部目标点：

   ```text
   x = arrow_forward
   y = arrow_right
   z = arrow_down
   ```

3. 通过 TF 把这个局部目标点从 `base_link` 转换到 `map`。
4. 转换完成后固定这个 `map` 坐标目标，不再跟着机器人实时变化。
5. 循环发布这个目标到 `/target`。
6. 同时读取当前机器人在 `map` 下的位置，计算当前位置和目标位置的距离误差、航向误差。
7. 当距离误差小于 `~arrive_dist`，航向误差小于 `~arrive_yaw_deg` 时，认为到达任务获取区。
8. 到达后保持当前位置 `~hold_seconds` 秒。
9. 发布 `/finished`，结束节点。

### 后续接真实箭头识别

后面箭头模型训练好后，优先保留这个运动框架，只替换“箭头参数来源”：

- 现在：`arrow_forward / arrow_right / arrow_down` 是启动参数。
- 后续：由箭头识别模块输出箭头相对机器人位置，再填入这三个量，或者改成订阅一个识别结果 topic。

## 子任务 2：读取 ArUco 编号并亮灯

脚本：

```bash
rosrun auv_control test_task3_2_get_task.py
```

### 当前测试假设

比赛里应该识别 ArUco 编号 `1~6`，再根据编号点亮对应颜色灯。现在 ArUco 识别模型还没有训练好，所以脚本默认用固定模拟序列：

```text
1,3,5,2,4,6
```

默认参数：

```text
~input_mode      = mock
~mock_aruco_ids  = [1, 3, 5, 2, 4, 6]
~light_seconds   = 3.0
~gap_seconds     = 0.5
~actuator_topic  = /auv_actuator_control
```

### 编号和颜色映射

规则来自 Task 3 比赛任务：

| ArUco 编号 | 目标颜色 |
| --- | --- |
| `1` / `2` | 黄色 |
| `3` / `4` | 绿色 |
| `5` / `6` | 红色 |

新接口 `/auv_actuator_control` 使用三个独立灯字段：

| 颜色 | `red_light` | `yellow_light` | `green_light` |
| --- | --- | --- | --- |
| 黄色 | `0` | `1` | `0` |
| 绿色 | `0` | `0` | `1` |
| 红色 | `1` | `0` | `0` |
| 熄灭 | `0` | `0` | `0` |

### 实现流程

1. 节点启动后读取 `~input_mode`。
2. 如果是 `mock` 模式，就从 `~mock_aruco_ids` 里按顺序取编号。
3. 如果是 `topic` 模式，就订阅 `~aruco_topic`，等待识别节点发布 `std_msgs/Int32` 编号。
4. 每次拿到一个 ArUco 编号后，通过映射表得到目标颜色。
5. 通过 `/auv_actuator_control` 发布灯光控制消息。
6. 每个颜色保持 `~light_seconds` 秒，默认 `3 秒`。
7. 两次亮灯之间熄灭 `~gap_seconds` 秒，默认 `0.5 秒`。
8. 模拟序列全部执行完，或者真实 topic 模式达到 `~max_topic_markers` 次后，熄灯并发布 `/finished`。

### 后续接真实 ArUco 识别

模型训练好后，不需要改亮灯逻辑，只需要让识别节点发布：

```text
topic: /task3/aruco_id
type:  std_msgs/Int32
data:  1~6
```

然后启动时切换参数：

```bash
rosrun auv_control test_task3_2_get_task.py _input_mode:=topic _aruco_topic:=/task3/aruco_id
```

如果一次任务只需要处理一个 ArUco，保持默认：

```text
~max_topic_markers = 1
```

如果需要连续测试多个真实识别结果，可以把它调大；设置为 `0` 表示不限制数量。

## 子任务 3：移动到管道区域并打开夹爪放球

脚本：

```bash
rosrun auv_control test_task3_3_inspect_and_drop.py
```

### 当前测试假设

比赛里应该识别黄色、绿色、红色三个管道区域，然后根据子任务 2 得到的目标颜色，移动到对应区域投放信标球。现在管道识别模型还没有训练好，所以脚本先假设：

- 已经识别到 `yellow / green / red` 三个颜色管道。
- 默认目标颜色是 `yellow`。
- 默认投放点在机器人前方 `0.50 m`、左侧 `0.30 m`。

默认参数：

```text
~target_mode          = mock
~target_color         = yellow
~mock_detected_colors = [yellow, green, red]
~drop_forward         = 0.50
~drop_left            = 0.30
~drop_down            = 0.00
```

由于 `base_link` 中 `y` 正方向是右方，所以代码里会把左侧 `0.30 m` 转成：

```text
x = +0.50
y = -0.30
z =  0.00
```

### 夹爪和灯光参数

脚本使用新接口 `/auv_actuator_control` 控制夹爪：

```text
~clamp_open   = 0x00
~clamp_closed = 0xFF
```

当前假设：

- `0x00` 表示夹爪全开，用于放球。
- `0xFF` 表示夹爪关闭，用于复位。

默认投放动作时间：

```text
~hold_seconds  = 1.0
~open_seconds  = 3.0
~close_seconds = 1.0
```

### 实现流程

1. 节点启动后创建 `/target`、`/finished`、`/auv_actuator_control` 发布器，并创建 TF 监听器。
2. 根据 `~target_mode` 决定目标来源。
3. 如果是 `mock` 模式，就用固定相对位移构造投放点。
4. 如果是 `topic` 模式，就订阅 `~target_topic`，等待识别节点发布 `geometry_msgs/PoseStamped`。
5. 把目标位姿转换到 `map` 坐标系。
6. 固定这个 `map` 下的投放目标，循环发布到 `/target`。
7. 读取机器人当前位姿，判断是否到达投放点。
8. 到达后保持 `~hold_seconds` 秒，同时可以点亮目标颜色灯，方便确认投放区域。
9. 发布夹爪打开值 `~clamp_open`，持续 `~open_seconds` 秒，完成放球。
10. 发布夹爪关闭值 `~clamp_closed`，持续 `~close_seconds` 秒，完成复位。
11. 熄灯，发布 `/finished`，结束节点。

### 后续接真实管道识别

模型训练好后，识别节点可以发布选中管道区域的目标位姿：

```text
topic: /task3/pipeline_target
type:  geometry_msgs/PoseStamped
```

这个 `PoseStamped` 可以直接在 `map` 坐标系下，也可以在 `base_link` 或相机坐标系下。脚本会尝试通过 TF 转到 `map`。

启动时切换参数：

```bash
rosrun auv_control test_task3_3_inspect_and_drop.py _target_mode:=topic _target_topic:=/task3/pipeline_target _target_color:=red
```

`~target_color` 仍然保留，是为了让脚本知道当前投放对应哪个颜色区域，并控制对应指示灯。

## 三个子任务之间的数据关系

当前三个脚本是独立测试脚本，每个脚本都可以单独运行：

```text
子任务 1：假设箭头结果 -> 生成任务获取区位置 -> 发布 /target
子任务 2：假设 ArUco 编号 -> 得到目标颜色 -> 发布 /auv_actuator_control 亮灯
子任务 3：假设目标管道位置 -> 发布 /target -> 发布 /auv_actuator_control 打开夹爪
```

后续如果要把三个子任务串成完整 Task 3 状态机，推荐数据流是：

```text
箭头识别结果
  -> 子任务 1 到达任务获取区
  -> ArUco 识别结果
  -> 子任务 2 得到 target_color 并亮灯
  -> 管道区域识别结果
  -> 子任务 3 根据 target_color 选择目标区域并投放信标球
```

也就是说，后续真正需要从模型接入的数据主要有三个：

- 箭头相对位置：用于替换子任务 1 的 `arrow_forward / arrow_right / arrow_down`。
- ArUco 编号：用于替换子任务 2 的 `mock_aruco_ids`。
- 管道目标位姿：用于替换子任务 3 的 `drop_forward / drop_left / drop_down` 或直接发布 `PoseStamped`。

## 常用测试命令

默认测试子任务 1：

```bash
rosrun auv_control test_task3_1_acquire_area.py
```

修改箭头假设位置：

```bash
rosrun auv_control test_task3_1_acquire_area.py _arrow_forward:=0.6 _arrow_right:=0.2
```

默认测试子任务 2：

```bash
rosrun auv_control test_task3_2_get_task.py
```

用真实 ArUco topic 测试子任务 2：

```bash
rosrun auv_control test_task3_2_get_task.py _input_mode:=topic _aruco_topic:=/task3/aruco_id
```

默认测试子任务 3：

```bash
rosrun auv_control test_task3_3_inspect_and_drop.py
```

修改投放颜色和投放相对位置：

```bash
rosrun auv_control test_task3_3_inspect_and_drop.py _target_color:=green _drop_forward:=0.5 _drop_left:=0.3
```

用真实管道目标位姿测试子任务 3：

```bash
rosrun auv_control test_task3_3_inspect_and_drop.py _target_mode:=topic _target_topic:=/task3/pipeline_target _target_color:=red
```
