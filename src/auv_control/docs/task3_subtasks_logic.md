# Task 3 三个测试子任务实现逻辑

本文档说明 `src/auv_control/test` 目录下 Task 3 三个测试子任务脚本的当前实现逻辑。

本次代码已经把真实感知模型输出接入到三个子任务里，同时保留原来的 mock 模式。这样模型、相机、TF 还没完全跑通时，可以继续用 mock 参数测试底盘运动和外设动作；等识别模型启动后，只需要切换 launch 参数，就可以读取真实识别结果。

相关脚本：

- `test_task3_1_acquire_area.py`：到达任务获取区
- `test_task3_2_get_task.py`：读取 ArUco 任务编号并亮灯
- `test_task3_3_inspect_and_drop.py`：移动到指定颜色区域并打开夹爪投放信标球

相关感知话题：

- `/obj/target_message`：`auv_control/TargetDetection`，用于箭头、形状、彩色框等单目标三维位置结果
- `/aruco/pose`：`geometry_msgs/PoseStamped`，用于 ArUco 位姿
- `/task3/aruco_id`：`std_msgs/Int32`，本次新增，用于给任务 3 直接输出 ArUco 编号

## 公共约定

坐标系约定：

- `map`：全局坐标系，底层控制器最终接收的目标位姿在这个坐标系下。
- `base_link`：机器人自身坐标系。
- 当前项目约定 `base_link` 中 `x` 表示前方，`y` 表示右方，`z` 表示向下。
- 所以“左侧 0.30 m”在代码里对应 `y = -0.30`。

主要控制话题：

- `/target`：发布 `geometry_msgs/PoseStamped`，告诉底层控制器机器人要去哪里。
- `/finished`：发布 `std_msgs/String`，表示当前测试子任务完成。
- `/auv_actuator_control`：发布 `auv_control/ActuatorControl`，控制新接口下的红、黄、绿三色灯和夹爪舵机。

## 子任务 1：到达任务获取区

### 本次代码修改说明

本次给子任务 1 增加了真实箭头识别结果接入，并把“找不到箭头怎么办”的基础搜索状态机写进了测试脚本。

原来的代码只使用写死参数：

```text
arrow_forward = 0.50
arrow_right   = 0.30
arrow_down    = 0.00
```

现在新增了 `input_mode`：

- `mock`：继续使用上面的固定偏移量。
- `topic`：订阅 `/obj/target_message`，搜索并锁定 `class_name == "arrow"` 的识别结果。

现在 `topic` 模式不再使用“识别到箭头后前方 0.50 m、右方 0.30 m”的固定偏移。识别稳定后，脚本会把箭头识别位姿转换到 `map` 坐标系，并把箭头所在的 90×90 cm 框区域作为目标点。

当前 `/obj/target_message` 只保证有目标三维位置，不保证有箭头朝向。因此默认：

```text
arrow_yaw_mode = current
```

也就是移动到箭头位置后，航向先保持当前方向。等识别模型后续把箭头方向写入 `pose.orientation` 后，可以改成：

```text
arrow_yaw_mode = detection
```

这样机器人会尝试把自身姿态调整到和箭头方向一致。

对应修改文件：

- `src/auv_control/test/test_task3_1_acquire_area.py`
- `src/auv_control/launch/task3_subtask1_acquire_area.launch`

### 感知来源

箭头识别不是单独的 `arrow_detection.py`，而是形状识别模型中的一个类别。

启动形状识别：

```bash
roslaunch stereo_depth test_shapes_detection.launch
```

形状识别模型输出：

```text
topic: /obj/target_message
type:  auv_control/TargetDetection
class_name: arrow
```

其中 `TargetDetection` 的主要字段是：

```text
pose       目标在 camera 坐标系下的三维位置
conf       置信度
type       一般为 center
class_name 目标类别，例如 arrow
```

### 子任务运行方式

mock 模式：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch
```

真实箭头识别模式：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch input_mode:=topic
```

如果后续识别模型已经输出箭头方向，可以这样启动：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch input_mode:=topic arrow_yaw_mode:=detection
```

### 实现流程

1. 节点启动后创建 `/target` 和 `/finished` 发布器，并创建 TF 监听器。
2. 如果是 `mock` 模式，直接使用 `arrow_forward / arrow_right / arrow_down` 生成 `base_link` 下的局部目标点。
3. 如果是 `topic` 模式，先根据启动时的 `base_link` 生成一组搜索点。
4. 每个搜索点都会尝试多个航向角，例如 `0、+30、-30、+60、-60、+90、-90` 度。
5. 每个航向角停留 `scan_hold_seconds` 秒，等待识别模型输出箭头。
6. 只接收 `class_name == "arrow"` 且置信度大于 `min_confidence` 的结果。
7. 为了避免误检，脚本要求连续 `stable_detection_count` 次识别位置稳定，位置抖动小于 `stable_position_tolerance`。
8. 稳定锁定后，把箭头识别位姿转换到 `map` 坐标系。
9. 将这个 `map` 下的箭头/框位置作为 `/target` 发布。
10. 如果 `arrow_yaw_mode=current`，目标航向保持当前航向；如果 `arrow_yaw_mode=detection`，后续模型提供真实方向后可直接使用识别方向。
11. 到达箭头所在框后保持 `hold_seconds` 秒。
12. 发布 `/finished`，结束节点。

如果所有搜索点和搜索角度都没有找到箭头，或者超过 `max_search_seconds`，脚本会发布失败信息并停止，不会继续假装完成。

### 当前搜索参数

默认搜索参数：

```text
search_step = 0.30
max_search_points = 9
scan_yaw_offsets_deg = 0 30 -30 60 -60 90 -90
scan_hold_seconds = 1.5
max_search_seconds = 60.0
stable_detection_count = 5
stable_position_tolerance = 0.15
```

含义：

- `search_step`：相邻搜索点距离，默认 0.30 m，小于 90×90 cm 框尺寸，比较适合先做保守搜索。
- `max_search_points`：最多搜索几个位置，默认 9 个，类似九宫格。
- `scan_yaw_offsets_deg`：每个位置原地转哪些角度看箭头。
- `stable_detection_count`：连续多少帧稳定识别才算真正找到箭头。
- `stable_position_tolerance`：这些识别点之间最大允许抖动距离。
- `max_search_seconds`：超过这个时间仍找不到箭头，就判定失败。

## 子任务 2：读取 ArUco 编号并亮灯

### 本次代码修改说明

本次给 ArUco 感知节点补了任务 3 能直接使用的编号话题，并把子任务 2 明确改成“只识别标志并亮灯”的测试节点。

这个子任务现在不做移动，也不发布 `/target`。它只做两件事：

```text
读取 ArUco 编号
根据编号点亮对应颜色灯
```

原来的 ArUco 节点主要发布：

```text
/aruco/pose
```

但任务 3 第二个子任务需要的是 ArUco 编号 `1~6`，而不是只要位姿。因此本次新增：

```text
topic: /task3/aruco_id
type:  std_msgs/Int32
data:  ArUco 编号
```

同时保留 `/aruco/pose` 不变。

本次还处理了 `aruco_detection260708.py` 里不标准的 `pose_msg.header.class_id` 写法。标准 `PoseStamped.header` 没有 `class_id` 字段，所以现在改为单独发布 `/task3/aruco_id`。

对应修改文件：

- `src/auv_control/test/test_task3_2_get_task.py`
- `src/auv_control/launch/task3_subtask2_get_task.launch`
- `src/stereo_depth/scripts/aruco_detection.py`
- `src/stereo_depth/scripts/aruco_detection260708.py`
- `src/stereo_depth/launch/find_aruco.launch`
- `src/stereo_depth/launch/test_aruco_detection260708.launch`
- `src/stereo_depth/launch/test_aruco_opencv.launch`

### 感知来源

启动 ArUco 识别：

```bash
roslaunch stereo_depth test_aruco_detection260708.launch
```

输出：

```text
/aruco/pose
/task3/aruco_id
```

子任务 2 只需要读取：

```text
topic: /task3/aruco_id
type:  std_msgs/Int32
```

### 编号和颜色映射

Task 3 规则：

| ArUco 编号 | 目标颜色 |
| --- | --- |
| `1` / `2` | 黄色 |
| `3` / `4` | 绿色 |
| `5` / `6` | 红色 |

对应 `/auv_actuator_control`：

| 颜色 | `red_light` | `yellow_light` | `green_light` |
| --- | --- | --- | --- |
| 黄色 | `0` | `1` | `0` |
| 绿色 | `0` | `0` | `1` |
| 红色 | `1` | `0` | `0` |
| 熄灭 | `0` | `0` | `0` |

### 子任务运行方式

真实 ArUco 编号模式，也是当前默认模式：

```bash
roslaunch auv_control task3_subtask2_get_task.launch
```

如果要离线模拟 ArUco 编号，可以切换到 mock 模式：

```bash
roslaunch auv_control task3_subtask2_get_task.launch input_mode:=mock
```

默认每次亮灯 `3` 秒：

```text
light_seconds = 3.0
```

默认识别到一个真实 ArUco 编号后就完成测试：

```text
max_topic_markers = 1
```

如果想让它一直识别、一直根据新编号亮灯，可以设置：

```bash
roslaunch auv_control task3_subtask2_get_task.launch max_topic_markers:=0
```

### 实现流程

1. 节点启动后读取 `input_mode`。
2. 默认 `topic` 模式，订阅 `/task3/aruco_id`，等待真实 ArUco 编号。
3. 如果切到 `mock` 模式，按 `mock_aruco_ids` 里的固定序列模拟识别编号。
4. 收到 ArUco 编号后，根据 `1/2 -> yellow`、`3/4 -> green`、`5/6 -> red` 得到目标颜色。
5. 通过 `/auv_actuator_control` 点亮对应颜色灯。
6. 每次亮灯保持 `light_seconds` 秒。
7. 灯熄灭 `gap_seconds` 秒。
8. 如果收到的编号不是 `1~6`，脚本会忽略这个编号，不会乱亮灯。
9. mock 序列完成，或 topic 模式达到 `max_topic_markers` 次后，发布 `/finished`。

## 子任务 3：检查指定颜色区域并投放信标球

### 本次代码修改说明

本次给子任务 3 增加了真实彩色框识别结果接入，并把投放逻辑改成“粗略靠近 + 视觉微调 + 停稳投放”的状态机。

原来的代码只使用 mock 假设：

```text
目标区域在机器人前方 0.50 m、左侧 0.30 m
```

现在新增了 `target_mode=detection`：

- `mock`：继续使用固定偏移量。
- `topic`：保留旧接口，读取外部发布的 `geometry_msgs/PoseStamped`。
- `detection`：直接订阅 `/obj/target_message`，读取彩色框识别结果，并在靠近后继续用视觉误差做小步微调。

在 `detection` 模式下，子任务会按照 `target_color` 过滤识别结果：

```text
target_color = yellow / green / red
```

也就是说，如果 `target_color=yellow`，脚本只接受 `class_name == "yellow"` 的彩色框识别结果。

这个子任务现在不会“识别到一次就直接开夹爪”。真实识别模式下会先要求连续稳定识别，然后粗略移动到方框中心附近，再根据方框在 `base_link` 下的前后、左右误差进行小步修正，确认机器人稳定在方框上方后才打开夹爪投放。

对应修改文件：

- `src/auv_control/test/test_task3_3_inspect_and_drop.py`
- `src/auv_control/launch/task3_subtask3_inspect_and_drop.launch`

### 感知来源

启动彩色框识别：

```bash
roslaunch stereo_depth test_rectangles_detection.launch
```

彩色框识别输出：

```text
topic: /obj/target_message
type:  auv_control/TargetDetection
class_name: red / green / yellow
```

### 子任务运行方式

mock 模式：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch
```

真实彩色框识别模式，例如目标颜色是黄色：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch target_mode:=detection target_color:=yellow
```

目标颜色是红色：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch target_mode:=detection target_color:=red
```

### 实现流程

1. 节点启动后创建 `/target`、`/finished`、`/auv_actuator_control` 发布器，并创建 TF 监听器。
2. 根据 `target_mode` 决定目标来源。
3. 如果是 `mock` 模式，使用 `drop_forward / drop_left / drop_down` 构造投放目标。
4. 如果是 `topic` 模式，订阅 `target_topic`，等待外部发布 `PoseStamped`。
5. 如果是 `detection` 模式，订阅 `/obj/target_message`，只接收 `class_name == target_color` 且置信度大于 `min_confidence` 的识别结果。
6. 为了避免误检，脚本要求连续 `stable_detection_count` 次识别位置稳定，位置抖动小于 `stable_position_tolerance`。
7. 稳定锁定后，把彩色框中心位姿转换到 `map` 坐标系，作为粗略靠近目标。
8. 机器人发布 `/target`，先移动到彩色框附近。
9. 到达粗略目标后进入视觉微调状态，不再只相信第一次识别位置。
10. 微调时把实时彩色框位姿转换到 `base_link` 坐标系，读取前后误差 `x` 和左右误差 `y`。
11. 如果 `x/y` 误差大于容许范围，就按 `fine_min_step ~ fine_max_step` 生成一个小步目标继续修正。
12. 如果 `x/y` 误差都小于阈值，并持续稳定 `fine_hold_seconds` 秒，认为机器人已经在方框上方。
13. 保持当前位置 `hold_seconds` 秒。
14. 发布夹爪打开值 `clamp_open`，保持 `open_seconds` 秒，完成投放。
15. 发布夹爪关闭值 `clamp_closed`，保持 `close_seconds` 秒。
16. 熄灯并发布 `/finished`，结束节点。

如果微调过程中短时间丢失彩色框，机器人会先保持当前目标继续等待；如果丢失超过 `frame_lost_timeout`，脚本会回到“重新稳定锁定彩色框”的状态。

### 当前可调参数

默认识别和粗定位参数：

```text
stable_detection_count = 5
stable_position_tolerance = 0.15
detection_timeout = 2.0
max_detection_wait_seconds = 60.0
coarse_arrive_dist = 0.18
coarse_arrive_yaw_deg = 8.0
```

默认微调参数：

```text
fine_tolerance_x = 0.08
fine_tolerance_y = 0.08
fine_max_step = 0.10
fine_min_step = 0.03
fine_gain = 0.8
fine_command_period = 0.4
fine_hold_seconds = 1.0
frame_lost_timeout = 2.0
```

含义：

- `fine_tolerance_x`：方框中心在机器人前后方向的允许误差，默认 0.08 m。
- `fine_tolerance_y`：方框中心在机器人左右方向的允许误差，默认 0.08 m。
- `fine_max_step`：每次微调最多移动多少米，默认 0.10 m，避免冲过头。
- `fine_min_step`：只要还没对准，每次至少移动多少米，默认 0.03 m，避免控制器收到太小的目标。
- `fine_gain`：误差到移动步长的比例系数。
- `fine_command_period`：多久重新根据视觉误差生成一次微调目标。
- `fine_hold_seconds`：连续对准多久才允许打开夹爪。
- `frame_lost_timeout`：微调中丢失目标多久后回到重新锁定状态。

## 三个子任务和识别模型的对应关系

```text
形状识别 test_shapes_detection.launch
  -> /obj/target_message class_name=arrow
  -> 子任务 1 到达任务获取区

ArUco 识别 test_aruco_detection260708.launch
  -> /task3/aruco_id
  -> 子任务 2 根据编号亮灯

彩色框识别 test_rectangles_detection.launch
  -> /obj/target_message class_name=yellow/green/red
  -> 子任务 3 移动到对应颜色区域并投放信标球
```

## 常用测试命令

启动箭头识别：

```bash
roslaunch stereo_depth test_shapes_detection.launch
```

启动子任务 1 真实识别模式：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch input_mode:=topic
```

启动 ArUco 识别：

```bash
roslaunch stereo_depth test_aruco_detection260708.launch
```

启动子任务 2 真实识别模式：

```bash
roslaunch auv_control task3_subtask2_get_task.launch input_mode:=topic
```

启动彩色框识别：

```bash
roslaunch stereo_depth test_rectangles_detection.launch
```

启动子任务 3 真实识别模式：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch target_mode:=detection target_color:=yellow
```

## 注意事项

1. 真实识别模式依赖 TF，尤其是 `camera` 到 `map` 或相关坐标系的转换。如果 TF 不完整，脚本会收到识别结果，但无法转换成 `/target` 所需的 `map` 坐标。
2. 子任务 1 在 `topic` 模式下会直接移动到识别到的箭头/90×90 cm 框位置；`mock` 模式才继续使用固定偏移参数。
3. 子任务 3 的 `target_color` 应该来自子任务 2 的 ArUco 编号结果；当前三个脚本还是独立测试脚本，所以需要手动通过 launch 参数指定。
4. mock 模式仍然保留，方便在没有相机、没有模型、没有 TF 的情况下先验证底盘控制和外设动作。
