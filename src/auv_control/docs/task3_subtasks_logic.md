# Task 3 三个测试子任务实现逻辑

更新：2026.7.14

本文档说明 `src/auv_control/test` 目录下 Task 3 三个测试子任务脚本的逻辑、实现步骤、可调参数和代码运行方式。

本次 Task 3 的测试脚本已经接入最新控制话题：

- 运动控制默认发布到 `/cmd/pose/ned`，消息类型为 `auv_control/PoseNEDcmd`。
- 旧版 `/target` 仍作为兼容调试接口保留，可通过 `motion_output=target` 或 `motion_output=both` 启用。
- 执行器控制发布到 `/cmd/actuator`，消息类型为 `auv_control/ActuatorControl`。

相关脚本：

| 子任务 | 脚本 | 功能 |
| --- | --- | --- |
| 子任务 1 | `test_task3_1_acquire_area.py` | 搜索箭头并到达任务获取区 |
| 子任务 2 | `test_task3_2_get_task.py` | 读取 ArUco 编号并点亮对应颜色灯 |
| 子任务 3 | `test_task3_3_inspect_and_drop.py` | 移动到指定颜色区域并投放信标球 |

## 公共约定

坐标系约定：

- `map`：全局坐标系，任务目标最终会转换到这个坐标系下。
- `base_link`：机器人自身坐标系。
- 当前项目约定 `base_link` 中 `x` 为前方，`y` 为右方，`z` 为向下。
- 因此“左侧 0.30 m”在代码中对应 `y = -0.30`。

主要话题：

| 话题 | 消息类型 | 用途 |
| --- | --- | --- |
| `/obj/target_message` | `auv_control/TargetDetection` | 箭头、彩色框等视觉目标的三维识别结果 |
| `/task3/aruco_id` | `std_msgs/Int32` | Task 3 直接使用的 ArUco 编号 |
| `/cmd/pose/ned` | `auv_control/PoseNEDcmd` | 最新运动控制整包指令 |
| `/target` | `geometry_msgs/PoseStamped` | 旧运动目标话题，保留用于兼容调试 |
| `/cmd/actuator` | `auv_control/ActuatorControl` | 三色灯和夹爪舵机控制 |
| `/finished` | `std_msgs/String` | 当前测试子任务完成或失败状态 |

子任务 1 和子任务 3 的默认运动输出：

```text
motion_output = cmd
pose_cmd_topic = /cmd/pose/ned
pose_cmd_mode = 4
pose_cmd_force = 0 0 0 0 0 0
```

`pose_cmd_mode=4` 对应最新驱动里的动力定位/定点模式。调试旧链路时可以把 `motion_output` 改成 `target` 或 `both`。

## 子任务 1：到达任务获取区

### 任务逻辑

子任务 1 的目标是找到前方地面箭头，并让机器人靠近箭头中心进入识别区。根据当前场地判断，箭头一定在机器人前方，只是不确定具体在前方偏左、偏右还是更远的位置，所以真实识别模式只搜索前方区域，不再向后搜索。

稳定识别到箭头后，脚本不会把箭头中心直接作为机器人中心目标点，而是根据箭头中心位置和箭头航向生成一个靠近目标：机器人保持和箭头航向一致，并让箭头中心落在机器人前方约 `0.30 m`，允许前后误差约 `0.10 m`，左右误差约 `0.10 m`。

高度上，箭头中心被当作地面点处理。运动目标会保持机器人对地最小距离 `min_ground_clearance = 0.40 m`。当前控制链路使用 NED 语义，`z` 向下，所以如果当前深度距离地面不足 40 cm，脚本会把目标深度限制到 `ground_z - 0.40`，避免继续贴近地面。

该子任务支持两种输入模式：

| 模式 | 参数 | 说明 |
| --- | --- | --- |
| mock | `input_mode=mock` | 不依赖相机，使用固定 `base_link` 偏移生成目标点 |
| topic | `input_mode=topic` | 订阅 `/obj/target_message`，搜索并锁定 `class_name == "arrow"` 的真实识别结果 |

真实识别模式下，脚本会先生成一组前方搜索点，并在每个搜索点尝试多个航向角。如果识别结果连续稳定，才会锁定箭头位置。这样可以避免偶然误检导致机器人直接跑偏。

当前靠近判定要求机器人航向和箭头航向一致，因此默认航向策略是：

```text
arrow_yaw_mode = detection
```

也就是使用 `TargetDetection.pose.orientation` 作为箭头方向。如果现场调试时视觉还没有稳定输出箭头航向，可以临时改成：

```text
arrow_yaw_mode = current
```

当前模型暂时还不能稳定识别箭头方向，但代码仍然按“后续模型会输出箭头方向”的逻辑实现；等模型把方向写进 `pose.orientation` 后，不需要再改控制逻辑。

### 实现步骤

1. 节点启动后创建 `/cmd/pose/ned`、兼容 `/target`、`/finished` 发布器，并创建 TF 监听器。
2. 读取 `input_mode`，决定使用 mock 固定点还是 topic 真实识别。
3. 如果是 `mock` 模式，使用 `arrow_forward / arrow_right / arrow_down` 在 `base_link` 下构造局部目标点，再转换到 `map`。
4. 如果是 `topic` 模式，根据启动时的 `base_link` 生成一组前方搜索点。
5. 搜索点顺序为原地、前方 0.30 m 一排、前方 0.60 m 一排、前方 0.90 m 一排，不包含任何后退点。
6. 每个搜索点都会尝试多个航向角，例如 `0、+20、-20、+40、-40、+60、-60` 度。
7. 每个航向角停留 `scan_hold_seconds` 秒，等待视觉模型输出箭头。
8. 只接收 `class_name == "arrow"` 且置信度大于 `min_confidence` 的识别结果。
9. 要求连续 `stable_detection_count` 次识别稳定，且位置抖动小于 `stable_position_tolerance`。
10. 稳定后把箭头识别位姿转换到 `map` 坐标系，并根据 `arrow_yaw_mode` 得到箭头航向。
11. 根据箭头中心和箭头航向生成靠近目标：目标点在箭头中心后方 `approach_distance` 处，机器人朝向和箭头航向一致。
12. 根据箭头中心的地面高度限制目标深度，保证对地距离不小于 `min_ground_clearance`。
13. 默认通过 `/cmd/pose/ned` 发布目标；如果 `motion_output=target` 或 `both`，同时发布兼容 `/target`。
14. 靠近过程中持续读取稳定箭头识别结果，并刷新靠近目标。
15. 当箭头中心在 `base_link` 前方约 `approach_distance`，左右偏差小于 `approach_lateral_tolerance`，且航向误差小于 `approach_yaw_tolerance_deg` 时，认为进入识别区。
16. 上述条件持续保持 `hold_seconds` 秒后，发布 `/finished` 并结束节点。

如果一整轮前方搜索点和扫描角度都走完仍未找到箭头，但累计搜索时间还没有超过 `max_search_seconds`，脚本会回到第一搜索点重新开始下一轮搜索。只有“一整轮搜索走完仍未找到箭头”并且累计搜索时间已经超过 `max_search_seconds` 时，才会发布失败信息并退出节点。

### 可调参数

基础输入参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input_mode` | `mock` | 输入模式，支持 `mock` 或 `topic` |
| `arrow_topic` | `/obj/target_message` | 箭头识别结果话题 |
| `arrow_class` | `arrow` | 需要匹配的识别类别 |
| `detection_frame` | `camera` | 识别结果缺少坐标系时使用的默认坐标系 |
| `min_confidence` | `0.35` | 最低置信度 |

mock 目标参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `arrow_forward` | `0.50` | 目标在机器人前方的距离 |
| `arrow_right` | `0.30` | 目标在机器人右侧的距离 |
| `arrow_down` | `0.00` | 目标在机器人下方的距离 |

搜索和稳定锁定参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `search_step` | `0.30` | 前方搜索点间距 |
| `max_search_points` | `14` | 最多搜索几个前方位置 |
| `scan_yaw_offsets_deg` | `0 20 -20 40 -40 60 -60` | 每个位置尝试的航向角 |
| `scan_hold_seconds` | `1.5` | 每个航向停留等待识别的时间 |
| `max_search_seconds` | `300.0` | 最大搜索时间，默认 5 分钟；必须完整走完搜索表且超过该时间才失败退出 |
| `stable_detection_count` | `5` | 连续多少次稳定识别才锁定 |
| `stable_position_tolerance` | `0.15` | 稳定识别允许的最大位置抖动 |
| `detection_timeout` | `2.0` | 识别结果过期时间 |

运动控制参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `motion_output` | `cmd` | `cmd` 发布 `/cmd/pose/ned`，`target` 发布 `/target`，`both` 同时发布 |
| `legacy_target_topic` | `/target` | 旧运动目标话题 |
| `pose_cmd_topic` | `/cmd/pose/ned` | 新运动控制话题 |
| `pose_cmd_mode` | `4` | PoseNEDcmd 控制模式 |
| `pose_cmd_force` | `0 0 0 0 0 0` | 六自由度力/力矩字段 |
| `arrive_dist` | `0.25` | 到达目标的位置容差 |
| `arrive_yaw_deg` | `8.0` | 到达目标的航向容差 |
| `min_ground_clearance` | `0.40` | 对地最小距离，NED 下目标深度不会深过 `ground_z - 0.40` |
| `hold_seconds` | `2.0` | 到达后保持时间 |

靠近箭头参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `approach_distance` | `0.30` | 进入识别区时箭头中心应在机器人前方的距离 |
| `approach_distance_tolerance` | `0.10` | 前后距离允许误差 |
| `approach_lateral_tolerance` | `0.10` | 左右偏差允许误差 |
| `approach_yaw_tolerance_deg` | `10.0` | 机器人航向和箭头航向允许误差 |

航向参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `arrow_yaw_mode` | `detection` | `detection` 使用识别航向，`current` 保持当前航向，`fixed` 使用固定航向 |
| `fixed_arrow_yaw_deg` | `0.0` | `arrow_yaw_mode=fixed` 时使用的航向角 |

## 子任务 2：读取 ArUco 编号并亮灯

### 任务逻辑

子任务 2 的目标是读取任务板上的 ArUco 编号，并根据编号点亮对应颜色灯。这个子任务只负责识别和亮灯，不控制机器人移动，也不发布 `/cmd/pose/ned` 或 `/target`。

ArUco 编号到颜色的规则：

| ArUco 编号 | 目标颜色 |
| --- | --- |
| `1` / `2` | yellow |
| `3` / `4` | green |
| `5` / `6` | red |

该子任务支持两种输入模式：

| 模式 | 参数 | 说明 |
| --- | --- | --- |
| topic | `input_mode=topic` | 默认模式，订阅 `/task3/aruco_id` 读取真实编号 |
| mock | `input_mode=mock` | 按 `mock_aruco_ids` 模拟编号序列，方便离线测试灯光 |

亮灯使用 `/cmd/actuator` 的三个独立灯字段：

| 颜色 | `red_light` | `yellow_light` | `green_light` |
| --- | --- | --- | --- |
| yellow | `0` | `1` | `0` |
| green | `0` | `0` | `1` |
| red | `1` | `0` | `0` |
| off | `0` | `0` | `0` |

### 实现步骤

1. 节点启动后创建 `/cmd/actuator` 和 `/finished` 发布器。
2. 读取 `input_mode`，决定读取真实 ArUco 话题还是 mock 编号序列。
3. 如果是 `topic` 模式，订阅 `/task3/aruco_id`，消息类型为 `std_msgs/Int32`。
4. 如果是 `mock` 模式，按 `mock_aruco_ids` 中的编号依次模拟识别结果。
5. 读取到编号后，要求最近 `stable_marker_count` 次编号一致，避免偶然跳变。
6. 根据 `1/2 -> yellow`、`3/4 -> green`、`5/6 -> red` 得到目标颜色。
7. 如果编号不在 `1~6` 范围内，忽略该编号，不亮灯。
8. 通过 `/cmd/actuator` 点亮对应颜色灯，保持 `light_seconds` 秒。
9. 灭灯并保持 `gap_seconds` 秒。
10. mock 序列结束，或 topic 模式处理数量达到 `max_topic_markers` 后，发布 `/finished`。

### 可调参数

基础输入参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input_mode` | `topic` | 输入模式，支持 `topic` 或 `mock` |
| `aruco_topic` | `/task3/aruco_id` | 真实 ArUco 编号话题 |
| `mock_aruco_ids` | `[1, 3, 5, 2, 4, 6]` | mock 模式下的编号序列 |
| `max_topic_markers` | `1` | topic 模式下处理几个真实编号后结束，`0` 表示一直运行 |
| `stable_marker_count` | `1` | 连续多少次相同编号才认为稳定 |
| `marker_timeout` | `1.0` | 编号样本过期时间 |

亮灯参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `light_seconds` | `3.0` | 每次亮灯持续时间 |
| `gap_seconds` | `0.5` | 灭灯间隔 |
| `actuator_topic` | `/cmd/actuator` | 执行器控制话题 |
| `light1` | `0` | 补光灯 1 亮度 |
| `light2` | `0` | 补光灯 2 亮度 |

执行器保持参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `heading_servo` | `128` | 航向舵机保持值 |
| `clamp_servo` | `255` | 夹爪舵机保持值 |
| `drive_cmd` | `0` | 推杆电机动作 |
| `drive_speed` | `0` | 推杆电机速度 |

## 子任务 3：检查指定颜色区域并投放信标球

### 任务逻辑

子任务 3 的目标是识别指定颜色的方框，逐步靠近到方框中心附近，并打开夹爪投放信标球。目标颜色后续可以来自子任务 2 识别到的 ArUco 编号，但当前为了调试稳定，先在 launch 参数里人为固定 `target_color`；该颜色可以随时改成 `yellow`、`green`、`red`，也可以临时改成模型实际输出的其他标签。

该子任务支持三种目标模式：

| 模式 | 参数 | 说明 |
| --- | --- | --- |
| mock | `target_mode=mock` | 使用固定 `base_link` 偏移作为投放目标 |
| topic | `target_mode=topic` | 读取外部发布的 `geometry_msgs/PoseStamped` |
| detection | `target_mode=detection` | 订阅 `/obj/target_message`，按 `target_color` 过滤彩色框识别结果 |

真实识别模式不会“识别到一次就直接开夹爪”。寻找彩色方框的逻辑沿用子任务 1 的前方搜索逻辑：机器人只搜索前方区域，在每个搜索点尝试多个航向角，稳定识别到指定颜色方框后再进入靠近。完整状态机如下：

```text
前方搜索指定颜色框
  -> 每个搜索点扫描多个航向角
  -> 稳定锁定指定颜色框
  -> 按实时中心误差分步靠近
  -> 进入粗容差后继续细对齐
  -> 停稳
  -> 打开夹爪投放
  -> 关闭夹爪并结束
```

粗靠近和细对齐阶段都会把实时彩色框位姿转换到 `base_link`，读取前后误差 `x` 和左右误差 `y`。目标不是“识别到一次就去那个固定点”，而是持续根据最新视觉中心位置生成小步移动目标。只有当方框中心连续保持在细对齐容差内，才允许投放。

### 实现步骤

1. 节点启动后创建 `/cmd/pose/ned`、兼容 `/target`、`/finished`、`/cmd/actuator` 发布器，并创建 TF 监听器。
2. 读取 `target_mode`，决定目标来源。
3. 如果是 `mock` 模式，使用 `drop_forward / drop_left / drop_down` 在 `base_link` 下构造投放目标。
4. 如果是 `topic` 模式，订阅 `target_topic`，等待外部发布 `PoseStamped`。
5. 如果是 `detection` 模式，订阅 `/obj/target_message`，只接收 `class_name == target_color` 且置信度大于 `min_confidence` 的结果。
6. detection 模式下，根据启动时的 `base_link` 生成和子任务 1 一致的前方搜索点，不向后搜索。
7. 搜索点顺序为原地、前方 0.30 m 一排、前方 0.60 m 一排、前方 0.90 m 一排。
8. 每个搜索点都会尝试多个航向角，例如 `0、+20、-20、+40、-40、+60、-60` 度。
9. 到达搜索位姿后保持 `scan_hold_seconds` 秒，等待指定颜色方框识别。
10. 要求连续 `stable_detection_count` 次识别稳定，且位置抖动小于 `stable_position_tolerance`。
11. 稳定锁定后，将彩色框中心转换到 `base_link`，计算中心相对期望位置 `center_target_forward / center_target_right` 的误差。
12. 粗靠近阶段按 `coarse_gain`、`coarse_min_step`、`coarse_max_step` 生成小步修正目标，并默认通过 `/cmd/pose/ned` 发布；如果 `motion_output=target` 或 `both`，同时发布兼容 `/target`。
13. 粗靠近期间持续读取新的稳定识别结果，按 `coarse_command_period` 刷新小步目标。
14. 当中心误差进入 `coarse_center_tolerance_x / coarse_center_tolerance_y` 后，进入细对齐状态。
15. 细对齐时使用最新彩色框位姿，按 `fine_gain`、`fine_min_step`、`fine_max_step` 继续生成更小的修正目标。
16. 如果误差进入 `fine_tolerance_x / fine_tolerance_y` 范围，并持续 `fine_hold_seconds` 秒，认为机器人已经稳定在彩色框中心附近。
17. 保持当前目标 `hold_seconds` 秒。
18. 发布 `clamp_open`，保持 `open_seconds` 秒，完成投放。
19. 发布 `clamp_closed`，保持 `close_seconds` 秒。
20. 熄灭指示灯，发布 `/finished` 并结束节点。

如果一整轮前方搜索点和扫描角度都走完仍未找到彩色方框，但累计搜索时间还没有超过 `max_search_seconds`，脚本会回到第一搜索点重新开始下一轮搜索。只有“一整轮搜索走完仍未找到方框”并且累计搜索时间已经超过 `max_search_seconds` 时，才会发布失败信息并退出节点。微调阶段短时间丢失彩色框时，机器人会继续保持当前目标等待视觉恢复；如果丢失超过 `frame_lost_timeout`，脚本会回到搜索状态。

### 可调参数

基础输入参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `target_mode` | `detection` | 目标模式，支持 `mock`、`topic`、`detection`；真实任务默认走识别 |
| `target_topic` | `/task3/pipeline_target` | topic 模式下外部目标话题 |
| `detection_topic` | `/obj/target_message` | detection 模式下彩色框识别话题 |
| `detection_frame` | `camera` | 识别结果缺少 frame_id 时使用的默认坐标系 |
| `target_color` | `yellow` | 人为固定的目标颜色；通常为 `yellow`、`green`、`red`，也可临时改成模型实际输出标签 |
| `min_confidence` | `0.2` | 最低识别置信度 |
| `mock_detected_colors` | `yellow green red` | mock 模式下认为可检测到的颜色 |
| `debug_log_detections` | `true` | 是否打印识别类别、置信度、坐标系和识别位置 |
| `debug_log_targets` | `true` | 是否打印当前发布的运动目标 |

mock 目标参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `drop_forward` | `0.50` | 投放目标在机器人前方的距离 |
| `drop_left` | `0.30` | 投放目标在机器人左侧的距离，代码中会转成负 `y` |
| `drop_down` | `0.00` | 投放目标在机器人下方的距离 |

搜索和稳定锁定参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `search_step` | `0.30` | 前方搜索点间距 |
| `max_search_points` | `14` | 最多搜索几个前方位置 |
| `scan_yaw_offsets_deg` | `0 20 -20 40 -40 60 -60` | 每个位置尝试的航向角 |
| `scan_hold_seconds` | `1.5` | 每个航向停留等待识别的时间 |
| `max_search_seconds` | `300.0` | 最大搜索时间，默认 5 分钟；必须完整走完搜索表且超过该时间才失败退出 |
| `search_arrive_dist` | `0.15` | 到达搜索位姿的位置容差 |
| `search_arrive_yaw_deg` | `8.0` | 到达搜索位姿的航向容差 |
| `stable_detection_count` | `5` | 连续多少次稳定识别才锁定 |
| `stable_position_tolerance` | `0.15` | 稳定识别允许的位置抖动 |
| `detection_timeout` | `2.0` | 识别结果过期时间 |

中心靠近参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `center_target_forward` | `0.00` | 投放确认时，方框中心期望落在 `base_link` 前方多少米 |
| `center_target_right` | `0.00` | 投放确认时，方框中心期望落在 `base_link` 右方多少米 |
| `coarse_center_tolerance_x` | `0.18` | 粗靠近阶段前后误差进入多少米后切到细对齐 |
| `coarse_center_tolerance_y` | `0.18` | 粗靠近阶段左右误差进入多少米后切到细对齐 |
| `coarse_max_step` | `0.20` | 粗靠近每次最大移动步长 |
| `coarse_min_step` | `0.05` | 粗靠近每次最小移动步长 |
| `coarse_gain` | `0.8` | 粗靠近误差到步长的比例系数 |
| `coarse_command_period` | `0.5` | 粗靠近多久重新生成一次小步目标 |

运动控制参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `motion_output` | `cmd` | `cmd` 发布 `/cmd/pose/ned`，`target` 发布 `/target`，`both` 同时发布 |
| `legacy_target_topic` | `/target` | 旧运动目标话题 |
| `pose_cmd_topic` | `/cmd/pose/ned` | 新运动控制话题 |
| `pose_cmd_mode` | `4` | PoseNEDcmd 控制模式 |
| `pose_cmd_force` | `0 0 0 0 0 0` | 六自由度力/力矩字段 |
| `arrive_dist` | `0.12` | mock/topic 简单模式到达距离容差 |
| `arrive_yaw_deg` | `5.0` | mock/topic 简单模式到达航向容差 |

视觉微调参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `fine_tolerance_x` | `0.08` | 前后方向允许误差 |
| `fine_tolerance_y` | `0.08` | 左右方向允许误差 |
| `fine_max_step` | `0.10` | 每次微调最大步长 |
| `fine_min_step` | `0.03` | 每次微调最小步长 |
| `fine_gain` | `0.8` | 误差到步长的比例系数 |
| `fine_command_period` | `0.4` | 多久重新生成一次微调目标 |
| `fine_hold_seconds` | `1.0` | 连续对准多久才允许投放 |
| `frame_lost_timeout` | `2.0` | 微调中目标丢失多久后重新锁定 |

投放和执行器参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `hold_seconds` | `1.0` | 投放前停稳时间 |
| `open_seconds` | `3.0` | 夹爪打开保持时间 |
| `close_seconds` | `1.0` | 夹爪关闭保持时间 |
| `actuator_topic` | `/cmd/actuator` | 执行器控制话题 |
| `clamp_open` | `0` | 夹爪打开舵机值 |
| `clamp_closed` | `255` | 夹爪关闭舵机值 |
| `heading_servo` | `128` | 航向舵机保持值 |
| `drive_cmd` | `0` | 推杆电机动作 |
| `drive_speed` | `0` | 推杆电机速度 |
| `light1` | `0` | 补光灯 1 亮度 |
| `light2` | `0` | 补光灯 2 亮度 |
| `show_color_light` | `true` | 投放过程中是否点亮目标颜色灯 |

## 代码操作步骤

### 1. 确认代码文件

Task 3 三个子任务主要对应以下文件：

```text
src/auv_control/test/test_task3_1_acquire_area.py
src/auv_control/test/test_task3_2_get_task.py
src/auv_control/test/test_task3_3_inspect_and_drop.py
src/auv_control/launch/task3_subtask1_acquire_area.launch
src/auv_control/launch/task3_subtask2_get_task.launch
src/auv_control/launch/task3_subtask3_inspect_and_drop.launch
src/auv_control/docs/task3_subtasks_logic.md
```

### 2. 编译工作空间

在机器人或 Linux ROS 环境中进入工作空间根目录后执行：

```bash
catkin_make_isolated \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/home/xhy/xhy_env/bin/python3.8
```

编译完成后加载环境：

```bash
source devel_isolated/setup.bash
```

### 3. 启动基础控制链路

先启动底层控制、TF、传感器和执行器节点：

```bash
roslaunch auv_control begin.launch
```

如果需要单独测试最新运动控制链路，可参考：

```bash
roslaunch auv_control test_debug_v2.launch
```

### 4. 子任务 1 操作

mock 模式，不依赖视觉：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch
```

真实箭头识别模式，先启动形状识别：

```bash
roslaunch stereo_depth test_shapes_detection.launch
```

再启动子任务 1：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch input_mode:=topic
```

如果要同时发布旧 `/target` 便于调试：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch input_mode:=topic motion_output:=both
```

### 5. 子任务 2 操作

真实 ArUco 模式，先启动 ArUco 识别：

```bash
roslaunch stereo_depth test_aruco_detection260708.launch
```

再启动子任务 2：

```bash
roslaunch auv_control task3_subtask2_get_task.launch
```

离线 mock 编号亮灯：

```bash
roslaunch auv_control task3_subtask2_get_task.launch input_mode:=mock
```

如果希望持续响应新的 ArUco 编号：

```bash
roslaunch auv_control task3_subtask2_get_task.launch max_topic_markers:=0
```

### 6. 子任务 3 操作

mock 模式，不依赖视觉：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch target_mode:=mock
```

真实彩色框识别模式，先启动彩色框识别：

```bash
roslaunch stereo_depth test_rectangles_detection.launch
```

再按目标颜色启动子任务 3：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch target_color:=yellow
```

其他颜色示例：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch target_mode:=detection target_color:=green
roslaunch auv_control task3_subtask3_inspect_and_drop.launch target_mode:=detection target_color:=red
```

如果要同时发布旧 `/target` 便于调试：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch target_mode:=detection target_color:=yellow motion_output:=both
```

### 7. 常用检查命令

查看运动控制输出：

```bash
rostopic echo /cmd/pose/ned
```

查看执行器输出：

```bash
rostopic echo /cmd/actuator
```

查看任务完成状态：

```bash
rostopic echo /finished
```

查看视觉识别输出：

```bash
rostopic echo /obj/target_message
```

查看 ArUco 编号输出：

```bash
rostopic echo /task3/aruco_id
```

### 8. 注意事项

1. 真实识别模式依赖 TF，尤其是 `camera`、`base_link`、`map` 之间的转换。如果 TF 不完整，脚本可以收到识别结果，但无法生成可用的 `map` 坐标目标。
2. 子任务 1 在 `topic` 模式下只搜索前方区域，并在稳定识别箭头后靠近到箭头中心前方约 0.30 m；`mock` 模式才使用固定偏移参数。
3. 子任务 3 的 `target_color` 后续应该来自子任务 2 的 ArUco 编号结果。当前为了调试稳定，先手动通过 launch 参数固定颜色。
4. mock 模式保留用于没有相机、没有模型、没有 TF 时先验证底盘和外设动作。
5. 如果底层仍在使用旧 `/target` 链路，可以先设置 `motion_output:=both`，确认新旧话题输出一致后再切回默认 `cmd`。
