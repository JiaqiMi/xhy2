# Task 3 三个测试子任务实现逻辑

更新：2026.7.16

本文档说明 `src/auv_control/test` 目录下 Task 3 三个测试子任务脚本的逻辑、实现步骤、可调参数和代码运行方式。

Task 3 当前使用的话题约定：

- 子任务1的运动控制默认发布到 `/cmd/pose/ned`，消息类型为 `auv_control/PoseNEDcmd`。
- 子任务1仍保留旧版 `/target` 兼容接口，可通过 `motion_output=target` 或 `motion_output=both` 启用。
- 子任务3通过 `operation_mode` 选择人工操作或自动寻找；只有自动模式发布运动指令。
- 子任务3自动模式订阅 `/status/auv`，消息类型为 `auv_control/AUVData`，读取当前位姿、控制模式和三轴速度。
- 执行器控制发布到 `/cmd/actuator`，消息类型为 `auv_control/ActuatorControl`。

相关脚本：

| 子任务 | 脚本 | 功能 |
| --- | --- | --- |
| 子任务 1 | `test_task3_1_acquire_area.py` | 搜索箭头并到达任务获取区 |
| 子任务 2 | `test_task3_2_get_task.py` | 读取 ArUco 编号并点亮对应颜色灯 |
| 子任务 3 | `test_task3_3_inspect_and_drop.py` | 人工/自动寻找指定颜色方框并执行灯光、夹爪动作 |

## 公共约定

坐标系约定：

- `map`：全局坐标系，任务目标最终会转换到这个坐标系下。
- `base_link`：机器人自身坐标系。
- 当前项目约定 `base_link` 中 `x` 为前方，`y` 为右方，`z` 为向下。
- 因此“左侧 0.30 m”在代码中对应 `y = -0.30`。

主要话题：

| 话题 | 消息类型 | 用途 |
| --- | --- | --- |
| `/obj/target_message` | `auv_control/TargetDetection` | 子任务1目标三维识别结果及子任务2鱼眼 ArUco 识别结果 |
| `/web/detections` | `std_msgs/String` | YOLO 每帧 `top_k` 全部候选 JSON，子任务3直接读取 |
| `/cmd/pose/ned` | `auv_control/PoseNEDcmd` | 最新运动控制整包指令 |
| `/status/auv` | `auv_control/AUVData` | 当前经纬度、深度、姿态、控制模式、线速度和角速度反馈 |
| `/target` | `geometry_msgs/PoseStamped` | 旧运动目标话题，保留用于兼容调试 |
| `/cmd/actuator` | `auv_control/ActuatorControl` | 三色灯和夹爪舵机控制；新协议通过 `mode` 选择下发类型 |
| `/finished` | `std_msgs/String` | 当前测试子任务完成或失败状态 |

子任务1默认使用 `mode=4` 动力定位。子任务2固定使用 `mode=3`，保持启动深度和航向，并通过启动点位置误差与 `/status/auv` 速度阻尼计算 `TX/TY` 保持水平定点。子任务3的人工模式不发布运动指令；自动模式使用以下控制输出：

```text
operation_mode = auto
pose_cmd_topic = /cmd/pose/ned
PoseNEDcmd.mode = 3
force = TX TY 0 0 0 0
```

`mode=3` 对应定深定向模式：下位机闭环保持深度和航向，子任务3通过 `TX/TY` 控制前后和左右移动，`TZ/MX/MY/MZ` 保持为 0。自动模式根据 `/status/auv.linear_velocity` 对 `TX/TY` 增加反向速度阻尼，并使用 `pose.depth`、`pose.yaw` 和 `control_mode` 校验动作条件。

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

子任务2用于人工定点测试：先由人工把机器人停在 ArUco 目标正前方，节点启动时记录当前 `map -> base_link` 位姿。控制固定使用 `mode=3`：下位机保持启动深度和当前航向，上层根据水平位置误差及 `/status/auv` 前右速度反馈计算小幅 `TX/TY`，抑制惯性并保持启动点。节点先定点悬停10秒，悬停期间模型消息只打印但不计入稳定帧；随后正式识别 ArUco，稳定确认后点亮对应颜色灯3秒，最后熄灯并结束任务。子任务2不会搜索或主动靠近目标。

```text
人工停到目标正前方
  -> 记录启动时 map -> base_link 位姿
  -> mode=3 保持启动深度、航向和水平定点并悬停10秒
  -> 清空悬停期间识别计数
  -> 连续3帧稳定识别同一个 ArUco ID
  -> 映射目标颜色并亮灯3秒
  -> 熄灯并发布 /finished，任务结束
```

ArUco 编号到颜色的规则：

| ArUco 编号 | 目标颜色 |
| --- | --- |
| `1` / `2` | yellow |
| `3` / `4` | green |
| `5` / `6` | red |

该子任务支持两种输入模式：

| 模式 | 参数 | 说明 |
| --- | --- | --- |
| topic | `input_mode=topic` | 默认模式，订阅 `/obj/target_message` 的 `TargetDetection` |
| mock | `input_mode=mock` | 按 `mock_aruco_ids` 模拟编号序列，方便离线测试灯光 |

亮灯使用 `/cmd/actuator` 的三个独立灯字段：

| 颜色 | `red_light` | `yellow_light` | `green_light` |
| --- | --- | --- | --- |
| yellow | `0` | `1` | `0` |
| green | `0` | `0` | `1` |
| red | `1` | `0` | `0` |
| off | `0` | `0` | `0` |

### 实现步骤

1. 节点启动后创建 `/cmd/pose/ned`、`/cmd/actuator` 和 `/finished` 发布器，创建 TF 监听器并订阅 `/status/auv`。
2. 在 `hold_pose_timeout` 时间内读取 `map -> base_link`，记录机器人的启动位置和姿态；读取失败则熄灯、发布失败状态并退出。
3. 等待有效 `/status/auv`；超过 `status_wait_timeout` 仍没有速度反馈则失败退出。
4. 持续向 `/cmd/pose/ned` 发布 `mode=3`：目标深度和航向固定为启动值，`TZ/MX/MY/MZ=0`。
5. 用启动点与当前 TF 的 XY 误差生成基础 `TX/TY`，再减去 `/status/auv` 的前右速度阻尼，并经过最大力和单周期推力变化限制。
6. 先定点悬停 `initial_hover_seconds`，默认累计10秒有效反馈时间。期间收到的模型帧只打印日志，不计入连续识别。
7. 悬停结束后清空识别缓存，再根据 `input_mode` 开始读取真实 ArUco 或 mock 编号。
8. 如果是 `topic` 模式，订阅 `/obj/target_message`，消息类型为 `auv_control/TargetDetection`。
9. 当 `type=aruco_not_detected` 或 `class_name=-1` 时认为本帧没有识别到标记，正常清零连续计数。
10. 其他消息依次从 `class_name` 和 `type` 解析编号，兼容 `1`、`aruco_1`、`ArUco ID 1` 等格式，并打印置信度、坐标系和三维位置。
11. 要求连续 `stable_marker_count` 帧编号一致；定点反馈缺失、无目标、低置信度、解析失败、编号跳变或消息间隔超时都会清零计数，默认连续3帧。
12. 根据 `1/2 -> yellow`、`3/4 -> green`、`5/6 -> red` 得到目标颜色；编号不在 `1~6` 时忽略。
13. 通过 `/cmd/actuator` 点亮对应颜色灯，消息使用 `actuator_mode=2`，默认保持3秒；亮灯期间继续执行 `mode=3` 定点反馈控制。
14. 熄灯并保持 `gap_seconds`，随后明确发布一帧 `mode=3、TX=TY=0`，topic 模式默认识别一个编号后发布 `/finished` 并结束任务。

### 可调参数

基础输入参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input_mode` | `topic` | 输入模式，支持 `topic` 或 `mock` |
| `start_aruco_model` | `true` | 是否由子任务2 launch 启动鱼眼 ArUco 识别链路 |
| `aruco_topic` | `/obj/target_message` | 鱼眼 ArUco 的 `TargetDetection` 输出话题 |
| `min_confidence` | `0.5` | 进入连续稳定帧统计的最低置信度；无目标帧通常为 `0.0` |
| `mock_aruco_ids` | `[1, 3, 5, 2, 4, 6]` | mock 模式下的编号序列 |
| `max_topic_markers` | `1` | topic 模式下处理几个真实编号后结束，`0` 表示一直运行 |
| `stable_marker_count` | `3` | 连续多少帧相同编号才认为稳定 |
| `marker_timeout` | `1.0` | 编号样本过期时间 |

定点悬停参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `pose_cmd_topic` | `/cmd/pose/ned` | `PoseNEDcmd` 定点控制话题 |
| `status_topic` | `/status/auv` | 当前控制模式和前右下速度反馈 |
| `status_timeout` | `0.5` | 状态消息有效时间 |
| `status_wait_timeout` | `5.0` | 启动时等待第一条有效状态消息的最长时间 |
| `status_linear_velocity_scale` | `1.0` | `/status/auv.linear_velocity` 到 `m/s` 的缩放 |
| `hold_map_frame` | `map` | 记录定点使用的固定坐标系 |
| `hold_base_frame` | `base_link` | 机器人本体坐标系 |
| `initial_hover_seconds` | `10.0` | 开始正式识别前的定点悬停时间 |
| `hold_pose_timeout` | `5.0` | 等待启动定点 TF 的最长时间 |
| `hold_forward_position_gain` | `600.0` | 前后位置误差到基础 TX 的比例 |
| `hold_lateral_position_gain` | `600.0` | 左右位置误差到基础 TY 的比例 |
| `hold_forward_velocity_damping` | `300.0` | 前向速度反向阻尼 |
| `hold_lateral_velocity_damping` | `300.0` | 右向速度反向阻尼 |
| `hold_max_force` | `120.0` | TX/TY 最大绝对值 |
| `hold_position_tolerance` | `0.02` | 水平位置误差死区，单位米 |
| `hold_speed_deadband` | `0.03` | 低于该水平速度时不施加速度阻尼，单位 `m/s` |
| `hold_force_step` | `50.0` | 单个控制周期 TX/TY 最大变化量 |
| `hold_tx_sign` | `1.0` | 前后控制方向；实机方向相反时改为 `-1.0` |
| `hold_ty_sign` | `1.0` | 左右控制方向；实机方向相反时改为 `-1.0` |

亮灯参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `actuator_mode` | `2` | 新执行器协议模式；子任务2控制三色指示灯，必须使用 `2` |
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

### 调试日志

子任务2会持续输出以下中文日志，方便同时结合鱼眼视频判断：

```text
人工定点测试流程：记录启动位置 -> mode=3定点悬停10.0s -> 连续3帧识别 -> 对应颜色亮灯3.0s
已记录启动定点：map 坐标=(x,y,z)，航向=...deg
/status/auv：mode=3，深度=...，航向=...，速度前右下=(vx,vy,vz)m/s
水平定点反馈：位置误差=(前...,右...)m，速度=(前...,右...)m/s，指令=(TX=...,TY=...)
[识别消息 #...] 本帧无 ArUco：conf=0.000，type='aruco_not_detected'，class_name='-1'
[识别消息 #...] 第 1/3 帧有效：ArUco ID=...
[识别消息 #...] 第 2/3 帧有效：ArUco ID=...
[识别消息 #...] 第 3/3 帧有效：ArUco ID=...
识别到 ArUco 编号=...，对应目标颜色=...，开始亮灯
灯光执行中：颜色=...，三色指示=(红...,黄...,绿...)，剩余 ...s
子任务2完成，已熄灯、清零 TX/TY 并发布 /finished
```

如果 TF 或 `/status/auv` 暂时不可用，日志会明确说明反馈缺失，识别连续帧计数会清零；反馈恢复后从第1帧重新累计。

## 子任务 3：识别指定颜色方框并执行投放

### 任务逻辑

子任务3通过 `operation_mode` 提供两种运行方式：

| 模式 | 参数 | 运动控制 | 使用场景 |
| --- | --- | --- | --- |
| 人工操作 | `manual`，默认 | 不创建 TF，不发布运动指令 | 人工把机器人移动到方框上方，只测试模型、灯光和夹爪 |
| 自动寻找 | `auto` | 发布 `/cmd/pose/ned`，固定使用 `mode=3` | 自动向前搜索方框，并根据中心像素完成前后、左右对齐 |

两种模式不会同时运行。人工模式是公共基础流程；自动模式只在人工模式已有的“识别、稳定确认、灯光、夹爪、结束”流程前增加运动搜索和中心对齐。切回 `manual` 后不会创建运动发布器，也不会与人工遥控争抢 `/cmd/pose/ned`。

子任务3 launch 默认包含 `test_rectangles_detection.launch`，并直接使用该文件内部的默认参数。其中 `task_mode=detect`、`detect_mode=2`，检测模式2对应 `/home/xhy/catkin_ws/models/rectangle0710.pt` 彩色方框模型，不使用 `test_shapes_detection.launch` 的通用形状模型。任务节点通过 `model_detection_topic` 订阅 YOLO 的 `/web/detections` JSON，并使用图像尺寸、中心像素和 bbox。

如果相机和方框模型已经在其他终端启动，可以设置 `start_rectangle_model=false`，避免相机、YOLO 和 Web 节点重复启动。

目标颜色通过 `target_color` 人为固定，默认是 `yellow`。模型标签可以直接是 `yellow`，也可以是 `yellow_box`、`yellow_rectangle` 等包含颜色单词的标签。

人工模式状态机：

```text
等待模型话题
  -> 从全部候选中过滤目标颜色和最低置信度
  -> 连续多帧检查中心抖动和检测框面积变化
  -> 稳定确认目标
  -> 点亮对应颜色灯并保持
  -> 打开夹爪
  -> 关闭夹爪、熄灯并发布 /finished
```

自动模式状态机：

```text
读取启动时 map -> base_link
  -> 记录启动深度和航向
  -> 等待 /status/auv
  -> mode=3 速度闭环悬停10秒
  -> 前进0.30m
  -> 距离未到时提前减推力，利用惯性和反向阻尼刹停
  -> 左移0.20m、右移0.20m回到该排中心
  -> 前进0.20m
  -> 左移0.20m、右移0.20m回到该排中心
  -> 前进0.10m
  -> 左移0.20m、右移0.20m回到该排中心
  -> 搜索期间连续3帧稳定识别目标颜色方框
  -> 根据方框中心像素误差和前右速度阻尼计算 TX/TY
  -> 方框中心连续5帧保持在允许范围
  -> mode=3，实际速度、深度误差和航向误差均满足动作阈值
  -> 同时点亮对应颜色灯、打开夹爪
  -> 速度闭环悬停3秒
  -> 关闭夹爪、熄灯并发布 /finished，任务结束
```

自动模式参考 Task1 巡线控制的底层方式，但不拟合路径：

- `PoseNEDcmd.mode=3`，下位机负责深度和航向闭环。
- 启动后先根据 `/status/auv.linear_velocity` 对水平速度施加反向阻尼并悬停10秒，给人工放置方框留出时间。速度闭环可以抑制漂移，但没有全局 XY 位置闭环，不能保证绝对位置完全不变。
- 悬停10秒是硬前置阶段；期间即使模型看到方框也不触发运动或夹爪，悬停结束后重新累计稳定识别帧。
- `/web/detections` 未启动或消息超时时暂停当前位移步骤，立即保持 `TX/TY=0`。
- `/status/auv` 未收到或超过 `status_timeout` 时，自动搜索和细对准立即暂停并清零 `TX/TY`；状态反馈恢复后才能继续。
- 搜索位移使用 `map -> base_link` 实际位姿计算，不使用固定运动时间估算距离。
- 前进或横移进入 `auto_search_*_braking_distance` 后，基础推力按剩余距离线性减小，速度阻尼会根据惯性自动形成减推力、滑行或反向刹车；达到目标距离后锁定刹停状态，实际水平速度和推力指令都归零后才切换下一步骤。
- 前进步骤只给 `TX`，横移步骤只给 `TY`，航向始终保持任务启动时记录的当前航向。
- 每次左移0.20m后再右移0.20m，回到当前搜索排的中心，再进入下一段前进。
- 任意搜索步骤中连续 `auto_search_stable_detection_count` 帧稳定识别到方框，都会立即退出分段搜索并进入方框中心视觉细对准，默认是3帧。
- 全部预设搜索路径执行完仍未识别方框时，原地悬停等待，直到 `max_wait_seconds` 超时失败。
- 对齐阶段使用图像水平误差控制 `TY`，使用图像垂直误差控制 `TX`，并分别减去右向、前向速度阻尼以减少越过中心和反复振荡。
- 推力每个控制周期最多变化 `auto_force_step`；方框连续居中5帧后，还要满足 `control_mode=3`、实际速度、深度误差、航向误差和 `TX/TY` 指令全部达到条件，才执行动作。
- 目标帧丢失时立即把 `TX/TY` 清零；丢失超过 `detection_timeout` 后重新进入搜索。
- 执行动作前记录当前 `map` 坐标作为最终定点；动作期间根据 TF 位置误差和 `/status/auv` 速度阻尼持续修正 `TX/TY`，并保持启动深度和航向。开灯、打开夹爪后默认定点保持3秒，再关闭夹爪、熄灯并结束任务。

默认相机方向假设为“图像上方对应机器人前方，图像右方对应机器人右方”。如果现场方向相反，不改公式，只把 `auto_tx_sign` 或 `auto_ty_sign` 调成 `-1`。

终端会对每个满足颜色和置信度要求的目标帧逐帧打印确认进度，不进行日志节流，例如：

```text
[模型帧 #101] 第 1/3 帧有效：yellow(conf=0.81, center=(320,210), bbox=(250,150,390,270))
[模型帧 #102] 第 2/3 帧有效：yellow(conf=0.84, center=(322,209), bbox=(251,149,391,269))
[模型帧 #103] 第 3/3 帧有效，连续稳定识别确认通过
[模型帧 #121] 居中确认第 1/5 帧有效：yellow(conf=0.88, center=(318,238), ...)
[模型帧 #125] 居中确认第 5/5 帧有效：yellow(conf=0.90, center=(321,241), ...)
```

如果中间一帧没有目标颜色或置信度不足，会打印“本帧无效”和连续计数清零信息。达到稳定帧数后，如果中心抖动或 bbox 面积变化超限，也会明确打印“连续稳定性未通过”。自动模式每秒打印前右下速度、角速度、中心像素误差和当前 `TX/TY` 指令；搜索到达距离后会打印刹停进度，动作前会打印实际停稳进度。可通过 `http://机器人IP:8080` 结合标注视频人工判断。

人工模式等待超过 `max_wait_seconds` 仍未稳定识别目标时失败退出；自动模式在搜索和对齐累计超过该时间仍未完成时清零推力、发布失败状态并退出。

### 实现步骤

1. 启动子任务3 launch；默认由它包含 `test_rectangles_detection.launch` 并启动相机、彩色方框模型和 Web。
2. 模型按 `top_k` 发布候选列表到 `/web/detections`。
3. 任务节点解析 JSON，打印模型类别、类别编号、置信度、中心像素和 bbox。
4. 按 `target_color` 和 `min_confidence` 过滤目标。
5. 人工模式要求连续 `stable_detection_count` 帧都出现目标颜色；自动搜索使用独立的 `auto_search_stable_detection_count`，默认3帧。
6. 计算连续帧中心像素最大抖动，要求不超过 `stable_center_tolerance_px`。
7. 计算连续帧检测框面积变化比例，要求不超过 `stable_area_tolerance_ratio`。
8. 人工模式稳定确认后直接进入动作阶段。
9. 自动模式订阅 `/status/auv`，读取 `control_mode`、`pose`、`linear_velocity` 和 `angular_velocity`，并检查位姿和速度数值有效性。
10. 自动模式启动时记录当前深度和航向，并使用速度阻尼悬停 `auto_initial_hover_seconds`。
11. 按“前进0.30m、左右各0.20m、前进0.20m、左右各0.20m、前进0.10m、左右各0.20m”的路径搜索。
12. 每段位移通过 TF 判断距离；进入提前刹车距离后逐步减小基础推力并叠加速度阻尼，达到距离后锁定反向刹停，停稳后切换下一段。
13. 自动模式锁定目标后，将中心像素误差修正力与反向速度阻尼相加，再经过限幅和变化步长限制后发布。
14. 方框连续居中5帧后，检查 `control_mode=3`、实际速度、深度误差、航向误差及 `TX/TY` 指令是否全部满足动作条件。
15. 构造 `ActuatorControl` 时设置 `mode=2`，选择“仅执行器”下发模式。
16. 动作前记录当前 `map` 坐标；自动模式同时点亮目标颜色灯并打开夹爪，使用位置误差和速度阻尼定点保持 `open_seconds`，默认3秒。
17. 关闭夹爪并熄灭颜色灯；经过 `close_seconds` 后发布 `/finished`，默认不额外等待。

### 可调参数

模型启动参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `start_rectangle_model` | `true` | 是否由子任务3 launch 启动彩色方框模型链路 |

相机、图像分割、Web、模型路径、模型置信度、候选数量和推理频率均使用 `test_rectangles_detection.launch` 内部默认值，不在子任务3 launch 中重复声明。

识别确认参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `operation_mode` | `manual` | `manual` 人工移动，`auto` 自动寻找和对齐 |
| `model_detection_topic` | `/web/detections` | Python 直接使用的 YOLO 全部候选 JSON 话题 |
| `target_color` | `yellow` | 人为固定的目标颜色，通常为 `yellow`、`green` 或 `red` |
| `min_confidence` | `0.35` | 进入连续有效帧统计的最低置信度 |
| `stable_detection_count` | `5` | 人工模式连续多少帧满足条件才稳定确认 |
| `auto_search_stable_detection_count` | `3` | 自动搜索时连续多少帧稳定识别后进入细对准 |
| `auto_center_stable_detection_count` | `5` | 自动细对准时连续多少帧位于中心范围后执行动作 |
| `stable_center_tolerance_px` | `40.0` | 连续帧目标中心允许的最大抖动像素 |
| `stable_area_tolerance_ratio` | `0.35` | 连续帧 bbox 面积允许的最大变化比例 |
| `detection_timeout` | `2.0` | 模型输出或相邻目标样本允许的最大间隔 |
| `max_wait_seconds` | `300.0` | 人工等待或自动搜索、对齐的累计最长时间 |

自动控制参数，仅 `operation_mode=auto` 时生效：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `pose_cmd_topic` | `/cmd/pose/ned` | 自动运动指令话题 |
| `status_topic` | `/status/auv` | 底层 `AUVData` 当前状态反馈话题 |
| `status_timeout` | `0.5` | 超过该时间未收到状态消息时暂停自动运动 |
| `status_linear_velocity_scale` | `1.0` | `linear_velocity` 转换为 `m/s` 的缩放系数 |
| `status_angular_velocity_scale` | `0.01745329252` | `angular_velocity` 转换为 `rad/s` 的缩放系数，默认从 `deg/s` 转换 |
| `auto_initial_hover_seconds` | `10.0` | 自动任务开始后的速度闭环悬停时间 |
| `auto_search_forward_force` | `120.0` | 未识别目标时的向前搜索 TX 原始推力；数值越大通常前进越快 |
| `auto_search_lateral_force` | `120.0` | 左右横移搜索共同使用的 TY 原始推力绝对值；数值越大通常横移越快 |
| `auto_search_first_forward_distance` | `0.30` | 第一段前进距离 |
| `auto_search_second_forward_distance` | `0.20` | 第二段前进距离 |
| `auto_search_third_forward_distance` | `0.10` | 第三段前进距离 |
| `auto_search_lateral_distance` | `0.20` | 每次向左或向右横移的距离 |
| `auto_search_distance_tolerance` | `0.03` | 实际位移距离允许误差 |
| `auto_search_forward_braking_distance` | `0.08` | 前进剩余多少距离时开始提前减速，单位 `m` |
| `auto_search_lateral_braking_distance` | `0.08` | 横移剩余多少距离时开始提前减速，单位 `m` |
| `auto_forward_gain` | `250.0` | 图像垂直归一化误差到 TX 的增益 |
| `auto_lateral_gain` | `250.0` | 图像水平归一化误差到 TY 的增益 |
| `auto_max_forward_force` | `180.0` | 自动对齐 TX 最大绝对值 |
| `auto_max_lateral_force` | `180.0` | 识别到方框后，视觉自动对齐阶段的 TY 最大绝对值 |
| `auto_min_correction_force` | `50.0` | 超出中心容差时的最小修正力 |
| `auto_force_step` | `50.0` | 每个控制周期 TX/TY 最大变化量；越小越平滑，但加减速越慢 |
| `auto_forward_velocity_damping` | `300.0` | 前向速度每增加 `1m/s` 所施加的反向 TX 推力 |
| `auto_lateral_velocity_damping` | `300.0` | 右向速度每增加 `1m/s` 所施加的反向 TY 推力 |
| `auto_search_stop_speed` | `0.05` | 搜索步骤允许切换时的最大水平速度，单位 `m/s` |
| `auto_action_max_horizontal_speed` | `0.03` | 允许开灯开夹爪的最大水平速度，单位 `m/s` |
| `auto_action_max_vertical_speed` | `0.03` | 允许开灯开夹爪的最大垂直速度绝对值，单位 `m/s` |
| `auto_action_max_yaw_rate` | `0.05` | 允许开灯开夹爪的最大航向角速度绝对值，单位 `rad/s` |
| `auto_action_max_depth_error` | `0.08` | 当前深度相对启动深度允许的最大误差，单位 `m` |
| `auto_action_max_yaw_error_deg` | `5.0` | 当前航向相对启动航向允许的最大误差，单位 `deg` |
| `auto_hold_forward_position_gain` | `600.0` | 最终定点前后位置误差到 TX 的增益 |
| `auto_hold_lateral_position_gain` | `600.0` | 最终定点左右位置误差到 TY 的增益 |
| `auto_hold_max_force` | `120.0` | 最终定点保持允许的最大 TX/TY 绝对值 |
| `auto_hold_position_tolerance` | `0.02` | 最终定点位置误差死区，单位 `m` |
| `auto_tx_sign` | `1.0` | 前后方向标定；方向相反改为 `-1.0` |
| `auto_ty_sign` | `1.0` | 左右方向标定；方向相反改为 `-1.0` |
| `auto_target_center_u_ratio` | `0.5` | 期望水平中心占图像宽度的比例 |
| `auto_target_center_v_ratio` | `0.5` | 期望垂直中心占图像高度的比例 |
| `auto_center_tolerance_u_px` | `35.0` | 水平中心允许误差 |
| `auto_center_tolerance_v_px` | `35.0` | 垂直中心允许误差 |
| `auto_image_width` | `640.0` | JSON 缺少图像宽度时的回退值 |
| `auto_image_height` | `480.0` | JSON 缺少图像高度时的回退值 |


越过目标：先增大刹车距离，再增加速度阻尼。
停得太早：减小刹车距离。
反向摆动：降低速度阻尼。
最终定点偏软：增加位置增益或最大定点推力。
最终定点来回振荡：降低位置增益或增加速度阻尼。
动作参数：


| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `hold_seconds` | `1.0` | 仅人工模式使用：稳定识别后、执行动作前的保持时间 |
| `open_seconds` | `3.0` | 开灯并打开夹爪后的速度闭环悬停时间 |
| `close_seconds` | `0.0` | 关闭夹爪并熄灯后的额外确认时间；默认立即结束 |
| `actuator_topic` | `/cmd/actuator` | 执行器控制话题 |
| `actuator_mode` | `2` | 新协议控制模式：`0` 不响应、`1` 仅补光灯、`2` 仅执行器；子任务3必须使用 `2` |
| `clamp_open` | `0` | 夹爪打开舵机值 |
| `clamp_closed` | `255` | 夹爪关闭舵机值 |

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

子任务2 launch 默认会包含鱼眼 ArUco 识别，直接启动：

```bash
roslaunch auv_control task3_subtask2_get_task.launch
```

运行 launch 前，先人工把机器人停到 ArUco 目标正前方并停止手动运动指令。节点取得 `map -> base_link` 和 `/status/auv` 后会立即接管 `/cmd/pose/ned`，使用 `mode=3` 保持启动深度、当前航向和水平位置10秒，然后才开始连续3帧识别和亮灯流程。

如果鱼眼 ArUco 已经在另一个终端启动：

```bash
roslaunch stereo_depth test_aruco_detection_fisheye.launch
roslaunch auv_control task3_subtask2_get_task.launch start_aruco_model:=false
```

离线 mock 编号亮灯：

```bash
roslaunch auv_control task3_subtask2_get_task.launch \
  input_mode:=mock \
  start_aruco_model:=false
```

如果希望持续响应新的 ArUco 编号：

```bash
roslaunch auv_control task3_subtask2_get_task.launch max_topic_markers:=0
```

### 6. 子任务 3 操作

子任务3 launch 默认会同时启动彩色方框模型和 Web。人工模式下，先人工控制机器人到达对应颜色方框并停止，再运行：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch \
  operation_mode:=manual \
  target_color:=yellow \
  min_confidence:=0.35
```

自动模式：先确认下位机、`map -> base_link` TF、`/cmd/pose/ned` 控制链路和 `/status/auv` 状态反馈已经启动，再运行：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch \
  operation_mode:=auto \
  target_color:=yellow
```

自动节点启动并取得 TF、`/status/auv` 后先进行10秒速度闭环悬停，此时将方框放在机器人前方约0.6米处。随后节点按预设的三排前进、左右横移路径开始搜索。

启动后可以通过 `http://机器人IP:8080` 查看方框模型标注视频。如果方框模型已由其他终端启动，运行子任务时增加：

```bash
start_rectangle_model:=false
```

第一次自动测试建议先把搜索力、自动对齐最大力和 `auto_hold_max_force` 调小，并同时观察中心误差、位置误差、`TX/TY` 和前右速度。越过0.2米目标时先增大提前刹车距离，再逐步增加对应速度阻尼；减速过早时减小刹车距离，减速过强或出现反向摆动时减小阻尼。最终定点偏软时增加位置增益，定点来回振荡时降低位置增益或增加阻尼。如果前后或左右方向相反，分别把 `auto_tx_sign`、`auto_ty_sign` 改成 `-1.0`。

### 7. 常用检查命令

查看运动控制输出：

```bash
rostopic echo /cmd/pose/ned
```

查看底层机器人状态反馈：

```bash
rostopic echo /status/auv
```

查看执行器输出：

```bash
rostopic echo /cmd/actuator
```

查看任务完成状态：

```bash
rostopic echo /finished
```

查看子任务3使用的完整模型候选：

```bash
rostopic echo /web/detections
```

查看鱼眼 ArUco 识别输出：

```bash
rostopic echo /obj/target_message
```

### 8. 注意事项

1. 子任务2需要可用的 `map -> base_link` TF 和 `/status/auv`，并在启动后持续发布 `/cmd/pose/ned` 的 `mode=3` 指令；启动前应停止人工运动控制，避免多个节点争抢控制话题。
2. 子任务3的 `manual` 模式只验证识别和执行器链路，不依赖 TF 或状态反馈，也不会发布运动指令；`auto` 模式需要 `map -> base_link` TF 和 `/status/auv`，并会发布 `/cmd/pose/ned`。
3. 子任务 1 在 `topic` 模式下只搜索前方区域，并在稳定识别箭头后靠近到箭头中心前方约 0.30 m；`mock` 模式才使用固定偏移参数。
4. 子任务 3 的 `target_color` 后续应该来自子任务 2 的 ArUco 编号结果。当前为了调试稳定，先手动通过 launch 参数固定颜色。
5. 子任务3 launch 默认包含并启动 `test_rectangles_detection.launch`；仅当模型已在外部启动时设置 `start_rectangle_model=false`。
6. 使用 `manual` 模式时，应先完成人工移动并停止手动控制；使用 `auto` 模式时，由节点执行悬停、搜索和细对准。
7. 当前代码按前、右、下解释 `/status/auv.linear_velocity[0:3]`，并通过 launch 缩放参数转换速度单位；消息缺失或超时会暂停自动控制。
8. 当前本地 `ActuatorControl.msg` 还没有 `mode` 字段。团队合并新消息定义后必须重新执行 `catkin_make`；字段不存在时子任务2和子任务3都会发布失败信息并退出，不会发送旧格式执行器指令。
