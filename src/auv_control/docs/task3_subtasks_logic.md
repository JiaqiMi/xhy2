# Task 3 三个测试子任务实现逻辑

更新：2026.7.17

本文档说明 `src/auv_control/test` 目录下 Task 3 三个测试子任务脚本的逻辑、实现步骤、可调参数和代码运行方式。

Task 3 当前使用的话题约定：

- 子任务1订阅 `/arrow/direction` 的二维箭头 JSON，并使用 `/status/auv` 反馈完成模式3细对准和定点。
- 子任务3通过 `operation_mode` 选择人工操作或自动寻找；只有自动模式发布运动指令。
- 子任务3自动模式订阅 `/status/auv`，消息类型为 `auv_control/AUVData`，读取当前位姿、控制模式和三轴速度。
- 执行器控制发布到 `/cmd/actuator`，消息类型为 `auv_control/ActuatorControl`。

相关脚本：

| 子任务 | 脚本 | 功能 |
| --- | --- | --- |
| 子任务 1 | `test_task3_1_acquire_area.py` | 从箭头后方直行搜索，完成航向对准、相机/本体补偿和最终定点 |
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
| `/arrow/direction` | `std_msgs/String` | 子任务1二维箭头中心、方向和置信度 JSON |
| `/obj/target_message` | `auv_control/TargetDetection` | 子任务2鱼眼 ArUco 识别结果 |
| `/web/detections` | `std_msgs/String` | YOLO 每帧 `top_k` 全部候选 JSON，子任务3直接读取 |
| `/cmd/pose/ned` | `auv_control/PoseNEDcmd` | 最新运动控制整包指令 |
| `/status/auv` | `auv_control/AUVData` | 当前经纬度、深度、姿态、控制模式、线速度和角速度反馈 |
| `/cmd/actuator` | `auv_control/ActuatorControl` | 三色灯和夹爪舵机控制；新协议通过 `mode` 选择下发类型 |
| `/finished` | `std_msgs/String` | 当前测试子任务完成或失败状态 |

子任务1和子任务3自动模式使用 `mode=3` 定深定向，通过 `TX/TY` 控制前后和左右运动；子任务2使用 `mode=4` 保持人工停好的启动位姿。子任务3的人工模式不发布运动指令。子任务1和子任务3自动模式使用以下控制输出：

```text
operation_mode = auto
pose_cmd_topic = /cmd/pose/ned
PoseNEDcmd.mode = 3
force = TX TY 0 0 0 0
```

`mode=3` 对应定深定向模式：下位机闭环保持深度和航向，任务脚本通过 `TX/TY` 控制前后和左右移动，`TZ/MX/MY/MZ` 保持为0。脚本根据 `/status/auv.linear_velocity` 对 `TX/TY` 增加反向速度阻尼，并使用深度、航向、控制模式、线速度和角速度判断机器人是否真正停稳。

## 子任务 1：箭头识别、细对准和定点

### 任务逻辑

测试前由人工把机器人放在地面箭头后方约 `0.5m`，箭头方向不要求与机器人启动航向一致。节点启动后先使用模式3在当前位置定深定向悬停10秒，然后保持当前航向低速向前移动。箭头连续稳定识别3帧后停止直行，先原地对齐机器人航向和箭头方向，再在保持航向的同时缓慢靠近并对准箭头中心。

箭头识别使用 `stereo_depth/test_arrow_detection.launch` 发布的 `/arrow/direction`，消息类型为 `std_msgs/String`。JSON 中直接提供 `valid`、`confidence`、`center.u/v`、`angle_deg` 和 `discrete_direction`。当前箭头模型没有三维深度，因此本版本不再读取 `/obj/target_message`，也不再使用“距离箭头中心0.30m”的三维靠近判定。

箭头角度约定如下：

```text
图像向右 = 0度
图像向上 = 90度
图像向左 = 180度
图像向下 = 270度
```

默认认为图像向上对应机器人前方，因此航向换算为：

```text
相对航向修正 = yaw_correction_sign * wrap(camera_forward_angle_deg - angle_deg)
目标map航向 = 当前map航向 + 相对航向修正
```

默认 `camera_forward_angle_deg=90`、`yaw_correction_sign=1`。如果现场机器人实际转向与箭头方向相反，只修改 `yaw_correction_sign=-1`，不改控制代码。

水平细对准参考子任务3：图像垂直误差控制前后推力 `TX`，图像水平误差控制左右推力 `TY`；同时使用 `/status/auv.linear_velocity` 施加反向速度阻尼。航向先单独连续满足3帧，之后视觉靠近阶段仍持续刷新目标航向，因此可以一边缓慢靠近一边保持箭头航向。

箭头在图像中心只表示“摄像头位于箭头正上方”，不能直接说明机器人本体中心也在箭头上方。`driver/static_tf_broadcaster.py` 已发布 `base_link -> camera` 静态变换，当前下视左相机相对本体约为前方 `0.658m`、右方 `-0.030m`。代码在摄像头连续居中5帧并停稳后读取 `map -> camera`，把此时的相机 `map` 坐标作为箭头位置，再调用 `driver/motion_supervisor_core.py` 的 `map_error_to_body()` 将 `map` 位置误差转换为本体前/右误差，控制 `base_link` 移到箭头位置。最终机器人航向与箭头一致，且机器人本体中心位于箭头正上方。

高度保持启动深度。若 `/status/auv.pose.altitude` 有效且低于 `0.40m`，代码只把目标深度向上修正，不会继续靠近地面；高度为0或无效时会打印警告并保持启动深度。

### 实现步骤

1. 启动真正的箭头模型 `test_arrow_detection.launch`，订阅 `/arrow/direction` JSON。
2. 订阅 `/status/auv`，读取 `control_mode`、深度、高度、航向、三轴线速度和角速度。
3. 读取 `map -> base_link`，记录启动位置、深度和航向，使用模式3原地悬停10秒；悬停期间箭头帧不参与计数。
4. 悬停结束后记录直行起点，保持当前深度和航向，以 `search_forward_force` 低速向前搜索；利用速度阻尼抑制惯性。
5. 每个模型消息打印中文帧日志；无箭头、低置信度、JSON错误、中心或角度无效均记为无效帧。
6. 连续3帧有效，并且中心抖动和角度抖动均在容差内后锁定箭头，立即停止直行并保持当前点。
7. 根据箭头二维角度计算目标航向，先只转航向、不执行视觉平移；航向连续满足3帧且机器人停稳后进入靠近阶段。
8. 根据箭头中心像素计算 `TX/TY`，叠加速度反向阻尼，缓慢靠近箭头；此阶段仍持续更新目标航向。
9. 如果箭头短暂丢失，立即停止视觉移动并保持丢失位置；超过 `detection_timeout` 仍未恢复则在当前点重新连续识别3帧，不继续盲目前进。
10. 摄像头中心进入像素容差且航向误差小于10度后连续累计5帧，并等待模式3、速度、角速度、深度误差及 `TX/TY=0` 全部达标。
11. 读取 `map -> camera`，把摄像头当前 `map` 坐标作为箭头位置，同时打印 `base_link`、`camera` 坐标及相对前/右偏移。
12. 调用 `map_error_to_body()` 将本体到箭头的 `map` 误差转成本体前/右误差，使用位置外环将 `base_link` 移到箭头位置。
13. 本体到位并停稳后连续悬停10秒，发布 `/finished` 并退出；直行超过1.2米未识别、总时间超过300秒或最终悬停30秒仍不稳定均按失败结束。

### 可调参数

识别和任务参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `start_arrow_model` | `true` | launch是否包含真正的箭头模型 |
| `arrow_topic` | `/arrow/direction` | 二维箭头JSON话题 |
| `min_confidence` | `0.35` | 最低置信度 |
| `detection_timeout` | `1.0` | 箭头丢失多久后重新等待稳定3帧 |
| `max_wait_seconds` | `300.0` | 等待或细对准最长时间 |
| `initial_hover_seconds` | `10.0` | 启动后模式3悬停时间 |
| `search_forward_force` | `80.0` | 直行搜索使用的前进推力 |
| `max_search_distance` | `1.20` | 直行搜索最大安全距离，m |
| `camera_frame` | `camera` | 下视左相机TF坐标系 |
| `stable_detection_count` | `3` | 首次锁定箭头所需连续有效帧数 |
| `stable_center_tolerance_px` | `40.0` | 首次3帧允许的中心最大抖动 |
| `stable_angle_tolerance_deg` | `12.0` | 首次3帧允许的角度最大抖动 |

运动控制参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `pose_cmd_topic` | `/cmd/pose/ned` | 模式3运动控制话题 |
| `status_topic` | `/status/auv` | 位姿、模式和速度反馈话题 |
| `forward_gain/lateral_gain` | `160/160` | 图像中心误差转为前后/左右推力的增益 |
| `max_forward_force/max_lateral_force` | `100/100` | 慢速视觉靠近最大前后/左右推力 |
| `min_correction_force` | `35` | 非零修正的最小推力 |
| `force_step` | `30` | 每个控制周期最大推力变化量 |
| `forward_velocity_damping/lateral_velocity_damping` | `300/300` | 前后/左右速度反向阻尼 |
| `tx_sign/ty_sign` | `1/1` | 现场运动方向相反时分别改为-1 |
| `hold_forward_position_gain/hold_lateral_position_gain` | `600/600` | 等待及最终定点的位置外环增益 |
| `hold_max_force` | `100` | 定点及本体坐标补偿最大水平力 |
| `min_ground_clearance` | `0.40` | 高度有效时的最低对地距离 |

图像和航向参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `image_width/image_height` | `640/480` | 箭头图像尺寸 |
| `target_center_u_ratio/target_center_v_ratio` | `0.5/0.5` | 目标中心在图像中的比例 |
| `center_tolerance_u_px/center_tolerance_v_px` | `35/35` | 最终中心允许像素误差 |
| `center_stable_detection_count` | `5` | 中心和航向同时满足的连续帧数 |
| `heading_stable_detection_count` | `3` | 开始视觉靠近前的连续航向对准帧数 |
| `camera_forward_angle_deg` | `90.0` | 图像中机器人前方对应的箭头角度 |
| `yaw_correction_sign` | `1.0` | 航向换算方向；转向相反时改为-1 |
| `yaw_tolerance_deg` | `10.0` | 当前航向与箭头目标航向允许误差 |
| `yaw_target_filter_alpha` | `0.35` | 航向目标低通系数 |
| `yaw_target_max_step_deg` | `10.0` | 每帧目标航向最大变化量 |

停稳和完成参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `max_horizontal_speed` | `0.03` | 成功前允许的最大水平速度，m/s |
| `max_vertical_speed` | `0.03` | 成功前允许的最大垂直速度，m/s |
| `max_yaw_rate` | `0.05` | 成功前允许的最大航向角速度，rad/s |
| `max_depth_error` | `0.08` | 成功前允许的深度误差，m |
| `final_position_tolerance` | `0.05` | 最终定点前后/左右位置容差，m |
| `final_hold_seconds` | `10.0` | 本体到达箭头上方后的连续稳定悬停时间 |
| `final_hold_timeout` | `30.0` | 最终稳定悬停最长等待时间 |

### 代码操作步骤

默认由子任务1 launch启动箭头模型和任务节点：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch
```

如果箭头模型已经单独启动：

```bash
roslaunch stereo_depth test_arrow_detection.launch
roslaunch auv_control task3_subtask1_acquire_area.launch start_arrow_model:=false
```

调试时建议同时查看：

```bash
rostopic echo /arrow/direction
rostopic echo /status/auv
rostopic echo /cmd/pose/ned
```

## 子任务 2：读取 ArUco 编号并亮灯

### 任务逻辑

子任务2用于人工定点测试：先由人工把机器人停在 ArUco 目标正前方，节点启动时记录当前 `map -> base_link` 位姿，并固定使用 `mode=4` 动力定位保持该点。节点先悬停10秒，悬停期间模型消息不进入识别窗口；随后开始60秒识别计时。识别使用最近10个模型帧组成的滑动循环队列，只要同一个合法ID在当前窗口中出现3次就立即成功，不要求连续，也不需要等待10帧填满。例如第1、第3和第7帧是同一个ID时，第7帧到达后立即确认。成功后点亮对应颜色灯3秒；60秒内未确认则失败结束。

```text
人工停到目标正前方
  -> 记录启动时 map -> base_link 位姿
  -> mode=4 动力定位悬停10秒
  -> 开始60秒识别计时并清空10帧窗口
  -> 每个模型帧入队，队列始终只保留最近10帧
  -> 任一ID在窗口内出现3次立即识别成功
  -> 映射目标颜色并亮灯3秒
  -> 熄灯并发布 /finished
  -> 任务节点退出，由roslaunch关闭摄像头、识别节点和Web
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

1. 节点启动后创建 `/cmd/pose/ned`、`/cmd/actuator` 和 `/finished` 发布器，并创建 TF 监听器。
2. 在 `hold_pose_timeout` 时间内读取 `map -> base_link`，记录机器人的启动位置和姿态；读取失败则熄灯、发布失败状态并退出。
3. 持续向 `/cmd/pose/ned` 发布固定启动位姿，控制模式为 `mode=4`，六自由度附加力全部为0。
4. 先定点悬停 `initial_hover_seconds`，默认10秒。期间收到的模型消息只打印，不进入识别窗口。
5. 悬停结束后清空识别窗口并开始 `recognition_timeout` 计时，默认60秒。
6. topic模式订阅 `/obj/target_message` 的 `TargetDetection`；`aruco_not_detected/-1`、低置信度、无法解析或范围外ID均作为空帧入队。
7. 每个模型帧入队后只保留最近 `recognition_window_size` 帧，默认10帧；空帧同样占据一个窗口位置。
8. 每次入队后立即统计各ID出现次数。任一ID达到 `required_match_count`，默认3次，立即确认成功，不等待窗口填满。
9. 根据 `1/2 -> yellow`、`3/4 -> green`、`5/6 -> red` 得到目标颜色，通过 `/cmd/actuator` 的 `mode=2` 点亮3秒；亮灯期间继续发布mode4定点。
10. 成功或60秒超时后熄灯并发布 `/finished`，任务节点退出。
11. 子任务节点在launch中设置 `required=true`；默认由该launch启动的鱼眼摄像头、ArUco识别节点和Web会随整条launch链路一起关闭。

### 可调参数

基础输入参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input_mode` | `topic` | 输入模式，支持 `topic` 或 `mock` |
| `start_aruco_model` | `true` | 是否由子任务2 launch 启动鱼眼 ArUco 识别链路 |
| `aruco_topic` | `/obj/target_message` | 鱼眼 ArUco 的 `TargetDetection` 输出话题 |
| `min_confidence` | `0.5` | 合法ID帧的最低置信度；不足时作为空帧入队 |
| `recognition_window_size` | `10` | 滑动循环队列保留的最近模型帧数 |
| `required_match_count` | `3` | 同一ID在窗口内达到该次数即成功，不要求连续 |
| `recognition_timeout` | `60.0` | 正式识别最长时间，超时失败退出 |
| `mock_aruco_ids` | `[1,-1,1,2,-1,-1,1]` | mock模型帧；第1、3、7帧为ID 1，`-1`为空帧 |
| `mock_frame_interval` | `0.2` | mock相邻模型帧时间间隔 |

定点悬停参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `pose_cmd_topic` | `/cmd/pose/ned` | `PoseNEDcmd` 定点控制话题 |
| `hold_map_frame` | `map` | 记录定点使用的固定坐标系 |
| `hold_base_frame` | `base_link` | 机器人本体坐标系 |
| `initial_hover_seconds` | `10.0` | 开始正式识别前的定点悬停时间 |
| `hold_pose_timeout` | `5.0` | 等待启动定点 TF 的最长时间 |

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
流程=mode4定点悬停10.0s -> 最近10帧内同ID达到3帧 -> 亮灯3.0s；识别最长60.0s
已记录 mode=4 定点：map 坐标=(x,y,z)
识别进行中，剩余...s，当前窗口=[1,-,1]
[识别帧 #7] 有效ID=1，该ID在最近10帧中=3/3；窗口=7/10，队列=[1,-,1,2,-,-,1]
识别成功：ArUco ID=1 在最近10帧中达到3帧，不再等待窗口填满
ArUco ID=1 确认成功，对应颜色=yellow，开始亮灯
灯光执行中：颜色=...，三色指示=(红...,黄...,绿...)，剩余 ...s
任务成功/失败：...；灯光已关闭，节点即将退出
```

识别日志会逐帧打印窗口内容、当前ID在窗口中的计数以及剩余时间。无目标和无效识别显示为 `-`，但仍占据一个窗口位置。

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

人工先把机器人放在箭头后方约0.5米并停止手动控制。子任务1 launch 默认同时启动箭头模型、相机和Web；节点启动后先悬停10秒，再自动低速向前搜索：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch
```

浏览器可通过 `http://机器人IP:8080` 对照箭头标注画面和中文帧日志。如果箭头模型已经在另一个终端启动：

```bash
roslaunch stereo_depth test_arrow_detection.launch
roslaunch auv_control task3_subtask1_acquire_area.launch start_arrow_model:=false
```

第一次运动测试建议先观察默认的 `search_forward_force=80`、`max_forward_force=100`、`max_lateral_force=100` 和 `hold_max_force=100`。搜索或靠近太快时继续减小对应推力；前后或左右运动方向相反时分别修改 `tx_sign`、`ty_sign`；机器人航向转动方向与箭头相反时修改 `yaw_correction_sign`。运行前必须确认 `begin.launch` 中的 `static_tf_broadcaster.py` 正在发布 `base_link -> camera`。

### 5. 子任务 2 操作

子任务2 launch 默认会包含鱼眼 ArUco 识别，直接启动：

```bash
roslaunch auv_control task3_subtask2_get_task.launch
```

运行 launch 前，先人工把机器人停到 ArUco 目标正前方并停止手动运动指令。节点取得 `map -> base_link` 后会立即接管 `/cmd/pose/ned`，使用 `mode=4` 保持启动位姿10秒，然后进入“最近10帧中同一ID出现3次”的60秒识别流程。

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

调整识别窗口或超时时间：

```bash
roslaunch auv_control task3_subtask2_get_task.launch \
  recognition_window_size:=10 \
  required_match_count:=3 \
  recognition_timeout:=60.0
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

查看子任务1箭头方向输出：

```bash
rostopic echo /arrow/direction
```

### 8. 注意事项

1. 子任务2需要可用的 `map -> base_link` TF，并在启动后持续发布 `/cmd/pose/ned` 的 `mode=4` 定点指令；启动前应停止人工运动控制，避免多个节点争抢控制话题。
2. 子任务3的 `manual` 模式只验证识别和执行器链路，不依赖 TF 或状态反馈，也不会发布运动指令；`auto` 模式需要 `map -> base_link` TF 和 `/status/auv`，并会发布 `/cmd/pose/ned`。
3. 子任务1只执行低速直行搜索，不执行横移或九宫格搜索；人工应把机器人放在箭头后方约0.5米。摄像头视觉居中后还会使用 `map -> camera` 和 `map_error_to_body()` 补偿相机与本体偏移，最终目标是 `base_link` 位于箭头正上方。
4. 子任务 3 的 `target_color` 后续应该来自子任务 2 的 ArUco 编号结果。当前为了调试稳定，先手动通过 launch 参数固定颜色。
5. 子任务3 launch 默认包含并启动 `test_rectangles_detection.launch`；仅当模型已在外部启动时设置 `start_rectangle_model=false`。
6. 使用 `manual` 模式时，应先完成人工移动并停止手动控制；使用 `auto` 模式时，由节点执行悬停、搜索和细对准。
7. 当前代码按前、右、下解释 `/status/auv.linear_velocity[0:3]`，并通过 launch 缩放参数转换速度单位；消息缺失或超时会暂停自动控制。
8. 当前最新 `ActuatorControl.msg` 已包含 `mode` 字段；同步消息定义后需要重新执行 `catkin_make`，否则运行环境仍可能加载旧消息类型。
9. 子任务2默认 `start_aruco_model=true`，任务节点退出时会关闭同一launch启动的摄像头、识别节点和Web。如果使用 `start_aruco_model=false` 并在外部终端单独启动模型，该外部摄像头不属于本launch，无法由子任务节点自动关闭。
