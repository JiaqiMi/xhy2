# Task 3 三个测试子任务实现逻辑

更新：2026.7.21

本文档说明 `src/auv_control/test` 目录下 Task 3 三个测试子任务脚本的逻辑、实现步骤、可调参数和代码运行方式。

Task 3 当前使用的话题约定：

- 子任务1订阅 `/arrow/direction` 的二维箭头 JSON，只通过 `motion_supervisor` 发布绝对位姿目标和取消指令。
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
| `/cmd/motion/goal` | `geometry_msgs/PoseStamped` | 子任务1发布 `map` 下最终 `base_link` 绝对目标 |
| `/cmd/motion/cancel` | `std_msgs/Empty` | 子任务1请求运动管理层主动刹停并定点 |
| `/motion/state` | `auv_control/MotionState` | 子任务1读取运动状态、实际误差、速度和输出诊断 |
| `/cmd/pose/ned` | `auv_control/PoseNEDcmd` | 最新运动控制整包指令 |
| `/status/auv` | `auv_control/AUVData` | 当前经纬度、深度、姿态、控制模式、线速度和角速度反馈 |
| `/cmd/actuator` | `auv_control/ActuatorControl` | 三色灯和夹爪舵机控制；新协议通过 `mode` 选择下发类型 |
| `/finished` | `std_msgs/String` | 当前测试子任务完成或失败状态 |

子任务1不再直接发布 `/cmd/pose/ned`，也不计算 `TX/TY/MZ`。它只发布 `map` 绝对目标，`motion_supervisor` 根据 TF 和速度反馈完成运动、主动刹停以及模式2到模式4的接管，并且必须是 `/cmd/pose/ned` 的唯一发布者。子任务1的接口为：

```text
目标：/cmd/motion/goal
取消：/cmd/motion/cancel
反馈：/motion/state
底层运动输出：仅由motion_supervisor发布/cmd/pose/ned
```

子任务2和子任务3暂时保持各自现有控制实现，本次修改不涉及它们。

## 子任务 1：箭头识别、细对准和定点

### 任务逻辑

人工先把机器人放在箭头后方约 `0.5m`。`motion_supervisor` 由基础控制系统统一启动，测试人员不再单独启动运动控制launch。任务节点不直接控制模式和推力，只生成 `map` 坐标系下最终 `base_link` 绝对目标。

完整流程：

```text
等待TF、/status/auv和/motion/state
→ 只读取一次启动位置，固定点HOVER稳定保持10秒
→ 保持当前航向，前进0.5m
→ 未发现时依次到第一层左0.2m、右0.2m并回中
→ 再前进0.3m，依次到第二层左0.2m、右0.2m
→ 搜索途中任意一帧箭头位置有效，立即发布cancel退出搜索
→ 刹停后累计箭头位置连续3帧有效，暂不要求方向
→ 等待主动刹停及mode4接管
→ 保持启动航向，只依据图像u误差逐个发布左右横移目标
→ bbox完整留在画面内、左右位置稳定5帧且方向稳定3帧后再次cancel
→ 定点重新确认完整箭头方向3帧
→ 细对准第一段持续读取最新位置和方向，只慢速横移和转向，禁止前后移动
→ 航向和左右位置稳定3帧且真实停稳后，开放慢速前后居中
→ 完整箭头中心和方向稳定5帧且真实停稳
→ base_link沿当前前方移动0.35m，途中继续按实时u误差和方向慢速修正
→ 实际到达后HOVER保持10秒并结束
```

最新模型由 `test_arrow_pose_detection.launch` 启动，使用YOLO Pose输出 `tip、tail_left、tail_right` 三个关键点并计算方向，结果仍通过 `/arrow/direction` 发布 JSON。`confidence` 是检测框和三个关键点置信度的最小值。首次锁定和粗对准只要求 `valid`、`confidence`、`center.u/v` 有效，即使 `angle_deg` 暂时没有输出也可以继续。方向确认和细对准还要求 `bbox` 四边均位于图像内、距边缘达到配置值且框尺寸不小于配置值；这样只有摄像头完整看到箭头时，方向帧才会计数。图像角度约定为右方 `0°`、上方 `90°`，默认图像上方对应机器人前方：

```text
相对航向修正 = yaw_correction_sign * wrap(camera_forward_angle_deg - angle_deg)
最终map航向 = 当前map航向 + 相对航向修正
```

粗对准阶段锁定发现箭头时的航向，只把图像 `u` 误差转换为左右横移小步；图像 `v` 误差只打印，不产生前后目标。只有左右位置和完整箭头方向都稳定后才进入细对准。细对准第一段继续实时读取每一帧完整箭头的位置和方向，每次最多横移 `0.03m`、转向 `3°`，并强制前后步长为0。航向和左右位置稳定后，第二段才把 `v` 误差转换为慢速前后小步，同时继续修正左右和yaw。

搜索和粗对准的每个小步在生成时只计算一次 `map` 绝对目标，并等待上一目标满足实际到达门槛。细对准需要持续看见箭头，因此不等待每个小步都完全HOVER，而是按 `fine_goal_min_interval` 节流，并从最新TF位姿生成受限的小目标；所有目标仍由 `motion_supervisor` 执行。初始悬停点只在启动时读取一次；机器人漂移时仍追踪原悬停点，绝不把漂移后的当前位置重新锁存为新悬停点。

方向确认只累计“完整箭头”帧，并检查连续角度抖动。代码不再用第一次得到的角度一次性发布完整转向目标，而是每次根据最新方向生成受限航向小步。中心和方向最终稳定后，`base_link` 按标定值沿当前前方移动 `0.35m`；前移过程中摄像头仍必须持续看到完整箭头，任务会使用实时 `u` 误差和方向慢速修正最终目标，`v` 误差只记录，不再抵消标定前移量。

任务不会把 `MotionState.HOVER` 单独视为准确到达，还会检查 `base_position_error`、航向误差、水平速度、航向角速度和深度误差。高度低于 `0.40m` 时只向上修正目标深度。

所有搜索点都相对初始固定悬停点和初始航向一次性计算：前 `0.5m` 后，以该中线点为中心搜索左、右各 `0.2m` 并回中；再沿中线前进 `0.3m`，重复左、右各 `0.2m`。路径任意阶段收到一帧有效位置都会立刻退出搜索，不会继续走剩余点。

失败、视觉丢失需要刹停或任务退出时，程序停止发布旧目标并发送 `/cmd/motion/cancel`。反馈超时、`SAFE`、全部搜索点仍未找到箭头或累计超过 `300s` 都会失败退出。

### 实现步骤

1. 订阅 `/motion/state`、`/status/auv`、`/arrow/direction`，读取 `map -> base_link` TF。
2. 只读取一次启动位姿并持续发布同一绝对目标；漂移时不更新目标点，等待机器人回到固定点并由模式4接管。
3. `HOVER` 和任务侧实际误差都通过后，连续保持 `initial_hover_seconds`。
4. 相对固定启动点生成7个搜索点：前0.5、第一层左、第一层右、第一层回中、再前0.3、第二层左、第二层右。
5. 每到一个点都等待真实到达门槛通过再进入下一点；每个箭头模型帧打印置信度、中心、位置是否有效以及方向是否提供。
6. 搜索中任意一帧位置有效就停止发布搜索目标并发送取消；刹停后重新累计位置连续3帧有效。
7. 位置3帧有效后锁定当前航向，只根据图像 `u` 误差生成左右横移目标；粗对准阶段前后偏置固定为0。
8. 粗对准阶段解析 `bbox`，只有箭头完整可见、左右位置稳定5帧且方向稳定3帧才取消刹停并准备细对准。
9. 定点重新确认完整箭头方向3帧后进入细对准第一段；该段只允许慢速左右横移和航向小步，前后步长恒为0。
10. 航向和左右位置连续3帧通过且最新目标真实到达后进入第二段，开放慢速前后、左右和yaw实时修正。
11. 细对准任一阶段完整箭头或方向丢失超过阈值都会取消、刹停并重新识别。
12. 中心和方向稳定5帧且真实到达后生成前方 `0.35m` 的目标；前移期间继续实时修正左右和yaw，并要求视觉跟踪稳定5帧。
13. 最终目标到达后连续定点10秒，发布 `/finished`；成功或失败退出前都先发送取消。

### 可调参数

以下默认值均指 `task3_subtask1_acquire_area.launch` 的实际默认值。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `rate` | `5.0` | 任务目标发布和阶段判断频率，单位 `Hz` |
| `start_arrow_model` | `true` | launch是否包含真正的箭头模型 |
| `arrow_topic` | `/arrow/direction` | 二维箭头JSON话题 |
| `min_confidence` | `0.35` | 最低方向置信度；新Pose链路取检测框和三个关键点置信度的最小值 |
| `detection_timeout` | `1.0` | 普通识别阶段的有效帧超时，单位 `s` |
| `visual_loss_cancel_seconds` | `0.5` | 运动中视觉丢失多久后取消刹停 |
| `stable_detection_count` | `3` | 首次锁定箭头位置所需连续有效帧数，不要求方向 |
| `stable_center_tolerance_px` | `40.0` | 首次3帧允许的中心最大抖动 |
| `stable_angle_tolerance_deg` | `12.0` | 方向确认允许的最大角度抖动 |
| `center_stable_detection_count` | `5` | 左右粗对准、最终居中和前移视觉跟踪连续有效帧数 |
| `heading_stable_detection_count` | `3` | 完整箭头方向稳定及进入细对准所需连续帧数 |
| `heading_aligned_detection_count` | `3` | 航向和左右位置通过后，开放前后移动所需连续帧数 |
| `full_arrow_edge_margin_px` | `15.0` | 完整箭头bbox距图像四边的最小像素距离 |
| `full_arrow_min_bbox_width_px` | `30.0` | 用于方向和细对准的bbox最小宽度 |
| `full_arrow_min_bbox_height_px` | `30.0` | 用于方向和细对准的bbox最小高度 |
| `motion_goal_topic` | `/cmd/motion/goal` | 绝对目标发布话题 |
| `motion_cancel_topic` | `/cmd/motion/cancel` | 主动刹停取消话题 |
| `motion_state_topic` | `/motion/state` | 运动状态反馈话题 |
| `motion_state_timeout` | `0.5` | 运动状态新鲜度限制，单位 `s` |
| `motion_startup_timeout` | `10.0` | 启动等待运动状态反馈上限 |
| `cancel_timeout` | `15.0` | 取消后等待刹停和HOVER的上限 |
| `status_topic` | `/status/auv` | 位姿、模式和速度反馈话题 |
| `status_timeout` | `0.5` | `/status/auv` 新鲜度限制，单位 `s` |
| `initial_hover_seconds` | `10.0` | 搜索前连续定点时间 |
| `search_initial_forward_distance` | `0.50` | 第一层搜索中线距启动点的前向距离 |
| `search_lateral_distance` | `0.20` | 每一层左、右搜索点距中线的距离 |
| `search_second_forward_distance` | `0.30` | 第一层结束后沿中线继续前进的距离 |
| `max_wait_seconds` | `300.0` | 搜索和对准累计超时 |
| `visual_lateral_gain_m` | `0.20` | 粗对准左右目标距离增益 |
| `visual_min_step_m` | `0.01` | 粗对准非零横移小步最小值 |
| `visual_max_step_m` | `0.08` | 粗对准单次横移小步最大值 |
| `visual_goal_min_interval` | `1.0` | 粗对准两个目标之间的最短时间 |
| `fine_forward_gain_m` | `0.10` | 航向稳定后，v误差到前后慢速目标的增益 |
| `fine_lateral_gain_m` | `0.10` | 细对准和最终前移阶段的左右慢速增益 |
| `fine_visual_min_step_m` | `0.005` | 细对准非零平移小步最小值 |
| `fine_visual_max_step_m` | `0.03` | 细对准单次平移小步最大值，建议先保持较小 |
| `fine_yaw_max_step_deg` | `3.0` | 每次实时方向更新允许的最大航向小步 |
| `fine_goal_min_interval` | `0.5` | 两个细对准实时目标之间的最短时间 |
| `visual_forward_sign/visual_lateral_sign` | `1/1` | 现场移动方向相反时改对应符号 |
| `center_tolerance_u_px/center_tolerance_v_px` | `35/35` | 图像中心允许误差 |
| `camera_forward_angle_deg` | `90.0` | 图像中机器人前方对应角度 |
| `yaw_correction_sign` | `1.0` | 航向修正方向；转反时改为-1 |
| `yaw_tolerance_deg` | `10.0` | 视觉箭头方向最终允许误差 |
| `base_link_forward_offset` | `0.35` | 从图像中心位置到base_link位于箭头上方的前移标定值 |
| `arrival_position_tolerance` | `0.05` | 任务阶段真实水平到达门槛 |
| `arrival_yaw_tolerance_deg` | `5.0` | 任务阶段真实航向到达门槛 |
| `arrival_max_horizontal_speed` | `0.02` | 到达时最大水平速度，单位 `m/s` |
| `arrival_max_yaw_rate_deg_s` | `0.5` | 到达时最大航向角速度 |
| `max_depth_error` | `0.08` | 到达和最终保持允许深度误差 |
| `min_ground_clearance` | `0.40` | 高度有效时的最低对地距离 |
| `final_hold_seconds` | `10.0` | 本体到达箭头上方后的连续稳定悬停时间 |
| `final_hold_timeout` | `30.0` | 最终稳定悬停最长等待时间 |

现场推荐调参顺序：

1. 先让机器人保持不动，对照网页视频和逐帧日志确认箭头中心、角度、方向标签及 `min_confidence` 正确。
2. 先减小三个搜索距离验证路径顺序和左右方向；搜索速度和刹车由 `motion_supervisor.yaml` 统一调节，不在任务脚本中调推力。
3. 粗对准先确认 `visual_lateral_sign`，确保机器人只左右横移且航向不变；再观察日志中的bbox四边和“完整可见”结果，调整边缘留白与最小框尺寸。
4. 细对准先把 `fine_visual_max_step_m` 和 `fine_yaw_max_step_deg` 保持在较小值，确认航向未稳定前日志中的前后步长始终为0。
5. 航向稳定后再确认 `visual_forward_sign` 和 `yaw_correction_sign`；若移动方向相反只改对应符号。
6. 最后结合网页图像和到达门槛日志调整细对准增益、位置、速度、航向和深度容差；底层速度、阻尼和刹车仍在 `motion_supervisor` 中调整。

### 调试日志

子任务1保留以下必要中文日志，建议和网页箭头标注画面同步观察：

1. 启动配置日志：打印完整任务流程、识别门槛、完整bbox门槛、粗/细对准步长、航向单步限制、运动到达门槛和最低离地保护。
2. 每帧识别日志：打印帧号、置信度、中心、u/v误差、bbox、完整可见结果、角度、方向和当前阶段。
3. 连续帧日志：分别打印位置锁定 `1/3`、左右粗对准 `1/5`、完整方向 `1/3`、航向对齐 `1/3`、最终居中 `1/5` 和base_link前移跟踪 `1/5`。
4. 控制目标日志：打印本体前/右小步、实时方向误差、航向小步和最终map目标；航向未稳定时会明确显示“允许前后=否”且前向小步为0。
5. 保护日志：打印普通有效帧与完整方向帧年龄、视觉丢失阈值、cancel原因、后续阶段和刹停等待进度。
6. 到达判定日志：打印HOVER、目标一致性、base_link实际误差、水平速度、航向角速度、深度误差及每项是否通过。

典型识别和控制日志如下：

```text
[箭头帧#37] 有效：conf=0.862，中心=(318.0,246.0)，误差=(u=-2.0,v=+6.0)px，bbox=(120,80,510,430)，完整可见=是
[箭头帧#39] 方向识别第3/3帧，完整箭头[通过]，平均角度=91.8deg，抖动=2.1/12.0deg
[箭头帧#52] 细对准实时目标：允许前后=否，本体偏置=(前+0.000,右-0.012)m，方向误差/航向小步=(+14.0/+3.0)deg
[箭头帧#71] 最终中心和航向第5/5帧有效，中心误差=(-4.0,+9.0)px，方向误差=+2.3deg
```

### 代码操作步骤

基础控制系统负责启动运动管理层；测试人员不重复启动。确认 `/motion/state` 正常，并且 `/cmd/pose/ned` 只有 `motion_supervisor` 一个发布者，再启动子任务1：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch
```

如果箭头模型已经单独启动：

```bash
roslaunch stereo_depth test_arrow_pose_detection.launch \
  start_camera:=false \
  start_splitter:=false
roslaunch auv_control task3_subtask1_acquire_area.launch start_arrow_model:=false
```

调试时建议同时查看：

```bash
rostopic echo /arrow/direction
rostopic echo /status/auv
rostopic echo /motion/state
rostopic info /cmd/pose/ned
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
| 人工操作 | `manual` | 不创建 TF 和运动接口，不发布运动指令 | 人工把机器人移动到方框上方，只测试模型、灯光和夹爪 |
| 自动寻找 | `auto`，launch 默认 | 发布 `/cmd/motion/goal`、`/cmd/motion/cancel`，读取 `/motion/state` | 自动搜索方框并根据中心像素完成前后、左右对齐 |

两种模式共用颜色过滤、稳定帧判断、灯光、夹爪和结束流程。人工模式不会连接运动接口；自动模式只在公共识别与执行流程前增加运动搜索和视觉对齐。

自动模式不再直接发布 `/cmd/pose/ned`，也不在任务节点中计算 `TX/TY`、速度阻尼、反向力、提前刹车或最终定点推力。所有底层 `mode=4` 定点控制、推力、阻尼和刹车统一由 `motion_supervisor` 完成。`motion_supervisor` 由基础控制系统统一启动；任务测试人员只检查接口已就绪，并确保它是 `/cmd/pose/ned` 的唯一发布者。

子任务3 launch 默认包含 `test_rectangles_detection.launch`，任务节点订阅 `/web/detections` JSON。模型已经在其他终端启动时设置 `start_rectangle_model=false`。目标颜色由 `target_color` 固定，默认 `red`，可改为 `yellow` 或 `green`。

人工模式：

```text
按颜色和置信度过滤候选
  -> 连续多帧检查中心抖动和bbox面积变化
  -> 稳定确认
  -> 点亮对应颜色灯并打开夹爪
  -> 关闭夹爪、熄灯并发布/finished
```

自动模式：

```text
等待TF、/status/auv和/motion/state
  -> 锁存一次启动map位置、深度和当前航向
  -> 发布固定启动点，稳定HOVER持续10秒
  -> 前进0.30m，左0.20m、右0.20m
  -> 前进0.20m，左0.20m、右0.20m
  -> 前进0.10m，左0.20m、右0.20m
  -> 搜索中连续3帧稳定识别方框
  -> cancel并等待主动刹停进入HOVER
  -> 根据中心像素误差发布前后、左右位置小步
  -> 方框连续5帧居中，锁定最终固定点并等待稳定HOVER
  -> 同时开对应颜色灯和夹爪
  -> 重发同一固定点并保持3秒
  -> 关闭夹爪和灯，发布/finished并结束
```

控制细节与保护：

- 启动位置只锁存一次。机器人漂移时仍追踪原固定点，不会用漂移后的当前位置更新悬停目标。
- 每段搜索只生成一个 `map` 绝对目标。只有 `MotionState.state==HOVER`、反馈目标与当前任务目标匹配，且实际位置、航向和速度门槛均通过，才切换下一段。
- 搜索顺序保持原逻辑：左移0.20m后右移0.20m，回到当前搜索排中心。
- 搜索中模型断流时发布取消并锁住当前位置；模型恢复后继续当前搜索步骤原来的绝对目标。
- 连续3帧稳定识别方框后立即退出搜索，先取消并刹停，再开始视觉细对准。
- 图像垂直误差生成前后位置小步，水平误差生成左右位置小步。每个模型帧最多更新一次，并限制单步距离和更新频率。
- 方框进入中心容差后立即锁定固定点。连续5帧居中只是视觉条件，仍需等待该固定点对应的稳定 `HOVER`。
- 动作前还检查 `/status/auv.control_mode==4`、垂直速度、启动深度误差和启动航向误差。
- 当前帧丢失目标时只取消一次并锁住当前位置；超过 `detection_timeout` 后返回当前搜索步骤。
- `/motion/state` 超时、进入 `SAFE`、取消超时或超过 `max_wait_seconds` 时失败退出，并关闭灯和夹爪。
- 完成、失败或节点关闭时发布 `/cmd/motion/cancel`，由 supervisor 刹停并定点接管。

默认相机方向为“图像上方对应机器人前方，图像右方对应机器人右方”。方向相反时修改 `auto_forward_sign` 或 `auto_lateral_sign`。

### 必要日志

识别日志逐帧打印：

```text
[模型帧 #101] 第 1/3 帧有效：red(conf=0.81, center=(320,210), bbox=(...))
[模型帧 #103] 第 3/3 帧有效，连续稳定识别确认通过
[模型帧 #121] 居中确认第 1/5 帧有效：red(...)
[模型帧 #125] 居中确认第 5/5 帧有效：red(...)
```

自动控制日志按 `log_interval` 打印：

```text
固定悬停点已锁存：map=(x,y,z)，后续不会跟随漂移位置更新
搜索步骤2/10进行中：向前移动0.30m，motion=TRANSLATE，base误差=...
运动反馈：state=HOVER，base_link误差=...，输出=(TX=...,TY=...,MZ=...)
搜索步骤2/10到达判定：反馈新鲜[通过]，state=TRANSLATE/HOVER[未通过]，目标一致[通过]，目标差值=(水平...,深度...,航向...)，实际到达[未通过]，实际门槛=(位置.../...m，速度.../...m/s，航向.../...deg，角速度.../...deg/s)，输出=(TX=...,TY=...,MZ=...)
视觉位置小步已发布：像素误差=(u=...,v=...)，本体偏置=(前...,右...)m
视觉目标更新间隔尚未满足：距上次发布...s，要求>=...s；本帧只更新识别结果
动作放行等待：居中=5/5帧；motion=.../HOVER；目标匹配误差=...；水平速度=.../...m/s；垂直速度=.../...m/s；航向角速度=.../...rad/s；mode=.../4
执行器指令已发布：mode=2，夹爪=...，颜色灯=(红...,黄...,绿...)
```

“到达判定”日志会逐项标出反馈新鲜度、状态、目标一致性和实际运动门槛是否通过。卡在某一步时先看第一个“未通过”项，再对照 `http://机器人IP:8080`、`rostopic echo /motion/state` 和 `rostopic echo /status/auv`。

### 实现步骤

1. 根据 `operation_mode` 决定是否创建 supervisor 运动接口和 TF。
2. 解析模型 JSON，按颜色和置信度过滤并打印候选。
3. 搜索阶段连续3帧检查中心抖动和 bbox 面积变化。
4. 自动模式锁存启动点，发布固定目标并稳定悬停10秒。
5. 按预设距离逐段生成绝对目标，每段等待匹配目标的 `HOVER`。
6. 稳定识别方框后发布取消，等待主动刹停。
7. 根据中心误差生成位置小步，方框连续5帧居中后锁定最终点。
8. 核对 HOVER、目标匹配、实际误差、速度、mode、深度和航向。
9. 发布 `ActuatorControl.mode=2`，开灯和夹爪并定点3秒。
10. 关闭夹爪和灯，发布 `/finished`，取消运动目标并退出。

### 可调参数

识别参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `operation_mode` | `auto` | `manual`人工移动，`auto`自动搜索 |
| `rate` | `5.0` | 主循环和目标重发频率，Hz |
| `log_interval/warning_log_interval` | `1.0/2.0` | 普通/异常日志节流周期，s |
| `start_rectangle_model` | `true` | 是否启动方框模型链路 |
| `model_detection_topic` | `/web/detections` | YOLO候选JSON话题 |
| `target_color` | `red` | yellow、green或red |
| `min_confidence` | `0.35` | 最低置信度 |
| `stable_detection_count` | `5` | 人工模式稳定帧数 |
| `auto_search_stable_detection_count` | `3` | 自动搜索锁定帧数 |
| `auto_center_stable_detection_count` | `5` | 自动居中确认帧数 |
| `stable_center_tolerance_px` | `40.0` | 搜索稳定帧中心最大抖动 |
| `stable_area_tolerance_ratio` | `0.35` | bbox面积最大变化比例 |
| `detection_timeout/max_wait_seconds` | `2.0/300.0` | 识别间隔/任务总超时，s |

supervisor接口和到达判定：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `motion_goal_topic` | `/cmd/motion/goal` | map绝对目标 |
| `motion_cancel_topic` | `/cmd/motion/cancel` | 主动刹停 |
| `motion_state_topic` | `/motion/state` | 运动状态反馈 |
| `motion_state_timeout` | `0.5` | 反馈最大年龄，s |
| `motion_startup_timeout/cancel_timeout` | `10.0/15.0` | 首份反馈/取消等待超时，s |
| `goal_match_position_tolerance` | `0.03` | 反馈目标与任务目标水平差，m |
| `goal_match_depth_tolerance` | `0.03` | 反馈目标与任务目标深度差，m |
| `goal_match_yaw_tolerance_deg` | `2.0` | 反馈目标与任务目标航向差 |
| `arrival_position_tolerance` | `0.05` | base_link实际水平误差，m |
| `arrival_yaw_tolerance_deg` | `5.0` | 实际航向误差 |
| `arrival_max_horizontal_speed` | `0.02` | 最大水平速度，m/s |
| `arrival_max_yaw_rate_deg_s` | `0.5` | 最大航向角速度，deg/s |
| `status_topic/status_timeout` | `/status/auv / 0.5` | AUV状态话题和超时 |
| `status_linear_velocity_scale` | `1.0` | 线速度换算为m/s的系数 |

搜索和视觉参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `auto_initial_hover_seconds` | `10.0` | 固定启动点稳定后的悬停时间 |
| `auto_search_first_forward_distance` | `0.30` | 第一段前进距离，m |
| `auto_search_second_forward_distance` | `0.20` | 第二段前进距离，m |
| `auto_search_third_forward_distance` | `0.10` | 第三段前进距离，m |
| `auto_search_lateral_distance` | `0.20` | 每次左右横移距离，m |
| `auto_visual_forward_gain_m` | `0.10` | 垂直误差到前后步长的增益 |
| `auto_visual_lateral_gain_m` | `0.10` | 水平误差到左右步长的增益 |
| `auto_visual_min_step_m/max_step_m` | `0.005/0.03` | 最小/最大单帧位置步长 |
| `auto_visual_goal_min_interval` | `0.50` | 视觉目标最短更新间隔，s |
| `auto_forward_sign/auto_lateral_sign` | `1.0/1.0` | 前后/左右方向标定 |
| `auto_target_center_u_ratio/v_ratio` | `0.5/0.5` | 期望图像中心比例 |
| `auto_center_tolerance_u_px/v_px` | `35/35` | 水平/垂直中心容差 |
| `auto_image_width/height` | `640/480` | 图像尺寸回退值 |

动作参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `auto_action_max_horizontal_speed` | `0.03` | 动作前最大水平速度，m/s |
| `auto_action_max_vertical_speed` | `0.03` | 动作前最大垂直速度，m/s |
| `auto_action_max_yaw_rate` | `0.05` | 动作前最大航向角速度，rad/s |
| `auto_action_max_depth_error` | `0.08` | 相对启动深度最大误差，m |
| `auto_action_max_yaw_error_deg` | `5.0` | 相对启动航向最大误差 |
| `hold_seconds/open_seconds/close_seconds` | `1.0/3.0/0.0` | 人工确认/开灯夹爪/关闭确认时间 |
| `actuator_mode` | `2` | 仅执行器模式 |
| `clamp_open/clamp_closed` | `0/255` | 夹爪开/关值 |

`motion_supervisor` 常用运动参数位于 `src/auv_control/config/motion_supervisor.yaml`，它们不是子任务3 launch 参数，修改后需要重启 supervisor：

| 参数 | 当前值 | 主要作用 |
| --- | --- | --- |
| `xy_max_speed` | `0.30` | 水平规划最大速度，m/s |
| `xy_position_gain` | `1.00` | 水平位置误差到目标速度的增益 |
| `xy_max_acceleration/xy_max_jerk` | `0.20/0.40` | 水平加速度和加加速度限制 |
| `max_tx_positive/negative` | `700/1000` | 前后运动阶段推力上限 |
| `max_ty_positive/negative` | `1200/2000` | 左右运动阶段推力上限 |
| `kv_x_positive/negative` | `700/1000` | 前后速度阻尼增益 |
| `kv_y_positive/negative` | `1200/2000` | 左右速度阻尼增益 |
| `brake_gain_tx_* / brake_gain_ty_*` | 按方向分别配置 | 主动刹车增益；过冲时结合实测速度调整 |
| `brake_acceleration_* / brake_margin_*` | 按方向分别配置 | 刹车距离估计和附加停车余量 |
| `capture_radius/capture_exit_radius` | `0.15/0.25` | 进入 mode=4 接管和重新跟踪的水平误差门槛，m |
| `control_center_hold_tolerance` | `0.03` | mode=4 当前控制中心位置死区，m |
| `horizontal_speed_threshold` | `0.015` | supervisor 判定水平停稳的速度上限，m/s |
| `yaw_tolerance_deg/yaw_rate_threshold_deg_s` | `5.0/0.3` | supervisor 判定航向到达和停转的门槛 |
| `goal_static_capture_seconds` | `0.80` | 目标静止后允许进入接管判定的时间，s |

任务 launch 中的 `arrival_*` 和 `auto_action_*` 是任务执行前的二次安全门槛，不会替代 supervisor 内部参数。视觉中心变化太慢时先适当增大 `auto_visual_min_step_m`、增益或最大步长；出现过冲和来回振荡时先减小增益、最大步长并增大 `auto_visual_goal_min_interval`。

现场调参顺序：先用人工模式验证识别和执行器；再缩短搜索距离验证 supervisor、坐标方向和到达日志；然后调整视觉方向符号、增益、步长和更新间隔；随后调整任务侧居中、到达和动作门槛；最后再依据实测速度和过冲调整 supervisor 的速度、推力、阻尼及主动刹车参数。

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

### 3. 确认基础控制接口

运动控制launch由基础控制系统统一启动，子任务1和子任务3测试人员不需要、也不应该重复启动 `motion_supervisor`。启动子任务前只检查 `/motion/state`、`/status/auv`、`/status/vel` 和 TF 已经正常，并检查 `/cmd/pose/ned` 只能有 `motion_supervisor` 一个发布者：

```bash
rostopic info /cmd/pose/ned
```

### 4. 子任务 1 操作

人工先把机器人放在箭头后方约0.5米并停止手动控制。子任务1 launch 默认同时启动箭头模型、相机和Web；节点启动后先固定点悬停10秒，再执行“前0.5米、左右各0.2米、再前0.3米、左右各0.2米”的搜索路径：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch
```

浏览器可通过 `http://机器人IP:8080` 对照箭头标注画面和中文帧日志。如果箭头模型已经在另一个终端启动：

```bash
roslaunch stereo_depth test_arrow_pose_detection.launch \
  start_camera:=false \
  start_splitter:=false
roslaunch auv_control task3_subtask1_acquire_area.launch start_arrow_model:=false
```

第一次运动测试建议先减小 `search_initial_forward_distance`、`search_lateral_distance` 和 `search_second_forward_distance`，并观察 `/motion/state` 的 `base_position_error`、速度、航向误差和状态切换。粗对准过快时减小 `visual_lateral_gain_m` 或 `visual_max_step_m`；细对准过快时减小 `fine_forward_gain_m`、`fine_lateral_gain_m`、`fine_visual_max_step_m` 或 `fine_yaw_max_step_deg`，也可以增大 `fine_goal_min_interval`。前后或左右方向相反时修改对应视觉方向符号，转向相反时修改 `yaw_correction_sign`。运动速度、推力、阻尼和刹车参数统一在 `motion_supervisor` 中调节。

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

自动模式：先按第3节确认基础控制接口已经就绪，并确认只有 supervisor 发布 `/cmd/pose/ned`，再运行：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch \
  operation_mode:=auto \
  target_color:=yellow
```

自动节点取得 TF、`/status/auv` 和 `/motion/state` 后，先锁存启动固定点。只有该固定点对应的 `HOVER` 和实际停稳门槛持续通过10秒，才按预设的三排前进、左右横移路径搜索。此时可将方框放在机器人前方约0.6米处。

启动后可以通过 `http://机器人IP:8080` 查看方框模型标注视频。如果方框模型已由其他终端启动，运行子任务时增加：

```bash
start_rectangle_model:=false
```

第一次自动测试建议先临时减小搜索距离，并观察 `/motion/state` 的状态、目标、`base_position_error`、速度和 `TX/TY/MZ`。细对准过快时减小 `auto_visual_*_gain_m` 或 `auto_visual_max_step_m`，也可增大 `auto_visual_goal_min_interval`。前后或左右方向相反时修改 `auto_forward_sign`、`auto_lateral_sign`。移动速度、推力、阻尼、刹车和定点强度统一在 `motion_supervisor` 中调整。

### 7. 常用检查命令

查看运动控制输出：

```bash
rostopic echo /cmd/pose/ned
```

查看任务发送给 motion_supervisor 的目标及运动反馈：

```bash
rostopic echo /cmd/motion/goal
rostopic echo /motion/state
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
2. 子任务3的 `manual` 模式只验证识别和执行器链路，不依赖 TF 或状态反馈，也不会发布运动指令；`auto` 模式需要 TF、`/status/auv` 和 `/motion/state`，只发布 supervisor 目标与取消请求。
3. 子任务1先直行搜索；位置识别稳定后只做左右横移粗对准。完整箭头方向稳定后先慢速横移和转向，航向稳定后才允许前后移动；最终前移期间仍实时跟踪完整箭头的位置和方向，并按 `base_link_forward_offset` 的现场标定值使 `base_link` 位于箭头正上方。
4. 子任务 3 的 `target_color` 后续应该来自子任务 2 的 ArUco 编号结果。当前为了调试稳定，先手动通过 launch 参数固定颜色。
5. 子任务3 launch 默认包含并启动 `test_rectangles_detection.launch`；仅当模型已在外部启动时设置 `start_rectangle_model=false`。
6. 使用 `manual` 模式时，应先完成人工移动并停止手动控制；使用 `auto` 模式时，由节点执行悬停、搜索和细对准。
7. 子任务3按前、右、下解释 `/status/auv.linear_velocity[0:3]`，该反馈只用于动作前垂直速度与状态安全核对；水平控制反馈使用 `/motion/state`。
8. 当前最新 `ActuatorControl.msg` 已包含 `mode` 字段；同步消息定义后需要重新执行 `catkin_make`，否则运行环境仍可能加载旧消息类型。
9. 子任务2默认 `start_aruco_model=true`，任务节点退出时会关闭同一launch启动的摄像头、识别节点和Web。如果使用 `start_aruco_model=false` 并在外部终端单独启动模型，该外部摄像头不属于本launch，无法由子任务节点自动关闭。
10. 子任务1或子任务3自动模式运行时，不要再启动其他直接发布 `/cmd/pose/ned` 的测试节点。
