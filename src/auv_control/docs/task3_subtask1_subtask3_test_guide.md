# 任务3子任务1和子任务3测试与控制调参说明

## 1. 文档用途

本文只说明以下两个任务节点的测试流程和运动控制问题：

- 子任务1：识别箭头、搜索、视觉对准航向，并使 `base_link` 到达箭头上方；
- 子任务3：识别指定颜色方框，人工或自动对准后开灯、打开夹爪并结束任务。

识别模型本身的漏检、误检和训练数据问题不在本文展开。本文重点回答：

1. 测试时各节点应该怎样启动；
2. 任务代码怎样调用 `motion_supervisor`；
3. 机器人为什么移动、刹停和进入定点；
4. 发生方向错误、过冲、抖动、不进入 `HOVER` 等问题时应该看什么日志、修改什么参数；
5. 如何判断问题在任务节点、`motion_supervisor`、下位机还是机械执行机构。

对应文件：

```text
src/auv_control/test/test_task3_1_acquire_area.py
src/auv_control/launch/task3_subtask1_acquire_area.launch
src/auv_control/test/test_task3_3_inspect_and_drop.py
src/auv_control/launch/task3_subtask3_inspect_and_drop.launch
src/auv_control/driver/motion_supervisor.py
src/auv_control/driver/motion_supervisor_core.py
src/auv_control/config/motion_supervisor.yaml
```

## 2. 必须先理解的控制边界

### 2.1 任务节点负责什么

子任务1和子任务3自动模式只负责：

- 读取识别结果；
- 读取 `map -> base_link` 当前位姿；
- 根据搜索步骤或图像误差计算目标；
- 发布 `/cmd/motion/goal`；
- 需要停止时发布 `/cmd/motion/cancel`；
- 订阅 `/motion/state` 判断目标是否真正到达；
- 根据任务阶段决定何时进入下一步或失败退出。

任务节点不负责：

- 不直接发布 `/cmd/pose/ned`；
- 不直接计算或发布 `TX/TY/MZ`；
- 不直接实现速度阻尼、停车距离和反向刹车；
- 不直接切换运动阶段的下位机模式；
- 不根据移动距离估算到达时间。

### 2.2 motion_supervisor负责什么

`motion_supervisor` 负责：

- 接收 `map` 坐标系下最终 `base_link` 目标；
- 根据 TF 获得当前位置和航向；
- 使用 `/status/vel` 获得前、右方向速度和航向角速度；
- 使用 `/status/auv.control_mode` 确认下位机模式；
- 把位置误差转换为目标速度；
- 把目标速度与实测速度的差转换为 `TX/TY/MZ`；
- 根据速度、减速度、延迟和余量估算停车距离；
- 主动刹停并在稳定后切换下位机定点模式；
- 发布 `/motion/state` 供任务节点判断当前状态。

### 2.3 三个容易混淆的“模式/状态”

| 字段 | 位置 | 含义 |
| --- | --- | --- |
| `PoseNEDcmd.mode=2` | `/cmd/pose/ned` | 运动和刹停阶段，下位机保持深度，上位机输出 `TX/TY/MZ` |
| `PoseNEDcmd.mode=4` | `/cmd/pose/ned` | 已满足捕获条件，交给下位机定点保持 |
| `AUVData.control_mode` | `/status/auv` | 下位机实际反馈的当前控制模式 |
| `MotionState.state` | `/motion/state` | 上位机状态机编号，不是下位机控制模式 |

特别注意：

- `/motion/state.state=4` 是 `TRANSLATE_BRAKE`，表示上位机正在跟踪或刹车；
- `/status/auv.control_mode=4` 才表示下位机实际进入定点控制；
- `/motion/state.state=8` 是 `HOVER`，表示目标到达并且下位机 `mode=4` 接管已确认。

## 3. 完整控制调用链

```text
识别模型
  -> /arrow/direction 或 /web/detections
  -> 子任务1/子任务3任务节点
  -> 计算 map 下 base_link 绝对目标
  -> /cmd/motion/goal (PoseStamped，任务约5Hz重复发布)
  -> motion_supervisor
       <- map -> base_link TF
       <- /status/vel 速度反馈
       <- /status/auv 下位机模式反馈
  -> /cmd/pose/ned (mode=2 + TX/TY/MZ，或 mode=4 + 零外力)
  -> debug_driver_v2
  -> 下位机

motion_supervisor
  -> /motion/state
  -> 子任务节点核对HOVER、目标一致性、实际误差和速度
  -> 通过后进入任务下一阶段
```

### 3.1 相对位移怎样变成绝对目标

任务需要“向前 `forward` 米、向右 `right` 米”时，会先读取当前航向 `yaw`，再计算一次：

```text
goal_x = current_x + cos(yaw) * forward - sin(yaw) * right
goal_y = current_y + sin(yaw) * forward + cos(yaw) * right
```

其中：

- `forward > 0`：向机器人前方；
- `right > 0`：向机器人右方；
- `right < 0`：向机器人左方。

算出 `map` 绝对目标后，任务会持续发布同一个目标。不能每个循环都使用“当前点+完整偏置”，否则机器人移动后目标也会继续向前移动，永远无法真正到达。

### 3.2 motion_supervisor怎样控制运动

当前控制器以 `20 Hz` 执行统一的 XY 和 yaw 闭环：

1. 计算 `map` 下位置误差和航向误差；
2. 把水平误差方向转换到机器人本体前、右坐标；
3. 根据距离、`xy_position_gain` 和 `xy_max_speed` 生成目标水平速度；
4. 使用 `xy_max_acceleration`、`xy_max_jerk` 限制目标速度变化；
5. 使用目标速度与 `/status/vel` 实测速度的误差计算 `TX/TY`；
6. 使用航向误差生成目标角速度，再根据实测角速度计算 `MZ`；
7. 使用下面的停车距离思想提前进入刹车：

```text
停车距离约等于 速度平方/(2*有效减速度) + 控制延迟*速度 + 停车余量
```

8. 水平位置、航向、水平速度、角速度和目标静止时间连续满足门槛后进入 `CAPTURE`；
9. 发布 `mode=4` 和零外力；
10. 收到本次接管之后的新 `/status/auv.control_mode=4` 反馈，进入 `HOVER`。

当前核心代码允许 XY 和 yaw 同时闭环。子任务1通过分阶段发布目标实现业务要求：粗对准阶段固定 yaw，航向对准阶段禁止前后移动，航向稳定后才开放前后移动。子任务3始终保持启动航向，只改变 XY。

### 3.3 任务为什么还要二次判断到达

收到 `HOVER` 后，两个任务仍会检查：

- `/motion/state` 是否新鲜；
- `MotionState.goal` 是否与任务最新目标一致；
- `base_position_error` 是否在任务容差内；
- `horizontal_speed` 是否足够小；
- `yaw_error/yaw_rate` 是否足够小；
- 子任务1额外检查实际深度；
- 子任务3动作前额外检查 `control_mode=4`、垂直速度、深度和航向。

因此，修改任务侧 `arrival_*` 参数只会改变“任务是否允许进入下一阶段”，不会改变机器人实际运动速度、推力和刹车行为。

## 4. 测试前准备

### 4.1 编译和加载环境

在机器人 ROS 工作空间根目录执行：

```bash
catkin_make_isolated \
  --cmake-args \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=/home/xhy/xhy_env/bin/python3.8

source devel_isolated/setup.bash
```

修改消息定义、Python节点安装项或控制代码后必须重新编译并重新加载环境。

### 4.2 基础控制由系统统一启动

子任务1和子任务3测试人员不单独启动 `motion_supervisor` 或它的运动控制launch。硬件驱动、TF、`/status/auv`、`/status/vel`、双目相机、图像分割和 `motion_supervisor` 由基础系统统一启动并保持运行。

任务侧只使用以下接口：

```text
/cmd/motion/goal
/cmd/motion/cancel
/motion/state
```

不要为了补齐缺失话题额外启动任何运动控制launch，否则可能出现两个控制器同时发布 `/cmd/pose/ned`。模型单独启动时设置 `start_camera:=false start_splitter:=false`，复用基础系统已经运行的相机和分割节点。

### 4.3 没有启动任务前先检查控制链路

```bash
rostopic info /cmd/pose/ned
rostopic hz /status/auv
rostopic hz /status/vel
rostopic hz /motion/state
rostopic echo -n 1 /motion/state
rostopic echo -n 1 /status/auv
rosrun tf tf_echo map base_link
```

必须满足：

- `/cmd/pose/ned` 只有 `/motion_supervisor` 一个发布者；
- `/status/auv`、`/status/vel`、`/motion/state` 持续更新；
- `map -> base_link` TF 持续更新且数值有限；
- `MotionState.startup_complete=true`；
- 无任务目标时最终应进入 `HOVER`；
- `/status/auv.control_mode=4`；
- `MotionState.state` 不能是 `SAFE`。

如果此处没有通过，不要启动自动任务，也不要先放宽任务超时。先修复 TF、速度反馈、状态反馈、下位机模式确认或重复发布者问题。

### 4.4 建议记录测试数据

子任务1：

```bash
rosbag record -O task3_subtask1_control.bag \
  /cmd/motion/goal /cmd/motion/cancel /motion/state /motion/diagnostics \
  /cmd/pose/ned /status/auv /status/vel /tf /tf_static /arrow/direction
```

子任务3：

```bash
rosbag record -O task3_subtask3_control.bag \
  /cmd/motion/goal /cmd/motion/cancel /motion/state /motion/diagnostics \
  /cmd/pose/ned /status/auv /status/vel /tf /tf_static \
  /web/detections /cmd/actuator /finished
```

`motion_supervisor` 还会把每个控制周期的数据写到：

```text
~/.ros/auv_logs/motion_supervisor/
```

排查过冲、刹车和推力问题时，CSV比只看终端日志更可靠。

## 5. 子任务1测试流程

### 5.1 第一步：单独验证箭头模型链路

终端2：

```bash
roslaunch stereo_depth test_arrow_pose_detection.launch \
  start_camera:=false \
  start_splitter:=false
```

检查：

```bash
rostopic hz /arrow/direction
rostopic echo /arrow/direction
```

浏览器查看：

```text
http://机器人IP:8080
```

最新链路使用YOLO Pose关键点模型，通过 `tip、tail_left、tail_right` 计算箭头方向。至少确认 `valid`、`confidence`、`center`、`bbox` 和 `angle_deg` 字段能够持续出现；其中 `confidence` 是检测框与三个关键点置信度的最小值。模型只识别到位置但没有有效方向时，任务可以退出搜索和进行左右粗对准，但不会进入航向细对准。

### 5.2 第二步：低风险验证搜索方向和运动状态

第一次运动测试建议临时缩短距离和视觉步长。这一步只验证控制方向和状态切换，不验证最终任务几何精度。

终端3：

```bash
roslaunch auv_control task3_subtask1_acquire_area.launch \
  start_arrow_model:=false \
  initial_hover_seconds:=3.0 \
  search_initial_forward_distance:=0.10 \
  search_lateral_distance:=0.05 \
  search_second_forward_distance:=0.10 \
  visual_max_step_m:=0.03 \
  fine_visual_max_step_m:=0.01 \
  fine_yaw_max_step_deg:=1.0 \
  fine_goal_min_interval:=1.0 \
  base_link_forward_offset:=0.10 \
  final_hold_seconds:=3.0
```

观察以下行为：

1. 启动后目标固定在第一次锁存的位置，机器人漂移时会返回该点；
2. 新鲜 `HOVER` 连续保持3秒后才开始搜索；
3. 前进目标应位于机器人前方；
4. 左移和右移方向应正确；
5. 每个搜索目标到达后才进入下一个点；
6. 任何一帧有效箭头位置出现后应立即发布 cancel，不再继续搜索路径；
7. cancel后应经历刹车、`CAPTURE`、`HOVER`，再定点累计3帧位置。

方向验证完成后恢复正式距离和时间。

### 5.3 第三步：验证粗对准

正式默认行为：

```text
固定启动点悬停10秒
-> 前进0.50m
-> 第一层左0.20m、右0.20m、回中
-> 再前进0.30m
-> 第二层左0.20m、右0.20m
```

搜索中出现第一帧有效位置后：

```text
立即cancel
-> 停稳并进入HOVER
-> 定点重新累计3帧位置
-> 进入左右粗对准
```

粗对准阶段：

- 只根据 `u` 误差做左右横移；
- `v` 误差只记录，不做前后移动；
- yaw目标保持当前固定航向；
- 每个粗对准小步必须真正到达后才允许生成下一步；
- 箭头完整留在图像中、中心左右稳定5帧并且方向稳定3帧后才进入下一步。

注意：粗对准阶段仍可能看到小量 `MZ`，这是 supervisor 为了保持固定 yaw，不代表任务已经开始按箭头方向转向。如果出现持续的大幅转向，应检查任务发布的目标 yaw、TF yaw和 `/motion/state.yaw_error`。

### 5.4 第四步：验证航向和前后细对准

粗对准通过后：

```text
cancel并定点
-> 再次复核完整箭头方向
-> 禁止前后移动，只慢速横移和转向
-> 航向及左右位置连续3帧通过并进入HOVER
-> 开放前后移动，慢速完成前后、左右和航向对准
-> 中心和方向连续5帧通过并进入HOVER
```

箭头方向误差计算为：

```text
方向误差 = yaw_correction_sign
         * normalize(camera_forward_angle_deg - 模型angle_deg)
```

默认 `camera_forward_angle_deg=90`。现场标定方法：让机器人航向与箭头实际方向一致，记录模型稳定输出的 `angle_deg`，该稳定值应接近 `camera_forward_angle_deg`。

### 5.5 第五步：验证base_link补偿和任务结束

摄像头位于箭头中心上方不等于 `base_link` 位于箭头上方。细对准通过后，代码只计算一次：

```text
沿机器人当前前方前移 base_link_forward_offset
```

默认 `base_link_forward_offset=0.35m`。前移过程中：

- 继续实时观察完整箭头；
- 只根据 `u` 误差修正左右位置；
- 继续修正航向；
- `v` 误差只记录，因为前后目标已经由0.35m安装偏置确定；
- 完整箭头、左右位置和航向连续5帧通过，且目标进入 `HOVER` 后进入最终保持；
- 最终 `HOVER` 和深度连续保持10秒，任务成功并发布 cancel。

如果最后机器人停在箭头前方，增大 `base_link_forward_offset`；如果越过箭头，减小该参数。该参数应依据相机光心、`base_link` 和目标期望对齐点的实际安装尺寸标定，不应该通过修改视觉增益补偿。

### 5.6 子任务1正常状态顺序

```text
启动定点悬停
-> 固定路径搜索箭头
-> 主动刹停并等待定点接管
-> 定点重新识别箭头
-> 箭头位置左右粗对准
-> 主动刹停
-> 定点确认箭头方向
-> 持续看箭头并慢速对齐航向
-> 航向对齐后慢速前后居中
-> base_link前移并持续视觉跟踪
-> 最终定点保持
-> finished
```

## 6. 子任务3测试流程

### 6.1 第一步：单独启动方框模型

终端2：

```bash
roslaunch stereo_depth test_rectangles_detection.launch \
  start_camera:=false \
  start_splitter:=false
```

检查：

```bash
rostopic hz /web/detections
rostopic echo /web/detections
```

浏览器查看：

```text
http://机器人IP:8080
```

### 6.2 第二步：先测人工模式

人工模式不创建 TF 运动接口，也不发布 `/cmd/motion/goal`。人工把机器人移动到目标方框位置后运行：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch \
  start_rectangle_model:=false \
  operation_mode:=manual \
  target_color:=red \
  min_confidence:=0.35
```

预期流程：

```text
按颜色和置信度过滤
-> 连续5帧稳定
-> 等待hold_seconds
-> 点亮对应颜色灯并打开夹爪
-> 保持open_seconds
-> 关闭夹爪并熄灯
-> 发布/finished
```

人工模式会真实发送执行器指令。测试前必须确认 `clamp_open`、`clamp_closed` 和三色灯协议值正确。

### 6.3 第三步：低风险验证自动搜索方向

第一次自动运动测试可临时使用：

```bash
roslaunch auv_control task3_subtask3_inspect_and_drop.launch \
  start_rectangle_model:=false \
  operation_mode:=auto \
  target_color:=red \
  auto_initial_hover_seconds:=3.0 \
  auto_search_first_forward_distance:=0.10 \
  auto_search_second_forward_distance:=0.05 \
  auto_search_third_forward_distance:=0.05 \
  auto_search_lateral_distance:=0.05 \
  auto_visual_forward_gain_m:=0.05 \
  auto_visual_lateral_gain_m:=0.05 \
  auto_visual_max_step_m:=0.01 \
  auto_visual_goal_min_interval:=1.0
```

这一步重点确认：

- 启动悬停点只锁存一次；
- 前进、左移和右移方向正确；
- 每个搜索步骤只产生一个绝对目标；
- 搜索步骤必须等最新目标的 `HOVER` 和任务实际到达门槛通过后才切换；
- 识别到连续3帧稳定方框后立即 cancel；
- 细对准根据图像 `v` 误差前后移动，根据 `u` 误差左右移动；
- 方框连续5帧居中后锁定最终点，不再继续叠加视觉目标；
- 只有稳定 `HOVER` 和动作门槛全部通过才会开灯、打开夹爪。

### 6.4 正式自动流程

```text
固定启动点稳定悬停10秒
-> 前进0.30m，左0.20m，右0.20m
-> 前进0.20m，左0.20m，右0.20m
-> 前进0.10m，左0.20m，右0.20m
-> 搜索中连续3帧稳定识别方框
-> cancel并等待HOVER
-> 根据中心像素误差发布前后、左右小步
-> 连续5帧居中
-> 锁定最终固定点并等待稳定HOVER
-> 开对应颜色灯并打开夹爪
-> 定点保持3秒
-> 关闭夹爪和灯
-> 发布/finished并cancel
```

搜索路径全部完成仍未识别到目标时，子任务3会保持最后定点继续等待，直到 `max_wait_seconds` 超时。目标在细对准中丢失超过 `detection_timeout` 时，会刹停并返回当前搜索步骤。

子任务3 launch 中任务节点没有设置 `required=true`。任务节点结束后，如果模型仍由同一个 launch 启动，roslaunch可能继续运行；看到 `/finished` 且夹爪和灯关闭后，可在该终端按 `Ctrl+C` 关闭剩余模型和网页节点。

## 7. 任务侧参数怎样调

### 7.1 先区分“目标生成”和“实际控制”

| 参数类型 | 能改变什么 | 不能改变什么 |
| --- | --- | --- |
| 搜索距离 | 每一阶段目标的位置 | 实际速度和推力 |
| 视觉增益/步长 | 每次视觉目标移动多少米 | supervisor速度闭环性能 |
| 视觉更新间隔 | 多久允许生成一个新目标 | 单个目标的底层控制频率 |
| 中心/到达容差 | 任务何时认为条件通过 | mode=2阶段的推力和刹车 |
| supervisor速度/推力参数 | 实际速度、响应和刹停 | 识别中心和任务状态机逻辑 |

### 7.2 子任务1关键参数

| 现象 | 优先调整参数 | 调整方向 |
| --- | --- | --- |
| 搜索范围过大或过小 | `search_initial_forward_distance`、`search_lateral_distance`、`search_second_forward_distance` | 按现场初始位置修改 |
| 粗对准左右方向相反 | `visual_lateral_sign` | `1.0`与`-1.0`互换 |
| 细对准前后方向相反 | `visual_forward_sign` | `1.0`与`-1.0`互换 |
| 航向修正方向相反 | `yaw_correction_sign` | `1.0`与`-1.0`互换 |
| 箭头对正时仍显示固定角度误差 | `camera_forward_angle_deg` | 设置为对正时模型稳定输出角度 |
| 粗对准太慢 | `visual_lateral_gain_m`、`visual_min_step_m`、`visual_max_step_m` | 小幅增大 |
| 粗对准来回越过中心 | 上述增益和最大步长、`visual_goal_min_interval` | 减小增益/最大步长，增大间隔 |
| 航向调整过快导致箭头离开画面 | `fine_yaw_max_step_deg`、`fine_goal_min_interval` | 减小角度步长，增大间隔 |
| 细对准平移过快 | `fine_forward_gain_m`、`fine_lateral_gain_m`、`fine_visual_max_step_m` | 减小 |
| 细对准几乎不移动 | `fine_visual_min_step_m`、增益、最大步长 | 先略增最小步长，再看是否受死区影响 |
| 箭头完整但始终不进入方向阶段 | `full_arrow_edge_margin_px`、`full_arrow_min_bbox_width_px/height_px` | 根据日志中的“不完整原因”小幅放宽 |
| 方向计数反复归零 | `stable_angle_tolerance_deg`、`heading_stable_detection_count` | 先检查方向数据，再适度增大角度抖动容差 |
| 视觉中心难以连续通过 | `center_tolerance_u_px/v_px`、`center_stable_detection_count` | 先略放宽像素容差，不要先放宽运动到达条件 |
| 最终base_link停在箭头前方/后方 | `base_link_forward_offset` | 前方未到增大，越过则减小 |
| 目标丢失时太敏感 | `visual_loss_cancel_seconds` | 小幅增大，但必须不大于 `detection_timeout` |
| 最终悬停经常计时清零 | `arrival_*`、`max_depth_error` | 根据“到达判定”中第一个未通过项调整 |
| 对地距离保护频繁改目标 | `min_ground_clearance`、`ground_clearance_goal_update_threshold` | 先核对高度单位和符号，再调整更新阈值 |

### 7.3 子任务3关键参数

| 现象 | 优先调整参数 | 调整方向 |
| --- | --- | --- |
| 自动搜索距离不合适 | `auto_search_*_distance` | 按场地缩短或增加 |
| 前后修正方向相反 | `auto_forward_sign` | `1.0`与`-1.0`互换 |
| 左右修正方向相反 | `auto_lateral_sign` | `1.0`与`-1.0`互换 |
| 方框靠近中心太慢 | `auto_visual_forward_gain_m`、`auto_visual_lateral_gain_m`、`auto_visual_min_step_m` | 小幅增大 |
| 反复穿过中心 | 视觉增益、`auto_visual_max_step_m`、`auto_visual_goal_min_interval` | 减小增益/最大步长，增大间隔 |
| 日志显示小步发布但机器人几乎不动 | `auto_visual_min_step_m` | 略增，并核对 supervisor 的位置死区 |
| 画面中心有固定偏置 | `auto_target_center_u_ratio/v_ratio` | 按相机安装偏置标定，不一定必须是0.5 |
| 居中计数反复清零 | `auto_center_tolerance_u_px/v_px` | 先看像素误差，再小幅放宽 |
| 搜索阶段3帧难以通过 | `auto_search_stable_detection_count`、`stable_center_tolerance_px`、`stable_area_tolerance_ratio` | 先看逐帧日志，再调整稳定门槛 |
| 已居中但不执行夹爪 | `arrival_*`、`auto_action_*`、`status_timeout` | 看动作放行日志中第一个未通过项 |
| `/status/auv`速度单位不为m/s | `status_linear_velocity_scale` | 依据协议换算，例如毫米每秒需乘0.001 |
| 总流程等待太久或太短 | `max_wait_seconds`、`detection_timeout` | 依据模型实际频率设置，不用于掩盖断流 |

## 8. motion_supervisor参数怎样调

参数文件：

```text
src/auv_control/config/motion_supervisor.yaml
```

修改后必须重启 `motion_supervisor`。一次只调整一类参数并保存对应 rosbag、CSV和视频时间点。

### 8.1 速度规划参数

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `xy_max_speed` | `0.30` | 最大水平参考速度，m/s |
| `xy_position_gain` | `1.00` | 位置误差到目标速度的比例 |
| `xy_max_acceleration` | `0.20` | 水平参考加速度限制，m/s² |
| `xy_max_jerk` | `0.40` | 水平参考加加速度限制 |
| `yaw_max_rate_deg_s` | `25.0` | 最大航向角速度 |
| `yaw_position_gain` | `1.50` | 航向误差到目标角速度的比例 |
| `yaw_max_acceleration_deg_s2` | `20.0` | 航向角加速度限制 |
| `yaw_max_jerk_deg_s3` | `60.0` | 航向角加加速度限制 |

第一次水池运动如果明显过快，可先把 `xy_max_speed` 降到 `0.05~0.10m/s`、`yaw_max_rate_deg_s` 降到 `5~10deg/s`，并相应降低加速度和jerk。确认方向、速度反馈和刹车都正确后再逐步增加。

### 8.2 跟踪推力参数

| 参数组 | 作用 |
| --- | --- |
| `max_tx_positive/negative` | 前后正常运动推力上限 |
| `max_ty_positive/negative` | 左右正常运动推力上限 |
| `max_mz_positive/negative` | 航向正常运动力矩上限 |
| `kv_x_positive/negative` | 前后速度误差到TX的增益 |
| `kv_y_positive/negative` | 左右速度误差到TY的增益 |
| `kp_yaw_positive/negative` | 航向角速度误差到MZ的增益 |
| `force_slew_per_cycle` | 正常运动时每周期力变化上限 |

调参判断：

- 参考速度已经很大但实测速度始终偏小，且力未到上限：增加对应 `kv/kp`；
- 力经常顶到上限仍无法达到速度：检查机械、推进器和环境阻力，再考虑提高对应上限；
- 小误差时输出跳变明显：降低增益或 `force_slew_per_cycle`；
- 正负方向表现不同：分别调整 positive/negative，不能只改一侧后直接类推另一侧。

### 8.3 主动刹车参数

| 参数组 | 作用 |
| --- | --- |
| `brake_gain_tx/ty/mz_*` | 实测速度到反向刹车力的增益 |
| `brake_max_tx/ty/mz_*` | 刹车阶段力上限 |
| `brake_acceleration_tx/ty_*` | 控制器假设的有效减速度 |
| `angular_brake_acceleration_mz_*` | 控制器假设的有效角减速度 |
| `control_delay` | 反馈、通信和执行综合延迟 |
| `brake_margin_tx/ty_*` | 水平提前停车附加余量 |
| `yaw_brake_margin_*` | 航向提前刹转附加角度 |

调参判断：

- 总是刹车太晚：增大对应 `brake_margin_*` 或 `control_delay`，也可适当减小控制器假设的有效减速度，使计算出的停车距离更长；
- 已经提前刹车但反向力不够：增加对应 `brake_gain_*` 或 `brake_max_*`；
- 刹车过早并且到目标前明显停住：减小停车余量或延迟估计；
- 刹车阶段大幅反向冲过头：减小刹车增益或刹车力上限；
- 参数后缀表示最终输出力的正负方向，必须结合CSV中的速度方向和 `TX/TY/MZ` 符号判断，不能只凭“前进/后退”文字猜测。

### 8.4 捕获和HOVER参数

| 参数 | 当前值 | 作用 |
| --- | ---: | --- |
| `capture_radius` | `0.15m` | 允许进入mode=4接管的水平误差 |
| `capture_exit_radius` | `0.25m` | 接管等待期间允许保留的水平误差 |
| `control_center_hold_tolerance` | `0.03m` | 控制中心位置死区 |
| `horizontal_speed_threshold` | `0.015m/s` | supervisor判定停稳的水平速度 |
| `yaw_tolerance_deg` | `5deg` | supervisor航向捕获容差 |
| `yaw_rate_threshold_deg_s` | `0.3deg/s` | supervisor停转门槛 |
| `stable_frames` | `5` | 捕获条件连续通过帧数 |
| `goal_static_capture_seconds` | `0.80s` | 目标保持不变多久后才允许捕获 |
| `mode_ack_timeout` | `1.0s` | 等待下位机mode=4确认的超时 |

如果长期停在 `CAPTURE`：

1. 先看 `base_position_error`、速度、航向误差和角速度是哪一项未通过；
2. 再看 `/status/auv.control_mode` 是否在接管之后反馈为4；
3. 如果是传感器噪声略高于门槛，可小幅放宽对应阈值；
4. 不要为了让状态变成 `HOVER` 一次性大幅放宽全部参数。

如果已经 `HOVER` 但任务不进入下一阶段，问题通常在任务侧更严格的 `arrival_*`、目标匹配或深度/动作门槛，不是 supervisor 的捕获参数。

## 9. 常见问题定位表

| 现象 | 先看什么 | 可能原因 | 优先处理 |
| --- | --- | --- | --- |
| 没有 `/motion/state` | `rosnode list`、启动日志 | supervisor未启动或编译失败 | 修复启动和编译 |
| 一直 `SAFE` 且提示等待反馈 | `/status/vel`、TF | 速度或TF未更新 | 修复反馈链路，不放宽任务超时 |
| `/cmd/pose/ned`有多个发布者 | `rostopic info /cmd/pose/ned` | 旧任务或测试节点仍在运行 | 只保留supervisor |
| 任务不发布 `/cmd/motion/goal` | 任务阶段和识别帧日志 | 仍在悬停、识别计数或保护状态 | 按状态日志检查 |
| goal已发布但MotionState.goal不变 | supervisor日志 | 目标frame错误、四元数无效或节点接错 | 检查拒绝目标日志 |
| `TRANSLATE`且力为0 | 误差、参考速度、实际速度 | 误差在死区、速度反馈异常或增益为0 | 查CSV和参数 |
| 力非0但机器人不动 | `/cmd/pose/ned`与硬件状态 | 驱动、推进器、机械阻力或通信 | 查下位机，不改视觉参数 |
| 机器人移动方向与图像修正相反 | goal目标方向 | 视觉方向符号错误 | 改任务的forward/lateral sign |
| goal方向正确但机器人实体反向 | `/cmd/pose/ned`力和实测速度 | 底层轴映射或有效性矩阵符号错误 | 查驱动/推进器映射 |
| 接近目标时过冲 | CSV速度、刹车开始距离 | 速度过高、减速度估计偏大、余量小、刹车力弱 | 先降速度，再调停车模型和刹车力 |
| 在目标附近来回振荡 | goal是否频繁变、速度和力 | 视觉小步太密、增益大或控制死区小 | 先调任务视觉步长/间隔，再调控制器 |
| 一直进不了HOVER | MotionState误差和reason | 速度/角速度/位置/航向/目标静止时间未通过 | 只调整未通过项 |
| `CAPTURE`后进入SAFE | `/status/auv.control_mode` | mode=4没有及时确认 | 查下位机模式反馈和`mode_ack_timeout` |
| HOVER后任务仍等待 | “到达判定”日志 | 最新目标不匹配或任务门槛更严格 | 调任务门槛或修复旧goal问题 |
| 子任务1粗对准时大幅转向 | goal yaw与yaw_error | 固定航向目标错误或TF航向异常 | 检查目标yaw和TF，不改箭头增益 |
| 子任务1转向后箭头离开画面 | 视频、yaw小步 | 转向太快或横移未跟上 | 降低fine yaw/平移步长和supervisor角速度 |
| 子任务1最终位置有固定前后偏差 | 实际base_link与箭头关系 | 安装偏置未标定 | 调`base_link_forward_offset` |
| 子任务3已居中但不开夹爪 | 动作放行日志 | mode、速度、深度、航向或目标到达未通过 | 调第一个未通过项 |
| 模型或分割节点反复重启 | `rosnode list`和launch日志 | begin与模型launch重复启动同名节点 | 按本文三终端方式启动 |

## 10. 推荐调参顺序

1. 不启动任务，确认 TF、`/status/vel`、`/status/auv` 和 supervisor 启动定点正常。
2. 使用小距离目标确认机器人前、后、左、右和yaw实际方向正确。
3. 降低 supervisor 最大速度、角速度和加速度，确认主动刹车能停住。
4. 单独运行模型，确认任务收到的中心、bbox和方向字段连续。
5. 使用缩短搜索距离测试任务状态切换和 cancel/HOVER。
6. 先校准视觉方向符号，再调整视觉增益、最小/最大步长和更新间隔。
7. 子任务1标定 `camera_forward_angle_deg` 和 `base_link_forward_offset`。
8. 根据“到达判定”日志调整任务侧位置、速度、航向和深度门槛。
9. 最后依据CSV实测数据精调 supervisor 推力、阻尼、停车距离和刹车参数。
10. 恢复正式搜索距离、10秒悬停和正式动作时间，完成整套流程测试。

一次测试只改一类参数。每次记录：参数文件版本、目标颜色/箭头位置、任务阶段、视频时间点、ROS日志、rosbag和supervisor CSV。这样才能判断修改是否真正改善，而不是偶然通过。
