# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

"小黄鱼"——基于 ROS 的 AUV（自主水下机器人）上位自动控制程序。通过 TCP 与 AUV 各硬件模块通信，实现数据采集、状态监控与运动控制。

源码仅关注 `src/auv_control/`，其余包（`oculus_ros/`、`stereo_depth/`、`stereo_splitter/`、`yolo_bridge/`）为感知相关，不在本项目的维护范围内。

## 构建与运行

```bash
# 构建
cd xhy2
catkin_make

# 启动完整驱动链路
roslaunch auv_control begin.launch
```

## 代码架构

```
src/auv_control/
├── driver/           # 硬件通信驱动（核心维护目标）
│   ├── debug_driver.py
│   ├── debug_driver_v2.py
│   ├── nav_driver.py
│   ├── sensor_status_node.py
│   └── sensor_actuator_node.py
├── scripts/          # 应用层 Python 节点
│   ├── [Control 层]  # 状态控制、键盘控制
│   ├── [Task 层]     # task1~task4 v2 任务执行节点
│   └── [Support 层]  # 数据保存、TF 变换、地图初始化、VO 融合
├── msg/              # 自定义 ROS 消息定义
├── launch/           # 启动文件
└── test/             # 测试脚本（不直接参与运行）
```

### Driver 层（核心）

驱动脚本统一位于 `src/auv_control/driver/`，负责通过 TCP 与 AUV 硬件模块通信。

所有 Driver 遵循特定的二进制协议。

#### `driver/debug_driver.py`
- **连接**：TCP `192.168.1.115:5063`
- **功能**：8Hz 发送 54 字节 ROV 扩展控制报文，同时接收并解析 AUV 回传数据
- **发布**：`/debug_auv_data` (AUVData)
- **订阅**：`/auv_control` (AUVPose)

#### `driver/debug_driver_v2.py`
- **连接**：TCP `192.168.1.115:5063`
- **功能**：支持定深(mode=2)、定深定向(mode=3)、定点(mode=4)三种模式，发送 54 字节 ROV 扩展控制帧
- **发布**：`/debug_auv_data` (AUVData)
- **订阅**：`/auv_control_cmd` (AUVControlCmd)

#### `driver/sensor_status_node.py`
- **连接**：TCP `192.168.1.115:5064`
- **功能**：纯 STATUS 帧接收节点，解析两路电源（电压、电流、功率）并发布，不发送任何下行控制帧
- **发布**：`/sensor_status` (SensorStatus)

#### `driver/sensor_actuator_node.py`
- **连接**：TCP `192.168.1.115:5064`
- **功能**：执行器控制与反馈节点，下发 CAMERA_LIGHT_SET + ACTUATOR_SET 指令，接收 ACK 和 ACTUATOR_FB 上行帧
- **发布**：`/auv_actuator_status` (ActuatorControl)
- **订阅**：`/auv_actuator_control` (ActuatorControl)

#### `driver/nav_driver.py`
- **连接**：TCP `192.168.1.115:5066`
- **功能**：接收 140 字节导航报文，解析 INS（经纬度/高度/速度/姿态）、GPS、DVL、IMU（陀螺/加速度/温度）、深度、时间、状态标志位
- **发布**：`/nav` (NavData)
- **可选**：支持原始报文和解析数据保存为 JSONL

### Control 层

- `state_control.py` — 任务状态机，管理 5 个任务的自动/手动切换与超时控制
- `keyboard_control.py` — 键盘输入节点，手动切换任务阶段与自动运行开关

### Task 层（v2）

- `task_v2_common.py` — 任务公共驱动模块，封装 TF 位姿获取、目标拆分与发布、执行器控制、亮灯/闪灯/旋转等通用功能
- `task1_v2.py` — 主管道检修：巡线前进，识别黄色泄漏标记（闪红灯）和黑色污染标记（闪绿灯+旋转），完成后移动到终点
- `task2_v2.py` — 环境监测与水样采集：移动到采样点，执行采水动作，返回起始区域上浮
- `task3_v2.py` — 识别指定管段并投放信标：获取任务编号，识别彩色管段，释放高尔夫球信标
- `task4_v2.py` — 返回起始区域并上浮
- `task1_v2_black_test.py` — Task1 黑色方形动作单独测试
- `task1_v2_line_test.py` — Task1 巡线运动功能单独测试
- `task1_v2_yellow_test.py` — Task1 黄色图形动作单独测试

### Support 层

- `data_saver.py` — 统一数据保存节点，将 debug/sensor/nav 消息写入 JSONL
- `auv_tf_handler.py` — 坐标系转换（base_link → map），支持 `/target`（PoseStamped）和 `/target_cmd`（AUVControlCmd）两条并行控制链路
- `static_tf_broadcaster.py` — 静态 TF 广播（base_link → imu/scan/hand/camera）
- `map_initer.py` — 地图坐标系原点初始化，取前 50 个惯导有效数据计算世界原点
- `vo_nav_fusion.py` — 视觉里程计与导航数据弱耦合融合（V1），发布 `/fusion/pose` 和 `/fusion/odom`

### 自定义消息

| 消息文件 | 用途 |
|---------|------|
| `AUVData.msg` | AUV 完整状态（位姿、传感器、电机力矩、速度） |
| `AUVPose.msg` | AUV 位姿（经纬度、深度、姿态角、速度） |
| `AUVMotor.msg` | 6 自由度力/力矩（TX/TY/TZ/MX/MY/MZ） |
| `AUVSensor.msg` | 传感器状态（温度、电压、电流、漏水报警等） |
| `AUVTime.msg` | UTC 时间 |
| `NavData.msg` | 完整导航数据（INS/GPS/DVL/IMU） |
| `SensorStatus.msg` | 两路电源状态（电压/电流/功率） |
| `ActuatorControl.msg` | 执行器控制（补光灯、舵机、推杆、指示灯） |
| `AUVControlCmd.msg` | V2 控制命令（mode + target + force） |
| `Keyboard.msg` | 键盘输入（run/mode） |
| `TargetDetection.msg` | 单目标检测结果 |
| `TargetDetection3.msg` | 三目标检测结果 |

### 启动文件

| 文件 | 用途 |
|------|------|
| `begin.launch` | 主启动：加载参数，启动 debug_driver、sensor_status_node、sensor_actuator_node、map_initer、static_tf、auv_tf_handler |
| `param.launch` | ROS 参数配置 |
| `static_tf.launch` | 静态 TF 独立启动 |
| `data_collection.launch` | 数据采集启动 |
| `test_debug_v2.launch` | debug_driver_v2 测试启动 |
| `task1_v2_test_launch.launch` | Task1 v2 完整测试 |
| `task1_v2_black_test_launch.launch` | Task1 黑色标记动作测试 |
| `task1_v2_line_test_launch.launch` | Task1 巡线测试 |
| `task1_v2_yellow_test_launch.launch` | Task1 黄色标记动作测试 |
| `task3_subtask1_acquire_area.launch` | Task3 子任务1：到达任务获取区域 |
| `task3_subtask2_get_task.launch` | Task3 子任务2：获取任务编号 |
| `task3_subtask3_inspect_and_drop.launch` | Task3 子任务3：识别管段并投放 |

## 修改规范

**每次修改必须在文件头部"记录"段落中标注修改内容**，格式沿用现有约定：

```python
"""
名称：xxx.py
功能：xxx
作者：xxx
监听：xxx          # 订阅的 topic
发布：xxx          # 发布的 topic
记录：
YYYY.M.D HH:MM
    修改描述1
YYYY.M.D HH:MM
    修改描述2
"""
```

日期使用当天的实际日期，描述简洁说明修改内容。

## 通信协议要点

- 所有 AUV 通信报文以 `FE FE` 为头、`FD FD` 为尾
- 校验方式为异或校验（XOR），校验范围不含尾部的 `FD FD`
- 数值多以大端序（big-endian）传输，部分字段有 `×100` 或 `×1e7` 的缩放
- nav 驱动使用独立的协议格式：`AA 55 5A A5` 帧头，140 字节定长，累加和校验
- sensor 执行器下行协议：`CAMERA_LIGHT_SET` (cmd=0x10) 控制补光灯/指示灯，`ACTUATOR_SET` (cmd=0x30) 控制舵机/推杆
- debug_driver_v2 控制帧偏移 44 固定为 0x00，力/力矩支持 0-10000 原始值
