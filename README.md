# 小黄鱼项目简介

## 项目背景

“小黄鱼”是一款面向浅海与水下结构探测的自主式水下机器人（AUV），具备较强的机动性和任务适应性。项目围绕低成本、高精度、视觉辅助导航与识别展开，结合目标定位与姿态控制等关键技术，旨在完成对水下二维码目标的识别、跟踪及定姿作业。

## 技术核心

- **视觉目标识别与定位**：

  - 利用双目相机结合YOLOv8分割模型，实现对水下二维码类目标（如标识点、维修接口等）的检测与分割；
  - 基于立体视觉原理与像素匹配，计算目标在相机坐标系下的3D位置，进而转换到载体与导航坐标系中。
- **航行器姿态控制**：

  - 通过 IMU 传感器获取载体在导航坐标系下的欧拉角；
  - 实现二维码坐标系的对准目标向量（如 z 轴）与载体本体坐标系（如 x 轴）的空间对齐；
  - 计算姿态变换所需的旋转角度，指导 AUV 进行精确对接或目标朝向控制。
- **坐标系转换与对齐策略**：

  - 采用 `ZYX` 欧拉角变换序列处理导航坐标系与相机/二维码坐标系之间的空间变换；
  - 利用旋转矩阵实现各坐标系之间的姿态解析和向量投影；
  - 支持对目标向量（如二维码 z 轴）在导航系的投影分析与角度计算，辅助控制系统调整航向。

## 关键成果

- 构建完整的坐标系转换链：二维码坐标系 → 相机坐标系 → 载体坐标系 → 导航坐标系；
- 实现二维码姿态标定与投影方向角度估计；
- 支持从视觉目标获取到航行器姿态控制闭环链路的构建。

## 应用前景

“小黄鱼”系统具备可扩展的目标识别、导航控制与环境感知能力，适用于：

- 海底构造巡检；
- 水下管线识别与定位；
- 自主作业中的目标引导与姿态校准任务。

## quick start

`git clone git@github.com:JiaqiMi/xhy2.git`
`cd xhy2`
`catkin_make`

## 分支说明

- main 主分支
- main-backup 实际部署分支备份
- control 控制更新分支
- lxy
- WolfFoox
- rice-local

## 视觉部分说明

识别“红色圆形”:

```bash
roslaunch stereo_depth test_red_circle_detection.launch
```

关键话题：

```text
/yolo_unified/target_center      geometry_msgs/PointStamped
/obj/target_message              auv_control/TargetDetection
```

---

识别“红色管线”:

```bash
roslaunch stereo_depth test_line_detection.launch
```

关键话题：

```text
/yolo_unified/line_bbox          geometry_msgs/LineBox
/obj/line_message                auv_control/TargetDetection3
```

---

识别巡线任务中的“三种形状”:

```bash
roslaunch stereo_depth test_shapes_detection.launch
```

关键话题：

```text
/yolo_unified/target_center      geometry_msgs/PointStamped
/obj/target_message              auv_control/TargetDetection

---

识别“黑色箭头”:

```bash
roslaunch stereo_depth test_arrow_detection.launch
```

关键话题：

```text
/yolo_unified/target_bbox        stereo_depth/BoundingBox
/arrow/direction                 std_msgs/String

---

识别“矩形框”:

```bash
roslaunch stereo_depth test_rectangles_detection.launch
```

关键话题：

```text
/yolo_unified/target_center      geometry_msgs/PointStamped
/obj/target_message              auv_control/TargetDetection

---

识别“aruco”:

```bash
roslaunch stereo_depth test_aruco_detection_fisheye.launch
```

关键话题：

```text
/yolo_unified/target_center      geometry_msgs/PointStamped
/obj/target_message              auv_control/TargetDetection

---


录制bag:

```bash

roslaunch stereo_depth test_rosbag.launch \
  mode:=record

roslaunch stereo_depth test_rosbag.launch \
  mode:=play \
  bag_file:=/home/xhy/xhy_records/stereo_input_0.bag
```
---


## hsx 控制侧

开启tf和map

```bash
roslaunch auv_control begin.launch
```

初始化世界坐标原点

```bash
roslaunch auv_control reset_world origin.launch
```

关键话题

```bash
/cmd/pose/ned (PoseNEDcmd.msg)
/cmd/actuator (ActuatorControl.msg)
/status/actuator (ActuatorControl.msg)
/status/power (SensorStatus.msg)
/status/auv (AUVData.msg)
```
