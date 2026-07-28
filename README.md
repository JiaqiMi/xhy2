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

### 一键运行

#### 激活环境

```bash
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel_isolated/setup.bash
source /home/xhy/xhy_env/bin/activate

export PYTHONPATH=/home/xhy/catkin_ws/devel_isolated/cv_bridge/lib/python3/dist-packages:$PYTHONPATH
```
这部分已经放到~/.bashrc中，理论上不需要重复执行。

#### 检测红色圆形

```bash
roslaunch stereo_depth test_red_circle_detection.launch
```

#### 检测红色管线

```bash
roslaunch stereo_depth test_line_detection.launch
```

#### 检测多类别形状

```bash
roslaunch stereo_depth test_shapes_detection.launch
```

#### 检测黑色箭头

```bash
roslaunch stereo_depth test_arrow_detection.launch
```

#### 检测三类矩形框

```bash
roslaunch stereo_depth test_rectangle_detection.launch
```

## 7. Shapes 多类别任务

### 7.1 启动

```bash
roslaunch stereo_depth test_shapes_detection.launch
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
```

---

识别“黑色箭头”:

```bash
roslaunch stereo_depth test_arrow_detection.launch
```

关键话题：

```text
/yolo_unified/target_bbox        stereo_depth/BoundingBox
/arrow/direction                 std_msgs/String
```

---

识别“矩形框”:

```bash
roslaunch stereo_depth test_rectangles_detection.launch
```

关键话题：

```text
/yolo_unified/target_center      geometry_msgs/PointStamped
/obj/target_message              auv_control/TargetDetection
```

---

识别“aruco”:

```bash
roslaunch stereo_depth test_aruco_detection_fisheye.launch
```

关键话题：

```text
/obj/target_message              auv_control/TargetDetection
```

---


录制bag:

```bash

roslaunch stereo_depth test_rosbag.launch \
  mode:=record \
  start_web:=true

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


<pre>[INFO] [1785035157.785452]: 正在发布 /fisheye_camera/image_raw: 15.13 FPS, 2560x1440
[INFO] [1785035162.833731]: 正在发布 /fisheye_camera/image_raw: 14.46 FPS, 2560x1440
[INFO] [1785035168.082419]: 正在发布 /fisheye_camera/image_raw: 13.91 FPS, 2560x1440
[INFO] [1785035173.122093]: 正在发布 /fisheye_camera/image_raw: 13.69 FPS, 2560x1440
[INFO] [1785035178.126619]: 正在发布 /fisheye_camera/image_raw: 14.19 FPS, 2560x1440
*** Added sample 49, p_x = 0.559, p_y = 0.342, p_size = 0.186, skew = 0.066
[INFO] [1785035183.161030]: 正在发布 /fisheye_camera/image_raw: 15.10 FPS, 2560x1440
[INFO] [1785035188.176037]: 正在发布 /fisheye_camera/image_raw: 14.96 FPS, 2560x1440
[INFO] [1785035193.221105]: 正在发布 /fisheye_camera/image_raw: 14.87 FPS, 2560x1440
[INFO] [1785035198.248960]: 正在发布 /fisheye_camera/image_raw: 14.92 FPS, 2560x1440
[INFO] [1785035203.301136]: 正在发布 /fisheye_camera/image_raw: 14.05 FPS, 2560x1440
[INFO] [1785035208.357257]: 正在发布 /fisheye_camera/image_raw: 15.67 FPS, 2560x1440
[INFO] [1785035213.362566]: 正在发布 /fisheye_camera/image_raw: 15.33 FPS, 2560x1440
[INFO] [1785035218.521839]: 正在发布 /fisheye_camera/image_raw: 14.73 FPS, 2560x1440
[INFO] [1785035223.594821]: 正在发布 /fisheye_camera/image_raw: 14.98 FPS, 2560x1440
[INFO] [1785035228.630748]: 正在发布 /fisheye_camera/image_raw: 15.09 FPS, 2560x1440
[INFO] [1785035233.638736]: 正在发布 /fisheye_camera/image_raw: 14.98 FPS, 2560x1440
[INFO] [1785035238.641502]: 正在发布 /fisheye_camera/image_raw: 15.19 FPS, 2560x1440
[INFO] [1785035243.656855]: 正在发布 /fisheye_camera/image_raw: 15.15 FPS, 2560x1440
*** Added sample 50, p_x = 0.620, p_y = 0.478, p_size = 0.183, skew = 0.013
[INFO] [1785035248.687144]: 正在发布 /fisheye_camera/image_raw: 14.31 FPS, 2560x1440
**** Calibrating ****
mono fisheye calibration...
[INFO] [1785035253.745080]: 正在发布 /fisheye_camera/image_raw: 13.05 FPS, 2560x1440
[INFO] [1785035258.785186]: 正在发布 /fisheye_camera/image_raw: 12.34 FPS, 2560x1440
[INFO] [1785035263.789221]: 正在发布 /fisheye_camera/image_raw: 11.75 FPS, 2560x1440
[INFO] [1785035268.889478]: 正在发布 /fisheye_camera/image_raw: 12.16 FPS, 2560x1440
[INFO] [1785035273.918401]: 正在发布 /fisheye_camera/image_raw: 14.13 FPS, 2560x1440
[INFO] [1785035278.965445]: 正在发布 /fisheye_camera/image_raw: 11.88 FPS, 2560x1440
D = [-3.644118474039142, 10.762022360724519, -13.215259515222344, 0.0]
K = [2852.7131934106355, 0.0, 1255.8019460475261, 0.0, 2871.6742979644077, 650.621689013765, 0.0, 0.0, 1.0]
R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P = [2852.7131934106355, 0.0, 1255.8019460475261, 0.0, 0.0, 2871.6742979644077, 650.621689013765, 0.0, 0.0, 0.0, 1.0, 0.0]
None
# oST version 5.0 parameters


[image]

width
2560

height
1440

[fisheye_camera]

camera matrix
2852.713193 0.000000 1255.801946
0.000000 2871.674298 650.621689
0.000000 0.000000 1.000000

distortion
-3.644118 10.762022 -13.215260 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
2852.713193 0.000000 1255.801946 0.000000
0.000000 2871.674298 650.621689 0.000000
0.000000 0.000000 1.000000 0.000000

[INFO] [1785035284.016772]: 正在发布 /fisheye_camera/image_raw: 13.07 FPS, 2560x1440
Traceback (most recent call last):
  File &quot;/home/xhy/catkin_ws/src/image_pipeline/camera_calibration/src/camera_calibration/camera_calibrator.py&quot;, line 274, in on_mouse
    self.c.do_save()
  File &quot;/home/xhy/catkin_ws/src/image_pipeline/camera_calibration/src/camera_calibration/calibrator.py&quot;, line 677, in do_save
    self.do_tarfile_save(tf) # Must be overridden in subclasses
  File &quot;/home/xhy/catkin_ws/src/image_pipeline/camera_calibration/src/camera_calibration/calibrator.py&quot;, line 1034, in do_tarfile_save
    taradd(name, cv2.imencode(&quot;.png&quot;, im)[1].tostring())
  File &quot;/home/xhy/catkin_ws/src/image_pipeline/camera_calibration/src/camera_calibration/calibrator.py&quot;, line 1022, in taradd
    if isinstance(buf, basestring):
NameError: name &apos;basestring&apos; is not defined
Traceback (most recent call last):
  File &quot;/home/xhy/catkin_ws/src/image_pipeline/camera_calibration/src/camera_calibration/camera_calibrator.py&quot;, line 274, in on_mouse
    self.c.do_save()
  File &quot;/home/xhy/catkin_ws/src/image_pipeline/camera_calibration/src/camera_calibration/calibrator.py&quot;, line 677, in do_save
    self.do_tarfile_save(tf) # Must be overridden in subclasses
  File &quot;/home/xhy/catkin_ws/src/image_pipeline/camera_calibration/src/camera_calibration/calibrator.py&quot;, line 1034, in do_tarfile_save
    taradd(name, cv2.imencode(&quot;.png&quot;, im)[1].tostring())
  File &quot;/home/xhy/catkin_ws/src/image_pipeline/camera_calibration/src/camera_calibration/calibrator.py&quot;, line 1022, in taradd
    if isinstance(buf, basestring):
NameError: name &apos;basestring&apos; is not defined
[INFO] [1785035289.109248]: 正在发布 /fisheye_camera/image_raw: 14.95 FPS, 2560x1440
[INFO] [1785035294.134910]: 正在发布 /fisheye_camera/image_raw: 14.70 FPS, 2560x1440
[INFO] [1785035299.174001]: 正在发布 /fisheye_camera/image_raw: 15.50 FPS, 2560x1440
[INFO] [1785035304.215586]: 正在发布 /fisheye_camera/image_raw: 13.08 FPS, 2560x1440
[INFO] [1785035309.235198]: 正在发布 /fisheye_camera/image_raw: 13.55 FPS, 2560x1440
[INFO] [1785035314.248592]: 正在发布 /fisheye_camera/image_raw: 13.96 FPS, 2560x1440
[INFO] [1785035319.278452]: </pre>
