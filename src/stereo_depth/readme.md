## 7. Shapes 多类别任务

### 7.1 启动

```bash
roslaunch stereo_depth test_shapes_detection.launch
```

默认模型：

```text
/home/xhy/catkin_ws/models/shapes0709.pt
```

模型类别由权重文件内部 `names` 决定。

### 7.2 核心话题

```text
/yolo_unified/annotated_image    sensor_msgs/Image
/web/detections                  std_msgs/String(JSON)
/yolo_unified/target_center      geometry_msgs/PointStamped
/obj/target_message              auv_control/TargetDetection
/web/pose                        std_msgs/String(JSON)
```

### 7.3 检查

```bash
rostopic echo -n 1 /web/detections
rostopic echo -n 1 /obj/target_message
rostopic echo -n 1 /web/pose
```

Web：

```text
http://192.168.1.117:8080
```

---

## 8. 彩色矩形框任务

### 8.1 启动

```bash
roslaunch stereo_depth test_rectangles_detection.launch
```

当前参数：

```text
task_mode=detect
detect_mode=2
model_path=""
```

因此 `yolo.py` 自动加载：

```text
/home/xhy/catkin_ws/models/rectangle0710.pt
```

### 8.2 核心话题

```text
/yolo_unified/annotated_image    sensor_msgs/Image
/web/detections                  std_msgs/String(JSON)
/yolo_unified/target_center      geometry_msgs/PointStamped
/obj/target_message              auv_control/TargetDetection
/web/pose                        std_msgs/String(JSON)
```

### 8.3 检查

```bash
rostopic echo -n 1 /web/detections
rostopic echo -n 1 /obj/target_message
rostopic echo -n 1 /web/pose
```

Web：

```text
http://192.168.1.117:8080
```

---

## 9. 黑色箭头二维方向任务

### 9.1 启动

```bash
roslaunch stereo_depth test_arrow_detection.launch
```

默认模型：

```text
/home/xhy/catkin_ws/models/arrow0709.pt
```

注意：当前 launch 显式设置了模型路径，因此 `detect_mode` 不负责模型选择。为了语义一致，后续可以把 `detect_mode` 改为 `4`。

### 9.2 YOLO 检测话题

```text
/yolo_unified/target_bbox        stereo_depth/BoundingBox
/web/detections                  std_msgs/String(JSON)
/yolo_unified/annotated_image    sensor_msgs/Image
```

### 9.3 箭头方向话题

#### 完整方向 JSON

```text
/arrow/direction
```

类型：

```text
std_msgs/String
```

有效格式：

```json
{
  "stamp": 1780000000.12,
  "source": "arrow_direction",
  "valid": true,
  "reason": "",
  "class_name": "arrow",
  "confidence": 0.91,
  "bbox": {
    "x1": 120,
    "y1": 80,
    "x2": 300,
    "y2": 220
  },
  "tip": {
    "u": 280,
    "v": 145
  },
  "tail": {
    "u": 145,
    "v": 150
  },
  "center": {
    "u": 210,
    "v": 148
  },
  "direction_2d": {
    "x": 0.999,
    "y": -0.037
  },
  "angle_rad": 0.037,
  "angle_deg": 2.1,
  "discrete_direction": "right"
}
```

无效格式：

```json
{
  "stamp": 1780000000.12,
  "source": "arrow_direction",
  "valid": false,
  "reason": "no_arrow_bbox",
  "class_name": "arrow",
  "confidence": 0.0,
  "bbox": null,
  "tip": null,
  "tail": null,
  "center": null,
  "direction_2d": null,
  "angle_rad": null,
  "angle_deg": null,
  "discrete_direction": "none"
}
```

无箭头时该话题仍然按固定频率发布。

查看：

```bash
rostopic hz /arrow/direction
rostopic echo /arrow/direction
```

#### 连续角度

```text
/arrow/angle_deg
```

类型：

```text
std_msgs/Float32
```

约定：

```text
向右 = 0°
向上 = 90°
向左 = 180°
向下 = 270°
```

无效时为：

```text
NaN
```

查看：

```bash
rostopic echo /arrow/angle_deg
```

#### 离散方向

```text
/arrow/discrete_direction
```

类型：

```text
std_msgs/String
```

可能值：

```text
right
up_right
up
up_left
left
down_left
down
down_right
none
```

#### 二维方向向量

```text
/arrow/direction_vector
```

类型：

```text
geometry_msgs/Vector3Stamped
```

有效格式：

```yaml
header:
  stamp: ...
  frame_id: "camera"
vector:
  x: 0.999
  y: -0.037
  z: 0.0
```

无效时为零向量。

#### 箭头标注图

```text
/arrow/annotated_image
```

类型：

```text
sensor_msgs/Image
```

其中绘制：

- YOLO bbox
- 黑色箭头轮廓
- 红色尖端
- 蓝色尾部
- 黄色中心
- 绿色方向箭头
- 连续角度
- 离散方向

### 9.4 当前没有箭头三维深度

`test_arrow_detection.launch` 没有启动 `stereo_depth_node.py`，因此当前箭头任务不发布：

```text
/obj/target_message
/web/pose
```

Web 页面中的“三维位置”区域显示暂无数据属于正常现象。

### 9.5 Web

箭头任务使用：

```text
vision_web_v2.launch
```

浏览器：

```text
http://192.168.1.117:8080
```

Web 会显示：

- 箭头方向标注画面
- YOLO 类别和置信度
- 箭头连续角度
- 离散方向
- 二维方向向量
- 尖端和尾部像素

---

## 10. ArUco 任务

### 10.1 当前启动命令

```bash
roslaunch stereo_depth test_aruco_detection_fisheye.launch
```

发布话题： obj/target_message, 内容如下：

```base
pose: 
  header: 
    seq: 0
    stamp: 
      secs: 1784127191
      nsecs: 163179636
    frame_id: "fisheye_camera"
  pose: 
    position: 
      x: 0.0
      y: 0.0
      z: 0.0
    orientation: 
      x: 0.0
      y: 0.0
      z: 0.0
      w: 1.0
conf: 0.0
type: "aruco_not_detected"
class_name: "-1"
```

## 11. Web 使用说明

### 11.1 普通任务

使用：

```text
vision_web.launch
```

默认订阅：

```text
/yolo_unified/annotated_image
/web/detections
/web/pose
```

### 11.2 箭头任务

使用：

```text
vision_web_v2.launch
```

默认订阅：

```text
/arrow/annotated_image
/web/detections
/web/pose
/arrow/direction
```

### 11.3 浏览器地址

```text
http://192.168.1.117:8080
```

设备 IP 变化时查看：

```bash
hostname -I
```

然后访问：

```text
http://设备IP:8080
```

### 11.4 服务检查

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/api/status
ss -lntp | grep 8080
```

---

## 12. 常用诊断命令

### 查看节点

```bash
rosnode list
```

### 查看话题

```bash
rostopic list | sort
```

### 查看某话题类型

```bash
rostopic type /web/detections
```

### 查看自定义消息定义

```bash
rosmsg show stereo_depth/BoundingBox
rosmsg show stereo_depth/LineBox
rosmsg show auv_control/TargetDetection
rosmsg show auv_control/TargetDetection3
```

### 查看发布者和订阅者

```bash
rostopic info /web/pose
rostopic info /arrow/direction
```

### 检查频率

```bash
rostopic hz /left/image_raw
rostopic hz /yolo_unified/annotated_image
rostopic hz /web/detections
rostopic hz /arrow/direction
```

### 查看模型实际加载路径

启动日志应出现：

```text
Final model path: ...
```

也可以检查参数：

```bash
rosparam get /yolo_unified_detector/task_mode
rosparam get /yolo_unified_detector/detect_mode
rosparam get /yolo_unified_detector/model_path
```

---

## 13. 推荐的一次性启动前检查

```bash
# 1. 加载环境
source /opt/ros/melodic/setup.bash
source ~/catkin_ws/devel_isolated/setup.bash
source /home/xhy/xhy_env/bin/activate
export PYTHONPATH=/home/xhy/catkin_ws/devel_isolated/cv_bridge/lib/python3/dist-packages:$PYTHONPATH

# 2. 检查模型
ls -lh \
  ~/catkin_ws/models/red_circle0713.pt \
  ~/catkin_ws/models/shapes0709.pt \
  ~/catkin_ws/models/rectangle0710.pt \
  ~/catkin_ws/models/arrow0709.pt

# 3. 检查相机
ls -l /dev/video0

# 4. 检查脚本
bash -n ~/catkin_ws/src/yolo_bridge/scripts/yolo_wrapper.sh
python -m py_compile ~/catkin_ws/src/yolo_bridge/scripts/yolo.py
python -m py_compile ~/catkin_ws/src/stereo_depth/scripts/stereo_depth_node.py
python -m py_compile ~/catkin_ws/src/stereo_depth/scripts/arrow_direction_node.py

# 5. 检查设备和端口占用
fuser /dev/video0 2>/dev/null
ss -lntp | grep 8080
```

## rosbag视频回放功能

```bash

roslaunch stereo_depth test_rosbag.launch \
  mode:=record

roslaunch stereo_depth test_rosbag.launch \
  mode:=play \
  bag_file:=/home/xhy/xhy_records/stereo_input_0.bag
```