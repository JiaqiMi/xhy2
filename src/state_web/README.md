# state_web

`state_web` 是 AUV 运行状态只读监控页面，显示三路原始相机、TF 实际
NED 位姿、`/cmd/pose/ned` 目标位姿、Z/Down 深度、反馈、控制指令和
设备健康状态。

## 启动

默认随控制侧启动：

```bash
roslaunch auv_control begin.launch
```

浏览器访问：

```text
http://<机器人IP>:8088
```

也可以独立启动：

```bash
roslaunch state_web state_web.launch
```

关闭随 `begin.launch` 启动：

```bash
roslaunch auv_control begin.launch enable_state_web:=false
```

## 说明

- 页面只订阅 ROS 话题，不发布控制消息。
- 双目和鱼眼相机驱动需要独立启动；缺少图像时对应画面显示离线。
- 位置图仅绘制 TF 实际位置和 `cmdned` 目标位置，不记录运行轨迹。
- 2D 位置图和深度图使用独立比例尺；两者均支持滚轮缩放，2D 图支持
  平面拖动，深度图支持上下拖动。
- “回到原点”同时重置 2D 位置图和深度图。
- 目标话题超时后保留最后一帧，并以灰色标记其超时状态。
- 核心状态按三轴紧凑排列，实际与目标数据上下对齐。
- 默认端口为 `8088`，可通过 `state_web_port` 或 `port` 参数修改。
