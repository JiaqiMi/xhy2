# state_web

`state_web` 是 AUV 运行状态只读监控页面，显示三路原始相机、NED
位置、Z/Down、航向、人工地平仪、反馈、控制指令和设备健康状态。

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
- `/world_origin` 更新后，所有已打开页面会自动清空本地轨迹。
- “绘制轨迹”默认关闭，轨迹仅保存在当前浏览器页面会话中。
- 默认端口为 `8088`，可通过 `state_web_port` 或 `port` 参数修改。
