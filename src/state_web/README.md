# state_web

`state_web` 是 AUV 运行状态只读监控页面，显示三路原始相机、执行器
指令与实际反馈、6S4P 电池与功率、TF 实际 NED 位姿、目标位姿和设备健康状态。

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

- 页面只订阅 ROS 话题，不发布控制消息；清除按钮只修改 Web 内存历史。
- 双目和鱼眼相机驱动需要独立启动；缺少图像时对应画面显示离线。
- 鱼眼在普通布局中仅从 Web 端居中裁成正方形，双击放大恢复完整 16:9；
  ROS 图像流和 ArUco 标注坐标不受影响。
- 执行器主卡以浅色显示 `/cmd/actuator` 指令、深色显示
  `/status/actuator` 实际反馈，叠加展示三色灯、向上开合的夹爪和推杆；
  同时显示指令/反馈年龄、Web 接收时间差和同步状态。接收时间差不等同于
  严格的硬件执行延迟；补光灯与航向舵机位于折叠详情。
- 电源1作为控制支路、电源2作为动力支路。页面只显示动力电压，并显示
  两路电流/功率及总功率。
- 电池按 6S4P 普通锂离子、单节 4 Ah、整包 16 Ah 估算；动力电压默认
  平滑 5 秒后按单节电压曲线计算 SOC 和剩余 Ah。
- 位置图按真实 TF 点位绘制 `base_link → camera` 箭头；`cmdned`
  目标位姿始终表示 `base_link`，不记录运行轨迹。
- 2D 位置图和深度图使用独立比例尺；两者均支持滚轮缩放，2D 图支持
  平面拖动，深度图支持上下拖动。
- “回到原点”同时重置 2D 位置图和深度图。
- 目标话题超时后保留最后一帧，并以灰色标记其超时状态。
- 视觉标注默认保留最后一次有效结果 `1.0` 秒，可通过
  `vision_overlay_hold` launch 参数调整；空检测结果不会立即清除标注。
- 左目有效三维识别结果会按检测时间戳转换到 `map` 并绘制在位置图上，
  每种图形默认滚动保留 20 条；可隐藏绘图或同时清除地图与 ArUco 历史。
- “显示位置标注”开关只控制地图视觉标记旁的置信度和 N/E 文字，
  关闭后圆点、方块、折线与箭头仍继续显示和滚动记录。
- 鱼眼标题显示最近 10 次合法 ArUco 结果；同一 ID 命中 3 次后按
  `1/2=黄、3/4=绿、5/6=红` 锁存期望亮灯颜色。
- `POST /api/vision-history/clear` 用于清除 Web 视觉历史，不影响视觉节点、
  ROS 话题或控制任务。
- 位置图与任务节点一致，使用 `lookupTransform(map, child, Time(0))`
  获取三路最新 TF；公共时间戳获取失败不影响已经取得的位姿。页面按 1 Hz
  最多绘制 1000 个轨迹点，并同时显示深粉色最新目标与浅粉色上一目标。
- `POST /api/base-trajectory/clear` 只清除 Web 中的 `base_link` 轨迹，
  不影响目标、水池范围或视觉历史。
- 核心状态按三轴紧凑排列，实际与目标数据上下对齐。
- 核心状态同时显示 `debug_driver` 控制模式和 `/motion/state` 状态机；
  位置误差与 Yaw 误差均按“目标减实际 TF”实时计算。
- `imu_frame`、`base_frame` 和 `camera_frame` 默认分别为 `imu`、
  `base_link` 和 `camera_center`，可通过 launch 参数覆盖。
- 默认端口为 `8088`，可通过 `state_web_port` 或 `port` 参数修改。
