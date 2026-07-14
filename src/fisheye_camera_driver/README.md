# RTSP 鱼眼相机驱动

适用于 ROS1 Melodic，通过 RTSP 读取鱼眼相机未经校正的画面，并发布标准
`sensor_msgs/Image`。

Python 节点固定使用 `/home/xhy/xhy_env/bin/python3.8`。虚拟环境需已安装
`opencv-python` 和 `Flask`；若缺少依赖，可执行：

```bash
pip install opencv-python Flask -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 启动驱动

```bash
roslaunch fisheye_camera_driver driver.launch
```

默认输出：

```text
/fisheye_camera/image_raw    sensor_msgs/Image
```

## 启动驱动和 Web 测试

```bash
roslaunch fisheye_camera_driver test_web.launch
```

浏览器访问：

```text
http://设备IP:8081
```

仅测试已启动的图像话题：

```bash
roslaunch fisheye_camera_driver test_web.launch start_driver:=false
```

## 参数覆盖示例

```bash
roslaunch fisheye_camera_driver test_web.launch \
  camera_host:=192.168.1.122 \
  camera_port:=554 \
  channel:=1 \
  subtype:=0 \
  web_port:=8081
```

也可在启动后检查节点私有参数：

```bash
rosparam get /rtsp_fisheye_driver
rosparam get /fisheye_web_image_test
rostopic hz /fisheye_camera/image_raw
```

若摄像头使用不同路径，可直接覆盖完整地址：

```bash
roslaunch fisheye_camera_driver driver.launch \
  rtsp_url:='rtsp://admin:admin@192.168.1.122:554/实际路径'
```

完整地址包含密码，避免将启动日志或终端历史发送给无关人员。
