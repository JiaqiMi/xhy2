# 常用指令说明

## 环境

#### 编译
catkin_make_isolated \
--cmake-args \
-DCMAKE_BUILD_TYPE=Release \
-DPYTHON_EXECUTABLE=/home/xhy/xhy_env/bin/python3.8



## 相机

#### 启动相机画面

sudo guvcview
或者
cheese

#### ros 启动相机话题

rosrun usb_cam usb_cam_node \
  _video_device:=/dev/video0 \
  _pixel_format:=mjpeg \
  _image_width:=1280 \
  _image_height:=480 \
  2> >(grep -v "No accelerated colorspace conversion found")

## control

roslaunch auv_control begin.launch