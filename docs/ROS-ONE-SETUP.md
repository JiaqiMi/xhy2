# 小黄鱼 ROS-O (ros-one) 迁移说明 — Jetson Orin NX / Ubuntu 22.04

> 分支:`ros-one-jammy` · 日期:2026-08-03
> 目标:在 Ubuntu 22.04 (JetPack 6.2) + 已装 ROS2 Humble 的板子上,让本 ROS1 (Melodic 时代) 项目原生运行,并为后续 ROS1/ROS2 混合(ros1_bridge)做好准备。

## 1. 背景

- 本项目是 ROS1 catkin 工作空间(rospy/roscpp),原开发环境为 Ubuntu 18.04 + ROS Melodic + 自建 Python3.8 venv(`/home/xhy/xhy_env`),并源码编译过 cv_bridge / geometry2。
- 板子(Jetson Orin NX, JetPack 6.2, Ubuntu 22.04 jammy, aarch64)只有 ROS2 Humble,没有任何 ROS1 运行时,Ubuntu 22.04 官方也从未提供过 ROS1。
- 解决方案:**ROS-O("ROS One")** — Noetic EOL 后的社区 ROS1 延续,Bielefeld 大学维护的 deb 仓库 https://ros.packages.techfak.net 直接提供 jammy/arm64 二进制包,安装到 `/opt/ros/one`,与 `/opt/ros/humble` 共存。

## 2. 板上系统级安装(不在 git 内,重刷系统后需重做)

```bash
# 2.1 ROS-O apt 源
sudo curl -sSL https://ros.packages.techfak.net/gpg.key -o /etc/apt/keyrings/ros-one-keyring.gpg
echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/ros-one-keyring.gpg] https://ros.packages.techfak.net jammy main" \
  | sudo tee /etc/apt/sources.list.d/ros1.list
sudo apt update

# 2.2 ROS-O 运行时 + 项目依赖
sudo apt install -y ros-one-ros-base python3-rosdep python3-catkin-tools \
  ros-one-cv-bridge ros-one-image-geometry ros-one-image-transport \
  ros-one-camera-calibration-parsers ros-one-camera-info-manager \
  ros-one-cmake-modules ros-one-eigen-conversions ros-one-tf \
  ros-one-tf2-geometry-msgs ros-one-nodelet-topic-tools ros-one-stereo-msgs \
  ros-one-usb-cam ros-one-compressed-image-transport \
  libgtk2.0-dev python3-flask python3-rospkg \
  libboost-dev libboost-system-dev libboost-thread-dev libboost-filesystem-dev \
  libopencv-superres4.5d libopencv-videostab4.5d
```

注意事项(装机时踩过的坑):
- **系统 apt 源已切到清华镜像**(`/etc/apt/sources.list`,原文件备份为 `sources.list.bak_20260803`)。原因:ports.ubuntu.com 经运营商 HTTP 透明缓存返回过期索引,导致 `libldap-dev` 等包 404 装不上。若再遇到 404,先 `sudo rm -rf /var/lib/apt/lists/* && sudo apt update`。
- **不要装 `libopencv-superres-dev` 等 -dev 包**:JetPack 自带 `libopencv-dev 4.8.0` 会冲突;只装 `*.4.5d` 运行库即可满足 ros-one cv_bridge 的链接需求。
- 板上有三套 OpenCV:系统 4.5.4d(ros-one 链接)、JetPack dev 4.8、`/usr/local` 自编 4.10(CUDA,Python 侧 `cv2` 实际用它)。C++ 统一链系统 4.5.4d,互不干扰。

## 3. 代码修改(全部在 `ros-one-jammy` 分支)

| 修改 | 文件范围 | 原因 |
|---|---|---|
| `/home/xhy/xhy_env*/bin/python*` → `/home/nvidia/venvs/xhy_ros2/bin/python` | 全部脚本 shebang 和 wrapper(105 文件,188 处) | 旧机器 py3.8 venv 不存在;板上统一用 `~/venvs/xhy_ros2`(py3.10 + NVIDIA torch 2.5 + ultralytics) |
| `/home/xhy/catkin_ws` → `/home/nvidia/catkin_ws` | launch 中模型/标定路径、脚本内路径 | 用户目录不同 |
| `/home/xhy/xhy_records` → `/home/nvidia/xhy_records` | rosbag 记录目录 | 同上(目录已创建) |
| `-std=c++11/14` → `-std=c++17` | image_pipeline 3 个 C++ 包的 CMakeLists | jammy 的 log4cxx 0.12 头文件要求 C++17 |
| 补 `#include <opencv2/calib3d.hpp>` | depth_image_proc 2 个 cpp、stereo_image_proc processor.h | OpenCV4 中 `undistortPoints/StereoBM/StereoSGBM` 不再被传递包含 |
| `src/CMakeLists.txt` 符号链接指向 `/opt/ros/one/.../toplevel.cmake` | 1 个符号链接 | 原指向 melodic,是死链 |
| 注释 `catkin_add_nosetests(test/test_state_web_core.py)` | state_web/CMakeLists.txt | 上游 PR 删了测试文件但没删引用,配置期报错 |
| 新增 `src/__MACOSX/CATKIN_IGNORE` | 标记文件 | 弃用 vendored cv_bridge 源码(改用 ros-one-cv-bridge 二进制) |
| 新增本文档 `docs/ROS-ONE-SETUP.md` | — | — |

另:`src/vision_opencv_ros2/`(ROS2 子模块,属 `ros2-humble` 分支)如果目录残留,内放 `CATKIN_IGNORE` 即可,勿加入本分支版本控制。
不再需要的历史补丁:`.rosinstall` 的 geometry2 源码编译(改用 `ros-one-tf2-*`)、`README` 中 `PYTHONPATH=...devel_isolated/cv_bridge...` 的 export。

## 4. `~/.bashrc` 环境切换(不在 git 内)

原文件备份:`~/.bashrc.bak_20260803`。尾部已改为:

```bash
use_ros1() {   # ROS1 (ROS-O) + 小黄鱼工作空间
  source /opt/ros/one/setup.bash
  [ -f ~/catkin_ws/devel_isolated/setup.bash ] && source ~/catkin_ws/devel_isolated/setup.bash
}
use_ros2() {   # ROS2 Humble
  source /opt/ros/humble/setup.bash
  [ -f ~/catkin_ws/install/setup.bash ] && source ~/catkin_ws/install/setup.bash
}
source ~/venvs/xhy_ros2/bin/activate
use_ros1   # 新终端默认 ROS1
```

**规则:一个终端只能用一套 ROS。** 要用 ROS2 就开新终端敲 `use_ros2`,不要在同一 shell 里先后调用两个函数。

## 5. 编译

```bash
cd ~/catkin_ws
source /opt/ros/one/setup.bash          # 新终端默认已 source,可省略
catkin_make_isolated -DCMAKE_BUILD_TYPE=Release
# 单包:catkin_make_isolated --pkg auv_control ...
source devel_isolated/setup.bash
```

不再需要旧的 `-DPYTHON_EXECUTABLE=...python3.8`;Python 节点通过 shebang 直接用 venv 解释器。

## 6. 运行

```bash
# 相机(需接上 USB 双目)
rosrun usb_cam usb_cam_node _video_device:=/dev/video0 _pixel_format:=mjpeg \
  _image_width:=1280 _image_height:=480

# 主控
roslaunch auv_control begin.launch

# 视觉检测示例
roslaunch stereo_depth test_red_circle_detection.launch
```

## 7. 已验证的测试(2026-08-03,全部通过)

| # | 测试 | 结果 |
|---|---|---|
| 1 | 16/16 包 `catkin_make_isolated` 编译 | ✅ 无错误 |
| 2 | roscore 启动 / rostopic | ✅ |
| 3 | 自定义消息生成(auv_control 11 个 + oculus 8 个 + srv) | ✅ `rosmsg show auv_control/AUVData` 正常 |
| 4 | venv 内 `cv_bridge` 图像双向转换(与 cv2 4.10 共存) | ✅ 像素级一致 |
| 5 | venv 内 `rospy` + `tf2_ros` 导入 | ✅ |
| 6 | torch 2.5.0+nv24.08 CUDA(Orin) | ✅ `cuda.is_available()=True` |
| 7 | 端到端:image_publisher 发测试图 → `yolo.py`(yolo11n.pt, GPU)→ 标注图话题 | ✅ 稳定 2.0 Hz 输出(等于设定推理率) |
| 8 | `begin.launch` / `test_red_circle_detection.launch` / `state_web.launch` / `test_yolo_detect.launch` 解析 | ✅ 节点清单完整解析 |
| 9 | 新终端默认环境(`ROS_DISTRO=one` + venv python) | ✅ |

**未测(缺硬件/数据,下水/接线后需回归):**
- 实体相机(`/dev/video*` 当前未接)、串口传感器(DVL/IMU/推进器,`/dev/ttyUSB*` 未接)、Oculus 声呐;
- 任务模型权重:`~/catkin_ws/models/*.pt` 不在板上(launch 里引用的 shapes/red_circle/holes 等模型需从训练机拷贝到 `/home/nvidia/catkin_ws/models/`)。

## 8. 后续:ROS1/ROS2 混合(规划)

- 用 ros-o 维护的 fork **github.com/ros-o/ros1_bridge**(README 明确支持 22.04 = ROS-O + Humble),需源码编译:单独 workspace,同时 source `/opt/ros/one` 和 `/opt/ros/humble`。
- 要过桥的自定义消息(如 `AUVData`、`TargetDetection`)需先建同名同字段的 ROS2 消息包;**消息定义每次变更后 bridge 必须重编**。
- 建议过桥消息集合稳定后再搭 bridge;新功能直接用 ROS2 写,通过 bridge 与现有 rospy 节点通信,逐包迁移。

## 9. 回滚

- 恢复 shell:`cp ~/.bashrc.bak_20260803 ~/.bashrc`
- 恢复 apt 源:`sudo cp /etc/apt/sources.list.bak_20260803 /etc/apt/sources.list`
- 卸载 ROS-O:`sudo apt remove "ros-one-*" && sudo rm /etc/apt/sources.list.d/ros1.list`
- 代码:切回 `main` 分支即可(本分支所有改动独立)。
