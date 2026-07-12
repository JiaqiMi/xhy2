# #!/bin/bash
# # LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 python /home/xhy/catkin_ws/src/yolo_bridge/scripts/yolo_v8_ros_bridge.py
# python /home/xhy/catkin_ws/src/yolo_bridge/scripts/yolo_v8_bridge_py38.py

#!/bin/bash

# 获取 detect_mode 参数
DETECT_MODE=$(rosparam get /yolo26n_detector/detect_mode)
TOP_K=$(rosparam get /yolo26n_detector/top_k)
VISUALIZATION=$(rosparam get /yolo26n_detector/visualization)
CONF_THRE=$(rosparam get /yolo26n_detector/conf_thre)
DETC_TYPE=$(rosparam get /yolo26n_detector/detc_type)


# 启动 Python 脚本并传入参数
python /home/xhy/catkin_ws/src/yolo_bridge/scripts/yolo26n_detection_web.py --detect_mode $DETECT_MODE --top_k $TOP_K --visualization $VISUALIZATION --conf_thre $CONF_THRE --detc_type $DETC_TYPE
