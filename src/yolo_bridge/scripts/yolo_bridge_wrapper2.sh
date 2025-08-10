#!/bin/bash

# 获取 detect_mode 参数
DETECT_MODE=$(rosparam get /yolov8_node2/detect_mode)
TOP_K=$(rosparam get /yolov8_node2/top_k)
VISUALIZATION=$(rosparam get /yolov8_node2/visualization)
CONF_THRE=$(rosparam get /yolov8_node2/conf_thre)
DETC_TYPE=$(rosparam get /yolov8_node2/detc_type)


# 启动 Python 脚本并传入参数
python /home/xhy/catkin_ws/src/yolo_bridge/scripts/yolo_bridge_node.py --detect_mode $DETECT_MODE --top_k $TOP_K --visualization $VISUALIZATION --conf_thre $CONF_THRE --detc_type $DETC_TYPE
