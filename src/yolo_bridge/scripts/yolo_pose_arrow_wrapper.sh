#!/bin/bash
set -euo pipefail

NODE_NAME="${NODE_NAME:-/yolo_arrow_pose_detector}"
SCRIPT="/home/xhy/catkin_ws/src/yolo_bridge/scripts/yolo_pose_arrow.py"
PYTHON_BIN="/home/xhy/xhy_env/bin/python3.8"
CV_BRIDGE_PY3="/home/xhy/catkin_ws/devel_isolated/cv_bridge/lib/python3/dist-packages"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[yolo_pose_wrapper] Python not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [ ! -f "${SCRIPT}" ]; then
    echo "[yolo_pose_wrapper] Script not found: ${SCRIPT}" >&2
    exit 1
fi

if [ -d "${CV_BRIDGE_PY3}" ]; then
    export PYTHONPATH="${CV_BRIDGE_PY3}:${PYTHONPATH:-}"
fi

get_param() {
    local name="$1"
    local default_value="$2"
    rosparam get "${NODE_NAME}/${name}" 2>/dev/null || echo "${default_value}"
}

MODEL_PATH=$(get_param model_path "")
TOP_K=$(get_param top_k 3)
VISUALIZATION=$(get_param visualization 0)
CONF_THRE=$(get_param conf_thre 0.20)
KEYPOINT_CONF_THRE=$(get_param keypoint_conf_thre 0.35)
INFER_RATE=$(get_param infer_rate 5.0)
IMGSZ=$(get_param imgsz 640)
DEVICE=$(get_param device 0)

INPUT_TOPIC=$(get_param input_topic /left/image_raw)
ANNOTATED_TOPIC=$(get_param annotated_topic /yolo_unified/annotated_image)
WEB_TOPIC=$(get_param web_topic /web/detections)
KEYPOINT_TOPIC=$(get_param keypoint_topic /yolo_unified/arrow_keypoints)
BBOX_TOPIC=$(get_param bbox_topic /yolo_unified/target_bbox)

if [ -z "${MODEL_PATH}" ]; then
    echo "[yolo_pose_wrapper] model_path is empty" >&2
    exit 1
fi

ARGS=(
    --model_path "${MODEL_PATH}"
    --top_k "${TOP_K}"
    --visualization "${VISUALIZATION}"
    --conf_thre "${CONF_THRE}"
    --keypoint_conf_thre "${KEYPOINT_CONF_THRE}"
    --infer_rate "${INFER_RATE}"
    --imgsz "${IMGSZ}"
    --device "${DEVICE}"
    --input_topic "${INPUT_TOPIC}"
    --annotated_topic "${ANNOTATED_TOPIC}"
    --web_topic "${WEB_TOPIC}"
    --keypoint_topic "${KEYPOINT_TOPIC}"
    --bbox_topic "${BBOX_TOPIC}"
)

echo "[yolo_pose_wrapper] node=${NODE_NAME}"
echo "[yolo_pose_wrapper] model=${MODEL_PATH}"
echo "[yolo_pose_wrapper] python=${PYTHON_BIN}"
echo "[yolo_pose_wrapper] cv_bridge_py3=${CV_BRIDGE_PY3}"
echo "[yolo_pose_wrapper] input=${INPUT_TOPIC}"
echo "[yolo_pose_wrapper] keypoints=${KEYPOINT_TOPIC}"
echo "[yolo_pose_wrapper] conf=${CONF_THRE}, kpt_conf=${KEYPOINT_CONF_THRE}"

exec "${PYTHON_BIN}" "${SCRIPT}" "${ARGS[@]}" "$@"
