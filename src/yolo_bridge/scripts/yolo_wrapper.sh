#!/bin/bash
set -euo pipefail

NODE_NAME="${NODE_NAME:-/yolo_unified_detector}"
SCRIPT="/home/xhy/catkin_ws/src/yolo_bridge/scripts/yolo.py"
PYTHON_BIN="/home/xhy/xhy_env/bin/python3.8"
CV_BRIDGE_PY3="/home/xhy/catkin_ws/devel_isolated/cv_bridge/lib/python3/dist-packages"

if [ ! -x "${PYTHON_BIN}" ]; then
    echo "[yolo_wrapper] Python not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [ ! -f "${SCRIPT}" ]; then
    echo "[yolo_wrapper] Script not found: ${SCRIPT}" >&2
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

TASK_MODE=$(get_param task_mode detect)
DETECT_MODE=$(get_param detect_mode 1)
MODEL_PATH=$(get_param model_path "")
TOP_K=$(get_param top_k 3)
VISUALIZATION=$(get_param visualization 0)
CONF_THRE=$(get_param conf_thre 0.2)
DETC_TYPE=$(get_param detc_type center)
OUTPUT_TYPE=$(get_param output_type quartiles)
INFER_RATE=$(get_param infer_rate 5.0)

INPUT_TOPIC=$(get_param input_topic /left/image_raw)
ANNOTATED_TOPIC=$(get_param annotated_topic /yolo_unified/annotated_image)
WEB_TOPIC=$(get_param web_topic /web/detections)
CENTER_TOPIC=$(get_param center_topic /yolo_unified/target_center)
BBOX_TOPIC=$(get_param bbox_topic /yolo_unified/target_bbox)
LINE_TOPIC=$(get_param line_topic /yolo_unified/line_bbox)

ARGS=(
    --task_mode "${TASK_MODE}"
    --detect_mode "${DETECT_MODE}"
    --top_k "${TOP_K}"
    --visualization "${VISUALIZATION}"
    --conf_thre "${CONF_THRE}"
    --detc_type "${DETC_TYPE}"
    --output_type "${OUTPUT_TYPE}"
    --infer_rate "${INFER_RATE}"
    --input_topic "${INPUT_TOPIC}"
    --annotated_topic "${ANNOTATED_TOPIC}"
    --web_topic "${WEB_TOPIC}"
    --center_topic "${CENTER_TOPIC}"
    --bbox_topic "${BBOX_TOPIC}"
    --line_topic "${LINE_TOPIC}"
)

if [ -n "${MODEL_PATH}" ]; then
    ARGS+=(--model_path "${MODEL_PATH}")
fi

echo "[yolo_wrapper] node=${NODE_NAME}"
echo "[yolo_wrapper] task_mode=${TASK_MODE}"
echo "[yolo_wrapper] model=${MODEL_PATH:-preset:${DETECT_MODE}}"
echo "[yolo_wrapper] python=${PYTHON_BIN}"
echo "[yolo_wrapper] cv_bridge_py3=${CV_BRIDGE_PY3}"

exec "${PYTHON_BIN}" "${SCRIPT}" "${ARGS[@]}" "$@"