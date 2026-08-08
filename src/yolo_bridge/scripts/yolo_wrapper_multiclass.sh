#!/bin/bash
set -euo pipefail

NODE_NAME="${NODE_NAME:-/yolo_unified_detector}"
CATKIN_WS="${CATKIN_WS:-${HOME}/catkin_ws}"
SCRIPT="${YOLO_SCRIPT:-${CATKIN_WS}/src/yolo_bridge/scripts/yolo_multiclass.py}"

# Prefer the current NVIDIA virtual environment; keep fallbacks for older xhy setups.
if [ -n "${YOLO_PYTHON_BIN:-}" ]; then
    PYTHON_BIN="${YOLO_PYTHON_BIN}"
elif [ -x "/home/nvidia/venvs/xhy_ros2/bin/python" ]; then
    PYTHON_BIN="/home/nvidia/venvs/xhy_ros2/bin/python"
elif [ -x "${HOME}/venvs/xhy_ros2/bin/python" ]; then
    PYTHON_BIN="${HOME}/venvs/xhy_ros2/bin/python"
elif [ -x "${HOME}/xhy_env/bin/python3.8" ]; then
    PYTHON_BIN="${HOME}/xhy_env/bin/python3.8"
else
    PYTHON_BIN="$(command -v python3)"
fi

CV_BRIDGE_PY3="${CATKIN_WS}/devel_isolated/cv_bridge/lib/python3/dist-packages"

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

TOP_K=$(get_param top_k 20)
MAX_DET=$(get_param max_det 30)
VISUALIZATION=$(get_param visualization 0)
CONF_THRE=$(get_param conf_thre 0.2)
CANDIDATE_CONF_THRE=$(get_param candidate_conf_thre 0.05)
CLASS_CONF_THRESHOLDS=$(get_param class_conf_thresholds "")
TARGET_SELECTION_MODE=$(get_param target_selection_mode global_best)
TARGET_PUB_QUEUE_SIZE=$(get_param target_pub_queue_size 10)
CLASS_WHITELIST=$(get_param class_whitelist "")
CLASS_PRIORITY=$(get_param class_priority "")

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
    --max_det "${MAX_DET}"
    --visualization "${VISUALIZATION}"
    --conf_thre "${CONF_THRE}"
    --candidate_conf_thre "${CANDIDATE_CONF_THRE}"
    --class_conf_thresholds "${CLASS_CONF_THRESHOLDS}"
    --target_selection_mode "${TARGET_SELECTION_MODE}"
    --target_pub_queue_size "${TARGET_PUB_QUEUE_SIZE}"
    --class_whitelist "${CLASS_WHITELIST}"
    --class_priority "${CLASS_PRIORITY}"
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

if [ -n "${MODEL_PATH}" ] && [ "${MODEL_PATH}" != "''" ] && [ "${MODEL_PATH}" != '""' ]; then
    ARGS+=(--model_path "${MODEL_PATH}")
fi

echo "[yolo_wrapper] node=${NODE_NAME}"
echo "[yolo_wrapper] task_mode=${TASK_MODE}"
echo "[yolo_wrapper] model=${MODEL_PATH:-preset:${DETECT_MODE}}"
echo "[yolo_wrapper] selection=${TARGET_SELECTION_MODE}"
echo "[yolo_wrapper] candidate_conf=${CANDIDATE_CONF_THRE}, default_class_conf=${CONF_THRE}"
echo "[yolo_wrapper] class_conf_thresholds=${CLASS_CONF_THRESHOLDS}"
echo "[yolo_wrapper] max_det=${MAX_DET}, web_top_k=${TOP_K}"
echo "[yolo_wrapper] python=${PYTHON_BIN}"
echo "[yolo_wrapper] script=${SCRIPT}"

exec "${PYTHON_BIN}" "${SCRIPT}" "${ARGS[@]}" "$@"
