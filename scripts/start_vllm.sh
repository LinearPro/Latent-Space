#!/bin/bash
set -euo pipefail

# Usage:
#   MODEL_PATH=/path/to/model scripts/start_vllm.sh
#   scripts/start_vllm.sh --model-path /path/to/model --served-model-name Qwen3-Thinking --gpus 0,1 --ports 8011,8022

MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-Thinking}"
GPUS="${GPUS:-0,1}"
PORTS="${PORTS:-8011,8022}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-50000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
REASONING_PARSER="${REASONING_PARSER:-deepseek_r1}"
ENABLE_REASONING="${ENABLE_REASONING:-1}"
LOG_DIR="${LOG_DIR:-logs}"
FOLLOW_LOG="${FOLLOW_LOG:-1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --served-model-name) SERVED_MODEL_NAME="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --ports) PORTS="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEMORY_UTILIZATION="$2"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL_SIZE="$2"; shift 2 ;;
        --reasoning-parser) REASONING_PARSER="$2"; shift 2 ;;
        --disable-reasoning) ENABLE_REASONING="0"; shift ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --no-follow-log) FOLLOW_LOG="0"; shift ;;
        -h|--help)
            sed -n '1,40p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "MODEL_PATH is required. Pass --model-path /path/to/model or set MODEL_PATH." >&2
    exit 1
fi

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
IFS=',' read -ra PORT_ARRAY <<< "$PORTS"

if [[ "${#GPU_ARRAY[@]}" -ne "${#PORT_ARRAY[@]}" ]]; then
    echo "--gpus and --ports must contain the same number of items." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

for idx in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$idx]}"
    port="${PORT_ARRAY[$idx]}"
    log_file="${LOG_DIR}/vllm_${port}.log"
    echo "Starting vLLM instance ${idx}: GPU ${gpu}, port ${port}, log ${log_file}"

    cmd=(
        python -m vllm.entrypoints.openai.api_server
        --model "$MODEL_PATH"
        --served-model-name "$SERVED_MODEL_NAME"
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
        --trust-remote-code
        --max-model-len "$MAX_MODEL_LEN"
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
        --port "$port"
    )

    if [[ "$ENABLE_REASONING" == "1" ]]; then
        cmd+=(--enable-reasoning --reasoning-parser "$REASONING_PARSER")
    fi

    CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1 &
done

echo "vLLM launch commands submitted."
for idx in "${!GPU_ARRAY[@]}"; do
    echo "  part $((idx + 1)): http://localhost:${PORT_ARRAY[$idx]} using GPU ${GPU_ARRAY[$idx]}"
done

if [[ "$FOLLOW_LOG" == "1" ]]; then
    echo "Following ${LOG_DIR}/vllm_${PORT_ARRAY[0]}.log. Press Ctrl+C to stop following logs; servers keep running."
    sleep 2
    tail -f "${LOG_DIR}/vllm_${PORT_ARRAY[0]}.log"
fi
