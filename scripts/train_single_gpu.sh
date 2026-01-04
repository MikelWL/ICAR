#!/bin/bash
# Single-GPU training for ICAR.
# - Baseline ViT-L-14 (no early exit)
# - Early-exit variants (layers 8/12/16/20)
#
# Usage:
#   scripts/train_single_gpu.sh coco all
#   scripts/train_single_gpu.sh coco baseline
#   scripts/train_single_gpu.sh coco variants
#   scripts/train_single_gpu.sh flickr all
#
# Note: For multi-GPU, replace the python calls with torchrun and --gpu-ids.

set -e

BASE_DIR=$(dirname $(dirname $(realpath $0)))
DATASET=${1:-"coco"}       # coco | flickr
MODE=${2:-"all"}           # baseline | variants | all
GPU_ID=${GPU_ID:-0}

LAYERS=(8 12 16 20)

COCO_ROOT=${COCO_ROOT:-"/path/to/coco-images"}
FLICKR_ROOT=${FLICKR_ROOT:-"/path/to/flickr30k"}

case "${DATASET}" in
    coco)
        MODEL_CONFIG="${BASE_DIR}/icar/configs/coco.yaml"
        DATA_ROOT="${COCO_ROOT}"
        BASELINE_DIR="${BASE_DIR}/checkpoints/baseline_coco"
        VARIANT_DIR="${BASE_DIR}/checkpoints/icar_coco"
        LOG_DIR="${BASE_DIR}/logs/train_coco"
        ;;
    flickr)
        MODEL_CONFIG="${BASE_DIR}/icar/configs/flickr30k.yaml"
        DATA_ROOT="${FLICKR_ROOT}"
        BASELINE_DIR="${BASE_DIR}/checkpoints/baseline_flickr"
        VARIANT_DIR="${BASE_DIR}/checkpoints/icar_flickr"
        LOG_DIR="${BASE_DIR}/logs/train_flickr"
        ;;
    *)
        echo "Unknown dataset: ${DATASET} (expected coco|flickr)"
        exit 1
        ;;
esac

mkdir -p "${LOG_DIR}"

run_baseline() {
    local log_file="${LOG_DIR}/baseline_${DATASET}.log"
    mkdir -p "${BASELINE_DIR}"

    echo "=============================================================="
    echo "Baseline training (${DATASET})"
    echo "Checkpoint dir: ${BASELINE_DIR}"
    echo "Log: ${log_file}"
    echo "=============================================================="

    python "${BASE_DIR}/scripts/train.py" \
        --config "${MODEL_CONFIG}" \
        --data-root "${DATA_ROOT}" \
        --baseline-only \
        --checkpoint-dir "${BASELINE_DIR}" \
        --gpu-id "${GPU_ID}" \
        2>&1 | tee "${log_file}"

    # Multi-GPU example (uncomment and edit):
    # torchrun --nproc_per_node=2 --master-port=29506 \
    #     "${BASE_DIR}/scripts/train.py" \
    #     --config "${MODEL_CONFIG}" \
    #     --data-root "${DATA_ROOT}" \
    #     --baseline-only \
    #     --checkpoint-dir "${BASELINE_DIR}" \
    #     --gpu-ids "0,1"
}

run_variants() {
    mkdir -p "${VARIANT_DIR}"

    for layer in "${LAYERS[@]}"; do
        local checkpoint_dir="${VARIANT_DIR}/layer_${layer}"
        local log_file="${LOG_DIR}/layer_${layer}_${DATASET}.log"
        mkdir -p "${checkpoint_dir}"

        echo "=============================================================="
        echo "Early-exit training (${DATASET}) - layer ${layer}"
        echo "Checkpoint dir: ${checkpoint_dir}"
        echo "Log: ${log_file}"
        echo "=============================================================="

        python "${BASE_DIR}/scripts/train.py" \
            --config "${MODEL_CONFIG}" \
            --data-root "${DATA_ROOT}" \
            --early-exit-layer "${layer}" \
            --checkpoint-dir "${checkpoint_dir}" \
            --gpu-id "${GPU_ID}" \
            2>&1 | tee "${log_file}"

        # Multi-GPU example (uncomment and edit):
        # torchrun --nproc_per_node=2 --master-port=29505 \
        #     "${BASE_DIR}/scripts/train.py" \
        #     --config "${MODEL_CONFIG}" \
        #     --data-root "${DATA_ROOT}" \
        #     --early-exit-layer "${layer}" \
        #     --checkpoint-dir "${checkpoint_dir}" \
        #     --gpu-ids "0,1"
    done
}

case "${MODE}" in
    baseline)
        run_baseline
        ;;
    variants)
        run_variants
        ;;
    all)
        run_baseline
        run_variants
        ;;
    *)
        echo "Unknown mode: ${MODE} (expected baseline|variants|all)"
        exit 1
        ;;
esac
