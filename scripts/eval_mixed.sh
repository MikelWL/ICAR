#!/bin/bash
# Mixed-dataset evaluation (instance + category) for ICAR.
# Evaluates COCO/Flickr30k test sets with 100k LAION-COCO distractors.
# Runs ICC-routed variants (layers 8/12/16/20) and baseline ViT-L-14.

set -e

BASE_DIR=$(dirname $(dirname $(realpath $0)))
RESULTS_DIR="${BASE_DIR}/results/mixed"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Data paths (override by exporting environment variables before running)
COCO_ROOT=${COCO_ROOT:-"/path/to/coco-images"}
FLICKR_ROOT=${FLICKR_ROOT:-"/path/to/flickr30k"}
LAION_ROOT=${LAION_ROOT:-"${BASE_DIR}/data/laion_coco_100k"}
COMPLEXITY_SCORES=${COMPLEXITY_SCORES:-"${BASE_DIR}/data/laion_coco_100k_metadata/complexity_scores.json"}
ICC_CHECKPOINT=${ICC_CHECKPOINT:-"${BASE_DIR}/data/ICC.pt"}

# Category label paths (external; provide files if you want category metrics)
COCO_CATEGORIES=${COCO_CATEGORIES:-"${BASE_DIR}/data/coco_test_categories.json"}
FLICKR_CATEGORIES=${FLICKR_CATEGORIES:-"${BASE_DIR}/data/flickr30k_test_categories.json"}
LAION_CATEGORIES=${LAION_CATEGORIES:-"${BASE_DIR}/data/laion_coco_100k/category_labels_laion_coco_100k.json"}

LAYERS=(8 12 16 20)

DATASET_CHOICE=${1:-"all"}  # coco | flickr | all

run_eval() {
    local dataset=$1
    local base_root=$2
    local category_labels=$3
    local baseline_checkpoint=$4
    local early_exit_dir=$5

    local dataset_results_dir="${RESULTS_DIR}/${dataset}"
    mkdir -p "${dataset_results_dir}"

    echo "=============================================================="
    echo "Dataset: ${dataset}"
    echo "Results: ${dataset_results_dir}"
    echo "=============================================================="

    # Baseline (full path) evaluation
    if [ -f "${baseline_checkpoint}" ]; then
        echo "\n[Baseline] ${dataset} + LAION-COCO (instance + category)"
        python "${BASE_DIR}/scripts/evaluate_mixed_preprocessed.py" \
            --config "${BASE_DIR}/icar/configs/coco.yaml" \
            --checkpoint "${baseline_checkpoint}" \
            --base-dataset "${dataset}" \
            --base-data-root "${base_root}" \
            --laion-data-root "${LAION_ROOT}" \
            --complexity-scores "${COMPLEXITY_SCORES}" \
            --batch-size 512 \
            --output-dir "${dataset_results_dir}/baseline" \
            --output-file "baseline_${TIMESTAMP}.json" \
            --device cuda \
            --eval-category \
            --laion-categories "${LAION_CATEGORIES}" \
            --coco-categories "${COCO_CATEGORIES}" \
            --flickr-categories "${FLICKR_CATEGORIES}"
    else
        echo "Baseline checkpoint not found: ${baseline_checkpoint}"
    fi

    # ICC-routed early-exit variants
    for layer in "${LAYERS[@]}"; do
        local checkpoint_path="${early_exit_dir}/layer_${layer}/latest_checkpoint.pt"
        if [ ! -f "${checkpoint_path}" ]; then
            echo "Checkpoint not found: ${checkpoint_path}"
            continue
        fi

        echo "\n[ICC-Routed] Layer ${layer} on ${dataset} + LAION-COCO (instance + category)"
        python "${BASE_DIR}/scripts/evaluate_mixed_preprocessed.py" \
            --config "${BASE_DIR}/icar/configs/coco.yaml" \
            --checkpoint "${checkpoint_path}" \
            --base-dataset "${dataset}" \
            --base-data-root "${base_root}" \
            --laion-data-root "${LAION_ROOT}" \
            --complexity-scores "${COMPLEXITY_SCORES}" \
            --early-exit-layer "${layer}" \
            --use-icc-routing \
            --icc-checkpoint "${ICC_CHECKPOINT}" \
            --icc-threshold 0.5 \
            --batch-size 512 \
            --output-dir "${dataset_results_dir}/layer_${layer}" \
            --output-file "icc_routing_${TIMESTAMP}.json" \
            --device cuda \
            --eval-category \
            --laion-categories "${LAION_CATEGORIES}" \
            --coco-categories "${COCO_CATEGORIES}" \
            --flickr-categories "${FLICKR_CATEGORIES}"
    done
}

mkdir -p "${RESULTS_DIR}"

case "${DATASET_CHOICE}" in
    coco)
        run_eval "mscoco" "${COCO_ROOT}" "${COCO_CATEGORIES}" \
            "${BASE_DIR}/checkpoints/baseline_coco/latest_checkpoint.pt" \
            "${BASE_DIR}/checkpoints/icar_coco"
        ;;
    flickr)
        run_eval "flickr30k" "${FLICKR_ROOT}" "${FLICKR_CATEGORIES}" \
            "${BASE_DIR}/checkpoints/baseline_flickr/latest_checkpoint.pt" \
            "${BASE_DIR}/checkpoints/icar_flickr"
        ;;
    all)
        run_eval "mscoco" "${COCO_ROOT}" "${COCO_CATEGORIES}" \
            "${BASE_DIR}/checkpoints/baseline_coco/latest_checkpoint.pt" \
            "${BASE_DIR}/checkpoints/icar_coco"
        run_eval "flickr30k" "${FLICKR_ROOT}" "${FLICKR_CATEGORIES}" \
            "${BASE_DIR}/checkpoints/baseline_flickr/latest_checkpoint.pt" \
            "${BASE_DIR}/checkpoints/icar_flickr"
        ;;
    *)
        echo "Usage: $0 [coco|flickr|all]"
        exit 1
        ;;
esac

echo "=============================================================="
echo "Mixed evaluation complete. Results in: ${RESULTS_DIR}"
echo "=============================================================="
