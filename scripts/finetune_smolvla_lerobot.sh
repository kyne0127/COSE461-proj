#!/usr/bin/env bash
# scripts/finetune_smolvla_lerobot.sh
# SmolVLA fine-tuning via lerobot-train
#
# 사용법:
#   ./scripts/finetune_smolvla_lerobot.sh
#   STEPS=5000 BATCH=2 ./scripts/finetune_smolvla_lerobot.sh

set -e

DATASET_REPO_ID="${DATASET_REPO_ID:-kyne0127/smolvla_pick_cube}"
DATASET_ROOT="${DATASET_ROOT:-$(pwd)/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/checkpoints/smolvla_pick_cube_ft}"
STEPS="${STEPS:-3000}"
BATCH="${BATCH:-4}"
WARMUP="${WARMUP:-200}"
SAVE_STEPS="${SAVE_STEPS:-500}"
BASE_MODEL="${BASE_MODEL:-lerobot/smolvla_base}"

echo "============================================================"
echo "  SmolVLA Fine-Tuning"
echo "  dataset  : ${DATASET_REPO_ID}"
echo "  root     : ${DATASET_ROOT}"
echo "  base     : ${BASE_MODEL}"
echo "  steps    : ${STEPS}  |  batch: ${BATCH}  |  warmup: ${WARMUP}"
echo "  save_dir : ${OUTPUT_DIR}"
echo "============================================================"

conda run -n vla-nlp \
    python -m lerobot.scripts.lerobot_train \
        --policy.type=smolvla \
        --policy.pretrained_path="${BASE_MODEL}" \
        --policy.freeze_vision_encoder=true \
        --policy.train_expert_only=true \
        --policy.train_state_proj=true \
        --policy.scheduler_warmup_steps="${WARMUP}" \
        --dataset.repo_id="${DATASET_REPO_ID}" \
        --dataset.root="${DATASET_ROOT}" \
        --batch_size="${BATCH}" \
        --steps="${STEPS}" \
        --save_freq="${SAVE_STEPS}" \
        --output_dir="${OUTPUT_DIR}"
