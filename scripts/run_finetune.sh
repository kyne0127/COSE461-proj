#!/usr/bin/env bash
# Fine-tuning 파이프라인 실행 스크립트
#
# 사용법:
#   bash scripts/run_finetune.sh              # 처음 실행 (새 run_id 자동 생성)
#   bash scripts/run_finetune.sh <run_id>     # 기존 run 이어서 (체크포인트 자동 탐색)
#
# 로그 모니터링:
#   tail -f logs/finetune_build.log     # 데이터 빌드 로그
#   tail -f logs/finetune_train.log     # 학습 로그
#
# 종료 후 재시작해도 자동으로 이어서 진행합니다.

set -euo pipefail
cd "$(dirname "$0")/.."

AMBRES_CKPT="/workspace/AmbRes/ckpt/AB5siP8DA78aA78wR5Y8Mw/checkpoint-390"
TRAIN_JSONL="dataset/finetune/train.jsonl"
VAL_JSONL="dataset/finetune/val.jsonl"
OUT_DIR="ckpt/grounding_ft"
BUILD_LOG="logs/finetune_build.log"
TRAIN_LOG="logs/finetune_train.log"
RUN_ID="${1:-}"

mkdir -p logs ckpt/grounding_ft dataset/finetune

# ── Step 1: 데이터셋 빌드 ────────────────────────────────────────────────────
if [ -f "$TRAIN_JSONL" ] && [ -f "$VAL_JSONL" ]; then
    echo "[build] train.jsonl, val.jsonl 이미 존재 — skip"
    echo "[build] 데이터를 새로 빌드하려면 dataset/finetune/*.jsonl 을 삭제 후 재실행"
else
    echo "[build] 데이터셋 빌드 시작... (로그: $BUILD_LOG)"
    HF_HOME=/workspace/hf_cache \
    python3 src/finetune/build_dataset.py \
        --out-dir dataset/finetune \
        --aug-flip --aug-brightness --aug-coord-noise \
        --resume \
        2>&1 | tee "$BUILD_LOG"
    echo "[build] 완료"
fi

# ── Step 2: 학습 ─────────────────────────────────────────────────────────────
echo "[train] 학습 시작... (로그: $TRAIN_LOG)"
if [ -n "$RUN_ID" ]; then
    echo "[train] run_id=$RUN_ID 에서 resume"
    RESUME_ARG="--run-id $RUN_ID"
else
    RESUME_ARG=""
fi

HF_HOME=/workspace/hf_cache \
python3 src/finetune/train.py \
    --ambres-ckpt "$AMBRES_CKPT" \
    --train-jsonl "$TRAIN_JSONL" \
    --val-jsonl   "$VAL_JSONL" \
    --out-dir     "$OUT_DIR" \
    --epochs 3 \
    --lr 1e-4 \
    --lora-r 8 \
    $RESUME_ARG \
    2>&1 | tee "$TRAIN_LOG"

echo "[train] 완료"
