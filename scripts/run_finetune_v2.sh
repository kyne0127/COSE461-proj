#!/usr/bin/env bash
# Re-fine-tuning v2: G₀-absent 샘플 추가된 데이터셋으로 재학습
#
# 변경점 vs v1:
#   - --no-g0-augment: 각 trial에 G₀ 없는 버전 추가 (count rule label)
#   - 출력 디렉터리: dataset/finetune_v2, ckpt/grounding_ft_v2
#
# 사용법:
#   bash scripts/run_finetune_v2.sh              # 새 학습
#   bash scripts/run_finetune_v2.sh <run_id>     # resume

set -euo pipefail
cd "$(dirname "$0")/.."

AMBRES_CKPT="/workspace/AmbRes/ckpt/AB5siP8DA78aA78wR5Y8Mw/checkpoint-390"
MANIFEST="dataset/manifest_train_v3.jsonl"
MANUAL_ANN="dataset/manual_annotations.json"
TRAIN_JSONL="dataset/finetune_v2/train.jsonl"
VAL_JSONL="dataset/finetune_v2/val.jsonl"
OUT_DIR="ckpt/grounding_ft_v2"
BUILD_LOG="logs/finetune_v2_build.log"
TRAIN_LOG="logs/finetune_v2_train.log"
RUN_ID="${1:-}"

mkdir -p logs "$OUT_DIR" dataset/finetune_v2

# ── Step 1: 데이터셋 빌드 ────────────────────────────────────────────────────
if [ -f "$TRAIN_JSONL" ] && [ -f "$VAL_JSONL" ]; then
    echo "[build] train.jsonl, val.jsonl 이미 존재 — skip"
    echo "[build] 재빌드하려면 dataset/finetune_v2/*.jsonl 삭제 후 재실행"
else
    echo "[build] 데이터셋 빌드 시작 (G₀-absent 포함)... (로그: $BUILD_LOG)"
    HF_HOME=/workspace/hf_cache \
    python3 src/finetune/build_dataset.py \
        --manifests "$MANIFEST" \
        --out-dir dataset/finetune_v2 \
        --manual-annotations "$MANUAL_ANN" \
        --aug-flip --aug-brightness --aug-coord-noise \
        --no-g0-augment \
        --counterfactual \
        --train-ratio 0.8 \
        2>&1 | tee "$BUILD_LOG"
    echo "[build] 완료"
fi

# 샘플 수 확인
echo "[build] 데이터셋 크기:"
wc -l "$TRAIN_JSONL" "$VAL_JSONL"

# G₀ 있는/없는 샘플 분포 확인
python3 -c "
import json
train = [json.loads(l) for l in open('$TRAIN_JSONL')]
val   = [json.loads(l) for l in open('$VAL_JSONL')]
no_g0 = sum(1 for e in train if e.get('meta', {}).get('no_g0'))
cf    = sum(1 for e in train if e.get('meta', {}).get('counterfactual'))
base  = len(train) - no_g0 - cf
print(f'  train: {len(train)} (base+aug={base}, no_g0={no_g0}, counterfactual={cf})')
print(f'  val:   {len(val)} (base only)')
from collections import Counter
dist = Counter(e['gold_state'] for e in train)
print('  Train gold state:')
for s, c in sorted(dist.items()):
    print(f'    {s:<35} {c}')
"

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
echo "[train] 체크포인트 위치:"
ls -d "$OUT_DIR"/*/checkpoint-* 2>/dev/null || echo "  (체크포인트 없음)"
