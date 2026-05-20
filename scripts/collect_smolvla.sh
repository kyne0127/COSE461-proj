#!/usr/bin/env bash
# scripts/collect_smolvla.sh
# SmolVLA 학습용 데이터 수집 (lerobot-record 사용)
#
# 사용법:
#   ./scripts/collect_smolvla.sh                          # 기본값으로 실행
#   ./scripts/collect_smolvla.sh --task "pick up the cube" --n 20
#   TASK="pick up the banana" N_EP=10 ./scripts/collect_smolvla.sh

set -e

# ── 파라미터 ──────────────────────────────────────────────────────────────────
TASK="${TASK:-pick up the cube}"
N_EP="${N_EP:-20}"
DATASET_ID="${DATASET_ID:-kyne0127/smolvla_pick_cup}"
# root는 full 경로 (lerobot이 root 자체를 새로 생성)
DATA_ROOT="${DATA_ROOT:-data/lerobot_datasets/${DATASET_ID}}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"   # 최대 에피소드 길이 (→ 로 언제든 조기종료)
RESET_TIME_S="${RESET_TIME_S:-10}"       # 리셋 대기 (→ 로 즉시 다음 에피소드 시작)
FPS="${FPS:-30}"

# ── 포트 ──────────────────────────────────────────────────────────────────────
FOLLOWER_PORT="/dev/ttyACM2"
LEADER_PORT="/dev/ttyACM1"

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --task)  TASK="$2";      shift 2 ;;
        --n)     N_EP="$2";     shift 2 ;;
        --id)    DATASET_ID="$2"; shift 2 ;;
        --root)  DATA_ROOT="$2";  shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  SmolVLA 데이터 수집"
echo "  task        : ${TASK}"
echo "  n_episodes  : ${N_EP}"
echo "  dataset_id  : ${DATASET_ID}"
echo "  data_root   : ${DATA_ROOT}"
echo "  fps         : ${FPS}  |  episode_time: ${EPISODE_TIME_S}s  |  reset_time: ${RESET_TIME_S}s"
echo "  follower    : ${FOLLOWER_PORT}"
echo "  leader      : ${LEADER_PORT}"
echo "============================================================"
echo ""
echo "  카메라 설정:"
echo "    egoview    : /dev/video0  640×480  @60fps"
echo "    global_view: /dev/video4  1344×376 @30fps"
echo "============================================================"
echo ""

# tasks.parquet 가 있을 때만 resume (실제 데이터가 있는 경우)
RESUME_FLAG=""
if [ -f "${DATA_ROOT}/meta/tasks.parquet" ]; then
    echo "  [INFO] 기존 데이터셋 감지 → resume 모드로 이어서 수집합니다."
    RESUME_FLAG="--resume=true"
fi

conda run -n vla-nlp \
    python -m lerobot.scripts.lerobot_record \
        --robot.type=so101_follower \
        --robot.port="${FOLLOWER_PORT}" \
        --robot.cameras="{
            egoview:     {type: opencv, index_or_path: 0, width: 640,  height: 480, fps: 60},
            global_view: {type: opencv, index_or_path: 4, width: 1344, height: 376, fps: 30}
        }" \
        --teleop.type=so101_leader \
        --teleop.port="${LEADER_PORT}" \
        --dataset.repo_id="${DATASET_ID}" \
        --dataset.root="${DATA_ROOT}" \
        --dataset.push_to_hub=false \
        --dataset.fps="${FPS}" \
        --dataset.episode_time_s="${EPISODE_TIME_S}" \
        --dataset.reset_time_s="${RESET_TIME_S}" \
        --dataset.num_episodes="${N_EP}" \
        --dataset.single_task="${TASK}" \
        --dataset.video=true \
        --dataset.streaming_encoding=true \
        --dataset.encoder_threads=2 \
        ${RESUME_FLAG}
