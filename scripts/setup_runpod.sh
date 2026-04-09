#!/bin/bash
# ============================================================
# scripts/setup_runpod.sh
# RunPod GPU Pod 초기 환경 설정 스크립트
#
# RunPod pod 접속 후 1회 실행:
#   bash scripts/setup_runpod.sh
#
# 또는 RunPod 템플릿 "On-Start Command"에 등록:
#   bash /workspace/scripts/setup_runpod.sh
# ============================================================
set -e

WORKSPACE="/workspace"
REPO_URL="${LEROBOT_REPO_URL:-}"        # 환경변수로 주입 가능
GRPC_PORT="${GRPC_PORT:-50051}"
LOG_DIR="/data/lerobot/logs"
DATA_DIR="/data/lerobot/datasets"
CKPT_DIR="/data/lerobot/checkpoints"

echo "========================================================"
echo "  LeRobot RunPod Setup"
echo "  $(date)"
echo "========================================================"

# ── 1. GPU 확인 ─────────────────────────────────────────────
echo ""
echo "[1/7] GPU 확인"
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi not available"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')" \
    2>/dev/null || echo "  PyTorch not yet installed"

# ── 2. 디렉터리 생성 ─────────────────────────────────────────
echo ""
echo "[2/7] 디렉터리 생성"
mkdir -p "$LOG_DIR" "$DATA_DIR" "$CKPT_DIR" "$WORKSPACE"
echo "  ✓ $DATA_DIR"
echo "  ✓ $CKPT_DIR"
echo "  ✓ $LOG_DIR"

# ── 3. 시스템 패키지 ─────────────────────────────────────────
echo ""
echo "[3/7] 시스템 패키지 설치"
apt-get update -qq
apt-get install -y -qq \
    git curl wget \
    libgl1-mesa-glx libglib2.0-0 \
    ffmpeg \
    openssh-client \
    > /dev/null
echo "  ✓ 시스템 패키지 설치 완료"

# ── 4. Python 패키지 ─────────────────────────────────────────
echo ""
echo "[4/7] Python 패키지 설치"

pip install --quiet --upgrade pip

# gRPC
pip install --quiet \
    grpcio>=1.60.0 \
    grpcio-tools>=1.60.0 \
    PyYAML>=6.0

# lerobot (official)
pip install --quiet "lerobot[all] @ git+https://github.com/huggingface/lerobot.git" \
    || pip install --quiet lerobot  # fallback to PyPI

# 추가 모델 의존성
pip install --quiet \
    diffusers>=0.27 \
    transformers>=4.40 \
    huggingface-hub>=0.22 \
    datasets>=2.18

echo "  ✓ Python 패키지 설치 완료"

# ── 5. 저장소 클론 ───────────────────────────────────────────
echo ""
echo "[5/7] 저장소 설정"
if [ -n "$REPO_URL" ]; then
    if [ -d "$WORKSPACE/.git" ]; then
        echo "  저장소가 이미 존재합니다. git pull ..."
        cd "$WORKSPACE" && git pull --quiet
    else
        echo "  클론 중: $REPO_URL"
        git clone --quiet "$REPO_URL" "$WORKSPACE"
    fi
    echo "  ✓ 저장소 준비 완료"
else
    echo "  LEROBOT_REPO_URL 미설정 — 스킵 (수동으로 코드를 업로드하세요)"
fi

# ── 6. proto 스텁 생성 ───────────────────────────────────────
echo ""
echo "[6/7] gRPC proto 스텁 생성"
cd "$WORKSPACE"
if [ -f "module/proto/lerobot.proto" ]; then
    python3 -m grpc_tools.protoc \
        -I module/proto \
        --python_out=module/proto \
        --grpc_python_out=module/proto \
        module/proto/lerobot.proto
    echo "  ✓ proto 스텁 생성 완료"
    ls module/proto/*.py 2>/dev/null | head -5 | sed 's/^/    /'
else
    echo "  WARNING: module/proto/lerobot.proto 없음 — 스킵"
fi

# ── 7. 서버 시작 ─────────────────────────────────────────────
echo ""
echo "[7/7] gRPC 서버 시작"
cd "$WORKSPACE"
LOG_FILE="$LOG_DIR/server.log"

# 이미 실행 중인 서버 종료
pkill -f "run_server.py" 2>/dev/null || true
sleep 1

# 백그라운드로 서버 시작
nohup python3 scripts/run_server.py \
    --config module/config/server.yaml \
    --port "$GRPC_PORT" \
    --log-level INFO \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "  서버 PID: $SERVER_PID"
sleep 3

# 서버 기동 확인
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "  ✓ gRPC 서버 실행 중 (포트 $GRPC_PORT)"
    echo "  로그: $LOG_FILE"
else
    echo "  ERROR: 서버 시작 실패. 로그 확인:"
    tail -20 "$LOG_FILE"
    exit 1
fi

# ── 완료 ─────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Setup 완료!"
echo ""
echo "  gRPC 서버 : 0.0.0.0:$GRPC_PORT"
echo "  데이터    : $DATA_DIR"
echo "  체크포인트: $CKPT_DIR"
echo "  서버 로그 : $LOG_FILE"
echo ""
echo "  데스크탑에서 접속:"
echo "    python scripts/open_tunnel.py \\"
echo "        --pod-id \$RUNPOD_POD_ID \\"
echo "        --ssh-port \$RUNPOD_SSH_PORT"
echo "========================================================"
