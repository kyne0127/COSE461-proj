#!/usr/bin/env bash
# =============================================================================
# scripts/setup_env.sh
# RunPod 재시작 후 환경 복구 스크립트
#
# 사용법:
#   bash scripts/setup_env.sh              # 전체 설치 (Molmo 다운로드 포함)
#   bash scripts/setup_env.sh --skip-molmo # Molmo 다운로드 건너뜀
#
# 수행 내용:
#   1. pip 패키지 설치 (grpcio, transformers, peft, outlines 등)
#   2. AmbRes 패키지 editable 설치 (/workspace/AmbRes)
#   3. lerobot-remote module editable 설치 (module/)
#   4. HF_HOME → /workspace/hf_cache 로 이동 (재시작 후에도 유지)
#   5. Molmo-7B-D-0924 다운로드 (없을 경우)
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AMBRES_ROOT="/workspace/AmbRes"
HF_CACHE_DIR="/workspace/hf_cache"
SKIP_MOLMO=false

# ── 인수 파싱 ──────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --skip-molmo) SKIP_MOLMO=true ;;
    *) echo "[warn] 알 수 없는 옵션: $arg" ;;
  esac
done

# ── 출력 헬퍼 ──────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${GREEN}[$(date +%H:%M:%S)] ▶ $1${NC}"; }
warn() { echo -e "${YELLOW}[warn] $1${NC}"; }
ok()   { echo -e "${GREEN}    ✓ $1${NC}"; }

# ── 0. 사전 확인 ───────────────────────────────────────────────────────────
step "환경 확인"
echo "  Python : $(python3 --version)"
echo "  pip    : $(pip --version | awk '{print $2}')"
echo "  REPO   : $REPO_ROOT"

if [[ ! -d "$AMBRES_ROOT" ]]; then
  echo -e "${RED}[error] /workspace/AmbRes 없음 — git clone 필요${NC}"
  echo "  git clone https://github.com/robot-learning-freiburg/AmbRes.git /workspace/AmbRes"
  exit 1
fi

# ── 1. pip 패키지 설치 ─────────────────────────────────────────────────────
step "pip 패키지 설치"

pip install --quiet \
  "transformers==4.48" \
  "peft>=0.9.0" \
  "accelerate>=0.26.0" \
  "datasets" \
  "sentencepiece" \
  "einops" \
  "outlines==0.1.14" \
  "shortuuid" \
  "jsonlines" \
  "opencv-python-headless" \
  "wandb" \
  "grpcio>=1.60.0" \
  "grpcio-tools>=1.60.0" \
  "pytest" \
  "pytest-mock" \
  "pyyaml" \
  "safetensors" \
  "huggingface_hub"

ok "pip 패키지 완료"

# ── 2. AmbRes editable 설치 ────────────────────────────────────────────────
step "AmbRes 패키지 설치 (editable)"

# pyproject.toml이 패키지 디렉터리로 설정되어 있는지 확인
if ! grep -q "packages.find" "$AMBRES_ROOT/pyproject.toml" 2>/dev/null; then
  warn "pyproject.toml에 packages.find 없음 — 패치 적용"
  sed -i 's/py-modules = \["ambres"\]//' "$AMBRES_ROOT/pyproject.toml"
  cat >> "$AMBRES_ROOT/pyproject.toml" << 'EOF'

[tool.setuptools.packages.find]
where = ["."]
include = ["ambres*"]
EOF
fi

# outlines 버전 고정이 pyproject.toml에 반영되어 있는지 확인
if grep -q '"outlines",' "$AMBRES_ROOT/pyproject.toml" 2>/dev/null; then
  sed -i 's/"outlines",/"outlines==0.1.14",/' "$AMBRES_ROOT/pyproject.toml"
fi

pip install --quiet -e "$AMBRES_ROOT" --no-deps
ok "AmbRes 설치 완료"

# ── 3. lerobot-remote module editable 설치 ────────────────────────────────
step "lerobot-remote module 설치 (editable)"

if [[ -f "$REPO_ROOT/module/pyproject.toml" ]]; then
  pip install --quiet -e "$REPO_ROOT/module" --no-deps
  ok "module 설치 완료"
else
  warn "module/pyproject.toml 없음 — 건너뜀"
fi

# ── 4. HF cache를 /workspace로 이동 ───────────────────────────────────────
step "HuggingFace 캐시 설정 (/workspace/hf_cache)"

mkdir -p "$HF_CACHE_DIR"

OLD_CACHE="$HOME/.cache/huggingface"
if [[ -d "$OLD_CACHE/hub/models--allenai--Molmo-7B-D-0924" ]] \
   && [[ ! -d "$HF_CACHE_DIR/hub/models--allenai--Molmo-7B-D-0924" ]]; then
  echo "  Molmo 캐시 이동 중 ($OLD_CACHE → $HF_CACHE_DIR) ..."
  echo "  크기: $(du -sh "$OLD_CACHE/hub/models--allenai--Molmo-7B-D-0924" 2>/dev/null | cut -f1)"
  mkdir -p "$HF_CACHE_DIR/hub"
  mv "$OLD_CACHE/hub/models--allenai--Molmo-7B-D-0924" "$HF_CACHE_DIR/hub/"
  ok "Molmo 캐시 이동 완료"
elif [[ -d "$HF_CACHE_DIR/hub/models--allenai--Molmo-7B-D-0924" ]]; then
  ok "Molmo 캐시 이미 /workspace/hf_cache에 있음"
fi

# HF_HOME 환경변수를 bashrc/profile에 영구 등록
HF_EXPORT_LINE="export HF_HOME=$HF_CACHE_DIR"
for rc_file in "$HOME/.bashrc" "$HOME/.profile"; do
  if [[ -f "$rc_file" ]] && ! grep -q "HF_HOME=$HF_CACHE_DIR" "$rc_file" 2>/dev/null; then
    echo "$HF_EXPORT_LINE" >> "$rc_file"
    ok "HF_HOME 등록: $rc_file"
  fi
done
export HF_HOME="$HF_CACHE_DIR"
ok "HF_HOME=$HF_CACHE_DIR (현재 세션에 적용)"

# ── 5. Molmo 모델 다운로드 ─────────────────────────────────────────────────
step "Molmo-7B-D-0924 모델 확인"

MOLMO_DIR="$HF_CACHE_DIR/hub/models--allenai--Molmo-7B-D-0924"

if [[ "$SKIP_MOLMO" == "true" ]]; then
  warn "--skip-molmo 옵션: Molmo 다운로드 건너뜀"
elif [[ -d "$MOLMO_DIR" ]]; then
  ok "Molmo 이미 있음: $MOLMO_DIR"
  echo "  크기: $(du -sh "$MOLMO_DIR" 2>/dev/null | cut -f1)"
else
  echo "  Molmo-7B 다운로드 시작 (~30GB, 시간이 걸립니다)..."
  python3 -c "
import os
os.environ['HF_HOME'] = '$HF_CACHE_DIR'
from huggingface_hub import snapshot_download
snapshot_download('allenai/Molmo-7B-D-0924', local_dir_use_symlinks=False)
print('다운로드 완료')
"
  ok "Molmo 다운로드 완료"
fi

# ── 6. AmbRes 체크포인트 확인 ─────────────────────────────────────────────
step "AmbRes 체크포인트 확인"

CKPT_REAL="$AMBRES_ROOT/ckpt/43qazb3XcrZF5rZWnjRPVm/checkpoint-200"
CKPT_SIM="$AMBRES_ROOT/ckpt/TtVVGPRoknCTELVSjH92xX/checkpoint-200"

if [[ -d "$CKPT_REAL" ]]; then
  ok "Real checkpoint: $CKPT_REAL"
else
  warn "Real checkpoint 없음 — 다운로드 필요"
  echo "  wget https://ambres.cs.uni-freiburg.de/download/ckpt.zip -O /tmp/ckpt.zip"
  echo "  python3 -c \"import zipfile; zipfile.ZipFile('/tmp/ckpt.zip').extractall('$AMBRES_ROOT')\""
fi

if [[ -d "$CKPT_SIM" ]]; then
  ok "Sim checkpoint: $CKPT_SIM"
fi

# ── 7. import 테스트 ───────────────────────────────────────────────────────
step "패키지 import 테스트"

python3 - << 'PYEOF'
import sys
results = []

packages = [
    ("grpc",        "import grpc"),
    ("transformers","import transformers"),
    ("peft",        "import peft"),
    ("outlines",    "import outlines.processors; outlines.processors.JSONLogitsProcessor"),
    ("ambres",      "from ambres.ambres_model import AmbresFSPrompt"),
    ("torch",       "import torch; _ = torch.__version__"),
    ("cv2",         "import cv2"),
    ("yaml",        "import yaml"),
]

all_ok = True
for name, code in packages:
    try:
        exec(code)
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        all_ok = False

import torch
print(f"\n  torch.cuda.is_available(): {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  → CUDA 불가 (Pod 재시작 직후라면 정상, GPU 기동 후 재확인)")

sys.exit(0 if all_ok else 1)
PYEOF

# ── 완료 ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  설치 완료!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  다음 단계:"
echo "  1. source ~/.bashrc   (HF_HOME 환경변수 적용)"
echo "  2. python scripts/run_server.py   (gRPC 서버 기동)"
echo "  3. python scripts/check_connection.py   (연결 확인)"
echo ""
