#!/bin/bash
# ============================================================
# proto 스텁 생성 스크립트
# 사용법: bash scripts/gen_proto.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_ROOT/module/proto"
PROTO_FILE="$PROTO_DIR/lerobot.proto"

echo "[gen_proto] Checking grpcio-tools ..."
python -c "import grpc_tools" 2>/dev/null || {
    echo "[gen_proto] Installing grpcio-tools ..."
    pip install grpcio-tools --quiet
}

echo "[gen_proto] Generating stubs from $PROTO_FILE ..."
python -m grpc_tools.protoc \
    -I "$PROTO_DIR" \
    --python_out="$PROTO_DIR" \
    --grpc_python_out="$PROTO_DIR" \
    "$PROTO_FILE"

echo "[gen_proto] ✓ Generated:"
ls "$PROTO_DIR"/*.py 2>/dev/null | head -10
