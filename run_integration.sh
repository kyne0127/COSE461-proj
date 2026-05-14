#!/usr/bin/env bash
# run_integration.sh — 실제 모델 통합 테스트 실행 + 타임스탬프 로그 저장
# Usage: ./run_integration.sh [pytest 추가 옵션]
#   예) ./run_integration.sh -k "TestEndToEnd"

set -euo pipefail
cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pytest_integration_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Integration Tests  $(date '+%Y-%m-%d %H:%M:%S') ===" | tee "$LOG_FILE"
echo "Log: $LOG_FILE"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null || echo 'N/A')"
echo ""

python -m pytest tests/test_integration.py -m integration -v -s "$@" 2>&1 | tee -a "$LOG_FILE"
EXIT=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "=== Finished: exit=$EXIT  log=$LOG_FILE ===" | tee -a "$LOG_FILE"
exit $EXIT
