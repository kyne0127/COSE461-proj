#!/usr/bin/env bash
# run_tests.sh — 단위 테스트 실행 + 타임스탬프 로그 저장
# Usage: ./run_tests.sh [pytest 추가 옵션]
#   예) ./run_tests.sh -k "test_role"
#       ./run_tests.sh --tb=long

set -euo pipefail
cd "$(dirname "$0")"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pytest_${TIMESTAMP}.log"
mkdir -p logs

echo "=== Unit Tests  $(date '+%Y-%m-%d %H:%M:%S') ===" | tee "$LOG_FILE"
echo "Log: $LOG_FILE"
echo ""

python -m pytest tests/ -m "not integration" -v "$@" 2>&1 | tee -a "$LOG_FILE"
EXIT=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "=== Finished: exit=$EXIT  log=$LOG_FILE ===" | tee -a "$LOG_FILE"
exit $EXIT
