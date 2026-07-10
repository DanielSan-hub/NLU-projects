#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs results

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/${RUN_TS}_full"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON:-python}"
DEVICE_VALUE="${DEVICE:-cuda}"
SEED_VALUE="${SEED:-1}"
NUM_WORKERS_VALUE="${NUM_WORKERS:-2}"

check_core_results() {
  ROOT_FOR_CHECK="$ROOT" "$PYTHON_BIN" - <<'PY'
import csv
import os
import sys
from pathlib import Path

root = Path(os.environ["ROOT_FOR_CHECK"])
required = [
    "LM/partA/results/results_partA.csv",
    "LM/partB/results/results_partB.csv",
    "NLU/partA/results/results_partA.csv",
    "NLU/partB/results/results_partB.csv",
]
missing = []
without_core_rows = []
for rel in required:
    path = root / rel
    if not path.exists():
        missing.append(rel)
        continue
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not any((row.get("mode") or "").strip().lower() == "core" for row in rows):
        without_core_rows.append(rel)

if missing or without_core_rows:
    print("Core results are not complete. Run scripts/run_core.sh before scripts/run_extras.sh.")
    for rel in missing:
        print(f"missing: {rel}")
    for rel in without_core_rows:
        print(f"no core rows: {rel}")
    sys.exit(1)
PY
}

check_core_results 2>&1 | tee "$LOG_DIR/preflight.log"

COMMON_ARGS=(--device "$DEVICE_VALUE" --seed "$SEED_VALUE" --num-workers "$NUM_WORKERS_VALUE")
if [[ "$DEVICE_VALUE" == "cpu" || "${ALLOW_CPU:-0}" == "1" ]]; then
  COMMON_ARGS+=(--allow-cpu)
fi
if [[ "${AMP:-1}" == "1" ]]; then
  COMMON_ARGS+=(--amp)
fi
if [[ "${PIN_MEMORY:-1}" == "1" ]]; then
  COMMON_ARGS+=(--pin-memory)
fi
if [[ "${TENSORBOARD:-0}" == "1" ]]; then
  COMMON_ARGS+=(--log-tensorboard)
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  COMMON_ARGS+=(--resume)
fi

run_part() {
  local part_dir="$1"
  local log_name="${part_dir//\//_}"
  local log_file="$LOG_DIR/${log_name}.log"
  echo "=== full ${part_dir} ==="
  "$PYTHON_BIN" "$part_dir/main.py" --mode full "${COMMON_ARGS[@]}" 2>&1 | tee "$log_file"
}

run_part "LM/partA"
run_part "LM/partB"
run_part "NLU/partA"
run_part "NLU/partB"

"$PYTHON_BIN" scripts/collect_results.py 2>&1 | tee "$LOG_DIR/collect_results.log"

echo "Full suite completed. Logs: ${LOG_DIR#$ROOT/}"
