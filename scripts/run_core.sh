#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs results

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/${RUN_TS}_core"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON:-python}"
DEVICE_VALUE="${DEVICE:-cuda}"
SEED_VALUE="${SEED:-1}"
NUM_WORKERS_VALUE="${NUM_WORKERS:-2}"

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
  echo "=== core ${part_dir} ==="
  "$PYTHON_BIN" "$part_dir/main.py" --mode core "${COMMON_ARGS[@]}" 2>&1 | tee "$log_file"
}

run_part "LM/partA"
run_part "LM/partB"
run_part "NLU/partA"
run_part "NLU/partB"

"$PYTHON_BIN" scripts/collect_results.py 2>&1 | tee "$LOG_DIR/collect_results.log"

echo "Core suite completed. Logs: ${LOG_DIR#$ROOT/}"
