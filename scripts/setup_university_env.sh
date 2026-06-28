#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_DIR="${ENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON:-python3}"
TORCH_VERSION="${TORCH_VERSION:-2.2.0}"
TORCH_CUDA="${TORCH_CUDA:-cu118}"

echo "== Preflight =="
command -v git >/dev/null 2>&1 || {
  echo "git is missing. Install it with your package manager or load the university module first."
  exit 1
}
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "$PYTHON_BIN is missing. Load/install Python 3.10+ first."
  exit 1
}

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required.")
if sys.version_info >= (3, 13):
    print("WARNING: Python 3.13+ may not match the pinned dependencies. Python 3.10 or 3.11 is preferred.")
print(f"Python OK: {sys.version.split()[0]}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found. CUDA smoke/core runs will fail unless the scheduler exposes a GPU later."
fi

echo "== Creating virtual environment: $ENV_DIR =="
"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

case "$TORCH_CUDA" in
  cu118|cu121|cpu)
    ;;
  *)
    echo "Unsupported TORCH_CUDA=$TORCH_CUDA. Use cu118, cu121, or cpu."
    exit 1
    ;;
esac

echo "== Installing PyTorch $TORCH_VERSION ($TORCH_CUDA) =="
python -m pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"

echo "== Installing project dependencies =="
python -m pip install -r requirements_project.txt

echo "== Import and CUDA check =="
python - <<'PY'
import torch
import transformers
import numpy

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("transformers:", transformers.__version__)
print("numpy:", numpy.__version__)
PY

echo
echo "Environment ready. Activate it with:"
echo "  source $ENV_DIR/bin/activate"
echo
echo "Next checks:"
echo "  bash scripts/smoke_all.sh"
echo "  bash scripts/run_core.sh"
