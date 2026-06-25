#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
python main.py --mode smoke --device "${DEVICE:-cuda}" --seed "${SEED:-1}" "${@}"
