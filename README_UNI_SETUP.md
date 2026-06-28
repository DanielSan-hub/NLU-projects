# University GPU Setup

Use this guide on the university Linux machine before launching the unattended experiments.

## 1. Connect And Check The Machine

```bash
ssh <user>@<university-host>
hostname
nvidia-smi
```

Expected target: NVIDIA Tesla V100 with about 16 GB GPU memory.

If the university uses a module system, inspect and load the available tools:

```bash
module avail
module load git
module load python/3.10
module load cuda/11.8
```

Module names vary by cluster. If `module` does not exist, skip this part.

If you have sudo on a plain Ubuntu machine:

```bash
sudo apt update
sudo apt install -y git tmux python3 python3-venv python3-pip
```

If you do not have sudo, use the provided Python/module/conda installation from the university.

## 2. Clone The Repository

```bash
cd ~
git clone https://github.com/DanielSan-hub/NLU-projects.git
cd NLU-projects
```

## 3. Create The Python Environment

Default setup uses PyTorch 2.2.0 with CUDA 11.8 wheels, which is usually a conservative choice for V100 machines:

```bash
bash scripts/setup_university_env.sh
source .venv/bin/activate
```

If `nvidia-smi` shows a recent driver and CUDA 12.1 is preferred:

```bash
TORCH_CUDA=cu121 bash scripts/setup_university_env.sh
source .venv/bin/activate
```

If the GPU is not visible yet because you are on a login node, create the environment anyway, then request an interactive GPU job before running smoke/core.

## 4. Quick Verification

```bash
python - <<'PY'
import torch
import transformers
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(transformers.__version__)
PY
```

CUDA must print `True` before running with `--device cuda`.

## 5. Run Smoke In Tmux

```bash
tmux new -s nlu_exam
cd ~/NLU-projects
source .venv/bin/activate
bash scripts/smoke_all.sh
```

Detach from tmux:

```text
Ctrl-b d
```

Reattach:

```bash
tmux attach -t nlu_exam
```

## 6. Run Core Experiments

After smoke passes:

```bash
source .venv/bin/activate
bash scripts/run_core.sh
```

Resume after interruption:

```bash
source .venv/bin/activate
RESUME=1 bash scripts/run_core.sh
```

## 7. Monitor Outputs

```bash
tail -f logs/*_core/*.log
python scripts/collect_results.py
python scripts/validate_submission.py
```

Important outputs:

- `logs/<timestamp>_smoke/`
- `logs/<timestamp>_core/`
- `results/master_results.csv`
- `results/summary.md`
- per-part `results/` folders with checkpoints and configs

## 8. Common Fixes

If CUDA is unavailable:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If `torch.cuda.is_available()` is `False`, make sure you are on a GPU node, not a login node.

If the CUDA wheel is incompatible with the driver, recreate the environment with the other wheel:

```bash
rm -rf .venv
TORCH_CUDA=cu121 bash scripts/setup_university_env.sh
```

or:

```bash
rm -rf .venv
TORCH_CUDA=cu118 bash scripts/setup_university_env.sh
```
