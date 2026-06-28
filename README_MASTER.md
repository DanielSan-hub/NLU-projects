# NLU Exam Project Runner

This repository contains four independent mini-projects:

- `LM/partA`: scratch GPT2 language modeling on PennTreeBank
- `LM/partB`: pretrained GPT2 with manual LoRA on PennTreeBank
- `NLU/partA`: scratch GPT2 for ATIS intent classification and slot filling
- `NLU/partB`: pretrained BERT and GPT2 for ATIS multitask NLU

All entry points are `main.py` files and all paths are relative to the repository root.

## Running In Tmux

On the university GPU machine:

```bash
tmux new -s nlu_exam
cd /path/to/NLU-2026-Labs
bash scripts/smoke_all.sh
```

Detach without stopping the run:

```text
Ctrl-b d
```

Reattach later:

```bash
tmux attach -t nlu_exam
```

If a session already exists:

```bash
tmux ls
tmux attach -t nlu_exam
```

## Smoke, Core, Full

Smoke tests run one or two batches for every part and stop immediately on the first failure:

```bash
bash scripts/smoke_all.sh
```

Core runs the mandatory experiments sequentially on CUDA:

```bash
bash scripts/run_core.sh
```

Extras runs the optional extensions only after core result rows already exist, without rerunning core:

```bash
bash scripts/run_extras.sh
```

Useful environment overrides:

```bash
PYTHON=python3 DEVICE=cuda SEED=1 NUM_WORKERS=2 AMP=1 PIN_MEMORY=1 bash scripts/run_core.sh
TENSORBOARD=1 bash scripts/run_core.sh
DEVICE=cpu ALLOW_CPU=1 bash scripts/smoke_all.sh
```

## Resume

Every mini-project supports `--resume`. For the unattended scripts:

```bash
RESUME=1 bash scripts/run_core.sh
RESUME=1 bash scripts/run_extras.sh
```

Individual runs are also supported:

```bash
python LM/partA/main.py --mode core --device cuda --resume
python LM/partB/main.py --mode core --device cuda --resume
python NLU/partA/main.py --mode core --device cuda --resume
python NLU/partB/main.py --mode core --device cuda --resume
```

## Outputs

Per-part CSV files:

- `LM/partA/results/results_partA.csv`
- `LM/partB/results/results_partB.csv`
- `NLU/partA/results/results_partA.csv`
- `NLU/partB/results/results_partB.csv`

Master outputs from `scripts/collect_results.py`:

- `results/master_results.csv`
- `results/summary.md`

Checkpoints and run configs are saved under each part:

- `LM/partA/results/<timestamp>_<mode>_seed<seed>/`
- `LM/partB/results/<timestamp>_<mode>_seed<seed>/`
- `NLU/partA/results/<timestamp>_<mode>_seed<seed>/`
- `NLU/partB/results/<timestamp>_<mode>_seed<seed>/`

Each part also writes `latest_run.txt` in its `results/` directory.

Logs from unattended scripts are timestamped:

- `logs/<timestamp>_smoke/`
- `logs/<timestamp>_core/`
- `logs/<timestamp>_extras/`

Validate the submission layout and latest artifacts:

```bash
python scripts/validate_submission.py
```
