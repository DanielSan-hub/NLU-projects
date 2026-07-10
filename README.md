# Transformer and NLU Systems

Four reproducible mini-projects exploring transformer language modelling and multitask natural-language understanding: from a decoder-only model built from scratch to parameter-efficient GPT-2 adaptation with manually implemented LoRA.

This repository contains my implementation for the University of Trento Natural Language Understanding course. I keep the original lab context for attribution, but the portfolio focus is model internals, evaluation discipline, and reproducible experiment tooling.

## What I implemented

- A GPT-2-style causal language model from scratch, including masked self-attention, transformer blocks, training, checkpointing, and perplexity evaluation.
- Manual LoRA injection into GPT-2's fused QKV projection, with frozen-base fine-tuning and trainable-parameter accounting.
- A scratch decoder model for joint ATIS intent classification and slot filling.
- BERT- and GPT-2-based multitask NLU systems with tokenizer/subword alignment and `-100` masking for non-label positions.
- A shared experiment workflow with smoke, core, and optional runs; deterministic seeds; CSV collection; report figures; and validation scripts.

## Project map

| Project | Model | Task / data | Main evaluation |
| --- | --- | --- | --- |
| [`LM/partA`](LM/partA) | Scratch GPT-2 | Penn Treebank language modelling | Perplexity |
| [`LM/partB`](LM/partB) | Pretrained GPT-2 + manual LoRA | Penn Treebank language modelling | Perplexity, trainable parameters |
| [`NLU/partA`](NLU/partA) | Scratch GPT-2 multitask model | ATIS intent + slot filling | Intent accuracy, slot F1, frame accuracy |
| [`NLU/partB`](NLU/partB) | Pretrained BERT and GPT-2 | ATIS intent + slot filling | Intent accuracy, slot F1, frame accuracy |

Each linked directory contains implementation notes, design choices, commands, and expected artifacts.

## Verified results

The public repository includes report artifacts and lightweight run metadata. Checkpoints are intentionally excluded because several are too large for a portfolio repo; the metrics below are copied from the local report artifacts in [`reports/REPORT_NOTES.md`](reports/REPORT_NOTES.md), with smoke-run CSV metadata preserved under [`results/`](results).

| Area | Best verified result | Source |
| --- | --- | --- |
| Scratch LM | `ablation_d_model_192`: dev PPL `62.624`, test PPL `53.651` | [`reports/LM_partA_report.md`](reports/LM_partA_report.md) |
| GPT-2 LoRA LM | mandatory `rank8_alpha16_qkv`: dev PPL `30.114`, test PPL `27.059`; optional `extra_rank16_alpha32_qkv`: test PPL `25.811` | [`reports/LM_partB_report.md`](reports/LM_partB_report.md) |
| Scratch multitask NLU | `ablation_n_heads_8`: test intent `0.960`, test slot F1 `0.919`, frame accuracy `0.772` | [`reports/NLU_partA_report.md`](reports/NLU_partA_report.md) |
| Pretrained multitask NLU | BERT core: test intent `0.976`, test slot F1 `0.951`, frame accuracy `0.868`; ontology run: frame accuracy `0.878` | [`reports/NLU_partB_report.md`](reports/NLU_partB_report.md) |

The checked-in [`results/master_results.csv`](results/master_results.csv) is a smoke-run aggregation snapshot generated on 2026-06-25. It is useful for validating the collection pipeline, while the report notes summarize the final university-machine runs.

## Five-minute review

1. Start with [`PROJECT_CARD.md`](PROJECT_CARD.md) for the design and evaluation overview.
2. Inspect [`LM/partB/model.py`](LM/partB/model.py) for the manual LoRA implementation.
3. Inspect [`NLU/partB/utils.py`](NLU/partB/utils.py) for subtoken/label alignment.
4. Inspect [`NLU/partA/model.py`](NLU/partA/model.py) for the scratch multitask architecture.
5. Run the dependency-free public snapshot check below.

## Reproduce and validate

Create an environment for the project code:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_project.txt
```

Run the lightweight repository check, which requires no model downloads or GPU:

```bash
python scripts/validate_public_snapshot.py
```

Run all short end-to-end checks:

```bash
DEVICE=cpu ALLOW_CPU=1 bash scripts/smoke_all.sh
```

On a CUDA machine, execute the mandatory experiments and collect their outputs:

```bash
bash scripts/run_core.sh
python scripts/collect_results.py
python scripts/validate_submission.py
```

See [`README_MASTER.md`](README_MASTER.md) for resume, `tmux`, TensorBoard, and optional-run instructions.

## Artifact policy

- Tracked: source code, datasets used by the course project, lab context, report notes, figures, CSV summaries, configuration JSON files, and small logs needed to verify the reported workflow.
- Not tracked: trained checkpoints (`*.pt`, `*.pth`, `*.ckpt`, `*.bin`, `*.safetensors`), virtual environments, caches, and local TensorBoard runs.

## Authorship and course context

My project implementation lives in `LM/`, `NLU/`, `scripts/`, `reports/`, and the generated result metadata. The `labs/`, `solutions/`, `exam/`, environment files, and parts of the assignment structure originate from the University of Trento NLU course repository and remain here for context. They should not be read as original work. Upstream acknowledgements and licensing are preserved in [`LICENSE`](LICENSE).

AI tools were used as a coding and review aid. I remain responsible for the design choices, source code, tests, and explanations in this repository.
