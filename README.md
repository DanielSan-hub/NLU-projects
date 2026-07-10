# Transformer & NLU Systems

Four reproducible mini-projects exploring transformer language modelling and multitask natural-language understanding: from a decoder-only model built from scratch to parameter-efficient adaptation with manually implemented LoRA.

This repository is my implementation for the University of Trento Natural Language Understanding course. It is published as an engineering portfolio: the focus is model internals, evaluation discipline, and reproducible experiment tooling—not unverified benchmark claims.

## What I implemented

- A GPT-2-style causal language model from scratch, including masked self-attention, transformer blocks, training, and perplexity evaluation.
- Manual LoRA injection into GPT-2's fused QKV projection, with frozen-base fine-tuning and trainable-parameter accounting.
- A scratch decoder model for joint ATIS intent classification and slot filling.
- BERT- and GPT-2-based multitask systems with tokenizer/subword alignment and `-100` masking for non-label positions.
- A shared experiment workflow with smoke, core, and optional runs; deterministic seeds; checkpoint resume; CSV collection; and validation scripts.

## Project map

| Project | Model | Task / data | Main evaluation |
| --- | --- | --- | --- |
| [`LM/partA`](LM/partA) | Scratch GPT-2 | Penn Treebank language modelling | Perplexity |
| [`LM/partB`](LM/partB) | Pretrained GPT-2 + manual LoRA | Penn Treebank language modelling | Perplexity, trainable parameters |
| [`NLU/partA`](NLU/partA) | Scratch GPT-2 multitask model | ATIS intent + slot filling | Intent accuracy, slot F1 |
| [`NLU/partB`](NLU/partB) | Pretrained BERT and GPT-2 | ATIS intent + slot filling | Intent accuracy, slot F1 |

Each linked directory contains implementation notes, design choices, commands, and expected artifacts.

## Five-minute review

For the most relevant technical material:

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

Run the lightweight repository check (no model downloads or GPU required):

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

## Current public snapshot

The source tree and command-line interfaces are checked in CI without requiring heavyweight ML dependencies. Trained checkpoints, datasets, logs, and generated result files are intentionally excluded from Git because they are large or reproducible artifacts. Quantitative results are not claimed here until the corresponding GPU runs and run metadata can be published together. The files in [`reports/`](reports) therefore document experiment designs and reporting criteria, not fabricated outcomes.

## Authorship and course context

My project implementation lives in `LM/`, `NLU/`, `scripts/`, and `reports/`. The `labs/`, `solutions/`, `exam/`, environment files, and parts of the assignment structure originate from the University of Trento NLU course repository and remain here for context. They should not be read as original work. Upstream acknowledgements and licensing are preserved in [`LICENSE`](LICENSE).

AI tools were used as a coding and review aid. I remain responsible for the design choices, source code, tests, and explanations in this repository.
