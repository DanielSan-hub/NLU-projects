# NLU/partA - Mini-Project 2A

Scratch GPT2-style decoder-only model for ATIS multitask NLU:

- intent classification
- slot filling

ATIS is treated as semantic frame extraction: the intent is the frame type, while slots fill frame arguments such as departure city, arrival city, time, airline, and fare constraints.

## Run

From the repository root:

```bash
python NLU/partA/main.py --mode smoke --device cuda
python NLU/partA/main.py --mode core --device cuda --amp --pin-memory --num-workers 2
python NLU/partA/main.py --mode full --device cuda --amp --pin-memory --num-workers 2
```

On a CPU-only debugging machine:

```bash
python NLU/partA/main.py --mode smoke --device cpu --allow-cpu
```

`--resume` resumes the latest run and each experiment loads its own `last.pt`.

## Data Setup

The code follows the Lab 05 structure:

- load ATIS JSON with `utterance`, `slots`, and `intent`
- stratified train/dev split by intent where possible
- `word2id`, `slot2id`, and `intent2id` through a small `Lang` class
- custom `Dataset`
- dynamic `collate_fn` with padding
- append a `<cls>` token at the end of each utterance
- assign the slot PAD label to `<cls>`
- ignore PAD and `<cls>` for slot loss and slot F1

Because GPT2 is causal, the global token must be final. A beginning CLS token could not attend to future words, while a final CLS token can attend to the full utterance.

## Model

`model.py` implements a scratch GPT2-style backbone:

- token embeddings
- learned positional embeddings
- causal masked multi-head self-attention
- feed-forward network
- transformer blocks
- final LayerNorm
- `slot_out = Linear(d_model, n_slots)`
- `intent_out = Linear(d_model, n_intents)`
- dropout before both output heads

The intent representation is the final valid token, which is the appended `<cls>` token.

## Loss

The objective is joint multitask learning:

```text
total_loss = slot_loss + intent_loss
```

`slot_loss` uses `ignore_index=slot_pad_id`, so both padding and the final CLS position are excluded from slot supervision.

## Metrics

Required metrics:

- intent accuracy
- slot CoNLL/BIO entity F1

PAD and CLS positions are excluded before computing slot F1.

The code also reports semantic frame accuracy on test: a sample is correct only if the intent is correct and all valid slot labels are correct.

## Core Experiments

`--mode core` writes `results/results_partA.csv` and runs:

- LR sweep
- one-at-a-time ablations:
  - `d_model`
  - `n_heads`
  - `num_layers`
  - `ff_dim`
- dropout before output heads

Suggested baseline:

```text
d_model=128, n_heads=4, num_layers=2, ff_dim=512, dropout=0.1
```

## Full Extras

`--mode full` runs core plus optional semantic-frame reporting in `results/results_partA_extra.csv`.

Optional metrics include:

- semantic frame accuracy
- ontology-derived illegal slot rate
- frame validity

These optional metrics do not replace intent accuracy or slot CoNLL F1.

## Best Result

The script prints the final best configuration at the end and saves it in the top-level run `summary.txt`. Use the best row in `results/results_partA.csv` for the report.

## AI Tool Usage Declaration

This project harness and implementation were completed with AI assistance. The code was reviewed and smoke-tested through the command-line entry points, and the final responsibility for interpreting results and reporting conclusions remains with the student.
