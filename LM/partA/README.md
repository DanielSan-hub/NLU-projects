# LM/partA - Mini-Project 1A

Scratch GPT2-style language modeling on PennTreeBank.

Metric: perplexity. Target: dev/test PPL below 250 after core training.

## Run

From the repository root:

```bash
python LM/partA/main.py --mode smoke --device cuda
python LM/partA/main.py --mode core --device cuda --amp --pin-memory --num-workers 2
python LM/partA/main.py --mode full --device cuda --amp --pin-memory --num-workers 2
```

On a CPU-only debugging machine:

```bash
python LM/partA/main.py --mode smoke --device cpu --allow-cpu
```

`--resume` resumes the latest run directory and each experiment reads its own `last.pt`.

## Architecture

The model in `model.py` is implemented from scratch with the Lab 04 GPT2 pattern:

- token embeddings
- learned positional embeddings
- dropout after token + positional embedding sum
- masked multi-head self-attention with a causal mask
- dropout after attention softmax
- dropout after attention output projection
- feed-forward network with dropout after the final linear layer
- pre-norm transformer blocks
- final normalization
- bias-free LM head

When `weight_tying=True`, the output classifier shares the exact same tensor object as the token embedding:

```python
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.token_embed.weight
assert self.lm_head.weight is self.token_embed.weight
```

Weight tying reduces parameters and acts as regularization by sharing the input token embedding matrix with the output classifier matrix. Each run logs parameter count before and after tying.

## Dataset

PennTreeBank train/dev/test files are tokenized with `GPT2TokenizerFast.from_pretrained("gpt2")`.

- `tokenizer.pad_token = tokenizer.eos_token`
- chunks are built with length `block_size + 1`
- `input_ids = tokenized[:, :-1]`
- `labels = tokenized[:, 1:]`
- pad labels are replaced with `-100`
- `CrossEntropyLoss` ignores padding
- non-pad tokens are counted for loss averaging and throughput

## Core Experiments

`--mode core` writes `results/results_partA.csv` and runs:

- LR sweep with fixed architecture: `1e-3`, `5e-4`, `3e-4`
- one-at-a-time architecture ablations:
  - `d_model`
  - `n_heads`
  - `num_layers`
  - `ff_dim`
- dropout ablation
- dropout + weight tying regularization experiment

Default baseline:

```text
d_model=128, n_heads=4, num_layers=2, ff_dim=512, dropout=0.1
```

The mandatory ablation table changes only one architectural hyperparameter at a time.

## Full Extras

`--mode full` first runs all core experiments, then writes optional runs to `results/results_partA_extra.csv`:

- throughput-aware reporting
- RMSNorm vs LayerNorm
- ReLU^2 vs GELU
- nanochat-inspired depth dial
- optional `x0` residual injection

## Overfitting Tracking

Every experiment logs `epoch_log.csv` with:

- train loss
- dev loss
- dev PPL
- best epoch
- train-dev loss gap

The results CSV includes `best_epoch`, `final_train_loss`, `best_dev_loss`, `train_dev_gap`, and `overfitting_note`.

If a larger/deeper model improves train loss but worsens dev PPL, the row is marked:

```text
decision=reject
notes=likely overfitting or optimization instability on small PTB
```

## Best And Worse Results

The script prints the final best configuration at the end of each run and stores it in the top-level run `summary.txt`.

For the report, use the best `dev_ppl` row in `results/results_partA.csv`. A typical worse run to discuss is a larger/deeper ablation that lowers train loss but increases dev PPL, suggesting overfitting or less stable optimization on small PTB.

## AI Tool Usage Declaration

This project harness and implementation were completed with AI assistance. The code was reviewed and smoke-tested by running the command-line entry points, and the final responsibility for interpreting results and reporting conclusions remain mine.
