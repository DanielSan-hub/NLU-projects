# LM Part B — experiment design

## Question

How effectively can manual low-rank adaptation specialize pretrained GPT-2 for Penn Treebank while training only a small fraction of its parameters?

## Evidence to report

- validation and test perplexity;
- LoRA rank, scaling, dropout, and targeted projection;
- total versus trainable parameter counts;
- a comparison against the required baseline under the same evaluation protocol.

## Status

The manual LoRA implementation and run harness are present. No quantitative result is claimed in this public snapshot; populate the final report from `LM/partB/results/results_partB.csv` after the GPU core run.
