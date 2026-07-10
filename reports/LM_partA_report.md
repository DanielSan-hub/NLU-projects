# LM Part A — experiment design

## Question

Can a compact GPT-2-style decoder trained from scratch learn a useful Penn Treebank language model, and how do the required architectural components affect stable training?

## Evidence to report

- validation and test perplexity with seed, epoch, and checkpoint;
- parameter count and model configuration;
- train/validation loss curves;
- explicit confirmation that causal and padding masks behave as intended.

## Status

The implementation and run harness are present. No quantitative result is claimed in this public snapshot; populate the final report from `LM/partA/results/results_partA.csv` after the GPU core run.
