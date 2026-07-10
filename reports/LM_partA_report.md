# LM PartA Report Notes

Scratch GPT2-style language modeling on Penn Treebank.

Best mandatory model: `ablation_d_model_192`

| Experiment | Dev PPL | Test PPL | Params | Notes |
| --- | ---: | ---: | ---: | --- |
| baseline_lr_5e-4 | 67.722 | 58.489 | 6.85M | reference |
| lr_sweep_1e-3 | 63.508 | 54.583 | 6.85M | best LR sweep |
| lr_sweep_3e-4 | 77.930 | 67.259 | 6.85M | slower optimization |
| ablation_d_model_192 | 62.624 | 53.651 | 10.37M | best scratch run |
| ablation_n_heads_8 | 68.158 | 58.521 | 6.85M | weaker than baseline |
| ablation_num_layers_3 | 66.790 | 57.487 | 7.04M | small gain vs baseline |
| dropout_0_2_weight_tying | 74.165 | 63.700 | 6.85M | too much dropout |

Implementation details to mention:

- GPT2 tokenizer with Penn Treebank train/dev/test.
- Causal masked multi-head self-attention.
- Required dropout placements were implemented.
- Padding labels ignored with `ignore_index=-100`.
- Weight tying: `lm_head.weight is token_embed.weight`.
- Best model beats the PPL target by a large margin.

Interpretation:

Increasing `d_model` improved perplexity most, but also increased the train-dev gap. Optional RMSNorm/ReLU2/depth experiments did not beat the best mandatory scratch result.
