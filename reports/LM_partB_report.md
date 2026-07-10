# LM PartB Report Notes

Pretrained GPT2 with manual LoRA on Penn Treebank.

Best mandatory model: `rank8_alpha16_qkv`

Best overall model including optional extras: `extra_rank16_alpha32_qkv`

| Experiment | Trainable Params | Dev PPL | Test PPL |
| --- | ---: | ---: | ---: |
| rank1_alpha2_qkv | 55K | 36.155 | 32.030 |
| rank2_alpha4_qkv | 111K | 33.733 | 29.974 |
| rank4_alpha8_qkv | 221K | 31.720 | 28.338 |
| rank8_alpha16_qkv | 442K | 30.114 | 27.059 |
| extra_rank16_alpha32_qkv | 885K | 28.637 | 25.811 |
| extra_v_only_r4_alpha8 | 74K | 35.387 | 31.412 |
| extra_q_only_r4_alpha8 | 74K | 49.119 | 43.250 |
| extra_k_only_r4_alpha8 | 74K | 49.377 | 43.608 |

Implementation details to mention:

- Loaded `openai-community/gpt2`.
- All pretrained parameters frozen.
- Manual LoRA, no PEFT.
- HuggingFace GPT2 fused `c_attn` handled by adding LoRA deltas to Q/K/V slices.
- LoRA A random, B exactly zero, so step-zero base and LoRA logits matched.
- Only LoRA parameters were trainable.

Interpretation:

LoRA strongly beat scratch GPT2 while training far fewer parameters. QKV LoRA was much better than Q-only or K-only; V-only helped but was not enough.
