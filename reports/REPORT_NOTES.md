# Report Notes

These notes are a larger source of material for the final one-page LM and NLU reports. The official report drafts are `LM_report.tex` and `NLU_report.tex`.

## Language Modeling

### Part A: Scratch GPT2 on Penn Treebank

Best mandatory scratch model:

| Experiment | Dev PPL | Test PPL | Trainable Params | Peak MB | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| ablation_d_model_192 | 62.624 | 53.651 | 10.37M | 2600 | Best scratch run |
| lr_sweep_1e-3 | 63.508 | 54.583 | 6.85M | 2537 | Best LR sweep |
| baseline_lr_5e-4 | 67.722 | 58.489 | 6.85M | 2537 | Reference |
| dropout_0_2_weight_tying | 74.165 | 63.700 | 6.85M | 2537 | Stronger dropout hurt |

Useful interpretation:

- All mandatory scratch runs beat the required PPL threshold of 250.
- Increasing `d_model` from 128 to 192 produced the best dev/test PPL.
- The train-dev gap for `d_model=192` was 0.439, higher than the baseline gap 0.252, so the larger model should be described as useful but closer to overfitting.
- Weight tying was active in the default architecture and reduced the output/classifier parameter burden by sharing token input and output embeddings.

Best optional scratch extra:

| Experiment | Dev PPL | Test PPL |
| --- | ---: | ---: |
| extra_depth_dial_3 | 65.752 | 56.334 |
| extra_relu2 | 65.946 | 56.881 |
| extra_rmsnorm | 68.123 | 58.961 |

The optional scratch runs did not beat the best mandatory `d_model=192` model.

### Part B: Pretrained GPT2 with Manual LoRA

Best mandatory LoRA model:

| Experiment | Trainable Params | Dev PPL | Test PPL |
| --- | ---: | ---: | ---: |
| rank8_alpha16_qkv | 442K | 30.114 | 27.059 |
| rank4_alpha8_qkv | 221K | 31.720 | 28.338 |
| rank2_alpha4_qkv | 111K | 33.733 | 29.974 |
| rank1_alpha2_qkv | 55K | 36.155 | 32.030 |

Best optional LoRA model:

| Experiment | Trainable Params | Dev PPL | Test PPL |
| --- | ---: | ---: | ---: |
| extra_rank16_alpha32_qkv | 885K | 28.637 | 25.811 |
| extra_qkv_dropout_r4_alpha8 | 221K | 31.862 | 28.387 |
| extra_v_only_r4_alpha8 | 74K | 35.387 | 31.412 |
| extra_q_only_r4_alpha8 | 74K | 49.119 | 43.250 |
| extra_k_only_r4_alpha8 | 74K | 49.377 | 43.608 |

Useful interpretation:

- LoRA strongly outperformed scratch GPT2 while training under 1M parameters in the best run.
- Rank scaling was monotonic in this range.
- QKV adapters are much better than Q-only or K-only; V-only is useful but not sufficient.
- Step-zero base-vs-LoRA logits matched because LoRA B was initialized to zero.

## NLU

### Part A: Scratch GPT2 on ATIS

Best mandatory scratch model by combined dev score:

| Experiment | Dev Intent | Dev Slot F1 | Test Intent | Test Slot F1 | Frame Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| ablation_n_heads_8 | 0.968 | 0.942 | 0.960 | 0.919 | 0.772 |
| ablation_d_model_192 | 0.962 | 0.946 | 0.953 | 0.927 | 0.774 |
| baseline_lr_5e-4 | 0.966 | 0.939 | 0.953 | 0.910 | 0.751 |
| lr_sweep_3e-4 | 0.964 | 0.940 | 0.960 | 0.920 | 0.772 |

Useful interpretation:

- The causal GPT2-style model works well when the CLS/global token is appended at the end.
- Slot F1 is high despite the decoder-only backbone.
- Larger representation improves slots, while more heads gives the best combined dev score.

### Part B: Pretrained BERT and GPT2 on ATIS

| Model | Dev Intent | Dev Slot F1 | Test Intent | Test Slot F1 | Frame Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERT core | 0.976 | 0.976 | 0.976 | 0.951 | 0.868 |
| GPT2 last | 0.964 | 0.929 | 0.957 | 0.884 | 0.672 |
| GPT2 mean | 0.972 | 0.937 | 0.968 | 0.915 | 0.777 |
| BERT ontology | 0.980 | 0.976 | 0.978 | 0.956 | 0.878 |

Useful interpretation:

- BERT is the best model for token-level NLU, consistent with bidirectional context.
- GPT2 is competitive for intent but weaker for slots with last-token pooling.
- Mean pooling improves GPT2 over last-token pooling, especially test slot F1.
- Ontology reporting slightly improves BERT frame metrics and gives a useful semantic-frame analysis.

## Reproducibility Details

- Seed: 1.
- Hardware used for final runs: Azure `Standard_NC8as_T4_v3`, Tesla T4 16GB.
- Scripts: `scripts/smoke_all.sh`, `scripts/run_core.sh`, `scripts/run_extras.sh`.
- Final validation passed with `scripts/validate_submission.py`.
- Final master CSV: `results/master_results.csv`.

## AI Tool Declaration

AI assistance was used to scaffold and debug the code, guide the university-machine setup, inspect logs, and draft report text. The final submitted code, experiment choices, and report interpretation should be checked and accepted by the student before submission.
