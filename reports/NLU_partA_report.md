# NLU PartA Report Notes

Scratch GPT2-style multitask NLU on ATIS.

Best mandatory model by combined dev score: `ablation_n_heads_8`

| Experiment | Dev Intent | Dev Slot F1 | Test Intent | Test Slot F1 | Frame Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline_lr_5e-4 | 0.966 | 0.939 | 0.953 | 0.910 | 0.751 |
| lr_sweep_1e-3 | 0.966 | 0.939 | 0.948 | 0.919 | 0.770 |
| lr_sweep_3e-4 | 0.964 | 0.940 | 0.960 | 0.920 | 0.772 |
| ablation_d_model_192 | 0.962 | 0.946 | 0.953 | 0.927 | 0.774 |
| ablation_n_heads_8 | 0.968 | 0.942 | 0.960 | 0.919 | 0.772 |
| ablation_num_layers_3 | 0.958 | 0.944 | 0.951 | 0.916 | 0.766 |
| dropout_before_heads_0_2 | 0.964 | 0.935 | 0.959 | 0.908 | 0.756 |

Implementation details to mention:

- ATIS split with intent stratification.
- Word, slot, and intent vocabularies.
- Decoder-only GPT2 backbone with causal mask.
- CLS/global token appended at the end, because beginning CLS cannot attend to future tokens.
- Slot loss ignores PAD and CLS.
- Joint loss: slot cross-entropy plus intent cross-entropy.

Interpretation:

The scratch causal model works well when the global token is final. More heads gave the best combined dev score; larger `d_model` improved slot F1 most.
