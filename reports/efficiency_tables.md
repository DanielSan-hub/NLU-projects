# GPU, Memory And Time Tables

Generated from final VM result CSV files. Smoke runs are excluded because they are functional checks, not performance experiments.

Important interpretation note: `peak_memory_mb` is `torch.cuda.max_memory_allocated`, i.e. peak tensor memory allocated by PyTorch during the experiment. It is not the full `nvidia-smi` process footprint and does not include all CUDA caching/reserved memory. It is still useful for comparing experiments within this project.

## Hardware Context

| Item | Value |
| --- | --- |
| GPU | NVIDIA Tesla T4 |
| GPU memory | 16GB |
| VM size | Azure Standard_NC8as_T4_v3 |
| Driver / CUDA shown by nvidia-smi | 555.42.06 / 12.5 |
| System RAM | about 54GiB |
| Execution | sequential, unattended via tmux/scripts |
| Core/extras AMP | enabled when launched through scripts unless overridden |

## Suite-Level Totals

| Scope | Experiments | Train time s | Train time min | Max peak CUDA MB |
| --- | --- | --- | --- | --- |
| LM/partA | 15 | 2160.9 | 36.02 | 2677.4 |
| LM/partB | 9 | 2961.0 | 49.35 | 2199.3 |
| NLU/partA | 9 | 170.1 | 2.83 | 95.5 |
| NLU/partB | 4 | 880.3 | 14.67 | 3569.9 |
| LM total | 24 | 5121.9 | 85.37 | 2677.4 |
| NLU total | 13 | 1050.3 | 17.51 | 3569.9 |
| Project total | 37 | 6172.3 | 102.87 | 3569.9 |

## Per-Experiment Efficiency

### LM/partA

| Experiment | Mode | Time s | Peak CUDA MB | Tokens/s | Trainable params | Metric |
| --- | --- | --- | --- | --- | --- | --- |
| ablation_d_model_192 | core | 177.94 | 2600.1 | 61505 | 10.37M | dev/test PPL 62.62/53.65 |
| ablation_ff_dim_768 | core | 140.50 | 2547.6 | 77894 | 6.98M | dev/test PPL 67.83/58.18 |
| ablation_n_heads_8 | core | 143.85 | 2564.5 | 76077 | 6.85M | dev/test PPL 68.16/58.52 |
| ablation_num_layers_3 | core | 147.28 | 2572.6 | 74306 | 7.04M | dev/test PPL 66.79/57.49 |
| baseline_lr_5e-4 | core | 132.04 | 2536.8 | 82886 | 6.85M | dev/test PPL 67.72/58.49 |
| dropout_0_0 | core | 138.18 | 2530.3 | 79202 | 6.85M | dev/test PPL 67.55/58.39 |
| dropout_0_2_weight_tying | core | 139.45 | 2536.8 | 78481 | 6.85M | dev/test PPL 74.17/63.70 |
| lr_sweep_1e-3 | core | 136.67 | 2536.8 | 80075 | 6.85M | dev/test PPL 63.51/54.58 |
| lr_sweep_3e-4 | core | 138.76 | 2536.8 | 78871 | 6.85M | dev/test PPL 77.93/67.26 |
| extra_depth_dial_1 | full | 121.52 | 2467.4 | 90059 | 4.95M | dev/test PPL 73.78/62.85 |
| extra_depth_dial_2 | full | 139.35 | 2536.8 | 78533 | 6.85M | dev/test PPL 67.72/58.49 |
| extra_depth_dial_3 | full | 191.63 | 2677.4 | 57111 | 11.01M | dev/test PPL 65.75/56.33 |
| extra_relu2 | full | 138.67 | 2551.8 | 78922 | 6.85M | dev/test PPL 65.95/56.88 |
| extra_rmsnorm | full | 135.85 | 2548.7 | 80557 | 6.85M | dev/test PPL 68.12/58.96 |
| extra_x0_residual | full | 139.24 | 2536.8 | 78600 | 6.85M | dev/test PPL 67.89/58.60 |

### LM/partB

| Experiment | Mode | Time s | Peak CUDA MB | Tokens/s | Trainable params | Metric |
| --- | --- | --- | --- | --- | --- | --- |
| rank1_alpha2_qkv | core | 329.36 | 2170.6 | 10352 | 55k | dev/test PPL 36.15/32.03 |
| rank2_alpha4_qkv | core | 337.26 | 2190.2 | 10109 | 111k | dev/test PPL 33.73/29.97 |
| rank4_alpha8_qkv | core | 334.85 | 2191.8 | 10182 | 221k | dev/test PPL 31.72/28.34 |
| rank8_alpha16_qkv | core | 334.64 | 2195.0 | 10188 | 442k | dev/test PPL 30.11/27.06 |
| extra_k_only_r4_alpha8 | full | 315.30 | 2150.8 | 10813 | 74k | dev/test PPL 49.38/43.61 |
| extra_q_only_r4_alpha8 | full | 315.18 | 2150.8 | 10817 | 74k | dev/test PPL 49.12/43.25 |
| extra_qkv_dropout_r4_alpha8 | full | 341.35 | 2199.3 | 9988 | 221k | dev/test PPL 31.86/28.39 |
| extra_rank16_alpha32_qkv | full | 335.18 | 2182.6 | 10172 | 885k | dev/test PPL 28.64/25.81 |
| extra_v_only_r4_alpha8 | full | 317.87 | 2150.8 | 10726 | 74k | dev/test PPL 35.39/31.41 |

### NLU/partA

| Experiment | Mode | Time s | Peak CUDA MB | Tokens/s | Trainable params | Metric |
| --- | --- | --- | --- | --- | --- | --- |
| ablation_d_model_192 | core | 18.59 | 87.3 |  | 897k | dev slot 0.946, test slot 0.927, frame 0.774 |
| ablation_ff_dim_768 | core | 18.39 | 85.3 |  | 665k | dev slot 0.943, test slot 0.919, frame 0.768 |
| ablation_n_heads_8 | core | 18.29 | 80.6 |  | 533k | dev slot 0.942, test slot 0.919, frame 0.772 |
| ablation_num_layers_3 | core | 22.38 | 95.5 |  | 731k | dev slot 0.944, test slot 0.916, frame 0.766 |
| baseline_lr_5e-4 | core | 18.34 | 73.7 |  | 533k | dev slot 0.939, test slot 0.910, frame 0.751 |
| dropout_before_heads_0_2 | core | 18.32 | 73.7 |  | 533k | dev slot 0.935, test slot 0.908, frame 0.756 |
| lr_sweep_1e-3 | core | 18.37 | 73.7 |  | 533k | dev slot 0.939, test slot 0.919, frame 0.770 |
| lr_sweep_3e-4 | core | 18.64 | 73.7 |  | 533k | dev slot 0.940, test slot 0.920, frame 0.772 |
| extra_semantic_frame_metrics | full | 18.75 | 72.6 |  | 533k | dev slot 0.947, test slot 0.917, frame 0.756 |

### NLU/partB

| Experiment | Mode | Time s | Peak CUDA MB | Tokens/s | Trainable params | Metric |
| --- | --- | --- | --- | --- | --- | --- |
| bert | core | 174.08 | 2997.6 |  | 109.60M | dev slot 0.976, test slot 0.951, frame 0.868 |
| gpt2 | core | 220.02 | 3552.4 |  | 124.56M | dev slot 0.929, test slot 0.884, frame 0.672 |
| bert_ontology_report | full | 218.47 | 3026.9 |  | 109.60M | dev slot 0.976, test slot 0.956, frame 0.878 |
| gpt2_mean_pool | full | 267.69 | 3569.9 |  | 124.56M | dev slot 0.937, test slot 0.915, frame 0.777 |
