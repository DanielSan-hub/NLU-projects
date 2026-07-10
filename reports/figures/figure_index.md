# Report Figure Index

Figures generated from the final VM CSV results.

Available formats:

- `*.svg`: editable vector source.
- `png/*.png`: high-resolution raster files, easiest to include with standard LaTeX `\includegraphics`.
- `pdf/*.pdf`: vector-ish browser export, useful for LaTeX if the template accepts PDF figures.

Example LaTeX usage:

```tex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/png/lm_ppl_overview.png}
  \caption{Main LM comparison. Lower perplexity is better.}
\end{figure}
```

Recommended minimal selection:

- LM report: `lm_ppl_overview.svg` and `lm_partb_lora_rank_sweep.svg`.
- NLU report: `nlu_partb_model_comparison.svg` and either `nlu_overall_test_metrics.svg` or `nlu_parta_core_metrics.svg`.
- Oral presentation or appendix: `project_best_summary.svg`, `lm_efficiency_tradeoff.svg`, `nlu_slot_frame_scatter.svg`.

## Captions

### `lm_ppl_overview.svg`

- Use: Recommended for LM report
- Caption: Main LM comparison: scratch GPT2 is below the PPL target, while manual LoRA on pretrained GPT2 roughly halves perplexity again.

### `lm_parta_ppl_gap.svg`

- Use: Supplemental LM figure
- Caption: Shows the mandatory one-at-a-time scratch GPT2 ablations and the overfitting-aware train-dev gap.

### `lm_partb_lora_rank_sweep.svg`

- Use: Recommended for LM report
- Caption: Rank/alpha sweep for manual LoRA. It is the cleanest visual evidence that larger low-rank adapters improve PTB adaptation.

### `lm_lora_target_ablation.svg`

- Use: Supplemental LM figure
- Caption: Optional LoRA target analysis: adapting all QKV sections is clearly stronger than adapting only Q or K.

### `lm_efficiency_tradeoff.svg`

- Use: Supplemental LM figure
- Caption: Shows the practical trade-off: scratch models are faster, but LoRA reaches much lower PPL.

### `nlu_parta_core_metrics.svg`

- Use: Recommended or supplemental NLU figure
- Caption: Mandatory scratch GPT2 NLU ablations. It highlights that d_model=192 is strongest for slot/frame, while heads=8 is strongest by combined dev score.

### `nlu_partb_model_comparison.svg`

- Use: Recommended for NLU report
- Caption: Core BERT/GPT2 comparison plus optional analyses. BERT wins clearly; GPT2 mean pooling recovers much of the gap.

### `nlu_overall_test_metrics.svg`

- Use: Recommended for NLU report
- Caption: Compact cross-part NLU comparison: BERT is best, while scratch GPT2 remains competitive and GPT2 mean pooling improves over last-token pooling.

### `nlu_slot_frame_scatter.svg`

- Use: Supplemental NLU figure
- Caption: Shows why semantic frame accuracy is stricter than slot F1 and separates BERT from GPT2 last-token pooling.

### `project_best_summary.svg`

- Use: Presentation / oral exam figure
- Caption: One-slide overview of the best result from each mini-project.

## Loss Curve Figures

These figures are generated from per-experiment `epoch_log.csv` files saved during the VM runs.

### `loss_lm_parta_selected_train_dev.svg`

- Use: LM training diagnostics
- Caption: Train/dev loss curves for selected scratch GPT2 runs. The wider model reaches the lowest dev loss, while heavier dropout learns more slowly.

### `loss_lm_parta_all_train.svg`

- Use: Supplemental LM diagnostics
- Caption: Train loss for all mandatory scratch GPT2 experiments.

### `loss_lm_partb_lora_rank_train_dev.svg`

- Use: Recommended LM training diagnostic
- Caption: Train/dev loss curves for the LoRA rank sweep.

### `loss_lm_partb_lora_extra_train_dev.svg`

- Use: Supplemental LM diagnostics
- Caption: Optional LoRA target/rank loss curves.

### `loss_nlu_parta_selected_train_dev.svg`

- Use: Recommended NLU training diagnostic
- Caption: Train/dev multitask loss for selected scratch GPT2 ATIS experiments.

### `loss_nlu_parta_slot_intent_components.svg`

- Use: Supplemental NLU diagnostics
- Caption: Slot and intent loss components show how the multitask objective evolves.

### `loss_nlu_partb_core_train_dev.svg`

- Use: Recommended NLU training diagnostic
- Caption: Train/dev loss curves for BERT and GPT2 core multitask fine-tuning.

### `loss_nlu_partb_extra_train_dev.svg`

- Use: Supplemental NLU diagnostics
- Caption: Loss curves for optional pretrained NLU analyses.
