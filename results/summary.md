# Master Results Summary

Generated: 2026-06-25 14:59:52

## Files

- Rows collected: 17
- Master CSV: `results/master_results.csv`

## Missing Result Files

- None

## Rows By Part

- `LM/partA`: 3
- `LM/partB`: 4
- `NLU/partA`: 4
- `NLU/partB`: 6

## Best Result Per Part

| Part | Experiment | Mode | Metric | Value | Source |
| --- | --- | --- | --- | ---: | --- |
| LM/partA | smoke_shapes_loss_ppl_checkpoint_csv | smoke | dev_ppl | 47720.5 | LM/partA/results/results_partA.csv |
| LM/partB | smoke_lora_qkv_r2_alpha4 | smoke | dev_ppl | 207.371 | LM/partB/results/results_partB.csv |
| NLU/partA | smoke_shapes_losses_metrics_checkpoint_csv | smoke | mean(dev intent acc, dev slot f1) | 0.332702 | NLU/partA/results/results_partA.csv |
| NLU/partB | bert | smoke | mean(dev intent acc, dev slot f1) | 0 | NLU/partB/results/results_partB.csv |
