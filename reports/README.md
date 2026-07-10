# Experiment reports

These files record the planned comparisons, metrics, and evidence required for each experiment. They intentionally contain no invented numerical results.

After a GPU core run, `scripts/collect_results.py` creates the aggregate outputs in `results/`. A final report should only promote a number from those outputs when the run configuration, seed, and relevant artifact are available alongside it.

| Design note | Experiment |
| --- | --- |
| [`LM_partA_report.md`](LM_partA_report.md) | Scratch GPT-2 on Penn Treebank |
| [`LM_partB_report.md`](LM_partB_report.md) | Pretrained GPT-2 with manual LoRA |
| [`NLU_partA_report.md`](NLU_partA_report.md) | Scratch multitask GPT-2 on ATIS |
| [`NLU_partB_report.md`](NLU_partB_report.md) | Pretrained BERT/GPT-2 multitask comparison |
