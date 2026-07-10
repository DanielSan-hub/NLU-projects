# NLU Part B — experiment design

## Question

How do pretrained encoder and decoder representations compare for joint ATIS intent classification and slot filling under consistent metrics?

## Evidence to report

- intent accuracy and slot F1 for BERT and GPT-2;
- tokenizer and first-subtoken alignment checks;
- label masking and pooling choices;
- matched run metadata plus qualitative errors for each architecture.

## Status

Both pretrained-model paths and the run harness are present. No quantitative result is claimed in this public snapshot; populate the final report from `NLU/partB/results/results_partB.csv` after the GPU core run.
