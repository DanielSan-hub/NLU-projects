# NLU Part A — experiment design

## Question

Can a scratch decoder share representations effectively between ATIS intent classification and slot filling?

## Evidence to report

- intent accuracy and slot F1 on the same split;
- both task-loss components and their weighting;
- handling of padding and ignored slot positions;
- error examples separating intent failures from boundary or slot-label failures.

## Status

The multitask architecture and run harness are present. No quantitative result is claimed in this public snapshot; populate the final report from `NLU/partA/results/results_partA.csv` after the GPU core run.
