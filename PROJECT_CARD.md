# Project card

## Research question

How do architecture choice and adaptation strategy change the implementation and evaluation of transformer systems across causal language modelling and joint intent/slot prediction?

## Systems

The repository compares four deliberately connected settings:

1. a decoder-only transformer trained from scratch;
2. a pretrained decoder adapted through a manual LoRA implementation;
3. a scratch decoder extended with two task heads for intent and slots;
4. pretrained encoder- and decoder-based multitask models with explicit subtoken alignment.

This progression makes otherwise hidden implementation details inspectable: causal masking, residual blocks, fused QKV adaptation, sequence pooling, label alignment, and loss masking.

## Evaluation discipline

- Language modelling: token-level cross-entropy and perplexity.
- Multitask NLU: intent accuracy and slot F1, keeping padding and non-first subtokens out of the slot loss with label `-100`.
- Reproducibility: explicit seeds, saved run configurations, resumable checkpoints, structured CSV outputs, smoke/core/optional run tiers, and a submission validator.
- Reporting: quantitative claims are tied to checked-in report notes, figures, CSV summaries, and run metadata. Large checkpoint files are excluded from Git.

## Result summary

- Scratch LM best verified run: `ablation_d_model_192`, dev PPL `62.624`, test PPL `53.651`.
- GPT-2 LoRA best mandatory run: `rank8_alpha16_qkv`, dev PPL `30.114`, test PPL `27.059`; optional `extra_rank16_alpha32_qkv` reached test PPL `25.811`.
- Scratch multitask NLU best dev-selected run: test intent accuracy `0.960`, test slot F1 `0.919`, frame accuracy `0.772`.
- Pretrained multitask NLU: BERT core reached test intent accuracy `0.976`, test slot F1 `0.951`, frame accuracy `0.868`; the ontology run reached frame accuracy `0.878`.

## Engineering highlights

- LoRA is inserted directly into GPT-2's fused attention projection instead of delegated to a PEFT wrapper.
- Scratch and pretrained models share comparable task boundaries while preserving architecture-specific tokenization and pooling choices.
- Shell runners fail fast and keep timestamped logs; aggregation code produces both machine-readable and reviewer-readable summaries.
- A dependency-free CI check validates repository structure, Python syntax, and required command-line interfaces.

## Limitations and next experiments

- Confidence intervals are not reported; the final runs use a fixed seed and should be repeated across seeds for stronger claims.
- The current scope is course-sized; stronger work would add broader error slices for rare intents and slots, calibration tests, and robustness checks under paraphrase or distribution shift.
- A natural safety-oriented extension is to test whether parameter-efficient adaptation changes robustness under distribution shift or adversarial paraphrase.
