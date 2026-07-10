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
- Reporting: no metric is presented without its associated run artifacts; the public snapshot currently makes no quantitative performance claim.

## Engineering highlights

- LoRA is inserted directly into GPT-2's fused attention projection instead of delegated to a PEFT wrapper.
- Scratch and pretrained models share comparable task boundaries while preserving architecture-specific tokenization and pooling choices.
- Shell runners fail fast and keep timestamped logs; aggregation code produces both machine-readable and reviewer-readable summaries.
- A dependency-free CI check validates repository structure, Python syntax, and required command-line interfaces.

## Limitations and next experiments

- Final GPU runs and confidence estimates still need to be published with hardware, seed, and configuration metadata.
- The current scope is course-sized; stronger work would add multiple seeds, ablations over LoRA rank/target modules, error slices for rare intents and slots, and calibration tests.
- A natural safety-oriented extension is to test whether parameter-efficient adaptation changes robustness under distribution shift or adversarial paraphrase.

