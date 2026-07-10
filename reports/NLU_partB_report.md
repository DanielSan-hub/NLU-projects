# NLU PartB Report Notes

Pretrained BERT and GPT2 for ATIS multitask NLU.

Best core model: BERT.

Best overall including optional extras: `bert_ontology_report`.

| Model | Dev Intent | Dev Slot F1 | Test Intent | Test Slot F1 | Frame Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERT core | 0.976 | 0.976 | 0.976 | 0.951 | 0.868 |
| GPT2 last | 0.964 | 0.929 | 0.957 | 0.884 | 0.672 |
| GPT2 mean | 0.972 | 0.937 | 0.968 | 0.915 | 0.777 |
| BERT ontology | 0.980 | 0.976 | 0.978 | 0.956 | 0.878 |

Implementation details to mention:

- BERT: `bert-base-uncased`, CLS pooling.
- GPT2: `openai-community/gpt2`, last valid token pooling in core.
- Subtoken alignment: gold slot label only on first subtoken.
- Non-first subtokens, special tokens, and padding set to `-100`.
- Slot loss uses `ignore_index=-100`; evaluation reconstructs only valid word-level labels.

Interpretation:

BERT is strongest because slot filling benefits from bidirectional context. GPT2 is competitive for intent but weaker for slots. Mean pooling improves GPT2 over last-token pooling.
