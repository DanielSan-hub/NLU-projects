# NLU/partB - Mini-Project 2B

Pretrained BERT and GPT2 for ATIS multitask NLU:

- intent classification
- slot filling

Core models:

- `bert-base-uncased`
- `openai-community/gpt2`

Large variants such as `bert-large` or `gpt2-medium` are not used by default.

## Run

From the repository root:

```bash
python NLU/partB/main.py --mode smoke --device cuda
python NLU/partB/main.py --mode core --device cuda --amp --pin-memory --num-workers 2
python NLU/partB/main.py --mode full --device cuda --amp --pin-memory --num-workers 2
```

On a CPU-only debugging machine:

```bash
python NLU/partB/main.py --mode smoke --device cpu --allow-cpu
```

`--resume` resumes the latest run and each model directory loads its own `last.pt`.

## BERT vs GPT2 For NLU

BERT is bidirectional, so each token representation can use both left and right context. Its intent representation uses `[CLS]`.

GPT2 is decoder-only and causal, so token representations cannot look ahead. For intent classification the implementation uses the last valid non-padding token representation, not an initial CLS token. An initial CLS would not be able to attend to future words.

## Subtokenization And `-100` Masking

ATIS slot labels are word-level, while BERT and GPT2 tokenizers produce subword tokens. The function `tokenize_and_align_labels` handles this by:

- splitting utterances into words
- tokenizing with `is_split_into_words=True`
- assigning the gold slot label only to the first subtoken of each word
- assigning `-100` to non-first subtokens
- assigning `-100` to special tokens
- assigning `-100` to padding tokens

The slot loss uses:

```python
CrossEntropyLoss(ignore_index=-100)
```

During evaluation, slot predictions are reconstructed only at positions where labels are not `-100`, then converted back to slot strings for CoNLL/BIO F1.

## Model Heads

Both models use:

- pretrained `AutoModel`
- `slot_classifier` over every token representation
- `intent_classifier` over the pooled sentence representation

BERT pooling:

```text
[CLS] representation
```

GPT2 pooling:

```text
last valid non-padding token representation
```

## Loss

Default multitask objective:

```text
total_loss = slot_loss + intent_loss
```

The code also supports:

```text
total_loss = lambda_slot * slot_loss + lambda_intent * intent_loss
```

Both lambdas default to `1.0`.

## Core Experiments

`--mode core` writes `results/results_partB.csv` and trains:

- BERT-base multitask
- GPT2 multitask

The script prints a final comparison table with dev/test intent accuracy and slot CoNLL F1.

## Full Extras

`--mode full` runs core plus optional analyses in `results/results_partB_extra.csv`:

- GPT2 mean-pooling ablation
- ontology-style illegal slot reporting for BERT

Optional metrics do not replace required intent accuracy or slot F1.

## Outputs

Each model run writes:

- `config.json`
- `labels.json`
- `alignment_and_shape_checks.json`
- `best.pt`
- `last.pt`
- `epoch_log.csv`
- `summary.txt`

Result columns:

```text
part,model_name,pretrained_model,mode,lr,batch_size,epochs,lambda_slot,lambda_intent,total_params,trainable_params,train_time_seconds,peak_memory_mb,intent_acc_dev,slot_f1_dev,intent_acc_test,slot_f1_test,semantic_frame_acc_test,checkpoint_path,notes
```

## AI Tool Usage Declaration

This project harness and implementation were completed with AI assistance. The code was reviewed and smoke-tested through command-line entry points, and the final responsibility for interpreting results and reporting conclusions remains with the student.
