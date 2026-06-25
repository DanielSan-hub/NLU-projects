# LM/partB - Mini-Project 1B

Pretrained GPT-2 fine-tuning on PennTreeBank with manually implemented LoRA.

Metric: perplexity. Target: PPL below 250 and better than the best scratch GPT-2 from `LM/partA`.

## Run

From the repository root:

```bash
python LM/partB/main.py --mode smoke --device cuda
python LM/partB/main.py --mode core --device cuda --amp --pin-memory --num-workers 2
python LM/partB/main.py --mode full --device cuda --amp --pin-memory --num-workers 2
```

On a CPU-only debugging machine:

```bash
python LM/partB/main.py --mode smoke --device cpu --allow-cpu
```

`--resume` resumes the latest run and each experiment loads its own `last_lora_adapters.pt`.

## Manual LoRA

The pretrained model is loaded from HuggingFace:

```text
openai-community/gpt2
```

All original GPT-2 parameters are frozen. LoRA is implemented manually without PEFT or external LoRA libraries. For a frozen projection `W`, the effective output is:

```text
output = W(x) + (alpha / rank) * B(A(x))
```

Initialization:

- `A` is initialized randomly with Kaiming initialization.
- `B` is initialized exactly to zeros.
- Therefore, at step 0 the LoRA delta is zero and GPT-2 behavior is unchanged.

The code tests this before training by running base GPT-2 and GPT-2+LoRA on the same batch and logging `max_abs_diff` between logits.

## Fused GPT-2 QKV Handling

HuggingFace GPT-2 uses a fused projection named `c_attn`, usually implemented as a `Conv1D`-style module rather than `nn.Linear`.

The implementation:

- inspects all `c_attn` modules before patching
- records module type and weight shape
- wraps fused `c_attn`
- preserves the frozen base output
- splits the fused output into Q/K/V sections
- adds LoRA deltas to the selected Q/K/V sections
- verifies wrapped output shape equals original output shape

Core mode applies LoRA to all Q/K/V sections of the fused `c_attn` projection.

## Frozen vs Trainable Parameters

Safety checks enforce:

- every original GPT-2 parameter is frozen
- at least one LoRA parameter is trainable
- only `lora_A` and `lora_B` parameters require gradients
- trainable parameter names are printed and saved to `trainable_params.txt`
- total params, trainable params, and trainable percentage are printed

Training uses AdamW over LoRA parameters only.

## Dataset

The dataset strategy matches `LM/partA`:

- PennTreeBank train/dev/test
- GPT-2 tokenizer
- `tokenizer.pad_token = tokenizer.eos_token`
- chunks of length `block_size + 1`
- `input_ids = tokenized[:, :-1]`
- `labels = tokenized[:, 1:]`
- pad positions are set to `-100`
- cross entropy ignores padding
- non-pad tokens are counted for loss and throughput

## Core Experiments

`--mode core` writes `results/results_partB.csv` and runs the mandatory rank/alpha sweep:

- `r=1, alpha=2`
- `r=2, alpha=4`
- `r=4, alpha=8`
- `r=8, alpha=16`

No full GPT-2 fine-tuning is performed in core mode.

## Full Extras

`--mode full` runs core plus optional experiments in `results/results_partB_extra.csv`:

- `r=16, alpha=32`
- target ablation: Q only
- target ablation: K only
- target ablation: V only
- QKV with LoRA dropout

Notes include efficiency-style reporting when a PartA baseline is available.

## Outputs

Each experiment directory contains:

- `best_lora_adapters.pt`
- `last_lora_adapters.pt`
- `config_best_lora.json`
- `trainable_params.txt`
- `c_attn_inspection.json`
- `lora_injection_report.json`
- `safety_checks.json`
- `epoch_log.csv`
- `summary.txt`

The result CSV columns are exactly:

```text
part,experiment_name,mode,rank,alpha,target_modules,lr,total_params,trainable_params,trainable_percent,train_time_seconds,tokens_per_second,peak_memory_mb,dev_ppl,test_ppl,partA_best_ppl,improves_over_partA,checkpoint_path,notes
```

## Best Result And PartA Comparison

If `LM/partA/results/results_partA.csv` exists, the script reads the best scratch dev/test PPL and prints whether each LoRA run improves over scratch. The top-level run `summary.txt` stores the best LoRA configuration.

## AI Tool Usage Declaration

This project harness and implementation were completed with AI assistance. The code was reviewed and smoke-tested through the command-line entry points, and the final responsibility for interpreting results and reporting conclusions remains with the student.
