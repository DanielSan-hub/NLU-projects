import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from model import BERT_MODEL, GPT2_MODEL


RESULT_COLUMNS = [
    "part",
    "model_name",
    "pretrained_model",
    "mode",
    "lr",
    "batch_size",
    "epochs",
    "lambda_slot",
    "lambda_intent",
    "total_params",
    "trainable_params",
    "train_time_seconds",
    "peak_memory_mb",
    "intent_acc_dev",
    "slot_f1_dev",
    "intent_acc_test",
    "slot_f1_test",
    "semantic_frame_acc_test",
    "checkpoint_path",
    "notes",
]


def load_atis(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for item in raw:
        words = item["utterance"].split()
        slots = item["slots"].split()
        if len(words) != len(slots):
            raise ValueError(f"Token/slot length mismatch for utterance: {item['utterance']}")
        records.append({"words": words, "slots": slots, "intent": item["intent"]})
    return records


def stratified_train_dev_split(records: list[dict], seed: int, dev_fraction: float = 0.1) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["intent"]].append(record)
    train_records: list[dict] = []
    dev_records: list[dict] = []
    for _, group in sorted(grouped.items()):
        group = list(group)
        rng.shuffle(group)
        if len(group) < 2:
            train_records.extend(group)
            continue
        dev_size = max(1, int(round(len(group) * dev_fraction)))
        dev_size = min(dev_size, len(group) - 1)
        dev_records.extend(group[:dev_size])
        train_records.extend(group[dev_size:])
    rng.shuffle(train_records)
    rng.shuffle(dev_records)
    return train_records, dev_records


def build_label_maps(records: list[dict]) -> tuple[dict[str, int], list[str], dict[str, int], list[str]]:
    slot_labels = sorted({slot for record in records for slot in record["slots"]})
    intent_labels = sorted({record["intent"] for record in records})
    slot2id = {label: idx for idx, label in enumerate(slot_labels)}
    intent2id = {label: idx for idx, label in enumerate(intent_labels)}
    return slot2id, slot_labels, intent2id, intent_labels


def load_splits(project_dir: Path, seed: int):
    data_dir = project_dir / "dataset" / "ATIS"
    train_full = load_atis(data_dir / "train.json")
    test_records = load_atis(data_dir / "test.json")
    train_records, dev_records = stratified_train_dev_split(train_full, seed)
    slot2id, id2slot, intent2id, id2intent = build_label_maps(train_full + test_records)
    return train_records, dev_records, test_records, slot2id, id2slot, intent2id, id2intent


def tokenize_and_align_labels(record: dict, tokenizer, slot2id: dict[str, int], intent2id: dict[str, int], max_length: int) -> dict:
    encoding = tokenizer(
        record["words"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_attention_mask=True,
    )
    try:
        word_ids = encoding.word_ids()
    except Exception as exc:
        raise RuntimeError("A fast tokenizer with word_ids() support is required for subtoken alignment.") from exc

    slot_labels = []
    first_subtoken_mask = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            slot_labels.append(-100)
            first_subtoken_mask.append(0)
        elif word_id != previous_word_id:
            slot_labels.append(slot2id[record["slots"][word_id]])
            first_subtoken_mask.append(1)
        else:
            slot_labels.append(-100)
            first_subtoken_mask.append(0)
        previous_word_id = word_id

    return {
        "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
        "intent_labels": torch.tensor(intent2id[record["intent"]], dtype=torch.long),
        "first_subtoken_mask": torch.tensor(first_subtoken_mask, dtype=torch.bool),
    }


class TokenizedATISDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        tokenizer,
        slot2id: dict[str, int],
        intent2id: dict[str, int],
        max_length: int,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.intent2id = intent2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return tokenize_and_align_labels(
            self.records[index],
            self.tokenizer,
            self.slot2id,
            self.intent2id,
            self.max_length,
        )


def make_dataloaders(project_dir: Path, config: dict, args, tokenizer):
    train_records, dev_records, test_records, slot2id, id2slot, intent2id, id2intent = load_splits(project_dir, args.seed)

    def dataset(records):
        return TokenizedATISDataset(records, tokenizer, slot2id, intent2id, config["max_length"])

    def loader(ds, shuffle):
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        return DataLoader(
            ds,
            batch_size=config["batch_size"],
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            generator=generator if shuffle else None,
        )

    metadata = {
        "n_slots": len(id2slot),
        "n_intents": len(id2intent),
        "id2slot": id2slot,
        "id2intent": id2intent,
        "slot2id": slot2id,
        "intent2id": intent2id,
        "train_examples": len(train_records),
        "dev_examples": len(dev_records),
        "test_examples": len(test_records),
    }
    return loader(dataset(train_records), True), loader(dataset(dev_records), False), loader(dataset(test_records), False), metadata


def multitask_loss(slot_logits, intent_logits, slot_labels, intent_labels, lambda_slot: float, lambda_intent: float):
    slot_loss = F.cross_entropy(slot_logits.reshape(-1, slot_logits.size(-1)), slot_labels.reshape(-1), ignore_index=-100)
    intent_loss = F.cross_entropy(intent_logits, intent_labels)
    return lambda_slot * slot_loss + lambda_intent * intent_loss, slot_loss, intent_loss


def _split_tag(tag: str) -> tuple[str, str | None]:
    if tag == "O":
        return "O", None
    if "-" not in tag:
        return "B", tag
    prefix, entity_type = tag.split("-", 1)
    return prefix, entity_type


def bio_entities(tags: list[str]) -> set[tuple[str, int, int]]:
    entities = set()
    start = None
    current_type = None
    for idx, tag in enumerate(tags + ["O"]):
        prefix, entity_type = _split_tag(tag)
        starts_new = prefix == "B" or (prefix == "I" and entity_type != current_type)
        ends_current = current_type is not None and (prefix in {"O", "B"} or entity_type != current_type)
        if ends_current:
            entities.add((current_type, start, idx - 1))
            start = None
            current_type = None
        if starts_new:
            start = idx
            current_type = entity_type
    return entities


def conll_f1(pred_sequences: list[list[str]], gold_sequences: list[list[str]]) -> float:
    true_positive = 0
    predicted = 0
    gold = 0
    for pred_tags, gold_tags in zip(pred_sequences, gold_sequences):
        pred_entities = bio_entities(pred_tags)
        gold_entities = bio_entities(gold_tags)
        true_positive += len(pred_entities & gold_entities)
        predicted += len(pred_entities)
        gold += len(gold_entities)
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@torch.no_grad()
def evaluate(model, loader, device, config: dict, metadata: dict, ontology: dict | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    intent_correct = 0
    semantic_frame_correct = 0
    pred_sequences: list[list[str]] = []
    gold_sequences: list[list[str]] = []
    illegal_slots = 0
    predicted_slots = 0
    id2slot = metadata["id2slot"]
    o_id = metadata["slot2id"].get("O")

    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        slot_labels = batch["slot_labels"].to(device, non_blocking=True)
        intent_labels = batch["intent_labels"].to(device, non_blocking=True)
        slot_logits, intent_logits = model(input_ids, attention_mask)
        loss, _, _ = multitask_loss(
            slot_logits,
            intent_logits,
            slot_labels,
            intent_labels,
            config["lambda_slot"],
            config["lambda_intent"],
        )
        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        intent_pred = intent_logits.argmax(dim=-1)
        intent_correct += (intent_pred == intent_labels).sum().item()
        slot_pred = slot_logits.argmax(dim=-1)

        for row_idx in range(batch_size):
            valid = slot_labels[row_idx] != -100
            pred_ids = slot_pred[row_idx][valid].detach().cpu().tolist()
            gold_ids = slot_labels[row_idx][valid].detach().cpu().tolist()
            pred_tags = [id2slot[idx] for idx in pred_ids]
            gold_tags = [id2slot[idx] for idx in gold_ids]
            pred_sequences.append(pred_tags)
            gold_sequences.append(gold_tags)
            if pred_ids == gold_ids and int(intent_pred[row_idx].item()) == int(intent_labels[row_idx].item()):
                semantic_frame_correct += 1
            if ontology is not None:
                valid_slots = ontology.get(int(intent_pred[row_idx].item()), set())
                for slot_id in pred_ids:
                    if o_id is not None and slot_id == o_id:
                        continue
                    predicted_slots += 1
                    if slot_id not in valid_slots:
                        illegal_slots += 1
        if config["max_eval_batches"] and step >= config["max_eval_batches"]:
            break

    return {
        "loss": total_loss / max(total_examples, 1),
        "intent_acc": intent_correct / max(total_examples, 1),
        "slot_f1": conll_f1(pred_sequences, gold_sequences),
        "semantic_frame_acc": semantic_frame_correct / max(total_examples, 1),
        "illegal_slot_rate": illegal_slots / predicted_slots if predicted_slots else 0.0,
    }


def build_intent_slot_ontology(loader, metadata: dict) -> dict[int, set[int]]:
    ontology: dict[int, set[int]] = defaultdict(set)
    o_id = metadata["slot2id"].get("O")
    for batch in loader:
        for intent_id, labels in zip(batch["intent_labels"], batch["slot_labels"]):
            valid = labels != -100
            for slot_id in labels[valid].tolist():
                if o_id is not None and slot_id == o_id:
                    continue
                ontology[int(intent_id.item())].add(int(slot_id))
    return ontology


def verify_alignment_and_shapes(model, batch, device, config: dict, metadata: dict, model_name: str) -> dict:
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    slot_labels = batch["slot_labels"].to(device)
    intent_labels = batch["intent_labels"].to(device)
    with torch.no_grad():
        slot_logits, intent_logits = model(input_ids, attention_mask)
        total_loss, slot_loss, intent_loss = multitask_loss(
            slot_logits,
            intent_logits,
            slot_labels,
            intent_labels,
            config["lambda_slot"],
            config["lambda_intent"],
        )

    ignored = int((slot_labels == -100).sum().item())
    valid = int((slot_labels != -100).sum().item())
    assert ignored > 0, "slot labels must contain -100 for ignored subtokens/special/pad tokens"
    assert valid > 0, "batch must contain valid slot labels"
    assert intent_labels.shape == (input_ids.size(0),)
    assert slot_logits.shape == (input_ids.size(0), input_ids.size(1), metadata["n_slots"])
    assert intent_logits.shape == (input_ids.size(0), metadata["n_intents"])
    assert torch.isfinite(total_loss)
    assert torch.isfinite(slot_loss)
    assert torch.isfinite(intent_loss)
    if model_name == "bert":
        assert model.pooling == "cls"
    if model_name.startswith("gpt2"):
        assert model.pooling in {"last", "mean"}
    return {
        "input_shape": tuple(input_ids.shape),
        "slot_logits_shape": tuple(slot_logits.shape),
        "intent_logits_shape": tuple(intent_logits.shape),
        "ignored_slot_labels": ignored,
        "valid_slot_labels": valid,
        "slot_loss": float(slot_loss.item()),
        "intent_loss": float(intent_loss.item()),
        "pooling": model.pooling,
    }


def model_configs_for_mode(mode: str) -> tuple[list[dict], list[dict]]:
    smoke_common = {
        "mode": "smoke",
        "epochs": 1,
        "batch_size": 2,
        "max_length": 32,
        "lr": 3e-5,
        "weight_decay": 0.01,
        "dropout": 0.1,
        "lambda_slot": 1.0,
        "lambda_intent": 1.0,
        "max_train_batches": 1,
        "max_eval_batches": 1,
        "use_ontology_gate": False,
    }
    core_common = {
        "mode": "core",
        "epochs": 4,
        "batch_size": 16,
        "max_length": 64,
        "lr": 3e-5,
        "weight_decay": 0.01,
        "dropout": 0.1,
        "lambda_slot": 1.0,
        "lambda_intent": 1.0,
        "max_train_batches": None,
        "max_eval_batches": None,
        "use_ontology_gate": False,
    }
    full_common = dict(core_common)
    full_common["mode"] = "full"
    full_common["epochs"] = 5

    if mode == "smoke":
        return [
            {**smoke_common, "model_name": "bert", "pretrained_model": BERT_MODEL, "pooling": "cls"},
            {**smoke_common, "model_name": "gpt2", "pretrained_model": GPT2_MODEL, "pooling": "last"},
        ], []
    if mode == "core":
        return [
            {**core_common, "model_name": "bert", "pretrained_model": BERT_MODEL, "pooling": "cls"},
            {**core_common, "model_name": "gpt2", "pretrained_model": GPT2_MODEL, "pooling": "last"},
        ], []
    if mode == "full":
        core = [
            {**core_common, "model_name": "bert", "pretrained_model": BERT_MODEL, "pooling": "cls"},
            {**core_common, "model_name": "gpt2", "pretrained_model": GPT2_MODEL, "pooling": "last"},
        ]
        extras = [
            {**full_common, "model_name": "gpt2_mean_pool", "pretrained_model": GPT2_MODEL, "pooling": "mean"},
            {**full_common, "model_name": "bert_ontology_report", "pretrained_model": BERT_MODEL, "pooling": "cls", "use_ontology_gate": True},
        ]
        return core, extras
    raise ValueError(f"Unknown mode={mode}")
