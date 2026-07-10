import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


PAD = "<pad>"
UNK = "<unk>"
CLS = "<cls>"
PAD_SLOT = "<pad>"

RESULT_COLUMNS = [
    "part",
    "experiment_name",
    "mode",
    "lr",
    "d_model",
    "n_heads",
    "num_layers",
    "ff_dim",
    "dropout",
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
    "decision",
    "notes",
]


class Lang:
    def __init__(self, specials: list[str] | None = None) -> None:
        self.id2item: list[str] = []
        self.item2id: dict[str, int] = {}
        for item in specials or []:
            self.add(item)

    def add(self, item: str) -> int:
        if item not in self.item2id:
            self.item2id[item] = len(self.id2item)
            self.id2item.append(item)
        return self.item2id[item]

    def encode(self, items: list[str], unk: str | None = None) -> list[int]:
        if unk is None:
            return [self.item2id[item] for item in items]
        unk_id = self.item2id[unk]
        return [self.item2id.get(item, unk_id) for item in items]

    def __len__(self) -> int:
        return len(self.id2item)


def load_atis(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for item in raw:
        tokens = item["utterance"].split()
        slots = item["slots"].split()
        if len(tokens) != len(slots):
            raise ValueError(f"Token/slot length mismatch for utterance: {item['utterance']}")
        records.append({"tokens": tokens, "slots": slots, "intent": item["intent"]})
    return records


def stratified_train_dev_split(
    records: list[dict],
    seed: int,
    dev_fraction: float = 0.1,
) -> tuple[list[dict], list[dict]]:
    by_intent: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_intent[record["intent"]].append(record)

    rng = random.Random(seed)
    train_records: list[dict] = []
    dev_records: list[dict] = []
    for intent, group in sorted(by_intent.items()):
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


def build_langs(train_records: list[dict], all_records: list[dict]) -> tuple[Lang, Lang, Lang]:
    word_lang = Lang([PAD, UNK, CLS])
    slot_lang = Lang([PAD_SLOT])
    intent_lang = Lang()
    for record in train_records:
        for token in record["tokens"]:
            word_lang.add(token)
    for record in all_records:
        for slot in record["slots"]:
            slot_lang.add(slot)
        intent_lang.add(record["intent"])
    return word_lang, slot_lang, intent_lang


class ATISDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        word_lang: Lang,
        slot_lang: Lang,
        intent_lang: Lang,
        max_tokens: int,
    ) -> None:
        self.records = records
        self.word_lang = word_lang
        self.slot_lang = slot_lang
        self.intent_lang = intent_lang
        self.max_tokens = max_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        tokens = record["tokens"][: self.max_tokens]
        slots = record["slots"][: self.max_tokens]
        input_ids = self.word_lang.encode(tokens, unk=UNK) + [self.word_lang.item2id[CLS]]
        slot_labels = self.slot_lang.encode(slots) + [self.slot_lang.item2id[PAD_SLOT]]
        slot_eval_mask = [1] * len(slots) + [0]
        cls_index = len(input_ids) - 1
        return {
            "input_ids": input_ids,
            "slot_labels": slot_labels,
            "intent_label": self.intent_lang.item2id[record["intent"]],
            "slot_eval_mask": slot_eval_mask,
            "cls_index": cls_index,
            "tokens": tokens,
        }


def make_collate_fn(pad_id: int, slot_pad_id: int):
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        attention_mask = []
        slot_labels = []
        slot_eval_mask = []
        cls_index = []
        intent_labels = []
        for item in batch:
            length = len(item["input_ids"])
            pad_len = max_len - length
            input_ids.append(item["input_ids"] + [pad_id] * pad_len)
            attention_mask.append([1] * length + [0] * pad_len)
            slot_labels.append(item["slot_labels"] + [slot_pad_id] * pad_len)
            slot_eval_mask.append(item["slot_eval_mask"] + [0] * pad_len)
            cls_index.append(item["cls_index"])
            intent_labels.append(item["intent_label"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
            "slot_eval_mask": torch.tensor(slot_eval_mask, dtype=torch.bool),
            "cls_index": torch.tensor(cls_index, dtype=torch.long),
            "intent_labels": torch.tensor(intent_labels, dtype=torch.long),
        }

    return collate


def make_dataloaders(project_dir: Path, config: dict, args):
    data_dir = project_dir / "dataset" / "ATIS"
    train_full = load_atis(data_dir / "train.json")
    test_records = load_atis(data_dir / "test.json")
    train_records, dev_records = stratified_train_dev_split(train_full, seed=args.seed)
    word_lang, slot_lang, intent_lang = build_langs(train_records, train_full + test_records)

    train_dataset = ATISDataset(train_records, word_lang, slot_lang, intent_lang, config["max_tokens"])
    dev_dataset = ATISDataset(dev_records, word_lang, slot_lang, intent_lang, config["max_tokens"])
    test_dataset = ATISDataset(test_records, word_lang, slot_lang, intent_lang, config["max_tokens"])
    collate_fn = make_collate_fn(word_lang.item2id[PAD], slot_lang.item2id[PAD_SLOT])

    def loader(dataset, shuffle):
        generator = torch.Generator()
        generator.manual_seed(args.seed)
        return DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=collate_fn,
            generator=generator if shuffle else None,
        )

    metadata = {
        "vocab_size": len(word_lang),
        "n_slots": len(slot_lang),
        "n_intents": len(intent_lang),
        "pad_id": word_lang.item2id[PAD],
        "cls_id": word_lang.item2id[CLS],
        "slot_pad_id": slot_lang.item2id[PAD_SLOT],
        "id2slot": slot_lang.id2item,
        "id2intent": intent_lang.id2item,
        "train_examples": len(train_records),
        "dev_examples": len(dev_records),
        "test_examples": len(test_records),
    }
    return loader(train_dataset, True), loader(dev_dataset, False), loader(test_dataset, False), metadata


def multitask_loss(slot_logits, intent_logits, slot_labels, intent_labels, slot_pad_id: int):
    slot_loss = F.cross_entropy(slot_logits.reshape(-1, slot_logits.size(-1)), slot_labels.reshape(-1), ignore_index=slot_pad_id)
    intent_loss = F.cross_entropy(intent_logits, intent_labels)
    return slot_loss + intent_loss, slot_loss, intent_loss


def _split_tag(tag: str) -> tuple[str, str | None]:
    if tag == "O" or tag == PAD_SLOT:
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
def evaluate(model, loader, device, config, metadata, max_batches: int | None = None, ontology: dict | None = None) -> dict:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    intent_correct = 0
    semantic_frame_correct = 0
    pred_slot_sequences: list[list[str]] = []
    gold_slot_sequences: list[list[str]] = []
    illegal_slots = 0
    predicted_slots = 0

    id2slot = metadata["id2slot"]
    slot_pad_id = metadata["slot_pad_id"]
    for step, batch in enumerate(loader, start=1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        cls_index = batch["cls_index"].to(device, non_blocking=True)
        slot_labels = batch["slot_labels"].to(device, non_blocking=True)
        intent_labels = batch["intent_labels"].to(device, non_blocking=True)
        slot_eval_mask = batch["slot_eval_mask"].to(device, non_blocking=True)
        slot_logits, intent_logits = model(input_ids, attention_mask, cls_index)
        loss, _, _ = multitask_loss(slot_logits, intent_logits, slot_labels, intent_labels, slot_pad_id)

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        intent_pred = intent_logits.argmax(dim=-1)
        intent_correct += (intent_pred == intent_labels).sum().item()
        slot_pred = slot_logits.argmax(dim=-1)

        for row_idx in range(batch_size):
            mask = slot_eval_mask[row_idx].bool()
            pred_ids = slot_pred[row_idx][mask].detach().cpu().tolist()
            gold_ids = slot_labels[row_idx][mask].detach().cpu().tolist()
            pred_tags = [id2slot[idx] for idx in pred_ids]
            gold_tags = [id2slot[idx] for idx in gold_ids]
            pred_slot_sequences.append(pred_tags)
            gold_slot_sequences.append(gold_tags)
            slots_match = pred_ids == gold_ids
            if slots_match and int(intent_pred[row_idx].item()) == int(intent_labels[row_idx].item()):
                semantic_frame_correct += 1
            if ontology is not None:
                valid = ontology.get(int(intent_pred[row_idx].item()), set())
                for slot_id in pred_ids:
                    if slot_id != slot_pad_id and id2slot[slot_id] != "O":
                        predicted_slots += 1
                        if slot_id not in valid:
                            illegal_slots += 1
        if max_batches and step >= max_batches:
            break

    slot_f1 = conll_f1(pred_slot_sequences, gold_slot_sequences)
    return {
        "loss": total_loss / max(total_examples, 1),
        "intent_acc": intent_correct / max(total_examples, 1),
        "slot_f1": slot_f1,
        "semantic_frame_acc": semantic_frame_correct / max(total_examples, 1),
        "illegal_slot_rate": illegal_slots / predicted_slots if predicted_slots else 0.0,
        "frame_validity": 1.0 - (illegal_slots / predicted_slots if predicted_slots else 0.0),
    }


def build_intent_slot_ontology(loader, metadata) -> dict[int, set[int]]:
    ontology: dict[int, set[int]] = defaultdict(set)
    id2slot = metadata["id2slot"]
    for batch in loader:
        for intent_id, slot_labels, mask in zip(batch["intent_labels"], batch["slot_labels"], batch["slot_eval_mask"]):
            valid = ontology[int(intent_id.item())]
            for slot_id, keep in zip(slot_labels.tolist(), mask.tolist()):
                if keep and id2slot[slot_id] != "O":
                    valid.add(slot_id)
    return ontology


def base_config() -> dict:
    return {
        "epochs": 20,
        "batch_size": 64,
        "max_tokens": 48,
        "lr": 5e-4,
        "weight_decay": 0.01,
        "d_model": 128,
        "n_heads": 4,
        "num_layers": 2,
        "ff_dim": 512,
        "dropout": 0.1,
        "max_train_batches": None,
        "max_eval_batches": None,
        "use_ontology_metrics": False,
    }


def with_overrides(name: str, overrides: dict, mode: str, notes: str = "") -> dict:
    cfg = base_config()
    cfg.update(overrides)
    cfg["experiment_name"] = name
    cfg["mode"] = mode
    cfg["experiment_notes"] = notes
    return cfg


def smoke_experiments() -> list[dict]:
    return [
        with_overrides(
            "smoke_shapes_losses_metrics_checkpoint_csv",
            {
                "epochs": 1,
                "batch_size": 4,
                "max_tokens": 24,
                "d_model": 64,
                "n_heads": 2,
                "num_layers": 1,
                "ff_dim": 256,
                "max_train_batches": 2,
                "max_eval_batches": 2,
            },
            "smoke",
            "1-2 batch verification of data shapes, slot/intent losses, metrics, checkpoint and CSV.",
        )
    ]


def core_experiments() -> list[dict]:
    return [
        with_overrides("baseline_lr_5e-4", {"lr": 5e-4}, "core", "Baseline fixed architecture."),
        with_overrides("lr_sweep_1e-3", {"lr": 1e-3}, "core", "LR sweep, fixed architecture."),
        with_overrides("lr_sweep_3e-4", {"lr": 3e-4}, "core", "LR sweep, fixed architecture."),
        with_overrides("ablation_d_model_192", {"d_model": 192}, "core", "One-at-a-time d_model ablation."),
        with_overrides("ablation_n_heads_8", {"n_heads": 8}, "core", "One-at-a-time n_heads ablation."),
        with_overrides("ablation_num_layers_3", {"num_layers": 3}, "core", "One-at-a-time num_layers ablation."),
        with_overrides("ablation_ff_dim_768", {"ff_dim": 768}, "core", "One-at-a-time ff_dim ablation."),
        with_overrides("dropout_before_heads_0_2", {"dropout": 0.2}, "core", "Dropout before slot_out and intent_out."),
    ]


def extra_experiments() -> list[dict]:
    return [
        with_overrides(
            "extra_semantic_frame_metrics",
            {"use_ontology_metrics": True},
            "full",
            "Optional semantic-frame/ontology reporting without replacing accuracy/F1.",
        )
    ]


def experiments_for_mode(mode: str) -> tuple[list[dict], list[dict]]:
    if mode == "smoke":
        return smoke_experiments(), []
    if mode == "core":
        return core_experiments(), []
    if mode == "full":
        return core_experiments(), extra_experiments()
    raise ValueError(f"Unknown mode={mode}")
