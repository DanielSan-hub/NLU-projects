import torch
from torch import nn


BERT_MODEL = "bert-base-uncased"
GPT2_MODEL = "openai-community/gpt2"


class PretrainedMultitaskNLU(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        n_slots: int,
        n_intents: int,
        pooling: str,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.slot_classifier = nn.Linear(hidden_size, n_slots)
        self.intent_classifier = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = self.dropout(outputs.last_hidden_state)
        slot_logits = self.slot_classifier(hidden)

        if self.pooling == "cls":
            pooled = hidden[:, 0]
        elif self.pooling == "last":
            last_index = attention_mask.sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            raise ValueError(f"Unknown pooling={self.pooling}")
        intent_logits = self.intent_classifier(pooled)
        return slot_logits, intent_logits


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError("NLU/partB requires transformers.") from exc

    tokenizer_kwargs = {"use_fast": True}
    if model_name == GPT2_MODEL:
        tokenizer_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if model_name == GPT2_MODEL and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_pretrained_multitask_model(model_name: str, tokenizer, n_slots: int, n_intents: int, pooling: str, dropout: float):
    try:
        from transformers import AutoModel
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise ImportError("NLU/partB requires transformers.") from exc

    encoder = AutoModel.from_pretrained(model_name)
    if len(tokenizer) != encoder.config.vocab_size:
        encoder.resize_token_embeddings(len(tokenizer))
    if getattr(encoder.config, "pad_token_id", None) is None:
        encoder.config.pad_token_id = tokenizer.pad_token_id

    model = PretrainedMultitaskNLU(
        encoder=encoder,
        hidden_size=encoder.config.hidden_size,
        n_slots=n_slots,
        n_intents=n_intents,
        pooling=pooling,
        dropout=dropout,
    )
    return model
