import math

import torch
from torch import nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_length: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}.")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        causal = torch.tril(torch.ones(max_length, max_length, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, max_length, max_length), persistent=False)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        causal = self.causal_mask[:, :, :seq_len, :seq_len]
        key_mask = attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~(causal & key_mask), torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.proj_dropout(self.out_proj(attended))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, max_length: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_length, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT2NLU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_slots: int,
        n_intents: int,
        max_length: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.max_length = max_length
        self.pad_id = pad_id
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Embedding(max_length, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    max_length=max_length,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.head_dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(d_model, n_slots)
        self.intent_out = nn.Linear(d_model, n_intents)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cls_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_length:
            raise ValueError(f"Sequence length {seq_len} exceeds max_length={self.max_length}.")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask)
        x = self.final_ln(x)
        head_input = self.head_dropout(x)
        slot_logits = self.slot_out(head_input)
        cls_repr = head_input[torch.arange(batch_size, device=input_ids.device), cls_index]
        intent_logits = self.intent_out(cls_repr)
        return slot_logits, intent_logits
