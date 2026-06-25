import math

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normed


def make_norm(norm_type: str, d_model: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model)
    if norm_type == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"Unknown norm_type={norm_type}")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float) -> None:
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
        causal = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", causal.view(1, 1, block_size, block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.proj_dropout(self.out_proj(attended))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float, activation: str = "gelu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        if self.activation == "gelu":
            x = F.gelu(x)
        elif self.activation == "relu2":
            x = F.relu(x).pow(2)
        else:
            raise ValueError(f"Unknown activation={self.activation}")
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_dim: int,
        block_size: int,
        dropout: float,
        norm_type: str = "layernorm",
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.ln1 = make_norm(norm_type, d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = make_norm(norm_type, d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT2LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        weight_tying: bool = True,
        norm_type: str = "layernorm",
        activation: str = "gelu",
        lambda_x0: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.weight_tying = weight_tying
        self.lambda_x0 = lambda_x0
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(block_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ff_dim=ff_dim,
                    block_size=block_size,
                    dropout=dropout,
                    norm_type=norm_type,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_ln = make_norm(norm_type, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

        self.param_count_before_tying = sum(p.numel() for p in self.parameters())
        if weight_tying:
            self.lm_head.weight = self.token_embed.weight
            assert self.lm_head.weight is self.token_embed.weight
        self.param_count_after_tying = sum(p.numel() for p in self.parameters())

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block_size={self.block_size}.")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_dropout(x)
        x0 = x
        for block in self.blocks:
            x = block(x)
            if self.lambda_x0:
                x = x + self.lambda_x0 * x0
        x = self.final_ln(x)
        return self.lm_head(x)
