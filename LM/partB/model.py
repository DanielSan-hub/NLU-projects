from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LoRAInjectionReport:
    module_name: str
    module_type: str
    weight_shape: tuple[int, ...]
    base_output_shape: tuple[int, ...]
    wrapped_output_shape: tuple[int, ...]
    target_sections: str


class LoRAFusedQKVConv1D(nn.Module):
    """Manual LoRA wrapper for HuggingFace GPT-2 fused c_attn Conv1D.

    GPT-2's c_attn projects hidden states to a fused [Q, K, V] tensor. The frozen
    base projection is preserved exactly, and LoRA deltas are added only to the
    selected Q/K/V sections. B matrices are initialized exactly to zero, so the
    wrapped module is functionally identical to the base module at step 0.
    """

    def __init__(
        self,
        base_module: nn.Module,
        rank: int,
        alpha: int,
        target_sections: str = "qkv",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base_module
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        if not hasattr(base_module, "weight") or not hasattr(base_module, "bias"):
            raise TypeError("GPT-2 c_attn wrapper expects a module with weight and bias attributes.")
        weight_shape = tuple(base_module.weight.shape)
        if len(weight_shape) != 2:
            raise ValueError(f"Expected 2D c_attn weight, got shape={weight_shape}.")

        self.in_features = int(weight_shape[0])
        self.out_features = int(weight_shape[1])
        if self.out_features % 3 != 0:
            raise ValueError(f"Expected fused QKV output dim divisible by 3, got {self.out_features}.")
        self.hidden_size = self.out_features // 3
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.target_sections = "".join(section for section in "qkv" if section in target_sections.lower())
        if not self.target_sections:
            raise ValueError("target_sections must include at least one of q, k, v.")
        self.dropout = nn.Dropout(dropout)

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for section in self.target_sections:
            a = nn.Parameter(torch.empty(self.in_features, rank))
            b = nn.Parameter(torch.zeros(rank, self.hidden_size))
            nn.init.kaiming_uniform_(a, a=5**0.5)
            # B must be exactly zero so the LoRA delta is zero at step 0.
            nn.init.zeros_(b)
            self.lora_A[section] = a
            self.lora_B[section] = b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x)
        pieces = list(base_output.split(self.hidden_size, dim=-1))
        dropped = self.dropout(x)
        for index, section in enumerate("qkv"):
            if section in self.target_sections:
                delta = dropped.matmul(self.lora_A[section]).matmul(self.lora_B[section])
                pieces[index] = pieces[index] + self.scaling * delta
        output = torch.cat(pieces, dim=-1)
        if output.shape != base_output.shape:
            raise RuntimeError(f"LoRA c_attn changed output shape from {base_output.shape} to {output.shape}.")
        return output


def freeze_pretrained_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _set_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if isinstance(parent, nn.ModuleList) else getattr(parent, part)
    child_name = parts[-1]
    if isinstance(parent, nn.ModuleList):
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def inspect_c_attn_modules(model: nn.Module) -> list[dict]:
    modules = []
    for name, module in model.named_modules():
        if name.endswith("c_attn") and hasattr(module, "weight"):
            modules.append(
                {
                    "name": name,
                    "type": module.__class__.__name__,
                    "weight_shape": tuple(module.weight.shape),
                    "bias_shape": tuple(module.bias.shape) if getattr(module, "bias", None) is not None else None,
                }
            )
    return modules


def inject_lora_into_gpt2_c_attn(
    model: nn.Module,
    rank: int,
    alpha: int,
    target_sections: str,
    dropout: float,
    sample_input_ids: torch.Tensor | None = None,
) -> list[LoRAInjectionReport]:
    reports: list[LoRAInjectionReport] = []
    c_attn_modules = inspect_c_attn_modules(model)
    if not c_attn_modules:
        raise RuntimeError("No GPT-2 c_attn modules found. Inspect model architecture before patching.")

    module_names = [item["name"] for item in c_attn_modules]
    original_modules = dict(model.named_modules())
    for name in module_names:
        base_module = original_modules[name]
        base_shape = None
        wrapped_shape = None
        if sample_input_ids is not None:
            hidden_size = int(base_module.weight.shape[0])
            probe = torch.zeros(
                sample_input_ids.size(0),
                sample_input_ids.size(1),
                hidden_size,
                device=base_module.weight.device,
                dtype=base_module.weight.dtype,
            )
            with torch.no_grad():
                base_shape = tuple(base_module(probe).shape)
        wrapped = LoRAFusedQKVConv1D(
            base_module=base_module,
            rank=rank,
            alpha=alpha,
            target_sections=target_sections,
            dropout=dropout,
        )
        wrapped.to(device=base_module.weight.device, dtype=base_module.weight.dtype)
        if sample_input_ids is not None:
            with torch.no_grad():
                wrapped_shape = tuple(wrapped(probe).shape)
            if base_shape != wrapped_shape:
                raise RuntimeError(f"{name} output shape changed from {base_shape} to {wrapped_shape}.")
        _set_module(model, name, wrapped)
        reports.append(
            LoRAInjectionReport(
                module_name=name,
                module_type=base_module.__class__.__name__,
                weight_shape=tuple(base_module.weight.shape),
                base_output_shape=base_shape or tuple(),
                wrapped_output_shape=wrapped_shape or tuple(),
                target_sections=target_sections,
            )
        )
    return reports


def trainable_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def assert_lora_training_safety(model: nn.Module) -> list[str]:
    names = trainable_parameter_names(model)
    assert names, "At least one LoRA parameter must be trainable."
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert "lora_A" in name or "lora_B" in name, f"Base parameter is trainable: {name}"
        else:
            if "lora_A" in name or "lora_B" in name:
                raise AssertionError(f"LoRA parameter is unexpectedly frozen: {name}")
    return names


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    missing = [name for name in state if name not in current]
    if missing:
        raise KeyError(f"LoRA checkpoint contains unknown keys: {missing[:5]}")
    current.update(state)
    model.load_state_dict(current)
