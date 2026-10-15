"""Low-rank adapters with frozen base weights and portable, verified state."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn

from .checkpoint import atomic_save, atomic_write_bytes, file_sha256
from .data import atomic_directory


@dataclasses.dataclass(frozen=True)
class AdapterConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ("q_proj", "v_proj")

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank < 1:
            raise ValueError("adapter rank must be positive")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("adapter alpha must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("adapter dropout must be in [0, 1)")
        if not self.targets or any(not isinstance(t, str) or not t for t in self.targets):
            raise ValueError("adapter targets must be nonempty module names")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("adapter targets must be unique")


class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable rank-r residual."""

    def __init__(self, base: nn.Linear, config: AdapterConfig):
        super().__init__()
        self.base = base
        self.scale = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        self.lora_a = nn.Parameter(base.weight.new_empty(config.rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.base.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = self.dropout(inputs) @ self.lora_a.T @ self.lora_b.T
        return self.base(inputs) + residual * self.scale

    def merged(self) -> nn.Linear:
        layer = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(self.base.weight + self.scale * (self.lora_b @ self.lora_a))
            if layer.bias is not None:
                layer.bias.copy_(self.base.bias)
        return layer.train(self.training)


def _fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _targets(model: nn.Module, config: AdapterConfig) -> list[tuple[str, nn.Linear]]:
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("model already contains adapters")
    found = [
        (name, module)
        for name, module in model.named_modules()
        if name.split(".")[-1] in config.targets and isinstance(module, nn.Linear)
    ]
    missing = set(config.targets) - {name.split(".")[-1] for name, _ in found}
    if missing:
        raise ValueError(f"adapter targets do not match linear modules: {sorted(missing)}")
    if any(config.rank > min(module.in_features, module.out_features) for _, module in found):
        raise ValueError("adapter rank exceeds a target layer dimension")
    parameters = list(model.named_parameters(remove_duplicate=False))
    for name, module in found:
        for other_name, parameter in parameters:
            if other_name == f"{name}.weight":
                continue
            shared = parameter is module.weight
            if parameter.device == module.weight.device and parameter.device.type != "meta":
                shared |= (
                    parameter.untyped_storage().data_ptr()
                    == module.weight.untyped_storage().data_ptr()
                )
            if shared:
                raise ValueError(
                    f"adapter target {name} shares weight storage with {other_name}; "
                    "choose an untied projection"
                )
    return found


def _replace(model: nn.Module, name: str, module: nn.Module) -> None:
    parent, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, child, module)


def apply_adapters(model: nn.Module, config: AdapterConfig = AdapterConfig()) -> nn.Module:
    """Attach adapters in place and freeze all other parameters."""
    targets = _targets(model, config)
    fingerprint = _fingerprint(model)
    model.requires_grad_(False)
    for name, layer in targets:
        _replace(model, name, LoRALinear(layer, config).train(layer.training))
    model._clearforge_adapter_config = config
    model._clearforge_adapter_base = fingerprint
    return model


def save_adapters(model: nn.Module, output: str | Path) -> Path:
    """Write adapter tensors and their base identity to a new directory."""
    config = getattr(model, "_clearforge_adapter_config", None)
    if config is None:
        raise ValueError("model has no registered adapters")
    weights = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if name.endswith((".lora_a", ".lora_b"))
    }
    with atomic_directory(output) as staging:
        atomic_save(weights, staging / "adapter.pt")
        metadata = {
            "format": "clearforge-lora-v1",
            "config": dataclasses.asdict(config),
            "base_sha256": model._clearforge_adapter_base,
            "weights_sha256": file_sha256(staging / "adapter.pt"),
        }
        atomic_write_bytes(json.dumps(metadata, sort_keys=True).encode(), staging / "adapter.json")
    return Path(output)


def load_adapters(model: nn.Module, directory: str | Path) -> nn.Module:
    """Validate base, tensor shapes and digest before changing the model."""
    directory = Path(directory)
    metadata = json.loads((directory / "adapter.json").read_text())
    if metadata.get("format") != "clearforge-lora-v1":
        raise ValueError("unsupported adapter format")
    config = AdapterConfig(**metadata["config"])
    targets = _targets(model, config)
    if _fingerprint(model) != metadata["base_sha256"]:
        raise ValueError("adapter base weights do not match")
    with (directory / "adapter.pt").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != metadata["weights_sha256"]:
            raise ValueError("adapter integrity check failed")
        stream.seek(0)
        weights = torch.load(stream, map_location="cpu", weights_only=True)
    expected = {}
    for name, module in targets:
        expected[f"{name}.lora_a"] = (config.rank, module.in_features)
        expected[f"{name}.lora_b"] = (module.out_features, config.rank)
    if not isinstance(weights, dict) or set(weights) != set(expected):
        raise ValueError("adapter tensor names do not match the configured targets")
    for name, shape in expected.items():
        value = weights[name]
        if (
            not isinstance(value, Tensor)
            or tuple(value.shape) != shape
            or not value.is_floating_point()
        ):
            raise ValueError(f"invalid adapter tensor: {name}")
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite adapter tensor: {name}")
    apply_adapters(model, config)
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        for name, value in weights.items():
            parameters[name].copy_(value)
    return model


def merge_adapters(model: nn.Module) -> nn.Module:
    """Fold low-rank updates into linear weights and remove adapter modules."""
    layers = [
        (name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    if not layers:
        raise ValueError("model has no adapters to merge")
    for name, module in layers:
        _replace(model, name, module.merged())
    model.requires_grad_(True)
    for name in ("_clearforge_adapter_config", "_clearforge_adapter_base"):
        if hasattr(model, name):
            delattr(model, name)
    return model
