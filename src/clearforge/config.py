"""Strict, immutable configuration and configuration-relative path resolution."""

from __future__ import annotations

import dataclasses
import math
import tomllib
from pathlib import Path
from typing import Any


def _integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _real(name: str, value: float, minimum: float, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum or 'infinity'}]")


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Decoder dimensions; defaults describe a nominal 100M-parameter model."""

    vocab_size: int = 32768
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int = 4
    intermediate_size: int = 2048
    max_seq_len: int = 1024
    dropout: float = 0.0
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    num_experts: int = 0
    experts_per_token: int = 2
    router_aux_loss_coef: float = 0.001
    yarn_factor: float = 1.0
    yarn_original_context: int = 2048
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "hidden_size",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "intermediate_size",
            "max_seq_len",
        ):
            _integer(name, getattr(self, name))
        if self.vocab_size < 8 or self.max_seq_len < 2:
            raise ValueError("vocab_size must be >= 8 and max_seq_len must be >= 2")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if (self.hidden_size // self.num_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        _real("dropout", self.dropout, 0.0, 1.0)
        if self.dropout == 1.0:
            raise ValueError("dropout must be less than 1")
        _real("rope_theta", self.rope_theta, 1.0)
        _integer("num_experts", self.num_experts, 0)
        _integer("experts_per_token", self.experts_per_token)
        if self.num_experts and self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token cannot exceed num_experts")
        _real("router_aux_loss_coef", self.router_aux_loss_coef, 0.0)
        _real("yarn_factor", self.yarn_factor, 1.0)
        _integer("yarn_original_context", self.yarn_original_context, 2)
        _real("yarn_beta_slow", self.yarn_beta_slow, 0.000001)
        _real("yarn_beta_fast", self.yarn_beta_fast, self.yarn_beta_slow)
        if self.yarn_factor > 1 and self.rope_theta <= 1:
            raise ValueError("YaRN requires rope_theta > 1")
        if type(self.tie_embeddings) is not bool:
            raise ValueError("tie_embeddings must be a boolean")


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """Optimization controls measured in optimizer updates, not microbatches."""

    max_steps: int = 10000
    batch_size: int = 4
    accumulation_steps: int = 8
    learning_rate: float = 0.0003
    weight_decay: float = 0.1
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "auto"
    precision: str = "bf16"
    keep_checkpoints: int = 3
    save_every: int = 100
    eval_every: int = 100
    eval_batches: int = 10
    deterministic: bool = True
    epochs: int | None = None
    compile_model: bool = False
    compile_backend: str = "inductor"
    activation_checkpointing: bool = False
    data_workers: int = 0
    pin_memory: bool = False
    log_every: int = 10

    def __post_init__(self) -> None:
        for name in (
            "max_steps",
            "batch_size",
            "accumulation_steps",
            "save_every",
            "keep_checkpoints",
            "eval_every",
            "eval_batches",
        ):
            _integer(name, getattr(self, name))
        if self.epochs is not None:
            _integer("epochs", self.epochs)
        _integer("data_workers", self.data_workers, 0)
        _integer("log_every", self.log_every, 0)
        for name in ("compile_model", "activation_checkpointing", "pin_memory"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.compile_backend not in {"inductor", "eager", "aot_eager"}:
            raise ValueError("compile_backend must be inductor, eager, or aot_eager")
        _integer("warmup_steps", self.warmup_steps, 0)
        _integer("seed", self.seed, 0)
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")
        _real("learning_rate", self.learning_rate, 0.0)
        _real("grad_clip", self.grad_clip, 0.0)
        if self.learning_rate == 0 or self.grad_clip == 0:
            raise ValueError("learning_rate and grad_clip must be positive")
        _real("weight_decay", self.weight_decay, 0.0)
        _real("min_lr_ratio", self.min_lr_ratio, 0.0, 1.0)
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        if not isinstance(self.device, str) or (
            self.device not in {"auto", "cpu", "cuda"}
            and not (self.device.startswith("cuda:") and self.device[5:].isdigit())
        ):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
        if type(self.deterministic) is not bool:
            raise ValueError("deterministic must be a boolean")


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Resolved configuration for one independently owned output directory."""

    model: ModelConfig
    training: TrainConfig
    data_dir: Path
    tokenizer_path: Path
    output_dir: Path
    stage: str = "pretrain"
    init_checkpoint: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, ModelConfig) or not isinstance(self.training, TrainConfig):
            raise ValueError("model and training must be validated configuration objects")
        if self.stage not in {"pretrain", "sft"}:
            raise ValueError("stage must be pretrain or sft")
        for name in ("data_dir", "tokenizer_path", "output_dir", "init_checkpoint"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, (str, Path)) or not str(value).strip():
                    raise ValueError(f"{name} must be a nonempty path")
                object.__setattr__(self, name, Path(value).expanduser().resolve())
            elif name != "init_checkpoint":
                raise ValueError(f"{name} is required")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible resolved configuration for provenance."""
        result = dataclasses.asdict(self)
        for key, value in result.items():
            if isinstance(value, Path):
                result[key] = str(value)
        return result


def _section(cls: type, value: Any, name: str) -> Any:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a TOML table")
    unknown = set(value) - {field.name for field in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"unknown {name} keys: {', '.join(sorted(unknown))}")
    return cls(**value)


def load_config(path: str | Path) -> RunConfig:
    """Read TOML without silently accepting misspelled keys or cwd-relative paths."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    unknown = set(value) - {field.name for field in dataclasses.fields(RunConfig)}
    if unknown:
        raise ValueError(f"unknown run keys: {', '.join(sorted(unknown))}")
    model = _section(ModelConfig, value.pop("model", {}), "model")
    training = _section(TrainConfig, value.pop("training", {}), "training")
    for name in ("data_dir", "tokenizer_path", "output_dir", "init_checkpoint"):
        if name in value:
            if not isinstance(value[name], str) or not value[name].strip():
                raise ValueError(f"{name} must be a nonempty string")
            resolved = Path(value[name]).expanduser()
            value[name] = resolved if resolved.is_absolute() else path.parent / resolved
    missing = {"data_dir", "tokenizer_path", "output_dir"} - set(value)
    if missing:
        raise ValueError(f"missing run keys: {', '.join(sorted(missing))}")
    return RunConfig(model=model, training=training, **value)
