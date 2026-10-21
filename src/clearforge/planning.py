"""Capacity planning for a run: parameters, token budget, memory, checkpoints, disk.

The estimates are analytic. They are precise for parameter, optimizer and
checkpoint sizes, and deliberately conservative for activations and logits so
that a plan which fits on paper has headroom in practice. Measure peak memory
with a short run before committing to a long budget.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, RunConfig
from .model import CausalLM

GIB = 1024**3
COMPUTE_OPTIMAL_TOKENS_PER_PARAMETER = 20.0
_MANIFEST_OVERHEAD_BYTES = 4 * 1024**2


def parameter_accounting(config: ModelConfig) -> dict[str, int]:
    """Count unique parameters on the meta device without allocating memory."""
    with torch.device("meta"):
        model = CausalLM(config)
    total = model.parameter_count()
    embedding = model.token_embedding.weight.numel()
    if not config.tie_embeddings:
        embedding += model.lm_head.weight.numel()
    return {"total": total, "embedding": embedding, "non_embedding": total - embedding}


def _activation_bytes(precision: str) -> int:
    return 4 if precision == "fp32" else 2


def memory_estimate(
    config: RunConfig, *, parameters: int, seq_length: int, device_type: str
) -> dict[str, int]:
    """Estimate peak training memory for one process in bytes.

    Weights, gradients and AdamW moments are float32. Activations count the
    tensors a decoder block keeps for backward (normalized inputs, projections,
    gated feed-forward intermediates) and, on CPU where scaled-dot-product
    attention materializes probabilities, the attention matrix. Logits include
    the mixed-precision projection, its float32 copy, and the loss gradient.
    """
    model, training = config.model, config.training
    activation = _activation_bytes(training.precision)
    tokens = training.batch_size * seq_length
    head_dim = model.hidden_size // model.num_heads
    kv_width = model.num_kv_heads * head_dim
    mlp_width = model.intermediate_size * (model.experts_per_token if model.num_experts else 1)
    per_layer = 9 * model.hidden_size + 2 * kv_width + 4 * mlp_width
    if device_type == "cpu":
        per_layer += model.num_heads * seq_length
    if training.activation_checkpointing:
        stored = model.num_layers * model.hidden_size + per_layer
    else:
        stored = model.num_layers * per_layer
    activations = tokens * (stored + 2 * model.hidden_size) * activation
    logits = tokens * model.vocab_size * (activation + 12)
    weight_cache = 0 if training.precision == "fp32" else 2 * parameters
    components = {
        "weights_bytes": 4 * parameters,
        "gradients_bytes": 4 * parameters,
        "optimizer_bytes": 8 * parameters,
        "autocast_weight_cache_bytes": weight_cache,
        "activations_bytes": activations,
        "logits_bytes": logits,
    }
    components["estimated_total_bytes"] = sum(components.values())
    return components


def checkpoint_estimate(parameters: int, train_rows: int, keep_checkpoints: int) -> dict[str, int]:
    """Size of one checkpoint (weights, moments, data cursor) and the retained set."""
    size = 12 * parameters + 8 * train_rows + _MANIFEST_OVERHEAD_BYTES
    return {"bytes": size, "retained_bytes": (keep_checkpoints + 1) * size}


def _packed_manifest(config: RunConfig) -> dict[str, Any] | None:
    path = config.data_dir / "manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("row_counts")
    if (
        type(manifest.get("seq_length")) is not int
        or not isinstance(rows, dict)
        or any(type(rows.get(split)) is not int for split in ("train", "validation"))
    ):
        raise ValueError(f"{path}: packed manifest lacks seq_length or row_counts")
    return manifest


def _free_disk_bytes(path: Path) -> int | None:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return shutil.disk_usage(candidate).free
    return None


def _device_type(config: RunConfig) -> str:
    requested = config.training.device
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda"


def plan_run(
    config: RunConfig,
    *,
    world_size: int = 1,
    device_memory_bytes: int | None = None,
    device_type: str | None = None,
) -> dict[str, Any]:
    """Describe what a configuration will consume before anything is allocated.

    ``world_size`` scales the global batch for a torchrun launch. The device
    memory defaults to the first visible CUDA device's capacity when the plan
    targets CUDA and a device is present; pass it explicitly to plan for other
    hardware. Warnings describe budget, memory and disk concerns; they never
    block the caller.
    """
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if device_memory_bytes is not None and (
        type(device_memory_bytes) is not int or device_memory_bytes < 1
    ):
        raise ValueError("device_memory_bytes must be a positive integer")
    device_type = device_type or _device_type(config)
    if device_type not in {"cpu", "cuda"}:
        raise ValueError("device_type must be cpu or cuda")
    model, training = config.model, config.training
    parameters = parameter_accounting(model)
    total = parameters["total"]
    manifest = _packed_manifest(config)
    seq_length = manifest["seq_length"] if manifest else model.max_seq_len
    sequences_per_update = training.batch_size * training.accumulation_steps * world_size
    tokens_per_update = sequences_per_update * seq_length
    max_steps = training.max_steps
    train_rows = manifest["row_counts"]["train"] if manifest else 0
    if training.epochs is not None and manifest:
        max_steps = training.epochs * -(-train_rows // sequences_per_update)
    total_sequences = sequences_per_update * max_steps
    total_tokens = total_sequences * seq_length
    schedule = {
        "max_steps": max_steps,
        "warmup_steps": training.warmup_steps,
        "lr_schedule": training.lr_schedule,
        "decay_steps": training.decay_steps,
        "world_size": world_size,
        "seq_length": seq_length,
        "sequences_per_update": sequences_per_update,
        "tokens_per_update": tokens_per_update,
        "total_sequences": total_sequences,
        "total_tokens": total_tokens,
        "tokens_per_parameter": total_tokens / total,
        "compute_optimal_tokens": int(COMPUTE_OPTIMAL_TOKENS_PER_PARAMETER * total),
        "training_flops": 6 * total * total_tokens,
    }
    data: dict[str, Any] | None = None
    if manifest:
        counts = manifest.get("token_counts", {}).get("train", {})
        arrays = [
            config.data_dir / f"{split}.{name}.npy"
            for split in ("train", "validation")
            for name in ("input_ids", "labels")
        ]
        data = {
            "manifest": str(config.data_dir / "manifest.json"),
            "train_rows": train_rows,
            "validation_rows": manifest["row_counts"]["validation"],
            "train_supervised_tokens": counts.get("supervised"),
            "train_input_tokens": counts.get("input"),
            "packed_bytes": sum(path.stat().st_size for path in arrays if path.is_file()),
            "epochs": total_sequences / train_rows if train_rows else None,
        }
    memory = memory_estimate(
        config, parameters=total, seq_length=seq_length, device_type=device_type
    )
    if device_memory_bytes is None and device_type == "cuda" and torch.cuda.is_available():
        index = 0
        if training.device.startswith("cuda:"):
            index = int(training.device[5:])
        if index < torch.cuda.device_count():
            device_memory_bytes = torch.cuda.mem_get_info(index)[1]
    memory["device_type"] = device_type
    memory["device_memory_bytes"] = device_memory_bytes
    memory["fits"] = (
        None
        if device_memory_bytes is None
        else memory["estimated_total_bytes"] <= device_memory_bytes
    )
    checkpoint = checkpoint_estimate(total, train_rows, training.keep_checkpoints)
    free = _free_disk_bytes(config.output_dir)
    disk = {
        "output_dir": str(config.output_dir),
        "free_bytes": free,
        "one_checkpoint_fits": None if free is None else free >= checkpoint["bytes"],
        "retention_fits": None if free is None else free >= checkpoint["retained_bytes"],
    }
    warnings: list[str] = []
    ratio = schedule["tokens_per_parameter"]
    if ratio < COMPUTE_OPTIMAL_TOKENS_PER_PARAMETER:
        warnings.append(
            f"training budget is {ratio:.1f} tokens per parameter; the compute-optimal "
            f"reference is about {COMPUTE_OPTIMAL_TOKENS_PER_PARAMETER:.0f}: raise "
            "max_steps or the global batch, or expand the corpus"
        )
    if data and data["epochs"] is not None and data["epochs"] > 4:
        warnings.append(
            f"the schedule makes {data['epochs']:.1f} passes over {train_rows} packed rows; "
            "expand the dataset or reduce the budget to limit memorization"
        )
    if data and data["epochs"] is not None and data["epochs"] < 1:
        warnings.append(
            f"the schedule consumes only {data['epochs']:.2f} of the packed training rows"
        )
    if memory["fits"] is False:
        warnings.append(
            f"estimated memory {memory['estimated_total_bytes'] / GIB:.1f} GiB exceeds device "
            f"memory {device_memory_bytes / GIB:.1f} GiB: reduce batch_size, enable "
            "activation_checkpointing, or shard the global batch across torchrun ranks"
        )
    if disk["one_checkpoint_fits"] is False:
        warnings.append(
            f"free disk {free / GIB:.1f} GiB cannot hold one checkpoint of "
            f"{checkpoint['bytes'] / GIB:.1f} GiB"
        )
    elif disk["retention_fits"] is False:
        warnings.append(
            f"free disk {free / GIB:.1f} GiB is below the {checkpoint['retained_bytes'] / GIB:.1f} "
            f"GiB needed for keep_checkpoints={training.keep_checkpoints} plus one in flight"
        )
    if total >= 250_000_000 and tokens_per_update < 65_536:
        warnings.append(
            f"global batch of {tokens_per_update} tokens is small for a {total / 1e6:.0f}M "
            "parameter model; raise accumulation_steps, batch_size, or world size"
        )
    return {
        "parameters": parameters,
        "schedule": schedule,
        "data": data,
        "memory": memory,
        "checkpoint": checkpoint,
        "disk": disk,
        "warnings": warnings,
    }


def summarize(plan: dict[str, Any]) -> str:
    """One readable line for logs and the start of a run."""
    schedule, memory, checkpoint = plan["schedule"], plan["memory"], plan["checkpoint"]
    parts = [
        f"{plan['parameters']['total']:,} parameters",
        f"{schedule['tokens_per_update']:,} tokens/update",
        f"{schedule['total_tokens'] / 1e9:.2f}B tokens over {schedule['max_steps']:,} updates",
        f"{schedule['tokens_per_parameter']:.1f} tokens/parameter",
        f"~{memory['estimated_total_bytes'] / GIB:.1f} GiB memory",
        f"~{checkpoint['bytes'] / GIB:.2f} GiB per checkpoint",
    ]
    if plan["data"] and plan["data"]["epochs"] is not None:
        parts.append(f"{plan['data']['epochs']:.2f} epochs")
    return "; ".join(parts)
