"""Native decoder optimization with auditable inputs and exact boundary resumes.

A run owns its output directory. Resume restores an optimizer boundary, including
random state and shuffled data position, and discards only metrics beyond that
checkpoint. SFT initialization transfers weights alone into a new run.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import shutil
import signal
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor

from .checkpoint import (
    FORMAT_VERSION,
    atomic_save,
    atomic_write_bytes,
    capture_rng_state,
    file_sha256,
    load_checkpoint,
    load_checkpoint_with_hash,
    publish_directory,
    publish_latest,
    restore_rng_state,
)
from .config import RunConfig
from .data import PackedDataset
from .model import CausalLM
from .tokenizer import ChatTokenizer


@dataclasses.dataclass(frozen=True)
class TrainResult:
    """The last durable optimizer boundary and its measured losses."""

    checkpoint: Path
    status: str
    step: int
    train_loss: float | None
    validation_loss: float | None


@dataclasses.dataclass
class _Inputs:
    identities: dict[str, Any]
    train: PackedDataset
    validation: PackedDataset


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _inputs(config: RunConfig) -> _Inputs:
    """Verify actual array contents and hashes before allocating a model or run."""
    manifest_path = config.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tokenizer_hash = file_sha256(config.tokenizer_path)
    tokenizer = ChatTokenizer.load(config.tokenizer_path)
    if manifest.get("mode") != config.stage:
        raise ValueError("packed dataset mode does not match the training stage")
    if manifest.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("packed dataset tokenizer identity does not match tokenizer_path")
    if manifest.get("vocab_size") != tokenizer.vocab_size:
        raise ValueError("packed vocabulary does not match the tokenizer")
    if config.model.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary must equal the trained tokenizer vocabulary")
    sequence = manifest.get("seq_length")
    if type(sequence) is not int or not 2 <= sequence <= config.model.max_seq_len:
        raise ValueError("packed sequence length exceeds model context or is invalid")
    hashes: dict[str, str] = {}
    for split in ("train", "validation"):
        names = [f"{split}.input_ids.npy", f"{split}.labels.npy"]
        arrays = []
        for name in names:
            path = config.data_dir / name
            actual_hash = file_sha256(path)
            if manifest.get("array_hashes", {}).get(name) != actual_hash:
                raise ValueError(f"packed array hash mismatch: {name}")
            hashes[name] = actual_hash
            arrays.append(np.load(path, mmap_mode="r", allow_pickle=False))
        inputs, labels = arrays
        if (
            inputs.ndim != 2
            or inputs.shape[1] != sequence
            or inputs.shape != labels.shape
            or inputs.dtype.kind not in "iu"
            or labels.dtype.kind != "i"
            or len(inputs) == 0
            or manifest.get("row_counts", {}).get(split) != len(inputs)
        ):
            raise ValueError(f"invalid or empty {split} packed arrays")
        for start in range(0, len(inputs), 1024):
            ids = inputs[start : start + 1024]
            targets = labels[start : start + 1024]
            supervised = targets != -100
            if np.any(ids < 0) or np.any(ids >= tokenizer.vocab_size):
                raise ValueError(f"out-of-vocabulary input in {split}")
            if np.any(targets[supervised] != ids[supervised]):
                raise ValueError(f"labels must align with supervised input IDs in {split}")
            if np.any(np.sum(supervised[:, 1:], axis=1) == 0):
                raise ValueError(f"all-ignored shifted supervision in {split}")
    identities = {
        "tokenizer_sha256": tokenizer_hash,
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "array_hashes": hashes,
    }
    return _Inputs(
        identities,
        PackedDataset(config.data_dir, "train"),
        PackedDataset(config.data_dir, "validation"),
    )


def _device(config: RunConfig) -> torch.device:
    requested = config.training.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError("requested CUDA device index is unavailable")
        with torch.cuda.device(device):
            if config.training.precision == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("bf16 is not supported on the selected CUDA device")
    elif config.training.precision != "fp32":
        raise ValueError("CPU training requires precision='fp32'")
    return device


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    return {
        "implementation": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "training.py",
                "checkpoint.py",
                "model.py",
                "data.py",
                "tokenizer.py",
                "config.py",
            )
        },
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "device": str(device),
        "cuda_version": torch.version.cuda,
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_build": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_hardware": [
            {
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
        "sdpa": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
    }


@contextlib.contextmanager
def _determinism(enabled: bool) -> Iterator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(enabled)
        if enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(config: RunConfig, device: torch.device) -> Any:
    if config.training.precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if config.training.precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def learning_rate_factor(index: int, *, max_steps: int, warmup_steps: int, minimum: float) -> float:
    """Multiplier for zero-based optimizer update index; last decay update reaches minimum."""
    if index < warmup_steps:
        return (index + 1) / warmup_steps
    decay_steps = max_steps - warmup_steps
    progress = min(1.0, max(0.0, (index - warmup_steps) / max(1, decay_steps - 1)))
    return minimum + (1.0 - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))


class _DataCursor:
    """Checkpointed epoch permutation and position, independent of model RNG."""

    def __init__(self, size: int, seed: int, state: dict[str, Any] | None = None):
        self.size, self.seed = size, seed
        if state is None:
            self.epoch, self.position = 0, 0
            self.order = self._order()
        else:
            self.epoch, self.position, self.order = (
                state["epoch"],
                state["position"],
                state["order"],
            )
            if (
                type(self.epoch) is not int
                or self.epoch < 0
                or type(self.position) is not int
                or not 0 <= self.position <= size
                or not isinstance(self.order, list)
                or any(type(index) is not int for index in self.order)
                or sorted(self.order) != list(range(size))
            ):
                raise ValueError("invalid checkpoint data cursor")

    def _order(self) -> list[int]:
        generator = torch.Generator().manual_seed((self.seed + self.epoch) % (2**63))
        return torch.randperm(self.size, generator=generator).tolist()

    def group(self, batch_size: int, accumulation: int) -> list[list[int]]:
        if self.position == self.size:
            self.epoch += 1
            self.position = 0
            self.order = self._order()
        batches = []
        for _ in range(accumulation):
            if self.position == self.size:
                break
            end = min(self.size, self.position + batch_size)
            batches.append(self.order[self.position : end])
            self.position = end
        return batches

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "position": self.position, "order": list(self.order)}


def _batch(dataset: PackedDataset, indices: list[int], device: torch.device) -> dict[str, Tensor]:
    rows = [dataset[index] for index in indices]
    return {
        key: torch.stack([row[key] for row in rows]).to(device) for key in ("input_ids", "labels")
    }


def _validation(
    model: CausalLM, dataset: PackedDataset, config: RunConfig, device: torch.device
) -> dict[str, float | int]:
    """Token-weighted evaluation that leaves training mode and all RNGs unchanged."""
    rng = capture_rng_state()
    was_training = model.training
    total_loss, tokens, batches = 0.0, 0, 0
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(dataset), config.training.batch_size):
                if batches >= config.training.eval_batches:
                    break
                indices = list(range(start, min(start + config.training.batch_size, len(dataset))))
                batch = _batch(dataset, indices, device)
                count = int((batch["labels"][:, 1:] != -100).sum().item())
                if count == 0:
                    raise ValueError("all-ignored validation supervision")
                with _autocast(config, device):
                    loss = model(**batch).loss
                if loss is None or not torch.isfinite(loss).item():
                    raise FloatingPointError("nonfinite validation loss")
                total_loss += float(loss.item()) * count
                tokens += count
                batches += 1
        if tokens == 0:
            raise ValueError("validation contains no supervised tokens")
        mean = total_loss / tokens
        return {
            "loss": mean,
            "perplexity": math.exp(mean) if mean < 709 else None,
            "perplexity_overflow": mean >= 709,
            "tokens": tokens,
            "batches": batches,
        }
    finally:
        model.train(was_training)
        restore_rng_state(rng)


@contextlib.contextmanager
def _stop_signals() -> Iterator[dict[str, bool]]:
    """Defer SIGINT/SIGTERM until the current optimizer update is durable."""
    state = {"requested": False}
    previous: dict[int, Any] = {}

    def request_stop(signum: int, frame: Any) -> None:
        state["requested"] = True

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, request_stop)
    try:
        yield state
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _read_resume_metrics(output: Path, payload: dict[str, Any]) -> bytes:
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != payload["run_id"]:
        raise ValueError("checkpoint belongs to a different output run")
    metrics = (output / "metrics.jsonl").read_bytes()
    length = payload["metrics"]["bytes"]
    if type(length) is not int or length < 0 or len(metrics) < length:
        raise ValueError("metrics journal is shorter than its checkpoint")
    prefix = metrics[:length]
    if hashlib.sha256(prefix).hexdigest() != payload["metrics"]["sha256"]:
        raise ValueError("metrics journal does not match checkpoint")
    return prefix


@contextlib.contextmanager
def _run_lock(output: Path) -> Iterator[None]:
    """Reject a concurrent writer, including a second process resuming this run."""
    with (output / ".run.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another process owns this training run") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validate_loaded_state(
    model: CausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    cursor: _DataCursor,
    config: RunConfig,
    step: int,
    scaler: torch.amp.GradScaler,
) -> None:
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all().item():
            raise ValueError("checkpoint contains nonfinite model weights")
    if scheduler.last_epoch != step:
        raise ValueError("checkpoint scheduler does not match optimizer step")
    schedule = scheduler.state_dict()
    if (
        schedule["base_lrs"] != [config.training.learning_rate]
        or schedule["_step_count"] != step + 1
        or schedule["_last_lr"] != [g["lr"] for g in optimizer.param_groups]
    ):
        raise ValueError("checkpoint scheduler state is inconsistent")
    scale = scaler.state_dict()
    if scale and (
        not math.isfinite(scale["scale"])
        or scale["scale"] <= 0
        or scale["growth_factor"] != 2.0
        or scale["backoff_factor"] != 0.5
        or scale["growth_interval"] != 2000
        or not 0 <= scale["_growth_tracker"] < 2000
    ):
        raise ValueError("checkpoint gradient scaler state is invalid")
    if cursor.order != cursor._order():
        raise ValueError("checkpoint data permutation does not match its epoch and seed")
    stride = config.training.batch_size * config.training.accumulation_steps
    updates_per_epoch = math.ceil(cursor.size / stride)
    expected_epoch = (step - 1) // updates_per_epoch if step else 0
    expected_position = (
        min(cursor.size, ((step - 1) % updates_per_epoch + 1) * stride) if step else 0
    )
    if (cursor.epoch, cursor.position) != (expected_epoch, expected_position):
        raise ValueError("checkpoint cursor does not match optimizer progress")
    expected_lr = config.training.learning_rate * learning_rate_factor(
        step,
        max_steps=config.training.max_steps,
        warmup_steps=config.training.warmup_steps,
        minimum=config.training.min_lr_ratio,
    )
    for group in optimizer.param_groups:
        if group["lr"] != expected_lr or group["weight_decay"] != config.training.weight_decay:
            raise ValueError("checkpoint optimizer hyperparameters do not match configuration")
        if group["betas"] != (0.9, 0.999) or group["eps"] != 1e-8 or group["amsgrad"]:
            raise ValueError("checkpoint AdamW settings do not match this trainer")
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if step == 0:
                if state:
                    raise ValueError("initial checkpoint unexpectedly contains optimizer moments")
                continue
            if not state or float(state["step"]) != step:
                raise ValueError("checkpoint AdamW step does not match optimizer progress")
            for name in ("exp_avg", "exp_avg_sq"):
                moment = state[name]
                if moment.shape != parameter.shape or not torch.isfinite(moment).all().item():
                    raise ValueError("checkpoint optimizer moment shape or values are invalid")
                if name == "exp_avg_sq" and (moment < 0).any().item():
                    raise ValueError("checkpoint second moments must be nonnegative")


def train(
    config: RunConfig, *, resume: str | Path | None = None, stop_after_steps: int | None = None
) -> TrainResult:
    """Train or exactly resume a native decoder run.

    ``stop_after_steps`` is an absolute optimizer boundary for a controlled
    pause; it does not alter the configured learning-rate horizon. Changing
    configuration, data, tokenizer, or runtime identity requires a new run.
    """
    if stop_after_steps is not None and (
        type(stop_after_steps) is not int or not 0 <= stop_after_steps <= config.training.max_steps
    ):
        raise ValueError("stop_after_steps must be an optimizer boundary within max_steps")
    if config.training.seed >= 2**63:
        raise ValueError("seed must be less than 2**63")
    if config.training.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = _device(config)
    inputs = _inputs(config)
    identities = {**inputs.identities, "runtime": _runtime_identity(device)}
    output = config.output_dir
    payload, loaded_hash = load_checkpoint_with_hash(resume) if resume is not None else (None, None)
    metrics_prefix = b""
    initial = None
    initial_hash = None
    if payload is not None:
        if payload["config"] != config.to_dict():
            raise ValueError("resume configuration differs from checkpoint")
        if payload["identities"] != identities:
            raise ValueError("resume dataset, tokenizer, or runtime identity differs")
        if payload["step"] > config.training.max_steps:
            raise ValueError("checkpoint exceeds configured training horizon")
        metrics_prefix = _read_resume_metrics(output, payload)
    else:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite an existing run directory: {output}")
        if config.stage == "sft":
            if config.init_checkpoint is None:
                raise ValueError("a fresh SFT run requires init_checkpoint")
            initial, initial_hash = load_checkpoint_with_hash(config.init_checkpoint)
            if initial["config"]["stage"] != "pretrain":
                raise ValueError("SFT initialization requires a native pretraining checkpoint")
            if initial["config"]["model"] != dataclasses.asdict(config.model):
                raise ValueError("initial checkpoint model configuration differs")
            if initial["identities"]["tokenizer_sha256"] != identities["tokenizer_sha256"]:
                raise ValueError("initial checkpoint tokenizer identity differs")
        elif config.init_checkpoint is not None:
            raise ValueError("init_checkpoint is for a new SFT run; use resume for pretraining")
    with _determinism(config.training.deterministic), contextlib.ExitStack() as locks:
        if payload:
            locks.enter_context(_run_lock(output))
            canonical_checkpoint = output / "checkpoints" / f"step-{payload['step']:08d}.pt"
            if (
                not canonical_checkpoint.is_file()
                or file_sha256(canonical_checkpoint) != loaded_hash
            ):
                raise ValueError("resume checkpoint does not match its owned immutable step file")
            metrics_prefix = _read_resume_metrics(output, payload)
        _seed(config.training.seed)
        model = CausalLM(config.model).to(device)
        if payload is not None or initial is not None:
            model.load_state_dict((payload or initial)["model"], strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda index: learning_rate_factor(
                index,
                max_steps=config.training.max_steps,
                warmup_steps=config.training.warmup_steps,
                minimum=config.training.min_lr_ratio,
            ),
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.training.precision == "fp16")
        cursor = _DataCursor(
            len(inputs.train), config.training.seed, payload["data_cursor"] if payload else None
        )
        step = payload["step"] if payload else 0
        if stop_after_steps is not None and stop_after_steps < step:
            raise ValueError("stop_after_steps precedes the resumed optimizer step")
        train_loss = payload.get("train_loss") if payload else None
        validation_loss = payload.get("validation_loss") if payload else None
        if payload:
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            scaler.load_state_dict(payload["scaler"])
            restore_rng_state(payload["rng"])
            _validate_loaded_state(model, optimizer, scheduler, cursor, config, step, scaler)
            run_id = payload["run_id"]
        else:
            run_id = uuid.uuid4().hex
        if not payload:
            _validate_loaded_state(model, optimizer, scheduler, cursor, config, step, scaler)
        # All compatibility checks and state loading precede output mutations.
        if payload is None:
            output.mkdir(parents=True, exist_ok=False)
            locks.enter_context(_run_lock(output))
            (output / "checkpoints").mkdir()
            run_manifest = {
                "run_id": run_id,
                "config": config.to_dict(),
                "identities": identities,
                "initial_checkpoint_sha256": initial_hash,
            }
            atomic_write_bytes(_json_bytes(run_manifest), output / "run.json")
        metrics_path = output / "metrics.jsonl"
        atomic_write_bytes(metrics_prefix, metrics_path)
        last_saved = step if payload else -1
        checkpoint = canonical_checkpoint if payload else output / "latest.pt"
        if payload:
            publish_latest(checkpoint, output / "latest.pt")

        def save() -> None:
            nonlocal last_saved, checkpoint
            if last_saved == step:
                return
            journal = metrics_path.read_bytes()
            checkpoint = output / "checkpoints" / f"step-{step:08d}.pt"
            state = {
                "format_version": FORMAT_VERSION,
                "run_id": run_id,
                "config": config.to_dict(),
                "identities": identities,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "rng": capture_rng_state(),
                "data_cursor": cursor.state_dict(),
                "metrics": {"bytes": len(journal), "sha256": hashlib.sha256(journal).hexdigest()},
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
            atomic_save(state, checkpoint)
            publish_latest(checkpoint, output / "latest.pt")
            last_saved = step
            owned = sorted((output / "checkpoints").glob("step-????????.pt"))
            for old in owned[: -config.training.keep_checkpoints]:
                if old != checkpoint and old.stem[5:].isdigit():
                    old.unlink()

        with _stop_signals() as stop:
            save()
            model.train()
            while step < config.training.max_steps:
                if stop["requested"] or (stop_after_steps is not None and step >= stop_after_steps):
                    break
                index_groups = cursor.group(
                    config.training.batch_size, config.training.accumulation_steps
                )
                # Gather on CPU first: divide by actual supervised tokens across
                # the whole update, not by nominal microbatch or sequence counts.
                batches = [
                    _batch(inputs.train, indices, torch.device("cpu")) for indices in index_groups
                ]
                counts = [int((batch["labels"][:, 1:] != -100).sum()) for batch in batches]
                total_tokens = sum(counts)
                if not total_tokens or any(count == 0 for count in counts):
                    raise ValueError("all-ignored training supervision")
                optimizer.zero_grad(set_to_none=True)
                weighted_loss = 0.0
                update_lr = optimizer.param_groups[0]["lr"]
                for cpu_batch, count in zip(batches, counts, strict=True):
                    batch = {key: value.to(device) for key, value in cpu_batch.items()}
                    with _autocast(config, device):
                        loss = model(**batch).loss
                    if loss is None or not torch.isfinite(loss).item():
                        raise FloatingPointError("nonfinite training loss")
                    weighted_loss += float(loss.detach().item()) * count / total_tokens
                    scaler.scale(loss * (count / total_tokens)).backward()
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.grad_clip, error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                train_loss = weighted_loss
                metric: dict[str, Any] = {
                    "step": step,
                    "epoch": cursor.epoch,
                    "cursor": cursor.position,
                    "tokens": total_tokens,
                    "train_loss": train_loss,
                    "learning_rate": update_lr,
                    "gradient_norm": float(norm.item()),
                }
                if step % config.training.eval_every == 0 or step == config.training.max_steps:
                    evaluation = _validation(model, inputs.validation, config, device)
                    validation_loss = float(evaluation["loss"])
                    metric["validation"] = evaluation
                with metrics_path.open("ab") as stream:
                    stream.write(_json_bytes(metric))
                    stream.flush()
                    os.fsync(stream.fileno())
                if step % config.training.save_every == 0:
                    save()
            save()
            status = (
                "complete"
                if step == config.training.max_steps
                else ("interrupted" if stop["requested"] else "paused")
            )
        return TrainResult(checkpoint, status, step, train_loss, validation_loss)


def evaluate(config: RunConfig, checkpoint: str | Path) -> dict[str, float | int]:
    """Evaluate a trusted native checkpoint against this config's validation data."""
    rng = capture_rng_state()
    try:
        device = _device(config)
        inputs = _inputs(config)
        payload = load_checkpoint(checkpoint)
        if payload["config"]["model"] != dataclasses.asdict(config.model):
            raise ValueError("evaluation model configuration differs from checkpoint")
        if payload["identities"]["tokenizer_sha256"] != inputs.identities["tokenizer_sha256"]:
            raise ValueError("evaluation tokenizer differs from checkpoint")
        with _determinism(config.training.deterministic):
            model = CausalLM(config.model).to(device)
            model.load_state_dict(payload["model"], strict=True)
            return _validation(model, inputs.validation, config, device)
    finally:
        restore_rng_state(rng)


def export(config: RunConfig, checkpoint: str | Path, output_dir: str | Path) -> Path:
    """Publish a portable native bundle atomically; refuse an existing destination.

    The bundle contains weights (loadable with ``weights_only=True``), model
    configuration, tokenizer, and hashes. It is not an ONNX or HF conversion.
    """
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"export destination already exists: {output}")
    payload, checkpoint_hash = load_checkpoint_with_hash(checkpoint)
    if payload["config"]["model"] != dataclasses.asdict(config.model):
        raise ValueError("export model configuration differs from checkpoint")
    tokenizer_bytes = config.tokenizer_path.read_bytes()
    tokenizer_hash = hashlib.sha256(tokenizer_bytes).hexdigest()
    if tokenizer_hash != payload["identities"]["tokenizer_sha256"]:
        raise ValueError("export tokenizer differs from checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        atomic_save(payload["model"], staging / "model.pt")
        atomic_write_bytes(_json_bytes(payload["config"]["model"]), staging / "model.json")
        atomic_write_bytes(tokenizer_bytes, staging / "tokenizer.json")
        provenance = {
            "format": "clearforge-native-v1",
            "step": payload["step"],
            "checkpoint_sha256": checkpoint_hash,
            "run_id": payload["run_id"],
            "identities": payload["identities"],
            "files": {
                name: file_sha256(staging / name)
                for name in ("model.pt", "model.json", "tokenizer.json")
            },
        }
        atomic_write_bytes(_json_bytes(provenance), staging / "provenance.json")
        # A second existence check avoids silently replacing a destination
        # created while the bundle was being serialized.
        if output.exists():
            raise FileExistsError(f"export destination already exists: {output}")
        publish_directory(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output
