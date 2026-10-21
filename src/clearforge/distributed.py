"""Synchronous native DDP training launched through torchrun.

A shared deterministic global cursor is partitioned across ranks. Empty ranks in
an epoch's tail run a zero-weight dummy microbatch so collective counts match.
Only rank zero owns output files; checkpoints retain each rank's random state.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import hashlib
import math
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import (
    FORMAT_VERSION,
    atomic_save,
    atomic_write_bytes,
    capture_rng_state,
    file_sha256,
    load_checkpoint_with_hash,
    publish_latest,
    restore_rng_state,
)
from .config import ModelConfig, RunConfig
from .model import CausalLM
from .training import (
    TrainResult,
    _autocast,
    _batch,
    _compile_execution,
    _DataCursor,
    _determinism,
    _enable_activation_checkpointing,
    _inputs,
    _json_bytes,
    _memory_guidance,
    _preflight,
    _progress_record,
    _read_resume_metrics,
    _run_lock,
    _runtime_identity,
    _seed,
    _stop_signals,
    _validate_loaded_state,
    _validation,
    build_optimizer,
    schedule_factor,
)


def _agree(function):
    """Propagate rank-local setup/IO errors before entering subsequent collectives."""
    value, error = None, None
    try:
        value = function()
    except Exception as exc:
        error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError("distributed operation failed: " + "; ".join(e for e in errors if e))
    return value


def _gather(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def train_distributed(
    config: RunConfig, *, resume: str | Path | None = None, stop_after_steps: int | None = None
) -> TrainResult:
    """Run synchronous CPU/Gloo or CUDA/NCCL training in every torchrun process.

    ``batch_size`` is per rank; global updates contain batch_size × world_size ×
    accumulation_steps sequences except epoch tails. Resume requires unchanged
    world size, rank mapping, runtime, data and configuration. Process-group
    failures are bounded by a timeout; torchrun tears down failed worker groups.
    """
    world = (
        int(os.environ.get("WORLD_SIZE", "1"))
        if not dist.is_initialized()
        else dist.get_world_size()
    )
    if world < 2:
        raise ValueError("distributed training requires at least two ranks")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda = config.training.device != "cpu" and torch.cuda.is_available()
    if config.training.device.startswith("cuda") and not cuda:
        raise ValueError("CUDA requested but unavailable")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group("nccl" if cuda else "gloo", timeout=datetime.timedelta(seconds=120))
    try:
        return _train(config, device, resume, stop_after_steps)
    finally:
        if owned_group:
            dist.destroy_process_group()


def _train(config, device, resume, stop_after_steps):
    rank, world = dist.get_rank(), dist.get_world_size()

    def inspect():
        if device.type == "cpu" and config.training.precision != "fp32":
            raise ValueError("CPU distributed training requires fp32")
        if (
            device.type == "cuda"
            and config.training.precision == "bf16"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("device does not support bf16")
        if config.training.seed >= 2**63:
            raise ValueError("seed must be less than 2**63")
        return _inputs(config)

    inputs = _agree(inspect)
    if config.training.epochs is not None:
        horizon = (
            math.ceil(
                len(inputs.train)
                / (config.training.batch_size * world * config.training.accumulation_steps)
            )
            * config.training.epochs
        )
        config = dataclasses.replace(
            config, training=dataclasses.replace(config.training, max_steps=horizon)
        )

    def boundary():
        if stop_after_steps is not None and (
            type(stop_after_steps) is not int
            or not 0 <= stop_after_steps <= config.training.max_steps
        ):
            raise ValueError("invalid optimizer pause boundary")

    _agree(boundary)
    signatures = _gather(
        {"config": config.to_dict(), "inputs": inputs.identities, "stop": stop_after_steps}
    )
    if any(value != signatures[0] for value in signatures):
        raise ValueError("rank configurations or dataset identities differ")
    identities = {
        **inputs.identities,
        "distributed": {
            "world_size": world,
            "backend": dist.get_backend(),
            "runtime_by_rank": _gather(_runtime_identity(device)),
            "implementation_sha256": file_sha256(__file__),
        },
    }
    output = config.output_dir
    with contextlib.ExitStack() as stack, _determinism(config.training.deterministic):
        if resume:
            _agree(lambda: stack.enter_context(_run_lock(output)) if rank == 0 else None)

        def preflight():
            payload, digest = load_checkpoint_with_hash(resume) if resume else (None, None)
            initial, initial_hash = None, None
            if payload:
                if payload["config"] != config.to_dict() or payload["identities"] != identities:
                    raise ValueError(
                        "distributed resume configuration, runtime, or world size differs"
                    )
                if not 0 <= payload["step"] <= config.training.max_steps:
                    raise ValueError("checkpoint exceeds configured horizon")
                if len(payload["rng_by_rank"]) != world:
                    raise ValueError("checkpoint rank RNG topology differs")
                canonical = output / "checkpoints" / f"step-{payload['step']:08d}.pt"
                if not canonical.is_file() or file_sha256(canonical) != digest:
                    raise ValueError("resume checkpoint differs from its owned step file")
                prefix = _read_resume_metrics(output, payload)
            else:
                prefix = b""
                if output.exists():
                    raise FileExistsError(f"output directory already exists: {output}")
                if config.stage == "sft" and config.init_checkpoint is None:
                    raise ValueError("SFT requires init_checkpoint")
                if config.init_checkpoint:
                    initial, initial_hash = load_checkpoint_with_hash(config.init_checkpoint)
                    if ModelConfig(**initial["config"]["model"]) != config.model:
                        raise ValueError("initial model configuration differs")
                    if (
                        initial["identities"]["tokenizer_sha256"]
                        != inputs.identities["tokenizer_sha256"]
                    ):
                        raise ValueError("initial tokenizer differs")
            return payload, initial, initial_hash, prefix, digest

        payload, initial, initial_hash, prefix, loaded_digest = _agree(preflight)
        checkpoint_identities = _gather((loaded_digest, initial_hash))
        if any(value != checkpoint_identities[0] for value in checkpoint_identities):
            raise ValueError("ranks loaded different checkpoint bytes")

        def setup():
            _seed(config.training.seed)
            model = CausalLM(config.model).to(device)
            if payload or initial:
                model.load_state_dict((payload or initial)["model"], strict=True)
            if config.training.activation_checkpointing:
                _enable_activation_checkpointing(model)
            optimizer = build_optimizer(model, config.training)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda step: schedule_factor(config.training, step)
            )
            scaler = torch.amp.GradScaler("cuda", enabled=config.training.precision == "fp16")
            cursor = _DataCursor(
                len(inputs.train), config.training.seed, payload["data_cursor"] if payload else None
            )
            step = payload["step"] if payload else 0
            if stop_after_steps is not None and stop_after_steps < step:
                raise ValueError("pause precedes resumed optimizer boundary")
            if payload:
                optimizer.load_state_dict(payload["optimizer"])
                scheduler.load_state_dict(payload["scheduler"])
                scaler.load_state_dict(payload["scaler"])
            global_config = dataclasses.replace(
                config,
                training=dataclasses.replace(
                    config.training, batch_size=config.training.batch_size * world
                ),
            )
            _validate_loaded_state(model, optimizer, scheduler, cursor, global_config, step, scaler)
            return model, optimizer, scheduler, scaler, cursor, step

        model, optimizer, scheduler, scaler, cursor, step = _agree(setup)
        ddp = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=config.model.num_experts > 1,
        )
        execution = _compile_execution(ddp, config)
        # Separate reproducible dropout streams; restore only after DDP construction.
        if payload:
            _agree(lambda: restore_rng_state(payload["rng_by_rank"][rank]))
        else:
            _seed((config.training.seed + rank) % 2**63)
        run_ids = _gather(
            payload["run_id"] if payload else (uuid.uuid4().hex if rank == 0 else None)
        )
        run_id = run_ids[0]
        # Rank zero owns the output volume, so it alone plans and checks disk space.
        plan = _agree(lambda: _preflight(config, device, world_size=world) if rank == 0 else None)

        def create_output():
            if rank != 0:
                return
            if not payload:
                output.mkdir(parents=True, exist_ok=False)
                stack.enter_context(_run_lock(output))
                (output / "checkpoints").mkdir()
                atomic_write_bytes(
                    _json_bytes(
                        {
                            "run_id": run_id,
                            "config": config.to_dict(),
                            "identities": identities,
                            "initial_checkpoint_sha256": initial_hash,
                            "plan": plan,
                        }
                    ),
                    output / "run.json",
                )
            atomic_write_bytes(prefix, output / "metrics.jsonl")

        _agree(create_output)
        executor = (
            stack.enter_context(ThreadPoolExecutor(max_workers=config.training.data_workers))
            if config.training.data_workers
            else None
        )
        train_loss = payload.get("train_loss") if payload else None
        validation_loss = payload.get("validation_loss") if payload else None
        checkpoint = output / "checkpoints" / f"step-{step:08d}.pt"
        last_saved = step if payload else -1

        def save():
            nonlocal checkpoint, last_saved
            if last_saved == step:
                return
            rank_rng = _gather(capture_rng_state())
            checkpoint = output / "checkpoints" / f"step-{step:08d}.pt"

            def write():
                if rank != 0:
                    return
                journal = (output / "metrics.jsonl").read_bytes()
                atomic_save(
                    {
                        "format_version": FORMAT_VERSION,
                        "run_id": run_id,
                        "config": config.to_dict(),
                        "identities": identities,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "rng": rank_rng[0],
                        "rng_by_rank": rank_rng,
                        "data_cursor": cursor.state_dict(),
                        "metrics": {
                            "bytes": len(journal),
                            "sha256": hashlib.sha256(journal).hexdigest(),
                        },
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                    },
                    checkpoint,
                )
                publish_latest(checkpoint, output / "latest.pt")
                owned = sorted((output / "checkpoints").glob("step-????????.pt"))
                for old in owned[: -config.training.keep_checkpoints]:
                    if old != checkpoint and old.stem[5:].isdigit():
                        old.unlink()

            _agree(write)
            last_saved = step

        with _stop_signals() as stop, _memory_guidance(plan):
            save()
            if payload:
                _agree(
                    lambda: publish_latest(checkpoint, output / "latest.pt") if rank == 0 else None
                )
            model.train()
            started, initial_step, token_sum = time.monotonic(), step, 0
            parameter_total = model.parameter_count()
            while step < config.training.max_steps:
                stop_tensor = torch.tensor(
                    int(
                        stop["requested"]
                        or (stop_after_steps is not None and step >= stop_after_steps)
                    ),
                    device=device,
                )
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                if stop_tensor.item():
                    break
                global_groups = cursor.group(
                    config.training.batch_size * world, config.training.accumulation_steps
                )

                def load_batches():
                    rank_groups = [
                        group[
                            rank * config.training.batch_size : (rank + 1)
                            * config.training.batch_size
                        ]
                        for group in global_groups
                    ]
                    if executor:
                        futures = [
                            executor.submit(
                                _batch,
                                inputs.train,
                                group or [global_groups[i][0]],
                                torch.device("cpu"),
                            )
                            for i, group in enumerate(rank_groups)
                        ]
                        batches = [future.result() for future in futures]
                    else:
                        batches = [
                            _batch(
                                inputs.train, group or [global_groups[i][0]], torch.device("cpu")
                            )
                            for i, group in enumerate(rank_groups)
                        ]
                    counts = [
                        int((batch["labels"][:, 1:] != -100).sum()) if group else 0
                        for group, batch in zip(rank_groups, batches)
                    ]
                    if config.training.pin_memory and device.type == "cuda":
                        batches = [
                            {key: value.pin_memory() for key, value in batch.items()}
                            for batch in batches
                        ]
                    return batches, counts

                batches, counts = _agree(load_batches)
                total = torch.tensor(sum(counts), dtype=torch.long, device=device)
                dist.all_reduce(total)
                total_tokens = int(total.item())
                optimizer.zero_grad(set_to_none=True)
                weighted_loss = 0.0
                update_lr = optimizer.param_groups[0]["lr"]
                for cpu_batch, count in zip(batches, counts):
                    batch = {
                        key: value.to(device, non_blocking=config.training.pin_memory)
                        for key, value in cpu_batch.items()
                    }
                    with _autocast(config, device):
                        loss = execution(**batch).loss
                    finite = torch.tensor(
                        int(loss is not None and torch.isfinite(loss).item()), device=device
                    )
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise FloatingPointError("nonfinite loss on a distributed rank")
                    weighted_loss += float(loss.detach()) * count / total_tokens
                    # DDP averages gradients; compensate to obtain the global token mean.
                    scaler.scale(loss * (world * count / total_tokens)).backward()
                scaler.unscale_(optimizer)
                norm = _agree(
                    lambda: torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.training.grad_clip, error_if_nonfinite=True
                    )
                )
                _agree(lambda: scaler.step(optimizer))
                scaler.update()
                scheduler.step()
                step += 1
                loss_sum = torch.tensor(weighted_loss, dtype=torch.float64, device=device)
                dist.all_reduce(loss_sum)
                train_loss = float(loss_sum.item())
                metric = {
                    "step": step,
                    "epoch": cursor.epoch,
                    "cursor": cursor.position,
                    "tokens": total_tokens,
                    "train_loss": train_loss,
                    "learning_rate": update_lr,
                    "gradient_norm": float(norm),
                }
                if step % config.training.eval_every == 0 or step == config.training.max_steps:
                    evaluation = _agree(
                        lambda: _validation(model, inputs.validation, config, device)
                        if rank == 0
                        else None
                    )
                    evaluation = _gather(evaluation)[0]
                    validation_loss = float(evaluation["loss"])
                    metric["validation"] = evaluation

                def journal():
                    if rank == 0:
                        with (output / "metrics.jsonl").open("ab") as stream:
                            stream.write(_json_bytes(metric))
                            stream.flush()
                            os.fsync(stream.fileno())

                _agree(journal)
                if step % config.training.save_every == 0:
                    save()
                token_sum += total_tokens
                if (
                    rank == 0
                    and config.training.log_every
                    and step % config.training.log_every == 0
                ):
                    elapsed = max(time.monotonic() - started, 1e-9)
                    progress = _progress_record(
                        step=step,
                        config=config,
                        loss=train_loss,
                        learning_rate=update_lr,
                        gradient_norm=float(norm),
                        epoch=cursor.epoch,
                        tokens_per_second=token_sum / elapsed,
                        eta_seconds=(config.training.max_steps - step)
                        * elapsed
                        / (step - initial_step),
                        parameters=parameter_total,
                        device=device,
                    )
                    progress["world_size"] = world
                    print(_json_bytes(progress).decode().strip(), file=sys.stderr, flush=True)
            save()
            interrupted = torch.tensor(int(stop["requested"]), device=device)
            dist.all_reduce(interrupted, op=dist.ReduceOp.MAX)
            status = (
                "complete"
                if step == config.training.max_steps
                else ("interrupted" if interrupted.item() else "paused")
            )
        return TrainResult(checkpoint, status, step, train_loss, validation_loss)
