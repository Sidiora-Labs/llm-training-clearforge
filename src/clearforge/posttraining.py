"""Native supervised, preference, distillation and online policy training drivers."""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from .adapters import AdapterConfig, _fingerprint, apply_adapters, merge_adapters
from .checkpoint import atomic_save, atomic_write_bytes, capture_rng_state, restore_rng_state
from .data import adapt_preference, adapt_rollout, atomic_directory, validate_messages
from .model import CausalLM
from .objectives import (
    clipped_value_loss,
    distillation_loss,
    dpo_loss,
    generalized_advantages,
    group_advantages,
    policy_loss,
    sequence_log_probabilities,
    token_log_probabilities,
)
from .tokenizer import ChatTokenizer
from .tools import serialize_messages
from .training import _DataCursor, _determinism, _run_lock, _runtime_identity, _seed, _stop_signals


@dataclasses.dataclass(frozen=True)
class PostTrainConfig:
    output_path: Path
    steps: int = 100
    batch_size: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cpu"
    beta: float = 0.1
    temperature: float = 2.0
    distillation_alpha: float = 0.5
    samples_per_prompt: int = 2
    max_new_tokens: int = 64
    sampling_temperature: float = 1.0
    policy_epochs: int = 2
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    gamma: float = 1.0
    gae_lambda: float = 0.95

    def __post_init__(self):
        object.__setattr__(self, "output_path", Path(self.output_path).resolve())
        for name in (
            "steps",
            "batch_size",
            "samples_per_prompt",
            "max_new_tokens",
            "policy_epochs",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        for name in ("learning_rate", "grad_clip", "temperature", "sampling_temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "beta", "value_coefficient"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("distillation_alpha", "gamma", "gae_lambda"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in [0, 1)")


@dataclasses.dataclass
class PostTrainResult:
    model: CausalLM
    checkpoint: Path
    step: int
    status: str


class ValueModel(nn.Module):
    """Independent native causal backbone and a trainable scalar value head."""

    def __init__(self, base: CausalLM):
        super().__init__()
        self.embedding = copy.deepcopy(base.token_embedding)
        self.layers = copy.deepcopy(base.layers)
        self.norm = copy.deepcopy(base.norm)
        self.head = nn.Linear(base.config.hidden_size, 1)
        self.requires_grad_(True)

    def forward(self, ids):
        hidden = self.embedding(ids)
        for layer in self.layers:
            result = layer(hidden)
            hidden = result[0] if isinstance(result, tuple) else result
        return self.head(self.norm(hidden)).squeeze(-1)


def _tokenizer_identity(tokenizer):
    return hashlib.sha256(tokenizer._tokenizer.to_str().encode()).hexdigest()


def _records(data):
    if isinstance(data, (str, Path)):
        with Path(data).open() as source:
            rows = [json.loads(line) for line in source if line.strip()]
    else:
        rows = copy.deepcopy(list(data))
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("data must contain nonempty JSON objects")
    return rows


def _prompt(row):
    # Preserve schemas as part of the prompt object through online generation.
    fields = {
        k: v for k, v in row.items() if k in {"prompt", "messages", "tools", "expected", "gt"}
    }
    return adapt_rollout(fields)


def _encode(row, tokenizer, objective):
    tools = row.get("tools")
    if objective in {"sft", "lora", "distillation"}:
        return tokenizer.encode_chat(
            validate_messages(row.get("messages"), tools=tools), tools=tools
        )
    if objective == "dpo":
        fields = {k: v for k, v in row.items() if k in {"prompt", "chosen", "rejected", "tools"}}
        pair = adapt_preference(fields)
        prompt = pair["chosen"][:-1]
        prefix, _ = tokenizer.encode_chat(prompt, tools=tools, add_generation_prompt=True)

        def response(messages):
            # Serialize the full final assistant, including structured calls/reasoning.
            # Historical assistant turns remain context-only for preference scoring.
            text = serialize_messages(messages, tools=tools)[-1][1]
            completion = tokenizer.encode(text) + [tokenizer.eos_id]
            return prefix + completion, [-100] * len(prefix) + completion

        return response(pair["chosen"]), response(pair["rejected"])
    return _prompt(row)


def _batch(examples, tokenizer, device):
    width = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), width), tokenizer.pad_id, dtype=torch.long, device=device)
    labels = torch.full_like(ids, -100)
    for index, (tokens, targets) in enumerate(examples):
        ids[index, : len(tokens)] = torch.tensor(tokens, device=device)
        labels[index, : len(targets)] = torch.tensor(targets, device=device)
    return ids, labels


def _frozen(model, device):
    return copy.deepcopy(model).to(device).eval().requires_grad_(False)


def _score(model, ids, labels):
    return sequence_log_probabilities(model(ids).logits, labels)


def _rollout(model, reference, critic, prompts, tokenizer, cfg, reward_fn, objective):
    device = next(model.parameters()).device
    examples, rewards, starts, groups = [], [], [], []
    model.eval()  # Importance ratios must not compare independent dropout masks.
    with torch.no_grad():
        for prompt in prompts:
            prefix, _ = tokenizer.encode_chat(
                prompt["messages"], tools=prompt.get("tools"), add_generation_prompt=True
            )
            if len(prefix) + cfg.max_new_tokens > model.config.max_seq_len:
                raise ValueError("prompt plus rollout budget exceeds model context")
            group = []
            for _ in range(cfg.samples_per_prompt):
                generated = model.generate(
                    torch.tensor([prefix], device=device),
                    max_new_tokens=cfg.max_new_tokens,
                    temperature=cfg.sampling_temperature,
                    eos_token_id=tokenizer.eos_id,
                )[0].tolist()
                completion = generated[len(prefix) :]
                if not completion:
                    raise ValueError("rollout contains no completion tokens")
                reward = float(
                    reward_fn(
                        prompt["messages"], tokenizer.decode(completion, skip_special_tokens=True)
                    )
                )
                if not math.isfinite(reward):
                    raise ValueError("reward callback returned a nonfinite value")
                examples.append((generated, [-100] * len(prefix) + completion))
                starts.append(len(prefix) - 1)
                rewards.append(reward)
                group.append(reward)
            groups.append(group)
        ids, labels = _batch(examples, tokenizer, device)
        old, mask = token_log_probabilities(model(ids).logits / cfg.sampling_temperature, labels)
        ref, _ = token_log_probabilities(reference(ids).logits / cfg.sampling_temperature, labels)
        if objective == "ppo":
            lengths = mask.sum(-1).tolist()
            width = max(lengths)
            completion_mask = (
                torch.arange(width, device=device)[None]
                < torch.tensor(lengths, device=device)[:, None]
            )
            old_values_full = critic(ids)[:, :-1]
            old_values = torch.zeros((len(examples), width), device=device)
            terminal_rewards = torch.zeros_like(old_values)
            for i, (start, length) in enumerate(zip(starts, lengths)):
                old_values[i, :length] = old_values_full[i, start : start + length]
                terminal_rewards[i, length - 1] = rewards[i]
            advantage_completion, returns = generalized_advantages(
                terminal_rewards, old_values, completion_mask, gamma=cfg.gamma, lam=cfg.gae_lambda
            )
            advantages = torch.zeros_like(old)
            for i, (start, length) in enumerate(zip(starts, lengths)):
                advantages[i, start : start + length] = advantage_completion[i, :length]
            value_data = (starts, lengths, completion_mask, old_values, returns)
        else:
            advantages = (
                group_advantages(torch.tensor(groups, device=device))
                .flatten()[:, None]
                .expand_as(old)
            )
            value_data = None
    return ids, labels, old, ref, mask, advantages, value_data, sum(rewards) / len(rewards)


def _export(model, tokenizer, output):
    with atomic_directory(output) as staging:
        merged = copy.deepcopy(model).cpu()
        if hasattr(merged, "_clearforge_adapter_config"):
            merge_adapters(merged)
        atomic_save(merged.state_dict(), staging / "model.pt")
        atomic_write_bytes(
            json.dumps(dataclasses.asdict(model.config)).encode(), staging / "model.json"
        )
        tokenizer.save(staging / "tokenizer.json")
        hashes = {}
        for name in ("model.pt", "model.json", "tokenizer.json"):
            with (staging / name).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        atomic_write_bytes(
            json.dumps(
                {"format": "clearforge-native-v1", "stage": "posttraining", "files": hashes}
            ).encode(),
            staging / "provenance.json",
        )


def train_posttraining(
    model: CausalLM,
    tokenizer: ChatTokenizer,
    data,
    config: PostTrainConfig,
    *,
    objective: str,
    reference: CausalLM | None = None,
    teacher: CausalLM | None = None,
    teacher_tokenizer: ChatTokenizer | None = None,
    reward_fn: Callable | None = None,
    reward_id: str | None = None,
    adapter_config: AdapterConfig | None = None,
    resume: str | Path | None = None,
    stop_after_steps: int | None = None,
) -> PostTrainResult:
    """Train a copy of the supplied native model; checkpoint after complete updates.

    Resume requires the same original base, reference/teacher, data, configuration,
    runtime and deterministic reward implementation identified by reward_id.
    Input model and teacher objects are never mutated. No data is truncated.
    """
    if objective not in {"sft", "lora", "dpo", "distillation", "ppo", "grpo", "cispo"}:
        raise ValueError("unknown posttraining objective")
    online = objective in {"ppo", "grpo", "cispo"}
    if online and (not callable(reward_fn) or not isinstance(reward_id, str) or not reward_id):
        raise ValueError("online training requires reward_fn and a stable reward_id")
    if objective in {"grpo", "cispo"} and config.samples_per_prompt < 2:
        raise ValueError("group training requires at least two samples per prompt")
    if objective == "dpo" and config.beta <= 0:
        raise ValueError("DPO beta must be positive")
    if stop_after_steps is not None and (
        type(stop_after_steps) is not int or not 0 <= stop_after_steps <= config.steps
    ):
        raise ValueError("invalid pause boundary")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model and tokenizer vocabulary differ")
    if objective == "distillation":
        if (
            teacher is None
            or teacher_tokenizer is None
            or _tokenizer_identity(teacher_tokenizer) != _tokenizer_identity(tokenizer)
        ):
            raise ValueError("distillation requires a teacher with the identical tokenizer")
        if teacher.config.vocab_size != model.config.vocab_size:
            raise ValueError("teacher vocabulary differs")
    rows = _records(data)
    encoded = [_encode(row, tokenizer, objective) for row in rows]
    if not online:
        sequences = [x for pair in encoded for x in pair] if objective == "dpo" else encoded
        limit = (
            min(model.config.max_seq_len, teacher.config.max_seq_len)
            if objective == "distillation"
            else model.config.max_seq_len
        )
        for ids, labels in sequences:
            if len(ids) > limit or not any(t != -100 for t in labels[1:]):
                raise ValueError("example exceeds context or has no supervised targets")
    else:
        for prompt in encoded:
            ids, _ = tokenizer.encode_chat(
                prompt["messages"], tools=prompt.get("tools"), add_generation_prompt=True
            )
            if len(ids) + config.max_new_tokens > model.config.max_seq_len:
                raise ValueError("prompt plus rollout budget exceeds model context")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config.device)
    cfg = dataclasses.asdict(config)
    cfg["output_path"] = str(config.output_path)
    adapters = adapter_config or (AdapterConfig() if objective == "lora" else None)
    identity = {
        "config": cfg,
        "objective": objective,
        "model_config": dataclasses.asdict(model.config),
        "base": _fingerprint(model),
        "tokenizer": _tokenizer_identity(tokenizer),
        "data": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "reference": _fingerprint(reference or model) if online or objective == "dpo" else None,
        "teacher": _fingerprint(teacher) if objective == "distillation" else None,
        "adapters": dataclasses.asdict(adapters) if adapters else None,
        "reward_id": reward_id,
        "runtime": _runtime_identity(device),
        "implementation": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("posttraining.py", "objectives.py", "adapters.py")
        },
    }
    output = config.output_path
    payload, resume_hash = None, None
    if resume:
        with Path(resume).open("rb") as stream:
            resume_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            payload = torch.load(stream, map_location="cpu", weights_only=False)
    if payload and (
        payload.get("format") != "clearforge-posttraining-v1" or payload["identity"] != identity
    ):
        raise ValueError("posttraining resume identity differs")
    if not payload and output.exists():
        raise FileExistsError(output)
    if payload and stop_after_steps is not None and stop_after_steps < payload["step"]:
        raise ValueError("pause boundary precedes checkpoint")
    with _determinism(True), contextlib.ExitStack() as ownership:
        if payload:
            ownership.enter_context(_run_lock(output))
            with Path(resume).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != resume_hash:
                    raise ValueError("resume checkpoint changed while acquiring run ownership")
        _seed(config.seed)
        policy = copy.deepcopy(model).to(device)
        frozen_reference = (
            _frozen(reference or model, device) if online or objective == "dpo" else None
        )
        if frozen_reference and frozen_reference.config != model.config:
            raise ValueError("reference architecture must match policy")
        frozen_teacher = _frozen(teacher, device) if objective == "distillation" else None
        if adapters:
            apply_adapters(policy, adapters)
        critic = ValueModel(model).to(device).eval() if objective == "ppo" else None
        parameters = [p for p in policy.parameters() if p.requires_grad]
        if critic is not None:
            parameters += list(critic.parameters())
        sparse_parameters = {
            id(parameter)
            for module in (policy, critic)
            if module is not None
            for name, parameter in module.named_parameters()
            if ".experts." in name
        }
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        cursor = _DataCursor(len(rows), config.seed, payload["cursor"] if payload else None)
        step = payload["step"] if payload else 0
        metrics = payload["metrics"] if payload else []
        if payload:
            policy.load_state_dict(payload["policy"])
            if critic is not None:
                critic.load_state_dict(payload["critic"])
            optimizer.load_state_dict(payload["optimizer"])
            restore_rng_state(payload["rng"])
            if type(step) is not int or not 0 <= step <= config.steps or len(metrics) != step:
                raise ValueError("checkpoint progress is inconsistent")
            updates_per_epoch = math.ceil(len(rows) / config.batch_size)
            expected_epoch = (step - 1) // updates_per_epoch if step else 0
            expected_position = (
                min(len(rows), ((step - 1) % updates_per_epoch + 1) * config.batch_size)
                if step
                else 0
            )
            if (cursor.epoch, cursor.position) != (
                expected_epoch,
                expected_position,
            ) or not cursor.order_is_consistent():
                raise ValueError("checkpoint data cursor is inconsistent")
            updates = step * (config.policy_epochs if online else 1)
            for group in optimizer.param_groups:
                if (
                    group["lr"] != config.learning_rate
                    or group["weight_decay"] != config.weight_decay
                    or group["betas"] != (0.9, 0.999)
                    or group["eps"] != 1e-8
                    or group["amsgrad"]
                ):
                    raise ValueError("checkpoint optimizer settings differ")
                for parameter in group["params"]:
                    state = optimizer.state.get(parameter, {})
                    if not torch.isfinite(parameter).all():
                        raise ValueError("checkpoint contains nonfinite weights")
                    if not updates:
                        if state:
                            raise ValueError("initial checkpoint contains optimizer moments")
                        continue
                    sparse = id(parameter) in sparse_parameters
                    if sparse and not state:
                        continue
                    progress = float(state.get("step", -1))
                    if (
                        not state
                        or not progress.is_integer()
                        or not 1 <= progress <= updates
                        or (not sparse and progress != updates)
                    ):
                        raise ValueError("checkpoint optimizer progress is inconsistent")
                    for name in ("exp_avg", "exp_avg_sq"):
                        moment = state[name]
                        if moment.shape != parameter.shape or not torch.isfinite(moment).all():
                            raise ValueError("checkpoint optimizer moments are invalid")
                        if name == "exp_avg_sq" and (moment < 0).any():
                            raise ValueError("checkpoint second moments must be nonnegative")
        else:
            output.mkdir(parents=True, exist_ok=False)
            ownership.enter_context(_run_lock(output))
        checkpoint = output / "latest.pt"
        with _stop_signals() as stop:

            def save():
                state = {
                    "format": "clearforge-posttraining-v1",
                    "identity": identity,
                    "step": step,
                    "policy": policy.state_dict(),
                    "critic": critic.state_dict() if critic is not None else None,
                    "optimizer": optimizer.state_dict(),
                    "cursor": cursor.state_dict(),
                    "rng": capture_rng_state(),
                    "metrics": metrics,
                }
                atomic_save(state, checkpoint)
                atomic_write_bytes(
                    "".join(json.dumps(m) + "\n" for m in metrics).encode(),
                    output / "metrics.jsonl",
                )

            save()
            while (
                step < config.steps
                and not stop["requested"]
                and (stop_after_steps is None or step < stop_after_steps)
            ):
                indices = cursor.group(config.batch_size, 1)[0]
                examples = [encoded[i] for i in indices]
                reward = None
                if online:
                    rollout = _rollout(
                        policy,
                        frozen_reference,
                        critic,
                        examples,
                        tokenizer,
                        config,
                        reward_fn,
                        objective,
                    )
                    ids, labels, old, ref, mask, advantages, value_data, reward = rollout
                    epochs = config.policy_epochs
                else:
                    policy.train()
                    epochs = 1
                losses = []
                for _ in range(epochs):
                    optimizer.zero_grad(set_to_none=True)
                    if objective == "dpo":
                        chosen_ids, chosen_labels = _batch(
                            [e[0] for e in examples], tokenizer, device
                        )
                        rejected_ids, rejected_labels = _batch(
                            [e[1] for e in examples], tokenizer, device
                        )
                        with torch.no_grad():
                            chosen_ref = _score(frozen_reference, chosen_ids, chosen_labels)
                            rejected_ref = _score(frozen_reference, rejected_ids, rejected_labels)
                        loss = dpo_loss(
                            _score(policy, chosen_ids, chosen_labels),
                            _score(policy, rejected_ids, rejected_labels),
                            chosen_ref,
                            rejected_ref,
                            beta=config.beta,
                        )
                    elif objective in {"sft", "lora", "distillation"}:
                        ids, labels = _batch(examples, tokenizer, device)
                        result = policy(ids, labels)
                        if objective == "distillation":
                            with torch.no_grad():
                                teacher_logits = frozen_teacher(ids).logits
                            loss = distillation_loss(
                                result.logits,
                                teacher_logits,
                                labels,
                                temperature=config.temperature,
                                alpha=config.distillation_alpha,
                            )
                        else:
                            loss = result.loss
                    else:
                        logp, _ = token_log_probabilities(
                            policy(ids).logits / config.sampling_temperature, labels
                        )
                        loss = policy_loss(
                            logp,
                            old,
                            advantages,
                            mask,
                            reference_logp=ref,
                            beta=config.beta,
                            clip_low=config.clip_ratio,
                            clip_high=config.clip_ratio,
                            algorithm=objective,
                        )
                        if objective == "ppo":
                            starts, lengths, completion_mask, old_values, returns = value_data
                            full_values = critic(ids)[:, :-1]
                            values = torch.stack(
                                [
                                    torch.nn.functional.pad(
                                        full_values[i, start : start + length],
                                        (0, completion_mask.shape[1] - length),
                                    )
                                    for i, (start, length) in enumerate(zip(starts, lengths))
                                ]
                            )
                            loss = loss + config.value_coefficient * clipped_value_loss(
                                values, old_values, returns, completion_mask, clip=config.clip_ratio
                            )
                    if loss is None or not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite posttraining loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in parameters):
                        raise FloatingPointError("nonfinite updated parameters")
                    losses.append(float(loss.detach()))
                step += 1
                metrics.append({"step": step, "loss": sum(losses) / len(losses), "reward": reward})
                save()
            if step == config.steps and not (output / "model").exists():
                _export(policy, tokenizer, output / "model")
    return PostTrainResult(
        policy, checkpoint, step, "complete" if step == config.steps else "paused"
    )
