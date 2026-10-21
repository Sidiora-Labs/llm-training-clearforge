"""On-policy GRPO/CISPO updates from real native multi-turn tool episodes."""

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

from .adapters import _fingerprint
from .checkpoint import atomic_save, atomic_write_bytes, capture_rng_state, restore_rng_state
from .data import adapt_rollout
from .objectives import group_advantages, policy_loss, token_log_probabilities
from .posttraining import (
    PostTrainConfig,
    PostTrainResult,
    _batch,
    _export,
    _frozen,
    _records,
    _tokenizer_identity,
)
from .rollout import EpisodeLimits, GenerationOptions, NativeBackend, ToolRegistry, run_episode
from .training import _DataCursor, _determinism, _run_lock, _runtime_identity, _seed, _stop_signals


@dataclasses.dataclass(frozen=True)
class AgentTrainConfig(PostTrainConfig):
    episode_limits: EpisodeLimits = dataclasses.field(default_factory=EpisodeLimits)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.episode_limits, EpisodeLimits):
            raise ValueError("episode_limits must be an EpisodeLimits object")
        if self.episode_limits.max_turns > 128:
            raise ValueError("agent training supports at most 128 sampled turns per episode")
        if self.samples_per_prompt < 2:
            raise ValueError("agent group training requires at least two samples per prompt")


def collect_agent_group(policy, reference, tokenizer, records, registry, config, reward_fn):
    """Collect exact native token traces, including malformed sampled actions.

    Each assistant turn is scored under its actual prompt. Earlier assistant
    turns and tool observations are context for later turns, never relabeled as
    newly sampled actions. The episode advantage applies to every sampled turn.
    """
    policy.eval()
    device = next(policy.parameters()).device
    backend = NativeBackend(policy, tokenizer)
    examples, owners, reward_groups, episodes = [], [], [], []
    generation = GenerationOptions(
        max_new_tokens=config.max_new_tokens, temperature=config.sampling_temperature
    )
    with torch.no_grad():
        for record in records:
            group = []
            for _ in range(config.samples_per_prompt):
                backend.traces.clear()
                episode = run_episode(
                    backend,
                    record["messages"],
                    registry,
                    generation=generation,
                    limits=config.episode_limits,
                )
                reward = float(reward_fn(copy.deepcopy(record), episode))
                if not math.isfinite(reward):
                    raise ValueError("agent reward must be finite")
                if not any(trace["generated_ids"] for trace in backend.traces):
                    raise ValueError(
                        f"episode produced no trainable sampled action: {episode.error}"
                    )
                index = len(episodes)
                for trace in backend.traces:
                    prompt, sampled = trace["prompt_ids"], trace["generated_ids"]
                    if not sampled:
                        continue
                    examples.append((prompt + sampled, [-100] * len(prompt) + sampled))
                    owners.append(index)
                group.append(reward)
                episodes.append(episode)
            reward_groups.append(group)
        ids, labels = _batch(examples, tokenizer, device)
        old, mask = token_log_probabilities(
            policy(ids).logits / config.sampling_temperature, labels
        )
        reference_logp, _ = token_log_probabilities(
            reference(ids).logits / config.sampling_temperature, labels
        )
        episode_advantages = group_advantages(torch.tensor(reward_groups, device=device)).flatten()
        advantages = episode_advantages[torch.tensor(owners, device=device)][:, None].expand_as(old)
    summary = {
        "reward": sum(map(sum, reward_groups)) / len(episodes),
        "episodes": len(episodes),
        "assistant_turns": len(examples),
        "sampled_tokens": int(mask.sum()),
        "tool_calls": sum(len(episode.tool_events) for episode in episodes),
        "statuses": {
            status: sum(e.status == status for e in episodes)
            for status in sorted({e.status for e in episodes})
        },
    }
    return ids, labels, old, reference_logp, mask, advantages, summary


def train_agent_policy(
    model,
    tokenizer,
    data,
    config: AgentTrainConfig,
    *,
    registry: ToolRegistry,
    registry_id: str,
    reward_fn: Callable,
    reward_id: str,
    objective: str = "grpo",
    reference=None,
    resume: str | Path | None = None,
    stop_after_steps: int | None = None,
) -> PostTrainResult:
    """Train a copy of a native policy through actual sampled tool episodes.

    Only explicit registered handlers can execute. Reward functions receive
    ``(record, EpisodeResult)``. Exact resume requires deterministic handlers and
    rewards, unchanged registry/reward identifiers and the same original model.
    """
    if objective not in {"grpo", "cispo"}:
        raise ValueError("agent objective must be grpo or cispo")
    if not isinstance(registry, ToolRegistry) or not callable(reward_fn):
        raise ValueError("agent training requires a ToolRegistry and reward callback")
    if any(not isinstance(value, str) or not value for value in (registry_id, reward_id)):
        raise ValueError("registry_id and reward_id must be stable nonempty identifiers")
    if stop_after_steps is not None and (
        type(stop_after_steps) is not int or not 0 <= stop_after_steps <= config.steps
    ):
        raise ValueError("invalid pause boundary")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary differs from tokenizer")
    definitions = registry.definitions
    rows = [adapt_rollout(row) for row in _records(data)]
    for row in rows:
        if row.get("tools") is not None and row["tools"] != definitions:
            raise ValueError("record tool schemas differ from the explicit registry")
        row["tools"] = definitions
        ids, _ = tokenizer.encode_chat(
            row["messages"], tools=definitions, add_generation_prompt=True
        )
        if len(ids) + config.max_new_tokens > model.config.max_seq_len:
            raise ValueError("agent prompt plus completion budget exceeds context")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config.device)
    cfg = dataclasses.asdict(config)
    cfg["output_path"] = str(config.output_path)
    identity = {
        "config": cfg,
        "objective": objective,
        "base": _fingerprint(model),
        "model_config": dataclasses.asdict(model.config),
        "trainable_parameters": [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ],
        "reference": _fingerprint(reference or model),
        "tokenizer": _tokenizer_identity(tokenizer),
        "data": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "registry_id": registry_id,
        "reward_id": reward_id,
        "runtime": _runtime_identity(device),
        "implementation": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "agent_training.py",
                "rollout.py",
                "tools.py",
                "objectives.py",
                "posttraining.py",
            )
        },
    }
    payload, digest = None, None
    if resume:
        with Path(resume).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            payload = torch.load(stream, map_location="cpu", weights_only=False)
        if (
            payload.get("format") != "clearforge-agent-training-v1"
            or payload["identity"] != identity
        ):
            raise ValueError("agent checkpoint identity differs")
    output = config.output_path
    if not payload and output.exists():
        raise FileExistsError(output)
    with _determinism(True), contextlib.ExitStack() as ownership:
        if payload:
            ownership.enter_context(_run_lock(output))
            with Path(resume).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("checkpoint changed before acquiring ownership")
        _seed(config.seed)
        policy = copy.deepcopy(model).to(device).eval()
        frozen_reference = _frozen(reference or model, device)
        if frozen_reference.config != model.config:
            raise ValueError("reference architecture differs")
        parameters = [p for p in policy.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("policy has no trainable parameters")
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        cursor = _DataCursor(len(rows), config.seed, payload["cursor"] if payload else None)
        step = payload["step"] if payload else 0
        metrics = payload["metrics"] if payload else []
        if stop_after_steps is not None and stop_after_steps < step:
            raise ValueError("pause precedes checkpoint")
        if payload:
            policy.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            restore_rng_state(payload["rng"])
            if type(step) is not int or not 0 <= step <= config.steps or len(metrics) != step:
                raise ValueError("checkpoint progress is inconsistent")
            updates = step * config.policy_epochs
            names = {id(p): name for name, p in policy.named_parameters()}
            for group in optimizer.param_groups:
                if (
                    group["lr"] != config.learning_rate
                    or group["weight_decay"] != config.weight_decay
                ):
                    raise ValueError("checkpoint optimizer settings differ")
                if group["betas"] != (0.9, 0.999) or group["eps"] != 1e-8 or group["amsgrad"]:
                    raise ValueError("checkpoint AdamW settings differ")
                for p in group["params"]:
                    state = optimizer.state.get(p, {})
                    expert = ".experts." in names[id(p)]
                    if not updates:
                        if state:
                            raise ValueError("initial checkpoint contains optimizer moments")
                        continue
                    if expert and not state:
                        continue
                    progress = float(state["step"]) if state else -1
                    valid = (
                        1 <= progress <= updates and progress.is_integer()
                        if expert
                        else progress == updates
                    )
                    if not valid:
                        raise ValueError("checkpoint optimizer progress is inconsistent")
                    for name in ("exp_avg", "exp_avg_sq"):
                        moment = state[name]
                        if (
                            moment.shape != p.shape
                            or not torch.isfinite(moment).all()
                            or (name == "exp_avg_sq" and (moment < 0).any())
                        ):
                            raise ValueError("checkpoint optimizer moments invalid")
            per_epoch = math.ceil(len(rows) / config.batch_size)
            expected = (
                (
                    (step - 1) // per_epoch,
                    min(len(rows), ((step - 1) % per_epoch + 1) * config.batch_size),
                )
                if step
                else (0, 0)
            )
            if (cursor.epoch, cursor.position) != expected or not cursor.order_is_consistent():
                raise ValueError("checkpoint cursor is inconsistent")
        if any(not torch.isfinite(p).all() for p in policy.parameters()):
            raise ValueError("policy contains nonfinite weights")
        if not payload:
            output.mkdir(parents=True, exist_ok=False)
            ownership.enter_context(_run_lock(output))
        checkpoint = output / "latest.pt"

        def save():
            atomic_save(
                {
                    "format": "clearforge-agent-training-v1",
                    "identity": identity,
                    "step": step,
                    "model": policy.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": capture_rng_state(),
                    "cursor": cursor.state_dict(),
                    "metrics": metrics,
                },
                checkpoint,
            )
            atomic_write_bytes(
                "".join(json.dumps(m, sort_keys=True) + "\n" for m in metrics).encode(),
                output / "metrics.jsonl",
            )

        with _stop_signals() as stop:
            save()
            while (
                step < config.steps
                and not stop["requested"]
                and (stop_after_steps is None or step < stop_after_steps)
            ):
                indices = cursor.group(config.batch_size, 1)[0]
                rollout = collect_agent_group(
                    policy,
                    frozen_reference,
                    tokenizer,
                    [rows[i] for i in indices],
                    registry,
                    config,
                    reward_fn,
                )
                ids, labels, old, ref, mask, advantages, summary = rollout
                losses = []
                for _ in range(config.policy_epochs):
                    optimizer.zero_grad(set_to_none=True)
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
                    if not torch.isfinite(loss):
                        raise FloatingPointError("nonfinite agent policy loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        parameters, config.grad_clip, error_if_nonfinite=True
                    )
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in parameters):
                        raise FloatingPointError("nonfinite updated policy")
                    losses.append(float(loss.detach()))
                step += 1
                metrics.append({"step": step, "loss": sum(losses) / len(losses), **summary})
                save()
            if step == config.steps and not (output / "model").exists():
                _export(policy, tokenizer, output / "model")
            status = (
                "complete"
                if step == config.steps
                else ("interrupted" if stop["requested"] else "paused")
            )
    return PostTrainResult(policy, checkpoint, step, status)
