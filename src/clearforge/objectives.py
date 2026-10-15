"""Differentiable objectives for preference, distillation and policy training."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean and match the values")
    count = mask.sum()
    if count.item() == 0:
        raise ValueError("objective has no supervised positions")
    return values.masked_fill(~mask, 0).sum() / count


def token_log_probabilities(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Shift aligned labels once and return per-target log probabilities and mask."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or labels.dtype != torch.long:
        raise ValueError("expected logits [batch, length, vocabulary] and aligned long labels")
    targets = labels[:, 1:]
    mask = targets != -100
    if not mask.any() or torch.any(mask & ((targets < 0) | (targets >= logits.shape[-1]))):
        raise ValueError("labels must contain valid supervised next-token targets")
    logp = logits[:, :-1].float().log_softmax(-1)
    selected = logp.gather(-1, targets.masked_fill(~mask, 0).unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~mask, 0), mask


def sequence_log_probabilities(logits: Tensor, labels: Tensor, *, average: bool = False) -> Tensor:
    values, mask = token_log_probabilities(logits, labels)
    counts = mask.sum(-1)
    if (counts == 0).any():
        raise ValueError("each sequence needs supervised targets")
    result = values.sum(-1)
    return result / counts if average else result


def dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    *,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Direct preference optimization using sequence-summed log probabilities."""
    if not math.isfinite(beta) or beta <= 0 or not 0 <= label_smoothing < 0.5:
        raise ValueError("beta must be positive and label_smoothing in [0, 0.5)")
    values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if any(v.shape != policy_chosen.shape for v in values) or policy_chosen.numel() == 0:
        raise ValueError("preference scores must have matching nonempty shapes")
    if any(not torch.isfinite(v).all() for v in values):
        raise ValueError("preference scores must be finite")
    margin = beta * (
        policy_chosen - policy_rejected - reference_chosen.detach() + reference_rejected.detach()
    )
    return (
        -(1 - label_smoothing) * F.logsigmoid(margin) - label_smoothing * F.logsigmoid(-margin)
    ).mean()


def distillation_loss(
    student: Tensor,
    teacher: Tensor,
    labels: Tensor,
    *,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Tensor:
    """Blend supervised CE with temperature-scaled teacher-to-student KL."""
    if student.shape != teacher.shape or student.ndim != 3:
        raise ValueError("teacher and student logits must have matching shapes")
    if not math.isfinite(temperature) or temperature <= 0 or not 0 <= alpha <= 1:
        raise ValueError("temperature must be positive and alpha in [0, 1]")
    logp, mask = token_log_probabilities(student, labels)
    student_soft = (student[:, :-1].float() / temperature).log_softmax(-1)
    teacher_soft = (teacher[:, :-1].detach().float() / temperature).log_softmax(-1)
    kl = F.kl_div(student_soft, teacher_soft, reduction="none", log_target=True).sum(-1)
    return alpha * masked_mean(kl, mask) * temperature**2 - (1 - alpha) * masked_mean(logp, mask)


def group_advantages(rewards: Tensor, *, epsilon: float = 1e-6) -> Tensor:
    """Normalize rewards within each prompt group; constant groups yield zero."""
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not torch.isfinite(rewards).all():
        raise ValueError("group rewards must be finite [prompts, samples>=2]")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be positive")
    centered = rewards - rewards.mean(-1, keepdim=True)
    return centered / (rewards.std(-1, unbiased=False, keepdim=True) + epsilon)


def policy_loss(
    logp: Tensor,
    old_logp: Tensor,
    advantages: Tensor,
    mask: Tensor,
    *,
    reference_logp: Tensor | None = None,
    beta: float = 0.0,
    clip_low: float = 0.2,
    clip_high: float = 0.2,
    algorithm: str = "grpo",
) -> Tensor:
    """PPO/GRPO clipped surrogate or CISPO detached importance-weighted objective."""
    if algorithm not in {"ppo", "grpo", "cispo"}:
        raise ValueError("algorithm must be ppo, grpo, or cispo")
    if not all(math.isfinite(v) for v in (beta, clip_low, clip_high)):
        raise ValueError("policy controls must be finite")
    if beta < 0 or not 0 <= clip_low < 1 or clip_high < 0:
        raise ValueError("invalid clipping or KL coefficient")
    if logp.shape != old_logp.shape or logp.shape != advantages.shape:
        raise ValueError("policy tensors must have matching shapes")
    if any(not torch.isfinite(v).all() for v in (logp, old_logp, advantages)):
        raise ValueError("policy tensors must be finite")
    ratio = (logp - old_logp.detach()).exp()
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    if algorithm == "cispo":
        surrogate = -clipped.detach() * advantages.detach() * logp
    else:
        surrogate = -torch.minimum(ratio * advantages.detach(), clipped * advantages.detach())
    if reference_logp is not None:
        if reference_logp.shape != logp.shape or not torch.isfinite(reference_logp).all():
            raise ValueError("reference probabilities must be finite and match the policy")
        delta = reference_logp.detach() - logp
        surrogate = surrogate + beta * (delta.exp() - delta - 1)
    elif beta:
        raise ValueError("KL coefficient requires reference probabilities")
    result = masked_mean(surrogate, mask)
    if not torch.isfinite(result):
        raise ValueError("nonfinite policy objective")
    return result


def generalized_advantages(
    rewards: Tensor, values: Tensor, mask: Tensor, *, gamma: float = 1.0, lam: float = 0.95
) -> tuple[Tensor, Tensor]:
    """GAE for right-padded completed episodes; terminal bootstrap is zero."""
    if rewards.shape != values.shape or rewards.ndim != 2 or mask.shape != values.shape:
        raise ValueError("rewards, values and mask must share [batch, time] shape")
    if mask.dtype != torch.bool or not mask.any() or not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("invalid GAE mask or discount controls")
    if torch.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError("GAE episodes must be right padded")
    if not torch.isfinite(rewards).all() or not torch.isfinite(values).all():
        raise ValueError("GAE inputs must be finite")
    advantages = torch.zeros_like(values)
    carry = torch.zeros_like(values[:, 0])
    for step in reversed(range(values.shape[1])):
        following = (
            mask[:, step + 1] if step + 1 < values.shape[1] else torch.zeros_like(mask[:, step])
        )
        next_value = values[:, step + 1].detach() if step + 1 < values.shape[1] else 0
        delta = rewards[:, step] + gamma * next_value * following - values[:, step].detach()
        carry = (delta + gamma * lam * carry * following) * mask[:, step]
        advantages[:, step] = carry
    return advantages, (advantages + values.detach()).masked_fill(~mask, 0)


def clipped_value_loss(
    values: Tensor, old_values: Tensor, returns: Tensor, mask: Tensor, *, clip: float = 0.2
) -> Tensor:
    if values.shape != old_values.shape or values.shape != returns.shape:
        raise ValueError("value tensors must have matching shapes")
    if not math.isfinite(clip) or clip < 0:
        raise ValueError("value clip must be finite and nonnegative")
    clipped = old_values.detach() + (values - old_values.detach()).clamp(-clip, clip)
    error = torch.maximum(
        (values - returns.detach()).square(), (clipped - returns.detach()).square()
    )
    return 0.5 * masked_mean(error, mask)
