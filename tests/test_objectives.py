import math

import pytest
import torch
from torch.nn import functional as F

from clearforge.objectives import (
    clipped_value_loss,
    distillation_loss,
    dpo_loss,
    generalized_advantages,
    group_advantages,
    masked_mean,
    policy_loss,
    sequence_log_probabilities,
    token_log_probabilities,
)


def test_token_scores_shift_once_and_ignore_prompt():
    logits = torch.tensor(
        [[[1.0, 2.0, 3.0], [4.0, 3.0, 1.0], [2.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], requires_grad=True
    )
    labels = torch.tensor([[-100, -100, 2, 1]])
    logp, mask = token_log_probabilities(logits, labels)
    expected = F.log_softmax(logits.float(), -1)
    torch.testing.assert_close(
        logp, torch.stack([torch.tensor(0.0), expected[0, 1, 2], expected[0, 2, 1]])[None]
    )
    assert mask.tolist() == [[False, True, True]]
    torch.testing.assert_close(sequence_log_probabilities(logits, labels), logp.sum(-1))
    torch.testing.assert_close(
        sequence_log_probabilities(logits, labels, average=True), logp.sum(-1) / 2
    )
    (-masked_mean(logp, mask)).backward()
    assert logits.grad[0, 0].abs().sum() == 0 and logits.grad[0, 3].abs().sum() == 0


def test_dpo_reference_formula_and_gradient_direction():
    chosen = torch.tensor([-2.0, -1.0], requires_grad=True)
    rejected = torch.tensor([-3.0, -4.0], requires_grad=True)
    reference_chosen = torch.tensor([-2.0, -1.0], requires_grad=True)
    reference_rejected = torch.tensor([-3.0, -4.0], requires_grad=True)
    loss = dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.2)
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert (chosen.grad < 0).all() and (rejected.grad > 0).all()
    assert reference_chosen.grad is None and reference_rejected.grad is None
    improved = dpo_loss(chosen + 1, rejected - 1, reference_chosen, reference_rejected, beta=0.2)
    assert improved < loss


def test_distillation_endpoints_and_teacher_frozen():
    torch.manual_seed(4)
    student = torch.randn(2, 4, 8, requires_grad=True)
    teacher = torch.randn(2, 4, 8, requires_grad=True)
    labels = torch.tensor([[-100, 1, 2, 3], [-100, -100, 4, -100]])
    logp, mask = token_log_probabilities(student, labels)
    torch.testing.assert_close(
        distillation_loss(student, teacher, labels, alpha=0), -masked_mean(logp, mask)
    )
    same = distillation_loss(student, student, labels, alpha=1)
    assert abs(same.item()) < 1e-6
    loss = distillation_loss(student, teacher, labels, temperature=3, alpha=0.7)
    loss.backward()
    assert student.grad.abs().sum() > 0 and teacher.grad is None
    assert student.grad[1, 0].abs().sum() == 0 and student.grad[1, 2].abs().sum() == 0


def test_group_advantages_constant_and_scale():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
    result = group_advantages(rewards)
    torch.testing.assert_close(result.mean(-1), torch.zeros(2))
    assert result[0, 0] < 0 and result[0, 2] > 0
    torch.testing.assert_close(result[1], torch.zeros(3))
    with pytest.raises(ValueError):
        group_advantages(torch.ones(2, 1))


@pytest.mark.parametrize("algorithm", ["ppo", "grpo", "cispo"])
def test_policy_clipping_matches_formula(algorithm):
    logp = torch.tensor([[math.log(1.5), math.log(0.5)]], requires_grad=True)
    old = torch.zeros_like(logp, requires_grad=True)
    advantage = torch.tensor([[1.0, -1.0]])
    mask = torch.ones_like(logp, dtype=torch.bool)
    loss = policy_loss(logp, old, advantage, mask, algorithm=algorithm)
    if algorithm == "cispo":
        expected = -(torch.tensor([[1.2, 0.8]]) * advantage * logp).mean()
    else:
        expected = torch.tensor(-0.2)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert old.grad is None
    if algorithm == "cispo":
        torch.testing.assert_close(logp.grad, torch.tensor([[-0.6, 0.4]]))
    else:
        torch.testing.assert_close(logp.grad, torch.zeros_like(logp))


def test_kl_penalty_zero_at_reference_and_positive_away():
    mask = torch.ones(1, 2, dtype=torch.bool)
    zeros = torch.zeros(1, 2)
    assert policy_loss(zeros, zeros, zeros, mask, reference_logp=zeros, beta=0.1) == 0
    assert policy_loss(zeros, zeros, zeros, mask, reference_logp=zeros + 1, beta=0.1) > 0


def test_gae_matches_discounted_returns_with_terminal_padding():
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    values = torch.zeros_like(rewards)
    advantage, returns = generalized_advantages(rewards, values, mask, gamma=0.5, lam=1)
    expected = torch.tensor([[0.25, 0.5, 1.0, 0.0], [1.0, 2.0, 0.0, 0.0]])
    torch.testing.assert_close(advantage, expected)
    torch.testing.assert_close(returns, expected)
    with pytest.raises(ValueError, match="right padded"):
        generalized_advantages(rewards, values, mask.flip(-1))


def test_value_clipping_and_mask():
    value = torch.tensor([[1.0, 9.0]], requires_grad=True)
    old = torch.zeros_like(value)
    target = torch.ones_like(value)
    mask = torch.tensor([[True, False]])
    loss = clipped_value_loss(value, old, target, mask, clip=0.2)
    assert loss.item() == pytest.approx(0.32)
    loss.backward()
    assert value.grad[0, 1] == 0


def test_empty_supervision_rejected():
    with pytest.raises(ValueError, match="supervised"):
        token_log_probabilities(torch.zeros(1, 3, 8), torch.full((1, 3), -100, dtype=torch.long))
    with pytest.raises(ValueError, match="supervised"):
        masked_mean(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
