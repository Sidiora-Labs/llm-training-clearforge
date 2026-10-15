"""Numerical contracts for the decoder, supervision, and bounded generation."""

import math
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from clearforge.config import ModelConfig
from clearforge.model import CausalAttention, CausalLM, RMSNorm, RotaryEmbedding

torch.set_num_threads(2)


def configuration(**kwargs):
    return replace(
        ModelConfig(
            vocab_size=31,
            hidden_size=32,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            max_seq_len=16,
        ),
        **kwargs,
    )


@pytest.fixture
def model():
    torch.manual_seed(19)
    return CausalLM(configuration()).eval()


def test_forward_and_tied_parameter_accounting(model):
    result = model(torch.tensor([[1, 2, 3]]))
    assert result.logits.shape == (1, 3, 31)
    assert result.loss is None
    assert model.lm_head.weight is model.token_embedding.weight
    untied = CausalLM(configuration(tie_embeddings=False))
    assert untied.lm_head.weight is not untied.token_embedding.weight
    assert untied.parameter_count() - model.parameter_count() == 31 * 32
    expected = 31 * 32 + 2 * (2 * 32 + 2 * 32 * 32 + 2 * 32 * 16 + 3 * 32 * 64) + 32
    assert model.parameter_count() == expected


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_attention_matches_explicit_causal_grouped_reference(kv_heads):
    torch.manual_seed(5)
    attention = CausalAttention(configuration(num_kv_heads=kv_heads)).eval()
    x = torch.randn(2, 5, 32)
    q = attention.rotary(attention.q_proj(x).view(2, 5, 4, 8).transpose(1, 2))
    k = attention.rotary(attention.k_proj(x).view(2, 5, kv_heads, 8).transpose(1, 2))
    v = attention.v_proj(x).view(2, 5, kv_heads, 8).transpose(1, 2)
    k, v = (a.repeat_interleave(4 // kv_heads, dim=1) for a in (k, v))
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(8)
    scores.masked_fill_(torch.ones(5, 5, dtype=torch.bool).triu(1), -torch.inf)
    expected = attention.out_proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(2, 5, 32))
    torch.testing.assert_close(attention(x), expected, atol=2e-7, rtol=2e-6)


def test_future_tokens_cannot_change_prefix_logits(model):
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = torch.tensor([[1, 2, 3, 29, 30, 7]])
    torch.testing.assert_close(model(first).logits[:, :3], model(changed).logits[:, :3])
    torch.testing.assert_close(model(first).logits[:, :3], model(first[:, :3]).logits)
    assert not torch.equal(model(first).logits[:, 3:], model(changed).logits[:, 3:])


def test_masked_right_padded_loss_matches_token_reference(model):
    ids = torch.tensor([[1, 2, 3, 4, 0, 0], [1, 7, 8, 9, 10, 11]])
    labels = torch.tensor([[-100, -100, 3, 4, -100, -100], [-100, 7, -100, 9, 10, 11]])
    result = model(ids, labels)
    losses = []
    for batch in range(2):
        for position in range(1, 6):
            target = labels[batch, position].item()
            if target != -100:
                losses.append(-result.logits[batch, position - 1].log_softmax(-1)[target])
    torch.testing.assert_close(result.loss, torch.stack(losses).mean())
    changed_padding = ids.clone()
    changed_padding[0, 4:] = torch.tensor([20, 21])
    torch.testing.assert_close(model(changed_padding, labels).loss, result.loss)
    labels[:, 0] = 30
    torch.testing.assert_close(model(ids, labels).loss, result.loss)


def test_aligned_labels_shift_exactly_once(model):
    ids = torch.tensor([[2, 5, 9]])
    labels = torch.tensor([[-100, -100, 13]])
    result = model(ids, labels)
    expected = F.cross_entropy(result.logits[:, 1], torch.tensor([13]))
    torch.testing.assert_close(result.loss, expected)


def test_finite_nonzero_gradients_reach_every_parameter(model):
    model.train()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    loss = model(ids, ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_rmsnorm_numeric_reference_and_bfloat16():
    norm = RMSNorm(4)
    x = torch.tensor([[1.0, 2.0, -3.0, 0.0]])
    expected = x / torch.sqrt(torch.tensor(14.0 / 4 + 1e-6))
    torch.testing.assert_close(norm(x), expected)
    assert norm(x.bfloat16()).dtype == torch.bfloat16
    torch.testing.assert_close(norm(torch.zeros(2, 4)), torch.zeros(2, 4))


def test_rotary_positions_and_length_independent_coordinates():
    rotary = RotaryEmbedding(4, 10000.0)
    x = torch.tensor([[[[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]]]])
    expected = torch.tensor(
        [[[[1.0, 0.0, 1.0, 0.0], [math.cos(1.0), math.sin(1.0), math.cos(0.01), math.sin(0.01)]]]]
    )
    torch.testing.assert_close(rotary(x), expected)
    torch.testing.assert_close(rotary(x[..., :1, :]), expected[..., :1, :])
    torch.testing.assert_close(rotary(x).square().sum(-1), x.square().sum(-1))


def test_residual_initialization_is_scaled_and_reproducible():
    config = configuration(hidden_size=256, intermediate_size=512)
    torch.manual_seed(7)
    first = CausalLM(config)
    torch.manual_seed(7)
    second = CausalLM(config)
    for name, parameter in first.named_parameters():
        torch.testing.assert_close(parameter, dict(second.named_parameters())[name], rtol=0, atol=0)
    assert first.token_embedding.weight.std().item() == pytest.approx(0.02, rel=0.05)
    for block in first.layers:
        assert block.attention.out_proj.weight.std().item() == pytest.approx(0.01, rel=0.05)
        assert block.mlp.down_proj.weight.std().item() == pytest.approx(0.01, rel=0.05)


@pytest.mark.parametrize(
    "ids",
    [
        [],
        torch.ones(2, 3),
        torch.ones(2, 3, dtype=torch.int32),
        torch.tensor([1, 2]),
        torch.empty((0, 3), dtype=torch.long),
        torch.empty((1, 0), dtype=torch.long),
        torch.tensor([[31]]),
        torch.tensor([[-1]]),
        torch.ones((1, 17), dtype=torch.long),
    ],
)
def test_invalid_inputs_fail_before_model_execution(model, ids):
    with pytest.raises(ValueError):
        model(ids)


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([1, 2, 3]),
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[1, -1, 2]]),
        torch.tensor([[1, 2, 31]]),
        torch.tensor([[-100, -100, -100]]),
        torch.tensor([[1, -100, -100]]),
    ],
)
def test_invalid_labels_and_no_shifted_supervision(model, labels):
    with pytest.raises(ValueError):
        model(torch.tensor([[1, 2, 3]]), labels)


def test_single_token_inference_but_not_training(model):
    assert model(torch.tensor([[1]])).logits.shape == (1, 1, 31)
    with pytest.raises(ValueError, match="supervised"):
        model(torch.tensor([[1]]), torch.tensor([[1]]))


def test_config_is_revalidated_before_allocation():
    with pytest.raises(ValueError):
        CausalLM({})
    config = configuration()
    object.__setattr__(config, "num_kv_heads", 3)
    with pytest.raises(ValueError, match="divisible"):
        CausalLM(config)


def test_dropout_active_in_training_and_disabled_in_eval():
    model = CausalLM(configuration(dropout=0.4)).train()
    ids = torch.tensor([[1, 2, 3, 4]])
    torch.manual_seed(6)
    first = model(ids).logits
    torch.manual_seed(7)
    second = model(ids).logits
    assert not torch.equal(first, second)
    model.eval()
    torch.testing.assert_close(model(ids).logits, model(ids).logits, rtol=0, atol=0)


def test_greedy_generation_matches_manual_full_context_and_restores_modes():
    model = CausalLM(configuration(dropout=0.3)).train()
    model.layers[0].mlp.eval()
    modes = [module.training for module in model.modules()]
    prompt = torch.tensor([[1, 2]])
    actual = model.generate(prompt, max_new_tokens=4)
    assert modes == [module.training for module in model.modules()]
    model.eval()
    expected = prompt.clone()
    for _ in range(4):
        expected = torch.cat((expected, model(expected).logits[:, -1].argmax(-1, keepdim=True)), 1)
    assert torch.equal(actual, expected)
    assert torch.equal(prompt, torch.tensor([[1, 2]]))


def test_generation_budget_and_zero_tokens(model):
    prompt = torch.tensor([[1, 2]])
    assert torch.equal(model.generate(prompt, max_new_tokens=0), prompt)
    assert model.generate(prompt, max_new_tokens=14).shape == (1, 16)
    with pytest.raises(ValueError, match="budget"):
        model.generate(prompt, max_new_tokens=15)


def test_eos_generation_and_existing_eos(model):
    prompt = torch.tensor([[1, 2]])
    next_id = model(prompt).logits[:, -1].argmax(-1).item()
    generated = model.generate(prompt, max_new_tokens=4, eos_token_id=next_id)
    assert generated.shape == (1, 2 if next_id == 2 else 3)
    assert generated[0, -1].item() == next_id
    assert torch.equal(model.generate(prompt, max_new_tokens=4, eos_token_id=2), prompt)


def test_seeded_sampling_and_top_k_one(model):
    ids = torch.tensor([[1, 2]])
    torch.manual_seed(123)
    first = model.generate(ids, max_new_tokens=5, temperature=0.7, top_k=5)
    torch.manual_seed(123)
    second = model.generate(ids, max_new_tokens=5, temperature=0.7, top_k=5)
    assert torch.equal(first, second)
    greedy = model.generate(ids, max_new_tokens=5)
    assert torch.equal(model.generate(ids, max_new_tokens=5, temperature=0.7, top_k=1), greedy)
    for position in range(2, first.shape[1]):
        candidates = model(first[:, :position]).logits[:, -1].topk(5).indices[0]
        assert first[0, position].item() in candidates.tolist()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": -1},
        {"max_new_tokens": True},
        {"max_new_tokens": 1.5},
        {"temperature": -1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": True},
        {"top_k": 0},
        {"top_k": 32},
        {"top_k": 1.5},
        {"top_k": True},
        {"eos_token_id": -1},
        {"eos_token_id": 31},
        {"eos_token_id": True},
    ],
)
def test_invalid_generation_controls(model, kwargs):
    with pytest.raises(ValueError):
        model.generate(torch.tensor([[1, 2]]), **({"max_new_tokens": 1} | kwargs))


def test_equal_length_batch_generation(model):
    ids = torch.tensor([[1], [2]])
    output = model.generate(ids, max_new_tokens=1)
    assert output.shape == (2, 2)
    assert torch.equal(output[:, :1], ids)


def test_nonfinite_generation_rejected_and_mode_restored(model):
    model.train()
    with torch.no_grad():
        model.lm_head.weight[0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        model.generate(torch.tensor([[1, 2]]), max_new_tokens=1)
    assert model.training


def test_low_temperature_is_numerically_valid(model):
    prompt = torch.tensor([[1, 2]])
    actual = model.generate(prompt, max_new_tokens=2, temperature=1e-100)
    assert torch.equal(actual, model.generate(prompt, max_new_tokens=2))


def test_ignored_batch_row_does_not_dilute_token_mean(model):
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    labels = torch.tensor([[-100, -100, -100], [-100, 5, 6]])
    torch.testing.assert_close(model(ids, labels).loss, model(ids[1:], labels[1:]).loss)


def test_cpu_bfloat16_autocast_backward(model):
    model.train()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(ids, ids)
    assert result.logits.dtype == torch.bfloat16
    assert result.loss.dtype == torch.float32
    result.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_device_mismatch_rejected(model):
    with pytest.raises(ValueError, match="same device"):
        model(torch.ones((1, 2), device="meta", dtype=torch.long))
    with pytest.raises(ValueError, match="shape and device"):
        model(torch.tensor([[1, 2]]), torch.ones((1, 2), device="meta", dtype=torch.long))


def test_attention_dropout_itself_is_disabled_in_eval():
    attention = CausalAttention(configuration(dropout=0.5)).train()
    x = torch.randn(1, 8, 32)
    torch.manual_seed(11)
    first = attention(x)
    torch.manual_seed(12)
    assert not torch.equal(first, attention(x))
    attention.eval()
    torch.testing.assert_close(attention(x), attention(x), rtol=0, atol=0)
