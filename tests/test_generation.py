"""Numerical parity for cached inference, sampling, sparse routing, and YaRN."""

import math
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from clearforge.config import ModelConfig
from clearforge.model import CausalLM, CausalLMOutput, RotaryEmbedding, SparseExperts


def config(**kwargs):
    return replace(
        ModelConfig(
            vocab_size=23,
            hidden_size=16,
            num_heads=4,
            num_kv_heads=2,
            num_layers=2,
            intermediate_size=32,
            max_seq_len=32,
        ),
        **kwargs,
    )


@pytest.fixture
def model():
    torch.manual_seed(3)
    return CausalLM(config()).eval()


@pytest.mark.parametrize("splits", [(1, 1, 1, 1, 1), (2, 3), (3, 2)])
def test_cached_chunk_logits_match_full(model, splits):
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    expected = model(ids).logits
    cache, outputs, start = None, [], 0
    for size in splits:
        output = model(ids[:, start : start + size], past_key_values=cache, use_cache=True)
        outputs.append(output.logits)
        cache = output.past_key_values
        start += size
        assert cache[0][0].shape == (2, 2, start, 4)
    torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-7, rtol=1e-5)


def test_cached_chunk_is_causal(model):
    past = model(torch.tensor([[1, 2]]), use_cache=True).past_key_values
    first = model(torch.tensor([[3, 4, 5]]), past_key_values=past).logits
    second = model(torch.tensor([[3, 9, 8]]), past_key_values=past).logits
    torch.testing.assert_close(first[:, :1], second[:, :1], rtol=0, atol=0)


def test_last_logits_and_supervision_contract(model):
    ids = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(model(ids, logits_to_keep=2).logits, model(ids).logits[:, -2:])
    with pytest.raises(ValueError, match="labels require"):
        model(ids, ids, logits_to_keep=1)
    with pytest.raises(ValueError, match="labels require"):
        model(ids, ids, past_key_values=model(ids, use_cache=True).past_key_values)


@pytest.mark.parametrize("mutation", ["layers", "pair", "shape", "length", "dtype", "context"])
def test_invalid_cache_rejected(model, mutation):
    cache = list(model(torch.tensor([[1, 2]]), use_cache=True).past_key_values)
    if mutation == "layers":
        cache.pop()
    elif mutation == "pair":
        cache[0] = (cache[0][0],)
    elif mutation == "shape":
        cache[0] = tuple(t[:, :1] for t in cache[0])
    elif mutation == "length":
        cache[1] = tuple(t[:, :, :1] for t in cache[1])
    elif mutation == "dtype":
        cache[0] = tuple(t.double() for t in cache[0])
    else:
        cache = [tuple(t.repeat(1, 1, 16, 1) for t in pair) for pair in cache]
    with pytest.raises(ValueError, match="cache"):
        model(torch.tensor([[3]]), past_key_values=cache)


def test_cached_greedy_matches_full_recomputation(model):
    prompt = torch.tensor([[1, 2], [3, 4]])
    expected = prompt.clone()
    for _ in range(5):
        expected = torch.cat((expected, model(expected).logits[:, -1].argmax(-1, keepdim=True)), 1)
    assert torch.equal(model.generate(prompt, max_new_tokens=5), expected)


def test_stream_matches_generate_and_restores_on_close(model):
    prompt = torch.tensor([[1, 2], [3, 4]])
    kwargs = dict(max_new_tokens=4, num_return_sequences=2, temperature=0.8, top_k=8, top_p=0.8)
    torch.manual_seed(9)
    expected = model.generate(prompt, **kwargs)
    torch.manual_seed(9)
    chunks = list(model.stream_generate(prompt, **kwargs))
    assert all(chunk.shape == (4, 1) for chunk in chunks)
    assert torch.equal(torch.cat([prompt.repeat_interleave(2, 0), *chunks], 1), expected)
    model.train()
    model.layers[0].eval()
    modes = [module.training for module in model.modules()]
    stream = model.stream_generate(prompt, max_new_tokens=3)
    next(stream)
    assert torch.is_grad_enabled()  # Generator must not leak thread-local no_grad across yield.
    stream.close()
    assert [module.training for module in model.modules()] == modes


def fixed_logits(model, monkeypatch, scores):
    def forward(ids, **kwargs):
        logits = scores.to(ids.device).view(1, 1, -1).expand(ids.shape[0], ids.shape[1], -1)
        return CausalLMOutput(logits)

    monkeypatch.setattr(model, "forward", forward)


def test_nucleus_retains_first_crossing_token(model, monkeypatch):
    scores = torch.full((23,), -100.0)
    scores[:3] = torch.tensor([0.5, 0.3, 0.2]).log()
    fixed_logits(model, monkeypatch, scores)
    outputs = model.generate(
        torch.tensor([[4]]).repeat(100, 1), max_new_tokens=1, temperature=1.0, top_p=0.6
    )
    assert set(outputs[:, -1].tolist()) == {0, 1}


def test_repetition_penalty_and_finished_rows(model, monkeypatch):
    scores = torch.full((23,), -100.0)
    scores[1], scores[2] = 4.0, 3.0
    fixed_logits(model, monkeypatch, scores)
    result = model.generate(
        torch.tensor([[1], [5]]), max_new_tokens=2, repetition_penalty=2.0, eos_token_id=5
    )
    assert result[0, 1].item() == 2
    assert result[1].tolist() == [5, 5, 5]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(top_p=0),
        dict(top_p=1.1),
        dict(top_p=float("nan")),
        dict(repetition_penalty=0),
        dict(num_return_sequences=0),
        dict(num_return_sequences=True),
    ],
)
def test_invalid_extended_generation(model, kwargs):
    with pytest.raises(ValueError):
        model.generate(torch.tensor([[1]]), max_new_tokens=1, **kwargs)


def test_sparse_experts_matches_explicit_reference():
    c = config(num_experts=3, experts_per_token=2, router_aux_loss_coef=0.03)
    module = SparseExperts(c)
    x = torch.randn(2, 3, c.hidden_size, requires_grad=True)
    actual, auxiliary = module(x)
    flat = x.reshape(-1, c.hidden_size)
    probabilities = module.router(flat).float().softmax(-1)
    weights, indices = probabilities.topk(2, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    expected = torch.stack(
        [
            sum(module.experts[indices[i, j].item()](flat[i]) * weights[i, j] for j in range(2))
            for i in range(len(flat))
        ]
    ).view_as(x)
    load = F.one_hot(indices, 3).float().mean((0, 1))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(auxiliary, 3 * (load * probabilities.mean(0)).sum() * 0.03)
    (actual.square().mean() + auxiliary).backward()
    assert torch.isfinite(x.grad).all()
    assert module.router.weight.grad.abs().sum() > 0


def test_moe_loss_and_cached_parity():
    model = CausalLM(config(num_experts=3, experts_per_token=2)).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    output = model(ids, ids)
    ce = F.cross_entropy(output.logits[:, :-1].reshape(-1, 23), ids[:, 1:].reshape(-1))
    torch.testing.assert_close(output.loss, ce + output.aux_loss)
    output.loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    cache = model(ids[:, :2], use_cache=True).past_key_values
    torch.testing.assert_close(
        model(ids[:, 2:], past_key_values=cache).logits, output.logits[:, 2:], atol=1e-7, rtol=1e-5
    )


def test_default_dense_state_keys_and_rotary_unchanged():
    base = config()
    explicit = config(num_experts=0, yarn_factor=1)
    torch.manual_seed(11)
    first = CausalLM(base)
    torch.manual_seed(11)
    second = CausalLM(explicit)
    assert first.state_dict().keys() == second.state_dict().keys()
    assert not any("expert" in name or "router" in name for name in first.state_dict())
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
    x = torch.randn(1, 2, 5, 4)
    torch.testing.assert_close(
        RotaryEmbedding(4, base.rope_theta)(x),
        RotaryEmbedding(4, base.rope_theta, base)(x),
        rtol=0,
        atol=0,
    )


def test_yarn_independent_frequency_reference_and_offsets():
    c = config(hidden_size=64, yarn_factor=4, yarn_original_context=128)
    x = torch.randn(1, 4, 6, 16)
    rotary = RotaryEmbedding(16, c.rope_theta, c)
    actual = rotary(x, offset=3)
    low = max(0, math.floor(16 * math.log(128 / (32 * 2 * math.pi)) / (2 * math.log(10000))))
    high = min(7, math.ceil(16 * math.log(128 / (2 * math.pi)) / (2 * math.log(10000))))
    pairs = []
    for i in range(8):
        ramp = min(1, max(0, (i - low) / max(high - low, 0.001)))
        frequency = 10000 ** (-2 * i / 16) * (1 - ramp + ramp / 4)
        angle = torch.arange(3, 9) * frequency
        cosine, sine = angle.cos() * (1 + 0.1 * math.log(4)), angle.sin() * (1 + 0.1 * math.log(4))
        pairs.extend(
            [
                x[..., 2 * i] * cosine - x[..., 2 * i + 1] * sine,
                x[..., 2 * i] * sine + x[..., 2 * i + 1] * cosine,
            ]
        )
    torch.testing.assert_close(actual, torch.stack(pairs, -1))
    torch.testing.assert_close(actual[..., 2:, :], rotary(x[..., 2:, :], offset=5))


def test_cpu_autocast_cache_dtype_and_continuation(model):
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        full = model(ids).logits
        cache = model(ids[:, :2], use_cache=True).past_key_values
        assert cache[0][0].dtype == torch.bfloat16
        actual = model(ids[:, 2:], past_key_values=cache).logits
    torch.testing.assert_close(actual, full[:, 2:], atol=0.002, rtol=0.02)
    with pytest.raises(ValueError, match="dtype"):
        model(ids[:, 2:], past_key_values=cache)


@pytest.mark.parametrize("value", [-1, 5, True, 1.5])
def test_invalid_logits_projection_count(model, value):
    with pytest.raises(ValueError, match="logits_to_keep"):
        model(torch.tensor([[1, 2]]), logits_to_keep=value)
