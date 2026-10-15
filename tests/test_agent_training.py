"""Train a tiny native tool policy, then exercise actual sampled agent updates."""

import copy
import dataclasses
import json

import pytest
import torch

from clearforge.agent_training import AgentTrainConfig, collect_agent_group, train_agent_policy
from clearforge.config import ModelConfig
from clearforge.data import SourceSpec, prepare_sources
from clearforge.model import CausalLM
from clearforge.posttraining import _batch
from clearforge.rollout import EpisodeLimits, ToolRegistry
from clearforge.tokenizer import ChatTokenizer, train_tokenizer

DEFINITION = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add integers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}
PROMPT = [{"role": "user", "content": "What is 2 plus 3?"}]


def add_integers(a, b):
    return a + b


def registry():
    tools = ToolRegistry()
    tools.register(DEFINITION, add_integers)
    return tools


@pytest.fixture(scope="module")
def trained_policy(tmp_path_factory):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    root = tmp_path_factory.mktemp("learned-tool-policy")
    conversations = []
    # Two plausible actions under one prompt make a genuine stochastic policy;
    # reinforcement rewards the mathematically correct action after execution.
    for operand in (3, 4):
        conversations.append(
            {
                "tools": [DEFINITION],
                "messages": PROMPT
                + [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "sum1",
                                "type": "function",
                                "function": {
                                    "name": "add",
                                    "arguments": json.dumps({"a": 2, "b": operand}),
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "sum1", "content": str(2 + operand)},
                    {"role": "assistant", "content": str(2 + operand)},
                ],
            }
        )
    source = root / "source.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in conversations))
    prepared = prepare_sources(
        [SourceSpec(source, "CC0")], root / "prepared", validation_fraction=0, seed=1, mode="sft"
    )
    tokenizer = ChatTokenizer.load(
        train_tokenizer(prepared / "train.jsonl", root / "tokenizer", vocab_size=512)
    )
    model = CausalLM(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=48,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=96,
            max_seq_len=512,
            dropout=0,
        )
    )
    batch = _batch(
        [tokenizer.encode_chat(row["messages"], tools=row["tools"]) for row in conversations],
        tokenizer,
        "cpu",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008)
    for step in range(240):
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch[0], batch[1]).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step > 70 and loss.item() < 0.025:
            break
    assert loss.item() < 0.04, "tiny policy must actually learn valid tool conversations"
    return model.eval(), tokenizer


def reward(record, episode):
    return float(episode.status == "complete" and episode.messages[-1].get("content") == "5")


def configuration(tmp_path):
    return AgentTrainConfig(
        tmp_path / "run",
        steps=2,
        learning_rate=0.0005,
        beta=0.01,
        samples_per_prompt=4,
        max_new_tokens=90,
        sampling_temperature=0.8,
        policy_epochs=2,
        episode_limits=EpisodeLimits(max_turns=3, max_generated_tokens=180, max_seconds=60),
    )


def test_real_tool_episode_masks_observations_and_prior_context(trained_policy, tmp_path):
    model, tokenizer = trained_policy
    reference = copy.deepcopy(model).requires_grad_(False)
    cfg = configuration(tmp_path)
    torch.manual_seed(42)
    ids, labels, old, ref, mask, advantages, summary = collect_agent_group(
        model, reference, tokenizer, [{"messages": PROMPT}], registry(), cfg, reward
    )
    assert summary["tool_calls"] >= 2, summary
    assert summary["assistant_turns"] >= 6, summary
    assert summary["sampled_tokens"] == int(mask.sum())
    for tokens, targets in zip(ids.tolist(), labels.tolist()):
        start = next(i for i, target in enumerate(targets) if target != -100)
        assert all(target == -100 for target in targets[:start])
        prefix = tokenizer.decode(tokens[:start])
        # At later turns, the real tool result is part of the excluded prefix.
        if "<clearforge_tool>" in prefix:
            assert "sum1" in prefix
        for token, target in zip(tokens[start:], targets[start:]):
            assert target == -100 or token == target
    assert any(
        "<clearforge_tool>"
        in tokenizer.decode(row[: next(i for i, t in enumerate(target) if t != -100)])
        for row, target in zip(ids.tolist(), labels.tolist())
    )
    assert torch.equal(old, ref)
    assert torch.isfinite(advantages).all()


@pytest.mark.parametrize("objective", ["grpo", "cispo"])
def test_actual_multiturn_updates_and_exact_resume(trained_policy, tmp_path, objective):
    model, tokenizer = trained_policy
    original = copy.deepcopy(model.state_dict())
    cfg = configuration(tmp_path)
    data = [{"messages": PROMPT}]
    kwargs = dict(
        registry=registry(),
        registry_id="add-integers-v1",
        reward_fn=reward,
        reward_id="answer-five-v1",
        objective=objective,
    )
    full = train_agent_policy(model, tokenizer, data, cfg, **kwargs)
    resumed_cfg = dataclasses.replace(cfg, output_path=tmp_path / "resumed")
    paused = train_agent_policy(model, tokenizer, data, resumed_cfg, stop_after_steps=1, **kwargs)
    resumed = train_agent_policy(
        model, tokenizer, data, resumed_cfg, resume=paused.checkpoint, **kwargs
    )
    assert full.status == resumed.status == "complete"
    assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
    assert any(
        not torch.equal(value, original[name]) for name, value in full.model.state_dict().items()
    )
    for name, value in full.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name])
    assert (cfg.output_path / "metrics.jsonl").read_bytes() == (
        resumed_cfg.output_path / "metrics.jsonl"
    ).read_bytes()
    metrics = [
        json.loads(line) for line in (cfg.output_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert sum(row["tool_calls"] for row in metrics) >= 4
    assert (cfg.output_path / "model/model.pt").exists()
    with pytest.raises(ValueError, match="identity"):
        train_agent_policy(
            model,
            tokenizer,
            data,
            cfg,
            resume=full.checkpoint,
            **{**kwargs, "registry_id": "changed"},
        )


def test_unregistered_tools_and_missing_reward_rejected(trained_policy, tmp_path):
    model, tokenizer = trained_policy
    cfg = configuration(tmp_path)
    with pytest.raises(ValueError, match="schemas"):
        train_agent_policy(
            model,
            tokenizer,
            [{"messages": PROMPT, "tools": []}],
            cfg,
            registry=registry(),
            registry_id="a",
            reward_fn=reward,
            reward_id="a",
        )
    assert not cfg.output_path.exists()


@pytest.mark.parametrize("corruption", ["betas", "initial_moments"])
def test_corrupt_resume_rejected_before_journal_changes(trained_policy, tmp_path, corruption):
    from clearforge.checkpoint import atomic_save

    model, tokenizer = trained_policy
    cfg = configuration(tmp_path)
    kwargs = dict(registry=registry(), registry_id="add-v1", reward_fn=reward, reward_id="five-v1")
    result = train_agent_policy(
        model, tokenizer, [{"messages": PROMPT}], cfg, stop_after_steps=0, **kwargs
    )
    payload = torch.load(result.checkpoint, weights_only=False)
    if corruption == "betas":
        payload["optimizer"]["param_groups"][0]["betas"] = (0.1, 0.999)
    else:
        parameter = next(model.parameters())
        payload["optimizer"]["state"][0] = {
            "step": torch.tensor(0.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    atomic_save(payload, result.checkpoint)
    before = (cfg.output_path / "metrics.jsonl").read_bytes()
    with pytest.raises(ValueError, match="AdamW|initial checkpoint"):
        train_agent_policy(
            model, tokenizer, [{"messages": PROMPT}], cfg, resume=result.checkpoint, **kwargs
        )
    assert (cfg.output_path / "metrics.jsonl").read_bytes() == before
