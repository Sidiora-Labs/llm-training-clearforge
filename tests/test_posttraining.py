import copy
import dataclasses
import json

import pytest
import torch

from clearforge.adapters import AdapterConfig
from clearforge.cli import _load_bundle
from clearforge.config import ModelConfig
from clearforge.data import SourceSpec, prepare_sources
from clearforge.model import CausalLM
from clearforge.objectives import sequence_log_probabilities
from clearforge.posttraining import PostTrainConfig, _batch, _encode, train_posttraining
from clearforge.tokenizer import ChatTokenizer, train_tokenizer


@pytest.fixture
def setup(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(8)
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"text": "Question answer useful helpful harmful text " * 20}) + "\n"
    )
    prepared = prepare_sources(
        [SourceSpec(source, "CC0")],
        tmp_path / "prepared",
        validation_fraction=0,
        seed=1,
        mode="pretrain",
    )
    tok = ChatTokenizer.load(
        train_tokenizer(prepared / "train.jsonl", tmp_path / "tokenizer", vocab_size=280)
    )
    model = CausalLM(
        ModelConfig(
            vocab_size=tok.vocab_size,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=32,
            max_seq_len=128,
            dropout=0.2,
        )
    )
    cfg = PostTrainConfig(tmp_path / "run", steps=3, learning_rate=0.005, max_new_tokens=4)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "helpful answer"},
            ]
        }
    ]
    return model, tok, cfg, rows


def test_lora_changes_only_adapters_and_exports_native(setup):
    model, tok, cfg, rows = setup
    original = copy.deepcopy(model.state_dict())
    result = train_posttraining(
        model,
        tok,
        rows,
        cfg,
        objective="lora",
        adapter_config=AdapterConfig(rank=2, alpha=4, dropout=0.1),
    )
    for name, value in model.state_dict().items():
        assert torch.equal(value, original[name])
    for name, value in result.model.state_dict().items():
        if ".lora_" not in name:
            assert torch.equal(value, original[name.replace(".base.", ".")])
    assert any(
        torch.count_nonzero(p) for n, p in result.model.named_parameters() if n.endswith(".lora_b")
    )
    exported = CausalLM(model.config)
    exported.load_state_dict(torch.load(cfg.output_path / "model/model.pt", weights_only=True))
    loaded, loaded_tok = _load_bundle(cfg.output_path / "model", torch.device("cpu"))
    prompt = torch.tensor([loaded_tok.encode("Question")])
    assert loaded.generate(prompt, max_new_tokens=2, temperature=0).shape[1] == prompt.shape[1] + 2
    assert result.status == "complete"


def test_dpo_improves_chosen_margin(setup):
    model, tok, cfg, _ = setup
    model.config = dataclasses.replace(model.config, dropout=0)
    rows = [{"prompt": "Question", "chosen": "helpful answer", "rejected": "harmful text"}]
    encoded = _encode(rows[0], tok, "dpo")

    def margin(policy):
        policy.eval()
        scores = []
        for row in encoded:
            ids, labels = _batch([row], tok, "cpu")
            scores.append(sequence_log_probabilities(policy(ids).logits, labels).item())
        return scores[0] - scores[1]

    before = margin(model)
    result = train_posttraining(
        model, tok, rows, dataclasses.replace(cfg, steps=10), objective="dpo"
    )
    assert margin(result.model) > before


def test_distillation_freezes_teacher_and_checks_tokenizer(setup):
    model, tok, cfg, rows = setup
    teacher = copy.deepcopy(model)
    with torch.no_grad():
        teacher.lm_head.weight.add_(0.01)
    original = copy.deepcopy(teacher.state_dict())
    result = train_posttraining(
        model, tok, rows, cfg, objective="distillation", teacher=teacher, teacher_tokenizer=tok
    )
    assert result.step == 3
    assert all(torch.equal(v, original[k]) for k, v in teacher.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())
    with pytest.raises(ValueError, match="identical tokenizer"):
        train_posttraining(
            model,
            tok,
            rows,
            dataclasses.replace(cfg, output_path=cfg.output_path.parent / "bad"),
            objective="distillation",
            teacher=teacher,
        )


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


@pytest.mark.parametrize(
    "objective", ["sft", "lora", "dpo", "distillation", "ppo", "grpo", "cispo"]
)
def test_updates_and_exact_resume(setup, objective):
    model, tok, cfg, rows = setup
    if objective in {"ppo", "grpo", "cispo"}:
        rows = [{"prompt": "Question"}]

    def reward(prompt, response):
        return sum(ord(c) for c in response) % 17 / 17

    kwargs = {"objective": objective}
    if objective == "dpo":
        rows = [{"prompt": "Question", "chosen": "helpful answer", "rejected": "harmful text"}]
    if objective == "distillation":
        kwargs.update(teacher=copy.deepcopy(model), teacher_tokenizer=tok)
    if objective in {"ppo", "grpo", "cispo"}:
        kwargs.update(reward_fn=reward, reward_id="unicode-mod17-v1")
    if objective == "lora":
        kwargs["adapter_config"] = AdapterConfig(rank=2, dropout=0.2)
    full = train_posttraining(model, tok, rows, cfg, **kwargs)
    cfg2 = dataclasses.replace(cfg, output_path=cfg.output_path.parent / "resumed")
    pause = train_posttraining(model, tok, rows, cfg2, stop_after_steps=1, **kwargs)
    resumed = train_posttraining(model, tok, rows, cfg2, resume=pause.checkpoint, **kwargs)
    same(full.model.state_dict(), resumed.model.state_dict())
    a = torch.load(full.checkpoint, weights_only=False)
    b = torch.load(resumed.checkpoint, weights_only=False)
    for key in ("optimizer", "cursor", "metrics", "critic"):
        same(a[key], b[key])
    assert all(torch.isfinite(p).all() for p in full.model.parameters())
    if objective != "lora":
        assert any(not torch.equal(a["policy"][k], v) for k, v in model.state_dict().items())


def test_rejects_missing_reward_and_overlength_without_output(setup):
    model, tok, cfg, rows = setup
    with pytest.raises(ValueError, match="reward_fn"):
        train_posttraining(model, tok, [{"prompt": "Question"}], cfg, objective="grpo")
    rows[0]["messages"][0]["content"] = "Question " * 200
    with pytest.raises(ValueError, match="context"):
        train_posttraining(model, tok, rows, cfg, objective="sft")
    assert not cfg.output_path.exists()


def test_sparse_expert_posttraining_resume(setup):
    model, tok, cfg, rows = setup
    model = CausalLM(dataclasses.replace(model.config, num_experts=4, experts_per_token=1))
    full = train_posttraining(model, tok, rows, cfg, objective="sft")
    cfg2 = dataclasses.replace(cfg, output_path=cfg.output_path.parent / "sparse-resumed")
    paused = train_posttraining(model, tok, rows, cfg2, objective="sft", stop_after_steps=1)
    resumed = train_posttraining(model, tok, rows, cfg2, objective="sft", resume=paused.checkpoint)
    same(full.model.state_dict(), resumed.model.state_dict())
