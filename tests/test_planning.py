"""Capacity plans must be exact where they can be and conservative elsewhere."""

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from clearforge.cli import main
from clearforge.config import ModelConfig, RunConfig, TrainConfig, load_config
from clearforge.data import SourceSpec, pack_dataset, prepare_sources
from clearforge.model import CausalLM
from clearforge.planning import (
    GIB,
    checkpoint_estimate,
    memory_estimate,
    parameter_accounting,
    plan_run,
    summarize,
)
from clearforge.tokenizer import ChatTokenizer, train_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def small_model(**kwargs):
    return ModelConfig(
        **{
            "vocab_size": 300,
            "hidden_size": 32,
            "num_layers": 2,
            "num_heads": 4,
            "num_kv_heads": 2,
            "intermediate_size": 64,
            "max_seq_len": 16,
            **kwargs,
        }
    )


@pytest.mark.parametrize("kwargs", [{}, {"tie_embeddings": False}, {"num_experts": 4}])
def test_parameter_accounting_matches_allocated_model(kwargs):
    config = small_model(**kwargs)
    counts = parameter_accounting(config)
    model = CausalLM(config)
    assert counts["total"] == model.parameter_count()
    embedding = model.token_embedding.weight.numel()
    if not config.tie_embeddings:
        embedding += model.lm_head.weight.numel()
    assert counts["embedding"] == embedding
    assert counts["non_embedding"] == counts["total"] - embedding


def test_memory_components_sum_and_respond_to_controls(tmp_path):
    base = RunConfig(
        small_model(),
        TrainConfig(batch_size=2, device="cpu", precision="fp32"),
        tmp_path / "data",
        tmp_path / "tok.json",
        tmp_path / "run",
    )
    parameters = parameter_accounting(base.model)["total"]
    plain = memory_estimate(base, parameters=parameters, seq_length=16, device_type="cpu")
    assert plain["estimated_total_bytes"] == sum(
        value for key, value in plain.items() if key != "estimated_total_bytes"
    )
    assert plain["weights_bytes"] == plain["gradients_bytes"] == 4 * parameters
    assert plain["optimizer_bytes"] == 8 * parameters
    assert plain["autocast_weight_cache_bytes"] == 0
    checkpointed = dataclasses.replace(
        base, training=dataclasses.replace(base.training, activation_checkpointing=True)
    )
    less = memory_estimate(checkpointed, parameters=parameters, seq_length=16, device_type="cpu")
    assert less["activations_bytes"] < plain["activations_bytes"]
    mixed = dataclasses.replace(
        base, training=dataclasses.replace(base.training, precision="bf16", device="cuda")
    )
    half = memory_estimate(mixed, parameters=parameters, seq_length=16, device_type="cuda")
    assert half["autocast_weight_cache_bytes"] == 2 * parameters
    assert half["activations_bytes"] < plain["activations_bytes"]
    assert half["logits_bytes"] == 2 * 16 * 300 * (2 + 12)
    bigger = memory_estimate(base, parameters=parameters, seq_length=16, device_type="cuda")
    assert bigger["activations_bytes"] < plain["activations_bytes"]


def test_checkpoint_estimate_is_a_lower_bound_on_real_checkpoints(tmp_path):
    torch.set_num_threads(2)
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": f"Document {i} " + "planning text " * (i % 4 + 1)}) + "\n"
            for i in range(20)
        )
    )
    prepared = prepare_sources(
        [SourceSpec(source, "CC0")],
        tmp_path / "prepared",
        validation_fraction=0.25,
        seed=3,
        mode="pretrain",
    )
    tokenizer = train_tokenizer(prepared / "train.jsonl", tmp_path / "tokenizer", vocab_size=280)
    packed = pack_dataset(prepared, tokenizer, tmp_path / "packed", seq_length=12, mode="pretrain")
    config = RunConfig(
        small_model(vocab_size=ChatTokenizer.load(tokenizer).vocab_size, max_seq_len=12),
        TrainConfig(
            max_steps=2,
            warmup_steps=1,
            batch_size=2,
            accumulation_steps=2,
            device="cpu",
            precision="fp32",
            save_every=1,
            eval_every=1,
            eval_batches=1,
        ),
        packed,
        tokenizer,
        tmp_path / "run",
    )
    from clearforge.training import train

    result = train(config)
    plan = json.loads((config.output_dir / "run.json").read_text())["plan"]
    estimate = checkpoint_estimate(
        plan["parameters"]["total"], plan["data"]["train_rows"], config.training.keep_checkpoints
    )
    assert estimate == plan["checkpoint"]
    assert result.checkpoint.stat().st_size <= estimate["bytes"]
    assert result.checkpoint.stat().st_size >= 12 * plan["parameters"]["total"]
    assert plan["data"]["epochs"] == pytest.approx(2 * 2 * 2 / plan["data"]["train_rows"])
    assert plan["data"]["train_supervised_tokens"] > 0
    assert plan["data"]["train_input_tokens"] >= plan["data"]["train_supervised_tokens"]


def test_plan_without_packed_data_uses_context_and_warns(tmp_path):
    config = RunConfig(
        small_model(),
        TrainConfig(max_steps=3, warmup_steps=1, batch_size=2, accumulation_steps=2, device="cpu"),
        tmp_path / "missing",
        tmp_path / "tok.json",
        tmp_path / "deep" / "run",
    )
    plan = plan_run(config, world_size=4, device_type="cuda", device_memory_bytes=1024)
    assert plan["data"] is None
    schedule = plan["schedule"]
    assert schedule["seq_length"] == 16
    assert schedule["sequences_per_update"] == 2 * 2 * 4
    assert schedule["tokens_per_update"] == 16 * 16
    assert schedule["total_tokens"] == 3 * 16 * 16
    assert schedule["training_flops"] == 6 * plan["parameters"]["total"] * schedule["total_tokens"]
    assert schedule["compute_optimal_tokens"] == 20 * plan["parameters"]["total"]
    assert plan["memory"]["device_type"] == "cuda"
    assert plan["memory"]["device_memory_bytes"] == 1024
    assert plan["memory"]["fits"] is False
    assert plan["disk"]["free_bytes"] > 0 and plan["disk"]["one_checkpoint_fits"] is True
    messages = " ".join(plan["warnings"])
    assert "tokens per parameter" in messages and "exceeds device memory" in messages
    assert "GiB" in summarize(plan)
    with pytest.raises(ValueError):
        plan_run(config, world_size=0)
    with pytest.raises(ValueError):
        plan_run(config, device_type="tpu")
    with pytest.raises(ValueError):
        plan_run(config, device_memory_bytes=0)


def test_shipped_billion_parameter_preset_plans_a_serious_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    plan = plan_run(
        load_config(ROOT / "configs" / "pretrain-1b.toml"),
        device_type="cuda",
        device_memory_bytes=80 * GIB,
    )
    assert plan["parameters"]["total"] == 1_041_852_416
    assert plan["schedule"]["tokens_per_parameter"] >= 20
    assert plan["schedule"]["tokens_per_update"] >= 2**18
    assert plan["memory"]["fits"] is True
    assert plan["memory"]["estimated_total_bytes"] > 16 * GIB
    assert plan["checkpoint"]["bytes"] > 11 * GIB
    assert not [w for w in plan["warnings"] if "tokens per parameter" in w or "small" in w]


def test_plan_command_reports_summary_and_rejects_bad_memory(capsys, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert (
        main(
            [
                "plan",
                str(ROOT / "configs" / "smoke.toml"),
                "--world-size",
                "2",
                "--device",
                "cpu",
                "--device-memory-gib",
                "0.5",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["summary"].startswith("106,816 parameters")
    assert report["schedule"]["world_size"] == 2
    assert report["memory"]["device_memory_bytes"] == GIB // 2
    assert report["memory"]["fits"] is True
    assert main(["plan", str(ROOT / "configs" / "smoke.toml"), "--device-memory-gib", "0"]) == 1
    assert "positive" in capsys.readouterr().err
