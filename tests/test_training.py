import dataclasses
import json
import os
import signal

import numpy as np
import pytest
import torch

from clearforge.checkpoint import atomic_save, capture_rng_state, load_checkpoint, publish_latest
from clearforge.config import ModelConfig, RunConfig, TrainConfig
from clearforge.data import PackedDataset, SourceSpec, pack_dataset, prepare_sources
from clearforge.model import CausalLM
from clearforge.tokenizer import ChatTokenizer, train_tokenizer
from clearforge.training import (
    _run_lock,
    _stop_signals,
    evaluate,
    export,
    learning_rate_factor,
    train,
)


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(2)
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": f"English document {i} " + "small useful text " * (i % 5 + 1)})
            + "\n"
            for i in range(24)
        )
    )
    prepared = prepare_sources(
        [SourceSpec(source, "CC0")],
        tmp_path / "prepared",
        validation_fraction=0.25,
        seed=8,
        mode="pretrain",
    )
    tokenizer = train_tokenizer(prepared / "train.jsonl", tmp_path / "tokenizer", vocab_size=280)
    packed = pack_dataset(prepared, tokenizer, tmp_path / "packed", seq_length=12, mode="pretrain")
    return RunConfig(
        ModelConfig(
            vocab_size=ChatTokenizer.load(tokenizer).vocab_size,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=32,
            max_seq_len=12,
            dropout=0.25,
        ),
        TrainConfig(
            max_steps=6,
            warmup_steps=2,
            batch_size=3,
            accumulation_steps=2,
            device="cpu",
            precision="fp32",
            save_every=1,
            eval_every=2,
            eval_batches=2,
        ),
        packed,
        tokenizer,
        tmp_path / "run",
    )


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal_tree(x, y)
    else:
        assert a == b


def test_exact_resume_with_dropout_rng_and_metrics(config):
    full = train(config)
    other = dataclasses.replace(config, output_dir=config.output_dir.parent / "resumed")
    paused = train(other, stop_after_steps=3)
    assert paused.status == "paused"
    # Simulate a durable journal suffix newer than the selected checkpoint.
    with (other.output_dir / "metrics.jsonl").open("ab") as stream:
        stream.write(b'{"stale":true}\n')
    resumed = train(other, resume=other.output_dir / "latest.pt")
    assert resumed.status == "complete" and resumed.step == 6
    a, b = load_checkpoint(full.checkpoint), load_checkpoint(resumed.checkpoint)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "data_cursor", "metrics"):
        equal_tree(a[key], b[key])
    assert (config.output_dir / "metrics.jsonl").read_bytes() == (
        other.output_dir / "metrics.jsonl"
    ).read_bytes()
    assert len(list(other.output_dir.joinpath("checkpoints").glob("step-*.pt"))) == 3
    assert (other.output_dir / "latest.pt").stat().st_ino == resumed.checkpoint.stat().st_ino


def test_token_weighted_accumulation_partial_epoch(config):
    size = len(PackedDataset(config.data_dir, "train"))
    settings = dataclasses.replace(
        config.training,
        max_steps=2,
        warmup_steps=0,
        batch_size=7,
        accumulation_steps=(size + 6) // 7,
    )
    small = dataclasses.replace(
        config, model=dataclasses.replace(config.model, dropout=0), training=settings
    )
    large = dataclasses.replace(
        small,
        output_dir=config.output_dir.parent / "large",
        training=dataclasses.replace(settings, batch_size=size, accumulation_steps=1),
    )
    a, b = load_checkpoint(train(small).checkpoint), load_checkpoint(train(large).checkpoint)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=1e-4, atol=1e-6)
    assert a["data_cursor"]["epoch"] == 1


@pytest.mark.parametrize("corruption", ["schedule", "negative_moment", "shape"])
def test_corrupt_state_rejected_before_journal_mutation(config, corruption):
    result = train(config, stop_after_steps=2)
    payload = load_checkpoint(result.checkpoint)
    if corruption == "schedule":
        payload["scheduler"]["base_lrs"] = [0.99]
    else:
        moment = next(iter(payload["optimizer"]["state"].values()))
        moment["exp_avg_sq"] = (
            torch.tensor([-1.0])
            if corruption == "shape"
            else -torch.ones_like(moment["exp_avg_sq"])
        )
    atomic_save(payload, result.checkpoint)
    publish_latest(result.checkpoint, config.output_dir / "latest.pt")
    before = (config.output_dir / "metrics.jsonl").read_bytes()
    with pytest.raises(ValueError):
        train(config, resume=result.checkpoint)
    assert (config.output_dir / "metrics.jsonl").read_bytes() == before


def test_preflight_collision_identity_and_run_lock(config):
    result = train(config, stop_after_steps=1)
    before = (config.output_dir / "metrics.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        train(config)
    changed = dataclasses.replace(
        config, training=dataclasses.replace(config.training, learning_rate=0.02)
    )
    with pytest.raises(ValueError, match="configuration"):
        train(changed, resume=result.checkpoint)
    with _run_lock(config.output_dir):
        with pytest.raises(RuntimeError, match="another"):
            train(config, resume=result.checkpoint)
    assert (config.output_dir / "metrics.jsonl").read_bytes() == before
    with config.tokenizer_path.open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="tokenizer"):
        train(config, resume=result.checkpoint)


def test_evaluation_preserves_rng_and_export_matches_native_model(config):
    result = train(config, stop_after_steps=1)
    before = capture_rng_state()
    metrics = evaluate(config, result.checkpoint)
    equal_tree(before, capture_rng_state())
    assert metrics["loss"] > 0
    out = export(config, result.checkpoint, config.output_dir.parent / "bundle")
    model = CausalLM(config.model).eval()
    model.load_state_dict(torch.load(out / "model.pt", weights_only=True))
    original = CausalLM(config.model).eval()
    original.load_state_dict(load_checkpoint(result.checkpoint)["model"])
    ids = PackedDataset(config.data_dir, "validation")[0]["input_ids"][None]
    assert torch.equal(model(ids).logits, original(ids).logits)
    assert ChatTokenizer.load(out / "tokenizer.json").vocab_size == config.model.vocab_size
    with pytest.raises(FileExistsError):
        export(config, result.checkpoint, out)


def test_stop_signal_and_cosine_boundaries():
    original = signal.getsignal(signal.SIGINT)
    with _stop_signals() as stop:
        os.kill(os.getpid(), signal.SIGINT)
        assert stop["requested"]
    assert signal.getsignal(signal.SIGINT) == original
    factors = [learning_rate_factor(i, max_steps=6, warmup_steps=2, minimum=0.1) for i in range(6)]
    assert factors[:3] == [0.5, 1.0, 1.0]
    assert factors[-1] == 0.1


def test_sft_transfers_weights_but_resets_optimizer_and_data(config):
    base = train(config, stop_after_steps=2)
    source = config.output_dir.parent / "chat.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": f"Question {i}"},
                        {"role": "assistant", "content": f"Useful answer {i}"},
                    ]
                }
            )
            + "\n"
            for i in range(24)
        )
    )
    prep = prepare_sources(
        [SourceSpec(source, "CC0")],
        source.parent / "sft-prepared",
        validation_fraction=0.25,
        seed=8,
        mode="sft",
    )
    packed = pack_dataset(
        prep, config.tokenizer_path, source.parent / "sft-packed", seq_length=12, mode="sft"
    )
    sft = dataclasses.replace(
        config,
        stage="sft",
        init_checkpoint=base.checkpoint,
        data_dir=packed,
        output_dir=source.parent / "sft-run",
    )
    start = train(sft, stop_after_steps=0)
    a, b = load_checkpoint(base.checkpoint), load_checkpoint(start.checkpoint)
    equal_tree(a["model"], b["model"])
    assert b["step"] == b["data_cursor"]["position"] == 0
    assert b["optimizer"]["state"] == {}
    trained = load_checkpoint(train(sft, resume=start.checkpoint, stop_after_steps=1).checkpoint)
    assert any(not torch.equal(trained["model"][k], a["model"][k]) for k in a["model"])
    equal_tree(a["model"], load_checkpoint(base.checkpoint)["model"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_cuda_mixed_precision_exact_resume(config, precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support bf16")
    cfg = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training, device="cuda", precision=precision, max_steps=3, warmup_steps=1
        ),
    )
    full = train(cfg)
    resumed_cfg = dataclasses.replace(cfg, output_dir=cfg.output_dir.parent / "cuda-resumed")
    pause = train(resumed_cfg, stop_after_steps=1)
    resumed = train(resumed_cfg, resume=pause.checkpoint)
    a, b = load_checkpoint(full.checkpoint), load_checkpoint(resumed.checkpoint)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "data_cursor", "metrics"):
        equal_tree(a[key], b[key])


@pytest.mark.parametrize(
    "settings",
    [
        {"activation_checkpointing": True},
        {"data_workers": 2},
        {"compile_model": True, "compile_backend": "eager"},
    ],
)
def test_execution_options_preserve_parameter_updates(config, settings):
    base = train(config)
    adjusted = dataclasses.replace(
        config,
        output_dir=config.output_dir.parent / "optimized",
        training=dataclasses.replace(config.training, **settings),
    )
    result = train(adjusted)
    expected = load_checkpoint(base.checkpoint)
    actual = load_checkpoint(result.checkpoint)
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value, rtol=0, atol=0)
    assert actual["step"] == expected["step"]


def test_new_pretraining_and_continued_sft_from_weights(config):
    base = train(config)
    continued = dataclasses.replace(
        config, init_checkpoint=base.checkpoint, output_dir=config.output_dir.parent / "continued"
    )
    start = train(continued, stop_after_steps=0)
    initial = load_checkpoint(start.checkpoint)
    original = load_checkpoint(base.checkpoint)
    assert initial["step"] == 0 and initial["optimizer"]["state"] == {}
    for key in original["model"]:
        assert torch.equal(initial["model"][key], original["model"][key])
    # Previous checkpoints can omit architecture fields that now have defaults.
    original["config"]["model"].pop("num_experts")
    legacy = config.output_dir.parent / "legacy.pt"
    atomic_save(original, legacy)
    older = dataclasses.replace(
        continued, init_checkpoint=legacy, output_dir=config.output_dir.parent / "legacy-start"
    )
    assert train(older, stop_after_steps=0).step == 0


def test_epoch_budget_and_progress(config, capsys):
    adjusted = dataclasses.replace(
        config, training=dataclasses.replace(config.training, epochs=2, warmup_steps=0, log_every=1)
    )
    result = train(adjusted)
    rows = len(PackedDataset(config.data_dir, "train"))
    expected_steps = 2 * __import__("math").ceil(
        rows / (config.training.batch_size * config.training.accumulation_steps)
    )
    assert result.step == expected_steps
    progress = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert progress[-1]["eta_seconds"] == 0
    assert progress[-1]["tokens_per_second"] > 0
    assert train(adjusted, resume=adjusted.output_dir / "latest.pt").step == expected_steps


@pytest.mark.parametrize(
    "settings",
    [
        {"data_workers": -1},
        {"data_workers": True},
        {"pin_memory": 1},
        {"epochs": 0},
        {"compile_model": "yes"},
        {"activation_checkpointing": 1},
        {"compile_backend": "unknown"},
        {"log_every": -1},
    ],
)
def test_invalid_execution_controls(config, settings):
    with pytest.raises(ValueError):
        dataclasses.replace(config.training, **settings)


def test_pause_boundary_uses_derived_epoch_horizon(config):
    cfg = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            max_steps=1,
            warmup_steps=0,
            epochs=1,
            batch_size=1,
            accumulation_steps=1,
        ),
    )
    result = train(cfg, stop_after_steps=2)
    assert result.step == 2 and result.status == "paused"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_inductor_checkpointed_dropout_exact_resume(config):
    import sysconfig
    from pathlib import Path

    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA bf16 unavailable")
    includes = [sysconfig.get_path("include"), *os.environ.get("CPATH", "").split(os.pathsep)]
    if not any((Path(path) / "Python.h").is_file() for path in includes if path):
        pytest.skip("Inductor requires Python development headers")
    cfg = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training,
            device="cuda",
            precision="bf16",
            max_steps=3,
            warmup_steps=1,
            compile_model=True,
            compile_backend="inductor",
            activation_checkpointing=True,
        ),
    )
    full = train(cfg)
    other = dataclasses.replace(cfg, output_dir=cfg.output_dir.parent / "compiled-resumed")
    paused = train(other, stop_after_steps=1)
    resumed = train(other, resume=paused.checkpoint)
    a, b = load_checkpoint(full.checkpoint), load_checkpoint(resumed.checkpoint)
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "data_cursor", "metrics"):
        equal_tree(a[key], b[key])
    assert (cfg.output_dir / "metrics.jsonl").read_bytes() == (
        other.output_dir / "metrics.jsonl"
    ).read_bytes()
