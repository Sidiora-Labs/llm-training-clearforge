"""Actual two-process Gloo integration, including uneven tails and dropout resume."""

import dataclasses
import json
import os
import socket
import traceback
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from clearforge.checkpoint import load_checkpoint
from clearforge.config import ModelConfig, RunConfig, TrainConfig
from clearforge.data import PackedDataset, SourceSpec, pack_dataset, prepare_sources
from clearforge.tokenizer import ChatTokenizer, train_tokenizer
from clearforge.training import train


def worker(rank, port, config, operation, result_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        CUDA_VISIBLE_DEVICES="",
    )
    torch.set_num_threads(1)
    try:
        if operation == "mismatch" and rank == 1:
            config = dataclasses.replace(
                config, training=dataclasses.replace(config.training, seed=43)
            )
        if operation == "resume":
            result = train(config, resume=config.output_dir / "latest.pt")
        else:
            result = train(config, stop_after_steps=1 if operation == "pause" else None)
        value = {"status": result.status, "step": result.step}
    except Exception:
        value = {"error": traceback.format_exc()}
    Path(result_dir, f"{rank}.json").write_text(json.dumps(value))


def launch(config, operation="full"):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    results = config.output_dir.parent / f"{config.output_dir.name}-{operation}-results"
    results.mkdir()
    context = mp.spawn(worker, args=(port, config, operation, str(results)), nprocs=2, join=False)
    import time

    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        if context.join(timeout=0.2):
            break
    else:
        for process in context.processes:
            process.terminate()
        for process in context.processes:
            process.join(5)
        pytest.fail("distributed workers did not terminate within 45 seconds")
    return [json.loads((results / f"{rank}.json").read_text()) for rank in range(2)]


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(1)
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"text": f"English document {i} " + "useful text " * (i % 3 + 1)}) + "\n"
            for i in range(15)
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
            dropout=0,
        ),
        TrainConfig(
            max_steps=3,
            warmup_steps=1,
            batch_size=2,
            accumulation_steps=2,
            device="cpu",
            precision="fp32",
            save_every=1,
            eval_every=2,
            eval_batches=2,
            log_every=0,
        ),
        packed,
        tokenizer,
        tmp_path / "ddp",
    )


def test_global_token_weighted_updates_and_empty_rank_tail(config):
    # One global microbatch spans the entire dataset, with rank one empty on
    # the final microbatch whenever the dataset is smaller than the per-rank batch.
    size = len(PackedDataset(config.data_dir, "train"))
    config = dataclasses.replace(
        config,
        training=dataclasses.replace(
            config.training, batch_size=size // 3 + 1, accumulation_steps=2
        ),
    )
    results = launch(config)
    assert all(x.get("status") == "complete" for x in results), results
    baseline = dataclasses.replace(
        config,
        output_dir=config.output_dir.parent / "single",
        training=dataclasses.replace(config.training, batch_size=config.training.batch_size * 2),
    )
    single = load_checkpoint(train(baseline).checkpoint)
    distributed = load_checkpoint(config.output_dir / "latest.pt")
    assert distributed["identities"]["distributed"]["world_size"] == 2
    assert len(distributed["rng_by_rank"]) == 2
    assert all(not name.startswith("module.") for name in distributed["model"])
    for name in single["model"]:
        torch.testing.assert_close(
            single["model"][name], distributed["model"][name], rtol=1e-4, atol=2e-6
        )
    assert distributed["data_cursor"] == single["data_cursor"]


def test_dropout_exact_resume(config):
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, dropout=0.25))
    assert all(r.get("status") == "complete" for r in launch(config))
    resumed = dataclasses.replace(config, output_dir=config.output_dir.parent / "resume")
    assert all(r.get("status") == "paused" for r in launch(resumed, "pause"))
    results = launch(resumed, "resume")
    assert all(r.get("status") == "complete" for r in results), results
    a, b = (
        load_checkpoint(config.output_dir / "latest.pt"),
        load_checkpoint(resumed.output_dir / "latest.pt"),
    )
    for name in a["model"]:
        assert torch.equal(a["model"][name], b["model"][name])
    assert a["data_cursor"] == b["data_cursor"]
    assert (config.output_dir / "metrics.jsonl").read_bytes() == (
        resumed.output_dir / "metrics.jsonl"
    ).read_bytes()


def test_rank_mismatch_fails_collectively_without_output(config):
    results = launch(config, "mismatch")
    assert all("rank configurations" in result.get("error", "") for result in results), results
    assert not config.output_dir.exists()


def test_sparse_expert_resume(config):
    config = dataclasses.replace(
        config, model=dataclasses.replace(config.model, num_experts=8, experts_per_token=1)
    )
    paused = launch(config, "pause")
    assert all(r.get("status") == "paused" for r in paused), paused
    resumed = launch(config, "resume")
    assert all(r.get("status") == "complete" for r in resumed), resumed


def test_changed_world_size_checkpoint_rejected_on_all_ranks(config):
    from clearforge.checkpoint import atomic_save, publish_latest

    assert all(r.get("status") == "paused" for r in launch(config, "pause"))
    latest = config.output_dir / "latest.pt"
    payload = load_checkpoint(latest)
    payload["identities"]["distributed"]["world_size"] = 3
    canonical = config.output_dir / "checkpoints" / "step-00000001.pt"
    atomic_save(payload, canonical)
    publish_latest(canonical, latest)
    before = (config.output_dir / "metrics.jsonl").read_bytes()
    results = launch(config, "resume")
    assert all("world size differs" in r.get("error", "") for r in results), results
    assert (config.output_dir / "metrics.jsonl").read_bytes() == before
