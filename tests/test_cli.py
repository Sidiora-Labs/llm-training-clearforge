"""Run the shipped offline workflow through the actual command entry point."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from clearforge.checkpoint import load_checkpoint
from clearforge.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_offline_pretrain_resume_sft_export_chat(tmp_path, monkeypatch, capsys):
    shutil.copytree(ROOT / "examples", tmp_path / "examples")
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    monkeypatch.chdir(tmp_path)

    def run(*args):
        status = main(["--threads", "2", *args])
        output = capsys.readouterr()
        assert status == 0, output.err
        return output.out

    run(
        "data",
        "prepare",
        "--source",
        "examples/pretrain.jsonl",
        "--license",
        "Apache-2.0",
        "--mode",
        "pretrain",
        "--output",
        "data/smoke-prepared",
        "--validation-fraction",
        "0.2",
    )
    run(
        "tokenizer",
        "train",
        "--input",
        "data/smoke-prepared/train.jsonl",
        "--output",
        "data/smoke-tokenizer",
        "--vocab-size",
        "512",
    )
    run(
        "data",
        "pack",
        "--input",
        "data/smoke-prepared",
        "--tokenizer",
        "data/smoke-tokenizer",
        "--output",
        "data/smoke-packed",
        "--seq-length",
        "64",
        "--mode",
        "pretrain",
    )
    paused = json.loads(run("train", "configs/smoke.toml", "--stop-after-steps", "4"))
    assert paused["status"] == "paused" and paused["step"] == 4
    finished = json.loads(
        run("train", "configs/smoke.toml", "--resume", "runs/smoke-pretrain/latest.pt")
    )
    assert finished["status"] == "complete" and finished["step"] == 8
    base = load_checkpoint("runs/smoke-pretrain/latest.pt")
    assert base["config"]["stage"] == "pretrain"
    metrics = json.loads(
        run("evaluate", "configs/smoke.toml", "--checkpoint", "runs/smoke-pretrain/latest.pt")
    )
    assert metrics["loss"] > 0 and metrics["tokens"] > 0
    run(
        "data",
        "prepare",
        "--source",
        "examples/chat.jsonl",
        "--license",
        "Apache-2.0",
        "--mode",
        "sft",
        "--output",
        "data/smoke-sft-prepared",
        "--validation-fraction",
        "0.2",
    )
    run(
        "data",
        "pack",
        "--input",
        "data/smoke-sft-prepared",
        "--tokenizer",
        "data/smoke-tokenizer",
        "--output",
        "data/smoke-sft-packed",
        "--seq-length",
        "64",
        "--mode",
        "sft",
    )
    sft = json.loads(run("train", "configs/smoke-sft.toml"))
    assert sft["status"] == "complete"
    tuned = load_checkpoint("runs/smoke-sft/latest.pt")
    assert tuned["config"]["stage"] == "sft" and tuned["run_id"] != base["run_id"]
    assert any(not torch.equal(tuned["model"][key], value) for key, value in base["model"].items())
    run(
        "export",
        "configs/smoke-sft.toml",
        "--checkpoint",
        "runs/smoke-sft/latest.pt",
        "--output",
        "runs/smoke-export",
    )
    assert (tmp_path / "runs/smoke-export/provenance.json").is_file()
    # Output quality is intentionally not asserted for an eight-update fixture model.
    run(
        "chat",
        "--model",
        "runs/smoke-export",
        "--prompt",
        "What is a library?",
        "--max-new-tokens",
        "8",
        "--device",
        "cpu",
    )
    assert (
        main(
            [
                "chat",
                "--model",
                "runs/smoke-export",
                "--prompt",
                "hello " * 100,
                "--max-new-tokens",
                "8",
                "--device",
                "cpu",
            ]
        )
        == 1
    )
    assert "exceeds" in capsys.readouterr().err
    # A fresh training invocation must never silently replace an existing run.
    assert main(["train", "configs/smoke.toml"]) == 1
    assert "exist" in capsys.readouterr().err


def test_module_entrypoint():
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(
        [sys.executable, "-m", "clearforge", "--version"],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "clearforge 0.1.0"


@pytest.mark.parametrize("name", ["pretrain-100m", "pretrain-300m", "pretrain-1b", "smoke"])
def test_doctor_counts_presets_without_allocation(name, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["doctor", "--config", str(ROOT / "configs" / f"{name}.toml")]) == 0
    report = json.loads(capsys.readouterr().out)
    expected = {
        "pretrain-100m": 100682496,
        "pretrain-300m": 304137216,
        "pretrain-1b": 1041852416,
        "smoke": 106816,
    }
    assert report["parameters"] == expected[name]


def test_advanced_training_conversion_and_generation_commands(tmp_path, capsys, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    from clearforge.config import ModelConfig
    from clearforge.model import CausalLM
    from clearforge.posttraining import _export
    from clearforge.tokenizer import SPECIAL_TOKENS, ChatTokenizer

    raw = Tokenizer(models.BPE())
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    from tokenizers import decoders

    raw.decoder = decoders.ByteLevel()
    raw.train_from_iterator(
        ["Question helpful answer " * 10],
        trainers.BpeTrainer(
            vocab_size=280,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer = ChatTokenizer(raw)
    torch.manual_seed(7)
    model = CausalLM(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=32,
            max_seq_len=128,
        )
    )
    bundle = tmp_path / "base"
    _export(model, tokenizer, bundle)
    data = tmp_path / "chat.jsonl"
    data.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "helpful answer"},
                ]
            }
        )
        + "\n"
    )
    config = tmp_path / "post.toml"
    config.write_text(
        '[training]\noutput_path="lora"\nsteps=2\ndevice="cpu"\n[adapter]\nrank=2\nalpha=4\n'
    )

    def run(*args):
        assert main(["--threads", "2", *map(str, args)]) == 0, capsys.readouterr().err
        return capsys.readouterr().out

    result = json.loads(
        run(
            "post-train",
            "--model",
            bundle,
            "--data",
            data,
            "--config",
            config,
            "--objective",
            "lora",
        )
    )
    assert result["status"] == "complete" and result["step"] == 2
    assert (tmp_path / "lora/adapter/adapter.json").is_file()
    merged = tmp_path / "merged"
    run(
        "merge-adapter",
        "--model",
        bundle,
        "--adapter",
        tmp_path / "lora/adapter",
        "--output",
        merged,
    )
    first = run(
        "chat",
        "--model",
        merged,
        "--prompt",
        "Question",
        "--device",
        "cpu",
        "--max-new-tokens",
        "4",
    )
    streamed = run(
        "chat",
        "--model",
        merged,
        "--prompt",
        "Question",
        "--device",
        "cpu",
        "--max-new-tokens",
        "4",
        "--stream",
    )
    assert first == streamed
    many = json.loads(
        run(
            "chat",
            "--model",
            merged,
            "--prompt",
            "Question",
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
            "--num-return-sequences",
            "2",
        )
    )
    assert many == [first.rstrip("\n")] * 2
    corpus = tmp_path / "text.jsonl"
    corpus.write_text(json.dumps({"text": "hello café <|user|>"}) + "\n")
    diagnostics = json.loads(run("tokenizer", "inspect", "--tokenizer", bundle, "--input", corpus))
    assert diagnostics["roundtrip_failures"] == 0 and diagnostics["records"] == 1
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {"messages": [{"role": "user", "content": "Question"}], "expected": first.rstrip("\n")}
        )
        + "\n"
    )
    report = json.loads(
        run(
            "evaluate-generation",
            "--model",
            merged,
            "--device",
            "cpu",
            "--cases",
            cases,
            "--output",
            tmp_path / "report.json",
            "--max-new-tokens",
            "4",
        )
    )
    assert report["passed"] == 1
    import types

    from clearforge.rollout import ToolRegistry

    callbacks = types.ModuleType("clearforge_test_callbacks")
    callbacks.registry = ToolRegistry
    callbacks.reward = lambda record, episode: float(episode.completion_tokens)
    monkeypatch.setitem(sys.modules, callbacks.__name__, callbacks)
    agent_data = tmp_path / "prompts.jsonl"
    agent_data.write_text(json.dumps({"prompt": "Question"}) + "\n")
    agent_config = tmp_path / "agent.toml"
    agent_config.write_text(
        '[training]\noutput_path="agent"\nsteps=1\nmax_new_tokens=4\npolicy_epochs=1\ndevice="cpu"\n[training.episode_limits]\nmax_turns=1\n'
    )
    agent_result = json.loads(
        run(
            "train-agent",
            "--model",
            bundle,
            "--data",
            agent_data,
            "--config",
            agent_config,
            "--registry",
            "clearforge_test_callbacks:registry",
            "--registry-id",
            "empty-v1",
            "--reward",
            "clearforge_test_callbacks:reward",
            "--reward-id",
            "length-v1",
        )
    )
    assert (
        agent_result["status"] == "complete"
        and (tmp_path / "agent/model/provenance.json").is_file()
    )
    pytest.importorskip("transformers")
    hf = tmp_path / "hf"
    run("convert-model", "--direction", "to-transformers", "--input", merged, "--output", hf)
    native = tmp_path / "native"
    run("convert-model", "--direction", "from-transformers", "--input", hf, "--output", native)
    assert (
        run(
            "chat",
            "--model",
            native,
            "--prompt",
            "Question",
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
        )
        == first
    )
