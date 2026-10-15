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
        "MIT",
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
        "MIT",
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
