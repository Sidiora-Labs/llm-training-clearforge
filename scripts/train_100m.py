#!/usr/bin/env python3
"""Pinned English corpus acquisition and a resumable native 100M chat training chain.

Launch with --launch. The detached supervisor freezes code and configuration,
serializes lifecycle changes with flock, and records stage failures truthfully.
Restart the same command to resume an interrupted pipeline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

PINS = {
    "pretrain": (
        "HuggingFaceFW/fineweb-edu",
        "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "ODC-BY-1.0",
    ),
    "sft": (
        "HuggingFaceTB/smol-smoltalk",
        "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
        "Apache-2.0",
    ),
}


def write_json(path, value):
    from clearforge.checkpoint import atomic_write_bytes

    atomic_write_bytes((json.dumps(value, indent=2) + "\n").encode(), path)


def status(root, stage, **extra):
    value = dict(stage=stage, updated_at=time.time(), pid=os.getpid(), **extra)
    write_json(root / "status.json", value)
    print(json.dumps(value), flush=True)


def download(root, kind):
    from huggingface_hub import HfApi, hf_hub_download

    from clearforge.checkpoint import file_sha256

    repo, revision, license_name = PINS[kind]
    info = HfApi(token=False).repo_info(
        repo, repo_type="dataset", revision=revision, files_metadata=True
    )
    files = sorted(
        (
            x
            for x in info.siblings
            if x.rfilename.endswith(".parquet")
            and (kind == "sft" or x.rfilename.startswith("sample/10BT/"))
        ),
        key=lambda x: x.rfilename,
    )
    if kind == "pretrain":
        files = files[:3]
    destination = root / "downloads" / kind
    destination.mkdir(parents=True, exist_ok=True)

    def one(item):
        path = Path(
            hf_hub_download(
                repo,
                item.rfilename,
                repo_type="dataset",
                revision=revision,
                local_dir=destination,
                token=False,
            )
        )
        digest = file_sha256(path)
        if path.stat().st_size != item.size or digest != item.lfs.sha256:
            raise ValueError(f"download digest mismatch: {path}")
        print(f"verified {path} ({item.size} bytes)", flush=True)
        return {"path": str(path), "filename": item.rfilename, "bytes": item.size, "sha256": digest}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        entries = list(pool.map(one, files))
    manifest = dict(repo=repo, revision=revision, license=license_name, files=entries)
    write_json(root / f"{kind}-download.json", manifest)
    return manifest


def convert(manifest, path, *, split="train"):
    import pyarrow.parquet as pq

    if path.exists():
        return
    temporary = path.with_suffix(".partial")
    count = 0
    with temporary.open("w") as out:
        for entry in manifest["files"]:
            if "smol-smoltalk" in manifest["repo"] and f"/{split}-" not in entry["filename"]:
                continue
            column = "text" if "fineweb" in manifest["repo"] else "messages"
            for batch in pq.ParquetFile(entry["path"]).iter_batches(
                batch_size=512, columns=[column]
            ):
                for value in batch.column(0).to_pylist():
                    out.write(json.dumps({column: value}, ensure_ascii=False) + "\n")
                    count += 1
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)
    print(f"converted {count} documents to {path}", flush=True)


def prepare(root, kind, manifest):
    from clearforge.data import SourceSpec, prepare_sources

    raw = root / f"{kind}.jsonl"
    convert(manifest, raw)
    prepared = root / f"{kind}-prepared"
    if not prepared.exists():
        prepare_sources(
            [SourceSpec(raw, manifest["license"], "en")],
            prepared,
            validation_fraction=0.001 if kind == "pretrain" else 0.02,
            seed=42,
            mode=kind,
        )
    return prepared


def train_worker(config_path):
    import torch

    from clearforge.config import load_config
    from clearforge.training import train

    torch.set_num_threads(8)
    config = load_config(config_path)
    latest = config.output_dir / "latest.pt"
    result = train(config, resume=latest if latest.exists() else None)
    print(
        json.dumps(
            {"result": result.status, "step": result.step, "checkpoint": str(result.checkpoint)}
        ),
        flush=True,
    )
    if result.status != "complete":
        raise RuntimeError(f"training stopped at {result.step}: {result.status}")


def run(root):
    from clearforge.data import SourceSpec, pack_dataset, prepare_sources
    from clearforge.tokenizer import train_tokenizer

    control = root / "control"
    status(root, "downloading_pretrain")
    manifest = download(root, "pretrain")
    status(root, "preparing_pretrain")
    prepared = prepare(root, "pretrain", manifest)
    tokenizer = root / "tokenizer" / "tokenizer.json"
    if not tokenizer.exists():
        subset = root / "tokenizer-subset.jsonl"
        with (prepared / "train.jsonl").open() as source, subset.open("w") as target:
            for _, line in zip(range(30000), source):
                target.write(line)
        subset_prepared = root / "tokenizer-prepared"
        if not subset_prepared.exists():
            prepare_sources(
                [SourceSpec(subset, PINS["pretrain"][2], "en")],
                subset_prepared,
                validation_fraction=0,
                seed=42,
                mode="pretrain",
            )
        status(root, "training_tokenizer", training_documents=30000)
        train_tokenizer(subset_prepared / "train.jsonl", tokenizer.parent, vocab_size=32768)
    packed = root / "pretrain-packed"
    if not packed.exists():
        status(root, "packing_pretrain")
        pack_dataset(prepared, tokenizer, packed, seq_length=1024, mode="pretrain")
    config = control / "english-100m-pretrain.toml"
    train_log = (root / "pretrain.log").open("a")
    process = subprocess.Popen(
        [sys.executable, str(control / "train_100m.py"), "--worker", str(config)],
        stdout=train_log,
        stderr=subprocess.STDOUT,
    )
    status(root, "pretraining_and_preparing_sft", training_pid=process.pid)
    try:
        sft_manifest = download(root, "sft")
        sft_prepared = prepare(root, "sft", sft_manifest)
        sft_packed = root / "sft-packed"
        if not sft_packed.exists():
            pack_dataset(sft_prepared, tokenizer, sft_packed, seq_length=1024, mode="sft")
        convert(sft_manifest, root / "heldout-chat.jsonl", split="test")
        rows = json.loads((sft_packed / "manifest.json").read_text())["row_counts"]["train"]
        # Two complete epochs: cursor drops no partial final update.
        steps = 2 * math.ceil(rows / 32)
        sft_config = control / "english-100m-sft.toml"
        original = (control / "english-100m-sft.template.toml").read_text()
        sft_config.write_text(original.replace("max_steps = 1", f"max_steps = {steps}"))
        write_json(root / "sft-schedule.json", dict(packed_rows=rows, epochs=2, max_steps=steps))
    except BaseException:
        # Preserve a healthy expensive pretraining process; supervisor restart must
        # not launch another writer while that child remains alive.
        status(
            root,
            "sft_preparation_failed_pretraining_continues",
            training_pid=process.pid,
            error=traceback.format_exc(),
        )
        process.wait()
        raise
    status(root, "pretraining", training_pid=process.pid, sft_ready=True)
    code = process.wait()
    train_log.close()
    if code:
        raise RuntimeError(f"pretraining worker exited {code}; see pretrain.log")
    status(root, "sft")
    with (root / "sft.log").open("a") as log:
        subprocess.run(
            [sys.executable, str(control / "train_100m.py"), "--worker", str(sft_config)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    status(root, "evaluating_and_exporting")
    finish(root, sft_config)
    status(root, "complete", export=str(root / "chat-model"))


def finish(root, config_path):
    import torch

    from clearforge.checkpoint import load_checkpoint
    from clearforge.config import load_config
    from clearforge.model import CausalLM
    from clearforge.tokenizer import ChatTokenizer
    from clearforge.training import evaluate, export

    torch.set_num_threads(8)
    cfg = load_config(config_path)
    checkpoint = cfg.output_dir / "latest.pt"
    write_json(root / "evaluation.json", evaluate(cfg, checkpoint))
    if not (root / "chat-model").exists():
        export(cfg, checkpoint, root / "chat-model")
    tokenizer = ChatTokenizer.load(cfg.tokenizer_path)
    model = CausalLM(cfg.model).to("cuda").eval()
    model.load_state_dict(load_checkpoint(checkpoint)["model"])
    examples = []
    with (root / "heldout-chat.jsonl").open() as source:
        for line in source:
            messages = json.loads(line)["messages"]
            first_user = next((i for i, m in enumerate(messages) if m["role"] == "user"), None)
            if first_user is None:
                continue
            prompt = messages[: first_user + 1]
            ids, _ = tokenizer.encode_chat(prompt, add_generation_prompt=True)
            if len(ids) > 768:
                continue
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                generated = model.generate(
                    torch.tensor([ids], device="cuda"),
                    max_new_tokens=128,
                    temperature=0,
                    eos_token_id=tokenizer.eos_id,
                )
            examples.append(
                dict(
                    messages=prompt,
                    response=tokenizer.decode(
                        generated[0, len(ids) :].tolist(), skip_special_tokens=True
                    ),
                )
            )
            if len(examples) == 20:
                break
    write_json(root / "heldout-generations.json", examples)


def verify_launch(root):
    """Prove a real optimizer update and durable checkpoint, not just a live PID."""
    import hashlib

    from clearforge.checkpoint import file_sha256, load_checkpoint

    state = json.loads((root / "status.json").read_text())
    if state["stage"] not in {
        "pretraining_and_preparing_sft",
        "pretraining",
        "sft",
        "evaluating_and_exporting",
        "complete",
    }:
        raise RuntimeError(f"training has not launched: {state['stage']}")
    checkpoint = root / "pretrain" / "latest.pt"
    payload = load_checkpoint(checkpoint)
    if payload["step"] < 1:
        raise RuntimeError("no completed optimizer update has been checkpointed yet")
    if payload["config"]["training"]["max_steps"] != 65536:
        raise RuntimeError("checkpoint does not belong to the full pretraining schedule")
    frozen = root / "control"
    manifest = json.loads((frozen / "runtime-manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_sha256(frozen / name) != digest:
            raise RuntimeError(f"frozen runtime changed: {name}")
    if (
        file_sha256(root / "pretrain-packed" / "manifest.json")
        != payload["identities"]["dataset_manifest_sha256"]
    ):
        raise RuntimeError("packed data manifest differs from checkpoint")
    if (
        file_sha256(root / "tokenizer" / "tokenizer.json")
        != payload["identities"]["tokenizer_sha256"]
    ):
        raise RuntimeError("tokenizer differs from checkpoint")
    with (root / "pretrain" / "metrics.jsonl").open("rb") as journal:
        prefix = journal.read(payload["metrics"]["bytes"])
    if hashlib.sha256(prefix).hexdigest() != payload["metrics"]["sha256"]:
        raise RuntimeError("checkpoint journal hash mismatch")
    records = [json.loads(line) for line in prefix.splitlines()]
    if not records or records[-1]["step"] != payload["step"]:
        raise RuntimeError("checkpoint has no matching optimizer metrics")
    if state["stage"] in {"pretraining_and_preparing_sft", "pretraining"}:
        pid = state.get("training_pid")
        command = Path(f"/proc/{pid}/cmdline")
        if not command.exists() or b"--worker" not in command.read_bytes():
            raise RuntimeError("pretraining worker is no longer running")
    evidence = {
        "verified_at": time.time(),
        "checkpoint_step": payload["step"],
        "checkpoint": str(checkpoint),
        "pipeline_stage": state["stage"],
        "last_metric": records[-1],
        "frozen_runtime_verified": True,
    }
    write_json(root / "launch-verification.json", evidence)
    print(json.dumps(evidence), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--verify-launch", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path("/home/ubuntu/clearforge/runs/english-100m")
    )
    args = parser.parse_args()
    if args.worker:
        train_worker(args.worker)
        return
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.verify_launch:
        verify_launch(root)
        return
    if args.launch:
        # Refuse duplicate supervisors before printing a misleading launch PID.
        with (root / "pipeline.lock").open("a") as probe:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = root / "status.json"
        if state_path.exists():
            training_pid = json.loads(state_path.read_text()).get("training_pid")
            if training_pid:
                command = Path(f"/proc/{training_pid}/cmdline")
                if command.exists() and b"--worker" in command.read_bytes():
                    raise RuntimeError("an existing training worker still owns this pipeline")
        repo = Path(__file__).resolve().parents[1]
        control = root / "control"
        control.mkdir(exist_ok=True)
        if not (control / "runtime").exists():
            shutil.copytree(repo / "src", control / "runtime")
            shutil.copy2(__file__, control / "train_100m.py")
            for kind in ("pretrain", "sft"):
                text = (
                    (repo / "configs" / f"english-100m-{kind}.toml")
                    .read_text()
                    .replace("@RUN_ROOT@", str(root))
                )
                name = f"english-100m-{kind}" + (".template" if kind == "sft" else "") + ".toml"
                (control / name).write_text(text)
        if not (control / "runtime-manifest.json").exists():
            import hashlib

            snapshot = {
                "created_at": time.time(),
                "files": {
                    str(path.relative_to(control)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(control.rglob("*"))
                    if path.is_file() and "__pycache__" not in path.parts
                },
                "python": sys.version,
                "git_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
            }
            write_json(control / "runtime-manifest.json", snapshot)
        env = {
            **os.environ,
            "PYTHONPATH": str(control / "runtime"),
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        }
        with (root / "pipeline.log").open("a") as log:
            proc = subprocess.Popen(
                [sys.executable, str(control / "train_100m.py"), "--root", str(root)],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"Pipeline supervisor PID {proc.pid}; status {root / 'status.json'}")
        return
    with (root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(root)
        except BaseException:
            status(root, "failed", error=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
