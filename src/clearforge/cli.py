"""Command-line entry points for reproducible training and native inference."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from . import __version__


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clearforge", description=__doc__)
    parser.add_argument("--version", action="version", version=f"clearforge {__version__}")
    parser.add_argument(
        "--threads", type=_positive, default=8, help="Torch CPU threads (default: 8)"
    )
    parser.add_argument("--debug", action="store_true", help="show exception tracebacks")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="inspect actual runtime and CUDA access")
    doctor.add_argument(
        "--config", type=Path, help="also count configured parameters without allocation"
    )
    data = commands.add_parser("data", help="prepare and pack auditable datasets")
    data_commands = data.add_subparsers(dest="operation", required=True)
    prepare = data_commands.add_parser("prepare", help="deduplicate and split JSONL records")
    prepare.add_argument("--source", type=Path, action="append", required=True)
    prepare.add_argument(
        "--license", required=True, help="license declaration applying to all sources"
    )
    prepare.add_argument("--language", default="en")
    prepare.add_argument("--mode", choices=("pretrain", "sft"), required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--validation-fraction", type=float, default=0.05)
    prepare.add_argument("--seed", type=int, default=42)
    pack = data_commands.add_parser("pack", help="write memory-mapped token arrays")
    pack.add_argument("--input", type=Path, required=True)
    pack.add_argument("--tokenizer", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--seq-length", type=_positive, required=True)
    pack.add_argument("--mode", choices=("pretrain", "sft"), required=True)
    tokenizer = commands.add_parser("tokenizer", help="train a byte-level BPE tokenizer")
    tokenizer_commands = tokenizer.add_subparsers(dest="operation", required=True)
    train_tokenizer = tokenizer_commands.add_parser("train")
    train_tokenizer.add_argument("--input", type=Path, required=True)
    train_tokenizer.add_argument("--output", type=Path, required=True)
    train_tokenizer.add_argument("--vocab-size", type=_positive, required=True)
    train_tokenizer.add_argument("--min-frequency", type=_positive, default=2)
    for command in ("train", "evaluate", "export"):
        sub = commands.add_parser(command)
        sub.add_argument("config", type=Path)
        if command == "train":
            sub.add_argument("--resume", type=Path)
            sub.add_argument(
                "--stop-after-steps", type=_positive, help="pause at this absolute update"
            )
        else:
            sub.add_argument("--checkpoint", type=Path, required=True)
        if command == "export":
            sub.add_argument("--output", type=Path, required=True)
    chat = commands.add_parser("chat", help="generate from a verified native export")
    chat.add_argument("--model", type=Path, required=True)
    chat.add_argument("--prompt", help="one user turn; omit for an interactive session")
    chat.add_argument("--system", help="optional initial system message")
    chat.add_argument("--device", default="auto")
    chat.add_argument("--max-new-tokens", type=_positive, default=128)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--top-k", type=_positive)
    chat.add_argument("--seed", type=int, default=42)
    return parser


def _tokenizer_file(path: Path) -> Path:
    return path / "tokenizer.json" if path.is_dir() else path


def _device(value: str) -> Any:
    import torch

    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value not in {"cpu", "cuda"} and not (value.startswith("cuda:") and value[5:].isdigit()):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _doctor(config: Path | None) -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {
        "clearforge": __version__,
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cpu_threads": torch.get_num_threads(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
    }
    for filename in ("cpu.max", "memory.max"):
        path = Path("/sys/fs/cgroup") / filename
        report[f"cgroup_{filename}"] = path.read_text().strip() if path.is_file() else None
    if report["cuda_available"]:
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                # Exercise allocation, a kernel, synchronization, and host readback.
                probe = torch.arange(16, device=f"cuda:{index}", dtype=torch.float32)
                if (probe.square().sum()).item() != 1240:
                    raise RuntimeError(f"CUDA device {index} failed the arithmetic probe")
                del probe
                free, total = torch.cuda.mem_get_info(index)
                report["gpus"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "free_bytes": free,
                        "total_bytes": total,
                        "bf16_supported": torch.cuda.is_bf16_supported(),
                        "arithmetic_probe": "passed",
                    }
                )
    if config is not None:
        from .config import load_config
        from .model import CausalLM

        with torch.device("meta"):
            model = CausalLM(load_config(config).model)
        report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
    return report


def _load_bundle(directory: Path, device: Any) -> tuple[Any, Any]:
    """Verify and load the same open file descriptions; never load training pickle."""
    import torch
    from tokenizers import Tokenizer

    from .config import ModelConfig
    from .model import CausalLM
    from .tokenizer import ChatTokenizer

    provenance = json.loads((directory / "provenance.json").read_text())
    names = {"model.pt", "model.json", "tokenizer.json"}
    if not isinstance(provenance, dict) or provenance.get("format") != "clearforge-native-v1":
        raise ValueError("unsupported native export format")
    hashes = provenance.get("files")
    if not isinstance(hashes, dict) or set(hashes) != names:
        raise ValueError("export must declare exactly the three required bundle files")
    with ExitStack() as stack:
        files = {}
        for name in sorted(names):
            stream = stack.enter_context((directory / name).open("rb"))
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != hashes[name]:
                raise ValueError(f"export integrity check failed: {name}")
            stream.seek(0)
            files[name] = stream
        config = ModelConfig(**json.load(files["model.json"]))
        tokenizer = ChatTokenizer(
            Tokenizer.from_str(files["tokenizer.json"].read().decode("utf-8"))
        )
        if tokenizer.vocab_size != config.vocab_size:
            raise ValueError("export tokenizer vocabulary does not match model")
        weights = torch.load(files["model.pt"], map_location="cpu", weights_only=True)
        with torch.device("meta"):
            model = CausalLM(config)
        model.load_state_dict(weights, strict=True, assign=True)
        if any(not torch.isfinite(p).all().item() for p in model.parameters()):
            raise ValueError("export contains nonfinite model weights")
        if config.tie_embeddings:
            if not torch.equal(model.lm_head.weight, model.token_embedding.weight):
                raise ValueError("export contains inconsistent tied embedding weights")
            model.lm_head.weight = model.token_embedding.weight
        # Detect ordinary in-place edits as well as path replacement during loading.
        for name, stream in files.items():
            stream.seek(0)
            if hashlib.file_digest(stream, "sha256").hexdigest() != hashes[name]:
                raise ValueError(f"export changed while loading: {name}")
        model.to(device).eval()
    return model, tokenizer


def _chat(args: argparse.Namespace) -> None:
    import torch

    device = _device(args.device)
    model, tokenizer = _load_bundle(args.model, device)
    torch.manual_seed(args.seed)
    history = [{"role": "system", "content": args.system}] if args.system else []
    while True:
        if args.prompt is not None:
            prompt = args.prompt
        else:
            try:
                prompt = input("You: ")
            except EOFError:
                return
            if prompt.strip() in {"/quit", "/exit"}:
                return
            if not prompt.strip():
                continue
        messages = history + [{"role": "user", "content": prompt}]
        ids, _ = tokenizer.encode_chat(messages, add_generation_prompt=True)
        output = model.generate(
            torch.tensor([ids], dtype=torch.long, device=device),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            eos_token_id=tokenizer.eos_id,
        )
        answer = tokenizer.decode(output[0, len(ids) :].tolist(), skip_special_tokens=True)
        print(answer if args.prompt is not None else f"Assistant: {answer}", flush=True)
        history = messages + [{"role": "assistant", "content": answer}]
        if args.prompt is not None:
            return


def _run(args: argparse.Namespace) -> Any:
    if args.command == "doctor":
        return _doctor(args.config)
    if args.command == "data":
        from .data import SourceSpec, pack_dataset, prepare_sources

        if args.operation == "prepare":
            return prepare_sources(
                [SourceSpec(path, args.license, args.language) for path in args.source],
                args.output,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                mode=args.mode,
            )
        return pack_dataset(
            args.input,
            _tokenizer_file(args.tokenizer),
            args.output,
            seq_length=args.seq_length,
            mode=args.mode,
        )
    if args.command == "tokenizer":
        from .tokenizer import train_tokenizer

        return {
            "tokenizer": train_tokenizer(
                args.input,
                args.output,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
            )
        }
    if args.command == "chat":
        return _chat(args)
    from .config import load_config
    from .training import evaluate, export, train

    config = load_config(args.config)
    if args.command == "train":
        return dataclasses.asdict(
            train(config, resume=args.resume, stop_after_steps=args.stop_after_steps)
        )
    if args.command == "evaluate":
        return evaluate(config, args.checkpoint)
    return {"export": export(config, args.checkpoint, args.output)}


def main(argv: list[str] | None = None) -> int:
    """Run a command, returning a stable nonzero status for user-facing failures."""
    args = _parser().parse_args(argv)
    try:
        import torch

        torch.set_num_threads(args.threads)
        result = _run(args)
        if result is not None:
            print(json.dumps(result, indent=2, default=str, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("clearforge: interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        if args.debug:
            raise
        print(f"clearforge: {error}", file=sys.stderr)
        return 1
