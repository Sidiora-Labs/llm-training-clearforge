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
    chat.add_argument("--top-p", type=float, default=1.0)
    chat.add_argument("--repetition-penalty", type=float, default=1.0)
    chat.add_argument("--stream", action="store_true")
    chat.add_argument("--num-return-sequences", type=_positive, default=1)
    diagnostics = tokenizer_commands.add_parser(
        "inspect", help="measure a tokenizer on a JSONL text corpus"
    )
    diagnostics.add_argument("--tokenizer", type=Path, required=True)
    diagnostics.add_argument("--input", type=Path, required=True)
    convert = data_commands.add_parser("convert", help="adapt chat, preference or rollout JSONL")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--license", required=True)
    convert.add_argument("--language", default="en")
    convert.add_argument(
        "--adapter", choices=("conversations", "preference", "rollout"), required=True
    )
    convert.add_argument("--output", type=Path, required=True)
    server = commands.add_parser(
        "serve", help="serve native exports through chat API and browser UI"
    )
    server.add_argument(
        "--model",
        action="append",
        required=True,
        help="NAME=EXPORT_DIRECTORY; repeat for multiple models",
    )
    server.add_argument("--device", default="auto")
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=_positive, default=8000)
    server.add_argument("--api-key-env", default="CLEARFORGE_API_KEY")
    server.add_argument("--max-tokens", type=_positive, default=4096)
    server.add_argument("--generation-timeout", type=float, default=120.0)
    interchange = commands.add_parser(
        "convert-model", help="convert native and Transformers model directories"
    )
    interchange.add_argument(
        "--direction", choices=("to-transformers", "from-transformers"), required=True
    )
    interchange.add_argument("--input", type=Path, required=True)
    interchange.add_argument("--output", type=Path, required=True)
    post = commands.add_parser(
        "post-train", help="train adapters, preferences, distillation or online policies"
    )
    post.add_argument("--model", type=Path, required=True)
    post.add_argument("--data", type=Path, required=True)
    post.add_argument(
        "--config", type=Path, required=True, help="TOML [training] and optional [adapter] settings"
    )
    post.add_argument(
        "--objective",
        choices=("sft", "lora", "dpo", "distillation", "ppo", "grpo", "cispo"),
        required=True,
    )
    post.add_argument("--reference", type=Path)
    post.add_argument("--teacher", type=Path)
    post.add_argument(
        "--reward", help="Python module:callable receiving prompt messages and response text"
    )
    post.add_argument("--reward-id", help="versioned identity of reward code and data")
    post.add_argument("--resume", type=Path)
    post.add_argument("--stop-after-steps", type=_positive)
    agent = commands.add_parser("train-agent", help="optimize native multi-turn tool policies")
    agent.add_argument("--model", type=Path, required=True)
    agent.add_argument("--data", type=Path, required=True)
    agent.add_argument("--config", type=Path, required=True)
    agent.add_argument("--objective", choices=("grpo", "cispo"), default="grpo")
    agent.add_argument(
        "--registry", required=True, help="Python module:factory returning a ToolRegistry"
    )
    agent.add_argument(
        "--reward", required=True, help="Python module:callable receiving record and EpisodeResult"
    )
    agent.add_argument("--reward-id", required=True)
    agent.add_argument(
        "--registry-id", required=True, help="versioned identity of tool handlers and external data"
    )
    agent.add_argument("--reference", type=Path)
    agent.add_argument("--resume", type=Path)
    agent.add_argument("--stop-after-steps", type=_positive)
    adapter = commands.add_parser(
        "merge-adapter", help="merge a saved LoRA adapter into a native export"
    )
    adapter.add_argument("--model", type=Path, required=True)
    adapter.add_argument("--adapter", type=Path, required=True)
    adapter.add_argument("--output", type=Path, required=True)
    suite = commands.add_parser("evaluate-generation", help="run scored generation cases")
    suite.add_argument("--model", type=Path, required=True)
    suite.add_argument("--cases", type=Path, required=True)
    suite.add_argument("--output", type=Path, required=True)
    suite.add_argument("--device", default="auto")
    suite.add_argument("--seed", type=int, default=42)
    suite.add_argument("--max-new-tokens", type=_positive, default=128)
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
        if args.num_return_sequences != 1 and (args.prompt is None or args.stream):
            raise ValueError("multiple completions require --prompt and non-streaming output")
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        settings = dict(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_id,
        )
        if args.stream:
            # Byte-level BPE pieces can split Unicode characters. Buffer only an
            # incomplete trailing character and emit complete decoded prefixes.
            generated, emitted = [], ""
            if args.prompt is None:
                print("Assistant: ", end="", flush=True)
            for token in model.stream_generate(inputs, **settings):
                generated.append(int(token[0, 0]))
                text = tokenizer.decode(generated, skip_special_tokens=True)
                stable = text.rstrip("\ufffd")
                if stable.startswith(emitted):
                    print(stable[len(emitted) :], end="", flush=True)
                    emitted = stable
            answer = tokenizer.decode(generated, skip_special_tokens=True)
            print(answer[len(emitted) :], flush=True)
        else:
            output = model.generate(
                inputs, num_return_sequences=args.num_return_sequences, **settings
            )
            answers = [
                tokenizer.decode(row[len(ids) :].tolist(), skip_special_tokens=True)
                for row in output
            ]
            answer = answers[0]
            if len(answers) > 1:
                print(json.dumps(answers, ensure_ascii=False), flush=True)
            else:
                print(answer if args.prompt is not None else f"Assistant: {answer}", flush=True)
        history = messages + [{"role": "assistant", "content": answer}]
        if args.prompt is not None:
            return


def _run(args: argparse.Namespace) -> Any:
    if args.command == "doctor":
        return _doctor(args.config)
    if args.command == "data":
        from .data import SourceSpec, pack_dataset, prepare_sources

        if args.operation == "convert":
            from .data import convert_dataset

            return convert_dataset(
                SourceSpec(args.source, args.license, args.language),
                args.output,
                adapter=args.adapter,
            )
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
        from .tokenizer import ChatTokenizer, train_tokenizer

        if args.operation == "inspect":
            from .evaluation import tokenizer_diagnostics

            return tokenizer_diagnostics(
                ChatTokenizer.load(_tokenizer_file(args.tokenizer)), args.input
            )
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
    if args.command == "serve":
        from .serving import serve

        models = {}
        for value in args.model:
            name, separator, path = value.partition("=")
            if not separator or not name or not path or name in models:
                raise ValueError("--model requires a unique NAME=EXPORT_DIRECTORY")
            models[name] = Path(path)
        serve(
            models,
            host=args.host,
            port=args.port,
            device=args.device,
            api_key=os.environ.get(args.api_key_env),
            max_tokens=args.max_tokens,
            generation_timeout=args.generation_timeout,
        )
        return None
    if args.command == "convert-model":
        from .interchange import export_bundle_transformers, import_transformers

        convert = (
            export_bundle_transformers
            if args.direction == "to-transformers"
            else import_transformers
        )
        return {"export": convert(args.input, args.output)}
    if args.command in {"post-train", "train-agent"}:
        return _post_train(args)
    if args.command == "merge-adapter":
        from .adapters import load_adapters
        from .posttraining import _export

        model, tokenizer = _load_bundle(args.model, _device("cpu"))
        load_adapters(model, args.adapter)
        _export(model, tokenizer, args.output)
        return {"export": args.output}
    if args.command == "evaluate-generation":
        from .evaluation import evaluate_generations
        from .rollout import GenerationOptions, NativeBackend

        model, tokenizer = _load_bundle(args.model, _device(args.device))
        return evaluate_generations(
            NativeBackend(model, tokenizer),
            args.cases,
            generation=GenerationOptions(max_new_tokens=args.max_new_tokens, temperature=0),
            seed=args.seed,
            output=args.output,
        )
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


def _post_train(args: argparse.Namespace) -> dict[str, Any]:
    import importlib
    import tomllib

    from .adapters import AdapterConfig, save_adapters
    from .posttraining import PostTrainConfig, train_posttraining

    with args.config.open("rb") as stream:
        document = tomllib.load(stream)
    if set(document) - {"training", "adapter"} or "training" not in document:
        raise ValueError("post-training TOML requires [training] and optional [adapter]")
    settings = dict(document["training"])
    if "output_path" in settings:
        settings["output_path"] = args.config.resolve().parent / settings["output_path"]
    settings["device"] = str(_device(settings.get("device", "auto")))
    if args.command == "train-agent":
        from .agent_training import AgentTrainConfig
        from .rollout import EpisodeLimits

        if "episode_limits" in settings:
            settings["episode_limits"] = EpisodeLimits(**settings["episode_limits"])
        config = AgentTrainConfig(**settings)
    else:
        config = PostTrainConfig(**settings)
    model, tokenizer = _load_bundle(args.model, _device("cpu"))
    reference = _load_bundle(args.reference, _device("cpu")) if args.reference else None
    if reference and reference[1]._tokenizer.to_str() != tokenizer._tokenizer.to_str():
        raise ValueError("reference tokenizer must match policy")
    teacher = _load_bundle(args.teacher, _device("cpu")) if getattr(args, "teacher", None) else None
    reward = None
    if args.reward:
        module, separator, name = args.reward.partition(":")
        if not separator or not module or not name:
            raise ValueError("--reward must be module:callable")
        reward = getattr(importlib.import_module(module), name)
        if not callable(reward) or not args.reward_id:
            raise ValueError("--reward requires a callable and --reward-id")
    adapter = None
    if "adapter" in document:
        values = dict(document["adapter"])
        if "targets" in values:
            values["targets"] = tuple(values["targets"])
        adapter = AdapterConfig(**values)
    if args.command == "train-agent":
        from .agent_training import train_agent_policy

        module, separator, name = args.registry.partition(":")
        if not separator or not module or not name:
            raise ValueError("--registry must be module:factory")
        if adapter is not None:
            raise ValueError("train-agent configuration does not accept [adapter]")
        registry = getattr(importlib.import_module(module), name)()
        result = train_agent_policy(
            model,
            tokenizer,
            args.data,
            config,
            registry=registry,
            registry_id=args.registry_id,
            reward_fn=reward,
            reward_id=args.reward_id,
            objective=args.objective,
            reference=reference[0] if reference else None,
            resume=args.resume,
            stop_after_steps=args.stop_after_steps,
        )
        return {
            "status": result.status,
            "step": result.step,
            "checkpoint": result.checkpoint,
            "export": config.output_path / "model" if result.status == "complete" else None,
        }
    result = train_posttraining(
        model,
        tokenizer,
        args.data,
        config,
        objective=args.objective,
        reference=reference[0] if reference else None,
        teacher=teacher[0] if teacher else None,
        teacher_tokenizer=teacher[1] if teacher else None,
        reward_fn=reward,
        reward_id=args.reward_id,
        adapter_config=adapter,
        resume=args.resume,
        stop_after_steps=args.stop_after_steps,
    )
    if result.status == "complete" and hasattr(result.model, "_clearforge_adapter_config"):
        adapter_path = config.output_path / "adapter"
        if not adapter_path.exists():
            save_adapters(result.model, adapter_path)
    return {
        "status": result.status,
        "step": result.step,
        "checkpoint": result.checkpoint,
        "export": config.output_path / "model" if result.status == "complete" else None,
    }


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
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        ImportError,
        AttributeError,
    ) as error:
        if args.debug:
            raise
        print(f"clearforge: {error}", file=sys.stderr)
        return 1
