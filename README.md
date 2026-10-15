<h1 align="center">Clearforge</h1>

<p align="center">
  <strong>A reproducible, English-first system for training small language models.</strong>
</p>

<p align="center">
  <a href="https://github.com/Sidiora-Labs"><img src="https://img.shields.io/badge/Built%20by-Sidiora%20Labs-0A0A0A?style=flat-square" alt="Built by Sidiora Labs" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-0A0A0A?style=flat-square" alt="Apache License 2.0" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-0A0A0A?style=flat-square" alt="Python 3.11 or newer" /></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/Version-0.1.0-0A0A0A?style=flat-square" alt="Version 0.1.0" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/small-chat-recipe.md">Train a chat model</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="LICENSE">License</a>
</p>

---

## Overview

Clearforge takes local text and conversation datasets through validation,
tokenization, pretraining, supervised chat fine-tuning, evaluation, and native
model export. It includes a compact PyTorch decoder, strict TOML configuration,
and a command-line interface with resumable runs and auditable data provenance.

## What it does

- Train native decoder models from scratch, then fine-tune them on assistant responses.
- Use nominal **100M, 300M, and 1B** presets, or define your own dimensions.
- Validate and deduplicate JSONL records before deterministic document splitting.
- Train byte-level BPE on training data only; preserve literal text and chat role boundaries.
- Pack pretraining documents efficiently and memory-map token arrays during training.
- Train on CPU, CUDA, or distributed workers with gradient accumulation, mixed precision, compilation, and activation checkpointing.
- Restore optimizer, scheduler, random generators, and data position from atomic checkpoints.
- Fine-tune with LoRA, DPO, distillation, PPO, GRPO, and CISPO.
- Use cached, streamed, or batched generation with nucleus sampling and repetition controls.
- Train dense or sparse expert models, with optional YaRN rotary scaling.
- Encode tool conversations and reasoning fields with validated call and result sequences.
- Evaluate loss and generation suites; import and export Transformers-compatible models.
- Serve a chat API, streaming completions, and a browser interface.

## Quick start

From this repository, install into an isolated Python environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
clearforge doctor
bash scripts/run_smoke.sh
```

Python 3.11 or newer is required. The smoke workflow runs offline after dependency
installation and uses the included original English fixtures. It exercises
pretraining, pause/resume, chat fine-tuning, evaluation, export, and generation.
Use fresh output directories when repeating it.

Read the [step-by-step guide](docs/getting-started.md) for the individual commands,
CUDA setup, and moving to larger experiments.

## Chat and serving

```bash
clearforge chat --model runs/chat-export --prompt "Hello!" --stream
python -m pip install -e '.[serving]'
clearforge serve --model clearforge=runs/chat-export
```

Open `http://127.0.0.1:8000` for the browser interface. The same server exposes
`/v1/models` and `/v1/chat/completions`. See [serving](docs/serving.md) for
multiple models, authentication, request limits, and streaming clients.

## Post-training and interoperability

[Post-training](docs/posttraining.md) covers adapter training, preference
optimization, teacher distillation, and reward-based policy updates. Use
[tool episodes](docs/rollout.md) and [agent training](docs/agent-training.md)
for multi-turn policies with registered tools.

```bash
python -m pip install -e '.[huggingface]'
clearforge convert-model --direction to-transformers \
  --input runs/chat-export --output runs/chat-transformers
clearforge evaluate-generation --model runs/chat-export \
  --cases examples/evaluation.jsonl --output runs/evaluation.json
```

## Presets

| Configuration | Parameters | Context | Intended use |
|---|---:|---:|---|
| `configs/smoke.toml` | 106,816 | 64 | CPU pretraining example |
| `configs/smoke-sft.toml` | 106,816 | 64 | CPU chat fine-tuning example |
| `configs/pretrain-100m.toml` | 100,682,496 | 1,024 | Small pretraining experiment |
| `configs/sft-100m.toml` | 100,682,496 | 1,024 | Chat fine-tuning of the 100M base |
| `configs/pretrain-300m.toml` | 304,137,216 | 1,024 | Larger pretraining experiment |
| `configs/pretrain-1b.toml` | 1,041,852,416 | 1,024 | Billion-parameter experiment |

Counts use the shipped vocabulary and tied embeddings. Run
`clearforge doctor --config configs/pretrain-100m.toml` to inspect CUDA access
and count parameters without allocating the model.

## Documentation

- [Getting started](docs/getting-started.md): executable pretraining and chat workflow.
- [Small chat training recipe](docs/small-chat-recipe.md): a complete 100M training path.
- [Data preparation](docs/data.md): formats, provenance, licensing, and split discipline.
- [Data pipeline details](docs/data-pipeline.md): integrity checks and packing boundaries.
- [Training and recovery](docs/training.md): optimization, precision, checkpoints, and limits.
- [Architecture](docs/architecture.md): components and decoder design.
- [Model options](docs/model.md): cached decoding, sparse experts, and rotary scaling.
- [Post-training](docs/posttraining.md): adapters, preferences, distillation, and policy objectives.
- [Interchange](docs/interchange.md): native and Transformers model conversion.
- [Serving](docs/serving.md): chat API and browser interface.
- [Rollouts](docs/rollout.md): native and HTTP generation with tool episodes.
- [Agent training](docs/agent-training.md): multi-turn policy optimization.
- [Evaluation](docs/evaluation.md): generation suites and tokenizer diagnostics.
- [Contributing](CONTRIBUTING.md): local checks and cg specification/fleet workflow.

## Requirements

Linux, Python 3.11 or newer, and a CPU or CUDA GPU. Distributed training uses
`torchrun` with Gloo on CPU or NCCL on CUDA. Optional `serving` and `huggingface`
extras install the API and Transformers integration dependencies.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, verification commands,
and the cg specification and fleet workflow. [CHANGELOG.md](CHANGELOG.md) records
project changes.

## License

Apache License 2.0. See [LICENSE](LICENSE). Copyright 2026 Sidiora Labs Inc.

The project license covers the code, documentation, and included original
fixtures. External datasets and model artifacts retain their own licensing
obligations.
