<h1 align="center">Clearforge</h1>

<p align="center">
  <strong>A reproducible, English-first system for training small language models.</strong>
</p>

<p align="center">
  <a href="https://github.com/Sidiora-Labs"><img src="https://img.shields.io/badge/Built%20by-Sidiora%20Labs-0A0A0A?style=flat-square" alt="Built by Sidiora Labs" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-0A0A0A?style=flat-square" alt="Apache License 2.0" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-0A0A0A?style=flat-square" alt="Python 3.11 or newer" /></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/Version-0.1.0%20unreleased-0A0A0A?style=flat-square" alt="Version 0.1.0 — unreleased" /></a>
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

> **Development status:** Version 0.1.0 is implemented and locally qualified, but
> has not been published. See [qualification results](docs/qualification.md) for
> the tested environment, GPU checks, and model-quality limitations.

## What it does

- Train native decoder models from scratch, then fine-tune them on assistant responses.
- Use nominal **100M, 300M, and 1B** presets, or define your own dimensions.
- Validate and deduplicate JSONL records before deterministic document splitting.
- Train byte-level BPE on training data only; preserve literal text and chat role boundaries.
- Pack pretraining documents efficiently and memory-map token arrays during training.
- Train on CPU or one CUDA GPU, with gradient accumulation and fp32, bf16, or fp16 controls.
- Restore optimizer, scheduler, random generators, and data position from atomic checkpoints.
- Evaluate token-weighted loss and export verified native bundles for local generation.

The implementation and documentation are in English. Source language declarations
are recorded but are not automatic language detection.

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

**The example model is a functional test, not a useful chatbot.** Producing a
capable chat model requires a substantially larger, carefully selected corpus,
training budget, and independent evaluation.

Read the [step-by-step guide](docs/getting-started.md) for the individual commands,
CUDA setup, and moving to larger experiments.

## Presets

| Configuration | Parameters | Context | Intended use |
|---|---:|---:|---|
| `configs/smoke.toml` | 106,816 | 64 | Offline pretraining qualification |
| `configs/smoke-sft.toml` | 106,816 | 64 | Offline chat fine-tuning qualification |
| `configs/pretrain-100m.toml` | 100,682,496 | 1,024 | Small pretraining experiment |
| `configs/sft-100m.toml` | 100,682,496 | 1,024 | Chat fine-tuning of the 100M base |
| `configs/pretrain-300m.toml` | 304,137,216 | 1,024 | Larger pretraining experiment |
| `configs/pretrain-1b.toml` | 1,041,852,416 | 1,024 | Billion-parameter experiment |

Counts use the shipped vocabulary and tied embeddings. Larger presets describe
model and optimizer settings; they do not include pretrained weights or production
datasets. Run `clearforge doctor --config configs/pretrain-100m.toml` to inspect
actual CUDA access and count parameters without allocating that model.

## Documentation

- [Getting started](docs/getting-started.md): executable pretraining and chat workflow.
- [Small chat training recipe](docs/small-chat-recipe.md): a complete 100M training path.
- [Data preparation](docs/data.md): formats, provenance, licensing, and split discipline.
- [Data pipeline details](docs/data-pipeline.md): integrity checks and packing boundaries.
- [Training and recovery](docs/training.md): optimization, precision, checkpoints, and limits.
- [Architecture](docs/architecture.md): components and decoder design.
- [Qualification](docs/qualification.md): measured results and tested environment.
- [Contributing](CONTRIBUTING.md): local checks and cg specification/fleet workflow.

## Scope

Version 0.1 targets Linux and single-process CPU or single-GPU training. It uses
full-parameter optimization of the native model. Distributed training, LoRA,
external pretrained-model import, quantized execution, KV-cached generation, and
production serving are outside this version. Training materializes full sequence
logits; measure memory with the intended vocabulary, context, and batch size.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, verification commands,
and the cg specification and fleet workflow. [CHANGELOG.md](CHANGELOG.md) records
project changes; the [source index](docs/development/source-index.md) connects
implementation and verification evidence.

## License

Apache License 2.0. See [LICENSE](LICENSE). Copyright 2026 Sidiora Labs Inc.

The project license covers the code, documentation, and included original
fixtures. External datasets and model artifacts retain their own licensing
obligations.
