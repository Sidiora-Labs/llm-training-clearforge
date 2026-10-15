# Getting started

## Installation

Use Linux with Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
clearforge doctor
```

For CUDA, install a PyTorch build appropriate to the GPU and driver using the
[PyTorch installation guide](https://pytorch.org/get-started/locally/).
`doctor` checks CUDA access and reports device information and available memory.

All commands below run from the repository root. Configuration paths resolve
relative to their TOML file. CLI data paths resolve relative to your current
directory. Output directories must be new. The complete sequence is also available
as `bash scripts/run_smoke.sh`.

## 1. Prepare text and train a tokenizer

```bash
clearforge data prepare --source examples/pretrain.jsonl --license Apache-2.0 \
  --language en --mode pretrain --output data/smoke-prepared \
  --validation-fraction 0.2
clearforge tokenizer train --input data/smoke-prepared/train.jsonl \
  --output data/smoke-tokenizer --vocab-size 512
clearforge data pack --input data/smoke-prepared --tokenizer data/smoke-tokenizer \
  --output data/smoke-packed --seq-length 64 --mode pretrain
```

The CLI accepts a tokenizer directory or its `tokenizer.json` for `data pack`.
The TOML `tokenizer_path` must identify the actual `tokenizer.json` file.
The fixture learns the preset's 512-token vocabulary; a different small corpus
may learn fewer tokens. Match `model.vocab_size` to the packed manifest exactly.

## 2. Pretrain and verify continuation

```bash
clearforge train configs/smoke.toml --stop-after-steps 4
clearforge train configs/smoke.toml --resume runs/smoke-pretrain/latest.pt
```

`--stop-after-steps` pauses at an absolute optimizer update without changing the
configured eight-update schedule. Resume uses the same configuration, inputs,
implementation, and execution environment. A completed run remains completed;
increasing `max_steps` changes the schedule and is rejected as an exact resume.

## 3. Fine-tune the base model for chat

```bash
clearforge data prepare --source examples/chat.jsonl --license Apache-2.0 \
  --language en --mode sft --output data/smoke-sft-prepared \
  --validation-fraction 0.2
clearforge data pack --input data/smoke-sft-prepared --tokenizer data/smoke-tokenizer \
  --output data/smoke-sft-packed --seq-length 64 --mode sft
clearforge train configs/smoke-sft.toml
```

Reuse the base tokenizer. The SFT configuration loads the pretraining checkpoint
into a new run with a fresh optimizer and data cursor. Only assistant content and
assistant end-of-turn tokens contribute to the loss.

## 4. Evaluate, export, and generate

```bash
clearforge evaluate configs/smoke-sft.toml --checkpoint runs/smoke-sft/latest.pt
clearforge export configs/smoke-sft.toml --checkpoint runs/smoke-sft/latest.pt \
  --output runs/smoke-export
clearforge chat --model runs/smoke-export --prompt "What is a library?" \
  --max-new-tokens 8 --device cpu
```

Omit `--prompt` for interactive chat; enter `/quit` to leave. The prompt, retained
conversation, and requested response budget must fit the model's context. An
oversized request fails explicitly. The example uses a 64-token context.

The example runs on CPU using the included dataset, with eight updates per stage.
It runs offline after dependency installation.

## Larger experiments

Read [data preparation](data.md), then build a licensed English corpus and an
independent validation partition. Copy a size preset into a new experiment file,
set its data/tokenizer/output paths, and match vocabulary and context dimensions.
Use a short run to measure memory before committing to a larger training budget.

The global `--threads` option defaults to eight CPU threads and comes before the
command, for example `clearforge --threads 4 train configs/experiment.toml`.
Keep it unchanged when resuming. All commands expose `--help`; global `--debug`
shows full exception tracebacks when diagnosing a failure.

See [training and recovery](training.md) for interruption handling, retention,
trust boundaries, and compatibility requirements.
