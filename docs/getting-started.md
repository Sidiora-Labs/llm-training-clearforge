# Getting started

Use Python 3.11 or newer in an isolated environment. Install a PyTorch build that
supports the intended hardware, then install Clearforge and its development tools:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
clearforge doctor
```

For CUDA installations, use the appropriate wheel from the
[PyTorch installation guide](https://pytorch.org/get-started/locally/). A driver
version alone does not establish that the installed PyTorch build supports a GPU.
The doctor command reports what this process can actually access.

## Run the offline example

Run these commands from the repository root. Each output directory must be new;
use a separate workspace or fresh paths when repeating the example.

```bash
clearforge data prepare --source examples/pretrain.jsonl --license MIT \
  --language en --mode pretrain --output data/smoke-prepared \
  --validation-fraction 0.2
clearforge tokenizer train --input data/smoke-prepared/train.jsonl \
  --output data/smoke-tokenizer --vocab-size 512
clearforge data pack --input data/smoke-prepared --tokenizer data/smoke-tokenizer \
  --output data/smoke-packed --seq-length 64 --mode pretrain
clearforge train configs/smoke.toml
clearforge evaluate configs/smoke.toml --checkpoint runs/smoke-pretrain/latest.pt
clearforge export configs/smoke.toml --checkpoint runs/smoke-pretrain/latest.pt \
  --output runs/smoke-export
clearforge chat --model runs/smoke-export --prompt "What is a library?" \
  --max-new-tokens 32
```

This run exercises the pipeline on a tiny model and fixture corpus. Its generated
text is not evidence of useful chat quality. The example intentionally runs on
CPU so that the basic workflow does not require a GPU or network access.

## Continue a run

Use `--resume` with the same configuration and unchanged data:

```bash
clearforge train configs/smoke.toml --resume runs/smoke-pretrain/latest.pt
```

A completed run remains completed; increasing `max_steps` is a different schedule
and is not an exact continuation of the recorded run. To inspect an interruption,
read the run's metrics and checkpoint step before resuming.

## Move to a larger experiment

Read [data preparation](data.md) before importing a real corpus. Train a tokenizer
on its training partition, pack the data with the chosen context length, and copy
one of the nominal 100M, 300M, or 1B configurations to a new experiment file.
Match vocabulary and sequence dimensions to the packed manifest. Check the actual
GPU allocation, then use a short run to measure memory and throughput.

See [training and recovery](training.md) for optimization and checkpoint semantics,
and [architecture](architecture.md) for component boundaries.
