# Qualification record

Local qualification performed on **2026-10-15**. These results describe the
implementation and small verification workloads, not the quality of a broadly
trained chat model. The project has not been published, and remote CI has not
been run.

## Environment

| Component | Observed value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu128 |
| NumPy | 2.5.4 |
| Tokenizers | 0.22.2 |
| GPU | NVIDIA B200 |
| CUDA arithmetic probe | Passed real allocation, kernel, and readback |
| GPU memory reported by Torch | 191,495,471,104 total bytes; 40,471,363,584 free bytes at inspection |
| CPU cgroup quota | 32 CPU equivalents |
| RAM cgroup limit | 68,719,476,736 bytes |

GPU free memory is an observation at one point in time, not a dedicated-memory
allocation guarantee. The process's CPU affinity mask exposed more logical CPUs
than its cgroup quota; explicit thread limits were used for small workloads.

## Automated and packaging checks

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python scripts/check_presets.py
cg spec lint
cg spec render --check
```

- Full hardware-enabled suite: **156 passed**, including exact resumed training
  with dropout on CPU, CUDA fp16, and CUDA bf16.
- CPU-only suite: **154 passed, 2 explicitly skipped** CUDA tests.
- Ruff lint and formatting passed; all six shipped presets load successfully.
- Source and wheel distributions built successfully. The wheel was installed
  into an isolated target directory, and `python -m clearforge --version` loaded
  that installed package successfully.

The local host lacks Python's `ensurepip`, so build isolation could not create
its temporary environment. After installing the declared Hatchling build backend
in the development environment, `python -m build --no-isolation` successfully
built both distributions. CI uses standard `python -m build` on a provisioned
Python installation. Local testing covered Python 3.12; the Python 3.11 CI matrix
is configured but has not been remotely executed.

## End-to-end CPU execution

The installed `clearforge` command executed `bash scripts/run_smoke.sh`, including
an actual process restart between update four and update eight, a separate chat
fine-tuning run, evaluation, native export, and generated output.

| Stage | Updates | Final validation loss | Validation supervised tokens |
|---|---:|---:|---:|
| Fixture pretraining | 8 | 5.550493 | 248 |
| Fixture chat fine-tuning | 8 | 5.136901 | 116 |

These measurements use the first two validation batches, not a full independent
benchmark. The eight-update chat fixture produced punctuation rather than useful
answers; that is expected for this functional workload and is not presented as
a trained chat assistant.

## GPU execution

A complete four-update fixture pretraining run used the normal trainer with CUDA
bf16, produced checkpoints, and evaluated successfully. Its validation loss was
5.840458 over 248 supervised tokens. Peak Torch tensor allocation was 72,287,744
bytes for this tiny run.

The **100,682,496-parameter preset** also completed three real AdamW updates on the
B200 at context 1,024, batch size four, and eight accumulation steps, using bf16
operations with float32 weights. Each update ran forward, loss, backward,
gradient clipping, and optimizer mutation. Losses and gradients were finite;
peak Torch tensor allocation was **5,588,494,336 bytes** (approximately 5.2 GiB).

That larger check used seeded synthetic token IDs to isolate the model/optimizer
runtime. It is not a corpus training run, chat-quality result, or production
throughput benchmark. The 300M and 1B presets were validated and counted on the
meta device; their full training memory and throughput have not been measured.

## Quality acceptance for a trained model

The complete 100M pretraining/SFT recipe is documented in
[small chat training](small-chat-recipe.md). Before calling its resulting model
usable, run it on a substantial licensed English corpus and evaluate held-out
conversation tasks. Record dataset revisions, tokenizer identity, training
budget, loss curves, representative generations, and failure cases. Passing
software tests and observing finite GPU updates do not establish answer quality.

## Presentation and distribution metadata

The subsequent convention alignment applies Apache-2.0 with Sidiora Labs Inc.
attribution consistently to the project and original fixtures. The CLI fixture
workflow was rerun with those declarations. Built wheel metadata and the bundled
license were checked against `pyproject.toml` and the root `LICENSE`; these
packaging checks do not alter the earlier numerical training results.
