# Training and recovery

A run is defined by its resolved TOML configuration, prepared data, tokenizer,
software environment, and checkpoint history. Paths in a configuration resolve
relative to that configuration file, not the shell's current directory.

## Presets

| Preset | Width | Layers | Query / KV heads | Feed-forward width |
|---|---:|---:|---:|---:|
| `pretrain-100m.toml` | 768 | 12 | 12 / 4 | 2048 |
| `pretrain-300m.toml` | 1024 | 24 | 16 / 4 | 2816 |
| `pretrain-1b.toml` | 2048 | 22 | 16 / 4 | 5504 |
| `smoke.toml` | 64 | 2 | 4 / 2 | 128 |

Parameter counts depend on vocabulary size and embedding tying. Training memory
includes weights, gradients, optimizer state, activations, and attention buffers.
Measure memory with the intended context length and batch size.

`max_steps` counts optimizer updates. `batch_size` counts sequences in a
microbatch; `accumulation_steps` controls the number of microbatches accumulated
for an update. The effective token count also depends on padding and supervised
positions. Metrics should be interpreted in tokens, not only examples.

## Precision and determinism

Use float32 on CPU. CUDA bfloat16 and float16 require compatible hardware and
runtime support. Weights and optimizer state remain float32. Keep hardware,
library versions, and thread settings consistent when resuming deterministic runs.

## Pretraining and chat fine-tuning

Pretraining teaches next-token prediction over text. Chat fine-tuning uses the
same vocabulary and model dimensions with assistant-only supervision. Initialize
chat fine-tuning from a base checkpoint as a new run with a fresh optimizer and
data cursor. Resume is reserved for continuing the same run and data identity.

Use a new output directory for a new experiment. Never overwrite a previous run
merely to avoid a compatibility error. If a checkpoint rejects changed data or
configuration, determine which input changed and either restore it or start a
new run intentionally.

## Evaluation

Validation reports token-weighted negative log-likelihood and perplexity.
Perplexity is comparable only when tokenization and evaluation data are held
fixed. Evaluate conversation quality with held-out prompts covering relevance,
instruction following, repetition, and multi-turn consistency.

## Checkpoint trust

Training checkpoints may contain Python serialization needed to restore optimizer
and random state. Load only checkpoints produced by trusted runs. Do not use an
untrusted checkpoint as a transport for model weights.

## Checkpoint recovery and retention

Each run owns its output directory. A nonblocking run lock prevents concurrent
trainers from writing to it. Checkpoints are written through a temporary file,
flushed, and atomically published at optimizer boundaries. `latest.pt` is an
atomic hard link to a retained `checkpoints/step-########.pt` file.

`keep_checkpoints` defaults to **3** and must be positive. After a new checkpoint
and latest pointer are durable, older numbered checkpoints are pruned. Unrelated
filenames are left alone. Allow disk space for the retained checkpoints, a new
checkpoint under construction, and exported weights. Optimizer checkpoints are
substantially larger than the model weights alone.

SIGINT and SIGTERM request a stop at the next optimizer boundary, followed by a
checkpoint. A hard kill or machine failure cannot save unfinished work. Resume
from the most recent durable checkpoint; the metrics journal is verified against
that checkpoint and any newer uncommitted suffix is removed atomically.

Resume requires the owned numbered checkpoint to remain in the run's checkpoint
directory, even when selecting it through `latest.pt`. To move a resumable run,
retain its complete directory and original resolved paths; path changes are
configuration changes and are rejected by exact resume. Use a native export for
portable inference instead.

Compatibility checks cover the full resolved configuration, tokenizer and packed
data hashes, implementation source hashes, library/runtime identity, thread
settings, and recorded CUDA hardware. Resume rejects incompatible changes.

## Metrics and validation coverage

`metrics.jsonl` contains one record per optimizer update: step, epoch/cursor,
supervised token count, token-weighted training loss, learning rate, gradient
norm, and periodic validation results. Accumulation stops at an epoch boundary;
a final partial group uses its actual supervised token count rather than a fixed
microbatch denominator.

Validation evaluates at most `eval_batches` batches from the beginning of the
validation split, using the configured batch size. The shipped presets use a
bounded subset. Increase that limit to cover the entire split when reporting
corpus-level results. Evaluation preserves random-generator state. Perplexity is
`null` with an explicit overflow flag when exponentiating loss would overflow.

## Resource and format boundaries

Weights and optimizer state are float32; mixed precision controls eligible
operations. Training materializes full `[batch, sequence, vocabulary]` logits
and an fp32 loss input. Include these allocations when choosing your batch size.

Generation uses cached attention keys and values, supports batched native decoding
and sampling controls, and stops at configured EOS or stop conditions. Exported
bundles include file hashes for integrity checks. Keep input artifacts unchanged
while a run is reading them.

Atomic directory publication uses Linux `renameat2(RENAME_NOREPLACE)`. Filesystems
must support file locking, hard links, and atomic renames.

## Epoch budgets and throughput controls

Set `epochs` to a positive integer to derive the optimizer horizon from packed
training rows. The final partial update belongs to its current epoch; rows are
not dropped. When `epochs` is set it determines the effective `max_steps`, and
warmup must fit that derived horizon. `stop_after_steps` remains an absolute
optimizer boundary within the effective horizon.

`init_checkpoint` can warm-start a new pretraining or SFT run from compatible
native weights. It requires matching model dimensions and tokenizer identity;
optimizer moments, schedule and data position start fresh. Exact continuation
uses `resume` instead.

The following optional `[training]` settings control execution:

| Setting | Default | Behavior |
|---|---:|---|
| `compile_model` | `false` | Compile the execution model with `torch.compile`. |
| `compile_backend` | `"inductor"` | Select `inductor`, `eager`, or `aot_eager`. |
| `activation_checkpointing` | `false` | Recompute decoder activations during backward. |
| `data_workers` | `0` | Load microbatches through this many CPU threads. |
| `pin_memory` | `false` | Pin host batches for asynchronous CUDA transfer. |
| `log_every` | `10` | Print progress, token throughput and ETA to stderr; zero disables it. |

Compilation has startup cost and may alter floating-point operation order.
Checkpointing exchanges additional computation for lower activation memory.
Execution settings are part of resume identity. Wall-clock progress stays out
of the deterministic metrics journal.

## Distributed training

Launch one process per device with `torchrun`; the ordinary train command selects
DDP when `WORLD_SIZE > 1`:

```bash
torchrun --standalone --nproc_per_node=2 -m clearforge train configs/pretrain-100m.toml
```

Use `device="cpu"` and `precision="fp32"` for CPU/Gloo; CUDA uses NCCL and
`LOCAL_RANK` to select each worker's device. The configured `batch_size` is per
rank. Global updates normally contain `batch_size × accumulation_steps ×
world_size` sequences. All ranks must use identical resolved configurations and
identical prepared artifacts. The output directory must be shared across ranks.

Each rank takes a disjoint slice of the same shuffled global cursor. Gradients
are weighted by the number of actual supervised tokens across all ranks, with
compensation for DDP's gradient averaging. An empty rank at an epoch tail runs a
zero-weight dummy microbatch, keeping collective counts aligned without adding
duplicate training supervision. Sparse expert models enable unused-parameter
tracking. Validation runs on rank zero and restores its RNG state.

Only rank zero writes the metrics journal and atomic checkpoint files. A
checkpoint stores unwrapped native model keys, shared optimizer/scheduler/scaler
state, the global data cursor, and separate Python/NumPy/Torch/CUDA RNG states
for every rank. Resume rejects changed world size, rank runtime or input
identities. A distributed checkpoint can initialize a new compatible run through
`init_checkpoint`; exact distributed resume requires the original topology.

SIGINT or SIGTERM received by any worker requests a coordinated checkpoint at an
optimizer boundary. Setup, input validation, journal and checkpoint errors are
reported collectively. Abrupt process or collective failures are handled by the
process-group timeout and torchrun worker supervision; recovery uses the last
durable checkpoint.
