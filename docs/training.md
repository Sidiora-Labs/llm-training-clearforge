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

Size labels are approximate. Exact parameter counts depend on vocabulary size
and embedding tying. A model's weight size is not its training memory requirement:
gradients, optimizer state, activations, and temporary attention buffers also
consume memory. Measure the selected sequence length and batch size on the
actual device before committing to a long run.

`max_steps` counts optimizer updates. `batch_size` counts sequences in a
microbatch; `accumulation_steps` controls the number of microbatches accumulated
for an update. The effective token count also depends on padding and supervised
positions. Metrics should be interpreted in tokens, not only examples.

## Precision and determinism

Use float32 on CPU. CUDA bfloat16 and float16 require compatible hardware and
runtime support. Mixed precision does not imply reduced-precision optimizer
state. Deterministic execution is useful for recovery testing and may reduce
throughput; reproducibility across different hardware or library versions is not
guaranteed.

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
fixed. A falling loss on a tiny fixture confirms optimization, not broad chat
quality. Use independent English prompts covering conversation, factual limits,
format adherence, repetition, and multi-turn consistency before describing a
model as useful.

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
settings, and recorded CUDA hardware. They intentionally reject changes that
would undermine the tested continuation guarantee. Resume equivalence was tested
with dropout on CPU and on the available CUDA device; it is not a guarantee
across arbitrary drivers, hardware, or environments.

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
and an fp32 loss input. There is no activation checkpointing, distributed
optimizer, or quantized training in this version. The larger presets have not
been qualified for production throughput or memory usage on every device.

Generation recomputes the sequence at every step and currently has no KV cache.
It supports batch size one, greedy decoding or temperature/top-k sampling, and
explicit EOS stopping. Exported hashes detect accidental changes; they do not
provide signed authenticity. Keep input artifacts immutable while reading them;
file checks are not a transactional snapshot of a filesystem under hostile
concurrent modification.

Atomic directory publication uses Linux `renameat2(RENAME_NOREPLACE)`. Filesystems
must support ordinary file locking, hard links, and atomic renames. The initial
release is qualified on local Linux storage, not network filesystems or Windows.
