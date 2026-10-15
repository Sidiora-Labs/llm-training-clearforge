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
