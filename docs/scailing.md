# Scaling to a billion parameters

This guide sizes a native 1B pretraining run and the corpus behind it. Every
number below comes from `clearforge plan`, which computes the same estimates
for any configuration; run it against your own file before trusting a budget.

```bash
clearforge plan configs/pretrain-1b.toml --device cuda --device-memory-gib 80
clearforge plan configs/pretrain-1b.toml --world-size 8 --device cuda --device-memory-gib 80
```

## The 1B preset

`configs/pretrain-1b.toml` defines a 1,041,852,416-parameter decoder: width
2048, 22 layers, 16 query and 4 key/value heads, SwiGLU width 5504, a
1,024-token context and tied embeddings over the shipped 32,768-token
vocabulary. The training section is tuned for long runs:

| Setting | Value | Reason |
|---|---:|---|
| `batch_size × accumulation_steps` | 4 × 64 | 262,144 tokens per update on one device. |
| `max_steps` | 80,000 | About 21 billion tokens, 20 tokens per parameter. |
| `learning_rate`, `warmup_steps` | 4e-4, 2,000 | Peak rate and warmup common for 1B decoders. |
| `lr_schedule`, `min_lr_ratio` | cosine, 0.1 | Decay to 4e-5 at the final update. |
| `adam_beta2` | 0.95 | Faster second-moment adaptation for large batches. |
| `optimizer_fused` | true | Fused AdamW kernel on CUDA. |
| `activation_checkpointing` | true | Keeps activations under one gibibyte at this batch. |
| `save_every`, `keep_checkpoints` | 1,000, 3 | About 47 GiB of retained checkpoints. |

`configs/sft-1b.toml` fine-tunes the resulting base on chat data with
65,536 supervised tokens per update, and `configs/pretrain-300m.toml` is the
same recipe at 304M parameters and 6.5 billion tokens.

## Memory

Weights, gradients and the two AdamW moments are float32: sixteen bytes per
parameter, 15.5 GiB for the 1B model before a single activation. Mixed
precision adds a two-byte weight cache. Activations and logits scale with the
microbatch, so `batch_size` is the lever that fits a run onto a device, while
`accumulation_steps` and the torchrun world size set the tokens per update
without changing memory. Estimates for the 1B preset at 1,024 tokens:

| `batch_size` | With activation checkpointing | Without |
|---:|---:|---:|
| 1 | 18.1 GiB | 19.7 GiB |
| 2 | 18.7 GiB | 21.8 GiB |
| 4 | 19.9 GiB | 26.2 GiB |
| 8 | 22.3 GiB | 34.9 GiB |
| 16 | 27.2 GiB | 52.4 GiB |

The estimate is deliberately conservative about the full-vocabulary logits
and their float32 loss copies, which cost about 0.44 GiB per sequence. Leave
headroom for the allocator and measure the `max_memory_allocated_gib` field in
the progress log during a short run before committing to a batch size. A 24 GiB
device runs the preset at `batch_size = 4`; 40 GiB and 80 GiB devices can raise
the microbatch and lower `accumulation_steps` to keep 262,144 tokens per update.

## Throughput and wall-clock

Training costs about `6 × parameters` floating-point operations per token:
1.3 × 10²⁰ for the preset's 21 billion tokens. The progress log reports the
achieved rate as `tflops`. At 400 TFLOPS sustained, a figure a single modern
accelerator reaches with bf16 and a healthy batch, the model processes about
64,000 tokens per second and the full schedule takes roughly 90 hours; eight
devices under `torchrun` divide that proportionally when the interconnect keeps
up. Compare the logged `tflops` with your hardware's peak to see whether data
loading (`data_workers`, `pin_memory`) or the microbatch is the bottleneck.

Keep `deterministic = true` for runs whose resume must be bit-exact. It
disables TF32 and nondeterministic kernels, which costs some throughput on
CUDA; a run that only needs reproducible data order and checkpoints can turn it
off at the cost of exact replay.

## Token budget and schedule

Twenty tokens per parameter is the compute-optimal reference from the Chinchilla
scaling study; models meant for inference are often trained well past it.
`clearforge plan` reports tokens per parameter and warns below the reference.
Budget before the run starts: `max_steps` is part of the cosine horizon and an
exact resume rejects a changed value.

The `wsd` schedule separates the budget decision from the data decision. A
stable phase at the peak learning rate can be checkpointed, evaluated and
continued into a cooldown of `decay_steps` updates whenever the corpus is
exhausted. To continue a finished stable-phase run on an expanded corpus, pack
the new data, start a new run with `init_checkpoint` pointing at the base
checkpoint, a short warmup and the same peak rate; optimizer moments start
fresh and the metrics journal records the boundary.

## Multiple devices

```bash
torchrun --standalone --nproc_per_node=8 -m clearforge train configs/pretrain-1b.toml
```

`batch_size` is per rank. Eight ranks at the preset's microbatch raise the
global update to 2,097,152 tokens; lower `accumulation_steps` to 8 to keep the
original 262,144, or keep the larger batch and raise `learning_rate` with care.
Rank zero runs the preflight plan with the launch's world size, owns the output
directory, and refuses a volume that cannot hold one checkpoint. Every rank
needs the memory in the table above; DDP replicates the optimizer state.

## Expanding the corpus

Twenty-one billion tokens is roughly 20 million packed sequences. Plan the
pipeline for that scale:

- **Describe sources in a manifest.** Each `[[source]]` carries its own license,
  language, size bounds, deterministic `sample_fraction` and `max_records`, so
  adding a corpus is a manifest change with an audit trail in `provenance.jsonl`
  and per-source counts in the preparation manifest. See
  [data preparation](data.md#corpus-manifests-and-selection).
- **Budget disk for packed arrays.** Arrays use the smallest integer width the
  vocabulary allows: with 32,768 tokens that is four bytes per position for IDs
  and labels together, about 78 GiB for 21 billion tokens, rather than 312 GiB
  as int64 pairs. The packing manifest records `token_counts`, `array_dtypes`
  and `row_counts`; `clearforge plan` reports packed bytes and how many passes
  the schedule makes over the rows.
- **Expect packing to be tokenizer-bound.** Documents are encoded in batches
  through the Rust tokenizer and cut into windows with array operations; the
  Python per-token loop is gone. `TOKENIZERS_PARALLELISM` controls the encoder's
  threads. Progress lines appear on stderr every 50,000 documents.
- **Keep startup verification in mind.** Training hashes every packed array
  before allocating the model. At one gibibyte per second that is a little over
  a minute for 78 GiB; it is the price of the dataset identity that exact
  resume enforces.
- **Reserve validation data independently.** Validation is a fixed fraction of
  deduplicated records; raise `eval_batches` for corpus-level numbers and keep
  external benchmarks out of both tokenizer training and pretraining.

## Checkpoints and disk

A 1B checkpoint holds float32 weights and both AdamW moments: 11.6 GiB plus the
data cursor. With `keep_checkpoints = 3` and one checkpoint in flight the run
needs about 47 GiB under `output_dir`, which the preflight check confirms
before the first update and `clearforge plan` reports as `retained_bytes`.
Each save writes the full file through a temporary name and renames it, so a
checkpoint interval of 1,000 updates adds a few seconds of serialization per
hour at the preset's throughput. Exported chat bundles hold only the 3.9 GiB of
weights.

## Before launching

1. `clearforge doctor` to confirm the accelerator, bf16 support and free disk.
2. `clearforge plan` with the real `data_dir` to check tokens per parameter,
   epochs over the packed rows, memory against the device and disk against
   retention.
3. `clearforge train CONFIG --stop-after-steps 20`, then read the progress
   log's `tflops` and `max_memory_allocated_gib` and the `plan` field in
   `run.json`.
4. `clearforge train CONFIG --resume runs/<name>/latest.pt` to continue the
   schedule once the short run looks right.
