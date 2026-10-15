# Implementation contracts

These interfaces are the integration boundary for the first release. Changes
require coordination with the component owners and matching integration tests.

## Configuration

`clearforge.config` provides frozen dataclasses and `load_config(path)`:

- `ModelConfig`: vocab_size, hidden_size, num_layers, num_heads, num_kv_heads,
  intermediate_size, max_seq_len, dropout, rope_theta, tie_embeddings.
- `TrainConfig`: max_steps, batch_size, accumulation_steps, learning_rate,
  weight_decay, warmup_steps, min_lr_ratio, grad_clip, seed, device, precision,
  save_every, eval_every, eval_batches, deterministic, keep_checkpoints.
- `RunConfig`: model, training, data_dir, tokenizer_path, output_dir, stage
  (`pretrain` or `sft`), init_checkpoint (optional).

Run paths resolve relative to the configuration file. Reject unknown keys,
invalid dimensions and invalid optimization parameters before allocating a model.

## Data and tokenizer

`clearforge.data.prepare_sources(sources, output_dir, *, validation_fraction,
seed, mode)` accepts `SourceSpec(path, license, language='en')` and creates
`train.jsonl`, `validation.jsonl`, and `manifest.json`. Inputs are UTF-8 JSONL:
pretraining records have `text`; supervised chat records have `messages` with
`role` and `content`. Validation splits are disjoint for exact canonical complete records.
Source hashes, licensing, exclusions, and split hashes are recorded. No inference
service or private conversation corpus is used implicitly.

`clearforge.tokenizer.train_tokenizer(train_path, output_dir, *, vocab_size,
min_frequency=2)` returns the saved tokenizer path. `ChatTokenizer.load(path)`
returns an object exposing `vocab_size`, `pad_id`, `bos_id`, `eos_id`,
`encode(text)`, `decode(ids, *, skip_special_tokens=False)`, `save(path)`, and
`encode_chat(messages, *, add_generation_prompt=False) -> (ids, labels)`.
Labels align with input IDs; user/system/padding labels are -100. The model,
not the dataset, applies the one-token next-token shift. Define role tokens
as reserved tokens. Plain content must not inject control-token IDs.

`clearforge.data.pack_dataset(prepared_dir, tokenizer_path, output_dir, *,
seq_length, mode)` produces memory-mappable arrays named
`train.input_ids.npy`, `train.labels.npy`, `validation.input_ids.npy`, and
`validation.labels.npy`, plus `manifest.json` with tokenizer and source hashes,
vocabulary size, sequence length, mode, and row counts. Tokenizer training uses
only training records. Pretraining packs documents with BOS/EOS boundaries and shared causal attention.
SFT must preserve assistant-only labels and exclude windows with no shifted
supervision. Never silently truncate a conversation. Explicitly record any
exclusions. `PackedDataset(data_dir, split)` implements length/getitem returning
`{'input_ids': LongTensor, 'labels': LongTensor}`. It must remain lazy/memory mapped.

## Model

`clearforge.model.CausalLM(ModelConfig)` is a PyTorch module.
`forward(input_ids, labels=None)` returns an object with `.logits` and `.loss`.
It uses causal SDPA, RoPE, RMSNorm, SwiGLU, grouped-query attention and optional
tied embeddings. Ignore -100 labels and shift labels once. The current implementation materializes full sequence logits; memory planning
must include them. Validate sequence length and head divisibility.
Provide `parameter_count()` and `generate(input_ids, *, max_new_tokens,
temperature=0.0, top_k=None, eos_token_id=None)` for batch-one or documented
batch behavior, with context-bound checks and EOS stopping.

## Training and checkpoints

`clearforge.training.train(config, *, resume=None, stop_after_steps=None)`
orchestrates native pretraining or supervised fine-tuning. CPU float32 works; CUDA bf16/fp16 are
capability-checked. Training steps count optimizer updates. Implement AdamW,
correct accumulation including final partial batches, warmup/cosine scheduling,
clipping, validation, and JSONL metrics. Reject all-ignored supervision and
nonfinite loss/gradients. No external logging service is enabled implicitly.

Checkpoints atomically persist model, optimizer, scheduler/step, RNG state
(Python/NumPy/Torch/CUDA), data-order/cursor, configuration, tokenizer identity,
and dataset identity. Resume rejects incompatible inputs before changing outputs.
Initialize SFT from a base checkpoint without resuming its optimizer or cursor.
Never overwrite another run unintentionally. Save at optimizer boundaries and
on controlled interruption. Resume-equivalence must be demonstrated with a
nonzero-dropout CPU fixture. Use trusted local checkpoint loading only; document
that pickle-based checkpoints are not safe to load from untrusted sources.

`clearforge.training.evaluate(config, checkpoint)` reports token-weighted loss
and perplexity on the validation split. An export function creates a portable
native bundle with model configuration, weights, tokenizer, and provenance.
No unsupported format should be advertised.

## CLI and documentation

The root integrates argparse commands: `data prepare`, `tokenizer train`,
`data pack`, `train`, `evaluate`, `export`, `chat`, and `doctor`.
Document the example workflow and the full pretraining-to-chat recipe.
Provide nominal 100M, 300M, and 1B presets plus a tiny smoke-test preset.
Record dataset and model licenses alongside their artifacts.
