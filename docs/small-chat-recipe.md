# Train a small English chat model

Train a 100M base model on English text, then fine-tune it on conversations
using supervised assistant responses.

## Complete pinned-corpus run

The repository includes a complete acquisition-to-export pipeline:

```bash
python -m pip install -e . -r requirements-training.txt
python scripts/train_100m.py --launch --root "$PWD/runs/english-100m"
```

This downloads three pinned FineWeb-Edu shards, validates their checksums,
deduplicates and splits documents, trains a 32,768-token BPE tokenizer on the
training split, and packs 1,024-token sequences. The 100,682,496-parameter model
then trains for 65,536 updates, with eight sequences per microbatch and four
accumulation steps: approximately 2.147 billion input positions. Training uses
bf16 on CUDA, with checkpoints and validation every 500 updates.

While pretraining runs, the pipeline prepares the pinned Smol-SmolTalk corpus.
It then fine-tunes the base for two epochs, evaluates the result, exports a native
chat bundle, and saves generations from the dataset's official held-out split.
Dataset revisions, licenses, source checksums and preparation manifests remain
inside the run directory.

The supervisor runs independently of the terminal. Inspect its stage and live
optimizer metrics with:

```bash
cat runs/english-100m/status.json
tail -f runs/english-100m/pretrain/metrics.jsonl
```

`pipeline.log`, `pretrain.log` and `sft.log` contain stage and worker output.
`control/` preserves the exact source and resolved configurations used for the
run, so subsequent repository changes do not change an active training job.
The latest durable weights are in `pretrain/latest.pt` or `sft/latest.pt`;
three checkpoints are retained per stage. Restart the same launch command to
resume a stopped pipeline after its previous supervisor and worker have exited.

After the first scheduled checkpoint, verify the running job and its identities:

```bash
python scripts/train_100m.py --verify-launch --root "$PWD/runs/english-100m"
```

The completed model and evaluation artifacts are `chat-model/`,
`evaluation.json`, and `heldout-generations.json` inside the run directory:

```bash
clearforge chat --model runs/english-100m/chat-model --prompt "Hello!" --stream
```

The following steps use your own corpora and allow a different training budget.

## 1. Assemble the training material

Provide two local UTF-8 JSONL files using the formats in [data preparation](data.md):

- `corpora/english-text.jsonl`: substantial, varied English text with a known license.
- `corpora/english-chat.jsonl`: high-quality user/assistant conversations, including
  ordinary questions, instructions, and multi-turn exchanges.

Prefer complete, readable documents and concise, accurate assistant answers.
Inspect language, duplication, broken formatting, answer quality, and domain
balance before training. Reserve a separate evaluation set.

Place your corpora at these paths and record their upstream revisions and
conversion procedures. Replace `YOUR_SOURCE_LICENSE` with the dataset license.

## 2. Build pretraining inputs

```bash
clearforge data prepare --source corpora/english-text.jsonl \
  --license YOUR_SOURCE_LICENSE --language en --mode pretrain \
  --output data/prepared-pretrain --validation-fraction 0.02
clearforge tokenizer train --input data/prepared-pretrain/train.jsonl \
  --output data/tokenizer --vocab-size 32768
clearforge data pack --input data/prepared-pretrain --tokenizer data/tokenizer \
  --output data/packed-pretrain --seq-length 1024 --mode pretrain
clearforge doctor --config configs/pretrain-100m.toml
```

Inspect `data/packed-pretrain/manifest.json`. Both splits must be nonempty, and
its vocabulary must match the preset. A smaller learned vocabulary requires the
same adjusted `model.vocab_size` in the pretraining and SFT configurations.

## 3. Set the budget and train the base

The default 100M preset runs 10,000 optimizer updates with four sequences per
microbatch and eight microbatches per update. At 1,024 tokens per sequence, that
is approximately 328 million input positions before accounting for padding,
shifts, masked boundaries, and partial groups. If the corpus is smaller, the
trainer cycles through it in a new deterministic order each epoch.

Choose `max_steps` before starting. Extending it during exact resume is rejected
because it changes the cosine schedule. Monitor held-out loss and generations
as training progresses.

```bash
clearforge train configs/pretrain-100m.toml --stop-after-steps 10
clearforge train configs/pretrain-100m.toml --resume runs/pretrain-100m/latest.pt
```

The first command verifies the real configuration and checkpoint path at ten
updates. The second continues the original schedule. Review actual memory and
storage use before a long run. The default keeps the latest three numbered
checkpoints; preserve a separately named copy if you need a particular earlier
base model for later experiments.

## 4. Fine-tune on conversations

```bash
clearforge data prepare --source corpora/english-chat.jsonl \
  --license YOUR_SOURCE_LICENSE --language en --mode sft \
  --output data/prepared-sft --validation-fraction 0.02
clearforge data pack --input data/prepared-sft --tokenizer data/tokenizer \
  --output data/packed-sft --seq-length 1024 --mode sft
clearforge train configs/sft-100m.toml
```

The SFT preset initializes from the completed base checkpoint, keeps the same
model and tokenizer, and starts a fresh optimizer at a lower learning rate.
The default budget is 2,500 updates. Adjust it using validation loss and sample
responses. Long conversations are windowed with one-token overlap;
later windows do not retain the complete earlier conversation. Curate examples
that fit the intended context when full conversational context is necessary.

## 5. Evaluate and use the export

```bash
clearforge evaluate configs/sft-100m.toml --checkpoint runs/sft-100m/latest.pt
clearforge export configs/sft-100m.toml --checkpoint runs/sft-100m/latest.pt \
  --output runs/chat-100m-export
clearforge chat --model runs/chat-100m-export --device cuda --max-new-tokens 128
```

Evaluate unseen prompts for readability, relevance, instruction following,
repetition, and multi-turn consistency. Compare checkpoints on tasks from your
intended application.
