# Data preparation

Clearforge accepts UTF-8 JSONL. Each line must be a complete JSON object; blank lines are rejected. Treat
source preparation as a reproducible build: retain the source version, declared
license, language, hashes, filtering counts, tokenizer, and packed-data manifest.

## Input contracts

Pretraining:

```json
{"text":"A compiler translates a program into another executable representation."}
```

Supervised conversation:

```json
{"messages":[{"role":"user","content":"What is a compiler?"},{"role":"assistant","content":"A compiler translates source code into another representation, often machine code."}]}
```

A conversation may begin with a system message. User and assistant turns must
follow the supported conversation order. Tool records are not silently converted
into ordinary assistant answers. Malformed data must be corrected rather than
allowed to contaminate a run unnoticed.

## English-first sources

Declare English sources explicitly and inspect their language distributions
before training. Clearforge records the supplied language in the manifest.

Two public starting points are [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
for English pretraining and [UltraChat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
for conversation fine-tuning. Read each dataset card and its current terms before
selecting a version. Pin an immutable revision and retain the upstream split and
any conversion procedure. These datasets are not downloaded automatically.

The repository includes original English example datasets under Apache-2.0.

## Split discipline

Deduplicate documents before creating training and validation partitions. Train
the tokenizer on the training partition only. Keep external benchmarks out of
both tokenizer training and model training. Review paraphrases, template overlap, and benchmark contamination during corpus curation.

Train and validation partitions must both contain supervised tokens. Very small
input files may fail this requirement; add independent examples instead of
copying training records into validation.

## Tokenizer and packing

The tokenizer is byte-level BPE with explicit conversation delimiters. A requested
vocabulary size is an upper bound on what a small corpus can learn; inspect the
actual vocabulary before training. Native model vocabulary must equal that
vocabulary, and fine-tuning must preserve its identity.

Pretraining concatenates documents with explicit end-of-document boundaries.
That is not the same as resetting attention between documents. Chat packing must
preserve role masks and report unsupported or excluded records. Inspect the
packing manifest rather than assuming every input character was retained.

Code licensing does not grant rights to training corpora or resulting models.
Do not mix secrets or private conversations into a public training corpus.

See [data pipeline details](data-pipeline.md) for exact integrity checks,
windowing behavior, and the context limits of long SFT conversations.
