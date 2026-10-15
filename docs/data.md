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

## Tool conversations and reasoning fields

The native conversation schema also supports function calls and tool results.
Keep tool definitions in a record's top-level `tools` list:

```json
{"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object","properties":{"x":{"type":"integer"},"y":{"type":"integer"}},"required":["x","y"],"additionalProperties":false}}}],"messages":[{"role":"user","content":"Add two and three."},{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"add","arguments":{"x":2,"y":3}}}]},{"role":"tool","tool_call_id":"call_1","content":"5"},{"role":"assistant","content":"The answer is 5."}]}
```

Assistant messages may include `reasoning_content` as a string. It is explicit
training data, supervised along with the assistant's answer; it is not synthesized
by the tokenizer. Call-only assistant messages may omit `content` or use `null`,
which normalizes to an empty string. Other content must be text.

Calls require a unique ID, `type: "function"`, a function name, and JSON-object
arguments. Native APIs accept arguments as either an object or a JSON string;
strings are parsed strictly and canonicalized. Duplicate keys and nonfinite JSON
numbers are rejected. Each result must match an outstanding call ID exactly once.
Parallel results may arrive in any order, but all must arrive before another
assistant turn. Call IDs cannot be reused within a conversation. Orphan results,
partial result batches, unknown functions, invalid arguments, and unexpected fields
fail with a locator rather than being silently rewritten.

A conversation ending in an assistant call is a valid training target. It cannot
be used directly as a generation prompt: provide every matching tool result first.
Generation prompts end with a user message or a fully answered tool-call batch.

Tool schemas support explicit JSON types, properties, required keys, boolean
`additionalProperties`, array items, enums, numeric bounds, string lengths, and
array lengths. Descriptions and titles are retained. Unsupported constructs such
as `$ref` and composition keywords are rejected. When definitions are supplied,
call names and arguments are checked against them. Without definitions, call
structure and result state are still validated, but function-specific arguments
cannot be checked.

### Stable serialization and inference

No vocabulary changes are required. Plain conversation token IDs remain exactly
the same as before tool support. Tool definitions append a deterministic JSON
block to the system context, or create a system message if absent. Extended
assistant messages use a `<clearforge_assistant>` JSON envelope inside the existing
assistant boundary; tool results use a `<clearforge_tool>` JSON envelope inside a
user boundary. These envelope markers are ordinary content bytes, not added
special tokens. Assistant envelopes and EOS are supervised; schemas and tool
results are masked. Literal reserved-token spellings in native content cannot
inject role boundaries.

```python
from clearforge.tools import parse_assistant_output

ids, labels = tokenizer.encode_chat(messages, tools=tools)
prompt_ids, _ = tokenizer.encode_chat(
    messages[:-1], tools=tools, add_generation_prompt=True
)
# After generating and decoding the assistant's output:
assistant_message = parse_assistant_output(decoded_text, tools=tools)
```

The parser accepts ordinary text or a complete structured envelope, validates
calls, and returns a normalized assistant message. It does not execute functions.
Dispatch only through an explicitly supplied function registry; append results
with their matching IDs before continuing generation. A malformed structured
output raises an error rather than being treated as executable tool arguments.

`tokenizer.apply_chat_template(messages, tools=..., tokenize=True)` returns native
IDs. With `tokenize=False`, it returns readable rendered text. To preserve literal
control-token safety, use native tokenization directly rather than re-encoding the
rendered string through an arbitrary tokenizer.

### Explicit dataset adapters

Use named adapters to convert another supported corpus format. Native preparation
does not silently interpret `conversations`, preference pairs, or rollout rows:

- `adapt_conversations(record)` converts `conversations` plus optional `tools` to
  native `messages`.
- `adapt_preference(record)` accepts complete `chosen`/`rejected` conversations, or
  a `prompt` string/message list plus two response strings. Both alternatives must
  share the same prompt, end with an assistant, and differ. Optional `tools` survive
  normalization.
- `adapt_rollout(record)` accepts exactly one of `prompt` or `messages`, validates
  generation readiness, and retains optional tools and JSON ground truth. `gt`
  explicitly maps to `expected`; specifying both is an error.

```python
from clearforge.data import SourceSpec, convert_dataset

converted = convert_dataset(
    SourceSpec("conversations.jsonl", license="Apache-2.0", language="en"),
    "data/converted", adapter="conversations",
)
```

File conversion streams one record at a time into `data.jsonl`, records the source
hash, license, language, row count, and output hash in a manifest, and refuses
output collisions. Malformed rows abort publication with a file/line locator.
Conversion preserves row order and does not split or filter by language. Feed a
converted chat corpus into ordinary SFT preparation; preference and rollout rows
belong to their respective post-training objectives. Retain the conversion
manifest alongside later preparation manifests for full lineage.
