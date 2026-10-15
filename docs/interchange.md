# Transformers interchange

Clearforge converts dense decoders with standard RoPE to the built-in Transformers
`LlamaForCausalLM` architecture. Exports contain `config.json`,
`model.safetensors`, a fast tokenizer, and a Jinja chat template. They load through
`AutoModelForCausalLM.from_pretrained` without custom model code or
`trust_remote_code=True`.

Install `transformers` and `safetensors` in the environment used for conversion.
The integration is qualified with Transformers 4.57.6. Conversion operates on
local files and does not download weights or contact model services.

## Export a native bundle

```python
from clearforge.interchange import export_bundle_transformers

export_bundle_transformers("runs/chat-export", "runs/chat-transformers")
```

To export an in-memory model, use
`export_transformers(model, tokenizer_path, output_dir)`. Do not mutate model
weights while exporting. Output directories must be new. Publication is atomic
and refuses existing destinations.

Load the result with standard tools:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "runs/chat-transformers"
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    path, local_files_only=True, use_safetensors=True
).eval()
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain how a library works."}],
    add_generation_prompt=True, return_tensors="pt",
)
output = model.generate(inputs, max_new_tokens=64, do_sample=False)
print(tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True))
```

The export permutes query and key projection rows from interleaved rotary pairs
to Llama's split-half coordinates. Value/output projections, GQA head counts,
SwiGLU weights, RMSNorm, and tied embeddings retain their semantics. Numerical
tests compare logits through the actual standard loader, then verify a native
round-trip preserves every float32 parameter exactly.

## Import a compatible local checkpoint

```python
from clearforge.interchange import import_transformers

import_transformers("runs/chat-transformers", "runs/imported-chat")
```

The result is a native inference bundle, usable by `clearforge chat --model`.
It is not a training checkpoint: optimizer, schedule, RNG, and data cursor state
are not available in a Transformers weights export.

Imports require standard Llama weights and a tokenizer containing Clearforge's
six control tokens with matching BOS/EOS/PAD IDs. This supports compatible
checkpoints rather than arbitrary published Llama models. The imported native
chat renderer uses Clearforge's message format; a checkpoint trained with a
different template needs an explicit format conversion before chat use.
Sharded safetensors archives are supported. Pickle weights, remote Python code,
quantized checkpoints, mismatched tensor keys, nonfinite weights, and inconsistent
tied embeddings are rejected. Imported weights use float32.

## Architecture and tokenizer boundaries

MoE, YaRN, non-SiLU activations, projection biases, nonstandard head dimensions,
and normalization epsilon other than `1e-6` are rejected explicitly. The mapping
preserves inference behavior. Transformers Llama does not reproduce Clearforge's
embedding and residual dropout during training; a nonzero native dropout setting
is recorded for round-trip restoration, not a claim of identical training dynamics.

Ordinary system/user/assistant text produces the same template token IDs in both
runtimes. Native tokenization safely encodes literal reserved control spellings
inside message content. Standard fast tokenizers match those spellings as special
tokens, so the exported template rejects such content rather than changing its
meaning. Use native tokenization for that case. This restriction concerns the
six control spellings, not ordinary angle brackets or Unicode text.

`interchange.json` records export file hashes and the source tokenizer hash.
Native imports record hashes of their source model/configuration/tokenizer files
in `provenance.json`. These hashes establish file identity, not publisher trust.

## Tools and reasoning messages

The exported template also renders tool definitions, structured assistant calls,
reasoning fields, and completed parallel tool-response batches using the same
vocabulary-preserving envelopes as the native tokenizer. Token-ID parity is
verified for Unicode, system-message injection, reasoning, and multi-turn tools.

Tool-call arguments may be mappings, which the template canonicalizes, or already
canonical JSON strings. Jinja cannot parse arbitrary JSON strings. Before passing
provider messages with string arguments directly to `apply_chat_template`, call
`clearforge.tools.validate_conversation(messages, tools=tools)` to normalize them,
or decode each arguments string into a mapping. The rollout HTTP client performs
this normalization automatically. Validate tool schemas and calls in application
code before dispatch; a chat template is a renderer, not an execution validator.
