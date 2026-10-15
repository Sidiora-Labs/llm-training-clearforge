# Generation evaluation

Run a versioned JSONL suite against a native model export:

```bash
clearforge evaluate-generation --model runs/chat-export \
  --cases examples/evaluation.jsonl --output runs/chat-evaluation.json \
  --max-new-tokens 128 --device cuda
```

Each record contains `messages`, an optional unique `id`, optional `tools`, and
one or more checks:

| Field | Check |
|---|---|
| `expected` | Exact response text |
| `contains` | Every listed substring occurs in the response |
| `expected_json` | Parsed response equals the supplied JSON value |
| `expected_tool_calls` | Function names, arguments and call order match; correlation IDs are ignored |

Cases without checks are saved as generation samples and excluded from the
pass-rate denominator. The report records individual answers, check results,
token counts, finish reasons and the sampling seed. Invalid cases fail before
any generation. Model or transport failures propagate to the caller.

The Python `evaluate_generations` function accepts a native or HTTP rollout
backend and `GenerationOptions`. The CLI uses greedy native generation and
restores the caller's random-generator state after evaluation.

## Tokenizer diagnostics

```bash
clearforge tokenizer inspect --tokenizer data/tokenizer \
  --input examples/pretrain.jsonl
```

This streams a JSONL text corpus and reports token count, vocabulary size,
characters and bytes per token, longest record, and exact text round-trip
failures. Literal chat-control spellings are encoded as text.
