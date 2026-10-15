# Generation backends and tool episodes

`clearforge.rollout` provides native cached generation, OpenAI-compatible HTTP
requests, and bounded multi-turn tool episodes. It does not supply executable
shell tools or assign rewards implicitly.

## Backends

`NativeBackend(model, tokenizer)` uses `stream_generate` and the native structured
assistant parser. `OpenAIBackend(base_url, model, tokenizer,
context_window=...)` calls a configured `/chat/completions` endpoint;
`SGLangBackend` uses the same OpenAI-compatible protocol. Include the API prefix
in the base URL, such as `http://127.0.0.1:30000/v1`.

All backends implement:

```python
completion = backend.complete(
    messages,
    tools=tool_definitions,
    generation=GenerationOptions(max_new_tokens=128, temperature=0.8, top_p=0.9),
)
```

A `Completion` contains the validated assistant `message`, `prompt_tokens`,
`completion_tokens`, and `finish_reason`. Native completions also contain exact
`prompt_ids` and `generated_ids`. `NativeBackend.traces` retains the latest 128
sampled turns, including outputs rejected by the structured parser and partial
outputs at a deadline. Training callers should clear this list before each
tracked episode. HTTP completions do not fabricate token sequences.

HTTP endpoints must use the same tokenizer and chat template as the local client.
The client checks reported prompt usage against the local encoding, checks token
budgets, requires one completion, and bounds response bytes and request time.
Redirects, environment proxies, missing usage, malformed responses, and unknown
finish reasons are rejected. TLS uses normal certificate verification. Optional
`api_key` values are sent in an Authorization header, not URL parameters.
Sampling extensions `top_k` and `repetition_penalty` are sent only when requested;
the configured server must support those fields. Requests are never retried
automatically.

## Explicit tool registration

Register trusted, importable top-level Python functions with JSON schemas:

```python
from clearforge.rollout import ToolRegistry

def add(a, b):
    return {"sum": a + b}

registry = ToolRegistry()
registry.register({
    "type": "function",
    "function": {
        "name": "add",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}, add)
```

Callbacks execute in spawned processes. They must be pickleable/importable;
lambdas and local closures are rejected. Place executable script entry points
under `if __name__ == "__main__":` as required by Python multiprocessing.
`functools.partial` of an importable function is supported. Workers enforce a
wall-clock timeout and result byte limit, and are terminated/reaped on failure.
Returned strings are used directly; other results must be finite JSON and are
serialized canonically. Child Python state does not mutate parent Python state.

These workers isolate execution lifecycle, not filesystem or network authority.
Registered handlers remain trusted application code. Their external side effects
are not rolled back by a later error. No handler is discovered or imported from
model output.

## Bounded episodes and rewards

```python
from clearforge.rollout import EpisodeLimits, GenerationOptions, run_episode

result = run_episode(
    backend,
    [{"role": "user", "content": "Add two and three."}],
    registry,
    generation=GenerationOptions(max_new_tokens=256, temperature=0.8),
    limits=EpisodeLimits(max_turns=4, max_tool_calls=4,
                         max_generated_tokens=1024, tool_timeout=5),
)
```

Each call batch is validated against the complete history and registered schemas
before executing any call in that batch. Call IDs cannot be reused. Results are
added as tool messages, then the assistant continues. Calls are not executed after
a truncated completion or on the last allowed turn. Context is never silently
truncated.

`EpisodeResult` records messages, accepted completions, token accounting, tool
observations, status, and an optional error. Status is `complete`, `length`,
`turn_limit`, `token_limit`, `tool_limit`, `output_limit`, or `error`. A partial
failed batch retains executed tool observations separately; incomplete tool
responses are not appended to the conversation. Applications can derive rewards
from these recorded outcomes and exact native sampling traces.

Limits cover turns, calls, generated tokens, total output bytes, individual tool
result bytes, and elapsed time. Native deadlines are checked between generated
tokens; an individual accelerator operation is not preempted. HTTP body reads
check the deadline between bounded reads. Platform DNS resolution is not governed
by Python's socket timeout. Reward callbacks and trainer checkpoint identity must
record the application's tool registry identity separately from the JSON schemas.
