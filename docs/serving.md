# Local chat service

Serve verified native exports through a browser interface or an OpenAI-compatible
chat completion endpoint. Models load once at startup; requests never select
arbitrary filesystem paths.

Install the server dependencies in the environment that contains Clearforge:

```bash
python -m pip install 'fastapi>=0.115,<1' 'uvicorn>=0.30,<1'
```

Start a local server with a named export:

```python
from clearforge.serving import serve

serve({"my-model": "runs/smoke-export"}, device="cpu")
```

Open `http://127.0.0.1:8000`. The interface includes model selection, conversation
history, system instructions, temperature, nucleus sampling, repetition penalty,
response length, stop, reset, and regenerate. Content renders as plain text;
model-generated markup never executes. All assets are served locally. Messages
and an optional API key live in browser memory and clear when the page reloads.
The server does not store conversation history: clients submit it with each turn.
Changing the selected model starts a new conversation.

## API

`GET /v1/models` lists loaded model IDs and context lengths. Register multiple
exports by passing additional name/path pairs to `serve` or `create_app`.
All registered models reside in memory, so account for their combined footprint.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":8,"temperature":0}'
```

A completion returns `id`, `object`, `created`, `model`, one assistant choice,
`finish_reason`, and prompt/completion/total token usage. Supported request fields:

| Field | Default | Meaning |
|---|---|---|
| `model` | required | Registered model ID |
| `messages` | required | Optional initial system, then alternating user/assistant; final user required |
| `max_tokens` | 128 | Requested response budget |
| `temperature` | 0.7 | Zero selects greedy decoding; accepted range 0–2 |
| `top_p` | 1 | Nucleus sampling threshold, greater than 0 and at most 1 |
| `top_k` | omitted | Optional vocabulary sampling limit |
| `repetition_penalty` | 1 | Positive repetition penalty, at most 10 |
| `seed` | omitted | Optional request-local sampling seed |
| `stream` | false | Enable server-sent events |
| `stream_options` | omitted | `{"include_usage":true}` adds a final usage chunk |
| `n` | 1 | Exactly one completion per request |

Unknown fields and unsupported message formats fail validation. A response budget
must fit the model context together with all submitted history. The server rejects
oversized requests instead of silently discarding messages. EOS counts toward
completion usage even though its structural token is omitted from displayed text.

For streaming, set `stream:true` and use `curl -N`. Events contain
`chat.completion.chunk` objects with a stable ID, role/content deltas, and a final
finish reason. The stream ends with `data: [DONE]`. If a generation fails after
headers are sent, an `error` object is emitted before `[DONE]`; clients must check
for that event. Errors detected before generation use HTTP status codes: 400 for
invalid requests or context budgets, 401 for authentication, 404 for unknown
models, 413 for oversized bodies, and 429 when another generation is active.
Nonstreaming model failures return 500 with a structured error.

## Access and resource controls

The default listener is loopback-only. To listen on another interface, supply
an API key explicitly, for example from an environment variable:

```python
import os
from clearforge.serving import serve

serve({"my-model": "runs/smoke-export"}, host="0.0.0.0", port=8000,
      api_key=os.environ["CLEARFORGE_API_KEY"], device="cuda",
      max_tokens=2048, max_body_bytes=1048576, generation_timeout=120)
```

API calls then require `Authorization: Bearer <key>`. The browser interface asks
for this key before connecting. Use a TLS reverse proxy for access beyond a
trusted local network; the built-in listener uses HTTP. No cross-origin access
is enabled. UI assets are public, while both API endpoints require the configured
key. Request bodies and credentials are not logged by the application.

One process permits one generation at a time across its loaded models, protecting
model state and request-local random seeds. Concurrent requests receive 429 rather
than entering an unbounded queue. Defaults limit requests to 1 MiB, 256 messages,
4,096 generated tokens and 120 seconds. The tokenizer's context limit may be lower.
The worker's output queue is bounded to 32 events. Client disconnection or Stop
signals cancellation; the worker checks between tokens and closes the generator
before releasing its lock. An already-running device operation must finish first.
A stalled client or timeout also cancels generation. Streaming buffers incomplete
UTF-8 byte sequences to avoid corrupting characters at token boundaries.

Use a single server process for a single loaded model set. Multiple processes
load separate copies of the models and hold independent locks. The server loads
only native inference exports, verifies their hashes, and never loads optimizer
checkpoints through API requests.

For ASGI integration, `create_app(model_paths, device=..., api_key=..., ...)`
returns a FastAPI application with the same limits as `serve`.

## Function-call conversations

Requests may supply a `tools` list of function definitions and assistant/tool
messages using `tool_calls`, `tool_call_id`, and optional `reasoning_content`.
Definitions and call arguments use the [validated tool schema](data.md#tool-conversations-and-reasoning-fields).
Tool definitions are included in the model's system context, so they count against
the prompt budget. A generation request may end after all outstanding tool results
have been returned, as well as after a user message. Missing, duplicate, unknown,
or partially answered call IDs fail before generation starts.

A model's complete structured assistant output is parsed and validated before
returning function calls. Nonstreaming responses expose `message.tool_calls` and
optional `message.reasoning_content`. Calls set `finish_reason` to `tool_calls`.
With SSE, ordinary text streams immediately; a structured response is buffered
until complete and valid, then emitted as a delta containing content, optional
reasoning, and indexed tool calls. Function arguments are canonical JSON strings.
Malformed or undeclared calls produce an error, never a partially validated call.

`tool_choice` accepts `auto` (default) or `none`. With `none`, generated calls are
rejected. Forced or named tool choices are not accepted. The server returns calls
but does not execute them; the client controls function dispatch, appends matching
`tool` result messages, and submits the next request. The browser chat interface
is for text conversations; use the API or rollout client for tool execution.

## Browser smoke check

An optional Chromium check exercises the real browser against a running service:
streamed generation, conversation history, regenerate, Stop, reset, server-error
recovery, model selection when multiple models are loaded, and the mobile settings
panel. It also captures desktop/mobile screenshots and JavaScript console results.
Use a local test export with at least a 256-token context; the check sends short
prompts and requests up to 128 response tokens.

Keep browser tooling separate from the training environment:

```bash
python -m venv /tmp/clearforge-browser-env
/tmp/clearforge-browser-env/bin/python -m pip install playwright
/tmp/clearforge-browser-env/bin/python -m playwright install --with-deps chromium
/tmp/clearforge-browser-env/bin/python tests/browser_smoke.py \
  --url http://127.0.0.1:8000 --output /tmp/clearforge-browser-check
```

Start the service first. Set `CLEARFORGE_API_KEY` when its API requires
authentication, or select another variable with `--api-key-env`. The check runs
explicitly and is not collected by the ordinary pytest suite. Its intentional
invalid request produces an expected HTTP 400 browser diagnostic; JavaScript
exceptions and other unexpected console errors fail the check.
