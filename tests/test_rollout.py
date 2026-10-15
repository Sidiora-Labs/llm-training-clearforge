"""Real native inference, local HTTP exchanges, and isolated tool episodes."""

import json
import threading
import time
from contextlib import contextmanager
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from clearforge.config import ModelConfig
from clearforge.model import CausalLM
from clearforge.rollout import (
    EpisodeLimits,
    GenerationOptions,
    NativeBackend,
    OpenAIBackend,
    RolloutError,
    SGLangBackend,
    ToolRegistry,
    run_episode,
)
from clearforge.tokenizer import SPECIAL_TOKENS, ChatTokenizer


def addition(a, b):
    return {"sum": a + b}


def write_marker(path, a, b):
    Path(path).write_text("ran")
    return a + b


def sleep_long():
    time.sleep(5)
    return "late"


def huge_output():
    return "x" * 1000


TOOL = {
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
}
CALL = {
    "id": "call1",
    "type": "function",
    "function": {"name": "add", "arguments": '{"a":2,"b":3}'},
}


@pytest.fixture
def tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["Hello world, add two numbers. The sum is five."],
        trainers.BpeTrainer(
            vocab_size=280,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        ),
    )
    return ChatTokenizer(tokenizer)


@contextmanager
def endpoint(callback):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            status, body, headers = callback(request)
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def response(tokenizer, request, message, reason="stop", completion_tokens=5):
    ids, _ = tokenizer.encode_chat(
        request["messages"], tools=request.get("tools", []), add_generation_prompt=True
    )
    body = {
        "choices": [{"index": 0, "message": message, "finish_reason": reason}],
        "usage": {"prompt_tokens": len(ids), "completion_tokens": completion_tokens},
    }
    return 200, json.dumps(body).encode(), {"Content-Type": "application/json"}


@pytest.mark.parametrize("backend_class", [OpenAIBackend, SGLangBackend])
def test_http_multiturn_tool_episode(tokenizer, backend_class):
    def answer(request):
        if request["messages"][-1]["role"] == "tool":
            assert json.loads(request["messages"][-1]["content"]) == {"sum": 5}
            return response(tokenizer, request, {"role": "assistant", "content": "Five."})
        return response(
            tokenizer,
            request,
            {"role": "assistant", "content": None, "tool_calls": [CALL]},
            "tool_calls",
        )

    registry = ToolRegistry()
    registry.register(TOOL, addition)
    with endpoint(answer) as (url, requests):
        backend = backend_class(url, "local-model", tokenizer, context_window=4096)
        result = run_episode(backend, [{"role": "user", "content": "Add 2 and 3"}], registry)
    assert result.status == "complete", result.error
    assert result.turns == 2 and result.completion_tokens == 10
    assert result.messages[-1]["content"] == "Five."
    assert result.tool_events[0]["result"] == '{"sum":5}'
    assert len(requests) == 2
    assert requests[0]["stream"] is False


def test_native_backend_real_cached_model_and_trace(tokenizer):
    torch.manual_seed(3)
    model = CausalLM(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            num_heads=2,
            num_kv_heads=1,
            num_layers=1,
            intermediate_size=32,
            max_seq_len=128,
        )
    ).train()
    backend = NativeBackend(model, tokenizer)
    prompt = [{"role": "user", "content": "Hello"}]
    completion = backend.complete(prompt, generation=GenerationOptions(max_new_tokens=3))
    assert completion.message["role"] == "assistant"
    assert completion.completion_tokens == len(completion.generated_ids)
    assert completion.prompt_tokens == len(completion.prompt_ids)
    assert backend.last_trace["generated_ids"] == completion.generated_ids
    assert model.training
    assert completion.finish_reason in {"length", "stop"}
    episode = run_episode(
        backend, prompt, ToolRegistry(), generation=GenerationOptions(max_new_tokens=2)
    )
    assert episode.status in {"length", "complete"}
    assert len(episode.completions) == 1


def test_context_budget_refused_without_http(tokenizer):
    with endpoint(lambda req: pytest.fail("must not send")) as (url, requests):
        backend = OpenAIBackend(url, "model", tokenizer, context_window=16)
        with pytest.raises(RolloutError, match="context window"):
            backend.complete(
                [{"role": "user", "content": "hello"}],
                generation=GenerationOptions(max_new_tokens=16),
            )
        assert not requests


@pytest.mark.parametrize(
    "fault",
    ["invalid_json", "oversized", "redirect", "missing_usage", "wrong_prompt", "bad_finish"],
)
def test_http_protocol_failures(tokenizer, fault):
    def answer(request):
        if fault == "invalid_json":
            return 200, b"{bad", {}
        if fault == "oversized":
            return 200, b"x" * 2001, {}
        if fault == "redirect":
            return 302, b"", {"Location": "http://127.0.0.1:1/private"}
        status, body, headers = response(tokenizer, request, {"role": "assistant", "content": "Hi"})
        value = json.loads(body)
        if fault == "missing_usage":
            value.pop("usage")
        elif fault == "wrong_prompt":
            value["usage"]["prompt_tokens"] += 1
        else:
            value["choices"][0]["finish_reason"] = "content_filter"
        return status, json.dumps(value).encode(), headers

    with endpoint(answer) as (url, requests):
        backend = OpenAIBackend(
            url, "model", tokenizer, context_window=1024, max_response_bytes=2000
        )
        with pytest.raises(RolloutError):
            backend.complete(
                [{"role": "user", "content": "Hello"}],
                generation=GenerationOptions(max_new_tokens=8),
            )
        assert len(requests) == 1


def test_http_timeout(tokenizer):
    def answer(request):
        time.sleep(0.2)
        return response(tokenizer, request, {"role": "assistant", "content": "late"})

    with endpoint(answer) as (url, _):
        backend = OpenAIBackend(url, "model", tokenizer, context_window=1024, timeout=0.03)
        with pytest.raises(RolloutError, match="request failed"):
            backend.complete([{"role": "user", "content": "hello"}])


def test_whole_call_batch_validated_before_side_effects(tokenizer, tmp_path):
    marker = tmp_path / "executed"

    registry = ToolRegistry()
    registry.register(TOOL, partial(write_marker, str(marker)))
    invalid = {
        "id": "call2",
        "type": "function",
        "function": {"name": "unregistered", "arguments": "{}"},
    }

    def answer(request):
        return response(
            tokenizer,
            request,
            {"role": "assistant", "content": "", "tool_calls": [CALL, invalid]},
            "tool_calls",
        )

    with endpoint(answer) as (url, _):
        backend = OpenAIBackend(url, "model", tokenizer, context_window=4096)
        result = run_episode(backend, [{"role": "user", "content": "Do it"}], registry)
    assert result.status == "error"
    assert not marker.exists()


@pytest.mark.parametrize("handler,expected", [(sleep_long, "timed out"), (huge_output, "failed")])
def test_tool_workers_bound_time_and_output(handler, expected):
    registry = ToolRegistry()
    registry.register(
        {"type": "function", "function": {"name": "test", "parameters": {"type": "object"}}},
        handler,
    )
    call = {"id": "t", "type": "function", "function": {"name": "test", "arguments": "{}"}}
    start = time.monotonic()
    with pytest.raises(RolloutError, match=expected):
        registry.execute(call, timeout=0.1 if handler is sleep_long else 5, max_result_bytes=30)
    assert time.monotonic() - start < 6


def test_limits_do_not_dispatch_last_turn_calls(tokenizer, tmp_path):
    marker = tmp_path / "executed"

    registry = ToolRegistry()
    registry.register(TOOL, partial(write_marker, str(marker)))
    with endpoint(
        lambda req: response(
            tokenizer, req, {"role": "assistant", "content": "", "tool_calls": [CALL]}, "tool_calls"
        )
    ) as (url, _):
        backend = OpenAIBackend(url, "model", tokenizer, context_window=4096)
        result = run_episode(
            backend,
            [{"role": "user", "content": "do it"}],
            registry,
            limits=EpisodeLimits(max_turns=1),
        )
    assert result.status == "turn_limit"
    assert not marker.exists()


@pytest.mark.parametrize(
    "kwargs",
    [dict(max_new_tokens=0), dict(temperature=float("nan")), dict(top_p=0), dict(top_k=True)],
)
def test_generation_options_validate(kwargs):
    with pytest.raises(ValueError):
        GenerationOptions(**kwargs)


def test_failed_native_parse_retains_sampled_training_trace(tokenizer, monkeypatch):
    from clearforge import rollout

    model = CausalLM(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=16,
            num_heads=2,
            num_kv_heads=1,
            num_layers=1,
            intermediate_size=32,
            max_seq_len=128,
        )
    )
    backend = NativeBackend(model, tokenizer)

    def malformed(*args, **kwargs):
        raise ValueError("malformed structured assistant envelope")

    monkeypatch.setattr(rollout, "parse_assistant_output", malformed)
    with pytest.raises(ValueError, match="malformed"):
        backend.complete(
            [{"role": "user", "content": "Hi"}], generation=GenerationOptions(max_new_tokens=2)
        )
    assert backend.traces[-1]["prompt_ids"]
    assert backend.traces[-1]["generated_ids"]
    assert len(backend.traces[-1]["generated_ids"]) <= 2
    assert model.training


def test_unpickleable_handlers_rejected():
    with pytest.raises(ValueError, match="importable pickleable"):
        ToolRegistry().register(TOOL, lambda a, b: a + b)


def test_reused_call_id_rejected_before_second_execution(tokenizer):
    registry = ToolRegistry()
    registry.register(TOOL, addition)
    with endpoint(
        lambda req: response(
            tokenizer, req, {"role": "assistant", "content": "", "tool_calls": [CALL]}, "tool_calls"
        )
    ) as (url, requests):
        backend = OpenAIBackend(url, "model", tokenizer, context_window=4096)
        result = run_episode(backend, [{"role": "user", "content": "Add"}], registry)
    assert result.status == "error"
    assert len(result.tool_events) == 1
    assert len(requests) == 2


def test_remote_cannot_hide_large_text_behind_false_usage(tokenizer):
    with endpoint(
        lambda req: response(
            tokenizer, req, {"role": "assistant", "content": "x" * 200}, completion_tokens=1
        )
    ) as (url, _):
        backend = OpenAIBackend(url, "model", tokenizer, context_window=1024)
        with pytest.raises(RolloutError, match="understated"):
            backend.complete([{"role": "user", "content": "Hi"}])
