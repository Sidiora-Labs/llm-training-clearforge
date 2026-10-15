"""Bounded native and HTTP generation with explicitly registered tool episodes."""

from __future__ import annotations

import copy
import json
import math
import multiprocessing
import pickle
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .tools import (
    canonical_json,
    parse_assistant_output,
    validate_conversation,
    validate_tool_calls,
    validate_tools,
)


class RolloutError(RuntimeError):
    """A backend, tool, or episode violated its execution contract."""


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    repetition_penalty: float = 1.0

    def __post_init__(self):
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        for name in ("temperature", "top_p", "repetition_penalty"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0 or not 0 < self.top_p <= 1 or self.repetition_penalty <= 0:
            raise ValueError("invalid sampling controls")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer")


@dataclass(frozen=True)
class Completion:
    message: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    prompt_ids: list[int] | None = None
    generated_ids: list[int] | None = None


class Backend(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        generation: GenerationOptions,
        deadline: float | None = None,
    ) -> Completion: ...


def _remaining(deadline: float | None, maximum: float) -> float:
    remaining = maximum if deadline is None else min(maximum, deadline - time.monotonic())
    if remaining <= 0:
        raise RolloutError("episode deadline exceeded")
    return remaining


def _prompt(tokenizer, messages, tools, context_window, generation):
    clean = validate_conversation(messages, tools=tools)
    ids, _ = tokenizer.encode_chat(clean, tools=tools, add_generation_prompt=True)
    if len(ids) + generation.max_new_tokens > context_window:
        raise RolloutError("prompt and completion budget exceed context window")
    return clean, ids


class NativeBackend:
    """Cached generation using the native tokenizer and structured assistant parser."""

    def __init__(self, model, tokenizer):
        if model.config.vocab_size != tokenizer.vocab_size:
            raise ValueError("model and tokenizer vocabularies differ")
        self.model, self.tokenizer = model, tokenizer
        self.lock = threading.Lock()
        self.last_trace: dict[str, list[int]] | None = None
        self.traces: list[dict[str, list[int]]] = []

    def complete(self, messages, *, tools=None, generation=None, deadline=None) -> Completion:
        import torch

        generation = generation or GenerationOptions()
        tools = validate_tools(tools or [])
        clean, ids = _prompt(
            self.tokenizer, messages, tools, self.model.config.max_seq_len, generation
        )
        if not self.lock.acquire(timeout=_remaining(deadline, 3600)):
            raise RolloutError("native generation lock deadline exceeded")
        emitted = []
        self.last_trace = {"prompt_ids": list(ids), "generated_ids": emitted}
        self.traces.append(self.last_trace)
        del self.traces[:-128]
        try:
            with closing(
                self.model.stream_generate(
                    torch.tensor(
                        [ids], dtype=torch.long, device=self.model.token_embedding.weight.device
                    ),
                    max_new_tokens=generation.max_new_tokens,
                    temperature=generation.temperature,
                    top_p=generation.top_p,
                    top_k=generation.top_k,
                    repetition_penalty=generation.repetition_penalty,
                    eos_token_id=self.tokenizer.eos_id,
                )
            ) as stream:
                for token in stream:
                    emitted.append(token.item())
                    _remaining(deadline, 3600)
            text = self.tokenizer.decode(emitted, skip_special_tokens=True)
            message = parse_assistant_output(text, tools=tools)
            message = validate_conversation([*clean, message], tools=tools)[-1]
            stopped = bool(emitted and emitted[-1] == self.tokenizer.eos_id)
            reason = (
                "tool_calls"
                if message.get("tool_calls") and stopped
                else "stop"
                if stopped
                else "length"
            )
            return Completion(message, len(ids), len(emitted), reason, list(ids), list(emitted))
        finally:
            self.last_trace = {"prompt_ids": list(ids), "generated_ids": list(emitted)}
            self.lock.release()


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _json_object(raw: bytes):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("duplicate JSON key")
            result[name] = value
        return result

    def constant(value):
        raise ValueError("nonfinite JSON value")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise RolloutError("backend returned invalid JSON") from error


class OpenAIBackend:
    """Non-streaming OpenAI chat completions with a matching local tokenizer.

    ``base_url`` includes the API prefix, usually ``http://host:port/v1``.
    Redirects and environment proxies are disabled. SGLang uses this wire format.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        tokenizer,
        *,
        context_window: int,
        api_key: str | None = None,
        timeout: float = 30,
        max_response_bytes: int = 1_048_576,
    ):
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("base_url must be an HTTP(S) API prefix without credentials or query")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("remote model name must be nonempty")
        if type(context_window) is not int or context_window < 2:
            raise ValueError("context_window must be at least two")
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive")
        if type(max_response_bytes) is not int or max_response_bytes < 1:
            raise ValueError("max_response_bytes must be positive")
        if api_key is not None and (
            not isinstance(api_key, str) or "\n" in api_key or "\r" in api_key
        ):
            raise ValueError("invalid API key")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.tokenizer, self.context_window = model, tokenizer, context_window
        self.api_key, self.timeout, self.max_response_bytes = api_key, timeout, max_response_bytes
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def complete(self, messages, *, tools=None, generation=None, deadline=None) -> Completion:
        generation = generation or GenerationOptions()
        tools = validate_tools(tools or [])
        clean, ids = _prompt(self.tokenizer, messages, tools, self.context_window, generation)
        payload = {
            "model": self.model,
            "messages": clean,
            "max_tokens": generation.max_new_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if generation.top_k is not None:
            payload["top_k"] = generation.top_k
        if generation.repetition_penalty != 1:
            payload["repetition_penalty"] = generation.repetition_penalty
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url, canonical_json(payload).encode(), headers, method="POST"
        )
        request_deadline = min(
            deadline if deadline is not None else math.inf, time.monotonic() + self.timeout
        )
        try:
            with self.opener.open(
                request, timeout=_remaining(request_deadline, self.timeout)
            ) as response:
                pieces, size = [], 0
                while True:
                    remaining = _remaining(request_deadline, self.timeout)
                    # read1 performs at most one underlying read, so a slow stream
                    # cannot indefinitely refresh the socket's per-read timeout.
                    if response.fp is not None:
                        response.fp.raw._sock.settimeout(remaining)
                    piece = response.read1(min(65536, self.max_response_bytes + 1 - size))
                    if not piece:
                        break
                    pieces.append(piece)
                    size += len(piece)
                    if size > self.max_response_bytes:
                        raise RolloutError("backend response exceeds byte budget")
                raw = b"".join(pieces)
        except urllib.error.HTTPError as error:
            raise RolloutError(f"backend returned HTTP status {error.code}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RolloutError("backend request failed") from error
        _remaining(deadline, self.timeout)
        document = _json_object(raw)
        if not isinstance(document, dict):
            raise RolloutError("backend response must be an object")
        choices = document.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise RolloutError("backend must return exactly one completion")
        choice = choices[0]
        reason = choice.get("finish_reason")
        if reason not in {"stop", "length", "tool_calls"}:
            raise RolloutError("backend returned unsupported finish reason")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise RolloutError("backend completion must be an assistant message")
        try:
            message = validate_conversation([*clean, message], tools=tools)[-1]
        except ValueError as error:
            raise RolloutError("backend returned an invalid assistant message") from error
        if reason == "tool_calls" and not message.get("tool_calls"):
            raise RolloutError("tool_calls finish reason lacks calls")
        usage = document.get("usage")
        if not isinstance(usage, dict) or any(
            type(usage.get(name)) is not int or usage[name] < 0
            for name in ("prompt_tokens", "completion_tokens")
        ):
            raise RolloutError("backend must report nonnegative integer token usage")
        prompt_tokens, completion_tokens = usage["prompt_tokens"], usage["completion_tokens"]
        if (
            completion_tokens > generation.max_new_tokens
            or prompt_tokens + completion_tokens > self.context_window
        ):
            raise RolloutError("backend exceeded token budget")
        # Usage is not allowed to understate even the visible text token count.
        # Tool-call transport can encode metadata differently, so its full token
        # accounting remains the server's explicit usage contract.
        if len(self.tokenizer.encode(message["content"])) > completion_tokens:
            raise RolloutError("backend understated completion token usage")
        if prompt_tokens != len(ids):
            raise RolloutError("remote tokenizer or chat template differs from local tokenizer")
        return Completion(message, prompt_tokens, completion_tokens, reason)


class SGLangBackend(OpenAIBackend):
    """SGLang's OpenAI-compatible /v1/chat/completions endpoint."""


@dataclass(frozen=True)
class EpisodeLimits:
    max_turns: int = 8
    max_tool_calls: int = 16
    max_generated_tokens: int = 1024
    max_output_bytes: int = 262_144
    max_tool_result_bytes: int = 16_384
    max_seconds: float = 120
    tool_timeout: float = 5

    def __post_init__(self):
        for name in (
            "max_turns",
            "max_tool_calls",
            "max_generated_tokens",
            "max_output_bytes",
            "max_tool_result_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_seconds", "tool_timeout"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")


def _tool_process(pipe, function, arguments, maximum):
    try:
        result = function(**arguments)
        text = result if isinstance(result, str) else canonical_json(result)
        if len(text.encode()) > maximum:
            raise RolloutError("tool result exceeds byte budget")
        pipe.send_bytes(canonical_json({"ok": True, "text": text}).encode())
    except BaseException as error:
        pipe.send_bytes(canonical_json({"ok": False, "error": type(error).__name__}).encode())
    finally:
        pipe.close()


class ToolRegistry:
    """Explicit trusted Python handlers; every invocation runs in a bounded child.

    Uses spawned workers. Handlers must be importable, pickleable callables and
    must not depend on mutating parent Python objects. External side effects
    are the registrant's responsibility.
    """

    def __init__(self):
        self._tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}

    def register(self, definition: dict[str, Any], handler: Callable[..., Any]) -> None:
        clean = validate_tools([definition])[0]
        name = clean["function"]["name"]
        if name in self._tools or not callable(handler):
            raise ValueError("tool name must be unique and handler callable")
        try:
            pickle.dumps(handler)
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            raise ValueError("tool handlers must be importable pickleable callables") from error
        self._tools[name] = (clean, handler)

    @property
    def definitions(self):
        return copy.deepcopy([value[0] for value in self._tools.values()])

    def execute(self, call: dict[str, Any], *, timeout: float, max_result_bytes: int) -> str:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("tool timeout must be finite and positive")
        if type(max_result_bytes) is not int or max_result_bytes < 1:
            raise ValueError("tool result byte budget must be positive")
        call = validate_tool_calls([call], tools=self.definitions)[0]
        handler = self._tools[call["function"]["name"]][1]
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=False)
        worker = context.Process(
            target=_tool_process,
            args=(child, handler, json.loads(call["function"]["arguments"]), max_result_bytes),
        )
        try:
            worker.start()
            child.close()
            if not parent.poll(timeout):
                raise RolloutError("tool execution timed out")
            try:
                result = _json_object(parent.recv_bytes(maxlength=max_result_bytes * 6 + 1024))
            except (EOFError, OSError) as error:
                raise RolloutError("tool worker failed") from error
            if not result.get("ok"):
                raise RolloutError(f"tool execution failed: {result.get('error', 'unknown')}")
            return result["text"]
        finally:
            if worker.pid is not None:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=1)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
            parent.close()
            child.close()


@dataclass
class EpisodeResult:
    messages: list[dict[str, Any]]
    status: str
    turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    completions: list[Completion] = field(default_factory=list)


def run_episode(
    backend: Backend,
    messages: list[dict[str, Any]],
    registry: ToolRegistry,
    *,
    generation: GenerationOptions | None = None,
    limits: EpisodeLimits | None = None,
) -> EpisodeResult:
    """Run bounded assistant/tool turns; return observations for external rewards.

    A malformed call batch is rejected before dispatching any call in that batch.
    No reward heuristic or implicit executable tool is supplied.
    """
    from dataclasses import replace

    generation, limits = generation or GenerationOptions(), limits or EpisodeLimits()
    definitions = registry.definitions
    clean = validate_conversation(messages, tools=definitions)
    result = EpisodeResult(copy.deepcopy(clean), "turn_limit")
    deadline, used_bytes, calls = time.monotonic() + limits.max_seconds, 0, 0
    for turn in range(limits.max_turns):
        remaining_tokens = limits.max_generated_tokens - result.completion_tokens
        if remaining_tokens <= 0:
            result.status = "token_limit"
            break
        try:
            _remaining(deadline, limits.max_seconds)
            completion = backend.complete(
                result.messages,
                tools=definitions,
                generation=replace(
                    generation, max_new_tokens=min(generation.max_new_tokens, remaining_tokens)
                ),
                deadline=deadline,
            )
            message = validate_conversation(
                [*result.messages, completion.message], tools=definitions
            )[-1]
            if (
                type(completion.completion_tokens) is not int
                or not 0
                <= completion.completion_tokens
                <= min(generation.max_new_tokens, remaining_tokens)
                or type(completion.prompt_tokens) is not int
                or completion.prompt_tokens < 0
            ):
                raise RolloutError("completion violated token accounting")
            if completion.finish_reason not in {"stop", "length", "tool_calls"}:
                raise RolloutError("completion has unsupported finish reason")
            message_bytes = len(canonical_json(message).encode())
            if used_bytes + message_bytes > limits.max_output_bytes:
                result.status = "output_limit"
                break
            used_bytes += message_bytes
            result.messages.append(message)
            result.completions.append(completion)
            result.turns = turn + 1
            result.prompt_tokens += completion.prompt_tokens
            result.completion_tokens += completion.completion_tokens
            if completion.finish_reason == "length":
                result.status = "length"
                break
            batch = message.get("tool_calls", [])
            if not batch:
                result.status = "complete"
                break
            batch = validate_tool_calls(batch, tools=definitions)
            if calls + len(batch) > limits.max_tool_calls:
                result.status = "tool_limit"
                break
            if turn + 1 == limits.max_turns:
                result.status = "turn_limit"
                break
            observations = []
            for call in batch:
                calls += 1
                text = registry.execute(
                    call,
                    timeout=_remaining(deadline, limits.tool_timeout),
                    max_result_bytes=min(
                        limits.max_tool_result_bytes, limits.max_output_bytes - used_bytes
                    ),
                )
                observation = {"role": "tool", "tool_call_id": call["id"], "content": text}
                size = len(canonical_json(observation).encode())
                if used_bytes + size > limits.max_output_bytes:
                    result.status = "output_limit"
                    return result
                used_bytes += size
                observations.append(observation)
                result.tool_events.append({"call": call, "result": text})
            result.messages.extend(observations)
        except (RolloutError, ValueError) as error:
            result.status = "error"
            result.error = str(error)
            break
    return result
