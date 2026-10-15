"""Local native-model chat service with bounded, cancellable generation."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import queue
import secrets
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import ClientDisconnect

from .cli import _device, _load_bundle

logger = logging.getLogger(__name__)


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=262144)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=200)
    messages: list[Message] = Field(min_length=1, max_length=256)
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    repetition_penalty: float = Field(default=1.0, gt=0, le=10)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: Literal[1] = 1


def _error(message: str, status: int = 400, code: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": code, "code": code}}, status_code=status
    )


def _tokens(model: Any, ids: torch.Tensor, options: ChatRequest, eos: int) -> Iterator[int]:
    """Consume cached single-sequence generation and always close its iterator."""
    iterator = model.stream_generate(
        ids,
        max_new_tokens=options.max_tokens,
        temperature=options.temperature,
        top_k=options.top_k,
        eos_token_id=eos,
        top_p=options.top_p,
        repetition_penalty=options.repetition_penalty,
    )
    try:
        for token in iterator:
            yield int(token[0, 0])
    finally:
        iterator.close()


class Generation:
    """One worker owns the model lock until its iterator is closed, even on disconnect."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        ids: list[int],
        options: ChatRequest,
        lock: threading.Lock,
        timeout: float,
    ) -> None:
        self.cancel = threading.Event()
        self.finished = threading.Event()
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=32)
        self.started = time.monotonic()
        self.timeout = timeout
        self.thread = threading.Thread(
            target=self._run,
            args=(model, tokenizer, ids, options, lock),
            daemon=True,
            name="clearforge-generation",
        )
        self.thread.start()

    def _put(self, value: dict[str, Any]) -> None:
        while not self.cancel.is_set():
            if time.monotonic() - self.started > self.timeout:
                self.cancel.set()
                return
            try:
                self.queue.put(value, timeout=0.05)
                return
            except queue.Full:
                continue

    def _run(
        self, model: Any, tokenizer: Any, ids: list[int], options: ChatRequest, lock: threading.Lock
    ) -> None:
        iterator = None
        try:
            device = next(model.parameters()).device
            devices = [device.index or 0] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                seed = options.seed if options.seed is not None else secrets.randbits(63)
                torch.random.default_generator.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.default_generators[device.index or 0].manual_seed(seed)
                iterator = _tokens(
                    model,
                    torch.tensor([ids], dtype=torch.long, device=device),
                    options,
                    tokenizer.eos_id,
                )
                generated: list[int] = []
                emitted = ""
                finish = "length"
                for token in iterator:
                    if self.cancel.is_set():
                        break
                    generated.append(token)
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    # Byte-level tokens may end in an incomplete UTF-8 character.
                    stable = text.rstrip("\ufffd")
                    if stable.startswith(emitted) and len(stable) > len(emitted):
                        self._put({"text": stable[len(emitted) :]})
                        emitted = stable
                    if token == tokenizer.eos_id:
                        finish = "stop"
                        break
                if not self.cancel.is_set():
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    if text.startswith(emitted) and len(text) > len(emitted):
                        self._put({"text": text[len(emitted) :]})
                    self._put(
                        {
                            "finish_reason": finish,
                            "usage": {
                                "prompt_tokens": len(ids),
                                "completion_tokens": len(generated),
                                "total_tokens": len(ids) + len(generated),
                            },
                        }
                    )
        except Exception:
            logger.exception("Model generation failed")
            self._put(
                {
                    "error": {
                        "message": "Model generation failed",
                        "type": "server_error",
                        "code": "generation_failed",
                    }
                }
            )
        finally:
            try:
                if iterator is not None:
                    iterator.close()
            finally:
                lock.release()
                self.finished.set()

    async def events(self, request: Request) -> Any:
        try:
            while True:
                if await request.is_disconnected():
                    return
                if time.monotonic() - self.started > self.timeout:
                    yield {
                        "error": {
                            "message": "Generation timed out",
                            "type": "server_error",
                            "code": "timeout",
                        }
                    }
                    return
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    if self.finished.is_set():
                        return
                    await asyncio.sleep(0.01)
                    continue
                yield item
                if "error" in item or "finish_reason" in item:
                    return
        finally:
            self.cancel.set()


def create_app(
    model_paths: Mapping[str, str | Path],
    *,
    device: str = "auto",
    api_key: str | None = None,
    max_tokens: int = 4096,
    max_body_bytes: int = 1048576,
    generation_timeout: float = 120.0,
) -> FastAPI:
    """Load verified exports once. All models share a generation/RNG lock per process."""
    if not model_paths or any(not isinstance(k, str) or not k or len(k) > 200 for k in model_paths):
        raise ValueError("provide at least one named model")
    if (
        type(max_tokens) is not int
        or max_tokens < 1
        or type(max_body_bytes) is not int
        or max_body_bytes < 1
        or isinstance(generation_timeout, bool)
        or not isinstance(generation_timeout, (int, float))
        or not math.isfinite(generation_timeout)
        or generation_timeout <= 0
    ):
        raise ValueError("server limits must be positive")
    if api_key is not None and not api_key:
        raise ValueError("API key cannot be empty")
    target = _device(device)
    models = {name: _load_bundle(Path(path), target) for name, path in model_paths.items()}
    lock = threading.Lock()
    app = FastAPI(title="Clearforge", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.models = models
    app.state.generation_lock = lock
    app.state.last_generation = None
    static = Path(__file__).with_name("web")

    def authorized(request: Request) -> bool:
        return api_key is None or secrets.compare_digest(
            request.headers.get("authorization", "").encode("utf-8"),
            ("Bearer " + api_key).encode("utf-8"),
        )

    @app.middleware("http")
    async def headers(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith("/v1/") and not authorized(request):
            return _error("Invalid or missing API key", 401, "authentication_error")
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; connect-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static / "index.html")

    @app.get("/assets/{name}")
    async def asset(name: str) -> Any:
        if name not in {"app.js", "style.css"}:
            return _error("Asset not found", 404, "not_found")
        return FileResponse(static / name)

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "local",
                    "context_length": model.config.max_seq_len,
                }
                for name, (model, _) in models.items()
            ],
        }

    @app.post("/v1/chat/completions")
    async def completion(request: Request) -> Any:
        try:
            body = bytearray()
            async for part in request.stream():
                if len(body) + len(part) > max_body_bytes:
                    return _error("Request body exceeds server limit", 413, "request_too_large")
                body.extend(part)
            options = ChatRequest.model_validate_json(body)
        except (ValidationError, ValueError):
            return _error("Invalid chat request: check fields, types, and generation limits")
        except ClientDisconnect:
            return _error("Client disconnected", 400)
        if options.model not in models:
            return _error("Unknown model", 404, "model_not_found")
        model, tokenizer = models[options.model]
        if options.max_tokens > max_tokens:
            return _error(f"max_tokens exceeds server limit of {max_tokens}")
        if options.top_k is not None and options.top_k > tokenizer.vocab_size:
            return _error("top_k exceeds model vocabulary")
        try:
            ids, _ = tokenizer.encode_chat(
                [m.model_dump() for m in options.messages], add_generation_prompt=True
            )
        except ValueError as exc:
            return _error(str(exc))
        if len(ids) + options.max_tokens > model.config.max_seq_len:
            return _error(
                "Prompt plus max_tokens exceeds model context; shorten history or response budget",
                code="context_length_exceeded",
            )
        if not lock.acquire(blocking=False):
            return _error(
                "Model is busy; retry after the current generation finishes", 429, "model_busy"
            )
        try:
            generation = Generation(model, tokenizer, ids, options, lock, generation_timeout)
        except Exception:
            lock.release()
            raise
        app.state.last_generation = generation
        identity = {
            "id": "chatcmpl-" + secrets.token_hex(12),
            "created": int(time.time()),
            "model": options.model,
        }
        if not options.stream:
            text = ""
            async for event in generation.events(request):
                if "error" in event:
                    return JSONResponse(event, status_code=500)
                text += event.get("text", "")
                if "finish_reason" in event:
                    return {
                        **identity,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": event["finish_reason"],
                            }
                        ],
                        "usage": event["usage"],
                    }
            return _error("Generation cancelled", 499, "cancelled")

        async def stream() -> Any:
            def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
                return (
                    "data: "
                    + json.dumps(
                        {
                            **identity,
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                        }
                    )
                    + "\n\n"
                )

            try:
                yield chunk({"role": "assistant", "content": ""})
                async for event in generation.events(request):
                    if "error" in event:
                        yield "data: " + json.dumps(event) + "\n\n"
                        break
                    if "text" in event:
                        yield chunk({"content": event["text"]})
                    if "finish_reason" in event:
                        yield chunk({}, event["finish_reason"])
                        if options.stream_options and options.stream_options.include_usage:
                            yield (
                                "data: "
                                + json.dumps(
                                    {
                                        **identity,
                                        "object": "chat.completion.chunk",
                                        "choices": [],
                                        "usage": event["usage"],
                                    }
                                )
                                + "\n\n"
                            )
                yield "data: [DONE]\n\n"
            finally:
                generation.cancel.set()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def serve(
    model_paths: Mapping[str, str | Path],
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "auto",
    api_key: str | None = None,
    **limits: Any,
) -> None:
    """Run one process; non-loopback listeners require an explicit API key."""
    import uvicorn

    if host not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise ValueError("Non-loopback serving requires an API key")
    uvicorn.run(
        create_app(model_paths, device=device, api_key=api_key, **limits), host=host, port=port
    )
