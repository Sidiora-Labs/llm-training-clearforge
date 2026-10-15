import dataclasses
import hashlib
import json
import threading

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from clearforge.config import ModelConfig
from clearforge.data import SourceSpec, prepare_sources
from clearforge.model import CausalLM
from clearforge.serving import ChatRequest, Generation, create_app, serve
from clearforge.tokenizer import train_tokenizer


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("serving")
    source = root / "source.jsonl"
    source.write_text(json.dumps({"text": "hello world local model"}) + "\n")
    prep = prepare_sources(
        [SourceSpec(source, "Apache-2.0")],
        root / "prep",
        validation_fraction=0,
        seed=1,
        mode="pretrain",
    )
    token = train_tokenizer(prep / "train.jsonl", root / "token", vocab_size=262)
    config = ModelConfig(
        vocab_size=262,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=32,
        max_seq_len=128,
    )
    model = CausalLM(config)
    path = root / "bundle"
    path.mkdir()
    torch.save(model.state_dict(), path / "model.pt")
    (path / "model.json").write_text(json.dumps(dataclasses.asdict(config)))
    (path / "tokenizer.json").write_bytes(token.read_bytes())
    (path / "provenance.json").write_text(
        json.dumps(
            {
                "format": "clearforge-native-v1",
                "files": {
                    name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                    for name in ["model.pt", "model.json", "tokenizer.json"]
                },
            }
        )
    )
    return path


@pytest.fixture
def app(bundle):
    return create_app({"tiny": bundle}, device="cpu", max_tokens=16)


def query(**changes):
    return {
        "model": "tiny",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.0,
        "max_tokens": 4,
        **changes,
    }


def events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def test_real_model_stream_matches_nonstream_and_usage(app):
    with TestClient(app) as client:
        normal = client.post("/v1/chat/completions", json=query())
        assert normal.status_code == 200, normal.text
        result = normal.json()
        streamed = client.post(
            "/v1/chat/completions", json=query(stream=True, stream_options={"include_usage": True})
        )
        assert streamed.status_code == 200
        parts = events(streamed)
        text = "".join(p["choices"][0]["delta"].get("content", "") for p in parts if p["choices"])
        assert text == result["choices"][0]["message"]["content"]
        assert parts[-1]["usage"] == result["usage"]
        assert result["usage"]["completion_tokens"] > 0
        assert result["usage"]["total_tokens"] == sum(
            result["usage"][k] for k in ["prompt_tokens", "completion_tokens"]
        )
        assert streamed.text.endswith("data: [DONE]\n\n")
        assert len({p["id"] for p in parts}) == 1
        assert app.state.last_generation.finished.wait(2)
        assert not app.state.generation_lock.locked()


@pytest.mark.parametrize(
    "change,status",
    [
        ({"model": "missing"}, 404),
        ({"max_tokens": 17}, 400),
        ({"messages": [{"role": "tool", "content": "x"}]}, 400),
        ({"messages": [{"role": "assistant", "content": "x"}]}, 400),
        ({"temperature": -1}, 400),
        ({"top_p": 0}, 400),
        ({"top_k": 1000}, 400),
        ({"n": 2}, 400),
        ({"unknown": True}, 400),
        ({"messages": [{"role": "user", "content": "x" * 130}]}, 400),
    ],
)
def test_request_errors_before_streaming(app, change, status):
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=query(stream=True, **change))
        assert response.status_code == status
        assert "error" in response.json()
        assert not app.state.generation_lock.locked()


def test_auth_body_limit_assets_and_busy(bundle):
    app = create_app({"tiny": bundle}, device="cpu", api_key="test-key", max_body_bytes=300)
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401
        headers = {"Authorization": "Bearer test-key"}
        models = client.get("/v1/models", headers=headers).json()["data"]
        assert models[0]["id"] == "tiny" and models[0]["context_length"] == 128
        assert (
            client.post("/v1/chat/completions", headers=headers, content="x" * 301).status_code
            == 413
        )
        assert (
            client.post("/v1/chat/completions", headers=headers, content="broken json").status_code
            == 400
        )
        assert client.get("/").status_code == 200
        for name in ["style.css", "app.js"]:
            assert client.get("/assets/" + name).status_code == 200
        assert client.get("/assets/nope").status_code == 404
        app.state.generation_lock.acquire()
        try:
            assert (
                client.post("/v1/chat/completions", headers=headers, json=query()).status_code
                == 429
            )
        finally:
            app.state.generation_lock.release()


def test_real_model_nonfinite_stream_error_releases_lock(app):
    model, _ = app.state.models["tiny"]
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=query(stream=True))
        assert response.status_code == 200
        assert any("error" in part for part in events(response))
        assert response.text.endswith("data: [DONE]\n\n")
        assert app.state.last_generation.finished.wait(2)
        assert not app.state.generation_lock.locked()
        response = client.post("/v1/chat/completions", json=query())
        assert response.status_code == 500


def test_cancel_real_generation_and_bounded_queue(bundle):
    app = create_app({"tiny": bundle}, device="cpu")
    model, tok = app.state.models["tiny"]
    ids, _ = tok.encode_chat([{"role": "user", "content": "x"}], add_generation_prompt=True)
    lock = threading.Lock()
    lock.acquire()
    work = Generation(model, tok, ids, ChatRequest(**query(max_tokens=100)), lock, 10)
    work.cancel.set()
    assert work.finished.wait(3)
    assert not lock.locked()
    assert work.queue.maxsize == 32
    assert not model.training


def test_generation_timeout(app):
    model, tok = app.state.models["tiny"]
    ids, _ = tok.encode_chat([{"role": "user", "content": "x"}], add_generation_prompt=True)
    lock = threading.Lock()
    lock.acquire()
    work = Generation(model, tok, ids, ChatRequest(**query()), lock, 0.000001)
    assert work.finished.wait(3)
    assert work.cancel.is_set()
    assert not lock.locked()


def test_public_listener_requires_auth(bundle):
    with pytest.raises(ValueError, match="API key"):
        serve({"tiny": bundle}, host="0.0.0.0")


def test_live_http_disconnect_stops_worker(app):
    import socket
    import time

    import httpx
    import uvicorn

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=3) as client:
            with client.stream(
                "POST", "/v1/chat/completions", json=query(stream=True, max_tokens=16)
            ) as response:
                assert response.status_code == 200
                assert next(response.iter_lines()).startswith("data: ")
            generation = app.state.last_generation
            assert generation.finished.wait(3)
            deadline = time.monotonic() + 3
            while not generation.cancel.is_set() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert generation.cancel.is_set()
            assert not app.state.generation_lock.locked()
    finally:
        server.should_exit = True
        worker.join(5)
        sock.close()
        assert not worker.is_alive()


def test_seed_reproducibility_and_rng_isolation(app):
    with TestClient(app) as client:
        before = torch.get_rng_state().clone()
        a = client.post(
            "/v1/chat/completions",
            json=query(temperature=0.8, seed=53, top_p=0.8, repetition_penalty=1.2),
        ).json()
        b = client.post(
            "/v1/chat/completions",
            json=query(temperature=0.8, seed=53, top_p=0.8, repetition_penalty=1.2),
        ).json()
        assert a["choices"] == b["choices"]
        assert torch.equal(before, torch.get_rng_state())


def tool_fixture():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        }
    ]
    call = {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    return tools, call


def test_real_model_accepts_schema_and_tool_result_history(app):
    model, tok = app.state.models["tiny"]
    # Same real native weights, a larger supported context for the JSON schema.
    model.config = dataclasses.replace(model.config, max_seq_len=1024)
    tools, call = tool_fixture()
    messages = [
        {"role": "user", "content": "Look up the value."},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "5"},
    ]
    expected, _ = tok.encode_chat(messages, tools=tools, add_generation_prompt=True)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=query(messages=messages, tools=tools))
        assert response.status_code == 200, response.text
        assert response.json()["usage"]["prompt_tokens"] == len(expected)
        messages[-1]["tool_call_id"] = "orphan"
        bad = client.post(
            "/v1/chat/completions", json=query(messages=messages, tools=tools, stream=True)
        )
        assert bad.status_code == 400


def test_tool_response_stream_protocol_and_nonstream(app, monkeypatch):
    from clearforge.tools import serialize_messages

    model, tok = app.state.models["tiny"]
    model.config = dataclasses.replace(model.config, max_seq_len=1024)
    tools, call = tool_fixture()
    messages = [
        {"role": "user", "content": "Look up the value."},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "A lookup is needed.",
            "tool_calls": [call],
        },
    ]
    response_text = serialize_messages(messages, tools=tools)[-1][1]

    # Fixed token fixture isolates protocol framing; real-model generation is tested separately.
    def fixed_tokens(*args, **kwargs):
        for token in tok.encode(response_text) + [tok.eos_id]:
            yield torch.tensor([[token]])

    monkeypatch.setattr(model, "stream_generate", fixed_tokens)
    with TestClient(app) as client:
        normal = client.post("/v1/chat/completions", json=query(tools=tools)).json()
        assert normal["choices"][0]["finish_reason"] == "tool_calls"
        assert normal["choices"][0]["message"]["tool_calls"] == [call]
        assert normal["choices"][0]["message"]["reasoning_content"] == "A lookup is needed."
        streamed = client.post("/v1/chat/completions", json=query(tools=tools, stream=True))
        parts = events(streamed)
        deltas = [p["choices"][0]["delta"] for p in parts]
        assert any(d.get("tool_calls") == [{"index": 0, **call}] for d in deltas)
        assert all("<clearforge_assistant>" not in d.get("content", "") for d in deltas)
        assert parts[-1]["choices"][0]["finish_reason"] == "tool_calls"
        forbidden = client.post("/v1/chat/completions", json=query(tools=tools, tool_choice="none"))
        assert forbidden.status_code == 500


def test_malformed_generated_tool_arguments_fail_stream(app, monkeypatch):
    from clearforge.tools import ASSISTANT_CLOSE, ASSISTANT_OPEN

    model, tok = app.state.models["tiny"]
    model.config = dataclasses.replace(model.config, max_seq_len=1024)
    tools, call = tool_fixture()
    call["function"]["name"] = "not_declared"
    text = ASSISTANT_OPEN + json.dumps({"content": "", "tool_calls": [call]}) + ASSISTANT_CLOSE

    def fixed_tokens(*args, **kwargs):
        for token in tok.encode(text) + [tok.eos_id]:
            yield torch.tensor([[token]])

    monkeypatch.setattr(model, "stream_generate", fixed_tokens)
    with TestClient(app) as client:
        result = client.post("/v1/chat/completions", json=query(tools=tools, stream=True))
        assert any("error" in part for part in events(result))
        assert not any(
            p.get("choices", [{}])[0].get("delta", {}).get("tool_calls")
            for p in events(result)
            if "choices" in p
        )


def test_generated_call_id_cannot_reuse_history(app, monkeypatch):
    from clearforge.tools import serialize_messages

    model, tok = app.state.models["tiny"]
    model.config = dataclasses.replace(model.config, max_seq_len=1024)
    tools, call = tool_fixture()
    messages = [
        {"role": "user", "content": "Look up"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": call["id"], "content": "5"},
    ]
    reply = serialize_messages([messages[0], messages[1]], tools=tools)[-1][1]

    def fixed_tokens(*args, **kwargs):
        for token in tok.encode(reply) + [tok.eos_id]:
            yield torch.tensor([[token]])

    monkeypatch.setattr(model, "stream_generate", fixed_tokens)
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions", json=query(messages=messages, tools=tools, stream=True)
        )
        assert any("error" in event for event in events(response))
        assert app.state.last_generation.finished.wait(2)
        assert not app.state.generation_lock.locked()
