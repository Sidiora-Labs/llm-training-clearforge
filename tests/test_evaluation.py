"""Evaluation contracts: scored denominators, structured calls and failure propagation."""

import json
from types import SimpleNamespace

import pytest
import torch

from clearforge.evaluation import evaluate_generations, tokenizer_diagnostics


class Backend:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        torch.rand(3)
        return SimpleNamespace(
            message=next(self.messages), prompt_tokens=3, completion_tokens=2, finish_reason="stop"
        )


def test_scores_explicit_checks_excludes_unscored_and_restores_rng(tmp_path):
    prompt = [{"role": "user", "content": "hello"}]
    cases = [
        {"id": "sample", "messages": prompt},
        {"id": "exact", "messages": prompt, "expected": "yes", "contains": ["ye", "s"]},
        {"id": "json", "messages": prompt, "expected_json": {"n": 2}},
        {"id": "wrong", "messages": prompt, "expected_json": {}},
    ]
    backend = Backend([{"content": x} for x in ("free text", "yes", '{"n":2}', "not json")])
    torch.manual_seed(19)
    original = torch.get_rng_state()
    output = tmp_path / "report.json"
    report = evaluate_generations(backend, cases, output=output)
    assert report["scored"] == 3 and report["passed"] == 2
    assert report["pass_rate"] == pytest.approx(2 / 3)
    assert report["results"][0]["passed"] is None
    assert torch.equal(original, torch.get_rng_state())
    assert json.loads(output.read_text()) == report


def test_tool_semantics_ignore_correlation_ids():
    def call(identifier, arguments):
        return {
            "id": identifier,
            "type": "function",
            "function": {"name": "lookup", "arguments": arguments},
        }

    backend = Backend([{"tool_calls": [call("actual", '{"city":"Paris"}')]}])
    report = evaluate_generations(
        backend,
        [
            {
                "messages": [{"role": "user", "content": "weather"}],
                "expected_tool_calls": [call("expected", {"city": "Paris"})],
            }
        ],
    )
    assert report["passed"] == 1


def test_invalid_suite_fails_before_generation_and_transport_errors_propagate():
    backend = Backend([])
    with pytest.raises(ValueError):
        evaluate_generations(
            backend,
            [{"messages": [{"role": "user", "content": "ok"}]}, {"messages": [], "contains": 3}],
        )
    assert backend.calls == 0
    with pytest.raises(StopIteration):
        evaluate_generations(backend, [{"messages": [{"role": "user", "content": "ok"}]}])


def test_tokenizer_diagnostics_rejects_nontext_and_empty(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"messages":[]}\n')
    with pytest.raises(ValueError, match="text"):
        tokenizer_diagnostics(None, path)
    path.write_text("")
    with pytest.raises(ValueError, match="tokens"):
        tokenizer_diagnostics(None, path)


def test_json_boolean_is_not_number_and_empty_tool_expectation():
    prompt = [{"role": "user", "content": "answer"}]
    backend = Backend([{"content": '{"flag":1}'}, {"content": "no tool needed"}])
    report = evaluate_generations(
        backend,
        [
            {"messages": prompt, "expected_json": {"flag": True}},
            {"messages": prompt, "expected_tool_calls": []},
        ],
    )
    assert report["results"][0]["passed"] is False
    assert report["results"][1]["passed"] is True


def test_expected_calls_validate_arguments_before_generation():
    backend = Backend([])
    case = {
        "messages": [{"role": "user", "content": "call"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                    },
                },
            }
        ],
        "expected_tool_calls": [
            {
                "id": "a",
                "type": "function",
                "function": {"name": "lookup", "arguments": {"n": True}},
            }
        ],
    }
    with pytest.raises(ValueError):
        evaluate_generations(backend, [case])
    assert backend.calls == 0


def test_all_cases_require_generation_ready_histories_before_running():
    backend = Backend([])
    with pytest.raises(ValueError, match="generation prompt"):
        evaluate_generations(
            backend,
            [
                {"messages": [{"role": "user", "content": "first"}]},
                {
                    "messages": [
                        {"role": "user", "content": "second"},
                        {"role": "assistant", "content": "already answered"},
                    ]
                },
            ],
        )
    assert backend.calls == 0
