"""Deterministic generation suites and tokenizer corpus diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import atomic_write_bytes


def tokenizer_diagnostics(tokenizer, path: Path) -> dict[str, Any]:
    """Measure every JSONL text record without retaining the corpus in memory."""
    records = characters = byte_count = tokens = roundtrip_failures = 0
    longest = 0
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get("text") if isinstance(row, dict) else None
            if not isinstance(text, str):
                raise ValueError(f"line {line_number}: expected a text string")
            encoded = tokenizer.encode(text)
            records += 1
            characters += len(text)
            byte_count += len(text.encode("utf-8"))
            tokens += len(encoded)
            longest = max(longest, len(encoded))
            roundtrip_failures += tokenizer.decode(encoded) != text
    if not records or not tokens:
        raise ValueError("diagnostic corpus must contain tokens")
    return {
        "records": records,
        "characters": characters,
        "bytes": byte_count,
        "tokens": tokens,
        "characters_per_token": characters / tokens,
        "bytes_per_token": byte_count / tokens,
        "max_record_tokens": longest,
        "roundtrip_failures": roundtrip_failures,
        "vocab_size": tokenizer.vocab_size,
    }


def evaluate_generations(backend, cases, *, generation=None, seed=42, output=None):
    """Run explicit exact, substring, JSON and tool-call checks on generated replies.

    Unscored prompts are retained as samples and excluded from pass-rate denominators.
    Exceptions propagate: an unavailable model is never counted as a failed answer.
    """
    import torch

    from .rollout import GenerationOptions
    from .tools import canonical_json, serialize_messages, validate_tool_calls, validate_tools

    if isinstance(cases, (str, Path)):
        with Path(cases).open(encoding="utf-8") as stream:
            cases = [json.loads(line) for line in stream if line.strip()]
    if not isinstance(cases, list) or not cases:
        raise ValueError("evaluation requires a nonempty list of cases")
    allowed = {
        "id",
        "messages",
        "tools",
        "expected",
        "contains",
        "expected_json",
        "expected_tool_calls",
    }
    identifiers = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) - allowed:
            raise ValueError(f"case {index}: unsupported evaluation fields")
        identifier = case.get("id", str(index))
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("case IDs must be unique nonempty strings")
        identifiers.add(identifier)
        tools = validate_tools(case["tools"]) if "tools" in case else None
        serialize_messages(case.get("messages"), tools=tools, add_generation_prompt=True)
        if "expected" in case and not isinstance(case["expected"], str):
            raise ValueError("expected must be a string")
        if "contains" in case and (
            not isinstance(case["contains"], list)
            or not all(isinstance(x, str) for x in case["contains"])
        ):
            raise ValueError("contains must be a list of strings")
        if "expected_json" in case:
            canonical_json(case["expected_json"])
        if "expected_tool_calls" in case:
            calls = case["expected_tool_calls"]
            if calls != []:
                validate_tool_calls(calls, tools=tools)
    results = []
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for index, case in enumerate(cases):
            completion = backend.complete(
                case["messages"],
                tools=case.get("tools"),
                generation=generation or GenerationOptions(temperature=0),
            )
            message = completion.message
            content = message.get("content") or ""
            checks = {}
            if "expected" in case:
                checks["exact"] = content == case["expected"]
            if "contains" in case:
                checks["contains"] = all(text in content for text in case["contains"])
            if "expected_json" in case:
                try:
                    checks["json"] = canonical_json(json.loads(content)) == canonical_json(
                        case["expected_json"]
                    )
                except (TypeError, ValueError):
                    checks["json"] = False
            if "expected_tool_calls" in case:
                # IDs are transport correlation labels, not semantic decisions.
                def semantics(calls):
                    return [
                        [
                            c["function"]["name"],
                            json.loads(c["function"]["arguments"])
                            if isinstance(c["function"]["arguments"], str)
                            else c["function"]["arguments"],
                        ]
                        for c in calls
                    ]

                checks["tool_calls"] = canonical_json(
                    semantics(message.get("tool_calls", []))
                ) == canonical_json(semantics(case["expected_tool_calls"]))
            results.append(
                {
                    "id": case.get("id", str(index)),
                    "message": message,
                    "checks": checks,
                    "passed": all(checks.values()) if checks else None,
                    "prompt_tokens": completion.prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "finish_reason": completion.finish_reason,
                }
            )
    scored = [row for row in results if row["passed"] is not None]
    report = {
        "seed": seed,
        "cases": len(results),
        "scored": len(scored),
        "passed": sum(row["passed"] for row in scored),
        "pass_rate": sum(row["passed"] for row in scored) / len(scored) if scored else None,
        "results": results,
    }
    if output is not None:
        atomic_write_bytes(
            (json.dumps(report, indent=2, allow_nan=False) + "\n").encode(), Path(output)
        )
    return report
