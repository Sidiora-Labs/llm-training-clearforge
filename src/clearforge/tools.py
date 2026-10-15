"""Validated tool conversations and deterministic, vocabulary-preserving wire text."""

from __future__ import annotations

import json
import re
from typing import Any

ASSISTANT_OPEN = "<clearforge_assistant>\n"
ASSISTANT_CLOSE = "\n</clearforge_assistant>"
TOOL_OPEN = "<clearforge_tool>\n"
TOOL_CLOSE = "\n</clearforge_tool>"
TOOLS_OPEN = "<clearforge_tools>\n"
TOOLS_CLOSE = "\n</clearforge_tools>"
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")
_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


def canonical_json(value: Any) -> str:
    """Stable JSON; nonfinite numbers and non-JSON objects are rejected."""

    def check(item: Any, depth: int = 0) -> None:
        if depth > 64:
            raise ValueError("JSON nesting is too deep")
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError("JSON object keys must be strings")
            for v in item.values():
                check(v, depth + 1)
        elif isinstance(item, list):
            for v in item:
                check(v, depth + 1)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError("unsupported JSON value")

    check(value)
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _parse(text: str, locator: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError("nonfinite JSON number")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"{locator}: invalid JSON") from exc


def _schema(schema: object, locator: str, depth: int = 0) -> dict[str, Any]:
    allowed = {
        "type",
        "description",
        "title",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }
    if depth > 12 or not isinstance(schema, dict) or set(schema) - allowed:
        raise ValueError(f"{locator}: unsupported JSON schema")
    kind = schema.get("type")
    if not isinstance(kind, str) or kind not in {
        "object",
        "array",
        "string",
        "integer",
        "number",
        "boolean",
        "null",
    }:
        raise ValueError(f"{locator}: schema requires a supported type")
    for key in ("description", "title"):
        if key in schema and not isinstance(schema[key], str):
            raise ValueError(f"{locator}: {key} must be text")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"{locator}: enum must be a nonempty list")
    if kind == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(props, dict) or any(not isinstance(k, str) for k in props):
            raise ValueError(f"{locator}: invalid properties")
        if (
            not isinstance(required, list)
            or any(not isinstance(k, str) or k not in props for k in required)
            or len(set(required)) != len(required)
        ):
            raise ValueError(f"{locator}: required must name unique declared properties")
        if type(schema.get("additionalProperties", True)) is not bool:
            raise ValueError(f"{locator}: additionalProperties must be boolean")
        for key, child in props.items():
            _schema(child, f"{locator}.{key}", depth + 1)
    elif any(k in schema for k in ("properties", "required", "additionalProperties")):
        raise ValueError(f"{locator}: object constraints require object type")
    if kind == "array":
        _schema(schema.get("items"), locator + ".items", depth + 1)
    elif "items" in schema:
        raise ValueError(f"{locator}: items requires array type")
    for low, high, kinds in (
        ("minLength", "maxLength", {"string"}),
        ("minItems", "maxItems", {"array"}),
        ("minimum", "maximum", {"integer", "number"}),
    ):
        for key in (low, high):
            if key in schema:
                v = schema[key]
                if (
                    kind not in kinds
                    or isinstance(v, bool)
                    or not isinstance(v, (float, int))
                    or (key not in {"minimum", "maximum"} and (type(v) is not int or v < 0))
                ):
                    raise ValueError(f"{locator}: invalid {key}")
        if low in schema and high in schema and schema[low] > schema[high]:
            raise ValueError(f"{locator}: inconsistent bounds")
    try:
        canonical_json(schema)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValueError(f"{locator}: invalid schema JSON") from exc
    return schema


def validate_arguments(value: Any, schema: dict[str, Any], locator: str = "arguments") -> None:
    """Validate the supported, explicitly checked JSON-schema subset."""
    schema = _schema(schema, locator + ".schema")
    kind = schema["type"]
    match = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "boolean": type(value) is bool,
        "null": value is None,
    }
    if not match[kind]:
        raise ValueError(f"{locator}: expected {kind}")
    if "enum" in schema and canonical_json(value) not in [
        canonical_json(v) for v in schema["enum"]
    ]:
        raise ValueError(f"{locator}: value outside enum")
    if kind == "object":
        props = schema.get("properties", {})
        if set(schema.get("required", [])) - set(value):
            raise ValueError(f"{locator}: missing required arguments")
        if not schema.get("additionalProperties", True) and set(value) - set(props):
            raise ValueError(f"{locator}: unknown arguments")
        for name, item in value.items():
            if name in props:
                validate_arguments(item, props[name], f"{locator}.{name}")
    if kind == "array":
        for i, item in enumerate(value):
            validate_arguments(item, schema["items"], f"{locator}[{i}]")
    for low, high, target in (
        ("minimum", "maximum", value),
        ("minLength", "maxLength", len(value) if kind == "string" else 0),
        ("minItems", "maxItems", len(value) if kind == "array" else 0),
    ):
        if low in schema and target < schema[low] or high in schema and target > schema[high]:
            raise ValueError(f"{locator}: value outside bounds")
    try:
        canonical_json(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{locator}: arguments must be finite JSON") from exc


def validate_tools(tools: object, locator: str = "tools") -> list[dict[str, Any]]:
    if not isinstance(tools, list) or len(tools) > 128:
        raise ValueError(f"{locator}: tools must be a list of at most 128 functions")
    clean, names = [], set()
    for i, tool in enumerate(tools):
        where = f"{locator}[{i}]"
        if (
            not isinstance(tool, dict)
            or set(tool) != {"type", "function"}
            or tool["type"] != "function"
        ):
            raise ValueError(f"{where}: expected function tool")
        f = tool["function"]
        if (
            not isinstance(f, dict)
            or set(f) - {"name", "description", "parameters", "strict"}
            or not isinstance(f.get("name"), str)
            or not _NAME.fullmatch(f["name"])
            or f["name"] in names
        ):
            raise ValueError(f"{where}: invalid or duplicate function name")
        if "description" in f and not isinstance(f["description"], str):
            raise ValueError(f"{where}: description must be text")
        if "strict" in f and type(f["strict"]) is not bool:
            raise ValueError(f"{where}: strict must be boolean")
        schema = _schema(f.get("parameters"), where + ".parameters")
        if schema["type"] != "object":
            raise ValueError(f"{where}: parameters must describe an object")
        names.add(f["name"])
        clean.append(_parse(canonical_json(tool), where))
    return clean


def validate_tool_calls(
    calls: object, locator: str = "tool_calls", *, tools: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if not isinstance(calls, list) or not calls or len(calls) > 128:
        raise ValueError(f"{locator}: calls must be a nonempty list of at most 128")
    schemas = (
        {t["function"]["name"]: t["function"]["parameters"] for t in validate_tools(tools)}
        if tools is not None
        else None
    )
    clean, ids = [], set()
    for i, call in enumerate(calls):
        where = f"{locator}[{i}]"
        if (
            not isinstance(call, dict)
            or set(call) != {"id", "type", "function"}
            or call["type"] != "function"
            or not isinstance(call["id"], str)
            or not _ID.fullmatch(call["id"])
            or call["id"] in ids
        ):
            raise ValueError(f"{where}: invalid or duplicate call ID")
        f = call["function"]
        if (
            not isinstance(f, dict)
            or set(f) != {"name", "arguments"}
            or not isinstance(f["name"], str)
            or not _NAME.fullmatch(f["name"])
        ):
            raise ValueError(f"{where}: invalid function")
        args = (
            _parse(f["arguments"], where + ".arguments")
            if isinstance(f["arguments"], str)
            else f["arguments"]
        )
        if not isinstance(args, dict) or any(not isinstance(k, str) for k in args):
            raise ValueError(f"{where}: arguments must be a JSON object")
        if schemas is not None:
            if f["name"] not in schemas:
                raise ValueError(f"{where}: undeclared function")
            validate_arguments(args, schemas[f["name"]], where + ".arguments")
        try:
            arguments = canonical_json(args)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError(f"{where}: invalid argument JSON") from exc
        clean.append(
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": f["name"], "arguments": arguments},
            }
        )
        ids.add(call["id"])
    return clean


def validate_conversation(
    messages: object, locator: str = "messages", *, tools: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{locator}: messages must be a nonempty list")
    if tools is not None:
        tools = validate_tools(tools, locator + ".tools")
    clean, seen, pending = [], set(), set()
    expected = "user"
    for i, message in enumerate(messages):
        where = f"{locator}[{i}]"
        if not isinstance(message, dict):
            raise ValueError(f"{where}: message must be an object")
        role = message.get("role")
        allowed = {"role", "content"} | (
            {"tool_calls", "reasoning_content"}
            if role == "assistant"
            else {"tool_call_id"}
            if role == "tool"
            else set()
        )
        if set(message) - allowed or role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"{where}: unsupported role or message fields")
        content = message.get(
            "content", "" if role == "assistant" and message.get("tool_calls") else None
        )
        if content is None and role == "assistant" and message.get("tool_calls"):
            content = ""
        if not isinstance(content, str):
            raise ValueError(f"{where}: content must be text")
        m = {"role": role, "content": content}
        if role == "system":
            if i != 0:
                raise ValueError(f"{where}: system role must be first")
        elif role != expected:
            raise ValueError(f"{where}: expected {expected} role")
        elif role == "user":
            expected = "assistant"
        elif role == "assistant":
            if "reasoning_content" in message:
                if not isinstance(message["reasoning_content"], str):
                    raise ValueError(f"{where}: reasoning_content must be text")
                m["reasoning_content"] = message["reasoning_content"]
            if "tool_calls" in message:
                calls = validate_tool_calls(
                    message["tool_calls"], where + ".tool_calls", tools=tools
                )
                pending = {c["id"] for c in calls}
                if pending & seen:
                    raise ValueError(f"{where}: call IDs cannot be reused")
                seen.update(pending)
                m["tool_calls"] = calls
                expected = "tool"
            else:
                expected = "user"
        else:
            ident = message.get("tool_call_id")
            if not isinstance(ident, str) or ident not in pending:
                raise ValueError(f"{where}: unknown or duplicate tool_call_id")
            m["tool_call_id"] = ident
            pending.remove(ident)
            expected = "tool" if pending else "assistant"
        clean.append(m)
    # A final assistant call is a valid training target; partially answered batches are not.
    if pending and clean[-1]["role"] != "assistant":
        raise ValueError(f"{locator}: incomplete tool response batch")
    return clean


def serialize_messages(
    messages: object,
    *,
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = False,
) -> list[tuple[str, str]]:
    clean = validate_conversation(messages, tools=tools)
    if add_generation_prompt and clean[-1]["role"] not in {"user", "tool"}:
        raise ValueError(
            "generation prompt requires a final user message or completed tool responses"
        )
    output = []
    if tools:
        schema_text = TOOLS_OPEN + canonical_json(validate_tools(tools)) + TOOLS_CLOSE
        if clean[0]["role"] == "system":
            clean[0]["content"] += "\n\n" + schema_text
        else:
            clean.insert(0, {"role": "system", "content": schema_text})
    for m in clean:
        role = m["role"]
        if role == "tool":
            text = (
                TOOL_OPEN
                + canonical_json({"content": m["content"], "tool_call_id": m["tool_call_id"]})
                + TOOL_CLOSE
            )
            role = "user"
        elif role == "assistant" and ("tool_calls" in m or "reasoning_content" in m):
            text = (
                ASSISTANT_OPEN
                + canonical_json({k: v for k, v in m.items() if k != "role"})
                + ASSISTANT_CLOSE
            )
        else:
            text = m["content"]
        output.append((role, text))
    return output


def parse_assistant_output(
    text: str, *, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Parse the exact structured envelope; validate before any caller dispatches a tool."""
    if not isinstance(text, str):
        raise ValueError("assistant output must be text")
    if not text.startswith(ASSISTANT_OPEN):
        if ASSISTANT_OPEN.rstrip() in text or ASSISTANT_CLOSE.strip() in text:
            raise ValueError("malformed structured assistant envelope")
        return {"role": "assistant", "content": text}
    if not text.endswith(ASSISTANT_CLOSE):
        raise ValueError("incomplete structured assistant envelope")
    value = _parse(text[len(ASSISTANT_OPEN) : -len(ASSISTANT_CLOSE)], "assistant output")
    if (
        not isinstance(value, dict)
        or set(value) - {"content", "reasoning_content", "tool_calls"}
        or not ({"tool_calls", "reasoning_content"} & set(value))
    ):
        raise ValueError("structured output requires calls or reasoning_content")
    return validate_conversation(
        [{"role": "user", "content": ""}, {**value, "role": "assistant"}], tools=tools
    )[-1]
