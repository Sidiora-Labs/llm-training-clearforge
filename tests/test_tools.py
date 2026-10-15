import copy
import json

import numpy as np
import pytest
import torch

from clearforge.config import ModelConfig
from clearforge.data import (
    SourceSpec,
    adapt_conversations,
    adapt_preference,
    adapt_rollout,
    convert_dataset,
    pack_dataset,
    prepare_sources,
    validate_messages,
)
from clearforge.model import CausalLM
from clearforge.tokenizer import ChatTokenizer, train_tokenizer
from clearforge.tools import (
    ASSISTANT_CLOSE,
    ASSISTANT_OPEN,
    canonical_json,
    parse_assistant_output,
    serialize_messages,
    validate_arguments,
    validate_tool_calls,
    validate_tools,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add integers",
            "parameters": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                "required": ["x", "y"],
                "additionalProperties": False,
            },
        },
    }
]


def call(ident="c1", args=None):
    return {
        "id": ident,
        "type": "function",
        "function": {"name": "add", "arguments": args if args is not None else {"x": 2, "y": 3}},
    }


def conversation():
    return [
        {"role": "user", "content": "Add two and three"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Use addition.",
            "tool_calls": [call()],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "5"},
        {"role": "assistant", "content": "The answer is 5."},
    ]


@pytest.fixture
def tokenizer(tmp_path):
    source = tmp_path / "text.jsonl"
    source.write_text(json.dumps({"text": "hello world"}) + "\n")
    prep = prepare_sources(
        [SourceSpec(source, "Apache-2.0")],
        tmp_path / "pretrain",
        validation_fraction=0,
        seed=1,
        mode="pretrain",
    )
    path = train_tokenizer(prep / "train.jsonl", tmp_path / "tok", vocab_size=262)
    return ChatTokenizer.load(path), path


def test_tools_roundtrip_masks_and_plain_message_identity(tokenizer):
    tok, _ = tokenizer
    plain = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    expected = [tok.bos_id]
    for m in plain:
        expected += [tok._special[f"<|{m['role']}|>"]] + tok.encode(m["content"]) + [tok.eos_id]
    assert tok.encode_chat(plain)[0] == expected
    messages = conversation()
    normalized = validate_messages(messages, tools=TOOLS)
    assert messages[1]["content"] is None
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == '{"x":2,"y":3}'
    rendered = serialize_messages(messages, tools=TOOLS)
    assert rendered[0][0] == "system" and rendered[3][0] == "user"
    assert parse_assistant_output(rendered[2][1], tools=TOOLS) == normalized[1]
    ids, labels = tok.encode_chat(messages, tools=TOOLS)
    expected_targets = []
    for role, text in rendered:
        if role == "assistant":
            expected_targets += tok.encode(text) + [tok.eos_id]
    assert [v for v in labels if v != -100] == expected_targets
    assert tok.vocab_size == 262
    assert tok.decode(ids) == tok.apply_chat_template(messages, tools=TOOLS, tokenize=False)
    assert tok.apply_chat_template(messages, tools=TOOLS) == ids
    prompt_ids, prompt_labels = tok.encode_chat(
        messages[:-1], tools=TOOLS, add_generation_prompt=True
    )
    assert prompt_ids[-1] == tok._special["<|assistant|>"] and prompt_labels[-1] == -100


def test_reserved_literals_do_not_inject_controls(tokenizer):
    tok, _ = tokenizer
    text = "".join(tok._special)
    messages = conversation()
    messages[0]["content"] = text
    messages[1]["reasoning_content"] = text
    messages[2]["content"] = text
    ids, _ = tok.encode_chat(messages, tools=TOOLS)
    # One BOS, one role token and one EOS per serialized message. No literal controls leak.
    assert sum(v in set(tok._special.values()) for v in ids) == 1 + 2 * 5
    assert text in tok.decode(ids)


@pytest.mark.parametrize(
    "mutation",
    [
        "orphan",
        "wrong_id",
        "duplicate_result",
        "missing_result",
        "early_answer",
        "repeat_call_id",
        "late_system",
        "null_reasoning",
        "unknown_field",
        "empty_calls",
    ],
)
def test_tool_state_machine_rejects_invalid_histories(mutation):
    m = conversation()
    if mutation == "orphan":
        m = [m[0], m[2]]
    elif mutation == "wrong_id":
        m[2]["tool_call_id"] = "other"
    elif mutation == "duplicate_result":
        m.insert(3, copy.deepcopy(m[2]))
    elif mutation == "missing_result":
        m[1]["tool_calls"].append(call("c2"))
        m = m[:3]
    elif mutation == "early_answer":
        m.pop(2)
    elif mutation == "repeat_call_id":
        m = m[:3] + [m[1]]
    elif mutation == "late_system":
        m.append({"role": "system", "content": "oops"})
    elif mutation == "null_reasoning":
        m[1]["reasoning_content"] = None
    elif mutation == "unknown_field":
        m[2]["unknown"] = True
    elif mutation == "empty_calls":
        m[1]["tool_calls"] = []
    with pytest.raises(ValueError):
        validate_messages(m, tools=TOOLS)


def test_parallel_calls_final_target_and_pending_prompt(tokenizer):
    tok, _ = tokenizer
    m = conversation()[:2]
    m[1]["tool_calls"].append(call("c2"))
    assert len(validate_messages(m, tools=TOOLS)) == 2
    with pytest.raises(ValueError, match="final user"):
        tok.encode_chat(m, tools=TOOLS, add_generation_prompt=True)
    m += [
        {"role": "tool", "tool_call_id": "c2", "content": "5"},
        {"role": "tool", "tool_call_id": "c1", "content": "5"},
    ]
    assert (
        tok.encode_chat(m, tools=TOOLS, add_generation_prompt=True)[0][-1]
        == tok._special["<|assistant|>"]
    )


@pytest.mark.parametrize(
    "args",
    [
        {"x": True, "y": 3},
        {"x": 1},
        {"x": 1, "y": 3, "z": 0},
        "[]",
        '{"x":1,"x":2,"y":3}',
        '{"x":NaN,"y":3}',
    ],
)
def test_arguments_validate_against_schema(args):
    with pytest.raises(ValueError):
        validate_tool_calls([call(args=args)], tools=TOOLS)


def test_schema_subset_and_unknown_functions():
    schema = {
        "type": "array",
        "items": {"type": "string", "enum": ["yes", "no"], "minLength": 2},
        "maxItems": 2,
    }
    validate_arguments(["yes", "no"], schema)
    with pytest.raises(ValueError):
        validate_arguments(["maybe"], schema)
    with pytest.raises(ValueError):
        validate_arguments(["yes"] * 3, schema)
    bad = copy.deepcopy(TOOLS)
    bad[0]["function"]["parameters"]["$ref"] = "external"
    with pytest.raises(ValueError):
        validate_tools(bad)
    c = call()
    c["function"]["name"] = "unknown"
    with pytest.raises(ValueError):
        validate_tool_calls([c], tools=TOOLS)
    with pytest.raises(ValueError):
        validate_tools(TOOLS + TOOLS)
    with pytest.raises(ValueError):
        validate_arguments(1, {"type": []})
    with pytest.raises(ValueError):
        canonical_json({1: "invalid"})


@pytest.mark.parametrize(
    "text",
    [
        ASSISTANT_OPEN + "{}" + ASSISTANT_CLOSE,
        ASSISTANT_OPEN + "broken" + ASSISTANT_CLOSE,
        ASSISTANT_OPEN + "{}",
        "prefix " + ASSISTANT_OPEN + "{}" + ASSISTANT_CLOSE,
        ASSISTANT_OPEN + '{"role":"user","reasoning_content":"x","content":"x"}' + ASSISTANT_CLOSE,
    ],
)
def test_malformed_model_output_fails_closed(text):
    with pytest.raises(ValueError):
        parse_assistant_output(text, tools=TOOLS)


def test_actual_packing_and_model_update_with_frozen_tokenizer(tokenizer, tmp_path):
    tok, tp = tokenizer
    source = tmp_path / "chat.jsonl"
    source.write_text(json.dumps({"messages": conversation(), "tools": TOOLS}) + "\n")
    original = tp.read_bytes()
    prep = prepare_sources(
        [SourceSpec(source, "Apache-2.0")],
        tmp_path / "sft",
        validation_fraction=0,
        seed=1,
        mode="sft",
    )
    prepared = json.loads((prep / "train.jsonl").read_text())
    assert prepared["tools"] == TOOLS
    packed = pack_dataset(prep, tp, tmp_path / "packed", seq_length=32, mode="sft")
    labels = np.load(packed / "train.labels.npy")
    expected = tok.encode_chat(conversation(), tools=TOOLS)[1]
    assert labels[labels != -100].tolist() == [v for v in expected if v != -100]
    assert tp.read_bytes() == original
    ids = torch.from_numpy(np.load(packed / "train.input_ids.npy")[:2].copy())
    y = torch.from_numpy(labels[:2].copy())
    model = CausalLM(
        ModelConfig(
            vocab_size=262,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=32,
            max_seq_len=32,
        )
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    before = next(model.parameters()).detach().clone()
    loss = model(ids, labels=y).loss
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()))


def test_explicit_adapters_and_conversion_lineage(tmp_path):
    m = conversation()
    assert adapt_conversations({"conversations": m, "tools": TOOLS})[
        "messages"
    ] == validate_messages(m, tools=TOOLS)
    pair = adapt_preference({"prompt": "hello", "chosen": "yes", "rejected": "no"})
    assert pair["chosen"][:-1] == pair["rejected"][:-1]
    with pytest.raises(ValueError):
        adapt_preference({"prompt": "x", "chosen": "same", "rejected": "same"})
    with pytest.raises(ValueError):
        adapt_preference(
            {
                "chosen": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "x"}],
                "rejected": [
                    {"role": "user", "content": "b"},
                    {"role": "assistant", "content": "y"},
                ],
            }
        )
    assert adapt_rollout({"prompt": "hello", "gt": {"result": 5}}) == {
        "messages": [{"role": "user", "content": "hello"}],
        "expected": {"result": 5},
    }
    with pytest.raises(ValueError):
        adapt_rollout({"messages": m[:2], "tools": TOOLS})
    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps({"conversations": m, "tools": TOOLS}) + "\n")
    out = convert_dataset(
        SourceSpec(path, "Apache-2.0"), tmp_path / "converted", adapter="conversations"
    )
    assert json.loads(out.read_text())["tools"] == TOOLS
    manifest = json.loads((out.parent / "manifest.json").read_text())
    assert manifest["records"] == 1 and manifest["source"]["license"] == "Apache-2.0"
    assert manifest["adapter"] == "conversations"
    path.write_text('{"conversations": []}\n')
    with pytest.raises(ValueError, match="input.jsonl:1"):
        convert_dataset(SourceSpec(path, "Apache-2.0"), tmp_path / "bad", adapter="conversations")
    assert not (tmp_path / "bad").exists()
