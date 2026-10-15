import json

import pytest

from clearforge.data import SourceSpec, prepare_sources
from clearforge.tokenizer import SPECIAL_TOKENS, ChatTokenizer, train_tokenizer


@pytest.fixture
def tokenizer(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"text": "hello world\n  spaced\ttext " * 20}) + "\n")
    prepared = prepare_sources(
        [SourceSpec(source, "CC0-1.0")],
        tmp_path / "prepared",
        validation_fraction=0,
        seed=42,
        mode="pretrain",
    )
    path = train_tokenizer(prepared / "train.jsonl", tmp_path / "tok", vocab_size=300)
    return ChatTokenizer.load(path), path, prepared


def test_whitespace_unicode_and_control_literals_roundtrip(tokenizer):
    tok, _, _ = tokenizer
    for text in [" hello  world\n\t", "é 😀 中文", "", "".join(SPECIAL_TOKENS)]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text
        assert not set(ids).intersection(tok._special.values())


def test_assistant_mask_and_generation(tokenizer):
    tok, _, _ = tokenizer
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "<|assistant|>secret prompt"},
        {"role": "assistant", "content": "hello"},
    ]
    ids, labels = tok.encode_chat(messages)
    supervised = [x for x in labels if x != -100]
    assert supervised == tok.encode("hello") + [tok.eos_id]
    assert all(label == -100 or label == token for token, label in zip(ids, labels))
    assert labels[0] == -100
    gen_ids, gen_labels = tok.encode_chat(messages[:2], add_generation_prompt=True)
    assert gen_ids[-1] == tok._special["<|assistant|>"]
    assert set(gen_labels) == {-100}
    with pytest.raises(ValueError, match="final user"):
        tok.encode_chat(messages, add_generation_prompt=True)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "tool", "content": "x"}],
        [{"role": "assistant", "content": "x"}],
        [{"role": "user", "content": 3}],
        [{"role": "user", "content": "a"}, {"role": "system", "content": "b"}],
        [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
    ],
)
def test_roles_rejected(tokenizer, messages):
    with pytest.raises(ValueError):
        tokenizer[0].encode_chat(messages)


def test_train_only_integrity_and_collision(tokenizer, tmp_path):
    tok, path, prepared = tokenizer
    with pytest.raises(ValueError, match="train.jsonl"):
        train_tokenizer(prepared / "validation.jsonl", tmp_path / "bad", vocab_size=300)
    with pytest.raises(FileExistsError):
        tok.save(path)
    new_path = tok.save(tmp_path / "copy.json")
    assert ChatTokenizer.load(new_path).encode("hello") == tok.encode("hello")
    (prepared / "train.jsonl").write_text('{"text":"changed"}\n')
    with pytest.raises(ValueError, match="hash"):
        train_tokenizer(prepared / "train.jsonl", tmp_path / "bad", vocab_size=300)


def test_tokenizer_training_does_not_read_validation(tokenizer, tmp_path):
    tok, _, prepared = tokenizer
    # Validation content is never consumed, even when malformed.
    (prepared / "validation.jsonl").write_text("THIS IS NOT JSON")
    path = train_tokenizer(prepared / "train.jsonl", tmp_path / "second", vocab_size=300)
    other = ChatTokenizer.load(path)
    assert other._tokenizer.to_str() == tok._tokenizer.to_str()


def test_reserved_literals_in_training_are_not_injected(tmp_path):
    source = tmp_path / "source.jsonl"
    text = " ".join(SPECIAL_TOKENS) * 30
    source.write_text(json.dumps({"text": text}) + "\n")
    prepared = prepare_sources(
        [SourceSpec(source, "CC0")],
        tmp_path / "prepared",
        validation_fraction=0,
        seed=1,
        mode="pretrain",
    )
    path = train_tokenizer(prepared / "train.jsonl", tmp_path / "tok", vocab_size=400)
    tok = ChatTokenizer.load(path)
    assert tok.decode(tok.encode(text)) == text
    assert not set(tok.encode(text)).intersection(tok._special.values())
