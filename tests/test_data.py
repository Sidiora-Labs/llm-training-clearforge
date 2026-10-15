import json

import numpy as np
import pytest
import torch

from clearforge.data import PackedDataset, SourceSpec, pack_dataset, prepare_sources
from clearforge.tokenizer import ChatTokenizer, train_tokenizer


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_content_disjoint_deterministic_multi_source(tmp_path):
    rows = [{"text": f"English document number {i}"} for i in range(100)]
    a = write(tmp_path / "a.jsonl", rows)
    b = write(tmp_path / "b.jsonl", rows[::-1])
    specs = [SourceSpec(a, "CC0"), SourceSpec(b, "MIT")]
    outputs = []
    for i, sources in enumerate([specs, specs[::-1]]):
        p = prepare_sources(
            sources, tmp_path / str(i), validation_fraction=0.3, seed=7, mode="pretrain"
        )
        outputs.append(p)
        manifest = json.loads((p / "manifest.json").read_text())
        assert manifest["duplicates_removed"] == 100
        train = set((p / "train.jsonl").read_text().splitlines())
        val = set((p / "validation.jsonl").read_text().splitlines())
        assert train and val and not train.intersection(val)
        assert len(train | val) == 100
    for split in ("train", "validation"):
        assert (outputs[0] / f"{split}.jsonl").read_bytes() == (
            outputs[1] / f"{split}.jsonl"
        ).read_bytes()


def test_language_declarations_and_license(tmp_path):
    a = write(tmp_path / "a.jsonl", [{"text": "English"}])
    b = write(tmp_path / "b.jsonl", [{"text": "bonjour"}])
    p = prepare_sources(
        [SourceSpec(a, "CC0"), SourceSpec(b, "CC-BY", "fr")],
        tmp_path / "prepared",
        validation_fraction=0,
        seed=1,
        mode="pretrain",
    )
    m = json.loads((p / "manifest.json").read_text())
    assert m["sources"][1]["excluded"] == "source_language_not_english"
    assert "no language detection" in m["language_policy"]
    with pytest.raises(ValueError, match="license"):
        prepare_sources(
            [SourceSpec(a, "")], tmp_path / "bad", validation_fraction=0, seed=1, mode="pretrain"
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("bad", ["not json", "[]", '{"text":0}', '{"text":""}', ""])
def test_malformed_locator_atomic_failure(tmp_path, bad):
    a = tmp_path / "bad.jsonl"
    a.write_text('{"text":"okay"}\n' + bad + "\n")
    with pytest.raises(ValueError, match=r"bad.jsonl:2"):
        prepare_sources(
            [SourceSpec(a, "CC0")], tmp_path / "out", validation_fraction=0, seed=1, mode="pretrain"
        )
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "out.lock").exists()


def test_sft_tails_masks_memmap_and_exclusions(tmp_path):
    messages = [
        {"role": "system", "content": "private policy"},
        {"role": "user", "content": "long prompt " * 20},
        {"role": "assistant", "content": "answer " * 21},
        {"role": "user", "content": "second prompt"},
        {"role": "assistant", "content": "tail"},
    ]
    a = write(
        tmp_path / "chat.jsonl",
        [{"messages": messages}, {"messages": [{"role": "user", "content": "no answer"}]}],
    )
    p = prepare_sources(
        [SourceSpec(a, "CC0")], tmp_path / "prep", validation_fraction=0, seed=1, mode="sft"
    )
    tp = train_tokenizer(p / "train.jsonl", tmp_path / "tok", vocab_size=300)
    tok = ChatTokenizer.load(tp)
    packed = pack_dataset(p, tp, tmp_path / "packed", seq_length=9, mode="sft")
    ds = PackedDataset(packed, "train")
    assert isinstance(ds.input_ids, np.memmap)
    assert ds[0]["input_ids"].dtype == torch.long
    assert len(PackedDataset(packed, "validation")) == 0
    actual = []
    for row in ds:
        labels = row["labels"].tolist()
        assert labels[0] == -100
        assert any(x != -100 for x in labels[1:])
        actual.extend(x for x in labels if x != -100)
    expected = [x for x in tok.encode_chat(messages)[1] if x != -100]
    assert actual == expected  # Every assistant target exactly once, including final EOS.
    assert (ds.labels[ds.input_ids == tok.pad_id] == -100).all()
    m = json.loads((packed / "manifest.json").read_text())
    assert m["exclusions"]["train"]["no_supervision_documents"] == 1
    assert m["exclusions"]["train"]["no_shifted_supervision_windows"] > 0
    with pytest.raises(FileExistsError):
        pack_dataset(p, tp, packed, seq_length=9, mode="sft")


def test_pretrain_preserves_all_tokens_and_detects_tampering(tmp_path):
    a = write(tmp_path / "input.jsonl", [{"text": "abc " * 31}])
    p = prepare_sources(
        [SourceSpec(a, "CC0")], tmp_path / "prep", validation_fraction=0, seed=1, mode="pretrain"
    )
    tp = train_tokenizer(p / "train.jsonl", tmp_path / "tok", vocab_size=280)
    tok = ChatTokenizer.load(tp)
    packed = pack_dataset(p, tp, tmp_path / "packed", seq_length=8, mode="pretrain")
    ds = PackedDataset(packed, "train")
    targets = [x for row in ds for x in row["labels"].tolist() if x != -100]
    assert targets == tok.encode("abc " * 31) + [tok.eos_id]
    (p / "train.jsonl").write_text('{"text":"tamper"}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        pack_dataset(p, tp, tmp_path / "bad", seq_length=8, mode="pretrain")
    assert not (tmp_path / "bad").exists()


def test_invalid_utf8_and_source_license_lineage(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_bytes(b'{"text":"ok"}\n{"text":"\xff"}\n')
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        prepare_sources(
            [SourceSpec(bad, "CC0")],
            tmp_path / "badout",
            validation_fraction=0,
            seed=1,
            mode="pretrain",
        )
    a = write(tmp_path / "a.jsonl", [{"text": "same"}])
    b = write(tmp_path / "b.jsonl", [{"text": "same"}])
    out = prepare_sources(
        [SourceSpec(a, "CC0"), SourceSpec(b, "MIT")],
        tmp_path / "out",
        validation_fraction=0,
        seed=1,
        mode="pretrain",
    )
    lineage = [json.loads(line) for line in (out / "provenance.jsonl").read_text().splitlines()]
    assert {r["license"] for r in lineage} == {"CC0", "MIT"}
    assert len({r["content_sha256"] for r in lineage}) == 1
    assert sum(r["duplicate"] for r in lineage) == 1


def test_no_supervision_fails_and_tokenizer_integrity(tmp_path):
    source = write(
        tmp_path / "chat.jsonl", [{"messages": [{"role": "user", "content": "only a prompt"}]}]
    )
    prep = prepare_sources(
        [SourceSpec(source, "CC0")], tmp_path / "prep", validation_fraction=0, seed=1, mode="sft"
    )
    tp = train_tokenizer(prep / "train.jsonl", tmp_path / "tok", vocab_size=280)
    with pytest.raises(ValueError, match="no shifted training"):
        pack_dataset(prep, tp, tmp_path / "packed", seq_length=8, mode="sft")
    assert not (tmp_path / "packed").exists()
    other = write(tmp_path / "other.jsonl", [{"text": "different training data"}])
    other_prep = prepare_sources(
        [SourceSpec(other, "MIT")],
        tmp_path / "otherprep",
        validation_fraction=0,
        seed=1,
        mode="pretrain",
    )
    tp.write_text(tp.read_text() + " ")
    with pytest.raises(ValueError, match="tokenizer hash mismatch"):
        pack_dataset(other_prep, tp, tmp_path / "wrong", seq_length=8, mode="pretrain")


def test_pretraining_tokenizer_reused_for_new_sft_corpus(tmp_path):
    source = write(tmp_path / "pretrain.jsonl", [{"text": "General English prose."}])
    prep = prepare_sources(
        [SourceSpec(source, "CC0")],
        tmp_path / "prep",
        validation_fraction=0,
        seed=1,
        mode="pretrain",
    )
    tp = train_tokenizer(prep / "train.jsonl", tmp_path / "tok", vocab_size=280)
    original_tokenizer_bytes = tp.read_bytes()
    messages = [
        {"role": "user", "content": "A completely new question"},
        {"role": "assistant", "content": "A new answer."},
    ]
    chat = write(tmp_path / "chat.jsonl", [{"messages": messages}])
    chat_prep = prepare_sources(
        [SourceSpec(chat, "MIT")], tmp_path / "chatprep", validation_fraction=0, seed=9, mode="sft"
    )
    packed = pack_dataset(chat_prep, tp, tmp_path / "sft", seq_length=8, mode="sft")
    dataset = PackedDataset(packed, "train")
    tok = ChatTokenizer.load(tp)
    targets = [x for row in dataset for x in row["labels"].tolist() if x != -100]
    assert targets == tok.encode("A new answer.") + [tok.eos_id]
    assert tp.read_bytes() == original_tokenizer_bytes
    manifest = json.loads((packed / "manifest.json").read_text())
    assert manifest["tokenizer_training_split_sha256"] != manifest["source_hashes"]["train"]
