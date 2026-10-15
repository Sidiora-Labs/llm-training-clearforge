"""Auditable, disk-backed document preparation and causal sequence packing.

Ingestion holds one JSONL record in memory and uses SQLite for global deduplication.
Packing holds one encoded document/conversation and one sequence window in memory.
Language filtering trusts source declarations; it does not detect language.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .checkpoint import publish_directory


@dataclass(frozen=True)
class SourceSpec:
    path: str | Path
    license: str
    language: str = "en"


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: str | Path, *, kind: str) -> tuple[dict[str, Any], str]:
    """Validate version and controls consumed by the pipeline, hashing the parsed bytes."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"{path}: malformed manifest") from exc
    if (
        not isinstance(value, dict)
        or type(value.get("format_version")) is not int
        or value["format_version"] != 1
    ):
        raise ValueError(f"{path}: unsupported manifest format_version")

    def digest(v: object) -> bool:
        return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)

    if kind == "prepared":
        if value.get("mode") not in ("pretrain", "sft"):
            raise ValueError(f"{path}: invalid manifest mode")
        hashes = value.get("split_hashes")
        counts = value.get("document_counts")
        if not isinstance(hashes, dict) or any(
            not digest(hashes.get(k)) for k in ("train", "validation")
        ):
            raise ValueError(f"{path}: invalid split_hashes")
        if not isinstance(counts, dict) or any(
            type(counts.get(k)) is not int or counts[k] < 0 for k in ("train", "validation")
        ):
            raise ValueError(f"{path}: invalid document_counts")
    elif kind == "tokenizer":
        if any(
            not digest(value.get(k))
            for k in ("tokenizer_sha256", "training_split_sha256", "source_manifest_sha256")
        ):
            raise ValueError(f"{path}: invalid tokenizer provenance hashes")
        if type(value.get("vocab_size")) is not int or value["vocab_size"] < 262:
            raise ValueError(f"{path}: invalid vocab_size")
    else:
        raise ValueError("unknown manifest kind")
    return value, hashlib.sha256(raw).hexdigest()


def verify_unchanged(path: str | Path, expected: str) -> None:
    if sha256(path) != expected:
        raise ValueError(f"{path}: input changed during processing")


@contextmanager
def atomic_directory(output_dir: str | Path) -> Iterator[Path]:
    """Publish without replacement; a sibling lock excludes cooperating writers."""
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        if output.exists():
            raise FileExistsError(output)
        tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        yield tmp
        if output.exists():
            raise FileExistsError(output)
        publish_directory(tmp, output)
        tmp = None
    finally:
        os.close(fd)
        lock.unlink()
        if tmp is not None:
            shutil.rmtree(tmp)


def validate_messages(
    messages: object, locator: str = "messages", *, tools: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    from .tools import validate_conversation

    return validate_conversation(messages, locator, tools=tools)


def records(path: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with Path(path).open("rb") as f:
        for line_no, line in enumerate(f, 1):
            locator = f"{path}:{line_no}"
            try:
                record = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeError) as e:
                raise ValueError(f"{locator}: malformed JSON record") from e
            if not isinstance(record, dict):
                raise ValueError(f"{locator}: expected an object")
            yield locator, record


def _mode(mode: str) -> None:
    if mode not in ("pretrain", "sft"):
        raise ValueError("mode must be pretrain or sft")


def prepare_sources(
    sources: Iterable[SourceSpec],
    output_dir: str | Path,
    *,
    validation_fraction: float,
    seed: int,
    mode: str,
) -> Path:
    _mode(mode)
    if not 0 <= validation_fraction < 1 or isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("invalid split fraction or seed")
    sources = list(sources)
    if not sources:
        raise ValueError("at least one source is required")
    with atomic_directory(output_dir) as tmp:
        db = sqlite3.connect(tmp / "dedup.sqlite")
        lineage = (tmp / "provenance.jsonl").open("w", encoding="utf-8")
        try:
            db.execute("CREATE TABLE docs (hash TEXT PRIMARY KEY, body TEXT, split TEXT)")
            provenance = []
            duplicates = 0
            for source in sources:
                if (
                    not isinstance(source, SourceSpec)
                    or not isinstance(source.license, str)
                    or not source.license.strip()
                ):
                    raise ValueError("every source requires a nonempty license declaration")
                if not isinstance(source.language, str) or not source.language.strip():
                    raise ValueError("source language must be a nonempty declaration")
                path = Path(source.path)
                info = {
                    "path": str(path.resolve()),
                    "license": source.license,
                    "language": source.language,
                    "sha256": sha256(path),
                    "records": 0,
                }
                provenance.append(info)
                if source.language.lower() not in ("en", "eng", "english"):
                    info["excluded"] = "source_language_not_english"
                    continue
                for locator, record in records(path):
                    if mode == "pretrain":
                        text = record.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(f"{locator}: text must be a nonempty string")
                        body = {"text": text}
                    else:
                        from .tools import validate_tools

                        tool_list = (
                            validate_tools(record["tools"], locator + ".tools")
                            if "tools" in record
                            else None
                        )
                        body = {
                            "messages": validate_messages(
                                record.get("messages"), locator, tools=tool_list
                            )
                        }
                        if tool_list is not None:
                            body["tools"] = tool_list
                    serialized = json.dumps(body, ensure_ascii=False, sort_keys=True)
                    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
                    score = int(hashlib.sha256(f"{seed}:{fingerprint}".encode()).hexdigest(), 16)
                    split = "validation" if score / 2**256 < validation_fraction else "train"
                    info["records"] += 1
                    cursor = db.execute(
                        "INSERT OR IGNORE INTO docs VALUES (?, ?, ?)",
                        (fingerprint, serialized, split),
                    )
                    duplicates += cursor.rowcount == 0
                    lineage.write(
                        json.dumps(
                            {
                                "locator": locator,
                                "source_sha256": info["sha256"],
                                "license": source.license,
                                "language": source.language,
                                "content_sha256": fingerprint,
                                "split": split,
                                "duplicate": cursor.rowcount == 0,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                if sha256(path) != info["sha256"]:
                    raise ValueError(f"{path}: source changed during preparation")
                db.commit()
            counts = {}
            hashes = {}
            for split in ("train", "validation"):
                counts[split] = 0
                with (tmp / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                    for (body,) in db.execute(
                        "SELECT body FROM docs WHERE split=? ORDER BY hash", (split,)
                    ):
                        f.write(body + "\n")
                        counts[split] += 1
                hashes[split] = sha256(tmp / f"{split}.jsonl")
            if not counts["train"]:
                raise ValueError("split has no training documents; change seed or split fraction")
        finally:
            db.close()
            lineage.close()
        (tmp / "dedup.sqlite").unlink()
        manifest = {
            "format_version": 1,
            "mode": mode,
            "seed": seed,
            "validation_fraction": validation_fraction,
            "sources": provenance,
            "duplicates_removed": duplicates,
            "document_counts": counts,
            "provenance_sha256": sha256(tmp / "provenance.jsonl"),
            "split_hashes": hashes,
            "language_policy": "declared English sources only; no language detection",
            "deduplication": "exact canonical record content across all included sources",
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return Path(output_dir)


def pack_dataset(
    prepared_dir: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    *,
    seq_length: int,
    mode: str,
) -> Path:
    from .tokenizer import ChatTokenizer

    _mode(mode)
    if isinstance(seq_length, bool) or not isinstance(seq_length, int) or seq_length < 2:
        raise ValueError("seq_length must be an integer >= 2")
    prepared = Path(prepared_dir)
    source_manifest, source_manifest_hash = read_manifest(
        prepared / "manifest.json", kind="prepared"
    )
    if source_manifest["mode"] != mode:
        raise ValueError("prepared mode mismatch")
    tokenizer_manifest_path = Path(tokenizer_path).parent / "manifest.json"
    tokenizer_manifest, tokenizer_manifest_hash = (
        read_manifest(tokenizer_manifest_path, kind="tokenizer")
        if tokenizer_manifest_path.exists()
        else ({}, None)
    )
    tokenizer_hash = sha256(tokenizer_path)
    expected_tokenizer_hash = tokenizer_manifest.get("tokenizer_sha256")
    if expected_tokenizer_hash is not None and tokenizer_hash != expected_tokenizer_hash:
        raise ValueError("tokenizer hash mismatch")
    tok = ChatTokenizer.load(tokenizer_path)
    if tokenizer_manifest and tokenizer_manifest["vocab_size"] != tok.vocab_size:
        raise ValueError("tokenizer vocabulary size mismatch")
    row_counts, exclusions = {}, {}
    with atomic_directory(output_dir) as tmp:
        for split in ("train", "validation"):
            path = prepared / f"{split}.jsonl"
            if sha256(path) != source_manifest["split_hashes"][split]:
                raise ValueError(f"{path}: prepared split hash mismatch")
            row_counts[split] = 0
            exclusions[split] = {"no_shifted_supervision_windows": 0, "no_supervision_documents": 0}
            raw_ids, raw_labels = tmp / f"{split}.ids.raw", tmp / f"{split}.labels.raw"
            with (
                raw_ids.open("wb") as ids_file,
                raw_labels.open("wb") as labels_file,
                (tmp / f"{split}.exclusions.jsonl").open("w") as excluded_file,
            ):
                count = 0
                pending_ids, pending_labels = [], []

                def write_window(x: list[int], y: list[int]) -> bool:
                    y = y.copy()
                    y[0] = -100
                    if not any(label != -100 for label in y[1:]):
                        return False
                    padding = seq_length - len(x)
                    np.asarray(x + [tok.pad_id] * padding, dtype=np.int64).tofile(ids_file)
                    np.asarray(y + [-100] * padding, dtype=np.int64).tofile(labels_file)
                    row_counts[split] += 1
                    return True

                for locator, record in records(path):
                    count += 1
                    if mode == "pretrain":
                        text = record.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(f"{locator}: text must be a nonempty string")
                        ids = [tok.bos_id] + tok.encode(text) + [tok.eos_id]
                        labels = [-100] + ids[1:]
                        # Buffer never exceeds one window; keep its final context token.
                        for token, label in zip(ids, labels):
                            pending_ids.append(token)
                            pending_labels.append(label)
                            if len(pending_ids) == seq_length:
                                write_window(pending_ids, pending_labels)
                                pending_ids, pending_labels = pending_ids[-1:], [-100]
                        continue
                    ids, labels = tok.encode_chat(
                        validate_messages(
                            record.get("messages"), locator, tools=record.get("tools")
                        ),
                        tools=record.get("tools"),
                    )
                    supervised = False
                    for start in range(0, len(ids) - 1, seq_length - 1):
                        x, y = ids[start : start + seq_length], labels[start : start + seq_length]
                        if not write_window(x, y):
                            exclusions[split]["no_shifted_supervision_windows"] += 1
                            excluded_file.write(
                                json.dumps(
                                    {
                                        "locator": locator,
                                        "token_offset": start,
                                        "reason": "window_has_no_shifted_supervision",
                                    }
                                )
                                + "\n"
                            )
                        else:
                            supervised = True
                    if not supervised:
                        exclusions[split]["no_supervision_documents"] += 1
                        excluded_file.write(
                            json.dumps(
                                {
                                    "locator": locator,
                                    "reason": "document_has_no_shifted_supervision",
                                }
                            )
                            + "\n"
                        )
                if mode == "pretrain" and len(pending_ids) > 1:
                    write_window(pending_ids, pending_labels)
                if count != source_manifest["document_counts"][split]:
                    raise ValueError(f"{path}: document count mismatch")
                verify_unchanged(path, source_manifest["split_hashes"][split])
            for name, raw in (("input_ids", raw_ids), ("labels", raw_labels)):
                shape = (row_counts[split], seq_length)
                target = tmp / f"{split}.{name}.npy"
                if not shape[0]:
                    np.save(target, np.empty(shape, dtype=np.int64))
                else:
                    source = np.memmap(raw, dtype=np.int64, mode="r", shape=shape)
                    dest = np.lib.format.open_memmap(target, mode="w+", dtype=np.int64, shape=shape)
                    for start in range(0, shape[0], 4096):
                        dest[start : start + 4096] = source[start : start + 4096]
                    dest.flush()
                    del source, dest
                raw.unlink()
        if not row_counts["train"]:
            raise ValueError("no shifted training supervision")
        verify_unchanged(tokenizer_path, tokenizer_hash)
        verify_unchanged(prepared / "manifest.json", source_manifest_hash)
        if tokenizer_manifest_hash is not None:
            verify_unchanged(tokenizer_manifest_path, tokenizer_manifest_hash)
        for split, expected in source_manifest["split_hashes"].items():
            if split in ("train", "validation"):
                verify_unchanged(prepared / f"{split}.jsonl", expected)
        manifest = {
            "format_version": 1,
            "mode": mode,
            "seq_length": seq_length,
            "vocab_size": tok.vocab_size,
            "row_counts": row_counts,
            "exclusions": exclusions,
            "tokenizer_sha256": tokenizer_hash,
            "tokenizer_training_split_sha256": tokenizer_manifest.get("training_split_sha256"),
            "source_manifest_sha256": source_manifest_hash,
            "source_hashes": source_manifest["split_hashes"],
            "packing": (
                "cross-document BOS/EOS, overlap one token, final padded tail"
                if mode == "pretrain"
                else "per-conversation, overlap one token, padded tails"
            )
            + "; labels shifted by model only",
        }
        manifest["array_hashes"] = {f.name: sha256(f) for f in tmp.glob("*.npy")}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return Path(output_dir)


class PackedDataset(Dataset):
    """Read-only lazy arrays; each returned tensor owns its small writable row copy."""

    def __init__(self, data_dir: str | Path, split: str) -> None:
        if split not in ("train", "validation"):
            raise ValueError("invalid split")
        self.input_ids = np.load(Path(data_dir) / f"{split}.input_ids.npy", mmap_mode="r")
        self.labels = np.load(Path(data_dir) / f"{split}.labels.npy", mmap_mode="r")
        if self.input_ids.shape != self.labels.shape or self.input_ids.ndim != 2:
            raise ValueError("packed arrays must have matching two-dimensional shapes")
        if self.input_ids.dtype != np.int64 or self.labels.dtype != np.int64:
            raise ValueError("packed arrays must be int64")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.from_numpy(self.input_ids[index].copy()),
            "labels": torch.from_numpy(self.labels[index].copy()),
        }


def adapt_conversations(record: dict[str, Any]) -> dict[str, Any]:
    """Explicitly convert the `conversations` chat format to the native schema."""
    from .tools import validate_tools

    if not isinstance(record, dict) or set(record) - {"conversations", "tools"}:
        raise ValueError("conversations adapter expects conversations and optional tools")
    tools = validate_tools(record["tools"]) if "tools" in record else None
    result = {"messages": validate_messages(record.get("conversations"), tools=tools)}
    if tools is not None:
        result["tools"] = tools
    return result


def adapt_preference(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize preference pairs; both alternatives must share the same prompt."""
    from .tools import validate_tools

    if not isinstance(record, dict) or set(record) - {"prompt", "chosen", "rejected", "tools"}:
        raise ValueError("preference adapter expects chosen/rejected, optional prompt/tools")
    tools = validate_tools(record["tools"]) if "tools" in record else None
    if "prompt" in record:
        prompt = record["prompt"]
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        prompt = validate_messages(prompt, "prompt", tools=tools)
        from .tools import serialize_messages

        serialize_messages(prompt, tools=tools, add_generation_prompt=True)
        if any(not isinstance(record.get(k), str) for k in ("chosen", "rejected")):
            raise ValueError("prompt form requires chosen/rejected response strings")
        alternatives = {
            k: prompt + [{"role": "assistant", "content": record[k]}]
            for k in ("chosen", "rejected")
        }
    else:
        alternatives = {k: record.get(k) for k in ("chosen", "rejected")}
    result = {k: validate_messages(v, k, tools=tools) for k, v in alternatives.items()}
    chosen, rejected = result["chosen"], result["rejected"]
    if (
        any(v[-1]["role"] != "assistant" for v in (chosen, rejected))
        or chosen[:-1] != rejected[:-1]
    ):
        raise ValueError("preferences require a shared prompt and final assistant alternatives")
    if chosen[-1] == rejected[-1]:
        raise ValueError("preference alternatives must differ")
    if tools is not None:
        result["tools"] = tools
    return result


def adapt_rollout(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize generation-ready prompts and optional JSON ground truth."""
    from .tools import canonical_json, serialize_messages, validate_tools

    allowed = {"prompt", "messages", "tools", "expected", "gt"}
    if (
        not isinstance(record, dict)
        or set(record) - allowed
        or ("prompt" in record) == ("messages" in record)
    ):
        raise ValueError("rollout adapter expects exactly one of prompt/messages")
    if "expected" in record and "gt" in record:
        raise ValueError("specify expected or gt, not both")
    prompt = record.get("prompt", record.get("messages"))
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    tools = validate_tools(record["tools"]) if "tools" in record else None
    prompt = validate_messages(prompt, tools=tools)
    serialize_messages(prompt, tools=tools, add_generation_prompt=True)
    result = {"messages": prompt}
    if tools is not None:
        result["tools"] = tools
    if "expected" in record or "gt" in record:
        expected = record.get("expected", record.get("gt"))
        result["expected"] = json.loads(canonical_json(expected))
    return result


def convert_dataset(source: SourceSpec, output_dir: str | Path, *, adapter: str) -> Path:
    """Stream an explicit format conversion with source licensing and hash lineage."""
    adapters = {
        "conversations": adapt_conversations,
        "preference": adapt_preference,
        "rollout": adapt_rollout,
    }
    if adapter not in adapters:
        raise ValueError("adapter must be conversations, preference, or rollout")
    if (
        not isinstance(source, SourceSpec)
        or not isinstance(source.license, str)
        or not source.license.strip()
        or not isinstance(source.language, str)
        or not source.language.strip()
    ):
        raise ValueError("conversion requires source license and language declarations")
    path = Path(source.path)
    original_hash = sha256(path)
    with atomic_directory(output_dir) as tmp:
        count = 0
        with (tmp / "data.jsonl").open("w", encoding="utf-8") as output:
            for locator, record in records(path):
                try:
                    normalized = adapters[adapter](record)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{locator}: {exc}") from exc
                output.write(
                    json.dumps(normalized, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                )
                count += 1
        if not count:
            raise ValueError("source has no records")
        verify_unchanged(path, original_hash)
        (tmp / "manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "kind": "explicit_dataset_conversion",
                    "adapter": adapter,
                    "source": {
                        "path": str(path.resolve()),
                        "sha256": original_hash,
                        "license": source.license,
                        "language": source.language,
                    },
                    "records": count,
                    "output_sha256": sha256(tmp / "data.jsonl"),
                },
                indent=2,
            )
            + "\n"
        )
    return Path(output_dir) / "data.jsonl"
