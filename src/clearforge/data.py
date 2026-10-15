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


@contextmanager
def atomic_directory(output_dir: str | Path) -> Iterator[Path]:
    """Publish a new directory only; an exclusive sibling lock prevents competing writers."""
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
        os.rename(tmp, output)
        tmp = None
    finally:
        os.close(fd)
        lock.unlink()
        if tmp is not None:
            shutil.rmtree(tmp)


def validate_messages(messages: object, locator: str = "messages") -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{locator}: messages must be a nonempty list")
    clean = []
    for i, message in enumerate(messages):
        where = f"{locator}[{i}]"
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError(f"{where}: expected exactly role and content")
        role, content = message["role"], message["content"]
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            raise ValueError(f"{where}: invalid role or non-string content")
        if role == "system" and i != 0:
            raise ValueError(f"{where}: system role must be first")
        if i and role == messages[i - 1]["role"]:
            raise ValueError(f"{where}: consecutive identical roles")
        if role == "assistant" and (i == 0 or messages[i - 1]["role"] != "user"):
            raise ValueError(f"{where}: assistant must follow user")
        clean.append({"role": role, "content": content})
    return clean


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
                        body = {"messages": validate_messages(record.get("messages"), locator)}
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
    source_manifest = json.loads((prepared / "manifest.json").read_text())
    if source_manifest["mode"] != mode:
        raise ValueError("prepared mode mismatch")
    tokenizer_manifest_path = Path(tokenizer_path).parent / "manifest.json"
    tokenizer_manifest = (
        json.loads(tokenizer_manifest_path.read_text()) if tokenizer_manifest_path.exists() else {}
    )
    expected_tokenizer_hash = tokenizer_manifest.get("tokenizer_sha256")
    if expected_tokenizer_hash is not None and sha256(tokenizer_path) != expected_tokenizer_hash:
        raise ValueError("tokenizer hash mismatch")
    tok = ChatTokenizer.load(tokenizer_path)
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
                for locator, record in records(path):
                    if mode == "pretrain":
                        ids = [tok.bos_id] + tok.encode(record["text"]) + [tok.eos_id]
                        labels = ids.copy()
                        labels[0] = -100
                    else:
                        ids, labels = tok.encode_chat(
                            validate_messages(record.get("messages"), locator)
                        )
                    supervised = False
                    # One-token overlap supplies the preceding token for every causal target.
                    for start in range(0, len(ids) - 1, seq_length - 1):
                        x, y = ids[start : start + seq_length], labels[start : start + seq_length]
                        y[0] = -100
                        if not any(label != -100 for label in y[1:]):
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
                            continue
                        supervised = True
                        padding = seq_length - len(x)
                        np.asarray(x + [tok.pad_id] * padding, dtype=np.int64).tofile(ids_file)
                        np.asarray(y + [-100] * padding, dtype=np.int64).tofile(labels_file)
                        row_counts[split] += 1
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
        manifest = {
            "format_version": 1,
            "mode": mode,
            "seq_length": seq_length,
            "vocab_size": tok.vocab_size,
            "row_counts": row_counts,
            "exclusions": exclusions,
            "tokenizer_sha256": sha256(tokenizer_path),
            "tokenizer_training_split_sha256": tokenizer_manifest.get("training_split_sha256"),
            "source_manifest_sha256": sha256(prepared / "manifest.json"),
            "source_hashes": source_manifest["split_hashes"],
            "packing": "per-document, overlap one token, padded tails; labels shifted by model only",
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
