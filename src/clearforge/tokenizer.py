"""Byte-level BPE with explicit chat boundaries and literal-safe content encoding."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .data import atomic_directory, records, sha256, validate_messages

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]


class ChatTokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._special = {s: tokenizer.token_to_id(s) for s in SPECIAL_TOKENS}
        if any(v is None for v in self._special.values()):
            raise ValueError("tokenizer lacks required control tokens")
        # Added-token matching must never interpret untrusted content as control syntax.
        document = json.loads(tokenizer.to_str())
        document["added_tokens"] = []
        document["post_processor"] = None
        self._content = Tokenizer.from_str(json.dumps(document))
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_id, self.bos_id, self.eos_id = [self._special[s] for s in SPECIAL_TOKENS[:3]]

    @classmethod
    def load(cls, path: str | Path) -> ChatTokenizer:
        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        ids = self._content.encode(text, add_special_tokens=False).ids
        if set(ids).intersection(self._special.values()):
            raise ValueError("content encoded as a reserved control token")
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(ids), skip_special_tokens=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".tokenizer.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(self._tokenizer.to_str())
                f.flush()
                os.fsync(f.fileno())
            # Atomic no-clobber publication, including concurrent writers.
            os.link(temporary, path)
        finally:
            os.unlink(temporary)
        return path

    def encode_chat(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool = False
    ) -> tuple[list[int], list[int]]:
        messages = validate_messages(messages)
        if add_generation_prompt and messages[-1]["role"] != "user":
            raise ValueError("generation prompt requires a final user message")
        ids, labels = [self.bos_id], [-100]
        for message in messages:
            role = message["role"]
            content = self.encode(message["content"])
            ids.extend([self._special[f"<|{role}|>"]] + content + [self.eos_id])
            labels.extend(
                [-100]
                + (content + [self.eos_id] if role == "assistant" else [-100] * (len(content) + 1))
            )
        if add_generation_prompt:
            ids.append(self._special["<|assistant|>"])
            labels.append(-100)
        return ids, labels


def train_tokenizer(
    train_path: str | Path, output_dir: str | Path, *, vocab_size: int, min_frequency: int = 2
) -> Path:
    path = Path(train_path)
    if path.name != "train.jsonl":
        raise ValueError("tokenizer input must be the prepared train.jsonl split")
    manifest_path = path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(path) != manifest["split_hashes"]["train"]:
        raise ValueError("training split hash mismatch")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size < 262:
        raise ValueError("vocab_size must accommodate 256 bytes and six control tokens")
    if isinstance(min_frequency, bool) or not isinstance(min_frequency, int) or min_frequency < 1:
        raise ValueError("min_frequency must be positive")

    def texts():
        for locator, record in records(path):
            if manifest["mode"] == "pretrain":
                yield record["text"]
            else:
                for message in validate_messages(record.get("messages"), locator):
                    yield message["content"]

    with atomic_directory(output_dir) as tmp:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        tokenizer.train_from_iterator(texts(), trainer=trainer)
        if sha256(path) != manifest["split_hashes"]["train"]:
            raise ValueError("training split changed during tokenizer training")
        tokenizer.save(str(tmp / "tokenizer.json"))
        (tmp / "manifest.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "training_split_sha256": sha256(path),
                    "tokenizer_sha256": sha256(tmp / "tokenizer.json"),
                    "source_manifest_sha256": sha256(manifest_path),
                    "vocab_size": tokenizer.get_vocab_size(),
                    "requested_vocab_size": vocab_size,
                    "min_frequency": min_frequency,
                    "special_tokens": SPECIAL_TOKENS,
                },
                indent=2,
            )
            + "\n"
        )
    return Path(output_dir) / "tokenizer.json"
