"""Durable checkpoints for trusted local runs.

Checkpoints contain Python and NumPy RNG state and therefore use pickle. Never
load a checkpoint from an untrusted source. Publication uses a same-filesystem
atomic replacement; an interrupted serialization cannot destroy the old file.
"""

from __future__ import annotations

import hashlib
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

FORMAT_VERSION = 1


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading its full contents into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG used by training, including every visible CUDA device."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore trusted state; reject a changed CUDA topology rather than guessing."""
    cuda_states = state["cuda"]
    if len(cuda_states) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("checkpoint CUDA device count does not match this process")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if cuda_states:
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_save(payload: dict[str, Any], path: str | Path) -> Path:
    """Serialize, fsync, then atomically publish a complete checkpoint.

    Parent directories must already exist. Before the atomic rename, failure
    leaves an existing destination untouched and removes the temporary file.
    """
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def publish_latest(checkpoint: str | Path, latest: str | Path) -> None:
    """Atomically point a regular hardlink at an immutable step checkpoint."""
    checkpoint, latest = Path(checkpoint), Path(latest)
    descriptor, temporary = tempfile.mkstemp(prefix=".latest.", dir=latest.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    temporary_path.unlink()
    try:
        os.link(checkpoint, temporary_path)
        os.replace(temporary_path, latest)
        _sync_directory(latest.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a **trusted** local checkpoint on CPU; this executes pickle decoding."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {
        "format_version", "run_id", "config", "identities", "model", "optimizer",
        "scheduler", "scaler", "step", "rng", "data_cursor", "metrics",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("not a complete Clearforge training checkpoint")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format version")
    if type(payload["step"]) is not int or payload["step"] < 0:
        raise ValueError("invalid checkpoint optimizer step")
    return payload
