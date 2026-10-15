import random

import numpy as np
import pytest
import torch

from clearforge.checkpoint import (
    atomic_save,
    capture_rng_state,
    publish_directory,
    publish_latest,
    restore_rng_state,
)


def test_atomic_failure_preserves_previous_and_cleans_temporary(tmp_path):
    path = tmp_path / "state.pt"
    atomic_save({"weight": torch.arange(9)}, path)
    before = path.read_bytes()
    with pytest.raises((AttributeError, TypeError)):
        atomic_save({"unserializable": lambda: None}, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_rng_roundtrip_all_generators():
    state = capture_rng_state()

    def draw():
        return (
            random.random(),
            np.random.random(),
            torch.rand(3),
            torch.rand(3, device="cuda").cpu() if torch.cuda.is_available() else torch.empty(0),
        )

    first = draw()
    restore_rng_state(state)
    second = draw()
    assert first[:2] == second[:2]
    assert all(torch.equal(a, b) for a, b in zip(first[2:], second[2:]))


def test_latest_is_atomic_hardlink_and_old_checkpoint_survives(tmp_path):
    first, second, latest = [tmp_path / x for x in ("one.pt", "two.pt", "latest.pt")]
    atomic_save({"step": 1}, first)
    publish_latest(first, latest)
    old = first.read_bytes()
    atomic_save({"step": 2}, second)
    publish_latest(second, latest)
    assert latest.stat().st_ino == second.stat().st_ino
    assert first.read_bytes() == old


def test_directory_publication_never_clobbers_empty_destination(tmp_path):
    source, dest = tmp_path / "source", tmp_path / "dest"
    source.mkdir()
    (source / "weights").write_text("complete")
    dest.mkdir()
    with pytest.raises(FileExistsError):
        publish_directory(source, dest)
    assert source.exists() and list(dest.iterdir()) == []
    dest.rmdir()
    publish_directory(source, dest)
    assert (dest / "weights").read_text() == "complete"
