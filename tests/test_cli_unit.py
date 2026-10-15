"""CLI boundary and export-integrity tests with real tokenizers and weights."""

import dataclasses
import hashlib
import json

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from clearforge.cli import _device, _load_bundle, main
from clearforge.config import ModelConfig
from clearforge.model import CausalLM
from clearforge.tokenizer import SPECIAL_TOKENS


@pytest.fixture
def bundle(tmp_path):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["Hello world. This is a test."],
        trainers.BpeTrainer(
            vocab_size=280,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        ),
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    config = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=32,
        max_seq_len=64,
    )
    (tmp_path / "model.json").write_text(json.dumps(dataclasses.asdict(config)))
    torch.save(CausalLM(config).state_dict(), tmp_path / "model.pt")
    _manifest(tmp_path)
    return tmp_path


def _manifest(path):
    hashes = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in ("model.pt", "model.json", "tokenizer.json")
    }
    (path / "provenance.json").write_text(
        json.dumps(
            {
                "format": "clearforge-native-v1",
                "files": hashes,
            }
        )
    )


def test_bundle_loads_real_weights(bundle):
    model, tokenizer = _load_bundle(bundle, torch.device("cpu"))
    assert model.config.vocab_size == tokenizer.vocab_size
    assert model.lm_head.weight is model.token_embedding.weight
    assert not model.training
    assert torch.isfinite(model(torch.tensor([[tokenizer.bos_id, 10]])).logits).all()


@pytest.mark.parametrize("filename", ["model.pt", "model.json", "tokenizer.json"])
def test_bundle_tamper_rejected(bundle, filename):
    with (bundle / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="integrity"):
        _load_bundle(bundle, torch.device("cpu"))


def test_bundle_rejects_unlisted_or_traversal_files(bundle):
    path = bundle / "provenance.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["../elsewhere"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly"):
        _load_bundle(bundle, torch.device("cpu"))


def test_nonfinite_weights_rejected(bundle):
    path = bundle / "model.pt"
    weights = torch.load(path, weights_only=True)
    next(iter(weights.values())).fill_(float("nan"))
    torch.save(weights, path)
    _manifest(bundle)
    with pytest.raises(ValueError, match="nonfinite"):
        _load_bundle(bundle, torch.device("cpu"))


def test_unknown_bundle_version_rejected(bundle):
    (bundle / "provenance.json").write_text('{"format":"future"}')
    with pytest.raises(ValueError, match="unsupported"):
        _load_bundle(bundle, torch.device("cpu"))


def test_missing_bundle_gives_concise_error(tmp_path, capsys):
    assert main(["chat", "--model", str(tmp_path), "--prompt", "Hello"]) == 1
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("value", ["mps", "cuda:-1", "cuda:no", "cpu:0"])
def test_device_rejects_unsupported_values(value):
    with pytest.raises(ValueError, match="device"):
        _device(value)


def test_help_does_not_require_training_module(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    assert "doctor" in capsys.readouterr().out


def test_threads_are_positive():
    with pytest.raises(SystemExit) as error:
        main(["--threads", "0", "doctor"])
    assert error.value.code == 2


def test_inconsistent_tied_weights_rejected(bundle):
    path = bundle / "model.pt"
    weights = torch.load(path, weights_only=True)
    weights["lm_head.weight"] = weights["lm_head.weight"].clone() + 1
    torch.save(weights, path)
    _manifest(bundle)
    with pytest.raises(ValueError, match="inconsistent tied"):
        _load_bundle(bundle, torch.device("cpu"))
