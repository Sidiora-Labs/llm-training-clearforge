"""Configuration errors must fail before data or accelerator allocation."""

import pytest

from clearforge.config import ModelConfig, TrainConfig, load_config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_size": 17},
        {"num_kv_heads": 5},
        {"num_heads": True},
        {"vocab_size": 0},
        {"max_seq_len": 1},
        {"dropout": 1.0},
        {"rope_theta": float("nan")},
        {"tie_embeddings": "yes"},
        {"hidden_size": 36, "num_heads": 12},
    ],
)
def test_reject_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"accumulation_steps": False},
        {"learning_rate": float("inf")},
        {"precision": "half"},
        {"device": "cuda:bad"},
        {"warmup_steps": 10001},
        {"learning_rate": 0},
        {"min_lr_ratio": -0.1},
        {"seed": -1},
    ],
)
def test_reject_invalid_training(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_paths_are_relative_to_config_not_process(tmp_path, monkeypatch):
    folder = tmp_path / "configs"
    folder.mkdir()
    config = folder / "run.toml"
    config.write_text('data_dir="../data"\ntokenizer_path="../tokens"\noutput_dir="../run"\n')
    monkeypatch.chdir("/")
    parsed = load_config(config)
    assert parsed.data_dir == tmp_path / "data"
    assert parsed.to_dict()["output_dir"] == str(tmp_path / "run")


@pytest.mark.parametrize("extra", ["typo=1", "[model]\nheadz=8", "[training]\nbatch_siz=4"])
def test_unknown_keys_fail(tmp_path, extra):
    config = tmp_path / "run.toml"
    config.write_text('data_dir="data"\ntokenizer_path="tokens"\noutput_dir="run"\n' + extra)
    with pytest.raises(ValueError, match="unknown"):
        load_config(config)


def test_missing_paths_fail(tmp_path):
    config = tmp_path / "run.toml"
    config.write_text("")
    with pytest.raises(ValueError, match="missing"):
        load_config(config)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_checkpoint_retention_requires_positive_integer(value):
    with pytest.raises(ValueError, match="keep_checkpoints"):
        TrainConfig(keep_checkpoints=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_experts=-1),
        dict(num_experts=1, experts_per_token=2),
        dict(experts_per_token=0),
        dict(router_aux_loss_coef=-1),
        dict(yarn_factor=0.5),
        dict(yarn_beta_fast=0.1),
        dict(yarn_original_context=1),
        dict(yarn_factor=2, rope_theta=1),
    ],
)
def test_invalid_architecture_controls(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_legacy_model_defaults_disable_extensions():
    model = ModelConfig(
        **{"vocab_size": 512, "hidden_size": 32, "num_heads": 4, "num_kv_heads": 2, "num_layers": 2}
    )
    assert model.num_experts == 0
    assert model.yarn_factor == 1.0
