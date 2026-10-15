import copy
import json

import pytest
import torch

from clearforge.adapters import (
    AdapterConfig,
    LoRALinear,
    apply_adapters,
    load_adapters,
    merge_adapters,
    save_adapters,
)
from clearforge.checkpoint import file_sha256
from clearforge.config import ModelConfig
from clearforge.model import CausalLM


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    return CausalLM(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            intermediate_size=32,
            max_seq_len=12,
        )
    ).eval()


def test_initial_identity_frozen_base_and_merge(model):
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    expected = model(ids).logits.detach()
    base = copy.deepcopy(model.state_dict())
    apply_adapters(model, AdapterConfig(rank=2, alpha=4))
    torch.testing.assert_close(model(ids).logits, expected, rtol=0, atol=0)
    assert all(
        p.requires_grad == name.endswith(("lora_a", "lora_b"))
        for name, p in model.named_parameters()
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.05)
    for _ in range(3):
        optimizer.zero_grad()
        model(ids, ids).loss.backward()
        optimizer.step()
    updated = model(ids).logits.detach()
    assert not torch.equal(updated, expected)
    for name, parameter in model.state_dict().items():
        if not name.endswith(("lora_a", "lora_b")):
            original = name.replace(".base.", ".")
            assert torch.equal(parameter, base[original])
    merge_adapters(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    torch.testing.assert_close(model(ids).logits, updated, rtol=1e-5, atol=1e-6)
    assert model.lm_head.weight is model.token_embedding.weight


def test_adapter_roundtrip_and_base_identity(model, tmp_path):
    plain = copy.deepcopy(model)
    apply_adapters(model, AdapterConfig(rank=2))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("lora_b"):
                p.fill_(0.1)
    save_adapters(model, tmp_path / "adapter")
    load_adapters(plain, tmp_path / "adapter")
    ids = torch.tensor([[1, 2, 3]])
    torch.testing.assert_close(model(ids).logits, plain(ids).logits, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_adapters(model, tmp_path / "adapter")
    other = copy.deepcopy(model)
    merge_adapters(other)
    with pytest.raises(ValueError, match="base weights"):
        load_adapters(other, tmp_path / "adapter")
    assert not any(isinstance(m, LoRALinear) for m in other.modules())


def test_invalid_adapter_preflight_does_not_mutate(model, tmp_path):
    pristine = copy.deepcopy(model)
    apply_adapters(model, AdapterConfig(rank=2))
    path = save_adapters(model, tmp_path / "adapter")
    values = torch.load(path / "adapter.pt", weights_only=True)
    values[next(iter(values))] = torch.zeros(1)
    torch.save(values, path / "adapter.pt")
    metadata = json.loads((path / "adapter.json").read_text())
    metadata["weights_sha256"] = file_sha256(path / "adapter.pt")
    (path / "adapter.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="invalid adapter tensor"):
        load_adapters(pristine, path)
    assert not any(isinstance(m, LoRALinear) for m in pristine.modules())
    assert all(p.requires_grad for p in pristine.parameters())


def test_bad_targets_and_repeated_injection(model):
    with pytest.raises(ValueError, match="targets"):
        apply_adapters(model, AdapterConfig(targets=("absent",)))
    assert all(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="dimension"):
        apply_adapters(model, AdapterConfig(rank=100))
    apply_adapters(model, AdapterConfig(rank=2))
    with pytest.raises(ValueError, match="already"):
        apply_adapters(model, AdapterConfig(rank=2))
