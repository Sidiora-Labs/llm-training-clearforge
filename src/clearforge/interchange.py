"""Local, safe-weight interchange with the standard Transformers Llama decoder."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .checkpoint import atomic_save, atomic_write_bytes, file_sha256
from .config import ModelConfig
from .data import atomic_directory
from .model import CausalLM
from .tokenizer import SPECIAL_TOKENS, ChatTokenizer

CHAT_TEMPLATE = r"""{%- if not messages %}{{ raise_exception('Messages must be nonempty') }}{% endif -%}
{%- set state = namespace(expected='user', pending=[], seen=[]) -%}
{%- set controls = ['<|pad|>', '<|bos|>', '<|eos|>', '<|system|>', '<|user|>', '<|assistant|>'] -%}
{{- bos_token -}}
{%- if tools and messages[0]['role'] != 'system' -%}
{%- set tool_text = '<clearforge_tools>\n' + (tools|tojson(ensure_ascii=false, sort_keys=true, separators=(',', ':'))) + '\n</clearforge_tools>' -%}
{%- for control in controls %}{% if control in tool_text %}{{ raise_exception('Literal reserved control tokens require the native tokenizer') }}{% endif %}{% endfor -%}
{{- '<|system|>' + tool_text + eos_token -}}
{%- endif -%}
{%- for message in messages -%}
{%- set role = message['role'] -%}
{%- if role not in ['system', 'user', 'assistant', 'tool'] %}{{ raise_exception('Unsupported message role') }}{% endif -%}
{%- if role == 'system' -%}
{%- if not loop.first %}{{ raise_exception('System message must be first') }}{% endif -%}
{%- elif role != state.expected %}{{ raise_exception('Unexpected conversation role') }}{% endif -%}
{%- set content = message.get('content') -%}
{%- if content is none and role == 'assistant' and message.get('tool_calls') %}{% set content = '' %}{% endif -%}
{%- if content is not string %}{{ raise_exception('Message content must be text') }}{% endif -%}
{%- if role == 'user' %}{% set state.expected = 'assistant' %}{% endif -%}
{%- if role == 'system' and tools %}{% set content = content + '\n\n<clearforge_tools>\n' + (tools|tojson(ensure_ascii=false, sort_keys=true, separators=(',', ':'))) + '\n</clearforge_tools>' %}{% endif -%}
{%- if role == 'assistant' -%}
{%- set payload = {'content': content} -%}
{%- if 'reasoning_content' in message -%}
{%- if message['reasoning_content'] is not string %}{{ raise_exception('Reasoning content must be text') }}{% endif -%}
{%- set payload = dict(payload, reasoning_content=message['reasoning_content']) -%}
{%- endif -%}
{%- if 'tool_calls' in message -%}
{%- if not message['tool_calls'] %}{{ raise_exception('Tool calls must be nonempty') }}{% endif -%}
{%- set calls = namespace(items=[]) -%}
{%- for call in message['tool_calls'] -%}
{%- if call['function']['arguments'] is not mapping and call['function']['arguments'] is not string %}{{ raise_exception('Tool arguments must be mappings or canonical JSON strings') }}{% endif -%}
{%- set arguments = call['function']['arguments'] if call['function']['arguments'] is string else call['function']['arguments']|tojson(ensure_ascii=false, sort_keys=true, separators=(',', ':')) -%}
{%- if call['id'] in state.seen %}{{ raise_exception('Duplicate tool call ID') }}{% endif -%}
{%- set state.seen = state.seen + [call['id']] -%}
{%- set state.pending = state.pending + [call['id']] -%}
{%- set function = {'name': call['function']['name'], 'arguments': arguments} -%}
{%- set calls.items = calls.items + [{'id': call['id'], 'type': 'function', 'function': function}] -%}
{%- endfor -%}
{%- set payload = dict(payload, tool_calls=calls.items) -%}
{%- set state.expected = 'tool' -%}
{%- else %}{% set state.expected = 'user' %}{% endif -%}
{%- if 'tool_calls' in message or 'reasoning_content' in message %}{% set content = '<clearforge_assistant>\n' + (payload|tojson(ensure_ascii=false, sort_keys=true, separators=(',', ':'))) + '\n</clearforge_assistant>' %}{% endif -%}
{%- elif role == 'tool' -%}
{%- if message['tool_call_id'] not in state.pending %}{{ raise_exception('Unknown or duplicate tool response ID') }}{% endif -%}
{%- set state.pending = state.pending|reject('equalto', message['tool_call_id'])|list -%}
{%- if not state.pending %}{% set state.expected = 'assistant' %}{% endif -%}
{%- set content = '<clearforge_tool>\n' + ({'content': content, 'tool_call_id': message['tool_call_id']}|tojson(ensure_ascii=false, sort_keys=true, separators=(',', ':'))) + '\n</clearforge_tool>' -%}
{%- set role = 'user' -%}
{%- endif -%}
{%- for control in controls %}{% if control in content %}{{ raise_exception('Literal reserved control tokens require the native tokenizer') }}{% endif %}{% endfor -%}
{{- '<|' + role + '|>' + content + eos_token -}}
{%- endfor -%}
{%- if state.pending and messages[-1]['role'] != 'assistant' %}{{ raise_exception('Incomplete tool response batch') }}{% endif -%}
{%- if add_generation_prompt -%}
{%- if messages[-1]['role'] not in ['user', 'tool'] or state.pending %}{{ raise_exception('Generation prompt requires a user or completed tool responses') }}{% endif -%}
{{- '<|assistant|>' -}}
{%- endif -%}
"""


def _libraries():
    try:
        from safetensors.torch import save_file
        from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast
    except ImportError as error:
        raise ImportError(
            "Transformers interchange requires transformers and safetensors"
        ) from error
    return save_file, AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _permutation(weight: Tensor, heads: int, *, inverse: bool = False) -> Tensor:
    """Convert interleaved rotary projection rows to split-half rows, or back."""
    rows, width = weight.shape
    dimension = rows // heads
    if inverse:
        return weight.reshape(heads, 2, dimension // 2, width).transpose(1, 2).reshape(rows, width)
    return weight.reshape(heads, dimension // 2, 2, width).transpose(1, 2).reshape(rows, width)


def _mapping(config: ModelConfig) -> dict[str, str]:
    names = {
        "token_embedding.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    for i in range(config.num_layers):
        native, hf = f"layers.{i}.", f"model.layers.{i}."
        names[native + "attention_norm.weight"] = hf + "input_layernorm.weight"
        names[native + "mlp_norm.weight"] = hf + "post_attention_layernorm.weight"
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            target = "o_proj" if projection == "out_proj" else projection
            names[native + f"attention.{projection}.weight"] = hf + f"self_attn.{target}.weight"
        for projection in ("gate_proj", "up_proj", "down_proj"):
            names[native + f"mlp.{projection}.weight"] = hf + f"mlp.{projection}.weight"
    return names


def _dense(config: ModelConfig) -> None:
    if config.num_experts or config.yarn_factor != 1:
        raise ValueError("standard Llama interchange supports dense models with unscaled RoPE only")


def export_transformers(
    model: CausalLM, tokenizer_path: str | Path, output_dir: str | Path
) -> Path:
    """Publish safetensors and a standard Llama config without custom model code."""
    save_file, _, LlamaConfig, TokenizerFast = _libraries()
    config = model.config
    _dense(config)
    tokenizer_path = Path(tokenizer_path)
    tokenizer = ChatTokenizer.load(tokenizer_path)
    tokenizer_hash = file_sha256(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocabulary differs from model")
    source = model.state_dict()
    if any(not torch.isfinite(value).all().item() for value in source.values()):
        raise ValueError("model contains nonfinite weights")
    if config.tie_embeddings and not torch.equal(
        source["token_embedding.weight"], source["lm_head.weight"]
    ):
        raise ValueError("inconsistent tied embedding weights")
    weights = {}
    for native, hf in _mapping(config).items():
        tensor = source[native].detach().cpu()
        if native.endswith("attention.q_proj.weight"):
            tensor = _permutation(tensor, config.num_heads)
        elif native.endswith("attention.k_proj.weight"):
            tensor = _permutation(tensor, config.num_kv_heads)
        # Independent storage allows safe serialization of tied names.
        weights[hf] = tensor.contiguous().clone()
    hf_config = LlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
        num_key_value_heads=config.num_kv_heads,
        max_position_embeddings=config.max_seq_len,
        rope_theta=config.rope_theta,
        rms_norm_eps=1e-6,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        attention_dropout=config.dropout,
        tie_word_embeddings=config.tie_embeddings,
        bos_token_id=tokenizer.bos_id,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
    )
    hf_config.architectures = ["LlamaForCausalLM"]
    hf_config.clearforge_model_config = dataclasses.asdict(config)
    with atomic_directory(output_dir) as staging:
        hf_config.save_pretrained(staging)
        save_file(weights, staging / "model.safetensors", metadata={"format": "pt"})
        fast = TokenizerFast(
            tokenizer_file=str(tokenizer_path),
            bos_token=SPECIAL_TOKENS[1],
            eos_token=SPECIAL_TOKENS[2],
            pad_token=SPECIAL_TOKENS[0],
            additional_special_tokens=SPECIAL_TOKENS[3:],
            model_max_length=config.max_seq_len,
            clean_up_tokenization_spaces=False,
        )
        fast.chat_template = CHAT_TEMPLATE
        fast.save_pretrained(staging)
        if file_sha256(tokenizer_path) != tokenizer_hash:
            raise ValueError("tokenizer changed during export")
        atomic_write_bytes(
            _json(
                {
                    "format": "clearforge-transformers-v1",
                    "tokenizer_source_sha256": tokenizer_hash,
                    "files": {
                        path.name: file_sha256(path) for path in staging.iterdir() if path.is_file()
                    },
                }
            ),
            staging / "interchange.json",
        )
    return Path(output_dir)


def _native_config(hf: Any) -> ModelConfig:
    if hf.model_type != "llama":
        raise ValueError("only standard Llama checkpoints are supported")
    if getattr(hf, "quantization_config", None) is not None:
        raise ValueError("quantized Transformers checkpoints are unsupported")
    if (
        hf.hidden_act != "silu"
        or hf.rms_norm_eps != 1e-6
        or hf.attention_bias
        or hf.mlp_bias
        or hf.rope_scaling is not None
        or hf.head_dim != hf.hidden_size // hf.num_attention_heads
        or getattr(hf, "pretraining_tp", 1) != 1
    ):
        raise ValueError("unsupported Llama activation, normalization, biases, RoPE or head layout")
    fields = dict(
        vocab_size=hf.vocab_size,
        hidden_size=hf.hidden_size,
        num_layers=hf.num_hidden_layers,
        num_heads=hf.num_attention_heads,
        num_kv_heads=hf.num_key_value_heads,
        intermediate_size=hf.intermediate_size,
        max_seq_len=hf.max_position_embeddings,
        rope_theta=hf.rope_theta,
        dropout=hf.attention_dropout,
        tie_embeddings=hf.tie_word_embeddings,
    )
    config = ModelConfig(**fields)
    recorded = getattr(hf, "clearforge_model_config", None)
    if recorded is not None:
        original = ModelConfig(**recorded)
        if any(getattr(original, key) != value for key, value in fields.items()):
            raise ValueError("recorded native configuration conflicts with Llama configuration")
        config = original
    _dense(config)
    return config


def import_transformers(directory: str | Path, output_dir: str | Path) -> Path:
    """Import a local standard Llama safetensors directory into a native bundle.

    The tokenizer must implement Clearforge's control vocabulary. No network,
    remote Python code, or pickle weights are loaded.
    """
    _, AutoModel, LlamaConfig, _ = _libraries()
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("import requires an existing local model directory")
    document = json.loads((directory / "config.json").read_text())
    if document.get("model_type") != "llama":
        raise ValueError("only standard Llama checkpoints are supported")
    hf_config = LlamaConfig.from_dict(document)
    config = _native_config(hf_config)
    tokenizer_path = directory / "tokenizer.json"
    tokenizer = ChatTokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("tokenizer vocabulary differs from model")
    if (
        hf_config.bos_token_id != tokenizer.bos_id
        or hf_config.eos_token_id != tokenizer.eos_id
        or hf_config.pad_token_id != tokenizer.pad_id
    ):
        raise ValueError("Llama control token IDs differ from native tokenizer")
    # Snapshot identities before decoding and reject concurrent replacements.
    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and (
            path.suffix == ".safetensors"
            or path.name in {"config.json", "tokenizer.json", "model.safetensors.index.json"}
        )
    ]
    identities = {path.name: file_sha256(path) for path in files}
    index_path = directory / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for name in index["weight_map"].values():
            if Path(name).name != name or not name.endswith(".safetensors"):
                raise ValueError("unsafe safetensors shard path")
    from safetensors import safe_open

    # Check tied names before Transformers aliases the parameters and could mask
    # inconsistent duplicated tensors in the source archive.
    tied = {}
    for path in files:
        if path.suffix == ".safetensors":
            with safe_open(path, framework="pt", device="cpu") as archive:
                for name in ("model.embed_tokens.weight", "lm_head.weight"):
                    if name in archive.keys():
                        tied[name] = archive.get_tensor(name)
    if (
        config.tie_embeddings
        and len(tied) == 2
        and not torch.equal(tied["model.embed_tokens.weight"], tied["lm_head.weight"])
    ):
        raise ValueError("inconsistent tied embedding weights")
    hf_model, information = AutoModel.from_pretrained(
        directory,
        config=hf_config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
        output_loading_info=True,
    )
    if any(
        information.get(name)
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("checkpoint has missing, unexpected, or mismatched model weights")
    source = hf_model.state_dict()
    weights = {}
    for native, hf in _mapping(config).items():
        tensor = source[hf].detach()
        if native.endswith("attention.q_proj.weight"):
            tensor = _permutation(tensor, config.num_heads, inverse=True)
        elif native.endswith("attention.k_proj.weight"):
            tensor = _permutation(tensor, config.num_kv_heads, inverse=True)
        weights[native] = tensor.contiguous().clone()
    if any(not torch.isfinite(value).all().item() for value in weights.values()):
        raise ValueError("checkpoint contains nonfinite model weights")
    if config.tie_embeddings and not torch.equal(
        weights["token_embedding.weight"], weights["lm_head.weight"]
    ):
        raise ValueError("inconsistent tied embedding weights")
    if any(file_sha256(directory / name) != digest for name, digest in identities.items()):
        raise ValueError("input checkpoint changed during import")
    with atomic_directory(output_dir) as staging:
        atomic_save(weights, staging / "model.pt")
        atomic_write_bytes(_json(dataclasses.asdict(config)), staging / "model.json")
        atomic_write_bytes(tokenizer_path.read_bytes(), staging / "tokenizer.json")
        provenance = {
            "format": "clearforge-native-v1",
            "source_format": "transformers-llama",
            "source_files": identities,
            "files": {
                name: file_sha256(staging / name)
                for name in ("model.pt", "model.json", "tokenizer.json")
            },
        }
        atomic_write_bytes(_json(provenance), staging / "provenance.json")
    return Path(output_dir)


def export_bundle_transformers(directory: str | Path, output_dir: str | Path) -> Path:
    """Verify a native export bundle and convert it to standard Transformers files."""
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    names = {"model.pt", "model.json", "tokenizer.json"}
    if (
        provenance.get("format") != "clearforge-native-v1"
        or set(provenance.get("files", {})) != names
    ):
        raise ValueError("invalid native bundle manifest")
    identities = provenance["files"]
    if any(file_sha256(directory / name) != identities[name] for name in names):
        raise ValueError("native bundle integrity mismatch")
    config = ModelConfig(**json.loads((directory / "model.json").read_text()))
    _dense(config)
    weights = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    with torch.device("meta"):
        model = CausalLM(config)
    model.load_state_dict(weights, strict=True, assign=True)
    if config.tie_embeddings:
        if not torch.equal(model.token_embedding.weight, model.lm_head.weight):
            raise ValueError("inconsistent tied embedding weights")
        model.lm_head.weight = model.token_embedding.weight
    if any(file_sha256(directory / name) != identities[name] for name in names):
        raise ValueError("native bundle changed during loading")
    return export_transformers(model.eval(), directory / "tokenizer.json", output_dir)
