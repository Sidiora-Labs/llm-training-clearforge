"""Causal decoder with cached inference, sparse experts, and rotary scaling."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from clearforge.config import ModelConfig


@dataclass
class CausalLMOutput:
    """Full token logits and optional mean loss over supervised next tokens."""

    logits: Tensor
    loss: Tensor | None = None
    past_key_values: tuple[tuple[Tensor, Tensor], ...] | None = None
    aux_loss: Tensor | None = None


class RMSNorm(nn.Module):
    """Root-mean-square normalization accumulated in float32."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """Interleaved rotary coordinates, calculated in float32 on the input device."""

    def __init__(self, head_dim: int, theta: float, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self.config = config

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        positions = torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=torch.float32)
        frequencies = self.theta ** (
            -torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim
        )
        scale = 1.0
        if self.config is not None and self.config.yarn_factor > 1:
            c = self.config

            def boundary(rotations: float) -> float:
                return (
                    self.head_dim
                    * math.log(c.yarn_original_context / (rotations * 2 * math.pi))
                    / (2 * math.log(self.theta))
                )

            low = max(0, math.floor(boundary(c.yarn_beta_fast)))
            high = min(self.head_dim // 2 - 1, math.ceil(boundary(c.yarn_beta_slow)))
            ramp = (
                (torch.arange(self.head_dim // 2, device=x.device) - low) / max(high - low, 0.001)
            ).clamp(0, 1)
            frequencies = frequencies * (1 - ramp + ramp / c.yarn_factor)
            scale = 1 + 0.1 * math.log(c.yarn_factor)
        angles = positions[:, None] * frequencies[None, :]
        cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        cosine, sine = cosine * scale, sine * scale
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)


class CausalAttention(nn.Module):
    """Grouped-query attention delegated to PyTorch's causal SDPA kernels."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.dropout = config.dropout
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta, config)

    def forward(self, x: Tensor, past_key_value=None, use_cache=False):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        offset = 0 if past_key_value is None else past_key_value[0].shape[2]
        q, k = self.rotary(q, offset), self.rotary(k, offset)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)
        mask = None
        if offset:
            mask = (
                torch.arange(k.shape[2], device=x.device)[None, :]
                <= torch.arange(offset, offset + length, device=x.device)[:, None]
            )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=not offset,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        output = self.out_proj(attended.transpose(1, 2).contiguous().view(batch, length, -1))
        return (output, (k, v)) if use_cache else output


class SwiGLU(nn.Module):
    """Gated feed-forward expansion with independent gate and value projections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseExperts(nn.Module):
    """Top-k routing with normalized expert weights and load-balancing loss."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(config) for _ in range(config.num_experts))

    def _route_probabilities(self, x: Tensor) -> Tensor:
        # Ranking experts is discontinuous: retain projection precision, not just
        # softmax precision. Casts also preserve gradients for half-stored weights.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.router.weight.float()).softmax(-1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        probabilities = self._route_probabilities(flat)
        weights, indices = probabilities.topk(self.config.experts_per_token, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        output = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == index)
            if tokens.numel():
                values = expert(flat[tokens]) * weights[tokens, slots, None].to(flat.dtype)
                output = output.index_add(0, tokens, values)
        load = F.one_hot(indices, self.config.num_experts).float().mean((0, 1))
        auxiliary = self.config.num_experts * (load * probabilities.mean(0)).sum()
        auxiliary = auxiliary * self.config.router_aux_loss_coef
        return output.view_as(x), auxiliary


class DecoderBlock(nn.Module):
    """Attention and feed-forward residuals with pre-normalization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SparseExperts(config) if config.num_experts else SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, past_key_value=None, use_cache=False):
        attention = self.attention(self.attention_norm(x), past_key_value, use_cache)
        attended, present = attention if use_cache else (attention, None)
        x = x + self.dropout(attended)
        feedforward = self.mlp(self.mlp_norm(x))
        values, auxiliary = (
            feedforward if isinstance(self.mlp, SparseExperts) else (feedforward, None)
        )
        return x + self.dropout(values), present, auxiliary


class CausalLM(nn.Module):
    """Decoder-only language model with aligned labels shifted exactly once."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if not isinstance(config, ModelConfig):
            raise ValueError("config must be a validated ModelConfig")
        config.__post_init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(DecoderBlock(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        # named_parameters deduplicates tied weights: each parameter is initialized once.
        for name, parameter in self.named_parameters():
            if parameter.ndim >= 2:
                residual = name.endswith(("attention.out_proj.weight", "down_proj.weight"))
                std = 0.02 / math.sqrt(2 * config.num_layers) if residual else 0.02
                nn.init.normal_(parameter, mean=0.0, std=std)

    def _validate_ids(self, input_ids: Tensor) -> None:
        if not isinstance(input_ids, Tensor) or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a torch.long tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty [batch, sequence] shape")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("input sequence exceeds max_seq_len")
        if input_ids.device != self.token_embedding.weight.device:
            raise ValueError("input_ids and model must be on the same device")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contains a token outside the vocabulary")

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        *,
        past_key_values=None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
    ) -> CausalLMOutput:
        self._validate_ids(input_ids)
        if type(use_cache) is not bool:
            raise ValueError("use_cache must be boolean")
        if type(logits_to_keep) is not int or not 0 <= logits_to_keep <= input_ids.shape[1]:
            raise ValueError("logits_to_keep must be between zero and input length")
        offset = 0
        if past_key_values is not None:
            if not isinstance(past_key_values, (tuple, list)) or len(past_key_values) != len(
                self.layers
            ):
                raise ValueError("cache must contain one key/value pair per layer")
            for pair in past_key_values:
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise ValueError("cache entries must be key/value pairs")
                k, v = pair
                if not all(isinstance(t, Tensor) and t.ndim == 4 for t in pair):
                    raise ValueError("cache tensors must have four dimensions")
                expected = (
                    input_ids.shape[0],
                    self.config.num_kv_heads,
                    k.shape[2],
                    self.config.hidden_size // self.config.num_heads,
                )
                if k.shape != expected or v.shape != expected or k.shape[2] < 1:
                    raise ValueError("cache tensor shape mismatch")
                expected_dtype = (
                    torch.get_autocast_dtype(input_ids.device.type)
                    if torch.is_autocast_enabled(input_ids.device.type)
                    else self.token_embedding.weight.dtype
                )
                if any(t.device != input_ids.device or t.dtype != expected_dtype for t in pair):
                    raise ValueError("cache device or dtype mismatch")
                if offset and offset != k.shape[2]:
                    raise ValueError("cache layers must have equal lengths")
                offset = k.shape[2]
        if offset + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("cached sequence exceeds max_seq_len")
        if labels is not None and (past_key_values is not None or logits_to_keep):
            raise ValueError("labels require full logits and no past cache")
        if labels is not None:
            if not isinstance(labels, Tensor) or labels.dtype != torch.long:
                raise ValueError("labels must be a torch.long tensor")
            if labels.shape != input_ids.shape or labels.device != input_ids.device:
                raise ValueError("labels must match input_ids shape and device")
            valid = labels != -100
            if torch.any(valid & ((labels < 0) | (labels >= self.config.vocab_size))):
                raise ValueError("labels must contain vocabulary IDs or -100")
            if not valid[:, 1:].any():
                raise ValueError("labels contain no supervised next tokens after shifting")
        x = self.embedding_dropout(self.token_embedding(input_ids))
        presents, auxiliary = [], None
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            x, present, aux = layer(x, past, use_cache)
            if use_cache:
                presents.append(present)
            if aux is not None:
                auxiliary = aux if auxiliary is None else auxiliary + aux
        hidden = self.norm(x)
        logits = self.lm_head(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        if loss is not None and auxiliary is not None:
            loss = loss + auxiliary
        return CausalLMOutput(logits, loss, tuple(presents) if use_cache else None, auxiliary)

    def parameter_count(self) -> int:
        """Count unique parameters, including each tied matrix only once."""
        return sum(parameter.numel() for parameter in self.parameters())

    def _generation_args(
        self,
        input_ids,
        max_new_tokens,
        temperature,
        top_k,
        eos_token_id,
        top_p,
        repetition_penalty,
        num_return_sequences,
    ):
        self._validate_ids(input_ids)
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("prompt plus generation budget exceeds max_seq_len")
        for name, value in (
            ("temperature", temperature),
            ("top_p", top_p),
            ("repetition_penalty", repetition_penalty),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
        if temperature < 0:
            raise ValueError("temperature must be nonnegative")
        if not 0 < top_p <= 1 or repetition_penalty <= 0:
            raise ValueError("top_p must be in (0, 1] and repetition_penalty positive")
        if type(num_return_sequences) is not int or num_return_sequences < 1:
            raise ValueError("num_return_sequences must be a positive integer")
        if top_k is not None and (
            type(top_k) is not int or not 1 <= top_k <= self.config.vocab_size
        ):
            raise ValueError("top_k must be an integer in [1, vocab_size]")
        if eos_token_id is not None and (
            type(eos_token_id) is not int or not 0 <= eos_token_id < self.config.vocab_size
        ):
            raise ValueError("eos_token_id must be a vocabulary ID")

    def stream_generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ) -> Iterator[Tensor]:
        """Yield [batch * returns, 1] tokens for equal-length unpadded prompts.

        Close the iterator when cancelling. Finished rows emit EOS until all rows
        finish. The requested prompt and generation budget must fit the context.
        """
        self._generation_args(
            input_ids,
            max_new_tokens,
            temperature,
            top_k,
            eos_token_id,
            top_p,
            repetition_penalty,
            num_return_sequences,
        )
        result = input_ids.repeat_interleave(num_return_sequences, dim=0)
        finished = torch.zeros(result.shape[0], dtype=torch.bool, device=result.device)
        if eos_token_id is not None:
            finished = result[:, -1] == eos_token_id
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        cache = None
        try:
            for _ in range(max_new_tokens):
                if finished.all():
                    break
                with torch.no_grad():
                    output = self(
                        result if cache is None else result[:, -1:],
                        past_key_values=cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    cache = output.past_key_values
                    scores = output.logits[:, -1].float()
                    if not torch.isfinite(scores).all():
                        raise ValueError("generation logits are nonfinite")
                    if repetition_penalty != 1:
                        seen = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, result, True)
                        adjusted = torch.where(
                            scores > 0, scores / repetition_penalty, scores * repetition_penalty
                        )
                        scores = torch.where(seen, adjusted, scores)
                    if temperature == 0:
                        token = scores.argmax(-1, keepdim=True)
                    else:
                        scores = (
                            (scores.double() - scores.double().max(-1, keepdim=True).values)
                            / temperature
                        ).float()
                        if top_k is not None:
                            values, indices = scores.topk(top_k, dim=-1)
                            scores = torch.full_like(scores, -torch.inf).scatter(1, indices, values)
                        if top_p < 1:
                            values, indices = scores.sort(dim=-1, descending=True)
                            remove = values.softmax(-1).cumsum(-1) > top_p
                            remove[:, 1:] = remove[:, :-1].clone()
                            remove[:, 0] = False
                            scores = scores.scatter(
                                1, indices, values.masked_fill(remove, -torch.inf)
                            )
                        token = torch.multinomial(scores.softmax(-1), 1)
                    if eos_token_id is not None:
                        token = torch.where(finished[:, None], eos_token_id, token)
                        finished |= token[:, 0] == eos_token_id
                    result = torch.cat((result, token), dim=1)
                yield token
        finally:
            for module, training in modes:
                module.training = training

    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
    ) -> Tensor:
        """Return prompts and cached continuations; sampling uses the caller's RNG."""
        chunks = list(
            self.stream_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=eos_token_id,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
            )
        )
        prompt = input_ids.repeat_interleave(num_return_sequences, dim=0)
        return torch.cat([prompt, *chunks], dim=1) if chunks else prompt
