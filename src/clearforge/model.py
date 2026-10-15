"""A pre-normalized causal decoder with explicit training and generation contracts.

Padding is represented by ignored labels, not an attention mask. Right-padded
training sequences are supported: padding must never precede real tokens.
Generation accepts one unpadded prompt and never silently discards context.
"""

from __future__ import annotations

import math
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

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x: Tensor) -> Tensor:
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
        frequencies = self.theta ** (
            -torch.arange(0, self.head_dim, 2, device=x.device, dtype=torch.float32) / self.head_dim
        )
        angles = positions[:, None] * frequencies[None, :]
        cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
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
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            self.rotary(q),
            self.rotary(k),
            v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return self.out_proj(attended.transpose(1, 2).contiguous().view(batch, length, -1))


class SwiGLU(nn.Module):
    """Gated feed-forward expansion with independent gate and value projections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DecoderBlock(nn.Module):
    """Attention and feed-forward residuals with pre-normalization."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size)
        self.attention = CausalAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        return x + self.dropout(self.mlp(self.mlp_norm(x)))


class CausalLM(nn.Module):
    """Decoder-only language model; loss shifts aligned labels exactly once.

    This implementation materializes full vocabulary logits. It does not claim
    fused loss, KV caching, sliding-window inference, or left-padding support.
    """

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
                residual = name.endswith(("attention.out_proj.weight", "mlp.down_proj.weight"))
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

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> CausalLMOutput:
        self._validate_ids(input_ids)
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
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.config.vocab_size).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return CausalLMOutput(logits=logits, loss=loss)

    def parameter_count(self) -> int:
        """Count unique parameters, including each tied matrix only once."""
        return sum(parameter.numel() for parameter in self.parameters())

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Return prompt plus continuation for one unpadded sequence.

        The complete requested budget must fit max_seq_len, even if EOS could
        stop earlier. An existing terminal EOS returns the prompt unchanged.
        Sampling uses the caller's Torch RNG. All module training flags are
        restored, including on failure; inference always disables dropout.
        """
        self._validate_ids(input_ids)
        if input_ids.shape[0] != 1:
            raise ValueError("generate supports batch size one only")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if input_ids.shape[1] + max_new_tokens > self.config.max_seq_len:
            raise ValueError("prompt plus generation budget exceeds max_seq_len")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a finite nonnegative number")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a finite nonnegative number")
        if top_k is not None and (
            type(top_k) is not int or not 1 <= top_k <= self.config.vocab_size
        ):
            raise ValueError("top_k must be an integer in [1, vocab_size]")
        if eos_token_id is not None and (
            type(eos_token_id) is not int or not 0 <= eos_token_id < self.config.vocab_size
        ):
            raise ValueError("eos_token_id must be a vocabulary ID")
        result = input_ids.clone()
        if max_new_tokens == 0 or (
            eos_token_id is not None and result[0, -1].item() == eos_token_id
        ):
            return result
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            for _ in range(max_new_tokens):
                scores = self(result).logits[:, -1, :].float()
                if not torch.isfinite(scores).all():
                    raise ValueError("generation logits are nonfinite")
                if temperature == 0:
                    next_token = scores.argmax(-1, keepdim=True)
                else:
                    # Center first to avoid positive overflow at very low temperatures.
                    scores = (
                        (scores.double() - scores.double().max(-1, keepdim=True).values)
                        / temperature
                    ).float()
                    if top_k is not None:
                        values, indices = torch.topk(scores, top_k, dim=-1)
                        choice = torch.multinomial(values.softmax(-1), 1)
                        next_token = indices.gather(-1, choice)
                    else:
                        next_token = torch.multinomial(scores.softmax(-1), 1)
                result = torch.cat((result, next_token), dim=1)
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        finally:
            for module, training in modes:
                module.training = training
        return result
