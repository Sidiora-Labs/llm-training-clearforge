# Decoder and generation

`CausalLM(ModelConfig(...))` implements a pre-normalized decoder with RMSNorm,
interleaved RoPE, grouped-query causal attention, SwiGLU, and optional tied token
embeddings. Dense defaults preserve existing parameter names and initialization.

## Forward and cache

```python
prefill = model(prompt_ids, use_cache=True, logits_to_keep=1)
next_step = model(
    next_ids,
    past_key_values=prefill.past_key_values,
    use_cache=True,
    logits_to_keep=1,
)
```

Inputs are nonempty `torch.long` tensors shaped `[batch, sequence]`. Each cache
entry is a `(key, value)` pair shaped `[batch, kv_heads, cached_tokens, head_dim]`.
The tuple contains one entry per layer. Cache length, batch, heads, device, and
active inference dtype must agree. RoPE starts at the cached position; multi-token
continuations use an offset causal mask. Cached and new tokens together must fit
`max_seq_len`. Reuse a cache only with the same model weights and precision.

`logits_to_keep=0` returns every new token's logits; a positive integer returns
that many trailing positions. Labels require full logits and no past cache.
Labels align with input IDs and use `-100` for ignored positions. The model shifts
labels once and averages cross-entropy over supervised next tokens. Right-padded
training is supported by ignored labels. Batched inference requires equal-length
unpadded prompts; there is no padding-mask interface.

## Sampling and streaming

`generate` returns prompts followed by generated tokens. `stream_generate` accepts
the same keyword arguments and yields only the new tokens, one tensor shaped
`[batch * num_return_sequences, 1]` at a time. Return sequences are grouped by
prompt. Generation uses the KV cache and computes only the final position's logits.

Controls are `max_new_tokens`, `temperature`, `top_k`, `top_p`,
`repetition_penalty`, `eos_token_id`, and `num_return_sequences`. Temperature zero
selects greedy decoding. Positive temperatures use the caller's Torch RNG;
seed it before a call for repeatable sampling. Repetition penalties apply to all
previously seen prompt and generated token IDs. Top-k filtering precedes nucleus
filtering, whose retained prefix includes the token crossing the probability
threshold. Finished batch rows emit EOS until every row finishes.

The full requested prompt plus generation budget must fit the context window.
An existing terminal EOS marks a prompt finished. Generation disables dropout
and restores every module's training flag after completion or failure. A streaming
consumer must close its iterator when cancelling:

```python
from contextlib import closing

with closing(model.stream_generate(ids, max_new_tokens=64, temperature=0.8,
                                   top_p=0.9, eos_token_id=eos_id)) as tokens:
    for token_batch in tokens:
        consume(token_batch)
        if cancelled():
            break
```

Serialize generation calls sharing a model because evaluation mode is module
state. Streaming does not leave a `no_grad` context active in consumer code.

## Sparse experts

Set `num_experts` above zero to replace each dense feed-forward layer with a
router and independent SwiGLU experts. `experts_per_token` selects the top-k
experts. Their softmax probabilities are renormalized over the selected set.
Experts process only their assigned tokens; outputs are combined with their
routing weights. Attention remains dense.

`router_aux_loss_coef` scales the load-balancing objective. For each layer it is
`coefficient * num_experts * sum(mean_assignment * mean_router_probability)`.
Assignments are averaged over tokens and selected slots. `output.aux_loss` is the
sum across layers; when labels are supplied, `output.loss` already includes it.
Do not add it again in the trainer. Sparse models have distinct checkpoint keys;
loading dense weights into them requires an explicit conversion.

## YaRN context scaling

`yarn_factor=1` preserves standard RoPE. Larger values enable frequency scaling
relative to `yarn_original_context`, using `yarn_beta_fast` and `yarn_beta_slow`
to define the frequency correction ramp. Low-frequency coordinates interpolate
by the scale factor while high-frequency coordinates retain their frequencies.
Both sine and cosine are multiplied by `1 + 0.1 * log(yarn_factor)`. Set
`max_seq_len` explicitly to the intended operational limit. Scaling changes
position encoding; evaluate long-context quality before using an extended window.
