# Post-training

`clearforge.posttraining.train_posttraining` trains a copy of a native model with
supervised fine-tuning, LoRA, DPO, teacher-logit distillation, PPO, GRPO or CISPO.
The input model, reference and teacher remain unchanged. The callable API accepts
an in-memory record list or a JSONL path; it validates every example before
training and rejects examples exceeding the model context.

```python
from clearforge.adapters import AdapterConfig
from clearforge.posttraining import PostTrainConfig, train_posttraining

result = train_posttraining(
    model, tokenizer, "conversations.jsonl",
    PostTrainConfig("runs/chat-adapter", steps=100, device="cuda"),
    objective="lora", adapter_config=AdapterConfig(rank=8, alpha=16),
)
```

`model` is a `CausalLM` and `tokenizer` is its matching `ChatTokenizer`. Drivers
use FP32 AdamW, gradient clipping and deterministic PyTorch execution. Each step
consumes a shuffled batch, including partial batches at epoch boundaries. Select
`objective="sft"` for full supervised fine-tuning or pass `adapter_config` with
another objective to update only adapters.

## Input formats

Supervised training and distillation use conversations:

```json
{"messages":[{"role":"user","content":"Explain photosynthesis."},{"role":"assistant","content":"Plants use light to convert water and carbon dioxide into sugars."}]}
```

Only assistant content and its closing EOS contribute to the supervised loss;
padding, prompts and role markers are masked. No example is silently truncated.

DPO accepts a prompt string or a conversation ending with a user turn, followed
by two response strings:

```json
{"prompt":"What is 2 + 2?","chosen":"4","rejected":"5"}
```

Alternatively, `chosen` and `rejected` may be full message lists with an identical
prefix and different final assistant responses. DPO sums log probabilities over
only the final response and EOS. Its reference defaults to a frozen copy of the
initial model; supply `reference=` to use another native model with the same
architecture and tokenizer. `beta` controls the preference margin.

Distillation requires `teacher=` and `teacher_tokenizer=`. Tokenizer serialization
and vocabulary must match the student's exactly. A teacher may have a different
native architecture; inputs must fit both contexts. `temperature` and
`distillation_alpha` blend teacher KL and supervised cross entropy on assistant
targets. Teacher weights never receive gradients.

## Online policy training

PPO, GRPO and CISPO accept `{"prompt":"..."}` or `{"prompt":[...messages...]}`.
Provide an explicit reward function and a stable identifier for its implementation:

```python
def reward(messages, response):
    return float(response.strip() == "4")

result = train_posttraining(
    model, tokenizer, [{"prompt": "What is 2 + 2?"}],
    PostTrainConfig("runs/policy", device="cuda", samples_per_prompt=4,
                    max_new_tokens=32, policy_epochs=2),
    objective="grpo", reward_fn=reward, reward_id="arithmetic-exact-v1",
)
```

Rewards must be finite. Reward callbacks must be deterministic for exact resume;
change `reward_id` whenever their behavior changes. No remote service is called
by the driver. Generation samples from the temperature-scaled policy without
top-k truncation, and importance probabilities use that same distribution.
Dropout is disabled for online policy updates so old and new probabilities refer
to the same policy. Completion tokens alone contribute to the objective.

PPO learns an independent native `ValueModel`: a causal backbone and scalar value
head. It uses terminal sequence rewards, generalized advantage estimation and a
clipped value loss. `gamma`, `gae_lambda` and `value_coefficient` control these
terms. GRPO and CISPO normalize rewards within each prompt's group and require
at least two samples. Constant-reward groups have zero reward advantage. CISPO
uses detached clipped importance ratios. All three support reference-policy KL
through `beta`, asymmetric-free ratio clipping via `clip_ratio`, and repeated
updates on each rollout through `policy_epochs`.

## Checkpoints and exports

`latest.pt` is atomically replaced after each complete supervised update or online
rollout's policy epochs. It contains model, critic, AdamW moments, shuffled data
cursor, Python/NumPy/Torch/CUDA RNG states, metrics and input identities. Only load
checkpoints you trust: they contain Python pickle state. `metrics.jsonl` records
step, loss and, for online training, mean reward.

Pause at a complete boundary with `stop_after_steps=...`. Resume using the same
original input model, data, tokenizer, configuration, reference/teacher and
`resume="runs/.../latest.pt"`. Runtime and implementation identities must match.
The configured `steps` is the total horizon, including completed steps. Output
collisions are refused for fresh runs, and an exclusive run lock prevents
concurrent writers. An interrupted update resumes from its last committed
checkpoint.

A completed run exports `model/model.pt`, `model/model.json`, `model/tokenizer.json`
and provenance in the native bundle format. LoRA updates are merged into the
exported weights; the checkpoint retains the adapters and optimizer for resume.
`PostTrainResult.model` exposes the trained model and `checkpoint`, `step` and
`status` describe the committed result.

## Command-line training

Create `post.toml` alongside the intended output directory:

```toml
[training]
output_path = "adapter-run"
steps = 1000
batch_size = 4
learning_rate = 0.0001
device = "cuda"

[adapter]
rank = 8
alpha = 16
dropout = 0.05
targets = ["q_proj", "v_proj"]
```

```bash
clearforge post-train --model runs/chat-export --data data/chat.jsonl \
  --config post.toml --objective lora
clearforge chat --model adapter-run/model --prompt "Hello!"
clearforge merge-adapter --model runs/chat-export \
  --adapter adapter-run/adapter --output runs/merged-export
```

The output directory is resolved relative to the TOML file. Other paths are
resolved from the working directory. Every completed objective writes a native
`model/` export. Adapter runs also write the reusable `adapter/` package.

Select `sft`, `lora`, `dpo`, `distillation`, `ppo`, `grpo`, or `cispo` with
`--objective`. DPO optionally accepts `--reference`; distillation requires
`--teacher`. Online objectives take `--reward package.module:function` and a
versioned `--reward-id`. The callable receives prompt messages and response
text and returns a finite numeric reward. Keep reward code and data fixed when
resuming. Resume with `--resume OUTPUT/latest.pt`; use `--stop-after-steps N`
to pause at an optimizer boundary.

Adapter targets must not share parameter storage with another module; in
particular, adapting `lm_head` requires `tie_embeddings=false`. Default attention
targets remain supported.
