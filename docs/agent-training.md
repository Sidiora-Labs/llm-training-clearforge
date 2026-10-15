# Training a tool-using agent policy

`train_agent_policy` runs the native policy through real assistant/tool episodes
and updates it with GRPO or CISPO. Each prompt produces a group of independently
sampled episodes. A supplied reward function scores each episode; the driver
normalizes those rewards within the prompt group and applies the resulting
advantage to all of that episode's sampled assistant turns.

```python
from clearforge.agent_training import AgentTrainConfig, train_agent_policy
from clearforge.rollout import EpisodeLimits, ToolRegistry

# Define handlers at module scope so isolated tool workers can import them.
def add(a, b):
    return a + b

registry = ToolRegistry()
registry.register({
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
            "additionalProperties": False,
        },
    },
}, add)

def reward(record, episode):
    answer = episode.messages[-1].get("content", "")
    return float(episode.status == "complete" and answer == record["expected"])

if __name__ == "__main__":
    result = train_agent_policy(
        model, tokenizer,
        [{"messages": [{"role": "user", "content": "What is 2 plus 3?"}],
          "expected": "5"}],
        AgentTrainConfig(
            "runs/arithmetic-agent", device="cuda", steps=100,
            samples_per_prompt=4, max_new_tokens=128, policy_epochs=2,
            episode_limits=EpisodeLimits(max_turns=4, max_tool_calls=4),
        ),
        registry=registry, registry_id="integer-add-v1",
        reward_fn=reward, reward_id="exact-answer-v1", objective="grpo",
    )
```

`model` and `tokenizer` are a matching native `CausalLM` and `ChatTokenizer`.
The input model stays unchanged. Use a model trained on the structured tool
conversation format so valid calls are within its learned behavior. The registry
executes only explicitly registered, schema-valid calls in bounded workers;
there is no implicit shell, network or filesystem tool.

## Data and rewards

Data may be a JSONL file or a list of rollout records. Each record contains
`messages` ending at a generation-ready user turn, with optional JSON `expected`
metadata for the reward function. `prompt` strings or message lists are accepted
through the rollout adapter. If a record includes `tools`, its definitions must
match the explicit registry. Otherwise the registry definitions are attached.

The reward callback receives `(record, EpisodeResult)`. It can inspect final
messages, tool events, completion status and errors, and must return a finite
number. Invalid sampled calls, tool failures and episode limits remain observable
outcomes for the reward function. No reward heuristic or remote grader is enabled
implicitly. Both supported objectives require at least two samples per prompt;
a group with identical rewards has zero reward advantage.

The same `PostTrainConfig` optimization controls apply: learning rate, clipping,
KL coefficient `beta`, sampling temperature, policy epochs, batch size and total
steps. `AgentTrainConfig.episode_limits` additionally bounds turns, calls, tokens,
output sizes and tool execution time. Agent training supports at most 128 sampled
turns per episode. Each optimizer boundary covers a complete group of episodes
and all its policy epochs.

## Which tokens receive gradients

The policy itself samples every assistant token. The driver retains the exact
prompt IDs and sampled IDs from each native generation call, including malformed
outputs. It does not retokenize generated assistant text or substitute a scripted
answer. Each assistant turn is scored under the actual context used to sample it.

Prompt tokens, tool definitions, tool observations, padding and prior turns in
a later turn's context receive ignored labels. Only newly sampled assistant
tokens, including a sampled EOS, contribute to that turn's policy likelihood.
Thus tool results influence subsequent decisions through context without being
trained as policy actions. The objective averages over sampled action tokens
across the update; longer episodes can contribute more tokens.

Generation uses the configured temperature with no top-k/nucleus filtering or
repetition penalty. Importance probabilities use the same temperature-scaled
distribution. Dropout is disabled during rollout and policy updates. The frozen
reference defaults to the original policy; supply `reference=` for another native
model with the same architecture and tokenizer. It contributes a KL regularizer
and receives no gradients. CISPO detaches its clipped importance ratios.

## Recovery and outputs

Every completed update atomically publishes `latest.pt` with policy weights,
AdamW moments, data cursor, all RNG states, metrics and implementation/input
identities. `metrics.jsonl` records loss, mean reward, episode statuses, assistant
turns, sampled tokens and executed tool calls. A completed run exports a standard
native bundle in `model/`.

Pause with `stop_after_steps` and resume with `resume="runs/.../latest.pt"`, using
the same original model, tokenizer, data, configuration, registry and rewards.
`registry_id` and `reward_id` are required version identifiers: change them when
handler or reward behavior changes. Exact continuation assumes deterministic
handlers/rewards and stable execution limits; external side effects and
wall-clock timeouts are not replayable state. SIGINT and SIGTERM request a stop
at the next complete update. Interrupted work resumes from the previous durable
boundary. Checkpoints use Python serialization and must come from trusted runs.

## Command line

Expose your trusted registry factory and reward function from an importable Python
module, for example `my_tasks:make_registry` and `my_tasks:reward`:

```bash
clearforge train-agent --model runs/chat/model --data tasks.jsonl \
  --config agent.toml --objective grpo \
  --registry my_tasks:make_registry --registry-id integer-add-v1 \
  --reward my_tasks:reward --reward-id exact-answer-v1
```

The configuration uses the same training fields as the Python API:

```toml
[training]
output_path = "runs/arithmetic-agent"
steps = 100
learning_rate = 0.00005
device = "cuda"
samples_per_prompt = 4
max_new_tokens = 128
policy_epochs = 2

[training.episode_limits]
max_turns = 4
max_tool_calls = 4
max_generated_tokens = 512
tool_timeout = 5
```

Use `--reference` for a separate native reference bundle, `--stop-after-steps`
for a controlled pause, and `--resume` to continue the exact checkpoint. Module
handlers must be top-level, importable callables because tool isolation uses
spawned processes; local closures and lambdas are rejected.
