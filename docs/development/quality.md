# Quality gates

Changes must have observable acceptance criteria in `spec/training-v1/spec.kvx`.
The `cg` task board records implementation status; the fleet records isolated task
branches and qualification. Generated spec mirrors are not edited by hand.

Required checks cover:

- Invalid configuration and unknown fields before model allocation.
- Malformed input, duplicate documents, source metadata, and split disjointness.
- Tokenizer round trips and literal control-token injection.
- Causal attention and a numerically checked next-token loss.
- Prompt and padding masking in supervised chat.
- Finite parameter updates and correct gradient accumulation.
- Checkpoint atomicity, incompatible-input rejection, and interruption recovery.
- Uninterrupted versus resumed CPU training with nonzero dropout.
- A real CLI workflow that prepares data, trains, evaluates, exports, and generates.

CPU tests are mandatory and do not require internet access. CUDA smoke tests are
additional evidence when a device is available; they are not silently replaced
by mocked accelerator calls. Record exact commands, dependency versions, and
results. Do not represent a smoke run as a trained general-purpose chatbot.
