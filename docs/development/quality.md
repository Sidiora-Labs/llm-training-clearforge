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

## Project presentation and licensing

Clearforge follows the Gideon and HyperMind Apache-2.0 convention, with
`Copyright 2026 Sidiora Labs Inc.` attribution. Keep the root license, Python
package metadata, README badge/footer, and bundled fixture license declarations
consistent. Codify uses MIT; it is not the licensing template for this project.

Use the centered project heading, monochrome flat-square badges, a concise
purpose statement, source-install instructions, documentation links, and a
license footer. Version and qualification statements must match actual package
metadata and evidence. Add repository, release, translation, and CI-status links
only when their targets exist; configured CI is not a passing remote CI run.
