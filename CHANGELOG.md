# Changelog

## Unreleased

- Cached, streamed and batched generation with nucleus sampling and repetition controls.
- Sparse expert decoders and YaRN rotary scaling.
- LoRA adapters, preference optimization, distillation, and PPO/GRPO/CISPO training.
- Multi-turn agent training with explicit tool registries and completion-only policy gradients.
- Distributed CPU/CUDA training, compilation, activation checkpointing, and data-loading controls.
- Standard Transformers model interchange with tokenizer and logit parity checks.
- Tool-aware conversation, preference, and rollout dataset adapters.
- Chat API, streaming responses, and responsive browser conversations.
- Generation evaluation suites, tokenizer diagnostics, and native/HTTP rollout backends.
- Pinned English corpus pipeline for the complete 100M pretraining and chat-training run.

## 0.1.0

- English text pretraining and supervised chat fine-tuning.
- Six model presets, including 100M, 300M, and 1B configurations.
- Dataset validation, deduplication, byte-level BPE, and token packing.
- Resumable training with bounded checkpoint retention.
- Evaluation, native model export, and interactive chat.
- Example datasets and a complete pretraining-to-chat walkthrough.
- Apache-2.0 licensing and Sidiora Labs Inc. branding.
