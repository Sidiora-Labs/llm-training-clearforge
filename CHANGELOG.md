# Changelog

## 0.1.0 — Unreleased

Initial development of the English-first native training workflow, including
strict configuration, corpus preparation, byte-level tokenization, causal decoder
models, restartable optimization, evaluation, and portable native export.

The initial implementation includes six validated presets, original offline
fixtures, an executable pretraining-to-chat workflow, bounded checkpoint
retention, corruption preflight checks, and native-bundle integrity verification.

See [qualification](docs/qualification.md) for actual local test results and
measured fixture runs. No public release or remote CI execution is implied.
