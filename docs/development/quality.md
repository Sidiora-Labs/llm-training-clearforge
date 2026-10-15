# Development checks

Track changes in `spec/training-v1/spec.kvx`. Run the task's verification command
before marking it complete. Regenerate spec mirrors with `cg spec render`.

```bash
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python scripts/check_presets.py
python -m build
```

Tests cover configuration validation, dataset integrity, tokenizer round trips,
causal attention, assistant masking, gradient accumulation, checkpoint recovery,
and the CLI workflow. CUDA tests run when a compatible device is available.

## Project conventions

Use Apache-2.0 with `Copyright 2026 Sidiora Labs Inc.` attribution. Keep the root
license, package metadata, README, and example dataset declarations consistent.

Use a centered heading, monochrome flat-square badges, concise setup instructions,
documentation links, and a license footer. Write documentation around what users
need to do. Keep commands and configuration details close to the relevant steps.
