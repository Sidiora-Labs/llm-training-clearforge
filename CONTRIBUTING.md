# Contributing

Read the implementation contracts in `docs/development/contracts.md` before
changing a component boundary. Keep changes focused and cover externally visible
behavior with tests that can detect a real regression.

## Local development

Install the development extras in an isolated Python environment. Run the CPU
test suite, formatting, and static checks before marking a task complete:

```bash
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

Tests must not download model weights, call paid APIs, or depend on private data.
Use small deterministic fixtures and temporary directories. Avoid global random
state changes that escape a test.

## Specification and fleet

```bash
cg brief
cg spec next
cg fleet begin <task-id> --agent <name>
```

Work in the returned task worktree. Set the reported `CG_AGENT`, `CG_ROLE`,
`CG_PARENT`, `CG_FEATURE`, and `CG_WAVE` environment values. Start the task, implement
its declared scope, run the checks, and create a task-tagged snapshot and Git
commit. Mark the task done only after its verification command passes. The
coordinator merges completed tasks with `cg fleet merge-up` and lands each wave
behind test and lint gates.

Document setup, configuration, error handling, and recovery behavior.

## Packaging and accelerator checks

```bash
python scripts/check_presets.py
python -m build
```

The GitHub Actions workflow checks Python 3.11 and 3.12 on CPU and builds source
and wheel distributions. CUDA resume tests run when compatible hardware
is available; otherwise pytest reports explicit skips. To check a CPU-only path
on a GPU machine, run `CUDA_VISIBLE_DEVICES='' python -m pytest -q`.

Use `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2` for small test fixtures on machines with
large CPU affinity masks. Keep Torch thread settings consistent across resume
comparisons. Specification worktree tests must import that worktree's source:
use `PYTHONPATH=src` if an editable install points at a different checkout.
