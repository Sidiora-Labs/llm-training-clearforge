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
its declared scope, run qualification, and create a task-tagged snapshot and Git
commit. Mark the task done only after its verification command passes. The
coordinator merges qualified tasks with `cg fleet merge-up` and lands a wave
behind test and lint gates. No remote publication is implied.

Document errors, recovery behavior, resource limits, and measured limitations.
Do not claim model quality from a functional smoke test or claim an export format
that downstream software has not successfully loaded.
