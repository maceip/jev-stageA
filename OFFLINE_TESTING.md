# Offline regression checks

Use the worktree's `uv` environment. `pyproject.toml` and `uv.lock` pin the CPU-side dependencies, including AgentDojo 0.1.35. The project is a collection of scripts, not an installable package.

```sh
/Users/mac/.local/bin/uv run --frozen python -m unittest discover -s tests -v
```

The tests must use the real AgentDojo types, parsing, defense adapters, and local runtime. Model inference and remote API boundaries are replaced with deterministic test doubles. This command must not load GPU weights, contact an agent or judge API, or start evaluation jobs.

For the Round 3 coordinator's executable-line metric, from an isolated worktree:

```sh
/Users/mac/.local/bin/uv run --frozen python /Users/mac/Documents/Pie/raffle/coverage_runner.py "$PWD" jev
```

That evaluator traces executable source lines. It does **not** calculate independent branch coverage, model accuracy, GPU throughput, or end-to-end defense efficacy. It measures nine selected offline modules; code outside that selection still needs direct behavioral assertions.

The lightweight dependencies do not include PyTorch, Transformers, SecJev model weights, or the private/local Kev training implementation. Real model runs require the separately configured research environment documented in `RUNBOOK.md`. Do not use this offline environment to restart or modify an active run.
