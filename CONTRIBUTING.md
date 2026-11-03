# Contributing

jev-stageA is an open research platform for JEV-based agent defenses. Contributions are welcome in three areas:

1. **New evaluations** — additional agent models, attack sets, and benchmarks run through the pipeline in [docs/evaluation.md](docs/evaluation.md). Report attack success and normal-task utility with Wilson 95% intervals, tune thresholds on suites disjoint from reporting suites, and state the exact model IDs, checkpoint revisions, and AgentDojo commit used.
2. **New defense layers** — RepE Sentinel probes, CounterSteer steering hooks, and argument-provenance features for the action monitor, integrated through the interfaces in [docs/defense-api.md](docs/defense-api.md). A new layer must fail closed: a detector that cannot produce a verdict raises an error instead of passing content through, and no layer converts another layer's block into an approval.
3. **New attack sets** — out-of-distribution injection templates in the style of `attacks_ood.py`. Each template must round-trip verbatim through the AgentDojo YAML environment encoding (the module enforces the round-trip check).

Ground rules:

- Tests exercise real code against real dependencies. The test suite contains no mocks; keep the test suite mock-free.
- Negative results are reportable results. A defense that fails a decision rule gets documented with the same prominence as a defense that passes a decision rule.
- Never commit API keys, tokens, or `.env` files. Model weights belong in the MetaPublic Hugging Face repositories, not in git.
- By contributing, you agree that your contribution is licensed under Apache-2.0.
