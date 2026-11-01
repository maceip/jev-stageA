# Track 2 — Measured request reuse, sounder replay, and 99.41% offline coverage

Agent: `agent-937c15-r3`; branch `raffle/agent-937c15-r3`; baseline `c1576e5d6c018c2c2edf27cca0d74a3c5ee3e7a6`. All changes are in the automatically provisioned `jev-stageA` worktree. No primary checkout, remote job, or GPU process was changed. All Python operations used `/Users/mac/.local/bin/uv run`.

## Results

| Measure | Baseline | Upgraded |
|---|---:|---:|
| Offline unittest methods | 0 | 19 |
| Executable-line coverage | 0% | 99.41% (1,174/1,181) |
| Nontrivial test LOC | 0 | 444 |
| Efficiency, coverage²/max(40,LOC) | 0 | 22.26 |
| Failures / errors / skips | — | 0 / 0 / 0 |
| Real archived benign replay agreement | not measured here | 91/91 |

The official meter is executable-line coverage, not independent branch coverage. The single table-driven test file exercises all nine target modules and real AgentDojo TaskSuite/FunctionsRuntime/pipelines with temporary YAML/JSONL/gzip fixtures. Only GPU/model and remote API boundaries are doubled. The scoring adapter outside the official target list is also behaviorally tested. Approximately four seconds per coverage run includes import overhead. JEV's test suite maximizes meaningful coverage but is not claimed to meet AWI's separate >=60 efficiency target.

## Concrete speed improvement, not a GPU-throughput claim

`score.py:13` adds an insertion-order-preserving SHA-256 key over the exact JSON request; `score.py:23` adds `CachedScorer` with a thread-safe bounded LRU, defensive copies, uncached failures, ordered multi-request results, and reuse counters. `secjev_scorer.py:37` keeps the public `score` interface and adds `score_many`, which runs unique requests sequentially using the established Kev interface. Object key order is intentionally retained because prompt serialization may depend on it. The cache is scoped to one scorer/checkpoint instance, preventing accidental cross-model reuse. `score.py:174` factors validated overlapping chunks and the CLI exposes `--cache-size`.

A read-only pass over `data/gptoss20b_stageA.tgz`, in order benign, important_instructions, ood_bland_note, ood_metadata_field, ood_quoted_reply, ood_zero_width, ood_llmail, measured:

| Initial-window request accounting | Count |
|---|---:|
| Valid episodes / excluded errors | 5,476 / 166 |
| Requests: tool chunks plus both action variants | 56,489 |
| Globally distinct exact request hashes | 26,125 |
| LRU capacity | 4,096 entries |
| LRU hits / potential backend forwards | 25,325 / 31,164 |
| Potential initial-forward avoidance | **44.83%** |

The calculation is `25,325 / 56,489`. It measures reuse opportunities, not inference timing; token-limit retries and adaptive window shrinkage were not simulated. The cache bounds entries rather than bytes and does not persist across processes. No speculative Kev batch API, GPU minibatching, or remote throughput result is claimed. Deterministic tests assert actual counting-backend forward reduction, concurrency safety, eviction, batch order, mutation isolation, and failure retry.

## Efficacy and analytical integrity repairs

1. **Actual adaptive monitor wiring.** `adaptive.py:232–234` now passes `a.mode`, `sj`, and `action` to `pipeline`. The baseline passed an ActionGate object in the mode position and omitted the monitor arguments. Tool arms could assert and action arms could use a plain executor. Regression tests run real pipelines for all seven defenses and verify meta/content distinction. This is a defect in the baseline source revision, not proof that archived experiments used that same revision.
2. **Model-visible error parity.** `score.decision_points`, both defense gates and adaptive gates now prefer a nonempty tool error over content, as the agent clients do. Redaction clears the original error channel so a flagged payload cannot remain visible there. Tests cover simultaneous ordinary content plus injected error, empty-error fallback, stop/redact behavior, and action history. Blocked action results carry an explicit marker.
3. **Held-out threshold selection.** `analyze.py:149–194` validates nonempty disjoint suites and calibrates solely on benign workspace/travel; `analyze.py:245–291` reports banking/slack as primary and labels full-suite results descriptive. Threshold candidates use exact `nextafter(score,+inf)` operating regions, including a pass-all threshold above one. Native and calibrated policies remain separate. This is a two/two split, not complete cross-fitting or independent confirmation: existing suites were already studied.
4. **Fail-closed score integrity.** `analyze.py:35–129` rejects missing/nonfinite/out-of-range/error scores, incomplete decision-point sets, missing model groups, duplicate episodes, and semantically conflicting duplicate scores. Timing-only duplicate differences are tolerated. A high early score cannot hide a later invalid score. Historical overlapping score shards now require explicit provenance selection; the Track 1 audit found no native/u95 decision changes from their small numeric differences.
5. **Observed execution, not proposals.** `replay.py:22` requires corresponding tool-result witnesses and excludes trailing unexecuted proposals and explicitly blocked calls. For tool-output gates, the full executor batch has already run when a result is flagged; intra-batch action cuts differ. Real archived benign replay matched 91/91 outcomes (workspace39, travel20, banking16, slack16). This validates those complete recordings, not every possible defended cut.
6. **Honest redaction bounds.** `replay.redact_bounds` and `analyze._redact_summary` return the observed outcome if no intervention occurs, and `[0,1]` for each binary outcome after any intervention. These are identification bounds, not Wilson intervals or predicted benefits. New continuations require live reruns.
7. **Adaptive resume/report identity.** `adaptive.py:241–265` records repeat, mode, threshold, model, attacker, and budget and will not silently reuse incomplete legacy records. `report.adaptive_table` preserves these distinctions rather than collapsing repeats and labels the recorded budget rather than assuming ten rounds.

## Paraphrase and corpus repairs

`make_paraphrases.py:28–87` has pure token/validation/JSON parsing helpers. It preserves ordinals, names, quoted strings, structured identifiers and their multiplicity; rejects substring substitutions such as 4 inside 14, introduced numeric/structured details, malformed candidate structures and whitespace duplicates. The task mapping is explicit and progress rows contain source-prompt hashes, so stale paraphrases are revalidated rather than silently reused. Lexical validation still cannot establish semantic equivalence. `verify_corpus.py` now fails descriptively for empty/fully filtered corpora instead of indexing an empty latency array.

## Reproduction and dependency scope

`pyproject.toml` + `uv.lock` pin the real CPU-side framework and parsing/API dependencies (AgentDojo0.1.35). `OFFLINE_TESTING.md` documents:

```sh
/Users/mac/.local/bin/uv run --frozen python -m unittest discover -s tests -v
/Users/mac/.local/bin/uv run --frozen python /Users/mac/Documents/Pie/raffle/coverage_runner.py "$PWD" jev
```

All source files parsed, the tests pass, and `git diff --check` passed. No GPU/Kev inference validation was performed. Remaining metric misses are one defensive chunk-count guard and six analysis validation/empty-data/fallback lines, not hidden failing tests.

## Limits that remain

Offline action-content inputs for later calls in an assistant batch still differ from sequential live inputs, because prior tool results enter live context before subsequent calls. Replay exactness is conditional on matching inputs and validated semantics. Ordinary score-file resume still lacks checkpoint/source fingerprints; adaptive resume does not fingerprint endpoints, judge configuration or SecJev checkpoint. Wilson intervals are descriptive, not valid optional-stopping guarantees, and task variants are correlated. No SPRT was added. No automatic action-authorization labeling was built from episode success: that would mistake trajectory outcomes for per-call authorization gold. Track 1 specifies a reviewed, leakage-safe labeling recipe instead. The work improves runnable machinery and rigor; it does not claim new measured GPU throughput or live attack-defense efficacy.
