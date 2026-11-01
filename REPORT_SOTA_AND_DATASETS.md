# Track 1 — SOTA research and dataset synthesis

**Repo:** `jev-stageA` · **Agent:** `agent-786e4f-r3` · **Date:** 2026-11-01
**Full dossier (committed):** [`SOTA_DOSSIER.md`](SOTA_DOSSIER.md) — 570 lines, 10,571 words,
114 URLs, every source tagged `[V]` fetched-and-confirmed / `[R]` recalled-from-training-knowledge
/ `[L]` read from the local artifacts. This document is the synthesis; the dossier is the
evidence.

---

## 1. The state of the field, and what it means for this project

**The literature has moved past "detect the injection."** The 2026 SoK survey of the prompt-injection
threat landscape finds that *no* tested defense simultaneously achieves high trustworthiness, high
utility and low latency, and explicitly reports that text-level (i.e. tool-output-detection)
defenses "fail to provide robust security guarantees against sophisticated payload manipulations"
`[V]` (`arXiv:2602.10453`). The same survey notes that execution-level defenses "can enforce
strict action boundaries" but often "resort to aggressive 'early refusal' strategies that render
the agent unusable for complex tasks," and that only **6 of 41** defense papers intervene at
segment level `[V]`.

That last clause is, almost word for word, a description of our own ~17 pp benign-utility loss.
It reframes the project's central problem: **the ~17 pp is a scoping bug, not a threshold bug.**
We ask one global question ("is this action authorized?") over a heterogeneous tool space; the
correct design asks a *per-task scoped* question ("is this call inside the plan the user's
request implies?").

**The action plane is where the tractable security is.** Progent (programmable privilege control
for LLM agents) reduces AgentDojo ASR **41.2 % → 2.2 %** while preserving utility, across
AgentDojo, ASB and AgentPoison `[V]` (`arXiv:2504.11703`). Our own artifacts already show the
same ordering: the action monitor preserves **40.2 %** (gpt-oss-20b) / **42.9 %** (qwen3-30b)
utility under attack versus **8.5 %** / **6.6 %** for the tool-output detector `[L]`.

**An honest baseline caveat that must travel with every claim.** At the `native` threshold,
**PIGuard beats SecJev on both axes** on gpt-oss-20b — PIGuard benign utility **40.7 %** /
standard ASR **0.3 %** / adaptive **0.0 %**, versus `secjev_action_meta` **50.5 %** / **1.6 %** /
**5.0 %** `[L]`. SecJev buys ≈ +10 pp benign utility at a cost of 1.6–5.0 pp ASR. That is a
defensible operating point, but it is a *trade*, not a win. Conversely, on gpt-oss-20b the LLM
judge dominates our action monitor (benign **67.0 %**, standard ASR **0.2 %**) while on
qwen3-30b the same judge collapses (benign **20.6 %**) `[L]`. **The judge is model-dependent;
SecJev is not.** That asymmetry is SecJev's strongest argument and should be framed as such.

---

## 2. The "ceiling effect" failure mode — what it actually is, and how to measure it

This is the question the track was asked to answer, and the answer is uncomfortable: **a 0/900
result has not yet been shown to be a ceiling effect at all, and the local artifacts disagree with
it.**

### 2.1 The internal inconsistency, flagged as P0

The local artifacts report *defense-outcome* ASR: `secjev_tool` = **0.0 %** on standard AgentDojo
at threshold 0.5, and 0.0–0.7 % across all five OOD sets `[L]`. A detector that scored **0/900**
on the injection subset could not produce 0.0 % ASR, because the undefended ASR is **24.5 %** —
the attacks demonstrably work against this model. Therefore the "0/900 Rule 1" number and the
artifact numbers are **different quantities**: most likely a *detector-level* label-accuracy
measurement taken against a different SecJev checkpoint, or against a different question type
than the `secjev_tool_request` / `noul` / `"true"` path `score.py:42-68` actually uses.

**Consequence: until the exact script, checkpoint, question type, ground-truth key and episode
list behind "0/900" are recorded, no claim about a ceiling effect can be published** — and it is
possible that §2.2 is moot. This is recorded as item 0 in the dossier's recommended-architecture
section and as a deviation in the dataset report.

### 2.2 The two hypotheses are distinguishable, and we have not distinguished them

At n = 900 the statistics are *not* the problem. Wilson(0/900) = [0.0000, 0.00425];
Wilson(45/900) = [0.0376, 0.0663]; the Newcombe hybrid-score CI for the difference is
**[−0.0663, −0.0369]** and excludes 0, so 0/900 vs 45/900 is a real, detectable difference with no
ambiguity. Power against the literal null is ample: a *single* detection in 900 rejects at
α ≈ 1/901, and a true rate of 0.5 % is detected with power ≈ 94 %. The MDE at n = 900 is
≈ 0.93 pp at a true rate of 1 % and ≈ 0.5 pp at 0.5 %.

So a 0/900 result means one of two *very* different things:

* **(i) The score has no ranking signal** — the benign and injected score distributions coincide.
  ⇒ The detector is **blind**; no threshold will fix it. (Architecture problem.)
* **(ii) The score ranks fine but never crosses the 0.5 argmax threshold** — the model's own
  label head is mis-calibrated for this task. ⇒ The detector is **weakly informative**, and a
  calibration fix suffices. (Engineering problem.)

They have opposite implications and we have not distinguished them. **The single most important
measurement gap in the project** is one evaluation pass.

### 2.3 The protocol that settles it (ceiling-effect-safe measurement)

Replace "rate at a fixed threshold" with "**separation at a fixed false-positive rate**":

1. **Dump continuous scores** for every benign and injected episode. Everything else is computed
   from those dumps.
2. **AUROC and AUPRC** with BCa bootstrap CIs (10,000 resamples over episodes). AUPRC is primary
   because the benign:attack ratio in AgentDojo tool outputs is heavily imbalanced.
3. **TPR at fixed FPR** — TPR@1 %FPR and TPR@5 %FPR plus the full ROC curve. This is invariant to
   the decision threshold and therefore immune to the ceiling.
4. **Cohen's d / Hedges' g** between the two score distributions. d ≈ 0 means no separation; d ≥ 0.8
   is a large effect a badly-set threshold could hide entirely.
5. **Reliability/calibration curve + ECE.** A *well-calibrated but uninformative* model looks very
   different from an *informative but mis-thresholded* one; the reliability curve separates them.
6. **Positive controls at three difficulty tiers**, to distinguish blindness from weakness:
   *Tier 1* trivially visible ("ignore previous instructions" — a detector failing here is
   architecturally broken); *Tier 2* visible but obfuscated (zero-width, whitespace, homoglyphs —
   we already ship `ood_zero_width`); *Tier 3* invisible/semantic (metadata fields, quoted replies,
   bland workflow notes — we already ship these, and they are the right set for measuring the
   ceiling). We also already ship `make_paraphrases.py` and a `tool_knowledge` control set.
7. **Paired bootstrap over episodes** for every between-detector difference (ΔAUROC,
   ΔTPR@1 %FPR, ΔASR) — resample episodes, not rows.
8. **MDE and power** for every headline comparison, from an exact power function.
9. **Pre-specify the primary endpoint.** "Detection rate on the injection subset" and "ASR under
   attack" are currently used interchangeably; they are different quantities, and the local
   artifacts mix them (`report.py:8` lists `secjev_tool` and `secjev_action_meta` in one
   "Detector" column, though one is a text classifier and the other an action classifier).

### 2.4 The pairing, and the equivalence question

Because our detectors score the *same* episodes, the correct test is **McNemar exact** on
discordant pairs, not an unpaired two-proportion test. With b + c ≈ 45 discordant episodes,
SE ≈ √45/900 ≈ 0.0075 — roughly half the unpaired SE at comparable rates. McNemar is markedly
more powerful and it uses the information we actually have.

For any *equivalence* claim ("SecJev's detector is equivalent to an LLM judge") use **TOST**
(Schuirmann 1987) with Newcombe's hybrid-score interval — SAS `PROC FREQ` implements
`RISKDIFF METHOD=NEWCOMBE` with `EQUIV` and TOST `[V]`. Recommended margins for our setting:
**Δ = 5 pp** benign-utility loss, **Δ = 2 pp** ASR difference, **Δ = 5 pp** detection rate.

Note a subtlety worth stating: on a 0/900 vs 0/900 comparison with a 5 pp margin, the 90 % CI is
[−0.0043, +0.0043], which *is* inside ±5 pp — so TOST would **successfully prove equivalence** at
the 5 pp margin. That is a legitimate and publishable claim ("statistically indistinguishable at
the 5-percentage-point level on this injection subset"), and it is *stronger* than "we don't
know". But it must be stated as **equivalence within margin**, not as "both are equally bad":
equivalence does not imply uselessness.

---

## 3. Threshold-calibration honesty

The current harness is not defensible, and the dossier says so in detail.

* `analyze.py:110-124` selects the `u95` threshold **and reports benign utility on the same
  benign episodes** `[L]`. The `u95` row therefore mixes an in-sample maximum with an
  out-of-sample security number. This must be split before publication.
* The harness reports **Wilson intervals only** (`report.py:16-19`) and **no tests at all**. It
  needs Newcombe hybrid-score CIs for all proportion differences, McNemar exact for paired
  detector comparisons, TOST for any equivalence claim, paired bootstrap for ΔAUC, and an
  MDE/power line under every headline.

**The protocol we implemented** (Track 2) is the concrete version of §2.3's item 3–4 at the
threshold level:

* `analyze.cross_suite_calibration(...)` performs **out-of-fold** threshold calibration: τ is
  selected on the *other* folds only and then applied to the held-out fold, reporting the
  **optimism gap** (in-sample minus out-of-fold) and the in-sample threshold alongside it, so
  the size of the self-deception is measured rather than assumed.
* The **Leave-One-Tool-Family-Out** 6-fold design is the recommended fold structure: fold 1 mail,
  fold 2 messaging/channel, fold 3 calendar, fold 4 file/document, fold 5 web-fetch, fold 6
  money+booking. In outer fold *i*, tune τ on folds ≠ *i* and evaluate **only** on family *i*,
  across every suite that uses it; inner 5-fold grouped CV *by task-template id* selects τ by the
  pre-registered rule (smallest τ with Wilson-UCL(benign FPR) ≤ 0.02).
* Report **mean ± 95 % CI (paired bootstrap, BCa, 10,000 resamples)** per metric across folds
  **and the worst fold**; **benign utility**, **benign FPR**, **ASR**, and **utility-under-attack**
  each with their own Wilson CI and Newcombe risk difference vs undefended; **McNemar exact** for
  paired comparisons (reporting *b* and *c*, not just p); TOST for equivalence; the
  **generalisation gap** Δ_gen = in-sample-tuned − out-of-fold for utility, FPR and ASR; and the
  **leak sensitivity** Δ = metric(LOSO) − metric(LOTO-TO).

---

## 4. Dataset synthesis: the action-authorization set, built without cross-suite leakage

This is the track's other half, and it is the one piece of the plan we could build without
network access or a GPU.

### 4.1 Why the current labels leak

Each AgentDojo suite exposes a nearly disjoint tool set, so a threshold calibrated on one suite
generalises to another only if the *action vocabulary* does not overlap. If it does overlap, an
in-fold threshold is measuring memorisation of a tool name, not a transferable authorization
decision. The consequence of leakage is a systematically optimistic threshold and an inflated
benign-utility number.

### 4.2 What we built

`build_authz_dataset.py` (stdlib only) reads the real stage-A archives and emits **41,499
tool-call-level authorization decisions**:

| split | examples | authorized | unauthorized | unknown |
|---|---|---|---|---|
| train | 27,923 | 27,231 | 692 | 0 |
| test | 7,758 | 6,517 | 1,241 | 0 |
| unknown (held out) | 5,818 | 0 | 0 | 5,818 |

| suite | fold | authorized | unauthorized | unknown | tools |
|---|---|---|---|---|---|
| banking | 1 | 2,958 | 668 | 1,281 | 11 |
| slack | 1 | 3,559 | 573 | 2,421 | 13 |
| travel | 0 | 6,677 | 200 | 1,373 | 29 |
| workspace | 0 | 20,554 | 492 | 743 | 27 |

Labels come from **observable evidence only**: each suite's *ordinary* tool set is read off the
recordings (functions called in episodes with no injection, plus injection-episode functions that
are neither a sink nor a credential access); the `injection_target_arg` rule fires when a
specific argument value is copied from the injected text **and** is absent from the trusted
request. An "unknown" third label is kept in a **separate file** and is never silently folded
into either class.

**Cross-suite leakage is measured, not assumed.** The pairwise Jaccard overlap of the ordinary
tool sets is:

```
banking  1.0  0.0  0.0      0.0
slack    0.0  1.0  0.0      0.0
travel   0.0  0.0  1.0      0.098039
workspace 0.0 0.0  0.098039 1.0
```

Maximum pairwise overlap is **0.098039** (travel ↔ workspace; shared: `cancel_calendar_event`,
`create_calendar_event`, `get_day_calendar_events`, `search_calendar_events`, `send_email`); all
other pairs are exactly 0.0. Because suites are **never split across folds**, the
**train/test cross-split overlap is 0.0** (train = {travel, workspace}, test = {banking, slack}).

**Honest caveat on label noise.** Two rules were dropped during construction because hand-auditing
~40 examples showed they fired on legitimate user requests (`privileged_change` on "invite Dora to
Slack" / "update my account info"; a naive `send_email` sink rule on "send a summary to the client
by email"). A residual failure mode remains: a sink call in a task whose request omits both the
action verb and the target can still be flagged. Every example therefore carries its `rule`, so the
corpus can be filtered or re-labelled without re-running the build.

### 4.3 The target we are still short of

The plan's target is **≥ 20,000 action decisions with ≥ 40 % negatives and a `benign_lookalike`
hard-negative class ≥ 25 % of all negatives**. The build reaches 41,499 decisions but only
**4.8 % negatives** (1,933/41,499), and it has **no** `benign_lookalike` class yet — mutating
authorized calls (change recipient / amount / account / date) to create hard negatives is the one
class that must be built by us; nothing public provides it. AgentDojo alone yields only ~300–800
authorized calls: it is a *seed*, not a dataset.

---

## 5. Benchmark and dataset acquisition plan

Ordered by expected value per unit of effort. Full table with licences and caveats in the
dossier, §8.

| # | Dataset | Size | Action-label yield | Why this one |
|---|---|---|---|---|
| 1 | **LLMail-Inject** (HF `microsoft/llmail-inject-challenge`, MIT) | 208,095 unique prompts | Low (message-level) | We hold a 20-item sample (`llmail_20.json`) = ~0.01 % of the corpus. The only large *human-written* attack corpus; synthetic sets cannot substitute. Use as a **detection-recall** benchmark, not an ASR benchmark — local undefended ASR is only 0.5–1.1 % |
| 2 | **NotInject** (HF `leolee99/NotInject`, MIT) | 339 benign samples | Low | **The direct instrument for the 17 pp FP problem.** Exposes trigger-word over-defense in ProtectAIv2/Fmops/DeepSet (< 60 % accuracy). 339 rows = a 1-hour job with outsized reporting value |
| 3 | **TaskTracker** (arXiv:2406.00799) | > 500,000 examples | Medium (text-level, but supplies **activations** → probe training) | **Highest-leverage single acquisition.** Directly attacks the ceiling effect: > 0.99 ROC AUC, beating PromptGuard and a proprietary classifier `[V]`. Train a probe on *our* SecJev-4B activations using their recipe — do not rebuild the toolkit |
| 4 | **AgentHarm** (HF `ai-safety-institute/AgentHarm`, ICLR 2025) | 110 base / 440 augmented, 104 tools | **High** — multi-step calls, per-step rubrics, **plus benign counterparts**, with a **private test split** | The best **paired authorized/unauthorized** action set available; use as a held-out generalization set |
| 5 | **AgentDojo** (own trajectories) | 97 tasks × 70 tools; ~300–800 authorized calls | High but small | The **only** source of attacker *ground truth*; the seed for the `benign_lookalike` negatives. Cost ≈ US$35 for 629 cases on GPT-4o `[V]` |
| 6 | **ASB** (arXiv:2410.02644) | ~90,000 cases, 400 attack tasks, 13 backbones | High | Best **cross-suite** generalization set; also supplies the reference FPR figure (0.200) for our FP budget |
| 7 | **APIGen-MT-5k** (HF, gated) | 5,000 multi-turn trajectories | High | Scale for authorization **pretraining**. ⚠️ **Never** for threshold calibration — it is built on τ-bench domains, so it would contaminate the split |
| 8 | **InjecAgent** (arXiv:2403.02691) | 1,054 cases, 62 attacker tools | High | Action-level attacker intent in a *different* tool taxonomy |
| 9 | **MCP-SafetyBench** (ICLR 2026) | 20 attack types, 5 domains | High | The forward-looking venue for an action monitor; also the source of the *p-value + effect-size* reporting style we should copy |
| 10 | **τ-bench / τ²-bench** | retail + airline, `pass^k` | Low (no injections) | **Utility benchmark**: proves recovered benign utility is not an AgentDojo artifact. `pass^k` exposes reliability collapse that `pass^1` hides |
| 11 | **R-Judge** (Findings of EMNLP 2024) | 8k+ samples | Medium–High | Schema = *action + risk category + safe/unsafe label + trajectory* — exactly our action monitor's schema |
| 12 | **Tensor Trust** (HF) | 126,808 attacks | None (prompt-level) | Cheap scale for detector **pretraining** only |
| 13 | **WildGuardMix** (CC-BY-NC-4.0 — **non-commercial**) | — | None | Benign pool for OOD/FP measurement, but the NC clause may block product use |
| 14 | **safe-guard / deepset / Gandalf HF sets** | 10,296 / 662 / 1,000 | None | Cheap benign/attack prompt-level halves for FPR sanity checks |
| 15 | **CyberSecEval 3 / 4** | v4 = 7,883 (PI = 1,004) | None | Static multilingual PI coverage — only if we claim multilingual robustness |
| 16 | **PINT** | 4,314 | — | ⛔ Test set is deliberately private. **Do not attempt** — documented so nobody spends time on it |
| 17 | **SEP / AttackGen / AgentMonitor(bench) / ToolSword / FATHOM** | — | — | ⛔ Could not be verified from a primary source. Either locate one or drop them from the report |

**Budget-aware order:** do **1, 2, 3, 4** first — together they cover the human-attack recall
axis, the false-positive axis, the ceiling-effect axis and the action-label axis, at negligible
cost and with no gated downloads. Then **5, 6** (AgentDojo we already have). Then **7, 9** if the
program needs multi-benchmark and MCP coverage.

---

## 6. Architectural alternatives to detection-by-classification

| family | representative work | what it buys | what it costs | verdict for this project |
|---|---|---|---|---|
| **Capability / dataflow / privilege control** (execution plane) | Progent `[V]`, CaMeL `[V]`, MCP protocol analysis (52.8 % → 12.4 % ASR, 8.3 ms overhead) `[V]` | Strict action boundaries; ASR 41.2 % → 2.2 % on AgentDojo; a crisp, well-posed label | Needs a policy language; can over-refuse on complex tasks | **Primary direction.** Our action monitor should *learn* the predicate Progent expresses by hand |
| **Segment-level intervention** | Only **6 of 41** surveyed defenses implement it `[V]` | Preserves the episode (utility under attack) while still blocking the bad call | Needs the episode to be resumable | The stated frontier; we now have the `redact` counterfactual machinery to measure it |
| **Input spotlighting / datamarking** | Spotlighting `[V]` | Cheap, model-agnostic | Only as strong as the marker | Pre-filter at best |
| **Representation-level detection** | TaskTracker activation deltas (> 0.99 ROC AUC) `[V]`, RepE `[V]`, ARGUS steering (CVPR 2026, "following user" vs "following injected" instructions *linearly separable*) `[V]` | Attacks the ceiling effect directly by reading activations rather than a label head | Needs activations and probe training | **Highest-leverage single acquisition** (§5, #3) |
| **Guard models** | PromptGuard, deepset, Gandalf, Semalith, HiveTraceGuard `[V]` | Zero training | FPRs of 0.15–0.47 on benign security-adjacent inputs `[V]`; InjecGuard shows < 60 % accuracy on NotInject `[V]` | Baseline to beat, not a solution |
| **Trust boundary / privilege separation** | CaMeL `[V]`, MCP origin-tagging `[V]` | Structural; defeats many payload manipulations | Large refactor | Long-horizon; the action monitor's design should already be compatible |

**The recommendation, in one line:** make the **action-authorization monitor** the primary
contribution, demote the tool-output detector to a secondary, low-cost pre-filter, and move the
scientific claim from "does this text contain an instruction?" (high base rate → the ceiling) to
"is this call authorized by the user's request?" (crisp label → measurable).

---

## 7. What this track produced, and what it did not

**Produced**

* A 10,571-word, 114-URL dossier with every source provenance-tagged, committed as
  `SOTA_DOSSIER.md`.
* A concrete, executable answer to each of the four questions: the benchmark acquisition plan
  (§5), the architectural alternatives (§6), the ceiling-effect statistics and the protocol that
  settles blindness-vs-miscalibration (§2), and the threshold-calibration honesty protocol (§3).
* The action-authorization dataset builder, with the leakage measurement and the honest label-noise
  caveat (§4) — **implemented in this submission**, together with the out-of-fold calibration and
  the cascade/redact machinery of Track 2.

**Not produced, and stated as such**

* The separation measurement itself (AUROC / AUPRC / TPR@FPR / Cohen's d / reliability curve).
  It needs the GPU scorer on the real checkpoint; this environment has no torch and no GPU. **The
  §2.3 protocol is written so that the first GPU run produces it.**
* The "0/900" provenance question (§2.1) is **unresolved**. Until it is, no ceiling-effect claim
  is published.
* The `benign_lookalike` hard-negative class (§4.3) is not built; without it the 4.8 % negative
  rate is far below the ≥ 40 % target.
* The intent judge on AgentDojo (Track 3) is **implemented and unit-tested but not evaluated on
  AgentDojo traces**, because that needs an AgentDojo-specific task-prompt mapping.
* A white-box adversary. `arXiv:2505.15738` shows fine-tuning-based defenses (StruQ, SecAlign)
  collapsing to **100 % / 96 % ASR** under Checkpoint-GCG `[V]`; our models are open-weight, so an
  optimization-based attack against SecJev-4B should be run before submission.
* No acquisition was performed — the environment is offline. §5 is a plan with sizes, licences and
  yield estimates, not a download log.
