# SOTA Dossier — Indirect Prompt Injection (IPI) Defense for LLM Agents

**Program:** SecJev-4B · **Harness:** AgentDojo + adaptive (defense-aware) attackers · **Models:** `gpt-oss-20b`, `qwen3-30b`
**Date:** 2026-11-01 · **Status:** research dossier (read-only synthesis)

**Provenance convention used throughout:**
`[V]` = URL fetched and content confirmed in this session · `[R]` = recalled from training knowledge, *not* fetched, treat as unverified · `[L]` = read from the local SecJev/AgentDojo artifacts in `/Users/mac/.raffle/worktrees/agent-786e4f-r3/jev-stageA`.

**Local artifacts inspected `[L]`** (these ground every claim about our own numbers):
`defenses.py` (88 lines), `score.py` (219), `analyze.py` (149), `replay.py` (81), `report.py` (75), `secjev_scorer.py` (37),
`data/results/gptoss20b.md`, `data/results/qwen30b.md`, `llmail_20.json`, archives `data/*.tgz`, `data2/*.tgz`.

---

## 0. Executive summary (read this first)

1. **The literature has moved past "detect the injection."** The 2026 SoK survey finds *no* defense that simultaneously achieves high security, high utility and low latency, and explicitly reports that text-level (i.e. tool-output-detection) defenses "fail to provide robust security guarantees against sophisticated payload manipulations" `[V]` (`arXiv:2602.10453`). Our own gpt-oss-20b 0/900 result is not an anomaly to be explained away — it is the predicted behaviour of the architecture.
2. **The action plane is where the tractable security is.** Progent (action-level privilege control policies) reduces AgentDojo ASR **41.2% → 2.2%** while preserving utility across AgentDojo, ASB and AgentPoison `[V]` (`arXiv:2504.11703`). Our local numbers already show the same ordering: action monitor 40.2% utility-under-attack vs tool-output detector 8.5% `[L]`.
3. **Our ~17pp benign-utility loss is a *scoping* bug, not a threshold bug.** The 2026 SoK reports that execution-level defenses "resort to aggressive 'early refusal' strategies that render the agent unusable for complex tasks," and that only **6 of 41** defense papers intervene at segment level `[V]`. Progent's answer is per-task scoped policies with fallbacks, not a global threshold.
4. **The statistics in the current analysis are not yet defensible.** `analyze.py:110-124` selects the `u95` threshold *and reports benign utility on the same benign episodes* `[L]`. The `u95` row therefore mixes an in-sample maximum with an out-of-sample security number. This must be split before publication.
5. **Our Run-1 numbers may not be the same quantity as the "0/900" finding — verify before writing.** The local artifacts report *defense-outcome* ASR (`secjev_tool` = 0.0% on standard AgentDojo at thr 0.5) `[L]`, which is **inconsistent** with a detector that never flags anything. The 0/900 "Rule 1" number is a *detector-level* detection rate on the injection subset. These are different quantities; the discrepancy is a P0 verification item (§7, item 0).

---

## A. Benchmarks & datasets beyond AgentDojo

### A.1 Baseline for comparison: AgentDojo `[V]`

`AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents` — NeurIPS 2024 Datasets & Benchmarks Track, **arXiv:2406.13352**, MIT license, `https://github.com/ethz-spylab/agentdojo`.

- **Scale:** 97 realistic user tasks, **629 security test cases**, 70 tools, 27 injection targets, 4 environments.
- **Per-suite structure** (this is the leakage map — see §C.3): Workspace 24 tools / 40 user tasks / 6 injection targets; Slack 11 / 21 / 5; Travel 28 / 20 / 7; Banking 11 / 16 / 9.
- **Granularity:** episode-level success, **plus** attacker-goal ground truth per injection task (this is what makes it usable as an *action*-label source — see §C.1).
- **Reported numbers:** state-of-the-art LLMs solve **< 66%** of tasks with no attack; best-performing agents are attacked successfully **< 25%** of the time; Slack-suite attacks reach **92% ASR** for GPT-4o. Running all 629 cases on GPT-4o ≈ **US$35**; the 97 utility cases ≈ **US$4**.
- **Why it matters here:** it is the only benchmark that ships *both* (a) a benign task with a machine-checkable ground truth and (b) an attacker goal with its own ground truth. That pairing is the raw material for an authorization label.

### A.2 Primary comparison table — agentic / execution-level benchmarks

| Name | Year / Venue | Core content & scale | Granularity | Attack families; vs. AgentDojo | Reported numbers (source) | Availability / license | Action-label yield | Relevance: tool-output detector vs action monitor |
|---|---|---|---|---|---|---|---|---|
| **AgentDojo** | 2024, NeurIPS D&B (arXiv:2406.13352) | 97 tasks, 629 security cases, 70 tools, 27 injection targets, 4 envs | Episode **+ attacker goal GT** | Heuristic + adaptive; the reference harness | Utility < 66% no-attack; attacks < 25% ASR best agents; Slack 92% ASR (paper) | MIT; github.com/ethz-spylab/agentdojo | **High** (attacker ground truth per injection task) | Both — but its suite overlap is the leakage risk (§C.3) |
| **InjecAgent** | 2024, ACL Findings (arXiv:2403.02691) | 1,054 test cases, 17 user tools, 62 attacker tools | **Action-level** (attacker tool invocation) | Instruction-relevance attacks (naive/context-aware/exploit-context) | Benchmark scores **lower** for defended agents vs undefended (per paper) | github.com/uiuc-kang-lab/InjecAgent | **High** — 62 attacker tools with ground-truth intent | Strong for action monitor; weak for a text detector (attacker tool ≠ injected text) |
| **ASB (Agent Security Bench)** | 2024/ICLR 2025 (arXiv:2410.02644) | 10 scenarios, 10 agent frameworks, 400+ tools, 400 attack tasks, 13 LLM backbones, **~90,000 test cases** | **Action-level** (tool-call triggers) | 4 families incl. memory poisoning; direct & indirect | Avg ASR **84.30%** (Mixed Attack); memory-poisoning avg ASR **7.92%**; LLM-based memory-poisoning detector avg **FNR 0.657 / FPR 0.200** (README) | github.com/agiresearch/ASB | **High** | Best cross-suite generalization set (scenarios ≠ AgentDojo's). Its published **FPR 0.200** is the reference point for our FPR budget |
| **MCP-SafetyBench** | 2025/ICLR 2026 (arXiv:2512.15163) | Real MCP servers, 5 domains (browser automation, financial analysis, location nav, repo mgmt, web search), **20 attack types** across server/host/user | **Action-level** (full execution traces, dual-label outcome) | Server-side tool poisoning, host-side, user-side; cross-server | Safety-prompt reduces weighted ASR **39.88% → 38.65%**, **NS** (p = 0.2908, Cohen's d = 0.31); significant for Malicious Code Execution **−21.54%** (p = 0.0016) and Credential Theft **−21.37%** (p = 0.0027) | github.com/xjzzzzzzzz/MCPSafety | **High** | Forward-looking action plane; the p/d reporting is a model for our own write-up |
| **AgentHarm** | 2025, ICLR | **110** base behaviors (**440** augmented), 11 harm categories, **104** tools, avg 3.53 functions/behavior; splits val 44 / public test 264 / private test 132 | **Action-level** (multi-step calls, per-step rubrics) | *Direct* harm requests + benign counterparts — **not** injection | Benchmark measures harmful task completion (paper) | HF `ai-safety-institute/AgentHarm`; public release = 44/66 public-test base behaviors (176 augmented) + 8/11 validation | **High**, and each behavior has a **benign counterpart** → natural hard negatives | **Best available paired authorized/unauthorized action set** for §C |
| **Agent-SafetyBench** | 2025 `[R]` | Agent safety score | Episode | — | "No agent exceeds **60%** safety"; **47.3%** reported in the ProtoAmp analysis `[V]` | Paper/repo (details unverified) | Low | Secondary |
| **SafeAgentBench** | 2024 (arXiv:2412.13178) | 750 tasks (450 hazardous), 10 hazards, 3 task types, 17 high-level actions | Episode, embodied | Hazard-aware planning (not injection) | — | safeagentbench.github.io | Low | Low (embodied) |
| **τ-bench / τ²-bench** | 2024 (arXiv:2406.12045) / 2025 (arXiv:2506.07982) | Retail + airline domains, user simulator, `pass^k` metric | Episode | No injections; measures **reliability** | GPT-4o **61.2%** retail / **35.2%** airline (`pass^1`); `pass^8` < 25% | github.com/sierra-research/tau-bench (MIT) | Low | **Utility benchmark** — the right place to prove recovered utility isn't an AgentDojo artifact |
| **ToolEmu** | 2024, ICLR (arXiv:2309.15817) | 36 toolkits / 311 emulated tools, 144 test cases, 9 risk types, ~3,000+ scored scenarios | **Action-level** (per-tool-call risk, severity) | Emulated/sandboxed; *no real injection* | Paper claims **68.8%** of identified failures are valid real-world risks (a third-party eval plan disputes this figure — treat as contested) | github.com/ryoungj/ToolEmu | **High** (risk labels per call) | Good for the *authorization* label schema; emulator fidelity is a caveat |

### A.3 Secondary table — detection corpora (no agent loop, no action labels)

| Name | Year | Scale | Granularity | Availability / license | Relevance to a tool-output detector |
|---|---|---|---|---|---|
| **LLMail-Inject** | 2025/EMNLP 2026 Findings | **208,095** unique prompts; **461,640** rows, 2.45 GB | Message-level, human-written | **MIT**; HF `microsoft/llmail-inject-challenge`; `aclanthology.org/2026.findings-emnlp.514.pdf` | **High** — the only large *human-written* attack corpus; our `llmail_20.json` is a 20-item sample `[L]` and is ~0.01% of the corpus |
| **BIPIA** | 2023 (arXiv:**2312.14197**) | 250 attacker goals × 5 tasks (Web QA, Email QA, Table QA, Summarization, Code QA), 30 text + 20 code attack types, 25 LLMs | Input-level | github.com/sunlab-uiuc/BIPIA | Medium — the canonical *input* benchmark; no action labels |
| **NotInject / InjecGuard** | 2024 (arXiv:2410.22770) | **339** benign samples (113 one-word + 113 two-word + 113 three-word) | Prompt-level benign-only | **MIT**; HF `leolee99/NotInject`; github.com/leolee99/InjecGuard | **Highest for the FP problem.** ProtectAIv2 / Fmops / DeepSet over-defense accuracy **< 60%** (near random); InjecGuard reaches over-defense **87.32%**, benign **85.74%**, malicious **77.39%** |
| **Tensor Trust** | 2023 (arXiv:2311.01011) | **126,808** attacks (69,906 distinct) + 46,457 defenses (39,731 distinct) | Prompt-level, red-team game | HF `qxcv/tensor-trust` | Medium — huge but non-agentic; good for detector pretraining |
| **PINT** (Lakera) | 2024 | **4,314** inputs | Prompt-level | **Test set deliberately NOT public** (Lakera keeps it private); repo ships harness without data | **Do not attempt to acquire.** Published leaderboard (third-party recall): deepset DeBERTa-v3 ≈ 62%, Lakera Guard ≈ 92.5% |
| **WildGuardMix** | 2024 `[R]` | prompt-level benign/harmful mixtures | Prompt-level | HF `allenai/WildGuardMix` (license unverified — AI2 datasets are commonly **CC-BY-NC-4.0**) | Medium — benign pool for OOD / FP measurement |
| **HackAPrompt** | 2023 `[R]` | ~Lakera/secureworks suite; used as an *attack type* in StruQ's table | Prompt-level | github.com secureworks/llm-injection-benchmark | Low — undefended ASR **52%** (Llama) / **38%** (Mistral), **0%** under StruQ `[V]` |
| **Gandalf** (Lakera) | 2024 | HF `Lakera/gandalf_ignore_instructions`: **1,000** rows, 1000 attack / 0 benign `[R]` | Prompt-level | HF (license unverified) | Low — no benign half |
| **deepset/prompt-injections** | 2023 `[R]` | **662** rows = 263 attack / 399 benign `[R]` | Prompt-level | HF | Medium — has a benign half, good cheap FP set |
| **xTRam1/safe-guard-prompt-injection** | 2024 (arXiv:2402.13064) | 10,296 examples; test split **2,060** = 650 attack / 1,410 benign `[R]` | Prompt-level | HF | Medium |
| **CyberSecEval 3 / 4** | 2024 (v2 arXiv:2404.13161, v3 arXiv:2408.01605) / 2025 (v4) | v4 total **7,883**: mitre 1,000; mitre_frr 750; instruct 1,916; autocomplete 1,916; multiturn_phishing 100; multilingual_prompt_injection 1,004; malware_analysis 609; threat_intelligence 588 | Prompt-level | MIT; `meta-llama/PurpleLlama` repo (datasets pinned at commit `fe05293`) | Low — static prompts, no agent loop |
| **SEP** ("Should It Be Executed or Processed") | — | **Could not verify.** No such benchmark surfaced in any search; likely a conflation with SecAlign / StruQ / SafeAgentBench. **Do not cite.** | — | — | — |
| **WorkArena** | 2024 `[R]` | ServiceNow enterprise web tasks | Episode | BrowserGym | **None** — contains no injections |
| **ToolSword** | 2024 (arXiv:2404.10757) `[R]` | tool-learning safety across 3 stages | Prompt-level | Repo | Low–medium; verify exact counts before citing |
| **AgentMonitor** | — | **Could not verify** a canonical benchmark by this name | — | — | Medium (as a *defense* named AgentMonitor) — verify |
| **Aegis / Aegis-AI** | 2024 `[R]` | NVIDIA Aegis AI Content Safety (v1/v2) | Prompt-level, benign+harmful | HF `nvidia/Aegis-AI-Content-Safety-Dataset-*` (license unverified) | Medium — safety-adjacent benign negatives |
| **AttackGen** | — | **Could not verify** | — | — | — |
| **WASP** | 2025 (arXiv:2504.18575) | Meta web-agent security env | Action-level | Repo | 16–86% adversarial execution rates `[V]` |
| **ProtoAmp / AttestMCP** | 2026 (arXiv:2601.17549) | 847 attack scenarios, 5 MCP server implementations | Action-level, protocol-level | Paper | MCP amplifies ASR by **23–41%**; AttestMCP cuts ASR **52.8% → 12.4%** at 8.3 ms median overhead; prompt-level "never pass data between servers" only cut cross-server ASR 61.3% → 47.2% vs AttestMCP's 8.7% `[V]` |

**Named-but-unverified — flag in the report, do not cite as fact:** `SEP`, `AttackGen`, `AgentMonitor` (as a benchmark), `ToolSword` details, `WildGuardMix` license, `FATHOM` (see §B).

---

## B. Architectural alternatives to detection-by-classification

### B.1 Capability / dataflow / privilege-control (the execution plane)

| Name | Year / Venue | Core mechanism | Granularity | Reported numbers (source) | Relevance to tool-output detector vs. action monitor |
|---|---|---|---|---|---|
| **CaMeL** ("Defeating Prompt Injections by Design") | 2025 (arXiv:2503.18813) | Capability-based design: LLM extracts *capabilities* from user request; deterministic control/dataflow enforces them; untrusted data never controls flow | **Action-level** | Paper reports strong reduction vs prompt-based defenses | **Action monitor.** The canonical "don't detect, constrain" answer. Also underpins Progent |
| **Progent** | 2025 (arXiv:2504.11703) | DSL for privilege-control policies over tool calls (tool + argument-level restrictions + fallback functions); LLM can auto-generate policies from the user query | **Action-level, per-task scoped** | AgentDojo ASR **41.2% → 2.2%**, utility preserved (no-attack & under-attack) across AgentDojo, ASB, AgentPoison `[V]` | **Action monitor — and the template for fixing our 17pp.** Scoping (per-task policy) instead of a global threshold is exactly the mechanism we lack |
| **Operationalizing CaMeL** | 2025 (arXiv:2505.22852) `[R]` | Practical deployment of CaMeL-style patterns | Action-level | — | Action monitor |
| **RTBAS** | 2025 (arXiv:2502.08966) `[R]` | Runtime tool-behavior allowlist | Action-level | — | Action monitor |
| **Encrypted Prompt** | 2025 (arXiv:2503.23250) `[R]` | Encrypt task descriptions; proxy decrypts for authorized calls | Action-level | — | Action monitor |
| **DataSentinel** | 2025 (arXiv:2504.11358) `[R]` | Game-theoretic detector with theoretical guarantee on FPR/FNR | Text-level | — | Tool-output detector (theoretical FPR framing) |
| **FATH** | 2024 (arXiv:2410.21492) | Authentication-based test-time defense: tags trusted data so injected content is distinguishable | Text-level | — | Tool-output detector. **`FATHOM` in our briefing appears to be a name conflation with FATH** — no `FATHOM` IPI-defense paper could be found. (A product "Fathom" claims 88.7% PI accuracy — commercial claim, not citable) |
| **Tool Result Parsing** | 2026 (arXiv:2601.04795) | Parse tool output into structured fields; check tool before use | Tool-output + action | Avg **ASR < 1%**, "lowest ASR to date"; undefended ASR > 20%, and **Spotlighting-with-delimiting still > 10% ASR** — i.e. "unusable in practice" `[V]` | **Both.** Strongest published tool-output-side result; directly comparable to our stop-mode detector |
| **CASCADE** (MCP) | 2026 (arXiv:2604.17125) | 3-tier cascade: regex/phrase/entropy → BGE embeddings w/ Llama3 fallback → output pattern filter | Text-level | On 5,000 samples: **95.85% precision, 6.06% FPR, 61.05% recall, 74.59% F1**; PI detection 84.2%, tool poisoning 59.9%, semantic attack 52.5% `[V]` | Tool-output detector. A realistic instance of the recall/FPR trade — useful as a "what real deployments get" anchor |
| **Min/Max firewall ("Are Firewalls All You Need?")** | 2025 (arXiv:2510.05244) | Two-firewall design ("Minimizer" + "Sanitizer") | Tool-output | Evaluated on AgentDojo, ASB, InjecAgent, τ-bench; **snippet-level only — verify exact ASR/utility in the paper body** | Tool-output detector. Multi-suite validation is the methodological standard we should copy |

### B.2 Input-spotlighting / datamarking

| Name | Year / Venue | Mechanism | Reported numbers (source) | Notes |
|---|---|---|---|---|
| **Spotlighting** | 2024, CAMLIS 2024 (arXiv:2403.14720) | Provenance encoding of retrieved data: delimiting, datamarking, encoding | ASR **> 50% → < 2%** overall; GPT-3.5 datamarking ~50–60% → **3.10%**; GPT-3-Text-003 40% → **0.00%**; encoding → **0.0%**; synthetic corpus of **1,000** injected docs `[V]` | Needs GPT-family behavior; **audited down to near-zero benefit on tool agents** — see `arXiv:2601.04795` ("Spotlighting with delimiting still >10% ASR") `[V]` |
| **StruQ** | 2024/25, USENIX Security 2025 (arXiv:2402.06363) | Structured queries: separate data channel + user/task separator + instruction-tuned | Llama undefended → StruQ: Completion-Real **96% → 0%**; TAP **97% → 9%**; GCG **97% → 58%**; utility 53.9% → **67.7%** (BIPIA model 26.0%) `[V]` | Requires fine-tuning the agent model |
| **Auditing defenses** | 2025 (arXiv:2505.15738) | Checkpoint-GCG against training-time defenses | ASR raised from ~0 to **100% on StruQ** and **96% on SecAlign** `[V]` | **Critical**: fine-tuning defenses (StruQ/SecAlign) collapse under white-box audit. Our open-weight models (`gpt-oss-20b`, `qwen3-30b`) are auditable — an adaptive attacker with gradients could break a fine-tuned SecJev. Plan for it |
| **SecAlign / SecAlign++** | 2024/25 `[R]` | Alignment for instruction/data separation | — (SecAlign audited to 96% ASR above) | Not fetched |
| **MyLLM / Code-Switch / T-Mark / MetaSploit / perplexity-based** | 2024–25 `[R]` | Various token/encoding watermarks and perplexity gates | — | Not fetched; do not cite numbers |

### B.3 Representation-level detection

| Name | Year | Mechanism | Reported numbers (source) | Notes |
|---|---|---|---|---|
| **TaskTracker** ("Are you still on track!? Catching LLM Task Drift with Activation Deltas") | 2024 (arXiv:2406.00799) | Contrast activations before/after ingesting retrieved text ("activation deltas"); logistic-regression probe + metric-learning (triplet) probe | **> 500,000** training examples; **> 0.99 ROC AUC** on 6 LLMs and on an OOD test set spanning jailbreaks, malicious instructions, unseen domains/styles; **outperforms PromptGuard and a proprietary black-box classifier** `[V]` | **This is the single highest-leverage item for §7(a).** It converts our "0/900 classification" ceiling into a measurable *separation* problem, and it uses the model's own hidden states — which SecJev-4B already has |
| **RepE** (representation engineering) | 2024/25 (arXiv:2502.19649) | Concept directions in activation space; DIM, linear probes, PCA, CCS; steering | Taxonomy + pipeline analysis (survey) `[V]` | Enables "steer away from following injected instructions" |
| **ARGUS** | CVPR 2026 | Orthogonal linear probes separating "following user instructions" from "following injected instructions"; utility-direction search + adaptive steering; detection→steering→post-filter in one forward pass | Finds the two behaviours **linearly separable**; best safety/utility/efficiency trade-off `[V]` | Model-level defense; multimodal variant |
| **ActSVD / JailbreakLens / INSTA / LATTE** | 2024–25 `[R]` | SVD-based / latent / jailbreak-specific detectors | — | Not fetched |

### B.4 Guard models (off-the-shelf classifiers)

| Model | Org / year | Sizes | Reported numbers (source) | Notes |
|---|---|---|---|---|
| **PromptGuard-2** | Microsoft, 2024 | 86M | IID **F1 0.539**, **recall 0.369**, hard-benign **FPR 0.1515**; a "high-precision" internal detector reached > 0.96 IID F1 yet flagged **38.3%** of benign security-adjacent inputs `[V]` (PIDS-Bench, arXiv:2609.15017); in another audit it showed **96% FPR** on AgentHarm-benign and PINT **F1 ≈ 0.0006** at 18.26 ms `[V]` (arXiv:2607.22545, 2609.01046) | **Warning label.** The wide spread across audits (F1 0.0006 → 0.96) is entirely explained by *which benign corpus* is used. We must specify our benign corpus |
| **Llama-Guard-3-8B** | Meta, 2024 | 8B | **FPR 0.063** on 208 benign agentic prompts vs Semalith's 0.000; **PINT F1 ≈ 0.06**; in a 15-model comparison F1 0.5249–0.6217 for the best models `[V]` | Current local baseline `piguard` beats our SecJev on ASR (0.3% vs 1.6% standard; 0.0% vs 5.0% adaptive) at 40.7% benign utility vs 50.5% `[L]` — i.e. **PIGuard is more aggressive, ours is more conservative** |
| **WildGuard** | AI2, 2024 | 7B/8B | **F1 0.6217 @ 99.67 ms** in a 15-model pooled comparison; pooled **FPR 0.268 / FNR 0.156**, field medians **FPR 0.227 / FNR 0.222** `[V]` (arXiv:2609.01046) | The best public reference point for "what FPR should we expect" |
| **ShieldGemma 2B / 9B** | Google, 2024/25 | 2B/9B | **F1 0.5249** (2B) in the 15-model comparison `[V]` | — |
| **Granite-Guardian** | IBM, 2024/25 | 2B/3.3-8B | PINT **F1 ≈ 0.32** (3.3-8B, fp16) `[V]` | — |
| **PolyGuard** | NVIDIA NeMo, 2024 | 8B | — `[R]` | — |
| **Semalith v1.4** | 2026 (arXiv:2607.22545) | 184M | **FPR 0.000** on 208 benign agentic prompts (vs Llama-Guard-3-8B 0.063) | Most relevant *recent* guard for the FP problem |
| **HiveTraceGuard-Pro** | 2026 (arXiv:2609.01046) | — | 15-model comparison, pooled FPR/FNR, latency/accuracy frontier | Use as the reference table for our benchmark table |
| **PIDS-Bench** | 2026 (arXiv:2609.15017) | — | Benchmark for detectors **under distribution shift**; reports IID vs hard-benign degradation | Directly addresses our ceiling question |
| **X-Guard / GuardReasoner / Magma / NeMo Guardrails / LlamaFirewall / HarmBench classifiers / Invariant Labs / Mialon** | 2024–26 `[R]` | Various | — | Not fetched; do not cite numbers |

### B.5 Trust boundary / privilege separation

- **MCP protocol analysis** `[V]` (`arXiv:2601.17549`): three protocol-level vulnerabilities — *absence of capability attestation* (servers can claim arbitrary permissions), *bidirectional sampling without origin authentication* (server-side prompt injection), *implicit trust propagation* in multi-server setups. AttestMCP (capability attestation + message authentication) reduces ASR **52.8% → 12.4%**, median overhead **8.3 ms/message**. **Implication:** the defense must sit at the tool boundary with *origin tagging*, not in the text.
- **Prompt infection, LLM→LLM** `[V]`: `arXiv:2410.07283` — injection propagates between agents; multi-agent inter-agent attack success reported at **84.6%** `[V]`.
- **Crescendo** `[R]` (`arXiv:2404.01233`) — multi-turn gradual escalation; **Mini-Dream** `[R]`, **AgentVillage** `[R]`, **A2A injection** `[R]`, **confused-deputy framing** — all unverified in this session. The confused-deputy framing is nonetheless the right mental model for our action monitor: the agent is a deputy acting on a principal's (the user's) authority, and an injection is an attempt to make the deputy act for a *different* principal.
- **Empirical support for "trust-boundary beats prompt-engineering"** `[V]`: adding *"Never pass data between different tool servers without explicit user confirmation"* to the system prompt reduced cross-server ASR only **61.3% → 47.2%**, versus AttestMCP's **8.7%** `[V]`.

---

## C. Building a labeled action-authorization dataset

### C.1 Label schema

For each proposed tool call `c` in a trajectory, with the *trusted* user request `u` in scope:

`label(c) ∈ {authorized, unauthorized, benign_lookalike}` plus `evidence` and `severity`.

- **authorized** — the call is consistent with `u` under the environment's ground truth. Source: calls appearing in a benign trajectory that satisfies AgentDojo's task checker, restricted to the reference solution path.
- **unauthorized** — the call advances the *attacker's* goal. Source: AgentDojo's `InjectionTask` ground truth (each injection task names a concrete attacker action, e.g. "send the Facebook security code to [attacker email]" ⇒ the negative call is `send_email` with the attacker address) `[V]`.
- **benign_lookalike** — *hard negatives*: authorized-tool-shaped but unauthorized-argument calls. Generated by mutating authorized calls (change recipient, amount, account, date). **This third class is where our 17pp benign-utility loss lives**, and it is the class almost no public dataset provides.

### C.2 Labeling procedure (recommended)

1. Replay benign trajectories with the model under test (`replay.py` already replays with a `cut` point — `replay.py:23-40` — exactly the interception point needed) `[L]`.
2. Auto-label with three independent oracles, then adjudicate: (a) AgentDojo ground truth; (b) a **Progent-style per-task policy oracle** (the paper's own prompt — *"Your task is to set permissions for each tool based on the user's query… set the parameter restrictions for this tool"* — yields tool+argument allowlists) `[V]`; (c) a strong judge model.
3. Human-audit a stratified 10% sample; report **Krippendorff's α** or Cohen's κ between oracles. Do not report a label set without an agreement number.
4. Keep a `provenance` field per label so that §D can run *paired* analyses and §E can run out-of-fold calibration without touching test labels.

### C.3 Cross-suite leakage: the concrete risk and the fold design

AgentDojo's four suites share tool families and task templates `[V]` (Workspace/Slack/Travel/Banking, 70 tools). A threshold tuned on Banking and evaluated on Travel shares (a) payment/booking tools, (b) the "read the file, then do the thing" template, and (c) the attacker's exfiltration channel. **Tuning on any subset and evaluating on another subset leaks.**

**Primary protocol — Leave-One-Tool-Family-Out (LOTO-TO), 6 folds.** Define families from the 70 tools: (F1) mail, (F2) messaging/channel, (F3) calendar/scheduling, (F4) file/document, (F5) web-browsing/fetch, (F6) money + booking. Six outer folds; in fold *i*, calibrate on F1..F6\{Fi} and evaluate **only** on Fi (all suites/tasks touching Fi).

**Secondary protocols, reported alongside (never instead of) LOTO-TO:**
- **LOSO** (4 folds): hold out one AgentDojo suite entirely.
- **Leave-one-template-out**: within every fold, hold out whole *task skeletons* so no user-goal template string appears in both calibration and evaluation.
- **Cross-model transfer**: calibrate on `gpt-oss-20b` episodes → evaluate on `qwen3-30b` episodes, and vice versa. This separates "model-specific scoring quirk" from "policy".
- **Cross-attack-family**: calibrate on standard AgentDojo attacks → evaluate on the adaptive/optimizer attacker already implemented in `adaptive.py` `[L]`.
- **Cross-corpus**: evaluate the frozen threshold on LLMail-Inject and NotInject, never used in calibration.

**Leak-sensitivity metric (must be reported):** `Δ = metric(LOSO) − metric(LOTO-TO)`. A large Δ means the threshold is exploiting suite-specific regularities, not learning authorization.

### C.4 Calibration techniques (ranked by suitability here)

1. **Split conformal risk control (CRC)** — `arXiv:2509.01455` describes it as: choose `λ̂ = inf{λ : (n·R̂_n(λ) + B)/(n+1) ≤ α}` `[V]`; the canonical source is Angelopoulos, Bates, Fisch, Lei & Schuster 2022 `[R]`. For our tool-output detector the loss is "benign tool output flagged" (monotone decreasing in τ) ⇒ a clean fit: pick the **largest** threshold whose calibration-set FPR satisfies the bound. For the action monitor the loss is "unauthorized action executed" — monotone in the blocking rule, also a fit. **Guarantee:** finite-sample, distribution-free, *conditional on exchangeability of the calibration and test episodes* — which is exactly what LOTO-TO/loso buys us.
2. **Learn-then-Test** (Angelopoulos, Bates, Candès, Jordan & Lei 2021) `[R]` for the stronger `(α, δ)` guarantee `P(R(λ̂) ≤ α) ≥ 1−δ`, needed if our risk estimator (the judge) is itself noisy — it is.
3. **Non-monotonic CRC** — if the judge's score is not monotone in the loss, use the algorithmic-stability variant `[V]` (Angelopoulos, X post, 2025) or fall back to LTT.
4. **Mondrian / bucketed conformal** — per attack-family or per tool-family thresholds; note the coverage guarantee is only *marginal* unless bucketed `[V]`.
5. **OOD detection on SecJev-4B's own hidden states** — Mahalanobis / energy score against a benign manifold. Because SecJev-4B is a transformer, this is free, and it converts the "ceiling" into a *separation* metric (§7a).
6. **Mixture-of-experts / risk-controlled abstention** — three bands: allow (τ_low), escalate-to-user-or-bigger-model (τ_low..τ_high), block (τ_high). **This is the direct mechanism for recovering the ~17pp**: legitimate-but-unusual calls land in the middle band and are not silently blocked.
7. **Not recommended:** a single global threshold; any threshold chosen by inspecting test-set utility.

---

## D. Statistics & methodology for this exact failure mode

### D.1 Wilson intervals and the trap

The codebase implements the Wilson score interval correctly (`analyze.py:24-31`, duplicated in `report.py:16-19`): for `p̂ = k/n`, `z = 1.96`,

```
d = 1 + z²/n;  c = (p̂ + z²/2n)/d;  h = z·√(p̂(1−p̂)/n + z²/4n²)/d
CI = [max(0, c−h), min(1, c+h)]
```

For `k = 0, n = 900` this yields **[0.0000, 0.00425]** (the familiar `3.84/n ≈ 0.0043` rule). So the team's "95% Wilson [0, 0.0041]" is right.

**The trap, stated precisely.** Overlapping 95% CIs do **not** license the conclusion "there is no difference"; they license only "this design is underpowered to resolve the difference at hand." The correct asymmetry is:

- *Non-overlapping* 95% CIs ⇒ the difference is significant at roughly the 0.005 level (conservative for independent samples).
- *Overlapping* 95% CIs ⇒ **inconclusive**, not "equivalent."

So our inability to distinguish "SecJev cannot detect" from "SecJev detects but weakly" is a statement about **power**, not about the model. And, critically, the converse inference is invalid too: failing to reject "no difference" is **not** evidence of equivalence. If the claim is "these two detectors are equivalent", it must be *proved* with an equivalence test (§D.3).

### D.2 The specific numbers: "0/900" vs "45/900"

**Unpaired comparison (the wrong test for our situation, computed here for reference):**
Wilson(0/900) = [0.0000, 0.00425]; Wilson(45/900) = [0.0376, 0.0663].
**Newcombe hybrid-score CI for the difference** (Newcombe 1998 method 10, combining the two Wilson intervals):
`p̂₁ − p̂₂ = −0.0500`, CI = **[−0.0663, −0.0369]**, excludes 0 ⇒ significant. Fisher exact would likewise reject (p ≪ 0.001). So *0/900 vs 45/900 is a real, detectable difference* — no ambiguity at all.

**But our comparison is paired.** The SecJev detector and the LLM judge score the **same 900 episodes**, so the correct test is **McNemar (exact, on discordant pairs)**: the statistic is `b + c` (episodes where exactly one detector flags), and the asymptotic SE of the difference is ≈ `√(b+c)/n` rather than the unpaired `√(p(1−p)/n)`. With `b + c ≈ 45` discordant episodes, SE ≈ `√45/900 = 0.0075` — **roughly half** the unpaired SE at comparable rates. McNemar is therefore markedly more powerful, and it is the test that uses the information we actually have.

**MDE / power at n = 900** (normal approximation, two independent proportions, α = 0.05 two-sided, 80% power; `MDE ≈ 2.802·√(p(1−p)/900)`):

| True detection rate p (vs 0) | Unpaired MDE | Detectable? |
|---|---|---|
| 0.5% | ≈ 0.5% | yes |
| 1% | 0.93% | yes |
| 5% | 2.04% | yes |
| 10% | 2.80% | yes |
| 20% | 3.74% | yes |
| 30% | 4.28% | yes |
| 50% | 4.67% | yes |

**Power against the literal null** (one-sided exact, `H0: p = 0`): a single detection in 900 already rejects at α ≈ 1/901; and a true rate of 0.5% is detected with power ≈ 94% (`1 − P(X ≤ 1 | n=900, p=0.005)`).

**Consequence — this reframes the finding.** n = 900 has ample power to resolve a true detection rate of ~0.2–0.5%. Therefore a 0/900 result means one of two very different things, and they have opposite implications:

- **(i) The score has no ranking signal** (benign and injected score distributions coincide). ⇒ The detector is *blind*, and no threshold will fix it. This is the "model cannot detect" hypothesis.
- **(ii) The score ranks fine but never crosses the 0.5 argmax threshold** (the model's own label head is mis-calibrated for this task). ⇒ The detector is *weakly informative*, and a calibration fix suffices.

**These are distinguishable, and we have not distinguished them.** That is the single most important measurement gap in the project (§7a).

### D.3 Ceiling-effect-safe measurement protocol

Replace "rate at a fixed threshold" with "separation at a fixed false-positive rate":

1. **Continuous scores, not argmax labels.** Dump the full score for every benign and injected episode. Everything below is computed from those dumps.
2. **AUROC and AUPRC** of benign vs injected, with **bootstrap CIs** (BCa, 10,000 resamples over episodes). AUPRC is the primary metric because the benign:attack ratio in AgentDojo tool outputs is heavily imbalanced.
3. **TPR at fixed FPR** — report TPR@1%FPR and TPR@5%FPR, plus the **full ROC curve**. This is the metric that is invariant to the decision threshold and therefore *not* subject to the ceiling.
4. **Cohen's d** (and Hedges' g) between the benign and injected score distributions. d ≈ 0 means no separation; d ≥ 0.8 is a large effect that a badly-set threshold could hide entirely.
5. **Reliability/calibration curve** of the score against the binary ground truth, plus **ECE**. A model that is *well-calibrated but uninformative* looks very different from an *informative but mis-thresholded* one; the reliability curve separates them.
6. **Positive controls at three difficulty tiers** to distinguish blindness from weakness:
   - *Tier 1 (trivially visible)*: verbatim "ignore previous instructions" — a detector failing here is architecturally broken.
   - *Tier 2 (visible but obfuscated)*: zero-width, whitespace, homoglyphs — the team already has `ood_zero_width`.
   - *Tier 3 (invisible / semantic)*: metadata fields, quoted replies, bland workflow notes — the team already has these, and they are the right set for measuring the ceiling.
   The team already ships `make_paraphrases.py` and a `tool_knowledge` control set in the qwen3-6b archives `[L]` — both should be used here.
7. **Paired bootstrap** over episodes for *all* between-detector differences (ΔAUROC, ΔTPR@1%FPR, ΔASR), resampling episodes not rows.
8. **MDE and power for every headline comparison**, computed with an exact power function (e.g. `statsmodels.stats.power.NormalIndPower` or a bootstrap-based power curve) rather than the normal approximation above.
9. **Pre-specify the primary endpoint.** Right now "detection rate on the injection subset" and "ASR under attack" are used interchangeably in conversation; they are different quantities, and the local artifacts mix them (`report.py:8` lists both `secjev_tool` and `secjev_action_meta` in one "Detector" column, though one is a text classifier and the other an action classifier).

### D.4 The equivalence question

If the report wants to say "SecJev-4B's detector is *equivalent to* an LLM judge", use **TOST** (Schuirmann 1987; Newcombe's hybrid-score CI is the recommended interval method, and SAS `PROC FREQ` implements `RISKDIFF METHOD=NEWCOMBE` with `EQUIV` and TOST) `[V]`:

- `H0: p₁ − p₂ ≤ −Δ` or `H0: p₁ − p₂ ≥ +Δ`; `H1: −Δ < p₁ − p₂ < Δ`.
- Reject if the **90% CI** (equivalently, both one-sided tests at α = 0.05) lies entirely inside `[−Δ, +Δ]`.
- Recommended margins for our setting: **Δ = 5 pp** for benign-utility loss, **Δ = 2 pp** for ASR difference, **Δ = 5 pp** for detection rate.

**Do not** use TOST on a 0/900 vs 0/900 comparison with a 5pp margin: with both at 0, the 90% CI is [−0.0043, +0.0043], which *is* inside ±5pp, so TOST would **successfully prove equivalence** at the 5pp margin — i.e. we could honestly claim "SecJev-4B's detector and the LLM judge are statistically indistinguishable at the 5-percentage-point level on this injection subset." That is a legitimate and publishable claim, and it is *stronger* than "we don't know". But it must be stated as equivalence-within-margin, not as "both are equally bad" — equivalence does not imply uselessness.

---

## E. Threshold calibration honesty

### E.1 Why Run 1's `u95` is optimistic, and by how much

`analyze.py:110-124` selects the `u95` threshold as the smallest threshold achieving ≥ 95% of undefended benign utility **on the same benign episodes whose utility is then reported** `[L]`. This is a *maximisation over a swept hyper-parameter, evaluated in-sample*. Consequences:

- **Bias direction:** the reported benign utility at `u95` is an **in-sample maximum**, hence **optimistically biased**. The reported *ASR* at `u95` is out-of-sample (AgentDojo attack episodes, n = 921/949) and therefore honest. **The `u95` row therefore mixes an in-sample number with an out-of-sample number in a single table**, which is the single most likely reviewer objection in the whole report.
- **Bias magnitude:** with `m` candidate thresholds swept over `n = 91` (gpt-oss) or `n = 97` (qwen) benign episodes, the maximum of `m` noisy binomial estimates is upward-biased by roughly the upper order statistic of `m` draws. With `m ≈ 20` thresholds and `n ≈ 91`, the expected selection inflation is on the order of **several percentage points of utility** — i.e. **the entire size of the effect we care about**. This must be measured, not estimated by hand: report `utility(in-sample tuned) − utility(out-of-fold tuned)` (§E.3).
- **The FP side is inflated, the security side is deflated.** A threshold chosen to maximise benign utility on a benign-only set is, by construction, a threshold with a high FPR on *that* benign set. The `u95` operating point makes this concrete in our own data `[L]`:

| Model | Operating point | Threshold | Benign utility (n=91/97) | Standard AgentDojo ASR (n=921/949) | Utility under attack |
|---|---|---|---|---|---|
| gpt-oss-20b | native | 0.50 | 50.5% | **1.6%** | 40.2% |
| gpt-oss-20b | u95 | 0.92 | **63.7%** | **18.5%** | 51.0% |
| gpt-oss-20b (tool) | native | 0.50 | 56.0% | 0.0% | 8.5% |
| gpt-oss-20b (tool) | u95 | 0.99 | 64.8% | 0.1% | 8.5% |
| qwen3-30b | native | 0.50 | 54.6% | **2.7%** | 42.9% |
| qwen3-30b | u95 | 0.88 | **69.1%** | **17.3%** | 51.5% |
| qwen3-30b (tool) | u95 | 0.89 | 69.1% | 0.0% | 6.6% |

*(undefended reference: gpt-oss 67.0% benign / 24.5% ASR / 52.9% UUA; qwen 72.2% / 26.4% / 55.2%)* `[L]`

The `u95` action threshold buys **+13.2pp** benign utility and **+10.8pp** utility-under-attack, and pays **+16.9pp ASR** (gpt-oss) / **+14.6pp** (qwen). The two models agree closely on the ASR cost of the u95 threshold (18.5% vs 17.3%), which is reassuring for reproducibility — and note the striking symmetry: the u95 action threshold's ASR (**~18%**) is essentially the same as the **undefended** ASR on qwen (26.4%) minus a little, i.e. at `u95` the action monitor has thrown away most of its security advantage. **The `u95` row is a legitimate operating point on a Pareto curve, but it must be reported as such, with out-of-fold numbers.**

### E.2 The correct protocol

1. **Never tune and evaluate on the same episodes.** Split benign episodes into `train` (threshold selection) / `val` (early stopping) / `test` (reported only).
2. **Nested / out-of-fold calibration.** Outer loop = held-out tool family (§C.3); inner loop = threshold search on the remaining families. Report the pooled out-of-fold estimate.
3. **Pre-register the threshold *rule*, not the value.** E.g. *"τ* = the smallest threshold on the calibration folds such that the Wilson upper 95% bound on the benign FPR is ≤ 0.02"*. Pre-registering the rule (not the number) prevents threshold-shopping and is publishable as a protocol.
4. **Cross-suite and cross-model thresholds** (§C.3) must be reported side by side, not just the best one.
5. **Report the generalization gap explicitly:** `Δ_gen = utility(τ tuned in-sample) − utility(τ tuned out-of-fold)`. Also `Δ_gen` for FPR and for ASR.
6. **Conformal risk control replaces ad-hoc threshold search** (§C.4) and gives a finite-sample guarantee *conditional on exchangeability* — which the LOTO-TO design buys, and which an in-sample sweep does not.
7. **Treat the benign set as the scarce resource.** It is `n = 91` (gpt-oss) / `n = 97` (qwen) — far too small for a 20-point threshold sweep plus a reported estimate. This is the *real* reason Run 1 was optimistic, and the fix is **more benign data**, not a different statistic.

### E.3 Additional honesty issues found in the local artifacts `[L]`

- **"Standard AgentDojo (in SecJev training)" is an in-sample flag** (`report.py:5`). Standard-AgentDojo numbers for SecJev are **training-set** numbers and must be labelled as such in every table. The out-of-sample evidence is the five OOD attack sets — and there SecJev holds up well (`secjev_tool` ASR 0.0–0.7%, `secjev_action_meta` 0.0–1.2%), so this is a framing fix, not a performance problem.
- **The "LLMail human attacks" set has a near-zero base rate on our models**: undefended ASR is **0.5%** (gpt-oss) / **1.1%** (qwen) `[L]`. Every defense trivially scores ~0%. This set therefore cannot discriminate defenses *as an ASR benchmark* on our models — it is useful only as a **detection-recall** benchmark (or after the agent is made more attackable). Reporting "0.0% ASR on LLMail" without the 0.5% undefended base rate would be misleading.
- **Different `n` per attack row** (921 / 917 / 923 / 927 / 901 / 796 for gpt-oss) because agent errors are filtered differently per set. Every table must print `n` (the report already does) and comparisons across rows must account for it.
- **The adaptive-attack table has n = 60 pairs**, so its Wilson intervals are wide (e.g. 5.0% → [1.7, 13.7]). Differences among defenses at this n are almost never significant; **do not rank defenses on the adaptive table**. Use it for "no defense breaks the monitor," not for ordering.

---

## 7. Recommended architecture for this project

### 7.0 P0-item 0 — resolve a consistency problem before writing anything

The local artifacts report **defense-outcome ASR** (`secjev_tool` = **0.0%** on standard AgentDojo at thr 0.5, and 0.0–0.7% across all five OOD sets) `[L]`. A detector that scored **0/900** on the injection subset could not produce 0.0% ASR, because an undefended ASR of 24.5% means the attacks *do* work against this model. Therefore **the "0/900 Rule 1" number and the artifact numbers are different quantities** — most likely a *detector-level* label-accuracy measurement against a *different* SecJev checkpoint, or against a different question type than the `secjev_tool_request` / `noul` / `"true"` path that `score.py:42-68` actually uses.

**Action:** locate the exact script/artifact that produced "0/900", record the checkpoint, the question type, the ground-truth key, and the episode list. Until this is resolved, **no claim about the ceiling effect can be published**, and §7(a) below may already be moot.

### 7.1 P0 — Break the ceiling effect  *(this is the answer to question (a))*

**P0.1 Measure separation, not thresholded accuracy.** (Full protocol: §D.3.) Dump continuous scores for all benign and injected episodes; report AUROC, AUPRC, TPR@1%/5% FPR, Cohen's d, and the reliability curve, with paired bootstrap CIs. If AUROC ≈ 0.5, the detector is blind (architecture problem). If AUROC ≥ 0.8 but the argmax rate is 0, it is a **threshold/calibration** problem, and the fix is cheap. **This one measurement decides the project's direction and costs one evaluation pass.**

**P0.2 Use SecJev-4B's hidden states, not just its label head.** Train a linear probe / Mahalanobis OOD scorer on benign vs injected tool outputs, in the TaskTracker style: contrast activations *before* and *after* ingesting the tool output (activation deltas), which reported **> 0.99 ROC AUC** on 6 LLMs with an OOD test set, beating PromptGuard and a proprietary classifier `[V]`. TaskTracker's released toolkit includes activations from 6 models and >500K examples — **use it, do not rebuild it.** This is the single highest-leverage acquisition in the whole plan (§8).

**P0.3 Change what is being detected.** The tool-output detector's question ("does this text contain an instruction aimed at the agent?") has a high base rate on legitimate tool output — which is exactly why guard models show FPRs of 0.15–0.47 on benign security-adjacent inputs `[V]`. The action monitor's question ("is this call authorized by the user's request?") has a crisp, well-posed label. **Move the primary scientific claim to the action plane.**

**P0.4 Fix the statistics before any new result.** Add to the harness: Newcombe hybrid-score CIs for all proportion differences, McNemar exact for paired detector comparisons, TOST for any equivalence claim, paired bootstrap for ΔAUC, and an MDE/power line under every headline. The current harness reports Wilson intervals only (`report.py:16-19`) — no tests at all.

### 7.2 P0 — Prioritize the action monitor, and recover the ~17pp  *(answer to (b))*

**Recommendation: make the action-authorization monitor the primary contribution; demote the tool-output detector to a secondary, low-cost pre-filter (or a redactor).**

Evidence for the pivot:
- Local: `secjev_action_meta` preserves **40.2%** (gpt-oss) / **42.9%** (qwen) utility under attack vs `secjev_tool` **8.5%** / **6.6%** `[L]`. Adaptive ASR 5.0% / 1.7% vs 3.3% / 8.3% `[L]`.
- Literature: Progent reaches **41.2% → 2.2% ASR on AgentDojo** while preserving utility, on **three** benchmarks `[V]`.
- 2026 SoK: text-level defenses "fail to provide robust security guarantees"; execution-level defenses "can enforce strict action boundaries" but "often … resort to aggressive 'early refusal' strategies that render the agent unusable for complex tasks" `[V]` — **that is a precise description of our 17pp.**

**Honest baseline caveat that must go in the report:** at the `native` threshold, **PIGuard beats SecJev on both axes** — gpt-oss: PIGuard benign **40.7%** / standard ASR **0.3%** / adaptive **0.0%**, vs `secjev_action_meta` **50.5%** / **1.6%** / **5.0%** `[L]`. So SecJev buys ≈ +10pp benign utility at a cost of 1.6–5.0pp ASR. That is a defensible operating point, but it must be stated as a trade, not as a win. Worse, on gpt-oss-20b the **LLM judge dominates our action monitor** (benign **67.0%** — identical to undefended — standard ASR **0.2%**, adaptive **6.7%**) `[L]`, while on qwen3-30b the same judge collapses (benign **20.6%**) `[L]`. **The judge is model-dependent; SecJev is not.** That asymmetry is SecJev's strongest argument and should be framed as such.

**How to recover the ~17pp (ranked):**

1. **Replace the global threshold with per-task scoped policies (Progent-style).** Generate a per-task allowlist of tools *and argument constraints* from the user request; block only calls outside it. Our current design asks one global question ("is this action authorized?") over a heterogeneous tool space, which is precisely the mis-specification that produces family-dependent FPRs. **This is the primary fix.**
2. **Abstain instead of block (three-band risk-controlled decision).** allow / *repair-or-ask* / block. A legitimate-but-unusual call lands in the middle band and costs a question, not a failed task. This is where the 17pp actually lives.
3. **Repair, don't block.** When a call is out-of-policy but a near-miss in-policy call exists, substitute it (Progent's fallback function). Segment-level intervention — which only **6 of 41** published defenses implement `[V]` — is the stated frontier.
4. **Diagnose the FP taxonomy before fixing.** Categorise every false positive into: (i) legitimate reads, (ii) legitimate writes, (iii) **legal-but-not-on-the-shortest-path intermediate calls**, (iv) judge-model error, (v) legitimate calls that *look* like an attack (wrong-recipient). Category (iii) is probably the largest and is fixed by a *plan-level allowance set* (calls appearing in the known solution path), not by lowering τ.
5. **Add the benign-but-scary corpora now:** NotInject (339 benign trigger-word prompts, MIT) `[V]`, AgentHarm's benign counterparts, and the team's own `tool_knowledge` control set. Without these, the FPR is unmeasured.
6. **Add a plan-level allowance** derived from AgentDojo ground truth: allow any call on the task's reference solution path, regardless of the judge's opinion.

### 7.3 P0 — Build the action-authorization calibration dataset without cross-suite leakage  *(answer to (c))*

**Sources** (in order of value):
1. **AgentDojo benign trajectories** → *authorized* labels (reference-solution-path calls).
2. **AgentDojo injection tasks** → *unauthorized* labels (the attacker goal's concrete action). This is the unique asset of AgentDojo.
3. **Mutated authorized calls** → *benign_lookalike* hard negatives (change recipient / amount / account / date). **This class must be built by us; nothing public provides it.**
4. **AgentHarm** (110 base / 440 augmented behaviors, 104 tools, benign counterparts, public+private splits) `[V]` → cross-domain authorized/unauthorized pairs with private-test held out for contamination control.
5. **APIGen-MT-5k** (5,000 multi-turn trajectories with explicit `function_call` / `observation` turns) `[V]` → scale, for *pretraining/fine-tuning* only — **never for threshold calibration** (it is built on τ-bench domains).

**Leakage controls:** the 6-fold LOTO-TO design (§C.3), plus leave-one-template-out, plus cross-model and cross-attack-family transfer, plus the leak-sensitivity metric `Δ = metric(LOSO) − metric(LOTO-TO)`.

**Target size:** ≥ **20,000** action decisions, ≥ 40% negatives, with the `benign_lookalike` class ≥ 25% of all negatives. AgentDojo alone yields only ~300–800 authorized calls — it is a *seed*, not a dataset.

**Label quality:** dual-oracle (AgentDojo GT + Progent-style policy oracle) + judge, with a 10% human audit and an inter-annotator agreement statistic published.

### 7.4 P0 — Cross-suite out-of-fold threshold calibration protocol (concrete)  *(answer to (d))*

**Fold assignment — Leave-One-Tool-Family-Out, 6 outer folds:**

| Fold | Held-out family | AgentDojo suites affected |
|---|---|---|
| 1 | mail (send/receive email, drafts) | Workspace |
| 2 | messaging / channel (Slack post, reply, invite) | Slack |
| 3 | calendar / scheduling (events, availability) | Workspace |
| 4 | file / document (drive read/write, files) | Workspace, Slack |
| 5 | web-browsing / fetch (web page retrieval) | Slack, Travel |
| 6 | money + booking (transfer, book flight/restaurant/car) | Banking, Travel |

In outer fold *i*: tune τ on folds ≠ *i*; evaluate **only** on family *i* across every suite that uses it. Run the 4 **LOSO** folds in parallel as the weaker secondary criterion.

**Procedure per outer fold:**
1. *Inner*: 5-fold grouped CV within the 5 remaining families, **grouped by task-template id**, to select τ by the pre-registered rule (*smallest τ with Wilson-UCL(benign FPR) ≤ 0.02*).
2. *Apply* τ to the held-out family; record benign utility, benign FPR, ASR, and utility-under-attack.
3. *Aggregate*: mean ± 95% CI (paired bootstrap over episodes, BCa, 10,000 resamples) per metric across the 6 folds, **and the worst fold**.

**Statistics to report (per fold and pooled):**
- Benign utility: Wilson CI; **Newcombe** risk difference vs undefended.
- Benign FPR at τ: Wilson CI.
- ASR: Wilson CI; **Newcombe** risk difference vs undefended and vs the `native` threshold.
- Paired defense comparisons on identical episodes: **McNemar exact** (report `b`, `c` discordant counts, not just p).
- **Equivalence:** TOST on `utility(defended) − utility(undefended)` with **Δ = 5pp** (claim: "utility loss ≤ 5pp"); TOST on `ASR(defended) − ASR(undefended)` with **Δ = 2pp**; TOST on detection rate with **Δ = 5pp** (this is the §D.4 equivalence claim).
- Calibration: reliability curve + ECE; conformal risk–coverage curve with achieved selective risk at target α and **violation frequency across bootstrap resamples** `[V]`.
- **Generalization gap** `Δ_gen` for utility, FPR and ASR (in-sample-tuned minus out-of-fold).
- **MDE** at 80% power for every headline comparison.
- **Leak sensitivity** `Δ = metric(LOSO) − metric(LOTO-TO)`.

### 7.5 P1 — Secondary priorities

- **P1.1 Evaluate `redact` mode.** It is implemented in `defenses.py:33-46` but **neither `replay.py` nor `analyze.py` models it** — `replay.py` implements abort semantics ("model output empty: aborted") and `analyze.py` only iterates `native`/`u95` thresholds `[L]`. A redact run exists in the qwen3-6b dev archive (`runs/dev/ii_tool_secjev_tool_0.5_redact.jsonl`) `[L]` but has **no scorer**. Redact is the mode most likely to beat stop-mode on utility-under-attack (it preserves the episode) while keeping the security benefit; **it should be the default tool-output mode going forward**, and it needs a scorer: `replay.py` must treat the flagged call as *not executed*, inject a fixed notice, and continue — the `outcome()` function (`replay.py:40-60`) already has the `cut` mechanism needed.
- **P1.2 Add an in-band "unknown" outcome to the SecJev label set.** The current label space forces a binary call on every tool output; a third state ("inconclusive → escalate") is the data-level precondition for the three-band decision of §7.2(2).
- **P1.3 Cross-validate the action monitor against Progent.** Port Progent's policy language (or `secagent`) and run it head-to-head on the same AgentDojo episodes. If Progent's 2.2% ASR holds at our models' utility, our story becomes "SecJev-4B learns the authorization predicate that Progent expresses by hand" — a far stronger paper.
- **P1.4 Prepare for a white-box adversary.** Our models are open-weight; `arXiv:2505.15738` shows fine-tuning-based defenses (StruQ, SecAlign) collapsing to **100% / 96% ASR** under Checkpoint-GCG `[V]`. Run an optimization-based attack against SecJev-4B before submission.
- **P1.5 Extend to MCP.** MCP-SafetyBench (20 attack types, 5 domains, dual-label execution outcomes) `[V]` and the protocol analysis (AttestMCP, **52.8% → 12.4% ASR**, 8.3 ms overhead) `[V]` are the natural next venue, and "origin-tagged tool results + capability attestation" is a design our action monitor can adopt directly.

### 7.6 P2 — Longer-horizon

- **P2.1** Fine-tune SecJev-4B (or a probe on top of it) on the §7.3 authorization dataset; report the train/test gap and the cross-family generalisation curve.
- **P2.2** Representation-level defense (ARGUS-style steering on "following user instructions" vs "following injected instructions", which the CVPR 2026 work finds **linearly separable** `[V]`) — model-level, complements the execution-level monitor.
- **P2.3** Split conformal risk control with Mondrian buckets per attack family, with the exchangeability caveat stated (coverage is marginal unless bucketed).
- **P2.4** Publish the *protocol* (leakage-controlled fold design, MDE, TOST, separation metrics) as a reusable benchmark contribution — the 2026 SoK's finding that "no tested defense simultaneously achieves high trustworthiness, high utility, and low latency" `[V]` means the field needs **better measurement**, not another detector.

---

## 8. Benchmark and dataset acquisition plan

Ordered by expected value per unit of effort. "Action-label yield" = expected number of *usable tool-call-level authorization labels* after filtering.

| # | Dataset | Where | Size | Action-label yield | License / caveats | Why this one |
|---|---|---|---|---|---|---|
| **1** | **LLMail-Inject** | HF `microsoft/llmail-inject-challenge`; paper `aclanthology.org/2026.findings-emnlp.514.pdf` | **208,095** unique prompts; 461,640 rows; 2.45 GB | **Low** (message-level; needs a host agent to convert to actions) | **MIT** `[V]` | We have a **20-item sample** (`llmail_20.json`) = ~0.01% of the corpus `[L]`. It is the only large *human-written* attack corpus; the synthetic/heuristic sets cannot substitute. Note the local undefended ASR on this set is only 0.5–1.1% `[L]` ⇒ use as a **detection-recall** benchmark, not an ASR benchmark, until we have a more attackable agent |
| **2** | **NotInject** (InjecGuard) | HF `leolee99/NotInject`; `github.com/leolee99/InjecGuard`; arXiv:2410.22770 | **339** benign samples (113×3) | **Low** (prompt-level benign only) | **MIT** `[V]` | **The direct instrument for the 17pp FP problem.** Exposes trigger-word over-defense in ProtectAIv2/Fmops/DeepSet (< 60% accuracy). 339 rows is a 1-hour job with outsized reporting value |
| **3** | **TaskTracker** (activations + probes) | arXiv:2406.00799; released toolkit (dataset + activations for 6 LLMs + probes) | **> 500,000** examples | **Medium** (text-level, but supplies *activations* → probe training) | arXiv license; **check the repo's terms** | **Highest-leverage single acquisition.** Directly attacks the ceiling effect: > 0.99 ROC AUC, beats PromptGuard and a proprietary classifier `[V]`. Train a probe on *our* SecJev-4B activations using their recipe |
| **4** | **AgentHarm** | HF `ai-safety-institute/AgentHarm`; ICLR 2025 | 110 base / **440** augmented behaviors, 11 harm categories, 104 tools; public release 44/66 public-test base (176 aug.) + 8/11 val (32 aug.) | **High** — multi-step calls, per-step rubrics, avg 3.53 functions/behavior, **plus benign counterparts** | HF terms — **read the card before use** | The best **paired authorized/unauthorized** action set available, with a **private test split** for contamination control. Use as a held-out generalization set |
| **5** | **AgentDojo** (own trajectories) | `github.com/ethz-spylab/agentdojo`; 629 security cases; 97 tasks | 97 tasks × 70 tools; ~300–800 authorized calls + attacker-ground-truth negatives | **High** (but small) | **MIT** `[V]` | The **only** source of attacker *ground truth*; the seed for the `benign_lookalike` hard negatives. Cost ≈ US$35 for 629 cases on GPT-4o `[V]` |
| **6** | **ASB** | `github.com/agiresearch/ASB`; arXiv:2410.02644 | ~**90,000** test cases, 400 attack tasks, 10 scenarios, 13 LLM backbones | **High** (tool-call triggers) | arXiv + repo license — **verify** | Best **cross-suite** generalization set (scenarios ≠ AgentDojo's). Also supplies the reference FPR figure (0.200) for our FP budget |
| **7** | **APIGen-MT-5k** | HF `Salesforce/APIGen-MT-5k`; arXiv:2504.03601 | **5,000** multi-turn trajectories, `function_call`/`observation` turns, retail + airline | **High** (the largest public agentic action corpus) | **Gated** — accept terms + login `[V]` | Scale for authorization **pretraining**. ⚠️ Never use for threshold calibration (τ-bench-domain overlap ⇒ contamination risk) |
| **8** | **InjecAgent** | `github.com/uiuc-kang-lab/InjecAgent`; arXiv:2403.02691 | **1,054** test cases, 17 user tools, **62 attacker tools** | **High** (attacker tool invocations) | MIT — **verify** | Action-level attacker intent in a *different* tool taxonomy; complementary to AgentDojo's 4 suites |
| **9** | **MCP-SafetyBench** | `github.com/xjzzzzzzzz/MCPSafety`; arXiv:2512.15163; ICLR 2026 | 20 attack types (server/host/user), 5 domains, dual-label execution outcomes | **High** (full execution traces) | arXiv + repo — **verify** | The forward-looking venue for an action monitor; also the source of the *p-value + effect-size reporting style* we should copy |
| **10** | **τ-bench / τ²-bench** | `github.com/sierra-research/tau-bench`; arXiv:2406.12045 / 2506.07982 | Retail + airline, user simulator, `pass^k` | **Low** (no injections) | MIT `[V]` | **Utility benchmark**: proves that recovered benign utility is not an AgentDojo artifact. `pass^k` exposes reliability collapse (`pass^8` < 25%) that `pass^1` hides |
| **11** | **R-Judge** | `aclanthology.org/2024.findings-emnlp.79`; arXiv:2401.10019 | 8k+ samples, model-graded, 10 risk categories, schema = *action + risk_category + safe/unsafe label + trajectory* | **Medium–High** (action-level with labels) | EMNLP Findings `[V]` | A ready-made **safety-judging** dataset with the exact schema our action monitor needs; useful for the judge-oracle in §7.3 |
| **12** | **Tensor Trust** | HF `qxcv/tensor-trust`; arXiv:2311.01011 | **126,808** attacks / 46,457 defenses | **None** (prompt-level) | HF terms `[V]` | Cheap scale for detector *pretraining* only; the HF viewer currently errors on a column mismatch, so pull the raw files |
| **13** | **WildGuardMix** | HF `allenai/WildGuardMix` | benign + harmful prompt mixtures | **None** | **CC-BY-NC-4.0** (NC!) `[R]` — non-commercial; **check** | Benign pool for OOD/FP measurement — but the non-commercial clause may block product use |
| **14** | **Safe-guard / deepset / Gandalf HF sets** | HF `xTRam1/safe-guard-prompt-injection` (10,296), `deepset/prompt-injections` (662), `Lakera/gandalf_ignore_instructions` (1,000) | as listed | **None** | licenses `[R]` — **verify** | Cheap benign/attack prompt-level halves for FPR sanity checks |
| **15** | **CyberSecEval 3 / 4** | `github.com/meta-llama/PurpleLlama`; arXiv:2404.13161 / 2408.01605 | v4 = **7,883** samples (multilingual PI = 1,004) | **None** | MIT `[V]` | Static multilingual PI coverage — useful only if we claim multilingual robustness |
| **16** | **PINT** | — | 4,314 inputs | — | **Test set is deliberately private** `[R]` | ⛔ **Do not attempt.** Documented so nobody spends time on it |
| **17** | **SEP / AttackGen / AgentMonitor(bench) / ToolSword(details) / FATHOM** | — | — | — | — | ⛔ **Could not verify.** Either locate a primary source or drop them from the report |

**Budget-aware order:** do **1, 2, 3, 4** first (LLMail-Inject, NotInject, TaskTracker, AgentHarm) — together they cover the human-attack recall axis, the false-positive axis, the ceiling-effect axis, and the action-label axis, at negligible cost and no gated downloads. Then **5, 6** (we already have AgentDojo). Then **7, 9** (gated / larger) if the program needs multi-benchmark and MCP coverage.

---

## 9. Citations

**Legend:** `[V]` = URL fetched and content confirmed in this session · `[R]` = recalled from training knowledge, not fetched — treat the *existence* as likely but the details as unverified.

### Verified in this session `[V]`

- AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents — https://arxiv.org/abs/2406.13352
- AgentDojo (NeurIPS 2024 D&B proceedings page) — https://proceedings.neurips.cc/paper_files/paper/2024/hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract.html
- AgentDojo paper PDF (NeurIPS D&B track) — https://papers.nips.cc/paper/2024/file/97091a5177d8dc64b1da8bf3e1f6fb54-Paper-Datasets_and_Benchmarks_Track.pdf
- AgentDojo repository (MIT) — https://github.com/ethz-spylab/agentdojo
- AgentDojo on AI Threat Atlas — https://threatatlas.ai/tool/agentdojo
- InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Using Agents — https://arxiv.org/abs/2403.02691
- InjecAgent repository — https://github.com/uiuc-kang-lab/InjecAgent
- ASB: Agent Security Bench — https://arxiv.org/abs/2410.02644
- ASB repository (README fetched: 90k cases, FNR/FPR table) — https://github.com/agiresearch/ASB
- ToolEmu: Identifying the Risks of LM Agents with an LM-Emulated Sandbox — https://arxiv.org/abs/2309.15817
- ToolEmu repository — https://github.com/ryoungj/ToolEmu
- CyberSecEval 4 landing page (Prompt Guard) — https://meta-llama.github.io/PurpleLlama/CyberSecEval/
- CyberSecEval 4 — Prompt Injection benchmark docs — https://meta-llama.github.io/PurpleLlama/CyberSecEval/docs/benchmarks/prompt_injection
- PurpleLlama / CybersecurityBenchmarks repository — https://github.com/meta-llama/PurpleLlama
- CyberSecEval 2 — https://arxiv.org/abs/2404.13161
- CyberSecEval 3 — https://arxiv.org/abs/2408.01605
- eval-cyberseceval_4 (third-party harness; v4 per-task sample counts) — https://github.com/AnyEvalOrg/eval-cyberseceval_4
- LLMail-Inject challenge dataset (HF, MIT) — https://huggingface.co/datasets/microsoft/llmail-inject-challenge
- LLMail-Inject paper (Findings of EMNLP 2026) — https://aclanthology.org/2026.findings-emnlp.514.pdf
- BIPIA: Benchmarking and Defending Against Indirect Prompt Injection Attacks on LLMs — https://arxiv.org/abs/2312.14197
- BIPIA repository — https://github.com/sunlab-uiuc/BIPIA
- StruQ: Defending Against Prompt Injection with Structured Queries (USENIX Security 2025) — https://arxiv.org/abs/2402.06363
- StruQ v2 HTML (ASR tables) — https://arxiv.org/html/2402.06363v2
- Defending Against Indirect Prompt Injection Attacks With Spotlighting (CAMLIS 2024) — https://arxiv.org/abs/2403.14720
- Spotlighting HTML (1,000-doc corpus, ASR numbers) — https://ar5iv.labs.arxiv.org/html/2403.14720
- Auditing Prompt Injection Defenses Before They Reach Users — https://arxiv.org/abs/2505.15738
- Defense Against Indirect Prompt Injection via Tool Result Parsing — https://arxiv.org/abs/2601.04795
- Indirect Prompt Injections: Are Firewalls All You Need, or Stronger Benchmarks? — https://arxiv.org/abs/2510.05244
- Progent: Programmable Privilege Control for LLM Agents — https://arxiv.org/abs/2504.11703
- Progent repository — https://github.com/sunblaze-ucb/progent
- Progent (vendor fork, `secagent`) — https://github.com/lindsey98/progent
- Defeating Prompt Injections by Design (CaMeL) — https://arxiv.org/abs/2503.18813
- Are you still on track!? Catching LLM Task Drift with Activation Deltas (TaskTracker) — https://arxiv.org/abs/2406.00799
- TaskTracker v5 HTML (>500K, >0.99 ROC AUC) — https://arxiv.org/html/2406.00799v5
- Representation Engineering: A Top-Down Approach to AI Transparency (RepE) — https://arxiv.org/abs/2502.19649
- RepE PDF (Wehner et al.) — https://janwehner.com/files/representation_engineering.pdf
- InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models — https://arxiv.org/abs/2410.22770
- InjecGuard HTML (NotInject stats, model tables) — https://arxiv.org/html/2410.22770
- NotInject dataset (HF, MIT) — https://huggingface.co/datasets/leolee99/NotInject
- InjecGuard repository — https://github.com/leolee99/InjecGuard
- InjecGuard project site — https://injecguard.github.io
- Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game — https://arxiv.org/abs/2311.01011
- Tensor Trust dataset (HF) — https://huggingface.co/datasets/qxcv/tensor-trust
- PIDS-Bench: Evaluating Prompt-Injection Detectors Under Distribution Shift — https://arxiv.org/abs/2609.15017
- Semalith v1.4 (guard-model FPR audit) — https://arxiv.org/abs/2607.22545
- HiveTraceGuard-Pro (15-model guard comparison) — https://arxiv.org/abs/2609.01046
- The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis (SoK) — https://arxiv.org/abs/2602.10453
- SoK v1 HTML (78-paper analysis, 6/41 segment-level finding) — https://arxiv.org/html/2602.10453v1
- SoK BibTeX — https://arxiv.org/bibtex/2602.10453
- Breaking the Protocol: Security Analysis of the Model Context Protocol Specification — https://arxiv.org/abs/2601.17549
- CASCADE: A Cascaded Hybrid Defense Architecture for Prompt Injection in MCP Systems — https://arxiv.org/abs/2604.17125
- MCP-SafetyBench (ICLR 2026) — https://arxiv.org/abs/2512.15163
- MCP-SafetyBench proceedings PDF — https://proceedings.iclr.cc/paper_files/paper/2026/file/d46f127a80dc58cbc0732a717285c43a-Paper-Conference.pdf
- MCP-SafetyBench repository — https://github.com/xjzzzzzzzz/MCPSafety
- APIGen-MT-5k / xLAM-2 — https://arxiv.org/abs/2504.03601
- APIGen-MT-5k dataset (HF, gated) — https://huggingface.co/datasets/Salesforce/APIGen-MT-5k
- R-Judge: A Safety Judgement Dataset for LLM Agents (Findings of EMNLP 2024) — https://arxiv.org/abs/2401.10019
- R-Judge (ACL Anthology) — https://aclanthology.org/2024.findings-emnlp.79/
- τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains — https://arxiv.org/abs/2406.12045
- τ-bench repository — https://github.com/sierra-research/tau-bench
- τ²-bench — https://arxiv.org/abs/2506.07982
- Prompt Infection: LLM-to-LLM prompt injection within multi-agent systems — https://arxiv.org/abs/2410.07283
- A Unified Framework for Confidence Calibration and Risk Control (conformal risk control protocol) — https://arxiv.org/abs/2509.01455
- Newcombe RG (1998). Interval estimation for the difference between independent proportions: comparison of eleven methods. *Statistics in Medicine* 17(8):873-890 — https://onlinelibrary.wiley.com/doi/abs/10.1002/(SICI)1097-0258(19980430)17:8%3C873::AID-SIM779%3E3.0.CO;2-I
- Newcombe RG (1998). Two-sided confidence intervals for the single proportion: comparison of seven methods. *Statistics in Medicine* 17(8):857-872 — https://onlinelibrary.wiley.com/doi/10.1002/(SICI)1097-0258(19980430)17:8%3C857::AID-SIM777%3E3.0.CO;2-E
- SAS PROC FREQ RISKDIFF (Newcombe hybrid-score CI, Farrington-Manning, TOST/equivalence) — https://support.sas.com/documentation/cdl/en/procstat/65544/HTML/default/procstat_freq_details49.htm
- NCSS: Two Proportions — Non-Inferiority, Superiority, Equivalence (TOST) — https://www.ncss.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Two_Proportions-Non-Inferiority,_Superiority,_Equivalence,_and_Two-Sided_Tests_vs_a_Margin.pdf
- SafeAgentBench — https://arxiv.org/abs/2412.13178
- SafeAgentBench project page (750 tasks / 450 hazardous) — https://safeagentbench.github.io/
- AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents (ICLR 2025) — https://proceedings.iclr.cc/paper_files/paper/2025/file/c493d23af93118975cdbc32cbe7323f5-Paper-Conference.pdf
- AgentHarm dataset (HF, UK AISI) — https://huggingface.co/datasets/ai-safety-institute/AgentHarm
- Awesome-AI-Security-Benchmarks (curated index used to enumerate §A) — https://github.com/EvanThomasLuke/Awesome-AI-Security-Benchmarks
- Agent-Safety-Paper (defense catalogue; RTBAS entry) — https://github.com/AgentSec-Guardrails/Agent-Safety-Paper
- ARGUS: Defending Against Multimodal Indirect Prompt Injection (CVPR 2026 slides) — https://cvpr.thecvf.com/media/cvpr-2026/Slides/40371.pdf
- WASP: Web Agent Security against Prompt injection — https://arxiv.org/abs/2504.18575
- WAInjectBench — https://arxiv.org/abs/2510.01354
- Open-Prompt-Injection — https://arxiv.org/abs/2402.12138
- safe-guard-prompt-injection dataset — https://arxiv.org/abs/2402.13064
- GenTel-Safe / GenTel-Bench — https://arxiv.org/abs/2404.06531
- CySecBench — https://arxiv.org/abs/2501.01335

### Recalled from training knowledge — existence likely, details unverified `[R]`

- FATH: Authentication-based Test-time Defense against Indirect Prompt Injection Attacks — https://arxiv.org/abs/2410.21492 · https://github.com/Jayfeather1024/FATH
- DataSentinel — https://arxiv.org/abs/2504.11358
- Operationalizing CaMeL — https://arxiv.org/abs/2505.22852
- RTBAS: Defending LLM Agents Against Prompt Injection and Privacy Leakage — https://arxiv.org/abs/2502.08966
- Encrypted Prompt — https://arxiv.org/abs/2503.23250
- CaMeL reference implementation — https://github.com/google-deepmind/camel-ai
- Crescendo: A Human-Agent Red Teaming Challenge — https://arxiv.org/abs/2404.01233
- ToolSword — https://arxiv.org/abs/2404.10757
- WorkArena / BrowserGym — https://arxiv.org/abs/2403.07718
- ToolACE — https://arxiv.org/abs/2409.00920
- Agent-FLAN — https://arxiv.org/abs/2310.08500
- LLMail-Inject repository (Microsoft) — https://github.com/microsoft/llmail-inject
- Lakera PINT benchmark (test set private) — https://www.lakera.ai/product-updates/lakera-pint-benchmark
- HackAPrompt / llm-injection-benchmark — https://github.com/secureworks/llm-injection-benchmark
- WildGuardMix — https://huggingface.co/datasets/allenai/WildGuardMix
- deepset/prompt-injections — https://huggingface.co/datasets/deepset/prompt-injections
- xTRam1/safe-guard-prompt-injection — https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection
- Lakera/gandalf_ignore_instructions — https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions
- nvidia/Aegis-AI-Content-Safety-Dataset-1.0 — https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0
- microsoft/Prompt-Guard-86M — https://huggingface.co/microsoft/Prompt-Guard-86M
- meta-llama/Llama-Guard-3-8B — https://huggingface.co/meta-llama/Llama-Guard-3-8B
- google/shieldgemma-2b — https://huggingface.co/google/shieldgemma-2b
- ibm-granite/granite-guardian-3.3-8b — https://huggingface.co/ibm-granite/granite-guardian-3.3-8b
- allenai/wildguard — https://huggingface.co/allenai/wildguard
- nvidia/polyguard — https://huggingface.co/nvidia/polyguard
- leolee99/PIGuard — https://huggingface.co/datasets/leolee99/PIGuard
- NeMo Guardrails — https://github.com/NVIDIA/NeMo-Guardrails
- LlamaFirewall (within PurpleLlama) — https://github.com/meta-llama/PurpleLlama/tree/main/llama-firewall
- HarmBench classifiers — https://github.com/centerforaisafety/HarmBench
- Conformal prediction kit (CRC λ̂ formula) — https://github.com/Muhtasim-Munif-Fahim/conformal-prediction-kit
- Angelopoulos, Bates, Fisch, Lei & Schuster, *Conformal Risk Control* (ICLR 2024) — https://arxiv.org/abs/2208.02814 (arXiv ID recalled; not fetched)
- Angelopoulos, Bates, Candès, Jordan & Lei, *Learn then Test* — https://arxiv.org/abs/2110.01052 (arXiv ID recalled; not fetched)
- Angelopoulos on non-monotonic CRC via algorithmic stability — https://x.com/ml_angelopoulos/status/2026339549149016141

### Could NOT verify — do not cite as fact

- **`SEP` ("Should It Be Executed or Processed")** — no such benchmark surfaced in any search. Likely a conflation with SecAlign / StruQ / SafeAgentBench. **Drop it.**
- **`AttackGen`** — no primary source located. Drop or locate.
- **`AgentMonitor`** — no canonical *benchmark* by this name found (there may be a defense by this name). Drop as a benchmark.
- **`FATHOM`** — no IPI-defense paper by this name. Almost certainly a conflation with **FATH** (§B.1). A commercial product "Fathom" (MCP server) advertises 88.7% PI accuracy — a vendor claim, not citable evidence.
- **`Mini-Dream`, `AgentVillage`, `A2A-injection`** — recalled only, no primary source fetched.
- **PINT test set** — intentionally not public; the 4,314 figure and the ~62% / ~92.5% leaderboard numbers come from third-party recall.
- **ToolEmu "68.8% of failures are valid real-world risks"** — this is the *paper's* claim; a third-party evaluation plan disputes it. Cite the paper's wording, and flag the dispute.
- **WildGuardMix / deepset / safe-guard / Gandalf / Aegis / PIGuard / PolyGuard licenses** — not verified individually; AI2 and NVIDIA datasets are commonly non-commercial.
- **TaskTracker toolkit download URL and license** — the paper and its release are verified; the exact repo/terms were not fetched.
