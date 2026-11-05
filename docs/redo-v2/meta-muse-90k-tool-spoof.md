# Meta Muse $90,000 prompt injection — synthetic reconstruction (2026-11-04)

## Public source

- [Securing Muse: More Than $1 Million in Bug Bounties Paid Out](https://muse.ai/blog/securing-muse)
  (David Singleton, 2026-11-02). Largest award $300,000 (pre-launch proxy-header
  auth flaw). The prompt-injection report, verbatim:

  > "Prompt injection disguised as Muse's own tools (bug bounty payment
  > $90,000). A researcher built a web page that, when a user asked Muse to
  > summarize it, imitated the status messages from Muse's browsing tool.
  > It got the agent to execute some actions which should have been caught.
  > This helped us identify some additional signals to add to our prompt
  > injection detection. We also changed how Muse handles suspicious tool
  > output. In our testing, the classifiers now flag the original attack
  > and 100 variations of it."

- Corroboration: [SecurityWeek coverage](https://www.securityweek.com/meta-shares-top-muse-bug-bounty-findings-as-payouts-reach-1-2-million/)
  (same facts, no extra payload detail).
- Architecture context: [How We Built Safety Into Muse](https://research.meta.ai/blog/security-and-safety-for-ai-agents-our-approach-with-muse)
  — browser sub-agent sees an accessibility-tree snapshot (rendered text,
  not raw DOM); independent PI classifiers inspect tool/file data entering
  context; Sentinel is sole permission authority for connector actions.

## Mechanism (reconstructed; exact payload not public)

1. Trigger is benign: user asks the agent to summarize an attacker web page.
2. Vector is untrusted tool output: rendered page text returned by the
   browsing tool (Spot 1 / tool-output-filter surface).
3. Core trick is channel spoofing, not a classic "ignore instructions"
   string: page text imitates the browsing tool's own status messages,
   so injected instructions arrive wearing trusted-tool framing.
4. Effect: agent executed actions that policy/Sentinel should have caught.
Unknowns (not in public sources): exact status-message syntax, exact
payload actions, whether text-only or image-carried. Synthetic rows
below therefore span 12 plausible status framings x 10 payload actions;
they reconstruct the mechanism class, not the literal researcher page.

## Synthetic dataset (addendum, honestly labelled synthetic)

Builder: `build_synthetic_muse_tool_spoof.py` (seed 20261104).
Outputs (also copied to jev-retrain droplet `/root/work/data/`,
SHA-256 ft `18994f85962dc948…`, canonical `83360e30641d2288…`):

- `data/external/redo_v2_out/synthetic_muse_spoof_ft.jsonl` — bench FT
  format identical to `train_spot1_t.jsonl` (state + noul `injection`
  question), ready to concatenate into the next Spot1/Joint FT build.
- `data/external/redo_v2_out/synthetic_muse_spoof_canonical.jsonl` —
  canonical redo rows, `source=synthetic-muse-tool-spoof`,
  `tier=S_synthetic`. Never report these as real-world traffic.
- 192 rows: 120 positives (12 forged-status framings x 10 payloads,
  head/mid/tail placement, 8 benign page covers) + 72 hard negatives
  (benign pages legitimately using status/tool vocabulary).

## Detection results (local scoring, 2026-11-04)

`score_synthetic_spoof.py`; full numbers in `synthetic-spoof-scores.json`.

| scorer | AUC | TPR@0.5 | FPR@0.5 |
|---|---|---|---|
| keyword baseline | 0.500 | 0.000 | 0.000 |
| SecJev-4B base | 0.9498 | 0.958 | 0.306 |
| SecJev Spot1 FT (HF published) | 0.9785 | 0.992 | 0.278 |
| SecJev Joint FT (HF published) | 0.9534 | 0.975 | 0.236 |

Findings:
- Explicit-instruction spoofs are detected by all SecJev variants
  (per-framing detection 0.9-1.0). The synthetic reconstruction does
  not evade our detectors the way the original evaded Muse's stack.
- Actionable weakness: 24-31% false positives on hard negatives —
  benign pages discussing status/tool vocabulary. The 72 synthetic
  negatives target exactly this for the next FT round.
- Implicit variants (no imperative payload; forged status declares
  consent/summary/redirect state) score lower under tool framing than
  identical attacker content as plain page text in a confounded paired
  probe (AUC 0.24-0.30, 36 rows) — channel spoofing measurably reduces
  suspicion. A payload-matched paired study is the follow-up before
  claiming an evasion rate.

## Muse-client fallback (not needed)

A faithful mechanism-level synthetic was buildable from public
sources, so no client extraction/testing was performed. Note the
public Muse client is the patched product (Meta states classifiers
now flag the original + 100 variations); any live-client test must
stay inside the Meta bug bounty scope/rules.
