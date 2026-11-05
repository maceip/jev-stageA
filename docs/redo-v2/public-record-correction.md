# Public-record correction — Tuesday v2 (PREPARED, NOT YET APPLIED)

**Status:** prepared locally. No Hugging Face repository has been modified in this session.
The Tuesday assessment listed publication posture (visible correction vs withdrawal) as a
decision belonging to the project owner. This document implements the conservative option —
a **visible correction** — and should be applied only after that decision is confirmed
and after gates G5–G8 pass.

## Why a correction is required

Three public repositories were published on 2026-11-03 from single-epoch fine-tunes:

- `macmacmacmac/SecJev-4B-Spot2-Redo`
- `macmacmacmac/SecJev-4B-Spot1-Redo`
- `macmacmacmac/SecJev-4B-Joint-Redo`

Their model cards state dev-only metrics from a corrected per-question re-evaluation, but:

1. Training ran **1 epoch** on `data/bench/*` **peer bench sets**, not the frozen
   `redo-v2-canonical-spot2-v0.2` splits.
2. Training-time dev metrics for spot1/joint were computed with `ft_action.evaluate`,
   which assumes authorization question keys; injection rows use `noul` `true`/`false`
   keys, so those metrics were **invalid**. (`ft_action.py` is now fixed with
   `verdict_score`.)
3. **No frozen-test, adaptive, or replication result exists** for these weights.
4. The six-arm benchmark compared these 1-epoch weights against the production hosted
   `ultraspeed_hosted` model — a structurally unfair comparison.

## Required errata text (identical on all three cards)

> **Correction notice (2026-11-04).** These weights are a 1-epoch research artifact trained
> on non-frozen peer bench sets. Training-time dev metrics for injection rows were computed
> with an incompatible score statistic and are withdrawn. No frozen-test, adaptive, or
> replication result supports these weights. They must not be used as evidence of a
> comparison against hosted JEV/Ultrafast models. A corrected, multi-epoch, frozen-split
> training run is in progress; see the project repository for the replacement artifacts.
> The earlier claim of a fine-tuned backbone applies to the LoRA adapter and decision head
> only.

## Superseded claims (retire, do not quietly edit)

| Claim | Status |
|---|---|
| "Redo FT checkpoint" implies a full fine-tune | Replace with "1-epoch adapter/head research artifact" |
| dev AUC 0.9911 / 0.9327 / 0.9898 as headline | Withdrawn as a training-time metric; replaced only after frozen dev re-evaluation |
| Any inference about beating hosted Ultraspeed | Unsupported; the comparison arm was unfair |
| `jev-train-1` H100 provenance | Accurate for those weights; that droplet no longer exists |

## Application procedure (after approval)

`tools/correct_public_record.py` writes the correction notice into each model card and
uploads it with the HF token from `tuesday_gates.canonical_secret("HF_TOKEN")`. It is
**fail-closed**: it refuses to run unless `TUESDAY_PUBLICATION_APPROVED=1` is set and the
frozen chain manifest verifies. Run it only after G7 replication.
