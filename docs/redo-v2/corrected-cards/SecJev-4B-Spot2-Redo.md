---

## Correction notice (2026-11-04)

These weights are a **1-epoch research artifact** trained on non-frozen peer bench sets.
Training-time dev metrics for injection rows were computed with an incompatible score
statistic and are **withdrawn**. No frozen-test, adaptive, or replication result supports
these weights. They must not be used as evidence of a comparison against hosted
JEV/Ultrafast models. A corrected multi-epoch frozen-split run supersedes this artifact;
see the project repository. The backbone was not fine-tuned: only the LoRA adapter and
the decision head were trained.
