# Prompt-injection technique catalog — pick-up record (2026-11-04)

## Resume checkpoint (WorkBuddy AI handoff)

- Source artifact: user-pasted WorkBuddy transcript (no local session
  file; the 4 PDFs exist only in the WorkBuddy cloud session as
  paper1_emnlp.pdf / paper2_cmc.pdf / paper3_arxiv1.pdf /
  paper4_arxiv2.pdf per ~/.workbuddy-ai/audit-log). No third-party
  artifact was modified.
- Identified papers, re-fetched publicly: Paper 1 = LLMail-Inject
  (arXiv:2506.09956, EMNLP Findings), Paper 2 = CMC survey
  (DOI 10.32604/cmc.2025.074081, 128-study taxonomy). Papers 3/4
  (arXiv1/arXiv2) not identifiable from local evidence — unknown.
- Latest user instruction in the transcript: find >=100 unique real
  prompt-injection techniques/attempts from the papers, their
  bibliographies, Google Scholar, and GitHub; DO NOT train until done.
- Safari bridge coordinator was down (127.0.0.1:47635 refused);
  Scholar/GitHub discovery ran through web search as fallback.

## Catalog (this build)

Builder `build_technique_catalog.py` (seed 20261104), verbatim
extraction only, dedup by normalized text:
128 unique attacks, 21 technique families, 7 sources
(llmail 46, bipia 16, gandalf 14, deepset 14, satml24 14,
injecagent 12, tensor-trust 12). License tiers: 88 trainable
(MIT/CC-BY), 28 eval-only (prior protocol), 12 excluded
(TensorTrust has no license file). Families follow CMC axes refined
by LLMail-observed patterns: cot_hijack (8), encoding_split_
reassembly (8), tool_call_forgery (7), unicode_obfuscation (3),
fake_system_admin (2), fake_user_turn (8), fake_assistant_prefill
(1), delimiter_escape (4), direct_override (14), roleplay_persona
(13), secret_extraction (23), authority_social_engineering (9),
retrieval_relevance_stuffing (2), indirect_email_injection (8),
task_redirect_qa (8), code_injection_payload (8), indirect_tool_harm
x5 subtypes (12). Files: technique_catalog_100.jsonl (SHA-256
4afff0a2095cf961…), technique_catalog_trainable.jsonl,
technique_catalog_canonical.jsonl, technique_catalog_summary.json
in data/external/redo_v2_out/, copied to jev-retrain:/root/work/data/.
Blocked: hackaprompt/hackaprompt-dataset is HF-gated (403).

## What is actually wrong with the training data (diagnosis)

1. Two different data regimes got conflated. The "569 rows" complaint
   is the original ft_clef proper protocol (train 569/dev 149/test
   220). The frozen redo sets are large (Spot-1 v0.2: 195,313;
   frozen-v0.3 CLEF: 146,984). Volume is not the binding constraint;
   composition is.
2. Class skew: frozen CLEF dev/test are 92% positive (1,695
   negatives of 21,977). Dev AUC saturates while operating FPR stays
   0.22 (CLEF seed1) — negatives are the scarce resource.
3. Domain/threshold non-transfer is measured, not hypothetical:
   SecJev Spot2-V3 dev AUC 0.9971 vs full-test AUC 0.8661, and the
   dev-calibrated threshold gives test FPR 0.221. Dev shares suites
   with train; test does not.
4. Technique thinness inside volume: LLMail dominates (~82% of the
   peer canon pre-cap per REDO-REPORT), paired attacker/benign calls
   come from small template sets (call-identity shortcut risk), and
   several catalogued families (cot_hijack, code_injection_payload,
   encoding_split_reassembly) were near-absent before this catalog.
Next step (not started): fold technique_catalog_canonical.jsonl as
an addendum into the next Spot1/Joint/CLEF data build, rebalance
negatives, and keep every frozen test untouched by catalog rows.
