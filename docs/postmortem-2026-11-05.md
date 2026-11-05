# postmortem: why jev training kept restarting (2026-11-05)

plain summary: the trained weights were rarely the problem. the data, the evaluation harness, the infrastructure, and the process around the training kept changing. each change made earlier results contestable. each contestable result triggered a redo. about 22 fine-tune stages ran, and at the end the repeatable, publishable evidence is thinner than the training volume suggests.

## timeline of redo triggers

1. run 1 (open-weight agents): thresholds tuned on the same normal tasks used for reporting. tuned numbers are optimistic by construction. result: rerun with split tuning/reporting suites.
2. run 2 (qwen3.6): all lambda nodes terminated mid-run. partial data never analysed. result: full restart as run 3.
3. run 3 static attacks: frontier agent hijacked 0 times in about 2,800 attacked runs. static sweeps cannot separate defenses. measurable signal moved to adaptive attacks and false-alarm cost after the static budget was spent.
4. zero-shot action monitor: about 17 points of normal-task loss. verdict: train the monitor. result: metadata fine-tunes (variants a–h).
5. metadata fine-tunes: catches were mostly early false alarms on reads, no transfer to unseen tool sets. verdict: the missing signal is tool-output content. result: content fine-tunes (variants c1–c6).
6. harness defects found by review mid-flight: tool_result.error bypass, replay executing never-run trailing calls, gpu lock omissions, causal_conv1d abi mismatch, yaml injection corruption, secjev_work import-time crash. the frozen analysis plan says any code change reruns affected evaluations. result: rescoring and replay sweeps after each fix.
7. c1 content monitor: live numbers good on one-try attacks, adaptive hijack 53–58%. cascade wiring followed. combined checks later measured no gain and large utility loss (72/249 vs 194/249). adaptive files for combined checks sit empty.
8. parallel agent sweep: seven raffle worktrees reimplemented scoring, calibration, and datasets in parallel on a 29-commit-stale base. result: rebase conflict debt and six fake-based test suites to repair (repaired: all suites green, fake-only fixes).
9. data redo (v2, then v0.3 frozen): label audit and split rebuilds. dev and test sets 92% positive with about 1,700 negatives in 22k rows, so fpr estimates stay unstable at exactly the thresholds that matter.
10. clef direction change: secjev kev checkpoints replaced mid-program by a clef-flash 9b joint head on hidden states (litert metal, then rocm droplet). new data builders, new scorer, new publish path. every prior secjev number became background instead of the product.
11. metric defects: ft_action.evaluate assumed authorization question keys, so training-log dev metrics for the tool-output redo were invalid and needed corrected re-evaluation. clef seed2 dev eval reported action guard n=0 while training continued and saved a "best" checkpoint on a combined auc of 0.4944.
12. infrastructure losses: first training droplet destroyed with raw bench outputs lost (headline numbers survive only in session notes). a full-size queue died at step 10 of 402 when the launching ssh session scope deactivated. clef seed1 first attempt oom-killed at 256 gb ram from materializing the train set. local disk reached 99–100% twice.
13. publication lag: weights sat staged and unuploaded, results sat in run logs, readme pointed at a hugging face organization that does not exist. from outside, nothing looked shipped, which created pressure for one more redo instead of one publication pass.

## root causes

- no enforced freeze. data, code, thresholds, model family, and terminology all changed inside single evaluation generations. a freeze existed on paper (docs/analysis-plan.md) and was overridden repeatedly by new findings and new directions.
- decision rules without stop discipline. the frozen gate said stop on rule 2 or rule 3 failure. the program continued through at least three further generations after the gate failed, without a new written gate.
- evaluation sample sizes too small to settle questions. 37 normal tasks on reporting suites give ±15-point intervals. every close call stayed arguable, and arguable results invite reruns.
- correctness audits ran after training, not before. label audits, metric-key bugs, and split leakage checks belong in front of gpu spend. in jev-stagea the audits repeatedly landed after full sweeps.
- too many parallel implementations of the same layer. seven raffle branches plus a shared dirty main worktree produced overlapping scorers, calibrators, and dataset builders. integration cost exceeded the value of parallel exploration.
- artifacts lived on ephemeral machines. lambda nodes and two droplets held canonical outputs. one droplet destruction already caused permanent data loss. harvest was manual and late.
- offline auc substituted for the product metric. most stages optimized dev auc on small, skewed sets. the product questions (adaptive hijack rate, work completed under attack, normal-task cost at a frozen threshold) stayed partially unmeasured: adaptive files for combined checks are still empty.

## what is actually done and kept (2026-11-05)

- public weights on hugging face under macmacmacmac: SecJev-4B-Spot1-Redo, SecJev-4B-Spot2-Redo, SecJev-4B-Joint-Redo, CLEF-Screen. dev-only metrics, defects disclosed on cards.
- droplet queue finished 2026-11-04 21:23 utc: secjev seed2 done (dev auc 0.9967, 7,192 frozen test scores, 0 failures), threshold variants done, ultraspeed zero-shot scores done, clef seed2 done with the n=0 eval defect noted in item 11.
- frozen-test replication exists for one secjev training: trained test auc 0.8802 vs base 0.8146 at a dev-calibrated threshold (replication/recomputed.json, harvested).
- all droplet outputs harvested 2026-11-05 to ft/droplet_harvest_20261105 (2.9 gb; caches excluded, backbones excluded, weights and scores included) and committed with the postmortem. the jev-retrain droplet (168.144.60.186) sits idle at load 0.00. idle gpu droplets cost money; harvest first, then destroy on owner instruction only.
- code state: main at 3fa5e06 plus harvest commit; six raffle branches rebased, test-fixed, verified green, unpushed.

## rules that stop the next redo cycle

1. freeze data, harness, thresholds, and reporting metric in one manifest with hashes before any gpu run. changes after freeze start a new named generation, never a silent patch.
2. audit labels and metric code on a 200-row dry run before full training. a stage that cannot print sane per-kind counts on dev does not train.
3. one product metric set per generation: adaptive hijack rate, utility under attack, normal-task cost, all at the frozen threshold, all with intervals sized by a power calculation done before recording.
4. harvest to local disk and commit in the same hour a run finishes. no canonical output lives only on a droplet.
5. one implementation per layer on main. parallel branches get a merge-or-drop decision inside one week.
6. publish weights and a results table at the end of each generation, including failed gates. unpublished results do not count as progress.
7. commit completed work immediately (project rule, .agents/memory/commit-policy.md). uncommitted work in shared trees caused at least two of the incidents in the timeline.
