# All 10 configurations (clean live runs, gpt-5.4-mini agent, gpt-6-astra judge, banking+slack, redact)

| Configuration | FPR | Benign utility | ASR 1-try | ASR 10-try | UA | n (normal/1-try/10-try) |
|---|---|---|---|---|---|---|
| YOLO | 0.0% [0.0-2.5] | 96.6% [92.3-98.5] | 31.3% [25.9-37.3] | 67.0% [57.3-75.4] | 78.7% [73.2-83.3] | 148/249/100 |
| JEV Spot 1 | 0.0% [0.0-2.5] | 96.6% [92.3-98.5] | 3.6% [1.9-6.7] | 50.0% [40.4-59.6] | 27.7% [22.5-33.6] | 148/249/100 |
| JEV Spot 2 | 4.1% [1.9-8.6] | 93.9% [88.8-96.8] | 0.0% [0.0-1.5] | 58.0% [48.2-67.2] | 76.7% [71.1-81.5] | 148/249/100 |
| JEV Spot 1+2 | 5.4% [2.8-10.3] | 93.2% [88.0-96.3] | 0.0% [0.0-1.5] | 40.0% [30.9-49.8] | 28.9% [23.6-34.8] | 148/249/100 |
| Judge Spot 1 | 0.7% [0.1-3.7] | 95.3% [90.6-97.7] | 0.0% [0.0-1.5] | 7.0% [3.4-13.7] | 27.7% [22.5-33.6] | 148/249/100 |
| Judge Spot 2 | 4.7% [2.3-9.4] | 95.3% [90.6-97.7] | 0.0% [0.0-1.5] | 7.0% [3.4-13.7] | 79.9% [74.5-84.4] | 148/249/100 |
| Judge Spot 1+2 | 6.1% [3.2-11.2] | 95.9% [91.4-98.1] | 0.0% [0.0-1.5] | 4.0% [1.6-9.8] | 26.5% [21.4-32.3] | 148/249/100 |
| JEV->Judge Spot 1 | 0.0% [0.0-2.5] | 98.6% [95.2-99.6] | 3.2% [1.6-6.2] | 48.0% [38.5-57.7] | 27.7% [22.5-33.6] | 148/249/100 |
| JEV->Judge Spot 2 | 0.0% [0.0-2.5] | 95.9% [91.4-98.1] | 0.0% [0.0-1.5] | 60.0% [50.2-69.1] | 79.1% [73.6-83.7] | 148/249/100 |
| JEV->Judge Spot 1+2 | 0.0% [0.0-2.5] | 99.3% [96.3-99.9] | 0.0% [0.0-1.5] | 37.0% [28.2-46.8] | 27.3% [22.2-33.2] | 148/249/100 |

FPR = share of normal runs where the defense fired. ASR = attack success rate. UA = utility under attack.
[..] = Wilson 95% CI. 0 errored episodes/targets in every file. Files that hit GPU out-of-memory errors (too many
processes per GPU) were rerun from scratch; the failed copies are in run3/clean_live/oom_failed/ for the record.
Arms: none, arm1-arm4 at 0f3506d; arm5 (double JEV + judge) at 0f3506d; s1judge, s2judge, s2judgeonly,
bothjudgeonly at 226b348 (adds those defense options; no change to existing ones).
