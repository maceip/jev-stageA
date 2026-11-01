# Action-authorization dataset

Inputs: `/Users/mac/.raffle/worktrees/agent-786e4f-r3/jev-stageA/data/gptoss20b_stageA.tgz, /Users/mac/.raffle/worktrees/agent-786e4f-r3/jev-stageA/data/qwen30b_stageA.tgz`  
Output: `/Users/mac/.raffle/worktrees/agent-786e4f-r3/jev-stageA/authz_out`  
Folds: 2 (suite -> fold: banking=1, slack=1, travel=0, workspace=0)

## Counts

| split | examples | authorized | unauthorized | unknown |
|---|---|---|---|---|
| train | 27923 | 27231 | 692 | 0 |
| test | 7758 | 6517 | 1241 | 0 |
| unknown (held out) | 5818 | 0 | 0 | 5818 |

## Per suite

| suite | fold | all | authorized | unauthorized | unknown | tools |
|---|---|---|---|---|---|---|
| banking | 1 | 4907 | 2958 | 668 | 1281 | 11 |
| slack | 1 | 6553 | 3559 | 573 | 2421 | 13 |
| travel | 0 | 8250 | 6677 | 200 | 1373 | 29 |
| workspace | 0 | 21789 | 20554 | 492 | 743 | 27 |

## Per fold

| fold | suites | labels |
|---|---|---|
| 0 | travel, workspace | authorized=27231, unauthorized=692, unknown=2116 |
| 1 | banking, slack | authorized=6517, unauthorized=1241, unknown=3702 |

## Label rules

| rule | all | train | test |
|---|---|---|---|
| ambiguous_attack_success | 5818 | 0 | 0 |
| cred_access | 45 | 0 | 45 |
| egress_sink | 225 | 84 | 141 |
| foreign_tool | 1 | 0 | 1 |
| injection_target_arg | 1662 | 608 | 1054 |
| ordinary_call_attack_resisted | 33044 | 26787 | 6257 |
| ordinary_call_no_injection | 704 | 444 | 260 |

## Cross-suite tool overlap (Jaccard)

| suite | banking | slack | travel | workspace |
|---|---|---|---|---|
| banking | 1.0 | 0.0 | 0.0 | 0.0 |
| slack | 0.0 | 1.0 | 0.0 | 0.0 |
| travel | 0.0 | 0.0 | 1.0 | 0.098039 |
| workspace | 0.0 | 0.0 | 0.098039 | 1.0 |

Maximum pairwise overlap: **0.098039** (travel vs workspace).

| suite A | suite B | shared tools | jaccard |
|---|---|---|---|
| banking | slack | 0 | 0.0 |
| banking | travel | 0 | 0.0 |
| banking | workspace | 0 | 0.0 |
| slack | travel | 0 | 0.0 |
| slack | workspace | 0 | 0.0 |
| travel | workspace | 5 | 0.098039 |

Train/test overlap: train = travel, workspace; test = banking, slack; maximum cross-split Jaccard = **0.0**; no shared tools
