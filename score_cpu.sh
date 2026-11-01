#!/bin/bash
# Classifier + LLM-judge scoring for each recorded set (SecJev runs on separate scoring hosts).
source ~/secjev-env/bin/activate; cd ~/jev-stageA
AGENT=${1:-gptoss20b}; PORT=${2:-8000}; MODEL=${3:-openai/gpt-oss-20b}
order=(benign important_instructions ood_bland_note ood_quoted_reply ood_zero_width ood_llmail)
check=("=== important_instructions" "=== ood:bland_note" "=== ood:metadata_field" "=== ood:zero_width" "ALL_DONE" "LLMAIL_DONE")
logf=(record_all.log record_all.log record_all.log record_all.log record_all.log record_llmail.log)
for i in "${!order[@]}"; do
  n=${order[$i]}
  until grep -q "${check[$i]}" ${logf[$i]} 2>/dev/null; do sleep 30; done
  R=runs/$AGENT/$n.jsonl; O=scores/$AGENT
  python score.py $R --group classifiers --outdir $O > logs/cls_$n.log 2>&1 &
  python score.py $R --group judge --port $PORT --model $MODEL --outdir $O > logs/judge_$n.log 2>&1 &
  wait
  echo "done $n $(date -u)"
done
echo SCORE_DONE $(date -u)
