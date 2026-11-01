#!/bin/bash
# Score each recorded attack set once its recording finishes; three detector groups in parallel.
source ~/secjev-env/bin/activate; export SECJEV_WORK=$HOME/secjev-4b-work; cd ~/jev-stageA
AGENT=${1:-gptoss20b}; PORT=${2:-8000}; MODEL=${3:-openai/gpt-oss-20b}
order=(benign important_instructions ood_bland_note ood_metadata_field ood_quoted_reply ood_zero_width ood_llmail)
next=(important_instructions ood:bland_note ood:metadata_field ood:quoted_reply ood:zero_width ALL_DONE LLMAIL_DONE)
for i in "${!order[@]}"; do
  n=${order[$i]}; m=${next[$i]}
  if [ "$m" = "LLMAIL_DONE" ]; then until grep -q LLMAIL_DONE record_llmail.log 2>/dev/null; do sleep 30; done
  elif [ "$m" = "ALL_DONE" ]; then until grep -q ALL_DONE record_all.log; do sleep 30; done
  else until grep -q "=== $m " record_all.log; do sleep 30; done; fi
  echo "scoring $n $(date -u)"
  R=runs/$AGENT/$n.jsonl; O=scores/$AGENT
  python score.py $R --group secjev --secjev ~/SecJev/SecJev-4B --outdir $O > logs/secjev_$n.log 2>&1 &
  python score.py $R --group classifiers --outdir $O > logs/cls_$n.log 2>&1 &
  python score.py $R --group judge --port $PORT --model $MODEL --outdir $O > logs/judge_$n.log 2>&1 &
  wait
  grep -h -E "scored|Error|Traceback" logs/*_$n.log
  echo "done $n $(date -u)"
done
echo SCORE_DONE $(date -u)
