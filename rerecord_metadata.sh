#!/bin/bash
# Re-record ood:metadata_field (first pass failed to load: unescaped quotes in YAML); classifier + judge scoring.
AGENT=${1:-astra}; MODEL=${2:-gpt-6.1-astra}; PORT=${3:-https://api.openai.com/v1}
cd ~/jev-stageA
until grep -q LLMAIL_DONE record_llmail.log 2>/dev/null; do sleep 30; done
if [ -f runs/$AGENT/ood_metadata_field.jsonl ] && grep -q ParserError runs/$AGENT/ood_metadata_field.jsonl; then
  mv runs/$AGENT/ood_metadata_field.jsonl runs/$AGENT/ood_metadata_field.yaml_error_pass1.jsonl
  rm -f scores/$AGENT/*/ood_metadata_field.*
fi
echo "=== metadata rerun $(date -u)"
~/harness-env/bin/python run_agent.py --attack ood:metadata_field --model $MODEL --workers 64 --out runs/$AGENT/ood_metadata_field.jsonl
echo "METADATA_RECORDED $(date -u)"
source ~/secjev-env/bin/activate
R=runs/$AGENT/ood_metadata_field.jsonl; O=scores/$AGENT
python score.py $R --group classifiers --outdir $O > logs/cls_meta2.log 2>&1 &
python score.py $R --group judge --port $PORT --model $MODEL --outdir $O > logs/judge_meta2.log 2>&1 &
wait
echo "METADATA_DONE $(date -u)"
