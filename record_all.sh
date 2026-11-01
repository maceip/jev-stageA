#!/bin/bash
# Record all attack sets for gpt-6.1-astra (only model used; no Qwen or gpt-oss-20b). Usage: record_all.sh [agent_tag] [model_name]
AGENT=${1:-astra}; MODEL=${2:-gpt-6.1-astra}
source ~/harness-env/bin/activate; cd ~/jev-stageA
for A in benign important_instructions ood:bland_note ood:metadata_field ood:quoted_reply ood:zero_width; do
  N=$(echo $A | tr ":" "_")
  echo "=== $A $(date -u)"
  python run_agent.py --attack $A --model $MODEL --workers 64 --out runs/$AGENT/$N.jsonl
done
echo "ALL_DONE $(date -u)"
echo "=== llmail $(date -u)" > record_llmail.log
python run_agent.py --attack ood:llmail --suites workspace --model $MODEL --workers 64 --out runs/$AGENT/ood_llmail.jsonl >> record_llmail.log 2>&1
echo "LLMAIL_DONE $(date -u)" >> record_llmail.log
