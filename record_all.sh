#!/bin/bash
# Record all attack sets for one agent model. Usage: record_all.sh <agent_tag> <served_model_name>
AGENT=${1:-gptoss20b}; MODEL=${2:-openai/gpt-oss-20b}
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
