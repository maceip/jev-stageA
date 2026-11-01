#!/bin/bash
# Stage A rerun, one agent on one box. Usage: stage2_box.sh <agent_tag> <served_model> <judge_url> <judge_model>
# 1) record undefended sets  2) score every set (SecJev x2 shards, classifiers, fixed LLM judge)  3) adaptive, undefended arm
AGENT=$1; MODEL=$2; JURL=$3; JMODEL=$4
cd ~/jev-stageA; mkdir -p runs/$AGENT scores/$AGENT logs adaptive
export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/env/bin/python
SETS="benign important_instructions tool_knowledge ood:bland_note ood:metadata_field"
for A in $SETS; do
  N=$(echo $A | tr ":" "_")
  extra=""; [ "$A" = benign ] && extra="--paraphrases paraphrases.json"
  echo "=== record $A $(date -u)"
  $PY run_agent.py --attack $A --suites workspace,travel,banking,slack --model $MODEL --workers 64 $extra --out runs/$AGENT/$N.jsonl
done
echo "RECORD_DONE $(date -u)"
for A in $SETS; do
  N=$(echo $A | tr ":" "_"); R=runs/$AGENT/$N.jsonl; O=scores/$AGENT
  echo "=== score $N $(date -u)"
  $PY score.py $R --group secjev --secjev ~/SecJev/SecJev-4B --shard 0/2 --outdir $O > logs/sj0_$N.log 2>&1 &
  $PY score.py $R --group secjev --secjev ~/SecJev/SecJev-4B --shard 1/2 --outdir $O > logs/sj1_$N.log 2>&1 &
  $PY score.py $R --group classifiers --outdir $O > logs/cls_$N.log 2>&1 &
  $PY score.py $R --group judge --port $JURL --model $JMODEL --outdir $O > logs/judge_$N.log 2>&1 &
  wait
  for g in sj0 sj1 cls judge; do grep -q Traceback logs/${g}_$N.log && { echo "FAILED scoring $g $N"; exit 1; }; done
done
echo "SCORE_DONE $(date -u)"
echo "=== adaptive none $(date -u)"
$PY adaptive.py --pairs adaptive_pairs200.json --defense none --model $MODEL --agent-url http://localhost:8000/v1 \
   --attacker-url $JURL --attacker-model $JMODEL --rounds 10 --workers 32 --repeat 0 --out adaptive/${AGENT}_none_r0.jsonl > logs/adapt_none_r0.log 2>&1
echo "ADAPT_NONE_DONE $(date -u)"
