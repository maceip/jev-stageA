#!/bin/bash
# On a helper GPU: score finished sets from the recording host (SecJev 2 shards + classifiers), push scores back.
SRC=$1; AGENT=$2
cd ~/jev-stageA; mkdir -p runs/$AGENT scores/$AGENT logs
export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/env/bin/python
order=(benign important_instructions tool_knowledge ood_bland_note ood_metadata_field)
marks=("=== record important_instructions" "=== record tool_knowledge" "=== record ood:bland_note" "=== record ood:metadata_field" "RECORD_DONE")
for i in "${!order[@]}"; do
  n=${order[$i]}
  until ssh $SRC "grep -q '${marks[$i]}' ~/jev-stageA/stage2.log"; do sleep 30; done
  scp -q $SRC:~/jev-stageA/runs/$AGENT/$n.jsonl runs/$AGENT/$n.jsonl
  for attempt in 1 2 3; do
    $PY score.py runs/$AGENT/$n.jsonl --group secjev --secjev ~/SecJev/SecJev-4B --shard 0/2 --outdir scores/$AGENT > logs/sj0_$n.log 2>&1 &
    $PY score.py runs/$AGENT/$n.jsonl --group secjev --secjev ~/SecJev/SecJev-4B --shard 1/2 --outdir scores/$AGENT > logs/sj1_$n.log 2>&1 &
    $PY score.py runs/$AGENT/$n.jsonl --group classifiers --outdir scores/$AGENT > logs/cls_$n.log 2>&1 &
    wait
    grep -q Traceback logs/sj0_$n.log logs/sj1_$n.log logs/cls_$n.log || break
    echo "retry $n (attempt $attempt)"
  done
  grep -q Traceback logs/sj0_$n.log logs/sj1_$n.log logs/cls_$n.log && { echo "FAILED $n"; exit 1; }
  ssh $SRC "mkdir -p ~/jev-stageA/scores/$AGENT/secjev ~/jev-stageA/scores/$AGENT/classifiers"
  scp -q scores/$AGENT/secjev/$n.*.jsonl $SRC:~/jev-stageA/scores/$AGENT/secjev/
  scp -q scores/$AGENT/classifiers/$n.*.jsonl $SRC:~/jev-stageA/scores/$AGENT/classifiers/
  echo "pushed $n $(date -u)"
done
echo PULL_SCORE_DONE $(date -u)
