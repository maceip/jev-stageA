#!/bin/bash
# A100: SecJev (2 shards) + classifiers for one run file. Usage: score_local3.sh <set>
n=$1; cd ~/jev-stageA
export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attempt in 1 2 3; do
  ~/env/bin/python score.py runs/mini/$n.jsonl --group secjev --secjev ~/SecJev/SecJev-4B --outdir scores/mini > sj0_$n.log 2>&1 &
  ~/env/bin/python score.py runs/mini/$n.jsonl --group classifiers --outdir scores/mini > cls_$n.log 2>&1 &
  wait
  grep -q Traceback sj0_$n.log cls_$n.log || { echo "SCORED $n $(date -u)"; exit 0; }
done
echo "FAILED $n"; exit 1
