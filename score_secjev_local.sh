#!/bin/bash
# SecJev scoring on the recording host itself, for sets whose recording has finished locally.
AGENT=$1; N=${2:-1}; shift 2; SETS="$@"
source ~/secjev-env/bin/activate; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export SECJEV_WORK=$HOME/secjev-4b-work SECJEV_TF32=1; cd ~/jev-stageA
marker() { case $1 in
  ood_quoted_reply) echo "record_all.log|=== ood:zero_width";;
  ood_zero_width) echo "record_all.log|ALL_DONE";;
  ood_llmail) echo "record_llmail.log|LLMAIL_DONE";; esac; }
for n in $SETS; do
  IFS="|" read -r lf pat <<< "$(marker $n)"
  until grep -q "$pat" $lf 2>/dev/null; do sleep 30; done
  for i in $(seq 0 $((N-1))); do
    python score.py runs/$AGENT/$n.jsonl --group secjev --secjev ~/SecJev/SecJev-4B --shard $i/$N --outdir scores/$AGENT > logs/sj_${n}_$i.log 2>&1 &
  done
  wait; echo "scored $n $(date -u)"
done
echo LOCAL_SECJEV_DONE $(date -u)
