#!/bin/bash
# On a scoring host: pull each finished run file from the recording host, score SecJev in N shards, push back.
SRC=$1; AGENT=$2; N=${3:-4}; shift 3; SETS="$@"
source ~/secjev-env/bin/activate; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export SECJEV_WORK=$HOME/secjev-4b-work; cd ~/jev-stageA
mkdir -p runs/$AGENT scores/$AGENT/secjev logs
done_marker() {  # set -> "logfile|pattern" signalling its recording finished on SRC
  case $1 in
    important_instructions) echo "record_all.log|=== ood:bland_note";;
    ood_bland_note) echo "record_all.log|=== ood:metadata_field";;
    ood_metadata_field) echo "$METALOG|$METAPAT";;
    ood_quoted_reply) echo "record_all.log|=== ood:zero_width";;
    ood_zero_width) echo "record_all.log|ALL_DONE";;
    ood_llmail) echo "record_llmail.log|LLMAIL_DONE";;
  esac
}
for n in $SETS; do
  IFS="|" read -r lf pat <<< "$(done_marker $n)"
  until ssh $SRC "grep -q '$pat' ~/jev-stageA/$lf" 2>/dev/null; do sleep 30; done
  scp -q $SRC:~/jev-stageA/runs/$AGENT/$n.jsonl runs/$AGENT/$n.jsonl
  for i in $(seq 0 $((N-1))); do
    python score.py runs/$AGENT/$n.jsonl --group secjev --secjev ~/SecJev/SecJev-4B --shard $i/$N --outdir scores/$AGENT > logs/sj_${n}_$i.log 2>&1 &
  done
  wait
  for attempt in 1 2 3; do   # a shard that crashed (e.g. GPU OOM on a long output) is rerun alone; it resumes
    failed=$(for i in $(seq 0 $((N-1))); do grep -q Traceback logs/sj_${n}_$i.log && echo $i; done)
    [ -z "$failed" ] && break
    for i in $failed; do
      echo "rerun shard $i of $n (attempt $attempt)"
      python score.py runs/$AGENT/$n.jsonl --group secjev --secjev ~/SecJev/SecJev-4B --shard $i/$N --outdir scores/$AGENT > logs/sj_${n}_$i.log 2>&1
    done
  done
  if [ -n "$failed" ] && grep -q Traceback $(for i in $failed; do echo logs/sj_${n}_$i.log; done); then echo "FAILED $n"; exit 1; fi
  grep -h -E "Traceback|Error" logs/sj_${n}_*.log | head -3
  ssh $SRC "mkdir -p ~/jev-stageA/scores/$AGENT/secjev"
  scp -q scores/$AGENT/secjev/$n.*of$N.jsonl $SRC:~/jev-stageA/scores/$AGENT/secjev/
  echo "pushed $n $(date -u)"
done
echo REMOTE_SECJEV_DONE $(date -u)
