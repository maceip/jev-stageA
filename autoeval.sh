#!/bin/bash
# poll for new epoch checkpoints of content runs and evaluate each once (content mode), on the given GPU
G=$1; shift
while true; do
  for r in "$@"; do for e in 1 2; do d=runs/$r/epoch$e; n=${r%%_*}_e$e
    if [ -d $d ] && [ ! -d evals/$n ]; then SKIPCLS=1 ./eval_variant.sh $d $n $G content; fi
  done; done; sleep 60
done
