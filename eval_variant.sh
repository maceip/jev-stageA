#!/bin/bash
# eval_variant.sh CKPT NAME GPU [content] : rescored action points for replay + held-out classification
cd ~; export SECJEV_WORK=~/secjev-4b-work SECJEV_TF32=1 CUDA_VISIBLE_DEVICES=$3
mkdir -p evals/$2
if [ "$4" = content ]; then F="--content"; T=fc_test.jsonl; else F=""; T=ft_test.jsonl; fi
env/bin/python rescore_action.py $1 runsmini/benign.jsonl evals/$2/benign.0aof0.jsonl $F > evals/$2/log 2>&1
env/bin/python rescore_action.py $1 runsmini/ood_bland_note.jsonl evals/$2/ood_bland_note_rs.0aof0.jsonl --suites banking,slack $F >> evals/$2/log 2>&1
echo RESCORE_DONE >> evals/$2/log
[ -z "$SKIPCLS" ] && env/bin/python eval_cls.py $1 $T evals/$2/cls.json >> evals/$2/log 2>&1
echo EVAL_DONE >> evals/$2/log
