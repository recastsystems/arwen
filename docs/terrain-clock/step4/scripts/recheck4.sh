#!/bin/bash
source /work/tclock/s4/env-s4.sh; cd $W4
echo "card $GPU_MUTEX_CARDS $(nvidia-smi --query-gpu=name --format=csv,noheader -i $GPU_MUTEX_CARDS)" > $W4/results/recheck4-card.txt
$PY $W4/recheck4.py $W4/recheck4-cells.json $W4/results/recheck4.json > $W4/logs/recheck4.log 2>&1
