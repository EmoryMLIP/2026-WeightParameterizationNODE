#!/bin/bash
# Joint selection of (readout warm start, Adam learning rate) on validation
# error at the 6000-iteration budget of the final sweep.
#
# The two interact. Warm-starting the readout with the regularized least-squares
# solution (as runExperiment.m does) helps on DCR, whose readout holds most of
# the parameters, and hurts on ELM and CDR, whose readouts are small. Adam's step
# is absolute, so MATLAB's lr=0.001 does not carry over: the range is centred on
# this pipeline's default of 1e-2.
#
# 3 datasets x warm start {0,1} x lr {1e-2,1e-3,1e-4} = 18 runs, seed 0,
# Legendre d=3. Usage: ./run_config_selection.sh <sweepId> [nWorkers]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="${1:?usage: run_config_selection.sh <sweepId> [nWorkers]}"; NPROC="${2:-6}"
ROOT="$(cd .. && pwd)"; RAW="$ROOT/results/raw/$SWEEP/python"; LOGS="$ROOT/results/raw/$SWEEP/logs"
mkdir -p "$RAW" "$LOGS"
"$(pwd)/.venv/bin/python" sweep_manifest.py start "$SWEEP" 18 "Readout warm start and ADAM learning rate selected on validation error: 3 datasets x warm start {0,1} x lr {1e-2,1e-3,1e-4}, seed 0, Legendre d=3."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$(pwd)"
for DATASET in ELM CDR DCR; do for WS in 0 1; do for LR in 1e-2 1e-3 1e-4; do
  echo "$DATASET $WS $LR"; done; done; done \
| xargs -P "$NPROC" -L 1 bash -c '
  set -euo pipefail
  DATASET=$0 WS=$1 LR=$2
  TAG="${DATASET}_NODE_legendre_d3_T1_adam_ws${WS}_lr${LR}_s0"
  OUT="'"$RAW"'/${TAG}.json"
  [ -f "$OUT" ] && { echo "skip  $TAG"; exit 0; }
  echo "start $TAG"
  '"$(pwd)"'/.venv/bin/python surrogate/train_surrogate_node.py \
      --dataset "$DATASET" --data_dir ../data --basis legendre --degree 3 --seed 0 \
      --niters 6000 --early_stop_patience 0 --test_freq 50 --grad_clip 1.0 --lr "$LR" \
      --warm_start_readout "$WS" --lr_decay_every 500 --lr_decay_gamma 0.5 --gpu -1 \
      --results_json "$OUT" > "'"$LOGS"'/${TAG}.log" 2>&1 \
    && echo "done  $TAG" || { echo "FAIL  $TAG"; exit 1; }
'
"$(pwd)/.venv/bin/python" sweep_manifest.py finish "$SWEEP"
echo "config selection $SWEEP finished"
