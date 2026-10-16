#!/bin/bash
# run_node_sweep.sh: the optimize-then-discretize (neural ODE) sweep behind
# tables/table3.tex.
#
#   3 datasets x 4 arms x 5 seeds = 60 runs
#
# Arms. 'none' is the static-weight neural ODE (d=0). 'monomial-raw' is the
# unscaled control: the monomial basis on [0,T] instead of tau in [-1,1], so the
# time-scaling claim is tested here as well as in the MATLAB half.
#
# Budget: batch size is nTrain/20 per dataset (--batch_size 0), so 6000
# iterations is 300 epochs for every dataset.
#
# The learning rate and readout warm start are read from dataset_config.py,
# which records them per dataset as selected in sweep 2026-10-05-node-config,
# so the script and the record cannot drift apart.
#
# Usage: ./run_node_sweep.sh <sweepId> [nWorkers]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SWEEP="${1:?usage: run_node_sweep.sh <sweepId> [nWorkers]}"
NPROC="${2:-8}"
ROOT="$(cd .. && pwd)"
RAW="$ROOT/results/raw/$SWEEP/python"
LOGS="$ROOT/results/raw/$SWEEP/logs"
PY="$(pwd)/.venv/bin/python"
mkdir -p "$RAW" "$LOGS"
"$(pwd)/.venv/bin/python" sweep_manifest.py start "$SWEEP" 60 "Neural ODE: 3 datasets x 4 arms (none, legendre, monomial, monomial-raw) x 5 seeds, continuous adjoint, settings from dataset_config.py."

# One BLAS thread per run: the pool already uses one core per run.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$(pwd)"

# Emit the job list, then run it through a fixed-size pool.
for SEED in 0 1 2 3 4; do
  for DATASET in ELM CDR DCR; do
    for BASIS in none legendre monomial monomial-raw; do
      DEG=3; [ "$BASIS" == "none" ] && DEG=0
      echo "$DATASET $BASIS $DEG $SEED"
    done
  done
done | xargs -P "$NPROC" -L 1 bash -c '
  set -euo pipefail
  DATASET=$0 BASIS=$1 DEG=$2 SEED=$3
  TAG="${DATASET}_NODE_${BASIS}_d${DEG}_T1_adam_s${SEED}"
  OUT="'"$RAW"'/${TAG}.json"
  if [ -f "$OUT" ]; then echo "skip  $TAG (already done)"; exit 0; fi
  echo "start $TAG"
  read LR WS < <('"$PY"' -c "import dataset_config,sys; c=dataset_config.get(sys.argv[1]); print(c[\"lr\"], c[\"warm_start\"])" "$DATASET")
  '"$PY"' surrogate/train_surrogate_node.py \
      --dataset "$DATASET" --data_dir ../data --basis "$BASIS" --degree "$DEG" \
      --lr "$LR" --warm_start_readout "$WS" \
      --seed "$SEED" --niters 6000 --early_stop_patience 0 --test_freq 50 \
      --grad_clip 1.0 --lr_decay_every 500 --lr_decay_gamma 0.5 --gpu -1 \
      --results_json "$OUT" > "'"$LOGS"'/${TAG}.log" 2>&1 \
    && echo "done  $TAG" || { echo "FAIL  $TAG"; exit 1; }
'
"$(pwd)/.venv/bin/python" sweep_manifest.py finish "$SWEEP"
echo "sweep $SWEEP finished"
