#!/usr/bin/env bash
# Launch any sweep driver, mint or resume its sweep, collect the ledger.
#
#   ./runNamedSweep.sh <driverFunction> <slug> [nWorkers] [sweepId] [note]
#
#   ./runNamedSweep.sh runConvergenceSweep convergence 5
#   ./runNamedSweep.sh runConvergenceSweep convergence 5 2026-10-05-convergence
#
# Passing an existing sweepId resumes it: every driver skips runs whose .mat is
# already on disk, so a killed sweep continues into the same directory.
set -uo pipefail
cd "$(dirname "$0")"
DRIVER="${1:?usage: runNamedSweep.sh <driverFunction> <slug> [nWorkers] [sweepId] [note]}"
SLUG="${2:?missing slug}"
NW="${3:-5}"
RESUME="${4:-}"
NOTE="${5:-}"
# Prefer MATLAB on PATH; fall back to the macOS install location.
MATLAB="${MATLAB:-$(command -v matlab || echo /Applications/MATLAB_R2026a.app/bin/matlab)}"
[ -x "$MATLAB" ] || { echo "no MATLAB at $MATLAB; set MATLAB=/path/to/matlab" >&2; exit 1; }

[ -f "$DRIVER.m" ] || { echo "no such driver: $DRIVER.m" >&2; exit 1; }

echo "== tests first =="
"$MATLAB" -batch "cd('../tests'); runTests" || { echo "tests failed; refusing to sweep" >&2; exit 1; }

if [ -n "$RESUME" ]; then
    SWEEP="$RESUME"
    [ -f "../../results/sweeps/$SWEEP.json" ] || { echo "no manifest for $SWEEP" >&2; exit 1; }
    echo "== resuming sweep $SWEEP with $DRIVER, $NW workers =="
else
    [ -n "$NOTE" ] || NOTE="$DRIVER; see its header for the grid"
    # MATLAB escapes a single quote by doubling it.
    ESCNOTE=${NOTE//\'/\'\'}
    SWEEP=$("$MATLAB" -batch "addpath(genpath('..')); id=newSweep('$SLUG','$ESCNOTE'); fprintf('SWEEPID=%s\n',id);" | sed -n 's/^SWEEPID=//p')
    [ -n "$SWEEP" ] || { echo "could not create sweep" >&2; exit 1; }
    echo "== sweep $SWEEP with $DRIVER, $NW workers =="
fi

LOGDIR="../../results/raw/$SWEEP/logs"
LOGSUF=""
[ -n "$RESUME" ] && LOGSUF="-resume$(date +%H%M)"
mkdir -p "$LOGDIR"
pids=()
for i in $(seq 1 "$NW"); do
    "$MATLAB" -batch "$DRIVER($i,$NW,'$SWEEP')" > "$LOGDIR/worker$i$LOGSUF.log" 2>&1 &
    pids+=($!)
done
echo "launched ${#pids[@]} workers; logs in results/raw/$SWEEP/logs/"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "== workers finished ($fail non-zero exits) =="

# Count what is on disk so a resumed sweep records its total.
OK=$(ls "../../results/raw/$SWEEP/matlab"/*.mat 2>/dev/null | wc -l | tr -d ' ')
FL=$(grep -h "^WORKER .* DONE" "$LOGDIR"/worker*"$LOGSUF".log | sed -E 's/.*fail=([0-9]+)/\1/' | paste -sd+ - | bc)
"$MATLAB" -batch "addpath(genpath('..')); finishSweep('$SWEEP',${OK:-0},${FL:-0}); collectResults; checkProvenance;"
echo "== done: sweep $SWEEP =="
