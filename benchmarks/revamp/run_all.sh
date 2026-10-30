#!/bin/bash
# The profiling runs of REVAMP.md section 2, one after the other, with per-run timeouts.
# Usage, from the repository root: benchmarks/revamp/run_all.sh [OUT_DIR]
#   PY:   Python interpreter (default: python)
#   WRAP: command prefix for each run, e.g. "taskset -c 3 prlimit --as=3500000000 --" as used for REVAMP.md
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-revamp_out}
PY=${PY:-python}
WRAP=${WRAP:-}
mkdir -p "$OUT"
LOG=$OUT/run_all.log
export PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg

run() { # timeout, then flatbench arguments
  local t=$1
  shift
  echo "$(date +%T) START $*" >> "$LOG"
  # shellcheck disable=SC2086 # WRAP is a command prefix and must be word-split
  timeout -k 10 "$t" $WRAP "$PY" "$HERE/flatbench.py" "$@" --out "$OUT" >> "$LOG" 2>&1
  echo "$(date +%T) END rc=$? $*" >> "$LOG"
}

peak_step() { # scenario -> step with the most agents on the map in its "b" timing run
  "$PY" -c "
import json
d = json.load(open('$OUT/$1_b_timing.json'))
s = [x for x in d['reps'][-1]['windows']['early']['steps'] if 'n_on_map' in x]
print(max(s, key=lambda x: x['n_on_map'])['t'])"
}

for c in b d a c; do run 300 S $c timing --reps 5; done
run 900 M b timing --reps 5
run 600 M b timing --reps 2 --seed 2 --tag _seed2
run 600 M b timing --reps 2 --seed 3 --tag _seed3
T0=$(peak_step M)
echo "M T0=$T0" >> "$LOG"
run 900 M d timing --reps 3
run 900 M a timing --reps 3 --early 50 --late 50 --t0 "$T0"
run 900 M c timing --reps 3 --early 30 --late 30 --t0 "$T0"
for c in a b c d; do run 300 S $c cprofile; done
run 600 M b cprofile
run 600 M d cprofile --early 200
run 600 M a cprofile --early 20 --late 20 --t0 "$T0"
run 600 M c cprofile --early 15 --late 15 --t0 "$T0"
run 900 L b timing
T0L=$(peak_step L)
echo "L T0=$T0L" >> "$LOG"
run 900 L a timing --early 10 --late 10 --t0 "$T0L"
run 1200 L c timing --early 10 --late 10 --t0 "$T0L"
run 900 L b cprofile --early 200
run 1200 L c cprofile --early 5
run 900 M b timing --reps 3 --tag _pass2
echo "$(date +%T) ALL DONE" >> "$LOG"
