#!/usr/bin/env bash
# Hard-sphere He collision-rate check (cases/microtests/stage2-collide) at 1 and
# 4 ranks. The criterion is analyze.py's own pre-registered one (exit 0 only on
# |R - 1| <= 0.05); its constants come from that file and the copied deck.
# Exit 0 only if both rank counts PASS. ROOT holds the staged repo and solver.
set -uo pipefail
root=$1
runs=$2
test ! -e "$runs" || { echo "refusing existing collision-check directory: $runs" >&2; exit 2; }
mkdir "$runs" || exit 2
exe=$root/sparta-build/src/spa_mpi_bigbig
case=$root/repo/cases/microtests/stage2-collide
rc=0
for ranks in 1 4; do
  out=$runs/$ranks
  mkdir "$out"
  cp "$root/repo/cases/data/he.species" "$root/repo/cases/data/he.vss" "$case/in.he_collide" "$out/" || exit 3
  if ((ranks > 1)); then
    # Preserve the original single-rank fixture; adapt only the copied MPI layout.
    sed -i '/^create_grid/a balance_grid rcb cell' "$out/in.he_collide"
  fi
  if (cd "$out" && mpirun -np "$ranks" "$exe" -in in.he_collide > run.log 2>&1); then
    python3 "$case/analyze.py" "$out/run.log" > "$out/analysis.txt" 2>&1 || rc=1
  else
    echo "VERDICT: FAIL (solver exited nonzero; see run.log)" > "$out/analysis.txt"
    rc=1
  fi
  echo "he-collision $ranks rank(s):"
  cat "$out/analysis.txt"
done
exit "$rc"
