#!/usr/bin/env bash
# Reusable short runtime checks; ROOT holds the staged repo and compiled solver.
set -euo pipefail
root=$1
runs=$2
test ! -e "$runs" || { echo "refusing existing runtime-smoke directory: $runs" >&2; exit 2; }
mkdir "$runs"
exe=$root/sparta-build/src/spa_mpi_bigbig
for ranks in 1 4; do
  out=$runs/he-parse-$ranks
  mkdir "$out"
  cp "$root/repo/cases/data/he.species" "$root/repo/cases/data/he.vss" "$out/"
  cp "$root/repo/cases/microtests/stage1-parse/in.he_parse" "$out/"
  if ((ranks > 1)); then
    # The original microtest was single-rank. gridcut=0 requires a clumped
    # parallel partition: change only decomposition in this derived copy.
    sed -i '/^create_grid/a balance_grid rcb cell' "$out/in.he_parse"
  fi
  (cd "$out"; mpirun -np "$ranks" "$exe" -in in.he_parse > run.log 2>&1)
done
out=$runs/he-stub-3d
mkdir "$out"
cp "$root/repo/cases/data/he.species" "$root/repo/cases/data/he.vss" "$out/"
cp "$root/repo/cases/microtests/emit-surf/"*.surf "$out/"
cp "$root/repo/cases/microtests/emit-surf/in.mflow_stub3d" "$out/"
# Reduced statistics/duration: runtime smoke, not the original steady flux gate.
(cd "$out"; mpirun -np 4 "$exe" -in in.mflow_stub3d \
  -var CAPM diffuse -var MFLOW 3.3076174e-8 -var NRHO 3.5e22 \
  -var FILLN 3.5e22 -var FNUM 1e11 -var DT 3e-7 -var STEPS 1000 \
  -var SEED 20260924 > run.log 2>&1)
python3 - "$runs" <<'PY'
import hashlib,json,pathlib,sys
runs=pathlib.Path(sys.argv[1]); checks={}
for name,step,deck in [('he-parse-1',100,'in.he_parse'),('he-parse-4',100,'in.he_parse'),('he-stub-3d',1000,'in.mflow_stub3d')]:
    text=(runs/name/'run.log').read_text()
    rows=[line.split() for line in text.splitlines() if line.split() and line.split()[0]==str(step)]
    assert rows and 'ERROR' not in text and 'Loop time of' in text,name
    row=rows[-1]
    assert int(row[2])>0 and int(row[4])>0,(name,row)
    if name.startswith('he-parse'): assert int(row[2])==10000,(name,row)
    else: assert int(row[6])>0,(name,row)
    checks[name]={'final_stats':row,'status':'pass',
        'deck_sha256':hashlib.sha256((runs/name/deck).read_bytes()).hexdigest()}
(runs/'sparta-smoke.json').write_text(json.dumps({
    'scope':'runtime smoke, not steady-state or convergence evidence',
    'parallel_parse_adaptation':'balance_grid rcb cell after create_grid',
    'checks':checks},indent=2)+'\n')
print(json.dumps(checks))
PY
