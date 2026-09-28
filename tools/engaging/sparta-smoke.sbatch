#!/usr/bin/env bash
#SBATCH --job-name=dsmc-sparta-smoke
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=sparta-%j.log
set -euo pipefail
root=$SLURM_SUBMIT_DIR
src=$root/sparta-source
build=$root/sparta-build
# One job per directory. Refuse before the tee and exit trap below would
# overwrite an earlier job's evidence (review F1: this used to exit 1 silently).
for f in "$build" "$root/sparta.exit" "$root/sparta-progress.log" "$root/acceptance.txt"; do
  test ! -e "$f" || { echo "refusing to reuse $root: $f exists. Submit from a new directory." >&2; exit 6; }
done
source "$root/smoke-setup.sh"
exec > >(tee "$root/sparta-progress.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$root/sparta.exit"' EXIT
date -u +%FT%TZ
hostname
module -t list
lscpu
mpirun -np 4 hostname
# Fail before an hour of build if an input is missing. $root/repo is the staged repository.
gate=${CTEST_GATE:-$root/repo/tools/wsl/ctest_gate.py}
for f in "$gate" "$root/repo/cases/microtests/stage2-collide/analyze.py" \
    "$root/sparta-runtime-smokes.sh" "$root/he-collision-runs.sh"; do
  test -f "$f" || { echo "missing input: $f" >&2; exit 3; }
done
# Pinned source: the same defaults as tools/wsl/build_sparta_cpu.sh:21-23;
# tools/engaging/test_engaging_setup.py asserts they agree.
SPARTA_URL=${SPARTA_URL:-https://github.com/sparta/sparta.git}
SPARTA_TAG=${SPARTA_TAG:-27Aug2026}
SPARTA_COMMIT=${SPARTA_COMMIT:-95b9abaa8bd548991cc3c3f1c58b34722f7ade74}
# Compute-node network is unverified; prefer cloning on the login node first.
if [ ! -e "$src" ]; then git clone "$SPARTA_URL" "$src"; fi
if [ -e "$src/.git" ]; then
  test -z "$(git -C "$src" status --porcelain)" || { echo "refusing dirty SPARTA source: $src" >&2; exit 4; }
  if [ "$(git -C "$src" rev-parse HEAD)" != "$SPARTA_COMMIT" ]; then
    git -C "$src" fetch --tags origin
    git -C "$src" checkout --detach "$SPARTA_COMMIT"
  fi
  test "$(git -C "$src" rev-parse HEAD)" = "$SPARTA_COMMIT" || { echo 'SPARTA commit mismatch' >&2; exit 5; }
  printf '%s %s %s\n' "$SPARTA_URL" "$SPARTA_TAG" "$SPARTA_COMMIT" > "$root/sparta-source.pin"
else
  # Transferred `git archive` trees (the 2026-09-24 route) carry no .git to check.
  echo "WARNING: $src is not a git checkout; pin $SPARTA_COMMIT not verified" >&2
  printf 'unverified: not a git checkout (expected %s)\n' "$SPARTA_COMMIT" > "$root/sparta-source.pin"
fi
cmake -C "$src/cmake/presets/mpi_big.cmake" -S "$src/cmake" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release -DSPARTA_MACHINE=mpi_bigbig \
  -DSPARTA_ENABLE_TESTING=ON -DCMAKE_INSTALL_PREFIX="$root/sparta-install"
grep -E 'SPARTA_(MACHINE|.*COMPILE_FLAGS)|MPI_CXX_COMPILER' "$build/CMakeCache.txt"
cmake --build "$build" --parallel 8
exe=$build/src/spa_mpi_bigbig
sha256sum "$exe" > "$root/sparta-executable.sha256"
ldd "$exe" > "$root/sparta-linkage.txt"
set +e
ctest --test-dir "$build" --output-on-failure -j 1 > "$root/ctest.log" 2>&1
ctest_rc=$?
set -e
printf '%s\n' "$ctest_rc" > "$root/ctest.exit"
tail -25 "$root/ctest.log"
set +e
python3 "$gate" "$root/ctest.log" --variant bigbig > "$root/ctest-gate.txt" 2>&1
gate_rc=$?
set -e
printf '%s\n' "$gate_rc" > "$root/ctest-gate.exit"
cat "$root/ctest-gate.txt"
# Acceptance is G10 option C (Arian, 2026-09-27): the runtime smokes plus the
# hard-sphere collision-rate check. ctest and its WSL gold gate above are
# information only: Engaging's libm makes trajectories diverge from the golds,
# which SPARTA's manual says to expect across machines (review F2).
set +e
bash "$root/sparta-runtime-smokes.sh" "$root" "$root/runtime-smokes"
smoke_rc=$?
bash "$root/he-collision-runs.sh" "$root" "$root/he-collision"
collide_rc=$?
set -e
verdict=PASS
if [ "$smoke_rc" != 0 ] || [ "$collide_rc" != 0 ]; then verdict=FAIL; fi
{
  echo "acceptance $verdict: runtime smokes exit $smoke_rc, He collision-rate check exit $collide_rc"
  echo "  runtime smokes: runtime-smokes/sparta-smoke.json"
  echo "  collision rate (criterion: cases/microtests/stage2-collide/analyze.py):"
  grep -H -e '^R = ' -e '^VERDICT' "$root"/he-collision/*/analysis.txt || true
  echo "information only (not part of acceptance): ctest exit $ctest_rc; $(head -n 1 "$root/ctest-gate.txt")"
} > "$root/acceptance.txt"
cat "$root/acceptance.txt"
test "$verdict" = PASS
