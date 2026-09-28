#!/usr/bin/env bash
# Pinned CPU MPI SPARTA recipe. Running this installs a new local solver build.
set -euo pipefail
GATE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/ctest_gate.py

variant=''
print_plan=0
while (($#)); do
  case "$1" in
    --variant) variant=${2:-}; shift 2 ;;
    --print-plan) print_plan=1; shift ;;
    *) echo "usage: $0 --variant plain|bigbig [--print-plan]" >&2; exit 2 ;;
  esac
done
case "$variant" in
  plain) preset=mpi; machine=mpi ;;
  bigbig) preset=mpi_big; machine=mpi_bigbig ;;
  *) echo 'variant must be plain or bigbig' >&2; exit 2 ;;
esac

SPARTA_URL=${SPARTA_URL:-https://github.com/sparta/sparta.git}
SPARTA_TAG=${SPARTA_TAG:-27Aug2026}
SPARTA_COMMIT=${SPARTA_COMMIT:-95b9abaa8bd548991cc3c3f1c58b34722f7ade74}
SPARTA_SOURCE_ROOT=${SPARTA_SOURCE_ROOT:-$HOME/src/sparta-$SPARTA_TAG}
SPARTA_BUILD_ROOT=${SPARTA_BUILD_ROOT:-$HOME/build/sparta-$SPARTA_TAG-$variant}
SPARTA_INSTALL_ROOT=${SPARTA_INSTALL_ROOT:-$HOME/opt/sparta-$SPARTA_TAG-$variant}
BUILD_JOBS=${BUILD_JOBS:-4}
# CTest includes four-rank MPI cases, so parallel test slots multiply ranks.
TEST_JOBS=${TEST_JOBS:-1}
case "$BUILD_JOBS:$TEST_JOBS" in *[!0-9:]*|:*|*:|0:*|*:0) echo 'BUILD_JOBS and TEST_JOBS must be positive integers' >&2; exit 2 ;; esac

if ((print_plan)); then
  cat <<EOF
Pinned source: $SPARTA_URL tag $SPARTA_TAG commit $SPARTA_COMMIT
Source root: $SPARTA_SOURCE_ROOT (must be clean, including when already at pinned HEAD)
Preset: $SPARTA_SOURCE_ROOT/cmake/presets/$preset.cmake
Expected flags: -DSPARTA_GZIP $([ "$variant" = bigbig ] && printf '%s' '-DSPARTA_BIGBIG')
cmake -C $SPARTA_SOURCE_ROOT/cmake/presets/$preset.cmake -S $SPARTA_SOURCE_ROOT/cmake -B $SPARTA_BUILD_ROOT -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$SPARTA_INSTALL_ROOT -DSPARTA_MACHINE=$machine -DSPARTA_ENABLE_TESTING=ON
cmake --build $SPARTA_BUILD_ROOT --parallel $BUILD_JOBS
ctest --test-dir $SPARTA_BUILD_ROOT --output-on-failure -j $TEST_JOBS | tee $SPARTA_BUILD_ROOT/ctest-full.log
python3 $GATE $SPARTA_BUILD_ROOT/ctest-full.log --variant $variant
cmake --install $SPARTA_BUILD_ROOT
Executable: $SPARTA_INSTALL_ROOT/bin/spa_$machine
EOF
  exit 0
fi

for program in git cmake ctest python3 sha256sum; do
  command -v "$program" >/dev/null || { echo "missing prerequisite: $program" >&2; exit 3; }
done
test -f "$GATE" || { echo "missing CTest gate: $GATE" >&2; exit 3; }
if [ -e "$SPARTA_BUILD_ROOT" ] || [ -e "$SPARTA_INSTALL_ROOT" ]; then
  echo 'refusing existing build/install root; choose fresh paths' >&2; exit 4
fi
if [ ! -e "$SPARTA_SOURCE_ROOT" ]; then
  git clone "$SPARTA_URL" "$SPARTA_SOURCE_ROOT"
elif [ ! -e "$SPARTA_SOURCE_ROOT/.git" ]; then
  echo "source root is not Git checkout: $SPARTA_SOURCE_ROOT" >&2; exit 4
fi
if [ -n "$(git -C "$SPARTA_SOURCE_ROOT" status --porcelain)" ]; then
  echo "refusing dirty SPARTA source checkout: $SPARTA_SOURCE_ROOT" >&2; exit 4
fi
actual=$(git -C "$SPARTA_SOURCE_ROOT" rev-parse HEAD)
if [ "$actual" != "$SPARTA_COMMIT" ]; then
  git -C "$SPARTA_SOURCE_ROOT" fetch --tags origin
  git -C "$SPARTA_SOURCE_ROOT" checkout --detach "$SPARTA_COMMIT"
fi
actual=$(git -C "$SPARTA_SOURCE_ROOT" rev-parse HEAD)
if [ "$actual" != "$SPARTA_COMMIT" ] || [ -n "$(git -C "$SPARTA_SOURCE_ROOT" status --porcelain)" ]; then
  echo 'source is not a clean copy of the requested commit' >&2; exit 5
fi
preset_path=$SPARTA_SOURCE_ROOT/cmake/presets/$preset.cmake
defaults_path=$SPARTA_SOURCE_ROOT/cmake/common/set/sparta_cmake_defaults.cmake
grep -q -- '-DSPARTA_GZIP' "$defaults_path" || { echo 'pinned source lacks default GZIP compile flag' >&2; exit 5; }
if [ "$variant" = bigbig ]; then
  grep -q -- '-DSPARTA_BIGBIG' "$preset_path" || { echo 'pinned BIGBIG preset lacks compile flag' >&2; exit 5; }
fi

cmake -C "$preset_path" -S "$SPARTA_SOURCE_ROOT/cmake" -B "$SPARTA_BUILD_ROOT" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$SPARTA_INSTALL_ROOT" \
  -DSPARTA_MACHINE="$machine" -DSPARTA_ENABLE_TESTING=ON
python3 - "$SPARTA_BUILD_ROOT/CMakeCache.txt" "$variant" "$machine" <<'PY'
import pathlib, sys
cache = pathlib.Path(sys.argv[1]).read_text()
variant, machine = sys.argv[2:]
required = (f"SPARTA_MACHINE:STRING={machine}", "-DSPARTA_GZIP", "SPARTA_ENABLE_TESTING:BOOL=ON")
if variant == "bigbig":
    required += ("-DSPARTA_BIGBIG",)
missing = [item for item in required if item not in cache]
if missing:
    raise SystemExit("CMake cache lacks required options: " + ", ".join(missing))
PY
cmake --build "$SPARTA_BUILD_ROOT" --parallel "$BUILD_JOBS"
# The parsed-log gate, not ctest's exit status, decides whether to install:
# BIGBIG accepts exactly the two documented circle.constant Maxlevel-cap
# failures and nothing else (ctest_gate.py). No `ctest -E`.
{ ctest --test-dir "$SPARTA_BUILD_ROOT" --output-on-failure -j "$TEST_JOBS" 2>&1 || true; } \
  | tee "$SPARTA_BUILD_ROOT/ctest-full.log"
python3 "$GATE" "$SPARTA_BUILD_ROOT/ctest-full.log" --variant "$variant"
cmake --install "$SPARTA_BUILD_ROOT"
exe=$SPARTA_INSTALL_ROOT/bin/spa_$machine
test -x "$exe"
python3 - "$exe.build.json" "$exe" "$SPARTA_URL" "$SPARTA_TAG" "$SPARTA_COMMIT" "$variant" "$preset_path" "$SPARTA_BUILD_ROOT/CMakeCache.txt" <<'PY'
import hashlib, json, pathlib, subprocess, sys
out, exe, url, tag, commit, variant, preset, cache = sys.argv[1:]
def first_line(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return (result.stdout or result.stderr).splitlines()[0]
    except (OSError, IndexError, subprocess.TimeoutExpired):
        return None
lines = pathlib.Path(cache).read_text().splitlines()
options = {key: line.split("=", 1)[1] for line in lines if ":" in line and "=" in line
           for key in ("SPARTA_MACHINE", "SPARTA_CXX_COMPILE_FLAGS", "SPARTA_DEFAULT_CXX_COMPILE_FLAGS",
                       "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "MPI_CXX_COMPILER")
           if line.startswith(key + ":")}
payload = {"source_url": url, "tag": tag, "commit": commit, "variant": variant,
           "preset": str(pathlib.Path(preset).resolve()), "cmake_options": options,
           "cmake_version": first_line(["cmake", "--version"]),
           "mpi_version": first_line(["mpirun", "--version"]),
           "compiler_version": first_line(["mpic++", "--version"]),
           "executable": str(pathlib.Path(exe).resolve()),
           "executable_sha256": hashlib.sha256(pathlib.Path(exe).read_bytes()).hexdigest()}
pathlib.Path(out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
printf 'Built %s\n' "$exe"
