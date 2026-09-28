#!/usr/bin/env bash
# Create a new Python venv + exclusive Julia depot for one DSMC root, and record
# hash-bound receipts. Tracked port of the package014 rehearsal installer
# (clean-environment rehearsal 2026-09-24, Engaging job 23708010).
# Run on a LOGIN node (ORCD: installs on login nodes, compute in jobs).
#
#   DSMC_ROOT=$W bash "$W/repo/tools/engaging/setup-env.sh"
#
# Required:  DSMC_ROOT  absolute working root. Its repo/ is a git clone or an
#                       extracted `dsmc_repository.py export`; the outputs
#                       python-venv/ julia-depot/ config/ env/ logs/ must be absent.
# Optional:  DSMC_REPO               source tree (default $DSMC_ROOT/repo)
#            SOURCE_ARCHIVE + SOURCE_ARCHIVE_SHA256
#                                    extract an immutable export into an absent repo/
#            EXPECTED_COMMIT         refuse unless the source is this commit
#            EXPECTED_MANIFEST_SHA256  refuse unless the export manifest matches
#            SITE_MODULE_SETUP       module-load script to copy and bind
#                                    (default repo/tools/engaging/smoke-setup.sh)
#            SOLVER_EXECUTABLE       record a pre-existing solver's hash here too
#            PIP_WHEELHOUSE          install offline from this wheel directory
# Writes: config/setup.sh (source it in every job and shell), env/environment.json,
#         env/inputs.sha256, env/python-freeze.txt, logs/*.
set -Eeuo pipefail
: "${DSMC_ROOT:?set DSMC_ROOT to your absolute working root}"
root=$(realpath -m -- "$DSMC_ROOT")
repo=$(realpath -m -- "${DSMC_REPO:-$root/repo}")
[[ "$root" = /* && "$repo" = /* ]] || { echo 'DSMC_ROOT/DSMC_REPO must be absolute POSIX paths' >&2; exit 2; }
for d in python-venv julia-depot config env logs; do
  [[ ! -e "$root/$d" && ! -L "$root/$d" ]] || { echo "refusing existing $root/$d; choose a new DSMC_ROOT" >&2; exit 2; }
done
mkdir -p "$root"
mkdir -m 700 "$root"/{config,env,logs}

if [[ -n "${SOURCE_ARCHIVE:-}" ]]; then
  : "${SOURCE_ARCHIVE_SHA256:?SOURCE_ARCHIVE needs SOURCE_ARCHIVE_SHA256}"
  [[ ! -e "$repo" && -f "$SOURCE_ARCHIVE" && ! -L "$SOURCE_ARCHIVE" ]] || {
    echo 'SOURCE_ARCHIVE must be a regular file and repo/ must be absent' >&2; exit 2; }
  install -m 600 "$SOURCE_ARCHIVE" "$root/source.tar.gz"
  printf '%s  %s\n' "$SOURCE_ARCHIVE_SHA256" "$root/source.tar.gz" | sha256sum -c -
  mkdir -m 700 "$repo"
  tar --no-same-owner --no-same-permissions -xzf "$root/source.tar.gz" -C "$repo"
  [[ -f "$repo/source-export.json" && ! -L "$repo/source-export.json" ]] || {
    echo 'source export manifest is missing or unsafe' >&2; exit 2; }
fi
[[ -d "$repo/tools" && -f "$repo/requirements-3d.txt" ]] || { echo "not a DSMC-sims tree: $repo" >&2; exit 2; }

module_setup=$(realpath -m -- "${SITE_MODULE_SETUP:-$repo/tools/engaging/smoke-setup.sh}")
[[ -f "$module_setup" && ! -L "$module_setup" ]] || { echo "module setup must be a regular file: $module_setup" >&2; exit 2; }
solver=
if [[ -n "${SOLVER_EXECUTABLE:-}" ]]; then
  solver=$(realpath -m -- "$SOLVER_EXECUTABLE")
  [[ -x "$solver" && ! -L "$SOLVER_EXECUTABLE" ]] || { echo "solver must be an executable regular file: $solver" >&2; exit 2; }
fi

install -m 700 "$module_setup" "$root/config/site-module-setup.sh"
source "$root/config/site-module-setup.sh"
command -v python3 julia mpirun >/dev/null
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11+ required: " + sys.version)'

# Source identity: git commit (clean tree) or export commit + manifest hash.
read -r commit manifest < <(python3 -c '
import sys; sys.path.insert(0, sys.argv[1] + "/tools")
from dsmc_repository import repository_identity
i = repository_identity(sys.argv[1]); print(i["commit"], i.get("manifest_sha256") or "-")' "$repo")
[[ -n "${commit:-}" ]] || { echo "cannot identify source commit in $repo" >&2; exit 2; }
[[ -z "${EXPECTED_COMMIT:-}" || "$commit" = "$EXPECTED_COMMIT" ]] || { echo "source commit $commit != EXPECTED_COMMIT" >&2; exit 2; }
[[ -z "${EXPECTED_MANIFEST_SHA256:-}" || "$manifest" = "$EXPECTED_MANIFEST_SHA256" ]] || { echo 'export manifest differs from EXPECTED_MANIFEST_SHA256' >&2; exit 2; }
verify=(python3 "$repo/tools/dsmc_repository.py" verify --root "$repo" --commit "$commit")
[[ "$manifest" = - ]] || verify+=(--manifest-sha256 "$manifest")
"${verify[@]}" | tee "$root/logs/source-verify-before.json"
# The maintained launchers use --project=tracer. Keep its committed inputs immutable.
chmod a-w "$repo/tracer/Project.toml" "$repo/tracer/Manifest.toml"

python3 -m venv "$root/python-venv"
venv_python="$root/python-venv/bin/python"
export PIP_CONFIG_FILE=/dev/null PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONNOUSERSITE=1
if [[ -n "${PIP_WHEELHOUSE:-}" ]]; then
  "$venv_python" -I -m pip install --isolated --no-index --find-links "$PIP_WHEELHOUSE" -r "$repo/requirements-3d.txt" |& tee "$root/logs/pip-install.log"
else
  "$venv_python" -I -m pip install --isolated -r "$repo/requirements-3d.txt" |& tee "$root/logs/pip-install.log"
fi
"$venv_python" -I -m pip check | tee "$root/logs/pip-check.log"
"$venv_python" -I -m pip freeze --all | LC_ALL=C sort > "$root/env/python-freeze.txt"

export JULIA_DEPOT_PATH="$root/julia-depot" JULIA_PROJECT="$repo/tracer" JULIA_NUM_PRECOMPILE_TASKS=1 JULIA_CPU_TARGET=generic
mkdir -m 700 "$JULIA_DEPOT_PATH"
julia --startup-file=no --project="$JULIA_PROJECT" --threads=1 -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); Pkg.status()' |& tee "$root/logs/julia-instantiate.log"
# A resolver that rewrote Project/Manifest fails here instead of passing silently.
"$venv_python" -I "${verify[@]:1}" | tee "$root/logs/source-verify-after.json"

# ponytail: no `set -e` here, unlike the rehearsal's rehearsal-setup.sh: this
# file is also sourced in interactive shells, where set -e ends the login on the
# first failing command. Every sbatch sets its own strict mode.
cat > "$root/config/setup.sh" <<EOF
#!/usr/bin/env bash
source $(printf '%q' "$root/config/site-module-setup.sh")
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export JULIA_DEPOT_PATH=$(printf '%q' "$root/julia-depot")
export JULIA_PROJECT=$(printf '%q' "$repo/tracer")
export JULIA_CPU_TARGET=generic
EOF
chmod 700 "$root/config/setup.sh"
sha256sum "$root/config/site-module-setup.sh" "$root/config/setup.sh" "$repo/requirements-3d.txt" \
  "$repo/tracer/Project.toml" "$repo/tracer/Manifest.toml" ${solver:+"$solver"} "$root/env/python-freeze.txt" > "$root/env/inputs.sha256"
DSMC_ROOT="$root" REPO="$repo" SOLVER="$solver" VENV_PYTHON="$venv_python" SOURCE_COMMIT="$commit" \
SOURCE_MANIFEST="$manifest" SOURCE_ARCHIVE_SHA256="${SOURCE_ARCHIVE_SHA256:-}" "$venv_python" -I - <<'PY'
import hashlib, importlib, json, os, shutil, sys
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
root = Path(os.environ['DSMC_ROOT'])
repo = Path(os.environ['REPO'])
venv = Path(os.environ['VENV_PYTHON']).parent.parent
if Path(sys.prefix) != venv or Path(sys.base_prefix) == Path(sys.prefix):
    raise SystemExit('environment receipt is not running in the new venv')
modules = {}
for name in ('numpy', 'pandas', 'scipy', 'psutil'):
    module_path = Path(importlib.import_module(name).__file__).resolve()
    try:
        module_path.relative_to(venv.resolve())
    except ValueError as exc:
        raise SystemExit(f'{name} imported outside new venv: {module_path}') from exc
    modules[name] = str(module_path)
solver = os.environ['SOLVER']
record = {
    'schema_version': 'dsmc-engaging-environment-v1', 'status': 'environment-ready',
    'source': {'repo': str(repo), 'commit': os.environ['SOURCE_COMMIT'],
               'manifest_sha256': None if os.environ['SOURCE_MANIFEST'] == '-' else os.environ['SOURCE_MANIFEST'],
               'archive_sha256': os.environ['SOURCE_ARCHIVE_SHA256'] or None},
    'python': {'venv': os.environ['VENV_PYTHON'], 'invoked_executable': sys.executable, 'sys_prefix': sys.prefix,
               'sys_base_prefix': sys.base_prefix, 'version': sys.version,
               'freeze_sha256': sha(root / 'env/python-freeze.txt'), 'module_paths': modules},
    'julia': {'executable': shutil.which('julia'), 'depot': str(root / 'julia-depot'), 'project': str(repo / 'tracer'),
              'project_sha256': sha(repo / 'tracer/Project.toml'), 'manifest_sha256': sha(repo / 'tracer/Manifest.toml'),
              'depot_path_mode': 'exclusive', 'cpu_target': os.environ.get('JULIA_CPU_TARGET')},
    'mpi': {'executable': shutil.which('mpirun')},
    'solver': {'path': solver, 'sha256': sha(solver)} if solver else None,
    'setup': {'copied_module_setup_sha256': sha(root / 'config/site-module-setup.sh'),
              'setup_sha256': sha(root / 'config/setup.sh')},
}
(root / 'env/environment.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
PY
echo "environment recorded at $root/env/environment.json; source $root/config/setup.sh in every shell and job"
