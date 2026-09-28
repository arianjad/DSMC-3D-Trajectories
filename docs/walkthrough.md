# Walkthrough: your own account on MIT Engaging, from clone to verified run

Status: **not yet run from this repository.** The steps use scripts that
exist in this release; no step below has been executed on Engaging from a
clone of `DSMC-3D-Trajectories`. If a step fails, stop, keep the output, and
report it (section 12).

Who this is for: someone with their **own** Engaging login and Slurm
allocation. You need ordinary SSH and Bash. You build your own Python and
Julia environments and your own SPARTA executable; nothing is borrowed from
another account.

| Step | What | Scheduler cost (requested) |
|---|---|---|
| 0 | Prerequisites | none |
| 1 | Clone the repository | none |
| 2 | Python and Julia environment | none (login node) |
| 3 | Build and accept SPARTA from source | 1 job: 8 CPUs, 24 GB, 1 h cap |
| 4 | Write your profiles | none |
| 5 | Prepare the B5 circle case | 1 job: 4 CPUs, 32 GB, 30 min cap |
| 6 | Submit the full workflow, check status | 1 job, rendered from your profile (4 CPUs by default) |
| 7 | Read the report | none |
| 8 | Copy to your desktop and verify, with one deliberate retry | none |
| 9 | Change one input (molecule source temperature) | 1 job: 1 CPU, 16 GB, 10 min cap |
| 10 | Deliberately invalid input | 1 job, as step 9 |
| 11 | Record and plot example trajectories | 1 job: 1 CPU, 16 GB, 10 min cap |
| 12 | Report what happened | none |

The B5 circle is a software execution test. A successful run says the
workflow works. It does **not** establish physical convergence.

Placeholders used below, replace every one:

| Placeholder | Meaning |
|---|---|
| `YOUR_KERBEROS` | your Engaging login name |
| `YOUR_ACCOUNT` | a Slurm account you may charge |
| `YOUR_PARTITION` | a partition that account may use |
| `RELEASE_TAG` | the release (tag or commit) you were told to use |

---

## 0. Prerequisites

- [ ] `ssh YOUR_KERBEROS@orcd-login.mit.edu` works, including Duo.
- [ ] You know an account/partition pair you may submit to. On a login node:
      `sacctmgr show assoc user=$USER format=account,partition`.
- [ ] Disk: budget at least 10 GB free where you put `$W` below. One full
      workflow archive is on the order of 1.2 GB, and the SPARTA build adds
      its own tree.
- [ ] The login node can reach GitHub, PyPI and the Julia package server.
      `setup-env.sh` accepts `PIP_WHEELHOUSE=<dir>` for offline Python
      installs.
- [ ] On your desktop (step 8 only): `ssh`/`scp`, and Python 3.11+ with
      `requirements-3d.txt` installed, in your own clone of this repository.

Write down the time you start each step (section 12).

---

## 1. Clone the repository

On an Engaging login node:

```bash
export W=$HOME/dsmc-work           # your working root; any NEW path is fine
test ! -e "$W" || { echo "choose a new W"; exit 1; }
mkdir -p "$W"
git clone https://github.com/arianjad/DSMC-3D-Trajectories.git "$W/repo"
git -C "$W/repo" checkout RELEASE_TAG
export REPO="$W/repo" ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION
```

Keep the clone clean: step 2 refuses a modified tree, and every run records
the commit it used.

---

## 2. Build your Python and Julia environment

On the login node (install on login nodes, compute in jobs):

```bash
DSMC_ROOT="$W" bash "$REPO/tools/engaging/setup-env.sh" 2>&1 | tee "$W/setup-env.log"
export PY="$W/python-venv/bin/python"
source "$W/config/setup.sh"
sha256sum -c "$W/env/inputs.sha256"
```

What `tools/engaging/setup-env.sh` does: refuses if any of `python-venv/`,
`julia-depot/`, `config/`, `env/`, `logs/` already exists under `$W`; loads
the modules in `tools/engaging/smoke-setup.sh` (copied into
`$W/config/site-module-setup.sh`); requires Python 3.11+; checks that your
clone is clean at its commit before and after Julia package resolution;
creates a new venv from `requirements-3d.txt`; instantiates the Julia project
in `tracer/` into your own `$W/julia-depot`; writes `$W/config/setup.sh`,
`$W/env/inputs.sha256` and `$W/env/environment.json`. Optional:
`EXPECTED_COMMIT=<sha>` refuses any other commit.

Expected: the last line starts `environment recorded at $W/env/environment.json`,
and `sha256sum -c` prints `OK` for every line. `config/setup.sh` has no
`set -e`, so sourcing it in your login shell is safe. If a module in
`smoke-setup.sh` no longer exists on Engaging, the script stops here: report
the `module load` error.

---

## 3. Build and accept SPARTA from source

The workflow needs a SPARTA MPI executable with 64-bit cell IDs, built from
upstream commit `95b9abaa8bd548991cc3c3f1c58b34722f7ade74` (tag `27Aug2026`).
One job builds it and decides whether to accept it:

```bash
mkdir "$W/sparta" && cd "$W/sparta"
cp "$REPO"/tools/engaging/{sparta-smoke.sbatch,smoke-setup.sh,sparta-runtime-smokes.sh,he-collision-runs.sh} .
ln -s "$REPO" repo   # the job reads the acceptance inputs from ./repo
# Clone on the login node, so the job never needs network from a compute node.
git clone https://github.com/sparta/sparta.git sparta-source
git -C sparta-source checkout --detach 95b9abaa8bd548991cc3c3f1c58b34722f7ade74
sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" sparta-smoke.sbatch | tee sparta-job.txt
```

The `--account`/`--partition` on the command line override the defaults
written inside `sparta-smoke.sbatch`.

What the job does (`tools/engaging/sparta-smoke.sbatch`):

- Refuses to start if `sparta-build/`, `sparta.exit`, `sparta-progress.log`
  or `acceptance.txt` already exists (exit 6). To retry, use a new directory.
  A missing input stops it before the build (exit 3).
- Refuses a modified `sparta-source/` (exit 4), and verifies the pinned
  commit (exit 5 on mismatch); records it in `sparta-source.pin`.
- Builds with the CMake `mpi_big` preset as `sparta-build/src/spa_mpi_bigbig`,
  and records its SHA-256 (`sparta-executable.sha256`) and linkage
  (`sparta-linkage.txt`).
- **Acceptance** (`acceptance.txt`) is two runtime checks:
  `sparta-runtime-smokes.sh` (three short runs that must finish without
  `ERROR` and with positive counters) and `he-collision-runs.sh`, the
  hard-sphere helium collision-rate test at 1 and 4 ranks. That test passes
  only if the measured rate is within 5 % of kinetic theory
  (`cases/microtests/stage2-collide/analyze.py`). The job's exit status is
  the verdict.
- It also runs SPARTA's own `ctest` suite and a regression gate on it. Their
  result is written on the last line of `acceptance.txt` as information only
  and does not affect acceptance; it can say FAIL on a correctly working
  build, because regression trajectories diverge across machines and math
  libraries.

When the job has finished:

```bash
sacct -j $(cat "$W/sparta/sparta-job.txt") --format=JobID,State,ExitCode,Elapsed   # expect COMPLETED 0:0
cat "$W/sparta/acceptance.txt"                  # first line: acceptance PASS
export SOLVER="$W/sparta/sparta-build/src/spa_mpi_bigbig"
```

If the first line says `acceptance FAIL`, stop and report `acceptance.txt`
and the `he-collision/*/analysis.txt` files.

---

## 4. Write your profiles

```bash
"$PY" "$REPO/tools/engaging/write_profiles.py" --root "$W" \
  --account "$ACCOUNT" --partition "$PARTITION" --solver "$SOLVER"
```

It prints three paths: `$W/site-profile.json` (the Slurm site profile),
`$W/solver-check-profile.json` (used by step 5's geometry check) and
`$W/profile-inputs.json` (hashes of both profiles and of the solver).

- It needs `$W/repo`, `$W/python-venv/bin/python` and `$W/config/setup.sh`
  from step 2, and refuses to overwrite existing profiles.
- Both profiles are checked with the same validators the workflow uses
  before anything is written.
- The site profile says `"site_runtime_verified": false`. Do not change that
  by hand: the readiness probe in step 5 writes the verified copy. A profile
  still marked unverified renders a job that refuses to run.
- Defaults: `--ranks 4`, 32 GB, 30 min, 40 GB disk cap. `--account none`
  omits `--account` from rendered jobs.

---

## 5. Prepare the B5 circle case (one short job)

The readiness probe must run inside an allocation.

```bash
export CASE="$REPO/cases/examples/b5-circle/case.json"
export DSMC_W="$W" DSMC_CASE="$CASE"
cat > "$W/prepare.sbatch" <<'BASH'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -Eeuo pipefail
source "$DSMC_W/config/setup.sh"
py="$DSMC_W/python-venv/bin/python"; repo="$DSMC_W/repo"
"$py" "$repo/tools/engaging/prepare_site.py" probe --profile "$DSMC_W/site-profile.json" --out "$DSMC_W/readiness"
"$py" "$repo/tools/dsmc.py" prepare "$DSMC_CASE" --out "$DSMC_W/prepared"
"$py" "$repo/tools/dsmc.py" inspect "$DSMC_W/prepared" --solver-check --profile "$DSMC_W/solver-check-profile.json" --out "$DSMC_W/checked"
BASH
sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --output="$W/prepare-%j.out" "$W/prepare.sbatch" | tee "$W/prepare-job.txt"
```

The probe checks the allocated runtime (Python packages, Julia and MPI
versions, free disk, the Julia tracer on a small synthetic fixture, and a
2-step SPARTA start at the profile's rank count) and writes
`$W/readiness/verified-profile.json`. `prepare` resolves the case into a new
directory; `inspect --solver-check` runs a zero-step SPARTA geometry check
and writes the checked preparation the full workflow uses.

When the job has finished:

```bash
sacct -j $(cat "$W/prepare-job.txt") --format=JobID,State,ExitCode,Elapsed
export PREPARED="$W/checked" PROFILE="$W/readiness/verified-profile.json"
"$PY" "$REPO/tools/engaging/prepare_site.py" verify --profile "$PROFILE"
```

Expected: `COMPLETED 0:0`, then JSON with `"status": "complete"`. If
verification fails, stop and keep the error. Never edit a receipt to make a
check pass.

---

## 6. Submit the full workflow and check status

```bash
export RUN="$W/runs/circle-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RUN" || { echo "choose a new RUN"; exit 1; }
mkdir -p "$RUN"
printf 'export W=%q\nexport RUN=%q\n' "$W" "$RUN" > "$RUN/operator.env"
"$PY" "$REPO/tools/dsmc.py" acceptance --suite slurm --submit \
  --prepared "$PREPARED" --profile "$PROFILE" --out "$RUN/acceptance"
```

Expected: `Slurm submitted: submitted`; the command returns immediately.
**Submit once.** If the reply is unclear (for example SSH dropped), look at
`$RUN/acceptance/` and `squeue -u $USER` before anything else. Re-running
with the same `--out` is refused.

Check status as often as you like:

```bash
"$PY" "$REPO/tools/dsmc.py" slurm-status --receipt "$RUN/acceptance/acceptance.json"
```

`queued` and `running` are normal. `complete` means the scheduler finished
**and** the full archive verified. `science_assessment: unassessed` is
expected for this example.

After a disconnect: `source "$RUN/operator.env"; source "$W/config/setup.sh"`,
then re-export `PY`, `REPO`, `ACCOUNT` and `PARTITION`.

---

## 7. Read the report

```bash
export ARCHIVE=$("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["profile"]["results_root"]+"/job-"+r["job_id"])' "$RUN/acceptance/submission.json")
echo "$ARCHIVE"
"$PY" -m json.tool "$ARCHIVE/workflow/molecules/report.json"
```

Read `launched_particles` (the denominator) and `forward_unique_particles`
(molecules that crossed the exit plane forward at least once). Plane event
rows count crossings, not distinct molecules.

| File below the archive | Contents |
|---|---|
| `case/case.json` | The case specification |
| `prepared/prepared.json` | Geometry, helium and molecule inputs after checking |
| `workflow/helium/manifest.json` | Helium run status, inputs and recorded outputs |
| `workflow/field/field-preparation.json` | The selected helium frame and how it was converted |
| `workflow/molecules/report.json` | Molecule counts, observables and numerical diagnostics |
| `workflow/molecules/events.csv`, `terminal.csv` | Plane crossings; each molecule's final state |

The archive also holds SPARTA restart files. Keep it as a unit.

---

## 8. Copy to your desktop and verify, with one deliberate retry

An incomplete copy must fail verification, and a fresh complete copy must
pass. There is no retry command: a retry is a new directory, a new copy and
a new verification.

On your desktop, in your own clone of this repository (Bash shown; Git Bash
works on Windows):

```bash
REMOTE='PASTE_THE_ARCHIVE_PATH_FROM_STEP_7'
LOGIN='YOUR_KERBEROS@orcd-login.mit.edu'
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

# Attempt 1: deliberately incomplete (manifest only)
R1="$HOME/dsmc-retrieval-$STAMP-attempt1"
mkdir -p "$R1/archive"
scp "$LOGIN:$REMOTE/archive-manifest.json" "$R1/archive/"
python tools/dsmc.py archive-verify "$R1/archive" --receipt "$R1/retrieval.json"
echo "exit $?"    # expected: 2, an error message, and no retrieval.json

# Attempt 2: the retry, into a NEW directory
R2="$HOME/dsmc-retrieval-$STAMP-attempt2"
mkdir "$R2"
scp -r "$LOGIN:$REMOTE" "$R2/"
python tools/dsmc.py archive-verify "$R2/$(basename "$REMOTE")" --receipt "$R2/retrieval.json"
echo "exit $?"    # expected: 0 and a complete retrieval receipt
```

Keep attempt 1's directory. Never overwrite a receipt or add files inside a
verified archive.

---

## 9. Change one input: molecule source temperature 1000 K to 1001 K

This reuses the helium field from step 6 (no new helium run) and changes one
value. Back on Engaging:

```bash
cat > "$RUN/source-trials.json" <<'JSON'
{"schema_version": "dsmc-molecule-batch-v1",
 "trials": [{"name": "source-1001K", "overrides": {"source": {"temperature_K": 1001.0}}}]}
JSON
export DSMC_W="$W" DSMC_RUN="$RUN" DSMC_ARCHIVE="$ARCHIVE" DSMC_TRIALS="$RUN/source-trials.json" DSMC_OUT="$RUN/source-trial"
cat > "$RUN/trial.sbatch" <<'BASH'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -Eeuo pipefail
source "$DSMC_W/config/setup.sh"
export JULIA_NUM_THREADS=1
"$DSMC_W/python-venv/bin/python" "$DSMC_W/repo/tools/dsmc.py" molecule-batch \
  "$DSMC_ARCHIVE/workflow/field" --trials "$DSMC_TRIALS" --out "$DSMC_OUT"
BASH
sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --output="$RUN/source-%j.out" "$RUN/trial.sbatch" | tee "$RUN/source-job.txt"
```

Check with `sacct -j $(cat "$RUN/source-job.txt") --format=JobID,State,ExitCode,Elapsed`,
then read `$RUN/source-trial/molecule-batch.json` and the child `report.json`.
The base temperature, 1000 K, is `molecules.source.temperature_K` in
`cases/examples/b5-circle/case.json`. This is a software exercise, not a
temperature-sensitivity result: both runs share one helium realization.

---

## 10. Deliberately invalid input

Ask for a point source with a nonzero radius. The tool must refuse it before
tracing any molecule:

```bash
cat > "$RUN/invalid-trials.json" <<'JSON'
{"schema_version": "dsmc-molecule-batch-v1",
 "trials": [{"name": "bad-point", "overrides": {"source": {"distribution": "point", "radius_m": 0.0127}}}]}
JSON
export DSMC_TRIALS="$RUN/invalid-trials.json" DSMC_OUT="$RUN/invalid-trial"
sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --output="$RUN/invalid-%j.out" "$RUN/trial.sbatch" | tee "$RUN/invalid-job.txt"
```

(`sbatch` passes your current environment to the job, so the job reads the
new `DSMC_TRIALS` and `DSMC_OUT`.)

Expected: the job fails with exit code 2, `$RUN/invalid-%j.out` contains a
message saying that a point source needs zero radius. No molecule is
traced. Copy the exact message into your report.

---

## 11. Record and plot example trajectories

This traces the case's molecules again on the helium field from step 6, and
also records every leg of the first five molecules in `paths.csv`. On
Engaging:

```bash
export DSMC_OUT="$RUN/trajectories"
cat > "$RUN/paths.sbatch" <<'BASH'
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -Eeuo pipefail
source "$DSMC_W/config/setup.sh"
export JULIA_NUM_THREADS=1
py="$DSMC_W/python-venv/bin/python"; repo="$DSMC_W/repo"
"$py" "$repo/tools/dsmc.py" molecule "$DSMC_ARCHIVE/workflow/field" --out "$DSMC_OUT" --trace-count 5
"$py" "$repo/tools/plot_tracer_paths3d.py" --prepared "$DSMC_OUT" "$DSMC_RUN/paths.png"
BASH
sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" \
  --output="$RUN/paths-%j.out" "$RUN/paths.sbatch" | tee "$RUN/paths-job.txt"
```

`DSMC_W`, `DSMC_RUN` and `DSMC_ARCHIVE` are still exported from step 9;
after a reconnect, export them again as there. The two
commands inside the job are the general form:
`python tools/dsmc.py molecule FIELD --out DIR --trace-count K`, then
`python tools/plot_tracer_paths3d.py --prepared DIR out.png`.

When the job has finished:

```bash
sacct -j $(cat "$RUN/paths-job.txt") --format=JobID,State,ExitCode,Elapsed   # expect COMPLETED 0:0
grep recorded_paths "$DSMC_OUT/transport-result.toml"                          # expect recorded_paths = 5
cmp "$DSMC_OUT/events.csv"   "$ARCHIVE/workflow/molecules/events.csv"   && echo events identical
cmp "$DSMC_OUT/terminal.csv" "$ARCHIVE/workflow/molecules/terminal.csv" && echo terminal identical
```

Recording draws no random numbers, and this run uses the same field, count
and seed as the workflow's molecule stage, so both `cmp` lines are expected
to print `identical`. If either reports a difference, that is a finding:
keep both directories and report it.

`paths.csv` holds one row per leg (start and end point, velocity, collision
count, time), in solver coordinates. Each molecule's last leg ends where it
hit a wall or crossed out of the domain, matching its row in `terminal.csv`;
the plotter refuses a file where that does not hold. Copy `$RUN/paths.png` to your
desktop with `scp` to look at it. It shows the recorded paths in two side
projections; nothing is interpolated.

---

## 12. Report what happened

Copy this into a file (for example `$W/outcome.md`) and send it back. Fill
one row per step, including the steps that went smoothly.

Name: ______  Date: ______  Release used (`git -C "$REPO" rev-parse HEAD`): ______

| Step | Expected | Observed (paste key output or error) | Wall time incl. queue | Stuck? Where, and what unblocked you |
|---|---|---|---|---|
| 0 Prerequisites | ssh + Duo work; account/partition known | | | |
| 1 Clone | repo at the named release | | | |
| 2 Environment | "environment recorded"; `sha256sum -c` all OK | | | |
| 3 SPARTA | `COMPLETED 0:0`; `acceptance.txt` first line `acceptance PASS` (paste the whole file) | | | |
| 4 Profiles | three paths printed | | | |
| 5 Prepare | `COMPLETED 0:0`; verify `status: complete` | | | |
| 6 Submit/status | `submitted`, then `complete` | | | |
| 7 Report | `launched_particles` = ___, `forward_unique_particles` = ___ | | | |
| 8 Retrieve + retry | attempt 1 fails; attempt 2 receipt complete | | | |
| 9 Change input | trial `COMPLETED`; report present | | | |
| 10 Invalid input | exit 2, point-radius message | | | |
| 11 Trajectories | `COMPLETED 0:0`; `recorded_paths = 5`; both `cmp` lines `identical`; `paths.png` present | | | |

Anything you had to guess, or any command you had to change: ______

First step that failed: ______
