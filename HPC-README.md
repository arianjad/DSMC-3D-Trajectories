# Command reference for an existing Engaging installation

This is the short version, for after you have completed
[`docs/walkthrough.md`](docs/walkthrough.md) steps 1-5 once. It uses the
same work root `$W`, and your own account and partition throughout. For
first-time setup (environment, SPARTA build, profiles, readiness probe) use
the walkthrough.

The example workflow runs SPARTA helium, freezes one helium frame, traces
1,000 example molecules and archives the result. It is an execution test and
does not establish physical convergence.

## 1. Load your installation

```bash
ssh YOUR_KERBEROS@orcd-login.mit.edu
```

On Engaging:

```bash
export W=$HOME/dsmc-work            # your work root from the walkthrough
export ACCOUNT=YOUR_ACCOUNT PARTITION=YOUR_PARTITION
source "$W/config/setup.sh"
export PY="$W/python-venv/bin/python" REPO="$W/repo"
export PREPARED="$W/checked" PROFILE="$W/readiness/verified-profile.json"
"$PY" "$REPO/tools/engaging/prepare_site.py" verify --profile "$PROFILE"
```

Expected: JSON with `status: complete`. This checks that the source and
runtime still match the readiness receipt; it launches nothing. If it fails,
stop and read the error. Do not edit a receipt to make the check pass. If
you changed the clone, the environment or the solver, redo walkthrough
step 5 into new directories.

## 2. Submit one fresh workflow

```bash
export RUN="$W/runs/circle-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$RUN" || { echo "choose a new RUN"; exit 1; }
mkdir -p "$RUN"
printf 'export W=%q\nexport RUN=%q\n' "$W" "$RUN" > "$RUN/operator.env"
"$PY" "$REPO/tools/dsmc.py" acceptance --suite slurm --submit \
  --prepared "$PREPARED" --profile "$PROFILE" --out "$RUN/acceptance"
```

Expected: `Slurm submitted: submitted`; the command returns immediately.
Each submission runs a **new helium realization**. Submit once; if the reply
is unclear, inspect `$RUN/acceptance/` and `squeue -u $USER` before doing
anything else.

## 3. Status and result

```bash
"$PY" "$REPO/tools/dsmc.py" slurm-status --receipt "$RUN/acceptance/acceptance.json"
```

`queued`/`running` are normal; `complete` means the scheduler finished and
the full archive verified; `science_assessment: unassessed` is expected.
After reconnecting: `source "$RUN/operator.env"`, then repeat section 1.

```bash
export ARCHIVE=$("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["profile"]["results_root"]+"/job-"+r["job_id"])' "$RUN/acceptance/submission.json")
"$PY" -m json.tool "$ARCHIVE/workflow/molecules/report.json"
```

`launched_particles` is the denominator; `forward_unique_particles` counts
molecules that crossed the exit plane forward. The archive layout is in
walkthrough step 7.

## 4. Retrieve and verify

Walkthrough step 8: copy the whole archive to a **new** directory on your
desktop, then

```bash
python tools/dsmc.py archive-verify COPIED_ARCHIVE --receipt NEW_RETRIEVAL_RECEIPT.json
```

Keep the receipt outside the archive. An interrupted copy is retried into a
new directory; never add files to a verified archive.

## 5. Reuse a field with changed molecule inputs

`dsmc.py molecule-batch` runs named trials on one existing field without a
new helium run. Walkthrough step 9 has the trials file and the one-CPU job
script; point `DSMC_ARCHIVE` at the archive and `DSMC_TRIALS`/`DSMC_OUT` at a
new trials file and a new output directory. Trials that share a field
cannot measure helium-run variability.

## 6. A new or edited case

Copy `cases/examples/b5-circle/case.json` **beside** the original (the case
refers to its deck and data files by relative path), edit the copy, and
repeat walkthrough step 5 with `CASE` pointing at it and new `readiness`,
`prepared` and `checked` directories, then section 2 with a new `RUN`.
`dsmc.py prepare` refuses an inconsistent case before anything is launched.
Changing timestep, particle weight, grid, source or gas policy changes the
scientific configuration and needs its own labeled run.

## If something fails

| Symptom | Action |
|---|---|
| SSH/Duo fails | Reauthenticate with ordinary `ssh`. |
| Readiness reports changed source or runtime | Keep the error; use the matching source/runtime or redo walkthrough step 5 into new directories. |
| Output directory already exists | Inspect its receipts; use a new name only for a genuinely new attempt. |
| Slurm pending | Read the reason in `squeue`; wait rather than resubmitting. |
| Failed, timed out, out of memory, or archive mismatch | Keep logs, receipts and scratch. Do not treat partial results as complete. |

Cancel a job with `scancel JOB_NUMBER`.
