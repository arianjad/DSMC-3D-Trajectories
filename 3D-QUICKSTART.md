# 3D workflow on your own machine

This runs the example case on one Linux machine (or Windows with SPARTA in
WSL): a **fresh** helium run, conversion of its final saved frame, and the
case-declared molecule ensemble. For a Slurm cluster, use
[`docs/walkthrough.md`](docs/walkthrough.md) (MIT Engaging, from a new
account) and [`HPC-README.md`](HPC-README.md).

The example (`cases/examples/b5-circle/`) is a software execution test. A
complete run with valid receipts does not establish a converged helium state,
an extraction efficiency or a geometry ranking.

In the commands below, every `NEW_...` or upper-case output name must be a
directory or file that does not exist yet; every command refuses to
overwrite. Keep outputs outside the clone.

## 1. Tools

- Python 3.11 or later with [`requirements-3d.txt`](requirements-3d.txt).
- Julia on `PATH`, with the project in `tracer/` instantiated
  (`julia --project=tracer -e 'using Pkg; Pkg.instantiate()'`).
- SPARTA with 64-bit cell IDs, built from upstream commit
  `95b9abaa8bd548991cc3c3f1c58b34722f7ade74` (tag `27Aug2026`), and MPI.
  `tools/wsl/build_sparta_cpu.sh --variant bigbig` is a pinned build recipe
  (`--print-plan` shows what it will do); walkthrough step 3 builds and
  accepts the same executable on a cluster.

Check Julia transport on the tracked synthetic field first (no SPARTA
needed):

```sh
python tools/verify_3d_fixture.py --out NEW_FIXTURE_CHECK
```

It checks field and geometry hashes and should record four clean plane
passages. It says nothing about helium.

Copy [`profiles/local.example.json`](profiles/local.example.json) outside
the repository and set `solver_executable`, `mpi_argv`, `ranks`, `threads`,
the storage roots and the memory and disk caps for your machine. On Windows
with SPARTA in WSL, the solver path and `python` in the profile are **WSL**
paths; Python and Julia themselves run on Windows.

## 2. Prepare and check the case

```sh
python tools/dsmc.py prepare cases/examples/b5-circle/case.json --out PREPARED
python tools/dsmc.py inspect PREPARED --solver-check --profile MY_PROFILE.json --out CHECKED
```

The solver check is a zero-step SPARTA clip-and-reread of the geometry;
`CHECKED` is the input to a full run. [`docs/3d-case-format.md`](docs/3d-case-format.md)
describes the case file and each stage.

## 3. Run, verify, read

```sh
python tools/dsmc.py run CHECKED --profile MY_PROFILE.json \
  --gas-policy carry-T-and-empty-ballistic --out WORKFLOW
python tools/verify_3d_workflow.py WORKFLOW --out NEW_VERIFICATION.json
```

Add `--helium-backend wsl` to `run` when SPARTA runs in WSL. The gas policy
carries sampled temperature into positive-density cells that lack one and
treats empty cells as ballistic; it is a declared model choice.

`WORKFLOW/workflow.json` should say `status: complete` and name three child
receipts: `helium/manifest.json`, `field/field-preparation.json` and
`molecules/molecule-run.json`. The verifier independently rechecks receipt
hashes, every recorded output, the whole selected helium frame, the station
logs and the molecule denominator. A failed stage keeps its receipts.

In `WORKFLOW/molecules/report.json`, `launched_particles` is the denominator
and `forward_unique_particles` counts molecules that crossed the exit plane
forward at least once; rows of `events.csv` are crossings, not molecules. The
example declares no acceptance cuts, so `science_assessment` is
`unassessed`.

## 4. More molecules on the same field

Re-trace molecules on the saved field without new helium, here also
recording the first five trajectories:

```sh
python tools/dsmc.py molecule WORKFLOW/field --out NEW_MOLECULES --trace-count 5
python tools/plot_tracer_paths3d.py --prepared NEW_MOLECULES paths.png
```

For named source or species trials, use `dsmc.py molecule-batch`
([format](docs/3d-case-format.md#trials-on-one-field)). Runs on one saved
field share one helium realization and cannot estimate helium-run
variability. [`tracer/README-3d.md`](tracer/README-3d.md) describes the
tracer's options and model.

A deliberately invalid input: copy the case file **beside** the original,
change `molecules.source.distribution` to `point` and keep its positive
`radius_m`. `dsmc.py prepare` should refuse it with exit code 2 and create no
output directory. Setting `radius_m` to 0 repairs it.

## 5. Cost and one-command acceptance

```sh
python tools/benchmark_3d.py CHECKED --profile MY_PROFILE.json \
  --gas-policy carry-T-and-empty-ballistic --out NEW_BENCHMARK
python tools/dsmc.py acceptance --suite local --profile MY_PROFILE.json --out NEW_ACCEPTANCE
```

The benchmark launches another fresh full run and records wall time, stage
timers, output bytes and sampled memory. Under WSL the Windows process tree
does not include the guest's MPI and Python memory. `acceptance --suite
local` runs the fixture, prepares and solver-checks the example, runs and
times a full workflow and verifies it; `acceptance.json` records each stage.
`--helium-backend auto` (the default there) selects WSL on Windows when the
profile names a POSIX solver path.

## 6. Share or move results

An analysis bundle (selected frame, molecule outputs, license notices; no
profile, solver, restart files or path-bound receipts):

```sh
python tools/dsmc.py export WORKFLOW --out NEW_BUNDLE
python tools/verify_3d_export.py COPIED_BUNDLE
```

`export` verifies the workflow first. The second command, run on the
receiving machine (Python 3.10 or later), checks every listed hash, the
selected raw frame, the case identity and the molecule counts. It does not
rerun anything.

A full archive (everything, including restart files), and its check after
copying:

```sh
python tools/dsmc.py archive WORKFLOW --out NEW_ARCHIVE
python tools/dsmc.py archive-verify COPIED_ARCHIVE --receipt NEW_RETRIEVAL.json
```

Keep the retrieval receipt outside the archive. `acceptance --suite handoff`
runs a new workflow, exports it, copies the bundle to a second directory and
verifies both; it is a same-machine check.

## 7. Continue a run from its final checkpoint

From a full archive, with a new seed and an absolute target step beyond the
checkpoint:

```sh
python tools/dsmc.py resume ARCHIVE --checkpoint ARCHIVE/workflow/helium/final.restart.base \
  --seed NEW_SEED --target-step 7200 --out NEW_RESUME
python tools/dsmc.py resume-run NEW_RESUME --parent-archive ARCHIVE --profile MY_PROFILE.json
```

Run these where the profile's solver runs. The resume uses the checkpoint's
MPI rank count unless `resume --ranks N` says otherwise, and the profile's
`ranks` must equal that count. `verification.json` is written only after the new restart,
field and diagnostics pass their checks. The resumed run starts a fresh
averaging window and is not a convergence assessment.
