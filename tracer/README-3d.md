# 3D molecule tracer

`ParticleTracing3D.jl` and `accumulators/crossing3d.jl` trace heavy molecules
through a frozen 3D helium field. Each molecule flies in a straight line
between hard-sphere collisions with helium sampled from the local cell, and
keeps its own clock; there is no shared timestep. Terminated molecules are
removed from the work arrays and their slots refilled. Expensive compatible
work is grouped, irregular geometry handling stays scalar, following the
selective-operation approach of the
[GeantV implementation paper](https://link.springer.com/article/10.1007/s41781-020-00048-6).
It is CPU code: GPU execution and MPI-distributed molecule transport are not
implemented.

## Trace molecules through a prepared B5 field

The usual route reads a completed `dsmc.py field` child (or the `field/`
directory of a workflow or archive) and uses its checked triangle roles, gas
membership, declared source, domain and observation planes. From the
repository root, with a new output directory:

```sh
python tools/dsmc.py molecule FIELD --out DIR
```

`DIR` receives `transport.toml`, `transport-result.toml`, `events.csv`,
`terminal.csv`, `report.json` and `molecule-run.json`. The receipt records the
parent field identity, the launched count and output hashes. `--count N`
overrides the case-declared count for a short run. The example case declares
no acceptance cuts, so `report.json` records crossing statistics but no
accepted yield. This route supports the B5 circle adapter shipped here.

### Record example trajectories

```sh
python tools/dsmc.py molecule FIELD --out DIR --trace-count K
```

`--trace-count K` (default 0, off; at most the launched count) also writes
`paths.csv` with every leg of molecules 1..K. `transport-result.toml` records
`recorded_paths = K` and the receipt lists `paths.csv` with its hash. Columns:
`particle_id, leg, x0_m, y0_m, z0_m, x1_m, y1_m, z1_m, vx_m_s, vy_m_s, vz_m_s,
collisions, time_s, initial_collisions`, in solver coordinates like
`terminal.csv`. The recording includes the terminating leg: the last leg of
each molecule ends at the clipped contact and carries the terminal row's
velocity, collision count and time. Recording draws no random numbers; on the
tracked tube fixture, `events.csv` and `terminal.csv` are byte-identical with
recording on or off. The stage rejects a `paths.csv` with a missing, reordered
or discontinuous leg, or whose last leg differs from `terminal.csv`
(`dsmc_molecule.read_recorded_paths`).

Plot the recorded paths (the plotter applies the same check):

```sh
python tools/plot_tracer_paths3d.py --prepared DIR paths.png
```

`molecule-batch` and `dsmc.py run` do not accept `--trace-count`.

### Several trials on one field

```sh
python tools/dsmc.py molecule-batch FIELD --trials TRIALS.json --out NEW
```

The trial file (`dsmc-molecule-batch-v1`) is described in
[`docs/3d-case-format.md`](../docs/3d-case-format.md). All trials share one
helium realization, so they probe molecule-side variability only.

## Direct command line

For an existing field directory with `cell.surfs` and `DS3FF.DAT`, and a new
output directory `RUN`:

```sh
mkdir RUN
julia --project=tracer --threads=4 tracer/ParticleTracing3D.jl \
  FIELD/cell.surfs FIELD/DS3FF.DAT -n 1000 --seed 42 \
  -M 191 -m 4 --sigma 1.3e-18 -T 500 -z 0.035 --saveall 1 \
  --stats RUN/cartesian.csv --spawnout RUN/spawns.csv \
  --receipt RUN/receipt.toml > RUN/legs.out
```

The species and source values are an example, not defaults for any
apparatus. To record crossings of a plane at **tracer z** = 0.070 m:

```sh
julia --project=tracer --threads=4 tracer/accumulators/crossing3d.jl \
  NEW_RUN_DIRECTORY 0.070 FIELD/cell.surfs FIELD/DS3FF.DAT \
  -n 1000 --seed 42 -T 500 -z 0.035
```

The crossing driver creates its directory and refuses an existing one. It
writes `legs.out`, `recs.csv`, `initial-collisions.csv` and `receipt.toml`;
`--trajprint N` histories go to `trajectories.out`. Stdout keeps the
12-column format, terminal records carry 17 significant digits, and output is
in particle-ID order. `crossing3d.jl` records paths against the inscribed
cylinder boundary described below; it is a different route from
`--trace-count` and cannot replace it.

Named output files must be distinct and new, and no output may be another
output's parent directory. These checks run before anything is written.
`--receipt FILE` overrides the crossing driver's default
`OUTDIR/receipt.toml`; keep a custom receipt beside the outputs (or in their
immediate parent) so the analyzer can validate it.

## Reproducibility

Each particle ID gets its own SplitMix64-derived Xoshiro stream
(`particle-v1`), so group size and thread count do not change a particle's
trajectory. The receipt records the seed, Julia version, input hashes,
arguments, escape bounds, diagnostics and completion status. `--seed 0`
chooses and records a nonzero random seed; the recorded value (a string, to
keep the full unsigned 64-bit range in TOML) can be passed back to `--seed`.
Exact replay also needs the same Julia version and numerical environment.

## Physical model and options

- `--flight local` (default): the gas at the start of a flight sets its
  collision rate and partner. `--flight field-aware` integrates optical depth
  in midpoint steps of `--flight-step METRES` (default 0.0001) and samples gas
  at the collision point. These are different numerical models; convergence
  in the step size is a check you have to run.
- `--rate approximate` (default) uses a mean-relative-speed approximation;
  `--rate maxwell` uses the analytic drifting-Maxwellian mean relative speed.
  Either combines with either flight mode, and the receipt records
  `collision_rate_model`. Hold the rate choice fixed when comparing step sizes.
- Both use straight, force-free flight, a constant elastic cross section and
  a finite 8-sigma envelope for collision-partner proposals. Nonzero
  `--omega`/`--pflip` and the legacy table sampler are rejected. The cell's
  sampled temperature is used; there is no global temperature.
- The helium is frozen: molecules do not alter it.
- `--missing-cell error` (default) rejects missing gas data;
  `--missing-cell nearest` substitutes the nearest cell and records every
  miss. It is a model choice, not a speed option. Malformed rows, nonfinite or
  negative gas values, all-zero density, off-lattice cells, duplicate leaves
  and overlapping ancestor leaves fail before transport. A field with several
  distinct gas subcells in one Cartesian parent needs a solver split map,
  which this release does not include.

`run3d(args)` captures each run's masses, cross section and source, so
reusing a loaded module for a new species or source uses that run's values.
`PreparedCase3D.jl` reuses validated gas and geometry across sequential
trials with run-local counters; `molecule-batch` is built on it.

## Resolved statistics

`--stats FILE` writes sparse native-cell Cartesian statistics.
`--exitstats FILE` writes the same quantities for the whole histories of
molecules that later exit the domain; it is not the velocity distribution at
an observation plane. Omitting either skips its accumulator. No 3D statistics
map averages over azimuth.

Coordinates and velocities are tracer `(x,y,z) = SPARTA (y,z,x)`. Each row
names the native cell and its centre and half-widths in metres. `n_events`
counts completed non-terminal legs. Velocity and position are taken at leg
start; age and collision ordinal at leg end. Variances and covariances are
central population moments, not raw second moments and not dwell-time
weighted. The terminating leg is excluded. A near-stationary initial molecule
gets one forced gas collision, counted separately.

## Analysis and geometry limits

`tools/tracer_analyze3d.py` uses signed x/y velocities and projected
direction angles. It needs an explicit `-N`, a complete spawn record, or a
complete colocated receipt; it does not infer the launched population from
surviving IDs. The `cartesian-3d-v2` schema uses circular means of projected
angles and measures widths after centring on that direction. A numerically
zero circular resultant has no defined direction or width and is reported as
NaN. A broad or multimodal sample need not have a useful single FWHM.

On the direct command line, walls are triangles but the escape boundary is an
inscribed cylinder, wall detection takes precedence over domain
classification, and source clipping tests visibility from a reference point.
This is not general volume membership for arbitrary concave geometry.
Terminal endpoints on that route are not clipped, and the terminal row's age
excludes the final leg. `--vr` supplies a fixed tracer-x drift for an
extended source. The prepared-case route (`dsmc.py molecule`) uses the
explicit triangle geometry, gas membership and planes of the checked case
instead.

## Checks

- `python tools/verify_3d_fixture.py --out NEW_DIR` runs the Julia tracer on
  the synthetic tube field in `tests/fixtures/3d/tube/` and checks hashes and
  four plane passages.
- `python -m pytest tests tools/test_plot_tracer_paths3d.py` runs the Python
  tests (`python -m pip install pytest` first; it is not a runtime requirement). They run neither Julia nor SPARTA; the fixture check above is the
  Julia check.
