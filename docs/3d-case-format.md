# 3D case format and stage commands

A case file (`dsmc-case-v1`) declares one geometry, helium model, numerics,
molecule ensemble and observables. The tools resolve it into immutable
directories, one per stage, each with a receipt that records hashes of its
inputs and outputs. Every command refuses an existing output directory.

The example is `cases/examples/b5-circle/case.json`: a 38.1 mm x 63.5 mm
cylindrical cell with a circular aperture, fed through a short inlet stub.
It is a software execution test. A completed run does not establish a
converged helium state, an extraction efficiency or a geometry ranking.

## Stages

From the repository root:

```sh
python tools/dsmc.py prepare cases/examples/b5-circle/case.json --out PREPARED
python tools/dsmc.py inspect PREPARED --json NEW_INSPECTION.json
```

`prepare` writes `prepared.json`, `geometry.json`, the five emitted `.surf`
files, `in.helium`, `he.species` and `he.vss`. `inspect` verifies the staged
hashes and reports triangle counts, opening area, boundary types, station
directions and throat regions. The 624 emitted physical triangles are
**pre-clip**; they are not SPARTA's final triangles.

With a SPARTA executable and a local profile, a solver check produces a new
checked preparation:

```sh
python tools/dsmc.py inspect PREPARED --solver-check --profile LOCAL_PROFILE.json --out CHECKED
```

It runs **zero transport steps** twice: first it captures SPARTA's clipped
surface as `physical-walls.surf`, then it rereads that frozen mesh.
`triangle-roles.json` binds each final triangle ID to wall or inlet-cap
behaviour and to the surface hash. The original preparation is unchanged. On
the pinned 64-bit-cell-ID SPARTA build the circle example has 676 triangles
after clipping, 48 of them inlet-cap facets. This is a geometry check, not a
populated field.

A full local run starts **new** helium, then prepares the field and traces the
declared molecules:

```sh
python tools/dsmc.py run CHECKED --profile LOCAL_PROFILE.json \
  --gas-policy carry-T-and-empty-ballistic --out WORKFLOW
```

`workflow.json` names the helium, field and molecule child receipts and keeps
the failed stage if one stops. The run never substitutes an earlier helium
run. With SPARTA installed in WSL and Python/Julia on Windows, add
`--helium-backend wsl` and use a profile whose solver path and Python are
valid inside WSL.

The stages can also be run one at a time from an existing helium run
directory:

```sh
python tools/dsmc.py frame HELIUM_RUN --timestep 4800 --json NEW_SELECTION.json
python tools/dsmc.py field CHECKED --parent HELIUM_RUN --timestep 4800 \
  --gas-policy carry-T-and-empty-ballistic --out FIELD
python tools/dsmc.py molecule FIELD --out MOLECULES
```

`frame` (or `--last-complete`) checks the whole gzip stream, header and row
count, and any recorded output hash, and reports newer invalid frames it
skipped; a complete frame from a failed run remains usable, with that status
recorded. `field` refuses a mismatched parent and preparation, an
off-schedule frame or a changed box; it copies and revalidates the frame,
converts it with the canonical surfaces and declared base grid, and writes
`field-preparation.json`. The gas policy is required: the one shown carries
sampled temperature into positive-density cells that lack a local temperature
and treats empty cells as ballistic. It is a declared reconstruction choice,
not a wall law. `molecule` writes `events.csv`, clipped `terminal.csv`,
`report.json` and `molecule-run.json`; see
[`tracer/README-3d.md`](../tracer/README-3d.md) for its options, including
trajectory recording.

`prepared.json` carries the frozen science inputs and the field receipt binds
its hash, so the original case file may be edited or moved after field
preparation.

## Trials on one field

```json
{
  "schema_version": "dsmc-molecule-batch-v1",
  "trials": [
    {"name": "baseline", "overrides": {"count": 100}},
    {"name": "source-probe", "overrides": {
      "count": 100,
      "seed": "20260923",
      "source": {"temperature_K": 500.0}
    }}
  ]
}
```

```sh
python tools/dsmc.py molecule-batch FIELD --trials TRIALS.json --out NEW
```

Each trial inherits the case's molecule settings and may override only
`count`, `seed`, `mass_u`, `gas_mass_u`, `cross_section_m2`, `flight`,
`collision_rate_model`, `flight_step_m`, and the `source` fields
`distribution`, `center_m`, `radius_m`, `drift_m_s`, `temperature_K`. The
batch checks the field before writing, snapshots the trial JSON and its hash,
and writes a receipt and report per trial. It starts no helium, so the trials
share one helium realization and cannot estimate helium-run variability.

## Case file

UTF-8 JSON with `schema_version: dsmc-case-v1`, `case_id`, `model`,
`geometry`, `helium`, `numerics`, `molecules` and `observables`, plus an
optional `assessment`. Unknown keys fail. Paths in a case resolve relative to
the case file; staged paths in `prepared.json` are relative to the prepared
directory.

- Units are in the field names and are SI, except molecule and gas masses in
  atomic mass units (`mass_u`, `gas_mass_u`).
- The molecule `seed` is a nonzero decimal UInt64 string; the SPARTA
  `solver_seed` is a positive integer.
- Point sources require `radius_m: 0`; uniform and Gaussian ball sources
  require a positive radius. Source temperature may be zero for a
  deterministic diagnostic.
- `flight`, `collision_rate_model`, `flight_step_m`, gas reconstruction,
  source distribution, wall action and RNG protocol are independent, explicit
  choices.
- Solver coordinates map to tracer coordinates as
  `(x,y,z)_solver -> (y,z,x)_tracer`.
- The example writes helium fields at steps 2,400 and 4,800. Its source
  offset along positive solver `y` is an example choice, not an apparatus
  direction.

The `b5-circle` adapter has a fixed domain, grid and `b5-s25-sampled-v1`
refinement. Boundary labels are explicit: wall (type 1) and emitting inlet
cap (type 2) on the cell, plus transparent flux stations. Other geometries
need their own adapter.

In plane acceptance, `angle_rad` is the polar deflection from the plane's
forward normal, `atan2(hypot(vu,vv),vn)`, on forward crossings. A null bound
means no cut; with all four bounds null the plane is **unassessed**, as in the
example.

## Profiles

`profiles/local.example.json` is a template for a **launch** profile: solver
executable and launcher, ranks and threads, storage roots and resource caps.
Preparation never reads a profile, and a profile cannot change the science
inputs in `prepared.json`. A launch receipt records which profile was used.
`profiles/slurm.engaging.example.json` is the Slurm counterpart; the
walkthrough writes a real one for your account.
