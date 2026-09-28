# DSMC-3D-Trajectories

Three-dimensional DSMC (direct simulation Monte Carlo) of the helium flow in a
cryogenic buffer-gas cell, and Monte Carlo trajectories of heavy molecules
through that flow.

The pipeline has three stages:

1. **Helium.** [SPARTA](https://sparta.github.io) computes the 3D helium flow
   in the cell from a case file (`cases/examples/b5-circle/case.json`).
2. **Field.** One saved, time-averaged helium frame is frozen and converted
   into the tracer's field format.
3. **Molecules.** A Julia tracer walks heavy molecules (the example uses BaF
   from a 1000 K source) through the frozen field, colliding with the local
   helium, and records wall contacts and crossings of an exit plane.

The coupling is one-way: molecules see the helium, the helium does not see
the molecules. The method follows Takahashi, Shlivko, Woolls and Hutzler,
*Phys. Rev. Research* **3**, 023018 (2021),
[doi:10.1103/PhysRevResearch.3.023018](https://doi.org/10.1103/PhysRevResearch.3.023018).

## Status

The example (`cases/examples/b5-circle/`) is a software execution test: a
short helium run (4,800 steps of 100 ns), two saved frames and 1,000
molecules. A completed run shows that the pipeline works. It does **not**
establish a converged helium state, an extraction efficiency, or a ranking of
geometries.

## Start here

- [`docs/walkthrough.md`](docs/walkthrough.md): from a new MIT Engaging
  account to a verified run, including building SPARTA from source. Start
  here if this is your first time.
- [`HPC-README.md`](HPC-README.md): the short command reference once your
  installation from the walkthrough exists.
- [`3D-QUICKSTART.md`](3D-QUICKSTART.md): the same workflow on one Linux or
  Windows/WSL machine.
- [`docs/3d-case-format.md`](docs/3d-case-format.md): the case file and each
  stage command.
- [`tracer/README-3d.md`](tracer/README-3d.md): the molecule tracer, its
  options, trajectory recording and model limits.
- [`docs/sparta-convergence.md`](docs/sparta-convergence.md): SPARTA's
  resolution guidance and what a convergence test needs.

## Requirements

- Linux cluster with Slurm and MPI (the scripts target MIT Engaging; the
  module line is in `tools/engaging/smoke-setup.sh`).
- Python 3.11 or later with the packages in `requirements-3d.txt`.
- Julia with the project in `tracer/` (`Project.toml`, `Manifest.toml`).
- SPARTA built from source at commit
  `95b9abaa8bd548991cc3c3f1c58b34722f7ade74` (tag `27Aug2026`) with 64-bit
  cell IDs. The walkthrough builds it for you; this repository contains no
  SPARTA source or binary.

`tools/engaging/setup-env.sh` creates the Python and Julia environments.

## Layout

| Path | Contents |
|---|---|
| `tools/dsmc.py` | Command-line entry point: prepare, inspect, run, acceptance, Slurm submit/status, archive, archive-verify, molecule-batch |
| `tools/` | The Python implementation behind `dsmc.py`, field conversion and verification tools |
| `tools/engaging/` | Environment setup, profile writer, SPARTA build-and-acceptance job, site readiness probe |
| `tracer/` | The Julia molecule tracer |
| `cases/examples/b5-circle/` | The example case |
| `cases/b5-lean/`, `cases/b5-slit-pair/` | Geometry generators and the helium input deck the example uses |
| `cases/data/` | Helium species and collision parameters |
| `cases/microtests/` | Small SPARTA inputs used to accept a SPARTA build |
| `tests/` | Python tests and a small synthetic tracer fixture |
| `profiles/` | Example machine profiles (templates; `/CHANGE_ME/` paths) |

`RELEASE-PROVENANCE.json` records the source commit this release was made
from and the SHA-256 of every file.

## License

MIT; see [`LICENSE`](LICENSE) and [`LICENSING.md`](LICENSING.md). The 3D
tracer is forked from Cal Miller's
[DSMC_Simulations](https://github.com/cal-miller-harvard/DSMC_Simulations).
