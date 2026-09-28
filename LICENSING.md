# Licensing and notices

Everything in this repository is released under the MIT License; see
[`LICENSE`](LICENSE).

## Attribution

The 3D tracer is forked from Cal Miller's DSMC_Simulations
(https://github.com/cal-miller-harvard/DSMC_Simulations, commit 543b42e).

## SPARTA (not included)

The helium solver is [SPARTA](https://sparta.github.io), developed at Sandia
National Laboratories. This repository contains no SPARTA source,
documentation or executable. The walkthrough has you clone and build SPARTA
yourself at commit `95b9abaa8bd548991cc3c3f1c58b34722f7ade74` (tag
`27Aug2026`); that checkout and your build are governed by SPARTA's own
license, the GNU General Public License version 2
(https://github.com/sparta/sparta/blob/95b9abaa8bd548991cc3c3f1c58b34722f7ade74/LICENSE).
The scripts here run SPARTA as a separate program and exchange files with it.

## Python and Julia packages

The packages named in `requirements-3d.txt`, `tracer/Project.toml` and
`tracer/Manifest.toml` are installed from their upstream distributors under
their own licenses. None is included in this repository.
