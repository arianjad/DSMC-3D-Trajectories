# What SPARTA says about convergence

SPARTA supplies numerical-resolution diagnostics and useful starting rules.
Passing those rules is not a convergence certificate for a particular helium
field or for a molecule observable computed from it. This note separates the
official guidance from the tests we infer from it. None of these tests is
performed by the example case in this repository, which is a short software
execution test.

## Official guidance

The SPARTA-hosted [Gallis DSMC numerical-error presentation](https://sparta.github.io/pdf/sparta_2023_improve_comp_efficiency.pdf)
(slides 4-10) separates four error controls: statistical sample size,
simulation particles per cell, cell size and timestep. It describes the
historical starting rules of roughly 20 simulation particles per cell, cells
narrower than about one third of the local mean free path, and timesteps below
about one quarter of the mean collision time. Its example shows that the
errors interact; asymptotic scaling is not automatically valid for a finite
run. "Particles" here means helium simulation particles, not traced molecules.

[`compute dt/grid`](https://sparta.github.io/doc/compute_dt_grid.html) forms
candidate local timesteps from collision and cell-transit limits and selects
the smallest. The manual suggests a collision fraction of 0.2 as a starting
point, and the same for the transit fraction. This is a safety margin, not a
measured error bound. Empty cells, split parent cells and zero-valued inputs
can produce zero candidates, so a raw minimum must be read with those cells
distinguished.

[`compute lambda/grid`](https://sparta.github.io/doc/compute_lambda_grid.html)
reports the local mean free path, mean collision time and mean-free-path to
cell-size ratios, and recommends time-averaged density and temperature as
inputs. For a streaming gas the temperature should come from `thermal/grid`,
which removes each cell's centre-of-mass motion; directed flow energy is not
thermal energy.

[`collide_modify`](https://sparta.github.io/doc/collide_modify.html) explains
that subcell collision-partner selection can sometimes permit cells as large
as the local mean free path, with reduced benefit in cut cells. That depends
on the collision algorithm and geometry and does not validate an arbitrary
coarse mesh.

[`fix ave/grid`](https://sparta.github.io/doc/fix_ave_grid.html) offers
separate sampling windows (`ave one`) and cumulative averaging
(`ave running`). Its averages combine particle tallies with their
denominators, so a plain average of displayed temperatures need not equal the
pooled estimate. Non-overlapping windows are not automatically statistically
independent (our reading, not a SPARTA guarantee).

[`stats_style`](https://sparta.github.io/doc/stats_style.html) defines `ncoll`
as accepted particle-particle collisions and `nattempt` as candidates; their
`ave` variants are cumulative since the current run began. Raising the
collision majorant raises attempts without raising accepted collisions
(`collide_modify`), so the acceptance ratio measures sampling efficiency and
has no universal steady-state target.

## Tests this implies

A practical test inferred from those controls, not a SPARTA-prescribed
tolerance:

1. **Map local resolution.** Cell width over mean free path, timestep over
   collision time, transit-time ratios and simulation particles per cell, in
   the active gas cells, especially the inlet, aperture and near-wall regions.
   A domain average can hide the cells that matter most for extraction.
2. **Test stationarity and sampling.** Compare separated time windows of local
   density, velocity and thermal temperature, together with inlet and outlet
   flux and the inventory balance. A flat total particle count alone does not
   establish a stationary spatial field. Estimate uncertainty from independent
   realizations or from blocks long enough to cover the temporal correlation.
3. **Test resolution.** Change timestep, particle weight and grid separately
   (timestep/2 and weight/2 are natural first steps), keeping physical run
   time, averaging duration, geometry and boundary conditions comparable. Grid
   refinement must keep adequate particle occupancy. One fine/coarse pair is
   sensitivity evidence; resolve a trend before claiming an order or an error
   bar.
4. **Test the intended output.** Declare which observable, region and error
   would change the conclusion, and run a fixed molecule ensemble through
   representative field variations. SPARTA's helium criteria say nothing about
   the molecule tracer's frozen-field, source, wall or collision-model
   accuracy.

For an observable Q, a useful requirement is that the resolved refinement
shifts and the statistical uncertainty are both small compared with the
declared error budget. Agreement within a very large statistical error is not
evidence of convergence. All of this concerns numerical error within the
chosen physical model; agreement with an apparatus is a separate question.

## Observables used elsewhere

Burt and Boyd's [2011 conference paper](https://doi.org/10.1063/1.3562653)
reviews inventory, collision-rate, energy and boundary balances as
steady-state criteria. Its Q test compares successive-window counts on
individual boundary faces, normalized by their statistical scatter, which
catches dilute-region transients that a global plateau misses. It assumes
suitable window lengths, roughly independent Poisson counts and boundaries
that feel the slowest transient.
[PICLas](https://piclas.readthedocs.io/en/stable/userguide/visu_output.html#particle-flow-and-surface-sampling)
samples density, bulk velocity, temperature, impacts, force and heat flux.
[Takahashi et al., Sec. II](https://doi.org/10.1103/PhysRevResearch.3.023018)
uses the helium density, velocity and temperature inside and beyond the
aperture for separate molecule tracing. These support the choice of
observables, not a universal tolerance.

Useful helium observables for a buffer-gas cell:

| Observable | Purpose |
|---|---|
| Regional density, axial and transverse mean velocity, thermal temperature | Direct inputs to molecule tracing; compare aperture and downstream profiles separately |
| Forward, reverse and net helium flux | Throughput, backflow and plume redistribution; keep spatial bins as well as totals |
| Real-atom inventory and inflow minus outflow minus dN/dt | Conservation and filling; the raw simulator particle count is affected by weighting |
| Accepted collision rate, attempts, ratio and mean energy | Supporting histories and sampler health, not stationarity tests on their own |
| Directional thermal moments, pressure tensor and heat flux | Tests of the scalar-temperature Maxwellian that the molecule tracer assumes |

[`compute pflux/grid`](https://sparta.github.io/doc/compute_pflux_grid.html)
and [`compute eflux/grid`](https://sparta.github.io/doc/compute_eflux_grid.html)
subtract the bulk streaming motion. A stationary expanding gas can keep an
anisotropic temperature and a nonzero heat flux; stationarity does not
require local equilibrium.

Fix the spatial bins from the geometry before comparing runs: just inside the
aperture, through the opening, and several downstream axial and radial bins.
Check sensitivity to the domain's downstream extent for the region you intend
to track molecules through.
