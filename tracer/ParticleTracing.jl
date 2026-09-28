# ParticleTracing.jl -- vendored into DSMC-sims.
#
# Upstream: https://github.com/cal-miller-harvard/DSMC_Simulations
#   path    ParticleTracing/ParticleTracing.jl
#   commit  543b42e ("simulation results")
#   license MIT in this repository (see LICENSING.md)
# Vendored 2026-08-19 for the particle-tracing integration
# (particle-tracing integration plan of 2026-08-19, Phase 1).
#
# Local diffs vs upstream (EXACTLY sixteen).  Diffs 1-11 changed behaviour or
# added flags; diff 12 is a pure structural extraction; diff 13 fixes an
# upstream threading defect; diff 14 opens the accumulator seam; diff 15
# removes the field-lookup cache; diff 16 repairs accumulator controls and the
# interpolation diagnostic.  Still not allowed without a fresh decision:
# renames, formatting passes, type annotations, performance work.  A
# structural change is admissible only if it is byte-identical under
# tracer/abcheck.sh (see diff 12). abcheck.sh, statsrace.sh and the microtests
# named below are validation harnesses in the private development repository;
# they are not part of this release:
#   1. CSV sink API: CSV.read(f, ...) -> CSV.read(f, DataFrame; ...) at every
#      call site (modern CSV.jl requires an explicit sink argument).
#   2. `using Statistics` added (mean/std on Vectors in
#      generate_lookup_table on modern package versions).
#   3. `--seed` flag (Int, default 0 = do not seed): Random.seed!(seed)
#      before the particle loop. Julia 1.9 task-local RNGs fork
#      deterministically from the seeded root, so threaded runs reproduce
#      AT A FIXED THREAD COUNT.
#   4. `--spawn` flag (String, default "point") with modes point, gaussball,
#      uniformball, uniformcell, plus --spawnsize and
#      --spawnrlo/--spawnrhi/--spawnzlo/--spawnzhi. Implemented ONLY inside
#      generateParticle() in main(); velocity sampling is unchanged.
#   5. `--trajprint` flag (Int, default 0): gates the per-leg print block.
#      Semantics generalized by diff 11 below.
#   6. Torn-stdout fix (Arian-approved 2026-08-19): the two per-particle
#      output sites used println(@sprintf(...)), which writes string and
#      newline as two separate locked writes -- at 16 threads ~1% of rows
#      fused/blanked (measured 43% under max contention, 200k-row A/B).
#      Changed to a single print(@sprintf("...\n", ...)) write, measured
#      0/200k tears. Output bytes otherwise identical.
#   7. `--sampler` flag (Arian-approved 2026-08-19; default "exact" per
#      Arian's follow-up decision -- "table" is the legacy upstream
#      sampler, kept only for A/B): "exact" replaces the lookup-table
#      rejection sampler in collide! with exact_partner(), a direct
#      flux-weighted Maxwellian rejection draw. Motivation (measured, session log
#      2026-08-19): the table sampler delivers only 70-85% of the exact
#      hard-sphere momentum transfer at relative speeds 20-200 m/s and
#      zero drag beyond the table's Umax = 1.5*max|v_flow| span; the
#      exact sampler matches a 2e7-sample reference MC within 1% and is
#      1.3-2.4x faster per collision. The exact mode implements the same
#      flux-weighted partner distribution as Takahashi et al. PRR 3,
#      023018 Appendix A (their Eq. A3), which this repo validates
#      against. Scatter kinematics are shared and untouched.
#   8. `--spawnclip` flag (Arian-approved 2026-08-19; default 1 = on):
#      rejects and resamples spawn positions that geometry separates from
#      the spawn region's reference point (SPAWN_REF) -- i.e. particles
#      born INSIDE a wall or outside the domain. Motivation (measured,
#      session log 2026-08-19): the plan's gaussball spawn (sigma 3.3 mm
#      at the Takahashi cell centre, wall at r = 6.35 mm) puts 15.7% of
#      draws in the metal (Rayleigh tail; 3 of 12 sample trajectories,
#      one surviving 24884 collisions inside the solid on nearest-gas-cell
#      field values). Upstream only tests wall CROSSINGS, so a particle
#      born in a wall is never stopped. The clipped ball is the truncated
#      Gaussian the condition physically means. Uses the existing
#      getCollision, adds no geometry code; the rejected count is reported
#      on stderr.
#   9. `uniformcell` now randomizes the azimuth, and the old behaviour is
#      preserved under the new name `uniformplane` (Arian-approved
#      2026-08-20). Upstream spawned every uniformcell particle at
#      [r, 0.0, z] -- azimuth exactly 0. An axisymmetric flow exerts no
#      azimuthal force, so angular momentum only diffuses and even ~9000
#      collisions do not randomize it: measured <v_x> = +15.05 +- 0.09 m/s
#      (175 sigma, azimuth chi2/dof = 1002) on the extracted ensemble,
#      against +0.26 and -0.03 m/s for the isotropic ball spawns. Scalar
#      observables are azimuth-independent by axisymmetry -- extraction,
#      v_f, |v_r|, collisions and transit time are unaffected, and the
#      Phase-4 numbers stand -- but any signed Cartesian statistic
#      inherits the asymmetry (a signed-v_x divergence read 15% low).
#      `uniformplane` reproduces the upstream stream exactly: r and z are
#      drawn in the original order and no further draw is made, so a
#      uniformplane run is bit-identical to the pre-diff-9 uniformcell run
#      at the same seed (verified against the Phase-4 output).
#  10. `--spawnout FILE` (Arian-approved 2026-08-20): writes idx,x,y,z per
#      particle through the same CSV/DataFrame path as `--stats`. The
#      trajectory rows carry only the terminating leg, so a birthplace is
#      otherwise unrecoverable and the paper's Fig.-5 extraction-versus-
#      initial-position map cannot be built. Deliberately a separate sink,
#      NOT extra stdout columns: the stdout stream stays byte-identical
#      (A/B-verified against `git show HEAD:` at n = 2000, seed 42), so
#      every existing parser and all Phase-4 data remain valid. Sampling
#      note: the conditional P(extract | z, r) does not depend on the spawn
#      proposal density, so for the MAP prefer a spawn uniform in the (z,r)
#      rectangle -- volume weighting puts bin counts proportional to r and
#      starves the near-axis bins where the structure is.
#  11. `--trajprint N` REPURPOSED (Arian-approved 2026-08-20): was "every mm
#      past max_x_geom", i.e. per-mm samples in the free plume only, which
#      nothing consumed. Now "every leg of particles 1..N", so full
#      trajectory histories come out of an ordinary run on ANY field,
#      species and spawn mode. This replaced tools/tracer_trajectories.jl, a
#      standalone recorder that re-derived SimulateParticles' setup and
#      propagate's loop verbatim and reimplemented generateParticle for the
#      gaussball mode ONLY -- so it silently ignored --spawn and drifted from
#      the tracer on every diff. Rows keep the 12-column format with a
#      NEGATIVE index, which `tracer_analyze.py` already filtered out, so
#      analysis is unaffected. Side effect: `max_x_geom` now feeds nothing
#      inside propagate; the parameter is left in place to keep vendored
#      churn minimal.
#  12. `build_field(geomFile, gridFile)` extracted from SimulateParticles
#      (Arian-approved 2026-08-20; tracer seam-refactor plan).
#      A PURE MOVE: the field/geometry setup -- bounds, geom, the grid frame,
#      the KDTree, the dmin fill loop, the lookup table, and the interpolate!
#      and getCollision closures -- now lives in its own function returning
#      (; interpolate!, getCollision, table, bounds, max_x_geom), and
#      SimulateParticles destructures it. Every moved line is byte-identical;
#      only four lines are new (the signature, the return, its end, and the
#      call). Motivation: interpolate! knn's on [z, r] and getCollision does
#      segment-vs-segment in the (z,r) plane, so those two closures are the
#      ONLY 2D-axisymmetric-specific code in the file -- the physics core is
#      already 3D Cartesian. Behind this seam a 3D or transient field is a
#      second build_field returning the same named tuple, not a fork.
#      Gate: tracer/abcheck.sh, 16 cells x two passes (threaded for
#      stdout/spawnout, serial for the then-racy stats/exitstats), 32/32 PASS
#      vs the pre-extraction file.
#  13. Threaded statistics reduction fixed (Arian-approved 2026-08-20).
#      Upstream ran merge!(allstats, stats) and merge!(boundstats, stats)
#      INSIDE the threaded particle loop with no synchronization, so every
#      thread read-modify-wrote the same OnlineStats objects. Measured
#      (tracer/statsrace.sh): 0.03-0.31% of leg samples silently LOST at 16
#      threads, the lost fraction itself random, and the binned CSVs not
#      reproducible run to run at fixed seed and thread count. Exactly zero
#      loss at one thread, which rules out every non-threading explanation.
#      Replaced with the standard parallel fold: the particle range is
#      partitioned explicitly, each chunk merges into its OWN partial (so a
#      partial is reachable by exactly one task and needs no lock), and the
#      partials are reduced into the run totals in fixed chunk order after
#      the loop. Fixes the loss AND the non-determinism; a lock would have
#      fixed only the loss, and would have put the thread-safety burden
#      outside the accumulator instead of in the reduction structure -- which
#      matters because this merge site is about to become public API.
#      NOT byte-identical: Threads.@spawn over an owned partition forks
#      task-local RNGs differently from Threads.@threads, so trajectories
#      differ. Validated statistically instead, n=5000 uniformcell, two seeds,
#      before vs after: extraction 0.3440/0.3630 -> 0.3432/0.3628 (4 and 1
#      particles out of 5000, against binomial sigma = 33), median v_z
#      141.02/140.93 unchanged to 2 dp, median |v_r| 32.76/33.67 ->
#      32.78/33.67. Leg conservation now EXACT at 1, 4 and 16 threads, and
#      all four artifacts reproduce run to run. Cost +0.8% wall time over 3
#      interleaved repeats. Reproducibility still requires --seed AND a fixed
#      thread count, as before. Caveat: Iterators.partition with
#      cld(n, nthreads()) leaves the last chunk short, and for n barely above
#      nthreads() it yields fewer chunks than threads -- irrelevant at science
#      n, worth knowing for tiny runs.
#  14. `make_stats` keyword on SimulateParticles (Arian-approved 2026-08-20):
#      the per-leg accumulator is now injectable instead of hard-wired. All
#      five StatsArray construction sites go through new_stats(), which
#      defaults to exactly the previous expression, so the default path is
#      byte-identical (abcheck.sh 16/16 PASS on all four artifacts).
#      THE ACCUMULATOR CONTRACT -- this is now public API:
#        updateStats!(acc, x, v, t, ncolls, lfree)  called once per leg
#        merge!(dst, src)                           fold; once per particle
#                                                   into its chunk partial,
#                                                   once per chunk into totals
#      Diff 13's ordered reduction is what makes this safe: an accumulator is
#      only ever touched by one task during the loop and folded in fixed order
#      afterwards, so it does NOT need to be thread-safe itself. Gated by
#      tracer/microtests/accumulator_seam.jl, which counts hook fires against
#      sum(outputs[:,10]) -- 1946855 == 1946855 legs, 216 == 200 particles +
#      16 chunks. Trajectory recording, arrival-time histograms and
#      transient/ablation density-vs-time are all this same shape.
#  15. Field-lookup cache REMOVED (Arian-approved 2026-09-04). Upstream cached
#      one cell's gas properties over a ball of radius dmin around that cell
#      and refreshed only when the molecule left the ball. Its dmin loop was
#      wrong two ways: the band was [0.2v, 1.2v] where the comment said 20 %,
#      and the `break` left only the inner property loop, so the scan kept the
#      LAST failing neighbour's distance instead of the nearest. With the
#      converter's vz written as exactly 0, every neighbour fails the band and
#      dmin is the distance to the 100th-nearest cell for 100.00 % of cells
#      (1.496 mm mean on the B5 field, against a 2.5 mm aperture), so a
#      molecule saw stale helium over millimetres. Measured with the hot-BaF
#      B5 source, n = 1e5, seed 42, on the mflow-inlet field
#      b5lean-smooth-4sccm: extraction 0.0570 through the shipped cache,
#      0.0523 with the radius forced to its 1.496 mm mean, 0.0671 +- 0.0008
#      with the radius forced to 0, against 0.0673 +- 0.0008 from the
#      independent 3D tracer's exact containing-cell lookup (0.2 sigma). On
#      the production field phase1-B5-cal2 the same source goes 0.0452 ->
#      0.0560 (+24 %; no-cache gate runs of 2026-09-04). The
#      corrected loop degenerates to dmin = 0 anyway (0 < 0 < 0 is false on
#      the first neighbour, the cell itself), so "no cache" IS the fix the
#      code intended: interpolate! now does an exact nearest-centre knn on
#      every call, which is also how Takahashi et al. sample the field.
#      Cost: the run-time nearest-neighbour count rises; column 8 of the grid
#      matrix and props[8] are kept at 0 so the layout that Field3D.jl and
#      propagate share is unchanged. NOT byte-identical by construction, so
#      abcheck.sh against a pre-diff-15 ref FAILS as expected; the gate is
#      tracer/microtests3d/gateA.sh (2D vs 3D on the same field) plus the 2D
#      physics microtests 3A/3B/3C (both green 2026-09-04). Every 2D
#      extraction number recorded before this date was measured through the
#      cache and reads low by a field-dependent 15-24 %.
#  16. Accumulator controls and interpolation diagnostic repaired
#      (Arian-approved 2026-09-22). `false` and `nothing` now both disable a
#      requested accumulator, disabled return values are defined as `nothing`,
#      and nonpositive particle counts fail directly. The 2D StatsArray
#      formulas/schema and the synthetic 11-column output-width probe are
#      unchanged. The 2D interpolation counter is atomic across particle tasks
#      and reset at the start of each SimulateParticles call.
#
# Upstream constraints found while porting (deliberately NOT patched -- they
# are requirements on the field files a converter must emit):
#   - generate_lookup_table builds Tmin:Tstep:Tmax and Umin:Ustep:Umax, so a
#     perfectly isothermal field (Tstep == 0) or a zero-velocity field
#     (Umax == 0) is degenerate and throws ArgumentError("range step cannot
#     be zero") / InexactError.

using Distributed
using Random
using LinearAlgebra
using NearestNeighbors
using CSV
using DataFrames
using Printf
using ArgParse
using OnlineStats
using SpecialFunctions
using Statistics

import Base: convert

# Data structures and functions for tracking trajectory statistics

# Statistics to be recorded for each grid cell. Currently includes velocity, time, number of collisions, and free path
struct TrajStats
    v::OnlineStat
    t::OnlineStat
    ncolls::OnlineStat
    lfree::OnlineStat
end

@inline TrajStats() = TrajStats(CovMatrix(2), Variance(), Variance(), Variance())

# Grid of TrajStats, including spatial information for indexing
struct StatsArray
    stats::Matrix{TrajStats}
    minr::Float64
    maxr::Float64
    rbins::Int
    minz::Float64
    maxz::Float64
    zbins::Int
    rstep::Float64
    zstep::Float64
end


@inline function StatsArray(minr, maxr, rbins, minz, maxz, zbins)
    stats = Array{TrajStats}(undef, rbins, zbins)
    for i in 1:rbins
        for j in 1:zbins
            stats[i,j] = TrajStats()
        end
    end
    return StatsArray(stats, minr, maxr, rbins, minz, maxz, zbins, rbins/(maxr-minr), zbins/(maxz-minz))
end

# Updates a StatsArray given a position and the value of the statistics
@inline function updateStats!(s::StatsArray, x, v, t, ncolls, lfree)
    r = sqrt(x[1]^2+x[2]^2)
    ridx = min(s.rbins, 1+floor(Int, s.rstep*(r-s.minr)))
    zidx = min(s.zbins, 1+floor(Int, s.zstep*(x[3]-s.minz)))
    fit!(s.stats[ridx, zidx].v, [(-x[2]*v[1]+x[1]*v[2])/sqrt(x[1]^2+x[2]^2),v[3]])
    fit!(s.stats[ridx, zidx].t, t)
    fit!(s.stats[ridx, zidx].ncolls, ncolls)
    fit!(s.stats[ridx, zidx].lfree, lfree)
    return s.stats[ridx, zidx]
end

# Merges two StatsArrays by merging each statistic for each cell
# TODO: Improve performance by only merging cells that the particle passed through (keep track of active cells)?
@inline function merge!(a::StatsArray, b::StatsArray)
    for i in 1:a.rbins
        for j in 1:a.zbins
            OnlineStats.merge!(a.stats[i,j].v, b.stats[i,j].v)
            OnlineStats.merge!(a.stats[i,j].t, b.stats[i,j].t)
            OnlineStats.merge!(a.stats[i,j].ncolls, b.stats[i,j].ncolls)
            OnlineStats.merge!(a.stats[i,j].lfree, b.stats[i,j].lfree)
        end
    end
    return a
end

# Converts a StatsArray to a matrix with rows r, z, n, t, tvar, vr, vz, vrcov, vzcov, vrvzcov, ncolls, ncollsvar, lfree, lfreevar
@inline function convert(::Type{Matrix}, s::StatsArray)
    M = Array{Float64}(undef, s.rbins*s.zbins, 14)
    idx = 1
    for i in 1:s.rbins
        for j in 1:s.zbins
            stats = s.stats[i,j]
            M[idx,1] = s.minr+(i-0.5)/s.rstep
            M[idx,2] = s.minz+(j-0.5)/s.zstep
            M[idx,3] = stats.t.n
            M[idx,4] = stats.t.μ
            M[idx,5] = stats.t.σ2
            M[idx,6] = stats.v.b[1]
            M[idx,7] = stats.v.b[2]
            M[idx,8] = stats.v.A[1,1]
            M[idx,9] = stats.v.A[2,2]
            M[idx,10] = stats.v.A[1,2]
            M[idx,11] = stats.ncolls.μ
            M[idx,12] = stats.ncolls.σ2
            M[idx,13] = stats.lfree.μ
            M[idx,14] = stats.lfree.σ2
            idx += 1
        end
    end
    return M
end

# Converts a StatsArray to a DataFrame with rows r, z, n, t, tvar, vr, vz, vrcov, vzcov, vrvzcov, ncolls, ncollsvar, lfree, lfreevar
@inline function convert(::Type{DataFrame}, s::StatsArray)
    m = convert(Matrix, s)
    df = DataFrame(m, [:r, :z, :n, :t, :tvar, :vr, :vz, :vrvar, :vzvar, :vrvzcov, :ncolls, :ncollsvar, :lfree, :lfreevar])
    return df
end

"""
    parse_commandline()

Parses command-line arguments. See function for possible arguments
"""
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "geom"
            help = "the file generated by SPARTA containing a list of surfaces"
            required = true
        "flow"
            help = "the file generated by SPARTA containing a dump of flow properties at each grid point"
            required = true
        "-n"
            help = "the number of particles to simulate."
            arg_type = Int
            default = 10000
        "-z"
            help = "axial position at which to spawn particles (m)"
            arg_type = Float64
            default = 0.035
        "-r"
            help = "radial position at which to spawn particles (m)"
            arg_type = Float64
            default = 0.0
        "--vz"
            help = "mean axial velocity at which to spawn particles (m/s)"
            arg_type = Float64
            default = 0.0
        "--vr"
            help = "mean radial velocity at which to spawn particles (m/s)"
            arg_type = Float64
            default = 0.0
        "-T"
            help = "temperature at which to spawn particles (K)"
            arg_type = Float64
            default = 0.0
        "-m"
            help = "mass of buffer gas atoms (AMU)"
            arg_type = Float64
            default = 4.0
        "-M"
            help = "mass of particles (AMU)"
            arg_type = Float64
            default = 191.0
        "--sigma"
            help = "collision cross section between buffer gas and particle (m^2)"
            arg_type = Float64
            default = 130E-20
        "--omega"
            help = "radial trap frequency (rad/s); sqrt(v/m) for a harmonic trap of V(x,y,z) = v*(x^2+y^2)"
            arg_type = Float64
            default = 0.0
        "--zmin"
            help = "minimum axial position of harmonic trap (m)"
            arg_type = Float64
            default = -Inf
        "--zmax"
            help = "maximum axial position of harmonic trap (m)"
            arg_type = Float64
            default = Inf
        "--pflip"
            help = "probability that collision flips spin"
            arg_type = Float64
            default = 0.0
        "--saveall"
            help = "saves all particles if nonzero or just those that leave the cell if zero"
            arg_type = Int
            default = 0
        "--stats"
            help = "filename to save average properties of trajectories"
            arg_type = String
            default = nothing
        "--exitstats"
            help = "filename to save average properties of trajectories where the particle hits the simulation boundary"
            arg_type = String
            default = nothing
        "--spawnout"
            help = "filename to save every particle's spawn position as idx,x,y,z; join on idx against the stdout rows to get extraction versus birthplace (the Fig.-5 map). Does not change the stdout stream."
            arg_type = String
            default = nothing
        "--seed"
            help = "RNG seed; 0 leaves the RNG unseeded. Threaded runs reproduce only at a fixed thread count."
            arg_type = Int
            default = 0
        "--spawn"
            help = "initial position mode: point, gaussball, uniformball, uniformcell (uniform over the cell VOLUME, azimuth randomized), or uniformplane (same r-z rectangle but every particle in the phi=0 half-plane, i.e. a uniform SLICE -- upstream's original uniformcell behaviour)"
            arg_type = String
            default = "point"
        "--spawnsize"
            help = "sigma of the gaussball mode, or radius of the uniformball mode (m)"
            arg_type = Float64
            default = 0.0
        "--spawnrlo"
            help = "uniformcell mode: minimum radius (m)"
            arg_type = Float64
            default = 0.0
        "--spawnrhi"
            help = "uniformcell mode: maximum radius (m)"
            arg_type = Float64
            default = 0.0
        "--spawnzlo"
            help = "uniformcell mode: minimum axial position (m)"
            arg_type = Float64
            default = 0.0
        "--spawnzhi"
            help = "uniformcell mode: maximum axial position (m)"
            arg_type = Float64
            default = 0.0
        "--trajprint"
            help = "if nonzero N, print a row for EVERY leg of particles 1..N (negative idx in column 1), giving full trajectory histories on any field, species and spawn mode. Cost is ~1 row per collision per traced particle, so keep N small (~12)."
            arg_type = Int
            default = 0
        "--sampler"
            help = "collision partner sampler: exact (direct flux-weighted Maxwellian rejection; default) or table (legacy upstream lookup-table rejection, kept for A/B against upstream behavior)"
            arg_type = String
            default = "exact"
        "--spawnclip"
            help = "if nonzero (default), reject spawn positions separated from the spawn region's reference point by geometry, i.e. born inside a wall or outside the domain; resamples until valid"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

# Parse command line arguments
args = parse_commandline()
# Set global constants
const MASS_PARTICLE = args["M"] # AMU
const MASS_BUFFER_GAS = args["m"] # AMU
const MASS_REDUCED = MASS_PARTICLE * MASS_BUFFER_GAS /
    (MASS_PARTICLE + MASS_BUFFER_GAS) # AMU
const kB = 8314.46 # AMU m^2 / (s^2 K)
const σ_BUFFER_GAS_PARTICLE = args["sigma"] # m^2
const TRAJPRINT = args["trajprint"] # 0 disables per-leg trajectory rows
# Footgun guard: --trajprint is a PARTICLE COUNT, not a boolean, and each
# traced particle emits one row per collision (~6300 on the Takahashi field).
# Confusing it with -n would try to write ~1e9 rows and serialize every
# thread on the stdout lock.
TRAJPRINT > 100 && @printf(stderr,
    "WARNING: --trajprint %d traces EVERY leg of %d particles -- expect of order %.0e stdout rows and heavy lock contention. Figures need ~12.\n",
    TRAJPRINT, TRAJPRINT, 6.3e3*TRAJPRINT)
args["sampler"] in ("table", "exact") || error("--sampler must be table or exact")
const EXACT_SAMPLER = args["sampler"] == "exact"
const SPAWNCLIP = args["spawnclip"]
# Reference point of the spawn region: the ball centre, or the centre of the
# uniformcell rectangle.  A candidate spawn is physical iff the straight line
# from this reference to it crosses no geometry (see SimulateParticles).
const SPAWN_REF = args["spawn"] in ("uniformcell", "uniformplane") ?
    [0.5*(args["spawnrlo"] + args["spawnrhi"]), 0.0, 0.5*(args["spawnzlo"] + args["spawnzhi"])] :
    [args["r"], 0.0, args["z"]]

# Setup for rejection sampling of collision velocities

# Parameters for proposal distribution for rejection sampling
struct SampleParams
    μ_vg::Float64
    σ_vg::Float64
    σ_θ::Float64
end

# Table of parameters for proposal distribution for rejection sampling for different values of temperature T and relative velocity U
struct LookupTable
    Tmin::Float64
    Tstep::Float64
    Tmax::Float64
    nT::Int64
    Umin::Float64
    Ustep::Float64
    Umax::Float64
    nU::Int64
    table::Matrix{SampleParams}
end

# Evaluates the PDF of a Gaussian distribution with mean μ and standard deviation σ at x
@inline function g(x, μ, σ)
    exp(-0.5*((x-μ)/σ)^2)/(σ*sqrt(2*π))
end

# Given a mean relative velocity u, temperature T and parameters for proposal distributions μ_vg, σ_vg, σ_θ, uses rejection sampling to draw relative speeds v_g and angles θ from the relative velocity. M describes the threshold for accepting a sample. M should be higher if the proposal distribution is farther from the correct distribution.
function sample(u, T, μ_vg, σ_vg, σ_θ, M=2.0)
    if T < 1E-2
        return (u, 0.0)
    end
    v_g = 0.0
    θ = 0.0
    bessel = 0.0
    i = 0
    imax = 50*M
    while true
        y = abs(μ_vg + σ_vg * Random.randn())
        bessel = SpecialFunctions.besseli(0, min(MASS_BUFFER_GAS*u*y/(kB*T), 10))
        f_y = exp(-MASS_BUFFER_GAS*(u^2+y^2)/(2*kB*T)) * y * bessel * MASS_BUFFER_GAS / (kB*T)
        r = f_y/(M*g(y, μ_vg, σ_vg))
        if Random.rand() < r
            v_g = y
            break
        end
        if i > imax
            println(stderr, "Maximum iterations exceeded in sampling v_g for u $u, T $T.")
            v_g = μ_vg
            break
        end
        i += 1
    end
    i = 0
    while true
        y = abs(σ_θ * Random.randn())
        f_y = exp(MASS_BUFFER_GAS * u * v_g * cos(y) / (kB*T)) / (pi * bessel)
        r = f_y/(2*M*g(y, 0, σ_θ))
        if Random.rand() < r && y < π
            θ = y
            break
        end
        if i > imax
            println(stderr, "Maximum iterations exceeded in sampling θ for u $u, T $T.")
            v_g = μ_vg
            break
        end
        i += 1
    end
    return v_g, θ
end

# Given temperature, mean relative velocity, and a lookup table, samples collision velocities
function sample(u, T, table, M=2.0)
    i_T = max(1,min(round(Int64, (T - table.Tmin) / table.Tstep), table.nT))
    i_U = max(1,min(round(Int64, (u - table.Umin) / table.Ustep), table.nU))
    p = table.table[i_T, i_U]
    return sample(max(table.Umin,min(u, table.Umax)), max(table.Tmin,min(T, table.Tmax)), p.μ_vg, 1.5*p.σ_vg, 3*p.σ_θ, M)
end

# Generates a lookup table of means and standard deviations for Gaussian proposal distributions for rejection sampling.
function generate_lookup_table(Tmin, Tstep, Tmax, Umin, Ustep, Umax, nsamples=100, Msample=20)
    Ts = Tmin:Tstep:Tmax
    Us = Umin:Ustep:Umax
    table = LookupTable(Tmin, Tstep, Tmax, length(Ts), Umin, Ustep, Umax, length(Us), Matrix{SampleParams}(undef, length(Ts), length(Us)))
    for (i, T) in enumerate(Ts)
        for (j, U) in enumerate(Us)
            σ_vg = 1.5*sqrt(8*kB*(T+0.2)/(π*MASS_BUFFER_GAS))
            σ_θ = 1.5*pi*σ_vg/(σ_vg+U)
            μ_vg = U + σ_vg
            vg_samples = zeros(nsamples)
            θ_samples = zeros(nsamples)
            for i in 1:nsamples
                vg_samples[i], θ_samples[i] = sample(U, T, μ_vg, σ_vg, σ_θ, Msample)
            end
            table.table[i,j] = SampleParams(mean(vg_samples), std(vg_samples), std(θ_samples))
        end
    end
    return table
end

"""
    collide!(v, vgx, vgy, vgz, T)

Accepts as input the velocity of a particle v, the mean velocity of a buffer gas atom vgx, vgy, vgz, and the buffer gas temperature T. Computes the velocity of the particle after they undergo a collision, treating the particles as hard spheres and assuming a random scattering parameter and buffer gas atom velocity (assuming the particles are moving slower than the buffer gas atoms). Follows Appendices B and C of Boyd 2017. Note that v and vg are modified.
"""
# Exact collision-partner sampler: draws the buffer gas thermal velocity from
# the flux-weighted (relative-speed-weighted) Maxwellian by rejection --
# propose u ~ Maxwell(T), accept with probability |w0-u|/gmax. w0 is the
# particle velocity in the gas mean-flow frame; gmax = |w0| + 8 sigma_c bounds
# the relative speed (acceptance tail beyond it ~1e-13). Statistically exact
# at any relative speed; validated < 1% against a 2e7-sample reference MC
# (session log 2026-08-19).
@inline function exact_partner(w0, T)
    s = sqrt(kB*T/MASS_BUFFER_GAS)
    gmax = LinearAlgebra.norm(w0) + 8.0*s
    gmax == 0.0 && return [0.0, 0.0, 0.0]  # T=0 gas at rest relative to particle: delta-function partner (also SimulateParticles' zero-velocity output_dim dummy call)
    while true
        u = s .* Random.randn(3)
        g = sqrt((w0[1]-u[1])^2+(w0[2]-u[2])^2+(w0[3]-u[3])^2)
        if Random.rand()*gmax < g
            return u
        end
    end
end

@inline function collide!(v, vg, T, table)
    if EXACT_SAMPLER
        vg = vg .+ exact_partner(v .- vg, T)
    else
    u = sqrt((v[1]-vg[1])^2+(v[2]-vg[2])^2+(v[3]-vg[3])^2)
    vgmag, θ = sample(u, T, table)
    if u < 1E-3
        vgdir = LinearAlgebra.normalize(Random.rand(3) .- 0.5)
    else
        vgdir = (vg - v) ./ u
    end
    vrand = LinearAlgebra.normalize(Random.rand(3) .- 0.5)
    vperp = LinearAlgebra.normalize(vrand .- dot(vrand, vgdir) .* vgdir)
    vg = v .+ vgmag .* (cos(θ) .* vgdir .+ sin(θ) .* vperp)
    end
    cosχ = 2*Random.rand() - 1
    sinχ = sqrt(1 - cosχ^2)
    θ = 2 * π * Random.rand()
    g = sqrt((v[1] - vg[1])^2 + (v[2] - vg[2])^2 + (v[3] - vg[3])^2)
    v[1] = MASS_PARTICLE * v[1] + MASS_BUFFER_GAS * (vg[1] + g * cosχ)
    v[2] = MASS_PARTICLE * v[2] + MASS_BUFFER_GAS * (vg[2] + g * sinχ * cos(θ))
    v[3] = MASS_PARTICLE * v[3] + MASS_BUFFER_GAS * (vg[3] + g * sinχ * sin(θ))
    v .= v ./ (MASS_PARTICLE + MASS_BUFFER_GAS)
end

"""
    freePropagate!(xnext, x, v, d, ω)

Updates xnext by propagating a particle at x with velocity v a distance d in a harmonic potential with frequency ω
"""
@inline function freePropagate!(xnext, x, v, t, ω)
    xnext[3] = x[3] + v[3]*t
    if ω != 0
        pm = sign(ω)
        ω = abs(ω)
        if pm > 0
            sint = sin(sqrt(2.0)*ω*t)
            cost = cos(sqrt(2.0)*ω*t)
            xnext[1] = x[1]*cost + v[1]*sint/(sqrt(2.0)*ω)
            xnext[2] = x[2]*cost + v[2]*sint/(sqrt(2.0)*ω)
            v[1] = v[1]*cost-2*x[1]*ω*sint
            v[2] = v[2]*cost-2*x[2]*ω*sint
        else
            sint = sinh(sqrt(2.0)*ω*t)
            cost = cosh(sqrt(2.0)*ω*t)
            xnext[1] = x[1]*cost + v[1]*sint/(sqrt(2.0)*ω)
            xnext[2] = x[2]*cost + v[2]*sint/(sqrt(2.0)*ω)
            v[1] = v[1]*cost+2*x[1]*ω*sint
            v[2] = v[2]*cost+2*x[2]*ω*sint
        end
    else
        xnext[1] = x[1] + v[1]*t
        xnext[2] = x[2] + v[2]*t
    end
end

# Propagates a particle to its next collision by updating xnext. Assumes starting position x, starting velocity v, a step distance of d, and a trap of frequency ω between zmin and zmax.
@inline function freePropagate!(xnext, x, v, d, ω, zmin, zmax)
    vmag = sqrt(v[1]^2+v[2]^2+v[3]^2)
    if vmag < 1E-6
        return
    end
    t = d/vmag
    x3next = x[3]+v[3]*t
    if x[3] > zmax && x3next < zmax
        t1 = (x[3] - zmax)/v[3]
        freePropagate!(xnext, x, v, t1, 0)
        xnext[3] = zmax
        d -= sqrt((x[1]-xnext[1])^2+(x[2]-xnext[2])^2+(x[3]-xnext[3])^2)
        return freePropagate!(xnext, deepcopy(xnext), v, d, ω, zmin, zmax)
    elseif x[3] < zmax && x3next > zmax
        t1 = (zmax - x[3])/v[3]
        freePropagate!(xnext, x, v, t1, ω)
        xnext[3] = zmax
        d -= sqrt((x[1]-xnext[1])^2+(x[2]-xnext[2])^2+(x[3]-xnext[3])^2)
        return freePropagate!(xnext, deepcopy(xnext), v, d, 0, zmin, zmax)
    elseif x[3] > zmin && x3next < zmin
        t1 = (x[3] - zmin)/v[3]
        freePropagate!(xnext, x, v, t1, 0)
        xnext[3] = zmin
        d -= sqrt((x[1]-xnext[1])^2+(x[2]-xnext[2])^2+(x[3]-xnext[3])^2)
        return freePropagate!(xnext, deepcopy(xnext), v, d, ω, zmin, zmax)
    elseif x[3] < zmin && x3next > zmin
        t1 = (zmin - x[3])/v[3]
        freePropagate!(xnext, x, v, t1, ω)
        xnext[3] = zmin
        d -= sqrt((x[1]-xnext[1])^2+(x[2]-xnext[2])^2+(x[3]-xnext[3])^2)
        return freePropagate!(xnext, deepcopy(xnext), v, d, 0, zmin, zmax)
    end
    if zmin < x[3] < zmax
        return freePropagate!(xnext, x, v, t, ω)
    else
        return freePropagate!(xnext, x, v, t, 0)
    end
end

"""
    freePath(vrel, T, ρ)

Accepts as input the velocity of a particle relative to a buffer gas atom v, the buffer gas temperature T and the buffer gas density ρ. Draws a distance the particle travels before it hits a buffer gas atom from an exponential distribution, accounting for a velocity-dependent mean free path. Note that this assumes that the gas properties don't change significantly over a mean free path.
"""
@inline function freePath(v, vrel, T, ρ)
    λ = sqrt(v[1]^2 + v[2]^2 + v[3]^2)/(ρ*σ_BUFFER_GAS_PARTICLE*sqrt(8*kB*T/(MASS_BUFFER_GAS*pi) + vrel^2))
    return min(-log(Random.rand()) * λ, 1000.0)
end

"""
    getIntersection(x1, x2, y2, x3, y3, x4, y4)

Returns whether the line segments ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4)) intersect. See "Faster Line Segment Intersection" from Graphics Gems III ed. David Kirk.
"""
@inline function getIntersection(x1, y1, x2, y2, x3, y3, x4, y4)
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    num = x4*(y1 - y3) + x1*(y3-y4) + x3*(y4-y1)
    if denom > 0
        if num < 0 || num > denom
            return false
        end
        num = x3*(y2 - y1) + x2*(y1 - y3) + x1*(y3-y2)
        if num < 0 || num > denom
            return false
        end
    else
        if num > 0 || num < denom
            return false
        end
        num = x3*(y2 - y1) + x2*(y1 - y3) + x1*(y3-y2)
        if num > 0 || num < denom
            return false
        end
    end
    return true
end


"""
    propagate(xinit, vin, interp!, getCollision, omega=0.0)

Accepts as input the position of a particle xinit, its velocity v, the function interp!, which takes as input a position and a vector and updates the vector to describe the gas [x, y, vgx, vgy, vgz, T, ρ], and the function getCollision(x1, x2), which returns whether if the segment from x1 to x2 intersects geometry. Computes the path of the particle until it getCollision returns true. Returns a vector of simulation results with elements x, y, z, xnext, ynext, znext, vx, vy, vz, collides, time.
"""
@inline function propagate(xinit, vin, interp!, getCollision, table, ω=0.0, zmin=-Inf, zmax=Inf, pflip=0.0, stats=nothing, max_x_geom=Inf, idx=0)
    x = deepcopy(xinit)
    props = zeros(8) # x, y, vgx, vgy, vgz, T, ρ, dmin
    interp!(props, x)
    xnext = deepcopy(x)
    v = deepcopy(vin)
    time = 0.0
    collides = 0
    
    # If nearly stationary, provide a first collision
    if LinearAlgebra.norm(v) < 1E-6
        collide!(v, view(props, 3:5), props[6], table)
        collides += 1
    end

    # Randomize the spin
    if rand() < 0.5
        ω = -ω
    end

    while true
        interp!(props, x)
        vrel = sqrt((v[1] - props[3])^2 + (v[2] - props[4])^2 + (v[3] - props[5])^2)
        dist = freePath(v, vrel, props[6], props[7])
        freePropagate!(xnext, x, v, dist, ω, zmin, zmax)
        if getCollision(x, xnext) != 0
            return (x[1], x[2], x[3], xnext[1], xnext[2], xnext[3], v[1], v[2], v[3], collides, time)
        else
            time += dist / LinearAlgebra.norm(v)
            collides += 1
            # Record every leg of the first TRAJPRINT particles (diff 11).
            if TRAJPRINT != 0 && idx <= TRAJPRINT
                print(@sprintf("%d %e %e %e %e %e %e %e %e %e %d %e\n", -idx,
                x[1], x[2], x[3], xnext[1], xnext[2], xnext[3], v[1], v[2], v[3], collides, time))
            end
        end
        if !isnothing(stats)
            updateStats!(stats, x, v, time, collides, dist)
        end
        x .= xnext
        collide!(v, view(props, 3:5), props[6], table)
        if rand() < pflip
            ω = -ω
        end
    end
end

# Tracks 2D interpolation lookups for the current SimulateParticles call.
# Atomic increments make the diagnostic truthful across particle tasks.
const interps = Threads.Atomic{Int}(0)

function build_field(geomFile, gridFile)
    bounds = Matrix(CSV.read(geomFile, DataFrame; header = ["min","max"], skipto=6, limit=2,ignorerepeated=true,delim=' '))
    geom = Matrix(CSV.read(geomFile, DataFrame; header=["ID","x1","y1","x2","y2"],skipto=10, ignorerepeated=true,delim=' '))
    max_x_geom = maximum(geom[:,[2,4]])
    griddf = CSV.read(gridFile, DataFrame; header=["x","y","T","ρ","ρm","vx","vy","vz"],skipto=10,ignorerepeated=true,delim=' ')

    # Reorder columns and only include grid cells with data
    DataFrames.select!(griddf, [:x, :y, :vx, :vy, :vz, :T, :ρ])
    griddf = griddf[griddf.T .> 0, :]
    # Column 8 is kept at 0 so props keeps its 8-slot layout; nothing reads it
    # since diff 15 removed the lookup cache.
    griddf[!, :dmin] .= 0
    grids = Matrix(griddf)

    # Make a tree for efficient nearest neighbor search
    kdtree = KDTree(transpose(Matrix(grids[:,1:2])); leafsize=10)

    # Set up the lookup table for sampling collision velocities
    Tmin = minimum(griddf.T)
    Tmax = maximum(griddf.T)
    Tstep = (Tmax - Tmin) / 20
    Umin = 0.0
    Umax = 1.5*maximum(sqrt.(griddf.vx.^2 .+ griddf.vy.^2))
    Ustep = (Umax - Umin) / 20
    table = generate_lookup_table(Tmin, Tstep, Tmax, Umin, Ustep, Umax)

    """
        interpolate!(props, x)
    
    Updates the gas properties props with the data from point x.
    """
    @inline function interpolate!(props, x)
        # props: x, y, vgx, vgy, vgz, T, ρ, (unused)
        # Exact nearest-centre lookup on every call (diff 15): the cell whose
        # centre is nearest in the (z, r) plane supplies the gas properties.
        Threads.atomic_add!(interps, 1)
        interp = view(grids, knn(kdtree, [x[3], sqrt(x[1]^2 + x[2]^2)], 1)[1][1], :)
        props[1] = interp[1]            # x
        props[2] = interp[2]            # y
        θ = atan(x[2],x[1])
        props[3] = interp[4] * cos(θ)   # vgx
        props[4] = interp[4] * sin(θ)   # vgy
        props[5] = interp[3]            # vgz
        props[6] = interp[6]            # T
        props[7] = interp[7]            # ρ
        props[8] = interp[8]
    end

    """
        getCollision(x1, x2)

    Checks whether the line between x1 and x2 intersects geometry or the boundary of the simulation region. Returns 0 if no collision, 1 if the particle collides with geometry, or 2 if the particle leaves the simulation bounds
    """
    @inline function getCollision(x1, x2)
        r1 = sqrt(x1[1]^2 + x1[2]^2)
        r2 = sqrt(x2[1]^2 + x2[2]^2)
        for i in 1:size(geom)[1]
            if getIntersection(geom[i,2],geom[i,3],geom[i,4],geom[i,5],x1[3],r1,x2[3],r2)
                return 1
            end
        end
        if x2[3] < bounds[1,1] || x2[3] > bounds[1,2] || r2 > bounds[2,2]
            return 2
        end
        return 0
    end

    return (; interpolate!, getCollision, table, bounds, max_x_geom)
end


"""
    SimulateParticles(geomFile, gridFile, nParticles, generateParticle)

    Runs nParticles simulations using the SPARTA outputs with paths geomFile and gridFile to define the surfaces and buffer gas properties. Generates each particle with position and velocity returned by generateParticle. Returns a nParticles by 11 matrix of simulation results, with columns x, y, z, xnext, ynext, znext, vx, vy, vz, collides, time.
"""
function SimulateParticles(
    geomFile, 
    gridFile, 
    nParticles,
    generateParticle,
    print_stuff=true,
    ω=0.0,
    zmin=-Inf,
    zmax=Inf,
    pflip=0.0,
    saveall=0,
    savestats=nothing,
    saveexitstats=nothing,
    rbins=100,
    zbins=100;
    savespawns=nothing,
    make_stats=nothing
    )

    nParticles isa Integer && !(nParticles isa Bool) && nParticles > 0 ||
        throw(ArgumentError("nParticles must be a positive integer, got $nParticles"))
    interps[] = 0
    (; interpolate!, getCollision, table, bounds, max_x_geom) = build_field(geomFile, gridFile)

    stats_enabled = !(savestats === nothing || savestats === false)
    exitstats_enabled = !(saveexitstats === nothing || saveexitstats === false)

    # Per-leg accumulator factory (diff 14).  Defaults to the bundled
    # StatsArray.  Pass make_stats to instrument a run with any accumulator
    # implementing updateStats!(acc, x, v, t, ncolls, lfree), called once per
    # leg, and merge!(dst, src), called to fold a particle into its chunk
    # partial and the partials into the run totals.  The reduction is what
    # provides thread-safety and determinism (diff 13), so an accumulator does
    # not have to be thread-safe itself.
    new_stats = isnothing(make_stats) ?
        (() -> StatsArray(bounds[2,1], bounds[2,2], rbins, bounds[1,1], bounds[1,2], zbins)) : make_stats

    # Runs one simulation to check the length of the output array
    output_dim = length(propagate(zeros(3), zeros(3), interpolate!, (x,y)->true, table))
    outputs = zeros(nParticles, output_dim)

    # Initializes statistics arrays
    allstats = nothing
    boundstats = nothing
    if stats_enabled
        allstats = new_stats()
    end
    if exitstats_enabled
        boundstats = new_stats()
    end
    
    # Per-chunk partial accumulators, reduced in fixed chunk order after the
    # loop (diff 13).  Upstream merged every particle straight into allstats /
    # boundstats from inside the threaded loop with no synchronization, which both
    # dropped samples and left the merge order -- hence the floating-point result --
    # varying run to run.  Owning the partition explicitly is what makes a chunk's
    # partial reachable by exactly one task, so no lock is needed.
    chunks = collect(Iterators.partition(1:nParticles, cld(nParticles, Threads.nthreads())))
    part_all = stats_enabled ? [new_stats() for _ in chunks] : nothing
    part_bound = exitstats_enabled ? [new_stats() for _ in chunks] : nothing

    spawn_rejects = Threads.Atomic{Int}(0)
    # --spawnout sink: the trajectory rows carry only the LAST leg, so the
    # birthplace has to be recorded here, where generateParticle's result is
    # still in scope.  Written from this function rather than returned so the
    # existing (outputs, boundstats, allstats) contract is untouched.
    spawns = isnothing(savespawns) ? nothing : zeros(nParticles, 3)
    @sync for (c, chunk) in enumerate(chunks)
        Threads.@spawn for i in chunk
            stats = nothing
            if exitstats_enabled || stats_enabled
                stats = new_stats()
            end
            xpart, vpart = generateParticle()
            if SPAWNCLIP != 0
                tries = 0
                while getCollision(SPAWN_REF, xpart) != 0
                    tries += 1
                    tries > 1000 && error("--spawnclip: 1000 consecutive spawn candidates were cut off from the spawn reference $(SPAWN_REF) by geometry; check the spawn region, -r/-z, and the .surfs file")
                    xpart, vpart = generateParticle()
                end
                tries > 0 && Threads.atomic_add!(spawn_rejects, tries)
            end
            isnothing(spawns) || (spawns[i,:] .= xpart)
            outputs[i,:] .= propagate(xpart, vpart, interpolate!, getCollision, table, ω, zmin, zmax, pflip, stats, max_x_geom, i)
            colltype = getCollision(outputs[i,[1,2,3]], outputs[i,[4,5,6]])
            if stats_enabled
                merge!(part_all[c], stats)
            end
            if colltype == 2 && exitstats_enabled
                merge!(part_bound[c], stats)
            end
            if print_stuff && (saveall != 0 || colltype == 2)
                print(@sprintf("%d %e %e %e %e %e %e %e %e %e %d %e\n", i,
                outputs[i,1], outputs[i,2], outputs[i,3], outputs[i,4], outputs[i,5], outputs[i,6], outputs[i,7], outputs[i,8], outputs[i,9], outputs[i,10], outputs[i,11]))
            end
        end
    end

    # Fixed-order reduction: the only place the run totals are written.
    for c in eachindex(chunks)
        isnothing(part_all) || merge!(allstats, part_all[c])
        isnothing(part_bound) || merge!(boundstats, part_bound[c])
    end
    if !isnothing(savespawns)
        CSV.write(savespawns, DataFrame(idx = 1:nParticles, x = spawns[:,1],
                                        y = spawns[:,2], z = spawns[:,3]))
    end
    if SPAWNCLIP != 0 && spawn_rejects[] > 0
        @printf(stderr, "Spawn candidates rejected by --spawnclip: %d (%.2f%% of %d draws)\n",
            spawn_rejects[], 100*spawn_rejects[]/(spawn_rejects[] + nParticles), spawn_rejects[] + nParticles)
    end

    return outputs, boundstats, allstats
end

"""
    main()

The main function starts a particle simulation based on the command line arguments, printing outputs to stdout and timing information to stderr.
"""
function main(args)

    nParticles = args["n"]

    if args["seed"] != 0
        Random.seed!(args["seed"])
    end

    println("idx x y z xnext ynext znext vx vy vz collides time")

    # Define particle generation
    boltzmann = sqrt(kB*args["T"]/MASS_PARTICLE)
    spawnmode = args["spawn"]
    spawnmode in ("point", "gaussball", "uniformball", "uniformcell", "uniformplane") ||
        error("unknown --spawn mode: $spawnmode")
    spawnPosition() =
        if spawnmode == "point"
            [args["r"], 0.0, args["z"]]
        elseif spawnmode == "gaussball"
            s = args["spawnsize"]
            [args["r"] + s*Random.randn(), s*Random.randn(), args["z"] + s*Random.randn()]
        elseif spawnmode == "uniformball"
            u = Random.randn(3)
            u .*= args["spawnsize"] * cbrt(Random.rand()) / LinearAlgebra.norm(u)
            [args["r"] + u[1], u[2], args["z"] + u[3]]
        else # uniformcell / uniformplane: uniform over the r-z rectangle,
             # volume weighted (r ~ r*dr).  uniformcell also draws the azimuth
             # uniformly, so the ensemble is genuinely uniform over the cell
             # VOLUME; uniformplane keeps every particle in the phi = 0
             # half-plane (upstream's behaviour), a uniform SLICE.
            rlo = args["spawnrlo"]
            rhi = args["spawnrhi"]
            zlo = args["spawnzlo"]
            zhi = args["spawnzhi"]
            r = sqrt(rlo^2 + Random.rand()*(rhi^2 - rlo^2))
            z = zlo + Random.rand()*(zhi - zlo)
            if spawnmode == "uniformplane"
                [r, 0.0, z]
            else
                ϕ = 2π*Random.rand()
                [r*cos(ϕ), r*sin(ϕ), z]
            end
        end
    generateParticle() = (
        spawnPosition(),
        [args["vr"] + Random.randn() * boltzmann, Random.randn() * boltzmann, args["vz"] + Random.randn() * boltzmann])

    # Set simulation parameters and run simulation
    nthreads = Threads.nthreads()
    @printf(stderr, "Threads: %d\n", nthreads)
    start = time()
    outputs, boundstats, allstats = SimulateParticles(
        args["geom"],
        args["flow"],
        nParticles,
        generateParticle,
        true,
        args["omega"],
        args["zmin"],
        args["zmax"],
        args["pflip"],
        args["saveall"],
        !isnothing(args["stats"]),
        !isnothing(args["exitstats"]);
        savespawns = args["spawnout"])
    runtime = time() - start
    if !isnothing(args["stats"])
        CSV.write(args["stats"], convert(DataFrame, allstats))
    end
    if !isnothing(args["exitstats"])
        CSV.write(args["exitstats"], convert(DataFrame, boundstats))
    end

    # Compute and display timing statistics
    @printf(stderr, "Time: %.3e\n",runtime)
    @printf(stderr, "Time per particle: %.3e\n", runtime/nParticles)
    @printf(stderr, "Time per collision: %.3e\n", runtime/sum(outputs[:,10]))
    @printf(stderr, "Interpolates: %.3e\n",interps[])
    @printf(stderr, "Collides: %.3e\n",sum(outputs[:,10]))

    return allstats
end

allstats = main(args)
