# Native-cell Cartesian trajectory statistics for the maintained 3D tracer.
#
# This file is local DSMC-sims code, introduced 2026-09-22 after the statistics
# audit of 2026-09-22 (statistics follow-up).  It is
# intentionally separate from the vendored 2D StatsArray: historical 2D column
# names and cylindrical formulas remain unchanged.

using DataFrames

import Base: convert

const _CARTESIAN_STATS3D_COLUMNS = (
    :cell_id,
    :cell_center_x_m,
    :cell_center_y_m,
    :cell_center_z_m,
    :cell_halfwidth_x_m,
    :cell_halfwidth_y_m,
    :cell_halfwidth_z_m,
    :n_events,
    :vx_mean_m_per_s,
    :vy_mean_m_per_s,
    :vz_mean_m_per_s,
    :vx_population_var_m2_per_s2,
    :vy_population_var_m2_per_s2,
    :vz_population_var_m2_per_s2,
    :vx_vy_population_cov_m2_per_s2,
    :vx_vz_population_cov_m2_per_s2,
    :vy_vz_population_cov_m2_per_s2,
    :leg_end_time_mean_s,
    :leg_end_time_population_var_s2,
    :collision_ordinal_mean,
    :collision_ordinal_population_var,
    :free_path_mean_m,
    :free_path_population_var_m2,
)

mutable struct _CartesianCellStats3D
    metadata::NTuple{6,Float64}
    n_events::Int
    v_mean::Vector{Float64}
    v_m2::Matrix{Float64}
    t_mean::Float64
    t_m2::Float64
    collision_ordinal_mean::Float64
    collision_ordinal_m2::Float64
    free_path_mean::Float64
    free_path_m2::Float64
end

function _CartesianCellStats3D(metadata::NTuple{6,Float64})
    return _CartesianCellStats3D(
        metadata,
        0,
        zeros(3),
        zeros(3, 3),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
end

"""
    CartesianStats3D(cell_index, cell_metadata)

Sparse native-field-cell statistics for 3D trajectories.

`cell_index(x)` must return a positive integer native field-row ID for the
Cartesian tracer position `x`, or throw when `x` cannot be assigned. On the
first event in a cell, `cell_metadata(id)` must return
`(center_x, center_y, center_z, halfwidth_x, halfwidth_y, halfwidth_z)` in
TRACER axes and metres. Half-widths must be positive.

`updateStats!` preserves the particle tracer's hook estimand: one event per
completed nonterminating free-flight leg. The hook supplies the leg-start
position and pre-collision Cartesian velocity, cumulative time and collision
ordinal at the leg end, and that leg's free path. The terminating leg is not
included. Thus `n_events` is a hook-event count, not a particle count, dwell
time, or path-length weight.

Velocity, leg-end time, collision ordinal, and free-path variances are
population central moments (division by `n_events`). Velocity covariances are
also population central moments. Updates use Welford recurrences and `merge!`
uses the corresponding pairwise correction.
"""
struct CartesianStats3D{CellIndex,CellMetadata}
    cell_index::CellIndex
    cell_metadata::CellMetadata
    cells::Dict{Int,_CartesianCellStats3D}
end

function CartesianStats3D(cell_index, cell_metadata)
    return CartesianStats3D(cell_index, cell_metadata,
                            Dict{Int,_CartesianCellStats3D}())
end

function _finite_scalar(value, name::AbstractString)
    x = try
        Float64(value)
    catch
        throw(ArgumentError("$name must be a finite real number"))
    end
    isfinite(x) || throw(ArgumentError("$name must be finite, got $value"))
    return x
end

function _finite_cartesian(value, name::AbstractString)
    length(value) >= 3 ||
        throw(ArgumentError("$name must have at least three Cartesian components"))
    return (
        _finite_scalar(value[1], name),
        _finite_scalar(value[2], name),
        _finite_scalar(value[3], name),
    )
end

function _native_cell_id(value)
    value isa Integer && !(value isa Bool) ||
        throw(ArgumentError("cell_index(x) must return a positive integer native cell ID, got $value"))
    value > 0 ||
        throw(ArgumentError("cell_index(x) must return a positive integer native cell ID, got $value"))
    return try
        Int(value)
    catch
        throw(ArgumentError("native cell ID $value cannot be represented as Int"))
    end
end

function _native_cell_metadata(value, id::Int)
    length(value) == 6 ||
        throw(ArgumentError("cell_metadata($id) must return six values: cx, cy, cz, hx, hy, hz"))
    metadata = ntuple(i -> _finite_scalar(value[i], "cell_metadata($id)[$i]"), 6)
    all(>(0.0), metadata[4:6]) ||
        throw(ArgumentError("cell_metadata($id) half-widths hx, hy, hz must be positive"))
    return metadata
end

@inline function _update_scalar(mean::Float64, m2::Float64,
                                value::Float64, n::Int)
    delta = value - mean
    new_mean = mean + delta / n
    new_m2 = m2 + delta * (value - new_mean)
    return new_mean, new_m2
end

"""
    updateStats!(stats::CartesianStats3D, x, v, t, ncolls, lfree)

Add one nonterminating-leg hook event. All position, velocity, and scalar
inputs must be finite. `t`, `ncolls`, and `lfree` retain the shared tracer
hook's cumulative-leg-end-time, collision-ordinal, and free-path meanings.
"""
function updateStats!(s::CartesianStats3D, x, v, t, ncolls, lfree)
    position = _finite_cartesian(x, "x")
    velocity = _finite_cartesian(v, "v")
    leg_end_time = _finite_scalar(t, "t")
    collision_ordinal = _finite_scalar(ncolls, "ncolls")
    free_path = _finite_scalar(lfree, "lfree")

    # Validate through the normalized tuple above, but preserve the caller's
    # position container for field backends that specialize `cell_index` on it.
    id = _native_cell_id(s.cell_index(x))
    cell = get!(s.cells, id) do
        _CartesianCellStats3D(_native_cell_metadata(s.cell_metadata(id), id))
    end

    n = cell.n_events + 1
    velocity_delta = ntuple(i -> velocity[i] - cell.v_mean[i], 3)
    @inbounds for i in 1:3
        cell.v_mean[i] += velocity_delta[i] / n
    end
    velocity_delta2 = ntuple(i -> velocity[i] - cell.v_mean[i], 3)
    @inbounds for i in 1:3, j in 1:3
        cell.v_m2[i, j] += velocity_delta[i] * velocity_delta2[j]
    end
    cell.t_mean, cell.t_m2 =
        _update_scalar(cell.t_mean, cell.t_m2, leg_end_time, n)
    cell.collision_ordinal_mean, cell.collision_ordinal_m2 =
        _update_scalar(cell.collision_ordinal_mean, cell.collision_ordinal_m2,
                       collision_ordinal, n)
    cell.free_path_mean, cell.free_path_m2 =
        _update_scalar(cell.free_path_mean, cell.free_path_m2, free_path, n)
    cell.n_events = n
    return cell
end

@inline function _merge_scalar(mean_a::Float64, m2_a::Float64, n_a::Int,
                               mean_b::Float64, m2_b::Float64, n_b::Int)
    n = n_a + n_b
    delta = mean_b - mean_a
    return mean_a + delta * n_b / n,
           m2_a + m2_b + delta * delta * (Float64(n_a) * n_b / n)
end

function _copy_cell(cell::_CartesianCellStats3D)
    return _CartesianCellStats3D(
        cell.metadata,
        cell.n_events,
        copy(cell.v_mean),
        copy(cell.v_m2),
        cell.t_mean,
        cell.t_m2,
        cell.collision_ordinal_mean,
        cell.collision_ordinal_m2,
        cell.free_path_mean,
        cell.free_path_m2,
    )
end

function _merge_cell!(a::_CartesianCellStats3D, b::_CartesianCellStats3D, id::Int)
    a.metadata == b.metadata ||
        throw(ArgumentError("native cell $id has inconsistent metadata across accumulators"))
    n_a = a.n_events
    n_b = b.n_events
    n_b == 0 && return a
    if n_a == 0
        replacement = _copy_cell(b)
        a.n_events = replacement.n_events
        a.v_mean .= replacement.v_mean
        a.v_m2 .= replacement.v_m2
        a.t_mean = replacement.t_mean
        a.t_m2 = replacement.t_m2
        a.collision_ordinal_mean = replacement.collision_ordinal_mean
        a.collision_ordinal_m2 = replacement.collision_ordinal_m2
        a.free_path_mean = replacement.free_path_mean
        a.free_path_m2 = replacement.free_path_m2
        return a
    end

    n = n_a + n_b
    velocity_delta = b.v_mean .- a.v_mean
    between_weight = Float64(n_a) * n_b / n
    a.v_m2 .+= b.v_m2 .+ between_weight .* (velocity_delta * velocity_delta')
    a.v_mean .+= (n_b / n) .* velocity_delta
    a.t_mean, a.t_m2 = _merge_scalar(a.t_mean, a.t_m2, n_a,
                                     b.t_mean, b.t_m2, n_b)
    a.collision_ordinal_mean, a.collision_ordinal_m2 =
        _merge_scalar(a.collision_ordinal_mean, a.collision_ordinal_m2, n_a,
                      b.collision_ordinal_mean, b.collision_ordinal_m2, n_b)
    a.free_path_mean, a.free_path_m2 =
        _merge_scalar(a.free_path_mean, a.free_path_m2, n_a,
                      b.free_path_mean, b.free_path_m2, n_b)
    a.n_events = n
    return a
end

"""
    merge!(destination::CartesianStats3D, source::CartesianStats3D)

Merge sparse touched-cell statistics with stable pairwise moment formulas.
Source cells are copied when first touched in `destination`, so the two
accumulators do not share mutable moment storage after the merge.
"""
function Base.merge!(a::CartesianStats3D, b::CartesianStats3D)
    for id in sort!(collect(keys(b.cells)))
        if haskey(a.cells, id)
            _merge_cell!(a.cells[id], b.cells[id], id)
        else
            a.cells[id] = _copy_cell(b.cells[id])
        end
    end
    return a
end

function _empty_cartesian_stats3d_dataframe()
    return DataFrame(
        cell_id = Int[],
        cell_center_x_m = Float64[],
        cell_center_y_m = Float64[],
        cell_center_z_m = Float64[],
        cell_halfwidth_x_m = Float64[],
        cell_halfwidth_y_m = Float64[],
        cell_halfwidth_z_m = Float64[],
        n_events = Int[],
        vx_mean_m_per_s = Float64[],
        vy_mean_m_per_s = Float64[],
        vz_mean_m_per_s = Float64[],
        vx_population_var_m2_per_s2 = Float64[],
        vy_population_var_m2_per_s2 = Float64[],
        vz_population_var_m2_per_s2 = Float64[],
        vx_vy_population_cov_m2_per_s2 = Float64[],
        vx_vz_population_cov_m2_per_s2 = Float64[],
        vy_vz_population_cov_m2_per_s2 = Float64[],
        leg_end_time_mean_s = Float64[],
        leg_end_time_population_var_s2 = Float64[],
        collision_ordinal_mean = Float64[],
        collision_ordinal_population_var = Float64[],
        free_path_mean_m = Float64[],
        free_path_population_var_m2 = Float64[],
    )
end

"""
    convert(DataFrame, stats::CartesianStats3D)

Return one row per touched native field cell, sorted by `cell_id`. The full
typed schema is returned when no cell was touched. Every `*_population_var*`
and `*_population_cov*` column is a central population moment.
"""
function convert(::Type{DataFrame}, s::CartesianStats3D)
    df = _empty_cartesian_stats3d_dataframe()
    for id in sort!(collect(keys(s.cells)))
        cell = s.cells[id]
        n = cell.n_events
        n > 0 || continue
        population_scale = 1.0 / n
        metadata = cell.metadata
        push!(df, (
            id,
            metadata[1], metadata[2], metadata[3],
            metadata[4], metadata[5], metadata[6],
            n,
            cell.v_mean[1], cell.v_mean[2], cell.v_mean[3],
            cell.v_m2[1, 1] * population_scale,
            cell.v_m2[2, 2] * population_scale,
            cell.v_m2[3, 3] * population_scale,
            cell.v_m2[1, 2] * population_scale,
            cell.v_m2[1, 3] * population_scale,
            cell.v_m2[2, 3] * population_scale,
            cell.t_mean,
            cell.t_m2 * population_scale,
            cell.collision_ordinal_mean,
            cell.collision_ordinal_m2 * population_scale,
            cell.free_path_mean,
            cell.free_path_m2 * population_scale,
        ))
    end
    @assert Tuple(propertynames(df)) == _CARTESIAN_STATS3D_COLUMNS
    return df
end
