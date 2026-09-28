# 3D gas lookup and triangle walls, shared by maintained and legacy drivers.
# Tracer axes (x,y,z) are SPARTA (y,z,x). Native gas cells retain Cartesian
# flow and statistics; no azimuth rotation or averaging is performed.
# Maintained Runtime3D selects validated column input, no lookup table, and
# strict missing-cell errors. Legacy callers retain explicit compatibility
# defaults below. The default escape remains the inherited inscribed cylinder;
# geometry_mode=:explicit loads only the field and requires a separate canonical
# contact callback. See README-3d.md for the legacy limitations.
const AXIS = 3   # tracer x[3] is the SPARTA axial coordinate

# set by build_field3d so the driver can report the lookup-fallback count
const field3d_misses = Ref{Any}(nothing)

# ------------------------------------------------------------------ readers

function _hdr(path, n)
    open(path) do io
        [readline(io) for _ in 1:n]
    end
end

"""
    read_surfs3d(path) -> (V0, E1, E2, LO, HI, box, max_axial)

Triangles from a 3D `cell.surfs` (written by tools/field2tracer3d.py), stored
in TRACER axes: tracer (x1,x2,x3) = SPARTA (y,z,x).  Columns of each returned
3xT matrix are one triangle: V0 a vertex, E1/E2 the two edge vectors, LO/HI
the axis-aligned bounding box.
"""
function read_surfs3d(path)
    h = _hdr(path, 9)
    h[3] == "ITEM: NUMBER OF SURFS" || error("$path: missing surface count header")
    h[9] == "ITEM: SURFS id v1x v1y v1z v2x v2y v2z v3x v3y v3z" || error("$path: unsupported 3D surface columns")
    box = [parse.(Float64, split(h[k])) for k in 6:8]
    df = CSV.read(path, DataFrame; header = false, skipto = 10,
                  ignorerepeated = true, delim = ' ')
    M = Matrix(df)                      # id v1x v1y v1z v2x ... v3z  (SPARTA)
    n = size(M, 1)
    n == parse(Int,h[4]) && n>0 && size(M,2)==10 || error("$path: truncated/extra surface rows or columns")
    all(x->x isa Real && isfinite(x),M) || error("$path: nonfinite/missing triangle coordinate")
    length(unique(M[:,1]))==n || error("$path: duplicate surface IDs")
    all(b->length(b)==2 && all(isfinite,b) && b[2]>b[1],box) || error("$path: invalid surface box")
    V0 = Array{Float64}(undef, 3, n); E1 = similar(V0); E2 = similar(V0)
    LO = similar(V0); HI = similar(V0)
    for t in 1:n
        # SPARTA (x,y,z) -> tracer (y,z,x)
        a = (M[t, 3], M[t, 4], M[t, 2])
        b = (M[t, 6], M[t, 7], M[t, 5])
        c = (M[t, 9], M[t, 10], M[t, 8])
        for d in 1:3
            V0[d, t] = a[d]
            E1[d, t] = b[d] - a[d]
            E2[d, t] = c[d] - a[d]
            LO[d, t] = min(a[d], b[d], c[d])
            HI[d, t] = max(a[d], b[d], c[d])
        end
        norm(cross(view(E1,:,t),view(E2,:,t)))>0 || error("$path: degenerate triangle $t")
    end
    # tracer-axis box: SPARTA rows are x, y, z -> tracer 1,2,3 = y, z, x
    tbox = [box[2], box[3], box[1]]
    return V0, E1, E2, LO, HI, tbox, maximum(HI[AXIS, :])
end

"""
    read_field3d(path) -> (C, F, box, base, nlev)

`C` is 6xN (cx cy cz hx hy hz) and `F` is 5xN (vgx vgy vgz T rho), both in
TRACER axes.  `base` is the base-grid cell count per tracer axis.
"""
function read_field3d(path)
    h = _hdr(path, 11)
    (h[1] == "ITEM: TIMESTEP" && h[9] == "ITEM: BASE GRID" &&
     startswith(h[11], "ITEM: CELLS")) || error("$path: not a DS3FF field file")
    sbox = [parse.(Float64, split(h[k])) for k in 6:8]      # SPARTA x, y, z
    sbase = parse.(Int, split(h[10]))
    df = CSV.read(path, DataFrame; header = false, skipto = 12,
                  ignorerepeated = true, delim = ' ')
    M = Matrix(df)      # xc yc zc hx temp nrho u v w   (SPARTA)
    n = size(M, 1)
    box = [sbox[2], sbox[3], sbox[1]]
    base = [sbase[2], sbase[3], sbase[1]]
    h0 = [0.5 * (box[d][2] - box[d][1]) / base[d] for d in 1:3]
    # SPARTA refines 2/2/2, so one level fixes all three half-sizes; hx is the
    # only one written and the converter asserts the other two follow.
    C = Array{Float64}(undef, 6, n)
    F = Array{Float64}(undef, 5, n)
    lev = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        L = round(Int, log2(h0[3] / M[i, 4]))
        lev[i] = L
        s = 2.0^(-L)
        C[1, i] = M[i, 2]; C[2, i] = M[i, 3]; C[3, i] = M[i, 1]
        C[4, i] = h0[1] * s; C[5, i] = h0[2] * s; C[6, i] = M[i, 4]
        F[1, i] = M[i, 8]; F[2, i] = M[i, 9]; F[3, i] = M[i, 7]
        F[4, i] = M[i, 5]; F[5, i] = M[i, 6]
    end
    return C, F, box, base, lev, h0
end

# Exact split-cell lookup is optional: the public release omits SplitField3D.jl
# (a translation of SPARTA GPL-2 routines). Without it, a field carrying
# converter split-map artifacts is refused, and duplicate gas rows in one box
# still fail at the lattice check in prepare_field3d.
if isfile(joinpath(@__DIR__,"SplitField3D.jl"))
    include(joinpath(@__DIR__,"SplitField3D.jl"))
else
    function load_split_field3d(grid_path, C)
        for name in ("split-cell-map.tsv","split-field-rows.tsv","split-parent-ids.txt",
                     "split-parent-ids.txt.json","split-field-manifest.toml")
            isfile(joinpath(dirname(grid_path),name)) &&
                error("$(joinpath(dirname(grid_path),name)) needs exact split-cell support, " *
                      "but tracer/SplitField3D.jl is not installed")
        end
        return Any[],zeros(Int32,size(C,2))
    end
end

# ------------------------------------------------------- triangle bin grid

struct BinGrid
    lo::NTuple{3,Float64}
    inv::NTuple{3,Float64}
    n::NTuple{3,Int}
    start::Vector{Int32}      # CSR offsets, length prod(n)+1
    items::Vector{Int32}
end

function BinGrid(LO, HI; target = 0.002, nmax = 64)
    lo = ntuple(d -> minimum(@view LO[d, :]), 3)
    hi = ntuple(d -> maximum(@view HI[d, :]), 3)
    n = ntuple(d -> clamp(ceil(Int, (hi[d] - lo[d]) / target), 1, nmax), 3)
    inv = ntuple(d -> n[d] / max(hi[d] - lo[d], eps()), 3)
    nb = prod(n)
    rng(t) = ntuple(d -> (clamp(floor(Int, (LO[d, t] - lo[d]) * inv[d]), 0, n[d] - 1),
                          clamp(floor(Int, (HI[d, t] - lo[d]) * inv[d]), 0, n[d] - 1)), 3)
    counts = zeros(Int32, nb)
    for t in 1:size(LO, 2)
        (a, b, c) = rng(t)
        for k in c[1]:c[2], j in b[1]:b[2], i in a[1]:a[2]
            counts[1+(k*n[2]+j)*n[1]+i] += 1
        end
    end
    start = Vector{Int32}(undef, nb + 1)
    start[1] = 1
    for b in 1:nb
        start[b+1] = start[b] + counts[b]
    end
    items = Vector{Int32}(undef, start[end] - 1)
    pos = copy(start)
    for t in 1:size(LO, 2)
        (a, b, c) = rng(t)
        for k in c[1]:c[2], j in b[1]:b[2], i in a[1]:a[2]
            p = 1 + (k * n[2] + j) * n[1] + i
            items[pos[p]] = t
            pos[p] += 1
        end
    end
    return BinGrid(lo, inv, n, start, items)
end

# Moller-Trumbore, restricted to the segment p + t*d with 0 <= t <= 1.
@inline function _seg_tri(px, py, pz, dx, dy, dz, V0, E1, E2, t)
    @inbounds begin
        e1x = E1[1, t]; e1y = E1[2, t]; e1z = E1[3, t]
        e2x = E2[1, t]; e2y = E2[2, t]; e2z = E2[3, t]
        hx = dy * e2z - dz * e2y
        hy = dz * e2x - dx * e2z
        hz = dx * e2y - dy * e2x
        a = e1x * hx + e1y * hy + e1z * hz
        abs(a) < 1e-30 && return false
        f = 1.0 / a
        sx = px - V0[1, t]; sy = py - V0[2, t]; sz = pz - V0[3, t]
        u = f * (sx * hx + sy * hy + sz * hz)
        (u < 0.0 || u > 1.0) && return false
        qx = sy * e1z - sz * e1y
        qy = sz * e1x - sx * e1z
        qz = sx * e1y - sy * e1x
        v = f * (dx * qx + dy * qy + dz * qz)
        (v < 0.0 || u + v > 1.0) && return false
        tt = f * (e2x * qx + e2y * qy + e2z * qz)
        return 0.0 <= tt <= 1.0
    end
end

# ---------------------------------------------------------------- the seam

"""
    make_walls(geomFile, zlo, zhi, rmax) -> (getCollision, max_x_geom, ntri, bins)

The geometry half of the seam, split out so tracer/microtests3d/walls3d.jl can
gate it against the 2D segment test without loading a 2.5-million-cell field.
"""
function make_walls(geomFile, zlo, zhi, rmax)
    V0, E1, E2, LO, HI, _, max_x_geom = read_surfs3d(geomFile)
    grid = BinGrid(LO, HI)
    ntri = size(V0, 2)

    # getCollision(x1, x2): 0 = no crossing, 1 = the leg hits a wall triangle,
    # 2 = it leaves the simulation bounds.  Geometry wins outright, exactly as
    # the 2D closure returns 1 before it ever tests the box.
    @inline function getCollision(p, q)
        @inbounds begin
            px, py, pz = p[1], p[2], p[3]
            dx, dy, dz = q[1] - px, q[2] - py, q[3] - pz
            slo = ntuple(d -> min(p[d], q[d]), 3)
            shi = ntuple(d -> max(p[d], q[d]), 3)
            ia = ntuple(d -> floor(Int, (slo[d] - grid.lo[d]) * grid.inv[d]), 3)
            ib = ntuple(d -> floor(Int, (shi[d] - grid.lo[d]) * grid.inv[d]), 3)
            span = 1
            skip = false
            for d in 1:3
                ib[d] < 0 && (skip = true)
                ia[d] >= grid.n[d] && (skip = true)
                span *= min(ib[d], grid.n[d] - 1) - max(ia[d], 0) + 1
            end
            if !skip
                if span > 256      # a near-vacuum free path can span the box
                    for t in 1:ntri
                        _seg_tri(px, py, pz, dx, dy, dz, V0, E1, E2, t) && return 1
                    end
                else
                    n1, n2 = grid.n[1], grid.n[2]
                    for k in max(ia[3], 0):min(ib[3], grid.n[3] - 1),
                        j in max(ia[2], 0):min(ib[2], n2 - 1),
                        i in max(ia[1], 0):min(ib[1], n1 - 1)
                        b = 1 + (k * n2 + j) * n1 + i
                        for s in grid.start[b]:(grid.start[b+1]-1)
                            t = grid.items[s]
                            (shi[1] < LO[1, t] || slo[1] > HI[1, t] ||
                             shi[2] < LO[2, t] || slo[2] > HI[2, t] ||
                             shi[3] < LO[3, t] || slo[3] > HI[3, t]) && continue
                            _seg_tri(px, py, pz, dx, dy, dz, V0, E1, E2, t) && return 1
                        end
                    end
                end
            end
            if q[3] < zlo || q[3] > zhi || q[1]^2 + q[2]^2 > rmax^2
                return 2
            end
            return 0
        end
    end
    return getCollision, max_x_geom, ntri, grid
end

"""Read and validate one gas field, building its reusable containing-cell lattice.
The returned arrays and index are shared read-only; each bound field below gets
its own mutable lookup counters and interpolation closure.
"""
function prepare_field3d(gridFile; reader=read_field3d,
                         missing_policy=:nearest, validate=false)
    missing_policy in (:error, :nearest) || error("unknown missing-cell policy")
    C, F, box, base, lev, h0 = reader(gridFile)
    validate && validate_field3d(C, F, box, base, lev, h0)
    ncell = size(C, 2)
    split_parents,split_by_row=load_split_field3d(gridFile,C)
    !isempty(split_parents) && missing_policy!=:error &&
        error("exact split-cell fields require strict missing-cell errors")
    nlev = maximum(lev) + 1

    # containing-cell lattice: one Dict per level, keyed on the integer index
    lo1, lo2, lo3 = box[1][1], box[2][1], box[3][1]
    dicts = [Dict{Int64,Int32}() for _ in 1:nlev]
    nL = [ntuple(d -> base[d] << (L - 1), 3) for L in 1:nlev]
    ivL = [ntuple(d -> 2.0^(L - 1) / (2 * h0[d]), 3) for L in 1:nlev]
    @inbounds for i in 1:ncell
        L = lev[i] + 1
        nx, ny, _ = nL[L]
        a = floor(Int, (C[1, i] - lo1) * ivL[L][1])
        b = floor(Int, (C[2, i] - lo2) * ivL[L][2])
        c = floor(Int, (C[3, i] - lo3) * ivL[L][3])
        key = (Int64(c) * ny + b) * nx + a
        if haskey(dicts[L],key)
            prior=Int(dicts[L][key])
            (validate && split_by_row[prior]>0 &&
             split_by_row[i]==split_by_row[prior]) ||
                error("duplicate field lattice cell at row $i without one shared solver split map")
        else
            dicts[L][key] = Int32(i)
        end
    end
    if validate
        for i in 1:ncell, L in 1:lev[i]
            nx,ny,_=nL[L]
            a=floor(Int,(C[1,i]-lo1)*ivL[L][1])
            b=floor(Int,(C[2,i]-lo2)*ivL[L][2])
            c=floor(Int,(C[3,i]-lo3)*ivL[L][3])
            haskey(dicts[L],(Int64(c)*ny+b)*nx+a) && error("overlapping ancestor/descendant field leaves at row $i")
        end
    end

    kdtree = missing_policy == :nearest ? KDTree(C[1:3, :]; leafsize = 16) : nothing
    Tmin, Tmax = extrema(@view F[4, :])
    Umax = 1.5 * sqrt(maximum(F[1, i]^2 + F[2, i]^2 + F[3, i]^2 for i in 1:ncell))
    (; C,F,box,base,lev,h0,ncell,nlev,lo1,lo2,lo3,dicts,nL,ivL,kdtree,
       Tmin,Tmax,Umax,split_parents,split_by_row,
       grid_path=abspath(gridFile),missing_policy,validated=validate)
end

function build_field3d(geomFile, gridFile; reader=read_field3d,
                       build_table=true, missing_policy=:nearest,
                       validate=false, geometry_mode=:legacy, prepared=nothing)
    missing_policy in (:error, :nearest) || error("unknown missing-cell policy")
    geometry_mode in (:legacy,:explicit) || error("unknown 3D geometry mode")
    data = isnothing(prepared) ? prepare_field3d(gridFile;reader,missing_policy,validate) : prepared
    if !isnothing(prepared)
        data.grid_path==abspath(gridFile) || error("prepared field belongs to another grid path")
        data.missing_policy==missing_policy || error("prepared field has a different missing-cell policy")
        validate && !data.validated && error("prepared field was not strictly validated")
    end
    C,F,box=data.C,data.F,data.box
    ncell,nlev=data.ncell,data.nlev
    lo1,lo2,lo3=data.lo1,data.lo2,data.lo3
    dicts,nL,ivL,kdtree=data.dicts,data.nL,data.ivL,data.kdtree
    split_parents,split_by_row=data.split_parents,data.split_by_row
    Tmin,Tmax,Umax=data.Tmin,data.Tmax,data.Umax
    misses = Threads.Atomic{Int}(0)
    lookups = Threads.Atomic{Int}(0)

    if geometry_mode==:legacy
        zlo, zhi = box[3][1], box[3][2]
        rmax = min(-box[1][1], box[1][2], -box[2][1], box[2][2])
        # The escape surface is inherited from the FIELD's box, so it is not the
        # same surface in every gate: the revolved 2D field is z <= 0.0926,
        # r <= 0.0195 (matching the 2D domain) while the real b5-3d-hires field is
        # z <= 0.115, r <= 0.025 (open-issues #9). These two variables force it.
        zhi = parse(Float64, get(ENV, "TRACER3D_ZHI", string(zhi)))
        rmax = parse(Float64, get(ENV, "TRACER3D_RMAX", string(rmax)))
        isfinite(zhi) && zhi>zlo && isfinite(rmax) && rmax>0 || error("invalid escape cylinder bounds")
        validate && (zhi>box[3][2] || rmax>minimum((box[1][2],-box[1][1],box[2][2],-box[2][1]))) &&
            error("escape cylinder exceeds the gas-field box")
        bounds = [zlo zhi; 0.0 rmax]
        getCollision, max_x_geom, ntri, grid = make_walls(geomFile, zlo, zhi, rmax)
    else
        # Explicit contact geometry is supplied separately with canonical roles.
        # Never infer an inscribed cylinder for translated/noncircular boxes.
        bounds = nothing
        getCollision = (p,q)->error("explicit field requires a role-aware contact callback")
        max_x_geom = NaN
        ntri = 0
        grid = nothing
    end

    Tstep = (Tmax - Tmin) / 20
    table = build_table ? generate_lookup_table(Tmin, Tstep, Tmax, 0.0, Umax / 20, Umax) : nothing
    if geometry_mode==:legacy
        @printf(stderr, "Field3D: %d cells, %d levels, %d triangles, bins %s; T %.4g-%.4g K, Umax %.4g m/s; escape surface z [%g, %g], r <= %g\n",
                ncell, nlev, ntri, string(grid.n), Tmin, Tmax, Umax, zlo, zhi, rmax)
    else
        @printf(stderr, "Field3D: %d cells, %d levels; explicit contact geometry; T %.4g-%.4g K, Umax %.4g m/s\n",
                ncell, nlev, Tmin, Tmax, Umax)
    end

    @inline function findcell(x1, x2, x3)
        @inbounds for L in nlev:-1:1
            nx, ny, nz = nL[L]
            iv = ivL[L]
            a = floor(Int, (x1 - lo1) * iv[1])
            (a < 0 || a >= nx) && continue
            b = floor(Int, (x2 - lo2) * iv[2])
            (b < 0 || b >= ny) && continue
            c = floor(Int, (x3 - lo3) * iv[3])
            (c < 0 || c >= nz) && continue
            r = get(dicts[L], (Int64(c) * ny + b) * nx + a, Int32(0))
            if r != 0
                parent=split_by_row[r]
                return parent==0 ? r :
                    split_field_row(split_parents[parent],(x1,x2,x3))
            end
        end
        return Int32(0)
    end

    """
        interpolate!(props, x)

    props layout is the vendored one -- propagate reads props[3:5] as the gas
    velocity, props[6] as T and props[7] as the number density.  props[1] is
    reused here as the current cell row (the 2D code stores the nearest cell
    centre there and, since vendored diff 15, looks it up afresh on every
    call); the hit test here is exact cell containment, so a cached row is
    only reused while the point is still inside that cell.
    """
    @inline function interpolate!(props, x)
        @inbounds begin
            r = unsafe_trunc(Int, props[1])
            if r > 0 && split_by_row[r]==0 &&
                        abs(x[1] - C[1, r]) <= C[4, r] &&
                        abs(x[2] - C[2, r]) <= C[5, r] &&
                        abs(x[3] - C[3, r]) <= C[6, r]
                return
            end
            Threads.atomic_add!(lookups, 1)
            # Historical drivers still report the shared counter. Maintained
            # runs use the private field counter and do not touch this global.
            validate || Threads.atomic_add!(interps, 1)
            j = findcell(x[1], x[2], x[3])
            if j == 0
                Threads.atomic_add!(misses, 1)
                missing_policy == :error && error("no gas field cell contains position $(Tuple(x)); repair/select a complete field, or explicitly select recorded nearest-cell compatibility")
                j = Int32(knn(kdtree, [x[1], x[2], x[3]], 1)[1][1])
            end
            props[1] = Float64(j)
            props[3] = F[1, j]; props[4] = F[2, j]; props[5] = F[3, j]
            props[6] = F[4, j]; props[7] = F[5, j]
        end
    end

    field3d_misses[] = misses
    # No nearest-cell substitution for spatial-statistics identity.
    cell_index(x) = begin
        j = findcell(x[1], x[2], x[3])
        j > 0 || error("cannot assign a 3D statistic to a missing gas cell at $(Tuple(x))")
        Int(j)
    end
    cell_metadata(j) = ntuple(d -> C[d,j], 6)
    return (; interpolate!, getCollision, table, bounds, max_x_geom,
             cell_index, cell_metadata, misses, lookups, box, ncell)
end
