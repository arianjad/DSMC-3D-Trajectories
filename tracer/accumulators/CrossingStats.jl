# Shared historical and maintained 3D crossing observable, extracted unchanged.
mutable struct Crossing
    recs::Vector{NTuple{12,Float64}}
    nlegs::Int
    nfwd::Int
    nback::Int
    have::Bool
    cx::Float64
    cy::Float64
    vx::Float64
    vy::Float64
    vz::Float64
    tc::Float64
    nc::Float64
    px::Float64
    py::Float64
    pz::Float64
    hasprev::Bool
    mism::Float64
end
Crossing() = Crossing(NTuple{12,Float64}[], 0, 0, 0, false,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, false, 0.0)

# x is the leg's START (the vendored file assigns x .= xnext on the next line),
# v the leg's velocity, lfree its free path, t and ncolls cumulative AFTER it.
@inline function updateStats!(a::Crossing, x, v, t, ncolls, lfree)
    vmag = sqrt(v[1]^2 + v[2]^2 + v[3]^2)
    dt = lfree / vmag
    ex = x[1] + v[1] * dt
    ey = x[2] + v[2] * dt
    ez = x[3] + v[3] * dt
    if a.hasprev                                   # V1: endpoint identity
        m = max(abs(x[1] - a.px), abs(x[2] - a.py), abs(x[3] - a.pz))
        m > a.mism && (a.mism = m)
    end
    a.px, a.py, a.pz, a.hasprev = ex, ey, ez, true
    a.nlegs += 1
    if (x[3] - XOBS) * (ez - XOBS) < 0             # strict straddle of the plane
        if ez > x[3]
            a.nfwd += 1
            if !a.have
                f = (XOBS - x[3]) / (ez - x[3])
                a.have = true
                a.cx = x[1] + f * (ex - x[1])
                a.cy = x[2] + f * (ey - x[2])
                a.vx, a.vy, a.vz = v[1], v[2], v[3]
                a.tc = t - dt + f * dt
                a.nc = ncolls - 1                  # collisions before the crossing
            end
        elseif a.have
            a.nback += 1                           # back-crossing after the first
        end
    end
    return a
end

merge!(a::Crossing, b::Crossing) = begin
    if isempty(b.recs)
        push!(a.recs, (Float64(b.nlegs), Float64(b.nfwd), Float64(b.nback),
                       b.have ? 1.0 : 0.0, b.cx, b.cy,
                       b.vx, b.vy, b.vz, b.tc, b.nc, b.mism))
    else
        append!(a.recs, b.recs)
    end
    a
end
