# Cartesian source adapter for the explicit-geometry transport path.
# Case-facing coordinates are SPARTA (x,y,z); the tracer uses (y,z,x).
using Random, LinearAlgebra

struct CartesianSource3D{F,G}
    sample::F
    gas_contains::G
end
(source::CartesianSource3D)(rng)=source.sample(rng)

function _source_vector3d(value,name)
    length(value)==3 && all(x->x isa Real && isfinite(x),value) ||
        error("$name must be a finite three-component solver-frame vector")
    Float64.((value[2],value[3],value[1]))
end

"""Build a point, Gaussian-ball, or uniform-ball source in tracer coordinates.

`geometry.gas_contains(x_tracer)` must test membership in the intended gas
component, including the declared domain. Extended-source rejection is
performed by `simulate3d` and reported there; the point centre is checked now.
`radius_m` is the per-axis standard deviation for the Gaussian source and the
hard sphere radius for the uniform-volume source.
"""
function make_source(spec::AbstractDict,geometry)
    required=Set(("distribution","center_m","radius_m","drift_m_s","temperature_K","mass_u"))
    Set(keys(spec))==required || error("Cartesian source needs exactly $(sort!(collect(required)))")
    hasproperty(geometry,:gas_contains) || error("source geometry lacks gas-component membership")
    contains=getproperty(geometry,:gas_contains)
    applicable(contains,[0.0,0.0,0.0]) || error("gas_contains must accept a tracer-frame point")
    mode=spec["distribution"]
    mode in ("point","gaussian-ball","uniform-ball") || error("unsupported Cartesian source distribution")
    center=_source_vector3d(spec["center_m"],"center_m")
    drift=_source_vector3d(spec["drift_m_s"],"drift_m_s")
    radius=spec["radius_m"]
    temperature=spec["temperature_K"]
    mass=spec["mass_u"]
    all(x->x isa Real && isfinite(x),(radius,temperature,mass)) || error("source scalars must be finite")
    radius>=0 && temperature>=0 && mass>0 || error("invalid source radius, temperature, or mass")
    if mode=="point"
        radius==0 || error("point source radius_m must be zero")
    else
        radius>0 || error("extended source radius_m must be positive")
    end
    mode=="point" && !contains(center) && error("point source is outside the intended gas component")
    thermal=sqrt(kB*temperature/mass)
    sample=rng->begin
        offset=if mode=="point"
            zeros(3)
        elseif mode=="gaussian-ball"
            radius.*randn(rng,3)
        else
            direction=randn(rng,3)
            while norm(direction)==0
                direction=randn(rng,3)
            end
            direction.*(radius*cbrt(rand(rng))/norm(direction))
        end
        center.+offset, drift.+thermal.*randn(rng,3)
    end
    CartesianSource3D(sample,contains)
end
