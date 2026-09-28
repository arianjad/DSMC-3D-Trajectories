# Opt-in numerical experiments; no change to the production tracer.
using SpecialFunctions: erf

"""Mean relative speed for a drifting Maxwell gas (Takahashi Appendix B).

`thermal` is sqrt(kB*T/m), not the mean thermal speed. All speeds are m/s.
"""
function maxwell_relative_speed(slip, thermal)
    slip >= 0 && thermal >= 0 || throw(DomainError((slip, thermal)))
    thermal == 0 && return slip
    q = slip / thermal
    q < 1e-4 && return thermal * sqrt(8/pi) *
        (1 + q*q/6 - q^4/120 + q^6/1680)
    return thermal * (sqrt(2/pi)*exp(-q*q/2) + (q + 1/q)*erf(q/sqrt(2)))
end

"""Consume one exponential optical depth along a straight flight.

`hazard(x)` returns inverse metres. Midpoint integration uses steps <= ds;
gas discontinuities are resolved only as ds -> 0. A subdivision never causes
a collision by itself. Return (endpoint, travelled distance, hit_wall).
The geometry callback is checked before accepting a gas collision.
"""
function optical_flight(x, direction, tau, hazard, wall, ds; maxsegments=10^7)
    ds > 0 && isfinite(ds) || throw(DomainError(ds))
    tau >= 0 && isfinite(tau) || throw(DomainError(tau))
    travelled = 0.0
    p = copy(x)
    for _ in 1:maxsegments
        end_of_step = x .+ direction .* (travelled + ds)
        wall_step = wall(p, end_of_step) != 0
        available = ds
        if wall_step
            # Keep the quadrature point inside the gas. Sampling beyond a wall
            # could otherwise create a false collision before that wall.
            lo, hi = 0.0, ds
            for _ in 1:40
                mid_distance = (lo+hi)/2
                if wall(p, p .+ direction .* mid_distance) != 0
                    hi = mid_distance
                else
                    lo = mid_distance
                end
            end
            available = lo
        end
        mid = p .+ direction .* (available/2)
        h = hazard(mid)
        h >= 0 && isfinite(h) || throw(DomainError(h, "invalid collision hazard"))
        if wall_step && (h == 0 || tau > h*available)
            return end_of_step, travelled + ds, true
        end
        step = h > 0 ? min(available, tau/h) : available
        endpoint = x .+ direction .* (travelled + step)
        wall(p, endpoint) != 0 && return endpoint, travelled + step, true
        if h > 0 && tau <= h*available
            return endpoint, travelled + step, false
        end
        tau -= h*ds
        travelled += ds
        p .= endpoint
    end
    error("optical_flight exceeded maxsegments; check domain, density and step")
end
