# Allocation-only variant of flight_math.jl; same arithmetic and callbacks.
# Returned endpoint is borrowed until the next call using this scratch object.
struct FlightScratch
    position::Vector{Float64}
    trial::Vector{Float64}
    midpoint::Vector{Float64}
    endpoint::Vector{Float64}
    direction::Vector{Float64}
    slip::Vector{Float64}
end
FlightScratch() = FlightScratch((zeros(3) for _ in 1:6)...)

function optical_flight!(scratch::FlightScratch, x, direction, tau, hazard, wall, ds; maxsegments=10^7)
    ds > 0 && isfinite(ds) || throw(DomainError(ds))
    tau >= 0 && isfinite(tau) || throw(DomainError(tau))
    travelled = 0.0
    p, end_of_step, mid, endpoint = scratch.position, scratch.trial, scratch.midpoint, scratch.endpoint
    p .= x
    for _ in 1:maxsegments
        end_of_step .= x .+ direction .* (travelled + ds)
        wall_step = wall(p, end_of_step) != 0
        available = ds
        if wall_step
            lo, hi = 0.0, ds
            for _ in 1:40
                mid_distance = (lo+hi)/2
                mid .= p .+ direction .* mid_distance
                if wall(p, mid) != 0
                    hi = mid_distance
                else
                    lo = mid_distance
                end
            end
            available = lo
        end
        mid .= p .+ direction .* (available/2)
        h = hazard(mid)
        h >= 0 && isfinite(h) || throw(DomainError(h, "invalid collision hazard"))
        if wall_step && (h == 0 || tau > h*available)
            return end_of_step, travelled + ds, true
        end
        step = h > 0 ? min(available, tau/h) : available
        endpoint .= x .+ direction .* (travelled + step)
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
