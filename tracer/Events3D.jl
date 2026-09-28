# Versioned observation-event artifact for the explicit-geometry path.
using CSV, DataFrames, LinearAlgebra, SHA

"""Write every directed plane crossing, including terminating-leg arrivals.

Case-facing Cartesian position and velocity columns are in solver (x,y,z)
axes. The local u/v coordinates and velocity projections use the plane basis.
`events_by_particle[id]` comes directly from `simulate3d(...; planes=...)`.
"""
function write_observable_events3d(path,events_by_particle,planes)
    (ispath(path) || islink(path)) && error("observable-event output already exists: $path")
    isempty(planes) && error("observable-event output needs declared planes")
    byname=Dict(p.name=>p for p in planes)
    length(byname)==length(planes) || error("duplicate observation-plane name")
    rows=DataFrame(schema_version=String[],particle_id=Int[],crossing_index=Int[],
        plane=String[],direction=String[],time_s=Float64[],
        x_m=Float64[],y_m=Float64[],z_m=Float64[],
        vx_m_s=Float64[],vy_m_s=Float64[],vz_m_s=Float64[],
        u_m=Float64[],v_m=Float64[],vu_m_s=Float64[],vv_m_s=Float64[],vn_m_s=Float64[])
    for (id,events) in enumerate(events_by_particle)
        for (number,event) in enumerate(events)
            haskey(byname,event.plane) || error("event references undeclared plane")
            event.direction in (:forward,:backward) || error("invalid plane crossing direction")
            plane=byname[event.plane]
            point=event.point
            vu,vv,vn=event.velocity_uvn
            velocity=vu.*plane.axis_u .+ vv.*plane.axis_v .+ vn.*plane.normal
            length(point)==3 && all(isfinite,point) && all(isfinite,velocity) &&
                all(isfinite,event.position_uv) && isfinite(event.time) && event.time>=0 ||
                error("nonfinite or negative-time observation event")
            push!(rows,("dsmc-observable-events-v1",id,number,event.plane,
                String(event.direction),event.time,
                point[3],point[1],point[2],velocity[3],velocity[1],velocity[2],
                event.position_uv[1],event.position_uv[2],vu,vv,vn))
        end
    end
    mkpath(dirname(abspath(path)))
    CSV.write(path,rows)
    (;schema="dsmc-observable-events-v1",rows=nrow(rows),
      bytes=filesize(path),sha256=bytes2hex(open(sha256,path)))
end
