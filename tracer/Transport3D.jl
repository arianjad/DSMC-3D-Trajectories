# SPDX-License-Identifier: MIT
# Derived collision/flight laws: ParticleTracing.jl (see LICENSING.md).
# Operation baskets own particle IDs; no common physical timestep is imposed.
using Random, LinearAlgebra, SHA
include(joinpath(@__DIR__, "experiments", "flight_math.jl"))
include(joinpath(@__DIR__, "experiments", "flight_scratch.jl"))

"""Immutable run-scoped transport parameters and collision-rate law."""
struct TransportPhysics3D
    mass_particle::Float64
    mass_buffer::Float64
    cross_section::Float64
    rate_model::Symbol
    function TransportPhysics3D(;mass_particle=MASS_PARTICLE,
                                mass_buffer=MASS_BUFFER_GAS,
                                cross_section=σ_BUFFER_GAS_PARTICLE,
                                rate_model=:approximate)
        mp=Float64(mass_particle); mb=Float64(mass_buffer)
        sigma=Float64(cross_section)
        isfinite(mp) && mp>0 || error("particle mass must be finite and positive")
        isfinite(mb) && mb>0 || error("buffer-gas mass must be finite and positive")
        isfinite(sigma) && sigma>0 || error("cross section must be finite and positive")
        rate_model in (:approximate,:maxwell) ||
            error("collision-rate model must be :approximate or :maxwell")
        new(mp,mb,sigma,rate_model)
    end
end

@inline function relative_speed3d(slip2, temperature, physics::TransportPhysics3D)
    physics.rate_model==:approximate ?
        sqrt(8kB*temperature/(physics.mass_buffer*pi)+slip2) :
        maxwell_relative_speed(sqrt(slip2),sqrt(kB*temperature/physics.mass_buffer))
end

# Explicit versioned stream, independent of task scheduling and basket width.
# SplitMix64 permutation is used only to derive each Xoshiro seed.
function particle_rng3d(seed::UInt64, id::Int)
    z = seed + UInt64(id) * 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    Random.Xoshiro(z ⊻ (z >> 31))
end

mutable struct Track3D{S,E}
    id::Int
    rng::Random.Xoshiro
    x::Vector{Float64}
    v::Vector{Float64}
    endpoint::Vector{Float64}
    props::Vector{Float64}
    scratch::FlightScratch
    elapsed::Float64
    collisions::Int
    initial_collisions::Int
    physics::TransportPhysics3D
    stats::S
    extra::E
    trace::Union{Nothing,IOBuffer}
end

# Scalar proposals avoid temporary vectors; the finite 8-sigma rejection
# envelope is inherited, not a claim of mathematically exact sampling.
@inline function partner3d(rng, v, props, physics=TransportPhysics3D())
    s = sqrt(kB * props[6] / physics.mass_buffer)
    w1,w2,w3 = v[1]-props[3],v[2]-props[4],v[3]-props[5]
    gmax = sqrt(w1*w1+w2*w2+w3*w3) + 8s
    isfinite(gmax) || error("nonfinite collision-partner scale; check gas/source values")
    gmax == 0 && return (props[3],props[4],props[5])
    while true
        u1,u2,u3 = s*randn(rng),s*randn(rng),s*randn(rng)
        g = sqrt((w1-u1)^2+(w2-u2)^2+(w3-u3)^2)
        rand(rng)*gmax < g && return (props[3]+u1,props[4]+u2,props[5]+u3)
    end
end

@inline function collide3d!(v, props, rng, physics=TransportPhysics3D())
    u1,u2,u3 = partner3d(rng,v,props,physics)
    c = 2rand(rng)-1; s = sqrt(1-c*c); theta=2pi*rand(rng)
    g = sqrt((v[1]-u1)^2+(v[2]-u2)^2+(v[3]-u3)^2)
    mp,mb=physics.mass_particle,physics.mass_buffer
    v[1] = (mp*v[1]+mb*(u1+g*c))/(mp+mb)
    v[2] = (mp*v[2]+mb*(u2+g*s*cos(theta)))/(mp+mb)
    v[3] = (mp*v[3]+mb*(u3+g*s*sin(theta)))/(mp+mb)
    nothing
end

# Gather only collision work. Geometry/field lookup keeps its scalar handler.
# Each column is one scalar quantity across particles for contiguous scatter
# arithmetic; random proposals stay per-particle and preserve rejection loops.
function collision_basket3d!(tracks, queue, work)
    if length(queue)==1
        p=tracks[only(queue)]
        collide3d!(p.v,p.props,p.rng,p.physics)
        rand(p.rng)
        return nothing
    end
    @inbounds for (j,k) in enumerate(queue)
        p=tracks[k]; v=p.v; u1,u2,u3=partner3d(p.rng,v,p.props,p.physics)
        c=2rand(p.rng)-1; s=sqrt(1-c*c); theta=2pi*rand(p.rng)
        g=sqrt((v[1]-u1)^2+(v[2]-u2)^2+(v[3]-u3)^2)
        work[j,1]=v[1]; work[j,2]=v[2]; work[j,3]=v[3]
        work[j,4]=u1+g*c; work[j,5]=u2+g*s*cos(theta); work[j,6]=u3+g*s*sin(theta)
    end
    @inbounds for d in 1:3
        @simd for j in eachindex(queue)
            physics=tracks[queue[j]].physics
            mp,mb=physics.mass_particle,physics.mass_buffer
            work[j,d]=(mp*work[j,d]+mb*work[j,d+3])/(mp+mb)
        end
    end
    @inbounds for (j,k) in enumerate(queue)
        p=tracks[k]
        p.v[1]=work[j,1]; p.v[2]=work[j,2]; p.v[3]=work[j,3]
        rand(p.rng) # reserved historical zero-probability spin-flip draw
    end
    nothing
end

function start_track3d(id, seed, generate, field, make_stats, make_extra,
                       spawnclip, spawnref, trace_count;
                       physics=TransportPhysics3D(),gas_contains=nothing)
    rng=particle_rng3d(seed,id)
    rejected=0
    x,v=generate(rng)
    while true
        length(x)==length(v)==3 && all(isfinite,x) && all(isfinite,v) || error("invalid source state for particle $id")
        (isnothing(gas_contains) ? (!spawnclip || field.getCollision(spawnref,x)==0) : gas_contains(x)) && break
        rejected+=1
        rejected<=1000 || error(isnothing(gas_contains) ?
            "source rejected 1000 candidates; reference-ray clipping requires a visible gas region" :
            "source rejected 1000 candidates outside the declared gas component")
        x,v=generate(rng)
    end
    p=Track3D(id,rng,Float64.(x),Float64.(v),zeros(3),zeros(8),FlightScratch(),
        0.0,0,0,physics,make_stats(),make_extra(),id<=trace_count ? IOBuffer() : nothing)
    field.interpolate!(p.props,p.x)
    if norm(p.v)<1e-6
        collide3d!(p.v,p.props,p.rng,p.physics)
        p.collisions=1; p.initial_collisions=1
    end
    rand(p.rng) # reserved initial spin draw (force-free model)
    return p,rejected
end

function advance_flight3d!(p, field, mode, ds; contact=nothing, planes=(), events=nothing)
    speed=sqrt(p.v[1]^2+p.v[2]^2+p.v[3]^2)
    isfinite(speed) && speed>0 || error("particle $(p.id) has zero/nonfinite speed after initialization")
    if mode==:local
        field.interpolate!(p.props,p.x)
        local_props=p.props
        slip2=(p.v[1]-local_props[3])^2+(p.v[2]-local_props[4])^2+
              (p.v[3]-local_props[5])^2
        rate=local_props[7]*p.physics.cross_section*
             relative_speed3d(slip2,local_props[6],p.physics)
        isfinite(rate) && rate>=0 || error("invalid collision rate for particle $(p.id)")
        dist=min(-log(rand(p.rng))*(speed/rate),1000.0)
        dt=dist/speed
        @inbounds for d in 1:3
            p.endpoint[d]=p.x[d]+p.v[d]*dt
        end
    else
        hazard(x)=begin
            field.interpolate!(p.props,x)
            hazard_props=p.props
            slip2=(p.v[1]-hazard_props[3])^2+(p.v[2]-hazard_props[4])^2+
                  (p.v[3]-hazard_props[5])^2
            hazard_props[7]*p.physics.cross_section*
                relative_speed3d(slip2,hazard_props[6],p.physics)/speed
        end
        p.scratch.direction .= p.v ./ speed
        collision=isnothing(contact) ? field.getCollision :
            ((a,b)->begin
                hit=contact(a,b)
                hit.kind==:none ? 0 : (hit.kind==:wall ? 1 : 2)
            end)
        endpoint,dist,_=optical_flight!(p.scratch,p.x,p.scratch.direction,
            -log(rand(p.rng)),hazard,collision,ds)
        p.endpoint .= endpoint
    end
    all(isfinite,p.endpoint) || error("nonfinite flight endpoint for particle $(p.id)")
    if !isnothing(contact)
        hit=contact(p.x,p.endpoint)
        p.endpoint .= hit.point
        if !isnothing(events)
            for plane in planes
                append!(events,score_segment(plane,p.x,hit,p.v,p.elapsed))
            end
        end
        if hit.kind!=:none
            p.elapsed+=norm(p.endpoint-p.x)/speed
            isnothing(p.trace) || write_leg3d(p.trace,-p.id,p.x,p.endpoint,p.v,p.collisions,p.elapsed)
            return hit.kind==:wall ? 1 : 2
        end
    else
        terminal=field.getCollision(p.x,p.endpoint)
        terminal!=0 && return terminal
    end
    p.elapsed+=dist/speed
    p.collisions+=1
    isnothing(p.stats) || updateStats!(p.stats,p.x,p.v,p.elapsed,p.collisions,dist)
    isnothing(p.extra) || updateStats!(p.extra,p.x,p.v,p.elapsed,p.collisions,dist)
    isnothing(p.trace) || write_leg3d(p.trace,-p.id,p.x,p.endpoint,p.v,p.collisions,p.elapsed)
    p.x .= p.endpoint
    # Local model deliberately uses START properties; field-aware uses END.
    mode==:local || field.interpolate!(p.props,p.x)
    return 0
end

function write_leg3d(io,id,x,y,v,n,t; digest=nothing)
    line=@sprintf("%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %d %.17g\n",
        id,x[1],x[2],x[3],y[1],y[2],y[3],v[1],v[2],v[3],n,t)
    isnothing(digest) || SHA.update!(digest,codeunits(line))
    write(io,line)
end

"""Grouped force-free test-particle transport. Each particle owns its RNG,
clock and finite terminal state. `alive` controls participation; collision
events do not kill particles. Batch/thread counts are implementation details.
Spatial statistics are opt-in outputs, efficiency improvements are defaults.
"""
function simulate3d(field,n,generate; seed=UInt64(42), mode=:local, ds=1e-4,
                    collectall=false,collectexit=false,make_extra=()->nothing,
                    spawnclip=true,spawnref=[0.0,0.0,0.035],trace_count=0,
                    batchsize=64,threaded=true,physics=TransportPhysics3D(),
                    contact=nothing,gas_contains=nothing,planes=())
    if isnothing(gas_contains) && hasproperty(generate,:gas_contains)
        gas_contains=getproperty(generate,:gas_contains)
    end
    n>0 || error("particle count must be positive")
    batchsize>0 || error("batch size must be positive")
    mode in (:local,:field_aware) || error("unknown flight mode")
    ds>0 && isfinite(ds) || error("flight integration step must be positive and finite")
    isnothing(contact) && !isempty(planes) && error("observation planes require explicit contact geometry")
    !isnothing(contact) && isnothing(gas_contains) &&
        error("explicit contact geometry requires a gas-component membership predicate")
    make_stats=(collectall||collectexit) ? (() -> CartesianStats3D(field.cell_index,field.cell_metadata)) : (() -> nothing)
    outputs=zeros(n,11); terminations=zeros(Int,n); initial=zeros(Int,n); spawns=zeros(n,3)
    traces=fill("",n); rejects=zeros(Int,n)
    events=isempty(planes) ? nothing : [NamedTuple[] for _ in 1:n]
    # Fixed chunk boundaries make statistical reduction independent of threads.
    chunks=collect(Iterators.partition(1:n,256))
    all_total=collectall ? make_stats() : nothing
    exit_total=collectexit ? make_stats() : nothing
    extra_total=make_extra()
    partials=Vector{Tuple{typeof(all_total),typeof(exit_total),typeof(extra_total)}}(undef,length(chunks))
    function run_chunk(ci)
        ids=chunks[ci]; capacity=min(batchsize,length(ids))
        firsttrack,r=start_track3d(first(ids),seed,generate,field,make_stats,make_extra,
            spawnclip,spawnref,trace_count;physics,gas_contains)
        tracks=[firsttrack]
        alive=fill(false,capacity); queue=Int[]; sizehint!(queue,capacity)
        work=zeros(capacity,6); nextid=first(ids)+1
        alive[1]=true; spawns[first(ids),:] .= firsttrack.x; rejects[first(ids)]=r
        stats_by_id=Vector{typeof(firsttrack.stats)}(undef,length(ids))
        extra_by_id=Vector{typeof(firsttrack.extra)}(undef,length(ids))
        while nextid<=last(ids) || any(alive)
            empty!(queue)
            for k in 1:capacity
                if !alive[k] && nextid<=last(ids)
                    p,r=start_track3d(nextid,seed,generate,field,make_stats,make_extra,
                        spawnclip,spawnref,trace_count;physics,gas_contains)
                    k>length(tracks) ? push!(tracks,p) : (tracks[k]=p)
                    spawns[nextid,:] .= p.x; rejects[nextid]=r
                    nextid+=1; alive[k]=true
                end
                alive[k] || continue
                p=tracks[k]; terminal=advance_flight3d!(p,field,mode,ds;
                    contact,planes,events=isnothing(events) ? nothing : events[p.id])
                if terminal==0
                    push!(queue,k)
                else
                    i=p.id
                    outputs[i,:] .= (p.x[1],p.x[2],p.x[3],p.endpoint[1],p.endpoint[2],p.endpoint[3],p.v[1],p.v[2],p.v[3],p.collisions,p.elapsed)
                    terminations[i]=terminal; initial[i]=p.initial_collisions
                    stats_by_id[i-first(ids)+1]=p.stats
                    extra_by_id[i-first(ids)+1]=p.extra
                    isnothing(p.trace) || (traces[i]=String(take!(p.trace)))
                    alive[k]=false
                end
            end
            collision_basket3d!(tracks,queue,work)
        end
        chunk_all=collectall ? make_stats() : nothing
        chunk_exit=collectexit ? make_stats() : nothing
        chunk_extra=make_extra()
        for (j,i) in enumerate(ids)
            collectall && merge!(chunk_all,stats_by_id[j])
            collectexit && terminations[i]==2 && merge!(chunk_exit,stats_by_id[j])
            isnothing(chunk_extra) || merge!(chunk_extra,extra_by_id[j])
        end
        partials[ci]=(chunk_all,chunk_exit,chunk_extra)
    end
    if threaded
        Threads.@threads for ci in eachindex(chunks)
            run_chunk(ci)
        end
    else
        foreach(run_chunk,eachindex(chunks))
    end
    for (pa,pe,px) in partials
        isnothing(all_total) || merge!(all_total,pa)
        isnothing(exit_total) || merge!(exit_total,pe)
        isnothing(extra_total) || merge!(extra_total,px)
    end
    (;outputs,terminations,initial_collisions=initial,spawns,traces,events,
      allstats=all_total,exitstats=exit_total,extra=extra_total,spawn_rejects=sum(rejects))
end
