# Run one provenance-staged explicit-geometry 3D molecule child.
# Python dsmc molecule owns the JSON case/field identity checks and writes the
# narrowly typed transport.toml. This file never edits the parent helium field.
using CSV, DataFrames, LinearAlgebra, NearestNeighbors, Printf, Random, SHA, TOML

const kB = 8314.46  # amu m^2 / (s^2 K), matching ParticleTracing.jl
include(joinpath(@__DIR__,"Field3D.jl"))
include(joinpath(@__DIR__,"ValidatedField3D.jl"))
include(joinpath(@__DIR__,"Statistics3D.jl"))
include(joinpath(@__DIR__,"Transport3D.jl"))
include(joinpath(@__DIR__,"Geometry3D.jl"))
include(joinpath(@__DIR__,"Source3D.jl"))
include(joinpath(@__DIR__,"Events3D.jl"))

function _required(config,key)
    haskey(config,key) || error("transport config lacks $key")
    config[key]
end

function _positive(value,name)
    value isa Real && isfinite(value) && value>0 || error("$name must be positive and finite")
    Float64(value)
end

function _vector3(value,name)
    value isa AbstractVector && length(value)==3 &&
        all(x->x isa Real && isfinite(x),value) || error("$name must be a finite 3-vector")
    Float64.(value)
end

function _domain3(value)
    value isa AbstractVector && length(value)==3 || error("domain_bounds_m needs three axes")
    box=zeros(3,2)
    for i in 1:3
        axis=value[i]
        axis isa AbstractVector && length(axis)==2 &&
            all(x->x isa Real && isfinite(x),axis) && axis[1]<axis[2] ||
            error("domain_bounds_m axis $i is invalid")
        box[i,:].=axis
    end
    box
end

_tracer3(value,name)=_vector3(value,name)[[2,3,1]]

function _membership_spec(config,adapter)
    adapter in ("b5-circle","b5-slit","cad") ||
        error("unsupported prepared-case geometry adapter")
    if adapter!="cad"
        return (;kind="checked-B5-axial-convex-cross-section")
    end
    ids=_required(config,"solid_boundary_ids")
    ids isa AbstractVector && !isempty(ids) &&
        all(id->id isa AbstractString && !isempty(id),ids) &&
        length(unique(ids))==length(ids) ||
        error("CAD solid_boundary_ids must be a nonempty list of distinct boundary names")
    reference=_tracer3(_required(config,"gas_reference_point_m"),"gas_reference_point_m")
    (;kind="closed-solid-seed-component",solid_boundary_ids=String.(ids),
      gas_reference_point=reference)
end

function _gas_membership(geometry,spec)
    spec.kind=="checked-B5-axial-convex-cross-section" &&
        return b5_radial_gas_component3d(geometry)
    ids=Set(spec.solid_boundary_ids)
    all(geometry.actions[t]!=:absorb || geometry.boundary_ids[t] in ids
        for t in eachindex(geometry.actions)) ||
        error("CAD solid_boundary_ids omit an absorbing physical boundary")
    closed_solid_gas_component3d(geometry,spec.solid_boundary_ids,
                                 spec.gas_reference_point)
end

function _write_result(path,record)
    temporary=path*".tmp"
    open(temporary,"w") do io
        TOML.print(io,record;sorted=true)
    end
    mv(temporary,path;force=true)
end

_artifact(path)=Dict("path"=>basename(path),"bytes"=>filesize(path),
                     "sha256"=>bytes2hex(open(sha256,path)))

function _planes(config)
    specs=_required(config,"planes")
    specs isa AbstractVector && !isempty(specs) || error("transport needs observation planes")
    planes=ObservablePlane3D[]
    for spec in specs
        spec isa AbstractDict || error("observation plane is not a table")
        push!(planes,observable_plane3d(_required(spec,"name"),
            _tracer3(_required(spec,"origin_m"),"plane origin"),
            _tracer3(_required(spec,"normal"),"plane normal"),
            _tracer3(_required(spec,"axis_u"),"plane axis_u"),
            _tracer3(_required(spec,"axis_v"),"plane axis_v")))
    end
    length(Set(p.name for p in planes))==length(planes) || error("duplicate observation-plane name")
    planes
end

function _terminal_rows(result)
    rows=DataFrame(particle_id=Int[],fate=String[],time_s=Float64[],
        x_m=Float64[],y_m=Float64[],z_m=Float64[],
        vx_m_s=Float64[],vy_m_s=Float64[],vz_m_s=Float64[],collisions=Int[])
    for id in eachindex(result.terminations)
        row=@view result.outputs[id,:]
        fate=result.terminations[id]==1 ? "wall" :
             result.terminations[id]==2 ? "domain-or-cap-exit" :
             error("molecule $id has no terminal state")
        push!(rows,(id,fate,row[11],row[6],row[4],row[5],
                    row[9],row[7],row[8],Int(row[10])))
    end
    rows
end

# Recorded legs of the first K molecules, in solver coordinates like terminal.csv.
# Transport3D writes every leg including the terminating one, whose end is the
# clipped contact and whose collision count/time equal the terminal row's.
# initial_collisions (1 after a zero-speed spawn collision, else 0) anchors the
# collision ordinals, so a dropped first leg cannot pass for such a spawn.
function _path_rows(result,count)
    rows=DataFrame(particle_id=Int[],leg=Int[],x0_m=Float64[],y0_m=Float64[],z0_m=Float64[],
        x1_m=Float64[],y1_m=Float64[],z1_m=Float64[],
        vx_m_s=Float64[],vy_m_s=Float64[],vz_m_s=Float64[],collisions=Int[],time_s=Float64[],
        initial_collisions=Int[])
    for id in 1:count
        for (leg,line) in enumerate(split(result.traces[id],'\n';keepempty=false))
            w=split(line)
            parse(Int,w[1])==-id || error("trace line of particle $id is labelled $(w[1])")
            a=parse.(Float64,w[2:10])
            push!(rows,(id,leg,a[3],a[1],a[2],a[6],a[4],a[5],a[9],a[7],a[8],
                        parse(Int,w[11]),parse(Float64,w[12]),result.initial_collisions[id]))
        end
    end
    rows
end

"""Prepare one checked geometry and gas lattice for sequential molecule trials."""
function prepare_transport3d(config_path::AbstractString)
    config=TOML.parsefile(abspath(config_path))
    _required(config,"schema_version")=="dsmc-molecule-transport-v1" ||
        error("unsupported transport config schema")
    domain=_domain3(_required(config,"domain_bounds_m"))
    adapter=_required(config,"geometry_adapter")
    membership_spec=_membership_spec(config,adapter)
    field_dir=abspath(_required(config,"field_dir"))
    isdir(field_dir) || error("bound field directory is missing")
    t=time()
    geometry=read_contact_geometry3d(field_dir,domain)
    gas=_gas_membership(geometry,membership_spec)
    field_data=prepare_field3d(joinpath(field_dir,"DS3FF.DAT");
        reader=read_validated_field3d,missing_policy=:error,validate=true)
    all(abs(field_data.box[i][j]-domain[[2,3,1][i],j])<=1e-12
        for i in 1:3 for j in 1:2) || error("field and prepared domain differ")
    (;field_dir,domain,adapter,membership_spec,geometry,gas,field_data,
      preparation_seconds=time()-t)
end

function run_prepared_case3d(config_path::AbstractString; prepared=nothing)
    config_path=abspath(config_path)
    isfile(config_path) || error("transport config is missing: $config_path")
    config=TOML.parsefile(config_path)
    _required(config,"schema_version")=="dsmc-molecule-transport-v1" ||
        error("unsupported transport config schema")
    out=dirname(config_path)
    result_path=joinpath(out,"transport-result.toml")
    (ispath(result_path) || islink(result_path)) && error("transport result already exists")
    for name in ("events.csv","terminal.csv","paths.csv")
        (ispath(joinpath(out,name)) || islink(joinpath(out,name))) &&
            error("transport output already exists: $name")
    end
    nparticles=_required(config,"count")
    nparticles isa Int && nparticles>0 || error("count must be a positive integer")
    trace_count=get(config,"trace_count",0)
    trace_count isa Int && 0<=trace_count<=nparticles ||
        error("trace_count must be an integer from 0 to count")
    seedstr=_required(config,"seed")
    seedstr isa AbstractString && occursin(r"^[0-9]+$",seedstr) ||
        error("seed must be a decimal UInt64 string")
    seed=parse(UInt64,seedstr)
    seed>0 || error("prepared-case seed must be nonzero")
    domain=_domain3(_required(config,"domain_bounds_m"))
    adapter=_required(config,"geometry_adapter")
    membership_spec=_membership_spec(config,adapter)
    flight=_required(config,"flight")
    flight in ("local","field-aware") || error("unsupported flight mode")
    rate=_required(config,"rate")
    rate in ("approximate","maxwell") || error("unsupported collision-rate model")
    ds=_positive(_required(config,"flight_step_m"),"flight_step_m")
    mass=_positive(_required(config,"mass_u"),"mass_u")
    gasmass=_positive(_required(config,"gas_mass_u"),"gas_mass_u")
    sigma=_positive(_required(config,"cross_section_m2"),"cross_section_m2")
    source=Dict{String,Any}(
        "distribution"=>_required(config,"source_distribution"),
        "center_m"=>_vector3(_required(config,"source_center_m"),"source_center_m"),
        "radius_m"=>_required(config,"source_radius_m"),
        "drift_m_s"=>_vector3(_required(config,"source_drift_m_s"),"source_drift_m_s"),
        "temperature_K"=>_required(config,"source_temperature_K"),
        "mass_u"=>mass)
    planes=_planes(config)
    field_dir=abspath(_required(config,"field_dir"))
    isdir(field_dir) || error("bound field directory is missing")
    record=Dict{String,Any}(
        "schema_version"=>"dsmc-molecule-transport-result-v1",
        "status"=>"started","launched_particles"=>nparticles,
        "seed"=>seedstr,"geometry_adapter"=>adapter,
        "membership"=>membership_spec.kind,
        "gas_reconstruction"=>"bound-field-policy-in-parent-receipt",
        "implementation_sha256"=>Dict(name=>bytes2hex(open(sha256,joinpath(@__DIR__,name)))
            for name in ("PreparedCase3D.jl","Field3D.jl","ValidatedField3D.jl",
                         "Statistics3D.jl","Transport3D.jl","Geometry3D.jl",
                         "Source3D.jl","Events3D.jl")))
    _write_result(result_path,record)
    try
        t=time()
        shared=isnothing(prepared) ? prepare_transport3d(config_path) : prepared
        shared.field_dir==field_dir || error("prepared transport belongs to another field")
        shared.adapter==adapter || error("prepared transport has a different geometry adapter")
        shared.domain==domain || error("prepared transport has a different domain")
        shared.membership_spec==membership_spec ||
            error("prepared transport has a different gas membership specification")
        g,gas=shared.geometry,shared.gas
        gas(_tracer3(source["center_m"],"source_center_m")) ||
            error("source center is outside the intended gas component")
        generator=make_source(source,(;gas_contains=gas))
        field=build_field3d(nothing,joinpath(field_dir,"DS3FF.DAT");
            reader=read_validated_field3d,build_table=false,
            missing_policy=:error,validate=true,geometry_mode=:explicit,
            prepared=shared.field_data)
        record["field_setup_seconds"]=time()-t
        record["preparation_reused"]=!isnothing(prepared)
        record["shared_preparation_seconds"]=shared.preparation_seconds
        physics=TransportPhysics3D(mass_particle=mass,mass_buffer=gasmass,
                                   cross_section=sigma,rate_model=Symbol(rate))
        t=time()
        result=simulate3d(field,nparticles,generator;seed,
            mode=flight=="local" ? :local : :field_aware,ds,
            contact=(a,b)->first_contact(g,a,b),planes,physics,
            collectall=false,collectexit=false,threaded=false,trace_count)
        record["transport_seconds"]=time()-t
        events=write_observable_events3d(joinpath(out,"events.csv"),result.events,planes)
        CSV.write(joinpath(out,"terminal.csv"),_terminal_rows(result))
        outputs=["events.csv","terminal.csv"]
        if trace_count>0
            CSV.write(joinpath(out,"paths.csv"),_path_rows(result,trace_count))
            record["recorded_paths"]=trace_count
            push!(outputs,"paths.csv")
        end
        record["event_rows"]=events.rows
        record["source_rejections"]=result.spawn_rejects
        record["wall_terminations"]=count(==(1),result.terminations)
        record["domain_or_cap_exits"]=count(==(2),result.terminations)
        record["collisions"]=sum(Int.(result.outputs[:,10]))
        record["field_searches"]=field.lookups[]
        record["missing_cell_queries"]=field.misses[]
        record["flight"]=flight
        record["collision_rate_model"]=rate
        record["artifacts"]=[_artifact(joinpath(out,name)) for name in outputs]
        record["status"]="complete"
        _write_result(result_path,record)
        record
    catch exc
        record["status"]="failed"
        record["error"]=sprint(showerror,exc)
        _write_result(result_path,record)
        rethrow()
    end
end

if abspath(PROGRAM_FILE)==abspath(@__FILE__)
    isempty(ARGS) && error("usage: julia --project=tracer tracer/PreparedCase3D.jl CONFIG.toml [CONFIG2.toml ...]")
    if length(ARGS)==1
        run_prepared_case3d(only(ARGS))
    else
        shared=prepare_transport3d(first(ARGS))
        for config_path in ARGS
            run_prepared_case3d(config_path;prepared=shared)
        end
    end
end
