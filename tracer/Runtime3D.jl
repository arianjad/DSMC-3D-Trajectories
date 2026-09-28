# Maintained 3D CLI and source factory. Load once, then call run3d(args).
using TOML, SHA
import Base: merge!

function options3d!(argv)
    opts=Dict("flight"=>"local","rate"=>"approximate","flight-step"=>"0.0001",
              "missing-cell"=>"error","receipt"=>"","seed"=>"")
    for name in keys(opts)
        flag="--"*name
        hits=findall(x -> x==flag || startswith(x,flag*"="),argv)
        length(hits)<=1 || error("duplicate $flag")
        isempty(hits) && continue
        i=only(hits)
        if argv[i]==flag
            i<length(argv) && !startswith(argv[i+1],"--") || error("missing value for $flag")
            opts[name]=argv[i+1]; deleteat!(argv,i:i+1)
        else
            opts[name]=split(argv[i],"=";limit=2)[2]; deleteat!(argv,i)
        end
        isempty(opts[name]) && error("missing value for $flag")
    end
    if "--help" in argv || "-h" in argv
        println("3D extensions: --flight local|field-aware; --rate approximate|maxwell; --flight-step METRES; --missing-cell error|nearest; --receipt FILE")
        println("3D default: particle-v1 RNG (thread independent), force-free flights, native-cell Cartesian statistics.")
        println("--vr is a fixed tracer-x drift, not an outward radial drift for extended sources.")
        println("--seed accepts 0..18446744073709551615; 0 creates and records a nonzero random master seed. Historical replay: ParticleTracing3DLegacy.jl.")
    end
    opts
end
const OPTIONS3D=options3d!(ARGS)
include(joinpath(@__DIR__,"vendored.jl"))
# Keep the vendored 2D parser unchanged; maintained 3D accepts the full RNG range.
isempty(OPTIONS3D["seed"]) || (args["seed"]=parse(UInt64,OPTIONS3D["seed"]))
include(joinpath(@__DIR__,"Field3D.jl"))
include(joinpath(@__DIR__,"ValidatedField3D.jl"))
include(joinpath(@__DIR__,"Statistics3D.jl"))
include(joinpath(@__DIR__,"Transport3D.jl"))

function validate_args3d(a,opts)
    a["n"]>0 || error("-n must be positive")
    for k in ("m","M","sigma")
        isfinite(a[k]) && a[k]>0 || error("$k must be finite and positive")
    end
    for k in ("T","spawnsize")
        isfinite(a[k]) && a[k]>=0 || error("$k must be finite and nonnegative")
    end
    for k in ("r","z","vr","vz","spawnrlo","spawnrhi","spawnzlo","spawnzhi")
        isfinite(a[k]) || error("$k must be finite")
    end
    a["omega"]==0 && a["pflip"]==0 || error("maintained 3D transport supports force-free flights only; nonzero omega/pflip need a corrected, validated force model")
    a["sampler"]=="exact" || error("maintained 3D transport supports the direct partner sampler only; legacy table sampler is available in the legacy entrypoint")
    a["seed"] isa Integer && !(a["seed"] isa Bool) && 0<=a["seed"]<=typemax(UInt64) ||
        error("--seed must be an integer in 0..18446744073709551615")
    a["trajprint"]>=0 || error("--trajprint must be nonnegative")
    a["spawnclip"] in (0,1) || error("--spawnclip must be 0 or 1")
    a["spawn"] in ("point","gaussball","uniformball","uniformcell","uniformplane") || error("unknown source mode")
    if a["spawn"] in ("uniformcell","uniformplane")
        0<=a["spawnrlo"]<=a["spawnrhi"] && a["spawnzlo"]<=a["spawnzhi"] || error("invalid source bounds")
    end
    opts["flight"] in ("local","field-aware") || error("--flight must be local or field-aware")
    opts["rate"] in ("approximate","maxwell") || error("--rate must be approximate or maxwell")
    ds=parse(Float64,opts["flight-step"])
    ds>0 && isfinite(ds) || error("--flight-step must be positive and finite")
    opts["missing-cell"] in ("error","nearest") || error("--missing-cell must be error or nearest")
end

function source3d(a)
    # Snapshot the source configuration, so a later trial cannot mutate this one.
    a=copy(a)
    mode=a["spawn"]; thermal=sqrt(kB*a["T"]/a["M"])
    function generate(rng)
        x=if mode=="point"
            [a["r"],0.0,a["z"]]
        elseif mode=="gaussball"
            s=a["spawnsize"]
            [a["r"]+s*randn(rng),s*randn(rng),a["z"]+s*randn(rng)]
        elseif mode=="uniformball"
            u=[randn(rng),randn(rng),randn(rng)]
            u .*= a["spawnsize"]*cbrt(rand(rng))/norm(u)
            [a["r"]+u[1],u[2],a["z"]+u[3]]
        else
            r=sqrt(a["spawnrlo"]^2+rand(rng)*(a["spawnrhi"]^2-a["spawnrlo"]^2))
            z=a["spawnzlo"]+rand(rng)*(a["spawnzhi"]-a["spawnzlo"])
            phi=mode=="uniformplane" ? 0.0 : 2pi*rand(rng)
            [r*cos(phi),r*sin(phi),z]
        end
        x,[a["vr"]+thermal*randn(rng),thermal*randn(rng),a["vz"]+thermal*randn(rng)]
    end
end

function write_receipt3d(path,receipt)
    mkpath(dirname(abspath(path)))
    open(path,"w") do io
        TOML.print(io,receipt;sorted=true)
    end
end

artifact3d(path) = Dict("path"=>abspath(path),"bytes"=>filesize(path),
    "sha256"=>bytes2hex(open(sha256,path)))

function output_identity3d(path)
    p=abspath(path); suffix=String[]
    while !ispath(p)
        pushfirst!(suffix,basename(p))
        parent=dirname(p)
        parent==p && error("cannot resolve output path: $path")
        p=parent
    end
    resolved=joinpath(realpath(p),suffix...)
    Sys.iswindows() ? lowercase(resolved) : resolved
end

function validate_outputs3d(a,receiptpath;extra_output_paths=Dict{String,String}())
    paths=Dict("receipt"=>receiptpath)
    for key in ("stats","exitstats","spawnout")
        a[key]===nothing || (paths[key]=a[key])
    end
    for (key,path) in extra_output_paths
        haskey(paths,key) && error("duplicate output role: $key")
        paths[key]=path
    end
    identities=Dict{String,String}()
    for (key,path) in paths
        isempty(path) && error("empty output path for $key")
        identity=output_identity3d(path)
        # Every role is a file: none may also be another output's directory.
        # Compare components so names such as stats.csv and stats.csv.extra
        # remain valid neighbors. Identities already resolve aliases/case.
        parts=splitpath(identity)
        for (other,role) in identities
            otherparts=splitpath(other)
            n=min(length(parts),length(otherparts))
            parts[1:n]==otherparts[1:n] &&
                error("output paths overlap: $key and $role at $path")
        end
        identities[identity]=key
        (ispath(path) || islink(path)) && error("output already exists: $path; select a new run path")
    end
    paths
end

function run3d(a;opts=OPTIONS3D,make_extra=()->nothing,output=stdout,write_terminal=true,
               completion_status="complete",extra_output_paths=Dict{String,String}())
    a=copy(a); opts=copy(opts)
    validate_args3d(a,opts)
    physics=TransportPhysics3D(mass_particle=a["M"],mass_buffer=a["m"],
        cross_section=a["sigma"],rate_model=Symbol(opts["rate"]))
    seed=a["seed"]==0 ? rand(RandomDevice(),UInt64(1):typemax(UInt64)) : UInt64(a["seed"])
    receiptpath=isempty(opts["receipt"]) ? joinpath("results","tracer3d","runs",string(time_ns()),"receipt.toml") : opts["receipt"]
    output_paths=validate_outputs3d(a,receiptpath;extra_output_paths)
    foreach(path->mkpath(dirname(abspath(path))),values(output_paths))
    receipt=Dict{String,Any}("schema"=>"dsmc-tracer3d-v1","status"=>"started","launched_particles"=>a["n"],
        "master_seed"=>string(seed),"rng_protocol"=>"particle-v1-splitmix64-xoshiro-julia-"*string(VERSION),
        "julia_threads"=>Threads.nthreads(),"flight"=>opts["flight"],"flight_step_m"=>parse(Float64,opts["flight-step"]),
        "collision_rate_model"=>opts["rate"],
        "missing_cell_policy"=>opts["missing-cell"],"tracer_axes"=>"SPARTA(y,z,x)",
        "statistics"=>"native-cell Cartesian; nonterminating leg-start position/velocity, leg-end age; population central moments",
        "exit_statistics"=>"whole nonterminating histories conditional on domain exit, not an exit-plane distribution",
        "terminal_row"=>"unclipped endpoint; time and collision count exclude terminal leg",
        "source_membership"=>a["spawnclip"]==1 ? "reference-ray visibility (requires visible gas region)" : "unchecked by explicit request",
        "collision_partner"=>"flux-weighted Maxwell proposals with inherited finite 8-sigma envelope",
        "near_stationary_initialization"=>"forced initial gas collision when speed < 1e-6 m/s; counted separately",
        "arguments"=>Dict(k=>(k=="seed" ? string(v) : v) for (k,v) in a if v!==nothing))
    write_receipt3d(receiptpath,receipt)
    @printf(stderr,"3D run receipt: %s; master seed %s\n",abspath(receiptpath),string(seed))
    try
        for key in ("geom","flow")
            receipt[key*"_sha256"]=open(sha256,a[key]) |> bytes2hex
        end
        receipt["implementation_sha256"]=Dict(file=>bytes2hex(open(sha256,joinpath(@__DIR__,file)))
            for file in ("Runtime3D.jl","Transport3D.jl","Field3D.jl","ValidatedField3D.jl",
                         "Statistics3D.jl","ParticleTracing.jl","vendored.jl",
                         "experiments/column_reader.jl","experiments/flight_math.jl","experiments/flight_scratch.jl"))
        t=time()
        field=build_field3d(a["geom"],a["flow"];reader=read_validated_field3d,build_table=false,
            missing_policy=Symbol(opts["missing-cell"]),validate=true)
        receipt["field_setup_seconds"]=time()-t
        receipt["escape_z_m"]=vec(field.bounds[1,:]); receipt["escape_radius_m"]=field.bounds[2,2]
        receipt["escape_surface"]="legacy inscribed cylinder; geometry-first terminal classification"
        t=time()
        result=simulate3d(field,a["n"],source3d(a);seed,physics,
            mode=opts["flight"]=="local" ? :local : :field_aware,ds=parse(Float64,opts["flight-step"]),
            collectall=a["stats"]!==nothing,collectexit=a["exitstats"]!==nothing,make_extra,
            spawnclip=a["spawnclip"]==1,spawnref=[a["r"],0.0,a["z"]],trace_count=a["trajprint"])
        receipt["transport_seconds"]=time()-t
        receipt["initial_collisions"]=sum(result.initial_collisions)
        receipt["collisions"]=Int(sum(result.outputs[:,10]))
        receipt["field_searches"]=field.lookups[]; receipt["missing_cell_queries"]=field.misses[]
        receipt["spawn_rejects"]=result.spawn_rejects
        receipt["wall_terminations"]=count(==(1),result.terminations)
        receipt["domain_exits"]=count(==(2),result.terminations)
        for (key,stats) in (("stats",result.allstats),("exitstats",result.exitstats))
            a[key]===nothing || CSV.write(a[key],convert(DataFrame,stats))
        end
        a["spawnout"]===nothing || CSV.write(a["spawnout"],DataFrame(idx=1:a["n"],x=result.spawns[:,1],y=result.spawns[:,2],z=result.spawns[:,3]))
        if write_terminal
            terminal_digest=SHA.SHA2_256_CTX()
            terminal_header="idx x y z xnext ynext znext vx vy vz collides time\n"
            terminal_bytes=write(output,terminal_header)
            SHA.update!(terminal_digest,codeunits(terminal_header))
            for i in 1:a["n"]
                terminal_bytes+=write(output,result.traces[i])
                SHA.update!(terminal_digest,codeunits(result.traces[i]))
                if a["saveall"]!=0 || result.terminations[i]==2
                    row=@view result.outputs[i,:]
                    terminal_bytes+=write_leg3d(output,i,view(row,1:3),view(row,4:6),view(row,7:9),Int(row[10]),row[11];digest=terminal_digest)
                end
            end
            flush(output)
            receipt["terminal_output_sha256"]=bytes2hex(SHA.digest!(terminal_digest))
            receipt["terminal_output_bytes"]=terminal_bytes
        end
        receipt["artifacts"]=Dict(key=>artifact3d(a[key]) for key in ("stats","exitstats","spawnout") if a[key]!==nothing)
        receipt["status"]=completion_status; write_receipt3d(receiptpath,receipt)
        @printf(stderr,"3D: %d particles, %d collisions, %d missing-cell queries, %.3f s transport\n",a["n"],receipt["collisions"],field.misses[],receipt["transport_seconds"])
        return result,receipt
    catch err
        receipt["status"]="failed"; receipt["error"]=sprint(showerror,err)
        write_receipt3d(receiptpath,receipt)
        rethrow()
    end
end
