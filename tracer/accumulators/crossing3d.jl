# Maintained 3D crossing driver. Terminal legs and hook crossings retain their
# established formats; the receipt supplies the launched population and RNG.
length(ARGS)>=4 || error("usage: crossing3d.jl OUTDIR XOBS <tracer arguments>")
const OUTDIR=popfirst!(ARGS)
const XOBS=parse(Float64,popfirst!(ARGS))
isfinite(XOBS) || error("observation plane must be finite")
ispath(OUTDIR) && error("select a new output directory: $OUTDIR already exists")
include(joinpath(@__DIR__,"..","Runtime3D.jl"))
include(joinpath(@__DIR__,"CrossingStats.jl"))
isempty(OPTIONS3D["receipt"]) && (OPTIONS3D["receipt"]=joinpath(OUTDIR,"receipt.toml"))
reserved_outputs=Dict(file=>joinpath(OUTDIR,file) for file in
    ("legs.out","recs.csv","initial-collisions.csv","trajectories.out"))
result,receipt=run3d(args;make_extra=Crossing,write_terminal=false,
    completion_status="transport_complete",extra_output_paths=reserved_outputs)
try
mkpath(OUTDIR)
receipt["observation_plane_tracer_z_m"]=XOBS
receipt["implementation_sha256"]["accumulators/CrossingStats.jl"]=bytes2hex(open(sha256,joinpath(@__DIR__,"CrossingStats.jl")))
receipt["implementation_sha256"]["accumulators/crossing3d.jl"]=bytes2hex(open(sha256,@__FILE__))
acc=result.extra
length(acc.recs)==args["n"] || error("crossing record count differs from launched count")
all(i->Int(acc.recs[i][1])+result.initial_collisions[i]==Int(result.outputs[i,10]),1:args["n"]) || error("hook legs + initial collisions != recorded collisions")
maximum(r[12] for r in acc.recs)<=1e-12 || error("crossing endpoint mismatch exceeds 1e-12 m")
open(joinpath(OUTDIR,"legs.out"),"w") do io
    println(io,"idx x y z xnext ynext znext vx vy vz collides time")
    for i in 1:args["n"]
        row=@view result.outputs[i,:]
        write_leg3d(io,i,view(row,1:3),view(row,4:6),view(row,7:9),Int(row[10]),row[11])
    end
end
open(joinpath(OUTDIR,"recs.csv"),"w") do io
    println(io,"idx,nlegs,nfwd,nback,have,cx,cy,vx,vy,vz,tc,nc,mism")
    for (k,r) in enumerate(acc.recs)
        @printf(io,"%d,%d,%d,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%.17g\n",
            k,Int(r[1]),Int(r[2]),Int(r[3]),Int(r[4]),r[5],r[6],r[7],r[8],r[9],r[10],Int(r[11]),r[12])
    end
end
CSV.write(joinpath(OUTDIR,"initial-collisions.csv"),DataFrame(idx=1:args["n"],initial_collisions=result.initial_collisions))
if args["trajprint"]>0
    open(joinpath(OUTDIR,"trajectories.out"),"w") do io
        println(io,"idx x y z xnext ynext znext vx vy vz collides time")
        foreach(t->write(io,t),result.traces)
    end
end
receipt["crossing_checks"]="passed: endpoint continuity and hook legs + initial collisions = total collisions"
for file in ("legs.out","recs.csv","initial-collisions.csv","trajectories.out")
    path=joinpath(OUTDIR,file)
    isfile(path) && (receipt["artifacts"][file]=artifact3d(path))
end
receipt["terminal_output_sha256"]=receipt["artifacts"]["legs.out"]["sha256"]
receipt["terminal_output_bytes"]=receipt["artifacts"]["legs.out"]["bytes"]
receipt["status"]="complete"
write_receipt3d(OPTIONS3D["receipt"],receipt)
println(stderr,"PASS crossing: endpoint continuity; hook legs + initial collisions = total collisions")
catch err
    receipt["status"]="failed"; receipt["error"]=sprint(showerror,err)
    write_receipt3d(OPTIONS3D["receipt"],receipt)
    rethrow()
end
