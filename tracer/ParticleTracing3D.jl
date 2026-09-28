# Maintained grouped 3D transport; see README-3d.md for protocols and limitations.
include(joinpath(@__DIR__, "Runtime3D.jl"))
result3d, receipt3d = run3d(args)
