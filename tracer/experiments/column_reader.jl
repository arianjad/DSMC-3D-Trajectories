# Opt-in allocation experiment. Parsing and coordinate/refinement rules match
# Field3D.jl; the DataFrame columns cross a function barrier instead of Matrix(df).
function read_field3d_columns(path)
    h = _hdr(path, 11)
    (h[1] == "ITEM: TIMESTEP" && h[9] == "ITEM: BASE GRID" &&
     startswith(h[11], "ITEM: CELLS")) || error("$path: not a DS3FF field file")
    sbox = [parse.(Float64, split(h[k])) for k in 6:8]
    sbase = parse.(Int, split(h[10]))
    df = CSV.read(path, DataFrame; header=false, skipto=12,
                  ignorerepeated=true, delim=' ')
    ncol(df) == 9 || error("column reader requires nine field columns")
    columns = ntuple(i -> df[!, i], 9)
    all(c -> eltype(c) <: Real, columns) || error("field columns must be numeric without missing values")
    return field3d_from_columns(columns, [sbox[2],sbox[3],sbox[1]],
                                [sbase[2],sbase[3],sbase[1]])
end

function field3d_from_columns(columns::Tuple, box, base)
    xc,yc,zc,hx,temp,nrho,u,v,w = columns
    # Keep the refinement vector concrete across this dynamic column dispatch.
    n = length(xc)::Int
    h0 = [0.5*(box[d][2]-box[d][1])/base[d] for d in 1:3]
    C = Array{Float64}(undef,6,n)
    F = Array{Float64}(undef,5,n)
    lev = Vector{Int}(undef,n)
    @inbounds for i in 1:n
        L = round(Int,log2(h0[3]/hx[i]))
        lev[i] = L
        s = 2.0^(-L)
        C[1,i] = yc[i]; C[2,i] = zc[i]; C[3,i] = xc[i]
        C[4,i] = h0[1]*s; C[5,i] = h0[2]*s; C[6,i] = hx[i]
        F[1,i] = v[i]; F[2,i] = w[i]; F[3,i] = u[i]
        F[4,i] = temp[i]; F[5,i] = nrho[i]
    end
    return C,F,box,base,lev,h0
end
