# Maintained reader validation. Existing experimental/legacy readers remain
# available for reproducing their recorded behavior.
include(joinpath(@__DIR__, "experiments", "column_reader.jl"))

function read_validated_field3d(path)
    h = _hdr(path, 11)
    length(h) == 11 && h[3] == "ITEM: NUMBER OF CELLS" || error("invalid DS3FF cell-count header")
    h[11] == "ITEM: CELLS xc yc zc hx temp nrho u v w" || error("unsupported DS3FF field column names/order")
    declared = tryparse(Int, h[4])
    declared !== nothing && declared > 0 || error("DS3FF must declare a positive cell count")
    C,F,box,base,lev,h0 = read_field3d_columns(path)
    size(C,2) == declared || error("truncated/extra DS3FF rows: declared $declared, parsed $(size(C,2))")
    validate_field3d(C,F,box,base,lev,h0)
    return C,F,box,base,lev,h0
end

function validate_field3d(C,F,box,base,lev,h0)
    all(isfinite,C) && all(isfinite,F) || error("nonfinite gas-field value")
    all(b -> b > 0,base) || error("base-grid dimensions must be positive")
    all(d -> length(box[d])==2 && all(isfinite,box[d]) && box[d][2]>box[d][1],1:3) || error("invalid 3D box bounds")
    size(C,2)>0 || error("empty gas field")
    all(t -> t>=0,@view F[4,:]) || error("negative gas temperature")
    all(n -> n>=0,@view F[5,:]) || error("negative gas density")
    any(n -> n>0,@view F[5,:]) || error("all-zero gas density field; unfilled output is not a usable populated helium field")
    @inbounds for i in axes(C,2)
        L=lev[i]
        0<=L<=30 || error("unsupported grid level $L at row $i")
        for d in 1:3
            half=C[d+3,i]; expected=h0[d]*2.0^(-L)
            half>0 && isapprox(half,expected;rtol=5e-6,atol=0) || error("non-octree cell half-width at row $i")
            width=2expected
            q=(C[d,i]-box[d][1])/width-0.5
            # Historical DS3FF fields used six significant decimal digits.
            # Permit that representation error, capped at 0.5% of a cell width;
            # never silently snap coordinates or accept a shifted half-cell.
            tol=min(0.005width,5e-6*max(abs(C[d,i]),abs(box[d][1]),abs(box[d][2]),width))
            abs(q-round(q))*width <= tol + 32eps(max(abs(C[d,i]),width)) || error("off-lattice cell center at row $i axis $d")
            -tol <= C[d,i]-half-box[d][1] && C[d,i]+half-box[d][2] <= tol || error("field cell outside box at row $i")
        end
    end
    return nothing
end
