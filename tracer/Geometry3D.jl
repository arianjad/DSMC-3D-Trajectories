# Explicit 3D contact and observation geometry for new prepared cases.
# Coordinates here are tracer (y,z,x)_SPARTA. Legacy Field3D.make_walls remains
# unchanged for replay. Caller must provide canonical, role-checked triangles.
using LinearAlgebra, TOML, SHA, CSV, DataFrames

struct ContactGeometry3D{B}
    V0::Matrix{Float64}
    E1::Matrix{Float64}
    E2::Matrix{Float64}
    LO::Matrix{Float64}
    HI::Matrix{Float64}
    actions::Vector{Symbol}
    boundary_ids::Vector{String}
    domain::Matrix{Float64}
    bins::B
end

"""Free-space test for a closed, oriented *solid-part* surface.

`solid_boundary_ids` must contain only the metal-solid boundary. In particular,
observation planes and gas emitter/exit caps are not part of that boundary.
The selected triangles must pair every directed edge with exactly one reverse
edge after `vertex_tol_m` coordinate quantization. The solid-angle sum is zero
in gas and has unit magnitude in metal. This is not an open-surface or gas
component classifier; the caller must separately establish that the free gas
region has the intended connected component.
"""
struct ClosedSolidFreeSpace3D{G}
    geometry::G
    triangles::Vector{Int}
end

"""One gas component of a closed, oriented solid boundary.

For an embedded, nonintersecting boundary, each disconnected closed surface
contributes its own signed winding number.
The vector of those winding numbers distinguishes a sealed cavity from the
exterior even though both have zero *total* solid winding. The reference point
must lie in the intended gas component and inside the solver domain.
"""
struct ClosedSolidGasComponent3D{G}
    geometry::G
    components::Vector{Vector{Int}}
    signature::Vector{Int8}
end

function closed_solid_free_space3d(g::ContactGeometry3D,solid_boundary_ids;
                                   vertex_tol_m=1e-11)
    isfinite(vertex_tol_m) && vertex_tol_m>0 ||
        error("solid vertex tolerance must be positive and finite")
    solid_boundary_ids isa AbstractString && error("solid boundary IDs must be a collection")
    ids=Set(String.(solid_boundary_ids))
    isempty(ids) && error("solid boundary IDs are empty")
    triangles=findall(id->id in ids,g.boundary_ids)
    isempty(triangles) && error("no triangles match the solid boundary IDs")
    Set(g.boundary_ids[triangles])==ids || error("a solid boundary ID has no triangles")
    all(g.actions[t]==:absorb for t in triangles) ||
        error("solid boundary includes a nonabsorbing triangle")
    vertex(x)=ntuple(i->round(Int64,x[i]/vertex_tol_m),3)
    edges=Dict{Tuple{NTuple{3,Int64},NTuple{3,Int64}},Int}()
    for t in triangles
        a=vertex(view(g.V0,:,t))
        b=vertex(view(g.V0,:,t).+view(g.E1,:,t))
        c=vertex(view(g.V0,:,t).+view(g.E2,:,t))
        length(Set((a,b,c)))==3 || error("solid triangle collapses at vertex tolerance")
        for edge in ((a,b),(b,c),(c,a))
            edges[edge]=get(edges,edge,0)+1
        end
    end
    all(count==1 && get(edges,(b,a),0)==1 for ((a,b),count) in edges) ||
        error("solid boundary is open, nonmanifold, or inconsistently oriented")
    ClosedSolidFreeSpace3D(g,triangles)
end

"""Signed winding number from oriented triangle solid angles.

Uses the van Oosterom-Strackee triangle formula
(doi:10.1109/TBME.1983.325207) and the closed-mesh winding classification of
Jacobson et al. (2013). A nonintegral value is a boundary/geometry ambiguity,
not a probabilistic gas fraction.
"""
function _solid_winding_number3d(g::ContactGeometry3D,triangles,p)
    length(p)==3 && all(isfinite,p) || error("solid membership needs a finite 3D point")
    px,py,pz=Float64.(p)
    total=0.0
    @inbounds for t in triangles
        ax=g.V0[1,t]-px; ay=g.V0[2,t]-py; az=g.V0[3,t]-pz
        bx=ax+g.E1[1,t]; by=ay+g.E1[2,t]; bz=az+g.E1[3,t]
        cx=ax+g.E2[1,t]; cy=ay+g.E2[2,t]; cz=az+g.E2[3,t]
        na=sqrt(ax*ax+ay*ay+az*az)
        nb=sqrt(bx*bx+by*by+bz*bz)
        nc=sqrt(cx*cx+cy*cy+cz*cz)
        min(na,nb,nc)>0 || error("source point lies on a solid vertex")
        det=ax*(by*cz-bz*cy)+ay*(bz*cx-bx*cz)+az*(bx*cy-by*cx)
        den=na*nb*nc+(ax*bx+ay*by+az*bz)*nc+
            (bx*cx+by*cy+bz*cz)*na+(cx*ax+cy*ay+cz*az)*nb
        total+=2atan(det,den)
    end
    total/(4pi)
end

solid_winding_number3d(m::ClosedSolidFreeSpace3D,p)=
    _solid_winding_number3d(m.geometry,m.triangles,p)

function _solid_component_signature3d(g,components,p)
    states=Int8[]
    for triangles in components
        w=_solid_winding_number3d(g,triangles,p)
        if abs(w)<1e-6
            push!(states,0)
        elseif abs(w-1)<1e-6
            push!(states,1)
        elseif abs(w+1)<1e-6
            push!(states,-1)
        else
            error("source point has ambiguous component winding number $w; check mesh and wall proximity")
        end
    end
    states
end

"""Build a gas-component predicate from a closed solid boundary and gas seed.

All selected physical triangles, including any absorbing numerical inlet cap
needed for closure, must be included. Closure and orientation are checked here;
self-intersection checks remain a preparation responsibility. The construction
does not accept an open B5 wall surface or certify a prepared CAD case by itself.
"""
function closed_solid_gas_component3d(g::ContactGeometry3D,solid_boundary_ids,
                                      reference;vertex_tol_m=1e-11)
    solid=closed_solid_free_space3d(g,solid_boundary_ids;vertex_tol_m)
    length(reference)==3 && all(isfinite,reference) &&
        all(g.domain[i,1]<reference[i]<g.domain[i,2] for i in 1:3) ||
        error("gas reference point must be finite and inside the solver domain")
    vertex(x)=ntuple(i->round(Int64,x[i]/vertex_tol_m),3)
    parent=Dict(t=>t for t in solid.triangles)
    function root(t)
        while parent[t]!=t
            parent[t]=parent[parent[t]]
            t=parent[t]
        end
        t
    end
    owners=Dict{Tuple{NTuple{3,Int64},NTuple{3,Int64}},Int}()
    for t in solid.triangles
        a=vertex(view(g.V0,:,t))
        b=vertex(view(g.V0,:,t).+view(g.E1,:,t))
        c=vertex(view(g.V0,:,t).+view(g.E2,:,t))
        for (u,v) in ((a,b),(b,c),(c,a))
            edge=minmax(u,v)
            if haskey(owners,edge)
                parent[root(t)]=root(owners[edge])
            else
                owners[edge]=t
            end
        end
    end
    grouped=Dict{Int,Vector{Int}}()
    for t in solid.triangles
        push!(get!(grouped,root(t),Int[]),t)
    end
    components=sort!(collect(values(grouped));by=first)
    signature=_solid_component_signature3d(g,components,reference)
    sum(signature)==0 || error("gas reference point is inside solid material")
    ClosedSolidGasComponent3D(g,components,signature)
end

function (m::ClosedSolidGasComponent3D)(p)
    length(p)==3 && all(isfinite,p) || error("gas membership needs a finite 3D point")
    all(m.geometry.domain[i,1]<=p[i]<=m.geometry.domain[i,2] for i in 1:3) || return false
    _solid_component_signature3d(m.geometry,m.components,p)==m.signature
end

"""B5-only gas membership for the axial, convex-cross-section tube and plume.

For the checked B5 circle/slit adapters, the beam axis is gas from the inlet
cap through the aperture and each transverse gas cross section is convex.
Thus a transverse segment from the axis to a point stays in the intended gas
component exactly when it crosses no physical wall. This geometry-specific
construction must only be bound to a checked B5 adapter and canonical surface;
it is not an arbitrary-CAD membership rule.
"""
struct B5RadialGasComponent3D{G}
    geometry::G
    cap_x::Float64
end

function b5_radial_gas_component3d(g::ContactGeometry3D;
                                   cap_boundary_id="inlet_cap",wall_boundary_id="wall")
    caps=findall(==(cap_boundary_id),g.boundary_ids)
    walls=findall(==(wall_boundary_id),g.boundary_ids)
    !isempty(caps) && !isempty(walls) || error("B5 gas membership needs wall and inlet-cap triangles")
    all(g.actions[t]==:absorb for t in vcat(caps,walls)) ||
        error("B5 gas membership needs absorbing wall and inlet-cap actions")
    all(g.actions[t]==:ignore || t in caps || t in walls for t in eachindex(g.actions)) ||
        error("B5 gas membership found an undeclared physical boundary")
    xs=vcat([g.V0[3,t] for t in caps],
            [g.V0[3,t]+g.E1[3,t] for t in caps],
            [g.V0[3,t]+g.E2[3,t] for t in caps])
    cap_x=first(xs)
    all(abs(x-cap_x)<1e-12 for x in xs) || error("B5 inlet cap is not planar at one axial coordinate")
    all(g.domain[i,1]<0<g.domain[i,2] for i in 1:2) &&
        g.domain[3,1]<cap_x<g.domain[3,2] ||
        error("B5 axis or inlet cap lies outside the solver domain")
    B5RadialGasComponent3D(g,cap_x)
end

function (m::B5RadialGasComponent3D)(p)
    length(p)==3 && all(isfinite,p) || error("B5 membership needs a finite 3D point")
    all(m.geometry.domain[i,1]<p[i]<m.geometry.domain[i,2] for i in 1:3) || return false
    abs(p[3]-m.cap_x)<1e-12 && error("B5 source point lies on the inlet-cap plane")
    p[3]<m.cap_x && return false
    first_contact(m.geometry,[0.0,0.0,Float64(p[3])],p).kind==:none
end

function (m::ClosedSolidFreeSpace3D)(p)
    length(p)==3 && all(isfinite,p) || error("solid membership needs a finite 3D point")
    all(m.geometry.domain[i,1]<=p[i]<=m.geometry.domain[i,2] for i in 1:3) || return false
    w=abs(solid_winding_number3d(m,p))
    w<1e-6 && return true
    abs(w-1)<1e-6 && return false
    error("source point has ambiguous solid winding number $w; check mesh and wall proximity")
end

"""Load a verified derivative of the canonical solver surface and role JSON.

`field2tracer3d.py --canonical-surface --triangle-roles` writes this bundle.
Its manifest binds the copied solver artifact, original role sidecar, tracer
triangle rows and ordered role table. The converter verifies the JSON's
surface hash and oriented triangle multiset before writing that manifest.
"""
function read_contact_geometry3d(directory,domain_solver)
    manifest=TOML.parsefile(joinpath(directory,"canonical-tracer.toml"))
    get(manifest,"schema_version",nothing)=="dsmc-canonical-tracer3d-v1" ||
        error("unsupported canonical tracer geometry schema")
    files=("canonical_surface_sha256"=>"physical-walls.surf",
           "triangle_roles_sha256"=>"triangle-roles.json",
           "tracer_surface_sha256"=>"cell.surfs",
           "tracer_roles_sha256"=>"triangle-roles.tsv")
    for (key,file) in files
        path=joinpath(directory,file)
        isfile(path) && bytes2hex(open(sha256,path))==get(manifest,key,nothing) ||
            error("canonical tracer artifact hash mismatch: $file")
    end
    V0,E1,E2,LO,HI,box,_=read_surfs3d(joinpath(directory,"cell.surfs"))
    domain=domain_solver[[2,3,1],:]
    all(box[i]==collect(domain[i,:]) for i in 1:3) ||
        error("tracer surface box differs from declared rectangular domain")
    roles=CSV.read(joinpath(directory,"triangle-roles.tsv"),DataFrame;delim='\t')
    names(roles)==["triangle_id","boundary_id","role","molecule_action"] ||
        error("unsupported canonical tracer role table")
    n=size(V0,2)
    get(manifest,"triangle_count",nothing)==n && nrow(roles)==n &&
        roles.triangle_id==collect(1:n) || error("canonical role IDs do not match surface rows")
    actions=Symbol.(roles.molecule_action)
    contact_geometry3d(V0,E1,E2,LO,HI,actions,roles.boundary_id,domain_solver)
end

"""Construct contact geometry from the current `read_surfs3d` triangle arrays.

`actions` are ordered by canonical triangle ID. `domain_solver` is a 3×2 box
in SPARTA (x,y,z) order; it is permuted once into tracer (y,z,x) order.
"""
function contact_geometry3d(V0,E1,E2,LO,HI,actions,boundary_ids,domain_solver)
    n=size(V0,2)
    n>0 && all(size(a)==(3,n) for a in (V0,E1,E2,LO,HI)) ||
        error("contact geometry needs nonempty, matching 3×N triangle arrays")
    length(actions)==length(boundary_ids)==n || error("triangle roles do not cover every canonical triangle")
    all(a->a in (:absorb,:exit,:ignore),actions) || error("unsupported molecule action in canonical roles")
    all(x->!isempty(x), boundary_ids) || error("empty canonical boundary ID")
    all(isfinite, V0) && all(isfinite,E1) && all(isfinite,E2) || error("nonfinite contact triangle")
    all(isfinite, LO) && all(isfinite,HI) || error("nonfinite contact bounds")
    all(LO .<= HI) || error("invalid contact triangle bounds")
    for t in 1:n
        norm(cross(view(E1,:,t),view(E2,:,t)))>0 || error("degenerate contact triangle $t")
    end
    size(domain_solver)==(3,2) && all(isfinite,domain_solver) &&
        all(domain_solver[:,1] .< domain_solver[:,2]) ||
        error("domain_solver needs three finite [low,high] pairs")
    domain=Matrix{Float64}(domain_solver[[2,3,1],:])
    ContactGeometry3D(Matrix{Float64}(V0),Matrix{Float64}(E1),Matrix{Float64}(E2),
        Matrix{Float64}(LO),Matrix{Float64}(HI),Symbol.(actions),String.(boundary_ids),
        domain,BinGrid(LO,HI))
end

@inline function _triangle_fraction3d(g::ContactGeometry3D,p,d,t)
    @inbounds begin
        e1x,e1y,e1z=g.E1[1,t],g.E1[2,t],g.E1[3,t]
        e2x,e2y,e2z=g.E2[1,t],g.E2[2,t],g.E2[3,t]
        hx=d[2]*e2z-d[3]*e2y
        hy=d[3]*e2x-d[1]*e2z
        hz=d[1]*e2y-d[2]*e2x
        det=e1x*hx+e1y*hy+e1z*hz
        abs(det)<1e-30 && return Inf
        invdet=1/det
        sx=p[1]-g.V0[1,t]; sy=p[2]-g.V0[2,t]; sz=p[3]-g.V0[3,t]
        u=(sx*hx+sy*hy+sz*hz)*invdet
        (u<0 || u>1) && return Inf
        qx=sy*e1z-sz*e1y
        qy=sz*e1x-sx*e1z
        qz=sx*e1y-sy*e1x
        v=(d[1]*qx+d[2]*qy+d[3]*qz)*invdet
        (v<0 || u+v>1) && return Inf
        f=(e2x*qx+e2y*qy+e2z*qz)*invdet
        return 0<=f<=1 ? f : Inf
    end
end

function _domain_exit_fraction(g::ContactGeometry3D,p,d)
    t=Inf
    @inbounds for axis in 1:3
        lo,hi=g.domain[axis,1],g.domain[axis,2]
        lo<=p[axis]<=hi || error("leg starts outside declared rectangular domain")
        if d[axis]>0
            t=min(t,(hi-p[axis])/d[axis])
        elseif d[axis]<0
            t=min(t,(lo-p[axis])/d[axis])
        end
    end
    0<=t<=1 ? t : Inf
end

function _candidate_bins3d(g::ContactGeometry3D,p,q)
    grid=g.bins
    lo=ntuple(i->min(p[i],q[i]),3)
    hi=ntuple(i->max(p[i],q[i]),3)
    ia=ntuple(i->floor(Int,(lo[i]-grid.lo[i])*grid.inv[i]),3)
    ib=ntuple(i->floor(Int,(hi[i]-grid.lo[i])*grid.inv[i]),3)
    any(i->ib[i]<0 || ia[i]>=grid.n[i],1:3) && return Int[]
    a=ntuple(i->max(ia[i],0),3)
    b=ntuple(i->min(ib[i],grid.n[i]-1),3)
    span=prod(b[i]-a[i]+1 for i in 1:3)
    span>256 && return 1:size(g.V0,2)
    candidates=Int[]
    n1,n2=grid.n[1],grid.n[2]
    for k in a[3]:b[3], j in a[2]:b[2], i in a[1]:b[1]
        bin=1+(k*n2+j)*n1+i
        for s in grid.start[bin]:(grid.start[bin+1]-1)
            push!(candidates,Int(grid.items[s]))
        end
    end
    candidates
end

"""Return the earliest absorbing wall, exit cap, or rectangular-domain exit.

`kind` is `:wall`, `:cap_exit`, `:domain_exit`, or `:none`; `fraction` is
relative to the proposed full segment. Ignored diagnostic triangles cannot
terminate it. A physical hit at the domain face wins a tie within 1 pm.
Different physical actions at the same contact are ambiguous and fail.
"""
function first_contact(g::ContactGeometry3D,start,finish)
    length(start)==length(finish)==3 && all(isfinite,start) && all(isfinite,finish) ||
        error("contact segment needs finite 3D endpoints")
    p=Float64.(start); q=Float64.(finish); d=q.-p
    dist=norm(d)
    td=_domain_exit_fraction(g,p,d)
    best=isfinite(td) ? td : 1.0
    kind=isfinite(td) ? :domain_exit : :none
    bid=""; tri=0
    tol=1e-12/max(dist,1e-12)
    endscan=p.+best.*d
    for t in _candidate_bins3d(g,p,endscan)
        g.actions[t]==:ignore && continue
        # Bounding-box rejection before the ray/triangle solve.
        any(axis->max(p[axis],endscan[axis])<g.LO[axis,t]-1e-12 ||
                  min(p[axis],endscan[axis])>g.HI[axis,t]+1e-12,1:3) && continue
        f=_triangle_fraction3d(g,p,d,t)
        f>best+tol && continue
        if f<best-tol || kind in (:none,:domain_exit)
            best=f; kind=g.actions[t]==:absorb ? :wall : :cap_exit
            bid=g.boundary_ids[t]; tri=t
        elseif abs(f-best)<=tol
            other=g.actions[t]==:absorb ? :wall : :cap_exit
            other==kind && g.boundary_ids[t]==bid ||
                error("ambiguous physical boundary actions at one contact")
            tri=min(tri,t)
        end
    end
    (;kind,fraction=best,point=p.+best.*d,boundary_id=bid,triangle_id=tri)
end

struct ObservablePlane3D
    name::String
    origin::Vector{Float64}
    normal::Vector{Float64}
    axis_u::Vector{Float64}
    axis_v::Vector{Float64}
end

"""Define a plane in tracer coordinates. Normal × axis_u must equal axis_v."""
function observable_plane3d(name,origin,normal,axis_u,axis_v)
    vectors=(origin,normal,axis_u,axis_v)
    all(v->length(v)==3 && all(isfinite,v),vectors) || error("plane needs four finite 3-vectors")
    n=Float64.(normal); u=Float64.(axis_u); v=Float64.(axis_v)
    all(abs(norm(a)-1)<1e-9 for a in (n,u,v)) || error("plane axes must be unit length")
    abs(dot(n,u))<1e-9 && abs(dot(n,v))<1e-9 && abs(dot(u,v))<1e-9 &&
        norm(cross(n,u)-v)<1e-9 || error("plane axes must be right-handed and orthogonal")
    ObservablePlane3D(String(name),Float64.(origin),n,u,v)
end

"""Score the travelled segment, including its terminal endpoint.

Crossing at the segment start is excluded to avoid double counting a previous
leg's endpoint. Arrival at the terminal contact is included. Coplanar travel
has no crossing. Returned time includes travel from `start` at constant speed.
"""
function score_segment(plane::ObservablePlane3D,start,contact,velocity,start_time)
    p=Float64.(start); q=Float64.(contact.point); v=Float64.(velocity)
    length(p)==length(q)==length(v)==3 && all(isfinite,p) && all(isfinite,q) &&
        all(isfinite,v) && isfinite(start_time) || error("invalid observation segment")
    speed=norm(v)
    speed>0 || error("observation segment has zero speed")
    a=dot(p-plane.origin,plane.normal)
    b=dot(q-plane.origin,plane.normal)
    direction=a<0 && b>=0 ? :forward : (a>0 && b<=0 ? :backward : :none)
    direction==:none && return NamedTuple[]
    f=-a/(b-a)
    x=p.+f.*(q.-p)
    tau=norm(x-p)/speed
    local_position=(dot(x-plane.origin,plane.axis_u),dot(x-plane.origin,plane.axis_v))
    local_velocity=(dot(v,plane.axis_u),dot(v,plane.axis_v),dot(v,plane.normal))
    [(;plane=plane.name,direction,fraction=f,point=x,position_uv=local_position,
       velocity_uvn=local_velocity,time=start_time+tau)]
end
