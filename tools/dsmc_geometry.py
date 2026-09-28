"""B5 circle/slit adapters and pre-solver geometry inspection."""
from __future__ import annotations

import hashlib
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
B5 = ROOT / "cases" / "b5-lean"
if str(B5) not in sys.path:
    sys.path.insert(0, str(B5))
import gen_b5_3d as circle  # noqa: E402
import gen_b5_3d_slit as slit  # noqa: E402
import gen_b5_aperture_pair as pair  # noqa: E402


def _identity(path):
    path = Path(path)
    resolved = path.resolve()
    try:
        name = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        name = str(resolved)
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def _surface(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    declared = {}
    section = None
    point_rows, triangle_rows = {}, {}
    for line in lines:
        fields = line.split()
        if not fields or fields[0] == "#":
            continue
        if len(fields) == 2 and fields[1] in ("points", "triangles"):
            declared[fields[1]] = int(fields[0])
            continue
        if fields[0] in ("Points", "Triangles"):
            section = fields[0]
            continue
        if section == "Points":
            if len(fields) != 4 or int(fields[0]) in point_rows:
                raise ValueError(f"{path}: malformed point row")
            point_rows[int(fields[0])] = tuple(float(v) for v in fields[1:])
        elif section == "Triangles":
            if len(fields) != 5 or int(fields[0]) in triangle_rows:
                raise ValueError(f"{path}: malformed triangle row")
            triangle_rows[int(fields[0])] = tuple(int(v) for v in fields[1:])
        else:
            raise ValueError(f"{path}: unexpected surface row")
    if declared != {"points": len(point_rows), "triangles": len(triangle_rows)}:
        raise ValueError(f"{path}: surface row counts differ from header")
    if set(point_rows) != set(range(1, len(point_rows)+1)) or set(triangle_rows) != set(range(1, len(triangle_rows)+1)):
        raise ValueError(f"{path}: point/triangle IDs are not contiguous 1..N")
    points = [point_rows[i] for i in range(1, len(point_rows)+1)]
    triangles = [triangle_rows[i] for i in range(1, len(triangle_rows)+1)]
    if any(not math.isfinite(v) for p in points for v in p):
        raise ValueError(f"{path}: nonfinite coordinate")
    if any(min(t[1:]) < 1 or max(t[1:]) > len(points) for t in triangles):
        raise ValueError(f"{path}: triangle index outside points")
    return points, triangles


def _normal(a, b, c):
    u = tuple(y-x for x, y in zip(a, b))
    v = tuple(y-x for x, y in zip(a, c))
    return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])


def _inspect_surface(path, *, physical):
    points, triangles = _surface(path)
    counts = Counter(t[0] for t in triangles)
    edges = Counter()
    cap_area = 0.0
    cap_normals_positive_x = True
    for typ, a, b, c in triangles:
        a, b, c = (i-1 for i in (a,b,c))
        normal = _normal(points[a], points[b], points[c])
        if math.hypot(*normal) <= 1e-22:
            raise ValueError(f"{path}: degenerate triangle")
        # SPARTA write_surf can duplicate point rows per triangle; compare
        # physical edge coordinates instead of file-local point IDs.
        for edge in ((a,b),(b,c),(c,a)):
            edges[tuple(sorted((points[edge[0]], points[edge[1]])))] += 1
        if physical and typ == 2:
            cap_area += math.hypot(*normal)/2
            cap_normals_positive_x &= normal[0] > 0
    if any(n > 2 for n in edges.values()):
        raise ValueError(f"{path}: nonmanifold edge")
    result = {"points": len(points), "triangles": len(triangles), "triangle_types": dict(sorted(counts.items())),
              "open_edges": sum(n == 1 for n in edges.values()),
              "nonmanifold_edges": sum(n > 2 for n in edges.values())}
    if physical:
        if set(counts) != {1, 2} or not cap_normals_positive_x or cap_area <= 0:
            raise ValueError(f"{path}: missing wall/cap roles or wrong cap normal")
        result.update({"emitter_cap_area_m2": cap_area, "emitter_cap_normal_positive_x": cap_normals_positive_x})
    else:
        if set(counts) != {1}:
            raise ValueError(f"{path}: station surface must use type 1")
    return result, points, triangles


def _mouth_area(points):
    ring = {(y,z) for x,y,z in points if abs(x-circle.XEXIT) < 1e-13 and 1e-7 < math.hypot(y,z) < 0.01}
    if len(ring) != circle.NTHETA:
        raise ValueError(f"aperture mouth: expected {circle.NTHETA} distinct vertices; found {len(ring)}")
    ring = sorted(ring, key=lambda p: math.atan2(p[1],p[0]))
    return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(ring,ring[1:]+ring[:1]))/2)


def prepare_b5(case: dict, out: Path | None) -> dict:
    """Return plan if out is None; otherwise export and inspect actual surfaces."""
    g = case["geometry"]
    if g["stub_preset"] != "s25":
        raise ValueError("geometry.stub_preset: first sampled adapter supports s25 only")
    kind = "circle" if g["adapter"] == "b5-circle" else "slit"
    aspect = g["aspect_vertical_over_horizontal"]
    width, height, _, polygon_area = slit.aperture_dimensions(aspect, "polygon")
    corner = math.hypot(width/2, height/2)
    station_radius = slit.aperture_station_radius(width, height)
    throat_l4 = math.ceil((corner + pair.THROAT_MARGIN)/pair.THROAT_INCREMENT)*pair.THROAT_INCREMENT
    regions = {"throat_l4_radius_m": throat_l4, "throat_buffer_l3_radius_m": throat_l4+1.6e-3,
               "eqthroat_radius_m": station_radius, "aperture_station_radius_m": station_radius}
    cell_name = pair.cell_name("" if kind == "circle" else "_slit", "s25")
    plan = {"cell_name": cell_name, "regions": regions}
    if out is None:
        return plan
    out = Path(out)
    if kind == "circle":
        circle.main("s25", out / "geometry-export")
    else:
        slit.main("s25", out / "geometry-export", aspect, "polygon")
    exported = out / "geometry-export"
    # Both arms use the same station acceptance, including the taller slit.
    pair.write_common_stations(exported, station_radius)
    names = {"physical_boundary": cell_name, "tube_fwd": "tube_fwd3d.surf", "tube_rev": "tube_rev3d.surf",
             "ap_fwd": "ap_fwd3d.surf", "ap_rev": "ap_rev3d.surf"}
    physical_producer = _identity(B5 / ("gen_b5_3d.py" if kind == "circle" else "gen_b5_3d_slit.py"))
    station_producer = _identity(B5 / "gen_b5_aperture_pair.py")
    artifacts = []
    inspections = {}
    for role, name in names.items():
        source = exported / name
        dest = out / name
        source.replace(dest)
        inspection, points, triangles = _inspect_surface(dest, physical=role == "physical_boundary")
        if role != "physical_boundary":
            intended_x = circle.XTUBE if role.startswith("tube") else circle.XAP
            if any(abs(p[0]-intended_x) > 1e-12 for p in points):
                raise ValueError(f"{role}: station is off its declared plane")
            signs = [math.copysign(1, _normal(points[a-1],points[b-1],points[c-1])[0]) for _,a,b,c in triangles]
            expected = -1 if role.endswith("fwd") else 1
            if any(sign != expected for sign in signs):
                raise ValueError(f"{role}: station normal has wrong direction")
        else:
            actual_area = _mouth_area(points)
            if not math.isclose(actual_area, polygon_area, rel_tol=2e-9):
                raise ValueError(f"{kind} aperture area {actual_area} differs from target {polygon_area}")
            inspection["aperture_area_m2"] = actual_area
        artifacts.append({"role": role, "path": name, "sha256": _identity(dest)["sha256"],
                          "producer": physical_producer if role == "physical_boundary" else station_producer})
        inspections[role] = inspection
    exported.rmdir()
    return {"schema_version": "dsmc-geometry-v1", "units": "m", "adapter": g["adapter"],
            "transform": g["transform"], "boundaries": g["boundaries"],
            "gas_reference_points_m": g["gas_reference_points_m"],
            "source_parameters": {"stub_preset": "s25", "aspect_vertical_over_horizontal": aspect,
                                  "area_basis": "polygon", "target_area_m2": polygon_area,
                                  "slit_width_sparta_y_m": width, "slit_height_sparta_z_m": height,
                                  "aperture_corner_radius_m": corner},
            "regions": regions, "observations": case["observables"],
            "artifacts": artifacts, "inspection": inspections,
            "physical_mesh_status": "emitted-before-SPARTA-clip"}
