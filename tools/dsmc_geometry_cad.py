#!/usr/bin/env python3
"""Normalize a labelled STEP/STL solid into an inspected pre-clip SPARTA surface.

This is geometry preparation only. It does not invent a helium inlet or a wall
scattering law, and pre-clip triangle IDs are not canonical solver IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from dsmc_mesh_crossings import find_nonincident_crossing


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _rotation(transform: dict) -> tuple[list[list[float]], list[float]]:
    if set(transform) != {"rotation", "translation_m"}:
        raise ValueError("transform needs rotation and translation_m only")
    matrix, translation = transform["rotation"], transform["translation_m"]
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix) or len(translation) != 3:
        raise ValueError("transform must have a 3x3 rotation and 3-vector translation")
    matrix = [[float(x) for x in row] for row in matrix]
    translation = [float(x) for x in translation]
    if any(not math.isfinite(x) for row in matrix for x in row) or any(not math.isfinite(x) for x in translation):
        raise ValueError("transform contains nonfinite values")
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[i][k] * matrix[j][k] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > 1e-8:
                raise ValueError("transform.rotation must be orthonormal")
    a, b, c = matrix
    det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    if abs(det - 1.0) > 1e-8:
        raise ValueError("transform.rotation must preserve handedness")
    return matrix, translation


def _parse_surf(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    declared, section = {}, None
    points, triangles = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) == 2 and fields[1] in ("points", "triangles"):
            declared[fields[1]] = int(fields[0])
            continue
        if fields == ["Points"] or fields == ["Triangles"]:
            section = fields[0]
            continue
        if section == "Points" and len(fields) == 4:
            number = int(fields[0])
            if number in points:
                raise ValueError(f"duplicate surf point {number}")
            points[number] = tuple(float(x) for x in fields[1:])
        elif section == "Triangles" and len(fields) == 4:
            number = int(fields[0])
            if number in triangles:
                raise ValueError(f"duplicate surf triangle {number}")
            triangles[number] = tuple(int(x) for x in fields[1:])
        else:
            raise ValueError(f"malformed untyped SPARTA surf row: {line[:90]}")
    if declared != {"points": len(points), "triangles": len(triangles)}:
        raise ValueError("SPARTA surf header counts do not match rows")
    if set(points) != set(range(1, len(points) + 1)) or set(triangles) != set(range(1, len(triangles) + 1)):
        raise ValueError("SPARTA surf IDs must be contiguous 1..N")
    p = [points[i] for i in range(1, len(points) + 1)]
    t = [triangles[i] for i in range(1, len(triangles) + 1)]
    if any(not math.isfinite(x) for point in p for x in point):
        raise ValueError("SPARTA surf has nonfinite coordinates")
    if any(min(row) < 1 or max(row) > len(p) or len(set(row)) != 3 for row in t):
        raise ValueError("SPARTA surf has invalid triangle vertices")
    return p, t


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _triangle(points, row):
    a, b, c = (points[i-1] for i in row)
    u = tuple(b[i]-a[i] for i in range(3))
    v = tuple(c[i]-a[i] for i in range(3))
    n = _cross(u, v)
    area = math.sqrt(sum(x*x for x in n))/2
    center = tuple((a[i]+b[i]+c[i])/3 for i in range(3))
    return a, n, area, center


def _components(rows: list[tuple[int, int, int]], selected: set[int]) -> int:
    if not selected:
        return 0
    parent = {i: i for i in selected}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    edges = defaultdict(list)
    for i in selected:
        a, b, c = rows[i]
        for edge in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted(edge))].append(i)
    for members in edges.values():
        first = find(members[0])
        for other in members[1:]:
            parent[find(other)] = first
    return len({find(i) for i in selected})


def _inspect(points, triangles, *, crossing_voxel_size_m=None):
    directed, undirected = Counter(), Counter()
    volume = area = 0.0
    for row in triangles:
        a, normal, triangle_area, _ = _triangle(points, row)
        if triangle_area <= 1e-20:
            raise ValueError("degenerate CAD triangle")
        area += triangle_area
        volume += sum(a[j]*normal[j] for j in range(3))/6
        i, j, k = row
        for edge in ((i, j), (j, k), (k, i)):
            directed[edge] += 1
            undirected[tuple(sorted(edge))] += 1
    if any(count != 1 for count in directed.values()) or any(count != 2 for count in undirected.values()):
        raise ValueError("CAD surface is cracked, overlapping, or inconsistently oriented")
    if volume <= 0:
        raise ValueError("CAD solid has reversed normals; expected outward-from-part")
    result = {"points": len(points), "triangles": len(triangles), "signed_solid_volume_m3": volume,
              "area_m2": area, "open_edges": 0, "nonmanifold_edges": 0}
    if crossing_voxel_size_m is not None:
        crossing = find_nonincident_crossing(points, triangles,
                                              voxel_size_m=crossing_voxel_size_m)
        if crossing["triangle_ids"] is not None:
            raise ValueError(f"CAD surface has crossing triangles {crossing['triangle_ids']} "
                             f"near {crossing['point_m']} m")
        result["nonincident_crossing_scan"] = crossing
    return result


def _select_roles(points, triangles, boundaries, required_roles):
    if not boundaries or not isinstance(boundaries, dict):
        raise ValueError("boundaries must explicitly define roles")
    if any(not isinstance(spec, dict) for spec in boundaries.values()):
        raise ValueError("each boundary role must be an object")
    defaults = [name for name, spec in boundaries.items() if spec.get("default") is True]
    if len(defaults) != 1:
        raise ValueError("exactly one default boundary is required")
    if not set(required_roles) <= set(boundaries):
        raise ValueError("missing declared boundary role: " + ", ".join(sorted(set(required_roles)-set(boundaries))))
    types = [spec["surface_type"] for spec in boundaries.values()]
    if any(type(t) is not int or t <= 0 for t in types) or len(set(types)) != len(types):
        raise ValueError("boundary surface_type values must be unique positive integers")
    def valid_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def disk_selector(name, selector):
        required = {"kind", "center_m", "plane_normal_unit", "radius_m", "axial_tolerance_m"}
        if set(selector) != required:
            raise ValueError(f"{name}: centroid_disk selector needs exactly {sorted(required)}")
        center = selector["center_m"]
        normal = selector["plane_normal_unit"]
        if (not isinstance(center, list) or len(center) != 3 or
                not isinstance(normal, list) or len(normal) != 3 or
                any(not valid_number(x) for x in center + normal)):
            raise ValueError(f"{name}: invalid centroid_disk center_m or plane_normal_unit")
        norm = math.sqrt(sum(x*x for x in normal))
        if abs(norm - 1.0) > 1e-8:
            raise ValueError(f"{name}: centroid_disk plane_normal_unit must have unit length")
        radius, axial_tolerance = selector["radius_m"], selector["axial_tolerance_m"]
        if (not valid_number(radius) or radius <= 0 or
                not valid_number(axial_tolerance) or axial_tolerance < 0):
            raise ValueError(f"{name}: invalid centroid_disk radius_m or axial_tolerance_m")
        return tuple(center), tuple(normal), radius, axial_tolerance

    for name, spec in boundaries.items():
        if spec.get("molecule_action") not in ("absorb", "exit", "ignore", "unspecified"):
            raise ValueError(f"{name}: unsupported molecule action")
        if name == defaults[0]:
            continue
        selector = spec.get("selector")
        if not isinstance(selector, dict) or selector.get("kind") not in ("centroid_box", "centroid_disk"):
            raise ValueError(f"{name}: expected an explicit centroid_box or centroid_disk selector")
    selected_by = {}
    measurements = {}
    for name, spec in boundaries.items():
        if name == defaults[0]:
            continue
        selector = spec["selector"]
        if selector["kind"] == "centroid_box":
            box = selector.get("bounds_m")
            if (not isinstance(box, list) or len(box) != 3 or any(not isinstance(v, list) or len(v) != 2 or
                    any(not valid_number(x) for x in v) or not v[0] < v[1] for v in box)):
                raise ValueError(f"{name}: invalid selector bounds")
        else:
            disk_center, disk_normal, disk_radius, axial_tolerance = disk_selector(name, selector)
        chosen, area = set(), 0.0
        for i, row in enumerate(triangles):
            origin, triangle_normal, triangle_area, center = _triangle(points, row)
            matches = False
            if selector["kind"] == "centroid_box":
                matches = all(box[axis][0] <= center[axis] <= box[axis][1] for axis in range(3))
            else:
                displacement = tuple(center[axis] - disk_center[axis] for axis in range(3))
                axial = sum(displacement[axis] * disk_normal[axis] for axis in range(3))
                radial_squared = max(0.0, sum(x*x for x in displacement) - axial*axial)
                normal_length = 2.0 * triangle_area
                vertices = (origin, *(points[index-1] for index in row[1:]))
                vertex_offsets = [sum((point[axis] - disk_center[axis]) * disk_normal[axis]
                                      for axis in range(3)) for point in vertices]
                vertex_in_disk = [
                    max(0.0, sum((point[axis] - disk_center[axis])**2 for axis in range(3)) -
                        offset*offset) <= disk_radius*disk_radius
                    for point, offset in zip(vertices, vertex_offsets)
                ]
                alignment = (abs(sum(triangle_normal[axis] * disk_normal[axis] for axis in range(3))) /
                             normal_length) if normal_length > 0 else 0.0
                # All vertices must lie in the selected plane band.  The fixed
                # alignment threshold rejects a nearby cylindrical wall even if
                # its centroid happens to fall inside the disk.
                matches = (abs(axial) <= axial_tolerance and radial_squared <= disk_radius*disk_radius and
                           all(abs(offset) <= axial_tolerance for offset in vertex_offsets) and
                           all(vertex_in_disk) and
                           alignment >= 1.0 - 1e-8)
            if matches:
                if i in selected_by:
                    raise ValueError(f"triangle {i+1} matches overlapping boundary selectors")
                selected_by[i] = name
                chosen.add(i)
                area += triangle_area
        if not chosen:
            raise ValueError(f"{name}: selector matched no triangles")
        expected, tolerance = spec["expected_area_m2"], spec["area_rel_tol"]
        if (type(expected) not in (int, float) or not math.isfinite(expected) or expected <= 0 or
                type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError(f"{name}: invalid area expectation")
        if abs(area/expected - 1) > tolerance:
            raise ValueError(f"{name}: matched area {area} differs from expected {expected}")
        components = _components(triangles, chosen)
        if components != spec["expected_components"]:
            raise ValueError(f"{name}: {components} components, expected {spec['expected_components']}")
        measurements[name] = {"triangles": len(chosen), "area_m2": area, "components": components}
    assigned = [selected_by.get(i, defaults[0]) for i in range(len(triangles))]
    measurements[defaults[0]] = {"triangles": assigned.count(defaults[0])}
    return assigned, measurements


def _check_inlet_normals(points, triangles, assigned, inlet):
    """Check that every numerical emitter face points into its declared gas side."""
    expected = inlet["normal_unit"]
    if (not isinstance(expected, list) or len(expected) != 3 or
            any(type(value) not in (int, float) or not math.isfinite(value)
                for value in expected) or
            abs(sum(value*value for value in expected) - 1) > 1e-8):
        raise ValueError("helium_inlet.normal_unit must be a finite unit vector")
    cosines = []
    for row, role in zip(triangles, assigned):
        if role == inlet["boundary_id"]:
            _, normal, area, _ = _triangle(points, row)
            cosines.append(sum(normal[i]*expected[i] for i in range(3))/(2*area))
    if not cosines or min(cosines) < 1-1e-8:
        raise ValueError("helium inlet cap triangles do not face the declared gas direction")
    return min(cosines)


def _mesh_with_gmsh(source: Path, stl: Path, size: float) -> str:
    try:
        import gmsh
    except (ImportError, OSError) as exc:
        raise RuntimeError("Gmsh is optional for B5 but required for STEP/STL CAD normalization") from exc
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("cad_import")
        if source.suffix.lower() in (".step", ".stp"):
            imported = gmsh.model.occ.importShapes(str(source))
            gmsh.model.occ.synchronize()
            if len([1 for dim, _ in imported if dim == 3]) != 1:
                raise ValueError("solid-part STEP must import as exactly one solid")
            gmsh.option.setNumber("Mesh.MeshSizeMin", size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", size)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.model.mesh.generate(2)
        elif source.suffix.lower() == ".stl":
            gmsh.merge(str(source))
        else:
            raise ValueError("CAD source must be STEP or STL")
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.StlOneSolidPerSurface", 0)
        gmsh.write(str(stl))
        return getattr(gmsh, "__version__", "unknown")
    finally:
        gmsh.finalize()


def prepare_cad(source: Path, unit_scale: float, transform: dict, boundaries: dict,
                out: Path, *, mesh_size_source_units: float = 1.0,
                converter: Path | None = None, expected: dict | None = None,
                required_roles: tuple[str, ...] = (), gas_reference_points_m: list | None = None,
                source_case: Path | None = None, solver_probe: dict | None = None,
                source_unit: str | None = None, provenance: str | None = None,
                helium_inlet: dict | None = None, case_id: str | None = None) -> dict:
    """Prepare an immutable, labelled pre-clip CAD derivative in a fresh dir."""
    source, out = Path(source).resolve(), Path(out).resolve()
    if not boundaries or not isinstance(boundaries, dict):
        raise ValueError("boundaries must explicitly define roles")
    if not source.is_file():
        raise ValueError(f"CAD source file missing: {source}")
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"CAD output already exists: {out}")
    if type(unit_scale) not in (int, float) or not math.isfinite(unit_scale) or unit_scale <= 0:
        raise ValueError("CAD unit_scale must be positive and finite")
    if (type(mesh_size_source_units) not in (int, float) or
            not math.isfinite(mesh_size_source_units) or mesh_size_source_units <= 0):
        raise ValueError("CAD mesh size must be positive and finite")
    rotation, translation = _rotation(transform)
    if not set(required_roles) <= set(boundaries):
        raise ValueError("missing declared boundary role: " + ", ".join(sorted(set(required_roles)-set(boundaries))))
    if helium_inlet is not None:
        if not isinstance(helium_inlet, dict) or set(helium_inlet) != {"boundary_id", "normal_unit"}:
            raise ValueError("helium_inlet needs boundary_id and normal_unit")
        inlet_id = helium_inlet["boundary_id"]
        if (not isinstance(inlet_id, str) or inlet_id not in required_roles or
                boundaries[inlet_id].get("role") != "emitter" or
                boundaries[inlet_id].get("default") is True):
            raise ValueError("helium_inlet must name a required, explicitly selected emitter boundary")
    if case_id is not None and (not isinstance(case_id, str) or not case_id.strip()):
        raise ValueError("case_id must be nonempty text")
    if converter is None:
        root = Path(os.environ.get("SPARTA_SOURCE_ROOT", Path.home() / "src/sparta-27Aug2026"))
        converter = root / "tools/stl2surf.py"
    converter = Path(converter).resolve()
    if not converter.is_file():
        raise ValueError(f"pinned SPARTA stl2surf.py is missing: {converter}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-cad-", dir=out.parent) as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir()
        staged_source = stage / ("source" + source.suffix.lower())
        shutil.copyfile(source, staged_source)  # CAD bytes are binary and immutable
        stl, untyped = stage / "normalized.stl", stage / "untyped.surf"
        gmsh_version = _mesh_with_gmsh(staged_source, stl, mesh_size_source_units)
        command = [sys.executable, str(converter), str(stl), str(untyped)]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
        (stage / "converter.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise ValueError(f"upstream stl2surf.py failed ({result.returncode}): {result.stderr[-500:]}")
        raw_points, triangles = _parse_surf(untyped)
        points = [tuple(sum(rotation[i][j] * (unit_scale*p[j]) for j in range(3)) + translation[i]
                        for i in range(3)) for p in raw_points]
        inspection = _inspect(points, triangles,
                              crossing_voxel_size_m=unit_scale*mesh_size_source_units)
        bbox = [[min(p[i] for p in points), max(p[i] for p in points)] for i in range(3)]
        inspection["bbox_m"] = bbox
        if expected:
            if any(abs(bbox[i][j]-expected["bbox_m"][i][j]) > expected["bbox_abs_tol_m"]
                   for i in range(3) for j in range(2)):
                raise ValueError("CAD bbox differs from declared expectation; check source units/transform")
            if abs(inspection["signed_solid_volume_m3"]/expected["solid_volume_m3"]-1) > expected["volume_rel_tol"]:
                raise ValueError("CAD solid volume differs from declared expectation; check units/orientation")
        assigned, role_measurements = _select_roles(points, triangles, boundaries, required_roles)
        if helium_inlet is not None:
            inspection["helium_inlet_normal_dot_min"] = _check_inlet_normals(
                points, triangles, assigned, helium_inlet)
        tagged = stage / "physical-walls.surf"
        with tagged.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(f"# Labelled CAD part surface; pre-clip, source SHA256 {_sha(staged_source)}\n\n")
            stream.write(f"{len(points)} points\n{len(triangles)} triangles\n\nPoints\n\n")
            for i, point in enumerate(points, 1):
                stream.write(f"{i} " + " ".join(f"{x:.17g}" for x in point) + "\n")
            stream.write("\nTriangles\n\n")
            for i, (row, role) in enumerate(zip(triangles, assigned), 1):
                stream.write(f"{i} {boundaries[role]['surface_type']} {row[0]} {row[1]} {row[2]}\n")
        roles = {"schema_version": "dsmc-triangle-roles-v1", "state": "preclip-not-canonical",
                 "surface_sha256": _sha(tagged), "records": [
                     {"triangle_id": i, "boundary_id": role, "role": boundaries[role]["role"],
                      "molecule_action": boundaries[role]["molecule_action"]}
                     for i, role in enumerate(assigned, 1)]}
        _write_json(stage / "triangle-roles.preclip.json", roles)
        geometry = {"schema_version": "dsmc-geometry-v1", "adapter": "cad-import", "case_id": case_id,
                    "units": "m", "source_unit": source_unit,
                    "source_kind": "solid-part", "solid_components": ["imported_solid"],
                    "source": {"path": staged_source.name,
                    "sha256": _sha(staged_source), "bytes": staged_source.stat().st_size},
                    "source_case": ({"path": str(source_case.resolve()), "sha256": _sha(source_case)}
                                    if source_case is not None else None),
                    "provenance": provenance,
                    "transform": transform, "unit_scale_m_per_source_unit": unit_scale,
                    "mesh_size_source_units": mesh_size_source_units,
                    "required_boundary_ids": list(required_roles),
                    "gas_reference_points_m": gas_reference_points_m,
                    "solver_probe": solver_probe,
                    "transport_readiness": "geometry-only",
                    "helium_inlet": helium_inlet,
                    "boundaries": boundaries, "inspection": {"physical_boundary": inspection,
                    "roles": role_measurements},
                    "physical_mesh_status": "emitted-before-SPARTA-clip",
                    "artifacts": [{"role": "physical_boundary", "path": tagged.name,
                                   "sha256": _sha(tagged)}],
                    "converter": {"path": str(converter), "sha256": _sha(converter),
                                  "argv": ["python", "stl2surf.py", "normalized.stl", "untyped.surf"]},
                    "gmsh_version": gmsh_version}
        _write_json(stage / "geometry.json", geometry)
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"CAD output appeared during preparation: {out}")
        stage.rename(out)
        return geometry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--converter", type=Path)
    args = parser.parse_args(argv)
    case = json.loads(args.case.read_text(encoding="utf-8"))
    if case.get("schema_version") != "dsmc-cad-geometry-v1":
        parser.error("unsupported CAD geometry case schema")
    keys = {"schema_version", "case_id", "source", "source_kind", "source_unit",
            "unit_scale_m_per_source_unit", "mesh_size_source_units", "transform", "expected",
            "boundaries", "required_boundary_ids", "gas_reference_points_m", "helium_inlet",
            "solver_probe", "transport_readiness", "provenance"}
    if set(case) != keys:
        parser.error(f"CAD case missing {sorted(keys - set(case))}; unknown keys {sorted(set(case) - keys)}")
    if case["source_kind"] != "solid-part":
        parser.error("CAD source_kind must be solid-part; gas-volume surfaces need a separate normals convention")
    unit_scales = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254}
    if (type(case["source_unit"]) is not str or case["source_unit"] not in unit_scales or
            case["unit_scale_m_per_source_unit"] != unit_scales[case["source_unit"]]):
        parser.error("CAD source unit and unit_scale_m_per_source_unit disagree")
    if case["transport_readiness"] != "geometry-only":
        parser.error("CAD adapter currently supports geometry-only preparation")
    required = case["required_boundary_ids"]
    if (not isinstance(required, list) or not required or
            any(not isinstance(item, str) or not item for item in required) or
            len(set(required)) != len(required)):
        parser.error("required_boundary_ids must be a nonempty list of unique labels")
    result = prepare_cad(args.case.parent / case["source"], case["unit_scale_m_per_source_unit"],
                         case["transform"], case["boundaries"], args.out,
                         mesh_size_source_units=case["mesh_size_source_units"],
                         converter=args.converter, expected=case["expected"],
                         required_roles=tuple(required),
                         gas_reference_points_m=case["gas_reference_points_m"],
                         source_case=args.case, solver_probe=case["solver_probe"],
                         source_unit=case["source_unit"], provenance=case["provenance"],
                         helium_inlet=case["helium_inlet"], case_id=case["case_id"])
    print(json.dumps({"case_id": case["case_id"], "out": str(args.out.resolve()),
                      "inspection": result["inspection"]}, indent=2))


if __name__ == "__main__":
    main()
