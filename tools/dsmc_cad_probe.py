#!/usr/bin/env python3
"""Freeze and verify a labelled, closed-solid CAD surface with SPARTA run 0.

Run inside Linux/WSL with --solver pointing to the built SPARTA executable.
The species/mixture declarations only let SPARTA initialize; no particles or
transport steps are run. A successful probe is not a runnable helium case.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from dsmc_geometry import _surface

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def geometry_signature(points, triangles):
    """Compare oriented geometry at 1e-11 m after SPARTA text formatting."""
    def oriented(row):
        a, b, c = (tuple(round(v, 11) for v in points[i-1]) for i in row[1:])
        return (row[0], min((a, b, c), (b, c, a), (c, a, b)))
    return Counter(oriented(row) for row in triangles)


def same_oriented_geometry(old_points, old_tris, new_points, new_tris):
    """Accept SPARTA's text-roundoff without losing the oriented mesh check."""
    if len(old_tris) != len(new_tris):
        return False
    # SPARTA usually retains triangle order but duplicates each vertex on write.
    # Decimal binning can disagree for coordinates only 5e-16 m apart when a
    # value straddles a rounding boundary, so compare ordered faces directly.
    if all(old[0] == new[0] and all(
            abs(old_points[i-1][axis] - new_points[j-1][axis]) <= 1e-11
            for i, j in zip(old[1:], new[1:]) for axis in range(3))
            for old, new in zip(old_tris, new_tris)):
        return True
    return geometry_signature(old_points, old_tris) == geometry_signature(new_points, new_tris)


def deck(surface, box, grid, *, clip, surfmax=1024):
    limits = " ".join(str(v) for axis in box for v in axis)
    count = " ".join(str(n) for n in grid)
    suffix = " clip file physical-walls.surf type" if clip else ""
    return ("dimension 3\nseed 12345\n"
            f"global gridcut 0.0 surfmax {surfmax} mem/limit 1024\n"
            "boundary o o o\n"
            f"create_box {limits}\ncreate_grid {count}\n"
            "species he.species He\nmixture gas He nrho 1.0e22 temp 4\n"
            "global fnum 1.0e18\n"
            f"read_surf {surface} type group cellwalls{suffix}\n"
            "surf_collide walls diffuse 4.0 1.0\n"
            "surf_modify cellwalls collide walls\nrun 0\n")


def one_rank_command(solver, name):
    """Use the site's MPI launch path even for one rank inside a Slurm job."""
    prefix = ["mpirun", "-np", "1"] if os.environ.get("SLURM_JOB_ID") else []
    return [*prefix, str(solver), "-in", name]


def run_probe(out, solver, name):
    command = one_rank_command(solver, name)
    result = subprocess.run(command, cwd=out,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            timeout=120, check=False)
    log = out / (name + ".log")
    log.write_bytes(result.stdout)
    lines = result.stdout.decode(errors="replace").splitlines()
    if result.returncode:
        raise ValueError(f"SPARTA {name} failed ({result.returncode}): {' | '.join(lines[-10:])}")
    text = "\n".join(lines)
    match = re.search(r"(\d+) (\d+) (\d+) = cells outside/inside/overlapping surfs", text)
    volume = re.search(r"([\d.eE+-]+) ([\d.eE+-]+) = cell-wise and global flow volume", text)
    if not match or not volume:
        raise ValueError(f"SPARTA {name}: missing classification or flow volume in {log}")
    counts = [int(v) for v in match.groups()]
    if not all(v > 0 for v in counts):
        raise ValueError(f"SPARTA {name}: expected outside, inside, and overlap cells")
    return {"command": command, "cells_outside_inside_overlapping": counts,
            "flow_volume_m3": float(volume.group(2)), "log": log.name,
            "log_sha256": sha(log)}


def write_tracer_surfs(path, box, points, triangles):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF SURFS\n")
        stream.write(f"{len(triangles)}\nITEM: BOX BOUNDS oo oo oo\n")
        for axis in box:
            stream.write(f"{axis[0]:.17g} {axis[1]:.17g}\n")
        stream.write("ITEM: SURFS id v1x v1y v1z v2x v2y v2z v3x v3y v3z\n")
        for i, (_, a, b, c) in enumerate(triangles, 1):
            xyz = [v for vertex in (points[a-1], points[b-1], points[c-1]) for v in vertex]
            stream.write(f"{i} " + " ".join(f"{v:.17g}" for v in xyz) + "\n")


def check(prepared: Path, solver: Path, out: Path) -> dict:
    prepared, solver, out = prepared.resolve(), solver.resolve(), out.resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"solver check output already exists: {out}")
    if out == prepared or out in prepared.parents or prepared in out.parents:
        raise ValueError("solver check output must be separate from the CAD preparation")
    if not solver.is_file():
        raise ValueError(f"solver executable missing: {solver}")
    geometry = json.loads((prepared / "geometry.json").read_text())
    if geometry["adapter"] != "cad-import" or geometry["transport_readiness"] != "geometry-only":
        raise ValueError("expected a geometry-only CAD preparation")
    probe = geometry["solver_probe"]
    if not probe:
        raise ValueError("CAD preparation lacks an explicit solver_probe")
    box, grid = probe["domain_bounds_m"], probe["base_grid"]
    surfmax = probe.get("surfmax", 1024)
    if type(surfmax) is not int or surfmax <= 0:
        raise ValueError("solver_probe.surfmax must be a positive integer")
    if (not isinstance(box, list) or len(box) != 3 or
            any(not isinstance(axis, list) or len(axis) != 2 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in axis) or
                axis[0] >= axis[1] for axis in box)):
        raise ValueError("solver_probe.domain_bounds_m needs three finite increasing pairs")
    bbox = geometry["inspection"]["physical_boundary"]["bbox_m"]
    if any(not box[i][0] < bbox[i][0] <= bbox[i][1] < box[i][1] for i in range(3)):
        raise ValueError("CAD solid must lie strictly inside the declared probe domain")
    if not isinstance(grid, list) or len(grid) != 3 or any(type(n) is not int or n <= 0 for n in grid):
        raise ValueError("solver_probe.base_grid needs three positive integers")
    tolerance = probe["flow_volume_rel_tol"]
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("solver_probe.flow_volume_rel_tol must be finite and between zero and one")
    if sha(prepared / "physical-walls.surf") != geometry["artifacts"][0]["sha256"]:
        raise ValueError("prepared CAD surface hash mismatch")
    out.mkdir(parents=True)
    try:
        source_name = geometry["source"]["path"]
        staged_source = (prepared / source_name).resolve()
        if staged_source.parent != prepared or not staged_source.is_file():
            raise ValueError("prepared CAD source path must be a file in its preparation")
        if sha(staged_source) != geometry["source"]["sha256"]:
            raise ValueError("prepared CAD source hash mismatch")
        shutil.copyfile(staged_source, out / staged_source.name)
        shutil.copyfile(prepared / "physical-walls.surf", out / "preclip.surf")
        species = ROOT / "cases/data/he.species"
        shutil.copyfile(species, out / "he.species")
        (out / "in.clip").write_text(deck("preclip.surf", box, grid, clip=True,
                                          surfmax=surfmax), encoding="utf-8")
        first = run_probe(out, solver, "in.clip")
        canonical = out / "physical-walls.surf"
        if not canonical.is_file():
            raise ValueError("SPARTA did not write the canonical surface")
        old_points, old_tris = _surface(out / "preclip.surf")
        points, triangles = _surface(canonical)
        if not same_oriented_geometry(old_points, old_tris, points, triangles):
            raise ValueError("SPARTA canonical triangles differ from oriented preclip geometry")
        by_type = {spec["surface_type"]: (name, spec) for name, spec in geometry["boundaries"].items()}
        if any(row[0] not in by_type for row in triangles):
            raise ValueError("canonical CAD surface has an unlabelled triangle type")
        records = [{"triangle_id": i, "boundary_id": by_type[row[0]][0],
                    "role": by_type[row[0]][1]["role"],
                    "molecule_action": by_type[row[0]][1]["molecule_action"]}
                   for i, row in enumerate(triangles, 1)]
        write_json(out / "triangle-roles.geometry-only.json", {
            "schema_version": "dsmc-triangle-roles-geometry-only-v1",
            "surface_sha256": sha(canonical), "records": records,
            "transport_readiness": "geometry-only"})
        write_tracer_surfs(out / "cell.surfs", box, points, triangles)
        (out / "in.canonical").write_text(deck("physical-walls.surf", box, grid,
                                               clip=False, surfmax=surfmax), encoding="utf-8")
        second = run_probe(out, solver, "in.canonical")
        solid = geometry["inspection"]["physical_boundary"]["signed_solid_volume_m3"]
        expected = math.prod(hi-lo for lo, hi in box) - solid
        for result in (first, second):
            if abs(result["flow_volume_m3"] / expected - 1) > tolerance:
                raise ValueError("solver flow volume disagrees with box minus CAD solid")
        if abs(first["flow_volume_m3"]-second["flow_volume_m3"]) > 1e-12*expected:
            raise ValueError("canonical reread changed SPARTA flow volume")
        geometry["physical_mesh_status"] = "canonical-SPARTA-clipped-geometry-only"
        geometry["artifacts"][0].update(path="physical-walls.surf", sha256=sha(canonical),
                                         producer={"path": str(solver), "sha256": sha(solver)})
        geometry["inspection"]["physical_boundary"]["triangles"] = len(triangles)
        geometry["inspection"]["physical_boundary"]["triangle_types"] = dict(Counter(row[0] for row in triangles))
        write_json(out / "geometry.json", geometry)
        receipt = {"status": "complete", "source_prepared": str(prepared),
                   "source_geometry_sha256": sha(prepared / "geometry.json"),
                   "solver": {"path": str(solver), "sha256": sha(solver)},
                   "species_sha256": sha(species), "expected_flow_volume_m3": expected,
                   "first": first, "canonical_reread": second,
                   "canonical_surface_sha256": sha(canonical),
                   "triangle_roles_sha256": sha(out / "triangle-roles.geometry-only.json"),
                   "tracer_surfs_sha256": sha(out / "cell.surfs")}
        write_json(out / "solver-check.json", receipt)
        return receipt
    except Exception as exc:
        write_json(out / "solver-check-failure.json", {"status": "failed", "error": str(exc),
                                                       "source_prepared": str(prepared)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(check(args.prepared, args.solver, args.out), indent=2))


if __name__ == "__main__":
    main()
