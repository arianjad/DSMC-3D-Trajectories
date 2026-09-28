#!/usr/bin/env python3
"""Run a short, source-only 3D helium diagnostic on solver-checked CAD.

This is a transport smoke test. It does not establish a steady flow, a
converged DSMC discretization, or an apparatus-specific mass-flow setting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _positive(value, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _validate_config(data: dict) -> None:
    keys = {"schema_version", "case_id", "geometry_case_id", "scope", "seed",
            "mass_flow_kg_s", "temperature_K", "mixture_number_density_m3",
            "timestep_s", "steps", "stats_every_steps"}
    optional = {"inlet_return_action", "grid_refinement"}
    fnum_keys = {"fnum_real_per_sim_particle", "fnum_m3"}
    if (set(data) - optional - fnum_keys != keys or
            len(set(data) & fnum_keys) != 1 or
            not set(data) <= keys | optional | fnum_keys or
            data["schema_version"] != "dsmc-cad-helium-smoke-v1"):
        raise ValueError("unsupported CAD helium smoke configuration")
    if data.get("inlet_return_action", "reflect") not in ("reflect", "vanish"):
        raise ValueError("inlet_return_action must be reflect or vanish")
    for key in ("case_id", "geometry_case_id", "scope"):
        if (not isinstance(data[key], str) or not data[key].strip() or
                "\n" in data[key] or "\r" in data[key]):
            raise ValueError(f"{key} must be nonempty single-line text")
    for key in ("seed", "steps", "stats_every_steps"):
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if data["steps"] % data["stats_every_steps"]:
        raise ValueError("steps must be a multiple of stats_every_steps")
    for key in ("mass_flow_kg_s", "temperature_K", "mixture_number_density_m3",
                "timestep_s", *(set(data) & fnum_keys)):
        _positive(data[key], key)


def _validate_geometry(checked: Path, solver: Path, config: dict) -> tuple[dict, dict, dict]:
    receipt_path, geometry_path = checked / "solver-check.json", checked / "geometry.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if receipt["status"] != "complete" or geometry["adapter"] != "cad-import":
        raise ValueError("CAD geometry needs a completed solver check")
    if geometry["physical_mesh_status"] != "canonical-SPARTA-clipped-geometry-only":
        raise ValueError("CAD geometry is not a checked canonical surface")
    if config["geometry_case_id"] != geometry["case_id"]:
        raise ValueError("transport configuration names a different CAD case")
    if _sha(solver) != receipt["solver"]["sha256"]:
        raise ValueError("solver differs from the binary used for CAD run-zero check")
    surf = checked / "physical-walls.surf"
    surf_hash = _sha(surf)
    if (surf_hash != receipt["canonical_surface_sha256"] or
            surf_hash != geometry["artifacts"][0]["sha256"]):
        raise ValueError("checked CAD surface hash mismatch")
    inlet = geometry.get("helium_inlet")
    if not inlet or set(inlet) != {"boundary_id", "normal_unit"}:
        raise ValueError("CAD geometry needs an explicit helium inlet")
    boundaries = geometry["boundaries"]
    if inlet["boundary_id"] not in boundaries:
        raise ValueError("helium inlet boundary is missing")
    cap = boundaries[inlet["boundary_id"]]
    if cap["role"] != "emitter" or cap["surface_type"] != 2:
        raise ValueError("helium inlet must be the type-2 emitter cap")
    if geometry["inspection"]["physical_boundary"].get("helium_inlet_normal_dot_min", -1) < 0.99:
        raise ValueError("emitter normals disagree with declared gas-side direction")
    # One uniform diffuse wall model is the scope of this first transport path.
    wall = {(b["temperature_K"], b["helium_wall_model"], b["helium_accommodation"])
            for b in boundaries.values()}
    if wall != {(config["temperature_K"], "diffuse", 1.0)}:
        raise ValueError("all CAD boundaries need the configured fully accommodating diffuse wall")
    if geometry["inspection"]["roles"][inlet["boundary_id"]]["triangles"] <= 0:
        raise ValueError("helium inlet cap has no triangles")
    return receipt, geometry, cap


FACE_DEADBAND = 1e-3   # cells; a station plane closer than this to a face counts as on it
CORNER_EPS = 1e-6      # cells; a grid corner closer than this to a region boundary is ambiguous


def region_margin(style: str, args: list, x) -> float:
    """Signed margin, >= 0 exactly where SPARTA's inclusive region inside() holds.

    region_block.cpp:71-77 and region_cylinder.cpp:87-110 (27Aug2026) use
    closed intervals; create_grid.cpp:640-670 applies them to the 8 corners.
    """
    if style == "block":
        return min(min(x[i] - args[2*i], args[2*i+1] - x[i]) for i in range(3))
    axis = "xyz".index(args[0])
    c1, c2, radius, lo, hi = args[1:]
    p, q = (i for i in range(3) if i != axis)
    return min(radius - math.hypot(x[p] - c1, x[q] - c2), x[axis] - lo, hi - x[axis])


def _region_bbox(style: str, args: list) -> list:
    if style == "block":
        return [(args[2*i], args[2*i+1]) for i in range(3)]
    axis = "xyz".index(args[0])
    c1, c2, radius, lo, hi = args[1:]
    p, q = (i for i in range(3) if i != axis)
    box = [None] * 3
    box[axis], box[p], box[q] = (lo, hi), (c1 - radius, c1 + radius), (c2 - radius, c2 + radius)
    return box


def _buffer_ok(outer: dict, inner: dict, margin: list) -> bool:
    """Inner region sits at least `margin[i]` inside the outer region on every side."""
    inner_box = _region_bbox(inner["style"], inner["args"])
    tol = [CORNER_EPS * m for m in margin]
    if outer["style"] == "block":
        return all(ilo - olo >= m - t and ohi - ihi >= m - t for (olo, ohi), (ilo, ihi), m, t
                   in zip(_region_bbox("block", outer["args"]), inner_box, margin, tol))
    axis = "xyz".index(outer["args"][0])
    c1, c2, radius, lo, hi = outer["args"][1:]
    p, q = (i for i in range(3) if i != axis)
    if inner["style"] == "cylinder" and inner["args"][0] == outer["args"][0]:
        far = math.hypot(inner["args"][1] - c1, inner["args"][2] - c2) + inner["args"][3]
    else:   # farthest bbox corner: exact for a block, conservative otherwise
        far = math.hypot(max(abs(v - c1) for v in inner_box[p]),
                         max(abs(v - c2) for v in inner_box[q]))
    return (inner_box[axis][0] - lo >= margin[axis] - tol[axis] and
            hi - inner_box[axis][1] >= margin[axis] - tol[axis] and
            radius - far >= max(margin[p], margin[q]) - max(tol[p], tol[q]))


def _refinement(config: dict, probe: dict) -> tuple[str, str]:
    """Region lines and `levels` suffix for nested static 2:1 refinement.

    Spec (config["grid_refinement"]): {"levels": [{"region_id", "style", "args"}, ...]},
    entry i defining SPARTA level i+2 by 2x2x2 subdivision of level-(i+1) cells
    with a corner inside the region (`inside any`). Styles: block
    [xlo, xhi, ylo, yhi, zlo, zhi] and cylinder [axis, c1, c2, radius, lo, hi].
    Enforced: every level-N region sits inside the level-(N-1) region with at
    least four level-(N-1) cells on every side (docs/sparta-reference.md sec. 1);
    no level-(N-1) grid corner lies on a region boundary (the corner test would
    then flip on round-off); the exit-station plane is off every cell face at
    each level that can reach it.
    """
    spec = config.get("grid_refinement")
    if spec is None:
        return "", ""
    if (not isinstance(spec, dict) or set(spec) != {"levels"} or
            not isinstance(spec["levels"], list) or not spec["levels"]):
        raise ValueError("grid_refinement needs a nonempty levels list")
    bounds = probe["domain_bounds_m"]
    h = [(hi - lo) / n for (lo, hi), n in zip(bounds, probe["base_grid"])]
    station = config.get("exit_station", {}).get("z_m")
    faces_checked = [h[2]] if station is not None else []
    lines, suffix, ids, previous = [], [], set(), None
    for level, entry in enumerate(spec["levels"], start=2):
        if not isinstance(entry, dict) or set(entry) != {"region_id", "style", "args"}:
            raise ValueError("refinement level needs region_id, style and args")
        rid, style, args = entry["region_id"], entry["style"], entry["args"]
        if (not isinstance(rid, str) or not re.fullmatch(r"refine[a-z0-9_]{0,40}", rid) or
                rid in ids):
            raise ValueError("refinement region_id must be a distinct refine* name")
        ids.add(rid)
        numbers = args[1:] if style == "cylinder" else args
        if (style not in ("block", "cylinder") or not isinstance(args, list) or
                len(args) != 6 or
                (style == "cylinder" and args[0] not in ("x", "y", "z")) or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in numbers)):
            raise ValueError(f"unsupported refinement region {rid}")
        box = _region_bbox(style, args)
        if any(lo >= hi for lo, hi in box):
            raise ValueError(f"refinement region {rid} is empty")
        # h is the level-(level-1) cell edge: the cells this region test subdivides.
        if previous is not None and not _buffer_ok(previous, entry, [4 * v for v in h]):
            raise ValueError(f"refinement region {rid} violates the 2:1 four-cell buffer")
        # Level-(level-1) corners live on the global lattice lo + i*h.
        ranges = [range(math.floor((blo - h[d] - lo) / h[d]), math.ceil((bhi + h[d] - lo) / h[d]) + 1)
                  for d, ((lo, _), (blo, bhi)) in enumerate(zip(bounds, box))]
        eps = CORNER_EPS * min(h)
        for i in ranges[0]:
            for j in ranges[1]:
                for k in ranges[2]:
                    x = (bounds[0][0] + i*h[0], bounds[1][0] + j*h[1], bounds[2][0] + k*h[2])
                    if abs(region_margin(style, args, x)) < eps:
                        raise ValueError(f"refinement region {rid} boundary passes through "
                                         f"a level-{level-1} grid corner at {x}")
        h = [v / 2 for v in h]
        if station is not None and box[2][0] - 2*h[2] <= station <= box[2][1] + 2*h[2]:
            faces_checked.append(h[2])
        lines.append(f"region {rid} {style} " + " ".join(
            v if isinstance(v, str) else f"{v:.17g}" for v in args) + "\n")
        suffix.append(f"region {level} {rid} 2 2 2")
        previous = entry
    for edge in faces_checked:
        u = (station - bounds[2][0]) / edge
        if abs(u - round(u)) < FACE_DEADBAND:
            raise ValueError(f"exit station z={station} lies on a cell face of edge {edge}")
    return "".join(lines), f" levels {len(spec['levels']) + 1} {' '.join(suffix)} inside any"


def _deck(config: dict, geometry: dict) -> str:
    probe = geometry["solver_probe"]
    bounds = " ".join(f"{v:.17g}" for axis in probe["domain_bounds_m"] for v in axis)
    grid = " ".join(str(n) for n in probe["base_grid"])
    refine_regions, refine_suffix = _refinement(config, probe)
    grid += refine_suffix
    surfmax = probe.get("surfmax", 1024)
    inlet_collision = ("surf_collide inlet vanish\nsurf_modify cap collide inlet\n"
                       if config.get("inlet_return_action", "reflect") == "vanish" else "")
    return (f"# {config['scope']}\n"
            f"seed {config['seed']}\nunits si\ndimension 3\nboundary o o o\n"
            f"global gridcut 0.0 surfmax {surfmax} mem/limit 1024\n"
            f"create_box {bounds}\n{refine_regions}create_grid {grid}\n"
            "global weight cell volume\nspecies he.species He\n"
            f"mixture hegas He nrho {config['mixture_number_density_m3']:.17g} "
            f"temp {config['temperature_K']:.17g}\n"
            f"global fnum {config.get('fnum_real_per_sim_particle', config.get('fnum_m3')):.17g}\n"
            "read_surf physical-walls.surf type group cellwalls\n"
            f"surf_collide walls diffuse {config['temperature_K']:.17g} 1.0\n"
            "surf_modify cellwalls collide walls\n"
            "group cap surf type 2\n"
            f"{inlet_collision}"
            f"fix src emit/surf hegas cap normal yes mflow "
            f"{config['mass_flow_kg_s']:.17g} {config['temperature_K']:.17g}\n"
            "collide vss hegas he.vss\n"
            f"stats {config['stats_every_steps']}\n"
            "stats_style step cpu np f_src[1] f_src[2]\n"
            "stats_modify flush yes\n"
            f"dump state particle hegas {config['steps']} particles.* x y z vx vy vz\n"
            f"timestep {config['timestep_s']:.17g}\nrun {config['steps']}\n")


def _summary(output: str, steps: int) -> dict:
    match = re.search(r"^\s*" + str(steps) + r"\s+[\d.eE+-]+\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
                      output, re.MULTILINE)
    if not match or "Loop time of" not in output:
        raise ValueError("SPARTA run did not report a final source count")
    nparticles, emitted_step, emitted_total = map(int, match.groups())
    if emitted_total <= 0 or nparticles > emitted_total:
        raise ValueError("source emission or particle accounting failed")
    return {"final_particles": nparticles, "emitted_final_step": emitted_step,
            "emitted_cumulative": emitted_total, "initial_particles": 0}


def _snapshot_summary(path: Path, expected_particles: int, normal: list) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if (len(lines) < 9 or lines[0] != "ITEM: TIMESTEP" or
            lines[2] != "ITEM: NUMBER OF ATOMS" or
            lines[8].strip() != "ITEM: ATOMS x y z vx vy vz"):
        raise ValueError("unexpected SPARTA particle snapshot format")
    if int(lines[3]) != expected_particles or len(lines) != 9 + expected_particles:
        raise ValueError("particle snapshot count differs from final SPARTA stats")
    forward = 0
    for line in lines[9:]:
        fields = [float(value) for value in line.split()]
        if len(fields) != 6 or any(not math.isfinite(value) for value in fields):
            raise ValueError("invalid particle snapshot row")
        forward += sum(fields[i+3]*normal[i] for i in range(3)) > 0
    return {"final_velocity_along_inlet_normal_positive": forward,
            "final_velocity_along_inlet_normal_nonpositive": expected_particles-forward,
            "source_direction_unit": normal}


def run(checked: Path, config_path: Path, solver: Path, out: Path) -> dict:
    checked, config_path, solver, out = (Path(p).resolve() for p in
                                         (checked, config_path, solver, out))
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"CAD flow output already exists: {out}")
    if out == checked or out in checked.parents or checked in out.parents:
        raise ValueError("flow output must be separate from solver-checked CAD geometry")
    if not solver.is_file():
        raise ValueError(f"missing solver: {solver}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    checked_receipt, geometry, cap = _validate_geometry(checked, solver, config)
    out.mkdir(parents=True)
    try:
        for name, source in (("physical-walls.surf", checked / "physical-walls.surf"),
                             ("he.species", ROOT / "cases/data/he.species"),
                             ("he.vss", ROOT / "cases/data/he.vss")):
            shutil.copyfile(source, out / name)
        deck = out / "in.helium"
        deck.write_text(_deck(config, geometry), encoding="utf-8", newline="\n")
        result = subprocess.run([str(solver), "-in", deck.name], cwd=out,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=120, check=False)
        log = out / "solver.log"
        log.write_bytes(result.stdout)
        if result.returncode:
            raise ValueError(f"SPARTA helium diagnostic failed ({result.returncode}); see {log}")
        summary = _summary(result.stdout.decode(errors="replace"), config["steps"])
        snapshot = out / f"particles.{config['steps']}"
        if not snapshot.is_file():
            raise ValueError("SPARTA did not write the final particle snapshot")
        snapshot_summary = _snapshot_summary(snapshot, summary["final_particles"],
                                             geometry["helium_inlet"]["normal_unit"])
        receipt = {"schema_version": "dsmc-cad-helium-smoke-result-v1", "status": "complete",
                   "scope": config["scope"], "case_id": config["case_id"],
                   "geometry_case_id": geometry["case_id"],
                   "geometry_sha256": _sha(checked / "geometry.json"),
                   "solver_check_sha256": _sha(checked / "solver-check.json"),
                   "canonical_surface_sha256": checked_receipt["canonical_surface_sha256"],
                   "config_sha256": _sha(config_path), "solver_sha256": _sha(solver),
                   "deck_sha256": _sha(deck), "log_sha256": _sha(log),
                   "snapshot_sha256": _sha(snapshot), "inlet_boundary_id": geometry["helium_inlet"]["boundary_id"],
                   "inlet_area_m2": geometry["inspection"]["roles"][geometry["helium_inlet"]["boundary_id"]]["area_m2"],
                   "wall_model": cap["helium_wall_model"],
                   "inlet_return_action": config.get("inlet_return_action", "reflect"),
                   "steps": config["steps"],
                   "elapsed_physical_s": config["steps"] * config["timestep_s"],
                   "source_mass_flow_kg_s": config["mass_flow_kg_s"],
                   **summary, **snapshot_summary}
        _write_json(out / "run.json", receipt)
        return receipt
    except Exception as exc:
        _write_json(out / "run-failure.json", {"status": "failed", "error": str(exc),
                                               "checked_geometry": str(checked)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checked", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.checked, args.config, args.solver, args.out), indent=2))


if __name__ == "__main__":
    main()
