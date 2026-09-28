"""Explicit SPARTA run-0 clipping check for one prepared 3D boundary."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from dsmc_case import _identity, _json_write, inspect_geometry, load_profile
from dsmc_geometry import _inspect_surface, _mouth_area, _surface
from dsmc_run import _executable, _sha, render_launch


def _probe_deck(deck: str, physical: str, canonical: bool) -> str:
    lines = deck.splitlines()
    matches = [i for i, line in enumerate(lines) if re.fullmatch(r"read_surf\s+\S+ type group cellwalls clip", line)]
    if len(matches) != 1:
        raise ValueError("helium deck must have exactly one physical read_surf ... clip command")
    read = (f"read_surf   {physical} type group cellwalls" if canonical else
            f"read_surf   {physical} type group cellwalls clip file physical-walls.surf type")
    return "\n".join([*lines[:matches[0]], read,
                       "surf_collide walls diffuse 4.0 1.0",
                       "surf_modify cellwalls collide walls", "run 0", ""])


def _run_probe(directory: Path, profile: dict, executable: str, deck_name: str, variables: dict, log_name: str):
    arguments = ["-in", deck_name]
    for name in ("SEED", "FILLN", "FNUM"):
        arguments += ["-var", name, str(variables[name])]
    argv = render_launch(profile, executable, arguments)
    with (directory / log_name).open("wb") as log:
        result = subprocess.run(argv, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = (directory / log_name).read_text(errors="replace").splitlines()[-8:]
        raise ValueError(f"solver run 0 failed ({result.returncode}) in {log_name}: {' | '.join(tail)}")
    return argv


def solver_check(prepared_path: Path, profile_path: Path, out: Path) -> Path:
    """Create a fresh checked derivative; leave the original preparation intact."""
    source = Path(prepared_path).resolve()
    if source.is_file():
        source = source.parent
    inspect_geometry(source)
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"solver-check: refusing existing output path {out}")
    profile = load_profile(profile_path)
    if profile["platform"] != "local":
        raise ValueError("solver-check currently requires a local profile")
    executable = _executable(profile["solver_executable"])
    prepared = json.loads((source / "prepared.json").read_text())
    if prepared["state"] != "prepared-unclipped":
        raise ValueError("solver-check requires an unclipped preparation")
    geometry = json.loads((source / "geometry.json").read_text())
    input_record = next(x for x in prepared["inputs"] if x["role"] == "physical_boundary")
    cell = input_record["staged"]["path"]
    science = prepared["science_inputs"]
    variables = {"SEED": science["numerics"]["solver_seed"],
                 "FILLN": science["helium"]["fill"]["number_density_m3"],
                 "FNUM": science["numerics"]["fnum"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    # A failed probe retains its log and case copy under a failed status file.
    shutil.copytree(source, out)
    try:
        deck = (out / "in.helium").read_text(encoding="utf-8")
        (out / "in.solver-check").write_text(_probe_deck(deck, cell, False), encoding="utf-8", newline="\n")
        first_argv = _run_probe(out, profile, executable, "in.solver-check", variables, "solver-check.log")
        canonical = out / "physical-walls.surf"
        if not canonical.is_file() or canonical.stat().st_size == 0:
            raise ValueError("solver run 0 did not write physical-walls.surf")
        inspection, points, triangles = _inspect_surface(canonical, physical=True)
        area = _mouth_area(points)
        if abs(area - geometry["source_parameters"]["target_area_m2"]) > 1e-12:
            raise ValueError("clipped solver aperture area differs from the prepared target")
        inspection["aperture_area_m2"] = area
        records = []
        roles = {1: ("wall", "absorb"), 2: ("inlet_cap", "absorb")}
        for number, (surface_type, *_) in enumerate(triangles, 1):
            if surface_type not in roles:
                raise ValueError(f"canonical triangle {number}: unlabelled type {surface_type}")
            boundary, action = roles[surface_type]
            records.append({"triangle_id": number, "boundary_id": boundary,
                            "role": boundary, "molecule_action": action})
        _json_write(out / "triangle-roles.json", {
            "schema_version": "dsmc-triangle-roles-v1", "surface_sha256": _sha(canonical),
            "records": records})
        (out / "in.canonical-check").write_text(_probe_deck(deck, canonical.name, True),
                                                 encoding="utf-8", newline="\n")
        second_argv = _run_probe(out, profile, executable, "in.canonical-check", variables,
                                 "canonical-check.log")
        changed_deck = re.sub(r"read_surf\s+\S+ type group cellwalls clip",
                              "read_surf   physical-walls.surf type group cellwalls", deck, count=1)
        (out / "in.helium").write_text(changed_deck, encoding="utf-8", newline="\n")
        geometry["physical_mesh_status"] = "canonical-SPARTA-clipped"
        geometry["artifacts"] = [({**item, "path": canonical.name, "sha256": _sha(canonical),
                                   "producer": {"path": executable, "sha256": _sha(executable)}}
                                  if item["role"] == "physical_boundary" else item)
                                 for item in geometry["artifacts"]]
        geometry["inspection"]["physical_boundary"] = inspection
        _json_write(out / "geometry.json", geometry)
        input_record["staged"] = {**_identity(canonical), "path": canonical.name}
        deck_record = next(x for x in prepared["inputs"] if x["role"] == "helium_deck")
        deck_record["staged"] = {**_identity(out / "in.helium"), "path": "in.helium"}
        prepared["geometry_artifact"] = {**_identity(out / "geometry.json"), "path": "geometry.json"}
        prepared["state"] = "solver-checked"
        prepared["solver_check"] = {"status": "complete", "executable_sha256": _sha(executable),
                                    "profile": profile["source_profile"], "first_command": first_argv,
                                    "canonical_command": second_argv,
                                    "physical_surface_sha256": _sha(canonical),
                                    "triangle_roles_sha256": _sha(out / "triangle-roles.json")}
        _json_write(out / "prepared.json", prepared)
        return out / "prepared.json"
    except Exception as exc:
        _json_write(out / "solver-check-failure.json", {"status": "failed", "error": str(exc),
                                                        "source_prepared": str(source)})
        raise
