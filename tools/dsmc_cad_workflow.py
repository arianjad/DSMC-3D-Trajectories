#!/usr/bin/env python3
"""Bridge a solver-checked CAD surface and explicit transport laws into HPC inputs."""
from __future__ import annotations

import copy
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

from dsmc_case import ROOT, _identity, _read_json, _json_write, _keys, _number, _integer, _seed, _vector
from dsmc_cad_field import (_check_initial_chain, _continuation_physics_differs,
                            _field_deck, _geometry_domain, _inspect_particle_dump,
                            _needs_preclone, _preclone_initial, _run_domain,
                            _validate_field_config, _write_exit_disks)


BRIDGE_VERSION = "dsmc-cad-workflow-preparation-v1"


def _same_file(path: Path, identity: dict, label: str) -> None:
    if not path.is_file() or _identity(path)["sha256"] != identity.get("sha256"):
        raise ValueError(f"{label} is missing or changed")


def _original_case(geometry: dict, checked: Path) -> Path:
    source = geometry["source_case"]
    if source is None:
        if type(geometry.get("case_id")) is not str or not geometry["case_id"]:
            raise ValueError("CAD geometry lacks a source-case identity")
        return checked / "geometry.json"
    historical = source["path"].replace("\\", "/")
    marker = "cases/"
    if marker not in historical:
        raise ValueError("CAD geometry lacks a repository source case")
    relative = historical[historical.index(marker):]
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        raise ValueError("CAD source case escapes repository")
    _same_file(path, source, "CAD source case")
    return path


def _validate_transport(data: dict, geometry: dict, case_id: str) -> None:
    _keys(data, ("schema_version", "case_id", "geometry_case_id", "model",
                 "molecules", "observables", "provenance", "geometry_transform",
                 "restart_every_steps"), optional=("boundary_actions",),
          where="CAD transport")
    if data["schema_version"] != "dsmc-cad-transport-v1":
        raise ValueError("unsupported CAD transport schema")
    if data["geometry_case_id"] != case_id or data["geometry_transform"] != geometry["transform"]:
        raise ValueError("CAD transport geometry identity or solver-to-tracer transform differs")
    _integer(data["restart_every_steps"], "CAD transport.restart_every_steps", minimum=1)
    actions = data.get("boundary_actions")
    if actions is None:
        if any(b.get("molecule_action") != "absorb" for b in geometry["boundaries"].values()):
            raise ValueError("CAD transport needs explicit molecule boundary actions")
    elif (type(actions) is not dict or set(actions) != set(geometry["boundaries"]) or
          any(action != "absorb" or
              geometry["boundaries"][name].get("molecule_action") not in ("absorb", "unspecified")
              for name, action in actions.items())):
        raise ValueError("CAD transport boundary actions differ from physical boundaries")
    for key in ("case_id", "provenance"):
        if type(data[key]) is not str or not data[key].strip():
            raise ValueError(f"CAD transport {key} is empty")
    _keys(data["model"], ("label", "provenance"), where="CAD transport.model")
    molecules = data["molecules"]
    _keys(molecules, ("mass_u", "gas_mass_u", "cross_section_m2", "source", "count",
                      "seed", "flight", "collision_rate_model", "flight_step_m",
                      "rng_protocol", "wall_action", "gas_reconstruction", "provenance"),
          where="CAD transport.molecules")
    for key in ("mass_u", "gas_mass_u", "cross_section_m2", "flight_step_m"):
        _number(molecules[key], f"molecules.{key}", positive=True)
    _integer(molecules["count"], "molecules.count", minimum=1)
    _seed(molecules["seed"], "molecules.seed")
    if (molecules["flight"] != "local" or molecules["collision_rate_model"] != "approximate" or
            molecules["rng_protocol"] != "particle-v1" or molecules["wall_action"] != "absorb" or
            molecules["gas_reconstruction"] != {"field": "frozen-scalar-temperature", "missing_cell": "error"}):
        raise ValueError("unsupported first CAD molecular transport law")
    source = molecules["source"]
    _keys(source, ("distribution", "center_m", "radius_m", "drift_m_s", "temperature_K"),
          where="CAD transport.molecules.source")
    center = _vector(source["center_m"], "molecules.source.center_m")
    _vector(source["drift_m_s"], "molecules.source.drift_m_s")
    if (source["distribution"] != "point" or source["radius_m"] != 0 or
            _number(source["temperature_K"], "molecules.source.temperature_K", positive=True) <= 0):
        raise ValueError("first CAD workflow needs an explicit point source")
    bounds = geometry["solver_probe"]["domain_bounds_m"]
    if any(not lo < x < hi for x, (lo, hi) in zip(center, bounds)):
        raise ValueError("molecule source is outside the CAD solver box")
    # The geometry case explicitly declares gas-side probes. A new source needs
    # an independent native gas-membership check before this first bridge expands.
    if center not in geometry["gas_reference_points_m"]:
        raise ValueError("molecule source is not a checked CAD gas reference point")
    observables = data["observables"]
    _keys(observables, ("planes", "field_regions"), where="CAD transport.observables")
    if not observables["planes"] or not observables["field_regions"]:
        raise ValueError("CAD transport needs a diagnostic plane and field region")
    for plane in observables["planes"]:
        _keys(plane, ("name", "origin_m", "normal", "axis_u", "axis_v", "acceptance"),
              where="CAD transport.plane")
        for key in ("origin_m", "normal", "axis_u", "axis_v"):
            _vector(plane[key], f"observables.plane.{key}")
        normal, u, v = (plane[key] for key in ("normal", "axis_u", "axis_v"))
        dot = lambda a, b: sum(x*y for x, y in zip(a, b))
        cross = [normal[1]*u[2]-normal[2]*u[1], normal[2]*u[0]-normal[0]*u[2], normal[0]*u[1]-normal[1]*u[0]]
        if (any(abs(dot(x, x)-1) > 1e-9 for x in (normal, u, v)) or
                any(abs(dot(a, b)) > 1e-9 for a, b in ((normal, u), (normal, v), (u, v))) or
                any(abs(a-b) > 1e-9 for a, b in zip(cross, v))):
            raise ValueError("CAD diagnostic plane axes are not right-handed orthonormal")
        _keys(plane["acceptance"], ("position_u_m", "position_v_m", "angle_rad", "time_s"),
              where="CAD transport.plane.acceptance")
        if any(value is not None for value in plane["acceptance"].values()):
            raise ValueError("first CAD diagnostic plane has no acceptance cuts")


def _checked(checked: Path, helium: dict, transport: dict) -> tuple[dict, dict, dict, Path]:
    geometry = _read_json(checked / "geometry.json")
    receipt = _read_json(checked / "solver-check.json")
    original = _original_case(geometry, checked)
    original_data = _read_json(original)
    case_id = original_data["case_id"]
    if (receipt.get("status") != "complete" or geometry.get("adapter") != "cad-import" or
            geometry.get("physical_mesh_status") != "canonical-SPARTA-clipped-geometry-only" or
            helium["geometry_case_id"] != case_id or
            helium["case_id"] != transport["case_id"]):
        raise ValueError("CAD workflow needs the matching completed canonical solver check")
    source_name = geometry["source"]["path"]
    if Path(source_name).name != source_name or Path(source_name).suffix.lower() not in (".step", ".stl"):
        raise ValueError("CAD source geometry name is invalid")
    _same_file(checked / source_name, geometry["source"], "CAD source geometry")
    wall = checked / "physical-walls.surf"
    wall_hash = _identity(wall)["sha256"]
    if (wall_hash != receipt.get("canonical_surface_sha256") or
            wall_hash != geometry["artifacts"][0]["sha256"]):
        raise ValueError("CAD canonical surface hash differs")
    roles_path = checked / "triangle-roles.geometry-only.json"
    roles = _read_json(roles_path)
    if (roles.get("schema_version") != "dsmc-triangle-roles-geometry-only-v1" or
            roles.get("surface_sha256") != wall_hash or
            _identity(roles_path)["sha256"] != receipt.get("triangle_roles_sha256")):
        raise ValueError("CAD geometry-only roles differ from the solver check")
    boundaries = geometry["boundaries"]
    inlet_id = geometry["helium_inlet"]["boundary_id"]
    if (boundaries[inlet_id]["role"] != "emitter" or
            boundaries[inlet_id]["surface_type"] != 2 or
            geometry["inspection"]["physical_boundary"].get("helium_inlet_normal_dot_min", -1) < .99):
        raise ValueError("CAD inlet cap lacks the checked type-2 gas-side direction")
    if (not all(b.get("helium_wall_model") == "diffuse" and
                b.get("helium_accommodation") == 1 and
                b.get("temperature_K") == helium["temperature_K"] for b in boundaries.values())):
        raise ValueError("CAD boundaries lack explicit compatible helium and molecule laws")
    records = roles.get("records")
    if (not isinstance(records, list) or not records or
            len(records) != geometry["inspection"]["physical_boundary"]["triangles"] or
            any(row.get("triangle_id") != i or
                row.get("boundary_id") not in boundaries or
                row.get("role") != boundaries[row["boundary_id"]]["role"] or
                row.get("molecule_action") != boundaries[row["boundary_id"]].get("molecule_action")
                for i, row in enumerate(records, 1))):
        raise ValueError("CAD canonical triangle boundary roles are incomplete or inconsistent")
    _validate_transport(transport, geometry, case_id)
    return geometry, receipt, roles, original


def _stage(out: Path, source: Path, name: str, role: str, items: list) -> None:
    target = out / name
    if target.exists():
        raise ValueError(f"duplicate staged CAD input: {name}")
    shutil.copyfile(source, target)
    items.append({"role": role, "transfer_mode": "binary", "source": _identity(source),
                  "staged": {**_identity(target), "path": name}})


def _stage_initial(helium: dict, checked: Path, receipt: dict, geometry: dict,
                   out: Path, items: list, deck: Path) -> dict:
    fill = helium.get("initial_fill", {})
    if "continuation" in helium:
        cont = helium["continuation"]
        parent_dir = (ROOT / cont["parent_run_dir"]).resolve()
        parent_config = (ROOT / cont["parent_config_path"]).resolve()
        if ROOT not in parent_dir.parents or ROOT not in parent_config.parents:
            raise ValueError("CAD continuation paths escape repository")
        run_path = parent_dir / "run.json"
        _same_file(run_path, {"sha256": cont["parent_run_sha256"]}, "CAD continuation receipt")
        parent = _read_json(run_path)
        parent_helium = _read_json(parent_config)
        _validate_field_config(parent_helium)
        box = geometry["solver_probe"]["domain_bounds_m"]
        same_geometry = (
            parent.get("geometry_sha256") == _identity(checked / "geometry.json")["sha256"] and
            parent.get("solver_check_sha256") == _identity(checked / "solver-check.json")["sha256"])
        # G3: another checked geometry is allowed only as a box extension of the parent's.
        domains = () if same_geometry else (
            _run_domain(parent_dir / "in.helium-field", parent),
            _geometry_domain(geometry, receipt["canonical_surface_sha256"]))
        if (parent.get("status") != "complete" or parent.get("case_id") != cont["parent_case_id"] or
                parent.get("config_sha256") != _identity(parent_config)["sha256"] or
                parent.get("steps") != cont["parent_steps"] or
                parent.get("canonical_surface_sha256") != receipt["canonical_surface_sha256"] or
                parent.get("solver_sha256") != receipt["solver"]["sha256"] or
                parent_helium.get("seed") == helium["seed"] or
                _continuation_physics_differs(parent_helium, helium, *domains)):
            raise ValueError("CAD continuation parent does not match this physical case")
        snapshot = parent_dir / f"particles.{cont['parent_steps']}"
        _same_file(snapshot, {"sha256": cont["parent_snapshot_sha256"]}, "CAD continuation particles")
        converted = out / f"initial-particles.{cont['parent_steps']}"
        # Unrefined parent -> refined child: stage the pre-clone onto the rendered deck.
        needs_preclone = _needs_preclone(parent_helium, helium)
        plain = out / f"converted-particles.{cont['parent_steps']}" if needs_preclone else converted
        info = _inspect_particle_dump(snapshot, cont["parent_steps"], parent["final_particles"],
                                      domains[0]["domain_bounds_m"] if domains else box, plain,
                                      box if domains else None)
        preclone = _preclone_initial(deck, plain, converted) if needs_preclone else None
        _stage(out, run_path, "continuation-parent-run.json", "continuation_parent_receipt", items)
        _stage(out, parent_config, "continuation-parent-config.json", "continuation_parent_config", items)
        if domains:
            _stage(out, parent_dir / "in.helium-field", "continuation-parent-deck.in",
                   "continuation_parent_deck", items)
        _stage(out, snapshot, f"continuation-parent-particles.{cont['parent_steps']}",
               "continuation_parent_particles", items)
        items.append({"role": "initial_particles", "transfer_mode": "binary", "source": _identity(snapshot),
                      "staged": {**_identity(converted), "path": converted.name}})
        result = {"kind": "continuation", "parent": cont, "converted": info}
        if preclone is not None:
            result["preclone"] = preclone
        return result
    if fill.get("model") == "cad-connected-cell-gas":
        source = (ROOT / fill["prefill_path"]).resolve()
        if ROOT not in source.parents:
            raise ValueError("CAD prefill path escapes repository")
        prefill = source.parent / "prefill.json"
        deck = source.parent / "in.prefill-candidates"
        record = _read_json(prefill)
        text = deck.read_text(encoding="utf-8")
        if (record.get("case_id") != fill.get("prefill_source_case_id", helium["case_id"]) or
                record.get("solid_step_sha256") != fill["solid_step_sha256"] or
                fill["solid_step_sha256"] != _identity(checked / geometry["source"]["path"])["sha256"] or
                record.get("canonical_surface_sha256") != receipt["canonical_surface_sha256"] or
                not math.isclose(record.get("number_density_m3", float("nan")),
                                 fill["number_density_m3"], rel_tol=1e-12) or
                record.get("initial_dump_sha256") != _identity(source)["sha256"] or
                f"seed {helium['seed']}\n" not in text or
                f"mixture fillgas He nrho {fill['number_density_m3']:.17g} temp {helium['temperature_K']:.17g}\n" not in text or
                f"global fnum {helium['fnum_real_per_sim_particle']:.17g}\n" not in text):
            raise ValueError("CAD prefill payload or producer laws differ")
        _stage(out, source, "initial-particles.0", "initial_particles", items)
        _stage(out, prefill, "cad-prefill.json", "prefill_receipt", items)
        _stage(out, deck, "cad-prefill-deck", "prefill_deck", items)
        return {"kind": "prefill", "receipt_sha256": _identity(prefill)["sha256"]}
    return {"kind": "empty-domain"}


def _prepare(checked: Path, helium_path: Path, transport_path: Path, out: Path) -> None:
    helium = _read_json(helium_path)
    transport = _read_json(transport_path)
    _validate_field_config(helium)
    geometry, receipt, roles, original = _checked(checked, helium, transport)
    out.mkdir()
    inputs = []
    for source, name, role in (
        (checked / "physical-walls.surf", "physical-walls.surf", "physical_boundary"),
        (checked / "triangle-roles.geometry-only.json", "triangle-roles.geometry-only.json", "geometry_only_roles"),
        (checked / "solver-check.json", "geometry-solver-check.json", "geometry_solver_check"),
        (checked / geometry["source"]["path"],
         "geometry-source" + Path(geometry["source"]["path"]).suffix.lower(), "geometry_source"),
        (original, "geometry-case.json", "geometry_source_case"),
        (helium_path, "cad-helium-config.json", "helium_config"),
        (transport_path, "cad-transport-config.json", "transport_config"),
        (ROOT / "cases/data/he.species", "he.species", "helium_species"),
        (ROOT / "cases/data/he.vss", "he.vss", "helium_collision"),
    ):
        _stage(out, source, name, role, inputs)
    # Keep the checked geometry bytes as the prepared geometry artifact.
    shutil.copyfile(checked / "geometry.json", out / "geometry.json")
    # Rendered before the initial state so a pre-clone reads the staged deck's grid.
    deck = _field_deck(helium, geometry, mpi_ranks=4)
    marker = f"run {helium['steps']}\n"
    if deck.count(marker) != 1:
        raise ValueError("CAD pilot deck has an unexpected run command")
    deck = deck.replace(marker, f"restart {transport['restart_every_steps']} cad.restart.%.*\n" + marker +
                        "write_restart final.restart.%\n")
    (out / "in.helium").write_text(deck, encoding="utf-8", newline="\n")
    initial = _stage_initial(helium, checked, receipt, geometry, out, inputs, out / "in.helium")
    if "exit_station" in helium:
        station = helium["exit_station"]
        bounds = geometry["solver_probe"]["domain_bounds_m"]
        surface_zmax = geometry["inspection"]["physical_boundary"]["bbox_m"][2][1]
        if (not surface_zmax + 1e-4 < station["z_m"] < bounds[2][1] - 1e-4 or
                station["radius_m"] >= min(bounds[0][1], -bounds[0][0],
                                           bounds[1][1], -bounds[1][0])):
            raise ValueError("CAD exit station must clear physical wall and domain")
        for disk in _write_exit_disks(out, station):
            inputs.append({"role": "diagnostic_exit_surface", "transfer_mode": "binary",
                           "source": _identity(helium_path),
                           "staged": {**_identity(disk), "path": disk.name}})
    runnable = copy.deepcopy(roles)
    actions = transport.get("boundary_actions", {name: spec["molecule_action"]
                                                 for name, spec in geometry["boundaries"].items()})
    for row in runnable["records"]:
        row["molecule_action"] = actions[row["boundary_id"]]
    runnable["schema_version"] = "dsmc-triangle-roles-v1"
    runnable["state"] = "solver-checked-runnable"
    runnable["transport_readiness"] = "explicit-helium-and-molecule-laws"
    runnable["physical_laws"] = {"helium_config_sha256": _identity(helium_path)["sha256"],
                                  "transport_config_sha256": _identity(transport_path)["sha256"]}
    _json_write(out / "triangle-roles.json", runnable)
    inputs.append({"role": "triangle_roles", "transfer_mode": "binary",
                   "source": _identity(checked / "triangle-roles.geometry-only.json"),
                   "staged": {**_identity(out / "triangle-roles.json"), "path": "triangle-roles.json"}})
    inputs.append({"role": "helium_deck", "transfer_mode": "binary", "source": _identity(helium_path),
                   "staged": {**_identity(out / "in.helium"), "path": "in.helium"}})
    n = {"deck_template": "in.helium", "domain_bounds_m": geometry["solver_probe"]["domain_bounds_m"],
         "base_grid": geometry["solver_probe"]["base_grid"], "refinement": "cad-native-grid",
         "weighting": "cell-volume", "fnum": helium["fnum_real_per_sim_particle"],
         "dt_s": helium["timestep_s"], "total_steps": helium["steps"],
         "warmup_steps": helium["steps"] - helium["field_sample_every_steps"]*helium["field_sample_count"],
         "solver_seed": helium["seed"],
         "field_every_steps": helium.get("field_frame_every_steps", helium["steps"]),
         "field_average": {"samples": helium["field_sample_count"],
                           "stride_steps": helium["field_sample_every_steps"],
                           "period_steps": helium["field_sample_count"]*helium["field_sample_every_steps"]},
         "restart_every_steps": transport["restart_every_steps"], "restart_prefix": "cad.restart"}
    runnable_boundaries = copy.deepcopy(geometry["boundaries"])
    for name, action in actions.items():
        runnable_boundaries[name]["molecule_action"] = action
    science = {"model": transport["model"],
               "geometry": {"adapter": "cad", "transform": geometry["transform"],
                            "boundaries": runnable_boundaries,
                            "gas_reference_points_m": geometry["gas_reference_points_m"],
                            "gas_reference_point_m": transport["molecules"]["source"]["center_m"],
                            "solid_boundary_ids": list(geometry["boundaries"]),
                            "source_case_id": transport["geometry_case_id"]},
               "helium": {"species_file": "he.species", "collision_file": "he.vss",
                          "wall_temperature_K": helium["temperature_K"], "wall_accommodation": 1.0,
                          "inlet": {"model": "emit/surf-mflow", "port": geometry["helium_inlet"]["boundary_id"],
                                    "mass_flow_kg_s": helium["mass_flow_kg_s"],
                                    "temperature_K": helium["temperature_K"],
                                    "return_action": helium["inlet_return_action"]},
                          "fill": {"number_density_m3": helium["mixture_number_density_m3"],
                                   "temperature_K": helium["temperature_K"], "initial_state": initial},
                          "provenance": helium["scope"]},
               "numerics": n, "molecules": transport["molecules"],
               "observables": transport["observables"]}
    prepared = {"schema_version": "dsmc-prepared-v1", "case_id": transport["case_id"],
                "science_inputs": science, "assessment": {"state": "unassessed"},
                "source_case": _identity(transport_path),
                "geometry_artifact": {**_identity(out / "geometry.json"), "path": "geometry.json"},
                "inputs": inputs,
                "declared_outputs": [
                    {"role": "helium_field", "pattern": "fields.*.gz", "required": True},
                    {"role": "helium_restart", "pattern": "cad.restart.*", "required": True},
                    {"role": "helium_final_restart", "pattern": "final.restart.*", "required": True},
                    {"role": "helium_particles", "pattern": "particles.*", "required": False}],
                "preparation_tools": [_identity(Path(__file__)),
                                      _identity(ROOT / "tools/dsmc_cad_field.py")],
                "state": "solver-checked",
                "solver_check": {"status": "complete",
                                 "triangle_roles_sha256": _identity(out / "triangle-roles.json")["sha256"],
                                 "solver": receipt["solver"],
                                 "source_solver_sha256": receipt["solver"]["sha256"]},
                "cad_workflow": {"schema_version": BRIDGE_VERSION,
                                 "checked_geometry_sha256": _identity(out / "geometry.json")["sha256"],
                                 "checked_solver_receipt_sha256": _identity(out / "geometry-solver-check.json")["sha256"],
                                 "geometry_source_case_sha256": _identity(out / "geometry-case.json")["sha256"],
                                 "helium_config_sha256": _identity(out / "cad-helium-config.json")["sha256"],
                                 "transport_config_sha256": _identity(out / "cad-transport-config.json")["sha256"],
                                 "original_geometry_roles_sha256": _identity(out / "triangle-roles.geometry-only.json")["sha256"],
                                 "canonical_surface_sha256": receipt["canonical_surface_sha256"],
                                 "source_solver_sha256": receipt["solver"]["sha256"],
                                 "deck_balance": "rcb-cell-rendered-for-four-ranks",
                                 "initial_condition": initial}}
    _json_write(out / "prepared.json", prepared)


def prepare_cad_workflow(checked: Path, helium_config: Path,
                         transport_config: Path, out: Path) -> Path:
    checked, helium_config, transport_config, out = map(lambda p: Path(p).resolve(),
                                                        (checked, helium_config, transport_config, out))
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"CAD workflow output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-", dir=out.parent) as temp:
        stage = Path(temp) / "prepared"
        _prepare(checked, helium_config, transport_config, stage)
        validate_prepared_cad(stage)
        stage.rename(out)
    return out / "prepared.json"


def validate_prepared_cad(prepared_dir: Path) -> dict:
    """Check bridge-specific CAD identities and laws from staged bytes only."""
    from dsmc_case import inspect_geometry

    path = Path(prepared_dir)
    if path.is_file():
        path = path.parent
    prepared = _read_json(path / "prepared.json")
    meta = prepared.get("cad_workflow", {})
    if (prepared.get("schema_version") != "dsmc-prepared-v1" or
            prepared.get("science_inputs", {}).get("geometry", {}).get("adapter") != "cad" or
            prepared.get("state") != "solver-checked" or
            meta.get("schema_version") != BRIDGE_VERSION):
        raise ValueError("not a checked CAD workflow preparation")
    inspect_geometry(path)
    for key, name in (("checked_geometry_sha256", "geometry.json"),
                      ("checked_solver_receipt_sha256", "geometry-solver-check.json"),
                      ("geometry_source_case_sha256", "geometry-case.json"),
                      ("helium_config_sha256", "cad-helium-config.json"),
                      ("transport_config_sha256", "cad-transport-config.json"),
                      ("original_geometry_roles_sha256", "triangle-roles.geometry-only.json"),
                      ("canonical_surface_sha256", "physical-walls.surf")):
        _same_file(path / name, {"sha256": meta.get(key)}, f"CAD bridge {name}")
    geometry = _read_json(path / "geometry.json")
    receipt = _read_json(path / "geometry-solver-check.json")
    source = _read_json(path / "geometry-case.json")
    helium = _read_json(path / "cad-helium-config.json")
    transport = _read_json(path / "cad-transport-config.json")
    _validate_field_config(helium)
    _validate_transport(transport, geometry, source["case_id"])
    if (receipt.get("status") != "complete" or
            receipt.get("canonical_surface_sha256") != meta["canonical_surface_sha256"] or
            receipt.get("solver", {}).get("sha256") != meta["source_solver_sha256"] or
            prepared["source_case"]["sha256"] != meta["transport_config_sha256"] or
            prepared["solver_check"].get("solver", {}).get("sha256") != meta["source_solver_sha256"] or
            prepared["solver_check"].get("status") != "complete" or
            prepared["science_inputs"]["numerics"]["restart_prefix"] != "cad.restart" or
            prepared["science_inputs"]["numerics"]["field_every_steps"] !=
            helium.get("field_frame_every_steps", helium["steps"]) or
            prepared["science_inputs"]["helium"]["inlet"]["return_action"] != helium["inlet_return_action"] or
            prepared["science_inputs"]["molecules"] != transport["molecules"] or
            prepared["science_inputs"]["observables"] != transport["observables"] or
            prepared["science_inputs"]["model"] != transport["model"] or
            prepared["science_inputs"]["geometry"]["boundaries"] != {
                name: {**spec, "molecule_action": transport.get("boundary_actions", {}).get(
                    name, spec["molecule_action"])} for name, spec in geometry["boundaries"].items()} or
            prepared["science_inputs"]["geometry"]["solid_boundary_ids"] != list(geometry["boundaries"]) or
            prepared["science_inputs"]["geometry"]["gas_reference_point_m"] != transport["molecules"]["source"]["center_m"]):
        raise ValueError("CAD bridge source, solver or runtime laws differ")
    deck = (path / "in.helium").read_text(encoding="utf-8")
    if (deck.count(f"restart {transport['restart_every_steps']} cad.restart.%.*\n") != 1 or
            deck.count("write_restart final.restart.%\n") != 1 or
            deck.count(f"run {helium['steps']}\n") != 1 or
            deck.count("dump dfields grid all "
                       f"{helium.get('field_frame_every_steps', helium['steps'])} fields.*.gz ") != 1):
        raise ValueError("CAD helium deck differs from the declared field/restart schedule")
    initial = meta.get("initial_condition", {})
    if initial.get("kind") == "continuation":
        step = helium["continuation"]["parent_steps"]
        staged_parent = _read_json(path / "continuation-parent-config.json")
        staged_run = _read_json(path / "continuation-parent-run.json")
        domains = () if (staged_run.get("geometry_sha256") == meta["checked_geometry_sha256"] and
                         staged_run.get("solver_check_sha256") ==
                         meta["checked_solver_receipt_sha256"]) else (
            _run_domain(path / "continuation-parent-deck.in", staged_run),
            _geometry_domain(geometry, meta["canonical_surface_sha256"]))
        if _continuation_physics_differs(staged_parent, helium, *domains):
            raise ValueError("CAD staged continuation parent does not match this physical case")
        _check_initial_chain(
            {"parent_snapshot": initial.get("converted"), "preclone": initial.get("preclone"),
             "initial_particle_sha256": _identity(path / f"initial-particles.{step}")["sha256"]},
            _identity(path / "in.helium")["sha256"], _needs_preclone(staged_parent, helium))
    roles = _read_json(path / "triangle-roles.json")
    original_roles = _read_json(path / "triangle-roles.geometry-only.json")
    expected_records = copy.deepcopy(original_roles.get("records"))
    for row in expected_records:
        row["molecule_action"] = prepared["science_inputs"]["geometry"]["boundaries"][row["boundary_id"]]["molecule_action"]
    if (roles.get("schema_version") != "dsmc-triangle-roles-v1" or
            roles.get("records") != expected_records or
            roles.get("surface_sha256") != meta["canonical_surface_sha256"] or
            roles.get("physical_laws") != {"helium_config_sha256": meta["helium_config_sha256"],
                                           "transport_config_sha256": meta["transport_config_sha256"]}):
        raise ValueError("CAD runnable roles differ from checked canonical triangles")
    return prepared


IMPORT_VERSION = "dsmc-cad-helium-import-v1"


def _pilot_file(pilot: Path, recorded: str, expected: str) -> Path:
    """Use a receipt's name, never its historical absolute path, for copying."""
    if (type(recorded) is not str or ".." in recorded.replace("\\", "/").split("/") or
            Path(recorded.replace("\\", "/")).name != expected or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", expected)):
        raise ValueError(f"unsafe or unexpected CAD pilot output path: {recorded!r}")
    path = pilot / expected
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"CAD pilot output missing: {expected}")
    return path


def _bound_pilot(prepared_dir: Path, pilot_dir: Path) -> tuple[dict, dict, dict, dict]:
    from dsmc_frames import validate_frame

    prepared = validate_prepared_cad(prepared_dir)
    record = _read_json(pilot_dir / "run.json")
    config = _read_json(prepared_dir / "cad-helium-config.json")
    geometry = _read_json(prepared_dir / "geometry.json")
    receipt = _read_json(prepared_dir / "geometry-solver-check.json")
    if (record.get("schema_version") != "dsmc-cad-helium-field-pilot-result-v1" or
            record.get("status") != "complete" or
            record.get("case_id") != prepared["case_id"] or
            record.get("geometry_case_id") != config["geometry_case_id"] or
            record.get("config_sha256") != _identity(prepared_dir / "cad-helium-config.json")["sha256"] or
            record.get("geometry_sha256") != _identity(prepared_dir / "geometry.json")["sha256"] or
            record.get("solver_check_sha256") != _identity(prepared_dir / "geometry-solver-check.json")["sha256"] or
            record.get("canonical_surface_sha256") != _identity(prepared_dir / "physical-walls.surf")["sha256"] or
            record.get("solver_sha256") != receipt["solver"]["sha256"] or
            record.get("inlet_boundary_id") != geometry["helium_inlet"]["boundary_id"] or
            record.get("inlet_return_action") != config["inlet_return_action"] or
            record.get("steps") != config["steps"] or
            record.get("field_sample_window_steps") !=
            config["field_sample_every_steps"] * config["field_sample_count"]):
        raise ValueError("CAD pilot is not the completed source for this prepared case")
    local_duration = config["steps"] * config["timestep_s"]
    if not math.isclose(record.get("elapsed_physical_s", float("nan")), local_duration,
                        rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError("CAD pilot local physical time differs from its source configuration")
    if "continuation" in config:
        cont = record.get("continuation", {})
        expected = config["continuation"]
        if (cont.get("parent_case_id") != expected["parent_case_id"] or
                cont.get("parent_run_sha256") != expected["parent_run_sha256"] or
                cont.get("parent_snapshot", {}).get("sha256") != expected["parent_snapshot_sha256"] or
                cont.get("initial_particle_sha256") !=
                _identity(prepared_dir / f"initial-particles.{expected['parent_steps']}")["sha256"] or
                not math.isclose(record.get("cumulative_elapsed_physical_s", float("nan")),
                                 cont.get("parent_elapsed_physical_s", float("nan")) + local_duration,
                                 rel_tol=1e-10, abs_tol=1e-15)):
            raise ValueError("CAD pilot continuation ancestry or cumulative time differs")
        # The pilot pre-clones onto its own deck; the staged file must be the same bytes.
        _check_initial_chain(cont, record.get("deck_sha256"),
                             "preclone" in prepared["cad_workflow"]["initial_condition"])
    elif config.get("initial_fill", {}).get("model") == "cad-connected-cell-gas":
        if (record.get("cad_prefill") != _read_json(prepared_dir / "cad-prefill.json") or
                _identity(prepared_dir / "initial-particles.0")["sha256"] !=
                record["cad_prefill"].get("initial_dump_sha256")):
            raise ValueError("CAD pilot prefill ancestry differs")
    elif "initial_fill" in config:
        if record.get("initial_condition") != config["initial_fill"]:
            raise ValueError("CAD pilot analytic initial state differs")
    elif record.get("initial_condition") != "empty-domain":
        raise ValueError("CAD pilot initial state differs")
    for name, key in (("solver.log", "log_sha256"), ("in.helium-field", "deck_sha256")):
        if _identity(_pilot_file(pilot_dir, name, name))["sha256"] != record.get(key):
            raise ValueError(f"CAD pilot {name} changed")
    frame_name = f"fields.{record['steps']}.gz"
    frame_path = _pilot_file(pilot_dir, record.get("frame", {}).get("path"), frame_name)
    frame = validate_frame(frame_path, expected_step=record["steps"])
    if any(frame[key] != record["frame"].get(key)
           for key in ("timestep", "bytes", "rows", "sha256", "box_bounds_m")):
        raise ValueError("CAD pilot selected frame differs from completed receipt")
    if frame["box_bounds_m"] != prepared["science_inputs"]["numerics"]["domain_bounds_m"]:
        raise ValueError("CAD pilot selected frame box differs from prepared case")
    snap = record.get("final_particle_snapshot", {})
    snap_name = f"particles.{record['steps']}"
    snap_path = _pilot_file(pilot_dir, snap.get("path"), snap_name)
    if (snap.get("timestep") != record["steps"] or
            snap.get("particles") != record.get("final_particles") or
            _identity(snap_path)["sha256"] !=
            snap.get("sha256")):
        raise ValueError("CAD pilot final particle snapshot differs")
    _inspect_particle_dump(snap_path, record["steps"], record["final_particles"],
                           geometry["solver_probe"]["domain_bounds_m"])
    return prepared, record, config, frame


def _copy_import(source: Path, out: Path, name: str, outputs: list[dict]) -> None:
    destination = out / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"duplicate imported CAD output {name}")
    shutil.copyfile(source, destination)
    outputs.append({"path": name.replace("\\", "/"), "bytes": destination.stat().st_size,
                    "sha256": _identity(destination)["sha256"]})


def _import_in_dir(prepared_dir: Path, pilot_dir: Path, out: Path) -> None:
    prepared, record, config, frame = _bound_pilot(prepared_dir, pilot_dir)
    out.mkdir()
    outputs: list[dict] = []
    for source, name in ((prepared_dir / "prepared.json", "prepared.json"),
                         (prepared_dir / "geometry.json", "source-geometry.json"),
                         (prepared_dir / "geometry-solver-check.json", "source-solver-check.json"),
                         (prepared_dir / "cad-helium-config.json", "source-helium-config.json"),
                         (prepared_dir / "physical-walls.surf", "source-physical-walls.surf"),
                         (pilot_dir / "run.json", "source-run.json"),
                         (pilot_dir / "in.helium-field", "source-deck.in"),
                         (pilot_dir / "solver.log", "source-solver.log"),
                         (pilot_dir / f"fields.{record['steps']}.gz", f"fields.{record['steps']}.gz"),
                         (pilot_dir / f"particles.{record['steps']}", f"particles.{record['steps']}")):
        _copy_import(source, out, name, outputs)
    # These names come from fixed pilot output contracts, never from arbitrary
    # recorded paths. Their checksums are already bound by the pilot receipt.
    ancillary = []
    if "body_inventory" in record:
        ancillary.append((record["body_inventory"], "body-inventory.out"))
    for name, identity in record.get("domain_balance", {}).items():
        if name not in ("boundary.out", "domain-inventory.out"):
            raise ValueError("unsupported CAD pilot balance output")
        ancillary.append((identity, name))
    if "exit_station" in record:
        station = record["exit_station"]
        ancillary.extend((({"path": name, "sha256": station[key]}, name) for name, key in (
            ("exit-fwd.surf", "fwd_surface_sha256"),
            ("exit-rev.surf", "rev_surface_sha256"),
            ("exit-fwd.out", "fwd_dump_sha256"),
            ("exit-rev.out", "rev_dump_sha256"))))
    for identity, name in ancillary:
        source = _pilot_file(pilot_dir, identity.get("path"), name)
        if _identity(source)["sha256"] != identity.get("sha256"):
            raise ValueError(f"CAD pilot diagnostic changed: {name}")
        _copy_import(source, out, name, outputs)
    ancestry = []
    for item in prepared["inputs"]:
        if item["role"] not in ("initial_particles", "continuation_parent_receipt",
                                "continuation_parent_config", "continuation_parent_particles",
                                "prefill_receipt", "prefill_deck"):
            continue
        name = item["staged"]["path"]
        if Path(name).name != name:
            raise ValueError("CAD preparation ancestry path is not flat")
        _copy_import(prepared_dir / name, out, f"ancestry/{name}", outputs)
        ancestry.append({"role": item["role"], "path": f"ancestry/{name}",
                         "sha256": item["staged"]["sha256"]})
    window = record["field_sample_window_steps"]
    offset = record.get("continuation", {}).get("parent_elapsed_physical_s", 0.0)
    sample_window = {
        "timestep": record["steps"], "local_start_step_exclusive": record["steps"] - window,
        "local_end_step": record["steps"],
        "samples": config["field_sample_count"],
        "stride_steps": config["field_sample_every_steps"],
        "period_steps": window,
        "cumulative_start_s": offset + (record["steps"] - window) * config["timestep_s"],
        "cumulative_end_s": record.get("cumulative_elapsed_physical_s", record["elapsed_physical_s"]),
    }
    manifest = {
        "schema_version": IMPORT_VERSION, "status": "complete", "exit_code": 0,
        "origin": {"kind": "imported-completed-CAD-field-pilot",
                   "source_pilot_original_path": str(pilot_dir)},
        "case_id": prepared["case_id"],
        "prepared": {"path": "prepared.json", "sha256": _identity(out / "prepared.json")["sha256"]},
        "source_pilot": {"receipt_path": "source-run.json",
                         "receipt_sha256": _identity(out / "source-run.json")["sha256"],
                         "config_sha256": record["config_sha256"],
                         "geometry_sha256": record["geometry_sha256"],
                         "solver_check_sha256": record["solver_check_sha256"],
                         "canonical_surface_sha256": record["canonical_surface_sha256"],
                         "solver_sha256": record["solver_sha256"],
                         "frame_sha256": frame["sha256"]},
        "end_step": record["steps"], "elapsed_physical_s": record["elapsed_physical_s"],
        "cumulative_elapsed_physical_s": record.get("cumulative_elapsed_physical_s",
                                                   record["elapsed_physical_s"]),
        "frame_windows": [sample_window],
        "restart": {"available": False, "kind": "particle-snapshot-only"},
        "ancestry": ancestry,
        "outputs": outputs,
    }
    _json_write(out / "manifest.json", manifest)
    validate_imported_cad(out, manifest)


def bind_cad_helium(prepared: Path, pilot_result: Path, out: Path) -> Path:
    """Import a completed CAD pilot without modifying or relabeling its source."""
    prepared_dir = Path(prepared).resolve()
    if prepared_dir.is_file():
        if prepared_dir.name != "prepared.json":
            raise ValueError("CAD import needs prepared.json or its directory")
        prepared_dir = prepared_dir.parent
    pilot_dir = Path(pilot_result).resolve()
    if pilot_dir.is_file():
        if pilot_dir.name != "run.json":
            raise ValueError("CAD import needs pilot run.json or its directory")
        pilot_dir = pilot_dir.parent
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"CAD import output already exists: {out}")
    if any(out == root or out in root.parents or root in out.parents
           for root in (prepared_dir, pilot_dir)):
        raise ValueError("CAD import output must be separate from its sources")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-import-", dir=out.parent) as temp:
        stage = Path(temp) / "imported"
        _import_in_dir(prepared_dir, pilot_dir, stage)
        stage.rename(out)
    return out / "manifest.json"


def validate_imported_cad(parent: Path, record: dict | None = None) -> dict:
    """Validate an imported parent using only files below its relocated root."""
    from dsmc_frames import validate_frame

    parent = Path(parent).resolve()
    manifest_path = parent / "manifest.json"
    data = _read_json(manifest_path) if record is None else record
    if record is not None and _read_json(manifest_path) != record:
        raise ValueError("imported CAD manifest changed after reading")
    if (data.get("schema_version") != IMPORT_VERSION or data.get("status") != "complete" or
            data.get("exit_code") != 0 or
            data.get("origin", {}).get("kind") != "imported-completed-CAD-field-pilot" or
            data.get("restart") != {"available": False, "kind": "particle-snapshot-only"}):
        raise ValueError("invalid imported CAD parent status or provenance")
    outputs = data.get("outputs")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("imported CAD parent lacks output inventory")
    seen = set()
    for item in outputs:
        name = item.get("path")
        if (type(name) is not str or name in seen or
                Path(name).is_absolute() or ".." in name.replace("\\", "/").split("/") or
                name.replace("\\", "/") != name):
            raise ValueError("unsafe or duplicate imported CAD inventory path")
        seen.add(name)
        source = (parent / name).resolve()
        if (parent not in source.parents or not source.is_file() or source.is_symlink() or
                source.stat().st_size != item.get("bytes") or
                _identity(source)["sha256"] != item.get("sha256")):
            raise ValueError(f"imported CAD output changed: {name}")
    required = {"prepared.json", "source-run.json", "source-geometry.json",
                "source-solver-check.json", "source-helium-config.json",
                "source-physical-walls.surf", "source-deck.in", "source-solver.log"}
    if not required <= seen:
        raise ValueError("imported CAD parent lacks a source binding")
    prepared = _read_json(parent / "prepared.json")
    receipt = _read_json(parent / "source-run.json")
    config = _read_json(parent / "source-helium-config.json")
    source = data.get("source_pilot", {})
    meta = prepared.get("cad_workflow", {})
    if (data.get("case_id") != prepared.get("case_id") or
            data.get("prepared", {}).get("sha256") != _identity(parent / "prepared.json")["sha256"] or
            source.get("receipt_sha256") != _identity(parent / "source-run.json")["sha256"] or
            source.get("config_sha256") != _identity(parent / "source-helium-config.json")["sha256"] or
            source.get("geometry_sha256") != _identity(parent / "source-geometry.json")["sha256"] or
            source.get("solver_check_sha256") != _identity(parent / "source-solver-check.json")["sha256"] or
            source.get("canonical_surface_sha256") != _identity(parent / "source-physical-walls.surf")["sha256"] or
            source.get("solver_sha256") != _read_json(parent / "source-solver-check.json")["solver"]["sha256"] or
            meta.get("checked_geometry_sha256") != source.get("geometry_sha256") or
            meta.get("checked_solver_receipt_sha256") != source.get("solver_check_sha256") or
            meta.get("helium_config_sha256") != source.get("config_sha256") or
            meta.get("canonical_surface_sha256") != source.get("canonical_surface_sha256") or
            receipt.get("config_sha256") != source["config_sha256"] or
            receipt.get("geometry_sha256") != source["geometry_sha256"] or
            receipt.get("solver_check_sha256") != source["solver_check_sha256"] or
            receipt.get("canonical_surface_sha256") != source["canonical_surface_sha256"] or
            receipt.get("solver_sha256") != source["solver_sha256"] or
            receipt.get("deck_sha256") != _identity(parent / "source-deck.in")["sha256"] or
            receipt.get("log_sha256") != _identity(parent / "source-solver.log")["sha256"] or
            receipt.get("schema_version") != "dsmc-cad-helium-field-pilot-result-v1" or
            receipt.get("status") != "complete" or
            receipt.get("case_id") != data["case_id"] or
            receipt.get("steps") != data.get("end_step") or
            config.get("steps") != data.get("end_step")):
        raise ValueError("imported CAD source identities differ")
    frame_name = f"fields.{data['end_step']}.gz"
    if frame_name not in seen:
        raise ValueError("imported CAD parent lacks selected frame")
    frame = validate_frame(parent / frame_name, expected_step=data["end_step"])
    if (frame["sha256"] != source.get("frame_sha256") or
            frame["sha256"] != receipt.get("frame", {}).get("sha256") or
            frame["rows"] != receipt["frame"].get("rows") or
            frame["box_bounds_m"] != receipt["frame"].get("box_bounds_m") or
            frame["box_bounds_m"] != prepared["science_inputs"]["numerics"]["domain_bounds_m"]):
        raise ValueError("imported CAD selected frame differs")
    snapshot = f"particles.{data['end_step']}"
    if (snapshot not in seen or
            _identity(parent / snapshot)["sha256"] !=
            receipt.get("final_particle_snapshot", {}).get("sha256")):
        raise ValueError("imported CAD final particle snapshot differs")
    _inspect_particle_dump(parent / snapshot, data["end_step"],
                           receipt["final_particle_snapshot"]["particles"],
                           prepared["science_inputs"]["numerics"]["domain_bounds_m"])
    diagnostics = {}
    if "body_inventory" in receipt:
        diagnostics["body-inventory.out"] = receipt["body_inventory"]["sha256"]
    for name, item in receipt.get("domain_balance", {}).items():
        if name not in ("boundary.out", "domain-inventory.out"):
            raise ValueError("unsupported imported CAD balance diagnostic")
        diagnostics[name] = item["sha256"]
    if "exit_station" in receipt:
        station = receipt["exit_station"]
        diagnostics.update({name: station[key] for name, key in (
            ("exit-fwd.surf", "fwd_surface_sha256"),
            ("exit-rev.surf", "rev_surface_sha256"),
            ("exit-fwd.out", "fwd_dump_sha256"),
            ("exit-rev.out", "rev_dump_sha256"))})
    if any(name not in seen or _identity(parent / name)["sha256"] != digest
           for name, digest in diagnostics.items()):
        raise ValueError("imported CAD diagnostic differs from original pilot")
    windows = data.get("frame_windows", [])
    if (len(windows) != 1 or windows[0].get("timestep") != data["end_step"] or
            windows[0].get("period_steps") != receipt.get("field_sample_window_steps") or
            windows[0].get("local_start_step_exclusive") !=
            data["end_step"] - receipt["field_sample_window_steps"] or
            windows[0].get("samples") != config.get("field_sample_count") or
            windows[0].get("stride_steps") != config.get("field_sample_every_steps") or
            not math.isclose(windows[0].get("cumulative_start_s", float("nan")),
                             receipt.get("continuation", {}).get("parent_elapsed_physical_s", 0.0) +
                             (data["end_step"] - receipt["field_sample_window_steps"]) *
                             config["timestep_s"], rel_tol=1e-10, abs_tol=1e-15) or
            not math.isclose(windows[0].get("cumulative_end_s", float("nan")),
                             receipt.get("cumulative_elapsed_physical_s",
                                         receipt["elapsed_physical_s"]),
                             rel_tol=1e-10, abs_tol=1e-15)):
        raise ValueError("imported CAD sample window differs from original pilot")
    expected_ancestry = {(item["role"], f"ancestry/{item['staged']['path']}",
                          item["staged"]["sha256"])
                         for item in prepared["inputs"] if item["role"] in (
                             "initial_particles", "continuation_parent_receipt",
                             "continuation_parent_config", "continuation_parent_particles",
                             "prefill_receipt", "prefill_deck")}
    actual_ancestry = {(item.get("role"), item.get("path"), item.get("sha256"))
                       for item in data.get("ancestry", [])}
    if actual_ancestry != expected_ancestry:
        raise ValueError("imported CAD ancestry inventory differs from preparation")
    for item in data.get("ancestry", []):
        if (item.get("path") not in seen or
                _identity(parent / item["path"])["sha256"] != item.get("sha256")):
            raise ValueError("imported CAD ancestry differs")
    return data
