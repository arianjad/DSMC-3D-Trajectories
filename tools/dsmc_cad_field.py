#!/usr/bin/env python3
"""Run a transient CAD helium pilot and save an averaged grid frame.

The initial domain can be empty, analytically prefilled, or continued from a
checked particle snapshot. A completed frame is structural evidence, not
proof of steady state, grid convergence, or a molecular-beam prediction.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from dsmc_cad_flow import (ROOT, _deck, _sha, _validate_config,
                           _validate_geometry, _write_json)
from dsmc_frames import validate_frame
from equilibrium import read_dump
from preclone_particles import preclone
from revolve_surf import Mesh, diagnostic_disk, write_surface

# Blocks must hold whole 10-step samples: the 2026-09-28 totals cadence (Arian,
# 2026-09-27 run plan, "Totals cadence"). The totals are now derived
# from the regional moments, so they sample on diagnostics.sample_every_steps (10 in
# every current config) and this rule is kept only as the recorded config gate.
TOTALS_SAMPLE_EVERY = 10
# The one ave/grid fix whose 12 columns every regional dump reads.
REGIONAL_FIX = "adiagunion"
KB_SI = 1.380649e-23  # J/K, SPARTA's boltz for units si (update.cpp:157 at 27Aug2026)


def _validate_diagnostics(diagnostics: object, steps: int) -> dict:
    """Validate opt-in block diagnostics without changing normal field output."""
    if not isinstance(diagnostics, dict) or set(diagnostics) != {
            "schema_version", "sample_every_steps", "block_every_steps", "regions"}:
        raise ValueError("unsupported CAD regional diagnostics")
    if diagnostics["schema_version"] != "dsmc-cad-regional-diagnostics-v1":
        raise ValueError("unsupported CAD regional diagnostics schema")
    for key in ("sample_every_steps", "block_every_steps"):
        if type(diagnostics[key]) is not int or diagnostics[key] <= 0:
            raise ValueError(f"diagnostics.{key} must be a positive integer")
    sample, block = diagnostics["sample_every_steps"], diagnostics["block_every_steps"]
    if block % sample or steps % block:
        raise ValueError("diagnostic sampling and blocks must divide the run")
    if block % TOTALS_SAMPLE_EVERY:
        raise ValueError(f"diagnostics.block_every_steps must be a multiple of "
                         f"{TOTALS_SAMPLE_EVERY}: regional totals sample every "
                         f"{TOTALS_SAMPLE_EVERY}th step within each block")
    regions = diagnostics["regions"]
    if not isinstance(regions, list) or not regions:
        raise ValueError("diagnostics.regions must be a nonempty list")
    names = set()
    for region in regions:
        if not isinstance(region, dict) or set(region) != {"name", "radius_m", "z_range_m"}:
            raise ValueError("unsupported diagnostic region")
        name = region["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", name):
            raise ValueError("diagnostic region name is invalid")
        if name in names:
            raise ValueError("diagnostic region names must be distinct")
        names.add(name)
        if (type(region["radius_m"]) not in (int, float) or
                not math.isfinite(region["radius_m"]) or region["radius_m"] <= 0):
            raise ValueError("diagnostic region radius is invalid")
        zrange = region["z_range_m"]
        if (not isinstance(zrange, list) or len(zrange) != 2 or
                any(type(value) not in (int, float) or not math.isfinite(value)
                    for value in zrange) or zrange[0] >= zrange[1]):
            raise ValueError("diagnostic region z range is invalid")
    return diagnostics


def _validate_diagnostic_geometry(diagnostics: dict, geometry: dict) -> None:
    bounds = geometry["solver_probe"]["domain_bounds_m"]
    radial_limit = min(bounds[0][1], -bounds[0][0], bounds[1][1], -bounds[1][0])
    for region in diagnostics["regions"]:
        zlo, zhi = region["z_range_m"]
        if (region["radius_m"] > radial_limit or zlo < bounds[2][0] or
                zhi > bounds[2][1]):
            raise ValueError("diagnostic region must lie inside the simulation domain")


def _validate_field_config(data: dict) -> dict:
    extra = {"nominal_sccm", "atoms_per_sccm_s", "helium_atom_mass_kg",
             "field_sample_every_steps", "field_sample_count"}
    required = {"schema_version", "case_id", "geometry_case_id", "scope",
                     "seed", "mass_flow_kg_s", "temperature_K",
                     "mixture_number_density_m3", "fnum_real_per_sim_particle",
                     "timestep_s", "steps", "stats_every_steps",
                     "inlet_return_action"} | extra
    if set(data) - {"initial_fill", "exit_station", "domain_balance",
                     "continuation", "diagnostics", "grid_refinement",
                     "field_frame_every_steps"} != required:
        raise ValueError("unsupported CAD helium field configuration")
    if data["schema_version"] != "dsmc-cad-helium-field-pilot-v1":
        raise ValueError("unsupported CAD helium field schema")
    base = {key: value for key, value in data.items()
            if key not in extra | {"initial_fill", "exit_station", "domain_balance",
                                   "continuation", "diagnostics", "field_frame_every_steps"}}
    base["schema_version"] = "dsmc-cad-helium-smoke-v1"
    _validate_config(base)
    for key in ("nominal_sccm", "atoms_per_sccm_s", "helium_atom_mass_kg"):
        value = data[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    expected = (data["nominal_sccm"] * data["atoms_per_sccm_s"] *
                data["helium_atom_mass_kg"])
    if not math.isclose(data["mass_flow_kg_s"], expected, rel_tol=1e-8):
        raise ValueError("mass_flow_kg_s disagrees with the declared SCCM conversion")
    for key in ("field_sample_every_steps", "field_sample_count"):
        if type(data[key]) is not int or data[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    window = data["field_sample_every_steps"] * data["field_sample_count"]
    if window > data["steps"] or data["steps"] % window:
        raise ValueError("field sample window must divide the run length")
    if "field_frame_every_steps" in data:
        # Opt-in intermediate frames: each dump must land on a completed ave/grid window.
        every = data["field_frame_every_steps"]
        if (type(every) is not int or every <= 0 or every % window or
                data["steps"] % every):
            raise ValueError("field_frame_every_steps must be a multiple of the field "
                             "sample window and divide the run length")
    if "initial_fill" in data:
        fill = data["initial_fill"]
        old = {"model", "number_density_m3", "region_radius_m",
               "region_z_range_m"}
        cad = {"model", "number_density_m3", "body_diagnostic_radius_m",
               "body_diagnostic_z_range_m", "solid_step_sha256", "prefill_path"}
        model = fill.get("model") if isinstance(fill, dict) else None
        if model == "effusive-aperture-starting-estimate":
            required = old
            optional = {"aperture_radius_m", "density_basis"}
        elif model == "cad-connected-cell-gas":
            required = cad
            optional = {"aperture_radius_m", "density_basis", "prefill_source_case_id"}
        else:
            raise ValueError("unsupported initial fill")
        if not required <= set(fill) or set(fill) - (required | optional):
            raise ValueError("unsupported initial fill")
        if ("density_basis" in fill and
                (not isinstance(fill["density_basis"], str) or
                 not fill["density_basis"].strip())):
            raise ValueError("initial_fill.density_basis must be a nonempty string")
        radius_key = ("body_diagnostic_radius_m" if fill["model"] == "cad-connected-cell-gas"
                      else "region_radius_m")
        range_key = ("body_diagnostic_z_range_m" if fill["model"] == "cad-connected-cell-gas"
                     else "region_z_range_m")
        numeric_keys = ["number_density_m3", radius_key]
        if "aperture_radius_m" in fill:
            numeric_keys.append("aperture_radius_m")
        for key in numeric_keys:
            value = fill[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"initial_fill.{key} must be finite and positive")
        zrange = fill[range_key]
        if (not isinstance(zrange, list) or len(zrange) != 2 or
                any(type(z) not in (int, float) or not math.isfinite(z) for z in zrange) or
                zrange[0] >= zrange[1]):
            raise ValueError("invalid initial fill z range")
        if fill["model"] == "cad-connected-cell-gas":
            if (not re.fullmatch(r"[0-9a-f]{64}", fill["solid_step_sha256"]) or
                    not isinstance(fill["prefill_path"], str) or
                    not fill["prefill_path"].startswith("generated/")):
                raise ValueError("CAD fill needs a pinned solid and generated particle path")
            if ("prefill_source_case_id" in fill and
                    (not isinstance(fill["prefill_source_case_id"], str) or
                     not fill["prefill_source_case_id"].strip())):
                raise ValueError("invalid prefill source case ID")
    if ("grid_refinement" in data and "continuation" not in data and
            data.get("initial_fill", {}).get("model") == "cad-connected-cell-gas"):
        # read_particles keeps weights: a saved level-1 prefill would start at 1/8^(k-1).
        raise ValueError("grid_refinement with a saved prefill is unsupported; "
                         "refine a continuation (pre-cloned) instead")
    if "domain_balance" in data and (data["domain_balance"] is not True or
                                     "initial_fill" not in data):
        raise ValueError("domain_balance requires an initial fill")
    if "continuation" in data:
        cont = data["continuation"]
        if (not isinstance(cont, dict) or set(cont) != {
                "parent_run_dir", "parent_config_path", "parent_case_id",
                "parent_run_sha256", "parent_snapshot_sha256", "parent_steps"} or
                data.get("initial_fill", {}).get("model") != "cad-connected-cell-gas" or
                data.get("domain_balance") is not True):
            raise ValueError("continuation needs a CAD-fill balance case and pinned parent")
        for key, prefix in (("parent_run_dir", "generated/"),
                            ("parent_config_path", "cases/")):
            if not isinstance(cont[key], str) or not cont[key].startswith(prefix):
                raise ValueError(f"continuation.{key} must be repository-relative")
        if not isinstance(cont["parent_case_id"], str) or not cont["parent_case_id"].strip():
            raise ValueError("continuation parent case ID is missing")
        for key in ("parent_run_sha256", "parent_snapshot_sha256"):
            if not isinstance(cont[key], str) or not re.fullmatch(r"[0-9a-f]{64}", cont[key]):
                raise ValueError(f"continuation.{key} must be a SHA-256 digest")
        if type(cont["parent_steps"]) is not int or cont["parent_steps"] <= 0:
            raise ValueError("continuation parent step must be positive")
    if "exit_station" in data:
        station = data["exit_station"]
        if (not isinstance(station, dict) or
                set(station) not in ({"z_m", "radius_m", "segments"},
                                     {"z_m", "radius_m", "segments", "sample_every_steps"})):
            raise ValueError("unsupported exit station")
        for key in ("z_m", "radius_m"):
            if type(station[key]) not in (int, float) or not math.isfinite(station[key]):
                raise ValueError(f"exit_station.{key} must be finite")
        if (station["radius_m"] <= 0 or type(station["segments"]) is not int or
                station["segments"] < 12):
            raise ValueError("invalid exit station radius or polygon")
        sample_every = station.get("sample_every_steps", 10)
        if (type(sample_every) is not int or sample_every <= 0 or
                1000 % sample_every):
            raise ValueError("exit station sample interval must divide 1000 steps")
    if "diagnostics" in data:
        _validate_diagnostics(data["diagnostics"], data["steps"])
    return base


def _field_deck(config: dict, geometry: dict, *, mpi_ranks: int = 1) -> str:
    if type(mpi_ranks) is not int or mpi_ranks < 1:
        raise ValueError("mpi_ranks must be a positive integer")
    deck = _deck(config, geometry)
    if mpi_ranks > 1:
        grid = re.findall(r"^create_grid .+$", deck, re.MULTILINE)
        if len(grid) != 1:
            raise ValueError("CAD source deck has an unexpected grid section")
        deck = deck.replace(grid[0] + "\n", grid[0] + "\nbalance_grid rcb cell\n")
    particle_dump = (f"dump state particle hegas {config['steps']} particles.* "
                     "x y z vx vy vz\n")
    if deck.count(particle_dump) != 1:
        raise ValueError("CAD source deck has an unexpected particle dump")
    deck = deck.replace(particle_dump,
                        particle_dump + "dump_modify state format float %.17g\n")
    if "initial_fill" in config:
        fill = config["initial_fill"]
        mixture_marker = "global fnum "
        collide_marker = "collide vss hegas he.vss\n"
        if deck.count(mixture_marker) != 1 or deck.count(collide_marker) != 1:
            raise ValueError("CAD source deck has an unexpected fill section")
        if "continuation" in config:
            step = config["continuation"]["parent_steps"]
            deck = deck.replace(collide_marker,
                                collide_marker +
                                f"read_particles initial-particles.{step} {step}\n")
        elif fill["model"] == "cad-connected-cell-gas":
            deck = deck.replace(collide_marker,
                                collide_marker + "read_particles initial-particles.0 0\n")
        else:
            deck = deck.replace(mixture_marker,
                                f"mixture fillgas He nrho {fill['number_density_m3']:.17g} "
                                f"temp {config['temperature_K']:.17g}\n" + mixture_marker)
            deck = deck.replace(collide_marker,
                                collide_marker +
                                f"region initcell cylinder z 0 0 {fill['region_radius_m']:.17g} "
                                f"{fill['region_z_range_m'][0]:.17g} "
                                f"{fill['region_z_range_m'][1]:.17g}\n"
                                "create_particles fillgas n 0 region initcell\n")
    if "exit_station" in config:
        collide_marker = "collide vss hegas he.vss\n"
        if deck.count(collide_marker) != 1:
            raise ValueError("CAD source deck has an unexpected collision section")
        station_block = (
            "surf_collide station transparent\n"
            "read_surf exit-fwd.surf type transparent group xf\n"
            "surf_modify xf collide station\n"
            "read_surf exit-rev.surf type transparent group xr\n"
            "surf_modify xr collide station\n")
        deck = deck.replace(collide_marker, collide_marker + station_block)
    if mpi_ranks > 1 and "initial_fill" in config:
        initialization = re.findall(r"^(?:read_particles|create_particles) .+$",
                                    deck, re.MULTILINE)
        if len(initialization) != 1:
            raise ValueError("CAD source deck has an unexpected particle initialization")
        deck = deck.replace(initialization[0] + "\n",
                            initialization[0] + "\nbalance_grid rcb part\n")
    marker = f"timestep {config['timestep_s']:.17g}\n"
    if deck.count(marker) != 1:
        raise ValueError("CAD source deck has an unexpected timestep section")
    nevery = config["field_sample_every_steps"]
    nrepeat = config["field_sample_count"]
    nfreq = nevery * nrepeat
    block = ("compute cfields grid all hegas n nrho u v w usq vsq wsq\n"
             "compute cthermal thermal/grid all hegas temp\n"
             f"fix afields ave/grid all {nevery} {nrepeat} {nfreq} "
             "c_cfields[*] c_cthermal[*] ave one\n"
             f"dump dfields grid all {config.get('field_frame_every_steps', config['steps'])} "
             "fields.*.gz "
             "idstr split xlo ylo zlo xhi yhi zhi vol f_afields[*]\n"
             "dump_modify dfields first no flush yes\n")
    if "initial_fill" in config:
        fill = config["initial_fill"]
        radius = fill.get("body_diagnostic_radius_m", fill.get("region_radius_m"))
        zrange = fill.get("body_diagnostic_z_range_m", fill.get("region_z_range_m"))
        block += ("compute ceqrho grid all hegas nrho\n"
                  "compute ceqvol property/grid all vol\n"
                  "variable eqinventory grid c_ceqrho[1]*c_ceqvol\n"
                  f"region eqbody cylinder z 0 0 {radius:.17g} "
                  f"{zrange[0]:.17g} "
                  f"{zrange[1]:.17g}\n"
                  "group eqbody grid region eqbody center\n"
                  "compute ceqbody reduce sum v_eqinventory subset eqbody\n"
                  "fix aeqbody ave/time 10 100 1000 c_ceqbody file body-inventory.out\n")
    if "exit_station" in config:
        sample_every = config["exit_station"].get("sample_every_steps", 10)
        block += ("group exitdiag surf union xf xr\n"
                  "compute cexit surf exitdiag hegas nflux norm flow\n"
                  f"fix aexit ave/surf exitdiag {sample_every} "
                  f"{1000 // sample_every} 1000 c_cexit[1]\n"
                  "dump dexf surf xf 1000 exit-fwd.out id f_aexit\n"
                  "dump_modify dexf first no flush yes\n"
                  "dump dexr surf xr 1000 exit-rev.out id f_aexit\n"
                  "dump_modify dexr first no flush yes\n")
    if config.get("domain_balance"):
        block += ("compute ceqall reduce sum v_eqinventory\n"
                  "fix aeqall ave/time 10 100 1000 c_ceqall file domain-inventory.out\n"
                  "compute ceqbound boundary hegas nflux\n"
                  "fix aeqbound ave/time 1 1000 1000 c_ceqbound[*] "
                  "mode vector file boundary.out\n")
    if "diagnostics" in config:
        diagnostics = config["diagnostics"]
        sample = diagnostics["sample_every_steps"]
        block_steps = diagnostics["block_every_steps"]
        block_count = block_steps // sample
        names = [region["name"] for region in diagnostics["regions"]]
        for region in diagnostics["regions"]:
            group = f"diag_{region['name']}"
            zlo, zhi = region["z_range_m"]
            block += (f"region {group} cylinder z 0 0 {region['radius_m']:.17g} "
                      f"{zlo:.17g} {zhi:.17g}\n"
                      f"group {group} grid region {group} center\n")
        # One accumulator over the union: each grid compute and ave/grid traverses the
        # whole local grid whatever its group (review 2026-09-28 finding 1). A cell's
        # value depends only on its own particles, so each region's dump is unchanged.
        # "diagunion" cannot collide with a "diag_<name>" region group.
        block += (f"group diagunion grid union {' '.join(f'diag_{n}' for n in names)}\n"
                  "compute cdiagunion_flow grid diagunion hegas n nrho u v w\n"
                  "compute cdiagunion_thermal thermal/grid diagunion hegas temp\n"
                  "compute cdiagunion_pflux pflux/grid diagunion hegas "
                  "momxx momyy momzz momxy momyz momxz\n"
                  f"fix {REGIONAL_FIX} ave/grid diagunion {sample} {block_count} {block_steps} "
                  "c_cdiagunion_flow[*] c_cdiagunion_thermal[*] c_cdiagunion_pflux[*] ave one\n")
        for name in names:
            block += (f"dump ddiag_{name} grid diag_{name} {block_steps} diagnostic-{name}.*.gz "
                      f"idstr split xlo ylo zlo xhi yhi zhi vol f_{REGIONAL_FIX}[*]\n"
                      f"dump_modify ddiag_{name} first no flush yes\n")
        # diagnostic-regional-totals.out is derived from these dumps after the run
        # (_write_regional_totals), not resampled by separate reductions.
        block += (f"variable vdiag_ncoll_block equal ncoll*{block_steps}\n"
                  f"variable vdiag_nattempt_block equal nattempt*{block_steps}\n"
                  f"fix adiag_collision ave/time 1 {block_steps} {block_steps} "
                  "v_vdiag_ncoll_block v_vdiag_nattempt_block ave one "
                  "file diagnostic-collision-history.out\n")
    return deck.replace(marker, block + marker)


def _field_summary(log: str, steps: int) -> dict:
    start = log.find("Step CPU Np f_src[1] f_src[2]")
    end = log.find("Loop time of", start)
    if start < 0 or end < 0:
        raise ValueError("SPARTA field pilot lacks a complete stats table")
    rows = {}
    for match in re.finditer(r"^\s*(\d+)\s+[\d.eE+-]+\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
                             log[start:end], re.MULTILINE):
        step, npart, inserted, cumulative = map(int, match.groups())
        rows[step] = (npart, inserted, cumulative)
    if 0 not in rows or steps not in rows or rows[steps][2] <= 0:
        raise ValueError("SPARTA field pilot lacks initial/final source counts")
    return {"initial_particles": rows[0][0], "final_particles": rows[steps][0],
            "emitted_final_step": rows[steps][1],
            "emitted_cumulative": rows[steps][2],
            "particle_stats": [{"step": step, "particles": row[0],
                                "emitted_cumulative": row[2]}
                               for step, row in sorted(rows.items())]}


def _diagnostic_frame(path: Path, step: int, fix: str) -> dict:
    """Check a complete opt-in regional grid block without accepting it as a field frame."""
    columns = ("idstr", "split", "xlo", "ylo", "zlo", "xhi", "yhi", "zhi", "vol",
               *(f"f_{fix}[{index}]" for index in range(1, 13)))
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            header = [next(source).strip() for _ in range(9)]
            if (header[0] != "ITEM: TIMESTEP" or int(header[1]) != step or
                    header[2] != "ITEM: NUMBER OF CELLS" or int(header[3]) <= 0 or
                    header[4] != "ITEM: BOX BOUNDS oo oo oo" or
                    header[8].split()[2:] != list(columns)):
                raise ValueError("diagnostic frame header differs from the rendered block")
            rows = sum(bool(line.strip()) for line in source)
            if rows != int(header[3]):
                raise ValueError("diagnostic frame row count differs from its header")
    except (EOFError, OSError, StopIteration, UnicodeError, ValueError) as exc:
        raise ValueError(f"incomplete diagnostic frame {path.name}: {exc}") from exc
    return {"path": path.name, "step": step, "rows": rows, "sha256": _sha(path)}


def _complete_block_history(path: Path, final_step: int, block_steps: int) -> dict:
    if not path.is_file():
        raise ValueError(f"SPARTA did not write {path.name}")
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]
    expected_steps = list(range(block_steps, final_step + 1, block_steps))
    if [int(row[0]) for row in rows] != expected_steps:
        raise ValueError(f"{path.name} lacks one or more complete diagnostic blocks")
    return {"path": path.name, "sha256": _sha(path), "blocks": len(rows)}


def _write_regional_totals(out: Path, diagnostics: dict, final_step: int) -> None:
    """Regional atom inventory and kinetic energy per block, from the pooled block moments.

    N_R = sum V n and E_R = sum V n (3/2 kB T + m |u|^2 / 2) over the region's dumped cells,
    with T SPARTA's raw thermal temperature (divides by N, compute_thermal_grid.cpp:249) and
    u, n, T from the same pooled window, this equals the block mean of the per-step
    sums of nrho*vol and kerho*vol the separate reductions used to sample on the same
    steps (review 2026-09-28 finding 2, recompute_totals.py). A cell with <= 1 pooled
    particle has T = 0, which is exact there. Values are the dumps' printed (%g) moments.
    The header and %g rows keep the fix ave/time file layout of the retired adiag_totals.
    """
    path = out / "diagnostic-regional-totals.out"
    if path.exists():
        raise ValueError(f"{path.name} already exists; it is derived after the run")
    mass = [float(line.split()[2]) for line in
            (out / "he.species").read_text(encoding="utf-8").splitlines()
            if line.split()[:1] == ["He"]]
    if len(mass) != 1:
        raise ValueError("he.species lacks one He row")
    names = [region["name"] for region in diagnostics["regions"]]
    lines = ["# Time-averaged data for fix adiag_totals",
             "# TimeStep " + " ".join(f"c_cdiag_{name}_inventory c_cdiag_{name}_energy_total"
                                      for name in names)]
    block = diagnostics["block_every_steps"]
    for step in range(block, final_step + 1, block):
        values = []
        for name in names:
            atoms = energy = 0.0
            with gzip.open(out / f"diagnostic-{name}.{step}.gz", "rt", encoding="utf-8") as source:
                for _ in range(9):
                    next(source)
                for row in source:
                    # idstr split 6-box vol | f[1..12] = n nrho u v w T momxx..momxz
                    items = row.split()
                    nrho, u, v, w, temp = map(float, items[10:15])
                    cell = float(items[8]) * nrho
                    atoms += cell
                    energy += cell * (1.5 * KB_SI * temp + 0.5 * mass[0] * (u * u + v * v + w * w))
            values += [atoms, energy]
        lines.append(f"{step} " + " ".join("%g" % value for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _regional_receipt(out: Path, config: dict) -> dict:
    """Check every regional block output of a completed run; return its receipt entry."""
    diagnostics = config["diagnostics"]
    block_steps = diagnostics["block_every_steps"]
    steps = range(block_steps, config["steps"] + 1, block_steps)
    fields = {region["name"]: [
        _diagnostic_frame(out / f"diagnostic-{region['name']}.{step}.gz", step, REGIONAL_FIX)
        for step in steps] for region in diagnostics["regions"]}
    _write_regional_totals(out, diagnostics, config["steps"])
    return {
        "definition": diagnostics,
        "blocks": config["steps"] // block_steps,
        "collision_history": _complete_block_history(
            out / "diagnostic-collision-history.out", config["steps"], block_steps),
        "regional_totals": {**_complete_block_history(
            out / "diagnostic-regional-totals.out", config["steps"], block_steps),
            "sample_every_steps": diagnostics["sample_every_steps"]},
        "fields": fields,
        "collision_history_semantics": (
            "nonoverlapping block totals derived from current-step ncoll and nattempt; "
            "not cumulative ncollave or nattemptave"),
        "regional_totals_semantics": (
            "derived after the run from each region's dumped block moments: "
            "sum over cells of vol*nrho and vol*nrho*(1.5 kB T + 0.5 m |u|^2), raw T, "
            "same pooled window and gas volume; equals the block mean of the per-step "
            "nrho*vol and kerho*vol sums, i.e. physical kinetic energy including directed "
            "flow, to the dumps' printed precision"),
    }


def _write_exit_disks(out: Path, station: dict) -> tuple[Path, Path]:
    disk = diagnostic_disk(station["z_m"], station["radius_m"],
                           station["segments"], 1, (-1.0, 0.0, 0.0), 1e-12)
    forward = Mesh(tuple((point[1], point[2], point[0]) for point in disk.points),
                   disk.triangles)
    reverse = Mesh(forward.points, tuple((kind, a, c, b)
                                         for kind, a, b, c in forward.triangles))
    fwd_path, rev_path = out / "exit-fwd.surf", out / "exit-rev.surf"
    write_surface(fwd_path, forward, "transparent exit station, -z normal for forward movers")
    write_surface(rev_path, reverse, "same disk, +z normal for reverse movers")
    return fwd_path, rev_path


def _inside_repo(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        raise ValueError(f"path escapes repository: {relative}")
    return path


def _prefill_deck_seed(deck: str) -> int:
    commands = [line.split("#", 1)[0].strip().split()
                for line in deck.splitlines()]
    seeds = [parts for parts in commands if parts and parts[0] == "seed"]
    if (len(seeds) != 1 or len(seeds[0]) != 2 or
            not re.fullmatch(r"[1-9][0-9]*", seeds[0][1])):
        raise ValueError("saved CAD prefill deck needs exactly one positive seed command")
    return int(seeds[0][1])


def _verified_prefill(prefill_path: Path, fill: dict, config: dict,
                      checked_receipt: dict) -> tuple[dict, int, str, str]:
    prefill_receipt = json.loads((prefill_path.parent / "prefill.json").read_text())
    deck_path = prefill_path.parent / "in.prefill-candidates"
    source_deck = deck_path.read_text(encoding="utf-8")
    prefill_seed = _prefill_deck_seed(source_deck)
    deck_sha256 = _sha(deck_path)
    source_case_id = fill.get("prefill_source_case_id", config["case_id"])
    expected_mix = (f"mixture fillgas He nrho {fill['number_density_m3']:.17g} "
                    f"temp {config['temperature_K']:.17g}\n")
    expected_fnum = f"global fnum {config['fnum_real_per_sim_particle']:.17g}\n"
    if (prefill_receipt["case_id"] != source_case_id or
            prefill_receipt["solid_step_sha256"] != fill["solid_step_sha256"] or
            prefill_receipt["canonical_surface_sha256"] !=
            checked_receipt["canonical_surface_sha256"] or
            prefill_receipt.get("prefill_seed", prefill_seed) != prefill_seed or
            prefill_receipt.get("candidate_deck_sha256", deck_sha256) != deck_sha256 or
            not math.isclose(prefill_receipt["number_density_m3"],
                             fill["number_density_m3"], rel_tol=1e-12) or
            prefill_receipt["initial_dump_sha256"] != _sha(prefill_path) or
            source_deck.count(expected_mix) != 1 or
            source_deck.count(expected_fnum) != 1):
        raise ValueError("CAD prefill provenance does not match this case")
    provenance = ("prefill-receipt-and-candidate-deck"
                  if {"prefill_seed", "candidate_deck_sha256"} <= set(prefill_receipt) else
                  "saved-candidate-deck; legacy receipt has no deck hash")
    return prefill_receipt, prefill_seed, deck_sha256, provenance


def _inspect_particle_dump(path: Path, step: int, count: int,
                           bounds: list, converted: Path | None = None,
                           inside: list | None = None) -> dict:
    """Check a six-column SPARTA dump; optionally add ID/type for read_particles.

    This preserves the saved decimal coordinates and velocities, but the dump
    itself has finite printed precision and does not preserve the solver RNG.
    `inside` (G3): a larger child box every particle position must lie in.
    """
    with path.open(encoding="utf-8") as source:
        header = [next(source) for _ in range(9)]
        if (header[0].strip() != "ITEM: TIMESTEP" or
                int(header[1]) != step or
                header[2].strip() != "ITEM: NUMBER OF ATOMS" or
                int(header[3]) != count or
                header[4].strip() != "ITEM: BOX BOUNDS oo oo oo" or
                header[8].strip() != "ITEM: ATOMS x y z vx vy vz"):
            raise ValueError("unexpected SPARTA particle dump header")
        for axis, line in zip(bounds, header[5:8]):
            pair = line.split()
            # SPARTA prints box bounds with %g (dump_particle.cpp:307-309).
            if (len(pair) != 2 or any(float(value) != float(f"{edge:g}")
                    for value, edge in zip(pair, axis))):
                raise ValueError("particle dump box differs from checked geometry")
        if converted is not None:
            target = converted.open("w", encoding="utf-8", newline="\n")
            header[8] = "ITEM: ATOMS id type x y z vx vy vz\n"
            target.writelines(header)
        else:
            target = None
        try:
            seen = 0
            for seen, line in enumerate(source, 1):
                values = line.split()
                if len(values) != 6 or any(not math.isfinite(float(x)) for x in values):
                    raise ValueError("invalid particle dump row")
                if inside is not None and any(not lo <= float(x) <= hi
                                              for x, (lo, hi) in zip(values, inside)):
                    raise ValueError("parent particle lies outside the child box")
                if target is not None:
                    target.write(f"{seen} 1 " + " ".join(values) + "\n")
            if seen != count:
                raise ValueError("particle dump row count differs from solver stats")
        finally:
            if target is not None:
                target.close()
    result = {"path": path.name, "sha256": _sha(path),
              "timestep": step, "particles": count}
    if converted is not None:
        result["converted_sha256"] = _sha(converted)
    return result


SAME_PHYSICS = ("geometry_case_id", "mass_flow_kg_s", "nominal_sccm",
                "atoms_per_sccm_s", "helium_atom_mass_kg", "temperature_K",
                "mixture_number_density_m3", "fnum_real_per_sim_particle",
                "timestep_s", "inlet_return_action", "initial_fill",
                "exit_station", "domain_balance")


LATTICE_TOL = 1e-6   # base cells; a parent box face this close to a child lattice plane is on it


def _geometry_domain(geometry: dict, surface_sha256: str) -> dict:
    probe = geometry["solver_probe"]
    return {"domain_bounds_m": probe["domain_bounds_m"], "base_grid": probe["base_grid"],
            "surface_sha256": surface_sha256}


def _run_domain(deck: Path, receipt: dict) -> dict:
    """Box, base grid and canonical surface a completed run's hash-bound deck used."""
    if not deck.is_file() or _sha(deck) != receipt.get("deck_sha256"):
        raise ValueError("continuation parent deck is missing or changed")
    text = deck.read_text(encoding="utf-8")
    box = re.findall(r"^create_box (.+)$", text, re.MULTILINE)
    grid = re.findall(r"^create_grid (\d+) (\d+) (\d+)", text, re.MULTILINE)
    values = [float(v) for v in box[0].split()] if len(box) == 1 else []
    if len(values) != 6 or len(grid) != 1:
        raise ValueError("continuation parent deck lacks one create_box/create_grid")
    return {"domain_bounds_m": [values[0:2], values[2:4], values[4:6]],
            "base_grid": [int(n) for n in grid[0]],
            "surface_sha256": receipt.get("canonical_surface_sha256")}


def _box_extends(parent: dict, child: dict) -> bool:
    """G3: the child box strictly contains the parent box, same base cell, same lattice, same surface."""
    if (parent["surface_sha256"] != child["surface_sha256"] or
            parent["domain_bounds_m"] == child["domain_bounds_m"]):
        return False
    for (plo, phi), (clo, chi), pn, cn in zip(parent["domain_bounds_m"], child["domain_bounds_m"],
                                              parent["base_grid"], child["base_grid"]):
        h = (chi - clo) / cn
        if (not math.isclose((phi - plo) / pn, h, rel_tol=1e-9) or not clo <= plo < phi <= chi or
                any(abs((edge - clo) / h - round((edge - clo) / h)) > LATTICE_TOL
                    for edge in (plo, phi))):
            return False
    return True


def _continuation_physics_differs(parent_config: dict, config: dict,
                                  parent_domain: dict | None = None,
                                  domain: dict | None = None) -> list[str]:
    """SAME_PHYSICS keys a continuation changes from its parent; empty means allowed.

    G1 rule (Arian, 2026-09-27, W3 plan N5): exit_station.z_m, and only z_m, may
    differ exactly when grid_refinement differs. Each added dyadic level can put
    the old plane on a cell face (doubled tally), so the station moves to a
    mid-cell plane of the new finest level. Radius, segments and sampling stay.

    G3 rule (W3 plan section 9): a caller whose checked geometry differs from the
    parent's passes both runs' domains (box, base grid, canonical surface).
    geometry_case_id may then differ exactly when `_box_extends`; any other
    domain change is reported as "domain". Without domains nothing changes.
    """
    differs = [key for key in SAME_PHYSICS if parent_config.get(key) != config.get(key)]
    old, new = parent_config.get("exit_station"), config.get("exit_station")
    if ("exit_station" in differs and isinstance(old, dict) and isinstance(new, dict) and
            parent_config.get("grid_refinement") != config.get("grid_refinement") and
            {**old, "z_m": None} == {**new, "z_m": None}):
        differs.remove("exit_station")
    if parent_domain is not None or domain is not None:
        if parent_domain is None or domain is None or not _box_extends(parent_domain, domain):
            differs.append("domain")
        elif "geometry_case_id" in differs:
            differs.remove("geometry_case_id")
    return differs


def _needs_preclone(parent_config: dict, config: dict) -> bool:
    """Unrefined parent -> refined child needs a pre-clone; refined -> identical refined does not.

    read_particles keeps particle weights (read_particles.cpp:206-219), so a
    level-1 snapshot read into level-k cells carries 1/8^(k-1) of its atoms.
    Undoing refinement (merging) or changing it is unsupported.
    """
    parent, child = parent_config.get("grid_refinement"), config.get("grid_refinement")
    if parent is not None and parent != child:
        raise ValueError("continuation from a refined parent needs the identical grid_refinement")
    return parent is None and child is not None


def _preclone_initial(deck: Path, converted: Path, target: Path) -> dict:
    """Pre-clone the converted snapshot onto `deck`'s grid; keep only hashes and counts."""
    result = preclone(deck, converted, target)
    # ponytail: the intermediate is the deterministic conversion of the hashed parent snapshot.
    converted.unlink()
    return {key: result[key] for key in (
        "schema_version", "deck_sha256", "input_sha256", "output_sha256",
        "input_particles", "output_particles", "levels")}


def _check_initial_chain(cont: dict, deck_sha256: str, preclone_expected: bool) -> None:
    """Parent snapshot -> converted file [-> pre-clone on this deck] -> file SPARTA read."""
    try:
        pre = cont.get("preclone")
        source = cont["parent_snapshot"]["converted_sha256"]
        if (pre is not None) != preclone_expected:
            raise ValueError("continuation pre-clone step is missing or unexpected")
        if pre is not None:
            levels = pre["levels"].values()
            if (pre["input_sha256"] != source or pre["deck_sha256"] != deck_sha256 or
                    pre["input_particles"] != cont["parent_snapshot"]["particles"] or
                    sum(v["input"] for v in levels) != pre["input_particles"] or
                    sum(v["output"] for v in levels) != pre["output_particles"]):
                raise ValueError("continuation pre-clone does not bind the parent snapshot to this deck")
            source = pre["output_sha256"]
        if cont["initial_particle_sha256"] != source:
            raise ValueError("continuation initial particles are not the recorded conversion")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"continuation initial-particle chain is incomplete: {exc!r}") from exc


def _initial_particle_count(cont: dict) -> int:
    """Particles SPARTA must report at step 0: the pre-clone output when there is one."""
    pre = cont.get("preclone")
    return pre["output_particles"] if pre is not None else cont["parent_snapshot"]["particles"]


def _prepare_continuation(config: dict, checked: Path, checked_receipt: dict,
                          geometry: dict, cap: dict, solver: Path, out: Path,
                          deck: Path | None = None) -> dict:
    cont = config["continuation"]
    parent_dir = _inside_repo(cont["parent_run_dir"])
    parent_config_path = _inside_repo(cont["parent_config_path"])
    parent_run_path = parent_dir / "run.json"
    if (not parent_run_path.is_file() or not parent_config_path.is_file() or
            _sha(parent_run_path) != cont["parent_run_sha256"]):
        raise ValueError("continuation parent receipt/configuration is missing or changed")
    parent = json.loads(parent_run_path.read_text(encoding="utf-8"))
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    _validate_field_config(parent_config)
    box = geometry["solver_probe"]["domain_bounds_m"]
    same_geometry = (parent.get("geometry_sha256") == _sha(checked / "geometry.json") and
                     parent.get("solver_check_sha256") == _sha(checked / "solver-check.json"))
    # G3: another checked geometry is allowed only as a box extension of the parent's.
    domains = () if same_geometry else (
        _run_domain(parent_dir / "in.helium-field", parent),
        _geometry_domain(geometry, checked_receipt["canonical_surface_sha256"]))
    if (parent.get("status") != "complete" or
            parent.get("case_id") != cont["parent_case_id"] or
            parent_config.get("case_id") != cont["parent_case_id"] or
            parent.get("config_sha256") != _sha(parent_config_path) or
            parent.get("steps") != cont["parent_steps"] or
            parent.get("frame", {}).get("timestep") != cont["parent_steps"] or
            parent.get("canonical_surface_sha256") !=
                checked_receipt["canonical_surface_sha256"] or
            parent.get("solver_sha256") != _sha(solver) or
            parent.get("wall_model") != cap["helium_wall_model"] or
            _continuation_physics_differs(parent_config, config, *domains)):
        raise ValueError("continuation parent physics or geometry differs")
    snapshot = parent_dir / f"particles.{cont['parent_steps']}"
    if not snapshot.is_file() or _sha(snapshot) != cont["parent_snapshot_sha256"]:
        raise ValueError("continuation particle snapshot is missing or changed")
    derived = out / f"initial-particles.{cont['parent_steps']}"
    needs_preclone = _needs_preclone(parent_config, config)
    if needs_preclone and (deck is None or not deck.is_file()):
        raise ValueError("a refined continuation needs the written deck SPARTA will read")
    converted = (out / f"converted-particles.{cont['parent_steps']}"
                 if needs_preclone else derived)
    snapshot_info = _inspect_particle_dump(
        snapshot, cont["parent_steps"], parent["final_particles"],
        domains[0]["domain_bounds_m"] if domains else box, converted,
        box if domains else None)
    extra = {"preclone": _preclone_initial(deck, converted, derived)} if needs_preclone else {}
    if domains:
        extra["parent_domain"] = domains[0]
    if "exit_station" in parent_config:
        # G1: the child's plane is receipt["exit_station"]["geometry"]["z_m"].
        extra["parent_exit_station_z_m"] = parent_config["exit_station"]["z_m"]
    return {
        **extra,
        "parent_case_id": parent["case_id"],
        "parent_run_sha256": cont["parent_run_sha256"],
        "parent_config_sha256": parent["config_sha256"],
        "parent_snapshot": snapshot_info,
        "initial_particle_file": derived.name,
        "initial_particle_sha256": _sha(derived),
        "parent_elapsed_physical_s": parent.get(
            "cumulative_elapsed_physical_s", parent["elapsed_physical_s"]),
        "random_stream": ("reused-seed; RNG reinitialized; particle snapshot is not an exact RNG restart"
                          if parent_config["seed"] == config["seed"] else
                          "new-seed; particle snapshot is not an exact RNG restart"),
    }


def _run_solver(command: list[str], out: Path, timeout_s: int):
    log_path = out / "solver.log"
    try:
        result = subprocess.run(
            command, cwd=out, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=None if timeout_s == 0 else timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        log_path.write_bytes(exc.stdout or b"")
        raise
    log_path.write_bytes(result.stdout)
    return result


def run(checked: Path, config_path: Path, solver: Path, out: Path,
        *, solver_timeout_s: int = 600, mpi_ranks: int = 1,
        mpirun: Path | None = None) -> dict:
    checked, config_path, solver, out = (Path(path).resolve() for path in
                                         (checked, config_path, solver, out))
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"CAD field output already exists: {out}")
    if out == checked or out in checked.parents or checked in out.parents:
        raise ValueError("field output must be separate from checked geometry")
    if not solver.is_file():
        raise FileNotFoundError(solver)
    if type(solver_timeout_s) is not int or solver_timeout_s < 0:
        raise ValueError("solver timeout must be a nonnegative integer")
    if type(mpi_ranks) is not int or mpi_ranks < 1:
        raise ValueError("mpi_ranks must be a positive integer")
    if mpi_ranks > 1:
        if mpirun is None:
            raise ValueError("multi-rank CAD field run requires --mpirun")
        mpirun = Path(mpirun).resolve()
        if not mpirun.is_file():
            raise FileNotFoundError(mpirun)
    elif mpirun is not None:
        raise ValueError("--mpirun requires --mpi-ranks greater than one")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = _validate_field_config(config)
    checked_receipt, geometry, cap = _validate_geometry(checked, solver, base)
    if "diagnostics" in config:
        _validate_diagnostic_geometry(config["diagnostics"], geometry)
    if "exit_station" in config:
        station = config["exit_station"]
        bounds = geometry["solver_probe"]["domain_bounds_m"]
        surface_zmax = geometry["inspection"]["physical_boundary"]["bbox_m"][2][1]
        if (not surface_zmax + 1e-4 < station["z_m"] < bounds[2][1] - 1e-4 or
                station["radius_m"] >= min(bounds[0][1], -bounds[0][0],
                                           bounds[1][1], -bounds[1][0])):
            raise ValueError("exit station must clear the physical surface and domain box")
    out.mkdir(parents=True)
    try:
        for name, source in (("physical-walls.surf", checked / "physical-walls.surf"),
                             ("he.species", ROOT / "cases/data/he.species"),
                             ("he.vss", ROOT / "cases/data/he.vss")):
            shutil.copyfile(source, out / name)
        prefill_receipt = continuation_receipt = None
        prefill_seed = prefill_deck_sha256 = prefill_seed_provenance = None
        # The deck is written first so a pre-clone reads the grid SPARTA will build.
        deck_path = out / "in.helium-field"
        deck_path.write_text(_field_deck(config, geometry, mpi_ranks=mpi_ranks),
                             encoding="utf-8", newline="\n")
        if "continuation" in config:
            continuation_receipt = _prepare_continuation(
                config, checked, checked_receipt, geometry, cap, solver, out, deck_path)
            _check_initial_chain(continuation_receipt, _sha(deck_path),
                                 "preclone" in continuation_receipt)
        elif config.get("initial_fill", {}).get("model") == "cad-connected-cell-gas":
            fill = config["initial_fill"]
            prefill_path = (ROOT / fill["prefill_path"]).resolve()
            if ROOT not in prefill_path.parents or not prefill_path.is_file():
                raise ValueError("CAD prefill particle file is missing or outside repository")
            (prefill_receipt, prefill_seed, prefill_deck_sha256,
             prefill_seed_provenance) = _verified_prefill(
                 prefill_path, fill, config, checked_receipt)
            shutil.copyfile(prefill_path, out / "initial-particles.0")
        if "exit_station" in config:
            _write_exit_disks(out, config["exit_station"])
        command = ([str(mpirun), "-np", str(mpi_ranks)] if mpi_ranks > 1 else [])
        command += [str(solver), "-in", deck_path.name]
        result = _run_solver(command, out, solver_timeout_s)
        log_path = out / "solver.log"
        if result.returncode:
            raise ValueError(f"SPARTA CAD field pilot failed ({result.returncode}); see {log_path}")
        log = result.stdout.decode(errors="replace")
        summary = _field_summary(log, config["steps"])
        if (continuation_receipt is not None and
                summary["initial_particles"] !=
                _initial_particle_count(continuation_receipt)):
            raise ValueError("SPARTA did not read the complete continuation snapshot")
        loop = re.search(rf"Loop time of ([\d.eE+-]+) on {mpi_ranks} procs", log)
        if not loop:
            raise ValueError("SPARTA did not report the requested-rank loop time")
        frame_path = out / f"fields.{config['steps']}.gz"
        frame = validate_frame(frame_path, expected_step=config["steps"])
        if "field_frame_every_steps" in config:
            every = config["field_frame_every_steps"]
            frames = []
            for step in range(every, config["steps"] + 1, every):
                path = out / f"fields.{step}.gz"
                if not path.is_file():
                    raise ValueError(f"SPARTA did not write intermediate frame {path.name}")
                item = validate_frame(path, expected_step=step)
                frames.append({"path": path.name, **{key: item[key] for key in (
                    "timestep", "rows", "bytes", "sha256", "box_bounds_m", "validation")}})
            if frames[-1]["sha256"] != frame["sha256"]:
                raise ValueError("final listed field frame differs from the selected frame")
        final_snapshot = _inspect_particle_dump(
            out / f"particles.{config['steps']}", config["steps"],
            summary["final_particles"], geometry["solver_probe"]["domain_bounds_m"])
        receipt = {
            "schema_version": "dsmc-cad-helium-field-pilot-result-v1",
            "status": "complete", "scope": config["scope"],
            "case_id": config["case_id"], "geometry_case_id": geometry["case_id"],
            "canonical_surface_sha256": checked_receipt["canonical_surface_sha256"],
            "geometry_sha256": _sha(checked / "geometry.json"),
            "solver_check_sha256": _sha(checked / "solver-check.json"),
            "config_sha256": _sha(config_path), "solver_sha256": _sha(solver),
            "deck_sha256": _sha(deck_path), "log_sha256": _sha(log_path),
            "frame": frame, "inlet_boundary_id": geometry["helium_inlet"]["boundary_id"],
            "final_particle_snapshot": final_snapshot,
            "inlet_area_m2": geometry["inspection"]["roles"][geometry["helium_inlet"]["boundary_id"]]["area_m2"],
            "wall_model": cap["helium_wall_model"],
            "inlet_return_action": config["inlet_return_action"],
            "nominal_sccm": config["nominal_sccm"],
            "source_mass_flow_kg_s": config["mass_flow_kg_s"],
            "initial_condition": ("particle-snapshot-continuation"
                                  if continuation_receipt is not None else
                                  config.get("initial_fill") or "empty-domain"),
            "field_sample_window_steps": (config["field_sample_every_steps"] *
                                          config["field_sample_count"]),
            "steps": config["steps"],
            "elapsed_physical_s": config["steps"] * config["timestep_s"],
            "solver_timeout_s": solver_timeout_s,
            "loop_time_s": float(loop.group(1)),
            **summary,
        }
        if "field_frame_every_steps" in config:
            receipt["field_frames"] = {"every_steps": config["field_frame_every_steps"],
                                       "frames": frames}
        if mpi_ranks > 1:
            receipt["mpi"] = {"ranks": mpi_ranks, "launcher": str(mpirun),
                              "launcher_sha256": _sha(mpirun), "command": command}
        if prefill_receipt is not None:
            receipt["cad_prefill"] = prefill_receipt
            receipt["prefill_seed"] = prefill_seed
            receipt["dynamics_seed"] = config["seed"]
            receipt["prefill_deck_sha256"] = prefill_deck_sha256
            receipt["prefill_seed_provenance"] = prefill_seed_provenance
        if continuation_receipt is not None:
            receipt["continuation"] = continuation_receipt
            receipt["cumulative_elapsed_physical_s"] = (
                continuation_receipt["parent_elapsed_physical_s"] +
                receipt["elapsed_physical_s"])
        if "initial_fill" in config:
            receipt["initial_fill_density_basis"] = config["initial_fill"].get(
                "density_basis", "unspecified")
            inventory = out / "body-inventory.out"
            if not inventory.is_file():
                raise ValueError("SPARTA did not write body inventory history")
            receipt["body_inventory"] = {"path": inventory.name,
                                         "sha256": _sha(inventory),
                                         "region": config["initial_fill"]}
        if "exit_station" in config:
            forward, reverse = (out / "exit-fwd.out", out / "exit-rev.out")
            fwd, rev = read_dump(forward), read_dump(reverse)
            shared = sorted(set(fwd) & set(rev))
            if (not shared or shared[-1] != config["steps"] or
                    set(fwd) != set(rev)):
                raise ValueError("exit station lacks matched completed windows")
            receipt["exit_station"] = {
                "geometry": config["exit_station"],
                "fwd_surface_sha256": _sha(out / "exit-fwd.surf"),
                "rev_surface_sha256": _sha(out / "exit-rev.surf"),
                "fwd_dump_sha256": _sha(forward), "rev_dump_sha256": _sha(reverse),
                "last_window_net_atoms_s": fwd[shared[-1]] - rev[shared[-1]],
                "last_window_net_sccm": ((fwd[shared[-1]] - rev[shared[-1]]) /
                                         config["atoms_per_sccm_s"]),
                "windows": len(shared),
            }
        if config.get("domain_balance"):
            paths = {name: out / name for name in
                     ("domain-inventory.out", "boundary.out")}
            if any(not path.is_file() for path in paths.values()):
                raise ValueError("SPARTA did not write domain balance histories")
            receipt["domain_balance"] = {
                name: {"path": path.name, "sha256": _sha(path)}
                for name, path in paths.items()}
        if "diagnostics" in config:
            receipt["regional_diagnostics"] = _regional_receipt(out, config)
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
    parser.add_argument("--solver-timeout-s", type=int, default=600,
                        help="wall-clock limit in seconds; 0 disables the limit")
    parser.add_argument("--mpi-ranks", type=int, default=1)
    parser.add_argument("--mpirun", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.checked, args.config, args.solver, args.out,
                         solver_timeout_s=args.solver_timeout_s,
                         mpi_ranks=args.mpi_ranks, mpirun=args.mpirun), indent=2))


if __name__ == "__main__":
    main()
