"""Strict, machine-independent inputs for the first 3D B5 preparation route."""
from __future__ import annotations

import hashlib
import copy
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_VERSION = "dsmc-case-v1"
PROFILE_VERSION = "dsmc-profile-v1"
PREPARED_VERSION = "dsmc-prepared-v1"
UINT64_MAX = 2**64 - 1


def _keys(value, required, optional=(), where="object"):
    if type(value) is not dict:
        raise ValueError(f"{where}: expected an object")
    missing = set(required) - value.keys()
    extra = value.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{where}: missing {sorted(missing)}; unknown keys {sorted(extra)}")


def _number(value, where, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{where}: expected a finite {'positive ' if positive else ''}number")
    return value


def _integer(value, where, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where}: expected an integer >= {minimum}")
    return value


def _string(value, where):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{where}: expected a nonempty string")
    return value


def _choice(value, allowed, where):
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{where}: expected one of {sorted(allowed)}")
    return value


def _vector(value, where, n=3):
    if type(value) is not list or len(value) != n:
        raise ValueError(f"{where}: expected {n} SI numeric components")
    return [_number(x, where) for x in value]


def _seed(value, where):
    if type(value) is not str or not re.fullmatch(r"[1-9][0-9]*", value) or int(value) > UINT64_MAX:
        raise ValueError(f"{where}: expected a nonzero decimal UInt64 string")
    return value


def _path(value, origin, where):
    _string(value, where)
    path = (origin / value).resolve()
    if not path.is_file():
        raise ValueError(f"{where}: file not found: {path}")
    return str(path)


def _read_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key!r}")
            result[key] = value
        return result
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_pairs,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"non-finite JSON value {x}")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: cannot read JSON: {exc}") from exc


def _identity(path):
    path = Path(path)
    resolved = path.resolve()
    try:
        name = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        name = str(resolved)  # External input remains explicit, never invented as repo content.
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def load_case(path: Path) -> dict:
    """Resolve and validate a versioned B5 case; paths are relative to its file."""
    path = Path(path).resolve()
    case = _read_json(path)
    _keys(case, ("schema_version", "case_id", "model", "geometry", "helium", "numerics",
                 "molecules", "observables"), ("assessment",), "case")
    if case["schema_version"] != CASE_VERSION:
        raise ValueError(f"schema_version: expected {CASE_VERSION}")
    _string(case["case_id"], "case_id")
    _keys(case["model"], ("label", "provenance"), where="model")
    for key in case["model"]:
        _string(case["model"][key], f"model.{key}")

    g = case["geometry"]
    _keys(g, ("adapter", "stub_preset", "aspect_vertical_over_horizontal", "area_basis",
              "length_unit", "transform", "boundaries", "gas_reference_points_m"), where="geometry")
    _choice(g["adapter"], ("b5-circle", "b5-slit"), "geometry.adapter")
    _choice(g["stub_preset"], ("s25", "5mm"), "geometry.stub_preset")
    _number(g["aspect_vertical_over_horizontal"], "geometry.aspect_vertical_over_horizontal", positive=True)
    if g["aspect_vertical_over_horizontal"] < 1:
        raise ValueError("geometry.aspect_vertical_over_horizontal: vertical slit requires aspect >= 1")
    _choice(g["area_basis"], ("polygon",), "geometry.area_basis")
    if g["length_unit"] != "m":
        raise ValueError("geometry.length_unit: use m; no implicit length conversion")
    _keys(g["transform"], ("solver_to_tracer_axes", "translation_m"), where="geometry.transform")
    if g["transform"]["solver_to_tracer_axes"] != ["y", "z", "x"] or g["transform"]["translation_m"] != [0, 0, 0]:
        raise ValueError("geometry.transform: B5 adapter requires solver (x,y,z) -> tracer (y,z,x), zero translation")
    _keys(g["boundaries"], ("wall", "inlet_cap", "tube_fwd", "tube_rev", "ap_fwd", "ap_rev"), where="geometry.boundaries")
    expected = {"wall": (1, "absorb"), "inlet_cap": (2, "absorb"),
                "tube_fwd": (1, "ignore"), "tube_rev": (1, "ignore"),
                "ap_fwd": (1, "ignore"), "ap_rev": (1, "ignore")}
    for name, (surface_type, action) in expected.items():
        b = g["boundaries"][name]
        _keys(b, ("surface_type", "molecule_action"), where=f"geometry.boundaries.{name}")
        if type(b["surface_type"]) is not int or b["surface_type"] != surface_type or b["molecule_action"] != action:
            raise ValueError(f"geometry.boundaries.{name}: expected type {surface_type}, molecule action {action}")
    if type(g["gas_reference_points_m"]) is not list or not g["gas_reference_points_m"]:
        raise ValueError("geometry.gas_reference_points_m: provide at least one point")
    for point in g["gas_reference_points_m"]:
        _vector(point, "geometry.gas_reference_points_m")

    h = case["helium"]
    _keys(h, ("species_file", "collision_file", "wall_temperature_K", "wall_accommodation",
              "inlet", "fill", "provenance"), where="helium")
    for key in ("species_file", "collision_file"):
        h[key] = _path(h[key], path.parent, f"helium.{key}")
    _number(h["wall_temperature_K"], "helium.wall_temperature_K", positive=True)
    _number(h["wall_accommodation"], "helium.wall_accommodation")
    if not 0 <= h["wall_accommodation"] <= 1:
        raise ValueError("helium.wall_accommodation: expected 0..1")
    _keys(h["inlet"], ("model", "port", "mass_flow_kg_s", "temperature_K"), where="helium.inlet")
    if h["inlet"]["model"] != "emit/surf-mflow" or h["inlet"]["port"] != "inlet_cap":
        raise ValueError("helium.inlet: B5 adapter requires emit/surf-mflow at inlet_cap")
    for key in ("mass_flow_kg_s", "temperature_K"):
        _number(h["inlet"][key], f"helium.inlet.{key}", positive=True)
    _keys(h["fill"], ("number_density_m3", "temperature_K"), where="helium.fill")
    for key in h["fill"]:
        _number(h["fill"][key], f"helium.fill.{key}", positive=True)
    _string(h["provenance"], "helium.provenance")

    n = case["numerics"]
    _keys(n, ("deck_template", "domain_bounds_m", "base_grid", "refinement", "weighting",
              "fnum", "dt_s", "total_steps", "warmup_steps", "solver_seed",
              "field_every_steps", "field_average", "restart_every_steps"), where="numerics")
    n["deck_template"] = _path(n["deck_template"], path.parent, "numerics.deck_template")
    if type(n["domain_bounds_m"]) is not list or len(n["domain_bounds_m"]) != 3:
        raise ValueError("numerics.domain_bounds_m: expected three [low, high] pairs in m")
    for bounds in n["domain_bounds_m"]:
        lo, hi = _vector(bounds, "numerics.domain_bounds_m", 2)
        if lo >= hi:
            raise ValueError("numerics.domain_bounds_m: low must be less than high")
    if type(n["base_grid"]) is not list or len(n["base_grid"]) != 3:
        raise ValueError("numerics.base_grid: expected three cell counts")
    for count in n["base_grid"]:
        _integer(count, "numerics.base_grid", minimum=1)
    if n["refinement"] != "b5-s25-sampled-v1" or n["weighting"] != "cell-volume":
        raise ValueError("numerics: unsupported refinement/weighting for first B5 adapter")
    for key in ("fnum", "dt_s"):
        _number(n[key], f"numerics.{key}", positive=True)
    for key in ("total_steps", "warmup_steps", "field_every_steps", "restart_every_steps"):
        _integer(n[key], f"numerics.{key}", minimum=1)
    if n["total_steps"] <= n["warmup_steps"] or n["total_steps"] % n["field_every_steps"]:
        raise ValueError("numerics.total_steps: must exceed warmup and end on a field step")
    _integer(n["solver_seed"], "numerics.solver_seed", minimum=1)
    _keys(n["field_average"], ("samples", "stride_steps", "period_steps"), where="numerics.field_average")
    for key in n["field_average"]:
        _integer(n["field_average"][key], f"numerics.field_average.{key}", minimum=1)

    m = case["molecules"]
    _keys(m, ("mass_u", "gas_mass_u", "cross_section_m2", "source", "count", "seed",
              "flight", "collision_rate_model", "flight_step_m", "rng_protocol", "wall_action",
              "gas_reconstruction", "provenance"), where="molecules")
    for key in ("mass_u", "gas_mass_u", "cross_section_m2", "flight_step_m"):
        _number(m[key], f"molecules.{key}", positive=True)
    _integer(m["count"], "molecules.count", minimum=1)
    _seed(m["seed"], "molecules.seed")
    _choice(m["flight"], ("local", "field-aware"), "molecules.flight")
    _choice(m["collision_rate_model"], ("approximate", "maxwell"), "molecules.collision_rate_model")
    if m["rng_protocol"] != "particle-v1" or m["wall_action"] != "absorb":
        raise ValueError("molecules: unsupported RNG protocol or wall action")
    _keys(m["source"], ("distribution", "center_m", "radius_m", "drift_m_s", "temperature_K"), where="molecules.source")
    _choice(m["source"]["distribution"], ("point", "uniform-ball", "gaussian-ball"), "molecules.source.distribution")
    _vector(m["source"]["center_m"], "molecules.source.center_m")
    _vector(m["source"]["drift_m_s"], "molecules.source.drift_m_s")
    radius = _number(m["source"]["radius_m"], "molecules.source.radius_m")
    temperature = _number(m["source"]["temperature_K"], "molecules.source.temperature_K")
    if (m["source"]["distribution"] == "point" and radius != 0 or
            m["source"]["distribution"] != "point" and radius <= 0):
        raise ValueError("molecules.source.radius_m: point requires zero; extended source requires positive")
    if temperature < 0:
        raise ValueError("molecules.source.temperature_K: expected a nonnegative number")
    _keys(m["gas_reconstruction"], ("field", "missing_cell"), where="molecules.gas_reconstruction")
    if m["gas_reconstruction"]["field"] != "frozen-scalar-temperature" or m["gas_reconstruction"]["missing_cell"] != "error":
        raise ValueError("molecules.gas_reconstruction: unsupported first-route policy")
    _string(m["provenance"], "molecules.provenance")

    o = case["observables"]
    _keys(o, ("planes", "field_regions"), where="observables")
    if type(o["planes"]) is not list or not o["planes"]:
        raise ValueError("observables.planes: expected at least one named plane")
    for plane in o["planes"]:
        _keys(plane, ("name", "origin_m", "normal", "axis_u", "axis_v", "acceptance"), where="observables.plane")
        _string(plane["name"], "observables.plane.name")
        for key in ("origin_m", "normal", "axis_u", "axis_v"):
            _vector(plane[key], f"observables.plane.{key}")
        a, b, c = (plane[key] for key in ("normal", "axis_u", "axis_v"))
        dot = lambda x, y: sum(i*j for i, j in zip(x, y))
        if any(abs(dot(x, x)-1) > 1e-9 for x in (a, b, c)) or any(abs(dot(x, y)) > 1e-9 for x, y in ((a,b),(a,c),(b,c))):
            raise ValueError("observables.plane: axes must be orthonormal")
        cross = [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
        if max(abs(x-y) for x, y in zip(cross, c)) > 1e-9:
            raise ValueError("observables.plane: normal cross axis_u must equal axis_v")
        _keys(plane["acceptance"], ("position_u_m", "position_v_m", "angle_rad", "time_s"), where="observables.plane.acceptance")
        for key, bounds in plane["acceptance"].items():
            if bounds is None:
                continue  # An exploratory plane is reported without a pass/fail acceptance.
            lo, hi = _vector(bounds, f"observables.plane.acceptance.{key}", 2)
            if lo >= hi:
                raise ValueError(f"observables.plane.acceptance.{key}: low must be less than high")
    if type(o["field_regions"]) is not list or not o["field_regions"] or any(type(x) is not str for x in o["field_regions"]):
        raise ValueError("observables.field_regions: expected nonempty list of names")
    if "assessment" in case:
        _keys(case["assessment"], ("intended_comparison", "tolerances"), where="assessment")
        _string(case["assessment"]["intended_comparison"], "assessment.intended_comparison")
        if type(case["assessment"]["tolerances"]) is not dict:
            raise ValueError("assessment.tolerances: expected object; empty means unassessed")
    case["source_case"] = _identity(path)
    return case


def load_profile(path: Path) -> dict:
    """Resolve launch paths only; a profile has no science-input fields."""
    path = Path(path).resolve()
    profile = _read_json(path)
    _keys(profile, ("schema_version", "profile_id", "platform", "solver_executable", "mpi_argv",
                    "python", "julia", "ranks", "threads", "scratch_root", "results_root",
                    "max_memory_gb", "max_disk_gb"), where="profile")
    if profile["schema_version"] != PROFILE_VERSION:
        raise ValueError(f"profile.schema_version: expected {PROFILE_VERSION}")
    _string(profile["profile_id"], "profile.profile_id")
    _choice(profile["platform"], ("local", "slurm"), "profile.platform")
    for key in ("solver_executable", "python", "julia", "scratch_root", "results_root"):
        value = _string(profile[key], f"profile.{key}")
        if key in ("scratch_root", "results_root") or "/" in value or "\\" in value:
            profile[key] = str((path.parent / value).resolve()) if not Path(value).is_absolute() else value
    if type(profile["mpi_argv"]) is not list or any(type(x) is not str or not x for x in profile["mpi_argv"]):
        raise ValueError("profile.mpi_argv: expected argv array")
    for key in ("ranks", "threads"):
        _integer(profile[key], f"profile.{key}", minimum=1)
    if profile["ranks"] > 1 and not profile["mpi_argv"]:
        raise ValueError("profile.mpi_argv: a multi-rank run needs an explicit launcher")
    for key in ("max_memory_gb", "max_disk_gb"):
        _number(profile[key], f"profile.{key}", positive=True)
    profile["source_profile"] = _identity(path)
    return profile


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _replace_once(text, pattern, replacement, label):
    updated, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"deck template: expected one {label}, found {count}")
    return updated


def _deck(case, cell_name, throat_l4, throat_buffer, eqthroat):
    n = case["numerics"]
    text = Path(n["deck_template"]).read_text(encoding="utf-8")
    expected = ("units       si", "dimension   3", "surf_collide walls diffuse 4.0 1.0",
                "fix         afields ave/grid all 12 100 1200", "dump        dfields grid all 2400 fields.*.gz")
    if any(item not in text for item in expected):
        raise ValueError("deck template: not the supported s25 sampled SI deck")
    fixed_lines = (
        "create_box  -0.005 0.115 -0.025 0.025 -0.025 0.025",
        "create_grid 150 63 63 levels 5 region 2 level2 2 2 2 region 3 level3 2 2 2 region 4 level4 2 2 2 region 5 tube_buffer_l5 2 2 2 inside any",
        "global      weight cell volume", "species     he.species He",
        "mixture     fillgas He nrho ${FILLN} temp 4", "global      fnum ${FNUM}",
        "collide     vss hegas he.vss", "surf_modify cellwalls collide walls",
        "group       cap surf type 2", "fix         src emit/surf hegas cap normal yes mflow ${MDOT} 4.0",
        "timestep    ${DT}", "run         200", "restart     24000 b5.restart.%",
        "variable    nmain equal ${STEPS}-200", "run         ${nmain}",
    )
    for line in fixed_lines:
        if text.splitlines().count(line) != 1:
            raise ValueError(f"deck template: fixed B5 sampled command differs: {line}")
    if n["domain_bounds_m"] != [[-0.005, 0.115], [-0.025, 0.025], [-0.025, 0.025]] or n["base_grid"] != [150,63,63]:
        raise ValueError("numerics: B5 sampled template has fixed domain/grid; edit requires a new adapter")
    if (n["warmup_steps"], n["field_every_steps"]) != (200,2400) or n["restart_every_steps"] not in (24000, 2400) or n["field_average"] != {"samples":100,"stride_steps":12,"period_steps":1200}:
        raise ValueError("numerics: B5 sampled template schedule differs from case")
    h = case["helium"]
    if h["wall_temperature_K"] != 4 or h["wall_accommodation"] != 1 or h["inlet"]["temperature_K"] != 4 or h["fill"]["temperature_K"] != 4:
        raise ValueError("helium: template has fixed 4 K diffuse wall/inlet/fill; changed conditions need a deck adapter")
    replacements = ((r"^read_surf   \S+ type group cellwalls clip$", f"read_surf   {cell_name} type group cellwalls clip", "physical read_surf"),
                    (r"^(region\s+throat_buffer_l3 cylinder x 0 0 )\S+( 0\.0607 0\.0668)$", rf"\g<1>{throat_buffer:.12g}\g<2>", "throat_buffer_l3"),
                    (r"^(region\s+throat_l4 cylinder x 0 0 )\S+( 0\.0615 0\.066)$", rf"\g<1>{throat_l4:.12g}\g<2>", "throat_l4"),
                    (r"^(region\s+eqthroat cylinder x 0 0 )\S+( 0\.062 0\.065)$", rf"\g<1>{eqthroat:.12g}\g<2>", "eqthroat"))
    for pattern, replacement, label in replacements:
        text = _replace_once(text, pattern, replacement, label)
    checkpoint_name = "b5.restart.%.*" if case["case_id"] == "b5-circle-s25-recovery-probe" else "b5.restart.%"
    text = _replace_once(text, r"^restart\s+24000\s+b5\.restart\.%$",
                         f"restart     {n['restart_every_steps']} {checkpoint_name}", "restart cadence")
    return text


def prepare_case(case: dict, out: Path) -> Path:
    """Atomically publish a fresh preparation. Does not run SPARTA or Julia."""
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"prepare: refusing existing output path {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-prepare-", dir=out.parent) as temporary:
        staging = Path(temporary) / "prepared"
        _prepare_case_in_directory(case, staging)
        if out.exists() or out.is_symlink():
            raise FileExistsError(f"prepare: output path appeared during preparation: {out}")
        staging.rename(out)
    return out / "prepared.json"


def _prepare_case_in_directory(case: dict, out: Path) -> None:
    from dsmc_geometry import prepare_b5

    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"prepare: refusing existing output path {out}")
    source = case["source_case"]
    source_path = ROOT / source["path"] if not Path(source["path"]).is_absolute() else Path(source["path"])
    if _identity(source_path)["sha256"] != source["sha256"]:
        raise ValueError("source case changed after validation; reload it before preparation")
    # All template/contract checks happen before making the destination.
    plan = prepare_b5(case, None)
    deck = _deck(case, plan["cell_name"], plan["regions"]["throat_l4_radius_m"],
                 plan["regions"]["throat_buffer_l3_radius_m"], plan["regions"]["eqthroat_radius_m"])
    out.mkdir(parents=True)
    geometry = prepare_b5(case, out)
    staged = []
    def add(source, dest_name, role):
        target = out / dest_name
        shutil.copyfile(source, target)
        staged.append({"role": role, "transfer_mode": "text-lf", "source": _identity(source),
                       "staged": {**_identity(target), "path": dest_name}})
    for name, role in (("species_file", "helium_species"), ("collision_file", "helium_collision")):
        add(case["helium"][name], Path(case["helium"][name]).name, role)
    deck_path = out / "in.helium"
    deck_path.write_text(deck, encoding="utf-8", newline="\n")
    staged.append({"role": "helium_deck", "transfer_mode": "text-lf",
                   "source": _identity(case["numerics"]["deck_template"]),
                   "staged": {**_identity(deck_path), "path": deck_path.name}})
    for item in geometry["artifacts"]:
        staged.append({"role": item["role"], "transfer_mode": "text-lf", "source": item["producer"],
                       "staged": {**_identity(out / item["path"]), "path": item["path"]}})
    _json_write(out / "geometry.json", geometry)
    science = copy.deepcopy({key: case[key] for key in ("model", "geometry", "helium", "numerics", "molecules", "observables")})
    for section, key in (("helium", "species_file"), ("helium", "collision_file"),
                         ("numerics", "deck_template")):
        science[section][key] = _identity(science[section][key])["path"]
    prepared = {
        "schema_version": PREPARED_VERSION, "case_id": case["case_id"],
        "science_inputs": science,
        "assessment": case.get("assessment", {"state": "unassessed"}),
        "source_case": case["source_case"],
        "geometry_artifact": {**_identity(out / "geometry.json"), "path": "geometry.json"},
        "inputs": staged, "declared_outputs": [
            {"role": "helium_field", "pattern": "fields.*.gz", "required": True},
            {"role": "helium_restart", "pattern": "b5.restart.*", "required": False},
            {"role": "helium_final_restart", "pattern": "final.restart.*", "required": True},
            {"role": "helium_history", "pattern": "equil-*.out", "required": True}],
        "preparation_tools": ([_identity(ROOT / "tools" / name) for name in
                               ("dsmc.py", "dsmc_case.py", "dsmc_geometry.py", "revolve_surf.py")]
                              + [_identity(ROOT / "cases" / "b5-lean" / name) for name in
                                 ("gen_b5_3d.py", "gen_b5_3d_slit.py", "gen_b5_aperture_pair.py")]),
        "state": "prepared-unclipped", "solver_check": "not-run",
    }
    _json_write(out / "prepared.json", prepared)


def inspect_geometry(prepared: Path) -> dict:
    path = Path(prepared)
    if path.is_dir():
        path = path / "prepared.json"
    data = _read_json(path)
    if data.get("schema_version") != PREPARED_VERSION:
        raise ValueError(f"{path}: unsupported prepared schema")
    geometry_path = path.parent / "geometry.json"
    if _identity(geometry_path)["sha256"] != data["geometry_artifact"]["sha256"]:
        raise ValueError("geometry.json hash differs from prepared.json")
    geometry = _read_json(geometry_path)
    for item in data["inputs"]:
        staged = item["staged"]
        staged_path = path.parent / staged["path"]
        if not staged_path.is_file() or _identity(staged_path)["sha256"] != staged["sha256"]:
            raise ValueError(f"prepared input changed or missing: {staged_path}")
    if data["state"] == "solver-checked":
        canonical = path.parent / "physical-walls.surf"
        roles = path.parent / "triangle-roles.json"
        if not canonical.is_file() or not roles.is_file():
            raise ValueError("solver-checked preparation lacks canonical surface or triangle roles")
        role_data = _read_json(roles)
        if (role_data.get("surface_sha256") != _identity(canonical)["sha256"] or
                data["solver_check"]["triangle_roles_sha256"] != _identity(roles)["sha256"]):
            raise ValueError("canonical surface and triangle-role identities differ")
    limitations = ["Unclipped preparations need inspect --solver-check before canonical solver triangles and role IDs exist."]
    if data.get("science_inputs", {}).get("geometry", {}).get("adapter") == "cad":
        limitations.append("CAD transport requires an explicit closed-solid boundary set, gas seed and runnable physical laws; geometry import alone is insufficient.")
    return {"schema_version": "dsmc-inspection-v1", "case_id": data["case_id"],
            "state": data["state"], "solver_check": data["solver_check"],
            "geometry": geometry["inspection"], "regions": geometry.get("regions", {}),
            "artifacts": {x["role"]: x["staged"]["sha256"] for x in data["inputs"]},
            "limitations": limitations}
