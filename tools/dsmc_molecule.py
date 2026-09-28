"""Run a provenance-bound molecule trial on a completed prepared field."""
from __future__ import annotations

import hashlib
import csv
import copy
import json
import math
import os
import re
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from dsmc_report import analyze_observable_events


ROOT = Path(__file__).resolve().parents[1]
FIELD_ARTIFACTS = {
    "prepared.json", "parent-manifest.json", "DS3FF.DAT",
    "source-row-metadata.csv.gz", "cell.surfs", "physical-walls.surf",
    "triangle-roles.json", "triangle-roles.tsv", "canonical-tracer.toml",
    "provenance.json",
}
SPLIT_FIELD_ARTIFACTS = {
    "split-cell-map.tsv", "split-field-rows.tsv", "split-parent-ids.txt",
    "split-parent-ids.txt.json", "split-field-manifest.toml",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _artifact(path: Path) -> dict:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}


def _confined(field_dir: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"invalid field artifact name: {name!r}")
    path = field_dir / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing regular field artifact: {name}")
    return path


def _preflight(field_dir: Path, count: int | None) -> tuple[dict, dict, int]:
    receipt_path = _confined(field_dir, "field-preparation.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema_version") != "dsmc-field-preparation-v1" or
            receipt.get("status") != "complete"):
        raise ValueError("field preparation receipt is not complete")
    frame_name = Path(receipt["frame"]["path"]).name
    split_lookup = receipt.get("split_lookup")
    if split_lookup is not None and (
            not isinstance(split_lookup, dict) or
            split_lookup.get("mode") != "solver-exact-split3d-v1"):
        raise ValueError("unsupported exact split lookup in field receipt")
    expected = FIELD_ARTIFACTS | {frame_name}
    if split_lookup is not None:
        expected |= SPLIT_FIELD_ARTIFACTS
    artifacts = receipt.get("artifacts")
    if (not isinstance(artifacts, list) or len(artifacts) != len(expected) or
            {a.get("path") for a in artifacts} != expected):
        raise ValueError("field receipt does not declare the expected artifacts")
    for item in artifacts:
        path = _confined(field_dir, item["path"])
        if path.stat().st_size != item.get("bytes") or _sha(path) != item.get("sha256"):
            raise ValueError(f"field artifact identity differs: {path.name}")
    prepared = json.loads((field_dir / "prepared.json").read_text(encoding="utf-8"))
    if (_sha(field_dir / "parent-manifest.json") !=
            receipt.get("parent", {}).get("manifest_sha256")):
        raise ValueError("staged helium parent manifest differs from field receipt")
    if (prepared.get("schema_version") != "dsmc-prepared-v1" or
            prepared.get("state") != "solver-checked" or
            prepared.get("solver_check", {}).get("status") != "complete" or
            prepared.get("case_id") != receipt.get("case_id") or
            _sha(field_dir / "prepared.json") != receipt.get("prepared", {}).get("sha256")):
        raise ValueError("prepared case identity or solver check differs from field receipt")
    science = prepared["science_inputs"]
    adapter = science["geometry"].get("adapter")
    if adapter not in ("b5-circle", "b5-slit", "cad") or not prepared.get("source_case", {}).get("sha256"):
        raise ValueError("field lacks a supported geometry adapter or recorded source case identity")
    roles = json.loads((field_dir / "triangle-roles.json").read_text(encoding="utf-8"))
    if (roles.get("schema_version") != "dsmc-triangle-roles-v1" or
            not roles.get("records") or
            any(row.get("molecule_action") not in ("absorb", "exit", "ignore")
                for row in roles["records"])):
        raise ValueError("canonical roles are not runnable")
    if adapter == "cad":
        geometry = science["geometry"]
        solid_ids = geometry.get("solid_boundary_ids")
        reference = geometry.get("gas_reference_point_m")
        if (not isinstance(solid_ids, list) or not solid_ids or
                any(not isinstance(name, str) or not name.strip() for name in solid_ids) or
                len(solid_ids) != len(set(solid_ids))):
            raise ValueError("CAD solid_boundary_ids must be distinct nonempty names")
        if any(not isinstance(row.get("boundary_id"), str) or not row["boundary_id"]
               for row in roles["records"]):
            raise ValueError("CAD canonical triangle roles need boundary IDs")
        present = {row["boundary_id"] for row in roles["records"]}
        absorbing = {row.get("boundary_id") for row in roles["records"]
                     if row["molecule_action"] == "absorb"}
        if not set(solid_ids) <= present or not absorbing <= set(solid_ids):
            raise ValueError("CAD solid_boundary_ids omit a canonical physical boundary")
        bounds = science["numerics"]["domain_bounds_m"]
        if (not isinstance(bounds, list) or len(bounds) != 3 or
                any(not isinstance(axis, list) or len(axis) != 2 or
                    any(type(edge) not in (int, float) or not math.isfinite(edge)
                        for edge in axis) or axis[0] >= axis[1] for axis in bounds)):
            raise ValueError("CAD solver domain bounds are invalid")
        if (not isinstance(reference, list) or len(reference) != 3 or
                any(type(x) not in (int, float) or not math.isfinite(x) or
                    not bounds[i][0] < x < bounds[i][1]
                    for i, x in enumerate(reference))):
            raise ValueError("CAD gas_reference_point_m must be inside the solver domain")
    canonical = tomllib.loads((field_dir / "canonical-tracer.toml").read_text(encoding="utf-8"))
    if (canonical.get("schema_version") != "dsmc-canonical-tracer3d-v1" or
            canonical.get("canonical_surface_sha256") != _sha(field_dir / "physical-walls.surf") or
            canonical.get("triangle_roles_sha256") != _sha(field_dir / "triangle-roles.json") or
            canonical.get("tracer_surface_sha256") != _sha(field_dir / "cell.surfs") or
            canonical.get("tracer_roles_sha256") != _sha(field_dir / "triangle-roles.tsv") or
            canonical.get("triangle_count") != receipt.get("canonical_triangles")):
        raise ValueError("canonical tracer identity differs")
    provenance = json.loads((field_dir / "provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("source_dump_sha256") != receipt["frame"].get("sha256") or
            provenance.get("source_timestep") != receipt["frame"].get("timestep") or
            provenance.get("converter_sha256") != receipt.get("implementation_sha256", {}).get("field2tracer3d.py") or
            provenance.get("source_row_metadata", {}).get("sha256") != _sha(field_dir / "source-row-metadata.csv.gz")):
        raise ValueError("converted field provenance differs")
    if provenance.get("split_lookup") != split_lookup:
        raise ValueError("split lookup differs between field receipt and provenance")
    if split_lookup is not None:
        for key, name in (("map", "split-cell-map.tsv"),
                          ("field_rows", "split-field-rows.tsv"),
                          ("ids", "split-parent-ids.txt"),
                          ("ids_receipt", "split-parent-ids.txt.json"),
                          ("manifest", "split-field-manifest.toml")):
            item = split_lookup.get(key, {})
            if item.get("path") != name or item.get("sha256") != _sha(field_dir / name):
                raise ValueError(f"split lookup artifact differs: {name}")
        validation = split_lookup.get("source_validation", {})
        if (validation.get("frame_sha256") != receipt["frame"].get("sha256") or
                validation.get("splitmap_sha256") != _sha(field_dir / "split-cell-map.tsv") or
                validation.get("ids_sha256") != _sha(field_dir / "split-parent-ids.txt")):
            raise ValueError("split lookup is not bound to the selected helium frame")
        ids_receipt = json.loads((field_dir / "split-parent-ids.txt.json").read_text(encoding="utf-8"))
        if (ids_receipt.get("source_frame_sha256") != receipt["frame"].get("sha256") or
                ids_receipt.get("ids_sha256") != _sha(field_dir / "split-parent-ids.txt")):
            raise ValueError("split parent IDs are not bound to the selected helium frame")
    molecules = science["molecules"]
    declared = molecules["count"]
    actual = declared if count is None else count
    if type(declared) is not int or declared <= 0 or type(actual) is not int or actual <= 0:
        raise ValueError("molecule count must be a positive integer")
    if molecules["source"]["distribution"] not in ("point", "uniform-ball", "gaussian-ball"):
        raise ValueError("unsupported molecule source distribution")
    if molecules["flight"] not in ("local", "field-aware") or molecules["collision_rate_model"] not in ("approximate", "maxwell"):
        raise ValueError("unsupported flight or collision rate model")
    planes = science["observables"]["planes"]
    if not planes or len({p["name"] for p in planes}) != len(planes):
        raise ValueError("observation planes missing or duplicated")
    return receipt, prepared, actual


PATH_COLUMNS = ["particle_id", "leg", "x0_m", "y0_m", "z0_m", "x1_m", "y1_m", "z1_m",
                "vx_m_s", "vy_m_s", "vz_m_s", "collisions", "time_s",
                "initial_collisions"]


def read_recorded_paths(paths_csv: Path, terminal_csv: Path,
                        count: int) -> tuple[dict, dict]:
    """Return ({id: [leg rows]}, {id: fate}) for particles 1..count.

    Each prepared-case path must be contiguous and ordered, legs 1..L, each leg
    starting where the previous ended. Non-terminal leg i carries collision
    count c0+i, where c0 is the row's initial_collisions (1 after a zero-speed
    spawn collision, else 0); the last leg is the terminating one and must
    equal its terminal.csv row: contact point, velocity, collision count, time.
    """
    def floats(row, keys):
        return [float(row[key]) for key in keys]

    with Path(terminal_csv).open(encoding="utf-8", newline="") as stream:
        terminal = {int(row["particle_id"]): row for row in csv.DictReader(stream)}
    paths: dict[int, list] = {}
    with Path(paths_csv).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != PATH_COLUMNS:
            raise ValueError("paths.csv has unexpected columns")
        last = None
        for row in reader:
            pid, leg = int(row["particle_id"]), int(row["leg"])
            if (pid != last and pid in paths) or leg != len(paths.setdefault(pid, [])) + 1:
                raise ValueError(f"particle {pid}: missing or reordered leg at leg {leg}")
            paths[pid].append(row)
            last = pid
    if set(paths) != set(range(1, count + 1)):
        raise ValueError(f"paths.csv holds particles {sorted(paths)}, expected 1..{count}")
    for pid, legs in paths.items():
        end, final = legs[-1], terminal.get(pid)
        if final is None:
            raise ValueError(f"particle {pid} has no terminal.csv row")
        for before, after in zip(legs, legs[1:]):
            if (floats(before, ("x1_m", "y1_m", "z1_m")) != floats(after, ("x0_m", "y0_m", "z0_m"))
                    or float(after["time_s"]) <= float(before["time_s"])):
                raise ValueError(f"particle {pid}: leg {after['leg']} does not continue leg {before['leg']}")
        collisions = [int(row["collisions"]) for row in legs]
        c0 = {int(row["initial_collisions"]) for row in legs}
        c0 = c0.pop() if len(c0) == 1 else -1
        if (c0 not in (0, 1) or collisions[:-1] != list(range(c0 + 1, c0 + len(legs))) or
                collisions[-1] != c0 + len(legs) - 1):
            raise ValueError(f"particle {pid}: collision ordinals {collisions[:4]}... skip or repeat a leg")
        if (floats(end, ("x1_m", "y1_m", "z1_m", "vx_m_s", "vy_m_s", "vz_m_s", "time_s")) !=
                floats(final, ("x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "time_s")) or
                collisions[-1] != int(final["collisions"])):
            raise ValueError(f"particle {pid}: last recorded leg differs from its terminal.csv row")
    return paths, {pid: terminal[pid]["fate"] for pid in paths}


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(part) for part in value) + "]"
    raise ValueError(f"unsupported transport TOML value: {value!r}")


def _transport_config(field_dir: Path, prepared: dict, count: int,
                      trace_count: int = 0) -> str:
    science = prepared["science_inputs"]
    molecule = science["molecules"]
    source = molecule["source"]
    values = {
        "schema_version": "dsmc-molecule-transport-v1",
        "field_dir": str(field_dir), "count": count, "seed": molecule["seed"],
        **({"trace_count": trace_count} if trace_count else {}),
        "mass_u": molecule["mass_u"], "gas_mass_u": molecule["gas_mass_u"],
        "cross_section_m2": molecule["cross_section_m2"],
        "flight": molecule["flight"], "rate": molecule["collision_rate_model"],
        "flight_step_m": molecule["flight_step_m"],
        "source_distribution": source["distribution"],
        "source_center_m": source["center_m"], "source_radius_m": source["radius_m"],
        "source_drift_m_s": source["drift_m_s"],
        "source_temperature_K": source["temperature_K"],
        "domain_bounds_m": science["numerics"]["domain_bounds_m"],
        "geometry_adapter": science["geometry"]["adapter"],
    }
    if values["geometry_adapter"] == "cad":
        values["solid_boundary_ids"] = science["geometry"]["solid_boundary_ids"]
        values["gas_reference_point_m"] = science["geometry"]["gas_reference_point_m"]
    lines = [f"{key} = {_toml_value(value)}" for key, value in values.items()]
    for plane in science["observables"]["planes"]:
        lines.extend(["", "[[planes]]"])
        lines.extend(f"{key} = {_toml_value(plane[key])}" for key in
                     ("name", "origin_m", "normal", "axis_u", "axis_v"))
    return "\n".join(lines) + "\n"


def _finalize_trial(out: Path, prepared: dict, actual: int, receipt: dict,
                    *, batch: bool = False, trace_count: int = 0) -> None:
    result_path = out / "transport-result.toml"
    result = tomllib.loads(result_path.read_text(encoding="utf-8"))
    if (result.get("schema_version") != "dsmc-molecule-transport-result-v1" or
            result.get("status") != "complete" or
            result.get("launched_particles") != actual or
            result.get("recorded_paths", 0) != trace_count):
        raise ValueError("Julia transport result does not evidence the requested launch")
    events = out / "events.csv"
    terminal = out / "terminal.csv"
    if not events.is_file() or not terminal.is_file():
        raise ValueError("Julia transport did not write events.csv and terminal.csv")
    outputs = {"events.csv", "terminal.csv"} | ({"paths.csv"} if trace_count else set())
    output_artifacts = result.get("artifacts")
    if (not isinstance(output_artifacts, list) or len(output_artifacts) != len(outputs) or
            {item.get("path") for item in output_artifacts} != outputs):
        raise ValueError("Julia transport result lacks output artifact identities")
    for item in output_artifacts:
        path = out / item["path"]
        if path.stat().st_size != item.get("bytes") or _sha(path) != item.get("sha256"):
            raise ValueError(f"Julia transport output identity differs: {path.name}")
    summary = analyze_observable_events(
        events, prepared["science_inputs"]["observables"]["planes"], actual)
    if summary["event_count"] != result.get("event_rows"):
        raise ValueError("Julia event-row count differs from analyzed events")
    with terminal.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows, None)
        if not header or len(header) != len(set(header)) or any(not key for key in header):
            raise ValueError("Julia terminal CSV lacks a valid header")
        terminal_rows = 0
        for row in rows:
            if len(row) != len(header):
                raise ValueError("Julia terminal CSV has a malformed row")
            terminal_rows += 1
    if terminal_rows != actual:
        raise ValueError("Julia terminal-row count differs from launched molecules")
    if trace_count:
        read_recorded_paths(out / "paths.csv", terminal, trace_count)
    report = {
        "schema_version": "dsmc-molecule-report-v1",
        "case_id": prepared["case_id"], "field": receipt["field"],
        "helium_parent": receipt["helium_parent"], "frame": receipt["frame"],
        "source": prepared["science_inputs"]["molecules"]["source"],
        "declared_count": receipt["declared_count"],
        "transport": result, "observables": summary,
        "science_assessment": "unassessed",
    }
    if batch:
        report["batch_trial"] = receipt["batch_trial"]
        report["effective_molecules"] = prepared["science_inputs"]["molecules"]
    _write_json(out / "report.json", report)
    names = ("transport.toml", "transport-result.toml", "events.csv",
             "terminal.csv", "report.json")
    if not batch:
        names = ("transport.toml", "transport.log", "transport-result.toml",
                 "events.csv", "terminal.csv", "report.json")
    if trace_count:
        names += ("paths.csv",)
    receipt["artifacts"] = [_artifact(out / name) for name in names]
    receipt["status"] = "complete"


def run_molecules(field_dir: Path, out: Path, *, count: int | None = None,
                  trace_count: int = 0) -> Path:
    """Verify parent before creation; retain a failed child receipt after creation.

    trace_count > 0 records the full paths of molecules 1..trace_count to
    paths.csv (see read_recorded_paths); events.csv and terminal.csv are
    unchanged by recording.
    """
    field_dir = Path(field_dir).resolve()
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"molecule output already exists: {out}")
    field_receipt, prepared, actual = _preflight(field_dir, count)
    if type(trace_count) is not int or not 0 <= trace_count <= actual:
        raise ValueError(f"trace_count must be an integer from 0 to the launched count {actual}")
    if field_dir == out or field_dir in out.parents or out in field_dir.parents:
        raise ValueError("molecule output must be separate from field directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    receipt_path = out / "molecule-run.json"
    command = ["julia", "--project=tracer", "--threads=1",
               "tracer/PreparedCase3D.jl", str(out / "transport.toml")]
    receipt = {
        "schema_version": "dsmc-molecule-run-v1", "status": "staging",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": prepared["case_id"],
        "field": {"path": str(field_dir),
                  "receipt_sha256": _sha(field_dir / "field-preparation.json"),
                  "artifacts": field_receipt["artifacts"]},
        "helium_parent": field_receipt["parent"], "frame": field_receipt["frame"],
        "declared_count": prepared["science_inputs"]["molecules"]["count"],
        "launched_count": actual, "command": command,
    }
    if trace_count:
        receipt["trace_count"] = trace_count
    _write_json(receipt_path, receipt)
    try:
        config = out / "transport.toml"
        config.write_text(_transport_config(field_dir, prepared, actual, trace_count),
                          encoding="utf-8", newline="\n")
        receipt["status"] = "transporting"
        _write_json(receipt_path, receipt)
        with (out / "transport.log").open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        receipt["exit_code"] = completed.returncode
        if completed.returncode:
            raise ValueError(f"Julia transport exited with code {completed.returncode}")
        _finalize_trial(out, prepared, actual, receipt, trace_count=trace_count)
    except BaseException as exc:
        receipt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        receipt["failure"] = str(exc)
        _write_json(receipt_path, receipt)
        raise
    _write_json(receipt_path, receipt)
    return receipt_path


def _batch_trials(data: dict, prepared: dict) -> list[tuple[str, dict]]:
    if (not isinstance(data, dict) or set(data) != {"schema_version", "trials"} or
            data["schema_version"] != "dsmc-molecule-batch-v1" or
            not isinstance(data["trials"], list) or not data["trials"]):
        raise ValueError("expected dsmc-molecule-batch-v1 with a nonempty trials list")
    allowed = {"count", "seed", "mass_u", "gas_mass_u", "cross_section_m2",
               "flight", "collision_rate_model", "flight_step_m", "source"}
    source_allowed = {"distribution", "center_m", "radius_m", "drift_m_s",
                      "temperature_K"}
    choices = {"flight": {"local", "field-aware"},
               "collision_rate_model": {"approximate", "maxwell"},
               "distribution": {"point", "uniform-ball", "gaussian-ball"}}
    positive = {"mass_u", "gas_mass_u", "cross_section_m2", "flight_step_m"}
    reserved = {"con", "prn", "aux", "nul"} | {
        f"{prefix}{number}" for prefix in ("com", "lpt") for number in range(1, 10)}
    seen = set()
    resolved = []
    for item in data["trials"]:
        if not isinstance(item, dict) or set(item) != {"name", "overrides"}:
            raise ValueError("each batch trial needs name and overrides only")
        name, overrides = item["name"], item["overrides"]
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or
                name.casefold() in seen or name.casefold() in reserved):
            raise ValueError(f"invalid or duplicate batch trial name: {name!r}")
        seen.add(name.casefold())
        if not isinstance(overrides, dict) or not set(overrides) <= allowed:
            raise ValueError(f"unsupported overrides in trial {name}")
        trial = copy.deepcopy(prepared)
        molecule = trial["science_inputs"]["molecules"]
        for key, value in overrides.items():
            if key == "source":
                if not isinstance(value, dict) or not set(value) <= source_allowed:
                    raise ValueError(f"unsupported source overrides in trial {name}")
                molecule["source"].update(value)
            else:
                molecule[key] = value
        for key in positive:
            value = molecule[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"trial {name}: {key} must be positive and finite")
        radius, temperature = (molecule["source"][key] for key in ("radius_m", "temperature_K"))
        if (type(radius) not in (int, float) or not math.isfinite(radius) or
                type(temperature) not in (int, float) or not math.isfinite(temperature) or
                temperature < 0):
            raise ValueError(f"trial {name}: source radius and temperature must be finite and nonnegative")
        for key in ("center_m", "drift_m_s"):
            value = molecule["source"][key]
            if (not isinstance(value, list) or len(value) != 3 or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
                raise ValueError(f"trial {name}: {key} must be a finite 3-vector")
        for key in ("flight", "collision_rate_model"):
            if not isinstance(molecule[key], str) or molecule[key] not in choices[key]:
                raise ValueError(f"trial {name}: unsupported {key}")
        distribution = molecule["source"]["distribution"]
        if not isinstance(distribution, str) or distribution not in choices["distribution"]:
            raise ValueError(f"trial {name}: unsupported source distribution")
        if (distribution == "point" and radius != 0 or
                distribution != "point" and radius <= 0):
            raise ValueError(f"trial {name}: point radius must be zero; extended source radius positive")
        count = molecule["count"]
        if type(count) is not int or count <= 0:
            raise ValueError(f"trial {name}: count must be a positive integer")
        seed = molecule["seed"]
        if (not isinstance(seed, str) or not re.fullmatch(r"[1-9][0-9]*", seed) or
                not 0 < int(seed) < 2**64):
            raise ValueError(f"trial {name}: seed must be a nonzero decimal UInt64 string")
        resolved.append((name, trial))
    return resolved


def run_molecule_batch(field_dir: Path, trials_path: Path, out: Path) -> Path:
    """Run independently reported trials in one Julia process on one checked field."""
    field_dir = Path(field_dir).resolve()
    trials_path = Path(trials_path).resolve()
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"molecule batch output already exists: {out}")
    if field_dir == out or field_dir in out.parents or out in field_dir.parents:
        raise ValueError("molecule batch output must be separate from field directory")
    if not trials_path.is_file():
        raise ValueError("batch trials JSON is missing")
    field_receipt, prepared, _ = _preflight(field_dir, None)
    spec_bytes = trials_path.read_bytes()
    trials = _batch_trials(json.loads(spec_bytes.decode("utf-8-sig")), prepared)
    spec_sha = hashlib.sha256(spec_bytes).hexdigest()
    command = ["julia", "--project=tracer", "--threads=1",
               "tracer/PreparedCase3D.jl"]
    for name, _ in trials:
        command.append(str(out / name / "transport.toml"))
    field_identity = {"path": str(field_dir),
                      "receipt_sha256": _sha(field_dir / "field-preparation.json"),
                      "artifacts": field_receipt["artifacts"]}
    batch_receipt = {
        "schema_version": "dsmc-molecule-batch-run-v1", "status": "staging",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": prepared["case_id"], "field": field_identity,
        "helium_parent": field_receipt["parent"], "frame": field_receipt["frame"],
        "trials": {"source_path": str(trials_path), "snapshot": "trials.json",
                   "bytes": len(spec_bytes), "sha256": spec_sha},
        "command": command, "children": {},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    batch_path = out / "molecule-batch.json"
    _write_json(batch_path, batch_receipt)
    children = []
    try:
        (out / "trials.json").write_bytes(spec_bytes)
        for index, (name, trial) in enumerate(trials):
            child = out / name
            child.mkdir()
            molecule = trial["science_inputs"]["molecules"]
            actual = molecule["count"]
            config = child / "transport.toml"
            config.write_text(_transport_config(field_dir, trial, actual),
                              encoding="utf-8", newline="\n")
            receipt = {
                "schema_version": "dsmc-molecule-run-v1", "status": "staging",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "case_id": prepared["case_id"], "field": field_identity,
                "helium_parent": field_receipt["parent"], "frame": field_receipt["frame"],
                "declared_count": prepared["science_inputs"]["molecules"]["count"],
                "launched_count": actual,
                "batch_trial": {"name": name, "index": index,
                                "trials_sha256": spec_sha},
                "command": command,
            }
            child_receipt = child / "molecule-run.json"
            _write_json(child_receipt, receipt)
            children.append((name, child, trial, actual, receipt))
        batch_receipt["status"] = "transporting"
        _write_json(batch_path, batch_receipt)
        with (out / "transport.log").open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        batch_receipt["exit_code"] = completed.returncode
        batch_receipt["transport_log"] = _artifact(out / "transport.log")
        failures = []
        for name, child, trial, actual, receipt in children:
            try:
                _finalize_trial(child, trial, actual, receipt, batch=True)
            except Exception as exc:
                receipt["status"] = "failed"
                receipt["failure"] = str(exc)
                failures.append(name)
            _write_json(child / "molecule-run.json", receipt)
            batch_receipt["children"][name] = {
                "status": receipt["status"],
                "path": str(child / "molecule-run.json"),
                "sha256": _sha(child / "molecule-run.json"),
            }
        if completed.returncode or failures:
            raise ValueError(f"batch transport exited {completed.returncode}; failed trials: {failures}")
        batch_receipt["status"] = "complete"
    except BaseException as exc:
        batch_receipt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        batch_receipt["failure"] = str(exc)
        for _, child, _, _, receipt in children:
            if receipt["status"] == "staging":
                receipt["status"] = batch_receipt["status"]
                receipt["failure"] = str(exc)
                _write_json(child / "molecule-run.json", receipt)
            batch_receipt["children"][receipt["batch_trial"]["name"]] = {
                "status": receipt["status"],
                "path": str(child / "molecule-run.json"),
                "sha256": _sha(child / "molecule-run.json"),
            }
        _write_json(batch_path, batch_receipt)
        raise
    _write_json(batch_path, batch_receipt)
    return batch_path
