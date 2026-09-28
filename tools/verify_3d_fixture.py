#!/usr/bin/env python3
"""Exercise real Julia transport on the tracked tiny synthetic 3D field."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from dsmc_report import analyze_observable_events


ROOT = Path(__file__).resolve().parents[1]
FIELD = ROOT / "tests" / "fixtures" / "3d" / "tube"
PLANES = [{"name": "plume", "origin_m": [1.5, 0, 0],
           "normal": [1, 0, 0], "axis_u": [0, 1, 0], "axis_v": [0, 0, 1],
           "acceptance": {"position_u_m": None, "position_v_m": None,
                          "angle_rad": None, "time_s": None}}]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _field_identity() -> list[dict]:
    manifest = tomllib.loads((FIELD / "canonical-tracer.toml").read_text(encoding="utf-8"))
    expected = {"canonical_surface_sha256": "physical-walls.surf",
                "triangle_roles_sha256": "triangle-roles.json",
                "tracer_surface_sha256": "cell.surfs",
                "tracer_roles_sha256": "triangle-roles.tsv"}
    if manifest.get("schema_version") != "dsmc-canonical-tracer3d-v1" or manifest.get("triangle_count") != 10:
        raise ValueError("small fixture has an invalid canonical format manifest")
    for key, name in expected.items():
        if _sha(FIELD / name) != manifest.get(key):
            raise ValueError(f"small fixture identity differs: {name}")
    roles = json.loads((FIELD / "triangle-roles.json").read_text(encoding="utf-8"))
    if (roles.get("schema_version") != "dsmc-triangle-roles-v1" or
            roles.get("surface_sha256") != _sha(FIELD / "physical-walls.surf") or
            [row.get("triangle_id") for row in roles.get("records", [])] != list(range(1, 11))):
        raise ValueError("small fixture triangle roles do not match its surface")
    names = ["DS3FF.DAT", "canonical-tracer.toml", *expected.values()]
    return [{"path": name, "bytes": (FIELD / name).stat().st_size,
             "sha256": _sha(FIELD / name)} for name in names]


def _config(field: Path) -> str:
    # All positions and bounds below are in solver (x,y,z) order.
    values = {
        "schema_version": "dsmc-molecule-transport-v1", "field_dir": str(field),
        "count": 4, "seed": "42", "mass_u": 156.325,
        "gas_mass_u": 4.002603, "cross_section_m2": 1e-20,
        "flight": "local", "rate": "approximate", "flight_step_m": 0.1,
        "source_distribution": "point", "source_center_m": [0.5, 0, 0],
        "source_radius_m": 0.0, "source_drift_m_s": [1000, 0, 0],
        "source_temperature_K": 0.0,
        "domain_bounds_m": [[-1, 2], [-2, 2], [-2, 2]],
        "geometry_adapter": "b5-circle",
    }
    lines = [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    for plane in PLANES:
        lines.extend(["", "[[planes]]"])
        lines.extend(f"{key} = {json.dumps(plane[key])}" for key in
                     ("name", "origin_m", "normal", "axis_u", "axis_v"))
    return "\n".join(lines) + "\n"


def verify(out: Path) -> Path:
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"fixture check output already exists: {out}")
    field = FIELD.resolve()
    identity = _field_identity()
    if out == field or out in field.parents or field in out.parents:
        raise ValueError("fixture output must be separate from tracked input")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    receipt_path = out / "fixture-check.json"
    command = ["julia", "--project=tracer", "--threads=1",
               "tracer/PreparedCase3D.jl", str(out / "transport.toml")]
    receipt = {"schema_version": "dsmc-3d-small-fixture-check-v1",
               "status": "staging", "created_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "synthetic field/geometry; no SPARTA parent or science claim",
               "field": {"path": str(field), "artifacts": identity}, "command": command}
    _write(receipt_path, receipt)
    try:
        (out / "transport.toml").write_text(_config(field), encoding="utf-8", newline="\n")
        receipt["status"] = "transporting"
        _write(receipt_path, receipt)
        with (out / "transport.log").open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, check=False)
        receipt["exit_code"] = completed.returncode
        if completed.returncode:
            raise ValueError(f"small-fixture Julia transport exited {completed.returncode}")
        result = tomllib.loads((out / "transport-result.toml").read_text(encoding="utf-8"))
        if (result.get("schema_version") != "dsmc-molecule-transport-result-v1" or
                result.get("status") != "complete" or result.get("launched_particles") != 4):
            raise ValueError("small-fixture transport result is incomplete")
        artifacts = result.get("artifacts")
        if (not isinstance(artifacts, list) or len(artifacts) != 2 or
                {item.get("path") for item in artifacts} != {"events.csv", "terminal.csv"}):
            raise ValueError("small-fixture transport result lacks output identities")
        for item in artifacts:
            path = out / item["path"]
            if path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
                raise ValueError(f"small-fixture output identity differs: {path.name}")
        summary = analyze_observable_events(out / "events.csv", PLANES, 4)
        if summary["event_count"] != result.get("event_rows"):
            raise ValueError("small-fixture event count differs from Julia result")
        with (out / "terminal.csv").open("r", encoding="utf-8", newline="") as stream:
            terminal = list(csv.DictReader(stream))
        if (len(terminal) != 4 or result.get("domain_or_cap_exits") != 4 or
                result.get("missing_cell_queries") != 0 or summary["event_count"] != 4):
            raise ValueError("small-fixture expected four clean forward passages")
        receipt["result"] = result
        receipt["observables"] = summary
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        receipt["failure"] = str(exc)
        _write(receipt_path, receipt)
        raise
    _write(receipt_path, receipt)
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.out)
    except (ValueError, FileExistsError, OSError, KeyError) as exc:
        parser.exit(2, f"verify_3d_fixture: {exc}\n")
    print(f"Small fixture check: {result}")


if __name__ == "__main__":
    main()
