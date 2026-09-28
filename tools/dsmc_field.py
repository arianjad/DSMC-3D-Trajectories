"""Freeze one selected helium frame as a provenance-bound 3D tracer field."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dsmc_case import inspect_geometry
from dsmc_frames import select_frame, validate_frame
from dsmc_resume_parent import SCHEMA as RESUME_SCHEMA


FIELD_POLICIES = {
    "positive-T-only": (False, False),
    "positive-T-and-empty-ballistic": (False, True),
    "carry-T": (True, False),
    "carry-T-and-empty-ballistic": (True, True),
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


def _same_box(left, right, tolerance=1e-12) -> bool:
    return (len(left) == len(right) == 3 and
            all(len(left[axis]) == len(right[axis]) == 2 and
                all(abs(left[axis][edge] - right[axis][edge]) <= tolerance
                    for edge in range(2)) for axis in range(3)))


def _preflight(prepared_dir: Path, parent_dir: Path, selection: dict, policy: str) -> tuple:
    if policy not in FIELD_POLICIES:
        raise ValueError(f"unsupported gas-field policy: {policy}")
    inspection = inspect_geometry(prepared_dir)
    if inspection["state"] != "solver-checked":
        raise ValueError("field preparation requires a solver-checked case")
    prepared_path = prepared_dir / "prepared.json"
    parent_path = parent_dir / "manifest.json"
    if not parent_path.is_file():
        raise ValueError(f"helium parent lacks a run manifest: {parent_path}")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    prepared_hash = _sha(prepared_path)
    imported = parent.get("schema_version") == "dsmc-cad-helium-import-v1"
    if parent.get("schema_version") not in ("dsmc-helium-attempt-v1", RESUME_SCHEMA,
                                            "dsmc-cad-helium-import-v1"):
        raise ValueError("helium parent has an unsupported manifest schema")
    if imported:
        from dsmc_cad_workflow import validate_imported_cad, validate_prepared_cad
        validate_prepared_cad(prepared_dir)
        validate_imported_cad(parent_dir, parent)
    parent_prepared = parent.get("prepared")
    if (parent.get("case_id") != prepared["case_id"] or
            not isinstance(parent_prepared, dict) or
            parent_prepared.get("sha256") != prepared_hash):
        raise ValueError("helium parent is not bound to this solver-checked preparation")
    roles_path = prepared_dir / "triangle-roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    if (roles.get("schema_version") != "dsmc-triangle-roles-v1" or
            not roles.get("records") or
            any(row.get("molecule_action") not in ("absorb", "exit", "ignore")
                for row in roles["records"])):
        raise ValueError("canonical triangle roles lack runnable molecule actions")
    frame = select_frame(parent_dir, selection)
    if frame["parent_manifest_sha256"] != _sha(parent_path):
        raise ValueError("helium parent manifest changed during frame selection")
    numerics = prepared["science_inputs"]["numerics"]
    expected_box = numerics["domain_bounds_m"]
    if not _same_box(frame["box_bounds_m"], expected_box):
        raise ValueError("selected helium frame box differs from prepared case")
    step = frame["timestep"]
    every = numerics["field_every_steps"]
    effective_end = (parent["end_step"] if parent["schema_version"] == RESUME_SCHEMA or imported
                     else numerics["total_steps"])
    if step <= 0 or step > effective_end or step % every:
        raise ValueError("selected helium frame timestep is outside the declared field schedule")
    if parent["schema_version"] == RESUME_SCHEMA:
        windows = [window for window in parent["frame_windows"] if window["timestep"] == step]
        if len(windows) != 1:
            raise ValueError("selected resumed field lacks one verified complete sampling window")
        frame["sample_window"] = windows[0]
    elif imported:
        windows = [window for window in parent["frame_windows"] if window["timestep"] == step]
        if len(windows) != 1:
            raise ValueError("imported CAD field lacks its original complete sampling window")
        frame["sample_window"] = windows[0]
    return prepared, prepared_hash, frame


def prepare_field(prepared: Path, parent: Path, selection: dict,
                  policy: str, out: Path, *, splitmap: Path | None = None,
                  splitmap_ids: Path | None = None) -> Path:
    """Create a new child directory; never alter the parent or prepared case."""
    prepared_dir = Path(prepared).resolve()
    if prepared_dir.is_file():
        if prepared_dir.name != "prepared.json":
            raise ValueError("field preparation needs a prepared directory or prepared.json")
        prepared_dir = prepared_dir.parent
    parent_dir = Path(parent).resolve()
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"field output already exists: {out}")
    if (splitmap is None) != (splitmap_ids is None):
        raise ValueError("field preparation needs both split map and parent IDs")
    if splitmap is not None:
        splitmap = Path(splitmap).resolve()
        splitmap_ids = Path(splitmap_ids).resolve()
        if not splitmap.is_file() or not splitmap_ids.is_file():
            raise ValueError("split map or selected parent IDs are missing")
    prepared_data, prepared_hash, frame = _preflight(
        prepared_dir, parent_dir, selection, policy)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    receipt_path = out / "field-preparation.json"
    receipt = {
        "schema_version": "dsmc-field-preparation-v1",
        "status": "staging", "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_id": prepared_data["case_id"],
        "prepared": {"path": str(prepared_dir), "sha256": prepared_hash},
        "parent": {"path": str(parent_dir),
                   "manifest_sha256": frame["parent_manifest_sha256"],
                   "manifest_status": frame["parent_manifest_status"]},
        "frame": frame, "requested_policy": policy,
        "splitmap_input": (None if splitmap is None else {
            "map_path": str(splitmap), "map_sha256": _sha(splitmap),
            "ids_path": str(splitmap_ids), "ids_sha256": _sha(splitmap_ids)}),
        "implementation_sha256": {
            name: _sha(Path(__file__).with_name(name)) for name in
            ("dsmc.py", "dsmc_case.py", "dsmc_field.py", "dsmc_frames.py",
             "field2tracer3d.py")},
        "sample_window": {"status": "declared-schedule-only",
                          **prepared_data["science_inputs"]["numerics"]["field_average"]},
    }
    if "sample_window" in frame:
        status = ("verified-imported-pilot-window" if frame["parent_manifest_status"] == "complete" and
                  prepared_data["science_inputs"]["geometry"]["adapter"] == "cad" and
                  json.loads((parent_dir / "manifest.json").read_text())["schema_version"] ==
                  "dsmc-cad-helium-import-v1" else "verified-resumed-window")
        receipt["sample_window"] = {"status": status, **frame["sample_window"]}
    _write_json(receipt_path, receipt)
    try:
        shutil.copyfile(prepared_dir / "prepared.json", out / "prepared.json")
        shutil.copyfile(parent_dir / "manifest.json", out / "parent-manifest.json")
        staged_frame = out / Path(frame["path"]).name
        shutil.copyfile(frame["path"], staged_frame)
        staged = validate_frame(staged_frame, expected_step=frame["timestep"])
        if (staged["sha256"] != frame["sha256"] or
                staged["rows"] != frame["rows"] or
                _sha(out / "prepared.json") != prepared_hash or
                _sha(out / "parent-manifest.json") != frame["parent_manifest_sha256"]):
            raise ValueError("staged parent, preparation or frame differs from selection")
        receipt["status"] = "converting"
        _write_json(receipt_path, receipt)
        from field2tracer3d import convert3d
        keep_unsampled, keep_zero_density = FIELD_POLICIES[policy]
        converted = convert3d(
            out, out, dumpname=staged_frame.name,
            canonical_surface=prepared_dir / "physical-walls.surf",
            triangle_roles=prepared_dir / "triangle-roles.json",
            base_grid=prepared_data["science_inputs"]["numerics"]["base_grid"],
            keep_unsampled=keep_unsampled, keep_zero_density=keep_zero_density,
            splitmap=splitmap, splitmap_ids=splitmap_ids)
        if (converted["source_dump_sha256"] != frame["sha256"] or
                converted["source_timestep"] != frame["timestep"] or
                converted["rows_read"] != frame["rows"] or
                not _same_box(converted["box_bounds"], frame["box_bounds_m"])):
            raise ValueError("converted field differs from selected helium frame")
        metadata = converted["source_row_metadata"]
        metadata_path = out / metadata["path"]
        if (metadata_path.stat().st_size != metadata["bytes"] or
                _sha(metadata_path) != metadata["sha256"]):
            raise ValueError("converted source-row metadata differs from provenance")
        names = ("prepared.json", "parent-manifest.json", staged_frame.name,
                 "DS3FF.DAT", "source-row-metadata.csv.gz", "cell.surfs",
                 "physical-walls.surf", "triangle-roles.json", "triangle-roles.tsv",
                 "canonical-tracer.toml", "provenance.json")
        if converted["split_lookup"] is not None:
            names += ("split-cell-map.tsv", "split-field-rows.tsv",
                      "split-parent-ids.txt", "split-parent-ids.txt.json",
                      "split-field-manifest.toml")
        receipt["artifacts"] = [_artifact(out / name) for name in names]
        receipt["split_lookup"] = converted["split_lookup"]
        receipt["field_policy"] = converted["field_policy"]
        receipt["field_cells"] = converted["emitted_cells"]
        receipt["canonical_triangles"] = converted["emitted_triangles"]
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        receipt["failure"] = str(exc)
        _write_json(receipt_path, receipt)
        raise
    _write_json(receipt_path, receipt)
    return receipt_path
