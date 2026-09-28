#!/usr/bin/env python3
"""Independently recheck a completed supported 3D workflow and its full field."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dsmc_molecule import _preflight
from dsmc_resume_parent import SCHEMA as RESUME_SCHEMA, validate_resume_parent


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cases" / "b5-slit-pair"))
from check_sample_probes import check_field, read_dump  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_ref(ref: dict) -> Path:
    path = Path(ref["path"])
    if not path.is_file() or path.stat().st_size != ref["bytes"] or _sha(path) != ref["sha256"]:
        raise ValueError(f"receipt-referenced file changed: {path}")
    return path


def _checked_output(root: Path, item: dict) -> Path:
    """Include retained checkpoint context without allowing an escaping path."""
    name = item["path"]
    if not isinstance(name, str) or not name:
        raise ValueError(f"unconfined helium output name: {name}")
    rel = Path(name)
    if (rel.is_absolute() or name.startswith("/") or
            ".." in rel.parts or "\\" in name or ":" in name):
        raise ValueError(f"unconfined helium output name: {name}")
    path = root / rel
    if (root.resolve() not in path.resolve().parents or
            any(part.is_symlink() for part in (path, *path.parents) if part != root.parent) or
            not path.is_file() or path.stat().st_size != item["bytes"] or
            _sha(path) != item["sha256"]):
        raise ValueError(f"helium parent output identity differs: {name}")
    return path


def _cad_diagnostics(helium: Path, prepared: dict, log: str, frame: dict,
                     end_step: int, *, imported: bool = False,
                     resumed: bool = False) -> dict:
    if imported and resumed:
        raise ValueError("CAD parent cannot be both imported and resumed")
    from dsmc_frames import validate_frame
    checked = validate_frame(helium / Path(frame["path"]).name, expected_step=frame["timestep"])
    bounds = prepared["science_inputs"]["numerics"]["domain_bounds_m"]
    if any(abs(a-b) > 1e-12 for left, right in zip(bounds, checked["box_bounds_m"])
           for a, b in zip(left, right)):
        raise ValueError("CAD field domain differs from its declared preparation")
    for output in prepared["declared_outputs"]:
        # A resumed parent's continued.restart inventory is already checked by
        # validate_resume_parent in _verify. Producer checkpoint names belong
        # to the fresh attempt, not the continuation's output contract.
        if (imported or resumed) and output.get("role") in ("helium_restart", "helium_final_restart"):
            continue
        if output["required"] and not any(p.is_file() and p.stat().st_size
                                          for p in helium.glob(output["pattern"])):
            raise ValueError(f"CAD required output is missing: {output['pattern']}")
    if imported and not (helium / f"particles.{end_step}").is_file():
        raise ValueError("imported CAD parent lacks its original final particle snapshot")
    # The CAD adapter fixes this stats layout in its checked deck. Values are
    # simulation-particle counts, not physical atom inventories under weighting.
    records = []
    for line in log.splitlines():
        cols = line.split()
        if len(cols) != 5:
            continue
        try:
            values = [float(value) for value in cols]
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in values):
            raise ValueError("CAD source statistics contain nonfinite values")
        if values[0] == end_step:
            records.append(values)
    if not records:
        raise ValueError("CAD final source statistics are missing")
    step, _, particles, inserted_step, inserted_total = records[-1]
    if (any(value < 0 or not value.is_integer() for value in
            (step, particles, inserted_step, inserted_total)) or inserted_total <= 0 or
            inserted_step > inserted_total):
        raise ValueError("CAD source statistics are invalid or contain no emission")
    return {"final_simulation_particles": int(particles),
            "inserted_final_step": int(inserted_step), "inserted_cumulative": int(inserted_total),
            "count_scope": "simulation-particle source counter for this attempt"}


def _splitmap_diagnostic(workflow: Path, prepared_hash: str,
                         helium: Path, frame_step: int) -> dict:
    """Recheck a retained zero-step CAD map against its exact source frame."""
    probe_dir = workflow / "splitmap"
    receipt_path = probe_dir / "splitmap-probe.json"
    probe = json.loads(receipt_path.read_text(encoding="utf-8"))
    parent_hash = _sha(helium / "manifest.json")
    frame_name = f"fields.{frame_step}.gz"
    frame_hash = _sha(helium / frame_name)
    selection = probe.get("selection", {})
    validation = probe.get("validation", {})
    status = probe.get("status")
    if (probe.get("schema_version") != "dsmc-cad-splitmap-probe-v1" or
            status not in ("complete", "not-needed") or
            probe.get("advanced_steps") != 0 or probe.get("ranks") != 1 or
            probe.get("prepared_sha256") != prepared_hash or
            probe.get("parent_manifest_sha256") != parent_hash or
            probe.get("timestep") != frame_step or
            selection.get("schema_version") != "sparta-splitmap-ids-v1" or
            selection.get("source_manifest_sha256") != parent_hash or
            selection.get("source_frame") != frame_name or
            selection.get("source_frame_sha256") != frame_hash):
        raise ValueError("CAD split-map probe is not bound to the selected helium frame")
    items = probe.get("artifacts")
    if not isinstance(items, list):
        raise ValueError("CAD split-map probe lacks an artifact inventory")
    if any(not isinstance(item, dict) or type(item.get("path")) is not str for item in items):
        raise ValueError("CAD split-map artifact inventory is malformed")
    names = {item["path"] for item in items}
    required = {"split-parent-ids.txt", "split-parent-ids.txt.json"}
    if status == "complete":
        required |= {"he.species", "he.vss", "in.splitmap", "probe.stdout",
                     "query-probe.stdout", "physical-walls.surf", "splitmap.tsv",
                     "queries.txt", "queries.txt.json", "answers.tsv"}
    if len(names) != len(items) or not required <= names:
        raise ValueError("CAD split-map probe omits required diagnostic artifacts")
    for item in items:
        name = item["path"]
        if type(name) is not str or Path(name).name != name:
            raise ValueError("CAD split-map artifact name is not flat")
        _checked_output(probe_dir, item)
    ids = probe_dir / "split-parent-ids.txt"
    sidecar = json.loads((probe_dir / "split-parent-ids.txt.json").read_text(encoding="utf-8"))
    if sidecar != selection or selection.get("ids_sha256") != _sha(ids):
        raise ValueError("CAD split-map selected IDs or sidecar differ")
    if status == "not-needed":
        if (selection.get("split_parent_ids") != 0 or validation or
                probe.get("native_probe_executed") is not False or
                "point_parity" in probe or "splitmap.tsv" in names or
                "exit_code" in probe):
            raise ValueError("CAD zero-split diagnostic incorrectly claims a native map")
        return {"status": "not-needed", "receipt_sha256": _sha(receipt_path),
                "split_parents": 0, "solver_execution": "not-run"}
    mapping = probe_dir / "splitmap.tsv"
    if (probe.get("exit_code") != 0 or probe.get("native_probe_executed") is not True or
            selection.get("split_parent_ids", 0) <= 0 or
            validation.get("schema_version") != "sparta-splitmap-frame-validation-v1" or
            validation.get("frame_sha256") != frame_hash or
            validation.get("ids_sha256") != _sha(ids) or
            validation.get("splitmap_sha256") != _sha(mapping) or
            validation.get("split_parents") != selection["split_parent_ids"] or
            probe.get("point_parity", {}).get("mismatch_count") != 0):
        raise ValueError("CAD completed split-map validation or point parity differs")
    from sparta_splitmap.validate_map import validate
    if validate(helium, frame_name, mapping, ids) != validation:
        raise ValueError("CAD split-map does not revalidate against the retained field")
    from sparta_splitmap.compare_queries import compare
    if compare(mapping, probe_dir / "queries.txt", probe_dir / "answers.tsv") != probe["point_parity"]:
        raise ValueError("CAD split-map point parity differs from retained native answers")
    for name in ("probe.stdout", "query-probe.stdout"):
        if not re.search(r"Loop time of [\d.eE+-]+ on 1 procs for 0 steps",
                         (probe_dir / name).read_text(encoding="utf-8", errors="replace")):
            raise ValueError("CAD split-map diagnostic log does not show zero steps")
    return {"status": "verified", "receipt_sha256": _sha(receipt_path),
            "solver_sha256": probe["solver"]["sha256"],
            "split_parents": validation["split_parents"],
            "mapped_frame_rows": validation["mapped_frame_rows"]}


def _verify(workflow: Path) -> dict:
    receipt = json.loads((workflow / "workflow.json").read_text(encoding="utf-8"))
    if receipt.get("schema_version") != "dsmc-local-workflow-v1" or receipt.get("status") != "complete":
        raise ValueError("local workflow receipt is not complete")
    for key in ("prepared", "profile"):
        _check_ref(receipt[key])
    children = receipt.get("children", {})
    if set(children) not in ({"helium", "field", "molecules"},
                             {"helium", "field", "molecules", "splitmap"}):
        raise ValueError("workflow lacks one of its three child receipts")
    paths = {key: _check_ref(ref) for key, ref in children.items()}
    parent = json.loads(paths["helium"].read_text(encoding="utf-8"))
    field_receipt, prepared, declared_count = _preflight(paths["field"].parent, None)
    molecule = json.loads(paths["molecules"].read_text(encoding="utf-8"))
    if (parent.get("status") != "complete" or parent.get("exit_code") != 0 or
            field_receipt.get("status") != "complete" or molecule.get("status") != "complete" or
            len({receipt["case_id"], parent["case_id"], prepared["case_id"]}) != 1 or
            molecule["launched_count"] != declared_count):
        raise ValueError("parent, field or molecule receipt disagrees with the workflow")
    helium = paths["helium"].parent
    adapter = prepared["science_inputs"]["geometry"]["adapter"]
    if adapter == "cad":
        from dsmc_cad_workflow import validate_prepared_cad
        validate_prepared_cad(Path(receipt["prepared"]["path"]).parent)
    elif adapter not in ("b5-circle", "b5-slit"):
        raise ValueError("workflow has an unsupported geometry adapter")
    resumed = parent.get("schema_version") == RESUME_SCHEMA
    imported = parent.get("schema_version") == "dsmc-cad-helium-import-v1"
    if resumed:
        validate_resume_parent(helium, parent)
    elif imported:
        if adapter != "cad":
            raise ValueError("imported CAD parent is bound to a non-CAD case")
        from dsmc_cad_workflow import validate_imported_cad
        validate_imported_cad(helium, parent)
    elif parent.get("schema_version") != "dsmc-helium-attempt-v1":
        raise ValueError("workflow has an unsupported helium parent")
    for item in parent.get("outputs", []):
        _checked_output(helium, item)
    frame = field_receipt["frame"]
    step = frame["timestep"]
    if step != receipt["frame_selection"]["timestep"]:
        raise ValueError("field frame differs from requested workflow step")
    frame_path = helium / Path(frame["path"]).name
    full_frame = check_field(frame_path, step)
    if _sha(frame_path) != frame["sha256"]:
        raise ValueError("selected helium frame hash differs from field receipt")
    splitmap = None
    if "splitmap" in children:
        if adapter != "cad" or paths["splitmap"] != workflow / "splitmap/splitmap-probe.json":
            raise ValueError("workflow has an unexpected split-map diagnostic child")
        splitmap = _splitmap_diagnostic(workflow, receipt["prepared"]["sha256"], helium, step)
    log_name = "source-solver.log" if imported else "run.log"
    log = (helium / log_name).read_text(encoding="utf-8", errors="replace")
    if "ERROR" in log:
        raise ValueError("SPARTA log contains ERROR")
    classifications = re.findall(
        r"(\d+)\s+(\d+)\s+(\d+) = cells outside/inside/overlapping surfs", log)
    corners = re.findall(
        r"(\d+)\s+(\d+) = cells overlapping surfs, overlap cells with unmarked corner pts", log)
    if (not classifications or
            (adapter != "cad" and any(int(a) <= int(b) for a, b, _ in classifications)) or
            any(sum(map(int, counts)) == 0 for counts in classifications) or
            not corners or any(int(unmarked) for _, unmarked in corners)):
        raise ValueError("SPARTA surface classification or corner marking is unhealthy")
    loops = [(float(seconds), int(steps)) for seconds, steps in re.findall(
        r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", log)]
    expected_steps = (parent["advanced_steps"] if resumed else parent["end_step"] if imported
                      else prepared["science_inputs"]["numerics"]["total_steps"])
    if sum(steps for _, steps in loops) != expected_steps:
        raise ValueError("SPARTA log does not account for all declared steps")
    cad_source = None
    if adapter == "cad":
        cad_source = _cad_diagnostics(helium, prepared, log, frame,
                                      parent["end_step"] if resumed else expected_steps,
                                      imported=imported, resumed=resumed)
        if imported:
            source_run = json.loads((helium / "source-run.json").read_text(encoding="utf-8"))
            if (cad_source["final_simulation_particles"] != source_run.get("final_particles") or
                    cad_source["inserted_final_step"] != source_run.get("emitted_final_step") or
                    cad_source["inserted_cumulative"] != source_run.get("emitted_cumulative")):
                raise ValueError("imported CAD source counts differ from original pilot receipt")
        stations, source = {}, []
    else:
        stations = {name: read_dump(helium / f"equil-{name}.out")
                    for name in ("tf", "tr", "af", "ar")}
        if any(not values or max(values.values()) <= 0 for values in stations.values()):
            raise ValueError("a required station stream is missing or nonpositive")
        source = [line.split() for line in (helium / "equil-source.out").read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
        if not source or not any(float(row[1]) > 0 for row in source):
            raise ValueError("source stream is missing or nonpositive")
    transport = paths["molecules"].parent
    for item in molecule.get("artifacts", []):
        name = item["path"]
        if Path(name).name != name:
            raise ValueError(f"unconfined molecule output name: {name}")
        path = transport / name
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            raise ValueError(f"molecule output identity differs: {name}")
    report = json.loads((transport / "report.json").read_text(encoding="utf-8"))
    if report["observables"]["launched_particles"] != declared_count:
        raise ValueError("report launched denominator differs from the case")
    with (transport / "terminal.csv").open("r", encoding="utf-8", newline="") as stream:
        terminal_rows = sum(1 for _ in csv.DictReader(stream))
    if terminal_rows != declared_count:
        raise ValueError("terminal record count differs from the launched denominator")
    return {
        "schema_version": "dsmc-3d-workflow-verification-v1", "status": "passed",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": {"path": str(workflow), "sha256": _sha(workflow / "workflow.json")},
        "case_id": receipt["case_id"], "frame": full_frame,
        "helium_steps": sum(steps for _, steps in loops),
        "helium_start_step": parent["start_step"] if resumed else 0,
        "helium_end_step": parent["end_step"] if resumed or imported else expected_steps,
        "helium_parent_kind": ("imported-completed-CAD-field-pilot" if imported else
                                "native-resumed" if resumed else "native-fresh"),
        "helium_restart_capability": ("particle-snapshot-only" if imported else "native-binary"),
        "helium_cumulative_elapsed_physical_s": (parent["cumulative_elapsed_physical_s"]
                                                  if imported else None),
        "helium_loop_seconds": sum(seconds for seconds, _ in loops),
        "station_atoms_per_s": {key: list(values.values())[-1] for key, values in stations.items()},
        "cad_source_counts": cad_source,
        "cad_splitmap_probe": splitmap,
        "source_rows": len(source), "unmarked_corners_max": max(int(b) for _, b in corners),
        "field_cells": field_receipt["field_cells"],
        "canonical_triangles": field_receipt["canonical_triangles"],
        "launched_molecules": declared_count,
        "forward_events": {p["name"]: p["forward_event_count"]
                           for p in report["observables"]["planes"]},
        "science_assessment": report["science_assessment"],
    }


def verify(workflow: Path, out: Path) -> Path:
    workflow, out = Path(workflow).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"verification output already exists: {out}")
    if out == workflow or out in workflow.parents or workflow in out.parents:
        raise ValueError("verification receipt must be separate from workflow outputs")
    result = _verify(workflow)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = verify(args.workflow, args.out)
    except (ValueError, FileExistsError, OSError, KeyError) as exc:
        parser.exit(2, f"verify_3d_workflow: {exc}\n")
    print(f"Workflow verification: {result}")


if __name__ == "__main__":
    main()
