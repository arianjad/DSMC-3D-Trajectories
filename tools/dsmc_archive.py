#!/usr/bin/env python3
"""Copy a verified 3D workflow into a relocation-safe full archive."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _files(root: Path, prefix: str) -> list[tuple[Path, str]]:
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"archive source contains a symlink: {path}")
        if path.is_file():
            result.append((path, f"{prefix}/{path.relative_to(root).as_posix()}"))
    return result


def _checked_sources(workflow: Path) -> tuple[Path, Path, dict, dict]:
    receipt = json.loads((workflow / "workflow.json").read_text(encoding="utf-8"))
    prepared_file = Path(receipt["prepared"]["path"])
    if not prepared_file.is_file() or _sha(prepared_file) != receipt["prepared"]["sha256"]:
        raise ValueError("workflow's prepared input is missing or changed")
    prepared = json.loads(prepared_file.read_text(encoding="utf-8"))
    if prepared["science_inputs"]["geometry"]["adapter"] == "cad":
        from dsmc_cad_workflow import validate_prepared_cad
        validate_prepared_cad(prepared_file.parent)
    return prepared_file.parent, checked_case(prepared), receipt, prepared


def checked_case(prepared: dict) -> Path:
    """The prepared run's source case: any file inside the repository, unchanged since prepare."""
    case_file = (ROOT / prepared["source_case"]["path"]).resolve()
    if ROOT.resolve() not in case_file.parents:
        raise ValueError("source case must be inside the repository")
    if not case_file.is_file() or _sha(case_file) != prepared["source_case"]["sha256"]:
        raise ValueError("source case is missing or changed")
    return case_file


def verify_archive(archive: Path) -> dict:
    archive = Path(archive).resolve()
    record = json.loads((archive / "archive-manifest.json").read_text(encoding="utf-8"))
    if record.get("schema_version") != "dsmc-3d-full-archive-v1" or record.get("status") != "complete":
        raise ValueError("full archive receipt is incomplete")
    if record.get("source_verification", {}).get("status") != "passed":
        raise ValueError("source workflow did not pass verification")
    listed = set()
    for item in record["files"]:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts or item["path"] in listed:
            raise ValueError("archive inventory contains an invalid path")
        listed.add(item["path"])
        path = archive / rel
        if (path.is_symlink() or not path.is_file() or
                path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]):
            raise ValueError(f"archived file is missing or changed: {item['path']}")
    if any(p.is_symlink() for p in archive.rglob("*")):
        raise ValueError("archive contains a symlink")
    actual = {p.relative_to(archive).as_posix() for p in archive.rglob("*") if p.is_file()}
    if actual != listed | {"archive-manifest.json"}:
        raise ValueError("archive contains unlisted or missing files")
    workflow = json.loads((archive / "workflow/workflow.json").read_text(encoding="utf-8"))
    prepared_hash = _sha(archive / "prepared/prepared.json")
    if (_sha(archive / "workflow/workflow.json") != record["source_workflow_sha256"] or
            prepared_hash != workflow["prepared"]["sha256"] or
            prepared_hash != record["source_prepared_sha256"]):
        raise ValueError("archived workflow or prepared identity differs")
    children = {"helium": "helium/manifest.json", "field": "field/field-preparation.json",
                "molecules": "molecules/molecule-run.json"}
    for name, path in children.items():
        if _sha(archive / "workflow" / path) != workflow["children"][name]["sha256"]:
            raise ValueError(f"archived {name} child receipt differs")
    if "splitmap" in workflow["children"]:
        probe_path = archive / "workflow/splitmap/splitmap-probe.json"
        if (_sha(probe_path) != workflow["children"]["splitmap"]["sha256"] or
                "workflow/splitmap/splitmap-probe.json" not in listed):
            raise ValueError("archived split-map child receipt differs")
        from verify_3d_workflow import _splitmap_diagnostic
        probe = _splitmap_diagnostic(archive / "workflow", prepared_hash,
                                     archive / "workflow/helium",
                                     workflow["frame_selection"]["timestep"])
        if probe != record["source_verification"].get("cad_splitmap_probe"):
            raise ValueError("archived CAD split-map diagnostic differs from source verification")
    elif record["source_verification"].get("cad_splitmap_probe") is not None:
        raise ValueError("archive claims a split-map diagnostic without a child")
    from dsmc_resume_parent import SCHEMA as RESUME_SCHEMA, validate_resume_parent
    helium = json.loads((archive / "workflow/helium/manifest.json").read_text(encoding="utf-8"))
    imported = helium.get("schema_version") == "dsmc-cad-helium-import-v1"
    if helium.get("schema_version") == RESUME_SCHEMA:
        validate_resume_parent(archive / "workflow/helium", helium)
    checkpoints = record["final_restart"]
    if imported:
        from dsmc_cad_workflow import validate_imported_cad
        validate_imported_cad(archive / "workflow/helium", helium)
        snapshot = f"workflow/helium/particles.{helium['end_step']}"
        capability = record.get("restart_capability")
        if (checkpoints or
                any(name.startswith("workflow/helium/final.restart.") for name in listed) or
                capability != {"kind": "particle-snapshot-only", "native_binary": False,
                               "snapshot_path": snapshot,
                               "snapshot_sha256": _sha(archive / snapshot)} or
                snapshot not in listed or
                helium["prepared"]["sha256"] != prepared_hash):
            raise ValueError("imported CAD archive has an invalid particle-only capability")
    else:
        if (record.get("restart_capability") not in (None,
                {"kind": "native-binary", "native_binary": True}) or
                not checkpoints or not any(x["path"].endswith(".base") for x in checkpoints)):
            raise ValueError("archive lacks a final native restart base file")
        if any(item["path"] not in listed for item in checkpoints):
            raise ValueError("archive checkpoint is absent from inventory")
        if helium.get("schema_version") == RESUME_SCHEMA:
            expected = {"workflow/helium/" + item["path"]: item["sha256"] for item in helium["final_restart"]}
            if {item["path"]: item["sha256"] for item in checkpoints} != expected:
                raise ValueError("archive resumed checkpoint inventory differs from helium parent")
    return {"status": "passed", "files": len(listed),
            "bytes": sum(item["bytes"] for item in record["files"]),
            "source_verification": record["source_verification"]}


def archive_workflow(workflow: Path, out: Path) -> Path:
    from verify_3d_workflow import _verify

    workflow, out = Path(workflow).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"archive output already exists: {out}")
    if out == workflow or out in workflow.parents or workflow in out.parents:
        raise ValueError("archive output must be separate from workflow")
    try:
        checked = _verify(workflow)
    except (AssertionError, KeyError) as exc:
        raise ValueError(f"source workflow verification failed: {exc}") from exc
    prepared_dir, case_file, receipt, _ = _checked_sources(workflow)
    if out == prepared_dir or out in prepared_dir.parents or prepared_dir in out.parents:
        raise ValueError("archive output must be separate from preparation")
    sources = [*_files(workflow, "workflow"), *_files(prepared_dir, "prepared"),
               (case_file, "case/case.json")]
    helium = json.loads((workflow / "helium/manifest.json").read_text(encoding="utf-8"))
    from dsmc_resume_parent import SCHEMA as RESUME_SCHEMA
    imported = helium.get("schema_version") == "dsmc-cad-helium-import-v1"
    if imported:
        from dsmc_cad_workflow import validate_imported_cad
        validate_imported_cad(workflow / "helium", helium)
        restart_paths = set()
        snapshot = workflow / "helium" / f"particles.{helium['end_step']}"
        restart_capability = {"kind": "particle-snapshot-only", "native_binary": False,
                              "snapshot_path": f"workflow/helium/{snapshot.name}",
                              "snapshot_sha256": _sha(snapshot)}
    else:
        restart_paths = ({"workflow/helium/" + item["path"] for item in helium["final_restart"]}
                         if helium.get("schema_version") == RESUME_SCHEMA else
                         {relative for _, relative in sources if relative.startswith("workflow/helium/final.restart.")})
        restart_capability = {"kind": "native-binary", "native_binary": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    needed = sum(source.stat().st_size for source, _ in sources)
    if shutil.disk_usage(out.parent).free < needed + 512 * 1024 * 1024:
        raise ValueError("insufficient free space for full archive plus 512 MiB reserve")
    out.mkdir()
    verification = {key: value for key, value in checked.items() if key != "workflow"}
    record = {"schema_version": "dsmc-3d-full-archive-v1", "status": "copying",
              "created_utc": _now(), "case_id": checked["case_id"], "files": [],
              "source_workflow_sha256": _sha(workflow / "workflow.json"),
              "source_prepared_sha256": _sha(prepared_dir / "prepared.json"),
              "source_profile_sha256": receipt["profile"]["sha256"],
              "source_verification": verification, "final_restart": [],
              "restart_capability": restart_capability}
    _write(out / "archive-manifest.json", record)
    try:
        for source, relative in sources:
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            digest = _sha(target)
            if digest != _sha(source):
                raise ValueError(f"archive copy changed: {relative}")
            item = {"path": relative, "bytes": target.stat().st_size, "sha256": digest}
            record["files"].append(item)
            if relative in restart_paths:
                record["final_restart"].append(item)
        record["status"] = "complete"
        record["finished_utc"] = _now()
        _write(out / "archive-manifest.json", record)
        verify_archive(out)
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["failure"] = str(exc)
        record["finished_utc"] = _now()
        _write(out / "archive-manifest.json", record)
        raise
    return out / "archive-manifest.json"


def verify_retrieval(archive: Path, receipt: Path) -> dict:
    """Record receiving-host verification separately from the immutable archive."""
    archive, receipt = Path(archive).resolve(), Path(receipt).resolve()
    if receipt == archive or archive in receipt.parents:
        raise ValueError("retrieval receipt must be outside the archive")
    if receipt.exists() or receipt.is_symlink():
        raise FileExistsError(receipt)
    checked = verify_archive(archive)
    record = {"schema_version": "dsmc-archive-retrieval-v1", "status": "complete",
              "verified_utc": _now(), "archive": str(archive),
              "manifest_sha256": _sha(archive / "archive-manifest.json"),
              "verification": checked}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return record
