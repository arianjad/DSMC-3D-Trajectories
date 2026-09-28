#!/usr/bin/env python3
"""Create a local, path-relative analysis bundle from one verified B5 workflow."""
from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from pathlib import Path

from dsmc_archive import checked_case
from verify_3d_workflow import _verify


ROOT = Path(__file__).resolve().parents[1]
FIELD_FILES = ("DS3FF.DAT", "cell.surfs", "source-row-metadata.csv.gz",
               "canonical-tracer.toml", "triangle-roles.json")
MOLECULE_FILES = ("events.csv", "terminal.csv")
STATIONS = ("tf", "tr", "af", "ar", "source", "regions", "inventory", "boundary")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def export(workflow: Path, out: Path) -> Path:
    workflow, out = Path(workflow).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"export output already exists: {out}")
    if out == workflow or out in workflow.parents or workflow in out.parents:
        raise ValueError("export must be separate from the source workflow")
    try:
        checked = _verify(workflow)
    except (AssertionError, KeyError) as exc:
        raise ValueError("source workflow verification failed") from exc
    receipt = json.loads((workflow / "workflow.json").read_text(encoding="utf-8"))
    parent = json.loads((workflow / "helium/manifest.json").read_text(encoding="utf-8"))
    prepared_path = Path(receipt["prepared"]["path"])
    if not prepared_path.is_file() or _sha(prepared_path) != receipt["prepared"]["sha256"]:
        raise ValueError("prepared case identity differs from workflow receipt")
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    source_case = checked_case(prepared)
    report = json.loads((workflow / "molecules/report.json").read_text(encoding="utf-8"))
    field = json.loads((workflow / "field/field-preparation.json").read_text(encoding="utf-8"))
    transport = tomllib.loads((workflow / "molecules/transport-result.toml").read_text(encoding="utf-8"))
    frame_name = Path(field["frame"]["path"]).name
    if frame_name != f"fields.{checked['helium_steps']}.gz":
        raise ValueError("selected raw frame is not the verified final gzip frame")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    files: list[dict] = []

    def add(source: Path, relative: str) -> None:
        dest = Path(relative)
        if dest.is_absolute() or ".." in dest.parts or not source.is_file():
            raise ValueError(f"invalid export source or path: {relative}")
        target = out / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = _sha(target)
        if digest != _sha(source):
            raise ValueError(f"export copy changed: {relative}")
        files.append({"path": dest.as_posix(), "bytes": target.stat().st_size,
                      "sha256": digest})

    try:
        add(source_case, "case/case.json")
        add(prepared_path.parent / "physical-walls.surf", "geometry/physical-walls.surf")
        add(prepared_path.parent / "triangle-roles.json", "geometry/triangle-roles.json")
        add(workflow / "helium/in.helium", "inputs/in.helium")
        add(workflow / "helium/he.species", "inputs/he.species")
        add(workflow / "helium/he.vss", "inputs/he.vss")
        add(workflow / "helium" / frame_name, f"helium/{frame_name}")
        add(workflow / "helium/run.log", "helium/run.log")
        for name in STATIONS:
            add(workflow / "helium" / f"equil-{name}.out",
                f"helium/equil-{name}.out")
        for name in FIELD_FILES:
            add(workflow / "field" / name, f"field/{name}")
        for name in MOLECULE_FILES:
            add(workflow / "molecules" / name, f"molecules/{name}")
        add(ROOT / "LICENSE", "LICENSE")
        add(ROOT / "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")
        _write(out / "case/science-inputs.json", prepared["science_inputs"])
        _write(out / "molecules/observables.json", {
            "case_id": report["case_id"], "science_assessment": report["science_assessment"],
            "observables": report["observables"]})
        for name in ("case/science-inputs.json", "molecules/observables.json"):
            target = out / name
            files.append({"path": name, "bytes": target.stat().st_size,
                          "sha256": _sha(target)})
        readme = out / "README.md"
        readme.write_text(
            "# Selected-frame B5 analysis bundle\n\n"
            "`export-manifest.json` lists every included file by relative path, size and SHA-256.\n"
            "`case/` holds the declared case and resolved science inputs; `inputs/` and\n"
            "`geometry/` hold the solver deck and checked physical surface; `helium/`\n"
            "holds the selected raw frame and diagnostic streams; `field/` holds the\n"
            "converted tracer field; `molecules/` holds events, terminal rows and\n"
            "observable summaries. The manifest lists omitted restart and other raw\n"
            "files, plus external software and machine settings. This is an analysis\n"
            "bundle, not a runnable environment or a converged science claim.\n",
            encoding="utf-8", newline="\n")
        files.append({"path": "README.md", "bytes": readme.stat().st_size,
                      "sha256": _sha(readme)})
        included_helium = {f"helium/{frame_name}", "helium/run.log",
                           *(f"helium/equil-{name}.out" for name in STATIONS)}
        omitted = [{"path": f"helium/{item['path']}", "bytes": item["bytes"],
                    "sha256": item["sha256"]}
                   for item in parent["outputs"]
                   if f"helium/{item['path']}" not in included_helium]
        bundle = {
            "schema_version": "dsmc-3d-analysis-export-v1", "status": "complete",
            "scope": "selected-frame analysis bundle; not a restart or executable environment",
            "case_id": checked["case_id"], "files": sorted(files, key=lambda x: x["path"]),
            "source_workflow_receipt_sha256": _sha(workflow / "workflow.json"),
            "source_prepared_sha256": _sha(prepared_path),
            "source_profile_sha256": receipt["profile"]["sha256"],
            "solver_sha256": parent["solver"]["sha256"],
            "repository_commit": (parent["repository"]["commit"] or
                                  receipt.get("repository", {}).get("commit")),
            "repository_dirty": (parent["repository"]["dirty"] if
                                 parent["repository"]["commit"] else
                                 receipt.get("repository", {}).get("dirty")),
            "repository_identity_status": ("recorded" if
                                           (parent["repository"]["commit"] or
                                            receipt.get("repository", {}).get("commit"))
                                           else "unavailable-in-parent-run"),
            "preparation_tools": prepared["preparation_tools"],
            "field_implementation_sha256": field["implementation_sha256"],
            "molecule_implementation_sha256": transport["implementation_sha256"],
            "selected_frame": {"path": f"helium/{frame_name}",
                               "sha256": field["frame"]["sha256"]},
            "gas_policy": receipt["gas_policy"],
            "verification": {"helium_steps": checked["helium_steps"],
                             "launched_molecules": checked["launched_molecules"],
                             "science_assessment": checked["science_assessment"]},
            "omitted_helium_outputs": omitted,
            "omitted_external": ["solver executable/build", "Julia/Python environments",
                                 "machine profile", "original absolute-path receipts"],
        }
        _write(out / "export-manifest.json", bundle)
        for item in bundle["files"]:
            path = out / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
                raise ValueError(f"export self-check failed: {item['path']}")
    except BaseException:
        _write(out / "export-failure.json", {
            "status": "failed", "source_workflow_receipt_sha256": _sha(workflow / "workflow.json")})
        raise
    return out / "export-manifest.json"
