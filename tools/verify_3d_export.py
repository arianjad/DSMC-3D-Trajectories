#!/usr/bin/env python3
"""Verify a relocated B5 selected-frame analysis bundle without its source run."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(name: object) -> PurePosixPath:
    if (not isinstance(name, str) or not name or "\\" in name or
            any(ord(char) < 32 for char in name)):
        raise ValueError(f"invalid bundle path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts:
        raise ValueError(f"bundle path escapes root: {name!r}")
    return path


def verify_export(bundle: Path) -> dict:
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "export-manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != "dsmc-3d-analysis-export-v1" or
            manifest.get("status") != "complete"):
        raise ValueError("analysis bundle manifest is not complete")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("analysis bundle has no file inventory")
    listed = {}
    for item in files:
        name = item["path"]
        relative = _relative(name)
        if name in listed:
            raise ValueError(f"duplicate bundle path: {name}")
        cursor = bundle
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"bundle path traverses a symlink: {name}")
        path = bundle.joinpath(*relative.parts)
        if (path.is_symlink() or not path.is_file() or
                path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]):
            raise ValueError(f"bundle file missing or changed: {name}")
        listed[name] = item
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("analysis bundle contains a symlink")
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*")
              if path.is_file()}
    if actual != set(listed) | {"export-manifest.json"}:
        raise ValueError("analysis bundle has an unlisted or missing file")
    required = {"LICENSE", "THIRD_PARTY_NOTICES.md", "README.md",
                "case/case.json", "case/science-inputs.json",
                "geometry/physical-walls.surf", "geometry/triangle-roles.json",
                "field/DS3FF.DAT", "field/canonical-tracer.toml",
                "molecules/terminal.csv", "molecules/events.csv",
                "molecules/observables.json"}
    if not required <= set(listed):
        raise ValueError("analysis bundle lacks a required study artifact")
    frame = manifest["selected_frame"]
    frame_name = frame["path"]
    if (frame_name not in listed or frame["sha256"] != listed[frame_name]["sha256"] or
            not frame_name.startswith("helium/fields.") or not frame_name.endswith(".gz")):
        raise ValueError("selected raw helium frame identity differs")
    with gzip.open(bundle / frame_name, "rt", encoding="utf-8") as stream:
        if stream.readline().strip() != "ITEM: TIMESTEP":
            raise ValueError("selected raw frame lacks a timestep header")
        frame_step = int(stream.readline().strip())
    if frame_step != manifest["verification"]["helium_steps"]:
        raise ValueError("selected raw frame step differs from manifest")
    case = json.loads((bundle / "case/case.json").read_text(encoding="utf-8"))
    summary = json.loads((bundle / "molecules/observables.json").read_text(encoding="utf-8"))
    if (case["case_id"] != manifest["case_id"] or
            summary["case_id"] != manifest["case_id"] or
            summary["science_assessment"] != manifest["verification"]["science_assessment"]):
        raise ValueError("case or science assessment differs from manifest")
    expected_launched = manifest["verification"]["launched_molecules"]
    if summary["observables"]["launched_particles"] != expected_launched:
        raise ValueError("molecule denominator differs from manifest")
    with (bundle / "molecules/terminal.csv").open(encoding="utf-8", newline="") as stream:
        terminal = list(csv.DictReader(stream))
    ids = [int(row["particle_id"]) for row in terminal]
    if len(ids) != expected_launched or len(set(ids)) != expected_launched:
        raise ValueError("terminal records do not cover the launched molecules once each")
    id_set = set(ids)
    with (bundle / "molecules/events.csv").open(encoding="utf-8", newline="") as stream:
        events = list(csv.DictReader(stream))
    if len(events) != summary["observables"]["event_count"] or any(
            row["schema_version"] != "dsmc-observable-events-v1" or
            int(row["particle_id"]) not in id_set for row in events):
        raise ValueError("event records differ from the observable summary")
    if any(item["path"] in listed for item in manifest["omitted_helium_outputs"]):
        raise ValueError("omitted helium output appears in bundle inventory")
    return {"status": "passed", "case_id": manifest["case_id"],
            "files": len(listed), "bytes": sum(item["bytes"] for item in files),
            "frame_step": frame_step, "launched_molecules": len(ids),
            "events": len(events), "science_assessment": summary["science_assessment"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_export(args.bundle), indent=2, sort_keys=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"verify_3d_export: {exc}\n")


if __name__ == "__main__":
    main()
