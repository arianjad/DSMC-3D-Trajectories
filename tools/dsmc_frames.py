"""Read-only selection of an independently complete SPARTA 3D gas frame.

This stage verifies compressed-stream completion and structural identity. The
field converter performs the later numerical/lattice checks on the chosen frame.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from pathlib import Path

from dsmc_resume_parent import SCHEMA as RESUME_SCHEMA, validate_resume_parent


FRAME_NAME = re.compile(r"^fields\.(\d+)\.gz$")
REQUIRED_COLUMNS = (
    "idstr", "split", "xlo", "ylo", "zlo", "xhi", "yhi", "zhi", "vol",
    *(f"f_afields[{i}]" for i in range(1, 10)),
)


def _manifest(run_dir: Path) -> dict | None:
    path = run_dir / "manifest.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") not in ("dsmc-helium-attempt-v1", RESUME_SCHEMA,
                                          "dsmc-cad-helium-import-v1"):
        raise ValueError(f"{path}: unsupported helium attempt manifest")
    if data["schema_version"] == RESUME_SCHEMA:
        validate_resume_parent(run_dir, data)
    elif data["schema_version"] == "dsmc-cad-helium-import-v1":
        from dsmc_cad_workflow import validate_imported_cad
        validate_imported_cad(run_dir, data)
    if data.get("status") not in ("running", "complete", "failed", "interrupted"):
        raise ValueError(f"{path}: unknown helium attempt status")
    return data


def validate_frame(path: Path, *, expected_step: int | None = None,
                   manifest: dict | None = None) -> dict:
    """Verify one complete gzip stream, header, row count and compressed bytes.

    A frame remains usable after a failed parent if this check passes. The
    caller must copy it to an immutable child and recheck the identity there.
    """
    path = Path(path)
    match = FRAME_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unsupported 3D frame name: {path.name}")
    name_step = int(match.group(1))
    if expected_step is not None and expected_step != name_step:
        raise ValueError(f"requested step {expected_step} differs from {path.name}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    compressed_sha = digest.hexdigest()
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            header = [next(stream).strip() for _ in range(9)]
            if (header[0] != "ITEM: TIMESTEP" or
                    header[2] != "ITEM: NUMBER OF CELLS" or
                    header[4] != "ITEM: BOX BOUNDS oo oo oo" or
                    not header[8].startswith("ITEM: CELLS ")):
                raise ValueError(f"{path}: unsupported SPARTA grid header")
            step = int(header[1])
            declared = int(header[3])
            if step != name_step or declared <= 0:
                raise ValueError(f"{path}: timestep/count differs from filename or is empty")
            bounds = []
            for line in header[5:8]:
                pair = [float(value) for value in line.split()]
                if len(pair) != 2 or not all(math.isfinite(value) for value in pair) or not pair[0] < pair[1]:
                    raise ValueError(f"{path}: invalid box bounds")
                bounds.append(pair)
            columns = header[8].split()[2:]
            if len(columns) != len(REQUIRED_COLUMNS) or set(columns) != set(REQUIRED_COLUMNS):
                raise ValueError(f"{path}: unsupported 3D gas columns")
            rows = sum(bool(line.strip()) for line in stream)
            if rows != declared:
                raise ValueError(f"{path}: declared {declared} cells but read {rows}")
    except (EOFError, OSError, UnicodeError, StopIteration) as exc:
        raise ValueError(f"{path}: incomplete or corrupt gzip grid: {exc}") from exc
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"{path}: frame changed during validation")
    if manifest is not None:
        outputs = [item for item in manifest.get("outputs", [])
                   if item.get("path") == path.name]
        if len(outputs) > 1:
            raise ValueError(f"{path}: duplicate frame outputs in manifest")
        if manifest["status"] == "complete" and len(outputs) != 1:
            raise ValueError(f"{path}: completed parent omitted this frame")
        if outputs and (outputs[0].get("bytes") != after.st_size or
                        outputs[0].get("sha256") != compressed_sha):
            raise ValueError(f"{path}: frame differs from parent manifest")
    return {"path": str(path.resolve()), "timestep": step, "rows": rows,
            "box_bounds_m": bounds, "bytes": after.st_size,
            "sha256": compressed_sha, "validation": "gzip-crc-header-row-count-v1"}


def select_frame(run_dir: Path, selection: dict) -> dict:
    """Select an exact timestep or the newest independently complete frame."""
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"helium run directory is missing: {run_dir}")
    if set(selection) == {"timestep"}:
        step = selection["timestep"]
        if type(step) is not int or step < 0:
            raise ValueError("frame timestep must be a nonnegative integer")
        candidates = [run_dir / f"fields.{step}.gz"]
        exact = True
    elif selection == {"last_complete": True}:
        candidates = sorted((path for path in run_dir.iterdir()
                             if FRAME_NAME.fullmatch(path.name)),
                            key=lambda path: int(FRAME_NAME.fullmatch(path.name).group(1)),
                            reverse=True)
        exact = False
    else:
        raise ValueError("select one frame by timestep or last_complete=true")
    manifest = _manifest(run_dir)
    rejected = []
    for path in candidates:
        try:
            frame = validate_frame(path, manifest=manifest)
        except (ValueError, FileNotFoundError) as exc:
            if exact:
                raise ValueError(f"exact frame selection failed: {exc}") from exc
            rejected.append({"path": path.name, "reason": str(exc)})
            continue
        return {**frame, "selection": selection,
                "parent_manifest_status": manifest["status"] if manifest else "absent",
                "parent_manifest_sha256": (
                    hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest()
                    if manifest else None),
                "rejected_newer_frames": rejected}
    raise ValueError(f"no complete 3D gas frame in {run_dir}: {rejected}")
