#!/usr/bin/env python3
"""Verify a completed same-host B5 restart attempt and its new sampled field."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from dsmc_archive import verify_archive
from dsmc_checkpoint import verify_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_resume(resume_dir: Path, parent_archive: Path) -> dict:
    resume_dir, parent_archive = Path(resume_dir).resolve(), Path(parent_archive).resolve()
    parent = verify_archive(parent_archive)
    prepared = json.loads((parent_archive / "prepared/prepared.json").read_text(encoding="utf-8"))
    numerics = prepared["science_inputs"]["numerics"]
    field_period = numerics["field_every_steps"]
    history_period = numerics["field_average"]["period_steps"]
    record = json.loads((resume_dir / "resume.json").read_text(encoding="utf-8"))
    if (record.get("schema_version") != "dsmc-b5-resume-preparation-v1" or
            record.get("status") != "prepared" or
            record.get("parent_archive_sha256") != _sha(parent_archive / "archive-manifest.json") or
            record.get("case_id") != parent["source_verification"]["case_id"]):
        raise ValueError("resume input receipt or parent archive identity differs")
    for item in record["inputs"]:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("resume input escapes attempt directory")
        path = resume_dir / rel
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"] or
                _sha(path) != item["sha256"]):
            raise ValueError(f"resume input changed: {rel}")
    start, target = record["checkpoint"]["expected_timestep"], record["target_step"]
    if (start != parent["source_verification"]["helium_steps"] or
            record["checkpoint"]["sha256"] != _sha(parent_archive / "workflow/helium/final.restart.base") or
            target <= start + 1):
        raise ValueError("resume checkpoint or target disagrees with parent")
    run = resume_dir / "run"
    log_path = run / "log.sparta"
    log = log_path.read_text(encoding="utf-8", errors="replace")
    if "ERROR" in log:
        raise ValueError("resumed SPARTA log contains ERROR")
    source_ranks = len([item for item in record["inputs"]
                        if re.fullmatch(r"run/final\.restart\.\d+", item["path"])])
    target_ranks = record.get("target_ranks", source_ranks)
    if (type(target_ranks) is not int or target_ranks <= 0 or
            not re.search(rf"^Running on {target_ranks} MPI task\(s\)$", log, flags=re.MULTILINE)):
        raise ValueError("resumed solver MPI rank count differs from prepared attempt")
    loops = [(float(seconds), int(steps)) for seconds, steps in re.findall(
        r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", log)]
    if sum(steps for _, steps in loops) != target - start or not re.search(
            rf"^\s*{target}\s+", log, flags=re.MULTILINE):
        raise ValueError("resumed solver log does not reach the declared timestep")
    corners = re.findall(
        r"(\d+)\s+(\d+) = cells overlapping surfs, overlap cells with unmarked corner pts", log)
    if (not corners or any(int(unmarked) for _, unmarked in corners) or
            not re.search(r"Particles stuck\s*=\s*0", log) or
            not re.search(r"Axisymm bad moves\s*=\s*0", log)):
        raise ValueError("resumed geometry or particle health check failed")
    expected_restart = {"continued.restart.base"} | {
        f"continued.restart.{i}" for i in range(target_ranks)}
    found_restart = {path.name for path in run.glob("continued.restart.*") if path.is_file()}
    if found_restart != expected_restart:
        raise ValueError("resumed final restart base/shards are incomplete")
    final_restart = [{"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
                     for path in sorted(run.glob("continued.restart.*"))]
    expected_fields = {f"fields.{step}.gz" for step in range(start + 1, target + 1)
                       if step % field_period == 0}
    found_fields = {path.name for path in run.glob("fields.*") if path.is_file()}
    if found_fields != expected_fields:
        raise ValueError(f"resumed field set differs: expected {sorted(expected_fields)}, "
                         f"found {sorted(found_fields)}")
    sys.path.insert(0, str(ROOT / "cases/b5-slit-pair"))
    from check_sample_probes import check_field, read_dump  # noqa: E402
    try:
        fields = [check_field(run / name, int(name.split(".")[1]))
                  for name in sorted(expected_fields)]
    except (AssertionError, IndexError, ValueError) as exc:
        raise ValueError(f"resumed field failed full-row check: {exc}") from exc
    expected_rows = parent["source_verification"]["frame"]["rows"]
    if any(field["rows"] != expected_rows for field in fields):
        raise ValueError("resumed field row count differs from parent geometry")
    station = {}
    if target >= ((start // history_period) + 1) * history_period:
        for name in ("tf", "tr", "af", "ar"):
            values = read_dump(run / f"equil-{name}.out")
            if not values or max(values.values()) <= 0:
                raise ValueError(f"resumed {name} station is empty or nonpositive")
            station[name] = list(values.values())[-1]
        source = [line.split() for line in (run / "equil-source.out").read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
        if not source or not any(float(row[1]) > 0 for row in source):
            raise ValueError("resumed source history is empty or nonpositive")
    return {"schema_version": "dsmc-b5-resume-verification-v1", "status": "passed",
            "case_id": record["case_id"], "checkpoint_step": start,
            "target_step": target, "resumed_steps": sum(steps for _, steps in loops),
            "source_ranks": source_ranks, "target_ranks": target_ranks,
            "loop_seconds": sum(seconds for seconds, _ in loops),
            "unmarked_corners_max": max(int(unmarked) for _, unmarked in corners),
            "fields": fields, "station_atoms_per_s": station,
            "final_restart": final_restart, "science_assessment": "unassessed"}


def verify_resume_checkpoint(resume_dir: Path, checkpoint_bundle: Path) -> dict:
    """Verify the resumed interval against the independently verified bundle."""
    resume_dir, checkpoint_bundle = Path(resume_dir).resolve(), Path(checkpoint_bundle).resolve()
    checkpoint = verify_checkpoint(checkpoint_bundle)
    manifest = json.loads((checkpoint_bundle / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    prepared = manifest["prepared"]["record"]
    if prepared.get("science_inputs", {}).get("geometry", {}).get("adapter") == "cad":
        return _verify_resume_checkpoint_cad(resume_dir, checkpoint_bundle, checkpoint, prepared)
    record = json.loads((resume_dir / "resume.json").read_text(encoding="utf-8"))
    if (record.get("schema_version") != "dsmc-checkpoint-resume-preparation-v1" or
            record.get("status") != "prepared" or
            record.get("checkpoint_bundle_sha256") != _sha(checkpoint_bundle / "checkpoint-manifest.json") or
            record.get("case_id") != prepared.get("case_id")):
        raise ValueError("checkpoint-resume preparation or bundle identity differs")
    for item in record.get("inputs", []):
        rel = Path(item.get("path", "")); path = resume_dir / rel
        if rel.is_absolute() or ".." in rel.parts or path.is_symlink() or not path.is_file() or \
                path.stat().st_size != item.get("bytes") or _sha(path) != item.get("sha256"):
            raise ValueError(f"checkpoint-resume input changed: {rel}")
    start, target = checkpoint["checkpoint_step"], record.get("target_step")
    if record.get("checkpoint", {}).get("step") != start or type(target) is not int or target <= start + 1:
        raise ValueError("checkpoint-resume target does not advance from verified checkpoint step")
    run = resume_dir / "run"; log = (run / "log.sparta").read_text(encoding="utf-8", errors="replace")
    if "ERROR" in log:
        raise ValueError("resumed SPARTA log contains ERROR")
    source_ranks, target_ranks = record.get("source_ranks"), record.get("target_ranks")
    if (source_ranks != checkpoint["source_ranks"] or type(target_ranks) is not int or target_ranks < 1 or
            not re.search(rf"^Running on {target_ranks} MPI task\(s\)$", log, flags=re.MULTILINE)):
        raise ValueError("resumed solver MPI rank count differs from prepared checkpoint attempt")
    loops = [(float(seconds), int(steps)) for seconds, steps in re.findall(
        r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", log)]
    if sum(steps for _, steps in loops) != target - start or not re.search(rf"^\s*{target}\s+", log, flags=re.MULTILINE):
        raise ValueError("resumed solver log does not reach its declared checkpoint-relative duration")
    corners = re.findall(r"(\d+)\s+(\d+) = cells overlapping surfs, overlap cells with unmarked corner pts", log)
    if (not corners or any(int(unmarked) for _, unmarked in corners) or
            not re.search(r"Particles stuck\s*=\s*0", log) or not re.search(r"Axisymm bad moves\s*=\s*0", log)):
        raise ValueError("resumed geometry or particle health check failed")
    expected_restart = {"continued.restart.base"} | {f"continued.restart.{rank}" for rank in range(target_ranks)}
    found_restart = {path.name for path in run.glob("continued.restart.*") if path.is_file()}
    if found_restart != expected_restart:
        raise ValueError("resumed final restart base/shards are incomplete")
    final_restart = []
    for name in sorted(expected_restart):
        path = run / name
        if path.stat().st_size <= 0:
            raise ValueError(f"resumed final restart is empty: {name}")
        final_restart.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    numerics = prepared["science_inputs"]["numerics"]
    expected_fields = {f"fields.{step}.gz" for step in range(start + 1, target + 1)
                       if step % numerics["field_every_steps"] == 0}
    found_fields = {path.name for path in run.glob("fields.*") if path.is_file()}
    if found_fields != expected_fields:
        raise ValueError("resumed field set does not contain only complete post-checkpoint windows")
    sys.path.insert(0, str(ROOT / "cases/b5-slit-pair"))
    from check_sample_probes import check_field  # noqa: E402
    fields = []
    expected_rows = checkpoint["diagnostics"]["dump_rows"]
    for name in sorted(expected_fields):
        path = run / name
        try:
            field = check_field(path, int(name.split(".")[1]))
        except (AssertionError, EOFError, OSError, UnicodeError, IndexError, ValueError) as exc:
            raise ValueError(f"resumed field failed full-row check: {name}: {exc}") from exc
        if field["rows"] != expected_rows:
            raise ValueError(f"resumed field rows differ from native grid: {name}")
        field["sha256"] = _sha(path)
        fields.append(field)
    return {"schema_version": "dsmc-checkpoint-resume-verification-v1", "status": "passed",
            "case_id": record["case_id"], "checkpoint_bundle_sha256": record["checkpoint_bundle_sha256"],
            "checkpoint_step": start, "target_step": target, "resumed_steps": sum(steps for _, steps in loops),
            "source_ranks": source_ranks, "target_ranks": target_ranks,
            "loop_seconds": sum(seconds for seconds, _ in loops),
            "unmarked_corners_max": max(int(unmarked) for _, unmarked in corners),
            "fields": fields, "final_restart": final_restart,
            "diagnostic_averaging": "fresh-from-checkpoint",
            "science_assessment": "unassessed"}


def _verify_resume_checkpoint_cad(resume_dir: Path, checkpoint_bundle: Path,
                                  checkpoint: dict, prepared: dict) -> dict:
    """Verify the bounded CAD 2000-to-4000 continuation from copied restart bytes."""
    from dsmc_cad_workflow import validate_prepared_cad
    from dsmc_frames import validate_frame

    validate_prepared_cad(checkpoint_bundle / "prepared")
    record = json.loads((resume_dir / "resume.json").read_text(encoding="utf-8"))
    if (record.get("schema_version") != "dsmc-checkpoint-resume-preparation-v1" or
            record.get("status") != "prepared" or record.get("adapter") != "cad" or
            record.get("checkpoint_bundle_sha256") != _sha(checkpoint_bundle / "checkpoint-manifest.json") or
            record.get("case_id") != prepared.get("case_id")):
        raise ValueError("CAD checkpoint-resume preparation or bundle identity differs")
    for item in record.get("inputs", []):
        rel = Path(item.get("path", "")); path = resume_dir / rel
        if rel.is_absolute() or ".." in rel.parts or path.is_symlink() or not path.is_file() or \
                path.stat().st_size != item.get("bytes") or _sha(path) != item.get("sha256"):
            raise ValueError(f"CAD checkpoint-resume input changed: {rel}")
    start, target = checkpoint["checkpoint_step"], record.get("target_step")
    numerics = prepared["science_inputs"]["numerics"]
    if (start != 2000 or target != 4000 or record.get("seed") != 20260929 or
            record.get("checkpoint", {}).get("step") != start or
            numerics.get("field_average") != {"samples": 100, "stride_steps": 10,
                                              "period_steps": 1000} or
            numerics.get("field_every_steps") != target):
        raise ValueError("CAD checkpoint-resume differs from the declared synthetic schedule")
    deck = (resume_dir / "run/in.resume").read_text(encoding="utf-8")
    if (deck.count("read_restart checkpoint.restart.%") != 1 or
            deck.count("write_restart continued.restart.%") != 1 or
            "read_particles " in deck or "create_particles " in deck or "read_surf " in deck or
            deck.count("fix src emit/surf hegas cap normal yes mflow ") != 1 or
            deck.count("collide vss hegas he.vss") != 1 or
            deck.count("surf_modify cellwalls collide walls") != 1):
        raise ValueError("CAD continuation does not restore the checked physical model exactly once")
    run = resume_dir / "run"; log = (run / "log.sparta").read_text(encoding="utf-8", errors="replace")
    if "ERROR" in log:
        raise ValueError("resumed SPARTA log contains ERROR")
    source_ranks, target_ranks = record.get("source_ranks"), record.get("target_ranks")
    if (source_ranks != checkpoint["source_ranks"] or type(target_ranks) is not int or target_ranks < 1 or
            not re.search(rf"^Running on {target_ranks} MPI task\(s\)$", log, flags=re.MULTILINE)):
        raise ValueError("resumed CAD solver MPI rank count differs from prepared checkpoint attempt")
    loops = [(float(seconds), int(steps)) for seconds, steps in re.findall(
        r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps", log)]
    if sum(steps for _, steps in loops) != target - start or not re.search(rf"^\s*{target}\s+", log, flags=re.MULTILINE):
        raise ValueError("resumed CAD solver log does not reach the declared timestep")
    corners = re.findall(r"(\d+)\s+(\d+) = cells overlapping surfs, overlap cells with unmarked corner pts", log)
    if (not corners or any(int(unmarked) for _, unmarked in corners) or
            not re.search(r"Particles stuck\s*=\s*0", log) or not re.search(r"Axisymm bad moves\s*=\s*0", log)):
        raise ValueError("resumed CAD geometry or particle health check failed")
    expected_restart = {"continued.restart.base"} | {f"continued.restart.{rank}" for rank in range(target_ranks)}
    found_restart = {path.name for path in run.glob("continued.restart.*") if path.is_file()}
    if found_restart != expected_restart:
        raise ValueError("resumed CAD final restart base/shards are incomplete")
    final_restart = []
    for name in sorted(expected_restart):
        path = run / name
        if path.stat().st_size <= 0:
            raise ValueError(f"resumed CAD final restart is empty: {name}")
        final_restart.append({"path": name, "bytes": path.stat().st_size, "sha256": _sha(path)})
    expected = "fields.4000.gz"
    found_fields = {path.name for path in run.glob("fields.*") if path.is_file()}
    if found_fields != {expected}:
        raise ValueError("resumed CAD field set does not contain the fresh final window only")
    frame = validate_frame(run / expected, expected_step=target)
    bounds = numerics["domain_bounds_m"]
    if any(abs(left - right) > 1e-12 for actual, expected_bounds in zip(frame["box_bounds_m"], bounds)
           for left, right in zip(actual, expected_bounds)):
        raise ValueError("resumed CAD field domain differs from checked preparation")
    if frame["rows"] != checkpoint["diagnostics"]["dump_rows"]:
        raise ValueError("resumed CAD field rows differ from native grid")
    first_sample = target - (numerics["field_average"]["samples"] - 1) * numerics["field_average"]["stride_steps"]
    if first_sample != 3010 or first_sample <= start + 1:
        raise ValueError("resumed CAD field is not a wholly post-checkpoint average")
    records = []
    for line in log.splitlines():
        tokens = line.split()
        if len(tokens) != 5:
            continue
        try:
            values = [float(value) for value in tokens]
        except ValueError:
            continue
        if values[0] == target:
            records.append(values)
    if not records or any(value < 0 or not value.is_integer() for value in records[-1][2:]) or records[-1][4] <= 0:
        raise ValueError("resumed CAD final source statistics are invalid or absent")
    field = {**frame, "file": expected}
    return {"schema_version": "dsmc-cad-checkpoint-resume-verification-v1", "status": "passed",
            "case_id": record["case_id"], "checkpoint_bundle_sha256": record["checkpoint_bundle_sha256"],
            "checkpoint_step": start, "target_step": target, "resumed_steps": sum(steps for _, steps in loops),
            "source_ranks": source_ranks, "target_ranks": target_ranks,
            "loop_seconds": sum(seconds for seconds, _ in loops),
            "unmarked_corners_max": max(int(unmarked) for _, unmarked in corners),
            "fields": [field], "final_restart": final_restart,
            "cad_source_counts": {"final_simulation_particles": int(records[-1][2]),
                                  "inserted_final_step": int(records[-1][3]),
                                  "inserted_cumulative": int(records[-1][4]),
                                  "count_scope": "simulation-particle source counter for resumed attempt"},
            "diagnostic_averaging": "fresh-from-checkpoint", "science_assessment": "unassessed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("resume_dir", type=Path)
    parser.add_argument("--parent-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        parser.exit(2, f"verification output already exists: {args.out}\n")
    try:
        result = verify_resume(args.resume_dir, args.parent_archive)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"verify_3d_resume: {exc}\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, args.out)
    print(f"Resume verification: {args.out}")


if __name__ == "__main__":
    main()
