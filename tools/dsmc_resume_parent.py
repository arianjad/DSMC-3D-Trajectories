"""Expose verified resumed helium as a parent for ordinary downstream stages."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

from dsmc_repository import sha


SCHEMA = "dsmc-helium-resume-attempt-v1"


def source_intervals(log: str) -> list[dict]:
    """Count simulator insertions once per run; emit/surf f[2] resets per run."""
    intervals, rows, columns = [], [], None
    for line in log.splitlines():
        tokens = line.split()
        if tokens[:1] == ["Step"] and "f_src[2]" in tokens:
            columns, rows = tokens, []
            continue
        if columns is not None and len(tokens) == len(columns):
            try:
                values = dict(zip(columns, map(float, tokens)))
            except ValueError:
                pass
            else:
                rows.append(values)
                continue
        loop = re.match(r"Loop time of .* on \d+ procs for (\d+) steps", line)
        if loop:
            if not rows or columns is None:
                raise ValueError("resume log lacks per-run source counter endpoints")
            first, last = rows[0], rows[-1]
            steps = int(loop[1])
            if (first["f_src[2]"] != 0 or last["Step"] - first["Step"] != steps or
                    last["f_src[2]"] < 0):
                raise ValueError("source counter baseline or step interval is inconsistent")
            intervals.append({"start_step": int(first["Step"]), "end_step": int(last["Step"]),
                              "reset_baseline": 0, "simulation_insertions": last["f_src[2]"]})
            rows, columns = [], None
    if not intervals:
        raise ValueError("resume log lacks complete source intervals")
    for left, right in zip(intervals, intervals[1:]):
        if left["end_step"] != right["start_step"]:
            raise ValueError("source intervals are discontinuous or overlap")
    return intervals


def frame_windows(start: int, end: int, numerics: dict, defined_after: int) -> list[dict]:
    average = numerics["field_average"]
    stride, count, period = (average[key] for key in ("stride_steps", "samples", "period_steps"))
    every = numerics["field_every_steps"]
    if every % period or count * stride > period:
        raise ValueError("unsupported resumed field averaging schedule")
    windows = []
    for step in range((start // every + 1) * every, end + 1, every):
        first = step - (count - 1) * stride
        if first <= defined_after:
            raise ValueError("saved resume frame includes an incomplete averaging window")
        windows.append({"path": f"fields.{step}.gz", "timestep": step,
                        "first_sample_step": first, "last_sample_step": step,
                        "sample_count": count, "stride_steps": stride})
    return windows


def _ref(path: Path, relative: str) -> dict:
    return {"path": relative, "sha256": sha(path), "bytes": path.stat().st_size}


def validate_resume_parent(parent: Path, record: dict) -> None:
    parent = Path(parent).resolve()
    if record.get("schema_version") != SCHEMA or record.get("status") != "complete":
        raise ValueError("resumed helium receipt is incomplete")
    if (record.get("exit_code") != 0 or record["end_step"] <= record["start_step"] or
            record["advanced_steps"] != record["end_step"] - record["start_step"]):
        raise ValueError("resumed helium interval is inconsistent")
    ancestry = {}
    for item in record["ancestry_files"]:
        relative = Path(item["path"])
        path = parent / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink() or
                not path.is_file() or path.stat().st_size != item["bytes"] or
                sha(path) != item["sha256"]):
            raise ValueError(f"resumed ancestry missing or changed: {relative}")
        if item["path"] in ancestry:
            raise ValueError("duplicate resumed ancestry reference")
        ancestry[item["path"]] = item
    intervals = record["source_count_intervals"]
    if (not intervals or intervals[0]["start_step"] != record["start_step"] or
            intervals[-1]["end_step"] != record["end_step"]):
        raise ValueError("source count intervals do not cover the resumed steps")
    for left, right in zip(intervals, intervals[1:]):
        if left["end_step"] != right["start_step"]:
            raise ValueError("resumed source intervals overlap or are discontinuous")
    for window in record["frame_windows"]:
        if (window["first_sample_step"] <= record["diagnostics_defined_after_step"] or
                window["last_sample_step"] > record["end_step"] or
                window["last_sample_step"] != window["timestep"] or
                window["last_sample_step"] - window["first_sample_step"] !=
                (window["sample_count"] - 1) * window["stride_steps"]):
            raise ValueError("resumed field has an incomplete sampling window")
    required = {"ancestry/resume.json", "ancestry/resume-run.json", "ancestry/verification.json",
                "ancestry/parent/prepared/prepared.json"}
    if not required.issubset(ancestry):
        raise ValueError("resumed parent lacks required ancestry receipts")
    for key in ("prepared", "checkpoint_bundle"):
        ref = record.get(key, {})
        if ancestry.get(ref.get("path")) != ref:
            raise ValueError(f"resumed {key} reference is not bound to ancestry")
    prepared = json.loads((parent / record["prepared"]["path"]).read_text(encoding="utf-8"))
    staging = json.loads((parent / "ancestry/resume.json").read_text(encoding="utf-8"))
    attempt = json.loads((parent / "ancestry/resume-run.json").read_text(encoding="utf-8"))
    checked = json.loads((parent / "ancestry/verification.json").read_text(encoding="utf-8"))
    if (attempt.get("status") != "complete" or attempt.get("exit_code") != 0 or
            attempt.get("resume_sha256") != ancestry["ancestry/resume.json"]["sha256"] or
            attempt.get("verification", {}).get("sha256") != ancestry["ancestry/verification.json"]["sha256"] or
            checked.get("status") != "passed" or
            checked.get("checkpoint_step") != record["start_step"] or
            checked.get("target_step") != record["end_step"] or
            checked.get("resumed_steps") != record["advanced_steps"] or
            checked.get("final_restart") != record.get("final_restart") or
            record.get("case_id") != prepared.get("case_id") or
            record.get("case_id") != checked.get("case_id") or
            record.get("solver") != attempt.get("solver") or
            record.get("profile") != attempt.get("profile")):
        raise ValueError("resumed parent disagrees with bound native verification or launcher")
    parent_hash = staging.get("checkpoint_bundle_sha256", staging.get("parent_archive_sha256"))
    if parent_hash != record["checkpoint_bundle"]["sha256"]:
        raise ValueError("resumed checkpoint parent differs from staged resume")
    windows = frame_windows(record["start_step"], record["end_step"],
                            prepared["science_inputs"]["numerics"], staging["diagnostics_defined_after_step"])
    if (record["diagnostics_defined_after_step"] != staging["diagnostics_defined_after_step"] or
            record["frame_windows"] != windows or
            {item["path"] for item in windows} != {item["file"] for item in checked["fields"]}):
        raise ValueError("resumed sampling windows differ from preparation and verified fields")
    outputs = {item["path"]: item for item in record["outputs"]}
    if len(outputs) != len(record["outputs"]):
        raise ValueError("duplicate resumed output reference")
    for name in {"run.log", "log.sparta", *(item["path"] for item in windows),
                 *(item["path"] for item in record["final_restart"])}:
        item = outputs.get(name, {})
        path = parent / name
        if (Path(name).name != name or path.is_symlink() or not path.is_file() or
                path.stat().st_size != item.get("bytes") or sha(path) != item.get("sha256")):
            raise ValueError(f"resumed output missing or changed: {name}")
    if any(outputs.get(item["path"]) != item for item in record["final_restart"]):
        raise ValueError("resumed final restart inventory differs from verified outputs")
    if (source_intervals((parent / "log.sparta").read_text(encoding="utf-8")) != intervals or
            outputs["run.log"]["sha256"] != outputs["log.sparta"]["sha256"]):
        raise ValueError("resumed source intervals differ from retained native log")


def normalize_resume(resume_dir: Path, parent_dir: Path) -> Path:
    """Add a provenance-bound parent receipt after the native resume verifier passes."""
    from verify_3d_resume import verify_resume

    resume, parent = Path(resume_dir).resolve(), Path(parent_dir).resolve()
    run = resume / "run"
    target = run / "manifest.json"
    if target.exists() or (run / "ancestry").exists():
        raise FileExistsError("resume already normalized or has an incomplete normalization")
    prepared_path = parent / "prepared/prepared.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    attempt = json.loads((resume / "resume-run.json").read_text(encoding="utf-8"))
    staging = json.loads((resume / "resume.json").read_text(encoding="utf-8"))
    verification_path = resume / "verification.json"
    if (attempt.get("status") != "complete" or attempt.get("exit_code") != 0 or
            attempt.get("resume_sha256") != sha(resume / "resume.json") or
            attempt.get("verification", {}).get("sha256") != sha(verification_path)):
        raise ValueError("resume launcher has not completed with bound verification")
    if (parent / "checkpoint-manifest.json").is_file():
        from verify_3d_resume import verify_resume_checkpoint
        checked = verify_resume_checkpoint(resume, parent)
    else:
        checked = verify_resume(resume, parent)
    if checked != json.loads(verification_path.read_text(encoding="utf-8")):
        raise ValueError("resume verification changed before downstream normalization")
    start, end = checked["checkpoint_step"], checked["target_step"]
    windows = frame_windows(start, end, prepared["science_inputs"]["numerics"],
                            staging["diagnostics_defined_after_step"])
    if {item["path"] for item in windows} != {item["file"] for item in checked["fields"]}:
        raise ValueError("verified fields disagree with complete resumed sampling windows")
    intervals = source_intervals((run / "log.sparta").read_text(encoding="utf-8"))
    ancestry = run / "ancestry"
    ancestry.mkdir()
    for name in ("resume.json", "resume-run.json", "verification.json"):
        shutil.copyfile(resume / name, ancestry / name)
    shutil.copytree(parent, ancestry / "parent")
    ancestry_files = [_ref(path, path.relative_to(run).as_posix())
                      for path in sorted(ancestry.rglob("*")) if path.is_file()]
    # Ordinary workflow verification reads run.log; retain the original log too.
    shutil.copyfile(run / "log.sparta", run / "run.log")
    outputs = [_ref(path, path.name) for path in sorted(run.iterdir()) if path.is_file()]
    parent_manifest = ("checkpoint-manifest.json" if (parent / "checkpoint-manifest.json").is_file()
                       else "archive-manifest.json")
    record = {"schema_version": SCHEMA, "status": "complete", "exit_code": 0,
              "case_id": prepared["case_id"], "prepared": _ref(prepared_path, "ancestry/parent/prepared/prepared.json"),
              "checkpoint_bundle": _ref(parent / parent_manifest, "ancestry/parent/" + parent_manifest),
              "source_attempt": staging.get("source_attempt", {"status": "complete", "kind": "full-archive"}),
              "start_step": start, "end_step": end, "advanced_steps": checked["resumed_steps"],
              "diagnostics_defined_after_step": staging["diagnostics_defined_after_step"],
              "frame_windows": windows, "source_count_intervals": intervals,
              "source_count_units": "simulation particles; not physical atoms or mass flux",
              "solver": attempt["solver"], "profile": attempt["profile"],
              "ancestry_files": ancestry_files, "outputs": outputs,
              "final_restart": checked["final_restart"], "science_assessment": "unassessed"}
    validate_resume_parent(run, record)
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return target
