#!/usr/bin/env python3
"""Measure one fresh local 3D workflow with a cold Python/Julia process start."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
SLURM_BENCHMARK_SCHEMA = "dsmc-slurm-benchmark-v1"


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


def _output_size(root: Path) -> tuple[int, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return len(files), sum(path.stat().st_size for path in files)


def _tree_rss(process: psutil.Process) -> int:
    try:
        members = [process, *process.children(recursive=True)]
    except psutil.Error:
        return 0
    total = 0
    for member in members:
        try:
            total += member.memory_info().rss
        except psutil.Error:
            pass
    return total


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe receipt: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"receipt is not a JSON object: {path}")
    return data


def _seconds(value: object) -> float | None:
    """Parse Slurm TotalCPU without treating a missing value as zero."""
    text = str(value or "").strip()
    if not text or text.upper() in {"UNKNOWN", "N/A", "NONE"}:
        return None
    long_clock = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2}):(\d{2})(\.\d+)?", text)
    short_clock = re.fullmatch(r"(\d{2}):(\d{2})(\.\d+)?", text)
    if long_clock is not None:
        days, hours, minutes, seconds, fraction = long_clock.groups()
    elif short_clock is not None:
        minutes, seconds, fraction = short_clock.groups()
        days = hours = "0"
    else:
        return None
    days, hours, minutes, seconds = (int(item or 0) for item in (days, hours, minutes, seconds))
    if minutes >= 60 or seconds >= 60:
        return None
    whole_seconds = (((days * 24) + hours) * 60 + minutes) * 60 + seconds
    return float(whole_seconds) + float(fraction or 0.0)


def _number(value: object) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _record_value(record: dict, *names: str) -> object:
    for name in names:
        if name in record:
            return record[name]
    return None


def _scheduler_records(records: object, job_id: str) -> list[dict]:
    if not isinstance(records, list):
        return []
    result = []
    for item in records:
        if not isinstance(item, dict):
            continue
        row = {
            "job_id": str(_record_value(item, "job_id", "JobIDRaw", "JobID") or "").strip(),
            "job_name": str(_record_value(item, "job_name", "JobName") or "").strip(),
            "state": str(_record_value(item, "state", "State") or "UNKNOWN").strip(),
            "exit_code": str(_record_value(item, "exit_code", "ExitCode") or "").strip(),
            "elapsed_raw": _record_value(item, "elapsed_raw", "ElapsedRaw"),
            "alloc_cpus": _record_value(item, "alloc_cpus", "AllocCPUS"),
            "total_cpu": _record_value(item, "total_cpu", "TotalCPU"),
            "max_rss": _record_value(item, "max_rss", "MaxRSS"),
        }
        if row["job_id"] == job_id or row["job_id"].startswith(job_id + "."):
            result.append(row)
    return result


def _parse_sacct(stdout: str, job_id: str) -> list[dict]:
    rows = []
    for line in stdout.splitlines():
        values = [value.strip() for value in line.split("|")]
        if len(values) != 8:
            continue
        rows.append(dict(zip(("job_id", "job_name", "state", "exit_code", "elapsed_raw",
                              "alloc_cpus", "total_cpu", "max_rss"), values)))
    return _scheduler_records(rows, job_id)


def _scheduler_evidence(submission: dict, job_id: str) -> dict:
    """Use receipt-bound accounting first, then one read-only sacct query."""
    scheduler = submission.get("scheduler")
    retained = scheduler.get("records") if isinstance(scheduler, dict) else None
    if retained is None and isinstance(submission.get("scheduler_evidence"), dict):
        retained = submission["scheduler_evidence"].get("records")
    rows = _scheduler_records(retained, job_id)
    if rows:
        return {"source": "retained", "records": rows}
    command = ["sacct", "-n", "-P", "-j", job_id,
               "--format=JobIDRaw%20,JobName%128,State%32,ExitCode%16,ElapsedRaw%20,AllocCPUS%20,TotalCPU%20,MaxRSS%20"]
    try:
        observed = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"source": "unavailable", "records": [], "command": command, "error": str(exc)}
    if observed.returncode:
        return {"source": "unavailable", "records": [], "command": command,
                "returncode": observed.returncode, "stderr": (observed.stderr or "").strip()}
    return {"source": "on-demand", "records": _parse_sacct(observed.stdout or "", job_id),
            "command": command}


def _duration_between(record: dict) -> float | None:
    try:
        start = datetime.fromisoformat(record["created_utc"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(record["finished_utc"].replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    seconds = (end - start).total_seconds()
    return seconds if seconds >= 0 else None


def _unknown(reason: str) -> dict:
    return {"value": None, "status": "unknown", "reason": reason}


def collect_slurm_benchmark(submission: Path, archive: Path, out: Path) -> Path:
    """Collect read-only Slurm cost evidence for one verified full archive."""
    from dsmc_archive import verify_archive

    submission = Path(submission).resolve()
    archive = Path(archive).resolve()
    if archive.name == "archive-manifest.json":
        archive = archive.parent
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"benchmark output already exists: {out}")
    if (out == archive or out in archive.parents or archive in out.parents or
            out == submission.parent or out in submission.parents):
        raise ValueError("benchmark output must be separate from submission and archive")
    source = _read(submission)
    if source.get("schema_version") not in ("dsmc-slurm-attempt-v1", "dsmc-bootstrap-submission-v1"):
        raise ValueError("unsupported Slurm submission receipt")
    job_id = source.get("job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("submission receipt lacks one numeric job_id")
    job_name = source.get("job_name")
    if not isinstance(job_name, str) or not job_name:
        raise ValueError("submission receipt lacks job_name")
    verified = verify_archive(archive)
    workflow = _read(archive / "workflow" / "workflow.json")
    prepared = _read(archive / "prepared" / "prepared.json")
    field = _read(archive / "workflow" / "field" / "field-preparation.json")
    molecules = _read(archive / "workflow" / "molecules" / "molecule-run.json")
    helium = _read(archive / "workflow" / "helium" / "manifest.json")
    if workflow.get("status") != "complete" or field.get("status") != "complete" or \
            molecules.get("status") != "complete" or helium.get("status") != "complete":
        raise ValueError("verified archive lacks complete workflow child receipts")
    manifest = _read(archive / "archive-manifest.json")
    prepared_sha = manifest.get("source_prepared_sha256")
    if not isinstance(prepared_sha, str) or workflow.get("prepared", {}).get("sha256") != prepared_sha:
        raise ValueError("archive workflow preparation identity differs")
    if source["schema_version"] == "dsmc-slurm-attempt-v1":
        bundle = source.get("bundle", {})
        if bundle.get("case_id") != workflow.get("case_id"):
            raise ValueError("submission bundle case differs from verified archive")
        submission_archive_binding = {"status": "case-bound", "case_id": workflow.get("case_id"),
                                     "bundle_render_manifest_sha256": bundle.get("render_manifest_sha256")}
    else:
        submission_archive_binding = {"status": "bootstrap-unbound",
            "reason": "bootstrap submission has no rendered-bundle identity; archive inputs were verified independently"}
    log = archive / "workflow" / "helium" / ("source-solver.log" if
        helium.get("schema_version") == "dsmc-cad-helium-import-v1" else "run.log")
    loops = []
    if log.is_file() and not log.is_symlink():
        loops = [(float(seconds), int(steps)) for seconds, steps in re.findall(
            r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps",
            log.read_text(encoding="utf-8", errors="replace"))]
    scheduler = _scheduler_evidence(source, job_id)
    parent = next((row for row in scheduler["records"] if row["job_id"] == job_id), None)
    if parent is not None and parent["job_name"] and parent["job_name"] != job_name:
        raise ValueError("scheduler parent job name differs from submission")
    elapsed = _number(parent["elapsed_raw"]) if parent else None
    alloc_cpus = _number(parent["alloc_cpus"]) if parent else None
    total_cpu_seconds = _seconds(parent["total_cpu"]) if parent else None
    phase_seconds = {name: value.get("duration_s") if isinstance(value, dict) and
                     isinstance(value.get("duration_s"), (int, float)) else None
                     for name, value in workflow.get("phases", {}).items()}
    per_task_rss = [{"job_id": row["job_id"], "max_rss": row["max_rss"]}
                    for row in scheduler["records"] if row.get("max_rss")]
    frame = field.get("frame", {})
    verified_work = manifest.get("source_verification", {})
    configured_average = prepared.get("science_inputs", {}).get("numerics", {}).get("field_average")
    helium_start = helium.get("start_step")
    helium_end = helium.get("end_step")
    helium_advanced = helium.get("advanced_steps")
    if not isinstance(helium_start, int):
        helium_start = verified_work.get("helium_start_step")
    if not isinstance(helium_end, int):
        helium_end = verified_work.get("helium_end_step")
    if not isinstance(helium_advanced, int):
        helium_advanced = verified_work.get("helium_steps")
    field_window = {"timestep": frame.get("timestep"),
                    "recorded_sample_window": frame.get("sample_window"),
                    "configured_average": configured_average}
    record = {
        "schema_version": SLURM_BENCHMARK_SCHEMA, "status": "complete", "collected_utc": _now(),
        "submission": {"path": str(submission), "sha256": _sha(submission),
                       "schema_version": source["schema_version"], "job_id": job_id, "job_name": job_name},
        "submission_archive_binding": submission_archive_binding,
        "archive": {"path": str(archive), "manifest_sha256": _sha(archive / "archive-manifest.json"),
                    "verification": verified, "retained_bytes": verified.get("bytes")},
        "scheduler": {**scheduler, "parent": parent},
        "comparable_identity": {
            "case_id": workflow.get("case_id"), "prepared_sha256": prepared_sha,
            "field_window": field_window,
            "actual_work": {"helium_start_step": helium_start,
                            "helium_end_step": helium_end,
                            "helium_advanced_steps": helium_advanced,
                            "solver_loop_steps": sum(steps for _, steps in loops) if loops else None,
                            "solver_ranks": helium.get("profile", {}).get("ranks"),
                            "field_rows": frame.get("rows"),
                            "molecule_launched_count": molecules.get("launched_count")}},
        "timing": {
            "queue_seconds": _unknown("queue timestamps were not used for execution cost"),
            "allocation_elapsed_seconds": elapsed,
            "allocated_cpu_hours": elapsed * alloc_cpus / 3600 if elapsed is not None and alloc_cpus is not None else None,
            "consumed_cpu_hours": total_cpu_seconds / 3600 if total_cpu_seconds is not None else None,
            "workflow_phase_seconds": phase_seconds,
            "solver_loop": {"seconds": sum(seconds for seconds, _ in loops) if loops else None,
                            "steps": sum(steps for _, steps in loops) if loops else None,
                            "records": len(loops), "log": log.relative_to(archive).as_posix()},
            "archive_copy_seconds": _duration_between(manifest),
            "retrieval_seconds": _unknown("no retained retrieval timing receipt"),
            "checkpoint_write_seconds": _unknown("no retained checkpoint-write timing measurement")},
        "resources": {
            "allocated_cpus": alloc_cpus,
            "consumed_cpu_seconds": total_cpu_seconds,
            "peak_memory_bytes": _unknown("Slurm MaxRSS is per-task, not a total allocation or MPI peak"),
            "per_task_max_rss": per_task_rss,
            "peak_working_bytes": _unknown("no retained peak scratch-storage measurement"),
            "retained_bytes": verified.get("bytes")},
        "full_frame": {key: frame.get(key) for key in ("timestep", "rows", "bytes", "sha256", "sample_window")},
        "molecules": {"declared_count": molecules.get("declared_count"),
                      "launched_count": molecules.get("launched_count")},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    target = out / "slurm-benchmark.json"
    _write(target, record)
    return target


def run_benchmark(prepared: Path, profile: Path, out: Path, *, gas_policy: str,
                  helium_backend: str = "native", sample_interval_s: float = 0.5) -> Path:
    prepared, profile, out = Path(prepared).resolve(), Path(profile).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"benchmark output already exists: {out}")
    prepared_file = prepared / "prepared.json" if prepared.is_dir() else prepared
    if prepared_file.name != "prepared.json" or not prepared_file.is_file() or not profile.is_file():
        raise ValueError("benchmark needs a prepared.json and an existing local profile")
    if out == prepared_file.parent or out in prepared_file.parent.parents or prepared_file.parent in out.parents:
        raise ValueError("benchmark output must be separate from prepared input")
    if out == profile or out in profile.parents:
        raise ValueError("benchmark output must be separate from profile")
    if sample_interval_s <= 0:
        raise ValueError("sample interval must be positive")
    workflow = out / "workflow"
    command = [sys.executable, str(ROOT / "tools" / "dsmc.py"), "run",
               str(prepared_file), "--profile", str(profile),
               "--gas-policy", gas_policy, "--helium-backend", helium_backend,
               "--out", str(workflow)]
    record = {
        "schema_version": "dsmc-3d-benchmark-v1", "status": "started",
        "created_utc": _now(), "command": command,
        "prepared_sha256": _sha(prepared_file), "profile_sha256": _sha(profile),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "logical_cpus": psutil.cpu_count(logical=True),
        "physical_cpus": psutil.cpu_count(logical=False),
        "helium_backend": helium_backend,
        "memory_scope": ("Windows process tree excludes WSL guest MPI/Python memory"
                         if os.name == "nt" and helium_backend == "wsl" else
                         "child process tree sampled; short peaks may be missed"),
        "sample_interval_s": sample_interval_s,
        "host_used_memory_start_bytes": psutil.virtual_memory().used,
        "host_cpu_percent_start": psutil.cpu_percent(interval=None),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    receipt_path = out / "benchmark.json"
    _write(receipt_path, record)
    t0 = time.perf_counter()
    child = None
    try:
        with (out / "pipeline.log").open("w", encoding="utf-8", newline="\n") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            process = psutil.Process(child.pid)
            peak_rss = 0
            peak_host_used = record["host_used_memory_start_bytes"]
            peak_host_cpu = 0.0
            samples = 0
            while child.poll() is None:
                peak_rss = max(peak_rss, _tree_rss(process))
                peak_host_used = max(peak_host_used, psutil.virtual_memory().used)
                peak_host_cpu = max(peak_host_cpu, psutil.cpu_percent(interval=None))
                samples += 1
                time.sleep(sample_interval_s)
            code = child.wait()
        record["exit_code"] = code
        record["wall_seconds"] = time.perf_counter() - t0
        record["host_process_tree_peak_rss_bytes"] = peak_rss
        record["host_used_memory_peak_bytes"] = peak_host_used
        record["host_cpu_percent_peak"] = peak_host_cpu
        record["samples"] = samples
        record["pipeline_log_sha256"] = _sha(out / "pipeline.log")
        if workflow.is_dir():
            nfiles, nbytes = _output_size(workflow)
            record["workflow_files"] = nfiles
            record["workflow_bytes"] = nbytes
        workflow_receipt = workflow / "workflow.json"
        if workflow_receipt.is_file():
            record["workflow_receipt_sha256"] = _sha(workflow_receipt)
            workflow_data = json.loads(workflow_receipt.read_text(encoding="utf-8"))
            record["workflow_status"] = workflow_data.get("status")
            record["phases"] = workflow_data.get("phases", {})
        if code or record.get("workflow_status") != "complete":
            raise ValueError(f"workflow exited {code} with status {record.get('workflow_status', 'missing')}")
        record["status"] = "complete"
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt) and child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["failure"] = str(exc)
        record.setdefault("wall_seconds", time.perf_counter() - t0)
        record["finished_utc"] = _now()
        _write(receipt_path, record)
        raise
    record["finished_utc"] = _now()
    _write(receipt_path, record)
    return receipt_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gas-policy", required=True, choices=(
        "positive-T-only", "positive-T-and-empty-ballistic", "carry-T",
        "carry-T-and-empty-ballistic"))
    parser.add_argument("--helium-backend", choices=("native", "wsl"), default="native")
    args = parser.parse_args(argv)
    try:
        path = run_benchmark(args.prepared, args.profile, args.out,
                             gas_policy=args.gas_policy,
                             helium_backend=args.helium_backend)
    except (ValueError, FileExistsError, OSError) as exc:
        parser.exit(2, f"benchmark_3d: {exc}\n")
    print(f"Benchmark: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
