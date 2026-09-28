#!/usr/bin/env python3
"""Collect a producer's terminal evidence beside an immutable recovered archive.

Collect on the Slurm login host. Copy the resulting package, then add an
unchanged verified full archive as ``full-archive/`` and run ``verify``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from dsmc_archive import verify_archive
from dsmc_checkpoint import verify_checkpoint


SCHEMA = "dsmc-recovery-evidence-v1"
SACCT_FORMAT = "JobIDRaw%20,JobName%128,State%32,ExitCode%16,Submit%32,Start%32,End%32"
INTERRUPTED = {"CANCELLED", "TIMEOUT", "FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _historical_join(root: str, *parts: str) -> str:
    """Keep captured Linux path syntax when verifying on Windows."""
    path = PurePosixPath(root) if root.startswith("/") else Path(root)
    return str(path.joinpath(*parts))


def _state(value: str) -> str:
    return str(value or "UNKNOWN").upper().split()[0].rstrip("+")


def _accounting(raw: str, job_id: str, job_name: str) -> tuple[str, list[dict]]:
    records = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 7:
            raise ValueError("sacct output has an unsupported row")
        if parts[0] != job_id and not parts[0].startswith(job_id + "."):
            raise ValueError("sacct returned a different job ID")
        records.append(dict(zip(("job_id", "job_name", "state", "exit_code", "submit", "start", "end"), parts)))
    parent = [item for item in records if item["job_id"] == job_id]
    if len(parent) > 1 or (parent and parent[0]["job_name"] != job_name):
        raise ValueError("sacct parent job name or row count differs from submission")
    if not parent:
        return "UNKNOWN", records
    return _state(parent[0]["state"]), records


def _copy(source: Path, target: Path) -> dict:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"missing or unsafe evidence source: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if _sha(source) != _sha(target):
        raise ValueError(f"evidence copy changed: {source}")
    return {"path": target.name, "bytes": target.stat().st_size, "sha256": _sha(target)}


def _bindings(evidence: Path, job_id: str) -> dict:
    submission = _read(evidence / "submission.json")
    source = _read(evidence / "checkpoint-source.json")
    helium = _read(evidence / "helium-manifest.json")
    render = _read(evidence / "render-manifest.json")
    site = _read(evidence / "site-profile.json")
    launch = _read(evidence / "launch-profile.json")
    prepared = _read(evidence / "prepared.json")
    rendered_files = {item["path"]: item for item in render.get("files", [])}
    bundle_path = submission.get("bundle", {}).get("path")
    historical_job = _historical_join(bundle_path or "", "job.sh")
    expected_submit = ["sbatch", "--parsable", "--job-name", submission.get("job_name"),
                       historical_job, bundle_path]
    if (submission.get("schema_version") != "dsmc-slurm-attempt-v1" or
            submission.get("job_id") != job_id or
            submission.get("submit_command") != expected_submit or
            submission.get("bundle", {}).get("receipt_kind") != "helium-only" or
            submission.get("bundle", {}).get("render_manifest_sha256") != _sha(evidence / "render-manifest.json") or
            submission.get("profile", {}).get("sha256") != _sha(evidence / "site-profile.json") or
            render.get("receipt_kind") != "helium-only" or
            render.get("case_id") != submission.get("bundle", {}).get("case_id") or
            render.get("site_id") != submission.get("bundle", {}).get("site_id") or
            render.get("repository_identity") != submission.get("bundle", {}).get("repository_identity") or
            site.get("site_id") != render.get("site_id") or
            launch.get("ranks") != site.get("ranks") or
            launch.get("solver_executable") != site.get("solver_executable") or
            source.get("schema_version") != "dsmc-checkpoint-source-v1" or
            source.get("prepared", {}).get("sha256") != _sha(evidence / "prepared.json") or
            prepared.get("case_id") != render.get("case_id") or
            source.get("prepared_record") != prepared or
            helium.get("schema_version") != "dsmc-helium-attempt-v1" or
            helium.get("case_id") != prepared.get("case_id") or
            helium.get("prepared", {}).get("sha256") != source["prepared"]["sha256"] or
            helium.get("solver", {}).get("sha256") != source.get("solver", {}).get("sha256") or
            helium.get("profile", {}).get("ranks") != source.get("producer", {}).get("ranks") or
            source.get("producer", {}).get("ranks") != launch.get("ranks")):
        raise ValueError("producer submission, source, manifest or render identity differs")
    for name in ("job.sh", "site-profile.json", "launch-profile.json", "prepared/prepared.json"):
        item = rendered_files.get(name)
        copied = evidence / ("prepared.json" if name == "prepared/prepared.json" else name)
        if item is None or item.get("sha256") != _sha(copied) or item.get("bytes") != copied.stat().st_size:
            raise ValueError(f"render inventory differs from retained {name}")
    return {"case_id": prepared["case_id"], "source_sha256": _sha(evidence / "checkpoint-source.json"),
            "render_sha256": _sha(evidence / "render-manifest.json"),
            "producer_manifest_status": helium.get("status"),
            "launch_manifest_sha256": source.get("attempt_identity", {}).get("sha256")}


def collect(submission_path: Path, helium_dir: Path, out: Path, *, runner=subprocess.run) -> Path:
    """Capture one exact Slurm parent and copied prelaunch/observed receipts."""
    submission_path, helium_dir, out = map(lambda path: Path(path).resolve(),
                                           (submission_path, helium_dir, out))
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if (out == helium_dir or out in helium_dir.parents or helium_dir in out.parents or
            out == submission_path.parent or out in submission_path.parents):
        raise ValueError("evidence package must be separate from producer inputs")
    submission = _read(submission_path)
    job_id = submission.get("job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("submission lacks one numeric job ID")
    bundle = Path(submission.get("bundle", {}).get("path", ""))
    if not bundle.is_absolute() or not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("submission has no safe rendered bundle")
    launch = _read(bundle / "launch-profile.json")
    expected = Path(launch["scratch_root"]) / f"job-{job_id}" / "workflow" / "helium"
    if not helium_dir.is_dir() or not expected.is_dir() or not os.path.samefile(expected, helium_dir):
        raise ValueError("helium directory is not the submitted job's derived run path")
    command = ["sacct", "-n", "-P", "-j", job_id, "--format=" + SACCT_FORMAT]
    observed = runner(command, capture_output=True, text=True, check=False, timeout=30)
    stdout, stderr = observed.stdout or "", observed.stderr or ""
    state, rows = _accounting(stdout, job_id, submission.get("job_name", "")) if observed.returncode == 0 else ("UNKNOWN", [])
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-evidence-", dir=out.parent) as temporary:
        stage = Path(temporary) / "package"; evidence = stage / "evidence"; evidence.mkdir(parents=True)
        sources = {"submission.json": submission_path,
                   "collector.py": Path(__file__).resolve(),
                   "checkpoint-source.json": helium_dir / "checkpoint-source.json",
                   "helium-manifest.json": helium_dir / "manifest.json",
                   "render-manifest.json": bundle / "render-manifest.json",
                   "site-profile.json": bundle / "site-profile.json",
                   "launch-profile.json": bundle / "launch-profile.json",
                   "prepared.json": bundle / "prepared/prepared.json",
                   "job.sh": bundle / "job.sh"}
        files = []
        for name, source in sources.items():
            item = _copy(source, evidence / name); files.append(item)
        for name, value in (("sacct.stdout", stdout), ("sacct.stderr", stderr)):
            target = evidence / name; target.write_text(value, encoding="utf-8", newline="\n")
            files.append({"path": name, "bytes": target.stat().st_size, "sha256": _sha(target)})
        bindings = _bindings(evidence, job_id)
        record = {"schema_version": SCHEMA, "captured_utc": datetime.now(timezone.utc).isoformat(),
                  "job_id": job_id, "job_name": submission["job_name"], "scheduler_state": state,
                  "interrupted_parent": state in INTERRUPTED, "sacct": {"command": command,
                      "exit_code": observed.returncode, "rows": rows},
                  "collector_sha256": _sha(evidence / "collector.py"),
                  "producer_run": {"observed": str(helium_dir), "derived": str(expected),
                                   "samefile_at_capture": True},
                  "bindings": bindings, "files": files}
        (evidence / "evidence.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        verify_evidence(stage)
        stage.rename(out)
    return out / "evidence/evidence.json"


def verify_evidence(package: Path) -> dict:
    package = Path(package).resolve(); evidence = package / "evidence"
    record = _read(evidence / "evidence.json")
    if record.get("schema_version") != SCHEMA:
        raise ValueError("unsupported recovery evidence")
    listed = set()
    for item in record.get("files", []):
        rel = Path(item.get("path", ""))
        if rel.is_absolute() or len(rel.parts) != 1 or rel.name in listed:
            raise ValueError("unsafe or duplicate evidence inventory")
        listed.add(rel.name); path = evidence / rel
        if path.is_symlink() or not path.is_file() or path.stat().st_size != item.get("bytes") or _sha(path) != item.get("sha256"):
            raise ValueError(f"evidence file changed: {rel}")
    required = {"submission.json", "collector.py", "checkpoint-source.json", "helium-manifest.json",
                "render-manifest.json", "site-profile.json", "launch-profile.json",
                "prepared.json", "job.sh", "sacct.stdout", "sacct.stderr"}
    if listed != required:
        raise ValueError("evidence inventory lacks a required producer or scheduler file")
    if {path.name for path in evidence.iterdir()} != listed | {"evidence.json"}:
        raise ValueError("evidence file set differs from inventory")
    job_id = record.get("job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("evidence has invalid job ID")
    launch = _read(evidence / "launch-profile.json")
    expected_run = _historical_join(launch["scratch_root"], f"job-{job_id}", "workflow", "helium")
    if (record.get("collector_sha256") != _sha(evidence / "collector.py") or
            record.get("producer_run", {}).get("derived") != expected_run or
            record["producer_run"].get("samefile_at_capture") is not True):
        raise ValueError("evidence collector or producer run binding differs")
    if record.get("bindings") != _bindings(evidence, job_id):
        raise ValueError("evidence producer bindings differ")
    sacct = record.get("sacct", {})
    if sacct.get("command") != ["sacct", "-n", "-P", "-j", job_id, "--format=" + SACCT_FORMAT]:
        raise ValueError("evidence scheduler command differs from exact job query")
    state, rows = _accounting((evidence / "sacct.stdout").read_text(encoding="utf-8"), job_id,
                              record.get("job_name", "")) if sacct.get("exit_code") == 0 else ("UNKNOWN", [])
    if (record.get("job_name") != _read(evidence / "submission.json").get("job_name") or
            sacct.get("rows") != rows or record.get("scheduler_state") != state or
            record.get("interrupted_parent") is not (state in INTERRUPTED)):
        raise ValueError("evidence scheduler state differs from retained sacct output")
    return {"status": "passed", "job_id": job_id, "parent_state": state,
            "interrupted_parent": state in INTERRUPTED, "source_sha256": record["bindings"]["source_sha256"]}


def verify_package(package: Path) -> dict:
    package = Path(package).resolve()
    parent = verify_evidence(package)
    if not parent["interrupted_parent"]:
        raise ValueError(f"producer is not terminally interrupted: {parent['parent_state']}")
    archive = package / "full-archive"
    if archive.is_symlink():
        raise ValueError("full archive must be an immutable copied directory")
    checked = verify_archive(archive)
    helium = _read(archive / "workflow/helium/manifest.json")
    checkpoint_dir = archive / "workflow/helium/ancestry/parent"
    checkpoint = verify_checkpoint(checkpoint_dir)
    bundle = _read(checkpoint_dir / "checkpoint-manifest.json")
    probe = _read(checkpoint_dir / "probe/native-probe.json")
    probe_job = probe.get("allocation", {}).get("job_id")
    if (helium.get("schema_version") != "dsmc-helium-resume-attempt-v1" or
            helium.get("status") != "complete" or helium.get("exit_code") != 0 or
            bundle.get("source_receipt", {}).get("sha256") != parent["source_sha256"] or
            _sha(checkpoint_dir / "checkpoint-source.json") != parent["source_sha256"] or
            checkpoint.get("checkpoint_step") != helium.get("start_step") or
            bundle.get("generation", {}).get("step") != helium.get("start_step") or
            probe.get("status") != "passed" or
            not isinstance(probe_job, str) or not probe_job.isdigit() or
            probe_job == parent["job_id"] or
            checked.get("source_verification", {}).get("status") != "passed"):
        raise ValueError("recovered archive does not bind source, native probe and successful child")
    return {"status": "passed", "producer_job_id": parent["job_id"],
            "producer_scheduler_state": parent["parent_state"], "recovered_child_status": "complete",
            "checkpoint_step": checkpoint["checkpoint_step"],
            "archive_manifest_sha256": _sha(archive / "archive-manifest.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect_cmd = commands.add_parser("collect")
    collect_cmd.add_argument("--submission", type=Path, required=True)
    collect_cmd.add_argument("--helium-dir", type=Path, required=True)
    collect_cmd.add_argument("--out", type=Path, required=True)
    verify_cmd = commands.add_parser("verify")
    verify_cmd.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (collect(args.submission, args.helium_dir, args.out) if args.command == "collect"
                  else verify_package(args.package))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"recovery evidence: {exc}\n")
    print(json.dumps(result if isinstance(result, dict) else {"evidence": str(result)}, indent=2))


if __name__ == "__main__":
    main()
