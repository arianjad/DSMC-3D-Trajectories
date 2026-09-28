#!/usr/bin/env python3
"""Launch and verify one prepared B5 restart with a local machine profile."""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

from dsmc_archive import verify_archive
from dsmc_checkpoint import verify_checkpoint
from dsmc_case import load_profile
from dsmc_run import _executable, _now, _sha, _write_json, render_launch
from verify_3d_resume import verify_resume, verify_resume_checkpoint


def _checked_inputs(resume: Path, record: dict) -> int:
    """Fail before launch if the staged attempt has changed or was already used."""
    schema = record.get("schema_version")
    if schema not in ("dsmc-b5-resume-preparation-v1", "dsmc-checkpoint-resume-preparation-v1") or record.get("status") != "prepared":
        raise ValueError("not a prepared supported resume attempt")
    run = resume / "run"
    if not run.is_dir() or (resume / "resume-run.json").exists() or (resume / "verification.json").exists():
        raise ValueError("resume attempt is missing or already launched")
    names = set()
    for item in record["inputs"]:
        rel = Path(item["path"])
        if (rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != "run" or
                rel.parts[1] in (".", "..") or ".." in rel.parts):
            raise ValueError("resume input path escapes run directory")
        source = resume / rel
        if (source.is_symlink() or not source.is_file() or
                source.stat().st_size != item["bytes"] or _sha(source) != item["sha256"]):
            raise ValueError(f"resume input changed: {rel}")
        if source.name in names:
            raise ValueError("duplicate staged resume input")
        names.add(source.name)
    if {path.name for path in run.iterdir()} != names:
        raise ValueError("run directory has missing or previous-attempt files")
    restart = "checkpoint.restart" if schema == "dsmc-checkpoint-resume-preparation-v1" else "final.restart"
    if "in.resume" not in names or f"{restart}.base" not in names:
        raise ValueError("resume deck or checkpoint base is missing")
    shards = {int(match.group(1)) for name in names
              if (match := re.fullmatch(re.escape(restart) + r"\.(\d+)", name))}
    if not shards or shards != set(range(len(shards))):
        raise ValueError("resume checkpoint shards are incomplete")
    return len(shards)


def run_resume_checkpoint(resume_dir: Path, checkpoint_bundle: Path, profile_path: Path) -> Path:
    """Run one local staged checkpoint recovery and retain a terminal receipt.

    Slurm submits the same staged directory through its batch wrapper; this
    helper intentionally remains the bounded local runner used by tests.
    """
    resume, bundle = Path(resume_dir).resolve(), Path(checkpoint_bundle).resolve()
    record = json.loads((resume / "resume.json").read_text(encoding="utf-8"))
    shard_count = _checked_inputs(resume, record)
    checked = verify_checkpoint(bundle)
    if (record.get("schema_version") != "dsmc-checkpoint-resume-preparation-v1" or
            record.get("checkpoint_bundle_sha256") != _sha(bundle / "checkpoint-manifest.json") or
            record["checkpoint"]["step"] != checked["checkpoint_step"] or
            record.get("source_ranks") != shard_count):
        raise ValueError("checkpoint bundle identity or staged resume input differs from preparation")
    adapter = record.get("adapter", "b5")
    if adapter == "cad":
        from dsmc_cad_workflow import validate_prepared_cad
        validate_prepared_cad(bundle / "prepared")
    elif adapter not in ("b5", "b5-circle", "b5-slit"):
        raise ValueError("checkpoint resume has an unsupported geometry adapter")
    profile = load_profile(profile_path)
    if profile["platform"] != "local":
        raise ValueError("resume-run needs a local profile")
    target_ranks = record.get("target_ranks", shard_count)
    if type(target_ranks) is not int or target_ranks <= 0 or profile["ranks"] != target_ranks:
        raise ValueError("resume-run profile differs from prepared target rank count")
    expected = "read_restart checkpoint.restart.%"
    if target_ranks != shard_count:
        expected += " balance rcb cell"
    lines = [line.strip() for line in (resume / "run/in.resume").read_text().splitlines()
             if line.strip().startswith("read_restart ")]
    if lines != [expected]:
        raise ValueError("prepared checkpoint restart command disagrees with rank counts")
    executable = _executable(profile["solver_executable"])
    argv = render_launch(profile, executable, ["-in", "in.resume"])
    manifest_path = resume / "resume-run.json"
    manifest = {"schema_version": "dsmc-checkpoint-resume-run-v1", "status": "running",
                "case_id": record["case_id"], "created_utc": _now(), "finished_utc": None,
                "resume_sha256": _sha(resume / "resume.json"),
                "checkpoint_bundle_sha256": _sha(bundle / "checkpoint-manifest.json"),
                "profile": {"profile_id": profile["profile_id"], "source": profile["source_profile"],
                            "ranks": profile["ranks"], "threads": profile["threads"]},
                "adapter": adapter, "source_ranks": shard_count, "target_ranks": target_ranks,
                "solver": {"path": executable, "sha256": _sha(Path(executable))}, "command": argv,
                "working_directory": str(resume / "run"), "exit_code": None, "failure": None,
                "verification": None}
    _write_json(manifest_path, manifest)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    child = None
    try:
        env = os.environ.copy(); env.update({key: str(profile["threads"]) for key in
                                              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        with (resume / "run/launcher.log").open("wb") as log:
            child = subprocess.Popen(argv, cwd=resume / "run", stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=(os.name != "nt"))
            code = child.wait()
        manifest["exit_code"] = code; (resume / "run/rc.sentinel").write_text(f"{code}\n", encoding="ascii")
        if code:
            raise RuntimeError(f"solver exited with status {code}")
        verified = verify_resume_checkpoint(resume, bundle)
        verification_path = resume / "verification.json"; _write_json(verification_path, verified)
        manifest["verification"] = {"path": "verification.json", "sha256": _sha(verification_path)}
        manifest["status"] = "complete"
    except KeyboardInterrupt as exc:
        if child is not None:
            _stop_child(child)
        manifest["status"] = "interrupted"; manifest["exit_code"] = 130
        manifest["failure"] = "local checkpoint resume launcher interrupted"
        (resume / "run/rc.sentinel").write_text("130\n", encoding="ascii")
        raise RuntimeError(manifest["failure"]) from exc
    except Exception as exc:
        if child is not None:
            _stop_child(child)
        manifest["status"] = "failed"; manifest["failure"] = str(exc)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        manifest["finished_utc"] = _now(); _write_json(manifest_path, manifest)
    return verification_path


def _stop_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    if os.name != "nt":
        os.killpg(child.pid, signal.SIGTERM)
    else:
        child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(child.pid, signal.SIGKILL)
        else:
            child.kill()
        child.wait()


def run_resume(resume_dir: Path, parent_archive: Path, profile_path: Path) -> Path:
    """Run one immutable staged attempt; terminal receipt records any failure."""
    resume, parent = Path(resume_dir).resolve(), Path(parent_archive).resolve()
    record = json.loads((resume / "resume.json").read_text(encoding="utf-8"))
    shard_count = _checked_inputs(resume, record)
    checked = verify_archive(parent)
    if (record["parent_archive_sha256"] != _sha(parent / "archive-manifest.json") or
            record["case_id"] != checked["source_verification"]["case_id"] or
            record["checkpoint"]["sha256"] != _sha(parent / "workflow/helium/final.restart.base")):
        raise ValueError("parent archive or checkpoint differs from resume preparation")
    profile = load_profile(profile_path)
    if profile["platform"] != "local":
        raise ValueError("resume-run needs a local profile")
    source_ranks = record.get("source_ranks", shard_count)
    target_ranks = record.get("target_ranks", shard_count)
    if (type(source_ranks) is not int or source_ranks != shard_count or
            type(target_ranks) is not int or target_ranks <= 0):
        raise ValueError("prepared restart rank counts disagree with checkpoint shards")
    if profile["ranks"] != target_ranks:
        raise ValueError(f"resume-run needs the prepared target rank count {target_ranks}")
    restart_lines = [line.strip() for line in (resume / "run/in.resume").read_text().splitlines()
                     if line.strip().startswith("read_restart ")]
    expected_restart_line = "read_restart final.restart.%"
    if target_ranks != source_ranks:
        expected_restart_line += " balance rcb cell"
    if restart_lines != [expected_restart_line]:
        raise ValueError("prepared restart redistribution command disagrees with rank counts")
    executable = _executable(profile["solver_executable"])
    argv = render_launch(profile, executable, ["-in", "in.resume"])
    if profile["mpi_argv"] and not shutil.which(argv[0]) and not Path(argv[0]).is_file():
        raise ValueError(f"launcher executable not found: {argv[0]}")
    manifest_path = resume / "resume-run.json"
    manifest = {
        "schema_version": "dsmc-b5-resume-run-v1", "status": "running",
        "case_id": record["case_id"], "created_utc": _now(), "finished_utc": None,
        "resume_sha256": _sha(resume / "resume.json"),
        "parent_archive_sha256": _sha(parent / "archive-manifest.json"),
        "profile": {"profile_id": profile["profile_id"],
                    "source": profile["source_profile"],
                    "ranks": profile["ranks"], "threads": profile["threads"]},
        "source_ranks": source_ranks, "target_ranks": target_ranks,
        "solver": {"path": executable, "sha256": _sha(Path(executable))},
        "command": argv, "working_directory": str(resume / "run"),
        "exit_code": None, "failure": None, "verification": None,
    }
    _write_json(manifest_path, manifest)
    old_sigterm = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    child = None
    try:
        env = os.environ.copy()
        env.update({key: str(profile["threads"]) for key in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        with (resume / "run/launcher.log").open("wb") as log:
            child = subprocess.Popen(argv, cwd=resume / "run", stdout=log,
                                     stderr=subprocess.STDOUT, env=env,
                                     start_new_session=(os.name != "nt"))
            code = child.wait()
        manifest["exit_code"] = code
        (resume / "run/rc.sentinel").write_text(f"{code}\n", encoding="ascii")
        if code:
            raise RuntimeError(f"solver exited with status {code}")
        verified = verify_resume(resume, parent)
        verification_path = resume / "verification.json"
        _write_json(verification_path, verified)
        manifest["verification"] = {"path": "verification.json", "sha256": _sha(verification_path)}
        manifest["status"] = "complete"
    except KeyboardInterrupt as exc:
        if child is not None:
            _stop_child(child)
        manifest["status"] = "interrupted"
        manifest["exit_code"] = 130
        manifest["failure"] = "local resume launcher interrupted"
        (resume / "run/rc.sentinel").write_text("130\n", encoding="ascii")
        raise RuntimeError(manifest["failure"]) from exc
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = str(exc)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_sigterm)
        manifest["finished_utc"] = _now()
        _write_json(manifest_path, manifest)
    return verification_path
