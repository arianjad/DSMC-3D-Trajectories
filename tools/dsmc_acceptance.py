#!/usr/bin/env python3
"""Run the tracked local 3D acceptance path in a fresh output directory."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dsmc_case import load_profile
from dsmc_pipeline import _wsl_path


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "cases/examples/b5-circle/case.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _backend(profile_raw: dict, selected: str) -> str:
    if selected != "auto":
        return selected
    solver = profile_raw["solver_executable"]
    return "wsl" if os.name == "nt" and solver.startswith("/") else "native"


def run_local_acceptance(profile: Path, out: Path, *, helium_backend: str = "auto") -> Path:
    profile, out = Path(profile).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"acceptance output already exists: {out}")
    if not profile.is_file() or not CASE.is_file():
        raise ValueError("acceptance needs a real local profile and the tracked B5 case")
    if out == profile or out in profile.parents:
        raise ValueError("acceptance output must be separate from the profile")
    launch = load_profile(profile)
    if launch["platform"] != "local":
        raise ValueError("local acceptance needs a local profile")
    raw = json.loads(profile.read_text(encoding="utf-8"))
    backend = _backend(raw, helium_backend)
    if backend not in ("native", "wsl"):
        raise ValueError(f"unsupported helium backend: {backend}")
    if backend == "wsl" and (os.name != "nt" or not shutil.which("wsl.exe")):
        raise ValueError("WSL acceptance requires wsl.exe on Windows")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    receipt_path = out / "acceptance.json"
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    receipt = {
        "schema_version": "dsmc-3d-local-acceptance-v1", "status": "running",
        "created_utc": _now(), "suite": "local", "helium_backend": backend,
        "case": {"path": str(CASE), "sha256": _sha(CASE)},
        "profile": {"path": str(profile), "sha256": _sha(profile)},
        "repository": {"commit": commit, "dirty": dirty},
        "stages": {},
    }
    _write(receipt_path, receipt)

    def stage(name: str, command: list[str]) -> None:
        log = out / f"{name}.log"
        started = _now()
        t0 = time.perf_counter()
        with log.open("w", encoding="utf-8", newline="\n") as stream:
            code = subprocess.run(command, cwd=ROOT, stdout=stream,
                                  stderr=subprocess.STDOUT, check=False).returncode
        receipt["stages"][name] = {
            "status": "complete" if code == 0 else "failed", "command": command,
            "started_utc": started, "finished_utc": _now(),
            "duration_s": time.perf_counter() - t0, "exit_code": code,
            "log": {"path": log.name, "sha256": _sha(log)},
        }
        _write(receipt_path, receipt)
        if code:
            raise RuntimeError(f"{name} exited {code}; inspect {log}")

    def preparation_command(*args: Path | str) -> list[str]:
        if backend == "wsl":
            return ["wsl.exe", "-e", raw["python"], _wsl_path(ROOT / "tools/dsmc.py"),
                    *(_wsl_path(value) if isinstance(value, Path) else value for value in args)]
        return [sys.executable, str(ROOT / "tools/dsmc.py"),
                *(str(value) for value in args)]

    try:
        stage("fixture", [sys.executable, str(ROOT / "tools/verify_3d_fixture.py"),
                          "--out", str(out / "fixture")])
        stage("prepare", preparation_command("prepare", CASE, "--out", out / "prepared"))
        stage("solver-check", preparation_command(
            "inspect", out / "prepared", "--solver-check", "--profile", profile,
            "--out", out / "checked"))
        stage("benchmark", [sys.executable, str(ROOT / "tools/benchmark_3d.py"),
                            str(out / "checked"), "--profile", str(profile),
                            "--gas-policy", "carry-T-and-empty-ballistic",
                            "--helium-backend", backend, "--out", str(out / "benchmark")])
        stage("verify", [sys.executable, str(ROOT / "tools/verify_3d_workflow.py"),
                         str(out / "benchmark/workflow"), "--out", str(out / "verification.json")])
        benchmark = json.loads((out / "benchmark/benchmark.json").read_text(encoding="utf-8"))
        verification = json.loads((out / "verification.json").read_text(encoding="utf-8"))
        fixture = json.loads((out / "fixture/fixture-check.json").read_text(encoding="utf-8"))
        if (benchmark["status"] != "complete" or verification["status"] != "passed" or
                fixture["status"] != "complete"):
            raise ValueError("an acceptance child receipt is incomplete")
        receipt["results"] = {
            "benchmark": {"path": "benchmark/benchmark.json",
                          "sha256": _sha(out / "benchmark/benchmark.json")},
            "verification": {"path": "verification.json", "sha256": _sha(out / "verification.json")},
            "fixture": {"path": "fixture/fixture-check.json",
                        "sha256": _sha(out / "fixture/fixture-check.json")},
            "wall_seconds": benchmark["wall_seconds"],
            "workflow_bytes": benchmark["workflow_bytes"],
            "sampled_process_tree_peak_rss_bytes": benchmark["host_process_tree_peak_rss_bytes"],
            "memory_scope": benchmark["memory_scope"],
            "science_assessment": verification["science_assessment"],
            "speedup_vs_baseline": None,
            "speedup_reason": "no matched prior acceptance run supplied",
        }
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        receipt["failure"] = str(exc)
        raise
    finally:
        receipt["finished_utc"] = _now()
        _write(receipt_path, receipt)
    return receipt_path
