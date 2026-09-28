#!/usr/bin/env python3
"""Exercise a fresh local B5 workflow and verify a copied analysis bundle."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from dsmc_acceptance import _now, _sha, _write, run_local_acceptance
from dsmc_export import export
from verify_3d_export import verify_export


def run_handoff_acceptance(profile: Path, out: Path, *, helium_backend: str = "auto") -> Path:
    profile, out = Path(profile).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"handoff output already exists: {out}")
    if not profile.is_file():
        raise ValueError("handoff acceptance needs a real local profile")
    out.mkdir(parents=True)
    receipt_path = out / "handoff.json"
    receipt = {
        "schema_version": "dsmc-3d-handoff-acceptance-v1",
        "status": "running", "suite": "handoff", "created_utc": _now(),
        "profile_sha256": _sha(profile), "stages": {},
        "scope": "same-host rehearsal; not a student walkthrough or another-machine test",
    }
    _write(receipt_path, receipt)

    def stage(name, action):
        started, t0 = _now(), time.perf_counter()
        try:
            result = action()
        except BaseException as exc:
            receipt["stages"][name] = {
                "status": "failed", "started_utc": started,
                "finished_utc": _now(), "duration_s": time.perf_counter() - t0,
                "failure": str(exc),
            }
            _write(receipt_path, receipt)
            raise
        receipt["stages"][name] = {
            "status": "complete", "started_utc": started,
            "finished_utc": _now(), "duration_s": time.perf_counter() - t0,
        }
        _write(receipt_path, receipt)
        return result

    try:
        local_path = stage("local", lambda: run_local_acceptance(
            profile, out / "local", helium_backend=helium_backend))
        local = json.loads(local_path.read_text(encoding="utf-8"))
        if local.get("status") != "complete":
            raise ValueError("local acceptance receipt is incomplete")
        bundle = out / "analysis-bundle"
        manifest = stage("export", lambda: export(out / "local/benchmark/workflow", bundle))
        source_check = stage("verify-export", lambda: verify_export(bundle))
        copied = out / "copied-bundle"
        stage("copy", lambda: shutil.copytree(bundle, copied))
        copy_check = stage("verify-copy", lambda: verify_export(copied))
        if source_check != copy_check or source_check.get("status") != "passed":
            raise ValueError("copied bundle verification differs from the export")
        receipt["results"] = {
            "local_receipt": {"path": "local/acceptance.json", "sha256": _sha(local_path)},
            "export_manifest": {"path": "analysis-bundle/export-manifest.json",
                                "sha256": _sha(manifest)},
            "copied_manifest": {"path": "copied-bundle/export-manifest.json",
                                "sha256": _sha(copied / "export-manifest.json")},
            "bundle_check": copy_check,
            "science_assessment": local["results"]["science_assessment"],
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
