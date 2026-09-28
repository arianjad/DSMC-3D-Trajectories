#!/usr/bin/env python3
"""Render, submit once, and reconcile a single-case Slurm B5 workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dsmc_archive import verify_archive
from dsmc_case import inspect_geometry
from dsmc_repository import repository_identity


ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"[A-Za-z0-9_.-]+\Z")
TIME = re.compile(r"\d{1,3}:\d{2}:\d{2}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, record: dict) -> None:
    """Atomically replace a JSON receipt after its exclusive first creation."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exclusive_json(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _posix_path(value: object, key: str) -> str:
    if (not isinstance(value, str) or not value.startswith("/") or
            any(ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ValueError(f"{key} must be an absolute Linux path without control characters")
    return value


def validate_profile(profile: dict) -> dict:
    required = {"schema_version", "site_id", "site_runtime_verified", "partition",
                "account", "walltime", "nodes", "ranks", "threads", "memory_gb",
                "max_disk_gb", "repository_root", "setup_script", "scratch_root",
                "results_root", "python", "julia", "solver_executable", "mpi_argv",
                "gas_policy"}
    optional = {"readiness_receipt", "splitmap_solver"}
    if not isinstance(profile, dict) or not required <= set(profile) or set(profile) - required - optional:
        raise ValueError("Slurm profile keys differ from dsmc-slurm-profile-v1")
    if profile["schema_version"] != "dsmc-slurm-profile-v1":
        raise ValueError("unsupported Slurm profile schema")
    for key in ("site_id", "partition"):
        if not isinstance(profile[key], str) or not NAME.fullmatch(profile[key]):
            raise ValueError(f"invalid Slurm {key}")
    if profile["account"] is not None and (not isinstance(profile["account"], str) or
                                           not NAME.fullmatch(profile["account"])):
        raise ValueError("invalid Slurm account")
    if type(profile["site_runtime_verified"]) is not bool:
        raise ValueError("site_runtime_verified must be boolean")
    if not isinstance(profile["walltime"], str) or not TIME.fullmatch(profile["walltime"]):
        raise ValueError("walltime must be H:MM:SS")
    hours, minutes, seconds = map(int, profile["walltime"].split(":"))
    if hours == 0 and minutes == 0 and seconds == 0 or minutes >= 60 or seconds >= 60:
        raise ValueError("walltime must be positive with minutes and seconds below 60")
    for key in ("nodes", "ranks", "threads", "memory_gb", "max_disk_gb"):
        if type(profile[key]) is not int or profile[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if profile["nodes"] != 1:
        raise ValueError("first Slurm route supports one node only")
    for key in ("repository_root", "setup_script", "scratch_root", "results_root",
                "solver_executable"):
        _posix_path(profile[key], key)
    if "readiness_receipt" in profile:
        _posix_path(profile["readiness_receipt"], "readiness_receipt")
    if "splitmap_solver" in profile:
        splitmap = profile["splitmap_solver"]
        if (not isinstance(splitmap, dict) or set(splitmap) != {"path", "sha256"} or
                not SHA256.fullmatch(str(splitmap.get("sha256", "")))):
            raise ValueError("splitmap_solver needs exactly an absolute path and lowercase SHA-256")
        _posix_path(splitmap["path"], "splitmap_solver.path")
    if profile["scratch_root"] == profile["results_root"]:
        raise ValueError("scratch_root and results_root must differ")
    for key in ("python", "julia"):
        if (not isinstance(profile[key], str) or not profile[key] or
                any(ord(char) < 32 or ord(char) == 127 for char in profile[key])):
            raise ValueError(f"invalid {key} executable")
    argv = profile["mpi_argv"]
    if (not isinstance(argv, list) or not argv or
            any(not isinstance(token, str) or not token or
                any(ord(char) < 32 or ord(char) == 127 for char in token) for token in argv) or
            not any("{ranks}" in token for token in argv)):
        raise ValueError("mpi_argv needs a site-chosen launcher and {ranks} placeholder")
    if profile["gas_policy"] not in ("carry-T-and-empty-ballistic", "positive-T-only",
                                     "positive-T-and-empty-ballistic", "carry-T"):
        raise ValueError("unsupported field gas policy")
    if profile["site_runtime_verified"] and any(
            "/CHANGE_ME/" in profile[key] for key in
            ("repository_root", "setup_script", "scratch_root", "results_root",
             "solver_executable")):
        raise ValueError("verified site profile still contains a placeholder path")
    return profile


def _verify_recovery_source(source_attempt: Path, copied_receipt: Path, prepared: Path) -> dict:
    """Bind an interrupted attempt to the exact prelaunch preparation bytes."""
    source_attempt, copied_receipt, prepared = (Path(item).resolve()
                                                 for item in (source_attempt, copied_receipt, prepared))
    source = source_attempt / "checkpoint-source.json"
    if source.is_symlink() or copied_receipt.is_symlink() or not source.is_file() or not copied_receipt.is_file():
        raise ValueError("recovery checkpoint source receipt is missing or unsafe")
    if _sha(source) != _sha(copied_receipt):
        raise ValueError("recovery checkpoint source changed after Slurm rendering")
    record = json.loads(source.read_text(encoding="utf-8"))
    expected = record.get("prepared", {}).get("sha256")
    actual = _sha(prepared / "prepared.json") if (prepared / "prepared.json").is_file() else None
    if not isinstance(expected, str) or expected != actual:
        raise ValueError("recovery checkpoint source preparation differs from rendered bundle")
    return {"status": "passed", "source_attempt": str(source_attempt), "sha256": expected}


def _cad_recovery_parameters(prepared: dict) -> dict:
    """Admit only the named CAD recovery experiment, never a generic deck."""
    science = prepared.get("science_inputs", {})
    numerics, helium = science.get("numerics", {}), science.get("helium", {})
    if (prepared.get("case_id") != "synthetic-off-axis-cad-hpc" or
            science.get("geometry", {}).get("adapter") != "cad" or
            prepared.get("cad_workflow", {}).get("schema_version") !=
            "dsmc-cad-workflow-preparation-v1" or
            numerics.get("total_steps") != 4000 or numerics.get("dt_s") != 1e-7 or
            numerics.get("solver_seed") != 20260927 or
            numerics.get("restart_prefix") != "cad.restart" or
            numerics.get("restart_every_steps") != 2000 or
            numerics.get("field_every_steps") != 4000 or
            numerics.get("field_average") != {"samples": 100, "stride_steps": 10,
                                               "period_steps": 1000} or
            helium.get("wall_temperature_K") != 4 or helium.get("wall_accommodation") != 1 or
            helium.get("inlet", {}).get("model") != "emit/surf-mflow"):
        raise ValueError("CAD Slurm recovery only supports the declared 2000-to-4000 synthetic fixture")
    return {"seed": 20260929, "target_step": 4000, "checkpoint_step": 2000,
            "checkpoint_name": "cad.restart.base.2000",
            "averaging": {"first_sample_step": 3010, "stride_steps": 10,
                          "sample_count": 100, "last_sample_step": 4000}}


def _splitmap_solver(profile: dict, *, required: bool) -> dict | None:
    solver = profile.get("splitmap_solver")
    if solver is None:
        if required:
            raise ValueError("full CAD Slurm workflow requires a hash-bound splitmap_solver")
        return None
    return solver


def _script(profile: dict, repository: dict | None, *, helium_only: bool = False,
            recovery_attempt: Path | None = None, recovery: dict | None = None,
            splitmap_solver: dict | None = None) -> str:
    if helium_only and recovery_attempt is not None:
        raise ValueError("helium_only and recovery_attempt are mutually exclusive")
    quote = shlex.quote
    directives = ["#!/bin/bash", f"#SBATCH --job-name=dsmc-{profile['site_id']}",
                  f"#SBATCH --partition={profile['partition']}",
                  f"#SBATCH --time={profile['walltime']}",
                  "#SBATCH --nodes=1", f"#SBATCH --ntasks={profile['ranks']}",
                  f"#SBATCH --cpus-per-task={profile['threads']}",
                  f"#SBATCH --mem={profile['memory_gb']}G",
                  "#SBATCH --output=slurm-%j.out"]
    if profile["account"]:
        directives.append(f"#SBATCH --account={profile['account']}")
    python = quote(profile["python"])
    repo = quote(profile["repository_root"])
    scratch = quote(profile["scratch_root"])
    results = quote(profile["results_root"])
    setup = quote(profile["setup_script"])
    body = ["set -Eeuo pipefail", "bundle_dir=$(cd -- \"${1:?Pass the rendered bundle directory}\" && pwd -P)",
            "[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo 'No numeric Slurm job ID' >&2; exit 2; }",
            "test -z \"${SLURM_ARRAY_TASK_ID:-}\" || { echo 'Arrays are not enabled for this bundle' >&2; exit 2; }"]
    if not profile["site_runtime_verified"]:
        body.append("echo 'Site runtime is unverified; render-only profile cannot execute' >&2; exit 2")
    body.extend([f"results_root={results}",
                 "attempt_dir=", "stage=setup",
                 "retain_failure() {",
                 "  local rc=$? failure_dir relative source target",
                 "  trap - EXIT",
                 "  if (( rc == 0 )); then return; fi",
                 "  set +e",
                 "  failure_dir=\"$results_root/job-$SLURM_JOB_ID.failed\"",
                 "  if mkdir -p -- \"$results_root\" && mkdir -- \"$failure_dir\"; then",
                 "    { printf 'schema_version=dsmc-slurm-failure-v1\\n'",
                 "      printf 'job_id=%s\\n' \"$SLURM_JOB_ID\"",
                 "      printf 'exit_code=%s\\n' \"$rc\"",
                 "      printf 'stage=%s\\n' \"$stage\"",
                 "      printf 'bundle_dir=%s\\n' \"$bundle_dir\"",
                 "      printf 'scratch_attempt=%s\\n' \"$attempt_dir\"",
                 "    } > \"$failure_dir/failure.txt\"",
                 "    if [[ -n $attempt_dir ]]; then",
                  "      for relative in workflow.json helium/manifest.json splitmap/splitmap-probe.json field/field-preparation.json molecules/molecule-run.json; do",
                 "        source=\"$attempt_dir/workflow/$relative\"",
                 "        target=\"$failure_dir/$relative\"",
                 "        if [[ -f $source && ! -L $source ]]; then",
                 "          mkdir -p -- \"${target%/*}\" && cp -- \"$source\" \"$target\" || echo \"Could not retain $relative\" >&2",
                 "        fi",
                 "      done",
                 "      for relative in checkpoint/checkpoint-manifest.json checkpoint/checkpoint-source.json checkpoint/probe/native-probe.json resume/resume.json resume/resume-run.json resume/verification.json resume/run/manifest.json; do",
                 "        source=\"$attempt_dir/$relative\"",
                 "        target=\"$failure_dir/recovery/$relative\"",
                 "        if [[ -f $source && ! -L $source ]]; then",
                 "          mkdir -p -- \"${target%/*}\" && cp -- \"$source\" \"$target\" || echo \"Could not retain $relative\" >&2",
                 "        fi",
                 "      done",
                 "    fi",
                 "    source=\"$results_root/job-$SLURM_JOB_ID/archive-manifest.json\"",
                 "    if [[ -f $source && ! -L $source ]]; then",
                 "      cp -- \"$source\" \"$failure_dir/archive-manifest.json\" || echo 'Could not retain archive manifest' >&2",
                 "    fi",
                 "  else",
                 "    echo \"Could not retain failure receipt at $failure_dir\" >&2",
                 "  fi",
                 "  exit \"$rc\"",
                 "}",
                 "trap retain_failure EXIT",
                 f"source {setup}",
                 "stage=verify-bundle",
                 f"{python} {repo}/tools/dsmc_slurm.py verify --bundle \"$bundle_dir\"",
                 "stage=verify-repository",
                 f"test -d {repo} || {{ echo 'Repository absent on compute node' >&2; exit 2; }}"])
    if repository is not None:
        body.append(f"{python} {repo}/tools/dsmc_repository.py verify --root {repo} "
                    f"--commit {quote(repository['commit'])}")
        if repository.get("manifest_sha256"):
            body[-1] += " --manifest-sha256 " + quote(repository["manifest_sha256"])
    if "readiness_receipt" in profile:
        body.extend(["stage=verify-site-readiness",
                     f"{python} {repo}/tools/engaging/prepare_site.py verify "
                      "--profile \"$bundle_dir/site-profile.json\""])
    if splitmap_solver is not None:
        splitmap_path = quote(splitmap_solver["path"])
        splitmap_sha = quote(splitmap_solver["sha256"])
        body.extend(["stage=verify-splitmap-solver",
                     f"test -f {splitmap_path} && test ! -L {splitmap_path} || "
                     "{ echo 'Split-map solver is missing or unsafe' >&2; exit 2; }",
                     f"splitmap_actual=$(sha256sum -- {splitmap_path} | awk '{{print $1}}')",
                     f"[[ $splitmap_actual == {splitmap_sha} ]] || "
                     "{ echo 'Split-map solver hash differs from rendered site profile' >&2; exit 2; }"])
    body.extend(["stage=prepare-scratch",
                 f"mkdir -p {scratch} {results}",
                 f"attempt_dir={scratch}/job-\"$SLURM_JOB_ID\"",
                 "test ! -e \"$attempt_dir\" || { echo 'Attempt path already exists' >&2; exit 2; }",
                 "mkdir \"$attempt_dir\""])
    if helium_only:
        body.extend(["stage=run-helium-only",
                     f"{python} {repo}/tools/dsmc_run.py --prepared \"$bundle_dir/prepared\" "
                     "--profile \"$bundle_dir/launch-profile.json\" "
                     "--run-dir \"$attempt_dir/workflow/helium\"",
                     "stage=retain-helium-only-receipt",
                     f"cp -- \"$attempt_dir/workflow/helium/manifest.json\" {results}/job-\"$SLURM_JOB_ID\".helium-only.json",
                     "echo \"Retained helium-only Slurm job $SLURM_JOB_ID; no full workflow archive was claimed\""])
    elif recovery_attempt is not None:
        source = quote(str(Path(recovery_attempt).resolve()))
        parameters = recovery or {"seed": 20260926, "target_step": 4800}
        checkpoint = parameters.get("checkpoint_name")
        checkpoint_arg = (" --checkpoint " + quote(str(Path(recovery_attempt).resolve() / checkpoint))
                          if checkpoint else "")
        splitmap_arg = (" --splitmap-solver " + quote(splitmap_solver["path"])
                        if splitmap_solver is not None else "")
        body.extend(["stage=verify-recovery-source",
                     f"{python} {repo}/tools/dsmc_slurm.py verify-recovery --source-attempt {source} "
                     "--receipt \"$bundle_dir/recovery-source.json\" --prepared \"$bundle_dir/prepared\"",
                      "stage=checkpoint-archive",
                      f"{python} {repo}/tools/dsmc.py checkpoint-archive {source} "
                     f"--profile \"$bundle_dir/launch-profile.json\"{checkpoint_arg} --out \"$attempt_dir/checkpoint\"",
                      "stage=resume-checkpoint",
                      f"{python} {repo}/tools/dsmc.py resume-checkpoint \"$attempt_dir/checkpoint\" "
                     f"--out \"$attempt_dir/resume\" --seed {parameters['seed']} "
                     f"--target-step {parameters['target_step']}",
                     "stage=resume-run",
                     f"{python} {repo}/tools/dsmc.py resume-run \"$attempt_dir/resume\" "
                     "--parent-archive \"$attempt_dir/checkpoint\" --profile \"$bundle_dir/launch-profile.json\"",
                     "stage=normalize-resume",
                     f"{python} {repo}/tools/dsmc.py normalize-resume \"$attempt_dir/resume\" --parent \"$attempt_dir/checkpoint\"",
                     "stage=run-resumed-workflow",
                      f"{python} {repo}/tools/dsmc.py finish \"$attempt_dir/checkpoint/prepared\" "
                      "--helium-receipt \"$attempt_dir/resume/run/manifest.json\" "
                      "--profile \"$bundle_dir/launch-profile.json\" --gas-policy "
                     f"{quote(profile['gas_policy'])}{splitmap_arg} --out \"$attempt_dir/workflow\"",
                     "stage=archive",
                     f"{python} {repo}/tools/dsmc.py archive \"$attempt_dir/workflow\" "
                     f"--out {results}/job-\"$SLURM_JOB_ID\"",
                     "echo \"Archived recovered Slurm job $SLURM_JOB_ID; scratch attempt retained at $attempt_dir\""])
    else:
        splitmap_arg = (" --splitmap-solver " + quote(splitmap_solver["path"])
                        if splitmap_solver is not None else "")
        body.extend(["stage=run-workflow",
                      f"{python} {repo}/tools/dsmc.py run \"$bundle_dir/prepared\" "
                      f"--profile \"$bundle_dir/launch-profile.json\" --gas-policy "
                     f"{quote(profile['gas_policy'])}{splitmap_arg} --out \"$attempt_dir/workflow\"",
                     "stage=archive",
                     f"{python} {repo}/tools/dsmc.py archive \"$attempt_dir/workflow\" "
                     f"--out {results}/job-\"$SLURM_JOB_ID\"",
                     "echo \"Archived Slurm job $SLURM_JOB_ID; scratch attempt retained at $attempt_dir\""])
    return "\n".join([*directives, "", *body, ""])


def verify_render(bundle: Path) -> dict:
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "render-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "dsmc-slurm-render-v1":
        raise ValueError("unsupported Slurm render manifest")
    listed = set()
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or item["path"] in listed:
            raise ValueError("invalid rendered bundle path")
        listed.add(item["path"])
        path = bundle / relative
        if (path.is_symlink() or not path.is_file() or
                path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]):
            raise ValueError(f"rendered file is missing or changed: {item['path']}")
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise ValueError("rendered bundle contains a symlink")
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    if actual != listed | {"render-manifest.json"}:
        raise ValueError("rendered bundle has unlisted or missing files")
    inspected = inspect_geometry(bundle / "prepared")
    if inspected["state"] != "solver-checked":
        raise ValueError("rendered preparation is not solver-checked")
    kind = manifest.get("receipt_kind", "full-workflow")
    if kind not in ("full-workflow", "helium-only", "recovery"):
        raise ValueError("rendered bundle has an unknown receipt kind")
    recovery = manifest.get("recovery")
    if (kind == "recovery") != isinstance(recovery, dict):
        raise ValueError("rendered recovery binding is inconsistent")
    if recovery:
        receipt = bundle / recovery.get("source_receipt", {}).get("path", "")
        if _sha(receipt) != recovery["source_receipt"].get("sha256"):
            raise ValueError("rendered recovery source receipt changed")
        _verify_recovery_source(Path(recovery.get("source_attempt", "")), receipt, bundle / "prepared")
    return {"status": "passed", "case_id": inspected["case_id"],
            "files": len(listed), "receipt_kind": kind,
            "site_runtime_verified": manifest["site_runtime_verified"]}


def render_slurm(prepared: Path, profile: dict, out: Path, *, helium_only: bool = False,
                 recovery_attempt: Path | None = None) -> Path:
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    if helium_only and recovery_attempt is not None:
        raise ValueError("helium_only and recovery_attempt are mutually exclusive")
    if prepared.is_file():
        prepared = prepared.parent
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"Slurm render output already exists: {out}")
    if out == prepared or out in prepared.parents or prepared in out.parents:
        raise ValueError("Slurm render output must be separate from preparation")
    if any(path.is_symlink() for path in prepared.rglob("*")):
        raise ValueError("prepared input contains a symlink")
    inspected = inspect_geometry(prepared)
    data = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    cad = data.get("science_inputs", {}).get("geometry", {}).get("adapter") == "cad"
    if cad:
        from dsmc_cad_workflow import validate_prepared_cad
        validate_prepared_cad(prepared)
    if inspected["state"] != "solver-checked" or (not cad and inspected["case_id"] not in (
            "b5-circle-s25-sampled-probe", "b5-slit-s25-sampled-probe",
            "b5-circle-s25-recovery-probe")):
        raise ValueError("Slurm renderer needs a supported checked workflow preparation")
    profile = validate_profile(profile)
    splitmap_solver = (_splitmap_solver(profile, required=not helium_only) if cad else None)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(prepared, out / "prepared", symlinks=False)
    recovery = None
    if recovery_attempt is not None:
        recovery_attempt = Path(recovery_attempt).resolve()
        source = recovery_attempt / "checkpoint-source.json"
        if source.is_symlink() or not source.is_file():
            raise ValueError("recovery attempt lacks a safe checkpoint-source.json")
        copied = out / "recovery-source.json"
        shutil.copyfile(source, copied)
        _verify_recovery_source(recovery_attempt, copied, out / "prepared")
        recovery = {"source_attempt": str(recovery_attempt),
                     "source_receipt": {"path": copied.name, "sha256": _sha(copied)}}
        if cad:
            recovery["resume"] = _cad_recovery_parameters(data)
        else:
            recovery["resume"] = {"seed": 20260926, "target_step": 4800}
    launch = {"schema_version": "dsmc-profile-v1", "profile_id": profile["site_id"] + "-launch",
              "platform": "local", "solver_executable": profile["solver_executable"],
              "mpi_argv": profile["mpi_argv"], "python": profile["python"],
              "julia": profile["julia"], "ranks": profile["ranks"],
              "threads": profile["threads"], "scratch_root": profile["scratch_root"],
              "results_root": profile["results_root"], "max_memory_gb": profile["memory_gb"],
              "max_disk_gb": profile["max_disk_gb"]}
    _json(out / "launch-profile.json", launch)
    _json(out / "site-profile.json", profile)
    repository = repository_identity(ROOT)
    if not isinstance(repository.get("commit"), str) or not repository["commit"]:
        raise ValueError("Slurm bundle needs a resolvable repository identity")
    (out / "job.sh").write_text(_script(profile, repository, helium_only=helium_only,
                                           recovery_attempt=recovery_attempt,
                                           recovery=(recovery or {}).get("resume"),
                                           splitmap_solver=splitmap_solver), encoding="utf-8", newline="\n")
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_symlink():
            raise ValueError("rendered preparation contains a symlink")
        if path.is_file():
            files.append({"path": path.relative_to(out).as_posix(),
                          "bytes": path.stat().st_size, "sha256": _sha(path)})
    manifest = {"schema_version": "dsmc-slurm-render-v1", "status": "rendered",
                "case_id": inspected["case_id"], "site_id": profile["site_id"],
                "site_runtime_verified": profile["site_runtime_verified"],
                "cluster_execution": "unverified",
                "receipt_kind": ("helium-only" if helium_only else "recovery" if recovery else "full-workflow"),
                "recovery": recovery, "splitmap_solver": splitmap_solver,
                "repository_identity": repository,
                "files": files}
    _json(out / "render-manifest.json", manifest)
    verify_render(out)
    return out / "job.sh"


def _profile_from_bundle(bundle: Path) -> dict:
    return validate_profile(json.loads((bundle / "site-profile.json").read_text(encoding="utf-8")))


def _verify_readiness(profile: dict) -> dict:
    """Require the site probe's binding before an irreversible scheduler call."""
    receipt = profile.get("readiness_receipt")
    if not receipt:
        raise ValueError("Slurm submission requires a readiness_receipt")
    try:
        from engaging.prepare_site import verify_readiness
    except ImportError as exc:
        raise RuntimeError("site readiness verifier is unavailable") from exc
    checked = verify_readiness(profile)
    if not isinstance(checked, dict) or checked.get("status") not in ("passed", "complete"):
        raise ValueError("site readiness receipt is not verified")
    return checked


def _read_receipt(receipt: Path) -> dict:
    return json.loads(Path(receipt).read_text(encoding="utf-8"))


def _bundle_identity(bundle: Path) -> dict:
    bundle = Path(bundle).resolve()
    checked = verify_render(bundle)
    manifest = bundle / "render-manifest.json"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    return {"path": str(bundle), "render_manifest_sha256": _sha(manifest),
            "case_id": checked["case_id"], "site_id": record["site_id"],
            "receipt_kind": checked["receipt_kind"], "repository_identity": record["repository_identity"]}


def _slurm_command(command: list[str], *, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)


def _query_command(command: list[str]) -> subprocess.CompletedProcess | None:
    try:
        return _slurm_command(command)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _job_id(value: str) -> str | None:
    match = re.fullmatch(r"\s*([0-9]+)(?:;[^\s;]+)?\s*", value)
    return match.group(1) if match else None


def submit_slurm(bundle: Path, receipt: Path) -> dict:
    """Submit exactly once, retaining a receipt before the scheduler can act."""
    bundle, receipt = Path(bundle).resolve(), Path(receipt).resolve()
    identity = _bundle_identity(bundle)
    profile = _profile_from_bundle(bundle)
    _verify_readiness(profile)
    token = uuid.uuid4().hex
    job_name = f"dsmc-{identity['site_id']}-{token}"
    started = _now()
    record = {
        "schema_version": "dsmc-slurm-attempt-v1", "status": "submitting",
        "created_utc": started, "submit_started_utc": started, "submit_user": (
            os.environ.get("USER") or os.environ.get("USERNAME")),
        "attempt_token": token, "job_name": job_name,
        "submit_command": ["sbatch", "--parsable", "--job-name", job_name,
                           str(bundle / "job.sh"), str(bundle)],
        "bundle": identity, "profile": {"path": str(bundle / "site-profile.json"),
                                             "sha256": _sha(bundle / "site-profile.json"),
                                             "results_root": profile["results_root"]},
        "scheduler": None,
    }
    _exclusive_json(receipt, record)
    try:
        completed = _slurm_command(record["submit_command"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        record.update({"status": "submission-unknown", "finished_utc": _now(),
                       "submission_error": str(exc)})
        _json(receipt, record)
        return record
    output = (completed.stdout or "").strip()
    job_id = _job_id(output) if completed.returncode == 0 else None
    if job_id:
        record.update({"status": "submitted", "finished_utc": _now(), "job_id": job_id,
                       "scheduler": {"job_id": job_id, "state": "SUBMITTED",
                                     "submission_stdout": output}})
    elif completed.returncode == 0:
        record.update({"status": "submission-unknown", "finished_utc": _now(),
                       "submission_error": "sbatch acknowledgement did not contain one job ID",
                       "submission_stdout": output})
    else:
        record.update({"status": "submission-rejected", "finished_utc": _now(),
                       "submission_exit_code": completed.returncode,
                       "submission_stdout": output, "submission_stderr": (completed.stderr or "").strip()})
    _json(receipt, record)
    return record


def _parse_pipe_records(text: str, fields: tuple[str, ...]) -> list[dict]:
    records = []
    for line in text.splitlines():
        values = line.strip().split("|")
        if len(values) != len(fields) or not any(values):
            continue
        records.append(dict(zip(fields, (value.strip() for value in values))))
    return records


def _state(value: object) -> str:
    return str(value or "UNKNOWN").upper().split()[0].rstrip("+")


def _failed_step(records: list[dict]) -> bool:
    good = {"COMPLETED", "PENDING", "CONFIGURING", "RUNNING", "COMPLETING"}
    for record in records:
        if not record.get("job_id") or "." not in record["job_id"]:
            continue
        state = _state(record.get("state"))
        exit_code = str(record.get("exit_code") or "")
        try:
            nonzero = any(int(part) != 0 for part in exit_code.split(":"))
        except ValueError:
            nonzero = bool(exit_code)
        if state not in good or nonzero:
            return True
    return False


def query_slurm(job_id: str) -> dict:
    """Collect both parent and step accounting, using queue state during lag."""
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("job_id must be numeric")
    fields = ("job_id", "job_name", "state", "exit_code")
    accounting = _query_command(["sacct", "-n", "-P", "-j", job_id,
                                 "--format=JobIDRaw%20,JobName%128,State%32,ExitCode%16"])
    records = _parse_pipe_records(accounting.stdout or "", fields) if accounting and accounting.returncode == 0 else []
    parent = next((record for record in records if record["job_id"] == job_id), None)
    queue = _query_command(["squeue", "-h", "-j", job_id, "-o", "%i|%j|%T"])
    queued = _parse_pipe_records(queue.stdout or "", ("job_id", "job_name", "state")) if queue and queue.returncode == 0 else []
    if parent is None:
        parent = next((record for record in queued if record["job_id"] == job_id), None)
    state = _state(parent.get("state")) if parent else "UNKNOWN"
    return {"job_id": job_id, "state": state,
            "exit_code": parent.get("exit_code") if parent else None,
            "job_name": parent.get("job_name") if parent else None,
            "steps": records, "accounting_available": bool(accounting and accounting.returncode == 0),
            "queue_available": bool(queue and queue.returncode == 0), "queue_records": queued,
            "step_failure": _failed_step(records)}


def _scheduler_time(value: object) -> datetime | None:
    """Interpret Slurm's local Submit time as an instant, rejecting placeholders."""
    text = str(value or "").strip()
    if not text or text.upper() in {"UNKNOWN", "N/A", "NONE"}:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return stamp.astimezone(timezone.utc)


def _find_attempts(record: dict) -> list[dict]:
    """Find only exact-name, same-user parent records after a lost acknowledgement."""
    job_name = record["job_name"]
    fields = ("job_id", "job_name", "user", "state", "exit_code", "submit")
    started = _scheduler_time(record["submit_started_utc"])
    if started is None:
        raise ValueError("attempt receipt has no valid submit start time")
    # sacct reads local host time. Query a whole-second padded window, then compare
    # the returned Submit instant against UTC bounds before accepting an identity.
    lower = started - timedelta(minutes=1)
    upper = datetime.now(timezone.utc) + timedelta(minutes=1)
    since = lower.astimezone().replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%S")
    accounting = _query_command(["sacct", "-n", "-P", "-X", "--starttime", since,
                                  "--name", job_name,
                                  "--format=JobIDRaw%20,JobName%128,User%64,State%32,ExitCode%16,Submit%32"])
    candidates = _parse_pipe_records(accounting.stdout or "", fields) if accounting and accounting.returncode == 0 else []
    queue = _query_command(["squeue", "-h", "-n", job_name, "-o", "%i|%j|%u|%V|%T"])
    candidates.extend(_parse_pipe_records(queue.stdout or "", ("job_id", "job_name", "user", "submit", "state"))
                      if queue and queue.returncode == 0 else [])
    user = record.get("submit_user")
    by_id = {}
    for candidate in candidates:
        if not candidate.get("job_id", "").isdigit():
            continue
        by_id[candidate["job_id"]] = {**by_id.get(candidate["job_id"], {}), **candidate}
    return [candidate for candidate in by_id.values()
            if candidate.get("job_name") == job_name and
            (not user or candidate.get("user") == user) and
            (submitted := _scheduler_time(candidate.get("submit"))) is not None and
            lower <= submitted <= upper]


def _receipt_status(receipt: Path) -> dict:
    record = _read_receipt(receipt)
    if record.get("schema_version") != "dsmc-slurm-attempt-v1":
        raise ValueError("unsupported Slurm attempt receipt")
    if record["status"] in ("submitting", "submission-unknown") and not record.get("job_id"):
        matches = _find_attempts(record)
        record["reconciliation"] = {"checked_utc": _now(), "matches": matches}
        if len(matches) == 1:
            record["job_id"] = matches[0]["job_id"]
            record["status"] = "submitted"
        else:
            record["status"] = "submission-unknown"
            _json(receipt, record)
            return {"status": "submission-unknown", "matches": len(matches),
                    "job_name": record["job_name"]}
    if not record.get("job_id"):
        return {"status": record["status"], "job_name": record["job_name"]}
    scheduler = query_slurm(record["job_id"])
    archive = Path(record["profile"]["results_root"]) / f"job-{record['job_id']}"
    if record.get("bundle", {}).get("receipt_kind") == "helium-only":
        result = _reconcile_helium_only(archive.with_name(archive.name + ".helium-only.json"), scheduler)
    else:
        result = reconcile_status(archive, scheduler)
    record.update({"status": result["status"], "scheduler": scheduler,
                   "status_checked_utc": _now(), "archive_dir": str(archive)})
    _json(receipt, record)
    return result


def status_slurm(receipt: Path) -> dict:
    """Reconcile an attempt receipt, or its enclosing nonblocking acceptance receipt."""
    receipt = Path(receipt).resolve()
    record = _read_receipt(receipt)
    if record.get("schema_version") == "dsmc-3d-slurm-acceptance-v1":
        attempt = receipt.parent / record["submission_receipt"]["path"]
        result = _receipt_status(attempt)
        record["scheduler"] = result
        record["status"] = "complete" if result.get("status") == "complete" else result.get("status")
        record["status_checked_utc"] = _now()
        _json(receipt, record)
        return result
    return _receipt_status(receipt)


def reconcile_status(run_dir: Path, scheduler_record: dict) -> dict:
    state = _state(scheduler_record.get("state"))
    job_id = scheduler_record.get("job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("scheduler record needs a numeric job_id")
    if state in ("PENDING", "CONFIGURING"):
        status = "queued"
    elif state in ("RUNNING", "COMPLETING"):
        status = "running"
    elif state in ("CANCELLED", "TIMEOUT", "PREEMPTED", "NODE_FAIL"):
        status = "interrupted"
    elif scheduler_record.get("step_failure"):
        status = "failed"
    elif state == "COMPLETED" and str(scheduler_record.get("exit_code")) in ("0", "0:0"):
        try:
            archive = verify_archive(run_dir)
            status = "complete"
        except (OSError, ValueError, KeyError):
            status = "failed-stageout"
    elif state in ("COMPLETED", "FAILED", "OUT_OF_MEMORY", "BOOT_FAIL"):
        status = "failed"
    else:
        status = "unknown"
    failed_stage = None
    if status == "failed":
        failure = Path(run_dir).with_name(Path(run_dir).name + ".failed") / "failure.txt"
        if failure.is_file() and not failure.is_symlink():
            values = dict(line.split("=", 1) for line in failure.read_text(encoding="utf-8").splitlines() if "=" in line)
            if values.get("schema_version") == "dsmc-slurm-failure-v1" and values.get("job_id") == job_id:
                failed_stage = values.get("stage")
                if failed_stage == "archive":
                    status = "failed-stageout"
    return {"job_id": job_id, "array_job_id": scheduler_record.get("array_job_id"),
            "array_task_id": scheduler_record.get("array_task_id"),
            "scheduler_state": state, "status": status,
            "step_failure": bool(scheduler_record.get("step_failure")), "failed_stage": failed_stage,
            "science_assessment": (archive["source_verification"].get("science_assessment")
                                   if status == "complete" else None)}


def _reconcile_helium_only(receipt: Path, scheduler_record: dict) -> dict:
    """Represent a retained helium stage without passing it as a full archive."""
    state = _state(scheduler_record.get("state")); job_id = scheduler_record.get("job_id")
    if not isinstance(job_id, str) or not job_id.isdigit():
        raise ValueError("scheduler record needs a numeric job_id")
    if state in ("PENDING", "CONFIGURING"):
        status = "queued"
    elif state in ("RUNNING", "COMPLETING"):
        status = "running"
    elif state in ("CANCELLED", "TIMEOUT", "PREEMPTED", "NODE_FAIL"):
        status = "interrupted"
    elif scheduler_record.get("step_failure") or state in ("FAILED", "OUT_OF_MEMORY", "BOOT_FAIL"):
        status = "failed"
    elif state == "COMPLETED" and str(scheduler_record.get("exit_code")) in ("0", "0:0"):
        try:
            record = json.loads(Path(receipt).read_text(encoding="utf-8"))
            status = "helium-only-complete" if record.get("status") == "complete" else "failed-stageout"
        except (OSError, ValueError, KeyError):
            status = "failed-stageout"
    else:
        status = "unknown"
    return {"job_id": job_id, "array_job_id": scheduler_record.get("array_job_id"),
            "array_task_id": scheduler_record.get("array_task_id"), "scheduler_state": state,
            "status": status, "step_failure": bool(scheduler_record.get("step_failure")),
            "science_assessment": None, "receipt_kind": "helium-only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--bundle", type=Path, required=True)
    recovery = commands.add_parser("verify-recovery")
    recovery.add_argument("--source-attempt", type=Path, required=True)
    recovery.add_argument("--receipt", type=Path, required=True)
    recovery.add_argument("--prepared", type=Path, required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("bundle", type=Path)
    submit.add_argument("--receipt", type=Path, required=True)
    status = commands.add_parser("status")
    status.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_render(args.bundle), indent=2))
    elif args.command == "verify-recovery":
        print(json.dumps(_verify_recovery_source(args.source_attempt, args.receipt, args.prepared), indent=2))
    elif args.command == "submit":
        print(json.dumps(submit_slurm(args.bundle, args.receipt), indent=2))
    elif args.command == "status":
        print(json.dumps(status_slurm(args.receipt), indent=2))


if __name__ == "__main__":
    main()
