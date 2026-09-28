#!/usr/bin/env python3
"""Probe the staged 3D runtime inside an allocation and bind a concrete profile."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from dsmc_repository import repository_identity, verify_repository, sha


def write(path: Path, record: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def executable(value: str) -> Path:
    path = shutil.which(value)
    if path is None:
        raise ValueError(f"missing executable: {value}")
    return Path(path).resolve()


def selected_python(value: str) -> Path:
    """Return the profile-selected Python without collapsing a venv symlink."""
    path = shutil.which(value)
    if path is None:
        raise ValueError(f"missing executable: {value}")
    return Path(path).absolute()


def runtime_requirements() -> dict[str, tuple[str, str]]:
    requirements = {}
    for line in (ROOT / "requirements-3d.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()  # pip comment rule: '#' starts a comment
        if not line:
            continue
        match = re.fullmatch(r"([a-zA-Z0-9_-]+)>=(\d+(?:\.\d+)*),<(\d+(?:\.\d+)*)", line)
        if not match:
            raise ValueError(f"unrecognized runtime requirement: {line}")
        name, low, high = match.groups()
        requirements[name] = (low, high)
    return requirements


def version_numbers(value: str) -> tuple[int, ...]:
    match = re.match(r"\d+(?:\.\d+)*", value)
    if match is None:
        raise ValueError(f"unrecognized package version: {value}")
    return tuple(map(int, match.group(0).split(".")))


def python_runtime_identity(profile_python: Path, requirements: dict[str, tuple[str, str]]) -> dict:
    """Query the selected interpreter, keeping its venv identity separate from its binary."""
    code = """
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

required = json.loads(sys.argv[1])
packages = {}
for name in required:
    module = importlib.import_module(name.replace("-", "_"))
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        raise RuntimeError(f"package has no module path: {name}")
    packages[name] = {
        "version": importlib.metadata.version(name),
        "module_path": str(Path(module_path).absolute()),
    }
print(json.dumps({
    "invoked_path": str(Path(sys.executable).absolute()),
    "prefix": str(Path(sys.prefix).absolute()),
    "base_prefix": str(Path(sys.base_prefix).absolute()),
    "packages": packages,
}, sort_keys=True))
"""
    result = subprocess.run(
        [str(profile_python), "-c", code, json.dumps(sorted(requirements))],
        capture_output=True, text=True, check=False, timeout=30)
    if result.returncode:
        raise ValueError(f"profile Python runtime probe failed: {result.stderr.strip()}")
    try:
        identity = json.loads(result.stdout)
        packages = identity["packages"]
        for name, (low, high) in requirements.items():
            version = packages[name]["version"]
            if not version_numbers(low) <= version_numbers(version) < version_numbers(high):
                raise ValueError(f"unsatisfied runtime requirement: {name}>={low},<{high}, found {version}")
            if not isinstance(packages[name]["module_path"], str):
                raise ValueError(f"package has no module path: {name}")
        identity["profile_path"] = str(profile_python)
        return identity
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("profile Python runtime probe returned an invalid identity") from exc


def verify_python_runtime(profile: dict, expected: dict) -> None:
    """Verify a current profile Python against a post-probe venv/package identity."""
    try:
        packages = expected["packages"]
        requirements = runtime_requirements()
        if set(packages) != set(requirements):
            raise ValueError("readiness receipt has invalid Python runtime identity")
        actual = python_runtime_identity(selected_python(profile["python"]), requirements)
    except (KeyError, TypeError) as exc:
        raise ValueError("readiness receipt has invalid Python runtime identity") from exc
    if actual != expected:
        raise ValueError("site Python runtime identity changed")


def artifact(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha(path)}


def splitmap_artifact(profile: dict) -> dict | None:
    """Check the optional diagnostic runtime at the exact declared site path."""
    declared = profile.get("splitmap_solver")
    if declared is None:
        return None
    path = Path(declared["path"])
    if path.is_symlink() or not path.is_file() or sha(path) != declared["sha256"]:
        raise ValueError("site split-map solver changed")
    return dict(declared)


def verify_readiness(profile: dict) -> dict:
    if not profile.get("site_runtime_verified") or not profile.get("readiness_receipt"):
        raise ValueError("submission requires a verified site-readiness receipt")
    receipt = json.loads(Path(profile["readiness_receipt"]).read_text(encoding="utf-8"))
    effective = {key: value for key, value in profile.items() if key != "readiness_receipt"}
    schema = receipt.get("schema_version")
    if (schema not in {"dsmc-site-readiness-v1", "dsmc-site-readiness-v2"} or
            receipt.get("status") != "complete" or receipt.get("verified_profile") != effective):
        raise ValueError("readiness receipt does not bind this profile")
    if Path(profile["repository_root"]).resolve() != ROOT:
        raise ValueError("site receipt must use the currently staged repository")
    verify_repository(ROOT, receipt["repository"])
    for name, item in receipt["artifacts"].items():
        if name == "splitmap_solver":
            continue
        if sha(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"site runtime artifact changed: {item['path']}")
    if schema == "dsmc-site-readiness-v2":
        if "python_runtime" not in receipt:
            raise ValueError("readiness receipt lacks Python runtime identity")
        verify_python_runtime(profile, receipt["python_runtime"])
    elif "python_runtime" in receipt:
        verify_python_runtime(profile, receipt["python_runtime"])
    if str(executable(profile["python"])) != receipt["executables"]["python"]:
        # Retain the binary-resolution check alongside v2's venv/package identity;
        # v1 receipts have only this weaker evidence for compatibility.
        raise ValueError("site executable resolution changed: python")
    for key in ("julia", "mpi"):
        requested = profile["mpi_argv"][0] if key == "mpi" else profile[key]
        if str(executable(requested)) != receipt["executables"][key]:
            raise ValueError(f"site executable resolution changed: {key}")
    # /home and /orcd/home can be separate mounts of the same shared payload.
    # The requested path is already bound by verified_profile; resolve() and
    # samefile() are not portable identities across those mounts. Verify the
    # binary reached through that exact requested path as well as the probe path.
    if sha(Path(profile["solver_executable"])) != receipt["artifacts"]["solver"]["sha256"]:
        raise ValueError("site solver changed")
    declared_splitmap = splitmap_artifact(profile)
    if declared_splitmap is not None:
        if receipt["artifacts"].get("splitmap_solver") != declared_splitmap:
            raise ValueError("site readiness receipt does not bind the split-map solver")
    return receipt


def probe(profile_path: Path, out: Path) -> Path:
    from dsmc_slurm import validate_profile
    profile = validate_profile(json.loads(profile_path.read_text(encoding="utf-8")))
    if not os.environ.get("SLURM_JOB_ID", "").isdigit():
        raise ValueError("site probe must run inside a Slurm allocation")
    if os.environ.get("LD_PRELOAD"):
        raise ValueError("site probe requires the native runtime without LD_PRELOAD")
    if Path(profile["repository_root"]).resolve() != ROOT:
        raise ValueError("profile repository differs from this staged source")
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": "dsmc-site-readiness-v2", "status": "running",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "job_id": os.environ["SLURM_JOB_ID"], "host": platform.node(),
              "repository": repository_identity(ROOT), "commands": [], "artifacts": {}}
    target = out / "site-readiness.json"
    write(target, record)

    def command(argv: list[str], name: str, timeout: int = 300) -> None:
        started = time.monotonic()
        with (out / (name + ".log")).open("w") as stream:
            result = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        record["commands"].append({"name": name, "argv": argv, "exit_code": result.returncode,
                                    "wall_seconds": time.monotonic() - started,
                                    "log": artifact(out / (name + ".log"))})
        write(target, record)
        if result.returncode:
            raise RuntimeError(f"{name} exited {result.returncode}")

    try:
        verify_repository(ROOT, record["repository"])
        profile_python = selected_python(profile["python"])
        record["executables"] = {key: str(executable(value)) for key, value in
            (("python", profile["python"]), ("julia", profile["julia"]), ("mpi", profile["mpi_argv"][0]))}
        if Path(sys.executable).absolute() != profile_python:
            raise ValueError("probe Python differs from profile Python")
        record["python_runtime"] = python_runtime_identity(profile_python, runtime_requirements())
        record["python_packages"] = {name: item["version"]
                                     for name, item in record["python_runtime"]["packages"].items()}
        record["artifacts"] = {key: artifact(path) for key, path in {
            "setup": Path(profile["setup_script"]), "solver": Path(profile["solver_executable"]),
            "python": Path(record["executables"]["python"]), "julia": Path(record["executables"]["julia"]),
            "mpi": Path(record["executables"]["mpi"]),
            "julia_project": ROOT / "tracer/Project.toml", "julia_manifest": ROOT / "tracer/Manifest.toml",
            "requirements": ROOT / "requirements-3d.txt"}.items()}
        checked_splitmap = splitmap_artifact(profile)
        if checked_splitmap is not None:
            record["artifacts"]["splitmap_solver"] = checked_splitmap
        record["storage"] = {}
        for key in ("scratch_root", "results_root"):
            folder = Path(profile[key]).resolve()
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=folder) as stream:
                stream.write(b"dsmc-site-probe\n"); stream.flush(); os.fsync(stream.fileno())
                assert Path(stream.name).read_bytes() == b"dsmc-site-probe\n"
            free = shutil.disk_usage(folder).free
            if free < profile["max_disk_gb"] * 1024**3:
                raise ValueError(f"insufficient free working space at {folder}")
            record["storage"][key] = {"path": str(folder), "free_bytes": free}
        command([profile["julia"], "--version"], "julia-version")
        command([profile["mpi_argv"][0], "--version"], "mpi-version")
        command([sys.executable, str(ROOT / "tools/verify_3d_fixture.py"), "--out", str(out / "fixture")], "julia-fixture")
        deck = out / "in.startup"
        deck.write_text("\n".join([
            "seed 20260924", "dimension 3", "global gridcut 0.0 nrho 1e20 fnum 1e8",
            "boundary p p p", "create_box 0 0.001 0 0.001 0 0.001", "create_grid 2 2 2",
            "balance_grid rcb cell", f"species {ROOT / 'cases/data/he.species'} He",
            "mixture hegas He temp 4.0", f"collide vss hegas {ROOT / 'cases/data/he.vss'}",
            "create_particles hegas n 1000", "timestep 1e-8", "stats 1", "stats_style step np", "run 2", ""]),
            encoding="utf-8", newline="\n")
        from dsmc_run import render_launch
        command(render_launch(profile, profile["solver_executable"],
                              ["-in", str(deck), "-log", str(out / "log.sparta")]), "mpi-startup")
        text = (out / "mpi-startup.log").read_text()
        if not re.search(r"Loop time of .* on " + str(profile["ranks"]) + r" procs for 2 steps", text):
            raise ValueError("native MPI startup did not execute the requested ranks and steps")
        profile["site_runtime_verified"] = True
        profile.pop("readiness_receipt", None)
        record["verified_profile"] = dict(profile)
        record["status"] = "complete"
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(target, record)
        profile["readiness_receipt"] = str(target)
        write(out / "verified-profile.json", profile)
        verify_readiness(profile)
    except BaseException as exc:
        record["status"] = "failed"
        record["failure"] = str(exc)
        write(target, record)
        raise
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    check = commands.add_parser("probe")
    check.add_argument("--profile", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "probe":
        print(probe(args.profile, args.out))
    else:
        receipt = verify_readiness(json.loads(args.profile.read_text(encoding="utf-8")))
        print(json.dumps({"status": receipt["status"], "job_id": receipt["job_id"]}))
