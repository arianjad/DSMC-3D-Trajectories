#!/usr/bin/env python3
"""Synchronous local helium attempt from an immutable prepared 3D case."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dsmc_case import inspect_geometry, load_profile
from dsmc_repository import repository_provenance


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, data):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _executable(value):
    path = shutil.which(value) if not Path(value).is_absolute() else value
    if not path or not Path(path).is_file():
        raise ValueError(f"solver executable not found: {value}")
    if os.name != "nt" and not os.access(path, os.X_OK):
        raise ValueError(f"solver executable is not executable: {path}")
    return str(Path(path).resolve())


def render_launch(profile: dict, executable: str, solver_args: list[str]) -> list[str]:
    """Render one launcher argv; no shell parsing and no implicit nested MPI."""
    substitutions = {"{ranks}": str(profile["ranks"]), "{threads}": str(profile["threads"])}
    prefix = []
    for token in profile["mpi_argv"]:
        for placeholder, value in substitutions.items():
            token = token.replace(placeholder, value)
        if "{" in token or "}" in token:
            raise ValueError(f"profile.mpi_argv: unsupported placeholder in {token!r}")
        prefix.append(token)
    return [*prefix, executable, *solver_args]


def _variables(science):
    h, n = science["helium"], science["numerics"]
    return {"MDOT": h["inlet"]["mass_flow_kg_s"], "FILLN": h["fill"]["number_density_m3"],
            "FNUM": n["fnum"], "DT": n["dt_s"], "STEPS": n["total_steps"],
            "SEED": n["solver_seed"]}


def _stage_inputs(prepared_dir, run_dir, prepared):
    records = []
    for item in prepared["inputs"]:
        source = (prepared_dir / item["staged"]["path"]).resolve()
        if source.parent != prepared_dir or not source.is_file():
            raise ValueError(f"prepared input path is invalid: {item['staged']['path']}")
        if _sha(source) != item["staged"]["sha256"]:
            raise ValueError(f"prepared input changed: {source}")
        dest = run_dir / source.name
        if dest.exists():
            raise ValueError(f"staged basenames collide: {dest.name}")
        mode = item.get("transfer_mode")
        if mode == "text-lf":
            content = source.read_bytes()
            dest.write_bytes(content.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
        elif mode == "binary":
            shutil.copyfile(source, dest)
        else:
            raise ValueError(f"unsupported staged transfer mode: {mode}")
        records.append({"role": item["role"], "transfer_mode": mode,
                        "prepared_path": item["staged"]["path"], "prepared_sha256": item["staged"]["sha256"],
                        "staged_path": dest.name, "staged_sha256": _sha(dest),
                        "bytes": dest.stat().st_size})
    return records


def _checkpoint_context(prepared_dir, run_dir):
    """Freeze preparation, geometry, roles, and generated inputs before launch."""
    context = run_dir / "checkpoint-context"
    context.mkdir()
    records = []
    for source in sorted(Path(prepared_dir).iterdir()):
        if source.is_symlink() or not source.is_file():
            continue
        target = context / source.name
        shutil.copyfile(source, target)
        if _sha(source) != _sha(target):
            raise ValueError(f"checkpoint context copy changed: {source.name}")
        records.append({"path": f"checkpoint-context/{source.name}", "bytes": target.stat().st_size,
                        "sha256": _sha(target)})
    required = {"prepared.json", "geometry.json"}
    if required - {Path(item["path"]).name for item in records}:
        raise ValueError("checkpoint context lacks prepared.json or geometry.json")
    return records


def _preflight_inputs(prepared):
    names = [Path(item["staged"]["path"]).name for item in prepared["inputs"]]
    if len(names) != len(set(names)):
        raise ValueError("prepared inputs have colliding staged basenames")
    for item in prepared["inputs"]:
        if "transfer_mode" not in item:
            raise ValueError("prepared input lacks transfer_mode; regenerate the preparation from its case")
        if item["transfer_mode"] not in ("text-lf", "binary"):
            raise ValueError(f"unsupported staged transfer mode: {item.get('transfer_mode')}")
    for item in prepared["declared_outputs"]:
        if "required" not in item or not isinstance(item["required"], bool):
            raise ValueError("prepared output lacks required flag; regenerate the preparation from its case")


def _outputs(run_dir, staged_names):
    result = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(run_dir).as_posix()
        if rel in staged_names or rel in ("manifest.json", "manifest.json.tmp"):
            continue
        result.append({"path": rel, "bytes": path.stat().st_size, "sha256": _sha(path)})
    return result


def _missing_outputs(run_dir, prepared):
    missing = []
    science = prepared["science_inputs"]
    n = science["numerics"]
    for step in range(n["field_every_steps"], n["total_steps"] + 1, n["field_every_steps"]):
        field = run_dir / f"fields.{step}.gz"
        if not field.is_file() or field.stat().st_size == 0:
            missing.append(field.name)
    for item in prepared["declared_outputs"]:
        if item["required"] and not any(p.is_file() and p.stat().st_size > 0 for p in run_dir.glob(item["pattern"])):
            missing.append(item["pattern"])
    return sorted(set(missing))


def _repo_identity():
    root = Path(__file__).resolve().parents[1]
    return {"root": str(root), **repository_provenance(root)}


def run_helium(prepared_path: Path, profile_path: Path, run_dir: Path) -> int:
    prepared_dir = Path(prepared_path).resolve()
    if prepared_dir.is_file():
        prepared_dir = prepared_dir.parent
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    inspect_geometry(prepared_dir)
    prepared = json.loads((prepared_dir / "prepared.json").read_text(encoding="utf-8"))
    _preflight_inputs(prepared)
    profile = load_profile(profile_path)
    if profile["platform"] != "local":
        raise ValueError("run_helium supports local profiles; Slurm uses Task 7 submission")
    executable = _executable(profile["solver_executable"])
    if prepared["science_inputs"]["geometry"]["adapter"] == "cad":
        from dsmc_cad_workflow import validate_prepared_cad
        validate_prepared_cad(prepared_dir)
        if _sha(executable) != prepared["solver_check"]["solver"]["sha256"]:
            raise ValueError("CAD solver differs from its canonical geometry check")
    variables = _variables(prepared["science_inputs"])
    arguments = ["-in", "in.helium"]
    for name, value in variables.items():
        arguments += ["-var", name, str(value)]
    argv = render_launch(profile, executable, arguments)
    if profile["mpi_argv"] and not shutil.which(argv[0]) and not Path(argv[0]).is_file():
        raise ValueError(f"launcher executable not found: {argv[0]}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(run_dir.parent).free < 512 * 1024 * 1024:
        raise ValueError(f"less than 512 MiB free at run destination: {run_dir.parent}")
    run_dir.mkdir()
    try:
        records = _stage_inputs(prepared_dir, run_dir, prepared)
        context = _checkpoint_context(prepared_dir, run_dir)
    except Exception as exc:
        _write_json(run_dir / "manifest.json", {
            "schema_version": "dsmc-helium-attempt-v1", "case_id": prepared["case_id"],
            "status": "failed", "stage": "staging", "created_utc": _now(),
            "finished_utc": _now(), "failure": f"input staging failed: {exc}",
            "profile": {"profile_id": profile["profile_id"], "source": profile["source_profile"]},
            "outputs": _outputs(run_dir, set())})
        return 8
    manifest = {
        "schema_version": "dsmc-helium-attempt-v1", "case_id": prepared["case_id"],
        "status": "running", "stage": "helium", "created_utc": _now(), "started_utc": None,
        "finished_utc": None, "exit_code": None, "failure": None,
        "prepared": {"path": str(prepared_dir), "sha256": _sha(prepared_dir / "prepared.json")},
        "profile": {"profile_id": profile["profile_id"], "source": profile["source_profile"],
                    "ranks": profile["ranks"], "threads": profile["threads"]},
        "repository": _repo_identity(),
        "solver": {"path": executable, "sha256": _sha(executable)},
        "command": argv, "working_directory": str(run_dir),
        "runtime": {"platform": platform.platform(), "threads_per_rank": profile["threads"]},
        "inputs": records, "outputs": [],
    }
    _write_json(run_dir / "manifest.json", manifest)
    # This immutable launch context is deliberately independent of a terminal
    # helium manifest. A later allocated publisher appends only observed restart
    # generations and its native read-restart probe; it must not infer them.
    _write_json(run_dir / "checkpoint-source.json", {
        "schema_version": "dsmc-checkpoint-source-v1",
        "attempt_identity": {"manifest": "manifest.json", "sha256": _sha(run_dir / "manifest.json"),
                             "status_at_launch": "running"},
        "prepared": manifest["prepared"], "prepared_record": prepared,
        "preparation_context": context,
        "active_inputs": [{"path": item["staged_path"], "bytes": item["bytes"], "sha256": item["staged_sha256"],
                           "role": item["role"]}
                          for item in records],
        "solver": manifest["solver"],
        "producer": {"ranks": profile["ranks"], "one_shard_per_rank": True,
                     "restart_prefix": prepared["science_inputs"]["numerics"].get("restart_prefix", "b5.restart"), "restart_command": next(
                         (" ".join(line.split()) for line in (run_dir / "in.helium").read_text(encoding="utf-8").splitlines()
                          if line.strip().startswith("restart ")), None)},
    })
    old_handler = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    child = None
    try:
        with (run_dir / "run.log").open("wb") as log:
            manifest["started_utc"] = _now()
            _write_json(run_dir / "manifest.json", manifest)
            env = os.environ.copy()
            env.update({key: str(profile["threads"]) for key in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
            child = subprocess.Popen(argv, cwd=run_dir, stdout=log, stderr=subprocess.STDOUT, env=env)
            code = child.wait()
        (run_dir / "rc.sentinel").write_text(str(code) + "\n", encoding="ascii")
        missing = _missing_outputs(run_dir, prepared) if code == 0 else []
        manifest["exit_code"] = code
        manifest["status"] = "complete" if code == 0 and not missing else "failed"
        if code != 0:
            manifest["failure"] = f"solver exited with status {code}"
        elif missing:
            manifest["failure"] = "missing required outputs: " + ", ".join(missing)
    except KeyboardInterrupt:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        manifest["status"] = "interrupted"
        manifest["failure"] = "local runner interrupted; reconcile outputs before reuse"
        manifest["exit_code"] = 130
        (run_dir / "rc.sentinel").write_text("130\n", encoding="ascii")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"runner error: {exc}"
        manifest["exit_code"] = 8
        (run_dir / "rc.sentinel").write_text("8\n", encoding="ascii")
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        manifest["finished_utc"] = _now()
        manifest["outputs"] = _outputs(run_dir, {x["staged_path"] for x in records})
        _write_json(run_dir / "manifest.json", manifest)
    return 0 if manifest["status"] == "complete" else (manifest["exit_code"] or 8)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        code = run_helium(args.prepared, args.profile, args.run_dir)
        print(f"RUN_DIR={args.run_dir.resolve()} STATUS={'complete' if code == 0 else 'failed'} RC={code}")
        return code
    except (ValueError, FileExistsError, OSError) as exc:
        print(f"run_helium: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
