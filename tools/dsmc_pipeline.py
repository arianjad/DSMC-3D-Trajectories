"""Compose a checked helium attempt, bound field and molecule child."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from dsmc_case import inspect_geometry, load_profile
from dsmc_field import FIELD_POLICIES, prepare_field
from dsmc_molecule import run_molecules
from dsmc_run import run_helium
from dsmc_repository import repository_provenance


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_identity() -> dict:
    return repository_provenance(ROOT)


def _receipt_ref(path: Path) -> dict:
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _check_host_dependencies() -> None:
    try:
        import pandas  # noqa: F401 - required by field2tracer3d.convert3d
    except ImportError as exc:
        raise ValueError("the invoking Python needs pandas for field conversion; "
                         "select an existing environment with pandas before launching helium") from exc
    if not shutil.which("julia"):
        raise ValueError("Julia is not on PATH for the molecule stage")


def _wsl_path(path: Path) -> str:
    completed = subprocess.run(
        ["wsl.exe", "-e", "wslpath", "-a", "-u", str(path)],
        capture_output=True, text=True, check=False)
    if completed.returncode or not completed.stdout.strip():
        raise ValueError(f"cannot map Windows path into WSL: {path}")
    return completed.stdout.strip()


def _run_wsl_helium(prepared: Path, profile: Path, parent: Path,
                    python_executable: str) -> int:
    if os.name != "nt" or not shutil.which("wsl.exe"):
        raise ValueError("WSL helium backend requires wsl.exe on Windows")
    command = ["wsl.exe", "-e", python_executable,
               _wsl_path(ROOT / "tools" / "dsmc_run.py"),
               "--prepared", _wsl_path(prepared),
               "--profile", _wsl_path(profile),
               "--run-dir", _wsl_path(parent)]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def run_pipeline(prepared: Path, profile: Path, out: Path, *, gas_policy: str,
                 helium_backend: str = "native", splitmap: Path | None = None,
                 splitmap_ids: Path | None = None, splitmap_solver: str | None = None) -> Path:
    """Run fresh helium through a separate selected-field and molecule child."""
    prepared_dir = Path(prepared).resolve()
    if prepared_dir.is_file():
        if prepared_dir.name != "prepared.json":
            raise ValueError("run needs a prepared directory or prepared.json")
        prepared_dir = prepared_dir.parent
    profile_path = Path(profile).resolve()
    out = Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"workflow output already exists: {out}")
    if (out == prepared_dir or out in prepared_dir.parents or
            prepared_dir in out.parents):
        raise ValueError("workflow output must be separate from prepared input")
    if not profile_path.is_file() or out == profile_path or out in profile_path.parents:
        raise ValueError("profile must exist outside workflow output")
    inspected = inspect_geometry(prepared_dir)
    check = inspected["solver_check"]
    if (inspected["state"] != "solver-checked" or
            not isinstance(check, dict) or check.get("status") != "complete"):
        raise ValueError("full workflow requires a completed solver-checked preparation")
    prepared_data = json.loads((prepared_dir / "prepared.json").read_text(encoding="utf-8"))
    adapter = prepared_data["science_inputs"]["geometry"]["adapter"]
    if adapter == "cad":
        from dsmc_cad_workflow import validate_prepared_cad
        validate_prepared_cad(prepared_dir)
    elif adapter not in ("b5-circle", "b5-slit"):
        raise ValueError("full local workflow needs a supported checked geometry adapter")
    split_inputs = _split_inputs(splitmap, splitmap_ids)
    _check_splitmap_options(split_inputs, splitmap_solver, helium_backend, adapter)
    if gas_policy not in FIELD_POLICIES:
        raise ValueError(f"unsupported field reconstruction policy: {gas_policy}")
    if helium_backend not in ("native", "wsl"):
        raise ValueError(f"unsupported helium backend: {helium_backend}")
    if helium_backend == "wsl" and (os.name != "nt" or not shutil.which("wsl.exe")):
        raise ValueError("WSL helium backend requires wsl.exe on Windows")
    launch = load_profile(profile_path)
    if launch["platform"] != "local":
        raise ValueError("full workflow needs a local profile; cluster submission is separate")
    # A POSIX interpreter path belongs to the guest. Windows Path.resolve in
    # load_profile must not turn /home/.../python into a host C:\home path.
    helium_python = (json.loads(profile_path.read_text(encoding="utf-8"))["python"]
                     if helium_backend == "wsl" else launch.get("python"))
    _check_host_dependencies()
    numerics = prepared_data["science_inputs"]["numerics"]
    final_step, field_every = numerics["total_steps"], numerics["field_every_steps"]
    if (type(final_step) is not int or final_step <= 0 or
            type(field_every) is not int or field_every <= 0 or
            final_step % field_every):
        raise ValueError("case has no scheduled final saved helium frame")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    receipt_path = out / "workflow.json"
    receipt = {
        "schema_version": "dsmc-local-workflow-v1", "status": "started",
        "created_utc": _now(), "case_id": prepared_data["case_id"],
        "repository": _repo_identity(),
        "prepared": _receipt_ref(prepared_dir / "prepared.json"),
        "profile": _receipt_ref(profile_path), "gas_policy": gas_policy,
        "helium_backend": helium_backend,
        "frame_selection": {"timestep": final_step},
        "children": {}, "phases": {},
    }
    _write_json(receipt_path, receipt)
    stage = "helium"
    stage_start = time.perf_counter()
    stage_utc = _now()

    def finish_phase(name: str, status: str) -> None:
        receipt["phases"][name] = {
            "status": status, "started_utc": stage_utc,
            "finished_utc": _now(), "duration_s": time.perf_counter() - stage_start,
        }

    try:
        parent = out / "helium"
        code = (run_helium(prepared_dir, profile_path, parent)
                if helium_backend == "native" else
                _run_wsl_helium(prepared_dir, profile_path, parent, helium_python))
        receipt["children"]["helium"] = _receipt_ref(parent / "manifest.json")
        receipt["helium_exit_code"] = code
        finish_phase(stage, "complete" if code == 0 else "failed")
        _write_json(receipt_path, receipt)
        if code != 0:
            raise ValueError(f"fresh helium attempt failed with exit code {code}")

    except BaseException as exc:
        if stage not in receipt["phases"]:
            finish_phase(stage, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        receipt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        receipt["failed_stage"] = stage
        receipt["failure"] = str(exc)
        receipt["finished_utc"] = _now()
        _write_json(receipt_path, receipt)
        raise
    return _finish_stages(prepared_dir, parent, out, receipt, gas_policy, **split_inputs,
                          splitmap_solver=splitmap_solver, splitmap_backend=helium_backend,
                          splitmap_python=helium_python or "python3")


def _split_inputs(splitmap: Path | None, splitmap_ids: Path | None) -> dict:
    if (splitmap is None) != (splitmap_ids is None):
        raise ValueError("workflow needs both split map and parent IDs")
    if splitmap is None:
        return {}
    paths = {"splitmap": Path(splitmap).resolve(), "splitmap_ids": Path(splitmap_ids).resolve()}
    sidecar = paths["splitmap_ids"].with_suffix(paths["splitmap_ids"].suffix + ".json")
    if any(path.is_symlink() or not path.is_file() for path in (*paths.values(), sidecar)):
        raise ValueError("workflow split map, parent IDs or ID receipt is missing")
    return paths


def _finish_stages(prepared: Path, parent: Path, out: Path, receipt: dict, gas_policy: str,
                   *, splitmap: Path | None = None, splitmap_ids: Path | None = None,
                   splitmap_solver: str | None = None, splitmap_backend: str = "native",
                   splitmap_python: str = "python3", splitmap_probe: Path | None = None) -> Path:
    receipt_path = out / "workflow.json"
    stage, started, started_utc = "field", time.perf_counter(), _now()
    try:
        if splitmap_solver is not None or splitmap_probe is not None:
            stage = "splitmap"
            if splitmap_probe is not None:
                if any(path.is_symlink() for path in splitmap_probe.rglob("*")):
                    raise ValueError("retained split-map probe contains a symlink")
                shutil.copytree(splitmap_probe, out / "splitmap")
                from verify_3d_workflow import _splitmap_diagnostic
                _splitmap_diagnostic(out, _sha(prepared / "prepared.json"), parent,
                                     receipt["frame_selection"]["timestep"])
                probe = json.loads((out / "splitmap/splitmap-probe.json").read_text(encoding="utf-8"))
            else:
                probe = _generate_splitmap(prepared, parent, out / "splitmap", splitmap_solver,
                                          splitmap_backend, splitmap_python,
                                          receipt["frame_selection"]["timestep"])
            receipt["children"][stage] = _receipt_ref(out / "splitmap/splitmap-probe.json")
            receipt["phases"][stage] = {"status": probe["status"], "started_utc": started_utc,
                "finished_utc": _now(), "duration_s": time.perf_counter() - started}
            if splitmap_probe is not None:
                receipt["phases"][stage].update(operation="retain-and-revalidate",
                    source_receipt=_receipt_ref(splitmap_probe / "splitmap-probe.json"))
            if probe["status"] == "complete":
                splitmap, splitmap_ids = out / "splitmap/splitmap.tsv", out / "splitmap/split-parent-ids.txt"
            elif probe["status"] != "not-needed":
                raise ValueError("split-map probe is incomplete")
            _write_json(receipt_path, receipt)
            stage, started, started_utc = "field", time.perf_counter(), _now()
        field = prepare_field(prepared, parent, receipt["frame_selection"], gas_policy, out / "field",
                              **_split_inputs(splitmap, splitmap_ids))
        receipt["children"]["field"] = _receipt_ref(field)
        receipt["phases"][stage] = {"status": "complete", "started_utc": started_utc,
                                     "finished_utc": _now(), "duration_s": time.perf_counter() - started}
        _write_json(receipt_path, receipt)
        stage, started, started_utc = "molecule", time.perf_counter(), _now()
        molecules = run_molecules(field.parent, out / "molecules")
        receipt["children"]["molecules"] = _receipt_ref(molecules)
        receipt["phases"][stage] = {"status": "complete", "started_utc": started_utc,
                                     "finished_utc": _now(), "duration_s": time.perf_counter() - started}
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        receipt["phases"][stage] = {"status": receipt["status"], "started_utc": started_utc,
                                     "finished_utc": _now(), "duration_s": time.perf_counter() - started}
        receipt.update(failed_stage=stage, failure=str(exc), finished_utc=_now())
        _write_json(receipt_path, receipt)
        raise
    receipt["finished_utc"] = _now()
    _write_json(receipt_path, receipt)
    return receipt_path


def _check_splitmap_options(inputs: dict, solver: str | None, backend: str, adapter: str) -> None:
    if solver is None:
        return
    if inputs:
        raise ValueError("choose an existing split map or a diagnostic solver, not both")
    if adapter != "cad" or backend not in ("native", "wsl"):
        raise ValueError("split-map generation needs CAD and a native or WSL backend")
    if backend == "native":
        if not Path(solver).is_file():
            raise ValueError("split-map diagnostic solver is missing")
    else:
        if os.name != "nt" or not shutil.which("wsl.exe"):
            raise ValueError("WSL split-map backend requires wsl.exe on Windows")
        checked = subprocess.run(["wsl.exe", "-e", "test", "-x", solver], check=False)
        if checked.returncode:
            raise ValueError("WSL split-map diagnostic solver is missing or not executable")


def _generate_splitmap(prepared: Path, parent: Path, out: Path, solver: str,
                       backend: str, python: str, timestep: int) -> dict:
    if backend == "native":
        from dsmc_splitmap import prepare_splitmap
        return prepare_splitmap(prepared, parent, Path(solver), out, timestep=timestep)
    command = ["wsl.exe", "-e", python, _wsl_path(ROOT / "tools/dsmc_splitmap.py"),
               "--prepared", _wsl_path(prepared), "--helium", _wsl_path(parent),
               "--solver", solver, "--out", _wsl_path(out), "--timestep", str(timestep)]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode:
        raise ValueError(f"WSL split-map probe failed: exit {result.returncode}")
    return json.loads((out / "splitmap-probe.json").read_text(encoding="utf-8"))


def finish_pipeline(prepared: Path, helium_receipt: Path, profile: Path, out: Path,
                    *, gas_policy: str, splitmap: Path | None = None,
                    splitmap_ids: Path | None = None, splitmap_solver: str | None = None,
                    splitmap_backend: str = "native", splitmap_probe: Path | None = None) -> Path:
    """Finish a verified helium parent without launching or relabeling that parent."""
    from dsmc_resume_parent import SCHEMA as RESUME_SCHEMA, validate_resume_parent

    prepared, helium_receipt, profile, out = (Path(p).resolve() for p in (prepared, helium_receipt, profile, out))
    if prepared.is_file():
        prepared = prepared.parent
    parent = helium_receipt.parent
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if any(out == path or out in path.parents or path in out.parents for path in (prepared, parent)):
        raise ValueError("downstream output must be separate from preparation and helium parent")
    inspected = inspect_geometry(prepared)
    split_inputs = _split_inputs(splitmap, splitmap_ids)
    original = json.loads(helium_receipt.read_text(encoding="utf-8"))
    if (inspected["state"] != "solver-checked" or original.get("status") != "complete" or
            original.get("exit_code") != 0 or original["prepared"]["sha256"] != _sha(prepared / "prepared.json")):
        raise ValueError("downstream workflow requires a complete helium parent bound to preparation")
    data = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    if splitmap_probe is not None:
        splitmap_probe = Path(splitmap_probe).resolve()
        if (split_inputs or splitmap_solver is not None or
                data["science_inputs"]["geometry"]["adapter"] != "cad"):
            raise ValueError("retained split-map probe requires CAD and excludes raw map or solver options")
        if (not (splitmap_probe / "splitmap-probe.json").is_file() or
                out == splitmap_probe or out in splitmap_probe.parents or splitmap_probe in out.parents):
            raise ValueError("retained split-map probe is missing or overlaps workflow output")
    _check_splitmap_options(split_inputs, splitmap_solver, splitmap_backend,
                           data["science_inputs"]["geometry"]["adapter"])
    if data["science_inputs"]["geometry"]["adapter"] == "cad":
        from dsmc_cad_workflow import validate_prepared_cad
        validate_prepared_cad(prepared)
    if original["schema_version"] == RESUME_SCHEMA:
        validate_resume_parent(parent, original)
        step = original["end_step"]
    elif original["schema_version"] == "dsmc-helium-attempt-v1":
        step = data["science_inputs"]["numerics"]["total_steps"]
    elif original["schema_version"] == "dsmc-cad-helium-import-v1":
        from dsmc_cad_workflow import validate_imported_cad
        validate_imported_cad(parent, original)
        step = original["end_step"]
    else:
        raise ValueError("unsupported helium parent schema")
    if gas_policy not in FIELD_POLICIES:
        raise ValueError("unsupported gas policy")
    load_profile(profile)
    _check_host_dependencies()
    if any(path.is_symlink() for path in parent.rglob("*")):
        raise ValueError("helium parent contains a symlink")
    out.mkdir(parents=True)
    shutil.copytree(parent, out / "helium")
    if _sha(out / "helium/manifest.json") != _sha(helium_receipt):
        raise ValueError("helium receipt changed during downstream staging")
    receipt = {"schema_version": "dsmc-local-workflow-v1", "status": "started", "created_utc": _now(),
               "case_id": data["case_id"], "repository": _repo_identity(),
               "prepared": _receipt_ref(prepared / "prepared.json"), "profile": _receipt_ref(profile),
               "gas_policy": gas_policy, "helium_backend": "retained-parent",
               "frame_selection": {"timestep": step}, "helium_exit_code": 0,
               "children": {"helium": _receipt_ref(out / "helium/manifest.json")},
               "phases": {"helium": {"status": "retained", "source_receipt": _receipt_ref(helium_receipt)}}}
    _write_json(out / "workflow.json", receipt)
    python = json.loads(profile.read_text(encoding="utf-8")).get("python", "python3")
    return _finish_stages(prepared, out / "helium", out, receipt, gas_policy, **split_inputs,
                          splitmap_solver=splitmap_solver, splitmap_backend=splitmap_backend,
                          splitmap_python=python, splitmap_probe=splitmap_probe)
