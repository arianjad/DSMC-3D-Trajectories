"""Publish native-read, one-shard-per-rank SPARTA checkpoint bundles.

The publisher deliberately copies first and probes the copied files. A restart
name or producer log is only a candidate: a complete bundle exists only when
the allocated SPARTA read emits diagnostics from that copy.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import tempfile
import uuid
from pathlib import Path

from dsmc_case import inspect_geometry, load_profile
from dsmc_run import _executable, render_launch

SCHEMA = "dsmc-checkpoint-bundle-v1"
SOURCE_SCHEMA = "dsmc-checkpoint-source-v1"
PROBE_SCHEMA = "dsmc-checkpoint-native-probe-v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if path == root or root not in path.parents or path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint source path is unsafe or missing: {value}")
    return path


def _inventory(root: Path, paths: list[str], *, nonempty: bool = True) -> list[dict]:
    result = []
    for relative in paths:
        path = _inside(root, relative)
        if nonempty and path.stat().st_size <= 0:
            raise ValueError(f"checkpoint payload is truncated or empty: {relative}")
        result.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path)})
    return result


def _same_inventory(expected: list[dict], observed: list[dict], label: str) -> None:
    if not isinstance(expected, list) or any(not isinstance(item, dict) for item in expected):
        raise ValueError(f"checkpoint source lacks immutable {label} inventory")
    wanted = {(item.get("path"), item.get("bytes"), item.get("sha256")) for item in expected}
    actual = {(item["path"], item["bytes"], item["sha256"]) for item in observed}
    if wanted != actual or len(wanted) != len(expected):
        raise ValueError(f"checkpoint {label} differs from immutable prelaunch source")


def _producer(source: dict) -> dict:
    if source.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("unsupported checkpoint source receipt")
    producer = source.get("producer")
    if not isinstance(producer, dict) or producer.get("one_shard_per_rank") is not True:
        raise ValueError("checkpoint source does not declare one-shard-per-rank output")
    if type(producer.get("ranks")) is not int or producer["ranks"] < 1:
        raise ValueError("checkpoint source has invalid producer rank count")
    numerics = source.get("prepared_record", {}).get("science_inputs", {}).get("numerics", {})
    expected = f"restart {numerics.get('restart_every_steps')} {producer.get('restart_prefix')}.%.*"
    if producer.get("restart_command") != expected:
        raise ValueError("checkpoint producer deck is not the supported one-shard numbered restart")
    return producer


def _generation(source: dict, attempt: Path, checkpoint: Path) -> dict:
    """Derive one candidate only from immutable rank layout and filename."""
    producer = _producer(source)
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.parent != attempt or not checkpoint.is_file():
        raise ValueError("checkpoint base is missing or outside the producer attempt")
    match = re.fullmatch(r"(.+)\.base\.(\d+)", checkpoint.name)
    if not match:
        raise ValueError("checkpoint base must use <prefix>.base.<step> numbering")
    prefix, step_text = match.groups(); step = int(step_text)
    if producer.get("restart_prefix") != prefix:
        raise ValueError("checkpoint base does not match immutable producer restart prefix")
    if checkpoint.stat().st_size < 8:
        raise ValueError("checkpoint base is truncated")
    # Pinned SPARTA 27Aug2026 write_restart.cpp writes MULTIPROC as the last
    # flag/value int pair in each base file; read_restart.cpp consumes it as
    # the actual number of shard files. MULTIPROC is enum value 45 there.
    with checkpoint.open("rb") as stream:
        stream.seek(-8, os.SEEK_END)
        flag, declared_files = struct.unpack("=ii", stream.read(8))
    if flag != 45 or declared_files != producer["ranks"]:
        raise ValueError("checkpoint base declares an unsupported shard count/layout")
    shards = [f"{prefix}.{rank}.{step}" for rank in range(producer["ranks"])]
    _inventory(attempt, [checkpoint.name, *shards])
    return {"step": step, "base": checkpoint.name, "shards": shards,
            "base_declared_files": declared_files,
            "diagnostic_averaging": {"discarded_until": step}}


def select_checkpoint(attempt: Path) -> Path:
    """Return latest complete candidate, retaining newer partial sets untouched."""
    attempt = Path(attempt).resolve()
    source = json.loads((attempt / "checkpoint-source.json").read_text(encoding="utf-8"))
    prefix = _producer(source).get("restart_prefix")
    if not isinstance(prefix, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", prefix):
        raise ValueError("checkpoint source lacks a safe immutable restart prefix")
    candidates = []
    for path in attempt.glob(f"{prefix}.base.*"):
        match = re.fullmatch(re.escape(prefix) + r"\.base\.(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for _, candidate in sorted(candidates, reverse=True):
        try:
            _generation(source, attempt, candidate)
        except ValueError:
            continue
        return candidate
    raise ValueError("no complete one-shard-per-rank checkpoint generation is available")


def _cad_probe_input(generation: dict, prepared_dir: Path, prepared: dict) -> str:
    """Render the CAD native-read probe from its checked staged preparation."""
    from dsmc_cad_workflow import validate_prepared_cad

    validate_prepared_cad(prepared_dir)
    helium = prepared["science_inputs"]["helium"]
    lines = (prepared_dir / "in.helium").read_text(encoding="utf-8").splitlines()
    normalized = lambda line: " ".join(line.split())
    physical = []
    for line in lines:
        words = normalized(line).split()
        if not words:
            continue
        if words[0] in {"collide", "surf_collide", "surf_modify"}:
            physical.append(normalized(line))
        elif words[0] == "fix" and len(words) > 1 and words[1] == "src":
            physical.append(normalized(line))
    if (len([line for line in physical if line == "collide vss hegas he.vss"]) != 1 or
            len([line for line in physical if line == "surf_modify cellwalls collide walls"]) != 1 or
            len([line for line in physical if line.startswith("surf_collide walls diffuse ")]) != 1 or
            len([line for line in physical if line.startswith("fix src emit/surf hegas cap normal yes mflow ")]) != 1 or
            helium.get("inlet", {}).get("model") != "emit/surf-mflow"):
        raise ValueError("CAD checkpoint probe cannot restore the checked physical wall/cap source laws")
    for line in physical:
        words = line.split()
        if ((words[0] == "collide" and line != "collide vss hegas he.vss") or
                (words[0] == "surf_collide" and not (line.startswith("surf_collide walls diffuse 4 ") or
                                                       line in {"surf_collide inlet vanish", "surf_collide station transparent"})) or
                (words[0] == "surf_modify" and len(words) > 1 and words[1] not in {"cellwalls", "cap", "xf", "xr"})):
            raise ValueError("CAD checkpoint probe has an undeclared physical restart command")
    # The checked producer deck runs from its staged-input directory. The
    # native-read probe runs from bundle/probe, beside bundle/prepared.
    physical = ["collide vss hegas ../prepared/inputs/he.vss"
                if line == "collide vss hegas he.vss" else line for line in physical]
    prefix = Path(generation["base"]).name.rsplit(".base.", 1)[0]
    return "\n".join((
        "# Generated CAD checkpoint-read probe. It must not advance physical time.",
        "seed 20260925",
        f"read_restart ../payload/{prefix}.%.{generation['step']}",
        *physical,
        "dump dprobe grid all 1 probe-grid.*.gz idstr split",
        "dump_modify dprobe first yes",
        "stats 0",
        "stats_style step np ngrid nsplit xlo xhi ylo yhi zlo zhi",
        "run 0", "",
    ))


def _probe_input(generation: dict, prepared_dir: Path | None = None) -> str:
    if prepared_dir is not None:
        prepared = json.loads((prepared_dir / "prepared.json").read_text(encoding="utf-8"))
        if prepared.get("science_inputs", {}).get("geometry", {}).get("adapter") == "cad":
            return _cad_probe_input(generation, prepared_dir, prepared)
    prefix = Path(generation["base"]).name.rsplit(".base.", 1)[0]
    return "\n".join((
        "# Generated checkpoint-read probe. It must not advance physical time.",
        "seed 20260925",
        f"read_restart ../payload/{prefix}.%.{generation['step']}",
        "collide vss hegas ../prepared/inputs/he.vss",
        "surf_collide walls diffuse 4.0 1.0",
        "surf_collide tp transparent",
        "surf_modify cellwalls collide walls",
        "surf_modify tf collide tp",
        "surf_modify tr collide tp",
        "surf_modify af collide tp",
        "surf_modify ar collide tp",
        "dump dprobe grid all 1 probe-grid.*.gz idstr split",
        "dump_modify dprobe first yes",
        "stats 0",
        "stats_style step np ngrid nsplit xlo xhi ylo yhi zlo zhi",
        "run 0", "",
    ))


def _parse_probe_log(log: Path, step: int) -> dict:
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    if any("Restart file used different # of processors" in line for line in lines):
        raise ValueError("native read reports producer processor count differs from declared ranks")
    wanted = ("step", "np", "ngrid", "nsplit", "xlo", "xhi", "ylo", "yhi", "zlo", "zhi")
    for index, line in enumerate(lines):
        header = line.split(); lowered = [item.lower() for item in header]
        if any(key not in lowered for key in wanted):
            continue
        positions = {key: lowered.index(key) for key in wanted}
        for data_line in lines[index + 1:]:
            values = data_line.split()
            if len(values) <= max(positions.values()):
                continue
            try:
                result = {key: float(values[position]) for key, position in positions.items()}
            except ValueError:
                continue
            if (all(math.isfinite(value) for value in result.values()) and result["step"].is_integer()
                    and int(result["step"]) == step and result["np"].is_integer()
                    and result["ngrid"].is_integer() and result["nsplit"].is_integer()
                    and result["np"] >= 0 and result["ngrid"] > 0 and 0 <= result["nsplit"] < result["ngrid"]):
                return {"timestep": step, "particles": int(result["np"]), "grid": int(result["ngrid"]),
                        "split_parents": int(result["nsplit"]),
                        "domain": [result[key] for key in ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")]}
    raise ValueError("native read probe log has no parseable step/np/ngrid/domain statistics row")


def _probe_grid(path: Path, step: int) -> dict:
    """Count the actual grid-dump rows; ngrid includes split parents, dumps do not."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            header = [next(stream).strip() for _ in range(9)]
            if (header[:4:2] != ["ITEM: TIMESTEP", "ITEM: NUMBER OF CELLS"] or
                    int(header[1]) != step or header[8] != "ITEM: CELLS idstr split"):
                raise ValueError("native probe grid dump has invalid header")
            declared = int(header[3])
            rows = 0
            for line in stream:
                if len(line.split()) != 2:
                    raise ValueError("native probe grid dump has malformed row")
                rows += 1
            if declared <= 0 or rows != declared:
                raise ValueError("native probe grid dump has incomplete rows")
    except (EOFError, OSError, UnicodeError, StopIteration) as exc:
        raise ValueError(f"native probe grid dump is incomplete: {exc}") from exc
    return {"path": f"probe/{path.name}", "sha256": _sha(path), "rows": rows}


def native_probe(bundle_stage: Path, generation: dict, profile: dict | Path, *, allocation_id: str | None = None,
                 runner=None) -> dict:
    """Run SPARTA ``read_restart`` against copied payload and record it.

    Production calls require an allocation identifier. ``runner`` exists solely
    to mock this native boundary in contract tests.
    """
    bundle_stage = Path(bundle_stage).resolve()
    profile = load_profile(profile) if isinstance(profile, Path) else dict(profile)
    allocation_id = allocation_id or os.environ.get("SLURM_JOB_ID")
    if not isinstance(allocation_id, str) or not allocation_id:
        raise ValueError("native checkpoint probe requires an allocated scheduler job identifier")
    if runner is None and os.environ.get("SLURM_JOB_ID") != allocation_id:
        raise ValueError("native checkpoint probe must run inside its declared Slurm allocation")
    if runner is None and profile.get("platform") != "local":
        raise ValueError("native checkpoint probe requires the concrete local launch profile inside its allocation")
    if type(profile.get("ranks")) is not int or profile["ranks"] < 1:
        raise ValueError("native checkpoint probe profile has invalid rank count")
    probe_dir = bundle_stage / "probe"; probe_dir.mkdir(exist_ok=False)
    input_path, log_path = probe_dir / "in.read-restart", probe_dir / "read-restart.log"
    input_path.write_text(_probe_input(generation, bundle_stage / "prepared"), encoding="utf-8", newline="\n")
    executable = _executable(profile["solver_executable"])
    argv = render_launch(profile, executable, ["-in", input_path.name])
    with log_path.open("wb") as stream:
        if runner is not None:
            completed = runner(argv, cwd=probe_dir, stdout=stream, stderr=subprocess.STDOUT, check=False)
            code = getattr(completed, "returncode", None)
        else:
            env = os.environ.copy()
            env.update({key: str(profile["threads"]) for key in
                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
            child = subprocess.Popen(argv, cwd=probe_dir, stdout=stream, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=(os.name != "nt"))
            try:
                code = child.wait(timeout=120)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if child.poll() is None:
                    if os.name != "nt": os.killpg(child.pid, signal.SIGTERM)
                    else: child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name != "nt": os.killpg(child.pid, signal.SIGKILL)
                    else: child.kill()
                    child.wait()
                raise ValueError("native checkpoint read timed out or was interrupted")
    if code != 0:
        raise ValueError(f"native checkpoint read failed with exit status {code}")
    diagnostics = _parse_probe_log(log_path, generation["step"])
    grid_dump = _probe_grid(probe_dir / f"probe-grid.{generation['step']}.gz", generation["step"])
    if (diagnostics["split_parents"] == 0 and grid_dump["rows"] != diagnostics["grid"]) or \
            grid_dump["rows"] < diagnostics["grid"] - diagnostics["split_parents"]:
        raise ValueError("native probe grid dump rows disagree with read-restart grid")
    diagnostics["dump_rows"] = grid_dump["rows"]
    payloads = _inventory(bundle_stage / "payload", [Path(generation["base"]).name, *generation["shards"]])
    probe = {"schema_version": PROBE_SCHEMA, "status": "passed",
             "allocation": {"scheduler": "slurm", "job_id": allocation_id},
             "producer_ranks": len(generation["shards"]), "probe_ranks": profile["ranks"],
             "solver": {"path": executable, "sha256": _sha(Path(executable))},
             "step": generation["step"], "diagnostics": diagnostics,
             "payload_sha256": {item["path"]: item["sha256"] for item in payloads},
             "command": argv, "input": {"path": "probe/in.read-restart", "sha256": _sha(input_path)},
             "log": {"path": "probe/read-restart.log", "sha256": _sha(log_path)}, "grid_dump": grid_dump}
    _write(probe_dir / "native-probe.json", probe)
    return probe


class _NativeReadRejected(ValueError):
    """A copied generation failed its allocated native-read gate."""


def _retain_rejection(stage: Path, out: Path, generation: dict, error: ValueError) -> None:
    rejected = out.with_name(out.name + ".rejections") / f"step-{generation['step']}-{uuid.uuid4().hex[:8]}"
    rejected.mkdir(parents=True)
    if (stage / "probe").is_dir():
        shutil.copytree(stage / "probe", rejected / "probe")
    _write(rejected / "rejection.json", {"step": generation["step"], "base": generation["base"],
                                         "shards": generation["shards"], "reason": str(error)})


def _archive_one(attempt: Path, checkpoint: Path, out: Path, *, profile: dict | Path | None = None,
                       allocation_id: str | None = None, runner=None) -> Path:
    """Publish one copied, allocated-native-probed checkpoint generation."""
    attempt, out = Path(attempt).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink() or out == attempt or out in attempt.parents or attempt in out.parents:
        raise FileExistsError("checkpoint output must be a fresh directory separate from its attempt")
    if profile is None:
        raise ValueError("checkpoint publication requires the allocated native probe profile")
    source_path = attempt / "checkpoint-source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    checkpoint = Path(checkpoint).resolve()
    generation = _generation(source, attempt, checkpoint)
    payloads = _inventory(attempt, [generation["base"], *generation["shards"]])
    active_paths = [item.get("path") for item in source.get("active_inputs", [])]
    context_paths = [item.get("path") for item in source.get("preparation_context", [])]
    if any(not isinstance(path, str) for path in active_paths + context_paths):
        raise ValueError("checkpoint source has invalid immutable input paths")
    active = _inventory(attempt, active_paths)
    for item in active:
        item["role"] = next(source_item.get("role") for source_item in source["active_inputs"]
                            if source_item.get("path") == item["path"])
    context = _inventory(attempt, context_paths)
    _same_inventory(source["active_inputs"], active, "active inputs")
    _same_inventory(source["preparation_context"], context, "preparation context")
    if not isinstance(source.get("prepared_record"), dict):
        raise ValueError("checkpoint source lacks checked immutable preparation")
    if _sha(Path(_executable((load_profile(profile) if isinstance(profile, Path) else profile)["solver_executable"]))) != source.get("solver", {}).get("sha256"):
        raise ValueError("allocated native probe solver differs from checkpoint producer")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-checkpoint-", dir=out.parent) as temp:
        stage = Path(temp) / "bundle"; (stage / "payload").mkdir(parents=True)
        copied = []
        for item in payloads:
            target = stage / "payload" / item["path"]
            shutil.copyfile(_inside(attempt, item["path"]), target)
            if _sha(target) != item["sha256"]: raise ValueError(f"checkpoint copy changed: {item['path']}")
            copied.append({**item, "path": (Path("payload") / item["path"]).as_posix(), "kind": "restart"})
        prepared_files = []
        for item in context:
            target = stage / "prepared" / Path(item["path"]).name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_inside(attempt, item["path"]), target)
            prepared_files.append({**item, "path": (Path("prepared") / target.name).as_posix()})
        active_files = []
        for item in active:
            target = stage / "prepared" / "inputs" / Path(item["path"]).name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(_inside(attempt, item["path"]), target)
            active_files.append({**item, "path": (Path("prepared/inputs") / target.name).as_posix()})
        mapping = {generation["base"]: "checkpoint.restart.base"}
        mapping.update({shard: f"checkpoint.restart.{index}" for index, shard in enumerate(generation["shards"])})
        try:
            probe = native_probe(stage, generation, profile, allocation_id=allocation_id, runner=runner)
        except ValueError as exc:
            _retain_rejection(stage, out, generation, exc)
            raise _NativeReadRejected(str(exc)) from exc
        probe_file = stage / "probe/native-probe.json"
        shutil.copyfile(source_path, stage / "checkpoint-source.json")
        manifest = {"schema_version": SCHEMA, "status": "complete", "source_attempt": source.get("attempt_identity"),
                    "source_receipt": {"path": "checkpoint-source.json", "sha256": _sha(source_path)},
                    "prepared": {"record": source["prepared_record"], "files": prepared_files, "active_inputs": active_files},
                    "solver": source.get("solver"), "producer": source["producer"], "generation": generation,
                    "native_probe": {"path": "probe/native-probe.json", "sha256": _sha(probe_file),
                                     "input": probe["input"], "log": probe["log"],
                                     "grid_dump": probe["grid_dump"], "diagnostics": probe["diagnostics"]},
                    "files": copied, "staging_name_mapping": mapping}
        _write(stage / "checkpoint-manifest.json", manifest)
        try:
            verify_checkpoint(stage)
        except ValueError as exc:
            _retain_rejection(stage, out, generation, exc)
            raise _NativeReadRejected(str(exc)) from exc
        stage.rename(out)
    return out / "checkpoint-manifest.json"


def archive_checkpoint(attempt: Path, checkpoint: Path | None, out: Path, *, profile: dict | Path | None = None,
                       allocation_id: str | None = None, runner=None) -> Path:
    """Publish the newest copied generation that passes a native read.

    An explicit selection is never silently replaced. Rejected source files
    remain in the attempt and probe evidence is kept beside the output path.
    """
    attempt = Path(attempt).resolve()
    if checkpoint is not None:
        return _archive_one(attempt, checkpoint, out, profile=profile,
                            allocation_id=allocation_id, runner=runner)
    source = json.loads((attempt / "checkpoint-source.json").read_text(encoding="utf-8"))
    prefix = _producer(source)["restart_prefix"]
    candidates = sorted(attempt.glob(f"{prefix}.base.*"),
                        key=lambda path: int(path.name.rsplit(".", 1)[1])
                        if path.name.rsplit(".", 1)[1].isdigit() else -1, reverse=True)
    failures = []
    for candidate in candidates:
        try:
            _generation(source, attempt, candidate)
        except ValueError:
            continue
        try:
            return _archive_one(attempt, candidate, out, profile=profile,
                                allocation_id=allocation_id, runner=runner)
        except _NativeReadRejected as exc:
            failures.append(f"{candidate.name}: {exc}")
    raise ValueError("no native-readable checkpoint generation is available" +
                     (": " + "; ".join(failures) if failures else ""))


def verify_checkpoint(bundle: Path) -> dict:
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("checkpoint bundle is not complete")
    source_file = _inside(bundle, manifest.get("source_receipt", {}).get("path", ""))
    if _sha(source_file) != manifest["source_receipt"].get("sha256"):
        raise ValueError("checkpoint source receipt changed after publication")
    source = json.loads(source_file.read_text(encoding="utf-8")); producer = _producer(source)
    generation = manifest.get("generation")
    if not isinstance(generation, dict) or producer != manifest.get("producer"):
        raise ValueError("checkpoint producer layout differs from immutable source receipt")
    expected_generation = _generation(source, bundle / "payload", bundle / "payload" / generation.get("base", ""))
    if expected_generation != generation:
        raise ValueError("checkpoint generation/layout differs from copied payload")
    expected_mapping = {generation["base"]: "checkpoint.restart.base"}
    expected_mapping.update({shard: f"checkpoint.restart.{rank}" for rank, shard in enumerate(generation["shards"])})
    if manifest.get("staging_name_mapping") != expected_mapping:
        raise ValueError("checkpoint staging mapping differs from one-shard-per-rank layout")
    restart_files = manifest.get("files")
    if not isinstance(restart_files, list) or len(restart_files) != len(generation["shards"]) + 1:
        raise ValueError("checkpoint restart inventory is incomplete")
    for item in restart_files:
        path = _inside(bundle, item.get("path", ""))
        if item.get("kind") != "restart" or path.stat().st_size != item.get("bytes") or _sha(path) != item.get("sha256"):
            raise ValueError(f"checkpoint payload changed: {item.get('path')}")
    prepared = manifest.get("prepared")
    if not isinstance(prepared, dict) or prepared.get("record") != source.get("prepared_record"):
        raise ValueError("checkpoint checked preparation differs from immutable source")
    prepared_receipt = _inside(bundle, "prepared/prepared.json")
    if (_sha(prepared_receipt) != source.get("prepared", {}).get("sha256") or
            json.loads(prepared_receipt.read_text(encoding="utf-8")) != source["prepared_record"]):
        raise ValueError("checkpoint prepared.json differs from immutable prelaunch preparation")
    try:
        inspect_geometry(bundle / "prepared")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"checkpoint copied preparation/geometry is not solver-ready: {exc}") from exc
    if source["prepared_record"].get("science_inputs", {}).get("geometry", {}).get("adapter") == "cad":
        try:
            from dsmc_cad_workflow import validate_prepared_cad
            validate_prepared_cad(bundle / "prepared")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"checkpoint copied CAD preparation is not runnable: {exc}") from exc
    for label, source_items, bundle_items in (("preparation context", source.get("preparation_context"), prepared.get("files")),
                                              ("active inputs", source.get("active_inputs"), prepared.get("active_inputs"))):
        if not isinstance(bundle_items, list): raise ValueError(f"checkpoint bundle lacks {label}")
        expected = {(item.get("bytes"), item.get("sha256")) for item in source_items or []}; actual = set()
        for item in bundle_items:
            path = _inside(bundle, item.get("path", "")); actual.add((item.get("bytes"), item.get("sha256")))
            if path.stat().st_size != item.get("bytes") or _sha(path) != item.get("sha256"):
                raise ValueError(f"checkpoint {label} payload changed: {item.get('path')}")
        if actual != expected or len(actual) != len(bundle_items):
            raise ValueError(f"checkpoint {label} inventory differs from immutable source")
    source_roles = {(item.get("path"), item.get("role")) for item in source.get("active_inputs", [])}
    bundle_roles = {(Path(item.get("path", "")).name, item.get("role")) for item in prepared["active_inputs"]}
    if source_roles != bundle_roles:
        raise ValueError("checkpoint active input roles differ from immutable prelaunch source")
    probe_meta = manifest.get("native_probe", {})
    probe_path = _inside(bundle, probe_meta.get("path", "")); probe = json.loads(probe_path.read_text(encoding="utf-8"))
    if _sha(probe_path) != probe_meta.get("sha256") or probe.get("schema_version") != PROBE_SCHEMA or probe.get("status") != "passed":
        raise ValueError("checkpoint native probe receipt changed or failed")
    for key in ("input", "log", "grid_dump"):
        item = probe_meta.get(key, {}); path = _inside(bundle, item.get("path", ""))
        if _sha(path) != item.get("sha256") or probe.get(key) != item:
            raise ValueError(f"checkpoint native probe {key} differs from manifest")
    payload_hashes = {Path(item["path"]).name: item["sha256"] for item in restart_files}
    if (probe.get("step") != generation["step"] or probe.get("producer_ranks") != producer["ranks"] or
            probe.get("probe_ranks") != producer["ranks"] or
            probe.get("solver", {}).get("sha256") != source.get("solver", {}).get("sha256") or
            manifest.get("solver") != source.get("solver") or
            probe.get("payload_sha256") != payload_hashes or probe.get("diagnostics") != probe_meta.get("diagnostics")):
        raise ValueError("checkpoint native probe does not bind copied payload/layout/solver")
    diagnostic = probe["diagnostics"]
    if (diagnostic.get("timestep") != generation["step"] or type(diagnostic.get("particles")) is not int or
            type(diagnostic.get("grid")) is not int or diagnostic["grid"] < math.prod(
                source["prepared_record"]["science_inputs"]["numerics"]["base_grid"]) or
            diagnostic["particles"] < 0 or type(diagnostic.get("split_parents")) is not int or
            type(diagnostic.get("dump_rows")) is not int or
            diagnostic["dump_rows"] != probe["grid_dump"].get("rows") or
            not isinstance(diagnostic.get("domain"), list) or len(diagnostic["domain"]) != 6):
        raise ValueError("checkpoint native probe lacks valid timestep/particle/domain/grid diagnostics")
    expected_domain = [value for pair in source["prepared_record"]["science_inputs"]["numerics"]["domain_bounds_m"]
                       for value in pair]
    if any(not math.isfinite(value) or not math.isclose(value, expected, rel_tol=0, abs_tol=1e-12)
           for value, expected in zip(diagnostic["domain"], expected_domain)):
        raise ValueError("checkpoint native probe domain differs from immutable preparation")
    return {"status": "passed", "checkpoint_step": generation["step"], "source_ranks": producer["ranks"],
            "native_probe": "allocated-read-restart", "diagnostics": diagnostic}
