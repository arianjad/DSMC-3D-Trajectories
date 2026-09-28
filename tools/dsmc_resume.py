#!/usr/bin/env python3
"""Prepare a fresh B5 SPARTA continuation from a verified full archive."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from dsmc_archive import verify_archive
from dsmc_checkpoint import verify_checkpoint


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(lines: list[str], pattern: str) -> str:
    matches = [line for line in lines if re.fullmatch(pattern, line.strip())]
    if len(matches) != 1:
        raise ValueError(f"parent helium deck needs exactly one {pattern!r} command")
    return matches[0]


def _between(lines: list[str], first: str, last: str) -> list[str]:
    starts = [i for i, line in enumerate(lines) if line.startswith(first)]
    ends = [i for i, line in enumerate(lines) if line.startswith(last)]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError(f"parent helium deck has no unique {first!r} to {last!r} section")
    return lines[starts[0]:ends[0]]


def _deck(source: str, prepared: dict, seed: int, target_step: int,
          *, rebalance_restart: bool = False, checkpoint_step: int | None = None,
          restart_basename: str = "final.restart") -> str:
    lines = source.splitlines()
    science = prepared["science_inputs"]
    helium, numerics = science["helium"], science["numerics"]
    if (prepared["case_id"] not in ("b5-circle-s25-sampled-probe",
                                     "b5-slit-s25-sampled-probe", "b5-circle-s25-recovery-probe") or
            helium["wall_temperature_K"] != 4 or helium["wall_accommodation"] != 1 or
            helium["inlet"]["temperature_K"] != 4 or
            helium["inlet"]["model"] != "emit/surf-mflow" or
            numerics["field_average"] != {"samples": 100, "stride_steps": 12,
                                           "period_steps": 1200} or
            numerics["field_every_steps"] != 2400 or
            numerics["restart_every_steps"] not in (24000, 2400)):
        raise ValueError("B5 restart renderer does not support changed physics or schedules")
    checkpoint_step = numerics["total_steps"] if checkpoint_step is None else checkpoint_step
    if type(checkpoint_step) is not int or checkpoint_step < 1 or target_step <= checkpoint_step + 1:
        raise ValueError("resume target must leave a sampled run after the first restored-physics step")
    if seed == numerics["solver_seed"]:
        raise ValueError("resume attempt needs a new explicit solver seed")
    _one(lines, r"seed\s+\$\{SEED\}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", restart_basename):
        raise ValueError("restart basename is unsafe")
    restart_command = f"read_restart {restart_basename}.%"
    if rebalance_restart:
        restart_command += " balance rcb cell"
    commands = [f"seed {seed}", restart_command]
    for pattern in (r"collide\s+vss hegas he\.vss",
                    r"surf_collide\s+walls diffuse 4\.0 1\.0",
                    r"surf_collide\s+tp transparent",
                    r"surf_modify\s+cellwalls collide walls",
                    r"surf_modify\s+tf collide tp",
                    r"surf_modify\s+tr collide tp",
                    r"surf_modify\s+af collide tp",
                    r"surf_modify\s+ar collide tp",
                    r"fix\s+src emit/surf hegas cap normal yes mflow \$\{MDOT\} 4\.0",
                    r"stats\s+1000",
                    r"stats_style\s+step cpu np nattempt ncoll f_src\[1\] f_src\[2\]",
                    r"stats_modify\s+flush yes"):
        line = _one(lines, pattern)
        commands.append(line.replace("${MDOT}", str(helium["inlet"]["mass_flow_kg_s"])))
    histories = _between(lines, "# --- histories", "# Sampling campaign:")
    if sum(line.lstrip().startswith("compute") for line in histories) < 10:
        raise ValueError("parent deck has too few history computes to restore")
    # The restart already stores grid and surface groups. Repeating their
    # definitions would make a new selection from region commands, which are
    # not stored and need not be reconstructed for these fixed B5 groups.
    histories = [line for line in histories
                 if not line.lstrip().startswith(("region ", "group "))]
    sampled = _between(lines, "fix         perf balance", "variable    nmain equal")
    if len([line for line in sampled if line.lstrip().startswith("dump ")]) != 1:
        raise ValueError("parent deck lacks one sampled field dump")
    _one(lines, r"write_restart\s+final\.restart\.%")
    # SPARTA always emits a dump at the first run step if the checkpoint step
    # is an exact multiple of its frequency, even with dump_modify first no.
    # At B5's step 4800 this would create an all-zero-density fields.4800.gz.
    # Advance one physical step with source/collisions/walls restored, then
    # define the new averaging fixes and dumps. The first sample is due at
    # step 4812, so no average sample is lost at 4801.
    commands.extend(["# Advance past the checkpoint dump boundary with physics restored.",
                     "run 1",
                     "# Histories and field averages start anew after the checkpoint.",
                     *histories, *sampled, f"run {target_step} upto",
                     "write_restart continued.restart.%", ""])
    # Fail closed if the producer acquires a new active model or diagnostic
    # command that this continuation did not restore. Whitespace in SPARTA
    # input is immaterial, and MDOT is the one resolved source placeholder.
    families = {"collide", "surf_collide", "surf_modify", "fix", "compute",
                "dump", "dump_modify", "restart", "stats", "stats_style",
                "stats_modify"}
    normalized = lambda line: " ".join(line.split()).replace(
        "${MDOT}", str(helium["inlet"]["mass_flow_kg_s"]))
    original_active = [normalized(line) for line in lines
                       if line.split() and line.split()[0] in families]
    resumed_active = [normalized(line) for line in commands
                      if line.split() and line.split()[0] in families]
    if sorted(original_active) != sorted(resumed_active):
        raise ValueError("parent helium deck has a model/diagnostic command not restored")
    if any(line.split() and line.split()[0] in
           {"react", "surf_react", "collide_modify", "custom"} for line in lines):
        raise ValueError("parent helium deck has unsupported restart-sensitive commands")
    return "\n".join(commands)


def _cad_deck(source: str, prepared: dict, seed: int, target_step: int, *,
              checkpoint_step: int, rebalance_restart: bool = False) -> str:
    """Render the declared CAD transient continuation without making new particles.

    A SPARTA restart retains the grid, surfaces, and surface groups.  This
    deliberately restores the collision/source laws and fresh diagnostics, but
    never repeats the source deck's ``read_particles``/``create_particles``
    initial condition.
    """
    science = prepared.get("science_inputs", {})
    helium, numerics = science.get("helium", {}), science.get("numerics", {})
    meta = prepared.get("cad_workflow", {})
    if (prepared.get("case_id") != "synthetic-off-axis-cad-hpc" or
            science.get("geometry", {}).get("adapter") != "cad" or
            meta.get("schema_version") != "dsmc-cad-workflow-preparation-v1" or
            numerics.get("total_steps") != 4000 or numerics.get("dt_s") != 1e-7 or
            numerics.get("solver_seed") != 20260927 or
            numerics.get("restart_every_steps") != 2000 or
            numerics.get("restart_prefix") != "cad.restart" or
            numerics.get("field_every_steps") != 4000 or
            numerics.get("field_average") != {"samples": 100, "stride_steps": 10,
                                               "period_steps": 1000} or
            helium.get("wall_temperature_K") != 4 or helium.get("wall_accommodation") != 1 or
            helium.get("inlet", {}).get("model") != "emit/surf-mflow" or
            helium.get("inlet", {}).get("temperature_K") != 4 or
            checkpoint_step != 2000 or target_step != 4000 or seed != 20260929):
        raise ValueError("CAD restart renderer only supports the declared 2000-to-4000 synthetic fixture")
    lines = source.splitlines()
    normalized = lambda line: " ".join(line.split())
    for pattern in (r"seed\s+20260927", r"restart\s+2000\s+cad\.restart\.%\.\*",
                    r"run\s+4000", r"write_restart\s+final\.restart\.%"):
        _one(lines, pattern)
    if any(normalized(line).startswith(("react ", "surf_react ", "collide_modify ", "custom "))
           for line in lines):
        raise ValueError("CAD source deck has unsupported restart-sensitive commands")

    physical, diagnostics = [], []
    for line in lines:
        words = normalized(line).split()
        if not words:
            continue
        head = words[0]
        if head in {"collide", "surf_collide", "surf_modify"}:
            physical.append(normalized(line))
        elif head == "fix" and len(words) > 1 and words[1] == "src":
            physical.append(normalized(line))
        elif head in {"stats", "stats_style", "stats_modify"}:
            physical.append(normalized(line))
        elif head in {"compute", "fix", "dump", "dump_modify"}:
            # Particle snapshots belong to the producer only.  The resumed
            # field is deliberately the final post-checkpoint field alone.
            if len(words) > 1 and words[1] == "state":
                continue
            diagnostics.append(normalized(line))
        elif head == "region" and len(words) > 1 and words[1] == "eqbody":
            diagnostics.append(normalized(line))
        elif head == "group" and len(words) > 1 and words[1] in {"eqbody", "exitdiag"}:
            diagnostics.append(normalized(line))
        elif head == "variable" and len(words) > 1 and words[1] == "eqinventory":
            diagnostics.append(normalized(line))
    if (len([line for line in physical if line.startswith("collide vss hegas he.vss")]) != 1 or
            len([line for line in physical if line.startswith("surf_collide walls diffuse 4 ")]) != 1 or
            len([line for line in physical if line.startswith("surf_modify cellwalls collide walls")]) != 1 or
            len([line for line in physical if line.startswith("fix src emit/surf hegas cap normal yes mflow ")]) != 1 or
            len([line for line in diagnostics if line.startswith("fix afields ave/grid all 10 100 1000 ")]) != 1 or
            len([line for line in diagnostics if line.startswith("dump dfields grid all 4000 fields.*.gz ")]) != 1):
        raise ValueError("CAD source deck lacks exactly one declared physical or field command")
    for line in physical:
        words = line.split()
        if ((words[0] == "collide" and line != "collide vss hegas he.vss") or
                (words[0] == "surf_collide" and not (line.startswith("surf_collide walls diffuse 4 ") or
                                                       line in {"surf_collide inlet vanish", "surf_collide station transparent"})) or
                (words[0] == "surf_modify" and len(words) > 1 and words[1] not in {"cellwalls", "cap", "xf", "xr"})):
            raise ValueError("CAD source deck has an undeclared physical restart command")
    restart = "read_restart checkpoint.restart.%"
    if rebalance_restart:
        restart += " balance rcb cell"
    # Define averages only after the first restored-physics step.  Thus the
    # retained frame uses samples 3010..4000 rather than producer history.
    return "\n".join([f"seed {seed}", restart, *physical,
                      "run 1", *diagnostics, f"run {target_step} upto",
                      "write_restart continued.restart.%", ""])


def prepare_resume(parent: Path, checkpoint: Path, out: Path, *, seed: int,
                   target_step: int, ranks: int | None = None) -> Path:
    """Stage an immutable archived checkpoint into a new, unrun attempt."""
    parent, out = Path(parent).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"resume output already exists: {out}")
    if out == parent or out in parent.parents or parent in out.parents:
        raise ValueError("resume output must be separate from parent archive")
    if type(seed) is not int or seed <= 0 or type(target_step) is not int or target_step <= 0:
        raise ValueError("resume seed and target step must be positive integers")
    checked = verify_archive(parent)
    manifest = json.loads((parent / "archive-manifest.json").read_text(encoding="utf-8"))
    prepared = json.loads((parent / "prepared/prepared.json").read_text(encoding="utf-8"))
    if checked["source_verification"]["case_id"] != prepared["case_id"]:
        raise ValueError("archive verification and preparation case IDs differ")
    if checked["source_verification"]["helium_steps"] != prepared["science_inputs"]["numerics"]["total_steps"]:
        raise ValueError("archived helium timestep differs from prepared numerics")
    base = (parent / "workflow/helium/final.restart.base").resolve()
    selected = Path(checkpoint).resolve()
    if selected != base:
        raise ValueError("only the verified final.restart.base checkpoint is supported")
    restart = sorted(manifest["final_restart"], key=lambda item: item["path"])
    names = {Path(item["path"]).name for item in restart}
    shards = {int(match.group(1)) for name in names
              if (match := re.fullmatch(r"final\.restart\.(\d+)", name))}
    if (len(restart) != len(names) or "final.restart.base" not in names or
            shards != set(range(len(restart) - 1))):
        raise ValueError("archived final restart base/shard set is incomplete")
    source_ranks = len(shards)
    if ranks is None:
        ranks = source_ranks
    if type(ranks) is not int or ranks <= 0:
        raise ValueError("resume rank count must be a positive integer")
    run_source = parent / "workflow/helium"
    source_deck = (run_source / "in.helium").read_text(encoding="utf-8")
    deck = _deck(source_deck, prepared, seed, target_step,
                 rebalance_restart=(ranks != source_ranks))
    selected_sources = [(parent / item["path"], Path(item["path"]).name)
                        for item in restart]
    selected_sources.extend([(run_source / "he.vss", "he.vss"),
                             (parent / "prepared/prepared.json", "prepared.json")])
    out.parent.mkdir(parents=True, exist_ok=True)
    needed = sum(path.stat().st_size for path, _ in selected_sources)
    if shutil.disk_usage(out.parent).free < needed + 512 * 1024 * 1024:
        raise ValueError("insufficient space for restart inputs plus 512 MiB reserve")
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-resume-", dir=out.parent) as temp:
        stage = Path(temp)
        run = stage / "run"
        run.mkdir()
        staged = []
        for source, name in selected_sources:
            destination = run / name
            shutil.copyfile(source, destination)
            if _sha(destination) != _sha(source):
                raise ValueError(f"restart staging changed {name}")
            staged.append({"path": f"run/{name}", "sha256": _sha(destination),
                           "bytes": destination.stat().st_size})
        script = run / "in.resume"
        script.write_text(deck, encoding="utf-8", newline="\n")
        staged.append({"path": "run/in.resume", "sha256": _sha(script),
                       "bytes": script.stat().st_size})
        receipt = {"schema_version": "dsmc-b5-resume-preparation-v1",
                   "status": "prepared", "case_id": prepared["case_id"],
                   "parent_archive_sha256": _sha(parent / "archive-manifest.json"),
                   "checkpoint": {"name": "final.restart.base", "sha256": _sha(base),
                                  "expected_timestep": checked["source_verification"]["helium_steps"]},
                   "seed": seed, "target_step": target_step,
                   "source_ranks": source_ranks, "target_ranks": ranks,
                   "averaging": "fresh-from-checkpoint",
                   "diagnostics_defined_after_step": checked["source_verification"]["helium_steps"] + 1,
                   "source_deck_sha256": _sha(run_source / "in.helium"),
                   "inputs": staged}
        (stage / "resume.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8", newline="\n")
        stage.rename(out)
    return out / "resume.json"


def prepare_resume_checkpoint(bundle: Path, out: Path, *, seed: int,
                              target_step: int, ranks: int | None = None) -> Path:
    """Stage a continuation from a native-probed incomplete-parent checkpoint bundle."""
    bundle, out = Path(bundle).resolve(), Path(out).resolve()
    if out.exists() or out.is_symlink() or out == bundle or out in bundle.parents or bundle in out.parents:
        raise FileExistsError("resume output must be fresh and separate from checkpoint bundle")
    if type(seed) is not int or seed <= 0 or type(target_step) is not int or target_step <= 0:
        raise ValueError("resume seed and target step must be positive integers")
    checked = verify_checkpoint(bundle)
    manifest = json.loads((bundle / "checkpoint-manifest.json").read_text(encoding="utf-8"))
    source = json.loads((bundle / "checkpoint-source.json").read_text(encoding="utf-8"))
    prepared = source.get("prepared_record")
    if not isinstance(prepared, dict) or "science_inputs" not in prepared or "case_id" not in prepared:
        raise ValueError("checkpoint source lacks the immutable prepared record needed for recovery")
    start, source_ranks = checked["checkpoint_step"], checked["source_ranks"]
    if target_step <= start + 1:
        raise ValueError("resume target must exceed the verified checkpoint step")
    ranks = source_ranks if ranks is None else ranks
    if type(ranks) is not int or ranks < 1:
        raise ValueError("resume rank count must be positive")
    mapping = manifest["staging_name_mapping"]
    source_deck = bundle / "prepared" / "inputs" / "in.helium"
    if source_deck is None or not source_deck.is_file():
        raise ValueError("checkpoint bundle lacks active helium deck")
    adapter = prepared.get("science_inputs", {}).get("geometry", {}).get("adapter")
    if adapter == "cad":
        from dsmc_cad_workflow import validate_prepared_cad
        validate_prepared_cad(bundle / "prepared")
        deck = _cad_deck(source_deck.read_text(encoding="utf-8"), prepared, seed, target_step,
                         checkpoint_step=start, rebalance_restart=(ranks != source_ranks))
    elif adapter in ("b5-circle", "b5-slit"):
        deck = _deck(source_deck.read_text(encoding="utf-8"), prepared, seed, target_step,
                     checkpoint_step=start, restart_basename="checkpoint.restart",
                     rebalance_restart=(ranks != source_ranks))
    else:
        raise ValueError("checkpoint resume has no supported checked geometry adapter")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}-resume-", dir=out.parent) as temp:
        stage, run = Path(temp), Path(temp) / "run"; run.mkdir()
        staged = []
        for original, staged_name in mapping.items():
            source_file = bundle / "payload" / original
            target = run / staged_name; shutil.copyfile(source_file, target)
            staged.append({"path": f"run/{staged_name}", "sha256": _sha(target), "bytes": target.stat().st_size})
        for item in manifest["prepared"]["active_inputs"]:
            source_file = bundle / item["path"]
            if source_file.name in ("in.helium", "he.vss"):
                target = run / source_file.name; shutil.copyfile(source_file, target)
                staged.append({"path": f"run/{target.name}", "sha256": _sha(target), "bytes": target.stat().st_size})
        script = run / "in.resume"; script.write_text(deck, encoding="utf-8", newline="\n")
        staged.append({"path": "run/in.resume", "sha256": _sha(script), "bytes": script.stat().st_size})
        receipt = {"schema_version": "dsmc-checkpoint-resume-preparation-v1", "status": "prepared",
                   "case_id": prepared["case_id"], "checkpoint_bundle_sha256": _sha(bundle / "checkpoint-manifest.json"),
                   "checkpoint_bundle": {"manifest": "checkpoint-manifest.json",
                                         "sha256": _sha(bundle / "checkpoint-manifest.json")},
                   "source_attempt": manifest["source_attempt"], "prepared": manifest["prepared"],
                   "adapter": adapter,
                   "checkpoint": {"step": start, "native_probe": checked["diagnostics"], "mapping": mapping},
                   "seed": seed, "target_step": target_step, "source_ranks": source_ranks,
                   "target_ranks": ranks, "averaging": "fresh-from-checkpoint",
                   "diagnostics_defined_after_step": start + 1, "inputs": staged}
        (stage / "resume.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        stage.rename(out)
    return out / "resume.json"
