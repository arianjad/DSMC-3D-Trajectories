"""Export an exact split-cell map for one completed CAD helium frame.

The separately built diagnostic solver runs zero timesteps on one MPI rank.
The ordinary helium solver, its particles and its saved outputs are untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from dsmc_cad_workflow import validate_prepared_cad
from dsmc_cad_probe import one_rank_command
from dsmc_repository import sha
from sparta_splitmap.ids_from_frame import select_ids
from sparta_splitmap.validate_map import validate
from sparta_splitmap.make_queries import generate


def probe_deck(source: str) -> str:
    """Reuse the supported CAD setup through surface reads, omitting particles."""
    lines = source.splitlines()
    stops = [i for i, line in enumerate(lines) if line.startswith("stats ")]
    if len(stops) != 1:
        raise ValueError("CAD map probe needs one declared stats boundary")
    setup = lines[:stops[0]]
    forbidden = ("run ", "include ", "jump ", "shell ", "read_restart ")
    if any(line.startswith(forbidden) for line in setup):
        raise ValueError("CAD map probe setup contains an unsupported command")
    setup = [line for line in setup if not line.startswith(
        ("balance_grid ", "read_particles ", "create_particles "))]
    if sum(line.startswith("read_surf physical-walls.surf ") for line in setup) != 1:
        raise ValueError("CAD map probe lacks one canonical wall surface")
    return "\n".join([*setup, "run 0", ""])


def prepare_splitmap(prepared: Path, helium: Path, solver: Path, out: Path,
                     *, timestep: int | None = None) -> dict:
    prepared, helium, solver, out = (Path(p).resolve() for p in (prepared, helium, solver, out))
    if prepared.is_file():
        prepared = prepared.parent
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if any(out == source or out in source.parents or source in out.parents
           for source in (prepared, helium)):
        raise ValueError("split-map output must be separate from helium and preparation")
    data = validate_prepared_cad(prepared)
    parent = json.loads((helium / "manifest.json").read_text(encoding="utf-8"))
    if (parent.get("status") != "complete" or parent.get("case_id") != data["case_id"] or
            parent.get("prepared", {}).get("sha256") != sha(prepared / "prepared.json")):
        raise ValueError("split-map helium parent differs from CAD preparation")
    if not solver.is_file() or solver.is_symlink():
        raise ValueError("a separate native split-map diagnostic solver is required")
    if sha(solver) == data["solver_check"]["solver"]["sha256"]:
        raise ValueError("split-map diagnostic solver must be separate from ordinary helium")
    step = timestep if timestep is not None else parent.get(
        "end_step", data["science_inputs"]["numerics"]["total_steps"])
    frame_name = f"fields.{step}.gz"
    out.mkdir(parents=True)
    receipt = {"schema_version": "dsmc-cad-splitmap-probe-v1", "status": "started",
               "prepared_sha256": sha(prepared / "prepared.json"),
               "parent_manifest_sha256": sha(helium / "manifest.json"),
               "solver": {"path": str(solver), "sha256": sha(solver)},
               "timestep": step, "advanced_steps": 0, "ranks": 1}
    receipt_path = out / "splitmap-probe.json"
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    save()
    try:
        ids = out / "split-parent-ids.txt"
        selection = select_ids(helium, frame_name, ids, allow_empty=True)
        receipt["selection"] = selection
        if not selection["split_parent_ids"]:
            receipt["status"] = "not-needed"
            receipt["native_probe_executed"] = False
            receipt["artifacts"] = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
                                    for path in (ids, ids.with_suffix(ids.suffix + ".json"))]
            save()
            return receipt
        try:  # optional: the public release omits this SPARTA-derived parity check
            from sparta_splitmap.compare_queries import compare
        except ModuleNotFoundError as exc:
            raise RuntimeError("frame has split parents; exact split-map support "
                               "(tools/sparta_splitmap/compare_queries.py) is not installed") from exc
        source = (prepared / "in.helium").read_text(encoding="utf-8")
        deck = probe_deck(source)
        for item in data["inputs"]:
            name = item["staged"]["path"]
            if name in ("he.species", "he.vss") or name.endswith(".surf"):
                shutil.copyfile(prepared / name, out / name)
        (out / "in.splitmap").write_text(deck, encoding="utf-8", newline="\n")
        environment = os.environ.copy()
        environment.update(SPARTA_SPLITMAP_OUT=str(out / "splitmap.tsv"),
                           SPARTA_SPLITMAP_IDS=str(ids), OMP_NUM_THREADS="1")
        # A caller's old point-query paths must not affect this new map.
        environment.pop("SPARTA_SPLITMAP_QUERY", None)
        environment.pop("SPARTA_SPLITMAP_ANSWER", None)
        command = one_rank_command(solver, "in.splitmap")
        receipt["command"] = command
        with (out / "probe.stdout").open("wb") as log:
            receipt["native_probe_executed"] = True
            completed = subprocess.run(command, cwd=out,
                                       env=environment, stdout=log, stderr=subprocess.STDOUT)
        receipt["exit_code"] = completed.returncode
        if completed.returncode:
            raise ValueError(f"CAD split-map native probe failed: exit {completed.returncode}")
        receipt["validation"] = validate(helium, frame_name, out / "splitmap.tsv", ids)
        original_map_sha = sha(out / "splitmap.tsv")
        receipt["queries"] = generate(out / "splitmap.tsv", out / "queries.txt")
        environment.update(SPARTA_SPLITMAP_QUERY=str(out / "queries.txt"),
                           SPARTA_SPLITMAP_ANSWER=str(out / "answers.tsv"))
        with (out / "query-probe.stdout").open("wb") as log:
            queried = subprocess.run(command, cwd=out,
                                     env=environment, stdout=log, stderr=subprocess.STDOUT)
        if queried.returncode or sha(out / "splitmap.tsv") != original_map_sha:
            raise ValueError("split-map query probe failed or changed the map")
        receipt["point_parity"] = compare(out / "splitmap.tsv", out / "queries.txt", out / "answers.tsv")
        if receipt["point_parity"]["mismatch_count"]:
            raise ValueError("split-map point classifications differ from native SPARTA")
        receipt["artifacts"] = [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
                                for path in sorted(out.iterdir()) if path.is_file() and path != receipt_path]
        receipt["status"] = "complete"
    except BaseException as exc:
        receipt.update(status="failed", failure=str(exc))
        save()
        raise
    save()
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--helium", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timestep", type=int)
    args = parser.parse_args()
    result = prepare_splitmap(args.prepared, args.helium, args.solver, args.out,
                              timestep=args.timestep)
    print(json.dumps({key: result[key] for key in ("status", "timestep", "advanced_steps")}, indent=2))
