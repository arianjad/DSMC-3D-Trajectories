#!/usr/bin/env python3
"""Write an unverified Slurm site profile and a solver-check launch profile for one root.

Tracked port of the package014 rehearsal's make-profiles.py. Run after
tools/engaging/setup-env.sh has created ROOT/python-venv and ROOT/config/setup.sh:

    python3 tools/engaging/write_profiles.py --root "$W" --account ACCOUNT \
        --partition PARTITION --solver /path/to/spa_mpi_bigbig

Writes ROOT/site-profile.json, ROOT/solver-check-profile.json and
ROOT/profile-inputs.json (hashes of both profiles and the solver). Refuses to
overwrite. Both profiles are checked with the maintained validators
(dsmc_slurm.validate_profile, dsmc_case.load_profile) before anything is written.
The site profile stays site_runtime_verified=false; `prepare_site.py probe`
inside an allocation produces the verified copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dsmc_case import load_profile  # noqa: E402
from dsmc_slurm import validate_profile  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(value: dict) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def profiles(root: Path, account: str | None, partition: str, solver: Path,
             ranks: int, site_id: str) -> tuple[dict, dict]:
    common = {"julia": "julia", "mpi_argv": ["mpirun", "-np", "{ranks}"],
              "python": str(root / "python-venv/bin/python"), "ranks": ranks, "threads": 1,
              "solver_executable": str(solver), "max_disk_gb": 40}
    site = {"schema_version": "dsmc-slurm-profile-v1", "site_id": site_id,
            "site_runtime_verified": False, "partition": partition, "account": account,
            "walltime": "00:30:00", "nodes": 1, "memory_gb": 32,
            "repository_root": str(root / "repo"), "setup_script": str(root / "config/setup.sh"),
            "scratch_root": str(root / "scratch"), "results_root": str(root / "results"),
            "gas_policy": "carry-T-and-empty-ballistic", **common}
    launch = {"schema_version": "dsmc-profile-v1", "profile_id": site_id + "-solver-check",
              "platform": "local", "max_memory_gb": 32,
              "scratch_root": str(root / "solver-check-scratch"),
              "results_root": str(root / "solver-check-results"), **common}
    return site, launch


def check(site: dict, launch: dict) -> None:
    validate_profile(dict(site))
    # The probe flips this flag; a placeholder path must fail at that point too.
    validate_profile({**site, "site_runtime_verified": True})
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "solver-check-profile.json"
        path.write_text(text(launch), encoding="utf-8")
        load_profile(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="working root holding repo/, python-venv/, config/")
    parser.add_argument("--account", required=True, help="Slurm account; 'none' omits --account")
    parser.add_argument("--partition", required=True)
    parser.add_argument("--solver", type=Path, required=True, help="SPARTA BIGBIG MPI executable")
    parser.add_argument("--ranks", type=int, default=4, help="MPI ranks (4 matches every passing B5 example)")
    parser.add_argument("--site-id", default="engaging-b5")
    args = parser.parse_args(argv)
    root, solver = args.root.resolve(), args.solver.resolve()
    missing = [p for p in (root / "repo", root / "python-venv/bin/python", root / "config/setup.sh")
               if not p.exists()]
    if args.root.is_symlink() or missing:
        raise SystemExit(f"root is incomplete (run setup-env.sh first): missing {[str(p) for p in missing]}")
    if not solver.is_file():
        raise SystemExit(f"solver is not a file: {solver}")
    account = None if args.account == "none" else args.account
    site, launch = profiles(root, account, args.partition, solver, args.ranks, args.site_id)
    check(site, launch)
    outputs = {name: root / name for name in
               ("site-profile.json", "solver-check-profile.json", "profile-inputs.json")}
    existing = [str(p) for p in outputs.values() if p.exists() or p.is_symlink()]
    if existing:
        raise SystemExit(f"refusing to overwrite {existing}")
    inputs = {"schema_version": "dsmc-engaging-profile-inputs-v1",
              "site_profile_sha256": hashlib.sha256(text(site).encode()).hexdigest(),
              "solver_check_profile_sha256": hashlib.sha256(text(launch).encode()).hexdigest(),
              "solver": str(solver), "solver_sha256": sha(solver),
              "account": account, "partition": args.partition,
              "ranks": args.ranks, "threads": 1, "memory_gb": 32, "max_disk_gb": 40}
    for name, value in (("site-profile.json", site), ("solver-check-profile.json", launch),
                        ("profile-inputs.json", inputs)):
        with outputs[name].open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text(value))
    for path in outputs.values():
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
