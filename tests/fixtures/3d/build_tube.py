"""Generate the tiny synthetic saved-field fixture used by 3D release checks.

This is a format/transport fixture, not a SPARTA result or B5 apparatus model.
Coordinates are metres; the source tube spans solver x=0..1 inside a larger
rectangular domain. Only Python's standard library is needed to regenerate it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path) -> None:
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"fixture output already exists: {out}")
    out.mkdir(parents=True)
    tracer = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0),
              (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
    triangles = [(1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3),
                 (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1),
                 (1, 2, 3), (1, 3, 4)]
    solver = [(p[2], p[0], p[1]) for p in tracer]
    surface = ["8 points", "10 triangles", "Points"]
    surface += [f"{i} {' '.join(map(str, p))}" for i, p in enumerate(solver, 1)]
    surface += ["Triangles"]
    surface += [f"{i} {1 if i <= 8 else 2} {' '.join(map(str, tri))}"
                for i, tri in enumerate(triangles, 1)]
    (out / "physical-walls.surf").write_text("\n".join(surface) + "\n", encoding="utf-8", newline="\n")
    records = [{"triangle_id": i, "boundary_id": "wall" if i <= 8 else "inlet_cap",
                "role": "wall" if i <= 8 else "inlet_cap", "molecule_action": "absorb"}
               for i in range(1, 11)]
    roles = {"schema_version": "dsmc-triangle-roles-v1",
             "surface_sha256": _sha(out / "physical-walls.surf"),
             "records": records, "fixture_only": True}
    (out / "triangle-roles.json").write_text(
        json.dumps(roles, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    (out / "triangle-roles.tsv").write_text(
        "triangle_id\tboundary_id\trole\tmolecule_action\n" +
        "".join(f"{r['triangle_id']}\t{r['boundary_id']}\t{r['role']}\tabsorb\n" for r in records),
        encoding="utf-8", newline="\n")
    header = ("ITEM: TIMESTEP\n0\nITEM: NUMBER OF SURFS\n10\n"
              "ITEM: BOX BOUNDS oo oo oo\n-1 2\n-2 2\n-2 2\n"
              "ITEM: SURFS id v1x v1y v1z v2x v2y v2z v3x v3y v3z\n")
    rows = []
    for i, tri in enumerate(triangles, 1):
        xyz = [coordinate for vertex in tri for coordinate in solver[vertex - 1]]
        rows.append(f"{i} {' '.join(map(str, xyz))}")
    (out / "cell.surfs").write_text(header + "\n".join(rows) + "\n",
                                      encoding="utf-8", newline="\n")
    canonical = {"schema_version": "dsmc-canonical-tracer3d-v1",
                 "canonical_surface_sha256": _sha(out / "physical-walls.surf"),
                 "triangle_roles_sha256": _sha(out / "triangle-roles.json"),
                 "tracer_surface_sha256": _sha(out / "cell.surfs"),
                 "tracer_roles_sha256": _sha(out / "triangle-roles.tsv"),
                 "triangle_count": 10}
    (out / "canonical-tracer.toml").write_text(
        "".join(f"{key} = {json.dumps(value)}\n" for key, value in canonical.items()),
        encoding="utf-8", newline="\n")
    (out / "DS3FF.DAT").write_text(
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF CELLS\n3\n"
        "ITEM: BOX BOUNDS oo oo oo\n-1 2\n-2 2\n-2 2\n"
        "ITEM: BASE GRID\n3 1 1\n"
        "ITEM: CELLS xc yc zc hx temp nrho u v w\n"
        "-0.5 0 0 0.5 4 1e15 0 0 0\n"
        "0.5 0 0 0.5 4 1e15 0 0 0\n"
        "1.5 0 0 0.5 4 1e15 0 0 0\n",
        encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
