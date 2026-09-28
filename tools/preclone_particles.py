#!/usr/bin/env python3
"""Pre-clone a SPARTA particle dump for `read_particles` onto a refined grid.

`read_particles` does not re-weight (read_particles.cpp:206-219), so a particle
saved on level-1 cells and read into a level-k cell under `global weight cell
volume` represents 1/8^(k-1) of its atoms. This tool writes each row
V_level1 / V_cell times (the clone count `post_weight` would produce,
particle.cpp:583-611) with identical x and v and fresh ids 1..M, as the
`id type x y z vx vy vz` file the continuation path feeds to read_particles.

The refined grid is read from the deck that will read the particles: its
`create_box`, `region` (block/cylinder) and `create_grid ... levels ... region`
lines, replicated with SPARTA's 8-corner test (create_grid.cpp:640-670). A
corner within 1e-6 cell of a region boundary is ambiguous and refused. The
input file is only read.

    python tools/preclone_particles.py --deck in.helium-field \
        --input particles.1000000 --output initial-particles.1000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from dsmc_cad_flow import CORNER_EPS, region_margin


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_grid(deck: str) -> dict:
    """Box, level-1 grid and ordered refinement levels from a SPARTA deck."""
    regions, box, grid = {}, None, None
    for line in deck.splitlines():
        words = line.split("#", 1)[0].split()
        if not words:
            continue
        if words[0] == "create_box":
            v = [float(w) for w in words[1:7]]
            box = [(v[0], v[1]), (v[2], v[3]), (v[4], v[5])]
        elif words[0] == "region" and len(words) > 2 and words[2] in ("block", "cylinder"):
            args = words[3:]
            if len(args) != 6:
                raise ValueError(f"unsupported region options: {line.strip()}")
            regions[words[1]] = (words[2], ([args[0]] if words[2] == "cylinder" else []) +
                                 [float(a) for a in args[(words[2] == "cylinder"):]])
        elif words[0] == "create_grid":
            if grid is not None:
                raise ValueError("deck has more than one create_grid")
            grid = words[1:]
    if box is None or grid is None:
        raise ValueError("deck lacks create_box or create_grid")
    n, rest = [int(w) for w in grid[:3]], grid[3:]
    nlevels, levels, inside, i = 1, {}, "any", 0
    while i < len(rest):
        if rest[i] == "levels":
            nlevels, i = int(rest[i+1]), i + 2
        elif rest[i] == "region":
            level, rid, factors = int(rest[i+1]), rest[i+2], [int(w) for w in rest[i+3:i+6]]
            if rid not in regions:
                raise ValueError(f"create_grid refers to undefined block/cylinder region {rid}")
            levels[level], i = (*regions[rid], factors), i + 6
        elif rest[i] == "inside" and rest[i+1] in ("any", "all"):
            inside, i = rest[i+1], i + 2
        else:
            raise ValueError(f"unsupported create_grid keyword {rest[i]}")
    if sorted(levels) != list(range(2, nlevels + 1)):
        raise ValueError("every level 2..L needs exactly one single-level region keyword")
    return {"box": box, "n": n, "inside": inside,
            "levels": [levels[k] for k in range(2, nlevels + 1)]}


def cell_of(x, grid: dict) -> tuple[int, int]:
    """(SPARTA level, level-1 volume / cell volume) of the cell holding x."""
    lo, hi = [], []
    for (blo, bhi), n, xd in zip(grid["box"], grid["n"], x):
        h = (bhi - blo) / n
        i = min(max(int((xd - blo) // h), 0), n - 1)
        lo.append(blo + i * h)
        hi.append(blo + (i + 1) * h)
    level, ratio = 1, 1
    for style, args, factors in grid["levels"]:
        eps = CORNER_EPS * min(b - a for a, b in zip(lo, hi))
        margins = [region_margin(style, args, (a, b, c))
                   for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])]
        if any(abs(m) < eps for m in margins):
            raise ValueError(f"region boundary passes through a level-{level} corner near {x}")
        inside = [m >= 0 for m in margins]
        if not (any(inside) if grid["inside"] == "any" else all(inside)):
            break
        for d, c in enumerate(factors):
            h = (hi[d] - lo[d]) / c
            j = min(max(int((x[d] - lo[d]) // h), 0), c - 1)
            lo[d], hi[d] = lo[d] + j * h, lo[d] + (j + 1) * h
        level, ratio = level + 1, ratio * math.prod(factors)
    return level, ratio


def preclone(deck_path: Path, source: Path, target: Path) -> dict:
    deck_path, source, target = (Path(p).resolve() for p in (deck_path, source, target))
    if target.exists() or target == source:
        raise FileExistsError(f"refusing to write {target}")
    grid = parse_grid(deck_path.read_text(encoding="utf-8"))
    with source.open(encoding="utf-8") as f:
        header = [next(f) for _ in range(9)]
        columns = header[8].split()[2:]
        if (header[0].strip() != "ITEM: TIMESTEP" or header[2].strip() != "ITEM: NUMBER OF ATOMS"
                or columns not in (["x", "y", "z", "vx", "vy", "vz"],
                                   ["id", "type", "x", "y", "z", "vx", "vy", "vz"])):
            raise ValueError("unexpected SPARTA particle dump header")
        rows = [line.split() for line in f if line.strip()]
    if len(rows) != int(header[3]):
        raise ValueError("particle row count differs from the dump header")
    by_level, out_rows = {}, []
    for row in rows:
        if len(row) != len(columns):
            raise ValueError("particle row has the wrong column count")
        kind, values = (row[1] if len(columns) == 8 else "1"), row[-6:]
        x = [float(v) for v in values[:3]]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("non-finite particle row")
        level, copies = cell_of(x, grid)
        by_level.setdefault(level, [0, 0, copies])
        by_level[level][0] += 1
        by_level[level][1] += copies
        out_rows.extend([(kind, values)] * copies)
    header[3] = f"{len(out_rows)}\n"
    header[8] = "ITEM: ATOMS id type x y z vx vy vz\n"
    with target.open("x", encoding="utf-8", newline="\n") as f:
        f.writelines(header)
        f.writelines(f"{i} {kind} {' '.join(values)}\n"
                     for i, (kind, values) in enumerate(out_rows, 1))
    return {"schema_version": "preclone-particles-v1",
            "deck": str(deck_path), "deck_sha256": _sha(deck_path),
            "input": str(source), "input_sha256": _sha(source),
            "output": str(target), "output_sha256": _sha(target),
            "input_particles": len(rows), "output_particles": len(out_rows),
            "levels": {str(k): {"input": v[0], "output": v[1], "copies_each": v[2]}
                       for k, v in sorted(by_level.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preclone(args.deck, args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
