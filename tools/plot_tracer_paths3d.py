#!/usr/bin/env python3
"""Render recorded 3D tracer legs in both transverse projections.

Prepared-case route (``--prepared MOLECULE_DIR OUT``): reads ``paths.csv`` from
``dsmc.py molecule --trace-count K``. Every leg is recorded, including the
terminating one, which already ends at the clipped contact. The input is
rejected unless each path is complete, ordered and ends at its ``terminal.csv``
row (``dsmc_molecule.read_recorded_paths``).

Legacy route (``STDOUT LEGS SURFS OUT``): the negative-index rows from
``--trajprint`` are individual non-terminal legs. The positive terminal row is
read from the matching ``legs.out`` and clipped against the actual 3D triangles
by ``tracer_analyze3d.clip_legs3d``.

Both draw recorded segments only; nothing is interpolated, smoothed or inferred.
"""
import argparse
import sys
import tomllib
from itertools import cycle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dsmc_molecule import read_recorded_paths
from tracer_analyze import COLS
from tracer_analyze3d import clip_legs3d, read_surfs3d


def trace_rows(stdout, count):
    rows = {pid: [] for pid in range(1, count + 1)}
    for line in Path(stdout).read_text().splitlines():
        words = line.split()
        if len(words) != len(COLS):
            continue
        try:
            index = int(words[0])
        except ValueError:
            continue
        pid = -index
        if index < 0 and pid in rows:
            rows[pid].append(np.asarray([float(value) for value in words[1:]], float))
    missing = [pid for pid, legs in rows.items() if not legs]
    if missing:
        raise ValueError("missing --trajprint legs for particle ids %s" % missing)
    return rows


def validate_rows(rows, terminal_collides):
    """Require every printed non-terminal leg through the terminal ordinal."""
    for pid, legs in rows.items():
        expected = int(terminal_collides[pid])
        ordinals = [int(leg[9]) for leg in legs]
        if len(legs) != expected or ordinals != list(range(1, expected + 1)):
            raise ValueError("particle %d has %d --trajprint legs with collision ordinals %s; "
                             "expected 1..%d through its terminal row" %
                             (pid, len(legs), ordinals[:4] + (["..."] if len(ordinals) > 4 else []), expected))


def terminal_rows(legs_path, pids, surfs):
    data = pd.read_csv(legs_path, sep=r"\s+")
    if list(data.columns) != COLS:
        raise ValueError("unexpected columns in %s" % legs_path)
    data = data.set_index("idx")
    if any(pid not in data.index for pid in pids):
        raise ValueError("legs.out lacks a terminal row for a plotted particle")
    terminal = data.loc[pids]
    terminal_collides = {pid: int(terminal.loc[pid, "collides"]) for pid in pids}
    tri, _, bounds, _ = read_surfs3d(surfs)
    start = terminal[["x", "y", "z"]].to_numpy()
    end = terminal[["xnext", "ynext", "znext"]].to_numpy()
    hit, code = clip_legs3d(start, end, tri, bounds)
    return ({pid: (start[i], hit[i], int(code[i])) for i, pid in enumerate(pids)},
            terminal_collides)


def legacy_paths(rows, terminal):
    """Legacy tracer (x,y,z) legs as solver (x,y,z) = tracer (z,x,y) segments."""
    solver = lambda p: np.asarray([p[2], p[0], p[1]])
    labels = {0: "unclipped endpoint", 1: "wall", 2: "domain boundary"}
    paths, fates = {}, {}
    for pid, legs in rows.items():
        start, hit, code = terminal[pid]
        paths[pid] = ([(solver(leg[:3]), solver(leg[3:6])) for leg in legs] +
                      [(solver(start), solver(hit))])
        fates[pid] = labels[code]
    return paths, fates


def load_prepared(molecule_dir):
    """Return ({id: [(start, end)] in solver xyz}, {id: fate}) after verification."""
    molecule_dir = Path(molecule_dir)
    result = tomllib.loads((molecule_dir / "transport-result.toml").read_text(encoding="utf-8"))
    count = result.get("recorded_paths", 0)
    if not count:
        raise ValueError("molecule run recorded no paths; rerun with --trace-count K")
    legs, fates = read_recorded_paths(molecule_dir / "paths.csv",
                                      molecule_dir / "terminal.csv", count)
    point = lambda row, suffix: np.asarray([float(row[k + suffix]) for k in "xyz"])
    return ({pid: [(point(row, "0_m"), point(row, "1_m")) for row in rows]
             for pid, rows in legs.items()}, fates)


def draw_segment(ax, start, end, transverse, color):
    # Solver coordinates: axial is x (index 0).
    ax.plot([start[0] * 1e3, end[0] * 1e3],
            [start[transverse] * 1e3, end[transverse] * 1e3],
            color=color, lw=0.65, alpha=0.8)


def path_colors(rows):
    return {pid: color for pid, color in zip(rows, cycle(["#0072B2", "#D55E00", "#009E73"]))}


def render(paths, fates, out, title):
    colors = path_colors(paths)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    views = ((1, "SPARTA y / tracer x (mm)", "horizontal"),
             (2, "SPARTA z / tracer y (mm)", "vertical"))
    for axis, (transverse, ylabel, name) in zip(axes, views):
        for pid, segments in paths.items():
            color = colors[pid]
            for start, end in segments:
                draw_segment(axis, start, end, transverse, color)
            start, hit = segments[0][0], segments[-1][1]
            axis.plot(start[0] * 1e3, start[transverse] * 1e3, "o",
                      ms=4, mfc="white", mec=color)
            axis.plot(hit[0] * 1e3, hit[transverse] * 1e3, "x", ms=5,
                      mew=1.3, color=color, label="particle %d (%s)" % (pid, fates[pid]))
        axis.set(title=name + " projection", xlabel="SPARTA x / tracer z (mm)", ylabel=ylabel)
        axis.set_aspect("equal")
        axis.legend(fontsize=8, loc="upper right")
    fig.suptitle(title)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s --prepared MOLECULE_DIR OUT | %(prog)s STDOUT LEGS SURFS OUT [--count N]")
    parser.add_argument("--prepared", metavar="MOLECULE_DIR",
                        help="dsmc molecule child directory run with --trace-count K")
    parser.add_argument("inputs", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--count", type=int, default=3, help="legacy route only")
    args = parser.parse_args(argv)
    if args.prepared:
        if len(args.inputs) != 1:
            parser.error("--prepared takes MOLECULE_DIR and one OUT path")
        out = args.inputs[0]
        paths, fates = load_prepared(args.prepared)
        title = "Recorded prepared-case legs including the clipped terminal leg; no inferred path"
    else:
        if len(args.inputs) != 4:
            parser.error("legacy route takes STDOUT LEGS SURFS OUT")
        stdout, legs, surfs, out = args.inputs
        if args.count <= 0:
            raise ValueError("--count must be positive")
        rows = trace_rows(stdout, args.count)
        terminal, terminal_collides = terminal_rows(legs, list(rows), surfs)
        validate_rows(rows, terminal_collides)
        paths, fates = legacy_paths(rows, terminal)
        title = "Recorded --trajprint legs plus actual 3D-clipped terminal leg; no inferred path"
    render(paths, fates, out, title)
    print("wrote %s from %d recorded paths" % (out, len(paths)))


if __name__ == "__main__":
    main()
