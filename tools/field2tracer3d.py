#!/usr/bin/env python3
"""Convert a 3D SPARTA grid dump (+ triangle .surf) into 3D tracer input.

Usage:
  python tools/field2tracer3d.py <3d-run-dir> --out <dir> [--dump fields.36000]
      [--canonical-surface <physical-walls.surf> --triangle-roles <triangle-roles.json>]
      [--geometry-only-roles]
      [--splitmap <solver-splitmap.tsv> --splitmap-ids <selected-ids.txt>]
  python tools/field2tracer3d.py --revolve <2d-run-dir> --surf <cell_*_3d.surf> --out <dir>
  python tools/field2tracer3d.py --synthetic T NRHO XLO XHI RMAX NX NR --out <dir>
  python tools/field2tracer3d.py --self-check

Emits <outdir>/DS3FF.DAT, <outdir>/cell.surfs, <outdir>/provenance.json.
Real 3D conversion also emits source-row-metadata.csv.gz with each retained
SPARTA row's emitted field row, bounds, ID, split index, cell volume and sample count. Its hash,
row count and source dump hash are recorded in provenance.json.

FILE FORMAT (read by tracer/Field3D.jl -- keep the two in sync)
---------------------------------------------------------------
DS3FF.DAT, 11 header lines then one row per GAS cell:

    ITEM: TIMESTEP / <step>
    ITEM: NUMBER OF CELLS / <n>
    ITEM: BOX BOUNDS oo oo oo / <xlo xhi> / <ylo yhi> / <zlo zhi>
    ITEM: BASE GRID / <nx ny nz>
    ITEM: CELLS xc yc zc hx temp nrho u v w
    ...

All coordinates are SPARTA's: x axial, (y,z) transverse.  The tracer's own
axes are (x1,x2,x3) = (y_sparta, z_sparta, x_sparta); Field3D.jl does that
permutation on load, exactly as the 2D path does (DS2FF column 1 = axial).

`hx` is the cell HALF-size along x.  The transverse half-sizes are not
written: SPARTA refines 2/2/2, so hy = hy_base * (hx/hx_base) and likewise
hz, and the converter ASSERTS that on every emitted cell.  Given the box,
the base grid and hx, a cell's refinement level and its integer lattice
index are exact, which is what lets Field3D.jl locate the containing cell in
O(levels) dictionary probes instead of guessing with a nearest-centre query.

cell.surfs is the same 9-line header the 2D writer emits, with

    ITEM: SURFS id v1x v1y v1z v2x v2y v2z v3x v3y v3z

By default only the run's own cell*.surf becomes tracer geometry. The optional
canonical pair uses the solver-clipped surface and its hash-bound role sidecar;
it writes the role-preserving tracer derivative and manifest. The transparent
ap_*3d.surf / tube_*3d.surf station files sit in the same directory and the
tracer kills a particle on every surface it is handed, so they are excluded
here for exactly the reason field2tracer.py excludes their 2D twins.
`--geometry-only-roles` explicitly permits a geometry-only sidecar whose
molecule actions are all unspecified. It validates field/surface structure but
the resulting bundle cannot be used for molecule transport.

TRAPS THIS TOOL EXISTS TO HANDLE
--------------------------------
  - The 3D dump's column 1 (`idstr`) is a STRING like "307053-4-8-5-1-6" and
    column 2 (`split`) an int, so a whole-file float parse raises.  Required
    columns are selected by their declared header names, then validated with
    `idstr` retained as text.
  - The dump is ~570 MB / 4.56 M cells; it is read in chunks and filtered to
    gas (T > 0, matching the vendored tracer's own filter) as it streams.
    Empty zero-volume solid rows are valid; gas values in zero volume fail.
    Tiny negative sampled-temperature cancellation values within 1e-9 K are
    recorded as nonpositive-T sentinels, never silently promoted to gas.
    --keep-unsampled carries temperature only into positive-density rows;
    --keep-zero-density retains empty positive-volume zero-moment rows as
    explicit ballistic cells. Both are recorded model choices.
  - Prepared cases pass their declared base grid via --base-grid. Inferring it
    from retained gas cells can pick a refined transverse width and fail the
    exact nested-lattice center check.
  - Split cells can share one bounding box. Without an explicit solver map,
    only rows with exactly equal decoded n/T/u/v/w/nrho values may share a
    lookup value; differing rows fail. With a receipt-bound solver split3d
    map, every retained split gas row stays distinct and gets an exact
    point-classification binding. Source-row inventory metadata is retained.
  - The revolve mode samples the 2D field by EXACT containing-cell lookup on
    the 2D adaptive lattice, not by nearest centre: the 2D grid spans 400 ->
    25 um, and a nearest-centre sample straddles refinement jumps.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HE_MASS = 6.6464731e-27  # filler, matches tools/field2tracer.py; tracer drops it


def fmt(v):
    return repr(float(v))


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
                              check=True).stdout.strip()
    except Exception:
        return None


# ----------------------------------------------------------------- geometry

def read_tris(path):
    """(points (P,3), tris (T,3) vertex indices, types (T,)) from a SPARTA 3D .surf."""
    with open(path, encoding="utf-8") as stream:
        lines = stream.read().splitlines()
    ip = lines.index("Points")
    it = lines.index("Triangles")
    pts = np.array([[float(t) for t in s.split()[1:4]]
                    for s in lines[ip + 1:it] if s.strip()])
    rec = [s.split() for s in lines[it + 1:] if s.strip()]
    tri = np.array([[int(r[2]), int(r[3]), int(r[4])] for r in rec]) - 1
    typ = np.array([int(r[1]) for r in rec])
    return pts, tri, typ


def write_surfs(outdir, step, box, verts):
    """verts: (T,3,3) triangle vertices in SPARTA coords."""
    rows = ["%d %s" % (i + 1, " ".join(fmt(c) for c in v.reshape(-1)))
            for i, v in enumerate(verts)]
    with open(os.path.join(outdir, "cell.surfs"), "w") as f:
        f.write(header(step, len(rows), box, "SURFS",
                       "SURFS id v1x v1y v1z v2x v2y v2z v3x v3y v3z"))
        f.write("\n".join(rows) + "\n")
    return len(rows)


def write_canonical_surfs(outdir, step, box, surface, roles,
                          allow_geometry_only_roles=False):
    """Convert solver-clipped triangles with their hash-bound molecule roles.

    The emitted TSV is a Julia-readable derivative. The JSON remains the
    authority and is copied byte-for-byte alongside it for provenance.
    """
    from pathlib import Path
    import shutil
    try:
        from .dsmc_geometry import _surface
    except ImportError:
        from dsmc_geometry import _surface

    surface, roles = Path(surface), Path(roles)
    role_data = json.loads(roles.read_text(encoding="utf-8"))
    surface_sha = hashlib.sha256(surface.read_bytes()).hexdigest()
    role_schema = role_data.get("schema_version")
    geometry_only = role_schema == "dsmc-triangle-roles-geometry-only-v1"
    if role_schema != "dsmc-triangle-roles-v1" and not (
            allow_geometry_only_roles and geometry_only):
        raise ValueError("unsupported canonical triangle-role schema")
    if role_data.get("surface_sha256") != surface_sha:
        raise ValueError("canonical surface hash differs from triangle roles")
    points, triangles = _surface(surface)
    if not triangles:
        raise ValueError("canonical surface has no triangles")
    pts = np.asarray(points, dtype=float)
    tri = np.asarray([record[1:] for record in triangles], dtype=int) - 1
    verts = pts[tri]
    if not np.isfinite(verts).all() or np.any(
            np.linalg.norm(np.cross(verts[:, 1]-verts[:, 0],
                                    verts[:, 2]-verts[:, 0]), axis=1) == 0):
        raise ValueError("nonfinite or degenerate canonical triangle")
    records = role_data.get("records")
    if not isinstance(records, list) or len(records) != len(tri):
        raise ValueError("canonical roles must cover every triangle")
    indexed = {}
    for rec in records:
        tid = rec.get("triangle_id")
        if type(tid) is not int or tid in indexed or not 1 <= tid <= len(tri):
            raise ValueError("invalid or duplicate canonical triangle ID")
        allowed_actions = (("unspecified",) if geometry_only else
                           ("absorb", "exit", "ignore"))
        if rec.get("molecule_action") not in allowed_actions:
            raise ValueError("unsupported canonical molecule action")
        for key in ("boundary_id", "role"):
            value = rec.get(key)
            if not isinstance(value, str) or not value or any(c in value for c in "\t\r\n"):
                raise ValueError(f"invalid canonical {key}")
        indexed[tid] = rec
    if set(indexed) != set(range(1, len(tri)+1)):
        raise ValueError("canonical role IDs have gaps")
    os.makedirs(outdir, exist_ok=True)
    write_surfs(outdir, step, box, verts)
    emitted = np.loadtxt(os.path.join(outdir, "cell.surfs"), skiprows=9, ndmin=2)
    converted = emitted[:, 1:].reshape((-1, 3, 3))
    # The oriented vertex order matters; this is a multiset comparison after
    # the same solver -> tracer axis permutation used by Field3D.jl.
    oriented = lambda array: Counter(map(tuple, array[:, :, [1, 2, 0]].reshape(-1, 9)))
    if oriented(converted) != oriented(verts):
        raise ValueError("solver/tracer oriented triangle multisets differ")
    with open(os.path.join(outdir, "triangle-roles.tsv"), "w", encoding="utf-8", newline="\n") as stream:
        stream.write("triangle_id\tboundary_id\trole\tmolecule_action\n")
        for tid in range(1, len(tri)+1):
            rec = indexed[tid]
            stream.write(f"{tid}\t{rec['boundary_id']}\t{rec['role']}\t{rec['molecule_action']}\n")
    dest = Path(outdir) / "triangle-roles.json"
    if roles.resolve() != dest.resolve():
        shutil.copyfile(roles, dest)
    canonical_dest = Path(outdir) / "physical-walls.surf"
    if surface.resolve() != canonical_dest.resolve():
        shutil.copyfile(surface, canonical_dest)
    tracer_surface = Path(outdir) / "cell.surfs"
    tracer_roles = Path(outdir) / "triangle-roles.tsv"
    with open(Path(outdir) / "canonical-tracer.toml", "w", encoding="utf-8", newline="\n") as stream:
        stream.write('schema_version = "dsmc-canonical-tracer3d-v1"\n')
        if geometry_only:
            stream.write('molecule_roles_state = "geometry-only"\n')
        stream.write(f'canonical_surface_sha256 = "{surface_sha}"\n')
        stream.write(f'triangle_roles_sha256 = "{hashlib.sha256(roles.read_bytes()).hexdigest()}"\n')
        stream.write(f'tracer_surface_sha256 = "{hashlib.sha256(tracer_surface.read_bytes()).hexdigest()}"\n')
        stream.write(f'tracer_roles_sha256 = "{hashlib.sha256(tracer_roles.read_bytes()).hexdigest()}"\n')
        stream.write(f'triangle_count = {len(tri)}\n')
    result = {"canonical_surface_sha256": surface_sha,
              "triangle_roles_sha256": hashlib.sha256(roles.read_bytes()).hexdigest(),
              "canonical_triangle_count": len(tri)}
    if geometry_only:
        result["molecule_roles_state"] = "geometry-only"
    return result


# -------------------------------------------------------------------- field

def header(step, n, box, kind, cols):
    return "\n".join([
        "ITEM: TIMESTEP", str(step),
        "ITEM: NUMBER OF %s" % kind, str(n),
        "ITEM: BOX BOUNDS oo oo oo",
        "%s %s" % (fmt(box[0][0]), fmt(box[0][1])),
        "%s %s" % (fmt(box[1][0]), fmt(box[1][1])),
        "%s %s" % (fmt(box[2][0]), fmt(box[2][1])),
        "ITEM: %s" % cols,
    ]) + "\n"


def write_field(outdir, step, box, base, cells):
    """cells: dict of equal-length 1D arrays xc yc zc hx temp nrho u v w."""
    os.makedirs(outdir, exist_ok=True)
    order = ["xc", "yc", "zc", "hx", "temp", "nrho", "u", "v", "w"]
    M = np.column_stack([np.asarray(cells[k], float) for k in order])
    path = os.path.join(outdir, "DS3FF.DAT")
    head = "\n".join([
        "ITEM: TIMESTEP", str(step),
        "ITEM: NUMBER OF CELLS", str(M.shape[0]),
        "ITEM: BOX BOUNDS oo oo oo",
        "%s %s" % (fmt(box[0][0]), fmt(box[0][1])),
        "%s %s" % (fmt(box[1][0]), fmt(box[1][1])),
        "%s %s" % (fmt(box[2][0]), fmt(box[2][1])),
        "ITEM: BASE GRID", "%d %d %d" % tuple(base),
        "ITEM: CELLS xc yc zc hx temp nrho u v w",
    ]) + "\n"
    with open(path, "w") as f:
        f.write(head)
        # Preserve round-trippable binary64 values in newly emitted fields.
        # Legacy files used approximately six significant digits and are
        # accepted by the bounded representation tolerance in lattice_check.
        np.savetxt(f, M, fmt="%.17g", delimiter=" ")
    return M.shape[0]


def read_field(path):
    """Inverse of write_field: (step, box, base, dict of arrays). Used by --self-check."""
    with open(path) as f:
        head = [next(f).rstrip("\n") for _ in range(11)]
    assert head[0] == "ITEM: TIMESTEP" and head[8] == "ITEM: BASE GRID", head
    step = int(head[1])
    box = [tuple(float(t) for t in head[k].split()) for k in (5, 6, 7)]
    base = tuple(int(t) for t in head[9].split())
    M = np.loadtxt(path, skiprows=11, ndmin=2)
    keys = head[10].split()[2:]
    return step, box, base, {k: M[:, i] for i, k in enumerate(keys)}


# SPARTA dumps coordinates with %g -- 6 significant digits -- so a size read
# back as a difference of two printed corners carries ~1e-3 relative error at
# the finest level.  Every lattice test below is therefore a rounding test
# with a loose tolerance, never an equality, and the exact sizes are always
# reconstructed from the box and the base grid rather than trusted from file.
LATTOL = 5e-3
# Legacy SPARTA dumps use %g (roughly six significant digits).  A valid centre
# can therefore miss its exact nested-lattice value slightly after its printed
# corners are decoded.  Bound that representation allowance both relative to
# the coordinate scale and as a small fraction of the local cell width.  Do
# not snap centres: the emitted value remains the decoded source value.
CENTER_COORD_RTOL = 5e-6
CENTER_CELL_FRAC_CAP = 5e-3

RAW_COLUMN_ALIASES = {
    "idstr": "idstr", "split": "split",
    "xlo": "xlo", "ylo": "ylo", "zlo": "zlo",
    "xhi": "xhi", "yhi": "yhi", "zhi": "zhi", "vol": "vol",
    "f_afields[1]": "n", "f_afields[2]": "nrho",
    "f_afields[3]": "u", "f_afields[4]": "v", "f_afields[5]": "w",
    "f_afields[9]": "T",
}
BOUND_COLUMNS = ["xlo", "ylo", "zlo", "xhi", "yhi", "zhi"]
EQUIVALENT_FIELD_COLUMNS = ["n", "nrho", "u", "v", "w", "T"]


def levels_of(h, h0, name):
    lev = np.rint(np.log2(h0 / h)).astype(int)
    if np.any(lev < 0) or np.any(np.abs(h0 / (h * 2.0 ** lev) - 1) > LATTOL):
        sys.exit("%s: half-size is not h_base/2^L (worst %.3g)"
                 % (name, np.abs(h0 / (h * 2.0 ** lev) - 1).max()))
    return lev


def lattice_check(box, base, centers, hx, name="cells"):
    """Require nested 2x sizes and centres; return each cell's level.

    ``centers`` is N x 3 in SPARTA coordinates.  ``None`` retains the old
    size-only utility behavior for callers that genuinely have no centres.
    """
    hx0 = 0.5 * (box[0][1] - box[0][0]) / base[0]
    lev = levels_of(hx, hx0, name)
    if centers is None:
        return lev
    centers = np.asarray(centers, float)
    if centers.ndim != 2 or centers.shape != (len(lev), 3):
        sys.exit("%s: expected N x 3 cell centres, got %s" % (name, centers.shape))
    eps = np.finfo(float).eps
    for axis, label in enumerate("xyz"):
        lo, hi = box[axis]
        h0 = 0.5 * (hi - lo) / base[axis]
        width = 2.0 * h0 / 2.0 ** lev
        q = (centers[:, axis] - lo) / width - 0.5
        offset = np.abs(q - np.rint(q)) * width
        scale = np.maximum.reduce([
            np.abs(centers[:, axis]), np.full(len(lev), abs(lo)),
            np.full(len(lev), abs(hi)), width,
        ])
        tolerance = np.minimum(CENTER_COORD_RTOL * scale,
                               CENTER_CELL_FRAC_CAP * width)
        tolerance += 64.0 * eps * np.maximum(scale, 1.0)
        bad = offset > tolerance
        if bad.any():
            j = int(np.flatnonzero(bad)[0])
            sys.exit(
                "%s: %d %s centres are off the nested lattice; first row %d "
                "offset %.9g m exceeds representation tolerance %.9g m "
                "(%.6g of cell width)"
                % (name, int(bad.sum()), label, j, offset[j], tolerance[j],
                   offset[j] / width[j]))
    return lev


def _open_dump(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") \
        else open(path, encoding="utf-8")


def validate_source_volumes(volume, n, nrho, temperature, path):
    """Allow SPARTA's empty zero-volume solid rows; reject gas in zero volume."""
    if np.any(volume < 0):
        sys.exit("%s: cell volumes must be nonnegative" % path)
    zero = volume == 0
    if np.any(zero & ((n != 0) | (nrho != 0) | (temperature != 0))):
        sys.exit("%s: zero-volume solid row carries gas samples, density or temperature" % path)
    return int(zero.sum())


def validate_source_thermodynamics(n, nrho, temperature, path,
                                   negative_temperature_tolerance_k=1e-9):
    """Recognize bounded moment-cancellation noise without making it gas."""
    if np.any(n < 0) or np.any(nrho < 0) or np.any(temperature < -negative_temperature_tolerance_k):
        sys.exit("%s: n, nrho or T falls below its physical/numerical bound" % path)
    return int(np.count_nonzero(temperature < 0))


def zero_density_field_rows(volume, n, nrho, u, v, w, temperature, path):
    """Identify explicit empty positive-volume cells for ballistic transport."""
    empty = (volume > 0) & (nrho == 0)
    if np.any(empty & ((n != 0) | (u != 0) | (v != 0) | (w != 0) |
                       (temperature != 0))):
        sys.exit("%s: zero-density cell has nonzero samples, velocity or temperature" % path)
    return empty


def _read_3d_header(path):
    try:
        with _open_dump(path) as f:
            head = [next(f).rstrip("\r\n") for _ in range(9)]
    except (OSError, EOFError, StopIteration, UnicodeError) as exc:
        sys.exit("%s: incomplete or unreadable field header: %s" % (path, exc))
    required = {
        0: "ITEM: TIMESTEP", 2: "ITEM: NUMBER OF CELLS",
        4: "ITEM: BOX BOUNDS", 8: "ITEM: CELLS",
    }
    for i, marker in required.items():
        if not head[i].startswith(marker):
            sys.exit("%s: expected %r on header line %d, got %r"
                     % (path, marker, i + 1, head[i]))
    try:
        step, ncells = int(head[1]), int(head[3])
        box = [tuple(float(t) for t in head[k].split()) for k in (5, 6, 7)]
    except (TypeError, ValueError) as exc:
        sys.exit("%s: invalid field header value: %s" % (path, exc))
    if step < 0 or ncells < 0:
        sys.exit("%s: timestep and declared row count must be nonnegative" % path)
    if any(len(b) != 2 for b in box) or not np.isfinite(np.asarray(box)).all() \
            or any(lo >= hi for lo, hi in box):
        sys.exit("%s: box bounds must be finite ordered pairs, got %s" % (path, box))
    tokens = head[8].split()
    if tokens[:2] != ["ITEM:", "CELLS"]:
        sys.exit("%s: malformed ITEM: CELLS header" % path)
    cols = tokens[2:]
    if len(cols) != len(set(cols)):
        sys.exit("%s: duplicate field column names" % path)
    missing = [c for c in RAW_COLUMN_ALIASES if c not in cols]
    if missing:
        sys.exit("%s: missing required field columns: %s" % (path, ", ".join(missing)))
    return step, ncells, box, cols


def _collapse_equivalent_boxes(d):
    """Collapse only exact-bound, exactly field-equivalent lookup rows."""
    dup = d.duplicated(BOUND_COLUMNS, keep=False)
    if not dup.any():
        return d, []
    metadata = []
    for bounds, group in d.loc[dup].groupby(BOUND_COLUMNS, sort=False, dropna=False):
        values = group[EQUIVALENT_FIELD_COLUMNS].to_numpy(dtype=float)
        equal = np.all(values == values[0], axis=0)
        if not equal.all():
            differing = [c for c, same in zip(EQUIVALENT_FIELD_COLUMNS, equal) if not same]
            ids = ["%s/split=%d" % (row.idstr, int(row.split))
                   for row in group[["idstr", "split"]].itertuples(index=False)]
            sys.exit(
                "unsupported multiple gas rows in one bounding box %s: differing %s "
                "for source rows %s; distinct split subcells need exact split-map support "
                "(--splitmap/--splitmap-ids and tracer/SplitField3D.jl)"
                % (tuple(float(v) for v in bounds), ", ".join(differing), ", ".join(ids)))
        rows = [{
            "idstr": str(row.idstr), "split": int(row.split),
            "volume": float(row.vol), "sample_count": float(row.n),
        } for row in group[["idstr", "split", "vol", "n"]].itertuples(index=False)]
        metadata.append({
            "bounds": [float(v) for v in bounds], "source_rows": rows,
            "source_volume": float(group["vol"].sum()),
            "source_inventory": float((group["nrho"] * group["vol"]).sum()),
        })
    return d.drop_duplicates(BOUND_COLUMNS, keep="first").reset_index(drop=True), metadata


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_source_row_metadata(outdir, rows, preserve_rows=False):
    """Keep each retained SPARTA subcell's bounds, ID, volume and samples.

    One-based field_row refers to DS3FF rows after equivalent-box collapse.
    Multiple source rows with one key remain separate here; the provenance
    also lists the equivalent groups explicitly.
    """
    source_columns = BOUND_COLUMNS + ["idstr", "split", "vol", "n"]
    if not all(column in rows for column in source_columns):
        missing = [column for column in source_columns if column not in rows]
        raise ValueError("source metadata lacks columns: %s" % missing)
    if preserve_rows:
        field_row = np.arange(1, len(rows) + 1)
    elif rows.duplicated(BOUND_COLUMNS).any():
        field_row = rows.groupby(BOUND_COLUMNS, sort=False, dropna=False).ngroup().to_numpy() + 1
    else:
        field_row = np.arange(1, len(rows) + 1)
    data = rows[source_columns].copy()
    data.insert(0, "field_row", field_row)
    columns = ["field_row"] + source_columns
    path = os.path.join(outdir, "source-row-metadata.csv.gz")
    os.makedirs(outdir, exist_ok=True)
    data.to_csv(path, index=False, compression="gzip", float_format="%.17g")
    return dict(path=os.path.basename(path), sha256=_file_sha256(path),
                bytes=os.path.getsize(path), rows=int(len(rows)),
                columns=columns, key="one-based emitted DS3FF field_row",
                semantics="retained SPARTA source rows before equivalent-box collapse")


# --------------------------------------------------------- mode A: real 3D

def convert3d(rundir, outdir, dumpname=None, chunk=400000, keep_unsampled=False,
              canonical_surface=None, triangle_roles=None, base_grid=None,
              keep_zero_density=False, allow_geometry_only_roles=False,
              splitmap=None, splitmap_ids=None):
    if (canonical_surface is None) != (triangle_roles is None):
        raise ValueError("canonical surface and triangle roles must be supplied together")
    if allow_geometry_only_roles and canonical_surface is None:
        raise ValueError("geometry-only roles require a canonical surface and sidecar")
    import pandas as pd
    if dumpname is None:
        cands = sorted((f for f in os.listdir(rundir) if f.startswith("fields.")),
                       key=lambda s: int(s.split(".")[1]))
        if not cands:
            sys.exit("no fields.* dump in %s" % rundir)
        dumpname = cands[-1]
    path = os.path.join(rundir, dumpname)
    step, ncells, box, cols = _read_3d_header(path)
    splitmap_validation = None
    split_parents = None
    if (splitmap is None) != (splitmap_ids is None):
        raise ValueError("exact split lookup needs both --splitmap and --splitmap-ids")
    if splitmap is not None:
        from sparta_splitmap.validate_map import load_map, validate as validate_splitmap
        splitmap_validation = validate_splitmap(
            Path(rundir), dumpname, Path(splitmap), Path(splitmap_ids))
        split_parents = load_map(Path(splitmap))
    source_names = list(RAW_COLUMN_ALIASES)
    names = [RAW_COLUMN_ALIASES[c] for c in source_names]
    keep, tot_inv, drop_inv, nsplit, nrow = [], 0.0, 0.0, 0, 0
    zero_volume_rows = 0
    near_zero_negative_temperature_rows = 0
    zero_density_rows = 0
    dxmax = 0.0
    numeric = [c for c in names if c != "idstr"]
    try:
        # Name every physical source column before selecting by name.  Pandas
        # returns integer ``usecols`` in file-position order, which silently
        # mislabeled values when a valid SPARTA header reordered fields.
        chunks = pd.read_csv(path, sep=r"\s+", skiprows=9, header=None,
                             names=cols, usecols=source_names, chunksize=chunk,
                             engine="c", dtype={"idstr": str})
        for ch in chunks:
            ch = ch.rename(columns=RAW_COLUMN_ALIASES)
            nrow += len(ch)
            vals = ch[numeric].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                row, col = np.argwhere(~np.isfinite(vals))[0]
                sys.exit("%s: nonfinite %s at data row %d"
                         % (path, numeric[int(col)], nrow - len(ch) + int(row) + 1))
            if ch["idstr"].isna().any() or (ch["idstr"].str.len() == 0).any():
                sys.exit("%s: empty idstr in field rows" % path)
            split = ch["split"].to_numpy(dtype=float)
            if np.any(split < 0) or np.any(split != np.rint(split)):
                sys.exit("%s: split indices must be nonnegative integers" % path)
            lo = ch[["xlo", "ylo", "zlo"]].to_numpy(dtype=float)
            hi = ch[["xhi", "yhi", "zhi"]].to_numpy(dtype=float)
            if np.any(hi <= lo):
                sys.exit("%s: cell bounds must have high > low" % path)
            zero_volume_rows += validate_source_volumes(
                ch["vol"].to_numpy(dtype=float), ch["n"].to_numpy(dtype=float),
                ch["nrho"].to_numpy(dtype=float), ch["T"].to_numpy(dtype=float), path)
            near_zero_negative_temperature_rows += validate_source_thermodynamics(
                ch["n"].to_numpy(dtype=float), ch["nrho"].to_numpy(dtype=float),
                ch["T"].to_numpy(dtype=float), path)
            for axis in range(3):
                scale = max(abs(box[axis][0]), abs(box[axis][1]),
                            box[axis][1] - box[axis][0])
                tol = CENTER_COORD_RTOL * scale
                if np.any(lo[:, axis] < box[axis][0] - tol) or \
                        np.any(hi[:, axis] > box[axis][1] + tol):
                    sys.exit("%s: cell bounds extend outside the declared box" % path)
            dxmax = max(dxmax, float((ch["xhi"] - ch["xlo"]).max()))
            tot_inv += float((ch["nrho"] * ch["vol"]).sum())
            # GAS FILTER, and it is a deletion, not a selection.  SPARTA's
            # temperature compute emits a HARD zero wherever the pooled sample
            # count is <= 1.  The legacy T>0 policy remains the default;
            # --keep-unsampled instead retains positive-density rows and carries
            # T from the nearest sampled centre.  This choice changes the field
            # model and is recorded below.
            g = (ch["nrho"] if keep_unsampled else ch["T"]).to_numpy() > 0
            if keep_zero_density:
                empty = zero_density_field_rows(
                    ch["vol"].to_numpy(dtype=float), ch["n"].to_numpy(dtype=float),
                    ch["nrho"].to_numpy(dtype=float), ch["u"].to_numpy(dtype=float),
                    ch["v"].to_numpy(dtype=float), ch["w"].to_numpy(dtype=float),
                    ch["T"].to_numpy(dtype=float), path)
                zero_density_rows += int(empty.sum())
                g |= empty
            drop_inv += float((ch["nrho"].to_numpy()[~g] * ch["vol"].to_numpy()[~g]).sum())
            nsplit += int((ch["split"].to_numpy()[g] != 0).sum())
            keep.append(ch[g])
    except SystemExit:
        raise
    except Exception as exc:
        sys.exit("%s: failed while reading field rows (truncated/corrupt stream): %s"
                 % (path, exc))
    if nrow != ncells:
        sys.exit("%s: declared %d field rows but decoded %d (truncated or extra data)"
                 % (path, ncells, nrow))
    if not keep:
        sys.exit("%s: declared field contains no data rows" % path)
    d = pd.concat(keep, ignore_index=True)
    del keep
    if d.empty:
        sys.exit("%s: no gas rows survive the %s field policy"
                 % (path, "nrho>0/temperature-carry" if keep_unsampled else "legacy T>0"))

    retained_source_rows = int(len(d))
    gas_inv = float((d["nrho"] * d["vol"]).sum())
    retained_source_volume = float(d["vol"].sum())
    source_rows = d
    if split_parents is None:
        d, duplicate_metadata = _collapse_equivalent_boxes(d)
    else:
        duplicate_metadata = []
        split_rows = d.loc[d["split"] > 0, ["idstr", "split"]]
        if split_rows.duplicated().any():
            raise ValueError("duplicate SPARTA parent/split row in retained gas")
        for idstr, split in split_rows.itertuples(index=False):
            parent = split_parents.get(str(idstr))
            if parent is None or not 1 <= int(split) <= parent["nsplit"]:
                raise ValueError(f"retained split row lacks solver map: {idstr}/{split}")
        for _, group in d.loc[d.duplicated(BOUND_COLUMNS, keep=False)].groupby(
                BOUND_COLUMNS, sort=False, dropna=False):
            if (group["idstr"].nunique() != 1 or
                    (group["split"] <= 0).any()):
                raise ValueError("duplicate Cartesian bounds do not share one split parent")
    source_metadata = write_source_row_metadata(
        outdir, source_rows, preserve_rows=split_parents is not None)
    del source_rows
    ndup = retained_source_rows - len(d)

    # carry T into the kept-but-unsampled cells from the nearest sampled
    # centre; their own n, u, v, w are real tallies and are left alone
    nfilled = 0
    if keep_unsampled:
        from scipy.spatial import cKDTree
        T = d["T"].to_numpy().astype(float).copy()
        density = d["nrho"].to_numpy(dtype=float)
        bad = (T <= 0) & (density > 0)
        donors = (T > 0) & (density > 0)
        nfilled = int(bad.sum())
        if nfilled and not donors.any():
            sys.exit("%s: --keep-unsampled retained %d rows but no positive-T row "
                     "exists from which to carry temperature" % (path, nfilled))
        if nfilled:
            C = np.column_stack([(d["xlo"] + d["xhi"]).to_numpy(),
                                 (d["ylo"] + d["yhi"]).to_numpy(),
                                 (d["zlo"] + d["zhi"]).to_numpy()]) * 0.5
            T[bad] = T[donors][cKDTree(C[donors]).query(C[bad], k=1)[1]]
            d = d.assign(T=T)

    hx = 0.5 * (d["xhi"] - d["xlo"]).to_numpy()
    hy = 0.5 * (d["yhi"] - d["ylo"]).to_numpy()
    hz = 0.5 * (d["zhi"] - d["zlo"]).to_numpy()
    if base_grid is None:
        # Compatibility fallback. Retained gas cells need not include a base
        # cell in every axis, so new prepared cases pass their declared grid.
        base = (int(round((box[0][1] - box[0][0]) / dxmax)),
                int(round((box[1][1] - box[1][0]) / (2 * hy[np.argmax(hx)]))),
                int(round((box[2][1] - box[2][0]) / (2 * hz[np.argmax(hx)]))))
        base_source = "inferred-from-retained-gas"
    else:
        if len(base_grid) != 3 or any(type(value) is not int or value <= 0 for value in base_grid):
            raise ValueError("base_grid must contain three positive integers")
        base = tuple(base_grid)
        base_source = "declared-case"
    centers = np.column_stack([
        0.5 * (d["xlo"] + d["xhi"]).to_numpy(),
        0.5 * (d["ylo"] + d["yhi"]).to_numpy(),
        0.5 * (d["zlo"] + d["zhi"]).to_numpy(),
    ])
    lev = lattice_check(box, base, centers, hx, "3D dump")
    # SPARTA prints cell corners with about six significant digits.  The
    # decoded difference passes the source-lattice check above, but can fail
    # the stricter downstream half-width invariant.  Emit the checked exact
    # lattice size; retain the source corners and volume in row metadata.
    exact_hx = (0.5 * (box[0][1] - box[0][0]) / base[0]) / 2.0 ** lev
    source_hx_max_rel_delta = float(np.max(np.abs(hx / exact_hx - 1.0)))
    hy0 = 0.5 * (box[1][1] - box[1][0]) / base[1]
    hz0 = 0.5 * (box[2][1] - box[2][0]) / base[2]
    bad = ((levels_of(hy, hy0, "3D dump y") != lev) |
           (levels_of(hz, hz0, "3D dump z") != lev))
    if bad.any():
        sys.exit("3D dump: %d cells refine anisotropically (2/2/2 assumed)" % bad.sum())

    cells = dict(xc=centers[:, 0], yc=centers[:, 1], zc=centers[:, 2],
                 hx=exact_hx, temp=d["T"].to_numpy(), nrho=d["nrho"].to_numpy(),
                 u=d["u"].to_numpy(), v=d["v"].to_numpy(), w=d["w"].to_numpy())

    os.makedirs(outdir, exist_ok=True)
    nc = write_field(outdir, step, box, base, cells)
    split_artifacts = None
    if split_parents is not None:
        import shutil
        map_out = Path(outdir) / "split-cell-map.tsv"
        shutil.copyfile(splitmap, map_out)
        rows_out = Path(outdir) / "split-field-rows.tsv"
        with rows_out.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("field_row\tidstr\tsplit\n")
            for rowno, (idstr, index) in enumerate(
                    d[["idstr", "split"]].itertuples(index=False, name=None), 1):
                if int(index) > 0:
                    stream.write(f"{rowno}\t{idstr}\t{int(index)}\n")
        ids_out = Path(outdir) / "split-parent-ids.txt"
        ids_receipt_source = Path(splitmap_ids).with_suffix(
            Path(splitmap_ids).suffix + ".json")
        ids_receipt_out = Path(outdir) / "split-parent-ids.txt.json"
        shutil.copyfile(splitmap_ids, ids_out)
        shutil.copyfile(ids_receipt_source, ids_receipt_out)
        manifest_out = Path(outdir) / "split-field-manifest.toml"
        split_hashes = {
            "field_sha256": _file_sha256(Path(outdir) / "DS3FF.DAT"),
            "source_frame_sha256": _file_sha256(path),
            "map_sha256": _file_sha256(map_out),
            "field_rows_sha256": _file_sha256(rows_out),
            "ids_sha256": _file_sha256(ids_out),
            "ids_receipt_sha256": _file_sha256(ids_receipt_out),
        }
        with manifest_out.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write('schema_version = "dsmc-split-field-v1"\n')
            for key, value in split_hashes.items():
                stream.write(f'{key} = "{value}"\n')
        split_artifacts = {
            "mode": "solver-exact-split3d-v1",
            "source_validation": splitmap_validation,
            "map": {"path": map_out.name, "sha256": _file_sha256(map_out)},
            "field_rows": {"path": rows_out.name, "sha256": _file_sha256(rows_out),
                           "rows": int((d["split"] > 0).sum())},
            "ids": {"path": ids_out.name, "sha256": _file_sha256(ids_out)},
            "ids_receipt": {"path": ids_receipt_out.name,
                            "sha256": _file_sha256(ids_receipt_out)},
            "manifest": {"path": manifest_out.name,
                         "sha256": _file_sha256(manifest_out)},
        }
    if canonical_surface is None:
        surfs = [f for f in os.listdir(rundir)
                 if f.startswith("cell") and f.endswith(".surf")]
        if len(surfs) != 1:
            sys.exit("expected exactly one cell*.surf in %s, found %s" % (rundir, surfs))
        pts, tri, typ = read_tris(os.path.join(rundir, surfs[0]))
        verts = pts[tri]
        ns = write_surfs(outdir, step, box, verts)
        geometry_prov = dict(surf_file=surfs[0],
                             triangle_types={str(k): int((typ == k).sum())
                                             for k in np.unique(typ)})
    else:
        geometry_prov = write_canonical_surfs(outdir, step, box,
                                               canonical_surface, triangle_roles,
                                               allow_geometry_only_roles)
        ns = geometry_prov["canonical_triangle_count"]
        verts = np.loadtxt(os.path.join(outdir, "cell.surfs"), skiprows=9,
                           ndmin=2)[:, 1:].reshape((-1, 3, 3))
    prov = dict(mode="sparta3d", source_run_dir=os.path.abspath(rundir),
                source_dump=dumpname, source_timestep=step,
                source_dump_sha256=_file_sha256(path),
                source_row_metadata=source_metadata,
                header_number_of_cells=ncells, rows_read=nrow,
                emitted_cells=nc, emitted_triangles=ns, **geometry_prov,
                box_bounds=box, base_grid=list(base), base_grid_source=base_source,
                 max_levels=int(lev.max()) + 1,
                 source_half_width_max_relative_rounding=source_hx_max_rel_delta,
                 emitted_half_width="exact-declared-lattice-after-source-validation",
                 inventory_all_cells=tot_inv, inventory_gas_cells=gas_inv,
                 zero_volume_solid_rows=zero_volume_rows,
                 retained_zero_density_rows=zero_density_rows,
                 near_zero_negative_temperature_rows=near_zero_negative_temperature_rows,
                 negative_temperature_tolerance_k=1e-9,
                 retained_source_rows=retained_source_rows,
                 retained_source_volume=retained_source_volume,
                 inventory_dropped_T0=drop_inv,
                 keep_unsampled=bool(keep_unsampled),
                 field_policy=("nrho-positive-nearest-positive-T" if keep_unsampled
                               else "legacy-positive-T-only") +
                              ("-plus-zero-density-ballistic" if keep_zero_density else ""),
                 temperature_carried_cells=nfilled,
                 split_rows_in_gas=nsplit, duplicate_boxes_collapsed=ndup,
                 duplicate_policy=("solver-exact-split3d-preserve-source-rows" if
                                   split_parents is not None else
                                   "exact-bounds-exact-decoded-fields-only"),
                 duplicate_field_tolerance={"rtol": 0.0, "atol": 0.0},
                 equivalent_duplicate_boxes=len(duplicate_metadata),
                 equivalent_duplicate_source_rows=duplicate_metadata,
                 split_lookup=split_artifacts,
                 lattice_center_coordinate_rtol=CENTER_COORD_RTOL,
                 lattice_center_cell_fraction_cap=CENTER_CELL_FRAC_CAP,
                 max_vertex_x=float(verts[:, :, 0].max()),
                converter="tools/field2tracer3d.py", converter_git_hash=git_hash(),
                converter_sha256=_file_sha256(__file__),
                date=datetime.now(timezone.utc).isoformat())
    with open(os.path.join(outdir, "provenance.json"), "w") as stream:
        json.dump(prov, stream, indent=2)
    return prov


# ------------------------------------------------------- mode B: 2D revolve

class Grid2D:
    """Exact containing-cell lookup on a 2D SPARTA adaptive lattice."""

    def __init__(self, xlo, ylo, xhi, yhi, box):
        self.box = box
        self.hx = 0.5 * (xhi - xlo)
        self.hy = 0.5 * (yhi - ylo)
        self.nx = int(round(0.5 * (box[0][1] - box[0][0]) / self.hx.max()))
        self.ny = int(round(0.5 * (box[1][1] - box[1][0]) / self.hy.max()))
        self.hx0 = 0.5 * (box[0][1] - box[0][0]) / self.nx
        self.hy0 = 0.5 * (box[1][1] - box[1][0]) / self.ny
        self.lev = levels_of(self.hx, self.hx0, "2D grid x")
        if np.any(levels_of(self.hy, self.hy0, "2D grid y") != self.lev):
            sys.exit("2D grid refines anisotropically")
        # per-level sorted key arrays; searchsorted beats a dict here because
        # the query is millions of points at a time
        self.maps = []
        for L in range(self.lev.max() + 1):
            m = self.lev == L
            i = np.rint((0.5 * (xlo + xhi)[m] - box[0][0])
                        / (2 * self.hx0 / 2 ** L) - 0.5).astype(np.int64)
            j = np.rint((0.5 * (ylo + yhi)[m] - box[1][0])
                        / (2 * self.hy0 / 2 ** L) - 0.5).astype(np.int64)
            key = j * (self.nx * 2 ** L) + i
            o = np.argsort(key)
            self.maps.append((key[o], np.flatnonzero(m)[o]))

    def find(self, x, y):
        """Row index of the containing cell, -1 if none. Vectorized."""
        out = np.full(x.shape, -1, np.int64)
        todo = np.ones(x.shape, bool)
        for L in range(len(self.maps) - 1, -1, -1):
            if not todo.any():
                break
            keys, rows = self.maps[L]
            dx, dy = 2 * self.hx0 / 2 ** L, 2 * self.hy0 / 2 ** L
            nxL, nyL = self.nx * 2 ** L, self.ny * 2 ** L
            i = np.floor((x[todo] - self.box[0][0]) / dx).astype(np.int64)
            j = np.floor((y[todo] - self.box[1][0]) / dy).astype(np.int64)
            inside = (i >= 0) & (i < nxL) & (j >= 0) & (j < nyL)
            k = np.where(inside, j * nxL + i, -1)
            pos = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
            hit = np.where(inside & (keys[pos] == k), rows[pos], -1)
            idx = np.flatnonzero(todo)
            out[idx[hit >= 0]] = hit[hit >= 0]
            todo[idx[hit >= 0]] = False
        return out


def last_frame_2d(path):
    L = open(path).read().splitlines()
    st = max(i for i, s in enumerate(L) if s.startswith("ITEM: TIMESTEP"))
    step = int(L[st + 1])
    box = [tuple(float(t) for t in L[st + k].split()) for k in (5, 6, 7)]
    D = np.array([[float(t) for t in s.split()] for s in L[st + 9:] if s.strip()])
    return step, box, D


def revolve(rundir, surfpath, outdir, maxlevel=None, dump="field.grid"):
    step, box2, D = last_frame_2d(os.path.join(rundir, dump))
    xc2, yc2 = D[:, 1], D[:, 2]
    xlo2, ylo2, xhi2, yhi2 = D[:, 3], D[:, 4], D[:, 5], D[:, 6]
    nrho2, u2, v2, T2 = D[:, 7], D[:, 8], D[:, 9], D[:, 10]
    g2 = Grid2D(xlo2, ylo2, xhi2, yhi2, box2)
    rmax = box2[1][1]
    box = [box2[0], (-rmax, rmax), (-rmax, rmax)]
    nx = g2.nx
    ny = nz = int(round(rmax / g2.hy0))
    base = (nx, ny, nz)
    nlev = g2.lev.max() + 1 if maxlevel is None else maxlevel

    # top-down 2x2x2 refinement: split a cell while the local 2D cell is finer
    hx0 = 0.5 * (box[0][1] - box[0][0]) / nx
    hy0 = 0.5 * (box[1][1] - box[1][0]) / ny
    hz0 = 0.5 * (box[2][1] - box[2][0]) / nz
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    lev = np.zeros(i.size, np.int64)
    keep_i, keep_j, keep_k, keep_l, keep_c = [], [], [], [], []
    for L in range(nlev):
        hx, hy, hz = hx0 / 2 ** L, hy0 / 2 ** L, hz0 / 2 ** L
        x = box[0][0] + (2 * i + 1) * hx
        y = box[1][0] + (2 * j + 1) * hy
        z = box[2][0] + (2 * k + 1) * hz
        r = np.hypot(y, z)
        c = g2.find(x, r)
        alive = (c >= 0) & (r <= rmax)
        # the same hard-zero deletion as the sparta3d filter above, inherited
        # here from the 2D field (which the vendored 2D build_field filters the
        # same way): 19681 of 27074 B5 cells survive it.  See that comment for
        # what it costs and what breaks if the extraction plane moves.
        alive[alive] &= T2[c[alive]] > 0
        # refine while the source cell is finer than this one
        finer = np.zeros(alive.shape, bool)
        if L < nlev - 1:
            finer[alive] = g2.hx[c[alive]] < hx - 1e-12
        done = alive & ~finer
        keep_i.append(i[done]); keep_j.append(j[done]); keep_k.append(k[done])
        keep_l.append(np.full(done.sum(), L)); keep_c.append(c[done])
        sel = np.flatnonzero(finer)
        if sel.size == 0:
            break
        o = np.array([0, 1])
        di, dj, dk = np.meshgrid(o, o, o, indexing="ij")
        i = (2 * i[sel][:, None] + di.ravel()[None, :]).ravel()
        j = (2 * j[sel][:, None] + dj.ravel()[None, :]).ravel()
        k = (2 * k[sel][:, None] + dk.ravel()[None, :]).ravel()
    i = np.concatenate(keep_i); j = np.concatenate(keep_j)
    k = np.concatenate(keep_k); lev = np.concatenate(keep_l)
    c = np.concatenate(keep_c)
    hx = hx0 / 2.0 ** lev
    x = box[0][0] + (2 * i + 1) * hx
    y = box[1][0] + (2 * j + 1) * hy0 / 2.0 ** lev
    z = box[2][0] + (2 * k + 1) * hz0 / 2.0 ** lev
    r = np.hypot(y, z)
    cs, sn = y / np.maximum(r, 1e-300), z / np.maximum(r, 1e-300)
    cells = dict(xc=x, yc=y, zc=z, hx=hx, temp=T2[c], nrho=nrho2[c],
                 u=u2[c], v=v2[c] * cs, w=v2[c] * sn)

    pts, tri, typ = read_tris(surfpath)
    verts = pts[tri]
    os.makedirs(outdir, exist_ok=True)
    nc = write_field(outdir, step, box, base, cells)
    ns = write_surfs(outdir, step, box, verts)
    vol = 8 * hx * (hy0 / 2.0 ** lev) * (hz0 / 2.0 ** lev)
    prov = dict(mode="revolve2d", source_run_dir=os.path.abspath(rundir),
                source_dump=dump, source_timestep=step, surf_file=surfpath,
                emitted_cells=nc, emitted_triangles=ns, box_bounds=box,
                base_grid=list(base), max_levels=int(lev.max()) + 1,
                source_cells_2d=int(D.shape[0]),
                source_gas_cells_2d=int((T2 > 0).sum()),
                inventory_gas_cells=float((cells["nrho"] * vol).sum()),
                triangle_types={str(t): int((typ == t).sum()) for t in np.unique(typ)},
                max_vertex_x=float(verts[:, :, 0].max()),
                rmax=rmax, converter="tools/field2tracer3d.py",
                converter_git_hash=git_hash(),
                date=datetime.now(timezone.utc).isoformat())
    with open(os.path.join(outdir, "provenance.json"), "w") as stream:
        json.dump(prov, stream, indent=2)
    return prov


# ------------------------------------------------------- mode C: synthetic

def synthetic(a, outdir):
    T, nrho, xlo, xhi, rmax, nx, nr = a
    nx, nr = int(nx), int(nr)
    box = [(xlo, xhi), (-rmax, rmax), (-rmax, rmax)]
    hx0 = 0.5 * (xhi - xlo) / nx
    h0 = 0.5 * rmax / nr          # HALF-size; 2*nr cells span [-rmax, rmax]
    i, j, k = np.meshgrid(np.arange(nx), np.arange(nr * 2), np.arange(nr * 2),
                          indexing="ij")
    i, j, k = i.ravel(), j.ravel(), k.ravel()
    x = xlo + (2 * i + 1) * hx0
    y = -rmax + (2 * j + 1) * h0
    z = -rmax + (2 * k + 1) * h0
    # no circular mask: the cylindrical bounds test kills a particle before it
    # can sample a corner cell, and a gap-free field lets the microtest gate
    # assert ZERO lookup fallbacks
    x, y, z = x.ravel(), y.ravel(), z.ravel()
    # near-uniform: a zero-width T range or zero flow makes the tracer's
    # lookup-table ranges degenerate (zero step), same guard as the 2D tool
    t = T * (0.9975 + 0.005 * (x - xlo) / (xhi - xlo))
    cells = dict(xc=x, yc=y, zc=z, hx=np.full(x.size, hx0), temp=t,
                 nrho=np.full(x.size, nrho), u=np.full(x.size, 0.1),
                 v=np.zeros(x.size), w=np.zeros(x.size))
    # one triangle strictly outside the domain: it can never be intersected,
    # but zero surfaces would leave max_x_geom undefined
    ox = xhi + (xhi - xlo)
    verts = np.array([[[ox, 0.0, 0.0], [ox, rmax, 0.0], [ox, 0.0, rmax]]])
    os.makedirs(outdir, exist_ok=True)
    nc = write_field(outdir, 0, box, (nx, 2 * nr, 2 * nr), cells)
    ns = write_surfs(outdir, 0, box, verts)
    prov = dict(mode="synthetic", T=T, nrho=nrho, box_bounds=box, rmax=rmax,
                nx=nx, nr=nr, flow_u=0.1, emitted_cells=nc, emitted_triangles=ns,
                base_grid=[nx, 2 * nr, 2 * nr], max_levels=1,
                inventory_gas_cells=float(nrho * 8 * hx0 * h0 * h0 * nc),
                max_vertex_x=float(ox), converter="tools/field2tracer3d.py",
                converter_git_hash=git_hash(),
                date=datetime.now(timezone.utc).isoformat())
    with open(os.path.join(outdir, "provenance.json"), "w") as stream:
        json.dump(prov, stream, indent=2)
    return prov


# ------------------------------------------------------------- self-check

def self_check():
    import tempfile
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print("%-46s %s %s" % (name, "PASS" if cond else "FAIL", detail))

    with tempfile.TemporaryDirectory() as td:
        # 1. synthetic round trip: values and inventory survive write -> read
        p = synthetic([4.0, 2.0e21, 0.0, 0.2, 0.02, 40, 10], td)
        step, box, base, c = read_field(os.path.join(td, "DS3FF.DAT"))
        chk("synthetic: header round-trips", step == 0 and base == (40, 20, 20)
            and box[0] == (0.0, 0.2), str(base))
        hx0, h0 = 0.5 * 0.2 / 40, 0.5 * 0.02 / 10
        vol = np.full(c["hx"].shape, 8 * hx0 * h0 * h0)
        inv = float((c["nrho"] * vol).sum())
        chk("synthetic: inventory round-trips <1e-9",
            abs(inv / p["inventory_gas_cells"] - 1) < 1e-9,
            "%.6e vs %.6e" % (inv, p["inventory_gas_cells"]))
        chk("synthetic: every cell inside the box",
            np.all(np.abs(c["yc"]) <= 0.02) and np.all(np.abs(c["zc"]) <= 0.02))
        span = (c["temp"].max() - c["temp"].min()) / 4.0
        chk("synthetic: T ramp preserved (non-degenerate)",
            abs(span - 0.005 * (1 - 1.0 / 40)) < 1e-9 and span > 0, "%.6g" % span)
        lev = lattice_check(
            box, base, np.column_stack([c["xc"], c["yc"], c["zc"]]),
            c["hx"], "synthetic")
        chk("synthetic: lattice level all 0", np.all(lev == 0))

        # 2. a KNOWN two-level field: 8 children of one base cell, exact volume
        box2 = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
        base2 = (2, 2, 2)
        xs, hs = [], []
        for a in (0.25, 0.75):
            for b in (0.25, 0.75):
                for cc in (0.25, 0.75):
                    xs.append((a, b, cc)); hs.append(0.25)
        # refine the first base cell into 8
        xs = xs[1:]; hs = hs[1:]
        for a in (0.125, 0.375):
            for b in (0.125, 0.375):
                for cc in (0.125, 0.375):
                    xs.append((a, b, cc)); hs.append(0.125)
        xs = np.array(xs); hs = np.array(hs)
        cells = dict(xc=xs[:, 0], yc=xs[:, 1], zc=xs[:, 2], hx=hs,
                     temp=np.full(hs.size, 4.0), nrho=np.full(hs.size, 1.0),
                     u=np.zeros(hs.size), v=np.zeros(hs.size), w=np.zeros(hs.size))
        d2 = os.path.join(td, "two")
        write_field(d2, 7, box2, base2, cells)
        _, bx, bs, cr = read_field(os.path.join(d2, "DS3FF.DAT"))
        volsum = float((8 * cr["hx"] ** 3).sum())
        chk("two-level: volumes tile the unit cube exactly",
            abs(volsum - 1.0) < 1e-12, "%.15g" % volsum)
        lev = lattice_check(
            bx, bs, np.column_stack([cr["xc"], cr["yc"], cr["zc"]]),
            cr["hx"], "two-level")
        chk("two-level: levels are 0 and 1", set(np.unique(lev)) == {0, 1})

        # 3. NEGATIVE CONTROL: an off-lattice cell must be rejected
        cells["hx"] = cells["hx"].copy(); cells["hx"][0] = 0.3
        d3 = os.path.join(td, "bad")
        write_field(d3, 7, box2, base2, cells)
        _, bx, bs, cb = read_field(os.path.join(d3, "DS3FF.DAT"))
        try:
            lattice_check(
                bx, bs, np.column_stack([cb["xc"], cb["yc"], cb["zc"]]),
                cb["hx"], "bad")
            caught = False
        except SystemExit:
            caught = True
        chk("negative control: off-lattice cell rejected", caught)
    print("SELF-CHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("rundir", nargs="?")
    p.add_argument("--out")
    p.add_argument("--dump")
    p.add_argument("--canonical-surface", help="solver-clipped physical-walls.surf for role-aware tracing")
    p.add_argument("--triangle-roles", help="hash-bound triangle-roles.json for --canonical-surface")
    p.add_argument("--geometry-only-roles", action="store_true",
                   help="accept a geometry-only role sidecar with unspecified molecule actions; output cannot run molecule transport")
    p.add_argument("--splitmap", help="one-rank solver split3d map for exact cut-cell lookup")
    p.add_argument("--splitmap-ids", help="receipt-bound split parent ID list for --splitmap")
    p.add_argument("--base-grid", nargs=3, type=int, metavar=("NX", "NY", "NZ"),
                   help="declared solver base-grid counts; required for prepared cases whose retained gas omits base cells")
    p.add_argument("--revolve", help="2D run dir to revolve into a 3D field")
    p.add_argument("--surf", help="3D .surf geometry for --revolve")
    p.add_argument("--maxlevel", type=int)
    p.add_argument("--keep-unsampled", action="store_true",
                   help="keep cells with nrho > 0 whose T is SPARTA's <=1-sample "
                        "sentinel, carrying T from the nearest sampled centre, "
                        "instead of deleting them (see convert3d)")
    p.add_argument("--keep-zero-density", action="store_true",
                   help="retain positive-volume zero-density zero-moment rows as explicit ballistic cells")
    p.add_argument("--synthetic", nargs=7, type=float,
                   metavar=("T", "NRHO", "XLO", "XHI", "RMAX", "NX", "NR"))
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        sys.exit(self_check())
    if not a.out:
        p.error("--out is required")
    if a.synthetic:
        prov = synthetic(a.synthetic, a.out)
    elif a.revolve:
        if not a.surf:
            p.error("--revolve needs --surf")
        prov = revolve(a.revolve, a.surf, a.out, a.maxlevel,
                       a.dump or "field.grid")
    else:
        prov = convert3d(a.rundir or sys.exit("need a run dir, --revolve or --synthetic"),
                         a.out, a.dump, keep_unsampled=a.keep_unsampled,
                         canonical_surface=a.canonical_surface,
                         triangle_roles=a.triangle_roles,
                         base_grid=a.base_grid,
                         keep_zero_density=a.keep_zero_density,
                         allow_geometry_only_roles=a.geometry_only_roles,
                         splitmap=a.splitmap, splitmap_ids=a.splitmap_ids)
    print("wrote %s: %d cells, %d triangles, %d levels, max vertex x = %.6g" %
          (a.out, prov["emitted_cells"], prov["emitted_triangles"],
           prov["max_levels"], prov["max_vertex_x"]))


if __name__ == "__main__":
    main()
