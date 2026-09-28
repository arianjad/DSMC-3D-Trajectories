#!/usr/bin/env python
"""Analyze a 3D tracer run (tracer/ParticleTracing3D.jl). 3D twin of tracer_analyze.py.

Usage
-----
    conda run -n claude-code python tools/tracer_analyze3d.py \
        <run>.out <field>/cell.surfs [-N 20000] [--spawnout sp.csv] \
        [--json out.json] [--outdir DIR]

The maintained 3D path reports the two signed Cartesian transverse components
separately.  It does not synthesize an azimuth or collapse source/exit data to
radius.  Directed-angle means are circular, and their histogram widths are
measured after unwrapping about that circular centre.  Compatibility JSON
aliases are explicitly labelled in `legacy_aliases`.
Two additional 3D-specific operations live here:

  read_surfs3d  the geometry is TRIANGLES, and the extraction plane is the
                largest triangle-vertex axial coordinate (0.0650 m for B5,
                identical to the 2D `cell_b5.surf`).

  clip_legs3d   `propagate` returns the UNCLIPPED end of the leg that killed
                the trajectory, exactly as in 2D, so the death point has to be
                recomputed here.  This is a real 3D clip -- Moller-Trumbore
                against every triangle, then the two axial planes and the
                cylinder r = rmax -- not the (z,r) projection the 2D tool
                uses.  Projecting would silently import the 48-gon's 0.11%
                inscribed-radius error into every death radius, and would be
                plain wrong the moment a geometry stops being a surface of
                revolution.

Figures show the two Cartesian axial projections independently.  `--spawnout`
adds Cartesian source projections rather than a cylindrical heat map.
"""
import argparse
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracer_analyze import COLS, fwhm  # noqa: E402

CIRCULAR_RESULTANT_TOL = 64.0 * np.finfo(float).eps


def _as_positive_int(value, label):
    if isinstance(value, bool):
        sys.exit(f"{label} must be a positive integer, got {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        sys.exit(f"{label} must be a positive integer, got {value!r}")
    if number <= 0 or number != value:
        sys.exit(f"{label} must be a positive integer, got {value!r}")
    return number


def _spawn_count(path):
    try:
        spawns = pd.read_csv(path)
    except Exception as exc:
        sys.exit(f"cannot read spawn record {path}: {exc}")
    if "idx" not in spawns.columns:
        sys.exit(f"spawn record {path} has no idx column")
    try:
        idx = spawns["idx"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        sys.exit(f"spawn record {path} has nonnumeric particle IDs: {exc}")
    if not len(idx) or not np.isfinite(idx).all() or np.any(idx != np.rint(idx)):
        sys.exit(f"spawn record {path} must contain finite integer particle IDs")
    ids = np.sort(idx.astype(np.int64))
    expected = np.arange(1, len(ids) + 1, dtype=np.int64)
    if not np.array_equal(ids, expected):
        sys.exit(f"spawn record {path} must contain each particle ID 1..N exactly once")
    return len(ids)


def _validate_terminal_output(receipt, outfile, receipt_path):
    """Bind a completed receipt to its terminal output when hashes are recorded."""
    out = Path(outfile).resolve()
    if "terminal_output_bytes" in receipt:
        expected_bytes = _as_positive_int(
            receipt["terminal_output_bytes"],
            f"{receipt_path}:terminal_output_bytes")
        actual_bytes = out.stat().st_size
        if actual_bytes != expected_bytes:
            sys.exit(f"terminal output byte count mismatch for {out}: receipt has "
                     f"{expected_bytes}, file has {actual_bytes}")
    if "terminal_output_sha256" in receipt:
        expected_hash = receipt["terminal_output_sha256"]
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 \
                or any(c not in "0123456789abcdefABCDEF" for c in expected_hash):
            sys.exit(f"{receipt_path}:terminal_output_sha256 must be 64 hex characters")
        digest = hashlib.sha256()
        with out.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual_hash = digest.hexdigest()
        if actual_hash != expected_hash.lower():
            sys.exit(f"terminal output hash mismatch for {out}: receipt has "
                     f"{expected_hash}, file has {actual_hash}")


def _read_receipt(path, outfile):
    path = Path(path).resolve()
    out = Path(outfile).resolve()
    allowed_parents = {out.parent, out.parent.parent}
    if path.parent not in allowed_parents:
        sys.exit(f"particle-count receipt {path} is not colocated with run output {out}")
    try:
        if path.suffix.lower() == ".toml":
            with path.open("rb") as f:
                data = tomllib.load(f)
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        sys.exit(f"cannot read particle-count receipt {path}: {exc}")
    if data.get("status") != "complete":
        sys.exit(f"particle-count receipt {path} is not complete")
    if path.suffix.lower() == ".toml":
        if data.get("schema") != "dsmc-tracer3d-v1":
            sys.exit(f"unsupported tracer receipt schema in {path}: {data.get('schema')!r}")
        _validate_terminal_output(data, out, path)
    return data


def _receipt_count(path, outfile):
    data = _read_receipt(path, outfile)
    if Path(path).suffix.lower() == ".toml":
        count = _as_positive_int(data.get("launched_particles"),
                                 f"{path}:launched_particles")
        return count, data
    candidates = []
    for key in ("n_particles", "particle_count", "spawn_count", "nparticles"):
        if key in data:
            candidates.append((key, data[key]))
    molecules = data.get("molecules")
    if isinstance(molecules, dict):
        for key in ("count", "n_particles", "particle_count"):
            if key in molecules:
                candidates.append((f"molecules.{key}", molecules[key]))
    if not candidates:
        return None, data
    counts = [(_as_positive_int(value, f"{path}:{key}"), key)
              for key, value in candidates]
    if len({count for count, _ in counts}) != 1:
        sys.exit(f"conflicting particle counts in receipt {path}: {candidates}")
    return counts[0][0], data


def resolve_nparticles(outfile, explicit=None, spawnout=None, receipt=None):
    """Return (N, evidence label), never an inference from surviving IDs."""
    evidence = []
    if explicit is not None:
        evidence.append((_as_positive_int(explicit, "-N/--nparticles"), "explicit"))
    if spawnout is not None:
        evidence.append((_spawn_count(spawnout), f"spawn:{Path(spawnout)}"))

    if receipt is not None:
        receipt_paths = [Path(receipt)]
        explicit_receipt = True
    else:
        out = Path(outfile).resolve()
        receipt_paths = [out.parent / "receipt.toml", out.parent / "manifest.json",
                         out.parent.parent / "manifest.json"]
        explicit_receipt = False
    seen = set()
    receipt_records = []
    for path in receipt_paths:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        count, data = _receipt_count(path, outfile)
        if count is not None:
            evidence.append((count, f"receipt:{path}"))
            receipt_records.append((path, data))
        elif explicit_receipt:
            sys.exit(f"particle-count receipt {path} has no recognized count field")

    if not evidence:
        sys.exit(
            "particle denominator is required: pass -N/--nparticles, provide a "
            "complete --spawnout record, or colocate a complete dsmc-tracer3d-v1 "
            "receipt.toml with the output (JSON --receipt with n_particles is also supported)")
    counts = {count for count, _ in evidence}
    if len(counts) != 1:
        sys.exit("particle denominator evidence disagrees: "
                 + "; ".join(f"{label}={count}" for count, label in evidence))
    receipt_record = next((r for r in receipt_records if r[0].suffix.lower() == ".toml"),
                          receipt_records[0] if receipt_records else None)
    return evidence[0][0], "+".join(label for _, label in evidence), receipt_record


def read_surfs3d(path, receipt_record=None, return_source=False):
    """(tris (T,3,3) in TRACER axes, xap, bounds [[zlo,zhi],[0,rmax]], seg_proj)."""
    lines = Path(path).read_text().splitlines()
    b = next(k for k, s in enumerate(lines) if s.startswith("ITEM: BOX BOUNDS"))
    box = np.array([[float(t) for t in lines[b + k].split()] for k in (1, 2, 3)])
    i = next(k for k, s in enumerate(lines) if s.startswith("ITEM: SURFS"))
    if "v1z" not in lines[i]:
        sys.exit("%s is a 2D surfs file; use tools/tracer_analyze.py" % path)
    v = np.array([[float(t) for t in s.split()[1:10]]
                  for s in lines[i + 1:] if s.strip()]).reshape(-1, 3, 3)
    # SPARTA (x,y,z) -> tracer (x1,x2,x3) = (y,z,x)
    tri = v[:, :, [1, 2, 0]]
    xap = float(v[:, :, 0].max())
    if receipt_record is not None and receipt_record[1].get("schema") == "dsmc-tracer3d-v1":
        receipt_path, receipt = receipt_record
        expected_hash = receipt.get("geom_sha256")
        if expected_hash is not None:
            actual_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                sys.exit(f"geometry hash mismatch: receipt {receipt_path} records "
                         f"{expected_hash}, but {path} is {actual_hash}")
        if "escape_z_m" not in receipt or "escape_radius_m" not in receipt:
            sys.exit(f"complete tracer receipt {receipt_path} lacks recorded escape bounds")
        try:
            zlo, zhi = (float(v) for v in receipt["escape_z_m"])
            rmax = float(receipt["escape_radius_m"])
        except (TypeError, ValueError) as exc:
            sys.exit(f"invalid escape bounds in receipt {receipt_path}: {exc}")
        if not np.isfinite([zlo, zhi, rmax]).all() or zlo >= zhi or rmax <= 0:
            sys.exit(f"invalid escape bounds in receipt {receipt_path}")
        for env_name, recorded in (("TRACER3D_ZHI", zhi), ("TRACER3D_RMAX", rmax)):
            if env_name in os.environ and not np.isclose(float(os.environ[env_name]), recorded,
                                                         rtol=0, atol=1e-15):
                sys.exit(f"{env_name}={os.environ[env_name]} disagrees with recorded "
                         f"run value {recorded} in {receipt_path}")
        bounds = np.array([[zlo, zhi], [0.0, rmax]])
        bounds_source = f"receipt:{receipt_path}"
    else:
        # Legacy outputs have no immutable escape metadata.  Preserve their
        # historical surface-box/environment interpretation and label it.
        rmax = float(os.environ.get(
            "TRACER3D_RMAX", min(-box[1, 0], box[1, 1], -box[2, 0], box[2, 1])))
        zhi = float(os.environ.get("TRACER3D_ZHI", box[0, 1]))
        bounds = np.array([[box[0, 0], zhi], [0.0, rmax]])
        bounds_source = "legacy-surface-box/environment"
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    seg = np.column_stack([e[:, 0, 2], np.hypot(e[:, 0, 0], e[:, 0, 1]),
                           e[:, 1, 2], np.hypot(e[:, 1, 0], e[:, 1, 1])])
    seg = np.unique(np.round(seg, 12), axis=0)
    result = (tri, xap, bounds, seg)
    return result + (bounds_source,) if return_source else result


def clip_legs3d(p1, p2, tri, bounds, chunk=200):
    """Clip each 3D leg to its first triangle or boundary crossing.

    Returns (points (N,3), code (N,)) with code 1 = wall, 2 = boundary,
    0 = neither.  Geometry wins outright whenever a triangle is hit, matching
    the tracer's own `getCollision`, which returns 1 before it tests the box.
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    t_wall = np.full(len(p1), np.inf)
    for a in range(0, len(p1), chunk):          # N x T arrays, chunked
        sl = slice(a, a + chunk)
        dd = d[sl]
        h = np.cross(dd[:, None, :], e2[None, :, :])
        det = np.einsum("tj,ntj->nt", e1, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1.0 / det
            s = p1[sl][:, None, :] - v0[None, :, :]
            u = f * np.einsum("ntj,ntj->nt", s, h)
            q = np.cross(s, e1[None, :, :])
            vv = f * np.einsum("nj,ntj->nt", dd, q)
            t = f * np.einsum("tj,ntj->nt", e2, q)
        good = ((det != 0) & (u >= 0) & (u <= 1) & (vv >= 0) & (u + vv <= 1)
                & (t >= 0) & (t <= 1))
        t_wall[sl] = np.where(good, t, np.inf).min(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tz = np.stack([(bounds[0, 0] - p1[:, 2]) / d[:, 2],
                       (bounds[0, 1] - p1[:, 2]) / d[:, 2]], axis=1)
        # cylinder |p_xy + t d_xy| = rmax
        A = (d[:, :2] ** 2).sum(1)
        B = 2 * (p1[:, :2] * d[:, :2]).sum(1)
        C = (p1[:, :2] ** 2).sum(1) - bounds[1, 1] ** 2
        disc = B * B - 4 * A * C
        sq = np.sqrt(np.where(disc > 0, disc, 0.0))
        tc = np.stack([(-B - sq) / (2 * A), (-B + sq) / (2 * A)], axis=1)
        tc = np.where((disc > 0)[:, None], tc, np.inf)
    tb = np.concatenate([tz, tc], axis=1)
    tb = np.where(np.isfinite(tb) & (tb > 0) & (tb <= 1), tb, np.inf)
    t_bnd = tb.min(axis=1)

    code = np.where(np.isfinite(t_wall), 1, np.where(np.isfinite(t_bnd), 2, 0))
    tmin = np.where(np.isfinite(t_wall), t_wall,
                    np.where(np.isfinite(t_bnd), t_bnd, 1.0))
    return p1 + tmin[:, None] * d, code


def _projected_kinematics(vtrans, vforward):
    """Actual signed velocity, line slope, and circular directed-angle metrics."""
    vtrans = np.asarray(vtrans, float)
    vforward = np.asarray(vforward, float)
    slope = np.full(vtrans.shape, np.nan)
    valid = np.isfinite(vtrans) & np.isfinite(vforward) & (vforward != 0)
    slope[valid] = vtrans[valid] / vforward[valid]
    angle = np.full(vtrans.shape, np.nan)
    directed = np.isfinite(vtrans) & np.isfinite(vforward) \
        & ((vtrans != 0) | (vforward != 0))
    angle[directed] = np.arctan2(vtrans[directed], vforward[directed])
    finite_angle = angle[directed]
    if finite_angle.size:
        mean_cos = float(np.mean(np.cos(finite_angle)))
        mean_sin = float(np.mean(np.sin(finite_angle)))
        resultant = float(np.hypot(mean_cos, mean_sin))
    else:
        mean_cos = mean_sin = resultant = float("nan")
    if np.isfinite(resultant) and resultant > CIRCULAR_RESULTANT_TOL:
        angle_mean = float(np.arctan2(mean_sin, mean_cos))
        # Move the branch cut opposite the sample's circular centre before a
        # linear histogram width is measured.  This preserves the full
        # directed population while making a narrow distribution around pi as
        # narrow as the same distribution around zero.
        angle_centered = (finite_angle - angle_mean + np.pi) % (2.0 * np.pi) - np.pi
        angle_fwhm = fwhm(angle_centered)
    else:
        # Opposed or exactly uniform directions have no meaningful circular
        # centre, so neither a mean direction nor a centre-dependent width is
        # reported.
        angle_mean = angle_fwhm = float("nan")
    return {
        "velocity_fwhm": fwhm(vtrans),
        "velocity_mean": float(np.mean(vtrans)) if len(vtrans) else float("nan"),
        "slope_mean": float(np.nanmean(slope)) if valid.any() else float("nan"),
        "slope_fwhm": fwhm(slope),
        "angle_mean_mrad": float(1e3 * angle_mean),
        "angle_fwhm_mrad": float(1e3 * angle_fwhm),
        "angle_resultant_length": resultant,
    }


def analyze3d(outfile, surfs, nparticles=None, spawnout=None, receipt=None):
    d = pd.read_csv(outfile, sep=r"\s+")
    if list(d.columns) != COLS:
        sys.exit(f"unexpected columns in {outfile}: {list(d.columns)}")
    try:
        numeric = d[COLS].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        sys.exit(f"{outfile}: nonnumeric trajectory value: {exc}")
    if not np.isfinite(numeric).all():
        row, column = np.argwhere(~np.isfinite(numeric))[0]
        sys.exit(f"{outfile}: nonfinite {COLS[int(column)]} at data row {int(row) + 1}")
    raw_idx = numeric[:, 0]
    if np.any(raw_idx == 0) or np.any(raw_idx != np.rint(raw_idx)):
        sys.exit(f"{outfile}: signed particle IDs must be nonzero integers")
    # Negative IDs are intentional trajectory-print rows.  Validate them first,
    # then retain only positive terminal rows for terminal-event statistics.
    d = d[raw_idx > 0].copy()
    n, n_source, receipt_record = resolve_nparticles(
        outfile, nparticles, spawnout, receipt)
    tri, xap, bounds, seg, bounds_source = read_surfs3d(
        surfs, receipt_record, return_source=True)
    idx = d["idx"].to_numpy(dtype=float)
    if not np.isfinite(idx).all() or np.any(idx != np.rint(idx)):
        sys.exit(f"{outfile}: terminal particle IDs must be finite integers")
    ids = idx.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        sys.exit(f"{outfile}: duplicate terminal particle IDs are unsupported")
    if len(ids) and ids.max() > n:
        sys.exit(f"{outfile}: terminal particle ID {ids.max()} exceeds denominator N={n}")

    hit, code = clip_legs3d(d[["x", "y", "z"]].to_numpy(),
                            d[["xnext", "ynext", "znext"]].to_numpy(), tri, bounds)
    d = d.assign(x_hit=hit[:, 0], y_hit=hit[:, 1], z_hit=hit[:, 2],
                 r_hit=np.hypot(hit[:, 0], hit[:, 1]), colltype=code)

    ext = d[d["z_hit"] > xap]             # clipped endpoint; see tracer_analyze.py
    other = d[d["z_hit"] <= xap]
    vf = ext["vz"].to_numpy()
    vx = ext["vx"].to_numpy()
    vy = ext["vy"].to_numpy()
    kx = _projected_kinematics(vx, vf)
    ky = _projected_kinematics(vy, vf)
    vfmed = float(np.median(vf)) if vf.size else float("nan")
    phi = np.arctan2(ext["vy"], ext["vx"]).to_numpy()
    if phi.size >= 120:
        cnt, _ = np.histogram(phi, bins=12, range=(-np.pi, np.pi))
        chi2 = float(((cnt - phi.size / 12.0) ** 2 / (phi.size / 12.0)).sum() / 11.0)
    else:
        chi2 = float("nan")
    axial = (ext["r_hit"].to_numpy() < 0.999 * bounds[1, 1]) if len(ext) \
        else np.zeros(0, bool)
    if axial.sum() >= 50:
        vf_a = float(np.median(vf[axial]))
        kx_a = _projected_kinematics(vx[axial], vf[axial])
        ky_a = _projected_kinematics(vy[axial], vf[axial])
    else:
        vf_a = float("nan")
        kx_a = ky_a = {"angle_fwhm_mrad": float("nan"),
                       "angle_resultant_length": float("nan")}

    res = {
        "outfile": str(outfile), "surfs": str(surfs), "dim": 3,
        "observable_schema": "cartesian-3d-v2",
        "aperture_exit_x_m": xap, "n": n,
        # Retain the historical key for report compatibility, but unsafe
        # max-surviving-ID inference is no longer an available code path.
        "n_from_max_idx": False, "n_source": n_source,
        "n_rows": int(len(d)), "n_extracted": int(len(ext)),
        "extraction": float(len(ext) / n) if n else float("nan"),
        "extraction_stderr": float(np.sqrt(len(ext) / n * (1 - len(ext) / n) / n))
                             if n else float("nan"),
        "vf_median_mps": vfmed,
        "vf_mean_mps": float(vf.mean()) if vf.size else float("nan"),
        "vf_fwhm_mps": fwhm(vf),
        "vx_fwhm_mps": kx["velocity_fwhm"], "vy_fwhm_mps": ky["velocity_fwhm"],
        "vx_mean_mps": kx["velocity_mean"], "vy_mean_mps": ky["velocity_mean"],
        "vx_over_vz_mean": kx["slope_mean"], "vy_over_vz_mean": ky["slope_mean"],
        "vx_over_vz_fwhm": kx["slope_fwhm"], "vy_over_vz_fwhm": ky["slope_fwhm"],
        "angle_x_mean_mrad": kx["angle_mean_mrad"],
        "angle_y_mean_mrad": ky["angle_mean_mrad"],
        "angle_x_fwhm_mrad": kx["angle_fwhm_mrad"],
        "angle_y_fwhm_mrad": ky["angle_fwhm_mrad"],
        "angle_x_resultant_length": kx["angle_resultant_length"],
        "angle_y_resultant_length": ky["angle_resultant_length"],
        "angle_estimator": {
            "direction": "directed atan2(v_transverse,v_z) in [-pi,pi]",
            "mean": "circular mean; NaN when the circular resultant is undefined",
            "fwhm": "histogram FWHM after unwrapping about the circular mean",
            "units": "mrad",
            "minimum_resultant_length": CIRCULAR_RESULTANT_TOL,
        },
        "azimuth_chi2_per_dof": chi2,
        "n_nonforward_extracted": int((vf <= 0).sum()),
        "n_radial_exit": int((~axial).sum()),
        "radial_exit_frac": float((~axial).mean()) if axial.size else float("nan"),
        "vf_median_axial_exit_mps": vf_a,
        "angle_x_fwhm_mrad_axial_exit": kx_a["angle_fwhm_mrad"],
        "angle_y_fwhm_mrad_axial_exit": ky_a["angle_fwhm_mrad"],
        "angle_x_resultant_length_axial_exit": kx_a["angle_resultant_length"],
        "angle_y_resultant_length_axial_exit": ky_a["angle_resultant_length"],
        # Compatibility aliases select the tracer x component and are never
        # used by this module's report or figures.  They do not azimuth-average.
        "vt_fwhm_mps": kx["velocity_fwhm"],
        "vt_fwhm_signed_vx_mps": kx["velocity_fwhm"],
        "divergence_fwhm_mrad": kx["angle_fwhm_mrad"],
        "divergence_fwhm_mrad_axial_exit": kx_a["angle_fwhm_mrad"],
        "legacy_aliases": {
            "vt_fwhm_mps": "vx_fwhm_mps; signed tracer-x component, no azimuth projection",
            "vt_fwhm_signed_vx_mps": "vx_fwhm_mps",
            "divergence_fwhm_mrad":
                "angle_x_fwhm_mrad; directed atan2(vx,vz), circular-center unwrap",
            "divergence_fwhm_mrad_axial_exit":
                "angle_x_fwhm_mrad_axial_exit; directed atan2(vx,vz), "
                "circular-center unwrap",
        },
        "collides_mean_extracted": float(ext["collides"].mean()) if len(ext) else float("nan"),
        "collides_mean_all_rows": float(d["collides"].mean()),
        "time_mean_extracted_s": float(ext["time"].mean()) if len(ext) else float("nan"),
        "time_estimator": "mean age at start of terminal leg; final-leg travel is excluded",
        # Keep the shared 2D/3D report contract: this is every row not past
        # the escape plane, regardless of whether a wall or a boundary ended
        # it. `n_wall_rows` remains the legacy JSON alias.
        "n_not_extracted": int(len(other)),
        "n_wall_rows": int(len(other)),
        "n_axial_exit": int(axial.sum()) if axial.size else 0,
        "extraction_axial": float(axial.sum() / n) if n and axial.size else float("nan"),
        "extraction_axial_stderr": float(np.sqrt(
            (axial.sum() / n) * (1 - axial.sum() / n) / n))
            if n and axial.size else float("nan"),
        "n_triangles": int(len(tri)), "rmax_m": float(bounds[1, 1]),
        "escape_bounds_source": bounds_source,
        # v_f, its widths and the divergence are read off the TERMINATING leg,
        # so they are properties of this plane and not of the cell: on
        # b5hires the median v_f runs 54.9 / 74.2 / 79.4 / 82.4 / 84.3 m/s at
        # escape_zhi = 65.1 / 70 / 80 / 92.6 / 115 mm.  Two runs are only
        # comparable on those metrics if this number matches.
        "escape_zhi_m": float(bounds[0, 1]),
    }
    return d, ext, other, tri, xap, res


def report3d(res):
    return "\n".join([
        f"  file             {Path(res['outfile']).name}",
        f"  n                {res['n']}   ({res['n_source']})",
        f"  rows / extracted {res['n_rows']} / {res['n_extracted']}"
        + (f"   (+{res['n_not_extracted']} not extracted)" if res["n_not_extracted"] else ""),
        f"  extraction       {res['extraction']:.4f} +- {res['extraction_stderr']:.4f}",
        f"  extraction axial {res['extraction_axial']:.4f} +- {res['extraction_axial_stderr']:.4f}"
        f"  ({res['n_axial_exit']} of {res['n_extracted']} extracted, downstream face only)",
        f"  v_z median       {res['vf_median_mps']:.1f} m/s   (mean {res['vf_mean_mps']:.1f})",
        f"  v_z FWHM         {res['vf_fwhm_mps']:.1f} m/s",
        f"  signed v_x/v_y   FWHM {res['vx_fwhm_mps']:.1f} / {res['vy_fwhm_mps']:.1f} m/s; "
        f"means {res['vx_mean_mps']:+.2f} / {res['vy_mean_mps']:+.2f}",
        f"  slopes x/y       mean {res['vx_over_vz_mean']:+.4g} / {res['vy_over_vz_mean']:+.4g}; "
        f"FWHM {res['vx_over_vz_fwhm']:.4g} / {res['vy_over_vz_fwhm']:.4g}",
        f"  angles x/y       directed atan2(vx,vz), atan2(vy,vz); circular mean "
        f"{res['angle_x_mean_mrad']:+.1f} / "
        f"{res['angle_y_mean_mrad']:+.1f} mrad",
        f"  angle FWHM x/y   {res['angle_x_fwhm_mrad']:.0f} / "
        f"{res['angle_y_fwhm_mrad']:.0f} mrad (all exits)",
        f"                     {res['angle_x_fwhm_mrad_axial_exit']:.0f} / "
        f"{res['angle_y_fwhm_mrad_axial_exit']:.0f} mrad (downstream face only; "
        f"{100*res['radial_exit_frac']:.0f}% left radially)",
        f"  exit azimuth     chi2/dof {res['azimuth_chi2_per_dof']:.1f}; "
        f"{res['n_nonforward_extracted']} extracted rows have v_z <= 0",
        f"  collisions       {res['collides_mean_extracted']:.0f} mean (extracted)",
        f"  age before final leg {res['time_mean_extracted_s']*1e3:.3f} ms mean (extracted)",
    ])


def _plot_geometry_projection(ax, tri, transverse_axis):
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    for edge in edges:
        ax.plot(edge[:, 2] * 1e3, edge[:, transverse_axis] * 1e3,
                color="black", lw=0.7, alpha=0.5)


def figures3d(d, ext, other, tri, xap, res, outdir, label, spawnout=None):
    """Cartesian 3D summaries; no radial collapse or synthetic azimuth."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
    fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.0))
    vf = ext["vz"].to_numpy()
    ax[0, 0].hist(vf, bins=60, color="#4477aa", alpha=0.8)
    ax[0, 0].set(xlabel="axial velocity vz (m/s)", ylabel="extracted molecules / bin",
                 title=f"(a) axial velocity: {len(ext)} extracted")

    vx, vy = ext["vx"].to_numpy(), ext["vy"].to_numpy()
    tx, ty = np.full(len(vf), np.nan), np.full(len(vf), np.nan)
    directed_x, directed_y = (vx != 0) | (vf != 0), (vy != 0) | (vf != 0)
    tx[directed_x] = 1e3 * np.arctan2(vx[directed_x], vf[directed_x])
    ty[directed_y] = 1e3 * np.arctan2(vy[directed_y], vf[directed_y])
    ax[0, 1].hist(tx[np.isfinite(tx)], bins=60, histtype="step", lw=1.5,
                  color="#cc6677", label="directed atan2(vx,vz)")
    ax[0, 1].hist(ty[np.isfinite(ty)], bins=60, histtype="step", lw=1.5,
                  color="#228833", label="directed atan2(vy,vz)")
    ax[0, 1].set(xlabel="signed projected angle (mrad)", ylabel="extracted molecules / bin",
                 title="(b) Cartesian projected angles")
    ax[0, 1].legend(frameon=False)

    other_take = slice(None) if len(other) <= 30000 else \
        slice(None, None, max(1, len(other) // 30000))
    for panel, coordinate, axis_index, tag in [
            (ax[1, 0], "x_hit", 0, "x"), (ax[1, 1], "y_hit", 1, "y")]:
        _plot_geometry_projection(panel, tri, axis_index)
        if len(other):
            panel.plot(other["z_hit"].to_numpy()[other_take] * 1e3,
                       other[coordinate].to_numpy()[other_take] * 1e3, ".", ms=1.3,
                       alpha=0.3, color="#cc6677", rasterized=True, label="not extracted")
        if len(ext):
            ext_take = slice(None) if len(ext) <= 30000 else \
                slice(None, None, max(1, len(ext) // 30000))
            panel.plot(ext["z_hit"].to_numpy()[ext_take] * 1e3,
                       ext[coordinate].to_numpy()[ext_take] * 1e3, ".", ms=1.3,
                       alpha=0.3, color="#4477aa", rasterized=True, label="extracted")
        panel.axvline(xap * 1e3, color="0.45", ls=":", lw=1.0)
        panel.set(xlabel="axial z (mm)", ylabel=f"transverse {tag} (mm)",
                  title=f"({'c' if tag == 'x' else 'd'}) terminal contacts: z-{tag}")
    fig.suptitle(f"{label}: Cartesian 3D summary | extraction {res['extraction']:.4f} "
                 f"+- {res['extraction_stderr']:.4f} (N={res['n']})")
    fig.tight_layout()
    summary = outdir / f"{label}_summary.png"
    fig.savefig(summary, dpi=150)
    plt.close(fig)
    made = [summary]

    if spawnout is not None:
        sp = pd.read_csv(spawnout)
        extracted = set(ext["idx"].astype(int))
        is_ext = sp["idx"].astype(int).isin(extracted).to_numpy()
        take = slice(None) if len(sp) <= 30000 else slice(None, None, max(1, len(sp) // 30000))
        fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
        for panel, coordinate, tag in [(axs[0], "x", "x"), (axs[1], "y", "y")]:
            panel.plot(sp["z"].to_numpy()[take] * 1e3,
                       sp[coordinate].to_numpy()[take] * 1e3, ".", ms=1.0,
                       alpha=0.18, color="0.4", rasterized=True, label="all spawns")
            selected = sp.loc[is_ext]
            panel.plot(selected["z"] * 1e3, selected[coordinate] * 1e3, ".", ms=1.2,
                       alpha=0.35, color="#4477aa", rasterized=True, label="extracted")
            panel.set(xlabel="initial axial z (mm)", ylabel=f"initial transverse {tag} (mm)",
                      title=f"source projection z-{tag}")
        axs[0].legend(frameon=False)
        fig.suptitle(f"{label}: Cartesian spawn positions (points, no radial bin averaging)")
        fig.tight_layout()
        spawn_path = outdir / f"{label}_spawn_cartesian.png"
        fig.savefig(spawn_path, dpi=150)
        plt.close(fig)
        made.append(spawn_path)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outfile")
    ap.add_argument("surfs")
    ap.add_argument("-N", "--nparticles", type=int, default=None)
    ap.add_argument("--spawnout", default=None)
    ap.add_argument("--receipt", default=None,
                    help="complete colocated JSON/TOML run receipt containing launched count")
    ap.add_argument("--json", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--heatbins", default=None,
                    help="unsupported legacy 2D radial-heatmap option; 3D uses Cartesian points")
    ap.add_argument("--label", default=None)
    a = ap.parse_args()
    if a.heatbins is not None:
        ap.error("--heatbins is a 2D radial-averaging option and is unsupported in 3D")

    d, ext, other, tri, xap, res = analyze3d(
        a.outfile, a.surfs, a.nparticles, a.spawnout, a.receipt)
    label = a.label or Path(a.outfile).stem
    print(report3d(res))
    if a.outdir:
        for p in figures3d(d, ext, other, tri, xap, res, a.outdir, label,
                           a.spawnout):
            print(f"  wrote            {p}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
        print(f"  wrote            {a.json}")


if __name__ == "__main__":
    main()
