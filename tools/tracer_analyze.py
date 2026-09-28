#!/usr/bin/env python
"""Analyze a ParticleTracing.jl run: extraction, forward velocity, divergence.

Usage
-----
    conda run -n claude-code python tools/tracer_analyze.py \
        <run>.out <field-dir>/cell.surfs \
        [-N 100000] [--spawnout <spawn.csv>] [--json out.json] [--outdir DIR]

(`--n` would be ambiguous with conda run's own options, hence `-N`.)

The tracer writes one whitespace-separated row per particle:

    idx x y z xnext ynext znext vx vy vz collides time

where (x,y,z) starts the LAST leg and (xnext,ynext,znext) is where the
particle died -- a wall, or the domain boundary.  Tracer z is the SPARTA
axial coordinate; the tracer's transverse pair (x,y) gives the SPARTA radius
r = sqrt(x^2+y^2).  Rows with idx < 0 come from `--trajprint` and are dropped.

Without `--saveall` the tracer prints ONLY particles that reached the domain
boundary, so the death map needs a `--saveall 1` run; the panel is skipped
(with a note) when no wall deaths are present.  Spawn positions are never in
the stdout stream -- pass a `--spawnout` CSV (idx,x,y,z) to get the Fig.-5
extraction heat map.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito, matching tools/plot_trajectories.py: blue = extracted, vermillion
# = wall-stuck, green = reference/prediction, black = measured summary.
BLUE, VERM, GREEN, BLACK, GRAY = "#0072B2", "#D55E00", "#009E73", "#000000", "#888888"
COLS = ["idx", "x", "y", "z", "xnext", "ynext", "znext",
        "vx", "vy", "vz", "collides", "time"]


def read_surfs(path):
    """Return (segments, xmax, bounds) from a SPARTA-style .surfs dump.

    xmax is the geometry's downstream extent -- the tracer's own
    `max_x_geom`, and the plane a particle must pass to count as extracted.
    bounds is [[zlo, zhi], [rlo, rhi]], the domain the tracer calls `bounds`.
    """
    lines = Path(path).read_text().splitlines()
    b = next(k for k, s in enumerate(lines) if s.startswith("ITEM: BOX BOUNDS"))
    bounds = np.array([[float(t) for t in lines[b + 1].split()],
                       [float(t) for t in lines[b + 2].split()]])
    i = next(k for k, s in enumerate(lines) if s.startswith("ITEM: SURFS"))
    seg = np.array([[float(t) for t in s.split()[1:5]]
                    for s in lines[i + 1:] if s.strip()])
    return seg, float(max(seg[:, 0].max(), seg[:, 2].max())), bounds


def clip_legs(p1, p2, seg, bounds):
    """Clip each (z, r) leg to its first geometry or boundary crossing.

    `propagate` returns the UNCLIPPED end of the leg that terminated the
    trajectory, so the raw (xnext, ynext, znext) sits past the wall -- final
    radii of 250-300 mm show up on a 19.9 mm domain.  This reproduces the
    tracer's own `getCollision` in the (z, r) plane: segment-segment tests
    against every surf first (geometry wins outright, exactly as upstream
    returns 1 before ever testing the box), then the zlo / zhi / rhi faces.

    Returns (points (N,2), code (N,)) with code 1 = wall, 2 = boundary,
    0 = neither (which the caller should not see for a terminated leg).
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    q1, e = seg[:, :2], seg[:, 2:] - seg[:, :2]
    with np.errstate(divide="ignore", invalid="ignore"):
        den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        w = q1[None, :, :] - p1[:, None, :]
        t = (w[:, :, 0] * e[None, :, 1] - w[:, :, 1] * e[None, :, 0]) / den
        u = (w[:, :, 0] * d[:, None, 1] - w[:, :, 1] * d[:, None, 0]) / den
    good = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    t_wall = np.where(good, t, np.inf).min(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tb = np.stack([(bounds[0, 0] - p1[:, 0]) / d[:, 0],
                       (bounds[0, 1] - p1[:, 0]) / d[:, 0],
                       (bounds[1, 1] - p1[:, 1]) / d[:, 1]], axis=1)
    tb = np.where(np.isfinite(tb) & (tb > 0) & (tb <= 1), tb, np.inf)
    t_bnd = tb.min(axis=1)

    code = np.where(np.isfinite(t_wall), 1, np.where(np.isfinite(t_bnd), 2, 0))
    tmin = np.where(np.isfinite(t_wall), t_wall,
                    np.where(np.isfinite(t_bnd), t_bnd, 1.0))
    return p1 + tmin[:, None] * d, code


def fwhm(v, nbins=None):
    """Full width at half maximum by half-max crossing of a smoothed histogram.

    Linear interpolation between the bracketing bin centres.  Returns nan when
    the sample is too small or the distribution never falls below half max
    inside the sampled range (a truncated tail), so a missing width is visible
    rather than silently fabricated.

    Bins default to Freedman-Diaconis clamped to [12, 80].  Measured against
    Gaussian samples (60 repeats per point): bias +6.5% / rms 11% at n = 200,
    +1.0% / 5.9% at n = 1000, and under +-1% above n = 5000.  A fixed bin count
    does far worse in the small-sample limit -- 60 bins reads 35% LOW at
    n = 200 -- which matters for the low-extraction end of a cross-section scan.
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 50:
        return float("nan")
    if nbins is None:
        nbins = int(np.clip(len(np.histogram_bin_edges(v, bins="fd")) - 1, 12, 80))
    h, e = np.histogram(v, bins=nbins)
    h = np.convolve(h.astype(float), np.ones(3) / 3.0, mode="same")
    c = 0.5 * (e[:-1] + e[1:])
    i = int(h.argmax())
    half = h[i] / 2.0

    def cross(rng, step):
        for j in rng:
            if h[j + step] < half:
                a, b = j, j + step
                return c[a] + (half - h[a]) * (c[b] - c[a]) / (h[b] - h[a])
        return None

    lo = cross(range(i, 0, -1), -1)
    hi = cross(range(i, len(h) - 1), +1)
    if lo is None or hi is None:
        return float("nan")
    return float(hi - lo)


def analyze(outfile, surfs, nparticles=None):
    d = pd.read_csv(outfile, sep=r"\s+")
    if list(d.columns) != COLS:
        sys.exit(f"unexpected columns in {outfile}: {list(d.columns)}")
    d = d[d["idx"] > 0]
    seg, xap, bounds = read_surfs(surfs)

    n = nparticles if nparticles else int(d["idx"].max())
    n_from_idx = nparticles is None

    hit, code = clip_legs(
        np.column_stack([d["z"], np.hypot(d["x"], d["y"])]),
        np.column_stack([d["znext"], np.hypot(d["xnext"], d["ynext"])]),
        seg, bounds)
    d = d.assign(z_hit=hit[:, 0], r_hit=hit[:, 1], colltype=code)

    # Score on the CLIPPED endpoint: `znext` is the unclipped free-path end of
    # the terminating leg, so a wall death whose free path would have carried
    # it past the plane counted as extracted (open-issues item 15, 0.49 % on
    # gate A).  Same clause in tracer_analyze3d.py; gate A compares the two.
    ext = d[d["z_hit"] > xap]
    other = d[d["z_hit"] <= xap]          # present only under --saveall

    vf = ext["vz"].to_numpy()
    vr = np.hypot(ext["vx"], ext["vy"]).to_numpy()

    # Divergence is the width of the angular profile in a plane through the
    # axis -- i.e. of ONE signed transverse component.  Taking v_x directly
    # would assume the ensemble is azimuthally uniform, which is NOT true for
    # the `point` and `uniformplane` spawns (every particle at azimuth 0), nor
    # for any pre-diff-9 `uniformcell` output: an axisymmetric flow exerts no
    # azimuthal force, so angular momentum only diffuses and even ~9000
    # collisions leave <v_x> = +15.05 +- 0.09 m/s (175 sigma).  Projecting
    # |v_r| onto a random lab azimuth is azimuth-independent by construction
    # and agrees with signed v_x to 1-2% wherever the ensemble IS uniform, so
    # it is used for every spawn mode.  The chi2 below keeps the raw
    # asymmetry visible instead of silently corrected away.
    vt = vr * np.cos(2.0 * np.pi * np.random.default_rng(20260820).random(vr.size))
    dvt = fwhm(vt)
    vfmed = float(np.median(vf)) if vf.size else float("nan")
    div = (2.0 * np.arctan(dvt / (2.0 * vfmed))
           if np.isfinite(dvt) and vfmed > 0 else float("nan"))

    # Keep the raw asymmetry visible rather than silently corrected for.
    phi = np.arctan2(ext["vy"], ext["vx"]).to_numpy()
    if phi.size >= 120:
        cnt, _ = np.histogram(phi, bins=12, range=(-np.pi, np.pi))
        chi2 = float(((cnt - phi.size / 12.0) ** 2 / (phi.size / 12.0)).sum() / 11.0)
    else:
        chi2 = float("nan")

    # A sizeable fraction of the beam leaves through the RADIAL domain face,
    # not the downstream one -- those are the wide-angle wings.  The full set
    # is the unbiased source divergence; the axial-exit subset is what a
    # skimmer of the domain's own aspect ratio would pass, and is the number
    # to compare against an experiment with a downstream aperture.
    axial = (ext["r_hit"].to_numpy() < 0.999 * bounds[1, 1]) if len(ext) \
        else np.zeros(0, bool)
    if axial.sum() >= 50:
        dvt_a = fwhm(vt[axial])
        vf_a = float(np.median(vf[axial]))
        div_a = (2.0 * np.arctan(dvt_a / (2.0 * vf_a))
                 if np.isfinite(dvt_a) and vf_a > 0 else float("nan"))
    else:
        dvt_a, vf_a, div_a = float("nan"), float("nan"), float("nan")

    res = {
        "outfile": str(outfile),
        "surfs": str(surfs),
        "aperture_exit_x_m": xap,
        "n": n,
        "n_from_max_idx": n_from_idx,
        "n_rows": int(len(d)),
        "n_extracted": int(len(ext)),
        "extraction": float(len(ext) / n) if n else float("nan"),
        "extraction_stderr": float(np.sqrt(len(ext) / n * (1 - len(ext) / n) / n))
                             if n else float("nan"),
        "vf_median_mps": vfmed,
        "vf_mean_mps": float(vf.mean()) if vf.size else float("nan"),
        "vf_fwhm_mps": fwhm(vf),
        "vt_fwhm_mps": dvt,
        "vt_fwhm_signed_vx_mps": fwhm(ext["vx"].to_numpy()) if len(ext) else float("nan"),
        "vx_mean_mps": float(ext["vx"].mean()) if len(ext) else float("nan"),
        "vy_mean_mps": float(ext["vy"].mean()) if len(ext) else float("nan"),
        "azimuth_chi2_per_dof": chi2,
        "divergence_fwhm_mrad": float(1e3 * div),
        "n_radial_exit": int((~axial).sum()),
        "radial_exit_frac": float((~axial).mean()) if axial.size else float("nan"),
        "vf_median_axial_exit_mps": vf_a,
        "divergence_fwhm_mrad_axial_exit": float(1e3 * div_a),
        "collides_mean_extracted": float(ext["collides"].mean()) if len(ext) else float("nan"),
        "collides_mean_all_rows": float(d["collides"].mean()),
        "time_mean_extracted_s": float(ext["time"].mean()) if len(ext) else float("nan"),
        # n_not_extracted = every row NOT past the escape plane, whatever
        # killed it (wall, radial domain face, upstream face) -- see
        # the 2026-09-05 tracer audit, defect 1: the old name
        # "n_wall_rows" mislabels this on any geometry with an open upstream
        # face (measured 0 such rows on the eight production arms, but the
        # name should not depend on that being true).  n_wall_rows kept as an
        # alias for one release so old readers of the JSON do not break.
        "n_not_extracted": int(len(other)),
        "n_wall_rows": int(len(other)),
        # n_axial_exit = extracted through the downstream face only (the
        # complement of n_radial_exit within n_extracted), i.e. a beam
        # observable comparable across the base/ap*/f* arms whose escape
        # planes and aperture areas differ 58x (tracer-audit.md Q1/Q2).
        "n_axial_exit": int(axial.sum()) if axial.size else 0,
        "extraction_axial": float(axial.sum() / n) if n and axial.size else float("nan"),
        "extraction_axial_stderr": float(np.sqrt(
            (axial.sum() / n) * (1 - axial.sum() / n) / n))
            if n and axial.size else float("nan"),
    }
    return d, ext, other, seg, xap, res


def figures(d, ext, other, seg, xap, res, outdir, label, spawnout=None,
            heatbins=(26, 13)):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False})
    made = []

    fig, ax = plt.subplots(1, 3, figsize=(15.0, 4.2))

    # (a) forward velocity
    vf = ext["vz"].to_numpy()
    ax[0].hist(vf, bins=70, color=BLUE, alpha=0.8)
    ax[0].axvline(res["vf_median_mps"], color=BLACK, lw=2,
                  label=f"median {res['vf_median_mps']:.1f} m/s")
    if np.isfinite(res["vf_fwhm_mps"]):
        ax[0].axvline(res["vf_median_mps"] - res["vf_fwhm_mps"] / 2, color=GRAY, ls=":", lw=1.5)
        ax[0].axvline(res["vf_median_mps"] + res["vf_fwhm_mps"] / 2, color=GRAY, ls=":", lw=1.5,
                      label=f"FWHM {res['vf_fwhm_mps']:.1f} m/s")
    ax[0].set_xlabel(r"forward velocity $v_f$ (m/s)")
    ax[0].set_ylabel("extracted molecules / bin")
    ax[0].set_title(f"(a) forward velocity — {res['n_extracted']} extracted")
    ax[0].legend(frameon=False, fontsize=9)

    # (b) transverse-vs-forward phase space.  Plot the SAME azimuth-projected
    # transverse component whose FWHM sets the divergence, so the two
    # half-angle lines actually bracket the plotted width.  (Plotting |v_r|
    # here instead would show a hole at 0 that is pure Jacobian -- for any
    # 2D-isotropic distribution P(|v_r|) goes as |v_r| and vanishes at the
    # origin -- and would sit entirely above the half-angle lines.)
    vt = (np.hypot(ext["vx"], ext["vy"]).to_numpy()
          * np.cos(2.0 * np.pi * np.random.default_rng(20260820).random(len(ext))))
    s = slice(None) if len(ext) <= 20000 else slice(None, None, max(1, len(ext) // 20000))
    ax[1].plot(vf[s], vt[s], ".", ms=1.5, alpha=0.25, color=BLUE, rasterized=True)
    if np.isfinite(res["divergence_fwhm_mrad"]):
        t = np.tan(0.5 * res["divergence_fwhm_mrad"] * 1e-3)
        xx = np.array([0.0, np.nanpercentile(vf, 99.5)])
        ax[1].plot(xx, t * xx, color=GREEN, lw=2,
                   label=f"FWHM {res['divergence_fwhm_mrad']:.0f} mrad (all exits)")
        ax[1].plot(xx, -t * xx, color=GREEN, lw=2)
        if np.isfinite(res["divergence_fwhm_mrad_axial_exit"]):
            ta_ = np.tan(0.5 * res["divergence_fwhm_mrad_axial_exit"] * 1e-3)
            ax[1].plot(xx, ta_ * xx, color=VERM, lw=2, ls="--",
                       label=f"{res['divergence_fwhm_mrad_axial_exit']:.0f} mrad "
                             f"(downstream face only)")
            ax[1].plot(xx, -ta_ * xx, color=VERM, lw=2, ls="--")
        ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    ax[1].set_xlabel(r"$v_f$ (m/s)")
    ax[1].set_ylabel(r"transverse $v_r$ on a random lab axis (m/s)")
    ax[1].set_title(f"(b) exit phase space — "
                    f"{100*res['radial_exit_frac']:.0f}% leave radially")

    # (c) death map, mirrored about the axis
    for s0 in seg:
        ax[2].plot([s0[0] * 1e3, s0[2] * 1e3], [s0[1] * 1e3, s0[3] * 1e3], color=BLACK, lw=1.2)
        ax[2].plot([s0[0] * 1e3, s0[2] * 1e3], [-s0[1] * 1e3, -s0[3] * 1e3], color=BLACK, lw=1.2)
    if len(other):
        rr = other["r_hit"].to_numpy() * 1e3            # clipped contact point
        zz = other["z_hit"].to_numpy() * 1e3
        sg = np.where(np.arange(len(zz)) % 2 == 0, 1.0, -1.0)   # mirror alternate points
        k = slice(None) if len(zz) <= 30000 else slice(None, None, max(1, len(zz) // 30000))
        ax[2].plot(zz[k], (rr * sg)[k], ".", ms=1.5, alpha=0.3, color=VERM, rasterized=True)
        ax[2].set_title(f"(c) wall deaths — {len(other)} particles")
    else:
        ax[2].set_title("(c) wall deaths — none in file (needs --saveall 1)")
    ax[2].axvline(xap * 1e3, color=GRAY, ls=":", lw=1.2)
    ax[2].set_xlabel("axial position z (mm)")
    ax[2].set_ylabel("radial position r (mm), mirrored")

    fig.suptitle(f"{label}  |  extraction {res['extraction']:.4f} "
                 f"± {res['extraction_stderr']:.4f}  (n = {res['n']})", fontsize=11)
    fig.tight_layout()
    p = outdir / f"{label}_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    if spawnout is not None:
        made.append(heatmap(ext, spawnout, seg, res, outdir, label, *heatbins))
    return made


def heatmap(ext, spawnout, seg, res, outdir, label, nz=26, nr=13, zref=None):
    """Two maps over the same birthplace grid: does a molecule born here get
    out, and if it does, how far off axis has it drifted downstream.

    Needs a `--spawnout` CSV (idx,x,y,z) of every particle's spawn point.
    Panel (a) is the paper's Fig.-5 quantity, (extracted spawns)/(all spawns)
    per bin.  Panel (b) takes only the extracted molecules from each bin and
    reports the mean radius they reach at the plane `zref`, obtained by
    running the recorded final state forward ballistically -- the last leg is
    already collisionless, so this is free flight, not a fit.  Molecules that
    left through the RADIAL face are carried to the same plane, which puts
    them past the simulated boundary; that is where they would be, and
    excluding them would bias the map toward the narrow core.
    """
    sp = pd.read_csv(spawnout)
    r0 = np.hypot(sp["x"], sp["y"]) * 1e3
    z0 = sp["z"].to_numpy() * 1e3
    hit = sp["idx"].isin(set(ext["idx"])).to_numpy()

    zb = np.linspace(z0.min(), z0.max(), nz + 1)
    rb = np.linspace(0.0, r0.max(), nr + 1)
    tot, _, _ = np.histogram2d(z0, r0, bins=[zb, rb])
    got, _, _ = np.histogram2d(z0[hit], r0[hit], bins=[zb, rb])
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(tot >= 20, got / tot, np.nan)
    # A volume-weighted spawn puts bin counts proportional to r, so the
    # near-axis column is always the thinnest.  Say so rather than letting
    # blank cells read as "no extraction".
    blank = int((tot < 20).sum())
    print(f"  heat map        {nz}x{nr} bins, {blank} blanked (<20 spawns); "
          f"counts/bin min {int(tot.min())} median {int(np.median(tot))} "
          f"max {int(tot.max())}")

    # Downstream radius of the extracted molecules, indexed by birthplace.
    zref = float(res["aperture_exit_x_m"] + 0.0465) if zref is None else zref
    j = sp.merge(ext[["idx", "x", "y", "z", "vx", "vy", "vz"]], on="idx",
                 suffixes=("0", ""))
    ok = (j["vz"] > 0) & (j["z"] < zref)
    j = j[ok]
    t = (zref - j["z"]).to_numpy() / j["vz"].to_numpy()
    rref = np.hypot(j["x"] + j["vx"] * t, j["y"] + j["vy"] * t).to_numpy() * 1e3
    jz0, jr0 = j["z0"].to_numpy() * 1e3, np.hypot(j["x0"], j["y0"]).to_numpy() * 1e3
    cnt, _, _ = np.histogram2d(jz0, jr0, bins=[zb, rb])
    ssum, _, _ = np.histogram2d(jz0, jr0, bins=[zb, rb], weights=rref)
    with np.errstate(invalid="ignore", divide="ignore"):
        rmap = np.where(cnt >= 20, ssum / cnt, np.nan)
    print(f"  downstream r    plane z = {1e3*zref:.0f} mm, {len(j)} of "
          f"{len(ext)} extracted usable; mean r {np.nanmin(rmap):.1f}-"
          f"{np.nanmax(rmap):.1f} mm across bins")

    fig, axs = plt.subplots(1, 2, figsize=(13.6, 4.4), sharex=True, sharey=True)
    for ax, m, cmap, lab, ttl in (
            (axs[0], frac, "viridis", "extraction probability",
             "(a) does a molecule born here get out?"),
            (axs[1], rmap, "magma", f"mean radius at z = {1e3*zref:.0f} mm (mm)",
             f"(b) if it does, how far off axis at z = {1e3*zref:.0f} mm?")):
        # Robust limits: a handful of thin bins otherwise set the whole scale
        # and flatten the structure everywhere else.
        lo, hi = (np.nanpercentile(m, [2, 98]) if np.isfinite(m).sum() > 20
                  else (np.nanmin(m), np.nanmax(m)))
        im = ax.pcolormesh(zb, rb, m.T, cmap=cmap, vmin=lo, vmax=hi)
        for s0 in seg:
            ax.plot([s0[0] * 1e3, s0[2] * 1e3], [s0[1] * 1e3, s0[3] * 1e3],
                    color="w", lw=1.5)
        ax.set_xlim(zb[0], zb[-1])
        ax.set_ylim(rb[0], rb[-1])
        fig.colorbar(im, ax=ax, label=lab)
        ax.set_xlabel("initial axial position (mm)")
        ax.set_title(ttl, fontsize=10.5)
        ax.grid(False)
    axs[0].set_ylabel("initial radial position (mm)")
    fig.suptitle(f"{label}: beam properties vs birthplace "
                 f"(bins with < 20 counts blank)", fontsize=11)
    fig.tight_layout()
    p = outdir / f"{label}_heatmap.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def report(res):
    w = res["n_from_max_idx"]
    return "\n".join([
        f"  file             {Path(res['outfile']).name}",
        f"  n                {res['n']}" + ("   (from max idx -- pass -N for the exact count)" if w else ""),
        f"  rows / extracted {res['n_rows']} / {res['n_extracted']}"
        + (f"   (+{res['n_not_extracted']} not extracted)" if res["n_not_extracted"] else ""),
        f"  extraction       {res['extraction']:.4f} +- {res['extraction_stderr']:.4f}",
        f"  extraction axial {res['extraction_axial']:.4f} +- {res['extraction_axial_stderr']:.4f}"
        f"  ({res['n_axial_exit']} of {res['n_extracted']} extracted, downstream face only)",
        f"  v_f median       {res['vf_median_mps']:.1f} m/s   (mean {res['vf_mean_mps']:.1f})",
        f"  v_f FWHM         {res['vf_fwhm_mps']:.1f} m/s",
        f"  v_t FWHM         {res['vt_fwhm_mps']:.1f} m/s  (azimuth-projected; "
        f"raw signed v_x gives {res['vt_fwhm_signed_vx_mps']:.1f})",
        f"  exit azimuth     chi2/dof {res['azimuth_chi2_per_dof']:.1f}, "
        f"<v_x> {res['vx_mean_mps']:+.2f} m/s"
        + ("   <-- NOT azimuthally uniform (point/uniformplane spawn, or a "
           "pre-diff-9 uniformcell run)"
           if res["azimuth_chi2_per_dof"] > 3 else ""),
        f"  divergence FWHM  {res['divergence_fwhm_mrad']:.0f} mrad  (all exits)",
        f"                   {res['divergence_fwhm_mrad_axial_exit']:.0f} mrad  "
        f"(downstream face only; {100*res['radial_exit_frac']:.0f}% left radially)",
        f"  collisions       {res['collides_mean_extracted']:.0f} mean (extracted)",
        f"  transit time     {res['time_mean_extracted_s']*1e3:.3f} ms mean (extracted)",
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outfile")
    ap.add_argument("surfs")
    ap.add_argument("-N", "--nparticles", type=int, default=None,
                    help="particles simulated (-n of the tracer run); "
                         "defaults to max idx, a slight underestimate")
    ap.add_argument("--spawnout", default=None,
                    help="CSV of spawn positions (idx,x,y,z) -> extraction heat map")
    ap.add_argument("--json", default=None, help="write results as JSON")
    ap.add_argument("--outdir", default=None, help="figure directory (default: no figures)")
    ap.add_argument("--heatbins", default="26,13",
                    help="heat-map bin counts as NZ,NR (default 26,13)")
    ap.add_argument("--label", default=None, help="figure basename (default: outfile stem)")
    a = ap.parse_args()

    d, ext, other, seg, xap, res = analyze(a.outfile, a.surfs, a.nparticles)
    label = a.label or Path(a.outfile).stem
    print(report(res))

    if a.outdir:
        hb = tuple(int(v) for v in a.heatbins.split(","))
        for p in figures(d, ext, other, seg, xap, res, a.outdir, label,
                         a.spawnout, hb):
            print(f"  wrote            {p}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2))
        print(f"  wrote            {a.json}")


if __name__ == "__main__":
    main()
