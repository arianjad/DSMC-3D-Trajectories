"""Station-pair diagnostics for approach to equilibrium.

Let ``balance = Qout/Qin``.  A station imbalance is a flow observable, not a
universal fractional distance from equilibrium.  Under the explicit reservoir
model ``Qout = c*N**p`` with unchanged inflow,

    N_eq = N * balance**(-1/p).

For the special linear case ``p=1``, ``(N_eq-N)/N = 1/balance-1`` while
``(N_eq-N)/N_eq = 1-balance``.  The mapping also assumes that the measured
station flow represents the same reservoir inventory.  This script reads the four
station dumps (tube fwd/rev, aperture fwd/rev; SPARTA `dump surf` of a
`fix ave/surf` nflux), stitches any number of run directories by timestep,
and reports balance per window with a block-scatter error bar.  Works for
the 2D decks (`tf.out` ...) and the 3D decks (`equil-tf.out` ...).

    python tools/equilibrium.py RUNDIR [RUNDIR ...] --dt 1e-7 --window 10 --out fig.png

By default this reports observables without declaring convergence.  Use
``--assessment 3d-balance`` for the validated 3D last-window rule, or
``--assessment b5-2d`` for the B5-specific 2D evidence requirements.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

SCCM = 4.478e17
PREFIXES = ("equil-", "measure-", "")


def read_dump(path):
    """Return {timestep: sum of column 2 over surfs} for a dump surf file."""
    out, ts, acc, mode = {}, None, 0.0, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP"):
                if ts is not None:
                    out[ts] = acc
                mode, acc = 1, 0.0
                continue
            if mode == 1:
                ts, mode = int(line), 2
                continue
            if line.startswith("ITEM: SURFS"):
                mode = 3
                continue
            if line.startswith("ITEM"):
                mode = 4
                continue
            if mode == 3:
                p = line.split()
                if len(p) >= 2:
                    acc += float(p[1])
    if ts is not None:
        out[ts] = acc
    return out


def find(rundir, name):
    for pre in PREFIXES:
        p = os.path.join(rundir, pre + name + ".out")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{name}.out not found under {rundir}")


def load(rundirs):
    series = {k: {} for k in ("tf", "tr", "af", "ar")}
    inv = {}
    for rd in rundirs:
        for k in series:
            series[k].update(read_dump(find(rd, k)))
        for p in glob.glob(os.path.join(rd, "*inventory.out")):
            for line in open(p):
                if not line.startswith("#"):
                    a, b = line.split()[:2]
                    inv[int(a)] = float(b)
    steps = sorted(set.intersection(*(set(v) for v in series.values())))
    steps = [s for s in steps if series["tf"][s] > 0]  # drop the step-0 empty dump
    nap = [series["af"][s] - series["ar"][s] for s in steps]
    ntb = [series["tf"][s] - series["tr"][s] for s in steps]
    return steps, nap, ntb, inv


def mean_sem(x):
    n = len(x)
    m = sum(x) / n
    v = sum((xi - m) ** 2 for xi in x) / max(n - 1, 1)
    return m, math.sqrt(v / n)


def windows(steps, nap, ntb, dt, w):
    rows = []
    for i in range(0, len(steps) - w + 1, w):
        a, ea = mean_sem(nap[i:i + w])
        b, eb = mean_sem(ntb[i:i + w])
        bal = a / b
        err = bal * math.sqrt((ea / a) ** 2 + (eb / b) ** 2)
        t0, t1 = steps[i] * dt, steps[i + w - 1] * dt
        rows.append((0.5 * (t0 + t1) * 1e3, a / SCCM, ea / SCCM, b / SCCM, eb / SCCM, bal, err))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rundirs", nargs="+")
    ap.add_argument("--dt", type=float, required=True, help="timestep in s")
    ap.add_argument("--window", type=int, default=10, help="dumps per window")
    ap.add_argument("--out", help="PNG path; omit for table only")
    ap.add_argument("--title", default="")
    ap.add_argument("--fit", action="store_true",
                    help="fit 1-balance = D0*exp(-t/tau) to the windows (leaky-reservoir model)")
    ap.add_argument("--tmin", type=float, default=0.0, help="ms; drop dumps before this")
    ap.add_argument("--tmax", type=float, default=float("inf"), help="ms; drop dumps after this")
    ap.add_argument("--assessment", choices=("report-only", "b5-2d", "3d-balance"),
                    default="report-only",
                    help="scientific interpretation to print (default: measurements only)")
    args = ap.parse_args(argv)

    steps, nap, ntb, inv = load(args.rundirs)
    keep = [i for i, s in enumerate(steps) if args.tmin <= s * args.dt * 1e3 <= args.tmax]
    steps, nap, ntb = [steps[i] for i in keep], [nap[i] for i in keep], [ntb[i] for i in keep]
    rows = windows(steps, nap, ntb, args.dt, args.window)
    if not rows:
        print(f"only {len(steps)} dumps so far, fewer than one window of {args.window}; nothing to score yet")
        return 0
    print(f"{len(steps)} dumps, {steps[0]*args.dt*1e3:.2f}-{steps[-1]*args.dt*1e3:.2f} ms, "
          f"window = {args.window} dumps = {args.window*(steps[1]-steps[0])*args.dt*1e3:.2f} ms")
    print(f"{'t_ms':>8} {'net_ap':>8} {'+-':>6} {'net_tube':>8} {'+-':>6} {'balance':>8} {'+-':>7} {'1-bal %':>8}")
    for r in rows:
        print(f"{r[0]:8.2f} {r[1]:8.3f} {r[2]:6.3f} {r[3]:8.3f} {r[4]:6.3f} {r[5]:8.4f} {r[6]:7.4f} {100*(1-r[5]):8.2f}")
    last = rows[-1]
    h = len(nap) // 2
    a2, _ = mean_sem(nap[h:])
    b2, _ = mean_sem(ntb[h:])
    print(f"last-half balance {a2/b2:.4f}  net_ap {a2/SCCM:.3f} SCCM  net_tube {b2/SCCM:.3f} SCCM")
    if inv:
        ks = sorted(k for k in inv if k >= steps[h])
        if len(ks) > 1:
            print(f"inventory over last half: {inv[ks[0]]:.4e} -> {inv[ks[-1]]:.4e} "
                  f"({100*(inv[ks[-1]]/inv[ks[0]]-1):+.2f} %)")
    metric = f"last window 1-balance = {100*(1-last[5]):+.2f} +- {100*last[6]:.2f} %"
    if args.assessment == "3d-balance":
        verdict = "CONVERGED" if abs(1 - last[5]) < 0.02 else "NOT YET"
        print(f"{metric}  ->  {verdict} (3D 2 % rule)")
    else:
        print(metric)
    if args.assessment == "b5-2d":
        print("2D B5 EQUILIBRIUM: INCONCLUSIVE from station histories alone.")
        print("Required: a contemporaneous body-density time series and a first-quarter comparison")
        print("of n(t) b(t)^(-1/p), using the B5 calibration p = 1.14 with autocorrelation-aware error.")
        print("That calibration is conditional on the B5 geometry and validated flow range; do not reuse it for arbitrary geometry.")
    fit = None
    if args.fit:
        import numpy as np
        from scipy.optimize import curve_fit
        tt = np.array([r[0] for r in rows])
        dd = np.array([1 - r[5] for r in rows])
        ee = np.array([r[6] for r in rows])
        f = lambda t, d0, tau: d0 * np.exp(-t / tau)
        try:
            if len(tt) < 3:
                raise RuntimeError("%d window(s); a two-parameter exponential needs "
                                   "at least 3 -- shorten --window or lengthen the run"
                                   % len(tt))
            p, cov = curve_fit(f, tt, dd, p0=(dd[0], 50.0), sigma=ee, absolute_sigma=True, maxfev=20000)
            span = tt[-1] - tt[0]
            d0err = math.sqrt(cov[0][0])
            if abs(p[0]) < 2 * d0err or not (0 < p[1] < 5 * span):
                print(f"fit: no resolved approach (D0 = {100*p[0]:+.2f} +- {100*d0err:.2f} %, "
                      f"tau = {p[1]:.0f} ms over a {span:.1f} ms span); the run is flat, tau is not measurable")
            else:
                fit = (p[0], p[1], math.sqrt(cov[1][1]))
                print(f"fit 1-balance = D0 exp(-t/tau): D0 = {100*p[0]:+.2f} %, tau = {p[1]:.1f} +- {fit[2]:.1f} ms")
        except (RuntimeError, TypeError) as exc:
            # TypeError is what scipy raises when points < parameters.
            print("fit failed:", exc)

    if not args.out:
        return 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [s * args.dt * 1e3 for s in steps]
    n = 3 if inv else 2
    fig, ax = plt.subplots(n, 1, figsize=(8, 2.6 * n), sharex=True)
    ax[0].plot(t, [x / SCCM for x in nap], ".", ms=2, color="C0", alpha=0.4, label="net aperture (per dump)")
    ax[0].plot(t, [x / SCCM for x in ntb], ".", ms=2, color="C1", alpha=0.4, label="net tube (per dump)")
    ax[0].errorbar([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], fmt="o-", ms=3, color="C0")
    ax[0].errorbar([r[0] for r in rows], [r[3] for r in rows], [r[4] for r in rows], fmt="s-", ms=3, color="C1")
    ax[0].set_ylabel("SCCM")
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].set_title(args.title)
    ax[1].axhspan(0.98, 1.02, color="0.9")
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].errorbar([r[0] for r in rows], [r[5] for r in rows], [r[6] for r in rows], fmt="o-", ms=3, color="C2")
    if fit:
        import numpy as np
        tt = np.linspace(rows[0][0], rows[-1][0], 200)
        ax[1].plot(tt, 1 - fit[0] * np.exp(-tt / fit[1]), "--", color="k", lw=1,
                   label=f"1 - D0 exp(-t/tau), tau = {fit[1]:.0f} ms")
        ax[1].legend(fontsize=8, loc="lower right")
    ax[1].set_ylabel("balance = out/in")
    if inv:
        ks = sorted(inv)
        ax[2].plot([k * args.dt * 1e3 for k in ks], [inv[k] for k in ks], "-", color="C3")
        ax[2].set_ylabel("inventory (atoms)")
    ax[-1].set_xlabel("time (ms)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
