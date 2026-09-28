#!/usr/bin/env python3
"""Stage-2 collision-convention gate (pre-registered).

Compares the measured SPARTA collision event rate in a homogeneous 4 K He box
against hard-sphere kinetic theory:

    events per particle per step  =  (1/2) n sigma <v_rel> dt
    <v_rel> = sqrt(16 kB T / (pi m))       (Maxwell, identical particles)
    n       = np * fnum / V                (exact, from the deck)

stats ncoll is the per-step instantaneous event count (doc/stats_style.txt:
"counts for the current timestep"); each accepted pair increments ncollide_one
once (src/collide.cpp:520). No differencing needed; samples are averaged.

Pre-registered verdicts on R = measured/expected over the steady window
(step >= 2000, excluding vremax break-in):
    PASS         |R - 1| <= 0.05
    INVESTIGATE  0.05 < |R - 1| <= 0.15
    FAIL         |R - 1| > 0.15   (a sigma = pi d^2/4 mixup would give R ~ 0.25)
Exit code 0 only on PASS.

Usage: python3 analyze.py run.log
"""
import math
import sys

KB = 1.380649e-23        # J/K
M_HE = 6.6464791e-27     # kg
SIGMA = 2.0e-19          # m^2, published hard-sphere cross section
FNUM = 2.0e9             # from deck
VOL = 1.0e-6             # m^3, from deck
DT = 1.0e-6              # s, from deck
STEADY_MIN_STEP = 2000

def parse_stats(path):
    rows = []
    cols = None
    with open(path) as f:
        for line in f:
            t = line.split()
            if t[:2] == ["Step", "CPU"]:
                cols = t
                continue
            if cols and len(t) == len(cols):
                try:
                    rows.append(dict(zip(cols, (float(x) for x in t))))
                except ValueError:
                    cols = None  # left the stats block
            elif cols and t and t[0] == "Loop":
                cols = None
    return rows

def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)

def main(path):
    rows = parse_stats(path)
    if not rows:
        sys.exit("no stats rows parsed from %s" % path)
    steady = [r for r in rows if r["Step"] >= STEADY_MIN_STEP]
    early = [r for r in rows if 0 < r["Step"] < STEADY_MIN_STEP]
    if len(steady) < 10:
        sys.exit("only %d steady samples; run too short" % len(steady))

    np_mean, _ = mean_se([r["Np"] for r in steady])
    t_mean, t_se = mean_se([r["c_myT"] for r in steady])
    ncoll_mean, ncoll_se = mean_se([r["Ncoll"] for r in steady])

    n_density = np_mean * FNUM / VOL
    def expected(temp):
        vrel = math.sqrt(16.0 * KB * temp / (math.pi * M_HE))
        return 0.5 * n_density * SIGMA * vrel * DT * np_mean  # events/step

    exp_nom = expected(4.0)
    exp_meas = expected(t_mean)
    r = ncoll_mean / exp_meas
    r_se = ncoll_se / exp_meas

    print("samples: %d steady (step>=%d), %d early" %
          (len(steady), STEADY_MIN_STEP, len(early)))
    print("Np mean            : %.1f" % np_mean)
    print("Temp mean          : %.6f K (SE %.2e)" % (t_mean, t_se))
    print("n (exact)          : %.4e m^-3" % n_density)
    print("measured ncoll/step: %.3f +- %.3f" % (ncoll_mean, ncoll_se))
    print("expected (T=4.000) : %.3f" % exp_nom)
    print("expected (T=meas)  : %.3f" % exp_meas)
    print("R = meas/exp(Tmeas): %.4f +- %.4f" % (r, r_se))
    if early:
        e_mean, e_se = mean_se([r_["Ncoll"] for r_ in early])
        print("early-window ncoll/step: %.3f +- %.3f (break-in check)" % (e_mean, e_se))

    dev = abs(r - 1.0)
    if dev <= 0.05:
        print("VERDICT: PASS")
        return 0
    print("VERDICT: %s" % ("INVESTIGATE" if dev <= 0.15 else "FAIL"))
    return 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "run.log"))
