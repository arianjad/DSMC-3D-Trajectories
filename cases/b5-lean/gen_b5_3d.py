#!/usr/bin/env python3
"""3D B5 hardware cell as a revolve of the gen_b5.py profile, emitting-cap inlet.

Same gas volume as cell_b5.surf: bore r 0.825 mm from the cap at XC to x = 0,
cell r 19.05 mm from 0 to 63.5 mm, 5 mm aperture channel through the 1.5 mm
plate (0.0635-0.0650), open to the plume box.  Built with the repo's own
`revolve_profile` (tools/revolve_surf.py) exactly as the emit-surf
microtests did, so the winding is the one already gated in 3D:
normals point INTO the fluid.

  start cap  type 2  at x = XC, radius RB   -> normal +x, the EMITTER
  walls      type 1                          -> normals into the gas
  end cap    type 99 at the channel exit     -> DROPPED (revolve_profile
             always seals the far end; here it would close the aperture)

Stations: transparent discs, mid-cell in x at the local finest level, radius
past the wall into the solid (tube 1.1 mm, aperture 3.8 mm), one disc per
orientation (fwd = -x normal registers +x movers; rev = reversed winding).

Grid this is designed for (in.he_b5_3d_mflow): base 0.8 mm cells from
x = -0.005, 6 levels of 2x refinement, tube_l6 25 um around the stub,
throat L4 100 um over the plate channel.  Non-face asserted for the cap,
the bore radius and both stations at every level that touches them.

Run: conda run -n claude-code python cases/b5-lean/gen_b5_3d.py [STUB]
where STUB is a key of STUBS below: "" (default, 4.9625 mm, writes
cell_b5_3d.surf and the four station discs) or "s25" (2.4875 mm, writes only
cell_b5_3d_s25.surf and reuses the same stations).  The box, the base lattice
and every coordinate downstream of the cell face are the same in both.
"""
import argparse
import math
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
import revolve_surf as gen  # noqa: E402

H = 0.8e-3                  # base cell (deck: create_grid 150 63 63 over 0.120 x 0.050 x 0.050)
XLO = -0.005
RB = 0.825e-3
RCELL = 0.01905
XPLATE, XEXIT, RAP = 0.0635, 0.0650, 0.0025
# Stub variants.  The value is the cap's index on the L6 lattice measured from
# XLO, in units of H/32 = 25 um; a HALF-INTEGER puts the cap mid-cell at L6 and
# therefore off a cell face at every coarser level too (asserted below).  The
# box and the base lattice never move, so every other coordinate in this file
# keeps the placement it was verified with.
STUBS = {"": 1.5, "s25": 100.5}     # 4.9625 mm and 2.4875 mm of simulated stub
XTUBE = -0.001975           # tube station: mid-cell at L5 (50 um) in the tube_buffer_l5 band
XAP = 0.064225              # aperture station: mid-cell at L4 (100 um) in the throat band
R_TUBE_ST, R_AP_ST = 0.0011, 0.0038
NTHETA = 48
QUANT = 1e-12
DROP = 99


def nonface(name, coord, origin, finest_level):
    for lev in range(1, finest_level + 1):
        h = H / 2 ** (lev - 1)
        idx = (coord - origin) / h
        assert abs(idx - round(idx)) > 1e-6, "%s on a cell face at L%d (index %.4f)" % (name, lev, idx)
    print("%-8s %.7g  non-face down to L%d (%.3g um)" % (name, coord, finest_level, H / 2 ** (finest_level - 1) * 1e6))


def reverse(mesh):
    return gen.Mesh(mesh.points, tuple((t, a, c, b) for (t, a, b, c) in mesh.triangles))


def main(tag="", outdir=None):
    if tag not in STUBS:
        raise ValueError("unknown stub variant %r" % tag)
    output_dir = Path(HERE) if outdir is None else Path(outdir)
    if outdir is not None and (output_dir.exists() or output_dir.is_symlink()):
        raise FileExistsError("refusing to overwrite existing output path: %s" % output_dir)
    XC = XLO + STUBS[tag] * H / 32
    suffix = ("_" + tag) if tag else ""
    print("stub variant %-4s cap x %.7g, simulated stub %.4f mm"
          % (tag or "(5mm)", XC, -XC * 1e3))
    nonface("cap x", XC, XLO, 6)
    nonface("tube st", XTUBE, XLO, 5)
    nonface("ap st", XAP, XLO, 4)
    # y/z: box -0.025..0.025 with 63 base cells (0.7937 mm), so the axis sits
    # at a cell CENTRE (index 31.5) and a radius r maps to index 31.5 + r/h.
    HY = 0.05 / 63
    for name, r, lev in (("bore r", RB, 6), ("r_cell", RCELL, 2), ("r_ap", RAP, 4)):
        for L in range(1, lev + 1):
            h = HY / 2 ** (L - 1)
            idx = (0.025 + r) / h
            assert abs(idx - round(idx)) > 1e-6, "%s on a y face at L%d (index %.4f)" % (name, L, idx)
        print("%-8s %.7g  non-face in y/z down to L%d (%.3g um)" % (name, r, lev, HY / 2 ** (lev - 1) * 1e6))

    # The last segment is the plate's DOWNSTREAM face, from the channel rim out
    # to R_OUT beyond the box half-diagonal (0.025*sqrt2 = 35.4 mm): after the
    # end cap is dropped the only open rim is at R_OUT, outside the box, and
    # `clip` removes it.  Ending at the channel rim instead leaves 48 unmatched
    # edges and read_surf fails its watertight check (seen 2026-09-04).
    R_OUT = 0.040
    profile = ((XC, RB), (0.0, RB), (0.0, RCELL), (XPLATE, RCELL), (XPLATE, RAP), (XEXIT, RAP), (XEXIT, R_OUT))
    types = [1, 1, 1, 1, 1, 1]
    full = gen.revolve_profile(profile, NTHETA, types, 2, DROP, QUANT)
    kept = tuple(t for t in full.triangles if t[0] != DROP)
    mesh = gen.Mesh(full.points, kept)
    assert len(full.triangles) - len(kept) == NTHETA, "expected exactly one dropped end cap"
    if outdir is not None:
        output_dir.mkdir(parents=True)
    roles = {1: "wall (diffuse 4 K)", 2: "start cap = EMITTER (diffuse 4 K, fix emit/surf mflow)"}
    cell_path = output_dir / ("cell_b5_3d%s.surf" % suffix)
    gen.write_surface(cell_path, mesh,
                      "B5 hardware cell revolve, %d azimuthal segments; end cap at the channel exit "
                      "dropped so the aperture is open; read with `type group cellwalls clip`, then "
                      "`group cap surf type 2`" % NTHETA, roles)
    print("wrote cell_b5_3d%s.surf: %d points, %d triangles (%d end-cap triangles dropped)"
          % (suffix, len(mesh.points), len(kept), len(full.triangles) - len(kept)))
    # Legacy in-place s25 output reuses the default's station files. An explicit
    # export is self-contained, so it writes those same four station files too.
    stations = (("tube", XTUBE, R_TUBE_ST), ("ap", XAP, R_AP_ST)) if (outdir is not None or not tag) else ()
    for name, x, r in stations:
        disc = gen.diagnostic_disk(x, r, NTHETA, 1, (-1.0, 0.0, 0.0), QUANT)
        fwd_path = output_dir / ("%s_fwd3d.surf" % name)
        gen.write_surface(fwd_path, disc,
                          "transparent %s station at x = %.7g, normal -x, registers +x movers" % (name, x))
        rev_path = output_dir / ("%s_rev3d.surf" % name)
        gen.write_surface(rev_path, reverse(disc),
                          "same disc reversed, normal +x, registers -x movers")
        print("wrote %s_fwd3d.surf / %s_rev3d.surf" % (name, name))
    poly = NTHETA / (2 * math.pi) * math.sin(2 * math.pi / NTHETA)
    print("48-gon cap area / circle = %.5f (emit/surf reproduces the polygon; mflow is exact regardless)" % poly)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate B5 3D surface inputs.")
    parser.add_argument("stub", nargs="?", choices=STUBS, default="")
    parser.add_argument("--outdir", type=Path,
                        help="fresh directory for a self-contained five-surface export")
    args = parser.parse_args()
    main(args.stub, args.outdir)
