#!/usr/bin/env python3
"""Export a B5 cell with a rectangular aperture through the existing plate.

SPARTA x is the beam axis.  A vertical slit runs along SPARTA z (tracer y);
its horizontal width runs along SPARTA y (tracer x).  The circular B5 input is
never changed.  The default area basis is the analytic 5-mm circle; pass
``--area-basis polygon`` to match its existing 48-gon aperture area instead.
"""
import argparse
import math
from pathlib import Path

import gen_b5_3d as base


R_OUT = 0.040
AP_STATION_MARGIN = 0.2e-3


def reverse(mesh):
    return base.reverse(mesh)


class Builder:
    def __init__(self):
        self.points, self.index, self.triangles = [], {}, []

    def point(self, value):
        point = tuple(base.gen._quantize_coordinate(v, base.QUANT) for v in value)
        if point not in self.index:
            self.index[point] = len(self.points)
            self.points.append(point)
        return self.index[point]

    def quad(self, corners, normal):
        a, b, c, d = (self.point(p) for p in corners)
        pa, pb, pc = (self.points[i] for i in (a, b, c))
        cross = ((pb[1] - pa[1]) * (pc[2] - pa[2]) - (pb[2] - pa[2]) * (pc[1] - pa[1]),
                 (pb[2] - pa[2]) * (pc[0] - pa[0]) - (pb[0] - pa[0]) * (pc[2] - pa[2]),
                 (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0]))
        if sum(a * b for a, b in zip(cross, normal)) < 0:
            b, d = d, b
        self.triangles.extend(((1, a, b, c), (1, a, c, d)))

    def mesh(self):
        return base.gen.Mesh(tuple(self.points), tuple(self.triangles))


def combine(*meshes):
    """Merge meshes by quantized coordinates, retaining shared seam vertices."""
    points, point_index, triangles = [], {}, []
    for mesh in meshes:
        remap = []
        for point in mesh.points:
            if point not in point_index:
                point_index[point] = len(points)
                points.append(point)
            remap.append(point_index[point])
        triangles.extend((typ, remap[a], remap[b], remap[c])
                         for typ, a, b, c in mesh.triangles)
    return base.gen.Mesh(tuple(points), tuple(triangles))


def aperture_dimensions(aspect, area_basis):
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError("--aspect must be finite and positive")
    circle_area = math.pi * base.RAP ** 2
    polygon_area = 0.5 * base.NTHETA * base.RAP ** 2 * math.sin(2 * math.pi / base.NTHETA)
    target = circle_area if area_basis == "circle" else polygon_area
    horizontal = math.sqrt(target / aspect)  # SPARTA y / tracer x
    vertical = aspect * horizontal           # SPARTA z / tracer y
    if not math.isfinite(horizontal) or not math.isfinite(vertical):
        raise ValueError("--aspect produces non-finite slit dimensions")
    if not slit_fits_body(horizontal, vertical):
        raise ValueError("slit corners must fit within the B5 body polygon")
    return horizontal, vertical, circle_area, polygon_area


def slit_fits_body(width, height):
    """Require every rectangular corner to lie inside the 48-sided body rim."""
    body = circular_loop(base.RCELL)
    for point in ((width / 2, height / 2), (width / 2, -height / 2),
                  (-width / 2, height / 2), (-width / 2, -height / 2)):
        for i, start in enumerate(body):
            end = body[(i + 1) % len(body)]
            cross = ((end[0] - start[0]) * (point[1] - start[1])
                     - (end[1] - start[1]) * (point[0] - start[0]))
            if cross < -1e-14:
                return False
    return True


def rectangular_loop(width, height):
    """48 counter-clockwise boundary points, starting at (y,z) = (+halfwidth,0).

    The loop has the B5 azimuthal count so it can be sewn to the cylindrical
    body ring.  Its corners are explicit: a ray-only rectangle would bevel
    them whenever an azimuth did not happen to land on a corner.
    """
    hy, hz = width / 2, height / 2
    points = [(hy, i * hz / 6) for i in range(7)]
    points += [(hy - i * 2 * hy / 12, hz) for i in range(1, 13)]
    points += [(-hy, hz - i * 2 * hz / 12) for i in range(1, 13)]
    points += [(-hy + i * 2 * hy / 12, -hz) for i in range(1, 13)]
    points += [(hy, -hz + i * hz / 6) for i in range(1, 6)]
    assert len(points) == base.NTHETA
    return points


def circular_loop(radius):
    return [(radius * math.cos(2 * math.pi * i / base.NTHETA),
             radius * math.sin(2 * math.pi * i / base.NTHETA))
            for i in range(base.NTHETA)]


def plate_and_slit(width, height):
    """Conforming plate annuli and duct walls, with normals into the gas."""
    b = Builder()
    inner = rectangular_loop(width, height)
    inlet_outer = circular_loop(base.RCELL)
    outlet_outer = circular_loop(R_OUT)
    for i in range(base.NTHETA):
        j = (i + 1) % base.NTHETA
        # Upstream annulus uses exactly the body endpoint ring, so its outer
        # edge is a shared boundary rather than a plate-interior T junction.
        b.quad(((base.XPLATE, *inlet_outer[i]), (base.XPLATE, *inlet_outer[j]),
                (base.XPLATE, *inner[j]), (base.XPLATE, *inner[i])), (-1.0, 0.0, 0.0))
        # Downstream annulus retains B5's open, clipped outer rim at R_OUT.
        b.quad(((base.XEXIT, *inner[i]), (base.XEXIT, *inner[j]),
                (base.XEXIT, *outlet_outer[j]), (base.XEXIT, *outlet_outer[i])), (1.0, 0.0, 0.0))
        # One duct quad per matching inner-loop edge prevents T junctions at
        # the rectangle sides and preserves inward gas normals.
        yi, zi = inner[i]
        yj, zj = inner[j]
        b.quad(((base.XPLATE, yi, zi), (base.XEXIT, yi, zi),
                (base.XEXIT, yj, zj), (base.XPLATE, yj, zj)),
               (0.0, -(yi + yj), -(zi + zj)))
    return b.mesh()


def cell_mesh(tag, aspect, area_basis):
    xc = base.XLO + base.STUBS[tag] * base.H / 32
    profile = ((xc, base.RB), (0.0, base.RB), (0.0, base.RCELL), (base.XPLATE, base.RCELL))
    full = base.gen.revolve_profile(profile, base.NTHETA, [1, 1, 1], 2, base.DROP, base.QUANT)
    cylindrical = base.gen.Mesh(full.points, tuple(t for t in full.triangles if t[0] != base.DROP))
    width, height, circle_area, polygon_area = aperture_dimensions(aspect, area_basis)
    return combine(cylindrical, plate_and_slit(width, height)), width, height, circle_area, polygon_area


def aperture_station_radius(width, height):
    """Cover every slit corner with a small diagnostic margin."""
    corner_radius = math.hypot(width / 2, height / 2)
    return max(base.R_AP_ST, corner_radius + AP_STATION_MARGIN)


def write_stations(outdir, width, height):
    # The tube stations retain their B5 radius.  The aperture stations measure
    # the entire rectangular opening, rather than imposing the circular B5
    # station's 3.8-mm finite-radius acceptance on a taller slit.
    for name, x, radius in (("tube", base.XTUBE, base.R_TUBE_ST),
                            ("ap", base.XAP, aperture_station_radius(width, height))):
        disc = base.gen.diagnostic_disk(x, radius, base.NTHETA, 1, (-1.0, 0.0, 0.0), base.QUANT)
        base.gen.write_surface(outdir / (name + "_fwd3d.surf"), disc,
                               "transparent %s station at x = %.7g, normal -x, registers +x movers" % (name, x))
        base.gen.write_surface(outdir / (name + "_rev3d.surf"), reverse(disc),
                               "same disc reversed, normal +x, registers -x movers")


def main(tag, outdir, aspect, area_basis):
    if tag not in base.STUBS:
        raise ValueError("unknown stub variant %r" % tag)
    if not math.isfinite(aspect) or aspect <= 0:
        raise ValueError("--aspect must be finite and positive")
    if outdir.exists() or outdir.is_symlink():
        raise FileExistsError("refusing to overwrite existing output path: %s" % outdir)
    mesh, width, height, circle_area, polygon_area = cell_mesh(tag, aspect, area_basis)
    base.nonface("cap x", base.XLO + base.STUBS[tag] * base.H / 32, base.XLO, 6)
    base.nonface("tube st", base.XTUBE, base.XLO, 5)
    base.nonface("ap st", base.XAP, base.XLO, 4)
    outdir.mkdir(parents=True)
    suffix = "_s25" if tag else ""
    path = outdir / ("cell_b5_3d_slit%s.surf" % suffix)
    base.gen.write_surface(path, mesh,
                           "B5 rectangular slit: beam SPARTA x/tracer z; vertical SPARTA z/tracer y; "
                           "area basis %s; aspect %.12g; horizontal %.12g m; vertical %.12g m" %
                           (area_basis, aspect, width, height),
                           {1: "wall (diffuse 4 K)", 2: "start cap = EMITTER (diffuse 4 K, fix emit/surf mflow)"})
    write_stations(outdir, width, height)
    print("wrote %s" % path)
    print("slit horizontal SPARTA-y/tracer-x = %.12g m; vertical SPARTA-z/tracer-y = %.12g m" % (width, height))
    print("areas: slit %.12g m2; analytic circle %.12g m2; existing 48-gon %.12g m2" %
          (width * height, circle_area, polygon_area))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stub", nargs="?", choices=base.STUBS, default="")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--aspect", type=float, default=5.0)
    parser.add_argument("--area-basis", choices=("circle", "polygon"), default="circle")
    args = parser.parse_args()
    main(args.stub, args.outdir, args.aspect, args.area_basis)
