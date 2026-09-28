#!/usr/bin/env python3
"""Export portable, matched 5-mm-circle versus 5:1-vertical-slit B5 inputs.

STUB selects the feed-stub preset shared by both arms: "" (4.9625 mm, deck
in.he_b5_3d_mflow) or "s25" (2.4875 mm, deck in.he_b5_3d_mflow_s25).
"""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import gen_b5_3d as circle
import gen_b5_3d_slit as slit


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ASPECT = 5.0
THROAT_MARGIN = 1.5e-3
THROAT_INCREMENT = 0.1e-3


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def suffix(stub):
    if stub not in circle.STUBS:
        raise ValueError("unknown stub variant %r" % stub)
    return ("_" + stub) if stub else ""


def deck_name(stub):
    return "in.he_b5_3d_mflow" + suffix(stub)


def cell_name(kind, stub):
    """kind is "" for the circle or "_slit"; matches the two generators' outputs."""
    return "cell_b5_3d%s%s.surf" % (kind, suffix(stub))


def common_geometry():
    width, height, _, polygon_area = slit.aperture_dimensions(ASPECT, "polygon")
    corner = math.hypot(width / 2, height / 2)
    station = slit.aperture_station_radius(width, height)
    throat_l4 = math.ceil((corner + THROAT_MARGIN) / THROAT_INCREMENT) * THROAT_INCREMENT
    return width, height, polygon_area, corner, station, throat_l4, throat_l4 + 1.6e-3


def write_common_stations(arm, aperture_radius):
    for name, x, radius in (("tube", circle.XTUBE, circle.R_TUBE_ST),
                            ("ap", circle.XAP, aperture_radius)):
        disc = circle.gen.diagnostic_disk(x, radius, circle.NTHETA, 1, (-1.0, 0.0, 0.0), circle.QUANT)
        circle.gen.write_surface(arm / (name + "_fwd3d.surf"), disc,
                                 "transparent %s station at x = %.7g, normal -x, registers +x movers" % (name, x))
        circle.gen.write_surface(arm / (name + "_rev3d.surf"), circle.reverse(disc),
                                 "same disc reversed, normal +x, registers -x movers")


def paired_deck(cell, throat_l4, throat_buffer, eqthroat, baseline=None, stub=""):
    text = (HERE / deck_name(stub)).read_text() if baseline is None else baseline
    replacements = (
        ("read_surf   %s" % cell_name("", stub), "read_surf   %s" % cell),
        ("region      throat_buffer_l3 cylinder x 0 0 0.0056 0.0607 0.0668",
         "region      throat_buffer_l3 cylinder x 0 0 %.12g 0.0607 0.0668" % throat_buffer),
        ("region      throat_l4 cylinder x 0 0 0.004 0.0615 0.066",
         "region      throat_l4 cylinder x 0 0 %.12g 0.0615 0.066" % throat_l4),
        ("region      eqthroat cylinder x 0 0 0.0038 0.062 0.065",
         "region      eqthroat cylinder x 0 0 %.12g 0.062 0.065" % eqthroat),
    )
    for old, new in replacements:
        count = text.count(old)
        if count != 1:
            raise ValueError("expected one baseline deck pattern, found %d: %s" % (count, old))
        text = text.replace(old, new, 1)
    return text


def write_arm(arm, deck, deck_file):
    (arm / deck_file).write_text(deck, newline="\n")
    for source in ("he.species", "he.vss"):
        shutil.copyfile(DATA / source, arm / source)


def export(outdir, stub=""):
    outdir = Path(outdir)
    if outdir.exists() or outdir.is_symlink():
        raise FileExistsError("refusing to overwrite existing output path: %s" % outdir)
    deck = deck_name(stub)
    circle_cell, slit_cell = cell_name("", stub), cell_name("_slit", stub)
    width, height, polygon_area, corner, station, throat_l4, throat_buffer = common_geometry()
    # Fail before creating output if the producing deck changed under us.
    circle_deck = paired_deck(circle_cell, throat_l4, throat_buffer, station, stub=stub)
    slit_deck = paired_deck(slit_cell, throat_l4, throat_buffer, station, stub=stub)
    outdir.mkdir(parents=True)
    circle_dir, slit_dir = outdir / "circle", outdir / "slit"
    circle.main(stub, circle_dir)
    slit.main(stub, slit_dir, ASPECT, "polygon")
    for arm in (circle_dir, slit_dir):
        write_common_stations(arm, station)
    write_arm(circle_dir, circle_deck, deck)
    write_arm(slit_dir, slit_deck, deck)
    source_hashes = {name: sha256(HERE / name) for name in
                     ("gen_b5_3d.py", "gen_b5_3d_slit.py", deck)}
    receipt = {
        "purpose": "matched default-circle versus approved 5:1 vertical-slit B5 export",
        "geometry": {"area_basis": "existing 48-gon", "aspect_vertical_over_horizontal": ASPECT,
                     "polygon_area_m2": polygon_area, "slit_area_m2": width * height,
                     "slit_horizontal_sparta_y_m": width, "slit_vertical_sparta_z_m": height,
                     "slit_corner_radius_m": corner,
                     "stub_preset": stub or "5mm",
                     "cap_x_m": circle.XLO + circle.STUBS[stub] * circle.H / 32},
        "common": {"aperture_station_radius_m": station, "throat_l4_radius_m": throat_l4,
                   "throat_buffer_l3_radius_m": throat_buffer, "eqthroat_radius_m": station,
                   "base_deck": deck, "grid_and_physics": "identical except cell surface filename"},
        "sources_sha256": source_hashes,
        "copied_data_sha256": {name: sha256(DATA / name) for name in ("he.species", "he.vss")},
    }
    (outdir / "PAIR_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", newline="\n")
    print("wrote matched pair (%s) to %s" % (deck, outdir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stub", nargs="?", choices=circle.STUBS, default="")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    export(args.outdir, args.stub)
