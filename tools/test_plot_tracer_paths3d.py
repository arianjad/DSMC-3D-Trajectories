#!/usr/bin/env python3
"""Integrity gates for recorded 3D trajectory-path input."""
import csv
import tempfile
from pathlib import Path

from plot_tracer_paths3d import load_prepared, main, path_colors, trace_rows, validate_rows


def leg(ordinal):
    return "-1 0 0 0 0 0 0 0 0 0 %d 0\n" % ordinal


def test_missing_negative_row_fails():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "stdout"
        path.write_text(leg(1) + leg(2) + leg(3))
        rows = trace_rows(path, 1)
        validate_rows(rows, {1: 3})

        path.write_text(leg(1) + leg(3))
        rows = trace_rows(path, 1)
        try:
            validate_rows(rows, {1: 3})
        except ValueError:
            pass
        else:
            raise AssertionError("removed negative-index leg was accepted")
    print("PASS")


def test_four_paths_receive_colors():
    assert len(path_colors({1: [], 2: [], 3: [], 4: []})) == 4


PATH_HEADER = ["particle_id", "leg", "x0_m", "y0_m", "z0_m", "x1_m", "y1_m", "z1_m",
               "vx_m_s", "vy_m_s", "vz_m_s", "collisions", "time_s", "initial_collisions"]
# Particle 1: two collision legs then the terminating wall leg (the prepared
# recorder writes it, with the terminal collision count and time).
# Particle 2: a zero-collision path whose only leg is terminal.
LEGS = [
    [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1.0, 0],
    [1, 2, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 2.0, 0],
    [1, 3, 1, 1, 0, 1, 2.5, 0, 0, 1.5, 0, 2, 3.0, 0],
    [2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0.5, 0],
]
# Same geometry, but particle 1 starts at rest: one spawn collision, no leg.
SPAWN_COLLIDED = [r[:11] + [r[11] + 1, r[12], 1] if r[0] == 1 else r for r in LEGS]
TERMINAL = [["particle_id", "fate", "time_s", "x_m", "y_m", "z_m",
             "vx_m_s", "vy_m_s", "vz_m_s", "collisions"],
            [1, "wall", 3.0, 1, 2.5, 0, 0, 1.5, 0, 2],
            [2, "domain-or-cap-exit", 0.5, 0, 0, 1, 0, 0, 2, 0],
            [3, "wall", 9.0, 0, 0, 0, 0, 0, 1, 4]]


def prepared_dir(root, legs, recorded=2, terminal=TERMINAL):
    root.mkdir()
    with (root / "paths.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows([PATH_HEADER, *legs])
    with (root / "terminal.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(terminal)
    (root / "transport-result.toml").write_text("recorded_paths = %d\n" % recorded)
    return root


def expect_rejected(root, why):
    try:
        load_prepared(root)
    except ValueError:
        return
    raise AssertionError("prepared paths accepted despite " + why)


def test_prepared_contract():
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)
        paths, fates = load_prepared(prepared_dir(base / "good", LEGS))
        assert [len(paths[1]), len(paths[2])] == [3, 1]
        assert fates == {1: "wall", 2: "domain-or-cap-exit"}
        spawn_terminal = [TERMINAL[0], TERMINAL[1][:9] + [3], *TERMINAL[2:]]
        load_prepared(prepared_dir(base / "spawn", SPAWN_COLLIDED, terminal=spawn_terminal))
        renumber = lambda legs: [[r[0], i + 1, *r[2:]] if r[0] == 1 else r
                                 for i, r in enumerate(legs)]
        cases = {
            "missing middle leg": [LEGS[0], LEGS[2], LEGS[3]],
            "missing middle leg, renumbered": renumber([LEGS[0], LEGS[2], LEGS[3]]),
            "missing first leg, renumbered": renumber(LEGS[1:]),
            "missing terminal leg": [LEGS[0], LEGS[1], LEGS[3]],
            "reordered legs": [LEGS[1], LEGS[0], LEGS[2], LEGS[3]],
            "reordered legs, renumbered": renumber([LEGS[1], LEGS[0], LEGS[2], LEGS[3]]),
            "interleaved particles": [LEGS[0], LEGS[3], LEGS[1], LEGS[2]],
            "missing particle": LEGS[:3],
            "moved final contact": LEGS[:2] + [LEGS[2][:6] + [2.4] + LEGS[2][7:]] + LEGS[3:],
        }
        for index, (why, legs) in enumerate(cases.items()):
            expect_rejected(prepared_dir(base / str(index), legs), why)
        expect_rejected(prepared_dir(base / "count", LEGS, recorded=3), "too few recorded paths")
    print("PASS prepared contract")


def test_prepared_render():
    with tempfile.TemporaryDirectory() as temporary:
        root = prepared_dir(Path(temporary) / "run", LEGS)
        out = Path(temporary) / "paths.png"
        main(["--prepared", str(root), str(out)])
        assert out.stat().st_size > 1000
    print("PASS prepared render")


if __name__ == "__main__":
    test_missing_negative_row_fails()
    test_four_paths_receive_colors()
    test_prepared_contract()
    test_prepared_render()
