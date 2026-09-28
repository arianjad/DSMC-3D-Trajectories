"""Validate and summarize explicit 3D observation events.

This is the event-analysis component. The enclosing molecule run will bind its
CSV, launched count and prepared case before writing a joined report.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path


EVENT_SCHEMA = "dsmc-observable-events-v1"
EVENT_COLUMNS = (
    "schema_version", "particle_id", "crossing_index", "plane", "direction",
    "time_s", "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
    "u_m", "v_m", "vu_m_s", "vv_m_s", "vn_m_s",
)


def _finite(row: dict, name: str, line: int) -> float:
    try:
        value = float(row[name])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"event line {line}: invalid {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"event line {line}: nonfinite {name}")
    return value


def _int(row: dict, name: str, line: int) -> int:
    value = row[name]
    if not value or not value.isdecimal() or value.startswith("0"):
        raise ValueError(f"event line {line}: invalid {name}")
    return int(value)


def _dot(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def _near(actual: float, expected: float, *, length=False) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9,
                        abs_tol=1e-10 if length else 1e-8)


def _mean(values) -> float | None:
    return sum(values) / len(values) if values else None


def _first_forward_stats(events: list[dict]) -> dict:
    """Summarize one earliest forward event per particle, so recrossers carry no extra weight."""
    first = {}
    for event in events:
        if event["direction"] == "forward":
            old = first.get(event["particle_id"])
            if old is None or (event["time_s"], event["crossing_index"]) < (
                    old["time_s"], old["crossing_index"]):
                first[event["particle_id"]] = event
    rows = list(first.values())
    azimuths = [math.atan2(e["vv_m_s"], e["vu_m_s"]) for e in rows
                if math.hypot(e["vu_m_s"], e["vv_m_s"]) > 0]
    c = _mean([math.cos(a) for a in azimuths])
    s = _mean([math.sin(a) for a in azimuths])
    resultant = math.hypot(c, s) if c is not None else None
    return {
        "sample": "first-forward-event-per-particle",
        "count": len(rows),
        "mean_u_m": _mean([e["u_m"] for e in rows]),
        "mean_v_m": _mean([e["v_m"] for e in rows]),
        "mean_vu_m_s": _mean([e["vu_m_s"] for e in rows]),
        "mean_vv_m_s": _mean([e["vv_m_s"] for e in rows]),
        "mean_arrival_time_s": _mean([e["time_s"] for e in rows]),
        "mean_polar_angle_rad": _mean([e["polar_angle_rad"] for e in rows]),
        "azimuth_sample_count": len(azimuths),
        "azimuth_resultant_length": resultant,
        "azimuth_circular_mean_rad": (math.atan2(s, c) if resultant is not None
                                      and resultant > 1e-12 else None),
    }


def analyze_observable_events(events_path: Path, planes: list[dict],
                              launched_particles: int) -> dict:
    """Return conditional plane statistics; launched count must come from run evidence.

    `angle_rad` is polar deflection from the plane's forward normal:
    atan2(hypot(vu,vv),vn). Acceptance is an inclusive cut on forward events.
    A molecule passes a plane if any of its forward events passes every declared
    cut. Null cuts mean the plane is exploratory and has no accepted yield.
    """
    if type(launched_particles) is not int or launched_particles <= 0:
        raise ValueError("launched_particles must be a positive evidenced integer")
    if not planes or any(type(p) is not dict for p in planes):
        raise ValueError("at least one declared observation plane is required")
    by_name = {p["name"]: p for p in planes}
    if len(by_name) != len(planes):
        raise ValueError("duplicate observation-plane name")
    by_plane = defaultdict(list)
    by_particle = defaultdict(list)
    with Path(events_path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(EVENT_COLUMNS):
            raise ValueError("event CSV header differs from dsmc-observable-events-v1")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"event line {line}: wrong column count")
            if row["schema_version"] != EVENT_SCHEMA:
                raise ValueError(f"event line {line}: unsupported schema")
            pid = _int(row, "particle_id", line)
            index = _int(row, "crossing_index", line)
            if pid > launched_particles:
                raise ValueError(f"event line {line}: particle ID exceeds launched count")
            name = row["plane"]
            if name not in by_name:
                raise ValueError(f"event line {line}: undeclared plane {name!r}")
            if row["direction"] not in ("forward", "backward"):
                raise ValueError(f"event line {line}: invalid direction")
            numbers = {key: _finite(row, key, line) for key in EVENT_COLUMNS[5:]}
            if numbers["time_s"] < 0:
                raise ValueError(f"event line {line}: negative arrival time")
            plane = by_name[name]
            point = tuple(numbers[k] for k in ("x_m", "y_m", "z_m"))
            velocity = tuple(numbers[k] for k in ("vx_m_s", "vy_m_s", "vz_m_s"))
            offset = tuple(x - o for x, o in zip(point, plane["origin_m"]))
            if (not _near(_dot(offset, plane["normal"]), 0, length=True) or
                    not _near(numbers["u_m"], _dot(offset, plane["axis_u"]), length=True) or
                    not _near(numbers["v_m"], _dot(offset, plane["axis_v"]), length=True) or
                    any(not _near(numbers[key], _dot(velocity, plane[axis]))
                            for key, axis in (("vu_m_s", "axis_u"),
                                              ("vv_m_s", "axis_v"),
                                              ("vn_m_s", "normal")))):
                raise ValueError(f"event line {line}: Cartesian and plane-local columns disagree")
            vn = numbers["vn_m_s"]
            if ((row["direction"] == "forward" and vn <= 0) or
                    (row["direction"] == "backward" and vn >= 0)):
                raise ValueError(f"event line {line}: direction disagrees with normal velocity")
            event = {**numbers, "particle_id": pid, "crossing_index": index,
                     "direction": row["direction"],
                     "polar_angle_rad": math.atan2(
                         math.hypot(numbers["vu_m_s"], numbers["vv_m_s"]), vn)}
            by_plane[name].append(event)
            by_particle[pid].append(event)
    for pid, rows in by_particle.items():
        indices = [row["crossing_index"] for row in rows]
        times = [row["time_s"] for row in rows]
        if indices != list(range(1, len(rows) + 1)) or times != sorted(times):
            raise ValueError(f"particle {pid}: crossings are not sequential in time")
    summaries = []
    for plane in planes:
        rows = by_plane[plane["name"]]
        forward = [row for row in rows if row["direction"] == "forward"]
        backward = [row for row in rows if row["direction"] == "backward"]
        acceptance = plane["acceptance"]
        has_cut = any(bounds is not None for bounds in acceptance.values())
        accepted = []
        if has_cut:
            for row in forward:
                values = {"position_u_m": row["u_m"], "position_v_m": row["v_m"],
                          "angle_rad": row["polar_angle_rad"], "time_s": row["time_s"]}
                if all(bounds is None or bounds[0] <= values[key] <= bounds[1]
                       for key, bounds in acceptance.items()):
                    accepted.append(row)
        accepted_ids = {row["particle_id"] for row in accepted}
        summaries.append({
            "name": plane["name"], "normal": plane["normal"],
            "axis_u": plane["axis_u"], "axis_v": plane["axis_v"],
            "event_count": len(rows), "forward_event_count": len(forward),
            "backward_event_count": len(backward),
            "forward_unique_particles": len({row["particle_id"] for row in forward}),
            "backward_unique_particles": len({row["particle_id"] for row in backward}),
            "first_forward": _first_forward_stats(forward),
            "acceptance": {"status": "evaluated" if has_cut else "unassessed",
                           "bounds": acceptance,
                           "angle_definition": "atan2(hypot(vu_m_s,vv_m_s),vn_m_s) from forward normal",
                           "event_count": len(accepted) if has_cut else None,
                           "unique_particles": len(accepted_ids) if has_cut else None,
                           "fraction_of_launched": len(accepted_ids) / launched_particles
                           if has_cut else None},
        })
    return {"schema_version": "dsmc-observable-summary-v1",
            "source_event_schema": EVENT_SCHEMA,
            "launched_particles": launched_particles,
            "denominator": "evidenced launched molecule count",
            "event_count": sum(len(rows) for rows in by_plane.values()),
            "planes": summaries}
