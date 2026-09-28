"""Revolve a 2D (x, r) profile into a SPARTA 3D triangle surf.

The only geometry helpers the lean 3D decks need (cases/b5-lean/gen_b5_3d.py).
Extracted 2026-09-04 from the retired Codex generator so the live path has no
dependency on it; the functions are byte-for-byte the ones that built every
tracked cell_b5_3d*.surf (regeneration is checked byte-identical).

Coordinates are quantized onto a decimal grid before entering a mesh so the
same profile gives IEEE-identical vertices on Windows and Linux.  Triangle
winding is the inverse of the fluid-volume outward normal, i.e. normals point
INTO the gas, which is what SPARTA's read_surf expects for a wall.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

GENERATOR_SCHEMA = 2
Vec3 = tuple[float, float, float]
Triangle = tuple[int, int, int, int]  # surface type, p1, p2, p3 (zero based)


@dataclass(frozen=True)
class Mesh:
    points: tuple[Vec3, ...]
    triangles: tuple[Triangle, ...]


def _float(value: float) -> str:
    if value == 0:
        return "0"
    return format(value, ".17g")


def _quantize_coordinate(value: float, quantum_m: float) -> float:
    if not math.isfinite(value) or not math.isfinite(quantum_m) or quantum_m <= 0:
        raise ValueError("coordinate quantization inputs must be finite and positive")
    scaled = value / quantum_m
    grid_index = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    if grid_index == 0:
        return 0.0
    return float(Decimal(grid_index) * Decimal(str(quantum_m)))


def _quantized_point(point: Vec3, quantum_m: float) -> Vec3:
    return tuple(_quantize_coordinate(value, quantum_m) for value in point)  # type: ignore[return-value]


def revolve_profile(
    profile: tuple[tuple[float, float], ...], ntheta: int,
    segment_types: list[int], start_type: int, end_type: int,
    coordinate_quantization_m: float,
) -> Mesh:
    """Revolve `profile` = ((x, r), ...) about x with `ntheta` segments.

    Segment i gets surface type segment_types[i]; the two axial ends are
    sealed with discs of type start_type / end_type (drop the end triangles
    yourself if the far end must stay open).
    """
    if len(segment_types) != len(profile) - 1:
        raise ValueError("segment type count does not match profile")
    points: list[Vec3] = []
    for x, radius in profile:
        for index in range(ntheta):
            theta = 2.0 * math.pi * index / ntheta
            points.append(_quantized_point(
                (x, radius * math.cos(theta), radius * math.sin(theta)),
                coordinate_quantization_m,
            ))
    start_center = len(points)
    points.append(_quantized_point((profile[0][0], 0.0, 0.0), coordinate_quantization_m))
    end_center = len(points)
    points.append(_quantized_point((profile[-1][0], 0.0, 0.0), coordinate_quantization_m))
    triangles: list[Triangle] = []
    for ring in range(len(profile) - 1):
        first = ring * ntheta
        second = (ring + 1) * ntheta
        surface_type = segment_types[ring]
        for index in range(ntheta):
            following = (index + 1) % ntheta
            a, an = first + index, first + following
            b, bn = second + index, second + following
            # This winding is the inverse of the fluid-volume outward normal.
            triangles.append((surface_type, a, b, bn))
            triangles.append((surface_type, a, bn, an))
    for index in range(ntheta):
        following = (index + 1) % ntheta
        triangles.append((start_type, start_center, index, following))
        last = (len(profile) - 1) * ntheta
        triangles.append((end_type, end_center, last + following, last + index))
    return Mesh(tuple(points), tuple(triangles))


def cylinder_fixture(xmin: float, xmax: float, radius: float, ntheta: int, coordinate_quantization_m: float) -> Mesh:
    """A closed tube: wall type 1, start cap type 2, end cap type 3."""
    return revolve_profile(
        ((xmin, radius), (xmax, radius)), ntheta, [1], 2, 3,
        coordinate_quantization_m,
    )


def diagnostic_disk(
    x: float, radius: float, ntheta: int, surface_type: int,
    expected_normal: tuple[float, float, float], coordinate_quantization_m: float,
) -> Mesh:
    """A transparent station disc at x with -x normal (registers +x movers)."""
    if expected_normal != (-1.0, 0.0, 0.0):
        raise ValueError("diagnostic_disk supports the -x normal only; reverse the winding for +x")
    points: list[Vec3] = [_quantized_point((x, 0.0, 0.0), coordinate_quantization_m)]
    for index in range(ntheta):
        theta = 2.0 * math.pi * index / ntheta
        points.append(_quantized_point(
            (x, radius * math.cos(theta), radius * math.sin(theta)),
            coordinate_quantization_m,
        ))
    triangles: list[Triangle] = []
    for index in range(ntheta):
        following = (index + 1) % ntheta
        # Reversed ring order gives -x; particles with +x velocity hit that face.
        triangles.append((surface_type, 0, following + 1, index + 1))
    return Mesh(tuple(points), tuple(triangles))


def write_surface(path: Path, mesh: Mesh, description: str, role_names: dict[int, str] | None = None) -> None:
    lines = [
        f"# {description}",
        f"# Generator schema {GENERATOR_SCHEMA}; coordinates are metres; triangle normals are intentional.",
    ]
    if role_names:
        lines.extend(f"# type {surface_type}: {role_names[surface_type]}" for surface_type in sorted(role_names))
    lines += ["", f"{len(mesh.points)} points", f"{len(mesh.triangles)} triangles", "", "Points", ""]
    for index, point in enumerate(mesh.points, 1):
        lines.append(f"{index} {_float(point[0])} {_float(point[1])} {_float(point[2])}")
    lines += ["", "Triangles", ""]
    for index, (surface_type, a, b, c) in enumerate(mesh.triangles, 1):
        lines.append(f"{index} {surface_type} {a + 1} {b + 1} {c + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
