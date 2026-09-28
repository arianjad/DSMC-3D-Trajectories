"""Generate deterministic interior point probes for SPARTA split3d parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

if __package__:
    from .validate_map import load_map, sha256
else:
    from validate_map import load_map, sha256


def clipped_facet_center(facet: tuple, bounds: tuple) -> list[float] | None:
    """Clip a triangle to the parent box before choosing its interior centroid."""
    polygon = [list(facet[start:start + 3]) for start in (0, 3, 6)]
    for axis in range(3):
        for bound, direction in ((bounds[axis], 1), (bounds[axis + 3], -1)):
            if not polygon:
                return None
            clipped = []
            previous = polygon[-1]
            previous_inside = direction * (previous[axis] - bound) >= 0
            for point in polygon:
                inside = direction * (point[axis] - bound) >= 0
                if inside != previous_inside:
                    fraction = (bound - previous[axis]) / (point[axis] - previous[axis])
                    clipped.append([previous[j] + fraction * (point[j] - previous[j]) for j in range(3)])
                if inside:
                    clipped.append(point)
                previous, previous_inside = point, inside
            polygon = clipped
    if len(polygon) < 3:
        return None
    return [sum(point[j] for point in polygon) / len(polygon) for j in range(3)]


def generate(map_path: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    parents = load_map(map_path)
    rng = random.Random(20260924)
    queries = []
    for idstr, parent in sorted(parents.items()):
        bounds = parent["bounds"]
        width = [bounds[j + 3] - bounds[j] for j in range(3)]
        queries.append((idstr, *parent["xsplit"]))
        for _ in range(8):
            point = [bounds[j] + (0.05 + 0.9 * rng.random()) * width[j]
                     for j in range(3)]
            queries.append((idstr, *point))
        # Uniform probes can miss very thin gas subcells. Probe every facet's
        # clipped portion at several offsets, leaving classification to SPARTA.
        for _, _, facet in parent["surfaces"]:
            center = clipped_facet_center(facet, bounds)
            if center is None:
                continue
            normal = facet[9:12]
            for scale in (1e-2, 1e-4, 1e-6):
                for sign in (-1, 1):
                    point = [center[j] + sign * min(width) * scale * normal[j] for j in range(3)]
                    if all(bounds[j] < point[j] < bounds[j + 3] for j in range(3)):
                        queries.append((idstr, *point))
    destination.write_text("".join(
        f"{idstr} {x:.17g} {y:.17g} {z:.17g}\n"
        for idstr, x, y, z in queries), encoding="utf-8", newline="\n")
    report = {
        "schema_version": "sparta-splitmap-queries-v1",
        "map_sha256": sha256(map_path),
        "queries_sha256": sha256(destination),
        "parent_count": len(parents),
        "query_count": len(queries),
        "sampling": "xsplit, eight deterministic interior points, every clipped facet both sides at 1e-2/1e-4/1e-6 cell width",
        "rng_seed": 20260924,
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splitmap", type=Path)
    parser.add_argument("queries", type=Path)
    args = parser.parse_args()
    print(json.dumps(generate(args.splitmap, args.queries), indent=2))
