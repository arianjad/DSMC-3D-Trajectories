"""Bounded check for nonincident triangle crossings in a CAD surface mesh."""
from __future__ import annotations

import itertools
import math
from collections import defaultdict


def _sub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _segment_face(a, b, face):
    """Return an interior edge/face crossing, or None (Moller-Trumbore)."""
    p, q, r = face
    direction, edge1, edge2 = _sub(b, a), _sub(q, p), _sub(r, p)
    h = _cross(direction, edge2)
    determinant = _dot(edge1, h)
    scale = math.sqrt(_dot(direction, direction) * _dot(edge1, edge1) * _dot(edge2, edge2))
    if abs(determinant) <= 1e-12 * scale:
        return None
    inv = 1.0 / determinant
    offset = _sub(a, p)
    u = _dot(offset, h) * inv
    v = _dot(direction, _cross(offset, edge1)) * inv
    t = _dot(edge2, _cross(offset, edge1)) * inv
    # Strict interior avoids treating the common edge of adjacent triangles as
    # a self-intersection. It still detects the deep sheet crossings seen in production CAD meshes.
    if min(u, v, 1-u-v, t, 1-t) <= 1e-9:
        return None
    return tuple(a[k] + t * direction[k] for k in range(3))


def _crossing(first, second):
    for source, target in ((first, second), (second, first)):
        for i in range(3):
            hit = _segment_face(source[i], source[(i+1) % 3], target)
            if hit is not None:
                return hit
    return None


def find_nonincident_crossing(points, triangles, *, voxel_size_m: float,
                              max_bucket_entries: int = 4_000_000,
                              max_candidate_pairs: int = 10_000_000):
    """Return the first strict crossing and scan counts, failing on excessive work.

    AABB voxel hashing bounds candidate search. Incident faces, coplanar overlap,
    and point/edge tangencies remain the solver run-0 check's responsibility.
    The input vertex IDs are one-based, as in SPARTA surface files.
    """
    if not math.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive and finite")
    buckets = defaultdict(list)
    boxes = []
    entries = pairs = 0
    for i, row in enumerate(triangles):
        face = tuple(points[index-1] for index in row)
        lo = tuple(min(p[axis] for p in face) for axis in range(3))
        hi = tuple(max(p[axis] for p in face) for axis in range(3))
        low = tuple(math.floor(x / voxel_size_m) for x in lo)
        high = tuple(math.floor(x / voxel_size_m) for x in hi)
        nentry = math.prod(high[axis] - low[axis] + 1 for axis in range(3))
        if entries + nentry > max_bucket_entries:
            raise ValueError("CAD crossing scan exceeded voxel-entry limit; use a resolved mesh or larger scan budget")
        cells = itertools.product(*(range(low[axis], high[axis]+1) for axis in range(3)))
        candidates = set()
        keys = []
        for cell in cells:
            keys.append(cell)
            candidates.update(buckets[cell])
        for j in candidates:
            other_row, other_lo, other_hi, other_face = boxes[j]
            if set(row).intersection(other_row):
                continue
            if any(lo[axis] > other_hi[axis] or other_lo[axis] > hi[axis] for axis in range(3)):
                continue
            pairs += 1
            if pairs > max_candidate_pairs:
                raise ValueError("CAD crossing scan exceeded candidate-pair limit; use a resolved mesh or larger scan budget")
            hit = _crossing(face, other_face)
            if hit is not None:
                return {"triangle_ids": [j+1, i+1], "point_m": list(hit),
                        "candidate_pairs_checked": pairs, "voxel_entries": entries}
        boxes.append((row, lo, hi, face))
        for cell in keys:
            buckets[cell].append(i)
        entries += nentry
    return {"triangle_ids": None, "candidate_pairs_checked": pairs,
            "voxel_entries": entries}
