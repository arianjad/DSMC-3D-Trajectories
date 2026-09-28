"""Bind a solver-exported split map to one completed SPARTA grid frame."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_map(path: Path) -> dict:
    parents = {}
    with path.open(encoding="utf-8") as stream:
        if next(stream).strip() != "SPARTA_SPLITMAP_V1":
            raise ValueError("unsupported split map schema")
        end = None
        for line in stream:
            row = line.strip().split("\t")
            if row[0] == "P":
                if len(row) != 13 or row[1] in parents:
                    raise ValueError("invalid or duplicate split parent")
                parents[row[1]] = {
                    "nsplit": int(row[2]), "xsub": int(row[3]),
                    "xsplit": tuple(map(float, row[4:7])),
                    "bounds": tuple(map(float, row[7:13])),
                    "volumes": {}, "surfaces": [],
                }
            elif row[0] == "C":
                if len(row) != 4 or row[1] not in parents:
                    raise ValueError("orphan split subcell")
                volumes = parents[row[1]]["volumes"]
                index = int(row[2])
                if index in volumes:
                    raise ValueError("duplicate split subcell index")
                volumes[index] = float(row[3])
            elif row[0] == "S":
                if len(row) != 16 or row[1] not in parents:
                    raise ValueError("orphan mapped surface")
                parents[row[1]]["surfaces"].append((int(row[2]), int(row[3]),
                                                     tuple(map(float, row[4:]))))
            elif row[0] == "END":
                if len(row) != 3 or end is not None:
                    raise ValueError("invalid split map terminator")
                end = (int(row[1]), int(row[2]))
            else:
                raise ValueError(f"unexpected split map record: {row[0]}")
    if end != (len(parents), len(parents)):
        raise ValueError("split map export is incomplete")
    for idstr, parent in parents.items():
        nsplit = parent["nsplit"]
        if nsplit < 2 or not 0 <= parent["xsub"] < nsplit:
            raise ValueError(f"invalid parent split count: {idstr}")
        if set(parent["volumes"]) != set(range(nsplit)):
            raise ValueError(f"incomplete subcell volumes: {idstr}")
        if not parent["surfaces"] or any(
                not 0 <= index < nsplit for _, index, _ in parent["surfaces"]):
            raise ValueError(f"invalid mapped surfaces: {idstr}")
        if not all(math.isfinite(x) for x in (*parent["xsplit"], *parent["bounds"],
                                              *parent["volumes"].values())):
            raise ValueError(f"nonfinite split map value: {idstr}")
    return parents


def validate(run: Path, frame_name: str, map_path: Path, ids_path: Path) -> dict:
    receipt_path = run / "run.json"
    frame = run / frame_name
    # A standalone conversion can inspect the original run.json. The generic
    # field stage passes a hash-checked staged copy with no run.json; the IDs
    # receipt still binds that copy to its completed source frame by SHA-256.
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete" or sha256(frame) != receipt["frame"]["sha256"]:
            raise ValueError("source frame differs from completed helium receipt")
    ids_receipt = json.loads(ids_path.with_suffix(ids_path.suffix + ".json").read_text())
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = run / "parent-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_manifest = ids_receipt.get("source_manifest_sha256")
        imported_source = manifest.get("source_pilot", {}).get("receipt_sha256")
        if recorded_manifest is not None:
            if recorded_manifest != sha256(manifest_path):
                raise ValueError("selected parent IDs differ from helium manifest")
        elif not (manifest.get("schema_version") == "dsmc-cad-helium-import-v1" and
                  imported_source is not None and
                  ids_receipt.get("source_run_json_sha256") == imported_source):
            raise ValueError("selected parent IDs are not bound to helium manifest")
        outputs = [item for item in manifest.get("outputs", []) if item.get("path") == frame_name]
        if len(outputs) != 1 or outputs[0].get("sha256") != sha256(frame):
            raise ValueError("split frame differs from helium manifest")
    elif "source_manifest_sha256" in ids_receipt:
        raise ValueError("manifest-bound parent IDs require the source manifest")
    if ((receipt_path.is_file() and
            ids_receipt["source_run_json_sha256"] != sha256(receipt_path)) or
            ids_receipt["source_frame_sha256"] != sha256(frame) or
            ids_receipt["ids_sha256"] != sha256(ids_path)):
        raise ValueError("selected parent IDs are not bound to this frame")
    parents = load_map(map_path)
    wanted = set(ids_path.read_text(encoding="utf-8").splitlines())
    if set(parents) != wanted:
        raise ValueError(f"map covers {len(parents)} parents; frame selects {len(wanted)}")
    seen = set()
    max_bound_error = 0.0
    max_volume_relative_error = 0.0
    rows = 0
    mapped_rows = 0
    with gzip.open(frame, "rt", encoding="utf-8") as stream:
        header = [next(stream).strip() for _ in range(9)]
        box = [tuple(map(float, line.split())) for line in header[5:8]]
        names = header[8].split()[2:]
        index = {name: names.index(name) for name in
                 ("idstr", "split", "xlo", "ylo", "zlo", "xhi", "yhi", "zhi", "vol")}
        for line in stream:
            values = line.split()
            rows += 1
            idstr = values[index["idstr"]]
            if idstr not in wanted:
                continue
            parent = parents[idstr]
            split = int(values[index["split"]])
            if not 1 <= split <= parent["nsplit"] or (idstr, split) in seen:
                raise ValueError(f"invalid or duplicate frame subcell {idstr}/{split}")
            seen.add((idstr, split))
            mapped_rows += 1
            bounds = [float(values[index[key]]) for key in
                      ("xlo", "ylo", "zlo", "xhi", "yhi", "zhi")]
            width = [parent["bounds"][j + 3] - parent["bounds"][j] for j in range(3)]
            for j, (actual, expected) in enumerate(zip(bounds, parent["bounds"])):
                error = abs(actual - expected)
                max_bound_error = max(max_bound_error, error)
                axis = j % 3
                scale = max(abs(actual), abs(expected), *(abs(x) for x in box[axis]),
                            width[axis])
                # SPARTA's text dump uses six significant digits; apply the
                # converter's bounded coordinate representation tolerance.
                tolerance = min(5e-6 * scale, 5e-3 * width[axis])
                tolerance += 64.0 * math.ulp(1.0) * max(scale, 1.0)
                if error > tolerance:
                    raise ValueError(f"frame/map bounds differ for {idstr}/{split}: "
                                     f"{error:.9g} m exceeds {tolerance:.9g} m")
            actual_volume = float(values[index["vol"]])
            expected_volume = parent["volumes"][split - 1]
            relative = abs(actual_volume - expected_volume) / max(expected_volume, 1e-300)
            max_volume_relative_error = max(max_volume_relative_error, relative)
            if relative > 5e-5:
                raise ValueError(f"frame/map volume differs for {idstr}/{split}")
    if rows != int(header[3]) or mapped_rows != sum(p["nsplit"] for p in parents.values()):
        raise ValueError("frame row count or mapped subcell coverage differs")
    report = {
        "schema_version": "sparta-splitmap-frame-validation-v1",
        "frame_sha256": sha256(frame),
        "splitmap_sha256": sha256(map_path),
        "ids_sha256": sha256(ids_path),
        "split_parents": len(parents),
        "mapped_frame_rows": mapped_rows,
        "mapped_surfaces": sum(len(p["surfaces"]) for p in parents.values()),
        "max_bound_error_m": max_bound_error,
        "max_volume_relative_error": max_volume_relative_error,
    }
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("frame_name")
    parser.add_argument("splitmap", type=Path)
    parser.add_argument("ids", type=Path)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    result = validate(args.run, args.frame_name, args.splitmap, args.ids)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
