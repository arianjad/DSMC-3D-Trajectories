"""Select split-parent IDs from one receipt-checked SPARTA grid frame.

The IDs drive the optional one-rank SPARTA split-map probe. All split parents
are selected, including solid subcells, so missing gas regions remain visible.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_ids(run: Path, frame_name: str, destination: Path, *, allow_empty: bool = False) -> dict:
    run = run.resolve()
    frame = run / frame_name
    receipt_file = run / "run.json"
    if receipt_file.is_file():
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError("helium run receipt is incomplete")
        frame_sha = sha256(frame)
        if frame_sha != receipt.get("frame", {}).get("sha256"):
            raise ValueError("frame differs from completed run receipt")
        receipt_key = "source_run_json_sha256"
    else:
        from dsmc_frames import select_frame
        receipt_file = run / "manifest.json"
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        if receipt.get("status") != "complete":
            raise ValueError("helium run receipt is incomplete")
        selected = select_frame(run, {"timestep": int(frame_name.split(".")[1])})
        if Path(selected["path"]).name != frame_name:
            raise ValueError("selected frame name differs")
        frame_sha = selected["sha256"]
        receipt_key = "source_manifest_sha256"
    opener = gzip.open if frame.suffix == ".gz" else open
    with opener(frame, "rt", encoding="utf-8") as stream:
        header = [next(stream).strip() for _ in range(9)]
        columns = header[8].split()[2:]
        if columns[:2] != ["idstr", "split"]:
            raise ValueError("SPARTA dump must begin with idstr split")
        expected = int(header[3])
        wanted = set()
        count = 0
        for line in stream:
            values = line.split()
            if len(values) != len(columns):
                raise ValueError(f"malformed frame row {count + 1}")
            count += 1
            if int(values[1]) > 0:
                wanted.add(values[0])
    if count != expected:
        raise ValueError(f"frame has {count} rows, header declares {expected}")
    if not wanted and not allow_empty:
        raise ValueError("frame has no split-parent IDs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text("".join(value + "\n" for value in sorted(wanted)),
                           encoding="utf-8", newline="\n")
    report = {
        "schema_version": "sparta-splitmap-ids-v1",
        receipt_key: sha256(receipt_file),
        "source_frame": frame.name,
        "source_frame_sha256": frame_sha,
        "frame_rows": count,
        "split_parent_ids": len(wanted),
        "ids_sha256": sha256(destination),
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("frame_name")
    parser.add_argument("ids_out", type=Path)
    args = parser.parse_args()
    print(json.dumps(select_ids(args.run, args.frame_name, args.ids_out), indent=2))
