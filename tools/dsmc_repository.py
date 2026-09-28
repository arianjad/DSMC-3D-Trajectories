"""Identify and verify a Git checkout or an immutable committed-tree export."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile

MANIFEST = "source-export.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_identity(root: Path) -> dict:
    root = Path(root).resolve()
    manifest = root / MANIFEST
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("schema_version") != "dsmc-source-export-v1":
            raise ValueError("unsupported repository export manifest")
        return {"kind": "export", "commit": data["commit"],
                "manifest_sha256": sha(manifest)}
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    return {"kind": "git", "commit": commit}


def verify_repository(root: Path, expected: dict) -> dict:
    root = Path(root).resolve()
    actual = repository_identity(root)
    if actual["commit"] != expected["commit"]:
        raise ValueError("repository commit differs from expected identity")
    if expected.get("manifest_sha256") and actual.get("manifest_sha256") != expected["manifest_sha256"]:
        raise ValueError("repository export manifest changed")
    if actual["kind"] == "git":
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=True)
        extra = subprocess.check_output(["git", "-C", str(root), "ls-files", "--others",
                                         "--exclude-standard", "--", "tools", "tracer"], text=True)
        if extra.strip():
            raise ValueError("untracked implementation files in checkout")
    else:
        data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        listed = set()
        for item in data["files"]:
            relative = PurePosixPath(item["path"])
            if relative.is_absolute() or ".." in relative.parts or str(relative) in listed:
                raise ValueError("invalid or duplicate export path")
            listed.add(str(relative))
            path = root / str(relative)
            if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
                raise ValueError("export path contains a symlink")
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                raise ValueError(f"exported source missing or changed: {relative}")
        for folder in ("tools", "tracer"):
            for path in (root / folder).rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts and path.relative_to(root).as_posix() not in listed:
                    raise ValueError(f"unlisted implementation file: {path.relative_to(root)}")
    return actual


def repository_provenance(root: Path) -> dict:
    """Record a checked export or the state of a development checkout."""
    root = Path(root).resolve()
    try:
        identity = repository_identity(root)
        if identity["kind"] == "export":
            verify_repository(root, identity)
            return {**identity, "dirty": False}
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
        return {**identity, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def export_repository(root: Path, out: Path) -> dict:
    """Export committed bytes, including on CRLF Windows checkouts."""
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        raw = Path(temp) / "source.tar"
        subprocess.run(["git", "-C", str(root), "-c", "core.autocrlf=false", "-c", "core.eol=lf",
                        "archive", "--format=tar", "--output", str(raw), commit], check=True)
        entries = []
        with tarfile.open(raw) as source, tarfile.open(out, "x:gz") as target:
            for member in source:
                if member.issym() or member.islnk():
                    raise ValueError(f"source export does not support links: {member.name}")
                content = source.extractfile(member) if member.isfile() else None
                if content is not None:
                    payload = content.read()
                    entries.append({"path": member.name, "bytes": len(payload),
                                    "sha256": hashlib.sha256(payload).hexdigest()})
                    target.addfile(member, io.BytesIO(payload))
                else:
                    target.addfile(member)
            record = {"schema_version": "dsmc-source-export-v1", "commit": commit, "files": entries}
            payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
            member = tarfile.TarInfo(MANIFEST)
            member.size = len(payload)
            member.mode = 0o644
            target.addfile(member, io.BytesIO(payload))
    return {"archive": str(out), "sha256": sha(out), "commit": commit, "files": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    export = commands.add_parser("export")
    export.add_argument("--root", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--commit", required=True)
    verify.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.action == "export":
        result = export_repository(args.root, args.out)
    else:
        result = verify_repository(args.root, {"commit": args.commit, "manifest_sha256": args.manifest_sha256})
    print(json.dumps(result, indent=2))
