"""Committed source exports retain identity across relocation and dirty checkouts."""
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from dsmc_repository import export_repository, repository_identity, verify_repository, repository_provenance


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        self.root.mkdir()
        (self.root / "tools").mkdir()
        (self.root / "tools/example.py").write_bytes(b"print('committed')\n")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                 "commit", "-qm", "fixture")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args], stderr=subprocess.STDOUT)

    def exported(self):
        archive = Path(self.temp.name) / "export.tar.gz"
        export_repository(self.root, archive)
        destination = Path(self.temp.name) / "moved"
        with tarfile.open(archive) as stream:
            stream.extractall(destination, filter="data")
        return destination

    def test_export_uses_commit_not_dirty_worktree(self):
        (self.root / "tools/example.py").write_text("dirty\n")
        destination = self.exported()
        identity = repository_identity(destination)
        self.assertEqual(identity["commit"], self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual((destination / "tools/example.py").read_bytes(), b"print('committed')\n")
        self.assertEqual(verify_repository(destination, identity), identity)
        self.assertEqual(repository_provenance(destination), {**identity, "dirty": False})

    def test_tampered_payload_and_new_code_are_rejected(self):
        destination = self.exported()
        identity = repository_identity(destination)
        extra = destination / "tools/injected.py"
        extra.write_text("print('untracked')")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            verify_repository(destination, identity)
        extra.unlink()
        (destination / "tools/example.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            verify_repository(destination, identity)

    def test_manifest_change_rejected_even_when_commit_unchanged(self):
        destination = self.exported()
        identity = repository_identity(destination)
        manifest = destination / "source-export.json"
        data = json.loads(manifest.read_text())
        data["files"] = []
        manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            verify_repository(destination, identity)

    def test_git_verifier_rejects_dirty_tracked_source(self):
        identity = repository_identity(self.root)
        verify_repository(self.root, identity)
        (self.root / "tools/example.py").write_text("changed")
        with self.assertRaises(subprocess.CalledProcessError):
            verify_repository(self.root, identity)


if __name__ == "__main__":
    unittest.main()
