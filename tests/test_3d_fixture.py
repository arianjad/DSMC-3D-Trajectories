"""Tiny tracked field reproducibility and provenance check."""
import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_3d_fixture import _field_identity  # noqa: E402

builder = ROOT / "tests/fixtures/3d/build_tube.py"
spec = importlib.util.spec_from_file_location("build_tube", builder)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrackedFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tracked = ROOT / "tests/fixtures/3d/tube"

    def test_regeneration_matches_all_tracked_bytes(self):
        rebuilt = self.root / "rebuilt"
        module.build(rebuilt)
        names = {p.name for p in self.tracked.iterdir() if p.is_file()}
        self.assertEqual(names, {p.name for p in rebuilt.iterdir()})
        self.assertEqual(len(names), 6)
        for name in names:
            self.assertEqual((self.tracked / name).read_bytes(), (rebuilt / name).read_bytes())

    def test_changed_field_artifact_fails_identity(self):
        copied = self.root / "altered"
        shutil.copytree(self.tracked, copied)
        (copied / "cell.surfs").write_text("changed", encoding="utf-8")
        with patch("verify_3d_fixture.FIELD", copied):
            with self.assertRaisesRegex(ValueError, "identity differs: cell.surfs"):
                _field_identity()


if __name__ == "__main__":
    unittest.main()
