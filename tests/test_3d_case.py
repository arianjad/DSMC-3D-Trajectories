"""Task 3 contract and actual circle/slit preparation checks."""
import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dsmc_case import inspect_geometry, load_case, load_profile, prepare_case  # noqa: E402


class CaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.circle = ROOT / "cases/examples/b5-circle/case.json"
        self.slit = ROOT / "cases/examples/b5-slit/case.json"

    def _case_copy(self, source, edit):
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        for section, key in (("helium", "species_file"), ("helium", "collision_file"),
                             ("numerics", "deck_template")):
            data[section][key] = str((Path(source).parent / data[section][key]).resolve())
        edit(data)
        path = self.root / f"case-{len(list(self.root.glob('case-*.json')))}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def _prepare(self, source, name):
        with contextlib.redirect_stdout(io.StringIO()):
            return prepare_case(load_case(source), self.root / name)

    @unittest.skip("needs cases/examples/b5-slit, which is not in this release")
    def test_circle_slit_preparation_matches_sampled_contract(self):
        circle = self._prepare(self.circle, "circle")
        slit = self._prepare(self.slit, "slit")
        a = inspect_geometry(circle)
        b = inspect_geometry(slit)
        for result, prepared in ((a, circle), (b, slit)):
            self.assertEqual(result["state"], "prepared-unclipped")
            self.assertEqual(result["solver_check"], "not-run")
            self.assertEqual(result["geometry"]["physical_boundary"]["triangle_types"], {"1": 576, "2": 48})
            self.assertTrue(result["geometry"]["physical_boundary"]["emitter_cap_normal_positive_x"])
            self.assertAlmostEqual(result["geometry"]["physical_boundary"]["aperture_area_m2"],
                                   1.9578928833007738e-5, delta=1e-13)
            deck = (prepared.parent / "in.helium").read_text()
            self.assertIn("dump        dfields grid all 2400 fields.*.gz", deck)
            self.assertIn("variable    nmain equal ${STEPS}-200", deck)
        circle_prep = json.loads(circle.read_text())
        slit_prep = json.loads(slit.read_text())
        self.assertEqual(circle_prep["science_inputs"]["numerics"]["total_steps"], 4800)
        self.assertEqual(slit_prep["science_inputs"]["numerics"]["solver_seed"], 20260919)
        self.assertEqual(circle_prep["science_inputs"]["molecules"]["seed"], "20260922")
        self.assertEqual(circle_prep["science_inputs"]["geometry"]["transform"]["solver_to_tracer_axes"], ["y", "z", "x"])
        self.assertNotEqual(a["artifacts"]["physical_boundary"], b["artifacts"]["physical_boundary"])
        self.assertEqual(a["artifacts"]["ap_fwd"], b["artifacts"]["ap_fwd"])
        self.assertEqual((circle.parent / "in.helium").read_text().replace("cell_b5_3d_s25.surf", "<cell>"),
                         (slit.parent / "in.helium").read_text().replace("cell_b5_3d_slit_s25.surf", "<cell>"))

    @unittest.skip("needs cases/examples/b5-slit, which is not in this release")
    def test_aspect_changes_mesh_station_and_regions(self):
        source = self._case_copy(self.slit, lambda d: d["geometry"].update(aspect_vertical_over_horizontal=4))
        changed = self._prepare(source, "changed")
        baseline = self._prepare(self.slit, "baseline")
        a, b = inspect_geometry(changed), inspect_geometry(baseline)
        self.assertNotEqual(a["artifacts"]["physical_boundary"], b["artifacts"]["physical_boundary"])
        self.assertNotEqual(a["artifacts"]["ap_fwd"], b["artifacts"]["ap_fwd"])
        self.assertNotEqual(a["regions"]["throat_l4_radius_m"], b["regions"]["throat_l4_radius_m"])
        self.assertAlmostEqual(a["geometry"]["physical_boundary"]["aperture_area_m2"],
                               b["geometry"]["physical_boundary"]["aperture_area_m2"], delta=1e-13)

    def test_profile_is_separate_and_not_embedded_in_preparation(self):
        first = self._prepare(self.circle, "one")
        default = load_profile(ROOT / "profiles/local.example.json")
        other_data = json.loads((ROOT / "profiles/local.example.json").read_text())
        other_data.update(profile_id="different-machine", ranks=3, threads=2,
                          solver_executable="different-sparta")
        other_path = self.root / "other-profile.json"
        other_path.write_text(json.dumps(other_data))
        other = load_profile(other_path)
        self.assertNotEqual(default["profile_id"], other["profile_id"])
        second = self._prepare(self.circle, "two")
        a, b = json.loads(first.read_text()), json.loads(second.read_text())
        self.assertEqual(a["science_inputs"], b["science_inputs"])
        self.assertNotIn("profile_id", first.read_text())
        other_data["dt_s"] = 2e-7
        other_path.write_text(json.dumps(other_data))
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            load_profile(other_path)

    def test_invalid_case_fields_fail_before_output(self):
        changes = [
            (lambda d: d.update(extra_executable="sh"), "unknown keys"),
            (lambda d: d["geometry"].update(length_unit="mm"), "length_unit"),
            (lambda d: d["geometry"]["transform"].update(solver_to_tracer_axes=["x", "y", "z"]), "transform"),
            (lambda d: d["geometry"]["boundaries"].pop("inlet_cap"), "inlet_cap"),
            (lambda d: d["molecules"].update(seed=18446744073709551615), "UInt64 string"),
            (lambda d: d["numerics"].update(total_steps=4799), "field step"),
        ]
        for index, (change, message) in enumerate(changes):
            with self.subTest(index=index):
                path = self._case_copy(self.circle, change)
                with self.assertRaisesRegex(ValueError, message):
                    prepare_case(load_case(path), self.root / f"bad-{index}")
                self.assertFalse((self.root / f"bad-{index}").exists())

    def test_full_uint64_molecule_seed_and_integrity(self):
        path = self._case_copy(self.circle, lambda d: d["molecules"].update(seed="18446744073709551615"))
        prepared = self._prepare(path, "maxseed")
        self.assertEqual(json.loads(prepared.read_text())["science_inputs"]["molecules"]["seed"],
                         "18446744073709551615")
        with self.assertRaises(FileExistsError):
            self._prepare(path, "maxseed")
        (prepared.parent / "in.helium").write_text("tamper")
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            inspect_geometry(prepared)

    def test_point_source_zero_radius_and_zero_temperature_match_runtime(self):
        def point(data):
            data["molecules"]["source"].update(
                distribution="point", radius_m=0, temperature_K=0)
            data["molecules"]["seed"] = "1"
        source = self._case_copy(self.circle, point)
        prepared = self._prepare(source, "point")
        actual = json.loads(prepared.read_text(encoding="utf-8"))["science_inputs"]["molecules"]
        self.assertEqual(actual["source"]["radius_m"], 0)
        self.assertEqual(actual["source"]["temperature_K"], 0)
        invalid = [
            (lambda d: d["molecules"]["source"].update(distribution="point", radius_m=0.01),
             "point requires zero"),
            (lambda d: d["molecules"]["source"].update(radius_m=0),
             "extended source requires positive"),
            (lambda d: d["molecules"]["source"].update(temperature_K=-1),
             "nonnegative"),
            (lambda d: d["molecules"].update(seed="0"), "nonzero decimal UInt64"),
        ]
        for change, message in invalid:
            with self.subTest(message=message):
                bad = self._case_copy(self.circle, change)
                with self.assertRaisesRegex(ValueError, message):
                    load_case(bad)

    def test_failed_export_leaves_no_named_output(self):
        import dsmc_geometry
        case = load_case(self.circle)
        real = dsmc_geometry.prepare_b5

        def broken_export(candidate, destination):
            if destination is None:
                return real(candidate, None)
            (destination / "partial.surf").write_text("incomplete")
            raise ValueError("synthetic export failure")

        with mock.patch.object(dsmc_geometry, "prepare_b5", side_effect=broken_export):
            with self.assertRaisesRegex(ValueError, "synthetic export failure"):
                prepare_case(case, self.root / "failed")
        self.assertFalse((self.root / "failed").exists())

    def test_cli_inspect_and_solver_check_boundary(self):
        prepared = self._prepare(self.circle, "cli")
        command = [sys.executable, str(ROOT / "tools/dsmc.py"), "inspect", str(prepared.parent)]
        ok = subprocess.run(command + ["--json", str(self.root / "inspect.json")], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertEqual(json.loads((self.root / "inspect.json").read_text())["solver_check"], "not-run")
        unsupported = subprocess.run(command + ["--solver-check"], capture_output=True, text=True)
        self.assertNotEqual(unsupported.returncode, 0)
        self.assertIn("requires --profile", unsupported.stderr)


if __name__ == "__main__":
    unittest.main()
