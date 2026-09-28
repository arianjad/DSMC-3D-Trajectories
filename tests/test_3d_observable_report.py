"""Check plane-event reporting on recrossings and malformed event data."""
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dsmc_report import EVENT_COLUMNS, EVENT_SCHEMA, analyze_observable_events  # noqa: E402


PLANE = {
    "name": "readout", "origin_m": [1.0, 0.0, 0.0],
    "normal": [1.0, 0.0, 0.0], "axis_u": [0.0, 1.0, 0.0],
    "axis_v": [0.0, 0.0, 1.0],
    "acceptance": {"position_u_m": [-0.2, 0.2],
                   "position_v_m": None, "angle_rad": [0.0, 0.2],
                   "time_s": [0.0, 0.25]},
}


def event(pid, index, direction, time, vu, vn, u=0.0):
    return dict(zip(EVENT_COLUMNS, (
        EVENT_SCHEMA, pid, index, "readout", direction, time,
        1.0, u, 0.0, vn, vu, 0.0, u, 0.0, vu, 0.0, vn,
    )))


class ObservableReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "events.csv"

    def write(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EVENT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def test_forward_acceptance_uses_polar_angle_and_launched_denominator(self):
        self.write([
            event(1, 1, "forward", 0.1, 1.0, 10.0, u=0.1),
            event(1, 2, "backward", 0.2, 0.0, -4.0),
            event(1, 3, "forward", 0.3, 1.0, 10.0),
            event(2, 1, "forward", 0.1, 2.0, 1.0),
        ])
        result = analyze_observable_events(self.path, [PLANE], 3)
        plane = result["planes"][0]
        self.assertEqual(result["event_count"], 4)
        self.assertEqual(plane["forward_event_count"], 3)
        self.assertEqual(plane["forward_unique_particles"], 2)
        self.assertEqual(plane["backward_event_count"], 1)
        self.assertEqual(plane["first_forward"]["count"], 2)
        self.assertAlmostEqual(plane["first_forward"]["mean_vu_m_s"], 1.5)
        self.assertAlmostEqual(plane["first_forward"]["mean_polar_angle_rad"],
                               (math.atan2(1, 10) + math.atan2(2, 1)) / 2)
        self.assertEqual(plane["acceptance"]["event_count"], 1)
        self.assertEqual(plane["acceptance"]["unique_particles"], 1)
        self.assertAlmostEqual(plane["acceptance"]["fraction_of_launched"], 1 / 3)

    def test_empty_exploratory_plane_is_unassessed(self):
        self.write([])
        plane = {**PLANE, "acceptance": {key: None for key in PLANE["acceptance"]}}
        result = analyze_observable_events(self.path, [plane], 5)
        summary = result["planes"][0]
        self.assertEqual(summary["event_count"], 0)
        self.assertIsNone(summary["first_forward"]["mean_arrival_time_s"])
        self.assertEqual(summary["acceptance"]["status"], "unassessed")
        self.assertIsNone(summary["acceptance"]["fraction_of_launched"])

    def test_inconsistent_plane_projection_rejected(self):
        row = event(1, 1, "forward", 0.1, 0.0, 1.0)
        row["u_m"] = 0.1
        self.write([row])
        with self.assertRaisesRegex(ValueError, "Cartesian and plane-local"):
            analyze_observable_events(self.path, [PLANE], 1)

    def test_crossing_order_and_launched_ids_rejected(self):
        self.write([event(1, 2, "forward", 0.1, 0.0, 1.0)])
        with self.assertRaisesRegex(ValueError, "not sequential"):
            analyze_observable_events(self.path, [PLANE], 1)
        self.write([event(2, 1, "forward", 0.1, 0.0, 1.0)])
        with self.assertRaisesRegex(ValueError, "exceeds launched"):
            analyze_observable_events(self.path, [PLANE], 1)


if __name__ == "__main__":
    unittest.main()
