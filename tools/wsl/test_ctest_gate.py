#!/usr/bin/env python3
"""Self-check for ctest_gate.py: a synthetic BIGBIG log passes, and every
mutation the gate exists to catch fails.

    python3 tools/wsl/test_ctest_gate.py

Fixture: fixtures/ctest-synthetic-bigbig.log is a short synthetic
`ctest --output-on-failure` log (8 tests) in the format CTest and SPARTA's
tools/testing/regression.py print. Its two failure blocks have the pattern the
gate accepts for a 64-bit-cell-ID build: circle.constant at 1 and 4 ranks,
compared against the 14Oct24 gold with the max norm, zero error in every
counter except Ngrid and Maxlevel, and Maxlevel 9 against a gold maximum of 8.
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / "ctest_gate.py"
FIXTURE = HERE / "fixtures" / "ctest-synthetic-bigbig.log"
LOG = FIXTURE.read_text(encoding="utf-8")


def run_gate(text, variant):
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "ctest.log"
        log.write_text(text, encoding="utf-8")
        r = subprocess.run([sys.executable, str(GATE), str(log), "--variant", variant],
                           capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


def sub(text, old, new, count=1):
    """Replace exactly `count` occurrences, or the mutation silently did nothing."""
    assert text.count(old) == count, (old, text.count(old))
    return text.replace(old, new)


def block_edit(text, test_no, old, new):
    """Replace `old` only inside test #test_no's failure block."""
    head, tail = text.split(f"Test #{test_no}: ", 1)
    block, rest = tail.split("        Start", 1)
    return head + f"Test #{test_no}: " + sub(block, old, new) + "        Start" + rest


PASSED_4 = "#4: mpi_bigbig.in.sphere.adjust.mpi_1 ...........................   Passed"
FAILED_4 = "#4: mpi_bigbig.in.sphere.adjust.mpi_1 ...........................***Failed"
FAILED_6 = "#6: mpi_bigbig.in.circle.constant.mpi_4 .........................***Failed"
SUMMARY = "75% tests passed, 2 tests failed out of 8"


def third_failure(text):
    text = sub(text, PASSED_4, FAILED_4)
    text = sub(text, SUMMARY, "62% tests passed, 3 tests failed out of 8")
    return sub(text, "\t6 - mpi_bigbig.in.circle.constant.mpi_4 (Failed)\n",
               "\t6 - mpi_bigbig.in.circle.constant.mpi_4 (Failed)\n"
               "\t4 - mpi_bigbig.in.sphere.adjust.mpi_1 (Failed)\n")


def all_pass(text):
    """A plain-style log: the two failures turned into passes, blocks dropped."""
    out = []
    skip = False
    for line in text.splitlines(keepends=True):
        if "circle.constant.mpi_" in line and "***Failed" in line:
            out.append(line.split("......")[0].rstrip() + " .....   Passed    1.00 sec\n")
            skip = True
            continue
        if skip and line.startswith("        Start"):
            skip = False
        if skip:
            continue
        if line.startswith("The following tests FAILED:"):
            break
        out.append(line)
    return sub("".join(out), SUMMARY, "100% tests passed, 0 tests failed out of 8")


MUTATIONS = {
    "third failing test": third_failure,
    "third failure, summary not updated": lambda t: sub(t, PASSED_4, FAILED_4),
    "nonzero Np in 3": lambda t: block_edit(t, 3, "Np       error  0.0", "Np       error  3.0"),
    "nonzero Ncoll in 6": lambda t: block_edit(t, 6, "Ncoll    error  0.0", "Ncoll    error  1.0"),
    "nonzero Step in 6": lambda t: block_edit(t, 6, "Step     error  0.0", "Step     error 250.0"),
    "Maxlevel 10 vs 8 in 3": lambda t: block_edit(t, 3, "Maxlevel  error  1.0", "Maxlevel  error  2.0"),
    "Maxlevel gold 9 in 6": lambda t: block_edit(t, 6, "wrt norm     8.0", "wrt norm     9.0"),
    "Maxlevel equal in 3": lambda t: block_edit(t, 3, "Maxlevel  error  1.0 wrt norm     8.0",
                                                 "Maxlevel  error  0.0 wrt norm     8.0"),
    "Maxlevel column missing in 6": lambda t: block_edit(t, 6, "Maxlevel  error  1.0 wrt norm     8.0\n", ""),
    "extra column in 3": lambda t: block_edit(t, 3, "Ngrid ", "Nfoo     error  0.0 wrt norm 1.0\nNgrid "),
    "3 compared to another gold": lambda t: block_edit(t, 3, "log.14Oct24.mpi_1.circle.constant ====",
                                                       "log.01Jan26.mpi_1.circle.constant ===="),
    "3 run error instead of compare": lambda t: block_edit(
        t, 3, "==== comparing log.mpi_1.circle.constant with log.14Oct24.mpi_1.circle.constant ====",
        "!!! no log.mpi_1.circle.constant"),
    "6 ERROR in output": lambda t: block_edit(t, 6, "read 9 log entries\n==",
                                              "read 9 log entries\nERROR: something\n=="),
    "6 L1 norm": lambda t: block_edit(t, 6, "default norm: max", "default norm: L1"),
    "6 timeout not compare failure": lambda t: sub(t, FAILED_6, FAILED_6.replace("***Failed", "***Timeout")),
    "only mpi_1 fails": lambda t: sub(sub(sub(
        t, FAILED_6, FAILED_6.replace("***Failed", "   Passed")),
        SUMMARY, "87% tests passed, 1 tests failed out of 8"),
        "\t6 - mpi_bigbig.in.circle.constant.mpi_4 (Failed)\n", ""),
    "a test missing from the log": lambda t: sub(
        t, "        Start 8: mpi_bigbig.in.torque.mpi_4\n"
           "8/8 Test #8: mpi_bigbig.in.torque.mpi_4 ..................................   Passed    1.00 sec\n", ""),
    "truncated before summary": lambda t: t.split("\n75% tests passed")[0],
}


class GateTests(unittest.TestCase):
    def test_synthetic_bigbig_log_passes(self):
        rc, out = run_gate(LOG, "bigbig")
        self.assertEqual(rc, 0, out)
        self.assertIn("PASS", out)

    def test_each_mutation_fails(self):
        for label, mutate in MUTATIONS.items():
            with self.subTest(label):
                mutated = mutate(LOG)
                self.assertNotEqual(mutated, LOG)
                rc, out = run_gate(mutated, "bigbig")
                self.assertNotEqual(rc, 0, out)
                self.assertIn("FAIL", out)

    def test_plain_variant(self):
        rc, out = run_gate(LOG, "plain")  # any failure is fatal for plain
        self.assertNotEqual(rc, 0, out)
        rc, out = run_gate(all_pass(LOG), "plain")  # positive control
        self.assertEqual(rc, 0, out)
        rc, out = run_gate(all_pass(LOG), "bigbig")  # bigbig wants exactly the two
        self.assertNotEqual(rc, 0, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
