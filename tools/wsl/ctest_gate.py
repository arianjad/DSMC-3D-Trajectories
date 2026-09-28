#!/usr/bin/env python3
"""Gate a SPARTA `ctest --output-on-failure` log, variant-aware.

    python3 ctest_gate.py LOG --variant plain|bigbig

plain:  every test passed, and the per-test result lines agree with CTest's
        own summary.
bigbig: the failure set is exactly <machine>.in.circle.constant.mpi_1 and
        .mpi_4, and inside each failure block regression.py compared against
        the 14Oct24 gold with the max norm, every column except Ngrid and
        Maxlevel shows error 0.0, and Maxlevel differs by exactly 1 from a
        gold maximum of 8 (the log prints |test - gold| and max|gold|, so this
        reads "9 vs 8"; 7 vs 8 would print the same line). Every other test
        passed. Anything else fails the gate.

Why the BIGBIG exception exists (roadmap probe P4,
2026-09-25): the golds are blessed on upstream's plain-cellint
runner and upstream CI does not run ctest for bigbig
(.github/workflows/main.yml:15-17 at 27Aug2026). The deck's `fix adapt` sets
no maxlevel, so the depth cap is the cell-ID bit width
(src/adapt_grid.cpp:394-429, src/spatype.h), which BIGBIG doubles (PR
sparta/sparta#610). Gold commit 9ef0941c passes both tests built plain and
fails both built BIGBIG with identical non-grid counters
(local BIGBIG build, 2026-09-23). Column text keys on
tools/testing/regression.py:397-400 ("NAME  error E wrt norm N").
"""
import argparse
import re
import sys

RESULT = re.compile(r"^\s*(\d+)/(\d+) Test\s+#(\d+): (\S+) \.*\s*(\*\*\*)?(\S+)")
START = re.compile(r"^\s*Start\s+\d+: ")
SUMMARY = re.compile(r"^(\d+)% tests passed, (\d+) tests failed out of (\d+)\s*$")
LISTED = re.compile(r"^\s*(\d+) - (\S+) \((.*)\)\s*$")
COLUMN = re.compile(r"^(\S+)\s+error\s+(\S+)(?:\s+wrt norm\s+(\S+))?\s*$")
ZERO_COLUMNS = ("Step", "Np", "Natt", "Ncoll", "Nscoll")
ALL_COLUMNS = set(ZERO_COLUMNS) | {"Ngrid", "Maxlevel"}


def parse(text):
    lines = text.splitlines()
    results, blocks, listed, summary = {}, {}, set(), None
    current = None
    in_listing = False
    for line in lines:
        m = RESULT.match(line)
        if m:
            idx, total, num, name, _, status = m.groups()
            if name in results:
                raise ValueError(f"duplicate result line for {name}")
            results[name] = (int(idx), int(total), status)  # idx is the "k/N" counter
            current = name
            blocks[name] = []
            continue
        if START.match(line):
            current = None
            continue
        m = SUMMARY.match(line)
        if m:
            summary = tuple(int(g) for g in m.groups())
            current = None
            continue
        if line.startswith("The following tests FAILED:"):
            in_listing = True
            continue
        if in_listing:
            m = LISTED.match(line)
            if m:
                listed.add(m.group(2))
                continue
            in_listing = False
        if current is not None:
            blocks[current].append(line)
    return results, blocks, listed, summary


def check_circle_block(name, block):
    """Reasons the failure block is not the documented Maxlevel-cap pattern."""
    why = []
    rank = name.rsplit(".", 1)[1]  # mpi_1 / mpi_4
    compare = f"==== comparing log.{rank}.circle.constant with log.14Oct24.{rank}.circle.constant ===="
    if compare not in block:
        why.append(f"{name}: no comparison against the 14Oct24 gold")
    if "default norm: max" not in block:
        why.append(f"{name}: comparison norm is not max")
    cols = {}
    for line in block:
        m = COLUMN.match(line)
        if m:
            col, err, norm = m.groups()
            if col in cols:
                why.append(f"{name}: column {col} reported twice")
            cols[col] = (float(err), None if norm is None else float(norm))
    if set(cols) != ALL_COLUMNS:
        why.append(f"{name}: columns {sorted(cols)} != {sorted(ALL_COLUMNS)}")
    for col in ZERO_COLUMNS:
        if col in cols and cols[col][0] != 0.0:
            why.append(f"{name}: {col} error {cols[col][0]} (must be 0.0)")
    if "Maxlevel" in cols and cols["Maxlevel"] != (1.0, 8.0):
        why.append(f"{name}: Maxlevel error/norm {cols['Maxlevel']} (must be 1.0 wrt gold max 8.0)")
    bangs = [l for l in block if l.startswith("!!!")]
    allowed = ["!!! test circle.constant FAILED", "!!! 1 of 1 tests failed"]
    if bangs != allowed:
        why.append(f"{name}: regression.py verdict lines {bangs} != {allowed}")
    if any("ERROR" in l or "Invalid logfile" in l or "Cannot compare" in l for l in block):
        why.append(f"{name}: run or log error in failure block")
    return why


def gate(text, variant):
    results, blocks, listed, summary = parse(text)
    why = []
    if summary is None:
        return ["no CTest summary line (incomplete log?)"]
    _, nfail, total = summary
    if len(results) != total:
        why.append(f"{len(results)} result lines but summary says {total} tests")
    if {n for n, _, _ in results.values()} != set(range(1, total + 1)):
        why.append("result counters k/N are not exactly 1..N")
    if any(t != total for _, t, _ in results.values()):
        why.append("a result line's denominator differs from the summary total")
    failed = {name for name, (_, _, status) in results.items() if status != "Passed"}
    if len(failed) != nfail:
        why.append(f"{len(failed)} non-passed result lines but summary says {nfail} failed")
    if listed != failed:
        why.append(f"FAILED listing {sorted(listed)} != non-passed results {sorted(failed)}")

    if variant == "plain":
        if failed:
            why.append(f"plain build must pass every test; failed: {sorted(failed)}")
        return why

    prefixes = {name.split(".in.", 1)[0] for name in results}
    if len(prefixes) != 1:
        return why + [f"test names carry several machine prefixes {sorted(prefixes)}"]
    machine = prefixes.pop()
    allowed = {f"{machine}.in.circle.constant.mpi_1", f"{machine}.in.circle.constant.mpi_4"}
    if failed != allowed:
        why.append(f"failure set {sorted(failed)} != allowed {sorted(allowed)}")
    for name in sorted(allowed & failed):
        if results[name][2] != "Failed":
            why.append(f"{name}: status {results[name][2]} (must be Failed)")
        why += check_circle_block(name, blocks[name])
    return why


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("log")
    ap.add_argument("--variant", required=True, choices=("plain", "bigbig"))
    args = ap.parse_args(argv)
    with open(args.log, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    try:
        why = gate(text, args.variant)
    except ValueError as exc:
        why = [str(exc)]
    if why:
        print(f"ctest gate FAIL ({args.variant}): {args.log}")
        for reason in why:
            print(f"  - {reason}")
        return 1
    results, _, _, summary = parse(text)
    nfail = summary[1]
    print(f"ctest gate PASS ({args.variant}): {summary[2] - nfail}/{summary[2]} passed"
          + (f", {nfail} accepted BIGBIG circle.constant Maxlevel-cap failures" if nfail else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
