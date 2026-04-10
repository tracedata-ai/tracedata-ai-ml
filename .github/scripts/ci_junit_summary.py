#!/usr/bin/env python3
"""Emit a GitHub Actions step-summary markdown table from one or more JUnit XML files.

Usage:
  ci_junit_summary.py [label:path ...]
  ci_junit_summary.py path1.xml path2.xml   # labels default to basename

Paths may be missing (skipped silently) so callers can use ``|| true``.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _testsuites_from_root(root: ET.Element) -> list[ET.Element]:
    if root.tag == "testsuites":
        return [c for c in root if c.tag == "testsuite"]
    if root.tag == "testsuite":
        return [root]
    return root.findall(".//testsuite")


def _aggregate_file(path: Path) -> tuple[int, int, int, int, int]:
    tree = ET.parse(path)
    root = tree.getroot()
    suites = _testsuites_from_root(root)
    total = failures = errors = skipped = 0
    for suite in suites:
        total += int(suite.attrib.get("tests", 0) or 0)
        failures += int(suite.attrib.get("failures", 0) or 0)
        errors += int(suite.attrib.get("errors", 0) or 0)
        skipped += int(suite.attrib.get("skipped", 0) or 0)
    passed = total - failures - errors - skipped
    return total, passed, failures, errors, skipped


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("_No JUnit files provided._", file=sys.stderr)
        return 0

    pairs: list[tuple[str, Path]] = []
    for raw in args:
        if ":" in raw:
            label, p = raw.split(":", 1)
            pairs.append((label.strip(), Path(p.strip())))
        else:
            p = Path(raw)
            pairs.append((p.name, p))

    rows: list[tuple[str, int, int, int, int, int]] = []
    for label, path in pairs:
        if not path.is_file():
            continue
        t, ok, f, e, s = _aggregate_file(path)
        rows.append((label, t, ok, f, e, s))

    if not rows:
        print("_No JUnit XML files found on disk (tests may have failed before report was written)._")
        return 0

    gt = gp = gf = ge = gs = 0
    for _label, t, ok, f, e, s in rows:
        gt += t
        gp += ok
        gf += f
        ge += e
        gs += s

    # Headline aggregate: maximal visibility, definitionally tied to JUnit (not invented).
    print("## Test volume (JUnit, auditable)")
    print("")
    print(
        f"**{gt}** reported test cases in this summary — **sum of JUnit `tests=` counters** "
        "from the XML below (parameterized cases count as multiple rows when the runner reports them that way)."
    )
    print("")
    if len(rows) > 1:
        print(
            "_Multiple reports are **added**: each file reflects a separate test command in CI, "
            "so the sum is “test executions recorded,” not deduplicated across files._"
        )
    else:
        print(
            "_Single report: this number matches the runner’s own JUnit total for that command._"
        )
    print("")
    print("### Breakdown by report")
    print("")
    print("| Suite | Total | Passed | Failed | Errors | Skipped |")
    print("|-------|------:|-------:|-------:|-------:|--------:|")
    for label, t, ok, f, e, s in rows:
        print(f"| {label} | {t} | {ok} | {f} | {e} | {s} |")
    if len(rows) > 1:
        print(f"| **Sum (all rows)** | {gt} | {gp} | {gf} | {ge} | {gs} |")
        print("")
        print(
            "_If the same logical tests appear in two uploaded reports, the sum can double-count those executions; "
            "the table above shows what each report claimed._"
        )
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
