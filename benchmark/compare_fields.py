#!/usr/bin/env python3
"""Compare OpenFOAM ASCII fields between two case dirs at a given time.

Usage: compare_fields.py <caseA> <caseB> <time> [fields...]
Prints max-abs and rms differences (per component for vectors).
"""
import re
import sys
import math


def read_field(path):
    with open(path) as f:
        txt = f.read()
    m = re.search(
        r"internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n"
        r"(\d+)\s*\n\(\n(.*?)\n\)\s*;",
        txt, re.S)
    if not m:
        mu = re.search(
            r"internalField\s+uniform\s+"
            r"(\(?[-0-9eE. +]+\)?)\s*;", txt)
        if mu:
            raise SystemExit(f"{path}: uniform field; nothing to compare")
        raise SystemExit(f"{path}: could not parse internalField")
    kind, n, body = m.group(1), int(m.group(2)), m.group(3)
    vals = []
    if kind == "scalar":
        vals = [[float(x)] for x in body.split()]
    else:
        for line in body.strip().split("\n"):
            nums = [float(x) for x in line.strip().strip("()").split()]
            vals.append(nums)
    assert len(vals) == n, f"{path}: expected {n}, got {len(vals)}"
    return vals


def main():
    a, b, t = sys.argv[1], sys.argv[2], sys.argv[3]
    fields = sys.argv[4:] or ["p", "U"]
    for fld in fields:
        va = read_field(f"{a}/{t}/{fld}")
        vb = read_field(f"{b}/{t}/{fld}")
        ncmp = len(va[0])
        for c in range(ncmp):
            diffs = [abs(x[c] - y[c]) for x, y in zip(va, vb)]
            mags = [abs(x[c]) for x in va]
            maxd = max(diffs)
            rms = math.sqrt(sum(d*d for d in diffs)/len(diffs))
            scale = max(mags) or 1.0
            comp = f"[{c}]" if ncmp > 1 else ""
            print(f"{fld}{comp}: maxAbsDiff={maxd:.3e} rmsDiff={rms:.3e} "
                  f"fieldMax={scale:.3e} relMax={maxd/scale:.3e}")


if __name__ == "__main__":
    main()
