"""Compare baseline vs DYTM eval results.

Usage:
    python -m eval.compare eval/results/baseline_*.csv eval/results/dytm_*.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_results(path: Path) -> dict[str, dict]:
    """Load a results CSV into a dict keyed by problem_id."""
    with open(path, encoding="utf-8") as f:
        return {row["problem_id"]: row for row in csv.DictReader(f)}


def compare(baseline_path: Path, dytm_path: Path) -> None:
    baseline = load_results(baseline_path)
    dytm = load_results(dytm_path)

    # Find common problems
    common = set(baseline.keys()) & set(dytm.keys())
    if not common:
        print("ERROR: No overlapping problem IDs found.")
        sys.exit(1)

    b_pass = sum(1 for pid in common if baseline[pid]["passed"] == "True")
    d_pass = sum(1 for pid in common if dytm[pid]["passed"] == "True")
    total = len(common)

    b_rate = b_pass / total * 100
    d_rate = d_pass / total * 100
    delta = d_rate - b_rate

    # Find problems DYTM fixed and broke
    dytm_fixed = [pid for pid in sorted(common)
                  if baseline[pid]["passed"] == "False" and dytm[pid]["passed"] == "True"]
    dytm_broke = [pid for pid in sorted(common)
                  if baseline[pid]["passed"] == "True" and dytm[pid]["passed"] == "False"]

    avg_retries = sum(int(dytm[pid].get("retries", 0)) for pid in common) / total

    print("=" * 60)
    print("  DYTM vs Baseline Comparison")
    print("=" * 60)
    print(f"  Problems compared: {total}")
    print()
    print(f"  Baseline pass@1:   {b_pass}/{total} = {b_rate:.1f}%")
    print(f"  DYTM pass@1:       {d_pass}/{total} = {d_rate:.1f}%")
    print(f"  Delta:             {delta:+.1f}%")
    print(f"  Avg DYTM retries:  {avg_retries:.1f}")
    print()

    if dytm_fixed:
        print(f"  DYTM FIXED ({len(dytm_fixed)} problems):")
        for pid in dytm_fixed:
            print(f"    + {pid}")

    if dytm_broke:
        print(f"  DYTM REGRESSED ({len(dytm_broke)} problems):")
        for pid in dytm_broke:
            print(f"    - {pid}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs DYTM")
    parser.add_argument("baseline", type=Path, help="Path to baseline CSV")
    parser.add_argument("dytm", type=Path, help="Path to DYTM CSV")
    args = parser.parse_args()
    compare(args.baseline, args.dytm)


if __name__ == "__main__":
    main()

