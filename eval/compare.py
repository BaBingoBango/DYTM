"""Compare baseline vs DYTM eval results.

Supports two-way (baseline vs DYTM) or three-way comparison
(baseline-weak, baseline-strong, DYTM cross-model) for controlled
cross-model experiments.

Usage:
    # Two-way (original)
    python -m eval.compare baseline.csv dytm.csv

    # Three-way (cross-model controlled experiment)
    python -m eval.compare baseline_weak.csv dytm_cross.csv --upgraded baseline_strong.csv
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


def _pass_count(results: dict[str, dict], pids: set[str]) -> int:
    return sum(1 for pid in pids if results[pid]["passed"] == "True")


def compare(baseline_path: Path, dytm_path: Path, upgraded_path: Path | None = None) -> None:
    baseline = load_results(baseline_path)
    dytm = load_results(dytm_path)
    upgraded = load_results(upgraded_path) if upgraded_path else None

    # Find common problems across all provided CSVs
    common = set(baseline.keys()) & set(dytm.keys())
    if upgraded:
        common &= set(upgraded.keys())
    if not common:
        print("ERROR: No overlapping problem IDs found.")
        sys.exit(1)

    total = len(common)
    b_pass = _pass_count(baseline, common)
    d_pass = _pass_count(dytm, common)

    b_rate = b_pass / total * 100
    d_rate = d_pass / total * 100
    delta_vs_baseline = d_rate - b_rate

    # Find problems DYTM fixed and broke vs baseline
    dytm_fixed = [pid for pid in sorted(common)
                  if baseline[pid]["passed"] == "False" and dytm[pid]["passed"] == "True"]
    dytm_broke = [pid for pid in sorted(common)
                  if baseline[pid]["passed"] == "True" and dytm[pid]["passed"] == "False"]

    avg_retries = sum(int(dytm[pid].get("retries", 0)) for pid in common) / total

    print("=" * 70)
    if upgraded:
        print("  Three-Way Cross-Model Comparison")
    else:
        print("  DYTM vs Baseline Comparison")
    print("=" * 70)
    print(f"  Problems compared: {total}")
    print()

    print(f"  Baseline (weak):   {b_pass}/{total} = {b_rate:.1f}%")
    delta_vs_upgraded = 0.0
    if upgraded:
        u_pass = _pass_count(upgraded, common)
        u_rate = u_pass / total * 100
        delta_vs_upgraded = d_rate - u_rate
        print(f"  Baseline (strong): {u_pass}/{total} = {u_rate:.1f}%  ← 'just use a better model'")
    print(f"  DYTM cross-model:  {d_pass}/{total} = {d_rate:.1f}%")
    print()
    print(f"  DYTM vs weak baseline:   {delta_vs_baseline:+.1f}%")
    if upgraded:
        print(f"  DYTM vs strong baseline: {delta_vs_upgraded:+.1f}%  ← THE KEY NUMBER")
        print()
        if delta_vs_upgraded > 0:
            print(f"  ✅ DYTM (cheap Spec + cheap Coder) BEATS the expensive model alone!")
        elif delta_vs_upgraded == 0:
            print(f"  ≈  DYTM matches the expensive model (but at lower cost).")
        else:
            print(f"  ⚠️  DYTM underperforms the expensive model by {abs(delta_vs_upgraded):.1f}%.")

        # Find problems where DYTM beats the upgraded baseline
        dytm_beats_upgraded = [pid for pid in sorted(common)
                               if upgraded[pid]["passed"] == "False" and dytm[pid]["passed"] == "True"]
        upgraded_beats_dytm = [pid for pid in sorted(common)
                               if upgraded[pid]["passed"] == "True" and dytm[pid]["passed"] == "False"]
        if dytm_beats_upgraded:
            print(f"\n  DYTM fixes (vs strong baseline): {len(dytm_beats_upgraded)} problems")
            for pid in dytm_beats_upgraded:
                print(f"    + {pid}")
        if upgraded_beats_dytm:
            print(f"\n  Strong baseline wins (vs DYTM): {len(upgraded_beats_dytm)} problems")
            for pid in upgraded_beats_dytm:
                print(f"    - {pid}")

    print(f"\n  Avg DYTM retries:  {avg_retries:.1f}")
    print()

    if dytm_fixed:
        print(f"  DYTM FIXED vs weak baseline ({len(dytm_fixed)} problems):")
        for pid in dytm_fixed:
            print(f"    + {pid}")

    if dytm_broke:
        print(f"  DYTM REGRESSED vs weak baseline ({len(dytm_broke)} problems):")
        for pid in dytm_broke:
            print(f"    - {pid}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs DYTM results")
    parser.add_argument("baseline", type=Path, help="Path to baseline CSV (weak/coder model)")
    parser.add_argument("dytm", type=Path, help="Path to DYTM CSV")
    parser.add_argument("--upgraded", type=Path, default=None,
                        help="Path to upgraded baseline CSV (strong model, zero-shot). "
                             "Enables three-way comparison to control for model quality.")
    args = parser.parse_args()
    compare(args.baseline, args.dytm, upgraded_path=args.upgraded)


if __name__ == "__main__":
    main()

