#!/usr/bin/env python3
"""Quick comparison of two DYTM eval runs."""
import csv, sys

def compare(old_path, new_path):
    with open(old_path) as f:
        old = list(csv.DictReader(f))
    with open(new_path) as f:
        new = list(csv.DictReader(f))

    old_by_id = {r['problem_id']: r for r in old}
    new_by_id = {r['problem_id']: r for r in new}

    old_pass = sum(1 for r in old if r['passed'] == 'True')
    new_pass = sum(1 for r in new if r['passed'] == 'True')
    old_pbt = sum(1 for r in old if r['pbt_passed'] == 'True')
    new_pbt = sum(1 for r in new if r['pbt_passed'] == 'True')
    n = len(new)

    print("=" * 70)
    print("  BEFORE vs AFTER Prompt Tuning")
    print("=" * 70)
    print(f"  Before: {old_pass}/{len(old)} = {old_pass/len(old)*100:.1f}%  (PBT: {old_pbt}/{len(old)} = {old_pbt/len(old)*100:.1f}%)")
    print(f"  After:  {new_pass}/{n} = {new_pass/n*100:.1f}%  (PBT: {new_pbt}/{n} = {new_pbt/n*100:.1f}%)")
    print(f"  Delta:  +{new_pass - old_pass} problems (+{(new_pass - old_pass)/n*100:.1f}%)")
    print()

    old_avg_retries = sum(int(r['retries']) for r in old) / len(old)
    new_avg_retries = sum(int(r['retries']) for r in new) / n
    print(f"  Avg retries: {old_avg_retries:.2f} -> {new_avg_retries:.2f}")
    print()

    fixed = []
    regressed = []
    for pid in sorted(set(list(old_by_id.keys()) + list(new_by_id.keys()))):
        o = old_by_id.get(pid)
        ne = new_by_id.get(pid)
        if not o or not ne:
            continue
        was_pass = o['passed'] == 'True'
        now_pass = ne['passed'] == 'True'
        if not was_pass and now_pass:
            fixed.append(pid)
        elif was_pass and not now_pass:
            regressed.append(pid)

    if fixed:
        print("  FIXED by prompt tuning:")
        for p in fixed:
            o = old_by_id[p]
            ne = new_by_id[p]
            print(f"    + {p:<20s}  was: pbt={o['pbt_passed']}, reason={o['failure_reason'] or 'none'}")
            print(f"      now: pbt={ne['pbt_passed']}, retries={ne['retries']}")

    if regressed:
        print("  REGRESSED:")
        for p in regressed:
            o = old_by_id[p]
            ne = new_by_id[p]
            print(f"    - {p:<20s}  was: passed, now: reason={ne['failure_reason']}")

    print()
    print("  STILL FAILING:")
    for r in sorted(new, key=lambda x: x['problem_id']):
        if r['passed'] == 'False':
            print(f"    {r['problem_id']:<20s}  pbt={r['pbt_passed']:<6s}  retries={r['retries']}  reason={r['failure_reason']}")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        compare(sys.argv[1], sys.argv[2])
    else:
        # Default: compare the two known runs
        compare(
            "eval/results/dytm_gpt-4o-mini_20260401_130315.csv",
            "eval/results/dytm_gpt-4o-mini_20260401_155238.csv",
        )

