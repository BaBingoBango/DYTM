"""Analyze regressions and fixes from full HumanEval run."""
import csv
import sys

dytm_file = "eval/results/dytm_gpt-4o-mini_20260401_162341.csv"
baseline_file = "eval/results/baseline_gpt-4o-mini_20260401_162139.csv"

regressed = [
    "HumanEval/108", "HumanEval/118", "HumanEval/133", "HumanEval/141",
    "HumanEval/160", "HumanEval/22", "HumanEval/56", "HumanEval/75", "HumanEval/95",
]
fixed = [
    "HumanEval/10", "HumanEval/120", "HumanEval/127", "HumanEval/134",
    "HumanEval/28", "HumanEval/35", "HumanEval/38", "HumanEval/4",
    "HumanEval/50", "HumanEval/74", "HumanEval/83", "HumanEval/93",
]

with open(dytm_file) as f:
    dytm_rows = {r["problem_id"]: r for r in csv.DictReader(f)}

print("=== REGRESSIONS (baseline passed, DYTM failed) ===")
for tid in sorted(regressed, key=lambda x: int(x.split("/")[1])):
    r = dytm_rows.get(tid, {})
    err = r.get("failure_reason", "")[:200]
    print(f"  {tid}: passed={r.get('passed')}, pbt={r.get('pbt_passed')}, retries={r.get('retries')}")
    if err:
        print(f"    reason: {err}")

print()
print("=== FIXES (baseline failed, DYTM passed) ===")
for tid in sorted(fixed, key=lambda x: int(x.split("/")[1])):
    r = dytm_rows.get(tid, {})
    print(f"  {tid}: passed={r.get('passed')}, pbt={r.get('pbt_passed')}, retries={r.get('retries')}")

# Overall failure analysis
print()
print("=== ALL DYTM FAILURES ===")
failures = []
for tid, r in sorted(dytm_rows.items(), key=lambda x: int(x[0].split("/")[1])):
    if r.get("passed") == "False":
        pbt = r.get("pbt_passed")
        retries = r.get("retries")
        err = r.get("failure_reason", "")[:100]
        failures.append((tid, pbt, retries, err))
        print(f"  {tid}: pbt={pbt}, retries={retries}, reason={err}")

print(f"\nTotal failures: {len(failures)}")

# Categorize
pbt_pass_canon_fail = sum(1 for _, pbt, _, _ in failures if pbt == "True")
pbt_fail_canon_fail = sum(1 for _, pbt, _, _ in failures if pbt == "False")
max_retries = sum(1 for _, _, r, _ in failures if r == "4")
error_failures = sum(1 for _, _, _, e in failures if e)

print(f"  PBT passed but canonical failed (weak contract): {pbt_pass_canon_fail}")
print(f"  PBT failed and canonical failed: {pbt_fail_canon_fail}")
print(f"  Hit max retries (4): {max_retries}")
print(f"  Had error: {error_failures}")


