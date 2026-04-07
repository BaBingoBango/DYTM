"""Quick comparison of multi-spec early results vs prior runs."""
import csv

# Multi-spec early results (from terminal output)
multi_spec = {
    "HumanEval/0": ("FAIL", "OK", 1),
    "HumanEval/1": ("PASS", "OK", 3),
    "HumanEval/2": ("PASS", "OK", 2),
    "HumanEval/3": ("PASS", "OK", 1),
    "HumanEval/4": ("PASS", "OK", 2),
    "HumanEval/5": ("PASS", "OK", 2),
    "HumanEval/6": ("PASS", "OK", 0),
    "HumanEval/7": ("PASS", "OK", 0),
    "HumanEval/8": ("PASS", "OK", 0),
    "HumanEval/9": ("FAIL", "OK", 1),
    "HumanEval/10": ("FAIL", "OK", 2),
    "HumanEval/11": ("PASS", "OK", 2),
    "HumanEval/12": ("PASS", "OK", 0),
    "HumanEval/13": ("PASS", "OK", 3),
    "HumanEval/14": ("FAIL", "OK", 1),
    "HumanEval/15": ("PASS", "FAIL", 4),
    "HumanEval/16": ("PASS", "OK", 2),
    "HumanEval/17": ("FAIL", "FAIL", 4),
    "HumanEval/18": ("PASS", "OK", 0),
    "HumanEval/19": ("PASS", "OK", 2),
    "HumanEval/20": ("PASS", "FAIL", 4),
    "HumanEval/21": ("PASS", "OK", 1),
    "HumanEval/22": ("PASS", "OK", 1),
    "HumanEval/23": ("PASS", "OK", 0),
}

# Load prior results
with open("eval/results/baseline_gpt-4o-mini_20260401_162139.csv") as f:
    base = {r["problem_id"]: r for r in csv.DictReader(f)}
with open("eval/results/dytm_gpt-4o-mini_20260401_162341.csv") as f:
    single = {r["problem_id"]: r for r in csv.DictReader(f)}

print("EARLY MULTI-SPEC RESULTS vs PRIOR RUNS (first 24 problems)")
print("=" * 80)
print(f"{'Problem':<20} {'Baseline':>8} {'Single-Spec':>12} {'Multi-Spec':>11}  Notes")
print("-" * 80)

new_regressions = []
new_fixes = []
for tid in sorted(multi_spec.keys(), key=lambda x: int(x.split("/")[1])):
    ms_pass, ms_pbt, ms_retries = multi_spec[tid]
    b_pass = "PASS" if base.get(tid, {}).get("passed") == "True" else "FAIL"
    s_pass = "PASS" if single.get(tid, {}).get("passed") == "True" else "FAIL"

    notes = ""
    if ms_pass == "FAIL" and s_pass == "PASS":
        notes = "⚠ NEW REGRESSION vs single-spec"
        new_regressions.append(tid)
    elif ms_pass == "FAIL" and b_pass == "PASS":
        notes = "⚠ regression vs baseline"
    elif ms_pass == "PASS" and s_pass == "FAIL":
        notes = "✓ FIXED vs single-spec"
        new_fixes.append(tid)
    elif ms_pass == "PASS" and b_pass == "FAIL":
        notes = "✓ fixed vs baseline"

    print(f"{tid:<20} {b_pass:>8} {s_pass:>12} {ms_pass:>11}  {notes}")

print()
ms_pass_count = sum(1 for p, _, _ in multi_spec.values() if p == "PASS")
b_pass_count = sum(1 for tid in multi_spec if base.get(tid, {}).get("passed") == "True")
s_pass_count = sum(1 for tid in multi_spec if single.get(tid, {}).get("passed") == "True")

print(f"Pass rate (first 24):  baseline={b_pass_count}/24  single-spec={s_pass_count}/24  multi-spec={ms_pass_count}/24")
print(f"New regressions vs single-spec: {len(new_regressions)}  {new_regressions}")
print(f"New fixes vs single-spec:       {len(new_fixes)}  {new_fixes}")

