"""Quick summary of all full-run results."""
import csv

files = {
    "baseline mini": "eval/results/baseline_gpt-4o-mini_20260401_162139.csv",
    "DYTM mini":     "eval/results/dytm_gpt-4o-mini_20260401_162341.csv",
    "baseline 4o":   "eval/results/baseline_gpt-4o_20260401_173513.csv",
    "DYTM cross":    "eval/results/dytm_spec-gpt-4o_coder-gpt-4o-mini_20260401_173451.csv",
}

data = {}
for label, path in files.items():
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        data[label] = rows
        total = len(rows)
        passed = sum(1 for r in rows if r["passed"] == "True")
        pbt = sum(1 for r in rows if r.get("pbt_passed") == "True")
        print(f"{label:20s}: {passed:3d}/{total} = {passed/total*100:5.1f}%  (pbt={pbt if pbt else 'n/a'})")
    except Exception as e:
        print(f"{label:20s}: ERROR {e}")

# Cross-model detail
print("\n--- Cross-model DYTM (spec:4o / coder:mini) failures ---")
cross = data.get("DYTM cross", [])
for r in sorted(cross, key=lambda x: int(x["problem_id"].split("/")[1])):
    if r["passed"] == "False":
        pbt = r.get("pbt_passed", "?")
        retries = r.get("retries", "?")
        reason = r.get("failure_reason", "")[:80]
        print(f"  {r['problem_id']:25s} pbt={pbt}  retries={retries}  {reason}")

# Compare cross-model vs baseline-4o
print("\n--- Compared to gpt-4o baseline ---")
base_4o_rows = {r["problem_id"]: r for r in data.get("baseline 4o", [])}
cross_rows = {r["problem_id"]: r for r in data.get("DYTM cross", [])}
fixed = []
regressed = []
for tid in sorted(base_4o_rows.keys(), key=lambda x: int(x.split("/")[1])):
    b = base_4o_rows[tid]["passed"] == "True"
    d = cross_rows.get(tid, {}).get("passed") == "True"
    if not b and d:
        fixed.append(tid)
    if b and not d:
        regressed.append(tid)
print(f"  Fixed by DYTM:     {len(fixed)}  {fixed}")
print(f"  Regressed by DYTM: {len(regressed)}  {regressed}")

