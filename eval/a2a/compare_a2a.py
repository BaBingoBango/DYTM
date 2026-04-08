"""Quick comparison of A2A baseline vs DYTM vs Hybrid results."""
import csv
import glob
import sys

baseline_path = "eval/a2a/results/a2a_baseline_20260407_222753.csv"

# Find latest DYTM and Hybrid results
dytm_files = sorted(glob.glob("eval/a2a/results/a2a_dytm_*.csv"))
hybrid_files = sorted(glob.glob("eval/a2a/results/a2a_hybrid_*.csv"))

dytm_path = dytm_files[-1] if dytm_files else None
hybrid_path = hybrid_files[-1] if hybrid_files else None

print(f"Comparing:")
print(f"  Baseline: {baseline_path}")
if dytm_path:
    print(f"  DYTM:     {dytm_path}")
if hybrid_path:
    print(f"  Hybrid:   {hybrid_path}")
print()


def load_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["task_id"]] = row
    return data


baseline = load_csv(baseline_path)
conditions = {"Baseline": baseline}

if dytm_path:
    conditions["DYTM"] = load_csv(dytm_path)
if hybrid_path:
    conditions["Hybrid"] = load_csv(hybrid_path)


def compare_pair(name_a, data_a, name_b, data_b):
    """Compare two conditions and print results."""
    improved, regressed, tied = [], [], []

    for tid in sorted(data_a.keys()):
        if tid not in data_b:
            continue
        a = int(data_a[tid]["final_intent_match"])
        b = int(data_b[tid]["final_intent_match"])
        delta = b - a
        if delta > 0:
            improved.append((tid, a, b, delta))
        elif delta < 0:
            regressed.append((tid, a, b, delta))
        else:
            tied.append((tid, a, b))

    print(f"=== {name_b} vs {name_a} ===")
    print(f"  Improved ({name_b} > {name_a}): {len(improved)}")
    for tid, a, b, d in improved[:10]:
        print(f"    {tid}: {a} -> {b} (+{d})")
    if len(improved) > 10:
        print(f"    ... and {len(improved) - 10} more")

    print(f"  Regressed ({name_b} < {name_a}): {len(regressed)}")
    for tid, a, b, d in regressed[:10]:
        print(f"    {tid}: {a} -> {b} ({d})")
    if len(regressed) > 10:
        print(f"    ... and {len(regressed) - 10} more")

    print(f"  Tied: {len(tied)}")
    print()


# Aggregate stats
print("=== AGGREGATE ===")
for name, data in conditions.items():
    scores = [int(data[t]["final_intent_match"]) for t in data]
    high = sum(1 for s in scores if s >= 4)
    drift = sum(1 for t in data if data[t]["drift_detected"].lower() == "true")
    avg = sum(scores) / len(scores)
    print(f"  {name:10s}: avg={avg:.2f}/5  high>=4={high}/{len(scores)} ({high/len(scores)*100:.0f}%)  drift={drift}/{len(scores)}")
print()

# Pairwise comparisons
if "DYTM" in conditions:
    compare_pair("Baseline", baseline, "DYTM", conditions["DYTM"])

if "Hybrid" in conditions:
    compare_pair("Baseline", baseline, "Hybrid", conditions["Hybrid"])

if "DYTM" in conditions and "Hybrid" in conditions:
    compare_pair("DYTM", conditions["DYTM"], "Hybrid", conditions["Hybrid"])

# On baseline failures
print("=== ON BASELINE FAILURES (<=3/5) ===")
bl_fails = [t for t in baseline if int(baseline[t]["final_intent_match"]) <= 3]
print(f"  Baseline failures: {len(bl_fails)}/50")

for name, data in conditions.items():
    if name == "Baseline":
        continue
    helped = sum(1 for t in bl_fails if t in data and int(data[t]["final_intent_match"]) > int(baseline[t]["final_intent_match"]))
    hurt = sum(1 for t in bl_fails if t in data and int(data[t]["final_intent_match"]) < int(baseline[t]["final_intent_match"]))
    same = sum(1 for t in bl_fails if t in data and int(data[t]["final_intent_match"]) == int(baseline[t]["final_intent_match"]))
    print(f"  {name:10s}: helped={helped}  hurt={hurt}  same={same}")

# Rejection pattern (DYTM & Hybrid)
print()
print("=== REJECTION PATTERN ===")
for name, data in conditions.items():
    if name == "Baseline":
        continue
    rej = [int(data[t]["contract_rejections"]) for t in data]
    print(f"  {name}: rej=0:{sum(1 for r in rej if r==0)}  rej=1:{sum(1 for r in rej if r==1)}  rej=2:{sum(1 for r in rej if r==2)}  rej=3+:{sum(1 for r in rej if r>=3)}")
