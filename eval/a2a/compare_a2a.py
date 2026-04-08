"""Quick comparison of A2A baseline vs DYTM results."""
import csv
import glob
import sys

baseline_path = "eval/a2a/results/a2a_baseline_20260407_222753.csv"

# Find latest DYTM result
dytm_files = sorted(glob.glob("eval/a2a/results/a2a_dytm_*.csv"))
if not dytm_files:
    print("No DYTM results found")
    sys.exit(1)
dytm_path = dytm_files[-1]  # latest
print(f"Comparing:\n  Baseline: {baseline_path}\n  DYTM:     {dytm_path}")

# Also show all DYTM runs for context
if len(dytm_files) > 1:
    print(f"  (All DYTM runs: {len(dytm_files)} total)")
print()

baseline = {}
with open(baseline_path) as f:
    for row in csv.DictReader(f):
        baseline[row["task_id"]] = row

dytm = {}
with open(dytm_path) as f:
    for row in csv.DictReader(f):
        dytm[row["task_id"]] = row

improved, regressed, tied = [], [], []

for tid in sorted(baseline.keys()):
    b = int(baseline[tid]["final_intent_match"])
    d = int(dytm[tid]["final_intent_match"])
    delta = d - b
    if delta > 0:
        improved.append((tid, b, d, delta))
    elif delta < 0:
        regressed.append((tid, b, d, delta))
    else:
        tied.append((tid, b, d))

print("=== IMPROVED (DYTM > Baseline) ===")
for tid, b, d, delta in improved:
    print(f"  {tid}: {b} -> {d} (+{delta})")
print(f"  Total: {len(improved)}")

print()
print("=== REGRESSED (DYTM < Baseline) ===")
for tid, b, d, delta in regressed:
    print(f"  {tid}: {b} -> {d} ({delta})")
print(f"  Total: {len(regressed)}")

print()
print("=== TIED ===")
for tid, b, d in tied:
    print(f"  {tid}: {b} = {d}")
print(f"  Total: {len(tied)}")

print()
print("=== AGGREGATE ===")
b_scores = [int(baseline[t]["final_intent_match"]) for t in baseline]
d_scores = [int(dytm[t]["final_intent_match"]) for t in dytm]
print(f"  Baseline avg: {sum(b_scores)/len(b_scores):.2f}")
print(f"  DYTM avg:     {sum(d_scores)/len(d_scores):.2f}")
print(f"  Delta:        {(sum(d_scores)-sum(b_scores))/len(b_scores):+.2f}")
print(f"  Baseline >=4: {sum(1 for s in b_scores if s>=4)}/50")
print(f"  DYTM >=4:     {sum(1 for s in d_scores if s>=4)}/50")

print()
print("=== ON BASELINE FAILURES (<=3/5) ===")
bl_fails = [
    (t, int(baseline[t]["final_intent_match"]), int(dytm[t]["final_intent_match"]))
    for t in baseline
    if int(baseline[t]["final_intent_match"]) <= 3
]
for tid, b, d in sorted(bl_fails):
    arrow = "UP" if d > b else ("DOWN" if d < b else "SAME")
    print(f"  {tid}: B={b} D={d} {arrow}")
helped = sum(1 for _, b, d in bl_fails if d > b)
hurt = sum(1 for _, b, d in bl_fails if d < b)
same = sum(1 for _, b, d in bl_fails if d == b)
print(f"  DYTM helped: {helped}/{len(bl_fails)}")
print(f"  DYTM hurt:   {hurt}/{len(bl_fails)}")
print(f"  Same:         {same}/{len(bl_fails)}")

print()
print("=== REJECTION PATTERN ===")
rej_counts = [int(dytm[t]["contract_rejections"]) for t in dytm]
print(f"  Tasks w/ 0 rejections: {sum(1 for r in rej_counts if r==0)}")
print(f"  Tasks w/ 1 rejection:  {sum(1 for r in rej_counts if r==1)}")
print(f"  Tasks w/ 2 rejections: {sum(1 for r in rej_counts if r==2)}")
print(f"  Tasks w/ 3+ rejections: {sum(1 for r in rej_counts if r>=3)}")

