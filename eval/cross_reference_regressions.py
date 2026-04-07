"""Cross-reference: do deterministic ambiguity detections predict DYTM regressions?"""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, ".")
from eval.deterministic_taxonomy import scan_prompt

# Load problems
problems = {}
with open("eval/benchmarks/HumanEval.jsonl") as f:
    for line in f:
        p = json.loads(line)
        problems[p["task_id"]] = p

# Load results
dytm_file = Path("eval/results/dytm_gpt-4o-mini_20260401_162341.csv")
base_file = Path("eval/results/baseline_gpt-4o-mini_20260401_162139.csv")

with open(dytm_file) as f:
    dytm = {r["problem_id"]: r for r in csv.DictReader(f)}
with open(base_file) as f:
    base = {r["problem_id"]: r for r in csv.DictReader(f)}

# Find regressions
regressions = [
    tid for tid in dytm
    if base.get(tid, {}).get("passed") == "True"
    and dytm[tid].get("passed") != "True"
]

print("REGRESSION PROBLEMS — Deterministic Ambiguity Scan")
print("=" * 70)
for tid in sorted(regressions, key=lambda x: int(x.split("/")[1])):
    prompt = problems[tid]["prompt"]
    analysis = scan_prompt(prompt)

    high_conf = [a for a in analysis.axes if a.confidence >= 0.7]

    print(f"\n{tid}:")
    if high_conf:
        for axis in high_conf:
            print(f"  ⚠ [{axis.confidence:.0%}] {axis.name}: {axis.description}")
            print(f"     matched: \"{axis.pattern_matched}\"")
            for i, interp in enumerate(axis.interpretations):
                print(f"     {i+1}. {interp}")
    elif analysis.axes:
        print(f"  Low-confidence axes only: {[a.name for a in analysis.axes]}")
    else:
        print(f"  ✗ No ambiguity detected (deterministic scan missed this)")

# Summary
print("\n" + "=" * 70)
print(f"Of {len(regressions)} regressions:")
caught = sum(1 for tid in regressions
             if any(a.confidence >= 0.7 for a in scan_prompt(problems[tid]["prompt"]).axes))
low_conf = sum(1 for tid in regressions
               if scan_prompt(problems[tid]["prompt"]).axes
               and not any(a.confidence >= 0.7 for a in scan_prompt(problems[tid]["prompt"]).axes))
missed = sum(1 for tid in regressions
             if not scan_prompt(problems[tid]["prompt"]).axes)
print(f"  Caught by high-confidence scan:  {caught}")
print(f"  Low-confidence match only:       {low_conf}")
print(f"  Completely missed:               {missed}")


