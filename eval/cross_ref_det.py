"""Cross-reference DYTM regressions/fixes with deterministic taxonomy."""
import csv, json
from eval.deterministic_taxonomy import scan_prompt

problems = []
with open('eval/benchmarks/HumanEval.jsonl') as f:
    for line in f:
        problems.append(json.loads(line))

dytm_results = {}
with open('eval/results/dytm_gpt-4o-mini_20260401_162341.csv') as f:
    dytm_results = {r['problem_id']: r for r in csv.DictReader(f)}
base_results = {}
with open('eval/results/baseline_gpt-4o-mini_20260401_162139.csv') as f:
    base_results = {r['problem_id']: r for r in csv.DictReader(f)}

print('=== REGRESSIONS (baseline PASS, DYTM FAIL) ===')
regressions = []
for p in problems:
    tid = p['task_id']
    bp = base_results.get(tid, {}).get('passed') == 'True'
    dp = dytm_results.get(tid, {}).get('passed') == 'True'
    if bp and not dp:
        a = scan_prompt(p['prompt'])
        strong = [x for x in a.axes if x.confidence >= 0.5]
        names = ', '.join(x.name for x in strong) if strong else 'none'
        print(f'  {tid:25s} axes={names}  interps={a.interpretation_count()}')
        regressions.append((tid, strong))

print(f'Total: {len(regressions)} regressions, {sum(1 for _,s in regressions if s)} with strong axes')

print()
print('=== FIXES (baseline FAIL, DYTM PASS) ===')
fixes = []
for p in problems:
    tid = p['task_id']
    bp = base_results.get(tid, {}).get('passed') == 'True'
    dp = dytm_results.get(tid, {}).get('passed') == 'True'
    if not bp and dp:
        a = scan_prompt(p['prompt'])
        strong = [x for x in a.axes if x.confidence >= 0.5]
        names = ', '.join(x.name for x in strong) if strong else 'none'
        print(f'  {tid:25s} axes={names}')
        fixes.append((tid, strong))

print(f'Total: {len(fixes)} fixes, {sum(1 for _,s in fixes if s)} with strong axes')

