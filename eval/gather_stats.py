"""Gather all stats for the micro-paper README."""
import csv

with open('eval/results/dytm_gpt-4o-mini_20260401_162341.csv') as f:
    dytm = list(csv.DictReader(f))
with open('eval/results/baseline_gpt-4o-mini_20260401_162139.csv') as f:
    base = list(csv.DictReader(f))
with open('eval/results/dytm_spec-gpt-4o_coder-gpt-4o-mini_20260401_173451.csv') as f:
    cross = list(csv.DictReader(f))

total = len(dytm)
passed = sum(1 for r in dytm if r['passed']=='True')
pbt_ok = sum(1 for r in dytm if r['pbt_passed']=='True')
retries = [int(r['retries']) for r in dytm]
times = [float(r['wall_clock_sec']) for r in dytm]

print(f"DYTM pass@1: {passed}/{total} = {passed/total*100:.1f}%")
print(f"PBT pass: {pbt_ok}/{total} = {pbt_ok/total*100:.1f}%")
print(f"Avg retries: {sum(retries)/total:.2f}")
print(f"Avg time: {sum(times)/total:.1f}s")
print(f"Total time: {sum(times):.0f}s = {sum(times)/60:.1f}m")
print(f"Retry dist: 0={retries.count(0)}, 1={retries.count(1)}, 2={retries.count(2)}, 3={retries.count(3)}, 4={retries.count(4)}")

weak = sum(1 for r in dytm if r['pbt_passed']=='True' and r['passed']!='True')
both_fail = sum(1 for r in dytm if r['pbt_passed']!='True' and r['passed']!='True')
print(f"Weak contracts (PBT pass, canonical fail): {weak}")
print(f"Both fail: {both_fail}")

reasons = {}
for r in dytm:
    reason = r.get('failure_reason','')
    if reason:
        reasons[reason] = reasons.get(reason,0)+1
for k,v in sorted(reasons.items(), key=lambda x:-x[1]):
    print(f"  failure: {k}: {v}")

cp = sum(1 for r in cross if r['passed']=='True')
cpbt = sum(1 for r in cross if r['pbt_passed']=='True')
print(f"\nCross-model pass@1: {cp}/{len(cross)} = {cp/len(cross)*100:.1f}%")
print(f"Cross-model PBT pass: {cpbt}/{len(cross)} = {cpbt/len(cross)*100:.1f}%")

bp = sum(1 for r in base if r['passed']=='True')
bt = [float(r['wall_clock_sec']) for r in base]
print(f"\nBaseline pass@1: {bp}/{len(base)} = {bp/len(base)*100:.1f}%")
print(f"Baseline avg time: {sum(bt)/len(bt):.1f}s")

print("\n=== REGRESSION/FIX LIST ===")
base_d = {r['problem_id']:r for r in base}
dytm_d = {r['problem_id']:r for r in dytm}
cross_d = {r['problem_id']:r for r in cross}
for tid in sorted(set(base_d)|set(dytm_d)):
    b = base_d.get(tid,{}).get('passed')=='True'
    d = dytm_d.get(tid,{}).get('passed')=='True'
    c = cross_d.get(tid,{}).get('passed')=='True'
    pbt = dytm_d.get(tid,{}).get('pbt_passed')=='True'
    ret = dytm_d.get(tid,{}).get('retries','?')
    if b and not d:
        label = "REGRESSION"
        pbt_s = "OK" if pbt else "FAIL"
        cross_s = "PASS" if c else "FAIL"
        print(f"  {label}: {tid:25s} pbt={pbt_s}  retries={ret}  cross={cross_s}")
    elif not b and d:
        label = "FIX"
        pbt_s = "OK" if pbt else "FAIL"
        cross_s = "PASS" if c else "FAIL"
        print(f"  {label}:        {tid:25s} pbt={pbt_s}  retries={ret}  cross={cross_s}")

