"""Quick check: how many HumanEval problems have extractable examples?"""
import json
from dytm.nodes.multi_spec import extract_docstring_examples

problems = []
with open("eval/benchmarks/HumanEval.jsonl") as f:
    for line in f:
        problems.append(json.loads(line))

with_ex = 0
without_ex = 0
total_ex = 0
no_ex_ids = []
for p in problems:
    ex = extract_docstring_examples(p["prompt"])
    if ex:
        with_ex += 1
        total_ex += len(ex)
    else:
        without_ex += 1
        no_ex_ids.append(p["task_id"])

print(f"Problems with examples:    {with_ex}/164 ({with_ex/164*100:.0f}%)")
print(f"Problems without examples: {without_ex}/164 ({without_ex/164*100:.0f}%)")
print(f"Total examples extracted:  {total_ex}")
print(f"Avg examples per problem:  {total_ex/164:.1f} (overall), {total_ex/with_ex:.1f} (when present)")
print()
print("Problems WITHOUT examples (filter disabled, score-only selection):")
for tid in no_ex_ids[:20]:
    print(f"  {tid}")
if len(no_ex_ids) > 20:
    print(f"  ... and {len(no_ex_ids) - 20} more")

