"""Analyze HumanEval prompts for ambiguity characteristics.

Classifies prompts into categories:
- SOLID: Has examples + clear algorithm description + type hints → one interpretation
- SOFT: Has examples but underspecified edge cases or vague wording
- FUZZY: Multiple reasonable interpretations of what "correct" means

This is a heuristic/static analysis — no LLM calls needed.
"""
import json
import re
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parent / "benchmarks" / "HumanEval.jsonl"


def analyze_prompt(problem: dict) -> dict:
    """Analyze a single HumanEval prompt for ambiguity signals."""
    prompt = problem["prompt"]
    task_id = problem["task_id"]
    docstring = ""
    # Extract docstring
    m = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if not m:
        m = re.search(r"'''(.*?)'''", prompt, re.DOTALL)
    if m:
        docstring = m.group(1).strip()

    # Count concrete examples (>>> lines)
    examples = re.findall(r'>>>\s*\w+', docstring)
    n_examples = len(examples)

    # Check for type hints in signature
    has_type_hints = bool(re.search(r'->\s*\w', prompt))

    # Check for vague/ambiguous language
    vague_phrases = [
        r'\bshould\b',      # "should" vs "must"
        r'\bcan be\b',
        r'\bmight\b',
        r'\bpossibly\b',
        r'\bfor example\b',  # just examples, not spec
        r'\bnote:\b',        # clarifications suggest ambiguity existed
        r'\bassume\b',       # assumptions = underspecified
        r'\bsigned digit\b', # the famous -123 problem
    ]
    vague_count = sum(1 for p in vague_phrases if re.search(p, docstring, re.IGNORECASE))

    # Check for edge case specification
    edge_case_words = [
        r'\bempty\b', r'\bzero\b', r'\bnegative\b', r'\bnone\b',
        r'\bsingle\b', r'\bno\b', r'\bnothing\b',
    ]
    edge_coverage = sum(1 for p in edge_case_words if re.search(p, docstring, re.IGNORECASE))

    # Check for mathematical/algorithmic precision
    has_formula = bool(re.search(r'[=<>]+.*[=<>]|sum|product|mean|median|sort|reverse', docstring, re.IGNORECASE))

    # Check if prompt provides helper functions (encode/decode patterns)
    has_helper = bool(re.search(r'def \w+\(', prompt.split('"""')[0] if '"""' in prompt else ''))
    # Actually check all of prompt for multiple def statements
    defs = re.findall(r'^def \w+', prompt, re.MULTILINE)
    has_helper = len(defs) > 1  # more than the target function

    # Check for "the first signed digit" type complexity
    has_tricky_semantics = bool(re.search(
        r'signed digit|operator precedence|left.to.right|bracket|paren|recurs',
        docstring, re.IGNORECASE
    ))

    # Scoring
    # More examples + type hints + formula = more solid
    # Vague language + tricky semantics + few examples = more fuzzy
    solidity_score = (
        min(n_examples, 4) * 2     # examples are strong (max 8)
        + (3 if has_type_hints else 0)
        + (2 if has_formula else 0)
        + edge_coverage
        - vague_count * 2
        - (3 if has_tricky_semantics else 0)
        - (2 if has_helper else 0)   # helpers add complexity
    )

    if solidity_score >= 8:
        category = "SOLID"
    elif solidity_score >= 4:
        category = "SOFT"
    else:
        category = "FUZZY"

    return {
        "task_id": task_id,
        "category": category,
        "solidity_score": solidity_score,
        "n_examples": n_examples,
        "has_type_hints": has_type_hints,
        "has_formula": has_formula,
        "has_helper": has_helper,
        "has_tricky_semantics": has_tricky_semantics,
        "vague_count": vague_count,
        "edge_coverage": edge_coverage,
        "docstring_len": len(docstring),
    }


def main():
    import csv

    problems = []
    with open(BENCHMARKS) as f:
        for line in f:
            problems.append(json.loads(line))

    analyses = [analyze_prompt(p) for p in problems]

    # Load DYTM results for correlation
    dytm_file = Path(__file__).resolve().parent / "results" / "dytm_gpt-4o-mini_20260401_162341.csv"
    base_file = Path(__file__).resolve().parent / "results" / "baseline_gpt-4o-mini_20260401_162139.csv"

    dytm_results = {}
    base_results = {}
    if dytm_file.exists():
        with open(dytm_file) as f:
            dytm_results = {r["problem_id"]: r for r in csv.DictReader(f)}
    if base_file.exists():
        with open(base_file) as f:
            base_results = {r["problem_id"]: r for r in csv.DictReader(f)}

    # Summary stats
    cats = {"SOLID": [], "SOFT": [], "FUZZY": []}
    for a in analyses:
        cats[a["category"]].append(a)

    print("=" * 70)
    print("HUMANEVAL AMBIGUITY ANALYSIS")
    print("=" * 70)

    for cat in ["SOLID", "SOFT", "FUZZY"]:
        items = cats[cat]
        print(f"\n{cat}: {len(items)} problems")

        if dytm_results:
            dytm_pass = sum(1 for a in items if dytm_results.get(a["task_id"], {}).get("passed") == "True")
            base_pass = sum(1 for a in items if base_results.get(a["task_id"], {}).get("passed") == "True")
            n = len(items)
            print(f"  Baseline pass rate: {base_pass}/{n} = {base_pass/n*100:.1f}%")
            print(f"  DYTM pass rate:     {dytm_pass}/{n} = {dytm_pass/n*100:.1f}%")
            delta = dytm_pass - base_pass
            print(f"  Delta:              {'+' if delta >= 0 else ''}{delta}")

            # Breakdown of DYTM failures
            failures = [a for a in items if dytm_results.get(a["task_id"], {}).get("passed") != "True"]
            regressions = [a for a in items
                          if base_results.get(a["task_id"], {}).get("passed") == "True"
                          and dytm_results.get(a["task_id"], {}).get("passed") != "True"]
            fixes = [a for a in items
                    if base_results.get(a["task_id"], {}).get("passed") != "True"
                    and dytm_results.get(a["task_id"], {}).get("passed") == "True"]
            if regressions:
                print(f"  Regressions: {[a['task_id'] for a in regressions]}")
            if fixes:
                print(f"  Fixes:       {[a['task_id'] for a in fixes]}")

    # Show FUZZY problems in detail
    print("\n" + "=" * 70)
    print("FUZZY PROBLEMS (most likely to benefit from multi-interpretation)")
    print("=" * 70)
    for a in sorted(cats["FUZZY"], key=lambda x: x["solidity_score"]):
        tid = a["task_id"]
        dr = dytm_results.get(tid, {})
        br = base_results.get(tid, {})
        dytm_status = "PASS" if dr.get("passed") == "True" else "FAIL"
        base_status = "PASS" if br.get("passed") == "True" else "FAIL"
        pbt = "pbt:OK" if dr.get("pbt_passed") == "True" else "pbt:FAIL"
        retries = dr.get("retries", "?")
        print(f"  {tid:25s} score={a['solidity_score']:3d}  "
              f"base={base_status}  dytm={dytm_status}  {pbt}  retries={retries}  "
              f"ex={a['n_examples']} hints={a['has_type_hints']} tricky={a['has_tricky_semantics']}")

    # Key insight: What percentage of DYTM regressions are in FUZZY/SOFT?
    print("\n" + "=" * 70)
    print("REGRESSION ANALYSIS BY CATEGORY")
    print("=" * 70)
    all_regressions = []
    for a in analyses:
        tid = a["task_id"]
        if (base_results.get(tid, {}).get("passed") == "True"
                and dytm_results.get(tid, {}).get("passed") != "True"):
            all_regressions.append(a)

    if all_regressions:
        for cat in ["SOLID", "SOFT", "FUZZY"]:
            in_cat = [a for a in all_regressions if a["category"] == cat]
            print(f"  {cat}: {len(in_cat)}/{len(all_regressions)} regressions")
            for a in in_cat:
                print(f"    {a['task_id']} (score={a['solidity_score']})")
    else:
        print("  No regressions found (or results files missing)")


if __name__ == "__main__":
    main()

