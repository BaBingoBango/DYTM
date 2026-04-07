"""Measure spec diversity: does temperature actually produce different contracts?

Runs 3 specs at different temperatures for a handful of HumanEval problems
and measures how much they actually differ. This tells us whether naive
temperature sampling is "good enough" or if we need a smarter mechanism.

Usage:
    python eval/measure_diversity.py [--limit N]
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env.local")
load_dotenv(_root / ".env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from dytm.nodes.spec import SPEC_SYSTEM_PROMPT
from dytm.nodes.multi_spec import (
    extract_docstring_examples,
    check_spec_example_consistency,
    score_spec,
)
from dytm.schemas.pbt import SpecOutput


def get_spec_at_temp(model_name: str, temp: float, prompt: str) -> dict | None:
    """Generate a single spec at a given temperature."""
    model = ChatOpenAI(model=model_name, temperature=temp, timeout=90, max_retries=2)
    structured = model.with_structured_output(SpecOutput)
    messages = [
        SystemMessage(content=SPEC_SYSTEM_PROMPT),
        HumanMessage(content=f"Task: {prompt}"),
    ]
    try:
        result = structured.invoke(messages)
        return result.pbt.model_dump()
    except Exception as e:
        print(f"  [WARN] temp={temp} failed: {e}")
        return None


def compare_specs(specs: list[dict]) -> dict:
    """Compare K specs and measure diversity."""
    if len(specs) < 2:
        return {"n_specs": len(specs), "diversity": "insufficient"}

    # Compare invariant test bodies (the most important part)
    all_invariant_bodies = []
    all_edge_inputs = []
    all_property_names = []

    for spec in specs:
        props = spec.get("properties", [])
        inv_bodies = set()
        edge_inputs = set()
        prop_names = set()
        for p in props:
            prop_names.add(p.get("name", ""))
            if p.get("type") == "invariant" and p.get("test_body"):
                # Normalize whitespace for comparison
                body = " ".join(p["test_body"].split())
                inv_bodies.add(body)
            if p.get("type") == "edge_case":
                edge_inputs.add(p.get("input", ""))
        all_invariant_bodies.append(inv_bodies)
        all_edge_inputs.append(edge_inputs)
        all_property_names.append(prop_names)

    # Jaccard similarity between property name sets
    name_pairs = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = all_property_names[i], all_property_names[j]
            if a | b:
                name_pairs.append(len(a & b) / len(a | b))
            else:
                name_pairs.append(1.0)

    # Jaccard similarity between invariant body sets
    inv_pairs = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = all_invariant_bodies[i], all_invariant_bodies[j]
            if a | b:
                inv_pairs.append(len(a & b) / len(a | b))
            else:
                inv_pairs.append(1.0)

    # Jaccard similarity between edge case input sets
    edge_pairs = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = all_edge_inputs[i], all_edge_inputs[j]
            if a | b:
                edge_pairs.append(len(a & b) / len(a | b))
            else:
                edge_pairs.append(1.0)

    return {
        "n_specs": len(specs),
        "avg_name_similarity": sum(name_pairs) / len(name_pairs) if name_pairs else 1.0,
        "avg_invariant_similarity": sum(inv_pairs) / len(inv_pairs) if inv_pairs else 1.0,
        "avg_edge_similarity": sum(edge_pairs) / len(edge_pairs) if edge_pairs else 1.0,
        "n_invariants_per_spec": [len(b) for b in all_invariant_bodies],
        "n_edges_per_spec": [len(e) for e in all_edge_inputs],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    # Pick specific problems that are interesting (mix of FUZZY + regression cases)
    parser.add_argument("--problems", type=str, default=None,
                        help="Comma-separated task IDs (e.g. 'HumanEval/5,HumanEval/56')")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    problems = []
    with open("eval/benchmarks/HumanEval.jsonl") as f:
        for line in f:
            problems.append(json.loads(line))

    if args.problems:
        target_ids = set(args.problems.split(","))
        problems = [p for p in problems if p["task_id"] in target_ids]
    else:
        problems = problems[:args.limit]

    temps = [0.0, 0.7, 1.0]
    print(f"Model: {args.model}")
    print(f"Temperatures: {temps}")
    print(f"Problems: {len(problems)}")
    print("=" * 70)

    all_results = []

    for p in problems:
        tid = p["task_id"]
        prompt = (
            f"Complete this Python function. The function name MUST be "
            f"'{p['entry_point']}'.\n\n{p['prompt']}"
        )
        examples = extract_docstring_examples(p["prompt"])

        print(f"\n--- {tid} (examples={len(examples)}) ---")

        specs = []
        for temp in temps:
            spec = get_spec_at_temp(args.model, temp, prompt)
            if spec:
                specs.append((temp, spec))
                consistent, matches, contradictions = \
                    check_spec_example_consistency(spec, examples)
                sc = score_spec(spec)
                n_inv = sum(1 for prop in spec["properties"]
                           if prop.get("type") == "invariant" and prop.get("test_body"))
                n_edge = sum(1 for prop in spec["properties"]
                            if prop.get("type") == "edge_case")
                print(f"  temp={temp}: score={sc:.1f}  inv={n_inv}  edge={n_edge}  "
                      f"consistent={consistent}  matches={matches}  contradictions={contradictions}")

                # Show invariant descriptions briefly
                for prop in spec["properties"]:
                    if prop.get("type") == "invariant":
                        desc = prop.get("expected", "")[:80]
                        print(f"    INV: {desc}")

        if len(specs) >= 2:
            diversity = compare_specs([s for _, s in specs])
            print(f"  DIVERSITY: names={diversity['avg_name_similarity']:.2f}  "
                  f"invariants={diversity['avg_invariant_similarity']:.2f}  "
                  f"edges={diversity['avg_edge_similarity']:.2f}")
            all_results.append(diversity)

    # Summary
    if all_results:
        print("\n" + "=" * 70)
        print("AGGREGATE DIVERSITY METRICS")
        print("=" * 70)
        avg_name = sum(r["avg_name_similarity"] for r in all_results) / len(all_results)
        avg_inv = sum(r["avg_invariant_similarity"] for r in all_results) / len(all_results)
        avg_edge = sum(r["avg_edge_similarity"] for r in all_results) / len(all_results)
        print(f"  Avg property name similarity:  {avg_name:.2f}  (1.0 = identical, 0.0 = totally different)")
        print(f"  Avg invariant body similarity: {avg_inv:.2f}")
        print(f"  Avg edge case similarity:      {avg_edge:.2f}")
        print()
        if avg_inv > 0.8:
            print("  ⚠ VERDICT: LOW DIVERSITY — Temperature sampling alone may not be enough.")
            print("    The model produces very similar invariants across temperatures.")
            print("    Consider: explicit interpretation prompting, or contrastive spec generation.")
        elif avg_inv > 0.5:
            print("  ✓ VERDICT: MODERATE DIVERSITY — Temperature sampling produces meaningful variation.")
            print("    Some invariants differ, which gives the selection heuristic something to work with.")
        else:
            print("  ✓ VERDICT: HIGH DIVERSITY — Temperature sampling produces substantially different specs.")
            print("    The selection heuristic has real choices to make.")


if __name__ == "__main__":
    main()

