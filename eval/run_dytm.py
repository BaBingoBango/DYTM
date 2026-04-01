"""DYTM eval — full verification loop on HumanEval benchmarks.

For each problem, runs the DYTM LangGraph pipeline (Operator -> Translator
-> Approval (auto) -> Doer), then validates the final code against the
benchmark's canonical test suite.

Usage:
    python -m eval.run_dytm [--limit N] [--model MODEL]
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env.local")
load_dotenv(_root / ".env")

from langchain_openai import ChatOpenAI

from dytm.graph import build_graph
from eval.loader import build_canonical_test_code, load_humaneval

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _run_canonical_tests(code: str, problem: dict, timeout: int = 30) -> bool:
    """Run the benchmark's canonical tests against generated code."""
    test_harness = build_canonical_test_code(problem)
    full_code = code + "\n\n" + test_harness

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_canonical.py"
        test_file.write_text(full_code, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                timeout=timeout,
                text=True,
                cwd=tmpdir,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False


def run_dytm(
    model_name: str = "gpt-4o-mini",
    limit: int | None = None,
    offset: int = 0,
    max_retries: int = 3,
) -> Path:
    """Run the DYTM eval and write results to CSV."""
    model = ChatOpenAI(model=model_name, temperature=0)
    app = build_graph(model)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"dytm_{model_name}_{timestamp}.csv"

    fieldnames = [
        "problem_id", "model", "mode", "passed", "pbt_passed",
        "retries", "wall_clock_sec", "failure_reason",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        problems = list(load_humaneval(limit=limit, offset=offset))
        total = len(problems)

        for i, problem in enumerate(problems):
            task_id = problem["task_id"]
            prompt = problem["prompt"]
            entry_point = problem["entry_point"]
            print(f"[{i+1}/{total}] {task_id} ... ", end="", flush=True)

            start = time.time()
            failure_reason = None
            pbt_passed = False
            retries = 0

            try:
                # Build the DYTM prompt from the benchmark problem
                dytm_prompt = (
                    f"Complete this Python function. The function name MUST be "
                    f"'{entry_point}'.\n\n{prompt}"
                )

                initial_state = {
                    "prompt": dytm_prompt,
                    "mode": "eval",
                    "retry_count": 0,
                    "max_retries": max_retries,
                }

                final_state = app.invoke(initial_state)

                code = final_state.get("code", "")
                pbt_passed = final_state.get("test_result") == "pass"
                retries = final_state.get("retry_count", 0)

                # Run canonical tests (ground truth)
                passed = _run_canonical_tests(code, problem)
                if not passed:
                    failure_reason = "canonical_tests_failed"
                    if not pbt_passed:
                        failure_reason = "pbt_and_canonical_failed"

            except Exception as exc:
                passed = False
                failure_reason = str(exc)[:200]

            elapsed = time.time() - start
            status = "PASS" if passed else "FAIL"
            pbt_tag = " (PBT:OK)" if pbt_passed else " (PBT:FAIL)"
            retry_tag = f" retries={retries}" if retries > 0 else ""
            print(f"{status}{pbt_tag}{retry_tag} ({elapsed:.1f}s)")

            writer.writerow({
                "problem_id": task_id,
                "model": model_name,
                "mode": "dytm",
                "passed": passed,
                "pbt_passed": pbt_passed,
                "retries": retries,
                "wall_clock_sec": round(elapsed, 2),
                "failure_reason": failure_reason,
            })
            f.flush()

    _print_summary(csv_path)
    return csv_path


def _print_summary(csv_path: Path) -> None:
    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    total = len(reader)
    passed = sum(1 for r in reader if r["passed"] == "True")
    pbt_passed = sum(1 for r in reader if r["pbt_passed"] == "True")
    avg_retries = sum(int(r["retries"]) for r in reader) / max(total, 1)
    print(f"\n{'='*50}")
    print(f"DYTM Results: {passed}/{total} = {passed/total*100:.1f}% pass@1")
    print(f"PBT pass rate: {pbt_passed}/{total} = {pbt_passed/total*100:.1f}%")
    print(f"Avg retries: {avg_retries:.1f}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Run DYTM eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    run_dytm(
        model_name=args.model,
        limit=args.limit,
        offset=args.offset,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()

