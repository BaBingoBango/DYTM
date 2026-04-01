"""Baseline eval — zero-shot code generation without DYTM verification.

For each HumanEval problem, calls the LLM once (zero-shot), then runs
the generated code against the benchmark's canonical test suite.
Logs results to CSV.

Usage:
    python -m eval.run_baseline [--limit N] [--model MODEL]
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

# Load env before importing langchain
_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env.local")
load_dotenv(_root / ".env")

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from eval.loader import build_canonical_test_code, load_humaneval

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BASELINE_SYSTEM_PROMPT = """\
You are an expert Python programmer. Complete the given function.
Return ONLY the complete function implementation, nothing else.
Do not include example usage, tests, or if __name__ blocks.
"""


def _run_canonical_tests(code: str, problem: dict, timeout: int = 30) -> bool:
    """Run the benchmark's canonical tests against generated code.

    Returns True if all tests pass, False otherwise.
    """
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
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False


def run_baseline(
    model_name: str = "gpt-4o-mini",
    limit: int | None = None,
    offset: int = 0,
) -> Path:
    """Run the baseline eval and write results to CSV.

    Returns the path to the results CSV.
    """
    model = ChatOpenAI(model=model_name, temperature=0)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"baseline_{model_name}_{timestamp}.csv"

    fieldnames = [
        "problem_id", "model", "mode", "passed",
        "wall_clock_sec", "failure_reason",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        problems = list(load_humaneval(limit=limit, offset=offset))
        total = len(problems)

        for i, problem in enumerate(problems):
            task_id = problem["task_id"]
            prompt = problem["prompt"]
            print(f"[{i+1}/{total}] {task_id} ... ", end="", flush=True)

            start = time.time()
            failure_reason = None

            try:
                messages = [
                    SystemMessage(content=BASELINE_SYSTEM_PROMPT),
                    HumanMessage(content=f"Complete this function:\n\n{prompt}"),
                ]
                response = model.invoke(messages)
                generated_code = response.content

                # Strip markdown fences if present
                if "```python" in generated_code:
                    generated_code = generated_code.split("```python", 1)[1]
                    generated_code = generated_code.split("```", 1)[0]
                elif "```" in generated_code:
                    generated_code = generated_code.split("```", 1)[1]
                    generated_code = generated_code.split("```", 1)[0]

                passed = _run_canonical_tests(generated_code, problem)
                if not passed:
                    failure_reason = "canonical_tests_failed"

            except Exception as exc:
                passed = False
                failure_reason = str(exc)[:200]

            elapsed = time.time() - start
            status = "PASS" if passed else "FAIL"
            print(f"{status} ({elapsed:.1f}s)")

            writer.writerow({
                "problem_id": task_id,
                "model": model_name,
                "mode": "baseline",
                "passed": passed,
                "wall_clock_sec": round(elapsed, 2),
                "failure_reason": failure_reason,
            })
            f.flush()

    # Print summary
    _print_summary(csv_path)
    return csv_path


def _print_summary(csv_path: Path) -> None:
    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    total = len(reader)
    passed = sum(1 for r in reader if r["passed"] == "True")
    print(f"\n{'='*50}")
    print(f"Baseline Results: {passed}/{total} = {passed/total*100:.1f}% pass@1")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Run baseline eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    run_baseline(model_name=args.model, limit=args.limit, offset=args.offset)


if __name__ == "__main__":
    main()

