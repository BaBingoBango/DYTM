"""Benchmark loader for HumanEval (and future MBPP).

Yields ``(task_id, prompt, entry_point, canonical_tests)`` tuples from
JSONL benchmark files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

BENCHMARKS_DIR = Path(__file__).resolve().parent / "benchmarks"


def load_humaneval(
    limit: int | None = None,
    offset: int = 0,
) -> Iterator[dict]:
    """Load HumanEval problems from the vendored JSONL file.

    Args:
        limit: Max number of problems to yield (None = all 164).
        offset: Skip the first *offset* problems.

    Yields:
        Dicts with keys: ``task_id``, ``prompt``, ``entry_point``,
        ``canonical_solution``, ``test``.
    """
    path = BENCHMARKS_DIR / "HumanEval.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"HumanEval.jsonl not found at {path}. "
            "Download it first — see Implementation Plan Phase 4."
        )

    count = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if limit is not None and count >= limit:
                break
            yield json.loads(line)
            count += 1


def build_canonical_test_code(problem: dict) -> str:
    """Build a runnable test file from a HumanEval problem.

    Combines the prompt (function signature), canonical solution, and
    the benchmark's test harness into a single executable Python string.

    For baseline eval: we paste the LLM's generated code instead of
    canonical_solution.

    Args:
        problem: A single HumanEval problem dict.

    Returns:
        A string of Python code that, when executed, runs the canonical
        ``check()`` function against the provided solution.
    """
    test_code = problem["test"]
    entry_point = problem["entry_point"]

    # The test harness calls check(candidate) — we need to wire it up
    return (
        f"{test_code}\n\n"
        f"check({entry_point})\n"
    )

