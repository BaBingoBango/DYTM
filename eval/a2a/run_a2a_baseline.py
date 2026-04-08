"""A2A Baseline — NL-only delegation from Planner to Coder.

Condition A of the A2A experiment:
  Planner writes NL description → Coder writes code → Planner gives NL
  feedback → Coder retries → … → Planner judges final output.

No PBTs, no contracts, no structured verification.

Usage:
    python -m eval.a2a.run_a2a_baseline [--tasks TASKS_JSON] [--limit N]
           [--planner-model MODEL] [--coder-model MODEL] [--max-rounds N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env.local")
load_dotenv(_root / ".env")

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from eval.a2a.planner import baseline_feedback, judge_output_majority
from eval.a2a.schemas import A2AResult, A2ATask, A2ATaskBank

console = Console()
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Coder system prompt (baseline — receives NL description only)
# ---------------------------------------------------------------------------

BASELINE_CODER_SYSTEM_PROMPT = """\
You are an expert Python programmer. You receive a task description from \
a Planner agent and must implement the requested Python function.

Rules:
1. Return ONLY the function code — no tests, no examples, no explanations.
2. Use Python 3.10+ type hints (lowercase list, dict, etc.).
3. Do NOT import external libraries (only builtins + math + collections).
4. The function name must match what the task specifies.
5. Handle edge cases carefully — think about empty inputs, boundary \
   conditions, and error cases.
"""

BASELINE_CODER_RETRY_PROMPT = """\
Your previous implementation was reviewed by the Planner who created this \
task. They found issues with your code.

ORIGINAL TASK:
{description}

YOUR PREVIOUS CODE:
```python
{prev_code}
```

PLANNER FEEDBACK:
{feedback}

Please fix your code to address this feedback. Return ONLY the corrected \
function code.
"""


def _strip_markdown_fences(code: str) -> str:
    """Remove markdown code fences if present."""
    if "```python" in code:
        code = code.split("```python", 1)[1]
        code = code.split("```", 1)[0]
    elif "```" in code:
        code = code.split("```", 1)[1]
        code = code.split("```", 1)[0]
    return code.strip()


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def _run_one_baseline(
    task: A2ATask,
    planner_model,
    coder_model,
    max_rounds: int,
    index: int,
    total: int,
) -> A2AResult:
    """Run the baseline condition on a single task."""
    start = time.time()
    llm_calls = 0

    # -- First attempt: Coder generates code from NL description --
    messages = [
        SystemMessage(content=BASELINE_CODER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Implement this function:\n\n{task.description}\n\n"
            f"The function must be named `{task.function_name}`."
        )),
    ]

    response = coder_model.invoke(messages)
    llm_calls += 1
    code = _strip_markdown_fences(response.content)

    # -- Feedback loop --
    rounds = 1
    first_attempt_code = code

    for round_num in range(1, max_rounds + 1):
        # Planner reviews
        fb = baseline_feedback(planner_model, task, code)
        llm_calls += 1

        if fb.satisfied:
            break

        if round_num >= max_rounds:
            break

        # Coder retries with NL feedback
        retry_content = BASELINE_CODER_RETRY_PROMPT.format(
            description=task.description,
            prev_code=code,
            feedback=fb.feedback,
        )
        retry_messages = [
            SystemMessage(content=BASELINE_CODER_SYSTEM_PROMPT),
            HumanMessage(content=retry_content),
        ]

        response = coder_model.invoke(retry_messages)
        llm_calls += 1
        code = _strip_markdown_fences(response.content)
        rounds += 1

    # -- Final judgment (3x majority vote) --
    judgment, votes = judge_output_majority(planner_model, task, code, n_votes=3)
    llm_calls += 3

    # -- Check first-attempt alignment --
    first_judgment = judge_output_majority(planner_model, task, first_attempt_code, n_votes=1)
    llm_calls += 1
    first_aligned = first_judgment[0].intent_match >= 4

    elapsed = time.time() - start

    result = A2AResult(
        task_id=task.task_id,
        condition="baseline",
        planner_model=planner_model.model if hasattr(planner_model, 'model') else "unknown",
        coder_model=coder_model.model if hasattr(coder_model, 'model') else "unknown",
        final_intent_match=judgment.intent_match,
        drift_detected=judgment.drift_detected,
        drift_category=judgment.drift_category,
        drift_description=judgment.drift_description,
        rounds_to_agreement=rounds,
        first_attempt_aligned=first_aligned,
        contract_rejections=0,
        pbt_retries=0,
        contract_faithfulness=None,
        total_llm_calls=llm_calls,
        wall_clock_sec=round(elapsed, 2),
        judgment_votes=votes,
    )

    # Print progress
    status = "✓" if judgment.intent_match >= 4 else "✗"
    drift_tag = f" drift={judgment.drift_category}" if judgment.drift_detected else ""
    console.print(
        f"  [{index + 1:>3}/{total}] {task.task_id:<12s} "
        f"{'[green]' + status + '[/green]' if judgment.intent_match >= 4 else '[red]' + status + '[/red]'}  "
        f"match={judgment.intent_match}/5  rounds={rounds}{drift_tag}  "
        f"{elapsed:.1f}s"
    )

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_a2a_baseline(
    tasks_path: str,
    planner_model_name: str = "claude-sonnet-4-20250514",
    coder_model_name: str = "gpt-4o-mini",
    max_rounds: int = 4,
    limit: int | None = None,
) -> Path:
    """Run the A2A baseline condition on all tasks."""
    # Load tasks
    with open(tasks_path, encoding="utf-8") as f:
        data = json.load(f)
    bank = A2ATaskBank(**data)
    tasks = bank.tasks[:limit] if limit else bank.tasks

    # Initialize models
    planner_model = ChatAnthropic(
        model=planner_model_name, temperature=0, timeout=90, max_retries=2,
    )
    coder_model = ChatOpenAI(
        model=coder_model_name, temperature=0, timeout=90, max_retries=2,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"a2a_baseline_{timestamp}.csv"

    console.print()
    console.print(Panel(
        f"[bold cyan]A2A Baseline[/bold cyan]  |  planner={planner_model_name}  "
        f"coder={coder_model_name}  max_rounds={max_rounds}\n"
        f"tasks={len(tasks)}",
        border_style="cyan",
    ))
    console.print()

    results: list[A2AResult] = []
    for i, task in enumerate(tasks):
        try:
            result = _run_one_baseline(
                task, planner_model, coder_model, max_rounds, i, len(tasks),
            )
            results.append(result)
        except Exception as exc:
            console.print(f"  [{i + 1}/{len(tasks)}] {task.task_id} ERROR: {exc}")

    # Write CSV
    fieldnames = [
        "task_id", "condition", "planner_model", "coder_model",
        "final_intent_match", "drift_detected", "drift_category",
        "drift_description", "rounds_to_agreement", "first_attempt_aligned",
        "contract_rejections", "pbt_retries", "contract_faithfulness",
        "total_llm_calls", "wall_clock_sec", "judgment_votes",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r.model_dump())

    _print_summary(results, csv_path)
    return csv_path


def _print_summary(results: list[A2AResult], csv_path: Path) -> None:
    """Print summary statistics."""
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    total = len(results)
    high_match = sum(1 for r in results if r.final_intent_match >= 4)
    drift_count = sum(1 for r in results if r.drift_detected)
    first_aligned = sum(1 for r in results if r.first_attempt_aligned)
    avg_rounds = sum(r.rounds_to_agreement for r in results) / total
    avg_match = sum(r.final_intent_match for r in results) / total

    console.print()
    console.print(Panel(
        f"[bold]A2A Baseline Summary[/bold]\n"
        f"  High match (≥4/5): {high_match}/{total} ({high_match / total * 100:.1f}%)\n"
        f"  Drift detected:    {drift_count}/{total} ({drift_count / total * 100:.1f}%)\n"
        f"  First-attempt OK:  {first_aligned}/{total} ({first_aligned / total * 100:.1f}%)\n"
        f"  Avg intent match:  {avg_match:.2f}/5\n"
        f"  Avg rounds:        {avg_rounds:.2f}\n"
        f"  Results: {csv_path}",
        border_style="green",
    ))


def main():
    parser = argparse.ArgumentParser(description="Run A2A baseline eval")
    parser.add_argument("--tasks", type=str, required=True,
                        help="Path to task bank JSON file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--planner-model", type=str,
                        default="claude-sonnet-4-20250514")
    parser.add_argument("--coder-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-rounds", type=int, default=4,
                        help="Max Planner↔Coder interaction rounds")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red bold]ERROR: ANTHROPIC_API_KEY not set.[/red bold]")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red bold]ERROR: OPENAI_API_KEY not set.[/red bold]")
        sys.exit(1)

    run_a2a_baseline(
        tasks_path=args.tasks,
        planner_model_name=args.planner_model,
        coder_model_name=args.coder_model,
        max_rounds=args.max_rounds,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

