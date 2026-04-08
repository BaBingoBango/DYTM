"""A2A DYTM — Planner delegates to Coder via DYTM protocol.

Condition B of the A2A experiment:
  Planner writes NL description → Spec generates PBTs → Translator renders
  CNL → Planner (adversarial Executive) reviews CNL → Coder writes code
  from approved contract → Doer runs PBTs → Planner judges final output.

The Planner acts as the Executive with access to its private intent_summary.

Usage:
    python -m eval.a2a.run_a2a_dytm [--tasks TASKS_JSON] [--limit N]
           [--planner-model MODEL] [--coder-model MODEL]
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
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel

from dytm.graph import build_graph  # noqa: F401 — kept for reference
from dytm.nodes.spec import make_spec_node
from dytm.nodes.translator import translator_node
from eval.a2a.planner import (
    adversarial_executive_review,
    judge_output_majority,
)
from eval.a2a.schemas import A2AResult, A2ATask, A2ATaskBank

console = Console()
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Custom approval node — Planner-as-adversarial-Executive
# ---------------------------------------------------------------------------

def _make_planner_executive_approval(planner_model, task: A2ATask):
    """Create an approval node where the Planner acts as adversarial Executive.

    Unlike the standard approval node which uses a generic Executive LLM,
    this one gives the Planner access to its private intent_summary — the
    information asymmetry that makes this experiment meaningful.
    """

    def planner_approval_node(state: dict) -> dict:
        cnl_text = state.get("cnl_text", "(no contract)")

        decision = adversarial_executive_review(planner_model, task, cnl_text)

        result = {
            "approved": decision.approved,
            "approval_feedback": decision.feedback,
        }

        if not decision.approved:
            result["spec_retry_count"] = state.get("spec_retry_count", 0) + 1

        return result

    return planner_approval_node


# ---------------------------------------------------------------------------
# Build a per-task graph with the Planner wired as Executive
# ---------------------------------------------------------------------------

def _build_a2a_graph(
    coder_model,
    spec_model,
    planner_model,
    task: A2ATask,
):
    """Build a DYTM graph with the Planner as adversarial Executive.

    We can't use the standard build_graph() directly because the approval
    node needs per-task context (the task's intent_summary). Instead, we
    build a custom graph that swaps in the Planner approval node.
    """
    from langgraph.graph import END, StateGraph
    from dytm.nodes.coder import make_coder_node
    from dytm.nodes.doer import doer_node
    from dytm.state import DYTMState

    spec_node = make_spec_node(spec_model)
    coder_node = make_coder_node(coder_model)
    approval_node = _make_planner_executive_approval(planner_model, task)

    def _route_after_approval(state: DYTMState) -> str:
        if state.get("approved"):
            return "coder"
        spec_retries = state.get("spec_retry_count", 0)
        max_spec_retries = state.get("max_spec_retries", 2)
        if spec_retries >= max_spec_retries:
            return "coder"
        return "spec"

    def _route_after_doer(state: DYTMState) -> str:
        if state.get("test_result") == "pass":
            return END
        if state.get("retry_count", 0) >= state.get("max_retries", 4):
            return END
        return "coder"

    graph = StateGraph(DYTMState)
    graph.add_node("spec", spec_node)
    graph.add_node("translator", translator_node)
    graph.add_node("approval", approval_node)
    graph.add_node("coder", coder_node)
    graph.add_node("doer", doer_node)

    graph.set_entry_point("spec")
    graph.add_edge("spec", "translator")
    graph.add_edge("translator", "approval")
    graph.add_edge("coder", "doer")

    graph.add_conditional_edges("approval", _route_after_approval)
    graph.add_conditional_edges("doer", _route_after_doer)

    return graph.compile()


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def _run_one_dytm(
    task: A2ATask,
    planner_model,
    coder_model,
    spec_model,
    max_retries: int,
    max_spec_retries: int,
    index: int,
    total: int,
) -> A2AResult:
    """Run the DYTM condition on a single task."""
    start = time.time()

    # Build per-task graph (Planner as Executive needs task context)
    app = _build_a2a_graph(coder_model, spec_model, planner_model, task)

    dytm_prompt = (
        f"Implement this Python function. The function name MUST be "
        f"'{task.function_name}'.\n\n{task.description}"
    )

    initial_state = {
        "prompt": dytm_prompt,
        "mode": "eval",  # triggers normal eval flow
        "retry_count": 0,
        "max_retries": max_retries,
        "spec_retry_count": 0,
        "max_spec_retries": max_spec_retries,
    }

    # Track contract rejections
    contract_rejections = 0
    final_state: dict = {}

    for step in app.stream(initial_state, stream_mode="updates"):
        for node_name, state_update in step.items():
            final_state.update(state_update)
            # Count contract rejections
            if node_name == "approval" and not state_update.get("approved", True):
                contract_rejections += 1

    code = final_state.get("code", "")
    pbt_retries = final_state.get("retry_count", 0)
    pbt_passed = final_state.get("test_result") == "pass"

    # -- Contract faithfulness check --
    # Ask the Planner: does the approved contract capture your intent?
    contract_faithfulness = None
    cnl = final_state.get("cnl_text", "")
    if cnl:
        try:
            from eval.a2a.planner import judge_output
            from eval.a2a.schemas import A2AJudgment
            # Quick single-call check (not full majority vote for cost)
            cf_decision = adversarial_executive_review(planner_model, task, cnl)
            contract_faithfulness = 5 if cf_decision.approved else 2
        except Exception:
            pass

    # -- Final judgment (3x majority vote) --
    judgment, votes = judge_output_majority(planner_model, task, code, n_votes=3)

    # -- First-attempt alignment --
    # In DYTM, "first attempt" = after the first Coder pass (before any PBT retries)
    first_aligned = judgment.intent_match >= 4 and pbt_retries == 0

    # Approximate LLM calls:
    # spec + (contract_rejections * spec_retry) + approval_calls + coder + (pbt_retries * coder) + 3 judgments
    llm_calls = (
        1  # spec
        + contract_rejections  # spec retries
        + (1 + contract_rejections)  # approval calls
        + 1  # coder
        + pbt_retries  # coder retries
        + 3  # judgment votes
    )

    elapsed = time.time() - start

    rounds = 1 + contract_rejections + pbt_retries

    result = A2AResult(
        task_id=task.task_id,
        condition="dytm",
        planner_model=planner_model.model if hasattr(planner_model, 'model') else "unknown",
        coder_model=coder_model.model if hasattr(coder_model, 'model') else "unknown",
        final_intent_match=judgment.intent_match,
        drift_detected=judgment.drift_detected,
        drift_category=judgment.drift_category,
        drift_description=judgment.drift_description,
        rounds_to_agreement=rounds,
        first_attempt_aligned=first_aligned,
        contract_rejections=contract_rejections,
        pbt_retries=pbt_retries,
        contract_faithfulness=contract_faithfulness,
        total_llm_calls=llm_calls,
        wall_clock_sec=round(elapsed, 2),
        judgment_votes=votes,
    )

    # Print progress
    status = "✓" if judgment.intent_match >= 4 else "✗"
    pbt_tag = "PBT:OK" if pbt_passed else "PBT:FAIL"
    drift_tag = f" drift={judgment.drift_category}" if judgment.drift_detected else ""
    rej_tag = f" rej={contract_rejections}" if contract_rejections > 0 else ""
    console.print(
        f"  [{index + 1:>3}/{total}] {task.task_id:<12s} "
        f"{'[green]' + status + '[/green]' if judgment.intent_match >= 4 else '[red]' + status + '[/red]'}  "
        f"match={judgment.intent_match}/5  {pbt_tag}{rej_tag}{drift_tag}  "
        f"{elapsed:.1f}s"
    )

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_a2a_dytm(
    tasks_path: str,
    planner_model_name: str = "claude-sonnet-4-20250514",
    coder_model_name: str = "gpt-4o-mini",
    spec_model_name: str | None = None,
    max_retries: int = 4,
    max_spec_retries: int = 2,
    limit: int | None = None,
) -> Path:
    """Run the A2A DYTM condition on all tasks."""
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
    spec_model = ChatOpenAI(
        model=spec_model_name or coder_model_name,
        temperature=0, timeout=90, max_retries=2,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"a2a_dytm_{timestamp}.csv"

    spec_tag = spec_model_name or coder_model_name

    console.print()
    console.print(Panel(
        f"[bold cyan]A2A DYTM[/bold cyan]  |  planner={planner_model_name}  "
        f"spec={spec_tag}  coder={coder_model_name}\n"
        f"tasks={len(tasks)}  max_retries={max_retries}  "
        f"max_spec_retries={max_spec_retries}",
        border_style="cyan",
    ))
    console.print()

    results: list[A2AResult] = []
    for i, task in enumerate(tasks):
        try:
            result = _run_one_dytm(
                task, planner_model, coder_model, spec_model,
                max_retries, max_spec_retries, i, len(tasks),
            )
            results.append(result)
        except Exception as exc:
            console.print(
                f"  [{i + 1}/{len(tasks)}] {task.task_id} "
                f"[red]ERROR: {str(exc)[:200]}[/red]"
            )

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
    total_rejections = sum(r.contract_rejections for r in results)
    tasks_with_rejections = sum(1 for r in results if r.contract_rejections > 0)

    # Rubber-stamp guardrail check
    first_attempt_approvals = sum(1 for r in results if r.contract_rejections == 0)
    approval_rate = first_attempt_approvals / total * 100

    guardrail_warning = ""
    if approval_rate > 90:
        guardrail_warning = (
            "\n  [bold red]⚠️  GUARDRAIL VIOLATION: First-attempt approval "
            f"rate is {approval_rate:.0f}% (>90%).[/bold red]\n"
            "  [red]The Executive prompt is too weak. Results are NOT valid. "
            "Tighten the adversarial prompt before reporting.[/red]"
        )
    elif approval_rate > 80:
        guardrail_warning = (
            f"\n  [yellow]⚠ First-attempt approval rate is {approval_rate:.0f}% "
            "(borderline). Consider tightening the Executive prompt.[/yellow]"
        )

    console.print()
    console.print(Panel(
        f"[bold]A2A DYTM Summary[/bold]\n"
        f"  High match (≥4/5):    {high_match}/{total} ({high_match / total * 100:.1f}%)\n"
        f"  Drift detected:       {drift_count}/{total} ({drift_count / total * 100:.1f}%)\n"
        f"  First-attempt OK:     {first_aligned}/{total} ({first_aligned / total * 100:.1f}%)\n"
        f"  Avg intent match:     {avg_match:.2f}/5\n"
        f"  Avg rounds:           {avg_rounds:.2f}\n"
        f"  Contract rejections:  {total_rejections} across {tasks_with_rejections} tasks\n"
        f"  First-attempt approve: {approval_rate:.0f}% (target: 50-80%)"
        f"{guardrail_warning}\n"
        f"  Results: {csv_path}",
        border_style="green",
    ))


def main():
    parser = argparse.ArgumentParser(description="Run A2A DYTM eval")
    parser.add_argument("--tasks", type=str, required=True,
                        help="Path to task bank JSON file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--planner-model", type=str,
                        default="claude-sonnet-4-20250514")
    parser.add_argument("--coder-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--spec-model", type=str, default=None,
                        help="Model for Spec node (default: same as coder)")
    parser.add_argument("--max-retries", type=int, default=4,
                        help="Max PBT-driven code retries")
    parser.add_argument("--max-spec-retries", type=int, default=2,
                        help="Max contract revision retries")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red bold]ERROR: ANTHROPIC_API_KEY not set.[/red bold]")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red bold]ERROR: OPENAI_API_KEY not set.[/red bold]")
        sys.exit(1)

    run_a2a_dytm(
        tasks_path=args.tasks,
        planner_model_name=args.planner_model,
        coder_model_name=args.coder_model,
        spec_model_name=args.spec_model,
        max_retries=args.max_retries,
        max_spec_retries=args.max_spec_retries,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()


