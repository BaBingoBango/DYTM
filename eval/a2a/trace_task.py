"""Verbose single-task trace for A2A DYTM debugging.

Runs one task through the full DYTM pipeline, printing every step:
  Spec output → CNL → Executive review → (loop) → Coder → Doer → Judgment
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env.local")
load_dotenv(_root / ".env")

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from eval.a2a.planner import adversarial_executive_review, judge_output_majority
from eval.a2a.schemas import A2ATask, A2ATaskBank

console = Console(width=120)


def trace_one_task(task: A2ATask, planner_model, coder_model, spec_model,
                   max_retries: int = 4, max_spec_retries: int = 3):
    """Run one task with full verbose output."""

    console.print(Panel(
        f"[bold]Task: {task.task_id}[/bold]\n\n"
        f"[cyan]Description:[/cyan]\n{task.description}\n\n"
        f"[yellow]Private Intent:[/yellow]\n{task.intent_summary}\n\n"
        f"[dim]Category: {task.category}  Ambiguity: {task.ambiguity_level}[/dim]",
        title="TASK", border_style="blue",
    ))

    # Build the graph
    from eval.a2a.run_a2a_dytm import _build_a2a_graph
    app = _build_a2a_graph(coder_model, spec_model, planner_model, task)

    dytm_prompt = (
        f"Implement this Python function. The function name MUST be "
        f"'{task.function_name}'.\n\n{task.description}"
    )

    initial_state = {
        "prompt": dytm_prompt,
        "mode": "eval",
        "retry_count": 0,
        "max_retries": max_retries,
        "spec_retry_count": 0,
        "max_spec_retries": max_spec_retries,
    }

    step_num = 0
    final_state = {}
    contract_rejections = 0

    for step in app.stream(initial_state, stream_mode="updates"):
        for node_name, state_update in step.items():
            step_num += 1
            final_state.update(state_update)

            if node_name == "spec":
                pbt = state_update.get("pbt_payload", {})
                props = pbt.get("properties", [])
                prop_summary = "\n".join(
                    f"  [{p.get('type', '?')}] {p.get('name', '?')}: {p.get('expected', '')[:100]}"
                    for p in props
                )
                console.print(Panel(
                    f"Function: {pbt.get('function_name', '?')}\n"
                    f"Properties ({len(props)}):\n{prop_summary}",
                    title=f"Step {step_num}: SPEC (attempt {final_state.get('spec_retry_count', 0)})",
                    border_style="cyan",
                ))

            elif node_name == "translator":
                cnl = state_update.get("cnl_text", "")
                # Show first 800 chars
                display = cnl[:800] + ("..." if len(cnl) > 800 else "")
                console.print(Panel(
                    display,
                    title=f"Step {step_num}: TRANSLATOR (CNL Contract)",
                    border_style="green",
                ))

            elif node_name == "approval":
                approved = state_update.get("approved", False)
                feedback = state_update.get("approval_feedback", "")
                if not approved:
                    contract_rejections += 1
                status = "[bold green]✓ APPROVED[/bold green]" if approved else "[bold red]✗ REJECTED[/bold red]"
                console.print(Panel(
                    f"{status}\n\n"
                    f"Feedback:\n{textwrap.fill(feedback or '(none)', 110)}",
                    title=f"Step {step_num}: EXECUTIVE REVIEW (rejection #{contract_rejections})" if not approved
                          else f"Step {step_num}: EXECUTIVE REVIEW",
                    border_style="green" if approved else "red",
                ))

            elif node_name == "coder":
                code = state_update.get("code", "")
                # Show first 600 chars
                display = code[:600] + ("..." if len(code) > 600 else "")
                retry = final_state.get("retry_count", 0)
                console.print(Panel(
                    f"```python\n{display}\n```",
                    title=f"Step {step_num}: CODER (retry {retry})",
                    border_style="yellow",
                ))

            elif node_name == "doer":
                test_result = state_update.get("test_result", "?")
                error = state_update.get("error_trace", "")
                counterex = state_update.get("counterexample", "")
                status = "[bold green]PASS[/bold green]" if test_result == "pass" else "[bold red]FAIL[/bold red]"
                detail = ""
                if counterex:
                    detail += f"\nCounterexample: {counterex[:200]}"
                if error:
                    detail += f"\nError: {error[:300]}"
                console.print(Panel(
                    f"Result: {status}{detail}",
                    title=f"Step {step_num}: DOER (PBT Execution)",
                    border_style="green" if test_result == "pass" else "red",
                ))

    # Final judgment
    console.print()
    console.print("[bold]Running final judgment (3x majority vote)...[/bold]")
    code = final_state.get("code", "")
    judgment, votes = judge_output_majority(planner_model, task, code, n_votes=3)

    console.print(Panel(
        f"Intent Match: [bold]{judgment.intent_match}/5[/bold]  Votes: {votes}\n"
        f"Drift Detected: {judgment.drift_detected}\n"
        f"Drift Category: {judgment.drift_category}\n"
        f"Drift Description:\n{textwrap.fill(judgment.drift_description or '(none)', 110)}\n\n"
        f"Contract Rejections: {contract_rejections}\n"
        f"PBT Retries: {final_state.get('retry_count', 0)}\n"
        f"PBT Passed: {final_state.get('test_result') == 'pass'}",
        title="FINAL JUDGMENT",
        border_style="bold magenta",
    ))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Trace a single A2A task verbosely")
    parser.add_argument("--tasks", type=str, default="eval/a2a/results/task_bank.json")
    parser.add_argument("--task-id", type=str, default="A2A/001",
                        help="Which task to trace (e.g. A2A/001)")
    parser.add_argument("--planner-model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--coder-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-spec-retries", type=int, default=3)
    args = parser.parse_args()

    with open(args.tasks) as f:
        bank = A2ATaskBank(**json.load(f))

    task = next((t for t in bank.tasks if t.task_id == args.task_id), None)
    if not task:
        print(f"Task {args.task_id} not found!")
        sys.exit(1)

    planner = ChatAnthropic(model=args.planner_model, temperature=0, timeout=90, max_retries=2)
    coder = ChatOpenAI(model=args.coder_model, temperature=0, timeout=90, max_retries=2)
    spec = ChatOpenAI(model=args.coder_model, temperature=0, timeout=90, max_retries=2)

    trace_one_task(task, planner, coder, spec,
                   max_retries=args.max_retries,
                   max_spec_retries=args.max_spec_retries)


if __name__ == "__main__":
    main()

