"""A2A Hybrid — DYTM pipeline + Planner NL feedback rounds.

Condition C of the A2A experiment:
  Phase 1 (DYTM): Planner → Spec → Contract → Executive → Coder → PBT loop
  Phase 2 (NL):   Planner reviews code → NL feedback → Coder revises
                   (revised code also runs PBTs to keep structural verification)

The Coder gets TWO feedback channels:
  - Deterministic PBT error traces (structural bugs)
  - Rich NL feedback from the Planner (intent nuances)

Usage:
    python -m eval.a2a.run_a2a_hybrid [--tasks TASKS_JSON] [--limit N]
           [--planner-model MODEL] [--coder-model MODEL]
           [--max-retries N] [--max-spec-retries N] [--max-nl-rounds N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
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

from dytm.nodes.spec import make_spec_node
from dytm.nodes.translator import translator_node
from dytm.state import DYTMState
from eval.a2a.planner import (
    adversarial_executive_review,
    baseline_feedback,
    judge_output_majority,
)
from eval.a2a.schemas import A2AResult, A2ATask, A2ATaskBank

console = Console()
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Maximum PBT test execution timeout (seconds)
PBT_TIMEOUT = 15


# ---------------------------------------------------------------------------
# Standalone PBT runner (outside the LangGraph graph)
# ---------------------------------------------------------------------------

def _run_pbt_standalone(code: str, pbt_payload: dict) -> tuple[bool, str]:
    """Compile and run PBTs on code, returning (passed, error_trace).

    Uses the Doer's compilation logic but runs outside the graph.
    """
    from dytm.nodes.doer import _compile_test_file
    from dytm.security import scan_code

    safe, reason = scan_code(code)
    if not safe:
        return False, f"Security scan failed: {reason}"

    try:
        test_source = _compile_test_file(code, pbt_payload)
    except Exception as exc:
        return False, f"Test compilation failed: {exc}"

    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_dytm_generated.py"
        test_file.write_text(test_source, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "--tb=long", "-q"],
                capture_output=True,
                timeout=PBT_TIMEOUT,
                text=True,
                cwd=tmpdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return False, f"PBT execution timed out after {PBT_TIMEOUT}s"

    if result.returncode == 0:
        return True, ""

    combined = ((result.stdout or "") + "\n" + (result.stderr or ""))[:3000]
    return False, combined.strip()


# ---------------------------------------------------------------------------
# Hybrid Coder NL-retry prompt
# ---------------------------------------------------------------------------

HYBRID_CODER_SYSTEM = """\
You are an expert Python programmer working with a verification contract \
AND natural language feedback from the task author.

You have TWO sources of guidance:
1. An APPROVED PBT contract specifying structural properties your code must satisfy.
2. Natural language feedback from the Planner who created the task, explaining \
   what's wrong with your current code in terms of INTENT.

Prioritise the Planner's NL feedback for understanding WHAT to build. \
Use the PBT contract to verify your implementation is structurally correct.

Rules:
- Return ONLY the function code — no tests, no examples, no explanations.
- Use Python 3.10+ type hints (lowercase list, dict, etc.).
- Do NOT import external libraries (only builtins + math + collections).
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


def _format_contract(pbt: dict) -> str:
    """Render PBT payload into human-readable contract text."""
    func_name = pbt["function_name"]
    lines = [f"Function name: {func_name}", "", "APPROVED VERIFICATION CONTRACT:"]
    for i, prop in enumerate(pbt["properties"], 1):
        ptype = prop["type"]
        expected = prop.get("expected", "")
        if ptype == "edge_case":
            inp = prop.get("input", "?")
            behavior = prop.get("expected_behavior", expected)
            lines.append(f"  {i}. EDGE CASE: {func_name}({inp}) must {behavior}")
        elif ptype == "output_type":
            lines.append(f"  {i}. RETURN TYPE: must return {expected}")
        elif ptype == "invariant":
            lines.append(f"  {i}. INVARIANT: {expected}")
        elif ptype == "membership":
            lines.append(f"  {i}. MEMBERSHIP: {expected}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Planner Executive approval node (same as DYTM runner)
# ---------------------------------------------------------------------------

def _make_planner_executive_approval(planner_model, task: A2ATask):
    def planner_approval_node(state: dict) -> dict:
        cnl_text = state.get("cnl_text", "(no contract)")
        previous_feedback = state.get("approval_feedback")
        attempt = state.get("spec_retry_count", 0)

        decision = adversarial_executive_review(
            planner_model, task, cnl_text,
            previous_feedback=previous_feedback,
            attempt=attempt,
        )

        result = {
            "approved": decision.approved,
            "approval_feedback": decision.feedback,
        }
        if not decision.approved:
            result["spec_retry_count"] = state.get("spec_retry_count", 0) + 1

        return result

    return planner_approval_node


# ---------------------------------------------------------------------------
# Phase 1: Build the DYTM graph (reused from run_a2a_dytm)
# ---------------------------------------------------------------------------

def _build_a2a_graph(coder_model, spec_model, planner_model, task: A2ATask):
    from langgraph.graph import END, StateGraph
    from dytm.nodes.coder import make_coder_node
    from dytm.nodes.doer import doer_node

    spec_node = make_spec_node(spec_model)
    coder_node = make_coder_node(coder_model)
    approval_node = _make_planner_executive_approval(planner_model, task)

    def _route_after_approval(state: DYTMState) -> str:
        if state.get("approved"):
            return "coder"
        if state.get("spec_retry_count", 0) >= state.get("max_spec_retries", 2):
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

def _run_one_hybrid(
    task: A2ATask,
    planner_model,
    coder_model,
    spec_model,
    max_retries: int,
    max_spec_retries: int,
    max_nl_rounds: int,
    index: int,
    total: int,
) -> A2AResult:
    """Run the hybrid condition on a single task."""
    start = time.time()

    # ===== PHASE 1: Standard DYTM pipeline =====
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

    contract_rejections = 0
    final_state: dict = {}

    for step in app.stream(initial_state, stream_mode="updates"):
        for node_name, state_update in step.items():
            final_state.update(state_update)
            if node_name == "approval" and not state_update.get("approved", True):
                contract_rejections += 1

    code = final_state.get("code", "")
    pbt_retries = final_state.get("retry_count", 0)
    pbt_passed = final_state.get("test_result") == "pass"
    pbt_payload = final_state.get("pbt_payload", {})

    # ===== PHASE 2: Planner NL feedback rounds =====
    nl_rounds = 0
    contract_text = _format_contract(pbt_payload) if pbt_payload else ""

    for nl_round in range(max_nl_rounds):
        # Planner reviews the code
        fb = baseline_feedback(planner_model, task, code)

        if fb.satisfied:
            break

        nl_rounds += 1

        # Build Coder retry prompt with NL feedback + contract
        pbt_context = ""
        if contract_text:
            pbt_context = f"\n\n{contract_text}\n"

        retry_content = (
            f"ORIGINAL TASK:\n{task.description}\n\n"
            f"FUNCTION NAME: {task.function_name}\n"
            f"{pbt_context}\n"
            f"YOUR PREVIOUS CODE:\n```python\n{code}\n```\n\n"
            f"PLANNER FEEDBACK (from the person who wrote this task):\n"
            f"{fb.feedback}\n\n"
            f"Please fix your code to address this feedback. "
            f"Return ONLY the corrected function code."
        )

        retry_messages = [
            SystemMessage(content=HYBRID_CODER_SYSTEM),
            HumanMessage(content=retry_content),
        ]

        response = coder_model.invoke(retry_messages)
        code = _strip_markdown_fences(response.content)

        # Run PBTs on revised code (structural verification still applies)
        if pbt_payload:
            pbt_ok, pbt_err = _run_pbt_standalone(code, pbt_payload)
            if not pbt_ok and pbt_err:
                # Give the Coder one more shot with PBT error trace too
                pbt_fix_content = (
                    f"ORIGINAL TASK:\n{task.description}\n\n"
                    f"FUNCTION NAME: {task.function_name}\n"
                    f"{pbt_context}\n"
                    f"YOUR CODE:\n```python\n{code}\n```\n\n"
                    f"This code addressed the Planner's feedback but FAILED "
                    f"the verification tests:\n{pbt_err[:2000]}\n\n"
                    f"Fix the code to pass both the Planner's requirements "
                    f"AND the verification contract. Return ONLY the function code."
                )
                pbt_fix_messages = [
                    SystemMessage(content=HYBRID_CODER_SYSTEM),
                    HumanMessage(content=pbt_fix_content),
                ]
                response = coder_model.invoke(pbt_fix_messages)
                code = _strip_markdown_fences(response.content)
                pbt_retries += 1

    # ===== FINAL JUDGMENT =====
    judgment, votes = judge_output_majority(planner_model, task, code, n_votes=3)

    first_aligned = judgment.intent_match >= 4 and pbt_retries == 0 and nl_rounds == 0

    # Approximate LLM calls
    llm_calls = (
        1                           # spec
        + contract_rejections       # spec retries
        + (1 + contract_rejections) # approval calls
        + 1                         # coder (initial)
        + pbt_retries               # coder PBT retries
        + nl_rounds                 # planner NL reviews
        + nl_rounds                 # coder NL retries
        + 3                         # judgment votes
    )

    elapsed = time.time() - start
    rounds = 1 + contract_rejections + pbt_retries + nl_rounds

    result = A2AResult(
        task_id=task.task_id,
        condition="hybrid",
        planner_model=planner_model.model if hasattr(planner_model, "model") else "unknown",
        coder_model=coder_model.model if hasattr(coder_model, "model") else "unknown",
        final_intent_match=judgment.intent_match,
        drift_detected=judgment.drift_detected,
        drift_category=judgment.drift_category,
        drift_description=judgment.drift_description,
        rounds_to_agreement=rounds,
        first_attempt_aligned=first_aligned,
        contract_rejections=contract_rejections,
        pbt_retries=pbt_retries,
        contract_faithfulness=None,
        total_llm_calls=llm_calls,
        wall_clock_sec=round(elapsed, 2),
        judgment_votes=votes,
    )

    # Print progress
    status = "✓" if judgment.intent_match >= 4 else "✗"
    pbt_tag = "PBT:OK" if pbt_passed else "PBT:FAIL"
    drift_tag = f" drift={judgment.drift_category}" if judgment.drift_detected else ""
    rej_tag = f" rej={contract_rejections}" if contract_rejections > 0 else ""
    nl_tag = f" nl={nl_rounds}" if nl_rounds > 0 else ""
    console.print(
        f"  [{index + 1:>3}/{total}] {task.task_id:<12s} "
        f"{'[green]' + status + '[/green]' if judgment.intent_match >= 4 else '[red]' + status + '[/red]'}  "
        f"match={judgment.intent_match}/5  {pbt_tag}{rej_tag}{nl_tag}{drift_tag}  "
        f"{elapsed:.1f}s"
    )

    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_a2a_hybrid(
    tasks_path: str,
    planner_model_name: str = "claude-sonnet-4-20250514",
    coder_model_name: str = "gpt-4o-mini",
    spec_model_name: str | None = None,
    max_retries: int = 4,
    max_spec_retries: int = 3,
    max_nl_rounds: int = 2,
    limit: int | None = None,
) -> Path:
    """Run the A2A hybrid condition on all tasks."""
    with open(tasks_path, encoding="utf-8") as f:
        data = json.load(f)
    bank = A2ATaskBank(**data)
    tasks = bank.tasks[:limit] if limit else bank.tasks

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
    csv_path = RESULTS_DIR / f"a2a_hybrid_{timestamp}.csv"

    spec_tag = spec_model_name or coder_model_name

    console.print()
    console.print(Panel(
        f"[bold magenta]A2A Hybrid[/bold magenta]  |  planner={planner_model_name}  "
        f"spec={spec_tag}  coder={coder_model_name}\n"
        f"tasks={len(tasks)}  max_retries={max_retries}  "
        f"max_spec_retries={max_spec_retries}  max_nl_rounds={max_nl_rounds}\n"
        f"Phase 1: DYTM pipeline (Spec→Contract→Executive→Coder→PBT loop)\n"
        f"Phase 2: Planner NL feedback rounds (up to {max_nl_rounds})",
        border_style="magenta",
    ))
    console.print()

    results: list[A2AResult] = []
    for i, task in enumerate(tasks):
        try:
            result = _run_one_hybrid(
                task, planner_model, coder_model, spec_model,
                max_retries, max_spec_retries, max_nl_rounds, i, len(tasks),
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
    tasks_with_nl = sum(1 for r in results if r.rounds_to_agreement > r.contract_rejections + r.pbt_retries + 1)

    console.print()
    console.print(Panel(
        f"[bold]A2A Hybrid Summary[/bold]\n"
        f"  High match (≥4/5):    {high_match}/{total} ({high_match / total * 100:.1f}%)\n"
        f"  Drift detected:       {drift_count}/{total} ({drift_count / total * 100:.1f}%)\n"
        f"  First-attempt OK:     {first_aligned}/{total} ({first_aligned / total * 100:.1f}%)\n"
        f"  Avg intent match:     {avg_match:.2f}/5\n"
        f"  Avg rounds:           {avg_rounds:.2f}\n"
        f"  Contract rejections:  {total_rejections}\n"
        f"  Tasks w/ NL feedback: {tasks_with_nl}\n"
        f"  Results: {csv_path}",
        border_style="green",
    ))


def main():
    parser = argparse.ArgumentParser(description="Run A2A hybrid eval")
    parser.add_argument("--tasks", type=str, required=True,
                        help="Path to task bank JSON file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--planner-model", type=str,
                        default="claude-sonnet-4-20250514")
    parser.add_argument("--coder-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--spec-model", type=str, default=None,
                        help="Model for Spec node (default: same as coder)")
    parser.add_argument("--max-retries", type=int, default=4,
                        help="Max PBT-driven code retries (Phase 1)")
    parser.add_argument("--max-spec-retries", type=int, default=3,
                        help="Max contract revision retries (Phase 1)")
    parser.add_argument("--max-nl-rounds", type=int, default=2,
                        help="Max Planner NL feedback rounds (Phase 2)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red bold]ERROR: ANTHROPIC_API_KEY not set.[/red bold]")
        sys.exit(1)
    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red bold]ERROR: OPENAI_API_KEY not set.[/red bold]")
        sys.exit(1)

    run_a2a_hybrid(
        tasks_path=args.tasks,
        planner_model_name=args.planner_model,
        coder_model_name=args.coder_model,
        spec_model_name=args.spec_model,
        max_retries=args.max_retries,
        max_spec_retries=args.max_spec_retries,
        max_nl_rounds=args.max_nl_rounds,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

