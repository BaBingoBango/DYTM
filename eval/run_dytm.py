"""DYTM eval — full verification loop on HumanEval benchmarks.

For each problem, runs the DYTM LangGraph pipeline (Operator -> Translator
-> Approval (auto) -> Doer), then validates the final code against the
benchmark's canonical test suite.

Now with **real-time observability**: streams graph execution node-by-node
so you can see exactly what the system is doing at each step.

Usage:
    python -m eval.run_dytm [--limit N] [--model MODEL] [--verbose] [--workers N]
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent
load_dotenv(_root / ".env.local")
load_dotenv(_root / ".env")

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from langchain_openai import ChatOpenAI

from dytm.graph import build_graph
from eval.loader import build_canonical_test_code, load_humaneval

console = Console()
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Thread lock for console + CSV writes in parallel mode
_print_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Colour / icon helpers
# ---------------------------------------------------------------------------

_NODE_STYLE = {
    "spec":       ("📋", "bold cyan"),
    "translator": ("📝", "bold yellow"),
    "approval":   ("👔", "bold green"),
    "coder":      ("💻", "bold blue"),
    "doer":       ("🧪", "bold magenta"),
}


def _node_tag(name: str) -> str:
    icon, style = _NODE_STYLE.get(name, ("▸", "dim"))
    return f"[{style}]{icon} {name.upper()}[/{style}]"


# ---------------------------------------------------------------------------
# Canonical test runner (unchanged)
# ---------------------------------------------------------------------------

def _run_canonical_tests(code: str, problem: dict, timeout: int = 30) -> tuple[bool, str]:
    """Run the benchmark's canonical tests against generated code.

    Returns (passed, error_detail).
    """
    test_harness = build_canonical_test_code(problem)
    # Add typing imports so code using List[float] etc. works
    preamble = "from typing import *\nfrom collections import *\nimport math\n\n"
    full_code = preamble + code + "\n\n" + test_harness

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
            if result.returncode == 0:
                return True, ""
            detail = (result.stdout or "") + "\n" + (result.stderr or "")
            return False, detail.strip()[-1000:]
        except subprocess.TimeoutExpired:
            return False, "Canonical tests timed out"
        except Exception as exc:
            return False, str(exc)[:500]


# ---------------------------------------------------------------------------
# Verbose helpers — show what each node produced
# ---------------------------------------------------------------------------

def _show_spec_output(state: dict, elapsed_ms: int, verbose: bool) -> None:
    """Log Spec node results."""
    pbt = state.get("pbt_payload", {})
    func_name = pbt.get("function_name", "?")
    n_props = len(pbt.get("properties", []))
    prop_types = [p.get("type", "?") for p in pbt.get("properties", [])]
    type_summary = ", ".join(f"{t}×{prop_types.count(t)}" for t in dict.fromkeys(prop_types))

    # Multi-spec info
    ms_candidates = state.get("_multi_spec_candidates")
    ms_consistent = state.get("_multi_spec_consistent")
    ms_mode = state.get("_multi_spec_mode", "")
    ms_tag = ""
    if ms_candidates is not None:
        ms_diff = state.get("_multi_spec_difficulty", "?")
        ms_axes = state.get("_multi_spec_ambiguities", 0)
        ms_tag = (
            f"  [dim yellow]multi-spec({ms_mode}, {ms_diff}, {ms_axes}axes): "
            f"{ms_consistent}/{ms_candidates} consistent[/dim yellow]"
        )

    console.print(
        f"  {_node_tag('spec')} → fn=[bold]{func_name}[/bold]  "
        f"props={n_props} ({type_summary}){ms_tag}  "
        f"[dim]{elapsed_ms}ms[/dim]"
    )

    if verbose:
        # Show property names
        for p in pbt.get("properties", []):
            ptype = p.get("type", "?")
            name = p.get("name", "?")
            expected = p.get("expected", "")[:60]
            console.print(f"    [dim]• {ptype}: {name} — {expected}[/dim]")


def _show_coder_output(state: dict, elapsed_ms: int, verbose: bool) -> None:
    """Log Coder node results."""
    code = state.get("code", "")
    console.print(
        f"  {_node_tag('coder')} → code={len(code.splitlines())}L  "
        f"[dim]{elapsed_ms}ms[/dim]"
    )

    if verbose:
        preview = "\n".join(code.splitlines()[:8])
        if len(code.splitlines()) > 8:
            preview += "\n    # ... (truncated)"
        console.print(Panel(
            Syntax(preview, "python", theme="monokai"),
            title="[dim]Generated Code Preview[/dim]",
            border_style="dim blue",
            expand=False,
            padding=(0, 1),
        ))


def _show_operator_output(state: dict, elapsed_ms: int, verbose: bool) -> None:
    """Log Operator node results (legacy)."""
    code = state.get("code", "")
    pbt = state.get("pbt_payload", {})
    func_name = pbt.get("function_name", "?")
    n_props = len(pbt.get("properties", []))
    prop_types = [p.get("type", "?") for p in pbt.get("properties", [])]
    type_summary = ", ".join(f"{t}×{prop_types.count(t)}" for t in dict.fromkeys(prop_types))

    console.print(
        f"  {_node_tag('operator')} → fn=[bold]{func_name}[/bold]  "
        f"props={n_props} ({type_summary})  "
        f"code={len(code.splitlines())}L  "
        f"[dim]{elapsed_ms}ms[/dim]"
    )

    if verbose:
        # Show first few lines of code
        preview = "\n".join(code.splitlines()[:8])
        if len(code.splitlines()) > 8:
            preview += "\n    # ... (truncated)"
        console.print(Panel(
            Syntax(preview, "python", theme="monokai"),
            title="[dim]Generated Code Preview[/dim]",
            border_style="dim cyan",
            expand=False,
            padding=(0, 1),
        ))


def _show_translator_output(state: dict, elapsed_ms: int, verbose: bool) -> None:
    """Log Translator node results."""
    cnl = state.get("cnl_text", "")
    n_lines = len(cnl.strip().splitlines())
    console.print(
        f"  {_node_tag('translator')} → CNL contract rendered ({n_lines} lines)  "
        f"[dim]{elapsed_ms}ms[/dim]"
    )

    if verbose:
        console.print(Panel(
            cnl.strip()[:600],
            title="[dim]CNL Contract[/dim]",
            border_style="dim yellow",
            expand=False,
            padding=(0, 1),
        ))


def _show_approval_output(state: dict, elapsed_ms: int, verbose: bool) -> None:
    """Log Approval node results."""
    approved = state.get("approved", False)
    feedback = state.get("approval_feedback", "")

    if approved:
        tag = "[green]APPROVED[/green]"
    else:
        tag = "[red]REJECTED[/red]"

    spec_retry = state.get("spec_retry_count", 0)
    retry_tag = f"  spec_retry={spec_retry}" if spec_retry > 0 else ""

    console.print(
        f"  {_node_tag('approval')} → {tag}{retry_tag}  [dim]{elapsed_ms}ms[/dim]"
    )

    if verbose and feedback:
        # Show first few lines of Executive reasoning
        short_feedback = feedback[:400]
        if len(feedback) > 400:
            short_feedback += "..."
        console.print(Panel(
            short_feedback,
            title="[dim]Executive Reasoning[/dim]",
            border_style="dim green" if approved else "dim red",
            expand=False,
            padding=(0, 1),
        ))


def _show_doer_output(state: dict, elapsed_ms: int, verbose: bool) -> None:
    """Log Doer node results."""
    test_result = state.get("test_result", "?")
    security = state.get("security_passed", False)
    retry = state.get("retry_count", 0)
    edge_only = state.get("_edge_only_fail", False)
    duplicate = state.get("_duplicate_code", False)

    if test_result == "pass" and duplicate:
        result_tag = "[bold green]PASS ✓[/bold green] [dim yellow](duplicate code → skip re-test)[/dim yellow]"
    elif test_result == "pass" and edge_only:
        result_tag = "[bold green]PASS ✓[/bold green] [dim yellow](edge-only fail → soft pass)[/dim yellow]"
    elif test_result == "pass":
        result_tag = "[bold green]PASS ✓[/bold green]"
    elif test_result == "fail" and edge_only:
        result_tag = "[bold red]FAIL ✗[/bold red] [dim yellow](edge-only — invariants OK)[/dim yellow]"
    elif test_result == "fail":
        result_tag = "[bold red]FAIL ✗[/bold red]"
    else:
        result_tag = f"[bold yellow]{test_result}[/bold yellow]"

    sec_tag = "[green]✓[/green]" if security else "[red]✗[/red]"
    console.print(
        f"  {_node_tag('doer')} → {result_tag}  "
        f"security={sec_tag}  retry_count={retry}  "
        f"[dim]{elapsed_ms}ms[/dim]"
    )

    if verbose and state.get("error_trace"):
        trace = state["error_trace"][:800]
        console.print(Panel(
            trace,
            title="[dim]Error Trace[/dim]",
            border_style="dim red",
            expand=False,
            padding=(0, 1),
        ))
    if verbose and state.get("counterexample"):
        console.print(f"    [dim red]Counterexample: {state['counterexample'][:200]}[/dim red]")

    if verbose and state.get("_test_source"):
        test_src = state["_test_source"]
        # Show the test file so we can debug compilation/import errors
        preview_lines = test_src.splitlines()[:40]
        if len(test_src.splitlines()) > 40:
            preview_lines.append("    # ... (truncated)")
        console.print(Panel(
            Syntax("\n".join(preview_lines), "python", theme="monokai", line_numbers=True),
            title="[dim]Generated Test File[/dim]",
            border_style="dim magenta",
            expand=False,
            padding=(0, 1),
        ))


# Dispatch table
_NODE_LOGGERS = {
    "spec": _show_spec_output,
    "operator": _show_operator_output,
    "translator": _show_translator_output,
    "approval": _show_approval_output,
    "coder": _show_coder_output,
    "doer": _show_doer_output,
}


# ---------------------------------------------------------------------------
# Single-problem runner (extracted for parallel use)
# ---------------------------------------------------------------------------

def _run_one_problem(
    problem: dict,
    app,
    max_retries: int,
    max_spec_retries: int,
    problem_timeout: int,
    verbose: bool,
    parallel: bool,
    index: int,
    total: int,
) -> dict:
    """Run DYTM on a single problem and return a result dict.

    In serial mode (parallel=False), prints rich output as it goes.
    In parallel mode (parallel=True), buffers output and prints at the end.
    """
    task_id = problem["task_id"]
    entry_point = problem["entry_point"]
    prompt = problem["prompt"]

    start = time.time()
    failure_reason = None
    pbt_passed = False
    retries = 0
    final_state: dict = {}
    passed = False

    try:
        dytm_prompt = (
            f"Complete this Python function. The function name MUST be "
            f"'{entry_point}'.\n\n{prompt}"
        )

        initial_state = {
            "prompt": dytm_prompt,
            "mode": "eval",
            "retry_count": 0,
            "max_retries": max_retries,
            "spec_retry_count": 0,
            "max_spec_retries": max_spec_retries,
        }

        if not parallel:
            # Serial mode — use SIGALRM timeout + rich streaming
            console.rule(
                f"[bold]{task_id}[/bold]  ({index+1}/{total})  fn={entry_point}",
                style="blue",
            )

            def _timeout_handler(signum, frame):
                raise TimeoutError(f"Problem timed out after {problem_timeout}s")
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(problem_timeout)

            try:
                node_start = time.time()
                for step in app.stream(initial_state, stream_mode="updates"):
                    for node_name, state_update in step.items():
                        node_elapsed_ms = int((time.time() - node_start) * 1000)
                        final_state.update(state_update)
                        logger = _NODE_LOGGERS.get(node_name)
                        if logger:
                            logger(final_state, node_elapsed_ms, verbose)
                        else:
                            console.print(f"  [dim]▸ {node_name} ({node_elapsed_ms}ms)[/dim]")
                        node_start = time.time()
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Parallel mode — no SIGALRM, no streaming, just invoke
            for step in app.stream(initial_state, stream_mode="updates"):
                for node_name, state_update in step.items():
                    final_state.update(state_update)

        code = final_state.get("code", "")
        pbt_passed = final_state.get("test_result") == "pass"
        retries = final_state.get("retry_count", 0)

        # Canonical tests
        passed, canon_detail = _run_canonical_tests(code, problem)

        if not parallel:
            canon_tag = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            console.print(f"  [bold]📊 CANONICAL[/bold] → {canon_tag}")
            if verbose and not passed and canon_detail:
                console.print(Panel(
                    canon_detail[:1000],
                    title="[dim]Canonical Test Error[/dim]",
                    border_style="dim red",
                    expand=False,
                    padding=(0, 1),
                ))

        if not passed:
            failure_reason = "canonical_tests_failed"
            if not pbt_passed:
                failure_reason = "pbt_and_canonical_failed"

    except Exception as exc:
        passed = False
        failure_reason = str(exc)[:200]
        if not parallel:
            console.print(f"  [bold red]💥 EXCEPTION:[/bold red] {failure_reason}")

    elapsed = time.time() - start

    result = {
        "problem_id": task_id,
        "model": "",  # filled by caller
        "spec_model": "",  # filled by caller
        "mode": "dytm",
        "passed": passed,
        "pbt_passed": pbt_passed,
        "retries": retries,
        "wall_clock_sec": round(elapsed, 2),
        "failure_reason": failure_reason,
    }

    if parallel:
        pbt_tag = "PBT:OK" if pbt_passed else "PBT:FAIL"
        status = "PASS ✓" if passed else "FAIL ✗"
        with _print_lock:
            console.print(
                f"  [{index+1:>3}/{total}] {task_id:<25s} "
                f"{'[green]' + status + '[/green]' if passed else '[red]' + status + '[/red]'}  "
                f"{pbt_tag}  retries={retries}  {elapsed:.1f}s"
            )
    else:
        pbt_tag = "[green]PBT:OK[/green]" if pbt_passed else "[red]PBT:FAIL[/red]"
        result_tag = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
        running_pct = ""  # can't compute running totals here in extracted fn
        console.print(
            f"  [bold]→ {result_tag}  {pbt_tag}  "
            f"retries={retries}  {elapsed:.1f}s[/bold]"
        )
        console.print()

    return result


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def run_dytm(
    model_name: str = "gpt-4o-mini",
    executive_model_name: str | None = None,
    spec_model_name: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    max_retries: int = 4,
    max_spec_retries: int = 2,
    problem_timeout: int = 180,
    verbose: bool = False,
    workers: int = 1,
    multi_spec_k: int = 1,
) -> Path:
    """Run the DYTM eval and write results to CSV."""
    model = ChatOpenAI(model=model_name, temperature=0, timeout=90, max_retries=2)

    executive_model = None
    if executive_model_name:
        executive_model = ChatOpenAI(
            model=executive_model_name, temperature=0, timeout=60, max_retries=2,
        )

    spec_model = None
    if spec_model_name:
        spec_model = ChatOpenAI(
            model=spec_model_name, temperature=0, timeout=90, max_retries=2,
        )

    app = build_graph(
        model,
        executive_model=executive_model,
        spec_model=spec_model,
        multi_spec_k=multi_spec_k,
    )

    exec_tag = executive_model_name or "auto-approve"
    spec_tag = spec_model_name or model_name
    ms_tag = f"  multi_spec_k={multi_spec_k}" if multi_spec_k > 1 else ""

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Include spec model in filename when it differs from coder model
    if multi_spec_k > 1:
        csv_path = RESULTS_DIR / f"dytm_multispec-{multi_spec_k}_{model_name}_{timestamp}.csv"
    elif spec_model_name and spec_model_name != model_name:
        csv_path = RESULTS_DIR / f"dytm_spec-{spec_model_name}_coder-{model_name}_{timestamp}.csv"
    else:
        csv_path = RESULTS_DIR / f"dytm_{model_name}_{timestamp}.csv"

    fieldnames = [
        "problem_id", "model", "spec_model", "mode", "passed", "pbt_passed",
        "retries", "wall_clock_sec", "failure_reason",
    ]

    problems = list(load_humaneval(limit=limit, offset=offset))
    total = len(problems)

    parallel = workers > 1
    if parallel and verbose:
        console.print("[yellow]⚠ Verbose mode disabled for parallel execution.[/yellow]")
        verbose = False

    console.print()
    console.print(Panel(
        f"[bold cyan]DYTM Eval (v2)[/bold cyan]  |  spec={spec_tag}  "
        f"coder={model_name}  executive={exec_tag}  workers={workers}{ms_tag}\n"
        f"problems={total}  max_retries={max_retries}  "
        f"max_spec_retries={max_spec_retries}  "
        f"timeout={problem_timeout}s  verbose={'on' if verbose else 'off'}",
        border_style="cyan",
    ))
    console.print()

    results: list[dict] = []

    if parallel:
        # --- Parallel execution ---
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_one_problem,
                    problem, app, max_retries, max_spec_retries,
                    problem_timeout, verbose, True, i, total,
                ): (i, problem)
                for i, problem in enumerate(problems)
            }
            for future in as_completed(futures):
                result = future.result()
                result["model"] = model_name
                result["spec_model"] = spec_tag
                results.append(result)

        # Sort by problem_id for consistent CSV ordering
        results.sort(key=lambda r: r["problem_id"])

    else:
        # --- Serial execution (original behavior) ---
        for i, problem in enumerate(problems):
            result = _run_one_problem(
                problem, app, max_retries, max_spec_retries,
                problem_timeout, verbose, False, i, total,
            )
            result["model"] = model_name
            result["spec_model"] = spec_tag
            results.append(result)

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    _print_summary(csv_path)
    return csv_path


def _print_summary(csv_path: Path) -> None:
    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    total = len(reader)
    if total == 0:
        console.print("[yellow]No results recorded.[/yellow]")
        return

    passed = sum(1 for r in reader if r["passed"] == "True")
    pbt_passed = sum(1 for r in reader if r["pbt_passed"] == "True")
    avg_retries = sum(int(r["retries"]) for r in reader) / total
    avg_time = sum(float(r["wall_clock_sec"]) for r in reader) / total
    total_time = sum(float(r["wall_clock_sec"]) for r in reader)

    table = Table(title="DYTM Eval Summary", border_style="cyan", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Canonical pass@1", f"{passed}/{total} = {passed/total*100:.1f}%")
    table.add_row("PBT pass rate", f"{pbt_passed}/{total} = {pbt_passed/total*100:.1f}%")
    table.add_row("Avg retries", f"{avg_retries:.2f}")
    table.add_row("Avg time/problem", f"{avg_time:.1f}s")
    table.add_row("Total time", f"{total_time:.0f}s ({total_time/60:.1f}m)")
    table.add_row("Results file", str(csv_path))

    # Failure breakdown
    failure_reasons: dict[str, int] = {}
    for r in reader:
        reason = r.get("failure_reason", "")
        if reason:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
    if failure_reasons:
        breakdown = "  ".join(f"{k}: {v}" for k, v in sorted(failure_reasons.items(), key=lambda x: -x[1])[:5])
        table.add_row("Top failures", breakdown)

    console.print()
    console.print(table)
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Run DYTM eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--max-spec-retries", type=int, default=2,
                        help="Max times the Executive can reject and Spec revises")
    parser.add_argument("--executive-model", type=str, default=None,
                        help="Model for Executive LLM approval (e.g. gpt-4o-mini). "
                             "If not set, contracts are auto-approved.")
    parser.add_argument("--spec-model", type=str, default=None,
                        help="Model for the Spec node only (e.g. gpt-4o). "
                             "If not set, uses --model for both Spec and Coder. "
                             "Use this for cross-model experiments.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show generated code, CNL contracts, and error traces")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of parallel workers (default: 1 = serial)")
    parser.add_argument("--multi-spec", type=int, default=1, metavar="K",
                        help="Number of spec interpretations to sample (default: 1 = single spec). "
                             "Use K=3 for multi-interpretation mode with temperature diversity.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red bold]ERROR: OPENAI_API_KEY not set.[/red bold]")
        sys.exit(1)

    run_dytm(
        model_name=args.model,
        executive_model_name=args.executive_model,
        spec_model_name=args.spec_model,
        limit=args.limit,
        offset=args.offset,
        max_retries=args.max_retries,
        max_spec_retries=args.max_spec_retries,
        verbose=args.verbose,
        workers=args.workers,
        multi_spec_k=args.multi_spec,
    )


if __name__ == "__main__":
    main()

