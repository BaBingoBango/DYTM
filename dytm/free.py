"""DYTM Free Mode — interactive CLI demo.

A Rich-powered REPL where the user plays the Executive role, experiencing
the full DYTM verification loop firsthand.

New v2 pipeline:
    Spec (PBTs only) → Translator (CNL) → Human Approval → Coder (fresh) → Doer (tests)

Usage:
    python -m dytm.free
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from dytm.nodes.coder import make_coder_node
from dytm.nodes.doer import doer_node
from dytm.nodes.spec import make_spec_node
from dytm.nodes.translator import translator_node

console = Console()


def _show_banner() -> None:
    console.print(Panel(
        "[bold cyan]DYTM - Do You Trust Me[/bold cyan]\n"
        "[dim]A deterministic verification layer for LLM agents[/dim]\n\n"
        "[dim]v2 pipeline: Spec → CNL → You (approve/reject) → Coder → Tests[/dim]\n\n"
        "Type a coding task. Type [bold]quit[/bold] to exit.",
        title="Free Mode",
        border_style="cyan",
    ))
    console.print()


def _show_cnl(cnl_text: str) -> None:
    console.print()
    console.print(Panel(
        cnl_text,
        title="Verification Contract",
        border_style="yellow",
    ))
    console.print()


def _show_result(state: dict, elapsed: float) -> None:
    code = state.get("code", "")
    retries = state.get("retry_count", 0)
    result = state.get("test_result", "unknown")

    console.print()
    if result == "pass":
        console.print(Panel(
            Syntax(code, "python", theme="monokai", line_numbers=True),
            title="[bold green]Verified Code[/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            Syntax(code, "python", theme="monokai", line_numbers=True),
            title="[bold red]Code (failed verification)[/bold red]",
            border_style="red",
        ))
        if state.get("error_trace"):
            console.print(Panel(
                state["error_trace"][:2000],
                title="Error Trace",
                border_style="red",
            ))

    table = Table(show_header=False, border_style="dim")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Result", "[green]PASS[/green]" if result == "pass" else "[red]FAIL[/red]")
    table.add_row("Retries", str(retries))
    table.add_row("Time", f"{elapsed:.1f}s")
    console.print(table)
    console.print()


def _get_approval() -> tuple[bool, str]:
    while True:
        answer = console.input("[yellow]Approve this contract? [Y]es / [N]o / [Q]uit: [/yellow]").strip().lower()
        if answer in ("y", "yes", ""):
            return True, ""
        elif answer in ("n", "no"):
            feedback = console.input("[dim]Feedback (optional, Enter to skip): [/dim]").strip()
            return False, feedback or "Contract rejected."
        elif answer in ("q", "quit"):
            raise KeyboardInterrupt
        else:
            console.print("[dim]Please enter Y, N, or Q.[/dim]")


def _run_task(spec_fn, coder_fn, prompt: str, max_retries: int = 4) -> dict:
    """Run one task through the DYTM v2 loop with interactive approval.

    Phase A: Spec generates PBTs only → Translator renders CNL → Human approves
    Phase B: Coder generates code (fresh context) → Doer runs tests → retry if needed
    """
    state = {
        "prompt": prompt,
        "mode": "free",
        "retry_count": 0,
        "max_retries": max_retries,
        "spec_retry_count": 0,
    }

    # --- Phase A: Generate & approve the contract (no code yet!) ---
    contract_approved = False
    while not contract_approved:
        with console.status("[bold cyan]Spec is generating verification properties..."):
            spec_result = spec_fn(state)
        state.update(spec_result)

        tr_result = translator_node(state)
        state.update(tr_result)

        _show_cnl(state["cnl_text"])
        approved, feedback = _get_approval()

        if approved:
            contract_approved = True
            state["approved"] = True
        else:
            state["approved"] = False
            state["approval_feedback"] = feedback
            state["spec_retry_count"] = state.get("spec_retry_count", 0) + 1
            console.print("[yellow]Regenerating contract with your feedback...[/yellow]")

    # --- Phase B: Coder generates code (FRESH context) → Doer tests ---
    approved_pbt = state["pbt_payload"]  # freeze the approved contract

    console.print()
    console.print("[bold cyan]Contract approved! Now generating code on a fresh context...[/bold cyan]")

    for attempt in range(max_retries + 1):
        with console.status("[bold cyan]Coder is writing code against the approved contract..."):
            coder_result = coder_fn(state)
        state.update(coder_result)
        # Ensure approved PBTs stay locked
        state["pbt_payload"] = approved_pbt

        with console.status("[bold cyan]Running Hypothesis PBTs..."):
            doer_result = doer_node(state)
        state.update(doer_result)

        if state.get("test_result") == "pass":
            return state

        # Show failure
        console.print(f"[red]Attempt {attempt + 1} FAILED[/red]")
        if state.get("error_trace"):
            console.print(f"[dim]{state['error_trace'][:500]}[/dim]")

        if attempt < max_retries:
            console.print("[yellow]Retrying — regenerating code against approved contract...[/yellow]")

    return state


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env.local")
    load_dotenv(project_root / ".env")

    if not os.environ.get("OPENAI_API_KEY"):
        console.print("[red]ERROR: OPENAI_API_KEY not found in .env.local or environment.[/red]")
        sys.exit(1)

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    spec_fn = make_spec_node(model)
    coder_fn = make_coder_node(model)

    _show_banner()

    try:
        while True:
            prompt = console.input("[bold green]You:[/bold green] ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                break

            start = time.time()
            final_state = _run_task(spec_fn, coder_fn, prompt)
            elapsed = time.time() - start

            _show_result(final_state, elapsed)

    except KeyboardInterrupt:
        pass

    console.print("\n[dim]Goodbye![/dim]")


if __name__ == "__main__":
    main()
