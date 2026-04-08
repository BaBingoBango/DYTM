"""Generate the A2A task bank using the Planner model.

Creates 50 complex coding tasks with hidden intent, suitable for
measuring intent drift between Planner and Coder agents.

Usage:
    python -m eval.a2a.generate_tasks [--planner-model MODEL] [--n-tasks N]
           [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent.parent
load_dotenv(_root / ".env.local")
load_dotenv(_root / ".env")

from langchain_anthropic import ChatAnthropic
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from eval.a2a.planner import generate_task_bank

console = Console()
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Generate A2A task bank")
    parser.add_argument("--planner-model", type=str,
                        default="claude-sonnet-4-20250514")
    parser.add_argument("--n-tasks", type=int, default=50)
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path (default: eval/a2a/results/task_bank.json)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red bold]ERROR: ANTHROPIC_API_KEY not set.[/red bold]")
        sys.exit(1)

    planner = ChatAnthropic(
        model=args.planner_model, temperature=0.7, timeout=120, max_retries=2,
    )

    console.print()
    console.print(Panel(
        f"[bold cyan]A2A Task Generation[/bold cyan]\n"
        f"  Model: {args.planner_model}\n"
        f"  Tasks: {args.n_tasks}",
        border_style="cyan",
    ))
    console.print()

    bank = generate_task_bank(planner, n_tasks=args.n_tasks)

    # Save
    output_path = args.output or str(RESULTS_DIR / "task_bank.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(bank.model_dump(), f, indent=2, ensure_ascii=False)

    # Print summary
    table = Table(title=f"Generated {len(bank.tasks)} Tasks", border_style="cyan")
    table.add_column("ID", style="bold")
    table.add_column("Category")
    table.add_column("Ambiguity")
    table.add_column("Function")
    table.add_column("Description (truncated)")

    categories: dict[str, int] = {}
    ambiguities: dict[str, int] = {}

    for task in bank.tasks:
        categories[task.category] = categories.get(task.category, 0) + 1
        ambiguities[task.ambiguity_level] = ambiguities.get(task.ambiguity_level, 0) + 1
        table.add_row(
            task.task_id,
            task.category,
            task.ambiguity_level,
            task.function_name,
            task.description[:80] + "…" if len(task.description) > 80 else task.description,
        )

    console.print(table)
    console.print()
    console.print(f"  Categories: {categories}")
    console.print(f"  Ambiguity:  {ambiguities}")
    console.print(f"  Saved to:   {output_path}")
    console.print()


if __name__ == "__main__":
    main()

