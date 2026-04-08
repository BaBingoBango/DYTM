"""Metrics computation and comparison for A2A experiment.

Loads baseline and DYTM result CSVs, computes all metrics from the A2A
Eval Plan (§3), and prints a formatted comparison table.

Usage:
    python -m eval.a2a.metrics <baseline.csv> <dytm.csv>
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _load_results(csv_path: str) -> list[dict]:
    """Load result CSV into list of dicts."""
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("true", "1", "yes")


def _parse_int(val: str, default: int = 0) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _parse_float(val: str, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def compute_metrics(results: list[dict], condition: str) -> dict:
    """Compute all A2A metrics for a single condition."""
    n = len(results)
    if n == 0:
        return {}

    intent_matches = [_parse_int(r["final_intent_match"]) for r in results]
    drift_detected = [_parse_bool(r["drift_detected"]) for r in results]
    rounds = [_parse_int(r["rounds_to_agreement"]) for r in results]
    first_aligned = [_parse_bool(r["first_attempt_aligned"]) for r in results]
    llm_calls = [_parse_int(r["total_llm_calls"]) for r in results]
    wall_clock = [_parse_float(r["wall_clock_sec"]) for r in results]

    # Primary metrics
    high_match = sum(1 for m in intent_matches if m >= 4)
    redo_count = sum(1 for m in intent_matches if m <= 2)
    drift_count = sum(drift_detected)

    metrics = {
        "condition": condition,
        "n_tasks": n,
        # Primary
        "high_match_rate": high_match / n,
        "redo_rate": redo_count / n,
        "drift_rate": drift_count / n,
        "avg_rounds": sum(rounds) / n,
        "avg_llm_calls": sum(llm_calls) / n,
        # Secondary
        "avg_intent_match": sum(intent_matches) / n,
        "first_attempt_aligned_rate": sum(first_aligned) / n,
        # Cost
        "avg_wall_clock": sum(wall_clock) / n,
        "total_wall_clock": sum(wall_clock),
    }

    # DYTM-specific
    if condition == "dytm":
        contract_rej = [_parse_int(r.get("contract_rejections", "0")) for r in results]
        pbt_retries = [_parse_int(r.get("pbt_retries", "0")) for r in results]

        tasks_with_rejections = sum(1 for c in contract_rej if c > 0)
        total_rejections = sum(contract_rej)

        metrics["misunderstanding_detection_rate"] = tasks_with_rejections / n
        metrics["total_contract_rejections"] = total_rejections
        metrics["avg_pbt_retries"] = sum(pbt_retries) / n

        # Rubber-stamp guardrail
        first_approve_rate = sum(1 for c in contract_rej if c == 0) / n
        metrics["first_attempt_approve_rate"] = first_approve_rate

    # Drift category breakdown
    drift_cats: dict[str, int] = {}
    for r in results:
        if _parse_bool(r["drift_detected"]):
            cat = r.get("drift_category", "other")
            drift_cats[cat] = drift_cats.get(cat, 0) + 1
    metrics["drift_categories"] = drift_cats

    # Judgment reliability (inter-vote agreement)
    try:
        vote_lists = [json.loads(r.get("judgment_votes", "[]")) for r in results]
        agreements = []
        for votes in vote_lists:
            if len(votes) >= 2:
                # Max vote count / total votes
                from collections import Counter
                c = Counter(votes)
                agreements.append(c.most_common(1)[0][1] / len(votes))
        if agreements:
            metrics["judgment_agreement"] = sum(agreements) / len(agreements)
    except Exception:
        pass

    return metrics


def print_comparison(baseline_metrics: dict, dytm_metrics: dict) -> None:
    """Print a formatted comparison of both conditions."""
    console.print()

    # ---- Primary Metrics ----
    table = Table(
        title="A2A Experiment — Primary Metrics",
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Metric", style="bold", min_width=30)
    table.add_column("Baseline (NL)", justify="right", min_width=15)
    table.add_column("DYTM", justify="right", min_width=15)
    table.add_column("Δ", justify="right", min_width=10)

    def _row(name: str, key: str, fmt: str = ".1%", higher_better: bool = True):
        b = baseline_metrics.get(key, 0)
        d = dytm_metrics.get(key, 0)
        delta = d - b
        if fmt == ".1%":
            b_str = f"{b * 100:.1f}%"
            d_str = f"{d * 100:.1f}%"
            delta_str = f"{delta * 100:+.1f}pp"
        elif fmt == ".2f":
            b_str = f"{b:.2f}"
            d_str = f"{d:.2f}"
            delta_str = f"{delta:+.2f}"
        elif fmt == ".1f":
            b_str = f"{b:.1f}"
            d_str = f"{d:.1f}"
            delta_str = f"{delta:+.1f}"
        else:
            b_str = str(b)
            d_str = str(d)
            delta_str = str(delta)

        # Color delta
        good = (delta > 0 and higher_better) or (delta < 0 and not higher_better)
        color = "green" if good else "red" if delta != 0 else "dim"
        table.add_row(name, b_str, d_str, f"[{color}]{delta_str}[/{color}]")

    _row("High match (≥4/5)", "high_match_rate")
    _row("Redo rate (≤2/5)", "redo_rate", higher_better=False)
    _row("Drift detected", "drift_rate", higher_better=False)
    _row("Avg intent match", "avg_intent_match", ".2f")
    _row("First-attempt aligned", "first_attempt_aligned_rate")
    _row("Avg rounds", "avg_rounds", ".2f", higher_better=False)
    _row("Avg LLM calls", "avg_llm_calls", ".1f", higher_better=False)
    _row("Avg wall-clock (s)", "avg_wall_clock", ".1f", higher_better=False)

    console.print(table)

    # ---- DYTM-specific ----
    if "misunderstanding_detection_rate" in dytm_metrics:
        console.print()
        dytm_table = Table(
            title="DYTM-Specific Metrics", border_style="magenta", show_lines=True,
        )
        dytm_table.add_column("Metric", style="bold", min_width=30)
        dytm_table.add_column("Value", justify="right", min_width=15)

        mdr = dytm_metrics["misunderstanding_detection_rate"]
        dytm_table.add_row(
            "Misunderstanding detection rate",
            f"{mdr * 100:.1f}%",
        )
        dytm_table.add_row(
            "Total contract rejections",
            str(dytm_metrics.get("total_contract_rejections", 0)),
        )
        dytm_table.add_row(
            "Avg PBT retries",
            f"{dytm_metrics.get('avg_pbt_retries', 0):.2f}",
        )

        fap = dytm_metrics.get("first_attempt_approve_rate", 0)
        fap_color = "red" if fap > 0.9 else "yellow" if fap > 0.8 else "green"
        dytm_table.add_row(
            "First-attempt approve rate",
            f"[{fap_color}]{fap * 100:.0f}%[/{fap_color}]"
            + (" ⚠️ GUARDRAIL" if fap > 0.9 else ""),
        )

        console.print(dytm_table)

    # ---- Drift category breakdown ----
    console.print()
    drift_table = Table(
        title="Drift Category Breakdown", border_style="yellow", show_lines=True,
    )
    drift_table.add_column("Category", style="bold")
    drift_table.add_column("Baseline", justify="right")
    drift_table.add_column("DYTM", justify="right")

    all_cats = set(
        list(baseline_metrics.get("drift_categories", {}).keys())
        + list(dytm_metrics.get("drift_categories", {}).keys())
    )
    for cat in sorted(all_cats):
        b = baseline_metrics.get("drift_categories", {}).get(cat, 0)
        d = dytm_metrics.get("drift_categories", {}).get(cat, 0)
        drift_table.add_row(cat, str(b), str(d))

    console.print(drift_table)

    # ---- Judgment reliability ----
    b_agree = baseline_metrics.get("judgment_agreement")
    d_agree = dytm_metrics.get("judgment_agreement")
    if b_agree or d_agree:
        console.print()
        console.print(
            f"  Judgment agreement: baseline={b_agree:.0%}  dytm={d_agree:.0%}"
            if b_agree and d_agree else ""
        )

    # ---- Verdict ----
    console.print()
    b_redo = baseline_metrics.get("redo_rate", 0)
    d_redo = dytm_metrics.get("redo_rate", 0)
    mdr = dytm_metrics.get("misunderstanding_detection_rate", 0)

    if d_redo < b_redo * 0.5 and mdr > 0.15:
        verdict = "[bold green]STRONG EVIDENCE[/bold green] — DYTM significantly reduces intent drift."
    elif d_redo < b_redo and mdr > 0.1:
        verdict = "[bold yellow]MODERATE EVIDENCE[/bold yellow] — DYTM shows promise but effect size is modest."
    elif d_redo >= b_redo:
        verdict = "[bold red]NO EVIDENCE[/bold red] — DYTM did not reduce intent drift on this task set."
    else:
        verdict = "[dim]INCONCLUSIVE — insufficient data or mixed signals.[/dim]"

    console.print(Panel(f"  Verdict: {verdict}", border_style="bold"))
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Compare A2A baseline vs DYTM")
    parser.add_argument("baseline_csv", type=str, help="Path to baseline results CSV")
    parser.add_argument("dytm_csv", type=str, help="Path to DYTM results CSV")
    args = parser.parse_args()

    baseline_results = _load_results(args.baseline_csv)
    dytm_results = _load_results(args.dytm_csv)

    baseline_metrics = compute_metrics(baseline_results, "baseline")
    dytm_metrics = compute_metrics(dytm_results, "dytm")

    print_comparison(baseline_metrics, dytm_metrics)


if __name__ == "__main__":
    main()


