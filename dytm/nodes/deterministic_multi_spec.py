"""Deterministic Multi-Spec — uses regex taxonomy instead of LLM for ambiguity detection.

This is the "deterministic distribution" variant of multi_spec.py.  Instead of
asking an LLM to identify ambiguities (which itself is probabilistic and adds
cost), it uses the regex-based deterministic_taxonomy scanner to identify
ambiguity axes, then generates one spec variant per interpretation combination.

Theoretical basis:
    For CODING TASK DESCRIPTIONS, the ambiguity space is BOUNDED.  Each task
    can only be misinterpreted along a finite set of axes (boundary conditions,
    empty handling, operator precedence, etc.), and each axis has 2-3 possible
    readings.  This module enumerates those readings deterministically and
    generates contrastive specs for each.

Pipeline:
    1. deterministic_taxonomy.scan_prompt() → list of ambiguity axes (no LLM)
    2. For each axis combination (up to K), build contrastive spec prompt
    3. Generate K specs using the Spec LLM with different interpretation guidance
    4. Filter by example consistency, select by restrictiveness score
"""

from __future__ import annotations

import itertools
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from dytm.nodes.spec import SPEC_SYSTEM_PROMPT
from dytm.schemas.pbt import SpecOutput
from dytm.state import DYTMState
from eval.deterministic_taxonomy import (
    DeterministicAnalysis,
    InterpretationAxis,
    scan_prompt,
)
from dytm.nodes.multi_spec import (
    check_spec_example_consistency,
    extract_docstring_examples,
    score_spec,
    _select_best_spec,
    _single_spec_with_feedback,
    _fallback_pbt,
)


def _build_deterministic_contrastive_prompt(
    task_prompt: str,
    axes: list[InterpretationAxis],
    interpretation_indices: tuple[int, ...],
) -> str:
    """Build a spec prompt steered by deterministic axis interpretations.

    Args:
        task_prompt: The raw HumanEval prompt.
        axes: Detected ambiguity axes from deterministic taxonomy.
        interpretation_indices: For each axis, which interpretation index to use.

    Returns:
        A prompt string with explicit interpretation guidance.
    """
    if not axes:
        return f"Task: {task_prompt}"

    guidance_lines = []
    for axis, interp_idx in zip(axes, interpretation_indices):
        chosen = axis.interpretations[min(interp_idx, len(axis.interpretations) - 1)]
        guidance_lines.append(
            f"- For \"{axis.name}\" ({axis.description}): "
            f"use this interpretation: \"{chosen}\""
        )

    guidance = "\n".join(guidance_lines)

    return (
        f"Task: {task_prompt}\n\n"
        f"INTERPRETATION GUIDANCE (deterministically identified ambiguities):\n"
        f"{guidance}\n\n"
        f"Generate a PBT specification consistent with the above interpretations. "
        f"Your invariant test_body code MUST implement the specific interpretation "
        f"described above."
    )


def _enumerate_interpretation_combos(
    axes: list[InterpretationAxis],
    max_k: int,
) -> list[tuple[int, ...]]:
    """Enumerate interpretation combinations, prioritising high-confidence axes.

    Always includes the "most likely" combo (index 0 for each axis) as the
    first entry.  Then adds alternatives ordered by axis confidence descending.

    Args:
        axes: Detected ambiguity axes.
        max_k: Maximum number of combinations to return.

    Returns:
        List of tuples, each tuple is one interpretation index per axis.
    """
    if not axes:
        return [()]

    # Sort axes by confidence descending for priority
    n = len(axes)

    # Default combo: interpretation 0 for all axes (most likely)
    default = tuple(0 for _ in axes)
    combos = [default]

    if max_k <= 1:
        return combos

    # Generate all possible combos
    ranges = [range(len(a.interpretations)) for a in axes]
    all_combos = list(itertools.product(*ranges))

    # Remove the default (already added)
    all_combos = [c for c in all_combos if c != default]

    # Sort by: prefer combos that differ on only ONE high-confidence axis
    # (single-axis flips are more informative than multi-axis flips)
    def _sort_key(combo: tuple[int, ...]) -> tuple[int, float]:
        n_diffs = sum(1 for a, b in zip(default, combo) if a != b)
        # Weight by confidence of the axes that differ
        conf_sum = sum(
            axes[i].confidence for i in range(n)
            if combo[i] != default[i]
        )
        return (n_diffs, -conf_sum)  # fewer diffs first, higher confidence first

    all_combos.sort(key=_sort_key)

    combos.extend(all_combos[:max_k - 1])
    return combos


def make_deterministic_multi_spec_node(
    model: BaseChatModel,
    k: int = 3,
    min_confidence: float = 0.4,
):
    """Create a multi-spec node using DETERMINISTIC ambiguity detection.

    Uses regex-based deterministic_taxonomy.scan_prompt() instead of an LLM
    Ambiguity Analyst.  This is:
    - Reproducible (same input → same ambiguity axes every time)
    - Free (no LLM call for analysis)
    - Fast (regex matching, not API call)
    - Theoretically grounded (finite enumeration of coding task ambiguities)

    Falls back to single-spec if no ambiguities detected.

    Args:
        model: LLM for spec generation (still uses LLM for specs, just not
               for ambiguity detection).
        k: Max number of interpretation variants to generate.
        min_confidence: Minimum confidence threshold for an axis to be included.

    Returns:
        A node function ``(DYTMState) -> dict`` suitable for LangGraph.
    """
    spec_structured = model.with_structured_output(SpecOutput)

    def deterministic_multi_spec_node(state: DYTMState) -> dict:
        prompt = state["prompt"]
        spec_retry = state.get("spec_retry_count", 0)
        approval_feedback = state.get("approval_feedback")

        # On retries (Executive rejection), fall back to single-spec
        if spec_retry > 0 and approval_feedback:
            return _single_spec_with_feedback(
                spec_structured, prompt, spec_retry, approval_feedback,
            )

        examples = extract_docstring_examples(prompt)

        # --- Phase 1: DETERMINISTIC ambiguity scan (no LLM) ---
        analysis = scan_prompt(prompt)

        # Filter axes by confidence threshold
        strong_axes = [
            a for a in analysis.axes if a.confidence >= min_confidence
        ]

        if not strong_axes:
            # No meaningful ambiguities — generate single spec
            messages = [
                SystemMessage(content=SPEC_SYSTEM_PROMPT),
                HumanMessage(content=f"Task: {prompt}"),
            ]
            try:
                result: SpecOutput = spec_structured.invoke(messages)
                return {
                    "pbt_payload": result.pbt.model_dump(),
                    "_multi_spec_candidates": 1,
                    "_multi_spec_consistent": 1,
                    "_multi_spec_mode": "deterministic_single",
                    "_multi_spec_difficulty": "unambiguous",
                    "_multi_spec_ambiguities": 0,
                }
            except Exception:
                return _fallback_pbt(prompt)

        # --- Phase 2: Enumerate interpretation combinations ---
        combos = _enumerate_interpretation_combos(strong_axes, max_k=k)

        # --- Phase 3: Generate one spec per combination ---
        candidates: list[tuple[dict, float, bool, int, int, int]] = []

        for variant_idx, combo in enumerate(combos):
            try:
                contrastive_prompt = _build_deterministic_contrastive_prompt(
                    prompt, strong_axes, combo,
                )
                messages = [
                    SystemMessage(content=SPEC_SYSTEM_PROMPT),
                    HumanMessage(content=contrastive_prompt),
                ]
                result: SpecOutput = spec_structured.invoke(messages)
                pbt = result.pbt.model_dump()

                consistent, matches, contradictions = \
                    check_spec_example_consistency(pbt, examples)
                restriction_score = score_spec(pbt)

                candidates.append((
                    pbt, restriction_score, consistent, matches,
                    contradictions, variant_idx,
                ))
            except Exception:
                continue

        if not candidates:
            return _fallback_pbt(prompt)

        # --- Phase 4: Select best spec ---
        selected = _select_best_spec(candidates, examples)

        # Build axis summary for diagnostics
        axis_summary = "; ".join(
            f"{a.name}({a.confidence:.0%})" for a in strong_axes
        )

        return {
            "pbt_payload": selected,
            "_multi_spec_candidates": len(candidates),
            "_multi_spec_consistent": sum(
                1 for _, _, c, _, _, _ in candidates if c
            ),
            "_multi_spec_mode": "deterministic_contrastive",
            "_multi_spec_difficulty": f"{len(strong_axes)}_axes",
            "_multi_spec_ambiguities": len(strong_axes),
            "_det_axes": axis_summary,
            "_det_n_combos": len(combos),
            "_det_n_interpretations": analysis.interpretation_count(),
        }

    return deterministic_multi_spec_node

