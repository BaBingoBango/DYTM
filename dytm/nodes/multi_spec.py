"""Multi-Spec node — generates K spec interpretations and selects the best.

Two-phase approach to interpretation diversity:

Phase 1: AMBIGUITY ANALYSIS
    An LLM call identifies what's actually ambiguous or tricky in the
    prompt — boundary conditions, semantic subtleties, algorithmic choices.
    This produces explicit "interpretation axes" to explore.

Phase 2: CONTRASTIVE SPEC GENERATION
    K spec calls, each guided to take a DIFFERENT stance on the identified
    ambiguities.  This produces semantically diverse contracts, not just
    textually diverse ones (which is all temperature sampling gives you).

Selection: Example-consistency filter + cross-spec invariant survival.

This addresses DYTM's main weakness on benchmarks: correlated spec errors.
When all temperatures produce the same wrong understanding, contrastive
prompting forces exploration of genuinely different interpretations.
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from dytm.nodes.spec import SPEC_SYSTEM_PROMPT
from dytm.schemas.pbt import SpecOutput
from dytm.state import DYTMState


# ---------------------------------------------------------------------------
# Phase 1: Ambiguity Analyst
# ---------------------------------------------------------------------------

AMBIGUITY_ANALYST_PROMPT = """\
You are an Ambiguity Analyst. Given a coding task description, identify the \
aspects that are AMBIGUOUS, UNDERSPECIFIED, or TRICKY — places where a \
reasonable programmer might interpret the requirements differently.

For each ambiguity, describe:
1. What aspect is ambiguous
2. Two or three different reasonable interpretations
3. Which interpretation is most likely correct

Focus on:
- Boundary conditions (< vs <=, inclusive vs exclusive)
- Edge case handling (empty input, zero, negative numbers)
- Algorithmic subtleties (operator precedence, signed digits, prefix validity vs count equality)
- Return type ambiguities (list vs tuple, int vs float)
- Semantic ambiguities (what does "closest" mean? what does "balanced" mean?)

Be SPECIFIC and TECHNICAL. Don't list generic ambiguities — identify the \
ones that would actually cause two implementations to produce different outputs.

If the task is completely unambiguous (clear examples, clear algorithm, \
no edge cases in doubt), say so — return an empty ambiguities list.
"""


class AmbiguityAxis(BaseModel):
    """A single ambiguous aspect of the task."""
    aspect: str = Field(description="What is ambiguous (e.g. 'bracket matching algorithm')")
    interpretations: list[str] = Field(
        description="2-3 different reasonable interpretations",
        min_length=2,
        max_length=3,
    )
    most_likely: int = Field(
        description="0-indexed: which interpretation is most likely correct",
        ge=0,
    )


class AmbiguityAnalysis(BaseModel):
    """Result of analyzing a task for ambiguities."""
    ambiguities: list[AmbiguityAxis] = Field(
        default_factory=list,
        description="List of ambiguous aspects. Empty if task is clear.",
    )
    is_ambiguous: bool = Field(
        description="True if any meaningful ambiguities were found",
    )
    difficulty: str = Field(
        description="'simple', 'moderate', or 'tricky' — overall assessment",
    )


# ---------------------------------------------------------------------------
# Phase 2: Contrastive Spec Prompts
# ---------------------------------------------------------------------------

def _build_contrastive_prompt(
    task_prompt: str,
    analysis: AmbiguityAnalysis,
    variant_idx: int,
) -> str:
    """Build a spec prompt that steers toward a specific interpretation.

    variant_idx=0: Use the "most likely" interpretation for each ambiguity.
    variant_idx=1+: Use an alternative interpretation for each ambiguity.
    """
    if not analysis.ambiguities:
        # No ambiguities found — just generate normally
        return f"Task: {task_prompt}"

    guidance_lines = []
    for axis in analysis.ambiguities:
        interps = axis.interpretations
        if variant_idx == 0:
            # Primary: use the most likely interpretation
            chosen_idx = min(axis.most_likely, len(interps) - 1)
        else:
            # Alternative: cycle through non-default interpretations
            non_default = [i for i in range(len(interps)) if i != axis.most_likely]
            chosen_idx = non_default[(variant_idx - 1) % len(non_default)]

        guidance_lines.append(
            f"- For \"{axis.aspect}\": use this interpretation: "
            f"\"{interps[chosen_idx]}\""
        )

    guidance = "\n".join(guidance_lines)

    return (
        f"Task: {task_prompt}\n\n"
        f"INTERPRETATION GUIDANCE — use these specific interpretations "
        f"for any ambiguous aspects:\n{guidance}\n\n"
        f"Generate a PBT specification that is consistent with the above "
        f"interpretations. Your invariant test_body code MUST implement "
        f"the specific interpretation described above."
    )


# ---------------------------------------------------------------------------
# Example extraction from docstrings
# ---------------------------------------------------------------------------

def extract_docstring_examples(prompt: str) -> list[dict]:
    """Extract concrete input→output examples from a HumanEval docstring.

    Parses ``>>> func(args)`` and ``>>> func(args) == expected`` patterns.
    Also handles multi-line assertions like ``>>> func(args) == [\\n    ...]``.

    Returns:
        List of dicts with keys:
        - ``call``: the function call expression (e.g. "f([1,2,3])")
        - ``expected``: the expected output as a string (e.g. "[3,2,1]")
        - ``func_name``: the function name extracted from the call
    """
    examples = []

    # Extract docstring
    docstring = ""
    m = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if not m:
        m = re.search(r"'''(.*?)'''", prompt, re.DOTALL)
    if m:
        docstring = m.group(1)

    lines = docstring.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Match >>> func(args) == expected
        m = re.match(r">>>\s*(.+?)\s*==\s*(.+)", line)
        if m:
            call = m.group(1).strip()
            expected = m.group(2).strip()

            # Handle multi-line expected values (e.g. lists spanning lines)
            # Check if expected starts with [ or ( but doesn't close
            open_count = expected.count("[") + expected.count("(") + expected.count("{")
            close_count = expected.count("]") + expected.count(")") + expected.count("}")
            while open_count > close_count and i + 1 < len(lines):
                i += 1
                next_line = lines[i].strip()
                if next_line.startswith(">>>") or next_line.startswith("..."):
                    break
                expected += " " + next_line
                open_count = expected.count("[") + expected.count("(") + expected.count("{")
                close_count = expected.count("]") + expected.count(")") + expected.count("}")

            # Extract function name
            fn_match = re.match(r"(\w+)\(", call)
            func_name = fn_match.group(1) if fn_match else None

            examples.append({
                "call": call,
                "expected": expected,
                "func_name": func_name,
            })
            i += 1
            continue

        # Match >>> func(args)\n value  (implicit return display)
        m2 = re.match(r">>>\s*(\w+\(.+\))\s*$", line)
        if m2 and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if not next_line.startswith(">>>") and next_line:
                call = m2.group(1).strip()
                fn_match = re.match(r"(\w+)\(", call)
                func_name = fn_match.group(1) if fn_match else None
                examples.append({
                    "call": call,
                    "expected": next_line,
                    "func_name": func_name,
                })
                i += 2
                continue

        i += 1

    return examples


# ---------------------------------------------------------------------------
# Spec consistency checking
# ---------------------------------------------------------------------------

def check_spec_example_consistency(
    pbt: dict,
    examples: list[dict],
    timeout: int = 5,
) -> tuple[bool, int, int]:
    """Check if a PBT contract's edge cases are consistent with docstring examples.

    For each docstring example, checks if ANY edge case in the contract
    contradicts it (claims a different return value for the same input).

    Args:
        pbt: The PBT payload dict.
        examples: Extracted docstring examples.
        timeout: Subprocess timeout for validation.

    Returns:
        (is_consistent, n_matches, n_contradictions)
        - is_consistent: True if no contradictions found
        - n_matches: number of examples confirmed by edge cases
        - n_contradictions: number of examples contradicted by edge cases
    """
    if not examples:
        return True, 0, 0

    func_name = pbt.get("function_name", "")
    edge_cases = [
        p for p in pbt.get("properties", [])
        if p.get("type") == "edge_case"
    ]

    n_matches = 0
    n_contradictions = 0

    for ex in examples:
        ex_call = ex["call"]
        ex_expected = ex["expected"]

        # Extract the args from the example call
        m = re.match(r"\w+\((.+)\)$", ex_call, re.DOTALL)
        if not m:
            continue
        ex_args = m.group(1).strip()

        for edge in edge_cases:
            edge_input = edge.get("input", "")
            edge_behavior = edge.get("expected_behavior", "")

            if not edge_input or not edge_behavior:
                continue

            # Normalize inputs for comparison
            # Strip outer parens from edge_input
            normalized_edge = edge_input.strip()
            if normalized_edge.startswith("(") and normalized_edge.endswith(")"):
                normalized_edge = normalized_edge[1:-1].strip()

            # Try to check if the inputs refer to the same call
            # This is a heuristic — compare normalized strings
            if not _inputs_match(ex_args, normalized_edge):
                continue

            # Inputs match — check if the expected outputs agree
            behavior_lower = edge_behavior.lower().strip()
            if behavior_lower.startswith("returns "):
                edge_expected = edge_behavior.split("returns", 1)[1].strip()

                # Try to compare values via execution
                match = _values_match(ex_expected, edge_expected)
                if match is True:
                    n_matches += 1
                elif match is False:
                    n_contradictions += 1
                # match is None → couldn't determine, skip

    is_consistent = n_contradictions == 0
    return is_consistent, n_matches, n_contradictions


def _inputs_match(a: str, b: str) -> bool:
    """Heuristic check if two input strings refer to the same arguments."""
    # Normalize whitespace
    a_norm = re.sub(r"\s+", "", a)
    b_norm = re.sub(r"\s+", "", b)

    if a_norm == b_norm:
        return True

    # Try evaluating both and comparing
    try:
        val_a = eval(a, {"__builtins__": {}})
        val_b = eval(b, {"__builtins__": {}})
        return val_a == val_b
    except Exception:
        return False


def _values_match(a: str, b: str) -> Optional[bool]:
    """Check if two expected value strings represent the same value.

    Returns True (match), False (contradiction), or None (can't determine).
    """
    # Normalize whitespace
    a_norm = re.sub(r"\s+", "", a.strip())
    b_norm = re.sub(r"\s+", "", b.strip())

    if a_norm == b_norm:
        return True

    # Try evaluating both
    try:
        val_a = eval(a, {"__builtins__": {}, "True": True, "False": False, "None": None})
        val_b = eval(b, {"__builtins__": {}, "True": True, "False": False, "None": None})
        return val_a == val_b
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Spec scoring — improved to value invariant specificity
# ---------------------------------------------------------------------------

def score_spec(pbt: dict) -> float:
    """Score a PBT contract by restrictiveness and specificity.

    Higher score = more restrictive = harder to accidentally satisfy = better.

    Scoring:
    - Each invariant property with test_body: +3
      - Bonus +1 if test_body contains comparison operators (more specific)
      - Bonus +1 if test_body uses assume() (preconditioned = sophisticated)
    - Each membership property with test_body: +2
    - Each output_type property with test_body: +1
    - Each edge_case: +0.5
    - Bonus for more properties overall: +0.1 per property
    """
    score = 0.0
    properties = pbt.get("properties", [])

    for prop in properties:
        ptype = prop.get("type", "")
        has_body = bool(prop.get("test_body"))
        body = prop.get("test_body", "")

        if ptype == "invariant" and has_body:
            score += 3.0
            # Specificity bonus: does the test_body compute something concrete?
            if any(op in body for op in ["==", "<=", ">=", "!=", "< ", "> "]):
                score += 1.0
            if "assume(" in body:
                score += 1.0
        elif ptype == "membership" and has_body:
            score += 2.0
        elif ptype == "output_type" and has_body:
            score += 1.0
        elif ptype == "edge_case":
            score += 0.5

        score += 0.1  # bonus per property

    return score


# ---------------------------------------------------------------------------
# Multi-Spec node factory
# ---------------------------------------------------------------------------

def make_multi_spec_node(
    model: BaseChatModel,
    k: int = 3,
    temperatures: list[float] | None = None,
):
    """Create a multi-spec node that generates K contract interpretations.

    Uses a two-phase approach:
    1. Ambiguity Analysis: identify what's tricky about the task
    2. Contrastive Generation: K specs that differ on the ambiguous aspects

    Falls back to temperature diversity if the analyst call fails.

    Args:
        model: Base LLM for all calls.
        k: Number of spec interpretations to generate.
        temperatures: Fallback temperatures if ambiguity analysis fails.
                      Defaults to [0.0, 0.7, 1.0] for K=3.

    Returns:
        A node function ``(DYTMState) -> dict`` suitable for LangGraph.
    """
    if temperatures is None:
        if k == 1:
            temperatures = [0.0]
        elif k == 2:
            temperatures = [0.0, 0.8]
        elif k == 3:
            temperatures = [0.0, 0.7, 1.0]
        else:
            temperatures = [round(i * 1.2 / (k - 1), 2) for i in range(k)]

    # Analyst model (low temp for reliable analysis)
    analyst_model = model
    if isinstance(model, ChatOpenAI):
        analyst_model = ChatOpenAI(
            model=model.model_name,
            temperature=0,
            timeout=60,
            max_retries=2,
        )
    analyst_structured = analyst_model.with_structured_output(AmbiguityAnalysis)

    # Spec model (temp=0 for spec gen, since diversity comes from prompting)
    spec_structured = model.with_structured_output(SpecOutput)

    # Fallback: temperature-diverse models (if analyst fails)
    fallback_models = []
    for temp in temperatures[:k]:
        if temp == 0.0:
            fallback_models.append(model.with_structured_output(SpecOutput))
        else:
            if isinstance(model, ChatOpenAI):
                variant = ChatOpenAI(
                    model=model.model_name,
                    temperature=temp,
                    timeout=90,
                    max_retries=model.max_retries,
                )
            else:
                variant = model.bind(temperature=temp)
            fallback_models.append(variant.with_structured_output(SpecOutput))

    def multi_spec_node(state: DYTMState) -> dict:
        """Generate K spec interpretations and select the best one.

        Phase 1: Analyze the task for ambiguities.
        Phase 2: Generate K specs, each exploring a different interpretation.
        Phase 3: Filter by example consistency, select by score.

        On Executive rejection retries, falls back to single-spec revision.
        """
        prompt = state["prompt"]
        spec_retry = state.get("spec_retry_count", 0)
        approval_feedback = state.get("approval_feedback")

        # On retries (Executive rejection), fall back to single-spec
        if spec_retry > 0 and approval_feedback:
            return _single_spec_with_feedback(
                spec_structured, prompt, spec_retry, approval_feedback,
            )

        examples = extract_docstring_examples(prompt)
        analysis = None
        use_contrastive = False

        # --- Phase 1: Ambiguity Analysis ---
        try:
            analyst_messages = [
                SystemMessage(content=AMBIGUITY_ANALYST_PROMPT),
                HumanMessage(content=f"Task:\n{prompt}"),
            ]
            analysis = analyst_structured.invoke(analyst_messages)
            # Gate: only use contrastive generation when the analyst sees
            # genuine, non-trivial ambiguity.  For "simple" problems the
            # default interpretation is almost always correct and steering
            # toward alternatives causes regressions.
            use_contrastive = (
                analysis.is_ambiguous
                and len(analysis.ambiguities) > 0
                and analysis.difficulty in ("moderate", "tricky")
            )
        except Exception:
            # Analyst failed — fall back to temperature diversity
            use_contrastive = False

        # --- Phase 2: Generate K specs ---
        candidates: list[tuple[dict, float, bool, int, int, int]] = []
        #                      pbt, score, consistent, matches, contradictions, variant_idx

        if use_contrastive and analysis is not None:
            # Contrastive generation: each spec gets different interpretation guidance
            for variant_idx in range(k):
                try:
                    contrastive_prompt = _build_contrastive_prompt(
                        prompt, analysis, variant_idx,
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
        else:
            # Fallback: temperature diversity (original approach)
            for i, structured_model in enumerate(fallback_models):
                try:
                    messages = [
                        SystemMessage(content=SPEC_SYSTEM_PROMPT),
                        HumanMessage(content=f"Task: {prompt}"),
                    ]
                    result: SpecOutput = structured_model.invoke(messages)
                    pbt = result.pbt.model_dump()

                    consistent, matches, contradictions = \
                        check_spec_example_consistency(pbt, examples)
                    restriction_score = score_spec(pbt)

                    candidates.append((
                        pbt, restriction_score, consistent, matches,
                        contradictions, i,
                    ))
                except Exception:
                    continue

        if not candidates:
            return _fallback_pbt(prompt)

        # --- Phase 3: Select the best spec ---
        selected = _select_best_spec(candidates, examples)

        return {
            "pbt_payload": selected,
            "_multi_spec_candidates": len(candidates),
            "_multi_spec_consistent": sum(
                1 for _, _, c, _, _, _ in candidates if c
            ),
            "_multi_spec_mode": "contrastive" if use_contrastive else "temperature",
            "_multi_spec_difficulty": analysis.difficulty if analysis else "unknown",
            "_multi_spec_ambiguities": len(analysis.ambiguities) if analysis else 0,
        }

    return multi_spec_node


def _select_best_spec(
    candidates: list[tuple[dict, float, bool, int, int, int]],
    examples: list[dict],
) -> dict:
    """Select the best spec from K candidates.

    Selection priority:
    1. Consistent with docstring examples (no contradictions)
    2. Most example matches (confirmed by edge cases)
    3. Highest restrictiveness/specificity score
    4. First candidate (variant 0 = most-likely interpretation)

    Args:
        candidates: List of (pbt, score, consistent, matches, contradictions, variant_idx)
        examples: Extracted docstring examples

    Returns:
        The selected PBT payload dict.
    """
    # Separate consistent and inconsistent
    consistent = [
        (pbt, score, m, c, vi)
        for pbt, score, cons, m, c, vi in candidates if cons
    ]
    inconsistent = [
        (pbt, score, m, c, vi)
        for pbt, score, cons, m, c, vi in candidates if not cons
    ]

    # Prefer consistent specs
    pool = consistent if consistent else inconsistent

    # Sort by: matches (desc), score (desc), variant_idx (asc for stability)
    # Add a bonus for variant 0 (most-likely interpretation) — when scores
    # are close, the default interpretation should win.  This prevents
    # alternative interpretations from winning on score noise alone.
    VARIANT_0_BONUS = 2.0
    pool.sort(
        key=lambda x: (
            x[2],                                          # matches desc
            x[1] + (VARIANT_0_BONUS if x[4] == 0 else 0), # score desc (variant 0 gets bonus)
            -x[4],                                         # variant_idx asc
        ),
        reverse=True,
    )

    return pool[0][0]


def _single_spec_with_feedback(
    model,
    prompt: str,
    spec_retry: int,
    feedback: str,
) -> dict:
    """Generate a single revised spec based on Executive feedback."""
    messages = [
        SystemMessage(content=SPEC_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Task: {prompt}\n\n"
            f"--- PREVIOUS CONTRACT REJECTED (attempt {spec_retry}) ---\n"
            f"Rejection reason:\n{feedback}\n\n"
            f"Please revise your specification to address this feedback."
        )),
    ]

    try:
        result: SpecOutput = model.invoke(messages)
        return {"pbt_payload": result.pbt.model_dump()}
    except Exception as exc:
        return _fallback_pbt(prompt)


def _fallback_pbt(prompt: str) -> dict:
    """Return a minimal fallback PBT when all spec generation fails."""
    fallback_name = "unknown_function"
    for word in prompt.split():
        if word.isidentifier() and word not in ("def", "the", "a", "an", "is", "if"):
            fallback_name = word
            break
    return {
        "pbt_payload": {
            "function_name": fallback_name,
            "properties": [
                {
                    "name": "spec_generation_failed",
                    "type": "edge_case",
                    "expected": "All spec interpretations failed",
                    "input": "None",
                    "expected_behavior": "raises RuntimeError",
                }
            ],
        },
    }

