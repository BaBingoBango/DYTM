"""Deterministic Ambiguity Taxonomy for coding task descriptions.

This module implements a DETERMINISTIC (no LLM) ambiguity scanner that
pattern-matches coding task prompts against a finite taxonomy of known
interpretation axes.

The key theoretical insight (from the user's observation):

    "An English sentence X can only have so many possible meanings"

For UNRESTRICTED English, this is intractable (quantifier scope alone is
factorial). But for CODING TASK DESCRIPTIONS, the ambiguity space is
BOUNDED and ENUMERABLE. Coding tasks cluster into a finite set of
ambiguity dimensions:

    1. Boundary conditions     (< vs <=, inclusive vs exclusive)
    2. Empty/null handling     (return empty, raise, return sentinel?)
    3. Operator precedence     (standard math vs left-to-right)
    4. Structural matching     (count-equality vs prefix-validity)
    5. Signed arithmetic       (first digit negative vs all digits)
    6. Tie-breaking            (first match, last match, arbitrary)
    7. Type coercion           (int→float, list→tuple)
    8. Duplicate handling      (keep first, keep all, deduplicate)

Each axis typically has 2-3 interpretations → total space is bounded.

This scanner is DETERMINISTIC — no LLM, no randomness, same input always
produces the same output. It can be used:
    (a) As a standalone ambiguity detector
    (b) As a pre-processing layer that "seeds" the LLM Ambiguity Analyst
        with known axes, ensuring it doesn't miss common patterns
    (c) As a research tool to classify HumanEval problems by ambiguity type
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class InterpretationAxis:
    """A single ambiguity dimension with enumerated interpretations."""
    name: str
    description: str
    interpretations: list[str]
    confidence: float  # 0-1: how confident we are this ambiguity exists
    pattern_matched: str  # the text pattern that triggered this axis


@dataclass
class DeterministicAnalysis:
    """Result of deterministic ambiguity scanning."""
    axes: list[InterpretationAxis] = field(default_factory=list)
    n_interpretations: int = 1  # product of all axis cardinalities
    is_ambiguous: bool = False

    def interpretation_count(self) -> int:
        """Total number of possible interpretation combinations."""
        count = 1
        for axis in self.axes:
            count *= len(axis.interpretations)
        return count


# ---------------------------------------------------------------------------
# Pattern library — each entry detects a specific ambiguity type
# ---------------------------------------------------------------------------

# Each pattern is: (regex, axis_name, description, interpretations, confidence)
# The regex matches against the docstring text.

AMBIGUITY_PATTERNS: list[tuple[re.Pattern, str, str, list[str], float]] = [

    # --- Boundary conditions ---
    (
        re.compile(r"closer\s+than|less\s+than|smaller\s+than|fewer\s+than", re.I),
        "boundary_strict_vs_inclusive",
        "Strict (<) vs inclusive (<=) comparison",
        [
            "Use strict comparison (<): 'closer than X' means distance < X",
            "Use inclusive comparison (<=): 'closer than X' means distance <= X",
        ],
        0.7,
    ),
    (
        re.compile(r"at\s+most|no\s+more\s+than|up\s+to|within", re.I),
        "boundary_inclusive",
        "Inclusive (<=) boundary — usually unambiguous but often implemented wrong",
        [
            "Use inclusive comparison (<=): 'at most X' means value <= X",
            "Use strict comparison (<): off-by-one error interpretation",
        ],
        0.4,
    ),
    (
        re.compile(r"greater\s+than|more\s+than|larger\s+than|exceeds", re.I),
        "boundary_strict_upper",
        "Strict (>) vs inclusive (>=) upper bound",
        [
            "Use strict comparison (>): 'greater than X' means value > X",
            "Use inclusive comparison (>=): 'greater than X' means value >= X",
        ],
        0.5,
    ),

    # --- Empty/null handling ---
    (
        re.compile(r"(list|array|string|sequence|input).*empty|empty.*(list|array|string|sequence|input)", re.I),
        "empty_input_handling",
        "Behavior on empty input",
        [
            "Return empty/zero/False for empty input (identity element)",
            "Raise ValueError/IndexError for empty input",
            "Return None/sentinel for empty input",
        ],
        0.5,
    ),

    # --- Bracket/parenthesis matching ---
    (
        re.compile(r"bracket|parenthes|paren|balanced|matching|nesting|open.*clos|<.*>", re.I),
        "structural_matching_algorithm",
        "How to verify structural matching",
        [
            "Count-based: count(open) == count(close)",
            "Prefix-validity: at every prefix, count(open) >= count(close), AND total counts equal",
            "Stack-based: push on open, pop on close, stack empty at end",
        ],
        0.8,
    ),

    # --- Operator precedence ---
    (
        re.compile(r"operator|expression|evaluat.*algebra|arithmetic.*express", re.I),
        "operator_precedence",
        "Standard math precedence vs left-to-right evaluation",
        [
            "Standard mathematical operator precedence (** > */ > +-)",
            "Left-to-right evaluation (process operators in order)",
        ],
        0.7,
    ),

    # --- Signed digit arithmetic ---
    (
        re.compile(r"signed\s+digit|negative.*digit|digit.*negative", re.I),
        "signed_digit_semantics",
        "How to compute signed digits of negative numbers",
        [
            "First digit is negative, remaining digits are positive: -123 → [-1, 2, 3]",
            "All digits carry the sign: -123 → [-1, -2, -3]",
            "Absolute value of digits: -123 → [1, 2, 3]",
        ],
        0.9,
    ),

    # --- Sorting stability / tie-breaking ---
    (
        re.compile(r"sort|order|rank|arrang", re.I),
        "sort_stability",
        "Tie-breaking behavior when elements compare equal",
        [
            "Stable sort: preserve original order for equal elements",
            "Unstable sort: no guarantee on equal-element ordering",
        ],
        0.3,
    ),

    # --- Round-trip / encode-decode ---
    (
        re.compile(r"encode|decode|encrypt|decrypt|compress|decompress|serialize|deserialize", re.I),
        "round_trip_invariant",
        "Whether a round-trip property holds",
        [
            "Perfect round-trip: decode(encode(x)) == x for all valid x",
            "Lossy: decode(encode(x)) ≈ x (some information lost)",
        ],
        0.6,
    ),

    # --- Duplicate handling ---
    (
        re.compile(r"unique|duplicat|distinct|repeated|occurrence", re.I),
        "duplicate_handling",
        "How to handle duplicate elements",
        [
            "Keep all duplicates as-is",
            "Remove duplicates, keeping first occurrence",
            "Remove duplicates, keeping last occurrence",
        ],
        0.4,
    ),

    # --- Prime number edge cases ---
    (
        re.compile(r"prime", re.I),
        "prime_edge_cases",
        "Edge cases for primality (is 1 prime? is 0? negative?)",
        [
            "1 is NOT prime (mathematical convention)",
            "1 IS prime (common misconception)",
        ],
        0.5,
    ),

    # --- Floating point comparison ---
    (
        re.compile(r"float|decimal|precision|round|truncat", re.I),
        "float_comparison",
        "How to compare floating point results",
        [
            "Exact equality (==)",
            "Approximate equality (abs(a-b) < epsilon)",
        ],
        0.4,
    ),

    # --- Return type when input is ambiguous ---
    (
        re.compile(r"return.*(?:list|string|number|int|float|bool)", re.I),
        "return_type_specificity",
        "Exact return type when not fully specified",
        [
            "Return the most specific type (int if whole number, float if decimal)",
            "Always return the declared type regardless of input",
        ],
        0.3,
    ),
]


def scan_prompt(prompt: str) -> DeterministicAnalysis:
    """Deterministically scan a coding task prompt for ambiguity axes.

    Args:
        prompt: The full HumanEval-style prompt including docstring.

    Returns:
        DeterministicAnalysis with all detected ambiguity axes.
    """
    # Extract docstring for matching
    docstring = ""
    m = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if not m:
        m = re.search(r"'''(.*?)'''", prompt, re.DOTALL)
    if m:
        docstring = m.group(1)

    # Also match against full prompt (catches function signatures)
    full_text = prompt + "\n" + docstring

    axes: list[InterpretationAxis] = []

    for pattern, name, description, interpretations, confidence in AMBIGUITY_PATTERNS:
        match = pattern.search(full_text)
        if match:
            axes.append(InterpretationAxis(
                name=name,
                description=description,
                interpretations=interpretations,
                confidence=confidence,
                pattern_matched=match.group(0),
            ))

    analysis = DeterministicAnalysis(
        axes=axes,
        is_ambiguous=len(axes) > 0,
    )
    analysis.n_interpretations = analysis.interpretation_count()

    return analysis


def format_analysis(analysis: DeterministicAnalysis, task_id: str = "") -> str:
    """Pretty-print an ambiguity analysis."""
    lines = []
    prefix = f"{task_id}: " if task_id else ""

    if not analysis.is_ambiguous:
        lines.append(f"{prefix}No ambiguities detected (deterministic scan)")
        return "\n".join(lines)

    lines.append(
        f"{prefix}{len(analysis.axes)} ambiguity axes → "
        f"{analysis.n_interpretations} possible interpretation combinations"
    )
    for axis in analysis.axes:
        lines.append(f"  [{axis.confidence:.0%}] {axis.name}: {axis.description}")
        lines.append(f"       matched: \"{axis.pattern_matched}\"")
        for i, interp in enumerate(axis.interpretations):
            lines.append(f"       {i+1}. {interp}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: scan HumanEval problems
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    benchmarks = Path(__file__).resolve().parent / "benchmarks" / "HumanEval.jsonl"
    problems = []
    with open(benchmarks) as f:
        for line in f:
            problems.append(json.loads(line))

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(problems)

    total_ambiguous = 0
    total_axes = 0
    axis_counts: dict[str, int] = {}

    for p in problems[:limit]:
        analysis = scan_prompt(p["prompt"])
        if analysis.is_ambiguous:
            total_ambiguous += 1
            total_axes += len(analysis.axes)
            for axis in analysis.axes:
                axis_counts[axis.name] = axis_counts.get(axis.name, 0) + 1
            print(format_analysis(analysis, p["task_id"]))
            print()

    print("=" * 60)
    print(f"SUMMARY: {total_ambiguous}/{min(limit, len(problems))} problems have "
          f"detectable ambiguities ({total_axes} total axes)")
    print()
    print("Axis frequency:")
    for name, count in sorted(axis_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:40s} {count:3d} problems")

