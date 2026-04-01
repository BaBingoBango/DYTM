"""Approval node — Executive gate between CNL contract and code generation.

Supports three approval modes:

- **eval**: An Executive LLM reads the CNL contract and the original task,
  then decides whether the contract is sufficient to verify correctness.
  If rejected, provides feedback for the Spec node to revise.
- **free**: The human Executive approves/rejects interactively.
- **auto**: Rubber-stamp (for unit tests / CI).

The Executive LLM is a SEPARATE model call with a FRESH context — it has
no memory of the Spec node's generation process.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from dytm.state import DYTMState


# ---------------------------------------------------------------------------
# Executive LLM system prompt
# ---------------------------------------------------------------------------

EXECUTIVE_SYSTEM_PROMPT = """\
You are the Executive in the DYTM ("Do You Trust Me") verification system.

Your job: review a Controlled Natural Language (CNL) verification contract \
and decide whether it is SUFFICIENT to verify that a correct implementation \
of the given task will pass, and that an incorrect implementation will fail.

You will receive:
1. The original task description
2. The CNL verification contract — a human-readable list of properties, \
each followed by its TEST IMPLEMENTATION (the actual Python test code \
that will run against the implementation)

You must decide: APPROVE or REJECT.

██████████████████████████████████████████████████████████████
YOUR #1 PRIORITY — DISCRIMINATING INVARIANT CHECK:
██████████████████████████████████████████████████████████████
The MOST IMPORTANT thing you check: does the contract contain at least one \
invariant or property-based test that captures the CORE INPUT→OUTPUT \
RELATIONSHIP?  A contract that only checks types and a few edge cases is \
WEAK — it will accept wrong implementations that happen to get those \
specific examples right.

Examples of STRONG contracts (APPROVE):
- "sort a list" with invariant: output == sorted(input)
- "number of collisions" with invariant: output == n * n
- "decode(encode(x))" with invariant: decode(encode(s)) == s
- "count vowels" with invariant: output == sum(1 for c in s if c in 'aeiou')

Examples of WEAK contracts (REJECT):
- "sort a list" with only: returns list, edge case [1] → [1], edge case [3,1] → [1,3]
  (MISSING: the invariant that output == sorted(input))
- "number of collisions" with only: returns int, edge case 1 → 1
  (MISSING: the formula output == n * n)
██████████████████████████████████████████████████████████████

APPROVE if:
- The contract has at least one DISCRIMINATING invariant testing the core behavior
- The return type is specified
- At least a few edge cases are present
- The edge case expected values are mathematically correct (see below)
- The test implementations correctly match their property descriptions (see below)

REJECT if:
- MISSING CORE INVARIANT: no invariant/property tests the fundamental \
  input→output relationship — only type checks and a handful of edge cases
- One or more edge case expected values are MATHEMATICALLY WRONG
- A test implementation contradicts its property description or has a boundary bug
- The contract is fundamentally wrong (contradicts the task)
- A Hypothesis test uses `assert` for a precondition instead of `assume()`

IMPORTANT guidelines:
- Do NOT reject for minor formatting issues (stray punctuation, minor wording)
- Do NOT invent requirements that are not stated in the original task
- Do NOT demand the contract handle behaviors the task doesn't specify
- If the contract has a strong core invariant + reasonable edge cases, APPROVE it
- Focus on SUBSTANCE over STYLE

CRITICAL — VERIFY EDGE CASE MATH:
For EVERY edge case in the contract, you MUST compute the expected value \
yourself step by step.  Wrong expected values poison the entire verification.
Examples of mistakes to catch:
  - "product of [-1, -2, -3] returns 6" — WRONG, should be -6
  - "sum of [1, 2, 3] returns 5" — WRONG, should be 6
If ANY edge case expected value is wrong, REJECT and specify which edge \
case is incorrect and what the correct value should be.

CRITICAL — VERIFY TEST IMPLEMENTATIONS:
Each property in the contract includes a TEST IMPLEMENTATION section \
showing the actual Python code that will run.  Check that:
1. Boundary conditions match the task wording: "closer than X" must use `<`, \
NOT `<=`.  "at most X" must use `<=`, NOT `<`.
2. Hypothesis strategies generate VALID inputs for the function (e.g. not \
`st.text()` for a function that only accepts parentheses).
3. Preconditioned tests use `assume()` NOT bare `assert` to filter inputs.
4. The test assertions are logically consistent with the property description.
5. No test property contradicts another.
If a test implementation has bugs, REJECT and specify what's wrong.

Start your response with APPROVE or REJECT, then explain your reasoning briefly.
"""


class ExecutiveDecision:
    """Simple container for the Executive LLM's decision."""

    def __init__(self, approved: bool, reasoning: str):
        self.approved = approved
        self.reasoning = reasoning


def _parse_executive_response(text: str) -> ExecutiveDecision:
    """Parse the Executive LLM's free-text response into a decision."""
    text_lower = text.lower()

    # Look for explicit decision markers near the start
    if "reject" in text_lower[:200]:
        return ExecutiveDecision(approved=False, reasoning=text)
    if "approve" in text_lower[:200]:
        return ExecutiveDecision(approved=True, reasoning=text)

    # Fallback: look for keywords anywhere
    reject_signals = ["reject", "insufficient", "missing", "too vague", "not enough"]
    approve_signals = ["approve", "sufficient", "adequate", "looks good", "well-covered"]

    reject_score = sum(1 for s in reject_signals if s in text_lower)
    approve_score = sum(1 for s in approve_signals if s in text_lower)

    approved = approve_score >= reject_score
    return ExecutiveDecision(approved=approved, reasoning=text)


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_approval_node(executive_model: BaseChatModel | None = None):
    """Create an approval node, optionally bound to an Executive LLM.

    Args:
        executive_model: If provided, used for LLM-based approval in eval mode.
                         If None, eval mode falls back to auto-approve.

    Returns:
        A node function ``(DYTMState) -> dict`` suitable for LangGraph.
    """

    def approval_node(state: DYTMState) -> dict:
        """Approve or reject the CNL contract.

        In eval mode: Executive LLM reviews the contract.
        In free mode: human-in-the-loop.
        In auto mode: rubber-stamp.
        """
        mode = state.get("mode", "eval")

        if mode == "auto":
            return {"approved": True, "approval_feedback": ""}

        if mode == "eval":
            return _executive_llm_approval(state, executive_model)

        # ---------- Free mode: human-in-the-loop ----------
        return _human_approval(state)

    return approval_node


def _executive_llm_approval(state: DYTMState, model: BaseChatModel | None) -> dict:
    """Use the Executive LLM to review the CNL contract."""
    if model is None:
        # No executive model configured — auto-approve
        return {"approved": True, "approval_feedback": ""}

    cnl_text = state.get("cnl_text", "(no contract)")
    prompt = state.get("prompt", "(no task)")

    # The CNL is now lossless — it includes both human-readable property
    # descriptions AND the executable test implementations rendered by the
    # Translator.  The Executive reviews a single, deterministic document.
    messages = [
        SystemMessage(content=EXECUTIVE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"TASK:\n{prompt}\n\n"
            f"VERIFICATION CONTRACT:\n{cnl_text}\n\n"
            f"Should this contract be APPROVED or REJECTED?  "
            f"Start your response with APPROVE or REJECT, then explain."
        )),
    ]

    response = model.invoke(messages)
    decision = _parse_executive_response(response.content)

    result = {
        "approved": decision.approved,
        "approval_feedback": decision.reasoning,
    }

    if not decision.approved:
        result["spec_retry_count"] = state.get("spec_retry_count", 0) + 1

    return result


def _human_approval(state: DYTMState) -> dict:
    """Prompt the human Executive for approval."""
    print()
    print(state.get("cnl_text", "(no contract generated)"))
    print()

    while True:
        answer = input("Do you approve this contract? [Y]es / [N]o / [Q]uit: ").strip().lower()
        if answer in ("y", "yes"):
            return {"approved": True, "approval_feedback": ""}
        elif answer in ("n", "no"):
            feedback = input("What's wrong with the contract? (optional, press Enter to skip): ").strip()
            return {
                "approved": False,
                "approval_feedback": feedback if feedback else "Contract rejected by Executive.",
                "spec_retry_count": state.get("spec_retry_count", 0) + 1,
            }
        elif answer in ("q", "quit"):
            raise KeyboardInterrupt("Executive quit the session.")
        else:
            print("Please enter Y, N, or Q.")

