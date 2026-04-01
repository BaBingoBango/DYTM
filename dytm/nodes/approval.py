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
2. The CNL verification contract (a human-readable list of properties)

You must decide: APPROVE or REJECT.

APPROVE if:
- The contract covers the core behavior of the task
- Edge cases cover boundary conditions (empty inputs, zeros, negatives, etc.)
- The invariants/properties would distinguish correct from incorrect implementations
- The return type is specified

REJECT if:
- The contract is too vague or generic
- Important edge cases are missing
- The invariants are tautological (would pass for any implementation)
- The contract doesn't match what the task is actually asking for
- The properties are insufficient to catch common bugs

Be STRICT but FAIR.  A good contract doesn't need to be perfect, but it \
must be strong enough to meaningfully verify correctness.

Respond with your decision and reasoning.
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

