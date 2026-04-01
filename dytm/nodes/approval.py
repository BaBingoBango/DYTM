"""Approval node — gate between CNL verification and code execution.

In **eval mode** the node auto-approves (rubber-stamps) the CNL so the
benchmark can run unattended.  In **free mode** it prompts the human
Executive for approval via ``input()``.  (The full Rich UI is layered on
in Phase 5; this keeps the node minimal and testable.)
"""

from __future__ import annotations

from dytm.state import DYTMState


def approval_node(state: DYTMState) -> dict:
    """Approve or reject the CNL contract.

    Args:
        state: The current LangGraph state.  Must contain ``mode`` and
               ``cnl_text``.

    Returns:
        A partial state update with ``approved`` set to ``True`` or
        ``False``.
    """
    if state.get("mode") == "eval":
        # Auto-approve for benchmark evaluation
        return {"approved": True}

    # ---------- Free mode: human-in-the-loop ----------
    print()
    print(state.get("cnl_text", "(no contract generated)"))
    print()

    while True:
        answer = input("Do you approve this contract? [Y]es / [N]o / [Q]uit: ").strip().lower()
        if answer in ("y", "yes"):
            return {"approved": True}
        elif answer in ("n", "no"):
            feedback = input("What's wrong with the contract? (optional, press Enter to skip): ").strip()
            return {
                "approved": False,
                "error_trace": feedback if feedback else "Contract rejected by Executive.",
            }
        elif answer in ("q", "quit"):
            raise KeyboardInterrupt("Executive quit the session.")
        else:
            print("Please enter Y, N, or Q.")

