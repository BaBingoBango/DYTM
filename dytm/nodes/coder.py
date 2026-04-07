"""Coder node — generates code on a FRESH context given approved PBTs.

This node is called AFTER the Executive has approved the verification
contract.  It receives only the task description and the approved PBT
specification — it has NO memory of the spec generation conversation.
This enforces the DYTM principle: the contract is independent of the code.

On retries (Doer test failures), the Coder gets the error trace and
counterexample to self-correct, but the PBT contract remains locked.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from dytm.schemas.pbt import CoderOutput
from dytm.state import DYTMState


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CODER_SYSTEM_PROMPT = """\
You are the Coder in the DYTM ("Do You Trust Me") verification system.

Your job: given a coding task and an APPROVED verification contract, \
write a Python function that passes all the specified tests.

Rules:
1. The function name MUST match the contract's function_name exactly.
2. Include type hints using Python 3.10+ syntax (use `list`, `dict`, `tuple`, `set` — NOT `List`, `Dict`, `Tuple`, `Set` from typing).
3. If you need Optional, use `X | None` syntax.
4. Do NOT import any external libraries (only builtins + math).
5. Do NOT include example usage or if __name__ blocks.
6. Return ONLY the function code — no tests, no explanations.
7. Think carefully about the edge cases listed in the contract.
8. The code must handle ALL properties and edge cases in the contract.
9. IMPORTANT: If the task provides HELPER FUNCTIONS (like `encode_cyclic`, \
`poly`, or any utility), you MUST include them in your output BEFORE \
your main function.  Copy them verbatim from the task.  The tests may \
call these helpers, so they must be defined in the same code block.

You will be tested against:
- Property-based tests (Hypothesis) checking invariants
- Concrete edge-case assertions

If your code fails any test, you'll get the error trace and a chance to fix it.
"""


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_coder_node(model: BaseChatModel):
    """Create a coder node function bound to a specific LLM.

    Args:
        model: Any LangChain chat model (OpenAI, Anthropic, Ollama, etc.).

    Returns:
        A node function ``(DYTMState) -> dict`` suitable for LangGraph.
    """
    structured_model = model.with_structured_output(CoderOutput)

    def coder_node(state: DYTMState) -> dict:
        """Generate code to satisfy the approved PBT contract.

        Always starts from a FRESH context — no memory of the spec
        generation step.  On retries, includes the error trace.

        If the LLM call fails (API error, structured output parse error),
        returns placeholder code that will fail the Doer's tests, feeding
        back into the retry loop instead of crashing the graph.
        """
        messages = [SystemMessage(content=CODER_SYSTEM_PROMPT)]

        prompt = state["prompt"]
        pbt = state["pbt_payload"]
        retry_count = state.get("retry_count", 0)
        error_trace = state.get("error_trace")
        counterexample = state.get("counterexample")

        # Build a clean description of the approved contract
        func_name = pbt["function_name"]
        contract_lines = [f"Function name: {func_name}", ""]
        contract_lines.append("APPROVED VERIFICATION CONTRACT:")
        for i, prop in enumerate(pbt["properties"], 1):
            ptype = prop["type"]
            expected = prop.get("expected", "")
            if ptype == "edge_case":
                inp = prop.get("input", "?")
                behavior = prop.get("expected_behavior", expected)
                contract_lines.append(f"  {i}. EDGE CASE: {func_name}({inp}) must {behavior}")
            elif ptype == "output_type":
                contract_lines.append(f"  {i}. RETURN TYPE: must return {expected}")
            elif ptype == "invariant":
                contract_lines.append(f"  {i}. INVARIANT: {expected}")
            elif ptype == "membership":
                contract_lines.append(f"  {i}. MEMBERSHIP: {expected}")

        contract_text = "\n".join(contract_lines)

        if retry_count > 0 and error_trace:
            user_content = (
                f"Task: {prompt}\n\n"
                f"{contract_text}\n\n"
                f"--- PREVIOUS ATTEMPT FAILED (attempt {retry_count}) ---\n"
                f"Error:\n{error_trace}\n"
            )
            if counterexample:
                user_content += f"\nMinimal counterexample:\n{counterexample}\n"
            user_content += (
                "\nPlease fix your code to address this failure. "
                "The verification contract above is LOCKED — your code must satisfy it.\n"
                "IMPORTANT: If an edge case expected value seems mathematically wrong "
                "but your code satisfies the invariant properties, prioritise matching "
                "the invariants — edge case expectations from the contract may "
                "occasionally contain arithmetic errors."
            )
        else:
            user_content = (
                f"Task: {prompt}\n\n"
                f"{contract_text}\n\n"
                f"Write the function `{func_name}` that satisfies all properties above."
            )

        messages.append(HumanMessage(content=user_content))

        try:
            result: CoderOutput = structured_model.invoke(messages)
        except Exception as exc:
            # LLM call failed — return placeholder code that will fail tests,
            # feeding back into the retry loop via the Doer.
            return {
                "code": (
                    f"def {func_name}(*args, **kwargs):\n"
                    f"    raise RuntimeError('Coder LLM call failed: {str(exc)[:100]}')\n"
                ),
            }

        return {
            "code": result.code,
        }

    return coder_node


