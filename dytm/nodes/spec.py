"""Spec node — LLM generates ONLY verification properties (PBTs), no code.

This is the first node in the new pipeline.  It asks the LLM to think
purely about *what* the function should do (its specification / contract),
WITHOUT writing any implementation.  This enforces the key DYTM insight:
the verification contract must be approved before any code exists.

The approved PBTs are then handed to the Coder node on a fresh context.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from dytm.schemas.pbt import SpecOutput
from dytm.state import DYTMState


# ---------------------------------------------------------------------------
# System prompt — specification only, NO code
# ---------------------------------------------------------------------------

SPEC_SYSTEM_PROMPT = """\
You are the Specification Writer in the DYTM ("Do You Trust Me") verification system.

Your job: given a natural-language coding task, produce a structured \
Property-Based Test (PBT) payload describing WHAT the function must do.

CRITICAL: You do NOT write any code.  You only write the verification \
contract — the testable properties that any correct implementation must \
satisfy.  Think of yourself as writing a formal specification.

The PBT payload must contain:
- `function_name`: the exact name the function must have.
- `properties`: a list of test properties.  Each property has:
  - `name`: short unique identifier (snake_case)
  - `type`: one of "output_type", "membership", "invariant", or "edge_case"
  - `expected`: human-readable description of expected behaviour (shown in the CNL contract)
  - `test_body`: (REQUIRED for output_type, membership, invariant) a complete Python test \
function including @given decorator and def line. This will be pasted directly into a pytest \
file and executed against the eventual code.
  - `input` (edge_case only): the specific input as a Python expression string
  - `expected_behavior` (edge_case only): what should happen, e.g. "raises ValueError" or "returns 5"

Property type guide:
- **output_type**: checks the return type. `expected` = simple type name (e.g. "int"). \
`test_body` = a Hypothesis test that calls the function with random valid inputs and checks isinstance.
- **membership**: checks a containment relationship. `expected` = human description. \
`test_body` = a Hypothesis test that checks the relationship.
- **invariant**: checks a general invariant over random inputs. `expected` = human description. \
`test_body` = a Hypothesis test that asserts the invariant.
- **edge_case**: checks specific inputs. Uses `input` and `expected_behavior` fields (NO test_body needed).

██████████████████████████████████████████████████████████████
MOST IMPORTANT — DISCRIMINATING INVARIANTS:
██████████████████████████████████████████████████████████████
Your #1 job is to write invariants that would DISTINGUISH a correct \
implementation from a plausible-but-wrong one.  A contract that only \
checks the return type and a few edge cases is WEAK and USELESS — \
it would accept buggy code.

Ask yourself: "What is the CORE RELATIONSHIP between input and output?"
Then express that as an invariant with test_body.

Examples of STRONG discriminating invariants:
- For "sum of a list": output == sum(input_list)
- For "collisions of n left + n right cars": output == n * n
- For "reverse a string": output == input[::-1]
- For "sort a list": output == sorted(input_list)
- For "decode(encode(x)) round-trip": decode(encode(s)) == s
- For "count vowels": output == sum(1 for c in s if c in 'aeiou')

These are the invariants that MATTER.  Without them, the contract is \
just checking types and a handful of examples — any buggy implementation \
that happens to get those examples right will pass.

You MUST include at least one invariant property with a test_body that \
checks the CORE INPUT→OUTPUT RELATIONSHIP over random inputs.
██████████████████████████████████████████████████████████████

IMPORTANT GUIDELINES:
- Generate at least 3-5 edge_case properties with CONCRETE inputs and expected behaviors.
- Edge case inputs must be valid Python expressions that can be passed directly to the function.
- For edge cases that should raise exceptions, use: expected_behavior="raises ExceptionType"
  BUT ONLY when the task EXPLICITLY says the function should raise for certain inputs.
  Do NOT invent error-handling requirements the task doesn't mention.
- For edge cases with concrete return values, use: expected_behavior="returns <value>"
- Think about: empty inputs, zero, negative numbers, single elements, large inputs, typical cases.
- For test_body: use `from hypothesis import given, settings, assume` and `hypothesis.strategies as st`. \
The test file already imports these.
- For test_body: the function under test is available by name in the test file scope.
- For test_body: use Python 3.10+ type hints (lowercase `list`, `dict`, etc.) — NOT `List`, `Dict` from typing.
- CRITICAL — STRATEGY DOMAIN: Hypothesis strategies in test_body MUST generate inputs \
that are VALID for the function under test.  For example:
  - If the function expects a string of parentheses: use `st.from_regex(r'[()\\s]*', fullmatch=True)`, \
NOT `st.text()`.
  - If the function expects a list of positive integers: use `st.lists(st.integers(min_value=1))`, \
NOT `st.lists(st.integers())`.
  - If the function expects floats in [0,1]: use `st.floats(min_value=0, max_value=1)`.
  - Always use `allow_nan=False, allow_infinity=False` with `st.floats()` unless the task \
specifically requires handling NaN/Inf.
  Using too-broad strategies (like `st.text()` for a parentheses parser) causes false failures \
when Hypothesis generates inputs outside the function's domain.
- CRITICAL — PRECONDITIONED TESTS: When a test needs inputs satisfying a precondition \
(e.g. "when no two elements are close, the result is False"), use `assume()` to filter, \
NOT a bare `assert` on the precondition.  Example:
  WRONG:  assert all(abs(a-b) > threshold for ...); result = f(x); assert result is False
  RIGHT:  assume(all(abs(a-b) > threshold for ...)); result = f(x); assert result is False
  `assume()` tells Hypothesis to skip examples that don't meet the precondition.  A bare \
`assert` crashes the test instead of skipping.  The test file already imports `assume`.
- CRITICAL — BOUNDARY PRECISION: In test_body, be precise about strict vs inclusive comparisons. \
Match the EXACT wording of the task:
  - "closer than X" / "less than X" → use `<` (strict), NOT `<=`
  - "at most X" / "no more than X" → use `<=` (inclusive)
  - "within X" / "up to X" → use `<=` (inclusive)
  - "greater than X" → use `>` (strict), NOT `>=`
  Getting `<` vs `<=` wrong causes Hypothesis to find boundary counterexamples \
(e.g., threshold=0.0 with identical elements) that fail the test but not the code.
- CRITICAL — ROUND-TRIP INVARIANTS: When the task involves encode/decode, \
serialize/deserialize, or any reversible transformation, ALWAYS include an invariant \
that checks `decode(encode(x)) == x` (or the equivalent) over random inputs. \
This is the single strongest property for these tasks.
- CRITICAL — HELPER FUNCTIONS IN TASK: When the task prompt PROVIDES helper \
functions (e.g. an `encode_cyclic` function, a `poly` evaluation function, \
or any utility), you MUST use them in your invariant test_body.  These helpers \
will be available in the test file (the Coder will include them).  Copy the \
helper function definition VERBATIM into your test_body so it's self-contained. \
For example, if the task provides `encode_cyclic(s)` and asks you to write \
`decode_cyclic(s)`, your test_body should define encode_cyclic inline and then \
assert `decode_cyclic(encode_cyclic(s)) == s`.
- CRITICAL — ALGORITHMIC PATTERN RECOGNITION: Identify the algorithmic pattern \
in the task and use the STRONGEST possible invariant for that pattern:
  - **Root-finding / solver**: "find x such that f(x) = 0" → invariant: \
`abs(f(result)) < 1e-4`.  Plug the output back into the given function \
and verify it's near zero.  If the task provides an evaluation function \
(like `poly(coeffs, x)`), use it directly.
  - **Encode/decode**: one function is given, write the inverse → invariant: \
`decode(encode(x)) == x` for random x.  Include the given function in test_body.
  - **Recursive sequences**: base cases + recurrence are given → invariant: \
`f(n) == f(n-1) + f(n-2) + ...` for random n > base cases.
  - **Sorting/ordering**: → invariant: `output == sorted(input)` or check \
that output is a permutation AND ordered.
  - **String transformation**: → invariant that computes the expected output \
from scratch (e.g. `output == input[::-1]` for reverse).
- Think carefully about what properties are NECESSARY and SUFFICIENT to ensure correctness.
- Consider: what invariants would distinguish a correct implementation from a buggy one?

Example for a "longest string in a list" task:

```json
{
  "function_name": "longest_string",
  "properties": [
    {
      "name": "returns_a_string",
      "type": "output_type",
      "expected": "str",
      "test_body": "@given(input_list=st.lists(st.text(min_size=1), min_size=1))\\n@settings(max_examples=50)\\ndef test_returns_a_string(input_list):\\n    result = longest_string(input_list)\\n    assert isinstance(result, str)"
    },
    {
      "name": "result_is_in_input",
      "type": "membership",
      "expected": "output is always a member of the input list",
      "test_body": "@given(input_list=st.lists(st.text(min_size=1), min_size=1))\\n@settings(max_examples=50)\\ndef test_result_is_in_input(input_list):\\n    result = longest_string(input_list)\\n    assert result in input_list"
    },
    {
      "name": "no_string_is_longer",
      "type": "invariant",
      "expected": "len(output) >= len(s) for all s in input_list",
      "test_body": "@given(input_list=st.lists(st.text(), min_size=1))\\n@settings(max_examples=50)\\ndef test_no_string_is_longer(input_list):\\n    result = longest_string(input_list)\\n    assert all(len(result) >= len(s) for s in input_list)"
    },
    {"name": "single_element", "type": "edge_case", "expected": "returns hello", "input": "['hello']", "expected_behavior": "returns 'hello'"},
    {"name": "empty_list_raises", "type": "edge_case", "expected": "raises ValueError", "input": "", "expected_behavior": "raises ValueError"}
  ]
}
```

Now write the specification for the given task.  Remember: NO code, only PBT properties.
"""


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_spec_node(model: BaseChatModel):
    """Create a spec node function bound to a specific LLM.

    Args:
        model: Any LangChain chat model (OpenAI, Anthropic, Ollama, etc.).

    Returns:
        A node function ``(DYTMState) -> dict`` suitable for LangGraph.
    """
    from langchain_core.messages import AIMessage

    structured_model = model.with_structured_output(SpecOutput)

    def spec_node(state: DYTMState) -> dict:
        """Call the LLM to generate PBT specification ONLY (no code).

        Uses chat-mode conversation: on retries, the full history of
        previous spec outputs and rejection feedback is preserved so the
        model can make targeted fixes without regressing on earlier fixes.

        If the LLM call fails (API error, structured output parse error),
        returns a minimal fallback PBT payload so the graph continues
        rather than crashing.
        """
        prompt = state["prompt"]
        spec_retry = state.get("spec_retry_count", 0)
        approval_feedback = state.get("approval_feedback")
        conversation = list(state.get("_spec_conversation") or [])

        # Build messages from conversation history
        messages = [SystemMessage(content=SPEC_SYSTEM_PROMPT)]

        if spec_retry == 0 or not conversation:
            # First attempt — fresh start
            messages.append(HumanMessage(content=f"Task: {prompt}"))
        else:
            # Retry — replay full conversation history as multi-turn chat
            for role, content in conversation:
                if role == "human":
                    messages.append(HumanMessage(content=content))
                elif role == "ai":
                    messages.append(AIMessage(content=content))

            # Add the latest rejection feedback
            if approval_feedback:
                messages.append(HumanMessage(content=(
                    f"--- CONTRACT REJECTED (attempt {spec_retry}) ---\n"
                    f"Rejection reason:\n{approval_feedback}\n\n"
                    f"Please revise your specification to address this feedback. "
                    f"IMPORTANT: Do NOT regress on fixes from previous rounds. "
                    f"Keep everything that was already correct and only fix "
                    f"what was flagged."
                )))

        try:
            result: SpecOutput = structured_model.invoke(messages)
        except Exception as exc:
            # LLM call failed — return a minimal fallback PBT so the graph
            # can continue (the Doer will fail, triggering a retry).
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
                            "expected": f"Spec LLM error: {str(exc)[:200]}",
                            "input": "None",
                            "expected_behavior": "raises RuntimeError",
                        }
                    ],
                },
            }

        # Serialize the spec output for conversation history
        import json
        spec_json = json.dumps(result.pbt.model_dump(), indent=2)

        # Build updated conversation history
        if spec_retry == 0 or not conversation:
            # First turn: human prompt + AI response
            new_conversation = [
                ("human", f"Task: {prompt}"),
                ("ai", spec_json),
            ]
        else:
            # Append the rejection + new AI response
            new_conversation = conversation.copy()
            if approval_feedback:
                new_conversation.append((
                    "human",
                    f"--- CONTRACT REJECTED (attempt {spec_retry}) ---\n"
                    f"Rejection reason:\n{approval_feedback}\n\n"
                    f"Please revise your specification."
                ))
            new_conversation.append(("ai", spec_json))

        return {
            "pbt_payload": result.pbt.model_dump(),
            "_spec_conversation": new_conversation,
        }

    return spec_node


