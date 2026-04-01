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
    structured_model = model.with_structured_output(SpecOutput)

    def spec_node(state: DYTMState) -> dict:
        """Call the LLM to generate PBT specification ONLY (no code).

        On first run, sends only the task prompt.  If the Executive LLM
        rejected the previous contract, appends the rejection feedback.
        """
        messages = [SystemMessage(content=SPEC_SYSTEM_PROMPT)]

        prompt = state["prompt"]
        spec_retry = state.get("spec_retry_count", 0)
        approval_feedback = state.get("approval_feedback")

        if spec_retry > 0 and approval_feedback:
            user_content = (
                f"Task: {prompt}\n\n"
                f"--- PREVIOUS CONTRACT REJECTED (attempt {spec_retry}) ---\n"
                f"Rejection reason:\n{approval_feedback}\n\n"
                f"Please revise your specification to address this feedback."
            )
        else:
            user_content = f"Task: {prompt}"

        messages.append(HumanMessage(content=user_content))

        result: SpecOutput = structured_model.invoke(messages)

        return {
            "pbt_payload": result.pbt.model_dump(),
        }

    return spec_node


