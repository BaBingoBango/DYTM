"""Operator node — LLM generates code + structured PBT JSON.

This is the only node in the pipeline that makes LLM calls.  It instructs
the model to produce both a Python function and a structured PBT payload
conforming to ``OperatorOutput``.  On retries, it appends the error trace
and counterexample from the previous Doer run so the model can self-correct.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from dytm.schemas.pbt import OperatorOutput
from dytm.state import DYTMState


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Operator in the DYTM ("Do You Trust Me") verification system.

Your job: given a natural-language coding task, produce TWO things:

1. **code** — a complete, self-contained Python function that solves the task.
   - Include type hints.
   - Do NOT import any external libraries (only builtins).
   - Do NOT include example usage or if __name__ blocks.

2. **pbt** — a structured Property-Based Test (PBT) payload describing \
testable properties of your code.  This payload will be used to \
*verify* your code before delivery.

The PBT payload must contain:
- `function_name`: the exact name of your function.
- `properties`: a list of test properties.  Each property has:
  - `name`: short unique identifier (snake_case)
  - `type`: one of "output_type", "membership", "invariant", or "edge_case"
  - `expected`: description of expected behaviour
  - `input` (edge_case only): the specific input as a Python expression string
  - `expected_behavior` (edge_case only): what should happen, e.g. "raises ValueError"

Property type guide:
- **output_type**: checks the return type (e.g. expected="int")
- **membership**: checks a containment relationship (e.g. expected="output in input_list")
- **invariant**: checks a general invariant over random inputs (e.g. expected="len(output) >= 0")
- **edge_case**: checks specific edge-case inputs with concrete expected behaviour

IMPORTANT GUIDELINES:
- Generate at least 2-3 edge_case properties with CONCRETE inputs and expected behaviors.
- Edge case inputs must be valid Python expressions that can be passed directly to the function.
- For edge cases that should raise exceptions, use: expected_behavior="raises ExceptionType"
- For edge cases with concrete return values, use: expected_behavior="returns <value>"
- Think about: empty inputs, zero, negative numbers, single elements, large inputs.

Example for a "longest string in a list" task:

code:
```
def longest_string(strings: list[str]) -> str:
    if not strings:
        raise ValueError("Empty list")
    return max(strings, key=len)
```

pbt:
```json
{
  "function_name": "longest_string",
  "properties": [
    {"name": "returns_a_string", "type": "output_type", "expected": "str"},
    {"name": "result_is_in_input", "type": "membership", "expected": "output in input_list"},
    {"name": "no_string_is_longer", "type": "invariant", "expected": "len(output) >= len(s) for all s in input_list"},
    {"name": "single_element", "type": "edge_case", "expected": "returns hello", "input": "['hello']", "expected_behavior": "returns hello"},
    {"name": "empty_list_raises", "type": "edge_case", "expected": "raises ValueError", "input": "", "expected_behavior": "raises ValueError"}
  ]
}
```

Now solve the given task.
"""


# ---------------------------------------------------------------------------
# Node factory
# ---------------------------------------------------------------------------

def make_operator_node(model: BaseChatModel):
    """Create an operator node function bound to a specific LLM.

    Args:
        model: Any LangChain chat model (OpenAI, Anthropic, Ollama, etc.).

    Returns:
        A node function ``(DYTMState) -> dict`` suitable for LangGraph.
    """
    structured_model = model.with_structured_output(OperatorOutput)

    def operator_node(state: DYTMState) -> dict:
        """Call the LLM to generate code + PBT payload.

        On first run, sends only the Executive's prompt.  On retries,
        appends the error trace and counterexample so the model can
        self-correct.
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Build the user message
        prompt = state["prompt"]
        retry_count = state.get("retry_count", 0)
        error_trace = state.get("error_trace")
        counterexample = state.get("counterexample")

        if retry_count > 0 and error_trace:
            user_content = (
                f"Task: {prompt}\n\n"
                f"--- PREVIOUS ATTEMPT FAILED (attempt {retry_count}) ---\n"
                f"Error:\n{error_trace}\n"
            )
            if counterexample:
                user_content += f"\nMinimal counterexample:\n{counterexample}\n"
            user_content += (
                "\nPlease fix your code and PBT properties to address this failure. "
                "Make sure edge_case inputs are concrete Python expressions."
            )
        else:
            user_content = f"Task: {prompt}"

        messages.append(HumanMessage(content=user_content))

        # Call the LLM with structured output
        result: OperatorOutput = structured_model.invoke(messages)

        return {
            "code": result.code,
            "pbt_payload": result.pbt.model_dump(),
        }

    return operator_node

