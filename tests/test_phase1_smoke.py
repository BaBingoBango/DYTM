"""Phase 1 smoke test."""
from dytm.schemas.pbt import PBTPayload, PBTProperty, OperatorOutput
from dytm.state import DYTMState

# Instantiate with the Stack Plan example
payload = PBTPayload(
    function_name="longest_string",
    properties=[
        PBTProperty(name="returns_a_string", type="output_type", expected="str"),
        PBTProperty(name="result_is_in_input", type="membership", expected="output in input_list"),
        PBTProperty(name="no_string_is_longer", type="invariant", expected="len(output) >= len(s) for all s in input_list"),
        PBTProperty(name="empty_list_raises", type="edge_case", expected="raises ValueError", input="[]", expected_behavior="raises ValueError"),
    ],
)

# Wrap in OperatorOutput
output = OperatorOutput(
    code="def longest_string(lst): return max(lst, key=len)",
    pbt=payload,
)

# Verify state can be constructed
state: DYTMState = {
    "prompt": "Write a function that returns the longest string",
    "code": output.code,
    "pbt_payload": output.pbt.model_dump(),
    "cnl_text": "",
    "approved": False,
    "test_result": "",
    "retry_count": 0,
    "max_retries": 3,
    "counterexample": None,
    "error_trace": None,
    "mode": "eval",
    "security_passed": False,
}

print(f"PBTPayload: {payload.function_name} with {len(payload.properties)} properties")
print(f"OperatorOutput: code={len(output.code)} chars, pbt={output.pbt.function_name}")
print(f"DYTMState: mode={state['mode']}, max_retries={state['max_retries']}")
print()
print("Phase 1 OK ✅")

