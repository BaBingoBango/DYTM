"""Phase 2 smoke tests — security scanner, translator, approval, and doer."""

from dytm.security import scan_code
from dytm.nodes.translator import translator_node
from dytm.nodes.approval import approval_node
from dytm.nodes.doer import doer_node

# =========================================================================
# 1. Security scanner
# =========================================================================

def test_security_safe_code():
    safe, reason = scan_code("def add(a, b): return a + b")
    assert safe is True
    assert reason is None
    print("  ✓ Safe code passes security scan")


def test_security_dangerous_import():
    safe, reason = scan_code("import os\nos.system('rm -rf /')")
    assert safe is False
    assert "os" in reason
    print("  ✓ Dangerous import (os) is blocked")


def test_security_dangerous_call():
    safe, reason = scan_code("exec('print(1)')")
    assert safe is False
    assert "exec" in reason
    print("  ✓ Dangerous call (exec) is blocked")


def test_security_from_import():
    safe, reason = scan_code("from subprocess import run")
    assert safe is False
    assert "subprocess" in reason
    print("  ✓ Dangerous from-import (subprocess) is blocked")


def test_security_syntax_error():
    safe, reason = scan_code("def foo(:")
    assert safe is False
    assert "SyntaxError" in reason
    print("  ✓ Syntax errors are caught")


# =========================================================================
# 2. Translator (Jinja2 → CNL)
# =========================================================================

SAMPLE_PBT = {
    "function_name": "longest_string",
    "properties": [
        {"name": "returns_a_string", "type": "output_type", "expected": "str"},
        {"name": "result_is_in_input", "type": "membership", "expected": "output in input_list"},
        {"name": "no_string_is_longer", "type": "invariant", "expected": "len(output) >= len(s) for all s in input_list"},
        {"name": "empty_list_raises", "type": "edge_case", "expected": "raises ValueError", "input": "[]", "expected_behavior": "raises ValueError"},
    ],
}


def test_translator_renders_cnl():
    state = {"pbt_payload": SAMPLE_PBT}
    result = translator_node(state)
    cnl = result["cnl_text"]

    assert "longest_string" in cnl
    assert "MUST return a value of type: str" in cnl
    assert "MUST satisfy: output in input_list" in cnl
    assert "invariant MUST hold" in cnl
    assert "EDGE CASE" in cnl
    assert "raises ValueError" in cnl
    print("  ✓ Translator renders CNL with all property types")
    print(f"    (CNL is {len(cnl)} chars)")


# =========================================================================
# 3. Approval (eval mode auto-approve)
# =========================================================================

def test_approval_eval_mode():
    state = {"mode": "eval", "cnl_text": "some contract"}
    result = approval_node(state)
    assert result["approved"] is True
    print("  ✓ Approval auto-approves in eval mode")


# =========================================================================
# 4. Doer (security + test compilation + execution)
# =========================================================================

def test_doer_rejects_unsafe_code():
    state = {
        "code": "import os\ndef foo(): return os.getcwd()",
        "pbt_payload": {"function_name": "foo", "properties": []},
        "retry_count": 0,
    }
    result = doer_node(state)
    assert result["test_result"] == "fail"
    assert result["security_passed"] is False
    assert "Security scan failed" in result["error_trace"]
    print("  ✓ Doer rejects unsafe code before execution")


def test_doer_passes_good_code():
    state = {
        "code": "def add(a, b):\n    return a + b",
        "pbt_payload": {
            "function_name": "add",
            "properties": [
                {
                    "name": "returns_correct_type",
                    "type": "edge_case",
                    "expected": "returns 5",
                    "input": "2, 3",
                    "expected_behavior": "returns 5",
                },
            ],
        },
        "retry_count": 0,
    }
    result = doer_node(state)
    assert result["test_result"] == "pass", f"Expected pass, got: {result}"
    assert result["security_passed"] is True
    print("  ✓ Doer passes good code with edge-case test")


def test_doer_catches_bad_code():
    # Function that raises on the edge case input
    state = {
        "code": "def divide(a, b):\n    return a / b",
        "pbt_payload": {
            "function_name": "divide",
            "properties": [
                {
                    "name": "zero_division",
                    "type": "edge_case",
                    "expected": "raises ZeroDivisionError",
                    "input": "1, 0",
                    "expected_behavior": "raises ZeroDivisionError",
                },
            ],
        },
        "retry_count": 0,
    }
    result = doer_node(state)
    assert result["test_result"] == "pass", f"Expected pass for raises test, got: {result}"
    assert result["security_passed"] is True
    print("  ✓ Doer verifies exception-raising edge case")


# =========================================================================
# Run all
# =========================================================================

if __name__ == "__main__":
    print()
    print("Phase 2 Smoke Tests")
    print("=" * 50)

    print("\n1. Security Scanner")
    test_security_safe_code()
    test_security_dangerous_import()
    test_security_dangerous_call()
    test_security_from_import()
    test_security_syntax_error()

    print("\n2. Translator")
    test_translator_renders_cnl()

    print("\n3. Approval")
    test_approval_eval_mode()

    print("\n4. Doer")
    test_doer_rejects_unsafe_code()
    test_doer_passes_good_code()
    test_doer_catches_bad_code()

    print()
    print("Phase 2 OK ✅")

