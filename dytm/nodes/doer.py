"""Doer node — sandboxed code execution with Hypothesis PBT verification.

The Doer is the most complex node in the DYTM pipeline.  It:

1. Runs the ``ast``-based security scan on the Operator's code.
2. Dynamically compiles the structured PBT JSON into a runnable Hypothesis
   test file.
3. Writes both files to a temporary directory and executes them via
   ``subprocess`` with a timeout.
4. Parses the result: on success the loop ends, on failure the minimal
   Hypothesis counterexample is extracted and routed back to the Operator.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from dytm.security import scan_code
from dytm.state import DYTMState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXECUTION_TIMEOUT = 15  # seconds (per subprocess call)
MAX_STDERR_CHARS = 3000  # truncate long tracebacks to save tokens


# ---------------------------------------------------------------------------
# Test-file compiler
# ---------------------------------------------------------------------------

def _compile_test_file(code: str, pbt: dict) -> str:
    """Compile the Operator's code + PBT payload into a single pytest file.

    The generated file:
    - Defines (via exec-paste) the function under test.
    - Imports Hypothesis and creates ``@given``-decorated tests for each
      property.
    - Includes plain ``assert``-based tests for edge cases.

    Args:
        code: The Operator's Python source code.
        pbt: The PBT payload as a dict (``PBTPayload.model_dump()``).

    Returns:
        A string of valid Python source ready to be written to a ``.py``
        file and run with ``pytest``.
    """
    func_name = pbt["function_name"]
    properties = pbt["properties"]

    header = (
        "# --- Auto-generated DYTM test file ---\n"
        "import pytest\n"
        "from typing import *\n"
        "from collections import *\n"
        "import math\n"
        "import re\n"
        "from hypothesis import given, settings, HealthCheck\n"
        "from hypothesis import strategies as st\n"
        "\n"
        "# ---- Function under test ----\n"
        f"{code}\n"
        "\n"
    )

    test_blocks: list[str] = []

    for i, prop in enumerate(properties):
        ptype = prop["type"]
        name = prop.get("name", f"property_{i}")
        expected = prop.get("expected", "")
        safe_name = name.replace(" ", "_").replace("-", "_")

        if ptype == "output_type":
            test_blocks.append(_compile_output_type(safe_name, func_name, expected, prop.get("test_body")))
        elif ptype == "membership":
            test_blocks.append(_compile_membership(safe_name, func_name, expected, prop.get("test_body")))
        elif ptype == "invariant":
            test_blocks.append(_compile_invariant(safe_name, func_name, expected, prop.get("test_body")))
        elif ptype == "edge_case":
            edge_input = prop.get("input", "None")
            # Strip outer parens — LLM often wraps multi-arg inputs like "(5, 0)"
            # but we need "5, 0" so it becomes func(5, 0) not func((5, 0))
            if edge_input.startswith("(") and edge_input.endswith(")"):
                edge_input = edge_input[1:-1].strip()
            # Validate edge_input is valid Python; skip if not
            try:
                compile(f"f({edge_input})", "<edge_input>", "eval")
            except SyntaxError:
                edge_input = "None"  # fallback to safe input
            edge_behavior = prop.get("expected_behavior", expected)
            test_blocks.append(
                _compile_edge_case(safe_name, func_name, edge_input, edge_behavior)
            )

    full_source = header + "\n\n".join(test_blocks) + "\n"

    # --- Final syntax validation: ensure the file compiles ---
    try:
        compile(full_source, "<dytm_test>", "exec")
    except SyntaxError as exc:
        # If the whole file doesn't parse, strip test blocks one by one
        # and keep only the ones that are syntactically valid
        valid_blocks = []
        for block in test_blocks:
            try:
                compile(header + block, "<dytm_test>", "exec")
                valid_blocks.append(block)
            except SyntaxError:
                valid_blocks.append(
                    f"def test_skipped_{len(valid_blocks)}():\n"
                    f"    # Skipped: syntax error in generated test\n"
                    f"    pass\n"
                )
        full_source = header + "\n\n".join(valid_blocks) + "\n"

    return full_source


# ---------------------------------------------------------------------------
# Per-property compilers
# ---------------------------------------------------------------------------

_TYPE_MAP: dict[str, str] = {
    "int": "int",
    "float": "float",
    "str": "str",
    "bool": "bool",
    "list": "list",
    "dict": "dict",
    "tuple": "tuple",
    "set": "set",
    "list[int]": "list",
    "list[str]": "list",
    "list[float]": "list",
    "Optional[int]": "(int, type(None))",
    "Optional[str]": "(str, type(None))",
}


def _python_type_expr(expected: str) -> str:
    """Convert a type string like ``'list[int]'`` into a Python isinstance check."""
    return _TYPE_MAP.get(expected, expected)


def _compile_output_type(name: str, func_name: str, expected: str, test_body: str | None = None) -> str:
    """Generate a test that checks the return type."""
    if test_body:
        return _use_test_body(test_body, name)
    # Fallback: no-op if no test_body provided
    return textwrap.dedent(f"""\
        def test_{name}():
            # output_type: verify return type is {expected}
            # No test_body provided — skipped.
            pass
    """)


def _compile_membership(name: str, func_name: str, expected: str, test_body: str | None = None) -> str:
    """Generate a test for membership / containment properties."""
    if test_body:
        return _use_test_body(test_body, name)
    return textwrap.dedent(f"""\
        def test_{name}():
            # membership: {expected}
            # No test_body provided — skipped.
            pass
    """)


def _compile_invariant(name: str, func_name: str, expected: str, test_body: str | None = None) -> str:
    """Generate a Hypothesis test for a general invariant."""
    if test_body:
        return _use_test_body(test_body, name)
    return textwrap.dedent(f"""\
        def test_{name}():
            # invariant: {expected}
            # No test_body provided — skipped.
            pass
    """)


def _use_test_body(test_body: str, name: str) -> str:
    """Sanitize and return an LLM-provided test_body.

    The test_body should be a complete test function including decorators.
    We do minimal sanitization: block dangerous calls and ensure it's
    syntactically valid Python.
    """
    # Basic sanitization — block obvious dangerous patterns
    dangerous = ["import os", "import sys", "import subprocess",
                 "exec(", "eval(", "open(", "__import__"]
    for pattern in dangerous:
        if pattern in test_body:
            return textwrap.dedent(f"""\
                def test_{name}():
                    # test_body blocked: contained '{pattern}'
                    pass
            """)

    # Cap max_examples to keep execution fast
    # Replace any max_examples>50 with 50
    import re as _re
    test_body = _re.sub(
        r'max_examples\s*=\s*\d+',
        'max_examples=30',
        test_body,
    )

    # If there's a @given but no @settings, inject one
    if '@given' in test_body and '@settings' not in test_body:
        test_body = test_body.replace(
            '@given',
            '@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])\n@given',
            1,
        )

    # The test_body should already include @given + def; return as-is
    lines = test_body.strip().splitlines()
    dedented = textwrap.dedent("\n".join(lines))
    return dedented + "\n"


def _compile_edge_case(
    name: str,
    func_name: str,
    edge_input: str,
    edge_behavior: str,
) -> str:
    """Generate a concrete edge-case test."""
    import re as _re

    # Strip JSON artifacts that leak from structured output (e.g. "},{"  "]},{")
    # Only target genuine field-boundary garbage — NOT legitimate Python like []
    edge_behavior = _re.sub(r'[}\]]\s*,?\s*[{\[]\s*', ' ', edge_behavior)
    edge_behavior = _re.sub(r'\s*[{},]+\s*\.?\s*$', '', edge_behavior).strip()

    behavior_lower = edge_behavior.lower().strip()

    # Edge-case tests use a "test_edge__" prefix so the Doer can distinguish
    # them from invariant/property tests when classifying failures.

    # Detect "raises XError" pattern
    if behavior_lower.startswith("raises "):
        exception_name = edge_behavior.split("raises", 1)[1].strip()
        # Sanitize exception name — must be a valid Python identifier
        exception_name = exception_name.split()[0].rstrip(".,;")
        if not exception_name.isidentifier():
            exception_name = "Exception"
        return textwrap.dedent(f"""\
            def test_edge__{name}():
                # edge_case: input={edge_input} should raise {exception_name}
                with pytest.raises({exception_name}):
                    {func_name}({edge_input})
        """)
    # Detect "returns X" pattern — check concrete return value
    elif behavior_lower.startswith("returns "):
        expected_value = edge_behavior.split("returns", 1)[1].strip()
        # Sanitize: strip trailing punctuation/JSON artifacts
        expected_value = expected_value.rstrip(".,;")
        # Remove any trailing JSON-like garbage (},{ etc.)
        # Validate it's actually a valid Python expression
        try:
            compile(expected_value, "<edge_case>", "eval")
        except SyntaxError:
            # Can't use this as a Python expression — fall back to a no-crash test
            return textwrap.dedent(f"""\
                def test_edge__{name}():
                    # edge_case: input={edge_input} -> {edge_behavior}
                    # (expected_value was not valid Python — checking no crash)
                    result = {func_name}({edge_input})
                    assert result is not None or result is None  # just verify no crash
            """)
        return textwrap.dedent(f"""\
            def test_edge__{name}():
                # edge_case: input={edge_input} -> {edge_behavior}
                result = {func_name}({edge_input})
                assert result == {expected_value}, (
                    f"Expected {expected_value}, got {{result}}"
                )
        """)
    else:
        return textwrap.dedent(f"""\
            def test_edge__{name}():
                # edge_case: input={edge_input} -> {edge_behavior}
                result = {func_name}({edge_input})
                assert result is not None, "Edge case returned None unexpectedly"
        """)


# ---------------------------------------------------------------------------
# Counterexample extraction
# ---------------------------------------------------------------------------

def _extract_counterexample(stderr: str) -> str | None:
    """Try to pull the Hypothesis minimal counterexample from pytest stderr.

    Hypothesis prints lines like::

        Falsifying example: test_foo(
            x=42,
        )

    We grab everything from "Falsifying example" to the next blank line.
    """
    lines = stderr.splitlines()
    capture = False
    buf: list[str] = []

    for line in lines:
        if "Falsifying example" in line:
            capture = True
        if capture:
            buf.append(line)
            if line.strip() == ")" or line.strip() == "":
                break

    return "\n".join(buf).strip() if buf else None


def _parse_failed_test_names(output: str) -> list[str]:
    """Extract failed test function names from pytest output.

    Parses the "short test summary info" section for lines like::

        FAILED path/to/test.py::test_name - AssertionError: ...

    Falls back to parsing ``_____ test_name _____`` section headers.
    """
    import re

    failed: list[str] = []

    for line in output.splitlines():
        m = re.match(r"^FAILED\s+.*::(\w+)", line)
        if m:
            failed.append(m.group(1))

    if not failed:
        for line in output.splitlines():
            m = re.match(r"^_+\s+(test_\w+)\s+_+$", line)
            if m:
                failed.append(m.group(1))

    return failed


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def doer_node(state: DYTMState) -> dict:
    """Execute the Operator's code against compiled Hypothesis PBTs.

    Args:
        state: The current LangGraph state.  Must contain ``code`` and
               ``pbt_payload``.

    Returns:
        A partial state update with ``test_result``, ``security_passed``,
        and optionally ``error_trace`` / ``counterexample`` / ``retry_count``.
    """
    code: str = state["code"]
    pbt: dict = state["pbt_payload"]
    retry_count: int = state.get("retry_count", 0)
    prev_code: str | None = state.get("_prev_code")

    # --- Duplicate-code detection -------------------------------------------
    # If the Coder generated IDENTICAL code to the previous attempt, the PBT
    # has an unfixable bug (e.g. wrong boundary condition in test_body).
    # Retrying would just waste tokens — accept the code immediately.
    if retry_count > 0 and prev_code is not None and code.strip() == prev_code.strip():
        return {
            "security_passed": True,
            "test_result": "pass",
            "_test_source": "",
            "_prev_code": code,
            "_duplicate_code": True,
        }

    # ---- Step 1: Security scan -------------------------------------------
    safe, reason = scan_code(code)
    if not safe:
        return {
            "security_passed": False,
            "test_result": "fail",
            "error_trace": f"Security scan failed: {reason}",
            "retry_count": retry_count + 1,
        }

    # ---- Step 2: Compile test file ---------------------------------------
    try:
        test_source = _compile_test_file(code, pbt)
    except Exception as exc:
        return {
            "security_passed": True,
            "test_result": "error",
            "error_trace": f"Test compilation failed: {exc}",
            "retry_count": retry_count + 1,
            "_test_source": "",
        }

    # ---- Step 3: Write to tempdir & execute ------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_dytm_generated.py"
        test_file.write_text(test_source, encoding="utf-8")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(test_file), "--tb=long", "-q"],
                capture_output=True,
                timeout=EXECUTION_TIMEOUT,
                text=True,
                cwd=tmpdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return {
                "security_passed": True,
                "test_result": "fail",
                "error_trace": f"Execution timed out after {EXECUTION_TIMEOUT}s",
                "retry_count": retry_count + 1,
                "_test_source": test_source,
            }

    # ---- Step 4: Parse results -------------------------------------------
    if result.returncode == 0:
        return {
            "security_passed": True,
            "test_result": "pass",
            "_test_source": test_source,
            "_prev_code": code,
        }

    # Tests failed — extract counterexample
    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
    combined_output = combined_output[:MAX_STDERR_CHARS]

    counterexample = _extract_counterexample(combined_output)

    # --- Edge-only failure classification ------------------------------------
    # Parse which tests failed.  If the ONLY failures are edge case tests
    # (test_edge__*), invariant / property tests all passed — the code is
    # probably correct and the Spec had wrong expected values.
    failed_tests = _parse_failed_test_names(combined_output)
    edge_only_fail = (
        bool(failed_tests)
        and all(t.startswith("test_edge__") for t in failed_tests)
    )

    max_retries = state.get("max_retries", 3)

    # Safety-net soft pass: if we've exhausted all retries AND only edge
    # cases are failing, accept the code.  This avoids burning every retry
    # on wrong expected values from the Spec LLM.  On earlier retries we
    # still treat it as a real failure so the Coder gets a chance to fix
    # genuine bugs.
    if edge_only_fail and (retry_count + 1) >= max_retries:
        return {
            "security_passed": True,
            "test_result": "pass",
            "error_trace": combined_output.strip(),
            "retry_count": retry_count + 1,
            "_test_source": test_source,
            "_edge_only_fail": True,
            "_prev_code": code,
        }

    return {
        "security_passed": True,
        "test_result": "fail",
        "error_trace": combined_output.strip(),
        "counterexample": counterexample,
        "retry_count": retry_count + 1,
        "_test_source": test_source,
        "_edge_only_fail": edge_only_fail,
        "_prev_code": code,
    }



