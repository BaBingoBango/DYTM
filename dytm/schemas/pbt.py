"""Pydantic models for DYTM's structured PBT payloads.

These schemas enforce the contract between the Operator (LLM) and the rest
of the pipeline.  The Operator's structured output must conform to
``OperatorOutput``, which is passed directly to
``model.with_structured_output(OperatorOutput)``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from typing import Literal


# ---------------------------------------------------------------------------
# Property types the Operator can emit
# ---------------------------------------------------------------------------
PropertyType = Literal["output_type", "membership", "invariant", "edge_case"]


class PBTProperty(BaseModel):
    """A single property-based test property.

    Each property maps 1:1 to a Hypothesis test that the Doer will compile
    and execute.

    Attributes:
        name: A short, unique identifier for the property
              (e.g. ``"returns_a_string"``).
        type: The category of property — determines how the Doer compiles it
              into a Hypothesis test and how the Translator renders it as CNL.
        expected: A human-/machine-readable description of the expected
                  behaviour (e.g. ``"list[int]"`` for ``output_type``, or an
                  invariant expression for ``invariant``).
        input: (edge_case only) The specific input that triggers the edge
               case.  Serialised as a JSON-compatible string.
        expected_behavior: (edge_case only) What should happen for the edge
                          case input (e.g. ``"raises ValueError"``).
    """

    name: str = Field(
        ...,
        description="Short unique identifier for this property",
    )
    type: PropertyType = Field(
        ...,
        description="Category of property: output_type, membership, invariant, or edge_case",
    )
    expected: str = Field(
        ...,
        description="Expected behaviour or type expression",
    )
    input: Optional[str] = Field(
        default=None,
        description="(edge_case only) Specific input triggering the edge case",
    )
    expected_behavior: Optional[str] = Field(
        default=None,
        description="(edge_case only) What should happen for the edge-case input",
    )
    test_body: Optional[str] = Field(
        default=None,
        description=(
            "Complete executable Python test function for this property "
            "(including @given decorator and def line). Required for "
            "output_type, membership, and invariant properties. The Doer "
            "pastes this directly into the test file."
        ),
    )


class PBTPayload(BaseModel):
    """The full suite of property-based test properties for a single function.

    Attributes:
        function_name: The name of the function under test.
        properties: Ordered list of properties the Doer must verify.
    """

    function_name: str = Field(
        ...,
        description="Name of the Python function under test",
    )
    properties: list[PBTProperty] = Field(
        ...,
        description="Ordered list of PBT properties to verify",
    )


class SpecOutput(BaseModel):
    """Structured output from the Spec node — PBT properties only, NO code.

    The Spec node generates a verification contract (what the code must do)
    *before* any code is written.  This enforces separation of concerns:
    the specification is approved independently of the implementation.

    Attributes:
        pbt: The structured PBT payload describing the verification
             properties.
    """

    pbt: PBTPayload = Field(
        ...,
        description="Structured PBT payload with verification properties",
    )


class CoderOutput(BaseModel):
    """Structured output from the Coder node — code only, given approved PBTs.

    The Coder node receives the approved PBT contract and generates code
    on a fresh context (no memory of the spec generation).

    Attributes:
        code: The complete Python source code of the generated function.
    """

    code: str = Field(
        ...,
        description="Complete Python source code of the generated function",
    )


class OperatorOutput(BaseModel):
    """Legacy: combined code + PBT output (kept for backward compat).

    Attributes:
        code: The complete Python source code of the generated function.
        pbt: The structured PBT payload describing the verification
             properties.
    """

    code: str = Field(
        ...,
        description="Complete Python source code of the generated function",
    )
    pbt: PBTPayload = Field(
        ...,
        description="Structured PBT payload with verification properties",
    )

