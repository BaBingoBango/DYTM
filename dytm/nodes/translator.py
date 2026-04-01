"""Translator node — deterministic JSON → CNL rendering.

This node contains **zero LLM calls**.  It loads a Jinja2 template and
renders the structured PBT payload into a human-readable Controlled Natural
Language (CNL) contract.  Because the mapping is purely template-based, the
CNL is a mathematically faithful, **lossless** description of the PBTs that
will actually run — including both the human-readable property descriptions
and the executable test implementations.

The CNL is the single document the Executive reviews.  Nothing is hidden.
"""

from __future__ import annotations

from pathlib import Path

import re

from jinja2 import Environment, FileSystemLoader

from dytm.state import DYTMState

# ---------------------------------------------------------------------------
# Template setup (loaded once at import time)
# ---------------------------------------------------------------------------
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATE_DIR)),
    trim_blocks=True,
    lstrip_blocks=True,
)
_cnl_template = _env.get_template("cnl.jinja2")

# JSON artifact pattern — strips stray },{ and similar from LLM output
_JSON_ARTIFACTS = re.compile(r'[{}\[\]]+[,.]?')


def _clean_expected(value: str) -> str:
    """Strip JSON artifacts from a property's expected field.

    Only targets genuine JSON field-boundary garbage (e.g. ``},{`` or ``],[``)
    that can leak from LLM structured output.  Does NOT strip legitimate
    Python bracket expressions like ``[]``, ``[1, 2]``, or ``(0, 1)``.
    """
    cleaned = value.strip()
    # Remove JSON field boundary artifacts: }...{ or ]...[
    # e.g. "returns hello},{next field" → "returns hello next field"
    cleaned = re.sub(r'[}\]]\s*,?\s*[{\[]\s*', ' ', cleaned)
    # Remove trailing lone { } , (but NOT [ or ] which may be valid Python)
    cleaned = re.sub(r'\s*[{},]+\s*\.?\s*$', '', cleaned)
    return cleaned.strip().rstrip('.') + '.' if cleaned else value


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def translator_node(state: DYTMState) -> dict:
    """Render PBT payload → CNL contract string.

    Reads ``state["pbt_payload"]`` (a dict) and writes the rendered CNL
    text to ``state["cnl_text"]``.  Cleans JSON artifacts from LLM output
    before rendering.

    Args:
        state: The current LangGraph state.

    Returns:
        A partial state update with ``cnl_text`` set.
    """
    pbt = state["pbt_payload"]

    # Deep-copy and clean properties before rendering
    clean_props = []
    for prop in pbt["properties"]:
        p = dict(prop)
        if "expected" in p:
            p["expected"] = _clean_expected(p["expected"])
        if "expected_behavior" in p and p["expected_behavior"]:
            p["expected_behavior"] = _clean_expected(p["expected_behavior"])
        clean_props.append(p)

    cnl_text = _cnl_template.render(
        function_name=pbt["function_name"],
        properties=clean_props,
    )

    return {"cnl_text": cnl_text}

