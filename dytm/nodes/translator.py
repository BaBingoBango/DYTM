"""Translator node — deterministic JSON → CNL rendering.

This node contains **zero LLM calls**.  It loads a Jinja2 template and
renders the structured PBT payload into a human-readable Controlled Natural
Language (CNL) contract.  Because the mapping is purely template-based, the
CNL is a mathematically faithful description of the PBTs that will actually
run.
"""

from __future__ import annotations

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def translator_node(state: DYTMState) -> dict:
    """Render PBT payload → CNL contract string.

    Reads ``state["pbt_payload"]`` (a dict) and writes the rendered CNL
    text to ``state["cnl_text"]``.

    Args:
        state: The current LangGraph state.

    Returns:
        A partial state update with ``cnl_text`` set.
    """
    pbt = state["pbt_payload"]

    cnl_text = _cnl_template.render(
        function_name=pbt["function_name"],
        properties=pbt["properties"],
    )

    return {"cnl_text": cnl_text}

