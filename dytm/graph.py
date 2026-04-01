"""LangGraph StateGraph definition for the DYTM verification loop.

Wires the four nodes — Operator → Translator → Approval → Doer — into a
cyclical graph with conditional edges for retry logic and approval gating.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from dytm.nodes.approval import approval_node
from dytm.nodes.doer import doer_node
from dytm.nodes.operator import make_operator_node
from dytm.nodes.translator import translator_node
from dytm.state import DYTMState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_approval(state: DYTMState) -> str:
    """Route after the Approval node.

    - Approved → doer (run the tests)
    - Rejected → operator (regenerate with feedback)
    """
    if state.get("approved"):
        return "doer"
    return "operator"


def _route_after_doer(state: DYTMState) -> str:
    """Route after the Doer node.

    - Tests pass → END
    - Max retries exceeded → END
    - Tests fail → operator (retry with counterexample)
    """
    if state.get("test_result") == "pass":
        return END
    if state.get("retry_count", 0) >= state.get("max_retries", 3):
        return END
    return "operator"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(model: BaseChatModel) -> CompiledStateGraph:
    """Build and compile the DYTM verification loop.

    Args:
        model: Any LangChain chat model to power the Operator node.

    Returns:
        A compiled LangGraph ``StateGraph`` ready to be invoked.
    """
    operator_node = make_operator_node(model)

    graph = StateGraph(DYTMState)

    # -- Add nodes ----------------------------------------------------------
    graph.add_node("operator", operator_node)
    graph.add_node("translator", translator_node)
    graph.add_node("approval", approval_node)
    graph.add_node("doer", doer_node)

    # -- Set entry point ----------------------------------------------------
    graph.set_entry_point("operator")

    # -- Linear edges -------------------------------------------------------
    graph.add_edge("operator", "translator")
    graph.add_edge("translator", "approval")

    # -- Conditional edges --------------------------------------------------
    graph.add_conditional_edges("approval", _route_after_approval)
    graph.add_conditional_edges("doer", _route_after_doer)

    return graph.compile()



