"""LangGraph StateGraph definition for the DYTM verification loop.

New pipeline (v2):
    Spec → Translator → Approval → Coder → Doer
                ↑ (reject)              ↑ (fail)
                └──────────┘            └──────┘

Key design decisions:
- Spec generates ONLY the PBT contract (no code)
- Approval is an Executive LLM (or human) reviewing the CNL
- Coder generates code on a FRESH context given approved PBTs
- On test failure, only Coder retries — the contract is locked
- On contract rejection, Spec retries with Executive feedback
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from dytm.nodes.approval import make_approval_node
from dytm.nodes.coder import make_coder_node
from dytm.nodes.doer import doer_node
from dytm.nodes.multi_spec import make_multi_spec_node
from dytm.nodes.spec import make_spec_node
from dytm.nodes.translator import translator_node
from dytm.state import DYTMState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_after_approval(state: DYTMState) -> str:
    """Route after the Approval node.

    - Approved → coder (generate code against the locked contract)
    - Rejected + retries left → spec (revise the contract)
    - Rejected + out of retries → coder (proceed with best-effort contract)
    """
    if state.get("approved"):
        return "coder"

    spec_retries = state.get("spec_retry_count", 0)
    max_spec_retries = state.get("max_spec_retries", 2)
    if spec_retries >= max_spec_retries:
        # Exhausted revision attempts — proceed with best-effort contract
        # rather than ending with no code at all
        return "coder"

    return "spec"


def _route_after_doer(state: DYTMState) -> str:
    """Route after the Doer node.

    - Tests pass → END
    - Max retries exceeded → END
    - Tests fail → coder (retry code generation, contract stays locked)
    """
    if state.get("test_result") == "pass":
        return END
    if state.get("retry_count", 0) >= state.get("max_retries", 4):
        return END
    return "coder"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    model: BaseChatModel,
    executive_model: BaseChatModel | None = None,
    spec_model: BaseChatModel | None = None,
    multi_spec_k: int = 1,
) -> CompiledStateGraph:
    """Build and compile the DYTM verification loop (v2).

    Args:
        model: Default LLM for Spec + Coder nodes.
        executive_model: Optional separate LLM for the Executive approval.
                         If None, eval mode auto-approves.
        spec_model: Optional separate LLM for the Spec node only.
                    If provided, Spec uses this model while Coder uses
                    ``model``.  This enables cross-model experiments
                    (e.g. strong Spec → weak Coder) to test whether DYTM
                    adds value beyond simply upgrading the model.
        multi_spec_k: Number of spec interpretations to sample.  If > 1,
                      uses the Multi-Spec node (temperature diversity +
                      example-consistency filtering) instead of single Spec.

    Returns:
        A compiled LangGraph ``StateGraph`` ready to be invoked.
    """
    base_spec_model = spec_model or model
    if multi_spec_k > 1:
        spec_node = make_multi_spec_node(base_spec_model, k=multi_spec_k)
    else:
        spec_node = make_spec_node(base_spec_model)
    coder_node_fn = make_coder_node(model)
    approval_node = make_approval_node(executive_model)

    graph = StateGraph(DYTMState)

    # -- Add nodes ----------------------------------------------------------
    graph.add_node("spec", spec_node)
    graph.add_node("translator", translator_node)
    graph.add_node("approval", approval_node)
    graph.add_node("coder", coder_node_fn)
    graph.add_node("doer", doer_node)

    # -- Set entry point ----------------------------------------------------
    graph.set_entry_point("spec")

    # -- Linear edges -------------------------------------------------------
    graph.add_edge("spec", "translator")
    graph.add_edge("translator", "approval")
    graph.add_edge("coder", "doer")

    # -- Conditional edges --------------------------------------------------
    graph.add_conditional_edges("approval", _route_after_approval)
    graph.add_conditional_edges("doer", _route_after_doer)

    return graph.compile()

