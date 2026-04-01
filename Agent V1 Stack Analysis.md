# DYTM V1 Tech Stack — Expert Recommendation

> **Design Principle for V1:** Every dependency must serve the evaluation. If it doesn't directly help you measure whether DYTM improves pass@1 on benchmarks, it's scope creep. Ship the paper, *then* productionize.

---

## ✅ KEEP — LangGraph (Orchestration)
**Verdict: Right choice, no change.**

LangGraph is the clear winner here. Your architecture is a **stateful, cyclical graph** with conditional routing on failure — that's literally what LangGraph was built for. The 4-node design (Operator → Translator → Approval → Doer → loop back) maps 1:1 to LangGraph's `StateGraph` with conditional edges.

Alternatives considered:
- **CrewAI / AutoGen** — too high-level, less control over routing logic
- **Raw Python loops** — you'd reinvent state management and checkpointing
- **DSPy** — different paradigm (prompt optimization), not a fit

---

## ⚠️ MODIFY — LLM Backend (OpenAI → Model-Agnostic)
**Verdict: Use LangChain's chat model abstraction. Primary eval on GPT-4o-mini, but design for swappability.**

GPT-4o-mini is a fine primary model, but for an academic paper proving DYTM improves on SOTA, **reviewers will ask: "Does this generalize beyond one model?"** You need results on at least 2-3 models.

**Recommendation:**
- Use `langchain_core`'s `BaseChatModel` interface (LangGraph gives you this for free)
- Primary eval: **GPT-4o-mini** (cheap, fast, good structured output)
- Secondary eval: **Claude 3.5 Sonnet** (strong coder, different architecture)
- Stretch: A local model via **Ollama** (e.g., Qwen2.5-Coder or DeepSeek) — shows DYTM helps *weak* models the most, which is a killer finding for a paper

Being model-agnostic costs you almost nothing in implementation but dramatically strengthens your results section.

---

## ✅ KEEP — Pydantic (Schema Enforcement)
**Verdict: Right choice, no change.**

Pydantic v2 for defining the structured PBT JSON schema is industry standard. Pairs perfectly with OpenAI's `response_format` and LangChain's `with_structured_output()`. No alternative worth considering.

---

## ✅ KEEP — Hypothesis (PBT Engine)
**Verdict: Right choice, no change.**

Hypothesis is the gold standard for Python PBT. The killer feature for DYTM is **automatic counterexample shrinking** — when a test fails, Hypothesis finds the *minimal* failing input, which is extremely token-efficient when fed back to the Operator for retry. That's a huge advantage over hand-written assertions.

Keep `hypothesis-jsonschema` for dynamically compiling JSON PBT specs into `@given` strategies.

---

## 🔴 CHANGE — CNL Engine (APE → Jinja2 Templates)
**Verdict: APE is overkill and risky for V1. Use deterministic template rendering instead.**

This is my strongest recommendation. APE (Attempto Parsing Engine) is:
- **Old Prolog-based software** with spotty maintenance
- Requires a **Docker container** just to run
- Adds infrastructure complexity that has nothing to do with your eval
- ACE syntax is rigid and hard to debug when things go wrong

Here's the key insight: **your own Agent Readme already describes the right approach** — *"maps it 1:1 to a Controlled Natural Language string using pre-defined text templates."* That's not APE. That's template rendering.

**Recommendation: Jinja2 templates**
```
# Example: JSON PBT payload → CNL via template
Input:  {"property": "output_type", "expected": "list", "of": "int"}
Output: "The function MUST return a list of integers."
```

- **Equally deterministic** — no LLM involved, pure mapping
- **Trivial to implement** — a few dozen templates cover HumanEval's scope
- **Easier for reviewers to reproduce** — no Docker dependency
- **Faster to iterate** — edit a template vs. debug Prolog

> **Future work (paper §8):** "In production, the template engine could be replaced with a formal CNL system such as APE to provide DRS-backed semantic verification of the contract itself."

---

## 🔴 CHANGE — Sandbox (E2B → Local Docker or subprocess)
**Verdict: E2B is a production tool. For batch eval, it's slow and expensive.**

Running HumanEval (164 problems) × multiple models × iterative retries = **hundreds to thousands of executions**. With E2B:
- ~$0.01-0.05 per sandbox session adds up
- Network latency on every execution
- External dependency that can go down during your eval run

**Recommendation for V1:** Use Python's **`subprocess`** with **`tempfile`** and **resource limits** (`timeout`, `ulimit`). This is what virtually every HumanEval benchmarking paper does. The security risk is negligible in a controlled eval environment — you're running known benchmark problems, not arbitrary user code.

```python
import subprocess, tempfile
result = subprocess.run(
    ["python", test_file.name],
    capture_output=True, timeout=30, text=True
)
```

If you want slightly more isolation, use **Docker containers** locally — you'd just need a simple `python:3.11-slim` image with Hypothesis pre-installed. But honestly, subprocess is fine for V1.

> **Save E2B for Phase 2** when you're running untrusted user code in production.

---

## ✅ KEEP — Python `ast` (Code Security)
**Verdict: Right choice, lightweight and built-in.**

Use `ast.parse()` + `ast.NodeVisitor` to scan for dangerous imports (`os`, `subprocess`, `shutil`, etc.) before execution. It's zero-dependency and fast.

---

## 🔴 ADD — Observability (LangSmith)
**Verdict: Not in your notes, but essential for debugging and paper figures.**

When your agent loop fails on problem #87, you need to know *why*. Was it the Operator generating bad PBTs? The Hypothesis test being too strict? A sandbox error?

**Recommendation: LangSmith** (free tier)
- Built-in LangGraph integration (one line of config)
- Full trace of every LLM call, every node transition, every retry
- You can export traces for paper figures showing the verification loop in action

Also add simple **CSV/JSON logging** for eval metrics (pass/fail, retry count, tokens used, wall-clock time per problem).

---

## ⚠️ MODIFY — Eval Benchmarks (HumanEval + MBPP, defer GAIA)
**Verdict: HumanEval alone is too narrow. Add MBPP. Defer GAIA.**

- **HumanEval** (164 problems) — the standard. You must have this.
- **MBPP** (974 problems) — broader, more diverse. Strengthens generalizability claims significantly. Easy to add since the format is similar.
- **GAIA** — complex multi-step reasoning tasks. Compelling but a *very* different eval setup (web browsing, file manipulation). Save for a follow-up paper or §8 future work.

---

## Final Recommended Stack

| Layer | Tool | Why |
|---|---|---|
| **Language** | Python 3.11+ | — |
| **Orchestration** | LangGraph | Stateful cyclical graphs with conditional edges |
| **LLM Interface** | LangChain chat models | Model-agnostic: swap OpenAI / Anthropic / Ollama |
| **Primary Eval Model** | GPT-4o-mini | Cheap, fast, great structured output |
| **Schema Enforcement** | Pydantic v2 | Structured PBT JSON validation |
| **PBT Engine** | Hypothesis + hypothesis-jsonschema | Gold standard, counterexample shrinking |
| **CNL Translation** | Jinja2 templates | Deterministic, zero infrastructure, easy to debug |
| **Code Security** | Python `ast` | Lightweight, built-in |
| **Sandbox** | `subprocess` + `tempfile` | Fast, free, standard for eval papers |
| **Observability** | LangSmith (free tier) | Trace agent loops, debug failures |
| **Eval Benchmarks** | HumanEval + MBPP | Standard + breadth |
| **Metrics Logging** | CSV/JSON | pass@1, retry count, tokens, latency |

### Key Dependencies (requirements.txt)
```
langgraph
langchain-openai
langchain-anthropic
pydantic>=2.0
hypothesis
hypothesis-jsonschema
jinja2
langsmith
```

### What's NOT in V1 (and why)
| Deferred | Reason |
|---|---|
| APE / Attempto | Infrastructure overhead, not needed to prove the concept |
| E2B | Cost/latency overhead for batch evals; save for production |
| GAIA benchmark | Different eval paradigm; save for paper #2 |
| Docker sandbox | subprocess is sufficient for controlled eval |
| FastAPI / web UI | No user-facing product in V1, just an eval harness |

---

*This stack minimizes time-to-first-eval-result while maximizing the strength of your paper's claims.*
