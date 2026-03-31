# DYTM V1 — Stack Plan

> **Goal:** Build an evaluation harness that proves DYTM's verification loop improves LLM pass@1 on code-generation benchmarks vs. a baseline (no verification) run.

---

## Architecture Overview

```
Executive (Simulated)
        │
        ▼
┌──────────────────────────────────────────────────┐
│                  LangGraph StateGraph             │
│                                                   │
│   ┌─────────────────┐    ┌──────────────────┐     │
│   │  1. Operator     │───▶│  2. Translator   │     │
│   │  (LLM: Generate  │    │  (Jinja2: JSON   │     │
│   │   Code + PBT JSON)│    │   → CNL string)  │     │
│   └─────────────────┘    └───────┬──────────┘     │
│           ▲                      │                 │
│           │ retry w/             ▼                 │
│           │ counterexample  ┌──────────────────┐   │
│           │                 │  3. Approval      │   │
│           │                 │  (Auto-approve    │   │
│   ┌───────┴──────────┐     │   for eval mode)  │   │
│   │  4. Doer          │◀───└──────────────────┘   │
│   │  (subprocess:     │                           │
│   │   run code +      │                           │
│   │   Hypothesis PBTs)│                           │
│   └──────────────────┘                            │
└──────────────────────────────────────────────────┘
        │
        ▼
  Eval Metrics (CSV/JSON)
```

---

## User Journey — Step by Step

Here's exactly what happens when a single problem (e.g., HumanEval #12) flows through the system, touching every component:

### Step 0: Setup
The **eval harness** (`eval/run_dytm.py`) loads a benchmark problem from **HumanEval/MBPP** and formats it as a prompt:
```
"Write a Python function that takes a list of strings and returns
the longest string. If multiple strings share the max length,
return the first one."
```
This prompt plays the role of the **Executive's intent**.

### Step 1: Operator Generates Code + PBT JSON
**Components touched:** `LangGraph` → `LangChain (model-agnostic)` → `Pydantic v2`

LangGraph enters the first node (`nodes/operator.py`). It calls the configured LLM (e.g., GPT-4o-mini via `langchain-openai`) with a system prompt instructing it to produce two things:

1. **The solution code** (a Python function)
2. **A structured PBT payload** — a JSON object conforming to the `Pydantic` schema

The LLM responds with structured output (via `with_structured_output()`), and Pydantic validates the JSON. If the schema is violated, Pydantic raises immediately — no bad data enters the pipeline.

Example PBT JSON payload:
```json
{
  "function_name": "longest_string",
  "properties": [
    {
      "name": "returns_a_string",
      "type": "output_type",
      "expected": "str"
    },
    {
      "name": "result_is_in_input",
      "type": "membership",
      "expected": "output in input_list"
    },
    {
      "name": "no_string_is_longer",
      "type": "invariant",
      "expected": "len(output) >= len(s) for all s in input_list"
    },
    {
      "name": "empty_list_raises",
      "type": "edge_case",
      "input": [],
      "expected_behavior": "raises ValueError"
    }
  ]
}
```

The generated code and validated PBT JSON are written into the **LangGraph state** (`TypedDict`).

### Step 2: Deterministic Translation (JSON → CNL)
**Components touched:** `Jinja2 Templates`

LangGraph enters the second node (`nodes/translator.py`). This node contains **zero LLM calls** — it is the deterministic firewall.

The Jinja2 template engine reads the PBT JSON and renders each property into a plain-English CNL contract using pre-defined templates:

```
VERIFICATION CONTRACT — longest_string
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. The function MUST return a value of type: str.
2. The returned value MUST be a member of the input list.
3. The following invariant MUST hold: the length of the output
   is greater than or equal to the length of every string in
   the input list.
4. EDGE CASE: When given an empty list, the function MUST raise
   a ValueError.
```

This CNL string is stored in the LangGraph state. Because it was produced by template rendering (not an LLM), it is a **mathematically faithful** description of the PBTs that will actually run.

### Step 3: Executive Approval
**Components touched:** `LangGraph` (conditional edge)

LangGraph enters the third node (`nodes/approval.py`).

- **In eval mode** (V1): Auto-approved. The node rubber-stamps the CNL and moves on. This isolates the variable being tested — we're measuring whether the *verification loop itself* helps, not whether a human is good at reviewing CNL.
- **In production mode** (future): A human (or upstream agent in a MAS) reads the CNL and approves or rejects. On rejection, LangGraph's conditional edge routes back to **Step 1** with feedback.

### Step 4: Code Security Scan
**Components touched:** `Python ast`

Before any execution, `security.py` runs `ast.parse()` on the Operator's generated code and walks the AST with a `NodeVisitor`. It checks for:
- Forbidden imports (`os`, `sys`, `subprocess`, `shutil`, `socket`, etc.)
- Forbidden function calls (`exec`, `eval`, `open`, `__import__`)
- Any attribute access on blocked modules

If the scan fails, the code is rejected and LangGraph routes back to **Step 1** with an error message. If it passes, we proceed to execution.

### Step 5: Doer Executes Code + Hypothesis PBTs
**Components touched:** `Hypothesis` + `hypothesis-jsonschema` → `subprocess` + `tempfile`

LangGraph enters the fourth node (`nodes/doer.py`). This is where the math meets the metal:

1. **Dynamic test compilation:** The Doer reads the PBT JSON and programmatically generates a Hypothesis test file. Each property becomes a `@given`-decorated test using strategies from `hypothesis` and `hypothesis-jsonschema`:

```python
from hypothesis import given, strategies as st

@given(input_list=st.lists(st.text(min_size=1), min_size=1))
def test_result_is_in_input(input_list):
    result = longest_string(input_list)
    assert result in input_list

@given(input_list=st.lists(st.text(min_size=1), min_size=1))
def test_no_string_is_longer(input_list):
    result = longest_string(input_list)
    assert all(len(result) >= len(s) for s in input_list)

def test_empty_list_raises():
    with pytest.raises(ValueError):
        longest_string([])
```

2. **Sandboxed execution:** The Operator's code and the generated test file are written to a `tempfile` directory. A `subprocess.run()` call executes them with a **30-second timeout**:

```python
result = subprocess.run(
    ["python", "-m", "pytest", test_file, "-x", "--tb=short"],
    capture_output=True, timeout=30, text=True
)
```

### Step 6: Pass or Retry?
**Components touched:** `LangGraph` (conditional edge) → optionally back to `LangChain`

LangGraph inspects the subprocess result:

- **All tests pass →** The solution is correct. LangGraph exits the loop and writes the result as ✅ **PASS** to the eval log.
- **Any test fails →** Hypothesis has already **shrunk** the failure to the minimal counterexample (e.g., `input_list=["a", "bb"]` — short, clear, token-cheap). LangGraph routes back to **Step 1** with the error trace and counterexample appended to the state. The Operator gets another shot. This loops up to a configurable max (e.g., 3 retries).
- **Max retries exceeded →** Written as ❌ **FAIL** to the eval log.

### Step 7: Metrics Logging
**Components touched:** `CSV/JSON logging`

Regardless of outcome, the eval harness logs:

| Field | Example |
|---|---|
| `problem_id` | `HumanEval/12` |
| `model` | `gpt-4o-mini` |
| `mode` | `dytm` (or `baseline`) |
| `passed` | `true` |
| `retries` | `1` |
| `total_tokens` | `2,847` |
| `wall_clock_sec` | `14.3` |
| `failure_reason` | `null` |

### Step 8: Observability
**Components touched:** `LangSmith`

Throughout all of the above, **LangSmith** has been silently tracing every LLM call, every node transition, every retry. After the eval run completes, you can open the LangSmith dashboard and:
- See exactly which problems needed retries and why
- Inspect the Operator's prompt/response at each attempt
- Identify patterns (e.g., "the model consistently fails edge cases on string problems")
- Export trace data for paper figures

### Step 9: Repeat for Baseline Comparison

The eval harness runs the same problems through a **baseline path** (`eval/run_baseline.py`): same model, same prompt, but zero-shot — no PBTs, no CNL, no verification loop. Just generate and test against the benchmark's canonical test suite. The Δ pass@1 between baseline and DYTM is the headline result.

---

## Free Mode — Interactive Demo

In addition to the automated eval harness, DYTM V1 includes a **Free Mode**: a terminal-based interactive session where a real user plays the Executive role.

### What It Is
A CLI REPL (read-eval-print loop) that runs the same LangGraph pipeline as the eval harness, but with **human-in-the-loop approval** instead of auto-approve. The user types a natural language coding request, watches the system generate PBTs and render the CNL contract, then approves or rejects it — experiencing the full DYTM verification loop firsthand.

### Why It Matters
- **Demos:** Show an advisor, reviewer, or conference audience exactly what DYTM feels like in 60 seconds.
- **Dogfooding:** Using it yourself surfaces UX issues with the CNL templates early (e.g., "this contract is unreadable" → fix the Jinja2 template).
- **Qualitative data:** If you run a small user study later, this is the interface participants interact with.

### How It Works

```
$ python -m dytm.free

╔══════════════════════════════════════════════╗
║           DYTM — Free Mode                   ║
║   Type a coding task. Type 'quit' to exit.   ║
╚══════════════════════════════════════════════╝

You: Write a function that checks if a number is prime.

🔧 Operator is generating code and verification properties...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERIFICATION CONTRACT — is_prime
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. The function MUST return a value of type: bool.
2. The function MUST return False for any input less than 2.
3. The function MUST return True for inputs 2, 3, 5, and 7.
4. The following invariant MUST hold: if the function returns
   True for n, then n has no divisors other than 1 and itself.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Do you approve this contract? [Y]es / [N]o / [Q]uit
You: y

✅ Contract approved. Running verification...

⏳ Attempt 1: Running Hypothesis PBTs...
   ❌ FAILED — test_no_divisors: Falsifying example: n=4
      is_prime(4) returned True, but 4 is divisible by 2.

🔄 Retrying with counterexample feedback...

⏳ Attempt 2: Running Hypothesis PBTs...
   ✅ ALL TESTS PASSED (4 properties verified)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESULT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Retries: 1 | Tokens: 1,923 | Time: 8.7s

You: _
```

### Rejection Flow

When the user rejects the CNL, they can provide feedback that gets routed back to the Operator:

```
Do you approve this contract? [Y]es / [N]o / [Q]uit
You: n

What's wrong with the contract? (optional, press Enter to skip)
You: It doesn't mention negative numbers.

🔄 Operator is regenerating with your feedback...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERIFICATION CONTRACT — is_prime (revised)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. The function MUST return a value of type: bool.
2. The function MUST return False for any input less than 2.
3. The function MUST return False for any negative input.
4. ...
```

### Implementation Details

- **Entry point:** `dytm/free.py` (runnable via `python -m dytm.free`)
- **Same graph, different approval node:** Free Mode uses the exact same `StateGraph` as the eval harness. The only difference is the approval node — it calls `input()` instead of auto-approving. This is a one-line config swap, not a separate codebase.
- **Rich terminal output:** Use the `rich` library for colored output, boxes, and spinners. One lightweight dependency that makes the demo dramatically more presentable.
- **No web server needed:** A terminal REPL is the simplest possible interface, zero infrastructure, runs anywhere.

---

## Stack Components

### 1. Orchestration — LangGraph

- **What:** Manages the multi-agent loop as a stateful, cyclical graph.
- **Why:** DYTM's verify-then-execute pipeline maps directly to a `StateGraph` with conditional edges. Built-in state persistence and checkpointing make debugging and replay trivial.
- **Key Detail:** A `TypedDict` holds the full state: prompt, generated code, PBT JSON payload, CNL string, execution logs, retry count.

### 2. LLM Interface — LangChain (Model-Agnostic)

- **What:** `langchain-openai` and `langchain-anthropic` provide a unified `BaseChatModel` interface.
- **Why:** Lets us swap models with one config change. Run the same eval across GPT-4o-mini, Claude 3.5 Sonnet, and a local model (Ollama) to show DYTM generalizes — a much stronger paper result.
- **Primary model:** GPT-4o-mini (cheap, fast, reliable structured output).
- **Secondary model(s):** Claude 3.5 Sonnet, and optionally a local model via Ollama (e.g., Qwen2.5-Coder) to demonstrate DYTM's impact on weaker models.

### 3. Schema Enforcement — Pydantic v2

- **What:** Defines strict schemas for the Operator's PBT JSON output.
- **Why:** Guarantees the LLM's structured output conforms to expected fields (`input_types`, `expected_invariants`, `edge_cases`, etc.). Works natively with `with_structured_output()` in LangChain.

### 4. PBT Engine — Hypothesis + hypothesis-jsonschema

- **What:** The Doer dynamically compiles PBT JSON specs into runnable `@given`-decorated Hypothesis tests.
- **Why:** Hypothesis is the Python gold standard for property-based testing. Its automatic **counterexample shrinking** finds the *minimal* failing input, which is extremely token-efficient when fed back to the Operator on retry — a core advantage of the DYTM loop.

### 5. CNL Translation — Jinja2 Templates

- **What:** A deterministic template engine that maps structured PBT JSON → human-readable Controlled Natural Language.
- **Why:** This is the "firewall" — no LLM touches this step. A set of pre-defined Jinja2 templates renders each PBT property into plain English (e.g., `{"property": "output_type", "expected": "list[int]"}` → *"The function MUST return a list of integers."*). Deterministic, zero-infrastructure, easy to debug.
- **Future work:** Could be replaced with a formal CNL system (e.g., APE / Attempto) for DRS-backed semantic verification.

### 6. Code Security — Python `ast`

- **What:** `ast.parse()` + `ast.NodeVisitor` scans generated code before execution.
- **Why:** Blocks dangerous imports (`os`, `subprocess`, `shutil`, etc.) and unauthorized function calls. Built-in to Python, zero dependencies.

### 7. Sandbox — `subprocess` + `tempfile`

- **What:** Executes generated code + Hypothesis tests in a subprocess with timeout and resource limits.
- **Why:** Fast, free, no external dependencies. Standard practice for benchmark eval papers. Sufficient isolation for controlled evaluation where inputs are known benchmark problems.
- **Future work:** Upgrade to E2B or Docker for production use with untrusted user code.

### 8. Observability — LangSmith

- **What:** Traces every LLM call, node transition, and retry in the LangGraph loop.
- **Why:** When problem #87 fails, you need to know if it was a bad PBT, a flawed generation, or a sandbox error. LangSmith integrates with LangGraph in one line of config. Free tier is sufficient.

### 9. Eval Benchmarks — HumanEval + MBPP

- **What:**
  - **HumanEval** — 164 hand-written Python problems (OpenAI standard).
  - **MBPP** — 974 crowd-sourced Python problems (Google standard).
- **Why:** HumanEval is the baseline every reviewer expects. MBPP adds breadth and strengthens generalizability claims. Both use the same format (function signature + docstring → implementation), so one eval harness covers both.

### 10. Metrics Logging — CSV / JSON

- **What:** Log per-problem results: pass/fail, retry count, tokens consumed, wall-clock time.
- **Why:** Feeds directly into paper tables and figures. Simple, portable, no extra dependencies.

---

## Dependencies

```
# requirements.txt
langgraph
langchain-openai
langchain-anthropic
langsmith
pydantic>=2.0
hypothesis
hypothesis-jsonschema
jinja2
rich
```

---

## Eval Protocol

1. **Baseline run:** Model solves each benchmark problem with zero-shot prompting (no DYTM). Record pass@1.
2. **DYTM run:** Same model, same problems, but routed through the full LangGraph verification loop. Executive approval is auto-accepted (simulated). Record pass@1, retry count, and token overhead.
3. **Compare:** Δ pass@1, cost of retries, and whether DYTM disproportionately helps weaker models.
4. **Repeat** across 2–3 models to show generalizability.

---

## What's Explicitly NOT in V1

| Deferred | Reason |
|---|---|
| APE / Attempto (Prolog CNL) | Heavy infrastructure; templates are equally deterministic for eval |
| E2B (cloud sandbox) | Cost & latency overhead for batch runs of 1000+ executions |
| GAIA benchmark | Different eval paradigm (multi-step reasoning); save for follow-up |
| Web UI / FastAPI | No user-facing product in V1 — this is a pure eval harness |
| iOS / Swift / Phase 2 | Separate product phase after academic validation |

---

## Project Structure (Planned)

```
DYTM/
├── dytm/
│   ├── __init__.py
│   ├── graph.py              # LangGraph StateGraph definition
│   ├── state.py              # TypedDict state schema
│   ├── free.py               # Free Mode CLI REPL (python -m dytm.free)
│   ├── nodes/
│   │   ├── operator.py       # Node 1: LLM generates code + PBT JSON
│   │   ├── translator.py     # Node 2: Jinja2 JSON → CNL
│   │   ├── approval.py       # Node 3: Auto-approve (eval) or human-in-the-loop (free)
│   │   └── doer.py           # Node 4: subprocess execution + Hypothesis
│   ├── schemas/
│   │   └── pbt.py            # Pydantic models for PBT payloads
│   ├── templates/
│   │   └── cnl.jinja2        # CNL rendering templates
│   └── security.py           # ast-based code scanner
├── eval/
│   ├── run_baseline.py       # Zero-shot baseline eval
│   ├── run_dytm.py           # DYTM loop eval
│   ├── benchmarks/           # HumanEval + MBPP problem sets
│   └── results/              # CSV/JSON output
├── tests/                    # Unit tests for the DYTM system itself
├── requirements.txt
├── README.md
└── Stack Plan.md             # ← this file
```

