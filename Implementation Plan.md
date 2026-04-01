# DYTM V1 — Implementation Plan

> **Strategy:** Build bottom-up — lay the data foundation (schemas, state), implement the four nodes, wire the LangGraph loop for a first end-to-end run, then layer on eval harness, Free Mode, and observability. Each phase produces a runnable, testable artifact. **First end-to-end loop arrives at Phase 3 (of 6).**

---

## Dependency Graph

```
Phase 1 (Schemas & State)
   └──▶ Phase 2 (Deterministic Nodes)
           └──▶ Phase 3 (Operator + Graph Wiring) ← first end-to-end loop 🎉
                   ├──▶ Phase 4 (Eval Harness + Benchmarks)
                   │       └──▶ Phase 6 (LangSmith + Multi-Model + Polish)
                   └──▶ Phase 5 (Free Mode CLI)
```

---

## Phase 1: Scaffolding, Schemas & State

| | |
|---|---|
| **Complexity** | 🟢 Low |
| **Depends on** | Nothing (first phase) |
| **Agent-delegatable?** | ✅ Fully — all files have clear specs, zero ambiguity |

### Work

1. **Create the directory tree** — `dytm/`, `dytm/nodes/`, `dytm/schemas/`, `dytm/templates/`, `eval/`, `eval/benchmarks/`, `eval/results/`, `tests/` — with `__init__.py` files where needed.

2. **Create `requirements.txt`** with all dependencies:
   ```
   langgraph
   langchain-openai
   langchain-anthropic
   langsmith
   pydantic>=2.0
   hypothesis
   hypothesis-jsonschema
   jinja2
   rich
   pytest
   ```

3. **Define Pydantic models** in `dytm/schemas/pbt.py`:
   - `PBTProperty` — fields: `name` (str), `type` (Literal: `"output_type"`, `"membership"`, `"invariant"`, `"edge_case"`), `expected` (str), optional `input`, optional `expected_behavior`
   - `PBTPayload` — fields: `function_name` (str), `properties` (list[PBTProperty])
   - `OperatorOutput` — wraps `code: str` + `pbt: PBTPayload` (this is the `with_structured_output()` target)

4. **Define LangGraph state** in `dytm/state.py` as a `TypedDict`:
   - `prompt`, `code`, `pbt_payload`, `cnl_text`, `approved` (bool), `test_result` (str), `retry_count` (int), `max_retries` (int, default 3), `counterexample` (optional str), `error_trace` (optional str), `mode` (Literal `"eval"` | `"free"`), `security_passed` (bool)

### Smoke Test
```bash
python -c "from dytm.schemas.pbt import PBTPayload; from dytm.state import DYTMState; print('Phase 1 OK')"
```
Imports succeed, and a sample `PBTPayload` can be instantiated with the example JSON from the Stack Plan.

---

## Phase 2: Deterministic Nodes (Translator, Approval, Doer, Security)

| | |
|---|---|
| **Complexity** | 🟡 Medium-High |
| **Depends on** | Phase 1 |
| **Agent-delegatable?** | ✅ Security scanner, translator, approval — fully delegatable. ⚠️ Doer node — partially delegatable; the Hypothesis strategy mapping requires careful design. Draft with agent, review/iterate manually. |

### Work

1. **`dytm/security.py`** — `ast.NodeVisitor` subclass that scans code for:
   - Forbidden imports: `os`, `sys`, `subprocess`, `shutil`, `socket`, `importlib`
   - Forbidden calls: `exec`, `eval`, `open`, `__import__`, `compile`
   - Blocked attribute access
   - Expose: `scan_code(code: str) -> tuple[bool, str | None]` returning `(safe, reason)`

2. **`dytm/templates/cnl.jinja2`** — Jinja2 template that iterates `PBTPayload.properties` and renders by type:
   - `output_type` → *"The function MUST return a value of type: {{expected}}."*
   - `membership` → *"The returned value MUST satisfy: {{expected}}."*
   - `invariant` → *"The following invariant MUST hold: {{expected}}."*
   - `edge_case` → *"EDGE CASE: When given {{input}}, the function MUST {{expected_behavior}}."*
   - Header block with function name

3. **`dytm/nodes/translator.py`** — Load Jinja2 template, render `state["pbt_payload"]` → CNL string, write to `state["cnl_text"]`. Zero LLM calls.

4. **`dytm/nodes/approval.py`** — If `state["mode"] == "eval"`: auto-approve. If `"free"`: call `input()` for Y/N. (Full Rich UI comes in Phase 5.)

5. **`dytm/nodes/doer.py`** — The most complex file in the project:
   - Call `security.scan_code()` on `state["code"]`. Fail fast if unsafe.
   - **Strategy mapping:** Map PBT property `type` → Hypothesis `@given` + assertion:
     - `output_type` → type-check assertion
     - `membership` → containment assertion
     - `invariant` → direct `assert` on the expression
     - `edge_case` → `pytest.raises` or specific behavior check
   - Write generated code + compiled test file to `tempfile.TemporaryDirectory`
   - Execute: `subprocess.run(["python", "-m", "pytest", ...], timeout=30)`
   - Parse result: returncode 0 → `state["test_result"] = "pass"`. Otherwise extract failure output → `state["error_trace"]`, `state["counterexample"]`, increment `state["retry_count"]`

### Smoke Test (`tests/test_phase2.py`)
- Pass known-safe and known-dangerous code through `scan_code` — verify correct accept/reject
- Render the `longest_string` PBT example through the translator — assert CNL contains "MUST return"
- Feed a known-good function + PBT through the Doer — assert pass
- Mutate the function to be wrong — assert the Doer catches it with a counterexample

---

## Phase 3: Operator Node + Graph Assembly → First End-to-End Loop 🎉

| | |
|---|---|
| **Complexity** | 🟡 Medium |
| **Depends on** | Phase 2 |
| **Agent-delegatable?** | ⚠️ Graph wiring is mechanical and delegatable. The Operator's system prompt requires human judgment and iterative tuning — this is where PBT quality is determined. Draft with agent, expect to iterate manually. |

### Work

1. **`dytm/nodes/operator.py`**:
   - Accept a `BaseChatModel` instance via dependency injection
   - System prompt instructs the LLM to produce a Python function AND a structured PBT payload. Include 1–2 few-shot examples of good PBT JSON (reuse `longest_string` example)
   - Call `model.with_structured_output(OperatorOutput)`
   - On retries: append `state["error_trace"]` and `state["counterexample"]` to messages
   - Write `code` and `pbt_payload` into state

2. **`dytm/graph.py`** — Wire the full `StateGraph(DYTMState)`:
   - Nodes: `operator` → `translator` → `approval` → `doer`
   - Conditional edge after `doer`: `pass` → END, `retry_count >= max_retries` → END, else → `operator`
   - Conditional edge after `approval`: `approved` → `doer`, not approved → `operator`
   - Compile and export as `app`

3. **Minimal entry point** (`dytm/__main__.py`):
   - Instantiate `ChatOpenAI(model="gpt-4o-mini")`
   - Build graph, invoke with a single hardcoded prompt

### Smoke Test
```bash
python -m dytm
```
Run against one problem. Observe: Operator produces code + PBT JSON → Translator renders CNL → Doer runs Hypothesis tests → either first-attempt pass or successful retry. Verify `state["test_result"] == "pass"` at termination.

**🎉 This is the milestone — the core DYTM loop is alive.**

---

## Phase 4: Eval Harness + Benchmarks

| | |
|---|---|
| **Complexity** | 🟡 Medium |
| **Depends on** | Phase 3 |
| **Agent-delegatable?** | ✅ Benchmark loading and CSV logging are mechanical. ⚠️ Baseline eval design (running canonical test suites) is partially delegatable with clear instructions. |

### Work

1. **`eval/benchmarks/`** — Download/vendor HumanEval (OpenAI JSONL) and MBPP (HuggingFace `datasets`). Write a loader module (`eval/loader.py`) that yields `(problem_id, prompt, canonical_tests)` tuples.

2. **`eval/run_baseline.py`** — For each benchmark problem:
   - Call LLM with zero-shot prompting (no PBTs, no loop)
   - Execute generated code against benchmark's canonical test suite via subprocess
   - Log pass/fail to CSV in `eval/results/`

3. **`eval/run_dytm.py`** — For each benchmark problem:
   - Invoke compiled LangGraph with `mode="eval"`
   - After DYTM loop terminates, run final code against **canonical tests** (ground-truth, separate from PBT verification)
   - Log: `problem_id`, `model`, `mode`, `passed`, `retries`, `total_tokens`, `wall_clock_sec`, `failure_reason`

4. **Comparison script** — Load both CSVs, print Δ pass@1, average retries, token overhead.

> ⚠️ **Key distinction:** The DYTM loop validates code against *LLM-generated* PBTs. The eval harness must *also* run the benchmark's ground-truth tests to measure true pass@1. These are two different checks — both must be implemented.

### Smoke Test
Run both `run_baseline.py` and `run_dytm.py` on the **first 10 HumanEval problems**. Verify CSVs appear in `eval/results/` with correct columns. Confirm DYTM pass rate ≥ baseline on this subset.

---

## Phase 5: Free Mode CLI

| | |
|---|---|
| **Complexity** | 🟢 Low-Medium |
| **Depends on** | Phase 3 (independent of Phase 4) |
| **Agent-delegatable?** | ✅ Fully — the spec in the Stack Plan is extremely detailed and prescriptive |

### Work

1. **`dytm/free.py`** — Rich-powered REPL:
   - Startup banner via `rich.panel.Panel`
   - Loop: read input → invoke graph with `mode="free"` → display CNL in styled panel → prompt Y/N/Q → show Hypothesis progress with `rich.status` spinners → display final code via `rich.syntax.Syntax` with retry/token/time stats
   - On rejection: collect optional feedback text, pass back to Operator via state
   - Handle `quit` / `Ctrl+C` gracefully

2. **Wire entry point** — `python -m dytm.free` via `__main__.py` or CLI argument

### Smoke Test
```bash
python -m dytm.free
```
Type *"Write a function that checks if a number is prime"*, approve the contract, watch the loop complete — matching the UX flow in the Stack Plan.

---

## Phase 6: Observability, Multi-Model & Polish

| | |
|---|---|
| **Complexity** | 🟢 Low |
| **Depends on** | Phase 4 |
| **Agent-delegatable?** | ✅ Fully — LangSmith is config, multi-model is a factory function, README is delegatable with review |

### Work

1. **LangSmith tracing** — Create `.env` with `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT=dytm-v1`. Confirm traces appear in dashboard after one run.

2. **Multi-model support** — Parameterize `run_baseline.py` and `run_dytm.py` with `--model` flag (e.g., `gpt-4o-mini`, `claude-3-5-sonnet`, `ollama/qwen2.5-coder`). Factory function returns the appropriate `ChatOpenAI` / `ChatAnthropic` / `ChatOllama`.

3. **README.md** — Setup instructions (API keys, `pip install`, how to run eval, how to run Free Mode), results format documentation.

### Smoke Test
Run `run_dytm.py --model gpt-4o-mini` on 5 problems, switch to `--model claude-3-5-sonnet` on the same 5. Verify both produce valid CSVs and traces appear in LangSmith.

---

## Agent Delegation Summary

| Phase | Can an agent build it? | Notes |
|---|---|---|
| **1 — Schemas & State** | ✅ Yes, fully | Pure data definitions, zero ambiguity |
| **2 — Deterministic Nodes** | ✅ Mostly | Security, translator, approval are straightforward. Doer's strategy mapping needs human review |
| **3 — Operator + Graph** | ⚠️ Partially | Graph wiring is mechanical. **Operator system prompt needs human iteration** — this is the creative core |
| **4 — Eval Harness** | ✅ Mostly | Loader + CSV logging are mechanical. Canonical test execution needs clear spec |
| **5 — Free Mode CLI** | ✅ Yes, fully | Stack Plan spec is prescriptive enough to build directly from |
| **6 — Observability** | ✅ Yes, fully | Config + factory function + README |

### Recommended Workflow
1. **Delegate Phase 1** entirely to an agent — it's just scaffolding.
2. **Delegate Phase 2** file-by-file (security → translator → approval → doer), reviewing the Doer carefully.
3. **Phase 3 is collaborative** — have the agent draft `operator.py` and `graph.py`, then manually tune the Operator's system prompt. This is where DYTM's effectiveness is won or lost.
4. **Delegate Phases 4–6** with clear task descriptions referencing the Stack Plan.

---

## Related Work & Positioning

Understanding the landscape is critical for the paper. Here's what to cite, what to borrow, and how DYTM differentiates.

### Must-Cite Papers

#### Iterative Refinement Systems (closest competitors)

| Paper | Core Idea | DYTM's Differentiator |
|---|---|---|
| **TiCoder** — *"User-Intent Formalization with Test-Driven Development"* (Microsoft Research) | Uses unit tests to clarify user intent before code generation | TiCoder uses example-based unit tests. DYTM uses *property-based* tests (strictly more expressive) and adds the CNL approval contract — a formal verification gate TiCoder lacks. |
| **Reflexion** — *"Language Agents with Verbal Reinforcement Learning"* (Shinn et al.) | LLM reflects on failures via verbal self-feedback, iterates | Reflexion's feedback signal is LLM-generated text — itself prone to hallucination. DYTM's signal is Hypothesis counterexamples: deterministic, minimal, mathematically grounded. |
| **Self-Refine** — *"Iterative Refinement with Self-Feedback"* (Madaan et al.) | LLM critiques and refines its own output in a loop | Same weakness as Reflexion: the refinement signal is probabilistic. DYTM's refinement is driven by a deterministic test engine. |
| **LEVER** — *"Learning to Verify LLM-Generated Code"* (Ni et al.) | Trains a separate neural verifier model to rank code candidates | LEVER's verifier is a trained (probabilistic) model — it can be wrong. DYTM's verifier is structural and deterministic. No additional training required. |
| **Clover** — *"Closed-Loop Verifiable Code Generation"* | Uses formal pre/post-conditions (Dafny-style proofs) for verification | Heavier formal methods approach. DYTM is lighter-weight (PBTs, not full proofs) and more practical for general code generation benchmarks. |

#### PBT + LLM (emerging area)

| Paper | Relevance |
|---|---|
| Recent workshop papers on *"Property-Based Testing for LLM-Generated Code"* | These treat PBT generation as the end goal (generate better tests). DYTM is different: PBT generation is a *means* — the verification layer in a larger agent loop, not the product itself. |
| **QuickCheck** lineage (Claessen & Hughes) and **Hypothesis** (MacIver) | Foundational PBT literature. Cite for the theoretical grounding of property-based testing and counterexample shrinking. |

#### CNL / Formal Methods

| Paper | Relevance |
|---|---|
| **Attempto Controlled English (ACE)** (Fuchs et al.) | Cite for the CNL concept and formal semantics. DYTM V1 uses Jinja2 templates as a lightweight CNL; ACE/APE is the principled upgrade path for future work. |
| **Controlled Natural Language** survey papers | Grounds the CNL design choice academically. |

### What to Borrow (Not Code — Methodology)

| Borrow | From | Why |
|---|---|---|
| **Eval methodology** | TiCoder, Reflexion, LEVER | How they measured pass@1, what baselines they compared against, how they reported token costs — follow the same structure so results are directly comparable |
| **Benchmark setup** | HumanEval papers, MBPP papers | Exact data loading, canonical test execution, pass@k calculation |
| **Retry prompt patterns** | Reflexion, Self-Refine | How they structured retry prompts with error feedback — proven templates to adapt for the Operator's retry messages |

### Why Build Fresh (Not Import Code)

1. **Architecture mismatch.** No existing paper has DYTM's exact pipeline (structured PBT generation + deterministic CNL + approval gate + sandboxed PBT execution). Adapting someone else's codebase would be more work than building clean on LangGraph.
2. **Better tooling.** TiCoder, Reflexion, and Self-Refine were built before LangGraph matured. Their codebases use ad-hoc loop management that you'd rip out anyway.
3. **Academic clarity.** Reviewers will examine your architecture. A clean, purpose-built implementation makes the contribution legible. A frankensteined codebase with imported modules obscures it.

### The Positioning Paragraph (for the paper)

> *"Unlike Reflexion (Shinn et al.) and Self-Refine (Madaan et al.), which rely on LLM-generated verbal feedback — itself subject to hallucination — DYTM's refinement signal is produced by a deterministic property-based testing engine (Hypothesis), ensuring that counterexamples are mathematically valid. Unlike LEVER (Ni et al.), which trains a probabilistic verifier, DYTM requires no additional training — verification is structural. And unlike TiCoder, which uses example-based unit tests to clarify intent, DYTM introduces property-based tests (which generalize over infinite inputs) combined with a Controlled Natural Language contract that enables the Executive to formally approve the verification criteria before execution begins."*

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Operator produces low-quality PBTs | High — weak tests = no improvement over baseline | Few-shot examples in system prompt; iterate on prompt in Phase 3 |
| Hypothesis strategy mapping too narrow | Medium — some benchmark problems can't be tested | Start with 4 property types; expand if benchmark coverage < 80% |
| PBTs pass but canonical tests fail | Medium — inflated DYTM results | Phase 4 always runs canonical tests as ground truth |
| Structured output failures on non-OpenAI models | Medium — blocks multi-model eval | Fallback to tool-calling or JSON-mode with manual parsing |
| Subprocess timeout on complex Hypothesis tests | Low — some problems hang | 30s timeout + max retry cap prevents infinite loops |

---

## V1 Findings & Post-Mortem

> Added after completing the full 164-problem HumanEval evaluation.

### Pipeline Evolution

The V1 pipeline evolved significantly from the original plan:

1. **V1 (Phases 1-3)**: Single Operator node generates code + PBTs together.
2. **V2 (current)**: Separated pipeline — **Spec** (PBTs only, no code) → **Translator** → **Executive Approval** (LLM gate) → **Coder** (code on fresh context) → **Doer**. Key insight: the verification contract must be approved *before any code exists*, and the Coder must have *no memory* of the Spec conversation.

### The Evaluation Gap — The Key Finding

The most important finding is not the benchmark numbers — it's that **HumanEval is the wrong evaluation for DYTM's core thesis**.

DYTM is a protocol for verifying *intent*. Its value comes from step 4 of the original description: "The executive approves or returns the CNL." This requires an independent intent-holder — a human who knows what they want and can confirm whether the contract captures it.

On HumanEval, the "intent" is a docstring. There is no independent intent-holder. The LLM reads the docstring to write the Spec, reads it again to approve the Spec, and reads it again to write the code. All three interpretations are correlated — when the model misunderstands the task, it misunderstands it at every stage, and verification provides zero signal.

**The regressions (9 problems)** are the evidence: DYTM adds an extra interpretation step (docstring → contract → code instead of docstring → code). Without a human to validate the contract, this extra step is pure risk — an additional opportunity for correlated misinterpretation.

**The fixes (12 problems)** come from DYTM's secondary mechanism — the PBT retry loop — not from intent verification. When the contract happens to be correct, Hypothesis counterexamples provide a powerful deterministic error-correction signal that verbal self-reflection (Reflexion, Self-Refine) cannot match.

### Risk Register Post-Mortem

| Original Risk | Occurred? | Notes |
|---|---|---|
| Operator produces low-quality PBTs | **Yes — this is the #1 failure mode** | 12/21 DYTM failures are "weak contracts" where PBTs pass but canonical tests fail. Extensive prompt tuning helped (from ~84% to ~87%) but hit a ceiling. The root cause is correlated errors: the same model misunderstands the task in both the Spec and the Code. |
| Hypothesis strategy mapping too narrow | Partially | Early issues with `st.text()` for parentheses parsers, broad float ranges for root-finding. Fixed with prompt guidance for strategy domains. |
| PBTs pass but canonical tests fail | **Yes — 57% of all failures** | This was correctly identified as a risk. The "edge-only-fail soft pass" heuristic helps but doesn't solve the core problem of weak invariants. |
| Structured output failures | Once | HumanEval/22 hit a JSON parse error (API 400). Rare but cost 1 problem. |
| Subprocess timeout | Rare | 180s timeout was sufficient. Some problems needed 3-4 retries but didn't timeout. |

### What Would a Correct Evaluation Look Like?

1. **User study**: N participants use Free Mode to solve coding tasks. Measure whether reviewing the CNL contract before code generation reduces intent mismatches compared to direct code generation (baseline). This tests the *primary* mechanism.

2. **Adversarial intent benchmark**: Create tasks where the spec is deliberately ambiguous (e.g., "sort the list" — ascending or descending?). Measure whether DYTM + human-in-the-loop catches the ambiguity and produces the user's intended behavior, while the baseline guesses wrong.

3. **MAS communication study**: In a multi-agent system, measure whether requiring DYTM CNL verification between agent pairs reduces cascading errors compared to free-form natural language delegation.

The HumanEval result (+1.8%) is a valid supporting data point showing the PBT retry loop provides secondary value. But it is not — and cannot be — evidence for or against DYTM's core thesis.

