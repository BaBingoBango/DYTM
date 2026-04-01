# DYTM

DYTM ("Do You Trust Me") is a verification layer for large language models using formal methods. Instead of relying on direct English to logic transformation, the following is assumed:

1. LLMs are untrustworthy by definition due to their probabilistic and largely unexplainable generations.
2. As a result of (1), LLMs cannot be relied upon to convert English to logic.

The situation could be modeled as:

- The **executive**, either a human being or an LLM agent. Either has an _intent_ (dubious in the case of the LLM but the analogy holds), which actually may or may not be a concrete logical thought (i.e. someone not even knowing what exactly they want but prompting anyway). We accept that this intent cannot be translated into logic due to natural language vagueness and unsure operators.
- The **operator**, an LLM agent. Technically, could also be thought of as a human being in a theoretical sense. This is the agent responsible for completing the executive's task.

We can recognize that the executive cannot transfer intent directly due to language constraints. To combat this, the DYTM requires a verification before task work begins:

1. The executive converts their intent into a prompt, which is sent to the operator.
2. The operator proposes a suite of property-based tests (PBTs) which describe what the operator thinks the executive meant. For instance, these could be code tests for a programming task, Excel functions for a spreadsheet task, or whatever verification layer is available.
3. The operator's PBTs map 1:1 to logic, and are therefore converted to an implementation of constrained natural logic (CNL).
4. The executive approves or returns the CNL, which is now a mathematically sound description of a verifier suite. If returned, the operator returns to step (2). If approved, the operator moves to step (5).
5. The operator completes the task, rather, it iterates on the task until its artifacts pass all of the PBTs. If the task fails over and over, this result can be returned (an LLM effort failure).
6. The executive gets the results, which mathematically match their original intent.

---

## V1 Implementation

The V1 implementation is a Python evaluation harness using LangGraph, Hypothesis PBTs, and Jinja2 CNL templates. The pipeline is:

```
Spec (LLM: generate PBT properties only)
  → Translator (Jinja2: deterministic JSON → CNL)
  → Approval (Executive LLM or human-in-the-loop)
  → Coder (LLM: generate code on fresh context)
  → Doer (subprocess: run Hypothesis PBTs in sandbox)
  → Pass? → Done. Fail? → Retry from Coder (contract is locked).
```

Key design principle: **separation of specification and implementation**. The Spec generates verification properties *before* any code exists. The Coder operates on a fresh context with only the approved contract — no memory of the Spec conversation. This enforces independence between the contract and the code.

---

## V1 Results — HumanEval (164 problems)

| Metric | DYTM | Baseline |
|--------|------|----------|
| **pass@1** | **143/164 = 87.2%** | 140/164 = 85.4% |
| Delta | **+1.8%** (net +3 problems) | — |
| Problems fixed by DYTM | 12 | — |
| Problems regressed by DYTM | 9 | — |
| Avg retries | 1.55 | 0 |
| Avg time/problem | 46.6s | ~3s |

Model: `gpt-4o-mini` for all nodes (Spec, Executive, Coder).

### What Worked: The PBT Retry Loop

DYTM **fixed 12 problems** the baseline got wrong — a **50% error recovery rate** on baseline failures. When the PBT contract correctly captures the task's invariants, the retry loop with Hypothesis counterexamples provides a powerful error-correction signal. This is the secondary mechanism of DYTM: deterministic feedback beats verbal self-reflection.

### What Didn't Work: Correlated Specification Errors

DYTM **regressed on 9 problems** the baseline got right. Root cause analysis of these regressions revealed three patterns:

1. **Wrong invariant** (necessary but not sufficient) — e.g., HumanEval/56 (`correct_bracketing`): the Spec generated `count('<') == count('>')` as the core invariant, which is necessary but not sufficient. Correct bracketing also requires prefix-validity. The count-based code passed the weak PBT but failed the canonical tests.

2. **Buggy oracle** (same misunderstanding in Spec and Code) — e.g., HumanEval/108 (`count_nums`): the task says "for negative numbers, the first signed digit is negative." The Spec's test implementation made *all* digits negative — the same mistake the Coder made. PBT and code agreed on the wrong answer.

3. **Buggy oracle** (missed operator precedence) — e.g., HumanEval/160 (`do_algebra`): the Spec's oracle evaluated left-to-right instead of respecting standard operator precedence. The Coder did the same. Both got `2 + 3 * 4 - 5 = 15` instead of `9`.

All three patterns share the same root cause: **the Spec model (gpt-4o-mini) makes the same misunderstandings as the Coder model (gpt-4o-mini)**. When both agree on the wrong interpretation, verification provides zero signal.

---

## The Core Insight: The Evaluation Gap

**DYTM's value proposition is bridging human intent → verified contract → code.** The critical step is a human (or independent agent) reviewing the CNL contract and confirming: "yes, this captures what I want."

**HumanEval cannot test this.** In a benchmark, the "intent" is encoded in a docstring. There is no independent intent-holder. The evaluation becomes:

```
Docstring (static text)
  → LLM interprets spec → generates contract
  → LLM interprets spec → "verifies" contract   ← CIRCULAR
  → LLM interprets spec → generates code
```

The contract and the code are both downstream interpretations of the same text by the same model. When the model misunderstands the docstring, it misunderstands it everywhere — in the Spec, in the Executive approval, and in the Coder. There is no human to say "wait, that's not what I meant."

**The regressions are the smoking gun for this.** The baseline goes docstring → code (one interpretation step). DYTM goes docstring → contract → code (two interpretation steps). Without a human to validate the contract, the extra step is pure risk — an additional opportunity to misinterpret.

### What HumanEval Actually Measures

HumanEval tests DYTM's **secondary** mechanism — the PBT retry loop as an error-correction tool — not its **primary** mechanism — intent verification via CNL contracts.

The +1.8% improvement is from the secondary mechanism alone, achieved *despite* the added risk of correlated misinterpretation from the extra specification step. The primary value proposition (a human reading a CNL contract and catching misunderstandings before any code is written) is fundamentally untestable on a static benchmark.

### Implications

1. **For evaluation**: DYTM's core thesis requires a **user study** or **interactive evaluation** — not a benchmark. The Free Mode demo (where a human reads the CNL and approves/rejects) is the correct demonstration of the primary mechanism.

2. **For heterogeneous models**: The correlated-error problem suggests DYTM's value scales with the *gap* between Spec quality and Coder quality. A stronger Spec model breaks the correlation — but then the question becomes whether the stronger model's improvement comes from DYTM or just from being a better model.

3. **For the DYTM thesis**: The system works as designed. The problem isn't DYTM — it's that HumanEval replaces the human with a docstring, voiding the entire approval gate. DYTM is a verification protocol for intent, and intent lives in a person's head, not in a spec.

---

## Heuristics

- A "smoother" LLM agent can be employed to improve readability of the CNL, as CNL is often hard to read and dense. This would be advisory/optional only, as the LLM could easily get this wrong/over-describe the CNL.

## Creating Tests

Another key element is the creation of tests for the environment at hand. For instance, code tests might be easier, while Excel tests or UI-based development tests might be harder.

It would also be interesting if "informational" tests could be set up without employing another LLM as judge, i.e. was X advice derived from Y pool of approved facts? Maybe this would require some sort of knowledge graph/data structure or database.

## Key Uses

- This could hopefully improve user trust in regular LLM "chatbots" by requiring users to verify intent in a way that will certainly come to pass by way of the tests.
- This could also possibly increase the power of MAS systems. Whereas traditional MASs converse in natural language with one another, requiring the DYTM CNL verification could reduce intent errors and smooth the flow of ideas between many agents. Such a MAS would probably be based on "duos" of executive & operator agents instead of singleton agents.

## Implementation

- An on-device implementation (e.g. Apple's Foundation Models framework) would be a robust way to demonstrate if the system is effective.
- A simpler code-only Python/web based implementation might serve as a stronger V1 for evaluation (i.e. does DYTM even help?).

---

## Running

### Setup
```bash
pip install -e .
# Set OPENAI_API_KEY in .env.local
```

### Free Mode (Human-in-the-Loop Demo)
```bash
python -m dytm.free
```

### Eval
```bash
# Baseline (zero-shot, no verification)
python -m eval.run_baseline --model gpt-4o-mini

# DYTM (full verification loop with Executive LLM)
python -m eval.run_dytm --model gpt-4o-mini --executive-model gpt-4o-mini -v

# Compare results
python -m eval.compare <baseline.csv> <dytm.csv>
```
