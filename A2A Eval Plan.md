# DYTM A2A Eval — Agent-to-Agent Intent Drift Experiment

> **Thesis**: DYTM reduces wasted computation in multi-agent systems by catching intent drift *before* execution begins. It is a communication protocol, not a correctness oracle.

> Last updated: April 7, 2026.

---

## 1. Motivation

### The HumanEval Lesson

Five experiments on HumanEval (see README.md §3–5) established that:

- DYTM's **secondary mechanism** (PBT retry loop) provides modest gains (+1.8% pass@1).
- DYTM's **primary mechanism** (intent verification via CNL contracts) is **untestable on a static benchmark** — there is no independent intent-holder to review the contract.
- Correlated errors (same model misunderstands at every stage) are the dominant failure mode.

### The Reframing

DYTM's value isn't "better code." It's **"the agents caught the misunderstanding before any code was written."**

In a multi-agent system, the catastrophic failure mode isn't a buggy function — it's Agent B confidently building the *wrong thing* because it misunderstood Agent A. The code can be beautifully written and still completely wrong.

DYTM interposes a verifiable checkpoint: "Here's what I think you want — is this right?" This is measurable without external ground truth, because we're measuring **internal consistency**, not output correctness.

---

## 2. Experiment Design

### 2.1 Actors

| Role | Description | Model |
|---|---|---|
| **Planner** (Agent A) | Generates complex coding tasks. Holds the intent. Acts as Executive in the DYTM condition. | `claude-3.5-sonnet` **(Anthropic — REQUIRED)** |
| **Spec** | (DYTM only) Reads the Planner's task description and generates PBT contract. | `gpt-4o-mini` **(OpenAI)** |
| **Coder** (Agent B) | Receives instructions and writes code. Never sees the Planner's internal intent — only the communicated description (baseline) or approved contract (DYTM). | `gpt-4o-mini` **(OpenAI)** |

> ⚠️ **HARD REQUIREMENT — Cross-Family Model Decorrelation**
>
> The Planner **MUST** be a different model family (different provider, different training pipeline) from the Spec/Coder. Using `gpt-4o` as Planner and `gpt-4o-mini` as Coder is **insufficient** — they share the same RLHF training, tokenization, and implicit biases. They will make correlated assumptions about ambiguous terms, default edge-case handling, and domain conventions, which is the exact confound this experiment must avoid.
>
> **Primary configuration**: Planner = Anthropic `claude-3.5-sonnet`, Spec/Coder = OpenAI `gpt-4o-mini`.
>
> **Ablation** (optional): Swap families (Planner = OpenAI `gpt-4o`, Spec/Coder = Anthropic `claude-3-haiku`) to check for asymmetric decorrelation effects. Report both.

### 2.2 Task Generation

The Planner generates **50 complex coding tasks**. Each task must include:

```json
{
  "task_id": "A2A/001",
  "description": "Natural language task description (what the Coder sees)",
  "intent_summary": "Planner's private summary of what it actually wants",
  "complexity_axes": ["edge_case_heavy", "implicit_constraints", "multi_step"],
  "ambiguity_level": "high"
}
```

**Task complexity requirements** (to ensure intent drift is possible):

- Tasks should involve **implicit constraints** (e.g., "process user data" — but what counts as invalid data?)
- Tasks should have **ambiguous edge cases** (e.g., "sort the results" — ascending? descending? stable?)
- Tasks should require **multi-step logic** (not single-function HumanEval-style problems)
- Tasks should include **domain-specific terms** that could be interpreted differently

**Task categories** (10 tasks each):

1. **Data transformation** — ETL-style operations with implicit schemas
2. **Business logic** — Rules with ambiguous boundary conditions
3. **String/text processing** — Parsing with unstated format assumptions
4. **Algorithmic** — Problems where the "obvious" approach has subtle incorrectness
5. **API/interface design** — Functions where the contract (params, returns, errors) is underconstrained

### 2.3 Conditions

#### Condition A: Baseline (Natural Language Only)

```
Planner writes NL description
        │
        ▼
Coder reads NL description
        │
        ▼
Coder writes code (up to R retries with NL feedback)
        │
        ▼
Planner judges output
```

- The Coder receives ONLY the Planner's natural language description.
- On failure, the Planner provides **natural language feedback** (like a Reflexion-style verbal retry): "That's not what I meant, the function should handle X by doing Y."
- Same number of retry opportunities as the DYTM condition (fair comparison).
- **No PBTs, no contracts, no structured verification.**

#### Condition B: DYTM Protocol

```
Planner writes NL description
        │
        ▼
Spec generates PBT contract (from NL description only)
        │
        ▼
Translator renders CNL (deterministic, no LLM)
        │
        ▼
Planner (as Executive) reviews CNL
        │  approve / reject + feedback
        ▼
Coder writes code (from approved contract only — fresh context)
        │
        ▼
Doer runs Hypothesis PBTs
        │  pass / fail + counterexample
        ▼
Planner judges output
```

- The Planner acts as Executive: it reviews the CNL contract and approves or rejects.
- The Coder sees ONLY the approved PBT contract — **not** the original NL description.
- On PBT failure, the Coder retries against the locked contract (existing DYTM loop).
- After the DYTM loop completes, the Planner judges the final output.

> ⚠️ **HARD REQUIREMENT — Adversarial Executive Prompting (Anti-Rubber-Stamp)**
>
> The Planner-as-Executive **MUST NOT** be a passive reviewer. At temperature=0 with a generic "approve or reject" prompt, the Planner will rubber-stamp any contract that looks structurally sound — defeating the entire purpose of the checkpoint. The Executive prompt must force **adversarial, intent-anchored review**.
>
> **Executive review prompt** (used when Planner reviews the CNL):
>
> ```
> You are the Planner who wrote the original task. A Spec agent has
> produced the following verification contract based on your task
> description. You must decide whether this contract FULLY captures
> your intent.
>
> YOUR ORIGINAL TASK:
> {description}
>
> YOUR PRIVATE INTENT (what you actually meant — the Coder will
> never see this):
> {intent_summary}
>
> THE PROPOSED CONTRACT:
> {cnl_text}
>
> REVIEW INSTRUCTIONS — Be adversarial. Ask yourself:
> 1. Does this contract MISS any implicit constraints from your intent?
> 2. Does it misinterpret any domain terms or boundary conditions?
> 3. Would a correct implementation of this contract STILL fail to
>    match what you actually wanted? If yes → REJECT.
> 4. Are any edge cases you care about absent from the contract?
> 5. Could a Coder satisfy every property in this contract and yet
>    produce the WRONG function? If yes → REJECT.
>
> You must output:
> - approved: true / false
> - feedback: (if rejected) what specifically is missing or wrong
>
> BIAS: When in doubt, REJECT. A false approval wastes the Coder's
> entire attempt. A false rejection only costs one Spec retry.
> ```
>
> **Rejection rate guardrail**: If the Planner approves >90% of contracts on the first attempt across all 50 tasks, the Executive prompt is too weak and must be tightened before results are considered valid. Target: 20–50% first-attempt rejection rate (indicating the checkpoint is doing real work).

### 2.4 Controls

- **Same retry budget**: Both conditions get the same maximum number of interaction rounds between Planner and Coder.
- **Same Coder model**: Identical model and temperature for the Coder in both conditions.
- **Same tasks**: Both conditions run on all 50 tasks. Order randomized.
- **Planner temperature = 0** for judging (deterministic evaluation).

---

## 3. Metrics

### 3.1 Primary Metrics (Communication Efficiency)

| Metric | Definition | Why It Matters |
|---|---|---|
| **Misunderstanding Detection Rate** | % of tasks where a misunderstanding was caught *before* the Coder wrote any code (DYTM: contract rejection; Baseline: N/A — always 0%) | Core thesis: early detection prevents wasted work |
| **Redo Rate** | % of tasks where the Coder's *first complete attempt* was judged by the Planner as "wrong intent" (not just buggy — fundamentally misunderstood) | Measures intent drift that survived to execution |
| **Rounds-to-Agreement** | Number of Planner↔Coder interaction rounds before the Planner accepts the output | Measures communication cost |
| **Total LLM Calls** | Sum of all LLM invocations across the full pipeline per task | Compute cost |

### 3.2 Secondary Metrics (Quality)

| Metric | Definition | Why It Matters |
|---|---|---|
| **Planner Satisfaction** | Planner LLM judges final output: "Does this match my intent?" (5-point scale or binary) | Intent fidelity from the intent-holder's perspective |
| **First-Attempt Alignment** | % of tasks where the Coder's first attempt matched the Planner's intent (no retries needed) | Measures initial communication clarity |
| **Contract Faithfulness** | (DYTM only) Planner rates the approved CNL contract: "Does this capture what I want?" | Measures Spec quality independently of Coder quality |

### 3.3 Diagnostic Metrics

| Metric | Definition |
|---|---|
| **Rejection Reason Taxonomy** | Categorize why the Planner rejected contracts (missing constraint, wrong interpretation, over-constrained, etc.) |
| **Drift Category** | Categorize the type of intent drift when it occurs (boundary condition, implicit constraint, domain term, etc.) — cross-reference with deterministic taxonomy from HumanEval experiments |
| **Wall-Clock Time** | Time-to-completion per task |
| **Token Cost** | Total tokens consumed per task per condition |

---

## 4. Planner Judgment Protocol

The Planner judges the Coder's output in a structured evaluation. This is the "ground truth" and must be carefully prompted.

### Judgment prompt template:

```
You are the Planner who originally created this task. You will now judge
whether the Coder's output matches YOUR intent.

YOUR ORIGINAL TASK:
{description}

YOUR PRIVATE INTENT SUMMARY:
{intent_summary}

THE CODER'S OUTPUT:
{code}

Judge the output on these dimensions:
1. INTENT MATCH (1-5): Does the code do what you meant? Not "is it correct
   code" but "is it the right thing?"
   - 5 = Perfect match to my intent
   - 4 = Mostly right, minor misunderstanding
   - 3 = Partially right, significant misunderstanding on one axis
   - 2 = Fundamentally misunderstood on multiple axes
   - 1 = Completely wrong thing

2. DRIFT DETECTED (yes/no): Is there any gap between what you meant and
   what the Coder built?

3. DRIFT DESCRIPTION (if yes): What specifically was misunderstood?

4. DRIFT CATEGORY: One of [boundary_condition, implicit_constraint,
   domain_term, edge_case, return_type, error_handling, ordering,
   performance, other]
```

### Why Planner-as-Judge is valid here

Unlike HumanEval (where the "judge" is a static test suite), the Planner *generated* the task. It holds the intent in its context window. Its judgment is: "does this match what I was thinking when I wrote the task?" This is a legitimate measurement of intent transmission fidelity.

**Limitation**: The Planner is still an LLM and its judgment is probabilistic. Mitigate by:
- Running each judgment 3× and taking majority vote
- Using temperature=0
- Including the `intent_summary` (private, never shown to Coder) as an anchor

---

## 5. Implementation Plan

### 5.1 New Files

```
eval/
├── a2a/
│   ├── __init__.py
│   ├── planner.py              # Task generation + Planner-as-Executive + Planner-as-Judge
│   ├── task_bank.py            # Generated/curated task inventory (50 tasks)
│   ├── run_a2a_baseline.py     # Condition A: NL-only delegation
│   ├── run_a2a_dytm.py         # Condition B: DYTM protocol delegation
│   ├── judge.py                # Planner judgment protocol
│   ├── metrics.py              # Compute all metrics from result CSVs
│   └── results/                # Output CSVs
```

### 5.2 Changes to Existing Code

| File | Change | Reason |
|---|---|---|
| `dytm/graph.py` | Add `planner_as_executive` option to `build_graph()` | Wire Planner LLM as the Executive approval node |
| `dytm/nodes/approval.py` | Support external model as Executive (already partially implemented) | Planner reviews CNL with its original intent in context |
| None (new files only) | — | The existing DYTM pipeline is reused as-is; only the eval harness is new |

### 5.3 Build Phases

| Phase | Work | Complexity |
|---|---|---|
| **Phase 1: Task Bank** | Generate/curate 50 tasks with Planner. Review and filter for quality. Ensure ambiguity axes are covered. | 🟡 Medium — task quality is critical |
| **Phase 2: Baseline Harness** | `run_a2a_baseline.py`: Planner → NL → Coder → NL feedback loop → Planner judges | 🟢 Low — straightforward LLM chain |
| **Phase 3: DYTM Harness** | `run_a2a_dytm.py`: Planner → DYTM pipeline (Planner as Executive) → Planner judges | 🟢 Low — reuses existing `build_graph()` |
| **Phase 4: Judgment + Metrics** | `judge.py` + `metrics.py`: structured evaluation, 3× majority vote, metric computation | 🟢 Low |
| **Phase 5: Run + Analyze** | Execute both conditions, compute metrics, generate tables | 🟢 Low |

### 5.4 Estimated Cost

| Item | Estimate |
|---|---|
| Task generation (50 tasks × ~1K tokens each) | ~$0.50 |
| Baseline condition (50 tasks × ~5 rounds × ~2K tokens) | ~$5 |
| DYTM condition (50 tasks × Spec + Translator + Executive + Coder + Doer) | ~$10 |
| Judgment (50 tasks × 2 conditions × 3 repeats × ~1K tokens) | ~$3 |
| **Total** | **~$20** |

---

## 6. Expected Outcomes

### 6.1 Hypothesis

DYTM reduces redo rate and rounds-to-agreement by catching misunderstandings at the contract stage. Specifically:

- **Misunderstanding Detection Rate**: Baseline = 0% (no checkpoint), DYTM > 20% (contract rejections that prevent wrong code)
- **Redo Rate**: DYTM < Baseline (fewer "built the wrong thing" failures)
- **Planner Satisfaction**: DYTM ≥ Baseline (intent fidelity is equal or better)
- **Total LLM Calls**: DYTM may be higher per-task (more pipeline stages), but lower in *wasted* calls (fewer full redos)

### 6.2 What Would Be Convincing

| Result | Interpretation |
|---|---|
| DYTM redo rate < 50% of baseline | Strong evidence for intent drift reduction |
| DYTM catches >15% of misunderstandings at contract stage | Validates the "checkpoint" mechanism |
| Planner satisfaction ≥ baseline despite more pipeline stages | DYTM doesn't hurt and may help |
| DYTM reduces total rounds-to-agreement | Counter-intuitive but powerful: the contract stage *adds* a round but *prevents* multiple redo rounds |

### 6.3 What Would Be Unconvincing

| Result | Interpretation |
|---|---|
| Redo rates are similar | Either tasks aren't ambiguous enough, or NL communication is sufficient for these tasks |
| DYTM adds cost without reducing redos | The contract stage is overhead without benefit — Planner-as-Executive doesn't catch drift |
| Planner satisfaction is lower with DYTM | The contract constrains the Coder in harmful ways |

### 6.4 Failure Modes to Watch

- **Tasks too easy**: If the Coder gets most tasks right on first attempt in both conditions, there's no room for DYTM to help. Ensure tasks are complex enough.
- **Planner-as-Executive rubber-stamps**: If the Planner approves >90% of first-attempt contracts, the Executive prompt is too weak. This is a **validity threat** — results under a rubber-stamping Executive are not interpretable. Tighten the adversarial prompt (see §2.3 Condition B) and re-run before reporting. Target first-attempt rejection rate: 20–50%.
- **Judgment noise**: If the Planner's satisfaction scores are inconsistent across 3× repeats, the measurement is unreliable. Report inter-agreement and consider human spot-checks on a subset.

---

## 7. Why This Escapes "LLMs All the Way Down"

The HumanEval problem: every component interprets the same static text, errors are correlated, and there's no independent intent-holder.

The A2A setup differs in three critical ways:

1. **The Planner IS the intent-holder.** It generated the task. It knows what it wants because it *defined* what it wants. When it reviews the CNL contract, it's comparing against its own mental model — not re-interpreting a static docstring.

2. **We're measuring communication, not correctness.** The metric isn't "does the code pass canonical tests" — it's "did the agents agree on what to build before building it?" This is an internal property of the system, observable without external ground truth.

3. **The ground truth doesn't need to be objectively "right."** The Planner's intent might be weird, suboptimal, or even wrong. That doesn't matter. What matters is: did the Coder build *what the Planner asked for*? DYTM is a fidelity protocol, not a correctness oracle.

**Analogy**: TCP doesn't guarantee you're sending the right data. It guarantees the data *arrives as sent*. DYTM is TCP for agent intent.

---

## 8. Relation to Prior Work

This experiment extends the DYTM project from benchmark evaluation (HumanEval, §3.1–3.5 of README) to **protocol evaluation**. It tests a different thesis:

| | HumanEval Experiments | A2A Experiment |
|---|---|---|
| **Thesis** | DYTM improves code generation accuracy | DYTM improves agent communication fidelity |
| **Ground truth** | Human-written canonical tests | Planner's self-reported intent |
| **Primary mechanism tested** | PBT retry loop (secondary) | Intent verification via CNL (primary) |
| **Independent intent-holder** | None (static docstring) | Planner LLM |
| **Escapes correlated errors** | No | Partially (different models, independent judgment) |
| **Commercially relevant to** | Code generation tools | Multi-agent orchestration platforms |

---

## 9. Open Questions

1. **Should the Planner see the Coder's code, or only the output/behavior?** Seeing code allows judging implementation intent; seeing only output tests black-box behavior. Recommend: show the code (the Planner is a technical agent, not an end user).

2. **How to handle tasks where the Planner's description is unambiguous?** These are "control" tasks — both conditions should succeed. If DYTM adds overhead on unambiguous tasks without benefit, that's useful to report (cost of the protocol on easy tasks).

3. **Should the baseline Coder get the Planner's feedback verbatim, or through a structured channel?** Recommend: verbatim NL feedback, to make the baseline as strong as possible. DYTM should win on the *hard* tasks, not by handicapping the baseline.

4. **~~Cross-model design~~** — RESOLVED: Cross-family decorrelation is a hard requirement (see §2.1). Primary: Planner = Claude, Spec/Coder = GPT. Optional ablation: swap families and report both.

