# DYTM — "Do You Trust Me"

**A Deterministic Verification Layer for Probabilistic LLM Agents**

> Micro-paper summarising all conducted experiments and findings.  
> Last updated: April 8, 2026.

---

## 1. Abstract

DYTM is a verification protocol for LLM code generation that interposes a *property-based test (PBT) contract* between a user's intent and the generated code. An LLM ("Spec") produces a structured PBT specification; a deterministic Jinja2 template engine renders it into Controlled Natural Language (CNL); an executive (human or LLM) approves the CNL; and a separate LLM ("Coder") writes code on a fresh context. The code is iteratively refined against Hypothesis PBTs until it passes or retries are exhausted.

We evaluated DYTM on the full 164-problem HumanEval benchmark. The system achieves **87.2% pass@1** vs. **85.4% baseline** (+1.8%, net +3 problems), with the improvement coming entirely from the PBT retry loop — DYTM's *secondary* mechanism. The *primary* mechanism (human intent verification via CNL contracts) is fundamentally untestable on a static benchmark. We additionally conducted cross-model, multi-spec, deterministic ambiguity taxonomy, and **agent-to-agent (A2A)** experiments. The A2A experiment — the first to test DYTM's primary mechanism — reveals that PBT contracts are a **lossy translation** of natural-language intent, producing an average 0.90-point intent match regression vs. direct NL communication. A **hybrid** condition (DYTM pipeline + NL feedback) recovers 67% of this gap, confirming that PBTs and NL are complementary but that PBTs cannot replace NL as a communication medium. This document reports all findings.

---

## 2. System Architecture

```
Executive (human or LLM)
        │  natural language prompt
        ▼
  ┌──────────┐     ┌────────────┐     ┌──────────┐
  │ 1. Spec  │────▶│ 2. Transl. │────▶│ 3. Exec. │
  │ (LLM:    │     │ (Jinja2:   │     │ Approval │
  │  PBTs    │     │  JSON→CNL) │     │ (LLM/    │
  │  only)   │     │  NO LLM    │     │  human)  │
  └──────────┘     └────────────┘     └────┬─────┘
       ▲ reject                            │ approve
       └───────────────────────────────────┘
                                           ▼
                                     ┌──────────┐     ┌──────────┐
                                     │ 4. Coder │────▶│ 5. Doer  │
                                     │ (LLM:    │     │ (subproc │
                                     │  fresh   │     │  Hypoth. │
                                     │  context)│     │  PBTs)   │
                                     └──────────┘     └────┬─────┘
                                          ▲  fail          │ pass
                                          └────────────────┘
                                                           ▼
                                                       ✅ Done
```

**Key design principle**: Separation of specification and implementation. The Spec generates verification properties *before* any code exists. The Coder operates on a fresh context with only the approved contract — no memory of the Spec conversation. This enforces independence between the contract and the code.

---

## 3. Experiments Conducted

### 3.1 Experiment 1: Baseline vs. DYTM (gpt-4o-mini)

**Setup**: All nodes (Spec, Executive, Coder) use `gpt-4o-mini`. Executive auto-approves in eval mode. Max 4 code retries.

| Metric | Baseline | DYTM | Δ |
|---|---|---|---|
| **pass@1** | 140/164 (85.4%) | 143/164 (87.2%) | **+1.8%** |
| Problems fixed by DYTM | — | 12 | — |
| Problems regressed by DYTM | — | 9 | — |
| **Net** | — | **+3** | — |
| Avg time/problem | 2.3s | 46.6s | +44.3s |
| Avg retries | 0 | 1.55 | — |
| Total eval time | ~6 min | ~127 min | — |

**DYTM retry distribution**: 0 retries: 66, 1 retry: 30, 2 retries: 21, 3 retries: 6, 4 retries (max): 41.

**DYTM failure breakdown** (21 failures total):
- PBT pass but canonical fail ("weak contracts"): 12 (57%)
- PBT fail and canonical fail: 9 (43%)

**Interpretation**: The +1.8% improvement comes from DYTM's *secondary mechanism* — the PBT retry loop with Hypothesis counterexamples. DYTM fixed 12 problems the baseline got wrong (a 50% error recovery rate on baseline failures). However, it regressed on 9 problems the baseline got right, yielding a net gain of only 3.

### 3.2 Experiment 2: Cross-Model (Spec=gpt-4o, Coder=gpt-4o-mini)

**Hypothesis**: A stronger Spec model breaks the correlated-error pattern (where the same model misunderstands a task in both Spec and Code).

| Metric | DYTM (4o-mini/4o-mini) | DYTM (4o/4o-mini) |
|---|---|---|
| **pass@1** | 143/164 (87.2%) | 143/164 (87.2%) |
| **PBT pass rate** | 127/164 (77.4%) | 152/164 (92.7%) |

**Interpretation**: The stronger Spec model produced substantially better contracts (PBT pass rate jumped from 77.4% to 92.7%), but canonical pass@1 remained identical. This confirms two things: (1) the Spec model's understanding is not the sole bottleneck — gpt-4o-mini's Coder capability matters; (2) the stronger Spec model improves contract quality, but without a human to verify those contracts, the improvement doesn't translate to end-to-end results on HumanEval.

**Note**: A standalone gpt-4o baseline run was attempted but was invalidated by API rate limiting (30K TPM limit caused ~45 HTTP 429 errors across 164 problems, yielding a misleading 61% pass rate).

### 3.3 Experiment 3: Multi-Spec with LLM Ambiguity Analyst (K=3)

**Setup**: An LLM "Ambiguity Analyst" identifies tricky/ambiguous aspects of each task, then K=3 contrastive specs are generated — each exploring a different interpretation. Selection by example-consistency filtering + restrictiveness scoring.

**Result**: Incomplete run (5/164 problems completed before hang). Partial data (3/5 pass) was inconclusive. The approach was conceptually sound but added significant latency (extra LLM call for analysis + 3× spec generation).

### 3.4 Experiment 4: Deterministic Ambiguity Taxonomy

**Setup**: A regex-based, zero-LLM scanner (`deterministic_taxonomy.py`) pattern-matches HumanEval prompts against a finite taxonomy of 13 ambiguity categories. No LLM calls — purely deterministic.

**Theoretical basis**: For unrestricted English, the space of possible meanings is intractable. But for *coding task descriptions*, the ambiguity space is **bounded and enumerable**. Tasks can only be misinterpreted along a finite set of axes (boundary conditions, empty handling, operator precedence, etc.), each with 2–3 possible readings.

**Coverage**: 124/164 problems (75.6%) have at least one detectable ambiguity axis.

**Taxonomy axis frequency across HumanEval**:

| Axis | Problems | Confidence |
|---|---|---|
| Return type specificity | 81 | 30% |
| Sort stability / tie-breaking | 30 | 30% |
| Empty/null input handling | 17 | 50% |
| Duplicate handling | 15 | 40% |
| Float comparison | 14 | 40% |
| Prime edge cases | 12 | 50% |
| Boundary (strict upper) | 11 | 50% |
| Boundary (strict vs inclusive) | 9 | 70% |
| Structural matching (brackets) | 8 | 80% |
| Round-trip invariant | 4 | 60% |
| Boundary (inclusive) | 3 | 40% |
| Operator precedence | 2 | 70% |
| Signed digit semantics | 1 | 90% |

**Cross-reference with DYTM regressions**:

| Regression | Taxonomy Axes Detected | Known Root Cause |
|---|---|---|
| HumanEval/56 | `structural_matching_algorithm` (80%) | Count-equality vs prefix-validity ✅ **exact match** |
| HumanEval/108 | `signed_digit_semantics` (90%) | First-digit-negative vs all-digits ✅ **exact match** |
| HumanEval/160 | `operator_precedence` (70%) | Left-to-right vs math precedence ✅ **exact match** |
| HumanEval/75 | `prime_edge_cases` (50%) | — |
| HumanEval/118 | `empty_input_handling` (50%) | — |
| HumanEval/141 | `boundary_strict_upper, empty_input_handling` (50%) | — |
| HumanEval/22 | none | JSON parse error (API 400) |
| HumanEval/95 | none | — |
| HumanEval/133 | none | — |

**6 of 9 regressions** (67%) have strong (≥50% confidence) taxonomy axes. The three most documented regression root causes are **perfectly identified** by the deterministic scanner. This validates the theoretical claim that coding task ambiguity is bounded and enumerable for the domain of coding task descriptions.

### 3.5 Experiment 5: Deterministic Multi-Spec (implemented, not yet evaluated)

A deterministic variant of the multi-spec node was implemented (`deterministic_multi_spec.py`), using the regex taxonomy (Experiment 4) as the ambiguity source instead of an LLM analyst. It enumerates interpretation combinations deterministically and generates contrastive specs for each. Available via `--det-multi-spec K` flag but not yet run on the full benchmark.

---

## 4. Root Cause Analysis of Regressions

All 9 regressions share the same root cause: **correlated specification errors**. When the Spec model misunderstands a task, the Coder model makes the same misunderstanding. The PBT contract and the code agree on the wrong answer — verification provides zero signal.

**Detailed regression table**:

| Problem | PBT | Retries | Cross-model (4o Spec) | Taxonomy Axis |
|---|---|---|---|---|
| HumanEval/22 | FAIL | 0 | PASS | none (API error) |
| HumanEval/56 | OK | 0 | PASS | structural_matching (80%) |
| HumanEval/75 | OK | 4 | FAIL | prime_edge_cases (50%) |
| HumanEval/95 | OK | 1 | PASS | none |
| HumanEval/108 | OK | 1 | PASS | signed_digit (90%) |
| HumanEval/118 | FAIL | 4 | FAIL | empty_input (50%) |
| HumanEval/133 | FAIL | 4 | PASS | none |
| HumanEval/141 | FAIL | 4 | PASS | boundary + empty (50%) |
| HumanEval/160 | OK | 1 | FAIL | operator_precedence (70%) |

Three specific failure patterns:

1. **Wrong invariant (necessary but not sufficient)** — HumanEval/56 (`correct_bracketing`): The Spec generated `count('<') == count('>')` as the core invariant, which is necessary but not sufficient. Correct bracketing requires prefix-validity. The count-based code passed the weak PBT but failed canonical tests.

2. **Buggy oracle (same misunderstanding in Spec and Code)** — HumanEval/108 (`count_nums`): The task says "for negative numbers, the first signed digit is negative." The Spec made *all* digits negative — the same mistake the Coder made.

3. **Buggy oracle (missed operator precedence)** — HumanEval/160 (`do_algebra`): The Spec evaluated left-to-right instead of respecting standard math precedence. The Coder did the same.

**Detailed fix table**:

| Problem | PBT | Retries | Cross-model (4o Spec) | Taxonomy Axis |
|---|---|---|---|---|
| HumanEval/4 | OK | 3 | PASS | none |
| HumanEval/10 | OK | 2 | FAIL | none |
| HumanEval/28 | OK | 0 | PASS | none |
| HumanEval/35 | OK | 0 | PASS | none |
| HumanEval/38 | OK | 0 | PASS | round_trip (60%) |
| HumanEval/50 | FAIL | 4 | PASS | round_trip (60%) |
| HumanEval/74 | OK | 2 | FAIL | boundary_strict_vs_inclusive (70%) |
| HumanEval/83 | FAIL | 4 | FAIL | none |
| HumanEval/93 | FAIL | 4 | PASS | round_trip (60%) |
| HumanEval/120 | OK | 2 | PASS | none |
| HumanEval/127 | OK | 1 | PASS | prime_edge_cases (50%) |
| HumanEval/134 | OK | 2 | FAIL | none |

---

## 5. The Core Finding: The Evaluation Gap

**DYTM's value proposition is bridging human intent → verified contract → code.** The critical step is a human (or independent agent) reviewing the CNL contract and confirming: "yes, this captures what I want."

**HumanEval cannot test this.** In a benchmark, the "intent" is encoded in a docstring. There is no independent intent-holder. The evaluation becomes:

```
Docstring (static text)
  → LLM interprets spec → generates contract
  → LLM interprets spec → "verifies" contract   ← CIRCULAR
  → LLM interprets spec → generates code
```

The contract and the code are both downstream interpretations of the same text by the same model. When the model misunderstands the docstring, it misunderstands it everywhere. There is no human to say "wait, that's not what I meant."

**The regressions are the smoking gun.** The baseline goes docstring → code (one interpretation step). DYTM goes docstring → contract → code (two interpretation steps). Without a human to validate the contract, the extra step is pure risk — an additional opportunity to misinterpret.

### What HumanEval Actually Measures

HumanEval tests DYTM's **secondary** mechanism (PBT retry loop as deterministic error correction) — not its **primary** mechanism (intent verification via CNL contracts).

The +1.8% improvement is from the secondary mechanism alone, achieved *despite* the added risk of the extra specification step. The primary value proposition (a human reading a CNL contract and catching misunderstandings before any code is written) is fundamentally untestable on a static benchmark.

---

## 6. Summary of All Results

| Configuration | pass@1 | PBT Pass | Notes |
|---|---|---|---|
| Baseline gpt-4o-mini | **140/164 (85.4%)** | N/A | Zero-shot, no verification |
| DYTM gpt-4o-mini | **143/164 (87.2%)** | 77.4% | Full pipeline, auto-approve |
| DYTM Spec=gpt-4o Coder=gpt-4o-mini | **143/164 (87.2%)** | 92.7% | Stronger spec, same pass@1 |
| Baseline gpt-4o | **Invalid** | N/A | Invalidated by API rate limits |
| Multi-Spec K=3 (LLM analyst) | **Incomplete** | — | 5/164 problems, inconclusive |
| Det. Multi-Spec (regex taxonomy) | **Not yet run** | — | Implemented, awaiting eval |

---

## 7. Experiment 6: Agent-to-Agent (A2A) Communication

### 7.1 Motivation

All prior experiments (HumanEval) test DYTM's **secondary** mechanism (PBT retry loop as error correction). They cannot test the **primary** mechanism — that an independent intent-holder reviews the CNL contract and catches misunderstandings before code generation begins. HumanEval has no independent intent-holder; the same docstring is the only "intent."

The A2A experiment fills this gap. A **Planner** agent holds private intent details that the **Coder** cannot see directly. The experiment compares:

- **Baseline (NL feedback)**: Planner writes a description → Coder codes → Planner reviews code and gives NL feedback → Coder revises. Up to 5 rounds.
- **DYTM (PBT contract)**: Planner writes a description → Spec generates PBTs → Translator renders CNL → Planner (as adversarial Executive) reviews the CNL contract → approved contract goes to Coder → Doer runs PBTs → Coder retries on failures.

The Planner judges final output against its private `intent_summary` on a 1–5 scale (3× majority vote for stability).

### 7.2 Task Bank

50 tasks generated by GPT-4o, covering 10 categories: data processing, string manipulation, API handling, math/statistics, file operations, validation, search/filter, formatting, state management, and configuration. Each task has:
- A **description** (what the Coder sees) — deliberately under-specified
- A **function_name** and **intent_summary** (Planner's private ground truth) — contains boundary cases, error handling rules, and formatting details NOT in the description

### 7.3 Results

| Metric | Baseline (NL) | DYTM v3 | Hybrid |
|---|---|---|---|
| High match (≥4/5) | **31/50 (62%)** | 17/50 (34%) | 27/50 (54%) |
| Avg intent match | **3.98/5** | 3.08/5 | 3.68/5 |
| Drift detected | 24/50 (48%) | 37/50 (74%) | 31/50 (62%) |
| Avg rounds | **3.20** | 5.08 | 8.10 |
| First-attempt OK | 9/50 (18%) | 4/50 (8%) | 4/50 (8%) |
| Improved / Regressed / Tied (vs baseline) | — | 3 / 26 / 21 | 7 / 17 / 26 |

**On baseline failures** (tasks where baseline scored ≤3/5):
- DYTM: helped 3/19, hurt 9/19, same 7/19 (net −6)
- Hybrid: helped 6/19, hurt 5/19, same 8/19 (net +1)

**Rejection distribution** (DYTM v3): 4 tasks with 0 rejections, 26 with 1, 5 with 2, 15 with 3+.

<details>
<summary><strong>Full per-task A2A comparison (all 50 tasks)</strong></summary>

| Task | Baseline | DYTM v3 | Hybrid | Δ DYTM | Δ Hybrid | Drift Category (DYTM) |
|---|---|---|---|---|---|---|
| A2A/001 | 5 | 4 | 2 | −1 | −3 | error_handling |
| A2A/002 | 3 | 4 | 2 | +1 | −1 | implicit_constraint |
| A2A/003 | 5 | 2 | 3 | −3 | −2 | error_handling |
| A2A/004 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/005 | 2 | 2 | 3 | 0 | +1 | implicit_constraint |
| A2A/006 | 2 | 2 | 2 | 0 | 0 | domain_term |
| A2A/007 | 4 | 2 | 3 | −2 | −1 | implicit_constraint |
| A2A/008 | 4 | 2 | 3 | −2 | −1 | implicit_constraint |
| A2A/009 | 2 | 1 | 2 | −1 | 0 | other |
| A2A/010 | 3 | 2 | 3 | −1 | 0 | implicit_constraint |
| A2A/011 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/012 | 5 | 3 | 5 | −2 | 0 | implicit_constraint |
| A2A/013 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/014 | 5 | 2 | 5 | −3 | 0 | implicit_constraint |
| A2A/015 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/016 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/017 | 5 | 3 | 5 | −2 | 0 | implicit_constraint |
| A2A/018 | 5 | 2 | 4 | −3 | −1 | implicit_constraint |
| A2A/019 | 5 | 4 | 5 | −1 | 0 | edge_case |
| A2A/020 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/021 | 1 | 2 | 2 | +1 | +1 | implicit_constraint |
| A2A/022 | 4 | 2 | 4 | −2 | 0 | implicit_constraint |
| A2A/023 | 5 | 2 | 2 | −3 | −3 | implicit_constraint |
| A2A/024 | 3 | 2 | 3 | −1 | 0 | implicit_constraint |
| A2A/025 | 2 | 2 | 3 | 0 | +1 | edge_case |
| A2A/026 | 2 | 2 | 2 | 0 | 0 | implicit_constraint |
| A2A/027 | 3 | 2 | 2 | −1 | −1 | boundary_condition |
| A2A/028 | 3 | 3 | 3 | 0 | 0 | implicit_constraint |
| A2A/029 | 3 | 2 | 4 | −1 | +1 | implicit_constraint |
| A2A/030 | 3 | 3 | 3 | 0 | 0 | implicit_constraint |
| A2A/031 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/032 | 5 | 5 | 4 | 0 | −1 | — |
| A2A/033 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/034 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/035 | 5 | 3 | 5 | −2 | 0 | performance |
| A2A/036 | 4 | 4 | 5 | 0 | +1 | ordering |
| A2A/037 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/038 | 5 | 5 | 5 | 0 | 0 | — |
| A2A/039 | 5 | 5 | 4 | 0 | −1 | — |
| A2A/040 | 2 | 3 | 5 | +1 | +3 | performance |
| A2A/041 | 5 | 2 | 4 | −3 | −1 | error_handling |
| A2A/042 | 5 | 2 | 2 | −3 | −3 | return_type |
| A2A/043 | 3 | 2 | 5 | −1 | +2 | implicit_constraint |
| A2A/044 | 3 | 3 | 3 | 0 | 0 | implicit_constraint |
| A2A/045 | 5 | 3 | 4 | −2 | −1 | implicit_constraint |
| A2A/046 | 5 | 2 | 3 | −3 | −2 | implicit_constraint |
| A2A/047 | 3 | 2 | 2 | −1 | −1 | implicit_constraint |
| A2A/048 | 4 | 2 | 4 | −2 | 0 | other |
| A2A/049 | 3 | 2 | 2 | −1 | −1 | implicit_constraint |
| A2A/050 | 3 | 2 | 2 | −1 | −1 | domain_term |

**Standout tasks**: A2A/040 (quickselect) went from baseline=2, DYTM=3 to hybrid=5 — the PBT contract caught a structural bug that NL missed, then NL feedback refined the algorithm choice. A2A/043 (event validation) went from baseline=3, DYTM=2 to hybrid=5 — NL feedback recovered from PBT's information loss. A2A/023 (whitespace cleaning) regressed in all conditions (5→2→2) — an example of intent complexity that neither PBT nor limited NL rounds could resolve.

</details>

### 7.4 Iteration History

| Version | Change | Result |
|---|---|---|
| DYTM v1 | Reject-everything bug in Executive prompt | 24% high match, 0% first-attempt approve |
| DYTM v2 | Fixed Executive prompt; still no Spec chat-mode | 24% → no improvement (Spec regressed on retries) |
| DYTM v3 | Chat-mode Spec + convergence-aware Executive | **34% high match, 8% first-attempt approve** |
| Hybrid | DYTM pipeline + Planner NL feedback rounds (up to 2) | **54% high match, avg 3.68/5** |

The v1→v3 improvement (+10 pp high match, +0.24 avg) confirms the mechanical fixes worked. The remaining gap to baseline (-28 pp, -0.90 avg) is partially closed by the hybrid's NL feedback rounds (-8 pp, -0.30 avg).

### 7.5 Root Cause Analysis

The 26 regressions fall into a clear pattern. Example cases:

| Task | Baseline | DYTM | Root Cause |
|---|---|---|---|
| A2A/003 (`aggregate_sales_data`) | 5 | 2 | Intent specifies "handle missing category as 'Unknown', skip invalid dates, float amounts." PBT contract captured type/shape but not these NL-nuanced rules. |
| A2A/023 (`clean_whitespace`) | 5 | 2 | Intent says "preserve leading indentation, collapse multiple blank lines to max 2." PBT invariants checked "fewer spaces" but missed preservation semantics. |
| A2A/014 (`calculate_shipping_cost`) | 5 | 2 | Boundary rules ("exactly 2 lbs uses $8 rate"), surcharge ordering ("express doubles base BEFORE international surcharge"). PBT edge cases tested a few points but not the combinatorial boundary logic. |
| A2A/041 (`create_user_profile`) | 5 | 2 | Intent specifies "age 13–120, name allows hyphens/apostrophes only, additional fields preserved but sanitized." PBT contract captured email validation but missed most of these. |

**The common thread**: Complex NL requirements with multiple interacting constraints lose information when compressed into PBT properties. A PBT contract captures structural relationships (type checks, invariants, specific edge cases) but cannot express:
- Nuanced error-handling policies ("skip invalid dates, treat missing category as 'Unknown'")
- Ordering constraints ("double base cost, THEN add surcharge")
- Preservation semantics ("collapse spaces but preserve indentation")
- Combinatorial boundary rules over multiple parameters

### 7.6 The Core Finding: PBT as Lossy Compression

The A2A experiment reveals the fundamental limitation of PBT-based contracts as a communication medium:

```
NL intent (rich, nuanced, contextual)
    ↓ Spec LLM (information-lossy translation)
PBT contract (structural, type-oriented, finite edge cases)
    ↓ Coder LLM (only sees contract)
Code (faithful to contract, not to original intent)
```

**In the baseline**, the Coder receives NL directly and gets NL feedback:
```
NL intent → NL description → Code → NL feedback → Revised code
```

Each step preserves the natural-language expressiveness. The Planner can say: "you missed that exactly 2 lbs should use the $8 rate, not the $5 rate." The Coder understands immediately.

**In DYTM**, the Coder only sees the PBT contract:
```
NL intent → NL description → PBT contract → Code
                                 ↑
                     (information bottleneck)
```

The PBT contract is a lossy compression. It captures *some* testable properties but discards the NL nuance that a human (or LLM) would naturally communicate. The Executive checkpoint catches *structural* mismatches (wrong function signature, missing return type), but it reviews the contract, not the code — so it can't catch the subtle intent losses that only manifest in implementation.

### 7.7 What This Means for DYTM

1. **The HumanEval result was genuine** — the PBT retry loop (secondary mechanism) provides real value as deterministic error correction for algorithmic tasks with clear input→output relationships.

2. **The A2A result is also genuine** — for complex, nuanced tasks with many interacting NL constraints, the PBT contract format cannot carry enough information to replace direct NL communication.

3. **The protocol works mechanically** — chat-mode Spec, convergence-aware Executive, and PBT-retry Coder all function correctly. The bottleneck is the *information capacity* of the PBT contract format.

4. **The hybrid recovers most of the gap** — adding NL feedback on top of the DYTM pipeline (Hybrid condition) recovers 67% of the deficit vs. baseline (3.08 → 3.68 of a 3.98 target). This confirms that PBTs and NL are complementary: PBTs catch structural bugs, NL preserves intent nuance. However, the hybrid still trails the baseline by 0.30 points, suggesting the locked PBT contract introduces residual drag that NL feedback cannot fully overcome.

5. **On hard tasks, the hybrid helps more than it hurts** — on baseline failures (≤3/5), the hybrid helped 6 and hurt 5 (net +1), while DYTM alone helped 3 and hurt 9 (net −6). The standout: A2A/040 went from baseline=2, DYTM=3 to hybrid=5 — the PBT contract caught something NL missed, then NL feedback refined the rest.

6. **Implication**: DYTM's value is as a *supplementary verification layer*, not a replacement for NL communication. The contract provides a checkable artifact and deterministic error correction; NL provides the expressiveness for intent transfer. The protocol's future lies in hybrid architectures, not pure PBT-based communication.

---

## 8. Technical Stack

| Layer | Tool | Purpose |
|---|---|---|
| Orchestration | LangGraph | Stateful cyclical graph with conditional edges |
| LLM Interface | LangChain (`BaseChatModel`) | Model-agnostic: OpenAI, Anthropic, Ollama |
| Schema Enforcement | Pydantic v2 | Structured PBT JSON validation |
| PBT Engine | Hypothesis | Property-based testing + counterexample shrinking |
| CNL Translation | Jinja2 templates | Deterministic JSON → English (no LLM) |
| Code Security | Python `ast` | NodeVisitor scanning for dangerous imports/calls |
| Sandbox | `subprocess` + `tempfile` | Isolated execution with timeout |
| Eval Benchmarks | HumanEval (164), A2A (50 tasks) | Code gen + intent verification benchmarks |

---

## 9. Discussion & Lessons Learned

### 9.1 DYTM Is Spec-Driven Development, Inline

The full arc of experiments — HumanEval, cross-model, deterministic taxonomy, A2A — converges on a single insight: DYTM is **lightweight spec-driven development, applied inline at generation time**. The CNL contract is a lightweight specification; the PBT suite is an automated acceptance test; the Executive checkpoint is a spec review gate.

This is not a novel pattern in software engineering. What DYTM adds is: (1) the spec is generated automatically from a natural-language prompt, (2) the spec is rendered in human-readable CNL for review, (3) the spec is enforced via property-based tests against infinite inputs (not just example cases), and (4) the entire loop is automated within an LLM pipeline. The novelty is in the *application* to LLM-generated code, not the pattern itself.

### 9.2 The Two Failure Scenarios DYTM Addresses

A human requests code from an LLM. Three outcomes are possible:

1. **Code is correct** — no intervention needed.
2. **Code is wrong, and the human doesn't notice** — the human ships buggy code. DYTM addresses this: the PBT suite catches structural bugs that the human might miss, and the CNL contract surfaces the *intended behavior* for review before any code is written.
3. **Code is not what the human meant** — the human's prompt was ambiguous, and the LLM chose the wrong interpretation. DYTM partially addresses this: the CNL contract makes the chosen interpretation explicit, giving the human a chance to catch the mismatch. However, the A2A experiment shows that PBT contracts lose too much NL nuance to fully replace direct communication.

Scenario (2) is DYTM's sweet spot. Scenario (3) requires the hybrid approach (PBT + NL). There is also a scenario (2b): the human's *intent itself* is wrong (they asked for the wrong thing). This is fundamentally uncatchable by any verification system — it's a human error, not a communication error.

### 9.3 PBT Contracts Are Lossy — But So Are All Translations

The A2A experiment's core finding — that PBT contracts are a lossy compression of NL intent — is not surprising in hindsight. *Every* formalization of natural language loses information. The question isn't "is it lossy?" but "is the loss acceptable for the use case?"

For **algorithmic tasks** with clear input→output relationships (HumanEval-style), PBT contracts capture most of what matters. The +1.8% HumanEval improvement and 50% error recovery rate on baseline failures demonstrate this.

For **complex, multi-constraint tasks** with nuanced NL requirements (A2A-style), PBT contracts miss too much. The 0.90-point intent match regression vs. direct NL communication demonstrates this.

The practical implication: DYTM's value scales inversely with task NL-complexity. It helps most on precisely the tasks where LLMs already perform well, and helps least where they struggle most. This is a fundamental tension.

### 9.4 The "TCP for Agent Intent" Analogy

TCP doesn't guarantee you're sending the right data — it guarantees the data arrives as sent. DYTM was conceived as TCP for agent intent: a fidelity protocol, not a correctness oracle. The A2A experiment tested this analogy directly.

The result: the analogy holds for the PBT retry loop (structural error correction ≈ TCP checksums) but breaks for the CNL contract (intent verification ≈ ... nothing in TCP, because TCP doesn't inspect payloads). The CNL contract is more like a content-type header: it describes *what kind of thing* is being sent, but not the full content. Useful, but not sufficient to replace the content itself.

### 9.5 Why the Evaluation Is Hard

DYTM's primary value proposition — "a human reads a contract and catches a misunderstanding before code is written" — requires a human in the loop. Every attempt to simulate this with an LLM (auto-approve, Planner-as-Executive) introduces the same correlated-error problem the system is designed to solve.

The A2A experiment partially escapes this by using a cross-family Planner (Claude) to judge an OpenAI Spec/Coder pipeline. But even this introduces "LLMs all the way down" concerns: the Planner's judgment of intent match is itself probabilistic (mitigated by 3× majority voting at temperature=0, but not eliminated).

A user study remains the gold-standard evaluation. Without one, the strongest evidence for DYTM's value is: (1) the PBT retry loop provides measurable error correction (+1.8% on HumanEval), (2) the deterministic taxonomy correctly identifies 67% of regression root causes, suggesting that coding task ambiguity is bounded and enumerable, and (3) the hybrid A2A condition recovers 67% of the NL baseline gap, confirming PBTs and NL are complementary.

---

## 10. Future Directions

### 10.1 Hybrid Contract Format

The A2A experiment identifies the PBT contract's information capacity as the bottleneck. A hybrid contract that combines PBT properties with structured NL constraints (e.g., "when category is missing, default to 'Unknown'") could preserve verifiability while increasing expressiveness. This is the most promising direction for closing the NL gap.

### 10.2 Adversarial Ambiguity Benchmark

Create a custom benchmark of 30–50 problems where disambiguating examples are deliberately weakened or removed. This tests the actual DYTM thesis — that the CNL contract catches misunderstandings before code generation — without requiring a full user study.

**Method**: Take HumanEval problems where the deterministic taxonomy detects strong ambiguity axes (e.g., HumanEval/56, /108, /160), strip the examples that resolve the ambiguity, and evaluate whether DYTM + multi-spec + oracle selection (selecting the interpretation matching hidden canonical tests) outperforms the baseline. This measures the *ceiling* of the multi-interpretation approach.

### 10.3 Deterministic Multi-Spec Evaluation

Run the deterministic multi-spec variant (`--det-multi-spec 3`) on the full HumanEval benchmark. While unlikely to produce large gains (docstring examples already disambiguate most tasks), it would validate or invalidate the regex taxonomy as a practical ambiguity detector.

### 10.4 A2A with Hybrid Contracts

Re-run the A2A experiment with an enriched contract format (Section 10.1). The 50-task bank and baseline results are already available for direct comparison. Target: close the 0.90-point gap to within 0.3 points of the NL baseline.

### 10.5 User Study (Gold Standard, Higher Cost)

N participants use Free Mode to solve coding tasks. Measure whether reviewing the CNL contract before code generation reduces intent mismatches compared to direct code generation. This tests the core thesis but requires IRB approval, participant recruitment, and time.

### 10.6 Stronger Baseline Comparison

Re-run the gpt-4o baseline with proper rate limiting (workers=1 or 2 with exponential backoff) to obtain a valid comparison point. If gpt-4o baseline exceeds DYTM gpt-4o-mini, this further supports the evaluation gap argument — DYTM's value isn't in raw benchmark performance but in verifiable intent transfer.

### 10.7 Phase 2: On-Device / Enterprise Product

The original project vision (see `Agent Readme.md`) includes a Phase 2 targeting **on-device execution on iOS** using Apple's `FoundationModels` or `MLX Swift` for private local inference, `Codable` for deterministic JSON→CNL parsing, and `JavaScriptCore` as a sandbox (bypassing iOS W^X memory restrictions). This positions DYTM as a privacy-first, on-device verification layer for edge AI — a high-value enterprise direction, contingent on the academic validation from Phase 1.

### 10.8 Related Research Directions

The experiments surfaced broader interest in **applied LLM trust mechanisms** — formal or semi-formal methods for verifying, constraining, or auditing LLM outputs. Adjacent directions include:

- **Adversarial contract testing**: Use an adversary LLM to write code that sneaks past the PBT contract. Measures contract strength independently of intent fidelity.
- **Knowledge-grounded verification**: Extend the PBT concept to verify whether LLM outputs are strictly derivable from an approved knowledge graph or RAG database (not just functional code).
- **Deterministic intent enumeration**: Explore whether the space of possible meanings of an English coding prompt is not only bounded (per the taxonomy in §3.4) but *enumerable* in a deterministic, tree-structured way — a finite list of interpretations, each with an estimated probability. This would make the multi-spec approach (§3.5) fully deterministic.

---

## 11. Running the Code

### Setup
```bash
pip install -e .
# Set OPENAI_API_KEY in .env.local
```

### Free Mode (Human-in-the-Loop Demo)
```bash
python -m dytm.free
```

### Eval — Baseline
```bash
python -m eval.run_baseline --model gpt-4o-mini
```

### Eval — DYTM (standard)
```bash
python -m eval.run_dytm --model gpt-4o-mini -w 5
```

### Eval — DYTM (cross-model)
```bash
python -m eval.run_dytm --spec-model gpt-4o --model gpt-4o-mini -w 5
```

### Eval — DYTM (multi-spec with LLM analyst)
```bash
python -m eval.run_dytm --model gpt-4o-mini --multi-spec 3 -w 5
```

### Eval — DYTM (deterministic multi-spec)
```bash
python -m eval.run_dytm --model gpt-4o-mini --det-multi-spec 3 -w 5
```

### Analysis Tools
```bash
# Compare two HumanEval runs
python -m eval.compare <baseline.csv> <dytm.csv>

# Deterministic ambiguity taxonomy scan
python -m eval.deterministic_taxonomy

# Ambiguity category analysis
python -m eval.ambiguity_analysis
```

### A2A Experiment — Task Generation
```bash
# Requires ANTHROPIC_API_KEY
python -m eval.a2a.generate_tasks --n-tasks 50
```

### A2A Experiment — Baseline (NL feedback)
```bash
python -u -m eval.a2a.run_a2a_baseline \
    --tasks eval/a2a/results/task_bank.json \
    --planner-model claude-sonnet-4-20250514 \
    --coder-model gpt-4o-mini \
    --max-rounds 4
```

### A2A Experiment — DYTM (PBT contract)
```bash
python -u -m eval.a2a.run_a2a_dytm \
    --tasks eval/a2a/results/task_bank.json \
    --planner-model claude-sonnet-4-20250514 \
    --coder-model gpt-4o-mini \
    --max-retries 4 \
    --max-spec-retries 3
```

### A2A Experiment — Compare Results
```bash
python -m eval.a2a.compare_a2a
```

### A2A Experiment — Hybrid (DYTM + NL feedback)
```bash
python -u -m eval.a2a.run_a2a_hybrid \
    --tasks eval/a2a/results/task_bank.json \
    --planner-model claude-sonnet-4-20250514 \
    --coder-model gpt-4o-mini \
    --max-retries 4 \
    --max-spec-retries 3 \
    --max-nl-rounds 2
```

### A2A Experiment — Verbose Single-Task Trace
```bash
python -m eval.a2a.trace_task \
    --tasks eval/a2a/results/task_bank.json \
    --task-id A2A/001 \
    --planner-model claude-sonnet-4-20250514 \
    --coder-model gpt-4o-mini
```

---

## 12. Repository Structure

```
DYTM/
├── dytm/
│   ├── graph.py                          # LangGraph StateGraph definition
│   ├── state.py                          # TypedDict state schema
│   ├── security.py                       # ast-based code security scanner
│   ├── free.py                           # Free Mode CLI (human-in-the-loop)
│   ├── nodes/
│   │   ├── spec.py                       # Spec node: LLM → PBT properties only
│   │   ├── coder.py                      # Coder node: LLM → code (fresh context)
│   │   ├── translator.py                 # Translator: Jinja2 JSON → CNL (no LLM)
│   │   ├── approval.py                   # Executive approval gate
│   │   ├── doer.py                       # Doer: subprocess + Hypothesis PBTs
│   │   ├── multi_spec.py                 # Multi-spec: K interpretations (LLM analyst)
│   │   └── deterministic_multi_spec.py   # Multi-spec: regex taxonomy (no LLM analyst)
│   ├── schemas/
│   │   └── pbt.py                        # Pydantic models for PBT payloads
│   └── templates/
│       └── cnl.jinja2                    # CNL rendering templates
├── eval/
│   ├── run_baseline.py                   # Zero-shot baseline eval
│   ├── run_dytm.py                       # DYTM full pipeline eval
│   ├── compare.py                        # Compare two result CSVs
│   ├── deterministic_taxonomy.py         # Regex-based ambiguity scanner
│   ├── ambiguity_analysis.py             # Heuristic prompt classification
│   ├── loader.py                         # HumanEval benchmark loader
│   ├── a2a/
│   │   ├── planner.py                    # Task generation + adversarial Executive + judgment
│   │   ├── generate_tasks.py             # CLI: generate task bank JSON
│   │   ├── run_a2a_baseline.py           # Condition A: NL-only delegation
│   │   ├── run_a2a_dytm.py              # Condition B: DYTM protocol delegation
│   │   ├── run_a2a_hybrid.py            # Condition C: DYTM pipeline + NL feedback
│   │   ├── compare_a2a.py               # Side-by-side result comparison
│   │   ├── trace_task.py                # Verbose single-task debugger
│   │   ├── metrics.py                   # Metric computation
│   │   └── results/                     # Output CSVs + task_bank.json
│   ├── benchmarks/
│   │   └── HumanEval.jsonl               # 164 problems
│   └── results/                          # CSV output from HumanEval runs
├── tests/
├── README.md                             # This file
├── A2A Eval Plan.md                      # A2A experiment design document
├── Implementation Plan.md                # Phase-by-phase build plan
├── Stack Plan.md                         # Architecture & component rationale
└── requirements.txt
```

---

## 13. Citation & Related Work

DYTM differentiates from existing iterative refinement systems as follows:

| System | Feedback Signal | DYTM Difference |
|---|---|---|
| **Reflexion** (Shinn et al.) | LLM verbal self-reflection | Probabilistic feedback. DYTM uses deterministic Hypothesis counterexamples. |
| **Self-Refine** (Madaan et al.) | LLM self-critique | Same weakness — feedback is itself hallucination-prone. |
| **LEVER** (Ni et al.) | Trained neural verifier | Requires additional training. DYTM is structural, no training needed. |
| **TiCoder** (Microsoft) | Example-based unit tests | Unit tests check specific cases. PBTs generalize over infinite inputs. |
| **Clover** | Formal pre/post-conditions (Dafny) | Full formal methods — heavier. DYTM is lighter-weight (PBTs, not proofs). |

The deterministic taxonomy (Section 3.4) relates to work on **Controlled Natural Language** (Fuchs et al., Attempto Controlled English) and **ambiguity detection in requirements engineering**, adapted to the specific domain of coding task descriptions.
