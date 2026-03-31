# DYTM ("Do You Trust Me?")
**A Deterministic Verification Layer for Probabilistic LLM Agents**

## 1. The Core Premise
DYTM acts as a mathematical firewall for Large Language Models (LLMs) and Multi-Agent Systems (MAS). The architecture assumes the following:
1. LLMs are probabilistic and inherently untrustworthy for direct execution due to hallucination risks.
2. Because of (1), an LLM cannot be relied upon to accurately translate its own execution logic into a natural language summary (it might hallucinate a safe description for dangerous code).
3. Trust requires deterministic verification: the user must mathematically approve the system invariants *before* execution begins.

## 2. The Agent Roles
The system models interaction through three distinct entities:
* **The Executive:** The entity with the initial intent. This can be a human user or a higher-level LLM agent in a MAS. The Executive's intent is often ambiguous or constrained by natural language.
* **The Operator (Planner):** An LLM agent that receives the Executive's prompt. It is responsible for translating the ambiguous intent into both functional code and strict, structured testing constraints.
* **The Doer (Executor):** An isolated, sandboxed runtime environment that executes the Operator's code and loops until all constraints are met.

## 3. The Verification Loop (The "Firewall")
To bridge the gap between ambiguous intent and safe execution, DYTM enforces the following pipeline:
1. **Intent Translation:** The Executive sends a natural language prompt to the Operator.
2. **Structured Generation:** The Operator generates the execution code AND a suite of Property-Based Tests (PBTs). *Crucially, these PBTs are generated as structured payloads (e.g., rigid JSON), not raw test code.*
3. **Deterministic Parsing (Logic -> CNL):** A deterministic, native engine reads the structured PBT payload and maps it 1:1 to a Controlled Natural Language (CNL) string using pre-defined text templates. No LLM is used in this translation step, eliminating hallucination.
4. **The Contract:** The Executive reads the deterministic CNL. By approving the CNL, the Executive is mathematically approving the underlying PBTs. If rejected, the Operator must try again.
5. **Sandboxed Execution:** Upon approval, the Doer agent executes the code in an isolated environment. It runs the PBTs against the output.
6. **Iterative Refinement:** If the PBTs fail, the Doer captures the error and forces the Operator to rewrite the code. This loops invisibly until the code mathematically satisfies the approved properties.
7. **Delivery:** The Executive receives the final artifact, which mathematically matches their original intent.

## 4. Key Use Cases
* **Consumer Trust:** Acts as a safety layer for enterprise chatbots, ensuring users (or faculty/researchers) can verify an AI's intended actions on their local files or data before any mutations occur.
* **MAS Stabilization:** In complex Multi-Agent Systems, replacing standard natural language communication with DYTM's CNL verification drastically reduces "intent drift" and hallucinated agreements between agents.
* **Information Verification:** Expanding the PBT concept to verify if an LLM's output was strictly derived from an approved Knowledge Graph or RAG database, rather than just testing functional code.

## 5. Implementation Strategy & Architecture
To balance academic validation with maximum enterprise value, the development is sequenced into two distinct phases:

### Phase 1: The Academic MVP (Web/Python)
**Goal:** Prove the multi-agent architecture improves task adherence and reduces errors for academic publication.
* **Orchestration:** Python (LangGraph) for stateful agent loops.
* **Testing Engine:** `Hypothesis` for dynamic property-based test generation and counterexample shrinking.
* **Evaluation:** Run an A/B test using a "dumb" local model against standard benchmarks (e.g., HumanEval, GAIA) to measure the delta in accuracy when DYTM is applied versus a baseline run.

### Phase 2: The Enterprise Moat (iOS/Native Swift)
**Goal:** Build a highly defensible, on-device product that solves the exact problem Big Tech faces with edge-device AI privacy and execution.
* **Local Inference:** Apple `FoundationModels` or `MLX Swift` for private, on-device LLM generation.
* **The Parser:** Native Swift code utilizing `Codable` to parse the Operator's JSON payload and map it to localized English UI strings.
* **The Sandbox:** Apple's `JavaScriptCore`. To bypass iOS kernel restrictions (Write XOR Execute memory rules), the Operator generates JavaScript instead of Swift. `JavaScriptCore` evaluates the logic instantly and securely without triggering dynamic compilation blockers.
* **The Engine:** A pre-compiled Swift binary manages the test execution natively, maintaining the deterministic firewall on the user's iPhone.
