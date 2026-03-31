# DYTM

DYTM ("Do You Trust Me") is a verirication layer for large language models using formal methods. Instead of relying on direct English to logic transformation, the following is assumed:

1. LLMs are untrustworthy by definition due to their probablistic and largely unexplainable generations.
2. As a result of (1), LLMs cannot be replied upon to convert English to logic.

The situation could be modeled as:

- The **executive**, either a human being or an LLM agent. Either has an _intent_ (dubious in the case of the LLM but the analogy holds), which actually may or may not be a concrete logical thought (i.e. someone not even knowing what exactly they want but prompting anyway). We accept that this intent cannot be translated into logic due to natural language vagueness and unsure operators.
- The **operator**, an LLM agent. Technically, could also be thought of as a human being in a theoretical sense. This is the agent responsible for completing the executive's task.

We can recognize that the executive cannot transfer intent directly due to language constraints. To combat this, the DYTM requires a verification before task work begins:

1. The executive coverts their intent into a prompt, which is sent to the operator.
2. The operator proposes a suite of property-based tests (PBTs) which describe what the operator thinks the executive meant. For instance, these could be code tests for a programming task, Excel functions for a spreadsheet task, or whatever verification layer is available.
3. The operator's PBTs map 1:1 to logic, and are therefore converted to an implementation of contstrained natural logic (CNL).
4. The executive approves or returns the CNL, which is now a mathematically sound description of a verifier suite. If returned, the operator returns to step (2). If approved, the operator moves to step (5).
5. The operator completes the task, rather, it iterates on the task until its artifacts pass all of the PBTs. If the task fails over and over, this result can be returned (an LLM effort failure).
6. The executive gets the results, which mathematically match their original intent.

## Heuristics

- A "smoother" LLM agent can be employed to improve readability of the CNL, as CNL is often hard to read and dense. This would be advisory/optional only, as the LLM could easily get this wrong/over-describe the CNL.

## Creating Tests

Another key element is the creation of tests for the environment at hand. For instance, code tests might be easier, while Excel tests or UI-based development tests might be harder.

It would also be interesting if "informational" tests could be set up without employing another LLM as judge, i.e. was X advice derived from Y pool of approved facts? Maybe this would require some sort of knowldege graph/data structure or database.

## Key Uses

- This could hopefully improve user trust in regular LLM "chatbots" by requiring users to verify intent in a way that will certainly come to pass by way of the tests.
- This could also possibly increase the power of MAS systems. Whereas traditional MASs converse in natural language with one another, requiring the DYTM CNL verification could reduce intent errors and smooth the flow of ideas between many agents. Such a MAS would probably be based on "duos" of executive & operator agents instead of singleton agents.

## Implementation

- An on-device implementation (e.g. Apple's Foundation Models framework) would be a robust way to demonstrate if the system is effective.
- A simpler code-only Python/web based implementation might serve as a stronger V1 for evaluation (i.e. does DYTM even help?).
