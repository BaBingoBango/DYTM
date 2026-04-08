"""Planner module — task generation, executive CNL review, and output judgment.

The Planner is an Anthropic model (claude-3.5-sonnet) that:
1. Generates complex coding tasks with hidden intent (task generation)
2. Acts as adversarial Executive reviewing CNL contracts (DYTM condition)
3. Provides NL feedback to the Coder (baseline condition)
4. Judges whether the Coder's final output matches its intent (both conditions)
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from eval.a2a.schemas import (
    A2ABaselineFeedback,
    A2AExecutiveDecision,
    A2AJudgment,
    A2ATask,
    A2ATaskBank,
)

# ---------------------------------------------------------------------------
# 1. Task Generation
# ---------------------------------------------------------------------------

TASK_GENERATION_SYSTEM_PROMPT = """\
You are a senior software architect generating complex coding tasks for a \
code generation benchmark. Each task will be given to an LLM Coder as a \
natural language description. The Coder must implement a Python function.

YOUR GOAL: Generate tasks where the description is realistic but contains \
IMPLICIT CONSTRAINTS and AMBIGUITIES that a Coder could plausibly \
misinterpret. You are NOT trying to be tricky for the sake of it — these \
should be tasks a real developer might delegate to an AI assistant.

For each task, you produce:
- `task_id`: sequential ID like "A2A/001"
- `function_name`: the exact function name
- `description`: what the Coder sees (realistic NL, may contain ambiguity)
- `intent_summary`: your PRIVATE notes on what you actually want, including \
  implicit constraints and edge-case expectations that are NOT fully explicit \
  in the description
- `category`: one of "data_transformation", "business_logic", \
  "string_text_processing", "algorithmic", "api_interface_design"
- `complexity_axes`: which axes apply
- `ambiguity_level`: "low", "medium", or "high"

IMPORTANT GUIDELINES:
- Each task should be a SINGLE Python function (not a class or module).
- The description should be 3-8 sentences — realistic, not artificially obscure.
- The intent_summary should spell out EVERY implicit constraint you care about.
- Include at least 2-3 edge cases in your intent_summary that are NOT in the \
  description.
- Aim for: 10 low ambiguity, 20 medium ambiguity, 20 high ambiguity tasks.
- Cover all 5 categories roughly evenly (10 each).
- Tasks should be solvable in 10-50 lines of Python.
"""


def generate_task_bank(
    planner_model: BaseChatModel,
    n_tasks: int = 50,
    batch_size: int = 10,
) -> A2ATaskBank:
    """Generate the full task bank using the Planner model.

    Generates in batches to stay within context limits and improve quality.

    Args:
        planner_model: The Planner LLM (must be Anthropic for decorrelation).
        n_tasks: Total number of tasks to generate.
        batch_size: Tasks per generation call.

    Returns:
        An A2ATaskBank containing all generated tasks.
    """
    structured_model = planner_model.with_structured_output(A2ATaskBank)
    all_tasks: list[A2ATask] = []
    categories = [
        "data_transformation",
        "business_logic",
        "string_text_processing",
        "algorithmic",
        "api_interface_design",
    ]

    for batch_idx in range(0, n_tasks, batch_size):
        remaining = min(batch_size, n_tasks - len(all_tasks))
        # Rotate through categories
        target_category = categories[batch_idx // batch_size % len(categories)]
        start_id = len(all_tasks) + 1

        user_content = (
            f"Generate {remaining} tasks for category '{target_category}'.\n"
            f"Start task_id numbering at A2A/{start_id:03d}.\n"
            f"Mix ambiguity levels: aim for 2 low, 4 medium, 4 high per batch.\n"
            f"Remember: the description should be realistic but contain implicit "
            f"constraints that could cause a Coder to misinterpret."
        )

        messages = [
            SystemMessage(content=TASK_GENERATION_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        try:
            result: A2ATaskBank = structured_model.invoke(messages)
            all_tasks.extend(result.tasks)
        except Exception as exc:
            print(f"  [WARN] Task generation batch {batch_idx} failed: {exc}")
            continue

    return A2ATaskBank(tasks=all_tasks)


# ---------------------------------------------------------------------------
# 2. Adversarial Executive Review (DYTM condition)
# ---------------------------------------------------------------------------

ADVERSARIAL_EXECUTIVE_SYSTEM_PROMPT = """\
You are the Planner who wrote the original task. A Spec agent has produced \
a verification contract based on your task description. You must decide \
whether to APPROVE or REJECT it.

REVIEW INSTRUCTIONS:
1. Does the contract capture the CORE BEHAVIOR you intended? If the Coder \
   follows this contract, will they build fundamentally the RIGHT function?
2. If you previously rejected a contract, check: did the Spec address your \
   feedback? If your main concerns are resolved, APPROVE even if minor \
   gaps remain.
3. Only REJECT if the contract would lead the Coder to build a \
   FUNDAMENTALLY WRONG function — not just an incomplete one.

IMPORTANT: A contract that captures 70-80% of your intent is GOOD ENOUGH. \
The PBT testing loop will catch implementation bugs. Your job is to prevent \
the Coder from building the WRONG THING, not to ensure the contract is \
perfect.

If this is a revision and the Spec has made a good-faith effort to address \
your previous feedback, lean toward APPROVE.
"""


def adversarial_executive_review(
    planner_model: BaseChatModel,
    task: A2ATask,
    cnl_text: str,
    previous_feedback: str | None = None,
    attempt: int = 0,
) -> A2AExecutiveDecision:
    """Have the Planner adversarially review a CNL contract.

    The Planner sees its original task description AND its private intent
    summary (which the Coder/Spec never see), and judges whether the
    contract is faithful.

    Args:
        planner_model: The Planner LLM (Anthropic).
        task: The original task (including private intent_summary).
        cnl_text: The CNL contract rendered by the Translator.
        previous_feedback: If this is a revision, the Executive's previous
            rejection feedback (so it can check convergence).
        attempt: Which review attempt this is (0-indexed).

    Returns:
        An A2AExecutiveDecision with approved/feedback.
    """
    structured_model = planner_model.with_structured_output(A2AExecutiveDecision)

    revision_context = ""
    if previous_feedback and attempt > 0:
        revision_context = (
            f"\n--- REVISION CONTEXT (attempt {attempt + 1}) ---\n"
            f"You previously REJECTED the contract with this feedback:\n"
            f"{previous_feedback}\n\n"
            f"The Spec has revised the contract. Check whether your main "
            f"concerns were addressed. If the Spec made a good-faith effort "
            f"to fix your issues, lean toward APPROVE.\n"
            f"---\n\n"
        )

    user_content = (
        f"YOUR ORIGINAL TASK DESCRIPTION:\n{task.description}\n\n"
        f"YOUR PRIVATE INTENT (what you actually want — the Coder will "
        f"never see this):\n{task.intent_summary}\n\n"
        f"{revision_context}"
        f"THE PROPOSED CONTRACT:\n{cnl_text}\n\n"
        f"Does this contract capture enough of your intent for a Coder to "
        f"build the right function? APPROVE or REJECT."
    )

    messages = [
        SystemMessage(content=ADVERSARIAL_EXECUTIVE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        result: A2AExecutiveDecision = structured_model.invoke(messages)
        return result
    except Exception as exc:
        # On parse failure, default to approve (don't penalize for infra errors)
        return A2AExecutiveDecision(
            approved=True,
            feedback=f"Executive review failed (auto-approved): {str(exc)[:200]}",
        )


# ---------------------------------------------------------------------------
# 3. Baseline NL Feedback (baseline condition)
# ---------------------------------------------------------------------------

BASELINE_FEEDBACK_SYSTEM_PROMPT = """\
You are the Planner who wrote the original task. A Coder has produced code \
based on your natural language description. Review the code and determine \
whether it matches YOUR intent.

If the code does NOT match your intent, provide specific natural language \
feedback explaining what's wrong and how to fix it. Be precise — mention \
specific edge cases, boundary conditions, or behaviors that are incorrect.

If the code matches your intent, mark it as satisfied.
"""


def baseline_feedback(
    planner_model: BaseChatModel,
    task: A2ATask,
    code: str,
) -> A2ABaselineFeedback:
    """Have the Planner provide NL feedback on the Coder's output (baseline).

    This is the baseline equivalent of the DYTM contract review — the Planner
    can only give unstructured natural language feedback.

    Args:
        planner_model: The Planner LLM (Anthropic).
        task: The original task (including private intent_summary).
        code: The Coder's generated code.

    Returns:
        An A2ABaselineFeedback with satisfied/feedback.
    """
    structured_model = planner_model.with_structured_output(A2ABaselineFeedback)

    user_content = (
        f"YOUR ORIGINAL TASK:\n{task.description}\n\n"
        f"YOUR PRIVATE INTENT (what you actually meant):\n"
        f"{task.intent_summary}\n\n"
        f"THE CODER'S OUTPUT:\n```python\n{code}\n```\n\n"
        f"Does this code match your intent?"
    )

    messages = [
        SystemMessage(content=BASELINE_FEEDBACK_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        result: A2ABaselineFeedback = structured_model.invoke(messages)
        return result
    except Exception:
        # On failure, assume not satisfied (conservative)
        return A2ABaselineFeedback(
            satisfied=False,
            feedback="Feedback generation failed — please retry.",
        )


# ---------------------------------------------------------------------------
# 4. Final Judgment (both conditions)
# ---------------------------------------------------------------------------

JUDGMENT_SYSTEM_PROMPT = """\
You are the Planner who originally created this task. You will now judge \
whether the Coder's output matches YOUR intent.

Judge the output on these dimensions:

1. INTENT MATCH (1-5): Does the code do what you meant? Not "is it correct \
   code" but "is it the right thing?"
   - 5 = Perfect match to my intent
   - 4 = Mostly right, minor misunderstanding
   - 3 = Partially right, significant misunderstanding on one axis
   - 2 = Fundamentally misunderstood on multiple axes
   - 1 = Completely wrong thing

2. DRIFT DETECTED (yes/no): Is there any gap between what you meant and \
   what the Coder built?

3. DRIFT DESCRIPTION (if yes): What specifically was misunderstood?

4. DRIFT CATEGORY: One of [boundary_condition, implicit_constraint, \
   domain_term, edge_case, return_type, error_handling, ordering, \
   performance, none, other]

Be STRICT. Compare the code against your private intent summary, not just \
the public description. A function that handles the "obvious" interpretation \
but misses your implicit constraints should score 3 or lower.
"""


def judge_output(
    planner_model: BaseChatModel,
    task: A2ATask,
    code: str,
) -> A2AJudgment:
    """Have the Planner judge the Coder's final output.

    Uses the Planner's private intent_summary as anchor for judgment.

    Args:
        planner_model: The Planner LLM (Anthropic).
        task: The original task (including private intent_summary).
        code: The Coder's final generated code.

    Returns:
        An A2AJudgment with intent_match score, drift info.
    """
    structured_model = planner_model.with_structured_output(A2AJudgment)

    user_content = (
        f"YOUR ORIGINAL TASK:\n{task.description}\n\n"
        f"YOUR PRIVATE INTENT SUMMARY:\n{task.intent_summary}\n\n"
        f"THE CODER'S OUTPUT:\n```python\n{code}\n```\n\n"
        f"Judge the output now."
    )

    messages = [
        SystemMessage(content=JUDGMENT_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        result: A2AJudgment = structured_model.invoke(messages)
        return result
    except Exception:
        # Worst-case fallback
        return A2AJudgment(
            intent_match=1,
            drift_detected=True,
            drift_description="Judgment call failed",
            drift_category="other",
        )


def judge_output_majority(
    planner_model: BaseChatModel,
    task: A2ATask,
    code: str,
    n_votes: int = 3,
) -> tuple[A2AJudgment, list[int]]:
    """Run judgment N times and return majority-voted result.

    Args:
        planner_model: The Planner LLM (Anthropic, temperature=0).
        task: The original task.
        code: The Coder's final code.
        n_votes: Number of judgment repetitions.

    Returns:
        Tuple of (best judgment, list of intent_match votes).
    """
    judgments: list[A2AJudgment] = []
    for _ in range(n_votes):
        j = judge_output(planner_model, task, code)
        judgments.append(j)

    votes = [j.intent_match for j in judgments]

    # Pick the judgment whose intent_match is closest to median
    sorted_votes = sorted(votes)
    median_vote = sorted_votes[len(sorted_votes) // 2]

    best = min(judgments, key=lambda j: abs(j.intent_match - median_vote))
    return best, votes

