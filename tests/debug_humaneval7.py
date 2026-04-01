"""Reproduce the HumanEval/7 regression to understand what DYTM did wrong."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env.local")

from langchain_openai import ChatOpenAI
from dytm.nodes.operator import make_operator_node
from dytm.nodes.translator import translator_node
from dytm.nodes.doer import doer_node, _compile_test_file
import json

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
operator_fn = make_operator_node(model)

# Load the problem
with open("eval/benchmarks/HumanEval.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d["task_id"] == "HumanEval/7":
            problem = d
            break

prompt = (
    f"Complete this Python function. The function name MUST be "
    f"'{problem['entry_point']}'.\n\n{problem['prompt']}"
)

state = {
    "prompt": prompt,
    "mode": "eval",
    "retry_count": 0,
    "max_retries": 3,
}

# Run just the operator to see what it generates
result = operator_fn(state)
state.update(result)

print("=== GENERATED CODE ===")
print(state["code"])
print()
print("=== PBT PAYLOAD ===")
print(json.dumps(state["pbt_payload"], indent=2))
print()

# See the compiled test
test_src = _compile_test_file(state["code"], state["pbt_payload"])
print("=== COMPILED TEST FILE ===")
print(test_src)
print()

# Run translator
tr = translator_node(state)
print("=== CNL ===")
print(tr["cnl_text"])
print()

# Run doer
doer_result = doer_node(state)
print("=== DOER RESULT ===")
print(f"test_result: {doer_result.get('test_result')}")
print(f"error_trace: {doer_result.get('error_trace', 'none')[:1000]}")

