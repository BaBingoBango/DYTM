"""DYTM entry point — ``python -m dytm``.

Runs a single hardcoded prompt through the full DYTM verification loop
as a quick integration test.  For the interactive demo, use
``python -m dytm.free`` (Phase 5).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def main() -> None:
    # Load API key from .env.local (or .env)
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env.local")
    load_dotenv(project_root / ".env")  # fallback

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found. Set it in .env.local or your environment.")
        sys.exit(1)

    # Import here so env is loaded before langchain reads the key
    from langchain_openai import ChatOpenAI
    from dytm.graph import build_graph

    print("=" * 60)
    print("  DYTM — Do You Trust Me (Phase 3 Smoke Test)")
    print("=" * 60)

    # Build the graph with GPT-4o-mini
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    app = build_graph(model)

    # Test prompt
    prompt = "Write a Python function called 'is_palindrome' that takes a string and returns True if it is a palindrome (case-insensitive, ignoring spaces), False otherwise."

    print(f"\nPrompt: {prompt}")
    print("-" * 60)

    # Invoke the full DYTM loop
    initial_state = {
        "prompt": prompt,
        "mode": "eval",
        "retry_count": 0,
        "max_retries": 3,
    }

    final_state = app.invoke(initial_state)

    # Display results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)

    print(f"\nTest result: {final_state.get('test_result', 'unknown')}")
    print(f"Retries: {final_state.get('retry_count', 0)}")
    print(f"Security passed: {final_state.get('security_passed', 'unknown')}")

    print(f"\n--- Generated Code ---")
    print(final_state.get("code", "(no code)"))

    print(f"\n--- CNL Contract ---")
    print(final_state.get("cnl_text", "(no contract)"))

    if final_state.get("error_trace") and final_state.get("test_result") != "pass":
        print(f"\n--- Error Trace ---")
        print(final_state.get("error_trace", ""))

    if final_state.get("test_result") == "pass":
        print("\n🎉 DYTM loop completed successfully! Code verified against PBTs.")
    else:
        print("\n❌ DYTM loop exhausted retries. Code did not pass all PBTs.")

    print()


if __name__ == "__main__":
    main()

