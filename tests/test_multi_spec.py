"""Tests for the multi-spec module — example extraction, consistency, scoring, contrastive prompting."""
import json
import pytest
from dytm.nodes.multi_spec import (
    extract_docstring_examples,
    check_spec_example_consistency,
    score_spec,
    _values_match,
    _inputs_match,
    _build_contrastive_prompt,
    AmbiguityAnalysis,
    AmbiguityAxis,
)


# ---------------------------------------------------------------------------
# Example extraction tests
# ---------------------------------------------------------------------------

class TestExtractExamples:
    def test_simple_equality(self):
        prompt = '''
def intersperse(numbers, delimeter):
    """ Insert delimeter between elements.
    >>> intersperse([], 4) == []
    >>> intersperse([1, 2, 3], 4) == [1, 4, 2, 4, 3]
    """
'''
        examples = extract_docstring_examples(prompt)
        assert len(examples) == 2
        assert examples[0]["call"] == "intersperse([], 4)"
        assert examples[0]["expected"] == "[]"
        assert examples[0]["func_name"] == "intersperse"
        assert examples[1]["expected"] == "[1, 4, 2, 4, 3]"

    def test_boolean_result(self):
        prompt = '''
def has_close_elements(numbers, threshold):
    """ Check if any two numbers are closer than threshold.
    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False
    """
'''
        examples = extract_docstring_examples(prompt)
        assert len(examples) == 2
        assert examples[0]["expected"] == "True"
        assert examples[1]["expected"] == "False"

    def test_multiline_expected(self):
        prompt = '''
def separate_paren_groups(paren_string):
    """ Separate groups of parentheses.
    >>> separate_paren_groups('(()()) ((())) () ((())()())') == [
    ...     '(()())', '((()))', '()', '((())()())'
    ... ]
    """
'''
        # Multi-line examples are tricky — we handle some cases
        examples = extract_docstring_examples(prompt)
        # At minimum we should extract the call
        assert len(examples) >= 1
        assert examples[0]["func_name"] == "separate_paren_groups"

    def test_no_examples(self):
        prompt = '''
def mystery(n):
    """ Compute the mystery function.
    """
'''
        examples = extract_docstring_examples(prompt)
        assert len(examples) == 0

    def test_implicit_return_display(self):
        prompt = '''
def add(a, b):
    """ Add two numbers.
    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    """
'''
        examples = extract_docstring_examples(prompt)
        assert len(examples) == 2
        assert examples[0]["expected"] == "5"
        assert examples[1]["expected"] == "0"

    def test_real_humaneval_prompt(self):
        """Test with actual HumanEval/5 prompt."""
        prompt = '''from typing import List


def intersperse(numbers: List[int], delimeter: int) -> List[int]:
    """ Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
    >>> intersperse([], 4)
    []
    >>> intersperse([1, 2, 3], 4)
    [1, 4, 2, 4, 3]
    """
'''
        examples = extract_docstring_examples(prompt)
        assert len(examples) == 2
        assert examples[0]["call"] == "intersperse([], 4)"
        assert examples[0]["expected"] == "[]"


# ---------------------------------------------------------------------------
# Value matching tests
# ---------------------------------------------------------------------------

class TestValueMatching:
    def test_identical_strings(self):
        assert _values_match("True", "True") is True

    def test_equivalent_values(self):
        assert _values_match("[1, 2, 3]", "[1,2,3]") is True

    def test_different_values(self):
        assert _values_match("5", "6") is False

    def test_list_comparison(self):
        assert _values_match("[1, 4, 2, 4, 3]", "[1, 4, 2, 4, 3]") is True
        assert _values_match("[1, 4, 2, 4, 3]", "[1, 2, 3]") is False

    def test_unparseable(self):
        assert _values_match("some text", "other text") is None


class TestInputMatching:
    def test_identical(self):
        assert _inputs_match("[1, 2, 3], 4", "[1, 2, 3], 4") is True

    def test_whitespace_difference(self):
        assert _inputs_match("[1, 2, 3], 4", "[1,2,3],4") is True

    def test_different(self):
        assert _inputs_match("[1, 2, 3]", "[4, 5, 6]") is False


# ---------------------------------------------------------------------------
# Consistency checking tests
# ---------------------------------------------------------------------------

class TestConsistency:
    def test_consistent_spec(self):
        pbt = {
            "function_name": "intersperse",
            "properties": [
                {"type": "edge_case", "name": "empty",
                 "input": "[], 4", "expected_behavior": "returns []",
                 "expected": "returns []"},
                {"type": "edge_case", "name": "basic",
                 "input": "[1, 2, 3], 4", "expected_behavior": "returns [1, 4, 2, 4, 3]",
                 "expected": "returns [1, 4, 2, 4, 3]"},
            ],
        }
        examples = [
            {"call": "intersperse([], 4)", "expected": "[]", "func_name": "intersperse"},
            {"call": "intersperse([1, 2, 3], 4)", "expected": "[1, 4, 2, 4, 3]",
             "func_name": "intersperse"},
        ]
        consistent, matches, contradictions = check_spec_example_consistency(pbt, examples)
        assert consistent is True
        assert contradictions == 0
        assert matches >= 1

    def test_inconsistent_spec(self):
        pbt = {
            "function_name": "intersperse",
            "properties": [
                {"type": "edge_case", "name": "wrong",
                 "input": "[1, 2, 3], 4", "expected_behavior": "returns [1, 2, 3, 4]",
                 "expected": "returns [1, 2, 3, 4]"},
            ],
        }
        examples = [
            {"call": "intersperse([1, 2, 3], 4)", "expected": "[1, 4, 2, 4, 3]",
             "func_name": "intersperse"},
        ]
        consistent, matches, contradictions = check_spec_example_consistency(pbt, examples)
        assert consistent is False
        assert contradictions >= 1

    def test_no_examples(self):
        pbt = {
            "function_name": "mystery",
            "properties": [
                {"type": "invariant", "name": "something",
                 "expected": "something holds", "test_body": "pass"},
            ],
        }
        consistent, matches, contradictions = check_spec_example_consistency(pbt, [])
        assert consistent is True


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------

class TestScoring:
    def test_invariant_heavy_scores_higher(self):
        rich = {
            "properties": [
                {"type": "invariant", "test_body": "assert x == y"},
                {"type": "invariant", "test_body": "assert a != b"},
                {"type": "output_type", "test_body": "type test"},
                {"type": "edge_case"},
                {"type": "edge_case"},
            ],
        }
        poor = {
            "properties": [
                {"type": "output_type", "test_body": "type test"},
                {"type": "edge_case"},
                {"type": "edge_case"},
            ],
        }
        assert score_spec(rich) > score_spec(poor)

    def test_empty_properties(self):
        assert score_spec({"properties": []}) == 0.0

    def test_no_test_body_invariant(self):
        """Invariant without test_body should score less."""
        with_body = {"properties": [{"type": "invariant", "test_body": "test"}]}
        without_body = {"properties": [{"type": "invariant"}]}
        assert score_spec(with_body) > score_spec(without_body)

    def test_specificity_bonus_for_comparisons(self):
        """Invariants with == or <= in test_body get a specificity bonus."""
        specific = {"properties": [
            {"type": "invariant", "test_body": "assert result == sorted(input_list)"},
        ]}
        vague = {"properties": [
            {"type": "invariant", "test_body": "assert isinstance(result, list)"},
        ]}
        assert score_spec(specific) > score_spec(vague)

    def test_assume_bonus(self):
        """Invariants using assume() get a sophistication bonus."""
        with_assume = {"properties": [
            {"type": "invariant", "test_body": "assume(len(x) > 0)\nassert result == x[0]"},
        ]}
        without_assume = {"properties": [
            {"type": "invariant", "test_body": "assert result == x[0]"},
        ]}
        assert score_spec(with_assume) > score_spec(without_assume)


# ---------------------------------------------------------------------------
# Contrastive prompting tests
# ---------------------------------------------------------------------------

class TestContrastivePrompting:
    def _make_analysis(self):
        return AmbiguityAnalysis(
            ambiguities=[
                AmbiguityAxis(
                    aspect="bracket matching algorithm",
                    interpretations=[
                        "Count-based: count('<') == count('>')",
                        "Prefix-validity: no prefix has more closing than opening",
                    ],
                    most_likely=1,
                ),
                AmbiguityAxis(
                    aspect="empty string handling",
                    interpretations=[
                        "Empty string returns True (vacuously correct)",
                        "Empty string returns False (no brackets to match)",
                    ],
                    most_likely=0,
                ),
            ],
            is_ambiguous=True,
            difficulty="moderate",
        )

    def test_variant_0_uses_most_likely(self):
        analysis = self._make_analysis()
        prompt = _build_contrastive_prompt("some task", analysis, 0)
        # variant 0 should use most_likely interpretation
        assert "Prefix-validity" in prompt
        assert "vacuously correct" in prompt

    def test_variant_1_uses_alternative(self):
        analysis = self._make_analysis()
        prompt = _build_contrastive_prompt("some task", analysis, 1)
        # variant 1 should use the NON-default interpretation
        assert "Count-based" in prompt
        assert "returns False" in prompt

    def test_no_ambiguities_returns_plain_prompt(self):
        analysis = AmbiguityAnalysis(
            ambiguities=[],
            is_ambiguous=False,
            difficulty="simple",
        )
        prompt = _build_contrastive_prompt("do something", analysis, 0)
        assert prompt == "Task: do something"
        prompt1 = _build_contrastive_prompt("do something", analysis, 1)
        assert prompt1 == "Task: do something"

    def test_variants_differ(self):
        """Different variant indices should produce different prompts."""
        analysis = self._make_analysis()
        p0 = _build_contrastive_prompt("task", analysis, 0)
        p1 = _build_contrastive_prompt("task", analysis, 1)
        assert p0 != p1

    def test_guidance_section_present(self):
        analysis = self._make_analysis()
        prompt = _build_contrastive_prompt("task", analysis, 0)
        assert "INTERPRETATION GUIDANCE" in prompt


# ---------------------------------------------------------------------------
# Ambiguity model tests (Pydantic validation)
# ---------------------------------------------------------------------------

class TestAmbiguityModels:
    def test_ambiguity_axis_min_interpretations(self):
        """Must have at least 2 interpretations."""
        with pytest.raises(Exception):
            AmbiguityAxis(
                aspect="test",
                interpretations=["only one"],
                most_likely=0,
            )

    def test_valid_analysis(self):
        a = AmbiguityAnalysis(
            ambiguities=[],
            is_ambiguous=False,
            difficulty="simple",
        )
        assert not a.is_ambiguous
        assert a.difficulty == "simple"

    def test_analysis_with_ambiguities(self):
        a = AmbiguityAnalysis(
            ambiguities=[
                AmbiguityAxis(
                    aspect="something",
                    interpretations=["a", "b"],
                    most_likely=0,
                ),
            ],
            is_ambiguous=True,
            difficulty="moderate",
        )
        assert len(a.ambiguities) == 1
        assert a.ambiguities[0].aspect == "something"

