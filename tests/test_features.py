import math

import pytest

from llm_failure_mining.features import (
    error_type, instruction_style, model_family, model_size_b, problem_text,
    size_bucket, stated_numbers,
)

SIBLING = (
    "Alice has 4 sisters and she also has 1 brother. How many sisters does Alice's brother have? "
    "Solve this problem and provide the final answer in following form: \"### Answer: \"."
)


def test_problem_text_stops_at_question_mark():
    assert problem_text(SIBLING).endswith("Alice's brother have?")


def test_stated_numbers_ignore_instruction_text():
    # Only 4 and 1 are in the problem; the instruction contains no numbers anyway.
    assert sorted(stated_numbers(SIBLING)) == [1, 4]


def test_stated_numbers_reads_words_and_articles():
    prompt = "Alice has four brothers and she also has a sister. How many sisters does Alice's brother have?"
    assert sorted(stated_numbers(prompt)) == [1, 4]


@pytest.mark.parametrize(
    "prompt,style",
    [
        (SIBLING, "standard"),
        ("... To answer the question, DO NOT OUTPUT ANY TEXT EXCEPT following format: ### Answer:", "restricted_format"),
        ("... think carefully and double check the path ... DO NOT OUTPUT ANY TEXT", "think_step_by_step"),
        ("### Problem: Alice has 4 sisters. How many? ### Answer: ", "base_completion"),
        ("Alice has 4 sisters and she also has 1 brother. How many sisters does Alice's brother have?", "bare_question"),
    ],
)
def test_instruction_style(prompt, style):
    assert instruction_style(prompt) == style


@pytest.mark.parametrize(
    "parsed,expected",
    [
        (5.0, "correct"),          # labelled correct
        (None, "no_parsable_answer"),
        (4.0, "off_by_one_low"),   # forgot to count Alice
        (6.0, "off_by_one_high"),
        (1.0, "copies_given_number"),
        (9.0, "other_wrong"),
    ],
)
def test_error_type_for_sibling_problem(parsed, expected):
    numbers = [4, 1]
    assert error_type(parsed, 5.0, parsed == 5.0, numbers) == expected


def test_error_type_detects_sums_and_wrong_reasoning():
    assert error_type(7.0, 10.0, False, [3, 4]) == "adds_given_numbers"
    # Right number but the AIW authors marked the reasoning as wrong.
    assert error_type(5.0, 5.0, False, [4, 1]) == "right_number_wrong_reasoning"
    assert error_type(float("nan"), 5.0, False, [4, 1]) == "no_parsable_answer"


@pytest.mark.parametrize(
    "name,size",
    [("llama-3-70b-chat", 70.0), ("mixtral-8x22b", 176.0), ("qwen1.5-1.8b-chat", 1.8), ("phi-2", 2.7)],
)
def test_model_size(name, size):
    assert model_size_b(name) == pytest.approx(size)


def test_closed_model_size_unknown():
    assert math.isnan(model_size_b("gpt-4o-2024-05-13"))
    assert size_bucket(float("nan")) == "unknown_closed"


@pytest.mark.parametrize(
    "name,family",
    [("codellama-34b-instruct", "codellama"), ("open-mixtral-8x7b", "mistral"),
     ("command-r-plus", "cohere"), ("nous-hermes-llama2-13b", "llama_finetune"), ("gpt-3.5-turbo", "gpt")],
)
def test_model_family(name, family):
    assert model_family(name) == family
