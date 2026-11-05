import json

import pytest

from llm_failure_mining.data import load_dataset, to_dataframe

GOOD = {
    "model": "m1", "prompt": "Alice has 1 sister. How many?", "model_response": "### Answer: 2",
    "parsed_answer": "2.0", "right_answer": "2", "correct": "True",
}


def test_types_are_normalised():
    df = to_dataframe([GOOD])
    row = df.iloc[0]
    assert row["parsed_answer"] == 2.0 and row["right_answer"] == 2.0
    assert bool(row["correct"]) is True


def test_invalid_records_are_dropped_and_counted():
    missing_field = {k: v for k, v in GOOD.items() if k != "correct"}
    bad_answer = {**GOOD, "right_answer": "unknown"}
    df = to_dataframe([GOOD, missing_field, bad_answer])
    assert len(df) == 1
    assert df.attrs["dropped_records"] == 2


def test_unparsable_answer_becomes_nan():
    df = to_dataframe([{**GOOD, "parsed_answer": None, "correct": False}])
    assert df["parsed_answer"].isna().all()


def test_empty_input_raises():
    with pytest.raises(ValueError):
        to_dataframe([])


def test_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_aiw_data"):
        load_dataset(tmp_path / "nope.json")


def test_prompt_key_groups_identical_prompts(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps([GOOD, {**GOOD, "model": "m2"}, {**GOOD, "prompt": "Other?"}]))
    df = load_dataset(path)
    assert df["prompt_key"].nunique() == 2


def test_sample_loads_without_drops(sample_path):
    df = load_dataset(sample_path)
    assert len(df) > 300
    assert df.attrs["dropped_records"] == 0
    assert df["correct"].any() and (~df["correct"]).any()
