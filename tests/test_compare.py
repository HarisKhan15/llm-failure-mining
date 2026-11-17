import json

from llm_failure_mining import compare


def _record(prompt, right, parsed, model="ollama:tiny"):
    return {
        "model": model, "prompt": prompt, "model_response": f"### Answer: {parsed}",
        "parsed_answer": parsed, "right_answer": right, "correct": parsed == right,
    }


SIB = "Alice has 4 sisters and she also has 1 brother. How many sisters does Alice's brother have?"


def test_compare_writes_report_and_profiles_sum_to_one(tmp_path, sample_path):
    new = [_record(SIB, 5, 4)] * 6 + [_record(SIB, 5, 5)] * 2 + [_record(SIB, 5, None)] * 2
    new_path = tmp_path / "new.json"
    new_path.write_text(json.dumps(new))

    code = compare.main(["--new", str(new_path), "--reference", str(sample_path), "--output", str(tmp_path / "out")])
    assert code == 0
    for name in ("COMPARISON.md", "summary.csv", "error_profiles.csv", "comparison.png"):
        assert (tmp_path / "out" / name).exists()

    import pandas as pd
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    new_row = summary[summary["group"] == "new_data"].iloc[0]
    assert new_row["accuracy"] == 0.2
    # 6 of the 8 failures are off-by-one
    assert new_row["off_by_one_share_of_sibling_failures"] == 0.75
    profiles = pd.read_csv(tmp_path / "out" / "error_profiles.csv")
    assert abs(profiles["new_data"].sum() - 1) < 0.01


def test_compare_reports_missing_file(tmp_path, sample_path):
    assert compare.main(["--new", str(tmp_path / "missing.json"), "--reference", str(sample_path)]) == 1
