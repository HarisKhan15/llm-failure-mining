import io
import json

import pytest

from llm_failure_mining import cli, collect_ollama
from llm_failure_mining.collect_ollama import collect, distinct_prompts, parse_answer


def test_end_to_end_on_sample(tmp_path):
    """Full pipeline on the committed sample: every output file is produced."""
    report = cli.run(cli.PROJECT_ROOT / "config.toml", use_sample=True, output_dir=tmp_path)
    expected = [
        "REPORT.md", "summary.json", "clusters.csv", "rules.csv", "prediction_summary.csv",
        "prediction_folds.csv", "feature_importance.csv", "error_types_by_model.csv",
        "figures/error_types.png", "figures/accuracy_by_style.png", "figures/errors_by_family.png",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert 0 < summary["overall_accuracy"] < 1
    assert abs(sum(summary["error_share_among_failures"].values()) - 1) < 0.01
    assert "Key finding 1" in report.read_text()


def test_cli_reports_missing_data(tmp_path, caplog):
    code = cli.main(["run", "--data", str(tmp_path / "missing.json"), "--output", str(tmp_path)])
    assert code == 1


@pytest.mark.parametrize(
    "text,value",
    [("Reasoning...\n### Answer: 5", 5.0), ("### Answer: **3**", 3.0),
     ("### Answer: 2 ... wait ### Answer: 4", 4.0), ("I cannot tell.", None)],
)
def test_parse_answer(text, value):
    assert parse_answer(text) == value


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_collect_with_mocked_ollama(monkeypatch):
    """Mocked HTTP: checks request handling and output schema, not a real model."""
    replies = iter(["Thinking. ### Answer: 5", "### Answer: 4"])

    def fake_urlopen(req, timeout):
        body = json.loads(req.data)
        assert body["stream"] is False and body["model"] == "tiny"
        return FakeResponse(json.dumps({"response": next(replies)}).encode())

    monkeypatch.setattr(collect_ollama.urllib.request, "urlopen", fake_urlopen)
    rows = collect([("Alice has 4 sisters and 1 brother. How many sisters does her brother have?", 5.0)],
                   model="tiny", trials=2)
    assert [r["correct"] for r in rows] == [True, False]
    assert rows[0]["model"] == "ollama:tiny"
    assert set(rows[0]) >= {"model", "prompt", "model_response", "parsed_answer", "right_answer", "correct"}


def test_distinct_prompts_deduplicates():
    recs = [{"prompt": "a", "right_answer": "1"}, {"prompt": "a", "right_answer": "1"}, {"prompt": "b", "right_answer": 2}]
    assert distinct_prompts(recs) == [("a", 1.0), ("b", 2.0)]
