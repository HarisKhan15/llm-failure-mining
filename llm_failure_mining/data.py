"""Load the AIW response records into a clean, validated pandas DataFrame.

Each record is one model answer to one prompt (one "trial"). The AIW authors
ran the same prompt many times per model, so identical short answers such as
"### Answer: 4" legitimately appear many times. We therefore do NOT remove
duplicates; instead the prediction stage splits data by prompt so repeated
trials of one prompt never land in both training and test sets.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REQUIRED_FIELDS = ("model", "prompt", "model_response", "parsed_answer", "right_answer", "correct")


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run 'python scripts/fetch_aiw_data.py' "
            "or use the sample with '--sample'."
        )
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list of records in {path}")
    return records


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Validate records and return a DataFrame with consistent column types.

    Records missing a required field are dropped and counted in
    ``df.attrs["dropped_records"]`` so the report can show data quality.
    """
    clean_rows = []
    dropped = 0
    for rec in records:
        if not all(field in rec for field in REQUIRED_FIELDS):
            dropped += 1
            continue
        right = _to_float(rec["right_answer"])
        if right is None or not isinstance(rec["prompt"], str):
            dropped += 1
            continue
        clean_rows.append(
            {
                "model": str(rec["model"]),
                "prompt": rec["prompt"],
                "response": rec["model_response"] if isinstance(rec["model_response"], str) else "",
                "parsed_answer": _to_float(rec["parsed_answer"]),
                "right_answer": right,
                # The AIW authors' label. It already includes their manual checks,
                # e.g. a right number reached through wrong reasoning is marked False.
                "correct": _to_bool(rec["correct"]),
                "manual_inspection": _to_bool(rec.get("manual_inspection", False)),
            }
        )

    df = pd.DataFrame(clean_rows)
    if df.empty:
        raise ValueError("No valid records after validation")
    # Stable integer id for each distinct prompt text, used for grouped splits.
    df["prompt_key"] = pd.factorize(df["prompt"])[0]
    df.attrs["dropped_records"] = dropped
    return df


def load_dataset(path: str | Path) -> pd.DataFrame:
    return to_dataframe(load_records(path))
