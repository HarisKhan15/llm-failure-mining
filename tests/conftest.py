import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLE = ROOT / "data" / "sample" / "aiw_sample.json"


@pytest.fixture(scope="session")
def sample_path() -> Path:
    assert SAMPLE.exists(), "Committed sample missing: data/sample/aiw_sample.json"
    return SAMPLE


@pytest.fixture(scope="session")
def featured_sample(sample_path):
    from llm_failure_mining.data import load_dataset
    from llm_failure_mining.features import add_features

    return add_features(load_dataset(sample_path))
