"""Load and validate the TOML configuration file."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_path: Path
    sample_path: Path
    seed: int
    output_dir: Path
    cluster_min_words: int
    cluster_max_responses: int
    k_min: int
    k_max: int
    svd_components: int
    rules_min_support: float
    rules_min_confidence: float
    rules_top_n: int
    n_splits: int


def load_config(path: str | Path, project_root: Path | None = None) -> Config:
    path = Path(path)
    root = project_root or path.resolve().parent
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    try:
        cfg = Config(
            data_path=root / raw["data"]["path"],
            sample_path=root / raw["data"]["sample_path"],
            seed=int(raw["run"]["seed"]),
            output_dir=root / raw["run"]["output_dir"],
            cluster_min_words=int(raw["clustering"]["min_words"]),
            cluster_max_responses=int(raw["clustering"]["max_responses"]),
            k_min=int(raw["clustering"]["k_min"]),
            k_max=int(raw["clustering"]["k_max"]),
            svd_components=int(raw["clustering"]["svd_components"]),
            rules_min_support=float(raw["rules"]["min_support"]),
            rules_min_confidence=float(raw["rules"]["min_confidence"]),
            rules_top_n=int(raw["rules"]["top_n"]),
            n_splits=int(raw["prediction"]["n_splits"]),
        )
    except KeyError as exc:
        raise ValueError(f"Missing configuration key: {exc}") from exc

    if not 2 <= cfg.k_min <= cfg.k_max:
        raise ValueError("clustering.k_min must be >= 2 and <= k_max")
    if not 0 < cfg.rules_min_support < 1:
        raise ValueError("rules.min_support must be between 0 and 1")
    if cfg.n_splits < 2:
        raise ValueError("prediction.n_splits must be at least 2")
    return cfg
