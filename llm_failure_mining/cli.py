"""Command-line entry point.

Examples:
    python -m llm_failure_mining.cli run              # full dataset
    python -m llm_failure_mining.cli run --sample     # small committed sample
    python -m llm_failure_mining.cli make-sample      # rebuild the sample file
    python -m llm_failure_mining.cli run --data data/collected/my.json --output results/my
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .clustering import cluster_failures
from .config import load_config
from .data import load_dataset, load_records
from .features import add_features
from .predict import evaluate, feature_importance, summarize
from .report import (
    describe, error_table_by, plot_accuracy_by_style, plot_error_types,
    plot_errors_by_family, save_json, write_report,
)
from .rules import mine_rules

PROJECT_ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("llm_failure_mining")


def run(config_path: Path, use_sample: bool, output_dir: Path | None = None,
        data_override: Path | None = None) -> Path:
    cfg = load_config(config_path, PROJECT_ROOT)
    data_path = data_override or (cfg.sample_path if use_sample else cfg.data_path)
    if output_dir:
        out = output_dir
    elif data_override:
        out = cfg.output_dir / Path(data_override).stem  # never overwrite the main results
    else:
        out = cfg.output_dir / "sample" if use_sample else cfg.output_dir
    started = time.time()

    log.info("1/6 Loading %s", data_path)
    df = add_features(load_dataset(data_path))
    (out / "figures").mkdir(parents=True, exist_ok=True)  # only after data loaded successfully
    log.info("    %d responses, %d models, %d prompts", len(df), df["model"].nunique(), df["prompt"].nunique())

    log.info("2/6 Descriptive statistics and error types")
    stats = describe(df)
    save_json(stats, out / "summary.json")
    family_table = error_table_by(df, "family")
    family_table.to_csv(out / "error_types_by_family.csv")
    error_table_by(df, "model").to_csv(out / "error_types_by_model.csv")
    error_table_by(df, "prompt_style").to_csv(out / "error_types_by_style.csv")
    plot_error_types(stats, out / "figures" / "error_types.png")
    plot_accuracy_by_style(stats, out / "figures" / "accuracy_by_style.png")
    plot_errors_by_family(family_table, out / "figures" / "errors_by_family.png")

    log.info("3/6 Clustering failed responses")
    clusters = cluster_failures(
        df, cfg.cluster_min_words, cfg.cluster_max_responses, cfg.k_min, cfg.k_max, cfg.svd_components, cfg.seed
    )
    clusters.summary.to_csv(out / "clusters.csv", index=False)
    log.info("    best k = %d", clusters.best_k)

    log.info("4/6 Mining association rules")
    rules = mine_rules(df, cfg.rules_min_support, cfg.rules_min_confidence, cfg.rules_top_n)
    rules.to_csv(out / "rules.csv", index=False)
    log.info("    %d rules kept", len(rules))

    log.info("5/6 Predicting failure with grouped cross-validation")
    folds = evaluate(df, cfg.n_splits, cfg.seed)
    folds.to_csv(out / "prediction_folds.csv", index=False)
    pred_summary = summarize(folds)
    pred_summary.to_csv(out / "prediction_summary.csv", index=False)
    importance = feature_importance(df, cfg.seed)
    importance.to_csv(out / "feature_importance.csv", index=False)

    log.info("6/6 Writing report")
    report = write_report(out, stats, clusters, rules, pred_summary, importance,
                          "sample" if use_sample else "full", folds)
    try:
        shown_path = str(Path(data_path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        shown_path = str(data_path)
    save_json({"data_path": shown_path, "seed": cfg.seed, "seconds": round(time.time() - started, 1),
               "best_k": clusters.best_k}, out / "run_info.json")
    log.info("Done in %.1f s. Report: %s", time.time() - started, report)
    return report


def make_sample(config_path: Path, per_group: int = 4) -> Path:
    """Build a small, stratified sample of the real data for demos and tests.

    Takes up to ``per_group`` responses for every (prompt, correct) pair so
    all prompts and both outcomes are represented.
    """
    cfg = load_config(config_path, PROJECT_ROOT)
    records = load_records(cfg.data_path)
    buckets: dict[tuple, list] = {}
    for rec in records:
        buckets.setdefault((rec["prompt"], bool(rec["correct"])), []).append(rec)
    sample = []
    for key in sorted(buckets, key=lambda k: (k[0], k[1])):
        group = buckets[key]
        step = max(1, len(group) // per_group)
        sample.extend(group[::step][:per_group])
    cfg.sample_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.sample_path.write_text(json.dumps(sample, indent=1), encoding="utf-8")
    log.info("Wrote %d records to %s", len(sample), cfg.sample_path)
    return cfg.sample_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine failure patterns of LLMs on the AIW benchmark.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run the full pipeline")
    run_p.add_argument("--sample", action="store_true", help="use the small committed sample")
    run_p.add_argument("--output", type=Path, default=None, help="output folder (default from config)")
    run_p.add_argument("--data", type=Path, default=None, help="analyse another JSON file in AIW format")
    sub.add_parser("make-sample", help="rebuild data/sample/aiw_sample.json from the full data")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "run":
            run(args.config, args.sample, args.output, args.data)
        else:
            make_sample(args.config)
    except (FileNotFoundError, ValueError) as exc:
        log.error("Error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
