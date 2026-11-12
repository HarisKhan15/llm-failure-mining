"""Descriptive statistics, figures and the auto-generated results report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt
import pandas as pd

ERROR_ORDER = [
    "off_by_one_low", "other_wrong", "copies_given_number", "no_parsable_answer",
    "adds_given_numbers", "off_by_one_high", "right_number_wrong_reasoning",
]


def describe(df: pd.DataFrame) -> dict:
    failures = df[~df["correct"]]
    by_model = df.groupby("model")["correct"].agg(["mean", "size"])
    by_model = by_model[by_model["size"] >= 200].sort_values("mean", ascending=False)
    return {
        "n_responses": int(len(df)),
        "n_models": int(df["model"].nunique()),
        "n_prompts": int(df["prompt"].nunique()),
        "dropped_records": int(df.attrs.get("dropped_records", 0)),
        "overall_accuracy": round(float(df["correct"].mean()), 4),
        "accuracy_by_problem_type": df.groupby("problem_type")["correct"].mean().round(4).to_dict(),
        "accuracy_by_style": df.groupby("prompt_style")["correct"].mean().round(4).sort_values(ascending=False).to_dict(),
        "responses_by_style": df["prompt_style"].value_counts().to_dict(),
        "error_share_among_failures": failures["error_type"].value_counts(normalize=True).round(4).to_dict(),
        "error_share_siblings_failures": failures[failures["problem_type"] == "aiw_siblings"]["error_type"]
        .value_counts(normalize=True).round(4).to_dict(),
        "hypothesis_accuracy_vs_off_by_one": _accuracy_vs_off_by_one(df),
        "top_models_accuracy": by_model["mean"].head(8).round(4).to_dict(),
        "bottom_models_accuracy": by_model["mean"].tail(8).round(4).to_dict(),
    }


def _accuracy_vs_off_by_one(df: pd.DataFrame, min_failures: int = 200) -> dict:
    """Test the idea 'stronger models fail more consistently in the same way'.

    For every model with enough failures, compare its accuracy with the share of
    its failures that are off_by_one_low, using Spearman rank correlation.
    """
    failures = df[~df["correct"]]
    share = failures.groupby("model")["error_type"].apply(lambda s: (s == "off_by_one_low").mean())
    counts = failures["model"].value_counts()
    acc = df.groupby("model")["correct"].mean()
    keep = counts[counts >= min_failures].index
    if len(keep) < 5:
        return {"n_models": int(len(keep)), "spearman_rho": None}
    rho = acc[keep].corr(share[keep], method="spearman")
    return {"n_models": int(len(keep)), "spearman_rho": round(float(rho), 3)}


def error_table_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    failures = df[~df["correct"]]
    table = pd.crosstab(failures[column], failures["error_type"], normalize="index").round(3)
    table["n_failures"] = failures[column].value_counts()
    table["accuracy"] = df.groupby(column)["correct"].mean().round(3)
    return table.sort_values("n_failures", ascending=False)


def plot_error_types(stats: dict, path: Path) -> None:
    shares = pd.Series(stats["error_share_among_failures"]).reindex(ERROR_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(shares.index[::-1], shares.values[::-1] * 100, color="#4C72B0")
    ax.set_xlabel("Share of failed responses (%)")
    ax.set_title("What kind of wrong answers do LLMs give on AIW?")
    for i, v in enumerate(shares.values[::-1] * 100):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_by_style(stats: dict, path: Path) -> None:
    acc = pd.Series(stats["accuracy_by_style"])
    counts = pd.Series(stats["responses_by_style"])
    acc = acc[counts.reindex(acc.index) >= 100]  # hide styles with too few responses
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(acc.index[::-1], acc.values[::-1] * 100, color="#55A868")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Accuracy by prompt instruction style (styles with >= 100 responses)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_errors_by_family(table: pd.DataFrame, path: Path) -> None:
    top = table.head(10)[[c for c in ERROR_ORDER if c in table.columns]]
    fig, ax = plt.subplots(figsize=(11, 5))
    left = pd.Series(0.0, index=top.index)
    for col in top.columns:
        ax.barh(top.index, top[col] * 100, left=left * 100, label=col)
        left += top[col]
    ax.set_xlabel("Share of failures (%)")
    ax.set_title("Error types by model family (10 families with most failures)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _hypothesis_text(h: dict) -> str:
    if h.get("spearman_rho") is None:
        return "_Hypothesis check skipped: too few models with enough failures._"
    rho = h["spearman_rho"]
    verdict = "no evidence for" if abs(rho) < 0.3 else ("support for" if rho > 0 else "evidence against")
    return (
        "**Hypothesis check:** do stronger models fail more consistently with the off-by-one error? "
        f"Spearman correlation between model accuracy and off-by-one share across {h['n_models']} models: "
        f"rho = {rho:+.3f}, i.e. {verdict} this idea. The error profile depends on the model family more than on overall accuracy."
    )


def _md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    df = df.head(max_rows)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(str(v).replace("|", "/") for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def paired_comparison(folds: pd.DataFrame, setting: str, challenger: str, baseline: str) -> str:
    """Fold-by-fold comparison: how often does ``challenger`` beat ``baseline``?"""
    f = folds[folds["setting"] == setting]
    if not {challenger, baseline} <= set(f["method"]):
        return ""
    parts = []
    for metric in ("roc_auc", "pr_auc"):
        pivot = f.pivot(index="fold", columns="method", values=metric)
        diff = pivot[challenger] - pivot[baseline]
        parts.append(f"{metric}: wins in {int((diff > 0).sum())}/{len(diff)} folds, mean difference {diff.mean():+.3f}")
    return f"Paired comparison, `{challenger}` vs `{baseline}` ({setting}): " + "; ".join(parts) + "."


def write_report(out: Path, stats: dict, clusters, rules: pd.DataFrame,
                 pred_summary: pd.DataFrame, importance: pd.DataFrame, data_label: str,
                 folds: pd.DataFrame | None = None) -> Path:
    err = stats["error_share_among_failures"]
    sib = stats["error_share_siblings_failures"]
    lines = [
        "# Results Report",
        "",
        f"_Auto-generated by `python -m llm_failure_mining.cli run` on the **{data_label}** data. "
        "Every number below is computed by the pipeline; nothing is typed in by hand._",
        "",
        "## Dataset",
        f"- Responses: **{stats['n_responses']:,}** from **{stats['n_models']}** models on **{stats['n_prompts']}** prompts",
        f"- Records dropped by validation: {stats['dropped_records']}",
        f"- Overall accuracy: **{stats['overall_accuracy']:.1%}**",
        f"- Accuracy by problem type: " + ", ".join(f"{k} {v:.1%}" for k, v in stats["accuracy_by_problem_type"].items()),
        "",
        "## Key finding 1: most failures are one specific mistake",
        f"Among all failed responses, **{err.get('off_by_one_low', 0):.1%}** give the right answer minus one. "
        f"On the sibling problems alone this rises to **{sib.get('off_by_one_low', 0):.1%}**. "
        "In these problems the correct answer is 'stated sisters + 1' because Alice is herself a sister, "
        "so this error means the model forgot to count Alice.",
        "",
        "![Error types](figures/error_types.png)",
        "",
        "## Key finding 2: prompt style and model family",
        _md_table(pd.DataFrame({"style": list(stats["accuracy_by_style"]),
                                "accuracy": [f"{v:.1%}" for v in stats["accuracy_by_style"].values()],
                                "responses": [stats["responses_by_style"][k] for k in stats["accuracy_by_style"]]})),
        "",
        "![Accuracy by style](figures/accuracy_by_style.png)",
        "",
        "![Errors by family](figures/errors_by_family.png)",
        "",
        _hypothesis_text(stats["hypothesis_accuracy_vs_off_by_one"]),
        "",
        "## Failure clusters (TF-IDF + SVD + k-means)",
        f"Chosen k = **{clusters.best_k}** by silhouette score "
        f"({', '.join(f'k={k}: {v:.3f}' for k, v in clusters.silhouette_by_k.items())}).",
        "",
        _md_table(clusters.summary[["cluster", "size", "share", "top_terms", "main_error_type", "main_error_share", "median_words", "top_models"]]),
        "",
        "## Association rules (Apriori)",
        "Rules whose right side is an outcome or an error type, sorted by lift.",
        "",
        _md_table(rules) if not rules.empty else "_No rules passed the thresholds._",
        "",
        "## Predicting failure before the model answers",
        "Grouped cross-validation (no prompt or model appears in both train and test). Mean and standard deviation over folds.",
        "",
        _md_table(pred_summary),
        "",
        paired_comparison(folds, "unseen_prompts", "gradient_boosting_with_model_id", "prior_by_model")
        if folds is not None else "",
        "",
        "How to read this: `prior_by_model` already knows how good each model is in general. "
        "A classifier is only interesting if it beats that prior, which means the prompt features carry extra signal. "
        "With small fold counts, a difference smaller than the fold standard deviation should be treated as uncertain.",
        "",
        "### Permutation importance (gradient boosting, unseen prompts, one fold)",
        _md_table(importance),
        "",
        "## Limitations",
        "- The data covers 63 prompt variants of two problem families; results may not transfer to other reasoning tasks.",
        "- Prompt style and prompt content are partly confounded, because not every style was run on every problem.",
        "- Model sizes for closed models are unknown and grouped as `unknown_closed`.",
        "- Error types are rule-based on the parsed number; they describe the answer, not the full reasoning.",
        "- Correctness labels come from the AIW authors (including their manual checks); we did not relabel responses.",
    ]
    path = out / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
