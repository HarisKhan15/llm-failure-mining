"""Predict whether a model will answer correctly, using only pre-answer features.

Inputs are prompt features (style, problem type, wording) and model features
(family, size, base vs. chat). The response text is NOT used, so the task is:
"Before asking, can we tell whether this model will fail on this prompt?"

Leakage control: the dataset contains many repeated trials of the same
(model, prompt) pair. A random split would put copies of the same question
in both training and test data and inflate the scores. We therefore use
grouped cross-validation in two settings:

* unseen_prompts: test prompts never appear in training (GroupKFold by prompt).
* unseen_models:  test models never appear in training (GroupKFold by model).

Baselines: a constant predictor (ROC-AUC 0.5) and a "prior" that predicts the
training accuracy of the same model (unseen prompts) or the same model family
(unseen models). A learned model is only useful if it beats the prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["prompt_style", "problem_type", "mention_order", "subject_name", "family", "size_bucket"]
NUMERIC = ["n_stated_numbers", "prompt_words", "log_size_b", "size_known"]
BOOLEAN = ["numbers_as_words", "is_base_model", "is_code_model"]
FEATURES = CATEGORICAL + NUMERIC + BOOLEAN


def prediction_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = df[CATEGORICAL + ["model", "n_stated_numbers", "prompt_words"] + BOOLEAN].copy()
    X["size_known"] = df["size_b"].notna().astype(float)
    X["log_size_b"] = np.log1p(df["size_b"].fillna(0.0))  # 0 when unknown; size_known flags it
    for col in BOOLEAN:
        X[col] = X[col].astype(float)
    return X[FEATURES + ["model"]]


def _preprocessor(with_model_id: bool = False) -> ColumnTransformer:
    categorical = CATEGORICAL + (["model"] if with_model_id else [])
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False), categorical),
            ("num", StandardScaler(), NUMERIC + BOOLEAN),
        ]
    )


def make_models(seed: int, with_model_id: bool = False) -> dict[str, Pipeline]:
    """Classifiers to compare.

    With ``with_model_id`` the exact model name is an extra input. That is only
    fair when test models were seen in training (the unseen_prompts setting),
    and it lets us ask: do prompt features add anything beyond knowing the model?
    """
    models = {
        "logistic_regression": Pipeline(
            [("prep", _preprocessor()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor()),
                ("clf", HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, random_state=seed)),
            ]
        ),
    }
    if with_model_id:
        models["gradient_boosting_with_model_id"] = Pipeline(
            [
                ("prep", _preprocessor(with_model_id=True)),
                ("clf", HistGradientBoostingClassifier(class_weight="balanced", max_iter=200, random_state=seed)),
            ]
        )
    return models


LEARNED = ("logistic_regression", "gradient_boosting", "gradient_boosting_with_model_id")


def _prior_scores(train: pd.DataFrame, test: pd.DataFrame, key: str) -> np.ndarray:
    rates = train.groupby(key)["correct"].mean()
    return test[key].map(rates).fillna(train["correct"].mean()).to_numpy()


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "balanced_accuracy": np.nan}
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "balanced_accuracy": balanced_accuracy_score(y_true, scores >= threshold),
    }


def evaluate(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Return per-fold metrics for every setting and method."""
    X = prediction_frame(df)
    y = df["correct"].to_numpy().astype(int)
    rows = []
    settings = {"unseen_prompts": ("prompt_key", "model"), "unseen_models": ("model", "family")}

    for setting, (group_col, prior_key) in settings.items():
        groups = df[group_col].to_numpy()
        splits = min(n_splits, len(np.unique(groups)))
        for fold, (tr, te) in enumerate(GroupKFold(n_splits=splits).split(X, y, groups)):
            train_df, test_df = df.iloc[tr], df.iloc[te]
            base_rate = y[tr].mean()
            # Thresholds: scores at or above the training base rate count as "correct".
            candidates = {
                "constant_baseline": np.full(len(te), base_rate),
                f"prior_by_{prior_key}": _prior_scores(train_df, test_df, prior_key),
            }
            for name, model in make_models(seed, with_model_id=(setting == "unseen_prompts")).items():
                model.fit(X.iloc[tr], y[tr])
                candidates[name] = model.predict_proba(X.iloc[te])[:, 1]

            for method, scores in candidates.items():
                # Balanced models output ~0.5 for an average case; priors output a rate.
                threshold = 0.5 if method in LEARNED else base_rate
                rows.append(
                    {"setting": setting, "fold": fold, "method": method, "n_test": len(te),
                     "test_positive_rate": round(float(y[te].mean()), 4),
                     **_metrics(y[te], scores, threshold)}
                )
    return pd.DataFrame(rows)


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    agg = fold_metrics.groupby(["setting", "method"])[["roc_auc", "pr_auc", "balanced_accuracy"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.round(3).reset_index()


def feature_importance(df: pd.DataFrame, seed: int = 42, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance on one held-out group of prompts.

    Importance = drop in ROC-AUC when one input column is shuffled.
    """
    X = prediction_frame(df)
    y = df["correct"].to_numpy().astype(int)
    tr, te = next(GroupKFold(n_splits=5).split(X, y, df["prompt_key"]))
    model = make_models(seed)["gradient_boosting"].fit(X[FEATURES].iloc[tr], y[tr])
    X = X[FEATURES]
    result = permutation_importance(
        model, X.iloc[te], y[te], scoring="roc_auc", n_repeats=n_repeats, random_state=seed
    )
    return (
        pd.DataFrame({"feature": FEATURES, "importance_mean": result.importances_mean,
                      "importance_std": result.importances_std})
        .sort_values("importance_mean", ascending=False)
        .round(4)
        .reset_index(drop=True)
    )
