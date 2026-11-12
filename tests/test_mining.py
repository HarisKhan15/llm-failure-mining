import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from llm_failure_mining.clustering import cluster_failures
from llm_failure_mining.predict import evaluate, prediction_frame, summarize
from llm_failure_mining.rules import build_transactions, mine_rules


def test_transactions_are_boolean_and_outcomes_complementary(featured_sample):
    t = build_transactions(featured_sample)
    assert t.dtypes.map(lambda d: d == bool).all()
    assert (t["outcome=FAIL"] ^ t["outcome=CORRECT"]).all()


def test_rules_have_single_target_consequent_and_lift_above_one(featured_sample):
    rules = mine_rules(featured_sample, min_support=0.05, min_confidence=0.5, top_n=50)
    assert not rules.empty
    assert rules["consequents"].str.match(r"^(outcome|error)=").all()
    assert not rules["antecedents"].str.contains("outcome=|error=").any()
    assert (rules["lift"] > 1).all()


def test_clustering_returns_consistent_summary(featured_sample):
    result = cluster_failures(featured_sample, min_words=5, max_responses=1000, k_min=2, k_max=3, svd_components=20)
    assert result.best_k in (2, 3)
    assert result.summary["size"].sum() == len(result.assignments)
    assert set(result.assignments["cluster"]) == set(range(result.best_k))


def test_prediction_features_exclude_response_information(featured_sample):
    X = prediction_frame(featured_sample)
    leaking = {"response", "parsed_answer", "error_type", "response_words", "length_bucket", "correct"}
    assert leaking.isdisjoint(X.columns)


def test_grouped_split_never_shares_prompts(featured_sample):
    X = prediction_frame(featured_sample)
    groups = featured_sample["prompt_key"]
    for tr, te in GroupKFold(n_splits=3).split(X, groups=groups):
        assert set(groups.iloc[tr]).isdisjoint(groups.iloc[te])


def test_evaluation_runs_and_constant_baseline_is_chance(featured_sample):
    folds = evaluate(featured_sample, n_splits=3, seed=0)
    summary = summarize(folds)
    const = summary[summary["method"] == "constant_baseline"]
    assert np.allclose(const["roc_auc_mean"].dropna(), 0.5)
    assert {"unseen_prompts", "unseen_models"} == set(summary["setting"])
    assert "gradient_boosting_with_model_id" in set(folds[folds.setting == "unseen_prompts"]["method"])
    assert "gradient_boosting_with_model_id" not in set(folds[folds.setting == "unseen_models"]["method"])
