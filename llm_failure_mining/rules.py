"""Association rule mining: which combinations of conditions go with failure?

Each response becomes a "transaction" (a shopping basket) of items such as
``style=restricted_format``, ``family=gpt`` or ``outcome=FAIL``. Apriori finds
item sets that appear together often, and we keep rules whose right-hand side
is a single outcome or error type, e.g.

    {problem=aiw_plus_cousins, size=small_<10B} -> {outcome=FAIL}

Support    = share of all responses containing every item of the rule.
Confidence = P(right side | left side).
Lift       = confidence / P(right side); lift > 1 means "more often than usual".
"""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

ITEM_COLUMNS = {
    "problem": "problem_type",
    "style": "prompt_style",
    "family": "family",
    "size": "size_bucket",
    "length": "length_bucket",
}


def build_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot table: one row per response, one boolean column per item."""
    items = pd.DataFrame(index=df.index)
    for prefix, column in ITEM_COLUMNS.items():
        items = items.join(pd.get_dummies(df[column], prefix=prefix, prefix_sep="=").astype(bool))
    items["base_model"] = df["is_base_model"].astype(bool)
    items["outcome=CORRECT"] = df["correct"].astype(bool)
    items["outcome=FAIL"] = ~df["correct"].astype(bool)
    for err in df["error_type"].unique():
        if err != "correct":
            items[f"error={err}"] = df["error_type"] == err
    return items


def mine_rules(
    df: pd.DataFrame, min_support: float = 0.02, min_confidence: float = 0.5, top_n: int = 25
) -> pd.DataFrame:
    transactions = build_transactions(df)
    frequent = apriori(transactions, min_support=min_support, use_colnames=True, max_len=4)
    if frequent.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)

    # Keep rules that explain an outcome or error type from describing items only.
    def is_target(items: frozenset) -> bool:
        return len(items) == 1 and next(iter(items)).startswith(("outcome=", "error="))

    def no_target(items: frozenset) -> bool:
        return not any(i.startswith(("outcome=", "error=")) for i in items)

    rules = rules[rules["consequents"].map(is_target) & rules["antecedents"].map(no_target)]
    rules = rules[rules["lift"] > 1.0]
    rules = rules.sort_values(["lift", "confidence"], ascending=False).head(top_n)

    out = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
    out["antecedents"] = out["antecedents"].map(lambda s: " AND ".join(sorted(s)))
    out["consequents"] = out["consequents"].map(lambda s: next(iter(s)))
    return out.round({"support": 4, "confidence": 3, "lift": 2}).reset_index(drop=True)
