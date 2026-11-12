"""Group failed responses by the words they use, to discover failure "themes".

Pipeline: TF-IDF text vectors -> truncated SVD (latent semantic analysis)
-> k-means. The number of clusters k is chosen by the silhouette score on
the SVD vectors. Each cluster is then described by its most important words,
its most common rule-based error type, and a short example.

We use TF-IDF instead of neural sentence embeddings so the run needs no
model download and gives identical results on every machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer


@dataclass
class ClusteringResult:
    assignments: pd.DataFrame      # one row per clustered response
    summary: pd.DataFrame          # one row per cluster
    silhouette_by_k: dict[int, float]
    best_k: int


def select_failures(df: pd.DataFrame, min_words: int, max_responses: int, seed: int) -> pd.DataFrame:
    failed = df[(~df["correct"]) & (df["response_words"] >= min_words)]
    if len(failed) > max_responses:
        failed = failed.sample(n=max_responses, random_state=seed)
    return failed.reset_index(drop=True)


def cluster_failures(
    df: pd.DataFrame,
    min_words: int = 20,
    max_responses: int = 8000,
    k_min: int = 4,
    k_max: int = 10,
    svd_components: int = 100,
    seed: int = 42,
) -> ClusteringResult:
    failed = select_failures(df, min_words, max_responses, seed)
    if len(failed) < k_max * 5:
        raise ValueError(f"Too few failed responses to cluster ({len(failed)}); lower clustering.min_words")

    vectorizer = TfidfVectorizer(
        stop_words="english", min_df=5, max_df=0.6, ngram_range=(1, 2), sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",  # words only; numbers vary per prompt
    )
    tfidf = vectorizer.fit_transform(failed["response"])
    n_comp = min(svd_components, tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    vectors = Normalizer(copy=False).fit_transform(svd.fit_transform(tfidf))

    silhouette_by_k: dict[int, float] = {}
    models: dict[int, KMeans] = {}
    sample_size = min(3000, len(failed))
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(vectors)
        silhouette_by_k[k] = float(
            silhouette_score(vectors, km.labels_, sample_size=sample_size, random_state=seed)
        )
        models[k] = km
    best_k = max(silhouette_by_k, key=silhouette_by_k.get)
    labels = models[best_k].labels_

    assignments = failed[["model", "prompt_key", "prompt_style", "problem_type", "error_type", "response_words"]].copy()
    assignments["cluster"] = labels

    # Describe clusters by the terms with the highest mean TF-IDF weight.
    terms = np.array(vectorizer.get_feature_names_out())
    rows = []
    for c in range(best_k):
        mask = labels == c
        mean_weights = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_terms = ", ".join(terms[mean_weights.argsort()[::-1][:8]])
        members = failed[mask]
        top_error = members["error_type"].value_counts()
        # Example: the response closest to the cluster centre, shortened.
        centre = models[best_k].cluster_centers_[c]
        idx = np.where(mask)[0][np.argmax(vectors[mask] @ centre)]
        example = " ".join(failed.loc[idx, "response"].split())[:300]
        rows.append(
            {
                "cluster": c,
                "size": int(mask.sum()),
                "share": round(float(mask.mean()), 4),
                "top_terms": top_terms,
                "main_error_type": top_error.index[0],
                "main_error_share": round(float(top_error.iloc[0] / mask.sum()), 3),
                "median_words": int(members["response_words"].median()),
                "top_models": ", ".join(members["model"].value_counts().index[:3]),
                "example": example,
            }
        )
    summary = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    return ClusteringResult(assignments, summary, silhouette_by_k, best_k)
