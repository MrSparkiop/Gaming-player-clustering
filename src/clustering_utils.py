"""Reusable helpers for the Gaming-Player-Clustering project.

This module isolates the repetitive, well-tested pieces of the analysis
(model-selection sweeps, a null-reference baseline, cluster profiling and a
couple of plotting utilities) so that the notebook stays readable and the
logic can be unit-tested in one place.

Every function that involves randomness takes an explicit ``random_state``
(default ``42``) so results are fully reproducible.

Author: Blagoy Hristov
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

RANDOM_STATE = 42


def set_plot_style() -> None:
    """Apply a consistent, readable Matplotlib style for the whole notebook."""
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 100,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "font.size": 11,
        }
    )


def kmeans_selection(
    X: np.ndarray,
    k_values: Iterable[int],
    random_state: int = RANDOM_STATE,
    n_init: int = 10,
    sample_size: int = 10_000,
) -> pd.DataFrame:
    """Fit K-Means for several ``k`` and return internal validation metrics.

    Parameters
    ----------
    X:
        Standardised feature matrix of shape ``(n_samples, n_features)``.
    k_values:
        Iterable of cluster counts to try (e.g. ``range(2, 11)``).
    random_state, n_init:
        Passed straight to :class:`sklearn.cluster.KMeans`.
    sample_size:
        Sub-sample size for the (O(n^2)) silhouette computation. Using a
        fixed, seeded sub-sample keeps the metric stable and affordable on
        ~40k rows.

    Returns
    -------
    pandas.DataFrame
        One row per ``k`` with columns ``inertia`` (WCSS), ``silhouette``,
        ``calinski_harabasz`` and ``davies_bouldin``.
    """
    records = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X)
        records.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(
                    X, labels, sample_size=sample_size, random_state=random_state
                ),
                "calinski_harabasz": calinski_harabasz_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
            }
        )
    return pd.DataFrame.from_records(records).set_index("k")


def null_silhouette_band(
    shape: tuple[int, int],
    k_values: Iterable[int],
    n_draws: int = 20,
    random_state: int = RANDOM_STATE,
    n_init: int = 10,
    sample_size: int = 10_000,
) -> pd.DataFrame:
    """Monte-Carlo silhouette *band* for a uniform-random null model.

    Rather than a single null draw, we generate ``n_draws`` independent
    structureless datasets (independent uniform features of the same shape,
    then standardised) and cluster each exactly as the real data. The spread
    across draws yields a reference band. If the real silhouette lies inside
    that band at every ``k``, the real "clusters" carry no more structure than
    a random partition of a uniform cloud — turning a hand-wavy "the scores
    look low" into a defensible comparison.

    Returns a DataFrame indexed by ``k`` with columns ``null_mean``,
    ``null_min``, ``null_max`` and ``null_std``.
    """
    rng = np.random.RandomState(random_state)
    draws: dict[int, list[float]] = {k: [] for k in k_values}
    for _ in range(n_draws):
        x_null = StandardScaler().fit_transform(rng.uniform(size=shape))
        for k in k_values:
            labels = KMeans(
                n_clusters=k, random_state=random_state, n_init=n_init
            ).fit_predict(x_null)
            draws[k].append(
                silhouette_score(
                    x_null, labels, sample_size=sample_size, random_state=random_state
                )
            )
    rows = {
        k: {
            "null_mean": float(np.mean(v)),
            "null_min": float(np.min(v)),
            "null_max": float(np.max(v)),
            "null_std": float(np.std(v)),
        }
        for k, v in draws.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("k")


def gmm_selection(
    X: np.ndarray,
    k_values: Iterable[int],
    random_state: int = RANDOM_STATE,
    n_init: int = 1,
) -> pd.DataFrame:
    """Fit Gaussian Mixture Models and return the BIC / AIC for each ``k``."""
    records = []
    for k in k_values:
        gmm = GaussianMixture(
            n_components=k, random_state=random_state, n_init=n_init
        ).fit(X)
        records.append({"k": k, "bic": gmm.bic(X), "aic": gmm.aic(X)})
    return pd.DataFrame.from_records(records).set_index("k")


def cluster_profile(
    frame: pd.DataFrame,
    labels: np.ndarray,
    features: Sequence[str],
) -> pd.DataFrame:
    """Return the per-cluster mean of ``features`` plus the cluster size.

    Used for post-hoc interpretation: given cluster assignments, describe what
    each cluster looks like on the original (un-scaled) feature scale.
    """
    profiled = frame[list(features)].copy()
    profiled["cluster"] = labels
    summary = profiled.groupby("cluster").mean().round(2)
    summary.insert(0, "size", profiled.groupby("cluster").size())
    return summary


def plot_metric_vs_k(
    metrics: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Line plot of a single selection metric against ``k`` (thin helper)."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(metrics.index, metrics[column], marker="o")
    ax.set_xlabel("Number of clusters $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    return ax
