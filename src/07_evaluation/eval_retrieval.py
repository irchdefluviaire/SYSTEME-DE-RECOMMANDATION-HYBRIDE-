"""
eval_retrieval.py
Module 07 — Métriques de retrieval (IR)

Fournit le calcul de :
  - Precision@K
  - Recall@K
  - NDCG@K  (Normalized Discounted Cumulative Gain)
  - MRR@K   (Mean Reciprocal Rank)
  - Intervalles de confiance par bootstrap (rééchantillonnage sans remise)

Aucune dépendance externe hormis numpy.
"""
import math
from collections import defaultdict
from typing import Callable

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Métriques par requête individuelle
# ─────────────────────────────────────────────────────────────────────────

def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Fraction de documents pertinents parmi les K premiers résultats."""
    if k <= 0:
        return 0.0
    return sum(1 for doc in retrieved[:k] if doc in relevant) / k


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    """Fraction des documents pertinents récupérés parmi les K premiers."""
    if not relevant:
        return 0.0
    return sum(1 for doc in retrieved[:k] if doc in relevant) / len(relevant)


def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    """NDCG@K — Normalized Discounted Cumulative Gain à l'ordre K."""
    dcg = sum(
        int(doc in relevant) / math.log2(i + 2)
        for i, doc in enumerate(retrieved[:k])
    )
    ideal_n = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(relevant: set, retrieved: list, k: int) -> float:
    """MRR@K — rang inverse du premier document pertinent dans les K premiers."""
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────
# Agrégation sur un ensemble de requêtes
# ─────────────────────────────────────────────────────────────────────────

def compute_metrics(
    results: dict[str, tuple[set, list]],
    k_values: list[int] | None = None,
    mrr_k: int = 10,
) -> dict[str, float]:
    """
    Calcule toutes les métriques IR moyennées sur l'ensemble des requêtes.

    Args:
        results  : {query_id: (relevant_set, ranked_retrieved_list)}
        k_values : valeurs de K pour P/R/NDCG  (défaut : [1, 5, 10])
        mrr_k    : K pour MRR                   (défaut : 10)

    Returns:
        Dictionnaire {metric_name: mean_value} arrondi à 4 décimales.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    agg: dict[str, list[float]] = defaultdict(list)

    for _qid, (rel, ret) in results.items():
        for k in k_values:
            agg[f"precision@{k}"].append(precision_at_k(rel, ret, k))
            agg[f"recall@{k}"].append(recall_at_k(rel, ret, k))
            agg[f"ndcg@{k}"].append(ndcg_at_k(rel, ret, k))
        agg[f"mrr@{mrr_k}"].append(mrr_at_k(rel, ret, mrr_k))

    return {key: round(float(np.mean(vals)), 4) for key, vals in sorted(agg.items())}


# ─────────────────────────────────────────────────────────────────────────
# Intervalles de confiance par bootstrap
# ─────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    results: dict[str, tuple[set, list]],
    metric_fn: Callable[[set, list, int], float],
    k: int = 10,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Intervalle de confiance bootstrap à (1 − alpha) pour une métrique donnée.

    Args:
        results     : {qid: (relevant_set, retrieved_list)}
        metric_fn   : une des fonctions ci-dessus (precision_at_k, ndcg_at_k, …)
        k           : paramètre K transmis à metric_fn
        n_bootstrap : nombre de ré-échantillonnages (défaut : 1 000)
        alpha       : niveau de signification (défaut : 0.05 → IC 95 %)

    Returns:
        (mean, lower_bound, upper_bound) arrondis à 4 décimales
    """
    rng       = np.random.default_rng(seed)
    query_ids = list(results.keys())
    n         = len(query_ids)
    scores    = [metric_fn(results[qid][0], results[qid][1], k) for qid in query_ids]
    mean_score = float(np.mean(scores))

    bootstrap_means: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bootstrap_means.append(float(np.mean([scores[i] for i in idx])))

    lo = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    hi = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return round(mean_score, 4), round(lo, 4), round(hi, 4)
